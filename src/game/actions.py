import itertools
import math
from typing import Optional

import numpy as np

from src.game import Game
from src.misc.utils import random_argmax, multi_dim_choice


def filter_illegal_and_normalize(
        action_probs: np.ndarray,  # array of shape [players-at-turn, a]
        game: Game,
        epsilon: float = 1e-5,
) -> np.ndarray:  # array of shape [players-at-turn, a]
    if len(action_probs.shape) != 2 or action_probs.shape[0] != game.num_players_at_turn()\
            or action_probs.shape[1] != game.num_actions:
        raise ValueError(f"Invalid array shape: {action_probs.shape}")
    # illegal actions of players at turn
    result = action_probs.copy()
    result += epsilon  # numerical stability
    for idx, player in enumerate(game.players_at_turn()):
        for action in game.illegal_actions(player):
            result[idx, action] = 0
    # normalize
    prob_sum = np.sum(result, axis=-1)
    # sanity check
    if np.any(prob_sum == 0) or np.any(np.isnan(prob_sum)) or np.any(np.isinf(prob_sum)):
        raise Exception(f"Invalid probability distribution in filter method: {action_probs} for game \n"
                        f"{game.get_str_repr()}\n {result=}\n"
                        f"{[game.available_actions(p) for p in range(action_probs.shape[0])]}")
    result = result / prob_sum[..., np.newaxis]
    return result


def apply_permutation(
        action_probs: np.ndarray,  # array of shape [p, a]
        perm: dict[int, int],
) -> np.ndarray:
    if len(action_probs.shape) != 2:
        raise ValueError(f"Invalid Array Shape: {action_probs.shape}")
    result = action_probs.copy()
    for k, v in perm.items():
        result[:, v] = action_probs[:, k]  # [p, a]
    return result


def add_dirichlet_noise(
        action_probs: np.ndarray,  # actions of shape (p, a)
        dirichlet_alpha: float,
        dirichlet_eps: float,
) -> np.ndarray:
    if len(action_probs.shape) != 2:
        raise ValueError(f"Invalid action array shape: {action_probs.shape}")
    if dirichlet_eps == 0:
        return action_probs
    # add dirichlet noise
    if dirichlet_alpha == math.inf:
        noise = np.ones_like(action_probs)
    else:
        alpha_list = [dirichlet_alpha for _ in range(action_probs.shape[1])]
        # noise = rng_actions.dirichlet(alpha_list, action_probs.shape[0])
        noise = np.random.dirichlet(alpha=alpha_list, size=action_probs.shape[0])
    new_actions = (1 - dirichlet_eps) * action_probs + dirichlet_eps * noise
    # normalize probability distribution
    prob_sum = np.sum(new_actions, axis=-1)
    if np.any(prob_sum == 0):
        raise Exception("Cannot normalize probability distribution with zero probs")
    result = new_actions / prob_sum[:, np.newaxis]
    return result


def compute_joint_probs(
        actions: np.ndarray,  # actions probabilities of shape (players_at_turn, a)
        game: Game,
) -> np.ndarray:
    """
    compute the joint probability given the individual action probs. Illegal actions have zero probability regardless
    of input and the resulting prob. dist. is normalized (sum to one)
    """
    if len(actions.shape) != 2 or actions.shape[0] != game.num_players_at_turn() \
            or actions.shape[1] != game.num_actions:
        raise ValueError(f"Wrong input shape for actions: {actions.shape}")
    # set prob of illegal actions to zero
    filtered_actions = filter_illegal_and_normalize(actions, game)
    # joint action is product of individual probs (independence assumption)
    arr_list = []
    for player_idx, player in enumerate(game.players_at_turn()):
        legal_action_probs = []
        for action in game.available_actions(player):
            legal_action_probs.append(filtered_actions[player_idx, action])
        # normalization is not necessary here, because we normalized above
        arr_list.append(legal_action_probs)
    prod_arr = np.asarray(list(itertools.product(*arr_list)))
    joint_probs = np.prod(prod_arr, axis=-1)
    if (np.abs(joint_probs.sum() - 1) > 0.001).item():  # sanity check
        raise ValueError(f"Invalid probability distribution: {joint_probs}")
    return joint_probs


def sample_joint_action(
        joint_action_list: list[tuple[int, ...]],  # joint actions indexing the joint probability parameter
        joint_probs: np.ndarray,  # array of shape (num_joint_actions, )
        temperature: float,
) -> tuple[int, ...]:
    if len(joint_probs.shape) != 1:
        raise ValueError(f"Wrong shape of joint prob array: {joint_probs.shape}")
    if len(joint_action_list) != joint_probs.shape[0]:
        raise ValueError(f"Wrong list sizes in filter function: {len(joint_action_list)} vs {joint_probs.shape}")
    if temperature == math.inf:
        # take maximum prob value
        max_idx = random_argmax(joint_probs)
        # max_idx = np.argmax(joint_probs, axis=0).item()
        return joint_action_list[max_idx]
    # apply temperature and normalize
    temp_probs = np.power(joint_probs, temperature)
    prob_sum = np.sum(temp_probs, axis=-1)
    if np.any(prob_sum == 0):
        raise Exception("Cannot normalize a probability distribution with zero values")
    norm_probs = temp_probs / prob_sum
    # sample
    rng = np.random.default_rng()
    joint_action = rng.choice(joint_action_list, 1, p=norm_probs)
    tuple_actions = tuple(joint_action.squeeze())
    return tuple_actions


def sample_individual_actions(
        action_probs: np.ndarray,  # actions probabilities of shape (p, a)
        temperature: float,
        invalid_mask: Optional[np.ndarray] = None,
) -> tuple[int, ...]:
    # sample one joint action from the individual probabilities of each player
    if len(action_probs.shape) != 2:
        raise ValueError(f"Wrong input shape for action probs: {action_probs}")
    if temperature == math.inf:
        # take maximum prob values
        action_list = []
        for p_idx in range(action_probs.shape[0]):
            action_list.append(random_argmax(action_probs[p_idx]))
        action_tuple = tuple(action_list)
        return action_tuple
    # apply temperature and normalize
    temp_probs = np.power(action_probs, temperature)
    if invalid_mask is not None:
        temp_probs[invalid_mask] = 0
    prob_sum = np.sum(temp_probs, axis=-1)
    if np.any(prob_sum == 0):
        raise Exception("Cannot normalize a probability distribution with zero values")
    norm_probs = temp_probs / prob_sum[..., np.newaxis]
    # sample
    ja_array = multi_dim_choice(norm_probs)
    return tuple(ja_array)

def q_values_from_individual_actions(
        action_probs: list[np.ndarray],  # len (num_p_at_turn - 1)
        game: Game,
        player: int,
        joint_action_values: np.ndarray,  # shape (num_ja, num_p_at_turn)
) -> np.ndarray:
    # sanity checks
    if len(action_probs) != game.num_players_at_turn() - 1:
        raise ValueError(f"Invalid number of policies: {action_probs}, expected {game.num_players_at_turn() - 1}")
    player_idx = game.players_at_turn().index(player)
    counter = 0
    for p_idx, p in enumerate(game.players_at_turn()):
        if p == player:
            continue
        if len(action_probs[counter]) != len(game.available_actions(p)):
            raise ValueError(f"Player {p} has available actions {game.available_actions(p)}, but invalid action probs:"
                             f"{action_probs=}")
    # joint policy
    grouped = itertools.product(*action_probs)
    grouped_arr = np.asarray(list(grouped), dtype=float)
    joint_pol = np.prod(grouped_arr, axis=1)[:, np.newaxis]
    # joint actions
    aa = [game.available_actions(p) for p in game.players_at_turn()]
    joint_aa = list(itertools.product(*aa))
    # joint enemy actions
    aa_enemy = [game.available_actions(p) for p in game.players_at_turn() if p != player]
    joint_aa_enemy = list(itertools.product(*aa_enemy))
    # dict of enemy joint actions and values
    enemy_ja_dict = dict(zip(joint_aa_enemy, joint_pol))
    q_vals = np.zeros(shape=len(game.available_actions(player)), dtype=float)
    for ja, ja_val in zip(joint_aa, joint_action_values):
        player_action = ja[player_idx]
        ja_lst = list(ja)
        del ja_lst[player_idx]
        enemy_ja = tuple(ja_lst)
        q_vals[player_action] += ja_val[player_idx] * enemy_ja_dict[enemy_ja]
    return q_vals
