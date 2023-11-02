import math
from collections import defaultdict

import numpy as np

from src.misc.utils import softmax_weighting
from src.search.node import Node


def filter_fully_explored(
        joint_action_list: list[tuple[int, ...]],
        joint_probs: np.ndarray,  # array of shape (num_joint_actions, )
        node: Node,
) -> np.ndarray:  # returns filtered and normalized joint probability
    if len(joint_probs.shape) != 1:
        raise ValueError(f"Wrong shape of joint prob array: {joint_probs.shape}")
    if len(joint_action_list) != joint_probs.shape[0]:
        raise ValueError(f"Wrong list sizes in filter function: {len(joint_action_list)} vs {joint_probs.shape}")
    # set probability of fully explored subtrees to zero
    for idx, joint_action in enumerate(joint_action_list):
        if node.children[joint_action].is_fully_explored():
            joint_probs[idx] = 0
    # normalize probability dist
    if np.sum(joint_probs) == 0:
        raise Exception("No subtree available for probability calculation")
    normalized = joint_probs / np.sum(joint_probs)
    return normalized


def max_player_action(
        action_values: dict[tuple[int, int], float],  # key is (player, action)
        node: Node,
        policy_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:  # value, action_prob
    """
    Compute the best response for each player given their player-action values. The actions are probabilities
    weighted by stretched sigmoid and normalized afterwards
    returns arrays of shape (num_player, ) and (players-at-turn, a)
    """
    values = np.zeros((node.game.num_players,), dtype=float)
    action_probs: np.ndarray = np.zeros((node.game.num_players_at_turn(), node.game.num_actions), dtype=float)
    for player_idx, player in enumerate(node.game.players_at_turn()):
        max_val = -math.inf
        cur_player_vals = []
        for action in node.game.available_actions(player):
            # find max values
            v = action_values[player, action]
            if v > max_val:
                max_val = v
            # action_list[idx] = action
            cur_player_vals.append(v)
        # compute action probability
        cur_action_probs = softmax_weighting(np.asarray(cur_player_vals), policy_temperature)
        for action_idx, action in enumerate(node.game.available_actions(player)):
            action_probs[player_idx, action] = cur_action_probs[action_idx]
        # save max value
        values[player] = max_val
    return values, action_probs


def min_enemy_response(
        node: Node,
) -> dict[tuple[int, int], float]:
    """
    Compute player-action values if the enemy responds to players move by minimizing players outcome
    """
    joint_action_values = node.get_joint_backward_estimate()
    # initialize min values with maximum possible value, key is (player, action)
    min_action_values: dict[tuple[int, int], float] = defaultdict(lambda: math.inf)
    # compute minimum possible output for every player-action pair
    for joint_action, values in joint_action_values.items():
        for idx, player in enumerate(node.game.players_at_turn()):
            if values[player] < min_action_values[player, joint_action[idx]]:
                min_action_values[player, joint_action[idx]] = values[player]
    return min_action_values


def avg_enemy_response(
        node: Node,
) -> dict[tuple[int, int], float]:
    """
    Compute player-action values if the enemy responds to players move with uniform random move selection.
    Player-action value is average of all possible outcomes given that action.
    """
    joint_action_values = node.get_joint_backward_estimate()
    # initialize decoupled action values, key is (player, action), value is (sum, counter)
    avg_action_values: dict[tuple[int, int], tuple[float, int]] = defaultdict(lambda: (0, 0))
    # compute average outcome for every player-action pair
    for joint_action, values in joint_action_values.items():
        for idx, player in enumerate(node.game.players_at_turn()):
            avg_action_values[player, joint_action[idx]] = (
                avg_action_values[player, joint_action[idx]][0] + values[player],
                avg_action_values[player, joint_action[idx]][1] + 1
            )
    # normalize
    avg_action_values_norm: dict[tuple[int, int], float] = defaultdict(lambda: 0)
    for player_action, sum_counter in avg_action_values.items():
        val_sum, counter = sum_counter
        if counter == 0:
            raise Exception("Cannot divide by counter of zero")
        avg_action_values_norm[player_action] = val_sum / counter
    return avg_action_values_norm


def compute_maxmin(
        node: Node,
        policy_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:  # returns values and action_probs
    """
    Compute maxmin value of the node. Value return array has shape (num_players, )
    Action Probs has shape (num_player_at_turn, num_actions)
    """
    player_action_vals = min_enemy_response(node)
    values, action_arr = max_player_action(player_action_vals, node, policy_temperature)
    return values, action_arr


def compute_maxavg(
        node: Node,
        policy_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:  # returns values and list of chosen actions
    """
    Compute maxavg value of the node. Value return array has shape (num_players, )
    Action Probs has shape (num_player_at_turn, num_actions)
    """
    player_action_vals = avg_enemy_response(node)
    values, action_arr = max_player_action(player_action_vals, node, policy_temperature)
    return values, action_arr


def inverse_action_indices(
        node: Node,
) -> dict[tuple[int, int], int]:
    """
    Builds index mapping (player, action) -> action_idx
    action_idx is the index of the action in the legal actions of the respective player.
    """
    res_dict = {}
    for player in node.game.players_at_turn():
        for action_idx, action in enumerate(node.game.available_actions(player)):
            res_dict[player, action] = action_idx
    return res_dict


def compute_q_values_as_arr(
        node: Node,
        player: int,
        action_probs: np.ndarray,  # shape (players_at_turn, num_actions)
) -> np.ndarray:  # shape (num_actions,)
    if node.children is None:
        raise Exception(f"Cannot compute q values on non-expanded node")
    # player and enemy indices
    player_idx = node.game.players_at_turn().index(player)
    enemies = list(enumerate(node.game.players_at_turn()))
    q_arr = np.zeros(shape=(node.game.num_actions,), dtype=float)
    # iterate all children
    for ja, ja_vals in node.get_joint_backward_estimate().items():
        # calculate joint enemy prob
        prob = 1
        for enemy_idx, enemy in enemies:
            if enemy != player:
                prob *= action_probs[enemy_idx, ja[enemy_idx]]
        # update q values
        q_arr[ja[player_idx]] += ja_vals[player] * prob
    return q_arr

def compute_q_values(
        node: Node,
        player: int,
        action_probs: np.ndarray,  # shape (players_at_turn, num_actions)
) -> list[float]:
    q_arr = compute_q_values_as_arr(
        node=node,
        player=player,
        action_probs=action_probs,
    )
    # remove illegal actions
    result_list = [q_arr[a] for a in node.game.available_actions(player)]
    return result_list


def action_indices_from_mask(
        actions: np.ndarray,  # shape (n,)
        is_invalid: np.ndarray,  # shape(n, num_actions), dtype bool -> true if invalid
) -> np.ndarray:  # action indices of shape (n,)
    if is_invalid.shape[0] != actions.shape[0]:
        raise ValueError(f"Invalid shapes: {actions.shape=}, {is_invalid.shape=}")
    # count invalid indices before chosen action
    offsets = np.cumsum(is_invalid, axis=-1)
    offset_per_action = np.take_along_axis(offsets, actions[..., np.newaxis], axis=1)
    action_indices = actions - np.squeeze(offset_per_action, axis=-1)
    return action_indices


def q_list_from_mask(
        q_arr: np.ndarray,  # shape(n, num_actions)
        is_valid: np.ndarray,  # shape(n, num_actions)
) -> list[list[float]]:
    if q_arr.shape[0] != is_valid.shape[0] or q_arr.shape[1] != is_valid.shape[1]:
        raise ValueError(f"Invalid shapes: {q_arr.shape=}, {is_valid.shape=}")
    result_list = []
    for i in range(q_arr.shape[0]):
        if is_valid[i].sum() == 0:
            raise ValueError(f"All actions are not allowed to be invalid")
        cur_q_arr = q_arr[i][is_valid[i]]
        result_list.append(list(cur_q_arr))
    return result_list

def ja_value_array(
        node: Node,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns ja_values sorted like itertools-combinations of shape (num_player, num_actions^num_player)
    Additionally returns joint action indices of shape (num_samples, num_actions^num_player, num_player)
    The arrays are from viewpoint of player, i.e. player actions are indexed first
    """
    if node.is_terminal():
        raise ValueError(f"Cannot compute joint actions on terminal node")
    num_p = node.game.num_players_at_turn()
    max_num_ja = node.game.num_actions ** num_p
    # init values and action with -2 to make errors obvious
    result_values = np.zeros(shape=(num_p, max_num_ja, num_p), dtype=float) - 2
    result_actions = np.zeros(shape=(num_p, max_num_ja, num_p), dtype=int) - 2
    backward_estimates = node.get_joint_backward_estimate()
    for ja_idx, item in enumerate(backward_estimates.items()):
        ja, values = item
        values_at_turn = values[node.game.players_at_turn()]
        for p_idx, player in enumerate(node.game.players_at_turn()):
            for offset in range(num_p):
                result_values[p_idx, ja_idx, offset] = values_at_turn[(p_idx + offset) % num_p]
                result_actions[p_idx, ja_idx, offset] = ja[(p_idx + offset) % num_p]
    return result_values, result_actions

