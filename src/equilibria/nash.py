import itertools
from typing import Optional

import numpy as np
import torch.multiprocessing as mp
from scipy.optimize import linprog

from src.cpp.lib import CPP_LIB


def calculate_nash_equilibrium(
        available_actions: list[list[int]],  # maps player(index of player_at_turn) to available actions
        joint_action_list: list[tuple[int, ...]],
        joint_action_value_arr: np.ndarray,  # shape (num_joint_actions, num_player_at_turn)
        use_cpp: bool = True,
        error_counter: Optional[mp.Value] = None
) -> tuple[list[float], list[np.ndarray]]:
    num_players = len(available_actions)
    if joint_action_value_arr.shape[1] != num_players:
        raise ValueError(f"Invalid array shape: {joint_action_value_arr.shape}")
    if use_cpp:
        value_list, policy_list = CPP_LIB.compute_nash(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_value_arr,
            error_counter=error_counter,
        )
        return value_list, policy_list
    else:
        if num_players != 2:
            raise ValueError("Nash Computation does not work for more than 2 players in python")
        # parse input
        shape = (len(available_actions[0]), len(available_actions[1]))
        p0_utils, p1_utils = np.empty(shape, dtype=float), np.empty(shape, dtype=float)
        # initialize normal form game
        for action_idx_0, action_0 in enumerate(available_actions[0]):
            for action_idx_1, action_1 in enumerate(available_actions[1]):
                ja_idx = joint_action_list.index((action_0, action_1))
                p0_utils[action_idx_0, action_idx_1] = joint_action_value_arr[ja_idx, 0]
                p1_utils[action_idx_0, action_idx_1] = joint_action_value_arr[ja_idx, 1]
        result = calculate_nash_equilibrium_python(p0_utils, p1_utils)
        if result is None:
            raise Exception(f"Nash could not be computed in python for unknown reason: {p0_utils=}, {p1_utils=}")
        values, action_probs = result
        value_list = []
        for player in range(2):
            value_list.append(values[player])
        policy_list = []
        for player in range(2):
            arr = np.empty(shape=(len(available_actions[player]),), dtype=float)
            for a_idx in range(len(available_actions[player])):
                arr[a_idx] = action_probs[(player, a_idx)]
            policy_list.append(arr)
        return value_list, policy_list


def feasibility(
        util1: np.ndarray,
        util2: np.ndarray,
        support1: list[int],
        support2: list[int],
) -> Optional[tuple[np.ndarray, dict[tuple[int, int], float]]]:  # values, action probs
    """
    Returns values of shape [2,] and action probabilities indexed by player-action -> prob
    """
    players = [0, 1]
    actions = [list(range(util1.shape[player])) for player in players]
    # constraint lists, _values is right hand side of equation
    equality_constraints = []
    equality_constraint_values = []
    inequality_constraints = []
    inequality_constraint_values = []
    # Create variable indices.
    # Each action has a probability of being chosen.
    variables = []
    for player in players:
        variables += [f"p{player}_{action}" for action in actions[player]]
    # Both players have a utility value
    variables += [f"v{player}" for player in players]
    # coefficient_template = np.zeros(len(variables), dtype=float)
    for player in players:
        # current player is always row-player
        support = support1 if player == 0 else support2
        other_support = support2 if player == 0 else support1
        other_player = 1 if player == 0 else 0
        util = util1 if player == 0 else util2.T
        # The player must be indifferent between all actions in the support and
        # prefer them over actions outside support. First and second constraint of Feasibility Program 1.
        for action in actions[player]:
            coefficients = np.zeros(len(variables), dtype=float)
            coefficients[variables.index(f"v{player}")] = -1
            for other_action in other_support:
                coefficients[variables.index(f"p{other_player}_{other_action}")] = util[action, other_action]
            if action in support:
                equality_constraints.append(coefficients)
                equality_constraint_values.append(0)
            else:
                inequality_constraints.append(coefficients)
                inequality_constraint_values.append(0)
        # All action probabilities of a player sum to 1.
        coefficients = np.zeros(len(variables), dtype=float)
        for action in actions[player]:
            coefficients[variables.index(f"p{player}_{action}")] = 1
        equality_constraints.append(coefficients)
        equality_constraint_values.append(1)
        # All actions in the support have a probability greater than or equal to 0,
        # and all other actions have a probability of 0.
        for action in actions[player]:
            coefficients = np.zeros(len(variables), dtype=float)
            coefficients[variables.index(f"p{player}_{action}")] = -1
            if action in support:
                inequality_constraints.append(coefficients)
                inequality_constraint_values.append(0)
            else:
                equality_constraints.append(coefficients)
                equality_constraint_values.append(0)
    # boundary conditions for variables
    bounds = [
        (0, 1) if variable.startswith("p") else (None, None)
        for variable in variables
    ]
    # Solve the problem.
    # we use zeros as coefficients c to make all solutions equally viable
    result = linprog(
        c=np.zeros(len(variables), dtype=float),
        A_ub=np.asarray(inequality_constraints),
        b_ub=np.asarray(inequality_constraint_values),
        A_eq=np.asarray(equality_constraints),
        b_eq=np.asarray(equality_constraint_values),
        method='highs',
        bounds=bounds,
    )

    if result.success:
        # parse probabilities
        player_action_res: dict[tuple[int, int], float] = {}
        for player in players:
            for a in actions[player]:
                player_action_res[(player, a)] = result.x[variables.index(f"p{player}_{a}")]
        # parse values
        value_res = np.zeros((2, ), dtype=float)
        for player in players:
            value_res[player] = result.x[variables.index(f"v{player}")]
        return value_res, player_action_res
    return None


def is_cond_dominated(
    util: np.ndarray,  # utilities for the current player
    action: int,  # action to check
    action_choices: list[int],  # action choices for current player
    action_responses: list[int],  # action choices for other player
    player: int,  # player id, either 1 or 2
) -> bool:
    """
    Check if an action is conditionally dominated given action profiles of both players
    """
    # transpose to make current player always the row player
    if player == 2:
        util = util.T
    # remove action from action choices if it exists
    if action in action_choices:
        action_choices = action_choices.copy()
        action_choices.remove(action)
    # check if any other possible action leads to strictly better utility
    for candidate in action_choices:
        if np.all(util[candidate, action_responses] > util[action, action_responses]):
            return True
    return False


def calculate_nash_equilibrium_python(
        util1: np.ndarray,
        util2: np.ndarray
) -> Optional[tuple[np.ndarray, dict[tuple[int, int], float]]]:
    """
    Calculates the Nash equilibrium of a 2-player game using Algorithm 1 of the Paper:
    https://www.sciencedirect.com/science/article/pii/S0899825606000935
    util1: The utility matrix of player 1.
    util2: The utility matrix of player 2.
    return: A Nash equilibrium of the game.
    """
    # Check if the utility matrices have the same shape.
    if util1.shape != util2.shape:
        raise ValueError("The utility matrices must have the same shape.")
    # all actions and supports
    actions1, actions2 = list(range(util1.shape[0])), list(range(util2.shape[1]))
    support_size_1 = list(range(1, len(actions1) + 1))
    support_size_2 = list(range(1, len(actions2) + 1))
    support_sizes = list(itertools.product(support_size_1, support_size_2))
    # The order of the support sizes is important for efficiency as stated in the paper.
    sorted_support_sizes = sorted(support_sizes, key=lambda x: (abs(x[0] - x[1]), x[0] + x[1]))
    # iterate all combinations of support sizes
    for x1, x2 in sorted_support_sizes:
        all_supports_1: list[tuple[int, ...]] = list(itertools.combinations(actions1, x1))
        for support1_tpl in all_supports_1:
            # convert tuple to list
            support1_list: list[int] = list(support1_tpl)
            # Prune search space for efficiency.
            actions2prime = [
                action
                for action in actions2
                if not is_cond_dominated(util2, action, actions2, support1_list, 2)
            ]
            if any(
                is_cond_dominated(util1, action, support1_list, actions2prime, 1)
                for action in support1_list
            ):
                continue
            # iterate over all possible support of player 2
            all_supports_2: list[tuple[int, ...]] = list(itertools.combinations(actions2prime, x2))
            for support2_tpl in all_supports_2:
                # convert tuple to list
                support2_list = list(support2_tpl)
                # check if a solution exists
                if any(
                    is_cond_dominated(util2, action, support2_list, support1_list, 2)
                    for action in support2_list
                ):
                    continue
                # solve linear feasibility program and return solution if successful
                solution = feasibility(util1, util2, support1_list, support2_list)
                if solution is not None:
                    values, action_probs = solution
                    return values, action_probs
    return None
