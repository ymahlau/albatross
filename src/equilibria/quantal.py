import numpy as np
import ctypes as ct

from src.cpp.lib import CPP_LIB


def compute_qne_equilibrium(
        available_actions: list[list[int]],  # maps player(index of player_at_turn) to available actions
        joint_action_list: list[tuple[int, ...]],
        joint_action_value_arr: np.ndarray,  # shape (num_joint_actions, num_player_at_turn)
        leader: int,
        num_iterations: int,
        temperature: float,
        random_prob: float = 0,
) -> tuple[list[float], list[np.ndarray]]:
    """
    Computes quantal nash equilibrium with the Regret Matching - Quantal Response Algorithm. For more info
    see the original paper:
    https://ojs.aaai.org/index.php/AAAI/article/view/16701
    """
    num_player = joint_action_value_arr.shape[1]
    if num_player != 2:
        raise ValueError(f"QR-RM only works for two player games")
    num_available_actions = np.asarray([len(available_actions[p]) for p in range(num_player)], dtype=ct.c_int)
    num_available_actions_p = num_available_actions.ctypes.data_as(ct.POINTER(ct.c_int))
    # available actions
    flat_action_list = [a for sublist in available_actions for a in sublist]
    available_actions_arr = np.asarray(flat_action_list, dtype=ct.c_int)
    available_actions_p = available_actions_arr.ctypes.data_as(ct.POINTER(ct.c_int))
    joint_actions_arr = np.asarray(joint_action_list, dtype=ct.c_int).flatten()
    joint_actions_p = joint_actions_arr.ctypes.data_as(ct.POINTER(ct.c_int))
    joint_action_value_arr_flat = joint_action_value_arr.astype(ct.c_double).flatten()
    joint_action_value_p = joint_action_value_arr_flat.ctypes.data_as(ct.POINTER(ct.c_double))
    # result arrays
    result_values = np.zeros(shape=(num_player,), dtype=ct.c_double)
    result_values_p = result_values.ctypes.data_as(ct.POINTER(ct.c_double))
    result_policies = np.zeros_like(available_actions_arr, dtype=ct.c_double)
    result_policies_p = result_policies.ctypes.data_as(ct.POINTER(ct.c_double))
    CPP_LIB.lib.rm_qr_cpp(
        num_available_actions_p,
        available_actions_p,
        joint_actions_p,
        joint_action_value_p,
        leader,
        num_iterations,
        temperature,
        random_prob,
        result_values_p,
        result_policies_p,
    )
    value_list = list(result_values)
    result_policy_list = []
    start_idx = 0
    for p in range(num_player):
        end_idx = start_idx + num_available_actions[p]
        result_policy_list.append(result_policies[start_idx:end_idx])
        start_idx = end_idx
    return value_list, result_policy_list


def compute_qse_equilibrium(
        available_actions: list[list[int]],  # maps player(index of player_at_turn) to available actions
        joint_action_list: list[tuple[int, ...]],
        joint_action_value_arr: np.ndarray,  # shape (num_joint_actions, num_player_at_turn)
        leader: int,
        temperature: float,
        num_iterations: int = 10,
        grid_size: int = 1000,
) -> tuple[list[float], list[np.ndarray]]:
    """
    Computes a Quantal Stackelberg Equilibrium: see the original paper:
    See https://www.ijcai.org/proceedings/2020/35 for more information
    """
    num_player = joint_action_value_arr.shape[1]
    if num_player != 2:
        raise ValueError(f"QR-RM only works for two player games")
    num_available_actions = np.asarray([len(available_actions[p]) for p in range(num_player)], dtype=ct.c_int)
    num_available_actions_p = num_available_actions.ctypes.data_as(ct.POINTER(ct.c_int))
    # available actions
    flat_action_list = [a for sublist in available_actions for a in sublist]
    available_actions_arr = np.asarray(flat_action_list, dtype=ct.c_int)
    available_actions_p = available_actions_arr.ctypes.data_as(ct.POINTER(ct.c_int))
    joint_actions_arr = np.asarray(joint_action_list, dtype=ct.c_int).flatten()
    joint_actions_p = joint_actions_arr.ctypes.data_as(ct.POINTER(ct.c_int))
    joint_action_value_arr_flat = joint_action_value_arr.astype(ct.c_double).flatten()
    joint_action_value_p = joint_action_value_arr_flat.ctypes.data_as(ct.POINTER(ct.c_double))
    # result arrays
    result_values = np.zeros(shape=(num_player,), dtype=ct.c_double)
    result_values_p = result_values.ctypes.data_as(ct.POINTER(ct.c_double))
    result_policies = np.zeros_like(available_actions_arr, dtype=ct.c_double)
    result_policies_p = result_policies.ctypes.data_as(ct.POINTER(ct.c_double))
    CPP_LIB.lib.qse_cpp(
        num_available_actions_p,
        available_actions_p,
        joint_actions_p,
        joint_action_value_p,
        leader,
        num_iterations,
        grid_size,
        temperature,
        result_values_p,
        result_policies_p,
    )
    value_list = list(result_values)
    result_policy_list = []
    start_idx = 0
    for p in range(num_player):
        end_idx = start_idx + num_available_actions[p]
        result_policy_list.append(result_policies[start_idx:end_idx])
        start_idx = end_idx
    return value_list, result_policy_list
