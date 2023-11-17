import numpy as np

from src.cpp.lib import CPP_LIB


def calculate_nash_equilibrium(
        available_actions: list[list[int]],  # maps player(index of player_at_turn) to available actions
        joint_action_list: list[tuple[int, ...]],
        joint_action_value_arr: np.ndarray,  # shape (num_joint_actions, num_player_at_turn)
        error_counter = None,
        use_cpp: bool = True,  # legacy, no longer necessary
) -> tuple[list[float], list[np.ndarray]]:
    num_players = len(available_actions)
    if joint_action_value_arr.shape[1] != num_players:
        raise ValueError(f"Invalid array shape: {joint_action_value_arr.shape}")
    value_list, policy_list = CPP_LIB.compute_nash(
        available_actions=available_actions,
        joint_action_list=joint_action_list,
        joint_action_value_arr=joint_action_value_arr,
        error_counter=error_counter,
    )
    return value_list, policy_list
    