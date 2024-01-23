from enum import Enum
from typing import Optional

import numpy as np

from src.cpp.lib import CPP_LIB


class SbrMode(Enum):
    MSA = 'MSA'
    EMA = 'EMA'
    ADAM = 'ADAM'
    BB = 'BB'
    POLYAK = 'POLYAK'  # n^-2/3 as step size
    NAGURNEY = 'NAGURNEY'  # step size by 1, 1/2, 1/2, 1/3, 1/3, 1/3, ... n times 1/n
    SRA = 'SRA'
    BB_POLYAK = 'BB_POLYAK'  # combination of different methods, does not work very well
    BB_NAGURNEY = 'BB_NAGURNEY'
    BB_MSA = 'BB_MSA'
    SRA_NAGURNEY = 'SRA_NAGURNEY'


def compute_logit_equilibrium(
        available_actions: list[list[int]],  # maps player(index of player_at_turn) to available actions
        joint_action_list: list[tuple[int, ...]],
        joint_action_value_arr: np.ndarray,  # shape (num_joint_actions, num_player_at_turn)
        num_iterations: int,
        epsilon: float,
        temperatures: list[float],
        initial_policies: Optional[list[np.ndarray]] = None,
        hp_0: Optional[float] = None,
        hp_1: Optional[float] = None,
        sbr_mode: SbrMode = SbrMode.MSA,
) -> tuple[list[float], list[np.ndarray], float]:  # values, policies, policy_error
    num_player = joint_action_value_arr.shape[1]
    # sanity checks
    if len(joint_action_value_arr.shape) != 2:
        raise ValueError(f"Invalid action value shape: {joint_action_value_arr.shape}")
    if initial_policies is not None:
        if len(initial_policies) != num_player:
            raise ValueError(f"Invalid initial policies: {initial_policies}")
        for idx, p in enumerate(initial_policies):
            if len(p.shape) != 1 or p.shape[0] != len(available_actions[idx]):
                raise ValueError(f"Invalid initial policy for player {idx}: {p}")
            if np.any(np.abs(np.sum(p) - 1) > 0.001) or np.any(np.isnan(p)) \
                    or np.any(np.isinf(p)):
                raise Exception(f"Initial policies do not yield a prob dist: {p}")
    if len(temperatures) == 1:
        temperatures = [temperatures[0] for _ in range(num_player)]
    if len(temperatures) != num_player:
        raise ValueError("Every Player needs a weighting config")
    # efficient c++ implementation of logit-equilibrium solver
    sbr_code = 0
    if sbr_mode == SbrMode.EMA or sbr_mode == SbrMode.EMA.value:
        sbr_code = 1
        if hp_0 is None:
            hp_0 = 0.5
        if hp_0 <= 0 or hp_0 >= 1:
            raise ValueError(f"Invalid EMA hyperparameter: {hp_0=}")
    elif sbr_mode == SbrMode.ADAM or sbr_mode == SbrMode.ADAM.value:
        sbr_code = 2
        if hp_0 is None:
            hp_0 = 0.01
        if hp_0 <= 0:
            raise ValueError(f"Invalid EMA hyperparameter: {hp_0=}")
    elif sbr_mode == SbrMode.BB or sbr_mode == SbrMode.BB.value:
        sbr_code = 3
    elif sbr_mode == SbrMode.POLYAK or sbr_mode == SbrMode.POLYAK.value:
        sbr_code = 4
    elif sbr_mode == SbrMode.NAGURNEY or sbr_mode == SbrMode.NAGURNEY.value:
        sbr_code = 5
    elif sbr_mode == SbrMode.BB_POLYAK or sbr_mode == SbrMode.BB_POLYAK.value:
        sbr_code = 6
    elif sbr_mode == SbrMode.BB_NAGURNEY or sbr_mode == SbrMode.BB_NAGURNEY.value:
        sbr_code = 7
    elif sbr_mode == SbrMode.BB_MSA or sbr_mode == SbrMode.BB_MSA.value:
        sbr_code = 8
    elif sbr_mode == SbrMode.SRA or sbr_mode == SbrMode.SRA.value:
        # Self-regulating averaging. hp_0 is gamma and hp_1 is large GAMMA.
        # the paper proposes gamma in [0.01, 0.5] and GAMMA in [1.5, 2]
        sbr_code = 9
        if hp_0 is None:
            hp_0 = 0.3
        if hp_1 is None:
            hp_1 = 1.8
        if hp_0 < 0.01 or hp_0 > 0.5:
            raise ValueError(f"Invalid EMA hyperparameter: {hp_0=}")
        if hp_1 < 1.5 or hp_1 > 2:
            raise ValueError(f"Invalid EMA hyperparameter: {hp_1=}")
    elif sbr_mode == SbrMode.SRA_NAGURNEY or sbr_mode == SbrMode.SRA_NAGURNEY.value:
        sbr_code = 10
    # default values to satisfy static types. Those are unused
    if hp_0 is None:
        hp_0 = -1
    if hp_1 is None:
        hp_1 = -1
    return CPP_LIB.compute_logit_equilibrium(
        available_actions=available_actions,
        joint_action_list=joint_action_list,
        joint_action_value_arr=joint_action_value_arr,
        num_iterations=num_iterations,
        epsilon=epsilon,
        temperatures=temperatures,
        initial_policies=initial_policies,
        hp_0=hp_0,
        hp_1=hp_1,
        sbr_mode=sbr_code,
    )

