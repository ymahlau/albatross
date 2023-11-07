import itertools
from enum import Enum

import numpy as np

from src.game.normal_form.normal_form import NormalFormConfig

class NFGType(Enum):
    GENERAL = 'GENERAL'
    ZERO_SUM = 'ZERO_SUM'
    FULL_COOP = 'FULL_COOP'

def get_random_matrix_cfg(
        num_actions_per_player: list[int],
        nfg_type: NFGType,
) -> NormalFormConfig:
    """
    Generates Normal Form game config for a matrix game with random (uniform) entries in the interval [-1, 1]
    """
    num_player = len(num_actions_per_player)
    if nfg_type == NFGType.ZERO_SUM and num_player != 2:
        raise ValueError(f"Zero sum games only work with two player, but got {num_player}")
    available_actions = [
        list(range(num_a)) for num_a in num_actions_per_player
    ]
    joint_actions = list(itertools.product(*available_actions))
    result_dict = {}
    for ja in joint_actions:
        if nfg_type == NFGType.ZERO_SUM:
            value_p0 = np.random.random(size=(1,))
            value_tpl = (value_p0.item(), - value_p0.item())
        elif nfg_type == NFGType.FULL_COOP:
            value_p0 = np.random.random(size=(1,))
            value_tpl = tuple([value_p0.item() for _ in range(num_player)])
        else:
            values = np.random.random(size=(num_player, )) * 2 - 1
            value_tpl = tuple(values.tolist())
        result_dict[ja] = value_tpl
    cfg = NormalFormConfig(ja_dict=result_dict)
    return cfg
