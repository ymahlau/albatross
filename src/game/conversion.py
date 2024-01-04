
import numpy as np
from src.game.overcooked.config import OvercookedGameConfig
from src.game.overcooked.overcooked import OvercookedGame as OvercookedGameFast
from src.game.overcooked_slow.overcooked import OvercookedSlowConfig, OvercookedGame as OvercookedGameSlow
from src.game.overcooked_slow.state import SimplifiedOvercookedState


def overcooked_slow_from_fast(game: OvercookedGameFast, layout_abbr: str) -> OvercookedGameSlow:
    abbrev_dict = {
        'cr': 'cramped_room',
        'aa': 'asymmetric_advantages',
        'co': 'coordination_ring',
        'fc': 'forced_coordination',
        'cc': 'counter_circuit_o_1order',
    }
    slow_game_cfg = OvercookedSlowConfig(
        overcooked_layout=abbrev_dict[layout_abbr],
        horizon=400,
        disallow_soup_drop=False,
        mep_reproduction_setting=True,
        mep_eval_setting=True,
        flat_obs=True,
    )
    game_slow = OvercookedGameSlow(slow_game_cfg)
    state_dict = game.generate_oc_state_dict()
    state = SimplifiedOvercookedState(state_dict)
    game_slow.set_state(state)
    return game_slow

def board_from_slow(game: OvercookedGameSlow) -> list[list[int]]:
    board_char_list = game.gridworld.terrain_mtx
    result_list = []
    charmap = {
        ' ': 0,
        'X': 1,
        'D': 2,
        'O': 3,
        'P': 4,
        'S': 5,
    }
    for row in board_char_list:
        row_list = []
        for char in row:
            row_list.append(charmap[char])
        result_list.append(row_list)
    return result_list
    
