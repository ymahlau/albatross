
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

def update_pot_state_arr(game: OvercookedGameSlow, state_arr: np.ndarray) -> np.ndarray:
    w, h = game.gridworld.shape
    state_arr = state_arr.reshape(h, w)
    a = 1
    

def overcooked_fast_from_slow(
    game: OvercookedGameSlow,
) -> OvercookedGameFast:
    # IMPORTANT: reset will produce a different result than the original as this changes the base config
    w, h = game.gridworld.shape
    board = board_from_slow(game)
    assert game.env is not None
    players = game.env.state.players
    player_items = [
        OvercookedGameFast.get_item_code_from_name(players[0].held_object.name if players[0].held_object is not None else None),
        OvercookedGameFast.get_item_code_from_name(players[0].held_object.name if players[0].held_object is not None else None)
    ]
    ors = [
        OvercookedGameFast.get_code_from_or(players[0].orientation),
        OvercookedGameFast.get_code_from_or(players[1].orientation)
    ]
    player_pos = (
        (players[0].position[0], players[0].position[1], ors[0], player_items[0]),
        (players[1].position[0], players[1].position[1], ors[1], player_items[1]),
    )
    base_cfg = OvercookedGameConfig(
        w=w,
        h=h,
        board=board,
        start_pos=player_pos,
    )
    new_game = OvercookedGameFast(base_cfg)
    state_arr = new_game.get_state_array()
    new_state_arr = update_pot_state_arr(game, state_arr)
    new_game.update_tile_states(new_state_arr.tolist())
    return new_game
    
