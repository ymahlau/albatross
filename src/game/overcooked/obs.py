import math

import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

from src.game.overcooked.state import get_pot_state
from src.game.overcooked.utils import OBJECT_NAMES


def get_general_features(gridworld: OvercookedGridworld) -> np.ndarray:  # shape (2, w, h, 6)
    # playing field
    w, h = gridworld.width, gridworld.height
    arr = np.zeros(shape=(2, w, h, 6))
    for x in range(w):
        for y in range(h):
            terrain = gridworld.get_terrain_type_at_pos((x, y))
            if terrain == ' ':  # playing field for player
                arr[:, x, y, 0] = 1
            elif terrain == 'P':  # pot location
                arr[:, x, y, 1] = 1
            elif terrain == 'X':  # counter
                arr[:, x, y, 2] = 1
            elif terrain == 'O':  # Onions
                arr[:, x, y, 3] = 1
            elif terrain == 'D':  # Dish
                arr[:, x, y, 4] = 1
            elif terrain == 'S':  # Serve
                arr[:, x, y, 5] = 1
            else:
                raise Exception(f"Unknown terrain type: {terrain}")
    return arr


def get_player_features(
        state: OvercookedState,
        gridworld: OvercookedGridworld,
) -> np.ndarray:  # shape (2, w, h, 6)
    w, h = gridworld.width, gridworld.height
    arr = np.zeros(shape=(2, w, h, 6))
    # positions
    pos = state.player_positions
    arr[0, pos[0][0], pos[0][1], 0] = 1
    arr[1, pos[1][0], pos[1][1], 0] = 1
    # field looked at
    ori = state.player_orientations
    f0 = (pos[0][0] + ori[0][0], pos[0][1] + ori[0][1])
    f1 = (pos[1][0] + ori[1][0], pos[1][1] + ori[1][1])
    arr[0, f0[0], f0[1], 1] = 1
    arr[1, f1[0], f1[1], 1] = 1
    # one hot item: onion, dish, soup, none
    for p in range(2):
        p_obj = state.players[p]
        item_id = 3 if not p_obj.has_object() else OBJECT_NAMES.index(p_obj.get_object().name)
        arr[p, :, :, 2 + item_id] += 1
    return arr


def get_pot_features(
        state: OvercookedState,
        gridworld: OvercookedGridworld,
) -> np.ndarray:    # shape (2, w, h, 4*num_pots)
    # num_onion, is_cooking, cook_time_remaining, is_done per pot
    w, h = gridworld.width, gridworld.height
    pot_states = get_pot_state(state, gridworld)
    num_pots = len(pot_states)
    arr = np.zeros(shape=(2, w, h, 4*num_pots))
    for idx, ps in enumerate(pot_states):
        if 0 <= ps <= 3:  # zero to three onions
            arr[:, :, :, 4 * idx] = ps / 3
            # arr[:, :, :, 4 * idx + 1] = 0  # no need to set is_cooking, is already zero
            arr[:, :, :, 4 * idx + 2] = 1
            # arr[:, :, :, 4 * idx + 3] = 0  # no need to set is_done, is already zero
        elif ps == 4:
            arr[:, :, :, 4 * idx] = 1
            # arr[:, :, :, 4 * idx + 1] = 0  # no need to set is_cooking, is already zero
            # arr[:, :, :, 4 * idx + 2] = 0  # no need to set remaining cook time, is already zero
            arr[:, :, :, 4 * idx + 3] = 1
        elif 5 <= ps <= 23:
            arr[:, :, :, 4 * idx] = 1
            arr[:, :, :, 4 * idx + 1] = 1
            arr[:, :, :, 4 * idx + 2] = (ps - 4) / 20
            # arr[:, :, :, 4 * idx + 3] = 0  # no need to set is_done, is already zero
        else:
            raise Exception(f"Unknown pot state")
    return arr


def temperature_features(
        gridworld: OvercookedGridworld,
        temperatures: list[float],
        single_temperature: bool,
) -> np.ndarray:  # shape (2, w, h, 1)
    w, h = gridworld.width, gridworld.height
    arr = np.zeros((2, w, h, 1), dtype=float)
    if single_temperature:
        arr[...] = temperatures[0]
    else:
        arr[0, ...] = temperatures[1]
        arr[1, ...] = temperatures[0]
    return arr


def get_obs_shape_and_padding(
        gridworld: OvercookedGridworld,
        temperature_input: bool,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:  # return obs-shape and padding
    # determine obs shape:
    # 6 general feat., 2*6 player features, 4*num_pots, + 1 time remaining + maybe temperature input
    num_pots = len(gridworld.get_pot_locations())
    z_dim = 19 + 4 * num_pots + int(temperature_input)
    w, h = gridworld.width, gridworld.height
    # determine padding for observations: left, right, up, down
    left, right, up, down = 0, 0, 0, 0
    w_pad, h_pad = w, h
    if w > h:
        num_padding = w - h
        up = math.ceil(num_padding / 2)
        down = num_padding - up
        h_pad = w
    elif w < h:
        num_padding = h - w
        up = math.ceil(num_padding / 2)
        down = num_padding - up
        w_pad = h
    # put together
    padding = (left, right, up, down)
    obs_shape = (w_pad, h_pad, z_dim)
    return obs_shape, padding

