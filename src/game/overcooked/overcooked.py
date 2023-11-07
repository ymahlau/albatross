import ctypes as ct
from typing import Optional

import numpy as np

from src.cpp.lib import Struct, CPP_LIB
from src.game.game import Game
from src.game.overcooked.config import OvercookedGameConfig


class OvercookedGame(Game):
    def __init__(self, cfg: OvercookedGameConfig, state_p: Optional[ct.POINTER(Struct)] = None):
        super().__init__(cfg)
        self.cfg = cfg
        # state pointer
        self.state_p = state_p
        if self.state_p is None:
            self._init_cpp()
        self.action_list = [0, 1, 2, 3, 4, 5]
        self.obs_dict: dict[int, np.ndarray] = dict()
        self.is_closed = False
        self.id_dict = {a: a for a in range(6)}

    def _init_cpp(self):
        board_np = np.asarray(self.cfg.board, dtype=ct.c_int)
        board_pt = board_np.ctypes.data_as(ct.POINTER(ct.c_int))
        start_pos_np = np.asarray(self.cfg.start_pos, dtype=ct.c_int)
        start_pos_pt = start_pos_np.ctypes.data_as(ct.POINTER(ct.c_int))
        self.state_p = CPP_LIB.lib.init_overcooked_cpp(
            self.cfg.w,
            self.cfg.h,
            board_pt,
            start_pos_pt,
            self.cfg.horizon,
            self.cfg.cooking_time,
            self.cfg.reward_cfg.placement_in_pot,
            self.cfg.reward_cfg.dish_pickup,
            self.cfg.reward_cfg.soup_pickup,
            self.cfg.reward_cfg.soup_delivery,
            self.cfg.reward_cfg.start_cooking,
        )

    def reset_saved_properties(self):
        self.obs_dict: dict[int, np.ndarray] = dict()

    def _step(self, actions: tuple[int, ...]) -> tuple[np.ndarray, bool, dict]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        # fill actions of players not at turn with zeros
        action_arr = np.zeros(shape=(2,), dtype=ct.c_int)
        action_arr[0] = actions[0]
        action_arr[1] = actions[1]
        action_p = action_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # perform step
        reward = CPP_LIB.lib.step_overcooked_cpp(self.state_p, action_p)
        self.reset_saved_properties()
        # compute return values
        reward_arr = np.asarray([reward, reward], dtype=float)
        done = self.turns_played >= self.cfg.horizon - 1  # turns played is only updated after the step
        return reward_arr, done, {}

    def close(self):
        if self.is_closed:
            raise ValueError("Cannot call close twice")
        self.is_closed = True
        CPP_LIB.lib.close_overcooked_cpp(self.state_p)

    def __del__(self):
        if not self.is_closed:
            self.close()

    def _reset(self):
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        CPP_LIB.lib.close_overcooked_cpp(self.state_p)
        self._init_cpp()
        self.reset_saved_properties()

    def render(self):
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        str_repr = self.get_str_repr()
        print(str_repr)

    def _get_copy(self) -> "Game":
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        state_p2 = CPP_LIB.lib.clone_overcooked_cpp(self.state_p)
        cpy = OvercookedGame(
            cfg=self.cfg,
            state_p=state_p2,
        )
        # copy properties
        cpy.obs_dict = self.obs_dict.copy()
        return cpy

    def __eq__(self, other: "OvercookedGame") -> bool:
        if not isinstance(other, OvercookedGame):
            return False
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        equals = CPP_LIB.lib.equals_overcooked_cpp(self.state_p, other.state_p)
        return equals

    def available_actions(self, player: int) -> list[int]:
        if self.turns_played < self.cfg.horizon:
            return self.action_list
        return []

    def players_at_turn(self) -> list[int]:
        if self.turns_played < self.cfg.horizon:
            return [1, 2]
        return []

    def players_alive(self) -> list[int]:
        return [1, 2]

    def is_terminal(self) -> bool:
        return self.turns_played >= self.cfg.horizon

    def get_symmetry_count(self):
        return 1

    def get_obs_shape(self, never_flatten=False) -> tuple[int, ...]:
        max_dim = max(self.cfg.h, self.cfg.w)
        return max_dim, max_dim, 16

    def get_obs(self, symmetry: Optional[int] = 0, temperatures: Optional[list[float]] = None,
                single_temperature: Optional[bool] = None) -> tuple[
        np.ndarray,
        dict[int, int],
        dict[int, int],
    ]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        arr = np.zeros(shape=self.get_obs_shape(), dtype=ct.c_float)
        arr_p = arr.ctypes.data_as(ct.POINTER(ct.c_float))
        CPP_LIB.lib.construct_overcooked_encoding_cpp(
            self.state_p,
            arr_p,
        )
        return arr, self.id_dict, self.id_dict

    def get_str_repr(self) -> str:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        arr = ct.create_string_buffer(self.cfg.w * self.cfg.h * 3)
        CPP_LIB.lib.char_overcooked_matrix_cpp(self.state_p, arr)
        str_repr = arr.value.decode("utf-8")
        return str_repr
