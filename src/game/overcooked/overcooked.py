import ctypes as ct
import random
from typing import Optional

import numpy as np

from src.cpp.lib import Struct, CPP_LIB
from src.game.game import Game
from src.game.overcooked.config import OvercookedGameConfig


class OvercookedGame(Game):
    def __init__(
        self, 
        cfg: OvercookedGameConfig, 
        state_p = None,  # Optional[ct.POINTER(Struct)]
        ):
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
        self.last_info: Optional[dict[str, int]] = None
        self.stuck_counter: list[int] = [0, 0]

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
        self.stuck_counter = [0, 0]
        self.last_info: Optional[dict[str, int]] = None
        
    def _test_stuck(self, cur_info: dict[str, int], player: int) -> bool:
        if self.last_info is None:
            return False
        for k, v in cur_info.items():
            if not k.startswith(f'p{player}'):
                continue
            if v != self.last_info[k]:
                return False
        return True

    def _step(self, actions: tuple[int, ...]) -> tuple[np.ndarray, bool, dict]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        # if agent is stuck perform random action
        if self.cfg.unstuck_behavior:
            for player in range(2):
                if self.stuck_counter[player] == 3 and actions[player] in [4, 5]:
                    action_list = list(actions)
                    action_list[player] = random.randint(0, 3)
                    actions = tuple(action_list)
        # fill actions of players not at turn with zeros
        action_arr = np.zeros(shape=(2,), dtype=ct.c_int)
        action_arr[0] = actions[0]
        action_arr[1] = actions[1]
        # perform step
        action_p = action_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        reward = CPP_LIB.lib.step_overcooked_cpp(self.state_p, action_p)
        self.obs_dict: dict[int, np.ndarray] = dict()
        # compute return values
        reward_arr = np.asarray([reward, reward], dtype=float) * self.cfg.reward_scaling_factor
        done = self.turns_played >= self.cfg.horizon - 1  # turns played is only updated after the step
        # test if stuck
        if self.cfg.unstuck_behavior:
            cur_info = self.get_player_info()
            for player in range(2):
                if self._test_stuck(cur_info, player):
                    self.stuck_counter[player] += 1
            self.last_info = cur_info
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
        cpy.stuck_counter = self.stuck_counter.copy()
        cpy.last_info = self.last_info.copy() if self.last_info is not None else None
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
            return [0, 1]
        return []

    def players_alive(self) -> list[int]:
        return [1, 2]

    def is_terminal(self) -> bool:
        return self.turns_played >= self.cfg.horizon

    def get_symmetry_count(self):
        return 1

    def get_obs_shape(self, never_flatten=False) -> tuple[int, ...]:
        max_dim = max(self.cfg.h, self.cfg.w)
        return max_dim, max_dim, 16 + self.cfg.temperature_input

    def get_obs(
            self, 
            symmetry: Optional[int] = 0, 
            temperatures: Optional[list[float]] = None,
            single_temperature: Optional[bool] = None
    ) -> tuple[
        np.ndarray,
        dict[int, int],
        dict[int, int],
    ]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        obs_list = []
        for player in range(2):
            arr = np.zeros(shape=self.get_obs_shape(), dtype=ct.c_float)
            arr_p = arr.ctypes.data_as(ct.POINTER(ct.c_float))
            temp_in = 0
            if temperatures is not None and not self.cfg.temperature_input:
                raise ValueError("Cannot process temperatures if cfg does not specify temp input")
            if self.cfg.temperature_input:
                single_temp = self.cfg.single_temperature_input if single_temperature is None else single_temperature
                if temperatures is None:
                    raise ValueError(f"Need temperatures to generate encoding")
                if single_temp and len(temperatures) != 1:
                    raise ValueError(f"Cannot process multiple temperatures if single temperature input specified")
                if not single_temp and len(temperatures) != self.num_players:
                    raise ValueError(f"Invalid temperature length: {temperatures}")
                temp_in = temperatures[0] if single_temp else temperatures[1 - player]
            # scale temperature input to reasonable range
            temp_in = temp_in / 10
            CPP_LIB.lib.construct_overcooked_encoding_cpp(
                self.state_p,
                arr_p,
                player,
                self.cfg.temperature_input,
                temp_in,
            )
            obs_list.append(arr)
        result = np.stack(obs_list, axis=0)
        return result, self.id_dict, self.id_dict

    def get_str_repr(self) -> str:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        arr = ct.create_string_buffer(self.cfg.w * self.cfg.h * 3)
        CPP_LIB.lib.char_overcooked_matrix_cpp(self.state_p, arr)
        str_repr = arr.value.decode("utf-8")
        return str_repr
    
    def get_player_info(self) -> dict[str, int]:
        arr = np.zeros(shape=(8,), dtype=ct.c_int)
        arr_p = arr.ctypes.data_as(ct.POINTER(ct.c_int))
        CPP_LIB.lib.get_player_infos_cpp(self.state_p, arr_p)
        return {
            "p0_x": arr[0],
            "p0_y": arr[1],
            "p0_or": arr[2],
            "p0_item": arr[3],
            "p1_x": arr[4],
            "p1_y": arr[5],
            "p1_or": arr[6],
            "p1_item": arr[7],
        }
