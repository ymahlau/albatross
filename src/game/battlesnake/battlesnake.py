import ctypes as ct
import math
from typing import Optional

import numpy as np
import torch

from src.cpp.lib import Struct, CPP_LIB
from src.game.game import Game
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig, post_init_battlesnake_cfg, validate_battlesnake_cfg
from src.game.battlesnake.battlesnake_enc import num_layers_general, layers_per_player, layers_per_enemy
from src.game.battlesnake.battlesnake_rewards import get_battlesnake_reward_func_from_cfg
from src.game.battlesnake.state import BattleSnakeState
from src.game.utils import int_to_perm

UP: int = 0
RIGHT: int = 1
DOWN: int = 2
LEFT: int = 3


class BattleSnakeGame(Game):
    def __init__(
            self,
            cfg: BattleSnakeConfig,
            state_p: Optional[ct.POINTER(Struct)] = None
    ):
        super().__init__(cfg)
        self.is_closed = False
        self.cfg = cfg
        self.turns_played = self.cfg.init_turns_played
        # state pointer
        self.state_p = state_p
        if self.state_p is None:
            # this is the first time the game was started, run post init and validate
            post_init_battlesnake_cfg(self.cfg)
            validate_battlesnake_cfg(self.cfg)
            self._init_cpp()
        # attributes for saving the current game state
        self.obs_dict: dict[int, np.ndarray] = dict()
        self.available_actions_save: dict[int, list[int]] = dict()
        self.players_at_turn_save: Optional[list[int]] = None
        self.players_at_turn_last: Optional[list[int]] = None  # property of last step
        self.players_alive_save: Optional[list[int]] = None
        self.players_alive_last: Optional[list[int]] = None  # property of last step
        self.reward_func = get_battlesnake_reward_func_from_cfg(self.cfg.reward_cfg)

    def _init_cpp(self):
        # snakes
        spawn_snakes_randomly = True if self.cfg.init_snake_pos is None else False
        if self.cfg.init_snake_pos is not None:
            # find the longest body, this determines the array shape
            snake_pos = {}  # we need a separate dict to not alter the config object
            body_lengths = []
            for s in range(self.cfg.num_players):
                cur_snake_pos = []
                for pos in self.cfg.init_snake_pos[s]:
                    cur_snake_pos.append((pos[0], pos[1]))
                cur_snake_pos = list(dict.fromkeys(cur_snake_pos))  # remove duplicates
                snake_pos[s] = cur_snake_pos
                body_lengths.append(len(cur_snake_pos))
            max_body_len = max(body_lengths)
            body_len_arr = np.asarray(body_lengths, dtype=ct.c_int)
            body_len_p = body_len_arr.ctypes.data_as(ct.POINTER(ct.c_int))
            snake_pos_arr = np.zeros(shape=(self.cfg.num_players, max_body_len, 2), dtype=ct.c_int) - 1
            for s in range(self.cfg.num_players):  # convert dictionary to numpy array
                for i, pos in enumerate(snake_pos[s]):
                    snake_pos_arr[s, i, 0] = pos[0]
                    snake_pos_arr[s, i, 1] = pos[1]
            snake_pos_p = snake_pos_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        else:
            body_len_p = ct.cast(0, ct.POINTER(ct.c_int))  # NULL-Pointer
            snake_pos_p = ct.cast(0, ct.POINTER(ct.c_int))  # NULL-Pointer
            max_body_len = -1
        # snake length
        snake_len_arr = np.asarray(self.cfg.init_snake_len, dtype=ct.c_int)
        snake_len_p = snake_len_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # snakes alive
        snake_alive_arr = np.asarray(self.cfg.init_snakes_alive, dtype=bool)
        snake_alive_p = snake_alive_arr.ctypes.data_as(ct.POINTER(ct.c_bool))
        # health
        snake_health_arr = np.asarray(self.cfg.init_snake_health, dtype=ct.c_int)
        snake_health_p = snake_health_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        snake_max_health_arr = np.asarray(self.cfg.max_snake_health, dtype=ct.c_int)
        snake_max_health_p = snake_max_health_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # food
        if self.cfg.init_food_pos is None:
            num_init_food = -1  # -1 is signal for random food spawning
            food_pos_p = ct.cast(0, ct.POINTER(ct.c_int))  # NULL-Pointer
        elif not self.cfg.init_food_pos:
            num_init_food = -2  # flag to indicate that no food should be spawned at beginning
            food_pos_p = ct.cast(0, ct.POINTER(ct.c_int))
        else:
            num_init_food = len(self.cfg.init_food_pos)
            np_arr = np.asarray(self.cfg.init_food_pos, dtype=ct.c_int)
            food_pos_p = np_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # hazards, we need to transpose because cpp uses flattened array (this is more efficient)
        hazard_arr = np.zeros(shape=(self.cfg.h, self.cfg.w), dtype=bool)
        for hazard_tile in self.cfg.init_hazards:
            hazard_arr[hazard_tile[1], hazard_tile[0]] = True
        hazards_p = hazard_arr.ctypes.data_as(ct.POINTER(ct.c_bool))
        # c++ call
        self.state_p = CPP_LIB.lib.init_cpp(
            self.cfg.w,
            self.cfg.h,
            self.cfg.num_players,
            self.cfg.min_food,
            self.cfg.food_spawn_chance,
            self.cfg.init_turns_played,
            spawn_snakes_randomly,
            body_len_p,
            max_body_len,
            snake_pos_p,
            num_init_food,
            food_pos_p,
            snake_alive_p,
            snake_health_p,
            snake_len_p,
            snake_max_health_p,
            self.cfg.wrapped,
            self.cfg.royale,
            self.cfg.shrink_n_turns,
            self.cfg.hazard_damage,
            hazards_p,
        )

    def reset_saved_properties(self):
        self.obs_dict: dict[int, np.ndarray] = dict()
        self.available_actions_save: dict[int, list[int]] = dict()
        self.players_at_turn_save = None
        self.players_at_turn_last = None
        self.players_alive_save = None
        self.players_alive_last = None

    def get_obs_shape(self, never_flatten=False) -> tuple[int, ...]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        # number layers
        num_enemies = 1 if self.cfg.ec.compress_enemies else (self.num_players - 1)
        offset = num_layers_general(self.cfg.ec) + layers_per_player(self.cfg.ec)
        z_dim = offset + num_enemies * layers_per_enemy(self.cfg.ec)
        # width and height
        width = self.cfg.w
        height = self.cfg.h
        if self.cfg.ec.centered:
            width = 2 * self.cfg.w - 1
            height = 2 * self.cfg.h - 1
        elif not self.cfg.wrapped:  # wrapped does not have a border
            # +1 on every side for border of field
            width = self.cfg.w + 2
            height = self.cfg.h + 2
        # number of dim
        if self.cfg.ec.flatten and not never_flatten:
            dim = width * height * z_dim
            return tuple([dim, ])
        return width, height, z_dim

    def _step(
            self,
            actions: tuple[int, ...],
    ) -> tuple[np.ndarray, bool, dict]:
        # test if actions are actually legal to perform
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        # fill actions of players not at turn with zeros
        action_arr = np.zeros(shape=(self.num_players,), dtype=ct.c_int)
        for idx, player in enumerate(self.players_at_turn()):
            action_arr[player] = actions[idx]
        # shift player alive and at turn
        self.players_at_turn_last = self.players_at_turn()
        self.players_alive_last = self.players_alive()
        # perform step
        action_p = action_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        CPP_LIB.lib.step_cpp(self.state_p, action_p)
        # reset saved properties
        self.obs_dict = dict()
        self.available_actions_save = dict()
        self.players_at_turn_save = None
        self.players_alive_save = None
        # compute return values
        done = self.is_terminal()
        rewards = self.reward_func(done, self.num_players, self.players_at_turn(), self.players_at_turn_last)
        return rewards, done, {}

    def _get_copy(self) -> "BattleSnakeGame":
        # clone the c++ env and initialize it on python side
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        state_p2 = CPP_LIB.lib.clone_cpp(self.state_p)
        cpy = BattleSnakeGame(
            cfg=self.cfg,
            state_p=state_p2,
        )
        # copy properties
        cpy.available_actions_save = self.available_actions_save.copy()
        cpy.obs_dict = self.obs_dict.copy()
        if self.players_at_turn_save is not None:
            cpy.players_at_turn_save = self.players_at_turn_save.copy()
        if self.players_at_turn_last is not None:
            cpy.players_at_turn_last = self.players_at_turn_last.copy()
        if self.players_alive_save is not None:
            cpy.players_alive_save = self.players_alive_save.copy()
        if self.players_alive_last is not None:
            cpy.players_alive_last = self.players_alive_last.copy()
        return cpy

    def close(self):
        CPP_LIB.lib.close_cpp(self.state_p)
        self.is_closed = True

    def _reset(self):
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        CPP_LIB.lib.close_cpp(self.state_p)
        self._init_cpp()
        self.reset_saved_properties()

    def available_actions(self, player: int) -> list[int]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        if player < 0 or player >= self.cfg.num_players:
            raise ValueError(f'Snake index out of range: {player}')
        if not self.is_player_alive(player):
            return []  # what is dead cannot move
        if self.cfg.all_actions_legal:
            return [0, 1, 2, 3]
        if player in self.available_actions_save:  # use saved actions from last function call
            return self.available_actions_save[player]
        # ask c++ lib what actions are legal
        legal_actions = np.zeros(shape=(self.cfg.num_actions,), dtype=ct.c_int)
        CPP_LIB.lib.actions_cpp(self.state_p, player, legal_actions)
        action_list = []
        for a in range(self.cfg.num_actions):
            if legal_actions[a]:
                action_list.append(a)
        self.available_actions_save[player] = action_list
        return action_list

    def players_at_turn(self) -> list[int]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        if self.players_at_turn_save is None:
            # only snakes with available actions are at turn
            self.players_at_turn_save = []
            for player in range(self.num_players):
                if self.is_player_alive(player) and self.available_actions(player):
                    self.players_at_turn_save.append(player)
        return self.players_at_turn_save

    def players_alive(self) -> list[int]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        # call c++
        if self.players_alive_save is None:
            res_arr = np.zeros(shape=(self.num_players,), dtype=bool)
            res_p = res_arr.ctypes.data_as(ct.POINTER(ct.c_bool))
            CPP_LIB.lib.alive_cpp(self.state_p, res_p)
            self.players_alive_save = []
            for player in range(self.num_players):
                if res_arr[player]:
                    self.players_alive_save.append(player)
        return self.players_alive_save

    def player_lengths(self) -> list[int]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        res_arr = np.zeros(shape=(self.num_players,), dtype=ct.c_int)
        res_p = res_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        CPP_LIB.lib.snake_length_cpp(self.state_p, res_p)
        return list(res_arr)

    def player_healths(self) -> list[int]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        res_arr = np.zeros(shape=(self.num_players,), dtype=ct.c_int)
        res_p = res_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        CPP_LIB.lib.snake_health_cpp(self.state_p, res_p)
        return list(res_arr)

    def player_pos(self, player: int) -> list[tuple[int, int]]:  # returns list of length BODY_LEN != SNAKE_LEN
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        # this is inefficient and should only be used for debugging
        body_len = CPP_LIB.lib.snake_body_length_cpp(self.state_p, player)
        res_arr = np.zeros(shape=(body_len, 2), dtype=ct.c_int)
        CPP_LIB.lib.snake_pos_cpp(self.state_p, player, res_arr.ctypes.data_as(ct.POINTER(ct.c_int)))
        return list(map(tuple, res_arr))

    def all_player_pos(self) -> dict[int, list[tuple[int, int]]]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        # this is inefficient and should only be used for debugging
        res_dict = {}
        for player in range(self.num_players):
            if self.is_player_alive(player):
                res_dict[player] = self.player_pos(player)
            else:
                res_dict[player] = []
        return res_dict

    def num_food(self) -> int:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        return CPP_LIB.lib.num_food_cpp(self.state_p)

    def get_hazards(self) -> np.ndarray:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        arr = np.zeros(shape=(self.cfg.w, self.cfg.h), dtype=bool)
        arr_p = arr.ctypes.data_as(ct.POINTER(ct.c_bool))
        CPP_LIB.lib.hazards_cpp(self.state_p, arr_p)
        return arr.T

    def food_pos(self) -> np.ndarray:  # returns array of shape (num_food, 2)
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        n = self.num_food()
        res_arr = np.zeros(shape=(n, 2), dtype=ct.c_int)
        CPP_LIB.lib.food_pos_cpp(self.state_p, res_arr.ctypes.data_as(ct.POINTER(ct.c_int)))
        return res_arr

    def is_terminal(self) -> bool:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        # a game has ended if no / only the last player alive is at turn
        if len(self.available_joint_actions()) == 0:
            return True
        if self.num_players == 1:
            return self.num_players_at_turn() == 0
        else:
            return self.num_players_at_turn() <= 1

    def area_control(
            self,
            weight: float = 1.0,  # weight of normal tile
            food_weight: float = 1.0,
            hazard_weight: float = 1.0,
            food_in_hazard_weight: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """
        Args:
            weight (): weight of normal tile
            food_weight (): weight of food tile
            hazard_weight (): weight of hazard tile
            food_in_hazard_weight (): weight of hazard tile that contains food

        Returns:
            Dictionary of:
            - area control for each player
            - food distance for each player  (w+h if not reachable)
            - tail distance for each player  (w+h if not reachable)
            - bool array indicating if tail is reachable for each player
            - bool array indicating if food is reachable for each player
        """
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        ac, fd, td, tr, fr = CPP_LIB.get_area_control(
            self.num_players,
            self.state_p,
            weight,
            food_weight,
            hazard_weight,
            food_in_hazard_weight
        )
        res_dict = {
            "area_control": ac,
            "food_distance": fd,
            "tail_distance": td,
            "tail_reachable": tr,
            "food_reachable": fr,
        }
        return res_dict

    def render(self):
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        str_repr = self.get_str_repr()
        print(str_repr)

    def get_str_repr(self) -> str:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        arr = ct.create_string_buffer(self.cfg.w * self.cfg.h * 3)
        CPP_LIB.lib.str_cpp(self.state_p, arr)
        str_repr = arr.value.decode("utf-8")
        return str_repr

    def _get_cpp_encoding(
            self,
            player: int,
            temperatures: Optional[list[float]],
            single_temperature: Optional[bool],
    ):
        obs_arr = np.zeros(shape=self.get_obs_shape(never_flatten=True), dtype=np.float32)
        obs_p = obs_arr.ctypes.data_as(ct.POINTER(ct.c_float))
        t_arr = np.asarray(temperatures, dtype=ct.c_float)
        t_p = t_arr.ctypes.data_as(ct.POINTER(ct.c_float))
        CPP_LIB.lib.custom_encode_cpp(
            self.state_p,
            obs_p,
            self.cfg.ec.include_current_food,
            self.cfg.ec.include_next_food,
            self.cfg.ec.include_board,
            self.cfg.ec.include_number_of_turns,
            self.cfg.ec.compress_enemies,
            player,
            self.cfg.ec.include_snake_body_as_one_hot,
            self.cfg.ec.include_snake_body,
            self.cfg.ec.include_snake_head,
            self.cfg.ec.include_snake_tail,
            self.cfg.ec.include_snake_health,
            self.cfg.ec.include_snake_length,
            self.cfg.ec.centered,
            self.cfg.ec.include_distance_map,
            self.cfg.ec.include_area_control,
            self.cfg.ec.include_food_distance,
            self.cfg.ec.include_hazards,
            self.cfg.ec.include_tail_distance,
            self.cfg.ec.include_num_food_on_board,
            self.cfg.ec.fixed_food_spawn_chance,
            self.cfg.ec.temperature_input,
            self.cfg.ec.single_temperature_input if single_temperature is None else single_temperature,
            t_p,
        )
        return obs_arr

    def _get_custom_state_encoding(
            self,
            player: int,
            perm: Optional[np.ndarray],
            temperatures: Optional[list[float]],
            single_temperature: Optional[bool],
    ) -> np.ndarray:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        player_list = self.players_at_turn()
        # if the player is dead we give him an arbitrary encoding (just to match shapes, it is discarded later)
        if player not in player_list:
            raise Exception("Cannot get an encoding for a dead snake (duh)")
        # check if we already computed an encoding for this player
        if player not in self.obs_dict or temperatures is not None:
            obs_arr = self._get_cpp_encoding(
                player=player,
                temperatures=temperatures,
                single_temperature=single_temperature,
            )
            self.obs_dict[player] = obs_arr.copy()
        else:
            obs_arr = np.copy(self.obs_dict[player])
        # rotate encodings of enemy players according to permutation
        if (not self.cfg.ec.compress_enemies) and self.cfg.num_players > 2:
            if perm is None:
                perm = np.random.permutation(self.cfg.num_players - 1)
            left, right = [], []
            offset = num_layers_general(self.cfg.ec) + layers_per_player(self.cfg.ec)
            num_enemy_layer = layers_per_enemy(self.cfg.ec)
            for sub_layer in range(num_enemy_layer):
                for enemy in range(0, self.cfg.num_players - 1):
                    left.append(offset + enemy * num_enemy_layer + sub_layer)
                for idx in perm:
                    right.append(offset + idx * num_enemy_layer + sub_layer)
            obs_arr[:, :, left] = obs_arr[:, :, right]
        return obs_arr

    def __eq__(self, other: "BattleSnakeGame"):
        if not isinstance(other, BattleSnakeGame):
            return False
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        equal = CPP_LIB.lib.equals_cpp(self.state_p, other.state_p)
        return equal

    def get_symmetry_count(self):
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        if self.cfg.ec.compress_enemies:
            return 8
        else:
            return 8 * math.factorial(self.cfg.num_players - 1)

    def get_obs(
            self,
            symmetry: Optional[int] = 0,
            temperatures: Optional[list[float]] = None,
            single_temperature: Optional[bool] = None,
    ) -> tuple[
        np.ndarray,
        dict[int, int],
        dict[int, int],
    ]:
        if self.is_closed:
            raise ValueError("Cannot call function on closed game")
        if self.is_terminal():
            raise ValueError("Cannot get encoding on terminal state")
        if not self.cfg.ec.temperature_input and temperatures is not None:
            raise ValueError(f"Cannot process temperatures if ec specifies no input")
        if self.cfg.ec.temperature_input:
            single_temp = self.cfg.ec.single_temperature_input if single_temperature is None else single_temperature
            if temperatures is None:
                raise ValueError(f"Need temperatures to generate encoding")
            if single_temp and len(temperatures) != 1:
                raise ValueError(f"Cannot process multiple temperatures if single temperature input specified")
            if not single_temp and len(temperatures) != self.num_players:
                raise ValueError(f"Invalid temperature length: {temperatures}")
        if symmetry is None:
            symmetry = np.random.randint(self.get_symmetry_count())
        # last 3 bits of symmetry represent rotation and flip
        sym_rot = symmetry % 8
        flip = (sym_rot % 2 == 1)  # if symmetry is odd then mirror it
        num_rot = math.floor(sym_rot / 2)
        # symmetry except last 3 bit describes player permutation
        sym_player = math.floor(symmetry / 8)
        perm = int_to_perm(sym_player, self.num_players - 1)
        # get encoding and stack them
        obs_list = []
        for player in self.players_at_turn():
            obs = self._get_custom_state_encoding(
                player=player,
                perm=perm,
                temperatures=temperatures,
                single_temperature=single_temperature,
            )
            obs_list.append(obs)
        obs = np.stack(obs_list)
        # apply rotation and flip
        obs_res = np.rot90(obs, k=num_rot, axes=(-3, -2))
        if flip:
            obs_res = np.flip(obs_res, axis=-2)
        # calculate action mapping by using offset: counterclockwise rotation of 90 is -1
        # flip is offset of 2 (left -> right, up -> down,...)
        # original: UP=0, RIGHT=1, DOWN=2, LEFT=3
        action_offset: int = -num_rot  # + 2*flip
        perm, inv_perm = dict(), dict()
        for a in range(self.cfg.num_actions):
            a_new = (a + action_offset) % self.cfg.num_actions
            if flip:
                if a_new == 2:
                    a_new = 0
                elif a_new == 0:
                    a_new = 2
            perm[a] = a_new
            inv_perm[a_new] = a
        # sanity check
        if obs_res.shape[0] != self.num_players_at_turn():
            raise Exception("Unknown Exception with observation shape")
        if self.cfg.ec.flatten:
            obs_res = obs_res.reshape(self.num_players_at_turn(), -1)
        result = obs_res.copy()  # necessary because of negative stride
        # result = torch.tensor(obs_res.copy(), dtype=torch.float32)
        return result, perm, inv_perm

    def __del__(self):
        if not self.is_closed:
            self.close()
            self.is_closed = True

    def get_bool_board_matrix(self) -> np.ndarray:
        if not self.cfg.constrictor:
            raise ValueError(f"Board matrix currently only supported in constrictor")
        arr = np.zeros((self.cfg.w, self.cfg.h), dtype=ct.c_int8)
        arr_p = arr.ctypes.data_as(ct.POINTER(ct.c_int8))
        CPP_LIB.lib.char_game_matrix_cpp(self.state_p, arr_p)
        return arr

    def get_state(self) -> BattleSnakeState:
        snakes_alive = self.players_alive()
        snakes_alive_bool = [i in snakes_alive for i in range(self.num_players)]
        player_pos = {i: self.player_pos(i) for i in range(self.num_players)}
        food_pos_arr = self.food_pos()
        food_list = [[food_pos_arr[i, 0], food_pos_arr[i, 1]] for i in range(food_pos_arr.shape[0])]
        snake_health = self.player_healths()
        snake_len = self.player_lengths()
        state = BattleSnakeState(
            snakes_alive=snakes_alive_bool,
            snake_pos=player_pos,
            food_pos=food_list,
            snake_health=snake_health,
            snake_len=snake_len,
        )
        return state

    def _set_state(self, state: BattleSnakeState):
        # close old state pointer
        CPP_LIB.lib.close_cpp(self.state_p)
        # find the longest body, this determines the array shape
        snake_pos = {}  # we need a separate dict to not alter the config object
        body_lengths = []
        for s in range(self.cfg.num_players):
            cur_snake_pos = []
            for pos in state.snake_pos[s]:
                cur_snake_pos.append((pos[0], pos[1]))
            cur_snake_pos = list(dict.fromkeys(cur_snake_pos))  # remove duplicates
            snake_pos[s] = cur_snake_pos
            body_lengths.append(len(cur_snake_pos))
        max_body_len = max(body_lengths)
        body_len_arr = np.asarray(body_lengths, dtype=ct.c_int)
        body_len_p = body_len_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        snake_pos_arr = np.zeros(shape=(self.cfg.num_players, max_body_len, 2), dtype=ct.c_int) - 1
        for s in range(self.cfg.num_players):  # convert dictionary to numpy array
            for i, pos in enumerate(snake_pos[s]):
                snake_pos_arr[s, i, 0] = pos[0]
                snake_pos_arr[s, i, 1] = pos[1]
        snake_pos_p = snake_pos_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # snake length
        snake_len_arr = np.asarray(state.snake_len, dtype=ct.c_int)
        snake_len_p = snake_len_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # snakes alive
        snake_alive_arr = np.asarray(state.snakes_alive, dtype=bool)
        snake_alive_p = snake_alive_arr.ctypes.data_as(ct.POINTER(ct.c_bool))
        # health
        snake_health_arr = np.asarray(state.snake_health, dtype=ct.c_int)
        snake_health_p = snake_health_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        snake_max_health_arr = np.asarray(self.cfg.max_snake_health, dtype=ct.c_int)
        snake_max_health_p = snake_max_health_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # food
        num_init_food = len(state.food_pos)
        np_arr = np.asarray(state.food_pos, dtype=ct.c_int)
        food_pos_p = np_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # hazards, we need to transpose because cpp uses flattened array (this is more efficient)
        hazard_arr = np.zeros(shape=(self.cfg.h, self.cfg.w), dtype=bool)
        for hazard_tile in self.cfg.init_hazards:
            hazard_arr[hazard_tile[1], hazard_tile[0]] = True
        hazards_p = hazard_arr.ctypes.data_as(ct.POINTER(ct.c_bool))
        self.state_p = CPP_LIB.lib.init_cpp(
            self.cfg.w,
            self.cfg.h,
            self.cfg.num_players,
            self.cfg.min_food,
            self.cfg.food_spawn_chance,
            self.cfg.init_turns_played,
            False,
            body_len_p,
            max_body_len,
            snake_pos_p,
            num_init_food,
            food_pos_p,
            snake_alive_p,
            snake_health_p,
            snake_len_p,
            snake_max_health_p,
            self.cfg.wrapped,
            self.cfg.royale,
            self.cfg.shrink_n_turns,
            self.cfg.hazard_damage,
            hazards_p,
        )
        self.reset_saved_properties()
