import ctypes as ct
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

file_path = Path(__file__)


class Struct(ct.Structure):
    pass


class CPPLibrary:
    def __init__(
            self,
    ):
        self.lib = ct.cdll.LoadLibrary(str(file_path.parent / 'compiled' / 'liblink.so'))
        self._init_functions()

    def _init_functions(self):
        self.lib.init_cpp.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_bool,
            ct.POINTER(ct.c_int),
            ct.c_int,
            ct.POINTER(ct.c_int),
            ct.c_int,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_bool),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.c_bool,
            ct.c_bool,
            ct.c_int,
            ct.c_int,
            ct.POINTER(ct.c_bool),
        ]
        self.lib.init_cpp.restype = ct.POINTER(Struct)
        self.lib.close_cpp.argtypes = [ct.POINTER(Struct)]
        self.lib.str_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.c_char_p,
        ]
        self.lib.clone_cpp.argtypes = [ct.POINTER(Struct)]
        self.lib.clone_cpp.restype = ct.POINTER(Struct)
        self.lib.actions_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.c_int,
            np.ctypeslib.ndpointer(dtype=ct.c_int, ndim=1, shape=(4,), flags='C_CONTIGUOUS'),
        ]
        self.lib.equals_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(Struct),
        ]
        self.lib.equals_cpp.restype = ct.c_bool
        self.lib.step_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(ct.c_int),
        ]
        self.lib.custom_encode_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(ct.c_float),
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_int,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_bool,
            ct.c_float,
            ct.c_bool,
            ct.c_bool,
            ct.POINTER(ct.c_float),
        ]
        self.lib.clone_cpp.argtypes = [ct.POINTER(Struct)]
        self.lib.clone_cpp.restype = ct.POINTER(Struct)
        self.lib.alive_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(ct.c_bool),
        ]
        self.lib.snake_length_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(ct.c_int),
        ]
        self.lib.snake_body_length_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.c_int,
        ]
        self.lib.snake_body_length_cpp.restype = ct.c_int
        self.lib.snake_pos_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.c_int,
            ct.POINTER(ct.c_int),
        ]
        self.lib.num_food_cpp.argtypes = [ct.POINTER(Struct)]
        self.lib.num_food_cpp.restype = ct.c_int
        self.lib.food_pos_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(ct.c_int),
        ]
        self.lib.turns_played_cpp.argtypes = [ct.POINTER(Struct)]
        self.lib.turns_played_cpp.restype = ct.c_int
        self.lib.snake_health_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(ct.c_int),
        ]
        self.lib.area_control_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_bool),
            ct.POINTER(ct.c_bool),
            ct.c_float,
            ct.c_float,
            ct.c_float,
            ct.c_float,
        ]
        self.lib.hazards_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(ct.c_bool),
        ]
        self.lib.logit_cpp.argtypes = [
            ct.c_int,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_double),
            ct.c_int,
            ct.c_double,
            ct.POINTER(ct.c_double),
            ct.c_bool,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]
        self.lib.logit_cpp.restype = ct.c_double
        self.lib.compute_nash_cpp.argtypes = [
            ct.c_int,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]
        self.lib.compute_nash_cpp.restype = ct.c_int
        self.lib.set_seed.argtypes = [
            ct.c_int,
        ]
        self.lib.mle_temperature_cpp.argtypes = [
            ct.c_double,
            ct.c_double,
            ct.c_int,
            ct.c_int,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_double),
            ct.c_bool,
        ]
        self.lib.mle_temperature_cpp.restype = ct.c_double
        self.lib.temperature_likelihood_cpp.argtypes = [
            ct.c_double,
            ct.c_int,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_double),
        ]
        self.lib.temperature_likelihood_cpp.restype = ct.c_double
        self.lib.rm_qr_cpp.argtypes = [
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_double),
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]
        self.lib.qse_cpp.argtypes = [
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_double),
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]
        self.lib.char_game_matrix_cpp.argtypes = [
            ct.POINTER(Struct),
            ct.POINTER(ct.c_int8),
        ]
        self.lib.init_overcooked_cpp.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
        ]
        self.lib.init_overcooked_cpp.restype = ct.POINTER(Struct)
        self.lib.clone_overcooked_cpp.argtypes = [ct.POINTER(Struct)]
        self.lib.clone_overcooked_cpp.restype = ct.POINTER(Struct)
        self.lib.close_overcooked_cpp.argtypes = [ct.POINTER(Struct)]
        self.lib.step_overcooked_cpp.argtypes = [ct.POINTER(Struct), ct.POINTER(ct.c_int)]
        self.lib.step_overcooked_cpp.restype = ct.c_double
        self.lib.char_overcooked_matrix_cpp.argtypes = [ct.POINTER(Struct), ct.c_char_p]
        self.lib.construct_overcooked_encoding_cpp.argtypes = [ct.POINTER(Struct), ct.POINTER(ct.c_float), ct.c_int, ct.c_bool, ct.c_float]
        self.lib.equals_overcooked_cpp.argtypes = [ct.POINTER(Struct), ct.POINTER(Struct)]
        self.lib.equals_overcooked_cpp.restype = ct.c_bool
        self.lib.get_player_infos_cpp.argtypes = [ct.POINTER(Struct), ct.POINTER(ct.c_int)]
        self.lib.get_state_oc_cpp.argtypes = [ct.POINTER(Struct), ct.POINTER(ct.c_int)]
        self.lib.update_tile_states_overcooked_cpp.argtypes = [ct.POINTER(Struct), ct.POINTER(ct.c_int)]

    def get_area_control(
            self,
            num_snakes: int,
            state_p,  # ct.POINTER(Struct)
            weight: float,
            food_weight: float,
            hazard_weight: float,
            food_in_hazard_weight: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # area array
        area_control_arr = np.zeros(shape=(num_snakes,), dtype=ct.c_float)
        area_control_p = area_control_arr.ctypes.data_as(ct.POINTER(ct.c_float))
        # food dist array
        food_dist_arr = np.zeros(shape=(num_snakes,), dtype=ct.c_int)
        food_dist_p = food_dist_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # tail dist array
        tail_dist_arr = np.zeros(shape=(num_snakes,), dtype=ct.c_int)
        tail_dist_p = tail_dist_arr.ctypes.data_as(ct.POINTER(ct.c_int))
        # reached tail array
        reached_tail_arr = np.zeros(shape=(num_snakes,), dtype=bool)
        reached_tail_p = reached_tail_arr.ctypes.data_as(ct.POINTER(ct.c_bool))
        # reached food array
        reached_food_arr = np.zeros(shape=(num_snakes,), dtype=bool)
        reached_food_p = reached_food_arr.ctypes.data_as(ct.POINTER(ct.c_bool))
        self.lib.area_control_cpp(
            state_p,
            area_control_p,
            food_dist_p,
            tail_dist_p,
            reached_tail_p,
            reached_food_p,
            weight,
            food_weight,
            hazard_weight,
            food_in_hazard_weight
        )
        return area_control_arr, food_dist_arr, tail_dist_arr, reached_tail_arr, reached_food_arr

    def compute_logit_equilibrium(
            self,
            available_actions: list[list[int]],  # maps player(index of player_at_turn) to available actions
            joint_action_list: list[tuple[int, ...]],
            joint_action_value_arr: np.ndarray,  # shape (num_joint_actions, num_player_at_turn)
            num_iterations: int,
            epsilon: float,
            temperatures: list[float],
            initial_policies: Optional[list[np.ndarray]] = None,
            hp_0: float = 0,
            hp_1: float = 0,
            sbr_mode: int = 0,
    ) -> tuple[list[float], list[np.ndarray], float]:
        # number of players and actions
        num_player = len(available_actions)
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
        # initialization
        initial_uniform = initial_policies is None
        initial_policies_p = ct.cast(0, ct.POINTER(ct.c_double))
        if not initial_uniform:
            flat_initial_policies = np.concatenate(initial_policies, axis=0).astype(ct.c_double)
            initial_policies_p = flat_initial_policies.ctypes.data_as(ct.POINTER(ct.c_double))
        # weighting
        temperatures_arr = np.asarray(temperatures, dtype=ct.c_double)
        temperatures_p = temperatures_arr.ctypes.data_as(ct.POINTER(ct.c_double))
        # result arrays
        result_values = np.zeros(shape=(num_player,), dtype=ct.c_double)
        result_values_p = result_values.ctypes.data_as(ct.POINTER(ct.c_double))
        result_policies = np.zeros_like(available_actions_arr, dtype=ct.c_double)
        result_policies_p = result_policies.ctypes.data_as(ct.POINTER(ct.c_double))
        pol_error = self.lib.logit_cpp(
            num_player,
            num_available_actions_p,
            available_actions_p,
            joint_actions_p,
            joint_action_value_p,
            num_iterations,
            epsilon,
            temperatures_p,
            initial_uniform,
            sbr_mode,
            hp_0,
            hp_1,
            initial_policies_p,
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
        return value_list, result_policy_list, pol_error

    def compute_nash(
            self,
            available_actions: list[list[int]],  # maps player(index of player_at_turn) to available actions
            joint_action_list: list[tuple[int, ...]],
            joint_action_value_arr: np.ndarray,  # shape (num_joint_actions, num_player_at_turn)
            error_counter = None,  # mp.Array
    ) -> tuple[list[float], list[np.ndarray]]:
        # number of players and actions
        num_player = len(available_actions)
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
        res_code: int = self.lib.compute_nash_cpp(
            num_player,
            num_available_actions_p,
            available_actions_p,
            joint_actions_p,
            joint_action_value_p,
            result_values_p,
            result_policies_p,
        )
        if res_code != 0:
            # raise Exception(f'Unknown c++ Nash error: {available_actions=}\n\n {joint_action_list=}\n\n'
            #                 f' {joint_action_value_arr=}')
            print('C++ Nash Error (probably due to numerical issues)', flush=True)
            if error_counter is not None:
                with error_counter.get_lock():
                    error_counter.value += 1
            avg_values = np.mean(joint_action_value_arr, axis=0)
            value_list = list(avg_values)
            result_policy_list = []
            for aa in available_actions:
                result_policy_list.append(np.ones(shape=(len(aa),), dtype=ct.c_float) / len(aa))
        else:
            value_list = list(result_values)
            result_policy_list = []
            start_idx = 0
            for p in range(num_player):
                end_idx = start_idx + num_available_actions[p]
                result_policy_list.append(result_policies[start_idx:end_idx])
                start_idx = end_idx
        return value_list, result_policy_list


CPP_LIB = CPPLibrary()
