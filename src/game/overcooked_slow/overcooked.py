from dataclasses import field, dataclass, MISSING
from typing import Optional

import numpy as np
import torch
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

from src.game.game import Game, GameConfig
from src.game.overcooked_slow.obs import get_general_features, get_obs_shape_and_padding, get_player_features, \
    get_pot_features, temperature_features
from src.game.overcooked_slow.state import SimplifiedOvercookedState, get_pot_state
from src.game.overcooked_slow.utils import OBJECT_NAMES


@dataclass
class OvercookedSlowConfig(GameConfig):
    num_actions: int = field(default=6)
    num_players: int = field(default=2)
    overcooked_layout: str = field(default="cramped_room")
    horizon: int = 100
    temperature_input: bool = False
    single_temperature_input: bool = True
    disallow_soup_drop: bool = True
    flat_obs: bool = False
    mep_reproduction_setting: bool = False
    mep_eval_setting: bool = False


class OvercookedGame(Game):
    """
    Overcooked game wrapper for: https://github.com/HumanCompatibleAI/overcooked_ai
    """
    def __init__(
            self,
            cfg: OvercookedSlowConfig,
            gridworld: Optional[OvercookedGridworld] = None,
            env: Optional[OvercookedEnv] = None
    ):
        super().__init__(cfg)
        self.cfg = cfg
        rew_shape_dict = None
        if self.cfg.mep_reproduction_setting:
            rew_shape_dict = {
                "PLACEMENT_IN_POT_REW": 3,
                "DISH_PICKUP_REWARD": 3,
                "SOUP_PICKUP_REWARD": 5,
                "DISH_DISP_DISTANCE_REW": 0.015,
                "POT_DISTANCE_REW": 0.03,
                "SOUP_DISTANCE_REW": 0.1,
            }
        if gridworld is None:
            self.gridworld: OvercookedGridworld = OvercookedGridworld.from_layout_name(
                cfg.overcooked_layout,
                rew_shaping_params=rew_shape_dict,
            )
        else:
            self.gridworld = gridworld
        self.env = env
        if env is None:
            self.env = OvercookedEnv.from_mdp(self.gridworld, horizon=self.cfg.horizon, info_level=1)
        # observations
        self.general_features: np.ndarray = get_general_features(self.gridworld)
        self.obs_save: Optional[torch.Tensor] = None  # save observation tensor to save compute
        self.id_dict = {a: a for a in range(self.cfg.num_actions)}
        if self.cfg.flat_obs:
            self.obs_shape_save = self.gridworld.get_featurize_state_shape()
        else:
            self.obs_shape_save, self.padding = get_obs_shape_and_padding(self.gridworld, self.cfg.temperature_input)
        self.padding_required = (self.gridworld.width != self.gridworld.height)

    def _step(self, actions: tuple[int, ...]) -> tuple[np.ndarray, bool, dict]:
        if len(actions) != 2:
            raise ValueError(f"Overcooked has two player with one action each")
        # save pot states before
        ps_before = self.get_pot_states()
        # step
        a0_converted = Action.ALL_ACTIONS[actions[0]]
        a1_converted = Action.ALL_ACTIONS[actions[1]]
        if self.env is None:
            raise Exception("self.env is None")
        _, single_reward, done, info = self.env.step((a0_converted, a1_converted))
        self.obs_save = None
        if self.cfg.mep_eval_setting:
            r = np.asarray([single_reward, single_reward], dtype=float)
            return r, done, {}
        r1, r2 =  info['shaped_r_by_agent'][0], info['shaped_r_by_agent'][1]
        if (not isinstance(r1, int) or not isinstance(r2, int)) and \
            (not isinstance(r1, float) or not isinstance(r2, float)):
            raise Exception("Wrong reward format in official overcooked implementation")
        full_reward = single_reward + r1 + r2
        # divide reward by horizon for estimate of maximum possible reward
        reward_arr = np.asarray([full_reward, full_reward], dtype=float)
        # additional reward not mentioned in paper for starting to cook
        if not self.cfg.mep_reproduction_setting:
            # divide reward for numerical stability
            reward_arr /= 10
            for psb, ps in zip(ps_before, self.get_pot_states()):
                if ps == 23:  # pot has started to cook
                    reward_arr += 0.5
        else:
            reward_arr /= 100
        return reward_arr, done, {}

    def _reset(self):
        self.env.reset(regen_mdp=False)
        self.obs_save = None

    def render(self):
        print(self.get_str_repr(), flush=True)

    def _get_copy(self) -> "OvercookedGame":
        env_cpy = self.env.copy()
        cpy = OvercookedGame(
            self.cfg,
            gridworld=self.gridworld,
            env=env_cpy,
        )
        cpy.obs_save = self.obs_save
        cpy.env.state = self.env.state.deepcopy()
        cpy.env._mlam = self.env.mlam
        cpy.env._mp = self.env.mp
        return cpy

    def __eq__(self, game: "OvercookedGame") -> bool:
        if not isinstance(game, OvercookedGame):
            return False
        return self.env.state == game.env.state

    def available_actions(self, player: int) -> list[int]:
        pos = self.get_player_positions()[player]
        orientation = self.env.state.player_orientations[player]
        faced_square = (pos[0] + orientation[0], pos[1] + orientation[1])
        # test if player has soup in hand and would drop it somewhere else than counter
        if self.cfg.disallow_soup_drop and not self.cfg.mep_reproduction_setting and self.get_player_held_item()[player] == 2:
            serv_loc = self.gridworld.get_serving_locations()
            if faced_square not in serv_loc:
                return list(range(5))
        # test if player has empty hands and is about to start cooking a dish
        if self.get_player_held_item()[player] == 3:
            pot_loc = self.gridworld.get_pot_locations()
            if faced_square in pot_loc:
                pot_state = self.gridworld.get_pot_states(self.env.state)
                legal_pots = self.gridworld.get_full_but_not_cooking_pots(pot_state)
                if faced_square not in legal_pots:
                    return list(range(5))
        return list(range(6))

    def players_at_turn(self) -> list[int]:
        return [] if self.is_terminal() else [0, 1]

    def players_alive(self) -> list[int]:
        return [] if self.is_terminal() else [0, 1]

    def is_terminal(self) -> bool:
        return self.env.is_done()

    def get_symmetry_count(self):
        return 1

    def get_obs_shape(self, never_flatten=False) -> tuple[int, ...]:
        return self.obs_shape_save

    def get_obs(
            self,
            symmetry: Optional[int] = 0,
            temperatures: Optional[list[float]] = None,
            single_temperature: Optional[bool] = None
    ) -> tuple[
        torch.Tensor,
        dict[int, int],
        dict[int, int],
    ]:
        # sanity check
        if self.is_terminal():
            raise ValueError("Cannot get encoding on terminal state")
        if not self.cfg.temperature_input and temperatures is not None:
            raise ValueError(f"Cannot process temperatures if ec specifies no input")
        single_temp = self.cfg.single_temperature_input if single_temperature is None else single_temperature
        if self.cfg.temperature_input:
            if temperatures is None:
                raise ValueError(f"Need temperatures to generate encoding")
            if single_temp and len(temperatures) != 1:
                raise ValueError(f"Cannot process multiple temperatures if single temperature input specified")
            if not single_temp and len(temperatures) != self.num_players:
                raise ValueError(f"Invalid temperature length: {temperatures}")
        # test if we have a saved observation tensor from previous call
        if self.obs_save is not None and not self.cfg.temperature_input:
            return self.obs_save, self.id_dict, self.id_dict
        # flat engineered observations
        if self.cfg.flat_obs:
            if self.cfg.temperature_input:
                raise ValueError(f"Cannot use flat obs with temperature input currently")
            obs = self.gridworld.featurize_state(self.env.state, self.env.mlam)
            obs_tensor = torch.tensor(np.asarray(obs), dtype=torch.float32)
            return obs_tensor, self.id_dict, self.id_dict
        # dynamic features
        player_obs = get_player_features(self.env.state, self.gridworld)
        pot_obs = get_pot_features(self.env.state, self.gridworld)
        time_layer = np.empty(shape=(2, pot_obs.shape[1], pot_obs.shape[2], 1), dtype=float)
        time_layer[...] = self.turns_played / self.cfg.horizon
        obs_list = [
            self.general_features,
            player_obs,
            np.flip(player_obs, axis=0),
            pot_obs,
            time_layer,
        ]
        if self.cfg.temperature_input:
            if temperatures is None:
                raise ValueError(f"Need temperatures to generate encoding")
            temp_features = temperature_features(self.gridworld, temperatures, single_temp)
            obs_list.append(temp_features)
        # merge arrays
        obs_arr = np.concatenate(obs_list, axis=-1)
        # apply padding if necessary
        if self.padding_required:
            obs_arr = np.pad(
                array=obs_arr,
                pad_width=((0, 0), (self.padding[0], self.padding[1]), (self.padding[2], self.padding[3]), (0, 0)),
            )
        obs = torch.tensor(obs_arr, dtype=torch.float32)
        self.obs_save = obs
        return obs, self.id_dict, self.id_dict

    def get_str_repr(self) -> str:
        return str(self.env)

    def close(self):
        pass  # nothing to close here

    def get_player_positions(self) -> tuple[tuple[int, int], tuple[int, int]]:
        # coordinate origin is top left square of board
        positions = self.env.state.player_positions
        return positions

    def get_player_orientations(self) -> tuple[int, int]:
        # 0: up, 1: down, 2: right, 3: left
        orientations_vector = self.env.state.player_orientations
        o1: int = Direction.ALL_DIRECTIONS.index(orientations_vector[0])
        o2: int = Direction.ALL_DIRECTIONS.index(orientations_vector[1])
        orientations = (o1, o2)
        return orientations

    def get_player_held_item(self) -> tuple[int, int]:
        # 0, 1, 2 -> index in OBJECT_NAMES. 3 encodes no item
        p0 = self.env.state.players[0]
        p0_item_id = 3 if not p0.has_object() else OBJECT_NAMES.index(p0.get_object().name)
        p1 = self.env.state.players[1]
        p1_item_id = 3 if not p1.has_object() else OBJECT_NAMES.index(p1.get_object().name)
        return p0_item_id, p1_item_id

    def get_pot_states(self) -> list[int]:  # length one or two, depending on number of pots
        pot_state_list = get_pot_state(self.env.state, self.gridworld)
        return pot_state_list

    def get_counter_states(self) -> list[int]:  # length number counters
        counter_locs = self.gridworld.get_counter_locations()
        state_list = [
            3 if not self.env.state.has_object(cl) else OBJECT_NAMES.index(self.env.state.get_object(cl).name)
            for cl in counter_locs
        ]
        return state_list

    def get_state(self) -> SimplifiedOvercookedState:
        simple_state = SimplifiedOvercookedState(self.env.state.to_dict())
        return simple_state

    def set_state(self, simple_state: SimplifiedOvercookedState):
        state = OvercookedState.from_dict(simple_state.state_dict)
        self.env.state = state
        self.obs_save = None
