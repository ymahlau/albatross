

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from imp_env.imp_env.owf_env import Struct_owf
from imp_env.imp_env.struct_env import Struct

from src.game.game import Game, GameConfig


class IMP_MODE(Enum):
    K_OF_N = 'K_OF_N'
    K_OF_N_CORRELATED = 'K_OF_N_CORRELATED'
    OWF = 'OWF'

@dataclass(kw_only=True)
class IMPConfig(GameConfig):
    num_actions: int = field(default=3)
    imp_mode: IMP_MODE
    campaign_cost: bool
    temperature_input: bool = False
    single_temperature_input: bool = True
    
    def __post_init__(self):
        if self.imp_mode == IMP_MODE.OWF:
            if self.num_players not in [2, 4, 10, 50, 100]:
                raise Exception(f"Invalid number of agents: {self.num_players}")
        else:
            if self.num_players not in [3, 5, 10, 50, 100]:
                raise Exception(f"Invalid number of agents: {self.num_players}")

class IMPGame(Game):
    def __init__(self, cfg: IMPConfig, env=None, obs_shape_save: Optional[tuple[int, ...]] = None):
        super().__init__(cfg)
        self.cfg = cfg
        
        if env is not None:
            self.env = env
        elif self.cfg.imp_mode == IMP_MODE.OWF:
            imp_cfg =  {
                "n_owt": int(self.cfg.num_players / 2),
                "lev": 3,
                "discount_reward": 1,
                "campaign_cost": self.cfg.campaign_cost
            }
            self.env = Struct_owf(config=imp_cfg)
        else:
            corresponding_k = {3:1, 5:2, 10:5, 50:25, 100:50}
            imp_cfg = {
                "n_comp": self.cfg.num_players,
                "discount_reward": 1,
                "k_comp": corresponding_k[self.cfg.num_players],
                "env_correlation": True if self.cfg.imp_mode == IMP_MODE.K_OF_N_CORRELATED else False,
                "campaign_cost": self.cfg.campaign_cost,
            }
            self.env = Struct(config=imp_cfg)
        self.done = False
        # get obs shape
        if obs_shape_save is None:
            if self.env.observations is None:
                raise Exception("This should never happen")
            self.obs_shape_save = self.env.observations['agent_0'].shape
        else:
            self.obs_shape_save = obs_shape_save
        
    def _step(self, actions: tuple[int, ...]) -> tuple[np.ndarray, bool, dict]:
        action_dict = {
            f"agent_{p}": actions[p] for p in self.players_at_turn()
        }
        _, reward_dict, done, _ = self.env.step(action_dict)
        reward_arr = np.asarray([reward_dict[ f"agent_{p}"] for p in self.players_at_turn()])
        self.done = done
        return reward_arr, done, {}
    
    def _reset(self):
        self.env.reset()
        self.done = False
    
    def close(self):
        pass
    
    def render(self):
        print(self.env)
    
    def _get_copy(self) -> "IMPGame":
        env2 = copy.deepcopy(self.env)
        cpy = IMPGame(self.cfg, env=env2, obs_shape_save=self.obs_shape_save)
        cpy.done = self.done
        cpy.obs_shape_save = self.obs_shape_save
        return cpy
    
    def __eq__(self, game: "Game") -> bool:
        own_obs = self.get_obs()[0]
        other_obs = game.get_obs()[0]
        if len(own_obs.shape) != len(other_obs.shape):
            return False
        if any([own_obs.shape[i] != other_obs.shape[i] for i in range(len(own_obs.shape))]):
            return False
        return np.all(own_obs == other_obs).item()

    def players_at_turn(self) -> list[int]:  # List of players, which can make a turn
        return [] if self.done else list(range(self.num_players))
    
    def players_alive(self) -> list[int]:
        return [] if self.done else list(range(self.num_players))
    
    def is_terminal(self) -> bool:
        return self.done
    
    def available_actions(self, player: int) -> list[int]:
        return [] if self.done else list(range(self.num_actions))

    def get_symmetry_count(self):
        return 1
    
    def get_obs_shape(self, never_flatten=False) -> tuple[int, ...]:
        raw_obs_shape = self.obs_shape_save
        if self.cfg.temperature_input:
            raw_obs_list = list(raw_obs_shape)
            if self.cfg.single_temperature_input:
                raw_obs_list[-1] += 1
            else:
                raw_obs_list[-1] += self.cfg.num_players - 1
            raw_obs_shape = tuple(raw_obs_list)
        return raw_obs_shape
    
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
        obs_dict = self.env.observations
        if obs_dict is None:
            raise Exception("This should never happen, probably game is terminal")
        obs_arr = np.asarray([obs_dict[f"agent_{p}"] for p in self.players_at_turn()])
        if temperatures is not None and not self.cfg.temperature_input:
                raise ValueError("Cannot process temperatures if cfg does not specify temp input")
        if self.cfg.temperature_input:
            temp_list = []
            for player in range(self.cfg.num_players):
                single_temp = self.cfg.single_temperature_input if single_temperature is None else single_temperature
                if temperatures is None:
                    raise ValueError(f"Need temperatures to generate encoding")
                if single_temp and len(temperatures) != 1:
                    raise ValueError(f"Cannot process multiple temperatures if single temperature input specified")
                if not single_temp and len(temperatures) != self.num_players:
                    raise ValueError(f"Invalid temperature length: {temperatures}")
                if single_temp:
                    temp_arr = np.asarray([temperatures[0]], dtype=float)
                else:
                    temp_arr = np.asarray([t for i, t in enumerate(temperatures) if i != player], dtype=float)
                temp_list.append(temp_arr)
            full_temp_arr = np.stack(temp_list, axis=0) / 10  # scale temperatures to reasonable range
            obs_arr = np.concatenate((obs_arr, full_temp_arr), axis=-1)
        id_dict = {a: a for a in range(self.num_actions)}
        return obs_arr, id_dict, id_dict
        
    
    def get_str_repr(self) -> str:
        return str(self.env)
    
    
