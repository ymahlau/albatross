import copy
import multiprocessing as mp
from dataclasses import field, dataclass
from pathlib import Path
import pickle
from typing import Optional, Any

import numpy as np
import torch

from src.agent import AgentConfig, Agent
from src.game.actions import filter_illegal_and_normalize, apply_permutation
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.conversion import overcooked_slow_from_fast
from src.game.game import Game
from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig, OvercookedGameConfig
from src.game.overcooked_slow.overcooked import OvercookedGame as OvercookedGameSlow
from src.game.overcooked.overcooked import OvercookedGame as OvercookedGameFast
from src.network import NetworkConfig
from src.network.flat_fcn import FlatFCNetworkConfig
from src.network.initialization import get_network_from_config
from src.network.simple_fcn import SimpleNetwork, SimpleNetworkConfig


@dataclass
class RandomAgentConfig(AgentConfig):
    """
    Agent that chooses a random legal action
    """
    name: str = field(default="RandomAgent")

class RandomAgent(Agent):
    def __init__(self, cfg: RandomAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _act(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs = None,  # mp.Array
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        probs = np.ones(shape=(game.num_players_at_turn(), game.cfg.num_actions,), dtype=float)
        player_idx = game.players_at_turn().index(player)
        filtered = filter_illegal_and_normalize(probs, game)[player_idx]
        return filtered, {}


@dataclass
class LegalRandomAgentConfig(AgentConfig):
    """
    Agent that chooses a random legal action by making a copy of the original game with filter for legal actions.
    """
    name: str = field(default="LegalRandomAgent")

class LegalRandomAgent(Agent):
    def __init__(self, cfg: LegalRandomAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _act(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs = None,  # mp.Array
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(game, BattleSnakeGame):
            raise ValueError("Can only use Legal random agent with Battlesnake game")
        # construct copy with all_actions_legal = False
        new_cfg = copy.deepcopy(game.cfg)
        new_cfg.all_actions_legal = False
        # parse info
        food_arr = game.food_pos()
        new_cfg.init_food_pos = [[int(food_arr[i, 0]), int(food_arr[i, 1])] for i in range(food_arr.shape[0])]
        if not new_cfg.init_food_pos:  # correct special case when no free place to spawn food
            new_cfg.min_food = 0
        player_pos = {}
        for p in range(game.num_players):
            player_pos[p] = [[int(p_list[0]), int(p_list[1])] for p_list in game.player_pos(p)]
        new_cfg.init_snake_pos = player_pos
        new_cfg.init_turns_played = game.turns_played
        new_cfg.init_snakes_alive = [p in game.players_alive() for p in range(game.num_players)]
        new_cfg.init_snake_health = [int(health) for health in game.player_healths()]
        new_cfg.init_snake_len = [int(length) for length in game.player_lengths()]
        # construct game
        new_game = BattleSnakeGame(new_cfg)
        if player not in new_game.players_at_turn():
            return np.eye(game.num_actions)[0], {}
        player_idx = new_game.players_at_turn().index(player)
        probs = np.ones(shape=(new_game.num_players_at_turn(), game.cfg.num_actions), dtype=float)
        filtered = filter_illegal_and_normalize(probs, new_game)[player_idx]
        return filtered, {}


@dataclass
class NetworkAgentConfig(AgentConfig):
    net_cfg: Optional[NetworkConfig] = None
    random_symmetry: bool = False
    temperature_input: bool = False
    init_temperatures: Optional[list[float]] = None
    single_temperature: bool = True
    name: str = field(default="NetworkAgent")

class NetworkAgent(Agent):
    def __init__(self, cfg: NetworkAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = torch.device('cpu')
        if cfg.net_cfg is not None:
            self.net = get_network_from_config(cfg.net_cfg)
            self.net.to(self.device)
            self.net.eval()
        self.temperatures = self.cfg.init_temperatures

    @torch.no_grad()
    def _act(
            self,
            game: Game,
            player: int,
            time_limit: float,
            iterations: Optional[int] = None,
            save_probs = None,  # mp.Array
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # observation of game env
        symmetry = None if self.cfg.random_symmetry else 0
        temp_obs_input = None
        if self.cfg.temperature_input and self.cfg.single_temperature:
            if self.temperatures is None:
                raise Exception("Temperatures is None")
            temp_obs_input = [self.temperatures[0]]
        elif self.cfg.temperature_input and not self.cfg.single_temperature:
            temp_obs_input = self.temperatures
        obs, _, inv_perm = game.get_obs(
            symmetry=symmetry,
            temperatures=temp_obs_input,
            single_temperature=self.cfg.single_temperature,
        )
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        # forward pass
        out_tensor = self.net(obs).cpu().detach()
        log_actions = self.net.retrieve_policy_tensor(out_tensor).cpu()
        action_probs = torch.nn.functional.softmax(log_actions, dim=-1).numpy()
        perm_probs = apply_permutation(action_probs, inv_perm)
        # filter and sample
        player_idx = game.players_at_turn().index(player)
        filtered_probs = filter_illegal_and_normalize(perm_probs, game)[player_idx]
        return filtered_probs, {'values': self.net.retrieve_value_tensor(out_tensor).numpy()}

@dataclass
class BCNetworkAgentConfig(AgentConfig):
    net_cfg: Optional[NetworkConfig] = None
    name: str = field(default="BC-NetworkAgent")

class BCNetworkAgent(Agent):
    def __init__(self, cfg: BCNetworkAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = torch.device('cpu')
        if cfg.net_cfg is not None:
            self.net = get_network_from_config(cfg.net_cfg)
            self.net.to(self.device)
            self.net.eval()

    @torch.no_grad()
    def _act(
            self,
            game: Game,
            player: int,
            time_limit: float,
            iterations: Optional[int] = None,
            save_probs = None,  # mp.Array
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(game, OvercookedGameFast):
            raise ValueError(f"BC-Network agent only works on overcooked")
        if self.net is None:
            raise Exception(f"Need network to act")
        if not isinstance(self.net, SimpleNetwork):
            raise Exception("BC Agent need simple network")
        # get bc probs
        converted = overcooked_slow_from_fast(game, layout_abbr=self.net.cfg.layout_abbrev)
        if converted.env is None:
            raise Exception("game.env is None")
        flat_obs = converted.env.featurize_state_mdp(converted.env.state)
        flat_arr = torch.tensor(np.asarray(flat_obs), dtype=torch.float32)
        net_out = self.net(flat_arr).cpu().detach().numpy()
        net_probs = np.exp(net_out) / np.exp(net_out).sum(axis=-1)[..., np.newaxis]
        filtered_probs = filter_illegal_and_normalize(net_probs, game)[player]
        filtered_probs[4] = 0  # set stay action to zero, as done in baseline methods (PECAN, etc..)
        filtered_probs = filtered_probs / np.sum(filtered_probs)
        return filtered_probs, {}


def infer_game_cfg_from_name(name: str) -> OvercookedGameConfig:
    if 'aa' in name:
        return AsymmetricAdvantageOvercookedConfig()
    elif 'cr' in name:
        return CrampedRoomOvercookedConfig()
    elif 'co' in name:
        return CoordinationRingOvercookedConfig()
    elif 'fc' in name:
        return ForcedCoordinationOvercookedConfig()
    elif 'cc' in name:
        return CounterCircuitOvercookedConfig()
    raise Exception(f"Could not infer game config from {name=}")

def bc_agent_from_file(path: Path) -> BCNetworkAgent:
    standard_net_cfg = SimpleNetworkConfig(layout_abbrev=path.name[:2])
    name_conversion = {
        'fc_0/kernel:0': 'fcn.lin_in.weight',
        'fc_0/bias:0': 'fcn.lin_in.bias',
        'fc_1/kernel:0': 'fcn.hidden.0.weight',
        'fc_1/bias:0': 'fcn.hidden.0.bias',
        'logits/kernel:0': 'fcn.lin_out.weight',
        'logits/bias:0': 'fcn.lin_out.bias',
    }
    standard_net_cfg.game_cfg = infer_game_cfg_from_name(path.name)
    net = get_network_from_config(standard_net_cfg)
    with open(path, 'rb') as f:
        state_dict = pickle.load(f)
    new_state_dict = {
        name_conversion[k]: torch.tensor(v.T) for k, v in state_dict.items()
    }
    net.load_state_dict(new_state_dict)
    agent_cfg = BCNetworkAgentConfig(net_cfg=net.cfg)
    agent = BCNetworkAgent(agent_cfg)
    agent.replace_net(net)
    return agent

