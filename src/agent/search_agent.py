import copy
import dataclasses
import multiprocessing as mp
from dataclasses import field, dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch

from src.agent import AgentConfig, Agent
from src.equilibria.logit import SbrMode
from src.game.game import Game
from src.game.actions import filter_illegal_and_normalize
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.initialization import get_game_from_config
from src.game.values import UtilityNorm
from src.network import NetworkConfig
from src.network.initialization import get_network_from_config, get_network_from_file
from src.search import SearchConfig
from src.search.backup_func import StandardBackupConfig
from src.search.config import FixedDepthConfig, NetworkEvalConfig, SpecialExtractConfig, NashBackupConfig, \
    CopyCatEvalConfig, LogitBackupConfig
from src.search.eval_func import AreaControlEvalConfig
from src.search.extraction_func import StandardExtractConfig
from src.search.initialization import get_search_from_config
from src.search.mcts import MCTSConfig
from src.search.sel_func import DecoupledUCTSelectionConfig


@dataclass(kw_only=True)
class SearchAgentConfig(AgentConfig):
    search_cfg: SearchConfig
    name: str = field(default="SearchAgent")
    deterministic: bool = False


class SearchAgent(Agent):
    """
    Search-based Agent that performs some form of search defined in the config. The best value is sampled from
    the resulting action-prob-distribution
    """

    def __init__(
            self,
            cfg: SearchAgentConfig,
    ) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.search = get_search_from_config(cfg.search_cfg)

    def _act(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs: Optional[mp.Array] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.cfg.deterministic:
            if not isinstance(game, BattleSnakeGame):
                raise NotImplementedError("Only battlesnake has conversion to deterministic implemented")
            cpy_cfg = copy.deepcopy(game.cfg)
            cpy_cfg.food_spawn_chance = 0
            state = game.get_state()
            game = get_game_from_config(cpy_cfg)
            game.set_state(state)
        player_idx = game.players_at_turn().index(player)
        values, action_probs, search_info = self.search(
            game=game,
            time_limit=time_limit,
            iterations=iterations,
            save_probs=save_probs,
            save_player_idx=player_idx,
            options=options,
        )
        info_dict = dataclasses.asdict(search_info)
        info_dict["values"] = values
        info_dict["all_action_probs"] = action_probs
        probs = action_probs[player_idx]
        return probs, info_dict


@dataclass(kw_only=True)
class AreaControlSearchAgentConfig(SearchAgentConfig):
    search_cfg: SearchConfig = field(default_factory=lambda: MCTSConfig(
        sel_func_cfg=DecoupledUCTSelectionConfig(exp_bonus=1.414),
        eval_func_cfg=AreaControlEvalConfig(utility_norm=UtilityNorm.NONE),
        backup_func_cfg=StandardBackupConfig(),
        extract_func_cfg=StandardExtractConfig(),
        expansion_depth=0,
        use_hot_start=True,
        optimize_fully_explored=False,
    ))
    name: str = field(default="AreaControlAgent")


@dataclass(kw_only=True)
class CopyCatSearchAgentConfig(SearchAgentConfig):
    search_cfg: SearchConfig = field(default_factory=lambda: MCTSConfig(
        sel_func_cfg=DecoupledUCTSelectionConfig(exp_bonus=2.0),
        eval_func_cfg=CopyCatEvalConfig(utility_norm=UtilityNorm.NONE),
        backup_func_cfg=StandardBackupConfig(),
        extract_func_cfg=StandardExtractConfig(),
        expansion_depth=0,
        use_hot_start=True,
        optimize_fully_explored=False,
    ))
    name: str = field(default="CopyCatAgent")


@dataclass(kw_only=True)
class SBRFixedDepthAgentConfig(SearchAgentConfig):
    search_cfg: SearchConfig = field(default_factory=lambda: FixedDepthConfig(
        eval_func_cfg=NetworkEvalConfig(net_cfg=None, utility_norm=UtilityNorm.NONE),
        backup_func_cfg=LogitBackupConfig(
            epsilon=0,
            num_iterations=200,
            sbr_mode=SbrMode.MSA,
        ),
        extract_func_cfg=SpecialExtractConfig(),
        average_eval=False,
    ))
    name: str = field(default="SBR-FixedDepthAgent")


@dataclass(kw_only=True)
class LookaheadAgentConfig(AgentConfig):
    search_cfg: SearchConfig
    net_cfg: Optional[NetworkConfig] = None
    temperature_input: bool = False
    init_temperatures: Optional[list[float]] = None
    single_temperature: bool = True
    additional_latency_sec: float = 0.01
    search_depth: int = 1
    search_weight: float = 0.5
    name: str = field(default="LookaheadAgent")


class LookaheadAgent(Agent):
    """
    Agent taking an average of network action prob output and n-step lookahead of fixed-depth search.
    This is useful to evaluate the network policy and values simultaneously.
    """

    def __init__(self, cfg: LookaheadAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = torch.device('cpu')
        # search
        self.search = get_search_from_config(self.cfg.search_cfg)
        if self.cfg.net_cfg is not None:
            self.net = get_network_from_config(self.cfg.net_cfg)
            self.search.replace_net(self.net)
        # sbr input temperatures
        self.temperatures = self.cfg.init_temperatures

    @torch.no_grad()
    def _act(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs: Optional[mp.Array] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # network action probs
        player_idx = game.players_at_turn().index(player)
        # obs input
        temp_obs_input = None
        if self.cfg.temperature_input and self.cfg.single_temperature:
            temp_obs_input = [self.temperatures[0]]
        elif self.cfg.temperature_input and not self.cfg.single_temperature:
            temp_obs_input = self.temperatures
        obs, _, _ = game.get_obs(symmetry=0, temperatures=temp_obs_input)
        # forward pass
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        net_out = self.net(obs)
        log_actions = self.net.retrieve_policy(net_out).cpu()
        net_probs = torch.nn.functional.softmax(log_actions, dim=-1).detach().numpy()
        filtered_net_probs = filter_illegal_and_normalize(net_probs, game)
        if save_probs is not None:
            save_probs[:] = filtered_net_probs[player_idx]
        filtered_probs_player = filtered_net_probs[player_idx]
        # search
        if time_limit is not None:
            time_limit -= self.cfg.additional_latency_sec
        # do not give save value and save idx as input because policy is better first estimate
        search_values, search_probs, info = self.search(
            game=game,
            iterations=self.cfg.search_depth,
            time_limit=time_limit,
            options=options,
        )
        # mixing and sample
        player_search_probs = search_probs[player_idx]
        policy_term = (1 - self.cfg.search_weight) * filtered_probs_player
        search_term = self.cfg.search_weight * player_search_probs
        mixed_probs = search_term + policy_term
        info_dict = dataclasses.asdict(info)
        return mixed_probs, info_dict


@dataclass(kw_only=True)
class TwoPlayerEvalAgentConfig(LookaheadAgentConfig):
    search_cfg: SearchConfig = field(default_factory=lambda: FixedDepthConfig(
        eval_func_cfg=NetworkEvalConfig(net_cfg=None, utility_norm=UtilityNorm.ZERO_SUM),
        backup_func_cfg=NashBackupConfig(),
        extract_func_cfg=SpecialExtractConfig(),
        average_eval=True,
        discount=1.0,
    ))
    search_depth: int = field(default=1)
    name: str = field(default="TwoPlayerEvalAgent")


@dataclass(kw_only=True)
class DoubleSearchAgentConfig(AgentConfig):
    search_cfg: SearchConfig
    search_cfg_2: SearchConfig
    net_path_2: Path
    device_str: str = 'cpu'
    compile_model_2: bool = False
    name: str = field(default="DoubleSearchAgent")
    deterministic: bool = False


class DoubleSearchAgent(Agent):
    def __init__(
            self,
            cfg: DoubleSearchAgentConfig,
    ) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.search = get_search_from_config(cfg.search_cfg)
        self.search2 = get_search_from_config(cfg.search_cfg_2)
        self.device = torch.device(self.cfg.device_str)
        net2 = get_network_from_file(self.cfg.net_path_2)
        temp_game = get_game_from_config(net2.cfg.game_cfg)
        net2 = net2.eval().to(self.device)
        if self.cfg.compile_model_2:
            net2 = torch.compile(
                model=net2,
                fullgraph=True,
                dynamic=False,
                mode='reduce-overhead'
            )
            temp_obs, _, _ = temp_game.get_obs()
            net2(torch.tensor(temp_obs, dtype=torch.float32).to(self.device))
        self.search2.replace_net(net2)

    def _act(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs: Optional[mp.Array] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.cfg.deterministic:
            if not isinstance(game, BattleSnakeGame):
                raise NotImplementedError("Only battlesnake has conversion to deterministic implemented")
            cpy_cfg = copy.deepcopy(game.cfg)
            cpy_cfg.food_spawn_chance = 0
            state = game.get_state()
            game = get_game_from_config(cpy_cfg)
            game.set_state(state)
        player_idx = game.players_at_turn().index(player)
        if game.num_players_at_turn() > 2:
            values, action_probs, search_info = self.search(
                game=game,
                time_limit=time_limit,
                iterations=iterations,
                save_probs=save_probs,
                save_player_idx=player_idx,
                options=options,
            )
        else:
            values, action_probs, search_info = self.search2(
                game=game,
                time_limit=time_limit,
                iterations=iterations,
                save_probs=save_probs,
                save_player_idx=player_idx,
                options=options,
            )
        info_dict = dataclasses.asdict(search_info)
        info_dict["values"] = values
        info_dict["all_action_probs"] = action_probs
        probs = action_probs[player_idx]
        return probs, info_dict

    def replace_device(self, device: torch.device):
        self.device = device
        if self.net is not None:
            self.net = self.net.to(device)
        self.search.replace_device(device)
        self.search2.replace_device(device)
