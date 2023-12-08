from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import torch

from src.game.game import Game
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.network import Network
from src.search.backup_func import get_backup_func_from_cfg, ExploitOtherBackupFunc
from src.search.config import SearchConfig
from src.search.core import cleanup
from src.search.eval_func import EvalFunc, NetworkEvalFunc, get_eval_func_from_cfg, EnemyExploitationEvalFunc
from src.search.extraction_func import get_extract_func_from_cfg
from src.search.node import Node


@dataclass
class SearchInfo:
    cleanup_time_ratio: float = 0
    eval_time_ratio: float = 0
    select_time_ratio: float = 0
    backup_time_ratio: float = 0
    extract_time_ratio: float = 0
    expansion_time_ratio: float = 0
    other_time_ratio: float = 0
    fully_explored: bool = False
    info: dict[str, Any] = field(default_factory=lambda: {})


class Search(ABC):
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self.eval_func = get_eval_func_from_cfg(cfg.eval_func_cfg)
        self.extract_func = get_extract_func_from_cfg(cfg.extract_func_cfg)
        self.root: Optional[Node] = None

    def __call__(
            self,
            game: Game,
            time_limit: Optional[float] = None,  # runtime limit in seconds
            iterations: Optional[int] = None,  # iteration limit
            save_probs = None,  # mp.Value, int value, to which current belief of best action is saved
            save_player_idx: Optional[int] = None,  # idx of player, whose action probs should be saved
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, np.ndarray, SearchInfo]:
        if game.is_terminal():
            game.render()
            raise ValueError("Cannot call search on terminal game")
        value, actions, info = self._compute(
            game=game,
            time_limit=time_limit,
            iterations=iterations,
            save_probs=save_probs,
            save_player_idx=save_player_idx,
            options=options,
        )
        # sanity checks
        for p_idx, p in enumerate(game.players_at_turn()):
            for a in game.illegal_actions(p):
                if actions[p_idx, a] > 0:
                    raise Exception("Search returned non-zero prob for illegal action")
        for player in game.players_not_at_turn():
            if value[player] != 0:
                game.render()
                if isinstance(game, BattleSnakeGame):
                    for p in range(game.num_players):
                        print(game.player_pos(p))
                raise Exception(f"Search gave dead player {player} a non-zero value: {value[player]}")
        return value, actions, info

    @abstractmethod
    def _compute(
            self,
            game: Game,
            time_limit: Optional[float] = None,  # runtime limit in seconds
            iterations: Optional[int] = None,  # iteration limit
            save_probs = None,  # mp.Value, int value, to which current belief of best action is saved
            save_player_idx: Optional[int] = None,  # idx of player, whose action probs should be saved
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, np.ndarray, SearchInfo]:
        """
        returns values and action probs for each player: arrays of shape
        - values: (num_players,)
        - action_probs: (num_players_at_turn, num_actions).
        - Additional info dictionary
        """
        raise NotImplementedError()

    def set_temperatures(self, temperatures: list[float]):
        if hasattr(self, "backup_func"):
            if hasattr(self.backup_func, "temperatures"): # type: ignore
                self.backup_func.temperatures = temperatures # type: ignore
            if hasattr(self.backup_func, "temperature"): # type: ignore
                self.backup_func.temperature = temperatures[0] # type: ignore
            if isinstance(self.backup_func, ExploitOtherBackupFunc): # type: ignore
                if hasattr(self.backup_func.backup_func, "temperatures"): # type: ignore
                    self.backup_func.backup_func.temperatures = temperatures # type: ignore
                if hasattr(self.backup_func.backup_func, "temperature"): # type: ignore
                    self.backup_func.backup_func.temperature = temperatures[0] # type: ignore
        if hasattr(self.eval_func, "temperatures"):
            self.eval_func.temperatures = temperatures # type: ignore
        if isinstance(self.eval_func, EnemyExploitationEvalFunc):
            self.eval_func.player_eval_func.temperatures = temperatures
            self.eval_func.temperatures = temperatures

    def replace_net(self, net: Network):
        if isinstance(self.eval_func, NetworkEvalFunc):
            self.eval_func.net = net
        elif isinstance(self.eval_func, EnemyExploitationEvalFunc):
            self.eval_func.player_eval_func.net = net

    def replace_device(self, device: torch.device):
        if isinstance(self.eval_func, NetworkEvalFunc):
            self.eval_func.device = device
            if self.eval_func.net is not None:
                self.eval_func.net = self.eval_func.net.to(device)
        elif isinstance(self.eval_func, EnemyExploitationEvalFunc):
            self.eval_func.player_eval_func.device = device
            if self.eval_func.player_eval_func.net is not None:
                self.eval_func.player_eval_func.net = self.eval_func.player_eval_func.net.to(device)
            self.eval_func.enemy_eval_func.device = device
            if self.eval_func.enemy_eval_func.net is not None:
                self.eval_func.enemy_eval_func.net = self.eval_func.enemy_eval_func.net.to(device)

    def _build_info(self, info: SearchInfo, full_time: float) -> SearchInfo:
        if full_time == 0:  # workaround for really fast search
            full_time = 1e-6
        info.other_time_ratio = full_time - info.cleanup_time_ratio - info.eval_time_ratio - info.backup_time_ratio \
            - info.extract_time_ratio - info.expansion_time_ratio - info.select_time_ratio
        info.cleanup_time_ratio /= full_time
        info.eval_time_ratio /= full_time
        info.select_time_ratio /= full_time
        info.backup_time_ratio /= full_time
        info.extract_time_ratio /= full_time
        info.expansion_time_ratio /= full_time
        info.other_time_ratio /= full_time
        if self.root is None:
            raise Exception("Root is None")
        info.fully_explored = self.root.is_fully_explored()
        return info

    def cleanup_root(self, exception_node: Optional[Node] = None):
        cleanup(self.root, exception_node)
        self.root = None
