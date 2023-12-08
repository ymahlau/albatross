from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

import numpy as np
import torch

from src.game.game import Game
from src.network import Network

@dataclass
class AgentConfig(ABC):
    name: str


class Agent(ABC):
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        # just for static type checking
        self.net = None
        self.device = None
        self.temperatures = None

    def __call__(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs = None,  # mp.Array
            options: Optional[dict[str, Any]] = None,
            single_temperature: Optional[bool] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:  # returns action probabilities and info
        if game.is_terminal():
            raise ValueError("Cannot call agent on terminal state")
        if not game.is_player_at_turn(player):
            raise ValueError("Calling agent which is not at turn")
        probs, info = self._act(
            game=game,
            player=player,
            time_limit=time_limit,
            iterations=iterations,
            save_probs=save_probs,
            options=options,
        )
        if save_probs is not None:
            save_probs[:] = probs
        return probs, info

    @abstractmethod
    def _act(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs = None,  # mp.Array
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError()

    def replace_net(self, net: Network):
        self.net = net
        if hasattr(self, 'search'):
            self.search.replace_net(net) # type: ignore

    def replace_device(self, device: torch.device):
        self.device = device
        if self.net is not None:
            self.net = self.net.to(device)
        if hasattr(self, 'search'):
            self.search.replace_device(device) # type: ignore

    def set_temperatures(self, temperatures: list[float]):
        self.temperatures = temperatures
        if hasattr(self, "search"):
            self.search.set_temperatures(temperatures) # type: ignore

    def reset_episode(self):
        """
        Subclasses can override this method. It is called whenever a new episode starts.
        """
        pass
