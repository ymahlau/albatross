from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from os.path import exists
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.game.game import GameConfig
from src.game.initialization import get_game_from_config
from src.network.fcn import HeadConfig
from src.network.utils import cleanup_state_dict


@dataclass
class NetworkConfig:
    game_cfg: Optional[GameConfig] = None
    predict_policy: bool = True
    predict_game_len: bool = False
    film_temperature_input: bool = False
    single_film_temperature: bool = True  # only used if film_temperature_input is True
    film_cfg: Optional[HeadConfig] = None


class Network(nn.Module, ABC):
    def __init__(
            self,
            cfg: NetworkConfig,
    ):
        super(Network, self).__init__()
        if cfg.game_cfg is None:
            raise ValueError("Cannot initialize Network without game config (need observation shape)")
        self.cfg = cfg
        self.game = get_game_from_config(self.cfg.game_cfg)

    def forward(self, x: torch.Tensor, temperatures: Optional[torch.Tensor] = None) -> torch.Tensor:
        if temperatures is not None and not self.cfg.film_temperature_input:
            raise ValueError(f"Cannot process temperatures in network (film not specified)")
        if self.cfg.film_temperature_input:
            if temperatures is None:
                raise ValueError(f"Temperature in network input cannot be None")
            if self.cfg.single_film_temperature:
                if len(temperatures.shape) != 2 or temperatures.shape[1] != 1:
                    raise ValueError(f"Invalid temperature shape: {temperatures.shape}")
            else:
                if len(temperatures.shape) != 2 or temperatures.shape[1] != self.cfg.game_cfg.num_players - 1:
                    raise ValueError(f"Invalid temperature shape: {temperatures.shape}")
        # divide film input by 10 to scale it appropriately
        if temperatures is not None:
            temperatures /= 10
        return self._forward_impl(x, temperatures)

    @abstractmethod
    def _forward_impl(self, x: torch.Tensor, temperatures: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def reset(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def num_params(self) -> int:
        num_p = sum(p.numel() if p.requires_grad else 0 for p in self.parameters())
        return num_p

    @cached_property
    def output_size(self) -> int:
        output_size = 1
        if self.cfg.predict_policy:
            output_size += self.cfg.game_cfg.num_actions
        if self.cfg.predict_game_len:
            output_size += 1
        return output_size

    def save(self, path: Path) -> None:
        to_save = self.state_dict()
        clean_to_save = cleanup_state_dict(to_save)
        clean_to_save['cfg'] = self.cfg
        torch.save(clean_to_save, path)

    def load(self, path: Path) -> None:
        if not exists(path):
            raise ValueError("Model checkpoint does not exist")
        state_dict = torch.load(path)
        del state_dict['cfg']  # in the load-function we ignore the previous config and use new one provided
        self.load_state_dict(state_dict)
        print('Successfully Loaded Model from existing Checkpoint!', flush=True)

    @staticmethod
    def retrieve_value(output_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output_tensor (): Output of the forward function
        Returns: Tensor (of shape (n), n players. (b,n) if output is a batch)
            containing the value part of the output, scalar tensor
        """
        value = output_tensor[..., -1]
        # sanity checks
        if not torch.any(torch.isfinite(value)) or torch.any(torch.isnan(value)):
            raise Exception(f"Network value output contains invalid numbers: {value}")
        return value

    def retrieve_policy(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output_tensor (): Output of the forward function
        Returns: Tensor (of shape (n,a), n players, a actions per player. (b,n,a) if output is a batch) containing
            the actions part of the output
        """
        if not self.cfg.predict_policy:
            raise ValueError("This network does not predict a policy")
        if self.cfg.predict_game_len:
            actions = output_tensor[..., 0:-2]
        else:
            actions = output_tensor[..., 0:-1]
        if not torch.any(torch.isfinite(actions)) or torch.any(torch.isnan(actions)):
            raise Exception(f"Network action output contains invalid numbers: {actions}")
        return actions

    def retrieve_length(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output_tensor (): Output of the forward function
        Returns: Tensor (of shape (n,a), n players, a actions per player. (b,n,a) if output is a batch) containing
            the game length prediction part of the output
        """
        if not self.cfg.predict_game_len:
            raise ValueError("This network does not predict game length")
        lengths = output_tensor[..., -2]
        if not torch.any(torch.isfinite(lengths)) or torch.any(torch.isnan(lengths)):
            raise Exception(f"Network lengths output contains invalid numbers: {lengths}")
        return lengths

    def __del__(self):
        self.game.close()
