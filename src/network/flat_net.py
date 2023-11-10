from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import torch
from torch import nn

from src.network import Network, NetworkConfig
from src.network.fcn import HeadConfig, head_from_cfg
from src.network.lff import LearnedFourierFeatures


@dataclass(kw_only=True)
class FlatNetworkConfig(NetworkConfig):
    value_head_cfg: HeadConfig
    policy_head_cfg: Optional[HeadConfig] = None
    length_head_cfg: Optional[HeadConfig] = None
    lff_features: bool = False
    lff_feature_expansion: int = 10


class FlatNet(Network, ABC):
    """
    Network for processing inputs that are flat (1d). Applications are games like Goofspiel or Oshi-Zumo
    """
    def __init__(
            self,
            cfg: FlatNetworkConfig,
    ):
        super().__init__(cfg)
        self.cfg = cfg
        # validate
        if len(self.game.get_obs_shape()) != 1:
            raise ValueError(f"Invalid input shape for fcn: {self.game.get_obs_shape()}")
        # lff
        in_channels = self.game.get_obs_shape()[0]
        self.transformation_out_features = in_channels
        if self.cfg.lff_features:
            self.transformation_out_features = in_channels * self.cfg.lff_feature_expansion
            self.lff = LearnedFourierFeatures(
                in_features=in_channels,
                out_features=self.transformation_out_features,
                conv_block=None,
                sin_cos=False,
                trainable=True,
            )
        self.value_head = head_from_cfg(
            cfg=self.cfg.value_head_cfg,
            input_size=self.latent_size,
            output_size=1
        )
        if self.cfg.predict_policy:
            self.policy_head = head_from_cfg(
                cfg=self.cfg.policy_head_cfg,
                input_size=self.latent_size,
                output_size=self.game.num_actions,
            )
        if self.cfg.predict_game_len:
            self.length_head = head_from_cfg(
                cfg=self.cfg.length_head_cfg,
                input_size=self.latent_size,
                output_size=1,
            )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                if m.weight is not None and m.bias is not None:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LearnedFourierFeatures):
                # iso initialization
                nn.init.normal_(m.layer.weight, 0, m.scale / m.in_features)
                nn.init.normal_(m.layer.bias, 0, 1)
                if m.sin_cos:
                    nn.init.zeros_(m.layer.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # lff if specified
        if self.cfg.lff_features:
            x = self.lff(x)
        # x has to be flattened array
        latent = self.backbone(x)
        # heads
        tensor_list = []
        if self.cfg.predict_policy:
            policy_out = self.policy_head(latent)
            tensor_list.append(policy_out)
        if self.cfg.predict_game_len:
            length_out = self.length_head(latent)
            tensor_list.append(length_out)
        value_out = self.value_head(latent)
        tensor_list.append(value_out)
        # post transform and cat
        out = torch.cat(tensor_list, dim=-1)
        return out

    @abstractmethod
    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    @cached_property
    def latent_size(self) -> int:
        raise NotImplementedError()
