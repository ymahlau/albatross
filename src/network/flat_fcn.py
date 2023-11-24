from dataclasses import dataclass, field
from functools import cached_property

import torch

from src.network.fcn import FCN, HeadConfig, SmallHeadConfig, WideHeadConfig
from src.network.flat_net import FlatNet, FlatNetworkConfig
from src.network.utils import NormalizationType, ActivationType


@dataclass
class FlatFCNetworkConfig(FlatNetworkConfig):
    num_layer: int = 1
    hidden_size: int = 256
    dropout_p: float = 0.1
    norm_type: NormalizationType = NormalizationType.NONE

class FlatFCN(FlatNet):
    def __init__(
            self,
            cfg: FlatFCNetworkConfig,
    ):
        super().__init__(cfg)
        self.cfg = cfg
        # backbone model
        self.model = FCN(
            input_size=self.transformation_out_features,
            output_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layer=cfg.num_layer,
            activation_type=ActivationType.LEAKY_RELU,
            dropout_p=self.cfg.dropout_p,
            norm_type=self.cfg.norm_type,
        )
        self.initialize_weights()

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        model_out = self.model(x)
        return model_out

    @cached_property
    def latent_size(self) -> int:
        return self.cfg.hidden_size


@dataclass
class SmallFlatFCNConfig(FlatFCNetworkConfig):
    num_layer: int = field(default=1)
    hidden_size: int = field(default=128)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())


@dataclass
class WideHeadFCNConfig(FlatFCNetworkConfig):
    policy_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())
