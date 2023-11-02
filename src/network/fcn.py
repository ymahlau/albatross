from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from src.network.utils import ActivationType, get_activation_func, NormalizationType, get_normalization_func


class FCN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layer: int,
            activation_type: ActivationType,
            dropout_p: float,
            norm_type: NormalizationType,
    ):
        super().__init__()
        self.num_layer = num_layer
        if num_layer < 1:
            raise ValueError(f"Invalid number of layers: {num_layer}")
        if num_layer == 1:
            self.linear = nn.Linear(input_size, output_size, bias=True)
        else:
            self.dropout_p = dropout_p
            self.use_dropout = self.dropout_p > 0
            if self.use_dropout:
                self.dropout = nn.Dropout(p=self.dropout_p, inplace=True)
            self.lin_bias = norm_type == NormalizationType.NONE or norm_type == NormalizationType.NONE.value
            self.norm = get_normalization_func(
                norm_type=norm_type,
                affine=True,
                num_features=hidden_size,
                normalized_shape=[hidden_size],
            )
            self.lin_in = nn.Linear(input_size, hidden_size, bias=self.lin_bias)
            self.hidden = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size, bias=self.lin_bias) for _ in range(num_layer-2)
            ])
            self.hidden_norms = nn.ModuleList([
                get_normalization_func(
                    norm_type=norm_type,
                    affine=True,
                    num_features=hidden_size,
                    normalized_shape=[hidden_size],
                )
                for _ in range(num_layer-2)
            ])
            self.lin_out = nn.Linear(hidden_size, output_size, bias=True)
            self.activ_func = get_activation_func(activation_type, inplace=False)

    def forward(self, x: torch.Tensor):
        if self.num_layer == 1:
            x = self.linear(x)
        else:
            x = self.lin_in(x)
            x = self.norm(x)
            x = self.activ_func(x)
            if self.use_dropout:
                x = self.dropout(x)
            for layer, hidden_norm in zip(self.hidden, self.hidden_norms):
                y = layer(x)
                y = hidden_norm(y)
                z = x + y  # residual connection
                z = self.activ_func(z)
                if self.use_dropout:
                    z = self.dropout(z)
                x = z
            x = self.lin_out(x)
        return x


@dataclass(kw_only=True)
class HeadConfig:
    num_layers: int
    hidden_size: Optional[int] = None
    activation_type: ActivationType = ActivationType.LEAKY_RELU
    normalization_type: NormalizationType = NormalizationType.GROUP_NORM
    dropout_p: float = 0.2
    final_activation: ActivationType = ActivationType.NONE

def head_from_cfg(
        cfg: HeadConfig,
        input_size: int,
        output_size: int,
) -> nn.Sequential:
    fcn = FCN(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        output_size=output_size,
        num_layer=cfg.num_layers,
        activation_type=cfg.activation_type,
        dropout_p=cfg.dropout_p,
        norm_type=cfg.normalization_type,
    )
    activation = get_activation_func(cfg.final_activation, inplace=True)
    seq = nn.Sequential(fcn, activation)
    return seq

@dataclass
class SmallHeadConfig(HeadConfig):
    num_layers: int = field(default=2)
    hidden_size: int = field(default=64)

@dataclass
class MediumHeadConfig(HeadConfig):
    num_layers: int = field(default=3)
    hidden_size: int = field(default=128)

@dataclass
class LargeHeadConfig(HeadConfig):
    num_layers: int = field(default=3)
    hidden_size: int = field(default=256)
