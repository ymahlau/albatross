from typing import Optional, Union, Type

import numpy as np
import torch
from torch import nn

from src.network.invariant_conv import InvariantConvolution


class LearnedFourierFeatures(nn.Module):
    """
    Original Paper: https://www.episodeyang.com/ffn/ICLR_2022_FFN.pdf
    """
    def __init__(
        self,
            in_features,
            out_features,
            conv_block: Optional[Union[Type[torch.nn.Conv2d], Type[InvariantConvolution]]] = None,
            scale: float = 1.0,
            sin_cos: bool = False,
            trainable: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.sin_cos = sin_cos
        self.out_features = out_features
        self.conv_block = conv_block
        self.use_conv = self.conv_block is not None
        self.scale = scale
        layer_out_features = self.out_features // 2 if self.sin_cos else self.out_features
        if self.conv_block is not None:
            self.layer = self.conv_block(in_features, layer_out_features, kernel_size=1, stride=1, bias=True)
        else:
            self.layer = nn.Linear(in_features, layer_out_features, bias=True)
        self.layer.requires_grad_(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv and len(x.shape) != 4:
            raise ValueError(f"Expected shape of length 4, but got {x.shape}")
        if not self.use_conv and len(x.shape) != 2:
            raise ValueError(f"Expected shape of length 2, but got {x.shape}")
        x = np.pi * self.layer(x)
        if self.sin_cos:
            if self.use_conv:
                # shape of input is (B, C, W, H)
                return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
            else:
                return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            return torch.sin(x)


