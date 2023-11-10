import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d
from torchvision.transforms.functional import rotate, vflip

from src.network import Network, NetworkConfig
from src.network.fcn import HeadConfig, head_from_cfg
from src.network.invariant_conv import InvariantConvolution
from src.network.lff import LearnedFourierFeatures
from src.network.utils import ActivationType, NormalizationType


class EquivarianceType(Enum):
    NONE = 'NONE'  # normal convolutions
    CONSTRAINED = 'CONSTRAINED'  # convolutions with constrained kernels
    POOLED = 'POOLED'  # pool over all possible symmetries


@dataclass(kw_only=True)
class VisionNetworkConfig(NetworkConfig):
    value_head_cfg: HeadConfig
    policy_head_cfg: Optional[HeadConfig] = None
    activation_type: ActivationType = ActivationType.LEAKY_RELU
    norm_type: NormalizationType = NormalizationType.GROUP_NORM
    eq_type: EquivarianceType = EquivarianceType.NONE
    lff_features: bool = False
    lff_feature_expansion: int = 40

class VisionNetwork(Network, ABC):
    def __init__(self, cfg: VisionNetworkConfig):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.policy_head_cfg is None and self.cfg.predict_policy:
            raise ValueError("Need head config for predicting policy")
        # input shape
        if len(self.game.get_obs_shape()) != 3:
            raise ValueError(f"Invalid input shape for resnet: {self.game.get_obs_shape()}")
        self.in_w = self.game.get_obs_shape()[0]
        self.in_h = self.game.get_obs_shape()[1]
        self.in_channels = self.game.get_obs_shape()[2]
        # equivariance convs
        if self.cfg.eq_type == EquivarianceType.CONSTRAINED or self.cfg.eq_type == EquivarianceType.CONSTRAINED.value:
            if self.cfg.predict_policy:
                raise ValueError(f"Cannot predict policy with constrained kernels (directional information is lost)")
            self.conv_class = InvariantConvolution
        else:
            self.conv_class = Conv2d
        # transformations
        if self.cfg.lff_features:
            self.transformation_out_features = self.in_channels * self.cfg.lff_feature_expansion
            self.lff = LearnedFourierFeatures(
                in_features=self.in_channels,
                out_features=self.transformation_out_features,
                conv_block=self.conv_class,
                sin_cos=False,
                trainable=True,
            )
        else:
            self.transformation_out_features = self.in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # heads
        self.value_head = head_from_cfg(
            self.cfg.value_head_cfg,
            input_size=self.latent_size,
            output_size=1,
        )
        if self.cfg.predict_policy:
            self.policy_head = head_from_cfg(
                self.cfg.policy_head_cfg,
                input_size=self.latent_size,
                output_size=self.game.num_actions,
            )
        # permutation for grouping the actions of all symmetries. A bit convoluted, but quick to compute.
        # Symmetries: 0,  1,  2,  3,  4,  5,  6,  7
        # ------------------------------------------
        #  UP (0)     0   2   3   3   2   0   1   1
        #  RIGHT (1)  1   1   0   2   3   3   2   0
        #  DOWN (2)   2   0   1   1   0   2   3   3
        #  LEFT (3)   3   3   2   0   1   1   0   2
        self.permutation = torch.tensor(np.asarray([
            0, 6, 11, 15, 18, 20, 25, 29,
            1, 5, 8,  14, 19, 23, 26, 28,
            2, 4, 9,  13, 16, 22, 27, 31,
            3, 7, 10, 12, 17, 21, 24, 30,
        ], dtype=int))

    def pre_transform(self, x: torch.Tensor) -> torch.Tensor:
        # we can only work with batched 4d tensors
        if len(x.shape) != 4:
            raise ValueError(f'Can only work with 4d-Tensors, but got {len(x.shape)}d')
        # convolutions expect input of shape (batch, channels, h, w)
        y = torch.permute(x, dims=[0, 3, 1, 2])
        # y = torch.transpose(x, 1, 3)
        # in pooled equivariance mode, add all possible symmetries
        if self.cfg.eq_type == EquivarianceType.POOLED or self.cfg.eq_type == EquivarianceType.POOLED.value:
            # do rotation/reflection and transpose back
            y = torch.transpose(y, -1, -2)
            sym_list = [
                y,
                vflip(y),
                rotate(y, -90),
                vflip(rotate(y, -90)),
                rotate(y, -180),
                vflip(rotate(y, -180)),
                rotate(y, -270),
                vflip(rotate(y, -270)),
            ]
            cat = torch.cat(sym_list, dim=0)
            out = torch.transpose(cat, -1, -2)
        else:
            out = y
        # possible do learned fourier feature transform
        if self.cfg.lff_features:
            out = self.lff(out)
        return out

    def _forward_impl(self, x: torch.Tensor, temperatures: Optional[torch.Tensor] = None) -> torch.Tensor:
        # backbone forward pass
        pre_transformed = self.pre_transform(x)
        backbone_out = self.backbone(pre_transformed)
        # pooling and feature transform
        pooled = self.avg_pool(backbone_out)
        latent = torch.flatten(pooled, 1)
        # heads
        tensor_list = []
        if self.cfg.predict_policy:
            policy_out = self.policy_head(latent)
            tensor_list.append(policy_out)
        value_out = self.value_head(latent)
        tensor_list.append(value_out)
        # post transform and cat
        post_transformed = self.post_transform(tensor_list)
        out = torch.cat(post_transformed, dim=-1)
        return out

    def post_transform(self, out_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.cfg.eq_type == EquivarianceType.POOLED or self.cfg.eq_type == EquivarianceType.POOLED.value:
            out = []
            counter = 0
            # policy
            if self.cfg.predict_policy:
                policy_out = out_list[counter]
                policy_reshaped = torch.reshape(policy_out, (8, -1, 4))
                policy_transposed = torch.transpose(policy_reshaped, 0, 1)  # shape (-1, 8, 4)
                policy_reshaped2 = torch.reshape(policy_transposed, (-1, 32))
                policy_permuted = policy_reshaped2[:, self.permutation]
                policy_grouped = torch.reshape(policy_permuted, (-1, 4, 8))
                policy_pooled = torch.mean(policy_grouped, dim=-1)  # shape(-1, 4)
                out.append(policy_pooled)
                counter += 1
            # value
            value_out = out_list[counter]
            value_reshaped = torch.reshape(value_out, (8, -1))
            value_pooled = torch.mean(value_reshaped, dim=0).unsqueeze(-1)
            out.append(value_pooled)
        else:
            out = out_list
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, InvariantConvolution):
                for weight_tensor in m.w:
                    # nn.init.kaiming_normal_(weight_tensor, mode='fan_out')
                    nn.init.orthogonal_(weight_tensor, gain=math.sqrt(2))
                if m.bias_params is not None:
                    nn.init.zeros_(m.bias_params)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
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

    @abstractmethod
    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    @cached_property
    def latent_size(self) -> int:
        raise NotImplementedError()
