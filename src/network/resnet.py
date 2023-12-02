from dataclasses import dataclass, field
from functools import cached_property
from typing import Union, Type

import torch
from torch import nn

from src.network.fcn import HeadConfig, LargeHeadConfig, MediumHeadConfig, SmallHeadConfig, WideHeadConfig
from src.network.invariant_conv import InvariantConvolution
from src.network.utils import NormalizationType, ActivationType, get_normalization_func, get_activation_func
from src.network.vision_net import VisionNetwork, VisionNetworkConfig


class ResNetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            in_w: int,
            in_h: int,
            kernel_size: int,
            padding: int,
            norm_type: NormalizationType,
            activation_type: ActivationType,
            conv_class: Union[Type[torch.nn.Conv2d], Type[InvariantConvolution]],

    ):
        super(ResNetBlock, self).__init__()
        # calculate resolution decrement
        if kernel_size % 2 == 0:
            raise ValueError(f"Only odd kernel sizes supported, but got even size {kernel_size}")
        # resolution_decrement = (kernel_size - 1) // 2 - padding
        resolution_decrement = kernel_size - 1 - 2 * padding
        total_decrement = 2 * resolution_decrement
        # if resolution is not the same, we need to downsample skip connection
        self.identity_resolution = kernel_size - 1 == 2 * padding
        if not self.identity_resolution:
            self.pool = nn.AvgPool2d(kernel_size=(total_decrement + 1, total_decrement + 1), stride=1, padding=0)
        # if channel number is not the same, we need to up/downsample skip connection
        self.identity_channel = in_channels == out_channels
        if not self.identity_channel:
            self.downsample = nn.Conv2d(in_channels, out_channels, stride=1, padding=0, kernel_size=1, bias=False)
        # normalization method
        self.norm1 = get_normalization_func(
            norm_type=norm_type,
            affine=True,
            num_features=out_channels,
            normalized_shape=[out_channels, in_w - resolution_decrement, in_h - resolution_decrement],
        )
        self.norm2 = get_normalization_func(
            norm_type=norm_type,
            affine=True,
            num_features=out_channels,
            normalized_shape=[out_channels, in_w - total_decrement, in_h - total_decrement],
        )
        # convolutions
        self.conv1 = conv_class(in_channels, out_channels, stride=1, padding=padding, kernel_size=kernel_size,
                                bias=False)
        self.conv2 = conv_class(out_channels, out_channels, stride=1, padding=padding, kernel_size=kernel_size,
                                bias=False)
        # activations
        self.activ_func = get_activation_func(activation_type)

    def forward(self, x: torch.Tensor):
        # feature connection
        y = self.conv1(x)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.activ_func(y)
        y = self.conv2(y)
        if self.norm2 is not None:
            y = self.norm2(y)
        # skip connection
        skip = x
        if not self.identity_resolution:
            skip = self.pool(skip)
        if not self.identity_channel:
            skip = self.downsample(skip)
        # add
        z = skip + y
        z = self.activ_func(z)
        return z


class ResNetGroup(nn.Module):
    def __init__(
            self,
            num_blocks: int,
            in_channels: int,
            out_channels: int,
            in_w: int,
            in_h: int,
            kernel_size: int,
            padding: int,
            norm_type: NormalizationType,
            activation_type: ActivationType,
            conv_class: Union[Type[torch.nn.Conv2d], Type[InvariantConvolution]],
    ):
        super().__init__()
        block_list = [
            ResNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                in_w=in_w,
                in_h=in_h,
                kernel_size=kernel_size,
                padding=padding,
                norm_type=norm_type,
                activation_type=activation_type,
                conv_class=conv_class,
            ),
        ]
        decrement = (kernel_size - 1 - 2 * padding)
        for _ in range(1, num_blocks):
            in_w -= 2 * decrement
            in_h -= 2 * decrement
            block_list.append(
                ResNetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    in_w=in_w,
                    in_h=in_h,
                    kernel_size=kernel_size,
                    padding=padding,
                    norm_type=norm_type,
                    activation_type=activation_type,
                    conv_class=conv_class,
                ),
            )
        self.seq = nn.Sequential(*block_list)

    def forward(self, x: torch.Tensor):
        y = self.seq(x)
        return y


class ResNet(VisionNetwork):
    def __init__(
            self,
            cfg: "ResNetConfig"
    ):
        super().__init__(cfg)
        self.cfg = cfg  # explicitly set config again to satisfy code analytics
        # backbone model
        group_list = []
        cur_in_channels = self.transformation_out_features
        cur_in_w = self.in_w
        cur_in_h = self.in_h
        for channels, num_blocks, kernel_size, padding, norm in cfg.layer_specs:
            group_list.append(
                ResNetGroup(
                    num_blocks=num_blocks,
                    in_channels=cur_in_channels,
                    out_channels=channels,
                    in_w=cur_in_w,
                    in_h=cur_in_h,
                    kernel_size=kernel_size,
                    padding=padding,
                    norm_type=cfg.norm_type if norm else NormalizationType.NONE,
                    activation_type=cfg.activation_type,
                    conv_class=self.conv_class,
                )
            )
            cur_in_channels = channels
            cur_in_w -= num_blocks * 2 * (kernel_size - 1 - 2 * padding)
            cur_in_h -= num_blocks * 2 * (kernel_size - 1 - 2 * padding)
            if cur_in_channels <= 0 or cur_in_w <= 0 or cur_in_h <= 0:
                raise ValueError(f"Invalid layer specs: {cfg.layer_specs}")
        self.backbone_model = nn.Sequential(*group_list)
        self.initialize_weights()

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone_model(x)
        return out

    @cached_property
    def latent_size(self) -> int:
        return self.cfg.layer_specs[-1][0]


@dataclass(kw_only=True)
class ResNetConfig(VisionNetworkConfig):
    layer_specs: list[list[int]]
    norm_type: NormalizationType = NormalizationType.GROUP_NORM


# channels, num_blocks, kernel_size, padding, norm
default_3x3 = [
    [32, 1, 3, 0, 1]
]


@dataclass
class ResNetConfig3x3(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_3x3)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: SmallHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: SmallHeadConfig())


# channels, num_blocks, kernel_size, padding, norm
default_5x5 = [
    [32, 2, 3, 1, 1],
    [48, 1, 3, 0, 1],
    [48, 1, 3, 1, 1],
    [64, 1, 3, 0, 1],
]


@dataclass
class ResNetConfig5x5(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_5x5)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig())


# channels, num_blocks, kernel_size, padding, norm
best_5x5 = [
    [32, 3, 3, 1, 1],
    [64, 2, 3, 1, 1],
    [128, 1, 3, 1, 1],
    [256, 1, 3, 0, 1],
]


@dataclass
class OvercookedResNetConfig5x5(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: best_5x5)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())


# channels, num_blocks, kernel_size, padding, norm
best_9x9 = [
    [32, 3, 3, 1, 1],
    [64, 2, 3, 1, 1],
    [64, 1, 3, 0, 1],
    [128, 2, 3, 1, 1],
    [256, 1, 3, 0, 1],
]


@dataclass
class OvercookedResNetConfig9x9(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: best_9x9)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())


# channels, num_blocks, kernel_size, padding, norm
best_8x8 = [
    [32, 3, 3, 1, 1],
    [64, 2, 3, 1, 1],
    [64, 1, 3, 0, 1],
    [128, 2, 3, 1, 1],
    [256, 1, 3, 1, 1],
]


@dataclass
class OvercookedResNetConfig8x8(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: best_8x8)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: WideHeadConfig())


default_7x7 = [
    [32, 2, 3, 1, 1],
    [48, 1, 3, 0, 1],
    [48, 1, 3, 1, 1],
    [64, 1, 3, 0, 1],
    [80, 2, 3, 1, 1],
    [96, 1, 3, 0, 1],
]


@dataclass
class ResNetConfig7x7(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_7x7)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig())


default_7x7_large = [
    [32, 3, 3, 1, 1],
    [48, 1, 3, 0, 1],
    [48, 3, 3, 1, 1],
    [64, 1, 3, 0, 1],
    [80, 3, 3, 1, 1],
    [96, 1, 3, 0, 1],
]


@dataclass
class ResNetConfig7x7Large(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_7x7_large)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: LargeHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: LargeHeadConfig())


@dataclass
class ResNetConfig7x7New(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_7x7_large)
    lff_features: bool = field(default=True)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.TANH)
    )


# channels, num_blocks, kernel_size, padding, norm
best_7x7 = [
    [32, 1, 3, 1, 1],
    [64, 1, 3, 0, 1],
    [96, 1, 3, 1, 1],
    [128, 1, 3, 0, 1],
    [256, 1, 3, 1, 1],
    [384, 1, 3, 0, 1],
]


@dataclass
class ResNetConfig7x7Best(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: best_7x7)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.TANH)
    )


# channels, num_blocks, kernel_size, padding, norm
default_centered_9x9 = [
    [32, 2, 3, 1, 1],
    [48, 1, 3, 0, 1],
    [48, 2, 3, 1, 1],
    [64, 1, 3, 0, 1],
    [80, 2, 3, 1, 1],
    [80, 1, 3, 0, 1],
    [96, 2, 3, 1, 1],
    [128, 1, 3, 0, 1],
]


@dataclass
class ResNetConfig9x9(ResNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_centered_9x9)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: LargeHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: LargeHeadConfig())


# channels, num_blocks, kernel_size, padding, norm
default_centered_11x11 = [
    [32, 1, 3, 1, 1],
    [64, 1, 3, 0, 1],  # 21-17
    [96, 1, 3, 0, 1],  # 17-13
    [128, 1, 3, 0, 1],  # 13-9
    [256, 1, 3, 0, 1],  # 9-5
    [384, 1, 3, 1, 1],
    [512, 1, 3, 0, 1],  # 5-1
]


@dataclass
class ResNetConfig11x11(ResNetConfig):
    predict_policy: bool = field(default=True)
    layer_specs: list[list[int]] = field(default_factory=lambda: default_centered_11x11)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.TANH)
    )
