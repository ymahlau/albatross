from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Union, Type

import torch
from torch import nn
from torch.nn import functional as F

from src.network.fcn import SmallHeadConfig, HeadConfig, MediumHeadConfig
from src.network.invariant_conv import InvariantConvolution
from src.network.utils import NormalizationType, ActivationType, get_normalization_func, get_activation_func
from src.network.vision_net import VisionNetworkConfig, VisionNetwork

"""
Implementation of MobileNet-v3 in Pytorch
Implementation adapted from:
https://github.com/pytorch/vision/blob/11bf27e37190b320216c349e39b085fb33aefed1/torchvision/models/mobilenetv3.py
Original Paper:
https://arxiv.org/abs/1905.02244
"""

LAST_EXPAND_FACTOR = 6

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        norm_type: NormalizationType,
        activation_type: ActivationType,
        conv_class: Union[Type[torch.nn.Conv2d], Type[InvariantConvolution]],
    ) -> None:
        if kernel_size == 1:
            conv_class = torch.nn.Conv2d
        padding = (kernel_size - 1) // 2
        norm_layer = get_normalization_func(norm_type, num_features=out_planes, affine=True)
        activation_layer = get_activation_func(activation_type, inplace=True)
        super().__init__(
            conv_class(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            norm_layer,
            activation_layer,
        )
        self.out_channels = out_planes


class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(
            self,
            input_channels: int,
            squeeze_factor: int = 4,
    ):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input_tensor: torch.Tensor, inplace: bool) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(input_tensor, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input_tensor, True)
        out = scale * input_tensor
        return out


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
            self,
            in_planes: int,
            exp_planes: int,
            out_planes: int,
            kernel_size: int,
            stride: int,
            use_se: bool,
            norm_type: NormalizationType,
            activation_type: ActivationType,
            conv_class: Union[Type[torch.nn.Conv2d], Type[InvariantConvolution]],
    ):
        super().__init__()
        self.use_res_connect = stride == 1 and in_planes == out_planes

        layers: list[nn.Module] = []
        # expand
        if exp_planes != in_planes:
            layers.append(ConvBNActivation(
                    in_planes=in_planes,
                    out_planes=exp_planes,
                    kernel_size=1,
                    stride=1,
                    norm_type=norm_type,
                    activation_type=activation_type,
                    conv_class=conv_class,
            ))
        # depth-wise
        layers.append(
            ConvBNActivation(
                in_planes=exp_planes,
                out_planes=exp_planes,
                kernel_size=kernel_size,
                norm_type=norm_type,
                activation_type=activation_type,
                stride=stride,
                conv_class=conv_class,
            ))
        if use_se:
            layers.append(SqueezeExcitation(input_channels=exp_planes))
        # projection
        layers.append(ConvBNActivation(
            in_planes=exp_planes,
            out_planes=out_planes,
            kernel_size=1,
            stride=1,
            norm_type=norm_type,
            activation_type=activation_type,
            conv_class=conv_class,
        ))
        self.block = nn.Sequential(*layers)
        self.out_channels = out_planes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        result = self.block(input_tensor)
        if self.use_res_connect:
            result += input_tensor
        return result


@dataclass(kw_only=True)
class MobileNetConfig(VisionNetworkConfig):
    layer_specs: list[list[int]]
    activation_type: ActivationType = field(default=ActivationType.HARDSWISH)


class MobileNetV3(VisionNetwork):
    def __init__(
            self,
            cfg: MobileNetConfig,
    ) -> None:
        """
        MobileNet V3 main class
        """
        super().__init__(cfg)
        self.cfg = cfg
        # building first layer
        layers: list[nn.Module] = []
        first_conv_output_channels = self.cfg.layer_specs[0][0]
        layers.append(
            ConvBNActivation(
                in_planes=self.transformation_out_features,
                out_planes=first_conv_output_channels,
                kernel_size=3,
                norm_type=self.cfg.norm_type,
                activation_type=self.cfg.activation_type,
                stride=1,
                conv_class=self.conv_class,
            )
        )
        # building inverted residual blocks
        for in_planes, exp_planes, out_planes, kernel_size, stride, use_se in self.cfg.layer_specs:
            layers.append(InvertedResidual(
                in_planes=in_planes,
                exp_planes=exp_planes,
                out_planes=out_planes,
                kernel_size=kernel_size,
                use_se=bool(use_se),
                stride=stride,
                norm_type=self.cfg.norm_type,
                activation_type=self.cfg.activation_type,
                conv_class=self.conv_class,
            ))
        # building last several layers
        self.LAST_EXPAND_FACTOR = 6
        last_conv_input_channels = self.cfg.layer_specs[-1][2]
        last_conv_output_channels = LAST_EXPAND_FACTOR * last_conv_input_channels
        layers.append(ConvBNActivation(
            in_planes=last_conv_input_channels,
            out_planes=last_conv_output_channels,
            kernel_size=1,
            stride=1,
            norm_type=self.cfg.norm_type,
            activation_type=self.cfg.activation_type,
            conv_class=self.conv_class,
        ))
        self.backbone_model = nn.Sequential(*layers)
        self.initialize_weights()

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone_model(x)
        return out

    @cached_property
    def latent_size(self) -> int:
        return LAST_EXPAND_FACTOR * self.cfg.layer_specs[-1][2]


# in_channels, exp_channels, out_channels, kernel_size, stride, se
default_3x3 = [
    [32, 64, 32, 3, 2, 1]
]
@dataclass
class MobileNetConfig3x3(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_3x3)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: SmallHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: SmallHeadConfig())


# in_channels, exp_channels, out_channels, kernel_size, stride, se
default_5x5 = [
    [32, 32, 32, 3, 1, 0],
    [32, 48, 32, 5, 1, 1],  # 11
    [32, 64, 48, 3, 2, 0],  # 5
    [48, 64, 48, 3, 2, 0],  # 3
]
@dataclass
class MobileNetConfig5x5(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_5x5)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig())

# in_channels, exp_channels, out_channels, kernel_size, stride, se
large_5x5 = [
    [32, 32, 32, 3, 1, 0],  # 11
    [32, 96, 64, 3, 2, 1],  # 5
    [64, 96, 64, 3, 1, 1],
    [64, 128, 96, 3, 2, 0],  # 3
    [96, 128, 96, 3, 1, 0],
]
@dataclass
class MobileNetConfig5x5Large(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: large_5x5)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig())


# in_channels, exp_channels, out_channels, kernel_size, stride, se
default_7x7 = [
    [32, 32, 32, 3, 1, 0],  # 15
    [32, 48, 32, 5, 1, 1],
    [32, 96, 64, 3, 2, 0],  # 8
    [64, 128, 64, 3, 1, 0],
    [64, 192, 64, 5, 1, 1],
    [64, 256, 64, 3, 2, 1],  # 4
    # [64, 256, 64, 3, 1, 1],
]
@dataclass
class MobileNetConfig7x7(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_7x7)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig(hidden_size=256, num_layers=2))
    value_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig(hidden_size=256, num_layers=2))

# in_channels, exp_channels, out_channels, kernel_size, stride, se
default_11x11 = [
    [32, 32, 32, 3, 1, 0],  # 21
    [32, 48, 32, 5, 1, 1],
    [32, 96, 48, 3, 2, 0],  # 11
    [48, 64, 48, 5, 1, 1],
    [48, 96, 64, 3, 2, 0],  # 6
    [64, 128, 64, 3, 1, 0],
    [64, 192, 64, 5, 1, 1],
    [64, 256, 128, 3, 2, 1],  # 3
    [128, 256, 128, 3, 1, 1],
]
@dataclass
class MobileNetConfig11x11(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_11x11)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig(hidden_size=256, num_layers=2))
    value_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig(hidden_size=256, num_layers=2))


# in_channels, exp_channels, out_channels, kernel_size, stride, se
default_11x11_large = [
    [96, 96, 96, 3, 1, 0],  # 21
    [96, 128, 96, 5, 1, 1],
    [96, 152, 128, 3, 2, 0],  # 11
    [128, 152, 128, 5, 1, 1],
    [128, 152, 152, 3, 2, 0],  # 6
    [152, 192, 152, 3, 1, 0],
    [152, 512, 256, 3, 2, 1],  # 3
    [256, 512, 256, 3, 1, 1],
]
@dataclass
class MobileNetConfig11x11Large(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_11x11_large)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: HeadConfig(hidden_size=256, num_layers=2))
    value_head_cfg: HeadConfig = field(default_factory=lambda: HeadConfig(hidden_size=256, num_layers=2))

# in_channels, exp_channels, out_channels, kernel_size, stride, se
incumbent_7x7 = [
    [64,  128, 64,  3, 1, 0],
    [64,  128, 64,  5, 1, 0],  # 15
    [64,  192, 128, 3, 2, 0],  # 8
    [128, 320, 128, 3, 1, 1],
    [128, 320, 128, 5, 1, 1],
    [128, 320, 192, 3, 2, 1],  # 4
    [192, 384, 192, 3, 1, 1],
    [192, 384, 192, 3, 1, 1],
]
@dataclass
class MobileNetConfig7x7Incumbent(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: incumbent_7x7)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.TANH)
    )
    
    
# in_channels, exp_channels, out_channels, kernel_size, stride, se
incumbent_9x9 = [
    [64,  128, 64,  3, 1, 0],
    [64,  128, 64,  5, 1, 0],  # 19
    [64,  192, 128, 3, 2, 0],  # 10
    [128, 320, 128, 3, 1, 1],
    [128, 320, 128, 5, 1, 1],
    [128, 320, 192, 3, 2, 1],  # 5
    [192, 384, 192, 3, 1, 1],
    [192, 384, 192, 3, 2, 1],  # 3
    [192, 384, 192, 3, 1, 1],
]
@dataclass
class MobileNetConfig9x9Incumbent(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: incumbent_9x9)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.TANH)
    )
    
# in_channels, exp_channels, out_channels, kernel_size, stride, se
incumbent_11x11 = [
    [64,  128, 64,  3, 1, 0],
    [64,  128, 64,  5, 1, 0],  # 21
    [64,  192, 128, 3, 2, 0],  # 11
    [128, 320, 128, 3, 1, 1],
    [128, 320, 128, 5, 1, 1],
    [128, 320, 192, 3, 2, 1],  # 6
    [192, 384, 192, 3, 1, 1],
    [192, 384, 192, 3, 2, 1],  # 3
    [192, 384, 192, 3, 1, 1],
]
@dataclass
class MobileNetConfig11x11Incumbent(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: incumbent_11x11)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.TANH)
    )


extrapolated_11x11 = [
    [64, 128, 64, 3, 1, 0],  # 21
    [64, 128, 64, 5, 1, 0],
    [64, 192, 128, 3, 2, 0],  # 11
    [128, 320, 128, 3, 1, 1],
    [128, 320, 128, 5, 1, 1],
    [128, 320, 192, 3, 2, 1],  # 6
    [192, 384, 192, 3, 1, 1],
    [192, 384, 192, 5, 1, 1],
    [192, 576, 384, 3, 2, 1],  # 3
    [384, 576, 384, 3, 1, 1],
    [384, 768, 384, 3, 1, 1],
]
@dataclass
class MobileNetConfig11x11Extrapolated(MobileNetConfig):
    predict_policy: bool = field(default=True)
    predict_game_len: bool = field(default=False)
    layer_specs: list[list[int]] = field(default_factory=lambda: extrapolated_11x11)
    lff_features: bool = field(default=True)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.TANH)
    )


overcooked_cramped = [
    [96,  128, 96,  3, 1, 0],  # 5
    [96,  192, 96,  5, 1, 0],
    [96,  192, 96,  3, 1, 0],
    [96,  256, 192, 3, 2, 1],  # 3
    [192, 320, 192, 3, 1, 1],
]
@dataclass
class MobileNetConfigOvercookedCramped(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: overcooked_cramped)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )

# in_channels, exp_channels, out_channels, kernel_size, stride, se
extrapolated_5x5 = [
    [64,  128, 64,  3, 1, 0],
    [64,  128, 64,  5, 1, 0],  # 9
    [64,  192, 128, 3, 2, 0],  # 5
    [128, 320, 128, 3, 1, 1],
    [128, 320, 192, 3, 2, 1],  # 3
    [192, 384, 192, 3, 1, 1],
]
@dataclass
class MobileNetConfig5x5Extrapolated(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: extrapolated_5x5)
    lff_features: bool = field(default=True)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.TANH)
    )


overcooked_asym_adv = [
    [64,  128, 64,  3, 1, 0],
    [64,  128, 64,  5, 1, 0],  # 9
    [64,  192, 128, 3, 2, 0],  # 5
    [128, 320, 128, 3, 1, 1],
    [128, 320, 192, 3, 2, 1],  # 3
    [192, 384, 192, 3, 1, 1],
]
@dataclass
class MobileNetConfigOvercookedAsymmetricAdvantage(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: overcooked_asym_adv)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )


overcooked_coord_ring = [
    [96,  128, 96,  3, 1, 0],  # 5
    [96,  192, 96,  5, 1, 0],
    [96,  192, 96,  3, 1, 0],
    [96,  256, 192, 3, 2, 1],  # 3
    [192, 320, 192, 3, 1, 1],
]
@dataclass
class MobileNetConfigOvercookedCoordinationRing(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: overcooked_coord_ring)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )


overcooked_forced_coord = [
    [96,  128, 96,  3, 1, 0],  # 5
    [96,  192, 96,  5, 1, 0],
    [96,  192, 96,  3, 1, 0],
    [96,  256, 192, 3, 2, 1],  # 3
    [192, 320, 192, 3, 1, 1],
]
@dataclass
class MobileNetConfigOvercookedForcedCoordination(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: overcooked_forced_coord)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )


overcooked_counter_circuit = [
    [64,  128, 64,  3, 1, 0],
    [64,  128, 64,  5, 1, 0],  # 8
    [64,  192, 128, 3, 2, 0],  # 4
    [128, 320, 128, 3, 1, 1],
    [128, 320, 192, 3, 2, 1],  # 2
    [192, 384, 192, 3, 1, 1],
]
@dataclass
class MobileNetConfigOvercookedCounterCircuit(MobileNetConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: overcooked_counter_circuit)
    lff_features: bool = field(default=False)
    lff_feature_expansion: int = field(default=27)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
