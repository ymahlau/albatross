from enum import Enum
from typing import Optional, Any

from torch import nn


def cleanup_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Removes _orig_mod. which is introduced by torch.compile()
    """
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.replace("_orig_mod.", "")] = state_dict[key]
    return new_state_dict


class ActivationType(Enum):
    NONE = 'NONE'
    LEAKY_RELU = 'LEAKY_RELU'
    RELU = 'RELU'
    TANH = 'TANH'
    SELU = 'SELU'
    SILU = 'SILU'
    SIGMOID = 'SIGMOID'
    HARDSWISH = 'HARDSWISH'


def get_activation_func(activation_type: ActivationType, inplace: bool = False) -> nn.Module:
    if activation_type == ActivationType.NONE or activation_type == ActivationType.NONE.value:
        return nn.Identity()
    elif activation_type == ActivationType.LEAKY_RELU or activation_type == ActivationType.LEAKY_RELU.value:
        return nn.LeakyReLU(inplace=inplace)
    elif activation_type == ActivationType.RELU or activation_type == ActivationType.RELU.value:
        return nn.ReLU(inplace=inplace)
    elif activation_type == ActivationType.TANH or activation_type == ActivationType.TANH.value:
        return nn.Tanh()
    elif activation_type == ActivationType.SELU or activation_type == ActivationType.SELU.value:
        return nn.SELU(inplace=inplace)
    elif activation_type == ActivationType.SILU or activation_type == ActivationType.SILU.value:
        return nn.SiLU(inplace=inplace)
    elif activation_type == ActivationType.SIGMOID or activation_type == ActivationType.SIGMOID.value:
        return nn.Sigmoid()
    elif activation_type == ActivationType.HARDSWISH or activation_type == ActivationType.HARDSWISH.value:
        return nn.Hardswish(inplace=inplace)
    else:
        raise ValueError(f"Unknown activation function type: {activation_type}")


class NormalizationType(Enum):
    NONE = 'NONE'
    BATCH_NORM = 'BATCH_NORM'
    LAYER_NORM = 'LAYER_NORM'
    INSTANCE_NORM = 'INSTANCE_NORM'
    GROUP_NORM = 'GROUP_NORM'


def get_normalization_func(
        norm_type: NormalizationType,
        affine: bool,
        num_features: Optional[int] = None,
        normalized_shape: Optional[list[int]] = None,
) -> nn.Module:
    if norm_type == NormalizationType.NONE or norm_type == NormalizationType.NONE.value:
        return nn.Identity()
    elif norm_type == NormalizationType.BATCH_NORM or norm_type == NormalizationType.BATCH_NORM.value:
        if num_features is None:
            raise ValueError("Need feature number for batch norm")
        return nn.BatchNorm2d(num_features=num_features, affine=affine)
    elif norm_type == NormalizationType.LAYER_NORM or norm_type == NormalizationType.LAYER_NORM.value:
        if normalized_shape is None:
            raise ValueError("Need norm shape for layer norm")
        return nn.LayerNorm(normalized_shape=normalized_shape, elementwise_affine=affine)
    elif norm_type == NormalizationType.INSTANCE_NORM or norm_type == NormalizationType.INSTANCE_NORM.value:
        if num_features is None:
            raise ValueError("Need feature number for instance norm")
        return nn.InstanceNorm2d(num_features=num_features, affine=affine)
    elif norm_type == NormalizationType.GROUP_NORM or norm_type == NormalizationType.GROUP_NORM.value:
        if num_features is None:
            raise ValueError("Need feature number for group norm")
        return nn.GroupNorm(num_groups=8, num_channels=num_features, affine=affine)
    else:
        raise ValueError(f"Unknown norm function type: {norm_type}")
