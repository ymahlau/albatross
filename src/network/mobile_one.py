import copy
from dataclasses import field, dataclass
from functools import cached_property

import torch
from torch import nn

import torch.nn.functional as F

from src.network import HeadConfig
from src.network.fcn import MediumHeadConfig, SmallHeadConfig
from src.network.utils import ActivationType
from src.network.vision_net import VisionNetworkConfig, VisionNetwork


class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            inference_mode: bool = False,
            use_se: bool = False,
            num_conv_branches: int = 1
    ) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = None
            if out_channels == in_channels and stride == 1:
                self.rbr_skip = nn.BatchNorm2d(num_features=in_channels)

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(
                    self._conv_bn(kernel_size=kernel_size, padding=padding)
                )
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0].conv.in_channels,
            out_channels=self.rbr_conv[0].conv.out_channels,
            kernel_size=self.rbr_conv[0].conv.kernel_size,
            stride=self.rbr_conv[0].conv.stride,
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation,
            groups=self.rbr_conv[0].conv.groups,
            bias=True
        )
        self.reparam_conv.weight.data = kernel
        if self.reparam_conv.bias is None:
            raise Exception("bias is None")
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')
        self.inference_mode = True

    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batch norm layer with preceding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batch_norm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            'conv',
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module(
            'bn',
            nn.BatchNorm2d(num_features=self.out_channels),
        )
        return mod_list


@dataclass(kw_only=True)
class MobileOneConfig(VisionNetworkConfig):
    layer_specs: list[list[int]]

class MobileOneNetwork(VisionNetwork):
    def __init__(
            self,
            cfg: MobileOneConfig,
    ) -> None:
        """
        Implementation of MobileOne Network. Code adapted from:
        - https://github.com/apple/ml-mobileone/blob/main/mobileone.py
        Original paper: https://arxiv.org/abs/2206.04040
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.inference_mode = False

        cur_in_channels = self.transformation_out_features
        stage_list = []
        for out_channels, num_blocks, num_se_blocks, stride, exp_factor, overparam_k in self.cfg.layer_specs:
            stage = self._make_stage(
                stride=stride,
                in_channels=cur_in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                num_se_blocks=num_se_blocks,
                overparam_k=overparam_k,
            )
            stage_list.append(stage)
            cur_in_channels = out_channels
        self.backbone_model = nn.Sequential(*stage_list)
        self.initialize_weights()

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        out = self.backbone_model(x)
        return out

    @staticmethod
    def _make_stage(
            stride: int,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            num_se_blocks: int,
            overparam_k: int,
    ) -> nn.Sequential:
        """ Build a stage of MobileOne model.
        :param stride: Stride of first block
        :param out_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :param overparam_k: Over-parameterization factor k
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for idx, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot exceed number of layers.")
            if idx >= (num_blocks - num_se_blocks):
                use_se = True
            # Depth-wise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    inference_mode=False,
                    use_se=use_se,
                    num_conv_branches=overparam_k,
                )
            )
            # Pointwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    inference_mode=False,
                    use_se=use_se,
                    num_conv_branches=overparam_k,
                )
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)

    @cached_property
    def latent_size(self) -> int:
        # first entry of the last stage is channels
        result = self.cfg.layer_specs[-1][0]
        return result


def reparameterize_model(model) -> MobileOneNetwork:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # save and delete game attribute for deepcopy
    cpy = model.game.get_copy()
    del model.game
    # Avoid editing original graph
    new_model = copy.deepcopy(model)
    for module in new_model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    # add copy to model again
    model.game = cpy
    new_model.game = cpy.get_copy()
    return new_model


# out_channels, num_blocks, num_se_blocks, stride, exp_factor, overparam_k
default_3x3 = [
    [64, 2, 1, 1, 2, 1],
]
@dataclass
class MobileOneConfig3x3(MobileOneConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_3x3)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: SmallHeadConfig())
    value_head_cfg: HeadConfig = field(default_factory=lambda: SmallHeadConfig())

# out_channels, num_blocks, num_se_blocks, stride, exp_factor, overparam_k
default_7x7 = [
    [64,  2, 0, 1, 3, 1],  # 13
    [128, 8, 0, 2, 3, 1],  # 13->7
    [256, 5, 2, 2, 3, 1],  # 7->4
    [512, 2, 2, 2, 4, 1],  # 4->2
]
@dataclass
class MobileOneConfig7x7(MobileOneConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: default_7x7)
    policy_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig(hidden_size=256, num_layers=2))
    value_head_cfg: HeadConfig = field(default_factory=lambda: MediumHeadConfig(hidden_size=256, num_layers=2))


# out_channels, num_blocks, num_se_blocks, stride, exp_factor, overparam_k
incumbent_7x7 = [
    [128,  1, 0, 1, 3, 1],  # 13
    [256, 4, 4, 2, 3, 1],  # 13->7
    [512, 2, 2, 2, 3, 1],  # 7->4
    [1024, 0, 0, 2, 4, 1],  # 4->2
]
@dataclass
class MobileOneIncumbentConfig7x7(MobileOneConfig):
    layer_specs: list[list[int]] = field(default_factory=lambda: incumbent_7x7)
    lff_features: bool = field(default=True)
    lff_feature_expansion: bool = field(default=25)
    policy_head_cfg: HeadConfig = field(
        default_factory=lambda: HeadConfig(num_layers=1, final_activation=ActivationType.NONE)
    )
    value_head_cfg: HeadConfig = field(
        default_factory= lambda: HeadConfig(
            num_layers=2,
            final_activation=ActivationType.TANH,
            dropout_p=0.3,
            hidden_size=368,
        )
    )
