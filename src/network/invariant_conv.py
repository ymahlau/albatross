from typing import Optional

import torch
from torch import nn


class InvariantConvolution(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int = 0,
            bias: bool = True,
            stride: int = 1,
    ):
        super().__init__()
        if kernel_size != 3 and kernel_size != 5:
            raise ValueError(f"Currently only kernel sizes of 3 and 5 are supported, but not {kernel_size}")
        # attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = padding
        # parameter
        if self.bias:
            self.bias_params = nn.Parameter(data=torch.zeros((out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias_params = None
        self.num_params = 3 if kernel_size == 3 else 6
        self.w = nn.ParameterList([
            nn.Parameter(data=torch.ones((out_channels, in_channels), dtype=torch.float32) * i)
            for i in range(self.num_params)
        ])
        # for efficiency in eval mode
        self.expanded_kernel: Optional[torch.Tensor] = None

    def generate_kernel(self) -> torch.Tensor:
        if self.kernel_size == 3:
            row0 = torch.stack([self.w[0], self.w[1], self.w[0]], dim=-1)
            row1 = torch.stack([self.w[1], self.w[2], self.w[1]], dim=-1)
            kernel = torch.stack([row0, row1, row0], dim=-1)
        else:
            row0 = torch.stack([self.w[0], self.w[1], self.w[2], self.w[1], self.w[0]], dim=-1)
            row1 = torch.stack([self.w[1], self.w[3], self.w[4], self.w[3], self.w[1]], dim=-1)
            row2 = torch.stack([self.w[2], self.w[4], self.w[5], self.w[4], self.w[2]], dim=-1)
            kernel = torch.stack([row0, row1, row2, row1, row0], dim=-1)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expanded_kernel is None:
            kernel = self.generate_kernel()
        else:
            kernel = self.expanded_kernel
        out = torch.nn.functional.conv2d(
            input=x,
            weight=kernel,
            bias=self.bias_params,
            stride=self.stride,
            padding=self.padding,
        )
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # train mode
            self.expanded_kernel = None
        else:
            # eval mode
            self.expanded_kernel = self.generate_kernel()

    def eval(self):
        super().eval()
        self.expanded_kernel = self.generate_kernel()
