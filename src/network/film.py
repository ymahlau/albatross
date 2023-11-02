import torch
from torch import nn


class FiLM(nn.Module):
    """
    Implementation of Feature-Wise Transformation (FiLM):
    https://distill.pub/2018/feature-wise-transformations/
    """
    def __init__(self, feature_size: int):
        super().__init__()
        self.feature_size = feature_size
        self.scale = nn.Linear(feature_size, feature_size, bias=True)
        self.shift = nn.Linear(feature_size, feature_size, bias=True)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x is network latent space, z is output of FiLM-Generator.
        # They need to have the same lengths
        scale_out = self.scale(z)
        shift_out = self.shift(z)
        result = scale_out * x + shift_out
        return result
