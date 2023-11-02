import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from src.game.initialization import get_game_from_config
from src.network import Network, NetworkConfig
from src.network.fcn import FCN
from src.network.utils import ActivationType, NormalizationType


@dataclass
class SimpleNetworkConfig(NetworkConfig):
    num_layer: int = 3
    hidden_size: int = 64


class SimpleNetwork(Network):
    def __init__(self, cfg: SimpleNetworkConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.game = get_game_from_config(self.cfg.game_cfg)
        self.fcn = FCN(
            input_size=self.game.get_obs_shape()[0],
            hidden_size=self.cfg.hidden_size,
            output_size=self.cfg.game_cfg.num_actions,
            num_layer=self.cfg.num_layer,
            activation_type=ActivationType.RELU,
            dropout_p=0,
            norm_type=NormalizationType.NONE,
        )

    def _forward_impl(
            self,
            x: torch.Tensor,
            temperatures: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.fcn(x)
        return out

    @staticmethod
    def retrieve_value(output_tensor: torch.Tensor) -> torch.Tensor:
        raise ValueError(f"Simple network does not predict a value")

    def retrieve_policy(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output_tensor (): Output of the forward function
        Returns: Tensor (of shape (n,a), n players, a actions per player. (b,n,a) if output is a batch) containing
            the actions part of the output
        """
        if not self.cfg.predict_policy:
            raise ValueError("This network does not predict a policy")
        if not torch.any(torch.isfinite(output_tensor)) or torch.any(torch.isnan(output_tensor)):
            raise Exception(f"Network action output contains invalid numbers: {output_tensor}")
        return output_tensor

    def retrieve_length(self, output_tensor: torch.Tensor) -> torch.Tensor:
        raise ValueError(f"Simple network does not predict a game length")

    def load_weights(self, weight_path: Path):
        """
        Convenience function, which loads the weights of keras model and loads it into this FCN.
        Used when training bc-model in overcooked with code from other repo for comparison.
        """
        with open(weight_path, "rb") as f:
            weight_list = pickle.load(f)
        tensor_list = [torch.tensor(w, dtype=torch.float32) for w in weight_list]
        new_state_dict = {
            'lin_in.weight': tensor_list[0].T,
            'lin_in.bias': tensor_list[1],
            'hidden.0.weight': tensor_list[2].T,
            'hidden.0.bias': tensor_list[3],
            'lin_out.weight': tensor_list[4].T,
            'lin_out.bias': tensor_list[5],
        }
        self.fcn.load_state_dict(new_state_dict)



