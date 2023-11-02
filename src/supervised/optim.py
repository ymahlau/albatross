from dataclasses import dataclass
from enum import Enum

import torch

from src.network import Network
from src.supervised.annealer import TemperatureAnnealingConfig, TemperatureAnnealer


class OptimType(Enum):
    ADAM = 'ADAM'
    ADAM_W = 'ADAM_W'


@dataclass
class OptimizerConfig:
    anneal_cfg: TemperatureAnnealingConfig
    optim_type: OptimType = OptimType.ADAM
    weight_decay: float = 0
    beta1: float = 0.9
    beta2: float = 0.99
    fused: bool = False


def get_optim_from_config(
        cfg: OptimizerConfig,
        net: Network,
) -> tuple[torch.optim.Optimizer, TemperatureAnnealer]:
    # construct optim
    if cfg.optim_type == OptimType.ADAM or cfg.optim_type == OptimType.ADAM.value:
        optim = torch.optim.Adam(
            params=net.parameters(),
            lr=cfg.anneal_cfg.init_temp,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            fused=cfg.fused,
        )
    elif cfg.optim_type == OptimType.ADAM_W or cfg.optim_type == OptimType.ADAM_W.value:
        optim = torch.optim.AdamW(
            params=net.parameters(),
            lr=cfg.anneal_cfg.init_temp,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            fused=cfg.fused,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.optim_type}")
    # annealer
    annealer = TemperatureAnnealer(cfg.anneal_cfg, optim)
    return optim, annealer
