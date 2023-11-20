import math
from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Optional

import torch


class AnnealingType(Enum):
    CONST = 'CONST'
    COSINE = 'COSINE'
    DOUBLE_COS = 'DOUBLE_COS'
    LINEAR = 'LINEAR'
    LOG = 'LOG'
    ENHANCED_COS = 'ENHANCED_COS'
    ENHANCED_DCOS = 'ENHANCED_DCOS'

@dataclass
class TemperatureAnnealingConfig:
    init_temp: float
    anneal_temps: list[float] = field(default_factory=lambda: [])
    anneal_types: list[AnnealingType] = field(default_factory=lambda: [])
    end_times_min: list[float] = field(default_factory=lambda: [])
    cyclic: bool = False
    sampling: bool = False

class TemperatureAnnealer:
    """
    Anneals the temperature with a specific function. If an Optimizer is given, then it updates its learning rate
    parameter.
    """
    def __init__(self, cfg: TemperatureAnnealingConfig, optim: Optional[torch.optim.Optimizer] = None):
        if len(cfg.anneal_types) != len(cfg.end_times_min) != len(cfg.anneal_temps):
            raise ValueError(f"List for annealing need to have same lengths")
        if cfg.cyclic:
            if len(cfg.anneal_types) != 1 or len(cfg.anneal_temps) != 1 or len(cfg.end_times_min) != 1:
                raise ValueError(f"Invalid cyclic scheduling: {cfg}")
        self.optim = optim
        self.cfg = cfg
        self.n_phases = len(self.cfg.end_times_min)

    def __call__(self, time_passed_min: float):
        # if sampling, draw random time between 0 and 1
        if self.cfg.sampling:
            time_passed_min = random.random()
        # if no scheduling specified, use const
        if not self.cfg.end_times_min:
            return self.cfg.init_temp
        # no annealing after max steps
        if not self.cfg.cyclic and time_passed_min >= self.cfg.end_times_min[-1]:
            return self.cfg.anneal_temps[-1]
        # determine current phase and its properties
        if self.cfg.cyclic:
            a_type = self.cfg.anneal_types[0]
            phase_time = time_passed_min % self.cfg.end_times_min[0]
            phase_length = self.cfg.end_times_min[0]
            start_temp = self.cfg.init_temp
            end_temp = self.cfg.anneal_temps[0]
        else:
            a_type, phase_time, phase_length, start_temp, end_temp = None, None, None, None, None
            for idx, cur_end in enumerate(self.cfg.end_times_min):
                if time_passed_min <= cur_end:
                    a_type = self.cfg.anneal_types[idx]
                    start_temp = self.cfg.anneal_temps[idx-1] if idx > 0 else self.cfg.init_temp
                    end_temp = self.cfg.anneal_temps[idx]
                    start_time = self.cfg.end_times_min[idx-1] if idx > 0 else 0
                    phase_time = time_passed_min - start_time
                    phase_length = cur_end - start_time
                    break
        if a_type is None or phase_time is None or start_temp is None or end_temp is None or phase_length is None:
            raise Exception(f"Unknown error occurred, this should never happen")
        # anneal function
        if a_type == AnnealingType.LINEAR or a_type == AnnealingType.LINEAR.value:
            t = phase_time * (end_temp - start_temp) / phase_length + start_temp
        elif a_type == AnnealingType.CONST or a_type == AnnealingType.CONST.value:
            t = start_temp
        elif a_type == AnnealingType.COSINE or a_type == AnnealingType.COSINE.value:
            time_ratio = phase_time / phase_length
            t = end_temp + 0.5 * (start_temp - end_temp) * (1 + math.cos(time_ratio * math.pi))
        elif a_type == AnnealingType.ENHANCED_COS or a_type == AnnealingType.ENHANCED_COS.value:
            time_ratio = phase_time / phase_length
            enhanced_time_ratio = math.log(time_ratio + 1) / (math.log(2))
            t = end_temp + 0.5 * (start_temp - end_temp) * (1 + math.cos(enhanced_time_ratio * math.pi))
        elif a_type == AnnealingType.DOUBLE_COS or a_type == AnnealingType.DOUBLE_COS.value:
            time_ratio = phase_time / phase_length
            scaled_time_ratio = 0.5 * (math.cos((1 + time_ratio) * math.pi) + 1)
            t = end_temp + 0.5 * (start_temp - end_temp) * (1 + math.cos(scaled_time_ratio * math.pi))
        elif a_type == AnnealingType.ENHANCED_DCOS or a_type == AnnealingType.ENHANCED_DCOS.value:
            time_ratio = phase_time / phase_length
            enhanced_time_ratio = math.log(time_ratio + 1) / (math.log(2))
            scaled_time_ratio = 0.5 * (math.cos((1 + enhanced_time_ratio) * math.pi) + 1)
            t = end_temp + 0.5 * (start_temp - end_temp) * (1 + math.cos(scaled_time_ratio * math.pi))
        elif a_type == AnnealingType.LOG or a_type == AnnealingType.LOG.value:
            scale = math.log(phase_length + 1)
            factor = (end_temp - start_temp) * math.log(phase_time + 1)
            t = start_temp + factor / scale
        else:
            raise ValueError(f"Invalid or not implemented anneal type: {a_type}")
        # update lr of optimizer
        if self.optim is not None:
            for param_group in self.optim.param_groups:
                param_group['lr'] = t
        return t
