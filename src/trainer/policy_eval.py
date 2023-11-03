from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

class PolicyEvalType(Enum):
    MC = 'MC'
    TD_0 = 'TD_0'
    TD_LAMBDA = 'TD_LAMBDA'


@dataclass
class PolicyEvalConfig:
    eval_type: PolicyEvalType = PolicyEvalType.MC
    lambda_val: float = 0.95
# cs.store(name="default", group='policy_eval_cfg', node=PolicyEvalConfig)


def policy_eval_config_from_structured(cfg) -> PolicyEvalConfig:
    return PolicyEvalConfig(**cfg)
