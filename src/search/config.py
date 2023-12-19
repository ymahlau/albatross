import math
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.equilibria.logit import SbrMode
from src.game.values import UtilityNorm
from src.network import NetworkConfig


# Backup Functions
###############################################################

@dataclass
class BackupFuncConfig(ABC):
    pass

@dataclass
class StandardBackupConfig(BackupFuncConfig):
    pass

@dataclass
class MaxMinBackupConfig(BackupFuncConfig):
    factor: float = 4.0

@dataclass
class MaxAvgBackupConfig(BackupFuncConfig):
    factor: float = 4.0

@dataclass
class NashBackupConfig(BackupFuncConfig):
    use_cpp: bool = True

@dataclass
class LogitBackupConfig(BackupFuncConfig):
    num_iterations: int = 300
    epsilon: float = 0
    hp_0: Optional[float] = None  # hyperparameters for logit equilibrium computation. Meaning depends on sbr mode
    hp_1: Optional[float] = None
    sbr_mode: SbrMode = SbrMode.MSA
    use_cpp: bool = True
    init_random: bool = True
    init_temperatures: Optional[list[float]] = None  # one temperature for every player

@dataclass
class Exp3BackupConfig(BackupFuncConfig):
    avg_backup: bool = False

@dataclass
class RegretMatchingBackupConfig(BackupFuncConfig):
    avg_backup: bool = False

@dataclass(kw_only=True)
class EnemyExploitationBackupConfig(BackupFuncConfig):
    init_temperatures: Optional[list[float]] = None  # length num_player
    exploit_temperature: float = 10
    average_eval: bool = False

@dataclass
class QNEBackupConfig(BackupFuncConfig):
    leader: int = 0
    num_iterations: int = 1000
    init_temperature: Optional[float] = None

@dataclass
class QSEBackupConfig(BackupFuncConfig):
    leader: int = 0
    num_iterations: int = 9
    grid_size: int = 200
    init_temperature: Optional[float] = None

@dataclass
class SBRLEBackupConfig(BackupFuncConfig):
    response_temperature: float
    init_temperature: Optional[float] = None
    leader: int = 0

@dataclass
class ExploitOtherBackupConfig(BackupFuncConfig):
    backup_cfg: BackupFuncConfig
    worst_case: bool
    leader: int = 0

@dataclass
class NashVsSBRBackupConfig(BackupFuncConfig):
    init_temperature: Optional[float] = None
    leader: int = 0


# Eval Function
##################################################

@dataclass
class EvalFuncConfig(ABC):
    utility_norm: UtilityNorm = UtilityNorm.NONE

@dataclass
class AreaControlEvalConfig(EvalFuncConfig):
    health_threshold: float = 1.0

@dataclass
class CopyCatEvalConfig(EvalFuncConfig):
    health_threshold: float = 1.0
    hazard_weight: float = 1.0
    tile_weight: float = 1.0
    food_weight: float = 1.0
    food_hazard_weight: float = 1.0
    area_control_weight: float = 2.0
    health_weight: float = 10.0

@dataclass
class NetworkEvalConfig(EvalFuncConfig):
    net_cfg: Optional[NetworkConfig] = None
    init_temperatures: Optional[list[float]] = None
    temperature_input: bool = False
    single_temperature: bool = True
    max_batch_size: int = 128
    random_symmetry: bool = False
    no_grad: bool = True
    min_clip_value: float = -1
    max_clip_value: float = 20


@dataclass
class DummyEvalConfig(EvalFuncConfig):
    pass


@dataclass(kw_only=True)
class EnemyExploitationEvalConfig(EvalFuncConfig):
    enemy_net_path: str
    init_temperatures: Optional[list[float]] = None
    net_cfg: Optional[NetworkConfig] = None
    max_batch_size: int = 128
    obs_temperature_input: bool = False
    precision: Optional[str] = None


@dataclass
class RandomRolloutEvalConfig(EvalFuncConfig):
    num_rollouts: int = 1


@dataclass
class InferenceServerEvalConfig(EvalFuncConfig):
    init_temperatures: Optional[list[float]] = None
    temperature_input: bool = False
    single_temperature: bool = True
    random_symmetry: bool = False
    min_clip_value: float = -math.inf
    max_clip_value: float = 50
    active_wait_time: float = 0
    policy_prediction: bool = True
    
@dataclass
class ResponseInferenceServerEvalConfig(EvalFuncConfig):
    random_symmetry: bool = False
    min_clip_value: float = -math.inf
    max_clip_value: float = 50
    active_wait_time: float = 0.05
    policy_prediction: bool = True


# Extraction Function
########################################################
@dataclass
class ExtractFuncConfig(ABC):
    pass


@dataclass
class StandardExtractConfig(ExtractFuncConfig):
    pass


@dataclass
class SpecialExtractConfig(ExtractFuncConfig):
    utility_norm: UtilityNorm = UtilityNorm.NONE
    min_clip_value: float = -math.inf
    max_clip_value: float = 50


@dataclass
class MeanPolicyExtractConfig(ExtractFuncConfig):
    pass


@dataclass
class PolicyExtractConfig(ExtractFuncConfig):
    pass


# Selection Function
#########################################################
@dataclass
class SelectionFuncConfig(ABC):
    pass


@dataclass
class DecoupledUCTSelectionConfig(SelectionFuncConfig):
    exp_bonus: float = 1.414


@dataclass
class AlphaZeroDecoupledSelectionConfig(SelectionFuncConfig):
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25
    exp_bonus: float = 2.0


@dataclass
class SampleSelectionConfig(SelectionFuncConfig):
    dirichlet_alpha: float = math.inf
    dirichlet_eps: float = 0.25
    temperature: float = 1.0


@dataclass
class Exp3SelectionConfig(SelectionFuncConfig):
    altered: bool = False
    random_prob: float = 0.1


@dataclass
class RegretMatchingSelectionConfig(SelectionFuncConfig):
    random_prob: float = 0.1
    informed_exp: bool = False


# Search
#########################################################

@dataclass
class SearchConfig(ABC):
    eval_func_cfg: EvalFuncConfig
    extract_func_cfg: ExtractFuncConfig
    discount: float = 0.99


@dataclass(kw_only=True)
class MCTSConfig(SearchConfig):
    sel_func_cfg: SelectionFuncConfig
    backup_func_cfg: BackupFuncConfig
    expansion_depth: int = 0  # leaf node is expanded by BFS of this depth (starting at leaf node)
    use_hot_start: bool = True
    optimize_fully_explored: bool = False  # compute statistics if a subtree is fully explored


@dataclass(kw_only=True)
class FixedDepthConfig(SearchConfig):
    backup_func_cfg: BackupFuncConfig
    average_eval: bool = False  # value of every node is average of backup and heuristic eval


@dataclass(kw_only=True)
class IterativeDeepeningConfig(FixedDepthConfig):
    backup_func_cfg: BackupFuncConfig


@dataclass(kw_only=True)
class SMOOSConfig(SearchConfig):
    eval_func_cfg: EvalFuncConfig
    extract_func_cfg: ExtractFuncConfig = field(default_factory=lambda: MeanPolicyExtractConfig())
    use_hot_start: bool = True
    exp_factor: float = 0.2
    informed_exp: bool = False  # multiply net probs with exploration
    exp_decay: bool = False  # logarithmic decay of exploration
    uct_explore: bool = False  # count-based exploration like in UCT
    enhanced_regret: bool = False  # subtractive regret calculation like regret matching
    regret_decay: float = 1.0
    lambda_val: float = 1.0
    relief_update: bool = True  # also calculate negative regret (relief) for chosen action
