import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.equilibria.logit import SbrMode
from src.game.values import ZeroSumNorm
from src.network import NetworkConfig


# Backup Functions
###############################################################
class BackupFuncType(Enum):
    STANDARD_BACKUP = 'STANDARD_BACKUP'
    MAXMIN_BACKUP = 'MAXMIN_BACKUP'
    MAXAVG_BACKUP = 'MAXAVG_BACKUP'
    NASH_BACKUP = 'NASH_BACKUP'
    SBR_BACKUP = 'SBR_BACKUP'
    RNAD_BACKUP = 'RNAD_BACKUP'
    EXP3 = 'EXP3'
    REGRET_MATCHING = 'REGRET_MATCHING'
    UNCERTAINTY = 'UNCERTAINTY'
    ENEMY_EXPLOIT = 'ENEMY_EXPLOIT'
    QSE = 'QSE'
    QNE = 'QNE'
    SBRLE = 'SBRLE'
    EXPLOIT_OTHER = 'EXPLOIT_OTHER'
    NASH_VS_SBR = 'NASH_VS_SBR'

@dataclass
class BackupFuncConfig:
    backup_type: BackupFuncType = MISSING

@dataclass
class StandardBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.STANDARD_BACKUP)

@dataclass
class MaxMinBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.MAXMIN_BACKUP)
    factor: float = 4.0

@dataclass
class MaxAvgBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.MAXAVG_BACKUP)
    factor: float = 4.0

@dataclass
class NashBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.NASH_BACKUP)
    use_cpp: bool = True

@dataclass
class SBRBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.SBR_BACKUP)
    num_iterations: int = 300
    epsilon: float = 0
    moving_average_factor: float = 0.9
    sbr_mode: SbrMode = SbrMode.CUMULATIVE_SUM
    use_cpp: bool = True
    init_random: bool = True
    init_temperatures: Optional[list[float]] = None  # one temperature for every player

@dataclass
class RNADBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.RNAD_BACKUP)
    num_iterations: int = 1000
    reg_factor: float = 0.2

@dataclass
class Exp3BackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.EXP3)
    avg_backup: bool = False

@dataclass
class RegretMatchingBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.REGRET_MATCHING)
    avg_backup: bool = False

@dataclass
class UncertaintyBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.UNCERTAINTY)
    lr: float = 0.1
    temperature: float = 2
    informed: bool = False
    use_children: bool = True

@dataclass
class EnemyExploitationBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.ENEMY_EXPLOIT)
    init_temperatures: Optional[list[float]] = None  # length num_player
    enemy_net_path: str = MISSING
    recompute_policy: bool = False
    exploit_temperature: float = 10
    average_eval: bool = False
    # sbr
    num_iterations: int = 300
    epsilon: float = 0
    moving_average_factor: float = 0.9
    sbr_mode: SbrMode = SbrMode.CUMULATIVE_SUM
    use_cpp: bool = True
    init_random: bool = True

@dataclass
class QNEBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.QNE)
    leader: int = 0
    num_iterations: int = 1000
    init_temperature: Optional[float] = None

@dataclass
class QSEBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.QSE)
    leader: int = 0
    num_iterations: int = 9
    grid_size: int = 200
    init_temperature: Optional[float] = None

@dataclass
class SBRLEBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.SBRLE)
    init_temperature: Optional[float] = None
    response_temperature: float = MISSING
    leader: int = 0

@dataclass
class ExploitOtherBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.EXPLOIT_OTHER)
    backup_cfg: BackupFuncConfig = MISSING
    leader: int = 0
    worst_case: bool = MISSING

@dataclass
class NashVsSBRBackupConfig(BackupFuncConfig):
    backup_type: BackupFuncType = field(default=BackupFuncType.NASH_VS_SBR)
    init_temperature: Optional[float] = None
    leader: int = 0


# Eval Function
##################################################
class EvalFuncType(Enum):
    AREA_CONTROL_EVAL = 'AREA_CONTROL_EVAL'
    COPY_CAT_EVAL = 'COPY_CAT_EVAL'
    NETWORK_EVAL = 'NETWORK_EVAL'
    DUMMY = 'DUMMY'
    OSHI_ZUMO = 'OSHI_ZUMO'
    ENEMY_EXPLOIT = 'ENEMY_EXPLOIT'
    ROLLOUT = 'ROLLOUT'
    POTENTIAL = 'POTENTIAL'

@dataclass
class EvalFuncConfig:
    zero_sum_norm: ZeroSumNorm = ZeroSumNorm.NONE
    eval_func_type: EvalFuncType = MISSING

@dataclass
class AreaControlEvalConfig(EvalFuncConfig):
    zero_sum_norm: ZeroSumNorm = field(default=ZeroSumNorm.NONE)
    eval_func_type: EvalFuncType = field(default=EvalFuncType.AREA_CONTROL_EVAL)
    health_threshold: float = 1.0

@dataclass
class CopyCatEvalConfig(EvalFuncConfig):
    eval_func_type: EvalFuncType = field(default=EvalFuncType.COPY_CAT_EVAL)
    health_threshold: float = 1.0
    hazard_weight: float = 1.0
    tile_weight: float = 1.0
    food_weight: float = 1.0
    food_hazard_weight: float = 1.0
    area_control_weight: float = 2.0
    health_weight: float = 10.0

@dataclass
class NetworkEvalConfig(EvalFuncConfig):
    eval_func_type: EvalFuncType = field(default=EvalFuncType.NETWORK_EVAL)
    net_cfg: Optional[NetworkConfig] = None
    init_temperatures: Optional[list[float]] = None
    temperature_input: bool = False
    single_temperature: bool = True
    obs_temperature_input: bool = False
    max_batch_size: int = 128
    random_symmetry: bool = False
    precision: Optional[str] = None
    no_grad: bool = True
    min_clip_value: float = -1
    max_clip_value: float = 20

@dataclass
class DummyEvalConfig(EvalFuncConfig):
    eval_func_type: EvalFuncType = field(default=EvalFuncType.DUMMY)

@dataclass
class EnemyExploitationEvalConfig(EvalFuncConfig):
    eval_func_type: EvalFuncType = field(default=EvalFuncType.ENEMY_EXPLOIT)
    init_temperatures: Optional[list[float]] = None
    net_cfg: Optional[NetworkConfig] = None
    max_batch_size: int = 128
    enemy_net_path: str = MISSING
    obs_temperature_input: bool = False
    precision: Optional[str] = None

@dataclass
class OshiZumoEvalConfig(EvalFuncConfig):
    eval_func_type: EvalFuncType = field(default=EvalFuncType.OSHI_ZUMO)

@dataclass
class RandomRolloutEvalConfig(EvalFuncConfig):
    eval_func_type: EvalFuncType = field(default=EvalFuncType.ROLLOUT)
    num_rollouts: int = 1

@dataclass
class OvercookedPotentialEvalConfig(EvalFuncConfig):
    eval_func_type: EvalFuncType = field(default=EvalFuncType.POTENTIAL)
    overcooked_layout: str = MISSING

# Extraction Function
########################################################
class ExtractFuncType(Enum):
    STANDARD_EXTRACT = 'STANDARD_EXTRACT'
    SPECIAL_EXTRACT = 'SPECIAL_EXTRACT'
    MEAN_POLICY = 'MEAN_POLICY'
    POLICY = 'POLICY'

@dataclass
class ExtractFuncConfig:
    extract_func_type: ExtractFuncType = MISSING

@dataclass
class StandardExtractConfig(ExtractFuncConfig):
    extract_func_type: ExtractFuncType = field(default=ExtractFuncType.STANDARD_EXTRACT)

@dataclass
class SpecialExtractConfig(ExtractFuncConfig):
    extract_func_type: ExtractFuncType = field(default=ExtractFuncType.SPECIAL_EXTRACT)
    zero_sum_norm: bool = False

@dataclass
class MeanPolicyExtractConfig(ExtractFuncConfig):
    extract_func_type: ExtractFuncType = field(default=ExtractFuncType.MEAN_POLICY)

@dataclass
class PolicyExtractConfig(ExtractFuncConfig):
    extract_func_type: ExtractFuncType = field(default=ExtractFuncType.POLICY)

# Selection Function
#########################################################
class SelectionFuncType(Enum):
    DECOUPLED_UCT = 'DECOUPLED_UCT'
    AZ_DECOUPLED = 'AZ_DECOUPLED'
    SAMPLE = 'SAMPLE'
    EXP3 = 'EXP3'
    REGRET_MATCHING = 'REGRET_MATCHING'
    UNCERTAINTY = 'UNCERTAINTY'

@dataclass
class SelectionFuncConfig:
    sel_func_type: SelectionFuncType = MISSING

@dataclass
class DecoupledUCTSelectionConfig(SelectionFuncConfig):
    sel_func_type: SelectionFuncType = field(default=SelectionFuncType.DECOUPLED_UCT)
    exp_bonus: float = 1.414

@dataclass
class AlphaZeroDecoupledSelectionConfig(SelectionFuncConfig):
    sel_func_type: SelectionFuncType = field(default=SelectionFuncType.AZ_DECOUPLED)
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25
    exp_bonus: float = 2.0

@dataclass
class SampleSelectionConfig(SelectionFuncConfig):
    sel_func_type: SelectionFuncType = field(default=SelectionFuncType.SAMPLE)
    dirichlet_alpha: float = math.inf
    dirichlet_eps: float = 0.25
    temperature: float = 1.0

@dataclass
class Exp3SelectionConfig(SelectionFuncConfig):
    sel_func_type: SelectionFuncType = field(default=SelectionFuncType.EXP3)
    altered: bool = False
    random_prob: float = 0.1

@dataclass
class RegretMatchingSelectionConfig(SelectionFuncConfig):
    sel_func_type: SelectionFuncType = field(default=SelectionFuncType.REGRET_MATCHING)
    random_prob: float = 0.1
    informed_exp: bool = False

@dataclass
class UncertaintySelectionConfig(SelectionFuncConfig):
    sel_func_type: SelectionFuncType = field(default=SelectionFuncType.UNCERTAINTY)
    informed: bool = False


# Search
#########################################################
class SearchType(Enum):
    MCTS = 'MCTS'
    ITERATIVE_DEEPENING = 'ITERATIVE_DEEPENING'
    FIXED_DEPTH = 'FIXED_DEPTH'
    SMOOS = 'SMOOS'

@dataclass
class SearchConfig:
    search_type: SearchType = MISSING
    eval_func_cfg: EvalFuncConfig = MISSING
    extract_func_cfg: ExtractFuncConfig = MISSING
    discount: float = 0.99

@dataclass
class MCTSConfig(SearchConfig):
    search_type: SearchType = field(default=SearchType.MCTS)
    sel_func_cfg: SelectionFuncConfig = MISSING
    backup_func_cfg: BackupFuncConfig = MISSING
    expansion_depth: int = 0  # leaf node is expanded by BFS of this depth (starting at leaf node)
    use_hot_start: bool = True
    optimize_fully_explored: bool = False  # compute statistics if a subtree is fully explored

@dataclass
class FixedDepthConfig(SearchConfig):
    search_type: SearchType = field(default=SearchType.FIXED_DEPTH)
    average_eval: bool = False  # value of every node is average of backup and heuristic eval
    backup_func_cfg: BackupFuncConfig = MISSING

@dataclass
class IterativeDeepeningConfig(FixedDepthConfig):
    search_type: SearchType = field(default=SearchType.ITERATIVE_DEEPENING)
    backup_func_cfg: BackupFuncConfig = MISSING

@dataclass
class SMOOSConfig(SearchConfig):
    search_type: SearchType = field(default=SearchType.SMOOS)
    eval_func_cfg: EvalFuncConfig = MISSING
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
