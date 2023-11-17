from dataclasses import dataclass, field
from typing import Optional

from src.agent import AgentConfig
from src.agent.one_shot import RandomAgentConfig
from src.agent.search_agent import AreaControlSearchAgentConfig
from src.game.game import GameConfig
from src.game.values import UtilityNorm
from src.network import NetworkConfig
from src.search import SearchConfig
from src.supervised.annealer import TemperatureAnnealingConfig
from src.supervised.optim import OptimizerConfig
from src.trainer.policy_eval import PolicyEvalConfig


@dataclass
class EvaluatorConfig:
    eval_rate_sec: float = 60
    num_episodes: int = 100
    temperature: float = 1.0
    enemy_iterations: int = 200
    enemy_cfgs: list[AgentConfig] = field(default_factory=lambda: [
        RandomAgentConfig(),
        AreaControlSearchAgentConfig(),
    ])
    prevent_draw: bool = True


@dataclass
class LoggerConfig:
    project_name: str = 'battlesnake_rl_test'
    buffer_gen: bool = False
    name: Optional[str] = None
    id: Optional[int] = None  # random seed
    updater_bucket_size: int = 50
    worker_episode_bucket_size: int = 100
    wandb_mode: str = 'online'


@dataclass(kw_only=True)
class UpdaterConfig:
    # optimizer
    optim_cfg: OptimizerConfig
    updates_until_distribution: int = 5
    # cuda
    use_gpu: bool = False
    gradient_max_norm: Optional[float] = 1.0
    utility_loss: UtilityNorm = UtilityNorm.NONE
    mse_policy_loss: bool = False


@dataclass
class CollectorConfig:
    buffer_size: int = int(1e3)
    batch_size: int = 128
    quick_start_buffer_path: Optional[str] = None
    start_wait_n_samples: int = int(1e3)
    log_every_sec: int = 60
    validation_percentage: float = 0  # percentage of samples directed to validation buffer


@dataclass
class WorkerConfig:
    search_cfg: SearchConfig
    policy_eval_cfg: PolicyEvalConfig
    anneal_cfgs: Optional[list[TemperatureAnnealingConfig]] = None
    search_iterations: int = 128
    temperature: float = 1
    max_random_start_steps: int = 0
    use_symmetries: bool = True
    quick_start: bool = True  # use uniform random policy at beginning
    max_game_length: float = 25  # only used for game length prediction
    prevent_draw: bool = False
    exploration_prob: float = 0.5


@dataclass
class SaverConfig:
    save_interval_sec: float = 6000


@dataclass
class InferenceServerConfig:
    use_gpu: bool = True
    statistics_every_sec: int = 60


@dataclass
class AlphaZeroTrainerConfig:
    game_cfg: GameConfig

    updater_cfg: UpdaterConfig
    collector_cfg: CollectorConfig
    worker_cfg: WorkerConfig
    evaluator_cfg: EvaluatorConfig
    logger_cfg: LoggerConfig
    saver_cfg: SaverConfig
    inf_cfg: InferenceServerConfig

    max_batch_size: int
    max_eval_per_worker: int  # maximum number of observation needing evaluation in a single process at a time
    net_cfg: NetworkConfig

    num_worker: int = 1
    num_inference_server: int = 1
    data_qsize: int = 10
    info_qsize: int = 100
    updater_out_qsize: int = 10
    updater_in_qsize: int = 100
    validator_data_qsize: int = 100
    distributor_out_qsize: int = 10
    prev_run_dir: Optional[str] = None  # path to hydra log dir of prev run, this means you want to continue training
    prev_run_idx: Optional[int] = None  # model index to continue training from
    init_new_network_params: bool = False  # only used if previous run dir is continued
    only_generate_buffer: bool = False
    restrict_cpu: bool = False  # only works on LINUX
    max_cpu_inference_server: Optional[int] = None
    max_cpu_updater: Optional[int] = None
    max_cpu_worker: Optional[int] = None
    max_cpu_log_dist_save_collect: Optional[int] = None
    max_cpu_evaluator: Optional[int] = None
    save_state: bool = False
    temperature_input: bool = False
    single_sbr_temperature: bool = True
    save_state_after_seconds: int = 36000
    compile_model: bool = False
    compile_mode: str = 'reduce-overhead'  # Can also be max_autotune (currently does not work on rtx3090
