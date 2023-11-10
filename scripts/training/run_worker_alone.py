import ctypes

import torch.multiprocessing as mp

from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player, survive_on_5x5
from src.game.values import ZeroSumNorm
from src.network.mobilenet_v3 import MobileNetConfig5x5
from src.network.resnet import ResNetConfig3x3
from src.network.utils import ActivationType
from src.network.vision_net import MediumHeadConfig, EquivarianceType
from src.search.backup_func import StandardBackupConfig
from src.search.config import SBRBackupConfig, SpecialExtractConfig, FixedDepthConfig, NashBackupConfig
from src.search.eval_func import NetworkEvalConfig
from src.search.extraction_func import StandardExtractConfig
from src.search.mcts import MCTSConfig
from src.search.sel_func import AlphaZeroDecoupledSelectionConfig
from src.supervised.annealer import TemperatureAnnealingConfig, AnnealingType
from src.trainer.az_worker import WorkerConfig, run_worker
from src.trainer.config import LoggerConfig, AlphaZeroTrainerConfig, UpdaterConfig, EvaluatorConfig, SaverConfig, \
    CollectorConfig, ValidatorConfig
from src.trainer.policy_eval import PolicyEvalConfig, PolicyEvalType


def lone_worker():
    temperature_input = True
    single_temperature = True
    obs_input_temperature = True
    # game_cfg = survive_on_5x5()
    game_cfg = perform_choke_2_player(fully_connected=False, centered=True)
    # game_cfg.all_actions_legal = False
    # game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
    # game_cfg = perform_choke_5x5_4_player(centered=True)
    if obs_input_temperature:
        game_cfg.ec.temperature_input = temperature_input
        game_cfg.ec.single_temperature_input = single_temperature

    eq_type = EquivarianceType.NONE
    # net_cfg = MobileNetConfig5x5(predict_policy=False, predict_game_len=False, game_cfg=game_cfg)
    net_cfg = ResNetConfig3x3(predict_policy=True, predict_game_len=False, eq_type=eq_type, lff_features=False,
                              game_cfg=game_cfg)

    net_cfg.value_head_cfg.final_activation = ActivationType.TANH
    net_cfg.length_head_cfg.final_activation = ActivationType.SIGMOID
    net_cfg.film_temperature_input = (not obs_input_temperature) and temperature_input
    net_cfg.film_cfg = MediumHeadConfig() if net_cfg.film_temperature_input else None

    # backup_cfg = StandardBackupConfig()
    # extract_cfg = StandardExtractConfig()
    batch_size = 128
    eval_func_cfg = NetworkEvalConfig(
        net_cfg=net_cfg,
        max_batch_size=batch_size,
        random_symmetry=False,
        temperature_input=temperature_input,
        single_temperature=single_temperature,
        obs_temperature_input=obs_input_temperature,
    )
    # backup_func_cfg = StandardBackupConfig()
    backup_cfg = SBRBackupConfig(
        num_iterations=200,
    )
    # backup_cfg = NashBackupConfig()
    # extract_func_cfg = StandardExtractConfig()
    extract_cfg = SpecialExtractConfig()
    # search_cfg = MCTSConfig(
    #     eval_func_cfg=NetworkEvalConfig(net_cfg=net_cfg, zero_sum_norm=ZeroSumNorm.NONE),
    #     sel_func_cfg=AlphaZeroDecoupledSelectionConfig(),
    #     backup_func_cfg=backup_cfg,
    #     extract_func_cfg=extract_cfg,
    #     use_hot_start=True,
    #     expansion_depth=0,
    #     optimize_fully_explored=False,
    # )
    search_cfg = FixedDepthConfig(
        eval_func_cfg=eval_func_cfg,
        backup_func_cfg=backup_cfg,
        extract_func_cfg=extract_cfg,
        average_eval=False,
        discount=0.95,
    )
    policy_eval_cfg = PolicyEvalConfig(
        eval_type=PolicyEvalType.TD_0,
        lambda_val=0.5,
    )
    worker_cfg = WorkerConfig(
        search_cfg=search_cfg,
        policy_eval_cfg=policy_eval_cfg,
        # anneal_cfgs=None,
        anneal_cfgs=[TemperatureAnnealingConfig(
            init_temp=1,
            end_times_min=[0.5],
            anneal_temps=[6],
            anneal_types=[AnnealingType.LINEAR],
            cyclic=True
        )],
        search_iterations=2,
        temperature=1,
        max_random_start_steps=0,
        num_gpu=0,
        use_symmetries=True,
        quick_start=False,
        max_game_length=8,
        prevent_draw=True,
        exploration_prob=0.5,
    )
    data_queue = mp.Queue(maxsize=10000)
    net_queue = mp.Queue(maxsize=1000)
    info_queue = mp.Queue(maxsize=10000)
    step_counter = mp.Value('i', 0, lock=True)  # the lock is important if multiple workers are running
    error_counter = mp.Value('i', 0, lock=True)
    episode_counter = mp.Value('i', 0, lock=True)
    stop_flag = mp.Value(ctypes.c_bool, lock=False)
    stop_flag.value = False
    logger_cfg = LoggerConfig(
        testing=True,
        name=None,
        id=None,
        updater_bucket_size=int(1e3),
        worker_episode_bucket_size=int(1e1),
        wandb_mode='offline',
    )
    trainer_cfg = AlphaZeroTrainerConfig(
        num_worker=1,  # IMPORTANT
        individual_gpu=True,  # only used if the use_gpu-flag in worker/updater config is true
        save_state=True,
        save_state_after_seconds=30,
        net_cfg=net_cfg,
        game_cfg=game_cfg,
        updater_cfg=UpdaterConfig(),
        worker_cfg=worker_cfg,
        evaluator_cfg=EvaluatorConfig(),
        logger_cfg=logger_cfg,
        saver_cfg=SaverConfig(),
        collector_cfg=CollectorConfig(),
        validator_cfg=ValidatorConfig(),
        data_qsize=10,
        info_qsize=100,
        updater_in_qsize=100,
        updater_out_qsize=10,
        distributor_out_qsize=10,
        prev_run_dir=None,
        prev_run_idx=None,
        only_generate_buffer=False,
        restrict_cpu=False,  # only works on LINUX
        max_cpu_updater=None,
        max_cpu_worker=None,
        max_cpu_distributor=1,
        max_cpu_logger=1,
        max_cpu_evaluator=3,
        max_cpu_saver=1,
        temperature_input=temperature_input,
        single_sbr_temperature=single_temperature,
        obs_temperature_input=obs_input_temperature,
    )
    # start
    run_worker(
        worker_id=0,
        trainer_cfg=trainer_cfg,
        data_queue=data_queue,
        net_queue=net_queue,
        stop_flag=stop_flag,
        info_queue=info_queue,
        step_counter=step_counter,
        episode_counter=episode_counter,
        gpu_idx=-1,
        cpu_list=None,
        state_dict=None,
        error_counter=error_counter,
        seed=0,
        debug=True,
    )

if __name__ == '__main__':
    lone_worker()
