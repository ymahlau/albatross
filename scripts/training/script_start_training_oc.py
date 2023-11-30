import math
import os
import sys
from pathlib import Path

import hydra
import yaml
import numpy as np

from src.agent.one_shot import LegalRandomAgentConfig, RandomAgentConfig
from src.agent.search_agent import AreaControlSearchAgentConfig
from src.equilibria.logit import SbrMode
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.battlesnake.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player
from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7_4_player_royale
from src.game.overcooked.config import CrampedRoomOvercookedConfig, OneStateCrampedRoomOvercookedConfig, Simple2CrampedRoomOvercookedConfig, Simple3CrampedRoomOvercookedConfig, Simple4CrampedRoomOvercookedConfig, SimpleCrampedRoomOvercookedConfig, TwoStateCrampedRoomOvercookedConfig
from src.game.values import UtilityNorm
from src.misc.const import PHI
from src.misc.serialization import serialize_dataclass
from src.network.fcn import MediumHeadConfig
from src.network.mobile_one import MobileOneConfig3x3
from src.network.mobilenet_v3 import MobileNetConfig3x3, MobileNetConfig5x5
from src.network.resnet import ResNetConfig3x3, ResNetConfig7x7Best, OvercookedResNetConfig5x5
from src.network.utils import ActivationType
from src.network.vision_net import EquivarianceType
from src.search.config import AlphaZeroDecoupledSelectionConfig, InferenceServerEvalConfig, ResponseInferenceServerEvalConfig, StandardBackupConfig, StandardExtractConfig, \
    DecoupledUCTSelectionConfig, LogitBackupConfig, FixedDepthConfig, SpecialExtractConfig, NashBackupConfig, \
    Exp3SelectionConfig, MeanPolicyExtractConfig, Exp3BackupConfig, RegretMatchingSelectionConfig, \
    RegretMatchingBackupConfig, SMOOSConfig, PolicyExtractConfig, EnemyExploitationEvalConfig, \
    EnemyExploitationBackupConfig
from src.search.eval_func import NetworkEvalConfig
from src.search.mcts import MCTSConfig
from src.supervised.annealer import TemperatureAnnealingConfig, AnnealingType
from src.trainer.az_evaluator import EvaluatorConfig
from src.trainer.az_trainer import AlphaZeroTrainerConfig
from src.trainer.az_updater import UpdaterConfig
from src.trainer.az_worker import WorkerConfig
from src.trainer.config import InferenceServerConfig, LoggerConfig, SaverConfig, CollectorConfig
from src.supervised.optim import OptimizerConfig, OptimType
from src.trainer.policy_eval import PolicyEvalType, PolicyEvalConfig
from start_training import main


def start_training_from_structured_configs():
    """
    Main method to start the training using dataclasses specified below
    """

    temperature_input = True
    single_temperature = True
    # game
    # game_cfg = perform_choke_2_player(fully_connected=False, centered=True)
    # game_cfg = CrampedRoomOvercookedConfig(horizon=10)
    # game_cfg = survive_on_7x7_4_player_royale()
    # game_cfg = perform_choke_5x5_4_player(centered=True)
    # game_cfg.all_actions_legal = False
    # game_cfg = OneStateCrampedRoomOvercookedConfig()
    # game_cfg = SimpleCrampedRoomOvercookedConfig()
    game_cfg = Simple4CrampedRoomOvercookedConfig()
    
    game_cfg.temperature_input = temperature_input
    game_cfg.single_temperature_input = single_temperature

    # network
    eq_type = EquivarianceType.NONE
    # net_cfg = ResNetConfig3x3(predict_policy=True, eq_type=eq_type, lff_features=False)
    # net_cfg = MobileNetConfig3x3(predict_policy=True, predict_game_len=False, eq_type=eq_type)
    # net_cfg = MobileOneConfig3x3(predict_policy=True, predict_game_len=False, eq_type=eq_type)
    # net_cfg = MobileNetConfig5x5(predict_policy=True, predict_game_len=False, eq_type=eq_type)
    # net_cfg = ResNetConfig7x7Best()
    net_cfg = OvercookedResNetConfig5x5(predict_policy=True, eq_type=eq_type, lff_features=False)

    # net_cfg = EquivariantMobileNetConfig3x3(predict_game_len=True)
    # search
    # eval_func_cfg = NetworkEvalConfig(zero_sum_norm=ZeroSumNorm.LINEAR)
    batch_size = 3000
    # eval_func_cfg = NetworkEvalConfig(
    #     max_batch_size=batch_size,
    #     random_symmetry=False,
    #     temperature_input=temperature_input,
    #     single_temperature=single_temperature,
    # )
    # enemy_path = Path(__file__).parent.parent.parent / 'trained_models' / 'choke_obs_in.pt'
    # eval_func_cfg = EnemyExploitationEvalConfig(
    #     enemy_net_path=str(enemy_path),
    #     obs_temperature_input=True,
    #     max_batch_size=batch_size,
    # )
    eval_func_cfg = InferenceServerEvalConfig(
        random_symmetry=False,
        temperature_input=temperature_input,
        single_temperature=single_temperature,
        min_clip_value=-math.inf,
        max_clip_value=math.inf,
        policy_prediction=net_cfg.predict_policy,
        utility_norm=UtilityNorm.FULL_COOP,
    )
    # eval_func_cfg = ResponseInferenceServerEvalConfig(
    #     random_symmetry= False,
    #     min_clip_value=-math.inf,
    #     max_clip_value=50,
    #     active_wait_time=0.05,
    #     policy_prediction=True,
    # )
    
    
    # sel_func_cfg = DecoupledUCTSelectionConfig(exp_bonus=1.414)  # 1.4)
    # sel_func_cfg = SampleSelectionConfig(dirichlet_alpha=math.inf, dirichlet_eps=0.25, temperature=1.0)
    # sel_func_cfg = AlphaZeroDecoupledSelectionConfig(exp_bonus=1.414, dirichlet_alpha=0.3, dirichlet_eps=0.25)
    # sel_func_cfg = Exp3SelectionConfig(random_prob=0.1)
    # sel_func_cfg = RegretMatchingSelectionConfig(random_prob=0.1)
    # sel_func_cfg = UncertaintySelectionConfig(informed=True)
    # backup_func_cfg = NashBackupConfig()
    backup_func_cfg = LogitBackupConfig(
        num_iterations=150,
        init_temperatures=[15 for _ in range(game_cfg.num_players)],
        sbr_mode=SbrMode.NAGURNEY,
    )
    # backup_func_cfg = EnemyExploitationBackupConfig(
    #     exploit_temperature=10,
    #     average_eval=False,
    # )
    # backup_func_cfg = RNADBackupConfig(
    #     num_iterations=1000,
    #     reg_factor=0.2,
    # )
    # backup_func_cfg = UncertaintyBackupConfig(lr=0.3, temperature=3, informed=True, use_children=False)
    # backup_func_cfg = StandardBackupConfig()
    # backup_func_cfg = Exp3BackupConfig()
    # backup_func_cfg = RegretMatchingBackupConfig(avg_backup=True)
    extraction_func_cfg = SpecialExtractConfig(
        utility_norm=UtilityNorm.FULL_COOP,
        min_clip_value=-math.inf,
        max_clip_value=30,
    )
    # extraction_func_cfg = StandardExtractConfig()
    # extraction_func_cfg = MeanPolicyExtractConfig()
    # extraction_func_cfg = PolicyExtractConfig()
    # search_cfg = MCTSConfig(
    #     eval_func_cfg=eval_func_cfg,
    #     sel_func_cfg=sel_func_cfg,
    #     backup_func_cfg=backup_func_cfg,
    #     extract_func_cfg=extraction_func_cfg,
    #     expansion_depth=0,
    #     use_hot_start=False,
    #     optimize_fully_explored=False,
    #     discount=0.95,
    # )
    search_cfg = FixedDepthConfig(
        eval_func_cfg=eval_func_cfg,
        backup_func_cfg=backup_func_cfg,
        extract_func_cfg=extraction_func_cfg,
        average_eval=False,
        discount=0.9,
    )
    # search_cfg = SMOOSConfig(
    #     eval_func_cfg=eval_func_cfg,
    #     exp_factor=0.15,
    #     enhanced_regret=True,
    #     relief_update=True,
    #     lambda_val=0,
    #     use_hot_start=True,
    #     informed_exp=True,
    # )
    # trainer setup
    policy_eval_cfg = PolicyEvalConfig(
        eval_type=PolicyEvalType.TD_0,
        lambda_val=0.5,
    )
    worker_cfg = WorkerConfig(
        search_cfg=search_cfg,
        policy_eval_cfg=policy_eval_cfg,
        
        temp_scaling_cfgs=(
            TemperatureAnnealingConfig(
                init_temp=5,
                end_times_min=[20, 40],
                anneal_temps=[5, 0],
                anneal_types=[AnnealingType.CONST, AnnealingType.LINEAR],
                cyclic=False,
                sampling=False,
            ),
            TemperatureAnnealingConfig(
                init_temp=10,
                end_times_min=[1],
                anneal_temps=[10],
                anneal_types=[AnnealingType.CONST],
                cyclic=True,
                sampling=False,
            ),
        ),
        # anneal_cfgs=None,
        anneal_cfgs=[TemperatureAnnealingConfig(
            init_temp=0,
            end_times_min=[1],
            anneal_temps=[1],
            anneal_types=[AnnealingType.COSINE],
            cyclic=True,
            sampling=True,
		)],
        
		# anneal_cfgs=[TemperatureAnnealingConfig(
        #     init_temp=1,
        #     end_times_min=[1],
        #     anneal_temps=[10],
        #     anneal_types=[AnnealingType.COSINE],
        #     cyclic=True,
        #     sampling=True,
        # ) for _ in range(game_cfg.num_players)],
        search_iterations=1,
        temperature=1,
        max_random_start_steps=0,
        use_symmetries=True,
        quick_start=False,
        max_game_length=8,
        prevent_draw=False,
        exploration_prob=0.5,
    )
    evaluator_cfg = EvaluatorConfig(
        eval_rate_sec=20,
        num_episodes=[100, 2],
        enemy_iterations=100,
        enemy_cfgs=[
            RandomAgentConfig()
            # LegalRandomAgentConfig(),
            # AreaControlSearchAgentConfig(),
            # CopyCatSearchAgentConfig()
        ],
        prevent_draw=False,
        self_play=True,
        switch_pos=True,
    )
    optim_cfg = OptimizerConfig(
        optim_type=OptimType.ADAM_W,
        anneal_cfg=TemperatureAnnealingConfig(
            init_temp=0,
            end_times_min=[4, 20, 24, 40],
            anneal_temps=[1e-3, 1e-5, 1e-3, 1e-6],
            anneal_types=[AnnealingType.LINEAR, AnnealingType.COSINE, AnnealingType.LINEAR, AnnealingType.COSINE],
        ),
        weight_decay=1e-4,
        beta1=0.9,
        beta2=0.99,
    )
    buffer_size = int(1e4)
    collector_cfg = CollectorConfig(
        buffer_size=buffer_size,
        quick_start_buffer_path=None,
        start_wait_n_samples=buffer_size,  # int(5e2),
        # quick_start_buffer_path=Path(__file__).parent.parent.parent / 'buffer' / 'choke_1e3.pt',
        log_every_sec=20,
    )
    updater_cfg = UpdaterConfig(
        updates_until_distribution=5,
        optim_cfg=optim_cfg,
        use_gpu=True,
        utility_loss=UtilityNorm.NONE,
        mse_policy_loss=False,
        policy_loss_factor=1,
        value_reg_loss_factor=0,
        utility_loss_factor=0,
        gradient_max_norm=100,
    )
    logger_cfg = LoggerConfig(
        project_name="test",
        buffer_gen=False,
        name=None,
        id=0,
        updater_bucket_size=100,
        worker_episode_bucket_size=5,
        wandb_mode='offline',
    )
    saver_cfg = SaverConfig(
        save_interval_sec=10,
    )
    inf_cfg = InferenceServerConfig(
        use_gpu=True,
    )
    trainer_cfg = AlphaZeroTrainerConfig(
        num_worker=30,  # IMPORTANT
        num_inference_server=1,
        save_state=False,
        save_state_after_seconds=30,
        net_cfg=net_cfg,
        game_cfg=game_cfg,
        updater_cfg=updater_cfg,
        worker_cfg=worker_cfg,
        evaluator_cfg=evaluator_cfg,
        logger_cfg=logger_cfg,
        saver_cfg=saver_cfg,
        collector_cfg=collector_cfg,
        inf_cfg=inf_cfg,
        max_batch_size=batch_size,
        max_eval_per_worker=batch_size*2,
        data_qsize=10,
        info_qsize=100,
        updater_in_qsize=100,
        updater_out_qsize=10,
        distributor_out_qsize=10,
        prev_run_dir=None,
        prev_run_idx=None,
        only_generate_buffer=False,
        restrict_cpu=True,  # only works on LINUX
        max_cpu_updater=1,
        max_cpu_worker=10,
        max_cpu_evaluator=1,
        max_cpu_log_dist_save_collect=1,
        max_cpu_inference_server=1,
        temperature_input=temperature_input,
        single_sbr_temperature=single_temperature,
        compile_model=False,
        compile_mode='max-autotune',
        merge_inference_update_gpu=True,
        proxy_net_path=None,
    )
    # initialize yaml file and hydra
    print(os.getcwd())
    exported_dict = serialize_dataclass(trainer_cfg)
    yaml_str = yaml.dump(exported_dict)
    # yaml_str = OmegaConf.to_yaml(trainer_cfg)
    yaml_str += 'hydra:\n  run:\n    dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_' + f'{logger_cfg.name}'
    config_name = 'debug_config'
    config_dir = Path(__file__).parent.parent.parent / 'config'
    cur_config_file = config_dir / f'{config_name}.yaml'
    with open(cur_config_file, 'w') as f:
        f.write(yaml_str)

    sys.argv = [
        'cmd',  # this is ignored
        'hydra.job.chdir=True',
        # ... other dynamically added parameters
    ]
    # hydra.main(version_base=None)(main)()
    hydra.main(config_path=str(config_dir), config_name=config_name, version_base=None)(main)()


if __name__ == '__main__':
    start_training_from_structured_configs()
