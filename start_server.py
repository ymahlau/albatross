import math
from pathlib import Path

import torch
import multiprocessing as mp

from src.agent.one_shot import NetworkAgentConfig
from src.agent.search_agent import LookaheadAgentConfig, SearchAgentConfig, DoubleSearchAgentConfig
from src.frontend.frontend_agent import FrontendAgent
from src.frontend.server import run_server
from src.frontend.utils import SupportedMode
from src.game.values import ZeroSumNorm
from src.search.config import MCTSConfig, NetworkEvalConfig, AlphaZeroDecoupledSelectionConfig, StandardBackupConfig, \
    StandardExtractConfig, IterativeDeepeningConfig, SBRBackupConfig, SpecialExtractConfig, FixedDepthConfig


def start_duel_agent():
    """
    Command for running battlesnake cli with this agent and remote server
    battlesnake.exe play -u http://127.0.0.1:8003 -u http://127.0.0.1:8004
        -u http://130.75.31.113:8003 -u http://130.75.31.113:8003 --browser
    """
    agent_cfg = SearchAgentConfig(
        search_cfg=MCTSConfig(
            sel_func_cfg=AlphaZeroDecoupledSelectionConfig(exp_bonus=1.414, dirichlet_eps=0),
            backup_func_cfg=StandardBackupConfig(),
            extract_func_cfg=StandardExtractConfig(),
            eval_func_cfg=NetworkEvalConfig(
                max_batch_size=1000,
                precision='32-true',
                no_grad=False,
                zero_sum_norm=ZeroSumNorm.LINEAR,
            ),
            discount=0.99,
            expansion_depth=0,
            use_hot_start=True,
            optimize_fully_explored=False,
        )
    )
    agent = FrontendAgent(
        agent_cfg=agent_cfg,
        net_path=Path(__file__).parent.parent / 'a_models' / 'duel_finished.pt',
        device=torch.device('cuda:0'),
        name='Deutschlange',
        mode=SupportedMode.DUELS,
        expected_latency_sec=0.1,
        draw_prevention_limit=25,
        compile_model=True,
    )
    port = 8010
    run_server(
        agent=agent,
        port=port
    )

def start_constrictor_agent():
    """
    Command for running battlesnake cli with this agent and remote server
    battlesnake.exe play -u http://127.0.0.1:8003 -u http://127.0.0.1:8004
        -u http://130.75.31.113:8003 -u http://130.75.31.113:8003 --browser
    """
    agent_cfg = SearchAgentConfig(
        search_cfg=MCTSConfig(
            sel_func_cfg=AlphaZeroDecoupledSelectionConfig(exp_bonus=1.414, dirichlet_eps=0),
            backup_func_cfg=StandardBackupConfig(),
            extract_func_cfg=StandardExtractConfig(),
            eval_func_cfg=NetworkEvalConfig(
                max_batch_size=1000,
                precision='32-true',
                no_grad=True,
                zero_sum_norm=ZeroSumNorm.NONE,
            ),
            discount=0.99,
            expansion_depth=0,
            use_hot_start=True,
            optimize_fully_explored=False,
        )
    )
    # agent_cfg = SearchAgentConfig(
    #     search_cfg=FixedDepthConfig(
    #         eval_func_cfg=NetworkEvalConfig(
    #             max_batch_size=100,
    #             precision='32-true',
    #             no_grad=False,
    #             zero_sum_norm=ZeroSumNorm.LINEAR,
    #         ),
    #         backup_func_cfg=SBRBackupConfig(num_iterations=100, init_temperatures=[10, 10, 10, 10]),
    #         extract_func_cfg=SpecialExtractConfig(),
    #         discount=0.99,
    #     )
    # )
    # agent_cfg = NetworkAgentConfig()

    agent = FrontendAgent(
        agent_cfg=agent_cfg,
        net_path=Path(__file__).parent.parent / 'a_models' / '4d11_smoos_pooled.pt',
        device=torch.device('cuda:0'),
        name='Deutschlange',
        mode=SupportedMode.CONSTRICTOR,
        expected_latency_sec=0.1,
        draw_prevention_limit=0,
        max_iterations=1000,
        compile_model=False,
    )
    port = 8011
    run_server(
        agent=agent,
        port=port
    )

def start_standard_agent():
    # agent_cfg = SearchAgentConfig(
    #     search_cfg=MCTSConfig(
    #         sel_func_cfg=AlphaZeroDecoupledSelectionConfig(exp_bonus=1.414, dirichlet_eps=0),
    #         backup_func_cfg=StandardBackupConfig(),
    #         extract_func_cfg=StandardExtractConfig(),
    #         eval_func_cfg=NetworkEvalConfig(
    #             max_batch_size=1000,
    #             precision='32-true',
    #             no_grad=True,
    #             zero_sum_norm=ZeroSumNorm.NONE,
    #         ),
    #         discount=0.99,
    #         expansion_depth=0,
    #         use_hot_start=True,
    #         optimize_fully_explored=False,
    #     )
    # )
    agent_cfg = DoubleSearchAgentConfig(
        search_cfg=MCTSConfig(
            sel_func_cfg=AlphaZeroDecoupledSelectionConfig(exp_bonus=1.414, dirichlet_eps=0),
            backup_func_cfg=StandardBackupConfig(),
            extract_func_cfg=StandardExtractConfig(),
            eval_func_cfg=NetworkEvalConfig(
                max_batch_size=1000,
                precision='32-true',
                no_grad=True,
                zero_sum_norm=ZeroSumNorm.NONE,
            ),
            discount=0.99,
            expansion_depth=0,
            use_hot_start=True,
            optimize_fully_explored=False,
        ),
        search_cfg_2=MCTSConfig(
            sel_func_cfg=AlphaZeroDecoupledSelectionConfig(exp_bonus=1.414, dirichlet_eps=0),
            backup_func_cfg=StandardBackupConfig(),
            extract_func_cfg=StandardExtractConfig(),
            eval_func_cfg=NetworkEvalConfig(
                max_batch_size=1000,
                precision='32-true',
                no_grad=True,
                zero_sum_norm=ZeroSumNorm.NONE,
            ),
            discount=0.99,
            expansion_depth=0,
            use_hot_start=True,
            optimize_fully_explored=False,
        ),
        device_str='cuda:1',
        compile_model_2=False,
        net_path_2=Path(__file__).parent.parent / 'a_models' / 'duct_large2_pooled.pt',
    )

    agent = FrontendAgent(
        agent_cfg=agent_cfg,
        net_path=Path(__file__).parent.parent / 'a_models' / 'standard_finished.pt',
        device=torch.device('cuda:1'),
        name='Deutschlange',
        mode=SupportedMode.STANDARD,
        expected_latency_sec=0.1,
        draw_prevention_limit=100,
        max_iterations=1000,
        compile_model=False,
    )
    port = 8012
    run_server(
        agent=agent,
        port=port
    )


def start_royale_agent():
    agent_cfg = SearchAgentConfig(
        search_cfg=MCTSConfig(
            sel_func_cfg=AlphaZeroDecoupledSelectionConfig(exp_bonus=1.414, dirichlet_eps=0),
            backup_func_cfg=StandardBackupConfig(),
            extract_func_cfg=StandardExtractConfig(),
            eval_func_cfg=NetworkEvalConfig(
                max_batch_size=1000,
                precision='32-true',
                no_grad=True,
                zero_sum_norm=ZeroSumNorm.NONE,
            ),
            discount=0.99,
            expansion_depth=0,
            use_hot_start=True,
            optimize_fully_explored=False,
        )
    )

    agent = FrontendAgent(
        agent_cfg=agent_cfg,
        net_path=Path(__file__).parent.parent / 'a_models' / 'royale_large_unfinished.pt',
        device=torch.device('cuda:1'),
        name='Deutschlange',
        mode=SupportedMode.STANDARD,
        expected_latency_sec=0.1,
        draw_prevention_limit=100,
        max_iterations=1000,
        compile_model=False,
    )
    port = 8013
    run_server(
        agent=agent,
        port=port
    )


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)  # this is important for using CUDA
    # print(f"{mp.get_start_method()=}")
    # start_duel_agent()
    # start_constrictor_agent()
    # start_standard_agent()
    start_royale_agent()
