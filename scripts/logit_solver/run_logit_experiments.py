from datetime import datetime
import itertools
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from scripts.logit_solver.solve_logit_parallel import compute_equilibrium_data_parallel, LogitSolverExperimentConfig, \
    EquilibriumData
from src.equilibria.logit import SbrMode
from src.game.normal_form.random_matrix import NFGType


def create_logit_data_func(
        num_iterations: int,
        save_path: str,
        sbr_mode_str: str,
        hp_0: Optional[float],
        hp_1: Optional[float],
        gt_path: Optional[str],
        nfg_type: NFGType,
        num_games: int,
        temperature_range: tuple[int, int],
        num_player: int,
        num_actions: int,
        num_procs: int,
):
    sbr_mode = SbrMode[sbr_mode_str]
    cfg = LogitSolverExperimentConfig(
        num_player=num_player,
        num_actions=num_actions,
        num_iterations=num_iterations,
        nfg_type=nfg_type,
        num_games=num_games,
        sbr_mode=sbr_mode,
        temperature_range=temperature_range,
        hp_0=hp_0,
        hp_1=hp_1,
    )
    data = compute_equilibrium_data_parallel(
        cfg=cfg,
        restrict_cpu=True,
        num_procs=num_procs,
        gt_path=gt_path,
    )
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Done!', flush=True)


def logit_data_from_gt(
    gt_path: str,
    num_iterations: int,
    save_path: Path,
    sbr_mode_str: str,
    hp_0: Optional[float],
    hp_1: Optional[float],
    num_procs: int,
):
    with open(gt_path, 'rb') as f:
        gt_data: EquilibriumData = pickle.load(f)
    sbr_mode = SbrMode[sbr_mode_str]
    cfg = LogitSolverExperimentConfig(
        num_player=gt_data.experiment_config.num_player,
        num_actions=gt_data.experiment_config.num_actions,
        num_iterations=num_iterations,
        nfg_type=gt_data.experiment_config.nfg_type,
        num_games=gt_data.experiment_config.num_games,
        sbr_mode=sbr_mode,
        temperature_range=gt_data.experiment_config.temperature_range,
        hp_0=hp_0,
        hp_1=hp_1,
    )
    data = compute_equilibrium_data_parallel(
        cfg=cfg,
        restrict_cpu=True,
        num_procs=num_procs,
        gt_path=gt_path,
    )
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Done!', flush=True)


# def merge_data():
#     path_low = (Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'old' /
#                 'ground_truth_zs_0_5.pkl')
#     path_high = (Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'old' /
#                  'ground_truth_zs_5_10.pkl')
#     path_merged = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'ground_truth_zs_0_10.pkl'
#     with open(path_low, 'rb') as f:
#         data_low: EquilibriumData = pickle.load(f)
#     with open(path_high, 'rb') as f:
#         data_high: EquilibriumData = pickle.load(f)
#     cfg_merged = LogitSolverExperimentConfig(
#         num_player=2,
#         num_actions=6,
#         num_iterations=10000000,
#         nfg_type=NFGType.ZERO_SUM,
#         num_games=100000,
#         sbr_mode=SbrMode.MSA,
#         temperature_range=(0, 10),
#         hp_0=None,
#         hp_1=None,
#     )
#     data_merged = EquilibriumData(
#         experiment_config=cfg_merged,
#         game_cfgs=data_low.game_cfgs + data_high.game_cfgs,
#         temperatures=np.concatenate((data_low.temperatures, data_high.temperatures)),
#         policies=np.concatenate((data_low.policies, data_high.policies), axis=0),
#         values=np.concatenate((data_low.values, data_high.values), axis=0),
#         errors=np.concatenate((data_low.errors, data_high.errors), axis=0),
#     )
#     with open(path_merged, 'wb') as f:
#         pickle.dump(data_merged, f)


def generate_gt_data():
    dir_path = Path(__file__).parent.parent.parent / 'a_data' / 'logit_solver'
    num_player = 2
    num_actions = 6
    nfg_type = NFGType.ZERO_SUM
    sbr_mode_str = 'MSA'
    hp_0 = None
    hp_1 = None
    num_games = int(1e5)
    gt_path = None
    temperature_range = (0, 10)
    num_iterations = int(1e7)
    num_procs = 30
    
    
    file_name = f'gt_{sbr_mode_str.lower()}_{nfg_type.value}.pkl'
    full_save_path = str(dir_path / file_name)
    print(f"{full_save_path=}", flush=True)
    create_logit_data_func(
        num_iterations=num_iterations,
        save_path=full_save_path,
        sbr_mode_str=sbr_mode_str,
        hp_0=hp_0,
        hp_1=hp_1,
        gt_path=gt_path,
        nfg_type=nfg_type,
        num_games=num_games,
        temperature_range=temperature_range,
        num_player=num_player,
        num_actions=num_actions,
        num_procs=num_procs,
    )


def step_size_experiment(experiment_id: int):
    hp_dict: dict[str, tuple[Optional[float], Optional[float]]] = {
        'EMA': (0.5, None),
        'MSA': (None, None),
        'POLYAK': (None, None),
        'NAGURNEY': (None, None),
        'SRA': (0.3, 1.8),
    }
    num_procs = 10
    
    pref_lists = [  # 3 * 5 * 40 = 600
        [NFGType.ZERO_SUM, NFGType.FULL_COOP, NFGType.GENERAL],
        list(hp_dict.keys()),
        list(range(50, 2001, 50)),  # 40
    ]
    prod = list(itertools.product(*pref_lists))
    nfg_type, sbr_mode_str, cur_iterations = prod[experiment_id]
    
    
    dir_path = Path(__file__).parent.parent.parent / 'a_data' / 'logit_solver'
    file_name = f'data_{sbr_mode_str.lower()}_{nfg_type.value}_{cur_iterations}.pkl'
    gt_name = f'gt_msa_{nfg_type.value}.pkl'
    full_save_path = dir_path / file_name
    gt_path = dir_path / gt_name
    print(f"{full_save_path=}", flush=True)
    logit_data_from_gt(
        num_iterations=cur_iterations,
        save_path=full_save_path,
        sbr_mode_str=sbr_mode_str,
        hp_0=hp_dict[sbr_mode_str][0],
        hp_1=hp_dict[sbr_mode_str][1],
        gt_path=str(gt_path),
        num_procs=num_procs,
    )

def temp():
    dir_path = Path(__file__).parent.parent.parent / 'a_data' / 'logit_solver'
    sbr_mode_str = 'MSA'
    nfg_type = NFGType.GENERAL
    gt_name = f'gt_{sbr_mode_str.lower()}_{nfg_type.value}.pkl'
    with open(dir_path / gt_name, 'rb') as f:
        data = pickle.load(f)
    a = 1
    

if __name__ == '__main__':
    # create_gt_logit_data()
    # generate_gt_data()
    step_size_experiment(0)
    # temp()
