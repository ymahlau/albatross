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
):
    sbr_mode = SbrMode[sbr_mode_str]
    cfg = LogitSolverExperimentConfig(
        num_player=2,
        num_actions=6,
        num_iterations=num_iterations,
        nfg_type=NFGType.FULL_COOP,
        num_games=200000,
        sbr_mode=sbr_mode,
        temperature_range=(0, 10),
        hp_0=hp_0,
        hp_1=hp_1,
    )
    data = compute_equilibrium_data_parallel(
        cfg=cfg,
        restrict_cpu=True,
        num_procs=8,
        gt_path=gt_path,
    )
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Done!', flush=True)

def merge_data():
    path_low = (Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'old' /
                'ground_truth_zs_0_5.pkl')
    path_high = (Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'old' /
                 'ground_truth_zs_5_10.pkl')
    path_merged = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'ground_truth_zs_0_10.pkl'
    with open(path_low, 'rb') as f:
        data_low: EquilibriumData = pickle.load(f)
    with open(path_high, 'rb') as f:
        data_high: EquilibriumData = pickle.load(f)
    cfg_merged = LogitSolverExperimentConfig(
        num_player=2,
        num_actions=6,
        num_iterations=10000000,
        nfg_type=NFGType.ZERO_SUM,
        num_games=100000,
        sbr_mode=SbrMode.MSA,
        temperature_range=(0, 10),
        hp_0=None,
        hp_1=None,
    )
    data_merged = EquilibriumData(
        experiment_config=cfg_merged,
        game_cfgs=data_low.game_cfgs + data_high.game_cfgs,
        temperatures=np.concatenate((data_low.temperatures, data_high.temperatures)),
        policies=np.concatenate((data_low.policies, data_high.policies), axis=0),
        values=np.concatenate((data_low.values, data_high.values), axis=0),
        errors=np.concatenate((data_low.errors, data_high.errors), axis=0),
    )
    with open(path_merged, 'wb') as f:
        pickle.dump(data_merged, f)

def temp():
    # save_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'temp2.pkl'
    # gt_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'temp.py.pkl'
    # create_logit_data_func(
    #     num_iterations=100,
    #     sbr_mode_str='POLYAK',
    #     save_path=str(save_path),
    #     gt_path=str(gt_path),
    #     hp_0=None,
    #     hp_1=None,
    # )
    # with open(save_path, 'rb') as f:
    #     data = pickle.load(f)
    path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'ema' / 'ema_zs_50.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    a = 1


def generate_experiment_data(experiment_id: int):
    sbr_mode_str = 'SRA_NAGURNEY'
    hp_0 = 1.5
    hp_1 = None
    all_iterations = np.arange(50, 2001, 50)
    cur_iterations = all_iterations[experiment_id].item()
    file_name = f'{sbr_mode_str.lower()}_zs_{cur_iterations}.pkl'
    save_path = (Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / f'{sbr_mode_str.lower()}'
                 / file_name)
    gt_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'ground_truth_zs_0_10.pkl'
    create_logit_data_func(
        num_iterations=cur_iterations,
        save_path=str(save_path),
        sbr_mode_str=sbr_mode_str,
        hp_0=hp_0,
        hp_1=hp_1,
        gt_path=str(gt_path),
    )


if __name__ == '__main__':
    # create_gt_logit_data()
    # temp()
    # merge_data()
    generate_experiment_data(0)
