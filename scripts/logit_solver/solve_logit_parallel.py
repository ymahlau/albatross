import multiprocessing as mp
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from src.equilibria.logit import SbrMode, compute_logit_equilibrium
from src.game.normal_form.normal_form import NormalFormConfig
from src.game.normal_form.random_matrix import NFGType, get_random_matrix_cfg


@dataclass
class LogitSolverExperimentConfig:
    num_player: int
    num_actions: int
    num_iterations: int
    nfg_type: NFGType
    num_games: int
    sbr_mode: SbrMode
    temperature_range: tuple[float, float]
    hp_0: Optional[float] = None
    hp_1: Optional[float] = None


@dataclass
class EquilibriumData:
    experiment_config: LogitSolverExperimentConfig
    game_cfgs: list[NormalFormConfig]
    temperatures: np.ndarray  # shape (num_games)
    policies: np.ndarray  # shape (num_games, num_players, num_actions)
    values: np.ndarray  # shape (num_games, num_players)
    errors: np.ndarray  # shape (num_games)

static_experiment_cfg: Optional[LogitSolverExperimentConfig] = None

def solve_nfg(cfg: NormalFormConfig, temperature: float, game_id: int):
    if static_experiment_cfg is None:
        raise Exception("Static experiment config is None")
    if game_id % 100 == 0:
        print(f"{datetime.now()} - {game_id=}", flush=True)
    aa = [list(range(static_experiment_cfg.num_actions)) for _ in range(static_experiment_cfg.num_player)]
    ja = list(cfg.ja_dict.keys())
    vals = np.asarray(list(cfg.ja_dict.values()), dtype=float)
    temps = [temperature for _ in range(static_experiment_cfg.num_player)]
    values, policies, error = compute_logit_equilibrium(
        available_actions=aa,
        joint_action_list=ja,
        joint_action_value_arr=vals,
        num_iterations=static_experiment_cfg.num_iterations,
        temperatures=temps,
        epsilon=0,
        sbr_mode=static_experiment_cfg.sbr_mode,
        hp_0=static_experiment_cfg.hp_0,
        hp_1=static_experiment_cfg.hp_1,
    )
    return values, policies, error


def init_process(
        cpu_list: Optional[list[int]],
        experiment_cfg: LogitSolverExperimentConfig,
):
    global static_experiment_cfg
    static_experiment_cfg = experiment_cfg
    pid = os.getpid()
    if cpu_list is not None:
        os.sched_setaffinity(pid, cpu_list)
        print(f"{datetime.now()} - Started computation with pid {pid} using restricted cpus: "
              f"{os.sched_getaffinity(pid)}", flush=True)
    else:
        print(f"{datetime.now()} - Started computation with pid {pid}", flush=True)

def compute_equilibrium_data_parallel(
        cfg: LogitSolverExperimentConfig,
        restrict_cpu: bool,
        num_procs: int,
        gt_path: Optional[str],
) -> EquilibriumData:
    # restrict cpu
    cpu_list = None
    if restrict_cpu:
        cpu_list = os.sched_getaffinity(0)
    # initialization
    print(f"{datetime.now()} - Main process with pid {os.getpid()} uses {cpu_list=}", flush=True)
    init_args = (cpu_list, cfg)
    gt_data: Optional[EquilibriumData] = None
    if gt_path is not None:
        with open(gt_path, 'rb') as f:
            gt_data = pickle.load(f)
    with mp.Pool(processes=num_procs, initargs=init_args, initializer=init_process) as pool:
        result_list_unfinished = []
        temperature_list = []
        game_cfg_list = []
        for game_id in range(cfg.num_games):
            if gt_data is None:
                game_cfg = get_random_matrix_cfg(
                    nfg_type=cfg.nfg_type,
                    num_actions_per_player=[cfg.num_actions for _ in range(cfg.num_player)]
                )
            else:
                game_cfg = gt_data.game_cfgs[game_id]
            temperature = random.random() * (cfg.temperature_range[1] - cfg.temperature_range[0])
            temperature += cfg.temperature_range[0]
            result_tpl_unfinished = pool.apply_async(
                func=solve_nfg,
                kwds={'cfg': game_cfg, 'temperature': temperature, 'game_id': game_id}
            )
            result_list_unfinished.append(result_tpl_unfinished)
            temperature_list.append(temperature)
            game_cfg_list.append(game_cfg)
        # wait for children to finish
        result_list = [res.get(timeout=None) for res in result_list_unfinished]
    # aggregate results
    full_values = np.stack([r[0] for r in result_list])
    full_policies = np.stack([r[1] for r in result_list])
    full_errors = np.stack([r[2] for r in result_list])
    temperatures = np.asarray(temperature_list)
    if gt_data is not None:
        game_cfg_list = []
        temperatures = np.asarray([])
    data = EquilibriumData(
        experiment_config=cfg,
        values=full_values,
        policies=full_policies,
        errors=full_errors,
        game_cfgs=game_cfg_list,
        temperatures=temperatures,
    )
    return data


