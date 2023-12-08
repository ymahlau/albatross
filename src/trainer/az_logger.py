import dataclasses
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import multiprocessing as mp
import wandb

from src.misc.utils import flatten_dict_rec
from src.trainer.config import AlphaZeroTrainerConfig
from src.trainer.utils import wait_for_obj_from_queue
import multiprocessing.sharedctypes as sc

"""
Single purpose of this logger is to send all logging info to wandb.
"""


def run_logger(
        trainer_cfg: AlphaZeroTrainerConfig,
        info_queue: mp.Queue,
        data_queue: mp.Queue,
        stop_flag: sc.Synchronized,
        step_counter: sc.Synchronized,
        episode_counter: sc.Synchronized,
        update_counter: sc.Synchronized,
        error_counter: sc.Synchronized,
        cpu_list: Optional[list[int]],
):
    # paths
    wandb_dir = Path(__file__).parent.parent.parent / 'wandb'
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    # init wandb
    logger_cfg = trainer_cfg.logger_cfg
    if logger_cfg.buffer_gen:
        logger_cfg.project_name = "battlesnake_rl_buffer"
    config_dict = dataclasses.asdict(trainer_cfg)
    flat_config_dict = flatten_dict_rec(config_dict)
    kwargs = {'project': logger_cfg.project_name, 'config': flat_config_dict, 'dir': wandb_dir,
              'mode': logger_cfg.wandb_mode, 'save_code': False}
    if logger_cfg.name is not None:
        name = logger_cfg.name if logger_cfg.id is None else f"{logger_cfg.name}_{logger_cfg.id}"
        kwargs["name"] = name
    run = wandb.init(**kwargs)
    if run is None:
        raise Exception("run is None, error of wandb")
    print(f"Wandb Dir: {run.dir}", flush=True)
    # restrict cpus
    pid = os.getpid()
    if cpu_list is not None:
        print(f"{datetime.now()} - CPU list in Logger: {cpu_list}")
        os.sched_setaffinity(pid, cpu_list)
        print(f'{datetime.now()} - Started Logging process with pid {pid} using cpus: {os.sched_getaffinity(pid)}',
              flush=True)
    else:
        print(f'{datetime.now()} - Started Logging process with pid {pid}', flush=True)
    # logging loop
    try:
        while not stop_flag.value:
            # wait for info message
            info_msg: Optional[Any] = wait_for_obj_from_queue(info_queue, stop_flag)
            if stop_flag.value:
                break
            if info_msg is None:
                raise Exception("Unknown error with queue")
            save_info(info_msg, info_queue, data_queue, step_counter, episode_counter, update_counter, error_counter)
    except KeyboardInterrupt:
        print('Detected Keyboard Interrupt in Logging process\n', flush=True)
    print("Logging process was stopped", flush=True)
    sys.exit(0)


def save_info(
        info_dict: dict[str, Any],
        info_queue: mp.Queue,
        data_queue: mp.Queue,
        step_counter: sc.Synchronized,
        episode_counter: sc.Synchronized,
        update_counter: sc.Synchronized,
        error_counter: sc.Synchronized,
) -> None:
    # add values to logging dict (x-Axis and status)
    info_dict["update_counter"] = update_counter.value
    info_dict["step_counter"] = step_counter.value
    info_dict["games_played"] = episode_counter.value
    info_dict["info_queue_size"] = info_queue.qsize()
    info_dict["data_queue_size"] = data_queue.qsize()
    info_dict["error_counter"] = error_counter.value
    # send to wandb
    wandb.log(info_dict)
    # print to console
    print(f"{datetime.now()} - {info_dict}", flush=True)
