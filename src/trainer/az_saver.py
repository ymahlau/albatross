import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp

from src.network.initialization import get_network_from_config
from src.trainer.config import AlphaZeroTrainerConfig
from src.trainer.utils import send_obj_to_queue, get_latest_obj_from_queue


def run_saver(
        trainer_cfg: AlphaZeroTrainerConfig,
        net_queue: mp.Queue,
        stop_flag: mp.Value,
        info_queue: mp.Queue,
        cpu_list: Optional[list[int]],
):
    net_cfg = trainer_cfg.net_cfg
    saver_cfg = trainer_cfg.saver_cfg
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    model_folder: Path = Path(os.getcwd()) / 'fixed_time_models'
    if not Path.exists(model_folder):
        model_folder.mkdir(parents=True, exist_ok=True)
    save_counter: int = 0
    last_save_time = time.time()
    net = get_network_from_config(net_cfg)
    # restrict cpus
    pid = os.getpid()
    if cpu_list is not None:
        print(f"{datetime.now()} - CPU list in Saver: {cpu_list}")
        os.sched_setaffinity(pid, cpu_list)
        print(f'{datetime.now()} - Started Saver process with pid {pid} using cpus: {os.sched_getaffinity(pid)}',
              flush=True)
    else:
        print(f'{datetime.now()} - Started Saver process with pid {pid}', flush=True)
    try:
        while not stop_flag.value:
            # check if enough time has passed
            time_passed = time.time() - last_save_time
            if time_passed > saver_cfg.save_interval_sec:
                maybe_state_dict = get_latest_obj_from_queue(net_queue)
                if maybe_state_dict is None:
                    print(f"Warning, Saver did not receive model...", flush=True)
                else:
                    net.load_state_dict(maybe_state_dict)
                    net.save(model_folder / f"m_{save_counter}.pt")
                    msg = {"save_counter": save_counter}
                    send_obj_to_queue(msg, info_queue, stop_flag)
                    save_counter += 1
                last_save_time = time.time()
            time.sleep(5)
    except KeyboardInterrupt:
        print("Detected CTRL-C in saver, closing...", flush=True)
    print("Successfully closed saver", flush=True)
    sys.exit(0)