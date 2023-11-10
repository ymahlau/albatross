import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.misc.utils import set_seed
from src.network.initialization import get_network_from_config, get_network_from_file
from src.trainer.config import AlphaZeroTrainerConfig
import multiprocessing as mp

from src.trainer.utils import get_latest_obj_from_queue


@dataclass
class InferenceServerStats:
    load_time_sum: float = 0


def run_inference_server(
        trainer_cfg: AlphaZeroTrainerConfig,
        net_queue: mp.Queue,
        stop_flag: mp.Value,
        info_queue: mp.Queue,
        input_rdy_arr: mp.Array,
        output_rdy_arr: mp.Array,
        input_arr: mp.Array,
        output_arr: mp.Array,
        cpu_list: Optional[list[int]],
        gpu_idx: Optional[int],
        prev_run_dir: Optional[Path],
        prev_run_idx: Optional[int],
):
    # important to avoid pytorch deadlocks
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    inf_cfg = trainer_cfg.inf_cfg
    # load initial network
    net_cfg = trainer_cfg.net_cfg
    net = get_network_from_config(net_cfg)
    if prev_run_dir is not None and not trainer_cfg.init_new_network_params:
        model_path = prev_run_dir / 'fixed_time_models' / f'm_{prev_run_idx}.pt'
        net = get_network_from_file(model_path)
    # cuda
    if inf_cfg.use_gpu:
        torch.cuda.init()
        device = torch.device('cuda' if gpu_idx is None else f'cuda:{gpu_idx}')
    else:
        device = torch.device('cpu')
    net = net.to(device)
    net = net.test()
    # compile
    if trainer_cfg.compile_model:
        net = torch.compile(
            model=net,
            dynamic=False,
            mode=trainer_cfg.compile_mode,
            fullgraph=True,
        )
    # restrict cpus
    pid = os.getpid()
    if cpu_list is not None:
        print(f"{datetime.now()} - CPU list in Updater: {cpu_list}")
        os.sched_setaffinity(pid, cpu_list)
        print(f'{datetime.now()} - Started Inference Server with pid {pid} using cpus: {os.sched_getaffinity(pid)}'
              f" using device {device}", flush=True)
    else:
        print(f'{datetime.now()} - Started Inference Server with pid {pid} using device {device}', flush=True)
    # statistics
    stats = InferenceServerStats()
    # processing loop
    try:
        while not stop_flag.value:
            # get the newest network state dictionary
            load_time_start = time.time()
            maybe_state_dict = get_latest_obj_from_queue(net_queue)
            if stop_flag.value:
                break
            if maybe_state_dict is not None:
                net.load_state_dict(maybe_state_dict)
                net = net.eval()
            stats.load_time_sum += time.time() - load_time_start
            # get ready input data
            input_rdy_cpy = np.array(np.copy(input_rdy_arr.value), dtype=bool)
            if np.all(input_rdy_cpy == 0):
                continue
            obs_filtered = input_arr.value[input_rdy_cpy]
            n = obs_filtered.shape[0]
            # forward pass for all encodings, but do not exceed max batch size
            if n <= trainer_cfg.max_batch_size:
                enc_tensor = torch.from_numpy(obs_filtered).to(device)
                out_tensor_with_grad = net(enc_tensor)
                out_tensor = out_tensor_with_grad.cpu().detach().float().numpy()
            else:
                start_idx = 0
                out_tensor_list = []
                end_idx_list = list(range(trainer_cfg.max_batch_size, n, trainer_cfg.max_batch_size))
                if end_idx_list[-1] < n:
                    end_idx_list.append(n)
                for end_idx in end_idx_list:
                    enc_tensor = torch.from_numpy(obs_filtered[start_idx:end_idx]).to(device)
                    out_tensor_part_with_grad = net(enc_tensor)
                    out_tensor_part = out_tensor_part_with_grad.cpu().detach().numpy()
                    out_tensor_list.append(out_tensor_part)
                    start_idx = end_idx
                out_tensor = np.concatenate(out_tensor_list, axis=0)
            # send output back to processes
            input_rdy_arr[input_rdy_cpy] = 0
            output_arr[input_rdy_cpy] = out_tensor
            output_rdy_arr[input_rdy_cpy] = 1
            if stop_flag.value:
                break
            # send statistics

    except KeyboardInterrupt:
        print('Detected Keyboard Interrupt in Inference Server\n', flush=True)
    # cleanup
    print(f'{datetime.now()} - Inference Server with pid {os.getpid()} terminated', flush=True)
    sys.exit(0)
