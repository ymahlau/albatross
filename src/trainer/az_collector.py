import os
import queue
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp

from src.game.initialization import get_game_from_config, buffer_config_from_game
from src.misc.replay_buffer import ReplayBuffer, BufferInputSample
from src.misc.utils import set_seed
from src.trainer.config import CollectorConfig, AlphaZeroTrainerConfig
from src.trainer.utils import send_obj_to_queue


@dataclass
class CollectorEssentials:
    buffer: ReplayBuffer
    data_queue: mp.Queue
    updater_queue: mp.Queue
    info_queue: mp.Queue
    stop_flag: mp.Value

@dataclass
class CollectorStatistics:
    samples_since_info: int = 0
    samples_since_info_buffer: int = 0
    samples_since_info_val: int = 0
    full_sample_counter: int = 0
    full_sample_counter_buffer: int = 0
    full_sample_counter_val: int = 0
    last_info_time: float = time.time()
    last_sample_info: float = time.time()
    last_save_buffer_time: float = time.time()
    buffer_add_time_sum: float = 0
    buffer_sample_time_sum: float = 0
    idle_time_sum: float = 0


def run_collector(
        trainer_cfg: AlphaZeroTrainerConfig,
        data_queue: mp.Queue,
        updater_queue: mp.Queue,
        updater_queue_maxsize: int,
        stop_flag: mp.Value,
        info_queue: mp.Queue,
        save_state: bool,
        save_state_after_seconds: int,
        cpu_list: Optional[list[int]],  # only works on Linux
        only_generate_buffer: bool,
        prev_run_dir: Optional[Path],
        seed: int,
        grouped_sampling: bool,
        film_temperature_sampling: bool,
):
    game_cfg = trainer_cfg.game_cfg
    collector_cfg = trainer_cfg.collector_cfg
    # important to avoid deadlocks
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    set_seed(seed)
    # init
    pid = os.getpid()
    state_save_dir = Path(os.getcwd()) / 'state'
    game = get_game_from_config(game_cfg)
    if only_generate_buffer:
        collector_cfg.validation_percentage = 0
    # buffer
    buffer_cfg = buffer_config_from_game(game, collector_cfg.buffer_size, trainer_cfg.single_sbr_temperature)
    buffer = ReplayBuffer(buffer_cfg, game_cfg=game_cfg)
    if prev_run_dir is not None and collector_cfg.quick_start_buffer_path is not None:
        raise ValueError(f"It is not possible to specify both prev run dir and quick start buffer")
    elif prev_run_dir is not None:
        buffer_path = prev_run_dir / 'state' / 'buffer.pt'
        if os.path.exists(buffer_path):
            buffer = ReplayBuffer.from_saved_file(buffer_path)
        else:
            print(f"{datetime.now()} - WARNING: Cannot load buffer from previous run directory ...........", flush=True)
    elif collector_cfg.quick_start_buffer_path is not None:
        buffer = ReplayBuffer.from_saved_file(Path(collector_cfg.quick_start_buffer_path))
    # essentials and statistics container
    stats = CollectorStatistics()
    essentials = CollectorEssentials(
        buffer=buffer,
        data_queue=data_queue,
        updater_queue=updater_queue,
        info_queue=info_queue,
        stop_flag=stop_flag,
    )
    if cpu_list is not None:
        print(f"{datetime.now()} - CPU list in Collector: {cpu_list}")
        os.sched_setaffinity(pid, cpu_list)
        print(f'{datetime.now()} - Started Collection process with pid {pid} using cpus: {os.sched_getaffinity(pid)}',
              flush=True)
    else:
        print(f'{datetime.now()} - Started Collection process with pid {pid}', flush=True)
    # collector loop
    try:
        while not stop_flag.value:
            # if we only want to generate the initial buffer, check size
            if only_generate_buffer:
                if len(buffer) == collector_cfg.buffer_size:
                    save_buffer(buffer, state_save_dir)
                    stop_flag.value = True
                    break
            # check if new data has arrived
            receive_new_data(essentials, stats, collector_cfg)
            if only_generate_buffer:
                continue
            # maybe save optimizer state
            if save_state and time.time() - stats.last_save_buffer_time > save_state_after_seconds:
                save_buffer(buffer, state_save_dir)
                stats.last_save_buffer_time = time.time()
            # sample data and send to updater
            if stats.full_sample_counter >= collector_cfg.start_wait_n_samples \
                    and essentials.updater_queue.qsize() < updater_queue_maxsize / 2:
                sample_data(
                    essentials=essentials,
                    stats=stats,
                    batch_size=collector_cfg.batch_size,
                    updater_queue_maxsize=updater_queue_maxsize,
                    grouped_sampling=grouped_sampling,
                    film_temperature_sampling=film_temperature_sampling,
                )
            else:
                idle_start = time.time()
                time.sleep(0.001)
                stats.idle_time_sum += time.time() - idle_start
            # log info
            if time.time() - stats.last_info_time > collector_cfg.log_every_sec:
                log_info(essentials, stats)
    except KeyboardInterrupt:
        print('Detected KeyboardInterrupt in Collection process\n', flush=True)
    print(f'{datetime.now()} - Successfully closed Collection {os.getpid()}', flush=True)
    sys.exit(0)


def data_to_buffer(
        essentials: CollectorEssentials,
        stats: CollectorStatistics,
        data: BufferInputSample,
):
    essentials.buffer.put(data)
    # stats
    cur_num_samples: int = data.obs.shape[0]
    stats.samples_since_info_buffer += cur_num_samples
    stats.full_sample_counter_buffer += cur_num_samples


def receive_new_data(
        essentials: CollectorEssentials,
        stats: CollectorStatistics,
        collector_cfg: CollectorConfig,
):
    buffer_time_start = time.time()
    num_samples = 0
    cur_q_size = essentials.data_queue.qsize()
    for _ in range(cur_q_size):
        try:
            data: BufferInputSample = essentials.data_queue.get_nowait()
        except queue.Empty:
            break
        data_to_buffer(essentials, stats, data)
        cur_num_samples: int = data.obs.shape[0]
        num_samples += cur_num_samples
        if essentials.stop_flag.value:
            break
    stats.samples_since_info += num_samples
    stats.full_sample_counter += num_samples
    stats.buffer_add_time_sum += time.time() - buffer_time_start


def sample_data(
        essentials: CollectorEssentials,
        stats: CollectorStatistics,
        batch_size: int,
        updater_queue_maxsize: int,
        grouped_sampling: bool,
        film_temperature_sampling: bool,
):
    sample_start = time.time()
    while essentials.updater_queue.qsize() < updater_queue_maxsize / 2:
        # sample
        sample = essentials.buffer.sample(batch_size, grouped=grouped_sampling, temperature=film_temperature_sampling)
        # add to queue
        essentials.updater_queue.put_nowait(sample)
    stats.buffer_sample_time_sum += time.time() - sample_start

def save_buffer(
        buffer: ReplayBuffer,
        state_save_dir: Path,
):
    if not Path.exists(state_save_dir):
        state_save_dir.mkdir(parents=True, exist_ok=True)
    buffer.save(state_save_dir / 'buffer.pt')

def log_info(
        essentials: CollectorEssentials,
        stats: CollectorStatistics,
):
    full_time = time.time() - stats.last_info_time
    other_time = full_time - stats.buffer_add_time_sum - stats.buffer_sample_time_sum - stats.idle_time_sum
    msg_data = {
        'collector_buffer_add_ratio': stats.buffer_add_time_sum / full_time,
        'collector_buffer_sample_ratio': stats.buffer_sample_time_sum / full_time,
        'collector_idle_ratio': stats.idle_time_sum / full_time,
        'collector_other_ratio': other_time / full_time,
        'updater_in_qsize': essentials.updater_queue.qsize(),
        'full_sample_counter': stats.full_sample_counter,
        'full_sample_counter_buffer': stats.full_sample_counter_buffer,
        'full_sample_counter_val': stats.full_sample_counter_val,
        'samples_per_min': stats.samples_since_info * 60 / full_time,
        'samples_per_min_buffer': stats.samples_since_info_buffer * 60 / full_time,
        'samples_per_min_val': stats.samples_since_info_val * 60 / full_time,
    }
    if stats.samples_since_info_buffer > 0:
        time_diff = time.time() - stats.last_sample_info
        buffer_swap_time = (essentials.buffer.cfg.capacity / stats.samples_since_info_buffer) * time_diff
        msg_data['buffer_swap_min'] = buffer_swap_time / 60
        stats.last_sample_info = time.time()
    stats.buffer_add_time_sum = 0
    stats.buffer_sample_time_sum = 0
    stats.idle_time_sum = 0
    stats.samples_since_info = 0
    stats.samples_since_info_buffer = 0
    stats.samples_since_info_val = 0
    stats.last_info_time = time.time()
    # send message
    send_obj_to_queue(msg_data, essentials.info_queue, essentials.stop_flag)
