import atexit
import os
import queue as q
import time
from pathlib import Path
from typing import Any, Optional

import torch.multiprocessing as mp

from src.misc.replay_buffer import ReplayBuffer, BufferInputSample
from src.network import Network


def send_obj_to_queue(
        obj: Any,
        queue: mp.Queue,
        stop_flag: mp.Value,
) -> None:
    while not stop_flag.value:
        try:
            queue.put(obj, timeout=1, block=True)
            break
        except q.Full:
            time.sleep(0.1)


def get_latest_obj_from_queue(
        queue: mp.Queue,
) -> Optional[Any]:
    obj = None
    while not queue.empty():  # make sure we have the very newest object available
        try:
            temp = queue.get_nowait()
            obj = temp
        except q.Empty:
            pass
    return obj


def wait_for_obj_from_queue(
        queue: mp.Queue,
        stop_flag: mp.Value,
        timeout: float = 1,
) -> Optional[Any]:
    obj = None
    while not stop_flag.value:
        try:
            obj = queue.get(block=True, timeout=timeout)
            break
        except q.Empty:
            time.sleep(0.1)
    return obj

def save_network(net: Network, name: str):
    save_dir = Path(os.getcwd()) / 'exit'
    save_dir.mkdir(exist_ok=True, parents=True)
    time_str = time.strftime("%m_%d__%H_%M_%S")
    net.save(save_dir / f"{time_str}_{name}.pt")


def register_exit_function(net: Network, name: str):
    kwargs = {'net': net, 'name': name}
    atexit.register(save_network, **kwargs)


def receive_data_to_buffer(
        data_queue: mp.Queue,
        buffer: ReplayBuffer,
        stop_flag: mp.Value,
) -> int:
    num_samples = 0
    while not data_queue.empty():
        try:
            data: BufferInputSample = data_queue.get_nowait()
        except q.Empty:
            break
        buffer.put(data)
        cur_num_samples: int = data.obs.shape[0]
        num_samples += cur_num_samples
        if stop_flag.value:
            break
    return num_samples
