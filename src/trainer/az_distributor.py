import os
import sys
import time
from datetime import datetime
from typing import Optional

import multiprocessing as mp

from src.trainer.utils import send_obj_to_queue

import queue as q
import multiprocessing.sharedctypes as sc

def run_distributor(
        in_queue: mp.Queue,
        net_queue_list: list[mp.Queue],
        stop_flag: sc.Synchronized,
        info_queue: mp.Queue,
        max_queue_size: int,
        info_bucket_size: int,
        cpu_list: Optional[list[int]] = None,  # only works on Linux
):
    # init
    pid = os.getpid()
    if cpu_list is not None:
        print(f"{datetime.now()} - CPU list in Distributor: {cpu_list}")
        os.sched_setaffinity(pid, cpu_list)
        print(f'{datetime.now()} - Started distribution process with pid {pid} using cpus: {os.sched_getaffinity(pid)}',
              flush=True)
    else:
        print(f'{datetime.now()} - Started distribution process with pid {pid}', flush=True)
    try:
        updates_since_info = 0
        update_needed_counter = 0
        useless_dists = 0
        avg_qsizes = []
        while not stop_flag.value:
            # check if new version of state dict is available
            maybe_state_dict = None
            counter = 0
            while not in_queue.empty():
                try:
                    temp = in_queue.get_nowait()
                    maybe_state_dict = temp
                    counter += 1
                except q.Empty:
                    break
            if counter > 1:
                useless_dists += counter - 1
            # maybe_state_dict = in_queue.get()
            if stop_flag.value:
                break
            if maybe_state_dict is None:
                time.sleep(0.1)
                continue
            # distribute updates
            useless = True
            cur_qsize_sum = 0
            for queue in net_queue_list:
                # put new item in
                if queue.empty():
                    useless = False
                    update_needed_counter += 1
                try:
                    queue.put_nowait(maybe_state_dict)
                except q.Full:
                    pass
                # if there are too many items in the queue, remove some old ones
                if queue.qsize() > max_queue_size / 2:
                    try:
                        queue.get_nowait()
                        queue.get_nowait()
                    except q.Empty:
                        pass
                cur_qsize_sum += queue.qsize()
            avg_qsizes.append(cur_qsize_sum / len(net_queue_list))
            useless_dists += int(useless)
            # maybe send info
            updates_since_info += 1
            if updates_since_info == info_bucket_size:
                msg_data = {
                    'dist_receive_ratio': update_needed_counter / (info_bucket_size * len(net_queue_list)),
                    'dist_useless_ratio': useless_dists / info_bucket_size,
                    'dist_avg_qsize': sum(avg_qsizes) / len(avg_qsizes)
                }
                send_obj_to_queue(msg_data, info_queue, stop_flag)
                avg_qsizes = []
                updates_since_info = 0
                update_needed_counter = 0
                useless_dists = 0
    except KeyboardInterrupt:
        print('Detected KeyboardInterrupt in Distribution process\n', flush=True)
    print(f'{datetime.now()} - Successfully closed Distributor {os.getpid()}', flush=True)
    sys.exit(0)
