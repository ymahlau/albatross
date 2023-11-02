import math
import os
from datetime import datetime
from typing import Optional


def basic_init_process(cpu_list: Optional[list[int]]):
    pid = os.getpid()
    if cpu_list is not None:
        print(f"{cpu_list=}")
        os.sched_setaffinity(pid, cpu_list)
        print(f"{datetime.now()} - Started computation with pid {pid} using restricted cpus: "
              f"{os.sched_getaffinity(pid)}", flush=True)
    else:
        print(f"{datetime.now()} - Started computation with pid {pid}", flush=True)


def partition_indices(
        start_idx: int,
        end_idx: int,
        min_size: int,
        max_num_partitions: int,
) -> list[tuple[int, int]]:
    size = end_idx - start_idx
    if max_num_partitions * min_size < size:
        # normal case, all partitions have size > min_size
        num_partitions = max_num_partitions
    else:
        # we have more max_partitions than size and min_size allow
        if size < min_size:
            num_partitions = 1
        else:
            num_partitions = math.floor(size / min_size)
    avg_part_size = size / num_partitions
    partitions = []
    for p_idx in range(num_partitions):
        start_idx = math.floor(p_idx * avg_part_size)
        end_idx = math.floor((p_idx + 1) * avg_part_size)
        partitions.append((start_idx, end_idx))
    return partitions
