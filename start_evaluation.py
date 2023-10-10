import itertools
import math
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import torch.multiprocessing as mp

from src.evaluation.play_parallel import ParallelPlayConfig, play_parallel


def main(cfg: ParallelPlayConfig):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    print(os.getcwd(), flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)
    play_parallel(cfg)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # this is important for using CUDA
    print(f"{mp.get_start_method()=}")
    config_path = Path(__file__).parent / 'config'
    config_name = 'config_tournament2'

    # if len(sys.argv) > 2 and sys.argv[1].startswith("config="):
    #     config_prefix = sys.argv[1].split("=")[-1]
    #     sys.argv.pop(1)
    #     arr_id = int(sys.argv[1])
    #     sys.argv.pop(1)
    #     t = math.floor(arr_id / 5) * 5
    #     seed = arr_id % 5
    #     config_name = f"{config_prefix}_t{t}_{seed}"
    if len(sys.argv) > 3 and sys.argv[1].startswith("config="):
        config_prefix = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)
        arr_id = int(sys.argv[1])
        sys.argv.pop(1)
        # t = math.floor(arr_id / 5)
        # prefix_arr = ['oc', 'oa', 'occ', 'of', 'or']
        # prefix_arr = list(range(50, 2001, 50))
        # prefix_arr = list(range(10))
        # seed = arr_id % 5
        # pref_lists = [
        #     list(range(1, 6)),
        #     [1] + list(range(5, 51, 5)),
        #     list(range(5)),
        # ]
        pref_lists = [
            # list(range(1, 6)),
            # [1] + list(range(5, 51, 5)),
            ['oc', 'oa', 'occ', 'of', 'or'],
            list(range(5)),
        ]
        prod = list(itertools.product(*pref_lists))
        tpl = prod[arr_id]
        # config_name = f"{config_prefix}_{tpl[0]}_{tpl[1]}_{tpl[2]}"
        config_name = f"{config_prefix}_{tpl[0]}_{tpl[1]}"
        # config_name = f"{config_prefix}_{prefix_arr[t]}_{seed}"
        # config_name = f"{config_prefix}_{seed}_{prefix_arr[t]}"
    elif len(sys.argv) > 2 and sys.argv[1].startswith("config="):
        config_name = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)
    print(f"{config_name=}", flush=True)
    hydra.main(config_path=str(config_path), config_name=config_name, version_base=None)(main)()
