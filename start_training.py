import itertools
import os
import sys
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from src.misc.serialization import deserialize_dataclass
from src.trainer.az_trainer import AlphaZeroTrainer, AlphaZeroTrainerConfig

# @hydra.main(version_base=None, config_name='config', config_path=str(Path(__file__).parent / 'config_generated'))
def main(cfg: AlphaZeroTrainerConfig):
    # torch.set_num_threads(1)
    # os.environ["OMP_NUM_THREADS"] = "1"
    print(os.getcwd(), flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    trainer_cfg = deserialize_dataclass(cfg_dict)
    trainer = AlphaZeroTrainer(trainer_cfg)
    if trainer_cfg.prev_run_dir is None:
        trainer.start_training()
    else:
        trainer.continue_training()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # this is important for using CUDA
    print(f"{mp.get_start_method()=}")
    config_path = Path(__file__).parent / 'config'
    config_name = 'config'

    if len(sys.argv) > 3 and sys.argv[1].startswith("config="):
        config_prefix = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)
        arr_id = int(sys.argv[1])
        sys.argv.pop(1)

        pref_lists = [
            # list(range(1, 6)),
            # [1] + list(range(5, 51, 5)),
            ['aa', 'cc', 'co', 'cr', 'fc'],
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
