import os
import sys
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from src.trainer.az_trainer import AlphaZeroTrainer, AlphaZeroTrainerConfig


# @hydra.main(version_base=None, config_name='config', config_path=str(Path(__file__).parent / 'config_generated'))
def main(cfg: AlphaZeroTrainerConfig):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    print(os.getcwd(), flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)
    a = 1
    # trainer_cfg = trainer_config_from_structured(cfg)
    # trainer = AlphaZeroTrainer(trainer_cfg)
    # if trainer_cfg.prev_run_dir is None:
    #     trainer.start_training()
    # else:
    #     trainer.continue_training()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # this is important for using CUDA
    print(f"{mp.get_start_method()=}")
    config_path = Path(__file__).parent / 'config'
    config_name = 'config'

    if len(sys.argv) > 1 and sys.argv[1].startswith("config="):
        config_name = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)
    hydra.main(config_path=str(config_path), config_name=config_name, version_base=None)(main)()
