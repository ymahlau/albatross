import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import multiprocessing as mp
from src.supervised.trainer import SupervisedTrainerConfig, SupervisedTrainer


def main(cfg: SupervisedTrainerConfig):
    print(os.getcwd())
    print(OmegaConf.to_yaml(cfg))
    trainer_cfg = supervised_trainer_config_from_structured(cfg)
    trainer = SupervisedTrainer(trainer_cfg)
    trainer()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # this is important for using CUDA
    print(f"{mp.get_start_method()=}")
    config_path = Path(__file__).parent / 'config'
    config_name = 'debug_config'

    if len(sys.argv) > 1 and sys.argv[1].startswith("config="):
        config_name = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)
    hydra.main(config_path=str(config_path), config_name=config_name, version_base=None)(main)()
