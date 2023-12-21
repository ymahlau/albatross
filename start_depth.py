import os
import sys
from pathlib import Path
import hydra

from omegaconf import OmegaConf
import multiprocessing as mp
from src.depth.depth_parallel import DepthSearchConfig, compute_different_depths_parallel
from src.misc.serialization import deserialize_dataclass


def main(cfg: DepthSearchConfig):
    print(os.getcwd(), flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    depth_cfg = deserialize_dataclass(cfg_dict)
    compute_different_depths_parallel(depth_cfg)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # this is important for using CUDA
    print(f"{mp.get_start_method()=}")
    config_path = Path(__file__).parent / 'config'
    config_name = 'config_depth'

    if len(sys.argv) > 1 and sys.argv[1].startswith("config="):
        config_name = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)
    hydra.main(config_path=str(config_path), config_name=config_name, version_base=None)(main)()
