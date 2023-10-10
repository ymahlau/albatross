import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from src.depth.depth_parallel import DepthSearchConfig, depth_search_config_from_structured, \
    compute_different_depths_parallel


def main(cfg: DepthSearchConfig):
    print(os.getcwd(), flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)
    depth_cfg = depth_search_config_from_structured(cfg)
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
