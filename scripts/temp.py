

import os
import shutil
import time

import torch
from src.game.initialization import get_game_from_config

from src.game.overcooked.config import CrampedRoomOvercookedConfig
from src.network.initialization import get_network_from_config
from src.network.resnet import OvercookedResNetConfig5x5


def main():
    device = torch.device('cuda')
    
    game_cfg = CrampedRoomOvercookedConfig()
    game = get_game_from_config(game_cfg)
    obs, _, _ = game.get_obs()
    obs_tensor = torch.tensor(obs).to(device)
    
    net_cfg = OvercookedResNetConfig5x5(game_cfg=game_cfg)
    net = get_network_from_config(net_cfg).to(device)
    net2 = get_network_from_config(net_cfg).to(device)
    
    
    # backends: ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
    net = torch.compile(
        model=net,
        dynamic=False,
        mode='default',
        # mode='max-autotune-no-cudagraphs',
        fullgraph=True,
        backend='cudagraphs',
    )
    
    start_time = time.time()
    net_out = net(obs_tensor)
    compile_time = time.time() - start_time
    print(f'compile done in {compile_time}')
    
    start_time = time.time()
    for _ in range(10):
        net_out = net(obs_tensor)
    end_time = time.time()
    print(f"{end_time - start_time =}")

if __name__ == '__main__':
    main()
