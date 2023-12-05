

import os
import shutil
import time

import torch
from src.game.imp_marl.imp_marl_wrapper import IMP_MODE, IMPConfig
from src.game.initialization import get_game_from_config

from src.game.overcooked.config import CrampedRoomOvercookedConfig
from src.network.initialization import get_network_from_config
from src.network.resnet import OvercookedResNetConfig5x5


def main():
    game_cfg = IMPConfig(num_players=3, imp_mode=IMP_MODE.K_OF_N, campaign_cost=False)
    game = get_game_from_config(game_cfg)
    
    reward_list = []
    for _ in range(20):
        r, _, _ = game.step((0, 0, 0))
        print(r)
    
    obs, _, _ = game.get_obs()
    
    game.step((0, 1, 0))
    obs2, _, _ = game.get_obs()
    
    game.step((0, 0, 0))
    obs3, _, _ = game.get_obs()
    
    a = 1

if __name__ == '__main__':
    main()
