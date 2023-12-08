import unittest

import torch

from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedSlowConfig
from src.game.overcooked.config import CrampedRoomOvercookedConfig
from src.network.initialization import get_network_from_config
from src.network.simple_fcn import SimpleNetworkConfig


class TestSimpleFCN(unittest.TestCase):
    def test_simple_fcn_obs(self):
        game_cfg_slow = CrampedRoomOvercookedSlowConfig(flat_obs=True)
        game_slow = get_game_from_config(game_cfg_slow)
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg.flat_obs = True
        game = get_game_from_config(game_cfg)
        net_cfg = SimpleNetworkConfig(game_cfg=game_cfg, layout_abbrev='cr')
        net = get_network_from_config(net_cfg)
        obs, _, _ = game_slow.get_obs()
        print(obs.shape)

        net_out = net(torch.tensor(obs))
        print(net_out)


