import unittest

from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedConfig
from src.network.initialization import get_network_from_config
from src.network.simple_fcn import SimpleNetworkConfig


class TestSimpleFCN(unittest.TestCase):
    def test_simple_fcn_obs(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg.flat_obs = True
        game = get_game_from_config(game_cfg)
        net_cfg = SimpleNetworkConfig(game_cfg=game_cfg)
        net = get_network_from_config(net_cfg)
        obs, _, _ = game.get_obs()
        print(obs.shape)

        net_out = net(obs)
        print(net_out)


