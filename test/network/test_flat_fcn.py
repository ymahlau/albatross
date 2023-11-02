import unittest
from pathlib import Path

import torch

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.network.flat_fcn import SmallFlatFCNConfig
from src.network.initialization import get_network_from_config, get_network_from_file


class TestFCN(unittest.TestCase):
    def test_fcn_game(self):
        gc = BattleSnakeConfig(
            w=3,
            h=3,
            num_players=4,
        )
        gc.ec.flatten = True
        game = BattleSnakeGame(gc)
        fcn_conf = SmallFlatFCNConfig(
            num_layer=5,
            hidden_size=64,
            predict_policy=True,
            lff_features=True,
            lff_feature_expansion=10,
        )
        fcn_conf.game_cfg = gc
        net = get_network_from_config(fcn_conf)
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(4, out.shape[0])
        self.assertEqual(5, out.shape[1])

    def test_save_load(self):
        temp_net_path = Path(__file__).parent / 'temp_net.pt'
        gc = BattleSnakeConfig(
            w=3,
            h=3,
            num_players=4,
        )
        gc.ec.flatten = True
        fcn_conf = SmallFlatFCNConfig(
            num_layer=2,
            hidden_size=64,
            predict_policy=True,
            lff_features=True,
            lff_feature_expansion=10,
        )
        fcn_conf.game_cfg = gc
        net = get_network_from_config(fcn_conf)
        net.save(temp_net_path)
        net.reset()
        net.load(temp_net_path)

        net2 = get_network_from_file(temp_net_path)

        for k in net.state_dict().keys():
            v1: torch.Tensor = net.state_dict()[k]
            v2: torch.Tensor = net2.state_dict()[k]
            self.assertTrue((v1 == v2).all().item())



