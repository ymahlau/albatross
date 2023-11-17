import unittest

import torch

from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.initialization import get_game_from_config
from src.network.initialization import get_network_from_config
from src.network.resnet import ResNetConfig3x3
from src.network.vision_net import EquivarianceType


class TestPooled(unittest.TestCase):
    def test_pooled_resnet_no_policy(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = ResNetConfig3x3(
            game_cfg=game_cfg,
            predict_policy=False,
            eq_type=EquivarianceType.POOLED,
        )
        net = get_network_from_config(net_cfg)
        net.eval()
        game = get_game_from_config(game_cfg)

        out_list = []
        for symmetry in range(8):
            obs, _, _, = game.get_obs(symmetry)
            out = net(torch.tensor(obs))
            out_list.append(out)
        for out in out_list[1:]:
            for player in range(2):
                self.assertAlmostEqual(out_list[0][player].item(), out[player].item(), places=5)
                self.assertAlmostEqual(out_list[0][player].item(), out[player].item(), places=5)

    def test_pooled_resnet_with_policy(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = ResNetConfig3x3(
            game_cfg=game_cfg,
            predict_policy=True,
            eq_type=EquivarianceType.POOLED,
        )
        net = get_network_from_config(net_cfg)
        net = net.eval()
        game = get_game_from_config(game_cfg)

        out_list = []
        for symmetry in range(8):
            obs, _, _, = game.get_obs(symmetry)
            out = net(torch.tensor(obs))
            out_list.append(out)
        for out in out_list[1:]:
            for player in range(2):
                self.assertAlmostEqual(out_list[0][player][-1].item(), out[player][-1].item(), places=5)

