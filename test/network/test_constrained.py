import time
import unittest

import torch

from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.initialization import get_game_from_config
from src.network.initialization import get_network_from_config
from src.network.mobilenet_v3 import MobileNetConfig3x3
from src.network.resnet import ResNetConfig3x3
from src.network.vision_net import EquivarianceType


class TestConstrained(unittest.TestCase):
    def test_constrained_resnet(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = ResNetConfig3x3(
            game_cfg=game_cfg,
            predict_policy=False,
            eq_type=EquivarianceType.CONSTRAINED,
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
                self.assertAlmostEqual(out_list[0][player][0].item(), out[player][0].item(), places=5)

    def test_constrained_mobile_net(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = MobileNetConfig3x3(
            game_cfg=game_cfg,
            predict_policy=False,
            eq_type=EquivarianceType.CONSTRAINED,
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
                self.assertAlmostEqual(out_list[0][player][0].item(), out[player][0].item(), places=5)

    def test_speed(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = ResNetConfig3x3(
            game_cfg=game_cfg,
            predict_policy=False,
            eq_type=EquivarianceType.CONSTRAINED,
        )
        net = get_network_from_config(net_cfg)
        game = get_game_from_config(game_cfg)
        obs, _, _, = game.get_obs()
        num_samples = 100
        net.train()
        # no export
        start_train = time.time()
        for _ in range(num_samples):
            out = net(torch.tensor(obs))
        train_time = time.time() - start_train
        print(f"{train_time=}")
        # with export
        net.eval()
        start_eval = time.time()
        for _ in range(num_samples):
            out = net(obs)
        eval_time = time.time() - start_eval
        print(f"{eval_time=}")

