import unittest

import torch

from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7
from src.game.initialization import get_game_from_config
from src.network.initialization import get_network_from_config
from src.network.mobile_one import MobileOneConfig3x3, MobileOneConfig7x7, reparameterize_model, \
    MobileOneIncumbentConfig7x7


class TestMobileOneNetwork(unittest.TestCase):
    def test_mobile_one_choke(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        game = get_game_from_config(game_cfg)
        net_cfg = MobileOneConfig3x3(game_cfg=game_cfg, predict_policy=True)
        net = get_network_from_config(net_cfg).eval()
        print(f"{net.num_params()=}")

        obs, _, _ = game.get_obs()
        net_out = net(torch.tensor(obs))
        self.assertEqual(2, len(net_out.shape))
        self.assertEqual(2, net_out.shape[0])
        self.assertEqual(5, net_out.shape[1])

    def test_mobile_one_7x7(self):
        game_cfg = survive_on_7x7()
        game = get_game_from_config(game_cfg)
        net_cfg = MobileOneConfig7x7(game_cfg=game_cfg, predict_policy=True)
        net = get_network_from_config(net_cfg).train()
        print(f"{net.num_params()=}")

        obs, _, _ = game.get_obs()
        net_out = net(torch.tensor(obs))
        self.assertEqual(2, len(net_out.shape))
        self.assertEqual(2, net_out.shape[0])
        self.assertEqual(5, net_out.shape[1])

    def test_parameterization(self):
        game_cfg = survive_on_7x7()
        game = get_game_from_config(game_cfg)
        net_cfg = MobileOneConfig7x7(game_cfg=game_cfg, predict_policy=True)
        net = get_network_from_config(net_cfg).eval()
        print(f"{net.num_params()=}")

        obs, _, _ = game.get_obs()
        net_out: torch.Tensor = net(torch.tensor(obs))

        new_net = reparameterize_model(net)
        new_net_out: torch.Tensor = new_net(torch.tensor(obs))

        self.assertTrue(torch.all(torch.abs(net_out - new_net_out) < 0.001).item())

    def test_mobile_one_incumbent(self):
        game_cfg = survive_on_7x7()
        game = get_game_from_config(game_cfg)
        game.render()
        net_cfg = MobileOneIncumbentConfig7x7(game_cfg=game_cfg, predict_policy=True)
        net = get_network_from_config(net_cfg).eval()
        print(f"{net.num_params()=}")

        obs, _, _ = game.get_obs()
        net_out = net(torch.tensor(obs))
        print(f"{net_out=}")
        self.assertEqual(2, net_out.shape[0])
