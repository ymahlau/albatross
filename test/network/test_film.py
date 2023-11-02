import unittest

import torch

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.battlesnake.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player
from src.network.fcn import MediumHeadConfig
from src.network.initialization import get_network_from_config
from src.network.mobilenet_v3 import MobileNetConfig3x3, MobileNetConfig5x5
from src.network.utils import ActivationType


class TestFilm(unittest.TestCase):
    def test_film_choke(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = MobileNetConfig3x3(
            game_cfg=game_cfg,
            film_temperature_input=True,
            film_cfg=MediumHeadConfig(final_activation=ActivationType.LEAKY_RELU),
        )
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)
        print(f"{net.num_params()=}")
        temperature = torch.tensor([[1], [2]], dtype=torch.float32)
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor, temperature)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(2, out.shape[0])
        self.assertEqual(5, out.shape[1])

    def test_film_multiplayer(self):
        game_cfg = perform_choke_5x5_4_player(centered=True)
        net_cfg = MobileNetConfig5x5(
            game_cfg=game_cfg,
            film_temperature_input=True,
            film_cfg=MediumHeadConfig(final_activation=ActivationType.LEAKY_RELU),
        )
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)
        print(f"{net.num_params()=}")
        temperature = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor, temperature)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(4, out.shape[0])
        self.assertEqual(5, out.shape[1])
