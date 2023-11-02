import time
import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.bootcamp.test_envs_11x11 import survive_on_11x11
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7
from src.game.initialization import get_game_from_config
from src.network.initialization import get_network_from_config
from src.network.resnet import ResNetConfig3x3, ResNetConfig5x5, ResNetConfig7x7, ResNetConfig9x9, ResNetConfig7x7New, \
    ResNetConfig11x11, ResNetConfig7x7Best


class TestResNet(unittest.TestCase):
    def test_resnet_game(self):
        gc = BattleSnakeConfig(
            w=3,
            h=3,
            num_players=4,
        )
        gc.ec.flatten = False
        game = BattleSnakeGame(gc)
        net_conf = ResNetConfig3x3(
            game_cfg=gc,
            predict_policy=True,
            predict_game_len=False,
        )
        net = get_network_from_config(net_conf)
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(4, out.shape[0])
        self.assertEqual(5, out.shape[1])

    def test_resnet_game_len(self):
        gc = BattleSnakeConfig(
            w=3,
            h=3,
            num_players=4,
        )
        gc.ec.flatten = False
        game = BattleSnakeGame(gc)
        net_conf = ResNetConfig3x3(
            game_cfg=gc,
            predict_policy=True,
            predict_game_len=True,
        )
        net = get_network_from_config(net_conf)
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(4, out.shape[0])
        self.assertEqual(6, out.shape[1])

    def test_value_resnet(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = ResNetConfig3x3(game_cfg=game_cfg, predict_policy=False, predict_game_len=False)
        net = get_network_from_config(net_cfg)
        game = get_game_from_config(game_cfg)
        obs, _, _, = game.get_obs()
        out = net(obs)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(1, out.shape[1])
        value = net.retrieve_value(out)
        self.assertEqual(1, len(value.shape))
        self.assertEqual(2, value.shape[0])

    def test_resnet_centered_5(self):
        gc = BattleSnakeConfig(
            w=5,
            h=5,
            num_players=4,
        )
        gc.ec.flatten = False
        gc.ec.centered = True
        game = BattleSnakeGame(gc)
        net_conf = ResNetConfig5x5(
            game_cfg=gc,
            predict_policy=True,
            predict_game_len=True,
        )
        net = get_network_from_config(net_conf)
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(4, out.shape[0])
        self.assertEqual(6, out.shape[1])

    def test_resnet_centered_7(self):
        gc = BattleSnakeConfig(
            w=7,
            h=7,
            num_players=4,
        )
        gc.ec.flatten = False
        gc.ec.centered = True
        game = BattleSnakeGame(gc)
        net_conf = ResNetConfig7x7(
            game_cfg=gc,
            predict_policy=True,
            predict_game_len=True,
        )
        net = get_network_from_config(net_conf)
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(4, out.shape[0])
        self.assertEqual(6, out.shape[1])

    def test_resnet_new_7x7(self):
        game_cfg = survive_on_7x7()
        game = get_game_from_config(game_cfg)
        game.render()
        net_conf = ResNetConfig7x7New(game_cfg=game_cfg)
        net = get_network_from_config(net_conf)
        print(f"{net.num_params()=}")
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        print(f"{out=}")
        self.assertEqual(2, len(out.shape))
        self.assertEqual(2, out.shape[0])
        self.assertEqual(5, out.shape[1])

    def test_resnet_best_7x7(self):
        game_cfg = survive_on_7x7()
        game = get_game_from_config(game_cfg)
        game.render()
        net_conf = ResNetConfig7x7Best(game_cfg=game_cfg)
        net = get_network_from_config(net_conf)
        print(f"{net.num_params()=}")
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        print(f"{out=}")
        self.assertEqual(2, len(out.shape))
        self.assertEqual(2, out.shape[0])
        self.assertEqual(5, out.shape[1])

    def test_resnet_centered_9(self):
        gc = BattleSnakeConfig(
            w=9,
            h=9,
            num_players=4,
        )
        gc.ec.flatten = False
        gc.ec.centered = True
        game = BattleSnakeGame(gc)
        net_conf = ResNetConfig9x9(
            game_cfg=gc,
            predict_policy=True,
            predict_game_len=True,
        )
        net = get_network_from_config(net_conf)
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(4, out.shape[0])
        self.assertEqual(6, out.shape[1])

    def test_resnet_centered_11(self):
        gc = survive_on_11x11()
        gc.ec.flatten = False
        gc.ec.centered = True
        game = BattleSnakeGame(gc)
        net_conf = ResNetConfig11x11(
            game_cfg=gc,
            predict_policy=True,
        )
        net = get_network_from_config(net_conf)
        print(f"{net.num_params()=}")
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(2, out.shape[0])
        self.assertEqual(5, out.shape[1])

    def test_resnet_centered_11_speed(self):
        gc = survive_on_11x11()
        gc.ec.flatten = False
        gc.ec.centered = True
        game = BattleSnakeGame(gc)
        net_conf = ResNetConfig11x11(
            game_cfg=gc,
            predict_policy=True,
        )
        net = get_network_from_config(net_conf)
        print(f"{net.num_params()=}")
        in_tensor, _, _ = game.get_obs()
        n = 10
        start_time = time.time()
        for _ in range(n):
            out = net(in_tensor)
        end_time = time.time()
        print(f"{(end_time - start_time) / n=}")

