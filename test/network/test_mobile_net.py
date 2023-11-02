import time
import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_11x11 import survive_on_11x11
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.battlesnake.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player, survive_on_5x5_constrictor
from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7
from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedConfig, AsymmetricAdvantageOvercookedConfig
from src.network.initialization import get_network_from_config
from src.network.mobilenet_v3 import MobileNetConfig3x3, MobileNetConfig7x7, MobileNetConfig5x5, \
    MobileNetConfig5x5Large, MobileNetConfig11x11, MobileNetConfig11x11Extrapolated, MobileNetConfig5x5Extrapolated, \
    MobileNetConfig7x7Incumbent, MobileNetConfigOvercookedCramped, MobileNetConfigOvercookedAsymmetricAdvantage
from src.network.resnet import ResNetConfig3x3, ResNetConfig7x7Large
from src.network.vision_net import EquivarianceType


class TestMobileNet(unittest.TestCase):

    def test_mobile_choke(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = MobileNetConfig3x3(game_cfg=game_cfg)
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)
        print(f"{net.num_params()=}")
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(2, out.shape[0])
        self.assertEqual(5, out.shape[1])

    def test_mobile_choke_game_len(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = MobileNetConfig3x3(game_cfg=game_cfg, predict_game_len=True)
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)
        print(f"{net.num_params()=}")
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        self.assertEqual(2, len(out.shape))
        self.assertEqual(2, out.shape[0])
        self.assertEqual(6, out.shape[1])

    def test_latency(self):  # unfair comparison due to parameter count
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = MobileNetConfig3x3(game_cfg=game_cfg)
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)
        resnet_cfg = ResNetConfig3x3(game_cfg=game_cfg)
        resnet = get_network_from_config(resnet_cfg)
        in_tensor, _, _ = game.get_obs()
        # mobile
        mobile_start = time.time()
        net(in_tensor)
        mobile_time = time.time() - mobile_start
        # resnet
        resnet_start = time.time()
        resnet(in_tensor)
        resnet_time = time.time() - resnet_start
        print(f"{mobile_time=}")
        print(f"{resnet_time=}")

    def test_5x5(self):
        game_cfg = perform_choke_5x5_4_player(centered=True)
        net_cfg = MobileNetConfig5x5(game_cfg=game_cfg)
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)

        in_tensor, _, _ = game.get_obs()
        print(f"{net.num_params()=}")
        # mobile
        mobile_start = time.time()
        net(in_tensor)
        mobile_time = time.time() - mobile_start
        print(f"{mobile_time=}")

    def test_5x5_large(self):
        game_cfg = perform_choke_5x5_4_player(centered=True)
        net_cfg = MobileNetConfig5x5Large(game_cfg=game_cfg, predict_policy=False, predict_game_len=True,
                                          eq_type=EquivarianceType.CONSTRAINED)
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)

        in_tensor, _, _ = game.get_obs()
        print(f"{net.num_params()=}")
        # mobile
        mobile_start = time.time()
        out = net(in_tensor)
        mobile_time = time.time() - mobile_start
        print(f"{mobile_time=}")

    def test_7x7(self):
        game_cfg = survive_on_7x7()
        net_cfg = MobileNetConfig7x7(game_cfg=game_cfg)
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)
        resnet_cfg = ResNetConfig7x7Large(game_cfg=game_cfg)
        resnet = get_network_from_config(resnet_cfg)
        in_tensor, _, _ = game.get_obs()
        print(f"{net.num_params()=}")
        print(f"{resnet.num_params()=}")
        # mobile
        mobile_start = time.time()
        net(in_tensor)
        mobile_time = time.time() - mobile_start
        # resnet
        resnet_start = time.time()
        resnet(in_tensor)
        resnet_time = time.time() - resnet_start
        print(f"{mobile_time=}")
        print(f"{resnet_time=}")

    def test_11x11(self):
        game_cfg = survive_on_11x11()
        net_cfg = MobileNetConfig11x11(game_cfg=game_cfg)
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)

        in_tensor, _, _ = game.get_obs()
        print(f"{net.num_params()=}")
        # mobile
        mobile_start = time.time()
        net(in_tensor)
        mobile_time = time.time() - mobile_start
        print(f"{mobile_time=}")

    def test_11x11_extrapolated(self):
        game_cfg = survive_on_11x11()
        net_cfg = MobileNetConfig11x11Extrapolated(game_cfg=game_cfg)
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)

        in_tensor, _, _ = game.get_obs()
        print(f"{net.num_params()=}")
        # mobile
        mobile_start = time.time()
        for _ in range(10):
            net(in_tensor)
        mobile_time = time.time() - mobile_start
        print(f"{mobile_time/10=}")

    def test_5x5_extrapolated(self):
        game_cfg = survive_on_5x5_constrictor()
        net_cfg = MobileNetConfig5x5Extrapolated(game_cfg=game_cfg)
        game = BattleSnakeGame(game_cfg)
        net = get_network_from_config(net_cfg)

        in_tensor, _, _ = game.get_obs()
        print(f"{net.num_params()=}")
        # mobile
        mobile_start = time.time()
        for _ in range(10):
            net(in_tensor)
        mobile_time = time.time() - mobile_start
        print(f"{mobile_time/10=}")

    def test_mobile_incumbent_7x7(self):
        game_cfg = survive_on_7x7()
        game = get_game_from_config(game_cfg)
        game.render()
        net_conf = MobileNetConfig7x7Incumbent(game_cfg=game_cfg)
        net = get_network_from_config(net_conf)
        print(f"{net.num_params()=}")
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        print(f"{out=}")
        self.assertEqual(2, len(out.shape))
        self.assertEqual(2, out.shape[0])
        self.assertEqual(5, out.shape[1])

    def test_mobile_overcooked_cramped(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = get_game_from_config(game_cfg)
        game.render()
        net_conf = MobileNetConfigOvercookedCramped(game_cfg=game_cfg)
        net = get_network_from_config(net_conf)
        print(f"{net.num_params()=}")
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        print(f"{out=}")

    def test_mobile_overcooked_asym(self):
        game_cfg = AsymmetricAdvantageOvercookedConfig()
        game = get_game_from_config(game_cfg)
        game.render()
        net_conf = MobileNetConfigOvercookedAsymmetricAdvantage(game_cfg=game_cfg)
        net = get_network_from_config(net_conf)
        print(f"{net.num_params()=}")
        in_tensor, _, _ = game.get_obs()
        out = net(in_tensor)
        print(f"{out=}")

