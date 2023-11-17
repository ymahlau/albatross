import unittest

import torch

from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedSlowConfig, AsymmetricAdvantageOvercookedSlowConfig, \
    CoordinationRingOvercookedSlowConfig, ForcedCoordinationOvercookedSlowConfig, CounterCircuitOvercookedSlowConfig
from src.network.initialization import get_network_from_config
from src.network.mobilenet_v3 import MobileNetConfigOvercookedCramped, MobileNetConfigOvercookedAsymmetricAdvantage, \
    MobileNetConfigOvercookedCoordinationRing, MobileNetConfigOvercookedForcedCoordination, \
    MobileNetConfigOvercookedCounterCircuit


class TestMobileNetOvercooked(unittest.TestCase):
    def test_cramped_room(self):
        game_cfg = CrampedRoomOvercookedSlowConfig()
        game = get_game_from_config(game_cfg)

        net_cfg = MobileNetConfigOvercookedCramped(game_cfg=game_cfg)
        net = get_network_from_config(net_cfg)
        print(f"{net.num_params()}")
        obs, _, _ = game.get_obs()
        net_out = net(torch.tensor(obs))
        self.assertEqual(2, obs.shape[0])

    def test_asymmetric_adv(self):
        game_cfg = AsymmetricAdvantageOvercookedSlowConfig()
        game = get_game_from_config(game_cfg)
        game.render()
        print(f"{game.get_obs_shape()=}")

        net_cfg = MobileNetConfigOvercookedAsymmetricAdvantage(game_cfg=game_cfg)
        net = get_network_from_config(net_cfg)
        print(f"{net.num_params()}")
        obs, _, _ = game.get_obs()
        net_out = net(torch.tensor(obs))
        print(f"{net_out=}")
        self.assertEqual(2, obs.shape[0])

    def test_coord_ring(self):
        game_cfg = CoordinationRingOvercookedSlowConfig()
        game = get_game_from_config(game_cfg)
        game.render()
        print(f"{game.get_obs_shape()=}")

        net_cfg = MobileNetConfigOvercookedCoordinationRing(game_cfg=game_cfg)
        net = get_network_from_config(net_cfg)
        print(f"{net.num_params()}")
        obs, _, _ = game.get_obs()
        net_out = net(torch.tensor(obs))
        print(f"{net_out=}")
        self.assertEqual(2, obs.shape[0])

    def test_forced_coord(self):
        game_cfg = ForcedCoordinationOvercookedSlowConfig()
        game = get_game_from_config(game_cfg)
        game.render()
        print(f"{game.get_obs_shape()=}")

        net_cfg = MobileNetConfigOvercookedForcedCoordination(game_cfg=game_cfg)
        net = get_network_from_config(net_cfg)
        print(f"{net.num_params()}")
        obs, _, _ = game.get_obs()
        net_out = net(torch.tensor(obs))
        print(f"{net_out=}")
        self.assertEqual(2, obs.shape[0])

    def test_counter_circuit(self):
        game_cfg = CounterCircuitOvercookedSlowConfig()
        game = get_game_from_config(game_cfg)
        game.render()
        print(f"{game.get_obs_shape()=}")

        net_cfg = MobileNetConfigOvercookedCounterCircuit(game_cfg=game_cfg)
        net = get_network_from_config(net_cfg)
        print(f"{net.num_params()}")
        obs, _, _ = game.get_obs()
        net_out = net(torch.tensor(obs))
        print(f"{net_out=}")
        self.assertEqual(2, obs.shape[0])
