import unittest

from src.game.initialization import get_game_from_config
from src.game.overcooked.config import CrampedRoomOvercookedConfig, AsymmetricAdvantageOvercookedConfig, \
    CoordinationRingOvercookedConfig, ForcedCoordinationOvercookedConfig, CounterCircuitOvercookedConfig


class TestLayouts(unittest.TestCase):
    def test_cramped_room(self):
        cfg = CrampedRoomOvercookedConfig()
        game = get_game_from_config(cfg)
        obs, _, _ = game.get_obs()
        print(obs.shape)

    def test_aa(self):
        cfg = AsymmetricAdvantageOvercookedConfig()
        game = get_game_from_config(cfg)
        obs, _, _ = game.get_obs()
        print(obs.shape)

    def test_cr(self):
        cfg = CoordinationRingOvercookedConfig()
        game = get_game_from_config(cfg)
        obs, _, _ = game.get_obs()
        print(obs.shape)

    def test_fc(self):
        cfg = ForcedCoordinationOvercookedConfig()
        game = get_game_from_config(cfg)
        obs, _, _ = game.get_obs()
        print(obs.shape)

    def test_cc(self):
        cfg = CounterCircuitOvercookedConfig()
        game = get_game_from_config(cfg)
        obs, _, _ = game.get_obs()
        print(obs.shape)
