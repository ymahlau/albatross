import unittest

from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedConfig


class TestObservationSpace(unittest.TestCase):
    def test_obs_input(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg.temperature_input = True
        game_cfg.single_temperature_input = True
        game = get_game_from_config(game_cfg)

        obs, _, _ = game.get_obs(temperatures=[5])
        print(obs.shape)
        self.assertEqual(24, obs.shape[3])

    def test_obs_multiple_input(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg.temperature_input = True
        game_cfg.single_temperature_input = False
        game = get_game_from_config(game_cfg)

        obs, _, _ = game.get_obs(temperatures=[2, 5])
        print(obs.shape)
        self.assertEqual(24, obs.shape[3])

    def test_obs_multiple_input_argument(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg.temperature_input = True
        game_cfg.single_temperature_input = True
        game = get_game_from_config(game_cfg)

        obs, _, _ = game.get_obs(temperatures=[2, 5], single_temperature=False)
        print(obs.shape)
        self.assertEqual(24, obs.shape[3])

    def test_flat_obs(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg.flat_obs = True
        game = get_game_from_config(game_cfg)
        obs_shape = game.get_obs_shape()
        print(f"{obs_shape=}")
        obs, _, _ = game.get_obs()
        print(f"{obs.shape=}")

