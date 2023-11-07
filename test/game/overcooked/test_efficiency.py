import time
import unittest

from src.game.initialization import get_game_from_config
from src.game.overcooked.config import CrampedRoomOvercookedConfig
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedSlowConfig

class TestEfficiency(unittest.TestCase):
    def test_compare_copy(self):
        game_cfg = CrampedRoomOvercookedSlowConfig()
        game_cfg2 = CrampedRoomOvercookedConfig()
        game = get_game_from_config(game_cfg)
        game2 = get_game_from_config(game_cfg2)

        num_samples = 1000

        start_time = time.time()
        for _ in range(num_samples):
            cpy = game.get_copy()
            cpy.step((0, 0))
            # cpy.reset()
            # cpy.step((0, 0))
        duration_overcooked = time.time() - start_time
        print(f"{duration_overcooked=}")

        start_time = time.time()
        for _ in range(num_samples):
            cpy2 = game2.get_copy()
            cpy2.step((0, 0))
            cpy2.reset()
            cpy2.step((0, 0))
        duration_fast = time.time() - start_time
        print(f"{duration_fast=}")

    def test_compare_step_reset(self):
        game_cfg = CrampedRoomOvercookedSlowConfig()
        game_cfg2 = CrampedRoomOvercookedConfig()
        game = get_game_from_config(game_cfg)
        game2 = get_game_from_config(game_cfg2)

        num_samples = 1000

        start_time = time.time()
        for _ in range(num_samples):
            _ = game.step((0, 0))
            game.reset()
        duration_overcooked = time.time() - start_time
        print(f"{duration_overcooked=}")

        start_time = time.time()
        for _ in range(num_samples):
            _ = game2.step((0, 0))
            game2.reset()
        duration_fast = time.time() - start_time
        print(f"{duration_fast=}")
