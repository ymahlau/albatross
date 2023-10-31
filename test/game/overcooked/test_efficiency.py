import time
import unittest

from omegaconf import OmegaConf

from src.game.battlesnake.bootcamp import survive_on_11x11
from src.game.initialization import get_game_from_config, game_config_from_structured
from src.game.overcooked.layouts import CrampedRoomOvercookedConfig


class TestEfficiency(unittest.TestCase):
    def test_compare_copy(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg2 = survive_on_11x11()
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
        duration_battlesnake = time.time() - start_time
        print(f"{duration_battlesnake=}")

    def test_compare_step_reset(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg2 = survive_on_11x11()
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
        duration_battlesnake = time.time() - start_time
        print(f"{duration_battlesnake=}")

    def test_compare_get_obs(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg2 = survive_on_11x11()
        game = get_game_from_config(game_cfg)
        game2 = get_game_from_config(game_cfg2)

        num_samples = 1000

        start_time = time.time()
        for _ in range(num_samples):
            _ = game.get_obs()
            game.obs_save = None
        duration_overcooked = time.time() - start_time
        print(f"{duration_overcooked=}")

        start_time = time.time()
        for _ in range(num_samples):
            _ = game2.get_obs()
            game2.obs_save = None
        duration_battlesnake = time.time() - start_time
        print(f"{duration_battlesnake=}")

    def test_hydra_speed(self):
        game_cfg = CrampedRoomOvercookedConfig()
        structured = OmegaConf.structured(game_cfg)
        new_cfg = game_config_from_structured(structured)

        game = get_game_from_config(game_cfg)
        new_game = get_game_from_config(new_cfg)

        start_time = time.time()
        for _ in range(1000):
            _ = game.get_copy()
        delta = time.time() - start_time
        print(f"original: {delta}")

        start_time = time.time()
        for _ in range(1000):
            _ = new_game.get_copy()
        delta = time.time() - start_time
        print(f"converted: {delta}")
