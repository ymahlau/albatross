import math
import time
import unittest

from src.game.actions import sample_individual_actions
from src.game.battlesnake.bootcamp.test_envs_11x11 import survive_on_11x11
from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedSlowConfig
from src.search.config import MCTSConfig, DecoupledUCTSelectionConfig, \
    StandardBackupConfig, StandardExtractConfig, DummyEvalConfig
from src.search.initialization import get_search_from_config


class TestOvercookedEval(unittest.TestCase):
    def test_overcooked_eval_cramped(self):
        game_cfg = CrampedRoomOvercookedSlowConfig()
        game_cfg.horizon = 30
        game = get_game_from_config(game_cfg)

        search_cfg = MCTSConfig(
            eval_func_cfg=DummyEvalConfig(),
            sel_func_cfg=DecoupledUCTSelectionConfig(),
            backup_func_cfg=StandardBackupConfig(),
            extract_func_cfg=StandardExtractConfig(),
            discount=0.99,
        )
        search = get_search_from_config(search_cfg)
        game.render()

        while not game.is_terminal():
            values, action_probs, _ = search(game, iterations=500)
            print(f"{values=}")
            print(f"{action_probs=}")
            ja = sample_individual_actions(action_probs, math.inf)
            game.step(ja)
            game.render()

    def test_overcooked_dummy_eval(self):
        game_cfg = CrampedRoomOvercookedSlowConfig()
        game = get_game_from_config(game_cfg)
        game_cfg2 = survive_on_11x11()
        game2 = get_game_from_config(game_cfg2)

        search_cfg = MCTSConfig(
            eval_func_cfg=DummyEvalConfig(),
            sel_func_cfg=DecoupledUCTSelectionConfig(),
            backup_func_cfg=StandardBackupConfig(),
            extract_func_cfg=StandardExtractConfig(),
            discount=0.99,
        )
        search = get_search_from_config(search_cfg)
        game.render()

        start_time = time.time()
        for _ in range(10):
            values, action_probs, _ = search(game, iterations=500)
            print(f"{values=}")
            print(f"{action_probs=}")
            ja = sample_individual_actions(action_probs, math.inf)
            game.step(ja)
            # game.render()
        overcooked_duration = time.time() - start_time
        print(f"{overcooked_duration=}")

        start_time = time.time()
        for _ in range(10):
            values, action_probs, _ = search(game2, iterations=500)
            print(f"{values=}")
            print(f"{action_probs=}")
            ja = sample_individual_actions(action_probs, math.inf)
            game2.step(ja)
            # game2.render()
        battlesnake_duration = time.time() - start_time
        print(f"{battlesnake_duration=}")

    def test_overcooked_search_iterations(self):
        game_cfg = CrampedRoomOvercookedSlowConfig()
        game = get_game_from_config(game_cfg)

        reward, _, _ = game.step((0, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 5))
        game.render()
        print(f"{reward}\n######################################")

        # reward, _, _ = game.step((5, 3))
        # game.render()
        # print(f"{reward}\n######################################")

        # reward, _, _ = game.step((4, 0))
        # game.render()
        # print(f"{reward}\n######################################")

        # reward, _, _ = game.step((4, 5))
        # game.render()
        # print(f"{reward}\n######################################")

        search_cfg = MCTSConfig(
            eval_func_cfg=DummyEvalConfig(),
            sel_func_cfg=DecoupledUCTSelectionConfig(exp_bonus=0.5),
            backup_func_cfg=StandardBackupConfig(),
            extract_func_cfg=StandardExtractConfig(),
            discount=0.99,
        )
        search = get_search_from_config(search_cfg)

        start_time = time.time()
        values, action_probs, _ = search(game, iterations=500)
        duration = time.time() - start_time
        print(f"{duration=}")
        print(f"{values=}")
        print(f"{action_probs=}")

