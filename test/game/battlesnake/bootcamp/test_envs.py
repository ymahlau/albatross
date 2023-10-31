import unittest
import random

import numpy as np

from src.game.battlesnake.battlesnake import UP, DOWN, BattleSnakeGame, LEFT, RIGHT
from src.game.battlesnake.bootcamp import die_on_a_hill
from src.game.battlesnake.bootcamp import cooperation_choke_5x5, randomization
from src.game.battlesnake.bootcamp import cooperation_7x7
from src.game.initialization import get_game_from_config


class TestEnvs(unittest.TestCase):
    def test_die_on_a_hill(self):
        game_cfg = die_on_a_hill()
        game = get_game_from_config(game_cfg)
        game.render()
        game.step((0, 2))
        game.render()
        self.assertEqual(2, game.num_players_alive())
        game.step((1, 3))
        self.assertEqual(0, game.num_players_alive())
        game.render()
        self.assertTrue(game.is_terminal())

    def test_cooperation_choke_5x5(self):
        game_cfg = cooperation_choke_5x5()
        game = BattleSnakeGame(game_cfg)
        game.render()
        game.step((UP, UP, DOWN, UP))
        game.render()
        self.assertEqual(4, game.num_players_alive())
        game.step((0, 0, 2, 0))
        game.render()
        self.assertEqual(2, game.num_players_alive())

    def test_randomization(self):
        game_cfg = randomization()
        game = BattleSnakeGame(game_cfg)
        game.render()
        game.step((UP, UP))
        game.render()
        game.step((UP, LEFT))
        game.render()
        self.assertEqual(2, game.num_players_alive())
        rewards, _, _ = game.step((UP, LEFT))
        self.assertTrue(game.is_terminal())
        self.assertEqual(1, rewards[0])
        self.assertEqual(-1, rewards[-1])

    def test_randomization_swap(self):
        game_cfg = randomization()
        game = BattleSnakeGame(game_cfg)
        game.render()
        game.step((UP, DOWN))
        game.render()
        game.step((RIGHT, DOWN))
        game.render()
        self.assertEqual(2, game.num_players_alive())
        game.step((DOWN, DOWN))
        game.render()
        self.assertFalse(game.is_terminal())
        rewards, _, _ = game.step((DOWN, LEFT))
        game.render()
        self.assertEqual(0, rewards[0])
        self.assertEqual(0, rewards[-1])

    def test_cooperation(self):
        game_cfg = cooperation_7x7()
        game = BattleSnakeGame(game_cfg)
        game.reset()

        game.render()
        game.step((0, 0))
        game.render()

        print(f"\n\nTesting different starting position:")
        for _ in range(10):
            game.reset()
            game.render()

    def test_cooperation_reward(self):
        game_cfg = cooperation_7x7()
        game = BattleSnakeGame(game_cfg)
        game.reset()
        game.render()

        while not game.is_terminal():
            cum_r = game.get_cum_rewards()
            self.assertTrue(np.all(cum_r >= 0))
            cur_r, _, _ = game.step(random.choice(game.available_joint_actions()))
            game.render()
            print(f"{cur_r=}")
            print('----------------------')
            if game.is_terminal():
                self.assertTrue(np.all(cur_r < 0))


