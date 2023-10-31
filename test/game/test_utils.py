import math
import unittest

import numpy as np

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.utils import int_to_perm, step_with_draw_prevention


class TestUtils(unittest.TestCase):
    def test_int_perm_small(self):
        n = 5
        fac = math.factorial(n)
        lst = []
        for i in range(fac):
            perm = int_to_perm(i, n)
            lst.append(perm)
        for i in range(fac):
            for j in range(i+1, fac):
                self.assertFalse((lst[i] == lst[j]).all())

    def test_zero_mapping(self):
        # seed zero should always map to identity
        for n in range(2, 20):
            perm = int_to_perm(0, n)
            self.assertTrue((perm == np.arange(n)).all())

    def test_draw_prevention(self):
        game_cfg = BattleSnakeConfig(
            w=5,
            h=5,
            num_players=2,
            init_snake_pos={0: [[2, 2]], 1: [[3, 3]]},
            init_food_pos=[],
            min_food=0,
        )
        game = BattleSnakeGame(game_cfg)
        game.render()
        cpy = game.get_copy()
        cpy.step((0, 3))
        cpy.render()
        self.assertEqual(0, cpy.num_players_at_turn())
        rewards = step_with_draw_prevention(game, (0, 3))
        game.render()
        self.assertEqual(2, game.num_players_at_turn())
        self.assertEqual(0, rewards[0])
        self.assertEqual(0, rewards[1])


