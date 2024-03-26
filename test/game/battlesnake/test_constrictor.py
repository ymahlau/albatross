import unittest

import numpy as np

from src.game.battlesnake.battlesnake import BattleSnakeGame, UP, RIGHT, LEFT, DOWN
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.bootcamp.test_envs_9x9 import survive_on_9x9_constrictor_4_player_coop


class TestConstrictor(unittest.TestCase):
    def test_constrictor_simple(self):
        game_conf = BattleSnakeConfig(
            w=5,
            h=5,
            num_players=2,
            all_actions_legal=True,
            constrictor=True,
        )
        game = BattleSnakeGame(game_conf)
        game.render()
        self.assertEqual(0, game.num_food())
        self.assertEqual(16, len(game.available_joint_actions()))

        game.step((UP, UP))
        game.render()
        self.assertEqual(2, game.num_players_alive())

    def test_long_game(self):
        init_snake_pos = {0: [[0, 0]], 1: [[4, 0]]}
        game_conf = BattleSnakeConfig(
            w=5,
            h=5,
            num_players=2,
            all_actions_legal=False,
            constrictor=True,
            init_snake_pos=init_snake_pos,
        )
        game = BattleSnakeGame(game_conf)
        game.render()

        for _ in range(4):
            game.step((UP, UP))
            game.render()
            self.assertEqual(2, game.num_players_alive())
        game.step((RIGHT, LEFT))
        game.render()
        self.assertEqual(2, game.num_players_alive())
        for _ in range(4):
            game.step((DOWN, DOWN))
            game.render()
            self.assertEqual(2, game.num_players_alive())
        self.assertEqual(1, len(game.available_joint_actions()))
        game.step((RIGHT, LEFT))
        game.render()
        self.assertEqual(0, game.num_players_alive())
        self.assertTrue(game.is_terminal())

    def test_bool_game_array(self):
        init_snake_pos = {0: [[0, 0]], 1: [[4, 0], [4, 1], [2, 2]]}
        game_conf = BattleSnakeConfig(
            w=5,
            h=5,
            num_players=2,
            all_actions_legal=False,
            constrictor=True,
            init_snake_pos=init_snake_pos,
        )
        game = BattleSnakeGame(game_conf)
        game.render()

        arr = game.get_bool_board_matrix()
        print(arr)
        self.assertEqual(7, np.sum(arr))
        
    def test_coop_9x9(self):
        game_cfg = survive_on_9x9_constrictor_4_player_coop()
        game_cfg.all_actions_legal = True
        game = BattleSnakeGame(game_cfg)
        game.render()
        
        r, _, _ = game.step((UP, UP, UP, UP))
        game.render()
        print(f"{r=}")
        
        r, _, _ = game.step((UP, UP, UP, UP))
        game.render()
        print(f"{r=}")

