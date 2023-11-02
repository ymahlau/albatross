import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.search import Node
from src.search.config import CopyCatEvalConfig
from src.search.eval_func import CopyCatEvalFunc


class TestEvalFunction(unittest.TestCase):
    def test_copy_cat(self):
        game_cfg = BattleSnakeConfig(
            num_players=4,
            w=11,
            h=11,
            init_snake_len=[3, 1, 5, 2],
            init_snake_health=[50, 25, 100, 10],
            init_snakes_alive=[True, True, False, True],
            init_snake_pos={
                0: [[0, 0], [1, 0], [2, 0]],
                1: [[5, 5]],
                2: [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
                3: [[8, 8], [8, 9]]
            },
            init_food_pos=[[4, 4], [3, 2], [1, 1]],
        )
        game = BattleSnakeGame(game_cfg)
        game.render()
        node = Node(
            game=game,
            parent=None,
            last_actions=None,
            discount=0.99,
            ignore_full_exploration=True,
        )
        eval_cfg = CopyCatEvalConfig()
        eval_func = CopyCatEvalFunc(eval_cfg)
        eval_func([node])
        print(node.value_sum)
