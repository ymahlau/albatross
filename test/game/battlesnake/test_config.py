import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeDuelsConfig
from src.game.battlesnake.battlesnake_enc import SimpleBattleSnakeEncodingConfig


class TestBattleSnakeConfig(unittest.TestCase):
    def test_duels_config(self):
        gc = BattleSnakeDuelsConfig()
        print(gc.ec)
        ec = SimpleBattleSnakeEncodingConfig()
        gc = BattleSnakeDuelsConfig(ec=ec)
        game = BattleSnakeGame(gc)
        game.reset()
        game.render()
