import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player


class TestState(unittest.TestCase):
    def test_state_choke(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=True)
        game = BattleSnakeGame(game_cfg)
        game.step((0, 0))
        game.render()

        state = game.get_state()
        print(state)

        game2 = BattleSnakeGame(game_cfg)
        game2.set_state(state)
        game2.render()

        game2.step((0, 0))
        game2.render()

        game.set_state(game2.get_state())
        game.render()
