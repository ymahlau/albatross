import unittest

from src.game.overcooked.layouts import AsymmetricAdvantageOvercookedConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.game.overcooked.state import get_albatross_oa_state


class TestState(unittest.TestCase):
    def test_oa_albatross_state(self):
        simple_state = get_albatross_oa_state(time_step=54)
        game_cfg = AsymmetricAdvantageOvercookedConfig()
        game = OvercookedGame(game_cfg)
        game.set_state(simple_state)
        game.render()
