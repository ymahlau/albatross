import unittest

from src.game.initialization import get_game_from_config
from src.game.normal_form.random_matrix import get_random_matrix_cfg, NFGType


class TestRandomMatrix(unittest.TestCase):
    def test_random_game_zero_sum(self):
        cfg = get_random_matrix_cfg([2, 2], nfg_type=NFGType.ZERO_SUM)
        game = get_game_from_config(cfg)
        game.render()
        self.assertEqual(2, game.num_players)
        self.assertEqual(2, game.num_actions)

    def test_random_game(self):
        cfg = get_random_matrix_cfg([3, 5, 2], nfg_type=NFGType.GENERAL)
        game = get_game_from_config(cfg)
        game.render()
        self.assertEqual(3, game.num_players)
        self.assertEqual(5, game.num_actions)

    def test_random_game_cooperative(self):
        cfg = get_random_matrix_cfg([3, 4, 2, 2], nfg_type=NFGType.FULL_COOP)
        game = get_game_from_config(cfg)
        game.render()
        self.assertEqual(4, game.num_players)
        self.assertEqual(4, game.num_actions)
