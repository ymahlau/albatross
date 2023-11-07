import unittest

from src.game.initialization import get_game_from_config
from src.game.normal_form.normal_form import NormalFormConfig
from src.search.config import FixedDepthConfig, DummyEvalConfig, NashBackupConfig, SpecialExtractConfig
from src.search.initialization import get_search_from_config


class TestNormalForm(unittest.TestCase):
    def test_prisoners_dilemma(self):
        ja_dict = {
            (0, 0): (1, 1),
            (1, 0): (0, 5),
            (0, 1): (5, 0),
            (1, 1): (3, 3),
        }
        cfg = NormalFormConfig(ja_dict=ja_dict)
        game = get_game_from_config(cfg)
        game.render()

        rewards, done, _ = game.step((0, 0))
        self.assertTrue(done)
        self.assertTrue(game.is_terminal())
        self.assertEqual(1, rewards[0])
        self.assertEqual(1, rewards[1])

        game.reset()
        self.assertFalse(game.is_terminal())

    def test_solve_nfg(self):
        ja_dict = {
            (0, 0): (1, 1),
            (1, 0): (0, 5),
            (0, 1): (5, 0),
            (1, 1): (3, 3),
        }
        cfg = NormalFormConfig(ja_dict=ja_dict)

        search_cfg = FixedDepthConfig(
            eval_func_cfg=DummyEvalConfig(),
            backup_func_cfg=NashBackupConfig(),
            extract_func_cfg=SpecialExtractConfig(),
            discount=1,
            average_eval=False,
        )
        search = get_search_from_config(search_cfg)
        game = get_game_from_config(cfg)

        values, probs, _ = search(game, iterations=1)
        print(f"{values=}")
        print(f"{probs=}")
        self.assertAlmostEqual(1, values[0])
        self.assertAlmostEqual(1, values[1])
        self.assertAlmostEqual(1, probs[0][0])
        self.assertAlmostEqual(1, probs[1][0])

