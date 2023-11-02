import unittest

from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.initialization import get_game_from_config
from src.search.config import StandardBackupConfig, DummyEvalConfig, DecoupledUCTSelectionConfig, \
    StandardExtractConfig, MCTSConfig
from src.search.mcts import MCTS


class TestDummyEval(unittest.TestCase):
    def test_dummy_choke(self):
        game_cfg = perform_choke_2_player(fully_connected=False, centered=True)
        env = get_game_from_config(game_cfg)
        sel_func_cfg = DecoupledUCTSelectionConfig(exp_bonus=1.41)
        eval_func_cfg = DummyEvalConfig()
        backup_func_cfg = StandardBackupConfig()
        extract_func_cfg = StandardExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=True,
            optimize_fully_explored=False,
        )
        mcts = MCTS(mcts_cfg)
        for _ in range(3):
            env.render()
            values, action_probs, info = mcts(env, iterations=1000)
            self.assertFalse(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()
