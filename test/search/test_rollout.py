import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.search.config import DecoupledUCTSelectionConfig, RandomRolloutEvalConfig, StandardBackupConfig, \
    StandardExtractConfig, MCTSConfig
from src.search.mcts import MCTS


class TestRollout(unittest.TestCase):
    def test_rollout_choke(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = DecoupledUCTSelectionConfig(exp_bonus=2.0)
        eval_func_cfg = RandomRolloutEvalConfig(num_rollouts=100)
        backup_func_cfg = StandardBackupConfig()
        extract_func_cfg = StandardExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=False,
            optimize_fully_explored=False,
        )
        mcts = MCTS(mcts_cfg)
        for _ in range(3):
            env.render()
            values, action_probs, info = mcts(env, iterations=100)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()
