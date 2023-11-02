import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.search.config import Exp3SelectionConfig, AreaControlEvalConfig, Exp3BackupConfig, MCTSConfig, \
    MeanPolicyExtractConfig
from src.search.mcts import MCTS


class TestExp3(unittest.TestCase):
    def test_exp3_choke(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = Exp3SelectionConfig(altered=False)
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = Exp3BackupConfig()
        extract_func_cfg = MeanPolicyExtractConfig()
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
            values, action_probs, info = mcts(env, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            # self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_exp3_choke_legal(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = False
        env = BattleSnakeGame(gc)
        sel_func_cfg = Exp3SelectionConfig(altered=False)
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = Exp3BackupConfig()
        extract_func_cfg = MeanPolicyExtractConfig()
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
        for _ in range(2):
            env.render()
            values, action_probs, info = mcts(env, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            # self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_exp3_choke_avg_backup(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = Exp3SelectionConfig(altered=False)
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = Exp3BackupConfig(avg_backup=True)
        extract_func_cfg = MeanPolicyExtractConfig()
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
        for _ in range(2):
            env.render()
            values, action_probs, info = mcts(env, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            # self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()
