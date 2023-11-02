import unittest

from src.game.actions import sample_individual_actions
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player
from src.game.values import ZeroSumNorm
from src.network.resnet import ResNetConfig3x3, ResNetConfig5x5
from src.search.backup_func import MaxMinBackupConfig
from src.search.eval_func import AreaControlEvalConfig, NetworkEvalConfig
from src.search.extraction_func import SpecialExtractConfig
from src.search.mcts import MCTS, MCTSConfig
from src.search.sel_func import SampleSelectionConfig


class TestMaxMin(unittest.TestCase):
    def test_maxmin_choke(self):
        for expansion_depth in [1, 2, 3]:
            gc = perform_choke_2_player(centered=False, fully_connected=False)
            env = BattleSnakeGame(gc)
            sel_func_cfg = SampleSelectionConfig()
            eval_func_cfg = AreaControlEvalConfig()
            backup_func_cfg = MaxMinBackupConfig()
            extract_func_cfg = SpecialExtractConfig()
            mcts_cfg = MCTSConfig(
                sel_func_cfg=sel_func_cfg,
                eval_func_cfg=eval_func_cfg,
                backup_func_cfg=backup_func_cfg,
                extract_func_cfg=extract_func_cfg,
                expansion_depth=expansion_depth,
                use_hot_start=False,
                optimize_fully_explored=True,
            )
            mcts = MCTS(mcts_cfg)
            for i in [2, 1, 0]:
                env.render()
                values, action_probs, info = mcts(env, iterations=100)
                self.assertAlmostEqual(0.99**(i+1), values[0], places=4)
                self.assertAlmostEqual(-(0.99**(i+1)), values[1], places=4)
                self.assertTrue(info.fully_explored)
                if i != 0:
                    self.assertGreater(action_probs[0, 0], action_probs[0, 1])
                    self.assertGreater(action_probs[0, 0], action_probs[0, 1])
                else:
                    self.assertGreater(action_probs[0, 3], action_probs[0, 0])
                env.step((0, 0))
            env.render()

    def test_maxmin_unoptimized(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = False
        env = BattleSnakeGame(gc)
        sel_func_cfg = SampleSelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = MaxMinBackupConfig()
        extract_func_cfg = SpecialExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=1,
            use_hot_start=False,
            optimize_fully_explored=False,
        )
        mcts = MCTS(mcts_cfg)
        for i in [1, 0]:
            env.render()
            values, action_probs, info = mcts(env, iterations=1000)
            self.assertAlmostEqual(0.99 ** i, values[0], delta=0.05)
            self.assertAlmostEqual(-(0.99 ** i), values[1], delta=0.05)
            self.assertFalse(info.fully_explored)
            self.assertGreater(action_probs[0, 0], action_probs[0, 3])
            self.assertGreater(action_probs[0, 0], action_probs[0, 1])
            actions = sample_individual_actions(action_probs, 1)
            env.step((0, 0))
        env.render()

    def test_maxmin_network(self):
        for optimize_full in [True, False]:
            gc = perform_choke_2_player(centered=False, fully_connected=False)
            gc.all_actions_legal = False
            env = BattleSnakeGame(gc)
            sel_func_cfg = SampleSelectionConfig()
            net_config = ResNetConfig3x3(game_cfg=gc)
            eval_func_cfg = NetworkEvalConfig(net_cfg=net_config, zero_sum_norm=ZeroSumNorm.LINEAR)
            backup_func_cfg = MaxMinBackupConfig()
            extract_func_cfg = SpecialExtractConfig()
            mcts_cfg = MCTSConfig(
                sel_func_cfg=sel_func_cfg,
                eval_func_cfg=eval_func_cfg,
                backup_func_cfg=backup_func_cfg,
                extract_func_cfg=extract_func_cfg,
                expansion_depth=1,
                use_hot_start=False,
                optimize_fully_explored=optimize_full,
            )
            mcts = MCTS(mcts_cfg)
            for i in [1, 0]:
                env.render()
                values, action_probs, info = mcts(env, iterations=1000)
                self.assertAlmostEqual(0.99 ** i, values[0], delta=0.05)
                self.assertAlmostEqual(-(0.99 ** i), values[1], delta=0.05)
                self.assertGreater(action_probs[0, 0], action_probs[0, 3])
                self.assertGreater(action_probs[0, 0], action_probs[0, 1])
                actions = sample_individual_actions(action_probs, 1)
                env.step((0, 0))
            env.render()

    def test_maxmin_4_player(self):
        for legal in [True, False]:
            gc = perform_choke_5x5_4_player(centered=True)
            gc.all_actions_legal = legal
            env = BattleSnakeGame(gc)
            sel_func_cfg = SampleSelectionConfig()
            net_config = ResNetConfig5x5(game_cfg=gc)
            eval_func_cfg = NetworkEvalConfig(net_cfg=net_config, zero_sum_norm=ZeroSumNorm.NONE)
            backup_func_cfg = MaxMinBackupConfig()
            extract_func_cfg = SpecialExtractConfig()
            mcts_cfg = MCTSConfig(
                sel_func_cfg=sel_func_cfg,
                eval_func_cfg=eval_func_cfg,
                backup_func_cfg=backup_func_cfg,
                extract_func_cfg=extract_func_cfg,
                expansion_depth=1,
                use_hot_start=False,
                optimize_fully_explored=False,
                discount=0.95
            )
            mcts = MCTS(mcts_cfg)
            env.step((0, 0, 0, 0))
            env.step((0, 0, 0, 0))
            env.step((0, 0, 0, 0))
            env.render()
            values, action_probs, info = mcts(env, iterations=100)
            print(action_probs)
            env.step((0, 0, 0, 0))
            env.render()
