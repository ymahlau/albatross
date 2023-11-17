import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.battlesnake.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player
from src.game.values import UtilityNorm
from src.search.backup_func import NashBackupConfig
from src.search.config import DecoupledUCTSelectionConfig
from src.search.eval_func import AreaControlEvalConfig
from src.search.extraction_func import SpecialExtractConfig
from src.search.mcts import MCTS, MCTSConfig
from src.search.sel_func import SampleSelectionConfig


class TestNash(unittest.TestCase):
    def test_nash_choke(self):
        for legal in [True, False]:
            for optimize in [True, False]:
                for hot_start in [True, False]:
                    gc = perform_choke_2_player(centered=False, fully_connected=False)
                    gc.all_actions_legal = legal
                    env = BattleSnakeGame(gc)
                    sel_func_cfg = SampleSelectionConfig()
                    eval_func_cfg = AreaControlEvalConfig()
                    backup_func_cfg = NashBackupConfig()
                    extract_func_cfg = SpecialExtractConfig()
                    mcts_cfg = MCTSConfig(
                        sel_func_cfg=sel_func_cfg,
                        eval_func_cfg=eval_func_cfg,
                        backup_func_cfg=backup_func_cfg,
                        extract_func_cfg=extract_func_cfg,
                        expansion_depth=1,
                        use_hot_start=hot_start,
                        optimize_fully_explored=optimize,
                    )
                    mcts = MCTS(mcts_cfg)
                    num_steps = 3 if legal else 2
                    for _ in range(num_steps):
                        env.render()
                        values, action_probs, info = mcts(env, iterations=100)
                        self.assertEqual(optimize, info.fully_explored)
                        self.assertTrue(values[0] > values[1])
                        env.step((0, 0))
                    env.render()

    def test_nash_multiplayer(self):
        gc = perform_choke_5x5_4_player(centered=True)
        game = BattleSnakeGame(gc)
        sel_func_cfg = DecoupledUCTSelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = NashBackupConfig()
        extract_func_cfg = SpecialExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=1,
            use_hot_start=True,
            optimize_fully_explored=False,
        )
        mcts = MCTS(mcts_cfg)
        for _ in range(3):
            game.render()
            values, action_probs, info = mcts(game, iterations=100)
            print(values)
            print(action_probs)
            game.step((0, 0, 0, 0))
        game.render()

    def test_zero_sum_extraction(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        sel_func_cfg = SampleSelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = NashBackupConfig()
        extract_func_cfg = SpecialExtractConfig(utility_norm=UtilityNorm.ZERO_SUM)
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=1,
            use_hot_start=False,
            optimize_fully_explored=True,
        )
        mcts = MCTS(mcts_cfg)
        for _ in range(3):
            env.render()
            values, action_probs, info = mcts(env, iterations=100)
            self.assertTrue(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()
