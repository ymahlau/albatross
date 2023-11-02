import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.values import UtilityNorm
from src.network.resnet import ResNetConfig3x3
from src.search.config import SampleSelectionConfig, AreaControlEvalConfig, SpecialExtractConfig, MCTSConfig, \
    LogitBackupConfig, NetworkEvalConfig
from src.search.mcts import MCTS


class TestIteratedEquilibriumBackup(unittest.TestCase):
    def test_choke_backup(self):
        for legal in [False, True]:
            for optimize in [False, True]:
                for hot_start in [True, False]:
                    gc = perform_choke_2_player(centered=False, fully_connected=False)
                    gc.all_actions_legal = legal
                    env = BattleSnakeGame(gc)
                    sel_func_cfg = SampleSelectionConfig()
                    eval_func_cfg = AreaControlEvalConfig()
                    backup_func_cfg = LogitBackupConfig(init_temperatures=[5, 5])
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

    def test_informed_initialization(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = False
        env = BattleSnakeGame(gc)
        sel_func_cfg = SampleSelectionConfig()
        net_cfg = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_cfg, value_norm_type=UtilityNorm.ZERO_SUM)
        backup_func_cfg = LogitBackupConfig(
            num_iterations=1000,
            epsilon=0,
            init_random=False,
            init_temperatures=[5, 5]
        )
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
        num_steps = 2
        for _ in range(num_steps):
            env.render()
            values, action_probs, info = mcts(env, iterations=100)
            self.assertEqual(False, info.fully_explored)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()
