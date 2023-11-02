import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.network.resnet import ResNetConfig3x3
from src.search.config import PolicyExtractConfig, UncertaintySelectionConfig, AreaControlEvalConfig, \
    UncertaintyBackupConfig, MCTSConfig, NetworkEvalConfig
from src.search.mcts import MCTS


class TestUncertainty(unittest.TestCase):

    def test_uncertainty_choke(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = UncertaintySelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = UncertaintyBackupConfig(lr=0.3, temperature=3)
        extract_func_cfg = PolicyExtractConfig()
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

    def test_uncertainty_informed_legal(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = False
        env = BattleSnakeGame(gc)
        sel_func_cfg = UncertaintySelectionConfig(informed=True)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        backup_func_cfg = UncertaintyBackupConfig(lr=0.3, temperature=2, informed=True)
        extract_func_cfg = PolicyExtractConfig()
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

    def test_uncertainty_no_children(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = UncertaintySelectionConfig(informed=True)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        backup_func_cfg = UncertaintyBackupConfig(lr=0.3, temperature=3, informed=True, use_children=False)
        extract_func_cfg = PolicyExtractConfig()
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
            env.step((0, 0))
        env.render()

    def test_uncertainty_multiple_runs(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        game = BattleSnakeGame(gc)
        sel_func_cfg = UncertaintySelectionConfig(informed=True)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        backup_func_cfg = UncertaintyBackupConfig(lr=0.3, temperature=3, informed=True, use_children=False)
        extract_func_cfg = PolicyExtractConfig()
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
        game.render()
        for _ in range(100):
            values, action_probs, info = mcts(game, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
