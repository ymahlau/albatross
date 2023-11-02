import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.network.resnet import ResNetConfig3x3
from src.search.config import AreaControlEvalConfig, MCTSConfig, MeanPolicyExtractConfig, \
    RegretMatchingSelectionConfig, RegretMatchingBackupConfig, NetworkEvalConfig
from src.search.initialization import get_search_from_config
from src.search.mcts import MCTS


class TestRegretMatching(unittest.TestCase):
    def test_regret_matching_choke(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = RegretMatchingSelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = RegretMatchingBackupConfig()
        extract_func_cfg = MeanPolicyExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=False,
            optimize_fully_explored=False,
            discount=0.99,
        )
        mcts = MCTS(mcts_cfg)
        for _ in range(3):
            env.render()
            values, action_probs, info = mcts(env, iterations=500)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_regret_matching_avg_backup(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = RegretMatchingSelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = RegretMatchingBackupConfig(avg_backup=True)
        extract_func_cfg = MeanPolicyExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=False,
            optimize_fully_explored=False,
            discount=0.95,
        )
        mcts = MCTS(mcts_cfg)
        for _ in range(3):
            env.render()
            values, action_probs, info = mcts(env, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_regret_matching_informed(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = RegretMatchingSelectionConfig(informed_exp=True)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        backup_func_cfg = RegretMatchingBackupConfig(avg_backup=True)
        extract_func_cfg = MeanPolicyExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=False,
            optimize_fully_explored=False,
            discount=0.95,
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

    def test_regret_matching_informed_legal(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = False
        env = BattleSnakeGame(gc)
        sel_func_cfg = RegretMatchingSelectionConfig(informed_exp=True)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        backup_func_cfg = RegretMatchingBackupConfig(avg_backup=True)
        extract_func_cfg = MeanPolicyExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=False,
            optimize_fully_explored=False,
            discount=0.95,
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

    def test_regret_matching_legal(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = False
        env = BattleSnakeGame(gc)
        sel_func_cfg = RegretMatchingSelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = RegretMatchingBackupConfig(avg_backup=True)
        extract_func_cfg = MeanPolicyExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=False,
            optimize_fully_explored=False,
            discount=0.95,
        )
        mcts = MCTS(mcts_cfg)
        for _ in range(2):
            env.render()
            values, action_probs, info = mcts(env, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_rm_multiple_runs(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        game = BattleSnakeGame(gc)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        sel_func_cfg = RegretMatchingSelectionConfig(random_prob=0.2)
        backup_func_cfg = RegretMatchingBackupConfig(avg_backup=True)
        extract_func_cfg = MeanPolicyExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=False,
            optimize_fully_explored=False,
            discount=0.99,
        )
        search = get_search_from_config(mcts_cfg)
        game.render()
        for _ in range(100):
            values, action_probs, info = search(game, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
