import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame, LEFT, UP, RIGHT
from src.game.bootcamp.test_envs_11x11 import survive_on_11x11
from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player
from src.game.bootcamp.test_envs_7x7 import survive_on_7x7_constrictor
from src.game.initialization import get_game_from_config
from src.game.values import ZeroSumNorm
from src.network.resnet import ResNetConfig3x3, ResNetConfig5x5, ResNetConfig7x7
from src.search.backup_func import StandardBackupConfig
from src.search.config import CopyCatEvalConfig
from src.search.eval_func import AreaControlEvalConfig, NetworkEvalConfig
from src.search.extraction_func import StandardExtractConfig
from src.search.initialization import get_search_from_config
from src.search.mcts import MCTS, MCTSConfig
from src.search.sel_func import DecoupledUCTSelectionConfig, \
    AlphaZeroDecoupledSelectionConfig


class TestStandard(unittest.TestCase):
    def test_standard_choke(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = DecoupledUCTSelectionConfig(exp_bonus=2.0)
        eval_func_cfg = AreaControlEvalConfig()
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
            self.assertFalse(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_network_choke_standard(self):
        gc = perform_choke_2_player(centered=True, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = DecoupledUCTSelectionConfig(exp_bonus=2.0)
        net_config = ResNetConfig3x3(game_cfg=gc)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config, zero_sum_norm=ZeroSumNorm.LINEAR)
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
            self.assertFalse(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_network_choke_alpha_zero(self):
        gc = perform_choke_2_player(centered=True, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig(exp_bonus=2.0)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config, zero_sum_norm=ZeroSumNorm.LINEAR)
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
            discount=0.95,
        )
        mcts = MCTS(mcts_cfg)
        for _ in range(3):
            env.render()
            values, action_probs, info = mcts(env, iterations=200)
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_standard_4_player(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig()
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config, zero_sum_norm=ZeroSumNorm.NONE)
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
        env.step((0, 0, 0, 0))
        env.step((0, 0, 0, 0))
        for _ in range(2):
            env.render()
            values, action_probs, info = mcts(env, iterations=200)
            print(values)
            # for enemy in range(1, 4):
            #     self.assertTrue(values[0] > values[enemy])
            env.step((0, 0, 0, 0))
        env.render()

    def test_return_only_legal_actions(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.all_actions_legal = False
        game = get_game_from_config(gc)
        net_cfg = ResNetConfig5x5(game_cfg=gc, predict_policy=True)
        search_cfg = MCTSConfig(
            eval_func_cfg=NetworkEvalConfig(net_cfg=net_cfg, zero_sum_norm=ZeroSumNorm.NONE),
            sel_func_cfg=AlphaZeroDecoupledSelectionConfig(),
            backup_func_cfg=StandardBackupConfig(),
            extract_func_cfg=StandardExtractConfig(),
            use_hot_start=False,
            expansion_depth=0,
            optimize_fully_explored=False,
        )
        search = get_search_from_config(search_cfg)
        for i in range(4):
            values, action_probs, info = search(game, iterations=20)
            for player in game.players_at_turn():
                for action in game.illegal_actions(player):
                    self.assertEqual(0, action_probs[player, action])
            game.step((0, 0, 0, 0))

    def test_no_network_init(self):
        gc = perform_choke_2_player(centered=True, fully_connected=False)
        env = BattleSnakeGame(gc)
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig(exp_bonus=2.0)
        # net_config = ResNetConfigCentered3x3(game_cfg=gc)
        eval_func_cfg = NetworkEvalConfig(zero_sum_norm=ZeroSumNorm.LINEAR)
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
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_special_choke_situation(self):
        gc = perform_choke_2_player(centered=True, fully_connected=False)
        game = BattleSnakeGame(gc)
        game.step((LEFT, UP))
        game.step((UP, UP))
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig(exp_bonus=2.0)
        eval_func_cfg = NetworkEvalConfig(zero_sum_norm=ZeroSumNorm.LINEAR)
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
        game.render()
        values, action_probs, info = mcts(game, iterations=500)
        game.step((RIGHT, LEFT))
        values, action_probs, info = mcts(game, iterations=500)
        self.assertEqual(0, values[0])
        self.assertEqual(0, values[0])

    def test_large_game(self):
        game_cfg = survive_on_7x7_constrictor()
        net_cfg = ResNetConfig7x7(game_cfg=game_cfg, predict_policy=True)
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig(exp_bonus=2.0)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_cfg, zero_sum_norm=ZeroSumNorm.LINEAR)
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
        game = get_game_from_config(game_cfg)
        game.render()
        values, action_probs, info = mcts(game, iterations=100)

    def test_copycat_large(self):
        gc = survive_on_11x11()
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        sel_func_cfg = DecoupledUCTSelectionConfig(exp_bonus=2.0)
        eval_func_cfg = CopyCatEvalConfig()
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
        env.render()
        values, action_probs, info = mcts(env, iterations=100)
        self.assertFalse(info.fully_explored)
