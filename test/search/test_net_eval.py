import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player
from src.network.fcn import MediumHeadConfig
from src.network.resnet import ResNetConfig5x5
from src.search.config import AlphaZeroDecoupledSelectionConfig, NetworkEvalConfig, StandardBackupConfig, \
    StandardExtractConfig, MCTSConfig
from src.search.mcts import MCTS


class TestNetworkEval(unittest.TestCase):
    def test_single_temperature_input_film(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig()
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=True,
                                     single_film_temperature=True)
        net_config.film_cfg = MediumHeadConfig()
        eval_func_cfg = NetworkEvalConfig(
            net_cfg=net_config,
            init_temperatures=[1, 2, 3, 4],
            single_temperature=True,
            obs_temperature_input=False,
            temperature_input=True,
        )
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
            values, action_probs, info = mcts(env, iterations=400)
            print(values)
            for enemy in range(1, 4):
                pass
                # self.assertTrue(values[0] > values[enemy])
            env.step((0, 0, 0, 0))
            mcts.set_temperatures([2, 3, 4, 5])
        env.render()

    def test_multiple_temperature_input_film(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig()
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=True,
                                     single_film_temperature=False)
        net_config.film_cfg = MediumHeadConfig()
        eval_func_cfg = NetworkEvalConfig(
            net_cfg=net_config,
            init_temperatures=[1, 2, 3, 4],
            single_temperature=False,
            obs_temperature_input=False,
            temperature_input=True,
        )
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
            values, action_probs, info = mcts(env, iterations=400)
            print(values)
            for enemy in range(1, 4):
                pass
                # self.assertTrue(values[0] > values[enemy])
            env.step((0, 0, 0, 0))
            mcts.set_temperatures([5, 6, 7, 8])
        env.render()

    def test_single_temperature_input_obs(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.ec.temperature_input = True
        gc.ec.single_temperature_input = True
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig()
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=False,
                                     single_film_temperature=False)
        # net_config.film_cfg = MediumHeadConfig()
        eval_func_cfg = NetworkEvalConfig(
            net_cfg=net_config,
            init_temperatures=[1, 2, 3, 4],
            single_temperature=True,
            obs_temperature_input=True,
            temperature_input=True,
        )
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
            values, action_probs, info = mcts(env, iterations=400)
            print(values)
            for enemy in range(1, 4):
                pass
                # self.assertTrue(values[0] > values[enemy])
            env.step((0, 0, 0, 0))
            mcts.set_temperatures([5, 6, 7, 8])
        env.render()

    def test_multiple_temperature_input_obs(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.ec.temperature_input = True
        gc.ec.single_temperature_input = False
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig()
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=False,
                                     single_film_temperature=False)
        # net_config.film_cfg = MediumHeadConfig()
        eval_func_cfg = NetworkEvalConfig(
            net_cfg=net_config,
            init_temperatures=[1, 2, 3, 4],
            single_temperature=False,
            obs_temperature_input=True,
            temperature_input=True,
        )
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
            values, action_probs, info = mcts(env, iterations=400)
            print(values)
            for enemy in range(1, 4):
                pass
                # self.assertTrue(values[0] > values[enemy])
            env.step((0, 0, 0, 0))
            mcts.set_temperatures([5, 6, 7, 8])
        env.render()
