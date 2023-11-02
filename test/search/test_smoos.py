import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.battlesnake.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player
from src.network.resnet import ResNetConfig3x3
from src.search.config import SMOOSConfig, AreaControlEvalConfig, NetworkEvalConfig
from src.search.initialization import get_search_from_config


class TestSMOOS(unittest.TestCase):
    def test_smoos_choke_standard(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        game = BattleSnakeGame(gc)
        smoos_cfg = SMOOSConfig(
            eval_func_cfg=AreaControlEvalConfig(),
            use_hot_start=False,
            exp_factor=0.2,
            informed_exp=False,  # multiply net probs with exploration
            exp_decay=False,  # logarithmic decay of exploration
            uct_explore=False,  # count-based exploration like in UCT
            enhanced_regret=False,  # subtractive regret calculation like regret matching
            regret_decay=1.0,
            lambda_val=1.0,
            relief_update=True,  # also calculate negative regret (relief) for chosen action
        )
        search = get_search_from_config(smoos_cfg)
        # game.step((0, 0))
        # game.step((0, 0))
        for _ in range(3):
            game.render()
            values, action_probs, info = search(game, iterations=100)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            # self.assertTrue(values[0] > values[1])
            game.step((0, 0))
        game.render()

    def test_smoos_informed(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        game = BattleSnakeGame(gc)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        smoos_cfg = SMOOSConfig(
            eval_func_cfg=eval_func_cfg,
            use_hot_start=False,
            exp_factor=0.2,
            informed_exp=True,  # multiply net probs with exploration
            exp_decay=False,  # logarithmic decay of exploration
            uct_explore=False,  # count-based exploration like in UCT
            enhanced_regret=False,  # subtractive regret calculation like regret matching
            regret_decay=1.0,
            lambda_val=1.0,
            relief_update=True,  # also calculate negative regret (relief) for chosen action
        )
        search = get_search_from_config(smoos_cfg)
        # game.step((0, 0))
        # game.step((0, 0))
        for _ in range(3):
            game.render()
            values, action_probs, info = search(game, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            # self.assertTrue(values[0] > values[1])
            game.step((0, 0))
        game.render()

    def test_smoos_informed_legal(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = False
        game = BattleSnakeGame(gc)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        smoos_cfg = SMOOSConfig(
            eval_func_cfg=eval_func_cfg,
            use_hot_start=False,
            exp_factor=0.2,
            informed_exp=True,  # multiply net probs with exploration
            exp_decay=False,  # logarithmic decay of exploration
            uct_explore=False,  # count-based exploration like in UCT
            enhanced_regret=False,  # subtractive regret calculation like regret matching
            regret_decay=1.0,
            lambda_val=1.0,
            relief_update=True,  # also calculate negative regret (relief) for chosen action
        )
        search = get_search_from_config(smoos_cfg)
        for _ in range(2):
            game.render()
            values, action_probs, info = search(game, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            # self.assertTrue(values[0] > values[1])
            game.step((0, 0))
        game.render()

    def test_smoos_regret(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        game = BattleSnakeGame(gc)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        smoos_cfg = SMOOSConfig(
            eval_func_cfg=eval_func_cfg,
            use_hot_start=False,
            exp_factor=0.2,
            informed_exp=True,  # multiply net probs with exploration
            exp_decay=False,  # logarithmic decay of exploration
            uct_explore=False,  # count-based exploration like in UCT
            enhanced_regret=True,  # subtractive regret calculation like regret matching
            regret_decay=1.0,
            lambda_val=1.0,
            relief_update=True,  # also calculate negative regret (relief) for chosen action
        )
        search = get_search_from_config(smoos_cfg)
        # game.step((0, 0))
        # game.step((0, 0))
        for _ in range(3):
            game.render()
            values, action_probs, info = search(game, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            # self.assertTrue(values[0] > values[1])
            game.step((0, 0))
        game.render()

    def test_smoos_uct_explore(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        game = BattleSnakeGame(gc)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        smoos_cfg = SMOOSConfig(
            eval_func_cfg=eval_func_cfg,
            use_hot_start=False,
            exp_factor=0.2,
            informed_exp=True,  # multiply net probs with exploration
            exp_decay=False,  # logarithmic decay of exploration
            uct_explore=True,  # count-based exploration like in UCT
            enhanced_regret=True,  # subtractive regret calculation like regret matching
            regret_decay=1.0,
            lambda_val=1.0,
            relief_update=True,  # also calculate negative regret (relief) for chosen action
        )
        search = get_search_from_config(smoos_cfg)
        for _ in range(3):
            game.render()
            values, action_probs, info = search(game, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            game.step((0, 0))
        game.render()

    def test_smoos_explore_decay(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        game = BattleSnakeGame(gc)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        smoos_cfg = SMOOSConfig(
            eval_func_cfg=eval_func_cfg,
            use_hot_start=False,
            exp_factor=0.5,
            informed_exp=True,  # multiply net probs with exploration
            exp_decay=True,  # logarithmic decay of exploration
            uct_explore=False,  # count-based exploration like in UCT
            enhanced_regret=True,  # subtractive regret calculation like regret matching
            regret_decay=1.0,
            lambda_val=1.0,
            relief_update=True,  # also calculate negative regret (relief) for chosen action
        )
        search = get_search_from_config(smoos_cfg)
        # game.step((0, 0))
        # game.step((0, 0))
        for _ in range(3):
            game.render()
            values, action_probs, info = search(game, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            game.step((0, 0))
        game.render()

    def test_smoos_lamda_backup(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        game = BattleSnakeGame(gc)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        smoos_cfg = SMOOSConfig(
            eval_func_cfg=eval_func_cfg,
            use_hot_start=False,
            exp_factor=0.5,
            informed_exp=True,  # multiply net probs with exploration
            exp_decay=True,  # logarithmic decay of exploration
            uct_explore=False,  # count-based exploration like in UCT
            enhanced_regret=True,  # subtractive regret calculation like regret matching
            regret_decay=1.0,
            lambda_val=0.5,
            relief_update=True,  # also calculate negative regret (relief) for chosen action
        )
        search = get_search_from_config(smoos_cfg)
        # game.step((0, 0))
        # game.step((0, 0))
        for _ in range(3):
            game.render()
            values, action_probs, info = search(game, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            game.step((0, 0))
        game.render()

    def test_smoos_enhanced_multiple_runs(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        game = BattleSnakeGame(gc)
        net_config = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config)
        smoos_cfg = SMOOSConfig(
            eval_func_cfg=eval_func_cfg,
            use_hot_start=False,
            exp_factor=0.1,
            informed_exp=True,  # multiply net probs with exploration
            exp_decay=False,  # logarithmic decay of exploration
            uct_explore=False,  # count-based exploration like in UCT
            enhanced_regret=True,  # subtractive regret calculation like regret matching
            regret_decay=0.9,
            lambda_val=0.0,
            relief_update=True,  # also calculate negative regret (relief) for chosen action
            discount=0.95,
        )
        search = get_search_from_config(smoos_cfg)
        game.render()
        for _ in range(100):
            values, action_probs, info = search(game, iterations=200)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)

    def test_smoos_multiplayer(self):
        game_cfg = perform_choke_5x5_4_player(centered=True)
        game = BattleSnakeGame(game_cfg)
        smoos_cfg = SMOOSConfig(
            eval_func_cfg=AreaControlEvalConfig(),
            use_hot_start=False,
            exp_factor=0.2,
            informed_exp=False,  # multiply net probs with exploration
            exp_decay=False,  # logarithmic decay of exploration
            uct_explore=False,  # count-based exploration like in UCT
            enhanced_regret=False,  # subtractive regret calculation like regret matching
            regret_decay=1.0,
            lambda_val=1.0,
            relief_update=True,  # also calculate negative regret (relief) for chosen action
        )
        search = get_search_from_config(smoos_cfg)
        # game.step((0, 0, 0, 0))
        # game.step((0, 0, 0, 0))
        for _ in range(4):
            game.render()
            values, action_probs, info = search(game, iterations=100)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertFalse(info.fully_explored)
            # self.assertTrue(values[0] > values[1])
            game.step((0, 0, 0, 0))
        game.render()
