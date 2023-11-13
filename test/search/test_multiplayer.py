import math
import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame, DOWN, LEFT, UP
from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7_constrictor_4_player
from src.game.values import UtilityNorm
from src.network.resnet import ResNetConfig7x7
from src.search.config import NetworkEvalConfig, SampleSelectionConfig, NashBackupConfig, SpecialExtractConfig, \
    MCTSConfig, AlphaZeroDecoupledSelectionConfig, StandardBackupConfig, StandardExtractConfig
from src.search.initialization import get_search_from_config


class TestMultiplayer(unittest.TestCase):
    def test_mcts_4_player_single_player_stuck(self):
        snake_pos = {0: [[0, 2], [1, 2], [1, 3]], 1: [[6, 0], [6, 1], [5, 1]],
                     2: [[5, 3], [4, 3], [4, 4]], 3: [[6, 4], [6, 5], [5, 5]]}
        game_cfg = survive_on_7x7_constrictor_4_player()
        game_cfg.init_snake_pos = snake_pos
        game = BattleSnakeGame(game_cfg)
        # network
        net_cfg = ResNetConfig7x7(game_cfg=game_cfg)
        # search
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_cfg, utility_norm=UtilityNorm.NONE)
        sel_func_cfg = SampleSelectionConfig(dirichlet_alpha=math.inf, dirichlet_eps=0.25, temperature=1.0)
        backup_func_cfg = NashBackupConfig()
        extraction_func_cfg = SpecialExtractConfig()
        search_cfg = MCTSConfig(
            eval_func_cfg=eval_func_cfg,
            sel_func_cfg=sel_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extraction_func_cfg,
            expansion_depth=1,
            use_hot_start=True,
            optimize_fully_explored=True,
        )
        game.render()
        search = get_search_from_config(search_cfg)
        values, action_probs, info = search(game, iterations=5)
        game.step((UP, LEFT, UP, DOWN))
        game.render()
        values, action_probs, info = search(game, iterations=5)
        game.step((UP, LEFT, DOWN))
        game.render()
        values, action_probs, info = search(game, iterations=5)
        print(values)

    def test_mcts_4_player_standard(self):
        snake_pos = {0: [[0, 2], [1, 2], [1, 3]], 1: [[6, 0], [6, 1], [5, 1]],
                     2: [[5, 3], [4, 3], [4, 4]], 3: [[6, 4], [6, 5], [5, 5]]}
        game_cfg = survive_on_7x7_constrictor_4_player()
        game_cfg.init_snake_pos = snake_pos
        game = BattleSnakeGame(game_cfg)
        # network
        net_cfg = ResNetConfig7x7(game_cfg=game_cfg, predict_policy=True)
        # search
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_cfg, utility_norm=UtilityNorm.NONE)
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig()
        backup_func_cfg = StandardBackupConfig()
        extraction_func_cfg = StandardExtractConfig()
        search_cfg = MCTSConfig(
            eval_func_cfg=eval_func_cfg,
            sel_func_cfg=sel_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extraction_func_cfg,
            expansion_depth=0,
            use_hot_start=False,
            optimize_fully_explored=False,
        )
        game.render()
        search = get_search_from_config(search_cfg)
        values, action_probs, info = search(game, iterations=200)
        print(values)
        game.step((UP, LEFT, UP, DOWN))
        game.render()
        values, action_probs, info = search(game, iterations=200)
        print(values)
        game.step((UP, LEFT, DOWN))
        game.render()
        values, action_probs, info = search(game, iterations=200)
        print(values)