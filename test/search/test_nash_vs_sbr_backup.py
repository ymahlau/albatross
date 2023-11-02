import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.extensive_form.random_efg import get_random_efg_config, EFGType
from src.game.initialization import get_game_from_config
from src.search.config import AreaControlEvalConfig, NashVsSBRBackupConfig, SpecialExtractConfig, FixedDepthConfig
from src.search.initialization import get_search_from_config


class TestNashVsSbrBackup(unittest.TestCase):
    def test_nash_vs_sbr_choke(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = NashVsSBRBackupConfig(init_temperature=5)
        extract_func_cfg = SpecialExtractConfig()
        mcts_cfg = FixedDepthConfig(
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
        )
        search = get_search_from_config(mcts_cfg)
        for _ in range(3):
            env.render()
            values, action_probs, info = search(env, iterations=100)
            print(f"{values=}")
            print(f"{action_probs=}")
            self.assertTrue(values[0] > values[1])
            env.step((0, 0))
        env.render()

    def test_nash_vs_sbr_random_efg(self):
        game_cfg = get_random_efg_config(
            num_actions=3,
            num_player=2,
            max_depth=2,
            efg_type=EFGType.ZERO_SUM,
        )
        game = get_game_from_config(game_cfg)
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = NashVsSBRBackupConfig(init_temperature=5)
        extract_func_cfg = SpecialExtractConfig()
        mcts_cfg = FixedDepthConfig(
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
        )
        search = get_search_from_config(mcts_cfg)
        values, action_probs, info = search(game, iterations=100)
        print(f"{values=}")
        print(f"{action_probs=}")
