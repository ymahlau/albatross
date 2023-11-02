import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.search.config import AreaControlEvalConfig, FixedDepthConfig, SpecialExtractConfig, QNEBackupConfig
from src.search.initialization import get_search_from_config


class TestQNEBackup(unittest.TestCase):
    def test_qne_choke(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        env = BattleSnakeGame(gc)
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = QNEBackupConfig(init_temperature=5)
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
