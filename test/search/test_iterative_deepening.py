import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.search.backup_func import NashBackupConfig
from src.search.eval_func import AreaControlEvalConfig
from src.search.extraction_func import SpecialExtractConfig
from src.search.iterative_deepening import IterativeDeepeningConfig, IterativeDeepening


class TestIterativeDeepening(unittest.TestCase):
    def test_id_simple(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = True
        game = BattleSnakeGame(gc)
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = NashBackupConfig()
        extract_func_cfg = SpecialExtractConfig()
        id_cfg = IterativeDeepeningConfig(
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
        )
        id_search = IterativeDeepening(id_cfg)
        for _ in range(3):
            game.render()
            values, action_probs, info = id_search(game, iterations=4)
            self.assertTrue(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            game.step((0, 0))
        game.render()
