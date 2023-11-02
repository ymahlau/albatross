import ctypes
import unittest

from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.initialization import get_game_from_config
from src.search.config import MCTSConfig, AreaControlEvalConfig, StandardBackupConfig, DecoupledUCTSelectionConfig, \
    StandardExtractConfig
from src.search.initialization import get_search_from_config
import multiprocessing as mp

class TestSearchProbs(unittest.TestCase):
    def test_search_probs(self):
        search_cfg = MCTSConfig(
            eval_func_cfg=AreaControlEvalConfig(),
            backup_func_cfg=StandardBackupConfig(),
            sel_func_cfg=DecoupledUCTSelectionConfig(),
            extract_func_cfg=StandardExtractConfig(),
        )
        search = get_search_from_config(search_cfg)
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        game = get_game_from_config(game_cfg)

        prob_arr = mp.Array(ctypes.c_float, 4, lock=False)

        search(
            game,
            iterations=200,
            save_probs=prob_arr,
            save_player_idx=0,
        )
        print([prob_arr[i] for i in range(4)])
        self.assertTrue(prob_arr[0] > 0)