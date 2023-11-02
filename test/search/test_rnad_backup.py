import unittest

from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.initialization import get_game_from_config
from src.game.values import ZeroSumNorm
from src.network.resnet import ResNetConfig3x3
from src.search.config import MCTSConfig, NetworkEvalConfig, SampleSelectionConfig, RNADBackupConfig, \
    SpecialExtractConfig
from src.search.initialization import get_search_from_config


class TestRNADBackup(unittest.TestCase):
    def test_rnad_choke(self):
        gc = perform_choke_2_player(fully_connected=False, centered=True)
        net_cfg = ResNetConfig3x3(game_cfg=gc, predict_policy=True)
        search_cfg = MCTSConfig(
            eval_func_cfg=NetworkEvalConfig(net_cfg=net_cfg, zero_sum_norm=ZeroSumNorm.NONE),
            sel_func_cfg=SampleSelectionConfig(),
            backup_func_cfg=RNADBackupConfig(),
            extract_func_cfg=SpecialExtractConfig(),
            expansion_depth=1,
            use_hot_start=True,
            optimize_fully_explored=False,
            discount=0.95,
        )
        search = get_search_from_config(search_cfg)
        game = get_game_from_config(gc)

        game.render()
        values, policies, info = search(game, iterations=500)
        print(f"{values=}")
        print(f"{policies=}")
