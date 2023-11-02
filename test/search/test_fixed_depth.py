import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.bootcamp.test_envs_5x5 import survive_on_5x5
from src.game.initialization import get_game_from_config
from src.game.values import ZeroSumNorm
from src.network.fcn import MediumHeadConfig
from src.network.resnet import ResNetConfig5x5, ResNetConfig3x3
from src.network.utils import ActivationType
from src.network.vision_net import EquivarianceType
from src.search.backup_func import NashBackupConfig
from src.search.config import NetworkEvalConfig, SBRBackupConfig
from src.search.eval_func import AreaControlEvalConfig
from src.search.extraction_func import SpecialExtractConfig
from src.search.fixed_depth import FixedDepthSearch, FixedDepthConfig
from src.search.initialization import get_search_from_config


class TestFixedDepthSearch(unittest.TestCase):
    def test_fixed_depth_simple(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = True
        game = BattleSnakeGame(gc)
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = NashBackupConfig()
        extract_func_cfg = SpecialExtractConfig()
        for average_eval in [True, False]:
            game.reset()
            search_cfg = FixedDepthConfig(
                eval_func_cfg=eval_func_cfg,
                backup_func_cfg=backup_func_cfg,
                extract_func_cfg=extract_func_cfg,
                average_eval=average_eval,
            )
            search = FixedDepthSearch(search_cfg)
            values, action_probs, info = search(game, iterations=1)
            self.assertFalse(info.fully_explored)
            self.assertTrue(values[0] > values[1])
            for _ in range(3):
                game.render()
                values, action_probs, info = search(game, iterations=4)
                self.assertTrue(info.fully_explored)
                self.assertTrue(values[0] > values[1])
                game.step((0, 0))

    def test_network_eval_max_batch_size(self):
        gc = survive_on_5x5()
        game = BattleSnakeGame(gc)
        net_cfg = ResNetConfig5x5(game_cfg=gc)
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_cfg, max_batch_size=64, zero_sum_norm=ZeroSumNorm.NONE)
        backup_func_cfg = NashBackupConfig()
        extract_func_cfg = SpecialExtractConfig()
        search_cfg = FixedDepthConfig(
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            average_eval=True,
        )
        search = get_search_from_config(search_cfg)
        game.reset()
        values, action_probs, info = search(game, iterations=4)
        self.assertFalse(info.fully_explored)

    def test_fd_choke_sbr(self):
        temperature_input = True
        single_temperature = True
        obs_input_temperature = True
        game_cfg = perform_choke_2_player(fully_connected=False, centered=True)
        if obs_input_temperature:
            game_cfg.ec.temperature_input = temperature_input
            game_cfg.ec.single_temperature_input = single_temperature

        eq_type = EquivarianceType.NONE
        # net_cfg = MobileNetConfig5x5(predict_policy=False, predict_game_len=False, game_cfg=game_cfg)
        net_cfg = ResNetConfig3x3(predict_policy=True, predict_game_len=False, eq_type=eq_type, lff_features=False,
                                  game_cfg=game_cfg)

        net_cfg.value_head_cfg.final_activation = ActivationType.TANH
        net_cfg.length_head_cfg.final_activation = ActivationType.SIGMOID
        net_cfg.film_temperature_input = (not obs_input_temperature) and temperature_input
        net_cfg.film_cfg = MediumHeadConfig() if net_cfg.film_temperature_input else None

        batch_size = 128
        eval_func_cfg = NetworkEvalConfig(
            net_cfg=net_cfg,
            max_batch_size=batch_size,
            random_symmetry=False,
            temperature_input=temperature_input,
            single_temperature=single_temperature,
            obs_temperature_input=obs_input_temperature,
        )
        backup_cfg = SBRBackupConfig(
            num_iterations=200,
        )

        extract_cfg = SpecialExtractConfig()

        search_cfg = FixedDepthConfig(
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_cfg,
            extract_func_cfg=extract_cfg,
            average_eval=False,
            discount=0.95,
        )
        search = get_search_from_config(search_cfg)
        search.set_temperatures([5, 5])
        game = get_game_from_config(game_cfg)
        game.step((0, 0))
        game.step((0, 0))

        values, probs, info = search(game, iterations=2)
        print(values)
        print(probs)
