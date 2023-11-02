import unittest

from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.initialization import get_game_from_config
from src.network.initialization import get_network_from_config
from src.network.resnet import ResNetConfig3x3
from src.network.vision_net import EquivarianceType


class TestLearnedFourierFeatures(unittest.TestCase):
    def test_lff_choke(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = ResNetConfig3x3(
            game_cfg=game_cfg,
            predict_policy=True,
            predict_game_len=True,
            eq_type=EquivarianceType.NONE,
            lff_features=True,
        )
        net = get_network_from_config(net_cfg)
        net.eval()
        game = get_game_from_config(game_cfg)

        obs, _, _, = game.get_obs()
        out = net(obs)

        self.assertEqual(2, len(out.shape))
        self.assertEqual(6, out.shape[1])




