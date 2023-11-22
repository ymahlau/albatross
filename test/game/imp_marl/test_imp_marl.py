import unittest
from src.game.imp_marl.imp_marl_wrapper import IMP_MODE, IMPConfig
from src.game.initialization import get_game_from_config

class TestIMPMARL(unittest.TestCase):
    def test_simple(self):
        game_cfg = IMPConfig(num_players=3, imp_mode=IMP_MODE.K_OF_N, campaign_cost=False)
        game = get_game_from_config(game_cfg)
        rewards, done, _ = game.step((0, 0, 2))
        print(rewards)
        while not game.is_terminal():
            # print(game.get_obs_shape())
            # print(game.get_obs())
            rewards, done, _ = game.step((0, 0, 0))
            print(rewards)
        print(game.turns_played)
        print(game.get_cum_rewards())
        