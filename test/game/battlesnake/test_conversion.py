import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame, UP
from src.game.battlesnake.enc_conversion import decode_encoding
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player


class TestConversion(unittest.TestCase):
    def test_choke_initial(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        game = BattleSnakeGame(game_cfg)
        obs, _, _ = game.get_obs(0)
        new_game_cfg = decode_encoding(game_cfg, obs[0])
        self.assertEqual(1, new_game_cfg.init_snake_pos[0][0][0])
        self.assertEqual(0, new_game_cfg.init_snake_pos[0][0][1])
        self.assertEqual(2, new_game_cfg.init_snake_pos[1][0][0])
        self.assertEqual(0, new_game_cfg.init_snake_pos[1][0][1])
        new_game_cfg = decode_encoding(game_cfg, obs[1])
        self.assertEqual(2, new_game_cfg.init_snake_pos[0][0][0])
        self.assertEqual(0, new_game_cfg.init_snake_pos[0][0][1])
        self.assertEqual(1, new_game_cfg.init_snake_pos[1][0][0])
        self.assertEqual(0, new_game_cfg.init_snake_pos[1][0][1])

    def test_choke_in_game(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        game = BattleSnakeGame(game_cfg)
        game.step((UP, UP))
        obs, _, _ = game.get_obs(0)
        new_game_cfg = decode_encoding(game_cfg, obs[0])
        self.assertEqual(1, new_game_cfg.init_snake_pos[0][0][0])
        self.assertEqual(1, new_game_cfg.init_snake_pos[0][0][1])
        self.assertEqual(2, new_game_cfg.init_snake_pos[1][0][0])
        self.assertEqual(1, new_game_cfg.init_snake_pos[1][0][1])
