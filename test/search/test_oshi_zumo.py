import unittest

from src.game.actions import sample_individual_actions
from src.game.initialization import get_game_from_config
from src.game.oshi_zumo.oshi_zumo import OshiZumoConfig, OshiZumoGame
from src.search.config import DecoupledUCTSelectionConfig, StandardBackupConfig, StandardExtractConfig, MCTSConfig, \
    OshiZumoEvalConfig
from src.search.mcts import MCTS


class TestOshiZumo(unittest.TestCase):
    def test_init(self):
        cfg = OshiZumoConfig(init_coins=10, board_size=2, min_bid=1)
        game = get_game_from_config(cfg)
        game.render()
        self.assertEqual(2, game.num_players_at_turn())
        self.assertEqual(2, game.num_players_alive())
        self.assertFalse(game.is_terminal())
        self.assertEqual(10, len(game.available_actions(0)))
        self.assertEqual(10, len(game.available_actions(1)))

        game.step((2, 4))
        game.render()
        game.step((3, 0))
        game.render()
        self.assertEqual(3, len(game.available_actions(0)))
        self.assertEqual(4, len(game.available_actions(1)))
        reward, done, _ = game.step((2, 0))
        game.render()
        self.assertEqual(-1, reward[0])
        self.assertEqual(1, reward[1])
        self.assertTrue(game.is_terminal())
        self.assertEqual(0, game.num_players_at_turn())
        self.assertEqual(0, game.num_players_alive())

    def test_zero_bid(self):
        cfg = OshiZumoConfig(init_coins=10, board_size=2, min_bid=0)
        game = get_game_from_config(cfg)
        game.render()
        self.assertEqual(2, game.num_players_at_turn())
        self.assertEqual(2, game.num_players_alive())
        self.assertFalse(game.is_terminal())
        self.assertEqual(11, len(game.available_actions(0)))
        self.assertEqual(11, len(game.available_actions(1)))

        game.step((0, 5))
        game.render()
        game.step((4, 0))
        game.render()

        rewards, _, _ = game.step((0, 0))
        self.assertEqual(0, rewards[0])
        self.assertEqual(0, rewards[1])
        self.assertTrue(game.is_terminal())

    def test_zero_bid_no_money(self):
        cfg = OshiZumoConfig(init_coins=2, board_size=1, min_bid=0)
        game = OshiZumoGame(cfg)
        game.render()
        self.assertFalse(game.is_terminal())
        game.step((2, 2))
        game.render()
        self.assertTrue(game.is_terminal())
        reward = game.compute_terminal_reward()
        self.assertEqual(0, reward[0])
        self.assertEqual(0, reward[1])

    def test_zero_bid_terminal(self):
        cfg = OshiZumoConfig(init_coins=2, board_size=1, min_bid=0)
        game = OshiZumoGame(cfg)
        game.render()
        self.assertFalse(game.is_terminal())
        game.step((0, 2))
        game.render()
        self.assertTrue(game.is_terminal())
        reward = game.compute_terminal_reward()
        self.assertEqual(1, reward[0])
        self.assertEqual(-1, reward[1])

    def test_obs(self):
        cfg = OshiZumoConfig(init_coins=10, board_size=2, min_bid=1)
        game = get_game_from_config(cfg)
        game.render()
        obs, _, _ = game.get_obs()
        self.assertEqual(2, len(obs.shape))
        self.assertEqual(2, obs.shape[0])
        self.assertEqual(25, obs.shape[1])
        self.assertEqual(25, game.get_obs_shape()[0])

        game.step((2, 4))
        game.render()
        obs, _, _ = game.get_obs()
        self.assertEqual(2, len(obs.shape))
        self.assertEqual(2, obs.shape[0])
        self.assertEqual(25, obs.shape[1])
        self.assertEqual(25, game.get_obs_shape()[0])
        self.assertEqual(1, obs[0, 11])
        self.assertEqual(1, obs[1, 9])

    def test_unary_encoding(self):
        cfg = OshiZumoConfig(init_coins=10, board_size=2, min_bid=1, unary_encoding=True)
        game = OshiZumoGame(cfg)
        game.render()
        obs, _, _ = game.get_obs()
        self.assertEqual(2, len(obs.shape))
        self.assertEqual(2, obs.shape[0])
        self.assertEqual(25, obs.shape[1])
        self.assertEqual(25, game.get_obs_shape()[0])
        for i in range(6, 16):
            self.assertAlmostEquals(1, obs[0, i].item())
            self.assertAlmostEquals(1, obs[1, i].item())
        game.step((2, 4))
        game.render()
        obs, _, _ = game.get_obs()
        print(obs)

    def test_search(self):
        cfg = OshiZumoConfig(init_coins=10, board_size=2, min_bid=1)
        game = get_game_from_config(cfg)

        sel_func_cfg = DecoupledUCTSelectionConfig(exp_bonus=1.414)
        eval_func_cfg = OshiZumoEvalConfig()
        backup_func_cfg = StandardBackupConfig()
        extract_func_cfg = StandardExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=False,
            optimize_fully_explored=False,
        )
        mcts = MCTS(mcts_cfg)
        game.render()
        while not game.is_terminal():
            values, policies, _ = mcts(game, iterations=500)
            actions = sample_individual_actions(policies, temperature=1)
            game.step(actions)
            game.render()
        print(game.players_at_turn())
