import unittest

import numpy as np

from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import LegalRandomAgentConfig
from src.game.actions import sample_individual_actions
from src.game.battlesnake.battlesnake import BattleSnakeGame, DOWN, LEFT
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.battlesnake_rewards import KillBattleSnakeRewardConfig
from src.game.battlesnake.bootcamp import survive_on_9x9_constrictor_4_player
from src.game.initialization import get_game_from_config


class TestRewards(unittest.TestCase):
    def test_win_no_legal(self):
        init_pos = {0: [[0, 1], [1, 1], [0, 2]], 1: [[2, 2], [2, 1], [2, 0]]}
        cfg = BattleSnakeConfig(w=3, h=3, num_players=2, constrictor=True, init_snake_pos=init_pos,
                                init_snake_len=[3, 3], init_snake_health=[5, 5], all_actions_legal=False)
        game = BattleSnakeGame(cfg)
        game.render()
        reward, done, info = game.step((DOWN, LEFT))
        game.render()
        self.assertTrue(done)
        self.assertEqual(1, reward[0])
        self.assertEqual(-1, reward[1])
        self.assertTrue(game.is_terminal())

    def test_multiplayer_standard_reward(self):
        game_cfg = survive_on_9x9_constrictor_4_player()
        game_cfg.all_actions_legal = True
        game = get_game_from_config(game_cfg)
        game.reset()
        game.render()
        game.step((0, 0, 0, 0))
        game.render()

        agent_cfg = LegalRandomAgentConfig()
        agent = get_agent_from_config(agent_cfg)
        while not game.is_terminal():
            actions = []
            for p in game.players_at_turn():
                probs, _ = agent(game=game, player=p)
                a = sample_individual_actions(probs[np.newaxis, ...], temperature=1)[0]
                actions.append(a)
            cur_reward, _, _ = game.step(tuple(actions))
            game.render()
            cum_rewards = game.get_cum_rewards()
            print(f"{cur_reward=}")
            print(f"{cum_rewards=}")
            for player in range(4):
                self.assertGreaterEqual(1, cum_rewards[player])
                self.assertLessEqual(-1, cum_rewards[player])
                self.assertGreaterEqual(1, cur_reward[player])
                self.assertLessEqual(-1, cur_reward[player])

    def test_multiplayer_kill_reward(self):
        game_cfg = survive_on_9x9_constrictor_4_player()
        game_cfg.all_actions_legal = True
        game_cfg.reward_cfg = KillBattleSnakeRewardConfig()
        game = get_game_from_config(game_cfg)
        num_games = 10

        for _ in range(num_games):
            game.reset()
            game.render()
            game.step((0, 0, 0, 0))
            game.render()

            agent_cfg = LegalRandomAgentConfig()
            agent = get_agent_from_config(agent_cfg)
            while not game.is_terminal():
                actions = []
                for p in game.players_at_turn():
                    probs, _ = agent(game=game, player=p)
                    a = sample_individual_actions(probs[np.newaxis, ...], temperature=1)[0]
                    actions.append(a)
                cur_reward, _, _ = game.step(tuple(actions))
                game.render()
                cum_rewards = game.get_cum_rewards()
                print(f"{cur_reward=}")
                print(f"{cum_rewards=}")
                for player in range(4):
                    self.assertGreaterEqual(1, cum_rewards[player])
                    self.assertLessEqual(-1, cum_rewards[player])
                    self.assertGreaterEqual(1, cur_reward[player])
                    self.assertLessEqual(-1, cur_reward[player])
            # terminal cumulative sum should be zero
            self.assertAlmostEqual(0, game.get_cum_rewards().sum().item())
