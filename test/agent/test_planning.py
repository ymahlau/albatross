import unittest

import numpy as np

from src.agent.initialization import get_agent_from_config
from src.agent.planning import AStarAgentConfig
from src.game.actions import sample_individual_actions
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig


class TestPlanning(unittest.TestCase):
    def test_a_star_fixed(self):
        player_pos = {0: [[0, 0]], 1: [[4, 4]]}
        food_pos = [[0, 4], [0, 3], [1, 4]]
        gc = BattleSnakeConfig(w=5, h=5, num_players=2, init_snake_pos=player_pos, init_food_pos=food_pos)
        game = BattleSnakeGame(gc)
        game.render()

        agent_cfg = AStarAgentConfig()
        agent = get_agent_from_config(agent_cfg)

        probs_1, info_1 = agent(game, 0)
        move_1 = sample_individual_actions(probs_1[np.newaxis, ...], 1)[0]
        self.assertTrue(info_1['success'])
        self.assertEqual(0, move_1)
        self.assertEqual(3, info_1['distance'])

        probs_2, info_2 = agent(game, 1)
        move_2 = sample_individual_actions(probs_2[np.newaxis, ...], 1)[0]
        self.assertTrue(info_2['success'])
        print(move_2)
        self.assertTrue(move_2 == 3 or move_2 == 2)
        self.assertEqual(3, info_2['distance'])

    def test_a_star_special(self):
        player_pos = {0: [[3, 4], [2, 4], [2, 3]], 1: [[2, 1], [1, 1], [1, 2]]}
        food_pos = [[2, 2], [0, 3]]
        gc = BattleSnakeConfig(w=5, h=5, num_players=2, init_snake_pos=player_pos, init_food_pos=food_pos)
        game = BattleSnakeGame(gc)
        game.render()

        agent_cfg = AStarAgentConfig()
        agent = get_agent_from_config(agent_cfg)

        probs_1, info_1 = agent(game, 0)
        move_1 = sample_individual_actions(probs_1[np.newaxis, ...], 1)[0]
        self.assertNotEqual(0, move_1)
        self.assertNotEqual(3, move_1)
