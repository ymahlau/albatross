import unittest

import numpy as np

from src.agent.initialization import get_agent_from_config
from src.agent.overcooked import GreedyHumanOvercookedAgentConfig
from src.game.actions import sample_individual_actions
from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedSlowConfig


class TestOvercooked(unittest.TestCase):
    def test_human_greedy(self):
        game_cfg = CrampedRoomOvercookedSlowConfig()
        game = get_game_from_config(game_cfg)
        game.render()
        agent_cfg = GreedyHumanOvercookedAgentConfig(
            high_level_sampling=True,
            overcooked_layout=game_cfg.overcooked_layout,
        )
        agent = get_agent_from_config(agent_cfg)
        while not game.is_terminal():
            a0_probs, _ = agent(game, 0)
            a0 = sample_individual_actions(a0_probs[np.newaxis, ...], 1)[0]
            a1_probs, _ = agent(game, 1)
            a1 = sample_individual_actions(a1_probs[np.newaxis, ...], 1)[0]
            print((a0, a1))
            game.step((a0, a1))
            print('####################################################')
            game.render()
        print(f"{game.get_cum_rewards()=}")

