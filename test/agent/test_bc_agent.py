import unittest
from pathlib import Path

import numpy as np

from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import BCNetworkAgentConfig
from src.game.actions import sample_individual_actions
from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedConfig
from src.network.initialization import get_network_from_file


# class TestBCOvercookedAgent(unittest.TestCase):
#     def test_bc_oc(self):
#         game_cfg = CrampedRoomOvercookedConfig()
#         game = get_game_from_config(game_cfg)
#         game.render()
#
#         path = Path(__file__).parent.parent.parent / 'trained_models' / 'bc_oc.pt'
#         net = get_network_from_file(path)
#         agent_cfg = BCNetworkAgentConfig()
#         agent = get_agent_from_config(agent_cfg)
#         agent.replace_net(net)
#
#         while not game.is_terminal():
#             a0_probs, _ = agent(game=game, player=0)
#             a1_probs, _ = agent(game=game, player=1)
#             probs = np.asarray([a0_probs, a1_probs])
#             ja = sample_individual_actions(probs, 1)
#             game.step(ja)
#             game.render()
