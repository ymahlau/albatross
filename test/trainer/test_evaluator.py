# import math
# import unittest
# from pathlib import Path
#
# import numpy as np
#
# from src.agent.search_agent import SBRFixedDepthAgentConfig, SearchAgent
# from src.game.actions import sample_individual_actions
# from src.game.initialization import get_game_from_config
# from src.network.initialization import get_network_from_file
#
#
# class TestEvaluator(unittest.TestCase):
#     def test_evaluation_choke(self):
#         net_path = Path(__file__).parent.parent.parent / 'trained_models' / 'choke.pt'
#         net = get_network_from_file(net_path)
#         net.eval()
#         game = get_game_from_config(net.cfg.game_cfg)
#         game.reset()
#         game.render()
#
#         value_agent_cfg = SBRFixedDepthAgentConfig()
#         value_agent = SearchAgent(value_agent_cfg)
#         value_agent.replace_net(net)
#         value_agent.search.set_temperatures([5, 5])
#
#         probs, _ = value_agent(game, player=0, iterations=1)
#         a = sample_individual_actions(probs[np.newaxis, ...], temperature=math.inf)[0]
#         self.assertEqual(0, a)
#
#         game.step((0, 0))
#         game.render()
#
#         probs, _ = value_agent(game, player=0, iterations=1)
#         a = sample_individual_actions(probs[np.newaxis, ...], temperature=math.inf)[0]
#         self.assertEqual(0, a)
