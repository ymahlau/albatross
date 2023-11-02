# import random
# import unittest
# from pathlib import Path
#
# import numpy as np
#
# from src.agent.albatross import AlbatrossAgentConfig
# from src.agent.initialization import get_agent_from_config
# from src.agent.one_shot import NetworkAgentConfig
# from src.agent.search_agent import SearchAgentConfig
# from src.game.actions import sample_individual_actions
# from src.game.battlesnake.battlesnake import BattleSnakeGame
# from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7_constrictor
# from src.game.initialization import get_game_from_config
# from src.search.config import FixedDepthConfig, EnemyExploitationEvalConfig, EnemyExploitationBackupConfig, \
#     SpecialExtractConfig
#
#
# class TestAlbatrossAgent(unittest.TestCase):
#     def test_albatross_agent_d7_fixed_temp(self):
#         game_cfg = survive_on_7x7_constrictor()
#         game_cfg.ec.temperature_input = True
#         game_cfg.ec.single_temperature_input = False
#         game = get_game_from_config(game_cfg)
#
#         proxy_path = Path(__file__).parent.parent.parent / 'trained_models' / 'd7_proxy.pt'
#         response_path = Path(__file__).parent.parent.parent / 'trained_models' / 'd7_response_inf.pt'
#
#         eval_conf = EnemyExploitationEvalConfig(
#             enemy_net_path=str(proxy_path),
#             obs_temperature_input=True,
#             net_cfg=None,
#         )
#         backup_conf = EnemyExploitationBackupConfig(
#             enemy_net_path=str(proxy_path),
#             exploit_temperature=5,
#         )
#         extract_conf = SpecialExtractConfig()
#         search_cfg = FixedDepthConfig(
#             eval_func_cfg=eval_conf,
#             backup_func_cfg=backup_conf,
#             extract_func_cfg=extract_conf,
#             average_eval=True,
#         )
#         search_agent_cfg = SearchAgentConfig(search_cfg=search_cfg)
#
#         agent_cfg = AlbatrossAgentConfig(
#             num_player=2,
#             agent_cfg=search_agent_cfg,
#             device_str='cpu',
#             response_net_path=str(response_path),
#             proxy_net_path=str(proxy_path),
#             fixed_temperatures=[5, 5]
#         )
#         agent = get_agent_from_config(agent_cfg)
#         game.render()
#         action_probs, info = agent(
#             game=game,
#             player=0,
#             iterations=1,
#         )
#         print(f"{action_probs=}")
#         print(f"{info=}")
#
#     def test_albatross_agent_d7_online(self):
#         game_cfg = survive_on_7x7_constrictor()
#         game_cfg.ec.temperature_input = True
#         game_cfg.ec.single_temperature_input = False
#         game = BattleSnakeGame(game_cfg)
#
#         proxy_path = Path(__file__).parent.parent.parent / 'trained_models' / 'd7_proxy.pt'
#         response_path = Path(__file__).parent.parent.parent / 'trained_models' / 'd7_response_inf.pt'
#
#         eval_conf = EnemyExploitationEvalConfig(
#             enemy_net_path=str(proxy_path),
#             obs_temperature_input=True,
#             net_cfg=None,
#         )
#         backup_conf = EnemyExploitationBackupConfig(
#             enemy_net_path=str(proxy_path),
#             exploit_temperature=5,
#         )
#         extract_conf = SpecialExtractConfig()
#         search_cfg = FixedDepthConfig(
#             eval_func_cfg=eval_conf,
#             backup_func_cfg=backup_conf,
#             extract_func_cfg=extract_conf,
#             average_eval=True,
#         )
#         search_agent_cfg = SearchAgentConfig(search_cfg=search_cfg)
#
#         agent_cfg = AlbatrossAgentConfig(
#             num_player=2,
#             agent_cfg=search_agent_cfg,
#             device_str='cpu',
#             response_net_path=str(response_path),
#             proxy_net_path=str(proxy_path),
#         )
#         agent = get_agent_from_config(agent_cfg)
#
#         while not game.is_terminal():
#             game.render()
#             print(f"{game.player_pos(0)=}")
#             action_probs, info = agent(
#                 game=game,
#                 player=0,
#                 iterations=1,
#             )
#             agent_action = sample_individual_actions(action_probs[np.newaxis, :], temperature=1)[0]
#             print(f"{action_probs=}")
#             print(f"{info=}")
#             random_action = random.choice(game.available_actions(1))
#             ja = (agent_action, random_action)
#             print(f"{ja=}")
#             print('###############################################')
#             game.step(ja)
#         game.render()
#
#     def test_albatross_agent_d7_online_pure_network(self):
#         game_cfg = survive_on_7x7_constrictor()
#         game_cfg.ec.temperature_input = True
#         game_cfg.ec.single_temperature_input = False
#         game = BattleSnakeGame(game_cfg)
#
#         proxy_path = Path(__file__).parent.parent.parent / 'trained_models' / 'd7_proxy.pt'
#         response_path = Path(__file__).parent.parent.parent / 'trained_models' / 'd7_response_inf.pt'
#
#         agent_cfg = NetworkAgentConfig(
#             temperature_input=True,
#             single_temperature=False,
#             obs_temperature_input=True,
#         )
#
#         agent_cfg = AlbatrossAgentConfig(
#             num_player=2,
#             agent_cfg=agent_cfg,
#             device_str='cpu',
#             response_net_path=str(response_path),
#             proxy_net_path=str(proxy_path),
#         )
#         agent = get_agent_from_config(agent_cfg)
#
#         while not game.is_terminal():
#             game.render()
#             print(f"{game.player_pos(0)=}")
#             action_probs, info = agent(
#                 game=game,
#                 player=0,
#                 iterations=1,
#             )
#             agent_action = sample_individual_actions(action_probs[np.newaxis, :], temperature=1)[0]
#             print(f"{action_probs=}")
#             print(f"{info=}")
#             random_action = random.choice(game.available_actions(1))
#             ja = (agent_action, random_action)
#             print(f"{ja=}")
#             print('###############################################')
#             game.step(ja)
#         game.render()
