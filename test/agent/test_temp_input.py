import unittest

from src.agent.one_shot import NetworkAgentConfig, NetworkAgent
from src.agent.search_agent import SearchAgent, SearchAgentConfig, LookaheadAgentConfig, LookaheadAgent
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player
from src.network.fcn import MediumHeadConfig
from src.network.resnet import ResNetConfig5x5
from src.search.config import SampleSelectionConfig, NetworkEvalConfig, NashBackupConfig, SpecialExtractConfig, \
    MCTSConfig, StandardBackupConfig, StandardExtractConfig, AlphaZeroDecoupledSelectionConfig


class TestTemperatureInput(unittest.TestCase):
    def test_network_agent_single_obs(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.ec.temperature_input = True
        gc.ec.single_temperature_input = True
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=False,
                                     single_film_temperature=False)
        agent_cfg = NetworkAgentConfig(net_cfg=net_config, obs_temperature_input=True, single_temperature=True,
                                       init_temperatures=[1, 2, 3, 4], temperature_input=True)
        agent = NetworkAgent(agent_cfg)
        probs, _ = agent(env, 0)
        self.assertTrue(probs[0] > 0)

    def test_network_agent_multiple_obs(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.ec.temperature_input = True
        gc.ec.single_temperature_input = False
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=False,
                                     single_film_temperature=False)
        agent_cfg = NetworkAgentConfig(net_cfg=net_config, obs_temperature_input=True, single_temperature=False,
                                       init_temperatures=[1, 2, 3, 4], temperature_input=True)
        agent = NetworkAgent(agent_cfg)
        probs, _ = agent(env, 0)
        self.assertTrue(probs[0] > 0)

    def test_network_agent_single_film(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.ec.temperature_input = False
        gc.all_actions_legal = True
        env = BattleSnakeGame(gc)
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=True,
                                     single_film_temperature=True, film_cfg=MediumHeadConfig())
        # agent_cfg = NetworkAgentConfig(net_cfg=net_config, obs_temperature_input=False, single_temperature=True,
        #                                init_temperatures=[1, 2, 3, 4], temperature_input=True)
        agent_cfg = NetworkAgentConfig(net_cfg=net_config, obs_temperature_input=False, single_temperature=True,
                                       temperature_input=True)
        agent = NetworkAgent(agent_cfg)
        agent.set_temperatures([5 for _ in range(4)])
        probs, _ = agent(env, 0)
        self.assertTrue(probs[0] > 0)

    # def test_network_agent_multiple_film(self):
    #     gc = perform_choke_5x5_4_player(centered=True)
    #     gc.ec.temperature_input = False
    #     gc.all_actions_legal = True
    #     env = BattleSnakeGame(gc)
    #     net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=True,
    #                                  single_film_temperature=False, film_cfg=MediumHeadConfig())
    #     agent_cfg = NetworkAgentConfig(net_cfg=net_config, obs_temperature_input=False, single_temperature=False,
    #                                    init_temperatures=[1, 2, 3, 4], temperature_input=True)
    #     agent = NetworkAgent(agent_cfg)
    #     probs, _ = agent(env, 0)
    #     self.assertTrue(probs[0] > 0)

    def test_search_agent(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.ec.temperature_input = False
        gc.all_actions_legal = True
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=True,
                                     single_film_temperature=False, film_cfg=MediumHeadConfig())
        sel_func_cfg = AlphaZeroDecoupledSelectionConfig()
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config, single_temperature=False, obs_temperature_input=False,
                                          init_temperatures=[1, 2, 3, 4], temperature_input=True)
        backup_func_cfg = StandardBackupConfig()
        extract_func_cfg = StandardExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=0,
            use_hot_start=True,
            optimize_fully_explored=False,
        )
        agent_cfg = SearchAgentConfig(search_cfg=mcts_cfg)
        agent = SearchAgent(agent_cfg)
        game = BattleSnakeGame(gc)
        probs, _ = agent(game, 0, iterations=50)
        print(probs)
        self.assertTrue(probs[0] > 0)

    def test_lookahead_agent(self):
        gc = perform_choke_5x5_4_player(centered=True)
        gc.ec.temperature_input = False
        gc.all_actions_legal = True
        net_config = ResNetConfig5x5(game_cfg=gc, predict_policy=True, film_temperature_input=True,
                                     single_film_temperature=True, film_cfg=MediumHeadConfig())
        sel_func_cfg = SampleSelectionConfig()
        eval_func_cfg = NetworkEvalConfig(net_cfg=net_config, single_temperature=True, obs_temperature_input=False,
                                          init_temperatures=[1, 2, 3, 4], temperature_input=True)
        backup_func_cfg = NashBackupConfig()
        extract_func_cfg = SpecialExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=1,
            use_hot_start=True,
            optimize_fully_explored=False,
        )
        agent_cfg = LookaheadAgentConfig(
            search_cfg=mcts_cfg,
            init_temperatures=[1, 2, 3, 4],
            single_temperature=True,
            obs_temperature_input=False,
            net_cfg=net_config,
            search_depth=10,
            temperature_input=True,
        )
        agent = LookaheadAgent(agent_cfg)
        game = BattleSnakeGame(gc)
        probs, _ = agent(game, 0)
        self.assertTrue(probs[0] > 0)
