import unittest
from typing import Optional

import numpy as np

from src.agent import Agent
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import RandomAgentConfig, NetworkAgentConfig, LegalRandomAgentConfig
from src.agent.planning import AStarAgentConfig
from src.agent.search_agent import SearchAgentConfig, TwoPlayerEvalAgentConfig
from src.game.actions import sample_individual_actions
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.bootcamp.test_envs_11x11 import survive_on_11x11
from src.game.battlesnake.bootcamp.test_envs_5x5 import survive_on_5x5
from src.game.game import Game
from src.game.initialization import get_game_from_config
from src.network.resnet import ResNetConfig5x5
from src.search.backup_func import NashBackupConfig, MaxMinBackupConfig
from src.search.config import MaxAvgBackupConfig
from src.search.eval_func import AreaControlEvalConfig
from src.search.extraction_func import SpecialExtractConfig
from src.search.mcts import MCTSConfig
from src.search.sel_func import SampleSelectionConfig


class TestOneShot(unittest.TestCase):
    @staticmethod
    def play_game_self(
            agent: Agent,
            game: Game,
            length: int = 10,
            time_limit: Optional[float] = 0.5,
            iterations: Optional[int] = None
    ):
        game.render()
        counter = 0
        while not game.is_terminal() and counter < length:
            actions = []
            for player in game.players_at_turn():
                probs, info = agent(game, player, time_limit=time_limit, iterations=iterations)
                if hasattr(agent, "search"):
                    print(f"{agent.search.root.visits}")
                a = sample_individual_actions(probs[np.newaxis, ...], 1)[0]
                actions.append(a)
            game.step(tuple(actions))
            game.render()
            counter += 1

    def test_search_agent(self):
        sel_func_cfg = SampleSelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = NashBackupConfig()
        extract_func_cfg = SpecialExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=1,
            use_hot_start=True,
            optimize_fully_explored=True,
        )
        agent_conf = SearchAgentConfig(search_cfg=mcts_cfg)
        agent = get_agent_from_config(agent_conf)
        gc = survive_on_11x11()
        game = BattleSnakeGame(gc)
        self.play_game_self(agent, game, length=100)

    def test_maxmin_search_agent(self):
        sel_func_cfg = SampleSelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = MaxMinBackupConfig()
        extract_func_cfg = SpecialExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=1,
            use_hot_start=True,
            optimize_fully_explored=True,
        )
        agent_conf = SearchAgentConfig(search_cfg=mcts_cfg)
        agent = get_agent_from_config(agent_conf)
        gc = survive_on_5x5()
        game = BattleSnakeGame(gc)
        self.play_game_self(agent, game, length=100)

    def test_maxavg_search_agent(self):
        sel_func_cfg = SampleSelectionConfig()
        eval_func_cfg = AreaControlEvalConfig()
        backup_func_cfg = MaxAvgBackupConfig()
        extract_func_cfg = SpecialExtractConfig()
        mcts_cfg = MCTSConfig(
            sel_func_cfg=sel_func_cfg,
            eval_func_cfg=eval_func_cfg,
            backup_func_cfg=backup_func_cfg,
            extract_func_cfg=extract_func_cfg,
            expansion_depth=1,
            use_hot_start=True,
            optimize_fully_explored=True,
        )
        agent_conf = SearchAgentConfig(search_cfg=mcts_cfg)
        agent = get_agent_from_config(agent_conf)
        gc = survive_on_5x5()
        game = BattleSnakeGame(gc)
        self.play_game_self(agent, game, length=100)

    def test_random_agent(self):
        agent_conf = RandomAgentConfig()
        agent = get_agent_from_config(agent_conf)
        gc = survive_on_11x11()
        game = BattleSnakeGame(gc)
        self.play_game_self(agent, game)

    def test_network_agent(self):
        game_cfg = survive_on_5x5()
        game_cfg.all_actions_legal = False
        game = get_game_from_config(game_cfg)
        net_cfg = ResNetConfig5x5(game_cfg=game_cfg, predict_policy=True)
        agent_cfg = NetworkAgentConfig(net_cfg=net_cfg, random_symmetry=True)
        agent = get_agent_from_config(agent_cfg)
        self.play_game_self(agent, game, length=100)

    def test_lookahead_agent(self):
        game_cfg = survive_on_5x5()
        game_cfg.all_actions_legal = False
        game = get_game_from_config(game_cfg)
        net_cfg = ResNetConfig5x5(game_cfg=game_cfg, predict_policy=True)
        agent_cfg = TwoPlayerEvalAgentConfig(net_cfg=net_cfg, search_depth=1, search_weight=0.5)
        agent = get_agent_from_config(agent_cfg)
        self.play_game_self(agent, game, length=100)

    def test_a_star_agent(self):
        game_cfg = survive_on_5x5()
        game_cfg.all_actions_legal = False
        game = get_game_from_config(game_cfg)
        agent_cfg = AStarAgentConfig()
        agent = get_agent_from_config(agent_cfg)
        self.play_game_self(agent, game, length=100)

    def test_legal_random_agent(self):
        game_cfg = survive_on_5x5()
        game_cfg.all_actions_legal = True
        game = get_game_from_config(game_cfg)
        agent_cfg = LegalRandomAgentConfig()
        agent = get_agent_from_config(agent_cfg)
        self.play_game_self(agent, game, length=100)
