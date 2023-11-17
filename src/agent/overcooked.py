from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
from overcooked_ai_py.agents.agent import GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS

from src.agent import AgentConfig, Agent
from src.game.game import Game

import multiprocessing as mp

from src.game.overcooked_slow.overcooked import OvercookedGame


@dataclass(kw_only=True)
class GreedyHumanOvercookedAgentConfig(AgentConfig):
    """
    Agent that chooses a random legal action
    """
    overcooked_layout: str
    name: str = field(default="Greedy Human Overcooked")
    high_level_sampling: bool = False  # sample from high level goals
    high_level_temperature: int = 1
    low_level_sampling: bool = False  # sample from low level goals
    low_level_temperature: int = 1

class GreedyHumanOvercookedAgent(Agent):
    def __init__(self, cfg: GreedyHumanOvercookedAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.gridworld = OvercookedGridworld.from_layout_name(cfg.overcooked_layout)
        # we need action manager and proxy agent for both players (reasons are weird implementation details
        # regarding unstuck in the overcooked library)
        params = {
            'start_orientations': True,
            'wait_allowed': False,
            'counter_goals': [],
            'counter_drop': [],
            'counter_pickup': [],
            'same_motion_goals': True
        }
        self.action_manager0 = MediumLevelActionManager(
            mdp=self.gridworld,
            mlam_params=params
        )
        self.proxy_agent0 = GreedyHumanModel(
            mlam=self.action_manager0,
            hl_boltzmann_rational=self.cfg.high_level_sampling,
            ll_boltzmann_rational=self.cfg.low_level_sampling,
            hl_temp=self.cfg.high_level_temperature,
            ll_temp=self.cfg.low_level_temperature,
            auto_unstuck=True,
        )
        self.action_manager1 = MediumLevelActionManager(
            mdp=self.gridworld,
            mlam_params=NO_COUNTERS_PARAMS
        )
        self.proxy_agent1 = GreedyHumanModel(
            mlam=self.action_manager0,
            hl_boltzmann_rational=self.cfg.high_level_sampling,
            ll_boltzmann_rational=self.cfg.low_level_sampling,
            hl_temp=self.cfg.high_level_temperature,
            ll_temp=self.cfg.low_level_temperature,
            auto_unstuck=True,
        )
        self.proxy_agent1.set_agent_index(1)

    def _act(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs = None,  # mp.Array
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(game, OvercookedGame):
            raise ValueError(f"Human overcooked agent only works with overcooked game")
        # we need to set the mdp of the agent again in case environment got reset
        self.proxy_agent0.set_mdp(self.gridworld)
        self.proxy_agent0.set_agent_index(0)
        self.proxy_agent1.set_mdp(self.gridworld)
        self.proxy_agent1.set_agent_index(1)
        # we can use player instead of player index, because overcooked has only two players
        if game.env is None:
            raise Exception("game.env is None")
        if player == 0:
            action_overcooked, _ = self.proxy_agent0.action(game.env.state)
        else:
            action_overcooked, _ = self.proxy_agent1.action(game.env.state)
        action = Action.ALL_ACTIONS.index(action_overcooked)
        probs = np.zeros(shape=(game.num_actions,), dtype=float)
        probs[action] = 1
        return probs, {}
