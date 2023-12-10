import copy
import pickle
import random
from dataclasses import field, dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch

from src.agent import AgentConfig, Agent
from src.game.game import Game
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.overcooked.overcooked import OvercookedGame
from src.modelling.mle import compute_all_likelihoods, compute_likelihood, compute_temperature_mle
from src.network.initialization import get_network_from_file
from src.search.config import FixedDepthConfig, NetworkEvalConfig, LogitBackupConfig, SpecialExtractConfig
from src.search.fixed_depth import FixedDepthSearch
from src.search.utils import compute_q_values


@dataclass(kw_only=True)
class AlbatrossAgentConfig(AgentConfig):
    num_player: int
    agent_cfg: AgentConfig
    device_str: str
    response_net_path: str
    proxy_net_path: str
    name: str = 'Albatross'
    min_temp: float = 0
    max_temp: float = 10
    init_temp: float = 10
    num_iterations: int = 10  # only used for mle
    fixed_temperatures: Optional[list[float]] = None
    estimate_log_path: Optional[str] = None
    noise_std: Optional[float] = None  # if not None, sample from normal
    additive_estimate_offset: float = 0
    sample_from_likelihood: bool = False
    num_likelihood_bins: int = int(1e4)
    num_samples: int = 1


class AlbatrossAgent(Agent):
    def __init__(self, cfg: AlbatrossAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        # init
        self.device = torch.device(self.cfg.device_str)
        from src.agent.initialization import get_agent_from_config  # local import to avoid circular import error
        self.agent = get_agent_from_config(self.cfg.agent_cfg)
        self.proxy_search = self._get_proxy_search()
        self.response_net = get_network_from_file(Path(self.cfg.response_net_path)).to(self.device)
        self.proxy_net = get_network_from_file(Path(self.cfg.proxy_net_path)).to(self.device)
        # replace net and device
        self.agent.replace_net(self.response_net)
        self.agent.replace_device(self.device)
        self.proxy_search.replace_net(self.proxy_net)
        self.proxy_search.replace_device(self.device)
        # temperature estimation attributes
        self.enemy_actions: dict[int, list[int]] = {p: [] for p in range(self.cfg.num_player)}
        self.enemy_util: dict[int, list[list[float]]] = {p: [] for p in range(self.cfg.num_player)}
        self.temp_estimates: dict[int, list[float]] = {p: [] for p in range(self.cfg.num_player)}
        self.last_player_at_turn: list[int] = []
        self.last_available_actions: dict[int, list[int]] = {}
        self.bins = np.linspace(self.cfg.min_temp, self.cfg.max_temp, self.cfg.num_likelihood_bins)
        if self.cfg.num_samples < 1:
            raise ValueError(f"Invalid sample number: {self.cfg.num_samples}")
        if self.cfg.noise_std is not None and self.cfg.sample_from_likelihood:
            raise ValueError("Cannot sample from both normal dist and likelihood")
        if self.cfg.fixed_temperatures is not None and self.cfg.sample_from_likelihood:
            raise ValueError("cannot sample from likelihood and use fixed temperatures")
        if self.cfg.noise_std is None and not self.cfg.sample_from_likelihood:
            if self.cfg.num_samples > 1:
                raise ValueError("Cannot use multiple samples if no samples is specified")

    def _act(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs = None,  # mp.Array
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # observe last actions taken
        # base_temperatures: list[float], one per player
        base_temperatures = []
        if game.turns_played == 0:
            # use initial temperatures
            base_temperatures = [self.cfg.init_temp for _ in range(game.num_players)]
        else:
            # parse last actions for temperature estimation
            last_actions = game.get_last_action()
            for p_idx, p in enumerate(self.last_player_at_turn):
                if last_actions is None:
                    raise Exception(f"Last action is None")
                action_idx = self.last_available_actions[p].index(last_actions[p_idx])
                self.enemy_actions[p].append(action_idx)
        if self.cfg.fixed_temperatures is not None:  # fixed temperatures
            base_temperatures = self.cfg.fixed_temperatures
        elif self.cfg.noise_std is not None and game.turns_played != 0:
            # do mle for all enemies
            base_temperatures = []
            for p in range(game.num_players):
                if p == player or not game.is_player_at_turn(p):
                    base_temperatures.append(0)  # we use temperature of zero for own player (does not matter)
                    continue
                temp_estimate = compute_temperature_mle(
                    min_temp=self.cfg.min_temp,
                    max_temp=self.cfg.max_temp,
                    num_iterations=self.cfg.num_iterations,
                    chosen_actions=self.enemy_actions[p],
                    utilities=self.enemy_util[p],
                )
                base_temperatures.append(temp_estimate)
        assert base_temperatures
        # sampling
        temperatures = None
        if self.cfg.noise_std is None and not self.cfg.sample_from_likelihood:
            # do not sample, just use base temperatures
            temperatures = np.asarray([base_temperatures])
        elif self.cfg.sample_from_likelihood:
            pass
        else:  # sample from normal
            pass
        
        if self.cfg.fixed_temperatures is not None:
            # use fixed temperatures
            temperatures = self.cfg.fixed_temperatures
            if self.cfg.noise_std is not None and self.cfg.num_samples == 1:
                for p in range(game.num_players):
                    noise = np.random.normal(0, self.cfg.noise_std)
                    temperatures[p] += noise
                    temperatures[p] = np.clip(temperatures[p], self.cfg.min_temp, self.cfg.max_temp).item()
        elif game.turns_played == 0:
            # use initial temperatures
            temperatures = [self.cfg.init_temp for _ in range(game.num_players)]
        elif self.cfg.sample_from_likelihood:
            # parse last actions for temperature estimation
            last_actions = game.get_last_action()
            for p_idx, p in enumerate(self.last_player_at_turn):
                if last_actions is None:
                    raise Exception(f"Last action is None")
                action_idx = self.last_available_actions[p].index(last_actions[p_idx])
                self.enemy_actions[p].append(action_idx)
            for p in range(game.num_players):
                if p == player or not game.is_player_at_turn(p):
                    probs.append(None)  # we use temperature of zero for own player (does not matter)
                    continue
                cur_probs = np.asarray(
                    compute_all_likelihoods(
                        chosen_actions=self.enemy_actions[p],
                        utilities=self.enemy_util[p],
                        min_temp=self.cfg.min_temp,
                        max_temp=self.cfg.max_temp,
                        resolution=self.cfg.num_likelihood_bins,
                    )
                )
                # cur_probs = np.zeros_like(bins)
                # for t_idx, t in enumerate(bins):
                #     cur_likelihood = compute_likelihood(
                #         temperature=t,
                #         chosen_actions=self.enemy_actions[p],
                #         utilities=self.enemy_util[p],
                #     )
                #     cur_probs[t_idx] = cur_likelihood
                cur_probs = np.exp(cur_probs)
                cur_probs = cur_probs / np.sum(cur_probs)  # normalize
                probs.append(cur_probs)
        else:
            # parse last actions for temperature estimation
            last_actions = game.get_last_action()
            for p_idx, p in enumerate(self.last_player_at_turn):
                if last_actions is None:
                    raise Exception(f"Last action is None")
                action_idx = self.last_available_actions[p].index(last_actions[p_idx])
                self.enemy_actions[p].append(action_idx)
            # do mle for all enemies
            temperatures = []
            for p in range(game.num_players):
                if p == player or not game.is_player_at_turn(p):
                    temperatures.append(0)  # we use temperature of zero for own player (does not matter)
                    continue
                temp_estimate = compute_temperature_mle(
                    min_temp=self.cfg.min_temp,
                    max_temp=self.cfg.max_temp,
                    num_iterations=self.cfg.num_iterations,
                    chosen_actions=self.enemy_actions[p],
                    utilities=self.enemy_util[p],
                )
                # print(f"{temp_estimate=}")
                if self.cfg.noise_std is not None and self.cfg.num_samples == 1:
                    noise = np.random.normal(0, self.cfg.noise_std)
                    temp_estimate += noise
                    temp_estimate = np.clip(temp_estimate, self.cfg.min_temp, self.cfg.max_temp).item()
                temp_estimate += self.cfg.additive_estimate_offset
                temperatures.append(temp_estimate)
                self.temp_estimates[p].append(temp_estimate)
        # copy game
        cpy = game.get_copy()
        if isinstance(game, BattleSnakeGame):
            cpy.cfg = copy.deepcopy(game.cfg)
            cpy.cfg.ec.temperature_input = True
            cpy.cfg.ec.single_temperature_input = False
        elif isinstance(game, OvercookedGame):
            cpy.cfg = copy.deepcopy(game.cfg)
            cpy.cfg.temperature_input = True
            cpy.cfg.single_temperature_input = False
        else:
            raise NotImplementedError()
        # iterate samples
        action_prob_list = []
        sampled_temps = []
        for p in range(game.num_players):
            if p == player or not game.is_player_at_turn(p):
                sampled_temps.append(np.zeros(shape=(self.cfg.num_samples,), dtype=float))  # we use temperature of zero for own player (does not matter)
            else:
                sampled_temps.append(np.random.choice(bins, size=(self.cfg.num_samples,), replace=True, p=probs[player]))
        for sample_idx in range(self.cfg.num_samples):
            if sampled_temps:
                cur_temp = [t[sample_idx].item() for t in sampled_temps]
                self.agent.set_temperatures(cur_temp)
            else:
                if not temperatures:
                    raise Exception('tthis should never happen')
                temp_arr = np.asarray(temperatures, dtype=float)
                cur_temp = temp_arr.copy()
                if self.cfg.noise_std is not None:
                    noise = np.random.normal(0, self.cfg.noise_std, len(temperatures))
                    cur_temp += noise
                self.agent.set_temperatures(list(cur_temp))
            cur_action_probs, info_dict = self.agent(
                game=cpy,
                time_limit=time_limit,
                iterations=iterations,
                save_probs=save_probs,
                player=player,
                options=options,
            )
            action_prob_list.append(cur_action_probs)
        action_probs = np.asarray(action_prob_list, dtype=float).mean(axis=0)
        # compute and update q
        q_dict = self._compute_q_values(cpy)
        for p in game.players_at_turn():
            self.enemy_util[p].append(q_dict[p])
        # save player at turn
        self.last_player_at_turn = game.players_at_turn()
        self.last_available_actions = {
            p: game.available_actions(p) for p in game.players_at_turn()
        }
        info_dict = {
            "all_action_probs": action_probs,
        }
        return action_probs, info_dict

    def _compute_q_values(
            self,
            game: Game,
    ) -> dict[int, list[float]]:
        if isinstance(game, BattleSnakeGame):
            game.cfg.ec.temperature_input = True
            game.cfg.ec.single_temperature_input = True
        elif isinstance(game, OvercookedGame):
            game.cfg.temperature_input = True
            game.cfg.single_temperature_input = True
        # search
        _, action_probs, _ = self.proxy_search(game, iterations=1)
        # compute q from root
        q_dict: dict[int, list[float]] = {}
        for p in game.players_at_turn():
            if self.proxy_search.root is None:
                raise Exception("Proxy Search root is None")
            q_dict[p] = compute_q_values(self.proxy_search.root, p, action_probs)
        return q_dict

    def reset_episode(self):
        if self.cfg.estimate_log_path is not None:
            for v_list in self.temp_estimates.values():
                if v_list:
                    rnd = random.randint(0, 2 ** 32 - 1)
                    with open(self.cfg.estimate_log_path + f"{rnd}.pkl", "wb") as f:
                        pickle.dump(v_list, f)
        self.temp_estimates = {p: [] for p in range(self.cfg.num_player)}
        self.enemy_actions = {p: [] for p in range(self.cfg.num_player)}
        self.enemy_util = {p: [] for p in range(self.cfg.num_player)}
        self.last_player_at_turn = []

    def _get_proxy_search(self) -> FixedDepthSearch:
        temperatures = [self.cfg.max_temp for _ in range(self.cfg.num_player)]
        search_cfg = FixedDepthConfig(
            eval_func_cfg=NetworkEvalConfig(
                temperature_input=True,
                single_temperature=True,
                init_temperatures=temperatures,
            ),
            backup_func_cfg=LogitBackupConfig(
                num_iterations=100,
                init_temperatures=temperatures,
            ),
            extract_func_cfg=SpecialExtractConfig()
        )
        search = FixedDepthSearch(search_cfg)
        return search
