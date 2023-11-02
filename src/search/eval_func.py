import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from lightning import Fabric
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS

from src.game.actions import apply_permutation, filter_illegal_and_normalize
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.oshi_zumo.oshi_zumo import OshiZumoGame
from src.game.overcooked.overcooked import OvercookedGame
from src.game.values import apply_zero_sum_norm
from src.network import Network
from src.network.initialization import get_network_from_config, network_config_from_structured, get_network_from_file
from src.search.config import NetworkEvalConfig, CopyCatEvalConfig, AreaControlEvalConfig, EvalFuncConfig, \
    EvalFuncType, DummyEvalConfig, OshiZumoEvalConfig, EnemyExploitationEvalConfig, RandomRolloutEvalConfig, \
    OvercookedPotentialEvalConfig
from src.search.node import Node


class EvalFunc(ABC):
    def __init__(self, cfg: EvalFuncConfig):
        self.cfg = cfg

    def __call__(self, node_list: list[Node]) -> None:
        heuristic_eval_list = []
        # filter terminal nodes
        for node in node_list:
            # sanity checks
            if node.visits > 0:
                raise Exception("Evaluation of an already visited node. This should never happen.")
            if node.is_terminal():
                node.visits = 1
                node.value_sum *= 0
            else:
                heuristic_eval_list.append(node)
        if not heuristic_eval_list:
            return  # all nodes given were terminal, no need to compute heuristic
        self._compute(heuristic_eval_list)

    @abstractmethod
    def _compute(self, nodel_list: list[Node]) -> None:
        # Computes estimate of the expected (discounted) reward sum from this state
        # and updates info dict containing additional information for later use (e.g. action tensor of network)
        raise NotImplementedError()

    def update_node(self, node: Node, values: np.ndarray):
        node.visits = 1
        # optionally squash values to be zero-sum
        norm_vals = apply_zero_sum_norm(values, self.cfg.zero_sum_norm)
        node.value_sum = norm_vals
        for player in node.game.players_not_at_turn():
            if node.value_sum[player] != 0:
                node.game.render()
                raise Exception(f"Player {player} has value {node.value_sum[player]}")


class AreaControlEvalFunc(EvalFunc):
    """
    GameState evaluation function, which does an area control evaluation. Area control is weighted by health.
    """

    def __init__(self, cfg: AreaControlEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute(self, node_list: list[Node]) -> None:
        for node in node_list:
            if not isinstance(node.game, BattleSnakeGame):
                raise Exception("Can only use Area control eval with battlesnake game")
            game: BattleSnakeGame = node.game
            ac_dict = node.game.area_control()
            ac = ac_dict["area_control"]
            ac_at_turn = ac[node.game.players_at_turn()]
            ac_at_turn_relative = (ac_at_turn - (np.sum(ac_at_turn) / node.game.num_players_at_turn()))
            ac_board_relative = ac_at_turn_relative / (game.cfg.w * game.cfg.h)

            health_arr = np.asarray(game.player_healths()).astype(float)
            health_relative = health_arr / np.asarray(node.game.cfg.max_snake_health)
            health_clipped = np.minimum(1.0, health_relative / self.cfg.health_threshold)
            health_at_turn = health_clipped[node.game.players_at_turn()]
            health_factor = health_at_turn - (np.sum(health_at_turn) / node.game.num_players_at_turn())

            result = (ac_board_relative + health_factor) / 2
            values = np.zeros(shape=(node.game.num_players,), dtype=float)
            values[node.game.players_at_turn()] = result
            self.update_node(node, values)


class CopyCatEvalFunc(EvalFunc):
    """
    GameState evaluation function for MCTS which does an area control evaluation.
    Inspired by https://github.com/m-schier/battlesnake-2019
    """

    def __init__(self, cfg: CopyCatEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute(self, node_list: list[Node]) -> None:
        for node in node_list:
            if not isinstance(node.game, BattleSnakeGame):
                raise ValueError("Can only use Area control eval with battlesnake game")
            values = self._copy_cat_eval_game(node.game)
            self.update_node(node, values)

    @staticmethod
    def _score_delta_advantage(x: np.ndarray):
        return (1.0 / (1.0 + np.exp(-(x * 0.2))) - 0.5) * 2.0

    def _copy_cat_eval_game(self, game: BattleSnakeGame) -> np.ndarray:
        # Get required information
        ac_dict = game.area_control(
            hazard_weight=self.cfg.hazard_weight,
            weight=self.cfg.tile_weight,
            food_weight=self.cfg.food_weight,
            food_in_hazard_weight=self.cfg.food_hazard_weight,
        )
        area_control = ac_dict["area_control"]
        food_distance = ac_dict["food_distance"].astype(float)
        food_reachable = ac_dict["food_reachable"]  # bool
        tail_reachable = ac_dict["tail_reachable"]  # bool
        snake_lengths = np.asarray(game.player_lengths(), dtype=float)
        snake_healths = np.asarray(game.player_healths(), dtype=float)
        players_at_turn = game.players_at_turn()
        num_at_turn = game.num_players_at_turn()
        # filter players not at turn
        turn_filter = np.zeros(shape=(game.num_players,), dtype=bool)
        turn_filter[players_at_turn] = True
        not_turn_filter = np.logical_not(turn_filter)
        # these metrics need filtering, all other already consider dead snakes
        snake_lengths[not_turn_filter] = 0
        snake_healths[not_turn_filter] = 0
        # delta length food metric, computes
        # - average lengths of other snakes
        # - max lengths of other snakes
        # avg
        other_length_sum = snake_lengths.sum() - snake_lengths
        other_length_avg = other_length_sum / (num_at_turn - 1)
        # max
        max_len_idx = np.argmax(snake_lengths)
        other_length_max = np.ones(shape=(game.num_players,), dtype=float) * snake_lengths[max_len_idx]
        if max_len_idx == 0:
            second_max = np.max(snake_lengths[1:])
        elif max_len_idx == game.num_players - 1:
            second_max = np.max(snake_lengths[:game.num_players - 1])
        else:
            second_max = np.maximum(np.max(snake_lengths[:max_len_idx]), np.max(snake_lengths[max_len_idx+1:]))
        other_length_max[max_len_idx] = second_max
        # interpolate and calculate advantage
        interpolated_length = 0.9 * other_length_max + 0.1 * other_length_avg
        delta_advantage = snake_lengths - interpolated_length
        delta_metric = self._score_delta_advantage(delta_advantage)
        delta_on_eat_metric = self._score_delta_advantage(delta_advantage + 1)
        food_multiplier = delta_on_eat_metric - delta_metric
        decayed_food = np.exp(-food_distance * 0.2)
        decayed_food[np.logical_not(food_reachable)] = 0
        food_score = 12 * (delta_metric + decayed_food * food_multiplier)
        # control score
        area_control_sum = np.sum(area_control)
        if area_control_sum < 0.001:
            area_control_normalized = np.zeros(shape=(game.num_players,), dtype=float)
        else:
            area_control_normalized = self.cfg.area_control_weight * ((area_control / area_control_sum) - 0.5)
        # hunger pressure score
        own_pressure = np.minimum(snake_healths - food_distance * 1.2 - 10, 0)
        logistic = np.power(0.85, -own_pressure)
        health_score = 2 * logistic - 1
        health_score[snake_healths > 75] = 1
        health_score *= self.cfg.health_weight  # Final health score
        # tail reachable score
        distance_score = - np.ones(shape=(game.num_players,), dtype=float)
        insufficient_space = area_control < snake_lengths
        sufficient_short = snake_lengths <= 6
        distance_score[insufficient_space] = -1
        distance_score[np.logical_not(tail_reachable)] = -1
        distance_score[sufficient_short] = 0
        distance_score *= 10
        # add final scores
        final_score = food_score + area_control_normalized + health_score + distance_score
        final_score[not_turn_filter] = 0
        # squash between -1 and 1
        normalized_result = np.tanh(0.05 * final_score)
        return normalized_result


class NetworkEvalFunc(EvalFunc):
    """
    Compute node evaluation based on Neural Network. Saves the action probs of the network as info
    """

    def __init__(self, cfg: NetworkEvalConfig):
        super().__init__(cfg)
        self.net: Optional[Network] = None
        if cfg.net_cfg is not None:
            self.net = get_network_from_config(cfg.net_cfg)
        self.cfg = cfg
        self.device = torch.device('cpu')
        self.temperatures = self.cfg.init_temperatures
        precision_type = self.cfg.precision if self.cfg.precision is not None else "32-true"
        self.fabric = Fabric(precision=precision_type)

    def _compute(self, node_list: list[Node]) -> None:
        if self.cfg.no_grad:
            with torch.no_grad():
                self._compute_helper(node_list)
        else:
            self._compute_helper(node_list)

    def _compute_helper(self, node_list: list[Node]) -> None:
        if self.cfg.temperature_input and self.temperatures is None:
            raise ValueError(f"Need temperature input for eval function")
        if not node_list:
            return
        if self.net is None:
            for node in node_list:
                uniform_actions = np.ones(shape=(node.game.num_players_at_turn(), node.game.num_actions,), dtype=float)
                filtered_actions = filter_illegal_and_normalize(uniform_actions, node.game)
                node.visits = 1
                node.value_sum *= 0
                node.info["net_action_probs"] = filtered_actions
            return
        # stack all encodings to single tensor, filter terminal nodes
        filtered_list = []  # list of non-terminal nodes
        encoding_list = []  # list of all encodings for all players
        inv_perm_list = []  # inverse permutations for nodes
        index_list = []  # end-indices of nodes (filtered list)
        temp_in_list = []  # sbr-temperature input
        # forward pass for every node
        for node in node_list:
            obs_temp_input = None
            if self.cfg.temperature_input and self.cfg.obs_temperature_input and self.cfg.single_temperature:
                obs_temp_input = [self.temperatures[0]]
            elif self.cfg.temperature_input and self.cfg.obs_temperature_input and not self.cfg.single_temperature:
                obs_temp_input = self.temperatures
            if self.cfg.random_symmetry:
                enc, perm, inv_perm = node.game.get_obs(None, obs_temp_input, self.cfg.single_temperature)
            else:
                enc, perm, inv_perm = node.game.get_obs(0, obs_temp_input, self.cfg.single_temperature)
            # encoding for every separate player
            for idx, player in enumerate(node.game.players_at_turn()):
                encoding_list.append(enc[idx])
                if self.net.cfg.film_temperature_input:
                    if self.cfg.single_temperature:
                        temp_in_list.append(torch.tensor([self.temperatures[0]], dtype=torch.float32))
                    else:
                        # multiple film inputs: Temperatures of all enemies, but dead players have temp of zero
                        cur_film_in = torch.zeros(size=(node.game.num_players - 1,), dtype=torch.float32)
                        counter = 0
                        for enemy in range(node.game.num_players):
                            if enemy != player:
                                if node.game.is_player_at_turn(enemy):
                                    cur_film_in[counter] = self.temperatures[enemy]
                                counter += 1
                        temp_in_list.append(cur_film_in)
            filtered_list.append(node)
            inv_perm_list.append(inv_perm)
            index_list.append(len(encoding_list))
        # forward pass for all encodings, but do not exceed max batch size
        with self.fabric.autocast():
            if len(encoding_list) <= self.cfg.max_batch_size:
                enc_tensor = torch.stack(encoding_list, dim=0).to(self.device)
                temp_tensor = None
                if self.cfg.temperature_input and self.net.cfg.film_temperature_input:
                    temp_tensor = torch.stack(temp_in_list, dim=0).to(self.device)
                out_tensor_with_grad = self.net(enc_tensor, temp_tensor)
                out_tensor = out_tensor_with_grad.cpu()
                out_tensor = out_tensor.detach()
            else:
                start_idx = 0
                out_tensor_list = []
                end_idx_list = list(range(self.cfg.max_batch_size, len(encoding_list), self.cfg.max_batch_size))
                if end_idx_list[-1] < len(encoding_list):
                    end_idx_list.append(len(encoding_list))
                for end_idx in end_idx_list:
                    enc_tensor = torch.stack(encoding_list[start_idx:end_idx], dim=0).to(self.device)
                    temp_tensor = None
                    if self.cfg.temperature_input and self.net.cfg.film_temperature_input:
                        temp_tensor = torch.stack(temp_in_list[start_idx:end_idx], dim=0).to(self.device)
                    out_tensor_part_with_grad = self.net(enc_tensor, temp_tensor)
                    out_tensor_part = out_tensor_part_with_grad.cpu()
                    out_tensor_part = out_tensor_part.detach()
                    out_tensor_list.append(out_tensor_part)
                    start_idx = end_idx
                out_tensor = torch.cat(out_tensor_list, dim=0)
        out_tensor = out_tensor.float()
        # retrieve values/actions
        values = self.net.retrieve_value(out_tensor).numpy()
        actions = None
        if self.net.cfg.predict_policy:
            raw_actions = self.net.retrieve_policy(out_tensor)
            # apply softmax to raw log-actions
            actions = torch.nn.functional.softmax(raw_actions, dim=-1).numpy()
            # sanity checks
            if np.any(np.isnan(actions)) or np.any(np.isinf(actions)) \
                    or np.any(np.abs(1 - np.sum(actions, axis=-1)) > 1e3):
                raise Exception(f"Invalid actions returned by network: {actions}")
        # update node stats
        start_idx = 0
        for node, inv_perm, end_idx in zip(filtered_list, inv_perm_list, index_list):
            # extract network output
            net_values = values[start_idx:end_idx]
            # clip
            net_values = np.clip(net_values, self.cfg.min_clip_value, self.cfg.max_clip_value)
            # compute values for each player, zero for dead players
            node_values = np.zeros((node.game.num_players,), dtype=float)
            for idx, player in enumerate(node.game.players_at_turn()):
                node_values[player] = net_values[idx]
            # apply permutation to actions and filter illegal
            if self.net.cfg.predict_policy:
                cur_action_probs = actions[start_idx:end_idx]
                perm_actions = apply_permutation(cur_action_probs, inv_perm)
                filtered_actions = filter_illegal_and_normalize(perm_actions, node.game)
                # add results to node stats. Shape is (num_players_at_turn, num_actions)
                node.info["net_action_probs"] = filtered_actions
            self.update_node(node, node_values)
            # update indices
            start_idx = end_idx


class DummyEvalFunc(EvalFunc):
    """
    GameState evaluation function, which does an area control evaluation. Area control is weighted by health.
    """

    def __init__(self, cfg: DummyEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute(self, node_list: list[Node]) -> None:
        if not node_list:
            return
        node_values = np.zeros((node_list[0].game.num_players,), dtype=float)
        for node in node_list:
            self.update_node(node, node_values)


class EnemyExploitationEvalFunc(EvalFunc):
    """
    GameState evaluation function, which does an area control evaluation. Area control is weighted by health.
    """

    def __init__(self, cfg: EnemyExploitationEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.temperatures = cfg.init_temperatures
        # player eval function
        player_eval_cfg = NetworkEvalConfig(
            zero_sum_norm=self.cfg.zero_sum_norm,
            net_cfg=cfg.net_cfg,
            init_temperatures=self.cfg.init_temperatures,
            temperature_input=True,
            single_temperature=False,
            obs_temperature_input=self.cfg.obs_temperature_input,
            max_batch_size=self.cfg.max_batch_size,
            random_symmetry=False,
            precision=self.cfg.precision,
        )
        self.player_eval_func = NetworkEvalFunc(player_eval_cfg)
        # network eval functions for every enemy
        self.enemy_net = get_network_from_file(Path(self.cfg.enemy_net_path))
        self.enemy_net = self.enemy_net.eval()
        enemy_eval_cfg = NetworkEvalConfig(
            zero_sum_norm=self.cfg.zero_sum_norm,
            net_cfg=self.enemy_net.cfg,
            init_temperatures=self.cfg.init_temperatures,
            temperature_input=True,
            single_temperature=True,
            obs_temperature_input=self.cfg.obs_temperature_input,
            max_batch_size=self.cfg.max_batch_size,
            random_symmetry=False,
            precision=self.cfg.precision,
        )
        self.enemy_eval_func = NetworkEvalFunc(enemy_eval_cfg)
        self.enemy_eval_func.net = self.enemy_net

    def _compute(self, node_list: list[Node]) -> None:
        if not node_list:
            return
        num_players = node_list[0].game.num_players
        # quick start mode
        if self.player_eval_func.net is None:
            for node in node_list:
                uniform_actions = np.ones(shape=(node.game.num_players_at_turn(), node.game.num_actions,), dtype=float)
                filtered_actions = filter_illegal_and_normalize(uniform_actions, node.game)
                node.visits = 1
                node.value_sum *= 0
                for player in range(num_players):
                    node.info[f'v{player}'] = node.value_sum
                    node.info[f'p{player}'] = filtered_actions
            return
        # compute enemy estimates
        for player in range(num_players):
            # update temperature of enemy eval function
            self.enemy_eval_func.temperatures = [self.temperatures[player]]
            self.enemy_eval_func(node_list)
            for node in node_list:
                # update stats of nodes, s.t. subsequent call does not override
                node.info[f'v{player}'] = node.value_sum
                node.info[f'p{player}'] = node.info['net_action_probs']
                node.visits = 0
                node.value_sum *= 0
        # compute own estimate (for every player)
        self.player_eval_func(node_list)


class OshiZumoEvalFunc(EvalFunc):
    """
    Simple evaluation function for the game of oshi-zumo. For more information, see page 44 of
    https://linkinghub.elsevier.com/retrieve/pii/S0004370216300285
    """

    def __init__(self, cfg: OshiZumoEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute(self, node_list: list[Node]) -> None:
        for node in node_list:
            if not isinstance(node.game, OshiZumoGame):
                raise ValueError(f"OshiZumo eval function only works for the game of oshi zumo")
            b = 0
            if node.game.p0_coins > node.game.p1_coins and node.game.zumo_pos >= node.game.cfg.board_size \
                    or node.game.p0_coins >= node.game.p1_coins and node.game.zumo_pos > node.game.cfg.board_size:
                b = 1
            elif node.game.p0_coins < node.game.p1_coins and node.game.zumo_pos <= node.game.cfg.board_size \
                    or node.game.p0_coins <= node.game.p1_coins and node.game.zumo_pos < node.game.cfg.board_size:
                b = -1
            divisor = node.game.cfg.min_bid if node.game.cfg.min_bid > 0 else 1
            term1 = (node.game.p0_coins - node.game.p1_coins) / divisor
            term2 = node.game.zumo_pos - node.game.cfg.board_size
            val0 = np.tanh(b / 2 + 1 / 3 * (term1 + term2))
            values = np.asarray([val0, -val0], dtype=float)
            self.update_node(node, values)


class RandomRolloutEvalFunc(EvalFunc):
    def __init__(self, cfg: RandomRolloutEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute(self, node_list: list[Node]) -> None:
        for node in node_list:
            value_sum = np.zeros(shape=(node_list[0].game.num_players,), dtype=float)
            for _ in range(self.cfg.num_rollouts):
                cpy = node.game.get_copy()
                while not cpy.is_terminal():
                    ja = random.choice(cpy.available_joint_actions())
                    rewards, done, info = cpy.step(ja)
                    value_sum += rewards
            values = value_sum / self.cfg.num_rollouts
            self.update_node(node, values)

class OvercookedPotentialEvalFunc(EvalFunc):
    """
    GameState evaluation function, which does an area control evaluation. Area control is weighted by health.
    """

    def __init__(self, cfg: OvercookedPotentialEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.gridworld = OvercookedGridworld.from_layout_name(cfg.overcooked_layout)
        self.action_manager = MediumLevelActionManager(
            mdp=self.gridworld,
            mlam_params=NO_COUNTERS_PARAMS
        )

    def _compute(self, node_list: list[Node]) -> None:
        for node in node_list:
            if not isinstance(node.game, OvercookedGame):
                raise Exception("Can only use Potential eval with Overcooked game")
            game: OvercookedGame = node.game
            potential_unscaled = self.gridworld.potential_function(
                state=game.env.state,
                mp=self.action_manager.motion_planner,
                gamma=node.discount
            )
            potential_scaled = potential_unscaled / game.cfg.horizon
            result = np.asarray([potential_scaled, potential_scaled], dtype=float)
            self.update_node(node, result)

def get_eval_func_from_cfg(cfg: EvalFuncConfig) -> EvalFunc:
    if cfg.eval_func_type == EvalFuncType.AREA_CONTROL_EVAL \
            or cfg.eval_func_type == EvalFuncType.AREA_CONTROL_EVAL.value:
        return AreaControlEvalFunc(cfg)
    elif cfg.eval_func_type == EvalFuncType.COPY_CAT_EVAL or cfg.eval_func_type == EvalFuncType.COPY_CAT_EVAL.value:
        return CopyCatEvalFunc(cfg)
    elif cfg.eval_func_type == EvalFuncType.NETWORK_EVAL or cfg.eval_func_type == EvalFuncType.NETWORK_EVAL.value:
        return NetworkEvalFunc(cfg)
    elif cfg.eval_func_type == EvalFuncType.DUMMY or cfg.eval_func_type == EvalFuncType.DUMMY.value:
        return DummyEvalFunc(cfg)
    elif cfg.eval_func_type == EvalFuncType.OSHI_ZUMO or cfg.eval_func_type == EvalFuncType.OSHI_ZUMO.value:
        return OshiZumoEvalFunc(cfg)
    elif cfg.eval_func_type == EvalFuncType.ENEMY_EXPLOIT or cfg.eval_func_type == EvalFuncType.ENEMY_EXPLOIT.value:
        return EnemyExploitationEvalFunc(cfg)
    elif cfg.eval_func_type == EvalFuncType.ROLLOUT or cfg.eval_func_type == EvalFuncType.ROLLOUT.value:
        return RandomRolloutEvalFunc(cfg)
    elif cfg.eval_func_type == EvalFuncType.POTENTIAL or cfg.eval_func_type == EvalFuncType.POTENTIAL.value:
        return OvercookedPotentialEvalFunc(cfg)
    else:
        raise ValueError(f"Unknown eval function config: {cfg}")


def eval_config_from_structured(cfg) -> EvalFuncConfig:
    if cfg.eval_func_type == EvalFuncType.AREA_CONTROL_EVAL \
            or cfg.eval_func_type == EvalFuncType.AREA_CONTROL_EVAL.value:
        return AreaControlEvalConfig(**cfg)
    elif cfg.eval_func_type == EvalFuncType.COPY_CAT_EVAL or cfg.eval_func_type == EvalFuncType.COPY_CAT_EVAL.value:
        return CopyCatEvalConfig(**cfg)
    elif cfg.eval_func_type == EvalFuncType.NETWORK_EVAL or cfg.eval_func_type == EvalFuncType.NETWORK_EVAL.value:
        net_cfg = network_config_from_structured(cfg.net_cfg)
        kwargs = dict(cfg)
        kwargs.pop("net_cfg")
        return NetworkEvalConfig(net_cfg=net_cfg, **kwargs)
    elif cfg.eval_func_type == EvalFuncType.DUMMY or cfg.eval_func_type == EvalFuncType.DUMMY.value:
        return DummyEvalConfig()
    elif cfg.eval_func_type == EvalFuncType.OSHI_ZUMO or cfg.eval_func_type == EvalFuncType.OSHI_ZUMO.value:
        return OshiZumoEvalConfig(**cfg)
    elif cfg.eval_func_type == EvalFuncType.ENEMY_EXPLOIT or cfg.eval_func_type == EvalFuncType.ENEMY_EXPLOIT.value:
        net_cfg = network_config_from_structured(cfg.net_cfg)
        kwargs = dict(cfg)
        kwargs.pop("net_cfg")
        return EnemyExploitationEvalConfig(net_cfg=net_cfg, **kwargs)
    elif cfg.eval_func_type == EvalFuncType.ROLLOUT or cfg.eval_func_type == EvalFuncType.ROLLOUT.value:
        return RandomRolloutEvalConfig(**cfg)
    elif cfg.eval_func_type == EvalFuncType.POTENTIAL or cfg.eval_func_type == EvalFuncType.POTENTIAL.value:
        return OvercookedPotentialEvalConfig(**cfg)
    else:
        raise ValueError(f"Unknown eval function config: {cfg}")
