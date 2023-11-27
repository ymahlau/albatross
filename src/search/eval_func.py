import multiprocessing as mp
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import multiprocessing.sharedctypes as sc

from src.game.actions import apply_permutation, filter_illegal_and_normalize, softmax
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.values import apply_utility_norm, UtilityNorm
from src.network import Network
from src.network.initialization import get_network_from_config, get_network_from_file
from src.search.config import NetworkEvalConfig, CopyCatEvalConfig, AreaControlEvalConfig, EvalFuncConfig, \
    DummyEvalConfig, EnemyExploitationEvalConfig, RandomRolloutEvalConfig, InferenceServerEvalConfig, ResponseInferenceServerEvalConfig
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


def update_node(node: Node, values: np.ndarray, utility_norm: UtilityNorm):
    node.visits = 1
    # optionally squash values to be zero-sum
    norm_vals = apply_utility_norm(values, utility_norm)
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
            update_node(node, values, self.cfg.utility_norm)


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
            update_node(node, values, self.cfg.utility_norm)

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


def gather_encodings(
        node_list: list[Node],
        temperature_input: bool,
        single_temperature: bool,
        random_symmetry: bool,
        temperatures: Optional[list[float]],
        only_get_player: Optional[int] = None
) -> tuple[
    list[Node],
    list[np.ndarray],
    list[dict[int, int]],
    list[int],
]:
    # stack all encodings to single tensor, filter terminal nodes
    filtered_list = []  # list of non-terminal nodes
    encoding_list = []  # list of all encodings for all players
    inv_perm_list = []  # inverse permutations for nodes
    index_list = []  # end-indices of nodes (filtered list)
    # forward pass for every node
    for node in node_list:
        obs_temp_input = None
        if temperature_input:
            if temperatures is None:
                raise ValueError(f"Need temperatures in Evaluation function to gather encodings")
            obs_temp_input = [temperatures[0]] if single_temperature else temperatures
        symmetry = None if random_symmetry else 0
        enc, perm, inv_perm = node.game.get_obs(
            symmetry=symmetry,
            temperatures=obs_temp_input,
            single_temperature=single_temperature
        )
        # encoding for every separate player
        if only_get_player is None:
            for idx, player in enumerate(node.game.players_at_turn()):
                encoding_list.append(enc[idx])
        else:
            if only_get_player not in node.game.players_at_turn():
                continue
            player_idx = node.game.players_at_turn().index(only_get_player)
            encoding_list.append(enc[player_idx])
        filtered_list.append(node)
        inv_perm_list.append(inv_perm)
        index_list.append(len(encoding_list))
    return filtered_list, encoding_list, inv_perm_list, index_list


def update_node_stats(
        filtered_list: list[Node],
        inv_perm_list: list[dict[int, int]],
        index_list: list[int],
        values: np.ndarray,
        actions: Optional[np.ndarray],
        min_clip_value: float,
        max_clip_value: float,
        utility_norm: UtilityNorm,
) -> None:
    start_idx = 0
    for node, inv_perm, end_idx in zip(filtered_list, inv_perm_list, index_list):
        # extract network output
        net_values = values[start_idx:end_idx]
        # clip
        net_values = np.clip(net_values, min_clip_value, max_clip_value)
        # compute values for each player, zero for dead players
        node_values = np.zeros((node.game.num_players,), dtype=float)
        for idx, player in enumerate(node.game.players_at_turn()):
            node_values[player] = net_values[idx]
        # apply permutation to actions and filter illegal
        if actions is not None:
            cur_action_probs = actions[start_idx:end_idx]
            perm_actions = apply_permutation(cur_action_probs, inv_perm)
            filtered_actions = filter_illegal_and_normalize(perm_actions, node.game)
            # add results to node stats. Shape is (num_players_at_turn, num_actions)
            node.info["net_action_probs"] = filtered_actions
        update_node(node, node_values, utility_norm)
        # update indices
        start_idx = end_idx


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
        filtered_list, encoding_list_np, inv_perm_list, index_list = gather_encodings(
            node_list=node_list,
            temperature_input=self.cfg.temperature_input,
            single_temperature=self.cfg.single_temperature,
            random_symmetry=self.cfg.single_temperature,
            temperatures=self.temperatures,
        )
        encoding_list = [torch.tensor(x, dtype=torch.float32) for x in encoding_list_np]
        # forward pass for all encodings, but do not exceed max batch size
        if len(encoding_list) <= self.cfg.max_batch_size:
            enc_tensor = torch.stack(encoding_list, dim=0).to(self.device)
            out_tensor_with_grad = self.net(enc_tensor)
            out_tensor = out_tensor_with_grad.cpu().detach().float().numpy()
        else:
            start_idx = 0
            out_tensor_list = []
            end_idx_list = list(range(self.cfg.max_batch_size, len(encoding_list), self.cfg.max_batch_size))
            if end_idx_list[-1] < len(encoding_list):
                end_idx_list.append(len(encoding_list))
            for end_idx in end_idx_list:
                enc_tensor = torch.stack(encoding_list[start_idx:end_idx], dim=0).to(self.device)
                out_tensor_part_with_grad = self.net(enc_tensor)
                out_tensor_part = out_tensor_part_with_grad.cpu().detach().float().numpy()
                out_tensor_list.append(out_tensor_part)
                start_idx = end_idx
            out_tensor = np.concatenate(out_tensor_list, axis=0)
        # retrieve values/actions
        values = self.net.retrieve_value(out_tensor)
        actions = None
        if self.net.cfg.predict_policy:
            raw_actions = self.net.retrieve_policy(out_tensor)
            # apply softmax to raw log-actions
            actions = softmax(raw_actions, 1)
            # sanity checks
            if np.any(np.isnan(actions)) or np.any(np.isinf(actions)) \
                    or np.any(np.abs(1 - np.sum(actions, axis=-1)) > 1e3):
                raise Exception(f"Invalid actions returned by network: {actions}")
        # update node stats
        update_node_stats(
            filtered_list=filtered_list,
            inv_perm_list=inv_perm_list,
            index_list=index_list,
            values=values,
            actions=actions if self.net.cfg.predict_policy else None,
            min_clip_value=self.cfg.min_clip_value,
            max_clip_value=self.cfg.max_clip_value,
            utility_norm=self.cfg.utility_norm,
        )


class DummyEvalFunc(EvalFunc):

    def __init__(self, cfg: DummyEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute(self, node_list: list[Node]) -> None:
        if not node_list:
            return
        node_values = np.zeros((node_list[0].game.num_players,), dtype=float)
        for node in node_list:
            update_node(node, node_values, self.cfg.utility_norm)


class EnemyExploitationEvalFunc(EvalFunc):
    def __init__(self, cfg: EnemyExploitationEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.temperatures = cfg.init_temperatures
        # player eval function
        player_eval_cfg = NetworkEvalConfig(
            utility_norm=self.cfg.utility_norm,
            net_cfg=cfg.net_cfg,
            init_temperatures=self.cfg.init_temperatures,
            temperature_input=True,
            single_temperature=False,
            max_batch_size=self.cfg.max_batch_size,
            random_symmetry=False,
        )
        self.player_eval_func = NetworkEvalFunc(player_eval_cfg)
        # network eval functions for every enemy
        self.enemy_net = get_network_from_file(Path(self.cfg.enemy_net_path))
        self.enemy_net = self.enemy_net.eval()
        enemy_eval_cfg = NetworkEvalConfig(
            utility_norm=self.cfg.utility_norm,
            net_cfg=self.enemy_net.cfg,
            init_temperatures=self.cfg.init_temperatures,
            temperature_input=True,
            single_temperature=True,
            max_batch_size=self.cfg.max_batch_size,
            random_symmetry=False,
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
                uniform_actions = np.ones(
                    shape=(node.game.num_players_at_turn(), node.game.num_actions,), 
                    dtype=float
                )
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
            if self.temperatures is None:
                raise Exception("This should never happen")
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
            update_node(node, values, self.cfg.utility_norm)


class InferenceServerEvalFunc(EvalFunc):
    def __init__(self, cfg: InferenceServerEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.temperatures = self.cfg.init_temperatures
        self.input_arr_np: Optional[np.ndarray] = None
        self.output_arr_np: Optional[np.ndarray] = None
        self.input_rdy_arr_np: Optional[np.ndarray] = None
        self.output_rdy_arr_np: Optional[np.ndarray] = None
        self.start_idx: Optional[int] = None
        self.max_length: Optional[int] = None
        self.stop_flag: Optional[sc.Synchronized] = None

    def _compute(self, node_list: list[Node]) -> None:
        if not node_list:
            return
        if self.input_arr_np is None or self.output_arr_np is None or self.input_rdy_arr_np is None \
                or self.output_rdy_arr_np is None or self.start_idx is None or self.max_length is None \
                or self.stop_flag is None:
            raise Exception('Need Arrays and indices for sending data to inference server!')
        # get observations from game states
        filtered_list, encoding_list_np, inv_perm_list, index_list = gather_encodings(
            node_list=node_list,
            temperature_input=self.cfg.temperature_input,
            single_temperature=self.cfg.single_temperature,
            random_symmetry=self.cfg.random_symmetry,
            temperatures=self.temperatures,
        )
        stacked_obs = np.stack(encoding_list_np, axis=0)
        if stacked_obs.shape[0] > self.max_length:
            raise ValueError("Received more observation than inference server allows. Increase max size!")
        # send observations to inference server
        end_idx = self.start_idx+stacked_obs.shape[0]
        self.input_arr_np[self.start_idx:end_idx] = stacked_obs
        self.input_rdy_arr_np[self.start_idx:end_idx] = 1
        # wait for inference to process data
        while not self.stop_flag.value:
            # check if data ready
            if np.all(self.output_rdy_arr_np[self.start_idx:end_idx] == 1):
                break
            # wait again
            time.sleep(self.cfg.active_wait_time)
        if self.stop_flag.value:  # program terminated
            return
        # gather outputs and clear rdy arr
        outputs = self.output_arr_np[self.start_idx:end_idx]
        self.output_rdy_arr_np[self.start_idx:end_idx] = 0
        # retrieve values/actions
        values = Network.retrieve_value(outputs)
        actions = None
        if self.cfg.policy_prediction:
            raw_actions = Network.retrieve_policy(outputs)
            # apply softmax to raw log-actions
            actions = softmax(raw_actions, 1)
            # sanity checks
            if np.any(np.isnan(actions)) or np.any(np.isinf(actions)) \
                    or np.any(np.abs(1 - np.sum(actions, axis=-1)) > 1e3):
                raise Exception(f"Invalid actions returned by network: {actions}")
        # update node stats
        update_node_stats(
            filtered_list=filtered_list,
            inv_perm_list=inv_perm_list,
            index_list=index_list,
            values=values,
            actions=actions if self.cfg.policy_prediction else None,
            min_clip_value=self.cfg.min_clip_value,
            max_clip_value=self.cfg.max_clip_value,
            utility_norm=self.cfg.utility_norm,
        )

    def update_arrays_and_indices(
            self,
            input_arr_np: np.ndarray,
            output_arr_np: np.ndarray,
            input_rdy_arr_np: np.ndarray,
            output_rdy_arr_np: np.ndarray,
            start_idx: int,
            max_length: int,
            stop_flag: sc.Synchronized,
    ):
        self.input_arr_np = input_arr_np
        self.output_arr_np = output_arr_np
        self.input_rdy_arr_np = input_rdy_arr_np
        self.output_rdy_arr_np = output_rdy_arr_np
        self.start_idx = start_idx
        self.max_length = max_length
        self.stop_flag = stop_flag
        
        
class ResponseInferenceEvalFunc(EvalFunc):
    def __init__(self, cfg: ResponseInferenceServerEvalConfig):
        super().__init__(cfg)
        self.cfg = cfg
        if not self.cfg.policy_prediction:
            raise ValueError("Need network to predict policy in response training")
        self.temperatures = []
        self.input_arr_np: Optional[np.ndarray] = None
        self.input_arr_resp_np: Optional[np.ndarray] = None
        self.output_arr_np: Optional[np.ndarray] = None
        self.output_arr_resp_np: Optional[np.ndarray] = None
        self.input_rdy_arr_np: Optional[np.ndarray] = None
        self.input_rdy_arr_resp_np: Optional[np.ndarray] = None
        self.output_rdy_arr_np: Optional[np.ndarray] = None
        self.output_rdy_arr_resp_np: Optional[np.ndarray] = None
        self.start_idx: Optional[int] = None
        self.max_length: Optional[int] = None
        self.stop_flag: Optional[sc.Synchronized] = None
        
    def _compute(self, node_list: list[Node]) -> None:
        if not node_list:
            return
        if self.input_arr_np is None or self.output_arr_np is None or self.input_rdy_arr_np is None \
                or self.output_rdy_arr_np is None or self.start_idx is None or self.max_length is None \
                or self.stop_flag is None or self.input_arr_resp_np is None \
                or self.output_arr_resp_np is None or self.input_rdy_arr_resp_np is None \
                or self.output_rdy_arr_resp_np is None:
            raise Exception('Need Arrays and indices for sending data to inference server!')
        if self.temperatures is None:
            raise Exception("Need temperature for response model evaluation")
        
        # get observations for proxy model
        node_list_proxy, encoding_list_proxy, inv_perm_list_proxy, index_list_proxy = [], [], [], []
        for player, temperature in enumerate(self.temperatures):
            filtered_list, encoding_list_np, inv_perm_list, index_list = gather_encodings(
                node_list=node_list,
                temperature_input=True,
                single_temperature=True,
                random_symmetry=self.cfg.random_symmetry,
                temperatures=[temperature],
                only_get_player=player,
            )
            node_list_proxy.append(filtered_list)
            encoding_list_proxy.append(encoding_list_np)
            inv_perm_list_proxy.append(inv_perm_list)
            index_list_proxy.append(index_list)
        # get encoding for response model
        filtered_list_resp, encoding_list_resp, inv_perm_list_resp, index_list_resp = gather_encodings(
            node_list=node_list,
            temperature_input=True,
            single_temperature=False,
            random_symmetry=self.cfg.random_symmetry,
            temperatures=self.temperatures,
        )
        # send to proxy inference server
        stacked_obs = np.asarray(encoding_list_proxy, dtype=np.float32)
        stacked_obs = stacked_obs.reshape(-1, *self.input_arr_np.shape[1:])
        if stacked_obs.shape[0] > self.max_length:
            raise ValueError("Received more observation than inference server allows. Increase max size!")
        end_idx_proxy = self.start_idx+stacked_obs.shape[0]
        self.input_arr_np[self.start_idx:end_idx_proxy] = stacked_obs
        self.input_rdy_arr_np[self.start_idx:end_idx_proxy] = 1
        # send to response inference server
        stacked_obs_resp = np.stack(encoding_list_resp, axis=0)
        if stacked_obs_resp.shape[0] > self.max_length:
            raise ValueError("Received more observation than inference server allows. Increase max size!")
        end_idx_resp = self.start_idx+stacked_obs_resp.shape[0]
        self.input_arr_resp_np[self.start_idx:end_idx_resp] = stacked_obs_resp
        self.input_rdy_arr_resp_np[self.start_idx:end_idx_resp] = 1
        # wait for both inference servers
        while not self.stop_flag.value:
            # check if data ready
            if np.all(self.output_rdy_arr_np[self.start_idx:end_idx_proxy] == 1) \
                    and np.all(self.output_rdy_arr_resp_np[self.start_idx:end_idx_resp] == 1):
                break
            # wait again
            time.sleep(self.cfg.active_wait_time)
        if self.stop_flag.value:  # program terminated
            return
        # gather outputs and clear rdy arr
        outputs_proxy = self.output_arr_np[self.start_idx:end_idx_proxy]
        self.output_rdy_arr_np[self.start_idx:end_idx_proxy] = 0
        outputs_resp = self.output_arr_resp_np[self.start_idx:end_idx_resp]
        self.output_rdy_arr_resp_np[self.start_idx:end_idx_resp] = 0
        # retrieve values/actions
        values_proxy = Network.retrieve_value(outputs_proxy)
        values_resp = Network.retrieve_value(outputs_resp)
        # apply softmax to raw log-actions
        actions_proxy = softmax(Network.retrieve_policy(outputs_proxy), 1)
        actions_resp = softmax(Network.retrieve_policy(outputs_resp), 1)
        # sanity checks
        if np.any(np.isnan(actions_proxy)) or np.any(np.isinf(actions_proxy)) \
                or np.any(np.abs(1 - np.sum(actions_proxy, axis=-1)) > 1e3):
            raise Exception(f"Invalid actions returned by proxy: {actions_proxy}")
        if np.any(np.isnan(actions_resp)) or np.any(np.isinf(actions_resp)) \
                or np.any(np.abs(1 - np.sum(actions_resp, axis=-1)) > 1e3):
            raise Exception(f"Invalid actions returned by response: {actions_resp}")
        # add proxy results
        start_idx = 0
        for player in range(len(self.temperatures)):
            cur_end_idx = start_idx + len(node_list_proxy[player])
            cur_values = values_proxy[start_idx:cur_end_idx]
            cur_values = np.clip(cur_values, self.cfg.min_clip_value, self.cfg.max_clip_value)
            cur_values = apply_utility_norm(cur_values, self.cfg.utility_norm)
            cur_actions = actions_proxy[start_idx:cur_end_idx]
            cur_inv_perm_list = inv_perm_list_proxy[player]
            for idx, node in enumerate(node_list_proxy[player]):
                node.info[f'v{player}'] = cur_values[idx]
                cur_node_actions = cur_actions[idx]
                perm_actions = apply_permutation(cur_node_actions[np.newaxis, :], cur_inv_perm_list[idx])[0]
                # filter illegal and normalize
                for action in node.game.illegal_actions(player):
                    perm_actions[action] = 0
                filtered_actions = perm_actions / perm_actions.sum()
                node.info[f'p{player}'] = filtered_actions
        # add response model results
        update_node_stats(
            filtered_list=filtered_list_resp,
            inv_perm_list=inv_perm_list_resp,
            index_list=index_list_resp,
            values=values_resp,
            actions=actions_resp,
            min_clip_value=self.cfg.min_clip_value,
            max_clip_value=self.cfg.max_clip_value,
            utility_norm=self.cfg.utility_norm,
        )
    
    def update_arrays_and_indices(
            self,
            input_arr_np: np.ndarray,
            input_arr_resp_np: np.ndarray,
            output_arr_np: np.ndarray,
            output_arr_resp_np: np.ndarray,
            input_rdy_arr_np: np.ndarray,
            input_rdy_arr_resp_np: np.ndarray,
            output_rdy_arr_np: np.ndarray,
            output_rdy_arr_resp_np: np.ndarray,
            start_idx: int,
            max_length: int,
            stop_flag: sc.Synchronized,
    ):
        self.input_arr_np = input_arr_np
        self.input_arr_resp_np = input_arr_resp_np
        self.output_arr_np = output_arr_np
        self.output_arr_resp_np = output_arr_resp_np
        self.input_rdy_arr_np = input_rdy_arr_np
        self.input_rdy_arr_resp_np = input_rdy_arr_resp_np
        self.output_rdy_arr_np = output_rdy_arr_np
        self.output_rdy_arr_resp_np = output_rdy_arr_resp_np
        self.start_idx = start_idx
        self.max_length = max_length
        self.stop_flag = stop_flag


def get_eval_func_from_cfg(cfg: EvalFuncConfig) -> EvalFunc:
    if isinstance(cfg, AreaControlEvalConfig):
        return AreaControlEvalFunc(cfg)
    elif isinstance(cfg, CopyCatEvalConfig):
        return CopyCatEvalFunc(cfg)
    elif isinstance(cfg, NetworkEvalConfig):
        return NetworkEvalFunc(cfg)
    elif isinstance(cfg, DummyEvalConfig):
        return DummyEvalFunc(cfg)
    elif isinstance(cfg, EnemyExploitationEvalConfig):
        return EnemyExploitationEvalFunc(cfg)
    elif isinstance(cfg, RandomRolloutEvalConfig):
        return RandomRolloutEvalFunc(cfg)
    elif isinstance(cfg, InferenceServerEvalConfig):
        return InferenceServerEvalFunc(cfg)
    elif isinstance(cfg, ResponseInferenceServerEvalConfig):
        return ResponseInferenceEvalFunc(cfg)
    else:
        raise ValueError(f"Unknown eval function config: {cfg}")
