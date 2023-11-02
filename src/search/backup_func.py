import itertools
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Any

import numpy as np
import torch.multiprocessing as mp

from src.cpp.lib import CPP_LIB
from src.equilibria.nash import calculate_nash_equilibrium
from src.equilibria.logit import compute_logit_equilibrium, SbrMode
from src.equilibria.quantal import compute_qne_equilibrium, compute_qse_equilibrium
from src.equilibria.responses import values_from_policies, best_response_from_q, smooth_best_response_from_q
from src.game.actions import filter_illegal_and_normalize, q_values_from_individual_actions
from src.search.config import MaxMinBackupConfig, BackupFuncConfig, StandardBackupConfig, NashBackupConfig, \
    BackupFuncType, MaxAvgBackupConfig, SBRBackupConfig, RNADBackupConfig, Exp3BackupConfig, \
    RegretMatchingBackupConfig, UncertaintyBackupConfig, EnemyExploitationBackupConfig, QNEBackupConfig, \
    QSEBackupConfig, SBRLEBackupConfig, ExploitOtherBackupConfig, NashVsSBRBackupConfig
from src.search.node import Node
from src.search.utils import compute_maxmin, compute_maxavg


class BackupFunc(ABC):
    def __init__(self, cfg: BackupFuncConfig):
        self.cfg = cfg

    def __call__(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            backup_values: Optional[dict[str, Any]] = None,
            options: Optional[dict[str, Any]] = None,
     ) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        # sanity checks
        if node.is_leaf():
            raise Exception("Cannot compute backup on a leaf node")
        values, maybe_action_probs, new_backup_values = self._compute_backup_values(
            node=node,
            child=child,
            values=values,
            options=options,
            backup_values=backup_values,
        )
        # update decoupled player-action visits and player-action values if child is given. key is (player, action).
        if child is not None:
            for idx, player in enumerate(node.game.players_at_turn()):
                node.player_action_value_sum[(player, child.last_actions[idx])] += values[player]
                node.player_action_visits[(player, child.last_actions[idx])] += 1
        # update node stats
        if node.is_fully_explored():
            # if a node is fully explored, then set its value to the exact outcome
            node.visits = 1
            node.value_sum = values
        else:
            node.visits += 1
            node.value_sum += values
        # In standard-mcts multiplayer, we do not want to update value_sums with terminal rewards.
        # The reward array of nodes already capture these rewards (do not punish dead players twice)
        for player in node.game.players_not_at_turn():
            node.value_sum[player] = 0
        # update action probabilities computed in backup
        if maybe_action_probs is not None:
            # sanity check
            if np.any(np.abs(np.sum(maybe_action_probs, axis=-1) - 1) > 0.001) or np.any(np.isnan(maybe_action_probs)) \
                    or np.any(np.isinf(maybe_action_probs)):
                raise Exception(f"Action probabilities do not yield a prob dist: {maybe_action_probs}")
            if "action_probs" not in node.info:
                node.info["action_probs"] = np.zeros((node.game.num_players_at_turn(), node.game.num_actions),
                                                     dtype=float)
            if "action_probs_count" not in node.info:
                node.info["action_probs_count"] = 0
            if node.is_fully_explored():
                node.info["action_probs"] = maybe_action_probs
                node.info["action_probs_count"] = 1
            else:
                node.info["action_probs"] += maybe_action_probs
                node.info["action_probs_count"] += 1
        return values, new_backup_values

    @abstractmethod
    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        """
        Computes the propagated backup value for every player.
        Returns
         - value array of shape (num_players, )
         - Optional array of action probabilities of shape (num_players_at_turn, num_actions)
        """
        raise NotImplementedError()


class StandardBackupFunc(BackupFunc):
    """
    Standard Backup of MCTS. Propagate the value of a leaf upwards to the root node
    """
    def __init__(self, cfg: StandardBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if node.is_fully_explored():
            # it does not make sense to consider fully explored subtrees since action probs and values would be skewed
            raise ValueError("Cannot use Standard Backup with optimization of fully explored subtrees")
        if values is None or child is None:
            raise ValueError("Cannot use standard backup without a value to propagate")
        if child.is_leaf():
            # first backup, values are already the backward estimate
            pass
        else:
            # propagate the values upwards, consider rewards and discount
            values = node.discount * (child.rewards + values)
        return values, None, None


class MaxMinBackupFunc(BackupFunc):
    """
    MaxMin Backup Function uses maximum of player outcome of minimum enemy responses as the backup value.
    Action probabilities are computed by stretched sigmoid weighting.
    """
    def __init__(self, cfg: MaxMinBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        # we do not use the values input, because we only propagate the maxmin value upwards
        values, actions = compute_maxmin(node, self.cfg.factor)
        return values, actions, None


class MaxAvgBackupFunc(BackupFunc):
    """
    MaxMin Backup Function uses maximum of player outcome of minimum enemy responses as the backup value.
    Action probabilities are computed by stretched sigmoid weighting.
    """
    def __init__(self, cfg: MaxAvgBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        # we do not use the values input, because we only propagate the maxmin value upwards
        values, actions = compute_maxavg(node, self.cfg.factor)
        return values, actions, None


class NashBackupFunc(BackupFunc):
    """
    Computes backup value and action by solving for a nash equilibrium.
    """
    def __init__(self, cfg: NashBackupConfig, error_counter: Optional[mp.Value] = None):
        super().__init__(cfg)
        self.cfg = cfg
        self.error_counter = error_counter

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        available_actions: list[list[int]] = []
        for player in node.game.players_at_turn():
            available_actions.append(node.game.available_actions(player))
        shape = (len(node.game.available_joint_actions()), node.game.num_players_at_turn())
        joint_action_value_arr = np.empty(shape=shape, dtype=float)
        joint_action_list = []
        joint_action_values = node.get_joint_backward_estimate()
        counter = 0
        for joint_action, joint_action_v in joint_action_values.items():
            joint_action_list.append(joint_action)
            for player_idx, player in enumerate(node.game.players_at_turn()):
                joint_action_value_arr[counter, player_idx] = joint_action_v[player]
            counter += 1
        values, policies = calculate_nash_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_value_arr,
            use_cpp=self.cfg.use_cpp,
        )
        # convert result to proper data format
        all_values = np.zeros(shape=(node.game.num_players,), dtype=float)
        action_probs = np.zeros(shape=(node.game.num_players_at_turn(), node.game.num_actions))
        for player_idx, player in enumerate(node.game.players_at_turn()):
            all_values[player] = values[player_idx]
            for action_idx, action in enumerate(node.game.available_actions(player)):
                action_probs[player_idx, action] = policies[player_idx][action_idx]
        return all_values, action_probs, None

class SBRBackupFunc(BackupFunc):
    """
    Computes backup action values and probabilities by solving for a Logit Equilibrium.
    """
    def __init__(self, cfg: SBRBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.temperatures = cfg.init_temperatures

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if self.temperatures is None:
            raise ValueError(f"Need temperature to compute sbr backup")
        # initialize solver inputs
        cur_temperatures = []
        for player in node.game.players_at_turn():
            cur_temperatures.append(self.temperatures[player])
        available_actions: list[list[int]] = []
        for player in node.game.players_at_turn():
            available_actions.append(node.game.available_actions(player))
        shape = (len(node.game.available_joint_actions()), node.game.num_players_at_turn())
        joint_action_value_arr = np.empty(shape=shape, dtype=float)
        joint_action_list = []
        joint_action_values = node.get_joint_backward_estimate()
        counter = 0
        for joint_action, joint_action_v in joint_action_values.items():
            joint_action_list.append(joint_action)
            for player_idx, player in enumerate(node.game.players_at_turn()):
                joint_action_value_arr[counter, player_idx] = joint_action_v[player]
            counter += 1
        if self.cfg.init_random:
            initial_policies = None
        else:
            if "net_action_probs" not in node.info:
                raise Exception("SBR backup with informed initialization needs network actions")
            initial_policies = []
            for player_idx, player in enumerate(node.game.players_at_turn()):
                cur_policy = np.zeros(shape=(len(node.game.available_actions(player)), ), dtype=float)
                for action_idx, action in enumerate(node.game.available_actions(player)):
                    # add epsilon for numerical stability
                    cur_policy[action_idx] = node.info["net_action_probs"][player_idx, action] + 1e-5
                cur_policy = cur_policy / np.sum(cur_policy)
                initial_policies.append(cur_policy)
        # solve for logit equilibrium
        values, policies = compute_logit_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_value_arr,
            num_iterations=self.cfg.num_iterations,
            epsilon=self.cfg.epsilon,
            temperatures=cur_temperatures,
            initial_policies=initial_policies,
            moving_average_factor=self.cfg.moving_average_factor,
            sbr_mode=self.cfg.sbr_mode,
            use_cpp=self.cfg.use_cpp,
        )
        # convert result to proper data format
        all_values = np.zeros(shape=(node.game.num_players,), dtype=float)
        action_probs = np.zeros(shape=(node.game.num_players_at_turn(), node.game.num_actions))
        for player_idx, player in enumerate(node.game.players_at_turn()):
            all_values[player] = values[player_idx]
            for action_idx, action in enumerate(node.game.available_actions(player)):
                action_probs[player_idx, action] = policies[player_idx][action_idx]
        return all_values, action_probs, None


class RNADBackupFunc(BackupFunc):
    """
    Computes backup action values and probabilities by solving Regularized Nash Dynamics:
    Paper: https://arxiv.org/abs/2206.15378
    """
    def __init__(self, cfg: RNADBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if "net_action_probs" not in node.info:
            raise ValueError("R-NAD Backup needs action probabilities of the network for regularization")
        # initialize solver inputs
        available_actions: list[list[int]] = []
        for player in node.game.players_at_turn():
            available_actions.append(node.game.available_actions(player))
        shape = (len(node.game.available_joint_actions()), node.game.num_players_at_turn())
        joint_action_value_arr = np.empty(shape=shape, dtype=float)
        joint_action_list: list[tuple[int, ...]] = []
        joint_action_values = node.get_joint_backward_estimate()
        counter = 0
        for joint_action, joint_action_v in joint_action_values.items():
            joint_action_list.append(joint_action)
            for player_idx, player in enumerate(node.game.players_at_turn()):
                joint_action_value_arr[counter, player_idx] = joint_action_v[player]
            counter += 1
        # policies
        reg_policy_list: list[np.ndarray] = []
        for player_idx, player in enumerate(node.game.players_at_turn()):
            cur_p = np.zeros(shape=(len(node.game.available_actions(player)),), dtype=float)
            for action_idx, action in enumerate(node.game.available_actions(player)):
                cur_p[action_idx] = node.info["net_action_probs"][player_idx][action]
            reg_policy_list.append(cur_p)
            # reg_policy_list.append(node.info["net_action_probs"][player_idx])
        # solve r-nad
        values, policies = CPP_LIB.compute_r_nad(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_value_arr,
            reg_policies=reg_policy_list,
            num_iteration=self.cfg.num_iterations,
            reg_factor=self.cfg.reg_factor,
            initial_policies=reg_policy_list,
        )
        # convert result to proper data format
        all_values = np.zeros(shape=(node.game.num_players,), dtype=float)
        action_probs = np.zeros(shape=(node.game.num_players_at_turn(), node.game.num_actions))
        for player_idx, player in enumerate(node.game.players_at_turn()):
            all_values[player] = values[player_idx]
            for action_idx, action in enumerate(node.game.available_actions(player)):
                action_probs[player_idx, action] = policies[player_idx][action_idx]
        return all_values, action_probs, None


class Exp3BackupFunc(BackupFunc):
    def __init__(self, cfg: Exp3BackupConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if node.is_fully_explored():
            # it does not make sense to consider fully explored subtrees since action probs and values would be skewed
            raise ValueError("Cannot use EXP3-Backup with optimization of fully explored subtrees")
        if "last_probs" not in node.info or "norm_value_sum" not in node.info or "last_actions" not in node.info:
            raise Exception("Need info to compute Exp3-Backup. Please use Exp3 selection function.")
        if values is None or child is None:
            raise ValueError("Cannot use exp3 backup without a value to propagate")
        if child.is_leaf():
            pass  # first backup, values are already the backward estimate
        elif self.cfg.avg_backup:
            values = child.backward_estimate()
        else:
            # propagate the values upwards, consider rewards and discount
            values = node.discount * (child.rewards + values)
        # Exp3 backup saves normed player-action sum
        for player_idx, player in enumerate(node.game.players_at_turn()):
            last_action = node.info["last_actions"][player_idx]
            last_prob = node.info["last_probs"][player_idx]
            node.info["norm_value_sum"][player, last_action] += values[player] / last_prob
        return values, None, None


class RegretMatchingBackupFunc(BackupFunc):
    def __init__(self, cfg: RegretMatchingBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if node.is_fully_explored():
            # it does not make sense to consider fully explored subtrees since action probs and values would be skewed
            raise ValueError("Cannot use Regret Matching with optimization of fully explored subtrees")
        if values is None or child is None:
            raise ValueError("Cannot use Regret Matching backup without a value to propagate")
        if "last_actions" not in node.info:
            raise Exception("Need last actions to compute Regret Matching Backup.")
        if "regret" not in node.info:
            raise Exception("Need regret to compute Regret Matching Backup.")
        # discount propagated value
        if child.is_leaf():
            # first backup, values are already the backward estimate
            pass
        elif self.cfg.avg_backup:
            values = child.backward_estimate()
        else:
            # propagate the values upwards, consider rewards and discount
            values = node.discount * (child.rewards + values)
        # compute regret for every action not chosen
        chosen_actions = node.info["last_actions"]
        regret_dict = node.info["regret"]
        for player_idx, player in enumerate(node.game.players_at_turn()):
            chosen_action_player = chosen_actions[player_idx]
            for action in node.game.available_actions(player):
                if action == chosen_action_player:
                    continue  # chosen action does not produce regret
                # update regret
                chosen_actions[player_idx] = action
                cur_child = node.children[tuple(chosen_actions)]
                if cur_child.visits > 0:
                    avg_val = cur_child.backward_estimate()[player]
                else:
                    avg_val = node.discount * cur_child.rewards[player]
                regret_dict[player, action] += avg_val - values[player]
                # reset chosen actions, important
                chosen_actions[player_idx] = chosen_action_player
        return values, None, None


class UncertaintyBackupFunc(BackupFunc):
    def __init__(self, cfg: UncertaintyBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _init_node(self, node: Node):
        # policy
        if self.cfg.informed:
            if "net_action_probs" not in node.info:
                raise ValueError("Cannot use informed initialization without network probs")
            probs = node.info["net_action_probs"]
        else:
            probs = np.ones(shape=(node.game.num_players_at_turn(), node.game.num_actions), dtype=float)
            probs = filter_illegal_and_normalize(probs, node.game)
        node.info["policy"] = probs
        # certainty
        if node.is_terminal():
            node.info["certainty"] = 1
        else:
            node.info["certainty"] = 0
        node.info["init_flag"] = True
        node.info["q"]: dict[tuple[int, int], float] = defaultdict(lambda: 0)  # key (player, action)

    @staticmethod
    def _backward_v(node: Node) -> np.ndarray:
        if "v" in node.info:
            v = node.info["v"]
        else:
            v = node.value_sum  # node is unvisited, no need to create new array
        values = node.discount * (node.rewards + v)
        return values

    @staticmethod
    def _certainty(node: Node) -> float:
        if node.is_terminal():
            return 1
        elif "certainty" in node.info:
            return node.info["certainty"]
        else:
            return 0

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if "init_flag" not in node.info:
            self._init_node(node)
        for player_idx, player in enumerate(node.game.players_at_turn()):
            chosen_action = child.last_actions[player_idx]
            available_actions = node.game.available_actions(player)
            # update q-values
            old_q = node.info["q"][player, chosen_action]
            new_q = (1 - self.cfg.lr) * old_q + self.cfg.lr * values[player]
            node.info["q"][player, chosen_action] = new_q
            # update policies
            q = np.asarray([node.info["q"][player, action] for action in available_actions], dtype=float)
            exp = np.exp(self.cfg.temperature * q)
            new_p = exp / np.sum(exp)
            for action_idx, action in enumerate(available_actions):
                node.info["policy"][player, action] = new_p[action_idx]
        # compute joint probs
        if self.cfg.use_children:
            # evaluation based on all children, computationally expensive but accurate
            arr_list = []
            action_list = []
            policy = node.info["policy"]
            for player_idx, player in enumerate(node.game.players_at_turn()):
                legal_action_probs = []
                for action in node.game.available_actions(player):
                    legal_action_probs.append(policy[player_idx, action])
                # normalization is not necessary here, because we normalized above
                arr_list.append(legal_action_probs)
                action_list.append(node.game.available_actions(player))
            prod_arr = np.asarray(list(itertools.product(*arr_list)))
            joint_probs = np.prod(prod_arr, axis=-1)
            joint_actions = list(itertools.product(*action_list))
            # update value and certainty
            c_child = 0
            values = np.zeros(shape=(node.game.num_players,), dtype=float)
            for ja_idx, ja in enumerate(joint_actions):
                child = node.children[ja]
                values += joint_probs[ja_idx] * self._backward_v(child)  # value already contains the confidence
                c_child += joint_probs[ja_idx] * self._certainty(child)
            c_own = 1 - (1 / np.sqrt(node.visits))
            certainty = (c_child + c_own) / 2
            node.info["v"] = values * certainty
            node.info["certainty"] = certainty
        else:
            values = child.backward_estimate()
            c_own = 1 - (1 / np.sqrt(node.visits))
            values *= c_own
            node.info["certainty"] = c_own
            if "v" in node.info:
                node.info["v"] = (1 - self.cfg.lr) * node.info["v"] + self.cfg.lr * values
            else:
                node.info["v"] = values
        return values, None, None


class EnemyExploitationBackupFunc(BackupFunc):
    """
    Backup Function that exploits the enemies bounded rationality. It computes a logit equilibrium based
    on all temperatures. This yields the enemies action probabilities, which are used to compute the players q-values.
    The resulting value and action are the smooth best response given these q-values.
    """
    def __init__(self, cfg: EnemyExploitationBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.temperatures = cfg.init_temperatures

    def _compute_enemy_policies(self, node: Node) -> np.ndarray:
        """
        WARNING: This function is largely untested
        """
        # initialize solver inputs: available actions
        cur_temperatures = []
        for player in node.game.players_at_turn():
            cur_temperatures.append(self.temperatures[player])
        available_actions: list[list[int]] = []
        for player in node.game.players_at_turn():
            available_actions.append(node.game.available_actions(player))
        # value estimates for every player based on the children
        num_p_turn = node.game.num_players_at_turn()
        shape = (num_p_turn, len(node.game.available_joint_actions()), num_p_turn)
        joint_action_value_arrays = np.empty(shape=shape, dtype=float)
        joint_action_list = []
        ja_counter = 0
        for ja, child in node.children.items():
            joint_action_list.append(ja)
            for enemy_idx, enemy in enumerate(node.game.players_at_turn()):
                enemy_values = child.info[f"v{enemy}"]
                for p_idx, p in enumerate(node.game.players_at_turn()):
                    joint_action_value_arrays[enemy_idx, ja_counter, p_idx] = enemy_values[p]
        # initial policies
        if self.cfg.init_random:
            initial_policies = [None for _ in range(num_p_turn)]
        else:
            initial_policies = []
            for enemy_idx, enemy in enumerate(node.game.players_at_turn()):
                cur_initial_policies = []
                for player_idx, player in enumerate(node.game.players_at_turn()):
                    cur_policy = np.zeros(shape=(len(node.game.available_actions(player)),), dtype=float)
                    for action_idx, action in enumerate(node.game.available_actions(player)):
                        # add epsilon for numerical stability
                        cur_policy[action_idx] = node.info[f"p{enemy}"][player_idx, action] + 1e-5
                    cur_policy = cur_policy / np.sum(cur_policy)
                    cur_initial_policies.append(cur_policy)
                initial_policies.append(cur_initial_policies)
        # solve enemy equilibria
        enemy_policies = np.zeros(shape=(num_p_turn, num_p_turn, node.game.num_actions), dtype=float)
        for enemy_idx, enemy in enumerate(node.game.players_at_turn()):
            _, cur_policies = compute_logit_equilibrium(
                available_actions=available_actions,
                joint_action_list=joint_action_list,
                joint_action_value_arr=joint_action_value_arrays[enemy_idx],
                num_iterations=self.cfg.num_iterations,
                epsilon=self.cfg.epsilon,
                temperatures=cur_temperatures,
                initial_policies=initial_policies[enemy_idx],
                moving_average_factor=self.cfg.moving_average_factor,
                sbr_mode=self.cfg.sbr_mode,
                use_cpp=self.cfg.use_cpp,
            )
            for player_idx, player in node.game.players_at_turn():
                for action_idx, action in node.game.available_actions(player):
                    enemy_policies[enemy_idx, player_idx, action] = cur_policies[player_idx][action_idx]
        return enemy_policies

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if self.temperatures is None:
            raise ValueError(f"Need temperature to compute sbr backup")
        if len(self.temperatures) != node.game.num_players:
            raise ValueError(f"Need temperature for every player")
        # enemy policies given their bounded rationality. array shape (num_p, num_p, num_a)
        if self.cfg.recompute_policy:
            enemy_policies = self._compute_enemy_policies(node)
        else:
            policy_list = [node.info[f"p{enemy}"] for enemy in node.game.players_at_turn()]
            enemy_policies = np.asarray(policy_list, dtype=float)
        # compute q-values
        num_p = node.game.num_players_at_turn()
        q_values = np.zeros(shape=(num_p, node.game.num_actions), dtype=float)
        for ja, child in node.children.items():
            # compute enemy joint action probabilities
            ja_probs = np.ones(shape=(node.game.num_players,), dtype=float)
            for player_idx, player in enumerate(node.game.players_at_turn()):
                for enemy_idx, enemy in enumerate(node.game.players_at_turn()):
                    if enemy == player:
                        continue
                    ja_probs[player] *= enemy_policies[enemy_idx, enemy_idx, ja[enemy_idx]]
            # add value scaled by joint action prob to q-values
            scaled_values = ja_probs * child.backward_estimate()
            for player_idx, player in enumerate(node.game.players_at_turn()):
                q_values[player_idx, ja[player_idx]] += scaled_values[player]
        # softmax over q-values using the players temperature
        if self.cfg.exploit_temperature == math.inf:
            best_actions = np.argmax(q_values, axis=-1)
            probs = np.zeros(shape=(num_p, node.game.num_actions), dtype=float)
            for player_idx, player in enumerate(node.game.players_at_turn()):
                probs[player_idx, best_actions[player_idx]] = 1
        else:
            exp_term = np.exp(self.cfg.exploit_temperature * q_values)
            probs = exp_term / exp_term.sum(axis=-1)[:, np.newaxis]
        filtered_probs = filter_illegal_and_normalize(probs, node.game)
        values_at_turn = (filtered_probs * q_values).sum(axis=-1)
        # expand values to all players, not just at turn
        values = np.zeros(shape=(node.game.num_players,), dtype=float)
        for player_idx, player in enumerate(node.game.players_at_turn()):
            values[player] = values_at_turn[player_idx]
        # reset old values
        if not self.cfg.average_eval:
            node.value_sum *= 0
            node.visits = 0
        return values, filtered_probs, {}


class QNEBackupFunc(BackupFunc):
    """
    Computes backup action values and probabilities by solving for a Quantal Nash Equilibrium
    """
    def __init__(self, cfg: QNEBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.temperature = cfg.init_temperature

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if self.temperature is None:
            raise ValueError(f"Need temperature to compute qne backup")
        # initialize solver inputs
        available_actions: list[list[int]] = []
        for player in node.game.players_at_turn():
            available_actions.append(node.game.available_actions(player))
        shape = (len(node.game.available_joint_actions()), node.game.num_players_at_turn())
        joint_action_value_arr = np.empty(shape=shape, dtype=float)
        joint_action_list = []
        joint_action_values = node.get_joint_backward_estimate()
        counter = 0
        for joint_action, joint_action_v in joint_action_values.items():
            joint_action_list.append(joint_action)
            for player_idx, player in enumerate(node.game.players_at_turn()):
                joint_action_value_arr[counter, player_idx] = joint_action_v[player]
            counter += 1
        # solve for quantal nash equilibrium
        values, policies = compute_qne_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_value_arr,
            leader=self.cfg.leader,
            num_iterations=self.cfg.num_iterations,
            temperature=self.temperature,
        )
        # convert result to proper data format
        all_values = np.zeros(shape=(node.game.num_players,), dtype=float)
        action_probs = np.zeros(shape=(node.game.num_players_at_turn(), node.game.num_actions))
        for player_idx, player in enumerate(node.game.players_at_turn()):
            all_values[player] = values[player_idx]
            for action_idx, action in enumerate(node.game.available_actions(player)):
                action_probs[player_idx, action] = policies[player_idx][action_idx]
        return all_values, action_probs, None


class QSEBackupFunc(BackupFunc):
    """
    Computes backup action values and probabilities by solving for a Quantal Stackelberg Equilibrium
    """
    def __init__(self, cfg: QSEBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.temperature = cfg.init_temperature

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if self.temperature is None:
            raise ValueError(f"Need temperature to compute qse backup")
        # initialize solver inputs
        available_actions: list[list[int]] = []
        for player in node.game.players_at_turn():
            available_actions.append(node.game.available_actions(player))
        shape = (len(node.game.available_joint_actions()), node.game.num_players_at_turn())
        joint_action_value_arr = np.empty(shape=shape, dtype=float)
        joint_action_list = []
        joint_action_values = node.get_joint_backward_estimate()
        counter = 0
        for joint_action, joint_action_v in joint_action_values.items():
            joint_action_list.append(joint_action)
            for player_idx, player in enumerate(node.game.players_at_turn()):
                joint_action_value_arr[counter, player_idx] = joint_action_v[player]
            counter += 1
        # solve for quantal stackelberg equilibrium
        values, policies = compute_qse_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_value_arr,
            leader=self.cfg.leader,
            num_iterations=self.cfg.num_iterations,
            temperature=self.temperature,
            grid_size=self.cfg.grid_size,
        )
        # convert result to proper data format
        all_values = np.zeros(shape=(node.game.num_players,), dtype=float)
        action_probs = np.zeros(shape=(node.game.num_players_at_turn(), node.game.num_actions))
        for player_idx, player in enumerate(node.game.players_at_turn()):
            all_values[player] = values[player_idx]
            for action_idx, action in enumerate(node.game.available_actions(player)):
                action_probs[player_idx, action] = policies[player_idx][action_idx]
        return all_values, action_probs, None


class SBRLEBackupFunc(BackupFunc):
    """
    Computes backup action values and probabilities by solving for a (Smooth) Best Response Logit Equilibrium
    """
    def __init__(self, cfg: SBRLEBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.temperature = cfg.init_temperature
        if cfg.leader != 0:
            raise NotImplementedError()

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if self.temperature is None:
            raise ValueError(f"Need temperature to compute sbrle backup")
        # initialize solver inputs
        available_actions: list[list[int]] = []
        for player in node.game.players_at_turn():
            available_actions.append(node.game.available_actions(player))
        shape = (len(node.game.available_joint_actions()), node.game.num_players_at_turn())
        joint_action_value_arr = np.empty(shape=shape, dtype=float)
        joint_action_list = []
        joint_action_values = node.get_joint_backward_estimate()
        counter = 0
        for joint_action, joint_action_v in joint_action_values.items():
            joint_action_list.append(joint_action)
            for player_idx, player in enumerate(node.game.players_at_turn()):
                joint_action_value_arr[counter, player_idx] = joint_action_v[player]
            counter += 1
        # first solve logit equilibrium
        le_values, le_policies = compute_logit_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_value_arr,
            num_iterations=1000,
            epsilon=0,
            temperatures=[self.temperature for _ in range(node.game.num_players)],
            initial_policies=None,
            sbr_mode=SbrMode.CUMULATIVE_SUM,
            use_cpp=True,
        )
        # then compute q values for leader
        other_policies = list(le_policies[1:])
        q_vals = q_values_from_individual_actions(
            action_probs=other_policies,
            game=node.game,
            player=0,
            joint_action_values=joint_action_value_arr,
        )
        # response
        if self.cfg.response_temperature == math.inf:
            sbrle_pol = best_response_from_q(q_vals)
        else:
            sbrle_pol = smooth_best_response_from_q(q_vals, self.cfg.response_temperature)
        if np.any(np.isnan(sbrle_pol)):
            raise Exception(f"Nan vals in policy, this should never happen")
        policies = [sbrle_pol] + other_policies
        values = values_from_policies(
            individual_policies=policies,
            ja_action_values=joint_action_value_arr,
        )
        # convert result to proper data format
        all_values = np.zeros(shape=(node.game.num_players,), dtype=float)
        action_probs = np.zeros(shape=(node.game.num_players_at_turn(), node.game.num_actions))
        for player_idx, player in enumerate(node.game.players_at_turn()):
            all_values[player] = values[player_idx]
            for action_idx, action in enumerate(node.game.available_actions(player)):
                action_probs[player_idx, action] = policies[player_idx][action_idx]
        return all_values, action_probs, None


class ExploitOtherBackupFunc(BackupFunc):
    """
    Computes backup action values and probabilities by solving for a (Smooth) Best Response Logit Equilibrium
    """
    def __init__(self, cfg: ExploitOtherBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.backup_func = get_backup_func_from_cfg(self.cfg.backup_cfg)
        if self.cfg.leader != 0:
            raise NotImplementedError()

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if node.game.num_players_at_turn() != 2:
            raise NotImplementedError()
        # first compute policy to exploit
        self.backup_func(
            node=node,
            child=child,
            values=values,
            options=options,
            backup_values=backup_values,
        )
        # policies = node.info["action_probs"]
        policy_0 = node.info["action_probs"][0]
        if policy_0 is None:
            raise ValueError(f"Cannot exploit backup function which does not compute a policy")
        # joint actions
        counter = 0
        joint_action_list = []
        shape = (len(node.game.available_joint_actions()), node.game.num_players_at_turn())
        joint_action_value_arr = np.empty(shape=shape, dtype=float)
        joint_action_values = node.get_joint_backward_estimate()
        for joint_action, joint_action_v in joint_action_values.items():
            joint_action_list.append(joint_action)
            for player_idx, player in enumerate(node.game.players_at_turn()):
                joint_action_value_arr[counter, player_idx] = joint_action_v[player]
            counter += 1
        if self.cfg.worst_case:
            # joint enemy actions
            aa_enemy = [node.game.available_actions(p) for p in node.game.players_at_turn() if p != self.cfg.leader]
            joint_aa_enemy = list(itertools.product(*aa_enemy))
            player_q_dict = {aa: 0 for aa in joint_aa_enemy}
            for ja, ja_val in joint_action_values.items():
                enemy_ja = tuple(list(ja)[1:])
                player_a = ja[0]
                player_q_dict[enemy_ja] += ja_val[0] * policy_0[player_a]
            # best response
            q_list = list(player_q_dict.items())
            min_q = min(q_list, key=lambda x: x[1])
            enemy_ja = min_q[0]
            enemy_policies = [
                np.zeros(shape=(len(node.game.available_actions(p)),), dtype=float)
                for p in node.game.players_at_turn() if p != self.cfg.leader
            ]
            for a_idx, a in enumerate(enemy_ja):
                enemy_policies[a_idx][a] = 1
            # merge policies and compute final values
            policies = [policy_0] + enemy_policies
        else:
            # joint player actions
            enemy_q_dict = {a: 0 for a in node.game.available_actions(1)}
            for ja, ja_val in joint_action_values.items():
                enemy_q_dict[ja[1]] += ja_val[1] * policy_0[ja[0]]
            # best response
            q_list = list(enemy_q_dict.items())
            max_q = max(q_list, key=lambda x: x[1])
            enemy_a = max_q[0]
            enemy_policy = np.zeros(shape=(len(node.game.available_actions(1)),), dtype=float)
            enemy_policy[enemy_a] = 1
            # merge policies and compute final values
            policies = [policy_0, enemy_policy]
        values = values_from_policies(
            individual_policies=policies,
            ja_action_values=joint_action_value_arr,
        )
        # expand values and policies ot all players/actions
        all_values = np.zeros(shape=(node.game.num_players,), dtype=float)
        action_probs = np.zeros(shape=(node.game.num_players_at_turn(), node.game.num_actions))
        for player_idx, player in enumerate(node.game.players_at_turn()):
            all_values[player] = values[player_idx]
            for action_idx, action in enumerate(node.game.available_actions(player)):
                action_probs[player_idx, action] = policies[player_idx][action_idx]
        # reset node values and policies from exploit backup call
        node.value_sum *= 0
        node.visits = 0
        del node.info['action_probs']
        del node.info['action_probs_count']
        return all_values, action_probs, None

class NashVsSBRBackupFunc(BackupFunc):
    def __init__(self, cfg: NashVsSBRBackupConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.temperature = cfg.init_temperature
        if self.cfg.leader != 0:
            raise NotImplementedError()

    def _compute_backup_values(
            self,
            node: Node,
            child: Optional[Node],
            values: Optional[np.ndarray],
            options: Optional[dict[str, Any]] = None,
            backup_values: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, Any]]]:
        if len(node.game.players_at_turn()) != 2:
            raise NotImplementedError()
        # nash solver arguments
        available_actions: list[list[int]] = []
        for player in node.game.players_at_turn():
            available_actions.append(node.game.available_actions(player))
        shape = (len(node.game.available_joint_actions()), node.game.num_players_at_turn())
        joint_action_value_arr = np.empty(shape=shape, dtype=float)
        true_joint_action_value_arr = np.empty(shape=shape, dtype=float)
        joint_action_list = []
        # joint action values are either leaf evaluation or nash backup values stored in info dict
        counter = 0
        for ja in node.game.available_joint_actions():
            joint_action_list.append(ja)
            child = node.children[ja]
            if child.is_leaf():
                vals = child.backward_estimate()
            else:
                vals = child.info['nash_values']
            true_vals = child.backward_estimate()
            for player_idx, player in enumerate(node.game.players_at_turn()):
                joint_action_value_arr[counter, player_idx] = vals[player]
                true_joint_action_value_arr[counter, player_idx] = true_vals[player]
            counter += 1
        # solve nash equilibrium
        values, policies = calculate_nash_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_value_arr,
            use_cpp=True,
        )
        node.info['nash_values'] = values
        # compute q values of follower
        leader_policy = [policies[0]]
        q_vals = q_values_from_individual_actions(
            action_probs=leader_policy,
            game=node.game,
            player=1,
            joint_action_values=joint_action_value_arr,
        )
        # compute smooth best response of follower
        sbrle_pol = smooth_best_response_from_q(q_vals, self.temperature)
        policies = leader_policy + [sbrle_pol]
        values = values_from_policies(
            individual_policies=policies,
            ja_action_values=true_joint_action_value_arr,
        )
        # convert result to proper data format
        all_values = np.zeros(shape=(node.game.num_players,), dtype=float)
        action_probs = np.zeros(shape=(node.game.num_players_at_turn(), node.game.num_actions))
        for player_idx, player in enumerate(node.game.players_at_turn()):
            all_values[player] = values[player_idx]
            for action_idx, action in enumerate(node.game.available_actions(player)):
                action_probs[player_idx, action] = policies[player_idx][action_idx]
        return all_values, action_probs, None


def get_backup_func_from_cfg(cfg: BackupFuncConfig) -> BackupFunc:
    if cfg.backup_type == BackupFuncType.STANDARD_BACKUP \
            or cfg.backup_type == BackupFuncType.STANDARD_BACKUP.value:
        return StandardBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.MAXMIN_BACKUP or cfg.backup_type == BackupFuncType.MAXMIN_BACKUP.value:
        return MaxMinBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.MAXAVG_BACKUP or cfg.backup_type == BackupFuncType.MAXAVG_BACKUP.value:
        return MaxAvgBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.NASH_BACKUP or cfg.backup_type == BackupFuncType.NASH_BACKUP.value:
        return NashBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.SBR_BACKUP or cfg.backup_type == BackupFuncType.SBR_BACKUP.value:
        return SBRBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.RNAD_BACKUP or cfg.backup_type == BackupFuncType.RNAD_BACKUP.value:
        return RNADBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.EXP3 or cfg.backup_type == BackupFuncType.EXP3.value:
        return Exp3BackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.REGRET_MATCHING or cfg.backup_type == BackupFuncType.REGRET_MATCHING.value:
        return RegretMatchingBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.UNCERTAINTY or cfg.backup_type == BackupFuncType.UNCERTAINTY.value:
        return UncertaintyBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.ENEMY_EXPLOIT or cfg.backup_type == BackupFuncType.ENEMY_EXPLOIT.value:
        return EnemyExploitationBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.QNE or cfg.backup_type == BackupFuncType.QNE.value:
        return QNEBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.QSE or cfg.backup_type == BackupFuncType.QSE.value:
        return QSEBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.SBRLE or cfg.backup_type == BackupFuncType.SBRLE.value:
        return SBRLEBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.NASH_VS_SBR or cfg.backup_type == BackupFuncType.NASH_VS_SBR.value:
        return NashVsSBRBackupFunc(cfg)
    elif cfg.backup_type == BackupFuncType.EXPLOIT_OTHER or cfg.backup_type == BackupFuncType.EXPLOIT_OTHER.value:
        return ExploitOtherBackupFunc(cfg)
    else:
        raise ValueError(f"Unknown backup type: {cfg}")


def backup_config_from_structured(cfg) -> BackupFuncConfig:
    if cfg.backup_type == BackupFuncType.STANDARD_BACKUP \
            or cfg.backup_type == BackupFuncType.STANDARD_BACKUP.value:
        return StandardBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.MAXMIN_BACKUP or cfg.backup_type == BackupFuncType.MAXMIN_BACKUP.value:
        return MaxMinBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.MAXAVG_BACKUP or cfg.backup_type == BackupFuncType.MAXAVG_BACKUP.value:
        return MaxAvgBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.NASH_BACKUP or cfg.backup_type == BackupFuncType.NASH_BACKUP.value:
        return NashBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.SBR_BACKUP or cfg.backup_type == BackupFuncType.SBR_BACKUP.value:
        return SBRBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.RNAD_BACKUP or cfg.backup_type == BackupFuncType.RNAD_BACKUP.value:
        return RNADBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.EXP3 or cfg.backup_type == BackupFuncType.EXP3.value:
        return Exp3BackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.REGRET_MATCHING or cfg.backup_type == BackupFuncType.REGRET_MATCHING.value:
        return RegretMatchingBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.UNCERTAINTY or cfg.backup_type == BackupFuncType.UNCERTAINTY.value:
        return UncertaintyBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.ENEMY_EXPLOIT or cfg.backup_type == BackupFuncType.ENEMY_EXPLOIT.value:
        return EnemyExploitationBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.QNE or cfg.backup_type == BackupFuncType.QNE.value:
        return QNEBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.QSE or cfg.backup_type == BackupFuncType.QSE.value:
        return QSEBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.SBRLE or cfg.backup_type == BackupFuncType.SBRLE.value:
        return SBRLEBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.NASH_VS_SBR or cfg.backup_type == BackupFuncType.NASH_VS_SBR.value:
        return NashVsSBRBackupConfig(**cfg)
    elif cfg.backup_type == BackupFuncType.EXPLOIT_OTHER or cfg.backup_type == BackupFuncType.EXPLOIT_OTHER.value:
        kwargs = dict(cfg)
        kwargs['backup_cfg'] = backup_config_from_structured(cfg.backup_cfg)
        return ExploitOtherBackupConfig(**kwargs)
    else:
        raise ValueError(f"Unknown backup type: {cfg}")
