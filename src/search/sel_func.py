import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from src.game.actions import add_dirichlet_noise, compute_joint_probs, sample_joint_action
from src.search.config import SampleSelectionConfig, AlphaZeroDecoupledSelectionConfig, DecoupledUCTSelectionConfig, \
    SelectionFuncConfig, SelectionFuncType, Exp3SelectionConfig, RegretMatchingSelectionConfig, \
    UncertaintySelectionConfig
from src.search.node import Node
from src.search.utils import filter_fully_explored


class SelectionFunc(ABC):
    def __init__(self, cfg: SelectionFuncConfig):
        self.cfg = cfg

    def __call__(self, node: Node) -> tuple[int, ...]:
        if node.is_leaf():
            raise ValueError("Cannot use selection function on leaf node")
        if node.is_fully_explored():
            raise ValueError("It is probably an error to call selection function in fully explored subtree")
        joint_actions = self._select_actions(node)
        if joint_actions not in node.children:
            raise Exception(f"Selected actions which are not in children: {joint_actions}")
        return joint_actions

    @abstractmethod
    def _select_actions(self, node: Node) -> tuple[int, ...]:
        # returns joint actions of all players
        raise NotImplementedError()


class DecoupledUCTSelectionFunc(SelectionFunc):
    """
    Decoupled UCT Action selection, according to http://mlanctot.info/files/papers/sm-tron-bnaic2013.pdf
    """
    def __init__(self, cfg: DecoupledUCTSelectionConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _select_actions(self, node: Node) -> tuple[int, ...]:
        action_list = []
        for idx, player in enumerate(node.game.players_at_turn()):
            max_uct = - math.inf
            best_action = -1
            non_explored_actions = []  # if some actions have never been explored, take a random one of them
            for action in node.game.available_actions(player):
                # exploitation term
                v = node.player_action_value_sum[(player, action)]
                n = node.player_action_visits[(player, action)]
                if n == 0:
                    non_explored_actions.append(action)
                    continue
                # exploration term
                unweighted = math.sqrt(math.log(node.visits) / n)
                exp_term = self.cfg.exp_bonus * unweighted
                uct = exp_term + v / n
                if uct > max_uct:
                    max_uct, best_action = uct, action
            if non_explored_actions:
                best_action = random.choice(non_explored_actions)
            action_list.append(best_action)
        return tuple(action_list)


class AlphaZeroDecoupledSelectionFunc(SelectionFunc):
    """
    Mixture of the selection function defined in AlphaZeroGo (https://www.nature.com/articles/nature24270)
    with decoupled selection for each player (http://mlanctot.info/files/papers/sm-tron-bnaic2013.pdf)
    Only in the root node, dirichlet noise is used.
    """
    def __init__(self, cfg: AlphaZeroDecoupledSelectionConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _select_actions(self, node: Node) -> tuple[int, ...]:
        if "net_action_probs" not in node.info:
            raise ValueError("Cannot use DecoupledUCT Selection function without network eval (need action probs)")
        action_list = []
        probs = node.info["net_action_probs"]
        if node.is_root():  # add noise to action probs in root
            probs = add_dirichlet_noise(probs, self.cfg.dirichlet_alpha, self.cfg.dirichlet_eps)
        for player_idx, player in enumerate(node.game.players_at_turn()):
            best_action, max_uct = -1, -math.inf
            non_explored_actions = []
            available_actions = node.game.available_actions(player)
            # if there is only one action, select this one
            if len(available_actions) == 1:
                action_list.append(available_actions[0])
                continue
            for action in available_actions:
                # exploitation term
                v = node.player_action_value_sum[(player, action)]
                n = node.player_action_visits[(player, action)]
                if n == 0:
                    # if an action was never selected before, do this first
                    non_explored_actions.append(action)
                    continue
                # exploration term
                ratio = math.sqrt(node.visits) / (1 + n)
                unweighted = ratio * probs[player_idx, action]
                exp_term = unweighted * self.cfg.exp_bonus
                uct = exp_term + v / n
                if uct > max_uct:
                    max_uct, best_action = uct, action
            # if there are non-explored actions, choose random one of them
            if non_explored_actions:
                # choose action with the highest probability given by network
                best_action, best_prob = -1, -math.inf
                for action in non_explored_actions:
                    if probs[player_idx, action] > best_prob:
                        best_action, best_prob = action, probs[player_idx, action]
            # sanity check
            if best_action == -1:
                raise Exception(f"AZ-UCT Selected -1 as best action for player {player}.\n"
                                f"Debug Info: {probs=},\n{non_explored_actions=}\n,{max_uct=}\n,"
                                f"{node.game.get_str_repr()=}\n, {node.player_action_value_sum=},\n"
                                f"{node.player_action_visits=}")
            action_list.append(best_action)
        return tuple(action_list)


class SampleSelectionFunc(SelectionFunc):
    """
    Computes best action according to backup-function. For exploration, we use dirichlet noise (in every node).
    Importantly, the action is sampled from the current belief, we do not choose the maximum value.
    """
    def __init__(self, cfg: SampleSelectionConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _select_actions(self, node: Node) -> tuple[int, ...]:
        if "action_probs" not in node.info or "action_probs_count" not in node.info:
            raise Exception("sample sel func currently only works with backup saving actions, todo if necessary")
        if node.visits == 0:
            raise Exception("sample selection function only works with expansion depth >= 1 currently")
        probs = node.info["action_probs"] / node.info["action_probs_count"]
        # add exploration bonus
        noisy_probs = add_dirichlet_noise(probs, self.cfg.dirichlet_alpha, self.cfg.dirichlet_eps)
        # sample joint action (not individual!)
        joint_action_list = node.game.available_joint_actions()
        joint_probs = compute_joint_probs(noisy_probs, node.game)
        # ignore all fully explored subtrees
        normalized_joint_probs = filter_fully_explored(joint_action_list, joint_probs, node)
        joint_action = sample_joint_action(joint_action_list, normalized_joint_probs, self.cfg.temperature)
        return joint_action


class Exp3SelectionFunc(SelectionFunc):
    def __init__(self, cfg: Exp3SelectionConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _select_actions(self, node: Node) -> tuple[int, ...]:
        if "norm_value_sum" not in node.info:
            node.info["norm_value_sum"]: dict[tuple[int, int], float] = defaultdict(lambda: 0)  # key (player, action)
        if "action_prob_sum" not in node.info:
            node.info["action_prob_sum"]: dict[tuple[int, int], float] = defaultdict(lambda: 0)  # key (player, action)
        action_list = []
        prob_list = []
        for player_idx, player in enumerate(node.game.players_at_turn()):
            available_actions = node.game.available_actions(player)
            num_actions = len(available_actions)
            # exp3 selection formula
            x_arr = np.asarray([node.info["norm_value_sum"][(player, a)] for a in available_actions], dtype=float)
            diff = x_arr[np.newaxis, :] - x_arr[:, np.newaxis]
            nu = self.cfg.random_prob / num_actions
            if self.cfg.altered:
                exps = np.exp(diff)
            else:
                exps = np.exp(nu * diff)
            exploit = (1 - self.cfg.random_prob) / np.sum(exps, axis=1)
            cur_probs = exploit + nu
            # sample
            cur_action = np.random.choice(available_actions, p=cur_probs)
            cur_action_idx = available_actions.index(cur_action)
            action_list.append(cur_action)
            prob_list.append(cur_probs[cur_action_idx])
            for action_idx, action in enumerate(available_actions):
                node.info["action_prob_sum"][player, action] += exploit[action_idx]
        node.info["last_probs"]: list[float] = prob_list
        node.info["last_actions"]: list[int] = action_list
        return tuple(action_list)


class RegretMatchingSelectionFunc(SelectionFunc):
    def __init__(self, cfg: RegretMatchingSelectionConfig):
        super().__init__(cfg)
        self.cfg = cfg

    @staticmethod
    def compute_exploit_probs(
            node: Node,
            player: int,
    ) -> np.ndarray:
        regret_dict = node.info["regret"]
        available_actions = node.game.available_actions(player)
        num_actions = len(available_actions)
        # compute positive regret sum
        regret_sum = 0
        for action in available_actions:
            regret_sum += max(0, regret_dict[player, action])
        if regret_sum > 0.001:
            # action prob proportional to positive regret
            prob_list = []
            for action in available_actions:
                prob_list.append(max(0, regret_dict[player, action]) / regret_sum)
            exploit = np.asarray(prob_list, dtype=float)
        else:
            # uniform distribution
            exploit = np.ones(shape=(num_actions,), dtype=float) / num_actions
        return exploit

    def _select_actions(self, node: Node) -> tuple[int, ...]:
        if "regret" not in node.info:
            node.info["regret"]: dict[tuple[int, int], float] = defaultdict(lambda: 0)  # key (player, action)
        if "action_prob_sum" not in node.info:
            node.info["action_prob_sum"]: dict[tuple[int, int], float] = defaultdict(lambda: 0)  # key (player, action)
        if self.cfg.informed_exp and "net_action_probs" not in node.info:
            raise ValueError(f"Need network action probs for informed selection")
        action_list = []
        for player_idx, player in enumerate(node.game.players_at_turn()):
            available_actions = node.game.available_actions(player)
            num_actions = len(available_actions)
            exploit = self.compute_exploit_probs(node, player)
            if self.cfg.informed_exp:
                # exploration prob is network policy
                explore = np.empty_like(exploit, dtype=float)
                net_probs = node.info["net_action_probs"]
                for action_idx, action in enumerate(available_actions):
                    explore[action_idx] = net_probs[player_idx, action]
            else:
                explore = np.ones(shape=(num_actions,), dtype=float) / num_actions
            probs = (1 - self.cfg.random_prob) * exploit + self.cfg.random_prob * explore
            probs /= np.sum(probs)
            # sample action
            cur_action = np.random.choice(available_actions, p=probs)
            action_list.append(cur_action)
            for action_idx, action in enumerate(available_actions):
                node.info["action_prob_sum"][player, action] += probs[action_idx] - (self.cfg.random_prob / num_actions)
        node.info["last_actions"]: list[int] = action_list
        return tuple(action_list)


class UncertaintySelectionFunc(SelectionFunc):
    def __init__(self, cfg: UncertaintySelectionConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _select_actions(self, node: Node) -> tuple[int, ...]:
        action_list = []
        for player_idx, player in enumerate(node.game.players_at_turn()):
            available_actions = node.game.available_actions(player)
            if "policy" in node.info:
                probs = [node.info["policy"][player_idx, action] for action in available_actions]
            elif self.cfg.informed:
                if "net_action_probs" not in node.info:
                    raise ValueError(f"Cannot use informed selection without network probs")
                probs = [node.info["net_action_probs"][player_idx, action] for action in available_actions]
            else:
                probs = [1 / len(available_actions) for _ in available_actions]
            probs = np.asarray(probs, dtype=float)
            probs /= np.sum(probs)  # numerical instabilities
            a = np.random.choice(available_actions, p=probs)
            action_list.append(a)
        return tuple(action_list)


def get_sel_func_from_cfg(cfg: SelectionFuncConfig) -> SelectionFunc:
    if cfg.sel_func_type == SelectionFuncType.DECOUPLED_UCT \
            or cfg.sel_func_type == SelectionFuncType.DECOUPLED_UCT.value:
        return DecoupledUCTSelectionFunc(cfg)
    elif cfg.sel_func_type == SelectionFuncType.AZ_DECOUPLED \
            or cfg.sel_func_type == SelectionFuncType.AZ_DECOUPLED.value:
        return AlphaZeroDecoupledSelectionFunc(cfg)
    elif cfg.sel_func_type == SelectionFuncType.SAMPLE or cfg.sel_func_type == SelectionFuncType.SAMPLE.value:
        return SampleSelectionFunc(cfg)
    elif cfg.sel_func_type == SelectionFuncType.EXP3 or cfg.sel_func_type == SelectionFuncType.EXP3.value:
        return Exp3SelectionFunc(cfg)
    elif cfg.sel_func_type == SelectionFuncType.REGRET_MATCHING \
            or cfg.sel_func_type == SelectionFuncType.REGRET_MATCHING.value:
        return RegretMatchingSelectionFunc(cfg)
    elif cfg.sel_func_type == SelectionFuncType.UNCERTAINTY \
            or cfg.sel_func_type == SelectionFuncType.UNCERTAINTY.value:
        return UncertaintySelectionFunc(cfg)
    else:
        raise ValueError(f"Unknown selection config type: {cfg}")


def selection_config_from_structured(cfg) -> SelectionFuncConfig:
    if cfg.sel_func_type == SelectionFuncType.DECOUPLED_UCT \
            or cfg.sel_func_type == SelectionFuncType.DECOUPLED_UCT.value:
        return DecoupledUCTSelectionConfig(**cfg)
    elif cfg.sel_func_type == SelectionFuncType.AZ_DECOUPLED \
            or cfg.sel_func_type == SelectionFuncType.AZ_DECOUPLED.value:
        return AlphaZeroDecoupledSelectionConfig(**cfg)
    elif cfg.sel_func_type == SelectionFuncType.SAMPLE or cfg.sel_func_type == SelectionFuncType.SAMPLE.value:
        return SampleSelectionConfig(**cfg)
    elif cfg.sel_func_type == SelectionFuncType.EXP3 or cfg.sel_func_type == SelectionFuncType.EXP3.value:
        return Exp3SelectionConfig(**cfg)
    elif cfg.sel_func_type == SelectionFuncType.REGRET_MATCHING \
            or cfg.sel_func_type == SelectionFuncType.REGRET_MATCHING.value:
        return RegretMatchingSelectionConfig(**cfg)
    elif cfg.sel_func_type == SelectionFuncType.UNCERTAINTY \
            or cfg.sel_func_type == SelectionFuncType.UNCERTAINTY.value:
        return UncertaintySelectionConfig(**cfg)
    else:
        raise ValueError(f"Unknown selection config type: {cfg}")
