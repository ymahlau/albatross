import multiprocessing as mp
import time
from collections import defaultdict
from typing import Optional, Any

import numpy as np

from src.game.game import Game
from src.search import Search, SearchInfo, Node
from src.search.config import SMOOSConfig
from src.search.core import cleanup, expand_node_to_depth
from src.search.sel_func import RegretMatchingSelectionFunc


class SMOOS(Search):
    """
    Simultaneous Move Online Outcome Sampling - Algorithm.
    Original paper: Algorithms for computing strategies in two-player simultaneous move games
    https://linkinghub.elsevier.com/retrieve/pii/S0004370216300285 (page 35)
    """
    def __init__(self, cfg: SMOOSConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute(
            self,
            game: Game,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs: Optional[mp.Array] = None,
            save_player_idx: Optional[int] = None,
            options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, np.ndarray, SearchInfo]:
        if time_limit is None and iterations is None:
            raise ValueError("You need to provide either time limit or iterations to Search")
        if game.is_terminal():
            raise ValueError("Cannot call tree search on terminal node")
        self.options = options
        start_time = time.time()
        root = None
        info = SearchInfo()
        action_probs, values = None, None
        # check if we re-used the same game
        if self.root is not None and game == self.root.game:
            root = self.root
        # check if the game state used in last call is same as provided now (i.e. food spawning randomness)
        if self.cfg.use_hot_start and self.root is not None and game.get_last_actions() is not None \
                and self.root.children is not None \
                and game.get_last_actions() in self.root.children \
                and self.root.visits > 0:  # hot start is of no use if node was never visited
            maybe_root = self.root.children[game.get_last_actions()]
            if maybe_root.game == game:
                root = maybe_root
                root.parent = None
                root.last_action = None
                root.rewards *= 0  # if a player died in the last turn, we do not want to punish him twice
                cleanup_time_start = time.time()
                cleanup(self.root, exception_node=root)
                info.cleanup_time_ratio += time.time() - cleanup_time_start
        # if we did not find a previous game state create new one
        if root is None:
            cleanup_time_start = time.time()
            cleanup(self.root)
            info.cleanup_time_ratio += time.time() - cleanup_time_start
            root = Node(
                parent=None,
                last_actions=None,
                discount=self.cfg.discount,
                game=game.get_copy(),
                ignore_full_exploration=True,
            )
            # evaluate the new root
            eval_time_start = time.time()
            self.eval_func([root])
            info.eval_time_ratio += time.time() - eval_time_start
        self.root = root
        # Start SMOOS - Algorithm
        updating_player_idx = 0
        updating_player = root.game.players_at_turn()[updating_player_idx]
        while True:
            if iterations is not None and root.visits > iterations:
                break
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            if root.is_fully_explored():  # if whole tree is explored, there is no point computing anymore
                break
            # select leaf
            select_time_start = time.time()
            leaf = self._smoos_select(self.root, updating_player)
            info.select_time_ratio += time.time() - select_time_start
            if not leaf.is_terminal() and leaf.visits > 0:
                # expand
                expansion_start_time = time.time()
                expand_node_to_depth(
                    node=leaf,
                    max_depth=1,
                    discount=self.cfg.discount,
                    ignore_full_exploration=True,
                )
                info.expansion_time_ratio += time.time() - expansion_start_time
                # select new leaf
                select_time_start = time.time()
                leaf = self._smoos_select(leaf, updating_player)
                info.select_time_ratio += time.time() - select_time_start
            # evaluate
            if leaf.visits == 0:
                eval_time_start = time.time()
                self.eval_func([leaf])
                info.eval_time_ratio += time.time() - eval_time_start
            # backup
            backup_time_start = time.time()
            self._smoos_backup(leaf, updating_player)
            info.backup_time_ratio += time.time() - backup_time_start
            # new updating player
            updating_player_idx = (updating_player_idx + 1) % root.game.num_players_at_turn()
            updating_player = root.game.players_at_turn()[updating_player_idx]
            # compute current belief of best actions, values for players. Save intermediate results
            if save_probs is not None and save_player_idx is not None:
                # only calculate intermediate results if necessary
                extract_time_start = time.time()
                values, action_probs = self.extract_func(root)
                player_probs = action_probs[save_player_idx]
                save_probs[:] = player_probs
                info.extract_time_ratio += time.time() - extract_time_start
        # sanity check
        if root.children is None:
            raise Exception("Was not able to compute single mcts iteration")
        extract_time_start = time.time()
        if save_probs is not None and save_player_idx is not None:
            if action_probs is None or values is None:
                raise Exception("Was not able to compute a single MCTS-Iteration in the given time limit")
        else:
            # if we did not compute intermediate results, calculate final results now
            values, action_probs = self.extract_func(root)
        if save_probs is not None and save_player_idx is not None:
            player_probs = action_probs[save_player_idx]
            save_probs[:] = player_probs
        info.extract_time_ratio += time.time() - extract_time_start
        # infos
        info = self._build_info(info, time.time() - start_time)
        return values, action_probs, info

    def _smoos_select(self, start_node: Node, updating_player: int) -> Node:
        node = start_node
        while not node.is_leaf():
            # in multiplayer, if updating player dies then do not select further
            if updating_player not in node.game.players_at_turn():
                break
            # maybe add info entries to node
            if "regret" not in node.info:
                node.info["regret"]: dict[tuple[int, int], float] = defaultdict(lambda: 0)  # key (player, action)
            if "action_prob_sum" not in node.info:
                node.info["action_prob_sum"]: dict[tuple[int, int], float] = defaultdict(lambda: 0)
            if self.cfg.informed_exp and "net_action_probs" not in node.info:
                raise ValueError("Need net action probs for informed exploration mode")
            # select based on regret matching
            action_list = []
            node.info["probs"]: list[np.ndarrray] = []
            for player_idx, player in enumerate(node.game.players_at_turn()):
                # compute exploitation action probs
                available_actions = node.game.available_actions(player)
                num_actions = len(available_actions)
                probs = RegretMatchingSelectionFunc.compute_exploit_probs(node, player)
                node.info["probs"].append(probs)
                # add exploration to updating player
                if player == updating_player:
                    if self.cfg.uct_explore:
                        # exploration distribution based on visit count
                        na_list = [node.player_action_visits[player, a] for a in available_actions]
                        na = np.asarray(na_list, dtype=float)
                        explore = 1 / (na + 1)
                        if self.cfg.informed_exp:
                            net_probs = node.info["net_action_probs"]
                            for action_idx, action in enumerate(available_actions):
                                explore[action_idx] *= net_probs[player_idx, action]
                        explore /= np.sum(explore)
                    elif self.cfg.informed_exp:
                        # exploration prob is network policy
                        explore = np.empty_like(probs, dtype=float)
                        for action_idx, action in enumerate(available_actions):
                            net_probs = node.info["net_action_probs"]
                            explore[action_idx] = net_probs[player_idx, action]
                    else:
                        # uniform exploration distribution
                        explore = 1 / num_actions
                    # exploration factor
                    cur_exp_factor = self.cfg.exp_factor
                    if self.cfg.exp_decay:
                        cur_exp_factor /= np.log(node.visits + 2)
                    # convex combination of exploration / exploitation
                    exp_probs = (1 - cur_exp_factor) * probs + cur_exp_factor * explore
                    exp_probs /= np.sum(exp_probs)  # correct small numerical errors
                    node.info["exp_probs"]: np.ndarray = exp_probs  # explore probs only for updating player
                    probs = exp_probs
                # sample action
                action = np.random.choice(available_actions, p=probs)
                action_list.append(action)
            # update node
            node.info["last_actions"] = action_list
            node = node.children[tuple(action_list)]
        return node

    def _smoos_backup(
            self,
            leaf: Node,
            updating_player: int,
    ) -> None:
        # initial statistics for standard SM-OOS
        node = leaf.parent
        values = leaf.backward_estimate()
        while node is not None:
            # value and prob stats
            probs = node.info["probs"]
            exp_probs = node.info["exp_probs"]
            last_actions = node.info["last_actions"]
            updating_player_idx = node.game.players_at_turn().index(updating_player)
            updating_player_action = last_actions[updating_player_idx]
            updating_player_action_idx = node.game.available_actions(updating_player).index(updating_player_action)
            p = probs[updating_player_idx][updating_player_action_idx]
            p_prime = exp_probs[updating_player_action_idx]
            ratio = p / p_prime
            # updating player updates their regret
            for action_idx, action in enumerate(node.game.available_actions(updating_player)):
                old_regret_dec = self.cfg.regret_decay * node.info["regret"][updating_player, action]
                # maybe update chosen action depending on configuration
                if action == updating_player_action:
                    # chosen action does not produce regret
                    if self.cfg.relief_update:
                        relief = (1 - p) / p_prime * values
                        node.info["regret"][updating_player, action] = old_regret_dec + relief[updating_player]
                    else:
                        node.info["regret"][updating_player, action] = old_regret_dec
                    continue
                # not chosen actions
                if self.cfg.enhanced_regret:
                    last_actions[updating_player_idx] = action
                    other_child = node.children[tuple(last_actions)]
                    last_actions[updating_player_idx] = updating_player_action
                    if other_child.visits > 0:
                        other_value = other_child.backward_estimate()[updating_player]
                    else:
                        other_value = node.discount * other_child.rewards[updating_player]
                    regret = ratio * (other_value - values)
                    # regret = other_value - values
                else:
                    regret = - ratio * values
                node.info["regret"][updating_player, action] = old_regret_dec + regret[updating_player]
            # other players update their policy
            for player_idx, player in enumerate(node.game.players_at_turn()):
                if player == updating_player:
                    continue
                for action_idx, action in enumerate(node.game.available_actions(player)):
                    node.info["action_prob_sum"][player, action] += probs[player_idx][action_idx]
            # propagate value upwards with td-lambda, save values for this nodes update
            value_save = np.clip(ratio * values, -node.discount, node.discount)
            # update values and visit counts
            node.value_sum += value_save
            node.visits += 1
            # value_save = values
            values = node.discount * (node.rewards + value_save)
            values = (1 - self.cfg.lambda_val) * node.backward_estimate() + self.cfg.lambda_val * values
            for player, action in zip(node.game.players_at_turn(), last_actions):
                node.player_action_visits[player, action] += 1
            # parent is new node
            node = node.parent
        return
