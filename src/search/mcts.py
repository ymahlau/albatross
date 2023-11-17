import multiprocessing as mp
import time
from typing import Optional, Any

import numpy as np

from src.game.game import Game
from src.search import Search, SearchInfo
from src.search.backup_func import StandardBackupFunc, MaxMinBackupFunc, NashBackupFunc, LogitBackupFunc, \
    get_backup_func_from_cfg
from src.search.config import MCTSConfig
from src.search.core import update_full_exploration, expand_node_to_depth, backup_depth_dict
from src.search.extraction_func import StandardExtractFunc, SpecialExtractFunc
from src.search.node import Node
from src.search.sel_func import get_sel_func_from_cfg, DecoupledUCTSelectionFunc, AlphaZeroDecoupledSelectionFunc


class MCTS(Search):
    def __init__(self, cfg: MCTSConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.sel_func = get_sel_func_from_cfg(cfg.sel_func_cfg)
        self.backup_func = get_backup_func_from_cfg(cfg.backup_func_cfg)
        self._validate()
        self.options = None

    def _validate(self):
        if isinstance(self.backup_func, StandardBackupFunc):
            assert self.cfg.expansion_depth == 0
            assert isinstance(self.extract_func, StandardExtractFunc)
        if isinstance(self.backup_func, MaxMinBackupFunc) or isinstance(self.backup_func, NashBackupFunc) \
                or isinstance(self.backup_func, LogitBackupFunc):
            assert self.cfg.expansion_depth > 0
            assert isinstance(self.extract_func, SpecialExtractFunc)
        if isinstance(self.sel_func, DecoupledUCTSelectionFunc) \
                or isinstance(self.sel_func, AlphaZeroDecoupledSelectionFunc):
            assert not self.cfg.optimize_fully_explored
        if self.cfg.expansion_depth == 0:
            assert not self.cfg.optimize_fully_explored

    def _compute(
            self,
            game: Game,
            time_limit: Optional[float] = None,  # runtime limit in seconds
            iterations: Optional[int] = None,  # iteration limit
            save_probs = None,  # int value, to which current belief of best action is saved
            save_player_idx: Optional[int] = None,  # idx of player, whose action probs should be saved
            options: Optional[dict[str, Any]] = None,
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
        last_actions = game.get_last_actions()
        if self.root is not None and game == self.root.game:
            root = self.root
        # check if the game state used in last call is same as provided now (i.e. food spawning randomness)
        elif self.cfg.use_hot_start and self.root is not None and last_actions is not None \
                and self.root.children is not None \
                and game.get_last_actions() in self.root.children \
                and self.root.visits > 0:  # hot start is of no use if node was never visited
            maybe_root = self.root.children[last_actions]
            if maybe_root.game == game:
                root = maybe_root
                root.parent = None
                root.last_actions = None
                if root.rewards is None:
                    raise Exception("This should never happen")
                root.rewards *= 0  # if a player died in the last turn, we do not want to punish him twice
                cleanup_time_start = time.time()
                self.cleanup_root(exception_node=root)
                info.cleanup_time_ratio += time.time() - cleanup_time_start
        # if we did not find a previous game state create new one
        if root is None:
            cleanup_time_start = time.time()
            self.cleanup_root()
            info.cleanup_time_ratio += time.time() - cleanup_time_start
            root = Node(
                parent=None,
                last_actions=None,
                discount=self.cfg.discount,
                game=game.get_copy(),
                ignore_full_exploration=(not self.cfg.optimize_fully_explored)
            )
            # evaluate the new root
            eval_time_start = time.time()
            self.eval_func([root])
            info.eval_time_ratio += time.time() - eval_time_start
        self.root = root
        while True:
            # check time limit and iterations
            if iterations is not None and root.visits > iterations:
                break
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            if root.is_fully_explored():  # if whole tree is explored, there is no point computing anymore
                break
            # select leaf node to expand
            select_time_start = time.time()
            leaf = self._select_leaf(root)
            info.select_time_ratio += time.time() - select_time_start
            # if the leaf was never visited, mark it for evaluation later
            to_eval_list = []
            if leaf.visits == 0:
                if self.cfg.expansion_depth > 0:
                    raise Exception(f"Found unvisited node with expansion depth > 0, this should never happen")
                to_eval_list.append(leaf)
            # expand the leaf if:
            # - not terminal -> obvious
            # and either - already visited and expansion depth == 0 -> standard mcts
            #            - expansion depth > 0 -> need to expand for equilibrium backup
            depth_dict = None
            if not leaf.is_terminal() and (leaf.visits != 0 or self.cfg.expansion_depth > 0):
                # expand leaf node up to specific depth. If expansion depth 0, expand one step and select new leaf later
                expansion_start_time = time.time()
                depth_dict, node_list, leaf_list = expand_node_to_depth(
                    node=leaf,
                    max_depth=self.cfg.expansion_depth if self.cfg.expansion_depth > 0 else 1,
                    discount=self.cfg.discount,
                    ignore_full_exploration=(not self.cfg.optimize_fully_explored)
                )
                info.expansion_time_ratio += time.time() - expansion_start_time
                # check if the whole tree is explored (only with expansion depth > 0, see validation check)
                if self.cfg.optimize_fully_explored:
                    update_full_exploration(leaf_list)
                # add new nodes to eval list for equilibrium backup, but not standard mcts
                if self.cfg.expansion_depth > 0:
                    to_eval_list += node_list
            # in standard mcts, we need to select a new leaf below the old one
            if not leaf.is_terminal() and leaf.visits != 0 and self.cfg.expansion_depth == 0:
                select_time_start = time.time()
                leaf = self._select_leaf(leaf)
                info.select_time_ratio += time.time() - select_time_start
                to_eval_list.append(leaf)
            # evaluate all nodes that are marked for evaluation (node statistics are added in place in the function)
            eval_time_start = time.time()
            self.eval_func(to_eval_list)
            info.eval_time_ratio += time.time() - eval_time_start
            # backup values on path to root
            backup_time_start = time.time()
            self._backup_mcts(leaf, depth_dict, options)
            info.backup_time_ratio += time.time() - backup_time_start
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
        info.info['root visits'] = self.root.visits
        return values, action_probs, info

    def _select_leaf(self, node: Node) -> Node:
        # selects a leaf starting from the given node using a selection function
        cur_node = node
        while not cur_node.is_leaf():
            joint_actions = self.sel_func(cur_node)
            if cur_node.children is None:
                raise Exception("This should never happen")
            cur_node = cur_node.children[joint_actions]
        # sanity check
        if self.cfg.optimize_fully_explored and cur_node.is_terminal():
            raise Exception("Found terminal leaf in sel-func. This is likely an issue with the sel-func")
        return cur_node

    def _backup_mcts(
            self,
            leaf: Node,
            depth_dict: Optional[dict[int, list[Node]]],
            options: Optional[dict[str, Any]] = None,
    ):
        # first backup new nodes from expansion using the depth dict
        if self.cfg.expansion_depth > 0 and depth_dict is not None:
            backup_depth_dict(depth_dict, self.cfg.expansion_depth, self.backup_func, options)
            values = None
        else:
            values = leaf.backward_estimate()
        # then update values on path from leaf to root
        cur_child = leaf
        cur_node = leaf.parent
        while cur_node is not None:
            values, backup_values = self.backup_func(cur_node, cur_child, values, options)
            cur_node = cur_node.parent
            if cur_child is None:
                raise Exception("This should never happen")
            cur_child = cur_child.parent
