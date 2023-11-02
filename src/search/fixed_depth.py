import multiprocessing as mp
import time
from typing import Optional, Any

import numpy as np

from src.game import Game
from src.search import Search, SearchInfo
from src.search.backup_func import StandardBackupFunc, get_backup_func_from_cfg
from src.search.config import FixedDepthConfig
from src.search.core import update_full_exploration, cleanup, expand_node_to_depth, backup_depth_dict
from src.search.extraction_func import StandardExtractFunc
from src.search.node import Node


class FixedDepthSearch(Search):
    def __init__(self, cfg: FixedDepthConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.backup_func = get_backup_func_from_cfg(cfg.backup_func_cfg)
        self._validate()

    def _validate(self):
        assert not isinstance(self.backup_func, StandardBackupFunc)
        assert not isinstance(self.extract_func, StandardExtractFunc)

    def _compute(
            self,
            game: Game,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs: Optional[mp.Array] = None,
            save_player_idx: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, np.ndarray, SearchInfo]:
        start_time = time.time()
        if iterations is None or iterations == 0:
            raise ValueError(f"Invalid iteration count: {iterations}")
        if game.is_terminal():
            raise ValueError("Cannot call tree search on terminal node")
        if iterations is None:
            raise ValueError("Fixed depth search does not work without depth")
        root = Node(
            parent=None,
            last_actions=None,
            discount=self.cfg.discount,
            game=game.get_copy(),
            ignore_full_exploration=False,
        )
        info = SearchInfo()
        # cleanup
        cleanup_start_time = time.time()
        cleanup(self.root)
        info.cleanup_time_ratio += time.time() - cleanup_start_time
        self.root = root
        # expand node up to depth
        expansion_start_time = time.time()
        depth_dict, node_list, leaf_list = expand_node_to_depth(root, iterations, self.cfg.discount, False)
        info.expansion_time_ratio += time.time() - expansion_start_time
        node_list.append(root)
        # evaluate. Maybe take average of heuristic and backup value to avoid being overconfident
        eval_start_time = time.time()
        if self.cfg.average_eval:
            self.eval_func(node_list)
        else:
            self.eval_func(leaf_list)
        info.eval_time_ratio += time.time() - eval_start_time
        # backup
        backup_start_time = time.time()
        backup_depth_dict(
            depth_dict,
            iterations if iterations > 0 else 1,
            self.backup_func,
            options
        )
        info.backup_time_ratio += time.time() - backup_start_time
        # fully explored subtrees. We update AFTER Backup because we do not want to mess up average
        update_full_exploration(leaf_list)
        # extract results
        extract_time_start = time.time()
        values, action_probs = self.extract_func(root)
        if save_probs is not None and save_player_idx is not None:
            player_probs = action_probs[save_player_idx]
            save_probs[:] = player_probs
        info.extract_time_ratio += time.time() - extract_time_start
        # infos
        info = self._build_info(info, time.time() - start_time)
        return values, action_probs, info
