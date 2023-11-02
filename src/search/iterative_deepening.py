import time
from typing import Optional, Any

import numpy as np
import torch.multiprocessing as mp

from src.game.game import Game
from src.search import SearchInfo, get_backup_func_from_cfg
from src.search.config import IterativeDeepeningConfig
from src.search.fixed_depth import FixedDepthSearch


class IterativeDeepening(FixedDepthSearch):
    """
    Performs iterative deepening search. The result is always saved and reused if possible.
    """
    def __init__(self, cfg: IterativeDeepeningConfig):
        super().__init__(cfg)
        self.backup_func = get_backup_func_from_cfg(cfg.backup_func_cfg)
        self.cfg = cfg

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
        if time_limit is None and iterations is None:
            raise ValueError("You need to provide either time limit or iterations to Search")
        if game.is_terminal():
            raise ValueError("Cannot call tree search on terminal node")
        max_depth = iterations
        info_sum = SearchInfo()
        if iterations is None:
            max_depth = 1000  # Any big integer
        values, action_probs = None, None
        final_search_depth = 0
        for depth in range(1, max_depth + 1):
            remaining_time = None
            if time_limit is not None:
                time_passed = time.time() - start_time
                remaining_time = time_limit - time_passed
                if remaining_time <= 0:
                    break
            values, action_probs, info = super()._compute(
                game=game,
                time_limit=remaining_time,
                iterations=depth,
                save_probs=save_probs,
                save_player_idx=save_player_idx,
                options=options,
            )
            final_search_depth = depth
            info_sum = self._add_info_sum(info_sum, info)
            if info.fully_explored:
                info_sum.fully_explored = True
                break
        if values is None or action_probs is None:
            raise Exception("Was not able to compute a single depth iteration")
        info_sum = self._calc_info_ratio(info_sum, final_search_depth)
        info_sum.info['depth'] = final_search_depth
        return values, action_probs, info_sum

    @staticmethod
    def _add_info_sum(info_sum: SearchInfo, info: SearchInfo) -> SearchInfo:
        info_sum.other_time_ratio += info.other_time_ratio
        info_sum.cleanup_time_ratio += info.cleanup_time_ratio
        info_sum.eval_time_ratio += info.eval_time_ratio
        info_sum.backup_time_ratio += info.backup_time_ratio
        info_sum.extract_time_ratio += info.extract_time_ratio
        info_sum.expansion_time_ratio += info.expansion_time_ratio
        return info_sum

    @staticmethod
    def _calc_info_ratio(info_sum: SearchInfo, num_iterations: int) -> SearchInfo:
        info_sum.other_time_ratio /= num_iterations
        info_sum.cleanup_time_ratio /= num_iterations
        info_sum.eval_time_ratio /= num_iterations
        info_sum.backup_time_ratio /= num_iterations
        info_sum.extract_time_ratio /= num_iterations
        info_sum.expansion_time_ratio /= num_iterations
        return info_sum
