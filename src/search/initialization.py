from src.search import Search
from src.search.config import SearchConfig, MCTSConfig, IterativeDeepeningConfig, FixedDepthConfig, SMOOSConfig
from src.search.fixed_depth import FixedDepthSearch
from src.search.iterative_deepening import IterativeDeepening
from src.search.mcts import MCTS
from src.search.smoos import SMOOS


def get_search_from_config(search_cfg: SearchConfig) -> Search:
    if isinstance(search_cfg, MCTSConfig):
        return MCTS(search_cfg)
    elif isinstance(search_cfg, IterativeDeepeningConfig):
        return IterativeDeepening(search_cfg)
    elif isinstance(search_cfg, FixedDepthConfig):
        return FixedDepthSearch(search_cfg)
    elif isinstance(search_cfg, SMOOSConfig):
        return SMOOS(search_cfg)
    else:
        raise ValueError(f"Unknown search config type: {search_cfg}")

