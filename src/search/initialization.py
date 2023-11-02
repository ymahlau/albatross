from src.search import Search
from src.search.backup_func import backup_config_from_structured
from src.search.config import SearchConfig, SearchType, MCTSConfig, IterativeDeepeningConfig, FixedDepthConfig, \
    SMOOSConfig
from src.search.eval_func import eval_config_from_structured
from src.search.extraction_func import extract_config_from_structured
from src.search.fixed_depth import FixedDepthSearch
from src.search.iterative_deepening import IterativeDeepening
from src.search.mcts import MCTS
from src.search.sel_func import selection_config_from_structured
from src.search.smoos import SMOOS


def get_search_from_config(search_cfg: SearchConfig) -> Search:
    if search_cfg.search_type == SearchType.MCTS or search_cfg.search_type == SearchType.MCTS.value:
        return MCTS(search_cfg)
    elif search_cfg.search_type == SearchType.ITERATIVE_DEEPENING \
            or search_cfg.search_type == SearchType.ITERATIVE_DEEPENING.value:
        return IterativeDeepening(search_cfg)
    elif search_cfg.search_type == SearchType.FIXED_DEPTH \
            or search_cfg.search_type == SearchType.FIXED_DEPTH.value:
        return FixedDepthSearch(search_cfg)
    elif search_cfg.search_type == SearchType.SMOOS \
            or search_cfg.search_type == SearchType.SMOOS.value:
        return SMOOS(search_cfg)
    else:
        raise ValueError(f"Unknown search config type: {search_cfg}")


def search_config_from_structured(search_cfg) -> SearchConfig:
    kwargs = dict(search_cfg)
    eval_func_cfg = eval_config_from_structured(search_cfg.eval_func_cfg)
    kwargs["eval_func_cfg"] = eval_func_cfg
    extract_func_cfg = extract_config_from_structured(search_cfg.extract_func_cfg)
    kwargs["extract_func_cfg"] = extract_func_cfg
    if search_cfg.search_type != SearchType.SMOOS and search_cfg.search_type != SearchType.SMOOS.value:
        backup_func_cfg = backup_config_from_structured(search_cfg.backup_func_cfg)
        kwargs["backup_func_cfg"] = backup_func_cfg
    if search_cfg.search_type == SearchType.MCTS or search_cfg.search_type == SearchType.MCTS.value:
        sel_func_cfg = selection_config_from_structured(search_cfg.sel_func_cfg)
        kwargs["sel_func_cfg"] = sel_func_cfg
        return MCTSConfig(**kwargs)
    elif search_cfg.search_type == SearchType.ITERATIVE_DEEPENING \
            or search_cfg.search_type == SearchType.ITERATIVE_DEEPENING.value:
        return IterativeDeepeningConfig(**kwargs)
    elif search_cfg.search_type == SearchType.FIXED_DEPTH \
            or search_cfg.search_type == SearchType.FIXED_DEPTH.value:
        return FixedDepthConfig(**kwargs)
    elif search_cfg.search_type == SearchType.SMOOS \
            or search_cfg.search_type == SearchType.SMOOS.value:
        return SMOOSConfig(**kwargs)
    else:
        raise ValueError(f"Unknown search config type: {search_cfg}")
