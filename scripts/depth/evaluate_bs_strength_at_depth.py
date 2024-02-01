import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import yaml

from src.game.battlesnake.bootcamp.test_envs_5x5 import survive_on_5x5
from src.depth.depth_parallel import DepthSearchConfig, compute_different_depths_parallel
from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7, survive_on_7x7_4_player, survive_on_7x7_constrictor, survive_on_7x7_constrictor_4_player
from src.misc.serialization import serialize_dataclass
from src.search.config import DecoupledUCTSelectionConfig, AreaControlEvalConfig, StandardBackupConfig, \
    StandardExtractConfig, MCTSConfig

def eval_bs_depth():
    game_dict = {
        'd7': survive_on_7x7_constrictor(),
        'nd7': survive_on_7x7(),
        '4nd7': survive_on_7x7_4_player(),
        '4d7': survive_on_7x7_constrictor_4_player(),
    }
    base_name = 'debug'
    
    pref_lists = [
        list(game_dict.keys()),
        list(range(20)),
    ]
    
    game_cfg = survive_on_7x7()

    sel_func_cfg = DecoupledUCTSelectionConfig()
    eval_func_cfg = AreaControlEvalConfig()
    backup_func_cfg = StandardBackupConfig()
    extract_func_cfg = StandardExtractConfig()
    mcts_cfg = MCTSConfig(
        sel_func_cfg=sel_func_cfg,
        eval_func_cfg=eval_func_cfg,
        backup_func_cfg=backup_func_cfg,
        extract_func_cfg=extract_func_cfg,
        expansion_depth=0,
        use_hot_start=False,
        optimize_fully_explored=False,
    )
    search_spec = {
        'duct': (mcts_cfg, None, 100, 5)
    }
    save_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'debug_data'
    cfg = DepthSearchConfig(
        game_cfg=game_cfg,
        search_specs=search_spec,
        num_samples=3,
        num_procs=2,
        step_temperature=5,
        step_iterations=None,  # if none, use gt result
        step_search='duct',
        draw_prevention=True,
        save_path=str(save_path),
        restrict_cpu=True,
        seed=42,
        include_ja_values=True,
        include_obs=True,
        device_str='cpu',
        min_available_actions=2,
        max_available_actions=3,
    )
    
    compute_different_depths_parallel(cfg)
    
    # initialize yaml file and hydra    
    # exported_dict = serialize_dataclass(cfg)
    # yaml_str = yaml.dump(exported_dict)
    # yaml_str += 'hydra:\n  run:\n    dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
    # config_name = 'debug_config'
    # config_dir = Path(__file__).parent.parent.parent / 'config'
    # cur_config_file = config_dir / f'{config_name}.yaml'
    # with open(cur_config_file, 'w') as f:
    #     f.write(yaml_str)

    # sys.argv = [
    #     'cmd',  # this is ignored
    #     'hydra.job.chdir=True',
    #     # ... other dynamically added parameters
    # ]
    # hydra.main(config_path=str(config_dir), config_name=config_name, version_base=None)(main)()


if __name__ == '__main__':
    eval_bs_depth()
