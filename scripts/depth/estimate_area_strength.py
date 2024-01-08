from datetime import datetime
import itertools
import math
import os
from pathlib import Path
import pickle
from matplotlib import pyplot as plt

import numpy as np
import scipy
import seaborn
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig
from src.agent.search_agent import AreaControlSearchAgentConfig
from src.game.battlesnake.bootcamp.test_envs_7x7 import (survive_on_7x7, survive_on_7x7_4_player,
    survive_on_7x7_constrictor, survive_on_7x7_constrictor_4_player)
from src.game.initialization import get_game_from_config
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def evaluate_bs_depth_strength(experiment_id: int):
    num_games_per_part = 5
    num_parts = 10
    search_iterations = np.arange(50, 2001, 50)
    # search_iterations = [50]
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth_strength'
    base_name = 'bs_5g_t1'
    # eval_az = False
    
    game_dict = {
        'd7': survive_on_7x7_constrictor(),
        'nd7': survive_on_7x7(),
        '4nd7': survive_on_7x7_4_player(),
        '4d7': survive_on_7x7_constrictor_4_player(),
    }
    
    pref_lists = [
        list(game_dict.keys()),
        list(range(5)),
        list(range(int(num_parts)))
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed, cur_game_id = prod[experiment_id]
    assert isinstance(prefix, str)
    # we do not want to set the same seed in every game and repeat the same play.
    # Therefore, set a different seed for every game and base seed
    set_seed((seed + 1) * cur_game_id)  
    game_cfg = game_dict[prefix]

    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'battlesnake'
    resp_path = net_path / f'{prefix}_resp_{seed}' / 'latest.pt'
    proxy_path = net_path / f'{prefix}_proxy_{seed}' / 'latest.pt'
    # az_path = net_path / f'{prefix}_{seed}' / 'latest.pt'
    
    net = get_network_from_file(resp_path).eval()
    alb_network_agent_cfg = NetworkAgentConfig(
        net_cfg=net.cfg,
        temperature_input=True,
        single_temperature=False,
        init_temperatures=[0, 0, 0, 0] if prefix.startswith('4') else [0, 0],
    )
    alb_online_agent_cfg = AlbatrossAgentConfig(
        num_player=4 if prefix.startswith('4') else 2,
        agent_cfg=alb_network_agent_cfg,
        device_str='cpu',
        response_net_path=str(resp_path),
        proxy_net_path=str(proxy_path),
        noise_std=None,
        num_samples=1,
        init_temp=5,
    )
    alb_online_agent = AlbatrossAgent(alb_online_agent_cfg)
    
    # net = get_network_from_file(az_path).eval()
    # az_cfg = NetworkAgentConfig(net_cfg=net.cfg)
    # az_agent = get_agent_from_config(az_cfg)
    # az_agent.replace_net(net)
    
    base_agent_cfg = AreaControlSearchAgentConfig()
    base_agent = get_agent_from_config(base_agent_cfg)
    
    game_cfg.ec.temperature_input = False
    game_cfg.ec.single_temperature_input = False
    game = get_game_from_config(game_cfg)
    
    print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}', flush=True)
    full_result_list_alb, full_length_list_alb, full_result_list_az, full_length_list_az = [], [], [], []
    
    full_save_path = save_path / f'{base_name}_{prefix}_{seed}_{cur_game_id}.pkl'
    # if os.path.exists(full_save_path):
    #     with open(full_save_path, 'rb') as f:
    #         last_result_dict = pickle.load(f)
    #     full_result_list_alb = last_result_dict['results_alb'].tolist()
    #     full_length_list_alb = last_result_dict['lengths_alb'].tolist()
    #     # if eval_az:
    #     #     full_result_list_az = last_result_dict['results_az'].tolist()
    #     #     full_length_list_az = last_result_dict['lengths_az'].tolist()
        
    #     num_complete_iterations = len(full_result_list_alb)
    #     search_iterations = search_iterations[num_complete_iterations:]
    
    
    for iteration_idx, cur_iterations in enumerate(search_iterations):
        # cur save path
        cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}_{cur_game_id}_{cur_iterations}.pkl'
        alb_online_agent.cfg.estimate_log_path = str(cur_log_path)
        
        print(f'Started evaluation with: {iteration_idx=}, {cur_iterations=}')
        
        results_alb, game_length_alb = do_evaluation(
            game=game,
            evaluee=alb_online_agent,
            opponent_list=[base_agent],
            num_episodes=[num_games_per_part],
            enemy_iterations=cur_iterations,
            temperature_list=[1],
            own_temperature=math.inf,
            prevent_draw=False,
            switch_positions=False,
            verbose_level=1,
            own_iterations=1,
        )
        full_result_list_alb.append(results_alb)
        full_length_list_alb.append(game_length_alb)
        
        # if eval_az:
        #     results_az, game_length_az = do_evaluation(
        #         game=game,
        #         evaluee=az_agent,
        #         opponent_list=[base_agent],
        #         num_episodes=[num_games_per_part],
        #         enemy_iterations=cur_iterations,
        #         temperature_list=[math.inf],
        #         own_temperature=math.inf,
        #         prevent_draw=True,
        #         switch_positions=False,
        #         verbose_level=1,
        #         own_iterations=1,
        #     )
        #     full_result_list_az.append(results_az)
        #     full_length_list_az.append(game_length_az)
        
        save_dict = {
            'results_alb': np.asarray(full_result_list_alb),
            'lengths_alb': np.asarray(full_length_list_alb),
        }
        # if eval_az:
        #     save_dict['results_az'] = np.asarray(full_result_list_az)
        #     save_dict['lengths_az'] = np.asarray(full_length_list_az)
        
        with open(full_save_path, 'wb') as f:
            pickle.dump(save_dict, f)

def get_area_strength_estimates():
    num_parts = 10
    num_seeds = 5
    search_iterations = np.arange(50, 2001, 50)
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth_strength'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'bs_depth_strength'
    base_name = 'bs_5g_t1_log'
    
    prefixes = ['d7', 'nd7', '4d7', '4nd7']
    
    min_action_count = 3
    
    for prefix in prefixes:
        depth_list = []
        for depth in search_iterations:
            cur_depth_list = []
            for seed in range(num_seeds):
                cur_seed_list = []
                for part in range(num_parts):
                    cur_path = save_path / f'{base_name}_{prefix}_{seed}_{part}_{int(depth)}.pkl'
                    with open(cur_path, 'rb') as f:
                        cur_data = pickle.load(f)
                    for cur_dict in cur_data['temp_estimates']:
                        if len(cur_dict[1]) >= min_action_count:
                            cur_seed_list.append(cur_dict[1][-1])
                        if prefix.startswith('4'):
                            if len(cur_dict[2]) >= min_action_count:
                                cur_seed_list.append(cur_dict[2][-1])
                            if len(cur_dict[3]) >= min_action_count:
                                cur_seed_list.append(cur_dict[3][-1])
                seed_arr = np.asarray(cur_seed_list)
                cur_depth_list.append(seed_arr.mean())
            depth_list.append(cur_depth_list)
        depth_arr = np.asarray(depth_list)
        mean_arr = depth_arr.mean(axis=-1)
        mean_arr = scipy.signal.savgol_filter(mean_arr, window_length=3, polyorder=1)
        std_arr = depth_arr.std(axis=-1)
        std_arr = scipy.signal.savgol_filter(std_arr, window_length=3, polyorder=1)
        
        plt.clf()
        plt.figure(dpi=600)
        seaborn.set_theme(style='whitegrid')
        
        plot_filled_std_curves(
            x=search_iterations,
            mean=mean_arr,
            std=std_arr,
            color=COLORS[0],
            lighter_color=LIGHT_COLORS[0],
            linestyle=LINESTYLES[0],
            label=None,
        )
        
        fontsize = 'medium'
        plt.xlabel('Enemy Search Iterations', fontsize=fontsize)
        plt.ylabel('Temperature Estimation', fontsize=fontsize)
        plt.xlim(search_iterations[0], search_iterations[-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f't1_smoothed_{prefix}.pdf')


def save_mean_strength_estimates():
    num_parts = 10
    num_seeds = 5
    search_iterations = np.arange(50, 2001, 50)
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth_strength'
    result_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth_strength_mean'
    base_name = 'bs_5g_t1_log'
    
    prefixes = ['d7', 'nd7', '4d7', '4nd7']
    
    min_action_count = 3
    
    for prefix in prefixes:
        depth_list = []
        for depth in search_iterations:
            cur_depth_list = []
            for seed in range(num_seeds):
                cur_seed_list = []
                for part in range(num_parts):
                    cur_path = save_path / f'{base_name}_{prefix}_{seed}_{part}_{int(depth)}.pkl'
                    with open(cur_path, 'rb') as f:
                        cur_data = pickle.load(f)
                    for cur_dict in cur_data['temp_estimates']:
                        if len(cur_dict[1]) >= min_action_count:
                            cur_seed_list.append(cur_dict[1][-1])
                        if prefix.startswith('4'):
                            if len(cur_dict[2]) >= min_action_count:
                                cur_seed_list.append(cur_dict[2][-1])
                            if len(cur_dict[3]) >= min_action_count:
                                cur_seed_list.append(cur_dict[3][-1])
                seed_arr = np.asarray(cur_seed_list)
                cur_depth_list.append(seed_arr.mean())
            depth_list.append(cur_depth_list)
        depth_arr = np.asarray(depth_list)
        mean_arr = depth_arr.mean(axis=-1)
        mean_arr = scipy.signal.savgol_filter(mean_arr, window_length=3, polyorder=1)
        
        cur_result_dict = {
            search_iterations[idx].item(): mean_arr[idx] for idx in range(len(search_iterations))
        }
        with open(result_path / f"{prefix}.pkl", 'wb') as f:
            pickle.dump(cur_result_dict, f)
        


if __name__ == '__main__':
    # evaluate_bs_depth_strength(0)
    # get_area_strength_estimates()
    save_mean_strength_estimates()
