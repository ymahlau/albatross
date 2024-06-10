from datetime import datetime
import itertools
import math
import os
from pathlib import Path
import pickle

from matplotlib import pyplot as plt
import numpy as np
import seaborn
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig
from src.agent.search_agent import AreaControlSearchAgentConfig
from src.game.battlesnake.bootcamp.test_envs_11x11 import survive_on_11x11_constrictor_4_player_coop
from src.game.battlesnake.bootcamp.test_envs_9x9 import survive_on_9x9_constrictor_4_player_coop
from src.game.initialization import get_game_from_config
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.search.config import SymmetricAreaControlEvalConfig
from src.trainer.az_evaluator import do_evaluation
from src.trainer.az_worker import discounted_rewards


def evaluate_bs_depth_func_coop(experiment_id: int):
    num_parts = 5
    num_games_per_part = 50
    search_iterations = [50] + list(range(100, 2001, 100))
    # search_iterations = np.arange(3000, 20001, 1000)
    # search_iterations = np.asarray([500])
    # search_iterations = np.arange(100, 1001, 100)
    # search_iterations = [int(5e4), int(1e5), int(5e5), int(1e6)]
    # search_iterations = [int(5e5), int(1e6)]
    # search_iterations = [int(5e6)]
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth_coop'
    # save_path = Path(__file__).parent.parent.parent / 'a_data' / 'temp'
    base_name = 'coop_ac_2k'
    eval_az = True
    
    game_dict = {
        # '4dc9': survive_on_9x9_constrictor_4_player_coop(),
        '4dc11': survive_on_11x11_constrictor_4_player_coop(),
    }
    
    pref_lists = [
        list(game_dict.keys()),
        list(range(5)),
        list(range(int(num_parts)))
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed, cur_game_id = prod[experiment_id]
    assert isinstance(prefix, str)
    # if 'n' in prefix:
    #     num_games_per_part = 50
    
    # we do not want to set the same seed in every game and repeat the same play.
    # Therefore, set a different seed for every game and base seed
    set_seed(cur_game_id + seed * num_parts)   
    game_cfg = game_dict[prefix]

    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'battlesnake_coop'
    resp_path = net_path / f'{prefix}_resp_{seed}.pt'
    proxy_path = net_path / f'{prefix}_proxy_{seed}.pt'
    az_path = net_path / f'{prefix}_{seed}.pt'
    # temp_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth_strength_mean'
    # with open(temp_path / f'{prefix}.pkl', 'rb') as f:
    #     mean_temps = pickle.load(f)
    
    net = get_network_from_file(resp_path).eval()
    alb_network_agent_cfg = NetworkAgentConfig(
        net_cfg=net.cfg,
        temperature_input=True,
        single_temperature=False,
        init_temperatures=[0, 0, 0, 0] if prefix.startswith('4') else [0, 0],
    )
    alb_online_agent_cfg = AlbatrossAgentConfig(
        num_player=4,
        agent_cfg=alb_network_agent_cfg,
        device_str='cpu',
        response_net_path=str(resp_path),
        proxy_net_path=str(proxy_path),
        noise_std=None,
        num_samples=1,
        init_temp=5,
    )
    alb_online_agent = AlbatrossAgent(alb_online_agent_cfg)
    
    net = get_network_from_file(az_path).eval()
    az_cfg = NetworkAgentConfig(net_cfg=net.cfg)
    az_agent = get_agent_from_config(az_cfg)
    az_agent.replace_net(net)
    
    base_agent_cfg = AreaControlSearchAgentConfig()
    base_agent_cfg.search_cfg.eval_func_cfg = SymmetricAreaControlEvalConfig()
    base_agent = get_agent_from_config(base_agent_cfg)
    
    game_cfg.ec.temperature_input = False
    game_cfg.ec.single_temperature_input = False
    game = get_game_from_config(game_cfg)
    
    print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}', flush=True)
    full_result_list_alb, full_length_list_alb, full_result_list_az, full_length_list_az = [], [], [], []
    
    full_save_path = save_path / f'{base_name}_{prefix}_{seed}_{cur_game_id}.pkl'
    if os.path.exists(full_save_path):
        with open(full_save_path, 'rb') as f:
            last_result_dict = pickle.load(f)
        full_result_list_alb = last_result_dict['results_alb'].tolist()
        full_length_list_alb = last_result_dict['lengths_alb'].tolist()
        if eval_az:
            full_result_list_az = last_result_dict['results_az'].tolist()
            full_length_list_az = last_result_dict['lengths_az'].tolist()
        
        num_complete_iterations = len(full_result_list_az)
        search_iterations = search_iterations[num_complete_iterations:]
    
    
    for iteration_idx, cur_iterations in enumerate(search_iterations):
        # cur save path
        cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}_{cur_game_id}_{cur_iterations}.pkl'
        alb_online_agent.cfg.estimate_log_path = str(cur_log_path)
        
        # cur_temp = mean_temps[cur_iterations.item()]
        # alb_online_agent.cfg.fixed_temperatures = [cur_temp for _ in range(game_cfg.num_players)]
        
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
            return_all_rewards=True,
        )
        full_result_list_alb.append(results_alb)
        full_length_list_alb.append(game_length_alb)
        
        if eval_az:
            results_az, game_length_az = do_evaluation(
                game=game,
                evaluee=az_agent,
                opponent_list=[base_agent],
                num_episodes=[num_games_per_part],
                enemy_iterations=cur_iterations,
                temperature_list=[1],
                own_temperature=math.inf,
                prevent_draw=False,
                switch_positions=False,
                verbose_level=1,
                own_iterations=1,
                return_all_rewards=True,
            )
            full_result_list_az.append(results_az)
            full_length_list_az.append(game_length_az)
        
        save_dict = {
            'results_alb': full_result_list_alb,
            'lengths_alb': full_length_list_alb,
        }
        # save_dict = {}
        if eval_az:
            save_dict['results_az'] = full_result_list_az
            save_dict['lengths_az'] = full_length_list_az
        
        with open(full_save_path, 'wb') as f:
            pickle.dump(save_dict, f)


def plot_bs_depth_coop():
    data_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth_coop'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'bs_depth_coop'
    num_parts = 5
    num_seeds = 5
    num_games_per_part = 50
    discount = 0.9
    
    # tmp_max_idx = 14
    search_iterations = np.asarray([50] + list(range(100, 2001, 100)))
    num_iterations = len(search_iterations)
    base_name = 'coop_ac_2k'
    
    
    full_list_az, full_list_alb = [], []
    for seed in range(num_seeds):
        for part in range(num_parts):
            file_name_alb = f'{base_name}_4dc11_{seed}_{part}.pkl'
            with open(data_path / file_name_alb, 'rb') as f:
                cur_dict = pickle.load(f)
            alb_results = cur_dict['results_alb']
            az_results = cur_dict['results_az']
            for t_idx, _ in enumerate(search_iterations):
                for game_idx in range(num_games_per_part):
                    cur_results = np.asarray(alb_results[t_idx][game_idx])
                    disc_results = discounted_rewards(cur_results, discount)
                    full_list_alb.append(disc_results[0])
                    
                    cur_results = np.asarray(az_results[t_idx][game_idx])
                    disc_results = discounted_rewards(cur_results, discount)
                    full_list_az.append(disc_results[0])
    full_arr_az = np.asarray(full_list_az).reshape(num_seeds, num_parts, num_iterations, num_games_per_part)
    full_arr_alb = np.asarray(full_list_alb).reshape(num_seeds, num_parts, num_iterations, num_games_per_part)
    
    az_plot_vals = full_arr_az.mean(axis=-1).mean(axis=1)
    alb_plot_vals = full_arr_alb.mean(axis=-1).mean(axis=1)
    
    plt.clf()
    plt.figure(dpi=600, figsize=(6, 4))
    seaborn.set_theme(style='whitegrid')
    
    # albatross
    plot_filled_std_curves(
        x=search_iterations,
        mean=alb_plot_vals.mean(axis=0),
        std=alb_plot_vals.std(axis=0),
        color=COLORS[1],
        lighter_color=LIGHT_COLORS[1],
        linestyle=LINESTYLES[0],
        label='Albatross',
    )
    
    # AlphaZero
    plot_filled_std_curves(
        x=search_iterations,
        mean=az_plot_vals.mean(axis=0),
        std=az_plot_vals.std(axis=0),
        color=COLORS[2],
        lighter_color=LIGHT_COLORS[2],
        linestyle=LINESTYLES[1],
        label='AlphaZero',
    )
    
    fontsize = 'x-large'
    plt.xlabel('Partner Search Iterations', fontsize=fontsize)
    plt.ylabel('Discounted Reward', fontsize=fontsize)
    plt.xlim(search_iterations[0], search_iterations[-1])
    plt.xticks([500, 1000, 1500, 2000], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.xscale('log')
    # plt.legend(fontsize=fontsize, loc='upper right', bbox_to_anchor=(1.03, 1.05))
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig(img_path / f'inf_100g_{abbrev}_depths.png')
    plt.savefig(img_path / f'4dc11.pdf', bbox_inches='tight', pad_inches=0.03, dpi=1000)
    


def plot_bs_depth_coop_game_length():
    data_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth_coop'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'bs_depth_coop'
    num_parts = 5
    num_seeds = 5
    num_games_per_part = 50
    
    # tmp_max_idx = 14
    search_iterations = np.asarray([50] + list(range(100, 2001, 100)))
    num_iterations = len(search_iterations)
    base_name = 'coop_ac_2k'
    
    
    full_list_az, full_list_alb = [], []
    for seed in range(num_seeds):
        for part in range(num_parts):
            file_name_alb = f'{base_name}_4dc11_{seed}_{part}.pkl'
            with open(data_path / file_name_alb, 'rb') as f:
                cur_dict = pickle.load(f)
                
            full_list_alb.append(
                np.asarray(cur_dict['lengths_alb']).squeeze()
            )
            full_list_az.append(
                np.asarray(cur_dict['lengths_az']).squeeze()
            )
            
    full_arr_az = np.asarray(full_list_az).reshape(num_seeds, num_parts, num_iterations, num_games_per_part)
    full_arr_alb = np.asarray(full_list_alb).reshape(num_seeds, num_parts, num_iterations, num_games_per_part)
    
    az_plot_vals = full_arr_az.mean(axis=-1).mean(axis=1)
    alb_plot_vals = full_arr_alb.mean(axis=-1).mean(axis=1)
    
    plt.clf()
    plt.figure(dpi=600, figsize=(6, 4))
    seaborn.set_theme(style='whitegrid')
    
    # albatross
    plot_filled_std_curves(
        x=search_iterations,
        mean=alb_plot_vals.mean(axis=0),
        std=alb_plot_vals.std(axis=0),
        color=COLORS[1],
        lighter_color=LIGHT_COLORS[1],
        linestyle=LINESTYLES[0],
        label='Albatross',
    )
    
    # AlphaZero
    plot_filled_std_curves(
        x=search_iterations,
        mean=az_plot_vals.mean(axis=0),
        std=az_plot_vals.std(axis=0),
        color=COLORS[2],
        lighter_color=LIGHT_COLORS[2],
        linestyle=LINESTYLES[1],
        label='AlphaZero',
    )
    
    fontsize = 'xx-large'
    plt.xlabel('Partner Search Iterations', fontsize=fontsize)
    plt.ylabel('Game Length', fontsize=fontsize)
    plt.xlim(search_iterations[0], search_iterations[-1])
    plt.xticks([500, 1000, 1500, 2000], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.xscale('log')
    # plt.legend(fontsize=fontsize, loc='upper right', bbox_to_anchor=(1.03, 1.05))
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig(img_path / f'inf_100g_{abbrev}_depths.png')
    plt.savefig(img_path / f'4dc11_length.png', bbox_inches='tight', pad_inches=0.03, dpi=1000)


if __name__ == '__main__':
    # evaluate_bs_depth_func_coop(0)
    plot_bs_depth_coop()
    # plot_bs_depth_coop_game_length()
