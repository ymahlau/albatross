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

from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7, survive_on_7x7_4_player, survive_on_7x7_constrictor, survive_on_7x7_constrictor_4_player
from src.game.initialization import get_game_from_config
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def alb_vs_proxy_at_temps(experiment_id: int):
    temperatures = np.linspace(0, 10, 100)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_proxy_temps'
    base_name = 'alb_proxy_temps'
    
    game_dict = {
        '4nd7': survive_on_7x7_4_player(),
        'd7': survive_on_7x7_constrictor(),
        'nd7': survive_on_7x7(),
        '4d7': survive_on_7x7_constrictor_4_player(),
    }
    
    pref_lists = [
        list(game_dict.keys()),
        list(range(5)),
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed = prod[experiment_id]
    assert isinstance(prefix, str)
    num_games_per_part = 100
    
    set_seed(seed)   
    game_cfg = game_dict[prefix]

    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'battlesnake'
    resp_path = net_path / f'{prefix}_resp_{seed}' / 'latest.pt'
    proxy_path = net_path / f'{prefix}_proxy_{seed}' / 'latest.pt'
    
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
    
    proxy_net = get_network_from_file(proxy_path)
    proxy_net = proxy_net.eval()
    proxy_net_agent_cfg = NetworkAgentConfig(
        net_cfg=proxy_net.cfg,
        temperature_input=True,
        single_temperature=True,
        init_temperatures=[10, 10],
    )
    proxy_net_agent = NetworkAgent(proxy_net_agent_cfg)
    proxy_net_agent.replace_net(proxy_net)
    
    game_cfg.ec.temperature_input = True
    game_cfg.ec.single_temperature_input = True
    game = get_game_from_config(game_cfg)
    
    print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}', flush=True)
    full_result_list_alb, full_length_list_alb = [], []
    full_save_path = save_path / f'{base_name}_{prefix}_{seed}.pkl'
    
    for idx, temperature in enumerate(temperatures):
        # cur save path
        cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}_{idx}.pkl'
        alb_online_agent.cfg.estimate_log_path = str(cur_log_path)
        
        proxy_net_agent.reset_episode()
        proxy_net_agent.set_temperatures([temperature for _ in range(game_cfg.num_players)])
        
        print(f'Started evaluation with: {idx=}, {temperature=}')
        
        results_alb, game_length_alb = do_evaluation(
            game=game,
            evaluee=alb_online_agent,
            opponent_list=[proxy_net_agent],
            num_episodes=[num_games_per_part],
            enemy_iterations=0,
            temperature_list=[1],
            own_temperature=math.inf,
            prevent_draw=False,
            switch_positions=False,
            verbose_level=1,
            own_iterations=1,
        )
        full_result_list_alb.append(results_alb)
        full_length_list_alb.append(game_length_alb)
        
        save_dict = {
            'results_alb': np.asarray(full_result_list_alb),
            'lengths_alb': np.asarray(full_length_list_alb),
        }
        
        with open(full_save_path, 'wb') as f:
            pickle.dump(save_dict, f)

def albfix_vs_proxy_at_temps(experiment_id: int):
    temperatures = np.linspace(0, 10, 100)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_proxy_temps'
    base_name = 'albfix_proxy_temps'
    
    game_dict = {
        '4nd7': survive_on_7x7_4_player(),
        'd7': survive_on_7x7_constrictor(),
        'nd7': survive_on_7x7(),
        '4d7': survive_on_7x7_constrictor_4_player(),
    }
    
    pref_lists = [
        list(game_dict.keys()),
        list(range(5)),
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed = prod[experiment_id]
    assert isinstance(prefix, str)
    num_games_per_part = 100
    
    set_seed(seed)   
    game_cfg = game_dict[prefix]

    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'battlesnake'
    resp_path = net_path / f'{prefix}_resp_{seed}' / 'latest.pt'
    proxy_path = net_path / f'{prefix}_proxy_{seed}' / 'latest.pt'
    
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
    
    proxy_net = get_network_from_file(proxy_path)
    proxy_net = proxy_net.eval()
    proxy_net_agent_cfg = NetworkAgentConfig(
        net_cfg=proxy_net.cfg,
        temperature_input=True,
        single_temperature=True,
        init_temperatures=[10, 10],
    )
    proxy_net_agent = NetworkAgent(proxy_net_agent_cfg)
    proxy_net_agent.replace_net(proxy_net)
    
    game_cfg.ec.temperature_input = True
    game_cfg.ec.single_temperature_input = True
    game = get_game_from_config(game_cfg)
    
    print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}', flush=True)
    full_result_list_alb, full_length_list_alb = [], []
    full_save_path = save_path / f'{base_name}_{prefix}_{seed}.pkl'
    
    if os.path.exists(full_save_path):
        with open(full_save_path, 'rb') as f:
            last_result_dict = pickle.load(f)
        full_result_list_alb = last_result_dict['results_alb'].tolist()
        full_length_list_alb = last_result_dict['lengths_alb'].tolist()
        
        num_complete_iterations = len(full_result_list_alb)
        temperatures = temperatures[num_complete_iterations:]
    
    for idx, temperature in enumerate(temperatures):        
        proxy_net_agent.reset_episode()
        proxy_net_agent.set_temperatures([temperature for _ in range(game_cfg.num_players)])
        
        alb_online_agent.cfg.fixed_temperatures = [temperature for _ in range(game_cfg.num_players)]
        
        print(f'Started evaluation with: {idx=}, {temperature=}')
        
        results_alb, game_length_alb = do_evaluation(
            game=game,
            evaluee=alb_online_agent,
            opponent_list=[proxy_net_agent],
            num_episodes=[num_games_per_part],
            enemy_iterations=0,
            temperature_list=[1],
            own_temperature=math.inf,
            prevent_draw=False,
            switch_positions=False,
            verbose_level=1,
            own_iterations=1,
        )
        full_result_list_alb.append(results_alb)
        full_length_list_alb.append(game_length_alb)
        
        save_dict = {
            'results_alb': np.asarray(full_result_list_alb),
            'lengths_alb': np.asarray(full_length_list_alb),
        }
        
        with open(full_save_path, 'wb') as f:
            pickle.dump(save_dict, f)


def plot_both_bs():
    temperatures = np.linspace(0, 10, 100)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_proxy_temps'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'bs_proxy_temps'
    
    game_dict = {
        'd7': survive_on_7x7_constrictor(),
        '4d7': survive_on_7x7_constrictor_4_player(),
        'nd7': survive_on_7x7(),
        '4nd7': survive_on_7x7_4_player(),
    }
    
    full_names = {
        'd7': 'Deterministic 2-Player',
        '4d7': 'Deterministic 4-Player',
        'nd7': 'Stochastic 2-Player',
        '4nd7': 'Stochastic 4-Player',
    }
    
    for idx, prefix in enumerate(game_dict.keys()):
        data, fix_data = [], []
        for seed in range(5):            
            full_save_path = save_path / f'alb_proxy_temps_{prefix}_{seed}.pkl'
            with open(full_save_path, 'rb') as f:
                cur_data = pickle.load(f)
            data.append(cur_data['results_alb'])
            
            full_save_path = save_path / f'albfix_proxy_temps_{prefix}_{seed}.pkl'
            with open(full_save_path, 'rb') as f:
                cur_data = pickle.load(f)
            fix_data.append(cur_data['results_alb'])
        
        data_arr = np.asarray(data)[:, :, 0, :].mean(axis=-1)
        data_arr_fix = np.asarray(fix_data)[:, :, 0, :].mean(axis=-1)
        
        plt.clf()
        plt.figure()
        seaborn.set_theme(style='whitegrid')
        
        
        plot_filled_std_curves(
            x=temperatures,
            mean=data_arr.mean(axis=0),
            std=data_arr.std(axis=0),
            color=COLORS[1],
            lighter_color=LIGHT_COLORS[1],
            linestyle=LINESTYLES[0],
            label='Alb. vs. Proxy' if idx == 0 else None,
            min_val=0,
        )
        
        plot_filled_std_curves(
            x=temperatures,
            mean=data_arr_fix.mean(axis=0),
            std=data_arr_fix.std(axis=0),
            color=COLORS[0],
            lighter_color=LIGHT_COLORS[0],
            linestyle=LINESTYLES[1],
            label='Alb.* vs. Proxy' if idx == 0 else None,
            min_val=0,
        )
        
        fontsize = 'xx-large'
        plt.xlim(temperatures[0], temperatures[-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.title(full_names[prefix], fontsize=fontsize)
        if idx == 0:
            plt.legend(fontsize='x-large')
        plt.ylabel('Reward', fontsize=fontsize)
        plt.xlabel('Temperature', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'bs_resp_proxy_{prefix}.pdf', bbox_inches='tight', pad_inches=0.0)
        

if __name__ == '__main__':
    # alb_vs_proxy_at_temps(0)
    # albfix_vs_proxy_at_temps(0)
    plot_both_bs()
    