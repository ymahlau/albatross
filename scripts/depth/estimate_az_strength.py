from datetime import datetime
import itertools
import math
from pathlib import Path
import pickle
import random
from matplotlib import pyplot as plt
import numpy as np
import scipy
import seaborn
from trueskill import Rating, rate
import trueskill
from scripts.depth.eval_tournament import play_single_game
from src.agent import Agent
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgentConfig
from src.agent.search_agent import AreaControlSearchAgentConfig
from src.game.actions import sample_individual_actions

from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7, survive_on_7x7_4_player, survive_on_7x7_constrictor, survive_on_7x7_constrictor_4_player
from src.game.game import Game
from src.game.initialization import get_game_from_config
from src.game.utils import step_with_draw_prevention
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file

def estimate_az(experiment_id: int):
    num_seeds = 5
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_az_strength'
    base_name = 'az_strength'
    num_games_per_part = 100
    depths = np.asarray(list(range(50, 2001, 50)), dtype=int)
    depth_dict = {
        x: d for x, d in enumerate(depths)
    }
    
    game_dict = {
        '4nd7': survive_on_7x7_4_player(),
        'd7': survive_on_7x7_constrictor(),
        'nd7': survive_on_7x7(),
        '4d7': survive_on_7x7_constrictor_4_player(),
    }
    
    pref_lists = [
        list(game_dict.keys()),
        list(range(num_seeds)),
    ]
    prod = list(itertools.product(*pref_lists))
    
    prefix, seed = prod[experiment_id]
    full_save_path = save_path / f'{base_name}_{prefix}_{seed}.pkl'
    num_agents = 4 if prefix.startswith("4") else 2
    print(f"{datetime.now()} - Started {prefix} with {seed=}", flush=True)
    print(f"{full_save_path=}", flush=True)
    
    set_seed(seed)   
    game_cfg = game_dict[prefix]
    
    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'battlesnake'
    resp_path = net_path / f'{prefix}_resp_{seed}' / 'latest.pt'
    proxy_path = net_path / f'{prefix}_proxy_{seed}' / 'latest.pt'
    az_path = net_path / f'{prefix}_{seed}' / 'latest.pt'
    
    net = get_network_from_file(resp_path).eval()
    alb_network_agent_cfg = NetworkAgentConfig(
        net_cfg=net.cfg,
        temperature_input=True,
        single_temperature=False,
        init_temperatures=[0, 0, 0, 0] if prefix.startswith('4') else [0, 0],
    )
    alb_online_agent_cfg = AlbatrossAgentConfig(
        num_player=num_agents,
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
    base_agent = get_agent_from_config(base_agent_cfg)
    
    agent_dict = {
        idx: base_agent for idx in range(len(depths))
    }
    agent_dict[len(depths)] = alb_online_agent  # albatross
    agent_dict[len(depths) + 1] = az_agent  # alphaZero
    
    game_cfg.ec.temperature_input = False
    game_cfg.ec.single_temperature_input = False
    game = get_game_from_config(game_cfg)
    
    cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}.pkl'
    alb_online_agent.cfg.estimate_log_path = str(cur_log_path)
    
    result_list = []
    for game_idx in range(num_games_per_part):
        # sample agents without replacement
        if num_agents == 2:
            cur_agent_list = [alb_online_agent, az_agent]
            cur_temperatures = [math.inf, math.inf]
            cur_iterations = [1, 1]
        else:
            sampled_indices = random.sample(range(len(depth_dict)), 2)    
            cur_agent_list = [alb_online_agent, az_agent] + [agent_dict[idx] for idx in sampled_indices]
            cur_iterations = [1, 1] + [depth_dict[idx] for idx in sampled_indices]
            cur_temperatures = [math.inf, math.inf, 1, 1]
        
        cur_result, cur_length = play_single_game(
            game=game,
            agent_list=cur_agent_list,
            iterations=cur_iterations,
            temperatures=cur_temperatures,
            prevent_draw=False,
            verbose_level=0,
        )
        result_list.append((cur_result, cur_length))
        
        with open(full_save_path, 'wb') as f:
            pickle.dump(result_list, f)
        print(f"{datetime.now()} - {game_idx}/{num_games_per_part}: {cur_result}, {cur_length}", flush=True)

def print_strength():
    num_seeds = 5
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_az_strength'
    base_name = 'az_strength'
    num_games_per_part = 100
    prefix = '4nd7'
    
    full_list = []
    for seed in range(num_seeds):
        cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}.pkl'
        with open(cur_log_path, 'rb') as f:
            data = pickle.load(f)
        last_entries = [e[1][-1] for e in data['temp_estimates']]
        full_list.append(last_entries)
    full_arr = np.asarray(full_list)
    print(np.mean(full_arr))



if __name__ == '__main__':
    # estimate_az(0)
    print_strength()
