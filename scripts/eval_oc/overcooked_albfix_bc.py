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
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig, bc_agent_from_file
from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, \
    CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig, OvercookedRewardConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation

def eval_albfix_vs_bc(experiment_id: int):
    num_games = 100
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc'
    base_name = 'albfix_bc_temps'
    
    game_dicts = {
        'aa': AsymmetricAdvantageOvercookedConfig(),
        'cc': CounterCircuitOvercookedConfig(),
        'co': CoordinationRingOvercookedConfig(),
        'cr': CrampedRoomOvercookedConfig(),
        'fc': ForcedCoordinationOvercookedConfig(),
    }
    
    pref_lists = [
        ['aa', 'cc', 'co', 'cr', 'fc'],
        list(range(5)),
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed = prod[experiment_id]
    set_seed(seed)
    game_cfg = game_dicts[prefix]

    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'overcooked'
    resp_path = net_path / f'resp_{prefix}_{seed}' / 'latest.pt'
    proxy_path = net_path / f'proxy_{prefix}_{seed}' / 'latest.pt'
    
    net = get_network_from_file(resp_path).eval()
    alb_network_agent_cfg = NetworkAgentConfig(
        net_cfg=net.cfg,
        temperature_input=True,
        single_temperature=False,
        init_temperatures=[0, 0],
    )
    alb_online_agent_cfg = AlbatrossAgentConfig(
        num_player=2,
        agent_cfg=alb_network_agent_cfg,
        device_str='cpu',
        response_net_path=str(resp_path),
        proxy_net_path=str(proxy_path),
        noise_std=None,
        min_temp=0,
        max_temp=10,
        fixed_temperatures=[10, 10],
        num_samples=1,
        init_temp=0,
        sample_from_likelihood=False,
    )
    alb_online_agent = AlbatrossAgent(alb_online_agent_cfg)

    
    bc_path = Path(__file__).parent.parent.parent / 'bc_state_dicts'/ f'{prefix}_{seed}.pkl'
    bc_agent = bc_agent_from_file(bc_path)
    
    reward_cfg = OvercookedRewardConfig(
        placement_in_pot=0,
        dish_pickup=0,
        soup_pickup=0,
        soup_delivery=20,
        start_cooking=0,
    )
    game_cfg.reward_cfg = reward_cfg
    game_cfg.temperature_input = True
    game_cfg.single_temperature_input = False
    game = OvercookedGame(game_cfg)
    
    print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}', flush=True)
    full_result_list = []
    temperatures = np.linspace(0, 10, 50)
    full_save_path = save_path / f'{base_name}_{prefix}_{seed}.pkl'
    
    if os.path.exists(full_save_path):
        with open(full_save_path, 'rb') as f:
            last_result_list = pickle.load(f)
        full_result_list = last_result_list
        
        num_complete_iterations = len(full_result_list)
        temperatures = temperatures[num_complete_iterations:]
    
    for t_idx, t in enumerate(temperatures):
        print(f'Started evaluation with: {t_idx=}, {t=}')
        alb_online_agent.cfg.fixed_temperatures = [t, t]
        
        results, _ = do_evaluation(
            game=game,
            evaluee=alb_online_agent,
            opponent_list=[bc_agent],
            num_episodes=[num_games],
            enemy_iterations=0,
            temperature_list=[1],
            own_temperature=1,
            prevent_draw=False,
            switch_positions=True,
            verbose_level=1,
        )
        full_result_list.append(results)
        with open(full_save_path, 'wb') as f:
            pickle.dump(full_result_list, f)

def plot_albfix_bs_bc():
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc_bc_fix'
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc'
    base_name = 'albfix_bc_temps'
    
    save_path_strength = Path(__file__).parent.parent.parent / 'a_data' / 'oc_strength'
    base_name_strength = 'bc_strength'
    
    temperatures = np.linspace(0, 10, 50)
    
    name_dict = {
        'cr': 'Cramped Rm.',
        'aa': 'Asym. Adv.',
        'co': 'Coord. Ring',
        'fc': 'Forced Coord.',
        'cc': 'Counter Circ.',
    }
    
    for idx, prefix in enumerate(name_dict.keys()):
        # results
        seed_list = []
        for seed in range(5):
            full_save_path = save_path / f'{base_name}_{prefix}_{seed}.pkl'
            with open(full_save_path, 'rb') as f:
                cur_data = pickle.load(f)
            seed_list.append(cur_data)
        full_arr = np.asarray(seed_list)[:, :, 0, :].mean(axis=-1)
        
        # temperature estimation
        estimation_list = []
        for seed in range(5):
            seed_list = []
            cur_log_path = save_path_strength / f'{base_name_strength}_log_{prefix}_{seed}.pkl'
            with open(cur_log_path, 'rb') as f:
                cur_log = pickle.load(f)
            game_dicts = cur_log['temp_estimates']
            for idx, game_dict in enumerate(game_dicts):
                enemy_idx = (idx + 1) % 2
                seed_list.append(game_dict[enemy_idx])
            estimation_list.append(seed_list)
        temp_arr = np.asarray(estimation_list).reshape(-1, 399)
        true_temp = temp_arr[:, -1].mean()
        
        plt.clf()
        plt.figure(figsize=(5, 5))
        seaborn.set_theme(style='whitegrid')
        
        plot_filled_std_curves(
            x=temperatures,
            mean=full_arr.mean(axis=0),
            std=full_arr.std(axis=0),
            color=COLORS[0],
            lighter_color=LIGHT_COLORS[0],
            linestyle=LINESTYLES[0],
            label='Alb. + BC',
            min_val=0,
        )
        plt.axvline(x=true_temp, color='xkcd:red', linestyle='solid', label='True Temp.')
        
        fontsize = 'x-large'
        plt.xlim(temperatures[0], temperatures[-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.title(name_dict[prefix], fontsize=fontsize)
        # plt.legend(fontsize='x-large', loc='lower right', bbox_to_anchor=(1.01, -0.01))
        plt.legend(fontsize='x-large')
        plt.ylabel('Reward', fontsize=fontsize)
        plt.xlabel('Fixed Temperature Input', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'legend_{base_name}_{prefix}.pdf', bbox_inches='tight', pad_inches=0.0)

if __name__ == '__main__':
    # eval_albfix_vs_bc(0)
    plot_albfix_bs_bc()
