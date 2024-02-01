from datetime import datetime
import itertools
from pathlib import Path
import pickle
from matplotlib import pyplot as plt

import numpy as np
import seaborn
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.one_shot import NetworkAgentConfig, bc_agent_from_file

from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig, OvercookedRewardConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves
from src.misc.utils import set_seed
from src.modelling.mle import compute_temperature_mle
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def estimate_bc_strength(experiment_id: int):
    num_games = 100
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc_strength'
    base_name = 'bc_strength'
    
    game_dicts = {
        'aa': AsymmetricAdvantageOvercookedConfig(),
        'cc': CounterCircuitOvercookedConfig(),
        'co': CoordinationRingOvercookedConfig(),
        'cr': CrampedRoomOvercookedConfig(),
        'fc': ForcedCoordinationOvercookedConfig(),
    }
    
    pref_lists = [
        ['aa', 'cc', 'co', 'cr', 'fc'],
        # ['cr'],
        list(range(5)),
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed = prod[experiment_id]
    set_seed(seed)
    game_cfg = game_dicts[prefix]

    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'overcooked'
    proxy_path = net_path / f'proxy_{prefix}_{seed}' / 'latest.pt'
    resp_path = net_path / f'resp_{prefix}_{seed}' / 'latest.pt'

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
        # fixed_temperatures=[0.1, 0.1],
        num_samples=1,
        init_temp=5,
        # num_likelihood_bins=int(2e3),
        # sample_from_likelihood=True,
    )
    alb_online_agent = AlbatrossAgent(alb_online_agent_cfg)
    
    reward_cfg = OvercookedRewardConfig(
        placement_in_pot=0,
        dish_pickup=0,
        soup_pickup=0,
        soup_delivery=20,
        start_cooking=0,
    )
    game_cfg.reward_cfg = reward_cfg
    game_cfg.temperature_input = True
    game_cfg.single_temperature_input = True
    game = OvercookedGame(game_cfg)
    
    bc_path = Path(__file__).parent.parent.parent / 'bc_state_dicts'/ f'{prefix}_{seed}.pkl'
    bc_agent = bc_agent_from_file(bc_path)
    
    print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}', flush=True)
    cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}.pkl'
    alb_online_agent.cfg.estimate_log_path = str(cur_log_path)
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
    with open(save_path / f'{base_name}_{prefix}_{seed}.pkl', 'wb') as f:
        pickle.dump(results, f)


def plot_bc_estimate_per_turn():
    name_dict = {
        'cr': 'Cramped Rm.',
        'aa': 'Asym. Adv.',
        'co': 'Coord. Ring',
        'fc': 'Forced Coord.',
        'cc': 'Counter Circ.',
    }
    num_seeds = 5
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc_strength'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc_strength'
    base_name = 'bc_strength'
    
    
    for idx, (prefix, full_name) in enumerate(name_dict.items()):
        estimation_list = []
        for seed in range(num_seeds):
            seed_list = []
            cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}.pkl'
            with open(cur_log_path, 'rb') as f:
                cur_log = pickle.load(f)
            game_dicts = cur_log['temp_estimates']
            for idx, game_dict in enumerate(game_dicts):
                enemy_idx = (idx + 1) % 2
                seed_list.append(game_dict[enemy_idx])
            estimation_list.append(seed_list)
        full_arr = np.asarray(estimation_list).reshape(-1, 399)
        
        plt.clf()
        plt.figure(figsize=(5, 4))
        seaborn.set_theme(style='whitegrid')
        
        x = np.arange(1, 400)
        plot_filled_std_curves(
            x=x,
            mean=full_arr.mean(axis=0),
            std=full_arr.std(axis=0),
            color=COLORS[0],
            lighter_color=LIGHT_COLORS[0],
            linestyle='solid',
            label='Maximum Likelihood\nEstimation',
            min_val=0,
        )
        
        fontsize = 'xx-large'
        plt.xlim(x[0], x[-1])
        plt.ylim(0, 4)
        plt.xticks([50, 150, 250, 350], fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel('Temp. Estimate', fontsize=fontsize)
        plt.xlabel('Episode Step', fontsize=fontsize)
        # if prefix == 'cr':
        # plt.legend(fontsize='x-large')
        # plt.title(full_name, fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'{base_name}_{prefix}.pdf', bbox_inches='tight', pad_inches=0.03)
        

def compute_bc_strength():
    game_dicts = {
        'aa': AsymmetricAdvantageOvercookedConfig(),
        'cc': CounterCircuitOvercookedConfig(),
        'co': CoordinationRingOvercookedConfig(),
        'cr': CrampedRoomOvercookedConfig(),
        'fc': ForcedCoordinationOvercookedConfig(),
    }
    num_seeds = 5
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc_strength'
    base_name = 'bc_strength'
    num_games = 100
    
    result_dict = {}
    
    for prefix in game_dicts.keys():
        action_list, utility_list = [], []
        for seed in range(num_seeds):
            full_save_path = save_path / f'{base_name}_log_{prefix}_{seed}.pkl'
            with open(full_save_path, 'rb') as f:
                cur_data = pickle.load(f)
            for game_id in range(num_games):
                enemy_pos = (game_id + 1) % 2
                cur_utils = cur_data['enemy_util'][game_id][enemy_pos]
                cur_actions = cur_data['enemy_actions'][game_id][enemy_pos]
                action_list.extend(cur_actions)
                utility_list.extend(cur_utils[:len(cur_actions)])
        result_dict[prefix] = compute_temperature_mle(
            min_temp=0,
            max_temp=10,
            num_iterations=40,
            chosen_actions=action_list,
            utilities=utility_list,
            use_line_search=True,
        )
    print(result_dict)
    with open(save_path / 'bc_temp_estimates.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
        
        
if __name__ == '__main__':
    # estimate_bc_strength(0)
    plot_bc_estimate_per_turn()
    # compute_bc_strength()
