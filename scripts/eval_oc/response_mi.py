

from collections import Counter
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn

from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves


def measure_mi():
    num_games_per_part = 20
    num_parts = 5
    num_seeds = 5
    num_actions = 6
    num_episode_steps = 399
    temperatures = np.linspace(0, 10, 20)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc_behavior'
    base_name = 'resp_proxy'
    
    name_dict = {
        'cr': 'Cramped Rm.',
        'aa': 'Asym. Adv.',
        'co': 'Coord. Ring',
        'fc': 'Forced Coord.',
        'cc': 'Counter Circ.',
    }
    
    mi_dict = {}
    for prefix in name_dict.keys():
        action_list_alb, action_list_proxy = [], []
        for seed in range(5):
            for t_idx, _ in enumerate(temperatures):
                for cur_part in range(num_parts):
                    cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}_{cur_part}_{t_idx}.pkl'
                    with open(cur_log_path, 'rb') as f:
                        cur_data = pickle.load(f)
                    for ep_idx, ep_actions in enumerate(cur_data['enemy_actions']):
                        if ep_idx == 0:
                            action_list_alb.append(ep_actions[0])
                            action_list_proxy.append(ep_actions[1])
                        else:
                            action_list_alb.append(ep_actions[1])
                            action_list_proxy.append(ep_actions[0])
        alb_action_arr = np.asarray(action_list_alb).reshape(num_seeds, len(temperatures), num_parts * num_games_per_part, num_episode_steps)
        proxy_action_arr = np.asarray(action_list_proxy).reshape(num_seeds, len(temperatures), num_parts * num_games_per_part, num_episode_steps)
        
        alb_marg_list, joint_list, proxy_marg_list = [], [], []
        for seed in range(num_seeds):
            for t_idx, _ in enumerate(temperatures):
                marg_alb_hist, _ = np.histogram(alb_action_arr[seed, t_idx], bins=num_actions)
                cur_alb_probs = marg_alb_hist / np.sum(marg_alb_hist)
                alb_marg_list.append(cur_alb_probs)
                
                proxy_alb_hist, _ = np.histogram(proxy_action_arr[seed, t_idx], bins=num_actions)
                cur_proxy_probs = proxy_alb_hist / np.sum(proxy_alb_hist)
                proxy_marg_list.append(cur_proxy_probs)
                
                pairs = list(zip(alb_action_arr[seed, t_idx].flatten(), proxy_action_arr[seed, t_idx].flatten()))
                tuple_counter = Counter(pairs)
                cur_joint_probs = np.empty((num_actions, num_actions), dtype=float)
                for a0 in range(num_actions):
                    for a1 in range(num_actions):
                        cur_joint_probs[a0, a1] = tuple_counter[(a0, a1)]
                cur_joint_probs = cur_joint_probs / np.sum(cur_joint_probs)
                joint_list.append(cur_joint_probs)
            
        alb_marg_arr = np.asarray(alb_marg_list).reshape(num_seeds, len(temperatures), num_actions)
        proxy_marg_arr = np.asarray(proxy_marg_list).reshape(num_seeds, len(temperatures), num_actions)
        joint_arr = np.asarray(joint_list).reshape(num_seeds, len(temperatures), num_actions, num_actions)
        
        mi = np.zeros(shape=(num_seeds, len(temperatures),), dtype=float)
        for a0 in range(num_actions):
            for a1 in range(num_actions):
                mi = mi + joint_arr[..., a0, a1] * np.log(joint_arr[..., a0, a1] / (alb_marg_arr[..., a0] * proxy_marg_arr[..., a1]))
        mi_dict[prefix] = mi
    with open(save_path / f'mutual_information.pkl', 'wb') as f:
        pickle.dump(mi_dict, f)


def plot_mi():
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc_behavior'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc_behavior'

    temperatures = np.linspace(0, 10, 20)
    
    name_dict = {
        'cr': 'Cramped Rm.',
        'aa': 'Asym. Adv.',
        'co': 'Coord. Ring',
        'fc': 'Forced Coord.',
        'cc': 'Counter Circ.',
    }
    
    with open(save_path / f'mutual_information.pkl', 'rb') as f:
        mi_dict = pickle.load(f)
    
    for prefix in name_dict:
        plt.clf()
        plt.figure(figsize=(5, 4))
        seaborn.set_theme(style='whitegrid')
        
        plot_filled_std_curves(
            x=temperatures,
            mean=mi_dict[prefix].mean(axis=0),
            std=mi_dict[prefix].std(axis=0),
            color='xkcd:almost black',
            lighter_color='xkcd:dark grey',
            linestyle=LINESTYLES[0],
            label=None,
            min_val=0,
        )
        
        fontsize = 'xx-large'
        plt.xlim(temperatures[0], temperatures[-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.title(name_dict[prefix], fontsize=fontsize)
        # if idx == 0:
            # plt.legend(fontsize='x-large', loc='lower right', bbox_to_anchor=(1.01, -0.01))
        plt.ylabel('Mutual Information', fontsize=fontsize)
        plt.xlabel('Temperature', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'mi_resp_proxy_notitle_{prefix}.pdf', bbox_inches='tight', pad_inches=0.0)
    
    

if __name__ == '__main__':
    # measure_mi()
    plot_mi()
