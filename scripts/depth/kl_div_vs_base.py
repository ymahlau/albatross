
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
from src.agent import Agent
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgentConfig
from src.agent.search_agent import AreaControlSearchAgentConfig
from src.game.actions import sample_individual_actions
from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7_4_player, survive_on_7x7_constrictor, survive_on_7x7_constrictor_4_player
from src.game.game import Game
from src.game.initialization import get_game_from_config
from src.game.utils import step_with_draw_prevention
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.misc.plotting import plot_filled_std_curves
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES

def kl_evaluation(
        game: Game,
        evaluee: Agent,
        opponent: Agent,
        num_episodes: int,
        enemy_iterations: int,
        temperature: float,
        prevent_draw: bool,
        switch_positions: bool,
        verbose_level: int = 0,
        own_temperature: float = math.inf,
        own_iterations: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    results_player, results_opponent = [], []
    # iterate episodes
    for ep in range(num_episodes):
        agent_pos = 0
        if switch_positions:
            agent_pos = ep % 2
        game.reset()
        evaluee.reset_episode()
        opponent.reset_episode()
        step_counter = 0
        while not game.is_terminal() and game.is_player_at_turn(0):
            joint_action_list: list[int] = []
            player_prob, opp_prob = None, None
            for player in game.players_at_turn():
                if player == agent_pos:  # agent to evaluate always plays as player 0
                    probs, info = evaluee(game, player=player, iterations=own_iterations)
                    probs[game.illegal_actions(player)] = 0
                    probs /= probs.sum()
                    player_prob = info['all_action_probs']
                    results_player.append(player_prob)
                    if verbose_level >= 2:
                        print(probs, flush=True)
                    action = sample_individual_actions(probs[np.newaxis, ...], own_temperature)[0]
                else:
                    probs, info = opponent(game, player=player, iterations=enemy_iterations)
                    probs[game.illegal_actions(player)] = 0
                    probs /= probs.sum()
                    opp_prob = info['all_action_probs']
                    results_opponent.append(opp_prob)
                    # print(probs)
                    action = sample_individual_actions(probs[np.newaxis, ...], temperature)[0]
                joint_action_list.append(action)
            assert player_prob is not None and opp_prob is not None
            if prevent_draw:
                step_with_draw_prevention(game, tuple(joint_action_list))
            else:
                game.step(tuple(joint_action_list))
            if verbose_level >= 2:
                print(joint_action_list, flush=True)
                game.render()
                print('#########################', flush=True)
            step_counter += 1
        cum_rewards = game.get_cum_rewards()
        if verbose_level >= 1:
            print(f"{datetime.now()} - {ep}: {cum_rewards}", flush=True)
    return results_player, results_opponent

def save_policies_at_depth(experiment_id: int):
    num_games_per_part = 5
    num_parts = 10
    search_iterations = np.arange(50, 2001, 50)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'kl_depth'
    base_name = 'az_base_kl'
    
    game_dict = {
        'd7': survive_on_7x7_constrictor(),
        # 'nd7': survive_on_7x7(),
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
    set_seed(cur_game_id + seed * num_parts)
    game_cfg = game_dict[prefix]

    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'battlesnake'
    resp_path = net_path / f'{prefix}_resp_{seed}' / 'latest.pt'
    proxy_path = net_path / f'{prefix}_proxy_{seed}' / 'latest.pt'
    az_path = net_path / f'{prefix}_{seed}' / 'latest.pt'
    
    # net = get_network_from_file(resp_path).eval()
    # alb_network_agent_cfg = NetworkAgentConfig(
    #     net_cfg=net.cfg,
    #     temperature_input=True,
    #     single_temperature=False,
    #     init_temperatures=[0, 0, 0, 0] if prefix.startswith('4') else [0, 0],
    # )
    # alb_online_agent_cfg = AlbatrossAgentConfig(
    #     num_player=4 if prefix.startswith('4') else 2,
    #     agent_cfg=alb_network_agent_cfg,
    #     device_str='cpu',
    #     response_net_path=str(resp_path),
    #     proxy_net_path=str(proxy_path),
    #     noise_std=2,
    #     num_samples=10,
    #     init_temp=5,
    # )
    # alb_online_agent = AlbatrossAgent(alb_online_agent_cfg)
    
    net = get_network_from_file(az_path).eval()
    az_cfg = NetworkAgentConfig(net_cfg=net.cfg)
    az_agent = get_agent_from_config(az_cfg)
    az_agent.replace_net(net)
    
    base_agent_cfg = AreaControlSearchAgentConfig()
    base_agent = get_agent_from_config(base_agent_cfg)
    
    game_cfg.ec.temperature_input = False
    game_cfg.ec.single_temperature_input = False
    game = get_game_from_config(game_cfg)
    
    full_save_path = save_path / f'{base_name}_{prefix}_{seed}_{cur_game_id}.pkl'
    print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}', flush=True)
    list_az, list_base = [], []
    
    for iteration_idx, cur_iterations in enumerate(search_iterations):
        print(f'Started evaluation with: {iteration_idx=}, {cur_iterations=}')
        
        az_pol, base_pol = kl_evaluation(
            game=game,
            evaluee=az_agent,
            opponent=base_agent,
            num_episodes=num_games_per_part,
            enemy_iterations=cur_iterations,
            temperature=math.inf,
            own_temperature=math.inf,
            prevent_draw=True,
            switch_positions=False,
            verbose_level=1,
            own_iterations=1,
        )
        list_az.append(az_pol)
        list_base.append(base_pol)
        
        save_dict = {
            'az_pol': list_az,
            'base_pol': list_base,
        }
        
        with open(full_save_path, 'wb') as f:
            pickle.dump(save_dict, f)

def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def rename_files_containing_string(directory, old_string, new_string):
    for filename in os.listdir(directory):
        if old_string in filename:
            new_filename = filename.replace(old_string, new_string)
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} to {new_filename}")

def compute_kl_div(p: np.ndarray, q: np.ndarray, epsilon: float = 0.001):
    # prefiltering:
    filter_zero_p = np.abs(p) < epsilon
    filter_one_p = np.abs(p - 1) < epsilon
    filter_zero_q = np.abs(q) < epsilon
    filter_one_q = np.abs(q - 1) < epsilon
    
    full_filter = np.logical_or.reduce(
        [filter_zero_p, filter_one_p, filter_zero_q, filter_one_q]
    )
    filter_reduced = full_filter.sum(axis=-1) < 4
    p_reduced = p[filter_reduced]
    q_reduced = q[filter_reduced]
    
    filter_zero_p_reduced = np.abs(p_reduced) < epsilon
    filter_zero_q_reduced = np.abs(q_reduced) < epsilon
    illegal_action_filter = np.logical_or(filter_zero_p_reduced, filter_zero_q_reduced)
    
    log_ratio = np.log(p_reduced / q_reduced)
    log_ratio = np.where(illegal_action_filter, 0, log_ratio)
    weighted_ratio = p_reduced * log_ratio
    kl_divs = np.sum(weighted_ratio, axis=-1)
    return kl_divs


def compute_same_argmax_ratio(p: np.ndarray, q: np.ndarray, epsilon=1e-5):
    filter_zero_p = np.abs(p) < epsilon
    filter_one_p = np.abs(p - 1) < epsilon
    filter_zero_q = np.abs(q) < epsilon
    filter_one_q = np.abs(q - 1) < epsilon
    
    full_filter = np.logical_or.reduce(
        [filter_zero_p, filter_one_p, filter_zero_q, filter_one_q]
    )
    filter_reduced = full_filter.sum(axis=-1) < 4
    p_reduced = p[filter_reduced]
    q_reduced = q[filter_reduced]
    
    p_argmax = np.argmax(p_reduced, axis=-1)
    q_argmax = np.argmax(q_reduced, axis=-1)
    same_argmax = p_argmax == q_argmax
    ratio = np.sum(same_argmax) / same_argmax.shape[0]
    return ratio


def plot_kl_div_at_depth():
    num_games_per_part = 5
    num_parts = 10
    search_iterations = np.arange(50, 2001, 50)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'kl_depth'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'kl_depth'
    plot_kl = False
    base_name = 'az_base_kl'
    # prefixes = ['d7', '4d7', '4nd7']
    prefixes = ['d7']
    
    for prefix in prefixes:
        policy_list_az = [[] for _ in search_iterations]
        policy_list_base = [[] for _ in search_iterations]
        for seed in range(5):
            for part in range(num_parts):
                cur_path = save_path / f'{base_name}_{prefix}_{seed}_{part}.pkl'
                with open(cur_path, 'rb') as f:
                    cur_dict = pickle.load(f)
                for full, new in zip(policy_list_az, cur_dict['az_pol']):
                    full.extend(flatten_list(new))
                for full, new in zip(policy_list_base, cur_dict['base_pol']):
                    full.extend(flatten_list(new))
                # policy_list_az.extend(flatten_list(cur_dict['az_pol']))
                # policy_list_base.extend(flatten_list(cur_dict['base_pol']))
        policy_arrs_az = [np.asarray(x).reshape(-1, 4) for x in policy_list_az]
        policy_arrs_base = [np.asarray(x).reshape(-1, 4) for x in policy_list_base]
        
        if plot_kl:
            kl_divs = []
            for p, q in zip(policy_arrs_base, policy_arrs_az):
                cur_kl = compute_kl_div(p=p, q=q)
                kl_divs.append(cur_kl)
            
            q1 = np.asarray([np.quantile(x, 0.25) for x in kl_divs])
            q2 = np.asarray([np.median(x) for x in kl_divs])
            q3 = np.asarray([np.quantile(x, 0.75) for x in kl_divs])
            
            plt.clf()
            plt.figure(dpi=600)
            seaborn.set_theme(style='whitegrid')
            
            plot_filled_std_curves(
                x=search_iterations,
                mean=q2,
                lower=q1,
                upper=q3,
                color=COLORS[0],
                lighter_color=LIGHT_COLORS[0],
                linestyle=LINESTYLES[0],
                label='q1-3',
            )
            plt.plot(search_iterations, [np.max(x) for x in kl_divs], label='max')
            plt.plot(search_iterations, [np.min(x) for x in kl_divs], label='min')
            plt.plot(search_iterations, [np.mean(x) for x in kl_divs], label='mean')
            
            fontsize = 'medium'
            plt.xlabel('Enemy Search Iterations', fontsize=fontsize)
            plt.ylabel('KL-Divergence', fontsize=fontsize)
            plt.xlim(search_iterations[0], search_iterations[-1])
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.legend(fontsize=fontsize)
            plt.tight_layout()
            plt.savefig(img_path / f'kl_{prefix}.pdf')
        else:
            argmax_ratio = []
            for p, q in zip(policy_arrs_base, policy_arrs_az):
                cur_ratio = compute_same_argmax_ratio(p=p, q=q)
                argmax_ratio.append(cur_ratio)
            ratio_arr = np.asarray(argmax_ratio)
            # smoothed_arr = scipy.signal.savgol_filter(ratio_arr, window_length=10, polyorder=1)
            
            
            plt.clf()
            plt.figure(dpi=600)
            seaborn.set_theme(style='whitegrid')
            
            plt.plot(search_iterations, ratio_arr)
            
            fontsize = 'medium'
            plt.xlabel('Enemy Search Iterations', fontsize=fontsize)
            plt.ylabel('Argmax ratio', fontsize=fontsize)
            plt.xlim(search_iterations[0], search_iterations[-1])
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.tight_layout()
            plt.savefig(img_path / f'argmax_{prefix}.pdf')
            

if __name__ == '__main__':
    # save_policies_at_depth(0)
    plot_kl_div_at_depth()
