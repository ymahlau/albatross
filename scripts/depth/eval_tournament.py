from datetime import datetime
from matplotlib.colors import LogNorm
import textalloc
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
from src.modelling.mle import compute_temperature_mle
from src.network.initialization import get_network_from_file
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES


def play_single_game(
    game: Game,
    agent_list: list[Agent],
    iterations: list[int],
    temperatures: list[float],
    prevent_draw: bool,
    verbose_level: int = 0
):
    game.reset()
    for agent in agent_list:
        agent.reset_episode()
    step_counter = 0
    while not game.is_terminal():
        joint_action_list: list[int] = []
        for player in game.players_at_turn():
            probs, _ = agent_list[player](game, player=player, iterations=iterations[player])
            probs[game.illegal_actions(player)] = 0
            probs /= probs.sum()
            if verbose_level >= 2:
                print(probs, flush=True)
            action = sample_individual_actions(probs[np.newaxis, ...], temperatures[player])[0]
            joint_action_list.append(action)
        if prevent_draw:
            step_with_draw_prevention(game, tuple(joint_action_list))
        else:
            game.step(tuple(joint_action_list))
        if verbose_level >= 2:
            print(joint_action_list, flush=True)
            game.render()
            print('#########################', flush=True)
        step_counter += 1
    # add rewards of player 0 to sum
    cum_rewards = game.get_cum_rewards()
    if verbose_level >= 1:
        print(f"{datetime.now()}: {cum_rewards}", flush=True)
    game.reset()
    for agent in agent_list:
        agent.reset_episode()
    return cum_rewards, step_counter



def play_tournament(experiment_id: int):
    num_seeds = 5
    num_parts = 10
    
    # depths = np.asarray(list(range(50, 2001, 50)), dtype=int)
    depths = list(range(100, 1001, 100)) + list(range(2000, 20001, 1000)) + [int(5e4), int(1e5)]
    depth_dict = {
        x: d for x, d in enumerate(depths)
    }
    depth_dict[len(depths)] = 1  # albatross
    depth_dict[len(depths) + 1] = 1  # alphaZero
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'trnmt'
    base_name = 'trnmt_large'
    num_games_per_part = 500
    
    
    game_dict = {
        '4nd7': survive_on_7x7_4_player(),
        # 'd7': survive_on_7x7_constrictor(),
        # 'nd7': survive_on_7x7(),
        # '4d7': survive_on_7x7_constrictor_4_player(),
    }
    
    pref_lists = [
        list(game_dict.keys()),
        list(range(num_seeds)),
        list(range(int(num_parts)))
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed, cur_part = prod[experiment_id]
    full_save_path = save_path / f'{base_name}_{prefix}_{seed}_{cur_part}.pkl'
    num_agents = 4 if prefix.startswith("4") else 2
    print(f"{datetime.now()} - Started {prefix} with {seed=}, {cur_part=}", flush=True)
    print(f"{full_save_path=}", flush=True)
    
    set_seed(cur_part + seed * num_parts)   
    game_cfg = game_dict[prefix]
    
    net_path = Path(__file__).parent.parent.parent.parent.parent / 'a_saved_runs' / 'battlesnake'
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
    
    cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}_{cur_part}.pkl'
    alb_online_agent.cfg.estimate_log_path = str(cur_log_path)
    
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
    
    result_list = []
    for game_idx in range(num_games_per_part):
        # sample agents without replacement
        sampled_indices = random.sample(range(len(depth_dict)), num_agents)
        cur_agent_list = [agent_dict[idx] for idx in sampled_indices]
        cur_iterations = [depth_dict[idx] for idx in sampled_indices]
        cur_temperatures = [math.inf if idx in [len(depths), len(depths) + 1] else 1 for idx in sampled_indices]
        
        cur_result, cur_length = play_single_game(
            game=game,
            agent_list=cur_agent_list,
            iterations=cur_iterations,
            temperatures=cur_temperatures,
            prevent_draw=False,
            verbose_level=0,
        )
        result_list.append((sampled_indices, cur_result, cur_length))
        
        with open(full_save_path, 'wb') as f:
            pickle.dump(result_list, f)
        print(f"{datetime.now()} - {game_idx}/{num_games_per_part}: {sampled_indices} - {cur_result}, {cur_length}", flush=True)
    


def plot_avg_outcomes():
    # depths = np.asarray(list(range(200, 2001, 200)), dtype=int)
    depths = list(range(100, 1001, 100)) + list(range(2000, 20001, 1000)) + [int(5e4), int(1e5)]
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'trnmt'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'trnmt'
    base_name = 'trnmt_small'
    prefix = '4nd7'
    num_seeds = 5
    num_parts = 10
    
    name_dict = {
        idx: f'd{x}' for idx, x in enumerate(depths)
    }
    name_dict[10] = 'Alb.'
    name_dict[11] = 'AlphaZero'
    labels = list(name_dict.values())
    
    result_dict = {
        x: [[] for _ in range(num_seeds)] for x in name_dict.keys()
    }
    
    for seed in range(num_seeds):
        for part in range(num_parts):
            full_save_path = save_path / f'{base_name}_{prefix}_{seed}_{part}.pkl'
            with open(full_save_path, 'rb') as f:
                cur_list = pickle.load(f)
            for tpl in cur_list:
                for idx, res in zip(tpl[0], tpl[1]):
                    result_dict[idx][seed].append(res)
    data_arr = np.asarray([
        [
            np.asarray(seed_vals).mean() for seed_vals in agent_vals
        ]
        for agent_vals in result_dict.values()
    ])
    mean_vals = data_arr.mean(axis=-1)
    std_vals = data_arr.std(axis=-1)
    
    plt.clf()
    plt.figure(dpi=600)
    # fig, ax = plt.subplots()
    seaborn.set_theme(style='whitegrid')

    ind = np.arange(len(name_dict))
    plt.bar(ind, mean_vals, yerr=std_vals, align='center', alpha=0.7, ecolor='black', capsize=10)

    fontsize = 'medium'
    plt.xlabel('Agents', fontsize=fontsize)
    plt.ylabel('Avg. Outcome', fontsize=fontsize)
    # plt.xlim(depths[0], depths[-1])
    # plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.gca().set_xticks(ind)
    plt.gca().set_xticklabels(labels)
    # plt.legend(fontsize='x-large')
    plt.xticks(fontsize='xx-small')
    plt.tight_layout()
    # plt.savefig(img_path / f'inf_100g_{abbrev}_depths.png')
    plt.savefig(img_path / f'{base_name}_mean_vals.pdf', bbox_inches='tight', pad_inches=0.0)
    

def plot_trueskill_ratings():
    # depths = np.asarray(list(range(200, 2001, 200)), dtype=int)
    # depths = np.asarray(list(range(50, 2001, 50)), dtype=int)
    depths = list(range(100, 1001, 100)) + list(range(2000, 20001, 1000)) + [int(5e4), int(1e5)]
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'trnmt_large'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'trnmt'
    base_name = 'trnmt_large'
    # prefixes = ['4nd7', '4d7', 'nd7', 'd7']
    prefixes = ['4nd7']
    num_seeds = 5
    num_parts = 10
    
    name_dict = {
        idx: f'd{x}' for idx, x in enumerate(depths)
    }
    name_dict[len(depths)] = 'Alb.'
    name_dict[len(depths) + 1] = 'AlphaZero'
    labels = list(name_dict.values())
    
    for prefix in prefixes:
        result_list = []
        for seed in range(num_seeds):
            rating_dict = {
                x: Rating() for x in name_dict.keys()
            }
            for part in range(num_parts):
                full_save_path = save_path / f'{base_name}_{prefix}_{seed}_{part}.pkl'
                with open(full_save_path, 'rb') as f:
                    cur_list = pickle.load(f)
                for tpl in cur_list:
                    ranks = (scipy.stats.rankdata(-tpl[1], method='dense') - 1).tolist()
                    rating_groups = [(rating_dict[idx],) for idx in tpl[0]]
                    new_rating_groups = trueskill.rate(rating_groups, ranks=ranks)
                    for idx, new_r in zip(tpl[0], new_rating_groups):
                        rating_dict[idx] = new_r[0]
            seed_means = [r.mu for r in rating_dict.values()]
            result_list.append(seed_means)
        
        data_arr = np.asarray(result_list)
        mean_vals = data_arr.mean(axis=0)
        std_vals = data_arr.std(axis=0)
        
        plt.clf()
        plt.figure(dpi=600)
        # fig, ax = plt.subplots()
        seaborn.set_theme(style='whitegrid')
        # plt.bar(ind, mean_vals, yerr=std_vals, align='center', alpha=0.7, ecolor='black', capsize=10)
        plt.boxplot(data_arr, labels=labels, showfliers=True)
        
        fontsize = 'medium'
        plt.xlabel('Agents', fontsize=fontsize)
        plt.ylabel('TrueSkill Score', fontsize=fontsize)
        # plt.xlim(depths[0], depths[-1])
        # plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # ind = np.arange(len(name_dict))
        # plt.gca().set_xticks(ind)
        # plt.gca().set_xticklabels(labels)
        # plt.legend(fontsize='x-large')
        plt.xticks(fontsize='xx-small')
        plt.title(prefix, fontsize=fontsize)
        plt.tight_layout()
        # plt.savefig(img_path / f'inf_100g_{abbrev}_depths.png')
        plt.savefig(img_path / f'{base_name}_{prefix}_boxplot.pdf', bbox_inches='tight', pad_inches=0.0)

def compute_temperatures_from_logs():
    # depths = list(range(100, 1001, 100)) + list(range(2000, 20001, 1000)) + [int(5e4), int(1e5)]
    log_path = Path(__file__).parent.parent.parent / 'a_data' / 'trnmt_large'
    base_name = 'trnmt_large'
    prefix = '4nd7'
    num_seeds = 5
    num_parts = 10
    
    # depths = np.asarray(list(range(50, 2001, 50)), dtype=int)
    depths = list(range(100, 1001, 100)) + list(range(2000, 20001, 1000)) + [int(5e4), int(1e5)]
    
    alb_idx = len(depths)
    az_idx = len(depths) + 1
    
    utility_dict = {idx: [] for idx in range(len(depths))}
    action_dict = {idx: [] for idx in range(len(depths))}
    utility_dict[len(depths) + 1] = []
    action_dict[len(depths) + 1] = []
    for seed in range(num_seeds):
        for part in range(num_parts):
            full_save_path = log_path / f'{base_name}_{prefix}_{seed}_{part}.pkl'
            with open(full_save_path, 'rb') as f:
                cur_results = pickle.load(f)
            full_save_path_log = log_path / f'{base_name}_log_{prefix}_{seed}_{part}.pkl'
            with open(full_save_path_log, 'rb') as f:
                cur_log = pickle.load(f)
            log_idx = 0
            for result_idx in range(len(cur_results)):
                cur_contestants = cur_results[result_idx][0]
                if alb_idx in cur_contestants:
                    for p_idx, p in enumerate(cur_contestants):
                        if p != alb_idx:
                            # sanity check
                            assert len(cur_log['temp_estimates'][log_idx][p_idx]) <= cur_results[result_idx][2] - 1
                            cur_actions = cur_log['enemy_actions'][log_idx][p_idx]
                            cur_utils = cur_log['enemy_util'][log_idx][p_idx]
                            action_dict[p].extend(cur_actions)
                            utility_dict[p].extend(cur_utils[:len(cur_actions)])
                    log_idx += 1
    result_dict = {}
    # do mle with all observations
    for idx, depth in enumerate(depths):
        result_dict[depth] = compute_temperature_mle(
            min_temp=0,
            max_temp=10,
            num_iterations=20,
            chosen_actions=action_dict[idx],
            utilities=utility_dict[idx],
            use_line_search=True,
        )
    result_dict['AlphaZero'] = compute_temperature_mle(
        min_temp=0,
        max_temp=10,
        num_iterations=20,
        chosen_actions=action_dict[az_idx],
        utilities=utility_dict[az_idx],
        use_line_search=True,
    )
    with open(log_path / 'temp_estimates.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
    
def compute_trueskill_ratings():
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'trnmt_large'
    base_name = 'trnmt_large'
    num_seeds = 5
    num_parts = 10
    
    # depths = np.asarray(list(range(50, 2001, 50)), dtype=int)
    depths = list(range(100, 1001, 100)) + list(range(2000, 20001, 1000)) + [int(5e4), int(1e5)]
    prefix = '4nd7'
    
    name_dict = {
        idx: f'd{x}' for idx, x in enumerate(depths)
    }
    name_dict[len(depths)] = 'Alb.'
    name_dict[len(depths) + 1] = 'AlphaZero'
        
    result_list = []
    for seed in range(num_seeds):
        rating_dict = {
            x: Rating() for x in name_dict.keys()
        }
        for part in range(num_parts):
            full_save_path = save_path / f'{base_name}_{prefix}_{seed}_{part}.pkl'
            with open(full_save_path, 'rb') as f:
                cur_list = pickle.load(f)
            for tpl in cur_list:
                ranks = (scipy.stats.rankdata(-tpl[1], method='dense') - 1).tolist()
                rating_groups = [(rating_dict[idx],) for idx in tpl[0]]
                new_rating_groups = trueskill.rate(rating_groups, ranks=ranks)
                for idx, new_r in zip(tpl[0], new_rating_groups):
                    rating_dict[idx] = new_r[0]
        seed_means = [r.mu for r in rating_dict.values()]
        result_list.append(seed_means)
    data_arr = np.asarray(result_list)
    with open(save_path / 'ratings.pkl', 'wb') as f:
        pickle.dump(data_arr, f)


def plot_trueskill_vs_temperature():
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'trnmt'
    base_name = 'trnmt_large'
    
    # depths = np.asarray(list(range(50, 2001, 50)), dtype=int)
    depths = list(range(100, 1001, 100)) + list(range(2000, 20001, 1000)) + [int(5e4), int(1e5)]
    prefix = '4nd7'
    
    name_dict = {
        idx: f'd{x}' for idx, x in enumerate(depths)
    }
    name_dict[len(depths)] = 'Alb.'
    name_dict[len(depths) + 1] = 'AlphaZero'
    
    temp_path = Path(__file__).parent.parent.parent / 'a_data' / 'trnmt_large' / 'temp_estimates.pkl'
    with open(temp_path, 'rb') as f:
        temps = pickle.load(f)
    
    data_path = Path(__file__).parent.parent.parent / 'a_data' / 'trnmt_large' / 'ratings.pkl'
    with open(data_path, 'rb') as f:
        data_arr = pickle.load(f)
    
    mean_vals = data_arr.mean(axis=0)
    
    az_temp = temps['AlphaZero']
    
    x_range = [3.5, 10.2]
    y_range = [23, 32]
    x = [temps[d] for d in depths]
    y = mean_vals[:-2]
    
    plt.clf()
    fig = plt.figure()
    # fig, ax = plt.subplots()
    seaborn.set_theme(style='whitegrid')
    
    cmap = seaborn.color_palette("coolwarm_r", as_cmap=True)
    # palette = seaborn.color_palette("coolwarm", 100)
    
    plt.scatter([10], [mean_vals[-2]], color=COLORS[1], marker='*', label='Albatross (ours)', s=130, edgecolors=(0, 0, 0, 1))
    plt.scatter(az_temp, mean_vals[-1], color=COLORS[2], label='AlphaZero', marker='v', s=90, edgecolors=(0, 0, 0, 1), clip_on=False, zorder=10)
    
    scatter = plt.scatter(x, y, c=depths, cmap=cmap, norm=LogNorm(), label='MCTS + Value Heuristic', s=75, alpha=0.9, edgecolors=(0, 0, 0, 1))
    plt.colorbar(scatter, label='MCTS Iterations')
    
    # plt.plot(x_range, [mean_vals[-2], mean_vals[-2]], color=COLORS[1], linestyle='dashed', label='Albatross (ours)')
    
    
    fontsize = 'medium'
    plt.xlabel('Temperature (Estimated Rationality)', fontsize=fontsize)
    plt.ylabel('TrueSkill Score', fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.legend(fontsize=fontsize, loc='center', bbox_to_anchor=(0.4, 0.75))
    
    annotated = [100, 5000, int(5e4), int(1e5)]
    annotated_indices = [depths.index(d) for d in annotated]
    text_list = [f'{d}' for d in annotated]
    x = [temps[d] for d in annotated]
    y = [mean_vals[d_idx] for d_idx in annotated_indices]
    # textalloc.allocate_text(
    #     fig,
    #     plt.gca(),
    #     x,
    #     y,
    #     text_list,
    #     x_scatter=x,
    #     y_scatter=y,
    #     textsize=15,
    #     margin=0.1,
    #     linecolor='black',
    #     seed=0,
    #     direction='north',
    # )
    
    plt.tight_layout()
    plt.savefig(img_path / f'{base_name}_{prefix}_scatter.pdf', bbox_inches='tight', pad_inches=0.02)

if __name__ == '__main__':
    # play_tournament(0)
    # plot_avg_outcomes()
    # plot_trueskill_ratings()
    # compute_trueskill_ratings()
    # compute_temperatures_from_logs()
    plot_trueskill_vs_temperature()
