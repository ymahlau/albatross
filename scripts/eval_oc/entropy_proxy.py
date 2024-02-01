from datetime import datetime
import itertools
import math
import os
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import torch
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig
from src.game.actions import sample_individual_actions

from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7, survive_on_7x7_4_player, survive_on_7x7_constrictor, survive_on_7x7_constrictor_4_player
from src.game.initialization import get_game_from_config
from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig, OvercookedRewardConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves
from src.misc.utils import set_seed, softmax_weighting
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def entropy_proxy_eval(experiment_id: int):
    num_games = 10
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc_behavior'
    base_name = 'proxy_probs'
    
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
    proxy_path = net_path / f'proxy_{prefix}_{seed}' / 'latest.pt'
        
    proxy_net = get_network_from_file(proxy_path)
    proxy_net = proxy_net.eval()
    proxy_net_agent_cfg = NetworkAgentConfig(
        net_cfg=proxy_net.cfg,
        temperature_input=True,
        single_temperature=True,
        init_temperatures=[5, 5],
    )
    proxy_net_agent = NetworkAgent(proxy_net_agent_cfg)
    proxy_net_agent.replace_net(proxy_net)
    
    proxy_net_agent2 = NetworkAgent(proxy_net_agent_cfg)
    proxy_net_agent2.replace_net(proxy_net)
    
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
    temperatures = np.linspace(0, 10, 50)
    all_probs = []
    for game_id in range(num_games):
        agent_pos = game_id % 2
        game.reset()
        proxy_net_agent.reset_episode()
        proxy_net_agent2.reset_episode()
        step_counter = 0
        while not game.is_terminal():
            if game.turns_played % 10 == 0:
                print(f'Step {game.turns_played}', flush=True)
            # do forward pass for different temperatures
            obs_list = []
            for t in temperatures:
                obs, _ , _ = game.get_obs(temperatures=[t, t])
                obs_list.append(obs)
            obs_tensor = torch.tensor(np.asarray(obs_list), dtype=torch.float32)
            obs_tensor = obs_tensor.reshape((2 * len(temperatures), obs_tensor.shape[2], obs_tensor.shape[3], obs_tensor.shape[4]))
            net_out = proxy_net(obs_tensor)
            policy_out = proxy_net.retrieve_policy_tensor(net_out).detach().numpy()
            exp_policy = np.exp(policy_out)
            probs = exp_policy / np.sum(exp_policy, axis=-1)[:, np.newaxis]
            probs = probs.reshape(len(temperatures), 2, game.num_actions)
            probs = np.transpose(probs, (1, 0, 2))
            all_probs.append(probs)
            # step
            joint_action_list: list[int] = []
            for player in game.players_at_turn():
                if player == agent_pos:  # agent to evaluate always plays as player 0
                    probs, _ = proxy_net_agent(game, player=player, iterations=1)
                    probs[game.illegal_actions(player)] = 0
                    probs /= probs.sum()
                    action = sample_individual_actions(probs[np.newaxis, ...], 1)[0]
                else:
                    probs, _ = proxy_net_agent2(game, player=player, iterations=1)
                    probs[game.illegal_actions(player)] = 0
                    probs /= probs.sum()
                    # print(probs)
                    action = sample_individual_actions(probs[np.newaxis, ...], 1)[0]
                joint_action_list.append(action)
            game.step(tuple(joint_action_list))
            step_counter += 1
        # add rewards of player 0 to sum
        cum_rewards = game.get_cum_rewards()
        print(f"{datetime.now()} - {game_id}: {cum_rewards}", flush=True)
        with open(save_path / f'{base_name}_{prefix}_{seed}.pkl', 'wb') as f:
            pickle.dump(all_probs, f)


def plot_entropy_proxy():
    name_dict = {
        'cr': 'Cramped Rm.',
        'aa': 'Asym. Adv.',
        'co': 'Coord. Ring',
        'fc': 'Forced Coord.',
        'cc': 'Counter Circ.',
    }
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc_behavior'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc_behavior'
    base_name = 'proxy_probs'
    temperatures = np.linspace(0, 10, 50)
    num_seeds = 5
    num_games = 10
    num_actions = 6
    
    for idx, (prefix, full_name) in enumerate(name_dict.items()):
        result_list = []
        for seed in range(num_seeds):
            with open(save_path / f'{base_name}_{prefix}_{seed}.pkl', 'rb') as f:
                cur_results = pickle.load(f)
            result_list.append(cur_results)
        result_arr = np.asarray(result_list)
        result_arr = result_arr.reshape(num_seeds, -1, len(temperatures), num_actions)
        entropy_arr = -np.sum(result_arr * np.log2(result_arr), axis=-1)
        entropy_arr = np.mean(entropy_arr, axis=1)
        
        plt.clf()
        plt.figure(figsize=(5, 4))
        seaborn.set_theme(style='whitegrid')
        
        plot_filled_std_curves(
            x=temperatures,
            mean=entropy_arr.mean(axis=0),
            std=entropy_arr.std(axis=0),
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
        # plt.title(full_name, fontsize=fontsize)
        # if idx == 0:
            # plt.legend(fontsize='x-large', loc='lower right', bbox_to_anchor=(1.01, -0.01))
        plt.ylabel('Proxy Policy Entropy', fontsize=fontsize)
        plt.xlabel('Temperature', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'proxy_entropy_notitle_{prefix}.pdf', bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    # entropy_proxy_eval(0)
    plot_entropy_proxy()
