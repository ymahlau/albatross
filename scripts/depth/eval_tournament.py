from datetime import datetime
import itertools
import math
from pathlib import Path
import pickle
import random
import numpy as np
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


def play_single_game(
    game: Game,
    agent_list: list[Agent],
    iterations: list[int],
    temperatures: list[float],
    prevent_draw: bool,
    verbose_level: int = 0,
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
    num_parts = 20
    
    depths = np.asarray(list(range(200, 2001, 200)), dtype=int)
    depth_dict = {
        x: d for x, d in enumerate(depths)
    }
    depth_dict[len(depths)] = 1  # albatross
    depth_dict[len(depths) + 1] = 1  # alphaZero
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth'
    base_name = 'trnmt_small'
    prefix = '4nd7'
    num_games_per_part = 100
    
    
    game_dict = {
        '4nd7': survive_on_7x7_4_player(),
        'd7': survive_on_7x7_constrictor(),
        'nd7': survive_on_7x7(),
        '4d7': survive_on_7x7_constrictor_4_player(),
    }
    
    pref_lists = [
        list(range(num_seeds)),
        list(range(int(num_parts)))
    ]
    prod = list(itertools.product(*pref_lists))
    seed, cur_part = prod[experiment_id]
    num_agents = 4 if prefix.startswith("4") else 2
    
    set_seed((seed + 1) * cur_part)  
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
        sampled_indices = random.sample(range(len(depth_dict)), 4)
        cur_agent_list = [agent_dict[idx] for idx in sampled_indices]
        cur_iterations = [depth_dict[idx] for idx in sampled_indices]
        
        cur_result, cur_length = play_single_game(
            game=game,
            agent_list=cur_agent_list,
            iterations=cur_iterations,
            temperatures=[math.inf for _ in range(num_agents)],
            prevent_draw=False,
            verbose_level=0,
        )
        result_list.append((sampled_indices, cur_result, cur_length))
        
        full_save_path = save_path / f'{base_name}_{prefix}_{seed}_{cur_part}.pkl'
        with open(full_save_path, 'wb') as f:
            pickle.dump(result_list, f)
        print(f"{datetime.now()} - {game_idx}: {sampled_indices} - {cur_result}, {cur_length}")
    




if __name__ == '__main__':
    play_tournament(0)
