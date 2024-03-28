from datetime import datetime
import itertools
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn

from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig
from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig, OvercookedRewardConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.misc.const import LINESTYLES
from src.misc.plotting import plot_filled_std_curves
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def temp_convergence_run(experiment_id: int):
    num_games_per_part = 20
    num_parts = 5
    temperatures = np.arange(0, 10.1, 1)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'temp_conv'
    base_name = 'temp_conv'
    
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
        list(range(num_parts)),
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed, cur_game_id = prod[experiment_id]
    set_seed(cur_game_id + seed * num_parts)
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
        fixed_temperatures=None,
        num_samples=1,
        init_temp=0,
        # num_likelihood_bins=int(2e3),
        sample_from_likelihood=False,
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
    for t_idx, t in enumerate(temperatures):
        print(f'Started evaluation with: {t_idx=}, {t=}', flush=True)
        proxy_net_agent.reset_episode()
        alb_online_agent.reset_episode()
        proxy_net_agent.set_temperatures([t, t])
        
        cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}_{cur_game_id}_{t_idx}.pkl'
        alb_online_agent.cfg.estimate_log_path = str(cur_log_path)
        
        results, _ = do_evaluation(
            game=game,
            evaluee=alb_online_agent,
            opponent_list=[proxy_net_agent],
            num_episodes=[num_games_per_part],
            enemy_iterations=0,
            temperature_list=[1],
            own_temperature=1,
            prevent_draw=False,
            switch_positions=True,
            verbose_level=1,
        )
        full_result_list.append(results)
        with open(save_path / f'{base_name}_{prefix}_{seed}.pkl', 'wb') as f:
            pickle.dump(full_result_list, f)


def plot_temp_convergence():
    num_parts = 5
    temperatures = np.arange(0, 10.1, 1)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'temp_conv'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'temp_conv'
    base_name = 'temp_conv'
    
    game_dicts = {
        'aa': AsymmetricAdvantageOvercookedConfig(),
        # 'cc': CounterCircuitOvercookedConfig(),
        # 'co': CoordinationRingOvercookedConfig(),
        # 'cr': CrampedRoomOvercookedConfig(),
        # 'fc': ForcedCoordinationOvercookedConfig(),
    }
    for prefix in game_dicts.keys():
        temp_arr_list, action_list, utility_list = [], [], []
        for seed in range(5):
            seed_list = []
            # for t_idx, _ in enumerate(temperatures):
            for t_idx in [10]:
                t_list = []
                for part in range(num_parts):
                    cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}_{part}_{t_idx}.pkl'
                    with open(cur_log_path, 'rb') as f:
                        cur_dict = pickle.load(f)
                    part_list = []
                    for game_dict in cur_dict['temp_estimates']:
                        if game_dict[0]:
                            part_list.append(game_dict[0])
                        else:
                            part_list.append(game_dict[1])
                    t_list.append(part_list)
                seed_list.append(t_list)
            temp_arr_list.append(seed_list)
        full_arr = np.asarray(temp_arr_list)
        
        for t_idx, t in enumerate(temperatures):
            cur_arr = full_arr[:, t_idx]
            cur_arr = cur_arr.reshape(-1, 399)
            x = np.arange(1, 400)
            
            plt.clf()
            plt.figure(figsize=(4, 4))
            seaborn.set_theme(style='whitegrid')
            
            plot_filled_std_curves(
                x=x,
                mean=cur_arr.mean(axis=0),
                std=cur_arr.std(axis=0),
                color='xkcd:almost black',
                lighter_color='xkcd:dark grey',
                linestyle=LINESTYLES[0],
                label=None,
                min_val=0,
            )
            
            fontsize = 'xx-large'
            plt.xlim(x[0], x[-1])
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            # plt.title(full_name, fontsize=fontsize)
            # if idx == 0:
                # plt.legend(fontsize='x-large', loc='lower right', bbox_to_anchor=(1.01, -0.01))
            plt.ylabel('Temp. Estimate', fontsize=fontsize)
            plt.xlabel('Episode Step', fontsize=fontsize)
            plt.tight_layout()
            plt.savefig(img_path / f'temp_conv_{prefix}_{t_idx}.pdf', bbox_inches='tight', pad_inches=0.0)
        


if __name__ == '__main__':
    # temp_convergence_run(0)
    plot_temp_convergence()
