from datetime import datetime
import itertools
import math
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pickle

import seaborn
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig, bc_agent_from_file
from src.agent.scripted import PlaceDishEverywhereAgentConfig, PlaceOnionAgentConfig, PlaceOnionDeliverAgentConfig, PlaceOnionEverywhereAgentConfig
from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, \
    CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig, OvercookedRewardConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def eval_scripted_oc(experiment_id: int):
    print(f'{datetime.now()} - Started eval script', flush=True)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'scripted_sweep_sampling' / 'half_clipped_normal'
    # game_cfg, prefix = AsymmetricAdvantageOvercookedConfig(), 'aa'
    # game_cfg, prefix = CoordinationRingOvercookedConfig(), 'co'
    # game_cfg, prefix = CounterCircuitOvercookedConfig(), 'cc'
    
    # init scripted agent
    # scripted_cfg = PlaceOnionAgentConfig()
    # scripted_cfg = PlaceOnionDeliverAgentConfig()
    # scripted_cfg = PlaceOnionEverywhereAgentConfig()
    # scripted_cfg = PlaceDishEverywhereAgentConfig()
    
    experiment_dict = {
        'aa_put_onions': (
            AsymmetricAdvantageOvercookedConfig(),
            PlaceOnionAgentConfig()
        ),
        'aa_put_onions_delivery': (
            AsymmetricAdvantageOvercookedConfig(),
            PlaceOnionDeliverAgentConfig(),
        ),
        'cc_dish_everywhere': (
            CounterCircuitOvercookedConfig(),
            PlaceDishEverywhereAgentConfig(),
        ),
        'cc_onions_everywhere': (
            CounterCircuitOvercookedConfig(),
            PlaceOnionEverywhereAgentConfig(),
        ),
        'co_dish_everywhere': (
            CoordinationRingOvercookedConfig(),
            PlaceDishEverywhereAgentConfig(),
        ),
        'co_onions_everywhere': (
            CoordinationRingOvercookedConfig(),
            PlaceOnionEverywhereAgentConfig(),
        )
    }
    
    # start_range = 0.6
    # all_end_temps = list(np.arange(0.8, 2.6, 0.2))
    all_end_temps = list(np.arange(0.5, 4.1, 0.5))
    scales = [0.5, 1.0, 1.5, 2, 3, 4]
    num_episodes = 100
    
    pref_lists = [
        list(experiment_dict.keys()),
        all_end_temps,
        scales
        # list(range(5)),
    ]
    prod = list(itertools.product(*pref_lists))
    experiment_key, temp, scale = prod[experiment_id]
    temp_idx = all_end_temps.index(temp)
    scale_idx = scales.index(scale)
    game_cfg, scripted_cfg = experiment_dict[experiment_key]
    prefix = experiment_key[:2]
    
    scripted_agent = get_agent_from_config(scripted_cfg)
    
    
    for seed in range(5):
        fname = f'{experiment_key}_{temp_idx}_{scale_idx}_{seed}'
        # fname = f'tmp.pkl'
        print(f"{fname=}", flush=True)    
        
        set_seed(seed)
        
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
            # fixed_temperatures=[9, 9],
            num_samples=1,
            init_temp=0,
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
        game_cfg.automatic_cook_start = False
        game = OvercookedGame(game_cfg)
        
        print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}', flush=True)
        
        # t_arr = np.random.uniform(start_range, sample_end_temp, size=(num_episodes, 400))
        t_arr = np.random.normal(temp, scale=scale, size=(num_episodes, 400))
        t_arr = np.clip(t_arr, 0, None)
        t_list = [[list(t) for t in t_arr]]
        
        results, _ = do_evaluation(
            game=game,
            evaluee=scripted_agent,
            opponent_list=[alb_online_agent],
            num_episodes=[100],
            enemy_iterations=0,
            temperature_list=t_list,
            own_temperature=1,
            prevent_draw=False,
            switch_positions=False,
            verbose_level=1,
        )
        with open(save_path / fname, 'wb') as f:
            pickle.dump(results, f)

def compute_scripted_curves():
    # for std_idx, std in enumerate([0.5, 1.0, 1.5, 2.0]):
    for std_idx, std in enumerate([0.5, 1.0, 1.5, 2, 3, 4]):
        save_path = Path(__file__).parent.parent.parent / 'a_data' / 'scripted_sweep_sampling' / 'half_clipped_normal'
        img_path = Path(__file__).parent.parent.parent / 'a_img' / 'scripted_sweep_sampling'
        experiment_names = [
            'aa_put_onions',
            'aa_put_onions_delivery',
            'cc_dish_everywhere',
            'cc_onions_everywhere',
            'co_dish_everywhere',
            'co_onions_everywhere',
        ]
        # all_sample_temps = np.arange(0.8, 2.6, 0.2)
        # all_sample_temps = np.arange(1.0, 2.2, 0.2)
        all_sample_temps = np.arange(0.5, 4.1, 0.5)
        num_episodes = 100
        
        for name in experiment_names:
            data_list = []
            for t_idx, t in enumerate(all_sample_temps):
                for seed in range(5):
                    fname = f'{name}_{t_idx}_{std_idx}_{seed}'
                    with open(save_path / fname, 'rb') as f:
                        cur_data = pickle.load(f)[0]
                    data_list.append(cur_data)
            data_arr = np.asarray(data_list)
            data_arr = data_arr.reshape(len(all_sample_temps), 5, num_episodes)
            data_arr = data_arr.mean(axis=-1)
            
            plt.clf()
            plt.figure()
            seaborn.set_theme(style='whitegrid')
            
            plot_filled_std_curves(
                x=all_sample_temps,
                mean=data_arr.mean(axis=-1),
                std=data_arr.std(axis=-1),
                color=COLORS[0],
                lighter_color=LIGHT_COLORS[0],
                label=None,
                linestyle=LINESTYLES[0],
            )
            
            fontsize = 'xx-large'
            plt.xlim(all_sample_temps[0], all_sample_temps[-1])
            plt.xticks(fontsize='medium')
            plt.yticks(fontsize=fontsize)
            # plt.title(name_dict[prefix], fontsize=fontsize)
            # if idx == 0:
                # plt.legend(fontsize='x-large', loc='lower right', bbox_to_anchor=(1.01, -0.01))
            plt.ylabel('Reward', fontsize=fontsize)
            plt.xlabel('Sampl. Temp.', fontsize=fontsize)
            plt.tight_layout()
            plt.savefig(img_path / f'half_clipped_normal_std{std_idx}_{name}.pdf', bbox_inches='tight', pad_inches=0.0)
            plt.close()
        

def compute_avg():
    path = Path(__file__).parent.parent.parent / 'a_data' / 'scripted_sweep_sampling' / 'half_clipped_normal'
    experiment_names = [
        'aa_put_onions',
        'aa_put_onions_delivery',
        'cc_dish_everywhere',
        'cc_onions_everywhere',
        'co_dish_everywhere',
        'co_onions_everywhere',
    ]
    
    for name in experiment_names:
        res_list = []
        print(f"\n\n{name}\n")
        for seed in range(5):
            with open(path / f'{name}_6_3_{seed}', 'rb') as f:
                res = pickle.load(f)
            res_list.append(res)
        full_arr = np.asarray(res_list)[:, 0]
        arr = full_arr.mean(axis=-1)
        print(arr)
        print(f"{arr.mean()=}")
        print(f"{arr.std()=}")
    


if __name__ == '__main__':
    # eval_scripted_oc(3)
    # compute_scripted_curves()
    compute_avg()
