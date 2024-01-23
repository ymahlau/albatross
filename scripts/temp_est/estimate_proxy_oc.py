from datetime import datetime
import itertools
from pathlib import Path
import pickle

import numpy as np
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig, bc_agent_from_file

from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig, OvercookedRewardConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def estimate_oc_proxy_strength(experiment_id: int):
    num_games = 50
    num_parts = 2
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc_strength'
    base_name = 'oc_proxy_strength'
    
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
        list(range(num_parts))
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed, part = prod[experiment_id]
    set_seed(part + seed * num_parts)   
    game_cfg = game_dicts[prefix]

    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'overcooked'
    proxy_path = net_path / f'proxy_{prefix}_{seed}' / 'latest.pt'
    resp_path = net_path / f'resp_{prefix}_{seed}' / 'latest.pt'
    
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
    
    print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}', flush=True)
    temperatures = np.linspace(0, 10, 20)
    for t_idx, t in enumerate(temperatures):
        print(f'Started evaluation with: {t_idx=}, {t=}')
        proxy_net_agent.reset_episode()
        proxy_net_agent.set_temperatures([t, t])
        cur_log_path = save_path / f'{base_name}_log_{prefix}_{seed}_{t_idx}_{part}.pkl'
        alb_online_agent.cfg.estimate_log_path = str(cur_log_path)
        
        results, _ = do_evaluation(
            game=game,
            evaluee=alb_online_agent,
            opponent_list=[proxy_net_agent],
            num_episodes=[num_games],
            enemy_iterations=0,
            temperature_list=[1],
            own_temperature=1,
            prevent_draw=False,
            switch_positions=True,
            verbose_level=1,
        )
        with open(save_path / f'{base_name}_{prefix}_{seed}_{t_idx}_{part}.pkl', 'wb') as f:
            pickle.dump(results, f)



if __name__ == '__main__':
    estimate_oc_proxy_strength(0)
