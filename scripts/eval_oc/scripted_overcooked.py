from datetime import datetime
import itertools
import math
from pathlib import Path
import numpy as np
import pickle
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig, bc_agent_from_file
from src.agent.scripted import PlaceDishEverywhereAgentConfig, PlaceOnionAgentConfig, PlaceOnionDeliverAgentConfig, PlaceOnionEverywhereAgentConfig
from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, \
    CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig, OvercookedRewardConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def eval_scripted_oc(seed: int):
    print(f'{datetime.now()} - Started eval script', flush=True)
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'scripted'
    game_cfg, prefix = AsymmetricAdvantageOvercookedConfig(), 'aa'
    # game_cfg, prefix = CoordinationRingOvercookedConfig(), 'co'
    # game_cfg, prefix = CounterCircuitOvercookedConfig(), 'cc'
    
    # init scripted agent
    scripted_cfg = PlaceOnionAgentConfig()
    # scripted_cfg = PlaceOnionDeliverAgentConfig()
    # scripted_cfg = PlaceOnionEverywhereAgentConfig()
    # scripted_cfg = PlaceDishEverywhereAgentConfig()
    
    scripted_agent = get_agent_from_config(scripted_cfg)
    
    # fname = f'{prefix}_dish_everywhere_{seed}.pkl'
    fname = f'tmp.pkl'
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
    results, _ = do_evaluation(
        game=game,
        evaluee=scripted_agent,
        opponent_list=[alb_online_agent],
        num_episodes=[100],
        enemy_iterations=0,
        temperature_list=[0.5],
        own_temperature=1,
        prevent_draw=False,
        switch_positions=False,
        verbose_level=1,
    )
    with open(save_path / fname, 'wb') as f:
        pickle.dump(results, f)
    

def compute_avg():
    path = Path(__file__).parent.parent.parent / 'a_data' / 'scripted'
    res_list = []
    for seed in range(5):
        with open(path / f'cc_dish_everywhere_{seed}.pkl', 'rb') as f:
            res = pickle.load(f)
        res_list.append(res)
    full_arr = np.asarray(res_list)[:, 0]
    arr = full_arr.mean(axis=-1)
    print(arr)
    print(f"{arr.mean()=}")
    print(f"{arr.std()=}")
    


if __name__ == '__main__':
    # eval_scripted_oc(0)
    compute_avg()
