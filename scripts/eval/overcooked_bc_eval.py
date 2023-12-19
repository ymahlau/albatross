

from datetime import datetime
import itertools
import math
from pathlib import Path
import pickle
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.one_shot import NetworkAgentConfig, bc_agent_from_file
from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CoordinationRingOvercookedConfig, \
    CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, ForcedCoordinationOvercookedConfig, OvercookedRewardConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def evaluate_overcooked_response(experiment_id):
    num_games = 2
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc'
    base_name = 'resp_20from200'
    
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
        # noise_std=2,
        # fixed_temperatures=[0.1, 0.1],
        num_samples=20,
        init_temp=0,
        num_likelihood_bins=int(2e3),
        sample_from_likelihood=True,
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
    game = OvercookedGame(game_cfg)
    
    bc_path = Path(__file__).parent.parent.parent / 'bc_state_dicts'/ f'{prefix}_{seed}.pkl'
    bc_agent = bc_agent_from_file(bc_path)
    
    print(f'{datetime.now()} - Started evaluation of {prefix} with {seed=}')
    results, _ = do_evaluation(
        game=game,
        evaluee=alb_online_agent,
        opponent_list=[bc_agent],
        num_episodes=[num_games],
        enemy_iterations=0,
        temperature_list=[1],
        own_temperature=math.inf,
        prevent_draw=False,
        switch_positions=True,
        verbose_level=1,
    )
    with open(save_path / f'{base_name}_{prefix}_{seed}.pkl', 'wb') as f:
        pickle.dump(results, f)
    


if __name__ == '__main__':
    evaluate_overcooked_response(0)
