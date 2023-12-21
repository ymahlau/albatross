

from pathlib import Path
from src.trainer.az_evaluator import do_evaluation


def evaluate_bs_depth_func():
    num_games = 100
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth'
    base_name = 'resp_proxy_temps_inf_1'
    
    game_dict = {
        'd7': (),
        '4d7': CounterCircuitOvercookedConfig(),
        'nd7': CoordinationRingOvercookedConfig(),
        '4nd7': CrampedRoomOvercookedConfig(),
    }
    
    pref_lists = [
        list(game_dict.keys()),
        list(range(5)),
    ]
    prod = list(itertools.product(*pref_lists))
    prefix, seed = prod[experiment_id]
    set_seed(seed)
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
        # fixed_temperatures=[10, 10],
        num_samples=1,
        init_temp=0,
        # num_likelihood_bins=int(2e3),
        # sample_from_likelihood=True,
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
    for t_idx, t in enumerate(np.linspace(0, 10, 100)):
        print(f'Started evaluation with: {t_idx=}, {t=}')
        proxy_net_agent.reset_episode()
        proxy_net_agent.set_temperatures([t, t])
        
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
        full_result_list.append(results)
        with open(save_path / f'{base_name}_{prefix}_{seed}.pkl', 'wb') as f:
            pickle.dump(full_result_list, f)
    

if __name__ == '__main__':
    evaluate_bs_depth_func()
