

import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np

import torch
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig, bc_agent_from_file
from src.game.actions import sample_individual_actions
from src.game.conversion import overcooked_slow_from_fast
from src.game.imp_marl.imp_marl_wrapper import IMP_MODE, IMPConfig
from src.game.initialization import get_game_from_config

from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig, CounterCircuitOvercookedConfig, CrampedRoomOvercookedConfig, OvercookedRewardConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedSlowConfig, CounterCircuitOvercookedSlowConfig
from src.network.initialization import get_network_from_config, get_network_from_file
from src.network.resnet import OvercookedResNetConfig5x5
from src.trainer.az_evaluator import do_evaluation


def old():
    # path = Path(__file__).parent.parent / 'bc_state_dicts' / 'aa_0.pkl'
    # agent = bc_agent_from_file(path)
    # a = 1
    game_cfg = CounterCircuitOvercookedConfig()
    game = OvercookedGame(game_cfg)
    
    game_cfg2 = CounterCircuitOvercookedSlowConfig(
        horizon=400,
        disallow_soup_drop=False,
        mep_reproduction_setting=True,
        mep_eval_setting=True,
        flat_obs=True,
    )
    game2 = get_game_from_config(game_cfg2)
    game2.render()
    for _ in range(100):
        game.reset()
        game2.reset()
        while not game.is_terminal():            
            game3 = overcooked_slow_from_fast(game, 'cc')
            
            game.render()
            game2.render()
            game3.render()
            
            if game2.get_str_repr() != game3.get_str_repr():
                a = 1
            print('#####################################################')
            
            actions = random.choice(game2.available_joint_actions())
            game.step(actions)
            game2.step(actions)


def main():
    bc_path = Path(__file__).parent.parent / 'bc_state_dicts'
    bc_agent = bc_agent_from_file(bc_path / 'cc_0.pkl')
    
    net_path = Path(__file__).parent.parent / 'a_models'
    proxy_net = get_network_from_file(net_path / 'proxy_cc_0.pt')
    proxy_net = proxy_net.eval()
    proxy_net_agent_cfg = NetworkAgentConfig(
        net_cfg=proxy_net.cfg,
        temperature_input=True,
        single_temperature=True,
        init_temperatures=[1.1, 1.1],
    )
    proxy_net_agent = NetworkAgent(proxy_net_agent_cfg)
    proxy_net_agent.replace_net(proxy_net)
    
    net = get_network_from_file(net_path / 'resp_cc_0.pt')
    net = net.eval()
    net_agent_cfg = NetworkAgentConfig(
        net_cfg=net.cfg,
        temperature_input=True,
        single_temperature=True,
        init_temperatures=[1.1, 1.1],
    )
    net_agent = NetworkAgent(net_agent_cfg)
    net_agent.replace_net(net)
    
    game_cfg = CounterCircuitOvercookedConfig(
        temperature_input=True,
        single_temperature_input=True,
        unstuck_behavior=True,
        reward_cfg=OvercookedRewardConfig(
            placement_in_pot=0,
            dish_pickup=0,
            soup_pickup=0,
            soup_delivery=20,
            start_cooking=0,
        )
    )
    game = OvercookedGame(game_cfg)
    
    t = 1
    
    results, _ = do_evaluation(
        game=game,
        evaluee=net_agent,
        opponent_list=[bc_agent],
        num_episodes=[4],
        enemy_iterations=0,
        temperature_list=[t, t],
        prevent_draw=False,
        switch_positions=True,
    )
    print(results)
    
    # while not game.is_terminal():
    #     game.render()
        
    #     a0_probs, _ = net_agent(game, 0)
    #     a0 = sample_individual_actions(a0_probs[np.newaxis, ...], t)[0]
        
    #     a1_probs, _ = net_agent(game, 1)
    #     a1 = sample_individual_actions(a1_probs[np.newaxis, ...], t)[0]
    #     print((a0, a1))
    #     print('############################################')
        
    #     game.step((a0, a1))
    # print(game.get_cum_rewards())


if __name__ == '__main__':
    main()
