import math
from pathlib import Path

import numpy as np
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig, RandomAgent, RandomAgentConfig
from src.game.actions import sample_individual_actions
from src.game.initialization import get_game_from_config
from src.game.overcooked.config import Simple2CrampedRoomOvercookedConfig

from src.network import Network
from src.network.initialization import get_network_from_file


def play_overcooked_example():
    path = Path(__file__).parent.parent.parent / 'outputs' / 'latest.pt'
    temperature_input = True
    single_temperature = True
    
    net = get_network_from_file(path).eval()
    game_cfg = net.cfg.game_cfg
    
    
    if game_cfg is None:
        raise Exception()
    # game_cfg = Simple2CrampedRoomOvercookedConfig()
    game = get_game_from_config(game_cfg)
    
    
    
    agent0 = NetworkAgent(
        NetworkAgentConfig(
            net_cfg=net.cfg, 
            temperature_input=temperature_input, 
            single_temperature=single_temperature, 
            init_temperatures=[5, 5]
        )
    )
    agent0.net = net
    
    # agent1 = RandomAgent(RandomAgentConfig())
    agent1 = NetworkAgent(
        NetworkAgentConfig(
            net_cfg=net.cfg, 
            temperature_input=temperature_input, 
            single_temperature=single_temperature, 
            init_temperatures=[5, 5]
            )
        )
    agent1.net = net
    
    agent_list = [
        agent1,
        agent0,
    ]
    sample_temperatures = [math.inf, math.inf]
    # sample_temperatures = [1, 1]
    
    # play
    # temperatures = np.linspace(0, 10, 15)
    temperatures = [5.]
    for t in temperatures:
        agent0.temperatures = [t, t]
        agent1.temperatures = [t, t]
        game.reset()
        game.render()
        # for _ in range(20):
        while not game.is_terminal():
            joint_action_list: list[int] = []
            prob_list = []
            for player in game.players_at_turn():
                invalid_mask = np.asarray([[a in game.illegal_actions(player) for a in range(game.num_actions)]], dtype=bool)
                probs, info = agent_list[player](
                    game,
                    player=player
                )
                print(f"{player}: {probs}, {info['values']}")
                action = sample_individual_actions(
                    probs[np.newaxis, ...],
                    temperature=sample_temperatures[player],
                    invalid_mask=invalid_mask,
                )[0]
                joint_action_list.append(action)
            ja_tuple = tuple(joint_action_list)
            print(f"######################################")
            rewards, _, _ = game.step(ja_tuple)
            
            print(f"{rewards=}")
            game.render()
        
        
    print(f"###########################")
    print(f"{game.get_cum_rewards()=}")


if __name__ == '__main__':
    play_overcooked_example()
