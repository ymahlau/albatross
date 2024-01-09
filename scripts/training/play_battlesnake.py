import math
from pathlib import Path
import pickle

import numpy as np
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgent, NetworkAgentConfig, RandomAgent, RandomAgentConfig
from src.agent.search_agent import AreaControlSearchAgentConfig
from src.game.actions import sample_individual_actions
from src.game.battlesnake.bootcamp.test_envs_7x7 import survive_on_7x7, survive_on_7x7_4_player, survive_on_7x7_constrictor, survive_on_7x7_constrictor_4_player
from src.game.initialization import get_game_from_config
from src.game.overcooked.config import Simple2CrampedRoomOvercookedConfig
from src.game.utils import step_with_draw_prevention
from src.misc.utils import set_seed

from src.network import Network
from src.network.initialization import get_network_from_file
from src.trainer.az_evaluator import do_evaluation


def play_battlesnake_example():
    # path = Path(__file__).parent.parent.parent / 'outputs' / 'luis_proxy_aa_0.pt'
    
    prefix = 'nd7'
    seed = 0
    cur_part = 8
    set_seed((seed + 1) * cur_part)
    sample_temperatures = [math.inf, math.inf]
    num_iterations = 500
    
    game_dict = {
        '4nd7': survive_on_7x7_4_player(),
        'd7': survive_on_7x7_constrictor(),
        'nd7': survive_on_7x7(),
        '4d7': survive_on_7x7_constrictor_4_player(),
    }
    game_cfg = game_dict[prefix]
    
    
    net_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'battlesnake'
    resp_path = net_path / f'{prefix}_resp_{seed}' / 'latest.pt'
    proxy_path = net_path / f'{prefix}_proxy_{seed}' / 'latest.pt'
    az_path = net_path / f'{prefix}_{seed}' / 'latest.pt'
    temp_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth_strength_mean'
    with open(temp_path / f'{prefix}.pkl', 'rb') as f:
        mean_temps = pickle.load(f)
    
    net = get_network_from_file(resp_path).eval()
    alb_network_agent_cfg = NetworkAgentConfig(
        net_cfg=net.cfg,
        temperature_input=True,
        single_temperature=False,
        init_temperatures=[0, 0, 0, 0] if prefix.startswith('4') else [0, 0],
    )
    alb_online_agent_cfg = AlbatrossAgentConfig(
        num_player=4 if prefix.startswith('4') else 2,
        agent_cfg=alb_network_agent_cfg,
        device_str='cpu',
        response_net_path=str(resp_path),
        proxy_net_path=str(proxy_path),
        # fixed_temperatures=[9.5, 9.5],
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
    
    game_cfg.ec.temperature_input = False
    game_cfg.ec.single_temperature_input = False
    game = get_game_from_config(game_cfg)
    
    agent_list = [
        alb_online_agent,
        base_agent,
    ]
    
    results_alb, game_length_alb = do_evaluation(
        game=game,
        evaluee=alb_online_agent,
        opponent_list=[base_agent],
        num_episodes=[5],
        enemy_iterations=num_iterations,
        temperature_list=[math.inf],
        own_temperature=math.inf,
        prevent_draw=False,
        switch_positions=False,
        verbose_level=2,
        own_iterations=1,
    )
    print(f"{results_alb=}")
    print(f"{game_length_alb=}")
    
    # game.reset()
    # game.render()
    # # for _ in range(50):
    # while not game.is_terminal():
    #     joint_action_list: list[int] = []
    #     prob_list = []
    #     for player in game.players_at_turn():
    #         invalid_mask = np.asarray([[a in game.illegal_actions(player) for a in range(game.num_actions)]], dtype=bool)
    #         probs, info = agent_list[player](
    #             game,
    #             player=player,
    #             iterations=num_iterations,
    #         )
    #         if player == 0:
    #             print(f"{player}: {probs}, {info['temperatures']}")
    #         if player == 1:
    #             print(f"{player}: {probs}, {info['values']}")
    #         action = sample_individual_actions(
    #             probs[np.newaxis, ...],
    #             temperature=sample_temperatures[player],
    #             invalid_mask=invalid_mask,
    #         )[0]
    #         joint_action_list.append(action)
    #     ja_tuple = tuple(joint_action_list)
    #     print(f"######################################")
    #     game.step(ja_tuple)
    #     # step_with_draw_prevention(game=game, joint_actions=ja_tuple)
        
    #     # print(f"{rewards=}")
    #     game.render()
        
        
    print(f"###########################")
    print(f"{game.get_cum_rewards()=}")
    print(f"{game.turns_played=}")

if __name__ == '__main__':
    play_battlesnake_example()
