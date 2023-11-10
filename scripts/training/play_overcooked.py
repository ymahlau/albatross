import math
from pathlib import Path

import numpy as np

from src.agent.albatross import AlbatrossAgentConfig
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgentConfig, BCNetworkAgentConfig
from src.game.actions import sample_individual_actions
from src.game.initialization import get_game_from_config
from src.game.overcooked.layouts import CoordinationRingOvercookedConfig, AsymmetricAdvantageOvercookedConfig, \
    CrampedRoomOvercookedConfig
from src.network.initialization import get_network_from_file


def play_oc_func():
    # game_cfg = CoordinationRingOvercookedConfig(horizon=400, mep_eval_setting=True)
    game_cfg = CoordinationRingOvercookedConfig()
    # game_cfg = AsymmetricAdvantageOvercookedConfig(horizon=400)
    # game_cfg = CrampedRoomOvercookedConfig(horizon=100)

    game = get_game_from_config(game_cfg)

    agent_cfg = NetworkAgentConfig()
    agent = get_agent_from_config(agent_cfg)
    model_path = Path(__file__).parent.parent.parent.parent / 'a_models' / 'or_duct.pt'
    net = get_network_from_file(model_path)
    net = net.eval()
    agent.replace_net(net)

    bc_agent_cfg = BCNetworkAgentConfig()
    bc_agent = get_agent_from_config(bc_agent_cfg)
    bc_model_path = Path(__file__).parent.parent.parent.parent / 'a_models' / 'bc' / 'or.pt'
    bc_net = get_network_from_file(bc_model_path)
    bc_net = bc_net.eval()
    bc_agent.replace_net(bc_net)

    alb_network_agent_cfg = NetworkAgentConfig(
        temperature_input=True,
        init_temperatures=[0 for _ in range(game_cfg.num_players)],
        single_temperature=False,
        obs_temperature_input=True,
    )
    # bc_gt_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'oc_alb' / 'bc_strength'
    # with open(bc_gt_path / f'gt_bc_{prefix}.pkl', 'rb') as f:
    #     gt = pickle.load(f)
    # gt_val = float(gt[seed])
    alb_online_agent_cfg = AlbatrossAgentConfig(
        agent_cfg=alb_network_agent_cfg,
        device_str='cpu',
        response_net_path=str(Path(__file__).parent.parent.parent.parent / 'a_models' / 'or_response.pt'),
        proxy_net_path=str(Path(__file__).parent.parent.parent.parent / 'a_models' / 'or_proxy.pt'),
        num_player=game_cfg.num_players,
        min_temp=0,
        max_temp=25,
        # noise_std=std.item(),
        # fixed_temperatures=[t.item() for _ in range(game_cfg.num_players)]
    )
    alb_agent = get_agent_from_config(alb_online_agent_cfg)

    # sample_temperatures = [1 for _ in range(2)]
    sample_temperatures = [0, 1]
    agent_list = [
        alb_agent,
        bc_agent,
    ]

    # play
    game.render()
    while not game.is_terminal():
        joint_action_list: list[int] = []
        for player in game.players_at_turn():
            invalid_mask = np.asarray([[a in game.illegal_actions(player) for a in range(game.num_actions)]],
                                      dtype=bool)
            probs, _ = agent_list[player](
                game,
                player=player
            )
            action = sample_individual_actions(
                probs[np.newaxis, ...],
                temperature=sample_temperatures[player],
                invalid_mask=invalid_mask,
            )[0]
            joint_action_list.append(action)
        ja_tuple = tuple(joint_action_list)
        rewards, _, _ = game.step(ja_tuple)

        print(f"######################################")
        game.render()
        print(f"{rewards=}")
    print(f"###########################")
    print(f"{game.get_cum_rewards()=}")


if __name__ == '__main__':
    play_oc_func()
