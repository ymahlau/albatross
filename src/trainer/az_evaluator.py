import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import torch.multiprocessing as mp

from src.agent import Agent
from src.agent.initialization import get_agent_from_config
from src.agent.one_shot import NetworkAgentConfig
from src.agent.search_agent import SBRFixedDepthAgentConfig, SearchAgent
from src.game.actions import sample_individual_actions
from src.game.game import Game
from src.game.initialization import get_game_from_config
from src.game.utils import step_with_draw_prevention
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_config, get_network_from_file
from src.trainer.config import EvaluatorConfig, AlphaZeroTrainerConfig
from src.trainer.utils import wait_for_obj_from_queue, send_obj_to_queue


@torch.no_grad()
def run_evaluator(
        trainer_cfg: AlphaZeroTrainerConfig,
        net_queue: mp.Queue,
        stop_flag: mp.Value,
        info_queue: mp.Queue,
        cpu_list: Optional[list[int]],  # only works on linux
        prev_run_dir: Optional[Path],
        prev_run_idx: Optional[int],
        seed: int,
):
    net_cfg = trainer_cfg.net_cfg
    game_cfg = trainer_cfg.game_cfg
    evaluator_cfg = trainer_cfg.evaluator_cfg
    # important for multiprocessing
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    set_seed(seed)
    # paths
    model_folder: Path = Path(os.getcwd()) / 'eval_models'
    # copy previous model to this directory
    if not Path.exists(model_folder):
        model_folder.mkdir(parents=True, exist_ok=True)
    # initialization
    game = get_game_from_config(game_cfg)
    # network
    net = get_network_from_config(net_cfg)
    if prev_run_dir is not None:
        net = get_network_from_file(prev_run_dir / 'fixed_time_models' / f'm_{prev_run_idx}.pt')
    net = net.eval()
    # initialization
    start_time = time.time()
    eval_counter = 0
    # value player, use 2x temperature due to double sampling (weighting and sampling)
    value_agent_cfg = SBRFixedDepthAgentConfig()
    if trainer_cfg.temperature_input:
        value_agent_cfg.search_cfg.eval_func_cfg.temperature_input = True
        value_agent_cfg.search_cfg.eval_func_cfg.single_temperature = trainer_cfg.single_sbr_temperature
        value_agent_cfg.search_cfg.eval_func_cfg.obs_temperature_input = trainer_cfg.obs_temperature_input
    value_agent = SearchAgent(value_agent_cfg)
    value_agent.replace_net(net)
    value_agent.set_temperatures([12.5 for _ in range(game.num_players)])
    # policy player
    policy_agent = None
    if net.cfg.predict_policy:
        policy_agent_cfg = NetworkAgentConfig(net_cfg=net_cfg)
        if trainer_cfg.temperature_input:
            policy_agent_cfg.temperature_input = True
            policy_agent_cfg.single_temperature = trainer_cfg.single_sbr_temperature
            policy_agent_cfg.obs_temperature_input = trainer_cfg.obs_temperature_input
        policy_agent = get_agent_from_config(policy_agent_cfg)
        policy_agent.set_temperatures([12.5 for _ in range(game.num_players)])
        policy_agent.replace_net(net)
    # opponents
    opponent_list = []
    for opponent_cfg in evaluator_cfg.enemy_cfgs:
        opponent_list.append(get_agent_from_config(opponent_cfg))
    # restrict cpus
    pid = os.getpid()
    if cpu_list is not None:
        print(f"{datetime.now()} - CPU list in Evaluator: {cpu_list}")
        os.sched_setaffinity(pid, cpu_list)
        print(f'{datetime.now()} - Started evaluation process with pid {pid} using cpus: {os.sched_getaffinity(pid)}',
              flush=True)
    else:
        print(f'{datetime.now()} - Started evaluation process with pid {pid}', flush=True)
    try:
        while not stop_flag.value:
            # do the evaluation
            last_eval_time = time.time()
            results_val, episode_lengths_val = do_evaluation(
                game=game,
                evaluee=value_agent,
                opponent_list=opponent_list,
                num_episodes=evaluator_cfg.num_episodes,
                enemy_iterations=evaluator_cfg.enemy_iterations,
                temperature=math.inf,
                prevent_draw=evaluator_cfg.prevent_draw,
            )
            eval_time_value = time.time() - last_eval_time
            time_played = (time.time() - start_time) / 60
            # data of value evaluation
            msg_data = generate_msg_and_print(
                evaluator_cfg=evaluator_cfg,
                results=results_val,
                episode_lengths=episode_lengths_val,
                is_value=True,
                time_played_min=time_played,
                eval_time=eval_time_value,
                eval_counter=eval_counter,
                input_dict={}
            )
            # policy evaluation
            if net_cfg.predict_policy:
                last_eval_time = time.time()
                results_pol, episode_lengths_pol = do_evaluation(
                    game=game,
                    evaluee=policy_agent,
                    opponent_list=opponent_list,
                    num_episodes=evaluator_cfg.num_episodes,
                    enemy_iterations=evaluator_cfg.enemy_iterations,
                    temperature=evaluator_cfg.temperature,
                    prevent_draw=evaluator_cfg.prevent_draw,
                )
                eval_time_policy = time.time() - last_eval_time
                # data of policy evaluation
                time_played = (time.time() - start_time) / 60
                msg_data = generate_msg_and_print(
                    evaluator_cfg=evaluator_cfg,
                    results=results_pol,
                    episode_lengths=episode_lengths_pol,
                    is_value=False,
                    time_played_min=time_played,
                    eval_time=eval_time_policy,
                    eval_counter=eval_counter,
                    input_dict=msg_data,
                )
            # send info to logging process
            send_obj_to_queue(msg_data, info_queue, stop_flag)
            if stop_flag.value:
                break
            # save model
            cur_model_path = model_folder / f"m_{eval_counter}.pt"
            net.save(cur_model_path)
            if stop_flag.value:
                break
            # update counter
            eval_counter += 1
            # wait until enough time has passed between evaluations
            time_passed_sec = time.time() - last_eval_time
            if time_passed_sec < evaluator_cfg.eval_rate_sec:
                time.sleep(evaluator_cfg.eval_rate_sec - time_passed_sec)
            # get the newest version of network, wait if not available yet
            maybe_state_dict = wait_for_obj_from_queue(net_queue, stop_flag, timeout=5)
            if stop_flag.value:
                break
            if maybe_state_dict is None:
                raise Exception("Unknown exception with queue")
            net.load_state_dict(maybe_state_dict)
            net.eval()
            value_agent.replace_net(net)
            if net_cfg.predict_policy:
                policy_agent.replace_net(net)
    except KeyboardInterrupt:
        print('Detected KeyboardInterrupt in Evaluation process\n', flush=True)
    print(f"{datetime.now()} - closed process {os.getpid()}", flush=True)
    sys.exit(0)


def generate_msg_and_print(
        evaluator_cfg: EvaluatorConfig,
        results: list[list[float]],
        episode_lengths: list[list[float]],
        is_value: bool,
        time_played_min: float,
        eval_time: float,
        eval_counter: int,
        input_dict: dict[str, Any],
) -> dict[str, Any]:
    id_str = 'onval' if is_value else 'onpol'
    avg_outcomes = [sum(x) / len(x) for x in results]
    avg_ep_len = [sum(x) / len(x) for x in episode_lengths]
    win_ratio = [sum([v == 1 for v in x]) / len(x) for x in results]
    draw_ratio = [sum([v == 0 for v in x]) / len(x) for x in results]
    loss_ratio = [sum([v == -1 for v in x]) / len(x) for x in results]
    input_dict[f'eval_time_{id_str}'] = eval_time
    for idx, opp_cfg in enumerate(evaluator_cfg.enemy_cfgs):
        input_dict[f'{opp_cfg.name}_{id_str}_avg_outcome'] = avg_outcomes[idx]
        input_dict[f'{opp_cfg.name}_{id_str}_std_outcome'] = np.std(results[idx]).item()
        input_dict[f'{opp_cfg.name}_{id_str}_median_outcome'] = np.median(results[idx]).item()
        input_dict[f'{opp_cfg.name}_{id_str}_avg_episode_len'] = avg_ep_len[idx]
        input_dict[f'{opp_cfg.name}_{id_str}_win_percentage'] = win_ratio[idx]
        input_dict[f'{opp_cfg.name}_{id_str}_draw_percentage'] = draw_ratio[idx]
        input_dict[f'{opp_cfg.name}_{id_str}_lose_percentage'] = loss_ratio[idx]
    print(f"{datetime.now()} - {eval_counter} Time: {time_played_min:.1f} min, avg outcomes: "
          f"{id_str}: {avg_outcomes}\n", flush=True)
    return input_dict


@torch.no_grad()
def do_evaluation(
        game: Game,
        evaluee: Agent,
        opponent_list: list[Agent],
        num_episodes: int,
        enemy_iterations: int,
        temperature: float,
        prevent_draw: bool,
) -> tuple[list[list[float]], list[list[int]]]:
    results = []
    episode_lengths = []
    # iterate opponents
    for opponent in opponent_list:
        cur_results = []
        cur_lengths = []
        # iterate episodes
        for i in range(num_episodes):
            game.reset()
            step_counter = 0
            while not game.is_terminal() and game.is_player_at_turn(0):
                joint_action_list: list[int] = []
                for player in game.players_at_turn():
                    if player == 0:  # agent to evaluate always plays as player 0
                        probs, _ = evaluee(game, player=0, iterations=1)
                        probs[game.illegal_actions(0)] = 0
                        probs /= probs.sum()
                        action = sample_individual_actions(probs[np.newaxis, ...], temperature)[0]
                    else:
                        probs, _ = opponent(game, player=player, iterations=enemy_iterations)
                        probs[game.illegal_actions(player)] = 0
                        probs /= probs.sum()
                        action = sample_individual_actions(probs[np.newaxis, ...], math.inf)[0]
                    joint_action_list.append(action)
                if prevent_draw:
                    step_with_draw_prevention(game, tuple(joint_action_list))
                else:
                    game.step(tuple(joint_action_list))
                step_counter += 1
            # add rewards of player 0 to sum
            cum_rewards = game.get_cum_rewards()
            cur_results.append(cum_rewards[0].item())
            cur_lengths.append(step_counter)
        results.append(cur_results)
        episode_lengths.append(cur_lengths)
    return results, episode_lengths