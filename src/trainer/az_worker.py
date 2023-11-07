import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp

from src.game.game import Game
from src.game.actions import sample_individual_actions, apply_permutation, filter_illegal_and_normalize
from src.game.initialization import get_game_from_config
from src.game.utils import step_with_draw_prevention
from src.misc.replay_buffer import BufferInputSample
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_config, get_network_from_file
from src.search import SearchInfo, Search
from src.search.initialization import get_search_from_config
from src.search.utils import compute_q_values
from src.supervised.annealer import TemperatureAnnealer
from src.trainer.config import WorkerConfig, AlphaZeroTrainerConfig
from src.trainer.policy_eval import PolicyEvalType
from src.trainer.utils import send_obj_to_queue, register_exit_function, get_latest_obj_from_queue


@dataclass
class WorkerStatistics:
    step_counter: mp.Value
    episode_counter: mp.Value
    load_time_sum: float = 0
    search_time_sum: float = 0
    idle_time_sum: float = 0
    data_conv_time_sum: float = 0
    step_time_sum: float = 0
    search_counter: int = 0
    episode_len_list: list[int] = field(default_factory=lambda: [])
    search_info_sum: SearchInfo = field(default_factory=lambda: SearchInfo())
    search_info_counter: int = 0
    last_info_time: float = time.time()
    process_start_time: float = time.time()
    reward_sum: float = 0
    reward_counter: int = 0


@torch.no_grad()
def run_worker(
        worker_id: int,
        trainer_cfg: AlphaZeroTrainerConfig,
        data_queue: mp.Queue,
        net_queue: mp.Queue,
        stop_flag: mp.Value,
        info_queue: mp.Queue,
        step_counter: mp.Value,
        episode_counter: mp.Value,
        error_counter: mp.Value,
        cpu_list: Optional[list[int]],
        gpu_idx: Optional[int],
        prev_run_dir: Optional[Path],
        prev_run_idx: Optional[int],
        seed: int,
        debug: bool = False,
):
    game_cfg = trainer_cfg.game_cfg
    net_cfg = trainer_cfg.net_cfg
    worker_cfg = trainer_cfg.worker_cfg
    # important for multiprocessing
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    set_seed(seed)
    # initialization
    search = get_search_from_config(worker_cfg.search_cfg)
    game = get_game_from_config(game_cfg)
    if worker_cfg.num_gpu != 0:
        device = torch.device('cuda' if gpu_idx is None else f'cuda:{gpu_idx}')
        torch.cuda.init()
    else:
        device = torch.device('cpu')
    # network
    net = None
    if net_cfg is not None:
        net = get_network_from_config(net_cfg)
        if prev_run_dir is not None:
            net = get_network_from_file(prev_run_dir / 'fixed_time_models' / f'm_{prev_run_idx}.pt')
        # net = fabric.setup_module(net, move_to_device=False)
        net = net.to(device)
        net = net.eval()  # We are not updating here, but evaluating during play to generate new data
        if not worker_cfg.quick_start:  # if quick start do not use network in search-eval
            search.replace_net(net)
            search.replace_device(device)
        else:
            print(f"Started Worker {worker_id} in Quick-Start mode without network", flush=True)
    else:
        print(f"Started Worker {worker_id} for Buffer Generation")
    # exit
    if net_cfg is not None:
        register_exit_function(net, name=f'worker_net_{worker_id}')
    # info for logging
    stats = WorkerStatistics(step_counter=step_counter, episode_counter=episode_counter)
    if hasattr(search, "backup_func"):
        if hasattr(search.backup_func, "error_counter"):
            search.backup_func.error_counter = error_counter
    annealer_list: Optional[list[TemperatureAnnealer]] = None
    if worker_cfg.anneal_cfgs is not None:
        if trainer_cfg.single_sbr_temperature:
            if len(worker_cfg.anneal_cfgs) != 1:
                raise ValueError(f"Please provide a single anneal_cfg when using single temperature")
            annealer_list = [TemperatureAnnealer(worker_cfg.anneal_cfgs[0])]
        else:
            annealer_list = [TemperatureAnnealer(cfg) for cfg in worker_cfg.anneal_cfgs]
    # restrict cpus
    pid = os.getpid()
    if cpu_list is not None:
        print(f"{datetime.now()} - CPU list in Worker: {cpu_list}")
        os.sched_setaffinity(pid, cpu_list)
        print(f'{datetime.now()} - Started Worker process with pid {pid} using cpus: {os.sched_getaffinity(pid)}'
              f" using device {device}", flush=True)
    else:
        print(f'{datetime.now()} - Started Worker process with pid {pid} using device {device}', flush=True)
    try:
        while not stop_flag.value:
            # check if new version of net is available
            load_time_start = time.time()
            maybe_state_dict = get_latest_obj_from_queue(net_queue)
            if stop_flag.value:
                break
            if maybe_state_dict is not None:
                net.load_state_dict(maybe_state_dict)
                net = net.eval()
                search.replace_net(net)
                search.replace_device(device)
            stats.load_time_sum += time.time() - load_time_start
            # for exploration, start game with a random number of random steps
            game.reset()
            num_random_steps = random.randint(0, worker_cfg.max_random_start_steps)
            game.play_random_steps(num_random_steps)
            episode_step_counter = num_random_steps
            # init
            game_list, value_list, reward_list, policy_list = [], [], [], []
            temp_list: Optional[list[list[float]]] = []
            # do search and send results
            while not game.is_terminal() and not stop_flag.value:
                if debug:
                    game.render()
                # search
                search_time_start = time.time()
                # parse options
                if annealer_list is not None:
                    time_passed = (time.time() - stats.process_start_time) / 60
                    if trainer_cfg.single_sbr_temperature:
                        temperature = annealer_list[0](time_passed)
                        cur_temp_list = [temperature for _ in range(game_cfg.num_players)]
                    else:
                        cur_temp_list = [a(time_passed) for a in annealer_list]
                    temp_list.append(cur_temp_list)
                    search.set_temperatures(cur_temp_list)
                # do the search
                values, action_probs, info = search(
                    game=game,
                    iterations=worker_cfg.search_iterations,
                )
                if debug:
                    print(f"{values=}")
                    print(f"{action_probs=}")
                    print(f"#############################")
                stats.search_time_sum += time.time() - search_time_start
                stats.search_counter += 1
                if stop_flag.value:
                    break
                # update stats
                idle_time_start = time.time()
                with step_counter.get_lock():
                    step_counter.value += 1
                stats.idle_time_sum += time.time() - idle_time_start
                episode_step_counter += 1
                # add results to episode list
                add_search_time(stats, info)
                game_list.append(game.get_copy())
                value_list.append(values)
                policy_list.append(action_probs)
                # make step
                step_time_start = time.time()
                rewards = make_step(worker_cfg, game, action_probs, search)
                reward_list.append(rewards)
                stats.reward_sum += np.sum(rewards).item()
                stats.step_time_sum += time.time() - step_time_start
            # after end of episode
            if debug:
                game.render()
                print("############################################")
            with episode_counter.get_lock():
                episode_counter.value += 1
            # print(reward_list, flush=True)
            stats.episode_len_list.append(episode_step_counter)
            stats.reward_counter += game.num_players
            # send results to queue
            if game_list:
                data_conv_time_start = time.time()
                if annealer_list is None:
                    temp_list = None
                sample = convert_training_data(game_list, value_list, reward_list, policy_list, trainer_cfg, temp_list)
                stats.data_conv_time_sum += time.time() - data_conv_time_start
                idle_time_start = time.time()
                send_obj_to_queue(sample, data_queue, stop_flag)
                stats.idle_time_sum += time.time() - idle_time_start
            # send info about episodes to logging process
            if len(stats.episode_len_list) == trainer_cfg.logger_cfg.worker_episode_bucket_size:
                time_passed = (time.time() - stats.process_start_time) / 60
                temps = None if not annealer_list else [a(time_passed) for a in annealer_list]
                send_info(stats, info_queue, stop_flag, temps)
                if stop_flag.value:
                    break
    except KeyboardInterrupt:
        print('Detected Keyboard Interrupt in Worker Process\n', flush=True)
    game.close()
    print(f'{datetime.now()} - Worker process {os.getpid()} is done', flush=True)
    sys.exit(0)


def make_step(
        worker_cfg: WorkerConfig,
        game: Game,
        action_probs: np.ndarray,  # action probabilities of shape (num_players_at_turn, num_actions)
        search: Search,
) -> np.ndarray:  # returns reward array of shape (players, )
    # add boltzmann exploration term
    uniform_actions = np.ones(shape=(game.num_players_at_turn(), game.num_actions), dtype=float)
    filtered_uniform = filter_illegal_and_normalize(uniform_actions, game)
    exp_action_probs = np.zeros_like(action_probs)
    for player_idx, player in enumerate(game.players_at_turn()):
        # compute q-values given uniform enemy
        q_vals = compute_q_values(node=search.root, player=player, action_probs=filtered_uniform)
        q_arr = np.asarray(q_vals, dtype=float)
        # softmax action selection
        cur_exp_probs = np.exp(q_arr) / np.sum(np.exp(q_arr))
        for action_idx, action in enumerate(game.available_actions(player)):
            exp_action_probs[player_idx, action] = cur_exp_probs[action_idx]
    # mix policy and exploration
    mixed_probs = (1 - worker_cfg.exploration_prob) * action_probs + worker_cfg.exploration_prob * exp_action_probs
    # sample
    joint_actions = sample_individual_actions(mixed_probs, temperature=worker_cfg.temperature)
    if worker_cfg.prevent_draw:
        rewards = step_with_draw_prevention(game=game, joint_actions=joint_actions)
    else:
        rewards, _, _ = game.step(joint_actions)
    return rewards


def add_search_time(stats: WorkerStatistics, info: SearchInfo):
    stats.search_info_sum.select_time_ratio += info.select_time_ratio
    stats.search_info_sum.other_time_ratio += info.other_time_ratio
    stats.search_info_sum.cleanup_time_ratio += info.cleanup_time_ratio
    stats.search_info_sum.eval_time_ratio += info.eval_time_ratio
    stats.search_info_sum.backup_time_ratio += info.backup_time_ratio
    stats.search_info_sum.extract_time_ratio += info.extract_time_ratio
    stats.search_info_sum.expansion_time_ratio += info.expansion_time_ratio
    stats.search_info_counter += 1


def send_info(
        stats: WorkerStatistics,
        info_queue: mp.Queue,
        stop_flag: mp.Value,
        temps: Optional[list[float]],
):
    full_time = time.time() - stats.last_info_time
    other_time = full_time - stats.load_time_sum - stats.search_time_sum - stats.idle_time_sum - stats.step_time_sum
    other_time -= stats.data_conv_time_sum
    msg_data = {
        'worker_load_ratio': stats.load_time_sum / full_time,
        'worker_search_ratio': stats.search_time_sum / full_time,
        'worker_idle_ratio': stats.idle_time_sum / full_time,
        'worker_data_ratio': stats.data_conv_time_sum / full_time,
        'worker_step_ratio': stats.step_time_sum / full_time,
        'worker_other_ratio': other_time / full_time,
        'avg_step_time': full_time / stats.search_counter,
        'worker_avg_episode_length': sum(stats.episode_len_list) / len(stats.episode_len_list),
        'search_cleanup_ratio': stats.search_info_sum.cleanup_time_ratio / stats.search_info_counter,
        'search_select_ratio': stats.search_info_sum.select_time_ratio / stats.search_info_counter,
        'search_other_ratio': stats.search_info_sum.other_time_ratio / stats.search_info_counter,
        'search_eval_ratio': stats.search_info_sum.eval_time_ratio / stats.search_info_counter,
        'search_backup_ratio': stats.search_info_sum.backup_time_ratio / stats.search_info_counter,
        'search_extract_ratio': stats.search_info_sum.extract_time_ratio / stats.search_info_counter,
        'search_expand_ratio': stats.search_info_sum.expansion_time_ratio / stats.search_info_counter,
        'worker_avg_outcome': stats.reward_sum / stats.reward_counter,
    }
    if full_time > 0:
        msg_data["steps_per_min"] = stats.search_counter * 60 / full_time
    if temps:
        for p, t_p in enumerate(temps):
            msg_data[f"sbr_temperature_{p}"] = t_p
    # send info
    idle_time_start = time.time()
    send_obj_to_queue(msg_data, info_queue, stop_flag)
    stats.idle_time_sum = time.time() - idle_time_start
    # reset values
    stats.last_info_time = time.time()
    stats.load_time_sum = 0
    stats.search_time_sum = 0
    stats.data_conv_time_sum = 0
    stats.step_time_sum = 0
    stats.search_counter = 0
    stats.reward_counter = 0
    stats.reward_sum = 0
    stats.episode_len_list = []
    stats.search_info_sum = SearchInfo()
    stats.search_info_counter = 0


def convert_training_data(
        game_list: list[Game],
        value_list: list[np.ndarray],  # shape (num_player,)
        reward_list: list[np.ndarray],  # shape (num_player,)
        policy_list: list[np.ndarray],  # shape (num_players_at_turn, num_actions)
        trainer_cfg: AlphaZeroTrainerConfig,
        temperature_list: Optional[list[list[float]]],
) -> BufferInputSample:
    value_tensor_list, policy_tensor_list, obs_tensor_list, game_len_list = [], [], [], []
    turns_list, player_list, symmetry_list = [], [], []
    temp_list = []
    # find game length for all players
    len_per_player = [-1 for _ in range(game_list[0].num_players)]
    for i, game in enumerate(game_list):
        for player in game.players_not_at_turn():
            if len_per_player[player] == -1:
                len_per_player[player] = i
    for i, pl in enumerate(len_per_player):
        if pl == -1:
            len_per_player[i] = len(game_list)
    # policy evaluation
    eval_value_list = compute_policy_evaluation(game_list, value_list, reward_list, trainer_cfg.worker_cfg)
    # construct
    counter = 0
    for game, values, action_probs in zip(game_list, eval_value_list, policy_list):
        # turn values and actions to tensor
        value_tensor = torch.tensor(values, dtype=torch.float32).cpu()
        # iterate symmetries
        sims = [0] if not trainer_cfg.worker_cfg.use_symmetries else list(range(game.get_symmetry_count()))
        for symmetry in sims:
            temp_obs_input = None
            if trainer_cfg.temperature_input and trainer_cfg.obs_temperature_input:
                if trainer_cfg.single_sbr_temperature:
                    temp_obs_input = [temperature_list[counter][0]]
                else:
                    temp_obs_input = temperature_list[counter]
            # shape (num_players_at_turn, *obs_shape)
            obs, perm, inverse_perm = game.get_obs(symmetry=symmetry, temperatures=temp_obs_input)
            # apply permutation to action probs
            perm_action = apply_permutation(action_probs, perm)
            perm_action_tensor = torch.tensor(perm_action, dtype=torch.float32).cpu()
            # iterate players
            for player_idx, player in enumerate(game.players_at_turn()):
                # parse temperature list only for film: either single or of all enemies
                if trainer_cfg.temperature_input and not trainer_cfg.obs_temperature_input:
                    cur_temp = temperature_list[counter]
                    if trainer_cfg.single_sbr_temperature:
                        temp_list.append([cur_temp[0]])  # single temperature
                    else:
                        t_list = [cur_temp[e] for e in range(game.num_players) if e != player]
                        temp_list.append(t_list)
                # value, policy and obs are tensors
                obs_tensor_list.append(obs[player_idx, ...])
                value_tensor_list.append(value_tensor[player])
                policy_tensor_list.append(perm_action_tensor[player_idx, :])
                # length, turn, player and symmetry are not
                length_val = (len_per_player[player] - counter) / trainer_cfg.worker_cfg.max_game_length
                game_len_list.append(length_val)
                turns_list.append(counter)
                player_list.append(player)
                symmetry_list.append(symmetry)
        # stack results
        counter += 1
    # concat results
    full_obs = np.stack(obs_tensor_list, axis=0)
    full_values = np.stack(value_tensor_list, axis=0)[:, np.newaxis]
    full_policies = np.stack(policy_tensor_list, axis=0)
    # convert to tensor
    full_len = torch.tensor(game_len_list, dtype=torch.float32).unsqueeze(1)
    full_turns = torch.tensor(turns_list, dtype=torch.float32).unsqueeze(1)
    full_players = torch.tensor(player_list, dtype=torch.float32).unsqueeze(1)
    full_symmetry = torch.tensor(symmetry_list, dtype=torch.float32).unsqueeze(1)
    full_temps = None
    if trainer_cfg.temperature_input and not trainer_cfg.obs_temperature_input:
        full_temps = torch.tensor(temp_list, dtype=torch.float32)
    sample = BufferInputSample(
        obs=full_obs,
        values=full_values,
        policies=full_policies,
        game_lengths=full_len,
        turns=full_turns,
        player=full_players,
        symmetry=full_symmetry,
        temperature=full_temps,
    )
    return sample


def compute_policy_evaluation(
        game_list: list[Game],
        value_list: list[np.ndarray],  # shape (num_player,)
        reward_list: list[np.ndarray],  # shape (num_player,)
        worker_cfg: WorkerConfig,
) -> list[np.ndarray]:
    method = worker_cfg.policy_eval_cfg.eval_type
    if method == PolicyEvalType.MC or method == PolicyEvalType.MC.value:
        # MC, compute discounted reward at each step
        reward_sum = np.zeros(shape=(game_list[0].num_players,), dtype=float)
        result_list = []
        for reward in reward_list[::-1]:
            cur_rewards = worker_cfg.search_cfg.discount * (reward + reward_sum)
            result_list.append(cur_rewards)
            reward_sum = cur_rewards
        return result_list[::-1]
    elif method == PolicyEvalType.TD_0 or method == PolicyEvalType.TD_0.value:
        # TD-0
        return value_list
    elif method == PolicyEvalType.TD_LAMBDA or method == PolicyEvalType.TD_LAMBDA.value:
        # TD-lambda, iteratively calculate all n-step returns and weight them
        result_arr = td_lambda(
            full_rewards=np.asarray(reward_list).T,
            full_values=np.asarray(value_list).T,
            ld=worker_cfg.policy_eval_cfg.lambda_val,
            discount=worker_cfg.search_cfg.discount,
        )
        result_list = list(result_arr.T)
        return result_list
    else:
        raise ValueError(f"Unknown policy eval type: {method}")


def td_lambda(
     full_rewards: np.ndarray,  # shape(player, T)
        full_values: np.ndarray,  # shape(player, T)
        ld: float,
        discount: float,
) -> np.ndarray:  # shape(player, T)
    result_arr = np.zeros_like(full_rewards)
    num_player = full_values.shape[0]
    length = full_values.shape[1]
    vals = np.zeros(shape=(num_player,), dtype=float)
    for t in range(length-1, -1, -1):
        vals = discount * (full_rewards[:, t] + vals)
        interpolated = ld * vals + (1 - ld) * full_values[:, t]
        result_arr[:, t] = interpolated
        vals = interpolated
    return result_arr


def td_lambda_inefficient(
        full_rewards: np.ndarray,  # shape(player, T)
        full_values: np.ndarray,  # shape(player, T)
        ld: float,
        discount: float,
        episode_len_player: list[int],
) -> np.ndarray:  # shape(player, T)
    """
    Just a reference implementation of the actual mathematical formula. This is vastly inefficient, do not use.
    """
    result_arr = np.zeros_like(full_rewards)
    for player, T in enumerate(episode_len_player):
        for t in range(T):
            return_ld_t = 0
            for n in range(T - t + 1):
                # value estimate
                if n == T - t:
                    return_t_n = 0
                else:
                    return_t_n = np.power(discount, n) * full_values[player, t + n]
                # construct n-step return
                if n > 0:
                    exponents = np.arange(1, n+1)
                    factors = np.power(discount, exponents)
                    rewards = full_rewards[player, t:t+n]
                    sum_terms = factors * rewards
                    return_t_n += np.sum(sum_terms)
                # add to lambda return
                if n == T - t:
                    # final term gets accumulated weights of geometric sum
                    return_ld_t += np.power(ld, T - t) * return_t_n
                else:
                    return_ld_t += (1 - ld) * np.power(ld, n) * return_t_n
            # write result to array
            result_arr[player, t] = return_ld_t
    return result_arr
