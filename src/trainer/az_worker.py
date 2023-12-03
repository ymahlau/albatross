from collections import defaultdict
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from src.game.actions import sample_individual_actions, apply_permutation, filter_illegal_and_normalize
from src.game.game import Game
from src.game.initialization import get_game_from_config
from src.game.overcooked.overcooked import OvercookedGame
from src.game.utils import step_with_draw_prevention
from src.misc.replay_buffer import BufferInputSample
from src.misc.utils import set_seed
from src.search import SearchInfo, Search
from src.search.eval_func import InferenceServerEvalFunc, ResponseInferenceEvalFunc
from src.search.initialization import get_search_from_config
from src.search.utils import compute_q_values
from src.supervised.annealer import TemperatureAnnealer
from src.trainer.config import WorkerConfig, AlphaZeroTrainerConfig
from src.trainer.policy_eval import PolicyEvalType
from src.trainer.utils import send_obj_to_queue
import multiprocessing.sharedctypes as sc

@dataclass
class WorkerStatistics:
    step_counter: sc.Synchronized
    episode_counter: sc.Synchronized
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
    infos = []


def run_worker(
        worker_id: int,
        trainer_cfg: AlphaZeroTrainerConfig,
        data_queue: mp.Queue,
        stop_flag: sc.Synchronized,
        info_queue: mp.Queue,
        step_counter: sc.Synchronized,
        episode_counter: sc.Synchronized,
        error_counter: sc.Synchronized,
        input_rdy_arr,  # mp.Array
        output_rdy_arr,  # mp.Array
        input_arr,  # mp.Array
        output_arr,  # mp.Array
        cpu_list: Optional[list[int]],
        seed: int,
        debug: bool = False,
        input_rdy_arr2 = None,  # mp.Array
        output_rdy_arr2 = None,  # mp.Array
        input_arr2 = None,  # mp.Array
        output_arr2 = None,  # mp.Array
):
    game_cfg = trainer_cfg.game_cfg
    worker_cfg = trainer_cfg.worker_cfg
    set_seed(seed)
    # torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    # initialization
    search = get_search_from_config(worker_cfg.search_cfg)
    if hasattr(search, "backup_func"):
        if hasattr(search.backup_func, "error_counter"): # type: ignore
            search.backup_func.error_counter = error_counter # type: ignore        
    # add inference server arrays to evaluation function of search
    game = get_game_from_config(trainer_cfg.game_cfg)
    obs_shape = game.get_obs_shape()
    out_shape = 1 + game.num_actions if trainer_cfg.net_cfg.predict_policy else 1
    worker_per_server = int(trainer_cfg.num_worker / trainer_cfg.num_inference_server)
    start_idx = (worker_id % worker_per_server) * trainer_cfg.max_eval_per_worker
    n = int(trainer_cfg.max_eval_per_worker * trainer_cfg.num_worker / trainer_cfg.num_inference_server)
    input_shape, output_shape = (n, *obs_shape), (n, out_shape)
    input_arr_np = np.frombuffer(input_arr, dtype=np.float32).reshape(input_shape)
    output_arr_np = np.frombuffer(output_arr, dtype=np.float32).reshape(output_shape)
    input_rdy_arr_np = np.frombuffer(input_rdy_arr, dtype=np.int32)
    output_rdy_arr_np = np.frombuffer(output_rdy_arr, dtype=np.int32)
    if input_rdy_arr2 is None or output_rdy_arr2 is None or input_arr2 is None or output_arr2 is None:
        if not isinstance(search.eval_func, InferenceServerEvalFunc):
            raise ValueError(f"Worker needs Inference Server Evaluation Function")
        # single inference server
        search.eval_func.update_arrays_and_indices(
            input_arr_np=input_arr_np,
            output_arr_np=output_arr_np,
            input_rdy_arr_np=input_rdy_arr_np,
            output_rdy_arr_np=output_rdy_arr_np,
            start_idx=start_idx,
            max_length=trainer_cfg.max_eval_per_worker,
            stop_flag=stop_flag,
        )
    else:
        if not isinstance(search.eval_func, ResponseInferenceEvalFunc):
            raise ValueError(f"Worker needs Inference Server Evaluation Function")
        # two inference server: first proxy and second response model
        input_arr_np2 = np.frombuffer(input_arr2, dtype=np.float32).reshape(input_shape)
        output_arr_np2 = np.frombuffer(output_arr2, dtype=np.float32).reshape(output_shape)
        input_rdy_arr_np2 = np.frombuffer(input_rdy_arr2, dtype=np.int32)
        output_rdy_arr_np2 = np.frombuffer(output_rdy_arr2, dtype=np.int32)
        search.eval_func.update_arrays_and_indices(
            input_arr_np=input_arr_np2,
            output_arr_np=output_arr_np2,
            input_rdy_arr_np=input_rdy_arr_np2,
            output_rdy_arr_np=output_rdy_arr_np2,
            input_arr_resp_np=input_arr_np,
            output_arr_resp_np=output_arr_np,
            input_rdy_arr_resp_np=input_rdy_arr_np,
            output_rdy_arr_resp_np=output_rdy_arr_np,
            start_idx=start_idx,
            max_length=trainer_cfg.max_eval_per_worker,
            stop_flag=stop_flag,
        )
    # info for logging
    stats = WorkerStatistics(step_counter=step_counter, episode_counter=episode_counter)
    # temperature scheduling
    annealer_list: Optional[list[TemperatureAnnealer]] = None
    if worker_cfg.anneal_cfgs is not None:
        if trainer_cfg.single_sbr_temperature:
            if len(worker_cfg.anneal_cfgs) != 1:
                raise ValueError(f"Please provide a single anneal_cfg when using single temperature")
            annealer_list = [TemperatureAnnealer(worker_cfg.anneal_cfgs[0])]
        else:
            annealer_list = [TemperatureAnnealer(cfg) for cfg in worker_cfg.anneal_cfgs]
    min_scaler, max_scaler = None, None
    if worker_cfg.temp_scaling_cfgs is not None:
        min_scaler = TemperatureAnnealer(worker_cfg.temp_scaling_cfgs[0])
        max_scaler = TemperatureAnnealer(worker_cfg.temp_scaling_cfgs[1])
    # restrict cpus
    pid = os.getpid()
    if cpu_list is not None:
        print(f"{datetime.now()} - CPU list in Worker: {cpu_list}")
        os.sched_setaffinity(pid, cpu_list)
        print(f'{datetime.now()} - Started Worker process {worker_id} with pid {pid} using cpus: '
              f'{os.sched_getaffinity(pid)}', flush=True)
    else:
        print(f'{datetime.now()} - Started Worker process {worker_id} with pid {pid}', flush=True)
    try:
        while not stop_flag.value:
            # for exploration, start game with a random number of random steps
            game.reset()
            num_random_steps = random.randint(0, worker_cfg.max_random_start_steps)
            game.play_random_steps(num_random_steps)
            episode_step_counter = num_random_steps
            # init
            game_list, value_list, reward_list, policy_list = [], [], [], []
            temp_list: list[list[float]] = []
            # parse options
            cur_temp_list = None
            if annealer_list is not None:
                time_passed = (time.time() - stats.process_start_time) / 60
                cur_min, cur_max = None, None
                if min_scaler is not None and max_scaler is not None:
                    cur_min, cur_max = min_scaler(time_passed), max_scaler(time_passed)
                if trainer_cfg.single_sbr_temperature:
                    t = annealer_list[0](time_passed)
                    if cur_min is not None and cur_max is not None:
                        t = t * (cur_max - cur_min) + cur_min
                    cur_temp_list = [t for _ in range(game_cfg.num_players)]
                else:
                    cur_temp_list = []
                    for idx in range(game_cfg.num_players):
                        t = annealer_list[idx](time_passed)
                        if cur_min is not None and cur_max is not None:
                            t = t * (cur_max - cur_min) + cur_min
                        cur_temp_list.append(t)
                search.set_temperatures(cur_temp_list)
            # response player
            resp_player = None
            if trainer_cfg.temperature_input and not trainer_cfg.single_sbr_temperature:
                # response model training: search result prob of one player and proxy prob of other players
                resp_player = random.choice(game.players_at_turn())
            # do search and send results
            while not game.is_terminal() and not stop_flag.value:
                if debug:
                    game.render()
                # search
                if cur_temp_list is not None:
                    temp_list.append(cur_temp_list)
                search_time_start = time.time()
                # do the search
                values, action_probs, info = search(
                    game=game,
                    iterations=worker_cfg.search_iterations,
                )
                # if worker_id == 0:
                #     stats.infos.append((values, action_probs, cur_temp_list))
                # if values[0] > 1.2:
                #     print(f"{values[0]:.2f}, {values[1]:.2f}, {action_probs=}".replace("\n", ""), flush=True)
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
                if resp_player is not None:
                    # response model training: search result prob of one player and proxy prob of other players
                    if resp_player in game.players_at_turn():
                        if search.root is None:
                            raise Exception("This should never happen")
                        player_at_turn_list = game.players_at_turn().copy()
                        action_list = [action_probs[resp_player]]
                        player_at_turn_list.remove(resp_player)
                        for p in player_at_turn_list:
                            action_list.append(search.root.info[f'p{p}'])
                        step_action_probs = np.asarray(action_list)
                    else:  # case: response player already died
                        step_action_probs = action_probs
                else:
                    step_action_probs = action_probs
                rewards = make_step(worker_cfg, game, step_action_probs, search)
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
                    temp_list = None # type: ignore
                sample = convert_training_data(game_list, value_list, reward_list, policy_list, trainer_cfg, temp_list)
                stats.data_conv_time_sum += time.time() - data_conv_time_start
                idle_time_start = time.time()
                send_obj_to_queue(sample, data_queue, stop_flag)
                stats.idle_time_sum += time.time() - idle_time_start
            # send info about episodes to logging process
            if len(stats.episode_len_list) == trainer_cfg.logger_cfg.worker_episode_bucket_size:
                time_passed = (time.time() - stats.process_start_time) / 60
                temps = None if not temp_list else random.choice(temp_list)
                send_info(stats, info_queue, stop_flag, temps)
                if stop_flag.value:
                    break
                # if worker_id == 0 and stats.infos:
                #     with open("/tmp/infos.pkl", 'wb') as f:
                #         pickle.dump(stats.infos, f)
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
        if search.root is None:
            raise Exception("search root is None")
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
        stop_flag: sc.Synchronized,
        temps: Optional[list[float]],
):
    full_time = time.time() - stats.last_info_time
    other_time = full_time - stats.search_time_sum - stats.idle_time_sum - stats.step_time_sum
    other_time -= stats.data_conv_time_sum
    msg_data = {
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
    value_tensor_list, policy_tensor_list, obs_tensor_list = [], [], []
    turns_list, player_list, symmetry_list = [], [], []
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
        # iterate symmetries
        sims = [0] if not trainer_cfg.worker_cfg.use_symmetries else list(range(game.get_symmetry_count()))
        for symmetry in sims:
            temp_obs_input = None
            if trainer_cfg.temperature_input:
                if temperature_list is None:
                    raise Exception("temperature list is None")
                if trainer_cfg.single_sbr_temperature:
                    temp_obs_input = [temperature_list[counter][0]]
                else:
                    temp_obs_input = temperature_list[counter]
            # shape (num_players_at_turn, *obs_shape)
            obs, perm, _ = game.get_obs(symmetry=symmetry, temperatures=temp_obs_input)
            # apply permutation to action probs
            perm_action = apply_permutation(action_probs, perm)
            # iterate players
            for player_idx, player in enumerate(game.players_at_turn()):
                # value, policy and obs are tensors
                obs_tensor_list.append(obs[player_idx, ...])
                value_tensor_list.append(values[player])
                policy_tensor_list.append(perm_action[player_idx, :])
                # turn, player and symmetry are not
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
    full_turns = np.asarray(turns_list, dtype=int)[:, np.newaxis]
    full_players = np.asarray(player_list, dtype=int)[:, np.newaxis]
    full_symmetry = np.asarray(symmetry_list, dtype=int)[:, np.newaxis]
    sample = BufferInputSample(
        obs=full_obs,
        values=full_values,
        policies=full_policies,
        turns=full_turns,
        player=full_players,
        symmetry=full_symmetry,
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
