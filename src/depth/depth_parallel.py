import copy
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import multiprocessing as mp

from src.depth.result_struct import DepthResultStruct, joint_action_from_struct, aggregate_structs, DepthResultEntry
from src.game import GameConfig, Game
from src.game.initialization import get_game_from_config, game_config_from_structured
from src.game.utils import step_with_draw_prevention
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_file
from src.search import Search, SearchConfig
from src.search.initialization import get_search_from_config, search_config_from_structured
from src.search.utils import compute_q_values_as_arr, ja_value_array

# global variables that need to be initialized by init function in each process
global_search_dict: Optional[dict[str, Search, int, int]] = None


def compute_different_depths(
        game: Game,
        episode: int,
        turn: int,
        include_values: bool,
        include_policies: bool,
        include_q_values: bool,
        include_obs: bool,
        include_ja_values: bool,
) -> DepthResultStruct:  # sample size: num_player at turn
    player_turn = game.players_at_turn()
    # iterate through all searches
    entry_dict = {}
    for name, v in global_search_dict.items():
        # print(f"{datetime.now()} - Starting search: {name}", flush=True)
        search, iterations, save_every_k = v
        values, policies, q_list = [], [], []
        # search with different depths
        for i in range(save_every_k, iterations + 1, save_every_k):
            # search
            cur_values, cur_policies, _ = search(
                game=game,
                iterations=i,
            )
            # sanity check
            if search.root is None:
                raise Exception("root is None, this should never happen")
            # add to result lists
            if include_values:
                values.append(cur_values[player_turn])
            if include_policies:
                policies.append(cur_policies)
            if include_q_values:
                cur_q_list = []
                for player in game.players_at_turn():
                    qs = compute_q_values_as_arr(search.root, player, cur_policies)
                    cur_q_list.append(qs)
                cur_q_arr = np.asarray(cur_q_list, dtype=float)
                q_list.append(cur_q_arr)
        ja_values, ja_actions = None, None
        if include_ja_values:
            ja_values, ja_actions = ja_value_array(search.root)
        result_entry = DepthResultEntry(
            k=np.asarray(save_every_k, dtype=int),
            values=np.stack(values, axis=1) if include_values else None,
            policies=np.stack(policies, axis=1) if include_policies else None,
            q_values=np.stack(q_list, axis=1) if include_q_values else None,
            ja_values=ja_values,
            ja_actions=ja_actions,
        )
        # add entry and reset search stats
        search.cleanup_root()
        entry_dict[name] = result_entry
    # observation
    obs = None
    if include_obs:
        obs_tensor, _, _ = game.get_obs(0)
        obs = obs_tensor.numpy()
    # legal actions filter
    legal_actions = np.zeros(shape=(game.num_players_at_turn(), game.num_actions), dtype=bool)
    for player_idx, player in enumerate(game.players_at_turn()):
        for action in game.available_actions(player):
            legal_actions[player_idx, action] = True
    # result struct
    result = DepthResultStruct(
        episode=np.asarray([episode for _ in player_turn], dtype=int),
        turn=np.asarray([turn for _ in player_turn], dtype=int),
        player=np.asarray(game.players_at_turn(), dtype=int),
        game_length=None,
        results=entry_dict,
        legal_actions=legal_actions,
        obs=obs,
    )
    return result


def compute_steps_async(
        game_cfg: GameConfig,
        num_samples: int,
        proc_id: int,
        step_temperature: float,
        step_iterations: Optional[int],  # if none, use gt result
        step_search: Optional[str],
        draw_prevention: bool,
        seed: int,
        include_values: bool,
        include_policies: bool,
        include_q_values: bool,
        include_ja_values: bool,
        include_obs: bool,
        min_available_actions: int,
        max_available_actions: int,
) -> DepthResultStruct:
    # set seed (important to prevent all workers doing the exact same work)
    set_seed(seed)
    # computes num_steps observations
    sample_counter = 0
    turn_counter = 0
    game = get_game_from_config(game_cfg)
    result_list = []
    episode_result_list = []
    # offset for episode id: Assume worst case that all episodes end after a single step
    episode_offset = num_samples * proc_id + 1
    episode_counter = episode_offset
    while sample_counter < num_samples:
        while not game.is_terminal() and sample_counter < num_samples:
            # test if min/max available actions is fulfilled
            valid = True
            for player in game.players_at_turn():
                num_aa = len(game.available_actions(player))
                if num_aa < min_available_actions or num_aa > max_available_actions:
                    valid = False
                    break
            # if at least one player does not fulfill requirements, make random step and do not search
            if not valid:
                game.play_random_steps(1)
                turn_counter += 1
                continue
            # compute results at different depths of the tree search
            cur_result = compute_different_depths(
                game=game,
                episode=episode_counter,
                turn=turn_counter,
                include_values=include_values,
                include_policies=include_policies,
                include_obs=include_obs,
                include_q_values=include_q_values,
                include_ja_values=include_ja_values,
            )
            episode_result_list.append(cur_result)
            sample_counter += game.num_players_at_turn()
            # do a step
            joint_action = joint_action_from_struct(
                struct=cur_result,
                game=game,
                step_iterations=step_iterations,
                step_temperature=step_temperature,
                step_search=step_search,
            )
            print(f"{datetime.now()} - Process {proc_id} computed {sample_counter} / {num_samples}", flush=True)
            if draw_prevention:
                step_with_draw_prevention(game, joint_action)
            else:
                game.step(joint_action)
            turn_counter += 1
        # aggregate results from this episode
        episode_agg = aggregate_structs(episode_result_list)
        example_k = list(global_search_dict.keys())[0]
        game_lengths = np.ones((episode_agg.results[example_k].values.shape[0],), dtype=int) * turn_counter
        episode_agg.game_length = game_lengths
        result_list.append(episode_agg)
        # reset variables
        game.reset()
        episode_result_list = []
        episode_counter += 1
        turn_counter = 0
        # print info
        print(f"{datetime.now()} - Process {os.getpid()} has computed {sample_counter} / {num_samples}", flush=True)
    result = aggregate_structs(result_list)
    return result


def init_process(
        search_specs: dict[str, tuple[SearchConfig, Optional[str], int, int]],
        device_str: str,
        cpu_list: Optional[list[int]],
):
    # important for multiprocessing
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    device = torch.device(device_str)
    # init search
    global global_search_dict
    global_search_dict = {}
    for k, v in search_specs.items():
        search = get_search_from_config(v[0])
        if v[1] is not None:
            net = get_network_from_file(Path(v[1]))
            search.replace_net(net)
        search.replace_device(device)
        global_search_dict[k] = (search, v[2], v[3])
    pid = os.getpid()
    if cpu_list is not None:
        os.sched_setaffinity(pid, cpu_list)
        print(f"{datetime.now()} - Started computation with pid {pid} using restricted cpus: "
              f"{os.sched_getaffinity(pid)}", flush=True)
    else:
        print(f"{datetime.now()} - Started computation with pid {pid}", flush=True)

@dataclass()
class DepthSearchConfig:
    # map name to tuple of  (search_cfg, net_path, iterations, save_every_k)
    search_specs: dict[str, Any]
    game_cfg: GameConfig
    num_samples: int
    num_procs: int
    device_str: str
    step_temperature: float
    step_iterations: Optional[int]  # if none, use max iteration count
    step_search: Optional[str]  # if None, choose random one of tree search results
    draw_prevention: bool
    save_path: Optional[str]
    restrict_cpu: bool
    seed: Optional[int]
    include_values: bool
    include_policies: bool
    include_q_values: bool
    include_ja_values: bool
    include_obs: bool
    min_available_actions: int  # inclusive bound
    max_available_actions: int  # inclusive bound

    def __post_init__(self):
        # sanity check
        # if self.num_samples % self.num_procs != 0:
        #     raise ValueError(f"Step count needs to be divisible by process count")
        for k, v in self.search_specs.items():
            if v[2] % v[3] != 0:
                raise ValueError(f"Iteration count needs to be divisible by k")

def depth_search_config_from_structured(cfg):
    search_spec_dict = {}
    for k, v in cfg.search_specs.items():
        search_spec_dict[k] = (
            search_config_from_structured(v[0]),
            v[1],
            v[2],
            v[3],
        )
    kwargs = dict(cfg)
    kwargs['search_specs'] = search_spec_dict
    kwargs['game_cfg'] = game_config_from_structured(cfg.game_cfg)
    result_cfg = DepthSearchConfig(**kwargs)
    return result_cfg


def compute_different_depths_parallel(
        cfg: DepthSearchConfig
) -> DepthResultStruct:
    if mp.get_start_method() != 'spawn':
        try:
            mp.set_start_method('spawn')  # this is important for using CUDA
        except RuntimeError:
            pass
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    # restrict cpu
    cpu_list = None
    if cfg.restrict_cpu:
        cpu_list = os.sched_getaffinity(0)
    if cfg.seed is not None:
        set_seed(cfg.seed)
    # initialization
    print(f"{datetime.now()} - Main process with pid {os.getpid()} uses {cpu_list=}", flush=True)
    init_args = (cfg.search_specs, cfg.device_str, cpu_list)
    # start pool
    with mp.Pool(processes=cfg.num_procs, initializer=init_process, initargs=init_args) as pool:
        result_list_unfinished = []
        kwargs = {
            'game_cfg': cfg.game_cfg,
            'num_samples': math.ceil(cfg.num_samples / cfg.num_procs),
            'step_temperature': cfg.step_temperature,
            'step_iterations': cfg.step_iterations,  # if none, use gt result
            'step_search': cfg.step_search,
            'draw_prevention': cfg.draw_prevention,
            'include_values': cfg.include_values,
            'include_policies': cfg.include_policies,
            'include_q_values': cfg.include_q_values,
            'include_ja_values': cfg.include_ja_values,
            'include_obs': cfg.include_obs,
            'min_available_actions': cfg.min_available_actions,
            'max_available_actions': cfg.max_available_actions,
        }
        # start child processes
        for child_id in range(cfg.num_procs):
            cur_kwargs = copy.deepcopy(kwargs)
            cur_kwargs['proc_id'] = child_id
            cur_kwargs['seed'] = random.randint(0, 2 ** 32 - 1)
            cur_result = pool.apply_async(
                func=compute_steps_async,
                kwds=cur_kwargs,
            )
            result_list_unfinished.append(cur_result)
        # wait for children to finish
        result_list = [res.get(timeout=None) for res in result_list_unfinished]
    # aggregate results
    full_results = aggregate_structs(result_list)
    # relabel the episodes: first find all episode labels and then relabel them
    label_counter = -1
    current_label = -1
    for idx in range(full_results.episode.shape[0]):
        item = full_results.episode[idx]
        if item != current_label:
            current_label = item
            label_counter += 1
        full_results.episode[idx] = label_counter
    # save results
    if cfg.save_path is not None:
        full_results.save(cfg.save_path)
    return full_results
