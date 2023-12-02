import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from src.game.actions import sample_individual_actions
from src.game.initialization import get_game_from_config
from src.modelling.mle import compute_temperature_mle
from src.network.initialization import get_network_from_file
from src.search.config import FixedDepthConfig, LogitBackupConfig, SpecialExtractConfig, NetworkEvalConfig
from src.search.initialization import get_search_from_config
from src.search.utils import compute_q_values_as_arr, q_list_from_mask


def estimate_training_strength():
    save_path = Path(__file__).parent.parent.parent.parent / 'nobackup' / 'data' / 'strength' / 'of_1e4.pkl'
    run_dir = Path(__file__).parent.parent.parent.parent / 'nobackup' / 'runs' / 'of'
    run_names = [f'of_duct_{s}' for s in range(5)]
    min_mle_temp = 0
    max_mle_temp = 100
    mle_iterations = 10
    time_stamps = 70
    num_estimates = 10000
    repeat_size = 1000
    num_actions = 6

    search_cfg = FixedDepthConfig(
        backup_func_cfg=LogitBackupConfig(num_iterations=100, init_temperatures=[20 for _ in range(2)]),
        extract_func_cfg=SpecialExtractConfig(),
        eval_func_cfg=NetworkEvalConfig(),
        discount=1,
        average_eval=False
    )
    all_results: dict[str, np.ndarray] = {}
    for n in run_names:
        print(f"{datetime.now()} - Starting {n}", flush=True)
        net_dir = run_dir / n / 'fixed_time_models'
        cur_result_arr = np.empty((time_stamps, 2 * num_estimates), dtype=float)
        cur_result_counter = 0
        # init search and networks
        net_dict = {}
        game_cfg = None
        gt_net = get_network_from_file(net_dir / f'm_{time_stamps}.pt')
        gt_search = get_search_from_config(search_cfg)
        gt_search.replace_net(gt_net)
        for t in range(time_stamps):
            net = get_network_from_file(net_dir / f'm_{t}.pt')
            net = net.eval()
            net_dict[t] = net
            if t == 0:
                game_cfg = net.game.cfg
        # init game
        game = get_game_from_config(game_cfg)
        game.reset()
        # iterate samples
        while cur_result_counter < 2 * num_estimates:
            # test min available actions
            if game.is_terminal():
                game.reset()
            if any(len(game.available_actions(p)) < 2 for p in range(2)):
                ja = random.choice(game.available_joint_actions())
                game.step(ja)
                continue
            # masks
            cur_legal_list = [[a in game.available_actions(p) for a in range(num_actions)] for p in range(2)]
            cur_legal_masks = np.asarray(cur_legal_list, dtype=bool)
            cur_illegal_masks = np.logical_not(cur_legal_masks)
            # ground truth estimate
            print(f"{datetime.now()} - Currently at estimate {cur_result_counter} / {2 * num_estimates}", flush=True)
            obs, _, _ = game.get_obs()
            net_out = gt_net(obs)
            gt_logits = gt_net.retrieve_policy(net_out).detach().numpy()
            gt_probs = np.exp(gt_logits) / np.sum(np.exp(gt_logits), axis=-1)[..., np.newaxis]
            gt_probs[cur_illegal_masks] = 0
            gt_probs /= gt_probs.sum(axis=-1)[..., np.newaxis]
            gt_search(
                game=game,
                iterations=1,
            )
            gtq_p0 = compute_q_values_as_arr(
                node=gt_search.root,
                action_probs=gt_probs,
                player=0,
            )
            gtq_p1 = compute_q_values_as_arr(
                node=gt_search.root,
                action_probs=gt_probs,
                player=1,
            )

            q_lists = [
                q_list_from_mask(q_arr=gtq_p0[np.newaxis, ...], is_valid=cur_legal_masks[np.newaxis, 0])[0],
                q_list_from_mask(q_arr=gtq_p1[np.newaxis, ...], is_valid=cur_legal_masks[np.newaxis, 1])[0],
            ]
            full_util_lists = [
                [q_lists[p] for _ in range(repeat_size)] for p in range(2)
            ]
            # iterate time steps
            for t_idx, t in enumerate(range(time_stamps)):
                # network policy prediction
                net = net_dict[t]
                net_out = net(obs)
                cur_logits = net.retrieve_policy(net_out).detach().numpy()
                cur_pol = np.exp(cur_logits) / np.sum(np.exp(cur_logits), axis=-1)[..., np.newaxis]
                for p in range(2):
                    # filter policy
                    filtered = cur_pol[p][cur_legal_masks[p]]
                    filtered /= np.sum(filtered)
                    # generate chosen actions for mle
                    end_indices = np.round(np.cumsum(filtered * repeat_size)).astype(int)
                    action_list = []
                    start_idx = 0
                    for a, end_idx in enumerate(end_indices):
                        cur_size = end_idx - start_idx
                        action_list += [a for _ in range(cur_size)]
                        start_idx = end_idx
                    if len(action_list) != repeat_size:
                        raise Exception(f"This should never happen, {len(action_list)=}, {repeat_size=}, {end_indices=}"
                                        f", {cur_pol=}, {filtered=},")
                    mle_estimate = compute_temperature_mle(
                        min_temp=min_mle_temp,
                        max_temp=max_mle_temp,
                        num_iterations=mle_iterations,
                        chosen_actions=action_list,
                        utilities=full_util_lists[p],
                    )
                    cur_result_arr[t_idx, cur_result_counter + p] = mle_estimate
            cur_result_counter += 2
            ja = sample_individual_actions(gt_probs, 1)
            game.step(ja)
        all_results[n] = cur_result_arr
    # save results
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)

