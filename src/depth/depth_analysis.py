import math
from datetime import datetime
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.depth.depth_parallel import DepthResultStruct
from src.game.actions import sample_individual_actions
from src.modelling.mle import compute_temperature_mle
from src.search.utils import action_indices_from_mask, q_list_from_mask


def estimate_argmax_error_percentage(
        results: DepthResultStruct
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:  # return name -> depth and error at depth (shape num_iter/k)
    result_dict = {}
    for name, entry in results.results.items():
        all_argmax = np.argmax(entry.policies, axis=-1)
        gt_argmax = all_argmax[:, -1]
        errors = all_argmax != gt_argmax[:, np.newaxis]
        error_percentage = np.mean(errors, axis=0)
        error_std = np.std(errors, axis=0)
        depth_axis = np.arange(entry.k, entry.k * entry.policies.shape[1] + 1, entry.k)
        result_dict[name] = (depth_axis, error_percentage, error_std)
    return result_dict


def estimate_kl_div_at_depth(
        results: DepthResultStruct,
) -> dict[str, tuple[np.ndarray, np.ndarray]] :  # return name -> depth and kl-div at depth
    result_dict = {}
    for name, entry in results.results.items():
        targets = entry.policies[:, -1, :]
        data = entry.policies.transpose((1, 0, 2))
        # smooth targets and data a little to avoid numerical and undefined errors
        offset = 1e-6
        smoothed_targets = targets + offset
        smoothed_targets /= smoothed_targets.sum(axis=-1)[..., np.newaxis]
        smoothed_data = data + offset
        smoothed_data /= smoothed_data.sum(axis=-1)[..., np.newaxis]
        # compute kl
        fraction = smoothed_data / smoothed_targets[np.newaxis, :, :]
        if np.isnan(fraction).any():  # sanity check
            raise Exception(f"Nan value in fraction of KL-divergence")
        log_arr = np.log(fraction)
        full_term = smoothed_data * log_arr
        kl = full_term.sum(axis=-1)
        kl = kl.transpose(1, 0)
        if np.isnan(kl).any() or np.isinf(kl).any():
            raise Exception(f"Nan or inf in kl divergence")
        mean_kl = np.mean(kl, axis=0)
        depth_axis = np.arange(entry.k, entry.k * entry.policies.shape[1] + 1, entry.k)
        result_dict[name] = (depth_axis, mean_kl)
    return result_dict


def distances_at_depth(
        results: DepthResultStruct,
        depth: int,
) -> tuple[list[str], np.ndarray]:  # names indexing rows/columns and 2d-distance matrix
    names = list(results.results.keys())
    policies = np.asarray([results.results[name].policies[:, depth, :] for name in names])
    # squared distance
    difference = policies[np.newaxis, :, :] - policies[:, np.newaxis, :]
    squared_difference = np.square(difference)
    distance_sum = np.sum(squared_difference, axis=-1)
    distance_sum[distance_sum < 0.0001] = 0  # avoid numerical issues
    distances = np.sqrt(distance_sum)
    avg_distances = np.mean(distances, axis=-1)
    if np.any(np.isnan(avg_distances)) or np.any(np.isinf(avg_distances)):
        raise ValueError(f"Error, this should never happen")
    return names, avg_distances


def estimate_strength_at_depth(
        results: DepthResultStruct,
        min_mle_temp: float,
        max_mle_temp: float,
        mle_iterations: int,
        sample_resolution: int,
        min_sample_temp: float,
        repeat_size: int = 1000,
        max_sample_temp: Optional[float] = None,
        ground_truth: Optional[str] = None,
) -> dict[str, np.ndarray]:  # shape (num_iter, resolution)
    # sanity check: illegal actions need to have zero prob
    result_dict = {}
    if sample_resolution > 1:
        if max_sample_temp is None:
            raise ValueError(f"Need minimum and maximum boundary on sampling if resolution is specified")
        sample_temperatures = np.linspace(min_sample_temp, max_sample_temp, sample_resolution)
    else:
        sample_temperatures = [min_sample_temp]
    gt_q_list = None
    if ground_truth is not None:
        entry = results.results[ground_truth]
        gt_q_list = q_list_from_mask(entry.q_values[:, -1, :], results.legal_actions)
    for name, entry in results.results.items():
        print(f"{datetime.now()} - Started computation for {name}", flush=True)
        num_games = entry.policies.shape[0]
        num_iter = entry.policies.shape[1]
        # convert q-values to list
        if ground_truth is None:
            gt_q_list = q_list_from_mask(entry.q_values[:, -1, :], results.legal_actions)
        # compute results
        result_arr = np.empty((num_games, num_iter, sample_resolution), dtype=float)
        for g_idx in range(num_games):
            print(f"{datetime.now()} - Game position {g_idx} / {num_games}", flush=True)
            cur_utils = gt_q_list[g_idx]
            full_util_list = [cur_utils for _ in range(repeat_size)]
            # legal
            cur_legal_mask = results.legal_actions[g_idx]
            cur_invalid_mask = np.logical_not(cur_legal_mask)
            for i in range(num_iter):
                # print(f"{datetime.now()} - Iteration {i=} / {num_iter}", flush=True)
                cur_policy = entry.policies[g_idx, i]
                for t_idx, t in enumerate(sample_temperatures):
                    # apply temperature and normalize
                    temp_probs = np.power(cur_policy, t)
                    temp_probs[cur_invalid_mask] = 0
                    prob_sum = np.sum(temp_probs, axis=-1)
                    if np.any(prob_sum == 0):
                        raise Exception("Cannot normalize a probability distribution with zero values")
                    norm_probs = temp_probs / prob_sum[..., np.newaxis]
                    # generate chosen actions for mle
                    end_indices = np.round(np.cumsum(norm_probs * repeat_size)).astype(int)
                    filtered = end_indices[cur_legal_mask]
                    action_list = []
                    start_idx = 0
                    for a, end_idx in enumerate(filtered):
                        cur_size = end_idx - start_idx
                        action_list += [a for _ in range(cur_size)]
                        start_idx = end_idx
                    if len(action_list) != repeat_size:
                        raise Exception(f"This should never happen, {len(action_list)=}, {repeat_size=}")
                    mle_estimate = compute_temperature_mle(
                        min_temp=min_mle_temp,
                        max_temp=max_mle_temp,
                        num_iterations=mle_iterations,
                        chosen_actions=action_list,
                        utilities=full_util_list,
                    )
                    result_arr[g_idx, i, t_idx] = mle_estimate
        result_dict[name] = result_arr
    return result_dict


def estimate_strength_at_depth_no_repeat(
        results: DepthResultStruct,
        min_mle_temp: float,
        max_mle_temp: float,
        mle_iterations: int,
        sample_resolution: int,
        min_sample_temp: float,
        max_sample_temp: Optional[float] = None,
        ground_truth: Optional[str] = None,
) -> dict[str, np.ndarray]:  # shape (num_iter, resolution)
    # sanity check: illegal actions need to have zero prob
    result_dict = {}
    invalid_mask = np.logical_not(results.legal_actions)
    if sample_resolution > 1:
        if max_sample_temp is None:
            raise ValueError(f"Need minimum and maximum boundary on sampling if resolution is specified")
        sample_temperatures = np.linspace(min_sample_temp, max_sample_temp, sample_resolution)
    else:
        sample_temperatures = [min_sample_temp]
    gt_q_list = None
    if ground_truth is not None:
        entry = results.results[ground_truth]
        gt_q_list = q_list_from_mask(entry.q_values[:, -1, :], results.legal_actions)
    for name, entry in results.results.items():
        print(f"{datetime.now()} - Started computation for {name}", flush=True)
        num_iter = entry.policies.shape[1]
        # convert q-values to list
        if ground_truth is None:
            gt_q_list = q_list_from_mask(entry.q_values[:, -1, :], results.legal_actions)
        # compute results
        result_arr = np.empty((num_iter, sample_resolution), dtype=float)
        for i in tqdm(range(num_iter)):
            # print(f"{datetime.now()} - Iteration {i=} / {num_iter}", flush=True)
            cur_probs = entry.policies[:, i, :]
            for t_idx, t in enumerate(sample_temperatures):
                # print(f"{datetime.now()} - {t_idx} / {len(sample_temperatures)}", flush=True)
                # sample actions given the action probs
                action_tuple = sample_individual_actions(cur_probs, t, invalid_mask)
                action_arr = np.asarray(action_tuple, dtype=int)
                # convert to action indices
                action_idx_arr = action_indices_from_mask(action_arr, invalid_mask)
                if np.max(action_idx_arr) >= results.legal_actions.shape[-1] or np.min(action_idx_arr) < 0:
                    raise Exception(f"Invalid action index sampled. Probably illegal actions has nonzero-prob.")
                # estimate temperature
                mle_estimate = compute_temperature_mle(
                    min_temp=min_mle_temp,
                    max_temp=max_mle_temp,
                    num_iterations=mle_iterations,
                    chosen_actions=list(action_idx_arr),
                    utilities=gt_q_list,
                )
                result_arr[i, t_idx] = mle_estimate
        result_dict[name] = result_arr
    return result_dict

