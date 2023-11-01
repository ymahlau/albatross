import random
from typing import Any, Optional

import numpy as np
import torch

from src.cpp.lib import CPP_LIB


def flatten_dict_rec(
        d: dict[str, Any],
        sep: str = ".",
        prefix: Optional[str] = None
) -> dict[str, Any]:
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            child_prefix = k if prefix is None else f"{prefix}{sep}{k}"
            child_dict = flatten_dict_rec(d=v, sep=sep, prefix=child_prefix)
            for child_k, child_v in child_dict.items():
                result[child_k] = child_v
        else:
            new_key = k if prefix is None else f"{prefix}{sep}{k}"
            result[new_key] = v
    return result


def set_seed(seed: int):
    CPP_LIB.lib.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def softmax_weighting(
        action_vals: np.ndarray,  # shape (num_actions,)
        temperature: float,
) -> np.ndarray:
    if len(action_vals.shape) != 1:
        raise ValueError(f"Invalid array shape: {action_vals}")
    weighted = np.exp(temperature * action_vals)
    normalized = weighted / np.sum(weighted)
    return normalized


def random_argmax(
        arr,  # array of shape (n,)
) -> int:  # scalar argmax
    max_val = np.max(arr)
    max_indices = np.where(arr == max_val)[0]
    random_index = np.random.choice(max_indices)
    return random_index


def multi_dim_choice(
        probs: np.ndarray,  # shape (n, num_items)
) -> np.ndarray:  # shape (n,)
    # draws random choice from the given probabilities. Returns the indices of chosen items
    cum_prob = np.cumsum(probs, axis=-1)
    samples = np.random.rand(probs.shape[0])[..., np.newaxis]
    below_bound = cum_prob < samples
    result_indices = np.sum(below_bound, axis=-1)
    return result_indices


def replace_keys(
        results: dict[str, Any],
        keymap: dict[str, str],
):
    replaced = {}
    for k, v in keymap.items():
        replaced[v] = results[k]
    return replaced


def symmetric_cartesian_product(
        name_list: list[str],
        players_per_game: int,
) -> list[tuple[str, ...]]:
    if len(name_list) < players_per_game:
        raise ValueError(f"Too few names for game")
    # computes symmetric cartesian product (ignores all symmetric and diagonal entries)
    sym_name_list: list[tuple[str, ...]] = []
    # first compute all tuples of names
    positions: list[int] = [0 for _ in range(players_per_game - 1)] + [1]
    while True:
        # add tuple at current position to product list
        cur_list: list[str] = []
        for pos in positions:
            cur_list.append(name_list[pos])
        sym_name_list.append(tuple(cur_list))
        # test which positions require update starting from last
        idx_to_reset = None
        for idx in range(players_per_game - 1, -1, -1):
            positions[idx] += 1
            if positions[idx] < len(name_list):
                break
            else:
                idx_to_reset = idx
        # stopping criterion
        if positions[0] == len(name_list) - 1:
            break
        # reset all positions after idx_to_reset
        if idx_to_reset is not None:
            reset_position = positions[idx_to_reset - 1]
            for idx in range(idx_to_reset, players_per_game):
                # we do not want diagonal entries, so reset last index to position + 1
                if idx == players_per_game - 1 and idx_to_reset == 1:
                    positions[idx] = reset_position + 1
                else:
                    positions[idx] = reset_position
    return sym_name_list


def kl_divergence(
        data: np.ndarray,  # shape (n,)
        target: np.ndarray,  # shape (n,)
        epsilon: float = 1e-5,
) -> float:
    # smooth to avoid numerical issues and nan/inf
    target_smoothed = target + epsilon
    target_smoothed /= np.sum(target_smoothed)
    data_smoothed = data + epsilon
    data_smoothed /= np.sum(data_smoothed)
    kl = target_smoothed * np.log(target_smoothed / data_smoothed)
    # sanity check
    if np.any(np.isinf(kl)) or np.any(np.isnan(kl)):
        raise Exception(f"nan or in in kl divergence")
    avg_kl = np.mean(kl).item()
    return avg_kl

def mse_diff(
        data: np.ndarray,  # shape (n,)
        target: np.ndarray,  # shape (n,)
) -> float:
    diff = data - target
    squared = np.square(diff)
    avg_squared = np.mean(squared)
    mse = np.sqrt(avg_squared).item()
    return mse