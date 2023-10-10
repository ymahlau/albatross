import itertools

import numpy as np


def smooth_best_response_from_q(
        q_values: np.ndarray,  # shape (num_actions,)
        temperature: float,
) -> np.ndarray:
    exp_term = np.exp(temperature * q_values)
    exp_term /= np.sum(exp_term)
    return exp_term


def best_response_from_q(
        q_values: np.ndarray,  # shape (num_actions,)
) -> np.ndarray:
    max_idx = np.argmax(q_values)
    arr = np.zeros_like(q_values, dtype=float)
    arr[max_idx] = 1
    return arr


def values_from_policies(
        individual_policies: list[np.ndarray],
        ja_action_values: np.ndarray,  # the array needs to be ordered according to itertools.product
) -> np.ndarray:
    grouped = itertools.product(*individual_policies)
    grouped_arr = np.asarray(list(grouped), dtype=float)
    joint_pol = np.prod(grouped_arr, axis=1)[:, np.newaxis]
    values = ja_action_values * joint_pol
    expected_vals = np.sum(values, axis=0)
    return expected_vals
