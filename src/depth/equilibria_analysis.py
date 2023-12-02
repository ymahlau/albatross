import os
import time
from datetime import datetime
import multiprocessing as mp
from typing import Optional

import numpy as np

from src.depth.result_struct import DepthResultEntry
from src.equilibria.quantal import compute_qne_equilibrium, compute_qse_equilibrium
from src.equilibria.logit import compute_logit_equilibrium
from src.misc.utils import kl_divergence, mse_diff


def compare_equilibria_async(
        entry: DepthResultEntry,
        temperature_estimates: np.ndarray,  # shape (num_iter/k)
        min_index: int,
        max_index: int,
        use_kl: bool,
) -> np.ndarray:  # model errors of shape (3, num_samples, num_iter/k)
    # computes le, qse and qne modelling error at different depths
    num_elements = max_index - min_index
    print(f"{datetime.now()} - {os.getpid()} is starting {min_index} - {max_index}", flush=True)
    le_errors = np.empty((num_elements, entry.policies.shape[1]), dtype=float)
    qse_errors = np.empty((num_elements, entry.policies.shape[1]), dtype=float)
    qne_errors = np.empty((num_elements, entry.policies.shape[1]), dtype=float)
    indices = list(range(min_index, max_index))
    for index_idx, i in enumerate(indices):  # iterate all samples
        aa, ja, vals = entry.get_aa_ja_and_values(index=i)
        num_p = vals.shape[-1]
        if num_p != 2:
            raise ValueError(f"Can only compare equilibria in games with two players")
        for iter_idx in range(entry.policies.shape[1]):  # iterate all iterations
            # print(f"{datetime.now()} - {iter_idx} / {entry.policies.shape[1]}", flush=True)
            # compute le
            le_vals, le_pol = compute_logit_equilibrium(
                available_actions=aa,
                joint_action_list=ja,
                joint_action_value_arr=vals,
                num_iterations=1000,
                epsilon=0,
                temperatures=[temperature_estimates[iter_idx] for _ in range(num_p)],
            )
            if use_kl:
                cur_le_error = kl_divergence(
                    data=le_pol[0],  # entry saves its data such that current player is always indexed first
                    target=entry.policies[i, iter_idx][aa[0]]
                )
            else:
                cur_le_error = mse_diff(
                    data=le_pol[0],
                    target=entry.policies[i, iter_idx][aa[0]]
                )
            le_errors[index_idx, iter_idx] = cur_le_error
            # compute qne
            qne_vals, qne_pol = compute_qne_equilibrium(
                available_actions=aa,
                joint_action_list=ja,
                joint_action_value_arr=vals,
                num_iterations=1000,
                temperature=temperature_estimates[iter_idx],
                leader=1,  # current player has bounded rationality, therefore other player is leader
            )
            if use_kl:
                cur_qne_error = kl_divergence(
                    data=qne_pol[0],  # entry saves its data such that current player is always indexed first
                    target=entry.policies[i, iter_idx][aa[0]]
                )
            else:
                cur_qne_error = mse_diff(
                    data=qne_pol[0],
                    target=entry.policies[i, iter_idx][aa[0]]
                )
            qne_errors[index_idx, iter_idx] = cur_qne_error
            # compute qse
            qse_vals, qse_pol = compute_qse_equilibrium(
                available_actions=aa,
                joint_action_list=ja,
                joint_action_value_arr=vals,
                num_iterations=9,
                grid_size=200,
                temperature=temperature_estimates[iter_idx],
                leader=1,  # current player has bounded rationality, therefore other player is leader
            )
            if use_kl:
                cur_qse_error = kl_divergence(
                    data=qse_pol[0],  # entry saves its data such that current player is always indexed first
                    target=entry.policies[i, iter_idx][aa[0]]
                )
            else:
                cur_qse_error = mse_diff(
                    data=qse_pol[0],
                    target=entry.policies[i, iter_idx][aa[0]]
                )
            qse_errors[index_idx, iter_idx] = cur_qse_error
    result = np.asarray([le_errors, qne_errors, qse_errors])
    return result


def init_process(
        cpu_list: Optional[list[int]],
):
    pid = os.getpid()
    if cpu_list is not None:
        os.sched_setaffinity(pid, cpu_list)
        print(f"{datetime.now()} - Started computation with pid {pid} using restricted cpus: "
              f"{os.sched_getaffinity(pid)}", flush=True)
    else:
        print(f"{datetime.now()} - Started computation with pid {pid}", flush=True)


def compare_equilibria(
        entry: DepthResultEntry,
        temperature_estimates: np.ndarray,  # shape (num_iter/k),
        chunk_size: int,
        num_processes: int,
        restrict_cpu: bool,
        use_kl: bool,
) -> np.ndarray:  # model errors of shape (3, num_samples, num_iter/k)
    num_samples = entry.policies.shape[0]
    print(f"{datetime.now()} - Started Equilibrium computation of size: {num_samples}", flush=True)
    chunk_starts = list(range(0, num_samples, chunk_size))
    chunk_ends = list(range(chunk_size, num_samples + chunk_size, chunk_size))
    if chunk_ends[-1] != num_samples:
        chunk_ends[-1] = num_samples
    cpu_list = None
    if restrict_cpu:
        cpu_list = os.sched_getaffinity(0)
    init_args = (cpu_list,)
    with mp.Pool(processes=num_processes, initializer=init_process, initargs=init_args) as pool:
        result_list = []
        for start, end in zip(chunk_starts, chunk_ends):
            cur_kwargs = {
                'entry': entry,
                'temperature_estimates': temperature_estimates,
                'min_index': start,
                'max_index': end,
                'use_kl': use_kl,
            }
            cur_result = pool.apply_async(
                func=compare_equilibria_async,
                kwds=cur_kwargs,
            )
            result_list.append(cur_result)
        final_result_list = [res.get(timeout=None) for res in result_list]
    arr = np.concatenate(final_result_list, axis=1)
    return arr

