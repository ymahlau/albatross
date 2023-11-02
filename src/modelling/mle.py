import ctypes as ct

import numpy as np

from src.cpp.lib import CPP_LIB


def compute_temperature_mle(
        min_temp: float,
        max_temp: float,
        num_iterations: int,
        chosen_actions: list[int],
        utilities: list[list[float]],
        use_line_search: bool = True,
) -> float:
    # number of time steps
    t = len(chosen_actions)
    # number of available actions
    num_aa = [len(u) for u in utilities]
    num_aa_arr = np.asarray(num_aa, dtype=ct.c_int)
    num_aa_p = num_aa_arr.ctypes.data_as(ct.POINTER(ct.c_int))
    # chosen actions
    chosen_actions_arr = np.asarray(chosen_actions, dtype=ct.c_int)
    chosen_actions_p = chosen_actions_arr.ctypes.data_as(ct.POINTER(ct.c_int))
    # utilities
    util_flat = [item for sublist in utilities for item in sublist]
    util_arr = np.asarray(util_flat, dtype=ct.c_double)
    util_p = util_arr.ctypes.data_as(ct.POINTER(ct.c_double))
    # c++ call
    temperature = CPP_LIB.lib.mle_temperature_cpp(
        min_temp,
        max_temp,
        num_iterations,
        t,
        chosen_actions_p,
        num_aa_p,
        util_p,
        use_line_search,
    )
    return temperature


def compute_likelihood(
        temperature: float,
        chosen_actions: list[int],
        utilities: list[list[float]],
) -> float:
    if len(utilities) != len(chosen_actions):
        raise ValueError(f"Invalid argument format: {len(utilities)} != {len(chosen_actions)}")
    # number of time steps
    t = len(chosen_actions)
    # number of available actions
    num_aa = [len(u) for u in utilities]
    num_aa_arr = np.asarray(num_aa, dtype=ct.c_int)
    num_aa_p = num_aa_arr.ctypes.data_as(ct.POINTER(ct.c_int))
    # chosen actions
    chosen_actions_arr = np.asarray(chosen_actions, dtype=ct.c_int)
    chosen_actions_p = chosen_actions_arr.ctypes.data_as(ct.POINTER(ct.c_int))
    # utilities
    util_flat = [item for sublist in utilities for item in sublist]
    util_arr = np.asarray(util_flat, dtype=ct.c_double)
    util_p = util_arr.ctypes.data_as(ct.POINTER(ct.c_double))
    # c++ call
    likelihood = CPP_LIB.lib.temperature_likelihood_cpp(
        temperature,
        t,
        chosen_actions_p,
        num_aa_p,
        util_p,
    )
    return likelihood


def compute_all_likelihoods(
        chosen_actions: list[int],
        utilities: list[list[float]],
        min_temp: float,
        max_temp: float,
        resolution: int,
) -> list[float]:
    temps = np.linspace(min_temp, max_temp, resolution)
    likelihoods = []
    for t in range(resolution):
        cur_l = compute_likelihood(
            temperature=temps[t].item(),
            chosen_actions=chosen_actions,
            utilities=utilities,
        )
        likelihoods.append(cur_l)
    return likelihoods
