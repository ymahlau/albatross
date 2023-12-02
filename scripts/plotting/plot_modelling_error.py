import pickle
from pathlib import Path

import numpy as np
from rliable import library as rly
import scipy
import seaborn
from matplotlib import pyplot as plt
from rliable import metrics

from src.depth.depth_analysis import estimate_strength_at_depth_no_repeat
from src.depth.equilibria_analysis import compare_equilibria
from src.depth.result_struct import DepthResultStruct
from src.misc.plotting import plot_filled_std_curves


def estimate_strength():
    path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'depth' / 'd7_area'
    result_struct = DepthResultStruct.from_file(path / 'depth_d7_all.npz')

    arr = estimate_strength_at_depth_no_repeat(
        results=result_struct,
        min_mle_temp=0,
        max_mle_temp=50,
        mle_iterations=9,
        sample_resolution=1,
        min_sample_temp=1,
        max_sample_temp=1,
        ground_truth=None,
    )
    with open(path / 'temperature_estimates.pkl', 'wb') as f:
        pickle.dump(arr, f)


def plot_strength():
    prefix = 'nd7'
    # prefix = 'd7'
    path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'depth' / f'{prefix}_area'
    with open(path / 'temperature_estimates.pkl', 'rb') as f:
        estimates = pickle.load(f)
    estimate_arr = estimates['area'][:, 0]

    x = np.arange(10, 10001, 25)
    colors = list(seaborn.color_palette('colorblind', n_colors=10))
    light_colors = list(seaborn.color_palette('pastel', n_colors=10))

    plt.clf()
    seaborn.set_theme(style='whitegrid')
    plt.figure(dpi=600)
    plt.tight_layout()

    fontsize = 'x-large'
    plt.plot(x, estimate_arr, color=colors[0])
    plt.xlabel('Search Iterations', fontsize=fontsize)
    plt.ylabel('Temperature Estimate', fontsize=fontsize)
    plt.xlim(x[0], x[-1])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(path / f'{prefix}_strength.pdf')
    plt.show()


def compute_equilibria():
    # path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'depth' / 'nd7_area'
    path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'nd7_depth_area'

    with open(path / 'temperature_estimates.pkl', 'rb') as f:
        estimates = pickle.load(f)
    estimate_arr = estimates['area'][:, 0]

    struct = DepthResultStruct.from_file(path / 'depth_nd7_all.npz')
    entry = struct.results['area']

    error_arr = compare_equilibria(
        entry=entry,
        temperature_estimates=estimate_arr,
        chunk_size=50,
        num_processes=50,
        restrict_cpu=True,
        use_kl=False,
    )
    with open(path / 'errors.pkl', 'wb') as f:
        pickle.dump(error_arr, f)


def plot_modelling_error():
    prefix = 'nd7'
    # prefix = 'd7'
    path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'depth' / f'{prefix}_area'
    data_path = path / f'{prefix}_errors_area_mse.pkl'
    img_path = path / f'{prefix}_errors_area_mse.pdf'

    with open(data_path, 'rb') as f:
        arr = pickle.load(f)

    # iqmean = scipy.stats.trim_mean(arr, proportiontocut=0.25, axis=1)
    # upper = np.quantile(arr, 0.75, axis=1)
    # lower = np.quantile(arr, 0.25, axis=1)
    def iqm(scores):
        return np.array([
            metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])
        ])

    d = {
        'QSE': arr[2][:, np.newaxis, :],
        'QNE': arr[1][:, np.newaxis, :],
        'LE': arr[0][:, np.newaxis, :],
    }
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        d,
        iqm,
        reps=100,
    )

    # arr = np.load(str(data_path))
    x = np.arange(10, 10001, 25)
    colors = list(seaborn.color_palette('colorblind', n_colors=10))
    light_colors = list(seaborn.color_palette('pastel', n_colors=10))

    plt.clf()
    seaborn.set_theme(style='whitegrid')
    plt.figure(dpi=600)
    plt.tight_layout()

    # qne = arr[1]
    # qne_mean = np.mean(qne, axis=0)
    # qne_std = np.std(qne, axis=0)
    plot_filled_std_curves(
        x=x,
        mean=iqm_scores['QNE'],
        lower=iqm_cis['QNE'][0],
        upper=iqm_cis['QNE'][1],
        # mean=iqmean[1],
        # lower=lower[1],
        # upper=upper[1],
        color=colors[0],
        lighter_color=light_colors[0],
        label='QNE',
        min_val=0,
    )
    # plt.plot(x, np.median(qne, axis=0), color='xkcd:blue')
    # plt.plot(x, np.quantile(qne, 0.9, axis=0), color='xkcd:blue')
    # plt.plot(x, np.quantile(qne, 0.1, axis=0), color='xkcd:blue')

    # qse = arr[2]
    # qse_mean = np.mean(qse, axis=0)
    # qse_std = np.std(qse, axis=0)
    plot_filled_std_curves(
        x=x,
        mean=iqm_scores['QSE'],
        lower=iqm_cis['QSE'][0],
        upper=iqm_cis['QSE'][1],
        # mean=iqmean[2],
        # lower=lower[2],
        # upper=upper[2],
        color=colors[1],
        lighter_color=light_colors[1],
        label='QSE',
        min_val=0,
    )
    # plt.plot(x, np.median(qse, axis=0), color='xkcd:green')
    # plt.plot(x, np.quantile(qse, 0.9, axis=0), color='xkcd:green')
    # plt.plot(x, np.quantile(qse, 0.1, axis=0), color='xkcd:green')

    # le
    # le = arr[0]
    # le_mean = np.mean(le, axis=0)
    # le_std = np.std(le, axis=0)
    plot_filled_std_curves(
        x=x,
        mean=iqm_scores['LE'],
        lower=iqm_cis['LE'][0],
        upper=iqm_cis['LE'][1],
        # mean=iqmean[0],
        # lower=lower[0],
        # upper=upper[0],
        color=colors[2],
        lighter_color=light_colors[2],
        label='LE',
        min_val=0,
    )
    # plt.plot(x, np.median(le, axis=0), color='xkcd:black')
    # plt.plot(x, np.quantile(le, 0.9, axis=0), color='xkcd:black')
    # plt.plot(x, np.quantile(le, 0.1, axis=0), color='xkcd:black')

    fontsize = 'x-large'
    plt.legend(fontsize=fontsize)
    # plt.ylim(0, 0.04)
    plt.xlabel('Search Iterations', fontsize=fontsize)
    plt.ylabel('IQM of MSE', fontsize=fontsize)
    plt.xlim(x[0], x[-1])
    plt.xticks([0, 2000, 4000, 6000, 8000, 10000], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.title('Modelling Errors at depths, Lower curves are median')
    plt.tight_layout()
    plt.savefig(img_path)
    plt.show()


if __name__ == '__main__':
    # estimate_strength()
    # plot_strength()
    # compute_equilibria()
    plot_modelling_error()
