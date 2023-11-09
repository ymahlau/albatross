import pickle
from pathlib import Path

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves


def get_data_list(prefix: str):
    data_list = []
    path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / prefix
    all_iterations = np.arange(50, 2001, 50)
    for cur_iterations in all_iterations:
        file_name = f'{prefix}_zs_{cur_iterations}.pkl'
        with open(path / file_name, 'rb') as f:
            cur_data = pickle.load(f)
        data_list.append(cur_data)
    return data_list

def plot_pol_errors():
    all_iterations = np.arange(50, 2001, 50)

    plt.clf()
    plt.figure(dpi=400)
    seaborn.set_theme(style='whitegrid')

    # methods = ['MSA', 'POLYAK', 'NAGURNEY']
    methods = ['MSA', 'POLYAK', 'NAGURNEY', 'SRA']

    for m_idx, m in enumerate(methods):
        data_list = get_data_list(m.lower())
        avg_errors = np.asarray([np.mean(d.errors) for d in data_list])
        std_errors = np.asarray([np.std(d.errors) for d in data_list])

        plot_filled_std_curves(
            x=all_iterations,
            mean=avg_errors,
            std=std_errors,
            color=COLORS[m_idx],
            lighter_color=LIGHT_COLORS[m_idx],
            linestyle=LINESTYLES[m_idx],
            label=m,
            min_val=0,
        )

    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Average Absolute Policy Error')
    plt.tight_layout()
    plt.xlim(all_iterations[0], all_iterations[-1])

    img_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'a_img' / 'zs_pol.png'
    plt.savefig(img_path)
    plt.show()


def plot_pol_errors_fc():
    all_iterations = np.arange(50, 2001, 50)

    plt.clf()
    plt.figure(dpi=400)
    seaborn.set_theme(style='whitegrid')

    # methods = ['MSA', 'POLYAK', 'NAGURNEY']
    # methods = ['MSA', 'POLYAK', 'NAGURNEY', 'SRA']
    methods = ['POLYAK', 'NAGURNEY']

    for m_idx, m in enumerate(methods):
        data_list = []
        path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / f"{m}_fc"
        all_iterations = np.arange(50, 2001, 50)
        for cur_iterations in all_iterations:
            file_name = f'{m}_zs_{cur_iterations}.pkl'
            with open(path / file_name, 'rb') as f:
                cur_data = pickle.load(f)
            data_list.append(cur_data)
        avg_errors = np.asarray([np.mean(d.errors) for d in data_list])
        std_errors = np.asarray([np.std(d.errors) for d in data_list])

        plot_filled_std_curves(
            x=all_iterations,
            mean=avg_errors,
            std=std_errors,
            color=COLORS[m_idx],
            lighter_color=LIGHT_COLORS[m_idx],
            linestyle=LINESTYLES[m_idx],
            label=m,
            min_val=0,
        )

    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Average Absolute Policy Error')
    plt.tight_layout()
    # plt.xlim(all_iterations[0], all_iterations[-1])

    img_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'a_img' / 'fc_pol.png'
    plt.savefig(img_path)
    plt.show()


def plot_value_errors():
    all_iterations = np.arange(50, 2001, 50)
    gt_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'ground_truth_zs_0_10.pkl'
    with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
    gt_values = gt_data.values[:, 0]

    plt.clf()
    plt.figure(dpi=400)
    seaborn.set_theme(style='whitegrid')

    methods = ['MSA', 'POLYAK', 'NAGURNEY', 'SRA', 'BB', 'BB_MSA', 'ADAM', 'EMA']
    # methods = ['MSA', 'POLYAK', 'NAGURNEY', 'SRA']

    for m_idx, m in enumerate(methods):
        data_list = get_data_list(m.lower())
        data_values = np.asarray([d.values[:, 0] for d in data_list])

        abs_err = np.abs(gt_values[np.newaxis, :] - data_values)
        avg_abs_err = np.mean(abs_err, axis=-1)
        std_abs_err = np.std(abs_err, axis=-1)

        plot_filled_std_curves(
            x=all_iterations,
            mean=avg_abs_err,
            std=std_abs_err,
            color=COLORS[m_idx],
            lighter_color=LIGHT_COLORS[m_idx],
            linestyle=LINESTYLES[m_idx],
            label=m,
            min_val=0,
        )
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Average Absolute Value Error')
    plt.tight_layout()
    plt.xlim(all_iterations[0], all_iterations[-1])

    img_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'a_img' / 'zs_val_all.png'
    plt.savefig(img_path)
    plt.show()


if __name__ == '__main__':
    # plot_pol_errors()
    plot_pol_errors_fc()
    # plot_value_errors()
