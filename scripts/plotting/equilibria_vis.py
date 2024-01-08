from datetime import datetime
import pickle
from pathlib import Path
import matplotlib

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.equilibria.logit import compute_logit_equilibrium, SbrMode


def compute_data():
    path = Path(__file__).parent.parent.parent / 'a_data' / 'eq'

    available_actions = [[0, 1], [0, 1]]
    joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    joint_action_values = np.asarray([[2, -2], [-1, 1], [2, -2], [-3, 3]])

    steps = 20
    epsilon = 0.01
    temperatures = [0.2]
    x_steps = np.linspace(epsilon, 1 - epsilon, steps)
    y_steps = np.linspace(epsilon, 1 - epsilon, steps)
    X, Y = np.meshgrid(x_steps, y_steps)
    dx = np.zeros(shape=(len(temperatures), steps, steps), dtype=float)
    dy = np.zeros(shape=(len(temperatures), steps, steps), dtype=float)

    for t_idx, t in enumerate(temperatures):
        print(f'{datetime.now()} - starting temp {t_idx}: {t}')
        for x_idx in tqdm(range(len(x_steps))):
            for y_idx in range(len(x_steps)):
                x, y = X[x_idx, y_idx], Y[x_idx, y_idx]
                init_policy = [
                    np.asarray([x, 1-x]),
                    np.asarray([y, 1-y]),
                ]

                _, policies, _ = compute_logit_equilibrium(
                    available_actions=available_actions,
                    joint_action_list=joint_action_list,
                    joint_action_value_arr=joint_action_values,
                    num_iterations=1,
                    epsilon=0,
                    initial_policies=init_policy,
                    sbr_mode=SbrMode.MSA,
                    temperatures=[t, t],
                )
                dst = np.asarray(policies) * 2 - np.asarray(init_policy)
                cur_dx = dst[0, 0] - x
                cur_dy = dst[1, 0] - y
                dx[t_idx, x_idx, y_idx] = cur_dx
                dy[t_idx, x_idx, y_idx] = cur_dy
    result_dict = {
        'dx': dx,
        'dy': dy,
        'x_steps': x_steps,
        'y_steps': y_steps,
        'X': X,
        'Y': Y,
        'temperatures': temperatures,
    }
    with open(path / 'example.pkl', 'wb') as f:
        pickle.dump(result_dict, f)


def plot_streams():
    path = Path(__file__).parent.parent.parent / 'a_data' / 'eq'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'eq'
    base_name = 'example'
    with open(path / f'{base_name}.pkl', 'rb') as f:
        result = pickle.load(f)

    x = result['x_steps']
    y = result['y_steps']
    dx = result['dx']
    dy = result['dy']
    temperatures = result['temperatures']

    # eqs = [
    #     [[0.5, 0.5]],
    #     [[0.5, 0.5]],
    #     [[0.5, 0.5], [0.99, 0.99], [0.01, 0.01]],
    # ]

    for t_idx, t in enumerate(temperatures):
        plt.clf()
        # seaborn.set_theme(style='whitegrid')
        plt.figure(dpi=600)
        # plt.streamplot(x, y, dx[t_idx], dy[t_idx], density=1, color='grey')
        # plt.quiver(x, y, dx[t_idx], dy[t_idx], angles='xy', scale=50, width=0.005, scale_units='width', color='blue')
        # u = 
        X, Y = np.meshgrid(x, y)
        plt.streamplot(X, Y, dx[t_idx], dy[t_idx], density=0.5, linewidth=3, arrowsize=4, # type: ignore
                       arrowstyle='fancy', color='xkcd:charcoal') # type: ignore
        # for i in range(ne.shape[0]):
        #     xy = ne[i]
        #     plt.scatter(xy[0], xy[1], marker='*', c='green', label='Nash Equilibrium', s=100)
        # eq = eqs[t_idx]
        # for eq_i in eq:
        #     plt.scatter(eq_i[0], eq_i[1], marker='*', c='red', s=75)

        # fontsize='xx-large'
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.gca().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        # plt.legend()
        # plt.xlabel('$\pi_0(a_0)$', fontsize=fontsize)
        # plt.ylabel('$\pi_1(a_0)$', fontsize=fontsize)
        # plt.xticks(fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(img_path / f'{base_name}_{t}.pdf')



if __name__ == '__main__':
    # compute_data()
    plot_streams()
