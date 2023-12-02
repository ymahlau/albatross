import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt, patches

from src.equilibria.logit import compute_logit_equilibrium
from src.equilibria.nash import calculate_nash_equilibrium
from src.equilibria.quantal import compute_qse_equilibrium, compute_qne_equilibrium
from src.equilibria.responses import smooth_best_response_from_q, best_response_from_q
from src.game.initialization import get_game_from_config
from src.game.normal_form.normal_form import NormalFormConfig, NormalFormGame
from src.game.normal_form.utils import q_values_from_nfg


def plot_equilibria_zero_sum():
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'equilibria' / 'equilibria_all.png'
    temperature = 0.30
    ja_dict = {
        (0, 0): (-4, 4),
        (0, 1): (-4, 4),
        (0, 2): (20, -20),
        (0, 3): (-7, 7),
        (1, 0): (-6, 6),
        (1, 1): (1, -1),
        (1, 2): (-4, 4),
        (1, 3): (9, -9),
    }

    # ja_dict = {
    #     (0, 0): (-5, 5),
    #     (0, 1): (-5, 5),
    #     (0, 2): (10, -10),
    #     (0, 3): (-7, 7),
    #     (1, 0): (-9, 9),
    #     (1, 1): (3, -3),
    #     (1, 2): (-3, 3),
    #     (1, 3): (10, -10),
    # }
    game_cfg = NormalFormConfig(ja_dict=ja_dict)
    game = NormalFormGame(game_cfg)
    x_prob = np.linspace(0, 1, 1000)

    # LE
    le_val, le_pol, _ = compute_logit_equilibrium(
        available_actions=[[0, 1], [0, 1, 2, 3]],
        joint_action_list=list(game_cfg.ja_dict.keys()),
        joint_action_value_arr=np.asarray(list(ja_dict.values()), dtype=float),
        temperatures=[temperature, temperature],
        num_iterations=10000,
        epsilon=0,
    )
    le_q0 = q_values_from_nfg(game, 0, [le_pol[1]])
    brle_val = np.max(le_q0)
    brle_idx = np.argmax(le_q0)
    brle_probs = [float(brle_idx == 0), float(brle_idx == 1)]
    # sbrle
    temperatures = np.linspace(0, 100, 1000)
    sbrle_probs, sbrle_vals = [], []
    for t in temperatures:
        cur_sbrle = smooth_best_response_from_q(le_q0, t)
        cur_val = cur_sbrle[0] * le_q0[0] + cur_sbrle[1] * le_q0[1]
        sbrle_probs.append(cur_sbrle[0])
        sbrle_vals.append(cur_val)
    # Best response
    br_utils = []
    br_qs = []
    for x in x_prob:
        p0_policy = [np.asarray([x, 1 - x])]
        q1 = q_values_from_nfg(game, 1, p0_policy)
        br1 = best_response_from_q(q1)
        q0 = q_values_from_nfg(game, 0, [br1])
        br_qs.append(q0)
        utility = x * q0[0] + (1 - x) * q0[1]
        br_utils.append(utility)
    br_qs = np.asarray(br_qs)
    # smooth best response
    sbr_utils = []
    sbr_qs = []
    for x in x_prob:
        p0_policy = [np.asarray([x, 1-x])]
        q1 = q_values_from_nfg(game, 1, p0_policy)
        sbr1 = smooth_best_response_from_q(q1, temperature)
        q0 = q_values_from_nfg(game, 0, [sbr1])
        sbr_qs.append(q0)
        utility = x * q0[0] + (1-x) * q0[1]
        sbr_utils.append(utility)
    sbr_qs = np.asarray(sbr_qs)
    # QSE
    qse_val, qse_pol = compute_qse_equilibrium(
        available_actions=[[0, 1], [0, 1, 2, 3]],
        joint_action_list=list(game_cfg.ja_dict.keys()),
        joint_action_value_arr=np.asarray(list(ja_dict.values()), dtype=float),
        temperature=temperature,
        leader=0,
        grid_size=int(1e4),
        num_iterations=15,
    )
    # QNE
    qne_val, qne_pol = compute_qne_equilibrium(
        available_actions=[[0, 1], [0, 1, 2, 3]],
        joint_action_list=list(game_cfg.ja_dict.keys()),
        joint_action_value_arr=np.asarray(list(ja_dict.values()), dtype=float),
        temperature=temperature,
        leader=0,
        num_iterations=10000,
        random_prob=0.05,
    )
    # temporary sanity check
    qne_all_vals = []
    for x in x_prob:
        q0 = q_values_from_nfg(game, 0, [qne_pol[1]])
        utility = x * q0[0] + (1 - x) * q0[1]
        qne_all_vals.append(utility)
    # NE
    ne_val, ne_pol = calculate_nash_equilibrium(
        available_actions=[[0, 1], [0, 1, 2, 3]],
        joint_action_list=list(game_cfg.ja_dict.keys()),
        joint_action_value_arr=np.asarray(list(ja_dict.values()), dtype=float),
    )

    # plt.rcParams.update({
    #     'text.latex.preamble': r'\usepackage{amsfonts}'
    # })
    plt.clf()
    seaborn.set_theme(style='whitegrid')
    colors = seaborn.color_palette('colorblind', n_colors=5)
    # plt.figure(figsize=(6.5, 6), dpi=600)
    plt.figure(dpi=600)
    plt.tight_layout()
    # Responses
    plt.plot(x_prob, sbr_utils, color=colors[0], linestyle='dashed', zorder=10, label='SBR')
    plt.plot(x_prob, br_utils, 'xkcd:almost black', linestyle='solid', zorder=10, label='BR / SE')
    plt.plot(x_prob, sbr_qs[:, 0], color=colors[1], linestyle='dashdot', zorder=10, label='SBR Util. X')
    # plt.plot(x_prob, qne_all_vals, color='red', label='QNE Utility')
    # plt.plot(x_prob, br_qs[:, 0], color='xkcd:grey', linestyle='dashdot', zorder=10, label='BR $Q_X$')
    # plt.plot(x_prob, sbr_qs[:, 1], color='grey', linestyle='dotted', zorder=10, label='SBR QY')
    # sbrle
    plt.plot(sbrle_probs, sbrle_vals, color=colors[2], linestyle='dotted', zorder=10, label='SBRLE')
    # QSE
    plt.scatter(qse_pol[0][0], qse_val[0], color='xkcd:almost black', marker='v', s=70, zorder=11, label='QSE')
    # QNE
    plt.scatter(qne_pol[0][0], qne_val[0], color='xkcd:almost black', marker='s', s=50, zorder=11, label='QNE')
    # LE
    plt.scatter(le_pol[0][0], le_val[0], color='xkcd:almost black', marker='*', s=80, zorder=11, label='LE')
    # BRLE
    plt.scatter(brle_probs[0], brle_val, color='xkcd:almost black', marker='D', s=50, zorder=11, label='BRLE',
                clip_on=False)
    # plt.scatter(sbrle_pol[0], sbrle_val, color='black', marker='p', s=50, zorder=11, label='SBRLE')
    # NE/SE
    plt.scatter(ne_pol[0][0], ne_val[0], color='xkcd:almost black', marker='o', s=50, zorder=11, label='NE / SSE')
    fontsize='large'
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    plt.xlabel('Probability of X', fontsize=fontsize)
    plt.ylabel('Utility', fontsize=fontsize)

    table_values = [
        [-4, -4, 20, -7],
        [-6,  1, -4,  9],
    ]

    table = plt.table(
        table_values,
        rowLabels=['X', 'Y'],
        cellLoc='center',
        rowLoc='center',
        loc='top',
        colWidths=[0.1 for _ in range(4)],
        edges='open',
        bbox=[0.25, 1.02, 0.5, 0.15],
        fontsize=fontsize,
    )
    table.scale(1, 1.5)

    # Remove edge lines around the row labels
    for (row, col), cell in table.get_celld().items():
        # cell.set_facecolor('pink')
        if row == 0 and col in (0, 3):
            cell.visible_edges = 'B'
        elif row == 0 and col in (1, 2):
            cell.visible_edges = 'BLR'
        elif row == 1 and col in (0, 3):
            cell.visible_edges = 'T'
        elif row == 1 and col in (1, 2):
            cell.visible_edges = 'TLR'
        # if key[1] == 0:
        #     cell.set_edgecolor('white')  # Set edge color to match the background
    # plt.gca().set_facecolor('pink')
    plt.ylim(-6, -3)
    plt.xlim(0, 1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.tight_layout()

    ax = plt.gca()
    # border = patches.Polygon(ax.get_position().get_points(), closed=True, edgecolor='black', facecolor='none',
    #                          linewidth=50)
    # ax.add_patch(border)
    # ax.patch.set_edgecolor('black')
    # ax.patch.set_linewidth(20)

    plt.savefig(img_path)
    plt.show()

if __name__ == '__main__':
    plot_equilibria_zero_sum()
