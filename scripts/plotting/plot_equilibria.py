import itertools
from pathlib import Path
from matplotlib.colors import ListedColormap
import textalloc

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
from src.misc.const import COLORS, LIGHT_COLORS


def plot_equilibria_zero_sum():
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'equilibria' / 'equilibria_new.pdf'
    temperature = 0.3
    # ja_dict = {
    #     (0, 0): (-4, 4),
    #     (0, 1): (-4, 4),
    #     (0, 2): (20, -20),
    #     (0, 3): (-7, 7),
    #     (1, 0): (-6, 6),
    #     (1, 1): (1, -1),
    #     (1, 2): (-4, 4),
    #     (1, 3): (9, -9),
    # }
    # ja_dict = {
    #     (0, 0): (-4, 4),
    #     (0, 1): (-7, 7),
    #     (1, 0): (-5, 5),
    #     (1, 1): (1, -1),
    # }
    ja_dict = {
        (0, 0): (-4, 4),
        (0, 1): (-7, 7),
        (1, 0): (-6, 6),
        (1, 1): (2, -2),
    }
    
    # ja_dict = {
    #     (0, 0): (-2, 2),
    #     (0, 1): (-4, 4),
    #     (1, 0): (-3, 3),
    #     (1, 1): (1, -1),
    # }
    
    # x1, x2, x3, x4 = 1, 2, 4, 3
    
    # ja_dict = {
    #     (0, 0): (x1, -x1),
    #     (0, 1): (x2, -x2),
    #     (1, 0): (x3, -x3),
    #     (1, 1): (x4, -x4),
    # }
    
    
    # table_values = [
    #     [-4, -4, 20, -7],
    #     [-6,  1, -4,  9],
    # ]
    table_values = [
        [-4, -7],
        [-6, 9],
    ]
    # available_actions = [[0, 1], [0, 1, 2, 3]]
    available_actions = [[0, 1], [0, 1]]

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
        available_actions=available_actions,
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
        available_actions=available_actions,
        joint_action_list=list(game_cfg.ja_dict.keys()),
        joint_action_value_arr=np.asarray(list(ja_dict.values()), dtype=float),
        temperature=temperature,
        leader=0,
        grid_size=int(1e4),
        num_iterations=15,
    )
    # QNE
    qne_val, qne_pol = compute_qne_equilibrium(
        available_actions=available_actions,
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
        available_actions=available_actions,
        joint_action_list=list(game_cfg.ja_dict.keys()),
        joint_action_value_arr=np.asarray(list(ja_dict.values()), dtype=float),
    )

    # plt.rcParams.update({
    #     'text.latex.preamble': r'\usepackage{amsfonts}'
    # })
    plt.clf()
    seaborn.set_theme(style='whitegrid')
    # colors = seaborn.color_palette('colorblind', n_colors=5)
    # plt.figure(figsize=(6.5, 6), dpi=600)
    fig = plt.figure()
    plt.tight_layout()
    
    plt.plot(x_prob, br_utils, color=COLORS[2], linestyle='dashed', zorder=10, label='BR / SE')
    # NE/SE
    plt.scatter(ne_pol[0][0], ne_val[0], color='xkcd:almost black', marker='o', s=50, zorder=11, label='NE / SSE')
    
    # Responses
    plt.plot(x_prob, sbr_utils, color='xkcd:almost black', linestyle='dashdot', zorder=10, label='SBR')
    plt.plot(x_prob, sbr_qs[:, 0], color='xkcd:grey', linestyle='dotted', zorder=10, label=None)
    plt.plot(x_prob, sbr_qs[:, 1], color='xkcd:grey', linestyle='dotted', zorder=10, label=None)
    # LE
    plt.scatter(le_pol[0][0], le_val[0], color='xkcd:almost black', marker='*', s=80, zorder=11, label='LE')
    # QNE
    plt.scatter(qne_pol[0][0], qne_val[0], color='xkcd:almost black', marker='s', s=50, zorder=11, label='QNE')
    # QSE
    plt.scatter(qse_pol[0][0], qse_val[0], color='xkcd:almost black', marker='v', s=70, zorder=11, label='QSE')
    # sbrle
    plt.plot(sbrle_probs, sbrle_vals, color=COLORS[1], linestyle='solid', zorder=10, label='SBRLE')
    
    # plt.plot(x_prob, br_qs[:, 0], color='xkcd:grey', linestyle='solid', zorder=10, label=None)
    # plt.plot(x_prob, br_qs[:, 1], color='xkcd:grey', linestyle='solid', zorder=10, label=None)
    
    # plt.plot(x_prob, qne_all_vals, color='red', label='QNE Utility')
    # plt.plot(x_prob, br_qs[:, 0], color='xkcd:grey', linestyle='dashdot', zorder=10, label='BR $Q_X$')
    # plt.plot(x_prob, sbr_qs[:, 1], color='grey', linestyle='dotted', zorder=10, label='SBR QY')
    
    # BRLE
    plt.scatter(brle_probs[0], brle_val, color='xkcd:almost black', marker='D', s=50, zorder=11, label='BRLE',
                clip_on=False)
    # plt.scatter(sbrle_pol[0], sbrle_val, color='black', marker='p', s=50, zorder=11, label='SBRLE')
    
    fontsize='large'
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize='medium')
    plt.xlabel('Probability of Action a', fontsize=fontsize)
    plt.ylabel('Expected Utility (Player 1)', fontsize=fontsize)
    
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.weight": 'bold',
    #     # "font.family": "serif",
    #     # "font.serif": ["Computer Modern Roman"],
    # })
    # text_list = ['$\\mathbf{\\tau_R} = 0$', '$\\tau = \\tau_R = 0.3$', '$\\tau_R \\rightarrow \\infty$']
    # x = [sbrle_probs[0], le_pol[0][0], sbrle_probs[-1]]
    # y = [sbrle_vals[0], le_val[0], sbrle_vals[-1]]
    # textalloc.allocate_text(
    #     fig,
    #     plt.gca(),
    #     x,
    #     y,
    #     text_list,
    #     x_scatter=x,
    #     y_scatter=y,
    #     textsize=12,
    #     margin=0,
    #     min_distance=0.03,
    #     max_distance=0.2,
    #     linecolor='xkcd:grey',
    #     seed=0,
    #     direction='northeast',
    #     fontweight='bold',
    # )
    # plt.rcParams.update({
    #     "text.usetex": False,
    #     "font.weight": 'normal',
    #     # "font.family": "serif",
    #     # "font.serif": ["Computer Modern Roman"],
    # })

    # table = plt.table(
    #     table_values,
    #     rowLabels=['X', 'Y'],
    #     cellLoc='center',
    #     rowLoc='center',
    #     loc='top',
    #     colWidths=[0.1 for _ in range(4)],
    #     edges='open',
    #     bbox=[0.25, 1.02, 0.5, 0.15],
    #     fontsize=fontsize,
    # )
    # table.scale(1, 1.5)

    # # Remove edge lines around the row labels
    # for (row, col), cell in table.get_celld().items():
    #     # cell.set_facecolor('pink')
    #     if row == 0 and col in (0, 3):
    #         cell.visible_edges = 'B'
    #     elif row == 0 and col in (1, 2):
    #         cell.visible_edges = 'BLR'
    #     elif row == 1 and col in (0, 3):
    #         cell.visible_edges = 'T'
    #     elif row == 1 and col in (1, 2):
    #         cell.visible_edges = 'TLR'
        # if key[1] == 0:
        #     cell.set_edgecolor('white')  # Set edge color to match the background
    # plt.gca().set_facecolor('pink')
    plt.ylim(-6, -3.5)
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

    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    


def plot_2d_equilibria():
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'equilibria'
    temperature = 0.3

    ja_dict = {
        (0, 0): (-4, 4),
        (0, 1): (-7, 7),
        (1, 0): (-6, 6),
        (1, 1): (2, -2),
    }
    
    # available_actions = [[0, 1], [0, 1, 2, 3]]
    available_actions = [[0, 1], [0, 1]]

    game_cfg = NormalFormConfig(ja_dict=ja_dict)
    game = NormalFormGame(game_cfg)
    x_prob = np.linspace(0, 1, 1000)

    # LE
    le_val, le_pol, _ = compute_logit_equilibrium(
        available_actions=available_actions,
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
    br_pol_c = []
    for x in x_prob:
        p0_policy = [np.asarray([x, 1 - x])]
        q1 = q_values_from_nfg(game, 1, p0_policy)
        br1 = best_response_from_q(q1)
        br_pol_c.append(br1[0])
        q0 = q_values_from_nfg(game, 0, [br1])
        br_qs.append(q0)
        utility = x * q0[0] + (1 - x) * q0[1]
        br_utils.append(utility)
    br_qs = np.asarray(br_qs)
    # smooth best response
    sbr_utils = []
    sbr_qs = []
    sbr_probs_c = []
    for x in x_prob:
        p0_policy = [np.asarray([x, 1-x])]
        q1 = q_values_from_nfg(game, 1, p0_policy)
        sbr1 = smooth_best_response_from_q(q1, temperature)
        sbr_probs_c.append(sbr1[0])
        q0 = q_values_from_nfg(game, 0, [sbr1])
        sbr_qs.append(q0)
        utility = x * q0[0] + (1-x) * q0[1]
        sbr_utils.append(utility)
    sbr_qs = np.asarray(sbr_qs)
    # QSE
    qse_val, qse_pol = compute_qse_equilibrium(
        available_actions=available_actions,
        joint_action_list=list(game_cfg.ja_dict.keys()),
        joint_action_value_arr=np.asarray(list(ja_dict.values()), dtype=float),
        temperature=temperature,
        leader=0,
        grid_size=int(1e4),
        num_iterations=15,
    )
    # QNE
    qne_val, qne_pol = compute_qne_equilibrium(
        available_actions=available_actions,
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
        available_actions=available_actions,
        joint_action_list=list(game_cfg.ja_dict.keys()),
        joint_action_value_arr=np.asarray(list(ja_dict.values()), dtype=float),
    )
    # expected utilities at all points in 2d space
    resolution = 100
    a_step, c_step = np.linspace(0, 1, resolution), np.linspace(0, 1, resolution)
    all_utils = np.zeros(shape=(resolution, resolution), dtype=float)
    for a_idx in range(resolution):
        for c_idx in range(resolution):
            a, c = a_step[a_idx], c_step[c_idx]
            all_utils[a_idx, c_idx] = a * c * ja_dict[(0, 0)][0] + a * (1 - c) * ja_dict[(0, 1)][0] \
                                    + (1-a) * c * ja_dict[(1, 0)][0] + (1-a) * (1-c) * ja_dict[(1, 1)][0]
    
    
    
    # plt.rcParams.update({
    #     'text.latex.preamble': r'\usepackage{amsfonts}'
    # })
    plt.clf()
    # seaborn.set_theme(style='whitegrid')
    # colors = seaborn.color_palette('colorblind', n_colors=5)
    # plt.figure(figsize=(6.5, 6), dpi=600)
    plt.figure()
    plt.tight_layout()
    
    # base image
    cmap = seaborn.color_palette("coolwarm_r", as_cmap=True)
    # cmap = seaborn.dark_palette("seagreen", as_cmap=True)
    # cmap = seaborn.color_palette("dark green", as_cmap=True)
    # ValueError: 'xkcd:dark green' is not a valid value for cmap; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
    # cmap = seaborn.light_palette("black", as_cmap=True) 'Greens_r'
    # cmap = list(seaborn.color_palette('colorblind', n_colors=10, as_cmap=True))[2]
    # my_cmap = ListedColormap(seaborn.color_palette([cmap]).as_hex())
    plt.imshow(all_utils.T, cmap='bone', interpolation='bicubic', extent=(0., 1., 0., 1.), origin='lower', vmax=2, vmin=-7)
    plt.colorbar()
    
    plt.plot(x_prob, br_pol_c, color=COLORS[2], linestyle='dashed', zorder=10, label='BR / SE')
    # # NE/SE
    plt.scatter(ne_pol[0][0], ne_pol[1][0], color='xkcd:almost black', marker='o', s=50, zorder=11, label='NE / SSE')
    
    # # Responses
    plt.plot(x_prob, sbr_probs_c, color='xkcd:almost black', linestyle='dashdot', zorder=10, label='SBR')
    # plt.plot(x_prob, sbr_qs[:, 0], color='xkcd:grey', linestyle='dotted', zorder=10, label=None)
    # plt.plot(x_prob, sbr_qs[:, 1], color='xkcd:grey', linestyle='dotted', zorder=10, label=None)
    # # LE
    plt.scatter(le_pol[0][0], le_pol[1][0], color='xkcd:almost black', marker='*', s=80, zorder=11, label='LE')
    # # QNE
    plt.scatter(qne_pol[0][0], qne_pol[1][0], color='xkcd:almost black', marker='s', s=50, zorder=11, label='QNE')
    # # QSE
    plt.scatter(qse_pol[0][0], qse_pol[1][0], color='xkcd:almost black', marker='v', s=70, zorder=11, label='QSE')
    # # sbrle
    tmp_var = [le_pol[1][0] for _ in sbrle_probs]
    plt.plot(sbrle_probs, tmp_var, color=COLORS[1], linestyle='solid', zorder=10, label='SBRLE')
    
    # plt.plot(x_prob, br_qs[:, 0], color='xkcd:grey', linestyle='solid', zorder=10, label=None)
    # plt.plot(x_prob, br_qs[:, 1], color='xkcd:grey', linestyle='solid', zorder=10, label=None)
    
    # plt.plot(x_prob, qne_all_vals, color='red', label='QNE Utility')
    # plt.plot(x_prob, br_qs[:, 0], color='xkcd:grey', linestyle='dashdot', zorder=10, label='BR $Q_X$')
    # plt.plot(x_prob, sbr_qs[:, 1], color='grey', linestyle='dotted', zorder=10, label='SBR QY')
    
    # BRLE
    plt.scatter(brle_probs[0], le_pol[1][0], color='xkcd:almost black', marker='D', s=50, zorder=11, label='BRLE', clip_on=False)
    # plt.scatter(sbrle_pol[0], sbrle_val, color='black', marker='p', s=50, zorder=11, label='SBRLE')
    
    fontsize='large'
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')
    plt.xlabel('Probability of Action a', fontsize=fontsize)
    plt.ylabel('Probability of Action c', fontsize=fontsize)
    
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.tight_layout()

    plt.savefig(img_path / 'equilibria_2d.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()



if __name__ == '__main__':
    plot_equilibria_zero_sum()
    # plot_2d_equilibria()
