from pathlib import Path
import pickle
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import seaborn
from scripts.logit_solver.solve_logit_parallel import EquilibriumData

from src.game.normal_form.random_matrix import NFGType
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves

def plot_policy_error():
    dir_path = Path(__file__).parent.parent.parent / 'a_data' / 'logit_solver'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'logit_solver'
    
    hp_dict: dict[str, tuple[Optional[float], Optional[float]]] = {
        # 'EMA': (0.5, None),
        'MSA': (None, None),
        'POLYAK': (None, None),
        'NAGURNEY': (None, None),
        'SRA': (0.3, 1.8),
    }
    nfg_type_names = {
        NFGType.ZERO_SUM: ('Zero-Sum', 'zs'),
        NFGType.FULL_COOP: ('Fully Cooperative', 'fc'),
        NFGType.GENERAL: ('General-Sum', 'gen'),
    }
    ylims = {
        NFGType.ZERO_SUM: (0, 0.035),
        NFGType.FULL_COOP: (0, 0.15),
        NFGType.GENERAL: (0, 0.2),
    }
    iterations = np.arange(50, 2001, 50)
    
    
    for nfg_type in nfg_type_names.keys():
        # gt_name = f'gt_msa_{nfg_type.value}.pkl'
        # gt_path = dir_path / gt_name
        # with open(gt_path, 'rb') as f:
        #     gt_data: EquilibriumData = pickle.load(f)

        error_dict = {k: [] for k in hp_dict.keys()}
        for sbr_mode_str in hp_dict.keys():
            for cur_iterations in iterations:
                file_name = f'data_{sbr_mode_str.lower()}_{nfg_type.value}_{cur_iterations}.pkl'
                full_save_path = dir_path / file_name
                with open(full_save_path, 'rb') as f:
                    cur_data: EquilibriumData = pickle.load(f)
                error_dict[sbr_mode_str].append(cur_data.errors)
        
        plt.clf()
        plt.figure()
        seaborn.set_theme(style='whitegrid')
        
        for m_idx, (sbr_mode_str, error_list) in enumerate(error_dict.items()):
            error_arr = np.asarray(error_list)
            
            plt.plot(iterations, error_arr.mean(axis=-1), color=COLORS[m_idx], linestyle=LINESTYLES[m_idx], label=sbr_mode_str)
            # plot_filled_std_curves(
            #     x=iterations,
            #     mean=error_arr.mean(axis=-1),
            #     std=error_arr.std(axis=-1),
            #     color=COLORS[m_idx],
            #     lighter_color=LIGHT_COLORS[m_idx],
            #     linestyle=LINESTYLES[m_idx],
            #     label=sbr_mode_str,
            #     min_val=0,
            # )
        
        fontsize = 'xx-large'
        plt.xlabel('Solver Iterations', fontsize=fontsize)
        plt.ylabel('Policy Error', fontsize=fontsize)
        plt.xlim(iterations[0], iterations[-1])
        # plt.ylim(ylims[nfg_type][0], ylims[nfg_type][1])
        plt.yscale('log')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize=fontsize)
        # plt.title(f'{nfg_type_names[nfg_type][0]}', fontsize=fontsize)
        if nfg_type_names[nfg_type][1] == 'gen':
            plt.legend(fontsize='large')
        plt.tight_layout()
        plt.savefig(img_path / f'pol_err_{nfg_type_names[nfg_type][1]}.pdf', bbox_inches='tight', pad_inches=0.03)


def plot_value_error():
    dir_path = Path(__file__).parent.parent.parent / 'a_data' / 'logit_solver'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'logit_solver'
    
    hp_dict: dict[str, tuple[Optional[float], Optional[float]]] = {
        # 'EMA': (0.5, None),
        'MSA': (None, None),
        'POLYAK': (None, None),
        'NAGURNEY': (None, None),
        'SRA': (0.3, 1.8),
    }
    nfg_type_names = {
        NFGType.ZERO_SUM: ('Zero-Sum', 'zs'),
        # NFGType.FULL_COOP: ('Fully Cooperative', 'fc'),
        # NFGType.GENERAL: ('General-Sum', 'gen'),
    }

    iterations = np.arange(50, 2001, 50)
    
    
    for nfg_type in nfg_type_names.keys():
        gt_name = f'gt_msa_{nfg_type.value}.pkl'
        gt_path = dir_path / gt_name
        with open(gt_path, 'rb') as f:
            gt_data: EquilibriumData = pickle.load(f)

        value_dict = {k: [] for k in hp_dict.keys()}
        for sbr_mode_str in hp_dict.keys():
            for cur_iterations in iterations:
                file_name = f'data_{sbr_mode_str.lower()}_{nfg_type.value}_{cur_iterations}.pkl'
                full_save_path = dir_path / file_name
                with open(full_save_path, 'rb') as f:
                    cur_data: EquilibriumData = pickle.load(f)
                value_dict[sbr_mode_str].append(cur_data.values)
        
        plt.clf()
        plt.figure()
        seaborn.set_theme(style='whitegrid')
        
        for m_idx, (sbr_mode_str, value_list) in enumerate(value_dict.items()):
            gt_values = gt_data.values[np.newaxis, :, 0]
            value_arr = np.asarray(value_list)[..., 0]
            error_arr = np.abs(gt_values - value_arr)
            
            plt.plot(iterations, error_arr.mean(axis=-1), color=COLORS[m_idx], linestyle=LINESTYLES[m_idx], label=sbr_mode_str)
            # plot_filled_std_curves(
            #     x=iterations,
            #     mean=error_arr.mean(axis=-1),
            #     std=error_arr.std(axis=-1),
            #     color=COLORS[m_idx],
            #     lighter_color=LIGHT_COLORS[m_idx],
            #     linestyle=LINESTYLES[m_idx],
            #     label=sbr_mode_str,
            #     min_val=0,
            # )
        
        fontsize = 'xx-large'
        # plt.xticks(fontsize='large')
        plt.xlabel('Solver Iterations', fontsize=fontsize)
        plt.ylabel('Value Error', fontsize=fontsize)
        plt.xlim(iterations[0], iterations[-1])
        # plt.ylim(0, 0.0008)
        plt.xticks(fontsize='large')
        plt.yticks(fontsize=fontsize)
        plt.yscale('log')
        # plt.title(f'{nfg_type_names[nfg_type][0]}', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'val_err_{nfg_type_names[nfg_type][1]}.pdf', bbox_inches='tight', pad_inches=0.03)


if __name__ == '__main__':
    plot_policy_error()
    plot_value_error()
