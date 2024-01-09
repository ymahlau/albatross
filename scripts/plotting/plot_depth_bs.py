
from pathlib import Path
import pickle
from matplotlib import pyplot as plt

import numpy as np
import scipy
import seaborn
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES

from src.misc.plotting import plot_filled_std_curves


def plot_bs_depth():
    data_path = Path(__file__).parent.parent.parent / 'a_data' / 'bs_depth'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'bs_depth'
    num_parts = 10
    num_seeds = 5
    depths = np.asarray(list(range(50, 2001, 50)), dtype=int)
    
    # prefix -> (alb, az)
    base_names =  {
        # 'd7': ('bs_az_alb_area_50_to_2000_inf_100games', 'bs_az_alb_area_50_to_2000_inf_100games'),
        # '4d7': ('bs_az_alb_area_50_to_2000_inf_100games', 'bs_az_alb_area_50_to_2000_inf_100games'),
        # '4nd7': ('bs_az_alb_area_50_to_2000_inf_100games', 'bs_az_alb_area_50_to_2000_inf_100games'),
        'nd7': ('bs_az_alb_area_50_to_2000_100games_retrained', 'bs_az_alb_area_50_to_2000_inf_100games')
    }
    
    for abbrev, (alb_base, az_base) in base_names.items():
        full_list_alb, full_list_az, length_list_alb, length_list_az  = [], [], [], []
        for seed in range(num_seeds):
            for part in range(num_parts):
                file_name_alb = f'{alb_base}_{abbrev}_{seed}_{part}.pkl'
                with open(data_path / file_name_alb, 'rb') as f:
                    cur_dict = pickle.load(f)
                full_list_alb.append(cur_dict['results_alb'])
                length_list_alb.append(cur_dict['lengths_alb'])
                
                file_name_az = f'{az_base}_{abbrev}_{seed}_{part}.pkl'
                with open(data_path / file_name_az, 'rb') as f:
                    cur_dict = pickle.load(f)
                full_list_az.append(cur_dict['results_az'])
                length_list_az.append(cur_dict['lengths_az'])
        full_arr_alb = np.concatenate(full_list_alb, axis=2)[:, 0, :]
        full_arr_az = np.concatenate(full_list_az, axis=2)[:, 0, :]
        length_arr_alb = np.concatenate(length_list_alb, axis=2)[:, 0, :]
        length_arr_az = np.concatenate(length_list_az, axis=2)[:, 0, :]
        
        # discount = 0.99
        # full_arr_alb = np.power(discount, length_arr_alb) * full_arr_alb
        # full_arr_az = np.power(discount, length_arr_az) * full_arr_az
        
        full_arr_alb = full_arr_alb.reshape(len(depths), num_seeds, -1).mean(axis=-1)
        full_arr_az = full_arr_az.reshape(len(depths), num_seeds, -1).mean(axis=-1)
        
        if abbrev == '4d7':
            full_arr_alb = scipy.signal.savgol_filter(full_arr_alb, window_length=5, polyorder=1, axis=0)
            full_arr_az = scipy.signal.savgol_filter(full_arr_az, window_length=5, polyorder=1, axis=0)
        
        # length_arr_alb = length_arr_alb.reshape(len(depths), num_seeds, -1).mean(axis=-1)
        # length_arr_az = length_arr_az.reshape(len(depths), num_seeds, -1).mean(axis=-1)
        
        plt.clf()
        plt.figure(dpi=600)
        seaborn.set_theme(style='whitegrid')
        
        # AlphaZero
        plot_filled_std_curves(
            x=depths,
            mean=full_arr_az.mean(axis=-1),
            std=full_arr_az.std(axis=-1),
            color=COLORS[0],
            lighter_color=LIGHT_COLORS[0],
            linestyle=LINESTYLES[0],
            label='AlphaZero',
        )
        
        # albatross
        plot_filled_std_curves(
            x=depths,
            mean=full_arr_alb.mean(axis=-1),
            std=full_arr_alb.std(axis=-1),
            color=COLORS[1],
            lighter_color=LIGHT_COLORS[1],
            linestyle=LINESTYLES[1],
            label='Albatross',
        )
        
        fontsize = 'large'
        plt.xlabel('Enemy Search Iterations', fontsize=fontsize)
        plt.ylabel('Reward', fontsize=fontsize)
        plt.xlim(depths[0], depths[-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if abbrev == 'd7':
            plt.legend(fontsize='x-large')
        plt.tight_layout()
        # plt.savefig(img_path / f'inf_100g_{abbrev}_depths.png')
        plt.savefig(img_path / f'bs_depth_{abbrev}.pdf', bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    plot_bs_depth()
