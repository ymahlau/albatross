
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
    lower_depths = np.asarray(list(range(50, 2001, 50)), dtype=int)
    # depths = np.arange(100, 1001, 100)
    upper_depths = np.arange(3000, 20001, 1000)
    upper_depths2_d7 = np.asarray([int(5e4), int(1e5), int(5e5), int(1e6)])
    upper_depths2_other = np.asarray([int(5e4), int(1e5)])
    upper_depths3_other = np.asarray([int(5e5)])
    # upper_depths4_d7 = np.asarray([int(5e6)])
    # full_depths_d7 = np.concatenate([lower_depths, upper_depths, upper_depths2_d7, upper_depths4_d7])
    full_depths_d7 = np.concatenate([lower_depths, upper_depths, upper_depths2_d7])
    full_depths_other = np.concatenate([lower_depths, upper_depths, upper_depths2_other, upper_depths3_other])
    
    upper_depths2_dict = {
        'd7': upper_depths2_d7,
        'nd7': upper_depths2_other,
        '4nd7': upper_depths2_other,
    }
    
    upper_depths3_dict = {
        'nd7': upper_depths3_other,
        '4nd7': upper_depths3_other,
    }
    
    # upper_depths4_dict = {
    #     'd7': upper_depths4_d7,
    # }
    
    full_depths_dict = {
        'd7': full_depths_d7,
        'nd7': full_depths_other,
        '4nd7': full_depths_other,
    }
    
    # prefix -> (alb, az)
    # base_names =  {
    #     'd7': ('base_sampl', 'base_sampl'),
    #     '4d7': ('base_sampl', 'base_sampl'),
    #     # '4nd7': ('base_sampl', 'base_sampl'),
    #     'nd7': ('base_sampl', 'base_sampl'),
    # }
    base_names =  {
        'd7': ('base_sampl', 'base_sampl'),
        # '4d7': ('base_sampl', 'base_sampl'),
        # 'nd7': ('base_sampl', 'base_sampl'),
        # '4nd7': ('base_sampl', 'base_sampl'),
    }
    
    upper_names = {
        'd7': ('upper_base_sampl', 'upper_base_sampl'),
        '4d7': ('upper_base_sampl', 'upper_base_sampl'),
        'nd7': ('upper_base_sampl', 'upper_base_sampl'),
        '4nd7': ('upper_base_sampl', 'upper_base_sampl'),
    }
    
    upper_names2 = {
        'd7': ('upper2_base_sampl', 'upper2_base_sampl'),
        '4d7': ('upper2_base_sampl', 'upper2_base_sampl'),
        'nd7': ('upper2_base_sampl', 'upper2_base_sampl'),
        '4nd7': ('upper2_base_sampl', 'upper2_base_sampl'),
    }
    
    upper_names3 = {
        'nd7': ('upper3_base_sampl', 'upper3_base_sampl'),
        '4nd7': ('upper3_base_sampl', 'upper3_base_sampl'),
    }
    
    upper_names4 = {
        'd7': ('upper4_base_sampl', 'upper4_base_sampl'),
    }
    
    full_names = {
        'd7': 'Tron',
        '4d7': 'Deterministic 4-Player',
        'nd7': 'Stochastic 2-Player',
        '4nd7': 'Stochastic 4-Player',
    }
    
    for abbrev, (alb_base, az_base) in base_names.items():
        alb_list, az_list = [], []
        
        upper_alb_base, upper_az_base = upper_names[abbrev]
        upper_alb_base2, upper_az_base2 = upper_names2[abbrev]
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
        lower_arr_alb = np.concatenate(full_list_alb, axis=2)[:, 0, :]
        lower_arr_az = np.concatenate(full_list_az, axis=2)[:, 0, :]
        
        lower_arr_alb = lower_arr_alb.reshape(len(lower_depths), num_seeds, -1).mean(axis=-1)
        lower_arr_az = lower_arr_az.reshape(len(lower_depths), num_seeds, -1).mean(axis=-1)
        alb_list.append(lower_arr_alb)
        az_list.append(lower_arr_az)
        
        full_list_alb, full_list_az, length_list_alb, length_list_az  = [], [], [], []
        for seed in range(num_seeds):
            for part in range(num_parts):
                file_name_alb = f'{upper_alb_base}_{abbrev}_{seed}_{part}.pkl'
                with open(data_path / file_name_alb, 'rb') as f:
                    cur_dict = pickle.load(f)
                full_list_alb.append(cur_dict['results_alb'])
                length_list_alb.append(cur_dict['lengths_alb'])
                
                file_name_az = f'{upper_az_base}_{abbrev}_{seed}_{part}.pkl'
                with open(data_path / file_name_az, 'rb') as f:
                    cur_dict = pickle.load(f)
                full_list_az.append(cur_dict['results_az'])
                length_list_az.append(cur_dict['lengths_az'])
        upper_arr_alb = np.concatenate(full_list_alb, axis=2)[:, 0, :]
        upper_arr_az = np.concatenate(full_list_az, axis=2)[:, 0, :]
        
        upper_arr_alb = upper_arr_alb.reshape(len(upper_depths), num_seeds, -1).mean(axis=-1)
        upper_arr_az = upper_arr_az.reshape(len(upper_depths), num_seeds, -1).mean(axis=-1)
        alb_list.append(upper_arr_alb)
        az_list.append(upper_arr_az)
        
        full_list_alb, full_list_az, length_list_alb, length_list_az  = [], [], [], []
        for seed in range(num_seeds):
            for part in range(5):  # only 5 parts
                file_name_alb = f'{upper_alb_base2}_{abbrev}_{seed}_{part}.pkl'
                with open(data_path / file_name_alb, 'rb') as f:
                    cur_dict = pickle.load(f)
                full_list_alb.append(cur_dict['results_alb'])
                length_list_alb.append(cur_dict['lengths_alb'])
                
                file_name_az = f'{upper_az_base2}_{abbrev}_{seed}_{part}.pkl'
                with open(data_path / file_name_az, 'rb') as f:
                    cur_dict = pickle.load(f)
                full_list_az.append(cur_dict['results_az'])
                length_list_az.append(cur_dict['lengths_az'])
        upper_arr_alb2 = np.concatenate(full_list_alb, axis=2)[:, 0, :]
        upper_arr_az2 = np.concatenate(full_list_az, axis=2)[:, 0, :]
        
        upper_arr_alb2 = upper_arr_alb2.reshape(len(upper_depths2_dict[abbrev]), num_seeds, -1).mean(axis=-1)
        upper_arr_az2 = upper_arr_az2.reshape(len(upper_depths2_dict[abbrev]), num_seeds, -1).mean(axis=-1)
        alb_list.append(upper_arr_alb2)
        az_list.append(upper_arr_az2)
        
        if abbrev in upper_depths3_dict:
            upper_alb_base3, upper_az_base3 = upper_names3[abbrev]
            full_list_alb, full_list_az, length_list_alb, length_list_az  = [], [], [], []
            for seed in range(num_seeds):
                for part in range(5):  # only 5 parts
                    file_name_alb = f'{upper_alb_base3}_{abbrev}_{seed}_{part}.pkl'
                    with open(data_path / file_name_alb, 'rb') as f:
                        cur_dict = pickle.load(f)
                    full_list_alb.append(cur_dict['results_alb'])
                    length_list_alb.append(cur_dict['lengths_alb'])
                    
                    file_name_az = f'{upper_az_base3}_{abbrev}_{seed}_{part}.pkl'
                    with open(data_path / file_name_az, 'rb') as f:
                        cur_dict = pickle.load(f)
                    full_list_az.append(cur_dict['results_az'])
                    length_list_az.append(cur_dict['lengths_az'])
            upper_arr_alb3 = np.concatenate(full_list_alb, axis=2)[:, 0, :]
            upper_arr_az3 = np.concatenate(full_list_az, axis=2)[:, 0, :]
            
            upper_arr_alb3 = upper_arr_alb3.reshape(len(upper_depths3_dict[abbrev]), num_seeds, -1).mean(axis=-1)
            upper_arr_az3 = upper_arr_az3.reshape(len(upper_depths3_dict[abbrev]), num_seeds, -1).mean(axis=-1)
            alb_list.append(upper_arr_alb3)
            az_list.append(upper_arr_az3)
            
        # if abbrev in upper_depths4_dict:
        #     upper_alb_base4, upper_az_base4 = upper_names4[abbrev]
        #     full_list_alb, full_list_az, length_list_alb, length_list_az  = [], [], [], []
        #     for seed in range(num_seeds):
        #         for part in range(5):  # only 5 parts
        #             file_name_alb = f'{upper_alb_base4}_{abbrev}_{seed}_{part}.pkl'
        #             with open(data_path / file_name_alb, 'rb') as f:
        #                 cur_dict = pickle.load(f)
        #             full_list_alb.append(cur_dict['results_alb'])
        #             length_list_alb.append(cur_dict['lengths_alb'])
                    
        #             file_name_az = f'{upper_az_base4}_{abbrev}_{seed}_{part}.pkl'
        #             with open(data_path / file_name_az, 'rb') as f:
        #                 cur_dict = pickle.load(f)
        #             full_list_az.append(cur_dict['results_az'])
        #             length_list_az.append(cur_dict['lengths_az'])
        #     upper_arr_alb4 = np.concatenate(full_list_alb, axis=2)[:, 0, :]
        #     upper_arr_az4 = np.concatenate(full_list_az, axis=2)[:, 0, :]
            
        #     upper_arr_alb4 = upper_arr_alb4.reshape(len(upper_depths4_dict[abbrev]), num_seeds, -1).mean(axis=-1)
        #     upper_arr_az4 = upper_arr_az4.reshape(len(upper_depths4_dict[abbrev]), num_seeds, -1).mean(axis=-1)
        #     alb_list.append(upper_arr_alb4)
        #     az_list.append(upper_arr_az4)
        
        
        full_arr_alb = np.concatenate(alb_list, axis=0)
        full_arr_az = np.concatenate(az_list, axis=0)
        
        # length_arr_alb = np.concatenate(length_list_alb, axis=2)[:, 0, :]
        # length_arr_az = np.concatenate(length_list_az, axis=2)[:, 0, :]
        
        # discount = 0.99
        # full_arr_alb = np.power(discount, length_arr_alb) * full_arr_alb
        # full_arr_az = np.power(discount, length_arr_az) * full_arr_az
        
        # full_arr_alb = np.concatenate([lower_arr_alb, upper_arr_alb, upper_arr_alb2], axis=0)
        # full_arr_az = np.concatenate([lower_arr_az, upper_arr_az, upper_arr_az2], axis=0)
                
        # if abbrev == '4d7':
        #     full_arr_alb = scipy.signal.savgol_filter(full_arr_alb, window_length=5, polyorder=1, axis=0)
        #     full_arr_az = scipy.signal.savgol_filter(full_arr_az, window_length=5, polyorder=1, axis=0)
        
        # length_arr_alb = length_arr_alb.reshape(len(depths), num_seeds, -1).mean(axis=-1)
        # length_arr_az = length_arr_az.reshape(len(depths), num_seeds, -1).mean(axis=-1)
        
        plt.clf()
        plt.figure(dpi=600, figsize=(6, 4))
        seaborn.set_theme(style='whitegrid')
        
        # albatross
        plot_filled_std_curves(
            x=full_depths_dict[abbrev],
            mean=full_arr_alb.mean(axis=-1),
            std=full_arr_alb.std(axis=-1),
            color=COLORS[1],
            lighter_color=LIGHT_COLORS[1],
            linestyle=LINESTYLES[0],
            label='Albatross',
        )
        
        # AlphaZero
        plot_filled_std_curves(
            x=full_depths_dict[abbrev],
            mean=full_arr_az.mean(axis=-1),
            std=full_arr_az.std(axis=-1),
            color=COLORS[2],
            lighter_color=LIGHT_COLORS[2],
            linestyle=LINESTYLES[1],
            label='AlphaZero',
        )
        
        fontsize = 'xx-large'
        plt.xlabel('Enemy Search Iterations', fontsize=fontsize)
        plt.ylabel('Reward', fontsize=fontsize)
        plt.xlim(full_depths_dict[abbrev][0], full_depths_dict[abbrev][-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xscale('log')
        # if abbrev != '4d7':
        #     plt.title(f'{full_names[abbrev]}', fontsize=fontsize)
        if abbrev == 'd7' or abbrev == '4d7':
            plt.legend(fontsize=fontsize, loc='upper right', bbox_to_anchor=(1.03, 1.05))
        plt.tight_layout()
        # plt.savefig(img_path / f'inf_100g_{abbrev}_depths.png')
        plt.savefig(img_path / f'full_sample_{abbrev}.pdf', bbox_inches='tight', pad_inches=0.03)


if __name__ == '__main__':
    plot_bs_depth()
