from scipy.signal import savgol_filter

from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn

from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES
from src.misc.plotting import plot_filled_std_curves


def plot_alphazero_runs():
    data_path = Path(__file__).parent.parent.parent / 'a_data' / 'az_wandb'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'az_wandb'
    
    full_names = {
        'd7': 'Deterministic 2-Player (Tron)',
        '4d7': 'Deterministic 4-Player',
        'nd7': 'Stochastic 2-Player',
        '4nd7': 'Stochastic 4-Player',
    }
    
    method_name_dict = {
        'duct': 'DUCT',
        'rm': 'RM',
        'smoos': 'SM-OOS',
        'sbr': 'LE',
        'exp3': 'EXP3',
        'nash': 'Nash',
    }
    for prefix in full_names.keys():
        cur_max_time = 48 if 'n' in prefix else 24
        runs_per_name = ['duct', 'rm', 'smoos', 'sbr'] if 'n' in prefix else ['duct', 'rm', 'smoos', 'sbr', 'exp3', 'nash']
        for m in ['pol', 'val']:
            if m == 'pol':
                save_dir = data_path / prefix / 'ac_pol_data.pkl'
            else:
                save_dir = data_path / prefix / 'ac_val_data.pkl'
            all_names = []
            for n in runs_per_name:
                num_seeds = 5
                for seed in range(num_seeds):
                    if prefix == 'd7':
                        all_names.append(f'{n}_{seed}')
                    else:
                        all_names.append(f'{prefix}_{n}_{seed}')
            with open(save_dir, 'rb') as f:
                result_dict = pickle.load(f)

            # interpolate data, we played tournaments concurrent to training and they ended at slightly different time points
            interpolated_dict = {}
            start_time = max([v[0][0] for k, v in result_dict.items()])
            end_time = min([v[-1][0] for k, v in result_dict.items()])
            steps = 15
            times = np.linspace(start_time, end_time, steps)
            x = np.linspace(start_time / 3600, cur_max_time, steps)
            for name in all_names:
                cur_data = result_dict[name]
                interp_arr = np.empty(shape=(times.shape[0],), dtype=float)
                for t_idx, t in enumerate(times):
                    # find data point above and below t
                    below_idx = np.sum(cur_data[:, 0] < t).item() - 1
                    above_idx = below_idx + 1
                    dx = cur_data[above_idx, 0] - cur_data[below_idx, 0]
                    dy = cur_data[above_idx, 1] - cur_data[below_idx, 1]
                    y_interp = cur_data[below_idx, 1] + dy * (t - cur_data[below_idx, 0]) / dx
                    interp_arr[t_idx] = y_interp
                interpolated_dict[name] = interp_arr
            # calculate mean and std
            to_print, val_dict = {}, {}
            for name in runs_per_name:
                if prefix == 'd7':
                    full_arr = np.asarray([interpolated_dict[f'{name}_{seed}'] for seed in range(5)])
                else:
                    full_arr = np.asarray([interpolated_dict[f'{prefix}_{name}_{seed}'] for seed in range(5)])
                if 'n' not in prefix:
                    full_arr: np.ndarray = savgol_filter(full_arr, axis=-1, window_length=5, polyorder=3)
                mean = np.mean(full_arr, axis=0)
                std = np.std(full_arr, axis=0)
                to_print[name] = mean, std
                val_dict[name] = full_arr[:, np.newaxis, :]
            
            # plot
            plt.clf()
            seaborn.set_theme(style='whitegrid')
            plt.figure()

            for name_idx, name in enumerate(runs_per_name):
                plot_filled_std_curves(
                    x=x,
                    mean=to_print[name][0],
                    std=to_print[name][1],
                    color=COLORS[name_idx],
                    lighter_color=LIGHT_COLORS[name_idx],
                    label=method_name_dict[name],
                    linestyle=LINESTYLES[name_idx],
                )
            # plt.title('Overcooked Training reward')
            fontsize = 'x-large'
            plt.xlabel('Training time [h]', fontsize=fontsize)
            plt.ylabel('Reward', fontsize=fontsize)
            plt.xlim(start_time / 3600, cur_max_time)
            plt.xticks(fontsize='large')
            plt.yticks(fontsize=fontsize)
            if m == 'pol' and prefix == 'd7':
                plt.legend(fontsize='large')
            # if m == 'pol':
            #     plt.title(f'{full_names[prefix]}', fontsize=fontsize)
            plt.tight_layout()
            plt.savefig(img_path / f'{prefix}_{m}.pdf', bbox_inches='tight', pad_inches=0.02)
        
if __name__ == '__main__':
    plot_alphazero_runs()
