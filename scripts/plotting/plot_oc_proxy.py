
from pathlib import Path
import pickle
from matplotlib import pyplot as plt

import numpy as np
import seaborn
from src.misc.const import COLORS, LIGHT_COLORS, LINESTYLES

from src.misc.plotting import plot_filled_std_curves


def plot_proxy_vs_proxy():
    name_dict = {
        'cr': 'Cramped Rm.',
        'aa': 'Asym. Adv.',
        'co': 'Coord. Ring',
        'fc': 'Forced Coord.',
        'cc': 'Counter Circ.',
    }
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc'
    base_name = 'proxy_proxy_temps_1'
    temperatures = np.linspace(0, 10, 100)
    num_seeds = 5
    
    for prefix, _ in name_dict.items():
        result_list = []
        for seed in range(num_seeds):
            with open(save_path / f'{base_name}_{prefix}_{seed}.pkl', 'rb') as f:
                cur_results = pickle.load(f)
            result_list.append(cur_results)
        result_arr = np.asarray(result_list)[:, :, 0, :]
        result_arr = np.transpose(result_arr, axes=(1, 0, 2)).reshape(len(temperatures), -1)
        
        plt.clf()
        plt.figure(dpi=600)
        seaborn.set_theme(style='whitegrid')
        plot_filled_std_curves(
            x=temperatures,
            mean=result_arr.mean(axis=-1),
            std=result_arr.std(axis=-1),
            color=COLORS[0],
            lighter_color=LIGHT_COLORS[0],
            linestyle=LINESTYLES[0],
            label=None,
            min_val=0,
        )
        
        fontsize = 'medium'
        plt.xlabel('Temperature', fontsize=fontsize)
        plt.ylabel('Reward', fontsize=fontsize)
        plt.xlim(temperatures[0], temperatures[-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'proxy_proxy_{prefix}.png')


def plot_resp_vs_proxy():
    name_dict = {
        'cr': 'Cramped Rm.',
        'aa': 'Asym. Adv.',
        'co': 'Coord. Ring',
        'fc': 'Forced Coord.',
        'cc': 'Counter Circ.',
    }
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc'
    base_name = 'resp_proxy_temps_inf_1'
    temperatures = np.linspace(0, 10, 100)
    num_seeds = 5
    
    for prefix, _ in name_dict.items():
        result_list = []
        for seed in range(num_seeds):
            with open(save_path / f'{base_name}_{prefix}_{seed}.pkl', 'rb') as f:
                cur_results = pickle.load(f)
            result_list.append(cur_results)
        result_arr = np.asarray(result_list)[:, :, 0, :]
        result_arr = np.transpose(result_arr, axes=(1, 0, 2)).reshape(len(temperatures), -1)
        
        plt.clf()
        plt.figure(dpi=600)
        seaborn.set_theme(style='whitegrid')
        plot_filled_std_curves(
            x=temperatures,
            mean=result_arr.mean(axis=-1),
            std=result_arr.std(axis=-1),
            color=COLORS[0],
            lighter_color=LIGHT_COLORS[0],
            linestyle=LINESTYLES[0],
            label=None,
            min_val=0,
        )
        
        fontsize = 'medium'
        plt.xlabel('Temperature', fontsize=fontsize)
        plt.ylabel('Reward', fontsize=fontsize)
        plt.xlim(temperatures[0], temperatures[-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'resp_proxy_{prefix}.png')



def plot_both():
    name_dict = {
        'cr': 'Cramped Rm.',
        'aa': 'Asym. Adv.',
        'co': 'Coord. Ring',
        'fc': 'Forced Coord.',
        'cc': 'Counter Circ.',
    }
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc'
    base_name_resp = 'resp_proxy_temps_inf_1'
    base_name_proxy = 'proxy_proxy_temps_1'
    temperatures = np.linspace(0, 10, 100)
    num_seeds = 5
    
    for idx, (prefix, full_name) in enumerate(name_dict.items()):
        result_list = []
        for seed in range(num_seeds):
            with open(save_path / f'{base_name_resp}_{prefix}_{seed}.pkl', 'rb') as f:
                cur_results = pickle.load(f)
            result_list.append(cur_results)
        result_arr = np.asarray(result_list)[:, :, 0, :]
        result_arr_resp = np.transpose(result_arr, axes=(1, 0, 2)).reshape(len(temperatures), -1)
        
        result_list = []
        for seed in range(num_seeds):
            with open(save_path / f'{base_name_proxy}_{prefix}_{seed}.pkl', 'rb') as f:
                cur_results = pickle.load(f)
            result_list.append(cur_results)
        result_arr = np.asarray(result_list)[:, :, 0, :]
        result_arr_proxy = np.transpose(result_arr, axes=(1, 0, 2)).reshape(len(temperatures), -1)
        
        plt.clf()
        plt.figure(figsize=(5, 5))
        seaborn.set_theme(style='whitegrid')
        
        plot_filled_std_curves(
            x=temperatures,
            mean=result_arr_resp.mean(axis=-1),
            std=result_arr_resp.std(axis=-1),
            color=COLORS[0],
            lighter_color=LIGHT_COLORS[0],
            linestyle=LINESTYLES[0],
            label='Alb + Proxy' if idx == 0 else None,
            min_val=0,
        )
        
        plot_filled_std_curves(
            x=temperatures,
            mean=result_arr_proxy.mean(axis=-1),
            std=result_arr_proxy.std(axis=-1),
            color=COLORS[1],
            lighter_color=LIGHT_COLORS[1],
            linestyle=LINESTYLES[1],
            label='Proxy + Proxy' if idx == 0 else None,
            min_val=0,
        )
        
        fontsize = 'xx-large'
        plt.xlim(temperatures[0], temperatures[-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(full_name, fontsize=fontsize)
        if idx == 0:
            plt.legend(fontsize=fontsize)
            plt.ylabel('Reward', fontsize=fontsize)
        plt.xlabel('Temperature', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'both_resp_proxy_{prefix}.pdf', bbox_inches='tight', pad_inches=0.0)


def plot_both_ood():
    name_dict = {
        'cr': 'Cramped Rm.',
        'aa': 'Asym. Adv.',
        'co': 'Coord. Ring',
        'fc': 'Forced Coord.',
        'cc': 'Counter Circ.',
    }
    
    save_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc'
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc'
    base_name_resp = 'resp_proxy_temps_inf_1'
    base_name_proxy = 'proxy_proxy_temps_1'
    base_name_proxy_ood = 'proxy_proxy_temps_1_ood'
    temperatures = np.linspace(0, 10, 100).tolist()
    temperatures_ood = np.linspace(-5, 0, 15)[:-1].tolist() + np.linspace(10, 15, 15)[1:].tolist()
    num_seeds = 5
    
    x_ood = np.concatenate([
            temperatures_ood[:14],
            temperatures,
            temperatures_ood[14:],
        ],
    )
    
    for idx, (prefix, full_name) in enumerate(name_dict.items()):
        result_list = []
        for seed in range(num_seeds):
            with open(save_path / f'{base_name_resp}_{prefix}_{seed}.pkl', 'rb') as f:
                cur_results = pickle.load(f)
            result_list.append(cur_results)
        result_arr = np.asarray(result_list)[:, :, 0, :]
        result_arr_resp = np.transpose(result_arr, axes=(1, 0, 2)).reshape(len(temperatures), -1)
        
        result_list = []
        for seed in range(num_seeds):
            with open(save_path / f'{base_name_proxy}_{prefix}_{seed}.pkl', 'rb') as f:
                cur_results = pickle.load(f)
            result_list.append(cur_results)
        result_arr = np.asarray(result_list)[:, :, 0, :]
        result_arr_proxy = np.transpose(result_arr, axes=(1, 0, 2)).reshape(len(temperatures), -1)
        
        result_list = []
        for seed in range(num_seeds):
            with open(save_path / f'{base_name_proxy_ood}_{prefix}_{seed}.pkl', 'rb') as f:
                cur_results = pickle.load(f)
            result_list.append(cur_results)
        result_arr = np.asarray(result_list)[:, :, 0, :]
        result_arr_proxy_ood = np.transpose(result_arr, axes=(1, 0, 2)).reshape(len(temperatures_ood), -1)
        result_arr_proxy_ood = np.concatenate([
                result_arr_proxy_ood[:14],
                result_arr_proxy,
                result_arr_proxy_ood[14:],
            ],
            axis=0,
        )
        
        plt.clf()
        plt.figure(figsize=(8, 8))
        seaborn.set_theme(style='whitegrid')
        
        plot_filled_std_curves(
            x=temperatures,
            mean=result_arr_resp.mean(axis=-1),
            std=result_arr_resp.std(axis=-1),
            color=COLORS[0],
            lighter_color=LIGHT_COLORS[0],
            linestyle=LINESTYLES[0],
            label='Alb + Proxy' if idx == 0 else None,
            min_val=0,
        )
        
        plot_filled_std_curves(
            x=x_ood,
            mean=result_arr_proxy_ood.mean(axis=-1),
            std=result_arr_proxy_ood.std(axis=-1),
            color=COLORS[1],
            lighter_color=LIGHT_COLORS[1],
            linestyle=LINESTYLES[1],
            label='Proxy + Proxy' if idx == 0 else None,
            min_val=0,
        )
        
        
        fontsize = 'xx-large'
        plt.xlim(x_ood[0], x_ood[-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(full_name, fontsize=fontsize)
        if idx == 0:
            plt.legend(fontsize=fontsize)
            plt.ylabel('Reward', fontsize=fontsize)
            plt.xlabel('Temperature', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(img_path / f'both_ood_resp_proxy_{prefix}.pdf', bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    # plot_proxy_vs_proxy()
    # plot_resp_vs_proxy()
    plot_both()
