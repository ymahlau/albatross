from pathlib import Path
import pickle
from matplotlib import pyplot as plt

import numpy as np
import seaborn

def load_albatross_data(base_name: str):
    data_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc'
    full_list = []
    prefix = 'cr'
    cur_list = []
    for seed in range(5):
        with open(data_path / f'{base_name}_{prefix}_{seed}.pkl', 'rb') as f:
            cur_list.append(pickle.load(f)[0])
    raw_data = np.mean(np.asarray(cur_list), axis=1)
    return raw_data

def plot_brle():
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc'
    method_dict = {
        'Alb. (SBRLE)': load_albatross_data('resp_mle_bc_1'),
        'Alb. (BRLE)': load_albatross_data('resp_brle_mle_bc_1')
    }
    data = list(method_dict.values())
    labels = list(method_dict.keys())
    
    plt.clf()
    seaborn.set_theme(style='whitegrid')
    plt.figure(figsize=(5, 5), dpi=600)
    
    # for idx, (name, data) in enumerate(method_dict.items()):
    #     x_list = [idx for _ in data]
    #     plt.scatter(x_list, data, label=name)
    
    plt.boxplot(data, labels=labels, showfliers=True)
    
    fontsize = 'medium'
    # plt.xlabel('Temperature', fontsize=fontsize)
    # plt.xticks([])
    plt.ylabel('Reward', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(img_path / f'brle.pdf')


if __name__ == '__main__':
    plot_brle()
