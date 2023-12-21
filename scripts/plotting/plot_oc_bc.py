


from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn


data_pecan = [
    (135.53223, 138.23088, 141.82909),
    (123.53823, 150.82459, 177.51124),
    (122.33883, 127.43628, 134.33283),
    (32.08396, 37.18141, 43.77811),
    (49.17541, 51.87406, 55.77211),
]

data_mep = [
    (119.64018, 128.33583, 135.23238), 
    (104.34783, 120.53973, 137.63118), 
    (101.04948, 109.14543, 117.24138), 
    (21.88906, 26.38681, 31.18441), 
    (36.88156, 45.27736, 53.67316)
]

data_trajdi = [
    (115.14243, 117.84108, 121.13943), 
    (84.25787, 91.75412, 100.14993), 
    (82.15892, 95.65217, 110.34483), 
    (21.58921, 24.28786, 28.18591), 
    (42.27886, 45.27736, 49.47526)
]

data_fcp = [
    (102.24888, 109.74513, 118.14093), 
    (78.26087, 83.65817, 90.25487), 
    (80.35982, 93.55322, 108.24588),
    (25.78711, 29.08546, 32.68366), 
    (27.28636, 33.28336, 40.17991)
]

data_pbt = [
    (101.94903, 113.34333, 130.13493), 
    (86.95652, 92.65367, 99.85007), 
    (58.47076, 70.46477, 79.16042), 
    (10.79460, 13.49325, 17.09145), 
    (25.48726, 32.38381, 39.88006)
]

data_sp = [
    (105.54723, 111.24438, 117.54123), 
    (75.56222, 91.75412, 108.84558), 
    (46.77661, 66.56672, 90.25487), 
    (17.99100, 23.08846, 28.48576), 
    (26.68666, 31.78411, 39.28036)
]

name_dict = {
    'cr': 'Cramped Rm.',
    'aa': 'Asym. Adv.',
    'co': 'Coord. Ring',
    'fc': 'Forced Coord.',
    'cc': 'Counter Circ.',
}

def load_albatross_data(base_name: str):
    data_path = Path(__file__).parent.parent.parent / 'a_data' / 'oc'
    full_list = []
    for prefix in name_dict.keys():
        # if prefix == 'aa' or prefix == 'fc':
        #     cur_data = np.zeros((3), dtype=float)
        # else:
        cur_list = []
        for seed in range(5):
            with open(data_path / f'{base_name}_{prefix}_{seed}.pkl', 'rb') as f:
                cur_list.append(pickle.load(f)[0])
        raw_data = np.mean(np.asarray(cur_list), axis=1)
        mean = np.mean(raw_data, axis=0)
        std = np.std(raw_data, axis=0)
        cur_data = np.stack([mean-std, mean, mean+std])
        full_list.append(cur_data)
    return np.asarray(full_list)

def main():
    img_name = 'temp.png'
    method_dict = {
        # 'Alb+BC': load_albatross_data('resp_20from200'),
        # 'Alb+Alb': load_albatross_data('resp_20from200_self'),
        # 'Alb+Proxy10': load_albatross_data('resp_20from200_proxy'),
        # 'Alb10+Proxy10': load_albatross_data('resp_proxy_inf'),
        # 'Proxy10+Proxy10': load_albatross_data('proxy_proxy_inf'),
        # 'Alb10+BC': load_albatross_data('resp10_bc'),
        # 'BC+BC': load_albatross_data('bc_bc_1'),
        # 'AlbN+BC': load_albatross_data('resp_20n1_bc'),
        # 'Alb10+Alb10': load_albatross_data('resp_resp_10'),
        # 'AlbMLE+BC': load_albatross_data('resp_mle_bc'),
        'AlbMLE1+BC': load_albatross_data('resp_mle_bc_1'),
        # 'Proxy10+BC': load_albatross_data('proxy_bc_10_inf'),
        'PECAN+BC': data_pecan,
        'MEP+BC': data_mep,
        'TrajeDi+BC': data_trajdi,
        'FCP+BC': data_fcp,
        'PBT+BC': data_pbt,
        'PPO+BC': data_sp,
    }
    
    # Sample data
    layouts = list(name_dict.values())
    methods = list(method_dict.keys())

    # # Generating random data for illustration
    # data = np.random.rand(len(layouts), len(methods))
    
    # # Generating random error bar values
    # error_min = np.random.rand(len(layouts), len(methods))
    # error_max = error_min + np.random.rand(len(layouts), len(methods))
    
    mean_data = np.asarray(list(method_dict.values()))[..., 1].T
    
    min_data = np.asarray(list(method_dict.values()))[..., 0].T
    min_err = mean_data - min_data
    
    max_data = np.asarray(list(method_dict.values()))[..., 2].T
    max_err = max_data - mean_data
    
    # Setting up positions for bars
    bar_width = 0.12
    bar_positions = np.arange(len(layouts))

    plt.clf()
    seaborn.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 10), dpi=600)
    
    
    # Creating grouped bar plot
    for i, setting in enumerate(methods):
        # plt.bar(bar_positions + i * bar_width, data[:, i], width=bar_width, label=setting)
        plt.bar(
            bar_positions + i * bar_width,
            mean_data[:, i],
            width=bar_width,
            yerr=[min_err[:, i], max_err[:, i]],
            label=setting,
            capsize=3,  # Specify the cap size for error bars
        )

    # Adding labels and title
    plt.xticks(bar_positions + bar_width * 3, layouts)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=int(len(method_dict)/3))

    # Displaying the plot
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'oc' / img_name
    plt.tight_layout()
    plt.savefig(img_path)


if __name__ == '__main__':
    main()

