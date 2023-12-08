import sys
from pathlib import Path

import multiprocessing as mp

from scripts.logit_solver.run_logit_experiments import generate_experiment_data, create_logit_data_func

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # this is important for using CUDA
    print(f"{mp.get_start_method()=}")

    # create_logit_data_func(
    #     num_iterations=10,
    #     save_path=str(Path(__file__).parent.parent / 'a_data' / 'logit_solver' / 'ground_truth_fc_0_10.pkl'),
    #     sbr_mode_str='MSA',
    #     hp_0=None,
    #     hp_1=None,
    #     gt_path=None,
    # )

    num_iterations = int(sys.argv[1])
    generate_experiment_data(num_iterations)



    # compute_equilibria()

    # n = int(sys.argv[1])
    # l_id = math.floor(n / 5)
    # seed = n % 5
    # estimate_bc_strength(layout_id=l_id, seed=seed)

    # play_albatross_oc()
    # play_albatross_oc_self()

