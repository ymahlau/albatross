import sys

import torch.multiprocessing as mp

from scripts.logit_solver.run_logit_experiments import create_gt_logit_data

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # this is important for using CUDA
    print(f"{mp.get_start_method()=}")

    num_iterations = int(sys.argv[1])

    create_gt_logit_data()



    # compute_equilibria()

    # n = int(sys.argv[1])
    # l_id = math.floor(n / 5)
    # seed = n % 5
    # estimate_bc_strength(layout_id=l_id, seed=seed)

    # play_albatross_oc()
    # play_albatross_oc_self()

