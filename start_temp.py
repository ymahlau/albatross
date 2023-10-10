import math
import sys
import torch.multiprocessing as mp

# from scripts.plots.plot_modelling_error import compute_equilibria

# from src.evaluation.estimate_bc_strength import estimate_bc_strength
# from scripts.runs.play_config_oc_self import play_albatross_oc_self
# from scripts.runs.play_config_oc_albatross import play_albatross_oc
# from src.depth.training_strength import estimate_training_strength
# from scripts.temp import main, merge_depth_logs, random_efg_experiment

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # this is important for using CUDA
    print(f"{mp.get_start_method()=}")

    # compute_equilibria()

    # n = int(sys.argv[1])
    # l_id = math.floor(n / 5)
    # seed = n % 5
    # estimate_bc_strength(layout_id=l_id, seed=seed)

    # play_albatross_oc()
    # play_albatross_oc_self()

