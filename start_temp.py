import sys
from pathlib import Path

import multiprocessing as mp
from scripts.depth.evaluate_bs_depth import evaluate_bs_depth_func
from scripts.eval.overcooked_bc_eval import evaluate_alb_proxy, evaluate_bc_bc, evaluate_overcooked_response, evaluate_overcooked_response_mle, evaluate_overcooked_response_normal, evaluate_proxy_bc, evaluate_proxy_proxy, evaluate_resp10_bc, evaluate_resp_proxy, evaluate_resp_resp, evaluate_resp_resp_10
from scripts.eval.overcooked_proxy_temps import eval_proxy_different_temps, eval_resp_proxy_different_temps

# from scripts.logit_solver.run_logit_experiments import generate_experiment_data, create_logit_data_func

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

    experiment_id = int(sys.argv[1])
    # evaluate_overcooked_response(experiment_id)
    # evaluate_bc_bc(experiment_id)
    # evaluate_resp_resp(experiment_id)
    # evaluate_alb_proxy(experiment_id)
    # evaluate_resp_proxy(experiment_id)
    # evaluate_proxy_proxy(experiment_id)
    # evaluate_resp10_bc(experiment_id)
    # evaluate_overcooked_response_normal(experiment_id)
    # eval_proxy_different_temps(experiment_id)
    # evaluate_overcooked_response_mle(experiment_id)
    # evaluate_resp_resp_10(experiment_id)
    # evaluate_proxy_bc(experiment_id)
    # eval_resp_proxy_different_temps(experiment_id)
    evaluate_bs_depth_func(experiment_id)
    
    
    # num_iterations = int(sys.argv[1])
    # generate_experiment_data(num_iterations)

    # compute_equilibria()

    # n = int(sys.argv[1])
    # l_id = math.floor(n / 5)
    # seed = n % 5
    # estimate_bc_strength(layout_id=l_id, seed=seed)

    # play_albatross_oc()
    # play_albatross_oc_self()

