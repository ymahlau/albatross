import sys
from pathlib import Path

import multiprocessing as mp
from scripts.depth.bs_proxy_temps import alb_vs_proxy_at_temps, albfix_vs_proxy_at_temps
from scripts.depth.estimate_area_strength import evaluate_bs_depth_strength
from scripts.depth.estimate_az_strength import estimate_az
from scripts.depth.evaluate_bs_depth import evaluate_bs_depth_func
from scripts.depth.evaluate_bs_depth_coop import evaluate_bs_depth_func_coop
from scripts.depth.kl_div_vs_base import save_policies_at_depth
from scripts.eval_oc.entropy_proxy import entropy_proxy_eval
from scripts.eval_oc.overcooked_albfix_bc import eval_albfix_vs_bc
from scripts.eval_oc.overcooked_bc_eval import evaluate_alb_proxy, evaluate_bc_bc, evaluate_overcooked_response, evaluate_overcooked_response_mle, evaluate_overcooked_response_normal, evaluate_proxy_bc, evaluate_proxy_proxy, evaluate_resp10_bc, evaluate_resp_proxy, evaluate_resp_resp, evaluate_resp_resp_10
from scripts.eval_oc.overcooked_proxy_temps import eval_proxy_different_temps, eval_resp_fixed_proxy, eval_resp_proxy_different_temps
from scripts.eval_oc.scripted_overcooked import eval_scripted_oc
from scripts.eval_oc.temp_convergence import temp_convergence_run
from scripts.logit_solver.run_logit_experiments import generate_gt_data, step_size_experiment
from scripts.temp_est.estimate_bc import estimate_bc_strength
from scripts.temp_est.estimate_proxy_oc import estimate_oc_proxy_strength

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
    # evaluate_bs_depth_func(experiment_id)
    # save_policies_at_depth(experiment_id)
    # evaluate_bs_depth_strength(experiment_id)
    # estimate_bc_strength(experiment_id)
    # estimate_oc_proxy_strength(experiment_id)
    # estimate_az(experiment_id)
    # generate_gt_data()
    # step_size_experiment()
    # eval_albfix_vs_bc(experiment_id)
    # alb_vs_proxy_at_temps(experiment_id)
    # albfix_vs_proxy_at_temps(experiment_id)
    # entropy_proxy_eval(experiment_id)
    # eval_resp_fixed_proxy(experiment_id)
    # eval_scripted_oc(experiment_id)
    # temp_convergence_run(experiment_id)
    evaluate_bs_depth_func_coop(experiment_id)
    
    
    # num_iterations = int(sys.argv[1])
    # generate_experiment_data(num_iterations)

    # compute_equilibria()

    # n = int(sys.argv[1])
    # l_id = math.floor(n / 5)
    # seed = n % 5
    # estimate_bc_strength(layout_id=l_id, seed=seed)

    # play_albatross_oc()
    # play_albatross_oc_self()

