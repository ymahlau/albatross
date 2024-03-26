

import numpy as np
from src.equilibria.logit import compute_logit_equilibrium
from src.modelling.mle import compute_likelihood, compute_temperature_mle


def main_matrix_game_repeated():
    # compute le with temperature 10 as ground truth best play
    ja_vals = np.asarray([[4, 4], [0, 0], [1, 1], [2, 2]])
    
    val, pol, err = compute_logit_equilibrium(
        available_actions=[[0, 1], [0, 1]],
        joint_action_list=[(0, 0), (0, 1), (1, 0), (1, 1)],
        joint_action_value_arr=ja_vals,
        num_iterations=int(1e6),
        epsilon=0,
        temperatures=[10, 10],
    )
    p1_logit_pol = pol[0]
    q_a = p1_logit_pol[0] * ja_vals[0, 1] + p1_logit_pol[1] * ja_vals[2, 1]
    q_b = p1_logit_pol[0] * ja_vals[1, 1] + p1_logit_pol[1] * ja_vals[3, 1]
    
    p2_actions = [0, 1, 0, 0]
    utils = [[q_a, q_b] for _ in range(len(p2_actions))]
    
    cur_temperature = compute_temperature_mle(
        min_temp=-10,
        max_temp=10,
        num_iterations=20,
        chosen_actions=p2_actions,
        utilities=utils,
    )
    
    val_log, pol_log, err = compute_logit_equilibrium(
        available_actions=[[0, 1], [0, 1]],
        joint_action_list=[(0, 0), (0, 1), (1, 0), (1, 1)],
        joint_action_value_arr=ja_vals,
        num_iterations=int(1e6),
        epsilon=0,
        temperatures=[cur_temperature, cur_temperature],
    )
    q_a_p1 = pol_log[1][0] * ja_vals[0, 1] + pol_log[1][1] * ja_vals[1, 1]
    q_b_p1 = pol_log[1][0] * ja_vals[2, 1] + pol_log[1][1] * ja_vals[3, 1]
    
    a = 1
    
    # for t in range(1, num_steps + 1):
    #     utils = [[q_a, q_b] for _ in range(t)]
        
    #     p2_actions = [0 for _ in range(t)]
        
    #     cur_temperature = compute_temperature_mle(
    #         min_temp=0,
    #         max_temp=10,
    #         num_iterations=20,
    #         chosen_actions=p2_actions,
    #         utilities=utils,
    #     )
    #     all_estimates.append(cur_temperature)        
        
    a = 1
    
    



if __name__ == '__main__':
    main_matrix_game_repeated()
