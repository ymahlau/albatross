//
// Created by mahla on 17/06/2023.
//

#include "../header/utils.h"
#include <cfloat>
#include <numeric>
#include <random>

random_device rd_quantal;
mt19937 gen_quantal(rd_quantal());

///////////////////////////////////////////////////////////////////
// Below is Code for Quantal Nash Equilibrium Computation
// See https://ojs.aaai.org/index.php/AAAI/article/view/16701
///////////////////////////////////////////////////////////////////

void rm_qr(
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        const int* joint_actions, // shape (prod(num_available_actions) * num_player)
        const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
        int leader,
        int num_iterations,
        double temperature,
        double random_prob,
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
){
    //compute number of joint actions and sum of all player actions
    int num_joint_actions = 1;
    int num_all_actions = 0;
    for (int p = 0; p < 2; p++){
        num_joint_actions *= num_available_actions[p];
        num_all_actions += num_available_actions[p];
    }
    // leader and follower actions
    int follower = 1 - leader;
    vector<int> leader_actions(num_available_actions[leader], 0);
    vector<double> leader_policy(num_available_actions[leader], 1.0 / num_available_actions[leader]);
    vector<double> leader_policy_sum(num_available_actions[leader], 0);
    int offset_leader = leader * num_available_actions[follower];
    for (int a_idx = 0; a_idx < num_available_actions[leader]; a_idx++){
        leader_actions[a_idx] = available_actions[offset_leader + a_idx];
    }
    vector<int> follower_actions(num_available_actions[follower], 0);
    vector<double> follower_policy(num_available_actions[follower], 1.0 / num_available_actions[follower]);
    int offset_follower = follower * num_available_actions[leader];
    for (int a_idx = 0; a_idx < num_available_actions[follower]; a_idx++){
        follower_actions[a_idx] = available_actions[offset_follower + a_idx];
    }
    // outcome/utility hashmap: leader action is first index: (a0, a1) -> (u0, u1)
    unordered_map<pair<int, int>, pair<double, double>, hashed_int_pair> outcomes;
    for (int ja_idx = 0; ja_idx < num_joint_actions; ja_idx++){
        int a0 = joint_actions[ja_idx * 2];
        int a1 = joint_actions[ja_idx * 2 + 1];
        double u0 = joint_action_values[ja_idx * 2];
        double u1 = joint_action_values[ja_idx * 2 + 1];
        if (leader == 0){
            outcomes[{a0, a1}] = {u0, u1};
        } else {
            outcomes[{a1, a0}] = {u1, u0};
        }
    }
    // initialize regret vector and q-values
    vector<double> regrets(num_available_actions[leader], 0);
    vector<double> follower_q(num_available_actions[follower], 0);
    vector<double> leader_q(num_available_actions[leader], 0);
    // iterate
    for (int i = 0; i < num_iterations; i ++){
        // compute positive regret sum
        double positive_regret_sum = 0;
        for (double r: regrets){
            if (r > 0){
                positive_regret_sum += r;
            }
        }
        // compute leader policy based on regret matching
        double uniform_p = 1.0 / num_available_actions[leader];
        if (positive_regret_sum > 1e-4){
            // policy proportional to regret
            for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++){
                // exploitation
                if (regrets[la_idx] > 0){
                    leader_policy[la_idx] = regrets[la_idx] / positive_regret_sum;
                } else {
                    leader_policy[la_idx] = 0;
                }
                // random exploration prob
                leader_policy[la_idx] = random_prob * uniform_p + (1 - random_prob) * leader_policy[la_idx];
            }
        } else {
            // uniform policy
            for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++){
                leader_policy[la_idx] = uniform_p;
            }
        }
        // update leader policy sum
        for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++){
            leader_policy_sum[la_idx] += leader_policy[la_idx];
        }
        // compute q-values for follower given the leader policy
        double exp_q_sum = 0;
        for (int fa_idx = 0; fa_idx < num_available_actions[follower]; fa_idx++){
            int fa = follower_actions[fa_idx];
            double cur_q = 0;
            for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++){
                cur_q += leader_policy[la_idx] * outcomes[{leader_actions[la_idx], fa}].second;
            }
            follower_q[fa_idx] = cur_q;
            exp_q_sum += exp(temperature * cur_q);
        }
        // compute followers quantal response based on q-values
        for (int fa_idx = 0; fa_idx < num_available_actions[follower]; fa_idx++){
            follower_policy[fa_idx] = exp(temperature * follower_q[fa_idx]) / exp_q_sum;
        }
        // compute q-values for leader
        for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++){
            int la = leader_actions[la_idx];
            double cur_q = 0;
            for (int fa_idx = 0; fa_idx < num_available_actions[follower]; fa_idx++){
                cur_q += follower_policy[fa_idx] * outcomes[{la, follower_actions[fa_idx]}].first;
            }
            leader_q[la_idx] = cur_q;
        }
        for (int other_a_idx = 0; other_a_idx < num_available_actions[leader]; other_a_idx++){
            double cur_regret = 0;
            for (int chosen_a_idx = 0; chosen_a_idx < num_available_actions[leader]; chosen_a_idx++){
                cur_regret += leader_policy[chosen_a_idx] * (leader_q[other_a_idx] - leader_q[chosen_a_idx]);
            }
            regrets[other_a_idx] += cur_regret;
        }
    }
    // extract final leader policy
    for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++){
        leader_policy[la_idx] = leader_policy_sum[la_idx] / (double) num_iterations;
    }
    // compute final follower q
    double exp_q_sum = 0;
    for (int fa_idx = 0; fa_idx < num_available_actions[follower]; fa_idx++){
        int fa = follower_actions[fa_idx];
        double cur_q = 0;
        for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++){
            cur_q += leader_policy[la_idx] * outcomes[{leader_actions[la_idx], fa}].second;
        }
        follower_q[fa_idx] = cur_q;
        exp_q_sum += exp(temperature * cur_q);
    }
    // compute followers quantal response based on q-values
    for (int fa_idx = 0; fa_idx < num_available_actions[follower]; fa_idx++){
        follower_policy[fa_idx] = exp(temperature * follower_q[fa_idx]) / exp_q_sum;
    }
    // update result arrays
    for (int a_idx = 0; a_idx < num_available_actions[0]; a_idx++){
        if (leader == 0){
            result_policies[a_idx] = leader_policy[a_idx];
        } else {
            result_policies[a_idx] = follower_policy[a_idx];
        }
    }
    int offset = num_available_actions[0];
    for (int a_idx = 0; a_idx < num_available_actions[1]; a_idx++){
        if (leader == 1){
            result_policies[a_idx + offset] = leader_policy[a_idx];
        } else {
            result_policies[a_idx + offset] = follower_policy[a_idx];
        }
    }
    // extract final values
    double l_v = 0;
    double f_v = 0;
    for (int fa_idx = 0; fa_idx < num_available_actions[follower]; fa_idx++) {
        int fa = follower_actions[fa_idx];
        double fa_prob = follower_policy[fa_idx];
        for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++) {
            int la = leader_actions[la_idx];
            double probs = leader_policy[la_idx] * fa_prob;
            l_v += outcomes[{la, fa}].first * probs;
            f_v += outcomes[{la, fa}].second * probs;
        }
    }
    if (leader == 0){
        result_values[0] = l_v;
        result_values[1] = f_v;
    } else {
        result_values[0] = f_v;
        result_values[1] = l_v;
    }
}






///////////////////////////////////////////////////////////////////
// Below is Code for Quantal Stackelberg equilibrium Computation
// See https://www.ijcai.org/proceedings/2020/35 for more information
///////////////////////////////////////////////////////////////////

double eval_dinkelbach_subproblem(
        unordered_map<pair<int, int>, pair<double, double>, hashed_int_pair>& outcomes,
        const vector<int>& leader_actions,
        const vector<int>& follower_actions,
        const vector<double>& leader_policy,
        double p,
        double temperature
){
    // calculates value of equation 6 in the paper
    double result = 0;
    for (int fa: follower_actions){
        // compute utility (leader and policy) for mixed leader and pure follower strategy
        double utility_leader = 0;
        double utility_follower = 0;
        for (int la_idx = 0; la_idx < (int)leader_actions.size(); la_idx++) {
            pair<double, double> u = outcomes[{leader_actions[la_idx], fa}];
            utility_leader += leader_policy[la_idx] * u.first;
            utility_follower += leader_policy[la_idx] * u.second;
        }
        //increment result by current term
        result += (utility_leader - p) * exp(temperature * utility_follower);
    }
    return result;
}

pair<double, vector<double>> solve_dinkelbach_subproblem(
        unordered_map<pair<int, int>, pair<double, double>, hashed_int_pair>& outcomes,
        const vector<int>& leader_actions,
        const vector<int>& follower_actions,
        double p,
        double temperature,
        int grid_size
){
    // solves equation 6 in the paper by finding the approximate maximum using grid search over probability simplex
    auto max_value = -DBL_MAX;
    vector<double> best_policy(leader_actions.size(), -1);
    vector<int> leader_action_indices(leader_actions.size());
    iota(leader_action_indices.begin(), leader_action_indices.end(), 0);
    // perform grid search for maximum
    //iterate support sizes
    for (int support_size = 1; support_size <= (int) leader_actions.size(); support_size++){
        // iterate supports with given support size
        deque<vector<int>> supports_of_size = combinations(leader_action_indices, support_size);
        for(vector<int> support: supports_of_size){
            vector<double> cur_policy(leader_actions.size(), 0);
            double value;
            if (support_size == 1){
                // only a single policy exists with support size one
                cur_policy[support[0]] = 1;
                value = eval_dinkelbach_subproblem(
                    outcomes,
                    leader_actions,
                    follower_actions,
                    cur_policy,
                    p,
                    temperature
                );
                // if the current evaluation was better than best value found so far, update it
                if (value > max_value){
                    max_value = value;
                    for (int la_idx = 0; la_idx < (int) leader_actions.size(); la_idx++){
                        best_policy[la_idx] = cur_policy[la_idx];
                    }
                }
            } else {
                // perform grid search within the current support
                double step_size = 1.0 / (double) grid_size;
                vector<int> steps(support_size - 1, 1);
                while (steps[0] <= grid_size - support_size + 1){
                    // set current policy
                    double prob_sum = 0;
                    for (int step_idx = 0; step_idx < support_size - 1; step_idx++){
                        double cur_prob = step_size * steps[step_idx];
                        cur_policy[support[step_idx]] = cur_prob;
                        prob_sum += cur_prob;
                    }
                    cur_policy[support[support_size - 1]] = 1 - prob_sum;
                    // evaluate current policy
                    value = eval_dinkelbach_subproblem(
                            outcomes,
                            leader_actions,
                            follower_actions,
                            cur_policy,
                            p,
                            temperature
                    );
                    // if the current evaluation was better than best value found so far, update it
                    if (value > max_value){
                        max_value = value;
                        for (int la_idx = 0; la_idx < (int) leader_actions.size(); la_idx++){
                            best_policy[la_idx] = cur_policy[la_idx];
                        }
                    }
                    // next grid point
                    for (int step_idx = support_size - 2; step_idx >= 0; step_idx--){
                        steps[step_idx] += 1;
                        // if next point is out of simplex, step further
                        int step_sum = 0;
                        for (int s: steps) step_sum += s;
                        if (step_idx != 0 and step_sum > grid_size - 1){
                            steps[step_idx] = 1;  // reset current step index and go to next dimension
                        } else {
                            break;  // this point is valid probability distribution
                        }
                    }
                }
            }
        }
    }
    return {max_value, best_policy};
}

void qse(
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        const int* joint_actions, // shape (prod(num_available_actions) * num_player)
        const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
        int leader,
        int num_iterations,
        int grid_size,
        double temperature,
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
){
    //compute number of joint actions and sum of all player actions
    int num_joint_actions = 1;
    int num_all_actions = 0;
    for (int p = 0; p < 2; p++){
        num_joint_actions *= num_available_actions[p];
        num_all_actions += num_available_actions[p];
    }
    // leader and follower actions
    int follower = 1 - leader;
    vector<int> leader_actions(num_available_actions[leader], 0);
    int offset_leader = leader * num_available_actions[follower];
    for (int a_idx = 0; a_idx < num_available_actions[leader]; a_idx++){
        leader_actions[a_idx] = available_actions[offset_leader + a_idx];
    }
    vector<int> follower_actions(num_available_actions[follower], 0);
    int offset_follower = follower * num_available_actions[leader];
    for (int a_idx = 0; a_idx < num_available_actions[follower]; a_idx++){
        follower_actions[a_idx] = available_actions[offset_follower + a_idx];
    }
    // Our algorithm does not work if there are more actions than grid_size
    if (grid_size <= available_actions[leader] + 1) return;
    // outcome/utility hashmap: leader action is first index: (a0, a1) -> (u0, u1)
    unordered_map<pair<int, int>, pair<double, double>, hashed_int_pair> outcomes;
    // also calculate lower and upper bound for leader value simultaneously
    auto lower_bound = DBL_MAX;
    auto upper_bound = -DBL_MAX;
    for (int ja_idx = 0; ja_idx < num_joint_actions; ja_idx++){
        int a0 = joint_actions[ja_idx * 2];
        int a1 = joint_actions[ja_idx * 2 + 1];
        double u0 = joint_action_values[ja_idx * 2];
        double u1 = joint_action_values[ja_idx * 2 + 1];
        double leader_util;
        if (leader == 0){
            outcomes[{a0, a1}] = {u0, u1};
            leader_util = u0;
        } else {
            outcomes[{a1, a0}] = {u1, u0};
            leader_util = u1;
        }
        if (leader_util < lower_bound) lower_bound = leader_util;
        if (leader_util > upper_bound) upper_bound = leader_util;
    }
    // lower bound on best policy
//    pair<double, vector<double>> init_solution = solve_dinkelbach_subproblem(
//        outcomes,
//        leader_actions,
//        follower_actions,
//        lower_bound,
//        temperature,
//        grid_size
//    );
//    vector<double> best_leader_policy = init_solution.second;
    // perform binary line search
    vector<double> leader_policy;
    for (int i = 0; i < num_iterations; i++){
        double p = (upper_bound + lower_bound) / 2.0;
        // solve subproblem
        pair<double, vector<double>> cur_solution = solve_dinkelbach_subproblem(
                outcomes,
                leader_actions,
                follower_actions,
                p,
                temperature,
                grid_size
        );
        // update boundary and best solution found
        leader_policy = cur_solution.second;
        if (cur_solution.first > 0){
            lower_bound = p;
        } else {
            upper_bound = p;
        }
    }
    // compute followers quantal response
    // compute q-values for follower given the leader policy
    double exp_q_sum = 0;
    vector<double> follower_q(num_available_actions[follower], 0);
    vector<double> follower_policy(num_available_actions[follower], -1);
    for (int fa_idx = 0; fa_idx < num_available_actions[follower]; fa_idx++){
        int f_a = follower_actions[fa_idx];
        double cur_q = 0;
        for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++){
            cur_q += leader_policy[la_idx] * outcomes[{leader_actions[la_idx], f_a}].second;
        }
        follower_q[fa_idx] = cur_q;
        exp_q_sum += exp(temperature * cur_q);
    }
    // compute followers quantal response based on q-values
    for (int f_a_idx = 0; f_a_idx < num_available_actions[follower]; f_a_idx++){
        follower_policy[f_a_idx] = exp(temperature * follower_q[f_a_idx]) / exp_q_sum;
    }
    // compute final values for both players
    double v_leader = 0;
    double v_follower = 0;
    for (int fa_idx = 0; fa_idx < num_available_actions[follower]; fa_idx++){
        int fa = follower_actions[fa_idx];
        for (int la_idx = 0; la_idx < num_available_actions[leader]; la_idx++){
            int la = leader_actions[la_idx];
            double joint_prob = follower_policy[fa_idx] * leader_policy[la_idx];
            pair<double, double> u = outcomes[{la, fa}];
            v_leader += joint_prob * u.first;
            v_follower += joint_prob * u.second;
        }
    }
    // update value array
    result_values[leader] = v_leader;
    result_values[follower] = v_follower;
    // update policy result array
    for (int a_idx = 0; a_idx < num_available_actions[0]; a_idx++){
        if (leader == 0){
            result_policies[a_idx] = leader_policy[a_idx];
        } else {
            result_policies[a_idx] = follower_policy[a_idx];
        }
    }
    int offset = num_available_actions[0];
    for (int a_idx = 0; a_idx < num_available_actions[1]; a_idx++){
        if (leader == 1){
            result_policies[a_idx + offset] = leader_policy[a_idx];
        } else {
            result_policies[a_idx + offset] = follower_policy[a_idx];
        }
    }
}
