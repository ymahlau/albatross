//
// Created by mahla on 09/03/2023.
//


#include <unordered_map>
#include <vector>
#include <list>
#include "../header/utils.h"

using namespace std;

const int MODE_MSA = 0;
const int MODE_EMA = 1;
const int MODE_ADAM = 2;
const int MODE_BB = 3;
const int MODE_2_3 = 4;
const int MODE_REPETITIONS = 5;
const int MODE_BB_2_3 = 6;
const int MODE_BB_REPETITIONS = 7;
const int MODE_BB_MSA = 8;
const int MODE_SRA = 9;
const int MODE_SRA_REPETITIONS = 10;

const double MIN_PROB = 1e-5;  // for numerical stability
const double MAX_PROB = 1 - MIN_PROB;


double compute_logit_cpp(
        int num_player_at_turn,
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        const int* joint_actions, // shape (prod(num_available_actions) * num_player)
        const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
        int num_iterations,
        double epsilon,  // squared policy error for early stopping
        const double* temperatures,
        bool initial_uniform,
        int mode,  // see above
        double hp_0,  //EMA alpha, ADAM lr,
        double hp_1,
        const double* initial_policies, // shape (sum(num_available_actions)), only used if not initial_uniform
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
){
    //compute number of joint actions and sum of all player actions
    int num_joint_actions = 1;
    int num_all_actions = 0;
    int counter = 0;  // the counter indexes player-action tuple in arrays
    for (int p = 0; p < num_player_at_turn; p++){
        num_joint_actions *= num_available_actions[p];
        num_all_actions += num_available_actions[p];
    }
    //policies / auxiliary policies
    vector<double> joint_probs(num_joint_actions, 1.0);
    vector<double> expected_outcomes(num_all_actions, 0);
    vector<double> last_pol(num_all_actions, 0);
    vector<double> last_aux_pol(num_all_actions, 0);
    vector<double> cur_pol(num_all_actions, 0);
    vector<double> cur_aux_pol(num_all_actions, 0);
    // initialize result policy
    if (not initial_uniform) {
        for (int i = 0; i < num_all_actions; i++) {
            cur_pol[i] = initial_policies[i];
        }
    } else {  // initialize uniformly
        for (int p = 0; p < num_player_at_turn; p++) {
            auto prob = 1.0 / num_available_actions[p];
            for (int a_idx = 0; a_idx < num_available_actions[p]; a_idx++) {
                cur_pol[counter] = prob;
                counter += 1;
            }
        }
    }
    // initialize inverted index mapping player action to indices in joint_actions
    unordered_map<pair<int, int>, list<int>, hashed_int_pair> inv_idx = construct_inverted_index(
            num_player_at_turn,
            num_available_actions,  //shape (num_player_at_turn,)
            available_actions,  // shape (sum(num_available_actions))
            joint_actions, // shape (prod(num_available_actions) * num_player)
            num_joint_actions  // == prod(num_available_actions)
    );
    //Adam setup
    double beta1 = 0.9;
    double beta2 = 0.999;
    vector<double> m(num_all_actions, 0.0);
    vector<double> v(num_all_actions, 0.0);
    double eps = 1e-8;
    double beta1_pow_t = beta1;
    double beta2_pow_t = beta2;
    // repetition setup
    int cur_repetition = 1;
    // SRA setup
    double sra_beta = 1;
    //iterate episodes
    for (int episode = 0; episode < num_iterations + 1; episode++) {
        //compute joint action probabilities
        fill(joint_probs.begin(), joint_probs.end(), 1.0);
        counter = 0;
        for (int p = 0; p < num_player_at_turn; p++) {
            for (int a_idx = 0; a_idx < num_available_actions[p]; a_idx++) {
                int action = available_actions[counter];
                for (int ja_idx: inv_idx[{p, action}]) {
                    joint_probs[ja_idx] *= cur_pol[counter];
                }
                counter += 1;
            }
        }
        //update step
        counter = 0;
        double abs_pol_change = 0;
        for (int p = 0; p < num_player_at_turn; p++) {
            //calculate expected outcomes
            int cur_counter = counter;
            for (int a_idx = 0; a_idx < num_available_actions[p]; a_idx++) {
                int action = available_actions[cur_counter];
                double action_prob = cur_pol[cur_counter];
                double outcome = 0.0;
                for (int ja_idx: inv_idx[{p, action}]) {
                    double cur_ja_value = joint_action_values[ja_idx * num_player_at_turn + p];
                    outcome += cur_ja_value * joint_probs[ja_idx];
                }
                expected_outcomes[cur_counter] = outcome / action_prob;
                cur_counter += 1;
            }
            //weight and normalize to get new policy
            vector<double> new_policy = weight_and_normalize(
                    num_available_actions[p],
                    &expected_outcomes[counter],
                    temperatures[p]
            );
            cur_counter = counter;
            for (int a_idx = 0; a_idx < num_available_actions[p]; a_idx++) {
                cur_aux_pol[cur_counter] = new_policy[a_idx];
                double cur_change = cur_aux_pol[cur_counter] - cur_pol[cur_counter];  // update squared error
                abs_pol_change += abs(cur_change);
                cur_counter++;
            }
            //update counter to next players start index
            counter += num_available_actions[p];
        }
        // stopping criterion
        if (episode == num_iterations or abs_pol_change < epsilon){
            break;
        }
        //determine step size
        double step_size;
        if (mode == MODE_MSA) {  // method of successive averages
            step_size = 1.0 / (episode + 2);
        } else if (mode == MODE_EMA) {  // exponential moving average (const step size)
            step_size = hp_0;
        } else if (mode == MODE_ADAM) {  // Adaptive Moment Estimation (Adam)
            for(int idx = 0; idx < num_all_actions; idx++){
                // reset last policy
                last_pol[idx] = cur_pol[idx];
                last_aux_pol[idx] = cur_aux_pol[idx];
                // calc grad, momentum and velocity
                double step = cur_aux_pol[idx] - cur_pol[idx];
                m[idx] = beta1 * m[idx] + (1.0 - beta1) * step;
                v[idx] = beta2 * v[idx] + (1.0 - beta2) * step * step;
                double m_hat = m[idx] / (1.0 - beta1_pow_t);
                double v_hat = v[idx] / (1.0 - beta2_pow_t);
                // update
                double result_step = hp_0 * m_hat / (sqrt(v_hat) + eps);  // hp_0 is learning rate
                cur_pol[idx] += result_step;
            }
            // parameter update
            beta1_pow_t *= beta1;
            beta2_pow_t *= beta2;
        }  else if (mode == MODE_BB or mode == MODE_BB_2_3 or mode == MODE_BB_REPETITIONS or mode == MODE_BB_MSA) {
            // barilai-borwain step size
            if (episode == 0){
                step_size = 1;
            } else {
                double pol_vec_prod = 0;
                double grad_vec_prod = 0;
                for(int idx = 0; idx < num_all_actions; idx++){
                    double del_x_i = last_aux_pol[idx] - last_pol[idx];
//                    double del_x_i = cur_pol[idx] - last_pol[idx];
                    double del_g_i = del_x_i - (cur_aux_pol[idx] - last_aux_pol[idx]);
                    pol_vec_prod += del_x_i * del_g_i;
                    grad_vec_prod += del_g_i * del_g_i;
                }
                step_size = pol_vec_prod / sqrt(grad_vec_prod);
            }
            // constraints on step size: has to be less than one
            if (step_size > MAX_PROB) step_size = MAX_PROB;
            double min_step_size = 0;
            if (mode == MODE_BB_2_3) {
                // lower constraint: polyak step size
                min_step_size = 1.0 / pow(episode + 2, 2.0 / 3.0);
            } else if (mode == MODE_BB_REPETITIONS) {
                // lower constraint: repetition step size
                min_step_size = 1.0 / (double) cur_repetition;
                if (episode + 1.5 > 0.5 * (cur_repetition * cur_repetition + cur_repetition)){
                    cur_repetition += 1;
                }
            } else if (mode == MODE_BB_MSA) {
                // lower constraint: msa
                min_step_size = 1.0 / (episode + 2);
            }
            if (step_size < min_step_size){
                step_size = min_step_size;
            }
        } else if (mode == MODE_2_3) {
            // Polyak step size
            step_size = 1.0 / pow(episode + 2, 2.0 / 3.0);
        } else if (mode == MODE_REPETITIONS) { // 1, 1/2, 1/2, 1/3, 1/3, 1/3, ... n times 1/n
            step_size = 1.0 / (double) cur_repetition;
            if (episode + 1.5 > 0.5 * (cur_repetition * cur_repetition + cur_repetition)) {
                cur_repetition += 1;
            }
        } else if (mode == MODE_SRA) {
            // Self-regulating averaging. hp_0 is gamma and hp_1 is large GAMMA
            // the paper proposes gamma in [0.01, 0.5] and GAMMA in [1.5, 2]
            double cur_diff = 0;
            double last_diff = 0;
            for (int idx = 0; idx < num_all_actions; idx++) {
                cur_diff += (cur_pol[idx] - cur_aux_pol[idx]) * (cur_pol[idx] - cur_aux_pol[idx]);
                last_diff += (last_pol[idx] - last_aux_pol[idx]) * (last_pol[idx] - last_aux_pol[idx]);
            }
            if (cur_diff < last_diff) {
                sra_beta += hp_0;
            } else {
                sra_beta += hp_1;
            }
            step_size = 1.0 / sra_beta;
        } else if (mode == MODE_SRA_REPETITIONS){
            double cur_diff = 0;
            double last_diff = 0;
            for (int idx = 0; idx < num_all_actions; idx++) {
                cur_diff += (cur_pol[idx] - cur_aux_pol[idx]) * (cur_pol[idx] - cur_aux_pol[idx]);
                last_diff += (last_pol[idx] - last_aux_pol[idx]) * (last_pol[idx] - last_aux_pol[idx]);
            }
            if (cur_diff < last_diff) {
                sra_beta = cur_repetition;
            } else {
                sra_beta += hp_0;
            }
            if (episode + 1.5 > 0.5 * (cur_repetition * cur_repetition + cur_repetition)) {
                cur_repetition += 1;
            }
            step_size = 1.0 / sra_beta;
        } else {
            exit(1);
        }
        //update and move cur_pol / cur_aux_pol to last
        for(int idx = 0; idx < num_all_actions; idx++){
            // update
            if (mode != MODE_ADAM){
                last_pol[idx] = cur_pol[idx];
                last_aux_pol[idx] = cur_aux_pol[idx];
                cur_pol[idx] = cur_pol[idx] + step_size * (cur_aux_pol[idx] - cur_pol[idx]);
            }
            // clip policy to avoid numerical over/underflow
            if (cur_pol[idx] > MAX_PROB){
                cur_pol[idx] = MAX_PROB;
            } else if(cur_pol[idx] < MIN_PROB) {
                cur_pol[idx] = MIN_PROB;
            }
        }
        //normalize to correct clipping and avoid accumulating numerical errors
        counter = 0;
        for (int p = 0; p < num_player_at_turn; p++) {
            double prob_sum = 0;
            int cur_counter = counter;
            for (int a_idx = 0; a_idx < num_available_actions[p]; a_idx++){  // calc prob sum
                prob_sum += cur_pol[cur_counter];
                cur_counter += 1;
            }
            cur_counter = counter;
            for (int a_idx = 0; a_idx < num_available_actions[p]; a_idx++){  // normalize
                cur_pol[cur_counter] /= prob_sum;
                cur_counter += 1;
            }
            counter = cur_counter;
        }
    }
    //update result policies
    for(int idx = 0; idx < num_all_actions; idx++){
        result_policies[idx] = cur_pol[idx];
    }
    //update result values
    for (int ja_idx = 0; ja_idx < num_joint_actions; ja_idx++){
        for (int p = 0; p < num_player_at_turn; p++) {
            double cur_ja_value = joint_action_values[ja_idx * num_player_at_turn + p];
            result_values[p] += cur_ja_value * joint_probs[ja_idx];
        }
    }
    // calculate absolute policy error
    double diff = 0;
    for(int idx = 0; idx < num_all_actions; idx++){
        diff += abs(cur_pol[idx] - cur_aux_pol[idx]);
    }
    return diff;
}
