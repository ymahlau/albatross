//
// Created by mahla on 09/03/2023.
//
// Computation of Nash Equilibrium according to paper:
// "Simple search methods for finding a Nash equilibrium"
// https://www.sciencedirect.com/science/article/pii/S0899825606000935

#include <vector>
#include <deque>
#include <algorithm>
#include "../header/utils.h"
#include "../alglib/optimization.h"

using namespace std;
using namespace alglib;

bool feasibility_2p(
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        unordered_map<pair<int, int>, pair<double, double>, hashed_int_pair>& action_value_map,
        const vector<int>& support1,
        const vector<int>& support2,
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
){
    //index helper
    int num_all_actions = num_available_actions[0] + num_available_actions[1];
    vector<int> support_indices1(support1.size());
    int idx_counter = 0;
    for (int a1: support1){
        for (int a_idx1 = 0; a_idx1 < num_available_actions[0]; a_idx1++){
            if (available_actions[a_idx1] == a1){
                support_indices1[idx_counter] = a_idx1;
                idx_counter += 1;
                break;
            }
        }
    }
    idx_counter = 0;
    vector<int> support_indices2(support2.size());
    for (int a2: support2){
        for (int a_idx2 = 0; a_idx2 < num_available_actions[1]; a_idx2++){
            if (available_actions[num_available_actions[0] + a_idx2] == a2){
                support_indices2[idx_counter] = a_idx2;
                idx_counter += 1;
                break;
            }
        }
    }
    //cost variables (all zero)
    // variables are first action probs of p1, then action probs of p2, the value of p1, value of p2
    int num_variables = num_all_actions + 2;  // policy probs and values
    real_1d_array c;
    c.setlength(num_variables);
    for (int i = 0; i < num_variables; i++){
        c[i] = 0;
    }
    //box constraints
    real_1d_array x_lower;
    real_1d_array x_upper;
    x_lower.setlength(num_variables);
    x_upper.setlength(num_variables);
    // box policy probability (between 0 and 1, always zero if not in support)
    for (int i = 0; i < num_all_actions; i++){
        x_upper[i] = 0;
        x_lower[i] = 0;
    }
    for (int a1_idx: support_indices1){
        x_upper[a1_idx] = 1;
    }
    for (int a2_idx: support_indices2){
        x_upper[num_available_actions[0] + a2_idx] = 1;
    }
    //box values (arbitrary)
    for (int i = num_all_actions; i < num_variables; i++){
        x_lower[i] = fp_neginf;
        x_upper[i] = fp_posinf;
    }
    //linear constraints
    // The player must be indifferent between all actions in the support and
    // prefer them over actions outside support. First and second constraint of Feasibility Program 1.
    int num_lin_constraints = num_all_actions + 2;
    real_2d_array a;
    real_1d_array a_lower;
    real_1d_array a_upper;
    a.setlength(num_lin_constraints, num_variables);
    a_lower.setlength(num_lin_constraints);
    a_upper.setlength(num_lin_constraints);
    for (int c_idx = 0; c_idx < num_lin_constraints; c_idx++){
        for (int v_idx = 0; v_idx < num_variables; v_idx++){
            a[c_idx][v_idx] = 0;
        }
    }
    // player 0
    int constraint_idx = 0;
    for (int a_idx = 0; a_idx < num_available_actions[0]; a_idx++){
        int a1 = available_actions[a_idx];
        // coefficient for value is -1
        a[constraint_idx][num_all_actions] = -1;
        // coefficients for enemy move probs is the utility of player 1
        for (int i = 0; i < (int) support2.size(); i++){
            int a2_idx = support_indices2[i];
            int a2 = support2[i];
            a[constraint_idx][num_available_actions[0] + a2_idx] = action_value_map[{a1, a2}].first;
        }
        a_upper[constraint_idx] = 0;
        //check if a2 in support, if yes equality constraint.
        bool in_support = false;
        for (int a1_supp: support1){
            if(a1_supp == a1){
                in_support = true;
                break;
            }
        }
        if (in_support){
            a_lower[constraint_idx] = 0;
        } else {
            a_lower[constraint_idx] = fp_neginf;
        }
        constraint_idx += 1;
    }
    // player 1
    for (int a_idx = 0; a_idx < num_available_actions[1]; a_idx++){
        int a2 = available_actions[num_available_actions[0] + a_idx];
        // coefficient for value is -1
        a[constraint_idx][num_all_actions + 1] = -1;
        // coefficients for enemy move probs is the utility of player 2
        for (int i = 0; i < (int) support1.size(); i++){
            int a1_idx = support_indices1[i];
            int a1 = support1[i];
            a[constraint_idx][a1_idx] = action_value_map[{a1, a2}].second;
        }
        a_upper[constraint_idx] = 0;
        //check if a2 in support, if yes equality constraint.
        bool in_support = false;
        for (int a2_supp: support2){
            if(a2_supp == a2){
                in_support = true;
                break;
            }
        }
        if (in_support){
            a_lower[constraint_idx] = 0;
        } else {
            a_lower[constraint_idx] = fp_neginf;
        }
        constraint_idx += 1;
    }
    //action probs need to sum to one within support
    for (int idx_1: support_indices1){
        a[constraint_idx][idx_1] = 1;
    }
    a_lower[constraint_idx] = 1;
    a_upper[constraint_idx] = 1;
    constraint_idx += 1;
    for (int idx_2: support_indices2){
        a[constraint_idx][num_available_actions[0] + idx_2] = 1;
    }
    a_lower[constraint_idx] = 1;
    a_upper[constraint_idx] = 1;
    //scale. 1 is a reasonable proxy for all variables
    real_1d_array s;
    s.setlength(num_variables);
    for (int i = 0; i < num_variables; i++){
        s[i] = 1;
    }
    // create linear program
    minlpstate state;
    minlpcreate(num_variables, state);
    // set constraint arrays
    minlpsetcost(state, c);
    minlpsetbc(state, x_lower, x_upper);
    minlpsetlc2dense(state, a, a_lower, a_upper, num_lin_constraints);
    minlpsetscale(state, s);
    // Solve
    minlpoptimize(state);
    // parse results
    minlpreport rep;
    real_1d_array x;
    minlpresults(state, x, rep);
    if (rep.terminationtype < 1 or rep.terminationtype > 4){
        return false;
    }
    for (int i = 0; i < num_all_actions; i++){
        result_policies[i] = x[i];
    }
    result_values[0] = x[num_all_actions];
    result_values[1] = x[num_all_actions + 1];
    return true;
}


// Check if an action is conditionally dominated given the action profiles of both players
bool is_cond_dominated_2p(
      int player,
      int action,
      const vector<int>& action_choices,  // possible actions for player
      const vector<int>& action_responses,  // possible actions for enemy (not player)
      unordered_map<pair<int, int>, pair<double, double>, hashed_int_pair>& action_value_map
){
    for (int a1: action_choices){
        if (a1 == action) continue;
        bool strictly_better = true;
        for (int a2: action_responses){
            if (player == 1){
                pair<double, double> v_actions = action_value_map[{action, a2}];
                pair<double, double> v_check = action_value_map[{a1, a2}];
                if (v_check.first <= v_actions.first){
                    strictly_better = false;
                    break;
                }
            } else {
                pair<double, double> v_actions = action_value_map[{a2, action}];
                pair<double, double> v_check = action_value_map[{a2, a1}];
                if (v_check.second <= v_actions.second) {
                    strictly_better = false;
                    break;
                }
            }
        }
        if (strictly_better) return true;
    }
    return false;
}

int compute_2p_nash(
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        const int* joint_actions, // shape (prod(num_available_actions) * num_player)
        const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
){
    //parse inputs for easier work with it later
    // input array shapes
    int num_joint_actions = 1;
    int num_all_actions = 0;
    for (int p = 0; p < 2; p++){
        num_joint_actions *= num_available_actions[p];
        num_all_actions += num_available_actions[p];
    }
    // joint action values
    unordered_map<pair<int, int>, pair<double, double>, hashed_int_pair> action_value_map;
    for (int ja_idx = 0; ja_idx < num_joint_actions; ja_idx++){
        int a1 = joint_actions[ja_idx * 2];
        int a2 = joint_actions[ja_idx * 2 + 1];
        double v1 = joint_action_values[ja_idx * 2];
        double v2 = joint_action_values[ja_idx * 2 + 1];
        action_value_map[{a1, a2}] = {v1, v2};
    }
    // available actions
    vector<int> actions1(num_available_actions[0]);
    vector<int> support_sizes_1(num_available_actions[0]);
    for (int i = 0; i < num_available_actions[0]; i++){
        support_sizes_1[i] = i+1;
        actions1[i] = available_actions[i];
    }
    vector<int> support_sizes_2(num_available_actions[1]);
    vector<int> actions2(num_available_actions[1]);
    for (int i = 0; i < num_available_actions[1]; i++){
        support_sizes_2[i] = i+1;
        actions2[i] = available_actions[num_available_actions[0] + i];
    }
    //joint support sizes
    vector<pair<int, int>> joint_support_sizes = cartesian_product(support_sizes_1, support_sizes_2);
    // The order of the support sizes is important for efficiency as stated in the paper.
    struct support_preference pref;
    sort(joint_support_sizes.begin(), joint_support_sizes.end(), pref);

    // ############### Start of Algorithm ###############################################
    // iterate all combinations of support sizes
    for (pair<int, int> cur_support_sizes: joint_support_sizes){
        // iterate support of player 1
        deque<vector<int>> all_supports_1 = combinations(actions1, cur_support_sizes.first);
        for (const vector<int>& cur_support1: all_supports_1){
            //prune search space for efficiency
            vector<int> actions2_pruned;
            for (int a2: actions2){
                if (not is_cond_dominated_2p(2, a2, actions2, cur_support1, action_value_map)){
                    actions2_pruned.push_back(a2);
                }
            }
            bool skip = false;
            for (int a1: cur_support1){
                if (is_cond_dominated_2p(1, a1, cur_support1, actions2_pruned, action_value_map)){
                    skip = true;
                    break;
                }
            }
            if (skip) continue;
            //iterate over supports of player 2
            deque<vector<int>> all_supports_2 = combinations(actions2_pruned, cur_support_sizes.second);
            for (const vector<int>& cur_support2: all_supports_2){
                //check if we can prune again
                skip = false;
                for (int a2: cur_support2){
                    if (is_cond_dominated_2p(2, a2, cur_support2, cur_support1, action_value_map)){
                        skip = true;
                        break;
                    }
                }
                if (skip) continue;
                // solve linear feasibility program and return solution if successful
                bool success = feasibility_2p(
                    num_available_actions,  //shape (num_player_at_turn,)
                    available_actions,  // shape (sum(num_available_actions))
                    action_value_map,
                    cur_support1,
                    cur_support2,
                    result_values,  // shape (num_players)
                    result_policies // shape (sum(num_available_actions))
                );
                if (success){
                    // if we found an equilibrium, we can immediately return.
                    return 0;
                }
            }
        }
    }
    return -1;
}

// Below is code for N - Player #################################################################################

struct nlc_info {
    int num_player;
    int num_variables;
    int num_all_actions;
    int num_eq_constraints;
    int num_non_eq_constraints;
    deque<deque<vector<int>>> enemy_ja;
    deque<deque<vector<double>>> ja_outcomes;
    unordered_map<pair<int, int>, int, hashed_int_pair> con_idx;
    unordered_map<pair<int, int>, int, hashed_int_pair> var_idx;
    vector<vector<int>> aa;  // available actions for each player
};

// This callback computes function value and jacobian-matrix of non-linear programming constraints
void nlc_func_jac (const real_1d_array &x, real_1d_array &fi, real_2d_array &jac, void *ptr){
    auto* info_p = (nlc_info*) ptr;
    // initialize jacobian and cost with zero
    for (int i = 0; i <= info_p->num_eq_constraints + info_p->num_non_eq_constraints; i++){
        fi[i] = 0;
        for (int v = 0; v < info_p->num_variables; v++){
            jac[i][v] = 0;
        }
    }
    // iterate joint enemy actions for all players
    for (int p = 0; p < info_p->num_player; p++){
        auto ja_it = info_p->enemy_ja[p].begin();
        auto out_it = info_p->ja_outcomes[p].begin();
        int num_ja = (int) info_p->enemy_ja[p].size();
        for (int ja_idx = 0; ja_idx < num_ja; ja_idx ++){
            vector<int> cur_ja = *ja_it;
            vector<double> cur_outcomes = *out_it;
            // compute joint action prob of enemies
            double prod = 1.0;
            int enemy_idx = 0;
            for (int enemy_a: cur_ja){
                if (enemy_idx == p) enemy_idx += 1;
                prod *= x[info_p->var_idx[{enemy_idx, enemy_a}]];
                enemy_idx += 1;
            }
            //update function values
            int outcome_idx = 0;
            for (int a: info_p->aa[p]){
                int cur_con_idx = info_p->con_idx[{p, a}];
                double cur_summand = prod * cur_outcomes[outcome_idx];
                fi[cur_con_idx] += cur_summand;
                //update jacobian for policy variables
                enemy_idx = 0;
                for (int enemy_a: cur_ja){
                    if (enemy_idx == p) enemy_idx += 1;
                    int cur_enemy_var_idx = info_p->var_idx[{enemy_idx, enemy_a}];
                    // we have to divide by probability of current variable to get gradient
                    double divisor = x[cur_enemy_var_idx];
                    if (divisor != 0 and cur_summand != 0) {
                        // if divisor is zero, then prod has to be zero
                        jac[cur_con_idx][cur_enemy_var_idx] += cur_summand / divisor;
                    }
                    enemy_idx += 1;
                }
                outcome_idx += 1;
            }
            //increase iterators
            ja_it++;
            out_it++;
        }
        // subtract value-variables from functions (originally right hand side of constraint)
        for (int a: info_p->aa[p]){
            int cur_con_idx = info_p->con_idx[{p, a}];
            fi[cur_con_idx] -= x[info_p->num_all_actions + p];
        }
        // gradient of value-variable is always -1
        for (int a: info_p->aa[p]) {
            int cur_con_idx = info_p->con_idx[{p, a}];
            jac[cur_con_idx][info_p->num_all_actions + p] = -1;
        }
    }
//    int raa = 1;
}

bool feasibility(
        int num_player,
        const deque<vector<int>>& supports,
        unordered_map<vector<int>, vector<double>, hashed_int_vector>& value_map,  // joint actions -> value
        const vector<vector<int>>& aa,  // available actions for each player
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
){
    // compute number of variables and constraints
    int num_all_actions = 0;
    for (const vector<int>& pv: aa){
        num_all_actions += (int) pv.size();
    }
    int num_all_support = 0;
    for (const vector<int>& s: supports){
        num_all_support += (int) s.size();
    }
    int num_variables = num_all_actions + num_player;  // policies and values
    int num_eq_constraints = num_all_support;
    int num_non_eq_constraints = num_all_actions - num_all_support;
    // compute joint enemy actions for every player
    deque<deque<vector<int>>> enemy_ja;
    deque<deque<vector<double>>> ja_outcomes;  // outcome of every action for current player given enemy joint action
    for (int p = 0; p < num_player; p++){
        //individual enemy actions
        deque<vector<int>> enemy_supports;
        int player_counter = 0;
        for(const vector<int>& s: supports){
            if (player_counter == p) {
                player_counter += 1;
                continue;
            }
            enemy_supports.push_back(s);
            player_counter += 1;
        }
        //joint enemy actions
        deque<vector<int>> ja_enemy_actions = general_cartesian_product(enemy_supports);
        enemy_ja.push_back(ja_enemy_actions);
        // values
        deque<vector<double>> player_values;
        int player_action_size = (int) aa[p].size();
        for (const vector<int>& ja: ja_enemy_actions){
            vector<double> cur_values(player_action_size);
            int counter = 0;
            for (int a: aa[p]){
                // construct full joint action
                vector<int> full_ja;
                full_ja = ja;
                full_ja.insert(full_ja.begin() + p, a);
                cur_values[counter] = value_map[full_ja][p];
                counter += 1;
            }
            player_values.push_back(cur_values);
        }
        ja_outcomes.push_back(player_values);
    }
    //create a hashmap player-action -> constraint index
    unordered_map<pair<int, int>, int, hashed_int_pair> con_idx;
    int constraint_counter = 1;  // first constraint is cost function, which we ignore
    for(int p = 0; p < num_player; p++){  // first equality constraints for actions in support
        for(int a: supports[p]){
            con_idx[{p, a}] = constraint_counter;
            constraint_counter += 1;
        }
    }
    for(int p = 0; p < num_player; p++){  // then inequality constraints for actions not in support
        for (int a: aa[p]){
            //check if in support and already included
            if (con_idx.find({p, a}) == con_idx.end()) {
                con_idx[{p, a}] = constraint_counter;
                constraint_counter += 1;
            }
        }
    }
    //create a hashmap for player-action -> variable index
    unordered_map<pair<int, int>, int, hashed_int_pair> var_idx;
    int variable_counter = 0;
    for(int p = 0; p < num_player; p++){
        for(int a: aa[p]){
            var_idx[{p, a}] = variable_counter;
            variable_counter += 1;
        }
    }
    //define struct for function value and jacobian computation
    struct nlc_info info = {
        num_player,
        num_variables,
        num_all_actions,
        num_eq_constraints,
        num_non_eq_constraints,
        enemy_ja,
        ja_outcomes,
        con_idx,
        var_idx,
        aa
    };
    //define box constraints
    real_1d_array lower_bounds;
    real_1d_array upper_bounds;
    lower_bounds.setlength(num_variables);
    upper_bounds.setlength(num_variables);
    variable_counter = 0;
    for (int p = 0; p < num_player; p++){  // bounds for policy
        vector<int> ps = supports[p];
        for (int a: aa[p]){
            lower_bounds[variable_counter] = 0;
            if (find(ps.begin(), ps.end(), a) != ps.end()){
                // in support
                upper_bounds[variable_counter] = 1;
            } else {
                // outside support
                upper_bounds[variable_counter] = 0;
            }
            variable_counter += 1;
        }
    }
    for (int p = 0; p < num_player; p++){  // bounds for value do not exist
        lower_bounds[num_all_actions + p] = fp_neginf;
        upper_bounds[num_all_actions + p] = fp_posinf;
    }
    // define linear constraint array (prob distribution needs to sum to one)
    int num_linear_constraints = num_player;
    real_2d_array lin_con;
    integer_1d_array lin_con_types;
    lin_con.setlength(num_linear_constraints, num_variables+1);
    for (int cons = 0; cons < num_linear_constraints; cons++){  // init all constraints with zeros first
        for (int var = 0; var < num_variables + 1; var++){
            lin_con[cons][var] = 0;
        }
    }
    lin_con_types.setlength(num_linear_constraints);
    variable_counter = 0;
    for (int p = 0; p < num_player; p++){
        // left hand side, sum of probabilities
        for (int a_idx = 0; a_idx < (int) aa[p].size(); a_idx++){
            lin_con[p][variable_counter] = 1.0;
            variable_counter += 1;
        }
        //right hand side of equation
        lin_con[p][num_variables] = 1.0;
        lin_con_types[p] = 0;  // zero signals equality constraint
    }
    //scale
    real_1d_array scale;
    scale.setlength(num_variables);
    for (int i = 0; i < num_variables; i++){
        scale[i] = 1.0;
    }
    // starting point, which is infeasible, might be a problem for some solver (edit: it is not for slp /sqp)
    real_1d_array x0;
    x0.setlength(num_variables);
    for (int i = 0; i < num_variables; i++){
        x0[i] = 0.0;
    }
    // create non-linear program
    minnlcstate state;
    minnlccreate(num_variables, x0, state);
    minnlcsetalgoslp(state);  // SLP is most robust, but also slow
//    minnlcsetalgosqp(state);  // SQP does not seem to yield speed improvement for this problem
//    double rho = 1000.0;
//    ae_int_t num_outer_iterations = 10;
//    minnlcsetalgoaul(state, rho, num_outer_iterations);
    minnlcsetscale(state, scale);
    //specify stopping criterion
    double eps_x = 0.000001;
    ae_int_t max_its = 0;  // zero means unlimited
    minnlcsetcond(state, eps_x, max_its);  // if both params zero, automatic stopping criterion (very bad)
    //add constraints
    minnlcsetbc(state, lower_bounds, upper_bounds);
    minnlcsetlc(state, lin_con, lin_con_types, num_linear_constraints);
    minnlcsetnlc(state, num_eq_constraints, num_non_eq_constraints); // constraint values are computed in every iter
    // activate sanity checks
//    minnlcoptguardsmoothness(state);
//    minnlcoptguardgradient(state, 1e-3);
    //optimize
    minnlcoptimize(state, nlc_func_jac, nullptr, &info);
    //fetch results
    minnlcreport rep;
    real_1d_array results;
    minnlcresults(state, results, rep);
    //fetch results of opt guard
//    optguardreport og_rep;
//    minnlcoptguardresults(state, og_rep);
    if (rep.terminationtype != 2) return false;
    //check if all constraints are satisfied
    double max_lc_err = 0.00075;
    double max_nlc_err = 0.02;
    if (rep.bcerr > max_lc_err or rep.lcerr > max_lc_err or rep.nlcerr > max_nlc_err) return false;
    //update policy and value
    variable_counter = 0;
    for (int i = 0; i < num_all_actions; i++){  // policy
        result_policies[variable_counter] = results[variable_counter];
        variable_counter += 1;
    }
    for (int p = 0; p < num_player; p++){
        result_values[p] = results[num_all_actions + p];
    }
    return true;
}


//Conditional Dominance of action by other_action
bool is_cond_dominated(
        int player,
        int action,
        int other_action,
        int num_player,
        const deque<deque<vector<int>>>& domains,
        unordered_map<vector<int>, vector<double>, hashed_int_vector>& value_map
){
    // compute action union of all other players
    deque<vector<int>> enemy_responses;
    for (int p = 0; p < num_player; p++){
        if (p == player) continue;
        vector<int> cur_union = element_union(domains[p]);
        enemy_responses.push_back(cur_union);
    }
    // cartesian product of enemy actions
    deque<vector<int>> joint_enemy_actions = general_cartesian_product(enemy_responses);
    for (vector<int> cur_ja: joint_enemy_actions){
        // add player actions to joint actions
        cur_ja.insert(cur_ja.begin() + player, action);
        double action_value = value_map[cur_ja][player];
        // replace action with other action
        cur_ja[player] = other_action;
        double other_action_value = value_map[cur_ja][player];
        if (other_action_value <= action_value) return false;
    }
    return true;
}

//Iterated Removal of Strictly Dominated Strategies
bool irsds(
        int num_player,
        deque<deque<vector<int>>>& domains,
        unordered_map<vector<int>, vector<double>, hashed_int_vector>& value_map,
        const vector<vector<int>>& aa
){
    bool changed;
    do {
        changed = false;
        //iterate player
        for (int p = 0; p < num_player; p++) {
            //compute union of actions in domain of this player
            vector<int> action_union = element_union(domains[p]);
            //iterate over union of actions
            for (int player_action: action_union) {
                // iterate over all other actions of this player
                for (int other_action: aa[p]) {
                    if (other_action == player_action) continue;
                    //check conditional dominance
                    if (is_cond_dominated(p, player_action, other_action, num_player, domains, value_map)) {
                        //remove all supports from player domain that contain the dominated action
                        deque<vector<int>> new_player_domain;
                        for(const vector<int>& cur_domain: domains[p]){
                            bool to_insert = true;
                            for (int a: cur_domain){
                                if (a == player_action){
                                    to_insert = false;
                                    break;
                                }
                            }
                            if (to_insert) new_player_domain.push_back(cur_domain);
                        }
                        changed = true;
                        // if new domain is empty, then the domain was inconsistent. return failure
                        if (new_player_domain.empty()) return false;
                        domains[p] = new_player_domain;
                        break;
                    }
                }
            }
        }
    } while (changed);
    return true;
}

bool recursive_backtrack(
        int num_player,
        deque<vector<int>> supports,
        deque<deque<vector<int>>> domains,
        int player_index,
        unordered_map<vector<int>, vector<double>, hashed_int_vector>& value_map,
        const vector<vector<int>>& aa,
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
){

    if (player_index == num_player){
        //check feasibility
        bool success = feasibility(num_player, supports, value_map, aa, result_values, result_policies);
        return success;
    }
    deque<vector<int>> player_domain = domains[player_index];
    int num_domains = (int) player_domain.size();
    // iterate all domains for current player
    for (int i = 0; i < num_domains; i++){
        supports[player_index] = player_domain.front();
        player_domain.pop_front();
        // construct new domain by fixing support of players up to current player
        deque<deque<vector<int>>> new_domains;
        for(int p = 0; p <= player_index; p++){  // players up to (including) current player
            deque<vector<int>> d = {supports[p]};
            new_domains.push_back(d);
        }
        for (int p = player_index + 1; p < num_player; p++){  // players after current player
            new_domains.push_back(domains[p]);
        }
        //check if strategies can be removed and profile is consistent
        bool consistent = irsds(num_player, new_domains, value_map, aa);
        if (consistent){
            // recurse one step deeper
            bool success = recursive_backtrack(
                    num_player,
                    supports,
                    new_domains,
                    player_index + 1,
                    value_map,
                    aa,
                    result_values,
                    result_policies
            );
            if (success) return true;
        }
    }
    return false;
}


int compute_nash(
        int num_player,
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        const int* joint_actions, // shape (prod(num_available_actions) * num_player)
        const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
){
    // input array shapes
    int num_joint_actions = 1;
    int num_all_actions = 0;
    for (int p = 0; p < num_player; p++){
        num_joint_actions *= num_available_actions[p];
        num_all_actions += num_available_actions[p];
    }
    vector<vector<int>> aa;  // index-able array of available actions
    int counter = 0;
    for (int p = 0; p < num_player; p++){
        vector<int> cur_avail_actions(num_available_actions[p]);
        for (int i = 0; i < num_available_actions[p]; i++){
            cur_avail_actions[i] = available_actions[counter];
            counter += 1;
        }
        aa.push_back(cur_avail_actions);
    }
    //hashmap for action tuple mapping to value outcomes of players
    unordered_map<vector<int>, vector<double>, hashed_int_vector> value_map;
    for (int ja_idx = 0; ja_idx < num_joint_actions; ja_idx++){
        vector<int> actions(num_player);
        vector<double> values(num_player);
        for (int p = 0; p < num_player; p++){
            actions[p] = joint_actions[ja_idx * num_player + p];
            values[p] = joint_action_values[ja_idx * num_player + p];
        }
        value_map[actions] = values;
    }
    // ################### Start of Algorithm ############################################
    //generate all support sizes
    deque<vector<int>> individual_support_sizes;
    for (int p = 0; p < num_player; p++){
        vector<int> cur_support_size(num_available_actions[p]);
        for (int i = 0; i < num_available_actions[p]; i++){
            cur_support_size[i] = i+1;
        }
        individual_support_sizes.push_back(cur_support_size);
    }
    //joint support sizes and sort by preference
    deque<vector<int>> joint_support_sizes = general_cartesian_product(individual_support_sizes);
    struct general_support_preference pref;
    sort(joint_support_sizes.begin(), joint_support_sizes.end(), pref);
    // iterate all joint support sizes
    for (vector<int> cur_joint_support_sizes: joint_support_sizes){
        //initialize supports and player domains
        deque<vector<int>> supports(num_player);
        deque<deque<vector<int>>> domains;
        for (int p = 0; p < num_player; p++){
            // domain consists of all support combinations of currently specified size
            deque<vector<int>> player_domain = combinations(aa[p], cur_joint_support_sizes[p]);
            domains.push_back(player_domain);
        }
        //recursive backtrack
        bool success = recursive_backtrack(num_player, supports, domains, 0, value_map, aa, result_values,
                                           result_policies);
        if (success) return 0;
    }
    return -1;
}
