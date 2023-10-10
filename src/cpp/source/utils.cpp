//
// Created by mahla on 13/03/2023.
//
#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include <list>
#include <algorithm>
#include <deque>
#include "../header/utils.h"

using namespace std;

vector<double> weight_and_normalize(
        int num_actions,
        const double *expected_outcomes,
        double temperature
){
    vector<double> new_policy(num_actions);
    //weight
    double sum = 0.0;
    for (int a_idx = 0; a_idx < num_actions; a_idx++){
        double prob = exp(temperature * expected_outcomes[a_idx]);
        new_policy[a_idx] = prob;
        sum += prob;
    }
    //normalize
    for (int a_idx = 0; a_idx < num_actions; a_idx++){
        new_policy[a_idx] /= sum;
    }
    return new_policy;
}

unordered_map<pair<int, int>, list<int>, hashed_int_pair> construct_inverted_index(
        int num_player_at_turn,
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        const int* joint_actions, // shape (prod(num_available_actions) * num_player)
        int num_joint_actions  // == prod(num_available_actions)
){
    // initialize inverted index mapping player action to indices in joint_actions
    unordered_map<pair<int, int>, list<int>, hashed_int_pair> inv_idx;
    int counter = 0;
    for (int p = 0; p < num_player_at_turn; p++){
        for (int a_idx = 0; a_idx < num_available_actions[p]; a_idx++){  //index of available action for player
            int action = available_actions[counter];
            inv_idx[{p, action}] = {};
            counter += 1;
        }
    }
    for (int ja_idx = 0; ja_idx < num_joint_actions; ja_idx++){  //joint action index in array
        for (int p = 0; p < num_player_at_turn; p++){
            int action = joint_actions[ja_idx * num_player_at_turn + p];
            inv_idx[{p, action}].push_back(ja_idx);
        }
    }
    return inv_idx;
}

vector<pair<int, int>> cartesian_product(const vector<int>& vector1, const vector<int>& vector2){
    vector<pair<int, int>> result(vector1.size() * vector2.size());
    int counter = 0;
    for (int v1: vector1){
        for (int v2: vector2){
            result[counter] = {v1, v2};
            counter += 1;
        }
    }
    return result;
}

deque<vector<int>> general_cartesian_product(const deque<vector<int>>& inputs){
    //parse input sizes and init helper variables
    int num_seqs = (int) inputs.size();
    int result_size = 1;
    vector<int> seq_sizes(num_seqs);
    int counter = 0;
    for (auto& cur_input : inputs){
        result_size *= (int) cur_input.size();
        seq_sizes[counter++] = (int) cur_input.size();
    }
    deque<vector<int>> result;
    vector<int> indices(num_seqs, 0);
    //iterate
    for(int i = 0; i < result_size; i++){
        // generate tuple given current indices
        vector<int> cur_tuple(num_seqs);
        int s_idx = 0;
        for (auto& cur_input : inputs){
            cur_tuple[s_idx] = cur_input[indices[s_idx]];
            s_idx += 1;
        }
        result.push_back(cur_tuple);
        // increment indices
        for (int s = 0; s < num_seqs; s++){
            indices[s] = (indices[s] + 1) % seq_sizes[s];
            if (indices[s] != 0) break;
        }
    }
    return result;
}

deque<vector<int>> combinations(const vector<int>& elements, int size){
    // implementation from rosetta-code: https://rosettacode.org/wiki/Combinations
    int n = (int) elements.size();
    vector<bool> bitmask(size, true); // K leading 1's
    bitmask.resize(n, false); // N-K trailing 0's
    deque<vector<int>> result = {};

    do {
        int counter = 0;
        vector<int> cur_vector(size);
        for(int i = 0; i < n; i++){
            if (bitmask[i]){
                cur_vector[counter] = elements[i];
                counter += 1;
            }
        }
        result.push_back(cur_vector);
    } while (prev_permutation(bitmask.begin(), bitmask.end()));
    return result;
}

vector<int> element_union(const deque<vector<int>>& inputs){
    vector<int> action_union;
    unordered_map<int, bool> is_in_union;
    for(const vector<int>& d: inputs){
        for (int a: d){
            if (is_in_union.find(a) == is_in_union.end()){
                is_in_union[a] = true;
                action_union.push_back(a);
            }
        }
    }
    return action_union;
}
