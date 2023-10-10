//
// Created by mahla on 13/03/2023.
//

#ifndef BATTLESNAKECPP_UTILS_H
#define BATTLESNAKECPP_UTILS_H

#include <unordered_map>
#include <vector>
#include <list>
#include <cmath>
#include <deque>
#include <cstdint>

using namespace std;

//simple hashing function for integer pairs. Collision free if ints are less than 0xffffffff.
struct hashed_int_pair {
    template <class T1, class T2>
    size_t operator () (const pair<T1,T2> &p) const {
        unsigned long long hash_1 = hash<T1>{}(p.first);
        unsigned long long hash_2 = hash<T2>{}(p.second);
        unsigned long long mask_1 = 0x00000000ffffffffull;
        unsigned long long mask_2 = 0xffffffff00000000ull;
        unsigned long long masked_h1 = mask_1 & hash_1;
        unsigned long long masked_h2 = (hash_2 << 32) & mask_2;
        unsigned long long hash = masked_h1 | masked_h2;
        return hash;
    }
};

struct hashed_int_vector {
    size_t operator () (const vector<int>& vec) const {
        unsigned long long seed = 0;
        std::hash<int> hasher;
        for (int v: vec){
            seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
        return seed;
    }
};

// weight and normalize expected outcomes using a weighting function
vector<double> weight_and_normalize(
        int num_actions,
        const double *expected_outcomes,
        double temperature
);

//construct inverted index mapping player-action to joint-action-indices
unordered_map<pair<int, int>, list<int>, hashed_int_pair> construct_inverted_index(
        int num_player_at_turn,
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        const int* joint_actions, // shape (prod(num_available_actions) * num_player)
        int num_joint_actions  // == prod(num_available_actions)
);

// Sort in increasing order of |x1 - x2| and secondly x1 + x2
struct support_preference {
    bool operator()(pair<int, int> a, pair<int, int> b) const {
        int diff_1 = abs(a.first - a.second);
        int diff_2 = abs(b.first - b.second);
        int sum_1 = a.first + a.second;
        int sum_2 = b.first + b.second;
        if (diff_1 != diff_2){
            return diff_1 < diff_2;
        }
        return sum_1 < sum_2;
    }
};

// Sort in increasing order of size and secondly max-unbalance
struct general_support_preference {
    bool operator()(const vector<int>& a, const vector<int>& b) const {
        // sums and max_diff for a
        int sum_a = 0;
        int min_a = INT16_MAX;
        int max_a = 0;
        for (int i : a){
            sum_a += i;
            if (i < min_a) min_a = i;
            if (i > max_a) max_a = i;
        }
        int diff_a = max_a - min_a;
        // sums and max_diff for a
        int sum_b = 0;
        int min_b = INT16_MAX;
        int max_b = 0;
        for (int i : b){
            sum_b += i;
            if (i < min_b) min_b = i;
            if (i > max_b) max_b = i;
        }
        int diff_b = max_b - min_b;
        // preference result
        if (sum_a != sum_b) return sum_a < sum_b;
        return diff_a < diff_b;
    }
};

vector<pair<int, int>> cartesian_product(const vector<int>& vector1, const vector<int>& vector2);
deque<vector<int>> general_cartesian_product(const deque<vector<int>>& inputs);
deque<vector<int>> combinations(const vector<int>& elements, int size);
vector<int> element_union(const deque<vector<int>>& inputs);

#endif //BATTLESNAKECPP_UTILS_H
