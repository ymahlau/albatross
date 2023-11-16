//
// Created by mahla on 11/3/2023.
//

#ifndef BATTLESNAKECPP_OVERCOOKED_H
#define BATTLESNAKECPP_OVERCOOKED_H


#include <vector>
#include <tuple>
#include <list>

using namespace std;

const int NO_ITEM = 0;
const int ONION_ITEM = 1;
const int DISH_ITEM = 2;
const int SOUP_ITEM = 3;

const int EMPTY_TILE = 0;
const int COUNTER_TILE = 1;
const int DISH_TILE = 2;
const int ONION_TILE = 3;
const int POT_TILE = 4;
const int SERVING_TILE = 5;

const int UP_ACTION = 0;
const int DOWN_ACTION = 1;
const int RIGHT_ACTION = 2;
const int LEFT_ACTION = 3;
const int STAY_ACTION = 4;
const int INTERACT_ACTION = 5;

const int EMPTY_POT = 0;
const int ONE_POT = 1;
const int TWO_POT = 2;
const int THREE_POT = 3;
const int DONE_POT = 4;
// cooking pot has state 4 + cooking_time_remaining if pot started cooking

using Coord = pair<int, int>;

struct Player{
    Coord position;
    int orientation;  // up, down, right, left like the actions
    int held_item;
};

struct OvercookedRewards{
    double placement_in_pot;
    double dish_pickup;
    double soup_pickup;
    double soup_delivery;
    double start_cooking;
};


struct OvercookedGameState{
    OvercookedGameState(vector<int> board, int turn, int w, int h, vector<Player> players, vector<int> tile_states,
                        int horizon, int cooking_time, OvercookedRewards reward_specs);
    OvercookedGameState(const OvercookedGameState& other);
    ~OvercookedGameState();
    OvercookedGameState& operator=(const OvercookedGameState& other);
    vector<int> board;
    int turn;
    int w;
    int h;
    vector<Player> players;
    vector<int> tile_states;  // same size as board with -1 where no pot/counter is located
    int horizon;
    int cooking_time;
    OvercookedRewards reward_specs;
    list<Coord> pot_positions;
    int dish_pickup_rewards_available;
    int dish_pickup_rewards_increment_next_round;
};

OvercookedGameState* init_overcooked(int w, int h, int* board, int* start_pos, int horizon, int cooking_time,
                                     double placement_in_pot_reward, double dish_pickup_reward,
                                     double soup_pickup_reward, double soup_delivery_reward, double soup_cooking);
OvercookedGameState* clone_overcooked(OvercookedGameState* state);
double step_overcooked(OvercookedGameState* state, int* actions);
void close_overcooked(OvercookedGameState* state);
void char_overcooked_matrix(OvercookedGameState* state, char* matrix);
void construct_overcooked_encoding(OvercookedGameState* state, float* arr, int player);
bool equals_overcooked(OvercookedGameState* state, OvercookedGameState* other);


#endif //BATTLESNAKECPP_OVERCOOKED_H

