//
// Created by mahla on 26/10/2022.
//

#ifndef BATTLESNAKECPP_BATTLESNAKE_HELPER_H
#define BATTLESNAKECPP_BATTLESNAKE_HELPER_H

#include <string>

#define UP 0
#define RIGHT 1
#define DOWN 2
#define LEFT 3

void set_seed_utils(int seed);

//initialization
deque<Coord> spawn_randomly(int w, int num_snakes);
deque<deque<Coord>> spawn_on_pos(const int* pos, const int* snake_body_lengths, int num_snakes, int max_body_length);
list<Coord> initialize_food(int w, int num_snakes, const vector<Coord>& snake_spawns);
list<Coord> food_on_pos(int* pos, int num_food);

//moving
Coord new_position(Coord old, int move, GameState* state);
bool in_bounds(GameState* state, Coord pos);
vector<int> possible_food_spawns(GameState* state);
void place_food_randomly(GameState* state);

//Debug
vector<int> human_repr(GameState* state);
void draw_to_arr(GameState* state, char* arr);

//State
void construct_custom_encoding(
    GameState* state,
    float* arr,
    bool include_current_food,
    bool include_next_food,
    bool include_board,
    bool include_number_of_turns,
    bool flatten_snakes,
    int player_snake,
    bool include_snake_body_as_one_hot,
    bool include_snake_body,
    bool include_snake_head,
    bool include_snake_tail,
    bool include_snake_health,
    bool include_snake_length,
    bool centered,
    bool include_dist_map,
    bool include_area_control,
    bool include_food_distance,
    bool include_hazards,
    bool include_tail_distance,
    bool include_num_food_on_board,
    float fixed_food_spawn_chance,
    bool include_temperatures,
    bool single_temperature,
    const float* temperatures
);
bool equals(GameState* state1, GameState* state2);

//State info
void alive(GameState* state, bool* arr);
void snake_length(GameState* state, int* arr);
int snake_body_length(GameState* state, int player);
void snake_pos(GameState* state, int player, int* arr);
int num_food(GameState* state);
void food_pos(GameState* state, int* arr);
int turns_played(GameState* state);
void snake_health(GameState* state, int* arr);
void area_control(
    GameState* state,
    float* area_arr,
    int* food_dist_arr,
    int* tail_dist_arr,
    bool* reached_tail,
    bool* reached_food,
    float weight,
    float food_weight,
    float hazard_weight,
    float food_in_hazard_weight
);
void hazards(GameState* state, bool* arr);
void char_game_matrix(GameState* state, char* matrix);

#endif //BATTLESNAKECPP_BATTLESNAKE_HELPER_H
