//
// Created by mahla on 24/10/2022.
//

#include <iostream>
#include "header/battlesnake.h"
#include "header/battlesnake_helper.h"
#include "header/logit.h"
#include "header/nash.h"
#include "header/mle.h"
#include "header/quantal.h"
#include "header/overcooked.h"

//g++ -c -fPIC link.cpp -o link.o
//g++ -shared -Wl,-soname,-liblink.so -o liblink.so link.o

extern "C" {
    GameState* init_cpp(
        int w,
        int h,
        int num_snakes,
        int min_food,
        int food_spawn_chance,
        int init_turns_played,
        bool spawn_snakes_randomly,
        int* snake_body_lengths,
        int max_body_length,
        int* snake_bodies,
        int num_init_food,
        int* food_spawns,
        bool* snake_alive,
        int* snake_health,
        int* snake_len,
        int* max_health,
        bool wrapped,
        bool royale,
        int shrink_n_turns,
        int hazard_damage,
        bool* init_hazards
    ){
        return init(
            w,
            h,
            num_snakes,
            min_food,
            food_spawn_chance,
            init_turns_played,
            spawn_snakes_randomly,
            snake_body_lengths,
            max_body_length,
            snake_bodies,
            num_init_food,
            food_spawns,
            snake_alive,
            snake_health,
            snake_len,
            max_health,
            wrapped,
            royale,
            shrink_n_turns,
            hazard_damage,
            init_hazards
        );
    }
    void step_cpp(GameState* state, int* actions){
        step(state, actions);
    }
    void str_cpp(GameState* state, char* arr){
        draw_to_arr(state, arr);
    }
    void custom_encode_cpp(
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
            bool include_distance_map,
            bool include_area_control,
            bool include_food_distance,
            bool include_hazards,
            bool include_tail_distance,
            bool include_num_food_on_board,
            float fixed_food_spawn_chance,
            bool include_temperatures,
            bool single_temperature,
            const float* temperatures
    ){
        construct_custom_encoding(
                state,
                arr,
                include_current_food,
                include_next_food,
                include_board,
                include_number_of_turns,
                flatten_snakes,
                player_snake,
                include_snake_body_as_one_hot,
                include_snake_body,
                include_snake_head,
                include_snake_tail,
                include_snake_health,
                include_snake_length,
                centered,
                include_distance_map,
                include_area_control,
                include_food_distance,
                include_hazards,
                include_tail_distance,
                include_num_food_on_board,
                fixed_food_spawn_chance,
                include_temperatures,
                single_temperature,
                temperatures
            );
    }

    GameState* clone_cpp(GameState* state){
        return clone(state);
    }
    void close_cpp(GameState* state){
        close(state);
    }
    void actions_cpp(GameState* state, int snake_id, int* actions){
        legal_actions(state, snake_id, actions);
    }
    bool equals_cpp(GameState* state1, GameState* state2){
        return equals(state1, state2);
    }
    void alive_cpp(GameState* state, bool* arr){
        alive(state, arr);
    }
    void snake_length_cpp(GameState* state, int* arr){
        snake_length(state, arr);
    }
    int snake_body_length_cpp(GameState* state, int player){
        return snake_body_length(state, player);
    }
    void snake_pos_cpp(GameState* state, int player, int* arr){
        snake_pos(state, player, arr);
    }
    void snake_health_cpp(GameState* state, int* arr){
        snake_health(state, arr);
    }
    int num_food_cpp(GameState* state){
        return num_food(state);
    }
    void food_pos_cpp(GameState* state, int* arr){
        food_pos(state, arr);
    }
    int turns_played_cpp(GameState* state){
        return turns_played(state);
    }
    void area_control_cpp(
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
    ){
        area_control(
                state,
                area_arr,
                food_dist_arr,
                tail_dist_arr,
                reached_tail,
                reached_food,
                weight,
                food_weight,
                hazard_weight,
                food_in_hazard_weight
        );
    }
    void hazards_cpp(GameState* state, bool* arr){
        hazards(state, arr);
    }
    double logit_cpp(
            int num_player_at_turn,
            const int* num_available_actions,  //shape (num_player_at_turn,)
            const int* available_actions,  // shape (sum(num_available_actions))
            const int* joint_actions, // shape (prod(num_available_actions) * num_player)
            const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
            int num_iterations,
            double epsilon,
            const double* temperatures,
            bool initial_uniform,
            int mode,
            double hp_0,
            double hp_1,
            const double* initial_policies, // shape (sum(num_available_actions)), only used if not initial_uniform
            double* result_values,  // shape (num_players)
            double* result_policies // shape (sum(num_available_actions))
    ){
        return compute_logit_cpp(
            num_player_at_turn,
            num_available_actions,
            available_actions,
            joint_actions,
            joint_action_values,
            num_iterations,
            epsilon,
            temperatures,
            initial_uniform,
            mode,
            hp_0,
            hp_1,
            initial_policies,
            result_values,
            result_policies
        );
    }

    int compute_nash_cpp(
            int num_player_at_turn,
            const int* num_available_actions,  //shape (num_player_at_turn,)
            const int* available_actions,  // shape (sum(num_available_actions))
            const int* joint_actions, // shape (prod(num_available_actions) * num_player)
            const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
            double* result_values,  // shape (num_players)
            double* result_policies // shape (sum(num_available_actions))
    ){
        if (num_player_at_turn == 2){
            int result = compute_2p_nash(
                num_available_actions,
                available_actions,
                joint_actions,
                joint_action_values,
                result_values,
                result_policies
            );
            return result;
        } else {
            int result = compute_nash(
                num_player_at_turn,
                num_available_actions,
                available_actions,
                joint_actions,
                joint_action_values,
                result_values,
                result_policies
            );
            return result;
        }
    }

    void set_seed(int seed) {
        set_seed_gym(seed);
        set_seed_utils(seed);
    }

    double mle_temperature_cpp(
        double min_temp,
        double max_temp,
        int num_iter,  // number gradient descent iterations
        int t,  // number of time steps
        const int* chosen_actions,  //actions chosen at each time step
        const int* num_actions,  //number of actions available at each time step
        const double* utils,  // flat utility array for every action in every time step
        bool use_line_search
    ){
        double temperature;
        if (use_line_search){
            temperature = compute_temperature_mle_line_search(
                    min_temp,
                    max_temp,
                    num_iter,
                    t,
                    chosen_actions,
                    num_actions,
                    utils
            );
        } else {
            temperature = compute_temperature_mle(
                    min_temp,
                    max_temp,
                    num_iter,
                    t,
                    chosen_actions,
                    num_actions,
                    utils
            );
        }
        return temperature;
    }

    double temperature_likelihood_cpp(
            double tau,
            int t,  // number of time steps
            const int* chosen_actions,  //actions chosen at each time step
            const int* num_actions,  //number of actions available at each time step
            const double* utils  // flat utility array for every action in every time step
    ){
        double likelihood = temperature_likelihood(
                tau,
                t,
                chosen_actions,
                num_actions,
                utils
        );
        return likelihood;
    }

    void rm_qr_cpp(
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
        rm_qr(
                num_available_actions,
                available_actions,
                joint_actions,
                joint_action_values,
                leader,
                num_iterations,
                temperature,
                random_prob,
                result_values,
                result_policies
        );
    }

    void qse_cpp(
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
        qse(
                num_available_actions,
                available_actions,
                joint_actions,
                joint_action_values,
                leader,
                num_iterations,
                grid_size,
                temperature,
                result_values,
                result_policies
        );
    }

    void char_game_matrix_cpp(
            GameState* state,
            char* matrix
    ){
       char_game_matrix(state, matrix);
    }

    OvercookedGameState* init_overcooked_cpp(
            int w,
            int h,
            int* board,
            int* start_pos,
            int horizon,
            int cooking_time,
            double placement_in_pot_reward,
            double dish_pickup_reward,
            double soup_pickup_reward,
            double soup_delivery_reward,
            double soup_cooking_reward
    ){
        return init_overcooked(
                w,
                h,
                board,
                start_pos,
                horizon,
                cooking_time,
                placement_in_pot_reward,
                dish_pickup_reward,
                soup_pickup_reward,
                soup_delivery_reward,
                soup_cooking_reward
        );
    }

    OvercookedGameState* clone_overcooked_cpp(OvercookedGameState* state){
        return clone_overcooked(state);
    }

    void close_overcooked_cpp(OvercookedGameState* state){
        close_overcooked(state);
    }

    double step_overcooked_cpp(OvercookedGameState* state, int* actions){
        return step_overcooked(state, actions);
    }

    void char_overcooked_matrix_cpp(
            OvercookedGameState* state,
            char* matrix
    ){
        char_overcooked_matrix(state, matrix);
    }

    void construct_overcooked_encoding_cpp(OvercookedGameState* state, float* arr){
        construct_overcooked_encoding(state, arr);
    }

    bool equals_overcooked_cpp(OvercookedGameState* state, OvercookedGameState* other){
        return equals_overcooked(state, other);
    }
}

