//
// Created by mahla on 17/06/2023.
//

#ifndef BATTLESNAKECPP_QUANTAL_H
#define BATTLESNAKECPP_QUANTAL_H


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
);


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
);

#endif //BATTLESNAKECPP_QUANTAL_H
