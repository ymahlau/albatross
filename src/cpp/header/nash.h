//
// Created by mahla on 09/03/2023.
//

#ifndef BATTLESNAKECPP_NASH_H
#define BATTLESNAKECPP_NASH_H

int compute_2p_nash(
    const int* num_available_actions,  //shape (num_player_at_turn,)
    const int* available_actions,  // shape (sum(num_available_actions))
    const int* joint_actions, // shape (prod(num_available_actions) * num_player)
    const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
    double* result_values,  // shape (num_players)
    double* result_policies // shape (sum(num_available_actions))
);

int compute_nash(
        int num_player,
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        const int* joint_actions, // shape (prod(num_available_actions) * num_player)
        const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
);

#endif //BATTLESNAKECPP_NASH_H
