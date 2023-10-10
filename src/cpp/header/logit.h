//
// Created by mahla on 09/03/2023.
//

#ifndef BATTLESNAKECPP_LOGIT_H
#define BATTLESNAKECPP_LOGIT_H

double compute_logit_cpp(
        int num_player_at_turn,
        const int* num_available_actions,  //shape (num_player_at_turn,)
        const int* available_actions,  // shape (sum(num_available_actions))
        const int* joint_actions, // shape (prod(num_available_actions) * num_player)
        const double* joint_action_values, // shape (prod(num_available_actions) * num_player)
        int num_iterations,
        double epsilon,
        const double* temperatures,
        bool initial_uniform,
        int mode,  // see above
        double hp_0,  //EMA alpha, ADAM lr, TODO: describe
        double hp_1,
        const double* initial_policies, // shape (sum(num_available_actions)), only used if not initial_uniform
        double* result_values,  // shape (num_players)
        double* result_policies // shape (sum(num_available_actions))
);

#endif //BATTLESNAKECPP_LOGIT_H
