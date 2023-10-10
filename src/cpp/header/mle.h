//
// Created by mahla on 14/05/2023.
//

#ifndef BATTLESNAKECPP_MLE_H
#define BATTLESNAKECPP_MLE_H

double compute_temperature_mle(
        double min_temp,
        double max_temp,
        int num_iter,  // number gradient descent iterations
        int t,  // number of time steps
        const int* chosen_actions,  //actions chosen at each time step
        const int* num_actions,  //number of actions available at each time step
        const double* utils  // flat utility array for every action in every time step
);

double compute_temperature_mle_line_search(
        double min_temp,
        double max_temp,
        int num_iter,  // number gradient descent iterations
        int t,  // number of time steps
        const int* chosen_actions,  //actions chosen at each time step
        const int* num_actions,  //number of actions available at each time step
        const double* utils  // flat utility array for every action in every time step
);


double temperature_likelihood(
        double tau,
        int t,  // number of time steps
        const int* chosen_actions,  //actions chosen at each time step
        const int* num_actions,  //number of actions available at each time step
        const double* utils  // flat utility array for every action in every time step
);


#endif //BATTLESNAKECPP_MLE_H
