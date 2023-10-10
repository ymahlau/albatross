//
// Created by mahla on 14/05/2023.
//

#include <cmath>
#include <iostream>

using namespace std;

double tau_grad(double tau, int t, const int* chosen_actions, const int* num_actions, const double* utils){
    double result = 0;
    int start_idx = 0;  // start index at every time step in utils array
    for (int t_i = 0; t_i < t; t_i++){
        //fetch utils, actions, ...
        int a = chosen_actions[t_i];
        int num_a = num_actions[t_i];
        double chosen_util = utils[start_idx + a];
        //compute gradient at each time step
        double dividend = 0;
        double divisor = 0;
        for (int a_i = 0; a_i < num_a; a_i++){
            double u = utils[start_idx + a_i];
            double exponent = exp(u * tau);
            dividend += u * exponent;
            divisor += exponent;
        }
        // add to result
        result += chosen_util - dividend / divisor;
        // increment index
        start_idx += num_a;
    }
    return result;
}


double compute_temperature_mle(
        double min_temp,
        double max_temp,
        int num_iter,  // number gradient descent iterations
        int t,  // number of time steps
        const int* chosen_actions,  //actions chosen at each time step
        const int* num_actions,  //number of actions available at each time step
        const double* utils  // flat utility array for every action in every time step
){
    double tau = (max_temp + min_temp) / 2.0;  // initial temperature
    double lr = 0.01;
    double beta1 = 0.9;
    double beta2 = 0.99;
    double m = 0;
    double v = 0;
    double eps = 1e-6;
    double beta1_pow_t = beta1;
    double beta2_pow_t = beta2;
    for (int i = 0; i < num_iter; i++){
        // compute step
        double grad = tau_grad(tau, t, chosen_actions, num_actions, utils);
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;
        double m_hat = m / (1.0 - beta1_pow_t);
        double v_hat = v / (1.0 - beta2_pow_t);
        double step = lr * m_hat / (sqrt(v_hat) + eps);
        // update tau
        tau += step;
        if (tau > max_temp) tau = max_temp;
        if (tau < min_temp) tau = min_temp;
        beta1_pow_t *= beta1;
        beta2_pow_t *= beta2;
    }
    return tau;
}


double compute_temperature_mle_line_search(
        double min_temp,
        double max_temp,
        int num_iter,  // number gradient descent iterations
        int t,  // number of time steps
        const int* chosen_actions,  //actions chosen at each time step
        const int* num_actions,  //number of actions available at each time step
        const double* utils  // flat utility array for every action in every time step
){
    double low_bound = min_temp;
    double upp_bound = max_temp;
    double result = (max_temp + min_temp) / 2.0;
    for (int i = 0; i < num_iter; i++){
        // calculate gradient at current best estimate
        double grad = tau_grad(result, t, chosen_actions, num_actions, utils);
        // shrink boundaries
        if (grad > 0){
            low_bound = result;
        } else if (grad < 0){
            upp_bound = result;
        }
        // update result as the middle between new bounds
        result = (low_bound + upp_bound) / 2.0;
    }

    return result;
}


/// Computes Likelihood function. For debugging and visualization purposes
double temperature_likelihood(
        double tau,
        int t,  // number of time steps
        const int* chosen_actions,  //actions chosen at each time step
        const int* num_actions,  //number of actions available at each time step
        const double* utils  // flat utility array for every action in every time step
){
    double result = 0;
    int start_idx = 0;  // start index at every time step in utils array
    for (int t_i = 0; t_i < t; t_i++){
        //fetch utils, actions, ...
        int a = chosen_actions[t_i];
        int num_a = num_actions[t_i];
        double chosen_util = utils[start_idx + a];
        //compute gradient at each time step
        double summand = chosen_util * tau;
        double sum = 0;
        for (int a_i = 0; a_i < num_a; a_i++){
            double u = utils[start_idx + a_i];
            sum += exp(u * tau);
        }
        // add to result
        result += summand - log(sum);
        // increment index
        start_idx += num_a;
    }
    return result;
}
