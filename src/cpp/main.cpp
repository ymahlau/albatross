#include <iostream>
#include "header/battlesnake.h"
#include "header/battlesnake_helper.h"
#include "header/logit.h"

#include "alglib/optimization.h"
#include "header/nash.h"
#include "header/utils.h"
#include "header/mle.h"
#include "header/quantal.h"
#include "header/overcooked.h"

using namespace std;
using namespace alglib;


int main() {
    int board[20] = {1, 1, 4, 1, 1, 3, 0, 0, 0, 3, 1, 0, 0, 0, 1, 1, 2, 1, 5, 1};
    int start_pos[6] = {3, 1, 0, 1, 2, 0};
    OvercookedGameState* state_p = init_overcooked(5, 4, board, start_pos, 400, 20, 3, 5, 3, 20, 0);
    int actions[2] = {2, 0};
    double reward = step_overcooked(state_p, actions);
    int actions2[2] = {5, 3};
    reward = step_overcooked(state_p, actions2);
    int a = 1;


//    int aa[] = {2, 3, 2, 3};
//    double utils[] = {
//        1.0, 0.5,
//        1.0, 0.5, 0,
//        1.0, 0,
//        1.0, 0, -1
//    };
//    int chosen[] = {0, 0, 1, 0};
//    int num_iter = 100;
//
//    double result = compute_temperature_mle(
//            0.1,
//            4,
//            num_iter,  // number gradient descent iterations
//            4,  // number of time steps
//            chosen,  //actions chosen at each time step
//            aa,  //number of actions available at each time step
//            utils  // flat utility array for every action in every time step
//    );
//    cout << result;

    /////////////////////////////////////////////////////////////7

//    int num_available_actions[] = {2, 2};
//    int available_actions[] = {0, 1, 0, 1};
//    int joint_actions[] = {
//            0, 0,
//            0, 1,
//            1, 0,
//            1, 1};
//    float joint_action_values[] = {
//            1, -1,
//            -1, 1,
//            -1, 1,
//            1, -1,
//    };
//    float result_values[] = {0, 0};
//    float result_policies[] = {0, 0, 0, 0};
//    int iterations = 10000;
//    float reg_factor = 0.2;
//    bool initial_uniform = true;
//    float initial_policies[] = {0.001, 0.999, 0.001, 0.999};
//    float reg_policy[] = {0.4, 0.6, 0.999, 0.001};
//    compute_r_nad_cpp_2p(
//            num_available_actions,
//            available_actions,
//            joint_actions,
//            joint_action_values,
//            reg_policy,
//            iterations,
//            reg_factor,
//            initial_uniform,
//            initial_policies,
//            result_values,
//            result_policies
//    );


    ///////////////////////////////////////////////////////////////////////////////////
//    int num_available_actions[] = {2, 2, 2};
//    int available_actions[] = {7, 3, 2, 1, 9, 4};
//    int joint_actions[] = {
//            7, 2, 9,
//            7, 2, 4,
//            7, 1, 9,
//            7, 1, 4,
//            3, 2, 9,
//            3, 2, 4,
//            3, 1, 9,
//            3, 1, 4};
//    float joint_action_values[] = {
//            7, 7, 7,
//            7, 7, 6,
//            1, 1, 2.3,
//            1, 2, 0,
//            6, 0, 0,
//            8, 5, 3,
//            6, 6.5, 1,
//            6, 5.5, 5
//    };
//    float result_values[] = {0, 0, 0};
//    float result_policies[] = {0, 0, 0, 0, 0, 0};
//    int num_available_actions[] = {3, 3, 3};
//    int available_actions[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
//    int joint_actions[] = {
//            0, 0, 0,
//            0, 0, 1,
//            0, 0, 2,
//            0, 1, 0,
//            0, 1, 1,
//            0, 1, 2,
//            0, 2, 0,
//            0, 2, 1,
//            0, 2, 2,
//
//            1, 0, 0,
//            1, 0, 1,
//            1, 0, 2,
//            1, 1, 0,
//            1, 1, 1,
//            1, 1, 2,
//            1, 2, 0,
//            1, 2, 1,
//            1, 2, 2,
//
//            2, 0, 0,
//            2, 0, 1,
//            2, 0, 2,
//            2, 1, 0,
//            2, 1, 1,
//            2, 1, 2,
//            2, 2, 0,
//            2, 2, 1,
//            2, 2, 2,
//    };
//    float joint_action_values[] = {
//            0, 0, 0,
//            -10, -10, 10,
//            1, 1, -1,
//            -10, 10, -10,
//            -1, 1, -1,
//            0, 0, 0,
//            1, -1, 1,
//            0, 0, 0,
//            10, -10, -10,
//
//            10, -10, -10,
//            1, -1, 1,
//            0, 0, 0,
//            1, 1, -1,
//            0, 0, 0,
//            -10, -10, 10,
//            0, 0, 0,
//            -10, 10, -10,
//            -1, 1, 1,
//
//            -1, 1, 1,
//            0, 0, 0,
//            -10, 10, -10,
//            0, 0, 0,
//            10, -10, -10,
//            1, -1, 1,
//            -10, -10, 10,
//            1, 1, -1,
//            0, 0, 0,
//    };
//    float result_values[] = {0, 0, 0};
//    float result_policies[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
//    compute_nash(3, num_available_actions, available_actions, joint_actions, joint_action_values, result_values,
//                 result_policies);


    ///////////////////////////////////////////////////////////////////////////////////////////

//    double a_t[] = {1, -1, 1, 1};
//    double al_t[] = {-1, fp_neginf};
//    double au_t[] = {fp_posinf, 1};
//    double c_t[] = {-0.1, -1};
//    vector<double> vec(2, 0);
//    double s_t[] = {1, 1};
//    double bndl_t[] = {-1, -1};
//    double bndu_t[] = {1, 1};
//
//    real_1d_array temparr;
//    temparr.setlength(2);
//    temparr[0] = -0.1;
//
//    real_2d_array a;
//    a.setcontent(2, 2, a_t);
//    real_1d_array al;
//    al.setcontent(2, al_t);
//    real_1d_array au;
//    au.setcontent(2, au_t);
//    real_1d_array c;
//    c.setcontent(2, c_t);
//    real_1d_array s;
//    s.setcontent(2, s_t);
//    real_1d_array bndl;
//    bndl.setcontent(2, bndl_t);
//    real_1d_array bndu;
//    bndu.setcontent(2, bndu_t);

//    real_2d_array a = "[[1,-1],[1,+1]]";
//    real_1d_array al = "[-1,-inf]";
//    real_1d_array au = "[+inf,+1]";
//    real_1d_array c = "[-0.1,-1]";
//    real_1d_array s = "[1,1]";
//    real_1d_array bndl = "[-1,-1]";
//    real_1d_array bndu = "[+1,+1]";
//    real_1d_array x;
//    minlpstate state;
//    minlpreport rep;
//    minlpcreate(2, state);
//
//
//    minlpsetcost(state, c);
//    minlpsetbc(state, bndl, bndu);
//    minlpsetlc2dense(state, a, al, au, 2);
//
//    minlpsetscale(state, s);
//
//    // Solve
//    minlpoptimize(state);
//    minlpresults(state, x, rep);
//    printf("%s\n", x.tostring(3).c_str()); // EXPECTED: [0,1]

//////////////////////////////////////////////////////////////////////////////////////////////////77

    int num_player = 2;
    int num_available_actions[] = {2, 2};
//    int num_available_actions[] = {2, 4};
    int available_actions[] = {0, 1, 0, 1};
//    int available_actions[] = {0, 1, 0, 1, 2, 3};
    int joint_actions[] = {0, 0, 0, 1, 1, 0, 1, 1};
//    int joint_actions[] = {0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3};
    //double joint_action_values[] = {1, 5, 3, 0, 4, 2, 2, 3};
    double joint_action_values[] = {1, 1, 5, 0, 0, 5, 3, 3};
//    double joint_action_values[] = {1, 1, 0, 0, 0, 0, 3, 3};
//    double joint_action_values[] = {-4, 4, -4, 4, 20, -20, -7, 7, -6, 6, 1, -1, -4, 4, 9, -9};
    int num_iterations = 100;
    double epsilon = 0;
//    double temperatures[] = {0.3, 0.3};
    double temperatures[] = {7, 7};
    bool initial_uniform = true;
    double* initial_policies = nullptr;
    int compute_mode = 10;
    double moving_avg = 0.9;
    double result_values[] = {0, 0};
    double result_policies[] = {0, 0, 0, 0, 0};

//    int num_player = 2;
//    int num_available_actions[] = {1, 2};
//    int available_actions[] = {0, 0, 3};
//    int joint_actions[] = {0, 0, 0, 3};
//    float joint_action_values[] = {0.38500001, -0.38500001, -0.38500001,  0.38500001};
//    int num_iterations = 100;
//    int weighting_type = 1;
//    float ratio = 10;
//    bool initial_uniform = true;
//    float* initial_policies = nullptr;
//    bool use_cumulative_sum = false;
//    float moving_avg = 0.9;
//    float result_values[] = {0, 0};
//    float result_policies[] = {0, 0, 0};

//    compute_sbr_cpp(
//            num_player,
//            num_available_actions,
//            available_actions,
//            joint_actions,
//            joint_action_values,
//            num_iterations,
//            epsilon,
//            factors,
//            initial_uniform,
//            initial_policies,
//            compute_mode,
//            moving_avg,
//            result_values,
//            result_policies
//    );
//    rm_qr(
//        num_available_actions,  //shape (num_player_at_turn,)
//        available_actions,  // shape (sum(num_available_actions))
//        joint_actions, // shape (prod(num_available_actions) * num_player)
//        joint_action_values, // shape (prod(num_available_actions) * num_player)
//        0,
//        10000,
//        0.3,
//        0.2,
//        result_values,  // shape (num_players)
//        result_policies // shape (sum(num_available_actions))
//    );

    compute_logit_cpp(
            num_player,
            num_available_actions,
            available_actions,
            joint_actions,
            joint_action_values,
            num_iterations,
            epsilon,
            temperatures,
            initial_uniform,
            compute_mode,  // mode
            0.5, // hp0
            1.5, // hp1
            initial_policies,
            result_values,
            result_policies
    );

//    compute_2p_nash(num_available_actions, available_actions, joint_actions, joint_action_values, result_values,
//                    result_policies);



/////////////////////////////////////////////////////////////////////////////////////


//    int w = 5;
//    int h = 5;
//    int num_snakes = 2;
//    int min_food = 0;
//    int food_spawn_chance = 0;
//    int init_turns_played = 0;
//    bool spawn_snakes_randomly = false;
//    int snake_body_lengths[] = {4, 3};
//    int max_body_length = 4;
//    int snake_bodies[] = {2, 2, 2, 1, 2, 0, 1, 0, 3, 3, 3, 2, 3, 1};
//    int num_init_food = 0;
//    int food_spawns[] = {};
//    int snake_health[] = {100, 100};
//    int snake_len[] = {4, 3};
//    int max_health[] = {100, 100};
//
////    GameState* state = init(w, h, num_snakes, 1, 25);
////    GameState* state1 = init_fixed(w, h, num_snakes, min_food, food_spawn_chance, snake_spawns, num_init_food, food_spawns);
////    GameState* state2 = clone(state1);
////    bool eq = equals(state1, state2);
////    cout << eq;
//    GameState* state = init(w, h, num_snakes, min_food, food_spawn_chance, init_turns_played, spawn_snakes_randomly,
//                            snake_body_lengths, max_body_length, snake_bodies, num_init_food, food_spawns, snake_health,
//                            snake_len, max_health);
//
//    cout << "created state\n";
//    char arr[300];
//    draw_to_arr(state, arr);
//    cout << arr;
//
//    int actions[] = {0, 0};
//    step(state, actions);
//
//    draw_to_arr(state, arr);
//    cout << arr;
//    cout << "stepped\n";
//
//    close(state);


    //int arr[] = {0, 0};
    //area_control(state, arr);

//    float arr[7*7*5];
//    construct_custom_encoding(
//            state,
//            arr,
//            false,
//            false,
//            true,
//            false,
//            true,
//            1,
//            false,
//            true,
//            true,
//            false,
//            false,
//            false
//    );


//    int actions[] = {RIGHT, LEFT};
//    float rewards[num_snakes];
//    int dead[num_snakes];
//
//    char arr[300];
//    draw_to_arr(state, arr);
//    cout << arr;
//
//    step(state, actions, rewards, dead);
//
//    draw_to_arr(state, arr);
//    cout << arr;
//
//    step(state, actions, rewards, dead);
//
//    draw_to_arr(state, arr);
//    cout << arr;


//    for(int i = 0; i < 10; i++){
//
//        step(state, actions, rewards, dead);
//
//        char arr2[300];
//        draw_to_arr(state, arr2);
//        cout << arr;
//    }





    return 0;
}
