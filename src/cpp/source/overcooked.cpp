//
// Created by mahla on 11/3/2023.
//

#include <utility>
#include <string>
#include <cmath>
#include <vector>

#include "../header/overcooked.h"


OvercookedGameState::OvercookedGameState(
        vector<int> board,
        int turn,
        int w,
        int h,
        vector<Player> players,
        vector<int> tile_states,
        int horizon,
        int cooking_time,
        const OvercookedRewards reward_specs
):
    board(std::move(board)),
    turn(turn),
    w(w),
    h(h),
    players(std::move(players)),
    tile_states(std::move(tile_states)),
    horizon(horizon),
    cooking_time(cooking_time),
    reward_specs(reward_specs),
    dish_pickup_rewards_available(0),
    dish_pickup_rewards_increment_next_round(0)
{
    for (int x = 0; x < w; x++){
        for (int y = 0; y < h; y++){
            int idx = x + w * y;
            if (this->board[idx] == POT_TILE){
                pot_positions.emplace_back(x, y);
            }
        }
    }
}


OvercookedGameState::OvercookedGameState(
        const OvercookedGameState& other
) = default;


OvercookedGameState::~OvercookedGameState(){
    board.clear();
    players.clear();
    tile_states.clear();
}


OvercookedGameState& OvercookedGameState::operator=(const OvercookedGameState& other)= default;


OvercookedGameState* init_overcooked(
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
    // convert input to proper format
    vector<int> board_vector(board, board + w * h);
    Player p0 = {Coord(start_pos[0], start_pos[1]), start_pos[2], NO_ITEM};
    Player p1 = {Coord(start_pos[3], start_pos[4]), start_pos[5], NO_ITEM};
    vector<Player> player_vector = {p0, p1};
    OvercookedRewards rewards = {
            placement_in_pot_reward,
            dish_pickup_reward,
            soup_pickup_reward,
            soup_delivery_reward,
            soup_cooking_reward
    };
    // create tile states. All non pot/counter tiles are -1, counters and pots are empty
    vector<int> tile_states(w*h, -1);
    for(int idx = 0; idx < w*h; idx++){
        if (board_vector[idx] == COUNTER_TILE or board_vector[idx] == POT_TILE) tile_states[idx] = NO_ITEM;
    }
    auto* state_p = new OvercookedGameState(
        board_vector,
        0,
        w,
        h,
        player_vector,
        tile_states,
        horizon,
        cooking_time,
        rewards
    );
    return state_p;
}


OvercookedGameState* clone_overcooked(OvercookedGameState* state){
    auto* copy = new OvercookedGameState(*state);
    return copy;
}


void close_overcooked(OvercookedGameState* state){
    delete state;
}


Coord position_after_action(const Player& player, int action){
    Coord new_pos = {player.position.first, player.position.second};
    switch(action){
        case UP_ACTION:
            new_pos.second -= 1;
            break;
        case DOWN_ACTION:
            new_pos.second += 1;
            break;
        case RIGHT_ACTION:
            new_pos.first += 1;
            break;
        case LEFT_ACTION:
            new_pos.first -= 1;
            break;
        default:
            break;
    }
    return new_pos;
}

Coord faced_tile(const Player& player){
    return position_after_action(player, player.orientation);
}

bool is_movement_action(const int action){
    return action < 4;
}


void step_resolve_orientation(OvercookedGameState* state, int player_id, int action){
    if (not is_movement_action(action)) return;
    state->players[player_id].orientation = action;
}


double step_resolve_helper(OvercookedGameState* state, int player_id, int action){
    double reward = 0;
    int other_player_id = 1 - player_id;
    Player other_player = state->players[other_player_id];
    if (action == STAY_ACTION) return 0;  // using the stay action, nothing happens
    // resolve action
    if (is_movement_action(action)){  // movement
        step_resolve_orientation(state, player_id, action);
        Coord new_pos = position_after_action(state->players[player_id], action);
        int new_tile_idx = new_pos.first + state->w * new_pos.second;
        if (state->board[new_tile_idx] != EMPTY_TILE) return 0;  // player tries to run into something
        if (new_pos == other_player.position) return 0;
        state->players[player_id].position = new_pos;
    } else {  // interaction
        Coord cur_faced_tile = faced_tile(state->players[player_id]);
        int faced_tile_idx = cur_faced_tile.first + state->w * cur_faced_tile.second;
        int faced_tile_type = state->board[faced_tile_idx];
        int faced_tile_state = state->tile_states[faced_tile_idx];
        if (state->players[player_id].held_item == NO_ITEM) {
            if (faced_tile_type == POT_TILE and faced_tile_state == THREE_POT) {
                // player faces a pot with three onions and holds no item. Start cooking.
                // all other interactions with pot require an item of the player
                state->tile_states[faced_tile_idx] = 4 + state->cooking_time;
                reward += state->reward_specs.start_cooking;
                state->dish_pickup_rewards_increment_next_round += 1;
            } else if (faced_tile_type == ONION_TILE){
                // player picks up an onion from dispenser
                state->players[player_id].held_item = ONION_ITEM;
            } else if (faced_tile_type == DISH_TILE){
                // player picks up a dish from dispenser
                state->players[player_id].held_item = DISH_ITEM;
                if (state->dish_pickup_rewards_available > 0){
                    reward += state->reward_specs.dish_pickup;
                    state->dish_pickup_rewards_available -= 1;
                }
            }
        } else if (state->players[player_id].held_item == ONION_ITEM){
            if (faced_tile_type == POT_TILE and faced_tile_state < THREE_POT){
                // player places onion in pot
                state->tile_states[faced_tile_idx] += 1;
                reward += state->reward_specs.placement_in_pot;
                state->players[player_id].held_item = NO_ITEM;
            } else if (faced_tile_type == COUNTER_TILE and faced_tile_state == NO_ITEM){
                // player places onion on counter
                state->tile_states[faced_tile_idx] = ONION_ITEM;
                state->players[player_id].held_item = NO_ITEM;
            }
        } else if (state->players[player_id].held_item == DISH_ITEM){
            if (faced_tile_type == POT_TILE and faced_tile_state == DONE_POT) {
                // player picks up a soup from a pot
                state->tile_states[faced_tile_idx] = EMPTY_POT;
                state->players[player_id].held_item = SOUP_ITEM;
                reward += state->reward_specs.soup_pickup;
            } else if (faced_tile_type == COUNTER_TILE and faced_tile_state == NO_ITEM){
                // player places dish on counter
                state->tile_states[faced_tile_idx] = DISH_ITEM;
                state->players[player_id].held_item = NO_ITEM;
            }
        } else {  // player.held_item == SOUP_ITEM
            if (faced_tile_type == SERVING_TILE){
                // player delivers soup at serving location
                state->players[player_id].held_item = NO_ITEM;
                reward += state->reward_specs.soup_delivery;
            } else if (faced_tile_type == COUNTER_TILE and faced_tile_state == NO_ITEM){
                // player places soup on counter
                state->tile_states[faced_tile_idx] = SOUP_ITEM;
                state->players[player_id].held_item = NO_ITEM;
            }
        }
    }
    return reward;
}


double step_overcooked(OvercookedGameState* state, int* actions){
    // update bookkeeping information
    state->dish_pickup_rewards_available += state->dish_pickup_rewards_increment_next_round;
    state->dish_pickup_rewards_increment_next_round = 0;
    // faced tiles and positions after possible move
    Coord faced_square_0 = faced_tile(state->players[0]);
    Coord faced_square_1 = faced_tile(state->players[1]);
    Coord new_pos_0 = position_after_action(state->players[0], actions[0]);
    Coord new_pos_1 = position_after_action(state->players[1], actions[1]);
    int faced_tile_idx_1 = faced_square_1.first + state->w * faced_square_1.second;
    int faced_tile_type_1 = state->board[faced_tile_idx_1];
    double reward = 0;
    // resolve actions
    if (actions[0] == INTERACT_ACTION and actions[1] == INTERACT_ACTION and faced_square_0 == faced_square_1
            and (faced_tile_type_1 == COUNTER_TILE or faced_tile_type_1 == POT_TILE)){
        // both player face the same counter/pot and want to interact with it
        if (faced_tile_type_1 == POT_TILE){
            if (state->tile_states[faced_tile_idx_1] == THREE_POT and (state->players[0].held_item == NO_ITEM
                    or state->players[1].held_item == NO_ITEM)){
                // let the pot start cooking
                state->tile_states[faced_tile_idx_1] = 4 + state->cooking_time;
                reward += state->reward_specs.start_cooking;
            } else if (state->tile_states[faced_tile_idx_1] < THREE_POT and (state->players[0].held_item == ONION_ITEM
                    or state->players[1].held_item == ONION_ITEM)){
                // at least one player tries to put an onion in the cooking pot
                if (state->players[0].held_item == ONION_ITEM and state->players[1].held_item == ONION_ITEM
                        and state->tile_states[faced_tile_idx_1] == TWO_POT){
                    // both players try to put an onion in a pot, but only one space is left. Do nothing
                } else {
                    // in all other cases resolve the interaction sequentially
                    reward += step_resolve_helper(state, 0, actions[0]);
                    reward += step_resolve_helper(state, 1, actions[1]);
                }
            }  // if both players try to retrieve a soup at the same time, no player gets the soup
        } else {  // faced tile is counter
            if (state->players[0].held_item == NO_ITEM and state->players[1].held_item != NO_ITEM){
                // p1 gives their item directly to p0
                state->players[0].held_item = state->players[1].held_item;
                state->players[1].held_item = NO_ITEM;
            } else if (state->players[1].held_item == NO_ITEM and state->players[0].held_item != NO_ITEM){
                // p0 gives their item directly to p1
                state->players[1].held_item = state->players[0].held_item;
                state->players[0].held_item = NO_ITEM;
            } // if none or both players hold an item, do nothing
        }
    } else if (is_movement_action(actions[0]) and is_movement_action(actions[1])){
        // Both players move
        if (new_pos_0 == new_pos_1){
            // both players try to move onto the same tile, which results in no movement at all. Only resolve directions
            step_resolve_orientation(state, 0, actions[0]);
            step_resolve_orientation(state, 1, actions[1]);
        } else {
            if (new_pos_0 == state->players[1].position) {
                // p0 moves to p1: resolve p1 first, then p0
                reward += step_resolve_helper(state, 1, actions[1]);
                reward += step_resolve_helper(state, 0, actions[0]);
            } else {
                // otherwise normal execution order
                reward += step_resolve_helper(state, 0, actions[0]);
                reward += step_resolve_helper(state, 1, actions[1]);
            }
        }
    } else {
        // in all other cases, the movement of players can be resolved sequentially
        reward += step_resolve_helper(state, 0, actions[0]);
        reward += step_resolve_helper(state, 1, actions[1]);
    }
    // update pots
    for (Coord pot_pos: state->pot_positions){
        int idx = pot_pos.first + pot_pos.second * state->w;
        if (state->tile_states[idx] > 4){
            state->tile_states[idx] -= 1;
        }
    }
    // update turns played
    state->turn += 1;
    return reward;
}


void char_overcooked_matrix(OvercookedGameState* state, char* matrix){
    int i = 0;
    matrix[i++] = '\n';
    for (int y = 0; y < state->h; y++){
        for (int x = 0; x < state->w; x++){
            int idx = state->w * y + x;
            bool skip = false;
            if (state->board[idx] == COUNTER_TILE){
                matrix[i++] = 'C';
            } else if (state->board[idx] == ONION_TILE){
                matrix[i++] = 'O';
            } else if (state->board[idx] == DISH_TILE){
                matrix[i++] = 'D';
            } else if (state->board[idx] == SERVING_TILE){
                matrix[i++] = 'S';
            } else if (state->board[idx] == POT_TILE){
                matrix[i++] = std::to_string(state->tile_states[idx])[0];
                if (state->tile_states[idx] > 9){
                    matrix[i++] = std::to_string(state->tile_states[idx])[1];
                    skip = true;
                }
            } else if (state->board[idx] == EMPTY_TILE){
                bool player_here = false;
                for (Player p: state->players){
                    if (x == p.position.first and y == p.position.second){
                        if (p.held_item == NO_ITEM){
                            matrix[i++] = 'P';
                        } else {
                            matrix[i++] = std::to_string(p.held_item)[0];
                        }
                        matrix[i++] = std::to_string(p.orientation)[0];
                        skip = true;
                        player_here = true;
                    }
                }
                if (not player_here){
                    matrix[i++] = ' ';
                }
            } else if (state->board[idx] == COUNTER_TILE){
                matrix[i++] = std::to_string(state->tile_states[idx])[0];
            }
            if (not skip) matrix[i++] = ' ';
        }
        matrix[i++] = '\n';
    }
    matrix[i++] = '\n';
}

int arr_idx(
        int x,
        int y,
        int z,
        int y_dim,
        int z_dim,
        int x_off,
        int y_off
){
    return (x + x_off) * y_dim * z_dim + (y + y_off) * z_dim + z;
}

void fill_layer_overcooked(float* arr, float value, int layer_id, int x_dim, int y_dim, int z_dim){
    for(int x = 0; x < x_dim; x++) {
        for (int y = 0; y < y_dim; y++) {
            arr[arr_idx(x, y, layer_id, y_dim, z_dim, 0, 0)] = value;
        }
    }
}

void construct_overcooked_encoding(
        OvercookedGameState* state, 
        float* arr, 
        int player,
        bool include_temperature,
        float temperature
){
    int x_dim = state->w;
    int y_dim = state->h;
    int z_dim = 16;
    if (include_temperature){
        z_dim += 1;
    }
    int x_pad = 0;
    int y_pad = 0;
    if (x_dim != y_dim){
        x_dim = max(x_dim, y_dim);
        y_dim = max(x_dim, y_dim);
        x_pad = x_dim - state->w;
        y_pad = y_dim - state->h;
    }
    int x_off = floor(x_pad / 2.0);
    int y_off = floor(y_pad / 2.0);
    int layer_id = 0;
    // player layer encodings
    int player_id = player;
    for (int p_idx = 0; p_idx < 2; p_idx++){
        Player p = state->players[player_id];
        // position
        arr[arr_idx(p.position.first, p.position.second, layer_id, y_dim, z_dim, x_off, y_off)] = 1;
        layer_id += 1;
        // orientation
        if (p.orientation == UP_ACTION){
            arr[arr_idx(p.position.first, p.position.second - 1, layer_id, y_dim, z_dim, x_off, y_off)] = 1;
        } else if (p.orientation == DOWN_ACTION){
            arr[arr_idx(p.position.first, p.position.second + 1, layer_id, y_dim, z_dim, x_off, y_off)] = 1;
        } else if (p.orientation == RIGHT_ACTION){
            arr[arr_idx(p.position.first + 1, p.position.second, layer_id, y_dim, z_dim, x_off, y_off)] = 1;
        } else if (p.orientation == LEFT_ACTION){
            arr[arr_idx(p.position.first - 1, p.position.second, layer_id, y_dim, z_dim, x_off, y_off)] = 1;
        }
        layer_id += 1;
        // held item features
        if (p.held_item == SOUP_ITEM){
            // soup position: layer 11
            int cur_idx = arr_idx(p.position.first, p.position.second, 11, y_dim, z_dim, x_off, y_off);
            arr[cur_idx] = 1;
        } else if (p.held_item == DISH_ITEM){
            // dish position: layer 12
            int cur_idx = arr_idx(p.position.first, p.position.second, 12, y_dim, z_dim, x_off, y_off);
            arr[cur_idx] = 1;
        } else if (p.held_item == ONION_ITEM){
            // onion position: layer 13
            int cur_idx = arr_idx(p.position.first, p.position.second, 13, y_dim, z_dim, x_off, y_off);
            arr[cur_idx] = 1;
        }
        // update player to other player
        player = 1 - player;
    }
    // base map + variable map encodings
    for (int x = 0; x < state->w; x++){
        for (int y = 0; y < state->h; y++){
            int tile_idx = x + state->w * y;
            int tile_type = state->board[tile_idx];
            int tile_state = state->tile_states[tile_idx];
            int cur_idx;
            if (tile_type == POT_TILE){
                // pot-location layer 4
                cur_idx = arr_idx(x, y, 4, y_dim, z_dim, x_off, y_off);
                arr[cur_idx] = 1;
                // pot state (only by position if multiple pots)
                if (state->pot_positions.size() > 1){
                    // number of onions in the pot: layer 9
                    cur_idx = arr_idx(x, y, 9, y_dim, z_dim, x_off, y_off);
                    arr[cur_idx] = (float) (min(3, tile_state) / 3.0);
                    // soup cook time remaining: layer 10
                    float time_remaining = 1.0;
                    if (tile_state > 3) time_remaining = (float) (tile_state - 4.0) / (float) state->cooking_time;
                    cur_idx = arr_idx(x, y, 10, y_dim, z_dim, x_off, y_off);
                    arr[cur_idx] = time_remaining;
                }
                // soup: layer 11
                if (tile_state == DONE_POT){
                    cur_idx = arr_idx(x, y, 11, y_dim, z_dim, x_off, y_off);
                    arr[cur_idx] = 1;
                }
            } else if (tile_type == COUNTER_TILE){
                // counter location: layer 5
                cur_idx = arr_idx(x, y, 5, y_dim, z_dim, x_off, y_off);
                arr[cur_idx] = 1;
                // positions of items dropped on counter
                if (tile_state == SOUP_ITEM){
                    // soup position: layer 11
                    cur_idx = arr_idx(x, y, 11, y_dim, z_dim, x_off, y_off);
                    arr[cur_idx] = 1;
                } else if (tile_state == DISH_ITEM){
                    // dish positions: layer 12
                    cur_idx = arr_idx(x, y, 12, y_dim, z_dim, x_off, y_off);
                    arr[cur_idx] = 1;
                } else if (tile_state == ONION_ITEM){
                    // onion positions: layer 13
                    cur_idx = arr_idx(x, y, 13, y_dim, z_dim, x_off, y_off);
                    arr[cur_idx] = 1;
                }
            } else if (tile_type == ONION_TILE){
                // onion dispenser location: layer 6
                cur_idx = arr_idx(x, y, 6, y_dim, z_dim, x_off, y_off);
                arr[cur_idx] = 1;
            } else if (tile_type == DISH_TILE){
                // dish dispenser location: layer 7
                cur_idx = arr_idx(x, y, 7, y_dim, z_dim, x_off, y_off);
                arr[cur_idx] = 1;
            } else if (tile_type == SERVING_TILE){
                // serving location: layer 8
                cur_idx = arr_idx(x, y, 8, y_dim, z_dim, x_off, y_off);
                arr[cur_idx] = 1;
            }
        }
    }
    // full layers: only if a single pot in environment
    if (state->pot_positions.size() == 1){
        Coord pot_pos = state->pot_positions.front();
        int cur_idx = pot_pos.first + state->w * pot_pos.second;
        int tile_state = state->tile_states[cur_idx];
        // number of onions in the pot: layer 9
        float relative_onions_in_pot = (float) (min(3, tile_state) / 3.0);
        fill_layer_overcooked(arr, relative_onions_in_pot, 9, x_dim, y_dim, z_dim);
        // soup cook time remaining: layer 10
        float time_remaining = 1.0;
        if (tile_state > 3) time_remaining = (float) (tile_state - 4.0) / (float) state->cooking_time;
        fill_layer_overcooked(arr, time_remaining, 10, x_dim, y_dim, z_dim);
    }
    // time features:
    // urgency if time_left < 40: layer 14
    if (state->turn > state->horizon - 40){
        fill_layer_overcooked(arr, 1, 14, x_dim, y_dim, z_dim);
    }
    // time step in environment: layer 15
    float time_remaining = (float) (state->horizon - state->turn) / (float) state->horizon;
    fill_layer_overcooked(arr, time_remaining, 15, x_dim, y_dim, z_dim);
    // temperature: layer 16
    if (include_temperature){
        fill_layer_overcooked(arr, temperature, 16, x_dim, y_dim, z_dim);
    }

}


bool equals_overcooked(OvercookedGameState* state, OvercookedGameState* other){
    // this assumes that the board is static and always equal for both game states
    if (state->players.front().position != other->players.front().position) return false;
    if (state->players.back().position != other->players.back().position) return false;
    if (state->players.front().held_item != other->players.front().held_item) return false;
    if (state->players.back().held_item != other->players.back().held_item) return false;
    if (state->players.front().orientation != other->players.front().orientation) return false;
    if (state->players.back().orientation != other->players.back().orientation) return false;
    if (state->turn != other->turn) return false;
    if (state->w != other->w or state->h != other->h) return false;
    for (int idx = 0; idx < state->w * state->h; idx++){
        if (state->tile_states[idx] != other->tile_states[idx]) return false;
    }
    return true;
}


