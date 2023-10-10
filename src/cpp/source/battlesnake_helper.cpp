//
// Created by mahla on 26/10/2022.
//

#include "../header/battlesnake.h"
#include "../header/battlesnake_helper.h"

#include <random>
#include <algorithm>
#include <cstdlib>

random_device rd_utils;
mt19937 gen_utils(rd_utils());

void set_seed_utils(int seed){
    gen_utils.seed(seed);
}

deque<Coord> spawn_randomly(int w, int num_snakes){
    //mn, md, mx := 1, (b.Width-1)/2, b.Width-2
    int mx = w - 2;
    int md = (w - 1) / 2;

    //snakes can either spawn in the corner with distance one to border or
    //in the middle of a side with distance one to border
    vector<Coord> available_spawns_corner;
    available_spawns_corner.reserve(4);
    if (w == 3){
        available_spawns_corner.emplace_back(0, 0);
        available_spawns_corner.emplace_back(0, 2);
        available_spawns_corner.emplace_back(2, 0);
        available_spawns_corner.emplace_back(2, 2);
    } else {
        available_spawns_corner.emplace_back(1, 1);
        available_spawns_corner.emplace_back(1, mx);
        available_spawns_corner.emplace_back(mx, 1);
        available_spawns_corner.emplace_back(mx, mx);
    }

    vector<Coord> available_spawns_side;
    available_spawns_side.reserve(4);
    if (w == 3){
        available_spawns_side.emplace_back(0, 1);
        available_spawns_side.emplace_back(1, 0);
        available_spawns_side.emplace_back(2, 1);
        available_spawns_side.emplace_back(1, 2);
    } else {
        available_spawns_side.emplace_back(1, md);
        available_spawns_side.emplace_back(md, 1);
        available_spawns_side.emplace_back(md, mx);
        available_spawns_side.emplace_back(mx, md);
    }

    std::shuffle(available_spawns_corner.begin(), available_spawns_corner.end(), gen_utils);
    std::shuffle(available_spawns_side.begin(), available_spawns_side.end(), gen_utils);

    //decide to fill corner or sides first
    unsigned random = abs((int)gen_utils());
    bool corner_first = random % 2;
    int counter = 0;
    deque<Coord> result;

    if(corner_first){
        for (int i = 0; i < 4 and counter < num_snakes; i++){
            result.push_back(available_spawns_corner[i]);
            counter++;
        }
        for (int i = 0; i < 4 and counter < num_snakes; i++){
            result.push_back(available_spawns_side[i]);
            counter++;
        }
    } else {
        for (int i = 0; i < 4 and counter < num_snakes; i++){
            result.push_back(available_spawns_side[i]);
            counter++;
        }
        for (int i = 0; i < 4 and counter < num_snakes; i++){
            result.push_back(available_spawns_corner[i]);
            counter++;
        }
    }
    return result;
}

deque<deque<Coord>> spawn_on_pos(const int* pos, const int* snake_body_lengths, int num_snakes, int max_body_length){
    // arr[x*y_dim*z_dim + y*z_dim + z] = value; <- convert 3d coordinate to flattened index
    deque<deque<Coord>> result;
    for(int s = 0; s < num_snakes; s++){
        int s_len = snake_body_lengths[s];
        deque<Coord> q;
        for (int i = 0; i < s_len; i++){
            int x = pos[s*max_body_length*2 + i*2 + 0];
            int y = pos[s*max_body_length*2 + i*2 + 1];
            q.emplace_back(x, y);
        }
        result.push_back(q);
    }
    return result;
}


list<Coord> initialize_food(int w, int num_snakes, const vector<Coord>& snake_spawns){
    list<Coord> food;
    //one food is always in the middle
    int mid = (w - 1) / 2;
    Coord center = Coord(mid, mid);
    food.push_back(center);

    // Up to 4 snakes can be placed such that food is nearby on small boards.
    // Otherwise, we skip this and only try to place food in the center.
    if (num_snakes > 4 || w == 3) {
        return food;
    }
    // Place 1 food within exactly 2 moves of each snake, but never towards the center or in a corner
    for (Coord snake_pos: snake_spawns){
        vector<Coord> possible_coords;
        possible_coords.emplace_back(snake_pos.first - 1, snake_pos.second - 1);
        possible_coords.emplace_back(snake_pos.first - 1, snake_pos.second + 1);
        possible_coords.emplace_back(snake_pos.first + 1, snake_pos.second - 1);
        possible_coords.emplace_back(snake_pos.first + 1, snake_pos.second + 1);

        deque<Coord> valid_coords;
        //check for all possible locations if they are valid
        for (Coord food_pos: possible_coords){
            // no more food in center on very small boards
            if (food_pos == center) {
                continue;
            }
            //ignore points already occupied
            bool is_already_occ = false;
            for (Coord fixed_food: food){
                if (fixed_food == food_pos){
                    is_already_occ = true;
                    break;
                }
            }
            if (is_already_occ){
                continue;
            }
            // Food must be further than snake from center on at least one axis
            bool is_away_from_center = false;
            if ((food_pos.first < snake_pos.first and snake_pos.first < center.first)
                or (food_pos.first > snake_pos.first and snake_pos.first > center.first)
                or (food_pos.second < snake_pos.second and snake_pos.second < center.second)
                or (food_pos.second > snake_pos.second and snake_pos.second > center.second)){
                is_away_from_center = true;
            }
            if (not is_away_from_center){
                continue;
            }
            // Don't spawn food in corners
            if ((food_pos.first == 0 or food_pos.first == w-1)
                and (food_pos.second == 0 or food_pos.second == w-1)){
                continue;
            }
            valid_coords.push_back(food_pos);
        }
        if (!valid_coords.empty()){
            // choose one random valid position
            unsigned index = abs((int)gen_utils()) % valid_coords.size();
            food.push_back(valid_coords[index]);
        }
    }
    return food;
}

list<Coord> food_on_pos(int* pos, int num_food){
    list<Coord> result;
    for(int i = 0; i < num_food; i++){
        result.emplace_back(pos[2*i], pos[2*i+1]);
    }
    return result;
}


Coord new_position(Coord old, int move, GameState* state){
    if (move == RIGHT){
        int new_x = old.first + 1;
        if (state->wrapped and new_x == state->w) new_x = 0;
        return {new_x, old.second};
    } else if (move == DOWN) {
        int new_y = old.second - 1;
        if (state->wrapped and new_y == -1) new_y = state->h - 1;
        return {old.first, new_y};
    } else if (move == LEFT) {
        int new_x = old.first - 1;
        if (state->wrapped and new_x == -1) new_x = state->w - 1;
        return {new_x, old.second};
    } else { //default move up
        int new_y = old.second + 1;
        if (state->wrapped and new_y == state->h) new_y = 0;
        return {old.first, new_y};
    }
}

bool in_bounds(GameState* state, Coord pos){
    return pos.first >= 0 and pos.second >= 0 and pos.first < state->w and pos.second < state->h;
}

vector<int> possible_food_spawns(GameState* state){
    vector<int> food_spawns(state->w*state->h, true);
    for(auto& s: state->snakes){
        if(not s->alive) continue;
        //food cannot spawn on top of body, except tail after it moved
        for(auto c: s->body){
            if(in_bounds(state, c)){
                food_spawns[c.second * state->w + c.first] = false;
            }
        }
        //include tail if it stays in place after move
        if(s->length > (int)s->body.size()){
            Coord tail = s->body.back();
            food_spawns[tail.second * state->w + tail.first] = false;
        }
        //food cannot spawn directly in front of snake
        Coord head = s->body.front();
        for(int a = 0; a < 4; a++){
            Coord new_pos = new_position(head, a, state);
            if(in_bounds(state, new_pos)){
                food_spawns[new_pos.second * state->w + new_pos.first] = false;
            }
        }
    }
    //food cannot spawn on top of existing food
    for(auto c: state->food){
        food_spawns[c.second * state->w + c.first] = false;
    }
    return food_spawns;
}


void place_food_randomly(GameState* state) {
    int num_food = (int) (state->food.size());
    int n_to_place = 0;
    //place the minimum amount of food
    if (num_food < state->min_food){
        n_to_place = state->min_food - num_food;
    }
    //randomly decide if to add another food
    if (((int) (abs((int)gen_utils()) % 100)) < state->food_spawn_chance){
        n_to_place++;
    }
    //all possible positions
    vector<int> food_spawns = possible_food_spawns(state);
    //int num_food_spawns = reduce(food_spawns.begin(), food_spawns.end());
    int num_food_spawns = 0;
    for (int food_spawn : food_spawns){
        num_food_spawns += food_spawn;
    }
    //sanity checks
    if(num_food_spawns < n_to_place){
        n_to_place = num_food_spawns;
    }
    if(num_food_spawns == 0) return;
    for (int i = 0; i < n_to_place; i++){
        // get random food spawn
        int rng = (int)(abs((int)gen_utils()) % num_food_spawns);
        int counter = 0;
        for(int j = 0; j < (int)food_spawns.size(); j++){
            if(food_spawns[j]){
                if (counter == rng){
                    // convert index to Coord
                    int x = j % state->w;
                    int y = j / state->w;
                    state->food.emplace_back(x, y);
                    food_spawns[j] = 0;
                    num_food_spawns--;
                    break;
                }
                counter++;
            }
        }
    }
}

///
/// \param state GameState
/// \return Vector describing an encoding of the state for human representation
vector<int> human_repr(GameState* state){
    vector<int> board(state->w * state->h, 0);
    //snake bodies marked with amount of time to live
    for (auto& s: state->snakes){
        if (not s->alive) continue;
        int n = (int) s->body.size();
        for(int i = 1; i < n; i++){
            Coord c = s->body[i];
            board[c.second * state->w + c.first] = s->length-i;
        }
        //mark head separately with -1
        board[s->body.front().second * state->w + s->body.front().first] = -1;
    }
    //food marked with -2
    for (Coord c: state->food){
        board[c.second * state->w + c.first] = -2;
    }
    //hazards marked with -3
    for (int i = 0; i < state->w * state->h; i++){
        if (board[i] == 0 and state->hazards[i]){
            board[i] = -3;
        }
    }
    return board;
}

void draw_to_arr(GameState* state, char* arr){
    vector<int> board = human_repr(state);
    int i = 0;
    arr[i++] = '\n';
    for (int y = state->h-1; y >= 0; y--){
        for (int x = 0; x < state->w; x++){
            int code = board[y * state->w + x];
            if (code > 0 and code < 10){ //body without head
                arr[i++] = (char)(48 + code);
            } else if (code > 0){ //empty
                arr[i++] = '1';
            } else if (code == 0){ //empty
                arr[i++] = 'O';
            } else if (code == -1){ //head
                arr[i++] = 'X';
            } else if (code == -2){ //food
                arr[i++] = '@';
            } else if (code == -3){ //hazard
                arr[i++] = '%';
            }
            arr[i++] = ' ';
        }
        arr[i++] = '\n';
    }
    arr[i++] = '\n';
}


int arr_idx(
        int x,
        int y,
        int z,
        int y_dim,
        int z_dim
){
    return x * y_dim * z_dim + y * z_dim + z;
}


bool in_encoding_bounds(
     int x,
     int y,
     int x_size,
     int y_size
){
    return x >= 0 and y >= 0 and x < x_size and y < y_size;
}


//this should only be called if encoding is centered and wrapped mode used
void all_tilings(
        float* arr,
        GameState* state,
        float value,
        int x,
        int y,
        int z,
        int x_dim,
        int y_dim,
        int z_dim
){
    //iterate all tilings, center:
    arr[arr_idx(x, y, z, y_dim, z_dim)] = value;
    //left
    if (in_encoding_bounds(x - state->w, y, x_dim, y_dim)){
        arr[arr_idx(x - state->w, y, z, y_dim, z_dim)] = value;
    }
    //right
    if (in_encoding_bounds(x + state->w, y, x_dim, y_dim)){
        arr[arr_idx(x + state->w, y, z, y_dim, z_dim)] = value;
    }
    //up
    if (in_encoding_bounds(x, y + state->h, x_dim, y_dim)){
        arr[arr_idx(x, y + state->h, z, y_dim, z_dim)] = value;
    }
    //down
    if (in_encoding_bounds(x, y - state->h, x_dim, y_dim)){
        arr[arr_idx(x, y - state->h, z, y_dim, z_dim)] = value;
    }
    //left up
    if (in_encoding_bounds(x - state->w, y + state->h, x_dim, y_dim)){
        arr[arr_idx(x - state->w, y + state->h, z, y_dim, z_dim)] = value;
    }
    //left down
    if (in_encoding_bounds(x - state->w, y - state->h, x_dim, y_dim)){
        arr[arr_idx(x - state->w, y - state->h, z, y_dim, z_dim)] = value;
    }
    //right up
    if (in_encoding_bounds(x + state->w, y + state->h, x_dim, y_dim)){
        arr[arr_idx(x + state->w, y + state->h, z, y_dim, z_dim)] = value;
    }
    //right down
    if (in_encoding_bounds(x - state->w, y - state->h, x_dim, y_dim)){
        arr[arr_idx(x - state->w, y - state->h, z, y_dim, z_dim)] = value;
    }
}


void fill_layer(
        float* arr,
        float value,
        int x_dim,
        int y_dim,
        int z_dim,
        int layer_id
){
    for(int x = 0; x < x_dim; x++) {
        for (int y = 0; y < y_dim; y++) {
            arr[arr_idx(x, y, layer_id, y_dim, z_dim)] = value;
        }
    }
}

void write_arr(
        GameState* state,
        float* arr,
        float value,
        int x,
        int y,
        int x_dim,
        int y_dim,
        int z_dim,
        int layer_id,
        bool wrapped,
        bool centered
){
    if (not wrapped or not centered) {
        arr[arr_idx(x, y, layer_id, y_dim, z_dim)] = value;
    } else {
        all_tilings(arr, state, value, x, y, layer_id, x_dim, y_dim, z_dim);
    }
}

void fill_body(
        GameState* state,
        float* arr,
        int snake_id,
        float value,
        int x_dim,
        int y_dim,
        int z_dim,
        int layer_id,
        bool wrapped,
        bool centered,
        int x_off,
        int y_off
){
    Snake* sp = state->snakes.at(snake_id);
    for (auto p: sp->body) {
        if (not in_bounds(state, p)) continue;
        write_arr(state, arr, value, p.first+x_off, p.second+y_off, x_dim, y_dim, z_dim, layer_id, wrapped, centered);
    }
}

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
) {
    // arr[x*y_dim*z_dim + y*z_dim + z] = value; <- convert 3d coordinate to flattened index
    // Calculate the dimensions of the state
    int x_dim;
    int y_dim;
    if (centered){
        x_dim = 2 * state->w - 1;
        y_dim = 2 * state->h - 1;
    } else {
        x_dim = state->w + 2;
        y_dim = state->h + 2;
        if (state->wrapped){  // we do not need borders in wrapped mode
            x_dim -= 2;
            y_dim -= 2;
        }
    }
    bool wrapped = state->wrapped;

    int general_layers = include_current_food + include_next_food + include_board + include_number_of_turns
            + include_dist_map + include_hazards + include_num_food_on_board;
    if (include_temperatures and single_temperature) general_layers += 1;
    int number_of_snake_layers = state->num_snakes;
    if (flatten_snakes) number_of_snake_layers = 2; // One for player snake, one for enemy snakes
    int layers_per_snake = include_snake_body + include_snake_head + include_snake_tail + include_snake_health
            + include_snake_length + include_area_control + include_food_distance + include_tail_distance
            + include_snake_body_as_one_hot;
    int enemy_snake_layers = layers_per_snake;
    if (include_temperatures and not single_temperature) enemy_snake_layers += 1;
    int z_dim = layers_per_snake + (number_of_snake_layers - 1) * enemy_snake_layers + general_layers;
    // offsets
    int x_off;
    int y_off;
    if (centered){
        Coord pos = state->snakes.at(player_snake)->body.front();
        x_off = state->w - pos.first - 1;
        y_off = state->h - pos.second - 1;
    } else {
        x_off = 1;
        y_off = 1;
        if (state->wrapped){  // no border in wrapped
            x_off = 0;
            y_off = 0;
        }
    }
    //precompute area control for all snakes
    auto* area_control_arr = (float*) calloc(state->num_snakes, sizeof(float));
    int* food_distance_arr = (int*) calloc(state->num_snakes, sizeof(int));
    int* tail_distance_arr = (int*) calloc(state->num_snakes, sizeof(int));
    bool* reached_tail_arr = (bool*) calloc(state->num_snakes, sizeof(bool));
    bool* reached_food_arr = (bool*) calloc(state->num_snakes, sizeof(bool));
    if (include_area_control or include_food_distance or include_tail_distance){
        area_control(state, area_control_arr, food_distance_arr, tail_distance_arr, reached_tail_arr, reached_food_arr,
                     1.0, 1.0, 1.0, 1.0);
    }
    //precompute max length
    int max_length = 0;
    for (Snake* s: state->snakes){
        if (s->length > max_length) max_length = s->length;
    }

    // Start with general layers
    int layer_id = 0;
    // Food layer
    if (include_current_food) {
        for (auto p: state->food){
            write_arr(state, arr, 1.0, p.first+x_off, p.second+y_off, x_dim, y_dim, z_dim, layer_id, wrapped, centered);
        }
        layer_id++;
    }
    // Next food layer
    if (include_next_food) {
        vector<int> food_spawns = possible_food_spawns(state);
        // count number of possible food spawns
        int num_food_spawns = 0;
        for (int food_spawn : food_spawns){
            num_food_spawns += food_spawn;
        }
        float percentage;
        if (fixed_food_spawn_chance >= 0){
            percentage = (float) (fixed_food_spawn_chance / 100.0);
        } else {
            percentage = (float) (state->food_spawn_chance / 100.0);
        }
        auto prob = (float) (percentage / (float) num_food_spawns);
        for(int x = 0; x < state->w; x++){
            for(int y = 0; y < state->h; y++){
                if(food_spawns[y * state->w + x]){
                    write_arr(state, arr, prob, x+x_off, y+y_off, x_dim, y_dim, z_dim, layer_id, wrapped, centered);
                }
            }
        }
        layer_id++;
    }
    // Board layer
    if (include_board) {
        for (int x = 0; x < x_dim; x++){
            for (int y = 0; y < y_dim; y++){
                if (x >= x_off and y >= y_off and x < x_off+state->w and y < y_off+state->h){
                    //playable space
                    arr[arr_idx(x, y, layer_id, y_dim, z_dim)] = 1.0;
                } else {
                    //blocked space
                    arr[arr_idx(x, y, layer_id, y_dim, z_dim)] = -1.0;
                }
            }
        }
        layer_id++;
    }
    // Number of turns layer
    if (include_number_of_turns) {
        auto value = (float) (state->turn / 100.0);
        fill_layer(arr, value, x_dim, y_dim, z_dim, layer_id);
        layer_id++;
    }
    //distance map layer
    if (include_dist_map) {
        auto max_dist = (float) (state->w + state->h - 2);
        Coord head = state->snakes.at(player_snake)->body.front();
        for(int x = 0; x < x_dim; x++){
            for(int y = 0; y < y_dim; y++){
                auto dist = (float) (std::abs(x - x_off - head.first) + std::abs(y - y_off - head.second));
                arr[arr_idx(x, y, layer_id, y_dim, z_dim)] = dist / max_dist;
            }
        }
        layer_id++;
    }
    //hazard layer
    if (include_hazards){
        for(int x = 0; x < state->w; x++){
            for(int y = 0; y < state->h; y++){
                float value = (float) (state->hazards[y * state->w + x]);
                write_arr(state, arr, value, x+x_off, y+y_off, x_dim, y_dim, z_dim, layer_id, wrapped, centered);
            }
        }
        layer_id++;
    }
    //number of food on board layer
    if (include_num_food_on_board){
        auto num_food =  (float) (((float) state->food.size()) / 10.0);
        fill_layer(arr, num_food, x_dim, y_dim, z_dim, layer_id);
        layer_id++;
    }
    // temperature layer (if single temperature provided)
    if (include_temperatures and single_temperature){
        float single_temp = temperatures[0] / (float) 10.0;
        fill_layer(arr, single_temp, x_dim, y_dim, z_dim, layer_id);
        layer_id++;
    }
    // Snake layers
    bool first_iteration = true;
    for(int s = player_snake; s < state->num_snakes; s++){
        // Skip the player snake since we already added it in the first iteration
        if (s == player_snake and not first_iteration) continue;
        // process enemy snakes
        Snake* snake = state->snakes.at(s);
        if(not snake->alive) {
            if (not flatten_snakes) layer_id += enemy_snake_layers;
            continue;
        }
        // Snake body layer
        if (include_snake_body) {
            auto body_counter = (float) snake->length;
            float decrement = 1.0;
            for(auto p: snake->body){
                if (not in_bounds(state, p)) continue;
                auto value = (float) (body_counter / 10.0);
                write_arr(state, arr, value, p.first+x_off, p.second+y_off, x_dim, y_dim, z_dim, layer_id,
                          wrapped, centered);
                body_counter -= decrement;
            }
            layer_id++;
        }
        //snake body one hot
        if (include_snake_body_as_one_hot) {
            fill_body(state, arr, s, 1.0, x_dim, y_dim, z_dim, layer_id, wrapped, centered, x_off, y_off);
            layer_id++;
        }
        // Snake head layer
        if (include_snake_head) {
            Coord head = snake->body.front();
            write_arr(state, arr, 1.0, head.first+x_off, head.second+y_off, x_dim, y_dim, z_dim, layer_id,
                      wrapped, centered);
            layer_id++;
        }
        // Snake tail layer
        if (include_snake_tail) {
            Coord tail = snake->body.back();
            write_arr(state, arr, 1.0, tail.first+x_off, tail.second+y_off, x_dim, y_dim, z_dim, layer_id,
                      wrapped, centered);
            layer_id++;
        }
        //health
        if (include_snake_health){
            auto value = (float) (snake->health / 100.0);
            if (not flatten_snakes or s == player_snake){
                fill_layer(arr, value, x_dim, y_dim, z_dim, layer_id);
            } else {
                fill_body(state, arr, s, value, x_dim, y_dim, z_dim, layer_id, wrapped, centered, x_off, y_off);
            }
            layer_id++;
        }
        //length
        if (include_snake_length){
            auto value = (float) ((snake->length - max_length) / 5.0);
            if (not flatten_snakes or s == player_snake){
                fill_layer(arr, value, x_dim, y_dim, z_dim, layer_id);
            } else {
                fill_body(state, arr, s, value, x_dim, y_dim, z_dim, layer_id, wrapped, centered, x_off, y_off);
            }
            layer_id++;
        }
        //area control
        if (include_area_control){
            auto value = (float) (((float) area_control_arr[s]) / ((float) state->w * (float) state->h));
            if (not flatten_snakes or s == player_snake){
                fill_layer(arr, value, x_dim, y_dim, z_dim, layer_id);
            } else {
                fill_body(state, arr, s, value, x_dim, y_dim, z_dim, layer_id, wrapped, centered, x_off, y_off);
            }
            layer_id++;
        }
        //food distance
        if (include_food_distance){
            auto value = (float) (((float) food_distance_arr[s]) / ((float) state->w + (float) state->h));
            if (not flatten_snakes or s == player_snake){
                fill_layer(arr, value, x_dim, y_dim, z_dim, layer_id);
            } else {
                fill_body(state, arr, s, value, x_dim, y_dim, z_dim, layer_id, wrapped, centered, x_off, y_off);
            }
            layer_id++;
        }
        //tail distance
        if (include_tail_distance){
            auto value = (float) (((float) tail_distance_arr[s]) / ((float) state->w + (float) state->h));
            if (not flatten_snakes or s == player_snake){
                fill_layer(arr, value, x_dim, y_dim, z_dim, layer_id);
            } else {
                fill_body(state, arr, s, value, x_dim, y_dim, z_dim, layer_id, wrapped, centered, x_off, y_off);
            }
            layer_id++;
        }
        //temperature
        if (include_temperatures and not single_temperature and s != player_snake){
            float value = temperatures[s] / (float) 10.0;
            if (flatten_snakes){
                fill_body(state, arr, s, value, x_dim, y_dim, z_dim, layer_id, wrapped, centered, x_off, y_off);
            } else {
                fill_layer(arr, value, x_dim, y_dim, z_dim, layer_id);
            }
            layer_id++;
        }

        // Adjust layer_id to account for the fact that we may be flattening snakes
        if (first_iteration) {
            first_iteration = false;
            s = -1;
        } else if (flatten_snakes) {
            layer_id -= enemy_snake_layers;
        }
    }
    //free memory
    free(area_control_arr);
    free(food_distance_arr);
    free(tail_distance_arr);
    free(reached_tail_arr);
    free(reached_food_arr);
}


bool equals(GameState* state1, GameState* state2){
//    if (state1 == state2) return true;
    if (state1->turn != state2->turn) return false;
    if (state1->w != state2->w) return false;
    if (state1->h != state2->h) return false;
    if (state1->num_snakes != state2->num_snakes) return false;
    if (state1->min_food != state2->min_food) return false;
    if (state1->food_spawn_chance != state2->food_spawn_chance) return false;
    if (state1->food.size() != state2->food.size()) return false;
    // check if food is equal. Can be in different order
    for (Coord c1: state1->food){
        bool exists = false;
        for (Coord c2: state2->food){
            if (c1 == c2){
                exists = true;
                break;
            }
        }
        if (not exists) return false;
    }
    for (Coord c2: state2->food){
        bool exists = false;
        for (Coord c1: state1->food){
            if (c1 == c2){
                exists = true;
                break;
            }
        }
        if (not exists) return false;
    }
    // check if snakes are equal
    if (state1->snakes.size() != state2->snakes.size()) return false;
    for (const auto s1: state1->snakes){
        Snake* s2 = state2->snakes.at(s1->id);
        if (s1->alive != s2->alive) return false;
        if (s1->length != s2->length) return false;
        if (s1->health != s2->health) return false;
        if (s1->body.size() != s2->body.size()) return false;
        // bodies are in order
        for(int i = 0; i < (int)s1->body.size(); i++){
            if(s1->body.at(i) != s2->body.at(i)) return false;
        }
    }
    return true;
}

void alive(GameState* state, bool* arr){
    for(const auto& s: state->snakes){
        arr[s->id] = s->alive;
    }
}

void snake_length(GameState* state, int* arr){
    for(const auto& s: state->snakes){
        arr[s->id] = s->length;
    }
}

int snake_body_length(GameState* state, int player){
    return (int) state->snakes.at(player)->body.size();
}

void snake_pos(GameState* state, int player, int* arr){
    Snake* s = state->snakes.at(player);
    int i = 0;
    for (auto c: s->body){
        arr[i] = c.first;
        arr[i+1] = c.second;
        i += 2;
    }
}

int num_food(GameState* state){
    return (int) state->food.size();
}

void food_pos(GameState* state, int* arr){
    int i = 0;
    for (auto c: state->food){
        arr[i] = c.first;
        arr[i+1] = c.second;
        i += 2;
    }
}

int turns_played(GameState* state){
    return state->turn;
}

void snake_health(GameState* state, int* arr){
    for(const auto& s: state->snakes){
        arr[s->id] = s->health;
    }
}

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
){
    //initialize frontiers and result map
    /* Encoding of the result map:
     * -1: unclaimed field
     * [0, num_snakes-1]: field is claimed securely by the snake with this id
     * [num_snakes, 2*num_snakes-1]: field was temporarily claimed this round by the snake with id: number - num_snakes
     * [2*num_snakes, 3*num_snakes-1]: field was claimed by snake with this id, but both two snakes died h2h here
     */
    vector<int> result(state->w * state->h, -1);
    vector<int> body_map(state->w * state->h, 0);
    // we need mapping from snake_id to index and vice versa, because some snakes might be dead and not in the arrays
    vector<int> snake_ids(state->num_snakes);  //mapping index -> snake_id
    vector<int> inverse_snake_ids(state->num_snakes);  //mapping snake_id -> index
    deque<deque<Coord>> frontiers;
    deque<deque<Coord>> bodies;
    deque<int> length_offset;  //amount of tiles the tail expands
    //initialization
    int alive_snake_count = 0;
    for(int i = 0; i < (int)(state->snakes.size()); i++){
        Snake* s = state->snakes.at(i);
        if (!s->alive) continue;
        //initial frontier
        deque<Coord> init_frontier;
        init_frontier.push_back(s->body.front());
        frontiers.push_back(init_frontier);
        //initial body
        deque<Coord> init_body(s->body);
        bodies.push_back(init_body);
        for(Coord c: s->body){
            body_map[c.second * state->w + c.first] = 1;
        }
        //offset
        int offset = s->length - (int)s->body.size();
        length_offset.push_back(offset);
        //save id
        snake_ids[alive_snake_count] = i;
        inverse_snake_ids[i] = alive_snake_count;
        alive_snake_count++;
        //set food and tail distance initially to max distance
        food_dist_arr[i] = state->w + state->h;
        tail_dist_arr[i] = state->w + state->h;
    }
    int counter = 1;
    while (true){
        //break condition: all frontiers are emtpy
        bool done = true;
        for(auto& frontier : frontiers){
            if (!frontier.empty()){
                done = false;
                break;
            }
        }
        if (done){
            break;
        }
        //remove tail if no offset present
        for(int i = 0; i < (int)(bodies.size()); i++){
            if(length_offset.at(i) > 0){
                length_offset[i]--;
            } else if (!bodies[i].empty()){
                Coord tail = bodies[i].back();
                body_map[tail.second * state->w + tail.first] = 0;
                bodies[i].pop_back();
            }
        }
        //move heads to new positions
        list<Coord> eval_at_end;
        for(int i = 0; i < (int)(frontiers.size()); i++){
            int cur_frontier_size = (int)(frontiers[i].size());
            for(int j = 0; j < cur_frontier_size; j++){
                //get current head and remove it
                Coord cur_head = frontiers[i].front();
                for (int move = 0; move < 4; move++){
                    Coord new_pos = new_position(cur_head, move, state);
                    //check if new position is valid
                    if (!in_bounds(state, new_pos)) continue;  // out of bounds
                    //field is occupied by someone's body
                    if (body_map[new_pos.second * state->w + new_pos.first]) continue;
                    // get current value on the board
                    int result_map_val = result[new_pos.second * state->w + new_pos.first];
                    // field is securely claimed
                    if (result_map_val >= 0 and result_map_val < state->num_snakes) continue;
                    //field was claimed temporarily this round by other snake or other snakes died both h2h
                    if (result_map_val >= state->num_snakes){
                        int other_id = result_map_val % state->num_snakes;
                        if (other_id == snake_ids[i]) {
                            // this field was already claimed this round by us. No need to claim again
                            continue;
                        }
                        int other_length = state->snakes.at(other_id)->length;
                        int our_length = state->snakes.at(snake_ids[i])->length;
                        // check which snake is longer
                        if (other_length > our_length){
                            continue;  // we lost and cannot claim this field
                        } else if (other_length < our_length){
                            // we can temporarily claim the field, no need to push eval_at_end as pos is already there
                            result[new_pos.second * state->w + new_pos.first] = snake_ids[i] + state->num_snakes;
                        } else {
                            // we would both die here
                            result[new_pos.second * state->w + new_pos.first] = snake_ids[i] + 2*state->num_snakes;
                        }
                    }
                    //field is empty, we can claim it temporarily
                    if (result_map_val == -1){
                        result[new_pos.second * state->w + new_pos.first] = snake_ids[i] + state->num_snakes;
                        eval_at_end.push_back(new_pos);
                    }
                }
                frontiers[i].pop_front();
            }
        }
        //evaluate all positions changed this round
        int num_pos_to_eval = (int)eval_at_end.size();
        for (int i = 0; i < num_pos_to_eval; i++){
            Coord cur_pos = eval_at_end.front();
            int value_at_pos = result[cur_pos.second * state->w + cur_pos.first];
            int snake_id = value_at_pos % state->num_snakes;
            if (value_at_pos >= 2*state->num_snakes){
                // all snakes entering this field would have died, so it is free to claim again
                result[cur_pos.second * state->w + cur_pos.first] = -1;
            } else if (value_at_pos >= state->num_snakes){
                // one snake successfully claimed this field
                result[cur_pos.second * state->w + cur_pos.first] = snake_id;
                frontiers[inverse_snake_ids[snake_id]].push_back(cur_pos);
                // maybe update food position if we did not already find a better food
                if (food_dist_arr[snake_id] > counter){
                    for (Coord f: state->food){
                        if (f == cur_pos){
                            food_dist_arr[snake_id] = counter;
                            reached_food[snake_id] = true;
                        }
                    }
                }
                // maybe update tail distance if we did not already find a better path
                if (tail_dist_arr[snake_id] > counter){
                    if (state->snakes.at(snake_id)->body.back() == cur_pos){
                        tail_dist_arr[snake_id] = counter;
                        reached_tail[snake_id] = true;
                    }

                }

            }
            eval_at_end.pop_front();
        }
        counter++;
    }
    //count tiles on result board

    for (int x = 0; x < state->w; x++){
        for (int y = 0; y < state->h; y++){
            int val = result[y * state->w + x];
            // check if the field is claimed by snake
            if (val >= 0 and val < state->num_snakes){
                //food
                bool continue_flag = false;
                for (Coord f: state->food) {
                    if (f.first == x and f.second == y) {
                        //food in hazards
                        if (state->hazards[y * state->w + x]){
                            area_arr[val] += food_in_hazard_weight;
                            continue_flag = true;
                            break;
                        } else {
                            area_arr[val] += food_weight;
                            continue_flag = true;
                            break;
                        }
                    }
                }
                if (continue_flag) continue;
                //hazards
                if (state->hazards[y * state->w + x]){
                    area_arr[val] += hazard_weight;
                    continue;
                }
                // normal tile
                area_arr[val] += weight;
            }
        }
    }
}

void hazards(GameState* state, bool* arr){
    for (int i = 0; i < state->w * state->h; i++){
        arr[i] = state->hazards[i];
    }
}

void char_game_matrix(GameState* state, char* matrix){
    char snake_counter = 2;
    for(const auto& s: state->snakes){
        if (not s->alive) continue;
        deque<Coord> cur_body = s->body;
        for (Coord c: cur_body){
            matrix[c.second * state->w + c.first] = 1;
        }
        Coord head = cur_body[0];
        matrix[head.second * state->w + head.first] = snake_counter;
        snake_counter += 1;
    }
}


