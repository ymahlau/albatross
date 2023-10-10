//
// Created by mahla on 26/10/2022.
//
#include <utility>
#include <random>

#include "../header/battlesnake.h"
#include "../header/battlesnake_helper.h"

using namespace std;

random_device rd_utils_gym;
mt19937 gen_utils_gym(rd_utils_gym());

void set_seed_gym(int seed){
    gen_utils_gym.seed(seed);
}

Snake::Snake(
        int id,
        bool alive,
        int health,
        int length,
        int w,
        int h,
        int max_health,
        deque<Coord> spawn_pos
):
        id(id),
        alive(alive),
        health(health),
        length(length),
        max_health(max_health),
        body(std::move(spawn_pos)),
        board(w*h, false)
{
    for (Coord& c: body){
        board[c.second * w + c.first] = true;
    }
}

Snake::Snake(const Snake& other):
    body(other.body),
    board(other.board)
{
    id = other.id;
    alive = other.alive;
    health = other.health;
    length = other.length;
    max_health = other.max_health;
}

Snake::~Snake(){
    body.clear();
    board.clear();
}

Snake& Snake::operator=(const Snake& other){
    id = other.id;
    alive = other.alive;
    health = other.health;
    length = other.length;
    max_health = other.max_health;
    body.clear();
    body = other.body;
    board.clear();
    board = other.board;
    return *this;
}

GameState::GameState(
        int w,
        int h,
        int num_snakes,
        int min_food,
        int food_spawn_chance,
        int init_turns_played,
        deque<deque<Coord>> snake_bodies,
        list<Coord> food_spawns,
        bool* snake_alive,
        int* snake_health,
        int* snake_len,
        int* max_health,
        bool wrapped,
        bool royale,
        int shrink_n_turns,
        int hazard_damage,
        vector<bool> init_hazards
):
        turn(init_turns_played),
        w(w),
        h(h),
        num_snakes(num_snakes),
        min_food(min_food),
        food_spawn_chance(food_spawn_chance),
        food(std::move(food_spawns)),
        wrapped(wrapped),
        royale(royale),
        shrink_n_turns(shrink_n_turns),
        hazard_damage(hazard_damage),
        hazards(std::move(init_hazards))
{
    //init snakes
    for (int i = 0; i < num_snakes; i++){
        auto* s = new Snake(i, snake_alive[i], snake_health[i], snake_len[i], w, h, max_health[i], snake_bodies[i]);
        snakes.push_back(s);
    }
}

GameState::GameState(const GameState& other):
    food(other.food),
    hazards(other.hazards)
{
    turn = other.turn;
    w = other.w;
    h = other.h;
    num_snakes = other.num_snakes;
    min_food = other.min_food;
    food_spawn_chance = other.food_spawn_chance;
    for (auto& sp: other.snakes){
        auto* s = new Snake(*sp);
        snakes.push_back(s);
    }
    wrapped = other.wrapped;
    royale = other.royale;
    shrink_n_turns = other.shrink_n_turns;
    hazard_damage = other.hazard_damage;
}

GameState::~GameState(){
    food.clear();
    while(not snakes.empty()){
        Snake* s = snakes.front();
        delete s;
        snakes.pop_front();
    }
    snakes.clear();
    hazards.clear();
}

GameState& GameState::operator=(const GameState& other){
    turn = other.turn;
    w = other.w;
    h = other.h;
    num_snakes = other.num_snakes;
    min_food = other.min_food;
    food_spawn_chance = other.food_spawn_chance;
    food.clear();
    food = other.food;
    deque<Snake*> temp_q;
    for (Snake* s : other.snakes){
        auto* temp_s = new Snake(*s);
        temp_q.push_back(temp_s);
    }
    while(not snakes.empty()){
        Snake* s = snakes.front();
        delete s;
        snakes.pop_front();
    }
    snakes.clear();
    snakes = temp_q;
    wrapped = other.wrapped;
    shrink_n_turns = other.shrink_n_turns;
    hazard_damage = other.hazard_damage;
    hazards = other.hazards;
    return *this;
}

GameState* init(
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
        int num_init_food,  // if this is -1, then spawn randomly
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
    // snake spawns
    deque<deque<Coord>> snake_qs;
    if (spawn_snakes_randomly){
        deque<Coord> heads = spawn_randomly(w, num_snakes);
        for (int i = 0; i < num_snakes; i++){
            deque<Coord> q;
            q.push_back(heads[i]);
            snake_qs.push_back(q);
        }
    } else {
        snake_qs = spawn_on_pos(snake_bodies, snake_body_lengths, num_snakes, max_body_length);
    }
    //food
    list<Coord> food_list;
    if (num_init_food == -2){
        food_list.resize(0);
    } else if (num_init_food == -1){
        vector<Coord> snake_heads;
        for (int i = 0; i < num_snakes; i++){
            snake_heads.push_back(snake_qs[i].front());
        }
        food_list = initialize_food(w, num_snakes, snake_heads);
    } else {
        food_list = food_on_pos(food_spawns, num_init_food);
    }
    //hazards
    vector<bool> init_hazard_vec(w*h, false);
    for (int i = 0; i < w*h; i++){
        init_hazard_vec[i] = init_hazards[i];
    }

    auto* state_p = new GameState(w, h, num_snakes, min_food, food_spawn_chance, init_turns_played,
                                  snake_qs, food_list, snake_alive, snake_health, snake_len, max_health, wrapped,
                                  royale, shrink_n_turns, hazard_damage, init_hazard_vec);
    return state_p;
}


GameState* clone(GameState* state){
    auto* copy = new GameState(*state);
    return copy;
}

void close(GameState* state){
    delete state;
}

list<Coord> move_snakes(GameState* state, const int* actions){
    list<Coord> food_to_delete;
    //iterate through live snakes
    for (Snake* s : state->snakes){
        if(not s->alive) continue;
        //move new head
        Coord old_pos = s->body.front();
        Coord new_pos = new_position(old_pos, actions[s->id], state);
        s->body.push_front(new_pos);

        //remove tail if no extension happening due to
        //1. no start of game, 2. no food consumed
        if (s->length < (int) s->body.size()){
            Coord tail = s->body.back();
            s->board[tail.second * state->w + tail.first] = false;
            s->body.pop_back();
        }
        //decrease health (normal and hazard damage)
        s->health--;
        if (state->royale and state->hazards[new_pos.second * state->w + new_pos.first]){
            s->health -= state->hazard_damage;
        }
        if (s->health < 0) s->health = 0;
        //food consumption
        for (Coord f: state->food){
            if (f == new_pos){
                s->health = s->max_health;
                s->length++;
                food_to_delete.push_back(f);
            }
        }
    }
    return food_to_delete;
}

list<int> calculate_deaths(GameState* state){
    list<int> snakes_to_kill;
    //calculate deaths
    for (auto s : state->snakes){
        if (!s->alive) continue;  // What is dead may never die
        Coord head = s->body.front();
        // 1. death by out of bounds
        // 2. death by starvation
        if ((not in_bounds(state, head)) or s->health <= 0) {
            snakes_to_kill.push_back(s->id);
            continue;
        }
        // 3. self collision
        if (s->board[head.second * state->w + head.first]){
            snakes_to_kill.push_back(s->id);
            continue;
        }
        // 4. collision with other snake
        for (auto other : state->snakes) {
            if(other->id == s->id) continue;  //we do not want to check own body again
            if(not other->alive) continue;  // What is dead may also never kill
            // plain body collision with other snake
            if(other->board[head.second * state->w + head.first]){
                snakes_to_kill.push_back(s->id);
                break;
            }
            //check if heads meet in the middle and current snake lost to other
            if(head == other->body.front() and s->length <= other->length){
                snakes_to_kill.push_back(s->id);
                break;
            }
        }
    }
    return snakes_to_kill;
}

void maybe_update_hazards(GameState* state){
    if (not state->royale) return;
    if (state->turn < state->shrink_n_turns) return;
    if (state->turn % state->shrink_n_turns != 0) return;
    //determine current boundaries of hazards
    int min_x = state->w;
    int min_y = state->w;
    int max_x = -1;
    int max_y = -1;
    for (int x = 0; x < state->w; x++){
        for (int y = 0; y < state->h; y++){
            if (not state->hazards[y * state->w + x]){
                if (x > max_x) max_x = x;
                if (x < min_x) min_x = x;
                if (y > max_y) max_y = y;
                if (y < min_y) min_y = y;
            }
        }
    }
    //determine direction for shrinking and do it
    int rng = abs((int) gen_utils_gym()) % 4;
    if (rng == 0){
        for (int y = 0; y < state->h; y++){
            state->hazards[y * state->w + min_x] = true;
        }
    } else if (rng == 1){
        for (int y = 0; y < state->h; y++){
            state->hazards[y * state->w + max_x] = true;
        }
    } else if (rng == 2){
        for (int x = 0; x < state->w; x++){
            state->hazards[min_y * state->w + x] = true;
        }
    } else {
        for (int x = 0; x < state->w; x++){
            state->hazards[max_y * state->w + x] = true;
        }
    }
}

void step(GameState* state, int* actions){
    /**
     * actions have to be ordered by the id of the snakes
     * See https://docs.battlesnake.com/guides/game/rules
     */
    //move snakes and apply damage
    list<Coord> food_to_delete = move_snakes(state, actions);
    // remove food from board
    for (Coord f: food_to_delete){
        state->food.remove(f);
    }
    //place new food
    place_food_randomly(state);
    //calculate deaths
    list<int> snakes_to_kill = calculate_deaths(state);
    //kill snakes
    for (int id: snakes_to_kill){
        state->snakes.at(id)->alive = false;
    }
    //draw new heads of live snakes
    for (auto &s : state->snakes){
        if (s->alive){
            Coord head = s->body.front();
            s->board[head.second * state->w + head.first] = true;
        }
    }
    //increase turn counter
    state->turn += 1;
    //draw new hazards
    maybe_update_hazards(state);
}


void legal_actions(GameState* state, int snake_id, int* actions){
    Coord head = state->snakes.at(snake_id)->body.front();
    for(int i = 0; i < 4; i++){
        Coord new_head = new_position(head, i, state);
        //check for out of bounds
        if(not in_bounds(state, new_head)){
            actions[i] = 0;
            continue;
        }
        //body collisions
        bool collision = false;
        for (auto &s : state->snakes){
            if(not s->alive) continue;
            if(s->board[new_head.second * state->w + new_head.first]){
                //check if tail would move out of the way next round
                if(s->body.back() != new_head or s->length > (int)(s->body.size())){
                    actions[i] = 0;
                    collision = true;
                    break;
                }
            }
        }
        //check if you would die out of health
        bool is_food = false;
        for (Coord f: state->food){  // if there is food, you cannot die out of health
            if (f == new_head){
                is_food = true;
                break;
            }
        }
        //if the snake has only one health and does not consume food, it dies
        int cur_health = state->snakes.at(snake_id)->health;
        if (not is_food and cur_health == 1){
            actions[i] = 0;
            continue;
        }
        //if there is a hazard, check for death by damage
        if (not is_food and state->hazards[new_head.second * state->w + new_head.first]
                and cur_health <= state->hazard_damage + 1){
            actions[i] = 0;
            continue;
        }
        if(not collision){
            actions[i] = 1;
        }
    }
}

