from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.battlesnake_enc import SimpleBattleSnakeEncodingConfig

# snake 1 should learn to take food and force a win
def two_snakes_one_food(fully_connected: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    init_snake_pos = {0: [[1, 0]], 1: [[2, 2]]}
    init_shake_health = [2, 2]
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=2,
        init_snake_pos=init_snake_pos,
        init_snake_health=init_shake_health,
        init_food_pos=[[0, 0]],
        init_snake_len=[1, 1],
        min_food=0,
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=True,
    )
    return gc


# value function of -1,-1 should be learned
def death_by_starvation_both_die(fully_connected: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    init_snake_pos = {0: [[1, 0]], 1: [[2, 2]]}
    init_shake_health = [1, 1]
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=2,
        init_snake_pos=init_snake_pos,
        init_snake_health=init_shake_health,
        init_food_pos=[],
        min_food=0,
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=True,
    )
    return gc


# value function of -1,1 should be learned
def death_by_starvation_only_1_dies(fully_connected: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    init_snake_pos = {0: [[1, 0]], 1: [[2, 2]]}
    init_shake_health = [1, 2]
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=2,
        init_snake_pos=init_snake_pos,
        init_snake_health=init_shake_health,
        init_food_pos=[],
        min_food=0,
        food_spawn_chance=0,
        init_snake_len=[1, 1],
        ec=ec,
        all_actions_legal=True,
    )
    return gc


# snake 1 should learn to force win
def perform_choke_2_player(fully_connected: bool, centered: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    ec.centered = centered
    init_snake_pos = {0: [[1, 0]], 1: [[2, 0]]}
    init_shake_health = [4, 4]
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=2,
        init_snake_pos=init_snake_pos,
        init_snake_health=init_shake_health,
        init_snake_len=[3, 3],
        init_food_pos=[],
        min_food=0,
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=True,
    )
    return gc


# Snakes should learn to both go up in the first state, to not take any risks
def choose_safe_option1(fully_connected: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    ec.include_snake_health = True
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=2,
        min_food=0,
        init_food_pos=[],
        init_snake_pos={0: [[2, 0]], 1: [[0, 0]]},
        init_snake_health=[2, 2],
        max_snake_health=[2, 2],
        init_snake_len=[0, 0],
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=True
    )
    gc.reward_cfg.living_reward = 1.0
    return gc


# Snake 1 should one of two valid options while snake 2 should go to the right (safe option)
def choose_safe_option2(fully_connected: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    ec.include_snake_health = True
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=2,
        min_food=0,
        init_food_pos=[],
        init_snake_pos={0: [[0, 0]], 1: [[1, 1]]},
        init_snake_health=[2, 2],
        max_snake_health=[2, 2],
        init_snake_len=[0, 0],
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=True
    )
    gc.reward_cfg.living_reward = 1.0
    return gc


# snakes should learn to not run into each other
def avoid_each_other(fully_connected: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    ec.include_snake_health = True
    ec.include_snake_body = False
    ec.include_current_food = False
    # ec.flatten_snakes = False
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=2,
        min_food=0,
        init_snake_pos={0: [[0, 0]], 1: [[2, 2]]},
        init_food_pos=[],
        init_snake_health=[10, 10],
        max_snake_health=[10, 10],
        init_snake_len=[0, 0],
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=False
    )
    gc.reward_cfg.living_reward = 0.5
    return gc


# snake should learn to avoid itself and the wall
def avoid_wall_and_self(fully_connected: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    ec.include_snake_health = True
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=1,
        min_food=0,
        init_food_pos=[],
        init_snake_pos={0: [[0, 0]]},
        init_snake_health=[5],
        max_snake_health=[5],
        init_snake_len=[3],
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=True
    )
    gc.reward_cfg.living_reward = 1.0
    return gc


# for debugging purposes
def map_with_food_single(fully_connected: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    ec.include_snake_health = True
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=1,
        min_food=0,
        init_food_pos=[[0, 1], [1, 1], [0, 2], [2, 2]],
        init_snake_pos={0: [[0, 0]]},
        init_snake_health=[5],
        max_snake_health=[5],
        init_snake_len=[3],
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=True
    )
    gc.reward_cfg.living_reward = 1.0
    return gc


# snake should learn to avoid the wall
def avoid_wall(fully_connected: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    ec.include_snake_health = True
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=1,
        min_food=0,
        init_food_pos=[],
        init_snake_pos={0: [[0, 0]]},
        init_snake_health=[5],
        max_snake_health=[5],
        init_snake_len=[0],
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=True
    )
    gc.reward_cfg.living_reward = 1.0
    return gc


def perform_choke_wrapped(fully_connected: bool, centered: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.flatten = fully_connected
    ec.centered = centered
    ec.include_board = False
    init_snake_pos = {0: [[0, 2], [0, 1], [0, 0], [2, 0], [2, 1], [2, 2]], 1: [[1, 1], [1, 2], [1, 0]]}
    init_shake_health = [6, 5]
    init_snake_len = [6, 3]
    gc = BattleSnakeConfig(
        w=3,
        h=3,
        num_players=2,
        init_snake_pos=init_snake_pos,
        init_snake_health=init_shake_health,
        init_snake_len=init_snake_len,
        init_food_pos=[],
        min_food=0,
        food_spawn_chance=0,
        ec=ec,
        all_actions_legal=True,
        wrapped=True
    )
    return gc


def die_on_a_hill():
    ec = SimpleBattleSnakeEncodingConfig(
        include_current_food=False,
        flatten=False,
        centered=True,
        compress_enemies=False,
    )
    game_cfg = BattleSnakeConfig(
        ec=ec,
        w=3,
        h=3,
        num_players=2,
        init_snake_pos={0: [[0, 0]], 1: [[2, 2]]},
        constrictor=True,
        all_actions_legal=True,
    )
    return game_cfg

