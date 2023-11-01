from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.battlesnake_enc import (SimpleBattleSnakeEncodingConfig, SimpleConstrictorEncodingConfig,
                                                  VanillaBattleSnakeEncodingConfig)
from src.game.battlesnake.battlesnake_rewards import KillBattleSnakeRewardConfig


def perform_choke_5x5(centered: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.centered = centered
    ec.compress_enemies = False
    gc = BattleSnakeConfig(
        w=5,
        h=5,
        num_players=2,
        init_food_pos=[],
        min_food=0,
        food_spawn_chance=0,
        ec=ec,
        init_snake_pos={0: [[0, 1]], 1: [[0, 0]]},
        init_snake_len=[3, 3],
        all_actions_legal=False,
    )
    return gc


def survive_on_5x5() -> BattleSnakeConfig:
    ec = VanillaBattleSnakeEncodingConfig()
    ec.compress_enemies = False
    gc = BattleSnakeConfig(
        w=5,
        h=5,
        num_players=2,
        init_food_pos=None,
        min_food=1,
        food_spawn_chance=15,
        ec=ec,
        init_snake_len=[3, 3],
        all_actions_legal=False,
    )
    return gc


def survive_on_5x5_constrictor() -> BattleSnakeConfig:
    ec = SimpleConstrictorEncodingConfig()
    ec.centered = True
    ec.compress_enemies = False
    gc = BattleSnakeConfig(
        w=5,
        h=5,
        num_players=2,
        ec=ec,
        all_actions_legal=False,
        constrictor=True,
    )
    return gc

def survive_on_5x5_constrictor_4player() -> BattleSnakeConfig:
    ec = SimpleConstrictorEncodingConfig()
    ec.centered = True
    ec.compress_enemies = True
    gc = BattleSnakeConfig(
        w=5,
        h=5,
        num_players=4,
        ec=ec,
        all_actions_legal=False,
        constrictor=True,
        reward_cfg=KillBattleSnakeRewardConfig(),
    )
    return gc


def avoid_each_other_4_snakes_5x5(centered: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.centered = centered
    gc = BattleSnakeConfig(
        w=5,
        h=5,
        num_players=4,
        init_food_pos=[],
        min_food=0,
        food_spawn_chance=0,
        ec=ec,
        init_snake_pos={0: [[0, 0]], 1: [[4, 4]], 2: [[0, 4]], 3: [[4, 0]]},
        init_snake_len=[1, 1, 1, 1],
        init_snake_health=[10, 10, 10, 10],
        all_actions_legal=True,
    )
    gc.reward_cfg.living_reward = 0.5
    return gc


def value_test_5x5_4p(centered: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.centered = centered
    gc = BattleSnakeConfig(
        w=5,
        h=5,
        num_players=4,
        init_snake_health=[1, 1, 2, 1],
        init_food_pos=[],
        min_food=0,
        food_spawn_chance=0,
        ec=ec,
        init_snake_pos={0: [[0, 0]], 1: [[4, 4]], 2: [[2, 2]], 3: [[4, 0]]},
        init_snake_len=[1, 1, 1, 1],
        all_actions_legal=False,
    )
    gc.reward_cfg.living_reward = 0.01
    return gc

def perform_choke_5x5_4_player(centered: bool) -> BattleSnakeConfig:
    ec = SimpleConstrictorEncodingConfig()
    ec.centered = centered

    gc = BattleSnakeConfig(
        w=5,
        h=5,
        num_players=4,
        ec=ec,
        init_snake_pos={0: [[3, 0]], 1: [[2, 0]], 2: [[1, 0]], 3: [[0, 0]]},
        all_actions_legal=True,
        constrictor=True,
    )
    return gc

def unstable_state_duel() -> BattleSnakeConfig:
    gc = BattleSnakeConfig(
        w=5,
        h=5,
        num_players=2,
        init_food_pos=[],
        min_food=0,
        food_spawn_chance=15,
        init_snake_pos={0: [[1, 0], [0, 0], [0, 1]], 1: [[2, 1], [3, 1], [3, 2], [2, 2]]},
        init_snake_health=[100, 100],
        init_snake_len=[3, 4],
        init_snakes_alive=[True, True],
        all_actions_legal=False,
    )
    return gc

def start_state_duel() -> BattleSnakeConfig:
    gc = BattleSnakeConfig(
        w=5,
        h=5,
        num_players=2,
        init_food_pos=[],
        min_food=0,
        food_spawn_chance=15,
        init_snake_pos={0: [[1, 1], [0, 1], [0, 0], [1, 0]], 1: [[3, 3], [3, 4], [4, 4], [4, 3]]},
        init_snake_health=[100, 100],
        init_snake_len=[4, 4],
        init_snakes_alive=[True, True],
        all_actions_legal=False,
    )
    return gc

def cooperation_choke_5x5() -> BattleSnakeConfig:
    ec = SimpleConstrictorEncodingConfig(centered=True)
    gc = BattleSnakeConfig(
        ec=ec,
        w=5,
        h=5,
        num_players=4,
        init_snake_pos={0: [[1, 2], [1, 1], [1, 0]], 1: [[3, 2], [3, 1], [3, 0]],
                        2: [[2, 4]], 3: [[2, 0]]},
        all_actions_legal=False,
        constrictor=True,
    )
    return gc

def randomization() -> BattleSnakeConfig:
    ec = SimpleConstrictorEncodingConfig(centered=True)
    gc = BattleSnakeConfig(
        ec=ec,
        w=5,
        h=5,
        num_players=2,
        init_snake_len=[6, 4],
        init_snake_health=[4, 4],
        init_food_pos=[],
        init_snake_pos={0: [[2, 1], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2]], 1: [[4, 3], [3, 3], [3, 2], [2, 2]]},
        all_actions_legal=True,
        constrictor=False,
        min_food=0,
        food_spawn_chance=0,
    )
    return gc
