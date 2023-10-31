from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.battlesnake_enc import SimpleBattleSnakeEncodingConfig, BestConstrictorEncodingConfig, \
    SimpleConstrictorEncodingConfig, VanillaBattleSnakeEncodingConfig
from src.game.battlesnake.battlesnake_rewards import KillBattleSnakeRewardConfig, CooperationBattleSnakeRewardConfig


def survive_on_7x7() -> BattleSnakeConfig:
    ec = VanillaBattleSnakeEncodingConfig()
    ec.compress_enemies = False
    gc = BattleSnakeConfig(
        w=7,
        h=7,
        num_players=2,
        init_food_pos=None,
        min_food=1,
        food_spawn_chance=15,
        ec=ec,
        init_snake_len=[3, 3],
        all_actions_legal=False,
    )
    return gc

def survive_on_7x7_4_player() -> BattleSnakeConfig:
    ec = VanillaBattleSnakeEncodingConfig()
    ec.compress_enemies = True
    gc = BattleSnakeConfig(
        w=7,
        h=7,
        num_players=4,
        init_food_pos=None,
        min_food=1,
        food_spawn_chance=15,
        ec=ec,
        init_snake_len=[3, 3, 3, 3],
        all_actions_legal=False,
        reward_cfg=KillBattleSnakeRewardConfig(),
    )
    return gc


def survive_on_7x7_constrictor() -> BattleSnakeConfig:
    ec = SimpleConstrictorEncodingConfig()
    ec.compress_enemies = False
    gc = BattleSnakeConfig(
        w=7,
        h=7,
        num_players=2,
        ec=ec,
        all_actions_legal=False,
        constrictor=True,
    )
    return gc


def survive_on_7x7_constrictor_4_player() -> BattleSnakeConfig:
    ec = SimpleConstrictorEncodingConfig()
    ec.compress_enemies = True
    gc = BattleSnakeConfig(
        w=7,
        h=7,
        num_players=4,
        ec=ec,
        all_actions_legal=False,
        constrictor=True,
        reward_cfg=KillBattleSnakeRewardConfig(),
    )
    return gc

def cooperation_7x7() -> BattleSnakeConfig:
    ec = SimpleConstrictorEncodingConfig()
    gc = BattleSnakeConfig(
        ec=ec,
        w=7,
        h=7,
        num_players=2,
        all_actions_legal=False,
        constrictor=True,
        reward_cfg=CooperationBattleSnakeRewardConfig(),
    )
    return gc


def survive_on_7x7_4_player_royale() -> BattleSnakeConfig:
    ec = VanillaBattleSnakeEncodingConfig()
    ec.compress_enemies = True
    ec.include_hazards = True
    gc = BattleSnakeConfig(
        w=7,
        h=7,
        royale=True,
        wrapped=False,
        num_players=4,
        min_food=1,
        food_spawn_chance=15,
        ec=ec,
        init_snake_len=[3, 3, 3, 3],
        all_actions_legal=False,
        shrink_n_turns=25,
        hazard_damage=15,
        reward_cfg=KillBattleSnakeRewardConfig(),
    )
    return gc


def title_cfg_4d7() -> BattleSnakeConfig:
    player_positions = {
        0: [
            [3, 4],
            [3, 3],
            [4, 3],
            [4, 4],
            [4, 5],
            [4, 6],
            [3, 6],
            [2, 6],
            [2, 5],
            [1, 5]
        ],
        1: [
            [1, 4],
            [1, 3],
            [1, 2],
            [2, 2],
            [2, 1],
            [2, 0],
            [1, 0],
            [0, 0],
            [0, 1],
            [1, 1],
        ],
        2: [
            [4, 1],
            [4, 2],
            [3, 2],
            [3, 1],
            [3, 0],
            [4, 0],
            [5, 0],
            [6, 0],
            [6, 1],
            [5, 1],
        ],
        3: [
            [5, 2],
            [6, 2],
            [6, 3],
            [5, 3],
            [5, 4],
            [6, 4],
            [6, 5],
            [6, 6],
            [5, 6],
            [5, 5],
        ]
    }
    ec = SimpleConstrictorEncodingConfig()
    game_cfg = BattleSnakeConfig(
        ec=ec,
        w=7,
        h=7,
        num_players=4,
        all_actions_legal=False,
        constrictor=True,
        reward_cfg=CooperationBattleSnakeRewardConfig(),
        init_snake_pos=player_positions,
    )
    return game_cfg
