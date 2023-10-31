from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.battlesnake_enc import SimpleBattleSnakeEncodingConfig, VanillaBattleSnakeEncodingConfig, \
    SimpleConstrictorEncodingConfig
from src.game.battlesnake.battlesnake_rewards import StandardBattleSnakeRewardConfig, KillBattleSnakeRewardConfig


def perform_choke_11x11(centered: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.include_current_food = False
    ec.centered = centered
    ec.compress_enemies = False
    gc = BattleSnakeConfig(
        w=11,
        h=11,
        num_players=2,
        min_food=0,
        food_spawn_chance=0,
        ec=ec,
        init_snake_pos={0: [[0, 1]], 1: [[0, 0]]},
        init_food_pos=[],
        init_snake_len=[3, 3],
        all_actions_legal=False,
        reward_cfg=StandardBattleSnakeRewardConfig(),
    )
    return gc

def survive_on_11x11() -> BattleSnakeConfig:
    ec = VanillaBattleSnakeEncodingConfig()
    ec.centered = True
    ec.compress_enemies = False
    gc = BattleSnakeConfig(
        w=11,
        h=11,
        num_players=2,
        min_food=1,
        food_spawn_chance=15,
        ec=ec,
        init_snake_len=[3, 3],
        all_actions_legal=False,
        reward_cfg=StandardBattleSnakeRewardConfig(),
    )
    return gc

def survive_on_11x11_4_player(centered: bool) -> BattleSnakeConfig:
    ec = VanillaBattleSnakeEncodingConfig()
    ec.centered = centered
    ec.compress_enemies = True
    gc = BattleSnakeConfig(
        w=11,
        h=11,
        num_players=4,
        min_food=1,
        food_spawn_chance=15,
        ec=ec,
        init_snake_len=[3, 3, 3, 3],
        all_actions_legal=False,
        reward_cfg=KillBattleSnakeRewardConfig(),
    )
    return gc


def survive_on_11x11_4_player_constrictor() -> BattleSnakeConfig:
    ec = SimpleConstrictorEncodingConfig()
    ec.compress_enemies = True
    gc = BattleSnakeConfig(
        w=11,
        h=11,
        num_players=4,
        ec=ec,
        all_actions_legal=False,
        reward_cfg=KillBattleSnakeRewardConfig(),
        constrictor=True,
    )
    return gc


def survive_on_11x11_4_player_wrapped_royale(centered: bool) -> BattleSnakeConfig:
    ec = SimpleBattleSnakeEncodingConfig()
    ec.centered = centered
    ec.include_board = False
    ec.compress_enemies = True
    gc = BattleSnakeConfig(
        w=11,
        h=11,
        royale=True,
        wrapped=True,
        num_players=4,
        min_food=1,
        food_spawn_chance=15,
        ec=ec,
        init_snake_len=[3, 3, 3, 3],
        all_actions_legal=False,
        reward_cfg=KillBattleSnakeRewardConfig(),
    )
    return gc


def survive_on_11x11_4_player_royale() -> BattleSnakeConfig:
    ec = VanillaBattleSnakeEncodingConfig()
    ec.compress_enemies = True
    ec.include_hazards = True
    gc = BattleSnakeConfig(
        w=11,
        h=11,
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

