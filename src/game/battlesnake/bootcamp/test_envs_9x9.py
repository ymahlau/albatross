from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.battlesnake_enc import BestBattleSnakeEncodingConfig


def survive_on_9x9_constrictor_4_player() -> BattleSnakeConfig:
    ec = BestBattleSnakeEncodingConfig()
    ec.include_num_food_on_board = False
    ec.include_current_food = False
    ec.include_next_food = False
    ec.compress_enemies = True
    ec.include_snake_health = False
    ec.include_snake_length = False
    ec.include_food_distance = False
    ec.include_tail_distance = False
    ec.centered = True

    gc = BattleSnakeConfig(
        w=9,
        h=9,
        num_players=4,
        ec=ec,
        all_actions_legal=False,
        constrictor=True,
    )
    return gc
