from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.game import GameConfig, Game
from src.game.overcooked_slow.overcooked import OvercookedConfig, OvercookedGame


def get_game_from_config(game_cfg: GameConfig) -> Game:
    if isinstance(game_cfg, BattleSnakeConfig):
        return BattleSnakeGame(game_cfg)
    elif isinstance(game_cfg, OvercookedConfig):
        return OvercookedGame(game_cfg)
    # elif game_cfg.game_type == GameType.OSHI_ZUMO or game_cfg.game_type == GameType.OSHI_ZUMO.value:
    #     return OshiZumoGame(game_cfg)
    # elif game_cfg.game_type == GameType.NORMAL_FORM or game_cfg.game_type == GameType.NORMAL_FORM.value:
    #     return NormalFormGame(game_cfg)
    # elif game_cfg.game_type == GameType.EXTENSIVE_FORM or game_cfg.game_type == GameType.EXTENSIVE_FORM.value:
    #     return ExtensiveFormGame(game_cfg)

    # elif game_cfg.game_type == GameType.EXPLOIT_RANDOM or game_cfg.game_type == GameType.EXPLOIT_RANDOM.value:
    #     return RandomExploitGame(game_cfg)
    else:
        raise ValueError(f"Unknown game type: {game_cfg}")


# def buffer_config_from_game(
#         game: Game,
#         capacity: int,
#         single_temperature: bool,
# ) -> ReplayBufferConfig:
#     buffer_cfg = ReplayBufferConfig(
#         obs_shape=game.get_obs_shape(),
#         num_actions=game.num_actions,
#         num_players=game.num_players,
#         num_symmetries=game.get_symmetry_count(),
#         capacity=capacity,
#         single_temperature=single_temperature,
#     )
#     return buffer_cfg
