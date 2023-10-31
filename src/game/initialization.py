from typing import Optional

from src.game import GameConfig, Game, GameType
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import bs_config_from_structured
from src.game.exploitability.random_exploit_game import RandomExploitGame, RandomExploitGameConfig
from src.game.extensive_form.extensive_form import ExtensiveFormGame, ExtensiveFormConfig
from src.game.normal_form.normal_form import NormalFormConfig, NormalFormGame
from src.game.oshi_zumo.oshi_zumo import OshiZumoGame, OshiZumoConfig
from src.game.overcooked.overcooked import OvercookedGame, OvercookedConfig
from src.misc.replay_buffer import ReplayBufferConfig


def get_game_from_config(game_cfg: GameConfig) -> Game:
    if game_cfg.game_type == GameType.BATTLESNAKE or game_cfg.game_type == GameType.BATTLESNAKE.value:
        return BattleSnakeGame(game_cfg)
    elif game_cfg.game_type == GameType.OSHI_ZUMO or game_cfg.game_type == GameType.OSHI_ZUMO.value:
        return OshiZumoGame(game_cfg)
    elif game_cfg.game_type == GameType.NORMAL_FORM or game_cfg.game_type == GameType.NORMAL_FORM.value:
        return NormalFormGame(game_cfg)
    elif game_cfg.game_type == GameType.EXTENSIVE_FORM or game_cfg.game_type == GameType.EXTENSIVE_FORM.value:
        return ExtensiveFormGame(game_cfg)
    elif game_cfg.game_type == GameType.OVERCOOKED or game_cfg.game_type == GameType.OVERCOOKED.value:
        return OvercookedGame(game_cfg)
    elif game_cfg.game_type == GameType.EXPLOIT_RANDOM or game_cfg.game_type == GameType.EXPLOIT_RANDOM.value:
        return RandomExploitGame(game_cfg)
    else:
        raise ValueError(f"Unknown game type: {game_cfg}")


def game_config_from_structured(cfg) -> Optional[GameConfig]:
    if cfg is None:
        return None
    if cfg.game_type == GameType.BATTLESNAKE or cfg.game_type == GameType.BATTLESNAKE.value:
        return bs_config_from_structured(cfg)
    elif cfg.game_type == GameType.OSHI_ZUMO or cfg.game_type == GameType.OSHI_ZUMO.value:
        return OshiZumoConfig(**cfg)
    elif cfg.game_type == GameType.NORMAL_FORM or cfg.game_type == GameType.NORMAL_FORM.value:
        kwargs = dict(cfg)
        new_ja_dict = {}
        for k, v in cfg.ja_dict.items():
            k_new = tuple([a for a in k])
            v_new = tuple([val for val in v])
            new_ja_dict[k_new] = v_new
        kwargs['ja_dict'] = new_ja_dict
        return NormalFormConfig(**kwargs)
    elif cfg.game_type == GameType.EXTENSIVE_FORM or cfg.game_type == GameType.EXTENSIVE_FORM.value:
        kwargs = dict(cfg)
        new_ja_dict = {}
        for k, v in cfg.ja_dict.items():
            k_new = tuple([tuple([a for a in k_i]) for k_i in k])
            v_new = tuple([val for val in v])
            new_ja_dict[k_new] = v_new
        kwargs['ja_dict'] = new_ja_dict
        return ExtensiveFormConfig(**kwargs)
    elif cfg.game_type == GameType.OVERCOOKED or cfg.game_type == GameType.OVERCOOKED.value:
        return OvercookedConfig(**cfg)
    elif cfg.game_type == GameType.EXPLOIT_RANDOM or cfg.game_type == GameType.EXPLOIT_RANDOM.value:
        kwargs = dict(cfg)
        kwargs['og_game_cfg'] = game_config_from_structured(cfg.og_game_cfg)
        return RandomExploitGameConfig(**cfg)
    else:
        raise ValueError(f"Unknown game type: {cfg}")

def buffer_config_from_game(
        game: Game,
        capacity: int,
        single_temperature: bool,
) -> ReplayBufferConfig:
    buffer_cfg = ReplayBufferConfig(
        obs_shape=game.get_obs_shape(),
        num_actions=game.num_actions,
        num_players=game.num_players,
        num_symmetries=game.get_symmetry_count(),
        capacity=capacity,
        single_temperature=single_temperature,
    )
    return buffer_cfg
