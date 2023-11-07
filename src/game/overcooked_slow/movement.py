from src.game.overcooked_slow.layouts import CrampedRoomOvercookedSlowConfig
from src.game.overcooked_slow.overcooked import OvercookedGame


def cramped_at_cook_start() -> OvercookedGame:
    game_cfg = CrampedRoomOvercookedSlowConfig()
    game = OvercookedGame(game_cfg)

    game.step((0, 2))
    game.step((3, 5))
    game.step((5, 3))
    game.step((4, 0))
    game.step((4, 5))
    game.step((2, 2))
    game.step((0, 5))
    game.step((5, 4))
    game.step((3, 3))
    game.step((5, 0))
    game.step((4, 5))
    
    return game


def cramped_before_cook_soup_ready() -> OvercookedGame:
    game_cfg = CrampedRoomOvercookedSlowConfig()
    game = OvercookedGame(game_cfg)

    game.step((0, 2))
    game.step((3, 5))
    game.step((5, 3))
    game.step((4, 0))
    game.step((4, 5))
    game.step((2, 2))
    game.step((0, 5))
    game.step((5, 4))
    game.step((3, 3))
    game.step((5, 0))
    game.step((4, 5))
    game.step((4, 5))

    for _ in range(13):
        reward, _, _ = game.step((4, 4))

    game.step((4, 1))
    game.step((4, 3))
    game.step((4, 1))
    game.step((4, 5))
    game.step((4, 2))

    return game

def cramped_at_cook_soup_ready() -> OvercookedGame:
    game = cramped_before_cook_soup_ready()
    game.step((4, 0))
    return game
