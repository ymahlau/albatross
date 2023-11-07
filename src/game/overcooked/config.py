from dataclasses import dataclass, field

from src.game.game import GameConfig


@dataclass
class OvercookedRewardConfig:
    placement_in_pot: float
    dish_pickup: float
    soup_pickup: float
    soup_delivery: float
    start_cooking: float


@dataclass(kw_only=True)
class OvercookedGameConfig(GameConfig):
    board: list[list[int]]
    start_pos: tuple[tuple[int, int, int], tuple[int, int, int]]  # x, y, orientation
    w: int
    h: int
    num_actions: int = field(default=6)
    num_players: int = field(default=2)
    reward_cfg: OvercookedRewardConfig = field(default_factory =lambda: OvercookedRewardConfig(
        placement_in_pot=3,
        dish_pickup=3,
        soup_pickup=5,
        soup_delivery=20,
        start_cooking=0,
    ))
    horizon: int = 400
    temperature_input: bool = False
    single_temperature_input: bool = True
    flat_obs: bool = False
    cooking_time: int = 20


"""
const int EMPTY_TILE = 0;
const int COUNTER_TILE = 1;
const int DISH_TILE = 2;
const int ONION_TILE = 3;
const int POT_TILE = 4;
const int SERVING_TILE = 5;
"""


@dataclass
class CrampedRoomOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=5)
    h: int = field(default=4)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 4, 1, 1],
        [3, 0, 0, 0, 3],
        [1, 0, 0, 0, 1],
        [1, 2, 1, 5, 1],
    ])
    start_pos: tuple[tuple[int, int, int], tuple[int, int, int]] = field(default_factory=lambda: ((1, 2, 0), (3, 1, 0)))
