from dataclasses import dataclass, field

from src.game.game import GameConfig


@dataclass
class OvercookedRewardConfig:
    placement_in_pot: float
    dish_pickup: float
    soup_pickup: float
    soup_delivery: float
    start_cooking: float
    
    def single_cooking_reward(self):
        return self.placement_in_pot * 3 + self.dish_pickup + self.start_cooking + self.soup_delivery


@dataclass(kw_only=True)
class OvercookedGameConfig(GameConfig):
    board: list[list[int]]
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]]  # x, y, orientation, held_item
    w: int
    h: int
    reward_scaling_factor: float = field(default=1)
    num_actions: int = field(default=6)
    num_players: int = field(default=2)
    reward_cfg: OvercookedRewardConfig = field(default_factory =lambda: OvercookedRewardConfig(
        placement_in_pot=3,
        dish_pickup=3,
        soup_pickup=5,
        soup_delivery=20,
        start_cooking=3,
    ))
    horizon: int = 400
    temperature_input: bool = False
    single_temperature_input: bool = True
    flat_obs: bool = False
    cooking_time: int = 20
    unstuck_behavior: bool = False


# const int EMPTY_TILE = 0;
# const int COUNTER_TILE = 1;
# const int DISH_TILE = 2;
# const int ONION_TILE = 3;
# const int POT_TILE = 4;
# const int SERVING_TILE = 5;

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
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((1, 2, 0, 0), (3, 1, 0, 0)))


@dataclass
class AsymmetricAdvantageOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=9)
    h: int = field(default=5)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [3, 0, 1, 5, 1, 3, 1, 0, 5],
        [1, 0, 0, 0, 4, 0, 0, 0, 1],
        [1, 0, 0, 0, 4, 0, 0, 0, 1],
        [1, 1, 1, 2, 1, 2, 1, 1, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((6, 2, 0, 0), (1, 3, 0, 0)))


@dataclass
class CoordinationRingOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=5)
    h: int = field(default=5)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 1, 4, 1],
        [1, 0, 0, 0, 4],
        [2, 0, 1, 0, 1],
        [3, 0, 0, 0, 1],
        [1, 3, 5, 1, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((2, 1, 0, 0), (1, 2, 0, 0)))


@dataclass
class ForcedCoordinationOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=5)
    h: int = field(default=5)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 1, 4, 1],
        [3, 0, 1, 0, 4],
        [3, 0, 1, 0, 1],
        [2, 0, 1, 0, 1],
        [1, 1, 1, 5, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((3, 1, 0, 0), (1, 2, 0, 0)))


@dataclass
class CounterCircuitOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=8)
    h: int = field(default=5)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 1, 4, 4, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [2, 0, 1, 1, 1, 1, 0, 5],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 3, 3, 1, 1, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((3, 3, 0, 0), (3, 1, 0, 0)))


@dataclass
class OneStateCrampedRoomOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=5)
    h: int = field(default=4)
    horizon: int = field(default=1)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 4, 1, 1],
        [3, 0, 0, 0, 3],
        [1, 0, 0, 0, 1],
        [1, 2, 1, 5, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((1, 2, 0, 0), (2, 1, 0, 1)))
    reward_scaling_factor: float = field(default=0.5)


@dataclass
class TwoStateCrampedRoomOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=5)
    h: int = field(default=4)
    horizon: int = field(default=2)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 4, 1, 1],
        [3, 0, 0, 0, 3],
        [1, 0, 0, 0, 1],
        [1, 2, 1, 5, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((1, 2, 0, 0), (2, 2, 0, 1)))
    reward_scaling_factor: float = field(default=0.5)


@dataclass
class SimpleCrampedRoomOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=5)
    h: int = field(default=4)
    horizon: int = field(default=2)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 4, 1, 1],
        [3, 0, 0, 0, 3],
        [1, 0, 0, 0, 1],
        [1, 2, 1, 5, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((2, 1, 0, 0), (2, 2, 0, 1)))
    reward_scaling_factor: float = field(default=0.5)


@dataclass
class Simple2CrampedRoomOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=5)
    h: int = field(default=4)
    horizon: int = field(default=6)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 4, 1, 1],
        [3, 0, 0, 0, 3],
        [1, 0, 0, 0, 1],
        [1, 2, 1, 5, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((1, 2, 0, 0), (3, 1, 0, 0)))
    reward_scaling_factor: float = field(default=0.5)
    
@dataclass
class Simple3CrampedRoomOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=5)
    h: int = field(default=4)
    horizon: int = field(default=15)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 4, 1, 1],
        [3, 0, 0, 0, 3],
        [1, 0, 0, 0, 1],
        [1, 2, 1, 5, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((1, 2, 0, 0), (3, 1, 0, 0)))
    reward_scaling_factor: float = field(default=0.5)

@dataclass
class Simple4CrampedRoomOvercookedConfig(OvercookedGameConfig):
    w: int = field(default=5)
    h: int = field(default=4)
    horizon: int = field(default=50)
    board: list[list[int]] = field(default_factory=lambda: [
        [1, 1, 4, 1, 1],
        [3, 0, 0, 0, 3],
        [1, 0, 0, 0, 1],
        [1, 2, 1, 5, 1],
    ])
    start_pos: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = field(default_factory=lambda: ((1, 2, 0, 0), (3, 1, 0, 0)))
    reward_scaling_factor: float = field(default=0.5)
