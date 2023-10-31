from dataclasses import dataclass
from typing import Optional


@dataclass
class BattleSnakeState:
    snakes_alive: list[bool]
    snake_pos: dict[int, list[list[int]]]  # includes dead snakes
    food_pos: list[list[int]]
    snake_health: list[int]  # includes dead snakes
    snake_len: list[int]  # includes dead snakes
