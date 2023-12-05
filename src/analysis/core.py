from dataclasses import dataclass
from tkinter import IntVar, Label
from typing import Optional


class IndexedLabel(Label):
    def __init__(self, x: int, y: int, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y

@dataclass
class GUIState:
    min_food: int
    w: int
    h: int
    num_players: int
    player_value: IntVar
    tool_value: IntVar
    action_values: list[IntVar]
    matrix_radio_var: IntVar
    buffer_idx: Optional[int]
    last_buffer_idx: Optional[int]

@dataclass
class GameState:
    turns_played: int
    snakes_alive: list[bool]
    snake_pos: dict[int, list[list[int]]]
    food_pos: list[list[int]]
    snake_health: list[int]
    snake_len: list[int]
