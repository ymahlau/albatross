from tkinter import Tk, Frame, Label, StringVar
from typing import Optional

from src.analysis.core import GUIState
from src.analysis.style import FRAME_PADDING, FONT_NAME, FONT_SIZE, COLOR_LIST
from src.game.battlesnake.battlesnake import BattleSnakeGame


class InfoFrame:
    def __init__(self, root: Tk, state: GUIState):
        self.state = state
        self.root = root
        self.f = Frame(root, padx=FRAME_PADDING, pady=FRAME_PADDING)
        font = (FONT_NAME, FONT_SIZE)

        # number of turns
        self.turn_label = Label(self.f, text='Turns:', font=font)
        self.turn_label.grid(row=0, column=0)
        self.turn_str_var = StringVar(self.root, value='-1')
        self.turn_var = Label(self.f, textvariable=self.turn_str_var, font=font)
        self.turn_var.grid(row=0, column=1)

        # buffer index
        self.buffer_label = Label(self.f, text='Buffer idx:', font=font)
        self.buffer_label.grid(row=0, column=2)
        self.buffer_str_var = StringVar(self.root, value='-1')
        self.buffer_var = Label(self.f, textvariable=self.buffer_str_var, font=font)
        self.buffer_var.grid(row=0, column=3)

        # player info stats
        self.health_label = Label(self.f, text='Health', font=font)
        self.health_label.grid(row=1, column=1)
        self.length_label = Label(self.f, text='Length', font=font)
        self.length_label.grid(row=1, column=2)
        self.area_label = Label(self.f, text='Area', font=font)
        self.area_label.grid(row=1, column=3)
        self.health_label_list, self.length_label_list, self.area_label_list = [], [], []
        self.area_var_list, self.length_var_list, self.health_var_list = [], [], []
        for player in range(self.state.num_players):
            Label(self.f, text=f"P{player}", font=font, fg=COLOR_LIST[player]).grid(row=player+2, column=0)
            # health
            health_str = StringVar(self.f, value='-1')
            health_label = Label(self.f, textvariable=health_str, font=font, fg=COLOR_LIST[player])
            health_label.grid(row=player+2, column=1)
            self.health_var_list.append(health_str)
            self.health_label_list.append(health_label)
            # length
            length_str = StringVar(self.f, value='-1')
            length_label = Label(self.f, textvariable=length_str, font=font, fg=COLOR_LIST[player])
            length_label.grid(row=player+2, column=2)
            self.length_var_list.append(length_str)
            self.length_label_list.append(length_label)
            # area
            area_str = StringVar(self.f, value='-1')
            area_label = Label(self.f, textvariable=area_str, font=font, fg=COLOR_LIST[player])
            area_label.grid(row=player+2, column=3)
            self.area_var_list.append(area_str)
            self.area_label_list.append(area_label)

    def update(self, game: BattleSnakeGame, buffer_idx: Optional[int]):
        self.turn_str_var.set(f"{game.turns_played}")
        ac_arr = game.area_control()["area_control"]
        for player in range(game.num_players):
            self.health_var_list[player].set(f"{game.player_healths()[player]}")
            self.length_var_list[player].set(f"{game.player_lengths()[player]}")
            self.area_var_list[player].set(f"{int(ac_arr[player])}")
        # buffer
        self.buffer_str_var.set(f"-")
        if buffer_idx is not None:
            self.buffer_str_var.set(f"{buffer_idx}")
