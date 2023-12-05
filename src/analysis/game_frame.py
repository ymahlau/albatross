from tkinter import Tk, Frame, StringVar
from typing import Callable

from src.analysis.core import GUIState, IndexedLabel
from src.analysis.style import FONT_NAME, TABLE_FONT_SIZE, FRAME_PADDING, COLOR_LIST
from src.game.battlesnake.battlesnake import BattleSnakeGame


class GameFrame:
    def __init__(
            self,
            root: Tk,
            on_click_callback: Callable,
            state: GUIState,
    ):
        self.root = root
        self.on_click_callback = on_click_callback
        self.state = state

        self.f = Frame(root, padx=FRAME_PADDING, pady=FRAME_PADDING)
        self.table: list[list[IndexedLabel]] = []
        self.string_vars: list[list[StringVar]] = []
        self.init_table()

    def init_table(self):
        # initialize the table displaying game grid
        for x in range(self.state.w):
            var_lst = []
            entry_lst = []
            for y in range(self.state.h):
                string_var = StringVar(value='')
                var_lst.append(string_var)
                entry = IndexedLabel(master=self.f, width=5, height=3, textvariable=string_var,
                                     font=(FONT_NAME, TABLE_FONT_SIZE),
                                     borderwidth=2, relief='groove', x=x, y=y)
                entry.bind("<Button-1>", self.on_click_callback)
                entry.grid(column=x, row=self.state.h - y - 1)
                entry_lst.append(entry)
            self.table.append(entry_lst)
            self.string_vars.append(var_lst)

    def update_table(self, game: BattleSnakeGame):
        # reset all old values
        for x in range(self.state.w):
            for y in range(self.state.h):
                self.string_vars[x][y].set("")
        # set player bodies
        for player in game.players_alive():
            pos_list = game.player_pos(player)
            for i, pos in enumerate(pos_list):
                x, y = pos
                self.string_vars[x][y].set(f'{i}')
                self.table[x][y].configure(fg=COLOR_LIST[player])
        # set food
        food_arr = game.food_pos()
        for f_idx in range(food_arr.shape[0]):
            fx, fy = food_arr[f_idx, 0], food_arr[f_idx, 1]
            self.string_vars[fx][fy].set(f'@')
            self.table[fx][fy].configure(fg='black')

