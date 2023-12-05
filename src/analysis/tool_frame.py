from tkinter import Tk, Frame, Label, Button, Radiobutton, IntVar
from typing import Callable

from src.analysis.core import GUIState
from src.analysis.style import FONT_NAME, FONT_SIZE, COLOR_LIST, FRAME_PADDING


class ToolFrame:
    def __init__(
            self,
            root: Tk,
            state: GUIState,
            on_button_callback: Callable,
    ):
        self.state = state
        self.on_button_callback = on_button_callback
        num_col = max(state.num_players, 4)
        font = (FONT_NAME, FONT_SIZE)
        # header
        self.f = Frame(root, padx=FRAME_PADDING, pady=FRAME_PADDING)
        self.header_label = Label(self.f, text='Tools', font=(FONT_NAME, FONT_SIZE + 2))
        self.header_label.grid(column=0, row=0, columnspan=num_col)
        # player selection
        self.player_label = Label(self.f, text='Player', font=font)
        self.player_label.grid(column=0, row=1, columnspan=num_col)
        self.player_button_list = []

        for i, p in enumerate(range(state.num_players)):
            b = Radiobutton(self.f, text=f'{p}', font=font, fg=COLOR_LIST[i], value=p,
                            variable=state.player_value, indicatoron=False)
            b.grid(column=i, row=2)
            self.player_button_list.append(b)
        # table tool selection
        self.table_label = Label(self.f, text='Table', font=font)
        self.table_label.grid(column=0, row=3, columnspan=num_col)

        h = Radiobutton(self.f, text='H', font=font, value=0, variable=state.tool_value, indicatoron=False)
        h.grid(column=0, row=4)
        t = Radiobutton(self.f, text='T', font=font, value=1, variable=state.tool_value, indicatoron=False)
        t.grid(column=1, row=4)
        # plus = Radiobutton(self.f, text='+', font=font, value=2, variable=tool_value, indicatoron=False)
        # plus.grid(column=2, row=4)
        # minus = Radiobutton(self.f, text='-', font=font, value=3, variable=tool_value, indicatoron=False)
        # minus.grid(column=3, row=4)
        delete = Radiobutton(self.f, text='X', font=font, value=2, variable=state.tool_value, indicatoron=False)
        delete.grid(column=2, row=4)
        food = Radiobutton(self.f, text='F', font=font, value=3, variable=state.tool_value, indicatoron=False)
        food.grid(column=3, row=4)
        # self.tool_buttons = [h, t, plus, minus, delete]
        self.tool_buttons = [h, t, delete]
        # health updater
        self.health_label = Label(self.f, text='Health', font=font)
        self.health_label.grid(column=0, row=5, columnspan=num_col)
        self.health_add = Button(self.f, text='+', font=font, command=self._h_plus_callback)
        self.health_add.grid(column=0, row=6, columnspan=2)
        self.health_sub = Button(self.f, text='-', font=font, command=self._h_minus_callback)
        self.health_sub.grid(column=2, row=6, columnspan=2)
        # length updater
        self.length_label = Label(self.f, text='Length', font=font)
        self.length_label.grid(column=0, row=7, columnspan=num_col)
        self.length_add = Button(self.f, text='+', font=font, command=self._l_plus_callback)
        self.length_add.grid(column=0, row=8, columnspan=2)
        self.length_sub = Button(self.f, text='-', font=font, command=self._l_minus_callback)
        self.length_sub.grid(column=2, row=8, columnspan=2)
        # flip and rotate buttons
        self.button_left = Button(self.f, text='Left90', font=font, command=self._left90_callback)
        self.button_left.grid(column=0, columnspan=num_col, row=9)
        self.button_right = Button(self.f, text='Right90', font=font, command=self._right90_callback)
        self.button_right.grid(column=0, columnspan=num_col, row=10)
        self.button_flip = Button(self.f, text='Flip', font=font, command=self._flip_callback)
        self.button_flip.grid(column=0, columnspan=num_col, row=11)

    def _h_plus_callback(self):
        self.on_button_callback("H+")

    def _h_minus_callback(self):
        self.on_button_callback("H-")

    def _l_plus_callback(self):
        self.on_button_callback("L+")

    def _l_minus_callback(self):
        self.on_button_callback("L-")

    def _left90_callback(self):
        self.on_button_callback("Left")

    def _right90_callback(self):
        self.on_button_callback("Right")

    def _flip_callback(self):
        self.on_button_callback("Flip")
