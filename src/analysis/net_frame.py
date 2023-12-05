from tkinter import Tk, Frame, Label, Button, StringVar, Radiobutton, Scale, DoubleVar, HORIZONTAL
from typing import Callable, Optional

import numpy as np
import torch.nn.functional

from src.analysis.core import GUIState
from src.analysis.style import FRAME_PADDING, FONT_SIZE, FONT_NAME, COLOR_LIST, BUFFER_COLOR
from src.game.actions import softmax
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.misc.utils import softmax_weighting
from src.network import Network
from src.misc.replay_buffer import ReplayBuffer


class NetFrame:
    def __init__(
            self,
            root: Tk,
            state: GUIState,
            buffer: Optional[ReplayBuffer],
            temp_input: bool,
            single_temp_input: bool,
    ):
        self.state = state
        self.root = root
        self.buffer = buffer
        self.temp_input = temp_input
        self.single_temp_input = single_temp_input
        self.f = Frame(root, padx=FRAME_PADDING, pady=FRAME_PADDING)
        font = (FONT_NAME, FONT_SIZE)
        # player action probs
        self.player_buttons = []
        self.player_vars = []
        self.player_value_labels = []
        self.player_value_vars: list[StringVar] = []
        self.num_crosses = self.state.num_players if self.buffer is None else self.state.num_players + 1
        for player in range(self.num_crosses):
            color = COLOR_LIST[player] if player < state.num_players else BUFFER_COLOR
            variable = self.state.action_values[player] if player != self.state.num_players else None
            width, height = 4, 2
            # value label
            value_var = StringVar(self.f, value='v')
            lbl = Label(self.f, textvariable=value_var, font=font)
            lbl.grid(row=0, column=3 * player, columnspan=3)
            self.player_value_vars.append(value_var)
            self.player_value_labels.append(lbl)
            # up
            up_var = StringVar(self.f, value="u")
            up = Radiobutton(self.f, textvariable=up_var, font=font, height=height, width=width,
                             fg=color, variable=variable, value=0, # type: ignore
                             indicatoron=False, command=self._on_action_click)
            up.grid(row=2, column=3 * player + 1)
            # right
            right_var = StringVar(self.f, value="r")
            right = Radiobutton(self.f, textvariable=right_var, font=font, height=height, width=width,
                                fg=color, variable=variable, value=1, # type: ignore
                                indicatoron=False, command=self._on_action_click)
            right.grid(row=3, column=3 * player + 2)
            # down
            down_var = StringVar(self.f, value="d")
            down = Radiobutton(self.f, textvariable=down_var, font=font, height=height, width=width,
                               fg=color, variable=variable, value=2, # type: ignore
                               indicatoron=False, command=self._on_action_click)
            down.grid(row=4, column=3 * player + 1)
            # left
            left_var = StringVar(self.f, value="l")
            left = Radiobutton(self.f, textvariable=left_var, font=font, height=height, width=width,
                               fg=color, variable=variable, value=3, # type: ignore
                               indicatoron=False, command=self._on_action_click)
            left.grid(row=3, column=3 * player)
            # save
            self.player_buttons.append([up, right, down, left])
            self.player_vars.append([up_var, right_var, down_var, left_var])
        self.move_btn = Button(self.f, text='Move', font=font)
        self.move_btn.grid(row=5, column=0, columnspan=3 * self.state.num_players)
        self.reset_btn = Button(self.f, text='Reset', font=font)
        self.reset_btn.grid(row=6, column=0, columnspan=3 * self.state.num_players)
        row_counter = 6
        if self.buffer is not None:
            self.buffer_prev_btn = Button(self.f, text='Prev', font=font)
            self.buffer_prev_btn.grid(row=7, column=0, columnspan=3 * self.state.num_players)
            self.buffer_next_btn = Button(self.f, text='Next', font=font)
            self.buffer_next_btn.grid(row=8, column=0, columnspan=3 * self.state.num_players)
            row_counter = 8
        if temp_input:
            # scale_label = Label(self.f, text="Temp", font=font)
            # scale_label.grid(row=row_counter + 1, column=0, columnspan=1)
            num_scales = 1 if self.single_temp_input else self.state.num_players
            self.scales = []
            self.scale_vars = []
            for i in range(num_scales):
                self.scale_vars.append(DoubleVar(value=3))
                self.scales.append(Scale(self.f, variable=self.scale_vars[i], from_=0, to=10, orient=HORIZONTAL,
                                         resolution=0.5))
                self.scales[i].grid(row=row_counter + 1, column=3*i, columnspan=3)
            row_counter += 1
        # player matrix (if two players)
        if self.state.num_players != 2:
            return
        # labels
        start_row = row_counter + 1
        Label(self.f, text='UP', fg=COLOR_LIST[0], font=font).grid(row=start_row + 1, column=0)
        Label(self.f, text='RIGHT', fg=COLOR_LIST[0], font=font).grid(row=start_row + 2, column=0)
        Label(self.f, text='DOWN', fg=COLOR_LIST[0], font=font).grid(row=start_row + 3, column=0)
        Label(self.f, text='LEFT', fg=COLOR_LIST[0], font=font).grid(row=start_row + 4, column=0)
        Label(self.f, text='UP', fg=COLOR_LIST[1], font=font).grid(row=start_row, column=1)
        Label(self.f, text='RIGHT', fg=COLOR_LIST[1], font=font).grid(row=start_row, column=2)
        Label(self.f, text='DOWN', fg=COLOR_LIST[1], font=font).grid(row=start_row, column=3)
        Label(self.f, text='LEFT', fg=COLOR_LIST[1], font=font).grid(row=start_row, column=4)
        # matrix vars
        self.matrix_values = [[StringVar(self.f, value='-') for _ in range(4)] for _ in range(4)]
        entry_w, entry_h = 8, 2
        for a0 in range(4):
            for a1 in range(4):
                rb = Radiobutton(self.f, textvariable=self.matrix_values[a0][a1], font=(FONT_NAME, FONT_SIZE-5),
                                 height=entry_h,
                                 width=entry_w, variable=self.state.matrix_radio_var, value=4 * a0 + a1,
                                 indicatoron=False, command=self._on_matrix_click)
                rb.grid(row=start_row+a0+1, column=a1+1)
                # rb.bind("<Button-1>", self._on_matrix_click)

    def output_from_game(
            self,
            game: BattleSnakeGame,
            net: Network
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        # observation
        temp_obs_input = None
        if self.temp_input and self.single_temp_input:
            temp_obs_input = [self.scale_vars[0].get()]
        elif self.temp_input and not self.single_temp_input:
            temp_obs_input = [self.scale_vars[i].get() for i in range(self.state.num_players)]
        obs, _, _, = game.get_obs(0, temperatures=temp_obs_input)
        # forward pass
        net_out = net(torch.tensor(obs)).detach().cpu().numpy()
        values = net.retrieve_value(net_out)
        probs = None
        if net.cfg.predict_policy:
            action_logits = net.retrieve_policy(net_out)
            probs = softmax(action_logits, temperature=1)
        return values, probs

    @torch.no_grad()
    def update(self, game: BattleSnakeGame, net: Network, buffer_idx: Optional[int]):
        # reset values
        for p in range(self.num_crosses):
            self.player_value_vars[p].set('-')
            for a in range(4):
                self.player_vars[p][a].set('-')
        if game.is_terminal():
            return
        values, probs = self.output_from_game(game, net)
        # new values
        for player_idx, player in enumerate(game.players_at_turn()):
            self.player_value_vars[player].set(f"{values[player_idx]:.2f}".replace("0.", "."))
            if probs is not None:
                for a in range(4):
                    self.player_vars[player][a].set(f"{probs[player_idx, a]:.2f}".lstrip("0"))
        # update buffer cross
        if buffer_idx is not None:
            if self.buffer is None:
                raise Exception("This should never happen")
            p_idx = game.num_players
            # value
            buffer_val = self.buffer.content.dc_val[buffer_idx].item()
            self.player_value_vars[p_idx].set(f"{buffer_val:.2f}".replace("0.", "."))
            # policy
            buffer_policy = self.buffer.content.dc_pol[buffer_idx]
            for a in range(4):
                self.player_vars[p_idx][a].set(f"{buffer_policy[a]:.2f}".lstrip("0"))
        # update matrix
        if self.state.num_players != 2:
            return
        # reset old values
        for a0 in range(4):
            for a1 in range(4):
                self.matrix_values[a0][a1].set("-")
        # set new values
        for ja in game.available_joint_actions():
            cpy = game.get_copy()
            if not isinstance(cpy, BattleSnakeGame):
                raise Exception("This should never happen")
            cpy.step(ja)
            if cpy.is_terminal():
                v = cpy.get_cum_rewards()
            else:
                v, _ = self.output_from_game(cpy, net)
            a0, a1 = ja
            v0_str = f"{v[0]:.2f}".replace("0.", ".").rstrip("0")
            if v0_str == ".":
                v0_str = "0"
            v1_str = f"{v[1]:.2f}".replace("0.", ".").rstrip("0")
            if v1_str == ".":
                v1_str = "0"
            self.matrix_values[a0][a1].set(f"{v0_str}, {v1_str}")

    def _on_action_click(self):
        # update matrix value
        self.state.matrix_radio_var.set(self.state.action_values[0].get() * 4 + self.state.action_values[1].get())

    def _on_matrix_click(self):
        # update action values
        a0 = int(self.state.matrix_radio_var.get() / 4)
        a1 = self.state.matrix_radio_var.get() % 4
        self.state.action_values[0].set(a0)
        self.state.action_values[1].set(a1)

    def set_move_callback(self, fnc: Callable):
        self.move_btn.configure(command=fnc)

    def set_reset_callback(self, fnc: Callable):
        self.reset_btn.configure(command=fnc)

    def set_prev_callback(self, fnc: Callable):
        self.buffer_prev_btn.configure(command=fnc)

    def set_next_callback(self, fnc: Callable):
        self.buffer_next_btn.configure(command=fnc)

    def set_temp_callback(self, fnc: Callable):
        for scale in self.scales:
            scale.configure(command=fnc)
