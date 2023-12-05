import copy
from tkinter import Tk, IntVar, Event
from typing import Optional

from src.analysis.core import GUIState, GameState
from src.analysis.game_frame import GameFrame
from src.analysis.info_frame import InfoFrame
from src.analysis.net_frame import NetFrame
from src.analysis.tool_frame import ToolFrame
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.enc_conversion import decode_encoding
from src.network import Network
from src.misc.replay_buffer import ReplayBuffer


class ModelAnalyser:
    def __init__(self, net: Network, buffer: Optional[ReplayBuffer] = None):
        self.root = Tk()
        self.init_game = net.game.get_copy()
        self.game = net.game.get_copy()
        if not isinstance(self.game, BattleSnakeGame):
            raise ValueError(f"GUI only works for battlesnake game")
        self.net = net
        self.net.eval()
        self.buffer = buffer
        self.temperature_input = self.game.cfg.ec.temperature_input
        self.single_temp_input = self.game.cfg.ec.single_temperature_input
        self.state = GUIState(
            min_food=self.game.cfg.min_food,
            w=self.game.cfg.w,
            h=self.game.cfg.h,
            num_players=self.game.num_players,
            player_value=IntVar(self.root, 0),
            tool_value=IntVar(self.root, 0),
            action_values=[IntVar(self.root, value=0) for _ in range(self.net.game.num_players)],
            matrix_radio_var=IntVar(self.root, 0),
            buffer_idx=None if self.buffer is None else 0,
            last_buffer_idx=None if self.buffer is None else 0,
        )
        if self.buffer is None:
            self.game_state = self.state_from_game()
        else:
            self.game_state = self.state_from_buffer(0)
        # tool frame
        self.tool_frame = ToolFrame(
            root=self.root,
            state=self.state,
            on_button_callback=self.tool_button_callback,
        )
        # game frame
        self.game_frame = GameFrame(
            root=self.root,
            on_click_callback=self.on_click_callback,
            state=self.state,
        )
        # info frame
        self.info_frame = InfoFrame(
            root=self.root,
            state=self.state,
        )
        # net frame
        self.net_frame = NetFrame(
            root=self.root,
            state=self.state,
            buffer=self.buffer,
            temp_input=self.temperature_input,
            single_temp_input=self.single_temp_input,
        )
        self._init()
        self.update_frames()

    def _init(self):
        self.root.title("Model Analyser")
        self.tool_frame.f.grid(column=0, row=0, rowspan=2)
        self.game_frame.f.grid(column=1, row=0, rowspan=2)
        self.info_frame.f.grid(column=2, row=0)
        self.net_frame.f.grid(column=2, row=1)
        self.net_frame.set_move_callback(self.do_move)
        self.net_frame.set_reset_callback(self.do_reset)
        if self.buffer is not None:
            self.net_frame.set_next_callback(self.do_next)
            self.net_frame.set_prev_callback(self.do_prev)
        if self.temperature_input:
            self.net_frame.set_temp_callback(self.do_temperature)

    def __call__(self):
        self.root.mainloop()

    def state_from_game(self) -> GameState:
        if not isinstance(self.game, BattleSnakeGame):
            raise ValueError("Analyser only works with Battlesnake game")
        # omega conf does not accept numpy types, so we have to carefully cast everything to python primitives
        food_arr = self.game.food_pos()
        food_list = [[int(food_arr[i, 0]), int(food_arr[i, 1])] for i in range(food_arr.shape[0])]
        player_pos = {}
        for p in range(self.game.num_players):
            player_pos[p] = [[int(p_list[0]), int(p_list[1])] for p_list in self.game.player_pos(p)]
        state = GameState(
            turns_played=self.game.turns_played,
            snakes_alive=[p in self.game.players_alive() for p in range(self.state.num_players)],
            snake_pos=player_pos,
            food_pos=food_list,
            snake_health=[int(health) for health in self.game.player_healths()],
            snake_len=[int(length) for length in self.game.player_lengths()],
        )
        return state

    def state_from_buffer(self, buffer_idx: int) -> GameState:
        if not isinstance(self.game, BattleSnakeGame):
            raise ValueError("Analyser only works with Battlesnake game")
        if self.buffer is None:
            raise Exception("This should never happen")
        obs = self.buffer.content.dc_obs[buffer_idx]
        cfg = decode_encoding(self.game.cfg, obs)
        state = GameState(
            turns_played=cfg.init_turns_played,
            snakes_alive=cfg.init_snakes_alive, # type: ignore
            snake_pos=cfg.init_snake_pos, # type: ignore
            food_pos=cfg.init_food_pos, # type: ignore
            snake_health=cfg.init_snake_health, # type: ignore
            snake_len=cfg.init_snake_len, # type: ignore
        )
        return state

    def update_game_from_state(self):
        cfg = copy.deepcopy(self.game.cfg)
        if not isinstance(cfg, BattleSnakeConfig):
            raise Exception("This should never happen")
        cfg.init_turns_played = self.game_state.turns_played
        cfg.init_snakes_alive = self.game_state.snakes_alive
        cfg.init_snake_pos = self.game_state.snake_pos
        cfg.init_snake_health = self.game_state.snake_health
        cfg.init_snake_len = self.game_state.snake_len
        cfg.init_food_pos = self.game_state.food_pos
        self.game = BattleSnakeGame(cfg)

    def update_frames(self):
        self.game_frame.update_table(self.game) # type: ignore
        self.info_frame.update(self.game, self.state.buffer_idx) # type: ignore
        self.net_frame.update(self.game, self.net, self.state.buffer_idx) # type: ignore

    def do_move(self):
        if self.game.is_terminal():
            return
        action_list = []
        for player in self.game.players_at_turn():
            action_list.append(self.state.action_values[player].get())
        joint_action = tuple(action_list)
        if joint_action not in self.game.available_joint_actions():
            return
        self.game.step(joint_action)
        self.game_state = self.state_from_game()
        if self.state.buffer_idx is not None:  # save current index in buffer
            self.state.last_buffer_idx = self.state.buffer_idx
        self.state.buffer_idx = None
        self.update_frames()

    def do_reset(self):
        self.game = self.init_game.get_copy()
        self.game.reset()
        self.game_state = self.state_from_game()
        if self.state.buffer_idx is not None:  # save current index in buffer
            self.state.last_buffer_idx = self.state.buffer_idx
        self.state.buffer_idx = None
        self.update_frames()

    def do_temperature(self, _: float):
        self.update_frames()

    def on_click_callback(self, event: Event):
        x, y = event.widget.x, event.widget.y
        tv: int = self.state.tool_value.get()
        if tv == 0:
            self._set_head(x, y)
        if tv == 1:
            self._set_tail(x, y)
        if tv == 2:
            self._delete(x, y)
        if tv == 3:
            self._food(x, y)
        self.update_game_from_state()
        if self.state.buffer_idx is not None:  # save current index in buffer
            self.state.last_buffer_idx = self.state.buffer_idx
        self.state.buffer_idx = None
        self.update_frames()

    def occupied_by_player(self, x: int, y: int) -> Optional[tuple[int, int]]:
        # returns player-body_index if it exists
        for player_idx, p_list in enumerate(self.game_state.snake_pos.values()):
            for body_idx, pos in enumerate(p_list):
                if pos[0] == x and pos[1] == y:
                    return player_idx, body_idx
        return None

    def occupied_by_food(self, x: int, y: int) -> Optional[int]:
        for idx, food_xy in enumerate(self.game_state.food_pos):
            if x == food_xy[0] and y == food_xy[1]:
                return idx
        return None

    def _set_head(self, x: int, y: int):
        if self.occupied_by_player(x, y) is not None:
            return  # illegal to overwrite already set position
        if self.occupied_by_food(x, y) is not None:
            return
        self.game_state.snake_pos[self.state.player_value.get()].insert(0, [x, y])

    def _set_tail(self, x: int, y: int):
        if self.occupied_by_player(x, y) is not None:
            return  # illegal to overwrite already set position
        if self.occupied_by_food(x, y) is not None:
            return
        self.game_state.snake_pos[self.state.player_value.get()].append([x, y])

    def _delete(self, x: int, y: int):
        maybe_idx = self.occupied_by_player(x, y)
        if maybe_idx is not None:
            player_idx, body_idx = maybe_idx
            if len(self.game_state.snake_pos[player_idx]) <= 1:  # cannot completely remove snake from board
                return
            del self.game_state.snake_pos[player_idx][body_idx]
        maybe_idx = self.occupied_by_food(x, y)
        if maybe_idx is not None:
            if len(self.game_state.food_pos) <= self.state.min_food:
                return
            del self.game_state.food_pos[maybe_idx]

    def _food(self, x: int, y: int):
        if self.occupied_by_player(x, y) is not None:
            return
        if self.occupied_by_food(x, y) is not None:
            self._delete(x, y)
            return
        self.game_state.food_pos.append([x, y])

    def tool_button_callback(self, keycode: str):
        if keycode in ["Left", "Right", "Flip"]:
            self._rotate_flip_callback(keycode)
        elif keycode in ["H+", "H-"]:
            self._update_health(keycode)
        elif keycode in ["L+", "L-"]:
            self._update_length(keycode)
        else:
            raise Exception(f"Unknown keycode: {keycode}")
        self.update_game_from_state()
        if self.state.buffer_idx is not None:  # save current index in buffer
            self.state.last_buffer_idx = self.state.buffer_idx
        self.state.buffer_idx = None
        self.update_frames()

    def _update_health(self, keycode: str):
        p = self.state.player_value.get()
        if keycode == "H+":
            self.game_state.snake_health[p] += 1
        elif keycode == "H-":
            self.game_state.snake_health[p] -= 1

    def _update_length(self, keycode: str):
        p = self.state.player_value.get()
        if keycode == "L+":
            self.game_state.snake_len[p] += 1
        elif keycode == "L-":
            self.game_state.snake_len[p] -= 1

    def _rotate_flip_callback(self, keycode: str):
        w, h = self.state.w, self.state.h
        # build new body position list
        for p, body_list in self.game_state.snake_pos.items():
            new_list = []
            for pos in body_list:
                x, y = pos[0], pos[1]
                if keycode == "Left":
                    new_list.append([-y + w - 1, x])
                elif keycode == "Right":
                    new_list.append([y, -x + h - 1])
                elif keycode == "Flip":
                    new_list.append([x, h - y - 1])
            self.game_state.snake_pos[p] = new_list
        # build new food list
        new_list = []
        for food_p in self.game_state.food_pos:
            x, y = food_p[0], food_p[1]
            if keycode == "Left":
                new_list.append([-y + w - 1, x])
            elif keycode == "Right":
                new_list.append([y, -x + h - 1])
            elif keycode == "Flip":
                new_list.append([x, h - y - 1])
        self.game_state.food_pos = new_list

    def do_next(self):
        last_idx = self.state.last_buffer_idx
        if last_idx is None:
            raise Exception("This should never happen")
        self.state.buffer_idx = (last_idx + 1) % len(self.buffer) # type: ignore
        self.state.last_buffer_idx = (last_idx + 1) % len(self.buffer) # type: ignore
        self.game_state = self.state_from_buffer(self.state.buffer_idx)
        self.update_game_from_state()
        self.update_frames()

    def do_prev(self):
        last_idx = self.state.last_buffer_idx
        if last_idx is None:
            raise Exception("This should never happen")
        self.state.buffer_idx = (last_idx - 1) % len(self.buffer) # type: ignore
        self.state.last_buffer_idx = (last_idx - 1) % len(self.buffer) # type: ignore
        self.game_state = self.state_from_buffer(self.state.buffer_idx)
        self.update_game_from_state()
        self.update_frames()