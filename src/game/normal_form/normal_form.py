from functools import cached_property
import itertools
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import torch

from src.game.game import Game, GameConfig


@dataclass(kw_only=True)
class NormalFormConfig(GameConfig):
    ja_dict: dict[tuple[int, ...], tuple[float, ...]]
    num_players: int = field(default=-1)  # filled automatically
    num_actions: int = field(default=-1)
    
    def __post__init___(self):
        self.num_players = len(list(self.ja_dict.keys())[0])
        max_a = -2
        for p in range(self.num_players):
            if max(self.aa_dict[p]) > max_a:
                max_a = max(self.aa_dict[p])        
        self.num_actions = max_a + 1
        
    @cached_property
    def aa_dict(self) -> dict[int, list[int]]:
        return {
            p: [] for p in range(self.num_players)
        }


class NormalFormGame(Game):
    def __init__(self, cfg: NormalFormConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.terminal_value: bool = False

    def _step(self, actions: tuple[int, ...]) -> tuple[np.ndarray, bool, dict]:
        rewards = np.asarray(self.cfg.ja_dict[actions], dtype=float)
        self.terminal_value = True
        return rewards, True, {}

    def _reset(self):
        self.terminal_value = False

    def render(self):
        print(self.get_str_repr(), flush=True)

    def _get_copy(self) -> "NormalFormGame":
        cpy = NormalFormGame(self.cfg)
        cpy.terminal_value = self.terminal_value
        return cpy

    def __eq__(self, game: "NormalFormGame") -> bool:
        if not isinstance(game, NormalFormGame):
            return False
        if self.is_terminal() != game.is_terminal():
            return False
        if self.get_last_actions() != game.get_last_actions():
            return False
        ja_dict = self.cfg.ja_dict
        other_ja_dict = game.cfg.ja_dict
        if len(ja_dict) != other_ja_dict:
            return False
        for k, v in ja_dict.items():
            if k not in other_ja_dict:
                return False
            if v != other_ja_dict[k]:
                return False
        return True

    def available_actions(self, player: int) -> list[int]:
        return self.cfg.aa_dict[player]

    def players_at_turn(self) -> list[int]:
        if self.terminal_value:
            return []
        else:
            return list(range(self.num_players))

    def players_alive(self) -> list[int]:
        if self.terminal_value:
            return []
        else:
            return list(range(self.num_players))

    def is_terminal(self) -> bool:
        return self.terminal_value

    def get_symmetry_count(self):
        return 1

    def get_obs_shape(self, never_flatten=False) -> tuple[int, ...]:
        raise Exception(f"It makes no sense to get the observation of a Normal-Form game")

    def get_obs(
            self,
            symmetry: Optional[int] = 0,
            temperatures: Optional[list[float]] = None,
            single_temperature: Optional[bool] = None
    ) -> tuple[
        torch.Tensor,
        dict[int, int],
        dict[int, int],
    ]:
        raise Exception(f"It makes no sense to get the observation of a Normal-Form game")

    def get_str_repr(self) -> str:
        result = ""
        for k, v in self.cfg.ja_dict.items():
            result += f"{k}: {v}\n"
        return result

    def close(self):
        # nothing to close here
        pass
