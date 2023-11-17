import itertools
import random
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np


@dataclass
class GameConfig:
    num_players: int
    num_actions: int  # max number of actions possible


class Game(ABC):
    """
    Interface of a Game for this Reinforcement Learning-Framework. It describes Simultaneous-Move Games with discrete
    Action spaces
    """
    def __init__(
            self,
            cfg: GameConfig
    ):
        self.cfg = cfg
        self._cum_rewards = np.zeros(shape=(self.cfg.num_players,), dtype=float)
        self._last_actions: Optional[tuple[int, ...]] = None
        self.turns_played = 0

    # reward, done, info as return
    # action is the joint action of all currently active players
    def step(self, actions: tuple[int, ...]) -> tuple[np.ndarray, bool, dict]:
        """
        reward, done, info = step(action)
        """
        if self.is_terminal():
            raise Exception('Cannot call step on terminal state')
        if len(actions) != self.num_players_at_turn():
            raise ValueError(f"Invalid action length: {actions}")
        if actions not in self.available_joint_actions():
            raise ValueError(f'Calling step with non-legal actions: {actions}')
        reward, done, info = self._step(actions)
        self._cum_rewards += reward
        self._last_actions = actions
        self.turns_played += 1
        return reward, done, info

    @abstractmethod
    def _step(self, actions: tuple[int, ...]) -> tuple[np.ndarray, bool, dict]:
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    def reset(self):
        self._cum_rewards = np.zeros(shape=(self.num_players,), dtype=float)
        self._last_actions = None
        self.turns_played = 0
        self._reset()

    @abstractmethod
    def _reset(self):
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()

    def get_copy(self) -> "Game":
        cpy = self._get_copy()
        cpy._last_actions = self._last_actions
        cpy._cum_rewards = self._cum_rewards.copy()
        cpy.turns_played = self.turns_played
        return cpy

    @abstractmethod
    def _get_copy(self) -> "Game":
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, game: "Game") -> bool:
        raise NotImplementedError()

    @abstractmethod
    def available_actions(self, player: int) -> list[int]:
        """
            Return list of currently available actions for player. Empty list if it is not players
                               turn
        """
        raise NotImplementedError()

    def available_joint_actions(self) -> list[tuple[int, ...]]:
        action_lists = []
        for player in range(self.num_players):
            current_list = self.available_actions(player)
            if current_list:
                action_lists.append(current_list)
        result = list(itertools.product(*action_lists))
        return result

    def illegal_actions(self, player: int) -> list[int]:
        if player < 0 or player >= self.num_players:
            raise ValueError(f'Snake index out of range: {player}')
        all_action_set = {i for i in range(self.num_actions)}
        legal_action_set: set[int] = set(self.available_actions(player))
        illegal_action_set = all_action_set - legal_action_set
        illegal_actions = list(illegal_action_set)
        return illegal_actions

    def illegal_joint_actions(self) -> list[tuple[int, ...]]:
        action_lists = [[i for i in range(self.num_actions)] for _ in range(self.num_players)]
        all_action_set: set[tuple[int, ...]] = set(itertools.product(*action_lists))
        available_joint_action_set: set[tuple[int, ...]] = set(self.available_joint_actions())
        illegal_action_set = all_action_set - available_joint_action_set
        return list(illegal_action_set)

    @abstractmethod
    def players_at_turn(self) -> list[int]:  # List of players, which can make a turn
        raise NotImplementedError()

    def players_not_at_turn(self) -> list[int]:  # list of players, which cannot make a turn
        result = set(range(self.num_players)) - set(self.players_at_turn())
        return list(result)

    def is_player_at_turn(self, player: int) -> bool:
        return player in self.players_at_turn()

    @property
    def num_players(self) -> int:
        return self.cfg.num_players

    def num_players_at_turn(self) -> int:
        return len(self.players_at_turn())

    @property
    def num_actions(self):
        return self.cfg.num_actions

    @abstractmethod
    def players_alive(self) -> list[int]:
        raise NotImplementedError()

    def players_not_alive(self) -> list[int]:
        result = set(range(self.num_players)) - set(self.players_alive())
        return list(result)

    def num_players_alive(self) -> int:
        return len(self.players_alive())

    def is_player_alive(self, player: int) -> bool:
        return player in self.players_alive()

    def get_last_actions(self) -> tuple[int, ...]:
        if self._last_actions is None:
            raise Exception("Last actions is None, probably because it is the start of the game.")
        return self._last_actions

    def set_last_actions(self, last_actions: Optional[tuple[int, ...]]):
        self._last_actions = last_actions

    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError()

    def get_cum_rewards(self) -> np.ndarray:
        return self._cum_rewards

    def set_cum_rewards(self, cum_rewards: np.ndarray):
        self._cum_rewards = cum_rewards

    def play_random_steps(self, steps: int):
        rndm = random.Random()
        while (not self.is_terminal()) and steps > 0:
            steps -= 1
            self.step(rndm.choice(self.available_joint_actions()))

    @abstractmethod
    def get_symmetry_count(self):
        """
            returns the total number of possible symmetries with the encoding
        """
        raise NotImplementedError()

    @abstractmethod
    def get_obs_shape(self, never_flatten=False) -> tuple[int, ...]:
        """
        Returns the shape of the observation-tensor describing a game state. If never_flatten, the output is
        multidimensional even if the game config defines it otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_obs(
            self,
            symmetry: Optional[int] = 0,
            temperatures: Optional[list[float]] = None,
            single_temperature: Optional[bool] = None,
    ) -> tuple[
        np.ndarray,
        dict[int, int],
        dict[int, int],
    ]:
        """
           Args:
               symmetry: number between 0 and get_symmetry_count()-1 that selects the symmetry.
                '0' applies no symmetry (identity function). if None is given, a random symmetry is chosen.
               temperatures: sbr temperatures if temperature is input of the model
               single_temperature: True if all snakes play with the same temperature. overwrites the config file
           Returns:
               1. symmetric encoding of shape (num_players_at_turn, *obs_shape)
               2. permutation of action: base state -> symmetric state
               3. inverse permutation of action: symmetric state -> base state
        """
        raise NotImplementedError()

    @abstractmethod
    def get_str_repr(self) -> str:
        """
        Returns string representation of current game state.
        """
        raise NotImplementedError()

    def get_last_action(self) -> Optional[tuple[int, ...]]:
        return self._last_actions
