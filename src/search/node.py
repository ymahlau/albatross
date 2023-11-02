from collections import defaultdict
from typing import Optional, Any

import numpy as np

from src.game import Game


class Node:
    def __init__(
            self,
            parent: Optional["Node"],
            last_actions: Optional[tuple[int, ...]],
            discount: float,
            game: Optional[Game],
            ignore_full_exploration: bool
    ):
        if parent is None and last_actions is not None or parent is not None and last_actions is None:
            raise ValueError("You either have to provide both parent and last actions or none of the two")
        if parent is None and game is None:
            raise ValueError("You either have to provide a game environment or the parent node")
        if parent is not None and game is not None:
            raise ValueError("You cannot provide a parent and a game")
        self.children: Optional[dict[tuple[int, ...], Node]] = None  # Key: joint action
        self.parent: Optional[Node] = parent
        self.last_actions = last_actions  # action that lead from last state to this state
        self.discount: float = discount
        self.ignore_full_exploration: bool = ignore_full_exploration

        # reward that was obtained when getting from parent with last_action to this for each player
        self.rewards: Optional[np.ndarray] = None
        if game is None and parent is not None:
            self.game: Game = parent.game.get_copy()
            rewards, done, info = self.game.step(last_actions)
            self.rewards = rewards
        else:
            # init game state
            self.game: Game = game
            # root has no additional rewards, no previous action or state
            self.rewards = np.zeros((self.game.num_players,), dtype=float)
        # Statistics
        self.visits: int = 0
        self.value_sum: np.ndarray = np.zeros((self.game.num_players,), dtype=float)
        self.player_action_value_sum: dict[tuple[int, int], float] = defaultdict(lambda: 0)
        self.player_action_visits: dict[tuple[int, int], int] = defaultdict(lambda: 0)
        self.num_children_fully_explored = 0
        # backup, selection and eval function can write information to this dictionary for later use (e.g. extraction)
        self.info: dict[str, Any] = {}

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.children is None

    def is_terminal(self):  # terminal implies also being a leaf
        return self.game.is_terminal()

    def is_fully_explored(self):
        if self.ignore_full_exploration:
            return False
        if self.is_terminal():
            return True
        if self.is_leaf():  # non-terminal leafs are by definition not fully explored
            return False
        if self.num_children_fully_explored > len(self.children):
            raise Exception("Cannot have more fully explored children than actual children")
        return self.num_children_fully_explored == len(self.children)

    def get_joint_backward_estimate(self) -> dict[tuple[int, ...], np.ndarray]:
        # joint values of node from the viewpoint of the previous game state
        if self.is_leaf():
            raise Exception("Cannot compute action values for leaf node")
        result_dict = {}
        for joint_action, child in self.children.items():
            if child.visits > 0:
                result_dict[joint_action] = child.backward_estimate_zero_fill()
            else:
                result_dict[joint_action] = self.discount * child.rewards
        return result_dict

    def backward_estimate_zero_fill(self) -> np.ndarray:
        # value of node from the viewpoint of the previous game state
        if self.is_terminal():
            return self.discount * self.rewards
        if self.visits == 0:
            return np.zeros_like(self.value_sum, dtype=float)
        return self.discount * (self.rewards + self.value_sum / self.visits)

    def backward_estimate(self) -> np.ndarray:
        # value of node from the viewpoint of the previous game state
        if self.is_terminal():
            return self.discount * self.rewards
        if self.visits == 0:
            raise Exception(f"Cannot compute backward estimate on unvisited node")
        return self.discount * (self.rewards + self.value_sum / self.visits)

    def forward_estimate(self) -> np.ndarray:
        if self.is_terminal():
            raise Exception(f"forward estimate is not defined for terminal nodes")
        if self.visits == 0:
            raise Exception(f"Cannot compute forward estimate on unvisited node")
        return self.value_sum / self.visits
