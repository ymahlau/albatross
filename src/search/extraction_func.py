from abc import ABC, abstractmethod

import numpy as np

from src.game.values import apply_utility_norm, UtilityNorm
from src.search.config import SpecialExtractConfig, ExtractFuncConfig, StandardExtractConfig, \
    MeanPolicyExtractConfig, PolicyExtractConfig
from src.search.node import Node


class ExtractFunc(ABC):
    def __init__(self, cfg: ExtractFuncConfig):
        self.cfg = cfg

    def __call__(self, root: Node) -> tuple[np.ndarray, np.ndarray]:
        if not root.is_root():
            raise ValueError("Cannot compute extraction function on non-root node")
        values, action_probs = self._compute_result(root)
        # sanity check
        if np.any(np.abs(np.sum(action_probs, axis=-1) - 1) > 0.001) or np.any(np.isnan(action_probs)) \
                or np.any(np.isinf(action_probs)):
            raise Exception(f"Action probabilities do not yield a prob dist: {action_probs}")
        return values, action_probs

    @abstractmethod
    def _compute_result(self, root: Node) -> tuple[np.ndarray, np.ndarray]:
        # given the root node, computes the values and action probs of each player
        # returns arrays of shape (num_players,) and (num_players, num_actions)
        raise NotImplementedError()


class StandardExtractFunc(ExtractFunc):
    """
    Standard value and action extraction: actions probs are player-action visit counts
    """
    def __init__(self, cfg: StandardExtractConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_result(self, root: Node) -> tuple[np.ndarray, np.ndarray]:
        if root.is_fully_explored():
            raise Exception("It is not wise to use standard extraction with optimization of fully explored subtrees")
        values = root.forward_estimate()
        # turn player-action dictionary to numpy array
        pa_visits: dict[tuple[int, int], int] = root.player_action_visits
        action_probs = np.zeros((root.game.num_players_at_turn(), root.game.num_actions))
        for idx, player in enumerate(root.game.players_at_turn()):
            for action in root.game.available_actions(player):
                action_probs[idx, action] = pa_visits[(player, action)]
        # normalize
        prob_sum = np.sum(action_probs, axis=-1)
        action_probs = action_probs / prob_sum[..., np.newaxis]
        return values, action_probs


class SpecialExtractFunc(ExtractFunc):
    """
    Extracts value and best actions from the tree search root node. E.g. for Nash or Maxmin
    """
    def __init__(self, cfg: SpecialExtractConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_result(self, root: Node) -> tuple[np.ndarray, np.ndarray]:
        if "action_probs" not in root.info or "action_probs_count" not in root.info:
            raise Exception("Special extraction only works with backup storing action probs")
        values = root.forward_estimate()
        values = apply_utility_norm(values, self.cfg.utility_norm)
        action_prob_sum: np.ndarray = root.info["action_probs"]
        action_prob_count: int = root.info["action_probs_count"]
        # sanity checks
        if len(action_prob_sum.shape) != 2 or action_prob_sum.shape[0] != root.game.num_players_at_turn() \
                or action_prob_sum.shape[1] != root.game.num_actions:
            raise Exception(f"Invalid action prob shape: {action_prob_sum.shape}")
        if action_prob_count < 1:
            raise Exception(f"Invalid action prob-count: {action_prob_count}")
        # if we fully explored the tree, value and probs are exactly calculated
        if root.is_fully_explored():
            action_probs = action_prob_sum
        else:
            action_probs = action_prob_sum / action_prob_count
        return values, action_probs


class MeanPolicyExtractFunc(ExtractFunc):
    def __init__(self, cfg: MeanPolicyExtractConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_result(self, root: Node) -> tuple[np.ndarray, np.ndarray]:
        if root.is_fully_explored():
            raise Exception("It is not wise to use standard extraction with optimization of fully explored subtrees")
        values = root.forward_estimate()
        # turn player-action dictionary to numpy array
        pa_sum = root.info["action_prob_sum"]
        action_probs = np.zeros((root.game.num_players_at_turn(), root.game.num_actions))
        for idx, player in enumerate(root.game.players_at_turn()):
            for action in root.game.available_actions(player):
                cur_prob = pa_sum[(player, action)] / root.visits
                action_probs[idx, action] = max(0, cur_prob)  # if exploration prob is removed, prob may be negative
        # normalize
        prob_sum = np.sum(action_probs, axis=-1)
        action_probs = action_probs / prob_sum[..., np.newaxis]
        return values, action_probs


class PolicyExtractFunc(ExtractFunc):
    def __init__(self, cfg: PolicyExtractConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _compute_result(self, root: Node) -> tuple[np.ndarray, np.ndarray]:
        if root.is_fully_explored():
            raise Exception("It is not wise to use policy extraction with optimization of fully explored subtrees")
        values = root.info["v"]
        policy = root.info["policy"]
        return values, policy


def get_extract_func_from_cfg(cfg: ExtractFuncConfig) -> ExtractFunc:
    if isinstance(cfg, StandardExtractConfig):
        return StandardExtractFunc(cfg)
    elif isinstance(cfg, SpecialExtractConfig):
        return SpecialExtractFunc(cfg)
    elif isinstance(cfg, MeanPolicyExtractConfig):
        return MeanPolicyExtractFunc(cfg)
    elif isinstance(cfg, PolicyExtractConfig):
        return PolicyExtractFunc(cfg)
    else:
        raise ValueError(f"Unknown extraction func type: {cfg}")

