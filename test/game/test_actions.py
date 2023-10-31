import math
import time
import unittest

import numpy as np
import torch.distributions

from src.game.actions import filter_illegal_and_normalize, sample_individual_actions, sample_joint_action
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig


class TestActions(unittest.TestCase):
    def test_action_epsilon(self):
        cfg = BattleSnakeConfig(num_players=2, w=5, h=5, all_actions_legal=True)
        game = BattleSnakeGame(cfg)

        actions = np.asarray([[0, 0, 0, 0], [0.3, 0.5, 0.1, 0.1]])
        filtered = filter_illegal_and_normalize(actions, game)
        self.assertEqual(0.25, filtered[0, 0])
        self.assertAlmostEqual(0.3, filtered[1, 0], places=2)  # epsilon skewers probs a bit

    def test_sample_individual_actions_argmax(self):
        action_probs = np.asarray([[0.2, 0.3, 0, 0.5], [0.3, 0.3, 0.2, 0.2]], dtype=float)
        ja = sample_individual_actions(action_probs, math.inf)
        self.assertTrue(ja in [(3, 0), (3, 1)])

    def test_sample_joint_actions(self):
        ja_list = [(0, 0), (0, 1), (0, 3), (1, 2), (4, 7)]
        ja_probs = np.asarray([0.3, 0.2, 0.3, 0, 0.2], dtype=float)
        ja = sample_joint_action(ja_list, ja_probs, math.inf)
        self.assertTrue(ja in [(0, 0), (0, 3)])

    def test_dirichlet_speed(self):
        alpha = 3
        eps = 0.25
        num_samples = 1000
        num_actions = 4
        num_player = 2
        action_probs = np.ones((num_player, num_actions), dtype=float) / num_actions

        print('\n')
        start = time.time()
        for _ in range(num_samples):
            alpha_list = [alpha for _ in range(action_probs.shape[1])]
            noise = np.random.default_rng().dirichlet(alpha_list, action_probs.shape[0])
            new_actions = (1 - eps) * action_probs + eps * noise
        print(f"{time.time() - start} - rng obj")

        start = time.time()
        for _ in range(num_samples):
            alpha_list = [alpha for _ in range(action_probs.shape[1])]
            noise = np.random.dirichlet(alpha=alpha_list, size=action_probs.shape[0])
            new_actions = (1 - eps) * action_probs + eps * noise
        print(f"{time.time() - start} - np method")

        start = time.time()
        for _ in range(num_samples):
            alpha_list = np.ones(shape=(action_probs.shape[1],), dtype=float) * alpha
            noise = np.random.dirichlet(alpha=alpha_list, size=action_probs.shape[0])
            new_actions = (1 - eps) * action_probs + eps * noise
        print(f"{time.time() - start} - np method -arr")

        start = time.time()
        alpha_list = [alpha for _ in range(action_probs.shape[1])]
        for _ in range(num_samples):
            noise = np.random.dirichlet(alpha=alpha_list, size=action_probs.shape[0])
            new_actions = (1 - eps) * action_probs + eps * noise
        print(f"{time.time() - start} - fixed alpha list ")

        start = time.time()
        alpha_list = [alpha for _ in range(action_probs.shape[1])]
        dirichlet = torch.distributions.Dirichlet(torch.tensor(alpha_list, dtype=torch.float32))
        pt_action_probs = torch.tensor(action_probs, dtype=torch.float32)
        for _ in range(num_samples):
            noise = dirichlet.sample(pt_action_probs.shape[:1])
            new_actions = (1 - eps) * pt_action_probs + eps * noise
        print(f"{time.time() - start} - pytorch")
