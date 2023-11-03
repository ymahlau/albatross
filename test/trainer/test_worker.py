import unittest

import numpy as np

from src.trainer.az_worker import td_lambda_inefficient, td_lambda


class TestWorker(unittest.TestCase):
    def test_lambda_return_inefficient(self):
        full_rewards = np.asarray([
            [0, 0, 0, 1],
            [0, 0, 0, -1],
        ], dtype=float)
        full_values = np.asarray([
            [0.6, 0.7, 0.8, 0.95],
            [-0.6, -0.7, -0.8, -0.95],
        ], dtype=float)
        episode_len_player = [4, 4]
        ld = 0.5
        discount = 0.95
        result_arr = td_lambda_inefficient(
            full_rewards=full_rewards,
            full_values=full_values,
            ld=ld,
            discount=discount,
            episode_len_player=episode_len_player,
        ).T
        for i in range(4):
            self.assertEqual(result_arr[i, 0], -result_arr[i, 1])
        self.assertEqual(0.95, result_arr[3, 0])
        self.assertEqual(-0.95, result_arr[3, 1])

    def test_lambda_return_last_zero(self):
        full_rewards = np.asarray([
            [0, 0],
            [0, 0],
        ], dtype=float)
        full_values = np.asarray([
            [0.5, 1],
            [-0.5, -1],
        ], dtype=float)
        episode_len_player = [2, 2]
        ld = 0.5
        discount = 1
        result_arr = td_lambda_inefficient(
            full_rewards=full_rewards,
            full_values=full_values,
            ld=ld,
            discount=discount,
            episode_len_player=episode_len_player,
        ).T
        for i in range(2):
            self.assertEqual(result_arr[i, 0], -result_arr[i, 1])
        self.assertEqual(0.5, result_arr[0, 0])
        self.assertEqual(-0.5, result_arr[0, 1])
        self.assertEqual(0.5, result_arr[1, 0])
        self.assertEqual(-0.5, result_arr[1, 1])

    def test_lambda_multiplayer_inefficient(self):
        full_rewards = np.asarray([
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 0, -1, 0],
        ], dtype=float)
        full_values = np.asarray([
            [0.6, 0.7, 0.8, 0.95],
            [-0.6, -0.7, -0.8, -0.95],
            [0, 0, -1, 0],
        ], dtype=float)
        episode_len_player = [4, 4, 3]
        ld = 0.5
        discount = 0.95
        result_arr = td_lambda_inefficient(
            full_rewards=full_rewards,
            full_values=full_values,
            ld=ld,
            discount=discount,
            episode_len_player=episode_len_player,
        ).T
        for i in range(4):
            self.assertEqual(result_arr[i, 0], -result_arr[i, 1])
        self.assertEqual(0.95, result_arr[3, 0])
        self.assertEqual(-0.95, result_arr[3, 1])
        self.assertEqual(-0.975, result_arr[2, 2])
        self.assertEqual(0, result_arr[3, 2])

    def test_lambda_efficient(self):
        full_rewards = np.asarray([
            [0, 0, 0, 1],
            [0, 0, 0, -1],
        ], dtype=float)
        full_values = np.asarray([
            [0.6, 0.7, 0.8, 0.95],
            [-0.6, -0.7, -0.8, -0.95],
        ], dtype=float)
        episode_len_player = [4, 4]
        ld = 0.5
        discount = 0.95
        result_arr = td_lambda(
            full_rewards=full_rewards,
            full_values=full_values,
            ld=ld,
            discount=discount,
        ).T
        inefficient_result_arr = td_lambda_inefficient(
            full_rewards=full_rewards,
            full_values=full_values,
            ld=ld,
            discount=discount,
            episode_len_player=episode_len_player,
        ).T
        for i in range(4):
            self.assertEqual(result_arr[i, 0], -result_arr[i, 1])
        self.assertEqual(0.95, result_arr[3, 0])
        self.assertEqual(-0.95, result_arr[3, 1])
        for p in range(2):
            for i in range(4):
                self.assertAlmostEqual(result_arr[i, p], inefficient_result_arr[i, p], places=6)

    def test_td_lambda_multiplayer(self):
        full_rewards = np.asarray([
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 0, -1, 0],
        ], dtype=float)
        full_values = np.asarray([
            [0.6, 0.7, 0.8, 0.95],
            [-0.6, -0.7, -0.8, -0.95],
            [0, 0, -1, 0],
        ], dtype=float)
        episode_len_player = [4, 4, 3]
        ld = 0.5
        discount = 0.95
        result_arr = td_lambda(
            full_rewards=full_rewards,
            full_values=full_values,
            ld=ld,
            discount=discount,
        ).T
        result_arr_inefficient = td_lambda_inefficient(
            full_rewards=full_rewards,
            full_values=full_values,
            ld=ld,
            discount=discount,
            episode_len_player=episode_len_player,
        ).T
        for i in range(4):
            self.assertEqual(result_arr[i, 0], -result_arr[i, 1])
        self.assertEqual(0.95, result_arr[3, 0])
        self.assertEqual(-0.95, result_arr[3, 1])
        self.assertEqual(-0.975, result_arr[2, 2])
        self.assertEqual(0, result_arr[3, 2])
        for p in range(3):
            for i in range(4):
                self.assertAlmostEqual(result_arr[i, p], result_arr_inefficient[i, p], places=6)
