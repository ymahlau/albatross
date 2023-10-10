import time
import unittest

import numpy as np

from src.equilibria.nash import calculate_nash_equilibrium_python, calculate_nash_equilibrium


class TestNashSolver(unittest.TestCase):
    def test_simple(self):
        arr1 = np.asarray([[1, -1], [-1, 1]], dtype=np.float32)
        arr2 = np.asarray([[-1, 1], [1, -1]], dtype=np.float32)

        values, action_probs = calculate_nash_equilibrium_python(arr1, arr2)
        print(values)
        print(action_probs)
        self.assertEqual(0, values[0])
        self.assertEqual(0, values[1])
        for player in range(2):
            for action in range(2):
                self.assertEqual(0.5, action_probs[(player, action)])

    def test_simple_general_interface(self):
        available_actions = [[3, 2], [5, 1]]
        joint_action_list = [(2, 1), (3, 5), (3, 1), (2, 5)]
        joint_action_values = np.asarray([[1, -1], [1, -1], [-1, 1], [-1, 1]], dtype=float)
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_values,
                                                          use_cpp=False)
        print(values)
        print(action_probs)
        self.assertEqual(0, values[0])
        self.assertEqual(0, values[1])
        for player in range(2):
            for action in range(2):
                self.assertEqual(0.5, action_probs[player][action])


    def test_asymmetric(self):
        arr1 = np.asarray([[3, 3], [2, 5], [0, 6]], dtype=np.float32)
        arr2 = np.asarray([[3, 2], [2, 6], [3, 1]], dtype=np.float32)

        values, action_probs = calculate_nash_equilibrium_python(arr1, arr2)
        print(values)
        print(action_probs)

    def test_degenerate(self):
        arr1 = np.asarray([[3, 3], [2, 5], [0, 6]], dtype=np.float32)
        arr2 = np.asarray([[3, 3], [2, 6], [3, 1]], dtype=np.float32)

        values, action_probs = calculate_nash_equilibrium_python(arr1, arr2)
        print(values)
        print(action_probs)

    def test_floating_point(self):
        arr1 = np.asarray([[3, 3], [2, 5], [0, 6]], dtype=np.float32) / 10.0
        arr2 = np.asarray([[3, 3], [2, 6], [3, 1]], dtype=np.float32) / 10.0
        start = time.time()
        values, action_probs = calculate_nash_equilibrium_python(arr1, arr2)
        print(time.time() - start)
        print(values)
        print(action_probs)

    def test_temp(self):
        arr1 = np.asarray([[5, 1], [1, 1]], dtype=np.float32)
        arr2 = np.asarray([[5, 1], [1, 1]], dtype=np.float32)
        start = time.time()
        values, action_probs = calculate_nash_equilibrium_python(arr1, arr2)
        print(time.time() - start)
        print(values)
        print(action_probs)

    def test_big(self):
        dim = 8
        arr1 = np.random.rand(dim, dim)
        arr2 = np.random.rand(dim, dim)
        start = time.time()
        values, action_probs = calculate_nash_equilibrium_python(arr1, arr2)
        print(time.time() - start)
        print(values)
        print(action_probs)
