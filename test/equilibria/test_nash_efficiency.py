import itertools
import time
import unittest

import numpy as np

from src.equilibria.nash import calculate_nash_equilibrium


class TestNashEfficiency(unittest.TestCase):
    def test_random(self):
        num_runs = 100
        dim = 2

        python_time = 0
        cpp_time = 0
        available_actions = [list(range(dim)) for _ in range(2)]
        joint_action_list = list(itertools.product(*available_actions))
        for _ in range(num_runs):
            joint_action_values = np.random.rand(len(joint_action_list), 2)
            start_time = time.time()
            values, policies = calculate_nash_equilibrium(
                available_actions,
                joint_action_list,
                joint_action_values,
                use_cpp=True,
            )
            python_time += time.time() - start_time
        print(f"{cpp_time=}")
