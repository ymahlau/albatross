import unittest

import numpy as np

from src.equilibria.quantal import compute_qse_equilibrium


class TestQSE(unittest.TestCase):
    def test_qse_matching_pennies(self):
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, -1], [-1, 1], [-1, 1], [1, -1]], dtype=np.float32)
        values, policies = compute_qse_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_values,
            leader=0,
            temperature=3,
        )
        print(f"{values=}")
        print(f"{policies=}")
        self.assertAlmostEqual(0, values[0])
        self.assertAlmostEqual(0, values[1])
        self.assertAlmostEqual(0.5, policies[0][0])
        self.assertAlmostEqual(0.5, policies[0][1])
        self.assertAlmostEqual(0.5, policies[1][0])
        self.assertAlmostEqual(0.5, policies[1][1])

    def test_qse_prisoners_dilemma(self):
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, 1], [5, 0], [0, 5], [3, 3]], dtype=np.float32)
        values, policies = compute_qse_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_values,
            leader=0,
            temperature=3,
        )
        print(f"{values=}")
        print(f"{policies=}")
        self.assertAlmostEqual(1, policies[0][0])
        self.assertAlmostEqual(0, policies[0][1])

    def test_asymmetric(self):
        available_actions = [[0, 1, 2], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        joint_action_values = np.asarray([[3, 3], [3, 2], [2, 2], [5, 6], [0, 3], [6, 1]], dtype=float)
        values, policies = compute_qse_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_values,
            leader=0,
            temperature=3,
        )
        print(f"{values=}")
        print(f"{policies=}")

    def test_paper_example(self):
        # from: https://ojs.aaai.org/index.php/AAAI/article/view/16701, Game 1
        available_actions = [[0, 1], [0, 1, 2, 3]]
        joint_action_list = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3),]
        joint_action_values = np.asarray([[-4, 4], [-5, 5], [8, -8], [4, -4], [-5, 5], [-4, 4], [-4, 4],[8, -8]],
                                         dtype=float)
        values, policies = compute_qse_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_values,
            leader=0,
            temperature=0.92,
            grid_size=10000,
            num_iterations=20,
        )
        print(f"{values=}")
        print(f"{policies=}")
