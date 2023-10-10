import unittest

import numpy as np

from src.equilibria.quantal import compute_qne_equilibrium


class TestQNE(unittest.TestCase):
    def test_qne_prisoners_dilemma(self):
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, 1], [5, 0], [0, 5], [3, 3]])
        values, policies = compute_qne_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_values,
            leader=0,
            num_iterations=100,
            temperature=3,
            random_prob=0,
        )
        print(f"{values=}")
        print(f"{policies=}")
        self.assertGreater(values[0], values[1])

    def test_unbalanced(self):
        # nash equilibrium would be [[1/6, 5/6], [1/4, 3/4]]
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, 5], [3, 0], [4, 2], [2, 3]])
        values, policies = compute_qne_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_values,
            leader=0,
            num_iterations=100,
            temperature=3,
            random_prob=0.1,
        )
        print(f"Leader 0:")
        print(f"{values=}")
        print(f"{policies=}")

        values, policies = compute_qne_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_values,
            leader=1,
            num_iterations=100,
            temperature=3,
            random_prob=0.1,
        )
        print(f"Leader 1:")
        print(f"{values=}")
        print(f"{policies=}")

    def test_paper_example(self):
        # from: https://ojs.aaai.org/index.php/AAAI/article/view/16701, Game 1
        available_actions = [[0, 1], [0, 1, 2, 3]]
        joint_action_list = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3),]
        joint_action_values = np.asarray([[-4, 4], [-5, 5], [8, -8], [4, -4], [-5, 5], [-4, 4], [-4, 4],[8, -8]],
                                         dtype=float)
        values, policies = compute_qne_equilibrium(
            available_actions=available_actions,
            joint_action_list=joint_action_list,
            joint_action_value_arr=joint_action_values,
            leader=0,
            num_iterations=100,
            temperature=0.92,
        )
        print(f"{values=}")
        print(f"{policies=}")
