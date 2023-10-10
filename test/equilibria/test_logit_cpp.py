import itertools
import time
import unittest
from random import random

import numpy as np

from src.equilibria.logit import compute_logit_equilibrium, SbrMode


class TestLogitEquilibriumCPP(unittest.TestCase):
    def test_prisoner_dilemma(self):
        # pd
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, 1], [5, 0], [0, 5], [3, 3]])
        for mode in SbrMode:
            print(f"{mode=}")
            values, policies, error = compute_logit_equilibrium(
                available_actions,
                joint_action_list,
                joint_action_values,
                100,
                0,
                [7.0, 7.0],
                sbr_mode=mode,
            )
            print(f"{policies=}")
            print(f"{error=}")
            print(f"################")
            self.assertAlmostEqual(values[0], values[1])
            self.assertGreater(policies[0][0].item(), policies[0][1].item())
            self.assertGreater(policies[1][0].item(), policies[1][1].item())

    def test_unbalanced(self):
        # nash equilibrium would be [[1/6, 5/6], [1/4, 3/4]]
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, 5], [3, 0], [4, 2], [2, 3]])
        values, policies, pol_err = compute_logit_equilibrium(
            available_actions,
            joint_action_list,
            joint_action_values,
            100,
            0,
            [7.0],
            hp_0=0.5,
            sbr_mode=SbrMode.EMA,
        )
        print(values)
        print(policies)

    def test_uniqueness(self):
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, 5], [3, 0], [4, 2], [2, 3]])
        equilibrium_list = []
        start_time = time.time()
        for _ in range(100):
            p0_policy, p1_policy = np.random.rand(2), np.random.rand(2)
            p0_policy, p1_policy = p0_policy / np.sum(p0_policy), p1_policy / np.sum(p1_policy)
            values, policies, pol_err = compute_logit_equilibrium(
                available_actions,
                joint_action_list,
                joint_action_values,
                2000,
                0,
                [4.0],
                hp_0=0.9,
                sbr_mode=SbrMode.MSA,
                initial_policies=[p0_policy, p1_policy]
            )
            print(f"{policies=}, {values=}")
            if not equilibrium_list:
                equilibrium_list.append(policies)
            else:
                for eq in equilibrium_list:
                    if np.any(np.abs(eq[0] - policies[0]) > 0.01) or np.any(np.abs(eq[1] - policies[1]) > 0.01):
                        equilibrium_list.append(policies)
                        break
        print(time.time() - start_time)
        if len(equilibrium_list) > 1:
            print(equilibrium_list)
        self.assertEqual(1, len(equilibrium_list))

    def test_non_uniqueness(self):
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[5, 5], [0, 0], [0, 0], [5, 5]])
        # mixed equilibrium
        v, p0, err0 = compute_logit_equilibrium(available_actions, joint_action_list, joint_action_values, 1000, 0,
                                                [7.0])
        for player in range(2):
            for action in range(2):
                self.assertAlmostEqual(0.5, p0[player][action].item(), places=2)
        # almost pure equilibrium 1
        initial_policy_1 = [np.asarray([0.1, 0.9]) for _ in range(2)]
        v, p1, err1 = compute_logit_equilibrium(available_actions, joint_action_list, joint_action_values, 1000, 0,
                                                [7.0], hp_0=0.9, sbr_mode=SbrMode.EMA,
                                                initial_policies=initial_policy_1)
        for player in range(2):
            for action in range(2):
                self.assertNotAlmostEqual(0.5, p1[player][action].item(), places=2)
        # almost pure equilibrium 2
        initial_policy_1 = [np.asarray([0.9, 0.1]) for _ in range(2)]
        v, p2, err2 = compute_logit_equilibrium(available_actions, joint_action_list, joint_action_values, 1000, 0,
                                                [7.0], hp_0=0.9, sbr_mode=SbrMode.EMA,
                                                initial_policies=initial_policy_1)
        for player in range(2):
            for action in range(2):
                self.assertNotAlmostEqual(0.5, p2[player][action].item(), places=2)
        print(p0)
        print(p1)
        print(p2)

    def test_multiplayer(self):
        available_actions = [[0, 1], [0, 1], [0, 1]]
        joint_action_list = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        x = 20
        joint_action_values = np.asarray([[x, x, x], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                          [x, x, x]])
        # mixed equilibrium
        v, policies, err = compute_logit_equilibrium(available_actions, joint_action_list, joint_action_values, 300, 0,
                                                     [7.0], hp_0=0.5, sbr_mode=SbrMode.EMA, )
        for player in range(3):
            for action in range(2):
                self.assertAlmostEqual(0.5, policies[player][action].item(), places=2)
        # almost pure equilibrium 1
        initial_policy_1 = [np.asarray([0.1, 0.9]) for _ in range(3)]
        v, policies, err = compute_logit_equilibrium(available_actions, joint_action_list, joint_action_values, 300, 0,
                                                     [7.0], hp_0=0.5, sbr_mode=SbrMode.EMA,
                                                     initial_policies=initial_policy_1)
        for player in range(3):
            for action in range(2):
                self.assertNotAlmostEqual(0.5, policies[player][action].item(), places=2)
        # almost pure equilibrium 2
        initial_policy_1 = [np.asarray([0.9, 0.1]) for _ in range(3)]
        v, policies, err = compute_logit_equilibrium(available_actions, joint_action_list, joint_action_values, 300, 0,
                                                     [7.0], hp_0=0.5, sbr_mode=SbrMode.EMA,
                                                     initial_policies=initial_policy_1)
        for player in range(3):
            for action in range(2):
                self.assertNotAlmostEqual(0.5, policies[player][action].item(), places=2)

    def test_unbalanced_epsilon(self):
        # nash equilibrium would be [[1/6, 5/6], [1/4, 3/4]]
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        epsilon = 1e-6
        num_iterations = int(1e6)
        joint_action_values = np.asarray([[1, 5], [3, 0], [4, 2], [2, 3]])
        values, policies, err = compute_logit_equilibrium(available_actions, joint_action_list, joint_action_values,
                                                          num_iterations, epsilon,
                                                          [7.0], hp_0=0.5,
                                                          sbr_mode=SbrMode.EMA)
        print(f"{values=}")
        print(f"{policies=}")

    def test_multiple_temperatures_pd(self):
        # pd
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, 1], [5, 0], [0, 5], [3, 3]])
        values, policies, err = compute_logit_equilibrium(
            available_actions,
            joint_action_list,
            joint_action_values,
            100,
            0,
            [0.5, 2],
        )
        print(policies)
        print(values)
        print(err)
        # self.assertAlmostEqual(values[0], values[1])
        # self.assertGreater(policies[0][0], policies[0][1])
        # self.assertGreater(policies[1][0], policies[1][1])

    def test_two_actions_zero_sum(self):
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, -1], [-1, 1], [-0.5, 0.5], [0.5, -0.5]])

        for _ in range(100):
            p0_policy, p1_policy = np.random.rand(2), np.random.rand(2)
            p0_policy, p1_policy = p0_policy / np.sum(p0_policy), p1_policy / np.sum(p1_policy)
            values, policies, err = compute_logit_equilibrium(available_actions, joint_action_list, joint_action_values,
                                                              300, 0, [8.0],
                                                              sbr_mode=SbrMode.MSA,
                                                              initial_policies=[p0_policy, p1_policy])
            print(f"{policies=}, {values=}")

    def test_four_actions_zero_sum(self):
        available_actions = [[0, 1, 2, 3], [0, 1, 2, 3]]
        joint_action_list = list(itertools.product(*available_actions))
        # joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values_single = [random() for _ in joint_action_list]
        joint_action_values = np.asarray([[x, -x] for x in joint_action_values_single])
        # joint_action_values = np.asarray([[1, -1], [-1, 1], [-0.5, 0.5], [0.5, -0.5]])

        for _ in range(100):
            p0_policy, p1_policy = np.random.rand(4), np.random.rand(4)
            p0_policy, p1_policy = p0_policy / np.sum(p0_policy), p1_policy / np.sum(p1_policy)
            values, policies, err = compute_logit_equilibrium(available_actions, joint_action_list, joint_action_values,
                                                              300, 0, [8.0],
                                                              sbr_mode=SbrMode.MSA,
                                                              initial_policies=[p0_policy, p1_policy])
            print(f"{policies=}, {values=}")
