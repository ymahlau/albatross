import unittest

import numpy as np

from src.search.utils import action_indices_from_mask, q_list_from_mask


class TestUtils(unittest.TestCase):
    def test_action_indices_from_mask(self):
        action_arr = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
        mask = np.asarray([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 0],
        ], dtype=bool)
        action_indices = action_indices_from_mask(
            actions=action_arr,
            is_invalid=mask,
        )
        gt = [0, 1, 1, 1, 0, 1, 1, 2]
        for i in range(8):
            self.assertEqual(gt[i], action_indices[i])

    def test_q_list_from_mask(self):
        q_arr = np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ], dtype=float)
        valid_mask = np.asarray([
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 1],
        ], dtype=bool)
        valid_q_list = q_list_from_mask(
            q_arr=q_arr,
            is_valid=valid_mask,
        )
        print(valid_q_list)
