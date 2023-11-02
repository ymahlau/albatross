import unittest

import numpy as np

from src.misc.utils import random_argmax, symmetric_cartesian_product


class TestUtils(unittest.TestCase):
    def test_random_argmax_simple(self):
        arr = np.array([1, 2, 3, 2, 3, 2, 1, 3, 2])
        result = random_argmax(arr)
        self.assertTrue(result in [2, 4, 7])

    def test_symmetric_cartesian(self):
        names = ["a", "b", "c", "d", "e", "f"]
        todo_list = symmetric_cartesian_product(
            name_list=names,
            players_per_game=2,
        )
        print(todo_list)
        self.assertEqual(15, len(todo_list))

    def test_symmetric_cartesian_4_player(self):
        names = ["a", "b", "c", "d", "e"]
        todo_list = symmetric_cartesian_product(
            name_list=names,
            players_per_game=4,
        )
        print(todo_list)
        self.assertEqual(65, len(todo_list))