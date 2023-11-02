import unittest

import numpy as np

from src.game.values import apply_utility_norm, UtilityNorm


class TestValue(unittest.TestCase):
    def test_values_simple(self):
        values = np.asarray([[0, 2], [4, 3]])
        normed = apply_utility_norm(values, norm=UtilityNorm.ZERO_SUM)
        self.assertEqual(-1, normed[0, 0])
        self.assertEqual(1, normed[0, 1])

    def test_values_shape(self):
        values = np.asarray([5, 3])
        normed = apply_utility_norm(values, norm=UtilityNorm.ZERO_SUM)
        self.assertEqual(1, normed[0])
        self.assertEqual(-1, normed[1])
