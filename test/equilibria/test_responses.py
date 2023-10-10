import unittest

import numpy as np

from src.equilibria.responses import values_from_policies


class TestResponses(unittest.TestCase):
    def test_value_from_policies(self):
        ja_vals = np.asarray([
            [1, -1],
            [2, -2],
            [3, -3],
            [4, -4],
        ], dtype=float)
        policies = [
            np.asarray([0.3, 0.7], dtype=float),
            np.asarray([0.9, 0.1], dtype=float)
        ]
        vals = values_from_policies(policies, ja_vals)
        print(f"{vals=}")
        self.assertEqual(2.5, vals[0].item())
        self.assertEqual(-2.5, vals[1].item())


