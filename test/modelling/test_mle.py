import unittest

from src.modelling.mle import compute_temperature_mle, compute_all_likelihoods


class TestMLE(unittest.TestCase):
    def test_mle_small(self):
        qs = [
            [1, 1.001, 0.5]
        ]
        actions = [1]
        estimate = compute_temperature_mle(
            min_temp=1,
            max_temp=10,
            chosen_actions=actions,
            utilities=qs,
            num_iterations=1000,
            use_line_search=False,
        )
        print(estimate)
        self.assertAlmostEqual(estimate, 10)

    def test_mle_multiple(self):
        qs = [
            [1, 1.001, 0.999],
            [1, 1.001, 0.999],
            [1, 1.001, 0.999],
            [1, 1.001, 0.999],
        ]
        actions = [0, 0, 0, 1]
        estimate = compute_temperature_mle(
            min_temp=1,
            max_temp=10,
            chosen_actions=actions,
            utilities=qs,
            num_iterations=1000,
            use_line_search=False,
        )
        all_likelihoods = compute_all_likelihoods(
            min_temp=1,
            max_temp=19,
            chosen_actions=actions,
            utilities=qs,
            resolution=100,
        )
        print(f"{all_likelihoods=}")
        print(f"{all_likelihoods.index(max(all_likelihoods))=}")
        print(f"{estimate=}")

    def test_line_search(self):
        qs = [[1, -1] for _ in range(10)]
        actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        estimate_list = []
        for i in range(1, 20):
            estimate = compute_temperature_mle(
                min_temp=0.1,
                max_temp=10,
                chosen_actions=actions,
                utilities=qs,
                num_iterations=i,
                use_line_search=True,
            )
            estimate_list.append(estimate)
        all_lkl = compute_all_likelihoods(
            chosen_actions=actions,
            utilities=qs,
            min_temp=0.1,
            max_temp=10,
            resolution=20,
        )
        print(f"{all_lkl=}")
        print(f"{estimate_list=}")

