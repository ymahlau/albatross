import unittest

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from src.game.actions import sample_individual_actions
from src.modelling.mle import compute_all_likelihoods


class TestLikelihood(unittest.TestCase):
    def test_likelihood_small(self):
        num_actions = 3
        num_utils = 50
        utilities = np.random.uniform(0, 1, (num_utils, num_actions))
        qs = [list(utils) for utils in utilities]

        actions = [np.random.choice(num_actions) for _ in range(num_utils)]
        actions = list(sample_individual_actions(utilities, 10))

        min_temp, max_temp, resolution = 0, 100, 200
        all_likelihoods = compute_all_likelihoods(
            min_temp=min_temp,
            max_temp=max_temp,
            chosen_actions=actions,
            utilities=qs,
            resolution=resolution,
        )
        # y = all_likelihoods
        y = np.exp(all_likelihoods)

        x = np.linspace(min_temp, max_temp, resolution)
        seaborn.set_theme(style='whitegrid')
        plt.clf()
        plt.plot(x, y)
        plt.show()
