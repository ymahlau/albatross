import unittest

import numpy as np
import torch
from src.game.values import UtilityNorm

from src.misc.replay_buffer import ReplayBufferConfig, ReplayBuffer, BufferInputSample
from src.supervised.loss import compute_utility_loss


class TestLoss(unittest.TestCase):
    def test_zero_sum_loss(self):
        buffer_cfg = ReplayBufferConfig(obs_shape=(5, 3), num_actions=2, num_players=2, num_symmetries=1, capacity=8,)
        buffer = ReplayBuffer(buffer_cfg)
        data1_size = 4
        data1 = BufferInputSample(
            obs=np.ones(shape=(data1_size, 5, 3), dtype=float),
            policies=np.ones(shape=(data1_size, 2), dtype=float),
            values=np.ones(shape=(data1_size, 1), dtype=float),
            turns=np.ones(shape=(data1_size, 1), dtype=float),
            player=np.ones(shape=(data1_size, 1), dtype=float),
            symmetry=np.ones(shape=(data1_size, 1), dtype=float),
        )
        data2_size = 4
        data2 = BufferInputSample(
            obs=np.ones(shape=(data2_size, 5, 3), dtype=float) + 1,
            policies=np.ones(shape=(data2_size, 2), dtype=float) + 1,
            values=np.ones(shape=(data2_size, 1), dtype=float) + 1,
            turns=np.ones(shape=(data2_size, 1), dtype=float) + 1,
            player=np.ones(shape=(data2_size, 1), dtype=float) + 1,
            symmetry=np.ones(shape=(data2_size, 1), dtype=float) + 1,
        )
        buffer.put(data1)
        buffer.put(data2)
        sample = buffer.sample(4, grouped=True)
        loss = compute_utility_loss(torch.tensor(sample.values), UtilityNorm.ZERO_SUM)
        self.assertGreater(loss.item(), 0)
        sample2 = buffer.sample(8, grouped=True)
        loss2 = compute_utility_loss(torch.tensor(sample2.values), UtilityNorm.ZERO_SUM)
        self.assertAlmostEqual(10, loss2.item(), places=5)
