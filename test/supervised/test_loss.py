import unittest

import torch

from src.misc.replay_buffer import ReplayBufferConfig, ReplayBuffer, BufferInputSample
from src.supervised.loss import compute_zero_sum_loss


class TestLoss(unittest.TestCase):
    def test_zero_sum_loss(self):
        buffer_cfg = ReplayBufferConfig(obs_shape=(5, 3), num_actions=2, num_players=2, num_symmetries=1, capacity=8,
                                        single_temperature=True)
        buffer = ReplayBuffer(buffer_cfg)
        data1_size = 4
        data1 = BufferInputSample(
            obs=torch.ones(size=(data1_size, 5, 3), dtype=torch.float32),
            policies=torch.ones(size=(data1_size, 2), dtype=torch.float32),
            values=torch.ones(size=(data1_size, 1), dtype=torch.float32),
            game_lengths=torch.ones(size=(data1_size, 1), dtype=torch.float32),
            turns=torch.ones(size=(data1_size, 1), dtype=torch.float32),
            player=torch.ones(size=(data1_size, 1), dtype=torch.float32),
            symmetry=torch.ones(size=(data1_size, 1), dtype=torch.float32),
            temperature=torch.ones(size=(data1_size, 1), dtype=torch.float32),
        )
        data2_size = 4
        data2 = BufferInputSample(
            obs=torch.ones(size=(data2_size, 5, 3), dtype=torch.float32) + 1,
            policies=torch.ones(size=(data2_size, 2), dtype=torch.float32) + 1,
            values=torch.ones(size=(data2_size, 1), dtype=torch.float32) + 1,
            game_lengths=torch.ones(size=(data2_size, 1), dtype=torch.float32) + 1,
            turns=torch.ones(size=(data2_size, 1), dtype=torch.float32) + 1,
            player=torch.ones(size=(data2_size, 1), dtype=torch.float32) + 1,
            symmetry=torch.ones(size=(data2_size, 1), dtype=torch.float32) + 1,
            temperature=torch.ones(size=(data1_size, 1), dtype=torch.float32) + 1,
        )
        buffer.put(data1)
        buffer.put(data2)
        sample = buffer.sample(4, grouped=True)
        loss = compute_zero_sum_loss(sample.values)
        self.assertGreater(loss.item(), 0)
        sample2 = buffer.sample(8, grouped=True)
        loss2 = compute_zero_sum_loss(sample2.values)
        self.assertAlmostEqual(10, loss2.item(), places=5)
