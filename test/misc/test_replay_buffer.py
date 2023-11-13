import dataclasses
import unittest
from pathlib import Path

import numpy as np

from src.misc.replay_buffer import ReplayBufferConfig, ReplayBuffer, BufferInputSample


class TestReplayBuffer(unittest.TestCase):
    def test_simple(self):
        buffer_cfg = ReplayBufferConfig(
            obs_shape=(5, 3),
            num_actions=2,
            num_players=2,
            num_symmetries=1,
            capacity=8,
        )
        buffer = ReplayBuffer(buffer_cfg)

        data1_size = 6
        data1 = BufferInputSample(
            obs=np.ones(shape=(data1_size, 5, 3), dtype=float),
            policies=np.ones(shape=(data1_size, 2), dtype=float),
            values=np.ones(shape=(data1_size, 1), dtype=float),
            turns=np.ones(shape=(data1_size, 1), dtype=float),
            player=np.ones(shape=(data1_size, 1), dtype=float),
            symmetry=np.ones(shape=(data1_size, 1), dtype=float),
        )
        buffer.put(data1)
        self.assertFalse(buffer.full())

        size = 2
        sample = buffer.sample(size)
        self.assertEqual(size, sample.obs.shape[0])
        self.assertEqual(size, sample.values.shape[0])
        self.assertEqual(size, sample.policies.shape[0])
        self.assertTrue(np.all(sample.obs == 1).item())
        self.assertTrue(np.all(sample.values == 1).item())
        self.assertTrue(np.all(sample.policies == 1).item())

        data2_size = 4
        data2 = BufferInputSample(
            obs=np.ones(shape=(data2_size, 5, 3), dtype=float),
            policies=np.ones(shape=(data2_size, 2), dtype=float),
            values=np.ones(shape=(data2_size, 1), dtype=float),
            turns=np.ones(shape=(data2_size, 1), dtype=float),
            player=np.ones(shape=(data2_size, 1), dtype=float),
            symmetry=np.ones(shape=(data2_size, 1), dtype=float),
        )
        buffer.put(data2)
        self.assertTrue(buffer.full())

        size = 2
        sample = buffer.sample(size)
        self.assertEqual(size, sample.obs.shape[0])
        self.assertEqual(size, sample.values.shape[0])
        self.assertEqual(size, sample.policies.shape[0])
        self.assertTrue(np.all(sample.obs == 1).item())
        self.assertTrue(np.all(sample.values == 1).item())
        self.assertTrue(np.all(sample.policies == 1).item())

    def test_large_emplace(self):
        buffer_cfg = ReplayBufferConfig(obs_shape=(5, 3), num_actions=2, num_players=2, num_symmetries=1, capacity=8)
        buffer = ReplayBuffer(buffer_cfg)

        data1_size = 10
        data1 = BufferInputSample(
            obs=np.ones(shape=(data1_size, 5, 3), dtype=float),
            policies=np.ones(shape=(data1_size, 2), dtype=float),
            values=np.ones(shape=(data1_size, 1), dtype=float),
            turns=np.ones(shape=(data1_size, 1), dtype=float),
            player=np.ones(shape=(data1_size, 1), dtype=float),
            symmetry=np.ones(shape=(data1_size, 1), dtype=float),
        )
        buffer.put(data1)
        self.assertTrue(buffer.full())

        size = 2
        sample = buffer.sample(size)
        self.assertEqual(size, sample.obs.shape[0])
        self.assertEqual(size, sample.values.shape[0])
        self.assertEqual(size, sample.policies.shape[0])
        self.assertTrue(np.all(sample.obs == 1).item())
        self.assertTrue(np.all(sample.values == 1).item())
        self.assertTrue(np.all(sample.policies == 1).item())

    def test_grouped(self):
        buffer_cfg = ReplayBufferConfig(obs_shape=(5, 3), num_actions=2, num_players=2, num_symmetries=1, capacity=8)
        buffer = ReplayBuffer(buffer_cfg)

        data1_size = 6
        data1 = BufferInputSample(
            obs=np.ones(shape=(data1_size, 5, 3), dtype=float),
            policies=np.ones(shape=(data1_size, 2), dtype=float),
            values=np.ones(shape=(data1_size, 1), dtype=float),
            turns=np.ones(shape=(data1_size, 1), dtype=float),
            player=np.ones(shape=(data1_size, 1), dtype=float),
            symmetry=np.ones(shape=(data1_size, 1), dtype=float),
        )
        buffer.put(data1)
        self.assertFalse(buffer.full())

        size = 2
        sample = buffer.sample(size, grouped=True)
        self.assertEqual(2, sample.obs.shape[0])
        self.assertEqual(2, sample.values.shape[0])
        self.assertEqual(2, sample.policies.shape[0])
        self.assertTrue(np.all(sample.obs == 1).item())
        self.assertTrue(np.all(sample.values == 1).item())
        self.assertTrue(np.all(sample.policies == 1).item())

    def test_buffer_save_load(self):
        save_path = Path(__file__).parent / 'temp_buffer.pt'

        buffer_cfg = ReplayBufferConfig(obs_shape=(5, 3), num_actions=2, num_players=2, num_symmetries=1, capacity=8)
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

        buffer.put(data1)
        buffer.save(save_path)
        buffer2 = ReplayBuffer.from_saved_file(save_path)
        b_dict = dataclasses.asdict(buffer.content)
        for k, v in dataclasses.asdict(buffer2.content).items():
            if not isinstance(v, np.ndarray):
                continue
            self.assertTrue(np.all(v == b_dict[k]).item())

    def test_shuffle(self):
        buffer_cfg = ReplayBufferConfig(obs_shape=(5, 3), num_actions=2, num_players=3, num_symmetries=1, capacity=8)
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

        buffer.shuffle()
        self.assertTrue(buffer.full())

    def test_shuffle_grouped(self):
        buffer_cfg = ReplayBufferConfig(obs_shape=(5, 3), num_actions=2, num_players=2, num_symmetries=1, capacity=8)
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

        buffer.shuffle()
        self.assertTrue(buffer.full())

    def test_retrieve(self):
        buffer_cfg = ReplayBufferConfig(obs_shape=(5, 3), num_actions=2, num_players=2, num_symmetries=1, capacity=8)
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
        sample = buffer.retrieve(3, 5)
        self.assertEqual(2, sample.obs.shape[0])
        sample2 = buffer.retrieve(1, 3, grouped=True)
        self.assertEqual(4, sample2.obs.shape[0])

    def test_buffer_decrease_capacity(self):
        buffer_cfg = ReplayBufferConfig(obs_shape=(5, 3), num_actions=2, num_players=3, num_symmetries=1, capacity=8)
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
        buffer.decrease_capacity(6)
        self.assertEqual(6, len(buffer))
        self.assertTrue(buffer.full())
        buffer.decrease_capacity(4)
        self.assertEqual(4, len(buffer))
        self.assertTrue(buffer.full())
