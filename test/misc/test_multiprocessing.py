import unittest

from src.misc.multiprocessing import partition_indices


class TestMultiprocessing(unittest.TestCase):
    def test_partitioning_normal(self):
        partitions = partition_indices(
            start_idx=0,
            end_idx=105,
            min_size=5,
            max_num_partitions=9,
        )
        print(partitions)
        self.assertEqual(9, len(partitions))

    def test_partitioning_min_bound(self):
        partitions = partition_indices(
            start_idx=0,
            end_idx=105,
            min_size=35,
            max_num_partitions=12,
        )
        print(partitions)
        self.assertEqual(3, len(partitions))

        partitions = partition_indices(
            start_idx=0,
            end_idx=106,
            min_size=35,
            max_num_partitions=12,
        )
        print(partitions)
        self.assertEqual(3, len(partitions))

        partitions = partition_indices(
            start_idx=0,
            end_idx=104,
            min_size=35,
            max_num_partitions=12,
        )
        print(partitions)
        self.assertEqual(2, len(partitions))
