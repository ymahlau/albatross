import unittest

import torch
from torch.utils.data import DataLoader

from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.initialization import get_game_from_config, buffer_config_from_game
from src.misc.replay_buffer import ReplayBuffer, BufferInputSample
from src.network.initialization import get_network_from_config
from src.network.resnet import ResNetConfig3x3
from src.supervised.episode import single_episode


class TestEpisode(unittest.TestCase):
    def test_episode_simple(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        net_cfg = ResNetConfig3x3(game_cfg=game_cfg)
        game = get_game_from_config(game_cfg)
        net = get_network_from_config(net_cfg)
        capacity = 64

        buffer_cfg = buffer_config_from_game(game, capacity, single_temperature=True)
        buffer = ReplayBuffer(buffer_cfg)
        for _ in range(int(capacity / 2)):
            obs, _, _ = game.get_obs()
            sample = BufferInputSample(
                obs=torch.tensor(obs, dtype=torch.float32),
                values=torch.ones((2, 1), dtype=torch.float32),
                policies=torch.ones((2, 4), dtype=torch.float32) / 4,
                game_lengths=torch.ones((2, 1), dtype=torch.float32) * 2,
                turns=torch.zeros((2, 1), dtype=torch.float32),
                player=torch.zeros((2, 1), dtype=torch.float32),
                symmetry=torch.zeros((2, 1), dtype=torch.float32),
                temperature=torch.ones(size=(2, 1), dtype=torch.float32),
            )
            buffer.put(sample)
        # compute loss
        val_loader = DataLoader(
            buffer,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )
        val_dict = single_episode(
            net=net,
            loader=val_loader,
            optim=None,
            device=torch.device('cpu'),
            use_zero_sum_loss=True,
            mode='valid',
            mse_policy_loss=True,
        )
        for k, v in val_dict.items():
            self.assertGreater(v, 0)
