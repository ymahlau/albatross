import copy
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ReplayBufferConfig:
    obs_shape: tuple[int, ...]
    num_actions: int
    num_players: int
    num_symmetries: int
    capacity: int
    single_temperature: bool


@dataclass
class BufferOutputSample:  # data structure used when sampling from buffer
    """
    shape is (n, *channel_shape)
    """
    obs: torch.Tensor
    values: torch.Tensor
    policies: torch.Tensor
    game_lengths: torch.Tensor
    temperature: Optional[torch.Tensor]


@dataclass
class BufferInputSample:  # data structure used for inserting new data into buffer
    # shape: (n, channel_shape)
    obs: torch.Tensor  # channels shape: obs_shape
    values: torch.Tensor  # channels shape: 1
    policies: torch.Tensor  # channels shape: num_actions
    game_lengths: torch.Tensor  # channels shape: 1
    turns: torch.Tensor  # channels shape: 1
    player: torch.Tensor  # channels shape: 1
    symmetry: torch.Tensor  # channels shape: 1
    temperature: Optional[torch.Tensor]  # channel shape: 1 or num_player-1


@dataclass
class ReplayBufferContent:
    # data channels: observation, value, policy, game length, sbr-temperature
    dc_obs: torch.Tensor
    dc_val: torch.Tensor
    dc_pol: torch.Tensor
    dc_len: torch.Tensor
    dc_temp: torch.Tensor
    # metadata channels: game_id, turns_played, player, symmetry
    mc_ids: torch.Tensor
    mc_turns: torch.Tensor
    mc_player: torch.Tensor
    mc_symmetry: torch.Tensor
    # control channels: start indices, offsets
    capacity_reached: bool
    idx: int
    game_idx: int

    @classmethod
    def empty(
            cls,
            obs_shape: tuple[int, ...],
            capacity: int, num_actions: int,
            num_players: int,
            single_temperature: bool,
    ) -> "ReplayBufferContent":
        obs_channel_shape = tuple([capacity] + list(obs_shape))
        temp_channel_size = 1 if single_temperature else num_players - 1
        return cls(
            dc_obs=torch.zeros(size=obs_channel_shape, dtype=torch.float32, requires_grad=False),
            dc_val=torch.zeros(size=(capacity, 1), dtype=torch.float32, requires_grad=False),
            dc_pol=torch.zeros(size=(capacity, num_actions), dtype=torch.float32, requires_grad=False),
            dc_len=torch.zeros(size=(capacity, 1), dtype=torch.float32, requires_grad=False),
            mc_ids=torch.zeros(size=(capacity, 1), dtype=torch.float32, requires_grad=False),
            mc_turns=torch.zeros(size=(capacity, 1), dtype=torch.float32, requires_grad=False),
            mc_player=torch.zeros(size=(capacity, 1), dtype=torch.float32, requires_grad=False),
            mc_symmetry=torch.zeros(size=(capacity, 1), dtype=torch.float32, requires_grad=False),
            capacity_reached=False,
            idx=0,
            game_idx=0,
            dc_temp=torch.zeros(size=(capacity, temp_channel_size), dtype=torch.float32, requires_grad=False),
        )


class ReplayBuffer(Dataset):
    """
    Replay buffer for experience replay.
    """

    def __init__(
            self,
            cfg: ReplayBufferConfig,
            content: Optional[ReplayBufferContent] = None,  # used to init the buffer with content
            game_cfg: Optional[Any] = None,  # optional config for saving and easier setup later
    ):
        if cfg.capacity % 2 != 0 and cfg.num_players == 2:
            raise ValueError(f"Please use even capacity for two players, it enables grouped sampling")
        # the buffer contains multiple tensors, each in a different channel
        super().__init__()
        self.cfg = cfg
        self.game_cfg = game_cfg
        self.temp_channel_size = 1 if self.cfg.single_temperature else self.cfg.num_players - 1
        if content is None:
            self.content = ReplayBufferContent.empty(
                obs_shape=self.cfg.obs_shape,
                capacity=self.cfg.capacity,
                num_actions=self.cfg.num_actions,
                num_players=self.cfg.num_players,
                single_temperature=self.cfg.single_temperature,
            )
        else:
            self.content = content

    def __len__(self):
        return self.cfg.capacity if self.content.capacity_reached else self.content.idx

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self.content.dc_obs[index]
        values=self.content.dc_val[index]
        policies=self.content.dc_pol[index]
        game_lengths=self.content.dc_len[index]
        return values, policies, obs, game_lengths

    def get_grouped(self, index: int) -> BufferOutputSample:
        if self.cfg.num_players != 2:
            raise Exception(f"Grouping with more than two players not implemented")
        if index >= len(self) / 2:
            raise ValueError(f"Index too large for grouped retrieval: {index=}, {len(self)}")
        sample = BufferOutputSample(
            obs=self.content.dc_obs[2 * index:2 * index + 2],
            values=self.content.dc_val[2 * index:2 * index + 2],
            policies=self.content.dc_pol[2 * index:2 * index + 2],
            game_lengths=self.content.dc_len[2 * index:2 * index + 2],
            temperature=self.content.dc_temp[2 * index:2 * index + 2],
        )
        return sample

    def sample(self, sample_size: int, grouped: bool = False, temperature: bool = False) -> BufferOutputSample:
        if grouped:
            if self.cfg.num_players != 2:
                raise ValueError(f"grouped sampling not implemented for more than two players")
            if sample_size % 2 != 0:
                raise ValueError(f"sample size needs to be even for grouped sampling")
            sampled_idx = np.random.choice(math.floor(len(self) / 2), size=math.floor(sample_size / 2), replace=False)
            obs = torch.stack([self.content.dc_obs[2 * sampled_idx], self.content.dc_obs[2 * sampled_idx + 1]])
            obs = torch.reshape(obs, [2 * obs.shape[1], *self.cfg.obs_shape])
            values = torch.stack([self.content.dc_val[2 * sampled_idx], self.content.dc_val[2 * sampled_idx + 1]])
            values = torch.reshape(values, [2 * values.shape[1], 1])
            policies = torch.stack([self.content.dc_pol[2 * sampled_idx], self.content.dc_pol[2 * sampled_idx + 1]])
            policies = torch.reshape(policies, [2 * policies.shape[1], self.cfg.num_actions])
            lengths = torch.stack([self.content.dc_len[2 * sampled_idx], self.content.dc_len[2 * sampled_idx + 1]])
            lengths = torch.reshape(lengths, [2 * lengths.shape[1], 1])
            temps = None
            if temperature:
                temps = torch.stack([self.content.dc_temp[2 * sampled_idx], self.content.dc_temp[2 * sampled_idx + 1]])
                temps = torch.reshape(temps, [2 * temps.shape[1], self.temp_channel_size])
            sample = BufferOutputSample(
                obs=obs,
                values=values,
                policies=policies,
                game_lengths=lengths,
                temperature=temps,
            )
        else:
            sampled_idx = np.random.choice(len(self), size=(sample_size,), replace=False)
            temps = self.content.dc_temp[sampled_idx] if temperature else None
            sample = BufferOutputSample(
                obs=self.content.dc_obs[sampled_idx],
                values=self.content.dc_val[sampled_idx],
                policies=self.content.dc_pol[sampled_idx],
                game_lengths=self.content.dc_len[sampled_idx],
                temperature=temps,
            )
        return sample

    def sample_single(self, grouped: bool = False, temperature: bool = False) -> BufferOutputSample:
        sample = self.sample(1, grouped, temperature)
        return sample

    def retrieve(
            self,
            start_idx: int,
            end_idx: int,
            grouped: bool = False,
            temperature: bool = False,
    ) -> BufferOutputSample:
        if start_idx >= end_idx or start_idx < 0 or end_idx > len(self):
            raise ValueError(f"Invalid indices: {start_idx=}, {end_idx=}")
        if grouped:
            if end_idx * 2 + 1 > len(self):
                raise ValueError(f"Indices are invalid for grouped retrieval: {start_idx=}, {end_idx=}")
            start_idx *= 2
            end_idx *= 2
        temps = self.content.dc_temp[start_idx:end_idx] if temperature else None
        sample = BufferOutputSample(
            obs=self.content.dc_obs[start_idx:end_idx],
            values=self.content.dc_val[start_idx:end_idx],
            policies=self.content.dc_pol[start_idx:end_idx],
            game_lengths=self.content.dc_len[start_idx:end_idx],
            temperature=temps,
        )
        return sample

    def shuffle(self):
        if not self.full():
            raise ValueError("Shuffling currently only implemented for full buffer")
        if self.cfg.num_players == 2:
            # grouped shuffling keeping time steps together
            half_indices = np.random.permutation(math.floor(len(self) / 2))
            full_indices = np.stack([2 * half_indices, 2 * half_indices + 1], axis=-1)
            shuffled_indices = full_indices.reshape(-1)
        else:
            shuffled_indices = np.random.permutation(len(self))
        self.content.dc_obs = self.content.dc_obs[shuffled_indices]
        self.content.dc_val = self.content.dc_val[shuffled_indices]
        self.content.dc_pol = self.content.dc_pol[shuffled_indices]
        self.content.dc_len = self.content.dc_len[shuffled_indices]
        self.content.mc_ids = self.content.mc_ids[shuffled_indices]
        self.content.mc_turns = self.content.mc_turns[shuffled_indices]
        self.content.mc_player = self.content.mc_player[shuffled_indices]
        self.content.mc_symmetry = self.content.mc_symmetry[shuffled_indices]
        self.content.dc_temp = self.content.dc_temp[shuffled_indices]

    def full(self) -> bool:
        return self.content.capacity_reached

    def save(self, path: Path):
        content_dict = dataclasses.asdict(self.content)
        content_dict['cfg'] = self.cfg
        if self.game_cfg is not None:
            content_dict['game_cfg'] = self.game_cfg
        torch.save(content_dict, path)

    @classmethod
    def from_saved_file(
            cls,
            file_path: Path,
    ):
        saved_dict = torch.load(file_path)
        game_cfg = None
        if "game_cfg" in saved_dict:
            game_cfg = saved_dict["game_cfg"]
            del saved_dict['game_cfg']
        cfg = saved_dict['cfg']
        del saved_dict['cfg']
        content = ReplayBufferContent(**saved_dict)
        obj = cls(
            cfg=cfg,
            game_cfg=game_cfg,
            content=content,
        )
        return obj

    def clear(self):
        self.content.idx = 0
        self.content.capacity_reached = False

    def decrease_capacity(
            self,
            new_capacity: int,
    ):
        if new_capacity > len(self):
            raise ValueError(f"Invalid new capacity: {new_capacity}")
        if self.cfg.num_players == 2 and new_capacity % 2 != 0:
            raise ValueError(f"Please choose even capacity with two players")
        if self.cfg.num_players == 2:
            half_indices = np.random.choice(math.floor(len(self) / 2), size=math.floor(new_capacity / 2), replace=False)
            full_indices = np.stack([2 * half_indices, 2 * half_indices + 1], axis=-1)
            indices = full_indices.reshape(-1)
        else:
            indices = np.random.choice(len(self), size=new_capacity, replace=False)
        self.content.dc_obs = self.content.dc_obs[indices].clone()
        self.content.dc_val = self.content.dc_val[indices].clone()
        self.content.dc_pol = self.content.dc_pol[indices].clone()
        self.content.dc_len = self.content.dc_len[indices].clone()
        self.content.mc_ids = self.content.mc_ids[indices].clone()
        self.content.mc_turns = self.content.mc_turns[indices].clone()
        self.content.mc_player = self.content.mc_player[indices].clone()
        self.content.mc_symmetry = self.content.mc_symmetry[indices].clone()
        self.content.dc_temp = self.content.dc_temp[indices].clone()
        self.content.capacity_reached = True
        self.content.idx = 0
        self.cfg.capacity = new_capacity

    def put(self, data: BufferInputSample) -> None:
        # sanity checks, make sure sizes are correct
        self._validate_input(data)
        n = data.obs.shape[0]
        if n > self.cfg.capacity:
            # subsample data input
            data.obs = data.obs[:self.cfg.capacity]
            data.values = data.values[:self.cfg.capacity]
            data.policies = data.policies[:self.cfg.capacity]
            data.game_lengths = data.game_lengths[:self.cfg.capacity]
            data.turns = data.turns[:self.cfg.capacity]
            data.player = data.player[:self.cfg.capacity]
            data.symmetry = data.symmetry[:self.cfg.capacity]
            if data.temperature is not None:
                data.temperature = data.temperature[:self.cfg.capacity]
            n = self.cfg.capacity
        # determine indices
        content_start_idx = self.content.idx
        content_end_idx = self.content.idx + n
        wrap_around_size = 0 if content_end_idx <= self.cfg.capacity else content_end_idx - self.cfg.capacity
        data_end_idx = n - wrap_around_size
        content_end_idx -= wrap_around_size
        # update channels
        self.content.dc_obs[content_start_idx:content_end_idx] = data.obs[:data_end_idx]
        self.content.dc_val[content_start_idx:content_end_idx] = data.values[:data_end_idx]
        self.content.dc_pol[content_start_idx:content_end_idx] = data.policies[:data_end_idx]
        self.content.dc_len[content_start_idx:content_end_idx] = data.game_lengths[:data_end_idx]
        self.content.mc_ids[content_start_idx:content_end_idx] = self.content.game_idx
        self.content.mc_turns[content_start_idx:content_end_idx] = data.turns[:data_end_idx]
        self.content.mc_player[content_start_idx:content_end_idx] = data.player[:data_end_idx]
        self.content.mc_symmetry[content_start_idx:content_end_idx] = data.symmetry[:data_end_idx]
        if data.temperature is not None:
            self.content.dc_temp[content_start_idx:content_end_idx] = data.temperature[:data_end_idx]
        if wrap_around_size > 0:
            # wrap around update for channels
            self.content.dc_obs[:wrap_around_size] = data.obs[data_end_idx:]
            self.content.dc_val[:wrap_around_size] = data.values[data_end_idx:]
            self.content.dc_pol[:wrap_around_size] = data.policies[data_end_idx:]
            self.content.dc_len[:wrap_around_size] = data.game_lengths[data_end_idx:]
            self.content.mc_ids[:wrap_around_size] = self.content.game_idx
            self.content.mc_turns[:wrap_around_size] = data.turns[data_end_idx:]
            self.content.mc_player[:wrap_around_size] = data.player[data_end_idx:]
            self.content.mc_symmetry[:wrap_around_size] = data.symmetry[data_end_idx:]
            if data.temperature is not None:
                self.content.dc_temp[:wrap_around_size] = data.temperature[data_end_idx:]
        # update indices
        self.content.game_idx = (self.content.game_idx + 1) % self.cfg.capacity
        self.content.idx = (self.content.idx + n) % self.cfg.capacity
        # check if buffer is full
        if wrap_around_size > 0 or self.content.idx == 0:
            self.content.capacity_reached = True

    def _validate_input(self, data: BufferInputSample):
        n = data.obs.shape[0]
        if len(data.obs.shape) != len(self.cfg.obs_shape) + 1 or data.obs.shape[0] != n:
            raise ValueError(f"Invalid obs shape: {data.obs.shape}")
        if len(data.values.shape) != 2 or data.values.shape[0] != n \
                or data.values.shape[1] != 1:
            raise ValueError(f"Invalid value shape: {data.values.shape}")
        if len(data.policies.shape) != 2 or data.policies.shape[0] != n \
                or data.policies.shape[1] != self.cfg.num_actions:
            raise ValueError(f"Invalid policy shape: {data.policies.shape}")
        if len(data.game_lengths.shape) != 2 or data.game_lengths.shape[0] != n \
                or data.game_lengths.shape[1] != 1:
            raise ValueError(f"Invalid game lengths shape: {data.game_lengths.shape}")
        if len(data.turns.shape) != 2 or data.turns.shape[0] != n \
                or data.turns.shape[1] != 1:
            raise ValueError(f"Invalid turns shape: {data.turns.shape}")
        if len(data.player.shape) != 2 or data.player.shape[0] != n \
                or data.player.shape[1] != 1:
            raise ValueError(f"Invalid player shape: {data.player.shape}")
        if len(data.symmetry.shape) != 2 or data.symmetry.shape[0] != n \
                or data.symmetry.shape[1] != 1:
            raise ValueError(f"Invalid symmetry shape: {data.symmetry.shape}")
        if data.temperature is not None and (len(data.temperature.shape) != 2 or data.temperature.shape[0] != n
                                             or data.temperature.shape[1] != self.temp_channel_size):
            raise ValueError(f"Invalid temperature shape: {data.temperature.shape}")
        if self.cfg.num_players == 2 and n % 2 != 0:
            raise ValueError(f"Incomplete sample for two players with odd length")


def split_buffer(
        buffer: ReplayBuffer,
        random_split: bool,
        validation_split: float = 0.1,
        test_split: float = 0.1
) -> tuple[ReplayBuffer, ReplayBuffer, ReplayBuffer]:
    if random_split:
        buffer.shuffle()
    train_buffer = ReplayBuffer(copy.deepcopy(buffer.cfg))
    val_buffer = ReplayBuffer(copy.deepcopy(buffer.cfg))
    test_buffer = ReplayBuffer(copy.deepcopy(buffer.cfg))
    split1 = math.floor((1 - validation_split - test_split) * len(buffer))
    split2 = math.floor((1 - test_split) * len(buffer))
    # keep buffer capacity even to allow grouped sampling
    if buffer.cfg.num_players == 2:
        if split1 % 2 != 0:
            split1 -= 1
        if split2 % 2 != 0:
            split2 -= 1

    def _replace_with_split(b: ReplayBuffer, start: int, end: int):
        b.content.dc_obs = buffer.content.dc_obs[start:end]
        b.content.dc_val = buffer.content.dc_val[start:end]
        b.content.dc_pol = buffer.content.dc_pol[start:end]
        b.content.dc_len = buffer.content.dc_len[start:end]
        b.content.dc_temp = buffer.content.dc_temp[start:end]
        b.content.mc_ids = buffer.content.mc_ids[start:end]
        b.content.mc_turns = buffer.content.mc_turns[start:end]
        b.content.mc_player = buffer.content.mc_player[start:end]
        b.content.mc_symmetry = buffer.content.mc_symmetry[start:end]
        # meta data
        b.content.idx = 0
        b.content.capacity_reached = True
        b.content.game_idx = buffer.content.game_idx
        # misc
        b.cfg.capacity = end - start
        b.game_cfg = buffer.game_cfg

    _replace_with_split(train_buffer, 0, split1)
    _replace_with_split(val_buffer, split1, split2)
    _replace_with_split(test_buffer, split2, len(buffer))
    return train_buffer, val_buffer, test_buffer
