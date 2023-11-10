import copy
import dataclasses
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ReplayBufferConfig:
    obs_shape: tuple[int, ...]
    num_actions: int
    num_players: int
    num_symmetries: int
    capacity: int


@dataclass
class BufferOutputSample:  # data structure used when sampling from buffer
    """
    shape is (n, *channel_shape)
    """
    obs: np.ndarray
    values: np.ndarray
    policies: np.ndarray


@dataclass
class BufferInputSample:  # data structure used for inserting new data into buffer
    # shape: (n, channel_shape)
    obs: np.ndarray  # channels shape: obs_shape
    values: np.ndarray  # channels shape: 1
    policies: np.ndarray  # channels shape: num_actions
    turns: np.ndarray  # channels shape: 1
    player: np.ndarray  # channels shape: 1
    symmetry: np.ndarray  # channels shape: 1


@dataclass
class ReplayBufferContent:
    # data channels: observation, value, policy, sbr-temperature
    dc_obs: np.ndarray
    dc_val: np.ndarray
    dc_pol: np.ndarray
    # metadata channels: game_id, turns_played, player, symmetry
    mc_ids: np.ndarray
    mc_turns: np.ndarray
    mc_player: np.ndarray
    mc_symmetry: np.ndarray
    # control channels: start indices, offsets
    capacity_reached: bool
    idx: int
    game_idx: int

    @classmethod
    def empty(
            cls,
            obs_shape: tuple[int, ...],
            capacity: int, num_actions: int,
    ) -> "ReplayBufferContent":
        obs_channel_shape = tuple([capacity] + list(obs_shape))
        return cls(
            dc_obs=np.zeros(shape=obs_channel_shape, dtype=float),
            dc_val=np.zeros(shape=(capacity, 1), dtype=float),
            dc_pol=np.zeros(shape=(capacity, num_actions), dtype=float),
            mc_ids=np.zeros(shape=(capacity, 1), dtype=int),
            mc_turns=np.zeros(shape=(capacity, 1), dtype=int),
            mc_player=np.zeros(shape=(capacity, 1), dtype=int),
            mc_symmetry=np.zeros(shape=(capacity, 1), dtype=int),
            capacity_reached=False,
            idx=0,
            game_idx=0,
        )


class ReplayBuffer:
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
        if content is None:
            self.content = ReplayBufferContent.empty(
                obs_shape=self.cfg.obs_shape,
                capacity=self.cfg.capacity,
                num_actions=self.cfg.num_actions,
            )
        else:
            self.content = content

    def __len__(self):
        return self.cfg.capacity if self.content.capacity_reached else self.content.idx

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = self.content.dc_obs[index]
        values = self.content.dc_val[index]
        policies = self.content.dc_pol[index]
        return values, policies, obs

    def get_grouped(self, index: int) -> BufferOutputSample:
        if self.cfg.num_players != 2:
            raise Exception(f"Grouping with more than two players not implemented")
        if index >= len(self) / 2:
            raise ValueError(f"Index too large for grouped retrieval: {index=}, {len(self)}")
        sample = BufferOutputSample(
            obs=self.content.dc_obs[2 * index:2 * index + 2],
            values=self.content.dc_val[2 * index:2 * index + 2],
            policies=self.content.dc_pol[2 * index:2 * index + 2],
        )
        return sample

    def sample(self, sample_size: int, grouped: bool = False) -> BufferOutputSample:
        if grouped:
            if self.cfg.num_players != 2:
                raise ValueError(f"grouped sampling not implemented for more than two players")
            if sample_size % 2 != 0:
                raise ValueError(f"sample size needs to be even for grouped sampling")
            sampled_idx = np.random.choice(math.floor(len(self) / 2), size=math.floor(sample_size / 2), replace=False)
            obs = np.stack([self.content.dc_obs[2 * sampled_idx], self.content.dc_obs[2 * sampled_idx + 1]])
            obs = np.reshape(obs, [2 * obs.shape[1], *self.cfg.obs_shape])
            values = np.stack([self.content.dc_val[2 * sampled_idx], self.content.dc_val[2 * sampled_idx + 1]])
            values = np.reshape(values, [2 * values.shape[1], 1])
            policies = np.stack([self.content.dc_pol[2 * sampled_idx], self.content.dc_pol[2 * sampled_idx + 1]])
            policies = np.reshape(policies, [2 * policies.shape[1], self.cfg.num_actions])
            sample = BufferOutputSample(
                obs=obs,
                values=values,
                policies=policies,
            )
        else:
            sampled_idx = np.random.choice(len(self), size=(sample_size,), replace=False)
            sample = BufferOutputSample(
                obs=self.content.dc_obs[sampled_idx],
                values=self.content.dc_val[sampled_idx],
                policies=self.content.dc_pol[sampled_idx],
            )
        return sample

    def sample_single(self, grouped: bool = False) -> BufferOutputSample:
        sample = self.sample(1, grouped)
        return sample

    def retrieve(
            self,
            start_idx: int,
            end_idx: int,
            grouped: bool = False,
    ) -> BufferOutputSample:
        if start_idx >= end_idx or start_idx < 0 or end_idx > len(self):
            raise ValueError(f"Invalid indices: {start_idx=}, {end_idx=}")
        if grouped:
            if end_idx * 2 + 1 > len(self):
                raise ValueError(f"Indices are invalid for grouped retrieval: {start_idx=}, {end_idx=}")
            start_idx *= 2
            end_idx *= 2
        sample = BufferOutputSample(
            obs=self.content.dc_obs[start_idx:end_idx],
            values=self.content.dc_val[start_idx:end_idx],
            policies=self.content.dc_pol[start_idx:end_idx],
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

    def full(self) -> bool:
        return self.content.capacity_reached

    def save(self, path: Path):
        content_dict = dataclasses.asdict(self.content)
        content_dict['cfg'] = self.cfg
        if self.game_cfg is not None:
            content_dict['game_cfg'] = self.game_cfg
        with open(path, 'wb') as f:
            pickle.dump(content_dict, f)

    @classmethod
    def from_saved_file(
            cls,
            file_path: Path,
    ):
        with open(file_path, 'rb') as f:
            saved_dict = pickle.load(f)
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
        self.content.dc_obs = self.content.dc_obs[indices]
        self.content.dc_val = self.content.dc_val[indices]
        self.content.dc_pol = self.content.dc_pol[indices]
        self.content.dc_len = self.content.dc_len[indices]
        self.content.mc_ids = self.content.mc_ids[indices]
        self.content.mc_turns = self.content.mc_turns[indices]
        self.content.mc_player = self.content.mc_player[indices]
        self.content.mc_symmetry = self.content.mc_symmetry[indices]
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
        self.content.mc_ids[content_start_idx:content_end_idx] = self.content.game_idx
        self.content.mc_turns[content_start_idx:content_end_idx] = data.turns[:data_end_idx]
        self.content.mc_player[content_start_idx:content_end_idx] = data.player[:data_end_idx]
        self.content.mc_symmetry[content_start_idx:content_end_idx] = data.symmetry[:data_end_idx]
        if wrap_around_size > 0:
            # wrap around update for channels
            self.content.dc_obs[:wrap_around_size] = data.obs[data_end_idx:]
            self.content.dc_val[:wrap_around_size] = data.values[data_end_idx:]
            self.content.dc_pol[:wrap_around_size] = data.policies[data_end_idx:]
            self.content.mc_ids[:wrap_around_size] = self.content.game_idx
            self.content.mc_turns[:wrap_around_size] = data.turns[data_end_idx:]
            self.content.mc_player[:wrap_around_size] = data.player[data_end_idx:]
            self.content.mc_symmetry[:wrap_around_size] = data.symmetry[data_end_idx:]
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
        if len(data.turns.shape) != 2 or data.turns.shape[0] != n \
                or data.turns.shape[1] != 1:
            raise ValueError(f"Invalid turns shape: {data.turns.shape}")
        if len(data.player.shape) != 2 or data.player.shape[0] != n \
                or data.player.shape[1] != 1:
            raise ValueError(f"Invalid player shape: {data.player.shape}")
        if len(data.symmetry.shape) != 2 or data.symmetry.shape[0] != n \
                or data.symmetry.shape[1] != 1:
            raise ValueError(f"Invalid symmetry shape: {data.symmetry.shape}")
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
