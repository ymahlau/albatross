import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.game.game import Game
from src.game.actions import sample_individual_actions
from src.misc.replay_buffer import ReplayBuffer, ReplayBufferConfig, ReplayBufferContent


@dataclass()
class DepthResultEntry:
    k: np.ndarray  # shape (), scalar: save every k search iterations
    values: Optional[np.ndarray]  # shape (num_samples, num_iter/k,)
    policies: Optional[np.ndarray]  # shape (num_samples, num_iter/k, num_actions)
    q_values: Optional[np.ndarray]  # shape (num_samples, num_iter/k, num_actions)
    ja_values: Optional[np.ndarray]  # shape (num_samples, num_actions^num_player, num_player) -> p0 is current player
    ja_actions: Optional[np.ndarray]  # shape (num_samples, num_actions^num_player, num_player)

    def get_aa_ja_and_values(
            self,
            index: int,
    ) -> tuple[list[list[int]], list[tuple[int, ...]], np.ndarray]:
        # computes available actions, joint action list and joint action value array
        ja = self.ja_actions[index]
        ja_value = self.ja_values[index]
        num_p_at_turn = ja.shape[-1]
        num_ja = np.argwhere(ja[:, 0] == -2)[0].item()
        ja_value_filtered = ja_value[:num_ja]
        joint_action_list = [tuple(ja[i]) for i in range(num_ja)]
        available_actions = []
        for p in range(num_p_at_turn):
            actions = set(ja[:, p]) - {-2}  # remove placeholder -2 if it is present
            available_actions.append(list(actions))
        return available_actions, joint_action_list, ja_value_filtered


@dataclass()
class DepthResultStruct:
    episode: np.ndarray  # shape num_samples
    turn: np.ndarray  # shape num_samples
    player: np.ndarray  # shape num_samples
    game_length: Optional[np.ndarray]  # shape num_samples, optional is just for intermediate computation
    results: dict[str, DepthResultEntry]
    legal_actions: np.ndarray  # shape (num_samples, num_actions)
    obs: Optional[np.ndarray]  # shape (num_samples, num_symmetries, *obs_shape)

    def save(self, path: Union[str, Path]):
        save_dict = {
            'episode': self.episode,
            'turn': self.turn,
            'player': self.player,
            'game_length': self.game_length,
            'legal_actions': self.legal_actions,
        }
        if self.obs is not None:
            save_dict['obs'] = self.obs
        for name, entry in self.results.items():
            save_dict[f'{name}_k'] = entry.k
            if entry.values is not None:
                save_dict[f'{name}_values'] = entry.values
            if entry.policies is not None:
                save_dict[f'{name}_policies'] = entry.policies
            if entry.q_values is not None:
                save_dict[f'{name}_q_values'] = entry.q_values
            if entry.ja_values is not None:
                save_dict[f'{name}_ja_values'] = entry.ja_values
            if entry.ja_actions is not None:
                save_dict[f'{name}_ja_actions'] = entry.ja_actions
        save_dict['names'] = np.asarray(list(self.results.keys()))
        np.savez_compressed(path, **save_dict)

    def __len__(self):
        return self.episode.shape[0]

    @classmethod
    def from_file(cls, path: Union[str, Path]):
        loaded = np.load(path)
        result_dict = {}
        for name in loaded['names']:
            result_dict[name] = DepthResultEntry(
                values=loaded[f'{name}_values'] if f'{name}_values' in loaded else None,
                policies=loaded[f'{name}_policies'] if f'{name}_policies' in loaded else None,
                q_values=loaded[f'{name}_q_values'] if f'{name}_q_values' in loaded else None,
                k=loaded[f'{name}_k'],
                ja_values=loaded[f'{name}_ja_values'] if f'{name}_ja_values' in loaded else None,
                ja_actions=loaded[f'{name}_ja_actions'] if f'{name}_ja_actions' in loaded else None,
            )
        cls_obj = cls(
            episode=loaded['episode'],
            turn=loaded['turn'],
            player=loaded['player'],
            game_length=loaded['game_length'],
            results=result_dict,
            legal_actions=loaded['legal_actions'],
            obs=loaded['obs'] if 'obs' in loaded else None,
        )
        return cls_obj

    def downsample(self, new_size: int):
        start_idx = len(self) - new_size
        end_idx = len(self)
        indices = np.asarray(list(range(start_idx, end_idx)), dtype=int)
        # downsample
        self._sort_from_idx_arr(indices)

    def sort_by_variance(self, search_name: str):
        # sorts the batch dimension wrt. the variance of q-value estimation during tree search
        qs = self.results[search_name].q_values
        if qs is None:
            raise ValueError(f"Cannot sort by variance without q-values")
        std = np.std(qs, axis=1)
        avg_std = np.mean(std, axis=-1)
        index_arr = np.argsort(avg_std)
        self._sort_from_idx_arr(index_arr)

    def sort_by_variance_of_gradients(self, search_name: str):
        # sorts the batch dimension by the variance of q-value estimation gradients wrt. search time
        # using finite differences
        qs = self.results[search_name].q_values
        if qs is None:
            raise ValueError(f"Cannot sort by variance without q-values")
        grads = qs[:, 1:, :] - qs[:, :-1, :]
        std = np.std(grads, axis=1)
        avg_std = np.mean(std, axis=-1)
        index_arr = np.argsort(avg_std)
        self._sort_from_idx_arr(index_arr)

    def _sort_from_idx_arr(self, index_arr: np.ndarray):
        # sort
        self.episode = self.episode[index_arr]
        self.turn = self.turn[index_arr]
        self.player = self.player[index_arr]
        self.game_length = self.game_length[index_arr]
        self.legal_actions = self.legal_actions[index_arr]
        if self.obs is not None:
            self.obs = self.obs[index_arr]
        for name, entry in self.results.items():
            if entry.values is not None:
                entry.values = entry.values[index_arr]
            if entry.policies is not None:
                entry.policies = entry.policies[index_arr]
            if entry.q_values is not None:
                entry.q_values = entry.q_values[index_arr]
            if entry.ja_values is not None:
                entry.ja_values = entry.ja_values[index_arr]
            if entry.ja_actions is not None:
                entry.ja_actions = entry.ja_actions[index_arr]

    def to_replay_buffer(self, search_name: str) -> ReplayBuffer:
        if self.obs is None:
            raise ValueError(f"Cannot convert to buffer without observation")
        entry = self.results[search_name]
        if entry.policies is None:
            raise ValueError(f"Cannot build buffer without policies")
        if entry.values is None:
            raise ValueError(f"Cannot build buffer without values")
        buffer_cfg = ReplayBufferConfig(
            obs_shape=self.obs.shape[1:],
            num_actions=entry.policies.shape[-1],
            num_players=np.max(self.player).item() + 1,
            num_symmetries=1,
            capacity=self.obs.shape[0],
            single_temperature=True,
        )
        content = ReplayBufferContent(
            dc_obs=torch.tensor(self.obs, dtype=torch.float32),
            dc_val=torch.tensor(entry.values, dtype=torch.float32).unsqueeze(-1),
            dc_pol=torch.tensor(entry.policies, dtype=torch.float32),
            dc_len=torch.tensor(self.game_length, dtype=torch.float32).unsqueeze(-1),
            dc_temp=torch.zeros(self.episode.shape, dtype=torch.float32).unsqueeze(-1),
            mc_ids=torch.zeros(self.episode.shape, dtype=torch.float32).unsqueeze(-1),
            mc_turns=torch.tensor(self.turn, dtype=torch.float32).unsqueeze(-1),
            mc_player=torch.tensor(self.player, dtype=torch.float32).unsqueeze(-1),
            mc_symmetry=torch.zeros(self.episode.shape, dtype=torch.float32).unsqueeze(-1),
            capacity_reached=True,
            idx=0,
            game_idx=0,
        )
        buffer = ReplayBuffer(
            cfg=buffer_cfg,
            content=content,
            game_cfg=None,
        )
        return buffer


def aggregate_structs(struct_list: list[DepthResultStruct]) -> DepthResultStruct:
    if not struct_list:
        raise ValueError("Cannot aggregate an empty list")
    # required parameters
    full_episodes = np.concatenate([s.episode for s in struct_list], axis=0)
    for s in struct_list:
        del s.episode
    full_turns = np.concatenate([s.turn for s in struct_list], axis=0)
    for s in struct_list:
        del s.turn
    full_player = np.concatenate([s.player for s in struct_list], axis=0)
    for s in struct_list:
        del s.player
    full_legal_actions = np.concatenate([s.legal_actions for s in struct_list], axis=0)
    for s in struct_list:
        del s.legal_actions
    full_game_length = None
    if struct_list[0].game_length is not None:
        full_game_length = np.concatenate([s.game_length for s in struct_list], axis=0)
        for s in struct_list:
            del s.game_length
    # observations are optional
    full_obs = None
    if struct_list[0].obs is not None:
        full_obs = np.concatenate([s.obs for s in struct_list], axis=0)
        for s in struct_list:
            del s.obs
    # result struct entries for every search
    new_entries = {}
    name_list = list(struct_list[0].results.keys())
    for name in name_list:
        full_values = None
        if struct_list[0].results[name].values is not None:
            full_values = np.concatenate([s.results[name].values for s in struct_list], axis=0)
            for s in struct_list:
                del s.results[name].values
        full_policies = None
        if struct_list[0].results[name].policies is not None:
            full_policies = np.concatenate([s.results[name].policies for s in struct_list], axis=0)
            for s in struct_list:
                del s.results[name].policies
        full_q_values = None
        if struct_list[0].results[name].q_values is not None:
            full_q_values = np.concatenate([s.results[name].q_values for s in struct_list], axis=0)
            for s in struct_list:
                del s.results[name].q_values
        full_ja_values = None
        if struct_list[0].results[name].ja_values is not None:
            full_ja_values = np.concatenate([s.results[name].ja_values for s in struct_list], axis=0)
            for s in struct_list:
                del s.results[name].ja_values
        full_ja_actions = None
        if struct_list[0].results[name].ja_actions is not None:
            full_ja_actions = np.concatenate([s.results[name].ja_actions for s in struct_list], axis=0)
            for s in struct_list:
                del s.results[name].ja_actions
        k = struct_list[0].results[name].k
        entry = DepthResultEntry(
            values=full_values,
            policies=full_policies,
            q_values=full_q_values,
            ja_values=full_ja_values,
            ja_actions=full_ja_actions,
            k=k,
        )
        new_entries[name] = entry
        for s in struct_list:
            del s.results[name]
    # construct resulting struct
    result = DepthResultStruct(
        episode=full_episodes,
        turn=full_turns,
        player=full_player,
        game_length=full_game_length,
        results=new_entries,
        legal_actions=full_legal_actions,
        obs=full_obs,
    )
    return result


def joint_action_from_struct(
        struct: DepthResultStruct,
        game: Game,
        step_temperature: float,
        step_iterations: Optional[int],  # if none, use latest iteration count
        step_search: Optional[str],
) -> tuple[int, ...]:
    # sanity check
    if step_search is None:
        keys = list(struct.results.keys())
        step_search = random.choice(keys)
    if struct.results[step_search].policies.shape[0] != game.num_players_at_turn():
        raise Exception("Struct has to contain only observations of current step for action sampling")
    if step_iterations is None:
        step_iterations = -1
    probs = struct.results[step_search].policies[:, step_iterations, :]
    ja = sample_individual_actions(probs, step_temperature)
    return ja
