import dataclasses
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import torch
import wandb
from torch.utils.data import DataLoader

from src.game.game import GameConfig
from src.misc.replay_buffer import ReplayBuffer, split_buffer
from src.misc.utils import flatten_dict_rec, set_seed
from src.network import NetworkConfig
from src.network.initialization import get_network_from_config
from src.supervised.optim import get_optim_from_config, OptimizerConfig
from src.supervised.episode import single_episode


@dataclass
class SupervisedTrainerConfig:
    buffer_path: str
    game_cfg: GameConfig
    net_cfg: NetworkConfig
    optim_cfg: OptimizerConfig
    num_episodes: Optional[int] = None
    time_limit_min: Optional[float] = None
    batch_size: int = 128
    use_gpu: bool = False
    num_worker_loader: int = 0
    run_name: Optional[str] = None
    logging_mode: str = 'online'
    project_name: str = 'battlesnake_rl_supervised'
    seed: Optional[int] = None
    random_split: bool = False
    save_every_episode: Optional[int] = None
    save_every_min: Optional[int] = None
    zero_sum_loss: bool = False
    validation_split: float = 0.1
    test_split: float = 0.1
    mse_policy_loss: bool = False
    compile_model: bool = False


class SupervisedTrainer:
    def __init__(self, cfg: SupervisedTrainerConfig):
        torch.set_float32_matmul_precision('high')
        self.start_time = time.time()
        if cfg.seed is not None:
            set_seed(cfg.seed)
        # sensible defaults
        if cfg.net_cfg.game_cfg is None:
            cfg.net_cfg.game_cfg = cfg.game_cfg
        # input validation
        if cfg.num_episodes is None and cfg.time_limit_min is None:
            raise ValueError(f"Need either episode count or time limit")
        if cfg.num_episodes is not None and cfg.time_limit_min is not None:
            raise ValueError(f"Can only specify episode count or time limit, not both")
        # init
        self.cfg = cfg
        self.device = torch.device('cuda' if self.cfg.use_gpu else 'cpu')
        # init training net
        self.net = get_network_from_config(cfg.net_cfg)
        self.net = self.net.to(self.device)
        if self.cfg.compile_model:
            self.compiled_net = torch.compile(
                model=self.net,
                fullgraph=True,
                dynamic=False,
                mode='default',
            )
            assert self.cfg.optim_cfg.fused is False
        self.optim, self.annealer = get_optim_from_config(cfg.optim_cfg, self.net)
        # buffer and loader
        buffer = ReplayBuffer.from_saved_file(Path(self.cfg.buffer_path))
        self.train_buffer, self.val_buffer, self.test_buffer = split_buffer(
            buffer=buffer,
            random_split=self.cfg.random_split,
            validation_split=self.cfg.validation_split,
            test_split=self.cfg.test_split,
        )
        self.train_loader = self._construct_loader(self.train_buffer, shuffle=True)
        self.val_loader = self._construct_loader(self.val_buffer, shuffle=False)
        if self.cfg.test_split > 0:
            self.test_loader = self._construct_loader(self.test_buffer, shuffle=False)
        # paths
        self.model_folder = Path(os.getcwd()) / 'models'
        self.model_folder.mkdir(parents=True, exist_ok=True)
        self.optim_folder = Path(os.getcwd())
        print(f"Supervised trainer uses device: {self.device}")

    def _construct_loader(self, buffer: ReplayBuffer, shuffle: bool) -> DataLoader:
        return DataLoader(
            buffer,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_worker_loader,
        )

    def __call__(self):
        # paths
        wandb_dir = Path(__file__).parent.parent.parent / 'wandb'
        wandb_dir.mkdir(parents=True, exist_ok=True)
        # init wandb
        flat_config_dict = flatten_dict_rec(dataclasses.asdict(self.cfg))
        kwargs = {
            'project': self.cfg.project_name,
            'config': flat_config_dict,
            'dir': wandb_dir,
            'mode': self.cfg.logging_mode,
            'save_code': False
        }
        if self.cfg.run_name is not None:
            name = self.cfg.run_name if self.cfg.seed is None else f"{self.cfg.run_name}_{self.cfg.seed}"
            kwargs["name"] = name
        wandb.init(**kwargs)
        # training loop
        ep_idx = 0
        last_save_time = self.start_time
        while True:
            # breaking conditions
            if self.cfg.num_episodes is not None and ep_idx == self.cfg.num_episodes:
                break
            if self.cfg.time_limit_min is not None and (time.time() - self.start_time) / 60 >= self.cfg.time_limit_min:
                break
            # training
            info = {}
            self.net.train()
            if self.cfg.compile_model:
                self.compiled_net.train()
            train_start = time.time()
            train_dict = single_episode(
                net=self.net if not self.cfg.compile_model else self.compiled_net,
                loader=self.train_loader,
                optim=self.optim,
                device=self.device,
                use_zero_sum_loss=self.cfg.zero_sum_loss,
                mode='train',
                mse_policy_loss=self.cfg.mse_policy_loss,
            )
            info['train_time'] = time.time() - train_start
            if self.cfg.num_episodes is not None:
                # anneal lr: End time is one, so use percentage of episodes as proxy
                time_passed = (ep_idx + 1) / self.cfg.num_episodes
            else:
                time_passed = (time.time() - self.start_time) / 60
            self.annealer(time_passed)
            info['learning_rate'] = self.optim.param_groups[0]["lr"]
            # validation
            self.net.eval()
            if self.cfg.compile_model:
                self.compiled_net.eval()
            val_start = time.time()
            val_dict = single_episode(
                net=self.net if not self.cfg.compile_model else self.compiled_net,
                loader=self.val_loader,
                optim=None,
                device=self.device,
                use_zero_sum_loss=self.cfg.zero_sum_loss,
                mode='valid',
                mse_policy_loss=self.cfg.mse_policy_loss,
            )
            info['val_time'] = time.time() - val_start
            # logging
            self._log(train_dict, val_dict, info, ep_idx)
            if self.cfg.save_every_episode is not None and ep_idx % self.cfg.save_every_episode == 0:
                self._save_state(ep_idx)
                last_save_time = time.time()
            if self.cfg.save_every_min is not None and (time.time() - last_save_time) / 60 >= self.cfg.save_every_min:
                self._save_state(ep_idx)
            ep_idx += 1
        # test
        if self.cfg.test_split > 0:
            self.net.eval()
            if self.cfg.compile_model:
                self.compiled_net.eval()
            test_start = time.time()
            test_dict = single_episode(
                net=self.net if not self.cfg.compile_model else self.compiled_net,
                loader=self.test_loader,
                optim=None,
                device=self.device,
                use_zero_sum_loss=self.cfg.zero_sum_loss,
                mode='test',
                mse_policy_loss=self.cfg.mse_policy_loss,
            )
            test_dict['test_time'] = time.time() - test_start
            test_dict['episode_counter'] = ep_idx
            wandb.log(test_dict)
        print('Stopped Supervised Training process')

    def _log(
            self,
            train_dict: dict[str, Any],
            val_dict: dict[str, Any],
            info: dict[str, float],
            episode_counter: int,
    ):
        info_dict = {}
        for k, v in train_dict.items():
            info_dict[k] = v
        for k, v in val_dict.items():
            info_dict[k] = v
        for k, v in info.items():
            info_dict[k] = v
        info_dict['episode_counter'] = episode_counter
        # send to wandb
        wandb.log(info_dict)
        # print to console
        print(f"{episode_counter} / {self.cfg.num_episodes}: {datetime.now()} - {info_dict}", flush=True)

    def _save_state(self, episode_counter: int):
        # optim
        torch.save(self.optim.state_dict(), self.optim_folder / 'optim.pt')
        # net
        self.net.save(self.model_folder / f"m_{episode_counter}.pt")
