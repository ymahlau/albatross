import math
import multiprocessing.sharedctypes as sc
import os
import pickle
import queue as q
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import multiprocessing as mp

from src.game.game import GameConfig
from src.game.initialization import get_game_from_config
from src.game.values import UtilityNorm
from src.misc.replay_buffer import BufferOutputSample
from src.misc.utils import set_seed
from src.network import Network
from src.network.initialization import get_network_from_config, get_network_from_file
from src.supervised.annealer import TemperatureAnnealer
from src.supervised.loss import compute_utility_loss, compute_value_loss, compute_policy_loss
from src.supervised.optim import get_optim_from_config
from src.trainer.config import UpdaterConfig, LoggerConfig, AlphaZeroTrainerConfig
from src.trainer.utils import send_obj_to_queue, wait_for_obj_from_queue


@dataclass
class UpdaterStatistics:
    update_counter: sc.Synchronized
    updates_since_info: int = 0
    updates_since_distribution: int = 0
    last_info_time: float = time.time()
    last_save_state_time: float = time.time()
    value_losses: list[float] = field(default_factory=lambda: [])
    policy_losses: list[float] = field(default_factory=lambda: [])
    utility_losses: list[float] = field(default_factory=lambda: [])
    val_reg_losses: list[float] = field(default_factory=lambda: [])
    losses: list[float] = field(default_factory=lambda: [])
    idle_time_sum: float = 0
    backward_time_sum: float = 0
    loss_time_sum: float = 0
    model_conv_time_sum: float = 0
    norm_sum: float = 0
    norm_min: float = math.inf
    norm_max: float = 0
    norm_n: int = 0
    process_start_time: float = time.time()
    started = False


@dataclass
class UpdaterEssentials:
    net: Network
    optim: torch.optim.Optimizer
    device: torch.device
    annealer: TemperatureAnnealer


def run_updater(
        trainer_cfg: AlphaZeroTrainerConfig,
        data_in_queue: mp.Queue,
        net_queue_out: mp.Queue,
        stop_flag: sc.Synchronized,
        info_queue: mp.Queue,
        update_counter: sc.Synchronized,
        save_state_after_seconds: int,
        cpu_list: Optional[list[int]],
        gpu_idx: Optional[int],
        prev_run_dir: Optional[Path],
        prev_run_idx: Optional[int],
        seed: int,
):
    updater_cfg = trainer_cfg.updater_cfg
    net_cfg = trainer_cfg.net_cfg
    game_cfg = trainer_cfg.game_cfg
    logger_cfg = trainer_cfg.logger_cfg
    # torch.autograd.set_detect_anomaly(True)
    state_save_dir = Path(os.getcwd()) / 'state'
    # important to avoid pytorch deadlocks
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    set_seed(seed)
    torch.set_float32_matmul_precision('medium')
    if updater_cfg.utility_loss != UtilityNorm.NONE and game_cfg.num_players != 2:
        raise ValueError(f"Can only use utility loss with two players")
    # initialization
    if net_cfg is None:
        raise Exception("Network config is None")
    net = get_network_from_config(net_cfg)
    if prev_run_dir is not None and not trainer_cfg.init_new_network_params:
        model_path = prev_run_dir / 'fixed_time_models' / f'm_{prev_run_idx}.pt'
        net = get_network_from_file(model_path)
    game = get_game_from_config(game_cfg)
    # cuda
    if updater_cfg.use_gpu:
        # torch.cuda.init()
        device = torch.device('cuda' if gpu_idx is None else f'cuda:{gpu_idx}')
    else:
        device = torch.device('cpu')
    net = net.to(device).train()
    # compile
    if trainer_cfg.compile_model:
        net = torch.compile(
            model=net,
            dynamic=False,
            mode=trainer_cfg.compile_mode,
            fullgraph=True,
        )
    print(f"{net.num_params()=}", flush=True)
    info_queue.put_nowait({"Network Parameter": net.num_params()})
    # optimizer
    optim, annealer = get_optim_from_config(updater_cfg.optim_cfg, net)
    # load previous optimizer state if necessary
    if prev_run_dir is not None and not trainer_cfg.init_new_network_params:
        optim_path = prev_run_dir / 'state' / 'optim.pt'
        if os.path.exists(optim_path):
            optim_state_dict = torch.load(optim_path)
            optim.load_state_dict(optim_state_dict)
        else:
            print(f"{datetime.now()} - WARNING: Cannot load optimizer from previous run directory ........", flush=True)
    essentials = UpdaterEssentials(net=net, optim=optim, device=device, annealer=annealer) # type: ignore
    stats = UpdaterStatistics(update_counter=update_counter)
    # restrict cpus
    pid = os.getpid()
    if cpu_list is not None:
        print(f"{datetime.now()} - CPU list in Updater: {cpu_list}")
        os.sched_setaffinity(pid, cpu_list)
        print(f'{datetime.now()} - Started Updater process with pid {pid} using cpus: {os.sched_getaffinity(pid)}'
              f" using device {device}", flush=True)
    else:
        print(f'{datetime.now()} - Started Updater process with pid {pid} using device {device}', flush=True)
    try:
        while not stop_flag.value:
            # save optimizer state
            if time.time() - stats.last_save_state_time > save_state_after_seconds \
                    and update_counter.value != 0:
                save_optim_state(optim, game_cfg, state_save_dir)
                stats.last_save_state_time = time.time()
            if stop_flag.value:
                break
            # update net
            perform_update(essentials, stats, updater_cfg, data_in_queue, stop_flag)
            # print info stats
            if stats.updates_since_info == logger_cfg.updater_bucket_size:
                log_info(
                    stats=stats,
                    essentials=essentials,
                    logger_cfg=logger_cfg,
                    info_queue=info_queue,
                    stop_flag=stop_flag,
                    net_queue_out=net_queue_out,
                )
            if stop_flag.value:
                break
            # send updated network state dict to other processes
            if stats.updates_since_distribution == updater_cfg.updates_until_distribution:
                distribute_net(essentials, stats, net_queue_out)

    except KeyboardInterrupt:
        print('Detected Keyboard Interrupt in Updater\n', flush=True)
    # cleanup
    game.close()
    print(f'{datetime.now()} - Update process with pid {os.getpid()} terminated', flush=True)
    sys.exit(0)


def distribute_net(
        essentials: UpdaterEssentials,
        stats: UpdaterStatistics,
        net_queue_out: mp.Queue,
):
    model_conv_time_start = time.time()
    essentials.net.eval()  # escnn state dict only works in eval mode, (weird, but ok)
    state_dict = essentials.net.state_dict()
    newest_state_dict = {}
    # state dict can only be sent to processes via cpu
    for k, v in state_dict.items():
        newest_state_dict[k] = v.cpu().numpy()
    stats.model_conv_time_sum += time.time() - model_conv_time_start
    # send
    idle_time_start = time.time()
    try:
        net_queue_out.put_nowait(newest_state_dict)
    except q.Full:
        pass
    essentials.net.train()
    # update info stats
    stats.idle_time_sum += time.time() - idle_time_start
    stats.updates_since_distribution = 0


def log_info(
        stats: UpdaterStatistics,
        essentials: UpdaterEssentials,
        logger_cfg: LoggerConfig,
        info_queue: mp.Queue,
        stop_flag: sc.Synchronized,
        net_queue_out: mp.Queue,
):
    # time usage
    full_time = time.time() - stats.last_info_time
    other_time = full_time - stats.idle_time_sum - stats.backward_time_sum - stats.model_conv_time_sum
    other_time -= stats.loss_time_sum
    msg_data = {
        'updater_idle_ratio': stats.idle_time_sum / full_time,
        'updater_backward_ratio': stats.backward_time_sum / full_time,
        'updater_model_conv_ratio': stats.model_conv_time_sum / full_time,
        'updater_loss_ratio': stats.loss_time_sum / full_time,
        'updater_update_ratio': (stats.loss_time_sum + stats.backward_time_sum) / full_time,
        'updater_other_ratio': other_time / full_time,
        'updater_value_avg_loss': sum(stats.value_losses) / logger_cfg.updater_bucket_size,
        'updater_value_std_loss': np.std(stats.value_losses).item(),
        'updater_value_median_loss': np.median(stats.value_losses).item(),
        'updater_value_max_loss': max(stats.value_losses),
        'updater_value_min_loss': min(stats.value_losses),
        'updater_val_reg_avg_loss': sum(stats.val_reg_losses) / logger_cfg.updater_bucket_size,
        'updater_val_reg_std_loss': np.std(stats.val_reg_losses).item(),
        'updater_val_reg_median_loss': np.median(stats.val_reg_losses).item(),
        'updater_val_reg_max_loss': max(stats.val_reg_losses),
        'updater_val_reg_min_loss': min(stats.val_reg_losses),
        'updater_avg_loss': sum(stats.losses) / logger_cfg.updater_bucket_size,
        'updater_std_loss': np.std(stats.losses).item(),
        'updater_median_loss': np.median(stats.losses).item(),
        'updater_max_loss': max(stats.losses),
        'updater_min_loss': min(stats.losses),
        'avg_update_time': full_time / logger_cfg.updater_bucket_size,
        'learning_rate': essentials.optim.param_groups[0]["lr"],
        'updater_queue_size': net_queue_out.qsize(),
    }
    if essentials.net.cfg.predict_policy:
        msg_data['updater_policy_avg_loss'] = sum(stats.policy_losses) / logger_cfg.updater_bucket_size
        msg_data['updater_policy_std_loss'] = np.std(stats.policy_losses).item()
        msg_data['updater_policy_median_loss'] = np.median(stats.policy_losses).item()
        msg_data['updater_policy_max_loss'] = max(stats.policy_losses)
        msg_data['updater_policy_min_loss'] = max(stats.policy_losses)
    if stats.utility_losses:
        msg_data['updater_utility_avg_loss'] = sum(stats.utility_losses) / logger_cfg.updater_bucket_size
        msg_data['updater_utility_std_loss'] = np.std(stats.utility_losses).item()
        msg_data['updater_utility_median_loss'] = np.median(stats.utility_losses).item()
        msg_data['updater_utility_max_loss'] = max(stats.utility_losses)
        msg_data['updater_utility_min_loss'] = max(stats.utility_losses)
    if full_time > 0:
        msg_data['updates_per_min'] = logger_cfg.updater_bucket_size * 60 / full_time
    if stats.norm_n > 0:
        msg_data['grad_norm'] = stats.norm_sum / stats.norm_n
        msg_data['grad_norm_min'] = stats.norm_min
        msg_data['grad_norm_max'] = stats.norm_max
        stats.norm_sum = 0
        stats.norm_max = 0
        stats.norm_min = math.inf
        stats.norm_n = 0
    # reset info values
    stats.last_info_time = time.time()
    stats.backward_time_sum = 0
    stats.loss_time_sum = 0
    stats.model_conv_time_sum = 0
    stats.idle_time_sum = 0
    stats.updates_since_info = 0
    stats.value_losses = []
    stats.policy_losses = []
    stats.utility_losses = []
    stats.val_reg_losses = []
    stats.losses = []
    # send message
    idle_time_start = time.time()
    send_obj_to_queue(msg_data, info_queue, stop_flag)
    stats.idle_time_sum += time.time() - idle_time_start


def perform_update(
        essentials: UpdaterEssentials,
        stats: UpdaterStatistics,
        updater_cfg: UpdaterConfig,
        data_in_queue: mp.Queue,
        stop_flag: sc.Synchronized,
):
    # sample data
    idle_time_start = time.time()
    maybe_sample = wait_for_obj_from_queue(data_in_queue, stop_flag)
    if not isinstance(maybe_sample, BufferOutputSample):
        raise Exception(f"Got unknown item from queue: {maybe_sample}")
    stats.idle_time_sum += time.time() - idle_time_start
    if maybe_sample is None:
        return
    if not stats.started:
        stats.started = True
        stats.process_start_time = time.time()
    # compute loss and update
    essentials.optim.zero_grad()
    loss_time_start = time.time()
    value_loss, val_output, policy_loss, utility_loss = compute_loss(
        sample=maybe_sample,
        net=essentials.net,
        device=essentials.device,
        utility_loss=updater_cfg.utility_loss,
        mse_policy_loss=updater_cfg.mse_policy_loss,
    )
    stats.loss_time_sum += time.time() - loss_time_start
    update_time_start = time.time()
    loss = torch.clone(value_loss)
    if policy_loss is not None:
        policy_loss *= updater_cfg.policy_loss_factor
        loss += policy_loss
    if utility_loss is not None:
        utility_loss *= updater_cfg.utility_loss_factor
        loss += utility_loss
    val_reg_loss = updater_cfg.value_reg_loss_factor * torch.abs(val_output).mean()
    loss += val_reg_loss
    loss.backward()
    # clip gradient norm
    if updater_cfg.gradient_max_norm is not None:
        norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            parameters=essentials.net.parameters(),
            max_norm=updater_cfg.gradient_max_norm,
            error_if_nonfinite=True,
        )
        stats.norm_sum += norm.item()
        stats.norm_n += 1
        if norm > stats.norm_max:
            stats.norm_max = norm.item()
        if norm < stats.norm_min:
            stats.norm_min = norm.item()
    # updater and annealer step
    essentials.optim.step()
    time_passed = (time.time() - stats.process_start_time) / 60
    essentials.annealer(time_passed)
    # update stats
    stats.losses += [loss.cpu().item()]
    stats.value_losses += [value_loss.cpu().item()]
    stats.val_reg_losses += [val_reg_loss.cpu().item()]
    if policy_loss is not None:
        stats.policy_losses += [policy_loss.cpu().item()]
    if utility_loss is not None:
        stats.utility_losses += [utility_loss.cpu().item()]
    stats.update_counter.value += 1
    stats.updates_since_distribution += 1
    stats.updates_since_info += 1
    stats.backward_time_sum += time.time() - update_time_start


def compute_loss(
        sample: BufferOutputSample,
        net: Network,
        device: torch.device,
        utility_loss: UtilityNorm,
        mse_policy_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    inputs = torch.from_numpy(sample.obs).to(device).float()
    values = torch.from_numpy(sample.values).to(device).float()
    policies = torch.from_numpy(sample.policies).to(device).float()
    outputs = net(inputs)
    val_output = net.retrieve_value_tensor(outputs).unsqueeze(-1)
    val_loss = compute_value_loss(val_output, values)
    action_loss, zero_sum_loss = None, None
    if net.cfg.predict_policy:
        action_output = net.retrieve_policy_tensor(outputs)
        action_loss = compute_policy_loss(action_output, policies, mse_policy_loss)
    if utility_loss != UtilityNorm.NONE:
        zero_sum_loss = compute_utility_loss(val_output, utility_loss)
    return val_loss, val_output, action_loss, zero_sum_loss


def save_optim_state(
        optimizer: torch.optim.Optimizer,
        game_cfg: GameConfig,
        state_save_dir: Path,
) -> None:
    if not Path.exists(state_save_dir):
        state_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), state_save_dir / 'optim.pt')
    with open(state_save_dir / 'game_cfg.pkl', 'wb') as f:
        pickle.dump(game_cfg, f)
