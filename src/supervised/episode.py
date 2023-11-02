from typing import Any, Optional

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.supervised.loss import compute_value_loss, compute_policy_loss, compute_length_loss, compute_zero_sum_loss


def single_episode(
        net,
        loader: DataLoader,
        optim: Optional[Optimizer],
        device: torch.device,
        use_zero_sum_loss: bool,
        mse_policy_loss: bool,
        mode: str,  # can be: train, valid or test
) -> dict[str, Any]:
    if mode == 'train' and optim is None:
        raise ValueError(f"Need optimizer in training mode")
    if mode == 'train':
        net = net.train()
    else:
        net = net.eval()
    losses, value_losses, policy_losses, zs_losses, length_losses = [], [], [], [], []
    grad_norms = []
    for value, policy, obs, length in loader:
        if mode == 'train':
            optim.zero_grad()
        # send to cuda
        value = value.to(device)
        obs = obs.to(device)
        # forward
        net_out = net(obs)
        # loss
        value_out = net.retrieve_value(net_out).unsqueeze(-1)
        val_loss = compute_value_loss(value_out, value)
        value_losses.append(val_loss.item())
        loss = val_loss
        if net.cfg.predict_policy:
            policy = policy.to(device)
            policy_out = net.retrieve_policy(net_out)
            pol_loss = compute_policy_loss(policy_out, policy, mse_policy_loss)
            loss += pol_loss
            policy_losses.append(pol_loss.item())
        if net.cfg.predict_game_len:
            length = length.to(device)
            len_out = net.retrieve_length(net_out)
            len_loss = compute_length_loss(len_out, length)
            loss += len_loss
            length_losses.append(len_loss.item())
        if use_zero_sum_loss:
            zero_sum_loss = compute_zero_sum_loss(value_out)
            loss += zero_sum_loss
            zs_losses.append(zero_sum_loss.item())
        # backward
        if mode == 'train':
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters=net.parameters(),
                max_norm=1,
                error_if_nonfinite=True,
            )
            optim.step()
            grad_norms.append(norm.item())
        # stats
        losses.append(loss.item())
    # parse statistics
    result = {
        f'{mode}_avg_loss': sum(losses) / len(losses),
        f'{mode}_std_loss': np.std(losses).item(),
        f'{mode}_median_loss': np.median(losses).item(),
        f'{mode}_max_loss': max(losses),
        f'{mode}_min_loss': min(losses),
        f'{mode}_value_avg_loss': sum(value_losses) / len(value_losses),
        f'{mode}_value_std_loss': np.std(value_losses).item(),
        f'{mode}_value_median_loss': np.median(value_losses).item(),
        f'{mode}_value_max_loss': max(value_losses),
        f'{mode}_value_min_loss': min(value_losses),
    }
    if net.cfg.predict_policy:
        result[f'{mode}_policy_avg_loss'] = sum(policy_losses) / len(policy_losses)
        result[f'{mode}_policy_std_loss'] = np.std(policy_losses).item()
        result[f'{mode}_policy_median_loss'] = np.median(policy_losses).item()
        result[f'{mode}_policy_max_loss'] = max(policy_losses)
        result[f'{mode}_policy_min_loss'] = min(policy_losses)
    if net.cfg.predict_game_len:
        result[f'{mode}_length_avg_loss'] = sum(length_losses) / len(length_losses)
        result[f'{mode}_length_std_loss'] = np.std(length_losses).item()
        result[f'{mode}_length_median_loss'] = np.median(length_losses).item()
        result[f'{mode}_length_max_loss'] = max(length_losses)
        result[f'{mode}_length_min_loss'] = min(length_losses)
    if use_zero_sum_loss:
        result[f'{mode}_zs_avg_loss'] = sum(zs_losses) / len(zs_losses)
        result[f'{mode}_zs_std_loss'] = np.std(zs_losses).item()
        result[f'{mode}_zs_median_loss'] = np.median(zs_losses).item()
        result[f'{mode}_zs_max_loss'] = max(zs_losses)
        result[f'{mode}_zs_min_loss'] = min(zs_losses)
    if mode == 'train':
        result['avg_grad_norm'] = sum(grad_norms) / len(grad_norms)
        result['max_grad_norm'] = max(grad_norms)
        result['min_grad_norm'] = min(grad_norms)
        result['std_grad_norm'] = np.std(grad_norms).item()
        result['median_grad_norm'] = np.median(grad_norms).item()
    return result
