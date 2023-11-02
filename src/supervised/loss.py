import torch


def compute_value_loss(
        value_pred: torch.Tensor,
        value_target: torch.Tensor,
) -> torch.Tensor:
    if len(value_pred.shape) != len(value_target.shape):
        raise ValueError(f"Invalid value shapes: {value_pred.shape=}, {value_target.shape=}")
    for i in range(len(value_pred.shape)):
        if value_pred.shape[i] != value_target.shape[i]:
            raise ValueError(f"Invalid value shapes: {value_pred.shape=}, {value_target.shape=}")
    loss = torch.nn.functional.mse_loss(value_pred, value_target)
    return loss


def compute_policy_loss(
        policy_pred_logits: torch.Tensor,
        policy_target: torch.Tensor,
        use_mse: bool,
) -> torch.Tensor:
    if len(policy_pred_logits.shape) != len(policy_target.shape):
        raise ValueError(f"Invalid value shapes: {policy_pred_logits.shape=}, {policy_target.shape=}")
    for i in range(len(policy_target.shape)):
        if policy_pred_logits.shape[i] != policy_target.shape[i]:
            raise ValueError(f"Invalid value shapes: {policy_pred_logits.shape=}, {policy_target.shape=}")
    if use_mse:
        pred_probs = torch.nn.functional.softmax(policy_pred_logits, dim=-1)
        loss = torch.nn.functional.mse_loss(pred_probs, policy_target)
    else:
        loss = torch.nn.functional.cross_entropy(policy_pred_logits, policy_target)
    return loss


def compute_length_loss(
        length_pred: torch.Tensor,
        length_target: torch.Tensor,
) -> torch.Tensor:
    if len(length_pred.shape) != len(length_target.shape):
        raise ValueError(f"Invalid value shapes: {length_pred.shape=}, {length_target.shape=}")
    for i in range(len(length_pred.shape)):
        if length_pred.shape[i] != length_target.shape[i]:
            raise ValueError(f"Invalid value shapes: {length_pred.shape=}, {length_target.shape=}")
    loss = torch.nn.functional.mse_loss(length_pred, length_target)
    return loss


def compute_zero_sum_loss(
        value_output: torch.Tensor,
) -> torch.Tensor:
    n = value_output.shape[0]
    if n % 2 != 0:
        raise ValueError(f"Value output needs even shape for zero-sum loss")
    p1_vals = value_output[:int(n/2)]
    p2_vals = value_output[int(n/2):]
    loss = torch.nn.functional.mse_loss(p1_vals, -p2_vals)
    return loss
