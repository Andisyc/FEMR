# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K-step executable return helper for FrontRES authority critic targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrontRESAuthorityReturn:
    """Done-masked K-step return and masks for event-level authority learning."""

    returns: torch.Tensor
    valid_mask: torch.Tensor
    steps: torch.Tensor
    bootstrap_mask: torch.Tensor


def _validate_time_tensor(name: str, tensor: torch.Tensor, reference_shape: torch.Size | tuple[int, ...]) -> None:
    if tuple(tensor.shape) != tuple(reference_shape):
        raise ValueError(f"{name} must have shape {tuple(reference_shape)}, got {tuple(tensor.shape)}.")


def compute_frontres_authority_k_step_return(
    exec_rewards: torch.Tensor,
    dones: torch.Tensor,
    event_mask: torch.Tensor,
    *,
    horizon: int,
    gamma: float,
    bootstrap_values: torch.Tensor | None = None,
) -> FrontRESAuthorityReturn:
    """Compute event-masked K-step executable returns.

    Args:
        exec_rewards: Reward tensor with shape ``(T, ...)``.
        dones: Done flags with the same shape. ``dones[t]`` stops credit after
            including ``exec_rewards[t]``.
        event_mask: Boolean mask selecting frames that own an authority target.
        horizon: Number of reward steps to include before optional bootstrap.
        gamma: Discount factor.
        bootstrap_values: Optional detached critic values with shape
            ``(T + 1, ...)``. ``bootstrap_values[t + horizon]`` is used only
            when the horizon fits in the rollout and no done was encountered.
    """

    if exec_rewards.ndim < 1:
        raise ValueError("exec_rewards must have a leading time dimension.")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}.")

    reward_shape = exec_rewards.shape
    _validate_time_tensor("dones", dones, reward_shape)
    _validate_time_tensor("event_mask", event_mask, reward_shape)
    if bootstrap_values is not None:
        expected_bootstrap_shape = (reward_shape[0] + 1, *reward_shape[1:])
        _validate_time_tensor("bootstrap_values", bootstrap_values, expected_bootstrap_shape)

    rewards = exec_rewards
    done_flags = dones.to(device=rewards.device, dtype=torch.bool)
    events = event_mask.to(device=rewards.device, dtype=torch.bool)
    returns = torch.zeros_like(rewards)
    steps = torch.zeros_like(rewards, dtype=torch.long)
    bootstrap_mask = torch.zeros_like(rewards, dtype=torch.bool)

    time_steps = rewards.shape[0]
    batch_shape = rewards.shape[1:]
    for start in range(time_steps):
        alive = torch.ones(batch_shape, device=rewards.device, dtype=torch.bool)
        total = torch.zeros(batch_shape, device=rewards.device, dtype=rewards.dtype)
        discount = 1.0
        step_count = torch.zeros(batch_shape, device=rewards.device, dtype=torch.long)

        for offset in range(horizon):
            src = start + offset
            if src >= time_steps:
                break
            active = alive
            total = total + float(discount) * rewards[src] * active.to(dtype=rewards.dtype)
            step_count = step_count + active.to(dtype=torch.long)
            alive = alive & (~done_flags[src])
            discount *= gamma

        if bootstrap_values is not None and start + horizon <= time_steps:
            boot_idx = start + horizon
            boot_active = alive
            total = total + float(discount) * bootstrap_values[boot_idx].detach().to(rewards.device) * boot_active.to(
                dtype=rewards.dtype
            )
            bootstrap_mask[start] = boot_active

        returns[start] = total
        steps[start] = step_count

    valid_mask = events & steps.gt(0)
    returns = returns * valid_mask.to(dtype=returns.dtype)
    steps = steps * valid_mask.to(dtype=steps.dtype)
    bootstrap_mask = bootstrap_mask & valid_mask
    return FrontRESAuthorityReturn(
        returns=returns,
        valid_mask=valid_mask,
        steps=steps,
        bootstrap_mask=bootstrap_mask,
    )
