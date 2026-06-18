"""State-alpha target construction for FrontRES.

This module owns the supervised signal for the state router.  It is intentionally
pure: the runner passes rollout evidence in, and receives target/mask tensors
plus scalar diagnostics back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrontRESStateAlphaTargets:
    target: torch.Tensor
    mask: torch.Tensor
    target_mean: float
    mask_mean: float


def build_state_alpha_targets(
    *,
    num_envs: int,
    n_exec: int,
    exec_perturbed: torch.Tensor,
    base_done: torch.Tensor,
    base_timeout: torch.Tensor | None,
    exec_floor: float,
    safe_floor: float,
    temp: float,
    device: torch.device,
) -> FrontRESStateAlphaTargets:
    """Build state-alpha labels from Noisy/GMT continuation and executable floor."""
    target = torch.zeros(num_envs, 1, device=device)
    mask = torch.zeros(num_envs, 1, device=device)
    if n_exec <= 0:
        return FrontRESStateAlphaTargets(target=target, mask=mask, target_mean=0.0, mask_mean=0.0)

    n = min(int(n_exec), int(exec_perturbed.numel()), int(base_done.numel()))
    if n <= 0:
        return FrontRESStateAlphaTargets(target=target, mask=mask, target_mean=0.0, mask_mean=0.0)

    exec_values = exec_perturbed[:n].to(device=device).view(-1)
    done = base_done[:n].to(device=device).view(-1) > 0
    if base_timeout is None:
        timeout = torch.zeros(n, device=device, dtype=torch.bool)
    else:
        timeout = base_timeout[:n].to(device=device).view(-1) > 0

    fall_now = done & (~timeout)
    alpha_temp = max(1.0e-6, float(temp))
    floor_mid = 0.5 * (float(exec_floor) + float(safe_floor))
    soft_label = torch.sigmoid((floor_mid - exec_values) / alpha_temp)
    soft_label = torch.where(fall_now, torch.ones_like(soft_label), soft_label)

    safe_signal = (~done) & (exec_values >= float(safe_floor))
    broken_signal = fall_now | (exec_values <= float(exec_floor))
    label_active = (broken_signal | safe_signal) & (~timeout)

    target[:n, 0] = soft_label.detach()
    mask[:n, 0] = label_active.to(mask.dtype).detach()
    target_mean = float(target[:n, 0][label_active].mean().item()) if bool(label_active.any().item()) else 0.0
    mask_mean = float(label_active.float().mean().item())
    return FrontRESStateAlphaTargets(target=target, mask=mask, target_mean=target_mean, mask_mean=mask_mean)
