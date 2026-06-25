# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Authority critic target arbitration for FrontRES."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrontRESAuthorityTargets:
    """Sanitized targets and masks for authority actor-critic learning."""

    behavior_return: torch.Tensor
    zero_return: torch.Tensor
    one_return: torch.Tensor
    mask: torch.Tensor
    harmful_full_write_mask: torch.Tensor
    conflict_mask: torch.Tensor


def resolve_frontres_authority_targets(
    *,
    behavior_return: torch.Tensor,
    zero_return: torch.Tensor,
    one_return: torch.Tensor,
    behavior_rho: torch.Tensor,
    mask: torch.Tensor,
    high_rho_threshold: float = 0.75,
    harmful_margin: float = 0.0,
) -> FrontRESAuthorityTargets:
    """Resolve conflicting behavior and endpoint authority targets.

    Endpoint ``rho=1`` evidence is useful when the executed behavior is far from
    full-write.  When the behavior authority is already near full-write, the
    realized behavior return is the more faithful local observation for that
    region of the authority axis.  If it is harmful relative to ``rho=0``, the
    endpoint-one target must not remain optimistically high.
    """

    target_behavior = torch.nan_to_num(behavior_return, nan=0.0, posinf=0.0, neginf=0.0)
    target_zero = torch.nan_to_num(zero_return, nan=0.0, posinf=0.0, neginf=0.0)
    target_one_raw = torch.nan_to_num(one_return, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

    rho = torch.nan_to_num(behavior_rho, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if rho.ndim == target_behavior.ndim and rho.shape[-1] != 1:
        rho_mean = rho.mean(dim=-1, keepdim=True)
    elif rho.ndim == target_behavior.ndim + 1:
        rho_mean = rho.mean(dim=-1)
    else:
        rho_mean = rho
    rho_mean = rho_mean.to(device=target_behavior.device, dtype=target_behavior.dtype)

    margin = float(harmful_margin)
    high_rho = rho_mean >= float(high_rho_threshold)
    harmful_behavior = target_behavior < (target_zero - margin)
    harmful_full_write = high_rho & harmful_behavior & valid_mask.gt(0.0)

    # If near-full behavior already proved harmful, treat it as the local
    # full-write target.  Otherwise keep the explicit rho=1 endpoint target.
    target_one = torch.where(harmful_full_write, target_behavior, target_one_raw)
    conflict = harmful_full_write & target_one_raw.gt(target_zero + margin)

    return FrontRESAuthorityTargets(
        behavior_return=target_behavior,
        zero_return=target_zero,
        one_return=target_one,
        mask=valid_mask,
        harmful_full_write_mask=harmful_full_write.to(dtype=target_behavior.dtype),
        conflict_mask=conflict.to(dtype=target_behavior.dtype),
    )

