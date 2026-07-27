# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy Stage-3 HSL rollout-label boundary.

FRS-TRAIN-v007 retains Stage-1 anti-DR initialization only. The previous
rollout-derived label is intentionally not materialized, audited, stored, or
consumed by any active Stage-3 route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

import torch


@dataclass(frozen=True)
class FrontRESHSLRolloutTargetResult:
    """Historical result shape retained only for import compatibility."""

    target: torch.Tensor
    weight: torch.Tensor
    harm_weight: torch.Tensor


def quat_to_rotvec_wxyz(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map wxyz unit quaternions to shortest-path rotation vectors."""

    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    xyz_norm = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, q[..., :1].clamp(min=eps))
    scale = torch.where(xyz_norm > eps, angle / xyz_norm.clamp(min=eps), 2.0 * torch.ones_like(xyz_norm))
    return xyz * scale


# B4: QUALITY-ACTION-01 keeps the retired Stage-3 HSL label fail-closed.
def build_frontres_hsl_rollout_target(
    runner: Any,
    *,
    command: Any,
    actions: torch.Tensor | None,
    dones: torch.Tensor | None,
    current_pos_correction: torch.Tensor | None,
    current_quat_correction: torch.Tensor | None,
    n_train: int,
    n_candidate: int,
    n_base: int,
    n_clean: int,
    quat_to_rotvec_wxyz: Any,
    write_transition: bool = True,
    enforce_training_enable_flag: bool = True,
) -> NoReturn:
    """Fail closed: v007 forbids every Stage-3 rollout-derived HSL label."""

    del (
        runner,
        command,
        actions,
        dones,
        current_pos_correction,
        current_quat_correction,
        n_train,
        n_candidate,
        n_base,
        n_clean,
        quat_to_rotvec_wxyz,
        write_transition,
        enforce_training_enable_flag,
    )
    raise RuntimeError(
        "FRS-TRAIN-v007 disables the legacy Stage-3 HSL rollout label; "
        "use Stage-1 proposal-only anti-DR initialization instead"
    )
