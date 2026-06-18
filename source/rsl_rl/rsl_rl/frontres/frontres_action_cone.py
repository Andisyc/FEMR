# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch


class FrontRESActionCone:
    """Own FrontRES task-space action-cone projection and per-mode masks."""

    def __init__(self, cfg: dict[str, Any], alg: Any):
        self.cfg = cfg
        self.alg = alg

    def project_task_target(self, command: Any, target: torch.Tensor) -> torch.Tensor:
        """Project supervised Delta SE(3) targets into executable action bounds.

        The supervised target is the anti-perturbation, but not every
        anti-perturbation is dynamically admissible. Ordinary root sink would
        require upward dz, which runtime projection blocks to avoid artificial
        lift. Training uses this same projected target to avoid actor/reward
        mismatch.
        """
        if target.numel() == 0 or target.shape[-1] < 6:
            return target

        projected = target.clone()
        n = min(projected.shape[0], command._frontres_pos_correction.shape[0])
        if n <= 0:
            return projected

        policy = getattr(self.alg, "policy", None)
        max_delta_pos = float(getattr(policy, "max_delta_pos", 0.3))
        max_delta_rpy = float(getattr(policy, "max_delta_rpy", 0.1))
        projected[:n, :3] = projected[:n, :3].clamp(-max_delta_pos, max_delta_pos)
        projected[:n, 3:6] = projected[:n, 3:6].clamp(-max_delta_rpy, max_delta_rpy)

        active_dims = getattr(
            self.alg,
            "frontres_active_task_dims",
            self.cfg.get("frontres_active_task_dims", None),
        )
        if active_dims is not None:
            mask = torch.zeros(6, device=projected.device, dtype=projected.dtype)
            for dim in active_dims:
                dim = int(dim)
                if 0 <= dim < 6:
                    mask[dim] = 1.0
            projected[:n, :6] = projected[:n, :6] * mask.view(1, 6)

        z_upper = torch.zeros(n, device=projected.device, dtype=projected.dtype)
        if hasattr(command, "jump_degree") and hasattr(command, "anchor_penetration_depth"):
            jump_degree = command.jump_degree[:n].to(projected.device).to(projected.dtype).clamp(0.0, 1.0)
            penetration = command.anchor_penetration_depth[:n].to(projected.device).to(projected.dtype)
            z_upper = (jump_degree * penetration).clamp(max=max_delta_pos)
        z_lower = torch.full_like(z_upper, -max_delta_pos)
        projected[:n, 2] = torch.minimum(torch.maximum(projected[:n, 2], z_lower), z_upper)
        return projected

    @staticmethod
    def mode_dim_mask(
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a per-env Delta SE(3) mask from perturbation families."""
        mask = torch.zeros(count, 6, device=device, dtype=dtype)
        for env_i, modes in enumerate(list(mode_groups)[:count]):
            mode_set = set(modes)
            if "planar" in mode_set:
                mask[env_i, 0] = 1.0
                mask[env_i, 1] = 1.0
            if "global_z" in mode_set:
                mask[env_i, 2] = 1.0
            if "local_rp" in mode_set:
                mask[env_i, 3] = 1.0
                mask[env_i, 4] = 1.0
            if "yaw" in mode_set:
                mask[env_i, 5] = 1.0
        return mask

    def apply_per_mode_supervised_mask(
        self,
        target: torch.Tensor,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
        count: int,
    ) -> torch.Tensor:
        if target.numel() == 0 or target.shape[-1] < 6 or count <= 0:
            return target
        masked = target.clone()
        n = min(count, masked.shape[0])
        mode_mask = self.mode_dim_mask(mode_groups, n, masked.device, masked.dtype)
        masked[:n, :6] = masked[:n, :6] * mode_mask
        return masked
