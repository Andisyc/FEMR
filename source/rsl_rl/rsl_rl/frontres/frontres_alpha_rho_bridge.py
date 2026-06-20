# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch


class FrontRESAlphaRhoBridge:
    """Write alpha/rho route signals into the live algorithm transition."""

    @staticmethod
    def write_rho_update_weight(
        transition: Any,
        *,
        num_envs: int,
        n_exec: int,
        rho_update_weight: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Write the per-env FrontRES rho-update weight.

        ``frontres_actor_gate`` is the legacy storage field name.  Conceptually
        this tensor is a rho-update validity/strength weight, not a hard actor
        gate and not the rho direction signal itself.
        """
        frontres_rho_update_weight = torch.zeros(num_envs, 1, device=device)
        frontres_rho_update_weight[:n_exec, 0] = rho_update_weight
        transition.frontres_actor_gate = frontres_rho_update_weight
        return frontres_rho_update_weight

    @staticmethod
    def write_sample_weight(
        transition: Any,
        *,
        num_envs: int,
        n_exec: int,
        sample_weight: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Legacy alias for write_rho_update_weight()."""
        return FrontRESAlphaRhoBridge.write_rho_update_weight(
            transition,
            num_envs=num_envs,
            n_exec=n_exec,
            rho_update_weight=sample_weight,
            device=device,
        )

    @staticmethod
    def write_actor_gate(
        transition: Any,
        *,
        num_envs: int,
        n_exec: int,
        actor_gate: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Legacy alias for write_rho_update_weight()."""
        return FrontRESAlphaRhoBridge.write_rho_update_weight(
            transition,
            num_envs=num_envs,
            n_exec=n_exec,
            rho_update_weight=actor_gate,
            device=device,
        )
