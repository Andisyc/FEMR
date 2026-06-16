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
    def write_actor_gate(
        transition: Any,
        *,
        num_envs: int,
        n_exec: int,
        actor_gate: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Write the per-env FrontRES actor gate to transition and return it.

        The actor gate is a credit-assignment signal.  Keeping this write in one
        bridge makes it easier to audit whether rho/actor updates are being
        suppressed by route masks or sample gates.
        """
        frontres_actor_gate = torch.zeros(num_envs, 1, device=device)
        frontres_actor_gate[:n_exec, 0] = actor_gate
        transition.frontres_actor_gate = frontres_actor_gate
        return frontres_actor_gate
