# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrontRESRolloutEvidence:
    """Core rollout comparison evidence for FEMR/FrontRES diagnostics."""

    repair_gain: torch.Tensor
    candidate_gain: torch.Tensor
    projection_gain: torch.Tensor


def compute_frontres_rollout_evidence(
    *,
    noisy_score: torch.Tensor,
    projected_score: torch.Tensor,
    candidate_score: torch.Tensor,
) -> FrontRESRolloutEvidence:
    """Compare Noisy, Candidate, and Projected in one executable score space."""
    return FrontRESRolloutEvidence(
        repair_gain=projected_score - noisy_score,
        candidate_gain=candidate_score - noisy_score,
        projection_gain=projected_score - candidate_score,
    )
