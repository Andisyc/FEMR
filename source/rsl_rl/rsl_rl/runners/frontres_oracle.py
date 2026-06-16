# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrontRESOracleUpperBound:
    """Diagnostic-only upper bound over Noisy, Projected, Candidate, Feasible."""

    gain: torch.Tensor
    pass_mask: torch.Tensor
    noisy_win: torch.Tensor
    projected_win: torch.Tensor
    candidate_win: torch.Tensor
    feasible_win: torch.Tensor


def compute_frontres_oracle_upper_bound(
    noisy_score: torch.Tensor,
    projected_score: torch.Tensor,
    candidate_score: torch.Tensor,
    feasible_score: torch.Tensor,
    *,
    margin: float = 0.0,
    enabled: bool = True,
) -> FrontRESOracleUpperBound:
    """Compute the optimistic diagnostic upper bound without actor credit.

    This answers whether the rollout evidence contains any branch that can beat
    Noisy/GMT. It must remain diagnostic-only: the winner source is not a policy
    action and should not be converted into PPO credit.
    """
    if not enabled:
        return FrontRESOracleUpperBound(
            gain=torch.zeros_like(projected_score),
            pass_mask=torch.zeros_like(projected_score),
            noisy_win=torch.ones_like(projected_score),
            projected_win=torch.zeros_like(projected_score),
            candidate_win=torch.zeros_like(projected_score),
            feasible_win=torch.zeros_like(projected_score),
        )

    scores = torch.stack((noisy_score, projected_score, candidate_score, feasible_score), dim=0)
    best, source = scores.max(dim=0)
    gain = best - noisy_score
    pass_mask = (gain > float(margin)).float()
    return FrontRESOracleUpperBound(
        gain=gain,
        pass_mask=pass_mask,
        noisy_win=(source == 0).float(),
        projected_win=(source == 1).float(),
        candidate_win=(source == 2).float(),
        feasible_win=(source == 3).float(),
    )
