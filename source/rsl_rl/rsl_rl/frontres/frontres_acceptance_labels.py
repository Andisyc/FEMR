# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FEMR acceptance labels from Noisy-vs-Candidate rollout evidence.

This module owns the Stage-2 HRL/admissibility label only.  It deliberately does
not build continuous rho targets, authority-critic returns, alpha routes, or
structured-rho carriers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FrontRESAcceptanceLabels:
    """Binary or masked acceptance labels for a Stage-1 proposal."""

    accept_gt: torch.Tensor
    accept_mask: torch.Tensor
    margin: torch.Tensor
    accept_frac: torch.Tensor
    reject_frac: torch.Tensor
    ignore_frac: torch.Tensor
    margin_mean: torch.Tensor
    beneficial_frac: torch.Tensor
    harmful_frac: torch.Tensor


def build_frontres_acceptance_labels(
    *,
    candidate_score: torch.Tensor,
    noisy_score: torch.Tensor | None = None,
    margin: torch.Tensor | None = None,
    positive_margin: float = 0.0,
    negative_margin: float | None = None,
) -> FrontRESAcceptanceLabels:
    """Build accept/reject labels from Candidate-vs-Noisy evidence.

    ``candidate_score`` and ``noisy_score`` should be executable scores in the
    same coordinate system.  The resulting label asks only whether the full
    Stage-1 proposal should be applied:

    - Candidate better than Noisy by margin -> ``accept_gt = 1``;
    - Candidate worse than Noisy by margin  -> ``accept_gt = 0``;
    - otherwise                             -> ``accept_mask = 0``.
    """

    if margin is None:
        if noisy_score is None:
            noisy_score = torch.zeros_like(candidate_score)
        margin = candidate_score - noisy_score
    margin = margin.view(-1)
    pos = max(0.0, float(positive_margin))
    neg = pos if negative_margin is None else max(0.0, float(negative_margin))

    accept = margin > pos
    reject = margin < -neg
    active = accept | reject
    dtype = margin.dtype
    accept_gt = accept.to(dtype=dtype).view(-1, 1)
    accept_mask = active.to(dtype=dtype).view(-1, 1)

    if margin.numel() == 0:
        zero = torch.tensor(0.0, device=margin.device, dtype=dtype)
        return FrontRESAcceptanceLabels(
            accept_gt=accept_gt,
            accept_mask=accept_mask,
            margin=margin.view(-1, 1),
            accept_frac=zero,
            reject_frac=zero,
            ignore_frac=zero,
            margin_mean=zero,
            beneficial_frac=zero,
            harmful_frac=zero,
        )

    return FrontRESAcceptanceLabels(
        accept_gt=accept_gt,
        accept_mask=accept_mask,
        margin=margin.view(-1, 1),
        accept_frac=accept.to(dtype=dtype).mean(),
        reject_frac=reject.to(dtype=dtype).mean(),
        ignore_frac=(~active).to(dtype=dtype).mean(),
        margin_mean=margin.mean(),
        beneficial_frac=(margin > 0.0).to(dtype=dtype).mean(),
        harmful_frac=(margin < 0.0).to(dtype=dtype).mean(),
    )


def expand_acceptance_labels_to_task_dims(
    labels: FrontRESAcceptanceLabels,
    *,
    task_dim: int = 6,
    dim_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand scalar accept/reject labels to task-space dimensions."""

    target = labels.accept_gt.expand(-1, int(task_dim)).clone()
    mask = labels.accept_mask.expand(-1, int(task_dim)).clone()
    if dim_mask is not None:
        dim_mask = dim_mask.to(device=mask.device, dtype=mask.dtype).view(1, -1)
        if dim_mask.shape[-1] != task_dim:
            raise ValueError(f"dim_mask must have {task_dim} dims, got {tuple(dim_mask.shape)}")
        mask = mask * dim_mask
    return target, mask
