"""Shared immutable row contract for FrontRES scenario lifecycles."""

from __future__ import annotations

import torch

from rsl_rl.frontres.frontres_segment_planning import FrontRESSegmentSample


def scenario_row_fields(
    sample: FrontRESSegmentSample,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Freeze Segment, source, attempt and K identities for scenario rows."""

    segment_ids = immutable_row_tensor("segment_ids", sample.segment_ids)
    source_index = getattr(sample, "source_index", None)
    trial_index = getattr(sample, "trial_index", None)
    horizon_k = getattr(sample, "horizon_k", None)
    if not isinstance(source_index, torch.Tensor):
        raise ValueError("scenario lifecycle requires sample.source_index")
    if not isinstance(trial_index, torch.Tensor):
        raise ValueError("scenario lifecycle requires sample.trial_index")
    if not isinstance(horizon_k, torch.Tensor):
        raise ValueError("scenario lifecycle requires sample.horizon_k")
    source_index = immutable_row_tensor("source_index", source_index)
    trial_index = immutable_row_tensor("trial_index", trial_index)
    horizon_k = immutable_row_tensor("horizon_k", horizon_k)
    count = int(segment_ids.numel())
    for name, value in (
        ("source_index", source_index),
        ("trial_index", trial_index),
        ("horizon_k", horizon_k),
    ):
        if int(value.numel()) != count:
            raise ValueError(f"sample.{name} must have {count} rows, got {int(value.numel())}")
    if bool((horizon_k <= 0).any().item()):
        raise ValueError("sample.horizon_k must be positive")
    return segment_ids, source_index, trial_index, horizon_k


def immutable_row_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    """Return a detached rank-1 lifecycle identity tensor."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {tuple(value.shape)}")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached lifecycle metadata")
    return value.detach().to(dtype=torch.long).clone()


__all__ = ["immutable_row_tensor", "scenario_row_fields"]
