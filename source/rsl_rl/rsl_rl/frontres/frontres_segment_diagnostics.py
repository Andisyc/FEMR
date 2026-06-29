from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


FORBIDDEN_ACCEPTANCE_KEYS = {
    "acceptance_gt",
    "acceptance_mask",
    "acceptance_margin",
    "acceptance_prob",
}


@dataclass(frozen=True)
class FrontRESSegmentReplaySummary:
    scalars: dict[str, float]
    stage: str
    objective: str


def summarize_segment_batch(
    sample: Any,
    reward_result: Any,
    reset_result: Any,
    action_stats: Any,
    sampler_stats: Any | None = None,
    stage: str = "stage3_segment_hrl",
    objective: str = "segment_replay_hrl",
) -> FrontRESSegmentReplaySummary:
    scalars: dict[str, float] = {}
    sources = tuple(getattr(sample, "source", ()))
    total = max(1, len(sources))
    scalars["segment/global_frac"] = sources.count("global") / total
    scalars["segment/replay_frac"] = sources.count("replay") / total
    scalars["segment/review_frac"] = sources.count("review") / total
    priority = getattr(sample, "priority", torch.zeros(0))
    scalars["segment/priority_mean"] = _mean(priority)
    scalars["segment/priority_p90"] = _quantile(priority, 0.9)
    if sampler_stats is not None:
        scalars["segment/replay_pool_size"] = float(getattr(sampler_stats, "replay_pool_size", 0))
    else:
        scalars["segment/replay_pool_size"] = float((priority > 0.0).sum().item()) if isinstance(priority, torch.Tensor) else 0.0

    solved = getattr(reward_result, "solved_mask", torch.zeros(0, dtype=torch.bool))
    hopeless = getattr(reward_result, "hopeless_mask", torch.zeros_like(solved))
    valid = getattr(reward_result, "valid_mask", torch.ones_like(solved))
    scalars["segment/solved_frac"] = _bool_mean(solved)
    scalars["segment/hopeless_frac"] = _bool_mean(hopeless)
    scalars["segment/active_frac"] = _bool_mean(valid & (~solved.bool()) & (~hopeless.bool())) if isinstance(valid, torch.Tensor) else 0.0
    scalars["segment/reset_success_frac"] = _bool_mean(getattr(reset_result, "success_mask", torch.zeros(0, dtype=torch.bool)))
    scalars["segment/preroll_frac"] = _bool_mean(getattr(reset_result, "preroll_mask", torch.zeros(0, dtype=torch.bool)))
    horizon = getattr(sample, "horizon_k", None)
    if horizon is None:
        horizon = getattr(reward_result, "horizon_k", None)
    scalars["segment/k"] = _mean(horizon) if isinstance(horizon, torch.Tensor) else float(horizon or 0.0)
    scalars["segment/score_noisy"] = _mean(getattr(reward_result, "score_noisy", torch.zeros(0)))
    scalars["segment/score_repaired"] = _mean(getattr(reward_result, "score_repaired", torch.zeros(0)))
    scalars["segment/score_clean"] = _mean(getattr(reward_result, "score_clean", torch.zeros(0)))
    scalars["segment/gain_over_noisy"] = _mean(getattr(reward_result, "gain_over_noisy", torch.zeros(0)))
    scalars["segment/fall_frac"] = _bool_mean(getattr(reward_result, "fall_flag", torch.zeros(0, dtype=torch.bool)))
    scalars["segment/contact_consistency"] = _mean(getattr(reward_result, "contact_consistency", torch.zeros(0)))
    scalars["segment/action_norm"] = float(getattr(action_stats, "action_norm_mean", 0.0))
    per_dim = getattr(action_stats, "per_dim_norm", torch.zeros(6))
    per_dim = per_dim.detach().flatten().float().cpu() if isinstance(per_dim, torch.Tensor) else torch.zeros(6)
    labels = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
    for i, label in enumerate(labels):
        scalars[f"segment/action_norm_{label}"] = float(per_dim[i].item()) if i < per_dim.numel() else 0.0
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    return FrontRESSegmentReplaySummary(scalars=scalars, stage=stage, objective=objective)


def format_segment_replay_log(summary: FrontRESSegmentReplaySummary) -> str:
    scalars = summary.scalars
    return (
        f"FrontRES Segment HRL active: stage={summary.stage} objective={summary.objective} "
        f"k={scalars.get('segment/k', 0.0):.0f} "
        f"mix=global:{scalars.get('segment/global_frac', 0.0):.2f}/"
        f"replay:{scalars.get('segment/replay_frac', 0.0):.2f}/"
        f"review:{scalars.get('segment/review_frac', 0.0):.2f} "
        f"gain={scalars.get('segment/gain_over_noisy', 0.0):.4f} "
        f"reset={scalars.get('segment/reset_success_frac', 0.0):.2f}"
    )


def segment_summary_to_scalars(summary: FrontRESSegmentReplaySummary) -> dict[str, float]:
    scalars = dict(summary.scalars)
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    return scalars


def _mean(value: torch.Tensor | None) -> float:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.float().mean().item())


def _quantile(value: torch.Tensor | None, q: float) -> float:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(torch.quantile(value.float().flatten(), q).item())


def _bool_mean(value: torch.Tensor | None) -> float:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.bool().float().mean().item())
