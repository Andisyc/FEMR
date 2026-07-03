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


def repair_effect_summary_to_scalars(summary: dict[str, Any]) -> dict[str, float]:
    scalars = {
        "segment/train_effect_noisy": _float(summary.get("score_noisy_mean")),
        "segment/train_effect_repaired": _float(summary.get("score_repaired_mean")),
        "segment/train_effect_gain": _float(summary.get("score_gain_mean")),
        "segment/train_effect_gain_pos_frac": _float(summary.get("score_gain_pos_frac")),
        "segment/train_effect_fall_rate": _float(summary.get("done_frac")),
        "segment/train_effect_valid_frac": _float(summary.get("storage_valid_frac")),
        "segment/train_effect_replay_candidates": _float(summary.get("sampler_replay_candidates")),
        "segment/train_effect_replay_pool_size": _float(summary.get("sampler_replay_pool_size")),
    }
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    return scalars


def format_segment_train_effect_log(summary: dict[str, Any]) -> str:
    scalars = repair_effect_summary_to_scalars(summary)
    return "\n".join(
        (
            "[FrontRES Segment Train Effect]",
            (
                "  score: "
                f"noisy={scalars['segment/train_effect_noisy']:.6f} "
                f"repaired={scalars['segment/train_effect_repaired']:.6f} "
                f"gain={scalars['segment/train_effect_gain']:.6f} "
                f"gain_pos={scalars['segment/train_effect_gain_pos_frac'] * 100.0:.1f}%"
            ),
            (
                "  data: "
                f"fall={scalars['segment/train_effect_fall_rate'] * 100.0:.1f}% "
                f"valid={scalars['segment/train_effect_valid_frac'] * 100.0:.1f}%"
            ),
            (
                "  replay: "
                f"candidates={scalars['segment/train_effect_replay_candidates']:.0f} "
                f"pool={scalars['segment/train_effect_replay_pool_size']:.0f}"
            ),
        )
    )


def motion_quality_summary_to_scalars(
    *,
    clean_positions: torch.Tensor | None = None,
    repaired_positions: torch.Tensor | None = None,
    noisy_positions: torch.Tensor | None = None,
    delta_se: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    valid = valid_mask.bool() if isinstance(valid_mask, torch.Tensor) else None
    return {
        "segment/motion_mpjpe_repaired_clean": _mpjpe(repaired_positions, clean_positions, valid),
        "segment/motion_mpjpe_noisy_clean": _mpjpe(noisy_positions, clean_positions, valid),
        "segment/motion_vel_error_repaired_clean": _diff_mpjpe(repaired_positions, clean_positions, valid, order=1),
        "segment/motion_acc_error_repaired_clean": _diff_mpjpe(repaired_positions, clean_positions, valid, order=2),
        "segment/motion_delta_se_norm": _delta_se_norm(delta_se, valid),
        "segment/motion_delta_z_up_frac": _delta_z_up_frac(delta_se, valid),
    }


def format_segment_motion_quality_log(scalars: dict[str, float]) -> str:
    return "\n".join(
        (
            "[FrontRES Segment Motion Quality]",
            (
                "  pose: "
                f"mpjpe_repaired={scalars.get('segment/motion_mpjpe_repaired_clean', 0.0):.6f} "
                f"mpjpe_noisy={scalars.get('segment/motion_mpjpe_noisy_clean', 0.0):.6f}"
            ),
            (
                "  dynamics: "
                f"vel_err={scalars.get('segment/motion_vel_error_repaired_clean', 0.0):.6f} "
                f"acc_err={scalars.get('segment/motion_acc_error_repaired_clean', 0.0):.6f}"
            ),
            (
                "  action: "
                f"delta_se_norm={scalars.get('segment/motion_delta_se_norm', 0.0):.6f} "
                f"dz_up={scalars.get('segment/motion_delta_z_up_frac', 0.0) * 100.0:.1f}%"
            ),
        )
    )


def periodic_eval_summary_to_scalars(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "segment/eval_episode_length": _float(summary.get("episode_length")),
        "segment/eval_success_rate": _float(summary.get("success_rate")),
        "segment/eval_fall_rate": _float(summary.get("fall_rate")),
        "segment/eval_mean_survival_steps": _float(summary.get("mean_survival_steps")),
        "segment/eval_continuous_rollout_gain": _float(summary.get("continuous_rollout_gain")),
    }


def format_segment_periodic_eval_log(summary: dict[str, Any]) -> str:
    scalars = periodic_eval_summary_to_scalars(summary)
    return "\n".join(
        (
            "[FrontRES Segment Periodic Eval]",
            (
                "  rollout: "
                f"episode_length={scalars['segment/eval_episode_length']:.1f} "
                f"survival={scalars['segment/eval_mean_survival_steps']:.1f}"
            ),
            (
                "  result: "
                f"success={scalars['segment/eval_success_rate'] * 100.0:.1f}% "
                f"fall={scalars['segment/eval_fall_rate'] * 100.0:.1f}% "
                f"gain={scalars['segment/eval_continuous_rollout_gain']:.6f}"
            ),
        )
    )


def segment_summary_to_scalars(summary: FrontRESSegmentReplaySummary) -> dict[str, float]:
    scalars = dict(summary.scalars)
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    return scalars


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())
    if value is None:
        return 0.0
    return float(value)


def _mpjpe(a: torch.Tensor | None, b: torch.Tensor | None, valid: torch.Tensor | None) -> float:
    if not _same_position_shape(a, b):
        return 0.0
    diff = torch.linalg.norm(a.float() - b.float(), dim=-1)
    return _masked_batch_mean(diff, valid)


def _diff_mpjpe(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    valid: torch.Tensor | None,
    *,
    order: int,
) -> float:
    if not _same_position_shape(a, b) or a.shape[1] <= order:
        return 0.0
    da = torch.diff(a.float(), n=order, dim=1)
    db = torch.diff(b.float(), n=order, dim=1)
    return _masked_batch_mean(torch.linalg.norm(da - db, dim=-1), valid)


def _same_position_shape(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and a.ndim >= 3


def _masked_batch_mean(value: torch.Tensor, valid: torch.Tensor | None) -> float:
    if value.numel() == 0:
        return 0.0
    if valid is not None and valid.shape[0] == value.shape[0]:
        value = value[valid]
        if value.numel() == 0:
            return 0.0
    return float(value.mean().item())


def _delta_se_norm(delta_se: torch.Tensor | None, valid: torch.Tensor | None) -> float:
    if not isinstance(delta_se, torch.Tensor) or delta_se.numel() == 0:
        return 0.0
    value = torch.linalg.norm(delta_se.float(), dim=-1)
    if valid is not None and valid.shape[0] == value.shape[0]:
        value = value[valid]
        if value.numel() == 0:
            return 0.0
    return float(value.mean().item())


def _delta_z_up_frac(delta_se: torch.Tensor | None, valid: torch.Tensor | None) -> float:
    if not isinstance(delta_se, torch.Tensor) or delta_se.ndim < 2 or delta_se.shape[-1] < 3:
        return 0.0
    value = delta_se[..., 2] > 0.0
    if valid is not None and valid.shape[0] == value.shape[0]:
        value = value[valid]
        if value.numel() == 0:
            return 0.0
    return float(value.float().mean().item())


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
