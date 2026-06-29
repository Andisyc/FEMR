from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch

_SAMPLER_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_segment_sampler.py"
_SAMPLER_SPEC = importlib.util.spec_from_file_location(
    "frontres_segment_sampler_live_module",
    _SAMPLER_PATH,
)
if _SAMPLER_SPEC is None or _SAMPLER_SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES Segment sampler from {_SAMPLER_PATH}.")
_SAMPLER_MODULE = importlib.util.module_from_spec(_SAMPLER_SPEC)
sys.modules[_SAMPLER_SPEC.name] = _SAMPLER_MODULE
_SAMPLER_SPEC.loader.exec_module(_SAMPLER_MODULE)

FrontRESSegmentRolloutEvidence = _SAMPLER_MODULE.FrontRESSegmentRolloutEvidence
FrontRESSegmentSample = _SAMPLER_MODULE.FrontRESSegmentSample
FrontRESSegmentSampler = _SAMPLER_MODULE.FrontRESSegmentSampler


def initialize_frontres_segment_live_sampler(runner: Any) -> None:
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    if not bool(getattr(boundary, "requested", False) and getattr(boundary, "live_runner_enabled", False)):
        return
    if getattr(runner, "_frontres_segment_sampler", None) is not None:
        return
    num_segments = _resolve_num_segments(runner)
    runner._frontres_segment_sampler = FrontRESSegmentSampler(
        num_segments=num_segments,
        global_frac=float(getattr(runner.alg, "frontres_segment_sampler_global_frac", 0.4)),
        replay_frac=float(getattr(runner.alg, "frontres_segment_sampler_replay_frac", 0.5)),
        review_frac=float(getattr(runner.alg, "frontres_segment_sampler_review_frac", 0.1)),
        seed=int(getattr(runner, "seed", 0) or 0),
        device=getattr(runner, "device", "cpu"),
    )
    print(
        "[FrontRES Segment Sampler Ready] "
        f"num_segments={num_segments} "
        f"global_frac={runner._frontres_segment_sampler.global_frac:.3f} "
        f"replay_frac={runner._frontres_segment_sampler.replay_frac:.3f} "
        f"review_frac={runner._frontres_segment_sampler.review_frac:.3f}",
        flush=True,
    )


def run_frontres_segment_sampler_step(
    runner: Any,
    *,
    init_at_random_ep_len: bool,
    update_step: int,
) -> dict[str, object]:
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    if sampler is None:
        return runner.run_frontres_segment_live_probe(init_at_random_ep_len=init_at_random_ep_len)

    sample = sampler.sample(_resolve_live_batch_size(runner))
    _print_sample_probe(update_step, sample)
    runner._frontres_segment_live_current_sample = sample
    try:
        summary = runner.run_frontres_segment_live_probe(init_at_random_ep_len=init_at_random_ep_len)
    finally:
        runner._frontres_segment_live_current_sample = None

    evidence = build_live_sampler_evidence(sample, summary, horizon_k=int(getattr(runner.alg, "frontres_segment_k", 1)))
    sampler.update(evidence)
    sampler_summary = summarize_sampler_step(sampler, sample)
    summary.update(sampler_summary)
    _print_sampler_summary(update_step, sampler_summary)
    return summary


def build_live_sampler_evidence(
    sample: FrontRESSegmentSample,
    summary: dict[str, object],
    *,
    horizon_k: int,
) -> FrontRESSegmentRolloutEvidence:
    ids = sample.segment_ids.detach().clone().long()
    n = int(ids.numel())
    device = ids.device
    reward_mean = _summary_float(summary, "storage_reward_mean", _summary_float(summary, "reward_mean", 0.0))
    gain_scalar = max(-1.0, min(1.0, reward_mean))
    score_repaired_scalar = max(0.0, min(1.0, 0.5 + 0.5 * gain_scalar))
    score_noisy_scalar = max(0.0, min(1.0, score_repaired_scalar - gain_scalar))
    valid = bool(_summary_int(summary, "ppo_valid_count", 0) > 0 and _summary_float(summary, "storage_valid_frac", 0.0) > 0.0)
    fall = bool(_summary_float(summary, "done_frac", 0.0) >= 0.5)
    return FrontRESSegmentRolloutEvidence(
        segment_ids=ids,
        reset_success=torch.ones(n, dtype=torch.bool, device=device),
        score_noisy=torch.full((n,), score_noisy_scalar, dtype=torch.float32, device=device),
        score_repaired=torch.full((n,), score_repaired_scalar, dtype=torch.float32, device=device),
        score_clean=torch.ones(n, dtype=torch.float32, device=device),
        gain_over_noisy=torch.full((n,), gain_scalar, dtype=torch.float32, device=device),
        fall_repaired=torch.full((n,), fall, dtype=torch.bool, device=device),
        contact_consistency=torch.ones(n, dtype=torch.float32, device=device),
        action_norm=torch.ones(n, dtype=torch.float32, device=device),
        valid_reward=torch.full((n,), valid, dtype=torch.bool, device=device),
        horizon_k=torch.full((n,), max(1, int(horizon_k)), dtype=torch.long, device=device),
    )


def summarize_sampler_step(sampler: FrontRESSegmentSampler, sample: FrontRESSegmentSample) -> dict[str, object]:
    stats = sampler.stats()
    counts = Counter(sample.source)
    stale_review_count = int(((sampler.staleness > 0.0) & sampler.solved & (~sampler.invalid)).sum().item())
    return {
        "sampler_update": True,
        "sampler_batch_size": int(sample.segment_ids.numel()),
        "sampler_source_global_count": int(counts.get("global", 0)),
        "sampler_source_replay_count": int(counts.get("replay", 0)),
        "sampler_source_review_count": int(counts.get("review", 0)),
        "sampler_replay_pool_size": int(stats.replay_pool_size),
        "sampler_review_pool_size": int(stats.review_pool_size),
        "sampler_priority_mean": float(stats.priority_mean),
        "sampler_priority_p90": float(stats.priority_p90),
        "sampler_solved_frac": float(stats.solved_frac),
        "sampler_hopeless_frac": float(stats.hopeless_frac),
        "sampler_stale_review_count": stale_review_count,
    }


def _resolve_num_segments(runner: Any) -> int:
    dataset = getattr(runner, "_frontres_segment_dataset", None)
    if dataset is not None and hasattr(dataset, "num_segments"):
        num_segments = dataset.num_segments()
        return max(1, int(num_segments))
    env = getattr(runner, "env", None)
    return max(1, int(getattr(env, "num_envs", 1) or 1))


def _resolve_live_batch_size(runner: Any) -> int:
    env = getattr(runner, "env", None)
    return max(1, int(getattr(env, "num_envs", 1) or 1))


def _summary_float(summary: dict[str, object], key: str, default: float) -> float:
    try:
        return float(summary.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _summary_int(summary: dict[str, object], key: str, default: int) -> int:
    try:
        return int(summary.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _print_sample_probe(update_step: int, sample: FrontRESSegmentSample) -> None:
    print(
        "[probe step22] sample: "
        f"update_step={update_step} "
        f"segment_ids={sample.segment_ids.detach().cpu().tolist()} "
        f"sources={list(sample.source)} "
        f"priority_mean={float(sample.priority.float().mean().detach().cpu()):.6f} "
        f"staleness_mean={float(sample.staleness.float().mean().detach().cpu()):.6f} "
        f"valid_count={int(sample.valid_mask.bool().sum().detach().cpu().item())}",
        flush=True,
    )


def _print_sampler_summary(update_step: int, summary: dict[str, object]) -> None:
    print(
        "[FrontRES Segment Sampler] "
        f"update_step={update_step} "
        f"global={int(summary['sampler_source_global_count'])} "
        f"replay={int(summary['sampler_source_replay_count'])} "
        f"review={int(summary['sampler_source_review_count'])} "
        f"replay_pool_size={int(summary['sampler_replay_pool_size'])} "
        f"review_pool_size={int(summary['sampler_review_pool_size'])} "
        f"priority_mean={float(summary['sampler_priority_mean']):.6f} "
        f"solved_frac={float(summary['sampler_solved_frac']):.4f} "
        f"hopeless_frac={float(summary['sampler_hopeless_frac']):.4f} "
        f"stale_review_count={int(summary['sampler_stale_review_count'])}",
        flush=True,
    )
