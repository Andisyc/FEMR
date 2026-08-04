"""Read-only sampler evidence projection and human-facing reporting."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import torch

from rsl_rl.frontres.frontres_segment_planning import (
    FrontRESSegmentRolloutEvidence,
    FrontRESSegmentSample,
)
from rsl_rl.frontres.frontres_segment_sampler import (
    FrontRESSegmentSampler,
)
from rsl_rl.runners.frontres_segment_probe_logging import (
    format_probe_number as _fmt_num,
    format_probe_percent as _fmt_pct,
    probe_log_block as _log_block,
)
from rsl_rl.runners.frontres_segment_runner_boundary import frontres_runner_cfg_get

_VERBOSE_PROBE_BATCH_LIMIT = 16


def _kv_lines(prefix: str, values: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"  {prefix}.{key}: {value}" for key, value in values.items())


def build_live_sampler_evidence(
    sample: FrontRESSegmentSample,
    summary: dict[str, object],
    *,
    horizon_k: int,
    reset_result: Any | None = None,
    print_probe: bool = True,
) -> FrontRESSegmentRolloutEvidence:
    """Construct sampler evidence from the formal paired Gain summary.

    Status: active formal evidence boundary; legacy score fields are compatibility
    payload only and cannot affect active sampler decisions.
    Upstream: live probe summary. Downstream: Segment sampler evidence update.
    Evidence: contract-confirmed by frontres_segment_live_sampler_contract.py.
    Gap: real simulator Gain population remains an S4 boundary.
    """
    ids = sample.segment_ids.detach().clone().long()
    row_count = _summary_int(summary, "evidence_row_count", int(ids.numel()))
    if 0 < row_count < int(ids.numel()):
        ids = ids[:row_count]
    n = int(ids.numel())
    device = ids.device
    horizon = _horizon_vector(horizon_k, n=n, device=device)
    reset_success = _reset_success_for_sample(reset_result, n=n, device=device)
    reward = _summary_vector(
        summary,
        keys=("evidence_reward_per_sample", "storage_reward_per_sample", "reward_per_sample"),
        n=n,
        device=device,
        default=_summary_float(summary, "storage_reward_mean", _summary_float(summary, "reward_mean", 0.0)),
    ).float()
    rollout_valid = _summary_bool_vector(
        summary,
        keys=("evidence_valid_mask_per_sample", "storage_valid_mask_per_sample"),
        n=n,
        device=device,
        default=bool(_summary_int(summary, "ppo_valid_count", 0) > 0 and _summary_float(summary, "storage_valid_frac", 0.0) > 0.0),
    )
    fall = _summary_bool_vector(
        summary,
        keys=("evidence_done_any_per_sample", "done_any_per_sample"),
        n=n,
        device=device,
        default=bool(_summary_float(summary, "done_frac", 0.0) >= 0.5),
    )
    score_noisy = _summary_vector(
        summary,
        keys=("score_noisy_per_sample", "noisy_score_per_sample", "baseline_score_per_sample"),
        n=n,
        device=device,
        default=float("nan"),
    ).float()
    score_repaired = _summary_vector(
        summary,
        keys=("score_repaired_per_sample", "repaired_score_per_sample"),
        n=n,
        device=device,
        default=float("nan"),
    ).float()
    gain_source = str(summary.get("gain_source", ""))
    if gain_source != "FRS-GAIN-v002":
        raise ValueError(
            "sampler evidence requires gain_source=FRS-GAIN-v002; "
            f"got {gain_source or 'UNCONFIRMED'}"
        )
    formal_gain = _required_gain_vector(summary, "gain_total_per_sample", n=n, device=device)
    gain_style = _required_gain_vector(summary, "gain_style_per_sample", n=n, device=device)
    gain_physics = _required_gain_vector(summary, "gain_physics_per_sample", n=n, device=device)
    repair_cost = _required_gain_vector(summary, "gain_repair_cost_per_sample", n=n, device=device)
    has_real_scores = torch.isfinite(score_noisy).all() and torch.isfinite(score_repaired).all()
    if has_real_scores:
        score_noisy = score_noisy.clamp(0.0, 1.0)
        score_repaired = score_repaired.clamp(0.0, 1.0)
    gain = formal_gain
    valid_reward = rollout_valid & reset_success
    if print_probe:
        print_frontres_sampler_evidence_probe(
            ids,
            reward,
            reset_success,
            rollout_valid,
            valid_reward,
            fall,
            gain,
            score_noisy=score_noisy,
            score_repaired=score_repaired,
            evidence_source=gain_source,
        )
    return FrontRESSegmentRolloutEvidence(
        segment_ids=ids,
        reset_success=reset_success,
        score_noisy=score_noisy,
        score_repaired=score_repaired,
        score_clean=torch.ones(n, dtype=torch.float32, device=device),
        gain_over_noisy=gain,
        fall_repaired=fall,
        contact_consistency=torch.ones(n, dtype=torch.float32, device=device),
        action_norm=torch.ones(n, dtype=torch.float32, device=device),
        valid_reward=valid_reward,
        horizon_k=horizon,
        gain_total=formal_gain,
        gain_style=gain_style,
        gain_physics=gain_physics,
        repair_cost=repair_cost,
        gain_source=gain_source,
    )


def _reset_success_for_sample(reset_result: Any | None, *, n: int, device: torch.device) -> torch.Tensor:
    if reset_result is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    success = getattr(reset_result, "success_mask", None)
    if success is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    success = success.to(device=device).bool().reshape(-1)
    if int(success.numel()) < n:
        raise ValueError(f"reset_success must have at least {n} rows, got {int(success.numel())}")
    if int(success.numel()) > n:
        success = success[:n]
    return success.detach()


def summarize_sampler_step(sampler: FrontRESSegmentSampler, sample: FrontRESSegmentSample) -> dict[str, object]:
    stats = sampler.stats()
    counts = Counter(sample.source)
    trial_counts = Counter(sample.trial_role)
    stale_review_count = int(((sampler.staleness > 0.0) & sampler.solved & (~sampler.invalid)).sum().item())
    return {
        "sampler_update": True,
        "sampler_batch_size": int(sample.segment_ids.numel()),
        "sampler_source_global_count": int(counts.get("global", 0)),
        "sampler_source_replay_count": int(counts.get("replay", 0)),
        "sampler_source_review_count": int(counts.get("review", 0)),
        "sampler_trial_policy_count": int(trial_counts.get("policy", 0)),
        "sampler_trial_search_count": int(trial_counts.get("search", 0)),
        "sampler_budget_trial_count_mean": float(sample.rollout_trial_count.float().mean().item())
        if isinstance(sample.rollout_trial_count, torch.Tensor) and sample.rollout_trial_count.numel() > 0
        else 0.0,
        "sampler_budget_horizon_mean": float(sample.horizon_k.float().mean().item())
        if isinstance(sample.horizon_k, torch.Tensor) and sample.horizon_k.numel() > 0
        else 0.0,
        "sampler_replay_pool_size": int(stats.replay_pool_size),
        "sampler_review_pool_size": int(stats.review_pool_size),
        "sampler_priority_mean": float(stats.priority_mean),
        "sampler_priority_p90": float(stats.priority_p90),
        "sampler_solved_frac": float(stats.solved_frac),
        "sampler_hopeless_frac": float(stats.hopeless_frac),
        "sampler_stale_review_count": stale_review_count,
    }


def _resolve_live_batch_size(runner: Any) -> int:
    env = getattr(runner, "env", None)
    return max(1, int(getattr(env, "num_envs", 1) or 1))


def _resolve_live_scorable_row_budget(runner: Any) -> int:
    """Return the FrontRES repair rows that can receive paired rollout scores."""
    batch_size = _resolve_live_batch_size(runner)
    cfg_present = getattr(runner, "cfg", None) is not None or getattr(runner, "alg_cfg", None) is not None
    if not cfg_present:
        return batch_size
    use_quartet_reward = bool(frontres_runner_cfg_get(runner, "frontres_candidate_rollout_enabled", False))
    divisor = 4 if use_quartet_reward else 3
    return max(1, batch_size // divisor)


def _resolve_live_max_horizon_k(runner: Any) -> int:
    alg = getattr(runner, "alg", None)
    return max(1, int(getattr(alg, "frontres_segment_max_horizon_k", getattr(alg, "frontres_segment_k", 1))))


def _sample_frontres_live_segment_rows(runner: Any, sampler: FrontRESSegmentSampler) -> FrontRESSegmentSample:
    row_budget = _resolve_live_scorable_row_budget(runner)
    max_horizon_k = _resolve_live_max_horizon_k(runner)
    if hasattr(sampler, "sample_rollout_rows"):
        return sampler.sample_rollout_rows(row_budget, max_horizon_k=max_horizon_k)
    return sampler.sample(row_budget, max_horizon_k=max_horizon_k)


def sample_frontres_live_segment_rows(runner: Any, sampler: FrontRESSegmentSampler) -> FrontRESSegmentSample:
    """Public compatibility seam for independently sampled legacy evaluation rows."""

    return _sample_frontres_live_segment_rows(runner, sampler)


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


def _horizon_vector(horizon_k: int | torch.Tensor | list[int] | tuple[int, ...], *, n: int, device: torch.device) -> torch.Tensor:
    if isinstance(horizon_k, torch.Tensor):
        horizon = horizon_k.to(device=device, dtype=torch.long).reshape(-1)
    elif isinstance(horizon_k, (list, tuple)):
        horizon = torch.tensor(list(horizon_k), dtype=torch.long, device=device).reshape(-1)
    else:
        return torch.full((n,), max(1, int(horizon_k)), dtype=torch.long, device=device)
    if int(horizon.numel()) < n:
        raise ValueError(f"horizon_k must have at least {n} rows, got {int(horizon.numel())}")
    return horizon[:n].clamp_min(1).detach().clone()


def _summary_vector(
    summary: dict[str, object],
    *,
    keys: tuple[str, ...],
    n: int,
    device: torch.device,
    default: float,
) -> torch.Tensor:
    for key in keys:
        if key not in summary:
            continue
        value = summary.get(key)
        tensor = _as_float_tensor(value, device=device)
        if tensor is None or int(tensor.numel()) == 0:
            continue
        if int(tensor.numel()) != n:
            raise ValueError(f"{key} must have {n} rows, got {int(tensor.numel())}")
        return tensor.reshape(-1).detach()
    return torch.full((n,), float(default), dtype=torch.float32, device=device)


def _summary_bool_vector(
    summary: dict[str, object],
    *,
    keys: tuple[str, ...],
    n: int,
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    for key in keys:
        if key not in summary:
            continue
        value = summary.get(key)
        tensor = _as_bool_tensor(value, device=device)
        if tensor is None or int(tensor.numel()) == 0:
            continue
        if int(tensor.numel()) != n:
            raise ValueError(f"{key} must have {n} rows, got {int(tensor.numel())}")
        return tensor.reshape(-1).detach()
    return torch.full((n,), bool(default), dtype=torch.bool, device=device)


def _as_float_tensor(value: object, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).reshape(-1)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.float32, device=device).reshape(-1)
    return None


def _as_bool_tensor(value: object, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device).bool().reshape(-1)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.bool, device=device).reshape(-1)
    return None


def _required_gain_vector(
    summary: dict[str, object],
    key: str,
    *,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Read one finite FRS-GAIN-v002 vector; never synthesize missing evidence."""

    if key not in summary:
        raise ValueError(f"sampler evidence requires {key}")
    value = _as_float_tensor(summary.get(key), device=device)
    if value is None or int(value.numel()) != n:
        got = 0 if value is None else int(value.numel())
        raise ValueError(f"{key} must have {n} rows, got {got}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{key} contains non-finite values")
    return value.detach()


def print_frontres_sampler_evidence_probe(
    ids: torch.Tensor,
    reward: torch.Tensor,
    reset_success: torch.Tensor,
    rollout_valid: torch.Tensor,
    valid_reward: torch.Tensor,
    fall: torch.Tensor,
    gain: torch.Tensor,
    *,
    score_noisy: torch.Tensor,
    score_repaired: torch.Tensor,
    evidence_source: str,
) -> None:
    print(
        _log_block(
            "[FrontRES Segment Evidence]",
            *_kv_lines(
                "evidence",
                {
                    "ids": _id_summary(ids),
                    "source": evidence_source,
                    "reset_valid": int(reset_success.bool().sum().detach().cpu().item()),
                    "rollout_valid": int(rollout_valid.bool().sum().detach().cpu().item()),
                    "valid_reward": int(valid_reward.bool().sum().detach().cpu().item()),
                    "fall_count": int(fall.bool().sum().detach().cpu().item()),
                },
            ),
            *_kv_lines(
                "score",
                {
                    "reward_min": _fmt_num(float(reward.min().detach().cpu().item()) if reward.numel() else 0.0),
                    "reward_max": _fmt_num(float(reward.max().detach().cpu().item()) if reward.numel() else 0.0),
                    "noisy": _fmt_num(float(score_noisy.mean().detach().cpu().item()) if score_noisy.numel() else 0.0),
                    "repaired": _fmt_num(
                        float(score_repaired.mean().detach().cpu().item()) if score_repaired.numel() else 0.0
                    ),
                    "gain": _fmt_num(float(gain.mean().detach().cpu().item()) if gain.numel() else 0.0),
                },
            ),
        ),
        flush=True,
    )


def frontres_sampler_verbose_probe_enabled(runner: Any, sample: FrontRESSegmentSample | None = None) -> bool:
    alg = getattr(runner, "alg", None)
    if bool(getattr(alg, "frontres_segment_verbose_probe", False)):
        return True
    if sample is None:
        return False
    return int(sample.segment_ids.numel()) <= _VERBOSE_PROBE_BATCH_LIMIT


def frontres_sampler_detail_log_enabled(runner: Any) -> bool:
    alg = getattr(runner, "alg", None)
    if bool(getattr(alg, "frontres_segment_verbose_probe", False)):
        return True
    count = int(getattr(runner, "_frontres_segment_live_detail_log_count", 0)) + 1
    runner._frontres_segment_live_detail_log_count = count
    warmup = max(0, int(getattr(alg, "frontres_segment_live_log_warmup", 3)))
    interval = max(1, int(getattr(alg, "frontres_segment_live_log_interval", 10)))
    return count <= warmup or count % interval == 0


def _id_summary(ids: torch.Tensor) -> str:
    ids = ids.detach().long().reshape(-1).cpu()
    count = int(ids.numel())
    if count == 0:
        return "count=0 id_min=-1 id_max=-1"
    return f"count={count} id_min={int(ids.min().item())} id_max={int(ids.max().item())}"


def _count_summary(items: tuple[str, ...] | list[str]) -> dict[str, int]:
    return dict(Counter(str(item) for item in items))


def _tensor_value_summary(name: str, value: object) -> str:
    if not isinstance(value, torch.Tensor):
        return f"{name}_count=0 {name}_min=0.000000 {name}_max=0.000000"
    tensor = value.detach().float().reshape(-1).cpu()
    if int(tensor.numel()) == 0:
        return f"{name}_count=0 {name}_min=0.000000 {name}_max=0.000000"
    return (
        f"{name}_count={int(tensor.numel())} "
        f"{name}_min={float(tensor.min().item()):.6f} "
        f"{name}_max={float(tensor.max().item()):.6f}"
    )


def _verbose_sample_lines(sample: FrontRESSegmentSample, *, verbose: bool) -> tuple[str, ...]:
    if not verbose:
        return ()
    horizon = sample.horizon_k.detach().cpu().tolist() if isinstance(sample.horizon_k, torch.Tensor) else []
    trial_index = sample.trial_index.detach().cpu().tolist() if isinstance(sample.trial_index, torch.Tensor) else []
    return (
        f"  sample.segment_ids: {sample.segment_ids.detach().cpu().tolist()}",
        f"  sample.sources: {list(sample.source)}",
        f"  sample.trial_roles: {list(sample.trial_role)}",
        f"  sample.trial_index: {trial_index}",
        f"  sample.budget_horizon: {horizon}",
    )


def frontres_sampler_verbose_batch_lines(
    sample: FrontRESSegmentSample,
    *,
    roles: tuple[str, ...],
    strength: object,
    verbose: bool,
) -> tuple[str, ...]:
    if not verbose:
        return ()
    strength_list = strength.detach().cpu().tolist() if isinstance(strength, torch.Tensor) else []
    return (
        f"  batch.segment_ids: {sample.segment_ids.detach().cpu().tolist()}",
        f"  batch.roles: {roles}",
        f"  batch.trial_roles: {list(sample.trial_role)}",
        f"  batch.strength: {strength_list}",
    )


def print_frontres_sampler_sample_probe(update_step: int, sample: FrontRESSegmentSample, *, verbose: bool = False) -> None:
    print(
            _log_block(
                "[FrontRES Segment Sample]",
                *_kv_lines(
                    "sample",
                    {
                        "update_step": update_step,
                        "ids": _id_summary(sample.segment_ids),
                        "source_counts": _count_summary(list(sample.source)),
                        "priority": _fmt_num(sample.priority.float().mean().detach().cpu()),
                        "staleness": _fmt_num(sample.staleness.float().mean().detach().cpu()),
                        "valid_count": int(sample.valid_mask.bool().sum().detach().cpu().item()),
                        "trial_role_counts": _count_summary(list(sample.trial_role)),
                        "budget_horizon": _tensor_value_summary("budget_horizon", sample.horizon_k),
                    },
                ),
                *_verbose_sample_lines(sample, verbose=verbose),
            ),
        flush=True,
    )


def print_frontres_sampler_summary(update_step: int, summary: dict[str, object]) -> None:
    print(
            _log_block(
                "[FrontRES Segment Sampler]",
                *_kv_lines(
                    "sampler",
                    {
                        "update_step": update_step,
                        "src": (
                            f"global:{int(summary['sampler_source_global_count'])},"
                            f"replay:{int(summary['sampler_source_replay_count'])},"
                            f"review:{int(summary['sampler_source_review_count'])}"
                        ),
                        "pool": (
                            f"replay:{int(summary['sampler_replay_pool_size'])},"
                            f"review:{int(summary['sampler_review_pool_size'])}"
                        ),
                        "trial": (
                            f"policy:{int(summary.get('sampler_trial_policy_count', 0))},"
                            f"search:{int(summary.get('sampler_trial_search_count', 0))},"
                            f"budget_mean:{_fmt_num(summary.get('sampler_budget_trial_count_mean', 0.0))},"
                            f"horizon_mean:{_fmt_num(summary.get('sampler_budget_horizon_mean', 0.0))}"
                        ),
                        "priority": _fmt_num(summary["sampler_priority_mean"]),
                        "useful": (
                            f"mean:{_fmt_num(summary.get('sampler_update_useful_mean', 0.0))},"
                            f"max:{_fmt_num(summary.get('sampler_update_useful_max', 0.0))}"
                        ),
                        "priority_flow": (
                            f"before:{_fmt_num(summary.get('sampler_update_priority_before_mean', 0.0))},"
                            f"after:{_fmt_num(summary.get('sampler_update_priority_after_mean', 0.0))},"
                            f"max:{_fmt_num(summary.get('sampler_update_priority_after_max', 0.0))}"
                        ),
                        "gain": (
                            f"mean:{_fmt_num(summary.get('sampler_update_gain_mean', 0.0))},"
                            f"pos:{_fmt_pct(summary.get('sampler_update_gain_pos_frac', 0.0))}"
                        ),
                        "oracle": (
                            f"gap:{_fmt_num(summary.get('sampler_update_oracle_gap_mean', 0.0))},"
                            f"confidence:{_fmt_num(summary.get('sampler_update_confidence_mean', 0.0))},"
                            f"delayed:{int(summary.get('sampler_update_delayed_regret_count', 0))}"
                        ),
                        "update": (
                            f"valid:{int(summary.get('sampler_update_valid_count', 0))},"
                            f"fall:{int(summary.get('sampler_update_fall_count', 0))},"
                            f"hopeless:{int(summary.get('sampler_update_hopeless_count', 0))},"
                            f"segments:{int(summary.get('sampler_update_segment_count', 0))},"
                            f"trials:{int(summary.get('sampler_update_trial_count', 0))},"
                            f"replay_candidates:{int(summary.get('sampler_update_replay_candidate_count', 0))}"
                        ),
                        "solved": _fmt_pct(summary["sampler_solved_frac"]),
                        "hopeless": _fmt_pct(summary["sampler_hopeless_frac"]),
                        "stale_review": int(summary["sampler_stale_review_count"]),
                    },
                ),
            ),
        flush=True,
    )


# Public, read-only formatting seams used by the sampler assembly shell.
summarize_frontres_sampler_ids = _id_summary
summarize_frontres_sampler_counts = _count_summary
summarize_frontres_sampler_tensor = _tensor_value_summary
resolve_frontres_live_scorable_row_budget = _resolve_live_scorable_row_budget
resolve_frontres_live_max_horizon_k = _resolve_live_max_horizon_k
