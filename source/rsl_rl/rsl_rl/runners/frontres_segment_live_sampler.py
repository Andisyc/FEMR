from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from types import SimpleNamespace
from typing import Any

import torch

import rsl_rl.frontres.frontres_local_scenario as _SAMPLER_MODULE
from rsl_rl.frontres.frontres_segment_dataset import load_stage1_cache_dataset
from rsl_rl.frontres.frontres_local_scenario import (
    FrontRESLocalScenarioLifecycle,
    FrontRESLocalScenarioMaterialization,
)
from rsl_rl.frontres.frontres_outer_scenario_replay import (
    FrontRESOuterReplayPlan,
    FrontRESOuterScenarioReplay,
    FrontRESScenarioKey,
    frontres_tensor_identity,
    isolated_frontres_perturbation_rng,
)
from rsl_rl.frontres.frontres_relational_scenario_replay import (
    FrontRESRelationalScenarioReplay,
)
from rsl_rl.frontres.frontres_segment_legacy_scenario import (
    FrontRESFixedNoisyScenarioLifecycle,
    FrontRESNoisyReferenceMaterialization,
)
from rsl_rl.frontres.frontres_segment_planning import (
    FrontRESFrozenPolicyTransactionPlan,
    FrontRESSegmentRolloutEvidence,
    FrontRESSegmentSample,
)
from rsl_rl.frontres.frontres_segment_storage_records import FrontRESV015GroupedCandidateMetadata
from rsl_rl.frontres.frontres_segment_sampler import (
    FrontRESSegmentSampler,
)
from rsl_rl.frontres.frontres_segment_warmup import (
    FRONTRES_V011_SELECTED_SEGMENT_COUNT,
    require_frontres_v013_campaign_schedule,
    resolve_frontres_k_stage_identity,
    sample_frontres_v013_dr_strength,
)
from rsl_rl.runners.frontres_formal_runtime_audit import print_sampler_audit
from rsl_rl.runners.frontres_segment_runner_boundary import frontres_runner_cfg_get as _runner_cfg_get
from rsl_rl.runners.frontres_segment_runtime_types import frontres_outer_scenario_replay
from rsl_rl.runners.frontres_segment_sampler_reporting import (
    build_live_sampler_evidence,
    frontres_sampler_detail_log_enabled,
    frontres_sampler_verbose_batch_lines,
    frontres_sampler_verbose_probe_enabled,
    print_frontres_sampler_evidence_probe,
    print_frontres_sampler_sample_probe,
    print_frontres_sampler_summary,
    resolve_frontres_live_max_horizon_k,
    resolve_frontres_live_scorable_row_budget,
    sample_frontres_live_segment_rows,
    summarize_frontres_sampler_counts,
    summarize_frontres_sampler_ids,
    summarize_frontres_sampler_tensor,
    summarize_sampler_step,
)
from rsl_rl.runners.frontres_segment_transaction import (
    FrontRESFrozenPolicySnapshot,
    FrontRESFrozenPolicyTransactionMetadata,
    FrontRESFormalTransactionAccumulator,
    FrontRESFormalTransactionPlan,
    bind_frontres_frozen_policy_transaction,
    capture_frontres_frozen_policy_snapshot,
    finalize_frontres_frozen_policy_transaction_metadata,
    validate_frontres_frozen_policy_transaction_plan,
)

_VERBOSE_PROBE_BATCH_LIMIT = 16
_LOG_SEPARATOR = "-" * 80


def _log_block(*lines: str) -> str:
    return "\n".join(("", _LOG_SEPARATOR, "", *lines))


def _kv_lines(prefix: str, values: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"  {prefix}.{key}: {value}" for key, value in values.items())


def _fmt_num(value: Any) -> str:
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    abs_value = abs(value)
    if abs_value != 0.0 and (abs_value >= 10000.0 or abs_value < 0.001):
        return f"{value:.3e}"
    return f"{value:.6f}"


def _fmt_pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


_validate_frozen_policy_transaction_plan = validate_frontres_frozen_policy_transaction_plan


def _resolve_num_segments(runner: Any) -> int:
    dataset = getattr(runner, "_frontres_segment_dataset", None)
    if dataset is not None and hasattr(dataset, "num_segments"):
        return max(1, int(dataset.num_segments()))
    env = getattr(runner, "env", None)
    return max(1, int(getattr(env, "num_envs", 1) or 1))


def initialize_frontres_segment_live_sampler(runner: Any) -> None:
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    if not bool(getattr(boundary, "requested", False) and getattr(boundary, "live_runner_enabled", False)):
        return
    _ensure_stage1_cache_dataset(runner)
    _ensure_stage1_index_reset_hook(runner)
    num_segments = _resolve_num_segments(runner)
    if getattr(runner, "_frontres_segment_sampler", None) is None:
        runner._frontres_segment_sampler = FrontRESSegmentSampler(
            num_segments=num_segments,
            global_frac=float(getattr(runner.alg, "frontres_segment_sampler_global_frac", 0.4)),
            replay_frac=float(getattr(runner.alg, "frontres_segment_sampler_replay_frac", 0.5)),
            review_frac=float(getattr(runner.alg, "frontres_segment_sampler_review_frac", 0.1)),
            seed=int(getattr(runner, "seed", 0) or 0),
            device=getattr(runner, "device", "cpu"),
        )
    if (
        bool(getattr(runner.alg, "frontres_formal_transaction_enabled", False))
        and getattr(runner, "_frontres_outer_scenario_replay", None) is None
    ):
        replay_cls = (
            FrontRESRelationalScenarioReplay
            if bool(getattr(runner.alg, "frontres_relational_actor_only", False))
            else FrontRESOuterScenarioReplay
        )
        runner._frontres_outer_scenario_replay = replay_cls(
            global_frac=float(getattr(runner.alg, "frontres_segment_sampler_global_frac", 0.4)),
            replay_frac=float(getattr(runner.alg, "frontres_segment_sampler_replay_frac", 0.5)),
            review_frac=float(getattr(runner.alg, "frontres_segment_sampler_review_frac", 0.1)),
            seed=int(getattr(runner, "seed", 0) or 0),
        )
    print(
        _log_block(
            "[FrontRES Segment Sampler Ready]",
            "  config: "
            f"num_segments={num_segments} "
            f"global_frac={runner._frontres_segment_sampler.global_frac:.3f} "
            f"replay_frac={runner._frontres_segment_sampler.replay_frac:.3f} "
            f"review_frac={runner._frontres_segment_sampler.review_frac:.3f}",
        ),
        flush=True,
    )


def ensure_frontres_readonly_reset_support(runner: Any) -> None:
    """Install cache-backed index/reset support without creating Replay or a sampler."""
    sampler_before = getattr(runner, "_frontres_segment_sampler", None)
    _ensure_stage1_cache_dataset(runner)
    _ensure_stage1_index_reset_hook(runner)
    if getattr(runner, "_frontres_segment_sampler", None) is not sampler_before:
        raise RuntimeError("read-only reset support must not create or replace the Segment sampler")


def ensure_frontres_policy_quality_reset_support(runner: Any) -> None:
    """Historical evaluator alias for the shared read-only reset owner."""

    ensure_frontres_readonly_reset_support(runner)


def _ensure_stage1_index_reset_hook(runner: Any) -> None:
    dataset = getattr(runner, "_frontres_segment_dataset", None)
    metadata = dataset.cache_metadata() if dataset is not None and hasattr(dataset, "cache_metadata") else None
    if not isinstance(metadata, dict) or not bool(metadata.get("index_only", False)):
        return
    cache_amass_root = str(metadata.get("amass_root", "") or "")
    if not cache_amass_root:
        raise ValueError("index-only Stage 1 dataset metadata is missing amass_root")
    from rsl_rl.frontres.frontres_segment_stage1_env_hooks import ensure_frontres_segment_index_reset_hook

    adapter = ensure_frontres_segment_index_reset_hook(
        runner.env,
        amass_root=cache_amass_root,
        robot_name=str(getattr(runner.alg, "frontres_segment_reset_robot_name", "robot")),
        trace=bool(getattr(runner.alg, "frontres_segment_reset_trace", True)),
    )
    probe = adapter.frontres_motion_loader_probe()
    live_amass_root = adapter.frontres_loaded_motion_root()
    filter_probe = None
    if hasattr(dataset, "filter_to_loaded_motion_paths"):
        filter_probe = dataset.filter_to_loaded_motion_paths(
            adapter.frontres_loaded_motion_paths(),
            amass_root=live_amass_root,
        )
    print(
        _log_block(
            "[FrontRES Segment Index Reset Hook Ready]",
            "  loader: "
            f"cache_amass_root={cache_amass_root} "
            f"live_amass_root={live_amass_root} "
            f"loaded_motion_count={probe.get('loaded_motion_count')} "
            f"all_motion_count={probe.get('all_motion_count')} "
            f"first_loaded_motion={probe.get('first_loaded_motion')}",
            "  index_filter: "
            f"{filter_probe if filter_probe is not None else 'not_applied'}",
        ),
        flush=True,
    )


def _ensure_stage1_cache_dataset(runner: Any) -> None:
    if getattr(runner, "_frontres_segment_dataset", None) is not None:
        return
    alg = getattr(runner, "alg", None)
    cache_dir = str(getattr(alg, "frontres_segment_cache_dir", "") or "")
    if not cache_dir:
        print(
            _log_block(
                "[FrontRES Segment Dataset]",
                "  cache_load: skipped reason=no_cache_dir",
            ),
            flush=True,
        )
        return
    include_boundary = bool(getattr(alg, "frontres_segment_include_boundary_diagnostic", False))
    shard_cache_size = max(1, int(getattr(alg, "frontres_segment_shard_cache_size", 8)))
    dataset = load_stage1_cache_dataset(
        cache_dir,
        device=getattr(runner, "device", "cpu"),
        include_boundary_diagnostic=include_boundary,
        shard_cache_size=shard_cache_size,
    )
    runner._frontres_segment_dataset = dataset
    metadata = dataset.cache_metadata() if hasattr(dataset, "cache_metadata") else None
    print(
            _log_block(
                "[FrontRES Segment Dataset Ready]",
                *_kv_lines(
                    "cache",
                    {
                        "cache_dir": cache_dir,
                        "num_segments": dataset.num_segments(),
                        "include_boundary_diagnostic": include_boundary,
                        "shard_cache_size": shard_cache_size,
                    },
                ),
                f"  metadata: {metadata}",
            ),
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

    sample = _sample_live_segment_rows(runner, sampler)
    detail_log = _live_detail_log_enabled(runner)
    verbose_probe = _verbose_probe_enabled(runner, sample)
    if detail_log:
        _print_sample_probe(update_step, sample, verbose=verbose_probe)
    batch = _build_current_segment_batch(runner, sample, update_step=update_step, print_probe=detail_log)
    runner._frontres_segment_live_current_sample = sample
    runner._frontres_segment_live_current_batch = batch
    runner._frontres_segment_live_detail_log_enabled = detail_log
    adapter = getattr(getattr(runner, "env", None), "_frontres_segment_index_reset_adapter", None)
    old_adapter_trace = getattr(adapter, "trace", None)
    if adapter is not None and old_adapter_trace is not None:
        adapter.trace = bool(detail_log)
    reset_result = None
    try:
        try:
            summary = runner.run_frontres_segment_live_probe(init_at_random_ep_len=init_at_random_ep_len)
            reset_result = getattr(runner, "_frontres_segment_live_current_reset_result", None)
        finally:
            if adapter is not None and old_adapter_trace is not None:
                adapter.trace = old_adapter_trace
            runner._frontres_segment_live_current_sample = None
            runner._frontres_segment_live_current_batch = None
            runner._frontres_segment_live_current_reset_request = None
            runner._frontres_segment_live_current_reset_result = None
            runner._frontres_segment_live_detail_log_enabled = True

        evidence = build_live_sampler_evidence(
            sample,
            summary,
            horizon_k=sample.horizon_k if isinstance(sample.horizon_k, torch.Tensor) else int(getattr(runner.alg, "frontres_segment_k", 1)),
            reset_result=reset_result,
            print_probe=detail_log,
        )
    finally:
        _close_fixed_noisy_scenarios(batch)
    update_probe = sampler.update_with_probe(evidence)
    sampler_summary = summarize_sampler_step(sampler, sample)
    sampler_summary.update(
        {
            "sampler_update_valid_count": update_probe.valid_count,
            "sampler_update_fall_count": update_probe.fall_count,
            "sampler_update_gain_mean": update_probe.gain_mean,
            "sampler_update_gain_pos_frac": update_probe.gain_pos_frac,
            "sampler_update_useful_mean": update_probe.useful_mean,
            "sampler_update_useful_max": update_probe.useful_max,
            "sampler_update_priority_before_mean": update_probe.priority_before_mean,
            "sampler_update_priority_after_mean": update_probe.priority_after_mean,
            "sampler_update_priority_after_max": update_probe.priority_after_max,
            "sampler_update_replay_candidate_count": update_probe.replay_candidate_count,
            "sampler_update_hopeless_count": update_probe.hopeless_count,
            "sampler_update_delayed_regret_count": update_probe.delayed_regret_count,
            "sampler_update_segment_count": update_probe.segment_count,
            "sampler_update_trial_count": update_probe.trial_count,
            "sampler_update_oracle_gap_mean": update_probe.oracle_gap_mean,
            "sampler_update_confidence_mean": update_probe.confidence_mean,
        }
    )
    summary.update(sampler_summary)
    # AUDIT-SAMPLER-01: 检查 Segment Replay 与 per-row K, 位于 rollout evidence -> sampler summary.
    # Result: PENDING_LIVE.
    print_sampler_audit(runner, update_step=update_step, sample=sample, batch=batch, summary=summary)
    if detail_log:
        _print_sampler_summary(update_step, sampler_summary)
    return summary


def _stage3_index_frontier_scale(runner: Any) -> float:
    value = getattr(runner, "_dr_scale", None)
    if value is not None:
        return float(value)
    return float(_runner_cfg_get(runner, "frontres_dr_scale", _runner_cfg_get(runner, "dr_scale_init", 1.0)))


def _stage3_index_progress(runner: Any, update_step: int) -> float:
    if getattr(runner, "_frontres_segment_sequence_eval_seed", None) is not None:
        # Evaluation compares policies under one fixed, fully materialized
        # perturbation curriculum rather than their training-time progress.
        return 1.0
    current_iter = int(getattr(runner, "current_learning_iteration", 0) or 0)
    max_iter = max(1, int(_runner_cfg_get(runner, "max_iterations", 1)))
    return max(0.0, min(1.0, (current_iter + int(update_step)) / float(max_iter)))


def _index_only_segment_batch(batch: Any) -> bool:
    families = tuple(getattr(batch, "perturbation_family", ()) or ())
    if families:
        return all(str(family) == "index_only" for family in families)
    specs = tuple(getattr(batch, "specs", ()) or ())
    return bool(specs) and all(str(getattr(spec, "perturbation_family", "")) == "index_only" for spec in specs)


def _build_stage3_index_perturbation_plan(runner: Any, batch: Any, *, update_step: int) -> Any | None:
    if not _index_only_segment_batch(batch):
        return None
    n = int(getattr(batch, "batch_size", int(batch.segment_ids.numel())))
    source_index = _source_index_for_batch(batch, n=n, device=batch.segment_ids.device)
    source_ids, source_inverse = torch.unique(source_index, sorted=True, return_inverse=True)
    source_count = int(source_ids.numel())
    alg = getattr(runner, "alg", None)
    if bool(getattr(alg, "frontres_formal_transaction_enabled", False)):
        schedule = tuple(getattr(alg, "frontres_segment_k_curriculum", ()) or ())
        require_frontres_v013_campaign_schedule(schedule)
        identity = resolve_frontres_k_stage_identity(
            schedule=schedule,
            committed_update_iteration=int(getattr(runner, "current_learning_iteration", 0) or 0),
            max_horizon_k=int(getattr(alg, "frontres_segment_max_horizon_k", 32)),
        )
        seq_idx = int(getattr(runner, "current_learning_iteration", 0) or 0) * 100000 + int(update_step)
        samples = tuple(
            sample_frontres_v013_dr_strength(identity, sample_key=seq_idx + int(source_id))
            for source_id in source_ids.detach().cpu().tolist()
        )
        source_strength = torch.tensor(
            [sample.strength for sample in samples],
            dtype=batch.perturbation_strength.dtype,
            device=batch.segment_ids.device,
        )
        perturbation_strength = source_strength.index_select(0, source_inverse)
        source_classes = tuple(sample.class_name for sample in samples)
        return SimpleNamespace(
            perturbation_family=tuple("local_rp" for _ in range(n)),
            perturbation_strength=perturbation_strength,
            source_index=source_index.detach().clone(),
            source_ids=source_ids.detach().clone(),
            source_perturbation_family=tuple("local_rp" for _ in range(source_count)),
            source_perturbation_strength=source_strength.detach().clone(),
            active_modes=("local_rp",),
            complexity="single",
            mix_mode="train-v013-four-class",
            mix_diag={name: source_classes.count(name) / max(1, source_count) for name in ("easy", "medium", "hard", "broken")},
            progress=float(identity.dr_progress),
            d_cap=float(identity.d_cap),
            dr_stage_fingerprint=str(identity.dr_stage_fingerprint),
            source_dr_class=source_classes,
            seq_idx=int(seq_idx),
        )
    # Historical non-formal routes retain the retired adaptive curriculum only
    # behind this explicit branch.  Importing it here keeps the formal
    # TRAIN-v015 route free of the old episode-length/frontier controller.
    from rsl_rl.frontres.frontres_dr_curriculum import sample_per_env_dr_strength, sample_perturbation_mix

    cfg = getattr(runner, "cfg", None) or getattr(runner, "alg_cfg", None) or {}
    eval_seed = getattr(runner, "_frontres_segment_sequence_eval_seed", None)
    if eval_seed is None:
        seq_idx = int(getattr(runner, "current_learning_iteration", 0) or 0) * 100000 + int(update_step)
    else:
        # Sequence eval must not change its corruption when comparing
        # checkpoints saved at different training iterations.
        seq_idx = int(eval_seed) * 100000 + int(update_step)
    progress = _stage3_index_progress(runner, update_step)
    mix_plan = sample_perturbation_mix(cfg, None, progress, seq_idx, source_count, is_frontres=True)
    frontier_scale = _stage3_index_frontier_scale(runner)
    dr_min = float(_runner_cfg_get(runner, "dr_min_scale", 0.0))
    dr_max = float(_runner_cfg_get(runner, "dr_max_scale", max(4.0, frontier_scale)))
    strength_plan = sample_per_env_dr_strength(
        cfg,
        frontier_scale,
        True,
        seq_idx,
        n_train=source_count,
        n_candidate=0,
        n_base=0,
        num_envs=source_count,
        dr_min=dr_min,
        dr_max=dr_max,
    )
    if strength_plan.scale_vector is None:
        source_strengths = [float(strength_plan.effective_scale)] * source_count
    else:
        source_strengths = [float(v) for v in strength_plan.scale_vector[:source_count]]
    source_family = tuple("+".join(group) for group in mix_plan.groups[:source_count])
    source_strength = torch.tensor(source_strengths, dtype=batch.perturbation_strength.dtype, device=batch.segment_ids.device)
    perturbation_strength = source_strength.index_select(0, source_inverse)
    perturbation_family = tuple(source_family[int(row)] for row in source_inverse.detach().cpu().tolist())
    return SimpleNamespace(
        perturbation_family=perturbation_family,
        perturbation_strength=perturbation_strength,
        source_index=source_index.detach().clone(),
        source_ids=source_ids.detach().clone(),
        source_perturbation_family=source_family,
        source_perturbation_strength=source_strength.detach().clone(),
        active_modes=tuple(mix_plan.active_modes),
        complexity=str(mix_plan.complexity),
        mix_mode=str(strength_plan.mix_mode),
        mix_diag=dict(strength_plan.diag),
        progress=float(progress),
        seq_idx=int(seq_idx),
    )


def _build_outer_replay_perturbation_plan(
    runner: Any,
    batch: Any,
    plan: FrontRESOuterReplayPlan,
) -> Any:
    plan.validate()
    n = int(getattr(batch, "batch_size", int(batch.segment_ids.numel())))
    source_index = _source_index_for_batch(batch, n=n, device=batch.segment_ids.device)
    if set(source_index.detach().cpu().tolist()) != set(range(8)):
        raise ValueError("TRAIN-v024 outer replay requires exactly eight source identities")
    selections = plan.selections
    source_strength = torch.tensor(
        [selection.perturbation_strength for selection in selections],
        dtype=batch.perturbation_strength.dtype,
        device=batch.segment_ids.device,
    )
    perturbation_strength = source_strength.index_select(0, source_index)
    perturbation_family = tuple(
        selections[int(source)].perturbation_family for source in source_index.detach().cpu().tolist()
    )
    curriculum = resolve_frontres_k_stage_identity(
        schedule=tuple(getattr(runner.alg, "frontres_segment_k_curriculum", ()) or ()),
        committed_update_iteration=int(getattr(runner, "current_learning_iteration", 0) or 0),
        max_horizon_k=int(getattr(runner.alg, "frontres_segment_max_horizon_k", 32)),
    )
    if curriculum.active_k != plan.active_k:
        raise RuntimeError("outer replay selection K differs from the active curriculum")
    classes = tuple(selection.dr_class for selection in selections)
    return SimpleNamespace(
        perturbation_family=perturbation_family,
        perturbation_strength=perturbation_strength,
        source_index=source_index.detach().clone(),
        source_ids=torch.arange(8, dtype=torch.long, device=batch.segment_ids.device),
        source_perturbation_family=tuple(selection.perturbation_family for selection in selections),
        source_perturbation_strength=source_strength.detach().clone(),
        active_modes=("local_rp",),
        complexity="single",
        mix_mode="train-v022-bounded-outer-scenario",
        mix_diag={name: classes.count(name) / 8.0 for name in set(classes)},
        progress=float(curriculum.dr_progress),
        d_cap=float(curriculum.d_cap),
        dr_stage_fingerprint=str(curriculum.dr_stage_fingerprint),
        source_dr_class=classes,
        seq_idx=int(getattr(runner, "current_learning_iteration", 0) or 0),
    )


def _attach_stage3_index_perturbation_plan(batch: Any, plan: Any | None) -> Any:
    if plan is None:
        return batch
    object.__setattr__(batch, "perturbation_strength", plan.perturbation_strength)
    object.__setattr__(batch, "stage3_index_perturbation_family", plan.perturbation_family)
    object.__setattr__(batch, "stage3_index_perturbation_strength", plan.perturbation_strength)
    object.__setattr__(batch, "stage3_index_perturbation_plan", plan)
    return batch


def _source_index_for_batch(batch: Any, *, n: int, device: torch.device | str) -> torch.Tensor:
    value = getattr(batch, "frontres_segment_source_index", None)
    if value is None:
        return torch.arange(n, dtype=torch.long, device=device)
    source_index = torch.as_tensor(value, dtype=torch.long, device=device).flatten()
    if int(source_index.numel()) != int(n) or bool((source_index < 0).any()):
        raise ValueError("frontres_segment_source_index must be a nonnegative [B] tensor")
    return source_index.detach().clone()


def _attach_frontres_segment_trial_plan(batch: Any, sample: FrontRESSegmentSample) -> Any:
    # QUALITY-DATA-01: 检查 sampled segment -> policy/search role -> gradient-bearing rows.
    # Result: PENDING_Q_EVIDENCE.
    # B1: sample source/global-replay-review identity 在 trial expansion 前可见.
    # B2: policy/search role 与 K/difficulty 在这里首次绑定到 batch rows.
    # B3: storage/PPO 消费前可统计 unique/repeat/staleness 与 valid policy rows.
    roles = tuple(getattr(sample, "trial_role", ()) or ())
    if roles and len(roles) == int(sample.segment_ids.numel()):
        object.__setattr__(batch, "frontres_segment_trial_role", roles)
    source_index = getattr(sample, "source_index", None)
    if isinstance(source_index, torch.Tensor) and int(source_index.numel()) == int(sample.segment_ids.numel()):
        object.__setattr__(batch, "frontres_segment_source_index", source_index.detach().clone())
    trial_index = getattr(sample, "trial_index", None)
    if isinstance(trial_index, torch.Tensor) and int(trial_index.numel()) == int(sample.segment_ids.numel()):
        object.__setattr__(batch, "frontres_segment_trial_index", trial_index.detach().clone())
    horizon_k = getattr(sample, "horizon_k", None)
    if isinstance(horizon_k, torch.Tensor) and int(horizon_k.numel()) == int(sample.segment_ids.numel()):
        object.__setattr__(batch, "frontres_segment_budget_horizon_k", horizon_k.detach().clone())
    return batch


def _require_frontres_future_offsets(runner: Any) -> tuple[int, ...]:
    raw = _runner_cfg_get(runner, "frontres_future_offsets", None)
    if raw is None or isinstance(raw, (str, bytes)):
        raise ValueError(
            "fixed Noisy Segment Replay requires explicit nonempty frontres_future_offsets; no legacy default is allowed"
        )
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    try:
        offsets = tuple(int(value) for value in raw)
    except TypeError as exc:
        raise ValueError("frontres_future_offsets must be an ordered sequence of positive integers") from exc
    if not offsets or any(offset <= 0 for offset in offsets) or tuple(sorted(set(offsets))) != offsets:
        raise ValueError(
            f"frontres_future_offsets must be nonempty, positive, ordered, and unique; got {offsets}"
        )
    return offsets


def _fixed_noisy_materializer_adapter(runner: Any) -> Any:
    env = getattr(runner, "env", None)
    candidates = (env, getattr(env, "unwrapped", None))
    for owner in candidates:
        adapter = getattr(owner, "_frontres_segment_index_reset_adapter", None)
        if callable(getattr(adapter, "materialize_frontres_fixed_noisy_tape", None)):
            return adapter
    raise RuntimeError(
        "fixed Noisy Segment Replay requires the Stage 1 index-reset adapter with a command-owned tape materializer"
    )


def _local_scenario_materializer_adapter(runner: Any) -> Any:
    env = getattr(runner, "env", None)
    candidates = (env, getattr(env, "unwrapped", None))
    for owner in candidates:
        adapter = getattr(owner, "_frontres_segment_index_reset_adapter", None)
        if callable(getattr(adapter, "materialize_frontres_local_scenario", None)):
            return adapter
    raise RuntimeError(
        "v015 local scenario requires the Stage 1 index-reset adapter with "
        "MultiMotionCommand.materialize_frontres_local_scenario()"
    )


def _local_scenario_candidate_eligibility(
    runner: Any,
    *,
    horizon_k: int,
    intent_horizon: int,
) -> Callable[[int], bool]:
    """Bind the active H/K budget to immutable Segment motion/frame specs."""

    dataset = getattr(runner, "_frontres_segment_dataset", None)
    specs_by_id = getattr(dataset, "_spec_by_id", None)
    if not isinstance(specs_by_id, dict):
        raise RuntimeError("v015 source eligibility requires the Stage-1 Segment spec index")
    adapter = _local_scenario_materializer_adapter(runner)
    is_materializable = getattr(adapter, "frontres_local_scenario_is_materializable", None)
    if not callable(is_materializable):
        raise RuntimeError("v015 source eligibility requires the Stage-1 unclamped frame-budget accessor")

    def candidate_is_eligible(segment_id: int) -> bool:
        spec = specs_by_id.get(int(segment_id))
        if spec is None:
            raise RuntimeError(f"v015 source eligibility cannot resolve segment_id={int(segment_id)}")
        motion_id = str(getattr(spec, "motion_id", ""))
        start_frame = getattr(spec, "start_frame", None)
        if not motion_id or start_frame is None:
            raise RuntimeError(f"v015 source eligibility requires motion/frame identity for segment_id={int(segment_id)}")
        return bool(
            is_materializable(
                motion_id=motion_id,
                start_frame=int(start_frame),
                horizon_k=int(horizon_k),
                intent_horizon=int(intent_horizon),
            )
        )

    return candidate_is_eligible


def _local_scenario_transaction_id(runner: Any, *, update_step: int) -> str:
    sequence = int(getattr(runner, "_frontres_local_scenario_transaction_sequence", 0)) + 1
    runner._frontres_local_scenario_transaction_sequence = sequence
    iteration = int(getattr(runner, "current_learning_iteration", 0) or 0)
    return f"frontres-local-scenario:i{iteration}:u{int(update_step)}:n{sequence}"


def _attach_frontres_local_scenarios(
    runner: Any,
    batch: Any,
    sample: FrontRESSegmentSample,
    *,
    update_step: int,
    transaction_id: str | None = None,
) -> Any:
    """Attach separated v015 local-scenario carriers once per selected source.

    This function is the selection/materialization boundary. The explicit
    Step 5A-S0 sentinel may consume its sealed carrier through later owners;
    this function itself does not reset, sample an action, execute GMT, or
    update PPO.
    """

    if not _index_only_segment_batch(batch):
        return batch
    if getattr(batch, "frontres_fixed_noisy_tape", None) is not None:
        raise RuntimeError("v015 local scenario cannot mix with a legacy complete fixed-Noisy tape")
    if getattr(batch, "frontres_segment_transaction_id", None) is not None:
        raise RuntimeError(
            "v015 local scenario collection cannot enter the frozen-policy transaction route before Step 2"
        )
    future_offsets = _require_frontres_future_offsets(runner)
    adapter = _local_scenario_materializer_adapter(runner)
    batch_size = int(batch.segment_ids.numel())
    source_index = _source_index_for_batch(batch, n=batch_size, device=batch.segment_ids.device)
    horizon_k = getattr(batch, "frontres_segment_budget_horizon_k", None)
    if not isinstance(horizon_k, torch.Tensor):
        raise ValueError("v015 local scenario requires source-aligned frontres_segment_budget_horizon_k")
    horizon_k = horizon_k.detach().to(device=batch.segment_ids.device, dtype=torch.long).flatten()
    if int(horizon_k.numel()) != batch_size or bool((horizon_k <= 0).any()):
        raise ValueError("frontres_segment_budget_horizon_k must be positive [B] data")
    families = tuple(
        str(value)
        for value in (
            getattr(batch, "stage3_index_perturbation_family", ()) or getattr(batch, "perturbation_family", ()) or ()
        )
    )
    strengths = getattr(batch, "stage3_index_perturbation_strength", getattr(batch, "perturbation_strength", None))
    if len(families) != batch_size or not isinstance(strengths, torch.Tensor):
        raise ValueError("v015 local scenario requires physical family and strength for every batch row")
    strengths = strengths.detach().to(device=batch.segment_ids.device, dtype=torch.float32).flatten()
    if int(strengths.numel()) != batch_size:
        raise ValueError("v015 local scenario strength must be [B]")
    specs = tuple(getattr(batch, "specs", ()) or ())
    if len(specs) != batch_size:
        raise ValueError("v015 local scenario requires one Stage 1 index spec per batch row")

    source_rows: dict[int, int] = {}
    source_reference: dict[int, tuple[str, int, int, str, float, str]] = {}
    for row, source in enumerate(source_index.detach().cpu().tolist()):
        spec = specs[row]
        motion_id = str(getattr(spec, "motion_id", ""))
        start_frame = getattr(spec, "start_frame", None)
        if not motion_id or start_frame is None:
            raise ValueError("v015 local scenario specs require motion_id and start_frame")
        segment_id = int(batch.segment_ids[row].item())
        source_identity = f"motion={motion_id}|frame={int(start_frame)}|segment={segment_id}"
        reference = (
            motion_id,
            int(start_frame),
            int(horizon_k[row].item()),
            families[row],
            float(strengths[row].item()),
            source_identity,
        )
        previous = source_reference.setdefault(int(source), reference)
        if previous != reference:
            raise ValueError(
                f"source_index={source} maps to multiple local scenario inputs: first={previous}, row_{row}={reference}"
            )
        source_rows.setdefault(int(source), row)

    if transaction_id is None:
        transaction_id = _local_scenario_transaction_id(runner, update_step=update_step)
    transaction_id = str(transaction_id)
    if not transaction_id:
        raise ValueError("v015 local scenario requires a non-empty immutable transaction id")
    x_t_identity_by_source = {source: reference[-1] for source, reference in source_reference.items()}
    intent_horizon = max(future_offsets)
    outer_replay_plan = getattr(batch, "frontres_outer_replay_plan", None)
    if outer_replay_plan is not None:
        if not isinstance(outer_replay_plan, FrontRESOuterReplayPlan):
            raise TypeError("outer replay batch contains a foreign selection plan")
        outer_replay_plan.validate()
        if outer_replay_plan.transaction_id != transaction_id:
            raise ValueError("outer replay plan transaction differs from local scenario lifecycle")

    def materialize(request: Any) -> Any:
        row = source_rows.get(int(request.source_index))
        if row is None:
            raise RuntimeError(f"missing selected source row for local scenario {request.source_index}")
        motion_id, start_frame, horizon, family, strength, _x_t_identity = source_reference[int(request.source_index)]
        if int(request.horizon_k) != horizon:
            raise RuntimeError("local scenario lifecycle changed the selected K horizon")
        if outer_replay_plan is None:
            payload = adapter.materialize_frontres_local_scenario(
                motion_id=motion_id,
                start_frame=start_frame,
                horizon_k=horizon,
                intent_horizon=intent_horizon,
                perturbation_family=family,
                perturbation_strength=strength,
            )
        else:
            selection = outer_replay_plan.selections[int(request.source_index)]
            if (
                selection.segment_id != int(request.segment_id)
                or selection.perturbation_family != family
                or selection.perturbation_strength != strength
            ):
                raise ValueError("outer replay selection changed before Scenario materialization")
            with isolated_frontres_perturbation_rng(
                selection.perturbation_seed,
                device=getattr(runner, "device", batch.segment_ids.device),
            ):
                payload = adapter.materialize_frontres_local_scenario(
                    motion_id=motion_id,
                    start_frame=start_frame,
                    horizon_k=horizon,
                    intent_horizon=intent_horizon,
                    perturbation_family=family,
                    perturbation_strength=strength,
                )
        if not isinstance(payload, dict):
            raise TypeError("local scenario adapter must return a dict payload")
        return FrontRESLocalScenarioMaterialization(
            current_root_artifact_t=payload.get("current_root_artifact_t"),
            clean_reference_t=payload.get("clean_reference_t"),
            intent_q29=payload.get("intent_q29"),
            clean_continuation=payload.get("clean_continuation"),
            expected_support=payload.get("expected_support"),
            expected_support_envelope=payload.get("expected_support_envelope"),
            provenance=payload.get("provenance"),
        )

    lifecycle = FrontRESLocalScenarioLifecycle(
        transaction_id=transaction_id,
        future_offsets=future_offsets,
        x_t_identity_by_source=x_t_identity_by_source,
        materialize_scenario=materialize,
    )
    rows = lifecycle.bind_rows(sample)
    artifacts = torch.stack([scenario.current_root_artifact_t for scenario in rows.scenarios], dim=0).to(
        batch.segment_ids.device
    )
    clean_reference_t = torch.stack([scenario.clean_reference_t for scenario in rows.scenarios], dim=0).to(
        batch.segment_ids.device
    )
    intent_q29 = torch.stack([scenario.intent_q29 for scenario in rows.scenarios], dim=0).to(batch.segment_ids.device)
    max_horizon = int(horizon_k.max().item())
    clean_continuation = torch.zeros(
        (batch_size, max_horizon, 65),
        dtype=artifacts.dtype,
        device=batch.segment_ids.device,
    )
    clean_continuation_mask = torch.zeros(
        (batch_size, max_horizon),
        dtype=torch.bool,
        device=batch.segment_ids.device,
    )
    expected_support = torch.zeros(
        (batch_size, max_horizon, 2), dtype=artifacts.dtype, device=batch.segment_ids.device
    )
    expected_support_envelope = torch.zeros(
        (batch_size, max_horizon, 6), dtype=artifacts.dtype, device=batch.segment_ids.device
    )
    for row, scenario in enumerate(rows.scenarios):
        continuation = scenario.clean_continuation.to(batch.segment_ids.device)
        length = int(continuation.shape[0])
        clean_continuation[row, :length] = continuation
        clean_continuation_mask[row, :length] = True
        expected_support[row, :length] = scenario.expected_support.to(batch.segment_ids.device)
        expected_support_envelope[row, :length] = scenario.expected_support_envelope.to(batch.segment_ids.device)
    if tuple(artifacts.shape) != (batch_size, 7):
        raise RuntimeError(f"local scenario current-root carrier must be [B,7], got {tuple(artifacts.shape)}")
    if tuple(intent_q29.shape) != (batch_size, intent_horizon + 1, 29):
        raise RuntimeError(
            "local scenario intent carrier must be [B,H_max+1,29], "
            f"got {tuple(intent_q29.shape)}"
        )
    if tuple(clean_continuation.shape) != (batch_size, max_horizon, 65):
        raise RuntimeError(
            "local scenario Clean continuation carrier must be [B,K_max,65], "
            f"got {tuple(clean_continuation.shape)}"
        )
    object.__setattr__(batch, "frontres_local_scenario_rows", rows)
    object.__setattr__(batch, "frontres_local_scenario_lifecycle", lifecycle)
    object.__setattr__(batch, "frontres_local_scenario_transaction_id", transaction_id)
    object.__setattr__(batch, "frontres_local_scenario_ids", rows.scenario_ids)
    object.__setattr__(batch, "frontres_local_scenario_hashes", rows.noisy_segment_hashes)
    object.__setattr__(batch, "frontres_local_scenario_x_t_identities", tuple(s.request.x_t_identity for s in rows.scenarios))
    object.__setattr__(batch, "frontres_local_scenario_current_root_artifact_t", artifacts.detach().clone())
    object.__setattr__(batch, "frontres_local_scenario_clean_reference_t", clean_reference_t.detach().clone())
    object.__setattr__(batch, "frontres_local_scenario_intent_q29", intent_q29.detach().clone())
    object.__setattr__(batch, "frontres_local_scenario_clean_continuation", clean_continuation.detach().clone())
    object.__setattr__(batch, "frontres_local_scenario_clean_continuation_lengths", rows.continuation_lengths)
    object.__setattr__(batch, "frontres_local_scenario_clean_continuation_mask", clean_continuation_mask.detach().clone())
    object.__setattr__(batch, "frontres_local_scenario_expected_support", expected_support.detach().clone())
    object.__setattr__(
        batch,
        "frontres_local_scenario_expected_support_envelope",
        expected_support_envelope.detach().clone(),
    )
    object.__setattr__(batch, "frontres_local_scenario_provenance", tuple(s.provenance for s in rows.scenarios))
    object.__setattr__(batch, "frontres_future_offsets", future_offsets)
    return batch


def close_frontres_local_scenarios(batch: Any) -> None:
    """Idempotently close every sealed scenario owned by one transaction batch."""

    lifecycle = getattr(batch, "frontres_local_scenario_lifecycle", None)
    rows = getattr(batch, "frontres_local_scenario_rows", None)
    if lifecycle is None or rows is None:
        return
    closed = list(getattr(batch, "frontres_local_scenario_closed_ids", ()) or ())
    closed_set = set(closed)
    for scenario_id in dict.fromkeys(rows.scenario_ids):
        scenario_id = str(scenario_id)
        if scenario_id in closed_set:
            continue
        lifecycle.close_scenario(scenario_id)
        closed.append(scenario_id)
        closed_set.add(scenario_id)
        object.__setattr__(batch, "frontres_local_scenario_closed_ids", tuple(closed))
    object.__setattr__(batch, "frontres_local_scenario_closed_ids", tuple(closed))


# Compatibility alias for historical focused contracts.
_close_frontres_local_scenarios = close_frontres_local_scenarios


def _fixed_noisy_transaction_id(runner: Any, *, update_step: int) -> str:
    sequence = int(getattr(runner, "_frontres_fixed_noisy_transaction_sequence", 0)) + 1
    runner._frontres_fixed_noisy_transaction_sequence = sequence
    iteration = int(getattr(runner, "current_learning_iteration", 0) or 0)
    return f"frontres-fixed-noisy:i{iteration}:u{int(update_step)}:n{sequence}"


def _attach_fixed_noisy_scenarios(
    runner: Any,
    batch: Any,
    sample: FrontRESSegmentSample,
    *,
    update_step: int,
) -> Any:
    """Bind one command-materialized Noisy tape per source before any reset/rollout."""

    if not _index_only_segment_batch(batch):
        return batch
    future_offsets = _require_frontres_future_offsets(runner)
    adapter = _fixed_noisy_materializer_adapter(runner)
    batch_size = int(batch.segment_ids.numel())
    source_index = _source_index_for_batch(batch, n=batch_size, device=batch.segment_ids.device)
    horizon_k = getattr(batch, "frontres_segment_budget_horizon_k", None)
    if not isinstance(horizon_k, torch.Tensor):
        raise ValueError("fixed Noisy Segment Replay requires source-aligned frontres_segment_budget_horizon_k")
    horizon_k = horizon_k.detach().to(device=batch.segment_ids.device, dtype=torch.long).flatten()
    if int(horizon_k.numel()) != batch_size or bool((horizon_k <= 0).any()):
        raise ValueError("frontres_segment_budget_horizon_k must be positive [B] data")
    common_frame_count = int(horizon_k.max().item()) + max(future_offsets)
    families = tuple(
        str(value)
        for value in (
            getattr(batch, "stage3_index_perturbation_family", ()) or getattr(batch, "perturbation_family", ()) or ()
        )
    )
    strengths = getattr(batch, "stage3_index_perturbation_strength", getattr(batch, "perturbation_strength", None))
    if len(families) != batch_size or not isinstance(strengths, torch.Tensor):
        raise ValueError("fixed Noisy Segment Replay requires physical family and strength for every batch row")
    strengths = strengths.detach().to(device=batch.segment_ids.device, dtype=torch.float32).flatten()
    if int(strengths.numel()) != batch_size:
        raise ValueError("fixed Noisy Segment Replay strength must be [B]")
    specs = tuple(getattr(batch, "specs", ()) or ())
    if len(specs) != batch_size:
        raise ValueError("fixed Noisy Segment Replay requires one Stage 1 index spec per batch row")

    source_rows: dict[int, int] = {}
    source_reference: dict[int, tuple[str, int, str, float]] = {}
    for row, source in enumerate(source_index.detach().cpu().tolist()):
        spec = specs[row]
        motion_id = str(getattr(spec, "motion_id", ""))
        start_frame = getattr(spec, "start_frame", None)
        if not motion_id or start_frame is None:
            raise ValueError("fixed Noisy Segment Replay specs require motion_id and start_frame")
        reference = (motion_id, int(start_frame), families[row], float(strengths[row].item()))
        previous = source_reference.setdefault(int(source), reference)
        if previous != reference:
            raise ValueError(
                f"source_index={source} maps to multiple materialization inputs: first={previous}, row_{row}={reference}"
            )
        source_rows.setdefault(int(source), row)

    bound_transaction_id = getattr(batch, "frontres_segment_transaction_id", None)
    if bound_transaction_id is None:
        transaction_id = _fixed_noisy_transaction_id(runner, update_step=update_step)
    else:
        transaction_id = str(bound_transaction_id)
        plan = getattr(batch, "frontres_segment_transaction_plan", None)
        snapshot = getattr(batch, "frontres_segment_frozen_policy_snapshot", None)
        _validate_frozen_policy_transaction_plan(plan)
        if not isinstance(snapshot, FrontRESFrozenPolicySnapshot):
            raise TypeError("pre-bound fixed Noisy transaction requires a captured frozen policy snapshot")
        if snapshot.transaction_id != transaction_id or str(plan.transaction_id) != transaction_id:
            raise ValueError("pre-bound fixed Noisy transaction identity is inconsistent")
        snapshot.verify_policy(getattr(getattr(runner, "alg", None), "policy", None))

    def materialize(request: Any) -> Any:
        row = source_rows.get(int(request.source_index))
        if row is None:
            raise RuntimeError(f"missing selected source row for fixed Noisy scenario {request.source_index}")
        motion_id, start_frame, family, strength = source_reference[int(request.source_index)]
        tape = adapter.materialize_frontres_fixed_noisy_tape(
            motion_id=motion_id,
            start_frame=start_frame,
            frame_count=common_frame_count,
            perturbation_family=family,
            perturbation_strength=strength,
        )
        return FrontRESNoisyReferenceMaterialization(
            reference_sequence=tape,
            provenance={
                "materializer_owner": "MultiMotionCommand",
                "motion_id": motion_id,
                "start_frame": start_frame,
                "perturbation_family": family,
                "perturbation_strength": strength,
                "carrier_feature_dim": int(tape.shape[-1]),
            },
        )

    lifecycle = FrontRESFixedNoisyScenarioLifecycle(
        transaction_id=transaction_id,
        future_offsets=future_offsets,
        materialize_reference=materialize,
    )
    rows = lifecycle.bind_rows(sample)
    tape = torch.stack([scenario.reference_sequence for scenario in rows.scenarios], dim=0).to(batch.segment_ids.device)
    if tuple(tape.shape) != (batch_size, common_frame_count, 65):
        raise RuntimeError(f"fixed Noisy tape must be [B,{common_frame_count},65], got {tuple(tape.shape)}")
    object.__setattr__(batch, "frontres_fixed_noisy_scenario_rows", rows)
    object.__setattr__(batch, "frontres_fixed_noisy_lifecycle", lifecycle)
    object.__setattr__(batch, "frontres_fixed_noisy_transaction_id", transaction_id)
    object.__setattr__(batch, "frontres_fixed_noisy_tape", tape.detach().clone())
    object.__setattr__(
        batch,
        "frontres_fixed_noisy_tape_lengths",
        torch.full((batch_size,), common_frame_count, dtype=torch.long, device=batch.segment_ids.device),
    )
    object.__setattr__(batch, "frontres_fixed_noisy_scenario_ids", rows.scenario_ids)
    object.__setattr__(batch, "frontres_fixed_noisy_segment_hashes", rows.noisy_segment_hashes)
    object.__setattr__(batch, "frontres_future_offsets", future_offsets)
    if bound_transaction_id is not None:
        finalize_frontres_frozen_policy_transaction_metadata(runner, batch)
    return batch


def _close_fixed_noisy_scenarios(batch: Any) -> None:
    lifecycle = getattr(batch, "frontres_fixed_noisy_lifecycle", None)
    rows = getattr(batch, "frontres_fixed_noisy_scenario_rows", None)
    if lifecycle is None or rows is None:
        return
    closed: list[str] = []
    for scenario_id in dict.fromkeys(rows.scenario_ids):
        lifecycle.close_scenario(scenario_id)
        closed.append(str(scenario_id))
    object.__setattr__(batch, "frontres_fixed_noisy_closed_scenario_ids", tuple(closed))


def _tensor_nonzero_frac(value: object) -> float:
    if not isinstance(value, torch.Tensor) or int(value.numel()) <= 0:
        return 0.0
    data = value.detach().reshape(-1)
    return float((data != 0).float().mean().cpu().item())


def _build_current_segment_batch(
    runner: Any,
    sample: FrontRESSegmentSample,
    *,
    update_step: int,
    print_probe: bool = True,
    v015_local_scenario_transaction_id: str | None = None,
    outer_replay_plan: FrontRESOuterReplayPlan | None = None,
) -> Any | None:
    dataset = getattr(runner, "_frontres_segment_dataset", None)
    if dataset is None or not hasattr(dataset, "get_segments"):
        if print_probe:
            alg = getattr(runner, "alg", None)
            cache_dir = str(getattr(alg, "frontres_segment_cache_dir", "") or "")
            sampler = getattr(runner, "_frontres_segment_sampler", None)
            sampler_segments = getattr(sampler, "num_segments", "n/a")
            print(
                _log_block(
                    "[FrontRES Segment Batch]",
                    *_kv_lines(
                        "skipped",
                        {
                            "reason": "no_dataset",
                            "cache_dir": cache_dir or "<empty>",
                            "has_dataset": dataset is not None,
                            "dataset_has_get_segments": hasattr(dataset, "get_segments"),
                            "sampler_segments": sampler_segments,
                        },
                    ),
                ),
                flush=True,
            )
        return None
    batch = dataset.get_segments(sample.segment_ids)
    _attach_frontres_segment_trial_plan(batch, sample)
    validation = dataset.validate_batch(batch) if hasattr(dataset, "validate_batch") else None
    valid_count = (
        int(validation.valid_mask.bool().sum().detach().cpu().item())
        if validation is not None and hasattr(validation, "valid_mask")
        else int(sample.segment_ids.numel())
    )
    dynamic_plan = (
        _build_outer_replay_perturbation_plan(runner, batch, outer_replay_plan)
        if outer_replay_plan is not None
        else _build_stage3_index_perturbation_plan(runner, batch, update_step=update_step)
    )
    batch = _attach_stage3_index_perturbation_plan(batch, dynamic_plan)
    if outer_replay_plan is not None:
        object.__setattr__(batch, "frontres_outer_replay_plan", outer_replay_plan)
    if v015_local_scenario_transaction_id is not None:
        batch = _attach_frontres_local_scenarios(
            runner,
            batch,
            sample,
            update_step=update_step,
            transaction_id=v015_local_scenario_transaction_id,
        )
    else:
        batch = _attach_fixed_noisy_scenarios(runner, batch, sample, update_step=update_step)
    roles = tuple(getattr(batch, "perturbation_role", ()))
    strength = getattr(batch, "perturbation_strength", None)
    dynamic_family = tuple(getattr(batch, "stage3_index_perturbation_family", ()) or ())
    verbose_probe = _verbose_probe_enabled(runner, sample)
    if print_probe:
        print(
            _log_block(
                "[FrontRES Segment Batch]",
                *_kv_lines(
                    "batch",
                    {
                        "update_step": update_step,
                        "ids": _id_summary(sample.segment_ids),
                        "valid_count": valid_count,
                        "role_counts": _count_summary(roles),
                        "trial_role_counts": _count_summary(tuple(getattr(batch, "frontres_segment_trial_role", ()) or ())),
                        "strength": _tensor_value_summary("strength", strength),
                        "budget_horizon": _tensor_value_summary(
                            "budget_horizon",
                            getattr(batch, "frontres_segment_budget_horizon_k", None),
                        ),
                        "strength_nonzero_frac": _fmt_pct(_tensor_nonzero_frac(strength)),
                        "dynamic_family_counts": _count_summary(dynamic_family),
                    },
                ),
                *_verbose_batch_lines(sample, roles=roles, strength=strength, verbose=verbose_probe),
            ),
            flush=True,
        )
    return batch


def build_frontres_current_segment_batch(
    runner: Any,
    sample: FrontRESSegmentSample,
    *,
    update_step: int,
    print_probe: bool = True,
    v015_local_scenario_transaction_id: str | None = None,
) -> Any | None:
    """Public compatibility seam for legacy evaluation batch construction."""

    return _build_current_segment_batch(
        runner,
        sample,
        update_step=update_step,
        print_probe=print_probe,
        v015_local_scenario_transaction_id=v015_local_scenario_transaction_id,
    )


def _sample_frontres_v015_transaction_sources(
    sampler: Any,
    *,
    selected_segment_count: int,
    max_horizon_k: int,
    candidate_is_eligible: Callable[[int], bool] | None = None,
) -> FrontRESSegmentSample:
    """Select exactly S distinct sources; TRAIN-v024 owns B8/M4, not sampler state."""

    if int(selected_segment_count) != FRONTRES_V011_SELECTED_SEGMENT_COUNT:
        raise RuntimeError("FRS-TRAIN-v024 requires exactly eight selected Scenario sources")
    max_draws = 64
    num_segments = int(getattr(sampler, "num_segments", 0) or 0)
    if 0 < num_segments < selected_segment_count:
        raise RuntimeError("FRS-TRAIN-v024 requires at least eight valid Scenario sources")

    candidates: list[FrontRESSegmentSample] = []
    candidate_ids: set[int] = set()

    drawn = 0
    draw_batch_size = 8
    while drawn < max_draws and len(candidates) < selected_segment_count:
        requested = min(draw_batch_size, max_draws - drawn)
        sampled = sampler.sample(requested, max_horizon_k=max_horizon_k)
        sampled_count = int(sampled.segment_ids.numel())
        if sampled_count <= 0:
            raise RuntimeError("v015 formal transaction sampler returned no candidate Segment")
        drawn += sampled_count
        for row in range(sampled_count):
            segment_id = int(sampled.segment_ids[row].item())
            if segment_id in candidate_ids:
                continue
            if candidate_is_eligible is not None and not bool(candidate_is_eligible(segment_id)):
                continue
            candidate_ids.add(segment_id)
            candidate = FrontRESSegmentSample(
                segment_ids=sampled.segment_ids[row : row + 1],
                source=(str(sampled.source[row]),),
                priority=sampled.priority[row : row + 1],
                staleness=sampled.staleness[row : row + 1],
                valid_mask=sampled.valid_mask[row : row + 1],
                segment_state=(
                    sampled.segment_state[row : row + 1]
                    if isinstance(sampled.segment_state, torch.Tensor)
                    else None
                ),
                rollout_trial_count=sampled.rollout_trial_count[row : row + 1],
                horizon_k=sampled.horizon_k[row : row + 1],
                budget_reason=(str(sampled.budget_reason[row]),),
                trial_role=("policy",),
                source_index=torch.zeros(1, dtype=torch.long, device=sampled.segment_ids.device),
                trial_index=torch.zeros(1, dtype=torch.long, device=sampled.segment_ids.device),
            )
            candidates.append(candidate)
            if len(candidates) == selected_segment_count:
                break

    if len(candidates) != selected_segment_count:
        raise RuntimeError(
            "FRS-TRAIN-v011 could not select exactly two eligible Segment sources "
            f"after {len(candidates)} distinct candidates"
        )
    selected = candidates

    device = selected[0].segment_ids.device
    segment_ids = torch.cat([sample.segment_ids for sample in selected], dim=0)
    segment_state_values = [sample.segment_state for sample in selected]
    segment_state = (
        torch.cat(segment_state_values, dim=0)
        if all(isinstance(value, torch.Tensor) for value in segment_state_values)
        else None
    )
    return FrontRESSegmentSample(
        segment_ids=segment_ids,
        source=tuple(str(sample.source[0]) for sample in selected),
        priority=torch.cat([sample.priority for sample in selected], dim=0),
        staleness=torch.cat([sample.staleness for sample in selected], dim=0),
        valid_mask=torch.cat([sample.valid_mask for sample in selected], dim=0),
        segment_state=segment_state,
        rollout_trial_count=torch.cat([sample.rollout_trial_count for sample in selected], dim=0),
        horizon_k=torch.cat([sample.horizon_k for sample in selected], dim=0),
        budget_reason=tuple(str(sample.budget_reason[0]) for sample in selected),
        trial_role=("policy",) * len(selected),
        source_index=torch.arange(len(selected), dtype=torch.long, device=device),
        trial_index=torch.zeros(len(selected), dtype=torch.long, device=device),
    )


def _outer_replay_base_sample(plan: FrontRESOuterReplayPlan, *, device: torch.device | str) -> FrontRESSegmentSample:
    plan.validate()
    segment_ids = torch.tensor(
        [selection.segment_id for selection in plan.selections],
        dtype=torch.long,
        device=device,
    )
    scenario_count = int(segment_ids.numel())
    return FrontRESSegmentSample(
        segment_ids=segment_ids,
        source=tuple(selection.source for selection in plan.selections),
        priority=torch.tensor([selection.score for selection in plan.selections], dtype=torch.float32, device=device),
        staleness=torch.tensor(
            [selection.staleness for selection in plan.selections], dtype=torch.float32, device=device
        ),
        valid_mask=torch.ones(scenario_count, dtype=torch.bool, device=device),
        segment_state=None,
        rollout_trial_count=torch.ones(scenario_count, dtype=torch.long, device=device),
        horizon_k=torch.full((scenario_count,), int(plan.active_k), dtype=torch.long, device=device),
        budget_reason=tuple(f"outer-{selection.source}" for selection in plan.selections),
        trial_role=("policy",) * scenario_count,
        source_index=torch.arange(scenario_count, dtype=torch.long, device=device),
        trial_index=torch.zeros(scenario_count, dtype=torch.long, device=device),
    )


def _outer_replay_scenario_keys(
    batch: Any,
    sample: FrontRESSegmentSample,
    plan: FrontRESOuterReplayPlan,
) -> tuple[FrontRESScenarioKey, ...]:
    rows = getattr(batch, "frontres_local_scenario_rows", None)
    specs = tuple(getattr(batch, "specs", ()) or ())
    if rows is None or len(specs) != int(sample.segment_ids.numel()):
        raise RuntimeError("outer replay key construction requires sealed Scenario rows and Stage-1 specs")
    result: list[FrontRESScenarioKey] = []
    for source, selection in enumerate(plan.selections):
        matches = torch.nonzero(sample.source_index == source, as_tuple=False).flatten()
        if int(matches.numel()) <= 0:
            raise RuntimeError(f"outer replay source {source} has no materialized rows")
        row = int(matches[0].item())
        scenario = rows.scenario_for_row(row)
        spec = specs[row]
        key = FrontRESScenarioKey(
            motion_id=str(getattr(spec, "motion_id", "")),
            start_frame=int(getattr(spec, "start_frame", -1)),
            segment_id=int(selection.segment_id),
            x_t_identity=str(scenario.request.x_t_identity),
            perturbation_family=selection.perturbation_family,
            perturbation_strength=selection.perturbation_strength,
            perturbation_seed=selection.perturbation_seed,
            noisy_segment_hash=scenario.noisy_segment_hash,
            horizon_k=int(scenario.request.horizon_k),
            future_intent_identity=frontres_tensor_identity(scenario.intent_q29),
            planned_support_identity=frontres_tensor_identity(
                scenario.expected_support,
                scenario.expected_support_envelope,
            ),
        )
        key.validate()
        result.append(key)
    if len(result) != 8 or len({key.digest for key in result}) != 8:
        raise RuntimeError("outer replay transaction requires eight distinct stable ScenarioKeys")
    return tuple(result)


@dataclass(frozen=True)
class FrontRESPreparedLocalTransaction:
    """Typed carrier from Scenario selection/materialization to formal collection."""

    sample: FrontRESSegmentSample
    batch: Any
    plan: FrontRESFormalTransactionPlan
    outer_replay_plan: FrontRESOuterReplayPlan
    outer_replay_scenario_keys: tuple[FrontRESScenarioKey, ...]

    def validate(self) -> None:
        if self.batch is None or self.plan is None:
            raise ValueError("prepared FrontRES transaction requires batch and frozen plan")
        self.outer_replay_plan.validate()
        if len(self.outer_replay_scenario_keys) != FRONTRES_V011_SELECTED_SEGMENT_COUNT:
            raise ValueError("prepared FrontRES transaction requires eight Scenario keys")
        for key in self.outer_replay_scenario_keys:
            key.validate()
        if self.outer_replay_plan.transaction_id != self.plan.transaction_id:
            raise ValueError("prepared FrontRES transaction lost its transaction identity")


def _prepare_frontres_v015_local_transaction_batch(
    runner: Any,
    *,
    route: str,
) -> FrontRESPreparedLocalTransaction:
    """Select and seal one complete v015 local transaction before reset.

    Status: active v015 selection owner for the bounded sentinel and ordinary
    formal Stage-3 route. Downstream is one local-scenario batch plus its frozen
    policy/expected-row plan. It selects no legacy fixed tape and does not reset
    the environment, sample an action, update priority, or step an optimizer.
    """

    alg = getattr(runner, "alg", None)
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    outer_replay = frontres_outer_scenario_replay(runner)
    if alg is None or sampler is None:
        raise RuntimeError("v015 local transaction requires initialized algorithm and segment sampler owners")
    if route not in {"sentinel", "training"}:
        raise ValueError(f"unknown v015 local transaction route={route!r}")
    sentinel_only = bool(getattr(alg, "frontres_local_sentinel_only", False))
    live_train_enabled = bool(getattr(alg, "frontres_segment_live_train_enabled", False))
    if route == "sentinel" and not sentinel_only:
        raise RuntimeError("v015 local sentinel batch requires its explicit config flag")
    if route == "training" and (sentinel_only or not live_train_enabled):
        raise RuntimeError("v015 formal training batch requires ordinary live training and rejects sentinel mode")
    env_count = int(getattr(getattr(runner, "env", None), "num_envs", 0) or 0)
    if env_count <= 0:
        raise RuntimeError("FRS-TRAIN-v011 local transaction requires a positive environment count")
    max_horizon = max(1, int(getattr(alg, "frontres_segment_max_horizon_k", 1) or 1))
    iteration = int(getattr(runner, "current_learning_iteration", 0) or 0)
    curriculum = resolve_frontres_k_stage_identity(
        schedule=tuple(getattr(alg, "frontres_segment_k_curriculum", ()) or ()),
        committed_update_iteration=iteration,
        max_horizon_k=max_horizon,
    )
    require_frontres_v013_campaign_schedule(tuple(getattr(alg, "frontres_segment_k_curriculum", ()) or ()))
    expected_repair_rows = FRONTRES_V011_SELECTED_SEGMENT_COUNT * int(curriculum.active_m)
    required_env_count = 2 * expected_repair_rows
    if env_count != required_env_count:
        raise RuntimeError(
            "FRS-TRAIN-v024 environment width must equal 2*B8*M4 Repair/Noisy rows: "
            f"active_m={curriculum.active_m} required={required_env_count} observed={env_count}"
        )
    repair_rows = env_count // 2
    configured_fingerprint = str(getattr(alg, "frontres_segment_k_curriculum_fingerprint", "") or "")
    if configured_fingerprint and configured_fingerprint != curriculum.schedule_fingerprint:
        raise RuntimeError("FRS-TRAIN-v011 sampler curriculum fingerprint drifted after config resolution")
    sequence_attr = f"_frontres_v015_local_{route}_sequence"
    sequence = int(getattr(runner, sequence_attr, 0) or 0) + 1
    setattr(runner, sequence_attr, sequence)
    transaction_id = f"frontres-v015-local-{route}:i{iteration}:n{sequence}"
    future_offsets = _require_frontres_future_offsets(runner)
    candidate_is_eligible = _local_scenario_candidate_eligibility(
        runner,
        horizon_k=curriculum.active_k,
        intent_horizon=max(future_offsets),
    )
    outer_replay_plan = outer_replay.plan(
        transaction_id=transaction_id,
        curriculum=curriculum,
        num_segments=int(getattr(sampler, "num_segments", 0) or 0),
        eligible=candidate_is_eligible,
        global_family=lambda _segment_id: "local_rp",
    )
    base_sample = _outer_replay_base_sample(
        outer_replay_plan,
        device=getattr(runner, "device", sampler.device),
    )
    base_ids = base_sample.segment_ids.detach().to(dtype=torch.long).clone()
    if int(base_ids.numel()) != FRONTRES_V011_SELECTED_SEGMENT_COUNT or int(
        torch.unique(base_ids).numel()
    ) != FRONTRES_V011_SELECTED_SEGMENT_COUNT:
        raise RuntimeError("FRS-TRAIN-v024 local transaction requires exactly eight distinct Scenario sources")
    snapshot = capture_frontres_frozen_policy_snapshot(runner, transaction_id=transaction_id)
    frozen_plan = sampler.plan_frozen_policy_transaction(
        base_ids,
        transaction_id=transaction_id,
        policy_snapshot_id=snapshot.policy_snapshot_id,
        max_horizon_k=max_horizon,
        minimum_policy_attempts=2,
        exact_policy_attempts=curriculum.active_m,
        active_horizon_k=curriculum.active_k,
    )
    if not bool((frozen_plan.horizon_k == curriculum.active_k).all().item()):
        raise RuntimeError("FRS-TRAIN-v011 sampler failed to produce a homogeneous-K transaction")
    if not bool((frozen_plan.base_trial_count == curriculum.active_m).all().item()):
        raise RuntimeError("FRS-TRAIN-v011 sampler failed to produce exact-M attempts per Segment")
    if int(frozen_plan.segment_ids.numel()) != repair_rows:
        raise RuntimeError(
            "v015 local transaction requires environment Repair rows to equal its complete selected transaction: "
            f"repair_rows={repair_rows} planned_attempts={int(frozen_plan.segment_ids.numel())}"
        )
    source_index = frozen_plan.source_index.detach().to(device=base_ids.device, dtype=torch.long)
    expanded_sample = FrontRESSegmentSample(
        segment_ids=frozen_plan.segment_ids.detach().clone(),
        source=tuple(str(base_sample.source[int(index)]) for index in source_index.tolist()),
        priority=base_sample.priority.index_select(0, source_index).detach().clone(),
        staleness=base_sample.staleness.index_select(0, source_index).detach().clone(),
        valid_mask=base_sample.valid_mask.index_select(0, source_index).detach().clone(),
        segment_state=(
            base_sample.segment_state.index_select(0, source_index).detach().clone()
            if isinstance(base_sample.segment_state, torch.Tensor)
            else None
        ),
        rollout_trial_count=frozen_plan.base_trial_count.index_select(0, source_index).detach().clone(),
        horizon_k=frozen_plan.horizon_k.detach().clone(),
        budget_reason=tuple(str(base_sample.budget_reason[int(index)]) for index in source_index.tolist()),
        trial_role=tuple(frozen_plan.trial_role),
        source_index=source_index.detach().clone(),
        trial_index=frozen_plan.trial_index.detach().clone(),
    )
    batch = _build_current_segment_batch(
        runner,
        expanded_sample,
        update_step=sequence,
        print_probe=True,
        v015_local_scenario_transaction_id=transaction_id,
        outer_replay_plan=outer_replay_plan,
    )
    if batch is None or getattr(batch, "frontres_local_scenario_rows", None) is None:
        raise RuntimeError("v015 local transaction failed to materialize a sealed local scenario batch")
    scenario_ids = tuple(str(value) for value in getattr(batch, "frontres_local_scenario_ids", ()) or ())
    hashes = tuple(str(value) for value in getattr(batch, "frontres_local_scenario_hashes", ()) or ())
    x_t_identities = tuple(str(value) for value in getattr(batch, "frontres_local_scenario_x_t_identities", ()) or ())
    provenance = tuple(getattr(batch, "frontres_local_scenario_provenance", ()) or ())
    specs = tuple(getattr(batch, "specs", ()) or ())
    row_count = int(expanded_sample.segment_ids.numel())
    if len(specs) != row_count or len(scenario_ids) != row_count or len(hashes) != row_count or len(x_t_identities) != row_count:
        raise RuntimeError("v015 local transaction batch lost source-aligned local scenario identities")
    if not provenance or any(not isinstance(value, Mapping) for value in provenance):
        raise RuntimeError("v015 local transaction batch lost local scenario provenance")
    intent_provenance = {str(value.get("intent_q29_provenance", "")) for value in provenance}
    intent_source = {str(value.get("intent_q29_source", "")) for value in provenance}
    if len(intent_provenance) != 1 or len(intent_source) != 1:
        raise RuntimeError("v015 local transaction requires one q29 provenance/source semantic owner per transaction")
    scenario_keys = _outer_replay_scenario_keys(batch, expanded_sample, outer_replay_plan)
    plan = FrontRESFormalTransactionPlan(
        snapshot=snapshot,
        motion_ids=tuple(str(getattr(spec, "motion_id", "")) for spec in specs),
        start_frames=torch.tensor(
            [int(getattr(spec, "start_frame", -1)) for spec in specs],
            dtype=torch.long,
            device=expanded_sample.segment_ids.device,
        ),
        segment_ids=expanded_sample.segment_ids,
        source_index=expanded_sample.source_index,
        trial_index=expanded_sample.trial_index,
        horizon_k=expanded_sample.horizon_k,
        scenario_ids=scenario_ids,
        noisy_segment_hashes=hashes,
        x_t_identities=x_t_identities,
        intent_q29_provenance=next(iter(intent_provenance)),
        intent_q29_source=next(iter(intent_source)),
    )
    plan.validate()
    result = FrontRESPreparedLocalTransaction(
        sample=expanded_sample,
        batch=batch,
        plan=plan,
        outer_replay_plan=outer_replay_plan,
        outer_replay_scenario_keys=scenario_keys,
    )
    result.validate()
    return result


def prepare_frontres_v015_local_sentinel_batch(runner: Any) -> FrontRESPreparedLocalTransaction:
    """Prepare the explicit bounded sentinel transaction."""

    return _prepare_frontres_v015_local_transaction_batch(runner, route="sentinel")


def prepare_frontres_v015_formal_training_batch(runner: Any) -> FrontRESPreparedLocalTransaction:
    """Prepare one complete ordinary Stage-3 transaction without legacy rows."""

    return _prepare_frontres_v015_local_transaction_batch(runner, route="training")


def prepare_frontres_v015_policy_quality_item_batch(runner: Any, item: Any) -> SimpleNamespace:
    """Materialize one fixed manifest item as an immutable two-role local batch.

    The manifest selects an existing Stage-1 index row by motion/frame identity.
    Its effective K is the executable-evidence budget, not the cache index
    window used to identify x_t. All Repair attempts use one source identity so
    zero/HSL/policy resets can reuse the same sealed scenario without invoking
    the training sampler or curriculum.
    """

    dataset = getattr(runner, "_frontres_segment_dataset", None)
    specs = tuple(getattr(dataset, "_specs", ()) or ())
    if dataset is None or not callable(getattr(dataset, "get_segments", None)) or not specs:
        raise RuntimeError("v015 quality manifest requires the initialized Stage-1 index dataset")
    motion_id = str(getattr(item, "motion_id", "")).lstrip("./")
    start_frame = int(getattr(item, "start_frame", -1))
    horizon_k = int(getattr(item, "effective_horizon_k", 0))
    matches = tuple(
        spec
        for spec in specs
        if str(getattr(spec, "motion_id", "")).lstrip("./") == motion_id
        and int(getattr(spec, "start_frame", -1)) == start_frame
    )
    if len(matches) != 1:
        cache_horizons = tuple(sorted({int(getattr(spec, "horizon_k", -1)) for spec in matches}))
        raise RuntimeError(
            "v015 quality manifest must resolve motion/start to exactly one loaded Segment identity: "
            f"motion={motion_id!r} frame={start_frame} execution_K={horizon_k} "
            f"matches={len(matches)} cache_horizons={cache_horizons}"
        )
    params = dict(getattr(item, "perturbation_parameters", ()) or ())
    strength_values = [params[name] for name in ("strength", "dr_scale", "scale") if name in params]
    if len(strength_values) != 1:
        raise ValueError("v015 quality manifest requires exactly one strength/dr_scale/scale parameter")
    strength = float(strength_values[0])
    family = str(getattr(item, "perturbation_family", ""))
    if not family or not math.isfinite(strength) or strength < 0.0:
        raise ValueError("v015 quality manifest has invalid perturbation family or strength")
    env_count = int(getattr(getattr(runner, "env", None), "num_envs", 0) or 0)
    if env_count != 8:
        raise RuntimeError("v015 bounded held-out quality requires exactly 8 envs (4 Repair + 4 Noisy)")
    repair_rows = env_count // 2
    device = torch.device(getattr(runner, "device", "cpu"))
    segment_id = int(matches[0].segment_id)
    segment_ids = torch.full((repair_rows,), segment_id, dtype=torch.long, device=device)
    source_index = torch.zeros(repair_rows, dtype=torch.long, device=device)
    sample = FrontRESSegmentSample(
        segment_ids=segment_ids,
        source=("heldout",) * repair_rows,
        priority=torch.ones(repair_rows, dtype=torch.float32, device=device),
        staleness=torch.zeros(repair_rows, dtype=torch.float32, device=device),
        valid_mask=torch.ones(repair_rows, dtype=torch.bool, device=device),
        segment_state=None,
        rollout_trial_count=torch.zeros(repair_rows, dtype=torch.long, device=device),
        horizon_k=torch.full((repair_rows,), horizon_k, dtype=torch.long, device=device),
        budget_reason=("heldout_manifest",) * repair_rows,
        trial_role=("policy",) * repair_rows,
        source_index=source_index,
        trial_index=torch.arange(repair_rows, dtype=torch.long, device=device),
    )
    batch = dataset.get_segments(segment_ids)
    _attach_frontres_segment_trial_plan(batch, sample)
    fixed_plan = SimpleNamespace(
        perturbation_family=(family,) * repair_rows,
        perturbation_strength=torch.full(
            (repair_rows,), strength, dtype=batch.perturbation_strength.dtype, device=device
        ),
        source_index=source_index,
        source_ids=torch.zeros(1, dtype=torch.long, device=device),
        source_perturbation_family=(family,),
        source_perturbation_strength=torch.tensor([strength], dtype=torch.float32, device=device),
        active_modes=(family,),
        complexity="heldout_fixed",
        mix_mode="heldout_fixed",
        mix_diag={"seed": int(getattr(item, "seed", -1))},
        progress=1.0,
        seq_idx=int(getattr(item, "seed", -1)),
    )
    batch = _attach_stage3_index_perturbation_plan(batch, fixed_plan)

    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        torch.manual_seed(int(getattr(item, "seed", -1)))
        batch = _attach_frontres_local_scenarios(
            runner,
            batch,
            sample,
            update_step=0,
            transaction_id=f"frontres-v015-quality:{item.comparison_signature}",
        )
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
    scenario_ids = tuple(getattr(batch, "frontres_local_scenario_ids", ()) or ())
    hashes = tuple(getattr(batch, "frontres_local_scenario_hashes", ()) or ())
    x_t = tuple(getattr(batch, "frontres_local_scenario_x_t_identities", ()) or ())
    if (
        len(scenario_ids) != repair_rows
        or len(set(scenario_ids)) != 1
        or len(set(hashes)) != 1
        or len(set(x_t)) != 1
    ):
        raise RuntimeError("v015 held-out manifest failed to seal one shared scenario/hash/x_t identity")
    return SimpleNamespace(sample=sample, batch=batch)


def prepare_frontres_fixed_k_m4_evaluation_batch(
    runner: Any,
    items: tuple[Any, Any],
    *,
    attempts_per_segment: int,
    allowed_horizons: tuple[int, ...],
    transaction_namespace: str,
    route_label: str,
) -> SimpleNamespace:
    """Materialize two fixed held-out Segments as one immutable exact-M4 batch."""

    # B1: 解析两个 manifest Segment, 产出 distinct source identities 和 exact-M row plan.
    if not isinstance(items, tuple) or len(items) != 2 or attempts_per_segment != 4:
        raise ValueError(f"{route_label} requires exactly two held-out Segments and M=4")
    if not allowed_horizons or any(int(value) <= 0 for value in allowed_horizons):
        raise ValueError(f"{route_label} requires explicit positive allowed horizons")
    if not transaction_namespace:
        raise ValueError(f"{route_label} requires a transaction namespace")
    dataset = getattr(runner, "_frontres_segment_dataset", None)
    resolve_spec = getattr(dataset, "resolve_segment_spec", None)
    if dataset is None or not callable(getattr(dataset, "get_segments", None)) or not callable(resolve_spec):
        raise RuntimeError(f"{route_label} requires the initialized Stage-1 index dataset")
    resolved = []
    families: list[str] = []
    strengths: list[float] = []
    requested_horizon: int | None = None
    for item in items:
        motion_id = str(getattr(item, "motion_id", "")).lstrip("./")
        start_frame = int(getattr(item, "start_frame", -1))
        horizon_k = int(getattr(item, "effective_horizon_k", 0))
        try:
            spec = resolve_spec(motion_id=motion_id, start_frame=start_frame)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{route_label} failed to resolve one Segment: "
                f"motion={motion_id!r} frame={start_frame}"
            ) from exc
        if horizon_k not in allowed_horizons:
            raise RuntimeError(
                f"{route_label} resolved a disallowed K for motion/start: "
                f"motion={motion_id!r} frame={start_frame} K={horizon_k}"
            )
        if requested_horizon is None:
            requested_horizon = horizon_k
        elif horizon_k != requested_horizon:
            raise RuntimeError(f"{route_label} requires one homogeneous horizon")
        params = dict(getattr(item, "perturbation_parameters", ()) or ())
        strength_values = [params[name] for name in ("strength", "dr_scale", "scale") if name in params]
        family = str(getattr(item, "perturbation_family", ""))
        if len(strength_values) != 1 or not family:
            raise ValueError(f"{route_label} requires one perturbation family and one strength")
        strength = float(strength_values[0])
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError(f"{route_label} perturbation strength must be finite and non-negative")
        resolved.append(spec)
        families.append(family)
        strengths.append(strength)
    if int(resolved[0].segment_id) == int(resolved[1].segment_id):
        raise ValueError(f"{route_label} requires two distinct Segment identities")
    env_count = int(getattr(getattr(runner, "env", None), "num_envs", 0) or 0)
    repair_rows = 2 * attempts_per_segment
    if env_count != 2 * repair_rows:
        raise RuntimeError(
            f"{route_label} requires 16 env rows (8 Repair + 8 Noisy): "
            f"observed={env_count}"
        )
    if requested_horizon is None:
        raise RuntimeError(f"{route_label} did not resolve a fixed horizon")
    device = torch.device(getattr(runner, "device", "cpu"))
    source_index = torch.arange(2, dtype=torch.long, device=device).repeat_interleave(attempts_per_segment)
    trial_index = torch.arange(attempts_per_segment, dtype=torch.long, device=device).repeat(2)
    segment_ids = torch.tensor(
        [int(resolved[int(source)].segment_id) for source in source_index.tolist()],
        dtype=torch.long,
        device=device,
    )
    horizon_k = torch.full((repair_rows,), requested_horizon, dtype=torch.long, device=device)
    sample = FrontRESSegmentSample(
        segment_ids=segment_ids,
        source=tuple("heldout" for _ in range(repair_rows)),
        priority=torch.ones(repair_rows, dtype=torch.float32, device=device),
        staleness=torch.zeros(repair_rows, dtype=torch.float32, device=device),
        valid_mask=torch.ones(repair_rows, dtype=torch.bool, device=device),
        segment_state=None,
        rollout_trial_count=torch.full((repair_rows,), attempts_per_segment, dtype=torch.long, device=device),
        horizon_k=horizon_k,
        budget_reason=tuple("heldout_manifest" for _ in range(repair_rows)),
        trial_role=tuple("policy" for _ in range(repair_rows)),
        source_index=source_index,
        trial_index=trial_index,
    )

    # B2: 安装固定 perturbation plan 并只 materialize 一次, 产出 source-shared sealed scenarios.
    batch = dataset.get_segments(segment_ids)
    _attach_frontres_segment_trial_plan(batch, sample)
    row_strength = torch.tensor(
        [strengths[int(source)] for source in source_index.tolist()],
        dtype=batch.perturbation_strength.dtype,
        device=device,
    )
    fixed_plan = SimpleNamespace(
        perturbation_family=tuple(families[int(source)] for source in source_index.tolist()),
        perturbation_strength=row_strength,
        source_index=source_index,
        source_ids=torch.arange(2, dtype=torch.long, device=device),
        source_perturbation_family=tuple(families),
        source_perturbation_strength=torch.tensor(strengths, dtype=torch.float32, device=device),
        active_modes=tuple(dict.fromkeys(families)),
        complexity="heldout_fixed",
        mix_mode="heldout_fixed",
        mix_diag={"seeds": tuple(int(getattr(item, "seed", -1)) for item in items)},
        progress=1.0,
        seq_idx=0,
    )
    batch = _attach_stage3_index_perturbation_plan(batch, fixed_plan)
    transaction_signature = hashlib.sha256(
        "|".join(str(getattr(item, "comparison_signature", "")) for item in items).encode("ascii")
    ).hexdigest()
    transaction_id = f"{transaction_namespace}:{transaction_signature}"
    try:
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        try:
            torch.manual_seed(int(transaction_signature[:16], 16) % (2**63 - 1))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(transaction_signature[:16], 16) % (2**63 - 1))
            batch = _attach_frontres_local_scenarios(
                runner,
                batch,
                sample,
                update_step=0,
                transaction_id=transaction_id,
            )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
        scenario_ids = tuple(str(value) for value in getattr(batch, "frontres_local_scenario_ids", ()) or ())
        hashes = tuple(str(value) for value in getattr(batch, "frontres_local_scenario_hashes", ()) or ())
        x_t = tuple(str(value) for value in getattr(batch, "frontres_local_scenario_x_t_identities", ()) or ())
        provenance = tuple(getattr(batch, "frontres_local_scenario_provenance", ()) or ())
        if any(len(values) != repair_rows for values in (scenario_ids, hashes, x_t, provenance)):
            raise RuntimeError(f"{route_label} materializer lost row-aligned scenario identity")
        for source in range(2):
            rows = [row for row, value in enumerate(source_index.tolist()) if int(value) == source]
            if any(len({values[row] for row in rows}) != 1 for values in (scenario_ids, hashes, x_t)):
                raise RuntimeError(f"{route_label} reset rows resampled or mixed one Segment scenario")
        intent_provenance = {str(value.get("intent_q29_provenance", "")) for value in provenance}
        intent_source = {str(value.get("intent_q29_source", "")) for value in provenance}
        if len(intent_provenance) != 1 or len(intent_source) != 1:
            raise RuntimeError(f"{route_label} requires one deployment q29 provenance owner")

        # B3: Seal evaluation rows without borrowing the B8 formal-training lifecycle owner.
        snapshot = capture_frontres_frozen_policy_snapshot(runner, transaction_id=transaction_id)
        batch_specs = tuple(getattr(batch, "specs", ()) or ())
        plan = FrontRESV015GroupedCandidateMetadata(
            transaction_id=snapshot.transaction_id,
            policy_snapshot_id=snapshot.policy_snapshot_id,
            motion_ids=tuple(str(getattr(spec, "motion_id", "")) for spec in batch_specs),
            start_frames=torch.tensor(
                [int(getattr(spec, "start_frame", -1)) for spec in batch_specs],
                dtype=torch.long,
                device=device,
            ),
            segment_ids=segment_ids,
            source_index=source_index,
            trial_index=trial_index,
            horizon_k=horizon_k,
            evidence_valid_step_count=horizon_k,
            trial_role=("policy",) * repair_rows,
            scenario_ids=scenario_ids,
            noisy_segment_hashes=hashes,
            x_t_identities=x_t,
            intent_q29_provenance=next(iter(intent_provenance)),
            intent_q29_source=next(iter(intent_source)),
        )
        return SimpleNamespace(sample=sample, batch=batch, plan=plan)
    except BaseException:
        close_frontres_local_scenarios(batch)
        raise


def prepare_frontres_policy_quality_fixed_k_m4_batch(
    runner: Any,
    items: tuple[Any, Any],
    *,
    attempts_per_segment: int,
) -> SimpleNamespace:
    """Historical EVAL-v004 wrapper over the version-neutral fixed-M4 owner."""

    return prepare_frontres_fixed_k_m4_evaluation_batch(
        runner,
        items,
        attempts_per_segment=attempts_per_segment,
        allowed_horizons=(8, 16),
        transaction_namespace="frontres-v018-quality",
        route_label="EVAL-v004 held-out policy quality",
    )


def prepare_frontres_action_gain_direction_fixed_k_m4_batch(
    runner: Any,
    items: tuple[Any, Any],
    *,
    attempts_per_segment: int,
) -> SimpleNamespace:
    """Active EVAL-v006 bounded-diagnostic K8/M4 materializer wrapper."""

    return prepare_frontres_fixed_k_m4_evaluation_batch(
        runner,
        items,
        attempts_per_segment=attempts_per_segment,
        allowed_horizons=(8,),
        transaction_namespace="frontres-v024-action-gain-direction",
        route_label="EVAL-v006 bounded action-Gain direction diagnostic",
    )

_print_evidence_probe = print_frontres_sampler_evidence_probe
_print_sample_probe = print_frontres_sampler_sample_probe
_print_sampler_summary = print_frontres_sampler_summary
_sample_live_segment_rows = sample_frontres_live_segment_rows
_live_detail_log_enabled = frontres_sampler_detail_log_enabled
_verbose_batch_lines = frontres_sampler_verbose_batch_lines
_verbose_probe_enabled = frontres_sampler_verbose_probe_enabled
_count_summary = summarize_frontres_sampler_counts
_id_summary = summarize_frontres_sampler_ids
_tensor_value_summary = summarize_frontres_sampler_tensor
_resolve_live_max_horizon_k = resolve_frontres_live_max_horizon_k
_resolve_live_scorable_row_budget = resolve_frontres_live_scorable_row_budget
