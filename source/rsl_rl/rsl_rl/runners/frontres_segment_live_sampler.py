from __future__ import annotations

from collections import Counter
import importlib.util
import math
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

_DATASET_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_segment_dataset.py"
_DATASET_SPEC = importlib.util.spec_from_file_location(
    "frontres_segment_dataset_live_module",
    _DATASET_PATH,
)
if _DATASET_SPEC is None or _DATASET_SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES Segment dataset from {_DATASET_PATH}.")
_DATASET_MODULE = importlib.util.module_from_spec(_DATASET_SPEC)
sys.modules[_DATASET_SPEC.name] = _DATASET_MODULE
_DATASET_SPEC.loader.exec_module(_DATASET_MODULE)

FrontRESSegmentRolloutEvidence = _SAMPLER_MODULE.FrontRESSegmentRolloutEvidence
FrontRESSegmentSample = _SAMPLER_MODULE.FrontRESSegmentSample
FrontRESSegmentSampler = _SAMPLER_MODULE.FrontRESSegmentSampler
load_stage1_cache_dataset = _DATASET_MODULE.load_stage1_cache_dataset

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


def initialize_frontres_segment_live_sampler(runner: Any) -> None:
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    if not bool(getattr(boundary, "requested", False) and getattr(boundary, "live_runner_enabled", False)):
        return
    if getattr(runner, "_frontres_segment_sampler", None) is not None:
        return
    _ensure_stage1_cache_dataset(runner)
    _ensure_stage1_index_reset_hook(runner)
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


def _ensure_stage1_index_reset_hook(runner: Any) -> None:
    dataset = getattr(runner, "_frontres_segment_dataset", None)
    metadata = dataset.cache_metadata() if dataset is not None and hasattr(dataset, "cache_metadata") else None
    if not isinstance(metadata, dict) or not bool(metadata.get("index_only", False)):
        return
    amass_root = str(metadata.get("amass_root", "") or "")
    if not amass_root:
        raise ValueError("index-only Stage 1 dataset metadata is missing amass_root")
    from rsl_rl.frontres.frontres_segment_stage1_env_hooks import ensure_frontres_segment_index_reset_hook

    adapter = ensure_frontres_segment_index_reset_hook(
        runner.env,
        amass_root=amass_root,
        robot_name=str(getattr(runner.alg, "frontres_segment_reset_robot_name", "robot")),
        trace=bool(getattr(runner.alg, "frontres_segment_reset_trace", True)),
    )
    probe = adapter.frontres_motion_loader_probe()
    filter_probe = None
    if hasattr(dataset, "filter_to_loaded_motion_paths"):
        filter_probe = dataset.filter_to_loaded_motion_paths(
            adapter.frontres_loaded_motion_paths(),
            amass_root=amass_root,
        )
    print(
        _log_block(
            "[FrontRES Segment Index Reset Hook Ready]",
            "  loader: "
            f"amass_root={amass_root} "
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

    sample = sampler.sample(_resolve_live_batch_size(runner))
    detail_log = _live_detail_log_enabled(runner)
    verbose_probe = _verbose_probe_enabled(runner, sample)
    if detail_log:
        _print_sample_probe(update_step, sample, verbose=verbose_probe)
    batch = _build_current_segment_batch(runner, sample, update_step=update_step, print_probe=detail_log)
    runner._frontres_segment_live_current_sample = sample
    runner._frontres_segment_live_current_batch = batch
    runner._frontres_segment_live_detail_log_enabled = detail_log
    reset_result = None
    try:
        summary = runner.run_frontres_segment_live_probe(init_at_random_ep_len=init_at_random_ep_len)
        reset_result = getattr(runner, "_frontres_segment_live_current_reset_result", None)
    finally:
        runner._frontres_segment_live_current_sample = None
        runner._frontres_segment_live_current_batch = None
        runner._frontres_segment_live_current_reset_request = None
        runner._frontres_segment_live_current_reset_result = None
        runner._frontres_segment_live_detail_log_enabled = True

    evidence = build_live_sampler_evidence(
        sample,
        summary,
        horizon_k=int(getattr(runner.alg, "frontres_segment_k", 1)),
        reset_result=reset_result,
        print_probe=detail_log,
    )
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
        }
    )
    summary.update(sampler_summary)
    if detail_log:
        _print_sampler_summary(update_step, sampler_summary)
    return summary


def _build_current_segment_batch(
    runner: Any,
    sample: FrontRESSegmentSample,
    *,
    update_step: int,
    print_probe: bool = True,
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
    validation = dataset.validate_batch(batch) if hasattr(dataset, "validate_batch") else None
    valid_count = (
        int(validation.valid_mask.bool().sum().detach().cpu().item())
        if validation is not None and hasattr(validation, "valid_mask")
        else int(sample.segment_ids.numel())
    )
    roles = tuple(getattr(batch, "perturbation_role", ()))
    strength = getattr(batch, "perturbation_strength", None)
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
                        "strength": _tensor_value_summary("strength", strength),
                    },
                ),
                *_verbose_batch_lines(sample, roles=roles, strength=strength, verbose=verbose_probe),
            ),
            flush=True,
        )
    return batch


def build_live_sampler_evidence(
    sample: FrontRESSegmentSample,
    summary: dict[str, object],
    *,
    horizon_k: int,
    reset_result: Any | None = None,
    print_probe: bool = True,
) -> FrontRESSegmentRolloutEvidence:
    ids = sample.segment_ids.detach().clone().long()
    n = int(ids.numel())
    device = ids.device
    reset_success = _reset_success_for_sample(reset_result, n=n, device=device)
    reward = _summary_vector(
        summary,
        keys=("storage_reward_per_sample", "reward_per_sample"),
        n=n,
        device=device,
        default=_summary_float(summary, "storage_reward_mean", _summary_float(summary, "reward_mean", 0.0)),
    ).float()
    rollout_valid = _summary_bool_vector(
        summary,
        keys=("storage_valid_mask_per_sample",),
        n=n,
        device=device,
        default=bool(_summary_int(summary, "ppo_valid_count", 0) > 0 and _summary_float(summary, "storage_valid_frac", 0.0) > 0.0),
    )
    fall = _summary_bool_vector(
        summary,
        keys=("done_any_per_sample",),
        n=n,
        device=device,
        default=bool(_summary_float(summary, "done_frac", 0.0) >= 0.5),
    )
    gain = reward.clamp(-1.0, 1.0)
    score_repaired = (0.5 + 0.5 * gain).clamp(0.0, 1.0)
    score_noisy = (score_repaired - gain).clamp(0.0, 1.0)
    valid_reward = rollout_valid & reset_success
    if print_probe:
        _print_evidence_probe(ids, reward, reset_success, rollout_valid, valid_reward, fall, gain)
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
        horizon_k=torch.full((n,), max(1, int(horizon_k)), dtype=torch.long, device=device),
    )


def _reset_success_for_sample(reset_result: Any | None, *, n: int, device: torch.device) -> torch.Tensor:
    if reset_result is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    success = getattr(reset_result, "success_mask", None)
    if success is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    success = success.to(device=device).bool().reshape(-1)
    if int(success.numel()) != n:
        raise ValueError(f"reset_success must have {n} rows, got {int(success.numel())}")
    return success.detach()


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


def _print_evidence_probe(
    ids: torch.Tensor,
    reward: torch.Tensor,
    reset_success: torch.Tensor,
    rollout_valid: torch.Tensor,
    valid_reward: torch.Tensor,
    fall: torch.Tensor,
    gain: torch.Tensor,
) -> None:
    print(
        "[probe step14] evidence_path: "
        f"count={int(ids.numel())} "
        f"id_min={int(ids.min().detach().cpu().item()) if ids.numel() else -1} "
        f"id_max={int(ids.max().detach().cpu().item()) if ids.numel() else -1} "
        f"reward_min={float(reward.min().detach().cpu().item()) if reward.numel() else 0.0:.6f} "
        f"reward_max={float(reward.max().detach().cpu().item()) if reward.numel() else 0.0:.6f} "
        f"reset_valid={int(reset_success.bool().sum().detach().cpu().item())} "
        f"rollout_valid={int(rollout_valid.bool().sum().detach().cpu().item())} "
        f"valid_reward={int(valid_reward.bool().sum().detach().cpu().item())} "
        f"fall_count={int(fall.bool().sum().detach().cpu().item())} "
        f"gain_mean={float(gain.mean().detach().cpu().item()) if gain.numel() else 0.0:.6f}",
        flush=True,
    )


def _verbose_probe_enabled(runner: Any, sample: FrontRESSegmentSample | None = None) -> bool:
    alg = getattr(runner, "alg", None)
    if bool(getattr(alg, "frontres_segment_verbose_probe", False)):
        return True
    if sample is None:
        return False
    return int(sample.segment_ids.numel()) <= _VERBOSE_PROBE_BATCH_LIMIT


def _live_detail_log_enabled(runner: Any) -> bool:
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
    return (
        f"  sample.segment_ids: {sample.segment_ids.detach().cpu().tolist()}",
        f"  sample.sources: {list(sample.source)}",
    )


def _verbose_batch_lines(
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
        f"  batch.strength: {strength_list}",
    )


def _print_sample_probe(update_step: int, sample: FrontRESSegmentSample, *, verbose: bool = False) -> None:
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
                    },
                ),
                *_verbose_sample_lines(sample, verbose=verbose),
            ),
        flush=True,
    )


def _print_sampler_summary(update_step: int, summary: dict[str, object]) -> None:
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
                        "update": (
                            f"valid:{int(summary.get('sampler_update_valid_count', 0))},"
                            f"fall:{int(summary.get('sampler_update_fall_count', 0))},"
                            f"hopeless:{int(summary.get('sampler_update_hopeless_count', 0))},"
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
