from __future__ import annotations

import json
import os
import math
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

_FORMAL_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_audit_training", Path(__file__).resolve().with_name("frontres_formal_runtime_audit.py")
)
_FORMAL_AUDIT_MODULE = importlib.util.module_from_spec(_FORMAL_AUDIT_SPEC)
assert _FORMAL_AUDIT_SPEC.loader is not None
_FORMAL_AUDIT_SPEC.loader.exec_module(_FORMAL_AUDIT_MODULE)
print_formal_route_audit = _FORMAL_AUDIT_MODULE.print_formal_route_audit

_DIAGNOSTICS_SPEC = importlib.util.spec_from_file_location(
    "frontres_segment_diagnostics",
    Path(__file__).resolve().parents[1] / "frontres" / "frontres_segment_diagnostics.py",
)
_DIAGNOSTICS_MODULE = importlib.util.module_from_spec(_DIAGNOSTICS_SPEC)
sys.modules[_DIAGNOSTICS_SPEC.name] = _DIAGNOSTICS_MODULE
_DIAGNOSTICS_SPEC.loader.exec_module(_DIAGNOSTICS_MODULE)
format_segment_motion_quality_log = _DIAGNOSTICS_MODULE.format_segment_motion_quality_log
format_segment_periodic_eval_log = _DIAGNOSTICS_MODULE.format_segment_periodic_eval_log
format_segment_train_effect_log = _DIAGNOSTICS_MODULE.format_segment_train_effect_log
action_distribution_health_summary = _DIAGNOSTICS_MODULE.action_distribution_health_summary
motion_quality_summary_to_scalars = _DIAGNOSTICS_MODULE.motion_quality_summary_to_scalars

try:
    from rsl_rl.runners.frontres_segment_live_probe import (
        _apply_current_segment_reset,
        _capture_paired_gain,
        _read_live_observations,
        _run_live_rollout_capture,
    )
except ModuleNotFoundError:
    def _apply_current_segment_reset(runner: Any) -> None:
        raise NotImplementedError("frontres_segment_live_probe import is unavailable.")

    def _read_live_observations(runner: Any) -> None:
        raise NotImplementedError("frontres_segment_live_probe import is unavailable.")

    def _capture_paired_gain(capture: Any) -> Any | None:
        raise NotImplementedError("frontres_segment_live_probe import is unavailable.")

    def _run_live_rollout_capture(
        runner: Any,
        observations: Any,
        *,
        rollout_steps: int,
        capture_motion_quality: bool = True,
        zero_segment_action: bool = False,
    ) -> None:
        raise NotImplementedError("frontres_segment_live_probe import is unavailable.")


_REQUIRED_SUMMARY_KEYS = (
    "update_steps",
    "update_count",
    "ppo_valid_count",
    "reward_mean",
    "storage_valid_frac",
    "ppo_total_loss_mean",
    "ppo_actor_loss_mean",
    "ppo_value_loss_mean",
    "ppo_approx_kl_mean",
    "ppo_clip_frac_mean",
)

_FINITE_SUMMARY_KEYS = (
    "reward_mean",
    "storage_valid_frac",
    "ppo_total_loss_mean",
    "ppo_actor_loss_mean",
    "ppo_value_loss_mean",
    "ppo_approx_kl_mean",
    "ppo_clip_frac_mean",
)

_LOG_SEPARATOR = "-" * 80
_V015_PROJECTION_TOLERANCE = 1.0e-8


_EVAL_GAIN_COMPONENTS = (
    "style_gain",
    "physics_gain",
    "repair_cost",
    "gain_total",
    "style_mpjpe_gain",
    "style_velocity_gain",
    "style_acceleration_gain",
    "style_root_orientation_gain",
    "physics_success_gain",
    "physics_survival_quality_repaired",
    "physics_survival_quality_noisy",
    "physics_survival_gain",
    "physics_zmp_gain",
    "physics_contact_gain",
    "repair_norm",
    "repair_temporal_change",
    "repair_clean_cost",
)

_EVAL_GAIN_PUBLIC_ALIASES = {
    "style_gain": "style",
    "physics_gain": "physics",
    "repair_cost": "repair_cost",
    "gain_total": "total",
    "style_mpjpe_gain": "style_mpjpe",
    "style_velocity_gain": "style_velocity",
    "style_acceleration_gain": "style_acceleration",
    "style_root_orientation_gain": "style_root_orientation",
    "physics_success_gain": "physics_success",
    "physics_survival_quality_repaired": "physics_survival_quality_repaired",
    "physics_survival_quality_noisy": "physics_survival_quality_noisy",
    "physics_survival_gain": "physics_survival",
    "physics_zmp_gain": "physics_zmp",
    "physics_contact_gain": "physics_contact",
    "repair_norm": "repair_norm",
    "repair_temporal_change": "repair_temporal",
    "repair_clean_cost": "repair_clean_cost",
}


def _add_public_gain_component_aliases(summary: dict[str, Any]) -> None:
    """Expose one stable log field for every canonical Gain component."""

    # B2: 将 GainResult 的内部字段名映射为评估日志公开字段名.
    # 该映射必须同时服务 sequence, periodic 和 per-motion summary.
    for component, public_name in _EVAL_GAIN_PUBLIC_ALIASES.items():
        summary[f"gain_{public_name}_mean"] = summary.get(
            f"gain_{component}_mean",
            float("nan"),
        )


def _reject_v015_legacy_evaluator(runner: Any, *, evaluator: str) -> None:
    """Reject a v015 layout before any legacy evaluator can sample or capture v002 Gain.

    函数名说明:
        `_reject_v015_legacy_evaluator` 是 legacy-evaluation isolation gate,
        不是 v015 evaluator owner. 它只阻止 quartet/repeated-actor/v002 route
        被误认为 local-K 或 deployment-composition v015 evidence.

    主链路:
        上游: periodic, offline, sequence legacy evaluator entry.
        下游: fail closed; v015 只能使用 candidate diagnostic projection 或后续
        专属 composition evaluator.

    语义:
        只按 explicit v015 future-intent layout version 判定. 没有该 version 的
        历史 route 保持历史行为, 但不能作为 v015 active consumer.
    """

    layout = getattr(runner, "_frontres_future_intent_layout", None)
    if isinstance(layout, Mapping):
        version = str(layout.get("version", ""))
    else:
        version = str(getattr(layout, "version", ""))
    if version.startswith("frontres-v015-future-intent-q29-"):
        raise RuntimeError(
            f"{evaluator} is a legacy v002/quartet evaluator and rejects v015 future-intent layouts; "
            "use the candidate-only v015 diagnostic projection until a separately authorized evaluator route exists"
        )


def _capture_eval_gain_summary(capture: Any) -> tuple[Any | None, dict[str, Any]]:
    """Adapt canonical paired Gain output to the evaluation summary contract.

    Status: legacy v002 diagnostic adapter, not the Gain formula owner.
    Upstream: ``frontres_gain.compute_segment_gain`` via quartet paired capture.
    Downstream: legacy periodic/sequence/per-motion formatters only.
    Evidence: code-confirmed; v015 entry points reject before reaching it.
    Gap: v015 uses the separate candidate-only diagnostic projection until a
    later evaluator route is explicitly authorized.
    """
    result = _capture_paired_gain(capture)
    summary: dict[str, Any] = {
        "gain_source": "UNCONFIRMED",
    }
    for component in _EVAL_GAIN_COMPONENTS:
        summary[f"gain_{component}_per_sample"] = []
        summary[f"gain_{component}_mean"] = float("nan")
    if result is None:
        return None, summary

    total = result.gain_total.detach().float().reshape(-1)
    finite = torch.isfinite(total)
    if not bool(finite.any().item()):
        return result, summary
    summary.update(
        {
            "gain_source": "FRS-GAIN-v002",
            "gain_total_pos_frac": float((total[finite] > 0.0).float().mean().cpu().item()),
        }
    )
    for component in _EVAL_GAIN_COMPONENTS:
        value = getattr(result, component, None)
        summary[f"gain_{component}_per_sample"] = _float_values(value)
        summary[f"gain_{component}_mean"] = _finite_tensor_mean(value)
    _add_public_gain_component_aliases(summary)
    return result, summary


def _finite_tensor_mean(value: Any) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return float("nan")
    data = value.detach().float().reshape(-1)
    data = data[torch.isfinite(data)]
    return float(data.mean().cpu().item()) if data.numel() else float("nan")


def run_frontres_segment_periodic_eval(
    runner: Any,
    *,
    iteration: int,
    train_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate an independently sampled segment batch without changing training sampler state."""
    _reject_v015_legacy_evaluator(runner, evaluator="periodic evaluation")
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    if sampler is None:
        raise RuntimeError("periodic eval requires initialized FrontRES Segment sampler")
    sample_rows = globals().get("_sample_live_segment_rows")
    batch_builder = globals().get("_build_current_segment_batch")
    if sample_rows is None or batch_builder is None:
        from rsl_rl.runners.frontres_segment_live_sampler import (
            _build_current_segment_batch as imported_batch_builder,
            _sample_live_segment_rows as imported_sample_rows,
        )

        sample_rows = imported_sample_rows
        batch_builder = imported_batch_builder
    eval_steps = max(
        int(getattr(runner.env, "max_episode_length", 1)),
        int(getattr(runner.alg, "frontres_segment_k", 1)),
    )
    with _temporary_sampler_sampling_state(sampler):
        sample = sample_rows(runner, sampler)
    batch = batch_builder(runner, sample, update_step=0, print_probe=False)
    if batch is None:
        raise RuntimeError("periodic eval could not build an independent FrontRES Segment batch")
    sample_count = int(getattr(sample, "segment_ids").numel())
    with _temporary_runner_segment_eval_state(runner, sample=sample, batch=batch):
        with _temporary_eval_detail_silence(runner):
            _apply_current_segment_reset(runner)
            reset_applied = getattr(runner, "_frontres_segment_live_current_reset_result", None) is not None
            observations = _read_live_observations(runner)
            runner.eval_mode()
            try:
                capture = _run_live_rollout_capture(runner, observations, rollout_steps=eval_steps)
            finally:
                runner.train_mode()
    done_any = capture.done_any
    done = done_any.detach().bool().reshape(-1) if done_any is not None else None
    survival = capture.survival_steps
    survival_flat = survival.detach().float().reshape(-1) if survival is not None else None
    summary = {
        "episode_length": float(capture.rollout_k),
        "success_rate": float((~done).float().mean().cpu().item()) if done is not None and done.numel() else 0.0,
        "fall_rate": float(done.float().mean().cpu().item()) if done is not None and done.numel() else 0.0,
        "mean_survival_steps": float(survival_flat.mean().cpu().item()) if survival_flat is not None and survival_flat.numel() else 0.0,
        "sample_count": float(sample_count),
    }
    _, gain_summary = _capture_eval_gain_summary(capture)
    summary.update(gain_summary)
    summary.update(
        motion_quality_summary_to_scalars(
            clean_positions=getattr(capture, "motion_clean_body_pos", None),
            repaired_positions=getattr(capture, "motion_repaired_body_pos", None),
            noisy_positions=getattr(capture, "motion_noisy_body_pos", None),
            delta_se=getattr(capture, "transition_actions", None),
            valid_mask=(~done) if done is not None else None,
        )
    )
    summary.update(_offline_eval_perturbation_summary(batch))
    summary.update(
        {
            "eval_batch_source": "independent_sampler",
            "eval_iteration": int(iteration),
            "eval_reset_applied": bool(reset_applied),
            "motion_ids": _offline_eval_motion_ids_from_batch(batch, sample_count),
            "start_frames": _offline_eval_start_frames_from_batch(batch, sample_count),
        }
    )
    return summary


def run_frontres_segment_offline_eval(
    runner: Any,
    *,
    num_eval_segments: int,
    rollout_steps: int,
) -> dict[str, Any]:
    _reject_v015_legacy_evaluator(runner, evaluator="offline evaluation")
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    if sampler is None:
        raise RuntimeError("offline eval requires initialized FrontRES Segment sampler")
    env_count = max(1, int(getattr(runner.env, "num_envs", num_eval_segments)))
    requested_count = max(1, int(num_eval_segments))
    if requested_count != env_count:
        print(
            "[FrontRES Segment Offline Eval] "
            f"requested_segments={requested_count} env_count={env_count} "
            "using env_count; set --num_envs to choose eval sample count",
            flush=True,
        )
    sample = sampler.sample(env_count)
    runner._frontres_segment_live_current_sample = sample
    batch_builder = globals().get("_build_current_segment_batch")
    if batch_builder is None:
        from rsl_rl.runners.frontres_segment_live_sampler import _build_current_segment_batch as batch_builder

    try:
        batch = _build_current_segment_batch(runner, sample, update_step=0, print_probe=True)
        runner._frontres_segment_live_current_batch = batch
        with _temporary_eval_detail_silence(runner):
            _apply_current_segment_reset(runner)
            observations = _read_live_observations(runner)
            runner.eval_mode()
            capture = _run_live_rollout_capture(runner, observations, rollout_steps=max(1, int(rollout_steps)))
    finally:
        runner._frontres_segment_live_current_sample = None
        runner._frontres_segment_live_current_batch = None
        runner._frontres_segment_live_current_reset_request = None
        runner._frontres_segment_live_current_reset_result = None
    summary = _offline_eval_summary(
        capture,
        sample_count=env_count,
        motion_ids=_offline_eval_motion_ids_from_batch(batch, env_count),
    )
    print(
        "\n".join(("", _LOG_SEPARATOR, "", _format_offline_eval_log(summary), "")),
        flush=True,
    )
    return summary


def run_frontres_segment_sequence_offline_eval(
    runner: Any,
    *,
    num_eval_sequences: int,
    rollout_steps: int,
    max_preroll_steps: int | None = None,
    sampler_seed: int | None = None,
) -> dict[str, Any]:
    _reject_v015_legacy_evaluator(runner, evaluator="sequence evaluation")
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    if sampler is None:
        raise RuntimeError("sequence offline eval requires initialized FrontRES Segment sampler")
    import torch

    # FRS3-EVAL-011: checkpoint replay history is training state, not an
    # evaluation-sequence selector. Reset it so model_1/model_2 see the same
    # deterministic candidate distribution before the fixed plan is built.
    eval_seed = int(sampler_seed if sampler_seed is not None else getattr(runner, "seed", 0) or 0)
    reset_eval_sampler = getattr(sampler, "reset_for_deterministic_eval", None)
    if not callable(reset_eval_sampler):
        raise RuntimeError("sequence eval sampler lacks deterministic reset contract")
    reset_eval_sampler(seed=eval_seed)
    print(
        "[FrontRES Segment Sequence Eval Sampler] "
        f"reset=1 seed={eval_seed} checkpoint_replay_state=ignored",
        flush=True,
    )

    env_count = max(1, int(getattr(runner.env, "num_envs", num_eval_sequences)))
    requested_count = max(1, int(num_eval_sequences))

    # FRS3-EVAL-007: collect enough candidate specs to cover requested unique motions.
    batch_builder = globals().get("_build_current_segment_batch")
    if batch_builder is None:
        from rsl_rl.runners.frontres_segment_live_sampler import _build_current_segment_batch as batch_builder
    sequence_plan_builder = globals().get("build_frontres_sequence_eval_plan")
    sequence_reset_batch_builder = globals().get("build_frontres_sequence_eval_reset_batch")
    sequence_item_segment_ids = globals().get("segment_ids_for_sequence_eval_item")
    if sequence_plan_builder is None or sequence_reset_batch_builder is None or sequence_item_segment_ids is None:
        from rsl_rl.runners.frontres_segment_sequence_eval import (
            build_frontres_sequence_eval_plan as sequence_plan_builder,
            build_frontres_sequence_eval_reset_batch as sequence_reset_batch_builder,
            segment_ids_for_sequence_eval_item as sequence_item_segment_ids,
        )

    specs: list[Any] = []
    sample_size = max(env_count, requested_count)
    plan = None
    last_plan_error: ValueError | None = None
    for _ in range(max(8, requested_count * 20)):
        sample = sampler.sample(sample_size)
        batch = batch_builder(runner, sample, update_step=0, print_probe=False)
        specs.extend(tuple(getattr(batch, "specs", ()) or ()))
        try:
            plan = sequence_plan_builder(
                specs,
                requested_sequences=requested_count,
                available_envs=env_count,
                eval_rollout_steps=max(1, int(rollout_steps)),
                max_preroll_steps=max_preroll_steps,
            )
            break
        except ValueError as exc:
            last_plan_error = exc
    if plan is None:
        if last_plan_error is not None:
            raise last_plan_error
        raise ValueError("sequence eval could not build a plan from sampled specs")
    previous_eval_seed = getattr(runner, "_frontres_segment_sequence_eval_seed", None)
    runner._frontres_segment_sequence_eval_seed = eval_seed
    print(
        "[FrontRES Segment Sequence Eval Perturbation] "
        f"fixed=1 seed={eval_seed} checkpoint_iteration_ignored=1",
        flush=True,
    )
    if max_preroll_steps is not None and int(max_preroll_steps) > 0:
        print(
            "[FrontRES Segment Sequence Eval Plan] "
            f"max_preroll_steps={int(max_preroll_steps)} "
            "set OFFLINE_EVAL_MAX_PREROLL_STEPS=0 for unbounded full evaluation",
            flush=True,
        )
    summaries: list[dict[str, Any]] = []
    previous_sample = getattr(runner, "_frontres_segment_live_current_sample", None)
    previous_batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    try:
        # FRS3-EVAL-008: evaluate one full motion sequence per pass to avoid staggered windows.
        for item_index, item in enumerate(plan.items, start=1):
            print(
                "[FrontRES Segment Sequence Eval Progress] "
                f"sequence={item_index}/{plan.sequence_count} "
                f"motion_id={item.motion_id} "
                f"reset_frame={item.reset_frame} "
                f"preroll_steps={item.preroll_steps} "
                f"eval_steps={item.eval_rollout_steps}",
                flush=True,
            )
            segment_ids = torch.tensor(
                sequence_item_segment_ids(item, env_count=env_count),
                dtype=torch.long,
                device=getattr(runner, "device", "cpu"),
            )
            sample = SimpleNamespace(segment_ids=segment_ids)
            eval_batch = batch_builder(runner, sample, update_step=0, print_probe=False)
            reset_batch = sequence_reset_batch_builder(eval_batch, item)
            runner._frontres_segment_live_current_sample = sample
            capture, observations, reset_request, reset_result = _run_sequence_eval_capture_for_item(
                runner,
                item=item,
                eval_batch=eval_batch,
                reset_batch=reset_batch,
                zero_segment_action=False,
            )
            summary = _offline_eval_summary(
                capture,
                sample_count=env_count,
                motion_ids=_offline_eval_motion_ids_from_batch(eval_batch, env_count),
            )
            # FRS3-EVAL-010: attach sequence boundaries to the per-motion eval summary.
            summary.update(
                {
                    "sequence_eval": True,
                    "motion_id": item.motion_id,
                    "reset_frame": float(item.reset_frame),
                    "preroll_steps": float(item.preroll_steps),
                    "eval_start_frame": float(item.eval_start_frame),
                }
            )
            summary.update(_offline_eval_perturbation_summary(reset_batch))
            summaries.append(summary)
            zero_capture, _, _, _ = _run_sequence_eval_capture_for_item(
                runner,
                item=item,
                eval_batch=eval_batch,
                reset_batch=reset_batch,
                zero_segment_action=True,
            )
            zero_summary = _offline_eval_summary(
                zero_capture,
                sample_count=env_count,
                motion_ids=_offline_eval_motion_ids_from_batch(eval_batch, env_count),
            )
            print(_format_sequence_eval_item_log(item_index, plan.sequence_count, summary), flush=True)
            print(
                _format_sequence_eval_differential_log(
                    item_index=item_index,
                    sequence_count=plan.sequence_count,
                    summary=summary,
                    zero_summary=zero_summary,
                    capture=capture,
                    zero_capture=zero_capture,
                ),
                flush=True,
            )
            print(
                _format_sequence_eval_debug_log(
                    item_index=item_index,
                    sequence_count=plan.sequence_count,
                    item=item,
                    eval_batch=eval_batch,
                    reset_batch=reset_batch,
                    capture=capture,
                    summary=summary,
                    scoring_observations=observations,
                    reset_request=reset_request,
                    reset_result=reset_result,
                ),
                flush=True,
            )
    finally:
        runner._frontres_segment_live_current_sample = previous_sample
        runner._frontres_segment_live_current_batch = previous_batch
        runner._frontres_segment_live_current_reset_request = None
        runner._frontres_segment_live_current_reset_result = None
        if previous_eval_seed is None:
            runner.__dict__.pop("_frontres_segment_sequence_eval_seed", None)
        else:
            runner._frontres_segment_sequence_eval_seed = previous_eval_seed

    summary = _sequence_offline_eval_summary(summaries, plan=plan, env_count=env_count)
    # FRS3-EVAL-011: print compact whole-sequence evaluation metrics.
    print(
        "\n".join(("", _LOG_SEPARATOR, "", _format_sequence_offline_eval_log(summary), "")),
        flush=True,
    )
    return summary


def _run_sequence_eval_capture_for_item(
    runner: Any,
    *,
    item: Any,
    eval_batch: Any,
    reset_batch: Any,
    zero_segment_action: bool,
) -> tuple[Any, Any, Any, Any]:
    runner._frontres_segment_live_current_batch = reset_batch
    with _temporary_eval_detail_silence(runner):
        # FRS3-EVAL-009: reset at frame0, silence reset trace, no-capture preroll, then score.
        _apply_current_segment_reset(runner)
        reset_request = getattr(runner, "_frontres_segment_live_current_reset_request", None)
        reset_result = getattr(runner, "_frontres_segment_live_current_reset_result", None)
        observations = _read_live_observations(runner)
        runner.eval_mode()
        if item.preroll_steps > 0:
            _run_live_rollout_capture(
                runner,
                observations,
                rollout_steps=item.preroll_steps,
                capture_motion_quality=False,
                zero_segment_action=False,
            )
            observations = _read_live_observations(runner)
        runner._frontres_segment_live_current_batch = eval_batch
        capture = _run_live_rollout_capture(
            runner,
            observations,
            rollout_steps=item.eval_rollout_steps,
            zero_segment_action=bool(zero_segment_action),
        )
    return capture, observations, reset_request, reset_result


def _sequence_offline_eval_summary(
    summaries: list[dict[str, Any]],
    *,
    plan: Any,
    env_count: int,
) -> dict[str, Any]:
    # FRS3-EVAL-012: merge per-sequence metrics into one reviewable result.
    if not summaries:
        return {"sequence_eval": True, "sequence_count": 0.0, "sample_count": 0.0}
    numeric_keys = {
        key
        for summary in summaries
        for key, value in summary.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    merged = {}
    for key in numeric_keys:
        values = [float(summary[key]) for summary in summaries if key in summary and math.isfinite(float(summary[key]))]
        merged[key] = sum(values) / float(len(values)) if values else float("nan")
    per_motion = []
    for summary in summaries:
        rows = tuple(summary.get("per_motion", ()) or ())
        if rows:
            per_motion.extend(rows)
    merged.update(
        {
            "sequence_eval": True,
            "sequence_count": float(len(summaries)),
            "requested_sequences": float(getattr(plan, "requested_sequences", len(summaries))),
            "env_count": float(env_count),
            "motion_ids": tuple(getattr(plan, "motion_ids", ())),
            "per_motion": per_motion,
            "gain_source": (
                "FRS-GAIN-v002"
                if all(summary.get("gain_source") == "FRS-GAIN-v002" for summary in summaries)
                else "UNCONFIRMED"
            ),
        }
    )
    return merged


def _format_sequence_eval_debug_log(
    *,
    item_index: int,
    sequence_count: int,
    item: Any,
    eval_batch: Any,
    reset_batch: Any,
    capture: Any,
    summary: Mapping[str, Any],
    scoring_observations: Any,
    reset_request: Any,
    reset_result: Any,
) -> str:
    lines = [
        "[FrontRES Segment Sequence Eval Debug]",
        (
            "  plan: "
            f"sequence={item_index}/{sequence_count} "
            f"segment_id={getattr(item, 'segment_id', 'missing')} "
            f"motion_id={getattr(item, 'motion_id', 'missing')} "
            f"reset_frame={getattr(item, 'reset_frame', 'missing')} "
            f"preroll_steps={getattr(item, 'preroll_steps', 'missing')} "
            f"eval_start_frame={getattr(item, 'eval_start_frame', 'missing')} "
            f"eval_steps={getattr(item, 'eval_rollout_steps', 'missing')} "
            f"horizon_k={getattr(item, 'segment_horizon_k', 'missing')}"
        ),
        f"  eval_batch: {_sequence_debug_batch(eval_batch)}",
        f"  reset_batch: {_sequence_debug_batch(reset_batch)}",
        f"  reset_request: {_sequence_debug_reset_request(reset_request)}",
        f"  reset_result: {_sequence_debug_reset_result(reset_result)}",
        f"  scoring_observations: {_sequence_debug_value(scoring_observations, max_items=8)}",
        (
            "  capture_roles: "
            f"rollout_k={getattr(capture, 'rollout_k', 'missing')} "
            f"n_train={getattr(capture, 'n_train', 'missing')} "
            f"n_candidate={getattr(capture, 'n_candidate', 'missing')} "
            f"n_base={getattr(capture, 'n_base', 'missing')} "
            f"n_clean={getattr(capture, 'n_clean', 'missing')}"
        ),
        (
            "  capture_shapes: "
            f"last_obs_shape={getattr(capture, 'last_obs_shape', 'missing')} "
            f"policy_action_shape={getattr(capture, 'action_shape', 'missing')} "
            f"env_action_shape={getattr(capture, 'env_action_shape', 'missing')} "
            f"reward_mean={float(getattr(capture, 'reward_mean', 0.0)):.6f} "
            f"done_frac={float(getattr(capture, 'done_frac', 0.0)):.6f}"
        ),
        f"  capture_legacy_reward_accum_raw: {_sequence_debug_value(getattr(capture, 'reward_accum', None))}",
        f"  capture_done_any: {_sequence_debug_value(getattr(capture, 'done_any', None))}",
        f"  capture_survival_steps: {_sequence_debug_value(getattr(capture, 'survival_steps', None))}",
        f"  capture_actor_update_mask: {_sequence_debug_value(getattr(capture, 'actor_update_mask', None))}",
        f"  raw_policy_action: {_sequence_debug_value(getattr(capture, 'env_actions', None))}",
        f"  segment_transition_actions: {_sequence_debug_actions(getattr(capture, 'transition_actions', None), capture)}",
        f"  policy_anti_rp_alignment: {_sequence_eval_anti_rp_alignment(capture)}",
        f"  transition_supervised_target: {_sequence_debug_value(getattr(capture, 'transition_supervised_target', None))}",
        (
            "  oracles: "
            f"{_sequence_eval_oracles(item, eval_batch, reset_batch, capture, summary, reset_request, reset_result)}"
        ),
        f"  differential_proxy: {_sequence_eval_differential_proxy(capture, summary)}",
        f"  action_distribution_health: {_sequence_eval_action_distribution_health(capture)}",
        f"  transition_log_probs: {_sequence_debug_value(getattr(capture, 'transition_log_probs', None))}",
        f"  transition_values: {_sequence_debug_value(getattr(capture, 'transition_values', None))}",
        f"  transition_means: {_sequence_debug_value(getattr(capture, 'transition_means', None))}",
        f"  transition_sigmas: {_sequence_debug_value(getattr(capture, 'transition_sigmas', None))}",
        f"  transition_obs: {_sequence_debug_value(getattr(capture, 'transition_obs', None), max_items=8)}",
        f"  transition_privileged_obs: {_sequence_debug_value(getattr(capture, 'transition_privileged_obs', None), max_items=8)}",
        f"  motion_clean_body_pos: {_sequence_debug_value(getattr(capture, 'motion_clean_body_pos', None), max_items=8)}",
        f"  motion_repaired_body_pos: {_sequence_debug_value(getattr(capture, 'motion_repaired_body_pos', None), max_items=8)}",
        f"  motion_noisy_body_pos: {_sequence_debug_value(getattr(capture, 'motion_noisy_body_pos', None), max_items=8)}",
        f"  motion_role_errors: {_sequence_debug_motion_errors(capture)}",
        (
            "  summary: "
            f"success={float(summary.get('success_rate', 0.0)):.6f} "
            f"fall={float(summary.get('fall_rate', 0.0)):.6f} "
            f"survival={float(summary.get('mean_survival_steps', 0.0)):.6f} "
            f"gain_source={summary.get('gain_source', 'UNCONFIRMED')} "
            f"gain_style={_fmt_eval_value(summary.get('gain_style_mean'))} "
            f"gain_physics={_fmt_eval_value(summary.get('gain_physics_mean'))} "
            f"gain_repair_cost={_fmt_eval_value(summary.get('gain_repair_cost_mean'))} "
            f"gain_total={_fmt_eval_value(summary.get('gain_total_mean'))} "
            f"mpjpe_repaired={float(summary.get('segment/motion_mpjpe_repaired_clean', 0.0)):.6f} "
            f"mpjpe_noisy={float(summary.get('segment/motion_mpjpe_noisy_clean', 0.0)):.6f} "
            f"delta_se_norm={float(summary.get('segment/motion_delta_se_norm', 0.0)):.6f}"
        ),
    ]
    return "\n".join(lines)


def _sequence_eval_action_distribution_health(capture: Any) -> dict[str, float | str | bool]:
    return action_distribution_health_summary(
        means=getattr(capture, "transition_means", None),
        sigmas=getattr(capture, "transition_sigmas", None),
        actions=getattr(capture, "transition_actions", None),
        supervised_target=getattr(capture, "transition_supervised_target", None),
    )


def _format_sequence_eval_differential_log(
    *,
    item_index: int,
    sequence_count: int,
    summary: Mapping[str, Any],
    zero_summary: Mapping[str, Any],
    capture: Any,
    zero_capture: Any,
) -> str:
    real_gain = float(summary.get("gain_total_mean", float("nan")))
    zero_gain = float(zero_summary.get("gain_total_mean", float("nan")))
    real_mpjpe = float(summary.get("segment/motion_mpjpe_repaired_clean", 0.0))
    zero_mpjpe = float(zero_summary.get("segment/motion_mpjpe_repaired_clean", 0.0))
    real_survival = float(summary.get("mean_survival_steps", 0.0))
    zero_survival = float(zero_summary.get("mean_survival_steps", 0.0))
    real_fall = float(summary.get("fall_rate", 0.0))
    zero_fall = float(zero_summary.get("fall_rate", 0.0))
    real_action_norm = _sequence_tensor_l2_mean(getattr(capture, "transition_actions", None))
    zero_action_norm = _sequence_tensor_l2_mean(getattr(zero_capture, "transition_actions", None))
    return "\n".join(
        (
            "[FrontRES Segment Sequence Eval Differential]",
            (
                f"  sequence: {item_index}/{sequence_count} "
                f"motion_id={summary.get('motion_id', 'missing')}"
            ),
            (
                "  real_policy: "
                f"gain_total={_fmt_eval_value(real_gain)} "
                f"mpjpe_repaired={real_mpjpe:.6f} "
                f"fall={real_fall:.6f} "
                f"survival={real_survival:.6f} "
                f"segment_action_norm={real_action_norm:.6f}"
            ),
            (
                "  zero_policy: "
                f"gain_total={_fmt_eval_value(zero_gain)} "
                f"mpjpe_repaired={zero_mpjpe:.6f} "
                f"fall={zero_fall:.6f} "
                f"survival={zero_survival:.6f} "
                f"segment_action_norm={zero_action_norm:.6f}"
            ),
            (
                "  real_minus_zero: "
                f"gain_total={_fmt_eval_value(real_gain - zero_gain)} "
                f"mpjpe={real_mpjpe - zero_mpjpe:.6f} "
                f"fall={real_fall - zero_fall:.6f} "
                f"survival={real_survival - zero_survival:.6f} "
                f"real_beats_zero_mpjpe={real_mpjpe < zero_mpjpe} "
                f"zero_action_is_zero={zero_action_norm <= 1e-8}"
            ),
        )
    )


def _sequence_eval_oracles(
    item: Any,
    eval_batch: Any,
    reset_batch: Any,
    capture: Any,
    summary: Mapping[str, Any],
    reset_request: Any,
    reset_result: Any,
) -> dict[str, bool | str]:
    eval_start = int(getattr(item, "eval_start_frame", -1))
    motion_id = str(getattr(item, "motion_id", ""))
    families = tuple(str(value) for value in getattr(reset_request, "perturbation_family", ()) or ())
    if not families:
        families = tuple(str(value) for value in getattr(reset_batch, "stage3_index_perturbation_family", ()) or ())
    return {
        "reset_frame0": int(getattr(item, "reset_frame", -1)) == 0,
        "reset_request_frame0": _sequence_values_all_int(getattr(reset_request, "start_frames", None), 0),
        "reset_batch_frame0": _sequence_specs_start_all(reset_batch, 0),
        "eval_batch_frame": _sequence_specs_start_all(eval_batch, eval_start),
        "motion_id_aligned": _sequence_specs_motion_all(eval_batch, motion_id)
        and _sequence_specs_motion_all(reset_batch, motion_id),
        "preroll_not_scored": int(getattr(capture, "rollout_k", -1)) == int(getattr(item, "eval_rollout_steps", -2)),
        "rp_only": bool(families)
        and set(families).issubset({"local_rp", "rp"})
        and float(summary.get("perturbation_non_rp_frac", 0.0)) <= 1e-6,
        "roles_aligned": int(getattr(capture, "n_train", 0)) == int(getattr(capture, "n_base", -1))
        and int(getattr(capture, "n_train", 0)) > 0,
        "metric_shapes_aligned": _sequence_same_tensor_shape(
            getattr(capture, "motion_clean_body_pos", None),
            getattr(capture, "motion_repaired_body_pos", None),
            getattr(capture, "motion_noisy_body_pos", None),
        ),
        "action_shape_visible": hasattr(getattr(capture, "transition_actions", None), "detach")
        or hasattr(getattr(capture, "env_actions", None), "detach"),
        "reset_success_all": _sequence_bool_tensor_all(getattr(reset_result, "success_mask", None)),
        "summary_motion_aligned": str(summary.get("motion_id", motion_id)) == motion_id,
    }


def _sequence_eval_differential_proxy(capture: Any, summary: Mapping[str, Any]) -> dict[str, float | bool]:
    raw_norm = _sequence_tensor_l2_mean(getattr(capture, "env_actions", None))
    segment_norm = _sequence_tensor_l2_mean(getattr(capture, "transition_actions", None))
    gain_total = float(summary.get("gain_total_mean", float("nan")))
    mpjpe_delta = float(summary.get("segment/motion_mpjpe_repaired_clean", 0.0)) - float(
        summary.get("segment/motion_mpjpe_noisy_clean", 0.0)
    )
    return {
        "raw_action_norm_mean": _round_float(raw_norm),
        "segment_action_norm_mean": _round_float(segment_norm),
        "raw_action_nonzero": raw_norm > 1e-8,
        "segment_action_nonzero": segment_norm > 1e-8,
        "gain_total_mean": _round_float(gain_total) if math.isfinite(gain_total) else float("nan"),
        "mpjpe_repaired_minus_noisy": _round_float(mpjpe_delta),
        "repaired_beats_noisy_mpjpe": mpjpe_delta < 0.0,
    }


def _sequence_specs_start_all(batch: Any, expected: int) -> bool:
    specs = tuple(getattr(batch, "specs", ()) or ())
    return bool(specs) and all(int(getattr(spec, "start_frame", -999999)) == int(expected) for spec in specs)


def _sequence_specs_motion_all(batch: Any, expected: str) -> bool:
    specs = tuple(getattr(batch, "specs", ()) or ())
    return bool(specs) and all(str(getattr(spec, "motion_id", "")) == str(expected) for spec in specs)


def _sequence_values_all_int(value: Any, expected: int) -> bool | str:
    if not hasattr(value, "detach"):
        return "missing"
    flat = value.detach().reshape(-1).cpu()
    if not bool(flat.numel()):
        return "empty"
    return all(int(item) == int(expected) for item in flat.tolist())


def _sequence_bool_tensor_all(value: Any) -> bool | str:
    if not hasattr(value, "detach"):
        return "missing"
    flat = value.detach().bool().reshape(-1).cpu()
    if not bool(flat.numel()):
        return "empty"
    return all(bool(item) for item in flat.tolist())


def _sequence_same_tensor_shape(*values: Any) -> bool:
    shapes = [tuple(value.shape) for value in values if hasattr(value, "shape")]
    return len(shapes) == len(values) and len(set(shapes)) == 1


def _sequence_tensor_l2_mean(value: Any) -> float:
    if not hasattr(value, "detach"):
        return 0.0
    tensor = value.detach().float().cpu()
    if tensor.numel() == 0:
        return 0.0
    if tensor.ndim >= 2:
        return float(tensor.reshape(tensor.shape[0], -1).norm(dim=1).mean().item())
    return float(tensor.reshape(-1).abs().mean().item())


def _sequence_debug_batch(batch: Any) -> str:
    if batch is None:
        return "missing"
    specs = tuple(getattr(batch, "specs", ()) or ())
    spec_rows = [
        {
            "segment_id": int(getattr(spec, "segment_id", -1)),
            "motion_id": str(getattr(spec, "motion_id", "")),
            "start_frame": int(getattr(spec, "start_frame", -1)),
            "horizon_k": int(getattr(spec, "horizon_k", -1)),
        }
        for spec in specs[:8]
    ]
    plan = getattr(batch, "stage3_index_perturbation_plan", None)
    return (
        f"segment_ids={_sequence_debug_value(getattr(batch, 'segment_ids', None))} "
        f"specs_head={spec_rows} "
        f"perturbation_family={tuple(getattr(batch, 'perturbation_family', ()) or ())[:8]} "
        f"perturbation_strength={_sequence_debug_value(getattr(batch, 'perturbation_strength', None))} "
        f"stage3_family={tuple(getattr(batch, 'stage3_index_perturbation_family', ()) or ())[:8]} "
        f"stage3_strength={_sequence_debug_value(getattr(batch, 'stage3_index_perturbation_strength', None))} "
        f"stage3_plan={_sequence_debug_plan(plan)}"
    )


def _sequence_debug_plan(plan: Any) -> str:
    if plan is None:
        return "missing"
    return (
        f"family={tuple(getattr(plan, 'perturbation_family', ()) or ())[:8]} "
        f"strength={_sequence_debug_value(getattr(plan, 'perturbation_strength', None))} "
        f"active_modes={tuple(getattr(plan, 'active_modes', ()) or ())} "
        f"complexity={getattr(plan, 'complexity', 'missing')} "
        f"mix_mode={getattr(plan, 'mix_mode', 'missing')} "
        f"progress={float(getattr(plan, 'progress', 0.0)):.6f} "
        f"seq_idx={int(getattr(plan, 'seq_idx', -1))} "
        f"mix_diag={dict(getattr(plan, 'mix_diag', {}) or {})}"
    )


def _sequence_debug_reset_request(request: Any) -> str:
    if request is None:
        return "missing"
    return (
        f"segment_ids={_sequence_debug_value(getattr(request, 'segment_ids', None))} "
        f"motion_ids={tuple(getattr(request, 'motion_ids', ()) or ())[:8]} "
        f"start_frames={_sequence_debug_value(getattr(request, 'start_frames', None))} "
        f"horizon_k={_sequence_debug_value(getattr(request, 'horizon_k', None))} "
        f"perturbation_family={tuple(getattr(request, 'perturbation_family', ()) or ())[:8]} "
        f"perturbation_strength={_sequence_debug_value(getattr(request, 'perturbation_strength', None))} "
        f"valid_mask={_sequence_debug_value(getattr(request, 'valid_mask', None))}"
    )


def _sequence_debug_reset_result(result: Any) -> str:
    if result is None:
        return "missing"
    return (
        f"success_mask={_sequence_debug_value(getattr(result, 'success_mask', None))} "
        f"direct_reset_mask={_sequence_debug_value(getattr(result, 'direct_reset_mask', None))} "
        f"preroll_mask={_sequence_debug_value(getattr(result, 'preroll_mask', None))} "
        f"velocity_mismatch={_sequence_debug_value(getattr(result, 'velocity_mismatch', None))}"
    )


def _sequence_debug_actions(value: Any, capture: Any) -> str:
    if not hasattr(value, "detach"):
        return _sequence_debug_value(value)
    tensor = value.detach().float().cpu()
    rows = []
    if tensor.ndim >= 2:
        labels = _sequence_role_labels(capture, int(tensor.shape[0]))
        flat = tensor.reshape(tensor.shape[0], -1)
        for index in range(min(int(flat.shape[0]), 8)):
            vec = flat[index, : min(int(flat.shape[1]), 6)]
            rows.append(
                {
                    "role": labels[index],
                    "delta": _round_list(vec.tolist()),
                    "pos_norm": _round_float(vec[:3].norm().item()) if vec.numel() >= 3 else 0.0,
                    "rpy_norm": _round_float(vec[3:6].norm().item()) if vec.numel() >= 6 else 0.0,
                }
            )
    return f"{_sequence_debug_value(value)} rows={rows}"


def _sequence_eval_anti_rp_alignment(capture: Any) -> dict[str, Any]:
    perturb_rp = getattr(capture, "transition_perturbation_rp", None)
    target = getattr(capture, "transition_supervised_target", None)
    actions = getattr(capture, "transition_actions", None)
    means = getattr(capture, "transition_means", None)
    if not hasattr(perturb_rp, "detach"):
        return {"available": False, "reason": "missing_transition_perturbation_rp"}
    rp = perturb_rp.detach().float().cpu()
    if rp.ndim != 2 or int(rp.shape[-1]) < 2:
        return {"available": False, "reason": f"bad_perturbation_shape={tuple(rp.shape)}"}
    n_train = max(0, min(int(getattr(capture, "n_train", 0)), int(rp.shape[0])))
    if n_train <= 0:
        return {"available": False, "reason": "no_train_rows"}
    rp = rp[:n_train, :2]
    anti = -rp
    result: dict[str, Any] = {
        "available": True,
        "perturb_rp_head": _round_list(rp[:4].tolist()),
        "anti_rp_head": _round_list(anti[:4].tolist()),
        "anti_rp_norm_mean": _round_float(rp.norm(dim=1).mean().item()),
        "max_delta_rpy": _round_float(float(getattr(capture, "max_delta_rpy", 0.0) or 0.0)),
    }
    target_rp = _sequence_eval_target_rp(target, n_train=n_train)
    if target_rp is not None:
        result["target_rp_head"] = _round_list(target_rp[:4].tolist())
        result["target_rp_norm_mean"] = _round_float(target_rp.norm(dim=1).mean().item())
        result["target_vs_anti_sign_agree_frac"] = _rp_sign_agree_frac(target_rp, anti)
        result["target_norm_over_anti_norm"] = _safe_norm_ratio(target_rp, anti)
    _add_rp_sign_stats(result, "action", actions, anti, n_train=n_train)
    _add_rp_sign_stats(result, "mean", means, anti, n_train=n_train)
    _add_policy_scaling_stats(result, actions, means, target_rp, capture, n_train=n_train)
    return result


def _sequence_eval_target_rp(value: Any, *, n_train: int) -> Any:
    if not hasattr(value, "detach"):
        return None
    tensor = value.detach().float().cpu()
    if tensor.ndim != 2 or int(tensor.shape[-1]) < 5:
        return None
    n = max(0, min(int(n_train), int(tensor.shape[0])))
    if n <= 0:
        return None
    return tensor[:n, 3:5]


def _rp_sign_agree_frac(value: Any, target: Any) -> Any:
    import torch

    valid = target.abs() > 1e-6
    if not bool(valid.any().item()):
        return None
    return _round_float((torch.sign(value[valid]) == torch.sign(target[valid])).float().mean().item())


def _safe_norm_ratio(value: Any, target: Any) -> Any:
    denom = float(target.norm(dim=1).mean().item())
    if abs(denom) <= 1e-8:
        return None
    return _round_float(value.norm(dim=1).mean().item() / denom)


def _add_policy_scaling_stats(
    result: dict[str, Any],
    actions: Any,
    means: Any,
    target_rp: Any,
    capture: Any,
    *,
    n_train: int,
) -> None:
    import torch

    if target_rp is None or not hasattr(means, "detach"):
        result["raw_to_delta_available"] = False
        return
    mean_tensor = means.detach().float().cpu()
    if mean_tensor.ndim != 2 or int(mean_tensor.shape[-1]) < 5:
        result["raw_to_delta_available"] = False
        result["raw_to_delta_reason"] = f"bad_mean_shape={tuple(mean_tensor.shape)}"
        return
    n = max(0, min(int(n_train), int(mean_tensor.shape[0]), int(target_rp.shape[0])))
    if n <= 0:
        result["raw_to_delta_available"] = False
        result["raw_to_delta_reason"] = "no_rows"
        return
    max_delta_rpy = float(getattr(capture, "max_delta_rpy", 0.0) or 0.0)
    mean_raw_rp = mean_tensor[:n, 3:5]
    mean_delta_rp = torch.tanh(mean_raw_rp) * max_delta_rpy
    result["raw_to_delta_available"] = True
    result["mean_delta_rp_head"] = _round_list(mean_delta_rp[:4].tolist())
    result["mean_delta_norm_over_target_norm"] = _safe_norm_ratio(mean_delta_rp, target_rp[:n])
    result["mean_raw_abs_max"] = _round_float(mean_raw_rp.abs().max().item())
    result["mean_raw_saturated_frac_abs_gt_2"] = _round_float(mean_raw_rp.abs().gt(2.0).float().mean().item())
    result["mean_delta_vs_target_sign_agree_frac"] = _rp_sign_agree_frac(mean_delta_rp, target_rp[:n])
    if hasattr(actions, "detach"):
        action_tensor = actions.detach().float().cpu()
        if action_tensor.ndim == 2 and int(action_tensor.shape[-1]) >= 5:
            action_rp = action_tensor[:n, 3:5]
            result["action_norm_over_target_norm"] = _safe_norm_ratio(action_rp, target_rp[:n])
            result["action_vs_target_sign_agree_frac"] = _rp_sign_agree_frac(action_rp, target_rp[:n])


def _add_rp_sign_stats(
    result: dict[str, Any],
    prefix: str,
    value: Any,
    anti_rp: Any,
    *,
    n_train: int,
) -> None:
    import torch

    if not hasattr(value, "detach"):
        result[f"{prefix}_available"] = False
        return
    tensor = value.detach().float().cpu()
    if tensor.ndim != 2 or int(tensor.shape[-1]) < 5:
        result[f"{prefix}_available"] = False
        result[f"{prefix}_reason"] = f"bad_shape={tuple(tensor.shape)}"
        return
    rp_action = tensor[:n_train, 3:5]
    valid = anti_rp.abs() > 1e-6
    if not bool(valid.any().item()):
        result[f"{prefix}_available"] = False
        result[f"{prefix}_reason"] = "zero_anti_rp"
        result[f"{prefix}_rp_head"] = _round_list(rp_action[:4].tolist())
        return
    action_sign = torch.sign(rp_action[valid])
    anti_sign = torch.sign(anti_rp[valid])
    result[f"{prefix}_available"] = True
    result[f"{prefix}_rp_head"] = _round_list(rp_action[:4].tolist())
    result[f"{prefix}_anti_sign_agree_frac"] = _round_float((action_sign == anti_sign).float().mean().item())
    result[f"{prefix}_same_as_perturb_frac"] = _round_float((action_sign == -anti_sign).float().mean().item())
    result[f"{prefix}_rp_norm_mean"] = _round_float(rp_action.norm(dim=1).mean().item())


def _sequence_debug_motion_errors(capture: Any) -> list[dict[str, float | int | str]]:
    clean = getattr(capture, "motion_clean_body_pos", None)
    repaired = getattr(capture, "motion_repaired_body_pos", None)
    noisy = getattr(capture, "motion_noisy_body_pos", None)
    repaired_err = _sequence_role_l2_mean(repaired, clean)
    noisy_err = _sequence_role_l2_mean(noisy, clean)
    if not repaired_err and not noisy_err:
        return []
    count = max(len(repaired_err), len(noisy_err))
    labels = _sequence_role_labels(capture, count)
    rows = []
    for index in range(min(count, 8)):
        rep = repaired_err[index] if index < len(repaired_err) else 0.0
        noi = noisy_err[index] if index < len(noisy_err) else 0.0
        rows.append(
            {
                "role": labels[index],
                "repaired_mpjpe": _round_float(rep),
                "noisy_mpjpe": _round_float(noi),
                "repair_delta": _round_float(rep - noi),
            }
        )
    return rows


def _sequence_role_l2_mean(value: Any, reference: Any) -> list[float]:
    if not (hasattr(value, "detach") and hasattr(reference, "detach")):
        return []
    lhs = value.detach().float().cpu()
    rhs = reference.detach().float().cpu()
    if tuple(lhs.shape) != tuple(rhs.shape) or lhs.ndim == 0:
        return []
    diff = lhs - rhs
    if diff.ndim >= 2 and int(diff.shape[-1]) in (2, 3, 4, 6):
        diff = diff.norm(dim=-1)
    else:
        diff = diff.abs()
    if diff.ndim == 1:
        per_role = diff
    else:
        per_role = diff.reshape(diff.shape[0], -1).mean(dim=1)
    return [float(item) for item in per_role.tolist()]


def _sequence_role_labels(capture: Any, count: int) -> list[str]:
    sizes = (
        ("train", max(0, int(getattr(capture, "n_train", 0)))),
        ("candidate", max(0, int(getattr(capture, "n_candidate", 0)))),
        ("base", max(0, int(getattr(capture, "n_base", 0)))),
        ("clean", max(0, int(getattr(capture, "n_clean", 0)))),
    )
    labels = []
    for name, size in sizes:
        labels.extend(f"{name}{index}" for index in range(size))
    labels.extend(f"env{index}" for index in range(len(labels), count))
    return labels[:count]


def _sequence_debug_value(value: Any, *, max_items: int = 16) -> str:
    if value is None:
        return "missing"
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        flat = tensor.reshape(-1)
        head = _round_list(flat[:max_items].tolist())
        if flat.numel() == 0:
            return f"shape={shape} dtype={dtype} empty=True"
        flat_float = flat.float()
        finite_frac = float(torch_isfinite(flat_float).float().mean().item())
        return (
            f"shape={shape} dtype={dtype} finite_frac={finite_frac:.3f} "
            f"min={float(flat_float.min().item()):.6f} "
            f"max={float(flat_float.max().item()):.6f} "
            f"mean={float(flat_float.mean().item()):.6f} "
            f"head={head}"
        )
    if isinstance(value, Mapping):
        return f"mapping_keys={tuple(value.keys())[:max_items]}"
    if isinstance(value, (list, tuple)):
        return repr(tuple(value[:max_items]))
    text = repr(value)
    return text if len(text) <= 240 else text[:237] + "..."


def torch_isfinite(value: Any) -> Any:
    import torch

    return torch.isfinite(value)


def _round_list(values: Any) -> list[Any]:
    rounded = []
    for value in values:
        if isinstance(value, float):
            rounded.append(_round_float(value))
        elif isinstance(value, (list, tuple)):
            rounded.append(_round_list(value))
        else:
            rounded.append(value)
    return rounded


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _fmt_eval_value(value: Any, *, percent: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "UNCONFIRMED"
    if not math.isfinite(numeric):
        return "UNCONFIRMED"
    return f"{numeric * 100.0:.1f}%" if percent else f"{numeric:.6f}"


def _format_eval_gain_line(summary: Mapping[str, Any], *, indent: str) -> str:
    return (
        f"{indent}gain: "
        f"source={summary.get('gain_source', 'UNCONFIRMED')} "
        f"style={_fmt_eval_value(summary.get('gain_style_mean'))} "
        f"physics={_fmt_eval_value(summary.get('gain_physics_mean'))} "
        f"physics_components=(success={_fmt_eval_value(summary.get('gain_physics_success_mean'))} "
        f"survival_quality=(repaired={_fmt_eval_value(summary.get('gain_physics_survival_quality_repaired_mean'))} "
        f"noisy={_fmt_eval_value(summary.get('gain_physics_survival_quality_noisy_mean'))} "
        f"gain={_fmt_eval_value(summary.get('gain_physics_survival_mean'))}) "
        f"zmp={_fmt_eval_value(summary.get('gain_physics_zmp_mean'))} "
        f"contact={_fmt_eval_value(summary.get('gain_physics_contact_mean'))}) "
        f"repair_cost={_fmt_eval_value(summary.get('gain_repair_cost_mean'))} "
        f"total={_fmt_eval_value(summary.get('gain_total_mean'))} "
        f"positive={_fmt_eval_value(summary.get('gain_total_pos_frac'), percent=True)}"
    )


def _format_sequence_eval_item_log(item_index: int, sequence_count: int, summary: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "[FrontRES Segment Sequence Eval Item]",
            (
                "  sequence: "
                f"{item_index}/{sequence_count} "
                f"motion_id={summary.get('motion_id', 'unknown')} "
                f"reset_frame={int(float(summary.get('reset_frame', 0.0)))} "
                f"preroll_steps={int(float(summary.get('preroll_steps', 0.0)))} "
                f"eval_start_frame={int(float(summary.get('eval_start_frame', 0.0)))}"
            ),
            (
                "  result: "
                f"success={float(summary.get('success_rate', 0.0)) * 100.0:.1f}% "
                f"fall={float(summary.get('fall_rate', 0.0)) * 100.0:.1f}% "
                f"survival={float(summary.get('mean_survival_steps', 0.0)):.1f}"
            ),
            _format_eval_gain_line(summary, indent="  "),
            (
                "  motion: "
                f"mpjpe_repaired={float(summary.get('segment/motion_mpjpe_repaired_clean', 0.0)):.6f} "
                f"mpjpe_noisy={float(summary.get('segment/motion_mpjpe_noisy_clean', 0.0)):.6f} "
                f"vel_err={float(summary.get('segment/motion_vel_error_repaired_clean', 0.0)):.6f} "
                f"acc_err={float(summary.get('segment/motion_acc_error_repaired_clean', 0.0)):.6f} "
                f"delta_se_norm={float(summary.get('segment/motion_delta_se_norm', 0.0)):.6f}"
            ),
            (
                "  perturbation: "
                f"family_counts={summary.get('perturbation_family_counts', {})} "
                f"strength_min={float(summary.get('perturbation_strength_min', 0.0)):.6f} "
                f"strength_mean={float(summary.get('perturbation_strength_mean', 0.0)):.6f} "
                f"strength_max={float(summary.get('perturbation_strength_max', 0.0)):.6f} "
                f"local_rp_frac={float(summary.get('perturbation_local_rp_frac', 0.0)) * 100.0:.1f}% "
                f"non_rp_frac={float(summary.get('perturbation_non_rp_frac', 0.0)) * 100.0:.1f}%"
            ),
        )
    )


def _format_sequence_offline_eval_log(summary: Mapping[str, Any]) -> str:
    lines = []
    per_motion = tuple(summary.get("per_motion", ()) or ())
    if per_motion:
        lines.append("[FrontRES Segment Sequence Eval / Per Motion]")
        for row in per_motion:
            lines.extend(
                (
                    f"  motion: id={row.get('motion_id', 'unknown')} samples={int(row.get('sample_count', 0))}",
                    (
                        "    result: "
                        f"success={float(row.get('success_rate', 0.0)) * 100.0:.1f}% "
                        f"fall={float(row.get('fall_rate', 0.0)) * 100.0:.1f}% "
                        f"survival={float(row.get('mean_survival_steps', 0.0)):.1f}"
                    ),
                    _format_eval_gain_line(row, indent="    "),
                    (
                        "    motion: "
                        f"mpjpe_repaired={float(row.get('segment/motion_mpjpe_repaired_clean', 0.0)):.6f} "
                        f"mpjpe_noisy={float(row.get('segment/motion_mpjpe_noisy_clean', 0.0)):.6f} "
                        f"vel_err={float(row.get('segment/motion_vel_error_repaired_clean', 0.0)):.6f} "
                        f"acc_err={float(row.get('segment/motion_acc_error_repaired_clean', 0.0)):.6f} "
                        f"delta_se_norm={float(row.get('segment/motion_delta_se_norm', 0.0)):.6f}"
                    ),
                )
            )
        lines.append("")

    lines.extend([
        "[FrontRES Segment Sequence Eval]",
        (
            "  rollout: "
            f"sequence_count={int(summary.get('sequence_count', 0))} "
            f"requested_sequences={int(summary.get('requested_sequences', 0))} "
            f"env_count={int(summary.get('env_count', 0))} "
            f"episode_length={float(summary.get('episode_length', 0.0)):.1f}"
        ),
        (
            "  result: "
            f"success={float(summary.get('success_rate', 0.0)) * 100.0:.1f}% "
            f"fall={float(summary.get('fall_rate', 0.0)) * 100.0:.1f}% "
            f"survival={float(summary.get('mean_survival_steps', 0.0)):.1f}"
        ),
        _format_eval_gain_line(summary, indent="  "),
        (
            "  motion: "
            f"mpjpe_repaired={float(summary.get('segment/motion_mpjpe_repaired_clean', 0.0)):.6f} "
            f"mpjpe_noisy={float(summary.get('segment/motion_mpjpe_noisy_clean', 0.0)):.6f} "
            f"vel_err={float(summary.get('segment/motion_vel_error_repaired_clean', 0.0)):.6f} "
            f"acc_err={float(summary.get('segment/motion_acc_error_repaired_clean', 0.0)):.6f} "
            f"delta_se_norm={float(summary.get('segment/motion_delta_se_norm', 0.0)):.6f}"
        ),
        f"  motion_ids: {tuple(summary.get('motion_ids', ()) or ())}",
    ])
    return "\n".join(lines)


def _format_offline_eval_log(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    per_motion = tuple(summary.get("per_motion", ()) or ())
    if per_motion:
        lines.append("[FrontRES Segment Offline Eval / Per Motion]")
        for row in per_motion:
            lines.extend(
                (
                    f"  motion: id={row.get('motion_id', 'unknown')} samples={int(row.get('sample_count', 0))}",
                    (
                        "    result: "
                        f"success={float(row.get('success_rate', 0.0)) * 100.0:.1f}% "
                        f"fall={float(row.get('fall_rate', 0.0)) * 100.0:.1f}% "
                        f"survival={float(row.get('mean_survival_steps', 0.0)):.1f}"
                    ),
                    _format_eval_gain_line(row, indent="    "),
                    (
                        "    motion: "
                        f"mpjpe_repaired={float(row.get('segment/motion_mpjpe_repaired_clean', 0.0)):.6f} "
                        f"mpjpe_noisy={float(row.get('segment/motion_mpjpe_noisy_clean', 0.0)):.6f} "
                        f"vel_err={float(row.get('segment/motion_vel_error_repaired_clean', 0.0)):.6f} "
                        f"acc_err={float(row.get('segment/motion_acc_error_repaired_clean', 0.0)):.6f}"
                    ),
                )
            )
        lines.append("")

    lines.extend(
        (
            "[FrontRES Segment Offline Eval / Mean]",
            (
                "  rollout: "
                f"sample_count={int(summary.get('sample_count', 0))} "
                f"episode_length={float(summary.get('episode_length', 0.0)):.1f} "
                f"survival={float(summary.get('mean_survival_steps', 0.0)):.1f}"
            ),
            (
                "  result: "
                f"success={float(summary.get('success_rate', 0.0)) * 100.0:.1f}% "
                f"fall={float(summary.get('fall_rate', 0.0)) * 100.0:.1f}%"
            ),
            _format_eval_gain_line(summary, indent="  "),
            (
                "  motion: "
                f"mpjpe_repaired={float(summary.get('segment/motion_mpjpe_repaired_clean', 0.0)):.6f} "
                f"mpjpe_noisy={float(summary.get('segment/motion_mpjpe_noisy_clean', 0.0)):.6f} "
                f"vel_err={float(summary.get('segment/motion_vel_error_repaired_clean', 0.0)):.6f} "
                f"acc_err={float(summary.get('segment/motion_acc_error_repaired_clean', 0.0)):.6f} "
                f"delta_se_norm={float(summary.get('segment/motion_delta_se_norm', 0.0)):.6f}"
            ),
        )
    )
    return "\n".join(lines)


def _offline_eval_summary(capture: Any, *, sample_count: int, motion_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    done_any = capture.done_any
    survival = capture.survival_steps
    done = done_any.detach().bool().reshape(-1) if done_any is not None else None
    survival_flat = survival.detach().float().reshape(-1) if survival is not None else None
    gain_result, gain_summary = _capture_eval_gain_summary(capture)
    summary = {
        "episode_length": float(capture.rollout_k),
        "success_rate": float((~done).float().mean().cpu().item()) if done is not None and done.numel() else 0.0,
        "fall_rate": float(done.float().mean().cpu().item()) if done is not None and done.numel() else 0.0,
        "mean_survival_steps": float(survival_flat.mean().cpu().item()) if survival_flat is not None and survival_flat.numel() else 0.0,
        "sample_count": float(sample_count),
    }
    summary.update(gain_summary)
    summary.update(
        motion_quality_summary_to_scalars(
            clean_positions=getattr(capture, "motion_clean_body_pos", None),
            repaired_positions=getattr(capture, "motion_repaired_body_pos", None),
            noisy_positions=getattr(capture, "motion_noisy_body_pos", None),
            delta_se=getattr(capture, "transition_actions", None),
            valid_mask=(~done) if done is not None else None,
        )
    )
    if motion_ids:
        unique_motion_ids = tuple(dict.fromkeys(str(motion_id) for motion_id in motion_ids))
        if len(unique_motion_ids) == 1:
            row = dict(summary)
            row["motion_id"] = unique_motion_ids[0]
            row["sample_count"] = float(sample_count)
            summary["per_motion"] = [row]
        else:
            summary["per_motion"] = _offline_eval_per_motion_summary(
                capture,
                sample_count=sample_count,
                motion_ids=motion_ids,
                gain_result=gain_result,
            )
    return summary


def _offline_eval_perturbation_summary(batch: Any) -> dict[str, Any]:
    families = tuple(
        getattr(batch, "stage3_index_perturbation_family", ())
        or getattr(batch, "perturbation_family", ())
        or tuple(str(getattr(spec, "perturbation_family", "")) for spec in (getattr(batch, "specs", ()) or ()))
    )
    strengths = getattr(batch, "stage3_index_perturbation_strength", None)
    if strengths is None:
        strengths = getattr(batch, "perturbation_strength", None)
    strength_values = _float_values(strengths)
    local_rp_count = sum(1 for family in families if str(family) == "local_rp")
    non_rp_count = sum(1 for family in families if str(family) and str(family) != "local_rp")
    total = max(1, len(families))
    return {
        "perturbation_family_counts": _count_items(families),
        "perturbation_local_rp_frac": float(local_rp_count) / float(total),
        "perturbation_non_rp_frac": float(non_rp_count) / float(total),
        "perturbation_strength_min": min(strength_values) if strength_values else 0.0,
        "perturbation_strength_mean": (sum(strength_values) / float(len(strength_values))) if strength_values else 0.0,
        "perturbation_strength_max": max(strength_values) if strength_values else 0.0,
    }


def _count_items(values: tuple[Any, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _float_values(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        return [float(item) for item in value.detach().float().reshape(-1).cpu().tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return []


def _offline_eval_motion_ids_from_batch(batch: Any, sample_count: int) -> tuple[str, ...]:
    specs = tuple(getattr(batch, "specs", ()) or ())
    motion_ids: list[str] = []
    for spec in specs[: max(0, int(sample_count))]:
        motion_ids.append(str(getattr(spec, "motion_id", "unknown")))
    return tuple(motion_ids)


def _offline_eval_start_frames_from_batch(batch: Any, sample_count: int) -> tuple[int, ...]:
    specs = tuple(getattr(batch, "specs", ()) or ())
    return tuple(int(getattr(spec, "start_frame", -1)) for spec in specs[: max(0, int(sample_count))])


def _offline_eval_per_motion_summary(
    capture: Any,
    *,
    sample_count: int,
    motion_ids: tuple[str, ...],
    gain_result: Any | None,
) -> list[dict[str, Any]]:
    n = min(sample_count, len(motion_ids), max(0, int(capture.n_train)), max(0, int(capture.n_base)))
    if n <= 0 or gain_result is None:
        return []

    done = capture.done_any.detach().bool().reshape(-1) if capture.done_any is not None else None
    survival = capture.survival_steps.detach().float().reshape(-1) if capture.survival_steps is not None else None
    grouped: dict[str, list[dict[str, float]]] = {}
    for index in range(n):
        scalars = motion_quality_summary_to_scalars(
            clean_positions=_slice_first_dim(getattr(capture, "motion_clean_body_pos", None), index),
            repaired_positions=_slice_first_dim(getattr(capture, "motion_repaired_body_pos", None), index),
            noisy_positions=_slice_first_dim(getattr(capture, "motion_noisy_body_pos", None), index),
            delta_se=_slice_first_dim(getattr(capture, "transition_actions", None), index),
        )
        row = {
            "sample_count": 1.0,
            "episode_length": float(capture.rollout_k),
            "success_rate": float((~done[index]).float().cpu().item()) if done is not None and index < int(done.numel()) else 0.0,
            "fall_rate": float(done[index].float().cpu().item()) if done is not None and index < int(done.numel()) else 0.0,
            "mean_survival_steps": (
                float(survival[index].cpu().item()) if survival is not None and index < int(survival.numel()) else 0.0
            ),
        }
        row.update(_gain_result_row_summary(gain_result, index))
        row.update(scalars)
        grouped.setdefault(motion_ids[index], []).append(row)

    summaries: list[dict[str, Any]] = []
    for motion_id, rows in grouped.items():
        keys = rows[0].keys()
        item: dict[str, Any] = {"motion_id": motion_id}
        for key in keys:
            values: list[float] = []
            for row in rows:
                try:
                    numeric = float(row[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    values.append(numeric)
            if values:
                item[key] = sum(values) / float(len(values))
            else:
                item[key] = rows[0].get(key, "UNCONFIRMED")
        item["sample_count"] = float(len(rows))
        summaries.append(item)
    return summaries


def _gain_result_row_summary(result: Any, index: int) -> dict[str, float]:
    row: dict[str, float] = {}
    for component in _EVAL_GAIN_COMPONENTS:
        value = getattr(result, component, None)
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or int(value.numel()) <= index:
            row[f"gain_{component}_mean"] = float("nan")
            continue
        item = value.detach().float().reshape(-1)[index]
        row[f"gain_{component}_mean"] = float(item.cpu().item()) if bool(torch.isfinite(item).item()) else float("nan")
    _add_public_gain_component_aliases(row)
    row.update(
        {
            "gain_source": "FRS-GAIN-v002",
            "gain_total_pos_frac": 1.0 if row["gain_total_mean"] > 0.0 else 0.0,
        }
    )
    return row


def _slice_first_dim(value: Any, index: int) -> Any:
    if value is None or not hasattr(value, "shape") or int(value.shape[0]) <= index:
        return None
    return value[index : index + 1]


class _temporary_eval_detail_silence:
    def __init__(self, runner: Any):
        self.runner = runner
        self.previous = getattr(runner, "_frontres_segment_live_detail_log_enabled", True)
        self.previous_index_reset_trace: list[tuple[Any, bool]] = []

    def __enter__(self) -> None:
        self.runner._frontres_segment_live_detail_log_enabled = False
        for adapter in _index_reset_adapters_for_runner(self.runner):
            self.previous_index_reset_trace.append((adapter, bool(getattr(adapter, "trace", False))))
            adapter.trace = False

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.runner._frontres_segment_live_detail_log_enabled = self.previous
        for adapter, trace in self.previous_index_reset_trace:
            adapter.trace = trace


class _temporary_sampler_sampling_state:
    """Keep periodic evaluation from advancing the training sampler sequence."""

    def __init__(self, sampler: Any):
        self.sampler = sampler
        self.seen = getattr(sampler, "seen", None)
        self.staleness = getattr(sampler, "staleness", None)
        self.seen_snapshot = self.seen.detach().clone() if hasattr(self.seen, "detach") else None
        self.staleness_snapshot = self.staleness.detach().clone() if hasattr(self.staleness, "detach") else None
        generator = getattr(sampler, "generator", None)
        self.generator_state = generator.get_state().clone() if hasattr(generator, "get_state") else None

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.seen_snapshot is not None:
            self.seen.copy_(self.seen_snapshot)
        if self.staleness_snapshot is not None:
            self.staleness.copy_(self.staleness_snapshot)
        generator = getattr(self.sampler, "generator", None)
        if self.generator_state is not None and hasattr(generator, "set_state"):
            generator.set_state(self.generator_state)


class _temporary_runner_segment_eval_state:
    """Install an eval batch for reset/rollout and restore the caller's runner fields."""

    _FIELDS = (
        "_frontres_segment_live_current_sample",
        "_frontres_segment_live_current_batch",
        "_frontres_segment_live_current_reset_request",
        "_frontres_segment_live_current_reset_result",
    )

    def __init__(self, runner: Any, *, sample: Any, batch: Any):
        self.runner = runner
        self.missing = object()
        self.previous = {name: getattr(runner, name, self.missing) for name in self._FIELDS}
        self.sample = sample
        self.batch = batch

    def __enter__(self) -> None:
        self.runner._frontres_segment_live_current_sample = self.sample
        self.runner._frontres_segment_live_current_batch = self.batch
        self.runner._frontres_segment_live_current_reset_request = None
        self.runner._frontres_segment_live_current_reset_result = None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for name, value in self.previous.items():
            if value is self.missing:
                if hasattr(self.runner, name):
                    delattr(self.runner, name)
            else:
                setattr(self.runner, name, value)


def _index_reset_adapters_for_runner(runner: Any) -> tuple[Any, ...]:
    envs = []
    env = getattr(runner, "env", None)
    if env is not None:
        envs.append(env)
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None:
            envs.append(unwrapped)
    adapters = []
    seen = set()
    for item in envs:
        adapter = getattr(item, "_frontres_segment_index_reset_adapter", None)
        if adapter is not None and id(adapter) not in seen and hasattr(adapter, "trace"):
            seen.add(id(adapter))
            adapters.append(adapter)
    return tuple(adapters)


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


def _live_train_status(summary: Mapping[str, Any]) -> str:
    if int(
        summary.get(
            "ppo_trust_region_rejected_count_sum",
            summary.get("ppo_trust_region_rejected_count", 0),
        )
        or 0
    ) > 0:
        return "WARN_TRUST_REGION_REJECTED"
    total_loss = float(summary["ppo_total_loss_mean"])
    actor_loss = float(summary["ppo_actor_loss_mean"])
    approx_kl = float(summary["ppo_approx_kl_mean"])
    clip_frac = float(summary["ppo_clip_frac_mean"])
    if not all(math.isfinite(v) for v in (total_loss, actor_loss, approx_kl, clip_frac)):
        return "BAD_NONFINITE"
    if abs(actor_loss) >= 1000.0 or abs(total_loss) >= 1000.0:
        return "BAD_LOSS_EXPLOSION"
    if clip_frac >= 0.3:
        return "WARN_HIGH_CLIP"
    if approx_kl < -0.001:
        return "WARN_NEG_KL"
    return "OK"


def _print_live_train_summary(
    runner: Any,
    *,
    local_iteration: int,
    num_learning_iterations: int,
    summary: Mapping[str, Any],
) -> None:
    motion_scalars = motion_quality_summary_to_scalars()
    for key in (
        "segment/motion_mpjpe_repaired_clean",
        "segment/motion_mpjpe_noisy_clean",
        "segment/motion_vel_error_repaired_clean",
        "segment/motion_acc_error_repaired_clean",
        "segment/motion_delta_se_norm",
        "segment/motion_delta_z_up_frac",
    ):
        if key in summary:
            motion_scalars[key] = float(summary[key])
    motion_scalars["segment/motion_delta_se_norm"] = float(
        summary.get("motion_delta_se_norm", motion_scalars["segment/motion_delta_se_norm"])
    )
    motion_scalars["segment/motion_delta_z_up_frac"] = float(
        summary.get("motion_delta_z_up_frac", motion_scalars["segment/motion_delta_z_up_frac"])
    )
    print(
        "\n".join(
            (
                "",
                _LOG_SEPARATOR,
                "",
                "[FrontRES Segment Live Train]",
                "  progress: "
                f"absolute_iter={runner.current_learning_iteration} "
                f"local={local_iteration}/{num_learning_iterations} "
                f"updates={int(summary['update_count'])}/{int(summary['update_steps'])} "
                "runner_learn=True",
                "  data: "
                f"valid={int(summary['ppo_valid_count'])} "
                f"valid_frac={_fmt_pct(summary['storage_valid_frac'])} "
                f"train_reward={_fmt_num(summary.get('train_reward_mean', summary['reward_mean']))} "
                f"env_reward={_fmt_num(summary.get('env_reward_mean', summary['reward_mean']))} "
                f"gain_total={_fmt_num(summary.get('gain_total_mean', float('nan')))}",
                "  trial: "
                f"policy={int(summary.get('trial_policy_count', summary.get('ppo_boundary_policy_rows', 0)))} "
                f"search={int(summary.get('trial_search_count', summary.get('ppo_boundary_search_rows', 0)))} "
                f"evidence={int(summary.get('ppo_boundary_evidence_rows', 0))} "
                f"ppo_valid={int(summary.get('ppo_boundary_eligible_rows', summary.get('ppo_valid_count', 0)))} "
                f"search_evidence_only={int(summary.get('ppo_boundary_search_evidence_only_rows', 0))} "
                f"policy_invalid={int(summary.get('ppo_boundary_policy_invalid_rows', 0))} "
                f"valid_policy={_fmt_pct(summary.get('ppo_boundary_valid_policy_frac', 0.0))} "
                f"valid_evidence={_fmt_pct(summary.get('ppo_boundary_valid_evidence_frac', 0.0))}",
                "  sampler: "
                f"gain={_fmt_num(summary.get('sampler_update_gain_mean', 0.0))} "
                f"gain_pos={_fmt_pct(summary.get('sampler_update_gain_pos_frac', 0.0))} "
                f"useful={_fmt_num(summary.get('sampler_update_useful_mean', 0.0))} "
                f"replay_candidates={int(summary.get('sampler_update_replay_candidate_count', 0))} "
                f"priority={_fmt_num(summary.get('sampler_priority_mean', 0.0))} "
                f"pool={int(summary.get('sampler_replay_pool_size', 0))} "
                f"hopeless={_fmt_pct(summary.get('sampler_hopeless_frac', 0.0))}",
                "  ppo: "
                f"phase={summary.get('ppo_warmup_phase', 'joint')} "
                f"phase_iter={int(summary.get('ppo_warmup_phase_iteration', 0))} "
                f"actor_weight={_fmt_num(summary.get('ppo_actor_loss_weight', 1.0))} "
                f"loss_total={_fmt_num(summary['ppo_total_loss_mean'])} "
                f"actor={_fmt_num(summary['ppo_actor_loss_mean'])} "
                f"value={_fmt_num(summary['ppo_value_loss_mean'])} "
                f"kl={_fmt_num(summary['ppo_approx_kl_mean'])} "
                f"clip={_fmt_pct(summary['ppo_clip_frac_mean'])} "
                f"status={_live_train_status(summary)}",
                "  trust: "
                f"accepted={int(summary.get('ppo_trust_region_accepted_min', summary.get('ppo_trust_region_accepted', 1)))} "
                f"rejected={int(summary.get('ppo_trust_region_rejected_count_sum', summary.get('ppo_trust_region_rejected_count', 0)))} "
                f"lr_before={_fmt_num(summary.get('ppo_adaptive_lr_before_first', summary.get('ppo_adaptive_lr_before', 0.0)))} "
                f"lr_after={_fmt_num(summary.get('ppo_adaptive_lr_after_last', summary.get('ppo_adaptive_lr_after', 0.0)))} "
                f"desired_kl={_fmt_num(summary.get('ppo_adaptive_lr_desired_kl_mean', summary.get('ppo_adaptive_lr_desired_kl', 0.0)))} "
                f"schedule={summary.get('ppo_trust_region_schedule', 'unknown')} "
                f"rollback={bool(summary.get('ppo_trust_region_rollback_enabled_min', summary.get('ppo_trust_region_rollback_enabled', 0)))} "
                f"max_retries={int(summary.get('ppo_trust_region_max_retries_max', summary.get('ppo_trust_region_max_retries', 0)))} "
                f"pre_lr_before={_fmt_num(summary.get('ppo_mosaic_pre_step_adaptive_lr_before_first', summary.get('ppo_mosaic_pre_step_adaptive_lr_before', 0.0)))} "
                f"pre_lr_after={_fmt_num(summary.get('ppo_mosaic_pre_step_adaptive_lr_after_last', summary.get('ppo_mosaic_pre_step_adaptive_lr_after', 0.0)))} "
                f"pre_kl={_fmt_num(summary.get('ppo_mosaic_pre_step_adaptive_lr_kl_mean', 0.0))}",
                "  scale: "
                f"adv_top1={_fmt_pct(summary.get('ppo_advantage_abs_top1_frac_mean', 0.0))} "
                f"old_sigma_min={_fmt_num(summary.get('ppo_old_sigma_min', 0.0))} "
                f"sigma_min={_fmt_num(summary.get('ppo_sigma_min', 0.0))} "
                f"post_mean_delta_l2={_fmt_num(summary.get('ppo_post_update_mean_delta_l2_mean', 0.0))} "
                f"post_mean_delta_max={_fmt_num(summary.get('ppo_post_update_mean_delta_max_abs', 0.0))}",
                "",
                format_segment_train_effect_log(dict(summary)),
                "",
                format_segment_motion_quality_log(motion_scalars),
                "",
            )
        ),
        flush=True,
    )


def _maybe_print_periodic_eval(runner: Any, summary: Mapping[str, Any]) -> None:
    # B1: Reject every alternate probe/eval route before formal live training starts.
    # B1: 验证唯一正式 route boundary, 拒绝 probe/eval 入口进入训练循环.
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    if not bool(getattr(boundary, "periodic_eval_enabled", False)):
        return
    interval = max(1, int(getattr(boundary, "periodic_eval_interval", 100)))
    iteration = int(getattr(runner, "current_learning_iteration", 0))
    if iteration <= 0 or iteration % interval != 0:
        return
    eval_hook = getattr(runner, "run_frontres_segment_periodic_eval", None)
    if not callable(eval_hook):
        raise NotImplementedError("periodic eval is enabled, but runner.run_frontres_segment_periodic_eval is missing.")
    eval_summary = eval_hook(iteration=iteration, train_summary=dict(summary))
    print(
        "\n".join(
            (
                "",
                _LOG_SEPARATOR,
                "",
                format_segment_periodic_eval_log(dict(eval_summary)),
                "",
            )
        ),
        flush=True,
    )


def _validate_live_update_summary(summary: Mapping[str, Any]) -> None:
    missing = [key for key in _REQUIRED_SUMMARY_KEYS if key not in summary]
    if missing:
        raise KeyError(f"FrontRES Segment live update summary missing keys: {missing}")


def _read_live_guard_cfg(runner: Any) -> tuple[bool, int, bool]:
    alg = getattr(runner, "alg", None)
    fail_on_invalid = bool(getattr(alg, "frontres_segment_live_fail_on_invalid_update", True))
    min_valid_count = max(0, int(getattr(alg, "frontres_segment_live_min_valid_count", 1)))
    fail_on_nonfinite = bool(getattr(alg, "frontres_segment_live_fail_on_nonfinite", True))
    return fail_on_invalid, min_valid_count, fail_on_nonfinite


def _validate_live_update_values(
    summary: Mapping[str, Any],
    *,
    fail_on_invalid: bool,
    min_valid_count: int,
    fail_on_nonfinite: bool,
) -> None:
    if fail_on_nonfinite:
        for key in _FINITE_SUMMARY_KEYS:
            value = float(summary[key])
            if not math.isfinite(value):
                raise FloatingPointError(f"FrontRES Segment live update produced non-finite {key}: {value}")
    if not fail_on_invalid:
        return
    update_count = int(summary["update_count"])
    valid_count = int(summary["ppo_valid_count"])
    if update_count <= 0:
        raise RuntimeError("FrontRES Segment live update produced update_count=0.")
    if valid_count < min_valid_count:
        raise RuntimeError(
            "FrontRES Segment live update has too few valid PPO samples: "
            f"ppo_valid_count={valid_count}, min_valid_count={min_valid_count}."
        )


def _v015_formal_update_summary(result: Any) -> dict[str, Any]:
    """Project a committed formal result without inventing legacy sampler metrics."""

    ppo = getattr(result, "ppo_result", None)
    if ppo is None:
        raise TypeError("v015 formal training requires a PPO result")
    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise TypeError("v015 formal training requires immutable update diagnostics")
    summary = {
        "transaction_id": str(getattr(result, "transaction_id", "")),
        "policy_snapshot_id": str(getattr(result, "policy_snapshot_id", "")),
        "update_steps": 1,
        "update_count": int(getattr(result, "update_invocation_count", 0)),
        "ppo_valid_count": int(getattr(result, "valid_row_count", 0)),
        "policy_attempt_count": int(getattr(result, "policy_attempt_count", 0)),
        "segment_count": int(getattr(result, "segment_count", 0)),
        "source_count": int(getattr(result, "source_count", 0)),
        "optimizer_step_before": int(getattr(result, "optimizer_step_before", -1)),
        "optimizer_step_after": int(getattr(result, "optimizer_step_after", -1)),
        "optimizer_step_delta": int(getattr(result, "optimizer_step_delta", -1)),
        "training_contract_id": str(diagnostics.get("training_contract_id", "")),
        "gain_contract_id": str(diagnostics.get("gain_contract_id", "")),
        "method_contract_id": str(diagnostics.get("method_contract_id", "")),
        "optimization_contract_id": str(diagnostics.get("optimization_contract_id", "")),
        "scalar_target_id": str(diagnostics.get("scalar_target_id", "")),
        "constraint_schema_id": str(diagnostics.get("constraint_schema_id", "")),
        "projection_schema_id": str(diagnostics.get("projection_schema_id", "")),
        "constraint_projection_status": str(diagnostics.get("constraint_projection_status", "")),
        "constraint_active_families": tuple(diagnostics.get("constraint_active_families", ())),
        "constraint_levels": dict(diagnostics.get("constraint_levels", {})),
        "constraint_gradient_norms": dict(diagnostics.get("constraint_gradient_norms", {})),
        "constraint_directional_derivatives": dict(diagnostics.get("constraint_directional_derivatives", {})),
        "constraint_dual_coefficients": dict(diagnostics.get("constraint_dual_coefficients", {})),
        "constraint_gram": tuple(diagnostics.get("constraint_gram", ())),
        "constraint_intent_directional_derivatives": dict(
            diagnostics.get("constraint_intent_directional_derivatives", {})
        ),
        "constraint_kkt_max_violation": float(diagnostics.get("constraint_kkt_max_violation", float("nan"))),
        "contact_constraint_advantage": tuple(diagnostics.get("contact_constraint_advantage", ())),
        "zmp_constraint_advantage": tuple(diagnostics.get("zmp_constraint_advantage", ())),
        "survival_constraint_advantage": tuple(diagnostics.get("survival_constraint_advantage", ())),
        "training_iteration": int(diagnostics.get("training_iteration", -1)),
        "curriculum_fingerprint": str(diagnostics.get("curriculum_fingerprint", "")),
        "k_stage_index": int(diagnostics.get("k_stage_index", -1)),
        "active_k": int(diagnostics.get("active_k", -1)),
        "k_stage_iteration": int(diagnostics.get("k_stage_iteration", -1)),
        "warmup_phase": str(diagnostics.get("warmup_phase", "")),
        "warmup_phase_iteration": int(diagnostics.get("warmup_phase_iteration", -1)),
        "actor_loss_weight": float(diagnostics.get("actor_loss_weight", float("nan"))),
        "critic_parameter_delta": dict(diagnostics.get("critic_parameter_delta", {})),
        "actor_std_parameter_delta": dict(diagnostics.get("actor_std_parameter_delta", {})),
        "ppo_total_loss_mean": float(ppo.total_loss.detach().cpu().item()),
        "ppo_actor_loss_mean": float(ppo.actor_loss.detach().cpu().item()),
        "ppo_value_loss_mean": float(ppo.value_loss.detach().cpu().item()),
        "ppo_approx_kl_mean": float(ppo.approx_kl),
        "ppo_clip_frac_mean": float(ppo.clip_frac),
        "grouped_motion_count": int(ppo.grouped_motion_count),
        "grouped_segment_count": int(ppo.grouped_segment_count),
        "grouped_attempt_count": int(ppo.grouped_attempt_count),
        "grouped_valid_step_count": int(ppo.grouped_valid_step_count),
        "grouped_motion_mass_shares": tuple(ppo.grouped_motion_mass_shares),
        "grouped_segment_mass_shares": tuple(ppo.grouped_segment_mass_shares),
        "grouped_attempt_mass_shares": tuple(ppo.grouped_attempt_mass_shares),
    }
    summary["v015_transaction_telemetry"] = _v015_sealed_transaction_telemetry(
        result,
        ppo=ppo,
    )
    return summary


def _v015_sealed_transaction_telemetry(result: Any, *, ppo: Any) -> dict[str, Any]:
    """Project immutable v006 objective/constraint reports into read-only live telemetry."""

    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("v015 formal result requires sealed transaction diagnostics")
    reports = diagnostics.get("v006_action_constraint_reports")
    if not isinstance(reports, tuple) or not reports:
        raise RuntimeError("v015 formal result requires immutable v006 action/constraint reports")

    def required_finite(name: str) -> float:
        if name not in diagnostics:
            raise RuntimeError(f"v015 formal result is missing {name} telemetry")
        value = float(diagnostics[name])
        if not math.isfinite(value):
            raise RuntimeError(f"v015 formal result has non-finite {name} telemetry")
        return value

    def required_identity(name: str, expected: str) -> str:
        value = str(diagnostics.get(name, ""))
        if value != expected:
            raise RuntimeError(f"v015 formal result requires {name}={expected!r}, got {value!r}")
        return value

    constraint_families = {"contact", "zmp", "survival"}

    def required_constraint_mapping(name: str, *, exact_families: bool) -> dict[str, float]:
        raw = diagnostics.get(name)
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"v015 formal result is missing {name} telemetry")
        values = {str(key): float(value) for key, value in raw.items()}
        keys = set(values)
        if (exact_families and keys != constraint_families) or (not exact_families and not keys <= constraint_families):
            raise RuntimeError(f"v015 formal result has invalid {name} families: {sorted(keys)}")
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"v015 formal result has non-finite {name} telemetry")
        return values

    method_contract_id = required_identity("method_contract_id", "FRS-METHOD-v016")
    optimization_contract_id = required_identity("optimization_contract_id", "FRS-PPO-v004")
    scalar_target_id = required_identity("scalar_target_id", "paired-intent-minus-repair-v1")
    constraint_schema_id = required_identity(
        "constraint_schema_id", "contact-loaded-phase_zmp-survival-physical-v2"
    )
    projection_schema_id = required_identity(
        "projection_schema_id", "grouped-first-order-constraint-projection-v1"
    )
    projection_status = str(diagnostics.get("constraint_projection_status", ""))
    allowed_projection_status = {
        "INTENT_FEASIBLE",
        "PROJECTED_INTENT",
        "CONSTRAINT_RECOVERY",
        "NO_EMPIRICAL_DIRECTION",
        "NO_COMMON_FIRST_ORDER_DESCENT",
    }
    if projection_status not in allowed_projection_status:
        raise RuntimeError(f"v015 formal result has invalid constraint projection status: {projection_status!r}")
    active_families = tuple(str(value) for value in diagnostics.get("constraint_active_families", ()))
    if len(set(active_families)) != len(active_families) or not set(active_families) <= constraint_families:
        raise RuntimeError(f"v015 formal result has invalid active constraint families: {active_families}")
    constraint_levels = required_constraint_mapping("constraint_levels", exact_families=True)
    constraint_gradient_norms = required_constraint_mapping("constraint_gradient_norms", exact_families=True)
    constraint_directional_derivatives = required_constraint_mapping(
        "constraint_directional_derivatives", exact_families=False
    )
    constraint_intent_directional_derivatives = required_constraint_mapping(
        "constraint_intent_directional_derivatives", exact_families=False
    )
    constraint_dual_coefficients = required_constraint_mapping(
        "constraint_dual_coefficients", exact_families=False
    )
    raw_gram = diagnostics.get("constraint_gram")
    if not isinstance(raw_gram, tuple):
        raise RuntimeError("v015 formal result is missing constraint_gram telemetry")
    constraint_gram = tuple(tuple(float(value) for value in row) for row in raw_gram)
    if any(len(row) != len(constraint_gram) for row in constraint_gram) or not all(
        math.isfinite(value) for row in constraint_gram for value in row
    ):
        raise RuntimeError("v015 formal result has invalid constraint_gram telemetry")
    constraint_kkt_max_violation = required_finite("constraint_kkt_max_violation")
    if not 0.0 <= constraint_kkt_max_violation <= _V015_PROJECTION_TOLERANCE:
        raise RuntimeError(
            "v015 formal result exceeds the checkpoint-v5 constraint projection tolerance: "
            f"kkt={constraint_kkt_max_violation:.9g} tolerance={_V015_PROJECTION_TOLERANCE:.9g}"
        )
    observed_kkt = max((max(0.0, value) for value in constraint_directional_derivatives.values()), default=0.0)
    if abs(observed_kkt - constraint_kkt_max_violation) > _V015_PROJECTION_TOLERANCE:
        raise RuntimeError(
            "v015 formal result has inconsistent constraint KKT telemetry: "
            f"reported={constraint_kkt_max_violation:.9g} observed={observed_kkt:.9g}"
        )

    transaction_id = str(getattr(result, "transaction_id", ""))
    fields: dict[str, list[Any]] = {
        "policy_actions": [],
        "valid_policy_row_mask": [],
        "intent_gain": [],
        "physics_gain": [],
        "repair_cost": [],
        "gain_total": [],
        "policy_values": [],
        "returns": [],
        "raw_advantages": [],
        "contact_constraint": [],
        "zmp_constraint": [],
        "survival_constraint": [],
        "zmp_applicable_repaired": [],
        "zmp_applicable_noisy": [],
        "zmp_constraint_applicable": [],
        "repaired_success": [],
        "noisy_success": [],
        "repaired_survival": [],
        "noisy_survival": [],
        "physics_survival_quality_repaired": [],
        "physics_survival_quality_noisy": [],
        "repaired_zmp_margin": [],
        "noisy_zmp_margin": [],
        "repaired_contact": [],
        "noisy_contact": [],
        "physics_success_gain": [],
        "physics_survival_gain": [],
        "physics_zmp_gain": [],
        "physics_contact_gain": [],
        "intent_quality_repaired": [],
        "intent_quality_noisy": [],
        "physics_admissible_repaired": [],
        "physics_admissible_noisy": [],
        "physics_deficit_repaired": [],
        "physics_deficit_noisy": [],
        "utility_repaired": [],
        "utility_noisy": [],
        "repair_penalty": [],
        "expected_support_steps": [],
        "actual_contact_repaired_steps": [],
        "actual_contact_noisy_steps": [],
        "zmp_margin_repaired_steps": [],
        "zmp_margin_noisy_steps": [],
        "zmp_applicable_steps": [],
        "zmp_applicable_noisy_steps": [],
        "support_transition_steps": [],
        "zmp_step_violation_repaired": [],
        "zmp_step_violation_noisy": [],
        "zmp_argmax_frame_repaired": [],
        "zmp_argmax_frame_noisy": [],
        "zmp_max_violation_repaired": [],
        "zmp_max_violation_noisy": [],
        "zmp_recovery_trajectory_repaired": [],
        "zmp_recovery_trajectory_noisy": [],
        "physics_valid_step_count": [],
        "scenario_ids": [],
        "noisy_segment_hashes": [],
        "x_t_identities": [],
        "horizon_k": [],
    }
    gain_source: str | None = None
    intent_provenance: str | None = None
    intent_source: str | None = None
    valid_gain_total: list[float] = []

    for report in reports:
        validate = getattr(report, "validate", None)
        if not callable(validate):
            raise TypeError("v015 live telemetry requires validated immutable reports")
        validate()
        if str(getattr(report, "transaction_id", "")) != transaction_id:
            raise RuntimeError("v015 live telemetry report has mixed transaction identity")
        if any(bool(getattr(report, name, True)) for name in ("return_feedback", "priority_feedback", "ppo_feedback")):
            raise RuntimeError("v015 live telemetry report cannot feed training state")
        current_gain_source = str(getattr(report, "gain_source", ""))
        current_provenance = str(getattr(report, "intent_q29_provenance", ""))
        current_source = str(getattr(report, "intent_q29_source", ""))
        if gain_source is None:
            gain_source = current_gain_source
            intent_provenance = current_provenance
            intent_source = current_source
        elif (
            current_gain_source != gain_source
            or current_provenance != intent_provenance
            or current_source != intent_source
        ):
            raise RuntimeError("v015 live telemetry reports have mixed Gain or q29 provenance")

        actions = tuple(tuple(float(value) for value in row) for row in report.policy_actions)
        valid = tuple(bool(value) for value in report.valid_policy_row_mask)
        components = {
            "intent_gain": tuple(float(value) for value in report.intent_gain),
            "physics_gain": tuple(float(value) for value in report.physics_gain),
            "repair_cost": tuple(float(value) for value in report.repair_cost),
            "gain_total": tuple(float(value) for value in report.gain_total),
            "policy_values": tuple(float(value) for value in report.policy_values),
            "returns": tuple(float(value) for value in report.returns),
            "raw_advantages": tuple(float(value) for value in report.raw_advantages),
            "contact_constraint": tuple(float(value) for value in report.contact_constraint),
            "zmp_constraint": tuple(float(value) for value in report.zmp_constraint),
            "survival_constraint": tuple(float(value) for value in report.survival_constraint),
            "repaired_success": tuple(float(value) for value in report.repaired_success),
            "noisy_success": tuple(float(value) for value in report.noisy_success),
            "repaired_survival": tuple(float(value) for value in report.repaired_survival),
            "noisy_survival": tuple(float(value) for value in report.noisy_survival),
            "physics_survival_quality_repaired": tuple(
                float(value) for value in report.physics_survival_quality_repaired
            ),
            "physics_survival_quality_noisy": tuple(
                float(value) for value in report.physics_survival_quality_noisy
            ),
            "repaired_zmp_margin": tuple(float(value) for value in report.repaired_zmp_margin),
            "noisy_zmp_margin": tuple(float(value) for value in report.noisy_zmp_margin),
            "repaired_contact": tuple(float(value) for value in report.repaired_contact),
            "noisy_contact": tuple(float(value) for value in report.noisy_contact),
            "physics_success_gain": tuple(float(value) for value in report.physics_success_gain),
            "physics_survival_gain": tuple(float(value) for value in report.physics_survival_gain),
            "physics_zmp_gain": tuple(float(value) for value in report.physics_zmp_gain),
            "physics_contact_gain": tuple(float(value) for value in report.physics_contact_gain),
            "intent_quality_repaired": tuple(float(value) for value in report.intent_quality_repaired),
            "intent_quality_noisy": tuple(float(value) for value in report.intent_quality_noisy),
            "physics_admissible_repaired": tuple(float(value) for value in report.physics_admissible_repaired),
            "physics_admissible_noisy": tuple(float(value) for value in report.physics_admissible_noisy),
            "physics_deficit_repaired": tuple(float(value) for value in report.physics_deficit_repaired),
            "physics_deficit_noisy": tuple(float(value) for value in report.physics_deficit_noisy),
            "utility_repaired": tuple(float(value) for value in report.utility_repaired),
            "utility_noisy": tuple(float(value) for value in report.utility_noisy),
            "repair_penalty": tuple(float(value) for value in report.repair_penalty),
        }
        row_count = len(actions)
        if len(valid) != row_count or any(len(values) != row_count for values in components.values()):
            raise RuntimeError("v015 live telemetry report rows are not aligned")
        fields["policy_actions"].extend(actions)
        fields["valid_policy_row_mask"].extend(valid)
        fields["scenario_ids"].extend(str(value) for value in report.scenario_ids)
        fields["noisy_segment_hashes"].extend(str(value) for value in report.noisy_segment_hashes)
        fields["x_t_identities"].extend(str(value) for value in report.x_t_identities)
        fields["horizon_k"].extend(int(value) for value in report.horizon_k)
        fields["physics_valid_step_count"].extend(int(value) for value in report.physics_valid_step_count)
        fields["zmp_applicable_repaired"].extend(bool(value) for value in report.zmp_applicable_repaired)
        fields["zmp_applicable_noisy"].extend(bool(value) for value in report.zmp_applicable_noisy)
        fields["zmp_constraint_applicable"].extend(bool(value) for value in report.zmp_constraint_applicable)
        fields["expected_support_steps"].extend(report.expected_support_steps)
        fields["actual_contact_repaired_steps"].extend(report.actual_contact_repaired_steps)
        fields["actual_contact_noisy_steps"].extend(report.actual_contact_noisy_steps)
        fields["zmp_margin_repaired_steps"].extend(report.zmp_margin_repaired_steps)
        fields["zmp_margin_noisy_steps"].extend(report.zmp_margin_noisy_steps)
        fields["zmp_applicable_steps"].extend(report.zmp_applicable_steps)
        fields["zmp_applicable_noisy_steps"].extend(report.zmp_applicable_noisy_steps)
        fields["support_transition_steps"].extend(report.support_transition_steps)
        fields["zmp_step_violation_repaired"].extend(report.zmp_step_violation_repaired)
        fields["zmp_step_violation_noisy"].extend(report.zmp_step_violation_noisy)
        fields["zmp_argmax_frame_repaired"].extend(report.zmp_argmax_frame_repaired)
        fields["zmp_argmax_frame_noisy"].extend(report.zmp_argmax_frame_noisy)
        fields["zmp_max_violation_repaired"].extend(report.zmp_max_violation_repaired)
        fields["zmp_max_violation_noisy"].extend(report.zmp_max_violation_noisy)
        fields["zmp_recovery_trajectory_repaired"].extend(report.zmp_recovery_trajectory_repaired)
        fields["zmp_recovery_trajectory_noisy"].extend(report.zmp_recovery_trajectory_noisy)
        for name, values in components.items():
            for is_valid, value in zip(valid, values):
                optional_zmp = name in {"repaired_zmp_margin", "noisy_zmp_margin", "physics_zmp_gain"}
                if is_valid:
                    if not math.isfinite(value) and not optional_zmp:
                        raise RuntimeError(f"v015 live telemetry valid {name} is nonfinite")
                    fields[name].append(value if math.isfinite(value) else None)
                    if name == "gain_total":
                        valid_gain_total.append(value)
                else:
                    if not math.isnan(value):
                        raise RuntimeError(f"v015 live telemetry invalid {name} must remain UNCONFIRMED")
                    fields[name].append(None)

    row_order = tuple(int(value) for value in diagnostics.get("v006_diagnostic_report_row_order", ()))
    flat_row_count = len(fields["policy_actions"])
    if sorted(row_order) != list(range(flat_row_count)):
        raise RuntimeError(
            "v015 live telemetry requires an exact diagnostic-to-PPO row permutation: "
            f"order={row_order} rows={flat_row_count}"
        )
    for name, values in fields.items():
        fields[name] = [values[index] for index in row_order]
    for name in (
        "contact_constraint_advantage",
        "zmp_constraint_advantage",
        "survival_constraint_advantage",
    ):
        raw_values = diagnostics.get(name)
        if not isinstance(raw_values, tuple) or len(raw_values) != flat_row_count:
            raise RuntimeError(f"v015 formal result requires row-aligned sealed {name} telemetry")
        values: list[float | None] = []
        for is_valid, raw_value in zip(fields["valid_policy_row_mask"], raw_values):
            value = float(raw_value)
            if bool(is_valid):
                if not math.isfinite(value):
                    raise RuntimeError(f"v015 formal result has non-finite valid {name} telemetry")
                values.append(value)
            else:
                if not math.isnan(value):
                    raise RuntimeError(f"v015 formal result invalid {name} must remain UNCONFIRMED")
                values.append(None)
        fields[name] = values
    valid_gain_total = [
        float(value)
        for value, is_valid in zip(fields["gain_total"], fields["valid_policy_row_mask"])
        if bool(is_valid)
    ]
    policy_row_count = len(fields["policy_actions"])
    expected_rows = int(getattr(result, "policy_attempt_count", -1))
    if policy_row_count != expected_rows or not valid_gain_total:
        raise RuntimeError(
            "v015 live telemetry row count disagrees with the sealed transaction: "
            f"reports={policy_row_count} expected={expected_rows}"
        )
    prepared_advantages = tuple(float(value) for value in getattr(ppo, "prepared_advantages", ()))
    valid_row_count = sum(bool(value) for value in fields["valid_policy_row_mask"])
    if len(prepared_advantages) != valid_row_count:
        raise RuntimeError(
            "v015 live telemetry requires one PPO-scaled advantage per valid policy row: "
            f"prepared={len(prepared_advantages)} valid={valid_row_count}"
        )
    prepared_iter = iter(prepared_advantages)
    fields["scaled_advantages"] = tuple(
        next(prepared_iter) if bool(is_valid) else None
        for is_valid in fields["valid_policy_row_mask"]
    )
    positive_fraction = sum(value > 0.0 for value in valid_gain_total) / len(valid_gain_total)
    negative_fraction = sum(value < 0.0 for value in valid_gain_total) / len(valid_gain_total)
    valid_actions = [
        row
        for row, is_valid in zip(fields["policy_actions"], fields["valid_policy_row_mask"])
        if is_valid
    ]
    if not valid_actions:
        raise RuntimeError("v015 live telemetry has no valid full-6D policy action")
    action_abs_values = [abs(value) for row in valid_actions for value in row]
    action_l2_values = [math.sqrt(sum(value * value for value in row)) for row in valid_actions]
    gradient_parameter_count = int(diagnostics.get("gradient_parameter_count", -1))
    gradient_nonzero_parameter_count = int(diagnostics.get("gradient_nonzero_parameter_count", -1))
    if gradient_parameter_count <= 0 or not 0 <= gradient_nonzero_parameter_count <= gradient_parameter_count:
        raise RuntimeError(
            "v015 formal result has invalid gradient parameter telemetry: "
            f"nonzero={gradient_nonzero_parameter_count} total={gradient_parameter_count}"
        )
    return {
        "transaction_id": transaction_id,
        "policy_snapshot_id": str(getattr(result, "policy_snapshot_id", "")),
        "gain_source": gain_source,
        "intent_q29_provenance": intent_provenance,
        "intent_q29_source": intent_source,
        **{name: tuple(values) for name, values in fields.items()},
        "policy_row_count": policy_row_count,
        "valid_policy_row_count": len(valid_gain_total),
        "positive_gain_fraction": float(positive_fraction),
        "negative_gain_fraction": float(negative_fraction),
        "harm_fraction": float(negative_fraction),
        "harm_definition": "gain_total<0",
        "action_abs_mean": float(sum(action_abs_values) / len(action_abs_values)),
        "action_abs_max": float(max(action_abs_values)),
        "action_l2_mean": float(sum(action_l2_values) / len(action_l2_values)),
        "return_mean": required_finite("return_mean"),
        "return_min": required_finite("return_min"),
        "return_max": required_finite("return_max"),
        "return_abs_mean": required_finite("return_abs_mean"),
        "advantage_mean": float(ppo.advantage_mean),
        "advantage_min": float(ppo.advantage_min),
        "advantage_max": float(ppo.advantage_max),
        "advantage_abs_mean": float(ppo.advantage_abs_mean),
        "advantage_abs_max": float(ppo.advantage_abs_max),
        "advantage_abs_top1_frac": float(ppo.advantage_abs_top1_frac),
        "advantage_scale": float(ppo.advantage_scale),
        "advantage_sign_flip_count": int(ppo.advantage_sign_flip_count),
        "grouped_reduction_active": bool(ppo.grouped_reduction_active),
        "grouped_transaction_advantage_rms": float(ppo.grouped_transaction_advantage_rms),
        "gradient_pre_clip_norm": required_finite("gradient_pre_clip_norm"),
        "gradient_post_clip_norm": required_finite("gradient_post_clip_norm"),
        "gradient_parameter_count": gradient_parameter_count,
        "gradient_nonzero_parameter_count": gradient_nonzero_parameter_count,
        "grouped_motion_mass_shares": tuple(ppo.grouped_motion_mass_shares),
        "grouped_segment_mass_shares": tuple(ppo.grouped_segment_mass_shares),
        "grouped_attempt_mass_shares": tuple(ppo.grouped_attempt_mass_shares),
        "update_count": int(getattr(result, "update_invocation_count", 0)),
        "optimizer_step_delta": int(getattr(result, "optimizer_step_delta", -1)),
        "training_contract_id": str(diagnostics.get("training_contract_id", "")),
        "gain_contract_id": str(diagnostics.get("gain_contract_id", "")),
        "method_contract_id": method_contract_id,
        "optimization_contract_id": optimization_contract_id,
        "scalar_target_id": scalar_target_id,
        "constraint_schema_id": constraint_schema_id,
        "projection_schema_id": projection_schema_id,
        "constraint_projection_status": projection_status,
        "constraint_active_families": active_families,
        "constraint_levels": constraint_levels,
        "constraint_gradient_norms": constraint_gradient_norms,
        "constraint_directional_derivatives": constraint_directional_derivatives,
        "constraint_dual_coefficients": constraint_dual_coefficients,
        "constraint_gram": constraint_gram,
        "constraint_intent_directional_derivatives": constraint_intent_directional_derivatives,
        "constraint_kkt_max_violation": constraint_kkt_max_violation,
        "training_iteration": int(diagnostics.get("training_iteration", -1)),
        "curriculum_fingerprint": str(diagnostics.get("curriculum_fingerprint", "")),
        "k_stage_index": int(diagnostics.get("k_stage_index", -1)),
        "active_k": int(diagnostics.get("active_k", -1)),
        "k_stage_iteration": int(diagnostics.get("k_stage_iteration", -1)),
        "warmup_phase": str(diagnostics.get("warmup_phase", "")),
        "warmup_phase_iteration": int(diagnostics.get("warmup_phase_iteration", -1)),
        "actor_loss_weight": float(diagnostics.get("actor_loss_weight", float("nan"))),
        "critic_parameter_delta": dict(diagnostics.get("critic_parameter_delta", {})),
        "actor_std_parameter_delta": dict(diagnostics.get("actor_std_parameter_delta", {})),
        "return_feedback": False,
        "priority_feedback": False,
        "ppo_feedback": False,
    }


def _require_v015_committed_result(runner: Any, result: Any) -> dict[str, Any]:
    """Prove exact-one commit before iteration advance or checkpoint save."""

    summary = _v015_formal_update_summary(result)
    if (
        summary["update_count"] != 1
        or summary["optimizer_step_delta"] != 1
        or summary["optimizer_step_after"] != summary["optimizer_step_before"] + 1
        or summary["ppo_valid_count"] <= 0
        or summary["policy_attempt_count"] <= 0
        or summary["grouped_attempt_count"] != summary["policy_attempt_count"]
    ):
        raise RuntimeError(f"v015 formal training result is not one complete grouped update: {summary}")
    state = getattr(runner, "_frontres_v015_checkpoint_transaction_state", None)
    receipt = state.get("receipt") if isinstance(state, Mapping) and state.get("state") == "committed" else None
    if not isinstance(receipt, Mapping):
        raise RuntimeError("v015 formal training requires a committed checkpoint receipt")
    if (
        str(receipt.get("transaction_id", "")) != summary["transaction_id"]
        or int(receipt.get("expected_policy_row_count", -1)) != summary["policy_attempt_count"]
        or int(receipt.get("collected_policy_attempt_count", -1)) != summary["policy_attempt_count"]
        or int(receipt.get("optimizer_step_delta", -1)) != 1
        or int(receipt.get("optimizer_step_before", -1)) != summary["optimizer_step_before"]
        or int(receipt.get("optimizer_step_after", -1)) != summary["optimizer_step_after"]
    ):
        raise RuntimeError("v015 formal training result disagrees with its committed checkpoint receipt")
    telemetry = summary.get("v015_transaction_telemetry")
    if (
        not isinstance(telemetry, Mapping)
        or telemetry.get("training_contract_id") != "FRS-TRAIN-v010"
        or telemetry.get("gain_contract_id") != "FRS-GAIN-v006"
    ):
        raise RuntimeError("v015 formal training requires exact v010/v006 telemetry identity")
    for name in (
        "curriculum_fingerprint",
        "k_stage_index",
        "active_k",
        "k_stage_iteration",
        "training_iteration",
    ):
        if receipt.get(name) != telemetry.get(name):
            raise RuntimeError(f"v015 formal training receipt/telemetry curriculum mismatch for {name}")
    if telemetry.get("warmup_phase") == "critic_only":
        actor_delta = telemetry.get("actor_std_parameter_delta")
        critic_delta = telemetry.get("critic_parameter_delta")
        if (
            not isinstance(actor_delta, Mapping)
            or not isinstance(critic_delta, Mapping)
            or float(actor_delta.get("param_delta_max_abs", float("nan"))) != 0.0
            or not float(critic_delta.get("param_delta_max_abs", 0.0)) > 0.0
        ):
            raise RuntimeError("FRS-TRAIN-v010 critic-only commit requires zero actor/std and nonzero Critic delta")
    return summary


def _print_v015_formal_train_summary(
    runner: Any,
    *,
    local_iteration: int,
    num_learning_iterations: int,
    summary: Mapping[str, Any],
) -> None:
    telemetry = summary.get("v015_transaction_telemetry")
    if not isinstance(telemetry, Mapping):
        raise RuntimeError("v015 formal train summary requires sealed transaction telemetry")
    print(
        "[FrontRES v015 Transaction Telemetry] "
        + json.dumps(dict(telemetry), sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )
    print(
        "[FrontRES v015 Formal Train] "
        f"absolute_iter={runner.current_learning_iteration} "
        f"local={local_iteration}/{num_learning_iterations} "
        f"transaction={summary['transaction_id']} "
        f"segments={summary['segment_count']} attempts={summary['policy_attempt_count']} "
        f"valid={summary['ppo_valid_count']} grouped_attempts={summary['grouped_attempt_count']} "
        f"step_delta={summary['optimizer_step_delta']} committed=1",
        flush=True,
    )


def _path_inside_log_dir(path: str, log_dir: str | None) -> bool:
    if log_dir is None:
        return False
    abs_path = os.path.abspath(path)
    abs_log_dir = os.path.abspath(log_dir)
    try:
        return os.path.commonpath([abs_path, abs_log_dir]) == abs_log_dir
    except ValueError:
        return False


def _print_checkpoint_save_probe(runner: Any, checkpoint_path: str) -> None:
    print(
        "\n".join(
            (
                "",
                _LOG_SEPARATOR,
                "",
                "[FrontRES Segment Live Checkpoint]",
                "  save.status: OK",
                f"  save.path: {checkpoint_path}",
                f"  save.in_log_dir: {_path_inside_log_dir(checkpoint_path, runner.log_dir)}",
                f"  save.iteration: {int(getattr(runner, 'current_learning_iteration', 0))}",
                "  route.runner_learn: True",
                "",
            )
        ),
        flush=True,
    )


def _print_checkpoint_save_failure(runner: Any, checkpoint_path: str, exc: BaseException) -> None:
    print(
        "\n".join(
            (
                "",
                _LOG_SEPARATOR,
                "",
                "[FrontRES Segment Live Checkpoint]",
                "  save.status: FAILED",
                f"  save.path: {checkpoint_path}",
                f"  save.in_log_dir: {_path_inside_log_dir(checkpoint_path, runner.log_dir)}",
                f"  save.iteration: {int(getattr(runner, 'current_learning_iteration', 0))}",
                f"  error.type: {type(exc).__name__}",
                f"  error.message: {str(exc)[:240]}",
                "",
            )
        ),
        flush=True,
    )


def _save_live_checkpoint(
    runner: Any,
    *,
    checkpoint_path: str,
    summary: Mapping[str, Any],
    required: bool,
    expected_v015_transaction_id: str | None = None,
) -> bool:
    if expected_v015_transaction_id is not None:
        state = getattr(runner, "_frontres_v015_checkpoint_transaction_state", None)
        receipt = state.get("receipt") if isinstance(state, Mapping) and state.get("state") == "committed" else None
        if (
            not isinstance(receipt, Mapping)
            or str(receipt.get("transaction_id", "")) != str(expected_v015_transaction_id)
            or int(receipt.get("optimizer_step_delta", -1)) != 1
        ):
            raise RuntimeError("v015 checkpoint trigger requires the matching exact-one committed transaction receipt")
    try:
        runner.save(checkpoint_path)
    except (OSError, RuntimeError) as exc:
        _print_checkpoint_save_failure(runner, checkpoint_path, exc)
        if required:
            raise
        return False
    _print_checkpoint_save_probe(runner, checkpoint_path)
    runner._record_frontres_checkpoint_probe(dict(summary), checkpoint_path)
    return True


def finalize_frontres_v015_local_sentinel_checkpoint(runner: Any, result: Any) -> str:
    """Persist and verify the checkpoint-v5 produced by one local sentinel update."""

    summary = _require_v015_committed_result(runner, result)
    telemetry = summary.get("v015_transaction_telemetry")
    if not isinstance(telemetry, Mapping):
        raise RuntimeError("v015 local sentinel checkpoint requires sealed transaction telemetry")
    existing = getattr(runner, "_frontres_v015_local_sentinel_telemetry", None)
    sealed = existing.get("sealed_transaction_evidence") if isinstance(existing, Mapping) else None
    if not isinstance(sealed, Mapping) or sealed.get("transaction_id") != summary["transaction_id"]:
        raise RuntimeError("v015 local sentinel checkpoint requires the matching final serialized evidence")

    previous_iteration = int(getattr(runner, "current_learning_iteration", -1))
    if previous_iteration < 0 or int(telemetry.get("training_iteration", -1)) != previous_iteration:
        raise RuntimeError("v015 local sentinel checkpoint iteration is not adjacent to its committed update")
    log_dir = str(getattr(runner, "log_dir", "") or "")
    if not log_dir:
        raise RuntimeError("v015 local sentinel checkpoint requires the formal runner log directory")
    next_iteration = previous_iteration + 1
    checkpoint_path = os.path.join(log_dir, f"model_{next_iteration}.pt")
    if os.path.exists(checkpoint_path):
        raise RuntimeError("v015 local sentinel checkpoint refuses to overwrite an existing artifact")
    runner.current_learning_iteration = next_iteration
    try:
        _save_live_checkpoint(
            runner,
            checkpoint_path=checkpoint_path,
            summary=summary,
            required=True,
            expected_v015_transaction_id=summary["transaction_id"],
        )
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        identity = payload.get("frontres_v015_checkpoint_identity") if isinstance(payload, Mapping) else None
        if not isinstance(identity, Mapping):
            raise RuntimeError("v015 local sentinel checkpoint has no checkpoint-v5 identity")
        required_identity = {
            "format": "frontres-v015-checkpoint-v5",
            "method_contract_id": "FRS-METHOD-v016",
            "training_contract_id": "FRS-TRAIN-v010",
            "gain_contract_id": "FRS-GAIN-v006",
            "optimization_contract_id": "FRS-PPO-v004",
            "scalar_target_id": "paired-intent-minus-repair-v1",
            "constraint_schema_id": "contact-loaded-phase_zmp-survival-physical-v2",
            "projection_schema_id": "grouped-first-order-constraint-projection-v1",
        }
        if any(identity.get(name) != value for name, value in required_identity.items()):
            raise RuntimeError("v015 local sentinel checkpoint contract identity drifted after serialization")
        physics_evidence = identity.get("physics_evidence")
        expected_physics = {
            "zmp_estimator_id": "contact-wrench-zmp-v1",
            "support_envelope_id": "clean-foot-pose-oriented-box-v1",
            "actual_contact_id": "contact-sensor-net-normal-force-threshold-v1",
            "expected_phase_id": "clean-foot-height-phase-v1",
        }
        if physics_evidence != expected_physics:
            raise RuntimeError("v015 local sentinel checkpoint Physics evidence identity drifted after serialization")
        transaction = identity.get("transaction")
        receipt = transaction.get("receipt") if isinstance(transaction, Mapping) else None
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("transaction_id") != summary["transaction_id"]
            or int(receipt.get("optimizer_step_delta", -1)) != 1
        ):
            raise RuntimeError("v015 local sentinel checkpoint lost its exact-one committed receipt")
        curriculum = identity.get("curriculum")
        if not isinstance(curriculum, Mapping) or int(curriculum.get("absolute_iteration", -1)) != next_iteration:
            raise RuntimeError("v015 local sentinel checkpoint lost its absolute iteration identity")
    except Exception:
        runner.current_learning_iteration = previous_iteration
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        raise

    checkpoint_evidence = {
        "path": checkpoint_path,
        "iteration": next_iteration,
        **required_identity,
        "physics_evidence": dict(physics_evidence),
        "transaction_id": summary["transaction_id"],
        "optimizer_step_delta": 1,
    }
    updated = dict(existing)
    updated["checkpoint_v5"] = checkpoint_evidence
    runner._frontres_v015_local_sentinel_telemetry = updated
    print(
        "[FrontRES v015 Checkpoint Sentinel] "
        + json.dumps(checkpoint_evidence, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return checkpoint_path


def _print_resume_probe(runner: Any) -> None:
    loaded_checkpoint_path = getattr(runner, "_frontres_last_loaded_checkpoint_path", None)
    if loaded_checkpoint_path is None:
        return
    print(
        "[FrontRES Segment Live Resume] "
        f"loaded_checkpoint_path={loaded_checkpoint_path} "
        f"resumed_iteration={int(getattr(runner, 'current_learning_iteration', 0))} "
        "runner_learn=True "
        "legacy_runner_learn=False",
        flush=True,
    )


def run_frontres_segment_live_training_loop(
    runner: Any,
    *,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = True,
) -> None:
    """运行正式 Stage 3 Segment Replay 训练迭代.

    函数名说明:
        `run_frontres_segment_live_training_loop` 是正式训练 route owner, 负责把
        已完成配置的 runner 送入重复的 live update; 它不是单次 rollout,
        eval 或离线 probe 入口.

    主链路:
        上游: `train.py` 在 `MODE=train` 且 live train 开启时调用本函数.
        下游: v015 每轮调用 formal transaction owner; legacy 配置才调用旧
        update loop. 只有 committed exact-one result 可递增 iteration 或保存.

    语义:
        进入本函数意味着正式 Stage 3 路由已唯一确定. 任何 probe/eval 路径
        都不得伪装成该训练循环.
    """
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    if not bool(getattr(boundary, "live_train_enabled", False)):
        raise ValueError("FrontRES Segment live training requires frontres_segment_live_train_enabled=True.")
    formal_v015 = bool(getattr(getattr(runner, "alg", None), "frontres_v015_formal_transaction_enabled", False))
    if formal_v015 and bool(getattr(boundary, "periodic_eval_enabled", False)):
        raise RuntimeError("v015 formal training rejects the legacy periodic evaluator")

    # B2: 冻结 live update loop 将消费的正式 iteration budget.
    num_learning_iterations = max(0, int(num_learning_iterations))
    # B3: 首次正式 update iteration 前截获 route identity.
    # AUDIT-ROUTE-01: 检查正式 Stage 3 路由, 位于 train dispatch -> live iteration loop.
    # Result: E70 LIVE PASS. 正式路径从 model_200 进入 absolute iter 700 joint
    # phase, actor_weight=1.0, 完成 4/4 accepted update 并保存 model_701;
    # GMT 仍冻结. 该证据只关闭 runtime connectivity, 不证明 policy quality.
    print_formal_route_audit(runner, num_learning_iterations=num_learning_iterations)
    if num_learning_iterations == 0:
        print(
            "[FrontRES Segment Live Train] "
            "num_learning_iterations=0 update_count=0 runner_learn=True",
            flush=True,
        )
        return

    _print_resume_probe(runner)
    last_checkpoint_probe_path: str | None = None

    for local_iteration in range(num_learning_iterations):
        if formal_v015:
            result = runner.run_frontres_v015_formal_training_transaction(
                init_at_random_ep_len=bool(init_at_random_ep_len and local_iteration == 0),
            )
            summary = _require_v015_committed_result(runner, result)
        else:
            summary = runner.run_frontres_segment_live_update_loop(
                init_at_random_ep_len=bool(init_at_random_ep_len and local_iteration == 0),
                runner_learn=True,
            )
            _validate_live_update_summary(summary)
            fail_on_invalid, min_valid_count, fail_on_nonfinite = _read_live_guard_cfg(runner)
            _validate_live_update_values(
                summary,
                fail_on_invalid=fail_on_invalid,
                min_valid_count=min_valid_count,
                fail_on_nonfinite=fail_on_nonfinite,
            )
        runner.current_learning_iteration += 1
        if formal_v015:
            _print_v015_formal_train_summary(
                runner,
                local_iteration=local_iteration + 1,
                num_learning_iterations=num_learning_iterations,
                summary=summary,
            )
        else:
            _print_live_train_summary(
                runner,
                local_iteration=local_iteration + 1,
                num_learning_iterations=num_learning_iterations,
                summary=summary,
            )
            _maybe_print_periodic_eval(runner, summary)
        if (
            runner.log_dir is not None
            and not runner.disable_logs
            and runner.save_interval > 0
            and runner.current_learning_iteration % runner.save_interval == 0
        ):
            checkpoint_path = os.path.join(runner.log_dir, f"model_{runner.current_learning_iteration}.pt")
            if _save_live_checkpoint(
                runner,
                checkpoint_path=checkpoint_path,
                summary=summary,
                required=False,
                expected_v015_transaction_id=(summary["transaction_id"] if formal_v015 else None),
            ):
                last_checkpoint_probe_path = checkpoint_path

    if runner.log_dir is not None and not runner.disable_logs:
        final_checkpoint_path = os.path.join(runner.log_dir, f"model_{runner.current_learning_iteration}.pt")
        if final_checkpoint_path != last_checkpoint_probe_path:
            _save_live_checkpoint(
                runner,
                checkpoint_path=final_checkpoint_path,
                summary=summary,
                required=True,
                expected_v015_transaction_id=(summary["transaction_id"] if formal_v015 else None),
            )
