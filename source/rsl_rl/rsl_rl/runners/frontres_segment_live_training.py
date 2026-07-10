from __future__ import annotations

import os
import math
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
        _read_live_observations,
        _run_live_rollout_capture,
    )
except ModuleNotFoundError:
    def _apply_current_segment_reset(runner: Any) -> None:
        raise NotImplementedError("frontres_segment_live_probe import is unavailable.")

    def _read_live_observations(runner: Any) -> None:
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


def run_frontres_segment_periodic_eval(
    runner: Any,
    *,
    iteration: int,
    train_summary: Mapping[str, Any],
) -> dict[str, Any]:
    eval_steps = max(
        int(getattr(runner.env, "max_episode_length", 1)),
        int(getattr(runner.alg, "frontres_segment_k", 1)),
    )
    with _temporary_eval_detail_silence(runner):
        _apply_current_segment_reset(runner)
        observations = _read_live_observations(runner)
        runner.eval_mode()
        capture = _run_live_rollout_capture(runner, observations, rollout_steps=eval_steps)
    done_any = capture.done_any
    survival = capture.survival_steps
    if done_any is None or survival is None:
        return {
            "episode_length": float(eval_steps),
            "success_rate": 0.0,
            "fall_rate": 0.0,
            "mean_survival_steps": 0.0,
            "continuous_rollout_gain": float(train_summary.get("score_gain_mean", 0.0)),
        }
    done = done_any.detach().bool().reshape(-1)
    survival_flat = survival.detach().float().reshape(-1)
    summary: dict[str, Any] = {
        "episode_length": float(eval_steps),
        "success_rate": float((~done).float().mean().cpu().item()),
        "fall_rate": float(done.float().mean().cpu().item()),
        "mean_survival_steps": float(survival_flat.mean().cpu().item()),
        "continuous_rollout_gain": float(train_summary.get("score_gain_mean", 0.0)),
    }
    if capture.reward_accum is not None:
        reward = capture.reward_accum.detach().float().reshape(-1)
        n_train = max(0, int(getattr(capture, "n_train", 0)))
        n_base = max(0, int(getattr(capture, "n_base", 0)))
        if n_train > 0 and n_base > 0 and reward.numel() >= n_train + n_base:
            repaired = reward[:n_train]
            noisy = reward[n_train : n_train + n_base]
            pair_count = min(int(repaired.numel()), int(noisy.numel()))
            if pair_count > 0:
                repaired = repaired[:pair_count]
                noisy = noisy[:pair_count]
                gain = repaired - noisy
                summary.update(
                    {
                        "score_noisy": float(noisy.mean().cpu().item()),
                        "score_repaired": float(repaired.mean().cpu().item()),
                        "continuous_rollout_gain": float(gain.mean().cpu().item()),
                    }
                )
    summary.update(
        motion_quality_summary_to_scalars(
            clean_positions=getattr(capture, "motion_clean_body_pos", None),
            repaired_positions=getattr(capture, "motion_repaired_body_pos", None),
            noisy_positions=getattr(capture, "motion_noisy_body_pos", None),
            delta_se=getattr(capture, "transition_actions", None),
            valid_mask=getattr(capture, "actor_update_mask", None),
        )
    )
    return summary


def run_frontres_segment_offline_eval(
    runner: Any,
    *,
    num_eval_segments: int,
    rollout_steps: int,
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    if sampler is None:
        raise RuntimeError("sequence offline eval requires initialized FrontRES Segment sampler")
    import torch

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
    merged = {
        key: sum(float(summary.get(key, 0.0)) for summary in summaries) / float(len(summaries))
        for key in numeric_keys
    }
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
        f"  capture_reward_accum: {_sequence_debug_value(getattr(capture, 'reward_accum', None))}",
        f"  capture_reward_pairs: {_sequence_debug_reward_pairs(capture)}",
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
            f"score_noisy={float(summary.get('score_noisy', 0.0)):.6f} "
            f"score_repaired={float(summary.get('score_repaired', 0.0)):.6f} "
            f"gain={float(summary.get('continuous_rollout_gain', 0.0)):.6f} "
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
    real_score = float(summary.get("score_repaired", 0.0))
    zero_score = float(zero_summary.get("score_repaired", 0.0))
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
                f"score_repaired={real_score:.6f} "
                f"gain={float(summary.get('continuous_rollout_gain', 0.0)):.6f} "
                f"mpjpe_repaired={real_mpjpe:.6f} "
                f"fall={real_fall:.6f} "
                f"survival={real_survival:.6f} "
                f"segment_action_norm={real_action_norm:.6f}"
            ),
            (
                "  zero_policy: "
                f"score_repaired={zero_score:.6f} "
                f"gain={float(zero_summary.get('continuous_rollout_gain', 0.0)):.6f} "
                f"mpjpe_repaired={zero_mpjpe:.6f} "
                f"fall={zero_fall:.6f} "
                f"survival={zero_survival:.6f} "
                f"segment_action_norm={zero_action_norm:.6f}"
            ),
            (
                "  real_minus_zero: "
                f"score={real_score - zero_score:.6f} "
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
    reward_pairs = _sequence_debug_reward_pairs(capture)
    gains = [float(row["gain"]) for row in reward_pairs if "gain" in row]
    raw_norm = _sequence_tensor_l2_mean(getattr(capture, "env_actions", None))
    segment_norm = _sequence_tensor_l2_mean(getattr(capture, "transition_actions", None))
    score_delta = float(summary.get("score_repaired", 0.0)) - float(summary.get("score_noisy", 0.0))
    mpjpe_delta = float(summary.get("segment/motion_mpjpe_repaired_clean", 0.0)) - float(
        summary.get("segment/motion_mpjpe_noisy_clean", 0.0)
    )
    return {
        "raw_action_norm_mean": _round_float(raw_norm),
        "segment_action_norm_mean": _round_float(segment_norm),
        "raw_action_nonzero": raw_norm > 1e-8,
        "segment_action_nonzero": segment_norm > 1e-8,
        "score_repaired_minus_noisy": _round_float(score_delta),
        "mpjpe_repaired_minus_noisy": _round_float(mpjpe_delta),
        "repaired_beats_noisy_mpjpe": mpjpe_delta < 0.0,
        "reward_pair_gain_mean": _round_float(sum(gains) / float(len(gains))) if gains else 0.0,
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


def _sequence_debug_reward_pairs(capture: Any) -> list[dict[str, float | int | bool]]:
    reward = getattr(capture, "reward_accum", None)
    if not hasattr(reward, "detach"):
        return []
    reward_per_step = reward.detach().float().reshape(-1).cpu() / float(max(1, int(getattr(capture, "rollout_k", 1))))
    n_train = max(0, int(getattr(capture, "n_train", 0)))
    n_base = max(0, int(getattr(capture, "n_base", 0)))
    base_start = n_train + max(0, int(getattr(capture, "n_candidate", 0)))
    n = min(n_train, n_base, max(0, int(reward_per_step.numel()) - base_start))
    done = getattr(capture, "done_any", None)
    survival = getattr(capture, "survival_steps", None)
    done_flat = done.detach().bool().reshape(-1).cpu() if hasattr(done, "detach") else None
    survival_flat = survival.detach().float().reshape(-1).cpu() if hasattr(survival, "detach") else None
    rows = []
    for index in range(min(n, 8)):
        repaired = float(reward_per_step[index].item())
        noisy = float(reward_per_step[base_start + index].item())
        rows.append(
            {
                "index": index,
                "repaired": _round_float(repaired),
                "noisy": _round_float(noisy),
                "gain": _round_float(repaired - noisy),
                "done": bool(done_flat[index].item()) if done_flat is not None and index < done_flat.numel() else False,
                "survival": (
                    _round_float(float(survival_flat[index].item()))
                    if survival_flat is not None and index < survival_flat.numel()
                    else 0.0
                ),
            }
        )
    return rows


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
            (
                "  score: "
                f"noisy={float(summary.get('score_noisy', 0.0)):.6f} "
                f"repaired={float(summary.get('score_repaired', 0.0)):.6f} "
                f"gain={float(summary.get('continuous_rollout_gain', 0.0)):.6f}"
            ),
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
                    (
                        "    score: "
                        f"noisy={float(row.get('score_noisy', 0.0)):.6f} "
                        f"repaired={float(row.get('score_repaired', 0.0)):.6f} "
                        f"gain={float(row.get('continuous_rollout_gain', 0.0)):.6f}"
                    ),
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
        (
            "  score: "
            f"noisy={float(summary.get('score_noisy', 0.0)):.6f} "
            f"repaired={float(summary.get('score_repaired', 0.0)):.6f} "
            f"gain={float(summary.get('continuous_rollout_gain', 0.0)):.6f}"
        ),
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
                    (
                        "    score: "
                        f"noisy={float(row.get('score_noisy', 0.0)):.6f} "
                        f"repaired={float(row.get('score_repaired', 0.0)):.6f} "
                        f"gain={float(row.get('continuous_rollout_gain', 0.0)):.6f}"
                    ),
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
            (
                "  score: "
                f"noisy={float(summary.get('score_noisy', 0.0)):.6f} "
                f"repaired={float(summary.get('score_repaired', 0.0)):.6f} "
                f"gain={float(summary.get('continuous_rollout_gain', 0.0)):.6f}"
            ),
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
    score = _offline_eval_score_summary(capture, sample_count=sample_count)
    summary = {
        "episode_length": float(capture.rollout_k),
        "success_rate": float((~done).float().mean().cpu().item()) if done is not None and done.numel() else 0.0,
        "fall_rate": float(done.float().mean().cpu().item()) if done is not None and done.numel() else 0.0,
        "mean_survival_steps": float(survival_flat.mean().cpu().item()) if survival_flat is not None and survival_flat.numel() else 0.0,
        "continuous_rollout_gain": float(score["gain"]),
        "score_noisy": float(score["noisy"]),
        "score_repaired": float(score["repaired"]),
        "sample_count": float(sample_count),
    }
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


def _offline_eval_score_summary(capture: Any, *, sample_count: int) -> dict[str, float]:
    if capture.reward_accum is None:
        return {"noisy": 0.0, "repaired": 0.0, "gain": 0.0}
    reward = capture.reward_accum.reshape(-1).detach().float() / float(max(1, int(capture.rollout_k)))
    n = min(sample_count, max(0, int(capture.n_train)), max(0, int(capture.n_base)))
    base_start = int(capture.n_train) + int(capture.n_candidate)
    if n <= 0 or int(reward.numel()) < base_start + n:
        repaired = float(reward.mean().cpu().item()) if reward.numel() else 0.0
        return {"noisy": 0.0, "repaired": repaired, "gain": repaired}
    repaired = reward[:n]
    noisy = reward[base_start : base_start + n]
    gain = repaired - noisy
    return {
        "noisy": float(noisy.mean().cpu().item()),
        "repaired": float(repaired.mean().cpu().item()),
        "gain": float(gain.mean().cpu().item()),
    }


def _offline_eval_motion_ids_from_batch(batch: Any, sample_count: int) -> tuple[str, ...]:
    specs = tuple(getattr(batch, "specs", ()) or ())
    motion_ids: list[str] = []
    for spec in specs[: max(0, int(sample_count))]:
        motion_ids.append(str(getattr(spec, "motion_id", "unknown")))
    return tuple(motion_ids)


def _offline_eval_per_motion_summary(capture: Any, *, sample_count: int, motion_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    if capture.reward_accum is None:
        return []
    reward = capture.reward_accum.reshape(-1).detach().float() / float(max(1, int(capture.rollout_k)))
    n = min(sample_count, len(motion_ids), max(0, int(capture.n_train)), max(0, int(capture.n_base)))
    base_start = int(capture.n_train) + int(capture.n_candidate)
    if n <= 0 or int(reward.numel()) < base_start + n:
        return []

    done = capture.done_any.detach().bool().reshape(-1) if capture.done_any is not None else None
    survival = capture.survival_steps.detach().float().reshape(-1) if capture.survival_steps is not None else None
    grouped: dict[str, list[dict[str, float]]] = {}
    for index in range(n):
        repaired_score = float(reward[index].cpu().item())
        noisy_score = float(reward[base_start + index].cpu().item())
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
            "score_noisy": noisy_score,
            "score_repaired": repaired_score,
            "continuous_rollout_gain": repaired_score - noisy_score,
        }
        row.update(scalars)
        grouped.setdefault(motion_ids[index], []).append(row)

    summaries: list[dict[str, Any]] = []
    for motion_id, rows in grouped.items():
        keys = rows[0].keys()
        item: dict[str, Any] = {"motion_id": motion_id}
        for key in keys:
            item[key] = sum(float(row.get(key, 0.0)) for row in rows) / float(len(rows))
        item["sample_count"] = float(len(rows))
        summaries.append(item)
    return summaries


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
                f"iter={runner.current_learning_iteration}/{num_learning_iterations} "
                f"updates={int(summary['update_count'])}/{int(summary['update_steps'])} "
                "runner_learn=True",
                "  data: "
                f"valid={int(summary['ppo_valid_count'])} "
                f"valid_frac={_fmt_pct(summary['storage_valid_frac'])} "
                f"train_reward={_fmt_num(summary.get('train_reward_mean', summary['reward_mean']))} "
                f"env_reward={_fmt_num(summary.get('env_reward_mean', summary['reward_mean']))} "
                f"gain={_fmt_num(summary.get('score_gain_mean', 0.0))}",
                "  sampler: "
                f"gain={_fmt_num(summary.get('sampler_update_gain_mean', 0.0))} "
                f"gain_pos={_fmt_pct(summary.get('sampler_update_gain_pos_frac', 0.0))} "
                f"useful={_fmt_num(summary.get('sampler_update_useful_mean', 0.0))} "
                f"replay_candidates={int(summary.get('sampler_update_replay_candidate_count', 0))} "
                f"priority={_fmt_num(summary.get('sampler_priority_mean', 0.0))} "
                f"pool={int(summary.get('sampler_replay_pool_size', 0))} "
                f"hopeless={_fmt_pct(summary.get('sampler_hopeless_frac', 0.0))}",
                "  ppo: "
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
) -> bool:
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
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    if not bool(getattr(boundary, "live_train_enabled", False)):
        raise ValueError("FrontRES Segment live training requires frontres_segment_live_train_enabled=True.")

    num_learning_iterations = max(0, int(num_learning_iterations))
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
        _print_live_train_summary(runner, num_learning_iterations=num_learning_iterations, summary=summary)
        _maybe_print_periodic_eval(runner, summary)
        if (
            runner.log_dir is not None
            and not runner.disable_logs
            and runner.save_interval > 0
            and runner.current_learning_iteration % runner.save_interval == 0
        ):
            checkpoint_path = os.path.join(runner.log_dir, f"model_{runner.current_learning_iteration}.pt")
            if _save_live_checkpoint(runner, checkpoint_path=checkpoint_path, summary=summary, required=False):
                last_checkpoint_probe_path = checkpoint_path

    if runner.log_dir is not None and not runner.disable_logs:
        final_checkpoint_path = os.path.join(runner.log_dir, f"model_{runner.current_learning_iteration}.pt")
        if final_checkpoint_path != last_checkpoint_probe_path:
            _save_live_checkpoint(runner, checkpoint_path=final_checkpoint_path, summary=summary, required=True)
