from __future__ import annotations

from collections import Counter, deque
import copy
from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

import torch

from rsl_rl.algorithms import FrontRESUnified
from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
)
from rsl_rl.frontres.frontres_segment_storage import (
    FrontRESSegmentRolloutStorage,
    FrontRESSegmentTransition,
)
from rsl_rl.frontres.frontres_segment_reset import (
    FrontRESSegmentResetAdapter,
    FrontRESSegmentResetResult,
    ensure_frontres_segment_live_reset_hook,
)
from rsl_rl.frontres.training_schedule import resolve_frontres_mode_state
from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout
from rsl_rl.runners.frontres_rollout_step import prepare_frontres_rollout_step


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


def _fmt_vec(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return "UNCONFIRMED"
    return "[" + ", ".join(_fmt_num(item) for item in value) + "]"


def _mean_sequence(value: Any, default: float = 0.0) -> float:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return float(default)
    return float(sum(float(item) for item in value) / len(value))


def _positive_fraction(value: Any) -> float:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return 0.0
    return sum(1 for item in value if float(item) > 0.0) / float(len(value))


def _shape_last_dim(shape: tuple[int, ...] | None) -> int | None:
    if shape is None or len(shape) == 0:
        return None
    return int(shape[-1])


def _delta_se_norm(actions: torch.Tensor | None) -> float:
    if not isinstance(actions, torch.Tensor) or actions.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(actions.detach().float(), dim=-1).mean().cpu().item())


def _delta_z_up_frac(actions: torch.Tensor | None) -> float:
    if not isinstance(actions, torch.Tensor) or actions.ndim < 2 or actions.shape[-1] < 3:
        return 0.0
    return float((actions.detach()[..., 2] > 0.0).float().mean().cpu().item())


def _probe_status(summary: dict[str, object]) -> str:
    if int(summary.get("ppo_trust_region_rejected_count", 0) or 0) > 0:
        return "WARN_TRUST_REGION_REJECTED"
    total_loss = float(summary.get("ppo_total_loss", 0.0))
    actor_loss = float(summary.get("ppo_actor_loss", 0.0))
    approx_kl = float(summary.get("ppo_approx_kl", 0.0))
    clip_frac = float(summary.get("ppo_clip_frac", 0.0))
    if not all(math.isfinite(v) for v in (total_loss, actor_loss, approx_kl, clip_frac)):
        return "BAD_NONFINITE"
    if abs(actor_loss) >= 1000.0 or abs(total_loss) >= 1000.0:
        return "BAD_LOSS_EXPLOSION"
    if clip_frac >= 0.3:
        return "WARN_HIGH_CLIP"
    if approx_kl < -0.001:
        return "WARN_NEG_KL"
    return "OK"


@dataclass
class FrontRESSegmentLiveObservations:
    obs: torch.Tensor
    privileged_obs: torch.Tensor
    teacher_obs: torch.Tensor
    ref_vel_estimator_obs: torch.Tensor | None


@dataclass
class FrontRESSegmentLiveRolloutCapture:
    rollout_k: int
    reward_mean: float
    done_frac: float
    last_obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...] | None
    env_action_shape: tuple[int, ...] | None
    transition_obs: torch.Tensor | None
    transition_privileged_obs: torch.Tensor | None
    transition_actions: torch.Tensor | None
    transition_log_probs: torch.Tensor | None
    transition_values: torch.Tensor | None
    transition_means: torch.Tensor | None
    transition_sigmas: torch.Tensor | None
    reward_accum: torch.Tensor | None
    done_any: torch.Tensor | None
    reward_steps: torch.Tensor | None = None
    done_steps: torch.Tensor | None = None
    actor_update_mask: torch.Tensor | None = None
    n_train: int = 0
    n_candidate: int = 0
    n_base: int = 0
    n_clean: int = 0
    survival_steps: torch.Tensor | None = None
    motion_clean_body_pos: torch.Tensor | None = None
    motion_repaired_body_pos: torch.Tensor | None = None
    motion_noisy_body_pos: torch.Tensor | None = None
    env_actions: torch.Tensor | None = None
    transition_perturbation_rp: torch.Tensor | None = None
    transition_supervised_target: torch.Tensor | None = None
    max_delta_rpy: float | None = None


def _verbose_probe_enabled(runner: Any, items: Any) -> bool:
    if bool(getattr(getattr(runner, "alg", object()), "frontres_segment_verbose_probe", False)):
        return True
    if isinstance(items, torch.Tensor):
        count = int(items.numel())
    else:
        try:
            count = len(items)
        except TypeError:
            count = int(items)
    return count <= _VERBOSE_PROBE_BATCH_LIMIT


def _id_summary(segment_ids: torch.Tensor) -> str:
    ids = segment_ids.detach().long().reshape(-1).cpu()
    count = int(ids.numel())
    if count == 0:
        return "count=0 id_min=None id_max=None"
    return f"count={count} id_min={int(ids.min().item())} id_max={int(ids.max().item())}"


def _tensor_range_summary(name: str, value: torch.Tensor) -> str:
    data = value.detach().long().reshape(-1).cpu()
    count = int(data.numel())
    if count == 0:
        return f"{name}_count=0 {name}_min=None {name}_max=None"
    return f"{name}_count={count} {name}_min={int(data.min().item())} {name}_max={int(data.max().item())}"


def _tensor_nonzero_frac(value: torch.Tensor) -> float:
    data = value.detach().reshape(-1)
    if int(data.numel()) <= 0:
        return 0.0
    return float((data != 0).float().mean().cpu().item())


def _safe_getattr(owner: Any, name: str) -> Any:
    try:
        return getattr(owner, name)
    except Exception as exc:  # pragma: no cover - diagnostic-only best effort.
        return f"<error {type(exc).__name__}: {exc}>"


def _tensor_debug_summary(name: str, value: Any, *, limit: int = _VERBOSE_PROBE_BATCH_LIMIT) -> str:
    if value is None:
        return f"  {name}: None"
    if not isinstance(value, torch.Tensor):
        return f"  {name}: {value}"
    data = value.detach()
    flat = data.reshape(-1)
    result: dict[str, Any] = {
        "shape": tuple(data.shape),
        "device": str(data.device),
        "dtype": str(data.dtype),
    }
    if int(flat.numel()) <= 0:
        result.update({"numel": 0, "finite": True, "nonzero_frac": "0.0%"})
        return f"  {name}: {result}"
    numeric = flat.float()
    result.update(
        {
            "numel": int(flat.numel()),
            "finite": bool(torch.isfinite(numeric).all().cpu().item()),
            "min": _fmt_num(numeric.min().cpu().item()),
            "max": _fmt_num(numeric.max().cpu().item()),
            "mean": _fmt_num(numeric.mean().cpu().item()),
            "abs_max": _fmt_num(numeric.abs().max().cpu().item()),
            "nonzero_frac": _fmt_pct((flat != 0).float().mean().cpu().item()),
        }
    )
    if int(flat.numel()) <= int(limit):
        result["values"] = flat.cpu().tolist()
    return f"  {name}: {result}"


def _optimizer_parameter_snapshots(policy: Any, optimizer: Any) -> tuple[tuple[str, torch.Tensor], dict[int, torch.Tensor]]:
    names = {id(param): name for name, param in policy.named_parameters()} if hasattr(policy, "named_parameters") else {}
    params: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()
    for group in getattr(optimizer, "param_groups", ()):
        for param in group.get("params", ()):
            if not isinstance(param, torch.Tensor) or id(param) in seen:
                continue
            seen.add(id(param))
            params.append((names.get(id(param), f"param_{len(params)}"), param))
    snapshots = {id(param): param.detach().clone() for _, param in params}
    return tuple(params), snapshots


def _parameter_delta_stats(
    params: tuple[tuple[str, torch.Tensor], ...],
    snapshots: dict[int, torch.Tensor],
) -> dict[str, Any]:
    total = len(params)
    changed = 0
    max_abs = 0.0
    l2_sq = 0.0
    first_changed = ""
    for name, param in params:
        before = snapshots.get(id(param))
        if before is None:
            continue
        delta = (param.detach() - before).float().reshape(-1)
        if int(delta.numel()) <= 0:
            continue
        param_max = float(delta.abs().max().cpu().item())
        if param_max > 0.0:
            changed += 1
            if not first_changed:
                first_changed = name
        max_abs = max(max_abs, param_max)
        l2_sq += float(delta.pow(2).sum().cpu().item())
    return {
        "param_delta_max_abs": max_abs,
        "param_delta_l2": math.sqrt(l2_sq),
        "param_delta_changed": changed,
        "param_delta_total": total,
        "param_delta_first_changed": first_changed,
    }


def _restore_optimizer_parameters(
    params: tuple[tuple[str, torch.Tensor], ...],
    snapshots: dict[int, torch.Tensor],
) -> None:
    for _, param in params:
        before = snapshots.get(id(param))
        if before is not None:
            param.data.copy_(before)


def _set_segment_optimizer_lr(alg: Any, lr: float) -> None:
    optimizer = getattr(alg, "optimizer", None)
    for group in getattr(optimizer, "param_groups", ()) or ():
        group["lr"] = float(lr)
    object.__setattr__(alg, "learning_rate", float(lr))


def _attach_ppo_update_diagnostics(result: Any, diagnostics: dict[str, Any]) -> None:
    for key, value in diagnostics.items():
        object.__setattr__(result, key, value)


def _post_update_segment_ppo_diagnostics(
    policy_adapter: Any,
    ppo_batch: FrontRESSegmentPPOBatch,
    ppo_cfg: FrontRESSegmentPPOConfig,
) -> dict[str, Any]:
    """Re-forward the same batch after optimizer.step and rename diagnostics as post-update.

    Status: active diagnostic boundary, not an optimizer or loss owner.
    Upstream: run_frontres_segment_single_update calls this after optimizer.step.
    Downstream: trust-region rollback, live summary, and PPO probe text consume these fields.
    Evidence: contract-confirmed by frontres_segment_live_single_update_contract.py.
    Gap: only proves same-batch post-step diagnostics, not long-horizon training quality.
    """
    with torch.no_grad():
        post_result = compute_frontres_segment_ppo_loss(policy_adapter, ppo_batch, ppo_cfg)
    post_kl = (
        float(post_result.distribution_kl_mean)
        if bool(post_result.distribution_kl_available)
        else float(post_result.logprob_approx_kl)
    )
    # compute_frontres_segment_ppo_loss names values by local forward timing.
    # Here that local "pre_update" means "before any further update", i.e. the
    # post-step distribution produced by the just-finished optimizer.step.
    post_raw_log_ratio_mean = float(post_result.pre_update_raw_log_ratio_mean)
    post_raw_log_ratio_min = float(post_result.pre_update_raw_log_ratio_min)
    post_raw_log_ratio_max = float(post_result.pre_update_raw_log_ratio_max)
    post_clamped_ratio_mean = float(post_result.pre_update_clamped_ratio_mean)
    post_clamped_ratio_max = float(post_result.pre_update_clamped_ratio_max)
    return {
        "post_update_distribution_kl_mean": float(post_result.distribution_kl_mean),
        "post_update_distribution_kl_available": bool(post_result.distribution_kl_available),
        "post_update_logprob_approx_kl": float(post_result.logprob_approx_kl),
        "post_update_raw_log_ratio_mean": post_raw_log_ratio_mean,
        "post_update_raw_log_ratio_min": post_raw_log_ratio_min,
        "post_update_raw_log_ratio_max": post_raw_log_ratio_max,
        "post_update_clamped_ratio_mean": post_clamped_ratio_mean,
        "post_update_clamped_ratio_max": post_clamped_ratio_max,
        "post_update_ratio_mean": post_clamped_ratio_mean,
        "post_update_ratio_max": post_clamped_ratio_max,
        "post_update_clip_frac": float(post_result.clip_frac),
        "post_update_approx_kl": post_kl,
        "post_update_mean_delta_l2_mean": float(post_result.distribution_mean_delta_l2_mean),
        "post_update_mean_delta_max_abs": float(post_result.distribution_mean_delta_max_abs),
        "post_update_old_sigma_min": float(post_result.old_sigma_min),
        "post_update_sigma_min": float(post_result.sigma_min),
        "post_update_raw_action_old_mean_l2_mean": float(post_result.raw_action_old_mean_l2_mean),
        "post_update_raw_action_old_mean_abs_max": float(post_result.raw_action_old_mean_abs_max),
        "post_update_raw_action_old_mean_abs_dim_mean": tuple(post_result.raw_action_old_mean_abs_dim_mean),
        "post_update_raw_action_old_mean_abs_dim_max": tuple(post_result.raw_action_old_mean_abs_dim_max),
        "post_update_old_sigma_dim_mean": tuple(post_result.old_sigma_dim_mean),
        "post_update_sigma_dim_mean": tuple(post_result.sigma_dim_mean),
        "post_update_distribution_mean_delta_dim_mean": tuple(post_result.distribution_mean_delta_dim_mean),
        "post_update_distribution_mean_delta_abs_dim_max": tuple(
            post_result.distribution_mean_delta_abs_dim_max
        ),
        "post_update_log_ratio_contrib_dim_mean": tuple(post_result.log_ratio_contrib_dim_mean),
        "post_update_log_ratio_contrib_abs_dim_max": tuple(post_result.log_ratio_contrib_abs_dim_max),
        "post_update_log_jacobian_dim_mean": tuple(post_result.log_jacobian_dim_mean),
        "post_update_log_jacobian_abs_dim_max": tuple(post_result.log_jacobian_abs_dim_max),
    }


def _apply_segment_adaptive_learning_rate(
    alg: Any,
    ppo_result: Any,
    *,
    kl_mean: float | None = None,
    allow_increase: bool = True,
) -> dict[str, Any]:
    optimizer = getattr(alg, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    desired_kl = getattr(alg, "desired_kl", None)
    schedule = str(getattr(alg, "schedule", "fixed")).lower()
    min_lr = float(getattr(alg, "frontres_segment_min_learning_rate", 1e-7))
    max_lr = float(getattr(alg, "frontres_segment_max_learning_rate", 1e-2))
    if not param_groups:
        return {
            "adaptive_lr_applied": 0,
            "adaptive_lr_before": 0.0,
            "adaptive_lr_after": 0.0,
            "adaptive_lr_desired_kl": float(desired_kl) if desired_kl is not None else 0.0,
            "adaptive_lr_schedule": schedule,
            "adaptive_lr_allow_increase": int(bool(allow_increase)),
        }
    lr_before = float(getattr(alg, "learning_rate", param_groups[0].get("lr", 0.0)))
    lr_after = lr_before
    if kl_mean is not None:
        kl_mean = float(kl_mean)
    elif bool(getattr(ppo_result, "distribution_kl_available", False)):
        kl_mean = float(getattr(ppo_result, "distribution_kl_mean", 0.0))
    else:
        kl_mean = float(getattr(ppo_result, "approx_kl", 0.0))
    applied = 0
    if desired_kl is not None and schedule == "adaptive" and math.isfinite(kl_mean):
        desired = float(desired_kl)
        if kl_mean > desired * 2.0:
            excess = kl_mean / max(desired * 2.0, 1e-12)
            lr_after = min(max_lr, max(min_lr, lr_before / max(1.5, math.sqrt(excess))))
        elif allow_increase and kl_mean < desired / 2.0 and kl_mean > 0.0:
            lr_after = min(max_lr, lr_before * 1.5)
        applied = int(lr_after != lr_before)
        _set_segment_optimizer_lr(alg, lr_after)
    return {
        "adaptive_lr_applied": applied,
        "adaptive_lr_before": lr_before,
        "adaptive_lr_after": lr_after,
        "adaptive_lr_kl_mean": kl_mean,
        "adaptive_lr_desired_kl": float(desired_kl) if desired_kl is not None else 0.0,
        "adaptive_lr_schedule": schedule,
        "adaptive_lr_min": min_lr,
        "adaptive_lr_max": max_lr,
        "adaptive_lr_allow_increase": int(bool(allow_increase)),
    }


def _family_mask_debug_lines(masks: Any) -> tuple[str, ...]:
    if not isinstance(masks, dict):
        return (f"  dr.family_masks: {masks}",)
    counts: dict[str, int] = {}
    values: dict[str, Any] = {}
    for family, mask in masks.items():
        if isinstance(mask, torch.Tensor):
            bool_mask = mask.detach().bool().reshape(-1)
            counts[str(family)] = int(bool_mask.sum().cpu().item())
            if int(bool_mask.numel()) <= _VERBOSE_PROBE_BATCH_LIMIT:
                values[str(family)] = bool_mask.cpu().tolist()
        else:
            values[str(family)] = mask
    return (
        f"  dr.family_mask_counts: {counts}",
        f"  dr.family_mask_values: {values}",
    )


def _perturber_debug_lines(runner: Any, *, rollout_step: int | None = None) -> tuple[str, ...]:
    command = _motion_command_for_runner(runner)
    if command is None:
        return ("  dr.motion_command: missing",)
    perturber = getattr(command, "perturber", None)
    if perturber is None:
        return ("  dr.perturber: missing",)
    cfg = getattr(perturber, "cfg", None)
    cfg_names = (
        "enable",
        "root_tilt_prob",
        "root_tilt_max_rad",
        "iid_prob_rp",
        "iid_std_rp",
        "iid_prob_xy",
        "iid_std_xy",
        "iid_prob_ya",
        "iid_std_ya",
        "iid_prob_z",
        "iid_std_z",
        "local_root_artifact_prob",
        "local_root_artifact_xy_std",
        "local_root_artifact_yaw_std",
        "iid_temporal_mode",
        "iid_burst_min_steps",
        "iid_burst_max_steps",
    )
    cfg_values = {name: getattr(cfg, name, None) for name in cfg_names} if cfg is not None else None
    lines = [
        f"  dr.rollout_step: {rollout_step if rollout_step is not None else 'n/a'}",
        f"  dr.cfg: {cfg_values}",
        f"  dr.scale_scalar: {_safe_getattr(perturber, '_dr_scale')}",
        _tensor_debug_summary("dr.scale_env", _safe_getattr(perturber, "_dr_scale_env")),
        *_family_mask_debug_lines(_safe_getattr(perturber, "_family_masks")),
    ]
    for name in (
        "_roll_state",
        "_pitch_state",
        "_iid_event_rp",
        "_iid_event_yaw",
        "_iid_event_xy",
        "_iid_event_z",
        "_iid_event_active",
        "_iid_event_start",
        "_artifact_yaw",
        "_artifact_xy",
        "_artifact_steps",
    ):
        lines.append(_tensor_debug_summary(f"dr.{name}", _safe_getattr(perturber, name)))
    for name in (
        "_cached_perturbed_pos",
        "_cached_perturbed_quat",
        "anchor_dr_delta_pos",
        "anchor_dr_delta_quat_correction",
        "_dr_supervised_target",
        "jump_degree",
    ):
        lines.append(_tensor_debug_summary(f"cmd.{name}", _safe_getattr(command, name)))
    return tuple(lines)


def _print_frontres_dr_runtime_probe(runner: Any, *, label: str, rollout_step: int | None = None) -> None:
    return
    # DR runtime diagnostic dump; uncomment when tracing live perturbation state.
    # if not _live_detail_log_enabled(runner):
    #     return
    # print(
    #     _log_block(
    #         "[FrontRES DR Runtime Probe]",
    #         f"  dr.label: {label}",
    #         *_perturber_debug_lines(runner, rollout_step=rollout_step),
    #     ),
    #     flush=True,
    # )


def _count_summary(values: tuple[Any, ...]) -> dict[str, int]:
    return dict(Counter(str(item) for item in values))


def _motion_summary(motion_ids: tuple[str, ...]) -> str:
    if not motion_ids:
        return "motion_count=0 unique_motion_count=0 first_motion=None"
    return (
        f"motion_count={len(motion_ids)} "
        f"unique_motion_count={len(set(motion_ids))} "
        f"first_motion={motion_ids[0]}"
    )


def _sequence_summary(values: Any, *, limit: int = _VERBOSE_PROBE_BATCH_LIMIT) -> Any:
    try:
        count = len(values)
    except TypeError:
        return values
    if count <= limit:
        return list(values)
    first = values[0] if count else None
    last = values[-1] if count else None
    result = {"count": count, "first": first, "last": last}
    if all(isinstance(item, int) for item in values):
        result.update({"min": min(values), "max": max(values)})
    else:
        result["unique_count"] = len(set(values))
    return result


def _verbose_reset_lines(request: Any, *, verbose: bool) -> tuple[str, ...]:
    if not verbose:
        return ()
    segment_ids = request.segment_ids.detach().long().reshape(-1).cpu().tolist()
    return (
        f"  reset.segment_ids: {_sequence_summary(segment_ids)}",
        f"  reset.mode: {_sequence_summary(tuple(request.mode))}",
    )


def _verbose_index_reset_lines(request: Any, *, verbose: bool) -> tuple[str, ...]:
    if not verbose:
        return ()
    segment_ids = request.segment_ids.detach().long().reshape(-1).cpu().tolist()
    start_frames = request.start_frames.detach().long().reshape(-1).cpu().tolist()
    horizon_k = request.horizon_k.detach().long().reshape(-1).cpu().tolist()
    strength = getattr(request, "perturbation_strength", None)
    strength_values = strength.detach().float().reshape(-1).cpu().tolist() if isinstance(strength, torch.Tensor) else ()
    return (
        f"  reset.segment_ids: {_sequence_summary(segment_ids)}",
        f"  reset.motion_ids: {_sequence_summary(tuple(request.motion_ids))}",
        f"  reset.start_frames: {_sequence_summary(start_frames)}",
        f"  reset.horizon_k: {_sequence_summary(horizon_k)}",
        f"  reset.perturbation_family: {_sequence_summary(tuple(getattr(request, 'perturbation_family', ())))}",
        f"  reset.perturbation_strength: {_sequence_summary(strength_values)}",
    )


def _should_print_once_or_verbose(owner: Any, flag_name: str) -> bool:
    if bool(getattr(owner, "frontres_segment_verbose_probe", False)):
        return True
    if bool(getattr(owner, flag_name, False)):
        return False
    setattr(owner, flag_name, True)
    return True


def _live_detail_log_enabled(runner: Any) -> bool:
    alg = getattr(runner, "alg", None)
    if bool(getattr(alg, "frontres_segment_verbose_probe", False)):
        return True
    return bool(getattr(runner, "_frontres_segment_live_detail_log_enabled", True))


class FrontRESSegmentLivePolicyAdapter:
    def __init__(self, alg: FrontRESUnified, privileged_observations: torch.Tensor | None):
        self.alg = alg
        self.privileged_observations = privileged_observations

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if bool(getattr(self.alg, "use_estimate_ref_vel", False)):
            raise NotImplementedError(
                "FrontRES Segment single-update sentinel does not yet store ref_vel_estimator observations."
            )
        self.alg.policy.act(observations)
        value_obs = self.privileged_observations if self.privileged_observations is not None else observations
        if actions.ndim != 2 or actions.shape[-1] != 6:
            raise ValueError(f"Segment PPO policy evaluation requires 6D Delta SE actions, got {tuple(actions.shape)}")
        action_mean = getattr(self.alg.policy, "action_mean", None)
        action_std = getattr(self.alg.policy, "action_std", None)
        mean_6d = None
        std_6d = None
        raw_actions = None
        log_jacobian_contrib = None
        if action_mean is not None and action_mean.ndim == 2 and action_mean.shape[-1] >= 6:
            mean_6d = action_mean[:, :6]
        if action_std is not None and action_std.ndim == 2 and action_std.shape[-1] >= 6:
            std_6d = action_std[:, :6]
        distribution = getattr(self.alg.policy, "distribution", None)
        if (
            distribution is not None
            and hasattr(distribution, "mean")
            and distribution.mean.ndim == 2
            and distribution.mean.shape[-1] >= 6
        ):
            logprob_parts = _segment_delta_se_log_prob_parts(
                self.alg.policy,
                actions,
                distribution.mean,
                distribution.stddev,
            )
            log_prob = logprob_parts["log_prob"]
            raw_actions = logprob_parts["raw_actions"]
            log_jacobian_contrib = logprob_parts["log_jacobian_contrib"]
        else:
            log_prob = _evaluate_segment_delta_se_log_prob(self.alg.policy, actions, alg=self.alg)
        entropy = getattr(self.alg.policy, "entropy", None)
        if callable(entropy):
            entropy = entropy()
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.reshape(-1)
            if entropy.numel() == 1 and actions.shape[0] != 1:
                entropy = entropy.expand(actions.shape[0])
        if _should_print_once_or_verbose(self.alg, "_frontres_segment_ppo_eval_trace_printed"):
            print(
                "[FrontRES Segment PPO Eval Trace] "
                f"batch_action_shape={tuple(actions.shape)} "
                f"policy_action_mean_shape={tuple(action_mean.shape) if action_mean is not None else None} "
                f"eval_mean_shape={tuple(mean_6d.shape) if mean_6d is not None else None} "
                f"log_prob_shape={tuple(log_prob.shape)} "
                "semantic=ppo_eval_uses_6d_delta_se",
                flush=True,
            )
        return {
            "log_prob": log_prob,
            "value": self.alg.policy.evaluate(value_obs).reshape(-1),
            "entropy": entropy if isinstance(entropy, torch.Tensor) else None,
            "mean": mean_6d,
            "sigma": std_6d,
            "raw_actions": raw_actions,
            "log_jacobian_contrib": log_jacobian_contrib,
        }


def _evaluate_segment_delta_se_log_prob(policy: Any, actions: torch.Tensor, *, alg: Any | None = None) -> torch.Tensor:
    distribution = getattr(policy, "distribution", None)
    if (
        distribution is not None
        and hasattr(distribution, "mean")
        and distribution.mean.ndim == 2
        and distribution.mean.shape[-1] >= 6
    ):
        return _evaluate_segment_delta_se_log_prob_from_stats(policy, actions, distribution.mean, distribution.stddev)
    if alg is not None and hasattr(alg, "_get_actor_log_prob"):
        return alg._get_actor_log_prob(actions).reshape(-1)
    if hasattr(policy, "get_actions_log_prob"):
        return policy.get_actions_log_prob(actions).reshape(-1)
    raise TypeError("policy must expose distribution or get_actions_log_prob for Segment PPO evaluation")


def _evaluate_segment_delta_se_log_prob_from_stats(
    policy: Any,
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return _segment_delta_se_log_prob_parts(policy, actions, mean, std)["log_prob"]


def _segment_delta_se_log_prob_parts(
    policy: Any,
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mean_6d = mean[:, :6].to(device=actions.device, dtype=actions.dtype)
    std_6d = std[:, :6].to(device=actions.device, dtype=actions.dtype)
    if int(getattr(policy, "num_task_corrections", 0)) > 0:
        max_delta_pos = float(getattr(policy, "max_delta_pos", 1.0))
        max_delta_rpy = float(getattr(policy, "max_delta_rpy", 1.0))
        max_d = torch.cat(
            [
                torch.full((3,), max_delta_pos, device=actions.device, dtype=actions.dtype),
                torch.full((3,), max_delta_rpy, device=actions.device, dtype=actions.dtype),
            ],
            dim=-1,
        )
        normalized = (actions / max_d).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(normalized)
        log_prob_dim = torch.distributions.Normal(mean_6d, std_6d).log_prob(raw)
        log_j_dim = torch.log(max_d) + torch.log(1.0 - normalized.pow(2) + 1e-6)
        return {
            "log_prob": log_prob_dim.sum(dim=-1) - log_j_dim.sum(dim=-1),
            "raw_actions": raw,
            "log_jacobian_contrib": log_j_dim,
        }
    log_prob_dim = torch.distributions.Normal(mean_6d, std_6d).log_prob(actions)
    return {
        "log_prob": log_prob_dim.sum(dim=-1),
        "raw_actions": actions,
        "log_jacobian_contrib": torch.zeros_like(actions),
    }


def run_frontres_segment_live_probe(runner: Any, init_at_random_ep_len: bool = True) -> dict[str, object]:
    single_update, storage_write = _resolve_probe_modes(runner)
    if init_at_random_ep_len:
        runner.env.episode_length_buf = torch.randint_like(
            runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
        )

    reset_result = _apply_current_segment_reset(runner)
    reset_skip_reason = str(getattr(runner, "_frontres_segment_live_current_reset_skip_reason", "") or "")
    _print_frontres_dr_runtime_probe(runner, label="after_current_segment_reset")
    observations = _read_live_observations(runner)
    runner.eval_mode()
    capture = _run_live_rollout_capture(runner, observations)
    summary = _initial_live_probe_summary(capture, storage_write=storage_write, single_update=single_update)
    _update_trial_metadata_summary(summary, runner, batch_size=_capture_batch_size(capture))
    _update_reset_summary(summary, reset_result, skip_reason=reset_skip_reason)

    if storage_write:
        segment_storage = build_live_segment_storage(runner, capture)
        storage_stats = segment_storage.stats()
        storage_batch = segment_storage.full_batch()
        _update_ppo_boundary_summary(summary, storage_batch.valid_mask)
        train_reward_mean = _valid_reward_mean(storage_batch.returns, storage_batch.valid_mask)
        summary.update(
            {
                "storage_size": storage_stats.size,
                "storage_valid_frac": storage_stats.valid_frac,
                "storage_reward_mean": storage_stats.reward_mean,
                "train_reward_mean": train_reward_mean,
                "storage_reward_per_sample": _float_list(storage_batch.returns),
                "storage_valid_mask_per_sample": _bool_list(storage_batch.valid_mask),
                "storage_segment_ids": _long_list(storage_batch.segment_ids),
            }
        )
        if single_update:
            ppo_result = run_frontres_segment_single_update(runner, storage_batch)
            summary.update(
                {
                    "ppo_update": bool(ppo_result.should_step),
                    "ppo_total_loss": float(ppo_result.total_loss.detach().cpu().item()),
                    "ppo_actor_loss": float(ppo_result.actor_loss.detach().cpu().item()),
                    "ppo_value_loss": float(ppo_result.value_loss.detach().cpu().item()),
                    "ppo_valid_count": int(ppo_result.valid_count),
                    "ppo_approx_kl": float(ppo_result.approx_kl),
                    "ppo_clip_frac": float(ppo_result.clip_frac),
                    "ppo_ratio_mean": float(ppo_result.ratio_mean),
                    "ppo_ratio_max": float(ppo_result.ratio_max),
                    "ppo_old_log_prob_mean": float(ppo_result.old_log_prob_mean),
                    "ppo_new_log_prob_mean": float(ppo_result.new_log_prob_mean),
                    "ppo_raw_log_ratio_mean": float(ppo_result.raw_log_ratio_mean),
                    "ppo_raw_log_ratio_min": float(ppo_result.raw_log_ratio_min),
                    "ppo_raw_log_ratio_max": float(ppo_result.raw_log_ratio_max),
                    "ppo_pre_update_raw_log_ratio_mean": float(
                        ppo_result.pre_update_raw_log_ratio_mean
                    ),
                    "ppo_pre_update_raw_log_ratio_min": float(
                        ppo_result.pre_update_raw_log_ratio_min
                    ),
                    "ppo_pre_update_raw_log_ratio_max": float(
                        ppo_result.pre_update_raw_log_ratio_max
                    ),
                    "ppo_pre_update_clamped_ratio_mean": float(
                        ppo_result.pre_update_clamped_ratio_mean
                    ),
                    "ppo_pre_update_clamped_ratio_max": float(
                        ppo_result.pre_update_clamped_ratio_max
                    ),
                    "ppo_pre_distribution_kl_mean": float(getattr(ppo_result, "distribution_kl_mean", 0.0)),
                    "ppo_pre_logprob_approx_kl": float(getattr(ppo_result, "logprob_approx_kl", 0.0)),
                    "ppo_distribution_kl_available": bool(
                        getattr(ppo_result, "distribution_kl_available", False)
                    ),
                    "ppo_post_update_distribution_kl_mean": float(
                        getattr(ppo_result, "post_update_distribution_kl_mean", 0.0)
                    ),
                    "ppo_post_update_logprob_approx_kl": float(
                        getattr(ppo_result, "post_update_logprob_approx_kl", 0.0)
                    ),
                    "ppo_post_update_ratio_mean": float(
                        getattr(ppo_result, "post_update_ratio_mean", 0.0)
                    ),
                    "ppo_post_update_ratio_max": float(
                        getattr(ppo_result, "post_update_ratio_max", 0.0)
                    ),
                    "ppo_post_update_raw_log_ratio_mean": float(
                        getattr(ppo_result, "post_update_raw_log_ratio_mean", 0.0)
                    ),
                    "ppo_post_update_raw_log_ratio_min": float(
                        getattr(ppo_result, "post_update_raw_log_ratio_min", 0.0)
                    ),
                    "ppo_post_update_raw_log_ratio_max": float(
                        getattr(ppo_result, "post_update_raw_log_ratio_max", 0.0)
                    ),
                    "ppo_post_update_clamped_ratio_mean": float(
                        getattr(ppo_result, "post_update_clamped_ratio_mean", 0.0)
                    ),
                    "ppo_post_update_clamped_ratio_max": float(
                        getattr(ppo_result, "post_update_clamped_ratio_max", 0.0)
                    ),
                    "ppo_post_update_clip_frac": float(
                        getattr(ppo_result, "post_update_clip_frac", 0.0)
                    ),
                    "ppo_advantage_mean": float(ppo_result.advantage_mean),
                    "ppo_advantage_min": float(ppo_result.advantage_min),
                    "ppo_advantage_max": float(ppo_result.advantage_max),
                    "ppo_advantage_abs_mean": float(ppo_result.advantage_abs_mean),
                    "ppo_advantage_abs_max": float(ppo_result.advantage_abs_max),
                    "ppo_advantage_abs_top1_frac": float(ppo_result.advantage_abs_top1_frac),
                    "ppo_distribution_mean_delta_l2_mean": float(
                        ppo_result.distribution_mean_delta_l2_mean
                    ),
                    "ppo_distribution_mean_delta_max_abs": float(
                        ppo_result.distribution_mean_delta_max_abs
                    ),
                    "ppo_old_sigma_min": float(ppo_result.old_sigma_min),
                    "ppo_sigma_min": float(ppo_result.sigma_min),
                    "ppo_param_delta_max_abs": float(getattr(ppo_result, "param_delta_max_abs", 0.0)),
                    "ppo_param_delta_l2": float(getattr(ppo_result, "param_delta_l2", 0.0)),
                    "ppo_param_delta_changed": int(getattr(ppo_result, "param_delta_changed", 0)),
                    "ppo_param_delta_total": int(getattr(ppo_result, "param_delta_total", 0)),
                    "ppo_param_delta_first_changed": str(getattr(ppo_result, "param_delta_first_changed", "")),
                    "ppo_param_grad_norm": float(getattr(ppo_result, "param_grad_norm", 0.0)),
                    "ppo_trust_region_rejected_count": int(
                        getattr(ppo_result, "trust_region_rejected_count", 0)
                    ),
                    "ppo_trust_region_accepted": int(getattr(ppo_result, "trust_region_accepted", 1)),
                    "ppo_trust_region_rollback_enabled": int(
                        getattr(ppo_result, "trust_region_rollback_enabled", 0)
                    ),
                    "ppo_trust_region_max_retries": int(
                        getattr(ppo_result, "trust_region_max_retries", 0)
                    ),
                    "ppo_trust_region_schedule": str(
                        getattr(ppo_result, "trust_region_schedule", "unknown")
                    ),
                    "ppo_trust_region_schedule_adaptive": int(
                        getattr(ppo_result, "trust_region_schedule_adaptive", 0)
                    ),
                    "ppo_adaptive_lr_before": float(getattr(ppo_result, "adaptive_lr_before", 0.0)),
                    "ppo_adaptive_lr_after": float(getattr(ppo_result, "adaptive_lr_after", 0.0)),
                    "ppo_adaptive_lr_kl_mean": float(getattr(ppo_result, "adaptive_lr_kl_mean", 0.0)),
                    "ppo_adaptive_lr_desired_kl": float(getattr(ppo_result, "adaptive_lr_desired_kl", 0.0)),
                    "ppo_mosaic_pre_step_adaptive_lr_before": float(
                        getattr(ppo_result, "mosaic_pre_step_adaptive_lr_before", 0.0)
                    ),
                    "ppo_mosaic_pre_step_adaptive_lr_after": float(
                        getattr(ppo_result, "mosaic_pre_step_adaptive_lr_after", 0.0)
                    ),
                    "ppo_mosaic_pre_step_adaptive_lr_kl_mean": float(
                        getattr(ppo_result, "mosaic_pre_step_adaptive_lr_kl_mean", 0.0)
                    ),
                    "ppo_segment_reject_adaptive_lr_after": float(
                        getattr(ppo_result, "segment_reject_adaptive_lr_after", 0.0)
                    ),
                    "ppo_post_update_mean_delta_l2_mean": float(
                        getattr(ppo_result, "post_update_mean_delta_l2_mean", 0.0)
                    ),
                    "ppo_post_update_mean_delta_max_abs": float(
                        getattr(ppo_result, "post_update_mean_delta_max_abs", 0.0)
                    ),
                    "ppo_post_update_old_sigma_min": float(
                        getattr(ppo_result, "post_update_old_sigma_min", 0.0)
                    ),
                    "ppo_post_update_sigma_min": float(
                        getattr(ppo_result, "post_update_sigma_min", 0.0)
                    ),
                    "ppo_post_update_raw_action_old_mean_l2_mean": float(
                        getattr(ppo_result, "post_update_raw_action_old_mean_l2_mean", 0.0)
                    ),
                    "ppo_post_update_raw_action_old_mean_abs_max": float(
                        getattr(ppo_result, "post_update_raw_action_old_mean_abs_max", 0.0)
                    ),
                    "ppo_post_update_raw_action_old_mean_abs_dim_mean": tuple(
                        getattr(ppo_result, "post_update_raw_action_old_mean_abs_dim_mean", ())
                    ),
                    "ppo_post_update_raw_action_old_mean_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_raw_action_old_mean_abs_dim_max", ())
                    ),
                    "ppo_post_update_old_sigma_dim_mean": tuple(
                        getattr(ppo_result, "post_update_old_sigma_dim_mean", ())
                    ),
                    "ppo_post_update_sigma_dim_mean": tuple(
                        getattr(ppo_result, "post_update_sigma_dim_mean", ())
                    ),
                    "ppo_post_update_distribution_mean_delta_dim_mean": tuple(
                        getattr(ppo_result, "post_update_distribution_mean_delta_dim_mean", ())
                    ),
                    "ppo_post_update_distribution_mean_delta_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_distribution_mean_delta_abs_dim_max", ())
                    ),
                    "ppo_post_update_log_ratio_contrib_dim_mean": tuple(
                        getattr(ppo_result, "post_update_log_ratio_contrib_dim_mean", ())
                    ),
                    "ppo_post_update_log_ratio_contrib_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_log_ratio_contrib_abs_dim_max", ())
                    ),
                    "ppo_post_update_log_jacobian_dim_mean": tuple(
                        getattr(ppo_result, "post_update_log_jacobian_dim_mean", ())
                    ),
                    "ppo_post_update_log_jacobian_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_log_jacobian_abs_dim_max", ())
                    ),
                }
            )
    _print_live_probe_summary(runner, capture, summary)
    return summary


def _apply_current_segment_reset(runner: Any) -> FrontRESSegmentResetResult | None:
    # FRS3-EVAL-013: apply the current index-only reset batch to the live env.
    batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    if batch is None:
        runner._frontres_segment_live_current_reset_skip_reason = "no_current_segment_batch"
        return None
    if _is_index_only_segment_batch(batch):
        return _apply_index_only_segment_reset(runner, batch)
    adapter = getattr(runner, "_frontres_segment_reset_adapter", None)
    if adapter is None:
        adapter = FrontRESSegmentResetAdapter(
            default_preroll_steps=int(getattr(runner.alg, "frontres_segment_preroll_steps", 0)),
            velocity_mismatch_tolerance=float(getattr(runner.alg, "frontres_segment_reset_velocity_tolerance", 1e-3)),
        )
        runner._frontres_segment_reset_adapter = adapter
    reset_mode = str(
        getattr(
            runner.alg,
            "frontres_segment_reset_mode",
            getattr(runner._frontres_segment_replay_boundary, "reset_mode", "auto"),
        )
    ).lower()
    request = adapter.build_request(batch, mode=reset_mode)
    _attach_trial_metadata_to_request(
        request,
        _current_trial_metadata(
            runner,
            batch_size=int(request.segment_ids.numel()),
            device=request.segment_ids.device,
        ),
    )
    if not _env_has_segment_reset_hook(runner.env):
        ensure_frontres_segment_live_reset_hook(
            runner.env,
            robot_name=str(getattr(runner.alg, "frontres_segment_reset_robot_name", "robot")),
            trace=bool(getattr(runner.alg, "frontres_segment_reset_trace", True)),
        )
    result = adapter.apply(runner.env, request)
    runner._frontres_segment_live_current_reset_request = request
    runner._frontres_segment_live_current_reset_result = result
    runner._frontres_segment_live_current_reset_skip_reason = ""
    verbose = _verbose_probe_enabled(runner, request.segment_ids)
    if _live_detail_log_enabled(runner):
        print(
            _log_block(
                "[FrontRES Segment Reset]",
                *_kv_lines(
                    "reset",
                    {
                        "ids": _id_summary(request.segment_ids),
                        "mode_counts": _count_summary(tuple(request.mode)),
                        "valid_count": int(request.valid_mask.detach().bool().sum().cpu().item()),
                        "success_frac": f"{float(result.success_mask.float().mean().detach().cpu().item()):.4f}",
                        "direct_frac": f"{float(result.direct_reset_mask.float().mean().detach().cpu().item()):.4f}",
                        "preroll_frac": f"{float(result.preroll_mask.float().mean().detach().cpu().item()):.4f}",
                        "velocity_mismatch_mean": f"{float(result.velocity_mismatch.float().mean().detach().cpu().item()):.6f}",
                    },
                ),
                *_verbose_reset_lines(request, verbose=verbose),
            ),
            flush=True,
        )
    return result


def _is_index_only_segment_batch(batch: Any) -> bool:
    families = tuple(getattr(batch, "perturbation_family", ()) or ())
    if families:
        return all(str(family) == "index_only" for family in families)
    specs = tuple(getattr(batch, "specs", ()) or ())
    return bool(specs) and all(str(getattr(spec, "perturbation_family", "")) == "index_only" for spec in specs)


def _apply_index_only_segment_reset(runner: Any, batch: Any) -> FrontRESSegmentResetResult | None:
    specs = tuple(getattr(batch, "specs", ()) or ())
    motion_ids = tuple(str(getattr(spec, "motion_id", "")) for spec in specs)
    start_frames = torch.tensor(
        [int(getattr(spec, "start_frame", 0) or 0) for spec in specs],
        dtype=torch.long,
        device=batch.segment_ids.device,
    )
    horizon_k = torch.tensor(
        [int(getattr(spec, "horizon_k", 1) or 1) for spec in specs],
        dtype=torch.long,
        device=batch.segment_ids.device,
    )
    trial_metadata = _current_trial_metadata(
        runner,
        batch_size=int(batch.segment_ids.numel()),
        device=batch.segment_ids.device,
        default_horizon_k=horizon_k,
    )
    horizon_k = trial_metadata.horizon_k
    perturbation_family = tuple(
        getattr(batch, "stage3_index_perturbation_family", ())
        or getattr(batch, "perturbation_family", ())
        or ()
    )
    perturbation_strength = getattr(
        batch,
        "stage3_index_perturbation_strength",
        getattr(batch, "perturbation_strength", None),
    )
    if not isinstance(perturbation_strength, torch.Tensor):
        perturbation_strength = torch.zeros_like(batch.segment_ids, dtype=torch.float32)
    perturbation_strength = perturbation_strength.to(device=batch.segment_ids.device, dtype=torch.float32).reshape(-1)
    request = SimpleNamespace(
        segment_ids=batch.segment_ids,
        motion_ids=motion_ids,
        start_frames=start_frames,
        horizon_k=horizon_k,
        perturbation_family=perturbation_family,
        perturbation_strength=perturbation_strength,
        valid_mask=torch.ones_like(batch.segment_ids, dtype=torch.bool),
    )
    _attach_trial_metadata_to_request(request, trial_metadata)
    hook = _index_segment_reset_hook(runner.env)
    if hook is None:
        runner._frontres_segment_live_current_reset_request = None
        runner._frontres_segment_live_current_reset_result = None
        runner._frontres_segment_live_current_reset_skip_reason = "index_only_segment_index"
        verbose = _verbose_probe_enabled(runner, batch.segment_ids)
        if _live_detail_log_enabled(runner):
            print(
                _log_block(
                    "[FrontRES Segment Reset]",
                    *_kv_lines(
                        "reset",
                        {
                            "skip_reason": "index_only_segment_index",
                            "ids": _id_summary(batch.segment_ids),
                            "motion": _motion_summary(motion_ids),
                            "start": _tensor_range_summary("start", start_frames),
                            "perturbation_family_counts": _count_summary(perturbation_family),
                            "request_strength_nonzero_frac": _fmt_pct(_tensor_nonzero_frac(perturbation_strength)),
                        },
                    ),
                    *_verbose_index_reset_lines(request, verbose=verbose),
                ),
                flush=True,
            )
        return None

    raw_result = hook(request)
    result = _index_reset_result_from_mapping(raw_result, request)
    runner._frontres_segment_live_current_reset_request = request
    runner._frontres_segment_live_current_reset_result = result
    runner._frontres_segment_live_current_reset_skip_reason = ""
    verbose = _verbose_probe_enabled(runner, request.segment_ids)
    if _live_detail_log_enabled(runner):
        print(
            _log_block(
                "[FrontRES Segment Reset]",
                *_kv_lines(
                    "reset",
                    {
                        "mode": "index_only",
                        "ids": _id_summary(request.segment_ids),
                        "motion": _motion_summary(motion_ids),
                        "start": _tensor_range_summary("start", request.start_frames),
                        "horizon": _tensor_range_summary("horizon", request.horizon_k),
                        "perturbation_family_counts": _count_summary(request.perturbation_family),
                        "request_strength_nonzero_frac": _fmt_pct(_tensor_nonzero_frac(request.perturbation_strength)),
                        "success_frac": f"{float(result.success_mask.float().mean().detach().cpu().item()):.4f}",
                    },
                ),
                *_verbose_index_reset_lines(request, verbose=verbose),
            ),
            flush=True,
        )
    return result


def _index_segment_reset_hook(env: Any) -> Any | None:
    for name in ("apply_frontres_segment_index_reset", "reset_to_frontres_segment_index", "set_frontres_segment_index"):
        if hasattr(env, name):
            return getattr(env, name)
    return None


def _index_reset_result_from_mapping(mapping: Any, request: Any) -> FrontRESSegmentResetResult:
    if isinstance(mapping, FrontRESSegmentResetResult):
        return mapping
    if mapping is None:
        mapping = {}
    count = int(request.segment_ids.numel())
    device = request.segment_ids.device
    success = _mapping_bool(mapping, ("success_mask", "reset_success", "valid_mask"), count, device, True)
    fall = _mapping_bool(mapping, ("fall_at_reset_mask", "fall_at_reset", "fall"), count, device, False)
    contact = _mapping_bool(mapping, ("contact_mismatch_mask", "contact_mismatch"), count, device, False)
    velocity = _mapping_float(mapping, ("velocity_mismatch",), count, device, 0.0)
    success = success & (~fall) & (~contact)
    zero = torch.zeros(count, dtype=torch.bool, device=device)
    diagnostics = {
        "reset_success_frac": float(success.float().mean().item()) if count else 0.0,
        "direct_frac": 0.0,
        "preroll_frac": 0.0,
        "invalid_static_frac": 0.0,
        "fall_at_reset_frac": float(fall.float().mean().item()) if count else 0.0,
        "contact_mismatch_frac": float(contact.float().mean().item()) if count else 0.0,
        "velocity_mismatch_mean": float(velocity.float().mean().item()) if count else 0.0,
        "reference_window_applied_frac": 0.0,
    }
    return FrontRESSegmentResetResult(
        success_mask=success,
        direct_reset_mask=zero,
        preroll_mask=zero,
        invalid_static_reset_mask=zero,
        fall_at_reset_mask=fall,
        contact_mismatch_mask=contact,
        velocity_mismatch=velocity,
        diagnostics=diagnostics,
    )


def _mapping_bool(mapping: dict[str, Any], names: tuple[str, ...], count: int, device: torch.device, default: bool) -> torch.Tensor:
    for name in names:
        if name in mapping:
            return mapping[name].to(device=device).bool().flatten()
    return torch.full((count,), default, dtype=torch.bool, device=device)


def _mapping_float(mapping: dict[str, Any], names: tuple[str, ...], count: int, device: torch.device, default: float) -> torch.Tensor:
    for name in names:
        if name in mapping:
            return mapping[name].to(device=device).float().flatten()
    return torch.full((count,), default, dtype=torch.float32, device=device)


def _env_has_segment_reset_hook(env: Any) -> bool:
    return any(hasattr(env, name) for name in ("apply_frontres_segment_reset", "reset_to_segment", "set_segment_state"))


def _update_reset_summary(
    summary: dict[str, object],
    result: FrontRESSegmentResetResult | None,
    *,
    skip_reason: str = "",
) -> None:
    if result is None:
        summary.update(
            {
                "segment_reset": False,
                "segment_reset_skip_reason": skip_reason or "not_requested",
                "segment_reset_success_frac": 0.0,
                "segment_reset_direct_frac": 0.0,
                "segment_reset_preroll_frac": 0.0,
                "segment_reset_invalid_static_frac": 0.0,
                "segment_reset_fall_frac": 0.0,
                "segment_reset_contact_mismatch_frac": 0.0,
                "segment_reset_velocity_mismatch_mean": 0.0,
                "segment_reference_window_applied_frac": 0.0,
            }
        )
        return
    diagnostics = result.diagnostics
    summary.update(
        {
            "segment_reset": True,
            "segment_reset_skip_reason": "",
            "segment_reset_success_frac": float(diagnostics.get("reset_success_frac", 0.0)),
            "segment_reset_direct_frac": float(diagnostics.get("direct_frac", 0.0)),
            "segment_reset_preroll_frac": float(diagnostics.get("preroll_frac", 0.0)),
            "segment_reset_invalid_static_frac": float(diagnostics.get("invalid_static_frac", 0.0)),
            "segment_reset_fall_frac": float(diagnostics.get("fall_at_reset_frac", 0.0)),
            "segment_reset_contact_mismatch_frac": float(diagnostics.get("contact_mismatch_frac", 0.0)),
            "segment_reset_velocity_mismatch_mean": float(diagnostics.get("velocity_mismatch_mean", 0.0)),
            "segment_reference_window_applied_frac": float(diagnostics.get("reference_window_applied_frac", 0.0)),
        }
    )


def _capture_batch_size(capture: FrontRESSegmentLiveRolloutCapture) -> int:
    for value in (capture.transition_actions, capture.reward_accum, capture.done_any):
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            return int(value.shape[0])
    return 0


def _current_trial_metadata(
    runner: Any,
    *,
    batch_size: int,
    device: torch.device | str,
    default_horizon_k: torch.Tensor | None = None,
) -> SimpleNamespace:
    batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    roles = getattr(batch, "frontres_segment_trial_role", None) if batch is not None else None
    if roles is None:
        trial_role = ("policy",) * int(batch_size)
    else:
        trial_role = tuple(str(item) for item in roles)
    if len(trial_role) < int(batch_size):
        trial_role = trial_role + ("baseline",) * (int(batch_size) - len(trial_role))
    if len(trial_role) != int(batch_size):
        raise ValueError(f"frontres_segment_trial_role must have {batch_size} rows, got {len(trial_role)}")

    default_source_index = torch.arange(batch_size, dtype=torch.long, device=device)
    default_trial_index = torch.zeros(batch_size, dtype=torch.long, device=device)
    if default_horizon_k is None:
        alg = getattr(runner, "alg", None)
        default_horizon = int(getattr(alg, "frontres_segment_k", 1) or 1)
        default_horizon_k = torch.full((batch_size,), default_horizon, dtype=torch.long, device=device)

    return SimpleNamespace(
        trial_role=trial_role,
        source_index=_trial_long_vector(
            getattr(batch, "frontres_segment_source_index", None) if batch is not None else None,
            name="frontres_segment_source_index",
            batch_size=batch_size,
            device=device,
            default=default_source_index,
        ),
        trial_index=_trial_long_vector(
            getattr(batch, "frontres_segment_trial_index", None) if batch is not None else None,
            name="frontres_segment_trial_index",
            batch_size=batch_size,
            device=device,
            default=default_trial_index,
        ),
        horizon_k=_trial_long_vector(
            getattr(batch, "frontres_segment_budget_horizon_k", None) if batch is not None else None,
            name="frontres_segment_budget_horizon_k",
            batch_size=batch_size,
            device=device,
            default=default_horizon_k,
        ),
    )


def _trial_long_vector(
    value: Any,
    *,
    name: str,
    batch_size: int,
    device: torch.device | str,
    default: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        tensor = default
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.tensor(list(value), dtype=torch.long)
    tensor = tensor.to(device=device, dtype=torch.long).reshape(-1)
    if int(tensor.numel()) < int(batch_size):
        expanded = default.to(device=device, dtype=torch.long).reshape(-1).detach().clone()
        if int(expanded.numel()) != int(batch_size):
            raise ValueError(f"{name} default must have {batch_size} rows, got {int(expanded.numel())}")
        expanded[: int(tensor.numel())] = tensor
        tensor = expanded
    if int(tensor.numel()) != int(batch_size):
        raise ValueError(f"{name} must have {batch_size} rows, got {int(tensor.numel())}")
    return tensor.detach()


def _attach_trial_metadata_to_request(request: Any, metadata: SimpleNamespace) -> None:
    object.__setattr__(request, "trial_role", metadata.trial_role)
    object.__setattr__(request, "source_index", metadata.source_index)
    object.__setattr__(request, "trial_index", metadata.trial_index)
    object.__setattr__(request, "budget_horizon_k", metadata.horizon_k)


def _update_trial_metadata_summary(
    summary: dict[str, object],
    runner: Any,
    *,
    batch_size: int,
) -> None:
    metadata = _current_trial_metadata(runner, batch_size=batch_size, device=getattr(runner, "device", "cpu"))
    role_counts = dict(Counter(metadata.trial_role))
    policy_count = int(role_counts.get("policy", 0))
    search_count = int(role_counts.get("search", 0))
    evidence_count = policy_count + search_count
    summary.update(
        {
            "trial_role_per_sample": list(metadata.trial_role),
            "trial_source_index_per_sample": _long_list(metadata.source_index),
            "trial_index_per_sample": _long_list(metadata.trial_index),
            "trial_horizon_k_per_sample": _long_list(metadata.horizon_k),
            "trial_role_counts": role_counts,
            "trial_policy_count": policy_count,
            "trial_search_count": search_count,
            "trial_horizon_summary": _tensor_range_summary("horizon", metadata.horizon_k),
            "ppo_boundary_evidence_rows": evidence_count,
            "ppo_boundary_policy_rows": policy_count,
            "ppo_boundary_search_rows": search_count,
            "ppo_boundary_eligible_rows": 0,
            "ppo_boundary_search_evidence_only_rows": search_count,
            "ppo_boundary_policy_invalid_rows": policy_count,
            "ppo_boundary_valid_policy_frac": 0.0,
            "ppo_boundary_valid_evidence_frac": 0.0,
        }
    )


def _update_ppo_boundary_summary(summary: dict[str, object], valid_mask: torch.Tensor) -> None:
    roles = tuple(str(item) for item in summary.get("trial_role_per_sample", ()))
    valid = valid_mask.detach().bool().reshape(-1).cpu()
    if not roles or len(roles) != int(valid.numel()):
        roles = ("policy",) * int(valid.numel())
    policy_mask = torch.tensor([role == "policy" for role in roles], dtype=torch.bool)
    search_mask = torch.tensor([role == "search" for role in roles], dtype=torch.bool)
    evidence_mask = policy_mask | search_mask
    policy_rows = int(policy_mask.sum().item())
    search_rows = int(search_mask.sum().item())
    eligible_rows = int(valid.sum().item())
    policy_invalid_rows = int((policy_mask & ~valid).sum().item())
    evidence_rows = int(evidence_mask.sum().item())
    summary.update(
        {
            "ppo_boundary_evidence_rows": evidence_rows,
            "ppo_boundary_policy_rows": policy_rows,
            "ppo_boundary_search_rows": search_rows,
            "ppo_boundary_eligible_rows": eligible_rows,
            "ppo_boundary_search_evidence_only_rows": search_rows,
            "ppo_boundary_policy_invalid_rows": policy_invalid_rows,
            "ppo_boundary_valid_policy_frac": float(eligible_rows / max(1, policy_rows)),
            "ppo_boundary_valid_evidence_frac": float(eligible_rows / max(1, evidence_rows)),
        }
    )


def _trial_metadata_priority_evidence(runner: Any, *, batch_size: int, device: torch.device | str) -> dict[str, Any]:
    metadata = _current_trial_metadata(runner, batch_size=batch_size, device=device)
    return {
        "trial_role": metadata.trial_role,
        "source_index": metadata.source_index,
        "trial_index": metadata.trial_index,
        "horizon_k": metadata.horizon_k,
    }


def _trial_metadata_ppo_update_mask(runner: Any, *, batch_size: int, device: torch.device | str) -> torch.Tensor:
    metadata = _current_trial_metadata(runner, batch_size=batch_size, device=device)
    return torch.tensor(
        [role == "policy" for role in metadata.trial_role],
        dtype=torch.bool,
        device=device,
    )


def build_live_segment_storage(runner: Any, capture: FrontRESSegmentLiveRolloutCapture) -> FrontRESSegmentRolloutStorage:
    if (
        capture.transition_obs is None
        or capture.transition_privileged_obs is None
        or capture.transition_actions is None
        or capture.transition_log_probs is None
        or capture.transition_values is None
        or capture.reward_accum is None
        or capture.done_any is None
    ):
        raise RuntimeError("FrontRES Segment live storage probe did not capture a valid first-step PPO tuple.")
    if capture.transition_actions.ndim != 2 or capture.transition_actions.shape[-1] != 6:
        raise ValueError(f"live storage probe requires 6D actions, got {tuple(capture.transition_actions.shape)}")

    batch_size = int(capture.transition_actions.shape[0])
    sample = getattr(runner, "_frontres_segment_live_current_sample", None)
    current_batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    sample_ids = getattr(sample, "segment_ids", None)
    sample_source = getattr(sample, "source", None)
    batch_ids = getattr(current_batch, "segment_ids", None)
    if sample_ids is not None and int(sample_ids.numel()) == batch_size:
        segment_ids = sample_ids.to(device=runner.device, dtype=torch.long).reshape(-1)
    elif batch_ids is not None and int(batch_ids.numel()) == batch_size:
        segment_ids = batch_ids.to(device=runner.device, dtype=torch.long).reshape(-1)
    else:
        segment_ids = torch.arange(batch_size, device=runner.device, dtype=torch.long)
    if sample_source is not None and len(sample_source) == batch_size:
        segment_source = tuple(str(item) for item in sample_source)
    else:
        segment_source = ("live_storage_probe",) * batch_size
    reset_mask = _current_reset_success_mask(runner, batch_size=batch_size, device=runner.device)
    rollout_valid_mask = ~capture.done_any.reshape(-1).bool().to(device=runner.device)
    if capture.actor_update_mask is not None:
        actor_update_mask = capture.actor_update_mask.reshape(-1).bool().to(device=runner.device)
        if int(actor_update_mask.numel()) != batch_size:
            raise ValueError(
                f"actor_update_mask must have {batch_size} rows, got {int(actor_update_mask.numel())}"
            )
    else:
        actor_update_mask = torch.ones(batch_size, device=runner.device, dtype=torch.bool)
    ppo_update_mask = _trial_metadata_ppo_update_mask(runner, batch_size=batch_size, device=runner.device)
    valid_mask = rollout_valid_mask & reset_mask & actor_update_mask & ppo_update_mask
    rewards = _segment_storage_rewards(capture, batch_size=batch_size, device=runner.device)
    action_mask = _live_segment_execution_action_mask(runner, capture.transition_actions)
    segment_storage = FrontRESSegmentRolloutStorage(
        capacity=batch_size,
        obs_shape=capture.transition_obs.shape[1:],
        action_dim=6,
        privileged_obs_shape=capture.transition_privileged_obs.shape[1:],
        device=runner.device,
    )
    segment_storage.add_transition(
        FrontRESSegmentTransition(
            observations=capture.transition_obs,
            privileged_observations=capture.transition_privileged_obs,
            actions=capture.transition_actions,
            old_log_probs=capture.transition_log_probs,
            values=capture.transition_values,
            rewards=rewards,
            valid_mask=valid_mask,
            reset_mask=reset_mask,
            segment_ids=segment_ids,
            segment_source=segment_source,
            old_means=capture.transition_means,
            old_sigmas=capture.transition_sigmas,
            action_mask=action_mask,
            priority_evidence=_trial_metadata_priority_evidence(
                runner,
                batch_size=batch_size,
                device=runner.device,
            ),
        )
    )
    reward_steps = _segment_storage_reward_steps(capture, batch_size=batch_size, device=runner.device)
    done_steps = _segment_storage_done_steps(capture, batch_size=batch_size, device=runner.device)
    if reward_steps is not None:
        alg = getattr(runner, "alg", None)
        segment_storage.compute_returns_and_advantages(
            reward_steps=reward_steps,
            done_steps=done_steps,
            horizon=max(1, int(getattr(alg, "frontres_segment_k", capture.rollout_k))),
            gamma=float(getattr(alg, "gamma", 1.0)),
        )
    return segment_storage


def _live_segment_execution_action_mask(runner: Any, actions: torch.Tensor) -> torch.Tensor:
    active_dims = getattr(getattr(runner, "alg", None), "frontres_active_task_dims", None)
    cfg = getattr(runner, "cfg", {}) or {}
    if active_dims is None:
        active_dims = cfg.get("frontres_active_task_dims", cfg.get("α1", None))
    mask = torch.ones_like(actions, dtype=torch.bool)
    if active_dims is None:
        return mask
    mask.zero_()
    proposal_dim = min(6, actions.shape[-1])
    for idx in active_dims:
        idx = int(idx)
        if 0 <= idx < proposal_dim:
            mask[:, idx] = True
    return mask


def _segment_storage_rewards(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    reward = capture.reward_accum.reshape(-1).to(device=device).float() / float(max(1, int(capture.rollout_k)))
    n_train = max(0, int(capture.n_train))
    n_candidate = max(0, int(capture.n_candidate))
    n_base = max(0, int(capture.n_base))
    base_start = n_train + n_candidate
    if n_train > 0 and n_base >= n_train and int(reward.numel()) >= base_start + n_train and batch_size == int(reward.numel()):
        reward = reward.clone()
        reward[:n_train] = reward[:n_train] - reward[base_start : base_start + n_train]
    if int(reward.numel()) != batch_size:
        raise ValueError(f"segment rewards must have {batch_size} rows, got {int(reward.numel())}")
    return reward


def _segment_storage_reward_steps(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if capture.reward_steps is None:
        return None
    reward_steps = capture.reward_steps.to(device=device, dtype=torch.float32)
    if reward_steps.ndim != 2:
        raise ValueError(f"segment reward_steps must be rank-2 [T, B], got {tuple(reward_steps.shape)}")
    if int(reward_steps.shape[1]) != batch_size:
        raise ValueError(f"segment reward_steps must have {batch_size} batch entries, got {int(reward_steps.shape[1])}")

    n_train = max(0, int(capture.n_train))
    n_candidate = max(0, int(capture.n_candidate))
    n_base = max(0, int(capture.n_base))
    base_start = n_train + n_candidate
    if n_train > 0 and n_base >= n_train and batch_size >= base_start + n_train:
        reward_steps = reward_steps.clone()
        reward_steps[:, :n_train] = reward_steps[:, :n_train] - reward_steps[:, base_start : base_start + n_train]
    return reward_steps


def _segment_storage_done_steps(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if capture.done_steps is None:
        return None
    done_steps = capture.done_steps.to(device=device).bool()
    if done_steps.ndim != 2:
        raise ValueError(f"segment done_steps must be rank-2 [T, B], got {tuple(done_steps.shape)}")
    if int(done_steps.shape[1]) != batch_size:
        raise ValueError(f"segment done_steps must have {batch_size} batch entries, got {int(done_steps.shape[1])}")
    return done_steps


def _select_segment_transition_actions(
    runner: Any,
    *,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 2:
        raise ValueError(f"live segment transition actions must be rank-2, got {tuple(actions.shape)}")
    if actions.shape[-1] == 6:
        log_probs = runner.alg.transition.actions_log_prob.detach().clone().reshape(-1)
        if _should_print_once_or_verbose(runner.alg, "_frontres_segment_live_probe_trace_printed"):
            print(
                "[FrontRES Segment Live Probe Trace] "
                f"raw_action_shape={tuple(actions.shape)} "
                f"segment_action_shape={tuple(actions.shape)} "
                f"log_prob_shape={tuple(log_probs.shape)} "
                "semantic=storage_uses_native_6d_delta_se_policy",
                flush=True,
            )
        return actions, log_probs
    if actions.shape[-1] < 6:
        raise ValueError(f"live segment transition actions must expose at least 6 Delta SE dims, got {tuple(actions.shape)}")

    segment_actions = actions[:, :6]
    action_mean = getattr(runner.alg.transition, "action_mean", None)
    action_sigma = getattr(runner.alg.transition, "action_sigma", None)
    if action_mean is not None and action_sigma is not None:
        log_probs = _evaluate_segment_delta_se_log_prob_from_stats(
            runner.alg.policy,
            segment_actions,
            action_mean,
            action_sigma,
        ).detach().clone().reshape(-1)
    elif hasattr(runner.alg.policy, "get_actions_log_prob_selected"):
        log_probs = runner.alg.policy.get_actions_log_prob_selected(actions, list(range(6))).detach().clone().reshape(-1)
    else:
        raise ValueError("12D live segment actions require action_mean/action_sigma to rebuild 6D log_prob.")
    if _should_print_once_or_verbose(runner.alg, "_frontres_segment_live_probe_trace_printed"):
        print(
            "[FrontRES Segment Live Probe Trace] "
            f"raw_action_shape={tuple(actions.shape)} "
            f"segment_action_shape={tuple(segment_actions.shape)} "
            f"log_prob_shape={tuple(log_probs.shape)} "
            "semantic=storage_uses_first_6_delta_se_dims",
            flush=True,
        )
    return segment_actions, log_probs


def _motion_perturber_from_runner(runner: Any) -> Any | None:
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    command_manager = getattr(env_raw, "command_manager", None)
    terms = getattr(command_manager, "_terms", {}) if command_manager is not None else {}
    motion_command = terms.get("motion") if hasattr(terms, "get") else None
    if motion_command is None:
        motion_command = getattr(env_raw, "command", None)
    return getattr(motion_command, "perturber", None)


def _snapshot_frontres_perturbation_rp(runner: Any, *, num_envs: int) -> torch.Tensor | None:
    perturber = _motion_perturber_from_runner(runner)
    roll_state = getattr(perturber, "_roll_state", None)
    pitch_state = getattr(perturber, "_pitch_state", None)
    if not isinstance(roll_state, torch.Tensor) or not isinstance(pitch_state, torch.Tensor):
        return None
    count = max(0, min(int(num_envs), int(roll_state.numel()), int(pitch_state.numel())))
    if count <= 0:
        return None
    rp = torch.stack(
        (
            roll_state[:count].detach().float(),
            pitch_state[:count].detach().float(),
        ),
        dim=-1,
    )
    iid_event_rp = getattr(perturber, "_iid_event_rp", None)
    if isinstance(iid_event_rp, torch.Tensor) and iid_event_rp.ndim == 2 and int(iid_event_rp.shape[0]) >= count:
        rp = rp + iid_event_rp[:count, :2].detach().float()
    family_masks = getattr(perturber, "_family_masks", None)
    if isinstance(family_masks, dict) and isinstance(family_masks.get("local_rp"), torch.Tensor):
        mask = family_masks["local_rp"][:count].to(device=rp.device, dtype=torch.bool)
        rp = rp * mask.to(dtype=rp.dtype).view(-1, 1)
    baseline_mask = getattr(perturber, "_baseline_mask", None)
    if isinstance(baseline_mask, torch.Tensor) and int(baseline_mask.numel()) >= count:
        mask = ~baseline_mask[:count].to(device=rp.device, dtype=torch.bool)
        rp = rp * mask.to(dtype=rp.dtype).view(-1, 1)
    return rp.detach().clone()


def _current_reset_success_mask(runner: Any, *, batch_size: int, device: torch.device | str) -> torch.Tensor:
    result = getattr(runner, "_frontres_segment_live_current_reset_result", None)
    if result is None:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    success_mask = getattr(result, "success_mask", None)
    if success_mask is None:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    success_mask = success_mask.to(device=device).bool().reshape(-1)
    if int(success_mask.numel()) != batch_size:
        raise ValueError(
            f"segment reset success mask must have {batch_size} rows, got {int(success_mask.numel())}"
        )
    return success_mask.detach()


def run_frontres_segment_single_update(runner: Any, storage_batch: Any) -> object:
    """Run one Stage 3 Segment PPO update on the isolated live Segment path.

    Status: active Segment Replay update boundary.
    Upstream: live probe/update loop passes storage_batch from rollout evidence.
    Downstream: FrontRESSegmentPPOBatch -> compute_frontres_segment_ppo_loss -> optimizer.step -> post diagnostics.
    Evidence: contract-confirmed by frontres_segment_live_single_update_contract.py.
    Gap: one fake/live-boundary update does not prove long live training quality.
    """
    runner.train_mode()
    # B1: Convert storage evidence into the algorithm-owned batch contract.
    ppo_batch = storage_batch.to_ppo_batch(FrontRESSegmentPPOBatch)
    policy_adapter = FrontRESSegmentLivePolicyAdapter(
        runner.alg,
        privileged_observations=storage_batch.privileged_observations,
    )
    ppo_cfg = FrontRESSegmentPPOConfig(
        clip_param=float(getattr(runner.alg, "clip_param", 0.2)),
        value_clip_param=float(getattr(runner.alg, "clip_param", 0.2)),
        value_loss_coef=float(getattr(runner.alg, "value_loss_coef", 1.0)),
        entropy_coef=float(getattr(runner.alg, "entropy_coef", 0.0)),
        use_clipped_value_loss=bool(getattr(runner.alg, "use_clipped_value_loss", True)),
        advantage_normalization=str(getattr(runner.alg, "frontres_segment_advantage_normalization", "scale_only")),
    )
    # B2: First forward is the pre-step loss and MOSAIC-style old/new KL source.
    ppo_result = compute_frontres_segment_ppo_loss(policy_adapter, ppo_batch, ppo_cfg)
    pre_step_lr_diagnostics = _apply_segment_adaptive_learning_rate(
        runner.alg,
        ppo_result,
        allow_increase=False,
    )
    optimizer_params, param_snapshots = _optimizer_parameter_snapshots(runner.alg.policy, runner.alg.optimizer)
    optimizer_state_snapshot = copy.deepcopy(runner.alg.optimizer.state_dict())
    grad_norm = 0.0
    post_update_diagnostics: dict[str, Any] = {}
    rejected_lr_diagnostics: dict[str, Any] = {}
    rejected_count = 0
    accepted = True
    max_retries = max(0, int(getattr(runner.alg, "frontres_segment_trust_region_max_retries", 2)))
    rollback_enabled = bool(getattr(runner.alg, "frontres_segment_trust_region_rollback", True))
    schedule = str(getattr(runner.alg, "schedule", "fixed")).lower()
    if ppo_result.should_step:
        # B3: The optimizer step is accepted only after a post-step same-batch
        # diagnostic pass says the policy distribution stayed inside the trust region.
        for attempt in range(max_retries + 1):
            if attempt > 0:
                ppo_result = compute_frontres_segment_ppo_loss(policy_adapter, ppo_batch, ppo_cfg)
            runner.alg.optimizer.zero_grad()
            ppo_result.total_loss.backward()
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                (param for _, param in optimizer_params),
                float(getattr(runner.alg, "max_grad_norm", 1.0)),
            )
            grad_norm = float(grad_norm_tensor.detach().cpu().item())
            runner.alg.optimizer.step()
            post_update_diagnostics = _post_update_segment_ppo_diagnostics(policy_adapter, ppo_batch, ppo_cfg)
            post_kl = float(post_update_diagnostics["post_update_approx_kl"])
            desired_kl = getattr(runner.alg, "desired_kl", None)
            reject = (
                rollback_enabled
                and desired_kl is not None
                and schedule == "adaptive"
                and math.isfinite(post_kl)
                and post_kl > float(desired_kl) * 2.0
            )
            if reject:
                _restore_optimizer_parameters(optimizer_params, param_snapshots)
                runner.alg.optimizer.load_state_dict(optimizer_state_snapshot)
                rejected_count += 1
                rejected_lr_diagnostics = _apply_segment_adaptive_learning_rate(
                    runner.alg,
                    ppo_result,
                    kl_mean=post_kl,
                )
                if attempt < max_retries:
                    continue
                accepted = False
            # B4: Keep legacy ratio_mean/ratio_max as post-step aliases for
            # existing logs, while explicit pre/post fields carry the white-box timing.
            object.__setattr__(ppo_result, "approx_kl", post_kl)
            object.__setattr__(ppo_result, "clip_frac", float(post_update_diagnostics["post_update_clip_frac"]))
            object.__setattr__(
                ppo_result,
                "ratio_mean",
                float(post_update_diagnostics["post_update_clamped_ratio_mean"]),
            )
            object.__setattr__(
                ppo_result,
                "ratio_max",
                float(post_update_diagnostics["post_update_clamped_ratio_max"]),
            )
            break
    if rejected_lr_diagnostics and not accepted:
        lr_diagnostics = rejected_lr_diagnostics
    else:
        lr_diagnostics = pre_step_lr_diagnostics
    diagnostics = _parameter_delta_stats(optimizer_params, param_snapshots)
    diagnostics["param_grad_norm"] = grad_norm
    diagnostics["trust_region_rejected_count"] = rejected_count
    diagnostics["trust_region_accepted"] = int(bool(accepted))
    diagnostics["trust_region_rollback_enabled"] = int(bool(rollback_enabled))
    diagnostics["trust_region_max_retries"] = max_retries
    diagnostics["trust_region_schedule_adaptive"] = int(schedule == "adaptive")
    diagnostics["trust_region_schedule"] = schedule
    for key, value in pre_step_lr_diagnostics.items():
        diagnostics[f"mosaic_pre_step_{key}"] = value
    for key, value in rejected_lr_diagnostics.items():
        diagnostics[f"segment_reject_{key}"] = value
    diagnostics.update(post_update_diagnostics)
    diagnostics.update(lr_diagnostics)
    _attach_ppo_update_diagnostics(ppo_result, diagnostics)
    runner.eval_mode()
    return ppo_result


def _resolve_probe_modes(runner: Any) -> tuple[bool, bool]:
    single_update = bool(
        runner._frontres_segment_replay_boundary.live_single_update_only
        or runner._frontres_segment_replay_boundary.live_update_loop_only
        or runner._frontres_segment_replay_boundary.live_train_enabled
    )
    storage_write = bool(runner._frontres_segment_replay_boundary.live_storage_write_only or single_update)
    if not (runner._frontres_segment_replay_boundary.live_probe_only or storage_write):
        raise ValueError(
            "FrontRES Segment live probe requires frontres_segment_live_probe_only=True "
            "or frontres_segment_live_storage_write_only=True "
            "or frontres_segment_live_single_update_only=True "
            "or frontres_segment_live_update_loop_only=True."
        )
    return single_update, storage_write


def _read_live_observations(runner: Any) -> FrontRESSegmentLiveObservations:
    obs, extras = runner.env.get_observations()
    obs_dict = extras.get("observations", {})
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type]
    privileged_obs = obs_dict.get(runner.privileged_obs_type, obs)
    teacher_obs = obs_dict.get(runner.teacher_obs_type)
    if teacher_obs is None:
        teacher_obs = privileged_obs
    ref_vel_estimator_obs = obs_dict.get(runner.ref_vel_estimator_obs_type)

    obs = runner._apply_obs_normalizer(obs.to(runner.device))
    privileged_obs = runner.privileged_obs_normalizer(privileged_obs.to(runner.device))
    teacher_obs = runner.teacher_obs_normalizer(teacher_obs.to(runner.device))
    if ref_vel_estimator_obs is not None:
        ref_vel_estimator_obs = ref_vel_estimator_obs.to(runner.device)
    return FrontRESSegmentLiveObservations(
        obs=obs,
        privileged_obs=privileged_obs,
        teacher_obs=teacher_obs,
        ref_vel_estimator_obs=ref_vel_estimator_obs,
    )


def _run_live_rollout_capture(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    *,
    rollout_steps: int | None = None,
    capture_motion_quality: bool = True,
    zero_segment_action: bool = False,
) -> FrontRESSegmentLiveRolloutCapture:
    # FRS3-EVAL-014: step the live env and optionally capture motion-quality frames.
    frontres_mode = resolve_frontres_mode_state(runner, FrontRESActorCritic)
    pair_layout = configure_frontres_pair_layout(runner, is_frontres=frontres_mode.is_frontres)
    rollout_k = max(
        1,
        int(
            rollout_steps
            if rollout_steps is not None
            else getattr(runner.alg, "frontres_segment_k", runner._frontres_segment_replay_boundary.segment_k)
        ),
    )
    vel_est_error_buffer = deque(maxlen=1)
    reward_sum = 0.0
    done_sum = 0.0
    reward_accum = None
    done_any = None
    reward_frames = []
    done_frames = []
    survival_steps = None
    actor_update_mask = None
    transition_obs = None
    transition_privileged_obs = None
    transition_actions = None
    transition_log_probs = None
    transition_values = None
    transition_means = None
    transition_sigmas = None
    transition_env_actions = None
    transition_perturbation_rp = None
    transition_supervised_target = None
    action_shape = None
    env_action_shape = None
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    max_delta_rpy = float(getattr(policy, "max_delta_rpy", 0.0)) if policy is not None else None
    clean_body_frames = []
    repaired_body_frames = []
    noisy_body_frames = []
    obs = observations.obs
    privileged_obs = observations.privileged_obs
    teacher_obs = observations.teacher_obs
    ref_vel_estimator_obs = observations.ref_vel_estimator_obs
    last_obs_shape = tuple(obs.shape)

    with torch.inference_mode():
        for rollout_step in range(rollout_k):
            step_plan = prepare_frontres_rollout_step(
                runner,
                obs=obs,
                privileged_obs=privileged_obs,
                teacher_obs=teacher_obs,
                ref_vel_estimator_obs=ref_vel_estimator_obs,
                obs_raw_for_gmt=None,
                vel_est_error_buffer=vel_est_error_buffer,
                iteration=runner.current_learning_iteration,
                rollout_step=rollout_step,
                is_frontres=frontres_mode.is_frontres,
                is_task_space_mode=frontres_mode.is_task_space_mode,
                n_train=pair_layout.n_train,
                n_candidate=pair_layout.n_candidate,
                n_base=pair_layout.n_base,
                n_clean=pair_layout.n_clean,
            )
            actions = step_plan.actions
            env_actions = step_plan.env_actions
            if bool(zero_segment_action) and actions is not None and frontres_mode.is_task_space_mode:
                actions = actions.detach().clone()
                actions[: max(0, min(int(pair_layout.n_train), int(actions.shape[0])))] = 0.0
                runner.alg.transition.actions = actions.detach()
                env_actions = _zero_segment_env_actions(
                    runner,
                    obs=obs,
                    actions=actions,
                    is_frontres=frontres_mode.is_frontres,
                    is_task_space_mode=frontres_mode.is_task_space_mode,
                    n_train=pair_layout.n_train,
                    n_candidate=pair_layout.n_candidate,
                )
            action_shape = tuple(actions.shape) if actions is not None else None
            env_action_shape = tuple(env_actions.shape)
            if rollout_step == 0 and actions is not None:
                transition_obs = runner.alg.transition.observations.detach().clone()
                transition_privileged_obs = runner.alg.transition.privileged_observations.detach().clone()
                transition_env_actions = env_actions.detach().clone()
                transition_perturbation_rp = _snapshot_frontres_perturbation_rp(
                    runner,
                    num_envs=int(actions.shape[0]),
                )
                supervised_target = getattr(runner.alg.transition, "supervised_target", None)
                if supervised_target is not None and supervised_target.ndim == 2 and supervised_target.shape[-1] >= 6:
                    transition_supervised_target = supervised_target.detach().clone()
                selected_actions, selected_log_probs = _select_segment_transition_actions(runner, actions=actions)
                transition_actions = selected_actions.detach().clone()
                transition_log_probs = selected_log_probs.detach().clone().reshape(-1)
                transition_values = runner.alg.transition.values.detach().clone().reshape(-1)
                action_mean = getattr(runner.alg.transition, "action_mean", None)
                action_sigma = getattr(runner.alg.transition, "action_sigma", None)
                if action_mean is not None and action_mean.ndim == 2 and action_mean.shape[-1] >= 6:
                    transition_means = action_mean[:, :6].detach().clone()
                if action_sigma is not None and action_sigma.ndim == 2 and action_sigma.shape[-1] >= 6:
                    transition_sigmas = action_sigma[:, :6].detach().clone()
                actor_update_mask = torch.zeros(actions.shape[0], device=runner.device, dtype=torch.bool)
                actor_update_mask[: max(0, min(int(pair_layout.n_train), actions.shape[0]))] = True

            obs, rewards, dones, infos = runner.env.step(env_actions.to(runner.env.device))
            _print_frontres_dr_runtime_probe(runner, label="after_env_step", rollout_step=rollout_step)
            rewards = rewards.to(runner.device)
            dones = dones.to(runner.device)
            reward_sum += float(rewards.mean().detach().cpu())
            done_sum += float(dones.float().mean().detach().cpu())
            reward_accum = rewards.detach().clone() if reward_accum is None else reward_accum + rewards.detach()
            reward_frames.append(rewards.detach().clone())
            done_frames.append(dones.detach().bool().clone())
            if done_any is None:
                done_any = torch.zeros_like(dones.detach(), dtype=torch.bool)
                survival_steps = torch.zeros_like(rewards.detach(), dtype=torch.float32)
            survival_steps = survival_steps + (~done_any).float()
            done_any = done_any | dones.detach().bool()
            if capture_motion_quality:
                clean_body, repaired_body, noisy_body = _capture_motion_quality_frame(runner, pair_layout)
                if clean_body is not None and repaired_body is not None and noisy_body is not None:
                    clean_body_frames.append(clean_body)
                    repaired_body_frames.append(repaired_body)
                    noisy_body_frames.append(noisy_body)

            obs, privileged_obs, teacher_obs, ref_vel_estimator_obs = _read_step_observations(runner, obs, infos)
            last_obs_shape = tuple(obs.shape)

    return FrontRESSegmentLiveRolloutCapture(
        rollout_k=rollout_k,
        reward_mean=reward_sum / float(rollout_k),
        done_frac=done_sum / float(rollout_k),
        last_obs_shape=last_obs_shape,
        action_shape=action_shape,
        env_action_shape=env_action_shape,
        transition_obs=transition_obs,
        transition_privileged_obs=transition_privileged_obs,
        transition_actions=transition_actions,
        transition_log_probs=transition_log_probs,
        transition_values=transition_values,
        transition_means=transition_means,
        transition_sigmas=transition_sigmas,
        reward_accum=reward_accum,
        done_any=done_any,
        reward_steps=torch.stack(reward_frames, dim=0) if reward_frames else None,
        done_steps=torch.stack(done_frames, dim=0) if done_frames else None,
        actor_update_mask=actor_update_mask,
        n_train=int(pair_layout.n_train),
        n_candidate=int(pair_layout.n_candidate),
        n_base=int(pair_layout.n_base),
        n_clean=int(pair_layout.n_clean),
        survival_steps=survival_steps,
        motion_clean_body_pos=_stack_motion_quality_frames(clean_body_frames),
        motion_repaired_body_pos=_stack_motion_quality_frames(repaired_body_frames),
        motion_noisy_body_pos=_stack_motion_quality_frames(noisy_body_frames),
        env_actions=transition_env_actions,
        transition_perturbation_rp=transition_perturbation_rp,
        transition_supervised_target=transition_supervised_target,
        max_delta_rpy=max_delta_rpy,
    )


def _zero_segment_env_actions(
    runner: Any,
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    is_frontres: bool,
    is_task_space_mode: bool,
    n_train: int,
    n_candidate: int,
) -> torch.Tensor:
    if is_task_space_mode:
        runner._apply_frontres_task_corrections(
            actions,
            n_train,
            allow_oracle=True,
            n_candidate=n_candidate if is_frontres else 0,
        )
        obs_corr, extras_corr = runner.env.get_observations()
        obs_corr_dict = extras_corr.get("observations", {})
        if runner.policy_obs_type is not None and runner.policy_obs_type in obs_corr_dict:
            obs_corr = obs_corr_dict[runner.policy_obs_type]
        obs_corr = runner._apply_obs_normalizer(obs_corr.to(runner.device))
        return runner.alg.policy.get_env_action(obs_corr, actions)
    if hasattr(runner.alg.policy, "get_env_action"):
        return runner.alg.policy.get_env_action(obs, actions)
    return actions


def _capture_motion_quality_frame(
    runner: Any,
    pair_layout: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    command = _motion_command_for_runner(runner)
    if command is None:
        return None, None, None
    clean_ref = getattr(command, "body_pos_w", None)
    if not isinstance(clean_ref, torch.Tensor):
        clean_ref = getattr(command, "body_pos_relative_w", None)
    robot_pos = getattr(command, "robot_body_pos_w", None)
    if not isinstance(clean_ref, torch.Tensor) or not isinstance(robot_pos, torch.Tensor):
        return None, None, None
    n_train = max(0, int(pair_layout.n_train))
    n_candidate = max(0, int(pair_layout.n_candidate))
    n_base = max(0, int(pair_layout.n_base))
    n_clean = max(0, int(pair_layout.n_clean))
    n = min(n_train, n_base, n_clean)
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    if n <= 0 or int(robot_pos.shape[0]) < base_start + n or int(clean_ref.shape[0]) < clean_start + n:
        return None, None, None
    return (
        _root_relative_body_pos(clean_ref[clean_start : clean_start + n]),
        _root_relative_body_pos(robot_pos[:n]),
        _root_relative_body_pos(robot_pos[base_start : base_start + n]),
    )


def _root_relative_body_pos(body_pos: torch.Tensor) -> torch.Tensor:
    if body_pos.ndim < 3 or int(body_pos.shape[-2]) <= 0:
        return body_pos.detach().clone()
    return (body_pos - body_pos[..., :1, :]).detach().clone()


def _motion_command_for_runner(runner: Any) -> Any | None:
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return None
    if hasattr(manager, "get_term"):
        try:
            return manager.get_term("motion")
        except Exception:
            return None
    terms = getattr(manager, "_terms", {})
    return terms.get("motion") if isinstance(terms, dict) else None


def _stack_motion_quality_frames(frames: list[torch.Tensor]) -> torch.Tensor | None:
    if not frames:
        return None
    return torch.stack(frames, dim=1)


def _read_step_observations(runner: Any, obs: torch.Tensor, infos: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    obs_dict = infos.get("observations", {})
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type].to(runner.device)
    else:
        obs = obs.to(runner.device)
    obs = runner._apply_obs_normalizer(obs)
    if runner.privileged_obs_type is not None and runner.privileged_obs_type in obs_dict:
        privileged_obs = runner.privileged_obs_normalizer(obs_dict[runner.privileged_obs_type].to(runner.device))
    else:
        privileged_obs = obs
    if runner.teacher_obs_type is not None and runner.teacher_obs_type in obs_dict:
        teacher_obs = runner.teacher_obs_normalizer(obs_dict[runner.teacher_obs_type].to(runner.device))
    else:
        teacher_obs = privileged_obs
    if runner.ref_vel_estimator_obs_type is not None and runner.ref_vel_estimator_obs_type in obs_dict:
        ref_vel_estimator_obs = obs_dict[runner.ref_vel_estimator_obs_type].to(runner.device)
    else:
        ref_vel_estimator_obs = None
    return obs, privileged_obs, teacher_obs, ref_vel_estimator_obs


def _initial_live_probe_summary(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    storage_write: bool,
    single_update: bool,
) -> dict[str, object]:
    score_summary = _paired_score_summary(capture)
    score_noisy = _mean_sequence(score_summary.get("score_noisy_per_sample", ()))
    score_repaired = _mean_sequence(score_summary.get("score_repaired_per_sample", ()))
    gains = score_summary.get("gain_over_noisy_per_sample", ())
    summary = {
        "reward_mean": capture.reward_mean,
        "env_reward_mean": capture.reward_mean,
        "train_reward_mean": capture.reward_mean,
        "score_noisy_mean": score_noisy,
        "score_repaired_mean": score_repaired,
        "score_gain_mean": score_repaired - score_noisy,
        "score_gain_pos_frac": _positive_fraction(gains),
        "motion_delta_se_norm": _delta_se_norm(capture.transition_actions),
        "motion_delta_z_up_frac": _delta_z_up_frac(capture.transition_actions),
        "done_frac": capture.done_frac,
        "valid_mask_frac": 1.0 - capture.done_frac,
        "reward_per_sample": _rollout_reward_per_sample(capture),
        "done_any_per_sample": _rollout_done_per_sample(capture),
        "storage_write": storage_write,
        "storage_size": 0,
        "storage_valid_frac": 0.0,
        "storage_reward_mean": 0.0,
        "storage_reward_per_sample": [],
        "storage_valid_mask_per_sample": [],
        "storage_segment_ids": [],
        "single_update": single_update,
        "ppo_update": False,
        "ppo_valid_count": 0,
        "ppo_total_loss": 0.0,
        "ppo_actor_loss": 0.0,
        "ppo_value_loss": 0.0,
        "ppo_approx_kl": 0.0,
        "ppo_clip_frac": 0.0,
        "ppo_pre_update_raw_log_ratio_mean": 0.0,
        "ppo_pre_update_raw_log_ratio_min": 0.0,
        "ppo_pre_update_raw_log_ratio_max": 0.0,
        "ppo_pre_update_clamped_ratio_mean": 0.0,
        "ppo_pre_update_clamped_ratio_max": 0.0,
        "ppo_pre_distribution_kl_mean": 0.0,
        "ppo_pre_logprob_approx_kl": 0.0,
        "ppo_distribution_kl_available": False,
        "ppo_post_update_distribution_kl_mean": 0.0,
        "ppo_post_update_logprob_approx_kl": 0.0,
        "ppo_post_update_ratio_mean": 0.0,
        "ppo_post_update_ratio_max": 0.0,
        "ppo_post_update_raw_log_ratio_mean": 0.0,
        "ppo_post_update_raw_log_ratio_min": 0.0,
        "ppo_post_update_raw_log_ratio_max": 0.0,
        "ppo_post_update_clamped_ratio_mean": 0.0,
        "ppo_post_update_clamped_ratio_max": 0.0,
        "ppo_post_update_clip_frac": 0.0,
        "ppo_param_delta_max_abs": 0.0,
        "ppo_param_delta_l2": 0.0,
        "ppo_param_delta_changed": 0,
        "ppo_param_delta_total": 0,
        "ppo_param_delta_first_changed": "",
        "ppo_param_grad_norm": 0.0,
        "ppo_trust_region_rejected_count": 0,
        "ppo_trust_region_accepted": 1,
    }
    summary.update(score_summary)
    summary.update(_motion_quality_summary(capture))
    return summary


def _motion_quality_summary(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, float]:
    try:
        from rsl_rl.frontres.frontres_segment_diagnostics import motion_quality_summary_to_scalars
    except ModuleNotFoundError:
        return {}
    return motion_quality_summary_to_scalars(
        clean_positions=capture.motion_clean_body_pos,
        repaired_positions=capture.motion_repaired_body_pos,
        noisy_positions=capture.motion_noisy_body_pos,
        delta_se=capture.transition_actions,
        valid_mask=capture.actor_update_mask,
    )


def _paired_score_summary(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, object]:
    if capture.reward_accum is None or capture.done_any is None:
        return {}
    n_train = max(0, int(capture.n_train))
    n_candidate = max(0, int(capture.n_candidate))
    n_base = max(0, int(capture.n_base))
    n_clean = max(0, int(capture.n_clean))
    n = min(n_train, n_base)
    if n <= 0:
        return {}
    reward = capture.reward_accum.reshape(-1).detach().float() / float(max(1, int(capture.rollout_k)))
    done = capture.done_any.reshape(-1).detach().bool()
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    if int(reward.numel()) < base_start + n:
        return {}
    clean = reward[clean_start : clean_start + n] if n_clean >= n and int(reward.numel()) >= clean_start + n else torch.ones(n, device=reward.device)
    noisy = reward[base_start : base_start + n]
    repaired = reward[:n]
    return {
        "evidence_row_count": n,
        "evidence_reward_per_sample": _float_list(repaired),
        "evidence_done_any_per_sample": _bool_list(done[:n]),
        "evidence_valid_mask_per_sample": _bool_list(~done[:n]),
        "score_repaired_per_sample": _float_list(repaired),
        "score_noisy_per_sample": _float_list(noisy),
        "gain_over_noisy_per_sample": _float_list(repaired - noisy),
        "score_clean_per_sample": _float_list(clean),
        "score_source": "b1_paired_env_rewards",
    }


def _rollout_reward_per_sample(capture: FrontRESSegmentLiveRolloutCapture) -> list[float]:
    if capture.reward_accum is None:
        return []
    reward = capture.reward_accum.reshape(-1).detach().float() / float(max(1, int(capture.rollout_k)))
    return _float_list(reward)


def _rollout_done_per_sample(capture: FrontRESSegmentLiveRolloutCapture) -> list[bool]:
    if capture.done_any is None:
        return []
    return _bool_list(capture.done_any.reshape(-1))


def _valid_reward_mean(rewards: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid = valid_mask.detach().bool().reshape(-1)
    reward = rewards.detach().float().reshape(-1)
    if int(valid.numel()) != int(reward.numel()):
        raise ValueError(f"valid_mask must have {int(reward.numel())} rows, got {int(valid.numel())}")
    if not bool(valid.any().item()):
        return 0.0
    return float(reward[valid].mean().cpu().item())


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().reshape(-1).cpu().tolist()]


def _bool_list(value: torch.Tensor) -> list[bool]:
    return [bool(item) for item in value.detach().bool().reshape(-1).cpu().tolist()]


def _long_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().long().reshape(-1).cpu().tolist()]


def _print_live_probe_summary(
    runner: Any,
    capture: FrontRESSegmentLiveRolloutCapture,
    summary: dict[str, object],
) -> None:
    """Print the human-facing live probe blocks without changing training state.

    Status: active diagnostic formatter.
    Upstream: run_frontres_segment_live_probe builds summary from rollout, storage, and PPO result.
    Downstream: terminal/log review only; no sampler, loss, optimizer, or checkpoint side effect.
    Evidence: contract-confirmed by frontres_segment_live_probe_contract.py.
    Gap: text presence does not prove live physics quality.
    """
    if not _live_detail_log_enabled(runner):
        return
    segment_action_shape = (
        tuple(capture.transition_actions.shape) if capture.transition_actions is not None else None
    )
    segment_delta_se_6d = bool(_shape_last_dim(segment_action_shape) == 6)
    score_noisy = _mean_sequence(summary.get("score_noisy_per_sample", ()))
    score_repaired = _mean_sequence(summary.get("score_repaired_per_sample", ()))
    print(
        _log_block(
            "[FrontRES Segment Live Probe]",
            *_kv_lines(
                "route",
                {
                    "objective": getattr(runner.alg, "frontres_training_objective", "n/a"),
                    "segment_id": "live_env_current",
                    "reset_mode": runner._frontres_segment_replay_boundary.reset_mode,
                },
            ),
            *_kv_lines(
                "reset",
                {
                    "enabled": bool(summary["segment_reset"]),
                    "reason": summary.get("segment_reset_skip_reason", "") or "applied",
                    "ok": _fmt_pct(summary["segment_reset_success_frac"]),
                    "direct": _fmt_pct(summary["segment_reset_direct_frac"]),
                    "preroll": _fmt_pct(summary["segment_reset_preroll_frac"]),
                    "vel_mismatch": _fmt_num(summary["segment_reset_velocity_mismatch_mean"]),
                    "ref_window": _fmt_pct(summary["segment_reference_window_applied_frac"]),
                },
            ),
            *_kv_lines(
                "rollout",
                {
                    "obs": capture.last_obs_shape,
                    "policy_action": capture.action_shape,
                    "policy_dim": _shape_last_dim(capture.action_shape),
                    "segment_action": segment_action_shape,
                    "segment_delta_se_6d": segment_delta_se_6d,
                    "env_action": capture.env_action_shape,
                    "env_dim": _shape_last_dim(capture.env_action_shape),
                    "k": capture.rollout_k,
                    "env_reward": _fmt_num(summary.get("env_reward_mean", summary["reward_mean"])),
                    "done": _fmt_pct(summary["done_frac"]),
                },
            ),
            *_kv_lines(
                "trial",
                {
                    "roles": summary.get("trial_role_counts", {}),
                    "policy": int(summary.get("trial_policy_count", 0) or 0),
                    "search": int(summary.get("trial_search_count", 0) or 0),
                    "horizon": summary.get("trial_horizon_summary", "horizon_count=0 horizon_min=None horizon_max=None"),
                },
            ),
            *_kv_lines(
                "ppo_boundary",
                {
                    "evidence": int(summary.get("ppo_boundary_evidence_rows", 0) or 0),
                    "policy": int(summary.get("ppo_boundary_policy_rows", 0) or 0),
                    "search": int(summary.get("ppo_boundary_search_rows", 0) or 0),
                    "ppo_valid": int(summary.get("ppo_boundary_eligible_rows", summary.get("ppo_valid_count", 0)) or 0),
                    "search_evidence_only": int(summary.get("ppo_boundary_search_evidence_only_rows", 0) or 0),
                    "policy_invalid": int(summary.get("ppo_boundary_policy_invalid_rows", 0) or 0),
                    "valid_policy": _fmt_pct(summary.get("ppo_boundary_valid_policy_frac", 0.0)),
                    "valid_evidence": _fmt_pct(summary.get("ppo_boundary_valid_evidence_frac", 0.0)),
                },
            ),
            *_kv_lines(
                "score",
                {
                    "source": summary.get("score_source", "synthetic_or_unavailable"),
                    "rows": int(summary.get("evidence_row_count", 0) or 0),
                    "noisy": _fmt_num(score_noisy),
                    "repaired": _fmt_num(score_repaired),
                    "gain": _fmt_num(summary.get("score_gain_mean", score_repaired - score_noisy)),
                    "valid": _fmt_pct(_mean_sequence(summary.get("evidence_valid_mask_per_sample", ()))),
                },
            ),
            *_kv_lines(
                "storage",
                {
                    "write": bool(summary["storage_write"]),
                    "size": int(summary["storage_size"]),
                    "mask_valid": _fmt_pct(summary["valid_mask_frac"]),
                    "valid_frac": _fmt_pct(summary["storage_valid_frac"]),
                    "train_reward": _fmt_num(summary.get("train_reward_mean", summary["storage_reward_mean"])),
                    "all_reward": _fmt_num(summary["storage_reward_mean"]),
                },
            ),
            *_kv_lines(
                "ppo",
                {
                    "single_update": bool(summary["single_update"]),
                    "update": bool(summary["ppo_update"]),
                    "valid": int(summary["ppo_valid_count"]),
                    "loss_total": _fmt_num(summary["ppo_total_loss"]),
                    "actor": _fmt_num(summary["ppo_actor_loss"]),
                    "value": _fmt_num(summary["ppo_value_loss"]),
                    "kl": _fmt_num(summary["ppo_approx_kl"]),
                    "clip": _fmt_pct(summary["ppo_clip_frac"]),
                    "status": _probe_status(summary),
                },
            ),
        ),
        flush=True,
    )
    if bool(summary.get("ppo_update", False)):
        # B1: Separate the same-batch PPO evidence by time. pre_* comes from
        # the loss forward before optimizer.step; post_* comes from the second
        # forward after optimizer.step on the same stored batch.
        print(
            _log_block(
                "[FrontRES Segment PPO Probe]",
                *_kv_lines(
                    "log_prob",
                    {
                        "old": _fmt_num(summary.get("ppo_old_log_prob_mean", 0.0)),
                        "new": _fmt_num(summary.get("ppo_new_log_prob_mean", 0.0)),
                    },
                ),
                *_kv_lines(
                    "pre_log_ratio",
                    {
                        "mean": _fmt_num(summary.get("ppo_pre_update_raw_log_ratio_mean", 0.0)),
                        "min": _fmt_num(summary.get("ppo_pre_update_raw_log_ratio_min", 0.0)),
                        "max": _fmt_num(summary.get("ppo_pre_update_raw_log_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "pre_ratio",
                    {
                        "clamped_mean": _fmt_num(summary.get("ppo_pre_update_clamped_ratio_mean", 0.0)),
                        "clamped_max": _fmt_num(summary.get("ppo_pre_update_clamped_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "kl",
                    {
                        "pre_distribution": _fmt_num(summary.get("ppo_pre_distribution_kl_mean", 0.0)),
                        "pre_logprob": _fmt_num(summary.get("ppo_pre_logprob_approx_kl", 0.0)),
                        "post_distribution": _fmt_num(
                            summary.get("ppo_post_update_distribution_kl_mean", 0.0)
                        ),
                        "post_logprob": _fmt_num(summary.get("ppo_post_update_logprob_approx_kl", 0.0)),
                        "distribution_available": bool(summary.get("ppo_distribution_kl_available", False)),
                    },
                ),
                *_kv_lines(
                    "post_log_ratio",
                    {
                        "mean": _fmt_num(summary.get("ppo_post_update_raw_log_ratio_mean", 0.0)),
                        "min": _fmt_num(summary.get("ppo_post_update_raw_log_ratio_min", 0.0)),
                        "max": _fmt_num(summary.get("ppo_post_update_raw_log_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "post_ratio",
                    {
                        "clamped_mean": _fmt_num(summary.get("ppo_post_update_clamped_ratio_mean", 0.0)),
                        "clamped_max": _fmt_num(summary.get("ppo_post_update_clamped_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "ratio_source",
                    {
                        "raw_action_old_mean_l2": _fmt_num(
                            summary.get("ppo_post_update_raw_action_old_mean_l2_mean", 0.0)
                        ),
                        "raw_action_old_mean_abs_max": _fmt_num(
                            summary.get("ppo_post_update_raw_action_old_mean_abs_max", 0.0)
                        ),
                        "raw_action_old_mean_abs_dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_raw_action_old_mean_abs_dim_mean", ())
                        ),
                        "raw_action_old_mean_abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_raw_action_old_mean_abs_dim_max", ())
                        ),
                    },
                ),
                *_kv_lines(
                    "ratio_sigma",
                    {
                        "old_dim_mean": _fmt_vec(summary.get("ppo_post_update_old_sigma_dim_mean", ())),
                        "new_dim_mean": _fmt_vec(summary.get("ppo_post_update_sigma_dim_mean", ())),
                    },
                ),
                *_kv_lines(
                    "ratio_mean_delta",
                    {
                        "dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_distribution_mean_delta_dim_mean", ())
                        ),
                        "abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_distribution_mean_delta_abs_dim_max", ())
                        ),
                    },
                ),
                *_kv_lines(
                    "ratio_contrib",
                    {
                        "log_ratio_dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_log_ratio_contrib_dim_mean", ())
                        ),
                        "log_ratio_abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_log_ratio_contrib_abs_dim_max", ())
                        ),
                        "log_jacobian_dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_log_jacobian_dim_mean", ())
                        ),
                        "log_jacobian_abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_log_jacobian_abs_dim_max", ())
                        ),
                    },
                ),
                *_kv_lines(
                    "trust",
                    {
                        "accepted": bool(summary.get("ppo_trust_region_accepted", 1)),
                        "rejected": int(summary.get("ppo_trust_region_rejected_count", 0)),
                        "lr_before": _fmt_num(summary.get("ppo_adaptive_lr_before", 0.0)),
                        "lr_after": _fmt_num(summary.get("ppo_adaptive_lr_after", 0.0)),
                        "desired_kl": _fmt_num(summary.get("ppo_adaptive_lr_desired_kl", 0.0)),
                        "schedule": str(summary.get("ppo_trust_region_schedule", "unknown")),
                        "rollback": bool(summary.get("ppo_trust_region_rollback_enabled", 0)),
                        "max_retries": int(summary.get("ppo_trust_region_max_retries", 0)),
                    },
                ),
                *_kv_lines(
                    "advantage",
                    {
                        "mean": _fmt_num(summary.get("ppo_advantage_mean", 0.0)),
                        "min": _fmt_num(summary.get("ppo_advantage_min", 0.0)),
                        "max": _fmt_num(summary.get("ppo_advantage_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "param_delta",
                    {
                        "max_abs": _fmt_num(summary.get("ppo_param_delta_max_abs", 0.0)),
                        "l2": _fmt_num(summary.get("ppo_param_delta_l2", 0.0)),
                        "changed": (
                            f"{int(summary.get('ppo_param_delta_changed', 0))}/"
                            f"{int(summary.get('ppo_param_delta_total', 0))}"
                        ),
                        "first": summary.get("ppo_param_delta_first_changed", ""),
                        "grad_norm": _fmt_num(summary.get("ppo_param_grad_norm", 0.0)),
                    },
                ),
            ),
            flush=True,
        )
