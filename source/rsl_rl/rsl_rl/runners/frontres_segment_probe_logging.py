"""Shared live-probe formatting and audit identity helpers."""





from __future__ import annotations





from collections import Counter


import hashlib


import math


from typing import Any


import torch





from rsl_rl.runners.frontres_segment_runtime_types import (


    FrontRESSegmentLiveRolloutCapture,


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


def _finite_mean(value: torch.Tensor | None) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return float("nan")
    flat = value.detach().float().reshape(-1)
    finite = torch.isfinite(flat)
    if not bool(finite.any().item()):
        return float("nan")
    return float(flat[finite].mean().cpu().item())


def _fmt_metric(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "UNCONFIRMED"
    return "UNCONFIRMED" if not math.isfinite(value) else _fmt_num(value)


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


_AUDIT_IDENTITY_KEYS = (
    "audit_transaction_id",
    "audit_batch_signature",
    "audit_role_signature",
    "audit_k_signature",
    "audit_segment_signature",
    "audit_row_count",
    "audit_identity_state",
)


def _audit_identity_kwargs(identity: dict[str, Any] | None) -> dict[str, Any]:
    """Return the compact identity fields shared by cards 15-17."""

    if not isinstance(identity, dict):
        return {
            "audit_transaction_id": "UNCONFIRMED",
            "audit_batch_signature": "UNCONFIRMED",
            "audit_role_signature": "UNCONFIRMED",
            "audit_k_signature": "UNCONFIRMED",
            "audit_segment_signature": "UNCONFIRMED",
            "audit_row_count": 0,
            "audit_identity_state": "UNCONFIRMED",
        }
    return {key: identity.get(key, "UNCONFIRMED") for key in _AUDIT_IDENTITY_KEYS}


def _capture_audit_identity_kwargs(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, Any]:
    return _audit_identity_kwargs(
        {
            "audit_transaction_id": capture.audit_transaction_id,
            "audit_batch_signature": capture.audit_batch_signature,
            "audit_role_signature": capture.audit_role_signature,
            "audit_k_signature": capture.audit_k_signature,
            "audit_segment_signature": capture.audit_segment_signature,
            "audit_row_count": capture.audit_row_count,
            "audit_identity_state": capture.audit_identity_state,
        }
    )


def _audit_identity_tuple(value: Any, batch_size: int, default: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().reshape(-1).cpu().tolist()
    try:
        items = tuple(value)
    except TypeError:
        items = ()
    if len(items) == batch_size:
        return items
    if len(items) > 0 and batch_size % len(items) == 0:
        return items * (batch_size // len(items))
    return (default,) * batch_size


def _new_live_audit_identity(
    runner: Any,
    *,
    pair_layout: Any,
    batch_size: int,
    horizon_k: torch.Tensor,
) -> dict[str, Any]:
    """Create one stable row identity for the current rollout capture.

    Status: active evidence identity owner.
    Upstream: current segment batch/reset request and rollout horizon.
    Downstream: paired Gain, Segment storage/returns, and diagnostics.
    Evidence: offline identity contract; live equality remains to be observed.
    """

    counter = int(getattr(runner, "_frontres_segment_audit_transaction_counter", 0)) + 1
    runner._frontres_segment_audit_transaction_counter = counter
    iteration = int(getattr(runner, "current_learning_iteration", 0))
    transaction_id = f"iter{iteration}:capture{counter}"
    current_batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    sample = getattr(runner, "_frontres_segment_live_current_sample", None)
    raw_segment_ids = getattr(current_batch, "segment_ids", None)
    if raw_segment_ids is None:
        raw_segment_ids = getattr(sample, "segment_ids", None)
    segment_ids = _audit_identity_tuple(raw_segment_ids, batch_size, -1)
    raw_roles = getattr(current_batch, "frontres_segment_trial_role", None)
    if raw_roles is None:
        raw_roles = getattr(sample, "trial_role", None)
    roles = tuple(str(item) for item in _audit_identity_tuple(raw_roles, batch_size, "UNCONFIRMED"))
    request = getattr(runner, "_frontres_segment_live_current_reset_request", None)
    motion_ids = tuple(
        str(item)
        for item in _audit_identity_tuple(getattr(request, "motion_ids", None), batch_size, "UNCONFIRMED")
    )
    start_frames = tuple(
        int(item)
        for item in _audit_identity_tuple(getattr(request, "start_frames", None), batch_size, -1)
    )
    horizon = tuple(int(item) for item in horizon_k.detach().long().reshape(-1).cpu().tolist())
    if len(horizon) != batch_size:
        horizon = (int(max(1, int(getattr(pair_layout, "rollout_k", 1)))),) * batch_size
    rows = tuple(zip(segment_ids, roles, motion_ids, start_frames, horizon))
    batch_signature = hashlib.sha1(repr(rows).encode("utf-8")).hexdigest()[:16]
    identity_state = (
        "complete"
        if all(item != "UNCONFIRMED" for item in motion_ids)
        and all(item >= 0 for item in start_frames)
        and all(item != "UNCONFIRMED" for item in roles)
        else "partial"
    )
    identity = {
        "audit_transaction_id": transaction_id,
        "audit_batch_signature": batch_signature,
        "audit_role_signature": "|".join(roles),
        "audit_k_signature": ",".join(str(item) for item in horizon),
        "audit_segment_signature": ",".join(str(item) for item in segment_ids),
        "audit_row_count": batch_size,
        "audit_identity_state": identity_state,
    }
    runner._frontres_segment_live_audit_identity = identity
    return identity


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


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().reshape(-1).cpu().tolist()]


def _bool_list(value: torch.Tensor) -> list[bool]:
    return [bool(item) for item in value.detach().bool().reshape(-1).cpu().tolist()]


def _long_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().long().reshape(-1).cpu().tolist()]


# Consumer-shaped public formatting and audit surface.
probe_log_block = _log_block
probe_kv_lines = _kv_lines
format_probe_number = _fmt_num
format_probe_percent = _fmt_pct
format_probe_vector = _fmt_vec
positive_fraction = _positive_fraction
finite_tensor_mean = _finite_mean
format_probe_metric = _fmt_metric
shape_last_dim = _shape_last_dim
delta_se_norm = _delta_se_norm
delta_z_up_fraction = _delta_z_up_frac
probe_status = _probe_status
verbose_probe_enabled = _verbose_probe_enabled
format_id_summary = _id_summary
audit_identity_kwargs = _audit_identity_kwargs
capture_audit_identity_kwargs = _capture_audit_identity_kwargs
new_live_audit_identity = _new_live_audit_identity
tensor_range_summary = _tensor_range_summary
tensor_nonzero_fraction = _tensor_nonzero_frac
print_frontres_dr_runtime_probe = _print_frontres_dr_runtime_probe
format_count_summary = _count_summary
format_motion_summary = _motion_summary
verbose_reset_lines = _verbose_reset_lines
verbose_index_reset_lines = _verbose_index_reset_lines
should_print_once_or_verbose = _should_print_once_or_verbose
live_detail_log_enabled = _live_detail_log_enabled
try_frontres_motion_command = _motion_command_for_runner
tensor_float_list = _float_list
tensor_bool_list = _bool_list
tensor_long_list = _long_list
