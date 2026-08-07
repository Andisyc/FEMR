"""FrontRES Segment reset, role layout, and sealed metadata adapter."""





from __future__ import annotations





from collections import Counter


from collections.abc import Mapping


from types import SimpleNamespace


from typing import Any


import torch


from rsl_rl.frontres.frontres_segment_reset import FrontRESSegmentResetAdapter, FrontRESSegmentResetResult, ensure_frontres_segment_live_reset_hook





from rsl_rl.runners.frontres_segment_runtime_types import (


    FrontRESSegmentLiveRolloutCapture,
    frontres_collection_batch,


)


from rsl_rl.runners.frontres_segment_probe_logging import (
    format_count_summary as _count_summary,
    tensor_float_list as _float_list,
    format_probe_percent as _fmt_pct,
    format_id_summary as _id_summary,
    probe_kv_lines as _kv_lines,
    live_detail_log_enabled as _live_detail_log_enabled,
    probe_log_block as _log_block,
    tensor_long_list as _long_list,
    format_motion_summary as _motion_summary,
    tensor_nonzero_fraction as _tensor_nonzero_frac,
    tensor_range_summary as _tensor_range_summary,
    verbose_index_reset_lines as _verbose_index_reset_lines,
    verbose_probe_enabled as _verbose_probe_enabled,
    verbose_reset_lines as _verbose_reset_lines,
)





def _apply_current_segment_reset(
    runner: Any,
    *,
    pair_layout: Any | None = None,
    local_scenario_execution_mode: str | None = None,
) -> FrontRESSegmentResetResult | None:
    # FRS3-EVAL-013: apply the current index-only reset batch to the live env.
    batch = frontres_collection_batch(runner)
    if batch is None:
        runner._frontres_segment_live_current_reset_skip_reason = "no_current_segment_batch"
        return None
    if _is_index_only_segment_batch(batch):
        return _apply_index_only_segment_reset(
            runner,
            batch,
            pair_layout=pair_layout,
            local_scenario_execution_mode=local_scenario_execution_mode,
        )
    if local_scenario_execution_mode is not None:
        raise RuntimeError("v017 local-scenario execution mode is only valid for an index-only sealed reset")
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
    trial_metadata = _current_trial_metadata(
        runner,
        batch_size=int(request.segment_ids.numel()),
        device=request.segment_ids.device,
    )
    _attach_trial_metadata_to_request(request, trial_metadata)
    _attach_frozen_transaction_metadata_to_request(
        request,
        runner=runner,
        batch=batch,
        trial_metadata=trial_metadata,
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


def apply_frontres_current_segment_reset(
    runner: Any,
    *,
    pair_layout: Any | None = None,
    local_scenario_execution_mode: str | None = None,
) -> FrontRESSegmentResetResult | None:
    """Public reset gateway for evaluators using the sealed Segment state."""

    return _apply_current_segment_reset(
        runner,
        pair_layout=pair_layout,
        local_scenario_execution_mode=local_scenario_execution_mode,
    )


def _is_index_only_segment_batch(batch: Any) -> bool:
    families = tuple(getattr(batch, "perturbation_family", ()) or ())
    if families:
        return all(str(family) == "index_only" for family in families)
    specs = tuple(getattr(batch, "specs", ()) or ())
    return bool(specs) and all(str(getattr(spec, "perturbation_family", "")) == "index_only" for spec in specs)


def _apply_index_only_segment_reset(
    runner: Any,
    batch: Any,
    *,
    pair_layout: Any | None = None,
    local_scenario_execution_mode: str | None = None,
) -> FrontRESSegmentResetResult | None:
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
    v015_local_scenario = getattr(batch, "frontres_local_scenario_rows", None) is not None
    if v015_local_scenario:
        _attach_frontres_local_scenario_to_index_request(request, batch)
        if local_scenario_execution_mode is not None:
            mode = str(local_scenario_execution_mode)
            if mode not in {"clean_baseline", "noisy_baseline", "repair_attempts"}:
                raise ValueError(f"unsupported v017 local-scenario execution mode: {mode!r}")
            # B1: mode 随 reset request 进入环境 owner；command carrier 尚未安装时禁止提前切换。
            request.frontres_local_scenario_execution_mode = mode
    else:
        if local_scenario_execution_mode is not None:
            raise RuntimeError("v017 local-scenario execution mode requires one sealed local scenario")
        _attach_fixed_noisy_tape_to_index_request(request, batch)
    if pair_layout is not None:
        request.frontres_role_env_ids = _frontres_reset_role_env_ids(
            pair_layout,
            source_count=int(batch.segment_ids.numel()),
            device=batch.segment_ids.device,
            v015_local=v015_local_scenario,
        )
    _attach_trial_metadata_to_request(request, trial_metadata)
    _attach_frozen_transaction_metadata_to_request(
        request,
        runner=runner,
        batch=batch,
        trial_metadata=trial_metadata,
    )
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


def _attach_fixed_noisy_tape_to_index_request(request: Any, batch: Any) -> None:
    tape = getattr(batch, "frontres_fixed_noisy_tape", None)
    if tape is None:
        return
    if not isinstance(tape, torch.Tensor) or tape.ndim != 3:
        raise ValueError(f"frontres_fixed_noisy_tape must be [B,L,65], got {getattr(tape, 'shape', None)}")
    batch_size = int(request.segment_ids.numel())
    if int(tape.shape[0]) != batch_size or int(tape.shape[-1]) != 65:
        raise ValueError("frontres_fixed_noisy_tape must align with reset rows and use the 65D carrier")
    offsets = tuple(int(value) for value in (getattr(batch, "frontres_future_offsets", ()) or ()))
    if not offsets or any(value <= 0 for value in offsets) or tuple(sorted(set(offsets))) != offsets:
        raise ValueError("frontres_future_offsets must be nonempty positive ordered offsets for fixed Noisy reset")
    lengths = getattr(batch, "frontres_fixed_noisy_tape_lengths", None)
    scenario_ids = tuple(str(value) for value in (getattr(batch, "frontres_fixed_noisy_scenario_ids", ()) or ()))
    hashes = tuple(str(value) for value in (getattr(batch, "frontres_fixed_noisy_segment_hashes", ()) or ()))
    if (
        not isinstance(lengths, torch.Tensor)
        or int(lengths.numel()) != batch_size
        or len(scenario_ids) != batch_size
        or len(hashes) != batch_size
    ):
        raise ValueError("fixed Noisy reset requires source-aligned tape lengths, scenario ids, and hashes")
    request.frontres_fixed_noisy_tape = tape.detach()
    request.frontres_fixed_noisy_tape_lengths = lengths.detach()
    request.frontres_fixed_noisy_scenario_ids = scenario_ids
    request.frontres_fixed_noisy_segment_hashes = hashes
    request.frontres_future_offsets = offsets


def _attach_frontres_local_scenario_to_index_request(request: Any, batch: Any) -> None:
    """Attach only the v015 split local carrier to an index-reset request.

    The request owns no actor-side Clean reference: q29 intent and the full 65D
    continuation remain separate fields for the command/reset owner to route to
    the actor and frozen GMT consumers respectively.
    """

    if getattr(batch, "frontres_fixed_noisy_tape", None) is not None:
        raise ValueError("v015 local reset request cannot mix a sealed local scenario with a legacy fixed Noisy tape")
    rows = getattr(batch, "frontres_local_scenario_rows", None)
    artifact = getattr(batch, "frontres_local_scenario_current_root_artifact_t", None)
    clean_reference_t = getattr(batch, "frontres_local_scenario_clean_reference_t", None)
    intent = getattr(batch, "frontres_local_scenario_intent_q29", None)
    continuation = getattr(batch, "frontres_local_scenario_clean_continuation", None)
    expected_support = getattr(batch, "frontres_local_scenario_expected_support", None)
    expected_support_envelope = getattr(batch, "frontres_local_scenario_expected_support_envelope", None)
    lengths = getattr(batch, "frontres_local_scenario_clean_continuation_lengths", None)
    mask = getattr(batch, "frontres_local_scenario_clean_continuation_mask", None)
    scenario_ids = tuple(str(value) for value in (getattr(batch, "frontres_local_scenario_ids", ()) or ()))
    hashes = tuple(str(value) for value in (getattr(batch, "frontres_local_scenario_hashes", ()) or ()))
    x_t_identities = tuple(str(value) for value in (getattr(batch, "frontres_local_scenario_x_t_identities", ()) or ()))
    provenance = tuple(getattr(batch, "frontres_local_scenario_provenance", ()) or ())
    offsets = tuple(int(value) for value in (getattr(batch, "frontres_future_offsets", ()) or ()))
    batch_size = int(request.segment_ids.numel())
    if (
        rows is None
        or not isinstance(artifact, torch.Tensor)
        or not isinstance(clean_reference_t, torch.Tensor)
        or not isinstance(intent, torch.Tensor)
        or not isinstance(continuation, torch.Tensor)
        or not isinstance(expected_support, torch.Tensor)
        or not isinstance(expected_support_envelope, torch.Tensor)
        or not isinstance(lengths, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or not offsets
        or any(value <= 0 for value in offsets)
        or tuple(sorted(set(offsets))) != offsets
        or tuple(artifact.shape) != (batch_size, 7)
        or tuple(clean_reference_t.shape) != (batch_size, 65)
        or tuple(intent.shape) != (batch_size, max(offsets) + 1, 29)
        or continuation.ndim != 3
        or tuple(continuation.shape[:1]) != (batch_size,)
        or int(continuation.shape[-1]) != 65
        or tuple(expected_support.shape) != tuple(continuation.shape[:2]) + (2,)
        or tuple(expected_support_envelope.shape) != tuple(continuation.shape[:2]) + (6,)
        or tuple(lengths.shape) != (batch_size,)
        or tuple(mask.shape) != tuple(continuation.shape[:2])
        or len(scenario_ids) != batch_size
        or len(hashes) != batch_size
        or len(x_t_identities) != batch_size
        or len(provenance) != batch_size
    ):
        raise ValueError("v015 local reset request requires one aligned sealed artifact, q29 intent, Clean continuation, identity, and provenance row")
    if any(not isinstance(value, Mapping) for value in provenance):
        raise ValueError("v015 local reset request requires mapping provenance for every local scenario row")
    request.frontres_local_scenario_rows = rows
    request.frontres_local_scenario_current_root_artifact_t = artifact.detach().clone()
    request.frontres_local_scenario_clean_reference_t = clean_reference_t.detach().clone()
    request.frontres_local_scenario_intent_q29 = intent.detach().clone()
    request.frontres_local_scenario_clean_continuation = continuation.detach().clone()
    request.frontres_local_scenario_expected_support = expected_support.detach().clone()
    request.frontres_local_scenario_expected_support_envelope = expected_support_envelope.detach().clone()
    request.frontres_local_scenario_clean_continuation_lengths = lengths.detach().clone()
    request.frontres_local_scenario_clean_continuation_mask = mask.detach().clone()
    request.frontres_local_scenario_ids = scenario_ids
    request.frontres_local_scenario_hashes = hashes
    request.frontres_local_scenario_x_t_identities = x_t_identities
    request.frontres_local_scenario_provenance = tuple(dict(value) for value in provenance)
    request.frontres_future_offsets = offsets


def _frontres_reset_role_env_ids(
    pair_layout: Any,
    *,
    source_count: int,
    device: torch.device,
    v015_local: bool = False,
) -> dict[str, torch.Tensor]:
    """将 sampled policy rows 映射到配对的 split-env role rows."""
    source_count = int(source_count)
    counts = (
        (("repair", int(getattr(pair_layout, "n_train", 0))), ("noisy", int(getattr(pair_layout, "n_base", 0))) )
        if v015_local
        else (
            ("policy", int(getattr(pair_layout, "n_train", 0))),
            ("candidate", int(getattr(pair_layout, "n_candidate", 0))),
            ("noisy", int(getattr(pair_layout, "n_base", 0))),
            ("clean", int(getattr(pair_layout, "n_clean", 0))),
        )
    )
    active_counts = [count for _, count in counts if count > 0]
    if not active_counts or any(count != source_count for count in active_counts):
        raise ValueError(
            "Segment index reset requires one split-env row per sampled source and active role; "
            f"source_count={source_count} role_counts={dict(counts)}"
        )
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for role, count in counts:
        if count > 0:
            result[role] = torch.arange(offset, offset + count, dtype=torch.long, device=device)
        offset += count
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
    source_state = _mapping_float(mapping, ("source_state_max_abs_diff",), count, device, 0.0)
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
        "source_state_max_abs_diff": float(source_state.max().item()) if count else 0.0,
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
    request: Any | None = None,
    skip_reason: str = "",
) -> None:
    # B1: Read the perturbation identity consumed by the reset owner.
    families = tuple(str(item) for item in (getattr(request, "perturbation_family", ()) or ()))
    strength = getattr(request, "perturbation_strength", None)
    strength_values = _float_list(strength) if isinstance(strength, torch.Tensor) else []
    # B2: Preserve distribution facts for diagnostics without changing reset behavior.
    summary.update(
        {
            "perturbation_family_counts": _count_summary(families),
            "perturbation_strength_min": min(strength_values) if strength_values else 0.0,
            "perturbation_strength_mean": (
                sum(strength_values) / float(len(strength_values)) if strength_values else 0.0
            ),
            "perturbation_strength_max": max(strength_values) if strength_values else 0.0,
        }
    )
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
                "segment_source_state_max_abs_diff": 0.0,
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
            "segment_source_state_max_abs_diff": float(diagnostics.get("source_state_max_abs_diff", 0.0)),
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
    batch = frontres_collection_batch(runner)
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
        horizon_k=_trial_horizon_vector(
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


def _trial_horizon_vector(
    value: Any,
    *,
    name: str,
    batch_size: int,
    device: torch.device | str,
    default: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return default.to(device=device, dtype=torch.long).reshape(-1).detach()
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(list(value), dtype=torch.long)
    tensor = tensor.to(device=device, dtype=torch.long).reshape(-1)
    if int(tensor.numel()) == int(batch_size):
        return tensor.detach()
    if int(tensor.numel()) > 0 and int(batch_size) % int(tensor.numel()) == 0:
        return tensor.repeat(int(batch_size) // int(tensor.numel())).detach()
    return _trial_long_vector(
        tensor,
        name=name,
        batch_size=batch_size,
        device=device,
        default=default,
    )


def _attach_trial_metadata_to_request(request: Any, metadata: SimpleNamespace) -> None:
    object.__setattr__(request, "trial_role", metadata.trial_role)
    object.__setattr__(request, "source_index", metadata.source_index)
    object.__setattr__(request, "trial_index", metadata.trial_index)
    object.__setattr__(request, "budget_horizon_k", metadata.horizon_k)


def _frozen_transaction_vector_has_rows(value: Any, *, batch_size: int) -> bool:
    return isinstance(value, torch.Tensor) and value.ndim == 1 and int(value.numel()) == int(batch_size)


def _same_frozen_transaction_vector(left: Any, right: Any, *, batch_size: int) -> bool:
    if not _frozen_transaction_vector_has_rows(left, batch_size=batch_size) or not _frozen_transaction_vector_has_rows(
        right,
        batch_size=batch_size,
    ):
        return False
    return torch.equal(
        left.detach().to(device="cpu", dtype=torch.long).reshape(-1),
        right.detach().to(device="cpu", dtype=torch.long).reshape(-1),
    ) and int(left.numel()) == int(batch_size)


def _current_frozen_transaction_metadata(
    runner: Any,
    *,
    batch_size: int,
    trial_metadata: SimpleNamespace,
) -> Any | None:
    """Fail closed when a sealed S1b transaction carrier disagrees with the selected batch."""

    batch = frontres_collection_batch(runner)
    metadata = getattr(batch, "frontres_segment_transaction_metadata", None) if batch is not None else None
    if metadata is None:
        return None
    validate = getattr(metadata, "validate", None)
    verify_policy = getattr(metadata, "verify_policy", None)
    if not callable(validate) or not callable(verify_policy):
        raise TypeError("frozen transaction metadata must provide validate() and verify_policy()")
    validate()
    for name in (
        "transaction_id",
        "policy_snapshot_id",
        "policy_state_hash",
        "motion_ids",
        "start_frames",
        "segment_ids",
        "source_index",
        "trial_index",
        "horizon_k",
        "trial_role",
        "noisy_segment_hashes",
    ):
        if not hasattr(metadata, name):
            raise TypeError(f"frozen transaction metadata is missing {name}")
    if not str(metadata.transaction_id) or not str(metadata.policy_snapshot_id) or not str(metadata.policy_state_hash):
        raise ValueError("frozen transaction metadata identity must be non-empty")
    if len(tuple(metadata.motion_ids)) != int(batch_size) or len(tuple(metadata.trial_role)) != int(batch_size):
        raise ValueError("frozen transaction metadata row count does not match the reset/storage batch")
    if len(tuple(metadata.noisy_segment_hashes)) != int(batch_size):
        raise ValueError("frozen transaction metadata requires one Noisy hash per batch row")
    for name in ("start_frames", "segment_ids", "source_index", "trial_index", "horizon_k"):
        if not _frozen_transaction_vector_has_rows(getattr(metadata, name), batch_size=batch_size):
            raise ValueError(f"frozen transaction metadata {name} must be [B]")
    for name, expected, actual in (
        ("segment_ids", metadata.segment_ids, getattr(batch, "segment_ids", None)),
        ("source_index", metadata.source_index, trial_metadata.source_index),
        ("trial_index", metadata.trial_index, trial_metadata.trial_index),
        ("horizon_k", metadata.horizon_k, trial_metadata.horizon_k),
    ):
        if not _same_frozen_transaction_vector(expected, actual, batch_size=batch_size):
            raise ValueError(f"frozen transaction metadata {name} disagrees with the selected batch")
    if tuple(str(value) for value in metadata.trial_role) != tuple(trial_metadata.trial_role):
        raise ValueError("frozen transaction metadata trial_role disagrees with the selected batch")
    if getattr(batch, "frontres_segment_transaction_id", None) != metadata.transaction_id:
        raise ValueError("batch transaction_id disagrees with frozen transaction metadata")
    if getattr(batch, "frontres_segment_policy_snapshot_id", None) != metadata.policy_snapshot_id:
        raise ValueError("batch policy_snapshot_id disagrees with frozen transaction metadata")
    if getattr(batch, "frontres_segment_policy_state_hash", None) != metadata.policy_state_hash:
        raise ValueError("batch policy_state_hash disagrees with frozen transaction metadata")
    if getattr(batch, "frontres_fixed_noisy_transaction_id", None) != metadata.transaction_id:
        raise ValueError("fixed Noisy tape transaction_id disagrees with frozen transaction metadata")
    batch_hashes = tuple(str(value) for value in (getattr(batch, "frontres_fixed_noisy_segment_hashes", ()) or ()))
    if batch_hashes != tuple(str(value) for value in metadata.noisy_segment_hashes):
        raise ValueError("fixed Noisy tape hash rows disagree with frozen transaction metadata")
    return metadata


def _attach_frozen_transaction_metadata_to_request(
    request: Any,
    *,
    runner: Any,
    batch: Any,
    trial_metadata: SimpleNamespace,
) -> Any | None:
    metadata = _current_frozen_transaction_metadata(
        runner,
        batch_size=int(request.segment_ids.numel()),
        trial_metadata=trial_metadata,
    )
    if metadata is None:
        return None
    if not _same_frozen_transaction_vector(metadata.segment_ids, request.segment_ids, batch_size=int(request.segment_ids.numel())):
        raise ValueError("frozen transaction metadata segment_ids disagree with reset request")
    request_motion_ids = getattr(request, "motion_ids", None)
    if request_motion_ids is not None and tuple(str(value) for value in request_motion_ids) != tuple(metadata.motion_ids):
        raise ValueError("frozen transaction metadata motion_ids disagree with reset request")
    request_start_frames = getattr(request, "start_frames", None)
    if request_start_frames is not None and not _same_frozen_transaction_vector(
        metadata.start_frames,
        request_start_frames,
        batch_size=int(request.segment_ids.numel()),
    ):
        raise ValueError("frozen transaction metadata start_frames disagree with reset request")
    for name, value in (
        ("frontres_segment_transaction_metadata", metadata),
        ("frontres_segment_transaction_id", metadata.transaction_id),
        ("frontres_segment_policy_snapshot_id", metadata.policy_snapshot_id),
        ("frontres_segment_policy_state_hash", metadata.policy_state_hash),
        ("frontres_segment_motion_ids", metadata.motion_ids),
        ("frontres_segment_start_frames", metadata.start_frames),
        ("frontres_segment_segment_ids", metadata.segment_ids),
        ("frontres_segment_source_index", metadata.source_index),
        ("frontres_segment_trial_index", metadata.trial_index),
        ("frontres_segment_budget_horizon_k", metadata.horizon_k),
        ("frontres_segment_trial_role", metadata.trial_role),
        ("frontres_segment_noisy_segment_hashes", metadata.noisy_segment_hashes),
    ):
        object.__setattr__(request, name, value)
    return metadata


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


# Public reset and sealed-metadata owner surface.
capture_frontres_batch_size = _capture_batch_size
current_frontres_trial_metadata = _current_trial_metadata
current_frontres_frozen_transaction_metadata = _current_frozen_transaction_metadata
frontres_trial_metadata_priority_evidence = _trial_metadata_priority_evidence
frontres_trial_metadata_ppo_update_mask = _trial_metadata_ppo_update_mask
frontres_reset_role_env_ids = _frontres_reset_role_env_ids
frontres_index_segment_reset_hook = _index_segment_reset_hook
frontres_index_reset_result_from_mapping = _index_reset_result_from_mapping
update_frontres_reset_summary = _update_reset_summary
update_frontres_trial_metadata_summary = _update_trial_metadata_summary
update_frontres_ppo_boundary_summary = _update_ppo_boundary_summary
