from __future__ import annotations

import math
import importlib.util
from pathlib import Path
from typing import Any, Mapping

import torch

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_runner",
    Path(__file__).resolve().parents[1] / "frontres" / "frontres_formal_runtime_probe.py",
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
configure_formal_runtime_probe = _AUDIT_MODULE.configure_formal_runtime_probe
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


def formal_runtime_audit_enabled(runner: Any) -> bool:
    enabled = bool(getattr(getattr(runner, "alg", None), "frontres_formal_runtime_audit", False))
    configure_formal_runtime_probe(enabled)
    return enabled


def _tensor_stats(value: Any) -> str:
    if not isinstance(value, torch.Tensor):
        return "missing"
    tensor = value.detach()
    numeric = tensor.float()
    finite = bool(torch.isfinite(numeric).all().item()) if tensor.numel() else True
    stats = f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} finite={int(finite)}"
    if tensor.numel() and numeric.dtype != torch.bool:
        stats += f" min={numeric.min().item():.6g} max={numeric.max().item():.6g} mean={numeric.mean().item():.6g}"
    return stats


def _summary_value(summary: Mapping[str, Any], key: str) -> str:
    value = summary.get(key, "missing")
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return str(value)


def _config_value(owner: Any, key: str, default: Any = "missing") -> Any:
    if isinstance(owner, Mapping):
        return owner.get(key, default)
    return getattr(owner, key, default)


def _emit_owner_snapshot(audit_id: str, **values: Any) -> None:
    """Print one compact owner-boundary snapshot for the Runtime Audit Atlas."""
    emit_formal_runtime_probe(audit_id, limit=2, **values)


def _role_slices(pair_layout: Any, batch_size: int) -> dict[str, slice]:
    counts = {
        "policy": int(getattr(pair_layout, "n_train", 0)),
        "candidate": int(getattr(pair_layout, "n_candidate", 0)),
        "noisy": int(getattr(pair_layout, "n_base", 0)),
        "clean": int(getattr(pair_layout, "n_clean", 0)),
    }
    if sum(counts.values()) != int(batch_size):
        return {"layout_error": slice(0, int(batch_size))}
    result: dict[str, slice] = {}
    start = 0
    for role, count in counts.items():
        result[role] = slice(start, start + count)
        start += count
    return result


def _role_tensor_stats(value: Any, pair_layout: Any, *, batch_size: int) -> dict[str, str]:
    if not isinstance(value, torch.Tensor) or value.ndim == 0 or int(value.shape[0]) != int(batch_size):
        return {"status": "missing_or_shape_mismatch"}
    return {
        role: _tensor_stats(value[role_slice])
        for role, role_slice in _role_slices(pair_layout, batch_size).items()
    }


def _role_true_counts(value: Any, pair_layout: Any, *, batch_size: int) -> dict[str, int | str]:
    if not isinstance(value, torch.Tensor) or value.ndim == 0 or int(value.shape[0]) != int(batch_size):
        return {"status": "missing_or_shape_mismatch"}
    mask = value.detach().bool().reshape(batch_size, -1).any(dim=-1)
    return {
        role: int(mask[role_slice].sum().item())
        for role, role_slice in _role_slices(pair_layout, batch_size).items()
    }


def _resolve_audit_robot(runner: Any) -> Any | None:
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    try:
        return scene["robot"]
    except (KeyError, TypeError):
        return getattr(scene, "robot", None)


def snapshot_reset_pair_state(runner: Any, pair_layout: Any) -> dict[str, Any]:
    """Measure whether quartet robot states are paired immediately after index reset."""
    robot = _resolve_audit_robot(runner)
    data = getattr(robot, "data", None)
    if data is None:
        return {"root_pair_error": "missing_robot", "joint_pair_error": "missing_robot"}
    root_pos = getattr(data, "root_pos_w", None)
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    env_origins = getattr(getattr(env, "scene", None), "env_origins", None)
    if (
        isinstance(root_pos, torch.Tensor)
        and isinstance(env_origins, torch.Tensor)
        and root_pos.ndim == 2
        and env_origins.ndim == 2
        and int(env_origins.shape[0]) >= int(root_pos.shape[0])
    ):
        root_pos = root_pos - env_origins[: root_pos.shape[0]].to(root_pos.device, root_pos.dtype)
    root_parts = [
        root_pos,
        getattr(data, "root_quat_w", None),
        getattr(data, "root_lin_vel_w", None),
        getattr(data, "root_ang_vel_w", None),
    ]
    joint_parts = [getattr(data, name, None) for name in ("joint_pos", "joint_vel")]
    root_state = torch.cat(root_parts, dim=-1) if all(isinstance(item, torch.Tensor) for item in root_parts) else None
    joint_state = torch.cat(joint_parts, dim=-1) if all(isinstance(item, torch.Tensor) for item in joint_parts) else None
    batch_size = int(sum(int(getattr(pair_layout, name, 0)) for name in ("n_train", "n_candidate", "n_base", "n_clean")))

    def pair_error(value: torch.Tensor | None) -> dict[str, str]:
        if value is None or value.ndim < 2 or int(value.shape[0]) < batch_size:
            return {"status": "missing_or_shape_mismatch"}
        slices = _role_slices(pair_layout, batch_size)
        policy = value[slices["policy"]].detach().float()
        result: dict[str, str] = {}
        for role, role_slice in slices.items():
            rows = value[role_slice].detach().float()
            count = min(int(policy.shape[0]), int(rows.shape[0]))
            if count == 0:
                result[role] = "count=0"
                continue
            delta = (rows[:count] - policy[:count]).abs()
            result[role] = f"count={count} max={delta.max().item():.6g} mean={delta.mean().item():.6g}"
        return result

    return {"root_pair_error": pair_error(root_state), "joint_pair_error": pair_error(joint_state)}


def snapshot_termination_terms(runner: Any, pair_layout: Any, *, batch_size: int) -> dict[str, Any]:
    """读取 env.step 当前时刻的 active termination term masks."""
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        return {"status": "missing_termination_manager"}
    names = tuple(getattr(manager, "active_terms", ()) or getattr(manager, "_term_names", ()) or ())
    get_term = getattr(manager, "get_term", None)
    term_dones = getattr(manager, "_term_dones", None)
    result: dict[str, Any] = {}
    for index, name in enumerate(names):
        value = get_term(name) if callable(get_term) else None
        if value is None and isinstance(term_dones, torch.Tensor) and term_dones.ndim == 2:
            if index < int(term_dones.shape[1]):
                value = term_dones[:, index]
        result[str(name)] = _role_true_counts(value, pair_layout, batch_size=batch_size)
    return result or {"status": "no_active_terms"}


def print_reset_lifecycle_audit(
    runner: Any,
    *,
    pair_layout: Any,
    phase: str,
    episode_before: Any = None,
    episode_randomized: Any = None,
    episode_after_reset: Any = None,
    pair_state: Mapping[str, Any] | None = None,
    rollout_step: int | None = None,
    dones: Any = None,
    time_outs: Any = None,
    terminated: Any = None,
    alive: Any = None,
    survival_steps: Any = None,
    first_done_step: Any = None,
    termination_terms: Mapping[str, Any] | None = None,
) -> None:
    """Emit role-aware reset and termination facts without changing rollout state."""
    if not formal_runtime_audit_enabled(runner):
        return
    batch_size = int(sum(int(getattr(pair_layout, name, 0)) for name in ("n_train", "n_candidate", "n_base", "n_clean")))
    values: dict[str, Any] = {"phase": phase, "batch_size": batch_size}
    if phase == "reset":
        values.update(
            episode_before=_role_tensor_stats(episode_before, pair_layout, batch_size=batch_size),
            episode_randomized=_role_tensor_stats(episode_randomized, pair_layout, batch_size=batch_size),
            episode_after_reset=_role_tensor_stats(episode_after_reset, pair_layout, batch_size=batch_size),
            **dict(pair_state or {}),
        )
    elif phase == "step":
        values.update(
            rollout_step=rollout_step,
            done=_role_true_counts(dones, pair_layout, batch_size=batch_size),
            time_out=_role_true_counts(time_outs, pair_layout, batch_size=batch_size),
            terminated=_role_true_counts(terminated, pair_layout, batch_size=batch_size),
            alive=_role_true_counts(alive, pair_layout, batch_size=batch_size),
            survival=_role_tensor_stats(survival_steps, pair_layout, batch_size=batch_size),
            termination_terms=dict(termination_terms or {}),
        )
    elif phase == "final":
        values["first_done_step"] = _role_tensor_stats(first_done_step, pair_layout, batch_size=batch_size)
    emit_formal_runtime_probe("AUDIT-RESET-LIFECYCLE-01", limit=128, **values)


def print_formal_route_audit(runner: Any, *, num_learning_iterations: int) -> None:
    if not formal_runtime_audit_enabled(runner):
        return
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    alg = getattr(runner, "alg", None)
    policy = getattr(alg, "policy", None)
    # AUDIT-PERTURB-01: 检查正式 Stage 3 扰动配置, 位于 stage preset -> sampler/rollout.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot(
        "AUDIT-PERTURB-01",
        specialist_mode=_config_value(getattr(runner, "cfg", None), "frontres_specialist_mode"),
        perturbation_channels=_config_value(getattr(runner, "cfg", None), "frontres_perturbation_channels"),
        dr_scale=getattr(runner, "_dr_scale", _config_value(getattr(runner, "cfg", None), "dr_scale_init")),
        max_horizon_k=getattr(alg, "frontres_segment_max_horizon_k", "missing"),
    )
    # AUDIT-HSL-LOAD-01: 检查 HSL actor 与 normalizer 已进入 Stage 3, 位于 checkpoint load -> live policy.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot(
        "AUDIT-HSL-LOAD-01",
        policy=type(policy).__name__,
        actor=type(getattr(policy, "residual_actor", None)).__name__,
        obs_normalizer=type(getattr(runner, "obs_normalizer", None)).__name__,
    )
    alternate = any(
        bool(getattr(boundary, name, False))
        for name in (
            "live_sentinel_only", "live_probe_only", "live_storage_write_only", "live_single_update_only",
            "live_update_loop_only", "offline_eval_only", "sequence_offline_eval_only",
        )
    )
    assert getattr(runner.alg, "frontres_training_objective", "") == "segment_replay_hrl"
    assert bool(getattr(boundary, "live_train_enabled", False)) and not alternate
    print(
        "[AUDIT-ROUTE-01] "
        f"objective={getattr(runner.alg, 'frontres_training_objective', 'missing')} "
        f"live_train={int(bool(getattr(boundary, 'live_train_enabled', False)))} "
        f"alternate_modes={int(alternate)} "
        f"iterations={int(num_learning_iterations)} current_iter={int(getattr(runner, 'current_learning_iteration', 0))} "
        f"checkpoint={getattr(runner, '_frontres_last_loaded_checkpoint_path', 'cold_start')}",
        flush=True,
    )


def print_sampler_audit(runner: Any, *, update_step: int, sample: Any, batch: Any, summary: Mapping[str, Any]) -> None:
    if not formal_runtime_audit_enabled(runner):
        return
    segment_ids = getattr(sample, "segment_ids", None)
    horizon_k = getattr(sample, "horizon_k", None)
    assert isinstance(segment_ids, torch.Tensor) and segment_ids.numel() > 0
    assert isinstance(horizon_k, torch.Tensor) and horizon_k.numel() == segment_ids.numel()
    assert bool((horizon_k > 0).all().item())
    # AUDIT-SEGDATA-01: 检查 cache/source/segment 身份, 位于 dataset lookup -> sampler sample.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot(
        "AUDIT-SEGDATA-01",
        segment_ids=_tensor_stats(segment_ids),
        source_index=_tensor_stats(getattr(sample, "source_index", None)),
    )
    # AUDIT-KPLAN-01: 检查 per-row K 与 rollout budget, 位于 sampler plan -> trial expansion.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot(
        "AUDIT-KPLAN-01",
        horizon_k=_tensor_stats(horizon_k),
        trial_roles=getattr(batch, "frontres_segment_trial_role", "missing"),
    )
    # AUDIT-KROLLOUT-01: 检查 reset/preroll/valid horizon, 位于 expanded trials -> scored rollout.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot(
        "AUDIT-KROLLOUT-01",
        reset_ok=_summary_value(summary, "reset_success_count"),
        valid=_summary_value(summary, "ppo_valid_count"),
        horizon_k=_tensor_stats(horizon_k),
    )
    print(
        "[AUDIT-SAMPLER-01] "
        f"update_step={update_step} segment_ids={_tensor_stats(getattr(sample, 'segment_ids', None))} "
        f"source_index={_tensor_stats(getattr(sample, 'source_index', None))} "
        f"horizon_k={_tensor_stats(getattr(sample, 'horizon_k', None))} "
        f"trial_roles={getattr(batch, 'frontres_segment_trial_role', 'missing')} "
        f"reset_ok={_summary_value(summary, 'reset_success_count')} "
        f"valid={_summary_value(summary, 'ppo_valid_count')} "
        f"priority_before={_summary_value(summary, 'sampler_update_priority_before_mean')} "
        f"priority_after={_summary_value(summary, 'sampler_update_priority_after_mean')}",
        flush=True,
    )


def print_rollout_storage_audit(
    runner: Any,
    *,
    capture: Any,
    summary: Mapping[str, Any],
    storage_batch: Any | None,
) -> None:
    if not formal_runtime_audit_enabled(runner):
        return
    observations = getattr(capture, "transition_obs", None)
    actions = getattr(capture, "transition_actions", None)
    obs_prefix = observations[..., :100] if isinstance(observations, torch.Tensor) and observations.shape[-1] >= 100 else None
    obs_suffix = observations[..., 100:] if isinstance(observations, torch.Tensor) and observations.shape[-1] >= 100 else None
    assert isinstance(observations, torch.Tensor) and observations.shape[-1] == 870
    assert isinstance(actions, torch.Tensor) and actions.shape[-1] == 6
    assert storage_batch is not None and getattr(storage_batch, "actions", None).shape[-1] == 6
    for value in (
        observations,
        actions,
        getattr(capture, "transition_means", None),
        getattr(capture, "transition_sigmas", None),
        getattr(storage_batch, "actions", None),
        getattr(storage_batch, "old_means", None),
        getattr(storage_batch, "old_sigmas", None),
        getattr(storage_batch, "returns", None),
    ):
        assert isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all().item())
    # AUDIT-PERTURB-02: 检查实际 rollout 扰动, 位于 perturbation application -> paired execution.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-PERTURB-02", perturb_rp=_tensor_stats(getattr(capture, "transition_perturbation_rp", None)), family=_summary_value(summary, "perturbation_family"), strength=_summary_value(summary, "perturbation_strength"))
    # AUDIT-OBS-01: 检查 100D balance + 770D GMT observation, 位于 env observation -> policy normalizer.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-OBS-01", obs=_tensor_stats(observations), prefix100=_tensor_stats(obs_prefix), suffix770=_tensor_stats(obs_suffix))
    # AUDIT-ACTION-01: 检查 full-6D distribution/action, 位于 policy distribution -> executed repair.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-ACTION-01", mean=_tensor_stats(getattr(capture, "transition_means", None)), sigma=_tensor_stats(getattr(capture, "transition_sigmas", None)), action=_tensor_stats(actions))
    # AUDIT-APPLY-01: 检查 executed Delta SE(3), 位于 task correction -> repaired reference.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-APPLY-01", action=_tensor_stats(actions), delta_norm=_summary_value(summary, "delta_se_norm"))
    # AUDIT-GMT-01: 检查 frozen GMT observation/execution, 位于 repaired reference -> GMT rollout.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-GMT-01", gmt_obs=_tensor_stats(obs_suffix), normalizer=type(getattr(runner, "obs_normalizer", None)).__name__)
    # AUDIT-PAIR-01: 检查 quartet roles 与有效行, 位于 trial plan -> paired rollout.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-PAIR-01", roles=_summary_value(summary, "trial_roles"), valid=_summary_value(summary, "ppo_valid_count"))
    # AUDIT-PAIR-EVIDENCE-01: 检查同 segment/K 的 paired evidence, 位于 rollout capture -> Gain.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-PAIR-EVIDENCE-01", noisy=_summary_value(summary, "score_noisy"), repaired=_summary_value(summary, "score_repaired"), gain=_summary_value(summary, "score_gain"))
    # AUDIT-GAIN-01: 检查 canonical Gain 分解, 位于 paired evidence -> storage reward.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-GAIN-01", style=_summary_value(summary, "gain_style_mean"), physics=_summary_value(summary, "gain_physics_mean"), repair=_summary_value(summary, "gain_repair_cost_mean"), total=_summary_value(summary, "gain_total_mean"))
    # AUDIT-RETURN-01: 检查 Gain -> reward -> returns, 位于 storage write -> PPO batch.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-RETURN-01", rewards=_tensor_stats(getattr(storage_batch, "rewards", None)), returns=_tensor_stats(getattr(storage_batch, "returns", None)), advantages=_tensor_stats(getattr(storage_batch, "advantages", None)))
def print_ppo_audit(runner: Any, *, result: Any) -> None:
    if not formal_runtime_audit_enabled(runner):
        return
    policy = runner.alg.policy
    gmt_params = tuple(getattr(policy, "gmt_policy", torch.nn.Module()).parameters())
    optimizer_ids = {
        id(param)
        for group in getattr(runner.alg.optimizer, "param_groups", ())
        for param in group.get("params", ())
    }
    valid_count = int(getattr(result, "valid_count", 0))
    update_observed = int(valid_count > 0)
    assert valid_count >= 0
    assert bool(torch.isfinite(result.total_loss).all().item())
    assert all(not param.requires_grad and id(param) not in optimizer_ids for param in gmt_params)
    # AUDIT-WARMUP-01: 检查 critic-only/actor-ramp/joint phase, 位于 warmup owner -> PPO loss.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot(
        "AUDIT-WARMUP-01",
        phase=getattr(result, "warmup_phase", "missing"),
        phase_iter=getattr(result, "warmup_phase_iteration", "missing"),
        actor_weight=getattr(result, "actor_loss_weight", "missing"),
    )
    # AUDIT-DIAG-01: 检查 diagnostics 来自最终 accepted update, 位于 PPO result -> live summary.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot(
        "AUDIT-DIAG-01",
        valid=valid_count,
        update_observed=update_observed,
        post_kl=getattr(result, "post_update_distribution_kl_mean", "missing"),
        accepted=getattr(result, "trust_region_accepted", "missing"),
    )
    print(
        "[AUDIT-PPO-01] "
        f"phase={getattr(result, 'warmup_phase', 'missing')} "
        f"phase_iter={getattr(result, 'warmup_phase_iteration', 'missing')} "
        f"actor_weight={getattr(result, 'actor_loss_weight', 'missing')} "
        f"valid={valid_count} update_observed={update_observed} "
        f"loss={float(result.total_loss.detach().cpu().item()):.6g} "
        f"grad_norm={getattr(result, 'param_grad_norm', 'missing')} "
        f"param_delta={getattr(result, 'param_delta_l2', 'missing')} "
        f"pre_kl={getattr(result, 'distribution_kl_mean', 'missing')} "
        f"post_kl={getattr(result, 'post_update_distribution_kl_mean', 'missing')} "
        f"trust_accepted={getattr(result, 'trust_region_accepted', 'missing')} "
        f"gmt_params={len(gmt_params)} gmt_trainable={sum(int(p.requires_grad) for p in gmt_params)} "
        f"gmt_in_optimizer={sum(int(id(p) in optimizer_ids) for p in gmt_params)}",
        flush=True,
    )


def print_checkpoint_payload_audit(runner: Any, *, path: str, payload: Mapping[str, Any]) -> None:
    if not formal_runtime_audit_enabled(runner):
        return
    required = (
        "model_state_dict",
        "optimizer_state_dict",
        "iter",
        "obs_norm_state_dict",
        "frontres_segment_sampler_state_dict",
        "frontres_gain_config",
        "frontres_segment_warmup_config",
    )
    missing = [key for key in required if key not in payload]
    assert not missing, f"formal Stage 3 checkpoint missing audit fields: {missing}"
    print(
        "[AUDIT-PERSIST-01] "
        f"path={path} iter={payload.get('iter', 'missing')} "
        f"model={int('model_state_dict' in payload)} optimizer={int('optimizer_state_dict' in payload)} "
        f"obs_norm={int('obs_norm_state_dict' in payload)} sampler={int('frontres_segment_sampler_state_dict' in payload)} "
        f"gain_config={int('frontres_gain_config' in payload)} "
        f"warmup={payload.get('frontres_segment_warmup_config', 'missing')}",
        flush=True,
    )


__all__ = [
    "formal_runtime_audit_enabled",
    "print_checkpoint_payload_audit",
    "print_formal_route_audit",
    "print_ppo_audit",
    "print_rollout_storage_audit",
    "print_sampler_audit",
]
