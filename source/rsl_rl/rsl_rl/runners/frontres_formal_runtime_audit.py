from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from rsl_rl.frontres.frontres_value_normalization import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FRONTRES_VALUE_NORMALIZER_DECAY,
    FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
    FrontRESValueNormalizerState,
)

from rsl_rl.frontres.frontres_formal_runtime_probe import configure_formal_runtime_probe, emit_formal_runtime_probe


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


def _policy_gain_steps_for_audit(capture: Any, n_train: int) -> torch.Tensor | None:
    """Return only policy-owned Gain steps for the Card 17 audit snapshot."""

    gain_steps = getattr(capture, "gain_steps", None)
    if not isinstance(gain_steps, torch.Tensor) or gain_steps.ndim != 2:
        return gain_steps
    if int(gain_steps.shape[1]) < int(n_train):
        return gain_steps
    # B2: quartet evidence contains non-policy rows that may be NaN by design;
    # Card 17 must inspect the same policy-row domain consumed by storage.
    return gain_steps[:, : int(n_train)]


def _summary_value(summary: Mapping[str, Any], key: str) -> str:
    value = summary.get(key, "missing")
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return str(value)


def _finite_tensor_mean(value: Any) -> float | None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    flat = value.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    return float(flat.mean().item()) if flat.numel() else None


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
    required_identity = {
        "frontres_method_contract_id": "FRS-METHOD-v025",
        "frontres_gain_contract_id": "FRS-GAIN-v008",
        "frontres_optimization_contract_id": "FRS-PPO-v012",
        "frontres_training_contract_id": "FRS-TRAIN-v024",
        "frontres_critic_support_context_id": "action-pre-support-plan-kmax32-v1",
    }
    for name, expected in required_identity.items():
        assert getattr(alg, name, None) == expected, f"AUDIT-B01 requires {name}={expected}"
    offsets = tuple(int(value) for value in (getattr(alg, "frontres_future_offsets", ()) or ()))
    checkpoint_path = str(getattr(runner, "_frontres_last_loaded_checkpoint_path", ""))
    assert offsets == (1, 2), "AUDIT-B01 requires future offsets (1,2)"
    assert checkpoint_path, "AUDIT-B01 requires a validated HSL-v2 initializer path"
    optimizer_groups = tuple(getattr(getattr(alg, "optimizer", None), "param_groups", ()) or ())
    optimizer_by_role = {
        str(group.get("frontres_role", "")): group
        for group in optimizer_groups
        if isinstance(group, Mapping)
    }
    assert len(optimizer_groups) == 2 and set(optimizer_by_role) == {"actor", "critic"}, (
        "AUDIT-B01 requires exact Actor/Critic optimizer groups"
    )
    actor_lr = float(optimizer_by_role["actor"].get("lr", float("nan")))
    critic_lr = float(optimizer_by_role["critic"].get("lr", float("nan")))
    assert actor_lr == 3.0e-7 and critic_lr == 1.0e-5, (
        "AUDIT-B01 requires initial Actor/Critic LR=3e-7/1e-5"
    )
    value_normalizer_state = getattr(alg, "frontres_critic_value_normalizer_state", None)
    assert getattr(alg, "frontres_critic_value_normalization", None) == FRONTRES_VALUE_NORMALIZATION_ID
    assert float(getattr(alg, "frontres_critic_value_normalizer_decay", float("nan"))) == FRONTRES_VALUE_NORMALIZER_DECAY
    assert float(getattr(alg, "frontres_critic_value_normalizer_scale_floor", float("nan"))) == FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR
    assert isinstance(value_normalizer_state, FrontRESValueNormalizerState)
    value_normalizer_state.validate()
    assert value_normalizer_state.update_count == int(getattr(runner, "current_learning_iteration", -1))
    assert int(num_learning_iterations) == 1, (
        "AUDIT-B01 is admitted only for one bounded transaction invocation"
    )
    # AUDIT-B01: 检查正式入口身份, 位于 HSL-v2 load -> ordinary Stage3 train.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-B01",
        limit=1,
        checkpoint=checkpoint_path,
        contracts="FRS-METHOD-v025/FRS-GAIN-v008/FRS-PPO-v012/FRS-TRAIN-v024",
        future_offsets=offsets,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        critic_value_normalization=FRONTRES_VALUE_NORMALIZATION_ID,
        critic_value_normalizer_decay=FRONTRES_VALUE_NORMALIZER_DECAY,
        critic_value_normalizer_scale_floor=FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
        critic_value_normalizer_update_count=value_normalizer_state.update_count,
        active_k=8,
        active_m=4,
        envs=64,
        iterations=int(num_learning_iterations),
        alternate_modes=int(alternate),
    )
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
    """Emit the retained legacy sampler snapshot.

    TRAIN-v018 K/M identity is owned by ``frontres_segment_warmup.py`` and the
    sealed formal transaction. This compatibility projection cannot prove the
    active K-step Curriculum.
    """
    if not formal_runtime_audit_enabled(runner):
        return
    segment_ids = getattr(sample, "segment_ids", None)
    horizon_k = getattr(sample, "horizon_k", None)
    assert isinstance(segment_ids, torch.Tensor) and segment_ids.numel() > 0
    assert isinstance(horizon_k, torch.Tensor) and horizon_k.numel() == segment_ids.numel()
    assert bool((horizon_k > 0).all().item())
    # AUDIT-SEGDATA-01: 检查 cache/source/segment 身份, 位于 dataset lookup -> sampler sample.
    # Result: E67 LIVE OBSERVED for one sample: cache_horizon_k=4 is the Stage 1
    # cache window; effective training K=8 is reported by K plan/rollout.
    _emit_owner_snapshot(
        "AUDIT-SEGDATA-01",
        segment_ids=_tensor_stats(segment_ids),
        source_index=_tensor_stats(getattr(sample, "source_index", None)),
    )
    print(
        "[AUDIT-SAMPLER-01] "
        f"update_step={update_step} segment_ids={_tensor_stats(getattr(sample, 'segment_ids', None))} "
        f"source_index={_tensor_stats(getattr(sample, 'source_index', None))} "
        f"horizon_k={_tensor_stats(getattr(sample, 'horizon_k', None))} "
        f"trial_roles={getattr(batch, 'frontres_segment_trial_role', 'missing')} "
        f"reset_success_frac={_summary_value(summary, 'segment_reset_success_frac')} "
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
    """Emit the retained legacy flat-probe snapshot.

    The active v017 transaction is audited by
    :func:`print_segment_replay_transaction_audit`; this function must not be
    used as evidence for the active 928D/grouped-scalar route.
    """
    if not formal_runtime_audit_enabled(runner):
        return
    observations = getattr(capture, "transition_obs", None)
    actions = getattr(capture, "transition_actions", None)
    raw_survival_steps = getattr(capture, "survival_steps", None)
    effective_horizon_k = getattr(capture, "horizon_k", None)
    survival_gain_steps = getattr(capture, "survival_gain_steps", None)
    obs_prefix = observations[..., :100] if isinstance(observations, torch.Tensor) and observations.shape[-1] >= 100 else None
    obs_suffix = observations[..., 100:] if isinstance(observations, torch.Tensor) and observations.shape[-1] >= 100 else None
    assert isinstance(observations, torch.Tensor) and observations.shape[-1] == 870
    assert isinstance(actions, torch.Tensor) and actions.shape[-1] == 6
    assert storage_batch is not None and getattr(storage_batch, "actions", None).shape[-1] == 6
    assert isinstance(raw_survival_steps, torch.Tensor)
    assert isinstance(effective_horizon_k, torch.Tensor)
    assert isinstance(survival_gain_steps, torch.Tensor)
    assert bool(torch.isfinite(raw_survival_steps.float()).all().item())
    assert bool(torch.isfinite(effective_horizon_k.float()).all().item())
    n_train = int(getattr(capture, "n_train", 0))
    if survival_gain_steps.ndim != 2 or int(survival_gain_steps.shape[1]) < n_train:
        raise ValueError("formal Gain audit requires [T,B] survival_gain_steps with policy rows")
    policy_survival_gain_steps = survival_gain_steps[:, :n_train]
    assert bool(torch.isfinite(policy_survival_gain_steps.float()).all().item())
    policy_survival_gain_sum = policy_survival_gain_steps.float().sum(dim=0)
    step_sum_mean = _finite_tensor_mean(policy_survival_gain_sum)
    final_survival_gain_mean = summary.get("gain_physics_survival_mean")
    sum_abs_error: float | str = "missing"
    if step_sum_mean is not None and isinstance(final_survival_gain_mean, (int, float)) and math.isfinite(float(final_survival_gain_mean)):
        sum_abs_error = abs(step_sum_mean - float(final_survival_gain_mean))
    for value in (
        observations,
        actions,
        getattr(capture, "transition_means", None),
        getattr(capture, "transition_sigmas", None),
        getattr(storage_batch, "actions", None),
        getattr(storage_batch, "old_means", None),
        getattr(storage_batch, "old_sigmas", None),
        getattr(storage_batch, "rewards", None),
        getattr(storage_batch, "returns", None),
        getattr(storage_batch, "advantages", None),
    ):
        assert isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all().item())
    # AUDIT-PERTURB-02: 检查实际 rollout 扰动, 位于 perturbation application -> paired execution.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot(
        "AUDIT-PERTURB-02",
        perturb_rp=_tensor_stats(getattr(capture, "transition_perturbation_rp", None)),
        family_counts=_summary_value(summary, "perturbation_family_counts"),
        strength_min=_summary_value(summary, "perturbation_strength_min"),
        strength_mean=_summary_value(summary, "perturbation_strength_mean"),
        strength_max=_summary_value(summary, "perturbation_strength_max"),
    )
    # AUDIT-OBS-01: 检查 100D balance + 770D GMT observation, 位于 env observation -> policy normalizer.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-OBS-01", obs=_tensor_stats(observations), prefix100=_tensor_stats(obs_prefix), suffix770=_tensor_stats(obs_suffix))
    # AUDIT-ACTION-01: 检查 full-6D distribution/action, 位于 policy distribution -> executed repair.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-ACTION-01", mean=_tensor_stats(getattr(capture, "transition_means", None)), sigma=_tensor_stats(getattr(capture, "transition_sigmas", None)), action=_tensor_stats(actions))
    # AUDIT-APPLY-01: 检查 executed Delta SE(3), 位于 task correction -> repaired reference.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-APPLY-01", action=_tensor_stats(actions), delta_norm=_summary_value(summary, "motion_delta_se_norm"))
    # AUDIT-GMT-01: 检查 frozen GMT observation/execution, 位于 repaired reference -> GMT rollout.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-GMT-01", gmt_obs=_tensor_stats(obs_suffix), normalizer=type(getattr(runner, "obs_normalizer", None)).__name__)
    # AUDIT-PAIR-01: 检查 quartet roles 与有效行, 位于 trial plan -> paired rollout.
    # Result: PENDING_LIVE.
    _emit_owner_snapshot("AUDIT-PAIR-01", roles=_summary_value(summary, "trial_role_counts"), valid=_summary_value(summary, "ppo_valid_count"))
    # AUDIT-PAIR-EVIDENCE-01: 检查同 segment/K 的 paired evidence, 位于 rollout capture -> Gain.
    # Result: E67 LIVE PASS for one capture: paired evidence shares
    # iter0:capture1 and batch b21ee717d66475f3.
    _emit_owner_snapshot(
        "AUDIT-PAIR-EVIDENCE-01",
        noisy=_summary_value(summary, "score_noisy"),
        repaired=_summary_value(summary, "score_repaired"),
        gain=_summary_value(summary, "score_gain"),
        audit_transaction_id=_summary_value(summary, "audit_transaction_id"),
        audit_batch_signature=_summary_value(summary, "audit_batch_signature"),
        audit_identity_state=_summary_value(summary, "audit_identity_state"),
    )
    # AUDIT-GAIN-01: 检查 v002 Gain 分解和 survival unit, 位于 paired evidence -> storage reward.
    # Result: E68 LIVE PASS: mixed-K Gain components are finite and canonical
    # total is forwarded with the same transaction/batch identity.
    _emit_owner_snapshot(
        "AUDIT-GAIN-01",
        contract="FRS-GAIN-v002",
        raw_survival_steps=_tensor_stats(raw_survival_steps[:n_train]),
        effective_horizon_k=_tensor_stats(effective_horizon_k[:n_train]),
        survival_quality_repaired=_summary_value(summary, "gain_physics_survival_quality_repaired_per_sample"),
        survival_quality_noisy=_summary_value(summary, "gain_physics_survival_quality_noisy_per_sample"),
        physics_survival_gain=_summary_value(summary, "gain_physics_survival_per_sample"),
        survival_gain_step_sum=_tensor_stats(policy_survival_gain_sum),
        survival_gain_sum_mean=step_sum_mean if step_sum_mean is not None else "missing",
        final_survival_gain_mean=_summary_value(summary, "gain_physics_survival_mean"),
        survival_gain_sum_abs_error=sum_abs_error,
        style=_summary_value(summary, "gain_style_mean"),
        physics=_summary_value(summary, "gain_physics_mean"),
        repair=_summary_value(summary, "gain_repair_cost_mean"),
        total=_summary_value(summary, "gain_total_mean"),
        audit_transaction_id=_summary_value(summary, "audit_transaction_id"),
        audit_batch_signature=_summary_value(summary, "audit_batch_signature"),
        audit_identity_state=_summary_value(summary, "audit_identity_state"),
    )
    # AUDIT-RETURN-01: 检查 Gain -> reward -> returns, 位于 storage write -> PPO batch.
    # Result: E68 LIVE PASS: mixed-K gain_steps is finite [T,8], survival
    # step-sum error is 0, and returns/advantages are finite.
    _emit_owner_snapshot(
        "AUDIT-RETURN-01",
        raw_survival_steps=_tensor_stats(raw_survival_steps[:n_train]),
        effective_horizon_k=_tensor_stats(effective_horizon_k[:n_train]),
        survival_gain_steps=_tensor_stats(policy_survival_gain_steps),
        survival_gain_step_sum=_tensor_stats(policy_survival_gain_sum),
        gain_steps=_tensor_stats(_policy_gain_steps_for_audit(capture, n_train)),
        rewards=_tensor_stats(getattr(storage_batch, "rewards", None)),
        returns=_tensor_stats(getattr(storage_batch, "returns", None)),
        advantages=_tensor_stats(getattr(storage_batch, "advantages", None)),
        audit_transaction_id=getattr(storage_batch, "audit_transaction_id", "UNCONFIRMED"),
        audit_batch_signature=getattr(storage_batch, "audit_batch_signature", "UNCONFIRMED"),
        audit_identity_state=getattr(storage_batch, "audit_identity_state", "UNCONFIRMED"),
    )


def print_segment_replay_transaction_audit(runner: Any, *, result: Any) -> None:
    """Audit the committed v017 Segment Replay transaction without mutation."""

    if not formal_runtime_audit_enabled(runner):
        return
    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise AssertionError("active Segment Replay audit requires immutable transaction diagnostics")
    required_identity = {
        "method_contract_id": "FRS-METHOD-v025",
        "gain_contract_id": "FRS-GAIN-v008",
        "optimization_contract_id": "FRS-PPO-v012",
        "training_contract_id": "FRS-TRAIN-v024",
    }
    for key, expected in required_identity.items():
        assert diagnostics.get(key) == expected, f"active Segment Replay audit requires {key}={expected}"

    active_m = int(diagnostics.get("active_m", 0))
    segment_count = int(diagnostics.get("selected_segment_count", 0))
    policy_row_count = int(diagnostics.get("policy_row_count", 0))
    assert segment_count == 8, "active campaign requires exactly eight selected Segments"
    assert active_m > 0 and policy_row_count == segment_count * active_m, (
        "active Segment Replay audit requires exactly M Repair rows for each selected Segment"
    )
    assert int(getattr(result, "segment_count", -1)) == segment_count
    assert int(getattr(result, "source_count", -1)) == segment_count
    assert int(getattr(result, "policy_attempt_count", -1)) == policy_row_count
    assert int(getattr(result, "valid_row_count", -1)) == policy_row_count
    assert int(getattr(result, "optimizer_step_delta", -1)) == 1
    assert int(getattr(result, "update_invocation_count", -1)) == 1

    motion_mass = tuple(float(value) for value in diagnostics.get("grouped_motion_mass_shares", ()))
    segment_mass = tuple(float(value) for value in diagnostics.get("grouped_segment_mass_shares", ()))
    attempt_mass = tuple(float(value) for value in diagnostics.get("grouped_attempt_mass_shares", ()))
    for name, values, expected_count in (
        ("motion", motion_mass, None),
        ("Segment", segment_mass, segment_count),
        ("attempt", attempt_mass, policy_row_count),
    ):
        assert values and all(math.isfinite(value) and value > 0.0 for value in values), (
            f"active Segment Replay audit requires positive finite {name} voting weights"
        )
        assert math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6), (
            f"active Segment Replay {name} voting weights must sum to one"
        )
        expected_weight = 1.0 / float(len(values))
        assert all(math.isclose(value, expected_weight, rel_tol=0.0, abs_tol=1e-6) for value in values), (
            f"active Segment Replay requires equal {name} voting weights"
        )
        if expected_count is not None:
            assert len(values) == expected_count, f"active Segment Replay has wrong {name} voting-weight count"

    _emit_owner_snapshot(
        "AUDIT-SEGMENT-REPLAY-01",
        transaction_id=getattr(result, "transaction_id", "missing"),
        policy_snapshot_id=getattr(result, "policy_snapshot_id", "missing"),
        segments=segment_count,
        attempts_per_segment=active_m,
        policy_rows=policy_row_count,
        valid_rows=getattr(result, "valid_row_count", "missing"),
        motion_voting_weights=motion_mass,
        segment_voting_weights=segment_mass,
        attempt_voting_weights=attempt_mass,
        optimizer_step_delta=getattr(result, "optimizer_step_delta", "missing"),
        update_invocations=getattr(result, "update_invocation_count", "missing"),
        contracts="FRS-METHOD-v025/FRS-GAIN-v008/FRS-PPO-v012/FRS-TRAIN-v024",
    )


def _print_one_action_k_audit_facts(
    runner: Any,
    *,
    roles: tuple[str, ...],
    provenance: tuple[str, ...],
    sources: tuple[str, ...],
    actions: torch.Tensor,
    horizon_k: torch.Tensor,
    gmt_action_shapes: tuple[tuple[int, ...], ...],
    gmt_actions_finite: bool,
    actor_forward_count: int,
    later_femr_action_count: int,
) -> None:
    from rsl_rl.runners.frontres_segment_runtime_types import frontres_observation_trace

    trace = dict(frontres_observation_trace(runner))
    expected_trace = {
        "current_command_dim": 58,
        "raw_observation_dim": 870,
        "q29_tail_dim": 58,
        "combined_observation_dim": 928,
        "normalized_observation_dim": 928,
        "femr_visible_dim": 158,
        "gmt_suffix_dim": 770,
        "gmt_input_dim": 770,
        "critic_current_observation_dim": 289,
        "critic_future_intent_dim": 58,
        "critic_support_context_dim": 102,
        "critic_observation_dim": 449,
    }
    for name, expected in expected_trace.items():
        assert int(trace.get(name, -1)) == expected, f"AUDIT-B03 requires {name}={expected}"
    for name in ("actor_segment_state_max_abs_diff", "critic_segment_state_max_abs_diff"):
        value = float(trace.get(name, float("nan")))
        assert value == 0.0, f"AUDIT-B03 requires exact shared {name}"
    for name in ("actor_raw_observation_max_abs_diff", "critic_raw_observation_max_abs_diff"):
        value = float(trace.get(name, float("nan")))
        assert math.isfinite(value) and value >= 0.0, f"AUDIT-B03 requires finite {name}"

    role_count = len(roles)
    repair_count = roles.count("repair")
    noisy_count = roles.count("noisy")
    assert role_count > 0 and repair_count == noisy_count and role_count == repair_count + noisy_count
    assert roles == ("repair",) * repair_count + ("noisy",) * noisy_count
    assert int(trace.get("role_row_count", -1)) == role_count
    assert len(provenance) == len(sources) == role_count
    assert all(value == "deployment_noisy_q29" for value in provenance)
    assert all(value and not any(token in value for token in ("clean", "root", "global")) for value in sources)

    assert isinstance(actions, torch.Tensor) and tuple(actions.shape) == (repair_count, 6)
    assert bool(torch.isfinite(actions).all().item())
    assert isinstance(horizon_k, torch.Tensor) and tuple(horizon_k.shape) == (role_count,)
    horizon_values = tuple(int(value) for value in torch.unique(horizon_k.detach()).cpu().tolist())
    assert len(horizon_values) == 1 and horizon_values[0] > 0
    active_k = horizon_values[0]
    assert len(gmt_action_shapes) == active_k and all(shape == (role_count, 29) for shape in gmt_action_shapes)
    assert gmt_actions_finite
    assert actor_forward_count == 1
    assert later_femr_action_count == 0
    # This counts one fresh frozen-GMT observation read per horizon step, not
    # the number of parallel role rows. M therefore must not scale it.
    assert int(trace.get("post_advance_gmt_read_count", -1)) == active_k

    policy = getattr(getattr(runner, "alg", None), "policy", None)
    alg = getattr(runner, "alg", None)
    assert getattr(alg, "frontres_critic_value_kind", None) == "state_value"
    assert getattr(alg, "frontres_critic_input_dim", None) == 449
    assert getattr(alg, "frontres_critic_support_context_id", None) == "action-pre-support-plan-kmax32-v1"
    assert getattr(alg, "frontres_critic_action_conditioned", None) is False
    assert getattr(alg, "frontres_critic_target_id", None) == "scenario-current-exact-m4-mean-symlog-v1"
    assert int(trace["critic_future_intent_dim"]) == int(trace["q29_tail_dim"])
    gmt_policy = getattr(policy, "gmt_policy", None)
    assert isinstance(gmt_policy, torch.nn.Module) and not gmt_policy.training
    assert all(not parameter.requires_grad for parameter in gmt_policy.parameters())

    # AUDIT-B03: 检查角色、Noisy provenance 与 158/449/770 权限分割.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-B03",
        limit=8,
        roles=roles,
        provenance=provenance,
        raw=trace["raw_observation_dim"],
        q29_tail=trace["q29_tail_dim"],
        combined=trace["combined_observation_dim"],
        actor=trace["femr_visible_dim"],
        critic_current=trace["critic_current_observation_dim"],
        critic_future=trace["critic_future_intent_dim"],
        critic_support=trace["critic_support_context_dim"],
        critic=trace["critic_observation_dim"],
        gmt=trace["gmt_suffix_dim"],
        actor_state_max_abs_diff=trace["actor_segment_state_max_abs_diff"],
        critic_state_max_abs_diff=trace["critic_segment_state_max_abs_diff"],
        actor_raw_max_abs_diff=trace["actor_raw_observation_max_abs_diff"],
        critic_raw_max_abs_diff=trace["critic_raw_observation_max_abs_diff"],
        critic_kind=getattr(alg, "frontres_critic_value_kind"),
        action_conditioned=int(bool(getattr(alg, "frontres_critic_action_conditioned"))),
    )
    # AUDIT-B04: 检查一次 FEMR action 后仅 frozen GMT 执行 K8.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-B04",
        limit=8,
        action_shape=tuple(actions.shape),
        action_finite=1,
        actor_forward_count=actor_forward_count,
        later_femr_action_count=later_femr_action_count,
        horizon_k=active_k,
        gmt_steps=len(gmt_action_shapes),
        gmt_action_shape=gmt_action_shapes[0],
        gmt_eval=int(not gmt_policy.training),
        gmt_trainable=sum(int(parameter.requires_grad) for parameter in gmt_policy.parameters()),
    )


def print_one_action_k_audit(runner: Any, *, evidence: Any) -> None:
    """Adapt legacy evaluator evidence to the shared B03/B04 assertions."""

    if not formal_runtime_audit_enabled(runner):
        return
    continuation = getattr(evidence, "continuation", None)
    gmt_actions = getattr(evidence, "frozen_gmt_env_actions", None)
    roles = tuple(getattr(evidence, "roles", ()))
    assert isinstance(continuation, torch.Tensor) and continuation.ndim == 3
    assert int(continuation.shape[0]) == len(roles) and int(continuation.shape[2]) == 65
    assert isinstance(gmt_actions, torch.Tensor) and gmt_actions.ndim == 3
    assert tuple(gmt_actions.shape) == (len(roles), int(continuation.shape[1]), 29)
    _print_one_action_k_audit_facts(
        runner,
        roles=roles,
        provenance=tuple(getattr(evidence, "intent_q29_provenance", ())),
        sources=tuple(str(value).lower() for value in getattr(evidence, "intent_q29_source", ())),
        actions=getattr(evidence, "policy_actions", None),
        horizon_k=getattr(evidence, "horizon_k", None),
        gmt_action_shapes=tuple(tuple(gmt_actions[:, step, :].shape) for step in range(gmt_actions.shape[1])),
        gmt_actions_finite=bool(torch.isfinite(gmt_actions).all().item()),
        actor_forward_count=int(getattr(evidence, "actor_forward_count", -1)),
        later_femr_action_count=int(getattr(evidence, "later_femr_action_count", -1)),
    )


def print_v017_repair_attempts_audit(
    runner: Any,
    *,
    roles: tuple[str, ...],
    provenance: tuple[str, ...],
    sources: tuple[str, ...],
    policy_actions: torch.Tensor,
    horizon_k: torch.Tensor,
    gmt_action_shapes: tuple[tuple[int, ...], ...],
    gmt_actions_finite: bool,
) -> None:
    """Consume B03/B04 facts at the active formal v017 Repair collector boundary."""

    if not formal_runtime_audit_enabled(runner):
        return
    assert getattr(runner, "_frontres_v015_one_action_k_phase", None) == "frozen", (
        "AUDIT-B04 requires one FEMR action followed by the frozen-GMT phase"
    )
    _print_one_action_k_audit_facts(
        runner,
        roles=roles,
        provenance=provenance,
        sources=tuple(str(value).lower() for value in sources),
        actions=policy_actions,
        horizon_k=horizon_k,
        gmt_action_shapes=gmt_action_shapes,
        gmt_actions_finite=gmt_actions_finite,
        actor_forward_count=1,
        later_femr_action_count=0,
    )


def print_phase_b_telemetry_audit(runner: Any, *, telemetry: Mapping[str, Any]) -> None:
    """Validate the final immutable serializer used by the formal transaction."""

    if not formal_runtime_audit_enabled(runner):
        return
    rows = int(telemetry.get("policy_row_count", -1))
    active_m = int(telemetry.get("active_m", -1))
    segment_count = int(telemetry.get("selected_segment_count", -1))
    source_index = tuple(int(value) for value in telemetry.get("source_index", ()))
    trial_index = tuple(int(value) for value in telemetry.get("trial_index", ()))
    scenario_ids = tuple(telemetry.get("scenario_ids", ()))
    noisy_hashes = tuple(telemetry.get("noisy_segment_hashes", ()))
    assert segment_count == 8 and active_m == 4 and rows == segment_count * active_m
    assert int(telemetry.get("role_row_count", -1)) == 2 * rows
    assert len(source_index) == len(trial_index) == len(scenario_ids) == len(noisy_hashes) == rows
    unique_sources = sorted(set(source_index))
    assert unique_sources == list(range(segment_count))
    for source in unique_sources:
        indices = [index for index, value in enumerate(source_index) if value == source]
        assert len(indices) == active_m and sorted(trial_index[index] for index in indices) == list(range(active_m))
        assert len({scenario_ids[index] for index in indices}) == 1
        assert len({noisy_hashes[index] for index in indices}) == 1

    # AUDIT-B02: 检查八个 sealed Scenario 各有 exact M=4 attempts.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-B02",
        limit=1,
        transaction_id=telemetry.get("transaction_id"),
        source_index=source_index,
        trial_index=trial_index,
        scenario_ids=scenario_ids,
        noisy_hashes=noisy_hashes,
        policy_rows=rows,
        role_rows=telemetry["role_row_count"],
    )

    gain_fields = (
        "intent_remaining_noisy",
        "intent_remaining_repaired",
        "physics_remaining_noisy",
        "physics_remaining_repaired",
        "intent_gain",
        "physics_gain",
        "recovery_pressure",
        "weighted_physics_gain",
        "repair_cost",
        "repair_penalty",
        "cost_free_score",
        "gain_total",
    )
    for name in gain_fields:
        values = tuple(float(value) for value in telemetry.get(name, ()))
        assert len(values) == rows and all(math.isfinite(value) for value in values), (
            f"AUDIT-B05 requires {rows} finite {name} rows"
        )
    assert tuple(telemetry.get("clean_execution_count", ())) == (1,) * segment_count
    assert tuple(telemetry.get("noisy_execution_count", ())) == (1,) * segment_count
    # AUDIT-B05: 检查 Clean/Noisy baseline 与三十二条完整 v008 Repair Gain.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-B05",
        limit=1,
        clean=telemetry["clean_execution_count"],
        noisy=telemetry["noisy_execution_count"],
        repair=rows,
        intent_gain=telemetry["intent_gain"],
        physics_gain=telemetry["physics_gain"],
        physics_remaining_noisy=telemetry["physics_remaining_noisy"],
        physics_remaining_repaired=telemetry["physics_remaining_repaired"],
        pressure=telemetry["recovery_pressure"],
        weighted_physics_gain=telemetry["weighted_physics_gain"],
        repair_cost=telemetry["repair_cost"],
        repair_penalty=telemetry["repair_penalty"],
        gain_total=telemetry["gain_total"],
    )

    gains = tuple(float(value) for value in telemetry["gain_total"])
    assert int(telemetry.get("active_k", -1)) == 8
    assert tuple(bool(value) for value in telemetry.get("valid_policy_row_mask", ())) == (True,) * rows
    critic_targets = tuple(float(value) for value in telemetry.get("critic_value_targets", ()))
    segment_targets = tuple(float(value) for value in telemetry.get("critic_segment_target_means", ()))
    actor_advantages = tuple(float(value) for value in telemetry.get("actor_advantages", ()))
    raw_returns = tuple(float(value) for value in telemetry.get("raw_returns", ()))
    utility_returns = tuple(float(value) for value in telemetry.get("utility_returns", ()))
    policy_values = tuple(float(value) for value in telemetry.get("policy_values", ()))
    raw_advantages = tuple(float(value) for value in telemetry.get("raw_advantages", ()))
    assert all(
        len(values) == rows and all(math.isfinite(value) for value in values)
        for values in (critic_targets, actor_advantages, raw_returns, utility_returns, policy_values, raw_advantages)
    )
    assert telemetry.get("return_utility_id") == "symmetric-log-gain-g0-1-v1"
    assert float(telemetry.get("return_utility_scale", float("nan"))) == 1.0
    assert raw_returns == gains
    assert len(segment_targets) == len(unique_sources) and all(math.isfinite(value) for value in segment_targets)
    expected_segment_targets: list[float] = []
    for segment_position, source in enumerate(unique_sources):
        indices = [index for index, value in enumerate(source_index) if value == source]
        expected_target = float(torch.tensor([utility_returns[index] for index in indices], dtype=torch.float32).mean().item())
        expected_segment_targets.append(expected_target)
        assert math.isclose(segment_targets[segment_position], expected_target, rel_tol=0.0, abs_tol=1e-6)
        assert all(
            math.isclose(critic_targets[index], expected_target, rel_tol=0.0, abs_tol=1e-6)
            for index in indices
        )
    for row in range(rows):
        assert math.isclose(raw_advantages[row], gains[row] - policy_values[row], rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(
            actor_advantages[row], utility_returns[row] - policy_values[row], rel_tol=0.0, abs_tol=1e-6
        )

    # AUDIT-B06: 检查 exact-M state-value target 与逐 attempt Actor advantage.
    expected_returns = torch.tensor(gains, dtype=torch.float32)
    expected_return_mean = float(expected_returns.mean().item())
    expected_return_min = float(expected_returns.min().item())
    expected_return_max = float(expected_returns.max().item())
    assert math.isclose(float(telemetry["return_mean"]), expected_return_mean, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(telemetry["return_min"]), expected_return_min, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(telemetry["return_max"]), expected_return_max, rel_tol=0.0, abs_tol=1e-6)
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-B06",
        limit=1,
        policy_rows=rows,
        active_k=telemetry.get("active_k"),
        valid=telemetry.get("valid_policy_row_mask"),
        source_index=source_index,
        gain_total=gains,
        policy_values=policy_values,
        segment_targets=segment_targets,
        critic_targets=critic_targets,
        actor_advantages=actor_advantages,
        return_mean=telemetry["return_mean"],
        return_min=telemetry["return_min"],
        return_max=telemetry["return_max"],
    )

    assert bool(telemetry.get("grouped_reduction_active", False))
    assert tuple(float(value) for value in telemetry.get("grouped_segment_mass_shares", ())) == (
        1.0 / segment_count,
    ) * segment_count
    assert tuple(float(value) for value in telemetry.get("grouped_attempt_mass_shares", ())) == (
        1.0 / rows,
    ) * rows
    assert int(telemetry.get("optimizer_step_delta", -1)) == 1
    assert int(telemetry.get("update_count", -1)) == 1
    outer_sources = tuple(telemetry.get("outer_replay_sources", ()))
    outer_keys = tuple(telemetry.get("outer_replay_scenario_key_digests", ()))
    outer_score_kind = str(telemetry.get("outer_replay_score_kind", ""))
    outer_calibration_values = tuple(
        float(value) for value in telemetry.get("outer_replay_critic_calibration_values", ())
    )
    outer_spread_values = tuple(
        float(value) for value in telemetry.get("outer_replay_repair_spread_values", ())
    )
    outer_priority_scores = tuple(
        float(value) for value in telemetry.get("outer_replay_priority_scores", ())
    )
    outer_target_means = tuple(
        float(value) for value in telemetry.get("outer_replay_critic_target_means", ())
    )
    outer_outcome_variances = tuple(
        float(value) for value in telemetry.get("outer_replay_outcome_variances", ())
    )
    outer_standard_errors = tuple(
        float(value) for value in telemetry.get("outer_replay_standard_errors", ())
    )
    outer_confidence_half_widths = tuple(
        float(value) for value in telemetry.get("outer_replay_confidence_half_widths", ())
    )
    outer_sample_counts = tuple(
        int(value) for value in telemetry.get("outer_replay_current_sample_counts", ())
    )
    assert int(telemetry.get("outer_replay_state_delta", -1)) == 1
    assert len(outer_sources) == segment_count and all(
        value in {"global", "replay", "review"} for value in outer_sources
    )
    assert len(outer_keys) == segment_count and len(set(outer_keys)) == segment_count and all(
        len(str(value)) == 64 for value in outer_keys
    )
    assert outer_score_kind in {"critic_calibration", "repair_spread"}
    expected_score_kind = "repair_spread" if telemetry.get("warmup_phase") == "joint" else "critic_calibration"
    assert outer_score_kind == expected_score_kind
    assert (
        len(outer_calibration_values)
        == len(outer_spread_values)
        == len(outer_priority_scores)
        == len(outer_target_means)
        == len(outer_outcome_variances)
        == len(outer_standard_errors)
        == len(outer_confidence_half_widths)
        == len(outer_sample_counts)
        == segment_count
    )
    assert all(
        math.isfinite(value) and value >= 0.0
        for value in (
            outer_calibration_values
            + outer_spread_values
            + outer_priority_scores
            + outer_outcome_variances
            + outer_standard_errors
            + outer_confidence_half_widths
        )
    )
    assert all(math.isfinite(value) for value in outer_target_means)
    assert outer_sample_counts == (int(telemetry["active_m"]),) * segment_count
    assert int(telemetry.get("actor_observation_dim", -1)) == 158
    assert int(telemetry.get("critic_observation_dim", -1)) == 449
    assert int(telemetry.get("gmt_observation_dim", -1)) == 770
    assert telemetry.get("critic_value_kind") == "state_value"
    assert telemetry.get("critic_support_context_id") == "action-pre-support-plan-kmax32-v1"
    assert telemetry.get("critic_action_conditioned") is False
    assert telemetry.get("critic_target_id") == "scenario-current-exact-m4-mean-symlog-v1"
    assert telemetry.get("gradient_clip_identity") == "separate-actor-critic-v1"
    assert telemetry.get("critic_value_normalization_id") == FRONTRES_VALUE_NORMALIZATION_ID
    value_scale = float(telemetry.get("critic_value_scale", float("nan")))
    value_decay = float(telemetry.get("critic_value_normalizer_decay", float("nan")))
    value_scale_floor = float(telemetry.get("critic_value_normalizer_scale_floor", float("nan")))
    value_count_before = int(telemetry.get("critic_value_normalizer_update_count_before", -1))
    value_count_after = int(telemetry.get("critic_value_normalizer_update_count_after", -1))
    raw_value_loss = float(telemetry.get("critic_raw_value_loss", float("nan")))
    scaled_value_loss = float(telemetry.get("critic_scaled_value_loss", float("nan")))
    assert math.isfinite(value_scale) and value_scale >= 1.0
    assert value_decay == FRONTRES_VALUE_NORMALIZER_DECAY
    assert value_scale_floor == FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR
    assert value_count_before >= 0 and value_count_after == value_count_before + 1
    assert math.isfinite(raw_value_loss) and raw_value_loss >= 0.0
    assert math.isfinite(scaled_value_loss) and scaled_value_loss >= 0.0
    assert math.isclose(scaled_value_loss, raw_value_loss / value_scale**2, rel_tol=1e-5, abs_tol=1e-7)
    max_norm = float(telemetry.get("gradient_clip_max_norm", float("nan")))
    assert max_norm == 0.5
    gradient_facts = {
        role: {
            name: float(telemetry.get(f"{role}_gradient_{name}", float("nan")))
            for name in ("pre_clip_norm", "post_clip_norm", "clip_coefficient")
        }
        for role in ("actor", "critic")
    }
    for facts in gradient_facts.values():
        assert all(math.isfinite(value) for value in facts.values())
        assert facts["pre_clip_norm"] >= 0.0
        assert 0.0 <= facts["post_clip_norm"] <= max_norm + 1e-6
        assert 0.0 < facts["clip_coefficient"] <= 1.0
        expected_coefficient = (
            1.0 if facts["pre_clip_norm"] <= max_norm else max_norm / (facts["pre_clip_norm"] + 1.0e-6)
        )
        assert math.isclose(
            facts["clip_coefficient"], expected_coefficient, rel_tol=1.0e-5, abs_tol=1.0e-7
        )
        if facts["pre_clip_norm"] == 0.0:
            assert facts["post_clip_norm"] == 0.0 and facts["clip_coefficient"] == 1.0
    actor_nonzero = int(telemetry.get("actor_gradient_nonzero_parameter_count", -1))
    critic_nonzero = int(telemetry.get("critic_gradient_nonzero_parameter_count", -1))
    assert 3.0e-7 <= float(telemetry.get("actor_learning_rate", float("nan"))) <= 1.0e-6
    assert float(telemetry.get("critic_learning_rate", float("nan"))) == 1.0e-5
    if telemetry.get("warmup_phase") == "low_dr_joint_init" and telemetry.get("k_stage_iteration") == 0:
        actor_delta = telemetry.get("actor_std_parameter_delta", {})
        critic_delta = telemetry.get("critic_parameter_delta", {})
        assert float(actor_delta.get("param_delta_max_abs", 0.0)) > 0.0
        assert float(critic_delta.get("param_delta_max_abs", 0.0)) > 0.0
        assert actor_nonzero > 0 and critic_nonzero > 0
    # AUDIT-B07: 检查 separate clip、grouped 等权、exact-one 和 coupled warmup.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-B07",
        limit=1,
        segment_mass=telemetry["grouped_segment_mass_shares"],
        attempt_mass=telemetry["grouped_attempt_mass_shares"],
        update_count=telemetry["update_count"],
        optimizer_step_delta=telemetry["optimizer_step_delta"],
        outer_replay_state_delta=telemetry["outer_replay_state_delta"],
        outer_replay_sources=outer_sources,
        outer_replay_score_kind=outer_score_kind,
        outer_replay_critic_calibration_values=outer_calibration_values,
        outer_replay_repair_spread_values=outer_spread_values,
        outer_replay_priority_scores=outer_priority_scores,
        outer_replay_critic_target_means=outer_target_means,
        outer_replay_outcome_variances=outer_outcome_variances,
        outer_replay_standard_errors=outer_standard_errors,
        outer_replay_confidence_half_widths=outer_confidence_half_widths,
        outer_replay_current_sample_counts=outer_sample_counts,
        outer_replay_pool_sizes=telemetry.get("outer_replay_pool_sizes"),
        actor_lr=telemetry["actor_learning_rate"],
        critic_lr=telemetry["critic_learning_rate"],
        max_norm=max_norm,
        actor_gradient={**gradient_facts["actor"], "nonzero_parameters": actor_nonzero},
        critic_gradient={**gradient_facts["critic"], "nonzero_parameters": critic_nonzero},
        phase=telemetry.get("warmup_phase"),
        actor_std_delta=telemetry.get("actor_std_parameter_delta"),
        critic_delta=telemetry.get("critic_parameter_delta"),
        critic_value_normalization=telemetry["critic_value_normalization_id"],
        critic_value_scale=value_scale,
        critic_raw_value_loss=raw_value_loss,
        critic_scaled_value_loss=scaled_value_loss,
        critic_value_normalizer_count=(value_count_before, value_count_after),
    )


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
    # Result: E68 LIVE OBSERVED: actor_warmup ramps from weight=0.002 to 0.040;
    # full actor/joint-RL behavior remains open.
    _emit_owner_snapshot(
        "AUDIT-WARMUP-01",
        phase=getattr(result, "warmup_phase", "missing"),
        phase_iter=getattr(result, "warmup_phase_iteration", "missing"),
        actor_weight=getattr(result, "actor_loss_weight", "missing"),
    )
    # AUDIT-DIAG-01: 检查 diagnostics 来自最终 accepted update, 位于 PPO result -> live summary.
    # Result: E68 LIVE OBSERVED: actor_weight ramps 0.002..0.040 with accepted
    # updates; full actor/joint-RL and population claims remain open.
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
    # B1: inspect the complete in-memory checkpoint-v16 envelope before serialization.
    required = (
        "model_state_dict",
        "optimizer_state_dict",
        "iter",
        "frontres_segment_sampler_state_dict",
        "frontres_segment_k_curriculum",
        "frontres_critic_value_normalizer_state_dict",
        "frontres_v015_checkpoint_identity",
    )
    missing = [key for key in required if key not in payload]
    assert not missing, f"formal Stage 3 checkpoint missing audit fields: {missing}"

    # B2: cross-check checkpoint-v16 identity without treating optional normalizers as unconditional.
    identity = payload["frontres_v015_checkpoint_identity"]
    assert isinstance(identity, Mapping), "formal Stage 3 checkpoint identity must be a mapping"
    assert identity.get("format") == "frontres-v024-checkpoint-v19", "formal audit requires checkpoint-v19"
    assert identity.get("method_contract_id") == "FRS-METHOD-v025", "formal audit requires FRS-METHOD-v025"
    assert identity.get("gain_contract_id") == "FRS-GAIN-v008", "formal audit requires FRS-GAIN-v008"
    assert identity.get("optimization_contract_id") == "FRS-PPO-v012", "formal audit requires FRS-PPO-v012"
    assert identity.get("training_contract_id") == "FRS-TRAIN-v024", "formal audit requires FRS-TRAIN-v024"
    assert identity.get("dr_curriculum_schema_id") == "nested-k-dr-four-class-v1", "formal audit requires TRAIN-v021 DR identity"
    assert identity.get("scalar_target_id") == "symmetric-log-recovery-aware-utility-v1"
    assert identity.get("return_utility") == {
        "identity": "symmetric-log-gain-g0-1-v1",
        "scale": 1.0,
        "placement": "per-attempt-before-current-exact-m4-mean",
    }
    assert identity.get("physics_schema_id") == "clean-anchored-contact-zmp-survival-v1"
    assert identity.get("grouped_schema_id") == "grouped-all-attempt-scalar-v1"
    assert identity.get("critic") == {
        "value_kind": "state_value",
        "input_dim": 449,
        "support_context_id": "action-pre-support-plan-kmax32-v1",
        "action_conditioned": False,
        "target_id": "scenario-current-exact-m4-mean-symlog-v1",
        "return_utility_id": "symmetric-log-gain-g0-1-v1",
        "return_utility_scale": 1.0,
    }
    assert identity.get("gradient_clip") == {
        "identity": "separate-actor-critic-v1",
        "max_norm": 0.5,
    }
    assert identity.get("critic_value_normalizer") == {
        "identity": FRONTRES_VALUE_NORMALIZATION_ID,
        "decay": FRONTRES_VALUE_NORMALIZER_DECAY,
        "scale_floor": FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
    }
    value_normalizer_state = FrontRESValueNormalizerState.from_state_dict(
        payload["frontres_critic_value_normalizer_state_dict"]
    )
    assert value_normalizer_state.update_count == int(payload["iter"])
    assert identity.get("gain") == {"beta": 0.02}, "formal audit requires the frozen v007 beta"
    assert "constraint_solver" not in identity and "projection_schema_id" not in identity
    assert "frontres_gain_config" not in payload, "active checkpoint-v16 must exclude legacy scalar Gain metadata"
    assert "dr_scale" not in payload and not any(str(key).startswith("frontres_gmt_frontier_") for key in payload), "active checkpoint-v16 must exclude legacy adaptive DR state"
    normalizer = identity.get("normalizer")
    if isinstance(normalizer, Mapping) and normalizer.get("mode") == "empirical_prefix_plus_frozen_gmt":
        assert "obs_norm_state_dict" in payload and "privileged_obs_norm_state_dict" in payload
    curriculum = identity.get("curriculum")
    assert isinstance(curriculum, Mapping), "formal Stage 3 checkpoint has no sealed curriculum identity"
    try:
        saved_schedule = tuple(tuple(row) for row in payload["frontres_segment_k_curriculum"])
        identity_schedule = tuple(tuple(row) for row in curriculum["schedule"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError("formal Stage 3 checkpoint has malformed curriculum schedule") from exc
    assert saved_schedule and saved_schedule == identity_schedule, (
        "formal Stage 3 checkpoint top-level and identity curriculum schedules differ"
    )
    active_k = int(curriculum.get("active_k", 0))
    assert active_k in {row[0] for row in identity_schedule}, "formal Stage 3 checkpoint active K is not scheduled"
    transaction = identity.get("transaction")
    assert isinstance(transaction, Mapping) and transaction.get("state") == "committed"
    receipt = transaction.get("receipt")
    assert isinstance(receipt, Mapping), "AUDIT-B08 requires the committed receipt"
    assert int(receipt.get("optimizer_step_delta", -1)) == 1
    assert int(receipt.get("selected_segment_count", -1)) == 8
    assert int(receipt.get("policy_row_count", -1)) == 32
    assert int(receipt.get("role_row_count", -1)) == 64
    assert int(receipt.get("active_k", -1)) == 8
    assert int(receipt.get("active_m", -1)) == 4
    gmt = identity.get("gmt")
    assert isinstance(gmt, Mapping), "AUDIT-B08 requires frozen GMT identity"
    assert int(gmt.get("normalizer_dim", -1)) == 770
    assert len(str(gmt.get("checkpoint_sha256", ""))) == 64
    assert len(str(gmt.get("normalizer_fingerprint", ""))) == 64
    layout = identity.get("future_intent_layout")
    assert isinstance(layout, Mapping), "AUDIT-B08 requires the 928 Actor / 158 Actor-prefix / 770 GMT layout identity"
    assert int(layout.get("actor_dim", -1)) == 928
    assert int(layout.get("prefix_dim", -1)) == 158
    assert int(layout.get("gmt_dim", -1)) == 770
    assert tuple(layout.get("future_offsets", ())) == (1, 2)
    optimizer_state = payload.get("optimizer_state_dict")
    groups = optimizer_state.get("param_groups") if isinstance(optimizer_state, Mapping) else None
    if isinstance(groups, list) and groups:
        groups_by_role = {
            str(group.get("frontres_role", "")): group for group in groups if isinstance(group, Mapping)
        }
        assert len(groups) == 2 and set(groups_by_role) == {"actor", "critic"}
        assert 3.0e-7 <= float(groups_by_role["actor"].get("lr", float("nan"))) <= 1.0e-6
        assert float(groups_by_role["critic"].get("lr", float("nan"))) == 1.0e-5

    # B3: Emit the exact coordinated identity immediately before torch.save.
    print(
        "[AUDIT-PERSIST-01] "
        f"path={path} iter={payload.get('iter', 'missing')} "
        f"model={int('model_state_dict' in payload)} optimizer={int('optimizer_state_dict' in payload)} "
        f"obs_norm={int('obs_norm_state_dict' in payload)} sampler={int('frontres_segment_sampler_state_dict' in payload)} "
        f"contracts={identity.get('method_contract_id')}/{identity.get('gain_contract_id')}/"
        f"{identity.get('optimization_contract_id')}/{identity.get('training_contract_id')} "
        f"scalar_target={identity.get('scalar_target_id')} beta={identity.get('gain', {}).get('beta', 'missing')} "
        f"curriculum={saved_schedule} stage={curriculum.get('k_stage_index', 'missing')} "
        f"active_k={active_k} phase={curriculum.get('phase', 'missing')} "
        f"fingerprint={curriculum.get('schedule_fingerprint', 'missing')}",
        flush=True,
    )
def print_checkpoint_reload_audit(
    runner: Any,
    *,
    path: str,
    payload: Mapping[str, Any],
    validated_identity: Mapping[str, Any],
    file_sha256: str,
) -> None:
    """Project a strictly validated post-``os.replace`` checkpoint-v16 readback."""

    if not formal_runtime_audit_enabled(runner):
        return
    identity = payload.get("frontres_v015_checkpoint_identity")
    assert isinstance(identity, Mapping) and dict(identity) == dict(validated_identity)
    assert len(file_sha256) == 64
    assert identity.get("format") == "frontres-v024-checkpoint-v19"
    critic = identity.get("critic")
    layout = identity.get("future_intent_layout")
    transaction = identity.get("transaction")
    assert (
        isinstance(critic, Mapping)
        and critic.get("input_dim") == 449
        and critic.get("support_context_id") == "action-pre-support-plan-kmax32-v1"
    )
    assert isinstance(layout, Mapping)
    assert int(layout.get("prefix_dim", -1)) == 158 and int(layout.get("gmt_dim", -1)) == 770
    assert isinstance(transaction, Mapping) and transaction.get("state") == "committed"
    receipt = transaction.get("receipt")
    assert isinstance(receipt, Mapping) and int(receipt.get("optimizer_step_delta", -1)) == 1
    assert int(receipt.get("policy_row_count", -1)) == 32 and int(receipt.get("role_row_count", -1)) == 64
    assert int(payload.get("iter", -1)) == int(identity.get("curriculum", {}).get("absolute_iteration", -2))
    gmt = identity.get("gmt")
    curriculum = identity.get("curriculum")
    gradient_clip = identity.get("gradient_clip")
    value_normalizer_identity = identity.get("critic_value_normalizer")
    value_normalizer_state = FrontRESValueNormalizerState.from_state_dict(
        payload.get("frontres_critic_value_normalizer_state_dict")
    )
    assert isinstance(gmt, Mapping) and len(str(gmt.get("checkpoint_sha256", ""))) == 64
    assert len(str(gmt.get("normalizer_fingerprint", ""))) == 64
    assert isinstance(curriculum, Mapping) and str(curriculum.get("schedule_fingerprint", ""))
    assert gradient_clip == {"identity": "separate-actor-critic-v1", "max_norm": 0.5}
    assert value_normalizer_identity == {
        "identity": FRONTRES_VALUE_NORMALIZATION_ID,
        "decay": FRONTRES_VALUE_NORMALIZER_DECAY,
        "scale_floor": FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
    }
    assert value_normalizer_state.update_count == int(payload["iter"])

    optimizer_state = payload.get("optimizer_state_dict")
    groups = optimizer_state.get("param_groups") if isinstance(optimizer_state, Mapping) else None
    assert isinstance(groups, list) and len(groups) == 2
    groups_by_role = {
        str(group.get("frontres_role", "")): group for group in groups if isinstance(group, Mapping)
    }
    assert set(groups_by_role) == {"actor", "critic"}
    lrs = (float(groups_by_role["actor"]["lr"]), float(groups_by_role["critic"]["lr"]))
    assert 3.0e-7 <= lrs[0] <= 1.0e-6 and lrs[1] == 1.0e-5

    # AUDIT-B08: atomic save -> strict loader/validator -> read-only identity projection.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-B08",
        limit=1,
        path=path,
        readback=1,
        file_sha256=file_sha256,
        checkpoint_format=identity["format"],
        contracts=(
            identity.get("method_contract_id"),
            identity.get("gain_contract_id"),
            identity.get("optimization_contract_id"),
            identity.get("training_contract_id"),
        ),
        transaction_id=receipt.get("transaction_id"),
        iteration=payload.get("iter"),
        active_k=receipt.get("active_k"),
        active_m=receipt.get("active_m"),
        policy_rows=receipt.get("policy_row_count"),
        role_rows=receipt.get("role_row_count"),
        optimizer_step_delta=receipt.get("optimizer_step_delta"),
        optimizer_lrs=lrs,
        layout=(layout["prefix_dim"], critic["input_dim"], layout["gmt_dim"]),
        gradient_clip=gradient_clip,
        critic_value_normalizer=value_normalizer_identity,
        critic_value_normalizer_update_count=value_normalizer_state.update_count,
        schedule_fingerprint=curriculum["schedule_fingerprint"],
        normalizer_mode=identity.get("normalizer", {}).get("mode"),
        gmt_sha256=gmt["checkpoint_sha256"],
        gmt_normalizer=gmt["normalizer_fingerprint"],
        runner_mutated=0,
    )


__all__ = [
    "formal_runtime_audit_enabled",
    "print_checkpoint_payload_audit",
    "print_checkpoint_reload_audit",
    "print_formal_route_audit",
    "print_one_action_k_audit",
    "print_v017_repair_attempts_audit",
    "print_phase_b_telemetry_audit",
    "print_ppo_audit",
    "print_rollout_storage_audit",
    "print_sampler_audit",
    "print_segment_replay_transaction_audit",
]
