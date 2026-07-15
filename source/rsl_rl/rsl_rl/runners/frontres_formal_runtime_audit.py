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
