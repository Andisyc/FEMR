"""FrontRES live policy adapter and exact-one optimizer update boundary."""





from __future__ import annotations





import copy


import math


from pathlib import Path


from typing import Any


import torch


from rsl_rl.algorithms import FrontRESUnified


from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
)


from rsl_rl.frontres.frontres_segment_warmup import frontres_segment_warmup_phase


from rsl_rl.runners.frontres_formal_runtime_audit import print_ppo_audit





from rsl_rl.frontres.frontres_policy_evaluation import (
    FrontRESSegmentLivePolicyAdapter,
    _evaluate_segment_delta_se_log_prob,
    _evaluate_segment_delta_se_log_prob_from_stats,
    _segment_delta_se_log_prob_parts,
)





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


def _clear_noncritic_grads(policy: Any, optimizer_params: tuple[tuple[str, torch.Tensor], ...]) -> None:
    """Hold the full-6D actor and its std fixed during DP-09 critic-only warmup."""
    critic = getattr(policy, "critic", None)
    critic_ids = {id(param) for param in critic.parameters()} if critic is not None else set()
    if not critic_ids:
        raise RuntimeError("DP-09 critic-only warmup requires policy.critic parameters.")
    for _, param in optimizer_params:
        if id(param) not in critic_ids:
            param.grad = None


def _set_segment_optimizer_lr(alg: Any, lr: float) -> None:
    optimizer = getattr(alg, "optimizer", None)
    if any(group.get("frontres_role") in {"actor", "critic"} for group in getattr(optimizer, "param_groups", ())):
        raise RuntimeError("FRS-TRAIN-v021 split-LR optimizer rejects group-wide LR mutation")
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
    if any(group.get("frontres_role") in {"actor", "critic"} for group in param_groups) and schedule != "fixed":
        raise RuntimeError("FRS-TRAIN-v021 split-LR optimizer rejects adaptive scheduling")
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


def run_frontres_segment_single_update(runner: Any, storage_batch: Any) -> object:
    # QUALITY-UPDATE-01: 检查 advantage/log-prob -> optimizer step -> accepted policy delta.
    # Result: PENDING_Q_EVIDENCE.
    # B1: step 前冻结 old/new distribution、advantage sign 与 held-out identity.
    # B2: backward/optimizer/trust 顺序记录 parameter 与 per-dim mean delta.
    # B3: accepted/rollback 后比较正负 advantage log-prob 方向.
    """Run one Stage 3 Segment PPO update on the isolated live Segment path.

    Status: active Segment Replay update boundary.
    Upstream: live probe/update loop passes storage_batch from rollout evidence.
    Downstream: FrontRESSegmentPPOBatch -> compute_frontres_segment_ppo_loss -> optimizer.step -> post diagnostics.
    Evidence: contract-confirmed by frontres_segment_live_single_update_contract.py.
    Gap: one fake/live-boundary update does not prove long live training quality.
    """
    if bool(getattr(runner.alg, "frontres_relational_actor_only", False)):
        return run_frontres_relational_actor_update(runner, storage_batch)
    runner.train_mode()
    # B1: Convert storage evidence into the algorithm-owned batch contract.
    ppo_batch = storage_batch.to_ppo_batch(FrontRESSegmentPPOBatch)
    policy_adapter = FrontRESSegmentLivePolicyAdapter(
        runner.alg,
        privileged_observations=storage_batch.privileged_observations,
    )
    warmup_phase = frontres_segment_warmup_phase(
        iteration=int(getattr(runner, "current_learning_iteration", 0)),
        critic_warmup_iterations=int(getattr(runner.alg, "frontres_segment_critic_warmup_iterations", 0)),
        actor_warmup_iterations=int(getattr(runner.alg, "frontres_segment_actor_warmup_iterations", 0)),
    )
    ppo_cfg = FrontRESSegmentPPOConfig(
        clip_param=float(getattr(runner.alg, "clip_param", 0.2)),
        value_clip_param=float(getattr(runner.alg, "clip_param", 0.2)),
        value_loss_coef=float(getattr(runner.alg, "value_loss_coef", 1.0)),
        entropy_coef=float(getattr(runner.alg, "entropy_coef", 0.0)),
        use_clipped_value_loss=bool(getattr(runner.alg, "use_clipped_value_loss", True)),
        advantage_normalization=str(getattr(runner.alg, "frontres_segment_advantage_normalization", "scale_only")),
        actor_loss_weight=warmup_phase.actor_loss_weight,
    )
    # B2: First forward is the pre-step loss and MOSAIC-style old/new KL source.
    ppo_result = compute_frontres_segment_ppo_loss(policy_adapter, ppo_batch, ppo_cfg)
    credit_result_path = str(getattr(runner, "_frontres_policy_quality_q2d_credit_result", "") or "")
    if credit_result_path and not Path(credit_result_path).exists():
        from rsl_rl.frontres.frontres_policy_quality_q2d import write_q2d_credit_tuple

        if ppo_batch.old_means is None or ppo_batch.old_sigmas is None:
            raise ValueError("Q2-D credit capture requires rollout old_means and old_sigmas")
        raw_actions = _segment_delta_se_log_prob_parts(
            runner.alg.policy,
            ppo_batch.actions,
            ppo_batch.old_means,
            ppo_batch.old_sigmas,
        )["raw_actions"]
        # QUALITY-CREDIT-01: capture the finalized Gain -> return -> advantage tuple
        # at the last read-only boundary before the official PPO optimizer step.
        write_q2d_credit_tuple(
            result_path=credit_result_path,
            raw_actions=raw_actions,
            bounded_actions=ppo_batch.actions,
            old_means=ppo_batch.old_means,
            old_sigmas=ppo_batch.old_sigmas,
            gains=storage_batch.rewards,
            returns=ppo_batch.returns,
            advantages=ppo_batch.advantages,
            valid_mask=ppo_batch.valid_mask,
            segment_ids=ppo_batch.segment_ids,
            audit_transaction_id=storage_batch.audit_transaction_id,
            audit_batch_signature=storage_batch.audit_batch_signature,
            audit_identity_state=storage_batch.audit_identity_state,
        )
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
    diagnostics["warmup_phase"] = warmup_phase.name
    diagnostics["warmup_phase_iteration"] = warmup_phase.phase_iteration
    diagnostics["actor_loss_weight"] = warmup_phase.actor_loss_weight
    for key, value in pre_step_lr_diagnostics.items():
        diagnostics[f"mosaic_pre_step_{key}"] = value
    for key, value in rejected_lr_diagnostics.items():
        diagnostics[f"segment_reject_{key}"] = value
    diagnostics.update(post_update_diagnostics)
    diagnostics.update(lr_diagnostics)
    _attach_ppo_update_diagnostics(ppo_result, diagnostics)
    # AUDIT-PPO-01: 检查 warmup/PPO/KL/Frozen GMT, 位于 optimizer diagnostics -> live summary.
    # Result: E68 LIVE OBSERVED: actor_warmup weight=0.002..0.040, accepted
    # updates, finite post-update KL, and frozen GMT ownership.
    print_ppo_audit(runner, result=ppo_result)
    runner.eval_mode()
    return ppo_result


def run_frontres_relational_actor_update(runner: Any, storage_batch: Any) -> object:
    """Run one explicit relational Actor-only update.

    This owner accepts only preference edges produced by the relational Gain
    adapter.  It rejects a scalar fallback, a Critic-owned optimizer group, or
    a no-edge transaction; the latter performs no optimizer step.
    """

    if not bool(getattr(runner.alg, "frontres_relational_actor_only", False)):
        raise RuntimeError("relational Actor update requires frontres_relational_actor_only=True")
    from rsl_rl.algorithms.frontres_segment_ppo import (
        FrontRESRelationalPPOBatch,
        FrontRESRelationalPPOConfig,
        compute_frontres_relational_actor_loss,
    )
    runner.train_mode()
    ppo_batch = storage_batch.to_grouped_ppo_candidate_batch(FrontRESRelationalPPOBatch)
    edges = tuple(getattr(storage_batch, "preference_edges", ()) or ())
    policy_adapter = FrontRESSegmentLivePolicyAdapter(
        runner.alg,
        privileged_observations=storage_batch.privileged_observations,
        actor_only=True,
    )
    cfg = FrontRESRelationalPPOConfig(
        clip_param=float(getattr(runner.alg, "clip_param", 0.2)),
        max_log_ratio=20.0,
    )
    result = compute_frontres_relational_actor_loss(policy_adapter, ppo_batch, edges, cfg)
    optimizer = runner.alg.optimizer
    optimizer_roles = {
        str(group.get("frontres_role", ""))
        for group in getattr(optimizer, "param_groups", ())
    }
    if optimizer_roles != {"actor"}:
        raise RuntimeError(
            "FRS-TRAIN-v025 relational route requires an Actor-only optimizer; "
            f"got roles={sorted(optimizer_roles)!r}"
        )
    critic = getattr(runner.alg.policy, "critic", None)
    if critic is not None and any(bool(param.requires_grad) for param in critic.parameters()):
        raise RuntimeError("FRS-TRAIN-v025 relational route requires frozen Critic parameters")
    if result.should_step:
        optimizer.zero_grad()
        result.total_loss.backward()
        actor_parameters = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group.get("params", ())
        )
        torch.nn.utils.clip_grad_norm_(actor_parameters, float(getattr(runner.alg, "max_grad_norm", 1.0)))
        optimizer.step()
    runner.eval_mode()
    return result


# Public policy/update seams used by storage and formal transaction owners.
snapshot_optimizer_parameters = _optimizer_parameter_snapshots
summarize_parameter_deltas = _parameter_delta_stats
evaluate_segment_delta_se_log_prob_from_stats = _evaluate_segment_delta_se_log_prob_from_stats
