# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES rollout-step preparation helpers.

This module owns the pre-env-step bridge from policy actions to executable
environment actions.  The runner keeps the main loop and calls env.step().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class FrontRESRolloutStepPlan:
    actions: torch.Tensor | None
    env_actions: torch.Tensor
    hsl_pos_snapshot: torch.Tensor | None
    hsl_quat_snapshot: torch.Tensor | None


@dataclass(frozen=True)
class FrontRESV015FrozenGMTStepPlan:
    """One frozen-GMT action and its command-owned Clean-C evidence."""

    env_actions: torch.Tensor
    continuation: torch.Tensor
    valid_mask: torch.Tensor
    cursor: torch.Tensor


def _append_future_intent_actor_context(runner: Any, obs: torch.Tensor) -> torch.Tensor:
    """Route the actor-only v015 q29 tail without admitting the legacy 65D tape."""

    append = getattr(runner, "_append_frontres_future_intent_context", None)
    if callable(append):
        return append(obs)
    batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    if isinstance(getattr(batch, "frontres_local_scenario_intent_q29", None), torch.Tensor):
        raise RuntimeError("v015 future-intent Segment Replay requires runner actor-context connectivity")
    return obs


def _uses_v015_future_intent_route(runner: Any) -> bool:
    """Return whether this runner is configured for the v015 q29 actor interface."""

    alg = getattr(runner, "alg", None)
    offsets = tuple(int(value) for value in (getattr(alg, "frontres_future_offsets", ()) or ()))
    return bool(offsets) or getattr(runner, "_frontres_future_intent_layout", None) is not None


def _motion_groups_for_runner(runner: Any) -> torch.Tensor | None:
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env, "command_manager") and "motion" in env.command_manager._terms):
        return None
    motion_command = env.command_manager._terms["motion"]
    if hasattr(motion_command, "env_motion_groups"):
        return motion_command.env_motion_groups.clone()
    return None


def _record_velocity_estimator_error(runner: Any, vel_est_error_buffer: Any) -> None:
    if not (hasattr(runner.alg, "last_estimated_ref_vel") and runner.alg.last_estimated_ref_vel is not None):
        return
    from whole_body_tracking.tasks.tracking.mdp import observations as mdp

    gt_ref_vel_b = mdp.ref_base_lin_vel_b(runner.env.unwrapped, "motion")
    vel_error = (runner.alg.last_estimated_ref_vel - gt_ref_vel_b).abs().mean(dim=-1)
    vel_est_error_buffer.extend(vel_error.cpu().numpy().tolist())


def _task_space_raw_proposal(policy: Any, actions: torch.Tensor) -> torch.Tensor:
    proposal_dim = min(6, actions.shape[-1])
    if proposal_dim <= 0:
        return actions[:, :0]
    scales = torch.empty(proposal_dim, device=actions.device, dtype=actions.dtype)
    pos_dim = min(3, proposal_dim)
    scales[:pos_dim] = float(getattr(policy, "max_delta_pos", 1.0))
    if proposal_dim > 3:
        scales[3:] = float(getattr(policy, "max_delta_rpy", 1.0))
    normalized = (actions[:, :proposal_dim] / scales.view(1, -1)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.atanh(normalized)


def _task_space_log_prob_from_stats(
    policy: Any,
    actions: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    dim = min(actions.shape[-1], mean.shape[-1], sigma.shape[-1])
    if hasattr(policy, "get_actions_log_prob_per_dim_from_stats"):
        dims = torch.arange(dim, device=actions.device)
        return policy.get_actions_log_prob_per_dim_from_stats(actions, mean, sigma, dims).sum(dim=-1)
    raw = _task_space_raw_proposal(policy, actions)
    dim = min(raw.shape[-1], mean.shape[-1], sigma.shape[-1])
    dist = torch.distributions.Normal(mean[:, :dim], sigma[:, :dim])
    log_prob = dist.log_prob(raw[:, :dim]).sum(dim=-1)
    proposal_dim = raw.shape[-1]
    scales = torch.empty(proposal_dim, device=actions.device, dtype=actions.dtype)
    pos_dim = min(3, proposal_dim)
    scales[:pos_dim] = float(getattr(policy, "max_delta_pos", 1.0))
    if proposal_dim > 3:
        scales[3:] = float(getattr(policy, "max_delta_rpy", 1.0))
    normalized = (actions[:, :proposal_dim] / scales.view(1, -1)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    log_j = (torch.log(scales).view(1, -1) + torch.log(1.0 - normalized.pow(2) + 1e-6)).sum(dim=-1)
    return log_prob - log_j


def _rewrite_task_space_log_prob(runner: Any, actions: torch.Tensor) -> None:
    runner.alg.transition.actions = actions.detach()
    policy = runner.alg.policy
    action_mean = getattr(runner.alg.transition, "action_mean", getattr(policy, "action_mean", None))
    action_sigma = getattr(runner.alg.transition, "action_sigma", getattr(policy, "action_std", None))
    if action_mean is not None and action_sigma is not None:
        runner.alg.transition.actions_log_prob = _task_space_log_prob_from_stats(
            policy,
            actions,
            action_mean,
            action_sigma,
        ).detach()
    elif hasattr(runner.alg, "_get_actor_log_prob"):
        runner.alg.transition.actions_log_prob = runner.alg._get_actor_log_prob(actions).detach()
    else:
        runner.alg.transition.actions_log_prob = policy.get_actions_log_prob(actions).detach()


def _apply_frontres_baseline_transition_override(
    runner: Any,
    *,
    actions: torch.Tensor,
    n_train: int,
    n_candidate: int,
    n_base: int,
    n_clean: int,
    is_task_space_mode: bool,
    use_explicit_baseline_count: bool,
) -> None:
    if use_explicit_baseline_count:
        zeros_gmt = torch.zeros(
            n_candidate + n_base + n_clean,
            runner.alg.transition.actions.shape[-1],
            device=runner.device,
        )
    else:
        zeros_gmt = torch.zeros_like(actions[n_train:])
    runner.alg.transition.actions[n_train:] = zeros_gmt
    if hasattr(runner.alg, "_get_actor_log_prob"):
        logp_zeros = runner.alg._get_actor_log_prob(runner.alg.transition.actions)[n_train:]
    else:
        mean_gmt = runner.alg.policy.action_mean[n_train:].clone()
        std_gmt = runner.alg.policy.action_std[n_train:]
        logp_zeros = torch.distributions.Normal(mean_gmt, std_gmt).log_prob(zeros_gmt).sum(dim=-1)
        if is_task_space_mode:
            logp_zeros = logp_zeros - (
                3 * math.log(runner.alg.policy.max_delta_pos)
                + 3 * math.log(runner.alg.policy.max_delta_rpy)
            )
    runner.alg.transition.actions_log_prob[n_train:] = logp_zeros

    frontres_mask = torch.zeros(runner.env.num_envs, 1, device=runner.device)
    frontres_mask[:n_train] = 1.0
    runner.alg.transition.frontres_mask = frontres_mask


def _build_env_actions_from_policy_actions(
    runner: Any,
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    is_frontres: bool,
    is_task_space_mode: bool,
    n_train: int,
    n_candidate: int,
    use_transition_actions_for_task_env_action: bool,
) -> torch.Tensor:
    if is_task_space_mode:
        runner._apply_frontres_task_corrections(
            actions,
            n_train,
            allow_oracle=False,
            n_candidate=n_candidate if is_frontres else 0,
        )
        obs_corr, extras_corr = runner.env.get_observations()
        obs_corr_dict = extras_corr.get("observations", {})
        if runner.policy_obs_type is not None and runner.policy_obs_type in obs_corr_dict:
            obs_corr = obs_corr_dict[runner.policy_obs_type]
        obs_corr = _append_future_intent_actor_context(runner, obs_corr.to(runner.device))
        obs_corr = runner._apply_obs_normalizer(obs_corr)
        task_actions = runner.alg.transition.actions if use_transition_actions_for_task_env_action else actions
        return runner.alg.policy.get_env_action(obs_corr, task_actions)
    if hasattr(runner.alg.policy, "get_env_action"):
        return runner.alg.policy.get_env_action(obs, actions)
    return actions


def _write_supervised_target_before_step(
    runner: Any,
    *,
    actions: torch.Tensor | None,
    iteration: int,
    rollout_step: int,
    is_task_space_mode: bool,
    n_train: int,
) -> None:
    if is_task_space_mode and _uses_v015_future_intent_route(runner):
        if float(getattr(runner.alg, "lambda_supervised", 0.0)) > 0.0:
            raise RuntimeError(
                "FRS-TRAIN-v008 forbids a nonzero Stage-3 online HSL target writer on the v015 route"
            )
        return
    if not (is_task_space_mode and getattr(runner.alg, "lambda_supervised", 0.0) > 0):
        return
    env_for_sup = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    for cmd_sup in env_for_sup.command_manager._terms.values():
        if hasattr(cmd_sup, "supervised_target"):
            sup_target = cmd_sup.supervised_target.clone().to(runner.device)
            sup_target = runner._frontres_action_cone.project_task_target(cmd_sup, sup_target)
            runner.alg.transition.supervised_target = sup_target
            runner._maybe_print_frontres_restore_debug(
                it=iteration,
                rollout_step=rollout_step,
                actions=actions,
                supervised_target=sup_target,
                n_train=n_train,
            )
            break


def _capture_hsl_snapshot_before_step(
    runner: Any,
    *,
    is_frontres: bool,
    is_task_space_mode: bool,
    n_train: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not (
        is_frontres
        and is_task_space_mode
        and bool(runner.cfg.get("frontres_hsl_rollout_label_enabled", False))
    ):
        return None, None
    if _uses_v015_future_intent_route(runner):
        raise RuntimeError(
            "FRS-TRAIN-v008 forbids legacy HSL rollout snapshots on the v015 Stage-3 route"
        )
    env_for_hsl_pre = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not hasattr(env_for_hsl_pre, "command_manager"):
        return None, None
    for term_hsl_pre in env_for_hsl_pre.command_manager._terms.values():
        if (
            hasattr(term_hsl_pre, "_frontres_pos_correction")
            and hasattr(term_hsl_pre, "_frontres_quat_correction")
        ):
            return (
                term_hsl_pre._frontres_pos_correction[:n_train].clone(),
                term_hsl_pre._frontres_quat_correction[:n_train].clone(),
            )
    return None, None


def prepare_frontres_rollout_step(
    runner: Any,
    *,
    obs: torch.Tensor,
    privileged_obs: torch.Tensor | None,
    teacher_obs: torch.Tensor | None,
    ref_vel_estimator_obs: torch.Tensor | None,
    obs_raw_for_gmt: torch.Tensor | None,
    vel_est_error_buffer: Any,
    iteration: int,
    rollout_step: int,
    is_frontres: bool,
    is_task_space_mode: bool,
    n_train: int,
    n_candidate: int,
    n_base: int,
    n_clean: int,
) -> FrontRESRolloutStepPlan:
    if getattr(runner, "_frontres_v015_one_action_k_phase", None) == "frozen":
        raise RuntimeError(
            "v015 one-action K collection forbids a second actor sample; use prepare_frontres_v015_frozen_gmt_step()"
        )
    actions = None
    if runner.training_type in ("mosaic", "frontres"):
        actions = runner.alg.act(
            obs,
            privileged_obs,
            teacher_obs=teacher_obs if runner.training_type == "mosaic" else None,
            ref_vel_estimator_obs=ref_vel_estimator_obs,
            motion_groups=_motion_groups_for_runner(runner),
        )
        if is_task_space_mode:
            _rewrite_task_space_log_prob(runner, actions)
        _record_velocity_estimator_error(runner, vel_est_error_buffer)
        if is_frontres:
            _apply_frontres_baseline_transition_override(
                runner,
                actions=actions,
                n_train=n_train,
                n_candidate=n_candidate,
                n_base=n_base,
                n_clean=n_clean,
                is_task_space_mode=is_task_space_mode,
                use_explicit_baseline_count=True,
            )
        env_actions = _build_env_actions_from_policy_actions(
            runner,
            obs=obs,
            actions=actions,
            is_frontres=is_frontres,
            is_task_space_mode=is_task_space_mode,
            n_train=n_train,
            n_candidate=n_candidate,
            use_transition_actions_for_task_env_action=False,
        )
    elif runner.training_type == "supervise":
        if obs_raw_for_gmt is None:
            raise RuntimeError("Supervise rollout requires raw observations for GMT action generation.")
        env_actions = runner.alg.policy.get_gmt_action(obs_raw_for_gmt)
        _ = runner.alg.act(obs, privileged_obs)
    else:
        actions = runner.alg.act(obs, privileged_obs)
        if is_task_space_mode:
            _rewrite_task_space_log_prob(runner, actions)
        if is_frontres:
            _apply_frontres_baseline_transition_override(
                runner,
                actions=actions,
                n_train=n_train,
                n_candidate=n_candidate,
                n_base=n_base,
                n_clean=n_clean,
                is_task_space_mode=is_task_space_mode,
                use_explicit_baseline_count=False,
            )
        env_actions = _build_env_actions_from_policy_actions(
            runner,
            obs=obs,
            actions=actions,
            is_frontres=is_frontres,
            is_task_space_mode=is_task_space_mode,
            n_train=n_train,
            n_candidate=n_candidate,
            use_transition_actions_for_task_env_action=True,
        )

    _write_supervised_target_before_step(
        runner,
        actions=actions,
        iteration=iteration,
        rollout_step=rollout_step,
        is_task_space_mode=is_task_space_mode,
        n_train=n_train,
    )
    hsl_pos_snapshot, hsl_quat_snapshot = _capture_hsl_snapshot_before_step(
        runner,
        is_frontres=is_frontres,
        is_task_space_mode=is_task_space_mode,
        n_train=n_train,
    )
    return FrontRESRolloutStepPlan(
        actions=actions,
        env_actions=env_actions,
        hsl_pos_snapshot=hsl_pos_snapshot,
        hsl_quat_snapshot=hsl_quat_snapshot,
    )


def _frontres_motion_command(runner: Any) -> Any:
    """Return the sole command owner used by the candidate-only K collector."""

    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    manager = getattr(env, "command_manager", None)
    get_term = getattr(manager, "get_term", None)
    if callable(get_term):
        command = get_term("motion")
    else:
        command = getattr(manager, "_terms", {}).get("motion") if manager is not None else None
    if command is None:
        raise RuntimeError("v015 one-action K collector requires env.command_manager motion command")
    return command


def prepare_frontres_v015_one_action_at_t(
    runner: Any,
    *,
    obs: torch.Tensor,
    privileged_obs: torch.Tensor | None,
    teacher_obs: torch.Tensor | None,
    ref_vel_estimator_obs: torch.Tensor | None,
    iteration: int,
    n_repair: int,
    n_noisy: int,
) -> FrontRESRolloutStepPlan:
    """Sample the unique v015 Repair policy tuple at the local scenario start t."""

    if n_repair <= 0 or n_noisy != n_repair or int(obs.shape[0]) != n_repair + n_noisy:
        raise ValueError(
            "v015 one-action K collector requires equal Repair/Noisy rows and an exact two-role observation batch"
        )
    if getattr(runner, "_frontres_v015_one_action_k_phase", None) is not None:
        raise RuntimeError("v015 one-action K collector is already active on this runner")
    runner._frontres_v015_one_action_k_phase = "acting"
    try:
        plan = prepare_frontres_rollout_step(
            runner,
            obs=obs,
            privileged_obs=privileged_obs,
            teacher_obs=teacher_obs,
            ref_vel_estimator_obs=ref_vel_estimator_obs,
            obs_raw_for_gmt=None,
            vel_est_error_buffer=[],
            iteration=iteration,
            rollout_step=0,
            is_frontres=True,
            is_task_space_mode=True,
            n_train=n_repair,
            n_candidate=0,
            n_base=n_noisy,
            n_clean=0,
        )
    except Exception:
        delattr(runner, "_frontres_v015_one_action_k_phase")
        raise
    quality_route = getattr(runner, "_frontres_v015_quality_action_route", None)
    if quality_route is not None:
        if quality_route not in {"zero", "hsl", "policy"} or not bool(
            getattr(getattr(runner, "alg", None), "frontres_v015_formal_transaction_enabled", False)
        ):
            delattr(runner, "_frontres_v015_one_action_k_phase")
            raise RuntimeError("v015 deterministic quality action is restricted to the formal quality route")
        transition = getattr(runner.alg, "transition", None)
        mean = getattr(transition, "action_mean", None)
        policy = runner.alg.policy
        if not isinstance(mean, torch.Tensor) or tuple(mean.shape[:2]) != (n_repair + n_noisy, 6):
            delattr(runner, "_frontres_v015_one_action_k_phase")
            raise RuntimeError("v015 quality route requires one raw 6D proposal mean per role row")
        if quality_route == "zero":
            deterministic = torch.zeros_like(mean[:, :6])
        else:
            deterministic = torch.cat(
                (
                    torch.tanh(mean[:, :3]) * float(policy.max_delta_pos),
                    torch.tanh(mean[:, 3:6]) * float(policy.max_delta_rpy),
                ),
                dim=-1,
            )
        deterministic[n_repair:] = 0.0
        _rewrite_task_space_log_prob(runner, deterministic)
        _apply_frontres_baseline_transition_override(
            runner,
            actions=deterministic,
            n_train=n_repair,
            n_candidate=0,
            n_base=n_noisy,
            n_clean=0,
            is_task_space_mode=True,
            use_explicit_baseline_count=True,
        )
        env_actions = _build_env_actions_from_policy_actions(
            runner,
            obs=obs,
            actions=deterministic,
            is_frontres=True,
            is_task_space_mode=True,
            n_train=n_repair,
            n_candidate=0,
            use_transition_actions_for_task_env_action=False,
        )
        plan = FrontRESRolloutStepPlan(
            actions=deterministic,
            env_actions=env_actions,
            hsl_pos_snapshot=plan.hsl_pos_snapshot,
            hsl_quat_snapshot=plan.hsl_quat_snapshot,
        )
    if plan.actions is None or tuple(plan.actions.shape) != (n_repair + n_noisy, 6):
        delattr(runner, "_frontres_v015_one_action_k_phase")
        raise RuntimeError("v015 one-action K collector requires one full-6D policy action per scored role row")
    runner._frontres_v015_one_action_k_phase = "frozen"
    return plan


def prepare_frontres_v015_frozen_gmt_step(
    runner: Any,
    *,
    gmt_observation_provider: Callable[[], torch.Tensor],
) -> FrontRESV015FrozenGMTStepPlan:
    """Advance Clean C, read its fresh observation, and run frozen GMT without another repair."""

    if getattr(runner, "_frontres_v015_one_action_k_phase", None) != "frozen":
        raise RuntimeError("v015 frozen-GMT step requires the unique t action to be sampled first")
    command = _frontres_motion_command(runner)
    advance = getattr(command, "advance_frontres_local_scenario_k_execution", None)
    if not callable(advance):
        raise RuntimeError("v015 frozen-GMT step requires command Clean-continuation advancement")
    continuation_state = advance()
    continuation = continuation_state.get("continuation") if isinstance(continuation_state, dict) else None
    valid_mask = continuation_state.get("valid_mask") if isinstance(continuation_state, dict) else None
    cursor = continuation_state.get("cursor") if isinstance(continuation_state, dict) else None
    if not callable(gmt_observation_provider):
        raise RuntimeError("v015 frozen-GMT step requires a post-advance observation provider")
    # C[offset] 必须先成为 command current reference, 然后 observation owner 才能
    # 构造同一 offset 的 GMT input. 反向顺序会产生一帧 reference lag.
    gmt_observations = gmt_observation_provider()
    if not isinstance(gmt_observations, torch.Tensor) or gmt_observations.ndim != 2:
        raise RuntimeError("v015 frozen-GMT observation provider must return a rank-2 tensor")
    batch_size = int(gmt_observations.shape[0])
    if (
        not isinstance(continuation, torch.Tensor)
        or tuple(continuation.shape) != (batch_size, 65)
        or not isinstance(valid_mask, torch.Tensor)
        or tuple(valid_mask.shape) != (batch_size,)
        or not isinstance(cursor, torch.Tensor)
        or tuple(cursor.shape) != (batch_size,)
    ):
        raise RuntimeError("v015 frozen-GMT command advance must return [N,65] C, [N] valid mask, and [N] cursor")
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    if (
        policy is None
        or not callable(getattr(policy, "_parse_observations", None))
        or not callable(getattr(policy, "_run_gmt_direct", None))
    ):
        raise RuntimeError("v015 frozen-GMT step requires the direct frozen-GMT execution adapter")
    action_dim = int(getattr(policy, "num_task_corrections", 0) or 6)
    if action_dim != 6:
        raise RuntimeError(f"v015 frozen-GMT step requires full-6D task-space FrontRES, got action_dim={action_dim}")
    # Do not call get_env_action(zero): that path records another FrontRES
    # correction state.  After t, only the command-owned C reference and the
    # frozen GMT execution adapter are allowed to advance.
    with torch.inference_mode():
        policy_obs, ref_vel, ref_vel_estimator_obs = policy._parse_observations(gmt_observations)
        env_actions = policy._run_gmt_direct(policy_obs, ref_vel, ref_vel_estimator_obs)
    if not isinstance(env_actions, torch.Tensor) or int(env_actions.shape[0]) != batch_size:
        raise RuntimeError("v015 frozen-GMT adapter must return one environment action per role row")
    return FrontRESV015FrozenGMTStepPlan(
        env_actions=env_actions.detach().clone(),
        continuation=continuation.detach().clone(),
        valid_mask=valid_mask.detach().bool().clone(),
        cursor=cursor.detach().long().clone(),
    )
