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
from typing import Any

import torch


@dataclass(frozen=True)
class FrontRESRolloutStepPlan:
    actions: torch.Tensor | None
    env_actions: torch.Tensor
    hsl_pos_snapshot: torch.Tensor | None
    hsl_quat_snapshot: torch.Tensor | None


def _frontres_authority_enabled(
    runner: Any,
    *,
    is_frontres: bool,
    is_task_space_mode: bool,
) -> bool:
    if not (is_frontres and is_task_space_mode):
        return False
    enabled_fn = getattr(runner.alg, "_authority_actor_critic_enabled", None)
    if callable(enabled_fn):
        return bool(enabled_fn())
    return bool(getattr(runner.alg, "frontres_authority_actor_critic_enabled", False))


def _frontres_authority_active_mask(runner: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    mask_fn = getattr(runner.alg, "_authority_active_task_dim_mask", None)
    if callable(mask_fn):
        return mask_fn(device=device, dtype=dtype)
    active_dims = getattr(runner.alg, "frontres_active_task_dims", runner.cfg.get("frontres_active_task_dims", None))
    if active_dims is None:
        return None
    dim_mask = torch.zeros(6, device=device, dtype=dtype)
    for idx in active_dims:
        idx = int(idx)
        if 0 <= idx < 6:
            dim_mask[idx] = 1.0
        elif 6 <= idx < 12:
            dim_mask[idx - 6] = 1.0
    return dim_mask


def _write_frontres_authority_transition_inputs(
    runner: Any,
    *,
    proposal_delta_se: torch.Tensor,
    authority_rho: torch.Tensor,
    event_start: torch.Tensor,
    event_active: torch.Tensor,
    event_step: torch.Tensor,
    event_duration: torch.Tensor,
    n_train: int,
) -> None:
    num_envs = int(runner.env.num_envs)
    n = max(0, min(int(n_train), num_envs, proposal_delta_se.shape[0], authority_rho.shape[0]))
    proposal_field = torch.zeros(num_envs, 6, device=runner.device, dtype=proposal_delta_se.dtype)
    rho_field = torch.zeros(num_envs, 6, device=runner.device, dtype=authority_rho.dtype)
    mask_field = torch.zeros(num_envs, 1, device=runner.device, dtype=authority_rho.dtype)
    event_start_field = torch.zeros(num_envs, 1, device=runner.device, dtype=authority_rho.dtype)
    event_active_field = torch.zeros(num_envs, 1, device=runner.device, dtype=authority_rho.dtype)
    event_step_field = torch.zeros(num_envs, 1, device=runner.device, dtype=authority_rho.dtype)
    event_duration_field = torch.zeros(num_envs, 1, device=runner.device, dtype=authority_rho.dtype)
    if n > 0:
        proposal_field[:n] = proposal_delta_se[:n].detach()
        rho_field[:n] = authority_rho[:n].detach()
        event_start_f = event_start[:n].view(-1, 1).to(device=runner.device, dtype=authority_rho.dtype)
        event_active_f = event_active[:n].view(-1, 1).to(device=runner.device, dtype=authority_rho.dtype)
        event_step_f = event_step[:n].view(-1, 1).to(device=runner.device, dtype=authority_rho.dtype)
        event_duration_f = event_duration[:n].view(-1, 1).to(device=runner.device, dtype=authority_rho.dtype)
        mask_field[:n] = event_start_f
        event_start_field[:n] = event_start_f
        event_active_field[:n] = event_active_f
        event_step_field[:n] = event_step_f
        event_duration_field[:n] = event_duration_f
    runner.alg.transition.proposal_delta_se = proposal_field
    runner.alg.transition.authority_action = rho_field
    runner.alg.transition.authority_rho = rho_field
    runner.alg.transition.authority_log_prob = torch.zeros(num_envs, 1, device=runner.device, dtype=authority_rho.dtype)
    runner.alg.transition.authority_mask = mask_field
    runner.alg.transition.authority_event_start = event_start_field
    runner.alg.transition.authority_event_active = event_active_field
    runner.alg.transition.authority_event_step = event_step_field
    runner.alg.transition.authority_event_duration = event_duration_field
    runner._frontres_authority_live_last = {
        "proposal_abs_mean": float(proposal_field[:n].abs().mean().item()) if n > 0 else 0.0,
        "rho_mean": float(rho_field[:n].mean().item()) if n > 0 else 0.0,
        "active_frac": float(event_active_field[:n].mean().item()) if n > 0 else 0.0,
        "query_frac": float(event_start_field[:n].mean().item()) if n > 0 else 0.0,
        "event_step_mean": float(event_step_field[:n].mean().item()) if n > 0 else 0.0,
        "event_duration_mean": float(event_duration_field[:n].mean().item()) if n > 0 else 0.0,
    }


def _current_frontres_authority_event(
    runner: Any,
    *,
    num_envs: int,
    n_train: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the environment-side perturbation event state for authority learning.

    When the environment does not expose temporal events, fall back to the old
    one-frame behavior so non-burst experiments remain unchanged.
    """

    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    motion_command = None
    if hasattr(env, "command_manager") and "motion" in env.command_manager._terms:
        motion_command = env.command_manager._terms["motion"]
    perturber = getattr(motion_command, "perturber", None)
    if perturber is not None and hasattr(perturber, "frontres_authority_event_state"):
        mode = str(getattr(getattr(perturber, "cfg", None), "iid_temporal_mode", "legacy")).lower()
        if mode == "legacy":
            event_start = torch.zeros(num_envs, device=device, dtype=torch.bool)
            event_active = torch.zeros(num_envs, device=device, dtype=torch.bool)
            event_start[:n_train] = True
            event_active[:n_train] = True
            event_step = torch.zeros(num_envs, device=device, dtype=dtype)
            event_duration = torch.ones(num_envs, device=device, dtype=dtype)
            return event_start, event_active, event_step, event_duration
        state = perturber.frontres_authority_event_state(num_envs)
        event_start = state["event_start"].to(device=device, dtype=torch.bool)
        event_active = state["event_active"].to(device=device, dtype=torch.bool)
        event_step = state["event_step"].to(device=device, dtype=dtype)
        event_duration = state["event_duration"].to(device=device, dtype=dtype)
        if event_start.numel() == num_envs:
            return event_start, event_active, event_step, event_duration

    event_start = torch.zeros(num_envs, device=device, dtype=torch.bool)
    event_active = torch.zeros(num_envs, device=device, dtype=torch.bool)
    event_start[:n_train] = True
    event_active[:n_train] = True
    event_step = torch.zeros(num_envs, device=device, dtype=dtype)
    event_duration = torch.ones(num_envs, device=device, dtype=dtype)
    return event_start, event_active, event_step, event_duration


def _ensure_frontres_authority_event_cache(
    runner: Any,
    *,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proposal = getattr(runner, "_frontres_authority_event_proposal", None)
    rho = getattr(runner, "_frontres_authority_event_rho", None)
    valid = getattr(runner, "_frontres_authority_event_valid", None)
    if proposal is None or proposal.shape != (num_envs, 6) or proposal.device != device:
        proposal = torch.zeros(num_envs, 6, device=device, dtype=dtype)
    if rho is None or rho.shape != (num_envs, 6) or rho.device != device:
        rho = torch.zeros(num_envs, 6, device=device, dtype=dtype)
    if valid is None or valid.shape != (num_envs,) or valid.device != device:
        valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
    runner._frontres_authority_event_proposal = proposal
    runner._frontres_authority_event_rho = rho
    runner._frontres_authority_event_valid = valid
    return proposal, rho, valid


def _apply_frontres_authority_rollout_action(
    runner: Any,
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    is_frontres: bool,
    is_task_space_mode: bool,
    n_train: int,
    rollout_step: int,
) -> torch.Tensor:
    if not _frontres_authority_enabled(
        runner,
        is_frontres=is_frontres,
        is_task_space_mode=is_task_space_mode,
    ):
        return actions
    policy = runner.alg.policy
    if actions.shape[-1] < 12 or int(getattr(policy, "task_conf_dim", 0)) != 6:
        raise RuntimeError(
            "FrontRES authority actor-critic requires 12D task action "
            "[proposal_delta_se(6), authority_rho(6)]."
        )
    if getattr(policy, "authority_actor", None) is None:
        raise RuntimeError("FrontRES authority actor-critic is enabled but policy.authority_actor is missing.")

    num_envs = int(runner.env.num_envs)
    proposal_delta_se = actions[:, :6].detach()
    active_mask = _frontres_authority_active_mask(runner, device=actions.device, dtype=actions.dtype)
    event_start, event_active, event_step, event_duration = _current_frontres_authority_event(
        runner,
        num_envs=num_envs,
        n_train=n_train,
        device=actions.device,
        dtype=actions.dtype,
    )
    cached_proposal, cached_rho, cached_valid = _ensure_frontres_authority_event_cache(
        runner,
        num_envs=num_envs,
        device=actions.device,
        dtype=actions.dtype,
    )
    inactive = ~event_active
    cached_valid[inactive] = False
    cached_proposal[inactive] = 0.0
    cached_rho[inactive] = 0.0
    query = event_active & (event_start | (~cached_valid))
    if query.any():
        fresh_rho = policy.get_authority_rho(
            obs,
            proposal_delta_se,
            active_task_dims=active_mask,
            detach_proposal=True,
        ).detach()
        cached_proposal[query] = proposal_delta_se[query]
        cached_rho[query] = fresh_rho[query]
        cached_valid[query] = True
    authority_rho = cached_rho.clone()
    proposal_for_execution = cached_proposal.clone()
    authority_actions = actions.clone()
    authority_actions[:n_train, :6] = proposal_for_execution[:n_train]
    authority_actions[:n_train, 6:12] = authority_rho[:n_train]
    runner.alg.transition.actions = authority_actions.detach()
    _write_frontres_authority_transition_inputs(
        runner,
        proposal_delta_se=proposal_for_execution,
        authority_rho=authority_rho,
        event_start=event_start,
        event_active=event_active,
        event_step=event_step,
        event_duration=event_duration,
        n_train=n_train,
    )
    if bool(runner.cfg.get("frontres_authority_live_debug", False)) and rollout_step == 0:
        stats = getattr(runner, "_frontres_authority_live_last", {})
        print(
            "[FrontRES authority live] "
            f"proposal_abs={stats.get('proposal_abs_mean', 0.0):.4f} "
            f"rho={stats.get('rho_mean', 0.0):.4f} "
            f"active={stats.get('active_frac', 0.0):.3f} "
            f"query={stats.get('query_frac', 0.0):.3f} "
            f"duration={stats.get('event_duration_mean', 0.0):.1f}"
        )
    return authority_actions


def _motion_groups_for_runner(runner: Any) -> torch.Tensor | None:
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env, "command_manager") and "motion" in env.command_manager._terms):
        return None
    motion_command = env.command_manager._terms["motion"]
    if hasattr(motion_command, "env_motion_groups"):
        return motion_command.env_motion_groups.clone()
    return None


def _state_alpha_policy_obs(runner: Any, obs: torch.Tensor, ref_vel_estimator_obs: torch.Tensor | None) -> torch.Tensor:
    if (
        getattr(runner.alg, "use_estimate_ref_vel", False)
        and getattr(runner.alg, "ref_vel_estimator", None) is not None
    ):
        estimator_input = ref_vel_estimator_obs if ref_vel_estimator_obs is not None else obs
        estimated = runner.alg.ref_vel_estimator(estimator_input)
        return torch.cat([obs, estimated], dim=-1)
    return obs


def _refresh_frontres_state_alpha_route(
    runner: Any,
    *,
    obs: torch.Tensor,
    ref_vel_estimator_obs: torch.Tensor | None,
    iteration: int,
    is_frontres: bool,
    is_task_space_mode: bool,
    n_train: int,
    n_base: int,
) -> None:
    if not (is_frontres and is_task_space_mode):
        return
    if (
        bool(runner.cfg.get("frontres_state_alpha_enabled", True))
        and hasattr(runner.alg.policy, "get_state_router_alpha")
    ):
        n_alpha_route = min(n_train, n_base)
        if n_alpha_route > 0:
            alpha_obs = _state_alpha_policy_obs(runner, obs, ref_vel_estimator_obs)
            alpha_prob = runner.alg.policy.get_state_router_alpha(alpha_obs[:n_alpha_route]).view(-1).detach()
            if alpha_prob.numel() < n_train:
                alpha_prob = torch.nn.functional.pad(alpha_prob, (0, n_train - alpha_prob.numel()), value=0.0)
            else:
                alpha_prob = alpha_prob[:n_train]
            alpha_route = alpha_prob >= float(runner.cfg.get("frontres_state_alpha_route_threshold", 0.70))
            min_iter = int(runner.cfg.get("frontres_state_alpha_route_min_iteration", 0))
            if int(iteration) < min_iter:
                alpha_route = torch.zeros_like(alpha_route, dtype=torch.bool)
            if not bool(runner.cfg.get("frontres_state_alpha_route_enabled", True)):
                alpha_route = torch.zeros_like(alpha_route, dtype=torch.bool)
            runner._frontres_stable_route_next_mask = alpha_route.detach()
            runner._frontres_state_alpha_prob_next = alpha_prob.detach()
            runner._frontres_state_alpha_pred_last = float(alpha_prob.mean().item())
            runner._frontres_state_alpha_route_last = float(alpha_route.float().mean().item())
            return
        runner._frontres_stable_route_next_mask = torch.zeros(0, device=runner.device, dtype=torch.bool)
        runner._frontres_state_alpha_prob_next = torch.zeros(0, device=runner.device)
        runner._frontres_state_alpha_pred_last = 0.0
        runner._frontres_state_alpha_route_last = 0.0
        return

    runner._frontres_stable_route_next_mask = torch.zeros(n_train, device=runner.device, dtype=torch.bool)
    runner._frontres_state_alpha_prob_next = torch.zeros(n_train, device=runner.device)
    runner._frontres_state_alpha_pred_last = 0.0
    runner._frontres_state_alpha_route_last = 0.0


def _record_velocity_estimator_error(runner: Any, vel_est_error_buffer: Any) -> None:
    if not (hasattr(runner.alg, "last_estimated_ref_vel") and runner.alg.last_estimated_ref_vel is not None):
        return
    from whole_body_tracking.tasks.tracking.mdp import observations as mdp

    gt_ref_vel_b = mdp.ref_base_lin_vel_b(runner.env.unwrapped, "motion")
    vel_error = (runner.alg.last_estimated_ref_vel - gt_ref_vel_b).abs().mean(dim=-1)
    vel_est_error_buffer.extend(vel_error.cpu().numpy().tolist())


def _rewrite_task_space_log_prob(runner: Any, actions: torch.Tensor) -> None:
    runner.alg.transition.actions = actions.detach()
    if hasattr(runner.alg, "_get_actor_log_prob"):
        runner.alg.transition.actions_log_prob = runner.alg._get_actor_log_prob(actions).detach()
    else:
        runner.alg.transition.actions_log_prob = runner.alg.policy.get_actions_log_prob(actions).detach()


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
            allow_oracle=True,
            n_candidate=n_candidate if is_frontres else 0,
        )
        obs_corr, extras_corr = runner.env.get_observations()
        obs_corr_dict = extras_corr.get("observations", {})
        if runner.policy_obs_type is not None and runner.policy_obs_type in obs_corr_dict:
            obs_corr = obs_corr_dict[runner.policy_obs_type]
        obs_corr = runner._apply_obs_normalizer(obs_corr.to(runner.device))
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
    if not (is_task_space_mode and getattr(runner.alg, "lambda_supervised", 0.0) > 0):
        return
    env_for_sup = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    for cmd_sup in env_for_sup.command_manager._terms.values():
        if hasattr(cmd_sup, "supervised_target"):
            sup_target = cmd_sup.supervised_target.clone().to(runner.device)
            sup_target = runner._frontres_action_cone.project_task_target(cmd_sup, sup_target)
            if bool(runner.cfg.get("frontres_per_mode_supervised_mask", True)):
                sup_mode_groups = list(
                    getattr(
                        runner,
                        "_frontres_curriculum_env_mode_groups",
                        [tuple(getattr(runner, "_frontres_curriculum_active_modes", ()))] * n_train,
                    )
                )[:n_train]
                sup_target = runner._frontres_action_cone.apply_per_mode_supervised_mask(
                    sup_target, sup_mode_groups, n_train
                )
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
            actions = runner._mask_frontres_task_actions(actions)
            _rewrite_task_space_log_prob(runner, actions)
            actions = _apply_frontres_authority_rollout_action(
                runner,
                obs=obs,
                actions=actions,
                is_frontres=is_frontres,
                is_task_space_mode=is_task_space_mode,
                n_train=n_train,
                rollout_step=rollout_step,
            )
        _refresh_frontres_state_alpha_route(
            runner,
            obs=obs,
            ref_vel_estimator_obs=ref_vel_estimator_obs,
            iteration=iteration,
            is_frontres=is_frontres,
            is_task_space_mode=is_task_space_mode,
            n_train=n_train,
            n_base=n_base,
        )
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
            actions = runner._mask_frontres_task_actions(actions)
            _rewrite_task_space_log_prob(runner, actions)
            actions = _apply_frontres_authority_rollout_action(
                runner,
                obs=obs,
                actions=actions,
                is_frontres=is_frontres,
                is_task_space_mode=is_task_space_mode,
                n_train=n_train,
                rollout_step=rollout_step,
            )
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
