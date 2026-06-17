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
