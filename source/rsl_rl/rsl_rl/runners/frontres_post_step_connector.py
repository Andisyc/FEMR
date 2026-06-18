# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-step FrontRES reward/evidence connector.

This helper owns the runner-side bridge between rollout evidence and the reward
payload passed to ``algorithm.process_env_step``.  It does not update the
algorithm, compute returns, or write storage directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rsl_rl.frontres.frontres_reward_diagnostics import accumulate_frontres_reward_diagnostics
from rsl_rl.frontres.frontres_reward_window import (
    FrontRESRewardContext,
    FrontRESRewardWindow,
    compose_frontres_reward_delta,
)
from rsl_rl.frontres.frontres_transition_payload import FrontRESAcceptancePayload


@dataclass
class FrontRESPostStepResult:
    rewards: torch.Tensor
    reward_window: FrontRESRewardWindow | None
    r_raw_gmt: torch.Tensor | None
    r_candidate_gmt: torch.Tensor | None
    r_clean_gmt: torch.Tensor | None
    prev_delta_q: torch.Tensor | None
    term_count: int
    step_count: int


def apply_frontres_post_step_reward_connector(
    runner: Any,
    *,
    locs: dict[str, Any],
    reward_context: FrontRESRewardContext | None,
    accept_payload: FrontRESAcceptancePayload | None,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    actions: torch.Tensor,
    diagnostic_sums: dict[str, float | int],
    prev_delta_q: torch.Tensor | None,
    term_count: int,
    step_count: int,
) -> FrontRESPostStepResult:
    """Convert post-env FrontRES evidence into rewards and diagnostics."""

    n_train = int(locs["N_train"])
    n_candidate = int(locs["N_candidate"])
    n_base = int(locs["N_base"])
    n_clean = int(locs["N_clean"])
    base_start = reward_context.base_start if reward_context is not None else int(locs["_base_start"])
    base_end = reward_context.base_end if reward_context is not None else int(locs["_base_end"])
    is_task_space_mode = bool(locs.get("_is_task_space_mode", False))

    r_raw_gmt = reward_context.r_raw_gmt if reward_context is not None else locs.get("r_raw_gmt")
    r_candidate_gmt = (
        reward_context.r_candidate_gmt if reward_context is not None else locs.get("r_candidate_gmt")
    )
    r_clean_gmt = reward_context.r_clean_gmt if reward_context is not None else locs.get("r_clean_gmt")
    diagnostic_locs = dict(locs)

    if is_task_space_mode:
        if reward_context is None:
            raise RuntimeError("FrontRES task-space post-step connector requires reward_context.")
        reward_window = reward_context.reward_window
        if reward_window is None:
            raise RuntimeError("FrontRES task-space post-step connector requires reward_window.")
        n_exec = int(reward_context.n_exec)
        reward_window = compose_frontres_reward_delta(
            cfg=runner.cfg,
            reward_window=reward_window,
            n_train=n_train,
            n_exec=n_exec,
            n_candidate=n_candidate,
            repair_gain=reward_context.repair_gain,
            candidate_gain=reward_context.candidate_gain,
            projection_gain=reward_context.projection_gain,
            r_step=reward_context.r_step,
            r_rescue=reward_context.r_rescue,
            intervention_cost=reward_context.intervention_cost,
            overcorrection_cost=reward_context.overcorrection_cost,
            w_exec=reward_context.w_exec,
            repair_scale=reward_context.repair_scale,
            w_geom=reward_context.w_geom,
            w_rescue=reward_context.w_rescue,
            w_exec_harm=reward_context.w_exec_harm,
            device=runner.device,
        )
        r_delta = reward_window.r_delta
        under_write = reward_window.under_write
        ranking_reward = reward_window.ranking_reward
        r_frontres_log = reward_context.exec_frontres.mean()
        r_clean_log = reward_context.exec_clean.mean()
        r_oracle_log = reward_context.exec_feasible.mean()
        r_base_log = reward_context.exec_perturbed.mean()
        r_rescue_log = reward_context.r_rescue.mean()
    else:
        reward_window = None
        r_raw_gmt = rewards[base_start:base_end].view(-1).clone()
        r_total = rewards[:n_train].view(-1)
        if n_train == n_base:
            r_delta = r_total - r_raw_gmt
            r_base_log = r_raw_gmt.mean()
        else:
            r_delta = r_total - r_raw_gmt.mean()
            r_base_log = r_raw_gmt.mean()
        r_rescue_log = 0.0
        under_write = None
        ranking_reward = None
        r_frontres_log = None
        r_clean_log = None
        r_oracle_log = None
        frontres_actor_gate = torch.zeros(runner.env.num_envs, 1, device=runner.device)
        frontres_actor_gate[:n_train, 0] = 1.0
        runner.alg.transition.frontres_actor_gate = frontres_actor_gate
        diagnostic_locs.update(
            {
                "_r_z": None,
                "_r_xy": None,
                "_r_rp": None,
                "_r_ya": None,
                "_actor_gate": None,
                "_exec_planar_log": None,
                "_exec_vertical_log": None,
                "_exec_task_log": None,
                "_dr_z_abs_log": None,
                "_dr_xy_abs_log": None,
                "_dr_rp_abs_log": None,
                "_dr_yaw_abs_log": None,
                "_corr_z_abs_log": None,
                "_corr_xy_abs_log": None,
                "_corr_rp_abs_log": None,
                "_corr_yaw_abs_log": None,
                "_frontres_actor_gate": frontres_actor_gate,
            }
        )

    if reward_context is not None:
        diagnostic_locs.update(
            {
                "_n_exec": reward_context.n_exec,
                "_n_pair": reward_context.n_pair,
                "_base_start": reward_context.base_start,
                "_base_end": reward_context.base_end,
                "_candidate_start": reward_context.candidate_start,
                "_candidate_end": reward_context.candidate_end,
                "_clean_start": reward_context.clean_start,
                "_clean_end": reward_context.clean_end,
                "_r_step": reward_context.r_step,
                "_r_rescue": reward_context.r_rescue,
                "_r_exec": reward_context.r_exec,
                "_r_z": getattr(reward_context, "r_z", None),
                "_r_xy": getattr(reward_context, "r_xy", None),
                "_r_rp": getattr(reward_context, "r_rp", None),
                "_r_ya": getattr(reward_context, "r_ya", None),
                "_dr_z_abs_log": reward_context.dr_z_abs_log,
                "_dr_xy_abs_log": reward_context.dr_xy_abs_log,
                "_dr_rp_abs_log": reward_context.dr_rp_abs_log,
                "_dr_yaw_abs_log": reward_context.dr_yaw_abs_log,
                "_corr_z_abs_log": reward_context.corr_z_abs_log,
                "_corr_xy_abs_log": reward_context.corr_xy_abs_log,
                "_corr_rp_abs_log": reward_context.corr_rp_abs_log,
                "_corr_yaw_abs_log": reward_context.corr_yaw_abs_log,
                "_rot_raw_to_clean": reward_context.rot_raw_to_clean,
                "_rot_raw_to_fr": reward_context.rot_raw_to_fr,
                "_e_raw": reward_context.e_raw,
                "_e_fr": reward_context.e_fr,
                "_exec_score_all": reward_context.exec_score_all,
                "_exec_components": reward_context.exec_components,
                "_feasible_components": reward_context.feasible_components,
                "_mode_groups": reward_context.mode_groups,
                "_exec_frontres": reward_context.exec_frontres,
                "_exec_candidate": reward_context.exec_candidate,
                "_exec_perturbed": reward_context.exec_perturbed,
                "_exec_clean": reward_context.exec_clean,
                "_exec_feasible": reward_context.exec_feasible,
                "_exec_planar_log": reward_context.exec_planar_log,
                "_exec_vertical_log": reward_context.exec_vertical_log,
                "_exec_task_log": reward_context.exec_task_log,
                "_intervention_cost": reward_context.intervention_cost,
                "_clean_bound_cost": reward_context.clean_bound_cost,
                "_side_cost": reward_context.side_cost,
                "_over_cost": reward_context.over_cost,
                "_overcorrection_cost": reward_context.overcorrection_cost,
                "_under_repair_penalty": reward_context.under_repair_penalty,
                "_action_activity": reward_context.action_activity,
                "_w_exec": reward_context.w_exec,
                "_repair_scale": reward_context.repair_scale,
                "_w_geom": reward_context.w_geom,
                "_w_rescue": reward_context.w_rescue,
                "_w_exec_harm": reward_context.w_exec_harm,
                "_repair_gain": reward_context.repair_gain,
                "_candidate_gain": reward_context.candidate_gain,
                "_projection_gain": reward_context.projection_gain,
                "_oracle_ub_gain": reward_context.oracle_ub_gain,
                "_oracle_ub_pass": reward_context.oracle_ub_pass,
                "_oracle_ub_noisy_win": reward_context.oracle_ub_noisy_win,
                "_oracle_ub_projected_win": reward_context.oracle_ub_projected_win,
                "_oracle_ub_candidate_win": reward_context.oracle_ub_candidate_win,
                "_oracle_ub_feasible_win": reward_context.oracle_ub_feasible_win,
                "_exec_floor": reward_context.exec_floor,
                "_exec_safe_floor": reward_context.exec_safe_floor,
                "_exec_floor_source": reward_context.exec_floor_source,
                "_candidate_floor_margin": reward_context.candidate_floor_margin,
                "_candidate_floor_pass": reward_context.candidate_floor_pass,
                "_candidate_floor_pass_frac": reward_context.candidate_floor_pass_frac,
                "_stable_route_next": reward_context.stable_route_next,
                "_stable_route_active": reward_context.stable_route_active,
                "_damage_gap": reward_window.damage_gap if reward_window is not None else None,
                "_oracle_clean_gap": reward_window.oracle_clean_gap if reward_window is not None else None,
                "_oracle_trust": reward_window.oracle_trust if reward_window is not None else None,
                "_repair_ratio": reward_window.repair_ratio if reward_window is not None else None,
                "_safe_gate": reward_window.safe_gate if reward_window is not None else None,
                "_repair_gate": reward_window.repair_gate if reward_window is not None else None,
                "_broken_gate": reward_window.broken_gate if reward_window is not None else None,
                "_window_mu": reward_window.window_mu if reward_window is not None else None,
                "_exec_gate": reward_window.exec_gate if reward_window is not None else None,
                "_cost_gate": reward_window.cost_gate if reward_window is not None else None,
                "_safe_frac": reward_window.safe_frac if reward_window is not None else None,
                "_repair_frac": reward_window.repair_frac if reward_window is not None else None,
                "_broken_frac": reward_window.broken_frac if reward_window is not None else None,
                "_safe_gap": reward_window.safe_gap if reward_window is not None else 0.0,
                "_broken_gap": reward_window.broken_gap if reward_window is not None else 0.0,
                "_learnable_route_mask": reward_window.learnable_route_mask if reward_window is not None else None,
                "_exec_weight": reward_window.exec_weight if reward_window is not None else None,
                "_cost_weight": reward_window.cost_weight if reward_window is not None else None,
                "_actor_gate": reward_window.actor_gate if reward_window is not None else None,
                "_harm_penalty": reward_window.harm_penalty if reward_window is not None else None,
                "_harm_penalty_exec": reward_window.harm_penalty_exec if reward_window is not None else None,
                "_harm_mag": reward_window.harm_mag if reward_window is not None else None,
                "_cost_exec": reward_window.cost_exec if reward_window is not None else None,
                "_effective_gain_bonus": reward_window.effective_gain_bonus if reward_window is not None else None,
                "_effective_gain_bonus_exec": reward_window.effective_gain_bonus_exec if reward_window is not None else None,
                "_reward_progress": reward_window.reward_progress if reward_window is not None else 0.0,
                "_constraint_progress": reward_window.constraint_progress if reward_window is not None else 0.0,
            }
        )
    if accept_payload is not None:
        diagnostic_locs.update(
            {
                "_accept_pref_target": accept_payload.accept_target,
                "_accept_pref_mask": accept_payload.accept_mask,
                "_pref_full_frac": accept_payload.pref_full_frac,
                "_pref_noop_frac": accept_payload.pref_noop_frac,
                "_pref_keep_frac": accept_payload.pref_keep_frac,
                "_pref_ignore_frac": accept_payload.pref_ignore_frac,
                "_pref_margin_mean": accept_payload.pref_margin_mean,
                "_pref_need_mean": accept_payload.pref_need_mean,
                "_pref_admiss_mean": accept_payload.pref_admiss_mean,
                "_pref_target_mean": accept_payload.pref_target_mean,
                "_tri_weight_repair_mean": accept_payload.tri_weight_repair_mean,
                "_tri_weight_noisy_mean": accept_payload.tri_weight_noisy_mean,
                "_tri_weight_stable_mean": accept_payload.tri_weight_stable_mean,
                "_pref_inertial_penalty_rho_mean": accept_payload.pref_inertial_penalty_rho_mean,
                "_pref_inertial_penalty_one_mean": accept_payload.pref_inertial_penalty_one_mean,
                "_rho_target_planar_mean": accept_payload.rho_target_planar_mean,
                "_rho_target_rp_mean": accept_payload.rho_target_rp_mean,
                "_rho_target_z_mean": accept_payload.rho_target_z_mean,
                "_rho_target_spread_mean": accept_payload.rho_target_spread_mean,
                "_grouped_rho_mask_mean": accept_payload.grouped_rho_mask_mean,
                "_rho_regret_up_planar_mean": accept_payload.rho_regret_up_planar_mean,
                "_rho_regret_up_rp_mean": accept_payload.rho_regret_up_rp_mean,
                "_rho_regret_up_z_mean": accept_payload.rho_regret_up_z_mean,
                "_rho_regret_down_planar_mean": accept_payload.rho_regret_down_planar_mean,
                "_rho_regret_down_rp_mean": accept_payload.rho_regret_down_rp_mean,
                "_rho_regret_down_z_mean": accept_payload.rho_regret_down_z_mean,
            }
        )

    if r_delta is None:
        raise RuntimeError("FrontRES post-step connector failed to compose r_delta.")

    rewards_mod = rewards.clone()
    if rewards_mod.dim() == 2:
        rewards_mod[:n_train] = r_delta.unsqueeze(-1)
        rewards_mod[n_train:] = 0.0
    else:
        rewards_mod[:n_train] = r_delta
        rewards_mod[n_train:] = 0.0
    rewards = rewards_mod

    lambda_smooth = float(getattr(runner.alg, "lambda_smooth", 0.0))
    smooth_penalty = None
    if lambda_smooth > 0.0 and prev_delta_q is not None:
        diff = actions[:n_train] - prev_delta_q[:n_train]
        smooth_penalty = -lambda_smooth * diff.pow(2).mean(dim=-1)
        if rewards.dim() == 2:
            rewards[:n_train] = rewards[:n_train] + smooth_penalty.unsqueeze(-1)
        else:
            rewards[:n_train] = rewards[:n_train] + smooth_penalty

    reg_penalty = None
    lambda_reg = float(locs.get("_lambda_reg", 0.0))
    if lambda_reg > 0.0 and bool(locs.get("_dr_done", False)):
        reg_penalty = -lambda_reg * actions[:n_train].pow(2).mean(dim=-1)
        if rewards.dim() == 2:
            rewards[:n_train] = rewards[:n_train] + reg_penalty.unsqueeze(-1)
        else:
            rewards[:n_train] = rewards[:n_train] + reg_penalty

    diagnostic_locs.update(
        {
            "r_delta": r_delta,
            "r_raw_gmt": r_raw_gmt,
            "r_candidate_gmt": r_candidate_gmt,
            "r_clean_gmt": r_clean_gmt,
            "_reward_window": reward_window,
            "_under_write": under_write,
            "_ranking_reward": ranking_reward,
            "_r_frontres_log": r_frontres_log,
            "_r_clean_log": r_clean_log,
            "_r_oracle_log": r_oracle_log,
            "_r_base_log": r_base_log,
            "_r_rescue_log": r_rescue_log,
            "_smooth_penalty": smooth_penalty,
            "_reg_penalty": reg_penalty,
        }
    )
    accumulate_frontres_reward_diagnostics(runner, diagnostic_sums, diagnostic_locs)

    term_count += int((dones[:n_train] > 0).sum().item())
    step_count += n_train

    done_mask = dones.bool().view(-1)
    if prev_delta_q is None:
        prev_delta_q = actions.clone()
    else:
        prev_delta_q = actions.clone()
        prev_delta_q[done_mask] = 0.0

    prev_pos_c = getattr(runner, "_prev_pos_correction", None)
    if prev_pos_c is not None:
        frontres_done_mask = done_mask[:n_train]
        if frontres_done_mask.any():
            prev_pos_c[frontres_done_mask] = 0.0
            runner._prev_pos_correction = prev_pos_c

    return FrontRESPostStepResult(
        rewards=rewards,
        reward_window=reward_window,
        r_raw_gmt=r_raw_gmt,
        r_candidate_gmt=r_candidate_gmt,
        r_clean_gmt=r_clean_gmt,
        prev_delta_q=prev_delta_q,
        term_count=term_count,
        step_count=step_count,
    )


# Readability alias used by the runner.
compute_frontres_reward = apply_frontres_post_step_reward_connector
