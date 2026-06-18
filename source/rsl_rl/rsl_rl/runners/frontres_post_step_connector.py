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
from rsl_rl.frontres.frontres_reward_window import FrontRESRewardWindow, compose_frontres_reward_delta


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
    rewards: torch.Tensor,
    dones: torch.Tensor,
    actions: torch.Tensor,
    reward_window: FrontRESRewardWindow | None,
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
    base_start = int(locs["_base_start"])
    base_end = int(locs["_base_end"])
    is_task_space_mode = bool(locs.get("_is_task_space_mode", False))

    r_raw_gmt = locs.get("r_raw_gmt")
    r_candidate_gmt = locs.get("r_candidate_gmt")
    r_clean_gmt = locs.get("r_clean_gmt")
    diagnostic_locs = dict(locs)

    if is_task_space_mode:
        if reward_window is None:
            raise RuntimeError("FrontRES task-space post-step connector requires reward_window.")
        n_exec = int(locs["_n_exec"])
        reward_window = compose_frontres_reward_delta(
            cfg=runner.cfg,
            reward_window=reward_window,
            n_train=n_train,
            n_exec=n_exec,
            n_candidate=n_candidate,
            repair_gain=locs["_repair_gain"],
            candidate_gain=locs["_candidate_gain"],
            projection_gain=locs["_projection_gain"],
            r_step=locs["_r_step"],
            r_rescue=locs["_r_rescue"],
            intervention_cost=locs["_intervention_cost"],
            overcorrection_cost=locs["_overcorrection_cost"],
            w_exec=locs["_w_exec"],
            repair_scale=locs["_repair_scale"],
            w_geom=locs["_w_geom"],
            w_rescue=locs["_w_rescue"],
            w_exec_harm=locs["_w_exec_harm"],
            device=runner.device,
        )
        r_delta = reward_window.r_delta
        under_write = reward_window.under_write
        ranking_reward = reward_window.ranking_reward
        r_frontres_log = locs["_exec_frontres"].mean()
        r_clean_log = locs["_exec_clean"].mean()
        r_oracle_log = locs["_exec_feasible"].mean()
        r_base_log = locs["_exec_perturbed"].mean()
        r_rescue_log = locs["_r_rescue"].mean()
    else:
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
