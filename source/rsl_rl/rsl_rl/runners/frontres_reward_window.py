# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES rollout reward-window construction.

The reward window lives in runners because it converts post-env rollout evidence
into per-sample reward weights and actor-credit gates before process_env_step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rsl_rl.runners.frontres_oracle import compute_frontres_oracle_upper_bound
from rsl_rl.runners.frontres_rollout_evidence import compute_frontres_rollout_evidence
from rsl_rl.runners.frontres_executable_floor import update_runner_executable_floor_stats


@dataclass
class FrontRESRewardWindow:
    r_exec: torch.Tensor
    damage_gap: torch.Tensor
    oracle_clean_gap: torch.Tensor
    oracle_trust: torch.Tensor
    repair_ratio: torch.Tensor
    safe_gate: torch.Tensor
    repair_gate: torch.Tensor
    broken_gate: torch.Tensor
    window_mu: torch.Tensor
    exec_gate: torch.Tensor
    cost_gate: torch.Tensor
    safe_frac: torch.Tensor
    repair_frac: torch.Tensor
    broken_frac: torch.Tensor
    safe_gap: float
    broken_gap: float
    learnable_route_mask: torch.Tensor
    exec_weight: torch.Tensor
    cost_weight: torch.Tensor
    actor_gate: torch.Tensor
    harm_penalty: torch.Tensor
    harm_penalty_exec: torch.Tensor
    harm_mag: torch.Tensor
    cost_exec: torch.Tensor
    effective_gain_bonus: torch.Tensor
    effective_gain_bonus_exec: torch.Tensor
    under_repair_penalty: torch.Tensor
    r_delta: torch.Tensor | None
    under_write: torch.Tensor | None
    ranking_reward: torch.Tensor | None
    reward_progress: float
    constraint_progress: float


@dataclass
class FrontRESRewardContext:
    """Post-step rollout evidence used to construct FrontRES reward and payloads."""

    candidate_start: int
    candidate_end: int
    base_start: int
    base_end: int
    clean_start: int
    clean_end: int
    n_exec: int
    n_pair: int
    r_raw_gmt: torch.Tensor
    r_clean_gmt: torch.Tensor
    r_candidate_gmt: torch.Tensor | None
    r_total: torch.Tensor
    cmd: Any
    use_clean: bool
    a_w: torch.Tensor
    a_raw: torch.Tensor
    a_fr: torch.Tensor
    q_w: torch.Tensor
    q_raw: torch.Tensor
    q_fr: torch.Tensor
    r_step: torch.Tensor
    r_rescue: torch.Tensor
    r_exec: torch.Tensor
    dr_z_abs_log: torch.Tensor
    dr_xy_abs_log: torch.Tensor
    dr_rp_abs_log: torch.Tensor
    dr_yaw_abs_log: torch.Tensor
    corr_z_abs_log: torch.Tensor
    corr_xy_abs_log: torch.Tensor
    corr_rp_abs_log: torch.Tensor
    corr_yaw_abs_log: torch.Tensor
    rot_raw_to_clean: torch.Tensor
    rot_raw_to_fr: torch.Tensor
    e_raw: torch.Tensor
    e_fr: torch.Tensor
    exec_score_all: torch.Tensor
    exec_components: Mapping[str, torch.Tensor]
    feasible_components: Mapping[str, torch.Tensor]
    mode_groups: list[tuple[str, ...]]
    exec_frontres: torch.Tensor
    exec_candidate: torch.Tensor
    exec_perturbed: torch.Tensor
    exec_clean: torch.Tensor
    exec_feasible: torch.Tensor
    exec_planar_log: torch.Tensor
    exec_vertical_log: torch.Tensor
    exec_task_log: torch.Tensor
    intervention_cost: torch.Tensor
    clean_bound_cost: torch.Tensor
    side_cost: torch.Tensor
    over_cost: torch.Tensor
    overcorrection_cost: torch.Tensor
    under_repair_penalty: torch.Tensor
    action_activity: torch.Tensor
    w_exec: float
    repair_scale: float
    w_geom: float
    w_rescue: float
    w_exec_harm: float
    repair_gain: torch.Tensor
    candidate_gain: torch.Tensor
    projection_gain: torch.Tensor
    oracle_ub_gain: torch.Tensor
    oracle_ub_pass: torch.Tensor
    oracle_ub_noisy_win: torch.Tensor
    oracle_ub_projected_win: torch.Tensor
    oracle_ub_candidate_win: torch.Tensor
    oracle_ub_feasible_win: torch.Tensor
    exec_floor: float
    exec_safe_floor: float
    exec_floor_source: str
    candidate_floor_margin: torch.Tensor
    candidate_floor_pass: torch.Tensor
    candidate_floor_pass_frac: torch.Tensor
    stable_route_next: torch.Tensor
    stable_route_active: torch.Tensor
    reward_window: FrontRESRewardWindow


def frontres_family_gain_std(
    runner: Any,
    mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
    gain: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample gain std from per-family EMA stats, then update stats."""

    if gain.numel() == 0:
        return torch.empty_like(gain)
    cfg = runner.cfg
    init_std = float(cfg.get("frontres_family_gain_initial_std", 0.01))
    min_std = float(cfg.get("frontres_family_gain_min_std", 0.002))
    alpha = float(cfg.get("frontres_family_gain_ema_alpha", 0.05))
    alpha = max(0.0, min(1.0, alpha))
    stats = getattr(runner, "_frontres_family_gain_stats", None)
    if stats is None:
        stats = {}
        runner._frontres_family_gain_stats = stats

    mode_groups_list = list(mode_groups)[: gain.shape[0]]
    if len(mode_groups_list) < gain.shape[0]:
        fallback_modes = ("planar", "yaw", "global_z", "local_rp")
        mode_groups_list.extend([fallback_modes] * (gain.shape[0] - len(mode_groups_list)))
    std = torch.full_like(gain, max(init_std, min_std))
    for idx, modes in enumerate(mode_groups_list):
        families = tuple(m for m in modes if m in ("planar", "yaw", "global_z", "local_rp"))
        if not families:
            families = ("all",)
        vals = []
        for family in families:
            entry = stats.get(family)
            if entry is None:
                vals.append(max(init_std, min_std))
            else:
                vals.append(max(float(entry.get("std", init_std)), min_std))
        std[idx] = sum(vals) / float(len(vals))

    with torch.no_grad():
        gain_detached = gain.detach()
        for family in ("planar", "yaw", "global_z", "local_rp"):
            mask_vals = [family in set(modes) for modes in mode_groups_list]
            if not any(mask_vals):
                continue
            mask = torch.tensor(mask_vals, device=gain.device, dtype=torch.bool)
            values = gain_detached[mask]
            if values.numel() == 0:
                continue
            batch_mean = values.mean().item()
            batch_var = values.var(unbiased=False).item() if values.numel() > 1 else 0.0
            entry = stats.get(family)
            if entry is None:
                mean = batch_mean
                var = max(batch_var, init_std * init_std)
            else:
                old_mean = float(entry.get("mean", 0.0))
                old_var = float(entry.get("var", init_std * init_std))
                mean = (1.0 - alpha) * old_mean + alpha * batch_mean
                var = (1.0 - alpha) * old_var + alpha * batch_var
            stats[family] = {
                "mean": mean,
                "var": max(var, min_std * min_std),
                "std": max(math.sqrt(max(var, 0.0)), min_std),
            }
    return std.clamp(min=min_std)


def build_frontres_reward_context(
    runner: Any,
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    infos: dict[str, Any],
    actions: torch.Tensor,
    n_train: int,
    n_candidate: int,
    n_base: int,
    n_clean: int,
    is_task_space_mode: bool,
    dr_scale: float,
    ppo_actor_weight_current: float,
    quat_to_rotvec_wxyz: Any,
    quat_mul_fn: Any,
    quat_inv_fn: Any,
    euler_xyz_from_quat_fn: Any,
    device: torch.device,
) -> FrontRESRewardContext | None:
    """Build post-env FrontRES reward evidence and reward window."""

    candidate_start = n_train
    candidate_end = candidate_start + n_candidate
    base_start = candidate_end
    base_end = base_start + n_base
    clean_start = base_end
    clean_end = clean_start + n_clean
    r_raw_gmt = rewards[base_start:base_end].view(-1).clone()
    r_clean_gmt = rewards[clean_start:clean_end].view(-1).clone()
    r_candidate_gmt = (
        rewards[candidate_start:candidate_end].view(-1).clone()
        if n_candidate > 0
        else None
    )
    r_total = rewards[:n_train].view(-1).clone()

    env_for_rdelta = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    cmd = env_for_rdelta.command_manager._terms.get("motion")
    use_clean = (
        cmd is not None
        and hasattr(cmd, "anchor_pos_w_original")
        and hasattr(cmd, "anchor_quat_w_original")
    )
    if not use_clean:
        return None

    a_w = cmd.anchor_pos_w_original
    a_raw = cmd.anchor_pos_w_raw
    a_fr = cmd.anchor_pos_w
    q_w = cmd.anchor_quat_w_original
    q_raw = cmd.anchor_quat_w_raw
    q_fr = cmd.anchor_quat_w

    def r_axis(dr, corr):
        return dr.abs() - (dr + corr).abs()

    def r_vec(dr_vec, corr_vec):
        return dr_vec.norm(dim=-1) - (dr_vec + corr_vec).norm(dim=-1)

    dr_z_fr = a_raw[:n_train, 2] - a_w[:n_train, 2]
    corr_z_fr = a_fr[:n_train, 2] - a_raw[:n_train, 2]
    r_z = r_axis(dr_z_fr, corr_z_fr)
    dr_xy_fr = a_raw[:n_train, :2] - a_w[:n_train, :2]
    corr_xy_fr = a_fr[:n_train, :2] - a_raw[:n_train, :2]
    r_xy = r_vec(dr_xy_fr, corr_xy_fr)

    def wrap_pi(angle: torch.Tensor):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    rot_raw_to_clean = quat_to_rotvec_wxyz(quat_mul_fn(quat_inv_fn(q_raw[:n_train]), q_w[:n_train]))
    rot_fr_to_clean = quat_to_rotvec_wxyz(quat_mul_fn(quat_inv_fn(q_fr[:n_train]), q_w[:n_train]))
    rot_raw_to_fr = quat_to_rotvec_wxyz(quat_mul_fn(quat_inv_fn(q_raw[:n_train]), q_fr[:n_train]))
    rp_raw = rot_raw_to_clean[:, :2]
    rp_fr = rot_fr_to_clean[:, :2]
    e_raw = rp_raw.norm(dim=-1)
    e_fr = rp_fr.norm(dim=-1)
    r_rp = e_raw - e_fr
    _, _, yaw_raw = euler_xyz_from_quat_fn(q_raw[:n_train])
    _, _, yaw_fr = euler_xyz_from_quat_fn(q_fr[:n_train])
    _, _, yaw_w = euler_xyz_from_quat_fn(q_w[:n_train])
    yaw_err_raw = wrap_pi(yaw_raw - yaw_w)
    yaw_corr = wrap_pi(yaw_fr - yaw_raw)
    r_ya = r_axis(yaw_err_raw, yaw_corr)

    restore_z_weight = float(runner.cfg.get("frontres_restore_z_weight", 0.3))
    restore_xy_weight = float(runner.cfg.get("frontres_restore_xy_weight", 0.3))
    restore_rp_weight = float(runner.cfg.get("frontres_restore_rp_weight", 0.15))
    restore_yaw_weight = float(runner.cfg.get("frontres_restore_yaw_weight", 0.02))
    r_step = (
        restore_z_weight * r_z
        + restore_xy_weight * r_xy
        + restore_rp_weight * r_rp
        + restore_yaw_weight * r_ya
    )
    dr_z_abs_log = dr_z_fr.abs().mean()
    dr_xy_abs_log = dr_xy_fr.norm(dim=-1).mean()
    dr_rp_abs_log = e_raw.mean()
    dr_yaw_abs_log = yaw_err_raw.abs().mean()
    corr_z_abs_log = corr_z_fr.abs().mean()
    corr_xy_abs_log = corr_xy_fr.norm(dim=-1).mean()
    corr_rp_abs_log = rot_raw_to_fr[:, :2].norm(dim=-1).mean()
    corr_yaw_abs_log = yaw_corr.abs().mean()

    n_pair = min(n_train, n_candidate if n_candidate > 0 else n_train, n_base, n_clean)
    fell_base = dones[base_start:base_start + n_pair].view(-1) > 0
    fell_fr = dones[:n_pair].view(-1) > 0
    r_rescue = torch.zeros(n_train, device=device)
    r_rescue_pair = torch.zeros(n_pair, device=device)
    rescue_mag = float(runner.cfg.get("r_rescue_magnitude", 0.5))
    r_rescue_pair[fell_base & ~fell_fr] = rescue_mag
    r_rescue_pair[fell_base & fell_fr] = -0.1 * rescue_mag
    r_rescue_pair[~fell_base & fell_fr] = -rescue_mag
    r_rescue[:n_pair] = r_rescue_pair

    r_exec = torch.zeros(n_train, device=device)
    n_exec = min(n_train, n_candidate if n_candidate > 0 else n_train, n_base, n_clean)
    executability = runner._frontres_executability
    active_modes = tuple(getattr(runner, "_frontres_curriculum_active_modes", ()))
    exec_score_all, exec_components = executability.exec_score(cmd, return_components=True)
    mode_groups = list(getattr(
        runner,
        "_frontres_curriculum_env_mode_groups",
        [tuple(getattr(runner, "_frontres_curriculum_active_modes", ()))] * n_exec,
    ))[:n_exec]
    exec_frontres = executability.exec_score_for_modes(
        exec_components, 0, n_exec, mode_groups=mode_groups, active_modes=active_modes
    )
    if n_candidate > 0:
        exec_candidate = executability.exec_score_for_modes(
            exec_components, candidate_start, n_exec, mode_groups=mode_groups, active_modes=active_modes
        )
    else:
        exec_candidate = exec_frontres.detach()
    exec_perturbed = executability.exec_score_for_modes(
        exec_components, base_start, n_exec, mode_groups=mode_groups, active_modes=active_modes
    )
    exec_clean = executability.exec_score_for_modes(
        exec_components, clean_start, n_exec, mode_groups=mode_groups, active_modes=active_modes
    )
    _, feasible_components = executability.feasible_oracle_exec_score(
        cmd, base_start, n_exec, return_components=True
    )
    exec_feasible = executability.exec_score_for_modes(
        feasible_components, 0, n_exec, mode_groups=mode_groups, active_modes=active_modes
    ).to(device).view(-1)
    exec_planar_log = exec_components["planar"][:n_exec].mean()
    exec_vertical_log = exec_components["vertical"][:n_exec].mean()
    exec_task_log = exec_components["task"][:n_exec].mean()

    intervention_cost = torch.zeros(n_train, device=device)
    clean_bound_cost = torch.zeros(n_train, device=device)
    side_cost = torch.zeros(n_train, device=device)
    over_cost = torch.zeros(n_train, device=device)
    under_repair_penalty = torch.zeros(n_train, device=device)
    action_activity = torch.zeros(n_train, device=device)
    if is_task_space_mode and actions.shape[-1] >= 6:
        delta = actions[:n_train, :6]
        max_delta = torch.tensor(
            [
                runner.alg.policy.max_delta_pos,
                runner.alg.policy.max_delta_pos,
                runner.alg.policy.max_delta_pos,
                runner.alg.policy.max_delta_rpy,
                runner.alg.policy.max_delta_rpy,
                runner.alg.policy.max_delta_rpy,
            ],
            device=device,
            dtype=delta.dtype,
        ).clamp(min=1e-6)
        weights = torch.tensor(
            runner.cfg.get(
                "frontres_intervention_cost_weights",
                [0.02, 0.02, 0.05, 0.30, 0.30, 0.10],
            ),
            device=device,
            dtype=delta.dtype,
        )
        intervention_cost = (weights * (delta / max_delta).pow(2)).sum(dim=-1)
        active_dims_cfg = runner.cfg.get("frontres_active_task_dims", None)
        if active_dims_cfg is not None:
            active_delta_dims = [
                int(idx) for idx in active_dims_cfg
                if 0 <= int(idx) < min(6, delta.shape[-1])
            ]
        else:
            active_delta_dims = list(range(min(6, delta.shape[-1])))
        if active_delta_dims:
            active_idx = torch.tensor(active_delta_dims, device=device, dtype=torch.long)
            action_activity = (delta[:, active_idx] / max_delta[active_idx]).pow(2).mean(dim=-1)
            target_delta = torch.cat(
                [
                    (a_w[:n_train] - a_raw[:n_train]),
                    rot_raw_to_clean,
                ],
                dim=-1,
            )
            corr_delta = torch.cat(
                [
                    (a_fr[:n_train] - a_raw[:n_train]),
                    rot_raw_to_fr,
                ],
                dim=-1,
            )
            target_active = target_delta[:, active_idx] / max_delta[active_idx]
            corr_active = corr_delta[:, active_idx] / max_delta[active_idx]
            target_norm = target_active.norm(dim=-1, keepdim=True)
            target_dir = target_active / target_norm.clamp(min=1e-6)
            parallel_scalar = (corr_active * target_dir).sum(dim=-1, keepdim=True)
            parallel = parallel_scalar * target_dir
            side = corr_active - parallel

            side_weight = float(runner.cfg.get("frontres_clean_bound_side_weight", 0.0))
            side_cost = max(side_weight, 0.0) * side.pow(2).sum(dim=-1)
            over_margin = float(runner.cfg.get("frontres_overcorrection_margin", 0.0))
            over_weight = float(runner.cfg.get("frontres_overcorrection_weight", 0.0))
            over = torch.relu(
                parallel_scalar.squeeze(-1)
                - target_norm.squeeze(-1)
                - max(over_margin, 0.0)
            )
            over_cost = max(over_weight, 0.0) * over.pow(2)
            clean_bound_cost = side_cost + over_cost
    overcorrection_cost = clean_bound_cost

    w_exec = float(runner.cfg.get("frontres_exec_reward_weight", 1.0))
    repair_scale = float(runner.cfg.get("frontres_repair_reward_scale", 1.0))
    w_geom = float(runner.cfg.get("frontres_geometry_reward_weight", 0.05))
    w_rescue = float(runner.cfg.get("frontres_rescue_reward_weight", 1.0))
    w_exec_harm = float(runner.cfg.get("frontres_executable_harm_weight", 1.0))
    rollout_evidence = compute_frontres_rollout_evidence(
        noisy_score=exec_perturbed,
        projected_score=exec_frontres,
        candidate_score=exec_candidate,
    )
    repair_gain = rollout_evidence.repair_gain
    candidate_gain = rollout_evidence.candidate_gain
    projection_gain = rollout_evidence.projection_gain
    oracle_ub = compute_frontres_oracle_upper_bound(
        exec_perturbed,
        exec_frontres,
        exec_candidate,
        exec_feasible,
        margin=float(runner.cfg.get("frontres_oracle_upper_bound_margin", 0.0)),
        enabled=bool(runner.cfg.get("frontres_oracle_upper_bound_diag_enabled", True)),
    )
    base_done_for_floor = dones[base_start:base_start + n_exec].view(-1) > 0
    timeout_for_floor = infos.get("time_outs", None)
    if timeout_for_floor is not None:
        timeout_for_floor = timeout_for_floor.to(device).view(-1)
        base_timeout_for_floor = timeout_for_floor[base_start:base_start + n_exec] > 0
    else:
        base_timeout_for_floor = torch.zeros(n_exec, device=device, dtype=torch.bool)
    mix_class_for_floor = getattr(runner, "_frontres_dr_mix_class_train", None)
    exec_floor, exec_safe_floor, exec_floor_source = update_runner_executable_floor_stats(
        runner,
        exec_perturbed,
        done=base_done_for_floor,
        timeout=base_timeout_for_floor,
        mix_class=mix_class_for_floor,
    )
    exec_floor_tensor = torch.full_like(exec_candidate, float(exec_floor))
    candidate_floor_margin = exec_candidate - exec_floor_tensor
    candidate_floor_pass = (candidate_floor_margin >= 0.0).to(candidate_floor_margin.dtype)
    candidate_floor_pass_frac = candidate_floor_pass.mean()
    stable_route_next = getattr(
        runner,
        "_frontres_stable_route_next_mask",
        torch.zeros_like(candidate_floor_margin, dtype=torch.bool),
    )
    stable_route_next = stable_route_next.to(device).view(-1).bool()
    if stable_route_next.numel() < n_exec:
        stable_route_next = torch.nn.functional.pad(
            stable_route_next,
            (0, n_exec - stable_route_next.numel()),
            value=False,
        )
    stable_route_next = stable_route_next[:n_exec]
    if n_exec > 0:
        alive_next = ~(dones[:n_exec].view(-1) > 0)
        stable_route_next = stable_route_next & alive_next
    runner._frontres_stable_route_next_mask = stable_route_next.detach()
    runner._frontres_candidate_floor_margin_last = float(candidate_floor_margin.mean().detach().item())
    runner._frontres_candidate_floor_pass_last = float(candidate_floor_pass_frac.detach().item())
    runner._frontres_stable_route_frac_last = float(
        stable_route_next.to(candidate_floor_margin.dtype).mean().detach().item()
    )
    stable_route_active = getattr(runner, "_frontres_stable_route_active_mask", None)
    if stable_route_active is None:
        stable_route_active = torch.zeros(n_exec, device=device, dtype=torch.bool)
    else:
        stable_route_active = stable_route_active.to(device).view(-1).bool()
        if stable_route_active.numel() < n_exec:
            stable_route_active = torch.nn.functional.pad(
                stable_route_active,
                (0, n_exec - stable_route_active.numel()),
                value=False,
            )
        stable_route_active = stable_route_active[:n_exec]
    reward_window = build_frontres_reward_window(
        runner=runner,
        cfg=runner.cfg,
        n_train=n_train,
        n_exec=n_exec,
        exec_clean=exec_clean,
        exec_perturbed=exec_perturbed,
        exec_feasible=exec_feasible,
        exec_frontres=exec_frontres,
        repair_gain=repair_gain,
        mode_groups=mode_groups,
        e_raw=e_raw,
        e_fr=e_fr,
        intervention_cost=intervention_cost,
        action_activity=action_activity,
        under_repair_penalty=under_repair_penalty,
        dr_scale=dr_scale,
        ppo_actor_weight_current=ppo_actor_weight_current,
        stable_route_active_mask=stable_route_active,
        device=device,
    )
    r_exec = reward_window.r_exec

    return FrontRESRewardContext(
        candidate_start=candidate_start,
        candidate_end=candidate_end,
        base_start=base_start,
        base_end=base_end,
        clean_start=clean_start,
        clean_end=clean_end,
        n_exec=n_exec,
        n_pair=n_pair,
        r_raw_gmt=r_raw_gmt,
        r_clean_gmt=r_clean_gmt,
        r_candidate_gmt=r_candidate_gmt,
        r_total=r_total,
        cmd=cmd,
        use_clean=use_clean,
        a_w=a_w,
        a_raw=a_raw,
        a_fr=a_fr,
        q_w=q_w,
        q_raw=q_raw,
        q_fr=q_fr,
        r_step=r_step,
        r_rescue=r_rescue,
        r_exec=r_exec,
        dr_z_abs_log=dr_z_abs_log,
        dr_xy_abs_log=dr_xy_abs_log,
        dr_rp_abs_log=dr_rp_abs_log,
        dr_yaw_abs_log=dr_yaw_abs_log,
        corr_z_abs_log=corr_z_abs_log,
        corr_xy_abs_log=corr_xy_abs_log,
        corr_rp_abs_log=corr_rp_abs_log,
        corr_yaw_abs_log=corr_yaw_abs_log,
        rot_raw_to_clean=rot_raw_to_clean,
        rot_raw_to_fr=rot_raw_to_fr,
        e_raw=e_raw,
        e_fr=e_fr,
        exec_score_all=exec_score_all,
        exec_components=exec_components,
        feasible_components=feasible_components,
        mode_groups=mode_groups,
        exec_frontres=exec_frontres,
        exec_candidate=exec_candidate,
        exec_perturbed=exec_perturbed,
        exec_clean=exec_clean,
        exec_feasible=exec_feasible,
        exec_planar_log=exec_planar_log,
        exec_vertical_log=exec_vertical_log,
        exec_task_log=exec_task_log,
        intervention_cost=intervention_cost,
        clean_bound_cost=clean_bound_cost,
        side_cost=side_cost,
        over_cost=over_cost,
        overcorrection_cost=overcorrection_cost,
        under_repair_penalty=reward_window.under_repair_penalty,
        action_activity=action_activity,
        w_exec=w_exec,
        repair_scale=repair_scale,
        w_geom=w_geom,
        w_rescue=w_rescue,
        w_exec_harm=w_exec_harm,
        repair_gain=repair_gain,
        candidate_gain=candidate_gain,
        projection_gain=projection_gain,
        oracle_ub_gain=oracle_ub.gain,
        oracle_ub_pass=oracle_ub.pass_mask,
        oracle_ub_noisy_win=oracle_ub.noisy_win,
        oracle_ub_projected_win=oracle_ub.projected_win,
        oracle_ub_candidate_win=oracle_ub.candidate_win,
        oracle_ub_feasible_win=oracle_ub.feasible_win,
        exec_floor=exec_floor,
        exec_safe_floor=exec_safe_floor,
        exec_floor_source=exec_floor_source,
        candidate_floor_margin=candidate_floor_margin,
        candidate_floor_pass=candidate_floor_pass,
        candidate_floor_pass_frac=candidate_floor_pass_frac,
        stable_route_next=stable_route_next,
        stable_route_active=stable_route_active,
        reward_window=reward_window,
    )


def build_frontres_reward_window(
    *,
    runner: Any,
    cfg: Mapping[str, Any],
    n_train: int,
    n_exec: int,
    exec_clean: torch.Tensor,
    exec_perturbed: torch.Tensor,
    exec_feasible: torch.Tensor,
    exec_frontres: torch.Tensor,
    repair_gain: torch.Tensor,
    mode_groups: Any,
    e_raw: torch.Tensor,
    e_fr: torch.Tensor,
    intervention_cost: torch.Tensor,
    action_activity: torch.Tensor,
    under_repair_penalty: torch.Tensor,
    dr_scale: float,
    ppo_actor_weight_current: float,
    stable_route_active_mask: torch.Tensor | None,
    device: torch.device,
) -> FrontRESRewardWindow:
    """Build reward gates, weights, harm terms, and actor-credit gates."""

    reward_dr_ref = float(
        cfg.get(
            "frontres_reward_scale_dr_reference",
            cfg.get("supervised_warmup_dr_scale", cfg.get("dr_scale_init", 1.0)),
        )
    )
    reward_dr_ref = max(reward_dr_ref, 1e-6)
    reward_dr_progress = max(0.0, min(1.0, float(dr_scale) / reward_dr_ref))
    reward_actor_progress = max(0.0, min(1.0, float(ppo_actor_weight_current)))
    reward_progress = reward_dr_progress * reward_actor_progress
    reward_progress = max(float(cfg.get("frontres_reward_progress_min", 0.0)), min(1.0, reward_progress))
    constraint_exp = float(cfg.get("frontres_constraint_progress_exponent", 2.0))
    constraint_exp = max(1.0, constraint_exp)
    constraint_progress = reward_progress ** constraint_exp

    gap_raw = exec_clean - exec_perturbed
    damage_gap = gap_raw.clamp(min=0.0)
    oracle_clean_gap = (exec_clean - exec_feasible).clamp(min=0.0)
    oracle_trust_tau = float(cfg.get("frontres_oracle_clean_gap_tau", 0.0))
    if oracle_trust_tau > 0.0:
        oracle_trust = torch.exp(-oracle_clean_gap / max(oracle_trust_tau, 1e-6))
    else:
        oracle_trust_threshold = float(cfg.get("frontres_oracle_clean_gap_threshold", 1e9))
        oracle_trust = (oracle_clean_gap <= oracle_trust_threshold).to(damage_gap.dtype)

    gap_floor = float(cfg.get("frontres_gap_floor_per_step", 0.005))
    repair_ratio = (repair_gain / damage_gap.clamp(min=gap_floor)).clamp(-1.0, 1.0)
    reward_signal_mode = str(cfg.get("frontres_exec_reward_signal", "gain")).lower()
    if reward_signal_mode in ("family_preference", "preference", "ranking"):
        gain_std = frontres_family_gain_std(runner, mode_groups, repair_gain.detach())
        tau = max(float(cfg.get("frontres_family_preference_tau", 1.0)), 1e-6)
        pref = torch.tanh((repair_gain / gain_std) / tau)
        alpha = float(cfg.get("frontres_family_preference_alpha", 0.7))
        alpha = max(0.0, min(1.0, alpha))
        scale = float(cfg.get("frontres_family_preference_scale", 0.02))
        exec_signal = scale * (alpha * pref + (1.0 - alpha) * repair_ratio)
    elif reward_signal_mode == "ratio":
        exec_signal = repair_ratio
    else:
        exec_signal = repair_gain
    r_exec = torch.zeros(n_train, device=device)
    r_exec[:n_exec] = exec_signal

    safe_gap = float(cfg.get("frontres_safe_gap_per_step", 0.003))
    broken_gap = float(cfg.get("frontres_broken_gap_per_step", 0.08))
    broken_gap = max(broken_gap, safe_gap + 1e-6)
    gate_temp = float(cfg.get("frontres_gap_gate_temp", 0.005))
    gate_temp = max(gate_temp, 1e-6)
    enter_window = torch.sigmoid((damage_gap - safe_gap) / gate_temp)
    exit_window = torch.sigmoid((broken_gap - damage_gap) / gate_temp)
    window_mu_raw = enter_window * exit_window
    gap_mid = 0.5 * (safe_gap + broken_gap)
    peak_enter_arg = max(-60.0, min(60.0, (gap_mid - safe_gap) / gate_temp))
    peak_exit_arg = max(-60.0, min(60.0, (broken_gap - gap_mid) / gate_temp))
    window_peak = (
        1.0 / (1.0 + math.exp(-peak_enter_arg))
        * 1.0 / (1.0 + math.exp(-peak_exit_arg))
    )
    window_mu = (window_mu_raw / max(window_peak, 1e-6)).clamp(0.0, 1.0)
    safe_gate = (1.0 - enter_window).clamp(0.0, 1.0)
    repair_gate = window_mu
    broken_gate = (1.0 - exit_window).clamp(0.0, 1.0)
    safe_frac = (damage_gap < safe_gap).float().mean()
    repair_frac = ((damage_gap >= safe_gap) & (damage_gap <= broken_gap)).float().mean()
    broken_frac = (damage_gap > broken_gap).float().mean()

    stable_route_active = stable_route_active_mask
    if stable_route_active is None:
        stable_route_active = torch.zeros(n_exec, device=device, dtype=torch.bool)
    else:
        stable_route_active = stable_route_active.to(device).view(-1).bool()
        if stable_route_active.numel() < n_exec:
            stable_route_active = torch.nn.functional.pad(
                stable_route_active,
                (0, n_exec - stable_route_active.numel()),
                value=False,
            )
        stable_route_active = stable_route_active[:n_exec]
    rho_space_for_route = str(cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
    if rho_space_for_route in ("tri_anchor", "tri-anchor", "tri"):
        learnable_route_mask = torch.ones_like(damage_gap)
    else:
        learnable_route_mask = (~stable_route_active).to(damage_gap.dtype)

    restore_min_ratio = float(cfg.get("frontres_min_restore_ratio", 0.0))
    restore_under_weight = float(cfg.get("frontres_under_repair_weight", 0.0))
    if restore_min_ratio > 0.0 and restore_under_weight > 0.0:
        restore_ratio = ((e_raw - e_fr) / e_raw.clamp(min=1e-6)).clamp(-1.0, 1.0)
        under = torch.relu(restore_min_ratio - restore_ratio)
        under_repair_penalty[:n_exec] = restore_under_weight * repair_gate * under[:n_exec].pow(2)

    selective_reward = bool(cfg.get("frontres_selective_reward_enabled", True))
    if selective_reward:
        exec_gate = repair_gate
        safe_cost_weight = float(cfg.get("frontres_safe_cost_weight", 1.0))
        repair_cost_weight = float(cfg.get("frontres_repair_cost_weight", 0.15))
        broken_cost_weight = float(cfg.get("frontres_broken_cost_weight", 1.0))
        cost_gate = (
            safe_cost_weight * safe_gate
            + repair_cost_weight * repair_gate
            + broken_cost_weight * broken_gate
        ).clamp(min=0.0)
    else:
        exec_gate = window_mu
        cost_gate = (1.0 - window_mu).clamp(0.0, 1.0)

    harm_eps = float(cfg.get("frontres_harm_epsilon", 0.001))
    harm_weight_cfg = float(cfg.get("frontres_harm_penalty_weight", 0.25))
    side_harm_weight = float(cfg.get("frontres_side_harm_weight", 0.0))
    side_harm_weight = max(0.0, min(1.0, side_harm_weight))
    harm_mag_raw = torch.relu(-repair_gain - max(harm_eps, 0.0))
    cost_exec = intervention_cost[:n_exec]
    harm_action_floor = float(cfg.get("frontres_harm_action_cost_floor", 0.001))
    harm_action_ref = float(cfg.get("frontres_harm_action_cost_ref", 0.01))
    harm_action_ref = max(harm_action_ref, harm_action_floor + 1e-6)
    harm_action_measure = action_activity[:n_exec]
    harm_action_gate = ((harm_action_measure - harm_action_floor) / (harm_action_ref - harm_action_floor)).clamp(0.0, 1.0)
    harm_mag = harm_mag_raw * harm_action_gate
    if selective_reward:
        broken_harm_weight = float(cfg.get("frontres_broken_harm_weight", 1.0))
        harm_weight = (
            repair_gate
            + broken_harm_weight * broken_gate
            + side_harm_weight * safe_gate
        ).clamp(0.0, 1.0)
    else:
        harm_weight = (window_mu + side_harm_weight * (1.0 - window_mu)).clamp(0.0, 1.0)
    harm_penalty_exec = harm_weight_cfg * harm_weight * harm_mag

    side_actor_weight = float(cfg.get("frontres_side_actor_gate_weight", 0.05))
    side_actor_weight = max(0.0, min(1.0, side_actor_weight))
    if selective_reward:
        actor_gate = (
            oracle_trust * repair_gate
            + side_actor_weight * (safe_gate + broken_gate)
        ).clamp(0.0, 1.0)
    else:
        actor_gate = (
            oracle_trust * window_mu + side_actor_weight * (1.0 - window_mu)
        ).clamp(0.0, 1.0)
    actor_gate = actor_gate * learnable_route_mask

    exec_weight = torch.zeros(n_train, device=device)
    cost_weight = torch.ones(n_train, device=device)
    exec_weight[:n_exec] = exec_gate
    cost_weight[:n_exec] = cost_gate
    harm_penalty = torch.zeros(n_train, device=device)
    harm_penalty[:n_exec] = harm_penalty_exec

    effective_gain_bonus_exec = torch.zeros(n_exec, device=device)
    if selective_reward:
        min_effective_gain = float(cfg.get("frontres_min_effective_gain", 0.006))
        bonus_weight = float(cfg.get("frontres_effective_gain_bonus_weight", 0.5))
        effective_gain_bonus_exec = bonus_weight * repair_gate * torch.relu(repair_gain - min_effective_gain)
    effective_gain_bonus = torch.zeros(n_train, device=device)
    effective_gain_bonus[:n_exec] = effective_gain_bonus_exec

    return FrontRESRewardWindow(
        r_exec=r_exec,
        damage_gap=damage_gap,
        oracle_clean_gap=oracle_clean_gap,
        oracle_trust=oracle_trust,
        repair_ratio=repair_ratio,
        safe_gate=safe_gate,
        repair_gate=repair_gate,
        broken_gate=broken_gate,
        window_mu=window_mu,
        exec_gate=exec_gate,
        cost_gate=cost_gate,
        safe_frac=safe_frac,
        repair_frac=repair_frac,
        broken_frac=broken_frac,
        safe_gap=safe_gap,
        broken_gap=broken_gap,
        learnable_route_mask=learnable_route_mask,
        exec_weight=exec_weight,
        cost_weight=cost_weight,
        actor_gate=actor_gate,
        harm_penalty=harm_penalty,
        harm_penalty_exec=harm_penalty_exec,
        harm_mag=harm_mag,
        cost_exec=cost_exec,
        effective_gain_bonus=effective_gain_bonus,
        effective_gain_bonus_exec=effective_gain_bonus_exec,
        under_repair_penalty=under_repair_penalty,
        r_delta=None,
        under_write=None,
        ranking_reward=None,
        reward_progress=reward_progress,
        constraint_progress=constraint_progress,
    )


def compose_frontres_reward_delta(
    *,
    cfg: Mapping[str, Any],
    reward_window: FrontRESRewardWindow,
    n_train: int,
    n_exec: int,
    n_candidate: int,
    repair_gain: torch.Tensor,
    candidate_gain: torch.Tensor,
    projection_gain: torch.Tensor,
    r_step: torch.Tensor,
    r_rescue: torch.Tensor,
    intervention_cost: torch.Tensor,
    overcorrection_cost: torch.Tensor,
    w_exec: float,
    repair_scale: float,
    w_geom: float,
    w_rescue: float,
    w_exec_harm: float,
    device: torch.device,
) -> FrontRESRewardWindow:
    """Compose ranking reward and final FrontRES delta reward."""

    ranking_enabled = bool(cfg.get("frontres_candidate_ranking_reward_enabled", True)) and n_candidate > 0
    if ranking_enabled:
        ranking_under_weight = float(cfg.get("frontres_candidate_underwrite_weight", 1.0))
        ranking_projection_weight = float(cfg.get("frontres_candidate_projection_weight", 0.25))
        ranking_harm_weight = float(cfg.get("frontres_candidate_harm_weight", 1.0))
        under_write = torch.relu(candidate_gain - repair_gain) * (candidate_gain > 0.0).to(repair_gain.dtype)
        ranking_exec = (
            repair_gain
            + ranking_projection_weight * projection_gain
            - ranking_under_weight * under_write
            - ranking_harm_weight * torch.relu(-repair_gain)
        )
        ranking_reward = torch.zeros(n_train, device=device)
        ranking_reward[:n_exec] = reward_window.exec_gate * ranking_exec
    else:
        under_write = torch.zeros_like(repair_gain)
        ranking_reward = torch.zeros(n_train, device=device)

    positive_reward = (
        float(w_exec) * float(repair_scale) * reward_window.exec_weight * reward_window.r_exec
        + float(w_exec) * float(repair_scale) * reward_window.effective_gain_bonus
        + float(w_geom) * r_step
        + float(cfg.get("frontres_candidate_ranking_reward_weight", 1.0)) * ranking_reward
        + float(w_rescue) * r_rescue
    )
    constraint_penalty = (
        float(w_exec_harm) * reward_window.harm_penalty
        + reward_window.cost_weight * intervention_cost
        + overcorrection_cost
        + reward_window.under_repair_penalty
    )
    r_delta = (
        float(reward_window.reward_progress) * positive_reward
        - float(reward_window.constraint_progress) * constraint_penalty
    )

    reward_window.r_delta = r_delta
    reward_window.under_write = under_write
    reward_window.ranking_reward = ranking_reward
    return reward_window
