# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rsl_rl.frontres.frontres_alpha_router import build_state_alpha_targets
from rsl_rl.frontres.frontres_reward_window import FrontRESRewardContext
from rsl_rl.frontres.frontres_structured_rho import build_structured_rho_carrier


@dataclass
class FrontRESAcceptancePayload:
    """Rollout-built acceptance/rho tensors and diagnostics for storage/logging."""

    accept_target: torch.Tensor
    accept_mask: torch.Tensor
    pref_full_frac: torch.Tensor
    pref_noop_frac: torch.Tensor
    pref_keep_frac: torch.Tensor
    pref_ignore_frac: torch.Tensor
    pref_margin_mean: torch.Tensor
    pref_need_mean: torch.Tensor
    pref_admiss_mean: torch.Tensor
    pref_target_mean: torch.Tensor
    tri_weight_repair_mean: torch.Tensor
    tri_weight_noisy_mean: torch.Tensor
    tri_weight_stable_mean: torch.Tensor
    pref_inertial_penalty_rho_mean: torch.Tensor
    pref_inertial_penalty_one_mean: torch.Tensor
    rho_target_planar_mean: torch.Tensor
    rho_target_rp_mean: torch.Tensor
    rho_target_z_mean: torch.Tensor
    rho_target_spread_mean: torch.Tensor
    grouped_rho_mask_mean: torch.Tensor
    rho_regret_up_planar_mean: torch.Tensor
    rho_regret_up_rp_mean: torch.Tensor
    rho_regret_up_z_mean: torch.Tensor
    rho_regret_down_planar_mean: torch.Tensor
    rho_regret_down_rp_mean: torch.Tensor
    rho_regret_down_z_mean: torch.Tensor


@dataclass
class FrontRESTriAnchorRhoPayload:
    """Tri-anchor rho targets plus the side tensors used by structured rho."""

    target_exec: torch.Tensor
    mask_exec: torch.Tensor
    rho_current: torch.Tensor
    rho_space: str
    grouped_targets_enabled: bool
    rho_direction_dim_from_regret: torch.Tensor | None
    candidate_planar: torch.Tensor | None
    candidate_rp: torch.Tensor | None
    candidate_z: torch.Tensor | None
    projected_planar: torch.Tensor | None
    projected_rp: torch.Tensor | None
    projected_z: torch.Tensor | None
    base_planar: torch.Tensor | None
    base_rp: torch.Tensor | None
    base_z: torch.Tensor | None
    rho_target_planar_mean: torch.Tensor
    rho_target_rp_mean: torch.Tensor
    rho_target_z_mean: torch.Tensor
    rho_target_spread_mean: torch.Tensor
    rho_regret_up_planar_mean: torch.Tensor
    rho_regret_up_rp_mean: torch.Tensor
    rho_regret_up_z_mean: torch.Tensor
    rho_regret_down_planar_mean: torch.Tensor
    rho_regret_down_rp_mean: torch.Tensor
    rho_regret_down_z_mean: torch.Tensor


@dataclass
class FrontRESNonTriAcceptanceTargetPayload:
    """Non-tri-anchor acceptance target tensors plus direct-target diagnostics."""

    target_exec: torch.Tensor
    mask_exec: torch.Tensor
    need: torch.Tensor | None
    admissibility: torch.Tensor | None


@dataclass
class FrontRESStructuredRhoPayload:
    """Structured-rho carrier tensors after optional override."""

    accept_target: torch.Tensor
    accept_mask: torch.Tensor
    target_exec: torch.Tensor
    mask_exec: torch.Tensor
    enabled: bool


def frontres_rho_current_from_actions(
    actions: torch.Tensor,
    *,
    n_exec: int,
    task_conf_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract the live rho coefficients from sampled task-space actions."""

    rho_current = torch.ones(n_exec, 6, device=device)
    if actions.shape[-1] >= 12 and task_conf_dim == 6:
        rho_current = actions[:n_exec, 6:12].detach().clamp(0.0, 1.0)
    elif actions.shape[-1] >= 7 and task_conf_dim == 1:
        rho_current = actions[:n_exec, 6:7].detach().clamp(0.0, 1.0).expand(-1, 6)
    return rho_current


def build_frontres_non_tri_acceptance_target_payload(
    *,
    cfg: Mapping[str, Any],
    rho_space: str,
    target_exec: torch.Tensor,
    mask_exec: torch.Tensor,
    n_exec: int,
    base_start: int,
    candidate_start: int,
    a_w: torch.Tensor,
    a_raw: torch.Tensor,
    a_fr: torch.Tensor,
    q_w: torch.Tensor,
    q_raw: torch.Tensor,
    q_fr: torch.Tensor,
    c_zero: torch.Tensor | None,
    c_one: torch.Tensor | None,
    rho_current: torch.Tensor,
    j_one: torch.Tensor,
    j_zero: torch.Tensor,
    j_rho: torch.Tensor,
    full_win: torch.Tensor,
    noop_win: torch.Tensor,
    keep_win: torch.Tensor,
    pref_margin: float,
    pref_gate: torch.Tensor,
    quat_to_rotvec_wxyz: Any,
    quat_mul_fn: Any,
    quat_inv_fn: Any,
    device: torch.device,
) -> FrontRESNonTriAcceptanceTargetPayload:
    """Build direct or legacy acceptance targets for non-tri-anchor rho spaces."""

    if rho_space in ("tri_anchor", "tri-anchor", "tri"):
        return FrontRESNonTriAcceptanceTargetPayload(
            target_exec=target_exec,
            mask_exec=mask_exec,
            need=None,
            admissibility=None,
        )

    if bool(cfg.get("frontres_acceptance_direct_target_enabled", False)):
        clean_pos = a_w[base_start:base_start + n_exec].detach()
        noisy_pos = a_raw[base_start:base_start + n_exec].detach()
        candidate_clean_pos = a_w[candidate_start:candidate_start + n_exec].detach()
        candidate_pos = a_fr[candidate_start:candidate_start + n_exec].detach()
        clean_q = q_w[base_start:base_start + n_exec].detach()
        noisy_q = q_raw[base_start:base_start + n_exec].detach()
        candidate_clean_q = q_w[candidate_start:candidate_start + n_exec].detach()
        candidate_q = q_fr[candidate_start:candidate_start + n_exec].detach()
        err0_vec = torch.cat(
            [
                (clean_pos - noisy_pos),
                quat_to_rotvec_wxyz(quat_mul_fn(quat_inv_fn(noisy_q), clean_q)),
            ],
            dim=-1,
        )
        err1_vec = torch.cat(
            [
                (candidate_clean_pos - candidate_pos),
                quat_to_rotvec_wxyz(quat_mul_fn(quat_inv_fn(candidate_q), candidate_clean_q)),
            ],
            dim=-1,
        )
        err0 = err0_vec.abs()
        err1 = err1_vec.abs()
        need = ((err0 - err1) / err0.clamp(min=1e-6)).clamp(0.0, 1.0)
        need_min = max(0.0, float(cfg.get("frontres_acceptance_need_min_error", 0.01)))
        need_mask = (err0 > need_min).to(need.dtype)
        admissibility = torch.ones(n_exec, device=device, dtype=need.dtype)
        if c_zero is not None and c_one is not None:
            inertial_margin = float(cfg.get("frontres_inertial_preference_margin", 0.05))
            temp = max(1e-6, float(cfg.get("frontres_acceptance_admissibility_temp", 0.20)))
            admissibility = torch.sigmoid((c_one - c_zero - inertial_margin) / temp)
        target_exec = torch.minimum(need, admissibility.view(-1, 1)).clamp(0.0, 1.0)
        mask_exec = need_mask * pref_gate.view(-1, 1)
        return FrontRESNonTriAcceptanceTargetPayload(
            target_exec=target_exec,
            mask_exec=mask_exec,
            need=need,
            admissibility=admissibility,
        )

    calib_step = float(cfg.get("frontres_acceptance_calibration_step", 0.5))
    calib_step = max(0.0, min(1.0, calib_step))
    denom = (j_one - j_zero).abs().clamp(min=1e-6)
    target_exec = rho_current.clone()
    raise_target = ((j_one - j_rho) / denom).clamp(0.0, 1.0)
    lower_target = ((j_zero - j_rho) / denom).clamp(0.0, 1.0)
    if bool(full_win.any().detach().item()):
        target_exec[full_win] = (
            rho_current[full_win] + calib_step * raise_target[full_win].view(-1, 1)
        ).clamp(0.0, 1.0)
    if bool(noop_win.any().detach().item()):
        target_exec[noop_win] = (
            rho_current[noop_win] - calib_step * lower_target[noop_win].view(-1, 1)
        ).clamp(0.0, 1.0)
    trainable = (full_win | noop_win | keep_win).to(target_exec.dtype)
    mask_exec = trainable.view(-1, 1) * pref_gate.view(-1, 1)
    return FrontRESNonTriAcceptanceTargetPayload(
        target_exec=target_exec,
        mask_exec=mask_exec,
        need=None,
        admissibility=None,
    )


def apply_frontres_structured_rho_payload(
    runner: Any,
    *,
    accept_target: torch.Tensor,
    accept_mask: torch.Tensor,
    target_exec: torch.Tensor,
    mask_exec: torch.Tensor,
    n_exec: int,
    rho_current: torch.Tensor,
    rho_update_weight: torch.Tensor,
    exec_perturbed: torch.Tensor,
    exec_feasible: torch.Tensor,
    exec_frontres: torch.Tensor,
    exec_candidate: torch.Tensor,
    state_alpha_target: torch.Tensor,
    rho_space: str,
    grouped_targets_enabled: bool,
    feasible_components: Mapping[str, torch.Tensor] | None,
    candidate_planar: torch.Tensor | None,
    candidate_rp: torch.Tensor | None,
    candidate_z: torch.Tensor | None,
    projected_planar: torch.Tensor | None,
    projected_rp: torch.Tensor | None,
    projected_z: torch.Tensor | None,
    base_planar: torch.Tensor | None,
    base_rp: torch.Tensor | None,
    base_z: torch.Tensor | None,
    pref_margin: float,
) -> FrontRESStructuredRhoPayload:
    """Optionally apply structured rho and write runner diagnostics."""

    structured_joint_enabled = runner._frontres_structured_joint_effective_enabled()
    if structured_joint_enabled:
        rho_floor = float(
            getattr(
                runner,
                "_frontres_exec_floor_value_last",
                runner.cfg.get(
                    "frontres_structured_joint_exec_floor",
                    runner.cfg.get("frontres_state_alpha_exec_floor", 0.0),
                ),
            )
        )
        rho_carrier = build_structured_rho_carrier(
            num_envs=runner.env.num_envs,
            n_exec=n_exec,
            rho_current=rho_current,
            rho_dim_weight=mask_exec.detach().clamp(min=0.0),
            rho_update_weight=rho_update_weight[:n_exec],
            exec_perturbed=exec_perturbed,
            exec_feasible=exec_feasible,
            exec_frontres=exec_frontres,
            exec_candidate=exec_candidate,
            state_alpha_target=state_alpha_target,
            live_alpha=getattr(runner, "_frontres_state_alpha_prob_next", None),
            rho_space=rho_space,
            grouped_targets_enabled=grouped_targets_enabled,
            feasible_components=feasible_components,
            candidate_planar=candidate_planar,
            candidate_rp=candidate_rp,
            candidate_z=candidate_z,
            projected_planar=projected_planar,
            projected_rp=projected_rp,
            projected_z=projected_z,
            base_planar=base_planar,
            base_rp=base_rp,
            base_z=base_z,
            pref_margin=pref_margin,
            rho_floor=rho_floor,
            directional_weight=max(
                0.0,
                float(runner.cfg.get("frontres_structured_joint_directional_weight", 1.0)),
            ),
            rho_center=float(runner.cfg.get("frontres_structured_joint_rho_center", 0.5)),
            center_drive_deadzone=max(
                0.0,
                float(runner.cfg.get("frontres_structured_joint_center_drive_deadzone", 0.10)),
            ),
            retention_weight=max(
                0.0,
                float(runner.cfg.get("frontres_structured_joint_retention_prior_weight", 0.0)),
            ),
            floor_penalty_weight=max(
                0.0,
                float(runner.cfg.get("frontres_structured_joint_floor_penalty_weight", 5.0)),
            ),
            full_bonus_weight=max(
                0.0,
                float(runner.cfg.get("frontres_structured_joint_full_repair_bonus_weight", 1.0)),
            ),
            joint_weight_floor=max(
                0.0,
                min(1.0, float(runner.cfg.get("frontres_structured_joint_weight_floor", 0.10))),
            ),
            use_rho_update_weight=bool(
                runner.cfg.get(
                    "frontres_structured_joint_use_sample_weight",
                    runner.cfg.get("frontres_structured_joint_use_actor_gate_weight", False),
                )
            ),
            device=runner.device,
        )
        # Legacy storage fields still use acceptance_* names.  On the live
        # structured-rho path they carry rho PPO advantages and weights.
        accept_target = rho_carrier.rho_advantage
        accept_mask = rho_carrier.rho_validity_weight
        target_exec = rho_carrier.rho_advantage_exec
        mask_exec = rho_carrier.rho_validity_weight_exec
        runner._frontres_structured_joint_adv_last = rho_carrier.adv_mean
        runner._frontres_structured_joint_weight_last = rho_carrier.weight_mean
        runner._frontres_structured_joint_rho_adv_last = rho_carrier.adv_mean
        runner._frontres_structured_joint_rho_weight_last = rho_carrier.weight_mean
        runner._frontres_structured_joint_rho_retention_last = rho_carrier.retention_mean
        runner._frontres_structured_joint_floor_violation_last = rho_carrier.floor_mean
        runner._frontres_structured_joint_full_bonus_last = rho_carrier.full_bonus_mean
        runner._frontres_structured_joint_rho_direction_last = rho_carrier.direction_mean
        runner._frontres_structured_joint_rho_centered_last = rho_carrier.centered_mean
        runner._frontres_structured_joint_rho_drive_last = rho_carrier.drive_mean
    else:
        runner._frontres_structured_joint_adv_last = 0.0
        runner._frontres_structured_joint_weight_last = 0.0
        runner._frontres_structured_joint_rho_adv_last = 0.0
        runner._frontres_structured_joint_rho_weight_last = 0.0
        runner._frontres_structured_joint_rho_retention_last = 0.0
        runner._frontres_structured_joint_floor_violation_last = 0.0
        runner._frontres_structured_joint_full_bonus_last = 0.0
        runner._frontres_structured_joint_rho_direction_last = 0.0
        runner._frontres_structured_joint_rho_centered_last = 0.0
        runner._frontres_structured_joint_rho_drive_last = 0.0
    return FrontRESStructuredRhoPayload(
        accept_target=accept_target,
        accept_mask=accept_mask,
        target_exec=target_exec,
        mask_exec=mask_exec,
        enabled=structured_joint_enabled,
    )


def build_frontres_tri_anchor_rho_payload(
    *,
    cfg: Mapping[str, Any],
    actions: torch.Tensor,
    n_exec: int,
    task_conf_dim: int,
    j_one: torch.Tensor,
    j_zero: torch.Tensor,
    pref_margin: float,
    pref_gate: torch.Tensor,
    exec_components: Mapping[str, torch.Tensor],
    candidate_start: int,
    base_start: int,
    regret_target_enabled: bool,
    device: torch.device,
) -> FrontRESTriAnchorRhoPayload:
    """Build tri-anchor rho targets from rollout counterfactual scores."""

    rho_current = frontres_rho_current_from_actions(
        actions,
        n_exec=n_exec,
        task_conf_dim=task_conf_dim,
        device=device,
    )
    rho_space = str(cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
    zero = torch.tensor(0.0, device=device)
    target_exec = torch.zeros(n_exec, 6, device=device)
    mask_exec = torch.zeros(n_exec, 6, device=device)
    grouped_targets_enabled = False
    rho_direction_dim_from_regret = None
    candidate_planar = candidate_rp = candidate_z = None
    projected_planar = projected_rp = projected_z = None
    base_planar = base_rp = base_z = None
    rho_target_planar_mean = zero
    rho_target_rp_mean = zero
    rho_target_z_mean = zero
    rho_target_spread_mean = zero
    rho_regret_up_planar_mean = zero
    rho_regret_up_rp_mean = zero
    rho_regret_up_z_mean = zero
    rho_regret_down_planar_mean = zero
    rho_regret_down_rp_mean = zero
    rho_regret_down_z_mean = zero

    if rho_space in ("tri_anchor", "tri-anchor", "tri"):
        rho_temp = max(1e-6, float(cfg.get("frontres_acceptance_rho_target_temp", 0.08)))
        calib_step = float(cfg.get("frontres_acceptance_calibration_step", 0.5))
        calib_step = max(0.0, min(1.0, calib_step))
        target_scalar = torch.sigmoid((j_one - j_zero - pref_margin) / rho_temp).clamp(0.0, 1.0)
        grouped_targets_enabled = bool(cfg.get("frontres_grouped_rho_target_enabled", True))
        if grouped_targets_enabled:
            comp = exec_components

            def component_branch(name: str, start: int, fallback: torch.Tensor) -> torch.Tensor:
                component = comp.get(name, fallback)
                return component[start:start + n_exec].to(device)

            def component_gain(name: str, fallback: torch.Tensor) -> torch.Tensor:
                return component_branch(name, candidate_start, fallback) - component_branch(name, base_start, fallback)

            def group_score(start: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                xy = component_branch("xy", start, comp["planar"])
                yaw = component_branch("yaw", start, comp["planar"])
                rp = component_branch("rp", start, comp["vertical"])
                z = component_branch("z", start, comp["vertical"])
                planar = 0.5 * (xy + yaw)
                return planar, rp, z

            if regret_target_enabled:
                candidate_planar, candidate_rp, candidate_z = group_score(candidate_start)
                projected_planar, projected_rp, projected_z = group_score(0)
                base_planar, base_rp, base_z = group_score(base_start)

                def regret_target(
                    candidate_score: torch.Tensor,
                    projected_score: torch.Tensor,
                    base_score: torch.Tensor,
                    rho_group: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    under = torch.relu(candidate_score - projected_score - pref_margin)
                    over = torch.relu(base_score - projected_score - pref_margin)
                    score_range = (candidate_score - base_score).abs().clamp(min=1e-6)
                    delta = calib_step * ((under - over) / score_range).clamp(-1.0, 1.0)
                    return (rho_group + delta).clamp(0.0, 1.0), under, over

                def regret_direction(
                    candidate_score: torch.Tensor,
                    base_score: torch.Tensor,
                    under: torch.Tensor,
                    over: torch.Tensor,
                ) -> torch.Tensor:
                    score_range = (candidate_score - base_score).abs() + pref_margin + 1.0e-6
                    return ((under - over) / score_range).clamp(-1.0, 1.0)

                rho_planar_current = (0.5 * (rho_current[:, 0] + rho_current[:, 1]) + rho_current[:, 5]) / 2.0
                rho_rp_current = 0.5 * (rho_current[:, 3] + rho_current[:, 4])
                rho_z_current = rho_current[:, 2]
                target_planar, regret_up_planar, regret_down_planar = regret_target(
                    candidate_planar, projected_planar, base_planar, rho_planar_current
                )
                target_rp, regret_up_rp, regret_down_rp = regret_target(
                    candidate_rp, projected_rp, base_rp, rho_rp_current
                )
                target_z, regret_up_z, regret_down_z = regret_target(
                    candidate_z, projected_z, base_z, rho_z_current
                )
                direction_planar = regret_direction(candidate_planar, base_planar, regret_up_planar, regret_down_planar)
                direction_rp = regret_direction(candidate_rp, base_rp, regret_up_rp, regret_down_rp)
                direction_z = regret_direction(candidate_z, base_z, regret_up_z, regret_down_z)
            else:
                xy_gain = component_gain("xy", comp["planar"])
                yaw_gain = component_gain("yaw", comp["planar"])
                rp_gain = component_gain("rp", comp["vertical"])
                z_gain = component_gain("z", comp["vertical"])
                planar_gain = 0.5 * (xy_gain + yaw_gain)
                target_planar = torch.sigmoid((planar_gain - pref_margin) / rho_temp).clamp(0.0, 1.0)
                target_rp = torch.sigmoid((rp_gain - pref_margin) / rho_temp).clamp(0.0, 1.0)
                target_z = torch.sigmoid((z_gain - pref_margin) / rho_temp).clamp(0.0, 1.0)
                regret_up_planar = torch.relu(planar_gain - pref_margin)
                regret_up_rp = torch.relu(rp_gain - pref_margin)
                regret_up_z = torch.relu(z_gain - pref_margin)
                regret_down_planar = torch.zeros_like(regret_up_planar)
                regret_down_rp = torch.zeros_like(regret_up_rp)
                regret_down_z = torch.zeros_like(regret_up_z)
                direction_planar = (planar_gain / (planar_gain.abs() + pref_margin + 1.0e-6)).clamp(-1.0, 1.0)
                direction_rp = (rp_gain / (rp_gain.abs() + pref_margin + 1.0e-6)).clamp(-1.0, 1.0)
                direction_z = (z_gain / (z_gain.abs() + pref_margin + 1.0e-6)).clamp(-1.0, 1.0)
            target_exec = torch.stack(
                [target_planar, target_planar, target_z, target_rp, target_rp, target_planar],
                dim=-1,
            )
            rho_direction_dim_from_regret = torch.stack(
                [direction_planar, direction_planar, direction_z, direction_rp, direction_rp, direction_planar],
                dim=-1,
            ).detach()
            mask_exec = pref_gate.view(-1, 1).expand(-1, 6).clone()
            rho_target_planar_mean = ((0.5 * (target_exec[:, 0] + target_exec[:, 1]) + target_exec[:, 5]) / 2.0).mean()
            rho_target_rp_mean = (0.5 * (target_exec[:, 3] + target_exec[:, 4])).mean()
            rho_target_z_mean = target_exec[:, 2].mean()
            rho_target_spread_mean = target_exec.std(dim=-1, unbiased=False).mean()
            rho_regret_up_planar_mean = regret_up_planar.mean()
            rho_regret_up_rp_mean = regret_up_rp.mean()
            rho_regret_up_z_mean = regret_up_z.mean()
            rho_regret_down_planar_mean = regret_down_planar.mean()
            rho_regret_down_rp_mean = regret_down_rp.mean()
            rho_regret_down_z_mean = regret_down_z.mean()
        else:
            target_exec = target_scalar.view(-1, 1).expand(-1, 6).clone()
            mask_exec = pref_gate.view(-1, 1).expand(-1, 6).clone()
            rho_target_planar_mean = target_scalar.mean()
            rho_target_rp_mean = target_scalar.mean()
            rho_target_z_mean = target_scalar.mean()
            rho_target_spread_mean = torch.zeros_like(target_scalar).mean()

    return FrontRESTriAnchorRhoPayload(
        target_exec=target_exec,
        mask_exec=mask_exec,
        rho_current=rho_current,
        rho_space=rho_space,
        grouped_targets_enabled=grouped_targets_enabled,
        rho_direction_dim_from_regret=rho_direction_dim_from_regret,
        candidate_planar=candidate_planar,
        candidate_rp=candidate_rp,
        candidate_z=candidate_z,
        projected_planar=projected_planar,
        projected_rp=projected_rp,
        projected_z=projected_z,
        base_planar=base_planar,
        base_rp=base_rp,
        base_z=base_z,
        rho_target_planar_mean=rho_target_planar_mean,
        rho_target_rp_mean=rho_target_rp_mean,
        rho_target_z_mean=rho_target_z_mean,
        rho_target_spread_mean=rho_target_spread_mean,
        rho_regret_up_planar_mean=rho_regret_up_planar_mean,
        rho_regret_up_rp_mean=rho_regret_up_rp_mean,
        rho_regret_up_z_mean=rho_regret_up_z_mean,
        rho_regret_down_planar_mean=rho_regret_down_planar_mean,
        rho_regret_down_rp_mean=rho_regret_down_rp_mean,
        rho_regret_down_z_mean=rho_regret_down_z_mean,
    )


def initialize_frontres_acceptance_payload(runner: Any) -> FrontRESAcceptancePayload:
    """Create default acceptance/rho payload tensors and reset last-step diagnostics."""

    zero = torch.tensor(0.0, device=runner.device)
    runner._frontres_rho_target_planar_last = 0.0
    runner._frontres_rho_target_rp_last = 0.0
    runner._frontres_rho_target_z_last = 0.0
    runner._frontres_rho_target_spread_last = 0.0
    runner._frontres_grouped_rho_mask_last = 0.0
    runner._frontres_rho_regret_up_planar_last = 0.0
    runner._frontres_rho_regret_up_rp_last = 0.0
    runner._frontres_rho_regret_up_z_last = 0.0
    runner._frontres_rho_regret_down_planar_last = 0.0
    runner._frontres_rho_regret_down_rp_last = 0.0
    runner._frontres_rho_regret_down_z_last = 0.0
    runner._frontres_structured_joint_adv_last = 0.0
    runner._frontres_structured_joint_weight_last = 0.0
    runner._frontres_structured_joint_rho_adv_last = 0.0
    runner._frontres_structured_joint_rho_weight_last = 0.0
    runner._frontres_structured_joint_rho_retention_last = 0.0
    runner._frontres_structured_joint_floor_violation_last = 0.0
    runner._frontres_structured_joint_full_bonus_last = 0.0
    runner._frontres_structured_joint_rho_direction_last = 0.0
    runner._frontres_structured_joint_rho_centered_last = 0.0
    runner._frontres_structured_joint_rho_drive_last = 0.0
    return FrontRESAcceptancePayload(
        accept_target=torch.zeros(runner.env.num_envs, 6, device=runner.device),
        accept_mask=torch.zeros(runner.env.num_envs, 6, device=runner.device),
        pref_full_frac=zero,
        pref_noop_frac=zero,
        pref_keep_frac=zero,
        pref_ignore_frac=torch.tensor(1.0, device=runner.device),
        pref_margin_mean=zero,
        pref_need_mean=zero,
        pref_admiss_mean=zero,
        pref_target_mean=zero,
        tri_weight_repair_mean=zero,
        tri_weight_noisy_mean=zero,
        tri_weight_stable_mean=zero,
        pref_inertial_penalty_rho_mean=zero,
        pref_inertial_penalty_one_mean=zero,
        rho_target_planar_mean=zero,
        rho_target_rp_mean=zero,
        rho_target_z_mean=zero,
        rho_target_spread_mean=zero,
        grouped_rho_mask_mean=zero,
        rho_regret_up_planar_mean=zero,
        rho_regret_up_rp_mean=zero,
        rho_regret_up_z_mean=zero,
        rho_regret_down_planar_mean=zero,
        rho_regret_down_rp_mean=zero,
        rho_regret_down_z_mean=zero,
    )


def summarize_frontres_acceptance_payload(
    runner: Any,
    *,
    accept_target: torch.Tensor,
    accept_mask: torch.Tensor,
    target_exec: torch.Tensor,
    mask_exec: torch.Tensor,
    structured_joint_enabled: bool,
    pref_margin: float,
    need: torch.Tensor | None,
    admissibility: torch.Tensor | None,
    j_one: torch.Tensor,
    j_rho: torch.Tensor,
    j_zero: torch.Tensor,
    tri_weight_repair_mean: torch.Tensor,
    tri_weight_noisy_mean: torch.Tensor,
    tri_weight_stable_mean: torch.Tensor,
    pref_inertial_penalty_rho_mean: torch.Tensor,
    pref_inertial_penalty_one_mean: torch.Tensor,
    rho_target_planar_mean: torch.Tensor,
    rho_target_rp_mean: torch.Tensor,
    rho_target_z_mean: torch.Tensor,
    rho_target_spread_mean: torch.Tensor,
    grouped_rho_mask_mean: torch.Tensor,
    rho_regret_up_planar_mean: torch.Tensor,
    rho_regret_up_rp_mean: torch.Tensor,
    rho_regret_up_z_mean: torch.Tensor,
    rho_regret_down_planar_mean: torch.Tensor,
    rho_regret_down_rp_mean: torch.Tensor,
    rho_regret_down_z_mean: torch.Tensor,
) -> FrontRESAcceptancePayload:
    """Summarize finalized acceptance targets for logging diagnostics."""

    zero = torch.tensor(0.0, device=runner.device)
    pref_full_frac = zero
    pref_noop_frac = zero
    pref_keep_frac = zero
    pref_ignore_frac = torch.tensor(1.0, device=runner.device)
    pref_margin_mean = zero
    pref_need_mean = zero
    pref_admiss_mean = zero
    pref_target_mean = zero

    active_pref = mask_exec.sum(dim=-1) > 0
    if bool(active_pref.any().detach().item()):
        mask_sum = mask_exec.sum(dim=-1).clamp(min=1e-6)
        target_sample = (target_exec * mask_exec).sum(dim=-1) / mask_sum
        active_target = target_sample[active_pref]
        if structured_joint_enabled:
            near = max(float(pref_margin), 1.0e-6)
            pref_full_frac = (active_target > near).float().mean()
            pref_noop_frac = (active_target < -near).float().mean()
            pref_keep_frac = (active_target.abs() <= near).float().mean()
        else:
            pref_full_frac = (active_target > 0.66).float().mean()
            pref_noop_frac = (active_target < 0.33).float().mean()
            pref_keep_frac = (
                ((active_target >= 0.33) & (active_target <= 0.66)).float().mean()
            )
        pref_target_mean = active_target.mean()
        if need is not None and admissibility is not None:
            need_sample = (need * mask_exec).sum(dim=-1) / mask_sum
            pref_need_mean = need_sample[active_pref].mean()
            admiss_sample = (admissibility.view(-1, 1) * mask_exec).sum(dim=-1) / mask_sum
            pref_admiss_mean = admiss_sample[active_pref].mean()
    pref_ignore_frac = (~active_pref).float().mean()

    best = torch.maximum(torch.maximum(j_one, j_rho), j_zero)
    second = torch.minimum(
        torch.maximum(j_one, j_rho),
        torch.maximum(torch.minimum(j_one, j_rho), j_zero),
    )
    pref_margin_mean = (best - second).mean()

    runner._frontres_rho_target_planar_last = float(rho_target_planar_mean.detach().item())
    runner._frontres_rho_target_rp_last = float(rho_target_rp_mean.detach().item())
    runner._frontres_rho_target_z_last = float(rho_target_z_mean.detach().item())
    runner._frontres_rho_target_spread_last = float(rho_target_spread_mean.detach().item())
    runner._frontres_grouped_rho_mask_last = float(grouped_rho_mask_mean.detach().item())
    runner._frontres_rho_regret_up_planar_last = float(rho_regret_up_planar_mean.detach().item())
    runner._frontres_rho_regret_up_rp_last = float(rho_regret_up_rp_mean.detach().item())
    runner._frontres_rho_regret_up_z_last = float(rho_regret_up_z_mean.detach().item())
    runner._frontres_rho_regret_down_planar_last = float(rho_regret_down_planar_mean.detach().item())
    runner._frontres_rho_regret_down_rp_last = float(rho_regret_down_rp_mean.detach().item())
    runner._frontres_rho_regret_down_z_last = float(rho_regret_down_z_mean.detach().item())

    return FrontRESAcceptancePayload(
        accept_target=accept_target,
        accept_mask=accept_mask,
        pref_full_frac=pref_full_frac,
        pref_noop_frac=pref_noop_frac,
        pref_keep_frac=pref_keep_frac,
        pref_ignore_frac=pref_ignore_frac,
        pref_margin_mean=pref_margin_mean,
        pref_need_mean=pref_need_mean,
        pref_admiss_mean=pref_admiss_mean,
        pref_target_mean=pref_target_mean,
        tri_weight_repair_mean=tri_weight_repair_mean,
        tri_weight_noisy_mean=tri_weight_noisy_mean,
        tri_weight_stable_mean=tri_weight_stable_mean,
        pref_inertial_penalty_rho_mean=pref_inertial_penalty_rho_mean,
        pref_inertial_penalty_one_mean=pref_inertial_penalty_one_mean,
        rho_target_planar_mean=rho_target_planar_mean,
        rho_target_rp_mean=rho_target_rp_mean,
        rho_target_z_mean=rho_target_z_mean,
        rho_target_spread_mean=rho_target_spread_mean,
        grouped_rho_mask_mean=grouped_rho_mask_mean,
        rho_regret_up_planar_mean=rho_regret_up_planar_mean,
        rho_regret_up_rp_mean=rho_regret_up_rp_mean,
        rho_regret_up_z_mean=rho_regret_up_z_mean,
        rho_regret_down_planar_mean=rho_regret_down_planar_mean,
        rho_regret_down_rp_mean=rho_regret_down_rp_mean,
        rho_regret_down_z_mean=rho_regret_down_z_mean,
    )


def build_and_write_frontres_acceptance_payload(
    runner: Any,
    *,
    actions: torch.Tensor,
    reward_context: FrontRESRewardContext,
    state_alpha_target: torch.Tensor,
    state_alpha_mask: torch.Tensor,
    quat_to_rotvec_wxyz: Any,
    quat_mul_fn: Any,
    quat_inv_fn: Any,
) -> FrontRESAcceptancePayload:
    """Build rollout acceptance/rho targets, write them to transition, and summarize diagnostics."""

    ctx = reward_context
    window = ctx.reward_window
    n_candidate = ctx.candidate_end - ctx.candidate_start
    n_exec = ctx.n_exec
    candidate_start = ctx.candidate_start
    base_start = ctx.base_start

    accept_payload = initialize_frontres_acceptance_payload(runner)
    accept_target = accept_payload.accept_target
    accept_mask = accept_payload.accept_mask
    pref_inertial_penalty_rho_mean = accept_payload.pref_inertial_penalty_rho_mean
    pref_inertial_penalty_one_mean = accept_payload.pref_inertial_penalty_one_mean
    tri_weight_repair_mean = accept_payload.tri_weight_repair_mean
    tri_weight_noisy_mean = accept_payload.tri_weight_noisy_mean
    tri_weight_stable_mean = accept_payload.tri_weight_stable_mean
    rho_target_planar_mean = accept_payload.rho_target_planar_mean
    rho_target_rp_mean = accept_payload.rho_target_rp_mean
    rho_target_z_mean = accept_payload.rho_target_z_mean
    rho_target_spread_mean = accept_payload.rho_target_spread_mean
    grouped_rho_mask_mean = accept_payload.grouped_rho_mask_mean
    rho_regret_up_planar_mean = accept_payload.rho_regret_up_planar_mean
    rho_regret_up_rp_mean = accept_payload.rho_regret_up_rp_mean
    rho_regret_up_z_mean = accept_payload.rho_regret_up_z_mean
    rho_regret_down_planar_mean = accept_payload.rho_regret_down_planar_mean
    rho_regret_down_rp_mean = accept_payload.rho_regret_down_rp_mean
    rho_regret_down_z_mean = accept_payload.rho_regret_down_z_mean

    structured_joint_requested = runner._frontres_structured_joint_effective_enabled()
    legacy_pref_enabled = bool(runner.cfg.get("frontres_acceptance_preference_enabled", True))
    pref_enabled = (legacy_pref_enabled or structured_joint_requested) and n_candidate > 0 and n_exec > 0
    if pref_enabled:
        pref_margin = max(float(runner.cfg.get("frontres_acceptance_preference_margin", 0.003)), 0.0)
        j_rho = ctx.repair_gain
        j_one = ctx.candidate_gain
        j_zero = torch.zeros_like(ctx.repair_gain)
        c_zero = None
        c_one = None
        if bool(runner.cfg.get("frontres_inertial_preference_enabled", False)):
            have_inertia = (
                hasattr(ctx.cmd, "robot_anchor_quat_w")
                and hasattr(ctx.cmd, "robot_anchor_ang_vel_w")
            )
            if have_inertia:
                robot_q = ctx.cmd.robot_anchor_quat_w[:n_exec].to(runner.device)
                robot_w = ctx.cmd.robot_anchor_ang_vel_w[:n_exec].to(runner.device)
                robot_p = (
                    ctx.cmd.robot_anchor_pos_w[:n_exec].to(runner.device)
                    if hasattr(ctx.cmd, "robot_anchor_pos_w")
                    else None
                )
                robot_v = (
                    ctx.cmd.robot_anchor_lin_vel_w[:n_exec].to(runner.device)
                    if hasattr(ctx.cmd, "robot_anchor_lin_vel_w")
                    else None
                )
                ang_w = float(runner.cfg.get("frontres_inertial_preference_ang_weight", 0.5))

                def branch_compat(branch_pos, branch_q):
                    rot_err = quat_to_rotvec_wxyz(quat_mul_fn(quat_inv_fn(robot_q), branch_q))[:, :3]
                    compat = torch.zeros(n_exec, device=runner.device, dtype=j_rho.dtype)
                    if branch_pos is not None and robot_p is not None and robot_v is not None:
                        pos_err = branch_pos - robot_p
                        compat = compat + (pos_err * robot_v).sum(-1) / (
                            pos_err.norm(dim=-1) * robot_v.norm(dim=-1) + 1e-8
                        )
                    compat = compat + ang_w * (rot_err * robot_w).sum(-1) / (
                        rot_err.norm(dim=-1) * robot_w.norm(dim=-1) + 1e-8
                    )
                    return torch.nan_to_num(compat, nan=0.0, posinf=0.0, neginf=0.0)

                pos_all = getattr(ctx.cmd, "anchor_pos_w", None)
                quat_all = getattr(ctx.cmd, "anchor_quat_w", None)
                if quat_all is not None:
                    noisy_pos = (
                        pos_all[base_start:base_start + n_exec].to(runner.device)
                        if pos_all is not None
                        else None
                    )
                    rho_pos = pos_all[:n_exec].to(runner.device) if pos_all is not None else None
                    one_pos = (
                        pos_all[candidate_start:candidate_start + n_exec].to(runner.device)
                        if pos_all is not None and n_candidate > 0
                        else None
                    )
                    c_zero = branch_compat(noisy_pos, quat_all[base_start:base_start + n_exec].to(runner.device))
                    c_rho = branch_compat(rho_pos, quat_all[:n_exec].to(runner.device))
                    c_one = branch_compat(one_pos, quat_all[candidate_start:candidate_start + n_exec].to(runner.device))
                    inertial_margin = float(runner.cfg.get("frontres_inertial_preference_margin", 0.05))
                    inertial_weight = max(0.0, float(runner.cfg.get("frontres_inertial_preference_weight", 0.0)))
                    penalty_rho = torch.relu(c_zero - c_rho + inertial_margin)
                    penalty_one = torch.relu(c_zero - c_one + inertial_margin)
                    j_rho = j_rho - inertial_weight * penalty_rho
                    j_one = j_one - inertial_weight * penalty_one
                    pref_inertial_penalty_rho_mean = penalty_rho.mean()
                    pref_inertial_penalty_one_mean = penalty_one.mean()

        full_win = (j_one > j_rho + pref_margin) & (j_one > j_zero + pref_margin)
        noop_win = (j_zero > j_rho + pref_margin) & (j_zero > j_one + pref_margin)
        keep_win = (j_rho > j_one + pref_margin) & (j_rho > j_zero + pref_margin)
        regret_target_enabled = bool(runner.cfg.get("frontres_acceptance_regret_target_enabled", True))
        if regret_target_enabled:
            regret_mask_floor = float(runner.cfg.get("frontres_acceptance_regret_soft_mask_floor", 1.0))
            regret_mask_floor = max(0.0, min(1.0, regret_mask_floor))
            repair_pref_gate = (regret_mask_floor + (1.0 - regret_mask_floor) * window.repair_gate).clamp(0.0, 1.0)
            oracle_pref_floor = float(runner.cfg.get("frontres_acceptance_regret_oracle_trust_floor", 0.25))
            oracle_pref_floor = max(0.0, min(1.0, oracle_pref_floor))
            oracle_pref_gate = (oracle_pref_floor + (1.0 - oracle_pref_floor) * window.oracle_trust).clamp(0.0, 1.0)
        else:
            repair_pref_gate = window.repair_gate
            oracle_pref_gate = window.oracle_trust
        pref_gate = (oracle_pref_gate * repair_pref_gate * window.learnable_route_mask).detach().clamp(0.0, 1.0)
        task_conf_dim = int(getattr(getattr(runner.alg, "policy", None), "task_conf_dim", 2))
        tri_rho_payload = build_frontres_tri_anchor_rho_payload(
            cfg=runner.cfg,
            actions=actions,
            n_exec=n_exec,
            task_conf_dim=task_conf_dim,
            j_one=j_one,
            j_zero=j_zero,
            pref_margin=pref_margin,
            pref_gate=pref_gate,
            exec_components=ctx.exec_components,
            candidate_start=candidate_start,
            base_start=base_start,
            regret_target_enabled=regret_target_enabled,
            device=runner.device,
        )
        target_exec = tri_rho_payload.target_exec
        mask_exec = tri_rho_payload.mask_exec
        rho_current = tri_rho_payload.rho_current
        rho_space = tri_rho_payload.rho_space
        grouped_targets_enabled = tri_rho_payload.grouped_targets_enabled
        rho_target_planar_mean = tri_rho_payload.rho_target_planar_mean
        rho_target_rp_mean = tri_rho_payload.rho_target_rp_mean
        rho_target_z_mean = tri_rho_payload.rho_target_z_mean
        rho_target_spread_mean = tri_rho_payload.rho_target_spread_mean
        rho_regret_up_planar_mean = tri_rho_payload.rho_regret_up_planar_mean
        rho_regret_up_rp_mean = tri_rho_payload.rho_regret_up_rp_mean
        rho_regret_up_z_mean = tri_rho_payload.rho_regret_up_z_mean
        rho_regret_down_planar_mean = tri_rho_payload.rho_regret_down_planar_mean
        rho_regret_down_rp_mean = tri_rho_payload.rho_regret_down_rp_mean
        rho_regret_down_z_mean = tri_rho_payload.rho_regret_down_z_mean
        non_tri_acceptance_payload = build_frontres_non_tri_acceptance_target_payload(
            cfg=runner.cfg,
            rho_space=rho_space,
            target_exec=target_exec,
            mask_exec=mask_exec,
            n_exec=n_exec,
            base_start=base_start,
            candidate_start=candidate_start,
            a_w=ctx.a_w,
            a_raw=ctx.a_raw,
            a_fr=ctx.a_fr,
            q_w=ctx.q_w,
            q_raw=ctx.q_raw,
            q_fr=ctx.q_fr,
            c_zero=c_zero,
            c_one=c_one,
            rho_current=rho_current,
            j_one=j_one,
            j_zero=j_zero,
            j_rho=j_rho,
            full_win=full_win,
            noop_win=noop_win,
            keep_win=keep_win,
            pref_margin=pref_margin,
            pref_gate=pref_gate,
            quat_to_rotvec_wxyz=quat_to_rotvec_wxyz,
            quat_mul_fn=quat_mul_fn,
            quat_inv_fn=quat_inv_fn,
            device=runner.device,
        )
        target_exec = non_tri_acceptance_payload.target_exec
        mask_exec = non_tri_acceptance_payload.mask_exec
        need = non_tri_acceptance_payload.need
        admissibility = non_tri_acceptance_payload.admissibility
        if bool(runner.cfg.get("frontres_per_mode_acceptance_preference_mask", True)):
            mode_dim_mask = runner._frontres_action_cone.mode_dim_mask(
                ctx.mode_groups, n_exec, runner.device, mask_exec.dtype
            )
            if regret_target_enabled and grouped_targets_enabled:
                mode_soft_floor = float(runner.cfg.get("frontres_acceptance_regret_per_mode_soft_floor", 1.0))
                mode_soft_floor = max(0.0, min(1.0, mode_soft_floor))
                mode_dim_mask = (mode_soft_floor + (1.0 - mode_soft_floor) * mode_dim_mask).clamp(0.0, 1.0)
            mask_exec = mask_exec * mode_dim_mask
        active_dims_cfg = runner.cfg.get("frontres_active_task_dims", None)
        if active_dims_cfg is not None:
            dim_mask = torch.zeros(6, device=runner.device, dtype=mask_exec.dtype)
            for idx in active_dims_cfg:
                idx = int(idx)
                if 0 <= idx < 6:
                    dim_mask[idx] = 1.0
                elif 6 <= idx < 12:
                    dim_mask[idx - 6] = 1.0
            mask_exec = mask_exec * dim_mask.view(1, -1)
        grouped_rho_mask_mean = mask_exec.mean()
        if rho_space in ("tri_anchor", "tri-anchor", "tri"):
            mask_sum_for_alpha = mask_exec.sum(dim=-1)
            target_mean_for_alpha = target_exec.mean(dim=-1).detach().clamp(0.0, 1.0)
            target_active_for_alpha = (
                (target_exec * mask_exec).sum(dim=-1) / mask_sum_for_alpha.clamp(min=1e-6)
            ).detach().clamp(0.0, 1.0)
            target_sample_for_alpha = torch.where(
                mask_sum_for_alpha > 0.0,
                target_active_for_alpha,
                target_mean_for_alpha,
            )
            tri_alpha_source = getattr(runner, "_frontres_state_alpha_prob_next", None)
            if isinstance(tri_alpha_source, torch.Tensor) and tri_alpha_source.numel() > 0:
                tri_alpha = tri_alpha_source.to(device=runner.device, dtype=target_exec.dtype).view(-1)
                if tri_alpha.numel() < n_exec:
                    tri_alpha = torch.nn.functional.pad(tri_alpha, (0, n_exec - tri_alpha.numel()), value=0.0)
                tri_alpha = tri_alpha[:n_exec].detach().clamp(0.0, 1.0)
            else:
                tri_alpha = state_alpha_target[:n_exec, 0].detach().clamp(0.0, 1.0)
            tri_weight_repair_mean = target_sample_for_alpha.mean()
            tri_weight_stable_mean = ((1.0 - target_sample_for_alpha) * tri_alpha).mean()
            tri_weight_noisy_mean = ((1.0 - target_sample_for_alpha) * (1.0 - tri_alpha)).mean()
        accept_target[:n_exec] = target_exec.detach()
        accept_mask[:n_exec] = mask_exec.detach()
        structured_rho_payload = apply_frontres_structured_rho_payload(
            runner,
            accept_target=accept_target,
            accept_mask=accept_mask,
            target_exec=target_exec,
            mask_exec=mask_exec,
            n_exec=n_exec,
            rho_current=rho_current,
            rho_update_weight=window.rho_update_weight,
            exec_perturbed=ctx.exec_perturbed,
            exec_feasible=ctx.exec_feasible,
            exec_frontres=ctx.exec_frontres,
            exec_candidate=ctx.exec_candidate,
            state_alpha_target=state_alpha_target,
            rho_space=rho_space,
            grouped_targets_enabled=grouped_targets_enabled,
            feasible_components=ctx.feasible_components,
            candidate_planar=tri_rho_payload.candidate_planar,
            candidate_rp=tri_rho_payload.candidate_rp,
            candidate_z=tri_rho_payload.candidate_z,
            projected_planar=tri_rho_payload.projected_planar,
            projected_rp=tri_rho_payload.projected_rp,
            projected_z=tri_rho_payload.projected_z,
            base_planar=tri_rho_payload.base_planar,
            base_rp=tri_rho_payload.base_rp,
            base_z=tri_rho_payload.base_z,
            pref_margin=pref_margin,
        )
        accept_target = structured_rho_payload.accept_target
        accept_mask = structured_rho_payload.accept_mask
        target_exec = structured_rho_payload.target_exec
        mask_exec = structured_rho_payload.mask_exec
        structured_joint_enabled = structured_rho_payload.enabled
        runner._frontres_state_alpha_mask_last = float(state_alpha_mask[:n_exec, 0].mean().detach().item())
        accept_payload = summarize_frontres_acceptance_payload(
            runner,
            accept_target=accept_target,
            accept_mask=accept_mask,
            target_exec=target_exec,
            mask_exec=mask_exec,
            structured_joint_enabled=structured_joint_enabled,
            pref_margin=pref_margin,
            need=need,
            admissibility=admissibility,
            j_one=j_one,
            j_rho=j_rho,
            j_zero=j_zero,
            tri_weight_repair_mean=tri_weight_repair_mean,
            tri_weight_noisy_mean=tri_weight_noisy_mean,
            tri_weight_stable_mean=tri_weight_stable_mean,
            pref_inertial_penalty_rho_mean=pref_inertial_penalty_rho_mean,
            pref_inertial_penalty_one_mean=pref_inertial_penalty_one_mean,
            rho_target_planar_mean=rho_target_planar_mean,
            rho_target_rp_mean=rho_target_rp_mean,
            rho_target_z_mean=rho_target_z_mean,
            rho_target_spread_mean=rho_target_spread_mean,
            grouped_rho_mask_mean=grouped_rho_mask_mean,
            rho_regret_up_planar_mean=rho_regret_up_planar_mean,
            rho_regret_up_rp_mean=rho_regret_up_rp_mean,
            rho_regret_up_z_mean=rho_regret_up_z_mean,
            rho_regret_down_planar_mean=rho_regret_down_planar_mean,
            rho_regret_down_rp_mean=rho_regret_down_rp_mean,
            rho_regret_down_z_mean=rho_regret_down_z_mean,
        )
        accept_target = accept_payload.accept_target
        accept_mask = accept_payload.accept_mask

    runner.alg.transition.acceptance_target = accept_target
    runner.alg.transition.acceptance_mask = accept_mask
    return accept_payload


def write_rho_update_weight(
    runner: Any,
    *,
    n_exec: int,
    rho_update_weight: torch.Tensor,
) -> torch.Tensor:
    """Write rollout rho-update weight into the active transition."""

    return runner._frontres_alpha_rho_bridge.write_rho_update_weight(
        runner.alg.transition,
        num_envs=runner.env.num_envs,
        n_exec=n_exec,
        rho_update_weight=rho_update_weight,
        device=runner.device,
    )


def write_frontres_sample_weight(
    runner: Any,
    *,
    n_exec: int,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Legacy alias for write_rho_update_weight()."""
    return write_rho_update_weight(
        runner,
        n_exec=n_exec,
        rho_update_weight=sample_weight,
    )


def write_frontres_actor_gate(
    runner: Any,
    *,
    n_exec: int,
    actor_gate: torch.Tensor,
) -> torch.Tensor:
    """Legacy alias for write_rho_update_weight()."""
    return write_rho_update_weight(
        runner,
        n_exec=n_exec,
        rho_update_weight=actor_gate,
    )


def write_frontres_state_alpha_payload(
    runner: Any,
    *,
    n_exec: int,
    exec_perturbed: torch.Tensor,
    dones: torch.Tensor,
    infos: dict[str, Any],
    base_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and write state-alpha rollout targets into the active transition."""

    state_alpha_target = torch.zeros(runner.env.num_envs, 1, device=runner.device)
    state_alpha_mask = torch.zeros(runner.env.num_envs, 1, device=runner.device)
    runner._frontres_state_alpha_target_last = 0.0
    runner._frontres_state_alpha_mask_last = 0.0
    if bool(runner.cfg.get("frontres_state_alpha_enabled", True)) and n_exec > 0:
        base_done = dones[base_start:base_start + n_exec].view(-1) > 0
        timeout = infos.get("time_outs", None)
        if timeout is not None:
            timeout = timeout.to(runner.device).view(-1)
            base_timeout = timeout[base_start:base_start + n_exec] > 0
        else:
            base_timeout = torch.zeros(n_exec, device=runner.device, dtype=torch.bool)
        alpha_exec_floor = float(
            getattr(
                runner,
                "_frontres_exec_floor_value_last",
                runner.cfg.get("frontres_state_alpha_exec_floor", 0.0),
            )
        )
        alpha_safe_floor = float(
            getattr(
                runner,
                "_frontres_exec_floor_safe_last",
                runner.cfg.get(
                    "frontres_state_alpha_safe_exec_floor",
                    max(alpha_exec_floor + 0.05, alpha_exec_floor),
                ),
            )
        )
        alpha_temp = max(1e-6, float(runner.cfg.get("frontres_state_alpha_temp", 0.08)))
        alpha_targets = build_state_alpha_targets(
            num_envs=runner.env.num_envs,
            n_exec=n_exec,
            exec_perturbed=exec_perturbed,
            base_done=base_done,
            base_timeout=base_timeout,
            exec_floor=alpha_exec_floor,
            safe_floor=alpha_safe_floor,
            temp=alpha_temp,
            device=runner.device,
        )
        state_alpha_target = alpha_targets.target
        state_alpha_mask = alpha_targets.mask
        runner._frontres_state_alpha_target_last = alpha_targets.target_mean
        runner._frontres_state_alpha_mask_last = alpha_targets.mask_mean
    runner.alg.transition.state_alpha_target = state_alpha_target
    runner.alg.transition.state_alpha_mask = state_alpha_mask
    return state_alpha_target, state_alpha_mask


# Readability aliases used by the runner.
write_rho_advantage = build_and_write_frontres_acceptance_payload
write_rho_groundtruth = build_and_write_frontres_acceptance_payload  # legacy alias
write_actor_sample_weight = write_frontres_sample_weight
write_alpha_groundtruth = write_frontres_state_alpha_payload
