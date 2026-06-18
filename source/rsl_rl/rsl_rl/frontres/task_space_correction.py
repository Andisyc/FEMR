# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES task-space correction application helpers."""

from __future__ import annotations

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_inv, quat_mul

from rsl_rl.frontres.frontres_executability import (
    quat_to_rotvec_wxyz as _quat_to_rotvec_wxyz,
    rotvec_to_quat_wxyz as _rotvec_to_quat_wxyz,
)
from rsl_rl.frontres.temporal_reference_cache import (
    frontres_raw_anchor_pose,
    frontres_update_temporal_reference_cache,
)
from rsl_rl.modules import FrontRESActorCritic


def mask_frontres_task_actions(self, actions: torch.Tensor) -> torch.Tensor:
    """Apply the configured task-space action cone to correction proposals.

    In task-space mode actions are
    [dx, dy, dz, droll, dpitch, dyaw, alpha...]. The action cone should zero
    inactive correction proposals. It should not force inactive sigmoid
    coefficients to zero: that value is on the boundary of the transformed
    action distribution and would corrupt PPO log-prob ratios. Coefficients on
    inactive axes are harmless once their proposal is zero.
    """
    active_dims = self.cfg.get("frontres_active_task_dims", None)
    if active_dims is None:
        return actions
    mask = torch.ones(actions.shape[-1], device=actions.device, dtype=actions.dtype)
    proposal_dim = min(6, actions.shape[-1])
    mask[:proposal_dim] = 0.0
    for idx in active_dims:
        idx = int(idx)
        if 0 <= idx < proposal_dim:
            mask[idx] = 1.0
    return actions * mask.view(1, -1)


def _reset_frontres_route_stats(self, n_train: int, device: torch.device) -> None:
    self._frontres_stable_route_applied_frac = 0.0
    self._frontres_stable_endpoint_frac = 0.0
    self._frontres_tri_weight_repair = 0.0
    self._frontres_tri_weight_noisy = 1.0
    self._frontres_tri_weight_stable = 0.0
    self._frontres_stable_route_active_mask = torch.zeros(n_train, device=device, dtype=torch.bool)


def _maybe_mix_oracle_task_correction(
    self,
    env_raw,
    task_corr: torch.Tensor,
    n_train: int,
    allow_oracle: bool,
) -> torch.Tensor:
    if not (allow_oracle and self.cfg.get("oracle_curriculum", False)):
        return task_corr
    for cmd_oracle in env_raw.command_manager._terms.values():
        if not hasattr(cmd_oracle, "supervised_target"):
            continue
        sup = cmd_oracle.supervised_target.to(task_corr.device)
        oracle_full = torch.zeros_like(task_corr)
        n = min(sup.shape[-1], oracle_full.shape[-1])
        oracle_full[:, :n] = sup[:, :n]

        fr_v = task_corr[:n_train, :n]
        or_v = oracle_full[:n_train, :n]
        if fr_v.numel() > 0:
            cos_s = (fr_v * or_v).sum(-1) / (fr_v.norm(dim=-1) * or_v.norm(dim=-1) + 1e-8)
            ema_alpha = 0.99
            prev_ema = getattr(self, "_oracle_cos_ema", 0.0)
            new_ema = ema_alpha * prev_ema + (1.0 - ema_alpha) * float(cos_s.mean())
            self._oracle_cos_ema = new_ema

            cos_lo = float(self.cfg.get("oracle_mix_cos_low", 0.3))
            cos_hi = float(self.cfg.get("oracle_mix_cos_high", 0.85))
            if new_ema < cos_lo:
                mix = 1.0
            elif new_ema < cos_hi:
                mix = 1.0 - (new_ema - cos_lo) / max(cos_hi - cos_lo, 1e-6)
            else:
                mix = 0.0
            self._oracle_mix = mix
            if mix > 0.0:
                task_corr = (1.0 - mix) * task_corr + mix * oracle_full
        break
    return task_corr


def _frontres_acceptance_coefficients(
    self,
    task_corr: torch.Tensor,
    n_train: int,
    policy,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    task_conf_dim = int(getattr(policy, "task_conf_dim", 2))
    acceptance = None
    if task_corr.shape[-1] >= 12 and task_conf_dim == 6:
        acceptance = task_corr[:n_train, 6:12].clone().clamp(0.0, 1.0)
        c_pos = acceptance[:, :3]
        c_rpy = acceptance[:, 3:6]
    elif task_corr.shape[-1] >= 7 and task_conf_dim == 1:
        rho_pos = task_corr[:n_train, 6:7].clone().clamp(0.0, 1.0)
        c_pos = torch.ones_like(rho_pos)
        c_rpy = torch.ones_like(rho_pos)
    else:
        c_pos = task_corr[:n_train, 6:7].clone()
        c_rpy = task_corr[:n_train, 7:8].clone()
    if objective == "supervised_restore":
        c_pos = torch.ones_like(c_pos)
        c_rpy = torch.ones_like(c_rpy)
    return c_pos, c_rpy, acceptance


def _frontres_contact_consistent_position_correction(
    self,
    cmd_term,
    pos_corr: torch.Tensor,
    n_train: int,
    task_corr: torch.Tensor,
) -> torch.Tensor:
    z_upper = torch.zeros_like(pos_corr[:, 2])
    if hasattr(cmd_term, "jump_degree"):
        jd = cmd_term.jump_degree[:n_train].to(task_corr.device).clamp(0.0, 1.0)
        contact_gate = (1.0 - jd).unsqueeze(-1)
        pos_corr[:, :2] = pos_corr[:, :2] * contact_gate
        if hasattr(cmd_term, "anchor_penetration_depth"):
            penetration = cmd_term.anchor_penetration_depth[:n_train].to(task_corr.device)
            z_upper = jd * penetration

    z_lower = torch.full_like(pos_corr[:, 2], -self.alg.policy.max_delta_pos)
    pos_corr[:, 2] = torch.maximum(pos_corr[:, 2], z_lower)
    pos_corr[:, 2] = torch.minimum(pos_corr[:, 2], z_upper)
    return pos_corr


def _frontres_tri_anchor_weights(
    self,
    c_pos: torch.Tensor,
    c_rpy: torch.Tensor,
    alpha_pos: torch.Tensor | None = None,
) -> None:
    self._frontres_tri_weight_repair = float(0.5 * (c_pos.mean() + c_rpy.mean()).detach().item())
    if alpha_pos is None:
        self._frontres_tri_weight_noisy = max(0.0, 1.0 - self._frontres_tri_weight_repair)
        self._frontres_tri_weight_stable = 0.0
        return
    self._frontres_tri_weight_stable = float(
        0.5
        * (
            ((1.0 - c_pos) * alpha_pos).mean()
            + ((1.0 - c_rpy) * alpha_pos).mean()
        ).detach().item()
    )
    self._frontres_tri_weight_noisy = max(
        0.0,
        1.0 - self._frontres_tri_weight_repair - self._frontres_tri_weight_stable,
    )


def _frontres_route_mask(self, n_train: int, task_corr: torch.Tensor) -> torch.Tensor | None:
    route_mask = getattr(self, "_frontres_stable_route_next_mask", None)
    if route_mask is None:
        return None
    route_mask = route_mask.to(device=task_corr.device).view(-1).bool()
    if route_mask.numel() < n_train:
        route_mask = torch.nn.functional.pad(route_mask, (0, n_train - route_mask.numel()), value=False)
    return route_mask[:n_train]


def _clear_frontres_stable_route_mask(self, n_train: int, task_corr: torch.Tensor) -> None:
    self._frontres_stable_route_active_mask = torch.zeros(n_train, device=task_corr.device, dtype=torch.bool)
    self._frontres_stable_route_applied_frac = 0.0


def _frontres_apply_stable_route_override(
    self,
    cmd_term,
    pos_corr: torch.Tensor,
    rpy_corr: torch.Tensor,
    c_pos: torch.Tensor,
    c_rpy: torch.Tensor,
    stable_pos_corr: torch.Tensor | None,
    stable_rpy_corr: torch.Tensor | None,
    n_train: int,
    task_corr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    route_mask = _frontres_route_mask(self, n_train, task_corr)
    if route_mask is None or not route_mask.any():
        _clear_frontres_stable_route_mask(self, n_train, task_corr)
        return pos_corr, rpy_corr, c_pos, c_rpy

    if stable_pos_corr is None or stable_rpy_corr is None:
        stable = frontres_stabilizing_candidate_correction(self, cmd_term, n_train, task_corr.device, task_corr.dtype)
        if stable is not None:
            stable_pos_corr, stable_rpy_corr = stable
    if stable_pos_corr is None or stable_rpy_corr is None:
        _clear_frontres_stable_route_mask(self, n_train, task_corr)
        return pos_corr, rpy_corr, c_pos, c_rpy

    pos_corr = torch.where(route_mask[:, None], stable_pos_corr, pos_corr)
    rpy_corr = torch.where(route_mask[:, None], stable_rpy_corr, rpy_corr)
    c_pos = torch.where(route_mask[:, None], torch.ones_like(c_pos), c_pos)
    c_rpy = torch.where(route_mask[:, None], torch.ones_like(c_rpy), c_rpy)
    self._frontres_stable_route_active_mask = route_mask.detach()
    self._frontres_stable_route_applied_frac = float(route_mask.to(task_corr.dtype).mean().detach().item())
    return pos_corr, rpy_corr, c_pos, c_rpy


def _compose_frontres_route_corrections(
    self,
    cmd_term,
    pos_corr: torch.Tensor,
    rpy_corr: torch.Tensor,
    c_pos: torch.Tensor,
    c_rpy: torch.Tensor,
    acceptance: torch.Tensor | None,
    n_train: int,
    n_candidate: int,
    task_corr: torch.Tensor,
    allow_oracle: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    objective = str(getattr(self.alg, "frontres_training_objective", "")).lower()
    rho_space = str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
    stable_to_repair = (
        objective == "hsl_hybrid"
        and acceptance is not None
        and rho_space in ("stable_to_repair", "stable-repair", "stable")
    )
    tri_anchor = (
        objective == "hsl_hybrid"
        and acceptance is not None
        and rho_space in ("tri_anchor", "tri-anchor", "tri")
    )

    stable_pos_corr = None
    stable_rpy_corr = None
    if stable_to_repair:
        stable = frontres_stabilizing_candidate_correction(self, cmd_term, n_train, task_corr.device, task_corr.dtype)
        if stable is not None:
            stable_pos_corr, stable_rpy_corr = stable
            pos_corr = stable_pos_corr + c_pos * (pos_corr - stable_pos_corr)
            rpy_corr = stable_rpy_corr + c_rpy * (rpy_corr - stable_rpy_corr)
            c_pos = torch.ones_like(c_pos)
            c_rpy = torch.ones_like(c_rpy)
            self._frontres_stable_endpoint_frac = 1.0
        else:
            self._frontres_stable_endpoint_frac = 0.0
    elif tri_anchor:
        stable = frontres_stabilizing_candidate_correction(self, cmd_term, n_train, task_corr.device, task_corr.dtype)
        if stable is not None:
            stable_pos_corr, stable_rpy_corr = stable
            alpha_source = getattr(self, "_frontres_state_alpha_prob_next", None)
            if alpha_source is None:
                alpha = torch.zeros(n_train, device=task_corr.device, dtype=task_corr.dtype)
            else:
                alpha = alpha_source.to(device=task_corr.device, dtype=task_corr.dtype).view(-1)
                if alpha.numel() < n_train:
                    alpha = torch.nn.functional.pad(alpha, (0, n_train - alpha.numel()), value=0.0)
                alpha = alpha[:n_train].clamp(0.0, 1.0)
            alpha_pos = alpha[:, None]
            pos_corr = c_pos * pos_corr + (1.0 - c_pos) * alpha_pos * stable_pos_corr
            rpy_corr = c_rpy * rpy_corr + (1.0 - c_rpy) * alpha_pos * stable_rpy_corr
            _frontres_tri_anchor_weights(self, c_pos, c_rpy, alpha_pos)
            c_pos = torch.ones_like(c_pos)
            c_rpy = torch.ones_like(c_rpy)
            self._frontres_stable_endpoint_frac = 0.0
        else:
            self._frontres_stable_endpoint_frac = 0.0
            _frontres_tri_anchor_weights(self, c_pos, c_rpy)

    if (
        (not tri_anchor)
        and allow_oracle
        and n_candidate > 0
        and bool(self.cfg.get("frontres_stable_route_enabled", True))
    ):
        return _frontres_apply_stable_route_override(
            self,
            cmd_term,
            pos_corr,
            rpy_corr,
            c_pos,
            c_rpy,
            stable_pos_corr,
            stable_rpy_corr,
            n_train,
            task_corr,
        )

    _clear_frontres_stable_route_mask(self, n_train, task_corr)
    return pos_corr, rpy_corr, c_pos, c_rpy


def _write_frontres_command_correction(
    self,
    cmd_term,
    pos_corr: torch.Tensor,
    rpy_corr: torch.Tensor,
    cand_pos_corr: torch.Tensor | None,
    cand_rpy_corr: torch.Tensor | None,
    n_train: int,
    n_candidate: int,
) -> None:
    cmd_term._frontres_pos_correction[:n_train].copy_(pos_corr)
    cmd_term._frontres_quat_correction[:n_train].copy_(_rotvec_to_quat_wxyz(rpy_corr))
    if n_candidate > 0 and cand_pos_corr is not None and cand_rpy_corr is not None:
        cand_start = n_train
        cand_end = cand_start + n_candidate
        cmd_term._frontres_pos_correction[cand_start:cand_end].copy_(cand_pos_corr)
        cmd_term._frontres_quat_correction[cand_start:cand_end].copy_(_rotvec_to_quat_wxyz(cand_rpy_corr))
    frontres_update_temporal_reference_cache(self, cmd_term, n_train)
    zero_start = n_train + n_candidate
    if zero_start < self.env.num_envs:
        cmd_term._frontres_pos_correction[zero_start:].zero_()
        cmd_term._frontres_quat_correction[zero_start:].zero_()
        cmd_term._frontres_quat_correction[zero_start:, 0] = 1.0


def apply_frontres_task_corrections(
    self,
    task_corr: torch.Tensor | None,
    n_train: int | None = None,
    *,
    allow_oracle: bool = False,
    n_candidate: int = 0,
) -> torch.Tensor | None:
    """Write FrontRES Delta SE(3) into the motion command before GMT/current env step.

    The policy samples/outputs Delta SE(3) from the noisy observation. This
    method applies the same conservative projection used by training rewards so
    a subsequent observation refresh exposes the corrected reference to GMT.
    """
    if task_corr is None:
        return None
    policy = getattr(getattr(self, "alg", None), "policy", None)
    if not isinstance(policy, FrontRESActorCritic):
        return task_corr
    if getattr(policy, "num_task_corrections", 0) <= 0:
        return task_corr

    task_corr = self._mask_frontres_task_actions(task_corr)
    env_raw = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
    if not (hasattr(env_raw, "command_manager") and hasattr(env_raw.command_manager, "_terms")):
        return task_corr

    if n_train is None:
        n_train = task_corr.shape[0]
    n_train = max(0, min(int(n_train), task_corr.shape[0], self.env.num_envs))
    n_candidate = max(0, min(int(n_candidate), max(0, self.env.num_envs - n_train)))
    _reset_frontres_route_stats(self, n_train, task_corr.device)
    task_corr = _maybe_mix_oracle_task_correction(self, env_raw, task_corr, n_train, allow_oracle)

    for cmd_term in env_raw.command_manager._terms.values():
        if not hasattr(cmd_term, "_frontres_pos_correction"):
            continue
        pos_corr = task_corr[:n_train, :3].clone()
        rpy_corr = task_corr[:n_train, 3:6].clone()
        objective = str(getattr(self.alg, "frontres_training_objective", "")).lower()
        c_pos, c_rpy, acceptance = _frontres_acceptance_coefficients(self, task_corr, n_train, policy, objective)
        pos_corr = _frontres_contact_consistent_position_correction(self, cmd_term, pos_corr, n_train, task_corr)
        cand_pos_corr = pos_corr[:n_candidate].clone() if n_candidate > 0 else None
        cand_rpy_corr = rpy_corr[:n_candidate].clone() if n_candidate > 0 else None
        pos_corr, rpy_corr, c_pos, c_rpy = _compose_frontres_route_corrections(
            self,
            cmd_term,
            pos_corr,
            rpy_corr,
            c_pos,
            c_rpy,
            acceptance,
            n_train,
            n_candidate,
            task_corr,
            allow_oracle,
        )
        pos_corr = pos_corr * c_pos
        rpy_corr = rpy_corr * c_rpy

        _write_frontres_command_correction(
            self,
            cmd_term,
            pos_corr,
            rpy_corr,
            cand_pos_corr,
            cand_rpy_corr,
            n_train,
            n_candidate,
        )
    return task_corr


def frontres_stabilizing_candidate_correction(
    self,
    cmd_term,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build a deterministic stable-manifold candidate for HRL route selection.

    This is not a new policy or rollout branch. It replaces the Projected route
    with a conservative upright reference when the previous full-HSL Candidate
    rollout fell below the executable floor.
    """
    if n <= 0:
        return None
    pose = frontres_raw_anchor_pose(self, cmd_term, n, device, dtype)
    if pose is None or not hasattr(cmd_term, "robot_anchor_quat_w"):
        return None
    raw_pos, raw_quat = pose
    robot_quat = cmd_term.robot_anchor_quat_w[:n].to(device=device, dtype=dtype)
    _, _, robot_yaw = euler_xyz_from_quat(robot_quat)
    zeros = torch.zeros_like(robot_yaw)
    stable_quat = quat_from_euler_xyz(zeros, zeros, robot_yaw)
    stable_quat = stable_quat / stable_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    stable_corr_quat = quat_mul(quat_inv(raw_quat), stable_quat)
    stable_rpy_corr = _quat_to_rotvec_wxyz(stable_corr_quat)
    stable_pos_corr = torch.zeros_like(raw_pos)

    max_delta_pos = float(getattr(getattr(self.alg, "policy", None), "max_delta_pos", 0.3))
    max_delta_rpy = float(getattr(getattr(self.alg, "policy", None), "max_delta_rpy", 0.1))
    stable_pos_corr = stable_pos_corr.clamp(-max_delta_pos, max_delta_pos)
    stable_rpy_corr = stable_rpy_corr.clamp(-max_delta_rpy, max_delta_rpy)

    active_dims = self.cfg.get("frontres_active_task_dims", None)
    if active_dims is not None:
        active = {int(dim) for dim in active_dims}
        for dim in range(3):
            if dim not in active:
                stable_pos_corr[:, dim] = 0.0
        for dim in range(3, 6):
            if dim not in active:
                stable_rpy_corr[:, dim - 3] = 0.0
    return stable_pos_corr, stable_rpy_corr
