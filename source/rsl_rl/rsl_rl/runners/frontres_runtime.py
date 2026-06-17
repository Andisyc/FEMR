"""FrontRES runtime correction and inference helpers.

This module owns deployment-time task-space correction application, temporal
reference cache handling, and restore debug printing. The runner keeps wrappers
for compatibility with the existing rollout code.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_inv, quat_mul

from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.runners.frontres_executability import (
    quat_to_rotvec_wxyz as _quat_to_rotvec_wxyz,
    rotvec_to_quat_wxyz as _rotvec_to_quat_wxyz,
)


def get_inference_policy_runner(self, device=None):
    self.eval_mode()  # switch to evaluation mode (dropout for example)
    if device is not None:
        self.alg.policy.to(device)
    if self.cfg["empirical_normalization"] and device is not None:
        self.obs_normalizer.to(device)

    is_task_space_frontres = (
        isinstance(self.alg.policy, FrontRESActorCritic)
        and getattr(self.alg.policy, "num_task_corrections", 0) > 0
    )

    if is_task_space_frontres:
        def policy(x):  # noqa: E306
            with torch.inference_mode():
                raw_obs = x.to(self.device)
                norm_obs = self._apply_obs_normalizer(raw_obs) if self.cfg["empirical_normalization"] else raw_obs
                if (
                    bool(self.cfg.get("frontres_state_alpha_enabled", True))
                    and hasattr(self.alg.policy, "get_state_router_alpha")
                ):
                    alpha_obs = norm_obs
                    self._frontres_state_alpha_prob_next = (
                        self.alg.policy.get_state_router_alpha(alpha_obs).view(-1).detach()
                    )
                correction = self.alg.policy.get_task_correction_inference(norm_obs)
                self._apply_frontres_task_corrections(correction, correction.shape[0], allow_oracle=False)
                obs_corr, extras_corr = self.env.get_observations()
                obs_corr_dict = extras_corr.get("observations", {})
                if self.policy_obs_type is not None and self.policy_obs_type in obs_corr_dict:
                    obs_corr = obs_corr_dict[self.policy_obs_type]
                obs_corr = obs_corr.to(self.device)
                norm_corr = self._apply_obs_normalizer(obs_corr) if self.cfg["empirical_normalization"] else obs_corr
                return self.alg.policy.get_env_action(norm_corr, correction)
        return policy

    policy = self.alg.policy.act_inference
    if self.cfg["empirical_normalization"]:
        policy = lambda x: self.alg.policy.act_inference(self._apply_obs_normalizer(x.to(self.device)))  # noqa: E731
    return policy

def apply_obs_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
    """Apply obs_normalizer, with partial pass-through for FrontRES task-space mode.

    IsaacLab places Optional obs terms (anchor_root_pos_error_w, anchor_root_rpy_error_w)
    BEFORE regular terms in the concatenated obs tensor, so the layout is:
      [0 : num_extra]           = anchor-error dims  (FrontRES-only, NOT in GMT training)
      [num_extra : num_extra+gmt_dim] = GMT-compatible dims (match GMT training obs exactly)

    where num_extra = obs_dim - gmt_dim  (= 30 = 6 dims/frame × 5 frames).

    We therefore normalize the LAST gmt_dim dims with the frozen GMT normalizer and
    optionally normalize the FIRST num_extra dims with Stage-1 empirical stats.
    Output shape is unchanged (800 dims); structure: [extra | gmt_part].
    """
    if self._frontres_gmt_obs_dim is not None and obs.shape[-1] > self._frontres_gmt_obs_dim:
        gmt_dim   = self._frontres_gmt_obs_dim
        num_extra = obs.shape[-1] - gmt_dim          # = 30 (anchor errors at front)
        extra     = obs[:, :num_extra]               # [0:30]   anchor errors
        gmt_part  = self.obs_normalizer(obs[:, num_extra:])  # [30:800] GMT-compatible → normalize
        _s1_mean = getattr(self, '_frontres_extra_mean', None)
        _s1_std  = getattr(self, '_frontres_extra_std',  None)
        if (_s1_mean is not None and _s1_std is not None
                and _s1_mean.shape[-1] == num_extra
                and _s1_std.shape[-1] == num_extra):
            extra = (extra - _s1_mean) / (_s1_std + 1e-8)
        return torch.cat([extra, gmt_part], dim=-1)  # [anchor_errors | normalized_gmt]
    return self.obs_normalizer(obs)

def mask_frontres_task_actions(self, actions: torch.Tensor) -> torch.Tensor:
    """Apply the configured task-space action cone to correction proposals.

    In task-space mode actions are
    [dx, dy, dz, droll, dpitch, dyaw, alpha...].  The action cone should
    zero inactive correction proposals.  It should not force inactive
    sigmoid coefficients to zero: that value is on the boundary of the
    transformed action distribution and would corrupt PPO log-prob ratios.
    Coefficients on inactive axes are harmless once their proposal is zero.
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

def apply_frontres_task_corrections(
    self,
    task_corr: torch.Tensor | None,
    n_train: int | None = None,
    *,
    allow_oracle: bool = False,
    n_candidate: int = 0,
) -> torch.Tensor | None:
    """Write FrontRES ΔSE3 into the motion command before GMT/current env step.

    The policy samples/outputs ΔSE3 from the noisy observation.  This method
    applies the same conservative projection used by training rewards so a
    subsequent observation refresh exposes the corrected reference to GMT.
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
    self._frontres_stable_route_applied_frac = 0.0
    self._frontres_stable_endpoint_frac = 0.0
    self._frontres_tri_weight_repair = 0.0
    self._frontres_tri_weight_noisy = 1.0
    self._frontres_tri_weight_stable = 0.0
    self._frontres_stable_route_active_mask = torch.zeros(
        n_train, device=task_corr.device, dtype=torch.bool
    )

    if allow_oracle and self.cfg.get("oracle_curriculum", False):
        for cmd_oracle in env_raw.command_manager._terms.values():
            if hasattr(cmd_oracle, "supervised_target"):
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

    for cmd_term in env_raw.command_manager._terms.values():
        if not hasattr(cmd_term, "_frontres_pos_correction"):
            continue
        pos_corr = task_corr[:n_train, :3].clone()
        rpy_corr = task_corr[:n_train, 3:6].clone()
        objective = str(getattr(self.alg, "frontres_training_objective", "")).lower()
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
        # Candidate rollout remains the full HSL route.  Stable-route
        # substitution is applied only to Projected/HRL envs so Candidate
        # stays a clean evidence source for the next floor test.
        cand_pos_corr = pos_corr[:n_candidate].clone() if n_candidate > 0 else None
        cand_rpy_corr = rpy_corr[:n_candidate].clone() if n_candidate > 0 else None
        stable_pos_corr = None
        stable_rpy_corr = None
        if stable_to_repair:
            stable = frontres_stabilizing_candidate_correction(self, cmd_term, n_train, task_corr.device, task_corr.dtype)
            if stable is not None:
                stable_pos_corr, stable_rpy_corr = stable
                # New HRL coordinate system:
                #   old fallback: projected = rho * repair
                #   active branch: projected = stable + rho * (repair - stable)
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
                self._frontres_tri_weight_repair = float(
                    0.5 * (c_pos.mean() + c_rpy.mean()).detach().item()
                )
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
                c_pos = torch.ones_like(c_pos)
                c_rpy = torch.ones_like(c_rpy)
                self._frontres_stable_endpoint_frac = 0.0
            else:
                self._frontres_stable_endpoint_frac = 0.0
                self._frontres_tri_weight_repair = float(
                    0.5 * (c_pos.mean() + c_rpy.mean()).detach().item()
                )
                self._frontres_tri_weight_noisy = max(0.0, 1.0 - self._frontres_tri_weight_repair)
                self._frontres_tri_weight_stable = 0.0
        if (
            (not tri_anchor)
            and allow_oracle
            and n_candidate > 0
            and bool(self.cfg.get("frontres_stable_route_enabled", True))
        ):
            route_mask = getattr(self, "_frontres_stable_route_next_mask", None)
            if route_mask is not None:
                route_mask = route_mask.to(device=task_corr.device).view(-1).bool()
                if route_mask.numel() < n_train:
                    route_mask = torch.nn.functional.pad(route_mask, (0, n_train - route_mask.numel()), value=False)
                route_mask = route_mask[:n_train]
                if route_mask.any():
                    if stable_pos_corr is None or stable_rpy_corr is None:
                        stable = frontres_stabilizing_candidate_correction(
                            self, cmd_term, n_train, task_corr.device, task_corr.dtype
                        )
                        if stable is not None:
                            stable_pos_corr, stable_rpy_corr = stable
                    if stable_pos_corr is not None and stable_rpy_corr is not None:
                        pos_corr = torch.where(route_mask[:, None], stable_pos_corr, pos_corr)
                        rpy_corr = torch.where(route_mask[:, None], stable_rpy_corr, rpy_corr)
                        c_pos = torch.where(route_mask[:, None], torch.ones_like(c_pos), c_pos)
                        c_rpy = torch.where(route_mask[:, None], torch.ones_like(c_rpy), c_rpy)
                        self._frontres_stable_route_active_mask = route_mask.detach()
                        self._frontres_stable_route_applied_frac = float(
                            route_mask.to(task_corr.dtype).mean().detach().item()
                        )
                    else:
                        self._frontres_stable_route_active_mask = torch.zeros(
                            n_train, device=task_corr.device, dtype=torch.bool
                        )
                        self._frontres_stable_route_applied_frac = 0.0
                else:
                    self._frontres_stable_route_active_mask = torch.zeros(
                        n_train, device=task_corr.device, dtype=torch.bool
                    )
                    self._frontres_stable_route_applied_frac = 0.0
            else:
                self._frontres_stable_route_active_mask = torch.zeros(
                    n_train, device=task_corr.device, dtype=torch.bool
                )
                self._frontres_stable_route_applied_frac = 0.0
        else:
            self._frontres_stable_route_active_mask = torch.zeros(
                n_train, device=task_corr.device, dtype=torch.bool
            )
            self._frontres_stable_route_applied_frac = 0.0
        # Legacy Noisy-to-Repair branch applies rho after route selection.
        # In Stable-to-Repair mode rho has already been absorbed into
        # stable + rho * (repair - stable), so c_pos/c_rpy were reset to 1.
        pos_corr = pos_corr * c_pos
        rpy_corr = rpy_corr * c_rpy

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
    return task_corr

def frontres_raw_anchor_pose(self, cmd_term, n: int, device: torch.device, dtype: torch.dtype):
    """Return raw reference pose before FrontRES correction."""
    if hasattr(cmd_term, "anchor_pos_w_raw"):
        raw_pos = cmd_term.anchor_pos_w_raw[:n].to(device=device, dtype=dtype)
    elif hasattr(cmd_term, "anchor_pos_w"):
        raw_pos = (
            cmd_term.anchor_pos_w[:n].to(device=device, dtype=dtype)
            - cmd_term._frontres_pos_correction[:n].to(device=device, dtype=dtype)
        )
    else:
        return None
    if hasattr(cmd_term, "anchor_quat_w_raw"):
        raw_quat = cmd_term.anchor_quat_w_raw[:n].to(device=device, dtype=dtype)
    elif hasattr(cmd_term, "anchor_quat_w"):
        written_q = cmd_term._frontres_quat_correction[:n].to(device=device, dtype=dtype)
        raw_quat = quat_mul(
            cmd_term.anchor_quat_w[:n].to(device=device, dtype=dtype),
            quat_inv(written_q),
        )
    else:
        return None
    raw_quat = raw_quat / raw_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return raw_pos, raw_quat

def frontres_stabilizing_candidate_correction(
    self,
    cmd_term,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build a deterministic stable-manifold candidate for HRL route selection.

    This is not a new policy or rollout branch.  It replaces the Projected
    route with a conservative upright reference when the previous full-HSL
    Candidate rollout fell below the executable floor.
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

def frontres_temporal_continuity_correction(
    self,
    cmd_term,
    n: int,
    hsl_pos_corr: torch.Tensor,
    hsl_rpy_corr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build the continuity candidate from the previous refined frame.

    HSL gives the clean-oriented repair for the current raw frame.  This
    helper is kept for legacy temporal-rejoin ablations; the active
    hsl_hybrid branch now uses PPO-owned per-axis acceptance over the HSL
    proposal rather than a temporal position rejoin.
    """
    if n <= 0:
        return None
    pose = frontres_raw_anchor_pose(self, cmd_term, n, hsl_pos_corr.device, hsl_pos_corr.dtype)
    if pose is None:
        return None
    raw_pos, raw_quat = pose

    cache = getattr(self, "_frontres_temporal_ref_cache", None)
    if cache is None:
        return None
    prev_raw_pos = cache.get("raw_pos")
    prev_raw_quat = cache.get("raw_quat")
    prev_ref_pos = cache.get("refined_pos")
    prev_ref_quat = cache.get("refined_quat")
    prev_valid = cache.get("valid")
    if (
        prev_raw_pos is None
        or prev_raw_quat is None
        or prev_ref_pos is None
        or prev_ref_quat is None
        or prev_valid is None
    ):
        return None
    if prev_raw_pos.shape[0] < n or prev_ref_pos.shape[0] < n:
        return None

    prev_raw_pos = prev_raw_pos[:n].to(device=raw_pos.device, dtype=raw_pos.dtype)
    prev_ref_pos = prev_ref_pos[:n].to(device=raw_pos.device, dtype=raw_pos.dtype)
    prev_raw_quat = prev_raw_quat[:n].to(device=raw_pos.device, dtype=raw_pos.dtype)
    prev_ref_quat = prev_ref_quat[:n].to(device=raw_pos.device, dtype=raw_pos.dtype)
    valid = prev_valid[:n].to(device=raw_pos.device).bool()

    raw_delta_pos = raw_pos - prev_raw_pos
    cont_ref_pos = prev_ref_pos + raw_delta_pos
    cont_pos_corr = cont_ref_pos - raw_pos

    raw_delta_quat = quat_mul(quat_inv(prev_raw_quat), raw_quat)
    cont_ref_quat = quat_mul(prev_ref_quat, raw_delta_quat)
    cont_corr_quat = quat_mul(quat_inv(raw_quat), cont_ref_quat)
    cont_rpy_corr = _quat_to_rotvec_wxyz(cont_corr_quat)

    max_delta_pos = float(getattr(getattr(self.alg, "policy", None), "max_delta_pos", 0.3))
    max_delta_rpy = float(getattr(getattr(self.alg, "policy", None), "max_delta_rpy", 0.1))
    cont_pos_corr = cont_pos_corr.clamp(-max_delta_pos, max_delta_pos)
    cont_rpy_corr = cont_rpy_corr.clamp(-max_delta_rpy, max_delta_rpy)

    z_upper = torch.zeros(n, device=raw_pos.device, dtype=raw_pos.dtype)
    if hasattr(cmd_term, "jump_degree") and hasattr(cmd_term, "anchor_penetration_depth"):
        jump_degree = cmd_term.jump_degree[:n].to(raw_pos.device).to(raw_pos.dtype).clamp(0.0, 1.0)
        penetration = cmd_term.anchor_penetration_depth[:n].to(raw_pos.device).to(raw_pos.dtype)
        z_upper = (jump_degree * penetration).clamp(max=max_delta_pos)
    z_lower = torch.full_like(z_upper, -max_delta_pos)
    cont_pos_corr[:, 2] = torch.minimum(torch.maximum(cont_pos_corr[:, 2], z_lower), z_upper)
    return cont_pos_corr, cont_rpy_corr, valid

def frontres_update_temporal_reference_cache(self, cmd_term, n: int) -> None:
    """Cache raw/refined reference poses after writing FrontRES corrections."""
    if n <= 0:
        return
    device = cmd_term._frontres_pos_correction.device
    dtype = cmd_term._frontres_pos_correction.dtype
    pose = frontres_raw_anchor_pose(self, cmd_term, n, device, dtype)
    if pose is None:
        return
    raw_pos, raw_quat = pose
    pos_corr = cmd_term._frontres_pos_correction[:n].detach().clone()
    quat_corr = cmd_term._frontres_quat_correction[:n].detach().clone()
    refined_pos = raw_pos.detach().clone() + pos_corr
    refined_quat = quat_mul(raw_quat, quat_corr)
    refined_quat = refined_quat / refined_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cache_size = int(self.env.num_envs)
    cache = getattr(self, "_frontres_temporal_ref_cache", None)
    if cache is None or cache.get("raw_pos", torch.empty(0, device=device)).shape[0] != cache_size:
        cache = {
            "raw_pos": torch.zeros(cache_size, 3, device=device, dtype=dtype),
            "raw_quat": torch.zeros(cache_size, 4, device=device, dtype=dtype),
            "refined_pos": torch.zeros(cache_size, 3, device=device, dtype=dtype),
            "refined_quat": torch.zeros(cache_size, 4, device=device, dtype=dtype),
            "valid": torch.zeros(cache_size, device=device, dtype=torch.bool),
        }
        cache["raw_quat"][:, 0] = 1.0
        cache["refined_quat"][:, 0] = 1.0
        self._frontres_temporal_ref_cache = cache
    cache["raw_pos"][:n].copy_(raw_pos.detach())
    cache["raw_quat"][:n].copy_(raw_quat.detach())
    cache["refined_pos"][:n].copy_(refined_pos.detach())
    cache["refined_quat"][:n].copy_(refined_quat.detach())
    cache["valid"][:n] = True

def frontres_invalidate_temporal_reference_cache(self, dones: torch.Tensor | None) -> None:
    """Drop temporal continuity state for environments that just reset."""
    cache = getattr(self, "_frontres_temporal_ref_cache", None)
    if cache is None or dones is None or "valid" not in cache:
        return
    valid = cache["valid"]
    done_mask = dones.to(device=valid.device).view(-1).bool()
    n = min(done_mask.numel(), valid.numel())
    if n <= 0:
        return
    valid[:n] &= ~done_mask[:n]

def maybe_print_frontres_restore_debug(
    self,
    it: int,
    rollout_step: int,
    actions: torch.Tensor | None,
    supervised_target: torch.Tensor | None,
    n_train: int,
) -> None:
    """Low-frequency consistency print for task-space FrontRES restore."""
    if actions is None or supervised_target is None:
        return
    if rollout_step != 0:
        return
    objective = str(getattr(self.alg, "frontres_training_objective", "")).lower()
    if objective not in ("supervised_restore", "basis_restore", "hsl_hybrid"):
        return
    interval = int(getattr(self.alg, "frontres_restore_debug_print_interval", self.cfg.get("frontres_restore_debug_print_interval", 10)))
    if interval <= 0 or int(it) % interval != 0:
        return
    if getattr(self, "_frontres_restore_debug_last_iter", None) == int(it):
        return
    self._frontres_restore_debug_last_iter = int(it)

    env_raw = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
    if not (hasattr(env_raw, "command_manager") and hasattr(env_raw.command_manager, "_terms")):
        return
    cmd_term = None
    for term in env_raw.command_manager._terms.values():
        needed = (
            "anchor_quat_w_original",
            "anchor_quat_w_raw",
            "_frontres_quat_correction",
        )
        if all(hasattr(term, name) for name in needed):
            cmd_term = term
            break
    if cmd_term is None:
        return

    n = max(0, min(int(n_train), actions.shape[0], supervised_target.shape[0]))
    if n <= 0:
        return

    raw_q = cmd_term.anchor_quat_w_raw[:n].to(self.device)
    clean_q = cmd_term.anchor_quat_w_original[:n].to(self.device)
    written_q = cmd_term._frontres_quat_correction[:n].to(self.device)
    have_pos_debug = (
        hasattr(cmd_term, "anchor_pos_w_raw")
        and hasattr(cmd_term, "anchor_pos_w_original")
        and hasattr(cmd_term, "_frontres_pos_correction")
    )
    if have_pos_debug:
        raw_p = cmd_term.anchor_pos_w_raw[:n].to(self.device)
        clean_p = cmd_term.anchor_pos_w_original[:n].to(self.device)
        written_p = cmd_term._frontres_pos_correction[:n].to(self.device)
        target_pos = supervised_target[:n, :3].detach()
        pred_pos = actions[:n, :3].detach()
    target = supervised_target[:n, 3:6].detach()
    pred = actions[:n, 3:6].detach()
    task_conf_dim = int(getattr(self.alg.policy, "task_conf_dim", 2))
    if actions.shape[-1] >= 12 and task_conf_dim == 6:
        conf_raw = actions[:n, 6:12].detach()
    elif actions.shape[-1] >= 7 and task_conf_dim == 1:
        conf_raw = actions[:n, 6:7].detach()
    elif actions.shape[-1] >= 8:
        conf_raw = actions[:n, 7:8].detach()
    else:
        conf_raw = torch.ones(n, 1, device=self.device)
    acceptance_hybrid = objective == "hsl_hybrid" and task_conf_dim == 6
    scalar_rejoin_hybrid = objective == "hsl_hybrid" and task_conf_dim == 1
    if objective == "supervised_restore" or scalar_rejoin_hybrid:
        conf_eff = torch.ones_like(conf_raw)
    else:
        conf_eff = conf_raw
    written = _quat_to_rotvec_wxyz(written_q)[:, :3]
    if acceptance_hybrid:
        applied = pred * conf_eff[:, 3:6]
    elif scalar_rejoin_hybrid:
        applied = written
    else:
        applied = pred * conf_eff

    clean_from_raw = _quat_to_rotvec_wxyz(quat_mul(quat_inv(raw_q), clean_q))[:, :3]
    corrected_q = quat_mul(raw_q, written_q)
    corrected_err = _quat_to_rotvec_wxyz(quat_mul(quat_inv(corrected_q), clean_q))[:, :3]
    alt_corrected_q = quat_mul(written_q, raw_q)
    alt_corrected_err = _quat_to_rotvec_wxyz(quat_mul(quat_inv(alt_corrected_q), clean_q))[:, :3]

    def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)

    valid = target.norm(dim=-1) > 1e-4
    if valid.any():
        cos_pred_target = _safe_cos(pred[valid], target[valid]).mean()
        cos_written_target = _safe_cos(written[valid], target[valid]).mean()
        sign_match = (torch.sign(pred[valid, :2]) == torch.sign(target[valid, :2])).float().mean()
    else:
        cos_pred_target = torch.tensor(0.0, device=self.device)
        cos_written_target = torch.tensor(0.0, device=self.device)
        sign_match = torch.tensor(0.0, device=self.device)

    noisy_err_norm = clean_from_raw[:, :2].norm(dim=-1)
    corrected_err_norm = corrected_err[:, :2].norm(dim=-1)
    alt_err_norm = alt_corrected_err[:, :2].norm(dim=-1)
    restore_gain = noisy_err_norm - corrected_err_norm
    if have_pos_debug:
        raw_pos_err = (clean_p - raw_p).norm(dim=-1)
        written_pos_err = (clean_p - (raw_p + written_p)).norm(dim=-1)
        candidate_pos_err = (clean_p - (raw_p + pred_pos)).norm(dim=-1)
        pos_valid = target_pos.norm(dim=-1) > 1e-4
        if pos_valid.any():
            pos_cos_pred = _safe_cos(pred_pos[pos_valid], target_pos[pos_valid]).mean()
            pos_cos_written = _safe_cos(written_p[pos_valid], target_pos[pos_valid]).mean()
        else:
            pos_cos_pred = torch.tensor(0.0, device=self.device)
            pos_cos_written = torch.tensor(0.0, device=self.device)
    candidate_rpy_err = (clean_from_raw[:, :2] - pred[:, :2]).norm(dim=-1)
    projected_rpy_err = (clean_from_raw[:, :2] - applied[:, :2]).norm(dim=-1)
    max_delta_rpy = float(getattr(getattr(self.alg, "policy", None), "max_delta_rpy", 0.4))
    sat_frac = (pred[:, :2].abs() > 0.95 * max_delta_rpy).float().mean()

    prev = getattr(self, "_frontres_restore_debug_prev_applied", None)
    if prev is not None and prev.shape == applied.shape:
        step_jump = (applied[:, :2] - prev[:, :2]).norm(dim=-1).mean()
    else:
        step_jump = torch.tensor(0.0, device=self.device)
    self._frontres_restore_debug_prev_applied = applied.detach().clone()

    def _vec(t: torch.Tensor, idx: int = 0) -> list[float]:
        vals = t[idx, :3].detach().cpu().tolist()
        return [round(float(v), 5) for v in vals]

    def _vec6(t: torch.Tensor) -> list[float]:
        vals = t.detach().mean(dim=0).cpu().tolist()
        return [round(float(v), 5) for v in vals]

    def _stats(t: torch.Tensor) -> str:
        vals = t.detach().reshape(-1)
        if vals.numel() == 0:
            return "mean=0.000 std=0.000 min=0.000 max=0.000"
        return (
            f"mean={float(vals.mean()):.3f} "
            f"std={float(vals.std(unbiased=False)):.3f} "
            f"min={float(vals.min()):.3f} "
            f"max={float(vals.max()):.3f}"
        )

    def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
        aa = a.detach().reshape(-1)
        bb = b.detach().reshape(-1)
        mask = torch.isfinite(aa) & torch.isfinite(bb)
        if int(mask.sum().item()) < 2:
            return 0.0
        aa = aa[mask] - aa[mask].mean()
        bb = bb[mask] - bb[mask].mean()
        denom = (aa.norm() * bb.norm()).clamp(min=1e-8)
        return float((aa * bb).sum() / denom)

    def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> float:
        if mask is None or not bool(mask.any()):
            return 0.0
        return float(t.detach()[mask].mean())

    def _branch_state_metrics(
        branch_pos: torch.Tensor | None,
        branch_quat: torch.Tensor,
        robot_pos: torch.Tensor | None,
        robot_quat: torch.Tensor,
        robot_lin_vel: torch.Tensor | None,
        robot_ang_vel: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rot_err = _quat_to_rotvec_wxyz(quat_mul(quat_inv(robot_quat), branch_quat))[:, :3]
        rot_dist = rot_err.norm(dim=-1)
        if branch_pos is not None and robot_pos is not None:
            pos_err = branch_pos - robot_pos
            pos_dist = pos_err.norm(dim=-1)
        else:
            pos_err = None
            pos_dist = torch.zeros_like(rot_dist)
        dist = pos_dist + rot_dist

        compat = torch.zeros_like(rot_dist)
        if pos_err is not None and robot_lin_vel is not None:
            compat = compat + (pos_err * robot_lin_vel).sum(-1) / (
                pos_err.norm(dim=-1) * robot_lin_vel.norm(dim=-1) + 1e-8
            )
        if robot_ang_vel is not None:
            compat = compat + 0.5 * (rot_err * robot_ang_vel).sum(-1) / (
                rot_err.norm(dim=-1) * robot_ang_vel.norm(dim=-1) + 1e-8
            )
        if robot_ang_vel is not None:
            rp_err = rot_err[:, :2]
            rp_vel = robot_ang_vel[:, :2]
            compat_rp = (rp_err * rp_vel).sum(-1) / (
                rp_err.norm(dim=-1) * rp_vel.norm(dim=-1) + 1e-8
            )
        else:
            compat_rp = torch.zeros_like(rot_dist)
        return dist, compat, compat_rp

    inertial_debug = None
    have_inertial_debug = (
        hasattr(cmd_term, "robot_anchor_quat_w")
        and hasattr(cmd_term, "robot_anchor_ang_vel_w")
    )
    if have_inertial_debug:
        robot_q = cmd_term.robot_anchor_quat_w[:n].to(self.device)
        robot_w = cmd_term.robot_anchor_ang_vel_w[:n].to(self.device)
        robot_p = None
        robot_v = None
        noisy_p = candidate_p = projected_p = clean_pos_branch = None
        if have_pos_debug and hasattr(cmd_term, "robot_anchor_pos_w"):
            robot_p = cmd_term.robot_anchor_pos_w[:n].to(self.device)
            noisy_p = raw_p
            candidate_p = raw_p + pred_pos
            clean_pos_branch = clean_p
            if acceptance_hybrid:
                projected_pos_corr = pred_pos * conf_eff[:, :3]
            else:
                projected_pos_corr = pred_pos * conf_eff
            projected_p = raw_p + projected_pos_corr
            if hasattr(cmd_term, "robot_anchor_lin_vel_w"):
                robot_v = cmd_term.robot_anchor_lin_vel_w[:n].to(self.device)

        noisy_q = raw_q
        candidate_q = quat_mul(raw_q, _rotvec_to_quat_wxyz(pred))
        projected_q = quat_mul(raw_q, _rotvec_to_quat_wxyz(applied))
        clean_branch_q = clean_q
        d_noisy, c_noisy, crp_noisy = _branch_state_metrics(
            noisy_p, noisy_q, robot_p, robot_q, robot_v, robot_w
        )
        d_proj, c_proj, crp_proj = _branch_state_metrics(
            projected_p, projected_q, robot_p, robot_q, robot_v, robot_w
        )
        d_cand, c_cand, crp_cand = _branch_state_metrics(
            candidate_p, candidate_q, robot_p, robot_q, robot_v, robot_w
        )
        d_clean, c_clean, crp_clean = _branch_state_metrics(
            clean_pos_branch, clean_branch_q, robot_p, robot_q, robot_v, robot_w
        )
        if noisy_p is not None and projected_p is not None:
            d_p0r1, c_p0r1, crp_p0r1 = _branch_state_metrics(
                noisy_p, candidate_q, robot_p, robot_q, robot_v, robot_w
            )
            d_ppr1, c_ppr1, crp_ppr1 = _branch_state_metrics(
                projected_p, candidate_q, robot_p, robot_q, robot_v, robot_w
            )
        else:
            d_p0r1 = c_p0r1 = crp_p0r1 = torch.zeros_like(d_noisy)
            d_ppr1 = c_ppr1 = crp_ppr1 = torch.zeros_like(d_noisy)
        margin = 0.05
        angle_inv = (d_proj > d_noisy).float()
        anti_inertia = (c_proj < c_noisy - margin).float()
        anti_cand = (c_cand < c_noisy - margin).float()
        anti_clean = (c_clean < c_noisy - margin).float()
        inertial_debug = {
            "d_noisy": d_noisy,
            "d_proj": d_proj,
            "d_cand": d_cand,
            "d_clean": d_clean,
            "c_noisy": c_noisy,
            "c_proj": c_proj,
            "c_cand": c_cand,
            "c_clean": c_clean,
            "crp_noisy": crp_noisy,
            "crp_proj": crp_proj,
            "crp_cand": crp_cand,
            "crp_clean": crp_clean,
            "d_p0r1": d_p0r1,
            "d_ppr1": d_ppr1,
            "c_p0r1": c_p0r1,
            "c_ppr1": c_ppr1,
            "crp_p0r1": crp_p0r1,
            "crp_ppr1": crp_ppr1,
            "angle_inv": angle_inv,
            "anti_inertia": anti_inertia,
            "anti_cand": anti_cand,
            "anti_clean": anti_clean,
            "inertial_gain": c_proj - c_noisy,
        }

    sample_idx = int(torch.argmax(noisy_err_norm).item())
    if acceptance_hybrid:
        gate_desc = (
            f"rho_pos={float(conf_raw[:, :3].mean()):.3f} "
            f"rho_rpy={float(conf_raw[:, 3:6].mean()):.3f}"
        )
    elif scalar_rejoin_hybrid:
        gate_desc = f"rho_pos={float(conf_raw.mean()):.3f}"
    else:
        gate_desc = f"conf_eff={float(conf_eff.mean()):.3f} conf_raw={float(conf_raw.mean()):.3f}"
    print(
        "[FrontRES restore debug] "
        f"it={int(it)} dr={float(getattr(self, '_dr_scale', 0.0)):.4f} "
        f"n={n} sample={sample_idx} "
        f"cos(pred,target)={float(cos_pred_target):+.4f} "
        f"cos(written,target)={float(cos_written_target):+.4f} "
        f"sign_xy={float(sign_match):.3f} {gate_desc} "
        f"sat={float(sat_frac):.3f} jump={float(step_jump):.5f}",
        flush=True,
    )
    print(
        "[FrontRES restore debug] "
        f"|raw-clean|={float(noisy_err_norm.mean()):.5f} "
        f"|corr-clean|={float(corrected_err_norm.mean()):.5f} "
        f"|altcorr-clean|={float(alt_err_norm.mean()):.5f} "
        f"gain={float(restore_gain.mean()):+.5f}",
        flush=True,
    )
    print(
        "[FrontRES restore debug] rpy counterfactual "
        f"noop={float(noisy_err_norm.mean()):.5f} "
        f"full_candidate={float(candidate_rpy_err.mean()):.5f} "
        f"projected={float(projected_rpy_err.mean()):.5f} "
        f"written={float(corrected_err_norm.mean()):.5f} "
        f"gain_full={float((noisy_err_norm - candidate_rpy_err).mean()):+.5f} "
        f"gain_proj={float((noisy_err_norm - projected_rpy_err).mean()):+.5f}",
        flush=True,
    )
    if have_pos_debug:
        print(
            "[FrontRES restore debug] position "
            f"|raw-clean|={float(raw_pos_err.mean()):.5f} "
            f"|written-clean|={float(written_pos_err.mean()):.5f} "
            f"|candidate-clean|={float(candidate_pos_err.mean()):.5f} "
            f"gain_cand={float((raw_pos_err - candidate_pos_err).mean()):+.5f} "
            f"gain_written={float((raw_pos_err - written_pos_err).mean()):+.5f} "
            f"cos(pred,target)={float(pos_cos_pred):+.4f} "
            f"cos(written,target)={float(pos_cos_written):+.4f}",
            flush=True,
        )
    if acceptance_hybrid:
        gate_mean = conf_raw.mean(dim=-1)
        proposal6 = actions[:n, :6].detach()
        target6 = supervised_target[:n, :6].detach()
        applied6 = proposal6 * conf_raw
        positive_gain = restore_gain > 0.0
        print(
            "[FrontRES gate debug] "
            f"pos({_stats(conf_raw[:, :3])}) "
            f"rpy({_stats(conf_raw[:, 3:6])}) "
            f"gate_gain_pos={_masked_mean(gate_mean, positive_gain):.3f} "
            f"gate_gain_neg={_masked_mean(gate_mean, ~positive_gain):.3f}",
            flush=True,
        )
        print(
            "[FrontRES gate debug] "
            f"corr(gate,damage)={_corr(gate_mean, noisy_err_norm):+.3f} "
            f"corr(gate,gain)={_corr(gate_mean, restore_gain):+.3f} "
            f"corr(gate,|proposal|)={_corr(gate_mean, proposal6.norm(dim=-1)):+.3f} "
            f"|target|={_vec6(target6.abs())} "
            f"|proposal|={_vec6(proposal6.abs())} "
            f"|applied|={_vec6(applied6.abs())}",
            flush=True,
        )
        if inertial_debug is not None:
            print(
                "[FrontRES inertial debug] "
                f"D noisy/proj/cand/clean="
                f"{float(inertial_debug['d_noisy'].mean()):.4f}/"
                f"{float(inertial_debug['d_proj'].mean()):.4f}/"
                f"{float(inertial_debug['d_cand'].mean()):.4f}/"
                f"{float(inertial_debug['d_clean'].mean()):.4f} "
                f"C noisy/proj/cand/clean="
                f"{float(inertial_debug['c_noisy'].mean()):+.3f}/"
                f"{float(inertial_debug['c_proj'].mean()):+.3f}/"
                f"{float(inertial_debug['c_cand'].mean()):+.3f}/"
                f"{float(inertial_debug['c_clean'].mean()):+.3f} "
                f"Crp noisy/proj/cand/clean="
                f"{float(inertial_debug['crp_noisy'].mean()):+.3f}/"
                f"{float(inertial_debug['crp_proj'].mean()):+.3f}/"
                f"{float(inertial_debug['crp_cand'].mean()):+.3f}/"
                f"{float(inertial_debug['crp_clean'].mean()):+.3f}",
                flush=True,
            )
            print(
                "[FrontRES inertial debug] mixed "
                f"D p0r1/pProjr1="
                f"{float(inertial_debug['d_p0r1'].mean()):.4f}/"
                f"{float(inertial_debug['d_ppr1'].mean()):.4f} "
                f"C p0r1/pProjr1="
                f"{float(inertial_debug['c_p0r1'].mean()):+.3f}/"
                f"{float(inertial_debug['c_ppr1'].mean()):+.3f} "
                f"Crp p0r1/pProjr1="
                f"{float(inertial_debug['crp_p0r1'].mean()):+.3f}/"
                f"{float(inertial_debug['crp_ppr1'].mean()):+.3f}",
                flush=True,
            )
            print(
                "[FrontRES inertial debug] "
                f"angle_inv={float(inertial_debug['angle_inv'].mean()):.3f} "
                f"anti_proj={float(inertial_debug['anti_inertia'].mean()):.3f} "
                f"anti_cand={float(inertial_debug['anti_cand'].mean()):.3f} "
                f"anti_clean={float(inertial_debug['anti_clean'].mean()):.3f} "
                f"corr(gate,inert_gain)={_corr(gate_mean, inertial_debug['inertial_gain']):+.3f} "
                f"sample_d={float(inertial_debug['d_noisy'][sample_idx]):.4f}->"
                f"{float(inertial_debug['d_proj'][sample_idx]):.4f} "
                f"sample_c={float(inertial_debug['c_noisy'][sample_idx]):+.3f}->"
                f"{float(inertial_debug['c_proj'][sample_idx]):+.3f}",
                flush=True,
            )
    print(
        "[FrontRES restore debug] sample vectors "
        f"clean_from_raw={_vec(clean_from_raw, sample_idx)} "
        f"target={_vec(target, sample_idx)} "
        f"pred={_vec(pred, sample_idx)} "
        f"applied={_vec(applied, sample_idx)} "
        f"written={_vec(written, sample_idx)} "
        f"residual={_vec(corrected_err, sample_idx)}",
        flush=True,
    )
