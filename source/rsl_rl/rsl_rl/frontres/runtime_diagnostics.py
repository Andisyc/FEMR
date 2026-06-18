# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES runtime diagnostic printing helpers."""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_inv, quat_mul

from rsl_rl.frontres.frontres_executability import (
    quat_to_rotvec_wxyz as _quat_to_rotvec_wxyz,
    rotvec_to_quat_wxyz as _rotvec_to_quat_wxyz,
)


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
