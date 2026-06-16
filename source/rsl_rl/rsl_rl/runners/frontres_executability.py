"""FrontRES executable-score and feasible-oracle ownership.

The runner should orchestrate rollout phases. This module owns the GMT
executability measurement used by FrontRES reward, labels, and diagnostics.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_inv, quat_mul, yaw_quat


def quat_to_rotvec_wxyz(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map wxyz unit quaternions to shortest-path rotation vectors."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    xyz_norm = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, q[..., :1].clamp(min=eps))
    scale = torch.where(xyz_norm > eps, angle / xyz_norm.clamp(min=eps), 2.0 * torch.ones_like(xyz_norm))
    return xyz * scale


def rotvec_to_quat_wxyz(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map local rotation vectors to wxyz unit quaternions."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    xyz_scale = torch.where(
        angle > eps,
        torch.sin(half) / angle.clamp(min=eps),
        0.5 * torch.ones_like(angle),
    )
    quat = torch.cat([torch.cos(half), rotvec * xyz_scale], dim=-1)
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=eps)


class FrontRESExecutabilityScorer:
    """GMT executability scoring and feasible in-cone oracle for FrontRES."""

    def __init__(self, cfg: dict[str, Any], alg: Any, device: str | torch.device):
        self.cfg = cfg
        self.alg = alg
        self.device = device

    def exec_score(self, command, return_components: bool = False):
        """Continuous executability score for the frozen GMT tracker."""
        n_envs = command.anchor_pos_w.shape[0]
        dtype = command.anchor_pos_w.dtype

        def cfg_float(name: str, default: float) -> float:
            return float(self.cfg.get(name, default))

        def quat_to_yaw_wxyz(q: torch.Tensor) -> torch.Tensor:
            _, _, yaw = euler_xyz_from_quat(q)
            return yaw

        def wrap_pi(a: torch.Tensor) -> torch.Tensor:
            return torch.atan2(torch.sin(a), torch.cos(a))

        anchor_xy_th = cfg_float("frontres_exec_anchor_xy_threshold", 0.35)
        anchor_yaw_th = cfg_float("frontres_exec_anchor_yaw_threshold", 0.45)
        anchor_xy_vel_std = cfg_float("frontres_exec_anchor_xy_vel_std", 1.0)
        anchor_yaw_rate_std = cfg_float("frontres_exec_anchor_yaw_rate_std", 1.0)

        anchor_xy_err = torch.norm(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
        anchor_xy_score = (1.0 - anchor_xy_err / max(anchor_xy_th, 1e-6)).clamp(-1.0, 1.0)
        anchor_yaw_err = wrap_pi(quat_to_yaw_wxyz(command.anchor_quat_w) - quat_to_yaw_wxyz(command.robot_anchor_quat_w)).abs()
        anchor_yaw_score = (1.0 - anchor_yaw_err / max(anchor_yaw_th, 1e-6)).clamp(-1.0, 1.0)
        anchor_xy_vel_err = torch.square(
            command.anchor_lin_vel_w[:, :2] - command.robot_anchor_lin_vel_w[:, :2]
        ).sum(dim=-1)
        anchor_yaw_rate_err = torch.square(command.anchor_ang_vel_w[:, 2] - command.robot_anchor_ang_vel_w[:, 2])
        anchor_xy_vel_score = torch.exp(
            (-anchor_xy_vel_err / max(anchor_xy_vel_std * anchor_xy_vel_std, 1e-6)).clamp(min=-50.0)
        )
        anchor_yaw_rate_score = torch.exp(
            (-anchor_yaw_rate_err / max(anchor_yaw_rate_std * anchor_yaw_rate_std, 1e-6)).clamp(min=-50.0)
        )

        body_names = list(getattr(command.cfg, "body_names", []))
        foot_names = self.cfg.get(
            "frontres_exec_foot_body_names",
            ["left_ankle_roll_link", "right_ankle_roll_link"],
        )
        foot_idx = [i for i, name in enumerate(body_names) if name in foot_names]
        if len(foot_idx) == 0:
            foot_idx = list(range(command.body_pos_relative_w.shape[1]))
        foot_xy_err = torch.norm(
            command.body_pos_relative_w[:, foot_idx, :2] - command.robot_body_pos_w[:, foot_idx, :2],
            dim=-1,
        )
        foot_z_err = (
            command.body_pos_relative_w[:, foot_idx, 2] - command.robot_body_pos_w[:, foot_idx, 2]
        ).abs()
        foot_z_th = cfg_float("frontres_exec_foot_phase_z_threshold", 0.12)
        foot_gate_temp = cfg_float("frontres_exec_foot_phase_gate_temp", 0.03)
        foot_xy_th = cfg_float("frontres_exec_foot_phase_xy_threshold", 0.25)
        foot_gate = torch.sigmoid((foot_z_th - foot_z_err) / max(foot_gate_temp, 1e-6))
        foot_phase_score_each = (1.0 - foot_xy_err / max(foot_xy_th, 1e-6)).clamp(-1.0, 1.0)
        foot_gate_den = foot_gate.sum(dim=-1).clamp(min=1e-6)
        foot_phase_score = (foot_gate * foot_phase_score_each).sum(dim=-1) / foot_gate_den

        w_xy = cfg_float("frontres_exec_anchor_xy_weight", 1.0)
        w_yaw = cfg_float("frontres_exec_anchor_yaw_weight", 1.0)
        w_xy_vel = cfg_float("frontres_exec_anchor_xy_vel_weight", 0.5)
        w_yaw_rate = cfg_float("frontres_exec_anchor_yaw_rate_weight", 0.5)
        w_foot_phase = cfg_float("frontres_exec_foot_phase_weight", 0.5)
        w_xy_sum = max(w_xy + w_xy_vel + w_foot_phase, 1e-6)
        xy_score = (
            w_xy * anchor_xy_score
            + w_xy_vel * anchor_xy_vel_score
            + w_foot_phase * foot_phase_score
        ) / w_xy_sum
        w_yaw_sum = max(w_yaw + w_yaw_rate, 1e-6)
        yaw_score = (w_yaw * anchor_yaw_score + w_yaw_rate * anchor_yaw_rate_score) / w_yaw_sum
        planar_score = 0.5 * (xy_score + yaw_score)

        anchor_z_th = cfg_float("frontres_exec_anchor_z_threshold", 0.25)
        anchor_ori_th = cfg_float("frontres_exec_anchor_ori_threshold", 0.20)
        ee_z_th = cfg_float("frontres_exec_ee_z_threshold", 0.25)
        anchor_z_err = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]).abs()
        anchor_z_score = (1.0 - anchor_z_err / max(anchor_z_th, 1e-6)).clamp(-1.0, 1.0)

        gravity = getattr(command.robot.data, "GRAVITY_VEC_W", None)
        if gravity is None:
            gravity = torch.zeros(n_envs, 3, device=self.device, dtype=dtype)
            gravity[:, 2] = -1.0
        rp_error_rotvec = quat_to_rotvec_wxyz(
            quat_mul(quat_inv(command.robot_anchor_quat_w), command.anchor_quat_w)
        )
        anchor_rp_err = torch.norm(rp_error_rotvec[:, :2], dim=-1)
        anchor_rp_score = (1.0 - anchor_rp_err / max(anchor_ori_th, 1e-6)).clamp(-1.0, 1.0)

        ee_names = self.cfg.get(
            "frontres_exec_ee_body_names",
            [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        )
        ee_idx = [i for i, name in enumerate(body_names) if name in ee_names]
        if len(ee_idx) == 0:
            ee_idx = list(range(command.body_pos_relative_w.shape[1]))
        ee_z_err = (
            command.body_pos_relative_w[:, ee_idx, 2] - command.robot_body_pos_w[:, ee_idx, 2]
        ).abs().amax(dim=-1)
        ee_z_score = (1.0 - ee_z_err / max(ee_z_th, 1e-6)).clamp(-1.0, 1.0)

        w_z = cfg_float("frontres_exec_anchor_z_weight", 1.0)
        w_ori = cfg_float("frontres_exec_anchor_ori_weight", 1.0)
        w_ee = cfg_float("frontres_exec_ee_z_weight", 1.0)
        w_z_sum = max(w_z + w_ee, 1e-6)
        z_score = (w_z * anchor_z_score + w_ee * ee_z_score) / w_z_sum
        rp_score = anchor_rp_score
        w_stab_sum = max(w_z + w_ori + w_ee, 1e-6)
        vertical_score = (w_z * anchor_z_score + w_ori * rp_score + w_ee * ee_z_score) / w_stab_sum

        vel_body_names = self.cfg.get("frontres_exec_velocity_body_names", None)
        if vel_body_names is None:
            vel_idx = list(range(command.body_lin_vel_w.shape[1]))
        else:
            vel_idx = [i for i, name in enumerate(body_names) if name in vel_body_names]
            if len(vel_idx) == 0:
                vel_idx = list(range(command.body_lin_vel_w.shape[1]))
        lin_std = cfg_float("frontres_exec_body_lin_vel_std", 1.0)
        ang_std = cfg_float("frontres_exec_body_ang_vel_std", 3.14)
        anchor_lin_std = cfg_float("frontres_exec_anchor_lin_vel_std", 1.0)
        lin_err = torch.square(command.body_lin_vel_w[:, vel_idx] - command.robot_body_lin_vel_w[:, vel_idx]).sum(
            dim=-1
        ).mean(dim=-1)
        ang_err = torch.square(command.body_ang_vel_w[:, vel_idx] - command.robot_body_ang_vel_w[:, vel_idx]).sum(
            dim=-1
        ).mean(dim=-1)
        anchor_lin_err = torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w).sum(dim=-1)
        lin_score = torch.exp((-lin_err / max(lin_std * lin_std, 1e-6)).clamp(min=-50.0))
        ang_score = torch.exp((-ang_err / max(ang_std * ang_std, 1e-6)).clamp(min=-50.0))
        anchor_lin_score = torch.exp((-anchor_lin_err / max(anchor_lin_std * anchor_lin_std, 1e-6)).clamp(min=-50.0))
        task_score = (lin_score + ang_score + anchor_lin_score) / 3.0

        planar_weight = cfg_float("frontres_exec_planar_weight", 1.0)
        vertical_weight = cfg_float("frontres_exec_vertical_weight", 0.25)
        task_weight = cfg_float("frontres_exec_task_weight", 0.25)
        score = planar_weight * planar_score + vertical_weight * vertical_score + task_weight * task_score
        score = torch.nan_to_num(score, nan=-1.0, posinf=1.0, neginf=-1.0)
        if return_components:
            return score, {
                "planar": torch.nan_to_num(planar_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "vertical": torch.nan_to_num(vertical_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "xy": torch.nan_to_num(xy_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "yaw": torch.nan_to_num(yaw_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "z": torch.nan_to_num(z_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "rp": torch.nan_to_num(rp_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "task": torch.nan_to_num(task_score, nan=0.0, posinf=1.0, neginf=0.0),
            }
        return score

    def feasible_oracle_exec_score(
        self,
        command,
        start: int,
        count: int,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Executability score after the best correction allowed by the active action cone."""
        if count <= 0:
            return torch.empty(0, device=self.device)

        needed = (
            "anchor_pos_w_original",
            "anchor_pos_w_raw",
            "anchor_quat_w_original",
            "anchor_quat_w_raw",
            "_frontres_pos_correction",
            "_frontres_quat_correction",
        )
        if not all(hasattr(command, name) for name in needed):
            score = self.exec_score(command, return_components=return_components)
            if return_components:
                full_score, components = score
                return full_score[start:start + count], {
                    key: value[start:start + count] for key, value in components.items()
                }
            return score[start:start + count]

        env_slice = slice(start, start + count)
        raw_pos = command.anchor_pos_w_raw[env_slice]
        clean_pos = command.anchor_pos_w_original[env_slice]
        raw_quat = command.anchor_quat_w_raw[env_slice]
        clean_quat = command.anchor_quat_w_original[env_slice]

        max_delta_pos = float(getattr(getattr(self.alg, "policy", None), "max_delta_pos", 0.3))
        max_delta_rpy = float(getattr(getattr(self.alg, "policy", None), "max_delta_rpy", 0.1))

        active_dims = getattr(self.alg, "frontres_active_task_dims", self.cfg.get("frontres_active_task_dims", None))
        active_set = None if active_dims is None else {int(dim) for dim in active_dims}

        def _active(dim: int) -> bool:
            return active_set is None or dim in active_set

        oracle_pos = torch.zeros_like(raw_pos)
        dpos_clean = clean_pos - raw_pos
        if _active(0):
            oracle_pos[:, 0] = dpos_clean[:, 0].clamp(-max_delta_pos, max_delta_pos)
        if _active(1):
            oracle_pos[:, 1] = dpos_clean[:, 1].clamp(-max_delta_pos, max_delta_pos)
        dz_clean = clean_pos[:, 2] - raw_pos[:, 2]
        z_upper = torch.zeros_like(dz_clean)
        if hasattr(command, "jump_degree") and hasattr(command, "anchor_penetration_depth"):
            jump_degree = command.jump_degree[env_slice].to(raw_pos.device).to(raw_pos.dtype).clamp(0.0, 1.0)
            penetration = command.anchor_penetration_depth[env_slice].to(raw_pos.device).to(raw_pos.dtype)
            z_upper = (jump_degree * penetration).clamp(max=max_delta_pos)
        if _active(2):
            z_lower = torch.full_like(dz_clean, -max_delta_pos)
            oracle_pos[:, 2] = torch.minimum(torch.maximum(dz_clean, z_lower), z_upper)

        correction_quat = quat_mul(quat_inv(raw_quat), clean_quat)
        correction_rotvec = quat_to_rotvec_wxyz(correction_quat)
        oracle_rotvec = torch.zeros_like(correction_rotvec)
        for dim in (3, 4, 5):
            if _active(dim):
                axis = dim - 3
                oracle_rotvec[:, axis] = correction_rotvec[:, axis].clamp(-max_delta_rpy, max_delta_rpy)
        oracle_quat = rotvec_to_quat_wxyz(oracle_rotvec)

        saved_pos = command._frontres_pos_correction[env_slice].clone()
        saved_quat = command._frontres_quat_correction[env_slice].clone()
        saved_body_pos = None
        saved_body_quat = None
        try:
            command._frontres_pos_correction[env_slice].copy_(oracle_pos)
            command._frontres_quat_correction[env_slice].copy_(oracle_quat)
            if hasattr(command, "body_pos_relative_w") and hasattr(command, "body_quat_relative_w"):
                saved_body_pos = command.body_pos_relative_w[env_slice].clone()
                saved_body_quat = command.body_quat_relative_w[env_slice].clone()

                body_count = len(getattr(command.cfg, "body_names", []))
                if body_count > 0:
                    anchor_pos = command.anchor_pos_w[env_slice]
                    anchor_quat = command.anchor_quat_w[env_slice]
                    robot_anchor_pos = command.robot_anchor_pos_w[env_slice]
                    robot_anchor_quat = command.robot_anchor_quat_w[env_slice]
                    body_pos = command.body_pos_w[env_slice]
                    body_quat = command.body_quat_w[env_slice]

                    anchor_pos_repeat = anchor_pos[:, None, :].repeat(1, body_count, 1)
                    anchor_quat_repeat = anchor_quat[:, None, :].repeat(1, body_count, 1)
                    robot_anchor_pos_repeat = robot_anchor_pos[:, None, :].repeat(1, body_count, 1)
                    robot_anchor_quat_repeat = robot_anchor_quat[:, None, :].repeat(1, body_count, 1)

                    delta_pos = robot_anchor_pos_repeat.clone()
                    delta_pos[..., 2] = anchor_pos_repeat[..., 2]
                    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_repeat, quat_inv(anchor_quat_repeat)))
                    command.body_quat_relative_w[env_slice].copy_(quat_mul(delta_ori, body_quat))
                    command.body_pos_relative_w[env_slice].copy_(
                        delta_pos + quat_apply(delta_ori, body_pos - anchor_pos_repeat)
                    )
            if return_components:
                feasible_score_all, feasible_components_all = self.exec_score(command, return_components=True)
                feasible_score = feasible_score_all[env_slice].clone()
                feasible_components = {
                    key: value[env_slice].clone() for key, value in feasible_components_all.items()
                }
            else:
                feasible_score = self.exec_score(command)[env_slice].clone()
        finally:
            command._frontres_pos_correction[env_slice].copy_(saved_pos)
            command._frontres_quat_correction[env_slice].copy_(saved_quat)
            if saved_body_pos is not None:
                command.body_pos_relative_w[env_slice].copy_(saved_body_pos)
            if saved_body_quat is not None:
                command.body_quat_relative_w[env_slice].copy_(saved_body_quat)
        if return_components:
            return feasible_score, feasible_components
        return feasible_score

    def exec_score_for_modes(
        self,
        components: dict[str, torch.Tensor],
        start: int,
        count: int,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...] | None = None,
        active_modes: tuple[str, ...] = (),
    ) -> torch.Tensor:
        """Select executable score components that match each sample's repair cone."""
        if count <= 0:
            return torch.empty(0, device=self.device)
        if mode_groups is None:
            if not active_modes:
                active_dims = getattr(self.alg, "frontres_active_task_dims", self.cfg.get("frontres_active_task_dims", None))
                if active_dims is None:
                    active_modes = ("planar", "yaw", "global_z", "local_rp")
                else:
                    dims = {int(dim) for dim in active_dims}
                    inferred = []
                    if 0 in dims or 1 in dims:
                        inferred.append("planar")
                    if 5 in dims:
                        inferred.append("yaw")
                    if 2 in dims:
                        inferred.append("global_z")
                    if 3 in dims or 4 in dims:
                        inferred.append("local_rp")
                    active_modes = tuple(inferred) if inferred else ("planar", "yaw", "global_z", "local_rp")
            mode_groups = [tuple(active_modes)] * count

        xy = components.get("xy", components["planar"])[start:start + count]
        yaw = components.get("yaw", components["planar"])[start:start + count]
        z = components.get("z", components["vertical"])[start:start + count]
        rp = components.get("rp", components["vertical"])[start:start + count]
        task = components["task"][start:start + count]
        score = torch.zeros(count, device=xy.device, dtype=xy.dtype)
        denom = torch.zeros_like(score)

        planar_weight = float(self.cfg.get("frontres_exec_cone_planar_weight", 1.0))
        yaw_weight = float(self.cfg.get("frontres_exec_cone_yaw_weight", planar_weight))
        vertical_weight = float(self.cfg.get("frontres_exec_cone_vertical_weight", 1.0))
        rp_weight = float(self.cfg.get("frontres_exec_cone_rp_weight", vertical_weight))
        task_weight = float(self.cfg.get("frontres_exec_cone_task_weight", 0.0))
        for idx, modes in enumerate(mode_groups[:count]):
            mode_set = set(modes)
            if "planar" in mode_set:
                score[idx] += planar_weight * xy[idx]
                denom[idx] += planar_weight
            if "yaw" in mode_set:
                score[idx] += yaw_weight * yaw[idx]
                denom[idx] += yaw_weight
            if "global_z" in mode_set:
                score[idx] += vertical_weight * z[idx]
                denom[idx] += vertical_weight
            if "local_rp" in mode_set:
                score[idx] += rp_weight * rp[idx]
                denom[idx] += rp_weight
            if task_weight > 0.0:
                score[idx] += task_weight * task[idx]
                denom[idx] += task_weight
        fallback = 0.25 * (xy + yaw + z + rp)
        score = torch.where(denom > 0.0, score / denom.clamp(min=1e-6), fallback)
        return torch.nan_to_num(score, nan=-1.0, posinf=1.0, neginf=-1.0)
