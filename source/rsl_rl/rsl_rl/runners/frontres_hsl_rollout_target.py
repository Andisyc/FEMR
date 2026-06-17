# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
from isaaclab.utils.math import quat_inv, quat_mul


def build_frontres_hsl_rollout_target(
    runner: Any,
    *,
    command: Any,
    actions: torch.Tensor | None,
    dones: torch.Tensor | None,
    current_pos_correction: torch.Tensor | None,
    current_quat_correction: torch.Tensor | None,
    n_train: int,
    n_candidate: int,
    n_base: int,
    n_clean: int,
    quat_to_rotvec_wxyz: Any,
) -> None:
    """Build and write HSL supervised labels from Clean/Noisy/FEMR rollout states."""

    if not bool(runner.cfg.get("frontres_hsl_rollout_label_enabled", False)):
        return
    if (
        command is None
        or actions is None
        or current_pos_correction is None
        or current_quat_correction is None
        or n_train <= 0
        or n_base <= 0
        or n_clean <= 0
    ):
        return
    if not hasattr(command, "robot") or not hasattr(command.robot, "data"):
        return
    data = command.robot.data
    if not (hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w")):
        return

    n = min(int(n_train), int(n_base), int(n_clean), actions.shape[0])
    if n <= 0:
        return

    base_start = int(n_train + max(0, int(n_candidate)))
    clean_start = int(base_start + n_base)
    device = runner.device
    dtype = actions.dtype
    valid_rollout = torch.ones(n, device=device, dtype=dtype)
    if dones is not None:
        done_flat = dones.to(device).view(-1)
        if done_flat.numel() >= clean_start + n:
            done_any = (
                done_flat[:n].bool()
                | done_flat[base_start:base_start + n].bool()
                | done_flat[clean_start:clean_start + n].bool()
            )
            valid_rollout = (~done_any).to(dtype)

    root_pos = data.root_pos_w.to(device=device, dtype=dtype)
    root_quat = data.root_quat_w.to(device=device, dtype=dtype)
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    origins = getattr(getattr(env_raw, "scene", None), "env_origins", None)
    if origins is None:
        origins = torch.zeros_like(root_pos)
    else:
        origins = origins.to(device=device, dtype=dtype)

    fr_pos = root_pos[:n] - origins[:n]
    noisy_pos = root_pos[base_start:base_start + n] - origins[base_start:base_start + n]
    clean_pos = root_pos[clean_start:clean_start + n] - origins[clean_start:clean_start + n]
    fr_quat = root_quat[:n]
    noisy_quat = root_quat[base_start:base_start + n]
    clean_quat = root_quat[clean_start:clean_start + n]

    front_to_clean_r = quat_to_rotvec_wxyz(quat_mul(quat_inv(fr_quat), clean_quat))
    noisy_to_clean_r = quat_to_rotvec_wxyz(quat_mul(quat_inv(noisy_quat), clean_quat))
    sim_residual = torch.cat([clean_pos - fr_pos, front_to_clean_r], dim=-1)

    current = torch.zeros(n, 6, device=device, dtype=dtype)
    current[:, :3] = current_pos_correction[:n].to(device=device, dtype=dtype)
    current[:, 3:6] = quat_to_rotvec_wxyz(
        current_quat_correction[:n].to(device=device, dtype=dtype)
    )

    eta = float(runner.cfg.get("frontres_hsl_rollout_eta", 1.0))
    label = current + eta * sim_residual
    label = runner._frontres_action_cone.project_task_target(command, label)
    if bool(runner.cfg.get("frontres_per_mode_supervised_mask", True)):
        mode_groups = list(getattr(
            runner,
            "_frontres_curriculum_env_mode_groups",
            [tuple(getattr(runner, "_frontres_curriculum_active_modes", ()))] * n,
        ))[:n]
        label = runner._frontres_action_cone.apply_per_mode_supervised_mask(label, mode_groups, n)

    rot_scale = float(runner.cfg.get("frontres_hsl_rot_error_scale", 0.25))
    noisy_err = (
        (noisy_pos - clean_pos).norm(dim=-1)
        + rot_scale * noisy_to_clean_r.norm(dim=-1)
    )
    front_err = (
        (fr_pos - clean_pos).norm(dim=-1)
        + rot_scale * front_to_clean_r.norm(dim=-1)
    )
    safe_th = float(runner.cfg.get("frontres_hsl_safe_threshold", 0.03))
    broken_th = float(runner.cfg.get("frontres_hsl_broken_threshold", 0.35))
    safe_tau = max(float(runner.cfg.get("frontres_hsl_safe_temperature", 0.01)), 1e-6)
    broken_tau = max(float(runner.cfg.get("frontres_hsl_broken_temperature", 0.05)), 1e-6)
    harm_tau = max(float(runner.cfg.get("frontres_hsl_harm_temperature", 0.02)), 1e-6)
    safe_w = torch.sigmoid((safe_th - noisy_err) / safe_tau)
    broken_w = torch.sigmoid((noisy_err - broken_th) / broken_tau)
    repair_w = torch.sigmoid((noisy_err - safe_th) / safe_tau) * torch.sigmoid(
        (broken_th - noisy_err) / broken_tau
    )
    harm_w = torch.sigmoid((front_err - noisy_err) / harm_tau)

    safe_scale = float(runner.cfg.get("frontres_hsl_safe_noop_weight", 1.0))
    broken_scale = float(runner.cfg.get("frontres_hsl_broken_noop_weight", 1.0))
    harm_scale = float(runner.cfg.get("frontres_hsl_harm_noop_weight", 2.0))
    noop_w = safe_scale * safe_w + broken_scale * broken_w + harm_scale * harm_w
    denom = (repair_w + noop_w).clamp(min=1e-6)

    objective = str(getattr(runner.alg, "frontres_training_objective", "")).lower()
    task_conf_dim = int(getattr(getattr(runner.alg, "policy", None), "task_conf_dim", 2))
    acceptance_hybrid = objective == "hsl_hybrid" and task_conf_dim == 6

    target_full = torch.zeros(runner.env.num_envs, 6, device=device, dtype=dtype)
    weight_full = torch.zeros(runner.env.num_envs, 1, device=device, dtype=dtype)
    harm_weight_full = torch.zeros(runner.env.num_envs, 1, device=device, dtype=dtype)
    max_weight = float(runner.cfg.get("frontres_hsl_max_sample_weight", 4.0))
    if acceptance_hybrid:
        target_full[:n] = label
        weight_full[:n, 0] = repair_w.clamp(max=max_weight) * valid_rollout
        harm_weight_full[:n, 0] = noop_w.clamp(max=max_weight) * valid_rollout
    else:
        mixed_label = label * (repair_w / denom).unsqueeze(-1)
        target_full[:n] = mixed_label
        weight_full[:n, 0] = denom.clamp(max=max_weight) * valid_rollout
        harm_weight_full[:n, 0] = harm_w * valid_rollout
    runner.alg.transition.supervised_target = target_full
    runner.alg.transition.supervised_weight = weight_full
    runner.alg.transition.supervised_harm_weight = harm_weight_full
