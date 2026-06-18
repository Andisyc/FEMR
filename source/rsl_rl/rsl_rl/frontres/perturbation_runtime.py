# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES perturbation runtime applicators."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


def snapshot_frontres_perturbation_target(runner: Any, *, is_frontres: bool) -> Any | None:
    """Snapshot base motion perturbation magnitudes before DR scaling mutates them."""
    if not is_frontres:
        return None
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env_raw, "cfg") and hasattr(env_raw.cfg, "motion_perturbations")):
        return None
    pt = env_raw.cfg.motion_perturbations
    return SimpleNamespace(
        float_prob=float(pt.float_prob),
        float_ratio=float(pt.float_ratio),
        sink_prob=float(pt.sink_prob),
        sink_ratio=float(pt.sink_ratio),
        foot_slip_prob=float(pt.foot_slip_prob),
        foot_slip_ratio=float(pt.foot_slip_ratio),
        lateral_drift_prob=float(getattr(pt, "lateral_drift_prob", 0.0)),
        lateral_drift_std=float(getattr(pt, "lateral_drift_std", 0.0)),
        root_tilt_prob=float(getattr(pt, "root_tilt_prob", 0.0)),
        root_tilt_max_rad=float(getattr(pt, "root_tilt_max_rad", 0.0)),
        joint_noise_prob=float(getattr(pt, "joint_noise_prob", 0.0)),
        joint_noise_std=float(getattr(pt, "joint_noise_std", 0.0)),
        iid_prob_z=float(getattr(pt, "iid_prob_z", 0.0)),
        iid_std_z=float(getattr(pt, "iid_std_z", 0.0)),
        iid_prob_xy=float(getattr(pt, "iid_prob_xy", 0.0)),
        iid_std_xy=float(getattr(pt, "iid_std_xy", 0.0)),
        iid_prob_rp=float(getattr(pt, "iid_prob_rp", 0.0)),
        iid_std_rp=float(getattr(pt, "iid_std_rp", 0.0)),
        iid_prob_ya=float(getattr(pt, "iid_prob_ya", 0.0)),
        iid_std_ya=float(getattr(pt, "iid_std_ya", 0.0)),
        local_root_artifact_prob=float(getattr(pt, "local_root_artifact_prob", 0.0)),
        local_root_artifact_xy_std=float(getattr(pt, "local_root_artifact_xy_std", 0.0)),
        local_root_artifact_yaw_std=float(getattr(pt, "local_root_artifact_yaw_std", 0.0)),
    )


def apply_frontres_dr_scale(
    runner: Any,
    *,
    scale: float,
    is_frontres: bool,
    perturb_target: Any | None,
) -> None:
    """Write the current scalar DR scale and active family gates to the perturber."""
    if not (is_frontres and perturb_target is not None):
        return
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env_raw, "command_manager") and "motion" in env_raw.command_manager._terms):
        return
    motion_command = env_raw.command_manager._terms["motion"]
    if not hasattr(motion_command, "perturber"):
        return

    def pt(attr: str, default: float = 0.0) -> float:
        return getattr(perturb_target, attr, default)

    modes = set(getattr(
        runner,
        "_frontres_curriculum_active_modes",
        ("planar", "yaw", "global_z", "local_rp"),
    ))
    planar = "planar" in modes
    yaw = "yaw" in modes
    global_z = "global_z" in modes
    local_rp = "local_rp" in modes

    motion_command.perturber.cfg.float_prob = pt("float_prob") if global_z else 0.0
    motion_command.perturber.cfg.float_ratio = pt("float_ratio") * scale
    motion_command.perturber.cfg.sink_prob = pt("sink_prob") if global_z else 0.0
    motion_command.perturber.cfg.sink_ratio = pt("sink_ratio") * scale
    motion_command.perturber.cfg.foot_slip_prob = pt("foot_slip_prob") if planar else 0.0
    motion_command.perturber.cfg.foot_slip_ratio = pt("foot_slip_ratio") * scale
    motion_command.perturber.cfg.lateral_drift_prob = pt("lateral_drift_prob") if planar else 0.0
    motion_command.perturber.cfg.lateral_drift_std = pt("lateral_drift_std") * scale
    motion_command.perturber.cfg.root_tilt_prob = pt("root_tilt_prob") if local_rp else 0.0
    motion_command.perturber.cfg.root_tilt_max_rad = pt("root_tilt_max_rad") * scale
    motion_command.perturber.cfg.joint_noise_prob = pt("joint_noise_prob")
    motion_command.perturber.cfg.joint_noise_std = pt("joint_noise_std") * scale
    motion_command.perturber.cfg.iid_prob_z = pt("iid_prob_z") if global_z else 0.0
    motion_command.perturber.cfg.iid_std_z = pt("iid_std_z") * scale
    motion_command.perturber.cfg.iid_prob_xy = pt("iid_prob_xy") if planar else 0.0
    motion_command.perturber.cfg.iid_std_xy = pt("iid_std_xy") * scale
    motion_command.perturber.cfg.iid_prob_rp = pt("iid_prob_rp") if local_rp else 0.0
    motion_command.perturber.cfg.iid_std_rp = pt("iid_std_rp") * scale
    motion_command.perturber.cfg.iid_prob_ya = pt("iid_prob_ya") if yaw else 0.0
    motion_command.perturber.cfg.iid_std_ya = pt("iid_std_ya") * scale
    motion_command.perturber.cfg.local_root_artifact_prob = pt("local_root_artifact_prob") if (planar or yaw) else 0.0
    # Local artifact magnitudes are multiplied by perturber._dr_scale at burst sampling time.
    motion_command.perturber.cfg.local_root_artifact_xy_std = pt("local_root_artifact_xy_std") if planar else 0.0
    motion_command.perturber.cfg.local_root_artifact_yaw_std = pt("local_root_artifact_yaw_std") if yaw else 0.0
    motion_command.perturber._dr_scale = float(scale)
    if hasattr(motion_command.perturber, "set_dr_scale_env"):
        motion_command.perturber.set_dr_scale_env(None)


def apply_frontres_dr_scale_env(
    runner: Any,
    *,
    scale_vec: torch.Tensor,
    is_frontres: bool,
    perturb_target: Any | None,
) -> None:
    """Write unscaled base magnitudes and per-env DR scale vector to the perturber."""
    if not (is_frontres and perturb_target is not None):
        return
    apply_frontres_dr_scale(
        runner,
        scale=1.0,
        is_frontres=is_frontres,
        perturb_target=perturb_target,
    )
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env_raw, "command_manager") and "motion" in env_raw.command_manager._terms):
        return
    motion_command = env_raw.command_manager._terms["motion"]
    if hasattr(motion_command, "perturber") and hasattr(motion_command.perturber, "set_dr_scale_env"):
        motion_command.perturber.set_dr_scale_env(scale_vec)


def apply_frontres_family_env_masks(
    runner: Any,
    *,
    groups: list[tuple[str, ...]],
    n_train: int,
    n_candidate: int,
    n_base: int,
) -> None:
    """Write per-family perturbation masks for projected/candidate/noisy branches."""
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env_raw, "command_manager") and "motion" in env_raw.command_manager._terms):
        return
    motion_command = env_raw.command_manager._terms["motion"]
    if not (
        hasattr(motion_command, "perturber")
        and hasattr(motion_command.perturber, "set_family_env_masks")
    ):
        return

    family_masks = {
        "planar": torch.zeros(runner.env.num_envs, dtype=torch.bool, device=runner.device),
        "yaw": torch.zeros(runner.env.num_envs, dtype=torch.bool, device=runner.device),
        "global_z": torch.zeros(runner.env.num_envs, dtype=torch.bool, device=runner.device),
        "local_rp": torch.zeros(runner.env.num_envs, dtype=torch.bool, device=runner.device),
    }
    candidate_start = int(n_train)
    base_start = int(n_train) + int(n_candidate)
    for env_i, group in enumerate(groups[: int(n_train)]):
        for mode in group:
            if mode in family_masks:
                family_masks[mode][env_i] = True
                if env_i < int(n_candidate):
                    family_masks[mode][candidate_start + env_i] = True
                if env_i < int(n_base):
                    family_masks[mode][base_start + env_i] = True
    # Clean-GMT envs intentionally remain all False; baseline masking keeps them clean.
    motion_command.perturber.set_family_env_masks(family_masks)
