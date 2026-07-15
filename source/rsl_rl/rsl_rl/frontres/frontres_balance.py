# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balance evidence adapter for the active Segment Gain path."""

from __future__ import annotations

from typing import Any

import torch


def _frontres_branch_balance_margin(
    runner: Any,
    cmd: Any,
    *,
    start: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return min(root, capture) balance margin for one quartet branch."""
    if count <= 0:
        return torch.empty(0, device=device)

    from whole_body_tracking.tasks.tracking.mdp.balance import frontres_balance_context_from_feet

    body_names = list(getattr(cmd.cfg, "body_names", []))
    foot_names = runner.cfg.get(
        "frontres_balance_foot_body_names",
        runner.cfg.get(
            "frontres_exec_foot_body_names",
            ["left_ankle_roll_link", "right_ankle_roll_link"],
        ),
    )
    foot_ids = [i for i, name in enumerate(body_names) if name in set(foot_names)]
    if len(foot_ids) != 2:
        raise RuntimeError(
            "FrontRES balance evidence requires exactly two foot body names in "
            f"command.cfg.body_names; got {foot_names!r} -> ids={foot_ids!r}."
        )

    end = start + count
    robot_data = cmd.robot.data
    root_xy = robot_data.root_pos_w[start:end, :2]
    root_vel_w = getattr(robot_data, "root_lin_vel_w", None)
    if root_vel_w is None:
        root_vel_w = getattr(cmd, "robot_anchor_vel_w", None)
    if root_vel_w is None:
        raise RuntimeError("FrontRES balance evidence requires root_lin_vel_w or robot_anchor_vel_w.")
    feet_pos = cmd.robot_body_pos_w[start:end, foot_ids, :]
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    env_origin_z = None
    if hasattr(env, "scene") and hasattr(env.scene, "env_origins"):
        env_origin_z = env.scene.env_origins[start:end, 2]
    context = frontres_balance_context_from_feet(
        root_xy,
        root_vel_w[start:end, :2],
        feet_pos[..., :2],
        feet_pos[..., 2],
        contact_height=float(runner.cfg.get("frontres_balance_contact_height", 0.08)),
        foot_radius=float(runner.cfg.get("frontres_balance_foot_radius", 0.04)),
        capture_height=float(runner.cfg.get("frontres_balance_capture_height", 0.8)),
        env_origin_z=env_origin_z,
    )
    return torch.minimum(context[:, -3], context[:, -2]).to(device)
