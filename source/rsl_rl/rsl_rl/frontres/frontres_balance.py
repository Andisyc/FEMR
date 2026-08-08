# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balance evidence adapter for the active Segment Gain path."""

from __future__ import annotations

import math
import re
from typing import Any

import torch


FRONTRES_ZMP_ESTIMATOR_ID = "contact-wrench-zmp-v1"
FRONTRES_SUPPORT_ENVELOPE_ID = "clean-foot-pose-oriented-box-v1"


def ensure_frontres_raw_contact_view(sensor: Any, *, num_envs: int) -> Any:
    """Install the raw-capable PhysX view used by FrontRES Physics evidence."""

    existing = getattr(sensor, "contact_physx_view", None)
    if int(getattr(sensor, "_frontres_raw_contact_capacity", 0)) > 0:
        return existing
    cfg = getattr(sensor, "cfg", None)
    if int(getattr(cfg, "max_contact_data_count", 0)) > 0:
        return existing
    physics_view = getattr(sensor, "_physics_sim_view", None)
    create_view = getattr(physics_view, "create_rigid_contact_view", None)
    body_names = getattr(sensor, "body_names", None)
    prim_path = getattr(cfg, "prim_path", None)
    filter_expr = getattr(cfg, "filter_prim_paths_expr", None)
    if not callable(create_view) or not isinstance(body_names, (list, tuple)) or not body_names:
        return existing
    if not isinstance(prim_path, str) or not isinstance(filter_expr, (list, tuple)) or not filter_expr:
        return existing

    parent = prim_path.rsplit("/", 1)[0]
    body_regex = r"(" + "|".join(re.escape(str(name)) for name in body_names) + r")"
    body_glob = f"{parent}/{body_regex}".replace(".*", "*")
    filter_glob = [str(expr).replace(".*", "*") for expr in filter_expr]
    raw_contacts_per_foot_env = 256
    capacity = max(raw_contacts_per_foot_env, int(num_envs) * raw_contacts_per_foot_env)
    raw_view = create_view(
        body_glob,
        filter_patterns=filter_glob,
        max_contact_data_count=capacity,
    )
    if int(getattr(raw_view, "count", int(num_envs))) != int(getattr(existing, "count", int(num_envs))):
        raise RuntimeError("raw contact view changed the ContactSensor body/env identity")
    if int(getattr(raw_view, "filter_count", len(filter_glob))) != int(
        getattr(existing, "filter_count", len(filter_glob))
    ):
        raise RuntimeError("raw contact view changed the ContactSensor filter identity")
    sensor._contact_physx_view = raw_view
    sensor._frontres_raw_contact_capacity = capacity
    return raw_view


def prepare_frontres_raw_contact_views(runner: Any) -> None:
    """Install both raw views before reset/step can create scored evidence."""

    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    scene = getattr(env, "scene", None)
    if scene is None:
        raise RuntimeError("v015 Physics preparation requires the formal IsaacLab scene")
    num_envs = int(getattr(env, "num_envs", 0))
    if num_envs <= 0:
        raise RuntimeError("v015 Physics preparation requires a positive env count")
    for name in ("frontres_left_foot_contacts", "frontres_right_foot_contacts"):
        try:
            sensor = scene[name]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"v015 Physics preparation is missing scene sensor {name}") from exc
        view = ensure_frontres_raw_contact_view(sensor, num_envs=num_envs)
        if int(getattr(sensor, "_frontres_raw_contact_capacity", 0)) <= 0:
            raise RuntimeError(f"v015 Physics preparation could not provision raw contact capacity for {name}")
        if view is not getattr(sensor, "_contact_physx_view", None):
            raise RuntimeError(f"v015 Physics preparation did not install the authoritative view for {name}")


def read_frontres_raw_filtered_contact_rows(
    sensor: Any,
    *,
    num_envs: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Unpack one filtered foot sensor into row-aligned raw contacts."""

    view = getattr(sensor, "contact_physx_view", None)
    if int(getattr(sensor, "_frontres_raw_contact_capacity", 0)) <= 0:
        raise RuntimeError("contact-wrench ZMP requires a raw view installed before the scored physics step")
    get_contact_data = getattr(view, "get_contact_data", None)
    if not callable(get_contact_data):
        raise RuntimeError("contact-wrench ZMP requires ContactSensor.contact_physx_view.get_contact_data")
    dt = float(getattr(sensor, "_sim_physics_dt", 0.0))
    if dt <= 0.0:
        raise RuntimeError("contact-wrench ZMP requires a positive ContactSensor physics dt")
    payload = get_contact_data(dt=dt)
    if not isinstance(payload, tuple) or len(payload) != 6:
        raise RuntimeError("unexpected IsaacLab raw contact-data payload")
    normal_force, points_w, normals_w, _distance, counts, starts = payload
    tensors = (normal_force, points_w, normals_w, counts, starts)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise RuntimeError("raw contact-data payload must contain tensors")
    counts = counts.to(device=device, dtype=torch.long).reshape(-1)
    starts = starts.to(device=device, dtype=torch.long).reshape(-1)
    if int(counts.numel()) != int(num_envs) or int(starts.numel()) != int(num_envs):
        raise RuntimeError("each v015 foot sensor must resolve exactly one body and one ground filter per env")
    capacity = int(getattr(sensor, "_frontres_raw_contact_capacity", 0))
    if capacity > 0 and int(counts.sum().item()) >= capacity:
        raise RuntimeError("contact-wrench ZMP raw contact buffer reached capacity; evidence may be truncated")
    max_contacts = max(1, int(counts.max().item()) if int(counts.numel()) else 0)
    points = torch.zeros(num_envs, 1, max_contacts, 3, device=device, dtype=torch.float32)
    normals = torch.zeros_like(points)
    forces = torch.zeros(num_envs, 1, max_contacts, device=device, dtype=torch.float32)
    valid = torch.zeros(num_envs, 1, max_contacts, device=device, dtype=torch.bool)
    normal_force = normal_force.to(device=device, dtype=torch.float32).reshape(-1)
    points_w = points_w.to(device=device, dtype=torch.float32).reshape(-1, 3)
    normals_w = normals_w.to(device=device, dtype=torch.float32).reshape(-1, 3)
    for env_id in range(num_envs):
        count = int(counts[env_id].item())
        start = int(starts[env_id].item())
        if count <= 0:
            continue
        stop = start + count
        if stop > int(normal_force.numel()) or stop > int(points_w.shape[0]) or stop > int(normals_w.shape[0]):
            raise RuntimeError("raw contact-data count/start exceeds the PhysX contact buffer")
        forces[env_id, 0, :count] = normal_force[start:stop].abs()
        points[env_id, 0, :count] = points_w[start:stop]
        normals[env_id, 0, :count] = normals_w[start:stop]
        valid[env_id, 0, :count] = True
    return points, forces, normals, valid


def pad_frontres_raw_contact_slots(
    raw: tuple[torch.Tensor, ...],
    *,
    contact_slots: int,
) -> tuple[torch.Tensor, ...]:
    """Right-pad one foot so both contact buffers share a contact axis."""

    points, forces, normals, valid = raw
    current = int(points.shape[2])
    if current > int(contact_slots) or int(contact_slots) <= 0:
        raise RuntimeError("raw contact-slot padding requires target C >= current C > 0")
    if current == int(contact_slots):
        return raw
    batch, feet = int(points.shape[0]), int(points.shape[1])
    padded_points = torch.zeros(batch, feet, contact_slots, 3, device=points.device, dtype=points.dtype)
    padded_forces = torch.zeros(batch, feet, contact_slots, device=forces.device, dtype=forces.dtype)
    padded_normals = torch.zeros(batch, feet, contact_slots, 3, device=normals.device, dtype=normals.dtype)
    padded_valid = torch.zeros(batch, feet, contact_slots, device=valid.device, dtype=torch.bool)
    padded_points[:, :, :current] = points
    padded_forces[:, :, :current] = forces
    padded_normals[:, :, :current] = normals
    padded_valid[:, :, :current] = valid.bool()
    return padded_points, padded_forces, padded_normals, padded_valid


def expected_support_and_envelope_from_foot_pose(
    foot_pos_w: torch.Tensor,
    foot_quat_w: torch.Tensor,
    *,
    contact_height: float,
    foot_half_length: float,
    foot_half_width: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive expected Contact phase and oriented support boxes from Clean feet."""

    if (
        foot_pos_w.ndim != 3
        or tuple(foot_pos_w.shape[1:]) != (2, 3)
        or tuple(foot_quat_w.shape) != (int(foot_pos_w.shape[0]), 2, 4)
        or not bool(torch.isfinite(foot_pos_w).all())
        or not bool(torch.isfinite(foot_quat_w).all())
    ):
        raise ValueError("expected support requires finite foot pose [T,2,3]/[T,2,4]")
    scalars = (float(contact_height), float(foot_half_length), float(foot_half_width))
    if (
        not all(math.isfinite(value) for value in scalars)
        or scalars[0] < 0.0
        or scalars[1] <= 0.0
        or scalars[2] <= 0.0
    ):
        raise ValueError("expected support thresholds/extents must be nonnegative/positive")
    support = foot_pos_w[..., 2] <= float(contact_height)
    envelope = support_envelope_from_foot_pose_and_mask(
        foot_pos_w,
        foot_quat_w,
        support,
        foot_half_length=foot_half_length,
        foot_half_width=foot_half_width,
    )
    return support.detach().clone(), envelope


def support_envelope_from_foot_pose_and_mask(
    foot_pos_w: torch.Tensor,
    foot_quat_w: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    foot_half_length: float,
    foot_half_width: float,
) -> torch.Tensor:
    """Build one oriented support box from an explicit two-foot support mask."""

    if (
        foot_pos_w.ndim != 3
        or tuple(foot_pos_w.shape[1:]) != (2, 3)
        or tuple(foot_quat_w.shape) != (int(foot_pos_w.shape[0]), 2, 4)
        or tuple(support_mask.shape) != (int(foot_pos_w.shape[0]), 2)
        or not bool(torch.isfinite(foot_pos_w).all())
        or not bool(torch.isfinite(foot_quat_w).all())
    ):
        raise ValueError("support envelope requires finite foot pose [B,2,3]/[B,2,4] and mask [B,2]")
    if (
        not math.isfinite(float(foot_half_length))
        or not math.isfinite(float(foot_half_width))
        or float(foot_half_length) <= 0.0
        or float(foot_half_width) <= 0.0
    ):
        raise ValueError("support-envelope half extents must be positive")
    support_mask = support_mask.bool()
    w, x, y, z = foot_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    rows: list[torch.Tensor] = []
    for frame_index in range(int(foot_pos_w.shape[0])):
        active = support_mask[frame_index]
        # The caller's explicit applicability mask owns no-support semantics.
        # Both feet only provide a finite geometric carrier for that masked row.
        if not bool(active.any()):
            active = torch.ones_like(active)
        mean_cos = torch.cos(yaw[frame_index, active]).mean()
        mean_sin = torch.sin(yaw[frame_index, active]).mean()
        norm = torch.sqrt(mean_cos.square() + mean_sin.square()).clamp_min(1.0e-8)
        cos_ref, sin_ref = mean_cos / norm, mean_sin / norm
        centers = foot_pos_w[frame_index, active, :2]
        center_x = cos_ref * centers[:, 0] + sin_ref * centers[:, 1]
        center_y = -sin_ref * centers[:, 0] + cos_ref * centers[:, 1]
        delta_yaw = yaw[frame_index, active] - torch.atan2(sin_ref, cos_ref)
        half_x = torch.cos(delta_yaw).abs() * float(foot_half_length) + torch.sin(delta_yaw).abs() * float(
            foot_half_width
        )
        half_y = torch.sin(delta_yaw).abs() * float(foot_half_length) + torch.cos(delta_yaw).abs() * float(
            foot_half_width
        )
        lower_x, upper_x = (center_x - half_x).min(), (center_x + half_x).max()
        lower_y, upper_y = (center_y - half_y).min(), (center_y + half_y).max()
        box_x, box_y = 0.5 * (lower_x + upper_x), 0.5 * (lower_y + upper_y)
        world_center = torch.stack((cos_ref * box_x - sin_ref * box_y, sin_ref * box_x + cos_ref * box_y))
        rows.append(
            torch.cat(
                (
                    world_center,
                    torch.stack((cos_ref, sin_ref)),
                    torch.stack((0.5 * (upper_x - lower_x), 0.5 * (upper_y - lower_y))),
                )
            )
        )
    return torch.stack(rows, dim=0).detach().clone()


# B2: Convert row-aligned raw foot-ground contacts into physical ZMP evidence.
def contact_wrench_zmp_xy(
    contact_points_w: torch.Tensor,
    normal_force_magnitudes: torch.Tensor,
    contact_normals_w: torch.Tensor,
    valid_contacts: torch.Tensor,
    *,
    vertical_force_epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal ZMP/CoP from raw foot-ground normal contact wrenches.

    Status: active FRS-GAIN-v006 Physics producer.
    Upstream: two filtered IsaacLab foot ContactSensor raw contact buffers.
    Downstream: ``expected_support_envelope_margin`` in the live probe.
    Evidence: deterministic golden/permutation/missing contracts; live values remain S4-only.

    Axes are ``[B, foot, contact, xyz]``. On the horizontal evaluation plane,
    each normal contact contributes its vertical force at its reported world
    contact point. Missing vertical resultant is explicit through ``valid``.
    """

    if (
        contact_points_w.ndim != 4
        or int(contact_points_w.shape[-1]) != 3
        or tuple(contact_normals_w.shape) != tuple(contact_points_w.shape)
        or tuple(normal_force_magnitudes.shape) != tuple(contact_points_w.shape[:-1])
        or tuple(valid_contacts.shape) != tuple(normal_force_magnitudes.shape)
    ):
        raise ValueError("contact-wrench ZMP requires points/normals [B,F,C,3] and force/mask [B,F,C]")
    if vertical_force_epsilon <= 0.0:
        raise ValueError("vertical_force_epsilon must be positive")
    finite = (
        torch.isfinite(contact_points_w).all(dim=-1)
        & torch.isfinite(contact_normals_w).all(dim=-1)
        & torch.isfinite(normal_force_magnitudes)
    )
    valid_contacts = valid_contacts.bool()
    if bool((valid_contacts & ~finite).any()):
        raise ValueError("valid raw contact-wrench entries must be finite")
    active = valid_contacts
    if bool((normal_force_magnitudes[active] < 0.0).any()):
        raise ValueError("normal contact-force magnitudes must be non-negative")
    vertical_force = normal_force_magnitudes * contact_normals_w[..., 2].abs()
    vertical_force = torch.where(active, vertical_force, torch.zeros_like(vertical_force))
    resultant = vertical_force.sum(dim=(1, 2))
    numerator = (vertical_force.unsqueeze(-1) * contact_points_w[..., :2]).sum(dim=(1, 2))
    valid = resultant > float(vertical_force_epsilon)
    zmp_xy = torch.full_like(numerator, float("nan"))
    zmp_xy[valid] = numerator[valid] / resultant[valid].unsqueeze(-1)
    return zmp_xy, valid


def expected_support_envelope_margin(
    zmp_xy_w: torch.Tensor,
    envelope_local: torch.Tensor,
    expected_support: torch.Tensor,
    *,
    env_origins_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return signed ZMP distance to a sealed Clean support envelope.

    ``envelope_local`` is ``[B,6] = center_xy, cos(yaw), sin(yaw), half_x,
    half_y``. Flight rows return NaN and are semantic N/A downstream.
    """

    if (
        tuple(zmp_xy_w.shape[-1:]) != (2,)
        or zmp_xy_w.ndim != 2
        or tuple(envelope_local.shape) != (int(zmp_xy_w.shape[0]), 6)
        or tuple(expected_support.shape) != (int(zmp_xy_w.shape[0]), 2)
    ):
        raise ValueError("support-envelope margin requires ZMP [B,2], envelope [B,6], support [B,2]")
    if not bool(torch.isfinite(envelope_local).all()):
        raise ValueError("sealed support envelope must be finite")
    center = envelope_local[:, :2]
    if env_origins_xy is not None:
        if tuple(env_origins_xy.shape) != tuple(center.shape):
            raise ValueError("env_origins_xy must match envelope centers [B,2]")
        center = center + env_origins_xy
    cos_yaw, sin_yaw = envelope_local[:, 2], envelope_local[:, 3]
    half = envelope_local[:, 4:6]
    if bool((half <= 0.0).any()):
        raise ValueError("support-envelope half extents must be positive")
    delta = zmp_xy_w - center
    local_x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    local_y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    margin = torch.minimum(half[:, 0] - local_x.abs(), half[:, 1] - local_y.abs())
    applicable = expected_support.bool().any(dim=-1)
    result = torch.full_like(margin, float("nan"))
    finite_zmp = torch.isfinite(zmp_xy_w).all(dim=-1)
    available = applicable & finite_zmp
    result[available] = margin[available]
    return result


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
