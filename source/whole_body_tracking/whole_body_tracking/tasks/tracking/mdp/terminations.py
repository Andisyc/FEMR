from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_termination",
    Path(__file__).resolve().parents[5] / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_formal_runtime_probe.py",
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe
formal_runtime_probe_enabled = _AUDIT_MODULE.formal_runtime_probe_enabled


def _frontres_quartet_role_values(command: MotionCommand, value: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """Return one diagnostic tensor view per active quartet role."""

    role_ids = {
        "policy": getattr(command, "_frontres_pair_train_ids", None),
        "candidate": getattr(command, "_frontres_pair_candidate_ids", None),
        "noisy": getattr(command, "_frontres_pair_base_ids", None),
        "clean": getattr(command, "_frontres_pair_clean_ids", None),
    }
    if any(not isinstance(env_ids, torch.Tensor) for env_ids in role_ids.values()):
        return None
    return {
        role: value.index_select(0, env_ids.to(value.device, dtype=torch.long))
        for role, env_ids in role_ids.items()
    }


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1)
    # NaN error (from NaN FrontRES correction) must terminate — IEEE 754: NaN > x = False,
    # so without this guard a NaN anchor silently disables all position terminations.
    return (error > threshold) | torch.isnan(error)


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # B1: 读取 termination 实际消费的世界系 reference/robot anchor z.
    reference_z = command.anchor_pos_w[:, -1]
    robot_z = command.robot_anchor_pos_w[:, -1]
    signed_error = reference_z - robot_z
    error = torch.abs(signed_error)
    result = (error > threshold) | torch.isnan(error)

    # B2: 将最终 reference z 分解为 clean/raw/correction, 并保留 frame identity.
    if formal_runtime_probe_enabled():
        clean_reference_z = command.anchor_pos_w_original[:, -1]
        raw_reference_z = command.anchor_pos_w_raw[:, -1]
        correction_z = command._frontres_pos_correction[:, -1]
        role_reference_z = _frontres_quartet_role_values(command, reference_z)
        if role_reference_z is not None:
            # B3: 在原 termination mask 返回前截获同一批 role, 不改变 done 语义.
            # AUDIT-ANCHOR-Z-01: 检查 command reference -> robot torso -> anchor_pos termination 数值链.
            # Result: E35 found stale raw_z; cache fix is inserted, so live result is stale-rerun-required.
            emit_formal_runtime_probe(
                "AUDIT-ANCHOR-Z-01",
                limit=2,
                reference_z=role_reference_z,
                robot_z=_frontres_quartet_role_values(command, robot_z),
                signed_error=_frontres_quartet_role_values(command, signed_error),
                abs_error=_frontres_quartet_role_values(command, error),
                terminated=_frontres_quartet_role_values(command, result),
                threshold=float(threshold),
                clean_reference_z=_frontres_quartet_role_values(command, clean_reference_z),
                raw_reference_z=_frontres_quartet_role_values(command, raw_reference_z),
                correction_z=_frontres_quartet_role_values(command, correction_z),
                time_steps=_frontres_quartet_role_values(command, command.time_steps),
                motion_indices=_frontres_quartet_role_values(command, command.env_motion_indices),
            )
    return result


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    error = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs()
    return (error > threshold) | torch.isnan(error)


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any((error > threshold) | torch.isnan(error), dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any((error > threshold) | torch.isnan(error), dim=-1)

def motion_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not hasattr(command, "motion_end_buf"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return command.motion_end_buf
