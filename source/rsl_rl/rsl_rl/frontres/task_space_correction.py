# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

"""Apply the active full-6D FrontRES Delta SE(3) command correction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import torch
from rsl_rl.frontres.frontres_executability import rotvec_to_quat_wxyz as _rotvec_to_quat_wxyz
from rsl_rl.modules import FrontRESActorCritic

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_correction",
    Path(__file__).resolve().with_name("frontres_formal_runtime_probe.py"),
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


def _frontres_contact_consistent_position_correction(
    runner,
    cmd_term,
    pos_corr: torch.Tensor,
    n_rows: int,
) -> torch.Tensor:
    """Keep horizontal and upward root corrections compatible with contact state."""
    if hasattr(cmd_term, "jump_degree"):
        jump_degree = cmd_term.jump_degree[:n_rows].to(device=pos_corr.device, dtype=pos_corr.dtype).clamp(0.0, 1.0)
        pos_corr[:, :2] *= (1.0 - jump_degree).unsqueeze(-1)
        if hasattr(cmd_term, "anchor_penetration_depth"):
            penetration = cmd_term.anchor_penetration_depth[:n_rows].to(device=pos_corr.device, dtype=pos_corr.dtype)
            z_upper = jump_degree * penetration
        else:
            z_upper = torch.zeros_like(pos_corr[:, 2])
    else:
        z_upper = torch.zeros_like(pos_corr[:, 2])

    max_delta_pos = float(getattr(runner.alg.policy, "max_delta_pos", 0.3))
    z_lower = torch.full_like(z_upper, -max_delta_pos)
    pos_corr[:, 2] = torch.minimum(torch.maximum(pos_corr[:, 2], z_lower), z_upper)
    return pos_corr


def _write_frontres_command_correction(
    runner,
    cmd_term,
    pos_corr: torch.Tensor,
    rpy_corr: torch.Tensor,
    candidate_pos_corr: torch.Tensor | None,
    candidate_rpy_corr: torch.Tensor | None,
    n_train: int,
    n_candidate: int,
) -> None:
    cmd_term._frontres_pos_correction[:n_train].copy_(pos_corr)
    cmd_term._frontres_quat_correction[:n_train].copy_(_rotvec_to_quat_wxyz(rpy_corr))
    if n_candidate > 0 and candidate_pos_corr is not None and candidate_rpy_corr is not None:
        start = n_train
        end = start + n_candidate
        cmd_term._frontres_pos_correction[start:end].copy_(candidate_pos_corr)
        cmd_term._frontres_quat_correction[start:end].copy_(_rotvec_to_quat_wxyz(candidate_rpy_corr))

    zero_start = n_train + n_candidate
    if zero_start < runner.env.num_envs:
        cmd_term._frontres_pos_correction[zero_start:].zero_()
        cmd_term._frontres_quat_correction[zero_start:].zero_()
        cmd_term._frontres_quat_correction[zero_start:, 0] = 1.0


def apply_frontres_task_corrections(
    runner,
    task_corr: torch.Tensor | None,
    n_train: int | None = None,
    *,
    allow_oracle: bool = False,
    n_candidate: int = 0,
) -> torch.Tensor | None:
    # QUALITY-ACTION-01: 检查 actor full-6D action -> task-space application buffers.
    # Result: Q-E3 OFFLINE SOURCE/SHAPE PASS; real application identity pending Q1-F.
    # B1: application 前保留 zero/HSL/policy raw bounded 6D action.
    # B2: contact-consistent correction 后记录真正写入的 pos/rpy correction.
    # B3: command/GMT 消费前比较 requested 与 executed action identity.
    """把 full-6D FrontRES correction 写入 motion command buffers.

    函数名说明:
        `apply_frontres_task_corrections` 是 task-space application owner, 只负责
        position/rotation correction 的表示转换与写入; 它不是 policy mask,
        reward gate 或 authority controller.

    主链路:
        上游: rollout step 提供 FrontRES sampled correction 和 split-env row count.
        下游: motion command term 读取 position/quaternion buffers, frozen GMT 随后
        跟踪修正后的 reference.

    语义:
        正式路径始终接收 full-6D Delta SE(3). `allow_oracle` 仅保留 wrapper API
        兼容性, 当前方法不存在 oracle/stable/authority 分支.
    """
    del allow_oracle
    # B1: 验证 active policy 拥有 full-6D Delta SE(3) action.
    if task_corr is None:
        return None
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    if not isinstance(policy, FrontRESActorCritic) or getattr(policy, "num_task_corrections", 0) != 6:
        return task_corr

    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    command_manager = getattr(env_raw, "command_manager", None)
    terms = getattr(command_manager, "_terms", None)
    if terms is None:
        return task_corr

    if n_train is None:
        n_train = int(task_corr.shape[0])
    n_train = max(0, min(int(n_train), int(task_corr.shape[0]), int(runner.env.num_envs)))
    n_candidate = max(0, min(int(n_candidate), int(runner.env.num_envs) - n_train))
    # B2: 转换 position/rotation 表示并写入 command-owned correction buffers.
    for cmd_term in terms.values():
        if not hasattr(cmd_term, "_frontres_pos_correction"):
            continue
        pos_corr = task_corr[:n_train, :3].clone()
        rpy_corr = task_corr[:n_train, 3:6].clone()
        pos_corr = _frontres_contact_consistent_position_correction(runner, cmd_term, pos_corr, n_train)
        candidate_pos_corr = pos_corr[:n_candidate].clone() if n_candidate else None
        candidate_rpy_corr = rpy_corr[:n_candidate].clone() if n_candidate else None
        _write_frontres_command_correction(
            runner,
            cmd_term,
            pos_corr,
            rpy_corr,
            candidate_pos_corr,
            candidate_rpy_corr,
            n_train,
            n_candidate,
        )
        # B3: AUDIT-APPLY-01 截获写入 command owner 的实际 correction.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-APPLY-01",
            task_corr=task_corr[:n_train],
            position_correction=cmd_term._frontres_pos_correction[:n_train],
            rotation_correction=cmd_term._frontres_quat_correction[:n_train],
            n_train=n_train,
            n_candidate=n_candidate,
        )
    return task_corr
