# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

"""Apply the active full-6D FrontRES Delta SE(3) command correction."""

from __future__ import annotations

import torch
from rsl_rl.frontres.frontres_executability import rotvec_to_quat_wxyz as _rotvec_to_quat_wxyz
from rsl_rl.frontres.frontres_formal_runtime_probe import emit_formal_runtime_probe
from rsl_rl.modules import FrontRESActorCritic


def _quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Multiply batches of wxyz quaternions without depending on simulator utilities."""

    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _world_delta_to_command_local_quat(raw_quat: torch.Tensor, world_delta_quat: torch.Tensor) -> torch.Tensor:
    """Convert a world-left Delta R into the host command's local-right buffer."""

    raw_unit = raw_quat / raw_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    raw_inverse = raw_unit.clone()
    raw_inverse[..., 1:] = -raw_inverse[..., 1:]
    return _quat_mul_wxyz(_quat_mul_wxyz(raw_inverse, world_delta_quat), raw_unit)


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
    # B1: host 使用 raw * buffer, 因此这里只转换 world-left Delta R 的表示.
    raw_quat = cmd_term.anchor_quat_w_raw[:n_train].to(device=rpy_corr.device, dtype=rpy_corr.dtype)
    train_quat = _world_delta_to_command_local_quat(raw_quat, _rotvec_to_quat_wxyz(rpy_corr))
    cmd_term._frontres_pos_correction[:n_train].copy_(pos_corr)
    cmd_term._frontres_quat_correction[:n_train].copy_(train_quat)
    if n_candidate > 0 and candidate_pos_corr is not None and candidate_rpy_corr is not None:
        start = n_train
        end = start + n_candidate
        candidate_raw_quat = cmd_term.anchor_quat_w_raw[start:end].to(
            device=candidate_rpy_corr.device,
            dtype=candidate_rpy_corr.dtype,
        )
        candidate_quat = _world_delta_to_command_local_quat(
            candidate_raw_quat,
            _rotvec_to_quat_wxyz(candidate_rpy_corr),
        )
        cmd_term._frontres_pos_correction[start:end].copy_(candidate_pos_corr)
        cmd_term._frontres_quat_correction[start:end].copy_(candidate_quat)

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
    # B1: 验证 active policy 拥有 finite full-6D world-frame Delta SE(3) action.
    if task_corr is None:
        return None
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    if not isinstance(policy, FrontRESActorCritic) or getattr(policy, "num_task_corrections", 0) != 6:
        return task_corr
    if task_corr.ndim != 2 or int(task_corr.shape[1]) != 6:
        raise ValueError(f"FrontRES task correction must have shape [B,6], got {tuple(task_corr.shape)}")
    if not bool(torch.isfinite(task_corr).all()):
        raise ValueError("FrontRES task correction must contain only finite values")

    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    command_manager = getattr(env_raw, "command_manager", None)
    terms = getattr(command_manager, "_terms", None)
    if terms is None:
        return task_corr

    if n_train is None:
        n_train = int(task_corr.shape[0])
    n_train = max(0, min(int(n_train), int(task_corr.shape[0]), int(runner.env.num_envs)))
    n_candidate = max(0, min(int(n_candidate), int(runner.env.num_envs) - n_train))
    # B2: 平移原样写入; rotation 只做 world-left -> host local-right 表示转换.
    for cmd_term in terms.values():
        if not hasattr(cmd_term, "_frontres_pos_correction"):
            continue
        if not hasattr(cmd_term, "anchor_quat_w_raw"):
            raise RuntimeError("FrontRES command term must expose anchor_quat_w_raw for world-frame rotation")
        pos_corr = task_corr[:n_train, :3].clone()
        rpy_corr = task_corr[:n_train, 3:6].clone()
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
