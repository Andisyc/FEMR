# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES temporal reference cache helpers."""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_inv, quat_mul

from rsl_rl.frontres.frontres_executability import quat_to_rotvec_wxyz as _quat_to_rotvec_wxyz


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


def frontres_temporal_continuity_correction(
    self,
    cmd_term,
    n: int,
    hsl_pos_corr: torch.Tensor,
    hsl_rpy_corr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build the continuity candidate from the previous refined frame.

    HSL gives the clean-oriented repair for the current raw frame. This helper
    is kept for legacy temporal-rejoin ablations; the active hsl_hybrid branch
    now uses PPO-owned per-axis acceptance over the HSL proposal rather than a
    temporal position rejoin.
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
