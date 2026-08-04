# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime probes for the active full-6D supervised restore path.

Status: active Stage 2 diagnostic boundary.
Upstream: full-6D actor actions, full-6D HSL target, motion command state.
Downstream: human-readable restore consistency evidence only.
Gap: this probe is offline/source-verified here; live population remains a
runtime test concern.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_inv, quat_mul

from rsl_rl.frontres.frontres_executability import (
    quat_to_rotvec_wxyz as _quat_to_rotvec_wxyz,
)


def maybe_print_frontres_restore_debug(
    self,
    it: int,
    rollout_step: int,
    actions: torch.Tensor | None,
    supervised_target: torch.Tensor | None,
    n_train: int,
) -> None:
    """Print low-frequency full-6D target, write, and restore diagnostics."""
    if actions is None or supervised_target is None or rollout_step != 0:
        return
    objective = str(getattr(self.alg, "frontres_training_objective", "")).lower()
    if objective != "supervised_restore":
        return
    interval = int(
        getattr(
            self.alg,
            "frontres_restore_debug_print_interval",
            self.cfg.get("frontres_restore_debug_print_interval", 10),
        )
    )
    if interval <= 0 or int(it) % interval != 0:
        return
    if getattr(self, "_frontres_restore_debug_last_iter", None) == int(it):
        return
    self._frontres_restore_debug_last_iter = int(it)

    env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
    terms = getattr(getattr(env, "command_manager", None), "_terms", {})
    cmd = next(
        (
            term
            for term in terms.values()
            if all(
                hasattr(term, name)
                for name in (
                    "anchor_quat_w_original",
                    "anchor_quat_w_raw",
                    "_frontres_quat_correction",
                )
            )
        ),
        None,
    )
    if cmd is None:
        return

    n = max(0, min(int(n_train), actions.shape[0], supervised_target.shape[0]))
    if n <= 0 or actions.shape[-1] < 6 or supervised_target.shape[-1] < 6:
        return

    device = self.device
    raw_q = cmd.anchor_quat_w_raw[:n].to(device)
    clean_q = cmd.anchor_quat_w_original[:n].to(device)
    written_q = cmd._frontres_quat_correction[:n].to(device)
    target6 = supervised_target[:n, :6].detach()
    pred6 = actions[:n, :6].detach()
    target_pos, target_rpy = target6[:, :3], target6[:, 3:6]
    pred_pos, pred_rpy = pred6[:, :3], pred6[:, 3:6]
    written_rpy = _quat_to_rotvec_wxyz(written_q)[:, :3]

    clean_from_raw = _quat_to_rotvec_wxyz(quat_mul(quat_inv(raw_q), clean_q))[:, :3]
    corrected_q = quat_mul(raw_q, written_q)
    corrected_err = _quat_to_rotvec_wxyz(quat_mul(quat_inv(corrected_q), clean_q))[:, :3]

    def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)

    def _mean_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        valid = b.norm(dim=-1) > 1e-4
        return _safe_cos(a[valid], b[valid]).mean() if valid.any() else torch.tensor(0.0, device=device)

    noisy_err = clean_from_raw[:, :2].norm(dim=-1)
    corrected_err_norm = corrected_err[:, :2].norm(dim=-1)
    restore_gain = noisy_err - corrected_err_norm
    prev = getattr(self, "_frontres_restore_debug_prev_action", None)
    if prev is not None and prev.shape == pred6.shape:
        step_jump = (pred6[:, :2] - prev[:, :2]).norm(dim=-1).mean()
    else:
        step_jump = torch.tensor(0.0, device=device)
    self._frontres_restore_debug_prev_action = pred6.clone()

    def _vec(t: torch.Tensor, idx: int) -> list[float]:
        return [round(float(v), 5) for v in t[idx, :3].detach().cpu().tolist()]

    print(
        "[FrontRES restore debug] "
        f"it={int(it)} mode=full6 n={n} "
        f"cos(pos)={float(_mean_cos(pred_pos, target_pos)):+.4f} "
        f"cos(rpy)={float(_mean_cos(pred_rpy, target_rpy)):+.4f} "
        f"mean_rp={float(pred_rpy[:, :2].norm(dim=-1).mean()):.5f} "
        f"jump_rp={float(step_jump):.5f}",
        flush=True,
    )
    print(
        "[FrontRES restore debug] "
        f"|raw-clean|={float(noisy_err.mean()):.5f} "
        f"|written-clean|={float(corrected_err_norm.mean()):.5f} "
        f"gain={float(restore_gain.mean()):+.5f}",
        flush=True,
    )
    sample_idx = int(torch.argmax(noisy_err).item())
    print(
        "[FrontRES restore debug] sample vectors "
        f"target_pos={_vec(target_pos, sample_idx)} "
        f"pred_pos={_vec(pred_pos, sample_idx)} "
        f"target_rpy={_vec(target_rpy, sample_idx)} "
        f"pred_rpy={_vec(pred_rpy, sample_idx)} "
        f"written_rpy={_vec(written_rpy, sample_idx)} "
        f"residual_rpy={_vec(corrected_err, sample_idx)}",
        flush=True,
    )
