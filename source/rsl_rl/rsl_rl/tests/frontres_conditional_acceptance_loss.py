# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: conditional rho acceptance loss sanity check.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_conditional_acceptance_loss.py

This test does not start IsaacLab and does not load a checkpoint.  It feeds a
hand-built batch into the formal FrontRES structured-rho algorithm loss, then
optimizes only synthetic rho logits.  It answers one narrow question:

    Can the current formal loss separate positive-advantage rho from
    negative-advantage rho, including pos and rpy dimensions?

If this test passes but live training keeps rho+adv ~= rho-adv, the likely
problem is checkpoint/history/optimization state or live evidence distribution,
not a local algebra bug in the loss.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified


class FakePolicy:
    task_conf_dim = 6

    def get_actions_log_prob_per_dim(self, actions: torch.Tensor, dims: list[int]) -> torch.Tensor:
        return torch.zeros(actions.shape[0], len(dims), device=actions.device, dtype=actions.dtype)

    def get_actions_log_prob_per_dim_from_stats(
        self,
        actions: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        dims: list[int],
    ) -> torch.Tensor:
        return torch.zeros(actions.shape[0], len(dims), device=actions.device, dtype=actions.dtype)


class FakeAlgorithm:
    def __init__(self, device: torch.device):
        self.device = device
        self.policy = FakePolicy()
        self.clip_param = 0.2
        self.frontres_structured_joint_rl_enabled = True
        self.frontres_structured_joint_rl_weight = 1.0
        self.frontres_structured_joint_rl_adv_clip = 5.0
        self.frontres_structured_joint_rl_normalize_advantage = False
        self.frontres_structured_joint_rl_loss_mode = "region_direct"
        self.frontres_structured_joint_prior_loss_weight = 1.0
        self.frontres_structured_joint_repair_loss_kind = "bce_logit"
        self.frontres_structured_joint_repair_loss_scale = 1.0
        self.frontres_reward_compute_live_debug = False

    def _structured_joint_rl_enabled(self) -> bool:
        return bool(self.frontres_structured_joint_rl_enabled) and self.frontres_structured_joint_rl_weight > 0.0


def _logit(p: float) -> float:
    p = min(1.0 - 1.0e-6, max(1.0e-6, float(p)))
    return math.log(p / (1.0 - p))


def _make_batch(device: torch.device, *, pos_init: float = 0.25, rpy_init: float = 0.25) -> SimpleNamespace:
    """Build samples where pos/rpy dimensions require different rho directions."""

    n = 6
    rho_adv = torch.tensor(
        [
            # pos dims positive, rpy dims positive
            [0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
            # pos dims negative, rpy dims negative
            [-0.70, -0.70, -0.70, -0.70, -0.70, -0.70],
            # pos wants write, rpy wants no-write
            [0.85, 0.85, 0.85, -0.65, -0.65, -0.65],
            # pos wants no-write, rpy wants write
            [-0.65, -0.65, -0.65, 0.85, 0.85, 0.85],
            # boundary prior says no-write for all dims
            [0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
            # weak near-zero evidence should barely matter
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
        device=device,
    )
    rho_mask = torch.ones_like(rho_adv)
    prior_authority = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device).view(n, 1)
    prior_target = torch.zeros(n, 6, dtype=torch.float32, device=device)
    init_rho = torch.tensor([pos_init, pos_init, pos_init, rpy_init, rpy_init, rpy_init], device=device).view(1, 6)
    init_rho_logits = torch.logit(init_rho.clamp(1e-6, 1.0 - 1e-6)).expand(n, 6).clone()
    init_mu = torch.zeros(n, 12, device=device)
    init_mu[:, 6:12] = init_rho_logits
    actions = torch.zeros(n, 12, device=device)
    actions[:, 6:12] = torch.sigmoid(init_rho_logits)
    return SimpleNamespace(
        n=n,
        obs=torch.zeros(n, 4, device=device),
        mu=init_mu,
        actions=actions,
        old_mu=torch.zeros(n, 12, device=device),
        old_sigma=torch.ones(n, 12, device=device),
        actions_log_prob=torch.zeros(n, 1, device=device),
        old_actions_log_prob=torch.zeros(n, 1, device=device),
        rho_adv=rho_adv,
        rho_mask=rho_mask,
        prior_authority=prior_authority,
        prior_target=prior_target,
    )


def _formal_loss(fake_alg: FakeAlgorithm, batch: SimpleNamespace, mu: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    actions = batch.actions.clone()
    actions[:, 6:12] = torch.sigmoid(mu.detach()[:, 6:12])
    return FrontRESUnified._compute_structured_joint_rl_loss(
        fake_alg,
        batch.obs,
        mu,
        actions,
        batch.old_mu,
        batch.old_sigma,
        batch.actions_log_prob,
        batch.old_actions_log_prob,
        batch.rho_adv,
        batch.rho_mask,
        batch.prior_authority,
        batch.prior_target,
        original_batch_size=batch.n,
    )


def run_conditional_acceptance_loss_check() -> None:
    device = torch.device("cpu")
    fake_alg = FakeAlgorithm(device)
    batch = _make_batch(device)
    mu = torch.nn.Parameter(batch.mu.clone())
    optimizer = torch.optim.Adam([mu], lr=0.15)

    _, metrics0 = _formal_loss(fake_alg, batch, mu)
    for _ in range(160):
        loss, _ = _formal_loss(fake_alg, batch, mu)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _, metrics1 = _formal_loss(fake_alg, batch, mu)
    rho = torch.sigmoid(mu.detach()[:, 6:12])
    pos_dims = slice(0, 3)
    rpy_dims = slice(3, 6)

    print("=== FrontRES Conditional Acceptance Loss TEST ONLY ===")
    print("This checks formal structured-rho loss without checkpoint or environment.")
    print(
        "initial: "
        f"rho+={metrics0['structured_joint_rl_rho_pos_adv_mean']:.3f}, "
        f"rho-={metrics0['structured_joint_rl_rho_neg_adv_mean']:.3f}, "
        f"pos/rpy={metrics0['structured_joint_rl_rho_pos_dim_mean']:.3f}/"
        f"{metrics0['structured_joint_rl_rho_rpy_dim_mean']:.3f}"
    )
    print(
        "final:   "
        f"rho+={metrics1['structured_joint_rl_rho_pos_adv_mean']:.3f}, "
        f"rho-={metrics1['structured_joint_rl_rho_neg_adv_mean']:.3f}, "
        f"pos/rpy={metrics1['structured_joint_rl_rho_pos_dim_mean']:.3f}/"
        f"{metrics1['structured_joint_rl_rho_rpy_dim_mean']:.3f}, "
        f"+pos/+rpy={metrics1['structured_joint_rl_rho_pos_adv_pos_dim_mean']:.3f}/"
        f"{metrics1['structured_joint_rl_rho_pos_adv_rpy_dim_mean']:.3f}, "
        f"-pos/-rpy={metrics1['structured_joint_rl_rho_neg_adv_pos_dim_mean']:.3f}/"
        f"{metrics1['structured_joint_rl_rho_neg_adv_rpy_dim_mean']:.3f}"
    )
    print("per-sample final rho:")
    names = ["all_pos", "all_neg", "pos_yes_rpy_no", "pos_no_rpy_yes", "boundary_no", "zero"]
    for i, name in enumerate(names):
        print(
            f"  {name:<15} "
            f"pos={rho[i, pos_dims].mean().item():.3f} "
            f"rpy={rho[i, rpy_dims].mean().item():.3f}"
        )

    if metrics1["structured_joint_rl_rho_pos_adv_mean"] <= metrics1["structured_joint_rl_rho_neg_adv_mean"] + 0.20:
        raise AssertionError("positive-advantage rho did not separate from negative-advantage rho")
    if metrics1["structured_joint_rl_rho_pos_adv_rpy_dim_mean"] <= metrics1["structured_joint_rl_rho_neg_adv_rpy_dim_mean"] + 0.20:
        raise AssertionError("rpy positive-advantage rho did not separate from rpy negative-advantage rho")
    if rho[4].mean().item() >= 0.20:
        raise AssertionError("boundary prior sample did not learn low rho")
    print("result: PASS")


if __name__ == "__main__":
    run_conditional_acceptance_loss_check()
