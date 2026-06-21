# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: train a tiny conditional rho head with the formal FrontRES loss.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_conditional_acceptance_head_training.py

This does not start IsaacLab and does not load a checkpoint.  It is stronger
than ``frontres_conditional_acceptance_loss.py`` because it does not optimize a
free rho value per sample.  Instead, a small shared linear head maps synthetic
observations to the 6D acceptance logits, and the formal FrontRES structured-rho
loss trains that head.

The test answers this question:

    If observation contains the condition, can the current formal loss train a
    policy head to write rho on positive-advantage dimensions and suppress rho
    on negative/boundary dimensions?

If this passes but live training keeps ``rho+adv ~= rho-adv``, the likely issue
is checkpoint/history, live observation separability, or live evidence
distribution rather than a local algebra bug in the structured-rho loss.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

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


class TinyAcceptanceHead(nn.Module):
    def __init__(self, obs_dim: int, *, init_rho: float):
        super().__init__()
        self.linear = nn.Linear(obs_dim, 12)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.linear.bias.data[6:12].fill_(_logit(init_rho))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


def _logit(p: float) -> float:
    p = min(1.0 - 1.0e-6, max(1.0e-6, float(p)))
    return math.log(p / (1.0 - p))


def _build_separable_batch(device: torch.device) -> SimpleNamespace:
    names = [
        "all_pos",
        "all_neg",
        "pos_yes_rpy_no",
        "pos_no_rpy_yes",
        "boundary_no",
        "zero",
    ]
    obs = torch.eye(len(names), dtype=torch.float32, device=device)
    rho_adv = torch.tensor(
        [
            [0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
            [-0.70, -0.70, -0.70, -0.70, -0.70, -0.70],
            [0.85, 0.85, 0.85, -0.65, -0.65, -0.65],
            [-0.65, -0.65, -0.65, 0.85, 0.85, 0.85],
            [0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
        device=device,
    )
    n = len(names)
    return SimpleNamespace(
        names=names,
        n=n,
        obs=obs,
        rho_adv=rho_adv,
        rho_mask=torch.ones_like(rho_adv),
        prior_authority=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device).view(n, 1),
        prior_target=torch.zeros(n, 6, dtype=torch.float32, device=device),
        old_mu=torch.zeros(n, 12, dtype=torch.float32, device=device),
        old_sigma=torch.ones(n, 12, dtype=torch.float32, device=device),
        actions_log_prob=torch.zeros(n, 1, dtype=torch.float32, device=device),
        old_actions_log_prob=torch.zeros(n, 1, dtype=torch.float32, device=device),
    )


def _formal_loss(fake_alg: FakeAlgorithm, batch: SimpleNamespace, mu: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    actions = torch.zeros(batch.n, 12, device=mu.device, dtype=mu.dtype)
    actions[:, 6:12] = torch.sigmoid(mu[:, 6:12])
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


def _train_head(batch: SimpleNamespace, fake_alg: FakeAlgorithm) -> tuple[TinyAcceptanceHead, dict[str, float]]:
    head = TinyAcceptanceHead(batch.obs.shape[-1], init_rho=0.25).to(batch.obs.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.08)
    last_metrics: dict[str, float] = {}
    for _ in range(260):
        mu = head(batch.obs)
        loss, last_metrics = _formal_loss(fake_alg, batch, mu)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return head, last_metrics


def _print_sample_table(names: list[str], rho: torch.Tensor) -> None:
    print("per-sample final rho:")
    for i, name in enumerate(names):
        print(
            f"  {name:<15} "
            f"pos={rho[i, :3].mean().item():.3f} "
            f"rpy={rho[i, 3:].mean().item():.3f}"
        )


def run_conditional_acceptance_head_training_check() -> None:
    device = torch.device("cpu")
    fake_alg = FakeAlgorithm(device)
    batch = _build_separable_batch(device)
    head, metrics = _train_head(batch, fake_alg)
    with torch.no_grad():
        rho = torch.sigmoid(head(batch.obs)[:, 6:12])

    print("=== FrontRES Conditional Acceptance Head TRAINING TEST ONLY ===")
    print("This checks formal structured-rho loss through a shared obs -> rho head.")
    print(
        "final metrics: "
        f"rho+={metrics['structured_joint_rl_rho_pos_adv_mean']:.3f}, "
        f"rho-={metrics['structured_joint_rl_rho_neg_adv_mean']:.3f}, "
        f"+pos/+rpy={metrics['structured_joint_rl_rho_pos_adv_pos_dim_mean']:.3f}/"
        f"{metrics['structured_joint_rl_rho_pos_adv_rpy_dim_mean']:.3f}, "
        f"-pos/-rpy={metrics['structured_joint_rl_rho_neg_adv_pos_dim_mean']:.3f}/"
        f"{metrics['structured_joint_rl_rho_neg_adv_rpy_dim_mean']:.3f}"
    )
    _print_sample_table(batch.names, rho)

    if rho[0].mean().item() <= 0.90:
        raise AssertionError("all_pos sample did not learn high rho")
    if rho[1].mean().item() >= 0.05:
        raise AssertionError("all_neg sample did not learn low rho")
    if rho[2, :3].mean().item() <= 0.90 or rho[2, 3:].mean().item() >= 0.05:
        raise AssertionError("pos_yes_rpy_no sample did not learn axis-conditional rho")
    if rho[3, :3].mean().item() >= 0.05 or rho[3, 3:].mean().item() <= 0.90:
        raise AssertionError("pos_no_rpy_yes sample did not learn axis-conditional rho")
    if rho[4].mean().item() >= 0.10:
        raise AssertionError("boundary prior sample did not learn low rho")
    # The zero-evidence row has no direct loss, but a shared neural head can
    # still move it through the shared bias/trunk.  Treat this as a diagnostic
    # for representation leakage, not as a formal structured-loss failure.
    zero_rho = rho[5].mean().item()
    if zero_rho <= 0.0 or zero_rho >= 1.0:
        raise AssertionError("zero-evidence sample produced invalid rho")
    print(f"zero-evidence drift diagnostic: initial=0.250 final={zero_rho:.3f}")
    print("result: PASS")


if __name__ == "__main__":
    run_conditional_acceptance_head_training_check()
