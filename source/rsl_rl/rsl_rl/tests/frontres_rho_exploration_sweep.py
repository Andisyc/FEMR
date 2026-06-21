# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: FrontRES rho exploration/update sweep.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_rho_exploration_sweep.py

This module does not start an environment.  It answers a narrow debugging
question: under the formal structured-rho loss, how much does rho move when the
sampled rho action is only a small distance from the policy mean?

The live log field to compare against is:

    act-mu = structured_joint_rl_rho_action_minus_mean_abs
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified


@dataclass(frozen=True)
class SweepCase:
    name: str
    action_delta_raw: float
    prior_weight: float
    action_std: float
    lr: float = 6.5e-5
    loss_weight: float = 1.0
    steps: int = 200


class SweepPolicy(torch.nn.Module):
    """Minimal policy matching FrontRES rho's bounded-action/logit contract."""

    task_conf_dim = 6

    def __init__(self, batch_size: int, init_rho: float, action_std: float):
        super().__init__()
        init_raw = torch.logit(torch.tensor(float(init_rho)).clamp(1e-6, 1.0 - 1e-6))
        action_mean = torch.zeros(batch_size, 12)
        action_mean[:, 6:12] = init_raw
        self.action_mean = torch.nn.Parameter(action_mean)
        self.register_buffer("action_std", torch.full((batch_size, 12), float(action_std)))

    @staticmethod
    def _rho_action_to_raw(actions: torch.Tensor, dims: list[int]) -> torch.Tensor:
        rho_action = actions[:, dims].clamp(1e-6, 1.0 - 1e-6)
        return torch.log(rho_action / (1.0 - rho_action))

    def get_actions_log_prob_per_dim(self, actions: torch.Tensor, dims: list[int]) -> torch.Tensor:
        action_raw = self._rho_action_to_raw(actions, dims)
        mu = self.action_mean[:, dims]
        sigma = self.action_std[:, dims].clamp(min=1e-6)
        return -0.5 * ((action_raw - mu) / sigma).pow(2) - torch.log(sigma)

    def get_actions_log_prob_per_dim_from_stats(
        self,
        actions: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        dims: list[int],
    ) -> torch.Tensor:
        action_raw = self._rho_action_to_raw(actions, dims)
        mu = old_mu[:, dims]
        sigma = old_sigma[:, dims].clamp(min=1e-6)
        return -0.5 * ((action_raw - mu) / sigma).pow(2) - torch.log(sigma)


class SweepAlgorithm(torch.nn.Module):
    """Minimal FrontRESUnified self object for formal loss calls."""

    def __init__(self, batch_size: int, init_rho: float, prior_weight: float, action_std: float):
        super().__init__()
        self.device = torch.device("cpu")
        self.policy = SweepPolicy(batch_size, init_rho=init_rho, action_std=action_std)
        self.clip_param = 0.2
        self.frontres_structured_joint_rl_enabled = True
        self.frontres_structured_joint_rl_weight = 1.0
        self.frontres_structured_joint_rl_adv_clip = 5.0
        self.frontres_structured_joint_rl_normalize_advantage = False
        self.frontres_structured_joint_prior_loss_weight = float(prior_weight)
        self.frontres_reward_compute_live_debug = False

    def _structured_joint_rl_enabled(self) -> bool:
        return bool(self.frontres_structured_joint_rl_enabled) and self.frontres_structured_joint_rl_weight > 0.0


def _make_live_like_batch(
    *,
    batch_size: int,
    init_rho: float,
    action_delta_raw: float,
    action_std: float,
) -> dict[str, torch.Tensor]:
    """Construct a live-like rho batch with controllable action-mean distance.

    The sign mix mirrors the recent log approximately:
        positive advantage: 66%
        negative advantage: 32%
        zero advantage:      2%

    All nonzero samples are placed above the current mean in raw rho space.  In
    that setup positive advantage pulls rho up, while negative advantage pushes
    rho down.  The net effect is therefore easy to inspect.
    """

    n_pos = int(round(batch_size * 0.66))
    n_neg = int(round(batch_size * 0.32))
    n_zero = batch_size - n_pos - n_neg

    init_raw = torch.logit(torch.tensor(float(init_rho)).clamp(1e-6, 1.0 - 1e-6))
    high_rho = torch.sigmoid(init_raw + float(action_delta_raw))
    mean_rho = torch.sigmoid(init_raw)

    actions = torch.zeros(batch_size, 12)
    actions[: n_pos + n_neg, 6:12] = high_rho
    if n_zero > 0:
        actions[n_pos + n_neg :, 6:12] = mean_rho

    rho_adv = torch.zeros(batch_size, 6)
    rho_adv[:n_pos, :] = 0.75
    rho_adv[n_pos : n_pos + n_neg, :] = -0.75

    rho_weight = torch.ones(batch_size, 6)

    prior_authority = torch.zeros(batch_size, 1)
    n_prior = int(round(batch_size * 0.225))
    if n_prior > 0:
        prior_authority[:n_prior, 0] = 1.0

    return {
        "obs": torch.ones(batch_size, 4),
        "actions": actions,
        "old_mu": torch.zeros(batch_size, 12),
        "old_sigma": torch.full((batch_size, 12), float(action_std)),
        "old_logp": torch.zeros(batch_size, 1),
        "new_logp": torch.zeros(batch_size, 1),
        "rho_adv": rho_adv,
        "rho_weight": rho_weight,
        "prior_authority": prior_authority,
        "prior_target": torch.zeros(batch_size, 6),
    }


def _loss_once(
    alg: SweepAlgorithm,
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    return FrontRESUnified._compute_structured_joint_rl_loss(
        alg,
        tensors["obs"],
        alg.policy.action_mean,
        tensors["actions"],
        tensors["old_mu"],
        tensors["old_sigma"],
        tensors["new_logp"],
        tensors["old_logp"],
        tensors["rho_adv"],
        tensors["rho_weight"],
        tensors["prior_authority"],
        tensors["prior_target"],
        original_batch_size=tensors["obs"].shape[0],
    )


def _run_case(case: SweepCase, *, batch_size: int = 100, init_rho: float = 0.45) -> dict[str, float]:
    torch.manual_seed(11)
    alg = SweepAlgorithm(
        batch_size=batch_size,
        init_rho=init_rho,
        prior_weight=case.prior_weight,
        action_std=case.action_std,
    )
    tensors = _make_live_like_batch(
        batch_size=batch_size,
        init_rho=init_rho,
        action_delta_raw=case.action_delta_raw,
        action_std=case.action_std,
    )
    tensors["old_mu"] = alg.policy.action_mean.detach().clone()

    loss, metrics = _loss_once(alg, tensors)
    (case.loss_weight * loss).backward()
    grad0 = float(alg.policy.action_mean.grad[:, 6:12].mean().detach().item())

    optimizer = torch.optim.Adam(alg.parameters(), lr=case.lr)
    for _ in range(case.steps):
        optimizer.zero_grad()
        loss, metrics = _loss_once(alg, tensors)
        (case.loss_weight * loss).backward()
        optimizer.step()

    final_rho = torch.sigmoid(alg.policy.action_mean[:, 6:12]).mean().detach().item()
    return {
        "init_rho": init_rho,
        "final_rho": float(final_rho),
        "delta_rho": float(final_rho - init_rho),
        "grad0": grad0,
        "rho_loss": float(metrics["structured_joint_rl_rho_loss"]),
        "prior_loss": float(metrics["structured_joint_rl_prior_loss"]),
        "weighted_loss": float(case.loss_weight * loss.detach().item()),
        "act_mu": float(metrics["structured_joint_rl_rho_action_minus_mean_abs"]),
        "ratio": float(metrics["structured_joint_rl_ratio_mean"]),
    }


def run_rho_exploration_sweep() -> None:
    cases = [
        SweepCase("live_act_mu_prior", action_delta_raw=0.01, prior_weight=1.0, action_std=0.01),
        SweepCase("small_act_mu_prior", action_delta_raw=0.02, prior_weight=1.0, action_std=0.01),
        SweepCase("med_act_mu_prior", action_delta_raw=0.05, prior_weight=1.0, action_std=0.05),
        SweepCase("large_act_mu_prior", action_delta_raw=0.10, prior_weight=1.0, action_std=0.10),
        SweepCase("live_act_mu_no_prior", action_delta_raw=0.01, prior_weight=0.0, action_std=0.01),
        SweepCase("med_act_mu_no_prior", action_delta_raw=0.05, prior_weight=0.0, action_std=0.05),
    ]

    print("=== FrontRES Rho Exploration Sweep TEST ONLY ===")
    print("One run = 200 Adam steps at lr=6.5e-5, using formal structured-rho loss.")
    print("name                  std   act_mu  prior  init   final  delta    grad0      ratio  rho_loss prior")
    print("-" * 108)
    for case in cases:
        result = _run_case(case)
        print(
            f"{case.name:<21} "
            f"{case.action_std:>5.3f} "
            f"{result['act_mu']:>7.4f} "
            f"{case.prior_weight:>5.2f} "
            f"{result['init_rho']:>6.3f} "
            f"{result['final_rho']:>6.3f} "
            f"{result['delta_rho']:>+7.4f} "
            f"{result['grad0']:>+9.5f} "
            f"{result['ratio']:>6.3f} "
            f"{result['rho_loss']:>+8.4f} "
            f"{result['prior_loss']:>6.4f}"
        )

    print()
    print(
        "Readout: if live_act_mu_prior has a tiny delta, the formal loss is live but "
        "rho exploration/update is too weak for fast training.  If no_prior moves "
        "much more than prior, prior is the dominant brake."
    )


def run_rho_update_strength_sweep() -> None:
    """Sweep optimizer strength while keeping the live-like batch fixed.

    This is the next local check after the exploration sweep.  The recent live
    logs show a valid advantage sign mix but tiny rho movement.  If this table
    only moves rho when lr or loss_weight is raised, the bottleneck is update
    strength rather than reward construction.
    """

    cases = [
        SweepCase(
            "base_lr_w1",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=6.5e-5,
            loss_weight=1.0,
        ),
        SweepCase(
            "lr_2e4_w1",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=2.0e-4,
            loss_weight=1.0,
        ),
        SweepCase(
            "lr_5e4_w1",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=5.0e-4,
            loss_weight=1.0,
        ),
        SweepCase(
            "lr_1e3_w1",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=1.0e-3,
            loss_weight=1.0,
        ),
        SweepCase(
            "base_lr_w3",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=6.5e-5,
            loss_weight=3.0,
        ),
        SweepCase(
            "base_lr_w5",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=6.5e-5,
            loss_weight=5.0,
        ),
        SweepCase(
            "lr_2e4_w3",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=2.0e-4,
            loss_weight=3.0,
        ),
        SweepCase(
            "lr_5e4_w3",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=5.0e-4,
            loss_weight=3.0,
        ),
    ]

    print()
    print("=== FrontRES Rho Update Strength Sweep TEST ONLY ===")
    print("Fixed live-like batch: act_mu_raw=0.01, prior=1.0, std=0.01, steps=200.")
    print("name          lr       w    act_mu  init   final  delta    grad0      weighted_loss")
    print("-" * 91)
    for case in cases:
        result = _run_case(case)
        print(
            f"{case.name:<13} "
            f"{case.lr:>8.1e} "
            f"{case.loss_weight:>4.1f} "
            f"{result['act_mu']:>7.4f} "
            f"{result['init_rho']:>6.3f} "
            f"{result['final_rho']:>6.3f} "
            f"{result['delta_rho']:>+7.4f} "
            f"{result['grad0']:>+9.5f} "
            f"{result['weighted_loss']:>+13.4f}"
        )

    print()
    print(
        "Readout: this table isolates update strength.  If only higher lr or "
        "loss weight produces visible rho motion, the formal signal is present "
        "but too weak under the current optimizer scale."
    )


if __name__ == "__main__":
    run_rho_exploration_sweep()
    run_rho_update_strength_sweep()
