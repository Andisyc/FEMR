# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: legacy FrontRES structured-rho PPO-clipped advantage check.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_advantage_learning.py

This module does not start an environment.  It explicitly exercises the legacy
`ppo_clipped` structured-rho path.  The current active FrontRES path is
`region_direct + bce_logit`, covered by frontres_rho_low_recovery_mechanism.py
and frontres_storage_algorithm_loss.py.
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
class AdvantageCase:
    name: str
    action_rho: float
    rho_advantage: float
    rho_weight: float
    prior_authority: float
    prior_target: float
    expected: str


class LearnableRhoPolicy(torch.nn.Module):
    """Minimal differentiable policy surface needed by structured-rho loss."""

    task_conf_dim = 6

    def __init__(self, batch_size: int, action_dim: int = 12, sigma: float = 0.35):
        super().__init__()
        self.action_mean = torch.nn.Parameter(torch.zeros(batch_size, action_dim))
        self.register_buffer("action_std", torch.full((batch_size, action_dim), float(sigma)))

    def get_actions_log_prob_per_dim(self, actions: torch.Tensor, dims: list[int]) -> torch.Tensor:
        mu = self.action_mean[:, dims]
        sigma = self.action_std[:, dims].clamp(min=1e-6)
        # Formal FrontRES stores rho actions after sigmoid, while action_mean
        # remains in raw Gaussian/logit space.
        action = actions[:, dims].clamp(1e-6, 1.0 - 1e-6)
        action = torch.log(action / (1.0 - action))
        return -0.5 * ((action - mu) / sigma).pow(2) - torch.log(sigma)

    def get_actions_log_prob_per_dim_from_stats(
        self,
        actions: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        dims: list[int],
    ) -> torch.Tensor:
        mu = old_mu[:, dims]
        sigma = old_sigma[:, dims].clamp(min=1e-6)
        action = actions[:, dims].clamp(1e-6, 1.0 - 1e-6)
        action = torch.log(action / (1.0 - action))
        return -0.5 * ((action - mu) / sigma).pow(2) - torch.log(sigma)


class LearnableAlgorithm(torch.nn.Module):
    """Minimal FrontRESUnified self object with a learnable policy."""

    def __init__(self, cases: list[AdvantageCase], prior_weight: float):
        super().__init__()
        self.device = torch.device("cpu")
        self.policy = LearnableRhoPolicy(len(cases))
        self.clip_param = 0.2
        self.frontres_structured_joint_rl_enabled = True
        self.frontres_structured_joint_rl_weight = 1.0
        self.frontres_structured_joint_rl_adv_clip = 5.0
        self.frontres_structured_joint_rl_normalize_advantage = False
        self.frontres_structured_joint_rl_loss_mode = "ppo_clipped"
        self.frontres_structured_joint_prior_loss_weight = float(prior_weight)
        self.frontres_reward_compute_live_debug = False

    def _structured_joint_rl_enabled(self) -> bool:
        return bool(self.frontres_structured_joint_rl_enabled) and self.frontres_structured_joint_rl_weight > 0.0


def _build_cases() -> list[AdvantageCase]:
    return [
        AdvantageCase(
            name="raise_rho",
            action_rho=0.70,
            rho_advantage=1.0,
            rho_weight=1.0,
            prior_authority=0.0,
            prior_target=0.0,
            expected="increase",
        ),
        AdvantageCase(
            name="lower_rho",
            action_rho=0.70,
            rho_advantage=-1.0,
            rho_weight=1.0,
            prior_authority=0.0,
            prior_target=0.0,
            expected="decrease",
        ),
        AdvantageCase(
            name="dead_at_mean",
            action_rho=0.50,
            rho_advantage=1.0,
            rho_weight=1.0,
            prior_authority=0.0,
            prior_target=0.0,
            expected="stay",
        ),
        AdvantageCase(
            name="safe_prior",
            action_rho=0.50,
            rho_advantage=0.0,
            rho_weight=1.0,
            prior_authority=1.0,
            prior_target=0.0,
            expected="decrease",
        ),
        AdvantageCase(
            name="masked",
            action_rho=0.70,
            rho_advantage=1.0,
            rho_weight=0.0,
            prior_authority=0.0,
            prior_target=0.0,
            expected="stay",
        ),
    ]


def _case_tensors(cases: list[AdvantageCase]) -> dict[str, torch.Tensor]:
    n = len(cases)
    actions = torch.zeros(n, 12)
    rho_advantage = torch.zeros(n, 6)
    rho_weight = torch.zeros(n, 6)
    prior_authority = torch.zeros(n, 1)
    prior_target = torch.zeros(n, 6)

    for i, case in enumerate(cases):
        actions[i, 6:12] = case.action_rho
        rho_advantage[i, :] = case.rho_advantage
        rho_weight[i, :] = case.rho_weight
        prior_authority[i, 0] = case.prior_authority
        prior_target[i, :] = case.prior_target

    return {
        "obs": torch.ones(n, 4),
        "actions": actions,
        "old_mu": torch.zeros(n, 12),
        "old_sigma": torch.full((n, 12), 0.35),
        "old_logp": torch.zeros(n, 1),
        "new_logp": torch.zeros(n, 1),
        "rho_advantage": rho_advantage,
        "rho_weight": rho_weight,
        "prior_authority": prior_authority,
        "prior_target": prior_target,
    }


def _check_expectation(case: AdvantageCase, initial: float, final: float) -> bool:
    if case.expected == "increase":
        return final > initial + 0.03
    if case.expected == "decrease":
        return final < initial - 0.03
    if case.expected == "stay":
        return abs(final - initial) < 0.02
    raise ValueError(f"unknown expectation: {case.expected}")


def run_advantage_learning_check() -> None:
    torch.manual_seed(7)
    cases = _build_cases()
    tensors = _case_tensors(cases)
    alg = LearnableAlgorithm(cases, prior_weight=1.0)
    optimizer = torch.optim.Adam(alg.parameters(), lr=0.02)

    initial_rho = torch.sigmoid(alg.policy.action_mean[:, 6]).detach().clone()
    final_metrics: dict[str, float] = {}

    for _ in range(80):
        optimizer.zero_grad()
        loss, metrics = FrontRESUnified._compute_structured_joint_rl_loss(
            alg,
            tensors["obs"],
            alg.policy.action_mean,
            tensors["actions"],
            tensors["old_mu"],
            tensors["old_sigma"],
            tensors["new_logp"],
            tensors["old_logp"],
            tensors["rho_advantage"],
            tensors["rho_weight"],
            tensors["prior_authority"],
            tensors["prior_target"],
            original_batch_size=len(cases),
        )
        loss.backward()
        optimizer.step()
        final_metrics = metrics

    final_rho = torch.sigmoid(alg.policy.action_mean[:, 6]).detach()

    print("=== FrontRES Advantage Learning TEST ONLY ===")
    print("This checks whether formal structured-rho loss can move rho away from 0.5.")
    print("name           rho_act adv   mask  prior_auth  prior_gt  initial  final   expected   result")
    print("-" * 96)
    failures: list[str] = []
    for i, case in enumerate(cases):
        initial = float(initial_rho[i].item())
        final = float(final_rho[i].item())
        passed = _check_expectation(case, initial, final)
        if not passed:
            failures.append(case.name)
        print(
            f"{case.name:<14} {case.action_rho:+.2f}  {case.rho_advantage:+.2f}  "
            f"{case.rho_weight:.1f}   {case.prior_authority:.1f}         {case.prior_target:.1f}      "
            f"{initial:.3f}    {final:.3f}   {case.expected:<9} {'PASS' if passed else 'FAIL'}"
        )
    print()
    print(
        "loss metrics: "
        f"rho_loss={final_metrics.get('structured_joint_rl_rho_loss', 0.0):+.4f}, "
        f"prior_loss={final_metrics.get('structured_joint_rl_prior_loss', 0.0):.4f}, "
        f"ratio={final_metrics.get('structured_joint_rl_rho_ratio_mean', 0.0):.3f}"
    )
    print(
        "meaning: raise/lower cases test PPO advantage direction; safe_prior tests the "
        "separate prior pull; dead_at_mean proves that advantage alone needs a sampled "
        "rho action away from the current mean."
    )
    if failures:
        raise AssertionError(f"Advantage learning check failed for: {', '.join(failures)}")


if __name__ == "__main__":
    run_advantage_learning_check()
