#!/usr/bin/env python3
"""TEST ONLY: active FEMR algorithm loss is masked HSL acceptance BCE.

This does not start an environment. It directly checks the Step-6 contract:
acceptance labels from storage train only the acceptance logits, while generic
PPO/structured-rho semantics are not required for this active loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified


class FakePolicy:
    task_conf_dim = 6
    num_task_corrections = 6


class FakeAlgorithm:
    def __init__(self, device: torch.device):
        self.device = device
        self.policy = FakePolicy()
        self.frontres_training_objective = "hsl_hybrid"
        self.frontres_acceptance_preference_weight = 1.0
        self.frontres_acceptance_preference_focal_gamma = 0.0
        self.frontres_acceptance_preference_balance_min = 1.0
        self.frontres_acceptance_preference_balance_max = 1.0
        self.frontres_active_task_dims = None
        self.frontres_structured_joint_rl_enabled = False
        self.frontres_structured_joint_rl_weight = 0.0
        self.frontres_structured_joint_rl_keep_legacy_bce = False
        self.frontres_authority_actor_critic_enabled = False
        self.frontres_authority_actor_loss_weight = 0.0
        self.frontres_authority_critic_loss_weight = 0.0
        self.ppo_actor_weight = 1.0

    def _structured_joint_rl_enabled(self) -> bool:
        return FrontRESUnified._structured_joint_rl_enabled(self)

    def _ppo_acceptance_only_mode(self) -> bool:
        return FrontRESUnified._ppo_acceptance_only_mode(self)

    def _authority_actor_critic_enabled(self) -> bool:
        return FrontRESUnified._authority_actor_critic_enabled(self)

    def _active_hsl_acceptance_loss_enabled(self) -> bool:
        return FrontRESUnified._active_hsl_acceptance_loss_enabled(self)


def run_hsl_acceptance_algorithm_loss_check() -> None:
    device = torch.device("cpu")
    alg = FakeAlgorithm(device)
    n = 4
    mu = torch.zeros(n, 12, device=device, requires_grad=True)
    logits = torch.tensor(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
            [0.5, -0.5, 1.5, -1.5, 2.5, -2.5],
            [-4.0, -3.0, -2.0, 2.0, 3.0, 4.0],
        ],
        device=device,
    )
    with torch.no_grad():
        mu[:, 6:12] = logits

    acceptance_gt = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        device=device,
    )
    acceptance_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

    loss, metrics = FrontRESUnified._compute_acceptance_preference_loss(
        alg,
        mu,
        acceptance_gt,
        acceptance_mask,
        original_batch_size=n,
    )
    expected_terms = F.binary_cross_entropy_with_logits(logits, acceptance_gt, reduction="none")
    expected = (expected_terms * acceptance_mask).sum() / acceptance_mask.sum().clamp(min=1e-6)
    torch.testing.assert_close(loss.detach(), expected.detach())

    if metrics["hsl_acceptance_loss_enabled"] != 1.0:
        raise AssertionError("active HSL acceptance loss did not report enabled")
    if metrics["acceptance_preference_mask_frac"] != 0.75:
        raise AssertionError(f"unexpected mask frac: {metrics['acceptance_preference_mask_frac']}")
    if not alg._active_hsl_acceptance_loss_enabled():
        raise AssertionError("active HSL acceptance helper should be enabled")
    if alg._structured_joint_rl_enabled():
        raise AssertionError("structured-rho must stay inactive in active FEMR Step 6")
    if alg._authority_actor_critic_enabled():
        raise AssertionError("authority actor-critic must stay inactive in active FEMR Step 6")

    loss.backward()
    if float(mu.grad[:, :6].abs().sum().item()) != 0.0:
        raise AssertionError("acceptance BCE leaked gradient into proposal logits")
    if float(mu.grad[:, 6:12].abs().sum().item()) <= 0.0:
        raise AssertionError("acceptance BCE produced no gradient on acceptance logits")
    if float(mu.grad[2, 6:12].abs().sum().item()) != 0.0:
        raise AssertionError("masked sample should not receive acceptance gradient")

    print("=== FrontRES HSL Acceptance Algorithm Loss TEST ONLY ===")
    print(f"loss={float(loss.detach()):.6f} expected={float(expected.detach()):.6f}")
    print(
        "metrics: "
        f"enabled={metrics['hsl_acceptance_loss_enabled']:.0f}, "
        f"mask={metrics['hsl_acceptance_mask_frac']:.3f}, "
        f"gt={metrics['hsl_acceptance_gt_mean']:.3f}, "
        f"prob={metrics['hsl_acceptance_prob_mean']:.3f}, "
        f"err={metrics['hsl_acceptance_abs_err']:.3f}"
    )
    print("gradient: proposal=0, acceptance>0, masked_sample=0")
    print("result: PASS")


def run_zero_mask_acceptance_loss_skip_check() -> None:
    device = torch.device("cpu")
    alg = FakeAlgorithm(device)
    n = 4
    mu = torch.zeros(n, 12, device=device, requires_grad=True)
    acceptance_gt = torch.ones(n, 6, device=device)
    acceptance_mask = torch.zeros(n, 6, device=device)

    loss, metrics = FrontRESUnified._compute_acceptance_preference_loss(
        alg,
        mu,
        acceptance_gt,
        acceptance_mask,
        original_batch_size=n,
    )
    if loss.requires_grad:
        raise AssertionError("zero-mask acceptance loss should be a skippable constant")
    if metrics["hsl_acceptance_loss_enabled"] != 0.0:
        raise AssertionError("zero-mask acceptance loss should not report enabled")

    base_loss = mu[:, :6].square().mean()
    base_loss.backward(retain_graph=True)
    base_grad = mu.grad.detach().clone()
    if loss.requires_grad:
        loss.backward()
    else:
        mu.grad = base_grad.clone()
    torch.testing.assert_close(mu.grad, base_grad)
    print("zero-mask acceptance loss: skippable without backward")


if __name__ == "__main__":
    run_hsl_acceptance_algorithm_loss_check()
    run_zero_mask_acceptance_loss_skip_check()
