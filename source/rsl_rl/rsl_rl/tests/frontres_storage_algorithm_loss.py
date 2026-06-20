# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: FrontRES storage -> structured-rho algorithm loss check.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_storage_algorithm_loss.py

This module does not start an environment.  It constructs hand-checkable
FrontRES storage fields, reads them back through RolloutStorage's formal
minibatch generator, then feeds the minibatch into the formal structured-rho
loss implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified
from rsl_rl.storage.rollout_storage import RolloutStorage


NAMES = ["safe", "raise_rho", "lower_rho", "low_rho_good", "deep_broken"]


class FakePolicy:
    """Minimal policy surface needed by structured-rho loss."""

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
    """Minimal FrontRESUnified self object for the unbound loss method."""

    def __init__(self, device: torch.device):
        self.device = device
        self.policy = FakePolicy()
        self.clip_param = 0.2
        self.frontres_structured_joint_rl_enabled = True
        self.frontres_structured_joint_rl_weight = 1.0
        self.frontres_structured_joint_rl_adv_clip = 5.0
        self.frontres_structured_joint_rl_normalize_advantage = False
        self.frontres_structured_joint_prior_loss_weight = 1.0
        self.frontres_reward_compute_live_debug = False

    def _structured_joint_rl_enabled(self) -> bool:
        return bool(self.frontres_structured_joint_rl_enabled) and self.frontres_structured_joint_rl_weight > 0.0


def _sample_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    """Hand-checkable values copied from the Reward Compute debug harness."""

    rho_adv = torch.tensor([0.000, 0.472, -0.566, 1.000, 0.435], device=device).view(-1, 1)
    rho_mask = torch.ones_like(rho_adv)
    prior_authority = torch.tensor([0.550, 0.000, 0.000, 0.000, 1.000], device=device).view(-1, 1)
    policy_rho = torch.tensor([0.750, 0.750, 0.750, 0.250, 0.750], device=device).view(-1, 1)
    return {
        "rho_adv": rho_adv.expand(-1, 6).clone(),
        "rho_mask": rho_mask.expand(-1, 6).clone(),
        "prior_authority": prior_authority,
        "rho_prior": torch.zeros(len(NAMES), 6, device=device),
        "policy_mu": torch.logit(policy_rho.clamp(1e-6, 1.0 - 1e-6)).expand(-1, 6).clone(),
    }


def _build_storage(device: torch.device) -> RolloutStorage:
    values = _sample_tensors(device)
    n = len(NAMES)
    storage = RolloutStorage(
        training_type="frontres",
        num_envs=n,
        num_transitions_per_env=1,
        obs_shape=(4,),
        privileged_obs_shape=None,
        actions_shape=(12,),
        device=device,
    )
    storage.yield_batch_indices = True

    transition = RolloutStorage.Transition()
    transition.observations = torch.arange(n * 4, device=device, dtype=torch.float32).view(n, 4)
    transition.actions = torch.zeros(n, 12, device=device)
    transition.actions[:, 6:12] = torch.sigmoid(values["policy_mu"])
    transition.rewards = torch.zeros(n, device=device)
    transition.dones = torch.zeros(n, device=device)
    transition.values = torch.zeros(n, 1, device=device)
    transition.actions_log_prob = torch.zeros(n, 1, device=device)
    transition.action_mean = torch.zeros(n, 12, device=device)
    transition.action_mean[:, 6:12] = values["policy_mu"]
    transition.action_sigma = torch.ones(n, 12, device=device)
    transition.frontres_mask = torch.ones(n, 1, device=device)
    transition.frontres_actor_gate = torch.ones(n, 1, device=device)
    transition.supervised_target = torch.zeros(n, 6, device=device)
    transition.supervised_weight = torch.ones(n, 1, device=device)
    transition.supervised_harm_weight = torch.zeros(n, 1, device=device)
    transition.acceptance_target = values["rho_adv"]
    transition.acceptance_mask = values["rho_mask"]
    transition.rho_prior_authority = values["prior_authority"]
    transition.rho_prior_target = values["rho_prior"]
    transition.state_alpha_target = torch.zeros(n, 1, device=device)
    transition.state_alpha_mask = torch.zeros(n, 1, device=device)
    transition.hidden_states = None

    storage.add_transitions(transition)
    storage.returns.zero_()
    storage.advantages.zero_()
    return storage


def _unpack_frontres_batch(batch: tuple[Any, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        obs=batch[0],
        actions=batch[2],
        advantages=batch[4],
        old_actions_log_prob=batch[6],
        old_mu=batch[7],
        old_sigma=batch[8],
        acceptance_target=batch[22],
        acceptance_mask=batch[23],
        rho_prior_authority=batch[24],
        rho_prior_target=batch[25],
        batch_idx=batch[-1],
    )


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{name} mismatch\nactual={actual}\nexpected={expected}")


def run_storage_algorithm_loss_check() -> None:
    device = torch.device("cpu")
    storage = _build_storage(device)
    expected = _sample_tensors(device)
    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    minibatch = _unpack_frontres_batch(batch)
    order = minibatch.batch_idx.detach().cpu().tolist()

    expected_adv = expected["rho_adv"][minibatch.batch_idx]
    expected_mask = expected["rho_mask"][minibatch.batch_idx]
    expected_prior_authority = expected["prior_authority"][minibatch.batch_idx]
    expected_prior = expected["rho_prior"][minibatch.batch_idx]

    _assert_close("storage acceptance_target/rho_adv", minibatch.acceptance_target, expected_adv)
    _assert_close("storage acceptance_mask/rho_loss_mask", minibatch.acceptance_mask, expected_mask)
    _assert_close("storage rho_prior_authority", minibatch.rho_prior_authority, expected_prior_authority)
    _assert_close("storage rho_prior_target", minibatch.rho_prior_target, expected_prior)

    fake_alg = FakeAlgorithm(device)
    loss, metrics = FrontRESUnified._compute_structured_joint_rl_loss(
        fake_alg,
        minibatch.obs,
        minibatch.old_mu,
        minibatch.actions,
        minibatch.old_mu,
        minibatch.old_sigma,
        torch.zeros(len(NAMES), 1, device=device),
        minibatch.old_actions_log_prob,
        minibatch.acceptance_target,
        minibatch.acceptance_mask,
        minibatch.rho_prior_authority,
        minibatch.rho_prior_target,
        original_batch_size=len(NAMES),
    )

    rho_loss_expected = -expected["rho_adv"].mean()
    prior_weight = expected["prior_authority"].expand(-1, 6)
    prior_error = torch.sigmoid(expected["policy_mu"]).pow(2)
    prior_loss_expected = (prior_error * prior_weight).sum() / prior_weight.sum().clamp(min=1e-6)
    total_expected = rho_loss_expected + prior_loss_expected

    _assert_close("algorithm rho_loss", torch.tensor(metrics["structured_joint_rl_rho_loss"]), rho_loss_expected)
    _assert_close("algorithm prior_loss", torch.tensor(metrics["structured_joint_rl_prior_loss"]), prior_loss_expected)
    _assert_close("algorithm total_loss", loss.detach().cpu(), total_expected.cpu())

    print("=== FrontRES Storage -> Algorithm Loss TEST ONLY ===")
    print(f"minibatch_order: {[NAMES[i] for i in order]}")
    print("storage fields: PASS")
    print(
        "loss check: "
        f"rho_loss={metrics['structured_joint_rl_rho_loss']:+.4f}, "
        f"prior_loss={metrics['structured_joint_rl_prior_loss']:.4f}, "
        f"total={loss.item():+.4f}"
    )
    print(
        "meaning: positive rho_adv lowers PPO loss; negative rho_adv raises it; "
        "safe/deep_broken prior acts only through separate regularization."
    )


if __name__ == "__main__":
    run_storage_algorithm_loss_check()
