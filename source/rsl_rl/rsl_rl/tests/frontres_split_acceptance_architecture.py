# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: FrontRES split acceptance architecture contract.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_split_acceptance_architecture.py

This test does not load GMT and does not start IsaacLab.  It constructs the
minimal pieces needed to call ``FrontRESActorCritic._frontres_raw_task_output``.

It checks the proposal-conditioned acceptance contract:

1. Stage-2 acceptance receives full/current-state observations plus the
   detached bounded Stage-1 Delta SE proposal.
2. An acceptance-only loss does not backpropagate into the Stage-1 proposal
   actor through the proposal feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic


class ProposalActor(nn.Module):
    """Small proposal network returning raw [Delta pos, Delta rpy, unused coeff]."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, 12, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            # Make proposal features nonzero and input-dependent.
            for row in range(6):
                self.linear.weight[row, row % obs_dim] = 0.25 + 0.05 * row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RecordingAcceptanceActor(nn.Module):
    """Acceptance head that records the exact input it receives."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.last_input: torch.Tensor | None = None
        with torch.no_grad():
            self.linear.weight.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        return self.linear(x)


def _build_minimal_policy(
    *,
    policy_obs_dim: int,
    full_obs_dim: int,
    proposal_dim: int = 6,
    acceptance_dim: int = 6,
) -> FrontRESActorCritic:
    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    nn.Module.__init__(policy)
    policy.num_task_corrections = proposal_dim
    policy.task_conf_dim = acceptance_dim
    policy.frontres_split_acceptance_head = True
    policy.max_delta_pos = 0.30
    policy.max_delta_rpy = 0.40
    policy.residual_actor = ProposalActor(policy_obs_dim)
    policy.acceptance_actor = RecordingAcceptanceActor(full_obs_dim + proposal_dim, acceptance_dim)
    return policy


def run_split_acceptance_architecture_check() -> None:
    torch.manual_seed(7)
    batch = 5
    policy_obs_dim = 8
    full_obs_dim = 13

    policy = _build_minimal_policy(policy_obs_dim=policy_obs_dim, full_obs_dim=full_obs_dim)
    policy_obs = torch.randn(batch, policy_obs_dim)
    full_obs = torch.randn(batch, full_obs_dim)
    policy._cached_full_policy_obs = full_obs

    raw = FrontRESActorCritic._frontres_raw_task_output(policy, policy_obs)
    acceptance_actor = policy.acceptance_actor
    assert isinstance(acceptance_actor, RecordingAcceptanceActor)
    assert acceptance_actor.last_input is not None

    raw_from_proposal_actor = policy.residual_actor(policy_obs)
    expected_proposal = torch.cat(
        [
            torch.tanh(raw_from_proposal_actor[:, :3]) * policy.max_delta_pos,
            torch.tanh(raw_from_proposal_actor[:, 3:6]) * policy.max_delta_rpy,
        ],
        dim=-1,
    )
    acceptance_input = acceptance_actor.last_input
    received_full_obs = acceptance_input[:, :full_obs_dim]
    received_proposal = acceptance_input[:, full_obs_dim:]

    torch.testing.assert_close(raw[:, :6], raw_from_proposal_actor[:, :6])
    torch.testing.assert_close(received_full_obs, full_obs)
    torch.testing.assert_close(received_proposal, expected_proposal)
    if received_proposal.requires_grad:
        raise AssertionError("Acceptance proposal feature must be detached from Stage-1 proposal graph.")

    policy.zero_grad(set_to_none=True)
    acceptance_only_loss = raw[:, 6:].sum()
    acceptance_only_loss.backward()

    proposal_grad = policy.residual_actor.linear.weight.grad
    acceptance_grad = policy.acceptance_actor.linear.weight.grad
    if proposal_grad is not None and float(proposal_grad.abs().max().item()) > 0.0:
        raise AssertionError("Acceptance-only loss leaked gradient into Stage-1 proposal actor.")
    if acceptance_grad is None or float(acceptance_grad.abs().sum().item()) <= 0.0:
        raise AssertionError("Acceptance-only loss did not train Stage-2 acceptance actor.")

    print("=== FrontRES Split Acceptance Architecture TEST ONLY ===")
    print("result: PASS")
    print(f"acceptance_input_dim={acceptance_input.shape[-1]} = full_obs({full_obs_dim}) + proposal(6)")
    print("proposal_feature=detached bounded Delta SE")
    print("acceptance_loss_grad: Stage-2 yes, Stage-1 no")


if __name__ == "__main__":
    run_split_acceptance_architecture_check()
