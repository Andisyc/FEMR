# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: why legacy single-stage rho advantage can pass toys but fail long runs.

This test does not start IsaacLab.  It isolates two failure modes that the old
single-stage Advantage Learning checks did not falsify:

1. A clean positive/negative advantage toy can pass because the gradient sign is
   mechanically correct.
2. Real FrontRES authority is conditional on the proposal.  If two samples have
   the same state but different proposals, an obs-only rho head cannot assign
   one high rho and one low rho.  The gradients average and the harmful proposal
   is not rejected.

Run from the repository root with:

    frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_legacy_advantage_failure_mode.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _logit(p: float) -> float:
    p = min(1.0 - 1.0e-6, max(1.0e-6, float(p)))
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class ConditionalCase:
    name: str
    state: float
    proposal: float
    true_accept: float
    projected_gain: float
    harmful: float


class ObsOnlyRho(torch.nn.Module):
    """Legacy-style toy rho head: the same state must share the same rho."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(_logit(0.5))

    def forward(self, state: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
        del proposal
        return torch.sigmoid(self.linear(state))


class ProposalAwareRho(torch.nn.Module):
    """Stage-2 toy rho head: rho can depend on state and proposal."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(_logit(0.5))

    def forward(self, state: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(torch.cat([state, proposal], dim=-1)))


def _run_clean_advantage_sign_check() -> dict[str, float]:
    """The old small test: positive advantage raises rho, negative lowers rho."""

    raw = torch.nn.Parameter(torch.tensor([[0.0], [0.0]], dtype=torch.float32))
    action_rho = torch.tensor([[0.70], [0.70]], dtype=torch.float32)
    action_logit = torch.logit(action_rho)
    old_mu = torch.zeros_like(raw)
    sigma = torch.full_like(raw, 0.35)
    advantage = torch.tensor([[+1.0], [-1.0]], dtype=torch.float32)
    opt = torch.optim.Adam([raw], lr=0.03)

    for _ in range(80):
        new_logp = -0.5 * ((action_logit - raw) / sigma).pow(2)
        old_logp = -0.5 * ((action_logit - old_mu) / sigma).pow(2)
        ratio = torch.exp((new_logp - old_logp).clamp(-10.0, 10.0))
        loss = (-advantage * ratio).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    rho = torch.sigmoid(raw.detach())
    return {
        "positive_final": float(rho[0, 0].item()),
        "negative_final": float(rho[1, 0].item()),
    }


def _train_classifier(
    model: torch.nn.Module,
    cases: list[ConditionalCase],
    *,
    target_kind: str,
    steps: int = 300,
) -> dict[str, float]:
    state = torch.tensor([[c.state] for c in cases], dtype=torch.float32)
    proposal = torch.tensor([[c.proposal] for c in cases], dtype=torch.float32)
    true_accept = torch.tensor([[c.true_accept] for c in cases], dtype=torch.float32)
    projected_gain = torch.tensor([[c.projected_gain] for c in cases], dtype=torch.float32)
    harmful = torch.tensor([[c.harmful] for c in cases], dtype=torch.float32)

    if target_kind == "true_no_regret":
        target = true_accept
    elif target_kind == "projected_gain_only":
        # This is the dangerous compression: the proposal looks geometrically
        # useful, so the target says "write" even when rollout says harmful.
        target = (projected_gain > 0.0).to(dtype=torch.float32)
    else:
        raise ValueError(f"unknown target_kind: {target_kind}")

    opt = torch.optim.Adam(model.parameters(), lr=0.04)
    for _ in range(steps):
        rho = model(state, proposal)
        loss = F.binary_cross_entropy(rho.clamp(1e-5, 1.0 - 1e-5), target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        rho = model(state, proposal)
        loss = F.binary_cross_entropy(rho.clamp(1e-5, 1.0 - 1e-5), target)
        pred = (rho >= 0.5).to(dtype=torch.float32)
        acc = (pred == true_accept).to(dtype=torch.float32).mean()
        harmful_accept = (rho * harmful).sum() / harmful.sum().clamp(min=1.0)
    return {
        "rho_good": float(rho[0, 0].item()),
        "rho_harmful": float(rho[1, 0].item()),
        "loss": float(loss.item()),
        "true_acc": float(acc.item()),
        "harmful_accept": float(harmful_accept.item()),
    }


def run_legacy_advantage_failure_mode_check() -> None:
    torch.manual_seed(41)
    clean = _run_clean_advantage_sign_check()

    cases = [
        ConditionalCase(
            name="same_state_good_proposal",
            state=1.0,
            proposal=+1.0,
            true_accept=1.0,
            projected_gain=+0.20,
            harmful=0.0,
        ),
        ConditionalCase(
            name="same_state_harmful_proposal",
            state=1.0,
            proposal=-1.0,
            true_accept=0.0,
            projected_gain=+0.18,
            harmful=1.0,
        ),
    ]

    obs_only_true = _train_classifier(ObsOnlyRho(), cases, target_kind="true_no_regret")
    proposal_true = _train_classifier(ProposalAwareRho(), cases, target_kind="true_no_regret")
    proposal_gain_only = _train_classifier(
        ProposalAwareRho(),
        cases,
        target_kind="projected_gain_only",
    )

    print("=== FrontRES Legacy Advantage Failure Mode TEST ONLY ===")
    print("A. old mechanical gradient toy")
    print(
        f"positive_adv_final={clean['positive_final']:.3f}, "
        f"negative_adv_final={clean['negative_final']:.3f}"
    )
    print("meaning: the old test can pass because advantage sign can move rho.")
    print()
    print("B. conditional proposal aliasing")
    print("case                         state proposal true_accept projected_gain harmful")
    print("-" * 78)
    for c in cases:
        print(
            f"{c.name:<28} {c.state:+.1f}   {c.proposal:+.1f}      "
            f"{c.true_accept:.0f}           {c.projected_gain:+.2f}          {c.harmful:.0f}"
        )
    print()
    print("model/target                 rho_good rho_harmful true_acc harmful_accept")
    print("-" * 74)
    print(
        "obs_only + no_regret         "
        f"{obs_only_true['rho_good']:.3f}    {obs_only_true['rho_harmful']:.3f}      "
        f"{obs_only_true['true_acc']:.3f}    {obs_only_true['harmful_accept']:.3f}"
    )
    print(
        "proposal_aware + no_regret   "
        f"{proposal_true['rho_good']:.3f}    {proposal_true['rho_harmful']:.3f}      "
        f"{proposal_true['true_acc']:.3f}    {proposal_true['harmful_accept']:.3f}"
    )
    print(
        "proposal_aware + gain_only   "
        f"{proposal_gain_only['rho_good']:.3f}    {proposal_gain_only['rho_harmful']:.3f}      "
        f"{proposal_gain_only['true_acc']:.3f}    {proposal_gain_only['harmful_accept']:.3f}"
    )
    print()
    print(
        "readout: old sign tests do not prove long-run success.  If rho cannot see "
        "proposal, same-state good/harmful proposals alias.  If the target is only "
        "projected gain, even a proposal-aware rho head accepts the harmful sample."
    )

    if not (clean["positive_final"] > 0.54 and clean["negative_final"] < 0.46):
        raise AssertionError("mechanical advantage sign check did not move rho as expected.")
    if obs_only_true["harmful_accept"] < 0.35:
        raise AssertionError("obs-only rho unexpectedly separated same-state harmful proposal.")
    if proposal_true["harmful_accept"] > 0.10 or proposal_true["rho_good"] < 0.90:
        raise AssertionError("proposal-aware no-regret target failed to separate good/harmful proposals.")
    if proposal_gain_only["harmful_accept"] < 0.90:
        raise AssertionError("gain-only target did not expose harmful acceptance failure.")


if __name__ == "__main__":
    run_legacy_advantage_failure_mode_check()
