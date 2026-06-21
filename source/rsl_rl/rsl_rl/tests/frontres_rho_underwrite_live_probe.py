# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: probe whether formal FrontRES rho underwrite can rescue low rho.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_rho_underwrite_live_probe.py

This does not start IsaacLab or an environment.  It feeds hand-checkable
Noisy/Projected/Candidate executable scores into the formal structured-rho
carrier, then applies the same region-direct rho loss shape used by the live
algorithm.  The test isolates two questions before spending time on short runs:

1. If Candidate is good but current Projected writes only a tiny correction,
   does the formal underwrite signal push rho upward?
2. If positive and negative evidence are mixed, can a policy learn different
   rho values only when the state features separate those cases?
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.frontres.frontres_structured_rho import build_structured_rho_carrier


@dataclass(frozen=True)
class ProbeCase:
    name: str
    noisy: float
    projected: float
    candidate: float
    prior_authority: float
    prior_gt: float
    expected: str


def _logit(p: float) -> float:
    p = min(1.0 - 1.0e-6, max(1.0e-6, float(p)))
    return math.log(p / (1.0 - p))


def _build_formal_adv(
    cases: list[ProbeCase],
    *,
    underwrite_weight: float,
    margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    n = len(cases)
    device = torch.device("cpu")
    noisy = torch.tensor([c.noisy for c in cases], dtype=torch.float32, device=device)
    projected = torch.tensor([c.projected for c in cases], dtype=torch.float32, device=device)
    candidate = torch.tensor([c.candidate for c in cases], dtype=torch.float32, device=device)

    carrier = build_structured_rho_carrier(
        num_envs=n,
        n_exec=n,
        rho_current=torch.full((n, 6), 0.04, dtype=torch.float32, device=device),
        rho_dim_weight=torch.ones(n, 6, dtype=torch.float32, device=device),
        rho_update_weight=torch.ones(n, 1, dtype=torch.float32, device=device),
        exec_perturbed=noisy,
        exec_feasible=candidate,
        exec_frontres=projected,
        exec_candidate=candidate,
        state_alpha_target=torch.zeros(n, 1, dtype=torch.float32, device=device),
        live_alpha=None,
        rho_space="noisy_to_repair",
        grouped_targets_enabled=False,
        feasible_components=None,
        candidate_planar=None,
        candidate_rp=None,
        candidate_z=None,
        projected_planar=None,
        projected_rp=None,
        projected_z=None,
        base_planar=None,
        base_rp=None,
        base_z=None,
        pref_margin=margin,
        rho_floor=0.0,
        directional_weight=1.0,
        rho_center=0.5,
        center_drive_deadzone=0.10,
        retention_weight=0.0,
        floor_penalty_weight=0.0,
        full_bonus_weight=0.0,
        underwrite_weight=float(underwrite_weight),
        joint_weight_floor=0.0,
        use_rho_update_weight=False,
        device=device,
    )
    rollout_gain = projected - noisy
    evidence_scale = (candidate - noisy).abs() + float(margin) + 1.0e-6
    raw_direction = (rollout_gain / evidence_scale).clamp(-1.0, 1.0)
    raw_direction = torch.where(
        rollout_gain.abs() > float(margin),
        raw_direction,
        torch.zeros_like(raw_direction),
    )
    accept_from_projected = (rollout_gain > float(margin)).to(raw_direction.dtype)
    underwrite = torch.relu(candidate - projected - float(margin))
    underwrite_direction = (underwrite / evidence_scale).clamp(0.0, 1.0)
    prior_authority = torch.tensor([c.prior_authority for c in cases], dtype=torch.float32, device=device).view(n, 1)
    prior_gt = torch.tensor([c.prior_gt for c in cases], dtype=torch.float32, device=device).view(n, 1)
    repairable_authority = (1.0 - prior_authority).clamp(0.0, 1.0)
    return {
        "noisy": noisy,
        "projected": projected,
        "candidate": candidate,
        "candidate_gain": candidate - noisy,
        "projected_gain": projected - noisy,
        "missing_write": candidate - projected,
        "adv": carrier.rho_advantage_exec[:, :1].detach(),
        "raw": raw_direction.view(n, 1).detach(),
        "underwrite": underwrite_direction.view(n, 1).detach(),
        "accept": accept_from_projected.view(n, 1).detach(),
        "weight": carrier.rho_validity_weight_exec[:, :1].detach(),
        "prior_authority": prior_authority,
        "repairable_authority": repairable_authority,
        "prior_gt": prior_gt,
    }


def _region_direct_update(
    adv: torch.Tensor,
    repairable_authority: torch.Tensor,
    prior_authority: torch.Tensor,
    prior_gt: torch.Tensor,
    *,
    steps: int = 80,
    lr: float = 0.20,
    init_rho: float = 0.04,
    shared_feature: bool = False,
) -> torch.Tensor:
    """Tiny optimizer using the live region-direct rho loss shape."""

    n = int(adv.shape[0])
    if shared_feature:
        features = torch.ones(n, 1)
        model = torch.nn.Linear(1, 1, bias=False)
    else:
        features = torch.eye(n)
        model = torch.nn.Linear(n, 1, bias=False)
    torch.nn.init.constant_(model.weight, _logit(init_rho))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        rho = torch.sigmoid(model(features))
        repair_weight = repairable_authority
        boundary_weight = prior_authority
        repair_loss = (-adv * rho * repair_weight).sum() / repair_weight.sum().clamp(min=1.0e-6)
        boundary_loss = ((rho - prior_gt).pow(2) * boundary_weight).sum()
        boundary_loss = boundary_loss / boundary_weight.sum().clamp(min=1.0e-6)
        loss = repair_loss + boundary_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.sigmoid(model(features)).detach().view(-1)


def run_case_table() -> list[ProbeCase]:
    cases = [
        ProbeCase(
            "low_rho_good_candidate",
            noisy=0.50,
            projected=0.505,
            candidate=0.60,
            prior_authority=0.0,
            prior_gt=0.0,
            expected="rho should rise: Candidate is good and Projected is slightly positive",
        ),
        ProbeCase(
            "low_rho_harmful_projected",
            noisy=0.50,
            projected=0.495,
            candidate=0.60,
            prior_authority=0.0,
            prior_gt=0.0,
            expected="rho should not rise: current-state Projected is harmful",
        ),
        ProbeCase(
            "clear_repair",
            noisy=0.50,
            projected=0.56,
            candidate=0.60,
            prior_authority=0.0,
            prior_gt=0.0,
            expected="rho should rise strongly",
        ),
        ProbeCase(
            "safe_boundary",
            noisy=0.95,
            projected=0.951,
            candidate=0.96,
            prior_authority=1.0,
            prior_gt=0.0,
            expected="boundary prior should pull rho down",
        ),
    ]
    print("=== FrontRES Rho Underwrite Live Probe TEST ONLY ===")
    print("A. formal carrier output")
    print(
        "name                        noisy  proj   cand   cand_gain proj_gain raw_adv under accept final_adv expected"
    )
    print("-" * 154)
    out = _build_formal_adv(cases, underwrite_weight=0.25)
    for i, case in enumerate(cases):
        print(
            f"{case.name:<27} "
            f"{out['noisy'][i].item():>5.3f} "
            f"{out['projected'][i].item():>5.3f} "
            f"{out['candidate'][i].item():>5.3f} "
            f"{out['candidate_gain'][i].item():>+9.3f} "
            f"{out['projected_gain'][i].item():>+9.3f} "
            f"{out['raw'][i, 0].item():>+7.3f} "
            f"{out['underwrite'][i, 0].item():>5.3f} "
            f"{out['accept'][i, 0].item():>6.1f} "
            f"{out['adv'][i, 0].item():>+9.3f} "
            f"{case.expected}"
        )
    print()
    print(
        "Readout: final_adv is the formal signal written to storage.  It should be positive for "
        "low_rho_good_candidate, negative for low_rho_harmful_projected, and near boundary-prior-only "
        "for safe_boundary."
    )
    return cases


def run_underwrite_sweep(cases: list[ProbeCase]) -> None:
    print()
    print("B. underwrite_weight sweep")
    print("weight  low_good_adv  harmful_adv  clear_repair_adv  low_good_rho_after  harmful_rho_after")
    print("-" * 96)
    for weight in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
        out = _build_formal_adv(cases, underwrite_weight=weight)
        rho_after = _region_direct_update(
            out["adv"],
            out["repairable_authority"],
            out["prior_authority"],
            out["prior_gt"],
        )
        print(
            f"{weight:>6.2f} "
            f"{out['adv'][0, 0].item():>13.3f} "
            f"{out['adv'][1, 0].item():>12.3f} "
            f"{out['adv'][2, 0].item():>17.3f} "
            f"{rho_after[0].item():>18.3f} "
            f"{rho_after[1].item():>17.3f}"
        )
    print()
    print(
        "Readout: if low_good_rho_after barely moves even when final_adv is positive, the problem is "
        "loss/update strength.  If harmful_rho_after rises, the underwrite rule is too permissive."
    )


def run_mixed_batch_probe() -> None:
    cases = []
    for _ in range(25):
        cases.append(ProbeCase("good", 0.50, 0.505, 0.60, 0.0, 0.0, "good"))
    for _ in range(25):
        cases.append(ProbeCase("harmful", 0.50, 0.495, 0.60, 0.0, 0.0, "harmful"))
    out = _build_formal_adv(cases, underwrite_weight=0.25)
    adv = out["adv"].view(-1)
    shared_rho = _region_direct_update(
        out["adv"],
        out["repairable_authority"],
        out["prior_authority"],
        out["prior_gt"],
        shared_feature=True,
    )
    separate_rho = _region_direct_update(
        out["adv"],
        out["repairable_authority"],
        out["prior_authority"],
        out["prior_gt"],
        shared_feature=False,
    )
    print()
    print("C. mixed good/harmful evidence")
    print(
        f"adv_mean={adv.mean().item():+.4f} "
        f"adv_pos/neg/zero={(adv > 1e-6).float().mean().item():.3f}/"
        f"{(adv < -1e-6).float().mean().item():.3f}/"
        f"{(adv.abs() <= 1e-6).float().mean().item():.3f}"
    )
    print(
        f"shared_feature_rho_mean={shared_rho.mean().item():.3f} "
        f"separate_feature_good_rho={separate_rho[:25].mean().item():.3f} "
        f"separate_feature_harmful_rho={separate_rho[25:].mean().item():.3f}"
    )
    print()
    print(
        "Readout: shared_feature is the 'policy cannot tell states apart' case.  separate_feature is the "
        "'current state contains enough information' case.  If only separate_feature succeeds, then the "
        "method is conceptually viable but the live observation/features may not expose the condition clearly."
    )


def main() -> None:
    cases = run_case_table()
    run_underwrite_sweep(cases)
    run_mixed_batch_probe()


if __name__ == "__main__":
    main()
