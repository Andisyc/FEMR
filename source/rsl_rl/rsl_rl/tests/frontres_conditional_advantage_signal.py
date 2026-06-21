# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: inspect whether FrontRES rho advantage carries conditional repair pressure.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_conditional_advantage_signal.py

This module does not start an environment.  It constructs hand-checkable
Noisy/Projected/Candidate executable scores and feeds them through the formal
structured-rho carrier.  The goal is to answer two debugging questions:

1. Candidate gain is high, but why can rho_adv be small?
2. In repairable samples, why can rho_adv signs be close to 50/50?
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.frontres.frontres_structured_rho import build_structured_rho_carrier


@dataclass(frozen=True)
class EvidenceRow:
    name: str
    noisy: float
    projected: float
    candidate: float
    expected: str


def _formal_rho_adv(rows: list[EvidenceRow], *, margin: float = 0.0) -> dict[str, torch.Tensor]:
    n = len(rows)
    device = torch.device("cpu")
    noisy = torch.tensor([row.noisy for row in rows], dtype=torch.float32, device=device)
    projected = torch.tensor([row.projected for row in rows], dtype=torch.float32, device=device)
    candidate = torch.tensor([row.candidate for row in rows], dtype=torch.float32, device=device)
    carrier = build_structured_rho_carrier(
        num_envs=n,
        n_exec=n,
        rho_current=torch.full((n, 6), 0.025, dtype=torch.float32, device=device),
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
        center_drive_deadzone=0.1,
        retention_weight=0.0,
        floor_penalty_weight=0.0,
        full_bonus_weight=0.0,
        joint_weight_floor=0.0,
        use_rho_update_weight=False,
        device=device,
    )
    formal_adv = carrier.rho_advantage_exec[:, 0].detach()
    candidate_gain = candidate - noisy
    projected_gain = projected - noisy
    missing_write_gain = candidate - projected
    candidate_pressure = missing_write_gain / (candidate_gain.abs() + margin + 1.0e-6)
    return {
        "noisy": noisy,
        "projected": projected,
        "candidate": candidate,
        "candidate_gain": candidate_gain,
        "projected_gain": projected_gain,
        "missing_write_gain": missing_write_gain,
        "formal_adv": formal_adv,
        "candidate_pressure": candidate_pressure.clamp(-1.0, 1.0),
    }


def run_single_case_table() -> None:
    rows = [
        EvidenceRow("candidate_high_projected_low", 0.50, 0.51, 0.60, "should strongly raise rho"),
        EvidenceRow("candidate_high_projected_high", 0.50, 0.58, 0.60, "rho already wrote useful repair"),
        EvidenceRow("candidate_bad_projected_low", 0.50, 0.49, 0.45, "should lower rho"),
        EvidenceRow("candidate_flat_projected_flat", 0.50, 0.50, 0.50, "no clear update"),
    ]
    out = _formal_rho_adv(rows)

    print()
    print("=== FrontRES Conditional Advantage Signal TEST ONLY ===")
    print("A. single hand-checkable rows")
    print("name                         noisy  proj   cand   cand_gain proj_gain formal_adv cand_pressure expected")
    print("-" * 122)
    for i, row in enumerate(rows):
        print(
            f"{row.name:<28} "
            f"{out['noisy'][i].item():>5.2f} "
            f"{out['projected'][i].item():>5.2f} "
            f"{out['candidate'][i].item():>5.2f} "
            f"{out['candidate_gain'][i].item():>+9.3f} "
            f"{out['projected_gain'][i].item():>+9.3f} "
            f"{out['formal_adv'][i].item():>+10.3f} "
            f"{out['candidate_pressure'][i].item():>+13.3f} "
            f"{row.expected}"
        )
    print()
    print(
        "Readout: formal_adv is the current training signal.  candidate_pressure is not used by training; "
        "it shows how much good Candidate evidence is left unwritten by the current Projected action."
    )


def run_repairable_mixture_table() -> None:
    rows: list[EvidenceRow] = []
    rows += [EvidenceRow("A_candidate_high_projected_low", 0.50, 0.51, 0.60, "raise rho") for _ in range(40)]
    rows += [EvidenceRow("B_candidate_high_projected_high", 0.50, 0.58, 0.60, "already high") for _ in range(20)]
    rows += [EvidenceRow("C_candidate_bad_projected_low", 0.50, 0.49, 0.45, "lower rho") for _ in range(35)]
    rows += [EvidenceRow("D_flat", 0.50, 0.50, 0.50, "zero") for _ in range(5)]
    out = _formal_rho_adv(rows)
    adv = out["formal_adv"]
    pressure = out["candidate_pressure"]

    print()
    print("B. repairable-like mixed batch")
    print(
        f"candidate_gain_mean={out['candidate_gain'].mean().item():+.4f} "
        f"projected_gain_mean={out['projected_gain'].mean().item():+.4f} "
        f"formal_adv_mean={adv.mean().item():+.4f} "
        f"candidate_pressure_mean={pressure.mean().item():+.4f}"
    )
    print(
        f"formal_adv sign pos/neg/zero: "
        f"{(adv > 1e-6).float().mean().item():.3f} / "
        f"{(adv < -1e-6).float().mean().item():.3f} / "
        f"{(adv.abs() <= 1e-6).float().mean().item():.3f}"
    )
    print(
        f"candidate_pressure sign pos/neg/zero: "
        f"{(pressure > 1e-6).float().mean().item():.3f} / "
        f"{(pressure < -1e-6).float().mean().item():.3f} / "
        f"{(pressure.abs() <= 1e-6).float().mean().item():.3f}"
    )
    print(
        "Readout: if candidate_pressure is strongly positive but formal_adv is weak or mixed, "
        "the current advantage is measuring Projected's already-written gain, not Candidate's write potential."
    )


def _train_two_state_policy(repair_adv: float, *, steps: int = 500, lr: float = 5.0e-2) -> tuple[float, float]:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    # Row 0: repairable state.  Row 1: boundary state.
    rho_adv = torch.tensor([[repair_adv], [0.0]], dtype=torch.float32)
    repairable_authority = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    boundary_authority = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    prior_target = torch.zeros(2, 1, dtype=torch.float32)
    model = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        rho = torch.sigmoid(model(features))
        repair_loss = (-rho_adv * rho * repairable_authority).sum() / repairable_authority.sum().clamp(min=1e-6)
        boundary_loss = ((rho - prior_target).pow(2) * boundary_authority).sum()
        boundary_loss = boundary_loss / boundary_authority.sum().clamp(min=1e-6)
        loss = repair_loss + boundary_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    rho = torch.sigmoid(model(features)).detach().view(-1)
    return float(rho[0].item()), float(rho[1].item())


def run_conditional_policy_toy() -> None:
    weak_repair_adv = 0.10
    strong_repair_adv = 0.90
    weak_repair_rho, weak_boundary_rho = _train_two_state_policy(weak_repair_adv)
    strong_repair_rho, strong_boundary_rho = _train_two_state_policy(strong_repair_adv)

    print()
    print("C. two-state conditional rho policy")
    print("signal             repair_adv  rho(repairable)  rho(boundary)")
    print("-" * 68)
    print(f"formal weak signal {weak_repair_adv:>10.3f} {weak_repair_rho:>16.3f} {weak_boundary_rho:>14.3f}")
    print(f"strong write signal{strong_repair_adv:>10.3f} {strong_repair_rho:>16.3f} {strong_boundary_rho:>14.3f}")
    print(
        "Readout: the model can represent conditional behavior.  If the repairable signal is too weak in live training, "
        "the learned policy can still become globally conservative even though the architecture is conditional."
    )


def main() -> None:
    run_single_case_table()
    run_repairable_mixture_table()
    run_conditional_policy_toy()


if __name__ == "__main__":
    main()
