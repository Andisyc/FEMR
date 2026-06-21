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


def _formal_rho_adv(
    rows: list[EvidenceRow],
    *,
    margin: float = 0.0,
    underwrite_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
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
    evidence_scale = candidate_gain.abs() + margin + 1.0e-6
    candidate_pressure = missing_write_gain / evidence_scale
    # Candidate should not override current-state execution evidence.  This
    # test-only proposal keeps Projected-vs-Noisy as the sign source, and uses
    # Candidate-vs-Projected only when Projected already improved over Noisy.
    accept_from_projected = (projected_gain > margin).to(formal_adv.dtype)
    underwrite_adv = torch.relu(missing_write_gain - margin) / evidence_scale
    proposed_adv = (formal_adv + float(underwrite_weight) * accept_from_projected * underwrite_adv).clamp(
        -1.0, 1.0
    )
    return {
        "noisy": noisy,
        "projected": projected,
        "candidate": candidate,
        "candidate_gain": candidate_gain,
        "projected_gain": projected_gain,
        "missing_write_gain": missing_write_gain,
        "formal_adv": formal_adv,
        "accept_from_projected": accept_from_projected,
        "underwrite_adv": underwrite_adv.clamp(0.0, 1.0),
        "proposed_adv": proposed_adv,
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
    print("name                         noisy  proj   cand   cand_gain proj_gain formal_adv proposed_adv expected")
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
            f"{out['proposed_adv'][i].item():>+12.3f} "
            f"{row.expected}"
        )
    print()
    print(
        "Readout: formal_adv is the current training signal.  proposed_adv is test-only.  It keeps the "
        "Projected sign, but adds Candidate under-write pressure only when Projected already improved over Noisy."
    )


def run_repairable_mixture_table() -> None:
    rows: list[EvidenceRow] = []
    rows += [EvidenceRow("A_candidate_high_projected_low", 0.50, 0.51, 0.60, "raise rho") for _ in range(40)]
    rows += [EvidenceRow("B_candidate_high_projected_high", 0.50, 0.58, 0.60, "already high") for _ in range(20)]
    rows += [EvidenceRow("C_candidate_bad_projected_low", 0.50, 0.49, 0.45, "lower rho") for _ in range(35)]
    rows += [EvidenceRow("D_flat", 0.50, 0.50, 0.50, "zero") for _ in range(5)]
    out = _formal_rho_adv(rows)
    adv = out["formal_adv"]
    proposed = out["proposed_adv"]
    pressure = out["candidate_pressure"]

    print()
    print("B. repairable-like mixed batch")
    print(
        f"candidate_gain_mean={out['candidate_gain'].mean().item():+.4f} "
        f"projected_gain_mean={out['projected_gain'].mean().item():+.4f} "
        f"formal_adv_mean={adv.mean().item():+.4f} "
        f"proposed_adv_mean={proposed.mean().item():+.4f} "
        f"candidate_pressure_mean={pressure.mean().item():+.4f}"
    )
    print(
        f"formal_adv sign pos/neg/zero: "
        f"{(adv > 1e-6).float().mean().item():.3f} / "
        f"{(adv < -1e-6).float().mean().item():.3f} / "
        f"{(adv.abs() <= 1e-6).float().mean().item():.3f}"
    )
    print(
        f"proposed_adv sign pos/neg/zero: "
        f"{(proposed > 1e-6).float().mean().item():.3f} / "
        f"{(proposed < -1e-6).float().mean().item():.3f} / "
        f"{(proposed.abs() <= 1e-6).float().mean().item():.3f}"
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


def run_live_like_underwrite_table() -> None:
    rows: list[EvidenceRow] = []
    # Candidate is consistently better than Noisy, but the currently executed
    # Projected action only writes a tiny amount and jitters around Noisy.
    rows += [EvidenceRow("good_candidate_projected_tiny_pos", 0.50, 0.505, 0.60, "rho should rise") for _ in range(25)]
    rows += [EvidenceRow("good_candidate_projected_tiny_neg", 0.50, 0.495, 0.60, "rho should rise") for _ in range(25)]
    out = _formal_rho_adv(rows)
    adv = out["formal_adv"]
    proposed = out["proposed_adv"]
    pressure = out["candidate_pressure"]

    print()
    print("C. live-like under-write batch")
    print(
        f"candidate_gain_mean={out['candidate_gain'].mean().item():+.4f} "
        f"projected_gain_mean={out['projected_gain'].mean().item():+.4f} "
        f"formal_adv_mean={adv.mean().item():+.4f} "
        f"proposed_adv_mean={proposed.mean().item():+.4f} "
        f"candidate_pressure_mean={pressure.mean().item():+.4f}"
    )
    print(
        f"formal_adv sign pos/neg/zero: "
        f"{(adv > 1e-6).float().mean().item():.3f} / "
        f"{(adv < -1e-6).float().mean().item():.3f} / "
        f"{(adv.abs() <= 1e-6).float().mean().item():.3f}"
    )
    print(
        f"proposed_adv sign pos/neg/zero: "
        f"{(proposed > 1e-6).float().mean().item():.3f} / "
        f"{(proposed < -1e-6).float().mean().item():.3f} / "
        f"{(proposed.abs() <= 1e-6).float().mean().item():.3f}"
    )
    print(
        f"candidate_pressure sign pos/neg/zero: "
        f"{(pressure > 1e-6).float().mean().item():.3f} / "
        f"{(pressure < -1e-6).float().mean().item():.3f} / "
        f"{(pressure.abs() <= 1e-6).float().mean().item():.3f}"
    )
    print(
        "Readout: this is the suspected live failure mode.  Candidate says repair is useful, "
        "but formal_adv follows the tiny Projected write and can cancel to about zero."
    )


def run_same_candidate_different_state_table() -> None:
    rows = [
        EvidenceRow("clear_good_state", 0.50, 0.56, 0.60, "same Candidate, current state accepts repair"),
        EvidenceRow("clear_bad_state", 0.50, 0.44, 0.60, "same Candidate, current state rejects repair"),
        EvidenceRow("low_write_good_state", 0.50, 0.505, 0.60, "same Candidate, but rho writes too little"),
        EvidenceRow("low_write_bad_state", 0.50, 0.495, 0.60, "same Candidate, tiny write looks harmful"),
    ]
    out = _formal_rho_adv(rows)

    print()
    print("D. same Candidate, different current-state execution")
    print("name                  noisy  proj   cand   cand_gain proj_gain formal_adv proposed_adv accept underwrite expected")
    print("-" * 140)
    for i, row in enumerate(rows):
        print(
            f"{row.name:<21} "
            f"{out['noisy'][i].item():>5.2f} "
            f"{out['projected'][i].item():>5.3f} "
            f"{out['candidate'][i].item():>5.2f} "
            f"{out['candidate_gain'][i].item():>+9.3f} "
            f"{out['projected_gain'][i].item():>+9.3f} "
            f"{out['formal_adv'][i].item():>+10.3f} "
            f"{out['proposed_adv'][i].item():>+12.3f} "
            f"{out['accept_from_projected'][i].item():>6.1f} "
            f"{out['underwrite_adv'][i].item():>10.3f} "
            f"{row.expected}"
        )
    print()
    print(
        "Readout: the first two rows test the real conditional question.  Candidate is identical, "
        "but the current-state execution result changes sign.  The last two rows show the weak-rho "
        "case: the conditional sign exists, but its magnitude is tiny because Projected barely moved."
    )


def run_underwrite_weight_sweep() -> None:
    rows = [
        EvidenceRow("clear_good_state", 0.50, 0.56, 0.60, "accepted repair"),
        EvidenceRow("clear_bad_state", 0.50, 0.44, 0.60, "rejected repair"),
        EvidenceRow("low_write_good_state", 0.50, 0.505, 0.60, "accepted but under-written"),
        EvidenceRow("low_write_bad_state", 0.50, 0.495, 0.60, "tiny harmful write"),
    ]
    weights = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

    print()
    print("E. underwrite_weight sweep")
    print("weight  clear_good  clear_bad  low_good  low_bad  live_like_mean")
    print("-" * 76)
    live_rows: list[EvidenceRow] = []
    live_rows += [EvidenceRow("good_candidate_projected_tiny_pos", 0.50, 0.505, 0.60, "") for _ in range(25)]
    live_rows += [EvidenceRow("good_candidate_projected_tiny_neg", 0.50, 0.495, 0.60, "") for _ in range(25)]
    for weight in weights:
        out = _formal_rho_adv(rows, underwrite_weight=weight)
        live_out = _formal_rho_adv(live_rows, underwrite_weight=weight)
        adv = out["proposed_adv"]
        live_mean = live_out["proposed_adv"].mean().item()
        print(
            f"{weight:>6.2f} "
            f"{adv[0].item():>11.3f} "
            f"{adv[1].item():>10.3f} "
            f"{adv[2].item():>9.3f} "
            f"{adv[3].item():>8.3f} "
            f"{live_mean:>15.3f}"
        )
    print()
    print(
        "Readout: weight=0 is current formal_adv.  A useful weight should strengthen low_write_good_state "
        "without changing clear_bad_state or low_write_bad_state into positive signals.  Values around "
        "0.25-0.50 are the first range to inspect because they lift the weak positive signal without "
        "immediately saturating every accepted repair to +1."
    )


def _train_two_state_policy(repair_adv: float, *, steps: int = 80, lr: float = 5.0e-1) -> tuple[float, float]:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    # Row 0: repairable state.  Row 1: boundary state.
    rho_adv = torch.tensor([[repair_adv], [0.0]], dtype=torch.float32)
    repairable_authority = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    boundary_authority = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    prior_target = torch.zeros(2, 1, dtype=torch.float32)
    model = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
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
    print("F. two-state conditional rho policy")
    print("signal             repair_adv  rho(repairable)  rho(boundary)")
    print("-" * 68)
    print(f"formal weak signal {weak_repair_adv:>10.3f} {weak_repair_rho:>16.3f} {weak_boundary_rho:>14.3f}")
    print(f"strong write signal{strong_repair_adv:>10.3f} {strong_repair_rho:>16.3f} {strong_boundary_rho:>14.3f}")
    print(
        "Readout: the model can represent conditional behavior.  Weak repair evidence moves the repairable state "
        "slowly, while the boundary prior can still push the boundary state down.  This separates architecture "
        "capacity from training-signal quality."
    )


def main() -> None:
    run_single_case_table()
    run_repairable_mixture_table()
    run_live_like_underwrite_table()
    run_same_candidate_different_state_table()
    run_underwrite_weight_sweep()
    run_conditional_policy_toy()


if __name__ == "__main__":
    main()
