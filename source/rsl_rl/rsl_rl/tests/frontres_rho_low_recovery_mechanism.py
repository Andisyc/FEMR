# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: compare rho recovery strength for low-rho FrontRES updates.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_rho_low_recovery_mechanism.py

This module does not start IsaacLab.  It builds the same formal structured-rho
advantage used by training, then compares small loss variants on a raw rho logit.
The goal is to make the low-rho failure inspectable:

- current_rho_linear is the old region_direct repairable-loss ablation.
- bce_logit is the active mature gate-style loss: BCEWithLogits weighted by
  |rho_adv|.
- raw_logit_linear is an intentionally aggressive reference, not a proposal.
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

from rsl_rl.frontres.frontres_structured_rho import build_structured_rho_carrier


@dataclass(frozen=True)
class EvidenceCase:
    name: str
    noisy: float
    projected: float
    candidate: float
    prior_authority: float = 0.0
    prior_gt: float = 0.0


def _logit(p: float) -> float:
    p = min(1.0 - 1.0e-6, max(1.0e-6, float(p)))
    return math.log(p / (1.0 - p))


def _formal_advantage(
    cases: list[EvidenceCase],
    *,
    underwrite_weight: float,
    pref_margin: float = 0.0,
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
        pref_margin=pref_margin,
        rho_floor=0.0,
        directional_weight=1.0,
        rho_center=0.5,
        center_drive_deadzone=0.10,
        retention_weight=0.0,
        floor_penalty_weight=0.0,
        full_bonus_weight=0.0,
        underwrite_weight=underwrite_weight,
        joint_weight_floor=0.0,
        use_rho_update_weight=False,
        device=device,
    )
    prior_authority = torch.tensor([c.prior_authority for c in cases], dtype=torch.float32, device=device).view(n, 1)
    prior_gt = torch.tensor([c.prior_gt for c in cases], dtype=torch.float32, device=device).view(n, 1)
    repairable_authority = (1.0 - prior_authority).clamp(0.0, 1.0)
    return {
        "noisy": noisy.view(n, 1),
        "projected": projected.view(n, 1),
        "candidate": candidate.view(n, 1),
        "candidate_gain": (candidate - noisy).view(n, 1),
        "projected_gain": (projected - noisy).view(n, 1),
        "adv": carrier.rho_advantage_exec[:, :1].detach(),
        "repairable_authority": repairable_authority,
        "prior_authority": prior_authority,
        "prior_gt": prior_gt,
    }


def _loss(
    kind: str,
    raw: torch.Tensor,
    *,
    adv: torch.Tensor,
    repairable_authority: torch.Tensor,
    prior_authority: torch.Tensor,
    prior_gt: torch.Tensor,
    repair_loss_scale: float = 1.0,
    prior_loss_weight: float = 1.0,
) -> torch.Tensor:
    rho = torch.sigmoid(raw)
    repairable_weight = repairable_authority
    boundary_weight = prior_authority

    if kind == "current_rho_linear":
        repair_loss = (-adv * rho * repairable_weight).sum()
        repair_loss = repair_loss / repairable_weight.sum().clamp(min=1.0e-6)
    elif kind == "bce_logit":
        target = (adv > 0.0).to(raw.dtype)
        repair_loss = F.binary_cross_entropy_with_logits(raw, target, reduction="none")
        repair_loss = (repair_loss * adv.abs() * repairable_weight).sum()
        repair_loss = repair_loss / repairable_weight.sum().clamp(min=1.0e-6)
    elif kind == "raw_logit_linear":
        repair_loss = (-adv * raw * repairable_weight).sum()
        repair_loss = repair_loss / repairable_weight.sum().clamp(min=1.0e-6)
    else:
        raise ValueError(f"Unknown loss kind: {kind}")

    boundary_error = (rho - prior_gt).pow(2)
    boundary_loss = (boundary_error * boundary_weight).sum()
    boundary_loss = boundary_loss / boundary_weight.sum().clamp(min=1.0e-6)
    return float(repair_loss_scale) * repair_loss + float(prior_loss_weight) * boundary_loss


def _run_optimizer(
    kind: str,
    data: dict[str, torch.Tensor],
    *,
    init_rho: float,
    steps: int = 80,
    lr: float = 0.20,
    repair_loss_scale: float = 1.0,
) -> dict[str, float]:
    raw = torch.nn.Parameter(torch.full_like(data["adv"], _logit(init_rho)))
    first_loss = _loss(
        kind,
        raw,
        adv=data["adv"],
        repairable_authority=data["repairable_authority"],
        prior_authority=data["prior_authority"],
        prior_gt=data["prior_gt"],
        repair_loss_scale=repair_loss_scale,
    )
    first_loss.backward()
    grad0 = raw.grad.detach().clone()
    raw.grad.zero_()
    optimizer = torch.optim.SGD([raw], lr=lr)
    for _ in range(steps):
        loss = _loss(
            kind,
            raw,
            adv=data["adv"],
            repairable_authority=data["repairable_authority"],
            prior_authority=data["prior_authority"],
            prior_gt=data["prior_gt"],
            repair_loss_scale=repair_loss_scale,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_rho = torch.sigmoid(raw.detach())
    return {
        "grad0_mean": float(grad0.mean().item()),
        "grad0_abs": float(grad0.abs().mean().item()),
        "final_rho_mean": float(final_rho.mean().item()),
        "final_rho_min": float(final_rho.min().item()),
        "final_rho_max": float(final_rho.max().item()),
    }


def _optimize_raw(
    kind: str,
    data: dict[str, torch.Tensor],
    *,
    init_rho: float,
    steps: int = 80,
    lr: float = 0.20,
    repair_loss_scale: float = 1.0,
) -> torch.Tensor:
    raw = torch.nn.Parameter(torch.full_like(data["adv"], _logit(init_rho)))
    optimizer = torch.optim.SGD([raw], lr=lr)
    for _ in range(steps):
        loss = _loss(
            kind,
            raw,
            adv=data["adv"],
            repairable_authority=data["repairable_authority"],
            prior_authority=data["prior_authority"],
            prior_gt=data["prior_gt"],
            repair_loss_scale=repair_loss_scale,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return raw.detach()


def run_single_positive_case() -> None:
    cases = [
        EvidenceCase("low_rho_good_candidate", noisy=0.50, projected=0.505, candidate=0.60),
    ]
    data = _formal_advantage(cases, underwrite_weight=0.25)
    print("=== FrontRES Low-Rho Recovery Mechanism TEST ONLY ===")
    print("A. one low-rho positive case")
    print(
        f"candidate_gain={data['candidate_gain'][0, 0].item():+.3f} "
        f"projected_gain={data['projected_gain'][0, 0].item():+.3f} "
        f"formal_adv={data['adv'][0, 0].item():+.3f}"
    )
    print("loss_kind             scale init_rho  grad0      grad_abs  final_rho")
    print("-" * 82)
    for kind in ("current_rho_linear", "bce_logit", "raw_logit_linear"):
        result = _run_optimizer(kind, data, init_rho=0.04, steps=80, lr=0.20)
        print(
            f"{kind:<21} "
            f"{1.0:>5.2f} "
            f"{0.04:>8.3f} "
            f"{result['grad0_mean']:>+9.5f} "
            f"{result['grad0_abs']:>9.5f} "
            f"{result['final_rho_mean']:>9.3f}"
        )
    print()
    print(
        "Readout: current_rho_linear is expected to have a much smaller initial gradient "
        "because it pushes sigmoid(rho_raw), not rho_raw itself."
    )


def run_init_rho_sweep() -> None:
    cases = [
        EvidenceCase("low_rho_good_candidate", noisy=0.50, projected=0.505, candidate=0.60),
    ]
    data = _formal_advantage(cases, underwrite_weight=0.25)
    print()
    print("B. initial rho sweep")
    print("init_rho  current_final  bce_final  current_grad_abs  bce_grad_abs")
    print("-" * 78)
    for init_rho in (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.50):
        current = _run_optimizer("current_rho_linear", data, init_rho=init_rho, steps=80, lr=0.20)
        bce = _run_optimizer("bce_logit", data, init_rho=init_rho, steps=80, lr=0.20)
        print(
            f"{init_rho:>8.3f} "
            f"{current['final_rho_mean']:>14.3f} "
            f"{bce['final_rho_mean']:>10.3f} "
            f"{current['grad0_abs']:>17.5f} "
            f"{bce['grad0_abs']:>12.5f}"
        )
    print()
    print(
        "Readout: if current_final barely moves for small init_rho while bce_final moves, "
        "the bottleneck is the loss parameterization, not the evidence."
    )


def run_bce_strength_sweep() -> None:
    cases = [
        EvidenceCase("low_rho_good_candidate", noisy=0.50, projected=0.505, candidate=0.60),
    ]
    data = _formal_advantage(cases, underwrite_weight=0.25)
    print()
    print("C. BCEWithLogits repair_loss_scale sweep")
    print("scale  grad_abs  final_rho  note")
    print("-" * 62)
    for scale in (0.05, 0.10, 0.25, 0.50, 1.00, 2.00):
        result = _run_optimizer(
            "bce_logit",
            data,
            init_rho=0.04,
            steps=80,
            lr=0.20,
            repair_loss_scale=scale,
        )
        if result["final_rho_mean"] < 0.15:
            note = "weak"
        elif result["final_rho_mean"] < 0.65:
            note = "useful range"
        else:
            note = "aggressive"
        print(
            f"{scale:>5.2f} "
            f"{result['grad0_abs']:>9.5f} "
            f"{result['final_rho_mean']:>10.3f} "
            f"{note}"
        )
    print()
    print(
        "Readout: this is the knob to inspect before touching formal training code. "
        "It does not create new evidence; it only controls how strongly the existing "
        "repair authority signal acts on the rho logit."
    )


def run_mixed_good_harmful_case() -> None:
    cases: list[EvidenceCase] = []
    for _ in range(8):
        cases.append(EvidenceCase("good", noisy=0.50, projected=0.505, candidate=0.60))
    for _ in range(8):
        cases.append(EvidenceCase("harmful", noisy=0.50, projected=0.495, candidate=0.60))
    data = _formal_advantage(cases, underwrite_weight=0.25)
    adv = data["adv"].view(-1)
    print()
    print("D. mixed good/harmful evidence, separate raw parameters")
    print(
        f"adv_mean={adv.mean().item():+.4f} "
        f"pos/neg={(adv > 1e-6).float().mean().item():.3f}/"
        f"{(adv < -1e-6).float().mean().item():.3f}"
    )
    print("loss_kind             scale good_rho  harmful_rho  rho_gap")
    print("-" * 72)
    for kind, scale in (
        ("current_rho_linear", 1.0),
        ("bce_logit", 1.0),
        ("bce_logit", 4.0),
        ("bce_logit", 8.0),
        ("raw_logit_linear", 1.0),
    ):
        raw = _optimize_raw(
            kind,
            data,
            init_rho=0.04,
            steps=80,
            lr=0.20,
            repair_loss_scale=scale,
        )
        rho = torch.sigmoid(raw).view(-1)
        good = float(rho[:8].mean().item())
        harmful = float(rho[8:].mean().item())
        print(f"{kind:<21} {scale:>5.1f} {good:>8.3f} {harmful:>12.3f} {good - harmful:>8.3f}")
    print()
    print(
        "Readout: this tells us whether the loss can separate 'write more' and 'write less' "
        "when the state representation is not the bottleneck."
    )


def main() -> None:
    run_single_positive_case()
    run_init_rho_sweep()
    run_bce_strength_sweep()
    run_mixed_good_harmful_case()


if __name__ == "__main__":
    main()
