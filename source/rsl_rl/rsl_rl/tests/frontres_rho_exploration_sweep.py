# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: FrontRES rho exploration/update sweep.

Run the newest clip diagnostic from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_rho_exploration_sweep.py

Run the older sweeps explicitly with:

    python source/rsl_rl/rsl_rl/tests/frontres_rho_exploration_sweep.py --section all

This module does not start an environment.  It answers a narrow debugging
question: under the formal structured-rho loss, how much does rho move when the
sampled rho action is only a small distance from the policy mean?

The live log field to compare against is:

    act-mu = structured_joint_rl_rho_action_minus_mean_abs
"""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class EvidenceCase:
    name: str
    expected: str
    region: str
    pos_fraction: float
    neg_fraction: float
    prior_fraction: float = 0.0
    prior_target: float = 0.0
    prior_weight: float = 0.0
    action_delta_raw: float = 0.01
    action_std: float = 0.01
    lr: float = 6.5e-5
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
        self.frontres_structured_joint_rl_loss_mode = "ppo_clipped"
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


def _make_minimal_evidence_batch(
    case: EvidenceCase,
    *,
    batch_size: int,
    init_rho: float,
) -> dict[str, torch.Tensor]:
    """Construct one clean conceptual case.

    This avoids mixing all forces at once.  Each EvidenceCase says exactly how
    much rollout evidence is positive, how much is negative, and whether a prior
    is allowed to pull rho.
    """

    n_pos = int(round(batch_size * case.pos_fraction))
    n_neg = int(round(batch_size * case.neg_fraction))
    n_pos = max(0, min(batch_size, n_pos))
    n_neg = max(0, min(batch_size - n_pos, n_neg))
    n_zero = batch_size - n_pos - n_neg

    init_raw = torch.logit(torch.tensor(float(init_rho)).clamp(1e-6, 1.0 - 1e-6))
    sampled_rho = torch.sigmoid(init_raw + float(case.action_delta_raw))
    mean_rho = torch.sigmoid(init_raw)

    actions = torch.zeros(batch_size, 12)
    actions[: n_pos + n_neg, 6:12] = sampled_rho
    if n_zero > 0:
        actions[n_pos + n_neg :, 6:12] = mean_rho

    rho_adv = torch.zeros(batch_size, 6)
    rho_adv[:n_pos, :] = 0.75
    rho_adv[n_pos : n_pos + n_neg, :] = -0.75
    rho_weight = torch.ones(batch_size, 6)

    prior_authority = torch.zeros(batch_size, 1)
    n_prior = int(round(batch_size * case.prior_fraction))
    if n_prior > 0:
        prior_authority[:n_prior, 0] = 1.0

    region = case.region.lower()
    repairable_authority = torch.ones(batch_size, 1) if region == "repairable" else torch.zeros(batch_size, 1)
    boundary_authority = torch.ones(batch_size, 1) if region in ("safe", "deep_broken") else torch.zeros(batch_size, 1)

    return {
        "obs": torch.ones(batch_size, 4),
        "actions": actions,
        "old_mu": torch.zeros(batch_size, 12),
        "old_sigma": torch.full((batch_size, 12), float(case.action_std)),
        "old_logp": torch.zeros(batch_size, 1),
        "new_logp": torch.zeros(batch_size, 1),
        "rho_adv": rho_adv,
        "rho_weight": rho_weight,
        "prior_authority": prior_authority,
        "prior_target": torch.full((batch_size, 6), float(case.prior_target)),
        "repairable_authority": repairable_authority,
        "boundary_authority": boundary_authority,
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


def _loss_once_unclipped_rho(
    alg: SweepAlgorithm,
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """TEST ONLY: same rho signal as production, without PPO ratio clipping."""

    n = tensors["obs"].shape[0]
    cols = int(getattr(alg.policy, "task_conf_dim", 6))
    rho_dims = list(range(6, 6 + cols))
    actions = tensors["actions"][:n]
    old_mu = tensors["old_mu"][:n]
    old_sigma = tensors["old_sigma"][:n]
    rho_adv = tensors["rho_adv"][:n, :cols].detach()
    rho_weight = tensors["rho_weight"][:n, :cols].detach().clamp(min=0.0)
    active = rho_weight > 1e-6
    zero = alg.policy.action_mean.sum() * 0.0
    if not bool(active.any().detach().item()):
        return zero, {"rho_loss": 0.0, "prior_loss": 0.0, "ratio": 1.0}

    new_logp = alg.policy.get_actions_log_prob_per_dim(actions, rho_dims)
    old_logp = alg.policy.get_actions_log_prob_per_dim_from_stats(actions, old_mu, old_sigma, rho_dims)
    ratio = torch.exp((new_logp[:, :cols] - old_logp[:, :cols]).clamp(-10.0, 10.0))
    rho_loss = (-rho_adv * ratio * rho_weight).sum() / rho_weight.sum().clamp(min=1e-6)

    rho_mean = torch.sigmoid(alg.policy.action_mean[:n, 6:6 + cols])
    prior_loss = zero
    prior_loss_weight = float(getattr(alg, "frontres_structured_joint_prior_loss_weight", 0.0))
    if prior_loss_weight > 0.0:
        prior_authority = tensors["prior_authority"][:n].detach().clamp(0.0, 1.0)
        if prior_authority.ndim == 1:
            prior_authority = prior_authority.view(-1, 1)
        prior_target = tensors["prior_target"][:n, :cols].detach().clamp(0.0, 1.0)
        prior_dim_weight = (prior_authority[:, :1] * active.to(rho_mean.dtype)).clamp(0.0, 1.0)
        if bool((prior_dim_weight > 1e-6).any().detach().item()):
            prior_loss = ((rho_mean - prior_target).pow(2) * prior_dim_weight).sum()
            prior_loss = prior_loss / prior_dim_weight.sum().clamp(min=1e-6)

    loss = rho_loss + prior_loss_weight * prior_loss
    return loss, {
        "rho_loss": float(rho_loss.detach().item()),
        "prior_loss": float(prior_loss.detach().item()),
        "ratio": float(ratio[active].detach().mean().item()),
    }


def _loss_once_direct_rho_mean(
    alg: SweepAlgorithm,
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """TEST ONLY: train repair authority directly on rho_mean.

    This removes PPO's sampled-action log-prob ratio entirely.  Positive rho
    advantage pushes rho_mean up; negative rho advantage pushes rho_mean down.
    It is the simplest check of whether the repair-authority signal itself is
    learnable before adding PPO safety machinery around it.
    """

    n = tensors["obs"].shape[0]
    cols = int(getattr(alg.policy, "task_conf_dim", 6))
    rho_adv = tensors["rho_adv"][:n, :cols].detach()
    rho_weight = tensors["rho_weight"][:n, :cols].detach().clamp(min=0.0)
    active = rho_weight > 1e-6
    zero = alg.policy.action_mean.sum() * 0.0
    if not bool(active.any().detach().item()):
        return zero, {"rho_loss": 0.0, "prior_loss": 0.0, "rho_mean": 0.0}

    rho_mean = torch.sigmoid(alg.policy.action_mean[:n, 6:6 + cols])
    rho_loss = (-rho_adv * rho_mean * rho_weight).sum() / rho_weight.sum().clamp(min=1e-6)

    prior_loss = zero
    prior_loss_weight = float(getattr(alg, "frontres_structured_joint_prior_loss_weight", 0.0))
    if prior_loss_weight > 0.0:
        prior_authority = tensors["prior_authority"][:n].detach().clamp(0.0, 1.0)
        if prior_authority.ndim == 1:
            prior_authority = prior_authority.view(-1, 1)
        prior_target = tensors["prior_target"][:n, :cols].detach().clamp(0.0, 1.0)
        prior_dim_weight = (prior_authority[:, :1] * active.to(rho_mean.dtype)).clamp(0.0, 1.0)
        if bool((prior_dim_weight > 1e-6).any().detach().item()):
            prior_loss = ((rho_mean - prior_target).pow(2) * prior_dim_weight).sum()
            prior_loss = prior_loss / prior_dim_weight.sum().clamp(min=1e-6)

    loss = rho_loss + prior_loss_weight * prior_loss
    return loss, {
        "rho_loss": float(rho_loss.detach().item()),
        "prior_loss": float(prior_loss.detach().item()),
        "rho_mean": float(rho_mean[active].detach().mean().item()),
    }


def _loss_once_region_authority_direct(
    alg: SweepAlgorithm,
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """TEST ONLY: direct rho learning with region-based teacher authority.

    repairable region: rollout evidence teaches rho.
    safe/deep_broken boundary region: prior teaches rho low.

    This is the code version of the concept:
        repairable -> listen to rollout evidence
        boundary   -> listen to prior
    """

    n = tensors["obs"].shape[0]
    cols = int(getattr(alg.policy, "task_conf_dim", 6))
    rho_adv = tensors["rho_adv"][:n, :cols].detach()
    rho_weight = tensors["rho_weight"][:n, :cols].detach().clamp(min=0.0)
    active = rho_weight > 1e-6
    zero = alg.policy.action_mean.sum() * 0.0
    if not bool(active.any().detach().item()):
        return zero, {
            "repairable_loss": 0.0,
            "boundary_loss": 0.0,
            "repairable_authority": 0.0,
            "boundary_authority": 0.0,
        }

    rho_mean = torch.sigmoid(alg.policy.action_mean[:n, 6:6 + cols])
    repairable_authority = tensors.get("repairable_authority", torch.ones(n, 1))[:n].detach().clamp(0.0, 1.0)
    boundary_authority = tensors.get("boundary_authority", torch.zeros(n, 1))[:n].detach().clamp(0.0, 1.0)
    if repairable_authority.ndim == 1:
        repairable_authority = repairable_authority.view(-1, 1)
    if boundary_authority.ndim == 1:
        boundary_authority = boundary_authority.view(-1, 1)

    repairable_weight = (repairable_authority[:, :1] * active.to(rho_mean.dtype)).clamp(0.0, 1.0)
    boundary_weight = (boundary_authority[:, :1] * active.to(rho_mean.dtype)).clamp(0.0, 1.0)

    repairable_loss = zero
    if bool((repairable_weight > 1e-6).any().detach().item()):
        repairable_loss = (-rho_adv * rho_mean * repairable_weight).sum()
        repairable_loss = repairable_loss / repairable_weight.sum().clamp(min=1e-6)

    prior_target = tensors["prior_target"][:n, :cols].detach().clamp(0.0, 1.0)
    boundary_loss = zero
    if bool((boundary_weight > 1e-6).any().detach().item()):
        boundary_loss = ((rho_mean - prior_target).pow(2) * boundary_weight).sum()
        boundary_loss = boundary_loss / boundary_weight.sum().clamp(min=1e-6)

    loss = repairable_loss + boundary_loss
    return loss, {
        "repairable_loss": float(repairable_loss.detach().item()),
        "boundary_loss": float(boundary_loss.detach().item()),
        "repairable_authority": float(repairable_authority.mean().detach().item()),
        "boundary_authority": float(boundary_authority.mean().detach().item()),
    }


def _compute_clip_diagnostics(
    alg: SweepAlgorithm,
    tensors: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Mirror the formal rho PPO clip math and print what it hides.

    The formal loss returns the mean ratio, but mean ratio can look harmless
    even when many positive or negative advantage dimensions are clipped.  This
    helper keeps the same action/logit convention as the production loss and
    only derives TEST ONLY diagnostics.
    """

    n = tensors["obs"].shape[0]
    cols = int(getattr(alg.policy, "task_conf_dim", 6))
    rho_dims = list(range(6, 6 + cols))
    actions = tensors["actions"][:n]
    old_mu = tensors["old_mu"][:n]
    old_sigma = tensors["old_sigma"][:n]
    rho_adv = tensors["rho_adv"][:n, :cols]
    rho_weight = tensors["rho_weight"][:n, :cols].clamp(min=0.0)
    active = rho_weight > 1e-6

    if not bool(active.any().detach().item()):
        return {
            "ratio_min": 1.0,
            "ratio_max": 1.0,
            "clip_frac": 0.0,
            "clip_pos_frac": 0.0,
            "clip_neg_frac": 0.0,
            "pos_clip_use_frac": 0.0,
            "neg_clip_use_frac": 0.0,
            "unclipped_loss": 0.0,
            "clipped_loss": 0.0,
            "selected_loss": 0.0,
            "clip_gap": 0.0,
        }

    new_logp = alg.policy.get_actions_log_prob_per_dim(actions, rho_dims)
    old_logp = alg.policy.get_actions_log_prob_per_dim_from_stats(actions, old_mu, old_sigma, rho_dims)
    log_ratio = new_logp[:, :cols] - old_logp[:, :cols]
    ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
    ratio_clipped = torch.clamp(ratio, 1.0 - alg.clip_param, 1.0 + alg.clip_param)

    surrogate = -rho_adv * ratio
    surrogate_clipped = -rho_adv * ratio_clipped
    selected = torch.max(surrogate, surrogate_clipped)
    clip_boundary = (ratio < 1.0 - alg.clip_param) | (ratio > 1.0 + alg.clip_param)
    clip_selected = surrogate_clipped > surrogate

    pos = active & (rho_adv > 1e-6)
    neg = active & (rho_adv < -1e-6)
    denom = rho_weight.sum().clamp(min=1e-6)

    def _frac(mask: torch.Tensor) -> float:
        return float(mask.float().mean().detach().item()) if int(mask.numel()) > 0 else 0.0

    def _masked_frac(mask: torch.Tensor, base: torch.Tensor) -> float:
        if not bool(base.any().detach().item()):
            return 0.0
        return float(mask[base].float().mean().detach().item())

    unclipped_loss = (surrogate * rho_weight).sum() / denom
    clipped_candidate_loss = (surrogate_clipped * rho_weight).sum() / denom
    selected_loss = (selected * rho_weight).sum() / denom

    active_ratio = ratio[active].detach()
    return {
        "ratio_min": float(active_ratio.min().item()),
        "ratio_max": float(active_ratio.max().item()),
        "clip_frac": _masked_frac(clip_boundary, active),
        "clip_pos_frac": _masked_frac(clip_boundary, pos),
        "clip_neg_frac": _masked_frac(clip_boundary, neg),
        "pos_clip_use_frac": _masked_frac(clip_selected, pos),
        "neg_clip_use_frac": _masked_frac(clip_selected, neg),
        "unclipped_loss": float(unclipped_loss.detach().item()),
        "clipped_loss": float(clipped_candidate_loss.detach().item()),
        "selected_loss": float(selected_loss.detach().item()),
        "clip_gap": float((selected_loss - unclipped_loss).detach().item()),
    }


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


def _run_clip_probe(
    case: SweepCase,
    *,
    checkpoints: tuple[int, ...],
    unclipped_rho: bool = False,
    loss_mode: str | None = None,
    batch_size: int = 100,
    init_rho: float = 0.45,
) -> list[dict[str, float]]:
    if loss_mode is None:
        loss_mode = "unclipped" if unclipped_rho else "clipped"

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

    optimizer = torch.optim.Adam(alg.parameters(), lr=case.lr)
    rows: list[dict[str, float]] = []
    checkpoint_set = set(checkpoints)
    max_step = max(checkpoints)
    for step in range(max_step + 1):
        if step in checkpoint_set:
            loss, metrics = _loss_once(alg, tensors)
            clip = _compute_clip_diagnostics(alg, tensors)
            rows.append(
                {
                    "step": float(step),
                    "rho": float(torch.sigmoid(alg.policy.action_mean[:, 6:12]).mean().detach().item()),
                    "formal_loss": float(loss.detach().item()),
                    "ratio": float(metrics["structured_joint_rl_ratio_mean"]),
                    **clip,
                }
            )
        if step == max_step:
            break
        optimizer.zero_grad()
        if loss_mode == "region_direct":
            loss, _ = _loss_once_region_authority_direct(alg, tensors)
        elif loss_mode == "direct":
            loss, _ = _loss_once_direct_rho_mean(alg, tensors)
        elif loss_mode == "unclipped":
            loss, _ = _loss_once_unclipped_rho(alg, tensors)
        else:
            loss, _ = _loss_once(alg, tensors)
        (case.loss_weight * loss).backward()
        optimizer.step()
    return rows


def _run_minimal_evidence_probe(
    case: EvidenceCase,
    *,
    loss_mode: str,
    batch_size: int = 100,
    init_rho: float = 0.45,
) -> dict[str, float | str]:
    torch.manual_seed(11)
    alg = SweepAlgorithm(
        batch_size=batch_size,
        init_rho=init_rho,
        prior_weight=case.prior_weight,
        action_std=case.action_std,
    )
    tensors = _make_minimal_evidence_batch(case, batch_size=batch_size, init_rho=init_rho)
    tensors["old_mu"] = alg.policy.action_mean.detach().clone()
    if loss_mode == "formal_region":
        alg.frontres_structured_joint_rl_loss_mode = "region_direct"

    optimizer = torch.optim.Adam(alg.parameters(), lr=case.lr)
    rho0 = float(torch.sigmoid(alg.policy.action_mean[:, 6:12]).mean().detach().item())
    for _ in range(case.steps):
        optimizer.zero_grad()
        if loss_mode == "region_direct":
            loss, _ = _loss_once_region_authority_direct(alg, tensors)
        elif loss_mode == "formal_region":
            loss, _ = _loss_once(alg, tensors)
        elif loss_mode == "direct":
            loss, _ = _loss_once_direct_rho_mean(alg, tensors)
        elif loss_mode == "unclipped":
            loss, _ = _loss_once_unclipped_rho(alg, tensors)
        else:
            loss, _ = _loss_once(alg, tensors)
        loss.backward()
        optimizer.step()

    rho1 = float(torch.sigmoid(alg.policy.action_mean[:, 6:12]).mean().detach().item())
    clip = _compute_clip_diagnostics(alg, tensors)
    delta = rho1 - rho0
    if case.expected == "up":
        ok = delta > 1e-5
    elif case.expected == "down":
        ok = delta < -1e-5
    else:
        ok = abs(delta) <= 1e-5
    return {
        "rho0": rho0,
        "rho1": rho1,
        "delta": delta,
        "clip_frac": clip["clip_frac"],
        "ratio_min": clip["ratio_min"],
        "ratio_max": clip["ratio_max"],
        "result": "PASS" if ok else "CHECK",
    }


def run_rho_minimal_evidence_cases() -> None:
    cases = [
        EvidenceCase(
            "safe_positive",
            expected="down",
            region="safe",
            pos_fraction=1.0,
            neg_fraction=0.0,
            prior_fraction=1.0,
            prior_target=0.0,
            prior_weight=1.0,
        ),
        EvidenceCase(
            "repairable_positive",
            expected="up",
            region="repairable",
            pos_fraction=1.0,
            neg_fraction=0.0,
        ),
        EvidenceCase(
            "repairable_negative",
            expected="down",
            region="repairable",
            pos_fraction=0.0,
            neg_fraction=1.0,
        ),
        EvidenceCase(
            "deep_positive",
            expected="down",
            region="deep_broken",
            pos_fraction=1.0,
            neg_fraction=0.0,
            prior_fraction=1.0,
            prior_target=0.0,
            prior_weight=1.0,
        ),
    ]
    modes = ("region_direct", "formal_region", "direct", "clipped")

    print()
    print("=== FrontRES Rho Minimal Evidence Cases TEST ONLY ===")
    print("Region authority: boundary regions listen to prior; repairable regions listen to rollout evidence.")
    print("case                region       mode          expect rho0   rho200 delta    clip  r_min  r_max  result")
    print("-" * 112)
    for case in cases:
        for mode in modes:
            result = _run_minimal_evidence_probe(case, loss_mode=mode)
            print(
                f"{case.name:<19} "
                f"{case.region:<12} "
                f"{mode:<13} "
                f"{case.expected:<6} "
                f"{result['rho0']:>5.3f} "
                f"{result['rho1']:>7.3f} "
                f"{result['delta']:>+7.4f} "
                f"{result['clip_frac']:>5.2f} "
                f"{result['ratio_min']:>6.3f} "
                f"{result['ratio_max']:>6.3f} "
                f"{result['result']:<5}"
            )

    print()
    print(
        "Readout: region_direct is the proposed concept test: safe/deep_broken "
        "must go down even with positive evidence, while repairable follows "
        "rollout evidence.  direct ignores region authority.  clipped shows the "
        "current PPO-style behavior for comparison."
    )


def run_rho_clip_diagnostics() -> None:
    cases = [
        SweepCase(
            "live_base",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=6.5e-5,
            loss_weight=1.0,
        ),
        SweepCase(
            "strong_lr_w3",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=5.0e-4,
            loss_weight=3.0,
        ),
        SweepCase(
            "wide_action",
            action_delta_raw=0.10,
            prior_weight=1.0,
            action_std=0.10,
            lr=6.5e-5,
            loss_weight=1.0,
        ),
    ]

    print()
    print("=== FrontRES Rho PPO Clip Diagnostics TEST ONLY ===")
    print("clip_frac means ratio outside [0.8, 1.2]; *_use means clipped term selected by max().")
    print(
        "case         step rho    r_min  r_max  clip  clip+ clip- use+  use-  "
        "unclip  clip_c selected gap"
    )
    print("-" * 112)
    for case in cases:
        rows = _run_clip_probe(case, checkpoints=(0, 1, 10, 50, 200))
        for row in rows:
            print(
                f"{case.name:<12} "
                f"{int(row['step']):>4d} "
                f"{row['rho']:>5.3f} "
                f"{row['ratio_min']:>6.3f} "
                f"{row['ratio_max']:>6.3f} "
                f"{row['clip_frac']:>5.2f} "
                f"{row['clip_pos_frac']:>5.2f} "
                f"{row['clip_neg_frac']:>5.2f} "
                f"{row['pos_clip_use_frac']:>5.2f} "
                f"{row['neg_clip_use_frac']:>5.2f} "
                f"{row['unclipped_loss']:>+7.3f} "
                f"{row['clipped_loss']:>+7.3f} "
                f"{row['selected_loss']:>+8.3f} "
                f"{row['clip_gap']:>+6.3f}"
            )

    print()
    print(
        "Readout: if clip/use fractions stay near zero, PPO clipping is not the "
        "main brake.  If they jump high while rho barely moves, the clipped "
        "surrogate is cutting off the effective rho update."
    )


def run_rho_unclipped_comparison() -> None:
    cases = [
        SweepCase(
            "live_base",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=6.5e-5,
            loss_weight=1.0,
        ),
        SweepCase(
            "strong_lr_w3",
            action_delta_raw=0.01,
            prior_weight=1.0,
            action_std=0.01,
            lr=5.0e-4,
            loss_weight=3.0,
        ),
        SweepCase(
            "wide_action",
            action_delta_raw=0.10,
            prior_weight=1.0,
            action_std=0.10,
            lr=6.5e-5,
            loss_weight=1.0,
        ),
    ]

    print()
    print("=== FrontRES Rho Loss Form Comparison TEST ONLY ===")
    print("Same toy batch and prior; compare PPO clipped, PPO unclipped, and direct rho_mean loss.")
    print("case         mode       rho0   rho200 delta    r_min  r_max  clip  selected_gap")
    print("-" * 86)
    for case in cases:
        rows_by_mode = [
            ("clipped", _run_clip_probe(case, checkpoints=(0, 200), loss_mode="clipped")),
            ("unclipped", _run_clip_probe(case, checkpoints=(0, 200), loss_mode="unclipped")),
            ("direct", _run_clip_probe(case, checkpoints=(0, 200), loss_mode="direct")),
        ]
        for mode, rows in rows_by_mode:
            start, end = rows[0], rows[-1]
            print(
                f"{case.name:<12} "
                f"{mode:<10} "
                f"{start['rho']:>5.3f} "
                f"{end['rho']:>7.3f} "
                f"{end['rho'] - start['rho']:>+7.4f} "
                f"{end['ratio_min']:>6.3f} "
                f"{end['ratio_max']:>6.3f} "
                f"{end['clip_frac']:>5.2f} "
                f"{end['clip_gap']:>+12.3f}"
            )

    print()
    print(
        "Readout: if direct moves rho but PPO modes do not, the repair-authority "
        "signal exists but is weakened by the PPO log-prob/ratio formulation.  "
        "If direct also barely moves, the toy advantage itself is too weak."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TEST ONLY FrontRES rho loss diagnostics.")
    parser.add_argument(
        "--section",
        choices=("minimal", "clip", "unclipped", "exploration", "update", "all"),
        default="minimal",
        help="Which diagnostic section to run. Defaults to minimal for quick iteration.",
    )
    args = parser.parse_args()

    if args.section in ("minimal", "all"):
        run_rho_minimal_evidence_cases()
    if args.section in ("exploration", "all"):
        run_rho_exploration_sweep()
    if args.section in ("update", "all"):
        run_rho_update_strength_sweep()
    if args.section in ("clip", "all"):
        run_rho_clip_diagnostics()
    if args.section in ("unclipped", "all"):
        run_rho_unclipped_comparison()


if __name__ == "__main__":
    main()
