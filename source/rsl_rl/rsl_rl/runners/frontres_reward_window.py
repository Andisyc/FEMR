# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES rollout reward-window construction.

The reward window lives in runners because it converts post-env rollout evidence
into per-sample reward weights and actor-credit gates before process_env_step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FrontRESRewardWindow:
    r_exec: torch.Tensor
    damage_gap: torch.Tensor
    oracle_clean_gap: torch.Tensor
    oracle_trust: torch.Tensor
    repair_ratio: torch.Tensor
    safe_gate: torch.Tensor
    repair_gate: torch.Tensor
    broken_gate: torch.Tensor
    window_mu: torch.Tensor
    exec_gate: torch.Tensor
    cost_gate: torch.Tensor
    safe_frac: torch.Tensor
    repair_frac: torch.Tensor
    broken_frac: torch.Tensor
    safe_gap: float
    broken_gap: float
    learnable_route_mask: torch.Tensor
    exec_weight: torch.Tensor
    cost_weight: torch.Tensor
    actor_gate: torch.Tensor
    harm_penalty: torch.Tensor
    harm_penalty_exec: torch.Tensor
    harm_mag: torch.Tensor
    cost_exec: torch.Tensor
    effective_gain_bonus: torch.Tensor
    effective_gain_bonus_exec: torch.Tensor
    under_repair_penalty: torch.Tensor
    r_delta: torch.Tensor | None
    under_write: torch.Tensor | None
    ranking_reward: torch.Tensor | None
    reward_progress: float
    constraint_progress: float


def build_frontres_reward_window(
    *,
    runner: Any,
    cfg: Mapping[str, Any],
    n_train: int,
    n_exec: int,
    exec_clean: torch.Tensor,
    exec_perturbed: torch.Tensor,
    exec_feasible: torch.Tensor,
    exec_frontres: torch.Tensor,
    repair_gain: torch.Tensor,
    mode_groups: Any,
    e_raw: torch.Tensor,
    e_fr: torch.Tensor,
    intervention_cost: torch.Tensor,
    action_activity: torch.Tensor,
    under_repair_penalty: torch.Tensor,
    dr_scale: float,
    ppo_actor_weight_current: float,
    stable_route_active_mask: torch.Tensor | None,
    device: torch.device,
) -> FrontRESRewardWindow:
    """Build reward gates, weights, harm terms, and actor-credit gates."""

    reward_dr_ref = float(
        cfg.get(
            "frontres_reward_scale_dr_reference",
            cfg.get("supervised_warmup_dr_scale", cfg.get("dr_scale_init", 1.0)),
        )
    )
    reward_dr_ref = max(reward_dr_ref, 1e-6)
    reward_dr_progress = max(0.0, min(1.0, float(dr_scale) / reward_dr_ref))
    reward_actor_progress = max(0.0, min(1.0, float(ppo_actor_weight_current)))
    reward_progress = reward_dr_progress * reward_actor_progress
    reward_progress = max(float(cfg.get("frontres_reward_progress_min", 0.0)), min(1.0, reward_progress))
    constraint_exp = float(cfg.get("frontres_constraint_progress_exponent", 2.0))
    constraint_exp = max(1.0, constraint_exp)
    constraint_progress = reward_progress ** constraint_exp

    gap_raw = exec_clean - exec_perturbed
    damage_gap = gap_raw.clamp(min=0.0)
    oracle_clean_gap = (exec_clean - exec_feasible).clamp(min=0.0)
    oracle_trust_tau = float(cfg.get("frontres_oracle_clean_gap_tau", 0.0))
    if oracle_trust_tau > 0.0:
        oracle_trust = torch.exp(-oracle_clean_gap / max(oracle_trust_tau, 1e-6))
    else:
        oracle_trust_threshold = float(cfg.get("frontres_oracle_clean_gap_threshold", 1e9))
        oracle_trust = (oracle_clean_gap <= oracle_trust_threshold).to(damage_gap.dtype)

    gap_floor = float(cfg.get("frontres_gap_floor_per_step", 0.005))
    repair_ratio = (repair_gain / damage_gap.clamp(min=gap_floor)).clamp(-1.0, 1.0)
    reward_signal_mode = str(cfg.get("frontres_exec_reward_signal", "gain")).lower()
    if reward_signal_mode in ("family_preference", "preference", "ranking"):
        gain_std = runner._frontres_family_gain_std(mode_groups, repair_gain.detach())
        tau = max(float(cfg.get("frontres_family_preference_tau", 1.0)), 1e-6)
        pref = torch.tanh((repair_gain / gain_std) / tau)
        alpha = float(cfg.get("frontres_family_preference_alpha", 0.7))
        alpha = max(0.0, min(1.0, alpha))
        scale = float(cfg.get("frontres_family_preference_scale", 0.02))
        exec_signal = scale * (alpha * pref + (1.0 - alpha) * repair_ratio)
    elif reward_signal_mode == "ratio":
        exec_signal = repair_ratio
    else:
        exec_signal = repair_gain
    r_exec = torch.zeros(n_train, device=device)
    r_exec[:n_exec] = exec_signal

    safe_gap = float(cfg.get("frontres_safe_gap_per_step", 0.003))
    broken_gap = float(cfg.get("frontres_broken_gap_per_step", 0.08))
    broken_gap = max(broken_gap, safe_gap + 1e-6)
    gate_temp = float(cfg.get("frontres_gap_gate_temp", 0.005))
    gate_temp = max(gate_temp, 1e-6)
    enter_window = torch.sigmoid((damage_gap - safe_gap) / gate_temp)
    exit_window = torch.sigmoid((broken_gap - damage_gap) / gate_temp)
    window_mu_raw = enter_window * exit_window
    gap_mid = 0.5 * (safe_gap + broken_gap)
    peak_enter_arg = max(-60.0, min(60.0, (gap_mid - safe_gap) / gate_temp))
    peak_exit_arg = max(-60.0, min(60.0, (broken_gap - gap_mid) / gate_temp))
    window_peak = (
        1.0 / (1.0 + math.exp(-peak_enter_arg))
        * 1.0 / (1.0 + math.exp(-peak_exit_arg))
    )
    window_mu = (window_mu_raw / max(window_peak, 1e-6)).clamp(0.0, 1.0)
    safe_gate = (1.0 - enter_window).clamp(0.0, 1.0)
    repair_gate = window_mu
    broken_gate = (1.0 - exit_window).clamp(0.0, 1.0)
    safe_frac = (damage_gap < safe_gap).float().mean()
    repair_frac = ((damage_gap >= safe_gap) & (damage_gap <= broken_gap)).float().mean()
    broken_frac = (damage_gap > broken_gap).float().mean()

    stable_route_active = stable_route_active_mask
    if stable_route_active is None:
        stable_route_active = torch.zeros(n_exec, device=device, dtype=torch.bool)
    else:
        stable_route_active = stable_route_active.to(device).view(-1).bool()
        if stable_route_active.numel() < n_exec:
            stable_route_active = torch.nn.functional.pad(
                stable_route_active,
                (0, n_exec - stable_route_active.numel()),
                value=False,
            )
        stable_route_active = stable_route_active[:n_exec]
    rho_space_for_route = str(cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
    if rho_space_for_route in ("tri_anchor", "tri-anchor", "tri"):
        learnable_route_mask = torch.ones_like(damage_gap)
    else:
        learnable_route_mask = (~stable_route_active).to(damage_gap.dtype)

    restore_min_ratio = float(cfg.get("frontres_min_restore_ratio", 0.0))
    restore_under_weight = float(cfg.get("frontres_under_repair_weight", 0.0))
    if restore_min_ratio > 0.0 and restore_under_weight > 0.0:
        restore_ratio = ((e_raw - e_fr) / e_raw.clamp(min=1e-6)).clamp(-1.0, 1.0)
        under = torch.relu(restore_min_ratio - restore_ratio)
        under_repair_penalty[:n_exec] = restore_under_weight * repair_gate * under[:n_exec].pow(2)

    selective_reward = bool(cfg.get("frontres_selective_reward_enabled", True))
    if selective_reward:
        exec_gate = repair_gate
        safe_cost_weight = float(cfg.get("frontres_safe_cost_weight", 1.0))
        repair_cost_weight = float(cfg.get("frontres_repair_cost_weight", 0.15))
        broken_cost_weight = float(cfg.get("frontres_broken_cost_weight", 1.0))
        cost_gate = (
            safe_cost_weight * safe_gate
            + repair_cost_weight * repair_gate
            + broken_cost_weight * broken_gate
        ).clamp(min=0.0)
    else:
        exec_gate = window_mu
        cost_gate = (1.0 - window_mu).clamp(0.0, 1.0)

    harm_eps = float(cfg.get("frontres_harm_epsilon", 0.001))
    harm_weight_cfg = float(cfg.get("frontres_harm_penalty_weight", 0.25))
    side_harm_weight = float(cfg.get("frontres_side_harm_weight", 0.0))
    side_harm_weight = max(0.0, min(1.0, side_harm_weight))
    harm_mag_raw = torch.relu(-repair_gain - max(harm_eps, 0.0))
    cost_exec = intervention_cost[:n_exec]
    harm_action_floor = float(cfg.get("frontres_harm_action_cost_floor", 0.001))
    harm_action_ref = float(cfg.get("frontres_harm_action_cost_ref", 0.01))
    harm_action_ref = max(harm_action_ref, harm_action_floor + 1e-6)
    harm_action_measure = action_activity[:n_exec]
    harm_action_gate = ((harm_action_measure - harm_action_floor) / (harm_action_ref - harm_action_floor)).clamp(0.0, 1.0)
    harm_mag = harm_mag_raw * harm_action_gate
    if selective_reward:
        broken_harm_weight = float(cfg.get("frontres_broken_harm_weight", 1.0))
        harm_weight = (
            repair_gate
            + broken_harm_weight * broken_gate
            + side_harm_weight * safe_gate
        ).clamp(0.0, 1.0)
    else:
        harm_weight = (window_mu + side_harm_weight * (1.0 - window_mu)).clamp(0.0, 1.0)
    harm_penalty_exec = harm_weight_cfg * harm_weight * harm_mag

    side_actor_weight = float(cfg.get("frontres_side_actor_gate_weight", 0.05))
    side_actor_weight = max(0.0, min(1.0, side_actor_weight))
    if selective_reward:
        actor_gate = (
            oracle_trust * repair_gate
            + side_actor_weight * (safe_gate + broken_gate)
        ).clamp(0.0, 1.0)
    else:
        actor_gate = (
            oracle_trust * window_mu + side_actor_weight * (1.0 - window_mu)
        ).clamp(0.0, 1.0)
    actor_gate = actor_gate * learnable_route_mask

    exec_weight = torch.zeros(n_train, device=device)
    cost_weight = torch.ones(n_train, device=device)
    exec_weight[:n_exec] = exec_gate
    cost_weight[:n_exec] = cost_gate
    harm_penalty = torch.zeros(n_train, device=device)
    harm_penalty[:n_exec] = harm_penalty_exec

    effective_gain_bonus_exec = torch.zeros(n_exec, device=device)
    if selective_reward:
        min_effective_gain = float(cfg.get("frontres_min_effective_gain", 0.006))
        bonus_weight = float(cfg.get("frontres_effective_gain_bonus_weight", 0.5))
        effective_gain_bonus_exec = bonus_weight * repair_gate * torch.relu(repair_gain - min_effective_gain)
    effective_gain_bonus = torch.zeros(n_train, device=device)
    effective_gain_bonus[:n_exec] = effective_gain_bonus_exec

    return FrontRESRewardWindow(
        r_exec=r_exec,
        damage_gap=damage_gap,
        oracle_clean_gap=oracle_clean_gap,
        oracle_trust=oracle_trust,
        repair_ratio=repair_ratio,
        safe_gate=safe_gate,
        repair_gate=repair_gate,
        broken_gate=broken_gate,
        window_mu=window_mu,
        exec_gate=exec_gate,
        cost_gate=cost_gate,
        safe_frac=safe_frac,
        repair_frac=repair_frac,
        broken_frac=broken_frac,
        safe_gap=safe_gap,
        broken_gap=broken_gap,
        learnable_route_mask=learnable_route_mask,
        exec_weight=exec_weight,
        cost_weight=cost_weight,
        actor_gate=actor_gate,
        harm_penalty=harm_penalty,
        harm_penalty_exec=harm_penalty_exec,
        harm_mag=harm_mag,
        cost_exec=cost_exec,
        effective_gain_bonus=effective_gain_bonus,
        effective_gain_bonus_exec=effective_gain_bonus_exec,
        under_repair_penalty=under_repair_penalty,
        r_delta=None,
        under_write=None,
        ranking_reward=None,
        reward_progress=reward_progress,
        constraint_progress=constraint_progress,
    )


def compose_frontres_reward_delta(
    *,
    cfg: Mapping[str, Any],
    reward_window: FrontRESRewardWindow,
    n_train: int,
    n_exec: int,
    n_candidate: int,
    repair_gain: torch.Tensor,
    candidate_gain: torch.Tensor,
    projection_gain: torch.Tensor,
    r_step: torch.Tensor,
    r_rescue: torch.Tensor,
    intervention_cost: torch.Tensor,
    overcorrection_cost: torch.Tensor,
    w_exec: float,
    repair_scale: float,
    w_geom: float,
    w_rescue: float,
    w_exec_harm: float,
    device: torch.device,
) -> FrontRESRewardWindow:
    """Compose ranking reward and final FrontRES delta reward."""

    ranking_enabled = bool(cfg.get("frontres_candidate_ranking_reward_enabled", True)) and n_candidate > 0
    if ranking_enabled:
        ranking_under_weight = float(cfg.get("frontres_candidate_underwrite_weight", 1.0))
        ranking_projection_weight = float(cfg.get("frontres_candidate_projection_weight", 0.25))
        ranking_harm_weight = float(cfg.get("frontres_candidate_harm_weight", 1.0))
        under_write = torch.relu(candidate_gain - repair_gain) * (candidate_gain > 0.0).to(repair_gain.dtype)
        ranking_exec = (
            repair_gain
            + ranking_projection_weight * projection_gain
            - ranking_under_weight * under_write
            - ranking_harm_weight * torch.relu(-repair_gain)
        )
        ranking_reward = torch.zeros(n_train, device=device)
        ranking_reward[:n_exec] = reward_window.exec_gate * ranking_exec
    else:
        under_write = torch.zeros_like(repair_gain)
        ranking_reward = torch.zeros(n_train, device=device)

    positive_reward = (
        float(w_exec) * float(repair_scale) * reward_window.exec_weight * reward_window.r_exec
        + float(w_exec) * float(repair_scale) * reward_window.effective_gain_bonus
        + float(w_geom) * r_step
        + float(cfg.get("frontres_candidate_ranking_reward_weight", 1.0)) * ranking_reward
        + float(w_rescue) * r_rescue
    )
    constraint_penalty = (
        float(w_exec_harm) * reward_window.harm_penalty
        + reward_window.cost_weight * intervention_cost
        + overcorrection_cost
        + reward_window.under_repair_penalty
    )
    r_delta = (
        float(reward_window.reward_progress) * positive_reward
        - float(reward_window.constraint_progress) * constraint_penalty
    )

    reward_window.r_delta = r_delta
    reward_window.under_write = under_write
    reward_window.ranking_reward = ranking_reward
    return reward_window
