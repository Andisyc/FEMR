# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES schedule and phase planning helpers.

This module keeps the pure schedule/phase/weight computations out of the runner
state-mutation layer. It should not write into runner/env/alg objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .frontres_dr_curriculum import (
    GMTFrontierState,
    allowed_perturbation_bases,
    choice_hash as frontres_curriculum_hash,
    choose_perturbation_choices,
    mode_complexity,
    sample_per_env_dr_strength,
    sample_perturbation_mix,
    sample_scalar_dr_strength,
    score_gmt_frontier,
    update_boundary_ema,
    update_gmt_frontier_state,
    warmup_perturbation_mode_groups,
)


@dataclass(frozen=True)
class FrontRESModeState:
    is_frontres: bool
    training_objective: str
    supervised_restore: bool
    hsl_restore: bool
    is_task_space_mode: bool
    critic_warmup_iters: int


@dataclass(frozen=True)
class FrontRESPairLayout:
    use_quartet_reward: bool
    n_train: int
    n_candidate: int
    n_base: int
    n_clean: int
    cur_reward_sum_gmt: torch.Tensor | None


@dataclass(frozen=True)
class FrontRESDRSetup:
    dr_max: float
    dr_min: float
    dr_ema_alpha: float
    dr_scale_init: float
    dr_scale: float
    r_delta_ema: float
    perturb_target: Any | None


@dataclass(frozen=True)
class FrontRESDRScaleEnvPlan:
    scale_vector: torch.Tensor | None
    mix_mode: str
    diag: dict[str, float]
    mix_class: torch.Tensor | None = None


@dataclass(frozen=True)
class FrontRESDRIterationPlan:
    dr_scale: float
    r_delta_ema: float
    effective_dr_scale: float | None
    dr_mix_mode: str | None
    mix_diag: dict[str, float] | None
    critic_warmup: bool
    actor_takeover_active: bool
    hsl_boundary_available: bool
    use_boundary: bool
    gmt_frontier_score: float | None
    gmt_frontier_decision: str | None
    applied: bool


def resolve_frontres_mode_state(runner: Any, policy_cls: type) -> FrontRESModeState:
    """Resolve the current FrontRES objective and task-space mode."""

    is_frontres = isinstance(runner.alg.policy, policy_cls)
    training_objective = str(getattr(
        runner.alg,
        "frontres_training_objective",
        runner.cfg.get("frontres_training_objective", "ppo_hrl"),
    )).lower()
    supervised_restore = (
        is_frontres and training_objective in ("supervised_restore", "basis_restore")
    )
    hsl_restore = (
        is_frontres
        and training_objective in ("supervised_restore", "basis_restore", "hsl_hybrid")
    )
    is_task_space_mode = (
        is_frontres and getattr(runner.alg.policy, "num_task_corrections", 0) > 0
    )
    return FrontRESModeState(
        is_frontres=is_frontres,
        training_objective=training_objective,
        supervised_restore=supervised_restore,
        hsl_restore=hsl_restore,
        is_task_space_mode=is_task_space_mode,
        critic_warmup_iters=int(runner.cfg.get("critic_warmup_iterations", 0)),
    )


def frontres_ppo_actor_weight_for_iter(
    runner: Any,
    *,
    iteration: int,
    is_frontres: bool,
    supervised_restore: bool,
) -> float:
    """Linear supervised-to-PPO takeover schedule for the current iteration."""

    if not (is_frontres and hasattr(runner.alg, "ppo_actor_weight")):
        return 1.0
    if supervised_restore:
        return 0.0
    actor_warmup = int(runner.alg_cfg.get(
        "ppo_actor_warmup_iterations",
        runner.cfg.get("ppo_actor_warmup_iterations", 0),
    ))
    actor_ramp = int(runner.alg_cfg.get(
        "ppo_actor_ramp_iterations",
        runner.cfg.get("ppo_actor_ramp_iterations", 0),
    ))
    phase_iter = max(0, iteration)
    if phase_iter < actor_warmup:
        return 0.0
    if actor_ramp > 0 and phase_iter < actor_warmup + actor_ramp:
        weight = (phase_iter - actor_warmup + 1) / float(actor_ramp)
        return max(0.0, min(1.0, weight))
    return 1.0


def frontres_curriculum_allowed_bases(runner: Any) -> tuple[str, ...]:
    """Map the active FrontRES output dimensions to repairable perturbation families."""

    return allowed_perturbation_bases(runner.cfg.get("frontres_active_task_dims", None))


def frontres_curriculum_choices(
    runner: Any,
    *,
    progress: float,
    seq_idx: int,
    is_frontres: bool,
) -> tuple[list[tuple[str, ...]], str]:
    stats = getattr(runner, "_frontres_boundary_ema", None)
    if stats is None:
        stats = getattr(runner, "_last_frontres_boundary_stats", None)
    return choose_perturbation_choices(
        runner.cfg,
        runner.cfg.get("frontres_active_task_dims", None),
        progress,
        seq_idx,
        boundary_stats=stats,
        is_frontres=is_frontres,
    )


def frontres_warmup_perturbation_mode_groups(
    runner: Any,
    *,
    seq_idx: int,
) -> list[tuple[str, ...]]:
    """Return perturbation families to mix inside one warmup update."""

    return warmup_perturbation_mode_groups(
        runner.cfg,
        runner.cfg.get("frontres_active_task_dims", None),
        seq_idx,
        current_active_modes=tuple(getattr(runner, "_frontres_curriculum_active_modes", ())),
    )


def frontres_mixed_dr_scale(
    runner: Any,
    *,
    frontier_scale: float,
    enabled: bool,
    seq_idx: int,
    dr_min: float,
    dr_max: float,
) -> tuple[float, str]:
    """Sample one easy/frontier/hard perturbation magnitude around the frontier."""

    return sample_scalar_dr_strength(
        runner.cfg,
        frontier_scale,
        enabled,
        seq_idx,
        dr_min=dr_min,
        dr_max=dr_max,
    )


def frontres_mixed_dr_scale_env(
    runner: Any,
    *,
    frontier_scale: float,
    enabled: bool,
    seq_idx: int,
    n_train: int,
    n_candidate: int,
    n_base: int,
    dr_min: float,
    dr_max: float,
) -> FrontRESDRScaleEnvPlan:
    """Sample per-env easy/frontier/hard DR strength for split rollout branches."""

    plan = sample_per_env_dr_strength(
        runner.cfg,
        frontier_scale,
        enabled,
        seq_idx,
        n_train=int(n_train),
        n_candidate=int(n_candidate),
        n_base=int(n_base),
        num_envs=runner.env.num_envs,
        dr_min=dr_min,
        dr_max=dr_max,
    )
    if plan.scale_vector is None:
        return FrontRESDRScaleEnvPlan(None, plan.mix_mode, plan.diag)
    scales = torch.tensor(plan.scale_vector, device=runner.device, dtype=torch.float32)
    mix_class = None
    if plan.mix_class is not None:
        mix_class = torch.tensor(
            plan.mix_class,
            device=runner.device,
            dtype=torch.long,
        )
    return FrontRESDRScaleEnvPlan(scales, plan.mix_mode, plan.diag, mix_class)


def _frontres_curriculum_progress(runner: Any, iteration: int) -> float:
    curriculum_iters = int(runner.cfg.get("frontres_curriculum_total_iterations", 1500))
    curriculum_iters = max(1, curriculum_iters)
    return min(1.0, max(0.0, int(iteration) / float(curriculum_iters)))


def _frontres_boundary_scale_step(
    runner: Any,
    *,
    ema: dict[str, float],
    dr_scale: float,
    dr_min: float,
    dr_max: float,
) -> float:
    safe = float(ema.get("safe", 0.0))
    repair = float(ema.get("repair", ema.get("fragile", 0.0)))
    broken = float(ema.get("broken", 0.0))
    gainpos = float(ema.get("positive_gain", 0.5))
    safe_hi = float(runner.cfg.get("frontres_boundary_safe_high", 0.45))
    broken_hi = float(runner.cfg.get("frontres_boundary_broken_high", 0.35))
    broken_target = float(runner.cfg.get("frontres_boundary_broken_target", 0.25))
    repair_lo = float(runner.cfg.get(
        "frontres_boundary_repair_low",
        runner.cfg.get("frontres_boundary_fragile_low", 0.45),
    ))
    repair_hi = float(runner.cfg.get(
        "frontres_boundary_repair_high",
        runner.cfg.get("frontres_boundary_fragile_high", 0.70),
    ))
    gain_hi = float(runner.cfg.get("frontres_boundary_positive_gain_high", 0.55))
    gain_lo = float(runner.cfg.get("frontres_boundary_positive_gain_low", 0.45))
    step = float(runner.cfg.get("frontres_boundary_dr_step", 0.03))
    factor = 1.0
    if broken > broken_hi:
        factor = 1.0 - step * min(3.0, 1.0 + (broken - broken_hi) / max(1.0 - broken_hi, 1e-6))
    elif safe > safe_hi and broken < broken_target:
        factor = 1.0 + step
    elif (repair_lo <= repair <= repair_hi) and gainpos > gain_hi and broken < broken_hi:
        factor = 1.0 + 0.5 * step
    elif gainpos < gain_lo and broken > broken_target:
        factor = 1.0 - 0.5 * step
    factor = max(0.80, min(1.10, factor))
    return max(dr_min, min(dr_max, dr_scale * factor))


def _frontres_pi_scale_step(
    runner: Any,
    *,
    dr_scale: float,
    dr_min: float,
    dr_max: float,
    r_delta_ema: float,
) -> float:
    kp = float(runner.cfg.get("dr_p_gain", 0.10))
    ki = float(runner.cfg.get("dr_i_gain", 0.01))
    dr_target = float(runner.cfg.get("dr_target_r_delta", 0.01))
    error = r_delta_ema - dr_target
    prev_err = getattr(runner, "_dr_prev_error", error)
    delta = kp * (error - prev_err) + ki * error
    runner._dr_prev_error = error
    return max(dr_min, min(dr_max, dr_scale + delta))


__all__ = [
    "FrontRESModeState",
    "FrontRESPairLayout",
    "FrontRESDRSetup",
    "FrontRESDRIterationPlan",
    "FrontRESDRScaleEnvPlan",
    "GMTFrontierState",
    "allowed_perturbation_bases",
    "frontres_curriculum_hash",
    "frontres_curriculum_allowed_bases",
    "frontres_curriculum_choices",
    "frontres_mixed_dr_scale",
    "frontres_mixed_dr_scale_env",
    "frontres_ppo_actor_weight_for_iter",
    "frontres_warmup_perturbation_mode_groups",
    "mode_complexity",
    "resolve_frontres_mode_state",
    "score_gmt_frontier",
    "sample_perturbation_mix",
    "update_boundary_ema",
    "update_gmt_frontier_state",
    "_frontres_boundary_scale_step",
    "_frontres_curriculum_progress",
    "_frontres_pi_scale_step",
]
