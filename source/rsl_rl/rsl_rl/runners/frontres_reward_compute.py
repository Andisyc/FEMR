# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone FrontRES Reward Compute debug harness.

This file intentionally copies the runner-side Reward Compute sequence without
being imported by ``on_policy_runner.py``.  It builds simple hand-checkable
FrontRES samples, then runs:

    rho update weight -> alpha groundtruth -> rho advantage -> reward

Run from the repository root with:

    python source/rsl_rl/rsl_rl/runners/frontres_reward_compute.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.frontres.frontres_action_cone import FrontRESActionCone
from rsl_rl.frontres.frontres_alpha_rho_bridge import FrontRESAlphaRhoBridge
from rsl_rl.frontres.frontres_oracle import compute_frontres_oracle_upper_bound
from rsl_rl.frontres.frontres_reward_diagnostics import (
    accumulate_frontres_reward_diagnostics,
    initialize_frontres_reward_diagnostic_sums,
    materialize_frontres_reward_diagnostic_means,
)
from rsl_rl.frontres.frontres_reward_window import (
    FrontRESRewardContext,
    FrontRESRewardWindow,
    build_frontres_reward_window,
    compose_frontres_reward_delta,
)
from rsl_rl.frontres.frontres_rollout_evidence import compute_frontres_rollout_evidence
from rsl_rl.frontres.frontres_transition_payload import (
    write_alpha_groundtruth,
    write_rho_update_weight,
    write_rho_advantage,
)


@dataclass
class FrontRESPostStepResult:
    rewards: torch.Tensor
    reward_window: FrontRESRewardWindow | None
    r_raw_gmt: torch.Tensor | None
    r_candidate_gmt: torch.Tensor | None
    r_clean_gmt: torch.Tensor | None
    prev_delta_q: torch.Tensor | None
    term_count: int
    step_count: int


def compute_frontres_reward(
    runner: Any,
    *,
    locs: dict[str, Any],
    reward_context: FrontRESRewardContext,
    accept_payload: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    actions: torch.Tensor,
    diagnostic_sums: dict[str, float | int],
    prev_delta_q: torch.Tensor | None,
    term_count: int,
    step_count: int,
) -> FrontRESPostStepResult:
    """Copied debug version of the runner-side task-space reward connector."""

    n_train = int(locs["N_train"])
    n_candidate = int(locs["N_candidate"])
    reward_window = reward_context.reward_window
    reward_window = compose_frontres_reward_delta(
        cfg=runner.cfg,
        reward_window=reward_window,
        n_train=n_train,
        n_exec=reward_context.n_exec,
        n_candidate=n_candidate,
        repair_gain=reward_context.repair_gain,
        candidate_gain=reward_context.candidate_gain,
        projection_gain=reward_context.projection_gain,
        r_step=reward_context.r_step,
        r_rescue=reward_context.r_rescue,
        intervention_cost=reward_context.intervention_cost,
        overcorrection_cost=reward_context.overcorrection_cost,
        w_exec=reward_context.w_exec,
        repair_scale=reward_context.repair_scale,
        w_geom=reward_context.w_geom,
        w_rescue=reward_context.w_rescue,
        w_exec_harm=reward_context.w_exec_harm,
        device=runner.device,
    )
    r_delta = reward_window.r_delta
    if r_delta is None:
        raise RuntimeError("Debug Reward Compute failed to compose r_delta.")

    rewards_mod = rewards.clone()
    rewards_mod[:n_train] = r_delta
    rewards_mod[n_train:] = 0.0

    diagnostic_locs = dict(locs)
    diagnostic_locs.update(
        {
            "_n_exec": reward_context.n_exec,
            "_n_pair": reward_context.n_pair,
            "_base_start": reward_context.base_start,
            "_base_end": reward_context.base_end,
            "_candidate_start": reward_context.candidate_start,
            "_candidate_end": reward_context.candidate_end,
            "_clean_start": reward_context.clean_start,
            "_clean_end": reward_context.clean_end,
            "_r_step": reward_context.r_step,
            "_r_rescue": reward_context.r_rescue,
            "_r_exec": reward_context.r_exec,
            "_r_z": reward_context.r_z,
            "_r_xy": reward_context.r_xy,
            "_r_rp": reward_context.r_rp,
            "_r_ya": reward_context.r_ya,
            "_dr_z_abs_log": reward_context.dr_z_abs_log,
            "_dr_xy_abs_log": reward_context.dr_xy_abs_log,
            "_dr_rp_abs_log": reward_context.dr_rp_abs_log,
            "_dr_yaw_abs_log": reward_context.dr_yaw_abs_log,
            "_corr_z_abs_log": reward_context.corr_z_abs_log,
            "_corr_xy_abs_log": reward_context.corr_xy_abs_log,
            "_corr_rp_abs_log": reward_context.corr_rp_abs_log,
            "_corr_yaw_abs_log": reward_context.corr_yaw_abs_log,
            "_rot_raw_to_clean": reward_context.rot_raw_to_clean,
            "_rot_raw_to_fr": reward_context.rot_raw_to_fr,
            "_e_raw": reward_context.e_raw,
            "_e_fr": reward_context.e_fr,
            "_exec_score_all": reward_context.exec_score_all,
            "_exec_components": reward_context.exec_components,
            "_feasible_components": reward_context.feasible_components,
            "_mode_groups": reward_context.mode_groups,
            "_exec_frontres": reward_context.exec_frontres,
            "_exec_candidate": reward_context.exec_candidate,
            "_exec_perturbed": reward_context.exec_perturbed,
            "_exec_clean": reward_context.exec_clean,
            "_exec_feasible": reward_context.exec_feasible,
            "_exec_planar_log": reward_context.exec_planar_log,
            "_exec_vertical_log": reward_context.exec_vertical_log,
            "_exec_task_log": reward_context.exec_task_log,
            "_intervention_cost": reward_context.intervention_cost,
            "_clean_bound_cost": reward_context.clean_bound_cost,
            "_side_cost": reward_context.side_cost,
            "_over_cost": reward_context.over_cost,
            "_overcorrection_cost": reward_context.overcorrection_cost,
            "_under_repair_penalty": reward_context.under_repair_penalty,
            "_action_activity": reward_context.action_activity,
            "_w_exec": reward_context.w_exec,
            "_repair_scale": reward_context.repair_scale,
            "_w_geom": reward_context.w_geom,
            "_w_rescue": reward_context.w_rescue,
            "_w_exec_harm": reward_context.w_exec_harm,
            "_repair_gain": reward_context.repair_gain,
            "_candidate_gain": reward_context.candidate_gain,
            "_projection_gain": reward_context.projection_gain,
            "_oracle_ub_gain": reward_context.oracle_ub_gain,
            "_oracle_ub_pass": reward_context.oracle_ub_pass,
            "_oracle_ub_noisy_win": reward_context.oracle_ub_noisy_win,
            "_oracle_ub_projected_win": reward_context.oracle_ub_projected_win,
            "_oracle_ub_candidate_win": reward_context.oracle_ub_candidate_win,
            "_oracle_ub_feasible_win": reward_context.oracle_ub_feasible_win,
            "_exec_floor": reward_context.exec_floor,
            "_exec_safe_floor": reward_context.exec_safe_floor,
            "_exec_floor_source": reward_context.exec_floor_source,
            "_candidate_floor_margin": reward_context.candidate_floor_margin,
            "_candidate_floor_pass": reward_context.candidate_floor_pass,
            "_candidate_floor_pass_frac": reward_context.candidate_floor_pass_frac,
            "_stable_route_next": reward_context.stable_route_next,
            "_stable_route_active": reward_context.stable_route_active,
            "_damage_gap": reward_window.damage_gap,
            "_oracle_clean_gap": reward_window.oracle_clean_gap,
            "_oracle_trust": reward_window.oracle_trust,
            "_repair_ratio": reward_window.repair_ratio,
            "_safe_score": reward_window.safe_score,
            "_repairable_score": reward_window.repairable_score,
            "_broken_score": reward_window.broken_score,
            "_safe_gate": reward_window.safe_score,
            "_repair_gate": reward_window.repairable_score,
            "_broken_gate": reward_window.broken_score,
            "_window_mu": reward_window.window_mu,
            "_exec_gate": reward_window.exec_gate,
            "_cost_gate": reward_window.cost_gate,
            "_safe_frac": reward_window.safe_frac,
            "_repair_frac": reward_window.repair_frac,
            "_broken_frac": reward_window.broken_frac,
            "_safe_gap": reward_window.safe_gap,
            "_broken_gap": reward_window.broken_gap,
            "_learnable_route_mask": reward_window.learnable_route_mask,
            "_exec_weight": reward_window.exec_weight,
            "_cost_weight": reward_window.cost_weight,
            "_rho_update_weight": reward_window.rho_update_weight,
            "_actor_gate": reward_window.rho_update_weight,
            "_harm_penalty": reward_window.harm_penalty,
            "_harm_penalty_exec": reward_window.harm_penalty_exec,
            "_harm_mag": reward_window.harm_mag,
            "_cost_exec": reward_window.cost_exec,
            "_effective_gain_bonus": reward_window.effective_gain_bonus,
            "_effective_gain_bonus_exec": reward_window.effective_gain_bonus_exec,
            "_reward_progress": reward_window.reward_progress,
            "_constraint_progress": reward_window.constraint_progress,
            "_accept_pref_target": accept_payload.accept_target,
            "_accept_pref_mask": accept_payload.accept_mask,
            "_pref_full_frac": accept_payload.pref_full_frac,
            "_pref_noop_frac": accept_payload.pref_noop_frac,
            "_pref_keep_frac": accept_payload.pref_keep_frac,
            "_pref_ignore_frac": accept_payload.pref_ignore_frac,
            "_pref_margin_mean": accept_payload.pref_margin_mean,
            "_pref_need_mean": accept_payload.pref_need_mean,
            "_pref_admiss_mean": accept_payload.pref_admiss_mean,
            "_pref_target_mean": accept_payload.pref_target_mean,
            "_tri_weight_repair_mean": accept_payload.tri_weight_repair_mean,
            "_tri_weight_noisy_mean": accept_payload.tri_weight_noisy_mean,
            "_tri_weight_stable_mean": accept_payload.tri_weight_stable_mean,
            "_pref_inertial_penalty_rho_mean": accept_payload.pref_inertial_penalty_rho_mean,
            "_pref_inertial_penalty_one_mean": accept_payload.pref_inertial_penalty_one_mean,
            "_rho_target_planar_mean": accept_payload.rho_target_planar_mean,
            "_rho_target_rp_mean": accept_payload.rho_target_rp_mean,
            "_rho_target_z_mean": accept_payload.rho_target_z_mean,
            "_rho_target_spread_mean": accept_payload.rho_target_spread_mean,
            "_grouped_rho_mask_mean": accept_payload.grouped_rho_mask_mean,
            "_rho_regret_up_planar_mean": accept_payload.rho_regret_up_planar_mean,
            "_rho_regret_up_rp_mean": accept_payload.rho_regret_up_rp_mean,
            "_rho_regret_up_z_mean": accept_payload.rho_regret_up_z_mean,
            "_rho_regret_down_planar_mean": accept_payload.rho_regret_down_planar_mean,
            "_rho_regret_down_rp_mean": accept_payload.rho_regret_down_rp_mean,
            "_rho_regret_down_z_mean": accept_payload.rho_regret_down_z_mean,
            "r_delta": r_delta,
            "r_raw_gmt": reward_context.r_raw_gmt,
            "r_candidate_gmt": reward_context.r_candidate_gmt,
            "r_clean_gmt": reward_context.r_clean_gmt,
            "_reward_window": reward_window,
            "_under_write": reward_window.under_write,
            "_ranking_reward": reward_window.ranking_reward,
            "_r_frontres_log": reward_context.exec_frontres.mean(),
            "_r_clean_log": reward_context.exec_clean.mean(),
            "_r_oracle_log": reward_context.exec_feasible.mean(),
            "_r_base_log": reward_context.exec_perturbed.mean(),
            "_r_rescue_log": reward_context.r_rescue.mean(),
            "_smooth_penalty": None,
            "_reg_penalty": None,
        }
    )
    accumulate_frontres_reward_diagnostics(runner, diagnostic_sums, diagnostic_locs)

    term_count += int((dones[:n_train] > 0).sum().item())
    step_count += n_train
    prev_delta_q = actions.clone()
    return FrontRESPostStepResult(
        rewards=rewards_mod,
        reward_window=reward_window,
        r_raw_gmt=reward_context.r_raw_gmt,
        r_candidate_gmt=reward_context.r_candidate_gmt,
        r_clean_gmt=reward_context.r_clean_gmt,
        prev_delta_q=prev_delta_q,
        term_count=term_count,
        step_count=step_count,
    )


@dataclass(frozen=True)
class DebugSample:
    name: str
    exec_noisy: float
    exec_frontres: float
    exec_candidate: float
    exec_clean: float
    rho_action: float
    expected: str


class FakeRunner:
    """Minimal runner surface needed by Reward Compute helpers."""

    def __init__(self, *, cfg: dict[str, Any], num_envs: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.env = SimpleNamespace(
            num_envs=num_envs,
            command_manager=SimpleNamespace(_terms={}),
        )
        self.alg = SimpleNamespace(
            transition=SimpleNamespace(),
            lambda_smooth=0.0,
            policy=SimpleNamespace(
                task_conf_dim=6,
                last_task_correction=torch.zeros(num_envs, 6, device=device),
            ),
        )
        self._frontres_action_cone = FrontRESActionCone(cfg, self.alg)
        self._frontres_alpha_rho_bridge = FrontRESAlphaRhoBridge()
        self._frontres_exec_floor_value_last = float(cfg["frontres_state_alpha_exec_floor"])
        self._frontres_exec_floor_safe_last = float(cfg["frontres_state_alpha_safe_exec_floor"])
        self._frontres_stable_route_next_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._frontres_stable_route_active_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._frontres_stable_route_applied_frac = 0.0
        self._frontres_stable_endpoint_frac = 0.0

    def _frontres_structured_joint_effective_enabled(self) -> bool:
        return bool(self.cfg.get("frontres_structured_joint_rl_enabled", False)) and float(
            self.cfg.get("frontres_structured_joint_rl_weight", 0.0)
        ) > 0.0


def _debug_cfg() -> dict[str, Any]:
    return {
        # Match the formal FrontRES config in rsl_rl_mosaic_cfg.py.  This
        # harness is for checking live training semantics, not toy curves.
        "frontres_gap_floor_per_step": 0.005,
        "frontres_safe_gap_per_step": 0.003,
        "frontres_broken_gap_per_step": 0.10,
        "frontres_gap_gate_temp": 0.005,
        "frontres_oracle_clean_gap_threshold": 1.0e9,
        "frontres_reward_scale_dr_reference": 1.25,
        "frontres_reward_progress_min": 0.0,
        "frontres_constraint_progress_exponent": 2.0,
        "frontres_selective_reward_enabled": True,
        "frontres_exec_reward_signal": "gain",
        "frontres_exec_reward_weight": 1.0,
        "frontres_repair_reward_scale": 1.0,
        "frontres_geometry_reward_weight": 0.0,
        "frontres_rescue_reward_weight": 0.0,
        "frontres_executable_harm_weight": 1.0,
        "frontres_harm_epsilon": 0.001,
        "frontres_harm_penalty_weight": 1.0,
        "frontres_side_harm_weight": 0.0,
        "frontres_harm_action_cost_floor": 0.001,
        "frontres_harm_action_cost_ref": 0.01,
        "frontres_side_actor_gate_weight": 0.05,
        "frontres_min_effective_gain": 0.008,
        "frontres_effective_gain_bonus_weight": 0.0,
        "frontres_candidate_ranking_reward_enabled": True,
        "frontres_candidate_ranking_reward_weight": 1.0,
        "frontres_candidate_underwrite_weight": 1.0,
        "frontres_candidate_projection_weight": 0.25,
        "frontres_candidate_harm_weight": 1.0,
        "frontres_rho_space": "noisy_to_repair",
        "frontres_acceptance_preference_enabled": True,
        "frontres_acceptance_preference_margin": 0.003,
        "frontres_acceptance_regret_target_enabled": True,
        "frontres_acceptance_regret_soft_mask_floor": 1.0,
        "frontres_acceptance_regret_oracle_trust_floor": 0.25,
        "frontres_acceptance_regret_per_mode_soft_floor": 1.0,
        "frontres_acceptance_rho_target_temp": 0.08,
        "frontres_acceptance_calibration_step": 0.5,
        "frontres_grouped_rho_target_enabled": True,
        "frontres_per_mode_acceptance_preference_mask": True,
        "frontres_active_task_dims": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "frontres_structured_joint_rl_enabled": True,
        "frontres_structured_joint_rl_weight": 1.0,
        "frontres_structured_joint_rl_adv_clip": 5.0,
        "frontres_structured_joint_directional_weight": 1.0,
        "frontres_structured_joint_rho_center": 0.5,
        "frontres_structured_joint_center_drive_deadzone": 0.10,
        "frontres_structured_joint_retention_prior_weight": 0.0,
        "frontres_structured_joint_floor_penalty_weight": 5.0,
        "frontres_structured_joint_full_repair_bonus_weight": 1.0,
        "frontres_structured_joint_weight_floor": 0.10,
        "frontres_structured_joint_use_sample_weight": True,
        "frontres_structured_joint_use_actor_gate_weight": True,
        "frontres_debug_rho_prior_beta": 0.5,
        "frontres_debug_repairable_prior_weight": 0.5,
        "frontres_state_alpha_enabled": False,
        "frontres_state_alpha_route_enabled": False,
        "frontres_stable_route_enabled": False,
        "frontres_state_alpha_exec_floor": 0.0,
        "frontres_state_alpha_safe_exec_floor": 0.05,
        "frontres_state_alpha_temp": 0.08,
        "frontres_mixed_dr_easy_weight": 0.45,
        "frontres_mixed_dr_frontier_weight": 0.40,
        "frontres_mixed_dr_hard_weight": 0.15,
        "frontres_mixed_dr_easy_factor": 0.75,
        "frontres_mixed_dr_frontier_factor": 1.00,
        "frontres_mixed_dr_hard_factor": 1.08,
        "frontres_oracle_upper_bound_diag_enabled": True,
        "frontres_oracle_upper_bound_margin": 0.0,
    }


def _debug_samples() -> list[DebugSample]:
    return [
        DebugSample("safe", 0.950, 0.949, 0.952, 0.952, 0.75, "prior should punish high rho"),
        DebugSample("raise_rho", 0.920, 0.945, 0.970, 0.980, 0.75, "evidence should reward high rho"),
        DebugSample("lower_rho", 0.930, 0.900, 0.880, 0.980, 0.75, "evidence should punish high rho"),
        DebugSample("keep_rho", 0.920, 0.965, 0.955, 0.980, 0.25, "evidence should avoid pushing higher rho"),
        DebugSample("deep_broken", 0.750, 0.760, 0.770, 0.980, 0.75, "prior should punish high rho unless evidence is strong"),
    ]


def _branch_scores(samples: list[DebugSample], device: torch.device) -> tuple[torch.Tensor, ...]:
    noisy = torch.tensor([s.exec_noisy for s in samples], device=device)
    frontres = torch.tensor([s.exec_frontres for s in samples], device=device)
    candidate = torch.tensor([s.exec_candidate for s in samples], device=device)
    clean = torch.tensor([s.exec_clean for s in samples], device=device)
    return noisy, frontres, candidate, clean


def _repeat_layout(
    *,
    frontres: torch.Tensor,
    candidate: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([frontres, candidate, noisy, clean], dim=0)


def build_debug_frontres_truth(
    runner: FakeRunner,
    samples: list[DebugSample],
) -> tuple[FrontRESRewardContext, torch.Tensor, torch.Tensor, dict[str, Any]]:
    device = runner.device
    n_train = len(samples)
    n_candidate = n_base = n_clean = n_train
    candidate_start = n_train
    candidate_end = candidate_start + n_candidate
    base_start = candidate_end
    base_end = base_start + n_base
    clean_start = base_end
    clean_end = clean_start + n_clean
    num_envs = clean_end

    exec_perturbed, exec_frontres, exec_candidate, exec_clean = _branch_scores(samples, device)
    rewards = _repeat_layout(
        frontres=exec_frontres,
        candidate=exec_candidate,
        noisy=exec_perturbed,
        clean=exec_clean,
    )
    dones = torch.zeros(num_envs, device=device)
    infos: dict[str, Any] = {}

    exec_components = {
        name: rewards.clone()
        for name in ("planar", "vertical", "task", "xy", "yaw", "rp", "z")
    }
    feasible_components = {name: rewards.clone() for name in exec_components}
    exec_feasible = exec_clean.clone()
    mode_groups = [("planar", "yaw", "global_z", "local_rp")] * n_train
    rollout_evidence = compute_frontres_rollout_evidence(
        noisy_score=exec_perturbed,
        projected_score=exec_frontres,
        candidate_score=exec_candidate,
    )
    oracle_ub = compute_frontres_oracle_upper_bound(
        exec_perturbed,
        exec_frontres,
        exec_candidate,
        exec_feasible,
        margin=0.0,
        enabled=True,
    )

    intervention_cost = torch.zeros(n_train, device=device)
    action_activity = torch.full((n_train,), 0.01, device=device)
    under_repair_penalty = torch.zeros(n_train, device=device)
    e_raw = (exec_clean - exec_perturbed).clamp(min=0.0)
    e_fr = (exec_clean - exec_frontres).clamp(min=0.0)
    reward_window = build_frontres_reward_window(
        runner=runner,
        cfg=runner.cfg,
        n_train=n_train,
        n_exec=n_train,
        exec_clean=exec_clean,
        exec_perturbed=exec_perturbed,
        exec_feasible=exec_feasible,
        exec_frontres=exec_frontres,
        repair_gain=rollout_evidence.repair_gain,
        mode_groups=mode_groups,
        e_raw=e_raw,
        e_fr=e_fr,
        intervention_cost=intervention_cost,
        action_activity=action_activity,
        under_repair_penalty=under_repair_penalty,
        dr_scale=1.0,
        ppo_actor_weight_current=1.0,
        stable_route_active_mask=torch.zeros(n_train, dtype=torch.bool, device=device),
        device=device,
    )

    zero = torch.zeros(n_train, device=device)
    zero_vec = torch.zeros(num_envs, 3, device=device)
    identity_quat = torch.zeros(num_envs, 4, device=device)
    identity_quat[:, 0] = 1.0
    actions = torch.zeros(num_envs, 12, device=device)
    rho_actions = torch.tensor([s.rho_action for s in samples], device=device).view(-1, 1)
    actions[:n_train, 6:12] = rho_actions.expand(-1, 6)

    context = FrontRESRewardContext(
        candidate_start=candidate_start,
        candidate_end=candidate_end,
        base_start=base_start,
        base_end=base_end,
        clean_start=clean_start,
        clean_end=clean_end,
        n_exec=n_train,
        n_pair=n_train,
        r_raw_gmt=exec_perturbed,
        r_clean_gmt=exec_clean,
        r_candidate_gmt=exec_candidate,
        r_total=exec_frontres,
        cmd=SimpleNamespace(),
        use_clean=True,
        a_w=zero_vec,
        a_raw=zero_vec,
        a_fr=zero_vec,
        q_w=identity_quat,
        q_raw=identity_quat,
        q_fr=identity_quat,
        r_z=zero,
        r_xy=zero,
        r_rp=zero,
        r_ya=zero,
        r_step=zero,
        r_rescue=zero,
        r_exec=reward_window.r_exec,
        dr_z_abs_log=torch.tensor(0.0, device=device),
        dr_xy_abs_log=torch.tensor(0.0, device=device),
        dr_rp_abs_log=torch.tensor(0.0, device=device),
        dr_yaw_abs_log=torch.tensor(0.0, device=device),
        corr_z_abs_log=torch.tensor(0.0, device=device),
        corr_xy_abs_log=torch.tensor(0.0, device=device),
        corr_rp_abs_log=torch.tensor(0.0, device=device),
        corr_yaw_abs_log=torch.tensor(0.0, device=device),
        rot_raw_to_clean=torch.zeros(n_train, 3, device=device),
        rot_raw_to_fr=torch.zeros(n_train, 3, device=device),
        e_raw=e_raw,
        e_fr=e_fr,
        exec_score_all=rewards,
        exec_components=exec_components,
        feasible_components=feasible_components,
        mode_groups=mode_groups,
        exec_frontres=exec_frontres,
        exec_candidate=exec_candidate,
        exec_perturbed=exec_perturbed,
        exec_clean=exec_clean,
        exec_feasible=exec_feasible,
        exec_planar_log=exec_frontres.mean(),
        exec_vertical_log=exec_frontres.mean(),
        exec_task_log=exec_frontres.mean(),
        intervention_cost=intervention_cost,
        clean_bound_cost=torch.zeros(n_train, device=device),
        side_cost=torch.zeros(n_train, device=device),
        over_cost=torch.zeros(n_train, device=device),
        overcorrection_cost=torch.zeros(n_train, device=device),
        under_repair_penalty=reward_window.under_repair_penalty,
        action_activity=action_activity,
        w_exec=1.0,
        repair_scale=1.0,
        w_geom=0.0,
        w_rescue=0.0,
        w_exec_harm=1.0,
        repair_gain=rollout_evidence.repair_gain,
        candidate_gain=rollout_evidence.candidate_gain,
        projection_gain=rollout_evidence.projection_gain,
        oracle_ub_gain=oracle_ub.gain,
        oracle_ub_pass=oracle_ub.pass_mask,
        oracle_ub_noisy_win=oracle_ub.noisy_win,
        oracle_ub_projected_win=oracle_ub.projected_win,
        oracle_ub_candidate_win=oracle_ub.candidate_win,
        oracle_ub_feasible_win=oracle_ub.feasible_win,
        exec_floor=runner._frontres_exec_floor_value_last,
        exec_safe_floor=runner._frontres_exec_floor_safe_last,
        exec_floor_source="debug-fixed",
        candidate_floor_margin=exec_candidate - runner._frontres_exec_floor_value_last,
        candidate_floor_pass=(exec_candidate >= runner._frontres_exec_floor_value_last).float(),
        candidate_floor_pass_frac=(exec_candidate >= runner._frontres_exec_floor_value_last).float().mean(),
        stable_route_next=torch.zeros(n_train, dtype=torch.bool, device=device),
        stable_route_active=torch.zeros(n_train, dtype=torch.bool, device=device),
        reward_window=reward_window,
    )
    return context, actions, dones, infos


def _quat_passthrough(q: torch.Tensor) -> torch.Tensor:
    return torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)


def _quat_mul_identity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return b


def _quat_inv_identity(q: torch.Tensor) -> torch.Tensor:
    return q


def _fmt_tensor(values: torch.Tensor, digits: int = 3) -> str:
    flat = values.detach().cpu().view(-1).tolist()
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in flat) + "]"


def _print_debug_boundary(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_sample_inputs(samples: list[DebugSample]) -> None:
    _print_debug_boundary("A. manual sample inputs")
    print("name       noisy  frontres  candidate  clean  damage_gap  repair_gain  candidate_gain")
    print("-" * 86)
    for sample in samples:
        damage_gap = max(sample.exec_clean - sample.exec_noisy, 0.0)
        repair_gain = sample.exec_frontres - sample.exec_noisy
        candidate_gain = sample.exec_candidate - sample.exec_noisy
        print(
            f"{sample.name:<10}"
            f"{sample.exec_noisy:>5.3f}  "
            f"{sample.exec_frontres:>8.3f}  "
            f"{sample.exec_candidate:>9.3f}  "
            f"{sample.exec_clean:>5.3f}  "
            f"{damage_gap:>10.3f}  "
            f"{repair_gain:>11.3f}  "
            f"{candidate_gain:>14.3f}"
        )


def _print_live_config(cfg: dict[str, Any]) -> None:
    _print_debug_boundary("0. live branch/config used by this debug harness")
    print(f"rho_space: {cfg['frontres_rho_space']}")
    print(f"structured_rho_advantage_learning: {cfg['frontres_structured_joint_rl_enabled']}")
    print(f"use_rho_update_weight (legacy config key): {cfg['frontres_structured_joint_use_sample_weight']}")
    print(
        "debug prior: "
        f"beta={cfg['frontres_debug_rho_prior_beta']}, "
        f"repairable_prior_weight={cfg['frontres_debug_repairable_prior_weight']}"
    )
    print(f"state_alpha_enabled: {cfg['frontres_state_alpha_enabled']}")
    print(f"stable_route_enabled: {cfg['frontres_stable_route_enabled']}")
    print(
        "gap floors/temp: "
        f"safe={cfg['frontres_safe_gap_per_step']}, "
        f"floor={cfg['frontres_gap_floor_per_step']}, "
        f"broken={cfg['frontres_broken_gap_per_step']}, "
        f"temp={cfg['frontres_gap_gate_temp']}"
    )
    print(
        "mixed DR bands: "
        f"easy={cfg['frontres_mixed_dr_easy_weight']}@{cfg['frontres_mixed_dr_easy_factor']}, "
        f"frontier={cfg['frontres_mixed_dr_frontier_weight']}@{cfg['frontres_mixed_dr_frontier_factor']}, "
        f"hard={cfg['frontres_mixed_dr_hard_weight']}@{cfg['frontres_mixed_dr_hard_factor']}"
    )


def _print_reward_window_debug(samples: list[DebugSample], truth: FrontRESRewardContext) -> None:
    window = truth.reward_window
    _print_debug_boundary("B. continuous sample-region scores")
    print(
        "name       damage_gap  safe_score  repair_score  broken_score  "
        "rho_update_weight exec_weight  cost_weight"
    )
    print("-" * 100)
    for i, sample in enumerate(samples):
        print(
            f"{sample.name:<10}"
            f"{window.damage_gap[i].item():>10.3f}  "
            f"{window.safe_score[i].item():>10.3f}  "
            f"{window.repairable_score[i].item():>12.3f}  "
            f"{window.broken_score[i].item():>12.3f}  "
            f"{window.rho_update_weight[i].item():>17.3f}  "
            f"{window.exec_weight[i].item():>11.3f}  "
            f"{window.cost_weight[i].item():>11.3f}"
        )


def _print_alpha_debug(
    samples: list[DebugSample],
    runner: Any,
    truth: FrontRESRewardContext,
    alpha_groundtruth: torch.Tensor,
    alpha_groundtruth_mask: torch.Tensor,
) -> None:
    _print_debug_boundary("C. alpha groundtruth")
    if not bool(runner.cfg.get("frontres_state_alpha_enabled", True)):
        print("alpha branch disabled in live config; target/mask should stay zero.")
    exec_floor = runner._frontres_exec_floor_value_last
    safe_floor = runner._frontres_exec_floor_safe_last
    temp = float(runner.cfg.get("frontres_state_alpha_temp", 0.08))
    floor_mid = 0.5 * (exec_floor + safe_floor)
    print(f"exec_floor={exec_floor:.3f}, safe_floor={safe_floor:.3f}, mid={floor_mid:.3f}, temp={temp:.3f}")
    print("alpha formula in code: sigmoid((mid - noisy_exec) / temp), active only below floor or above safe")
    print("name       noisy_exec  formula_alpha  written_alpha  alpha_mask")
    print("-" * 68)
    formula = torch.sigmoid((floor_mid - truth.exec_perturbed) / max(temp, 1.0e-6))
    for i, sample in enumerate(samples):
        print(
            f"{sample.name:<10}"
            f"{truth.exec_perturbed[i].item():>10.3f}  "
            f"{formula[i].item():>13.3f}  "
            f"{alpha_groundtruth[i, 0].item():>13.3f}  "
            f"{alpha_groundtruth_mask[i, 0].item():>10.1f}"
        )


def _rho_drive_from_action(rho_current: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    rho_center = max(0.0, min(1.0, float(cfg.get("frontres_structured_joint_rho_center", 0.5))))
    deadzone = max(0.0, float(cfg.get("frontres_structured_joint_center_drive_deadzone", 0.10)))
    centered = (2.0 * (rho_current.detach() - rho_center)).clamp(-1.0, 1.0)
    if deadzone > 1.0e-6:
        return torch.where(
            centered.abs() >= deadzone,
            torch.sign(centered),
            centered / deadzone,
        )
    return torch.sign(centered)


def _rho_loss_mask_from_action_cone(
    *,
    cfg: dict[str, Any],
    n_exec: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    active_dims = set(int(dim) for dim in cfg.get("frontres_active_task_dims", list(range(12))))
    # A rho dimension can train only if both its proposal dimension and its rho
    # action dimension are enabled by the action cone.
    per_dim = torch.tensor(
        [1.0 if (dim in active_dims and dim + 6 in active_dims) else 0.0 for dim in range(6)],
        device=device,
        dtype=dtype,
    )
    return per_dim.view(1, 6).expand(n_exec, 6).clone()


def apply_debug_prior_plus_evidence_rho_advantage(
    runner: Any,
    truth: FrontRESRewardContext,
    actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Debug-only rho advantage design from the note contract.

    This intentionally lives only in the standalone harness.  It tests the
    concept before we touch the formal training path.
    """
    cfg = runner.cfg
    n = int(truth.n_exec)
    rho_current = actions[:n, 6:12].detach().clamp(0.0, 1.0)
    rho_drive = _rho_drive_from_action(rho_current, cfg)
    pref_margin = float(cfg.get("frontres_acceptance_preference_margin", 0.0))

    candidate_regret = torch.relu(truth.exec_candidate[:n].detach() - truth.exec_frontres[:n].detach() - pref_margin)
    noisy_regret = torch.relu(truth.exec_perturbed[:n].detach() - truth.exec_frontres[:n].detach() - pref_margin)
    evidence_scale = (truth.exec_candidate[:n].detach() - truth.exec_perturbed[:n].detach()).abs() + pref_margin + 1.0e-6
    evidence_direction = ((candidate_regret - noisy_regret) / evidence_scale).clamp(-1.0, 1.0)
    rho_evidence_advantage = evidence_direction.view(-1, 1) * rho_drive

    window = truth.reward_window
    repairable_prior_weight = float(cfg.get("frontres_debug_repairable_prior_weight", 0.5))
    prior_direction = (
        repairable_prior_weight * window.repairable_score[:n].detach()
        - window.safe_score[:n].detach()
        - window.broken_score[:n].detach()
    ).clamp(-1.0, 1.0)
    rho_prior_advantage = prior_direction.view(-1, 1) * rho_drive
    beta = float(cfg.get("frontres_debug_rho_prior_beta", 0.5))
    rho_advantage = rho_evidence_advantage + beta * rho_prior_advantage
    rho_loss_mask = _rho_loss_mask_from_action_cone(
        cfg=cfg,
        n_exec=n,
        device=runner.device,
        dtype=rho_advantage.dtype,
    )

    runner.alg.transition.acceptance_target = torch.zeros_like(runner.alg.transition.acceptance_target)
    runner.alg.transition.acceptance_mask = torch.zeros_like(runner.alg.transition.acceptance_mask)
    runner.alg.transition.acceptance_target[:n, :6] = rho_advantage.detach()
    runner.alg.transition.acceptance_mask[:n, :6] = rho_loss_mask.detach()
    return {
        "rho_current": rho_current,
        "rho_drive": rho_drive,
        "evidence_direction": evidence_direction,
        "rho_evidence_advantage": rho_evidence_advantage,
        "prior_direction": prior_direction,
        "rho_prior_advantage": rho_prior_advantage,
        "rho_advantage": rho_advantage,
        "rho_loss_mask": rho_loss_mask,
    }


def _print_rho_debug(
    samples: list[DebugSample],
    runner: Any,
    rho_payload: Any,
    rho_debug: dict[str, torch.Tensor],
) -> None:
    _print_debug_boundary("D. rho advantage learning")
    rho_advantage = runner.alg.transition.acceptance_target
    rho_loss_mask = runner.alg.transition.acceptance_mask
    print("debug override: acceptance_target=rho_advantage, acceptance_mask=rho_loss_mask")
    print(
        "formula: rho_advantage = evidence_advantage + "
        f"{runner.cfg['frontres_debug_rho_prior_beta']:.3f} * prior_advantage"
    )
    print("rho columns: dx dy dz roll pitch yaw")
    print(
        "name       rho  drive  ev_dir prior_dir ev_adv prior_adv final_adv loss_mask"
    )
    print("-" * 104)
    for i, sample in enumerate(samples):
        print(
            f"{sample.name:<10}"
            f"{rho_debug['rho_current'][i].mean().item():>4.2f} "
            f"{rho_debug['rho_drive'][i].mean().item():>6.2f} "
            f"{rho_debug['evidence_direction'][i].item():>7.3f} "
            f"{rho_debug['prior_direction'][i].item():>9.3f} "
            f"{rho_debug['rho_evidence_advantage'][i].mean().item():>7.3f} "
            f"{rho_debug['rho_prior_advantage'][i].mean().item():>9.3f} "
            f"{rho_advantage[i].mean().item():>9.3f} "
            f"{rho_loss_mask[i].mean().item():>9.1f}"
        )
    print(
        "rho diag means: "
        f"planar={float(rho_payload.rho_target_planar_mean):.3f}, "
        f"rp={float(rho_payload.rho_target_rp_mean):.3f}, "
        f"z={float(rho_payload.rho_target_z_mean):.3f}, "
        f"group_weight={float(rho_payload.grouped_rho_mask_mean):.3f} "
        "(formal payload before debug override)"
    )


def _print_reward_debug(
    samples: list[DebugSample],
    truth: FrontRESRewardContext,
    frontres_reward: FrontRESPostStepResult,
) -> None:
    window = frontres_reward.reward_window
    if window is None:
        return
    _print_debug_boundary("E. final reward composition")
    print("name       r_exec  harm_penalty  intervention_cost  r_delta")
    print("-" * 64)
    for i, sample in enumerate(samples):
        print(
            f"{sample.name:<10}"
            f"{window.r_exec[i].item():>6.3f}  "
            f"{window.harm_penalty[i].item():>12.3f}  "
            f"{truth.intervention_cost[i].item():>17.3f}  "
            f"{frontres_reward.rewards[i].item():>7.3f}"
        )


def run_debug_reward_compute() -> None:
    device = torch.device("cpu")
    samples = _debug_samples()
    cfg = _debug_cfg()
    runner = FakeRunner(cfg=cfg, num_envs=len(samples) * 4, device=device)
    frontres_truth, actions, dones, infos = build_debug_frontres_truth(runner, samples)

    _print_live_config(cfg)
    _print_sample_inputs(samples)
    _print_reward_window_debug(samples, frontres_truth)

    write_rho_update_weight(
        runner,
        n_exec=frontres_truth.n_exec,
        rho_update_weight=frontres_truth.reward_window.rho_update_weight,
    )
    alpha_groundtruth, alpha_groundtruth_mask = write_alpha_groundtruth(
        runner,
        n_exec=frontres_truth.n_exec,
        exec_perturbed=frontres_truth.exec_perturbed,
        dones=dones,
        infos=infos,
        base_start=frontres_truth.base_start,
    )
    _print_alpha_debug(samples, runner, frontres_truth, alpha_groundtruth, alpha_groundtruth_mask)

    rho_payload = write_rho_advantage(
        runner,
        actions=actions,
        reward_context=frontres_truth,
        state_alpha_target=alpha_groundtruth,
        state_alpha_mask=alpha_groundtruth_mask,
        quat_to_rotvec_wxyz=_quat_passthrough,
        quat_mul_fn=_quat_mul_identity,
        quat_inv_fn=_quat_inv_identity,
    )
    rho_debug = apply_debug_prior_plus_evidence_rho_advantage(runner, frontres_truth, actions)
    _print_rho_debug(samples, runner, rho_payload, rho_debug)

    diagnostic_sums = initialize_frontres_reward_diagnostic_sums()
    frontres_reward = compute_frontres_reward(
        runner,
        locs={
            "N_train": len(samples),
            "N_candidate": len(samples),
            "N_base": len(samples),
            "N_clean": len(samples),
            "_is_task_space_mode": True,
            "_lambda_reg": 0.0,
            "_dr_done": False,
        },
        reward_context=frontres_truth,
        accept_payload=rho_payload,
        rewards=torch.zeros(len(samples) * 4, device=device),
        dones=dones,
        actions=actions,
        diagnostic_sums=diagnostic_sums,
        prev_delta_q=None,
        term_count=0,
        step_count=0,
    )
    _print_reward_debug(samples, frontres_truth, frontres_reward)

    means = materialize_frontres_reward_diagnostic_means(
        diagnostic_sums,
        is_frontres=True,
        is_task_space_mode=True,
        term_count=frontres_reward.term_count,
        step_count=frontres_reward.step_count,
    )

    header = (
        "name       noisy  frontres  candidate  clean  gap   gain  "
        "rho_update alpha mask  rho_adv  rho_mask reward expected"
    )
    print(header)
    print("-" * len(header))
    for i, sample in enumerate(samples):
        rho_loss_mask = runner.alg.transition.acceptance_mask[i].mean().item()
        rho_advantage = runner.alg.transition.acceptance_target[i].mean().item()
        print(
            f"{sample.name:<10}"
            f"{sample.exec_noisy:>5.3f}  "
            f"{sample.exec_frontres:>8.3f}  "
            f"{sample.exec_candidate:>9.3f}  "
            f"{sample.exec_clean:>5.3f}  "
            f"{frontres_truth.reward_window.damage_gap[i].item():>4.3f}  "
            f"{frontres_truth.repair_gain[i].item():>5.3f}  "
            f"{runner.alg.transition.frontres_actor_gate[i, 0].item():>6.3f} "
            f"{alpha_groundtruth[i, 0].item():>5.3f} "
            f"{alpha_groundtruth_mask[i, 0].item():>4.1f}  "
            f"{rho_advantage:>7.3f} "
            f"{rho_loss_mask:>8.3f} "
            f"{frontres_reward.rewards[i].item():>6.3f} "
            f"{sample.expected}"
        )

    print("\nDiagnostics means:")
    for key in (
        "frontres_damage_gap_mean",
        "frontres_actor_gate_mean",  # legacy diagnostic name for rho_update_weight
        "frontres_repair_gain_mean",
        "frontres_train_reward_mean",
        "frontres_accept_pref_mask_mean",
        "frontres_state_alpha_target_mean",
        "frontres_rho_target_planar_mean",
    ):
        print(f"  {key}: {means.get(key)}")


if __name__ == "__main__":
    run_debug_reward_compute()
