# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone FrontRES Reward Compute debug harness.

This file intentionally copies the runner-side Reward Compute sequence without
being imported by ``on_policy_runner.py``.  It builds simple hand-checkable
FrontRES samples, then runs:

    sample weight -> alpha groundtruth -> rho groundtruth -> reward

Run from the repository root with:

    PYTHONPATH=source/rsl_rl python -m rsl_rl.runners.frontres_reward_compute

or:

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
    initialize_frontres_reward_diagnostic_sums,
    materialize_frontres_reward_diagnostic_means,
)
from rsl_rl.frontres.frontres_reward_window import (
    FrontRESRewardContext,
    build_frontres_reward_window,
)
from rsl_rl.frontres.frontres_rollout_evidence import compute_frontres_rollout_evidence
from rsl_rl.frontres.frontres_transition_payload import (
    write_actor_sample_weight,
    write_alpha_groundtruth,
    write_rho_groundtruth,
)
from rsl_rl.runners.frontres_post_step_connector import compute_frontres_reward


@dataclass(frozen=True)
class DebugSample:
    name: str
    exec_noisy: float
    exec_frontres: float
    exec_candidate: float
    exec_clean: float
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
        return False


def _debug_cfg() -> dict[str, Any]:
    return {
        "frontres_gap_floor_per_step": 0.01,
        "frontres_safe_gap_per_step": 0.03,
        "frontres_broken_gap_per_step": 0.50,
        "frontres_gap_gate_temp": 0.04,
        "frontres_oracle_clean_gap_threshold": 1.0e9,
        "frontres_reward_scale_dr_reference": 1.0,
        "frontres_reward_progress_min": 1.0,
        "frontres_constraint_progress_exponent": 1.0,
        "frontres_selective_reward_enabled": True,
        "frontres_exec_reward_signal": "gain",
        "frontres_exec_reward_weight": 1.0,
        "frontres_repair_reward_scale": 1.0,
        "frontres_geometry_reward_weight": 0.0,
        "frontres_rescue_reward_weight": 0.0,
        "frontres_executable_harm_weight": 1.0,
        "frontres_harm_epsilon": 0.001,
        "frontres_harm_penalty_weight": 0.25,
        "frontres_side_harm_weight": 0.0,
        "frontres_harm_action_cost_floor": 0.0,
        "frontres_harm_action_cost_ref": 0.01,
        "frontres_side_actor_gate_weight": 0.05,
        "frontres_min_effective_gain": 0.006,
        "frontres_effective_gain_bonus_weight": 0.0,
        "frontres_candidate_ranking_reward_enabled": False,
        "frontres_rho_space": "tri_anchor",
        "frontres_acceptance_preference_enabled": True,
        "frontres_acceptance_preference_margin": 0.003,
        "frontres_acceptance_regret_target_enabled": True,
        "frontres_acceptance_regret_soft_mask_floor": 0.0,
        "frontres_acceptance_regret_oracle_trust_floor": 0.0,
        "frontres_acceptance_regret_per_mode_soft_floor": 0.0,
        "frontres_acceptance_rho_target_temp": 0.08,
        "frontres_acceptance_calibration_step": 0.5,
        "frontres_grouped_rho_target_enabled": True,
        "frontres_per_mode_acceptance_preference_mask": True,
        "frontres_active_task_dims": [0, 1, 2, 3, 4, 5],
        "frontres_state_alpha_enabled": True,
        "frontres_state_alpha_exec_floor": 0.50,
        "frontres_state_alpha_safe_exec_floor": 0.60,
        "frontres_state_alpha_temp": 0.08,
        "frontres_oracle_upper_bound_diag_enabled": True,
        "frontres_oracle_upper_bound_margin": 0.0,
    }


def _debug_samples() -> list[DebugSample]:
    return [
        DebugSample("safe", 0.950, 0.949, 0.952, 0.952, "low sample weight, no strong repair"),
        DebugSample("repairable", 0.550, 0.750, 0.850, 0.900, "high sample weight, positive reward"),
        DebugSample("harmful", 0.700, 0.450, 0.800, 0.880, "negative reward, reject current repair"),
        DebugSample("broken", 0.050, 0.080, 0.120, 0.900, "low sample weight, weak actor signal"),
        DebugSample("keep", 0.600, 0.780, 0.760, 0.900, "keep current repair"),
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
    actions[:n_train, 6:12] = 0.5

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


def run_debug_reward_compute() -> None:
    device = torch.device("cpu")
    samples = _debug_samples()
    cfg = _debug_cfg()
    runner = FakeRunner(cfg=cfg, num_envs=len(samples) * 4, device=device)
    frontres_truth, actions, dones, infos = build_debug_frontres_truth(runner, samples)

    write_actor_sample_weight(
        runner,
        n_exec=frontres_truth.n_exec,
        actor_gate=frontres_truth.reward_window.actor_gate,
    )
    alpha_groundtruth, alpha_groundtruth_mask = write_alpha_groundtruth(
        runner,
        n_exec=frontres_truth.n_exec,
        exec_perturbed=frontres_truth.exec_perturbed,
        dones=dones,
        infos=infos,
        base_start=frontres_truth.base_start,
    )
    rho_groundtruth = write_rho_groundtruth(
        runner,
        actions=actions,
        reward_context=frontres_truth,
        state_alpha_target=alpha_groundtruth,
        state_alpha_mask=alpha_groundtruth_mask,
        quat_to_rotvec_wxyz=_quat_passthrough,
        quat_mul_fn=_quat_mul_identity,
        quat_inv_fn=_quat_inv_identity,
    )

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
        accept_payload=rho_groundtruth,
        rewards=torch.zeros(len(samples) * 4, device=device),
        dones=dones,
        actions=actions,
        diagnostic_sums=diagnostic_sums,
        prev_delta_q=None,
        term_count=0,
        step_count=0,
    )
    means = materialize_frontres_reward_diagnostic_means(
        diagnostic_sums,
        is_frontres=True,
        is_task_space_mode=True,
        term_count=frontres_reward.term_count,
        step_count=frontres_reward.step_count,
    )

    header = (
        "name       noisy  frontres  candidate  clean  gap   gain  "
        "weight alpha mask  rho_mean rho_mask reward expected"
    )
    print(header)
    print("-" * len(header))
    for i, sample in enumerate(samples):
        rho_mask = runner.alg.transition.acceptance_mask[i].mean().item()
        rho_target = runner.alg.transition.acceptance_target[i].mean().item()
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
            f"{rho_target:>8.3f} "
            f"{rho_mask:>8.3f} "
            f"{frontres_reward.rewards[i].item():>6.3f} "
            f"{sample.expected}"
        )

    print("\nDiagnostics means:")
    for key in (
        "frontres_damage_gap_mean",
        "frontres_actor_gate_mean",
        "frontres_repair_gain_mean",
        "frontres_train_reward_mean",
        "frontres_accept_pref_mask_mean",
        "frontres_state_alpha_target_mean",
        "frontres_rho_target_planar_mean",
    ):
        print(f"  {key}: {means.get(key)}")


if __name__ == "__main__":
    run_debug_reward_compute()
