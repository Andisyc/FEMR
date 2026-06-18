# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES reward diagnostic accumulation.

This module keeps rollout diagnostics out of the runner's main training loop.
The values are logging-only and do not define storage, loss, or policy behavior.
"""

from __future__ import annotations

from typing import Any


def _item(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _mean_item(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "mean"):
        return float(value.mean().item())
    return float(value)


def initialize_frontres_reward_diagnostic_sums() -> dict[str, float | int]:
    """Create per-rollout FrontRES diagnostic accumulators."""

    keys = (
        "rdelta", "baseline", "r_z", "r_xy", "r_rp", "r_yaw", "r_rescue",
        "r_exec", "r_geom", "intervention_cost", "clean_bound_cost",
        "clean_bound_side_cost", "over_cost", "under_repair_cost",
        "reward_frontres", "reward_clean", "reward_candidate", "reward_oracle",
        "exec_planar", "exec_vertical", "exec_task", "damage_gap",
        "oracle_clean_gap", "oracle_trust", "repair_gain", "candidate_gain",
        "projection_gain", "underwrite", "accept_pref_mask", "accept_pref_full",
        "accept_pref_noop", "accept_pref_keep", "accept_pref_ignore",
        "accept_pref_margin", "accept_pref_need", "accept_pref_admiss",
        "accept_pref_target", "inertial_pref_penalty_rho",
        "inertial_pref_penalty_one", "positive_gain_frac", "repair_ratio",
        "exec_signal", "weighted_exec_signal", "train_reward",
        "effective_gain_bonus", "safe_cost", "repair_cost", "broken_cost",
        "behavior_fit", "repair_fit_rate", "repair_fit_gain",
        "restore_ratio_rp", "residual_rp_abs", "corr_roll_bias",
        "corr_pitch_bias", "harm_rate", "harm_mag", "safe_harm_rate",
        "broken_harm_rate", "safe_abstain_cost", "broken_abstain_cost",
        "window_mu", "safe_frac", "repair_frac", "broken_frac",
        "candidate_floor_margin", "candidate_floor_pass", "exec_floor_value",
        "exec_floor_safe", "exec_floor_adaptive", "exec_floor_safe_count",
        "exec_floor_broken_count", "stable_route_frac", "stable_endpoint_frac",
        "tri_weight_repair", "tri_weight_noisy", "tri_weight_stable",
        "rho_target_planar", "rho_target_rp", "rho_target_z",
        "rho_target_spread", "grouped_rho_mask", "rho_regret_up_planar",
        "rho_regret_up_rp", "rho_regret_up_z", "rho_regret_down_planar",
        "rho_regret_down_rp", "rho_regret_down_z", "state_alpha_pred",
        "state_alpha_target", "state_alpha_mask", "state_alpha_route",
        "structured_joint_adv", "structured_joint_weight",
        "structured_joint_rho_adv", "structured_joint_rho_weight",
        "structured_joint_rho_retention", "structured_joint_floor_violation",
        "structured_joint_full_bonus", "structured_joint_rho_direction",
        "structured_joint_rho_centered", "structured_joint_rho_drive",
        "oracle_ub_gain", "oracle_ub_pass", "oracle_ub_projected_win",
        "oracle_ub_candidate_win", "oracle_ub_feasible_win",
        "oracle_ub_noisy_win", "actor_gate", "exec_gate", "cost_gate",
        "dr_z_abs", "dr_xy_abs", "dr_rp_abs", "dr_yaw_abs", "corr_z_abs",
        "corr_xy_abs", "corr_rp_abs", "corr_yaw_abs", "smooth_penalty",
        "reg_penalty", "delta_pos_abs", "delta_rpy_abs", "delta_z_abs",
        "jump_degree", "reward_progress", "constraint_progress",
    )
    sums: dict[str, float | int] = {key: 0.0 for key in keys}
    sums["shaping_steps"] = 0
    sums["reward_diag_steps"] = 0
    return sums


_FRONTRES_SHAPING_MEAN_KEYS = (
    "rdelta", "baseline", "r_z", "r_xy", "r_rp", "r_yaw", "r_rescue",
    "dr_z_abs", "dr_xy_abs", "dr_rp_abs", "dr_yaw_abs", "corr_z_abs",
    "corr_xy_abs", "corr_rp_abs", "corr_yaw_abs", "smooth_penalty",
    "reg_penalty", "jump_degree",
)

_FRONTRES_REWARD_DIAG_MEAN_KEYS = (
    "r_exec", "r_geom", "intervention_cost", "clean_bound_cost",
    "clean_bound_side_cost", "over_cost", "under_repair_cost",
    "reward_frontres", "reward_clean", "reward_candidate", "reward_oracle",
    "exec_planar", "exec_vertical", "exec_task", "damage_gap",
    "oracle_clean_gap", "oracle_trust", "repair_gain", "candidate_gain",
    "projection_gain", "underwrite", "oracle_ub_gain", "oracle_ub_pass",
    "oracle_ub_projected_win", "oracle_ub_candidate_win",
    "oracle_ub_feasible_win", "oracle_ub_noisy_win", "accept_pref_mask",
    "accept_pref_full", "accept_pref_noop", "accept_pref_keep",
    "accept_pref_ignore", "accept_pref_margin", "accept_pref_need",
    "accept_pref_admiss", "accept_pref_target", "inertial_pref_penalty_rho",
    "inertial_pref_penalty_one", "positive_gain_frac", "repair_ratio",
    "exec_signal", "weighted_exec_signal", "train_reward",
    "effective_gain_bonus", "safe_cost", "repair_cost", "broken_cost",
    "reward_progress", "constraint_progress", "behavior_fit",
    "repair_fit_rate", "repair_fit_gain", "restore_ratio_rp",
    "residual_rp_abs", "corr_roll_bias", "corr_pitch_bias", "harm_rate",
    "harm_mag", "safe_harm_rate", "broken_harm_rate", "safe_abstain_cost",
    "broken_abstain_cost", "window_mu", "safe_frac", "repair_frac",
    "broken_frac", "candidate_floor_margin", "candidate_floor_pass",
    "exec_floor_value", "exec_floor_safe", "exec_floor_adaptive",
    "exec_floor_safe_count", "exec_floor_broken_count", "stable_route_frac",
    "stable_endpoint_frac", "tri_weight_repair", "tri_weight_noisy",
    "tri_weight_stable", "rho_target_planar", "rho_target_rp",
    "rho_target_z", "rho_target_spread", "grouped_rho_mask",
    "rho_regret_up_planar", "rho_regret_up_rp", "rho_regret_up_z",
    "rho_regret_down_planar", "rho_regret_down_rp", "rho_regret_down_z",
    "state_alpha_pred", "state_alpha_target", "state_alpha_mask",
    "state_alpha_route", "structured_joint_adv", "structured_joint_weight",
    "structured_joint_rho_adv", "structured_joint_rho_weight",
    "structured_joint_rho_retention", "structured_joint_floor_violation",
    "structured_joint_full_bonus", "structured_joint_rho_direction",
    "structured_joint_rho_centered", "structured_joint_rho_drive",
    "actor_gate", "exec_gate", "cost_gate",
)


def _metric_mean(is_frontres: bool, steps: int, value: float | int) -> float | None:
    if not is_frontres or steps <= 0:
        return None
    return float(value) / float(steps)


def materialize_frontres_reward_diagnostic_means(
    sums: dict[str, float | int],
    *,
    is_frontres: bool,
    is_task_space_mode: bool,
    term_count: int,
    step_count: int,
) -> dict[str, float | None]:
    """Build final per-rollout FrontRES diagnostic mean values for logging."""

    shaping_steps = int(sums.get("shaping_steps", 0))
    reward_diag_steps = int(sums.get("reward_diag_steps", 0))
    means: dict[str, float | None] = {}

    for key in _FRONTRES_SHAPING_MEAN_KEYS:
        means[f"frontres_{key}_mean"] = _metric_mean(
            is_frontres, shaping_steps, sums.get(key, 0.0)
        )
    for key in _FRONTRES_REWARD_DIAG_MEAN_KEYS:
        means[f"frontres_{key}_mean"] = _metric_mean(
            is_frontres, reward_diag_steps, sums.get(key, 0.0)
        )

    means["frontres_survival_rate"] = (
        1.0 - float(term_count) / float(step_count)
        if is_frontres and step_count > 0
        else None
    )
    means["frontres_delta_pos_abs_mean"] = (
        _metric_mean(is_frontres, shaping_steps, sums.get("delta_pos_abs", 0.0))
        if is_task_space_mode
        else None
    )
    means["frontres_delta_rpy_abs_mean"] = (
        _metric_mean(is_frontres, shaping_steps, sums.get("delta_rpy_abs", 0.0))
        if is_task_space_mode
        else None
    )
    means["frontres_delta_z_abs_mean"] = (
        _metric_mean(is_frontres, shaping_steps, sums.get("delta_z_abs", 0.0))
        if not is_task_space_mode
        else None
    )
    return means


def expose_frontres_reward_diagnostic_sums(
    sums: dict[str, float | int],
    locs: dict[str, Any],
) -> None:
    """Mirror accumulator dict values into legacy runner local names."""

    for key, value in sums.items():
        locs[f"_frontres_{key}_sum"] = value
    locs["_frontres_shaping_steps"] = int(sums.get("shaping_steps", 0))
    locs["_frontres_reward_diag_steps"] = int(sums.get("reward_diag_steps", 0))


def accumulate_frontres_reward_diagnostics(
    runner: Any,
    sums: dict[str, float | int],
    locs: dict[str, Any],
) -> None:
    """Accumulate one rollout step of FrontRES reward diagnostics."""

    sums["rdelta"] += _mean_item(locs.get("r_delta"))
    sums["baseline"] += _item(locs.get("_r_base_log"))

    if locs.get("_r_z") is not None:
        n_exec = int(locs["_n_exec"])
        n_train = int(locs["N_train"])
        r_exec = locs["_r_exec"]
        exec_weight = locs["_exec_weight"]
        cost_exec = locs["_cost_exec"]
        repair_gain = locs["_repair_gain"]
        damage_gap = locs["_damage_gap"]
        window_mu = locs["_window_mu"]
        harm_mag = locs["_harm_mag"]
        safe_gap = float(locs["_safe_gap"])
        broken_gap = float(locs["_broken_gap"])
        eps_fit = 1e-6

        sums["r_z"] += _mean_item(locs["_r_z"])
        sums["r_xy"] += _mean_item(locs["_r_xy"])
        sums["r_rp"] += _mean_item(locs["_r_rp"])
        sums["r_yaw"] += _mean_item(locs["_r_ya"])
        sums["r_rescue"] += float(locs["_r_rescue_log"])
        sums["r_exec"] += _mean_item(r_exec)
        sums["r_geom"] += _mean_item(locs["_r_step"])
        sums["intervention_cost"] += _mean_item(locs["_intervention_cost"])
        sums["clean_bound_cost"] += _mean_item(locs["_clean_bound_cost"])
        sums["clean_bound_side_cost"] += _mean_item(locs["_side_cost"])
        sums["over_cost"] += _mean_item(locs["_over_cost"])
        sums["under_repair_cost"] += _mean_item(locs["_under_repair_penalty"])
        sums["reward_frontres"] += _item(locs["_r_frontres_log"])
        sums["reward_clean"] += _item(locs["_r_clean_log"])
        sums["reward_candidate"] += _mean_item(locs["_exec_candidate"])
        sums["reward_oracle"] += _item(locs["_r_oracle_log"])
        sums["exec_planar"] += _item(locs["_exec_planar_log"])
        sums["exec_vertical"] += _item(locs["_exec_vertical_log"])
        sums["exec_task"] += _item(locs["_exec_task_log"])
        sums["damage_gap"] += _mean_item(damage_gap)
        sums["oracle_clean_gap"] += _mean_item(locs["_oracle_clean_gap"])
        sums["oracle_trust"] += _mean_item(locs["_oracle_trust"])
        sums["repair_gain"] += _mean_item(repair_gain)
        sums["candidate_gain"] += _mean_item(locs["_candidate_gain"])
        sums["projection_gain"] += _mean_item(locs["_projection_gain"])
        sums["underwrite"] += _mean_item(locs["_under_write"])
        sums["accept_pref_mask"] += _mean_item((locs["_accept_pref_mask"][:n_exec].sum(dim=-1) > 0).float())
        sums["accept_pref_full"] += _item(locs["_pref_full_frac"])
        sums["accept_pref_noop"] += _item(locs["_pref_noop_frac"])
        sums["accept_pref_keep"] += _item(locs["_pref_keep_frac"])
        sums["accept_pref_ignore"] += _item(locs["_pref_ignore_frac"])
        sums["accept_pref_margin"] += _item(locs["_pref_margin_mean"])
        sums["accept_pref_need"] += _item(locs["_pref_need_mean"])
        sums["accept_pref_admiss"] += _item(locs["_pref_admiss_mean"])
        sums["accept_pref_target"] += _item(locs["_pref_target_mean"])
        sums["inertial_pref_penalty_rho"] += _item(locs["_pref_inertial_penalty_rho_mean"])
        sums["inertial_pref_penalty_one"] += _item(locs["_pref_inertial_penalty_one_mean"])
        sums["positive_gain_frac"] += _mean_item((repair_gain > 0.0).float())
        sums["repair_ratio"] += _mean_item(locs["_repair_ratio"])
        sums["exec_signal"] += _mean_item(r_exec[:n_exec])
        sums["weighted_exec_signal"] += _mean_item(exec_weight[:n_exec] * r_exec[:n_exec])
        sums["train_reward"] += _mean_item(locs["r_delta"][:n_exec])
        sums["reward_progress"] += float(locs["_reward_progress"])
        sums["constraint_progress"] += float(locs["_constraint_progress"])
        sums["effective_gain_bonus"] += _mean_item(locs["_effective_gain_bonus"][:n_exec])
        sums["safe_cost"] += _mean_item(locs["_safe_gate"] * cost_exec)
        sums["repair_cost"] += _mean_item(locs["_repair_gate"] * cost_exec)
        sums["broken_cost"] += _mean_item(locs["_broken_gate"] * cost_exec)

        mu_sum = window_mu.sum().clamp(min=eps_fit)
        repair_fit_num = (window_mu * repair_gain).sum()
        repair_fit_gap = (window_mu * damage_gap).sum().clamp(min=eps_fit)
        repair_fit_rate = repair_fit_num / repair_fit_gap
        repair_fit_gain = repair_fit_num / mu_sum
        harm_indicator = (harm_mag > 0.0).float()
        harm_rate = (window_mu * harm_indicator).sum() / mu_sum
        harm_mag_fit = (window_mu * harm_mag).sum() / mu_sum
        safe_mask = (damage_gap < safe_gap).float()
        broken_mask = (damage_gap > broken_gap).float()
        safe_den = safe_mask.sum().clamp(min=eps_fit)
        broken_den = broken_mask.sum().clamp(min=eps_fit)
        safe_harm_rate = (safe_mask * harm_indicator).sum() / safe_den
        broken_harm_rate = (broken_mask * harm_indicator).sum() / broken_den
        safe_abstain_cost = (safe_mask * cost_exec).sum() / safe_den
        broken_abstain_cost = (broken_mask * cost_exec).sum() / broken_den
        behavior_fit_num = (
            (window_mu * repair_gain).sum()
            + locs["_effective_gain_bonus_exec"].sum()
            - locs["_harm_penalty_exec"].sum()
            - (locs["_cost_gate"] * cost_exec).sum()
        )
        behavior_fit_den = (
            (window_mu * damage_gap).sum()
            + (locs["_cost_gate"] * cost_exec).sum()
        ).clamp(min=eps_fit)
        behavior_fit = behavior_fit_num / behavior_fit_den
        sums["behavior_fit"] += _item(behavior_fit)
        sums["repair_fit_rate"] += _item(repair_fit_rate)
        sums["repair_fit_gain"] += _item(repair_fit_gain)

        restore_eval_min = float(runner.cfg.get("frontres_restore_eval_min_error", 1e-3))
        restore_eval_mask = locs["_e_raw"] > max(restore_eval_min, 1e-8)
        if restore_eval_mask.any():
            restore_ratio_rp = 1.0 - (
                locs["_e_fr"][restore_eval_mask] / locs["_e_raw"][restore_eval_mask].clamp(min=1e-6)
            )
            sums["restore_ratio_rp"] += _mean_item(restore_ratio_rp)
        else:
            sums["restore_ratio_rp"] += 0.0
        sums["residual_rp_abs"] += _mean_item(locs["_e_fr"])
        sums["corr_roll_bias"] += _mean_item(locs["_rot_raw_to_fr"][:, 0])
        sums["corr_pitch_bias"] += _mean_item(locs["_rot_raw_to_fr"][:, 1])
        sums["harm_rate"] += _item(harm_rate)
        sums["harm_mag"] += _item(harm_mag_fit)
        sums["safe_harm_rate"] += _item(safe_harm_rate)
        sums["broken_harm_rate"] += _item(broken_harm_rate)
        sums["safe_abstain_cost"] += _item(safe_abstain_cost)
        sums["broken_abstain_cost"] += _item(broken_abstain_cost)
        sums["window_mu"] += _mean_item(window_mu)
        sums["safe_frac"] += _item(locs["_safe_frac"])
        sums["repair_frac"] += _item(locs["_repair_frac"])
        sums["broken_frac"] += _item(locs["_broken_frac"])
        sums["candidate_floor_margin"] += _mean_item(locs["_candidate_floor_margin"])
        sums["candidate_floor_pass"] += _item(locs["_candidate_floor_pass_frac"])

        for key, attr in (
            ("exec_floor_value", "_frontres_exec_floor_value_last"),
            ("exec_floor_safe", "_frontres_exec_floor_safe_last"),
            ("exec_floor_adaptive", "_frontres_exec_floor_adaptive_last"),
            ("exec_floor_safe_count", "_frontres_exec_floor_safe_count_last"),
            ("exec_floor_broken_count", "_frontres_exec_floor_broken_count_last"),
            ("stable_route_frac", "_frontres_stable_route_applied_frac"),
            ("stable_endpoint_frac", "_frontres_stable_endpoint_frac"),
            ("tri_weight_repair", "_frontres_tri_weight_repair"),
            ("tri_weight_noisy", "_frontres_tri_weight_noisy"),
            ("tri_weight_stable", "_frontres_tri_weight_stable"),
            ("rho_target_planar", "_frontres_rho_target_planar_last"),
            ("rho_target_rp", "_frontres_rho_target_rp_last"),
            ("rho_target_z", "_frontres_rho_target_z_last"),
            ("rho_target_spread", "_frontres_rho_target_spread_last"),
            ("grouped_rho_mask", "_frontres_grouped_rho_mask_last"),
            ("rho_regret_up_planar", "_frontres_rho_regret_up_planar_last"),
            ("rho_regret_up_rp", "_frontres_rho_regret_up_rp_last"),
            ("rho_regret_up_z", "_frontres_rho_regret_up_z_last"),
            ("rho_regret_down_planar", "_frontres_rho_regret_down_planar_last"),
            ("rho_regret_down_rp", "_frontres_rho_regret_down_rp_last"),
            ("rho_regret_down_z", "_frontres_rho_regret_down_z_last"),
            ("state_alpha_pred", "_frontres_state_alpha_pred_last"),
            ("state_alpha_target", "_frontres_state_alpha_target_last"),
            ("state_alpha_mask", "_frontres_state_alpha_mask_last"),
            ("state_alpha_route", "_frontres_state_alpha_route_last"),
            ("structured_joint_adv", "_frontres_structured_joint_adv_last"),
            ("structured_joint_weight", "_frontres_structured_joint_weight_last"),
            ("structured_joint_rho_adv", "_frontres_structured_joint_rho_adv_last"),
            ("structured_joint_rho_weight", "_frontres_structured_joint_rho_weight_last"),
            ("structured_joint_rho_retention", "_frontres_structured_joint_rho_retention_last"),
            ("structured_joint_floor_violation", "_frontres_structured_joint_floor_violation_last"),
            ("structured_joint_full_bonus", "_frontres_structured_joint_full_bonus_last"),
            ("structured_joint_rho_direction", "_frontres_structured_joint_rho_direction_last"),
            ("structured_joint_rho_centered", "_frontres_structured_joint_rho_centered_last"),
            ("structured_joint_rho_drive", "_frontres_structured_joint_rho_drive_last"),
        ):
            default = 1.0 if key == "tri_weight_noisy" else 0.0
            sums[key] += float(getattr(runner, attr, default))

        sums["oracle_ub_gain"] += _mean_item(locs["_oracle_ub_gain"])
        sums["oracle_ub_pass"] += _mean_item(locs["_oracle_ub_pass"])
        sums["oracle_ub_projected_win"] += _mean_item(locs["_oracle_ub_projected_win"])
        sums["oracle_ub_candidate_win"] += _mean_item(locs["_oracle_ub_candidate_win"])
        sums["oracle_ub_feasible_win"] += _mean_item(locs["_oracle_ub_feasible_win"])
        sums["oracle_ub_noisy_win"] += _mean_item(locs["_oracle_ub_noisy_win"])
        sums["actor_gate"] += _mean_item(locs["_actor_gate"])
        sums["exec_gate"] += _mean_item(locs["_exec_gate"])
        sums["cost_gate"] += _mean_item(locs["_cost_gate"])
        sums["reward_diag_steps"] += 1
        sums["dr_z_abs"] += _item(locs["_dr_z_abs_log"])
        sums["dr_xy_abs"] += _item(locs["_dr_xy_abs_log"])
        sums["dr_rp_abs"] += _item(locs["_dr_rp_abs_log"])
        sums["dr_yaw_abs"] += _item(locs["_dr_yaw_abs_log"])
        sums["corr_z_abs"] += _item(locs["_corr_z_abs_log"])
        sums["corr_xy_abs"] += _item(locs["_corr_xy_abs_log"])
        sums["corr_rp_abs"] += _item(locs["_corr_rp_abs_log"])
        sums["corr_yaw_abs"] += _item(locs["_corr_yaw_abs_log"])

    sums["smooth_penalty"] += _mean_item(locs.get("_smooth_penalty"))
    sums["reg_penalty"] += _mean_item(locs.get("_reg_penalty"))

    if locs.get("_is_task_space_mode"):
        task_corr = getattr(runner.alg.policy, "last_task_correction", None)
        if task_corr is not None:
            n_train = int(locs["N_train"])
            sums["delta_pos_abs"] += _mean_item(task_corr[:n_train, :3].abs())
            sums["delta_rpy_abs"] += _mean_item(task_corr[:n_train, 3:6].abs())
        env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
        for command_term in env.command_manager._terms.values():
            if hasattr(command_term, "jump_degree"):
                sums["jump_degree"] += _mean_item(command_term.jump_degree[: int(locs["N_train"])])
                break
    else:
        delta_z = getattr(runner.alg.policy, "last_delta_z", None)
        if delta_z is not None:
            sums["delta_z_abs"] += _mean_item(delta_z.abs())

    sums["shaping_steps"] += 1
