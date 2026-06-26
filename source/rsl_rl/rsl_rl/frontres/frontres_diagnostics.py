"""Console diagnostics for FrontRES runner logs.

This module is intentionally formatting-only.  It centralizes FrontRES metric
names so runner branches cannot drift into different interpretations of the same
concept.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MetricMap = Mapping[str, Any]


def _value(metrics: MetricMap, key: str, default: float = 0.0) -> Any:
    value = metrics.get(key, default)
    return default if value is None else value


def _first_value(metrics: MetricMap, keys: tuple[str, ...], default: float = 0.0) -> Any:
    for key in keys:
        value = metrics.get(key, None)
        if value is not None:
            return value
    return default


def _fmt_active_value(metrics: MetricMap, key: str, active_key: str) -> str:
    if float(_value(metrics, active_key)) <= 1.0e-6:
        return "n/a"
    return f"{_value(metrics, key):.3f}"


def _rho_space(cfg: MetricMap) -> str:
    return str(cfg.get("frontres_rho_space", "noisy_to_repair"))


def _is_tri_anchor(cfg: MetricMap) -> bool:
    return _rho_space(cfg).lower() in ("tri_anchor", "tri-anchor", "tri")


def _structured_joint_enabled(cfg: MetricMap) -> bool:
    return (
        bool(cfg.get("frontres_structured_joint_rl_enabled", False))
        and float(cfg.get("frontres_structured_joint_rl_weight", 0.0)) > 0.0
        and str(cfg.get("frontres_training_objective", "")).lower() == "hsl_hybrid"
    )


def _authority_actor_critic_enabled(cfg: MetricMap, loss_dict: MetricMap | None = None) -> bool:
    if bool(cfg.get("frontres_authority_actor_critic_enabled", False)):
        return True
    if loss_dict is not None and float(_value(loss_dict, "authority_actor_critic_enabled", 0.0)) > 0.5:
        return True
    return False


def _active_hsl_acceptance_enabled(cfg: MetricMap, loss_dict: MetricMap | None = None) -> bool:
    if str(cfg.get("frontres_training_objective", "")).lower() != "hsl_hybrid":
        return False
    if _structured_joint_enabled(cfg) or _authority_actor_critic_enabled(cfg, loss_dict):
        return False
    if loss_dict is None:
        return True
    return (
        float(_value(loss_dict, "hsl_acceptance_path_enabled", 0.0)) > 0.5
        or float(_value(loss_dict, "hsl_acceptance_loss_enabled", 0.0)) > 0.5
        or float(_value(loss_dict, "lambda_acceptance_preference", 0.0)) > 0.0
    )


def _show_legacy_rho_diag(cfg: MetricMap) -> bool:
    return (not _structured_joint_enabled(cfg)) or bool(
        cfg.get("frontres_structured_joint_show_legacy_rho_diag", False)
    )


def format_frontres_floor_alpha_diagnostics(
    locs: MetricMap,
    loss_dict: MetricMap,
    *,
    pad: int,
) -> str:
    """Format executable-floor and state-alpha diagnostics."""
    lines: list[str] = []

    if locs.get("frontres_candidate_floor_margin_mean") is not None:
        lines.append(
            f"{'cand floor pass/margin:':>{pad}} "
            f"{_value(locs, 'frontres_candidate_floor_pass_mean'):.3f} / "
            f"{_value(locs, 'frontres_candidate_floor_margin_mean'):+.4f}\n"
        )
        lines.append(
            f"{'exec floor val/safe/adapt:':>{pad}} "
            f"{_value(locs, 'frontres_exec_floor_value_mean'):+.4f} / "
            f"{_value(locs, 'frontres_exec_floor_safe_mean'):+.4f} / "
            f"{_value(locs, 'frontres_exec_floor_adaptive_mean'):.3f}\n"
        )
        lines.append(
            f"{'exec floor cnt s/b:':>{pad}} "
            f"{_value(locs, 'frontres_exec_floor_safe_count_mean'):.0f} / "
            f"{_value(locs, 'frontres_exec_floor_broken_count_mean'):.0f}\n"
        )

    show_state_alpha = (
        locs.get("frontres_state_alpha_pred_mean") is not None
        and (
            _value(loss_dict, "lambda_state_alpha", 0.0) > 0.0
            or abs(float(_value(locs, "frontres_state_alpha_mask_mean", 0.0))) > 1.0e-6
            or abs(float(_value(locs, "frontres_state_alpha_route_mean", 0.0))) > 1.0e-6
            or abs(float(_value(loss_dict, "state_alpha_loss", 0.0))) > 1.0e-8
        )
    )
    if show_state_alpha:
        lines.append(
            f"{'state alpha p/t/m/hard:':>{pad}} "
            f"{_value(locs, 'frontres_state_alpha_pred_mean'):.3f} / "
            f"{_value(locs, 'frontres_state_alpha_target_mean'):.3f} / "
            f"{_value(locs, 'frontres_state_alpha_mask_mean'):.3f} / "
            f"{_value(locs, 'frontres_state_alpha_route_mean'):.3f}\n"
        )
        if loss_dict.get("state_alpha_loss", None) is not None:
            lines.append(
                f"{'state alpha loss/acc:':>{pad}} "
                f"{_value(loss_dict, 'state_alpha_loss'):.4f} / "
                f"{_value(loss_dict, 'state_alpha_acc'):.3f}\n"
            )

    return "".join(lines)


def format_frontres_route_rho_diagnostics(
    locs: MetricMap,
    cfg: MetricMap,
    *,
    pad: int,
) -> str:
    """Format live route, structured-rho, and optional legacy-rho diagnostics."""
    lines: list[str] = []
    if _authority_actor_critic_enabled(cfg) or _active_hsl_acceptance_enabled(cfg):
        return ""

    structured_adv = _first_value(
        locs,
        ("frontres_structured_joint_rho_adv_mean", "frontres_structured_joint_adv_mean"),
        None,
    )
    structured_weight = _first_value(
        locs,
        ("frontres_structured_joint_rho_weight_mean", "frontres_structured_joint_weight_mean"),
        0.0,
    )
    if structured_adv is not None:
        lines.append(
            f"{'rho directional d/c/drv:':>{pad}} "
            f"{_value(locs, 'frontres_structured_joint_rho_direction_mean'):+.3f} / "
            f"{_value(locs, 'frontres_structured_joint_rho_centered_mean'):+.3f} / "
            f"{_value(locs, 'frontres_structured_joint_rho_drive_mean'):+.3f}\n"
        )
        lines.append(
            f"{'rho underwrite raw/uw/acc:':>{pad}} "
            f"{_value(locs, 'frontres_structured_joint_rho_raw_direction_mean'):+.3f} / "
            f"{_value(locs, 'frontres_structured_joint_rho_underwrite_mean'):.3f} / "
            f"{_value(locs, 'frontres_structured_joint_rho_accept_mean'):.3f} "
            f"(w={float(cfg.get('frontres_structured_joint_underwrite_weight', 0.0)):.2f})\n"
        )
        lines.append(
            f"{'rho constrained adv:':>{pad}} "
            f"{structured_adv:+.4f} / "
            f"retp={_value(locs, 'frontres_structured_joint_rho_retention_mean'):+.4f}, "
            f"floor={_value(locs, 'frontres_structured_joint_floor_violation_mean'):+.4f}, "
            f"full={_value(locs, 'frontres_structured_joint_full_bonus_mean'):+.4f}\n"
        )
        lines.append(f"{'rho weight active:':>{pad}} {structured_weight:.3f}\n")
        lines.append(f"{'rho update mode:':>{pad}} structured_adv rho-only\n")

    if locs.get("frontres_candidate_floor_margin_mean") is not None:
        lines.append(f"{'rho space:':>{pad}} {_rho_space(cfg)}\n")
        if _is_tri_anchor(cfg):
            lines.append(
                f"{'tri route w R/N/S:':>{pad}} "
                f"{_value(locs, 'frontres_tri_weight_repair_mean'):.3f} / "
                f"{_value(locs, 'frontres_tri_weight_noisy_mean'):.3f} / "
                f"{_value(locs, 'frontres_tri_weight_stable_mean'):.3f}\n"
            )
            show_hard_route = (
                bool(cfg.get("frontres_state_alpha_route_enabled", False))
                or abs(float(_value(locs, "frontres_stable_endpoint_frac_mean"))) > 1.0e-6
                or abs(float(_value(locs, "frontres_stable_route_frac_mean"))) > 1.0e-6
            )
            if show_hard_route:
                lines.append(
                    f"{'tri hard end/route:':>{pad}} "
                    f"{_value(locs, 'frontres_stable_endpoint_frac_mean'):.3f} / "
                    f"{_value(locs, 'frontres_stable_route_frac_mean'):.3f}\n"
                )
        else:
            show_stable_route = (
                bool(cfg.get("frontres_state_alpha_route_enabled", False))
                or abs(float(_value(locs, "frontres_stable_endpoint_frac_mean"))) > 1.0e-6
                or abs(float(_value(locs, "frontres_stable_route_frac_mean"))) > 1.0e-6
            )
            if not show_stable_route:
                return "".join(lines)
            lines.append(
                f"{'stable endpoint frac:':>{pad}} "
                f"{_value(locs, 'frontres_stable_endpoint_frac_mean'):.3f}\n"
            )
            lines.append(
                f"{'stable route frac:':>{pad}} "
                f"{_value(locs, 'frontres_stable_route_frac_mean'):.3f}\n"
            )

    if (
        _is_tri_anchor(cfg)
        and locs.get("frontres_rho_target_planar_mean") is not None
        and _show_legacy_rho_diag(cfg)
    ):
        target_label = "legacy rho diag p/r/z:" if _structured_joint_enabled(cfg) else "rho target grp p/r/z:"
        spread_label = (
            "legacy rho spread/mask:"
            if _structured_joint_enabled(cfg)
            else "rho target spread/weight:"
        )
        lines.append(
            f"{target_label:>{pad}} "
            f"{_value(locs, 'frontres_rho_target_planar_mean'):.3f} / "
            f"{_value(locs, 'frontres_rho_target_rp_mean'):.3f} / "
            f"{_value(locs, 'frontres_rho_target_z_mean'):.3f}\n"
        )
        lines.append(
            f"{spread_label:>{pad}} "
            f"{_value(locs, 'frontres_rho_target_spread_mean'):.3f} / "
            f"{_value(locs, 'frontres_grouped_rho_mask_mean'):.3f}\n"
        )
        lines.append(
            f"{'rho regret up/dn p/r/z:':>{pad}} "
            f"{_value(locs, 'frontres_rho_regret_up_planar_mean'):.4f}/"
            f"{_value(locs, 'frontres_rho_regret_down_planar_mean'):.4f} / "
            f"{_value(locs, 'frontres_rho_regret_up_rp_mean'):.4f}/"
            f"{_value(locs, 'frontres_rho_regret_down_rp_mean'):.4f} / "
            f"{_value(locs, 'frontres_rho_regret_up_z_mean'):.4f}/"
            f"{_value(locs, 'frontres_rho_regret_down_z_mean'):.4f}\n"
        )

    return "".join(lines)


def format_frontres_preference_diagnostics(
    locs: MetricMap,
    loss_dict: MetricMap,
    cfg: MetricMap,
    *,
    pad: int,
    structured_label: str = "joint adv pos/neg/near/ign:",
) -> str:
    """Format acceptance/preference diagnostics that are still active."""
    lines: list[str] = []
    structured_joint_live = _structured_joint_enabled(cfg) or bool(
        _value(loss_dict, "structured_joint_rl_enabled", 0.0) > 0.5
    ) or bool(_value(loss_dict, "lambda_structured_joint_rl", 0.0) > 0.0)
    authority_live = _authority_actor_critic_enabled(cfg, loss_dict)
    if _active_hsl_acceptance_enabled(cfg, loss_dict):
        return ""

    if locs.get("frontres_accept_pref_mask_mean") is not None and not structured_joint_live and not authority_live:
        rho_space = _rho_space(cfg).lower()
        if structured_joint_live:
            pref_label = structured_label
        elif rho_space in ("stable_to_repair", "stable-repair", "stable"):
            pref_label = "accept pref repair/stable/keep/ign:"
        elif rho_space in ("tri_anchor", "tri-anchor", "tri"):
            pref_label = "accept pref repair/fallback/keep/ign:"
        else:
            pref_label = "accept pref full/noop/keep/ign:"
        lines.append(
            f"{pref_label:>{pad}} "
            f"{_value(locs, 'frontres_accept_pref_full_mean'):.3f} / "
            f"{_value(locs, 'frontres_accept_pref_noop_mean'):.3f} / "
            f"{_value(locs, 'frontres_accept_pref_keep_mean'):.3f} / "
            f"{_value(locs, 'frontres_accept_pref_ignore_mean'):.3f} "
            f"(mask={_value(locs, 'frontres_accept_pref_mask_mean'):.3f}, "
            f"margin={_value(locs, 'frontres_accept_pref_margin_mean'):+.4f})\n"
        )
        lines.append(
            f"{'accept mask raw/mode/dim:':>{pad}} "
            f"{_value(locs, 'frontres_accept_pref_raw_mask_mean'):.3f} / "
            f"{_value(locs, 'frontres_accept_pref_mode_mask_mean'):.3f} / "
            f"{_value(locs, 'frontres_accept_pref_dim_mask_mean'):.3f}\n"
        )

    if (
        bool(cfg.get("frontres_acceptance_direct_target_enabled", False))
        and locs.get("frontres_accept_pref_need_mean") is not None
    ):
        accept_target = locs.get(
            "frontres_accept_pref_target_mean",
            _value(loss_dict, "acceptance_preference_target_mean"),
        )
        lines.append(
            f"{'accept need/admiss/tgt:':>{pad}} "
            f"{_value(locs, 'frontres_accept_pref_need_mean'):.3f} / "
            f"{_value(locs, 'frontres_accept_pref_admiss_mean'):.3f} / "
            f"{accept_target:.3f}\n"
        )

    if (
        bool(cfg.get("frontres_inertial_preference_enabled", False))
        and locs.get("frontres_inertial_pref_penalty_rho_mean") is not None
    ):
        lines.append(
            f"{'inert pref pen rho/cand:':>{pad}} "
            f"{_value(locs, 'frontres_inertial_pref_penalty_rho_mean'):.3f} / "
            f"{_value(locs, 'frontres_inertial_pref_penalty_one_mean'):.3f}\n"
        )

    return "".join(lines)


def format_frontres_hsl_acceptance_diagnostics(
    locs: MetricMap,
    loss_dict: MetricMap,
    cfg: MetricMap,
    *,
    pad: int,
) -> str:
    """Format active FEMR HSL proposal + acceptance diagnostics."""

    if not _active_hsl_acceptance_enabled(cfg, loss_dict):
        return ""

    lines: list[str] = []
    loss = loss_dict.get("acceptance_preference_loss", None)
    if loss is not None:
        status = "active" if _value(loss_dict, "hsl_acceptance_mask_frac") > 0.0 else "active_zero_mask"
        lines.append(
            f"{'acceptance loss:':>{pad}} {loss:.4f} "
            f"(lambda={_value(loss_dict, 'lambda_acceptance_preference'):.3f}, "
            f"status={status}, "
            f"mask={_value(loss_dict, 'hsl_acceptance_mask_frac'):.3f}, "
            f"gt={_value(loss_dict, 'hsl_acceptance_gt_mean'):.3f}, "
            f"prob={_value(loss_dict, 'hsl_acceptance_prob_mean'):.3f}, "
            f"err={_value(loss_dict, 'hsl_acceptance_abs_err'):.3f})\n"
        )

    if locs.get("frontres_accept_pref_mask_mean") is not None:
        lines.append(
            f"{'accept labels a/r/i:':>{pad}} "
            f"{_value(locs, 'frontres_accept_pref_full_mean'):.3f} / "
            f"{_value(locs, 'frontres_accept_pref_noop_mean'):.3f} / "
            f"{_value(locs, 'frontres_accept_pref_ignore_mean'):.3f} "
            f"(mask={_value(locs, 'frontres_accept_pref_mask_mean'):.3f}, "
            f"margin={_value(locs, 'frontres_accept_pref_margin_mean'):+.4f})\n"
        )

    if loss_dict.get("frontres_accept_pos_mean", None) is not None:
        lines.append(
            f"{'accept prob pos/rpy:':>{pad}} "
            f"{_value(loss_dict, 'frontres_accept_pos_mean'):.3f} / "
            f"{_value(loss_dict, 'frontres_accept_rpy_mean'):.3f} "
            f"(active={_value(loss_dict, 'frontres_accept_active_frac'):.3f})\n"
        )

    if loss_dict.get("frontres_proposal_ratio", None) is not None:
        lines.append(
            f"{'proposal ratio/leakage:':>{pad}} "
            f"{_value(loss_dict, 'frontres_proposal_ratio'):.3f} / "
            f"{_value(loss_dict, 'frontres_axis_leakage'):.3f}\n"
        )

    return "".join(lines)


def format_frontres_optimization_diagnostics(loss_dict: MetricMap, *, pad: int) -> str:
    """Format FrontRES optimization/update diagnostics."""
    lines: list[str] = []
    sal = loss_dict.get("state_alpha_loss", None)
    if sal is not None and _value(loss_dict, "lambda_state_alpha", 0.0) > 0.0:
        lines.append(
            f"{'state alpha loss:':>{pad}} {sal:.4f} "
            f"(λ={_value(loss_dict, 'lambda_state_alpha'):.3f}, "
            f"mask={_value(loss_dict, 'state_alpha_mask_frac'):.3f}, "
            f"tgt={_value(loss_dict, 'state_alpha_target_mean'):.3f}, "
            f"pred={_value(loss_dict, 'state_alpha_pred_mean'):.3f}, "
            f"acc={_value(loss_dict, 'state_alpha_acc'):.3f})\n"
        )

    authority_loss = loss_dict.get("authority_loss", None)
    authority_active = _value(loss_dict, "authority_active_frac", 0.0)
    authority_enabled = float(_value(loss_dict, "authority_actor_critic_enabled", 0.0)) > 0.5
    if authority_loss is not None and authority_enabled:
        lines.append(
            f"{'authority AC loss:':>{pad}} {authority_loss:.4f} "
            f"(actor={_value(loss_dict, 'authority_actor_loss'):+.4f}, "
            f"critic={_value(loss_dict, 'authority_critic_loss'):.4f}, "
            f"beh/0/1={_value(loss_dict, 'authority_critic_behavior_loss'):.4f}/"
            f"{_value(loss_dict, 'authority_critic_zero_loss'):.4f}/"
            f"{_value(loss_dict, 'authority_critic_one_loss'):.4f}, "
            f"λa={_value(loss_dict, 'lambda_authority_actor'):.3f}, "
            f"λa_eff={_value(loss_dict, 'lambda_authority_actor_effective'):.3f}, "
            f"λq={_value(loss_dict, 'lambda_authority_critic'):.3f}, "
            f"active={authority_active:.3f})\n"
        )
        lines.append(
            f"{'authority takeover:':>{pad}} "
            f"actor_phase={_value(loss_dict, 'authority_actor_phase_weight'):.3f}, "
            f"warmup={_value(loss_dict, 'authority_actor_warmup_active'):.0f}, "
            f"ramp={_value(loss_dict, 'authority_actor_ramp_active'):.0f}\n"
        )
        lines.append(
            f"{'authority return/Q:':>{pad}} "
            f"ret={_value(loss_dict, 'authority_return_mean'):+.4f}, "
            f"ret0/1={_value(loss_dict, 'authority_return_zero_mean'):+.4f}/"
            f"{_value(loss_dict, 'authority_return_one_mean'):+.4f}, "
            f"Qbeh={_value(loss_dict, 'authority_q_behavior_mean'):+.4f}, "
            f"Qact={_value(loss_dict, 'authority_q_actor_mean'):+.4f}\n"
        )
        lines.append(
            f"{'authority Q 0/act/1:':>{pad}} "
            f"{_value(loss_dict, 'authority_q_zero_mean'):+.4f} / "
            f"{_value(loss_dict, 'authority_q_actor_mean'):+.4f} / "
            f"{_value(loss_dict, 'authority_q_one_mean'):+.4f} "
            f"(1-0={_value(loss_dict, 'authority_q_one_minus_zero_mean'):+.4f}, "
            f"act-0={_value(loss_dict, 'authority_q_actor_minus_zero_mean'):+.4f})\n"
        )
        lines.append(
            f"{'authority actor ready:':>{pad}} "
            f"ready={_value(loss_dict, 'authority_actor_ready_frac'):.3f}, "
            f"accept={_value(loss_dict, 'authority_actor_ready_accept_frac'):.3f}, "
            f"reject={_value(loss_dict, 'authority_actor_ready_reject_frac'):.3f}, "
            f"harmful={_value(loss_dict, 'authority_harmful_full_write_frac'):.3f}, "
            f"conflict={_value(loss_dict, 'authority_target_conflict_frac'):.3f}\n"
        )
        lines.append(
            f"{'authority rho μ/σ/min/max:':>{pad}} "
            f"{_value(loss_dict, 'authority_rho_mean'):.3f} / "
            f"{_value(loss_dict, 'authority_rho_std'):.3f} / "
            f"{_value(loss_dict, 'authority_rho_min'):.3f} / "
            f"{_value(loss_dict, 'authority_rho_max'):.3f} "
            f"(near0={_value(loss_dict, 'authority_rho_near_zero_frac'):.3f}, "
            f"near1={_value(loss_dict, 'authority_rho_near_one_frac'):.3f})\n"
        )
        lines.append(
            f"{'authority rho dims:':>{pad}} "
            f"dx={_value(loss_dict, 'authority_rho_dx_mean'):.3f}, "
            f"dy={_value(loss_dict, 'authority_rho_dy_mean'):.3f}, "
            f"dz={_value(loss_dict, 'authority_rho_dz_mean'):.3f}, "
            f"r={_value(loss_dict, 'authority_rho_roll_mean'):.3f}, "
            f"p={_value(loss_dict, 'authority_rho_pitch_mean'):.3f}, "
            f"y={_value(loss_dict, 'authority_rho_yaw_mean'):.3f}\n"
        )
        lines.append(
            f"{'authority buckets L/M/H:':>{pad}} "
            f"ret={_value(loss_dict, 'authority_return_low_rho_mean'):+.3f}/"
            f"{_value(loss_dict, 'authority_return_mid_rho_mean'):+.3f}/"
            f"{_value(loss_dict, 'authority_return_high_rho_mean'):+.3f}, "
            f"Q={_value(loss_dict, 'authority_q_actor_low_rho_mean'):+.3f}/"
            f"{_value(loss_dict, 'authority_q_actor_mid_rho_mean'):+.3f}/"
            f"{_value(loss_dict, 'authority_q_actor_high_rho_mean'):+.3f}, "
            f"|Δ|={_value(loss_dict, 'authority_proposal_abs_low_rho_mean'):.3f}/"
            f"{_value(loss_dict, 'authority_proposal_abs_mid_rho_mean'):.3f}/"
            f"{_value(loss_dict, 'authority_proposal_abs_high_rho_mean'):.3f}\n"
        )
        lines.append(
            f"{'authority temporal:':>{pad}} "
            f"event-level, K={_value(loss_dict, 'authority_return_k_horizon', _value(loss_dict, 'authority_k_horizon')):.0f}, "
            f"mode={loss_dict.get('authority_temporal_mode', 'single')}, "
            f"events={_value(loss_dict, 'authority_event_count'):.0f}, "
            f"active={_value(loss_dict, 'authority_event_active_frac'):.3f}, "
            f"query={_value(loss_dict, 'authority_event_mask_frac'):.3f}, "
            f"dur={_value(loss_dict, 'authority_event_duration_mean'):.1f}\n"
        )

    sjl = loss_dict.get("structured_joint_rl_loss", None)
    if sjl is not None and _value(loss_dict, "lambda_structured_joint_rl") > 0.0:
        region_direct = _value(loss_dict, "structured_joint_rl_mode_region_direct") > 0.5
        if region_direct:
            lines.append(
                f"{'rho region loss:':>{pad}} {sjl:.4f} "
                f"(λ={_value(loss_dict, 'lambda_structured_joint_rl'):.3f}, "
                f"enabled={_value(loss_dict, 'structured_joint_rl_enabled'):.0f}, "
                f"rep={_value(loss_dict, 'structured_joint_rl_repairable_loss'):+.4f}, "
                f"bound={_value(loss_dict, 'structured_joint_rl_boundary_loss'):.4f}, "
                f"repair_bce={_value(loss_dict, 'structured_joint_rl_repair_loss_is_bce'):.0f}, "
                f"rscale={_value(loss_dict, 'structured_joint_rl_repair_loss_scale'):.3f}, "
                f"adv={_value(loss_dict, 'structured_joint_rl_adv_mean'):+.4f}, "
                f"|adv|={_value(loss_dict, 'structured_joint_rl_adv_abs_mean'):.4f}, "
                f"w_act={_value(loss_dict, 'structured_joint_rl_weight_mean'):.3f}, "
                f"dim={_value(loss_dict, 'structured_joint_rl_dim_active_mean'):.3f}, "
                f"p_auth={_value(loss_dict, 'structured_joint_rl_prior_authority_mean'):.3f}, "
                f"r_auth={_value(loss_dict, 'structured_joint_rl_repairable_authority_mean'):.3f}, "
                f"p_rho={_value(loss_dict, 'structured_joint_rl_prior_rho_mean'):.3f}, "
                f"rho={_value(loss_dict, 'structured_joint_rl_rho_mean'):.3f}, "
                f"|rho-.5|={_value(loss_dict, 'structured_joint_rl_rho_abs_from_half'):.3f}, "
                f"near.5={_value(loss_dict, 'structured_joint_rl_rho_near_half_frac'):.3f}, "
                f"generic={_value(loss_dict, 'ppo_actor_weight'):.3f}, "
                f"raw={_value(loss_dict, 'raw_ppo_actor_weight'):.3f})\n"
            )
        else:
            lines.append(
                f"{'joint rl loss:':>{pad}} {sjl:.4f} "
                f"(λ={_value(loss_dict, 'lambda_structured_joint_rl'):.3f}, "
                f"enabled={_value(loss_dict, 'structured_joint_rl_enabled'):.0f}, "
                f"adv={_value(loss_dict, 'structured_joint_rl_adv_mean'):+.4f}, "
                f"|adv|={_value(loss_dict, 'structured_joint_rl_adv_abs_mean'):.4f}, "
                f"w_act={_value(loss_dict, 'structured_joint_rl_weight_mean'):.3f}, "
                f"w_all={_value(loss_dict, 'structured_joint_rl_weight_all_mean'):.3f}, "
                f"dim={_value(loss_dict, 'structured_joint_rl_dim_active_mean'):.3f}, "
                f"prior={_value(loss_dict, 'structured_joint_rl_prior_loss'):.4f}, "
                f"p_auth={_value(loss_dict, 'structured_joint_rl_prior_authority_mean'):.3f}, "
                f"p_rho={_value(loss_dict, 'structured_joint_rl_prior_rho_mean'):.3f}, "
                f"rho={_value(loss_dict, 'structured_joint_rl_rho_mean'):.3f}, "
                f"|rho-.5|={_value(loss_dict, 'structured_joint_rl_rho_abs_from_half'):.3f}, "
                f"near.5={_value(loss_dict, 'structured_joint_rl_rho_near_half_frac'):.3f}, "
                f"act-mu={_value(loss_dict, 'structured_joint_rl_rho_action_minus_mean_abs'):.4f}, "
                f"generic={_value(loss_dict, 'ppo_actor_weight'):.3f}, "
                f"raw={_value(loss_dict, 'raw_ppo_actor_weight'):.3f}, "
                f"rho_ratio={_value(loss_dict, 'structured_joint_rl_ratio_mean'):.3f})\n"
            )
        lines.append(
            f"{'rho adv sign pos/neg/zero:':>{pad}} "
            f"{_value(loss_dict, 'structured_joint_rl_adv_pos_frac'):.3f} / "
            f"{_value(loss_dict, 'structured_joint_rl_adv_neg_frac'):.3f} / "
            f"{_value(loss_dict, 'structured_joint_rl_adv_near_zero_frac'):.3f}\n"
        )
        lines.append(
            f"{'rho by adv +/−:':>{pad}} "
            f"{_value(loss_dict, 'structured_joint_rl_rho_pos_adv_mean'):.3f} / "
            f"{_value(loss_dict, 'structured_joint_rl_rho_neg_adv_mean'):.3f} "
            f"(|adv|={_value(loss_dict, 'structured_joint_rl_adv_pos_abs_mean'):.3f} / "
            f"{_value(loss_dict, 'structured_joint_rl_adv_neg_abs_mean'):.3f})\n"
        )
        lines.append(
            f"{'rho region means:':>{pad}} "
            f"repair={_value(loss_dict, 'structured_joint_rl_rho_repairable_mean'):.3f}, "
            f"boundary={_value(loss_dict, 'structured_joint_rl_rho_boundary_mean'):.3f}, "
            f"repair_pos={_value(loss_dict, 'structured_joint_rl_repairable_pos_frac'):.3f}\n"
        )
        lines.append(
            f"{'rho dims pos/rpy:':>{pad}} "
            f"active={_value(loss_dict, 'structured_joint_rl_active_frac_pos_dim'):.3f} / "
            f"{_value(loss_dict, 'structured_joint_rl_active_frac_rpy_dim'):.3f}, "
            f"all={_fmt_active_value(loss_dict, 'structured_joint_rl_rho_pos_dim_mean', 'structured_joint_rl_active_frac_pos_dim')} / "
            f"{_fmt_active_value(loss_dict, 'structured_joint_rl_rho_rpy_dim_mean', 'structured_joint_rl_active_frac_rpy_dim')}, "
            f"+adv={_fmt_active_value(loss_dict, 'structured_joint_rl_rho_pos_adv_pos_dim_mean', 'structured_joint_rl_active_frac_pos_dim')} / "
            f"{_fmt_active_value(loss_dict, 'structured_joint_rl_rho_pos_adv_rpy_dim_mean', 'structured_joint_rl_active_frac_rpy_dim')}, "
            f"-adv={_fmt_active_value(loss_dict, 'structured_joint_rl_rho_neg_adv_pos_dim_mean', 'structured_joint_rl_active_frac_pos_dim')} / "
            f"{_fmt_active_value(loss_dict, 'structured_joint_rl_rho_neg_adv_rpy_dim_mean', 'structured_joint_rl_active_frac_rpy_dim')}\n"
        )
        lines.append(
            f"{'adv+ pos/rpy frac:':>{pad}} "
            f"{_fmt_active_value(loss_dict, 'structured_joint_rl_adv_pos_frac_pos_dim', 'structured_joint_rl_active_frac_pos_dim')} / "
            f"{_fmt_active_value(loss_dict, 'structured_joint_rl_adv_pos_frac_rpy_dim', 'structured_joint_rl_active_frac_rpy_dim')}\n"
        )
    return "".join(lines)
