"""Legacy live-probe summary construction and serialization."""





from __future__ import annotations





from typing import Any


import torch





from rsl_rl.runners.frontres_segment_runtime_types import (


    FrontRESSegmentLiveRolloutCapture,


)


from rsl_rl.runners.frontres_segment_probe_logging import (
    tensor_bool_list as _bool_list,
    capture_audit_identity_kwargs as _capture_audit_identity_kwargs,
    delta_se_norm as _delta_se_norm,
    delta_z_up_fraction as _delta_z_up_frac,
    finite_tensor_mean as _finite_mean,
    tensor_float_list as _float_list,
    format_probe_metric as _fmt_metric,
    format_probe_number as _fmt_num,
    format_probe_percent as _fmt_pct,
    format_probe_vector as _fmt_vec,
    probe_kv_lines as _kv_lines,
    live_detail_log_enabled as _live_detail_log_enabled,
    probe_log_block as _log_block,
    positive_fraction as _positive_fraction,
    probe_status as _probe_status,
    shape_last_dim as _shape_last_dim,
    tensor_range_summary as _tensor_range_summary,
)


from rsl_rl.runners.frontres_segment_live_storage import (
    capture_frontres_averaged_repair_scores as _capture_averaged_repair_scores,
    capture_frontres_averaged_rewards as _capture_averaged_rewards,
    capture_frontres_paired_gain as _capture_paired_gain,
)





def _initial_live_probe_summary(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    storage_write: bool,
    single_update: bool,
) -> dict[str, object]:
    """Build the active live summary from canonical paired Gain evidence.

    Status: active diagnostic boundary.
    Upstream: captured paired rollout and ``_capture_paired_gain``.
    Downstream: sampler evidence, update-loop aggregation, and train logs.
    Evidence: Step 7 implementation path; legacy per-row score fields remain
    only for sampler compatibility and are not active train diagnostics.
    Gap: real simulator component population remains an S4 boundary.
    """
    legacy_score_compatibility = _paired_score_summary(capture)
    paired_gain = _capture_paired_gain(capture)
    gain_summary = _paired_gain_summary(capture)
    gain_total_pos_frac = (
        _positive_fraction(_float_list(paired_gain.gain_total))
        if paired_gain is not None
        else float("nan")
    )
    summary = {
        **_capture_audit_identity_kwargs(capture),
        "rollout_k": capture.rollout_k,
        "rollout_horizon_summary": _tensor_range_summary("horizon", capture.horizon_k)
        if isinstance(capture.horizon_k, torch.Tensor)
        else f"horizon_count=0 horizon_min={capture.rollout_k} horizon_max={capture.rollout_k}",
        "reward_mean": capture.reward_mean,
        "env_reward_mean": capture.reward_mean,
        "train_reward_mean": capture.reward_mean,
        "gain_total_pos_frac": gain_total_pos_frac,
        "motion_delta_se_norm": _delta_se_norm(capture.transition_actions),
        "motion_delta_z_up_frac": _delta_z_up_frac(capture.transition_actions),
        "done_frac": capture.done_frac,
        "valid_mask_frac": 1.0 - capture.done_frac,
        "reward_per_sample": _rollout_reward_per_sample(capture),
        "done_any_per_sample": _rollout_done_per_sample(capture),
        "storage_write": storage_write,
        "storage_size": 0,
        "storage_valid_frac": 0.0,
        "storage_reward_mean": 0.0,
        "storage_reward_per_sample": [],
        "storage_valid_mask_per_sample": [],
        "storage_segment_ids": [],
        "single_update": single_update,
        "ppo_update": False,
        "ppo_valid_count": 0,
        "ppo_total_loss": 0.0,
        "ppo_actor_loss": 0.0,
        "ppo_value_loss": 0.0,
        "ppo_approx_kl": 0.0,
        "ppo_clip_frac": 0.0,
        "ppo_pre_update_raw_log_ratio_mean": 0.0,
        "ppo_pre_update_raw_log_ratio_min": 0.0,
        "ppo_pre_update_raw_log_ratio_max": 0.0,
        "ppo_pre_update_clamped_ratio_mean": 0.0,
        "ppo_pre_update_clamped_ratio_max": 0.0,
        "ppo_pre_distribution_kl_mean": 0.0,
        "ppo_pre_logprob_approx_kl": 0.0,
        "ppo_distribution_kl_available": False,
        "ppo_post_update_distribution_kl_mean": 0.0,
        "ppo_post_update_logprob_approx_kl": 0.0,
        "ppo_post_update_ratio_mean": 0.0,
        "ppo_post_update_ratio_max": 0.0,
        "ppo_post_update_raw_log_ratio_mean": 0.0,
        "ppo_post_update_raw_log_ratio_min": 0.0,
        "ppo_post_update_raw_log_ratio_max": 0.0,
        "ppo_post_update_clamped_ratio_mean": 0.0,
        "ppo_post_update_clamped_ratio_max": 0.0,
        "ppo_post_update_clip_frac": 0.0,
        "ppo_param_delta_max_abs": 0.0,
        "ppo_param_delta_l2": 0.0,
        "ppo_param_delta_changed": 0,
        "ppo_param_delta_total": 0,
        "ppo_param_delta_first_changed": "",
        "ppo_param_grad_norm": 0.0,
        "ppo_trust_region_rejected_count": 0,
        "ppo_trust_region_accepted": 1,
    }
    # Compatibility vectors are retained for sampler evidence serialization;
    # no legacy scalar is used by active diagnostics or training aggregation.
    summary.update(legacy_score_compatibility)
    summary.update(gain_summary)
    summary.update(_motion_quality_summary(capture))
    return summary


def _motion_quality_summary(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, float]:
    try:
        from rsl_rl.frontres.frontres_segment_reporting import motion_quality_summary_to_scalars
    except ModuleNotFoundError:
        return {}
    positions = capture.motion_repaired_body_pos
    temporal_mask = None
    valid_mask = capture.actor_update_mask
    if isinstance(positions, torch.Tensor):
        batch_size, time_steps = int(positions.shape[0]), int(positions.shape[1])
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask[:batch_size]
        if isinstance(capture.horizon_k, torch.Tensor):
            horizon = capture.horizon_k[:batch_size].to(device=positions.device, dtype=torch.long)
            temporal_mask = torch.arange(time_steps, device=positions.device).view(1, -1) < horizon.view(-1, 1)
    return motion_quality_summary_to_scalars(
        clean_positions=capture.motion_clean_body_pos,
        repaired_positions=capture.motion_repaired_body_pos,
        noisy_positions=capture.motion_noisy_body_pos,
        delta_se=capture.transition_actions,
        valid_mask=valid_mask,
        temporal_mask=temporal_mask,
    )


def _paired_score_summary(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, object]:
    """Return legacy executable-score vectors for compatibility evidence only.

    Status: legacy compatibility boundary, not an active training diagnostic.
    Upstream: paired rollout capture. Downstream: sampler evidence compatibility
    fields and migration tests only. Evidence: Step 6C/7 audit.
    Gap: the active route must use ``_paired_gain_summary`` instead.
    """
    if capture.done_any is None:
        return {}
    n_train = max(0, int(capture.n_train))
    n_candidate = max(0, int(capture.n_candidate))
    n_base = max(0, int(capture.n_base))
    n_clean = max(0, int(capture.n_clean))
    n = min(n_train, n_base)
    if n <= 0:
        return {}
    score = _capture_averaged_repair_scores(capture)
    done = capture.done_any.reshape(-1).detach().bool()
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    if int(score.numel()) < base_start + n:
        return {}
    clean = score[clean_start : clean_start + n] if n_clean >= n and int(score.numel()) >= clean_start + n else torch.ones(n, device=score.device)
    noisy = score[base_start : base_start + n]
    repaired = score[:n]
    return {
        "evidence_row_count": n,
        "evidence_reward_per_sample": _float_list(repaired),
        "evidence_done_any_per_sample": _bool_list(done[:n]),
        "evidence_valid_mask_per_sample": _bool_list(~done[:n]),
        "score_repaired_per_sample": _float_list(repaired),
        "score_noisy_per_sample": _float_list(noisy),
        "gain_over_noisy_per_sample": _float_list(repaired - noisy),
        "score_clean_per_sample": _float_list(clean),
        "score_source": "repair_executability",
    }


def _paired_gain_summary(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, object]:
    result = _capture_paired_gain(capture)
    if result is None:
        return {**_capture_audit_identity_kwargs(capture), "gain_source": "UNCONFIRMED"}
    return {
        **_capture_audit_identity_kwargs(capture),
        "gain_source": "FRS-GAIN-v002",
        "gain_style_per_sample": _float_list(result.style_gain),
        "gain_physics_per_sample": _float_list(result.physics_gain),
        "gain_repair_cost_per_sample": _float_list(result.repair_cost),
        "gain_total_per_sample": _float_list(result.gain_total),
        "gain_style_mean": _finite_mean(result.style_gain),
        "gain_physics_mean": _finite_mean(result.physics_gain),
        "gain_repair_cost_mean": _finite_mean(result.repair_cost),
        "gain_total_mean": _finite_mean(result.gain_total),
        "gain_style_mpjpe_mean": _finite_mean(result.style_mpjpe_gain),
        "gain_style_velocity_mean": _finite_mean(result.style_velocity_gain),
        "gain_style_acceleration_mean": _finite_mean(result.style_acceleration_gain),
        "gain_style_root_orientation_mean": _finite_mean(result.style_root_orientation_gain),
        "gain_physics_success_mean": _finite_mean(result.physics_success_gain),
        "gain_physics_survival_quality_repaired_per_sample": _float_list(result.physics_survival_quality_repaired),
        "gain_physics_survival_quality_noisy_per_sample": _float_list(result.physics_survival_quality_noisy),
        "gain_physics_survival_per_sample": _float_list(result.physics_survival_gain),
        "gain_physics_survival_quality_repaired_mean": _finite_mean(result.physics_survival_quality_repaired),
        "gain_physics_survival_quality_noisy_mean": _finite_mean(result.physics_survival_quality_noisy),
        "gain_physics_survival_mean": _finite_mean(result.physics_survival_gain),
        "gain_physics_zmp_mean": _finite_mean(result.physics_zmp_gain),
        "gain_physics_contact_mean": _finite_mean(result.physics_contact_gain),
        "gain_repair_norm_mean": _finite_mean(result.repair_norm),
        "gain_repair_temporal_mean": _finite_mean(result.repair_temporal_change),
        "gain_repair_clean_cost_per_sample": _float_list(result.repair_clean_cost),
        "gain_repair_clean_cost_mean": _finite_mean(result.repair_clean_cost),
    }


def _rollout_reward_per_sample(capture: FrontRESSegmentLiveRolloutCapture) -> list[float]:
    if capture.reward_accum is None:
        return []
    reward = _capture_averaged_rewards(capture)
    return _float_list(reward)


def _rollout_done_per_sample(capture: FrontRESSegmentLiveRolloutCapture) -> list[bool]:
    if capture.done_any is None:
        return []
    return _bool_list(capture.done_any.reshape(-1))


def _valid_reward_mean(rewards: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid = valid_mask.detach().bool().reshape(-1)
    reward = rewards.detach().float().reshape(-1)
    if int(valid.numel()) != int(reward.numel()):
        raise ValueError(f"valid_mask must have {int(reward.numel())} rows, got {int(valid.numel())}")
    if not bool(valid.any().item()):
        return 0.0
    return float(reward[valid].mean().cpu().item())


def _print_live_probe_summary(
    runner: Any,
    capture: FrontRESSegmentLiveRolloutCapture,
    summary: dict[str, object],
) -> None:
    """Print the human-facing live probe blocks without changing training state.

    Status: active diagnostic formatter.
    Upstream: run_frontres_segment_live_probe builds summary from rollout, storage, and PPO result.
    Downstream: terminal/log review only; no sampler, loss, optimizer, or checkpoint side effect.
    Evidence: contract-confirmed by frontres_segment_live_probe_contract.py.
    Gap: text presence does not prove live physics quality.
    """
    if not _live_detail_log_enabled(runner):
        return
    segment_action_shape = (
        tuple(capture.transition_actions.shape) if capture.transition_actions is not None else None
    )
    segment_delta_se_6d = bool(_shape_last_dim(segment_action_shape) == 6)
    print(
        _log_block(
            "[FrontRES Segment Live Probe]",
            *_kv_lines(
                "route",
                {
                    "objective": getattr(runner.alg, "frontres_training_objective", "n/a"),
                    "segment_id": "live_env_current",
                    "reset_mode": runner._frontres_segment_replay_boundary.reset_mode,
                },
            ),
            *_kv_lines(
                "reset",
                {
                    "enabled": bool(summary["segment_reset"]),
                    "reason": summary.get("segment_reset_skip_reason", "") or "applied",
                    "ok": _fmt_pct(summary["segment_reset_success_frac"]),
                    "direct": _fmt_pct(summary["segment_reset_direct_frac"]),
                    "preroll": _fmt_pct(summary["segment_reset_preroll_frac"]),
                    "vel_mismatch": _fmt_num(summary["segment_reset_velocity_mismatch_mean"]),
                    "ref_window": _fmt_pct(summary["segment_reference_window_applied_frac"]),
                },
            ),
            *_kv_lines(
                "rollout",
                {
                    "obs": capture.last_obs_shape,
                    "policy_action": capture.action_shape,
                    "policy_dim": _shape_last_dim(capture.action_shape),
                    "segment_action": segment_action_shape,
                    "segment_delta_se_6d": segment_delta_se_6d,
                    "env_action": capture.env_action_shape,
                    "env_dim": _shape_last_dim(capture.env_action_shape),
                    "k": capture.rollout_k,
                    "horizon": summary.get("rollout_horizon_summary", "unavailable"),
                    "env_reward": _fmt_num(summary.get("env_reward_mean", summary["reward_mean"])),
                    "done": _fmt_pct(summary["done_frac"]),
                },
            ),
            *_kv_lines(
                "trial",
                {
                    "roles": summary.get("trial_role_counts", {}),
                    "policy": int(summary.get("trial_policy_count", 0) or 0),
                    "search": int(summary.get("trial_search_count", 0) or 0),
                    "horizon": summary.get("trial_horizon_summary", "horizon_count=0 horizon_min=None horizon_max=None"),
                },
            ),
            *_kv_lines(
                "ppo_boundary",
                {
                    "evidence": int(summary.get("ppo_boundary_evidence_rows", 0) or 0),
                    "policy": int(summary.get("ppo_boundary_policy_rows", 0) or 0),
                    "search": int(summary.get("ppo_boundary_search_rows", 0) or 0),
                    "ppo_valid": int(summary.get("ppo_boundary_eligible_rows", summary.get("ppo_valid_count", 0)) or 0),
                    "search_evidence_only": int(summary.get("ppo_boundary_search_evidence_only_rows", 0) or 0),
                    "policy_invalid": int(summary.get("ppo_boundary_policy_invalid_rows", 0) or 0),
                    "valid_policy": _fmt_pct(summary.get("ppo_boundary_valid_policy_frac", 0.0)),
                    "valid_evidence": _fmt_pct(summary.get("ppo_boundary_valid_evidence_frac", 0.0)),
                },
            ),
            *_kv_lines(
                "gain",
                {
                    "source": summary.get("gain_source", "UNCONFIRMED"),
                    "style": _fmt_metric(summary.get("gain_style_mean")),
                    "physics": _fmt_metric(summary.get("gain_physics_mean")),
                    "repair_cost": _fmt_metric(summary.get("gain_repair_cost_mean")),
                    "total": _fmt_metric(summary.get("gain_total_mean")),
                    "mpjpe": _fmt_metric(summary.get("gain_style_mpjpe_mean")),
                    "velocity": _fmt_metric(summary.get("gain_style_velocity_mean")),
                    "acceleration": _fmt_metric(summary.get("gain_style_acceleration_mean")),
                    "root_orientation": _fmt_metric(summary.get("gain_style_root_orientation_mean")),
                    "success": _fmt_metric(summary.get("gain_physics_success_mean")),
                    "survival_quality_repaired": _fmt_metric(summary.get("gain_physics_survival_quality_repaired_mean")),
                    "survival_quality_noisy": _fmt_metric(summary.get("gain_physics_survival_quality_noisy_mean")),
                    "survival_quality": _fmt_metric(summary.get("gain_physics_survival_mean")),
                    "zmp": _fmt_metric(summary.get("gain_physics_zmp_mean")),
                    "contact": _fmt_metric(summary.get("gain_physics_contact_mean")),
                    "repair_norm": _fmt_metric(summary.get("gain_repair_norm_mean")),
                    "repair_temporal": _fmt_metric(summary.get("gain_repair_temporal_mean")),
                },
            ),
            *_kv_lines(
                "storage",
                {
                    "write": bool(summary["storage_write"]),
                    "size": int(summary["storage_size"]),
                    "mask_valid": _fmt_pct(summary["valid_mask_frac"]),
                    "valid_frac": _fmt_pct(summary["storage_valid_frac"]),
                    "train_reward": _fmt_num(summary.get("train_reward_mean", summary["storage_reward_mean"])),
                    "all_reward": _fmt_num(summary["storage_reward_mean"]),
                },
            ),
            *_kv_lines(
                "ppo",
                {
                    "single_update": bool(summary["single_update"]),
                    "update": bool(summary["ppo_update"]),
                    "valid": int(summary["ppo_valid_count"]),
                    "loss_total": _fmt_num(summary["ppo_total_loss"]),
                    "actor": _fmt_num(summary["ppo_actor_loss"]),
                    "value": _fmt_num(summary["ppo_value_loss"]),
                    "kl": _fmt_num(summary["ppo_approx_kl"]),
                    "clip": _fmt_pct(summary["ppo_clip_frac"]),
                    "status": _probe_status(summary),
                },
            ),
        ),
        flush=True,
    )
    if bool(summary.get("ppo_update", False)):
        # B1: Separate the same-batch PPO evidence by time. pre_* comes from
        # the loss forward before optimizer.step; post_* comes from the second
        # forward after optimizer.step on the same stored batch.
        print(
            _log_block(
                "[FrontRES Segment PPO Probe]",
                *_kv_lines(
                    "log_prob",
                    {
                        "old": _fmt_num(summary.get("ppo_old_log_prob_mean", 0.0)),
                        "new": _fmt_num(summary.get("ppo_new_log_prob_mean", 0.0)),
                    },
                ),
                *_kv_lines(
                    "pre_log_ratio",
                    {
                        "mean": _fmt_num(summary.get("ppo_pre_update_raw_log_ratio_mean", 0.0)),
                        "min": _fmt_num(summary.get("ppo_pre_update_raw_log_ratio_min", 0.0)),
                        "max": _fmt_num(summary.get("ppo_pre_update_raw_log_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "pre_ratio",
                    {
                        "clamped_mean": _fmt_num(summary.get("ppo_pre_update_clamped_ratio_mean", 0.0)),
                        "clamped_max": _fmt_num(summary.get("ppo_pre_update_clamped_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "kl",
                    {
                        "pre_distribution": _fmt_num(summary.get("ppo_pre_distribution_kl_mean", 0.0)),
                        "pre_logprob": _fmt_num(summary.get("ppo_pre_logprob_approx_kl", 0.0)),
                        "post_distribution": _fmt_num(
                            summary.get("ppo_post_update_distribution_kl_mean", 0.0)
                        ),
                        "post_logprob": _fmt_num(summary.get("ppo_post_update_logprob_approx_kl", 0.0)),
                        "distribution_available": bool(summary.get("ppo_distribution_kl_available", False)),
                    },
                ),
                *_kv_lines(
                    "post_log_ratio",
                    {
                        "mean": _fmt_num(summary.get("ppo_post_update_raw_log_ratio_mean", 0.0)),
                        "min": _fmt_num(summary.get("ppo_post_update_raw_log_ratio_min", 0.0)),
                        "max": _fmt_num(summary.get("ppo_post_update_raw_log_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "post_ratio",
                    {
                        "clamped_mean": _fmt_num(summary.get("ppo_post_update_clamped_ratio_mean", 0.0)),
                        "clamped_max": _fmt_num(summary.get("ppo_post_update_clamped_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "ratio_source",
                    {
                        "raw_action_old_mean_l2": _fmt_num(
                            summary.get("ppo_post_update_raw_action_old_mean_l2_mean", 0.0)
                        ),
                        "raw_action_old_mean_abs_max": _fmt_num(
                            summary.get("ppo_post_update_raw_action_old_mean_abs_max", 0.0)
                        ),
                        "raw_action_old_mean_abs_dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_raw_action_old_mean_abs_dim_mean", ())
                        ),
                        "raw_action_old_mean_abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_raw_action_old_mean_abs_dim_max", ())
                        ),
                    },
                ),
                *_kv_lines(
                    "ratio_sigma",
                    {
                        "old_dim_mean": _fmt_vec(summary.get("ppo_post_update_old_sigma_dim_mean", ())),
                        "new_dim_mean": _fmt_vec(summary.get("ppo_post_update_sigma_dim_mean", ())),
                    },
                ),
                *_kv_lines(
                    "ratio_mean_delta",
                    {
                        "dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_distribution_mean_delta_dim_mean", ())
                        ),
                        "abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_distribution_mean_delta_abs_dim_max", ())
                        ),
                    },
                ),
                *_kv_lines(
                    "ratio_contrib",
                    {
                        "log_ratio_dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_log_ratio_contrib_dim_mean", ())
                        ),
                        "log_ratio_abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_log_ratio_contrib_abs_dim_max", ())
                        ),
                        "log_jacobian_dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_log_jacobian_dim_mean", ())
                        ),
                        "log_jacobian_abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_log_jacobian_abs_dim_max", ())
                        ),
                    },
                ),
                *_kv_lines(
                    "trust",
                    {
                        "accepted": bool(summary.get("ppo_trust_region_accepted", 1)),
                        "rejected": int(summary.get("ppo_trust_region_rejected_count", 0)),
                        "lr_before": _fmt_num(summary.get("ppo_adaptive_lr_before", 0.0)),
                        "lr_after": _fmt_num(summary.get("ppo_adaptive_lr_after", 0.0)),
                        "desired_kl": _fmt_num(summary.get("ppo_adaptive_lr_desired_kl", 0.0)),
                        "schedule": str(summary.get("ppo_trust_region_schedule", "unknown")),
                        "rollback": bool(summary.get("ppo_trust_region_rollback_enabled", 0)),
                        "max_retries": int(summary.get("ppo_trust_region_max_retries", 0)),
                    },
                ),
                *_kv_lines(
                    "advantage",
                    {
                        "mean": _fmt_num(summary.get("ppo_advantage_mean", 0.0)),
                        "min": _fmt_num(summary.get("ppo_advantage_min", 0.0)),
                        "max": _fmt_num(summary.get("ppo_advantage_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "param_delta",
                    {
                        "max_abs": _fmt_num(summary.get("ppo_param_delta_max_abs", 0.0)),
                        "l2": _fmt_num(summary.get("ppo_param_delta_l2", 0.0)),
                        "changed": (
                            f"{int(summary.get('ppo_param_delta_changed', 0))}/"
                            f"{int(summary.get('ppo_param_delta_total', 0))}"
                        ),
                        "first": summary.get("ppo_param_delta_first_changed", ""),
                        "grad_norm": _fmt_num(summary.get("ppo_param_grad_norm", 0.0)),
                    },
                ),
            ),
            flush=True,
        )


# Public report boundary used only by the legacy probe use case.
build_initial_live_probe_summary = _initial_live_probe_summary
print_live_probe_summary = _print_live_probe_summary
valid_reward_mean = _valid_reward_mean
