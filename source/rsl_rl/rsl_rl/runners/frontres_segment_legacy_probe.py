"""Legacy Segment live-probe use case retained behind the public facade."""





from __future__ import annotations





from typing import Any


import torch


from rsl_rl.frontres.training_schedule import resolve_frontres_mode_state


from rsl_rl.modules import FrontRESActorCritic


from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout


from rsl_rl.runners.frontres_formal_runtime_audit import print_rollout_storage_audit





from rsl_rl.runners.frontres_segment_probe_logging import (
    tensor_bool_list as _bool_list,
    tensor_float_list as _float_list,
    tensor_long_list as _long_list,
    print_frontres_dr_runtime_probe as _print_frontres_dr_runtime_probe,
)


from rsl_rl.runners.frontres_segment_live_policy import (


    run_frontres_segment_single_update,


)


from rsl_rl.runners.frontres_segment_live_reset import (
    apply_frontres_current_segment_reset as _apply_current_segment_reset,
    capture_frontres_batch_size as _capture_batch_size,
    update_frontres_ppo_boundary_summary as _update_ppo_boundary_summary,
    update_frontres_reset_summary as _update_reset_summary,
    update_frontres_trial_metadata_summary as _update_trial_metadata_summary,
)


from rsl_rl.runners.frontres_segment_live_storage import (


    build_live_segment_storage,


)


from rsl_rl.runners.frontres_segment_one_action_k import (
    read_frontres_live_observations as _read_live_observations,
    resolve_frontres_probe_modes as _resolve_probe_modes,
)


from rsl_rl.runners.frontres_segment_live_rollout import (
    run_frontres_live_rollout_capture as _run_live_rollout_capture,
)


from rsl_rl.runners.frontres_segment_probe_reporting import (
    build_initial_live_probe_summary as _initial_live_probe_summary,
    print_live_probe_summary as _print_live_probe_summary,
    valid_reward_mean as _valid_reward_mean,
)





def run_frontres_segment_live_probe(runner: Any, init_at_random_ep_len: bool = True) -> dict[str, object]:
    single_update, storage_write = _resolve_probe_modes(runner)
    episode_before = runner.env.episode_length_buf.detach().clone()
    if init_at_random_ep_len:
        runner.env.episode_length_buf = torch.randint_like(
            runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
        )
    episode_randomized = runner.env.episode_length_buf.detach().clone()

    frontres_mode = resolve_frontres_mode_state(runner, FrontRESActorCritic)
    pair_layout = configure_frontres_pair_layout(runner, is_frontres=frontres_mode.is_frontres)
    reset_result = _apply_current_segment_reset(runner, pair_layout=pair_layout)
    episode_after_reset = runner.env.episode_length_buf.detach().clone()
    reset_skip_reason = str(getattr(runner, "_frontres_segment_live_current_reset_skip_reason", "") or "")
    _print_frontres_dr_runtime_probe(runner, label="after_current_segment_reset")
    observations = _read_live_observations(runner)
    runner.eval_mode()
    capture = _run_live_rollout_capture(
        runner,
        observations,
        reset_lifecycle={
            "episode_before": episode_before,
            "episode_randomized": episode_randomized,
            "episode_after_reset": episode_after_reset,
        },
        pair_layout=pair_layout,
    )
    summary = _initial_live_probe_summary(capture, storage_write=storage_write, single_update=single_update)
    _update_trial_metadata_summary(summary, runner, batch_size=_capture_batch_size(capture))
    _update_reset_summary(
        summary,
        reset_result,
        request=getattr(runner, "_frontres_segment_live_current_reset_request", None),
        skip_reason=reset_skip_reason,
    )

    storage_batch = None
    if storage_write:
        segment_storage = build_live_segment_storage(runner, capture)
        storage_stats = segment_storage.stats()
        storage_batch = segment_storage.full_batch()
        _update_ppo_boundary_summary(summary, storage_batch.valid_mask)
        train_reward_mean = _valid_reward_mean(storage_batch.returns, storage_batch.valid_mask)
        summary.update(
            {
                "storage_size": storage_stats.size,
                "storage_valid_frac": storage_stats.valid_frac,
                "storage_reward_mean": storage_stats.reward_mean,
                "train_reward_mean": train_reward_mean,
                "storage_reward_per_sample": _float_list(storage_batch.returns),
                "storage_valid_mask_per_sample": _bool_list(storage_batch.valid_mask),
                "storage_segment_ids": _long_list(storage_batch.segment_ids),
            }
        )
        if single_update:
            ppo_result = run_frontres_segment_single_update(runner, storage_batch)
            summary.update(
                {
                    "ppo_update": bool(ppo_result.should_step),
                    "ppo_total_loss": float(ppo_result.total_loss.detach().cpu().item()),
                    "ppo_actor_loss": float(ppo_result.actor_loss.detach().cpu().item()),
                    "ppo_value_loss": float(ppo_result.value_loss.detach().cpu().item()),
                    "ppo_valid_count": int(ppo_result.valid_count),
                    "ppo_approx_kl": float(ppo_result.approx_kl),
                    "ppo_clip_frac": float(ppo_result.clip_frac),
                    "ppo_ratio_mean": float(ppo_result.ratio_mean),
                    "ppo_ratio_max": float(ppo_result.ratio_max),
                    "ppo_old_log_prob_mean": float(ppo_result.old_log_prob_mean),
                    "ppo_new_log_prob_mean": float(ppo_result.new_log_prob_mean),
                    "ppo_raw_log_ratio_mean": float(ppo_result.raw_log_ratio_mean),
                    "ppo_raw_log_ratio_min": float(ppo_result.raw_log_ratio_min),
                    "ppo_raw_log_ratio_max": float(ppo_result.raw_log_ratio_max),
                    "ppo_pre_update_raw_log_ratio_mean": float(
                        ppo_result.pre_update_raw_log_ratio_mean
                    ),
                    "ppo_pre_update_raw_log_ratio_min": float(
                        ppo_result.pre_update_raw_log_ratio_min
                    ),
                    "ppo_pre_update_raw_log_ratio_max": float(
                        ppo_result.pre_update_raw_log_ratio_max
                    ),
                    "ppo_pre_update_clamped_ratio_mean": float(
                        ppo_result.pre_update_clamped_ratio_mean
                    ),
                    "ppo_pre_update_clamped_ratio_max": float(
                        ppo_result.pre_update_clamped_ratio_max
                    ),
                    "ppo_pre_distribution_kl_mean": float(getattr(ppo_result, "distribution_kl_mean", 0.0)),
                    "ppo_pre_logprob_approx_kl": float(getattr(ppo_result, "logprob_approx_kl", 0.0)),
                    "ppo_distribution_kl_available": bool(
                        getattr(ppo_result, "distribution_kl_available", False)
                    ),
                    "ppo_post_update_distribution_kl_mean": float(
                        getattr(ppo_result, "post_update_distribution_kl_mean", 0.0)
                    ),
                    "ppo_post_update_logprob_approx_kl": float(
                        getattr(ppo_result, "post_update_logprob_approx_kl", 0.0)
                    ),
                    "ppo_post_update_ratio_mean": float(
                        getattr(ppo_result, "post_update_ratio_mean", 0.0)
                    ),
                    "ppo_post_update_ratio_max": float(
                        getattr(ppo_result, "post_update_ratio_max", 0.0)
                    ),
                    "ppo_post_update_raw_log_ratio_mean": float(
                        getattr(ppo_result, "post_update_raw_log_ratio_mean", 0.0)
                    ),
                    "ppo_post_update_raw_log_ratio_min": float(
                        getattr(ppo_result, "post_update_raw_log_ratio_min", 0.0)
                    ),
                    "ppo_post_update_raw_log_ratio_max": float(
                        getattr(ppo_result, "post_update_raw_log_ratio_max", 0.0)
                    ),
                    "ppo_post_update_clamped_ratio_mean": float(
                        getattr(ppo_result, "post_update_clamped_ratio_mean", 0.0)
                    ),
                    "ppo_post_update_clamped_ratio_max": float(
                        getattr(ppo_result, "post_update_clamped_ratio_max", 0.0)
                    ),
                    "ppo_post_update_clip_frac": float(
                        getattr(ppo_result, "post_update_clip_frac", 0.0)
                    ),
                    "ppo_advantage_mean": float(ppo_result.advantage_mean),
                    "ppo_advantage_min": float(ppo_result.advantage_min),
                    "ppo_advantage_max": float(ppo_result.advantage_max),
                    "ppo_advantage_abs_mean": float(ppo_result.advantage_abs_mean),
                    "ppo_advantage_abs_max": float(ppo_result.advantage_abs_max),
                    "ppo_advantage_abs_top1_frac": float(ppo_result.advantage_abs_top1_frac),
                    "ppo_distribution_mean_delta_l2_mean": float(
                        ppo_result.distribution_mean_delta_l2_mean
                    ),
                    "ppo_distribution_mean_delta_max_abs": float(
                        ppo_result.distribution_mean_delta_max_abs
                    ),
                    "ppo_old_sigma_min": float(ppo_result.old_sigma_min),
                    "ppo_sigma_min": float(ppo_result.sigma_min),
                    "ppo_param_delta_max_abs": float(getattr(ppo_result, "param_delta_max_abs", 0.0)),
                    "ppo_param_delta_l2": float(getattr(ppo_result, "param_delta_l2", 0.0)),
                    "ppo_param_delta_changed": int(getattr(ppo_result, "param_delta_changed", 0)),
                    "ppo_param_delta_total": int(getattr(ppo_result, "param_delta_total", 0)),
                    "ppo_param_delta_first_changed": str(getattr(ppo_result, "param_delta_first_changed", "")),
                    "ppo_param_grad_norm": float(getattr(ppo_result, "param_grad_norm", 0.0)),
                    "ppo_warmup_phase": str(getattr(ppo_result, "warmup_phase", "joint")),
                    "ppo_warmup_phase_iteration": int(getattr(ppo_result, "warmup_phase_iteration", 0)),
                    "ppo_actor_loss_weight": float(getattr(ppo_result, "actor_loss_weight", 1.0)),
                    "ppo_trust_region_rejected_count": int(
                        getattr(ppo_result, "trust_region_rejected_count", 0)
                    ),
                    "ppo_trust_region_accepted": int(getattr(ppo_result, "trust_region_accepted", 1)),
                    "ppo_trust_region_rollback_enabled": int(
                        getattr(ppo_result, "trust_region_rollback_enabled", 0)
                    ),
                    "ppo_trust_region_max_retries": int(
                        getattr(ppo_result, "trust_region_max_retries", 0)
                    ),
                    "ppo_trust_region_schedule": str(
                        getattr(ppo_result, "trust_region_schedule", "unknown")
                    ),
                    "ppo_trust_region_schedule_adaptive": int(
                        getattr(ppo_result, "trust_region_schedule_adaptive", 0)
                    ),
                    "ppo_adaptive_lr_before": float(getattr(ppo_result, "adaptive_lr_before", 0.0)),
                    "ppo_adaptive_lr_after": float(getattr(ppo_result, "adaptive_lr_after", 0.0)),
                    "ppo_adaptive_lr_kl_mean": float(getattr(ppo_result, "adaptive_lr_kl_mean", 0.0)),
                    "ppo_adaptive_lr_desired_kl": float(getattr(ppo_result, "adaptive_lr_desired_kl", 0.0)),
                    "ppo_mosaic_pre_step_adaptive_lr_before": float(
                        getattr(ppo_result, "mosaic_pre_step_adaptive_lr_before", 0.0)
                    ),
                    "ppo_mosaic_pre_step_adaptive_lr_after": float(
                        getattr(ppo_result, "mosaic_pre_step_adaptive_lr_after", 0.0)
                    ),
                    "ppo_mosaic_pre_step_adaptive_lr_kl_mean": float(
                        getattr(ppo_result, "mosaic_pre_step_adaptive_lr_kl_mean", 0.0)
                    ),
                    "ppo_segment_reject_adaptive_lr_after": float(
                        getattr(ppo_result, "segment_reject_adaptive_lr_after", 0.0)
                    ),
                    "ppo_post_update_mean_delta_l2_mean": float(
                        getattr(ppo_result, "post_update_mean_delta_l2_mean", 0.0)
                    ),
                    "ppo_post_update_mean_delta_max_abs": float(
                        getattr(ppo_result, "post_update_mean_delta_max_abs", 0.0)
                    ),
                    "ppo_post_update_old_sigma_min": float(
                        getattr(ppo_result, "post_update_old_sigma_min", 0.0)
                    ),
                    "ppo_post_update_sigma_min": float(
                        getattr(ppo_result, "post_update_sigma_min", 0.0)
                    ),
                    "ppo_post_update_raw_action_old_mean_l2_mean": float(
                        getattr(ppo_result, "post_update_raw_action_old_mean_l2_mean", 0.0)
                    ),
                    "ppo_post_update_raw_action_old_mean_abs_max": float(
                        getattr(ppo_result, "post_update_raw_action_old_mean_abs_max", 0.0)
                    ),
                    "ppo_post_update_raw_action_old_mean_abs_dim_mean": tuple(
                        getattr(ppo_result, "post_update_raw_action_old_mean_abs_dim_mean", ())
                    ),
                    "ppo_post_update_raw_action_old_mean_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_raw_action_old_mean_abs_dim_max", ())
                    ),
                    "ppo_post_update_old_sigma_dim_mean": tuple(
                        getattr(ppo_result, "post_update_old_sigma_dim_mean", ())
                    ),
                    "ppo_post_update_sigma_dim_mean": tuple(
                        getattr(ppo_result, "post_update_sigma_dim_mean", ())
                    ),
                    "ppo_post_update_distribution_mean_delta_dim_mean": tuple(
                        getattr(ppo_result, "post_update_distribution_mean_delta_dim_mean", ())
                    ),
                    "ppo_post_update_distribution_mean_delta_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_distribution_mean_delta_abs_dim_max", ())
                    ),
                    "ppo_post_update_log_ratio_contrib_dim_mean": tuple(
                        getattr(ppo_result, "post_update_log_ratio_contrib_dim_mean", ())
                    ),
                    "ppo_post_update_log_ratio_contrib_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_log_ratio_contrib_abs_dim_max", ())
                    ),
                    "ppo_post_update_log_jacobian_dim_mean": tuple(
                        getattr(ppo_result, "post_update_log_jacobian_dim_mean", ())
                    ),
                    "ppo_post_update_log_jacobian_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_log_jacobian_abs_dim_max", ())
                    ),
                }
            )
    # AUDIT-PERTURB-02..AUDIT-RETURN-01: 检查 perturb/obs/action/GMT/pair/Gain/return owner 边界.
    # Result: PENDING_LIVE.
    print_rollout_storage_audit(runner, capture=capture, summary=summary, storage_batch=storage_batch)
    _print_live_probe_summary(runner, capture, summary)
    return summary
