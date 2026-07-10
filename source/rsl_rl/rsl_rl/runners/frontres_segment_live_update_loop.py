from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

_LOG_SEPARATOR = "-" * 80
_LIVE_SAMPLER_PATH = Path(__file__).resolve().with_name("frontres_segment_live_sampler.py")
_LIVE_SAMPLER_SPEC = importlib.util.spec_from_file_location(
    "frontres_segment_live_sampler_update_loop_module",
    _LIVE_SAMPLER_PATH,
)
if _LIVE_SAMPLER_SPEC is None or _LIVE_SAMPLER_SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES Segment live sampler from {_LIVE_SAMPLER_PATH}.")
_LIVE_SAMPLER_MODULE = importlib.util.module_from_spec(_LIVE_SAMPLER_SPEC)
sys.modules[_LIVE_SAMPLER_SPEC.name] = _LIVE_SAMPLER_MODULE
_LIVE_SAMPLER_SPEC.loader.exec_module(_LIVE_SAMPLER_MODULE)
run_frontres_segment_sampler_step = _LIVE_SAMPLER_MODULE.run_frontres_segment_sampler_step


def _fmt_num(value: Any) -> str:
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    abs_value = abs(value)
    if abs_value != 0.0 and (abs_value >= 10000.0 or abs_value < 0.001):
        return f"{value:.3e}"
    return f"{value:.6f}"


def _fmt_pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _loop_status(total_loss: float, actor_loss: float, approx_kl: float, clip_frac: float) -> str:
    if not all(math.isfinite(v) for v in (total_loss, actor_loss, approx_kl, clip_frac)):
        return "BAD_NONFINITE"
    if abs(actor_loss) >= 1000.0 or abs(total_loss) >= 1000.0:
        return "BAD_LOSS_EXPLOSION"
    if clip_frac >= 0.3:
        return "WARN_HIGH_CLIP"
    if approx_kl < -0.001:
        return "WARN_NEG_KL"
    return "OK"


def _mean_optional_metric(metrics: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in metrics if key in item and math.isfinite(float(item[key]))]
    if not values:
        return float("nan")
    return sum(values) / float(len(values))


def _min_optional_metric(metrics: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in metrics if key in item and math.isfinite(float(item[key]))]
    if not values:
        return float("nan")
    return min(values)


def _min_valid_distribution_metric(metrics: list[dict[str, Any]], key: str) -> float:
    values = [
        float(item[key])
        for item in metrics
        if bool(item.get("ppo_update", False))
        and bool(item.get("ppo_distribution_kl_available", False))
        and key in item
        and math.isfinite(float(item[key]))
    ]
    if not values:
        return float("nan")
    return min(values)


def _should_print_update_loop_summary(runner: Any) -> bool:
    alg = getattr(runner, "alg", None)
    if bool(getattr(alg, "frontres_segment_verbose_probe", False)):
        return True
    count = int(getattr(runner, "_frontres_segment_live_update_loop_summary_count", 0)) + 1
    runner._frontres_segment_live_update_loop_summary_count = count
    warmup = max(0, int(getattr(alg, "frontres_segment_live_log_warmup", 3)))
    interval = max(1, int(getattr(alg, "frontres_segment_live_log_interval", 10)))
    return count <= warmup or count % interval == 0


def run_frontres_segment_live_update_loop(
    runner: Any,
    init_at_random_ep_len: bool = True,
    *,
    runner_learn: bool = False,
) -> dict[str, float | int]:
    boundary = runner._frontres_segment_replay_boundary
    if not (boundary.live_update_loop_only or boundary.live_train_enabled):
        raise ValueError(
            "FrontRES Segment live update loop requires frontres_segment_live_update_loop_only=True "
            "or frontres_segment_live_train_enabled=True."
        )
    update_steps = max(
        1,
        int(getattr(runner.alg, "frontres_segment_live_update_steps", boundary.live_update_steps)),
    )
    metrics = []
    for update_step in range(update_steps):
        metrics.append(
            run_frontres_segment_sampler_step(
                runner,
                init_at_random_ep_len=bool(init_at_random_ep_len and update_step == 0),
                update_step=update_step,
            )
        )
    update_count = sum(1 for item in metrics if bool(item["ppo_update"]))
    valid_count = sum(int(item["ppo_valid_count"]) for item in metrics)
    env_reward_mean = sum(float(item.get("env_reward_mean", item["reward_mean"])) for item in metrics) / float(update_steps)
    train_reward_mean = sum(float(item.get("train_reward_mean", item["reward_mean"])) for item in metrics) / float(update_steps)
    score_noisy_mean = sum(float(item.get("score_noisy_mean", 0.0)) for item in metrics) / float(update_steps)
    score_repaired_mean = sum(float(item.get("score_repaired_mean", 0.0)) for item in metrics) / float(update_steps)
    score_gain_mean = sum(float(item.get("score_gain_mean", 0.0)) for item in metrics) / float(update_steps)
    score_gain_pos_frac = sum(float(item.get("score_gain_pos_frac", 0.0)) for item in metrics) / float(update_steps)
    done_frac = sum(float(item.get("done_frac", 0.0)) for item in metrics) / float(update_steps)
    motion_delta_se_norm = sum(float(item.get("motion_delta_se_norm", 0.0)) for item in metrics) / float(update_steps)
    motion_delta_z_up_frac = sum(float(item.get("motion_delta_z_up_frac", 0.0)) for item in metrics) / float(update_steps)
    motion_mpjpe_repaired_clean = _mean_optional_metric(metrics, "segment/motion_mpjpe_repaired_clean")
    motion_mpjpe_noisy_clean = _mean_optional_metric(metrics, "segment/motion_mpjpe_noisy_clean")
    motion_vel_error_repaired_clean = _mean_optional_metric(metrics, "segment/motion_vel_error_repaired_clean")
    motion_acc_error_repaired_clean = _mean_optional_metric(metrics, "segment/motion_acc_error_repaired_clean")
    storage_valid_frac = sum(float(item["storage_valid_frac"]) for item in metrics) / float(update_steps)
    trial_policy_count = sum(int(item.get("trial_policy_count", item.get("ppo_boundary_policy_rows", 0))) for item in metrics)
    trial_search_count = sum(int(item.get("trial_search_count", item.get("ppo_boundary_search_rows", 0))) for item in metrics)
    ppo_boundary_evidence_rows = sum(
        int(item.get("ppo_boundary_evidence_rows", int(item.get("trial_policy_count", 0)) + int(item.get("trial_search_count", 0))))
        for item in metrics
    )
    ppo_boundary_search_evidence_only_rows = sum(
        int(item.get("ppo_boundary_search_evidence_only_rows", item.get("trial_search_count", 0))) for item in metrics
    )
    ppo_boundary_policy_invalid_rows = sum(
        int(item.get("ppo_boundary_policy_invalid_rows", max(0, int(item.get("trial_policy_count", 0)) - int(item.get("ppo_valid_count", 0)))))
        for item in metrics
    )
    if valid_count > 0 and trial_policy_count <= 0 and ppo_boundary_evidence_rows <= 0:
        trial_policy_count = valid_count
        ppo_boundary_evidence_rows = valid_count
    ppo_boundary_valid_policy_frac = float(valid_count / max(1, trial_policy_count))
    ppo_boundary_valid_evidence_frac = float(valid_count / max(1, ppo_boundary_evidence_rows))
    total_loss_mean = sum(float(item["ppo_total_loss"]) for item in metrics) / float(update_steps)
    actor_loss_mean = sum(float(item["ppo_actor_loss"]) for item in metrics) / float(update_steps)
    value_loss_mean = sum(float(item["ppo_value_loss"]) for item in metrics) / float(update_steps)
    approx_kl_mean = sum(float(item["ppo_approx_kl"]) for item in metrics) / float(update_steps)
    clip_frac_mean = sum(float(item["ppo_clip_frac"]) for item in metrics) / float(update_steps)
    trust_region_rejected_count = sum(int(item.get("ppo_trust_region_rejected_count", 0)) for item in metrics)
    trust_region_accepted_min = min(int(item.get("ppo_trust_region_accepted", 1)) for item in metrics)
    trust_region_rollback_enabled_min = min(
        int(item.get("ppo_trust_region_rollback_enabled", 0)) for item in metrics
    )
    trust_region_max_retries_max = max(int(item.get("ppo_trust_region_max_retries", 0)) for item in metrics)
    trust_region_schedule_set = sorted(
        {str(item.get("ppo_trust_region_schedule", "unknown")) for item in metrics}
    )
    trust_region_schedule = ",".join(trust_region_schedule_set)
    adaptive_lr_before_first = float(metrics[0].get("ppo_adaptive_lr_before", 0.0))
    adaptive_lr_after_last = float(metrics[-1].get("ppo_adaptive_lr_after", 0.0))
    adaptive_lr_desired_kl_mean = (
        sum(float(item.get("ppo_adaptive_lr_desired_kl", 0.0)) for item in metrics) / float(update_steps)
    )
    mosaic_pre_step_lr_before_first = float(metrics[0].get("ppo_mosaic_pre_step_adaptive_lr_before", 0.0))
    mosaic_pre_step_lr_after_last = float(metrics[-1].get("ppo_mosaic_pre_step_adaptive_lr_after", 0.0))
    mosaic_pre_step_lr_kl_mean = (
        sum(float(item.get("ppo_mosaic_pre_step_adaptive_lr_kl_mean", 0.0)) for item in metrics)
        / float(update_steps)
    )
    ppo_advantage_abs_top1_frac_mean = _mean_optional_metric(metrics, "ppo_advantage_abs_top1_frac")
    ppo_old_sigma_min = _min_valid_distribution_metric(metrics, "ppo_old_sigma_min")
    ppo_sigma_min = _min_valid_distribution_metric(metrics, "ppo_sigma_min")
    ppo_post_update_mean_delta_l2_mean = _mean_optional_metric(metrics, "ppo_post_update_mean_delta_l2_mean")
    ppo_post_update_mean_delta_max_abs = _mean_optional_metric(metrics, "ppo_post_update_mean_delta_max_abs")
    sampler_update_count = sum(1 for item in metrics if bool(item.get("sampler_update", False)))
    sampler_global_count = sum(int(item.get("sampler_source_global_count", 0)) for item in metrics)
    sampler_replay_count = sum(int(item.get("sampler_source_replay_count", 0)) for item in metrics)
    sampler_review_count = sum(int(item.get("sampler_source_review_count", 0)) for item in metrics)
    sampler_replay_pool_size = int(metrics[-1].get("sampler_replay_pool_size", 0))
    sampler_priority_mean = float(metrics[-1].get("sampler_priority_mean", 0.0))
    sampler_solved_frac = float(metrics[-1].get("sampler_solved_frac", 0.0))
    sampler_hopeless_frac = float(metrics[-1].get("sampler_hopeless_frac", 0.0))
    sampler_stale_review_count = int(metrics[-1].get("sampler_stale_review_count", 0))
    sampler_update_gain_mean = sum(float(item.get("sampler_update_gain_mean", 0.0)) for item in metrics) / float(update_steps)
    sampler_update_gain_pos_frac = sum(float(item.get("sampler_update_gain_pos_frac", 0.0)) for item in metrics) / float(update_steps)
    sampler_update_useful_mean = sum(float(item.get("sampler_update_useful_mean", 0.0)) for item in metrics) / float(update_steps)
    sampler_update_replay_candidate_count = sum(int(item.get("sampler_update_replay_candidate_count", 0)) for item in metrics)
    sampler_update_priority_before_mean = sum(
        float(item.get("sampler_update_priority_before_mean", 0.0)) for item in metrics
    ) / float(update_steps)
    sampler_update_priority_after_mean = sum(
        float(item.get("sampler_update_priority_after_mean", 0.0)) for item in metrics
    ) / float(update_steps)
    if _should_print_update_loop_summary(runner):
        print(
            "\n".join(
                (
                    "",
                    _LOG_SEPARATOR,
                    "",
                    "[FrontRES Segment Live Update Loop]",
                    "  route: "
                    f"objective={getattr(runner.alg, 'frontres_training_objective', 'n/a')} "
                    f"runner_learn={runner_learn}",
                    "  update: "
                    f"updates={update_count}/{update_steps} "
                    f"valid={valid_count} "
                    f"valid_frac={_fmt_pct(storage_valid_frac)} "
                    f"train_reward={_fmt_num(train_reward_mean)} "
                    f"env_reward={_fmt_num(env_reward_mean)} "
                    f"gain={_fmt_num(score_gain_mean)}",
                    "  trial: "
                    f"policy={trial_policy_count} "
                    f"search={trial_search_count} "
                    f"evidence={ppo_boundary_evidence_rows} "
                    f"ppo_valid={valid_count} "
                    f"search_evidence_only={ppo_boundary_search_evidence_only_rows} "
                    f"policy_invalid={ppo_boundary_policy_invalid_rows} "
                    f"valid_policy={_fmt_pct(ppo_boundary_valid_policy_frac)} "
                    f"valid_evidence={_fmt_pct(ppo_boundary_valid_evidence_frac)}",
                    "  ppo: "
                    f"loss_total={_fmt_num(total_loss_mean)} "
                    f"actor={_fmt_num(actor_loss_mean)} "
                    f"value={_fmt_num(value_loss_mean)} "
                    f"kl={_fmt_num(approx_kl_mean)} "
                    f"clip={_fmt_pct(clip_frac_mean)} "
                    "status="
                    f"{'WARN_TRUST_REGION_REJECTED' if trust_region_rejected_count > 0 else _loop_status(total_loss_mean, actor_loss_mean, approx_kl_mean, clip_frac_mean)}",
                    "  trust: "
                    f"accepted={trust_region_accepted_min} "
                    f"rejected={trust_region_rejected_count} "
                    f"lr_before={_fmt_num(adaptive_lr_before_first)} "
                    f"lr_after={_fmt_num(adaptive_lr_after_last)} "
                    f"desired_kl={_fmt_num(adaptive_lr_desired_kl_mean)} "
                    f"schedule={trust_region_schedule} "
                    f"rollback={bool(trust_region_rollback_enabled_min)} "
                    f"max_retries={trust_region_max_retries_max} "
                    f"pre_lr_before={_fmt_num(mosaic_pre_step_lr_before_first)} "
                    f"pre_lr_after={_fmt_num(mosaic_pre_step_lr_after_last)} "
                    f"pre_kl={_fmt_num(mosaic_pre_step_lr_kl_mean)}",
                    "  scale: "
                    f"adv_top1={_fmt_pct(ppo_advantage_abs_top1_frac_mean)} "
                    f"old_sigma_min={_fmt_num(ppo_old_sigma_min)} "
                    f"sigma_min={_fmt_num(ppo_sigma_min)} "
                    f"post_mean_delta_l2={_fmt_num(ppo_post_update_mean_delta_l2_mean)} "
                    f"post_mean_delta_max={_fmt_num(ppo_post_update_mean_delta_max_abs)}",
                    "  sampler: "
                    f"global={sampler_global_count} "
                    f"replay={sampler_replay_count} "
                    f"review={sampler_review_count} "
                    f"pool={sampler_replay_pool_size} "
                    f"priority={_fmt_num(sampler_priority_mean)} "
                    f"solved={_fmt_pct(sampler_solved_frac)} "
                    f"hopeless={_fmt_pct(sampler_hopeless_frac)} "
                    f"stale_review={sampler_stale_review_count}",
                    "  sampler_update: "
                    f"gain={_fmt_num(sampler_update_gain_mean)} "
                    f"gain_pos={_fmt_pct(sampler_update_gain_pos_frac)} "
                    f"useful={_fmt_num(sampler_update_useful_mean)} "
                    f"replay_candidates={sampler_update_replay_candidate_count} "
                    f"priority_before={_fmt_num(sampler_update_priority_before_mean)} "
                    f"priority_after={_fmt_num(sampler_update_priority_after_mean)}",
                    "",
                    _LOG_SEPARATOR,
                    "",
                )
            ),
            flush=True,
        )
    return {
        "update_steps": update_steps,
        "update_count": update_count,
        "ppo_valid_count": valid_count,
        "reward_mean": train_reward_mean,
        "train_reward_mean": train_reward_mean,
        "env_reward_mean": env_reward_mean,
        "score_noisy_mean": score_noisy_mean,
        "score_repaired_mean": score_repaired_mean,
        "score_gain_mean": score_gain_mean,
        "score_gain_pos_frac": score_gain_pos_frac,
        "done_frac": done_frac,
        "motion_delta_se_norm": motion_delta_se_norm,
        "motion_delta_z_up_frac": motion_delta_z_up_frac,
        "segment/motion_mpjpe_repaired_clean": motion_mpjpe_repaired_clean,
        "segment/motion_mpjpe_noisy_clean": motion_mpjpe_noisy_clean,
        "segment/motion_vel_error_repaired_clean": motion_vel_error_repaired_clean,
        "segment/motion_acc_error_repaired_clean": motion_acc_error_repaired_clean,
        "storage_valid_frac": storage_valid_frac,
        "trial_policy_count": trial_policy_count,
        "trial_search_count": trial_search_count,
        "ppo_boundary_evidence_rows": ppo_boundary_evidence_rows,
        "ppo_boundary_policy_rows": trial_policy_count,
        "ppo_boundary_search_rows": trial_search_count,
        "ppo_boundary_eligible_rows": valid_count,
        "ppo_boundary_search_evidence_only_rows": ppo_boundary_search_evidence_only_rows,
        "ppo_boundary_policy_invalid_rows": ppo_boundary_policy_invalid_rows,
        "ppo_boundary_valid_policy_frac": ppo_boundary_valid_policy_frac,
        "ppo_boundary_valid_evidence_frac": ppo_boundary_valid_evidence_frac,
        "ppo_total_loss_mean": total_loss_mean,
        "ppo_actor_loss_mean": actor_loss_mean,
        "ppo_value_loss_mean": value_loss_mean,
        "ppo_approx_kl_mean": approx_kl_mean,
        "ppo_clip_frac_mean": clip_frac_mean,
        "ppo_trust_region_rejected_count_sum": trust_region_rejected_count,
        "ppo_trust_region_accepted_min": trust_region_accepted_min,
        "ppo_trust_region_rollback_enabled_min": trust_region_rollback_enabled_min,
        "ppo_trust_region_max_retries_max": trust_region_max_retries_max,
        "ppo_trust_region_schedule": trust_region_schedule,
        "ppo_adaptive_lr_before_first": adaptive_lr_before_first,
        "ppo_adaptive_lr_after_last": adaptive_lr_after_last,
        "ppo_adaptive_lr_desired_kl_mean": adaptive_lr_desired_kl_mean,
        "ppo_mosaic_pre_step_adaptive_lr_before_first": mosaic_pre_step_lr_before_first,
        "ppo_mosaic_pre_step_adaptive_lr_after_last": mosaic_pre_step_lr_after_last,
        "ppo_mosaic_pre_step_adaptive_lr_kl_mean": mosaic_pre_step_lr_kl_mean,
        "ppo_advantage_abs_top1_frac_mean": ppo_advantage_abs_top1_frac_mean,
        "ppo_old_sigma_min": ppo_old_sigma_min,
        "ppo_sigma_min": ppo_sigma_min,
        "ppo_post_update_mean_delta_l2_mean": ppo_post_update_mean_delta_l2_mean,
        "ppo_post_update_mean_delta_max_abs": ppo_post_update_mean_delta_max_abs,
        "sampler_update_count": sampler_update_count,
        "sampler_global_count": sampler_global_count,
        "sampler_replay_count": sampler_replay_count,
        "sampler_review_count": sampler_review_count,
        "sampler_replay_pool_size": sampler_replay_pool_size,
        "sampler_priority_mean": sampler_priority_mean,
        "sampler_solved_frac": sampler_solved_frac,
        "sampler_hopeless_frac": sampler_hopeless_frac,
        "sampler_stale_review_count": sampler_stale_review_count,
        "sampler_update_gain_mean": sampler_update_gain_mean,
        "sampler_update_gain_pos_frac": sampler_update_gain_pos_frac,
        "sampler_update_useful_mean": sampler_update_useful_mean,
        "sampler_update_replay_candidate_count": sampler_update_replay_candidate_count,
        "sampler_update_priority_before_mean": sampler_update_priority_before_mean,
        "sampler_update_priority_after_mean": sampler_update_priority_after_mean,
    }
