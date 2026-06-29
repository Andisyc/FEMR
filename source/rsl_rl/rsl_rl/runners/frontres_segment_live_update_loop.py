from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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
    reward_mean = sum(float(item["reward_mean"]) for item in metrics) / float(update_steps)
    storage_valid_frac = sum(float(item["storage_valid_frac"]) for item in metrics) / float(update_steps)
    total_loss_mean = sum(float(item["ppo_total_loss"]) for item in metrics) / float(update_steps)
    actor_loss_mean = sum(float(item["ppo_actor_loss"]) for item in metrics) / float(update_steps)
    value_loss_mean = sum(float(item["ppo_value_loss"]) for item in metrics) / float(update_steps)
    approx_kl_mean = sum(float(item["ppo_approx_kl"]) for item in metrics) / float(update_steps)
    clip_frac_mean = sum(float(item["ppo_clip_frac"]) for item in metrics) / float(update_steps)
    sampler_update_count = sum(1 for item in metrics if bool(item.get("sampler_update", False)))
    sampler_global_count = sum(int(item.get("sampler_source_global_count", 0)) for item in metrics)
    sampler_replay_count = sum(int(item.get("sampler_source_replay_count", 0)) for item in metrics)
    sampler_review_count = sum(int(item.get("sampler_source_review_count", 0)) for item in metrics)
    sampler_replay_pool_size = int(metrics[-1].get("sampler_replay_pool_size", 0))
    sampler_priority_mean = float(metrics[-1].get("sampler_priority_mean", 0.0))
    sampler_solved_frac = float(metrics[-1].get("sampler_solved_frac", 0.0))
    sampler_hopeless_frac = float(metrics[-1].get("sampler_hopeless_frac", 0.0))
    sampler_stale_review_count = int(metrics[-1].get("sampler_stale_review_count", 0))
    print(
        "[FrontRES Segment Live Update Loop] "
        f"objective={getattr(runner.alg, 'frontres_training_objective', 'n/a')} "
        f"update_steps={update_steps} "
        f"update_count={update_count} "
        f"ppo_valid_count={valid_count} "
        f"reward_mean={reward_mean:.6f} "
        f"storage_valid_frac={storage_valid_frac:.4f} "
        f"ppo_total_loss_mean={total_loss_mean:.6f} "
        f"ppo_actor_loss_mean={actor_loss_mean:.6f} "
        f"ppo_value_loss_mean={value_loss_mean:.6f} "
        f"ppo_approx_kl_mean={approx_kl_mean:.6f} "
        f"ppo_clip_frac_mean={clip_frac_mean:.6f} "
        f"sampler_update_count={sampler_update_count} "
        f"sampler_global_count={sampler_global_count} "
        f"sampler_replay_count={sampler_replay_count} "
        f"sampler_review_count={sampler_review_count} "
        f"sampler_replay_pool_size={sampler_replay_pool_size} "
        f"sampler_priority_mean={sampler_priority_mean:.6f} "
        f"sampler_solved_frac={sampler_solved_frac:.4f} "
        f"sampler_hopeless_frac={sampler_hopeless_frac:.4f} "
        f"sampler_stale_review_count={sampler_stale_review_count} "
        f"runner_learn={runner_learn}",
        flush=True,
    )
    return {
        "update_steps": update_steps,
        "update_count": update_count,
        "ppo_valid_count": valid_count,
        "reward_mean": reward_mean,
        "storage_valid_frac": storage_valid_frac,
        "ppo_total_loss_mean": total_loss_mean,
        "ppo_actor_loss_mean": actor_loss_mean,
        "ppo_value_loss_mean": value_loss_mean,
        "ppo_approx_kl_mean": approx_kl_mean,
        "ppo_clip_frac_mean": clip_frac_mean,
        "sampler_update_count": sampler_update_count,
        "sampler_global_count": sampler_global_count,
        "sampler_replay_count": sampler_replay_count,
        "sampler_review_count": sampler_review_count,
        "sampler_replay_pool_size": sampler_replay_pool_size,
        "sampler_priority_mean": sampler_priority_mean,
        "sampler_solved_frac": sampler_solved_frac,
        "sampler_hopeless_frac": sampler_hopeless_frac,
        "sampler_stale_review_count": sampler_stale_review_count,
    }
