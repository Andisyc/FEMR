from __future__ import annotations

import os
import math
from collections.abc import Mapping
from typing import Any


_REQUIRED_SUMMARY_KEYS = (
    "update_steps",
    "update_count",
    "ppo_valid_count",
    "reward_mean",
    "storage_valid_frac",
    "ppo_total_loss_mean",
    "ppo_actor_loss_mean",
    "ppo_value_loss_mean",
    "ppo_approx_kl_mean",
    "ppo_clip_frac_mean",
)

_FINITE_SUMMARY_KEYS = (
    "reward_mean",
    "storage_valid_frac",
    "ppo_total_loss_mean",
    "ppo_actor_loss_mean",
    "ppo_value_loss_mean",
    "ppo_approx_kl_mean",
    "ppo_clip_frac_mean",
)


def _validate_live_update_summary(summary: Mapping[str, Any]) -> None:
    missing = [key for key in _REQUIRED_SUMMARY_KEYS if key not in summary]
    if missing:
        raise KeyError(f"FrontRES Segment live update summary missing keys: {missing}")


def _read_live_guard_cfg(runner: Any) -> tuple[bool, int, bool]:
    alg = getattr(runner, "alg", None)
    fail_on_invalid = bool(getattr(alg, "frontres_segment_live_fail_on_invalid_update", True))
    min_valid_count = max(0, int(getattr(alg, "frontres_segment_live_min_valid_count", 1)))
    fail_on_nonfinite = bool(getattr(alg, "frontres_segment_live_fail_on_nonfinite", True))
    return fail_on_invalid, min_valid_count, fail_on_nonfinite


def _validate_live_update_values(
    summary: Mapping[str, Any],
    *,
    fail_on_invalid: bool,
    min_valid_count: int,
    fail_on_nonfinite: bool,
) -> None:
    if fail_on_nonfinite:
        for key in _FINITE_SUMMARY_KEYS:
            value = float(summary[key])
            if not math.isfinite(value):
                raise FloatingPointError(f"FrontRES Segment live update produced non-finite {key}: {value}")
    if not fail_on_invalid:
        return
    update_count = int(summary["update_count"])
    valid_count = int(summary["ppo_valid_count"])
    if update_count <= 0:
        raise RuntimeError("FrontRES Segment live update produced update_count=0.")
    if valid_count < min_valid_count:
        raise RuntimeError(
            "FrontRES Segment live update has too few valid PPO samples: "
            f"ppo_valid_count={valid_count}, min_valid_count={min_valid_count}."
        )


def _path_inside_log_dir(path: str, log_dir: str | None) -> bool:
    if log_dir is None:
        return False
    abs_path = os.path.abspath(path)
    abs_log_dir = os.path.abspath(log_dir)
    try:
        return os.path.commonpath([abs_path, abs_log_dir]) == abs_log_dir
    except ValueError:
        return False


def _print_checkpoint_save_probe(runner: Any, checkpoint_path: str) -> None:
    print(
        "[FrontRES Segment Live Checkpoint] "
        f"saved_checkpoint_path={checkpoint_path} "
        f"in_log_dir={_path_inside_log_dir(checkpoint_path, runner.log_dir)} "
        f"iteration={int(getattr(runner, 'current_learning_iteration', 0))} "
        "runner_learn=True",
        flush=True,
    )


def _print_resume_probe(runner: Any) -> None:
    loaded_checkpoint_path = getattr(runner, "_frontres_last_loaded_checkpoint_path", None)
    if loaded_checkpoint_path is None:
        return
    print(
        "[FrontRES Segment Live Resume] "
        f"loaded_checkpoint_path={loaded_checkpoint_path} "
        f"resumed_iteration={int(getattr(runner, 'current_learning_iteration', 0))} "
        "runner_learn=True "
        "legacy_runner_learn=False",
        flush=True,
    )


def run_frontres_segment_live_training_loop(
    runner: Any,
    *,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = True,
) -> None:
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    if not bool(getattr(boundary, "live_train_enabled", False)):
        raise ValueError("FrontRES Segment live training requires frontres_segment_live_train_enabled=True.")

    num_learning_iterations = max(0, int(num_learning_iterations))
    if num_learning_iterations == 0:
        print(
            "[FrontRES Segment Live Train] "
            "num_learning_iterations=0 update_count=0 runner_learn=True",
            flush=True,
        )
        return

    _print_resume_probe(runner)
    last_checkpoint_probe_path: str | None = None

    for local_iteration in range(num_learning_iterations):
        summary = runner.run_frontres_segment_live_update_loop(
            init_at_random_ep_len=bool(init_at_random_ep_len and local_iteration == 0),
            runner_learn=True,
        )
        _validate_live_update_summary(summary)
        fail_on_invalid, min_valid_count, fail_on_nonfinite = _read_live_guard_cfg(runner)
        _validate_live_update_values(
            summary,
            fail_on_invalid=fail_on_invalid,
            min_valid_count=min_valid_count,
            fail_on_nonfinite=fail_on_nonfinite,
        )
        runner.current_learning_iteration += 1
        print(
            "[FrontRES Segment Live Train] "
            f"iteration={runner.current_learning_iteration} "
            f"num_learning_iterations={num_learning_iterations} "
            f"update_steps={int(summary['update_steps'])} "
            f"update_count={int(summary['update_count'])} "
            f"ppo_valid_count={int(summary['ppo_valid_count'])} "
            f"reward_mean={float(summary['reward_mean']):.6f} "
            f"storage_valid_frac={float(summary['storage_valid_frac']):.4f} "
            f"ppo_total_loss_mean={float(summary['ppo_total_loss_mean']):.6f} "
            f"ppo_actor_loss_mean={float(summary['ppo_actor_loss_mean']):.6f} "
            f"ppo_value_loss_mean={float(summary['ppo_value_loss_mean']):.6f} "
            f"ppo_approx_kl_mean={float(summary['ppo_approx_kl_mean']):.6f} "
            f"ppo_clip_frac_mean={float(summary['ppo_clip_frac_mean']):.6f} "
            "runner_learn=True",
            flush=True,
        )
        if (
            runner.log_dir is not None
            and not runner.disable_logs
            and runner.save_interval > 0
            and runner.current_learning_iteration % runner.save_interval == 0
        ):
            checkpoint_path = os.path.join(runner.log_dir, f"model_{runner.current_learning_iteration}.pt")
            runner.save(checkpoint_path)
            _print_checkpoint_save_probe(runner, checkpoint_path)
            last_checkpoint_probe_path = checkpoint_path
            runner._record_frontres_checkpoint_probe(dict(summary), checkpoint_path)

    if runner.log_dir is not None and not runner.disable_logs:
        final_checkpoint_path = os.path.join(runner.log_dir, f"model_{runner.current_learning_iteration}.pt")
        runner.save(final_checkpoint_path)
        if final_checkpoint_path != last_checkpoint_probe_path:
            _print_checkpoint_save_probe(runner, final_checkpoint_path)
