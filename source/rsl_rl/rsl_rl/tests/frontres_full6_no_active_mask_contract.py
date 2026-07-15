#!/usr/bin/env python3
"""Static contract for the current full-6D FrontRES method boundary."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> None:
    rollout = _read("source/rsl_rl/rsl_rl/runners/frontres_rollout_step.py")
    runner = _read("source/rsl_rl/rsl_rl/runners/on_policy_runner.py")
    live_probe = _read("source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py")
    storage = _read("source/rsl_rl/rsl_rl/storage/rollout_storage.py")
    ppo = _read("source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py")
    algorithm = _read("source/rsl_rl/rsl_rl/algorithms/frontres_unified.py")
    warmup = _read("source/rsl_rl/rsl_rl/runners/frontres_warmup.py")
    task_correction = _read("source/rsl_rl/rsl_rl/frontres/task_space_correction.py")
    runner_package = _read("source/rsl_rl/rsl_rl/frontres/__init__.py")
    dr_sweep = _read("source/rsl_rl/rsl_rl/runners/frontres_dr_sweep_eval.py")

    assert "_mask_frontres_task_actions" not in rollout
    assert "build_frontres_task_action_mask" not in runner
    assert "frontres_active_task_dims" not in runner
    assert "_live_segment_execution_action_mask" not in live_probe
    assert "action_mask" not in storage
    assert "action_mask" not in ppo
    assert "frontres_active_task_dims" not in warmup
    assert "temporal_reference_cache" not in task_correction
    assert "_frontres_temporal_ref_cache" not in task_correction
    assert "temporal_reference_cache" not in runner_package
    assert "state_router_alpha" not in dr_sweep
    assert "frontres_executable_floor" not in runner_package
    assert "frontres_actor_gate" not in storage
    assert "frontres_actor_gate_batch" not in algorithm
    assert "frontres_executable_floor" not in _read("source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py")
    assert "frontres_executable_floor" not in _read(
        "source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py"
    )
    print("frontres_full6_no_active_mask_contract: ok")


if __name__ == "__main__":
    main()
