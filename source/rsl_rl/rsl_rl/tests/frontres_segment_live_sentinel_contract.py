#!/usr/bin/env python3
"""Stage 3 live sentinel contract.

This test proves that the real Stage 3 startup boundary can be entered only in
sentinel/probe modes, while storage writes and PPO/update training remain
explicitly disabled.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl"
if str(RSL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_boundary_live_sentinel_log() -> None:
    module = _load(
        "frontres_segment_runner_boundary",
        RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_runner_boundary.py",
    )
    boundary = module.FrontRESSegmentRunnerBoundary.from_train_cfg(
        {
            "algorithm": {
                "frontres_training_objective": "segment_replay_hrl",
                "frontres_segment_replay_enabled": True,
                "frontres_segment_live_runner_enabled": True,
                "frontres_segment_live_sentinel_only": True,
                "frontres_segment_live_probe_only": False,
                "frontres_segment_live_storage_write_only": False,
                "frontres_segment_live_single_update_only": False,
                "frontres_segment_live_update_loop_only": False,
                "frontres_segment_live_train_enabled": False,
                "frontres_segment_live_update_steps": 4,
                "frontres_segment_k": 4,
                "frontres_segment_reset_mode": "auto",
            }
        }
    )
    boundary.assert_live_runner_ready()
    log = boundary.sentinel_log()
    assert log is not None
    required = [
        "FrontRES Segment Live Sentinel",
        "objective=segment_replay_hrl",
        "segment_k=4",
        "reset_mode=auto",
        "live_runner=True",
        "sentinel_only=True",
        "storage=independent",
        "ppo_action=delta_se3_6d",
        "training_update=disabled",
    ]
    for needle in required:
        assert needle in log, needle


def test_boundary_live_probe_log() -> None:
    module = _load(
        "frontres_segment_runner_boundary_probe",
        RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_runner_boundary.py",
    )
    boundary = module.FrontRESSegmentRunnerBoundary.from_train_cfg(
        {
            "algorithm": {
                "frontres_training_objective": "segment_replay_hrl",
                "frontres_segment_replay_enabled": True,
                "frontres_segment_live_runner_enabled": True,
                "frontres_segment_live_sentinel_only": False,
                "frontres_segment_live_probe_only": True,
                "frontres_segment_live_storage_write_only": False,
                "frontres_segment_live_single_update_only": False,
                "frontres_segment_live_update_loop_only": False,
                "frontres_segment_live_train_enabled": False,
                "frontres_segment_live_update_steps": 4,
                "frontres_segment_k": 4,
                "frontres_segment_reset_mode": "auto",
            }
        }
    )
    boundary.assert_live_runner_ready()
    log = boundary.probe_log()
    assert log is not None
    required = [
        "FrontRES Segment Live Probe Ready",
        "objective=segment_replay_hrl",
        "segment_k=4",
        "reset_mode=auto",
        "live_runner=True",
        "probe_only=True",
        "storage_write=False",
        "ppo_update=False",
    ]
    for needle in required:
        assert needle in log, needle


def test_boundary_live_storage_probe_log() -> None:
    module = _load(
        "frontres_segment_runner_boundary_storage_probe",
        RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_runner_boundary.py",
    )
    boundary = module.FrontRESSegmentRunnerBoundary.from_train_cfg(
        {
            "algorithm": {
                "frontres_training_objective": "segment_replay_hrl",
                "frontres_segment_replay_enabled": True,
                "frontres_segment_live_runner_enabled": True,
                "frontres_segment_live_sentinel_only": False,
                "frontres_segment_live_probe_only": False,
                "frontres_segment_live_storage_write_only": True,
                "frontres_segment_live_single_update_only": False,
                "frontres_segment_live_update_loop_only": False,
                "frontres_segment_live_train_enabled": False,
                "frontres_segment_live_update_steps": 4,
                "frontres_segment_k": 4,
                "frontres_segment_reset_mode": "auto",
            }
        }
    )
    boundary.assert_live_runner_ready()
    log = boundary.probe_log()
    assert log is not None
    required = [
        "FrontRES Segment Live Probe Ready",
        "objective=segment_replay_hrl",
        "segment_k=4",
        "reset_mode=auto",
        "live_runner=True",
        "probe_only=True",
        "storage_write=True",
        "ppo_update=False",
    ]
    for needle in required:
        assert needle in log, needle


def test_boundary_live_single_update_probe_log() -> None:
    module = _load(
        "frontres_segment_runner_boundary_single_update_probe",
        RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_runner_boundary.py",
    )
    boundary = module.FrontRESSegmentRunnerBoundary.from_train_cfg(
        {
            "algorithm": {
                "frontres_training_objective": "segment_replay_hrl",
                "frontres_segment_replay_enabled": True,
                "frontres_segment_live_runner_enabled": True,
                "frontres_segment_live_sentinel_only": False,
                "frontres_segment_live_probe_only": False,
                "frontres_segment_live_storage_write_only": False,
                "frontres_segment_live_single_update_only": True,
                "frontres_segment_live_update_loop_only": False,
                "frontres_segment_live_train_enabled": False,
                "frontres_segment_live_update_steps": 4,
                "frontres_segment_k": 4,
                "frontres_segment_reset_mode": "auto",
            }
        }
    )
    boundary.assert_live_runner_ready()
    log = boundary.probe_log()
    assert log is not None
    required = [
        "FrontRES Segment Live Probe Ready",
        "objective=segment_replay_hrl",
        "segment_k=4",
        "reset_mode=auto",
        "live_runner=True",
        "probe_only=True",
        "storage_write=True",
        "ppo_update=True",
    ]
    for needle in required:
        assert needle in log, needle


def test_boundary_live_update_loop_probe_log() -> None:
    module = _load(
        "frontres_segment_runner_boundary_update_loop_probe",
        RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_runner_boundary.py",
    )
    boundary = module.FrontRESSegmentRunnerBoundary.from_train_cfg(
        {
            "algorithm": {
                "frontres_training_objective": "segment_replay_hrl",
                "frontres_segment_replay_enabled": True,
                "frontres_segment_live_runner_enabled": True,
                "frontres_segment_live_sentinel_only": False,
                "frontres_segment_live_probe_only": False,
                "frontres_segment_live_storage_write_only": False,
                "frontres_segment_live_single_update_only": False,
                "frontres_segment_live_update_loop_only": True,
                "frontres_segment_live_train_enabled": False,
                "frontres_segment_live_update_steps": 4,
                "frontres_segment_k": 4,
                "frontres_segment_reset_mode": "auto",
            }
        }
    )
    boundary.assert_live_runner_ready()
    log = boundary.probe_log()
    assert log is not None
    required = [
        "FrontRES Segment Live Probe Ready",
        "objective=segment_replay_hrl",
        "segment_k=4",
        "update_steps=4",
        "reset_mode=auto",
        "live_runner=True",
        "probe_only=True",
        "storage_write=True",
        "ppo_update=True",
    ]
    for needle in required:
        assert needle in log, needle


def test_boundary_live_train_log() -> None:
    module = _load(
        "frontres_segment_runner_boundary_train",
        RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_runner_boundary.py",
    )
    boundary = module.FrontRESSegmentRunnerBoundary.from_train_cfg(
        {
            "algorithm": {
                "frontres_training_objective": "segment_replay_hrl",
                "frontres_segment_replay_enabled": True,
                "frontres_segment_live_runner_enabled": True,
                "frontres_segment_live_sentinel_only": False,
                "frontres_segment_live_probe_only": False,
                "frontres_segment_live_storage_write_only": False,
                "frontres_segment_live_single_update_only": False,
                "frontres_segment_live_update_loop_only": False,
                "frontres_segment_live_train_enabled": True,
                "frontres_segment_live_update_steps": 4,
                "frontres_segment_k": 4,
                "frontres_segment_reset_mode": "auto",
            }
        }
    )
    boundary.assert_live_runner_ready()
    log = boundary.train_log()
    assert log is not None
    required = [
        "FrontRES Segment Live Train Ready",
        "objective=segment_replay_hrl",
        "segment_k=4",
        "update_steps=4",
        "reset_mode=auto",
        "live_runner=True",
        "runner_learn=True",
        "storage=independent",
        "ppo_action=delta_se3_6d",
    ]
    for needle in required:
        assert needle in log, needle


def test_live_sentinel_is_not_training_mode() -> None:
    train = (ROOT / "scripts" / "rsl_rl" / "train.py").read_text()
    runner = (RSL_ROOT / "rsl_rl" / "runners" / "on_policy_runner.py").read_text()
    live_probe = (RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_live_probe.py").read_text()
    live_update_loop = (RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_live_update_loop.py").read_text()
    live_training = (RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_live_training.py").read_text()
    algorithm = (RSL_ROOT / "rsl_rl" / "algorithms" / "frontres_unified.py").read_text()
    assert '"--frontres_segment_live_sentinel_only"' in train
    assert '"--frontres_segment_live_probe_only"' in train
    assert '"--frontres_segment_live_storage_write_only"' in train
    assert '"--frontres_segment_live_single_update_only"' in train
    assert '"--frontres_segment_live_update_loop_only"' in train
    assert '"--frontres_segment_live_update_steps"' in train
    assert "agent_cfg.max_iterations = 0" in train
    assert "live_sentinel_only or live_probe_only or live_storage_only or live_single_update_only or live_update_loop_only" in train
    assert "live_train_enabled = not (" in train
    assert '_set_if_present(alg_cfg, "frontres_segment_live_sentinel_only", live_sentinel_only)' in train
    assert '_set_if_present(alg_cfg, "frontres_segment_live_probe_only", live_probe_only)' in train
    assert '_set_if_present(alg_cfg, "frontres_segment_live_storage_write_only", live_storage_only)' in train
    assert '_set_if_present(alg_cfg, "frontres_segment_live_single_update_only", live_single_update_only)' in train
    assert '_set_if_present(alg_cfg, "frontres_segment_live_update_loop_only", live_update_loop_only)' in train
    assert '_set_if_present(alg_cfg, "frontres_segment_live_train_enabled", live_train_enabled)' in train
    assert '_set_if_present(alg_cfg, "frontres_segment_live_update_steps", live_update_steps)' in train
    assert "sentinel_log()" in runner
    assert "probe_log()" in runner
    assert "train_log()" in runner
    assert "run_frontres_segment_live_probe" in runner
    assert "run_frontres_segment_live_probe_helper(self" in runner
    assert "run_frontres_segment_live_update_loop" in runner
    assert "run_frontres_segment_live_update_loop_helper(" in runner
    assert "learn_frontres_segment_live" in runner
    assert "run_frontres_segment_live_training_loop" in runner
    assert "_run_frontres_segment_single_update" in runner
    assert "FrontRESSegmentRolloutStorage" not in runner
    assert "FrontRESSegmentTransition" not in runner
    assert "compute_frontres_segment_ppo_loss" not in runner
    assert "FrontRESSegmentPPOConfig" not in runner
    assert '"  storage: "' in live_probe
    assert "write={bool(summary['storage_write'])}" in live_probe
    assert "from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout" in live_probe
    assert "from rsl_rl.frontres.training_schedule import configure_frontres_pair_layout" not in live_probe
    assert "FrontRESSegmentRolloutStorage" in live_probe
    assert "FrontRESSegmentTransition" in live_probe
    assert "compute_frontres_segment_ppo_loss" in live_probe
    assert "FrontRESSegmentPPOConfig" in live_probe
    assert "build_live_segment_storage" in live_probe
    assert "run_frontres_segment_single_update" in live_probe
    assert "_run_live_rollout_capture" in live_probe
    assert "size={int(summary['storage_size'])}" in live_probe
    assert "valid_frac={_fmt_pct(summary['storage_valid_frac'])}" in live_probe
    assert "update={bool(summary['ppo_update'])}" in live_probe
    assert "FrontRES Segment Live Update Loop" in live_update_loop
    assert "runner_learn={runner_learn}" in live_update_loop
    assert "FrontRES Segment Live Train" in live_training
    assert "runner_learn=True" in live_training
    assert "FrontRES Segment live update summary missing keys" in live_training
    assert "FrontRES Segment live update produced update_count=0" in live_training
    assert "FrontRES Segment live update produced non-finite" in live_training
    assert "too few valid PPO samples" in live_training
    assert "Stage 3 Segment Replay live mode reached FrontRESUnified.update" in algorithm
    assert "runner will execute exactly one PPO optimizer step and exit" in algorithm
    assert "PPO optimizer steps and exit" in algorithm
    assert "PPO optimizer steps per iteration" in algorithm


def main() -> None:
    test_boundary_live_sentinel_log()
    test_boundary_live_probe_log()
    test_boundary_live_storage_probe_log()
    test_boundary_live_single_update_probe_log()
    test_boundary_live_update_loop_probe_log()
    test_boundary_live_train_log()
    test_live_sentinel_is_not_training_mode()
    print("result: PASS")


if __name__ == "__main__":
    main()
