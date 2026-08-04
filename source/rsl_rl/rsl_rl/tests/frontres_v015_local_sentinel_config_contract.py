#!/usr/bin/env python3
"""Deterministic S2 contract for the explicit v015 pre-live sentinel entrypoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
BOUNDARY_PATH = RSL_ROOT / "runners" / "frontres_segment_runner_boundary.py"
TRAIN_PATH = ROOT / "scripts" / "rsl_rl" / "train.py"
RUNNER_PATH = RSL_ROOT / "runners" / "on_policy_runner.py"
ALG_PATH = RSL_ROOT / "algorithms" / "frontres_unified.py"
RSL_CFG_PATH = RSL_ROOT / "modules" / "rsl_rl_cfg.py"
TASK_CFG_PATH = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "rsl_rl_cfg.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _v015_cfg(**overrides):
    cfg = {
        "frontres_training_objective": "segment_replay_hrl",
        "frontres_segment_replay_enabled": True,
        "frontres_segment_live_runner_enabled": True,
        "frontres_local_sentinel_only": True,
        "frontres_formal_transaction_enabled": True,
        "frontres_future_offsets": (1, 2),
        "frontres_future_intent_layout_version": "frontres-v015-future-intent-q29-v1",
        "frontres_segment_k": 8,
        "frontres_segment_max_horizon_k": 8,
    }
    cfg.update(overrides)
    return {"algorithm": cfg}


def test_t_v015_boundary_is_explicit_and_legacy_exclusive() -> None:
    boundary_module = _load("frontres_v015_local_sentinel_boundary", BOUNDARY_PATH)
    boundary = boundary_module.FrontRESSegmentRunnerBoundary.from_train_cfg(_v015_cfg())

    assert boundary.local_sentinel_only
    boundary.assert_live_runner_ready()
    assert boundary.sentinel_log() is None
    log = boundary.local_sentinel_log()
    assert log is not None
    assert "frontres_local_sentinel=True" in log
    assert "future_offsets=(1, 2)" in log

    mixed = boundary_module.FrontRESSegmentRunnerBoundary.from_train_cfg(
        _v015_cfg(frontres_segment_live_sentinel_only=True)
    )
    try:
        mixed.assert_live_runner_ready()
    except ValueError as exc:
        assert "legacy" in str(exc).lower()
    else:
        raise AssertionError("v015 sentinel unexpectedly coexisted with the legacy sentinel mode")
    print("[T-config/T-legacy-isolation] explicit v015 sentinel accepts q29 H and rejects legacy sentinel mixing", flush=True)


def test_t_entrypoint_and_algorithm_route_are_dedicated() -> None:
    train = TRAIN_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    algorithm = ALG_PATH.read_text(encoding="utf-8")
    rsl_cfg = RSL_CFG_PATH.read_text(encoding="utf-8")
    task_cfg = TASK_CFG_PATH.read_text(encoding="utf-8")

    assert "--frontres_local_sentinel_only" in train
    assert "--frontres_v015_future_offsets" in train
    assert "runner.run_frontres_local_identity_sentinel" in train
    assert "runner.finalize_frontres_local_sentinel_checkpoint" in train
    assert "def run_frontres_local_identity_sentinel(" in runner
    assert "def finalize_frontres_local_sentinel_checkpoint(" in runner
    assert "frontres_local_sentinel_only: bool = False" in algorithm
    assert "frontres_local_sentinel_only: bool = False" in rsl_cfg
    assert "frontres_local_sentinel_only: bool = False" in task_cfg
    method_start = runner.index("def run_frontres_local_identity_sentinel(")
    method_end = runner.index("\n    def ", method_start + 1)
    assert "run_frontres_segment_single_update" not in runner[method_start:method_end]
    print("[T-entrypoint/T-no-legacy-update] train dispatches only the dedicated v015 sentinel owner", flush=True)


def main() -> None:
    test_t_v015_boundary_is_explicit_and_legacy_exclusive()
    test_t_entrypoint_and_algorithm_route_are_dedicated()
    print("frontres_v015_local_sentinel_config_contract: ok", flush=True)


if __name__ == "__main__":
    main()
