#!/usr/bin/env python3
"""Deterministic contract for the three independent Evaluation capabilities."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
DIAGNOSTICS_PATH = RSL_ROOT / "frontres" / "frontres_segment_diagnostics.py"
RUNNERS_ROOT = RSL_ROOT / "runners"
sys.path.insert(0, str(ROOT / "source" / "rsl_rl"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_value_error(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_t_three_capabilities_and_training_isolation() -> None:
    diagnostics = _load(
        "rsl_rl.frontres.frontres_segment_diagnostics",
        DIAGNOSTICS_PATH,
    )
    protocol = diagnostics.build_frontres_v015_composition_evaluation_protocol(
        reference_stream_id="demo/deployment_motion.npz",
        frame_count=8,
        femr_action_count=8,
    )
    protocol.validate()
    text = diagnostics.format_frontres_v015_composition_evaluation_protocol(protocol)
    assert protocol.evaluation_kind == "deployment_composition_protocol"
    assert protocol.return_feedback is False
    assert protocol.priority_feedback is False
    assert protocol.ppo_feedback is False
    assert "deployment_reference_stream" in text
    assert "local_return_feedback=0 replay_priority_feedback=0 ppo_feedback=0" in text
    _expect_value_error(lambda: replace(protocol, priority_feedback=True).validate())

    assert not (RUNNERS_ROOT / "frontres_segment_training_evaluation.py").exists()
    assert not (RUNNERS_ROOT / "frontres_segment_legacy_sequence_eval.py").exists()
    runner_source = (RUNNERS_ROOT / "on_policy_runner.py").read_text(encoding="utf-8")
    live_training_source = (RUNNERS_ROOT / "frontres_segment_live_training.py").read_text(
        encoding="utf-8"
    )
    train_source = (ROOT / "scripts" / "rsl_rl" / "train.py").read_text(encoding="utf-8")

    for retired in (
        "run_frontres_segment_periodic_eval",
        "run_frontres_segment_offline_eval",
        "run_frontres_segment_sequence_offline_eval",
        "frontres_segment_periodic_eval_enabled",
        "frontres_segment_offline_eval_only",
        "frontres_segment_sequence_offline_eval_only",
    ):
        assert retired not in runner_source
        assert retired not in train_source
        assert retired not in live_training_source

    for active in (
        "run_frontres_policy_quality_eval",
        "run_frontres_v015_deployment_composition_eval",
        "evaluate_frontres_dr_sweep",
    ):
        assert active in runner_source

    quality_source = (RUNNERS_ROOT / "frontres_policy_quality_eval.py").read_text(encoding="utf-8")
    composition_source = (RUNNERS_ROOT / "frontres_segment_sequence_eval.py").read_text(
        encoding="utf-8"
    )
    dr_source = (RUNNERS_ROOT / "frontres_dr_sweep_eval.py").read_text(encoding="utf-8")
    assert "run_frontres_v015_policy_quality_heldout_eval" in quality_source
    assert "run_frontres_v015_deployment_composition_eval" in composition_source
    assert "evaluate_frontres_dr_sweep" in dr_source

    print(
        "[T-evaluation-authority] held-out quality, deployment composition, and DR sweep remain independent; embedded periodic/offline routes are absent",
        flush=True,
    )


def main() -> None:
    test_t_three_capabilities_and_training_isolation()
    print("frontres_v015_evaluation_isolation_contract: ok", flush=True)


if __name__ == "__main__":
    main()
