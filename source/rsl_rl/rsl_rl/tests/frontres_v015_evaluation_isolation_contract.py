#!/usr/bin/env python3
"""Deterministic Step 3C contract for v015 diagnostics and evaluation isolation."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
GAIN_CONSUMER_TEST = RSL_ROOT / "tests" / "frontres_v015_gain_consumer_contract.py"
DIAGNOSTICS_PATH = RSL_ROOT / "frontres" / "frontres_segment_diagnostics.py"
LIVE_TRAINING_PATH = RSL_ROOT / "runners" / "frontres_segment_live_training.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_owners():
    gain_contract = _load("frontres_v015_evaluation_gain_helper", GAIN_CONSUMER_TEST)
    one_action, helper, commands, hooks, setup, live_probe, gain, sampler = gain_contract._load_owners()
    diagnostics = _load("rsl_rl.frontres.frontres_segment_diagnostics", DIAGNOSTICS_PATH)
    live_training = _load("rsl_rl.runners.frontres.frontres_segment_live_training", LIVE_TRAINING_PATH)
    return gain_contract, one_action, helper, commands, hooks, setup, live_probe, gain, sampler, diagnostics, live_training


def _assert_same_with_nan(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(torch.nan_to_num(actual, nan=0.0), torch.nan_to_num(expected, nan=0.0))


def _expect_value_error(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _expect_runtime_error(fn) -> str:
    try:
        fn()
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("expected RuntimeError")


def test_t_diagnostic_and_v003_evaluator(
    gain_contract,
    one_action,
    helper,
    commands,
    hooks,
    setup,
    live_probe,
    gain,
    diagnostics,
) -> None:
    legacy_calls: list[str] = []
    original_segment_gain = gain.compute_segment_gain
    original_capture_gain = live_probe._capture_paired_gain

    def legacy_forbidden(*_args, **_kwargs):
        legacy_calls.append("legacy")
        raise AssertionError("v015 diagnostics must not call a Clean-global legacy owner")

    gain.compute_segment_gain = legacy_forbidden
    live_probe._capture_paired_gain = legacy_forbidden
    try:
        captured = gain_contract._capture_consumer(
            one_action, helper, commands, hooks, setup, live_probe
        )
        returned = captured.result.return_evidence
        priority = captured.result.priority_evidence
        return_before = returned.gain_total.detach().clone()
        priority_before = priority.gain_total.detach().clone()
        report = diagnostics.build_frontres_v015_local_evaluation_report(captured.result)
    finally:
        gain.compute_segment_gain = original_segment_gain
        live_probe._capture_paired_gain = original_capture_gain

    report.validate()
    assert not legacy_calls
    valid = returned.policy_row_valid.bool()
    assert report.evaluation_kind == "local_k_candidate_only"
    assert report.gain_source == "FRS-GAIN-v003-intent-physics-local-repair"
    assert report.intent_q29_provenance == "deployment_noisy_q29"
    assert report.intent_q29_source == "motion_internal_q29"
    assert report.valid_policy_row_count == int(valid.sum().item())
    assert abs(report.intent_gain_mean - float(returned.intent_gain[valid].mean().item())) < 1e-6
    assert abs(report.physics_gain_mean - float(returned.physics_gain[valid].mean().item())) < 1e-6
    assert abs(report.repair_cost_mean - float(returned.repair_cost[valid].mean().item())) < 1e-6
    assert abs(report.gain_total_mean - float(returned.gain_total[valid].mean().item())) < 1e-6
    _assert_same_with_nan(returned.gain_total, return_before)
    _assert_same_with_nan(priority.gain_total, priority_before)

    no_valid_report = replace(
        report,
        valid_policy_row_count=0,
        intent_gain_mean=float("nan"),
        physics_gain_mean=float("nan"),
        repair_cost_mean=float("nan"),
        gain_total_mean=float("nan"),
        gain_total_pos_frac=float("nan"),
    )
    no_valid_report.validate()
    _expect_value_error(lambda: replace(no_valid_report, gain_total_mean=0.0).validate())

    text = diagnostics.format_frontres_v015_local_evaluation_report(report)
    assert "[FrontRES v015 Local-K Evaluation]" in text
    assert "provenance=deployment_noisy_q29" in text
    assert "source=motion_internal_q29" in text
    assert "physics: gain=" in text and "repair: cost=" in text
    assert "style=" not in text and "Clean" not in text
    print(
        "[T-diagnostic/T-evaluator/T-no-v002-fallback/T-no-zero-fill] sealed v003 candidate evidence formats intent/physics/cost/source only",
        flush=True,
    )


def test_t_composition_protocol_and_legacy_isolation(
    gain_contract,
    one_action,
    helper,
    commands,
    hooks,
    setup,
    live_probe,
    diagnostics,
    live_training,
) -> None:
    captured = gain_contract._capture_consumer(
        one_action, helper, commands, hooks, setup, live_probe
    )
    returned = captured.result.return_evidence
    priority = captured.result.priority_evidence
    return_before = returned.gain_total.detach().clone()
    priority_before = priority.gain_total.detach().clone()

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
    _assert_same_with_nan(returned.gain_total, return_before)
    _assert_same_with_nan(priority.gain_total, priority_before)

    _expect_value_error(lambda: replace(protocol, priority_feedback=True).validate())
    try:
        diagnostics.build_frontres_v015_composition_evaluation_protocol(
            reference_stream_id="demo/deployment_motion.npz",
            frame_count=8,
            femr_action_count=8,
            return_evidence=returned,
        )
    except TypeError:
        pass
    else:
        raise AssertionError("composition protocol accepted local return evidence")

    runner = SimpleNamespace(
        _frontres_future_intent_layout=SimpleNamespace(version="frontres-v015-future-intent-q29-v1")
    )
    calls = (
        lambda: live_training.run_frontres_segment_periodic_eval(runner, iteration=1, train_summary={}),
        lambda: live_training.run_frontres_segment_offline_eval(runner, num_eval_segments=1, rollout_steps=1),
        lambda: live_training.run_frontres_segment_sequence_offline_eval(runner, num_eval_sequences=1, rollout_steps=1),
    )
    for call in calls:
        error = _expect_runtime_error(call)
        assert "legacy v002/quartet evaluator" in error

    print(
        "[T-composition-isolation] deployment protocol has no local feedback, and every legacy evaluator rejects v015 before capture",
        flush=True,
    )


def main() -> None:
    owners = _load_owners()
    test_t_diagnostic_and_v003_evaluator(*owners[:8], owners[9])
    test_t_composition_protocol_and_legacy_isolation(
        owners[0], owners[1], owners[2], owners[3], owners[4], owners[5], owners[6], owners[9], owners[10]
    )
    print("frontres_v015_evaluation_isolation_contract: ok", flush=True)


if __name__ == "__main__":
    main()
