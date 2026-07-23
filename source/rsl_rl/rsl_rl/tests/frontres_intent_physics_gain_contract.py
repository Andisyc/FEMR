#!/usr/bin/env python3
"""Deterministic S1 contracts for the FRS-GAIN-v004 pure owner."""
from __future__ import annotations

from dataclasses import fields, replace
import importlib.util
import inspect
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "frontres_intent_physics_gain_target",
    ROOT / "rsl_rl" / "frontres" / "frontres_gain.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _q29(value: float, *, batch_size: int = 1, steps: int = 2) -> torch.Tensor:
    tensor = torch.zeros(batch_size, steps, 29)
    tensor[..., 0] = value
    return tensor


def _evidence(
    *,
    intent: torch.Tensor | None = None,
    repaired: torch.Tensor | None = None,
    noisy: torch.Tensor | None = None,
    provenance: str = "deployment_noisy_q29",
    source: str = "motion_internal_q29",
    action_steps: torch.Tensor | None = None,
    **kwargs: object,
):
    intent = _q29(0.0, batch_size=2) if intent is None else intent
    repaired = _q29(0.1, batch_size=int(intent.shape[0])) if repaired is None else repaired
    noisy = _q29(0.4, batch_size=int(intent.shape[0])) if noisy is None else noisy
    action_steps = torch.zeros(int(intent.shape[0]), 6) if action_steps is None else action_steps
    payload = {
        "intent_q29": intent,
        "repaired_q29": repaired,
        "noisy_q29": noisy,
        "intent_q29_provenance": provenance,
        "intent_q29_source": source,
        "repair_action_steps": action_steps,
        "repaired_success": torch.ones(int(intent.shape[0]), dtype=torch.bool),
        "noisy_success": torch.zeros(int(intent.shape[0]), dtype=torch.bool),
        "repaired_survival": torch.full((int(intent.shape[0]),), 4.0),
        "noisy_survival": torch.full((int(intent.shape[0]),), 2.0),
        "effective_horizon_k": 4,
        "repaired_contact_violation": torch.zeros(int(intent.shape[0])),
        "noisy_contact_violation": torch.ones(int(intent.shape[0])),
        "repaired_zmp_violation": torch.zeros(int(intent.shape[0])),
        "noisy_zmp_violation": torch.zeros(int(intent.shape[0])),
    }
    payload.update(kwargs)
    return MODULE.FrontRESIntentPhysicsGainInput(**payload)


def _compute(evidence, **config: float):
    return MODULE.compute_intent_physics_local_repair_gain(
        evidence,
        config=MODULE.FrontRESIntentPhysicsGainConfig(**config),
    )


def test_value_sign_and_alias() -> None:
    result = _compute(_evidence())
    torch.testing.assert_close(result.intent_q29_noisy_error, torch.full((2,), 0.4))
    torch.testing.assert_close(result.intent_q29_repaired_error, torch.full((2,), 0.1))
    torch.testing.assert_close(result.intent_gain, torch.full((2,), 0.3))
    torch.testing.assert_close(result.physics_gain, torch.full((2,), 0.75))
    torch.testing.assert_close(result.repair_cost, torch.zeros(2))
    assert bool((result.gain_total > 2.0).all())
    assert bool(result.physics_admissible_repaired.all())
    assert not bool(result.physics_admissible_noisy.any())
    torch.testing.assert_close(result.style_gain, result.intent_gain)
    assert bool((result.intent_gain > 0.0).all())

    negative = _compute(_evidence(repaired=_q29(0.7, batch_size=2)))
    assert bool((negative.intent_gain < 0.0).all())
    print("[T-value/T-sign] fixed-I q29 error produces signed Noisy-to-Repair intent gain", flush=True)


def test_noop_and_fixed_target_invariant() -> None:
    noop = _compute(_evidence(
        repaired=_q29(0.4, batch_size=2),
        noisy_success=torch.ones(2, dtype=torch.bool),
        noisy_survival=torch.full((2,), 4.0),
        noisy_contact_violation=torch.zeros(2),
    ))
    torch.testing.assert_close(noop.intent_gain, torch.zeros(2))
    torch.testing.assert_close(noop.gain_total, torch.zeros(2))

    shifted_target = _compute(_evidence(intent=_q29(0.2, batch_size=2)))
    torch.testing.assert_close(shifted_target.intent_gain, torch.full((2,), 0.1))
    assert not torch.equal(shifted_target.intent_gain, noop.intent_gain)
    print("[T-noop/T-invariant] no-op has zero intent gain; the fixed I target is active", flush=True)


def test_root_clean_exclusion_and_provenance_rejection() -> None:
    field_names = {field.name for field in fields(MODULE.FrontRESIntentPhysicsGainInput)}
    assert not any(token in name.lower() for name in field_names for token in ("clean", "root", "global"))
    assert set(inspect.signature(MODULE.compute_intent_physics_local_repair_gain).parameters) == {
        "evidence",
        "config",
    }

    base = _evidence()
    try:
        MODULE.FrontRESIntentPhysicsGainInput(
            **{**base.__dict__, "clean_root_positions": torch.zeros(2, 3)}
        )
    except TypeError:
        pass
    else:
        raise AssertionError("v003 typed gain input accepted a Clean/root field")

    for rejected in (
        replace(base, intent_q29_provenance="clean_q29"),
        replace(base, intent_q29_source="root_global_q29"),
    ):
        try:
            _compute(rejected)
        except ValueError:
            pass
        else:
            raise AssertionError("v003 accepted prohibited q29 provenance")
    print("[T-root-exclusion/T-provenance] no Clean/root/global input channel reaches intent fidelity", flush=True)


def test_optional_components_are_unconfirmed_not_zero() -> None:
    result = _compute(_evidence())
    assert torch.isnan(result.intent_qvel_gain).all()
    assert torch.isnan(result.intent_qacc_gain).all()
    assert torch.isnan(result.repair_temporal_change).all()
    torch.testing.assert_close(result.repair_cost, torch.zeros(2))

    partial = replace(_evidence(), intent_qvel=_q29(0.0, batch_size=2))
    try:
        _compute(partial)
    except ValueError:
        pass
    else:
        raise AssertionError("partial qvel evidence was silently accepted")
    print("[T-unconfirmed] absent qvel/qacc/one-action temporal terms stay NaN, never zero-filled", flush=True)


def test_paired_physics_k_normalization_and_full_six_d_cost() -> None:
    actions = torch.zeros(2, 1, 6)
    actions[0, 0, 0] = 1.0
    actions[1, 0, 5] = 2.0
    evidence = _evidence(
        intent=_q29(0.0, batch_size=1),
        repaired=_q29(0.0, batch_size=1),
        noisy=_q29(0.0, batch_size=1),
        action_steps=actions,
        repaired_success=torch.tensor([True]),
        noisy_success=torch.tensor([True]),
        repaired_survival=torch.tensor([4.0]),
        noisy_survival=torch.tensor([2.0]),
        repaired_zmp_margin=torch.tensor([0.3]),
        noisy_zmp_margin=torch.tensor([0.1]),
        repaired_contact=torch.tensor([1.0]),
        noisy_contact=torch.tensor([0.5]),
        noisy_contact_violation=torch.zeros(1),
    )
    result = _compute(evidence)
    expected_norm = torch.tensor([(1.0 + 2.0) / 2.0])
    expected_temporal = torch.tensor([5.0**0.5])
    expected_cost = (expected_norm + expected_temporal) / 2.0
    torch.testing.assert_close(result.physics_survival_gain, torch.tensor([0.5]))
    torch.testing.assert_close(result.physics_gain, torch.tensor([0.3]))
    torch.testing.assert_close(result.repair_norm, expected_norm)
    torch.testing.assert_close(result.repair_temporal_change, expected_temporal)
    torch.testing.assert_close(result.repair_cost, expected_cost)
    expected_penalty = 0.15 * expected_cost / (1.0 + expected_cost)
    expected_total = torch.tensor([2.5]) - expected_penalty
    torch.testing.assert_close(result.gain_total, expected_total)
    print("[T-pair/T-full6] K-normalized paired physics and all-6D repair cost remain explicit", flush=True)


def test_phase_conditioning_and_lexicographic_dominance() -> None:
    expected = torch.tensor([[[1, 1]], [[1, 0]], [[1, 0]], [[1, 1]]], dtype=torch.float32)
    actual = expected.clone()
    zmp = torch.tensor([[-0.2], [-0.2], [0.1], [0.1]])
    valid = torch.ones(4, 1, dtype=torch.bool)
    phase = MODULE.evaluate_phase_conditioned_physics(
        expected, actual, zmp, valid, timing_tolerance=1, recovery_window=1
    )
    torch.testing.assert_close(phase["contact_violation"], torch.zeros(1))
    assert not bool(phase["zmp_applicable_steps"][1, 0])
    assert bool(phase["zmp_applicable_steps"][2, 0])
    torch.testing.assert_close(phase["zmp_step_violation"][:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))

    extra_step = actual.clone()
    extra_step[0, 0, 0] = 0
    strict = MODULE.evaluate_phase_conditioned_physics(
        torch.ones_like(expected), extra_step, torch.ones_like(zmp), valid, timing_tolerance=0
    )
    assert float(strict["contact_violation"][0]) > 0.0

    unsafe_better_intent = _compute(_evidence(
        repaired=_q29(0.0, batch_size=2),
        noisy=_q29(0.4, batch_size=2),
        repaired_contact_violation=torch.ones(2),
        noisy_contact_violation=torch.zeros(2),
        noisy_success=torch.ones(2, dtype=torch.bool),
        noisy_survival=torch.full((2,), 4.0),
    ))
    assert bool((unsafe_better_intent.gain_total < 0.0).all())
    flight = MODULE.evaluate_phase_conditioned_physics(
        torch.zeros(2, 1, 2), torch.zeros(2, 1, 2), torch.full((2, 1), float("nan")),
        torch.ones(2, 1, dtype=torch.bool), timing_tolerance=0, recovery_window=0,
    )
    torch.testing.assert_close(flight["zmp_violation"], torch.zeros(1))
    permutation = torch.tensor([1, 0])
    two = MODULE.evaluate_phase_conditioned_physics(
        expected.repeat(1, 2, 1), actual.repeat(1, 2, 1), zmp.repeat(1, 2), valid.repeat(1, 2)
    )
    permuted = MODULE.evaluate_phase_conditioned_physics(
        expected.repeat(1, 2, 1).index_select(1, permutation),
        actual.repeat(1, 2, 1).index_select(1, permutation),
        zmp.repeat(1, 2).index_select(1, permutation),
        valid.repeat(1, 2).index_select(1, permutation),
    )
    torch.testing.assert_close(permuted["zmp_violation"], two["zmp_violation"].index_select(0, permutation))
    torch.testing.assert_close(
        permuted["zmp_step_violation"], two["zmp_step_violation"].index_select(1, permutation)
    )
    print("[T-phase/T-lexicographic] planned transitions recover; unsafe Intent cannot compensate Physics", flush=True)


def main() -> None:
    test_value_sign_and_alias()
    test_noop_and_fixed_target_invariant()
    test_root_clean_exclusion_and_provenance_rejection()
    test_optional_components_are_unconfirmed_not_zero()
    test_paired_physics_k_normalization_and_full_six_d_cost()
    test_phase_conditioning_and_lexicographic_dominance()
    print("frontres_intent_physics_gain_contract: ok", flush=True)


if __name__ == "__main__":
    main()
