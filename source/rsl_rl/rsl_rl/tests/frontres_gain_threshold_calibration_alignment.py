#!/usr/bin/env python3
"""Semantic pseudo-samples for the v010 threshold-calibration producer/consumer.

It uses the production public boundaries ``build_clean_calibration`` and
``apply_clean_relative_calibration`` with hand-checkable pseudo repeats and an
independent quantile calculation.  It does not claim simulator connectivity or
real calibration quality.
"""

from __future__ import annotations

from dataclasses import replace
import math

from rsl_rl.frontres.frontres_clean_relative_calibration import (
    CleanCalibration,
    CleanCalibrationObservation,
    CleanReference,
    apply_clean_relative_calibration,
    build_clean_calibration,
)
from rsl_rl.frontres.frontres_hierarchical_gain_candidate import Outcome


def _independent_quantile(values: tuple[float, ...], coverage: float) -> float:
    """Handwritten order-statistic oracle; does not call production helpers."""

    if len(values) < 2:
        raise AssertionError("noise oracle requires repeated observations")
    ordered = sorted(abs(float(value)) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(coverage * len(ordered)) - 1))
    result = ordered[index]
    if not math.isfinite(result) or result <= 0.0:
        raise AssertionError("noise oracle requires a positive finite resolution")
    return result


def _observations() -> tuple[CleanCalibrationObservation, ...]:
    values = (
        (0.040, 0.0100, 0.0300, 0.010, 0.020, 0.0050),
        (0.041, 0.0105, 0.0308, 0.011, 0.022, 0.0060),
        (0.039, 0.0095, 0.0292, 0.009, 0.018, 0.0040),
        (0.042, 0.0110, 0.0312, 0.012, 0.023, 0.0065),
        (0.038, 0.0090, 0.0288, 0.008, 0.017, 0.0035),
    )
    return tuple(
        CleanCalibrationObservation(
            domain_id="robot-gmt-cache-pseudo",
            scenario_id="scenario-threshold-pseudo",
            repeat_id=f"repeat-{index}",
            capture_margin=capture,
            capture_margin_trend=trend,
            zmp_applicable=True,
            zmp_margin=zmp,
            linear_momentum_error=linear,
            angular_momentum_error=angular,
            support_drift=drift,
        )
        for index, (capture, trend, zmp, linear, angular, drift) in enumerate(values)
    )


def _all_pair_differences(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        values[left] - values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def _expected_resolutions(
    observations: tuple[CleanCalibrationObservation, ...], coverage: float
) -> tuple[float, ...]:
    field_names = (
        "capture_margin",
        "capture_margin_trend",
        "zmp_margin",
        "linear_momentum_error",
        "angular_momentum_error",
        "support_drift",
    )
    return tuple(
        _independent_quantile(
            _all_pair_differences(tuple(float(getattr(row, name)) for row in observations)),
            coverage,
        )
        for name in field_names
    )


def _calibration(observations: tuple[CleanCalibrationObservation, ...]) -> CleanCalibration:
    return build_clean_calibration(
        observations,
        calibration_id="clean-calibration-pseudo-v2",
        domain_id="robot-gmt-cache-pseudo",
        field_schema_id="frontres-clean-relative-fields-v1",
        horizon_k=8,
        timestep_seconds=0.02,
        coverage=0.95,
    )


def _clean(calibration: CleanCalibration) -> CleanReference:
    return CleanReference(
        domain_id=calibration.domain_id,
        scenario_id="scenario-threshold-pseudo",
        capture_margin=0.04,
        capture_margin_trend=0.01,
        zmp_applicable=True,
        zmp_margin=0.03,
    )


def _ordinary() -> Outcome:
    return Outcome(
        capture_margin=0.04,
        capture_margin_trend=0.01,
        zmp_applicable=True,
        zmp_margin=0.03,
        linear_momentum_error=0.0,
        angular_momentum_error=0.0,
        support_drift=0.0,
    )


def main() -> None:
    observations = _observations()
    expected = _expected_resolutions(observations, 0.95)
    calibration = _calibration(observations)
    actual = (
        calibration.capture_margin_resolution,
        calibration.capture_trend_resolution,
        calibration.zmp_margin_resolution,
        calibration.linear_momentum_resolution,
        calibration.angular_momentum_resolution,
        calibration.support_drift_resolution,
    )
    assert actual == expected
    assert calibration.repeated_sample_count == 5
    assert calibration.repeated_pair_count == 10
    assert calibration.source_scenario_ids == ("scenario-threshold-pseudo",)
    assert len(calibration.source_observation_hash) == 64
    assert len(calibration.artifact_hash) == 64

    # C4: input row order cannot change resolutions or artifact identity.
    permuted = _calibration(tuple(reversed(observations)))
    assert permuted == calibration

    # Sensitivity: the oracle must reject a plausible half-resolution mutant.
    try:
        assert actual[0] == expected[0] * 0.5
    except AssertionError:
        pass
    else:
        raise AssertionError("resolution oracle did not detect controlled mutant")

    clean = _clean(calibration)
    ordinary = _ordinary()

    # C1: no physical change, independent repeated-baseline oracle -> SAME bins.
    same = apply_clean_relative_calibration(
        ordinary, clean, calibration, outcome_scenario_id=clean.scenario_id
    )
    assert same.recovery_bins == (0, 0, 0, 0, 0, 0)
    assert same.absolute_physics_valid and same.inside_clean_domain

    # C2: sub-resolution change must not create a directional bin.
    sub_resolution = apply_clean_relative_calibration(
        replace(
            ordinary,
            capture_margin=ordinary.capture_margin + 0.5 * calibration.capture_margin_resolution,
            linear_momentum_error=0.5 * calibration.linear_momentum_resolution,
        ),
        clean,
        calibration,
        outcome_scenario_id=clean.scenario_id,
    )
    assert sub_resolution.recovery_bins == (0, 0, 0, 0, 0, 0)

    # C2: a supra-resolution one-sided change must be observable.
    supra_resolution = apply_clean_relative_calibration(
        replace(
            ordinary,
            capture_margin=ordinary.capture_margin + 2.2 * calibration.capture_margin_resolution,
        ),
        clean,
        calibration,
        outcome_scenario_id=clean.scenario_id,
    )
    assert supra_resolution.recovery_bins[0] > 0

    # C3: hard Physics remains invalid for ordinary improvements.
    hard_failure = apply_clean_relative_calibration(
        replace(ordinary, survival_ok=False, survival_failure_duration=0.2),
        clean,
        calibration,
        outcome_scenario_id=clean.scenario_id,
    )
    assert not hard_failure.absolute_physics_valid
    no_load = apply_clean_relative_calibration(
        replace(ordinary, expected_support_no_load=0.2),
        clean,
        calibration,
        outcome_scenario_id=clean.scenario_id,
    )
    assert not no_load.absolute_physics_valid

    # C3/C4: invalid, N/A mismatch and identity swaps fail closed.
    for malformed in (
        replace(ordinary, capture_margin=float("nan")),
        replace(ordinary, zmp_applicable=False, zmp_margin=0.0),
    ):
        try:
            apply_clean_relative_calibration(
                malformed, clean, calibration, outcome_scenario_id=clean.scenario_id
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid/N/A evidence must fail closed")

    try:
        apply_clean_relative_calibration(
            ordinary, clean, calibration, outcome_scenario_id="wrong-scenario"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Scenario identity mutation must be detected")

    # Producer identity, N/A, and evidence validity fail closed.
    for malformed_rows in (
        observations[:1],
        (replace(observations[0], domain_id="wrong-domain"), *observations[1:]),
        (replace(observations[0], capture_margin=float("nan")), *observations[1:]),
        (
            replace(observations[0], zmp_applicable=False, zmp_margin=None),
            *observations[1:],
        ),
    ):
        try:
            _calibration(tuple(malformed_rows))
        except ValueError:
            pass
        else:
            raise AssertionError("malformed repeated calibration evidence must fail closed")

    for tampered in (
        replace(calibration, artifact_hash="0" * 64),
        replace(calibration, field_units=(("capture_margin", "rad"),)),
    ):
        try:
            tampered.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("artifact hash/unit mutation must fail closed")

    print("frontres_gain_threshold_calibration_alignment: MODULE-CORRECT")


if __name__ == "__main__":
    main()
