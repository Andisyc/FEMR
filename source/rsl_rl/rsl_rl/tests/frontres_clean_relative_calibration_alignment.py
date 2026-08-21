"""Semantic pseudo-samples for the Clean-relative calibration boundary."""

from dataclasses import replace

from rsl_rl.frontres.frontres_clean_relative_calibration import (
    CleanCalibration,
    CleanCalibrationObservation,
    CleanReference,
    apply_clean_relative_calibration,
    build_clean_calibration,
)
from rsl_rl.frontres.frontres_hierarchical_gain_candidate import Outcome


def _calibration() -> CleanCalibration:
    values = (
        (0.04, 0.010, 0.030, 0.050, 0.100, 0.020),
        (0.05, 0.015, 0.035, 0.075, 0.150, 0.030),
        (0.03, 0.005, 0.025, 0.025, 0.050, 0.010),
    )
    observations = tuple(
        CleanCalibrationObservation(
            domain_id="robot-gmt-cache-test",
            scenario_id="scenario-1",
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
    return build_clean_calibration(
        observations,
        calibration_id="clean-calibration-test-v1",
        domain_id="robot-gmt-cache-test",
        field_schema_id="frontres-clean-relative-fields-v1",
        horizon_k=8,
        timestep_seconds=0.02,
        coverage=0.95,
    )


def main() -> None:
    calibration = _calibration()
    clean = CleanReference(
        domain_id=calibration.domain_id,
        scenario_id="scenario-1",
        capture_margin=0.04,
        capture_margin_trend=0.01,
        zmp_applicable=True,
        zmp_margin=0.03,
    )
    ordinary = Outcome(
        capture_margin=0.04,
        capture_margin_trend=0.01,
        zmp_applicable=True,
        zmp_margin=0.03,
        linear_momentum_error=0.0,
        angular_momentum_error=0.0,
        support_drift=0.0,
    )

    same = apply_clean_relative_calibration(
        ordinary, clean, calibration, outcome_scenario_id=clean.scenario_id
    )
    assert same.recovery_bins == (0, 0, 0, 0, 0, 0)
    assert same.inside_clean_domain

    sub_resolution = apply_clean_relative_calibration(
        replace(
            ordinary,
            capture_margin=0.035,
            linear_momentum_error=0.01,
            angular_momentum_error=0.02,
            support_drift=0.005,
        ),
        clean,
        calibration,
        outcome_scenario_id=clean.scenario_id,
    )
    assert sub_resolution.recovery_bins == (0, 0, 0, 0, 0, 0)
    assert sub_resolution.inside_clean_domain

    normal_boundary = apply_clean_relative_calibration(
        replace(ordinary, angular_momentum_error=calibration.angular_momentum_resolution),
        clean,
        calibration,
        outcome_scenario_id=clean.scenario_id,
    )
    assert normal_boundary.recovery_bins == (0, 0, 0, 0, 0, 0)
    assert normal_boundary.inside_clean_domain

    clear_degradation = apply_clean_relative_calibration(
        replace(
            ordinary,
            angular_momentum_error=calibration.angular_momentum_resolution + 0.001,
        ),
        clean,
        calibration,
        outcome_scenario_id=clean.scenario_id,
    )
    assert clear_degradation.recovery_bins == (0, 0, 0, 0, -1, 0)
    assert not clear_degradation.inside_clean_domain

    absolute_violation = apply_clean_relative_calibration(
        replace(ordinary, capture_margin=-0.001),
        clean,
        calibration,
        outcome_scenario_id=clean.scenario_id,
    )
    assert not absolute_violation.absolute_physics_valid
    assert not absolute_violation.inside_clean_domain

    try:
        apply_clean_relative_calibration(
            ordinary,
            replace(clean, domain_id="wrong-domain"),
            calibration,
            outcome_scenario_id=clean.scenario_id,
        )
    except ValueError as error:
        assert "domain" in str(error)
    else:
        raise AssertionError("calibration/reference domain mismatch must fail closed")

    try:
        apply_clean_relative_calibration(
            ordinary,
            clean,
            calibration,
            outcome_scenario_id="wrong-scenario",
        )
    except ValueError as error:
        assert "Scenario" in str(error)
    else:
        raise AssertionError("Clean/Repair Scenario mismatch must fail closed")

    try:
        replace(calibration, angular_momentum_resolution=0.0).validate()
    except ValueError as error:
        assert "resolution" in str(error)
    else:
        raise AssertionError("zero calibration resolution must fail closed")

    try:
        replace(calibration, repeated_pair_count=2.5).validate()
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("fractional repeated-pair count must fail closed")

    print("frontres_clean_relative_calibration_alignment: MODULE-CORRECT")


if __name__ == "__main__":
    main()
