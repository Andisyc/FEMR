"""Pseudo-data contract for the raw Clean telemetry reducer."""

from __future__ import annotations

from pathlib import Path

import torch

from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[4]
install_frontres_contract_packages(ROOT / "source" / "rsl_rl" / "rsl_rl")

from rsl_rl.frontres.frontres_segment_evidence import FrontRESExecutedKTrajectory
from rsl_rl.runners.frontres_clean_calibration_telemetry import (
    FrontRESCleanCalibrationHardEventError,
    FrontRESCleanRawWindow,
    build_clean_calibration_measurement,
)


def _trajectory(*, velocity_offset: float = 0.0, no_load: bool = False) -> FrontRESExecutedKTrajectory:
    k_steps = 3
    contact = torch.ones(k_steps, 1, 2, dtype=torch.float32)
    expected = torch.ones(k_steps, 1, 2, dtype=torch.float32)
    if no_load:
        contact[-1] = 0.0
    return FrontRESExecutedKTrajectory(
        joint_pos=torch.zeros(k_steps, 1, 29),
        root_pos=torch.tensor([[[0.0, 0.0, 0.8]]] * k_steps),
        root_quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * k_steps),
        key_body_pos=torch.zeros(k_steps, 1, 2, 3),
        root_lin_vel=torch.tensor([[[velocity_offset, 0.0, 0.0]]] * k_steps),
        root_ang_vel=torch.zeros(k_steps, 1, 3),
        foot_pos=torch.tensor([[[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]]] * k_steps),
        contact=contact,
        zmp_margin=torch.full((k_steps, 1), 0.04),
        survival=torch.ones(k_steps, 1),
        valid_mask=torch.ones(k_steps, 1),
        env_origin=torch.zeros(k_steps, 1, 3),
    )


def _window(repeat_id: str, trajectory: FrontRESExecutedKTrajectory) -> FrontRESCleanRawWindow:
    return FrontRESCleanRawWindow(
        repeat_id=repeat_id,
        trajectory=trajectory,
        expected_support=torch.ones(trajectory.joint_pos.shape[0], 1, 2),
        timestep_seconds=0.02,
    )


def main() -> None:
    reference = _window("repeat-0", _trajectory())
    first = build_clean_calibration_measurement(
        reference=reference,
        candidate=reference,
        domain_id="pseudo-domain",
        scenario_id="scenario-0",
    )
    assert first.hard_events.survival_ok
    assert first.hard_events.expected_support_no_load == 0.0
    assert first.observation.linear_momentum_error == 0.0
    assert first.observation.support_drift == 0.0

    second = build_clean_calibration_measurement(
        reference=reference,
        candidate=_window("repeat-1", _trajectory(velocity_offset=0.01)),
        domain_id="pseudo-domain",
        scenario_id="scenario-0",
    )
    assert second.observation.linear_momentum_error > 0.0
    assert second.hard_events.zmp_applicable_step_count == 3

    try:
        build_clean_calibration_measurement(
            reference=reference,
            candidate=_window("repeat-bad", _trajectory(no_load=True)),
            domain_id="pseudo-domain",
            scenario_id="scenario-0",
        )
    except FrontRESCleanCalibrationHardEventError as exc:
        assert exc.repeat_id == "repeat-bad"
        assert exc.hard_events.expected_support_no_load > 0
        assert exc.hard_events.survival_ok is True
    else:
        raise AssertionError("hard Clean events must fail closed")

    print("frontres_clean_calibration_telemetry_alignment: MODULE-CORRECT")


if __name__ == "__main__":
    main()
