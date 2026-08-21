"""Typed raw Clean telemetry producer for the v010 calibration route.

This module is deliberately downstream of the existing execution-frame and
relational-outcome owners.  It does not define a second Gain, reward, or
physics metric.  A repeated Clean window is compared against the first Clean
window only to express measurement noise in the existing Clean-relative
fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rsl_rl.frontres.frontres_clean_relative_calibration import (
    CleanCalibrationObservation,
    CleanHardEventEvidence,
)
from rsl_rl.frontres.frontres_relational_outcome import build_frontres_relational_outcome
from rsl_rl.frontres.frontres_segment_evidence import FrontRESExecutedKTrajectory


@dataclass(frozen=True)
class FrontRESCleanRawWindow:
    """One complete K-step Clean execution and its sealed support plan."""

    repeat_id: str
    trajectory: FrontRESExecutedKTrajectory
    expected_support: torch.Tensor
    timestep_seconds: float

    def validate(self) -> None:
        if not isinstance(self.repeat_id, str) or not self.repeat_id:
            raise ValueError("raw Clean window requires a non-empty repeat_id")
        if not isinstance(self.trajectory, FrontRESExecutedKTrajectory):
            raise TypeError("raw Clean window requires FrontRESExecutedKTrajectory")
        self.trajectory.validate()
        k_steps = int(self.trajectory.joint_pos.shape[0])
        if (
            not isinstance(self.expected_support, torch.Tensor)
            or tuple(self.expected_support.shape) != (k_steps, 1, 2)
            or self.expected_support.requires_grad
            or not bool(torch.isfinite(self.expected_support.float()).all().item())
        ):
            raise ValueError("raw Clean window expected_support must be detached finite [K,1,2]")
        if bool(((self.expected_support != 0) & (self.expected_support != 1)).any().item()):
            raise ValueError("raw Clean window expected_support must be binary")
        if isinstance(self.timestep_seconds, bool) or not isinstance(self.timestep_seconds, (int, float)):
            raise ValueError("raw Clean window timestep_seconds must be real")
        if not float(self.timestep_seconds) > 0.0:
            raise ValueError("raw Clean window timestep_seconds must be positive")


@dataclass(frozen=True)
class FrontRESCleanTelemetryMeasurement:
    observation: CleanCalibrationObservation
    hard_events: CleanHardEventEvidence


def build_clean_calibration_observation(
    *,
    reference: FrontRESCleanRawWindow,
    candidate: FrontRESCleanRawWindow,
    domain_id: str,
    scenario_id: str,
) -> CleanCalibrationObservation:
    """Convert one raw Clean repeat into the existing calibration schema.

    The first repeat is its own reference, so its relative dynamic-error and
    support-drift fields are exactly zero.  Later repeats use the same first
    Clean trajectory.  This is a measurement-noise construction, not a second
    semantic score.  Hard physics failures are rejected because they cannot
    be used to estimate sensor/estimator tolerance.
    """

    return build_clean_calibration_measurement(
        reference=reference,
        candidate=candidate,
        domain_id=domain_id,
        scenario_id=scenario_id,
    ).observation


def build_clean_calibration_measurement(
    *,
    reference: FrontRESCleanRawWindow,
    candidate: FrontRESCleanRawWindow,
    domain_id: str,
    scenario_id: str,
) -> FrontRESCleanTelemetryMeasurement:
    """Build continuous calibration fields and preserve hard-event labels."""

    reference.validate()
    candidate.validate()
    if reference.timestep_seconds != candidate.timestep_seconds:
        raise ValueError("Clean repeat timestep identity mismatch")
    if tuple(reference.trajectory.joint_pos.shape) != tuple(candidate.trajectory.joint_pos.shape):
        raise ValueError("Clean repeats must have identical K-step state shapes")
    if not torch.equal(reference.expected_support, candidate.expected_support):
        raise ValueError("Clean repeats changed the sealed expected-support plan")
    if not domain_id or not scenario_id:
        raise ValueError("Clean telemetry observation requires domain and Scenario identity")

    zero_action = torch.zeros(6, device=candidate.trajectory.joint_pos.device, dtype=torch.float32)
    outcome = build_frontres_relational_outcome(
        clean=reference.trajectory,
        repair=candidate.trajectory,
        expected_support=reference.expected_support,
        repair_action=zero_action,
    )
    if (
        not outcome.survival_ok
        or outcome.expected_support_no_load > 0.0
        or outcome.unplanned_support_switch > 0.0
        or outcome.illegal_contact_duration > 0.0
    ):
        raise RuntimeError(
            "raw Clean calibration window contains a hard Physics event; "
            "it cannot define measurement-noise tolerance"
        )
    observation = CleanCalibrationObservation(
        domain_id=domain_id,
        scenario_id=scenario_id,
        repeat_id=candidate.repeat_id,
        capture_margin=float(outcome.capture_margin),
        capture_margin_trend=float(outcome.capture_margin_trend),
        zmp_applicable=bool(outcome.zmp_applicable),
        zmp_margin=None if outcome.zmp_margin is None else float(outcome.zmp_margin),
        linear_momentum_error=float(outcome.linear_momentum_error),
        angular_momentum_error=float(outcome.angular_momentum_error),
        support_drift=float(outcome.support_drift),
    )
    hard_events = CleanHardEventEvidence(
        survival_ok=bool(outcome.survival_ok),
        survival_failure_duration=float(outcome.survival_failure_duration),
        expected_support_no_load=float(outcome.expected_support_no_load),
        unplanned_support_switch=float(outcome.unplanned_support_switch),
        illegal_contact_duration=float(outcome.illegal_contact_duration),
        valid_step_count=int(candidate.trajectory.valid_mask[:, 0].bool().sum().item()),
        zmp_applicable_step_count=int(candidate.trajectory.zmp_margin[:, 0].isfinite().sum().item()),
    )
    hard_events.validate()
    return FrontRESCleanTelemetryMeasurement(observation=observation, hard_events=hard_events)


__all__ = (
    "FrontRESCleanRawWindow",
    "FrontRESCleanTelemetryMeasurement",
    "build_clean_calibration_measurement",
    "build_clean_calibration_observation",
)
