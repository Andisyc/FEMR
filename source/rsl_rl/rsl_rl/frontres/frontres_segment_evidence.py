"""Immutable one-action-K, paired Physics and scalar-return evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rsl_rl.frontres.frontres_gain import FrontRESRecoveryAwareGainInput


@dataclass(frozen=True)
class FrontRESExecutedKTrajectory:
    """Framework-free K-step execution snapshot for one logical role."""

    joint_pos: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    key_body_pos: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    foot_pos: torch.Tensor
    contact: torch.Tensor
    zmp_margin: torch.Tensor
    survival: torch.Tensor
    valid_mask: torch.Tensor

    def validate(self) -> None:
        if self.joint_pos.ndim != 3 or int(self.joint_pos.shape[-1]) != 29:
            raise ValueError("v017 executed trajectory requires joint_pos [K,B,29]")
        k_steps, batch_size = int(self.joint_pos.shape[0]), int(self.joint_pos.shape[1])
        shapes = {
            "root_pos": (k_steps, batch_size, 3),
            "root_quat": (k_steps, batch_size, 4),
            "root_lin_vel": (k_steps, batch_size, 3),
            "root_ang_vel": (k_steps, batch_size, 3),
            "foot_pos": (k_steps, batch_size, 2, 3),
            "contact": (k_steps, batch_size, 2),
            "zmp_margin": (k_steps, batch_size),
            "survival": (k_steps, batch_size),
            "valid_mask": (k_steps, batch_size),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"v017 executed trajectory {name} must be {shape}")
        if (
            self.key_body_pos.ndim != 4
            or tuple(self.key_body_pos.shape[:2]) != (k_steps, batch_size)
            or int(self.key_body_pos.shape[-1]) != 3
            or int(self.key_body_pos.shape[-2]) <= 0
        ):
            raise ValueError("v017 executed trajectory key_body_pos must be [K,B,J,3]")
        binary = (self.contact, self.survival)
        if any(bool(((value != 0) & (value != 1)).any()) for value in binary):
            raise ValueError("v017 executed trajectory Contact/survival must be binary")
        required = (
            self.joint_pos,
            self.root_pos,
            self.root_quat,
            self.key_body_pos,
            self.root_lin_vel,
            self.root_ang_vel,
            self.foot_pos,
        )
        if any(value.requires_grad or not bool(torch.isfinite(value.float()).all()) for value in required):
            raise ValueError("v017 executed trajectory must be detached and finite")
        finite_zmp = torch.isfinite(self.zmp_margin.float())
        if bool(torch.isinf(self.zmp_margin.float()).any()):
            raise ValueError("v017 ZMP may be finite or semantic N/A, never infinite")


@dataclass(frozen=True)
class FrontRESSegmentBaselineEvidence:
    """Exactly one executed Clean and one fixed Noisy baseline for a Segment."""

    transaction_id: str
    policy_snapshot_id: str
    scenario_id: str
    noisy_segment_hash: str
    x_t_identity: str
    source_index: int
    segment_id: int
    horizon_k: int
    expected_support: torch.Tensor
    clean: FrontRESExecutedKTrajectory
    noisy: FrontRESExecutedKTrajectory
    clean_execution_count: int = 1
    noisy_execution_count: int = 1

    def validate(self) -> None:
        identities = (
            self.transaction_id,
            self.policy_snapshot_id,
            self.scenario_id,
            self.noisy_segment_hash,
            self.x_t_identity,
        )
        if any(not isinstance(value, str) or not value for value in identities):
            raise ValueError("v017 baseline requires complete immutable identity")
        if int(self.source_index) < 0 or int(self.segment_id) < 0 or int(self.horizon_k) <= 0:
            raise ValueError("v017 baseline has invalid source/segment/K identity")
        if int(self.clean_execution_count) != 1 or int(self.noisy_execution_count) != 1:
            raise ValueError("v017 baseline must execute Clean and Noisy exactly once")
        self.clean.validate()
        self.noisy.validate()
        expected = (int(self.horizon_k), 1, 2)
        if tuple(self.expected_support.shape) != expected:
            raise ValueError(f"v017 baseline expected_support must be {expected}")
        if bool(((self.expected_support != 0) & (self.expected_support != 1)).any()):
            raise ValueError("v017 expected support must be binary")
        if int(self.clean.joint_pos.shape[1]) != 1 or int(self.noisy.joint_pos.shape[1]) != 1:
            raise ValueError("v017 baseline stores one authoritative execution per Segment")
        _validate_v017_role_zmp(self.clean, self.expected_support, role="Clean")
        _validate_v017_role_zmp(self.noisy, self.expected_support, role="Noisy")


@dataclass(frozen=True)
class FrontRESRepairAttemptEvidence:
    """One Repair policy row and its K-step execution consequence."""

    transaction_id: str
    policy_snapshot_id: str
    scenario_id: str
    noisy_segment_hash: str
    x_t_identity: str
    source_index: int
    segment_id: int
    trial_index: int
    horizon_k: int
    policy_observation: torch.Tensor
    policy_privileged_observation: torch.Tensor
    policy_action: torch.Tensor
    policy_log_prob: torch.Tensor
    policy_value: torch.Tensor
    policy_mean: torch.Tensor
    policy_sigma: torch.Tensor
    repair: FrontRESExecutedKTrajectory

    def validate(self) -> None:
        identities = (
            self.transaction_id,
            self.policy_snapshot_id,
            self.scenario_id,
            self.noisy_segment_hash,
            self.x_t_identity,
        )
        if any(not isinstance(value, str) or not value for value in identities):
            raise ValueError("v017 Repair attempt requires complete immutable identity")
        if min(int(self.source_index), int(self.segment_id), int(self.trial_index)) < 0 or int(self.horizon_k) <= 0:
            raise ValueError("v017 Repair attempt has invalid source/segment/trial/K identity")
        self.repair.validate()
        if int(self.repair.joint_pos.shape[1]) != 1 or int(self.repair.joint_pos.shape[0]) != int(self.horizon_k):
            raise ValueError("v017 Repair attempt must contain one [K,1,...] execution")
        for name, value, shape in (
            ("policy_action", self.policy_action, (6,)),
            ("policy_mean", self.policy_mean, (6,)),
            ("policy_sigma", self.policy_sigma, (6,)),
            ("policy_log_prob", self.policy_log_prob, ()),
            ("policy_value", self.policy_value, ()),
        ):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"v017 Repair attempt {name} must be {shape}")
        if self.policy_observation.ndim != 1 or self.policy_privileged_observation.ndim != 1:
            raise ValueError("v017 Repair attempt policy observations must be one row")
        tensors = (
            self.policy_observation,
            self.policy_privileged_observation,
            self.policy_action,
            self.policy_log_prob,
            self.policy_value,
            self.policy_mean,
            self.policy_sigma,
        )
        if any(value.requires_grad or not bool(torch.isfinite(value.float()).all()) for value in tensors):
            raise ValueError("v017 Repair policy tuple must be detached and finite")


@dataclass(frozen=True)
class FrontRESSealedRecoveryAwareGainBatch:
    """Complete B-Scenario baselines plus exact-M Repair consistency boundary.

    The Gain evidence owner is shared by formal B8 training and bounded B2
    evaluation.  Training batch cardinality remains owned and enforced by the
    formal transaction/PPO boundary, not by this evidence value object.
    """

    baselines: tuple[FrontRESSegmentBaselineEvidence, ...]
    attempts: tuple[FrontRESRepairAttemptEvidence, ...]
    active_m: int

    def validate(self) -> None:
        scenario_count = len(self.baselines)
        if scenario_count < 1 or int(self.active_m) < 2:
            raise ValueError("sealed Gain batch requires at least one Scenario and M>=2")
        for baseline in self.baselines:
            baseline.validate()
        for attempt in self.attempts:
            attempt.validate()
        if len(self.attempts) != scenario_count * int(self.active_m):
            raise ValueError("sealed Gain batch requires exact B x M Repair attempts")
        by_source = {int(value.source_index): value for value in self.baselines}
        expected_sources = set(range(scenario_count))
        if len(by_source) != scenario_count or set(by_source) != expected_sources:
            raise ValueError("sealed Gain batch requires distinct contiguous source identities 0..B-1")
        seen: set[tuple[int, int]] = set()
        counts = {source: 0 for source in by_source}
        for attempt in self.attempts:
            key = (int(attempt.source_index), int(attempt.trial_index))
            if key in seen or int(attempt.source_index) not in by_source:
                raise ValueError("v017 sealed Gain batch repeats or mixes Repair attempts")
            seen.add(key)
            counts[int(attempt.source_index)] += 1
            baseline = by_source[int(attempt.source_index)]
            if (
                attempt.transaction_id != baseline.transaction_id
                or attempt.policy_snapshot_id != baseline.policy_snapshot_id
                or attempt.scenario_id != baseline.scenario_id
                or attempt.noisy_segment_hash != baseline.noisy_segment_hash
                or attempt.x_t_identity != baseline.x_t_identity
                or int(attempt.segment_id) != int(baseline.segment_id)
                or int(attempt.horizon_k) != int(baseline.horizon_k)
            ):
                raise ValueError("v017 sealed Gain batch mixes baseline/Repair identity")
            _validate_v017_role_zmp(attempt.repair, baseline.expected_support, role="Repair")
        if any(count != int(self.active_m) for count in counts.values()):
            raise ValueError("v017 sealed Gain batch requires exact M attempts per Segment")
        if len({value.transaction_id for value in self.baselines}) != 1 or len(
            {value.policy_snapshot_id for value in self.baselines}
        ) != 1:
            raise ValueError("v017 sealed Gain batch requires one transaction and frozen policy")

    def to_gain_input(self) -> FrontRESRecoveryAwareGainInput:
        """Alias each immutable baseline to its M Repair rows without re-execution."""

        self.validate()
        ordered = tuple(sorted(self.attempts, key=lambda value: (value.source_index, value.trial_index)))
        baseline_by_source = {int(value.source_index): value for value in self.baselines}

        def role_tensor(role: str, field: str) -> torch.Tensor:
            values = []
            for attempt in ordered:
                if role == "repaired":
                    trajectory = attempt.repair
                else:
                    trajectory = getattr(baseline_by_source[int(attempt.source_index)], role)
                values.append(getattr(trajectory, field)[:, 0])
            return torch.stack(values, dim=1).detach().clone()

        expected_support = torch.stack(
            [baseline_by_source[int(attempt.source_index)].expected_support[:, 0] for attempt in ordered],
            dim=1,
        ).detach().clone()
        fields: dict[str, torch.Tensor] = {}
        for role in ("clean", "noisy", "repaired"):
            for field in (
                "joint_pos",
                "root_pos",
                "root_quat",
                "key_body_pos",
                "root_lin_vel",
                "root_ang_vel",
                "foot_pos",
                "contact",
                "zmp_margin",
                "survival",
                "valid_mask",
            ):
                fields[f"{role}_{field}"] = role_tensor(role, field)
        return FrontRESRecoveryAwareGainInput(
            **fields,
            expected_support=expected_support,
            repair_actions=torch.stack([value.policy_action for value in ordered], dim=0).detach().clone(),
        )

    @property
    def ordered_attempts(self) -> tuple[FrontRESRepairAttemptEvidence, ...]:
        self.validate()
        return tuple(sorted(self.attempts, key=lambda value: (value.source_index, value.trial_index)))


def _validate_v017_role_zmp(
    trajectory: FrontRESExecutedKTrajectory,
    expected_support: torch.Tensor,
    *,
    role: str,
) -> None:
    expected = expected_support.bool()
    applicable = trajectory.valid_mask.bool() & expected.any(dim=-1) & trajectory.contact.bool().any(dim=-1)
    finite = torch.isfinite(trajectory.zmp_margin.float())
    if not bool(finite[applicable].all()) or bool(finite[~applicable].any()):
        raise ValueError(f"v017 {role} ZMP must be finite exactly on loaded expected-support steps")
