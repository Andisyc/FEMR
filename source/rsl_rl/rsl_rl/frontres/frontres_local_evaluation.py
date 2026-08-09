"""Read-only v015 local-K and deployment-composition evaluation owner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from rsl_rl.frontres.frontres_gain import FrontRESRecoveryAwareGainResult
from rsl_rl.frontres.frontres_segment_evidence import FrontRESSealedRecoveryAwareGainBatch


@dataclass(frozen=True)
class FrontRESSegmentReplaySummary:
    scalars: dict[str, float]
    stage: str
    objective: str


_V015_GAIN_SOURCE = "FRS-GAIN-v006-loaded-support-zmp-applicability"
_V015_LOCAL_EVALUATION_KIND = "local_k_candidate_only"
_V015_COMPOSITION_EVALUATION_KIND = "deployment_composition_protocol"


@dataclass(frozen=True)
class FrontRESV017LocalEvaluationReport:
    """Read-only projection of the exact evidence consumed by FRS-GAIN-v008."""

    transaction_id: str
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    policy_actions: tuple[tuple[float, ...], ...]
    valid_policy_row_mask: tuple[bool, ...]
    intent_remaining_noisy: tuple[float, ...]
    intent_remaining_repaired: tuple[float, ...]
    physics_remaining_noisy: tuple[float, ...]
    physics_remaining_repaired: tuple[float, ...]
    intent_channel_noisy: tuple[tuple[float, ...], ...]
    intent_channel_repaired: tuple[tuple[float, ...], ...]
    physics_channel_noisy: tuple[tuple[float | None, ...], ...]
    physics_channel_repaired: tuple[tuple[float | None, ...], ...]
    support_foot_drift_noisy: tuple[float | None, ...]
    support_foot_drift_repaired: tuple[float | None, ...]
    intent_gain: tuple[float, ...]
    physics_gain: tuple[float, ...]
    recovery_pressure: tuple[float, ...]
    weighted_physics_gain: tuple[float, ...]
    repair_cost: tuple[float, ...]
    repair_penalty: tuple[float, ...]
    cost_free_score: tuple[float, ...]
    gain_total: tuple[float, ...]
    policy_values: tuple[float, ...]
    raw_advantages: tuple[float, ...]
    clean_execution_count: tuple[int, ...]
    noisy_execution_count: tuple[int, ...]
    expected_support_steps: tuple[tuple[tuple[float, float], ...], ...]
    contact_clean_steps: tuple[tuple[tuple[float, float], ...], ...]
    contact_noisy_steps: tuple[tuple[tuple[float, float], ...], ...]
    contact_repair_steps: tuple[tuple[tuple[float, float], ...], ...]
    zmp_clean_steps: tuple[tuple[float | None, ...], ...]
    zmp_noisy_steps: tuple[tuple[float | None, ...], ...]
    zmp_repair_steps: tuple[tuple[float | None, ...], ...]
    survival_clean_steps: tuple[tuple[float, ...], ...]
    survival_noisy_steps: tuple[tuple[float, ...], ...]
    survival_repair_steps: tuple[tuple[float, ...], ...]
    contact_violation_repair_steps: tuple[tuple[bool, ...], ...]
    zmp_applicable_repair_steps: tuple[tuple[bool, ...], ...]
    zmp_violation_repair_steps: tuple[tuple[float | None, ...], ...]
    zmp_recovery_repair_steps: tuple[tuple[float | None, ...], ...]
    unplanned_contact_repair_steps: tuple[tuple[bool, ...], ...]
    lateral_roll_repair_steps: tuple[tuple[float, ...], ...]
    lateral_roll_cumulative_mean_repair_steps: tuple[tuple[float, ...], ...]
    sustained_lean_repair: tuple[bool, ...]
    intent_scales: tuple[float, ...]
    physics_scales: tuple[float, ...]
    translation_repair_scale: float
    rotation_repair_scale: float
    beta: float
    gain_contract_id: str = "FRS-GAIN-v008"

    @property
    def policy_row_count(self) -> int:
        return len(self.policy_actions)

    def validate(self) -> None:
        # B1: 校验 transaction 行对齐, 产出完整 Repair-row report boundary.
        count = self.policy_row_count
        row_fields = (
            self.scenario_ids,
            self.noisy_segment_hashes,
            self.valid_policy_row_mask,
            self.intent_remaining_noisy,
            self.intent_remaining_repaired,
            self.physics_remaining_noisy,
            self.physics_remaining_repaired,
            self.intent_channel_noisy,
            self.intent_channel_repaired,
            self.physics_channel_noisy,
            self.physics_channel_repaired,
            self.support_foot_drift_noisy,
            self.support_foot_drift_repaired,
            self.intent_gain,
            self.physics_gain,
            self.recovery_pressure,
            self.weighted_physics_gain,
            self.repair_cost,
            self.repair_penalty,
            self.cost_free_score,
            self.gain_total,
            self.policy_values,
            self.raw_advantages,
            self.expected_support_steps,
            self.contact_clean_steps,
            self.contact_noisy_steps,
            self.contact_repair_steps,
            self.zmp_clean_steps,
            self.zmp_noisy_steps,
            self.zmp_repair_steps,
            self.survival_clean_steps,
            self.survival_noisy_steps,
            self.survival_repair_steps,
            self.contact_violation_repair_steps,
            self.zmp_applicable_repair_steps,
            self.zmp_violation_repair_steps,
            self.zmp_recovery_repair_steps,
            self.unplanned_contact_repair_steps,
            self.lateral_roll_repair_steps,
            self.lateral_roll_cumulative_mean_repair_steps,
            self.sustained_lean_repair,
        )
        if not self.transaction_id or count <= 0 or any(len(value) != count for value in row_fields):
            raise ValueError("v017 local report lost transaction or Repair-row alignment")
        if any(len(action) != 6 for action in self.policy_actions) or not all(self.valid_policy_row_mask):
            raise ValueError("v017 local report requires valid full-6D Repair rows")
        scalar_fields = (
            self.intent_remaining_noisy,
            self.intent_remaining_repaired,
            self.physics_remaining_noisy,
            self.physics_remaining_repaired,
            self.intent_gain,
            self.physics_gain,
            self.recovery_pressure,
            self.weighted_physics_gain,
            self.repair_cost,
            self.repair_penalty,
            self.cost_free_score,
            self.gain_total,
            self.policy_values,
            self.raw_advantages,
        )
        if any(not all(math.isfinite(float(item)) for item in values) for values in scalar_fields):
            raise ValueError("v017 local report required scalars must be finite")
        # B2: support-foot drift 仅在 expected loaded-support exposure 上适用, N/A 必须显式保留.
        if any(
            value is not None and not math.isfinite(float(value))
            for values in (self.support_foot_drift_noisy, self.support_foot_drift_repaired)
            for value in values
        ):
            raise ValueError("v017 local report support-foot drift must be finite or semantic N/A")
        if any(len(row) != 6 or not all(math.isfinite(float(item)) for item in row) for rows in (
            self.intent_channel_noisy,
            self.intent_channel_repaired,
        ) for row in rows):
            raise ValueError("v017 local report requires finite six-channel Intent decomposition")
        if any(
            len(row) != 4
            or any(row[index] is None or not math.isfinite(float(row[index])) for index in (0, 3))
            or any(not (row[index] is None or math.isfinite(float(row[index]))) for index in (1, 2))
            for rows in (self.physics_channel_noisy, self.physics_channel_repaired)
            for row in rows
        ):
            raise ValueError("v017 local report permits N/A only for support-foot drift and phase-ZMP")
        if (
            len(self.intent_scales) != 6
            or len(self.physics_scales) != 4
            or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in self.intent_scales + self.physics_scales)
            or not math.isfinite(float(self.translation_repair_scale))
            or float(self.translation_repair_scale) <= 0.0
            or not math.isfinite(float(self.rotation_repair_scale))
            or float(self.rotation_repair_scale) <= 0.0
            or not math.isfinite(float(self.beta))
            or float(self.beta) < 0.0
        ):
            raise ValueError("v017 local report requires complete fixed Gain scale/beta identity")
        if any(value != 1 for value in self.clean_execution_count + self.noisy_execution_count):
            raise ValueError("v017 local report requires one Clean and one Noisy execution per Segment")


def build_frontres_v017_local_evaluation_report(
    evidence: FrontRESSealedRecoveryAwareGainBatch,
    gain: FrontRESRecoveryAwareGainResult,
) -> FrontRESV017LocalEvaluationReport:
    """Serialize owner evidence without recomputing any Gain component."""

    # B1: 只读取 sealed evidence 与 Gain owner 输出, 不重算任何训练标量.
    evidence.validate()
    attempts = evidence.ordered_attempts
    baseline_by_source = {value.source_index: value for value in evidence.baselines}

    def role_steps(role: str, field: str) -> tuple[Any, ...]:
        values = []
        for attempt in attempts:
            trajectory = attempt.repair if role == "repair" else getattr(baseline_by_source[attempt.source_index], role)
            value = getattr(trajectory, field)[:, 0].detach().to(device="cpu")
            if field == "zmp_margin":
                values.append(tuple(float(item) if math.isfinite(float(item)) else None for item in value.tolist()))
            else:
                values.append(tuple(value.tolist()))
        return tuple(values)

    contact_violation_rows: list[tuple[bool, ...]] = []
    zmp_applicable_rows: list[tuple[bool, ...]] = []
    zmp_violation_rows: list[tuple[float | None, ...]] = []
    zmp_recovery_rows: list[tuple[float | None, ...]] = []
    unplanned_rows: list[tuple[bool, ...]] = []
    lateral_roll_rows: list[tuple[float, ...]] = []
    cumulative_roll_rows: list[tuple[float, ...]] = []
    sustained_lean_rows: list[bool] = []
    for attempt in attempts:
        baseline = baseline_by_source[attempt.source_index]
        expected = baseline.expected_support[:, 0].detach().cpu().bool()
        repair = attempt.repair
        actual = repair.contact[:, 0].detach().cpu().bool()
        valid = repair.valid_mask[:, 0].detach().cpu().bool()
        margin = repair.zmp_margin[:, 0].detach().cpu().float()
        contact_violation = valid & torch.any(expected != actual, dim=-1)
        applicable = valid & expected.any(dim=-1) & actual.any(dim=-1)
        violation = torch.where(applicable, torch.clamp_min(-margin, 0.0), torch.full_like(margin, float("nan")))
        recovery = _v017_zmp_recovery_projection(violation, applicable)
        unplanned = _v017_unplanned_contact_transitions(expected, actual, valid, tolerance=1)
        roll = _v017_lateral_roll(repair.root_quat[:, 0].detach().cpu().float())
        cumulative = torch.cumsum(roll, dim=0) / torch.arange(1, int(roll.numel()) + 1, dtype=roll.dtype)
        sustained = bool(
            int(roll.numel()) >= 3
            and torch.all(torch.sign(roll[-3:]) == torch.sign(roll[-1]))
            and abs(float(cumulative[-1])) >= math.radians(5.0)
        )
        contact_violation_rows.append(tuple(bool(value) for value in contact_violation.tolist()))
        zmp_applicable_rows.append(tuple(bool(value) for value in applicable.tolist()))
        zmp_violation_rows.append(
            tuple(float(value) if math.isfinite(float(value)) else None for value in violation.tolist())
        )
        zmp_recovery_rows.append(tuple(float(value) if value is not None else None for value in recovery))
        unplanned_rows.append(tuple(bool(value) for value in unplanned.tolist()))
        lateral_roll_rows.append(tuple(float(value) for value in roll.tolist()))
        cumulative_roll_rows.append(tuple(float(value) for value in cumulative.tolist()))
        sustained_lean_rows.append(sustained)

    policy_values = torch.stack([value.policy_value for value in attempts]).reshape(-1)

    def finite_rows(value: torch.Tensor) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(item) for item in row) for row in value.detach().cpu().tolist())

    def optional_rows(value: torch.Tensor) -> tuple[tuple[float | None, ...], ...]:
        return tuple(
            tuple(None if math.isnan(float(item)) else float(item) for item in row)
            for row in value.detach().cpu().tolist()
        )

    def optional_scalars(value: torch.Tensor) -> tuple[float | None, ...]:
        return tuple(
            None if math.isnan(float(item)) else float(item)
            for item in value.detach().cpu().tolist()
        )

    # B2: 将适用值序列化为 finite scalar, 将 owner 的语义 N/A 序列化为 None.
    report = FrontRESV017LocalEvaluationReport(
        transaction_id=attempts[0].transaction_id,
        scenario_ids=tuple(value.scenario_id for value in attempts),
        noisy_segment_hashes=tuple(value.noisy_segment_hash for value in attempts),
        policy_actions=tuple(tuple(float(item) for item in value.policy_action.detach().cpu().tolist()) for value in attempts),
        valid_policy_row_mask=(True,) * len(attempts),
        intent_remaining_noisy=tuple(float(value) for value in gain.intent_remaining_noisy.detach().cpu().tolist()),
        intent_remaining_repaired=tuple(float(value) for value in gain.intent_remaining_repaired.detach().cpu().tolist()),
        physics_remaining_noisy=tuple(float(value) for value in gain.physics_remaining_noisy.detach().cpu().tolist()),
        physics_remaining_repaired=tuple(float(value) for value in gain.physics_remaining_repaired.detach().cpu().tolist()),
        intent_channel_noisy=finite_rows(gain.intent_channel_noisy),
        intent_channel_repaired=finite_rows(gain.intent_channel_repaired),
        physics_channel_noisy=optional_rows(gain.physics_channel_noisy),
        physics_channel_repaired=optional_rows(gain.physics_channel_repaired),
        support_foot_drift_noisy=optional_scalars(gain.support_foot_drift_noisy),
        support_foot_drift_repaired=optional_scalars(gain.support_foot_drift_repaired),
        intent_gain=tuple(float(value) for value in gain.intent_gain.detach().cpu().tolist()),
        physics_gain=tuple(float(value) for value in gain.physics_gain.detach().cpu().tolist()),
        recovery_pressure=tuple(float(value) for value in gain.recovery_pressure.detach().cpu().tolist()),
        weighted_physics_gain=tuple(float(value) for value in gain.weighted_physics_gain.detach().cpu().tolist()),
        repair_cost=tuple(float(value) for value in gain.repair_cost.detach().cpu().tolist()),
        repair_penalty=tuple(float(value) for value in gain.repair_penalty.detach().cpu().tolist()),
        cost_free_score=tuple(float(value) for value in gain.cost_free_score.detach().cpu().tolist()),
        gain_total=tuple(float(value) for value in gain.gain_total.detach().cpu().tolist()),
        policy_values=tuple(float(value) for value in policy_values.detach().cpu().tolist()),
        raw_advantages=tuple(float(value) for value in (gain.gain_total - policy_values).detach().cpu().tolist()),
        clean_execution_count=tuple(value.clean_execution_count for value in evidence.baselines),
        noisy_execution_count=tuple(value.noisy_execution_count for value in evidence.baselines),
        expected_support_steps=tuple(
            tuple(baseline_by_source[value.source_index].expected_support[:, 0].detach().cpu().tolist()) for value in attempts
        ),
        contact_clean_steps=role_steps("clean", "contact"),
        contact_noisy_steps=role_steps("noisy", "contact"),
        contact_repair_steps=role_steps("repair", "contact"),
        zmp_clean_steps=role_steps("clean", "zmp_margin"),
        zmp_noisy_steps=role_steps("noisy", "zmp_margin"),
        zmp_repair_steps=role_steps("repair", "zmp_margin"),
        survival_clean_steps=role_steps("clean", "survival"),
        survival_noisy_steps=role_steps("noisy", "survival"),
        survival_repair_steps=role_steps("repair", "survival"),
        contact_violation_repair_steps=tuple(contact_violation_rows),
        zmp_applicable_repair_steps=tuple(zmp_applicable_rows),
        zmp_violation_repair_steps=tuple(zmp_violation_rows),
        zmp_recovery_repair_steps=tuple(zmp_recovery_rows),
        unplanned_contact_repair_steps=tuple(unplanned_rows),
        lateral_roll_repair_steps=tuple(lateral_roll_rows),
        lateral_roll_cumulative_mean_repair_steps=tuple(cumulative_roll_rows),
        sustained_lean_repair=tuple(sustained_lean_rows),
        intent_scales=tuple(gain.intent_scales),
        physics_scales=tuple(gain.physics_scales),
        translation_repair_scale=float(gain.translation_repair_scale),
        rotation_repair_scale=float(gain.rotation_repair_scale),
        beta=float(gain.beta),
    )
    # B3: 最终 report 在离开 owner 前 fail closed, 产出可安全序列化的只读诊断.
    report.validate()
    return report


def _v017_zmp_recovery_projection(
    violation: torch.Tensor,
    applicable: torch.Tensor,
) -> tuple[float | None, ...]:
    """Expose post-worst-frame phase-ZMP recovery without inventing N/A values."""

    indices = torch.nonzero(applicable, as_tuple=False).flatten()
    if int(indices.numel()) == 0:
        return tuple(None for _ in range(int(violation.numel())))
    worst = int(indices[torch.argmax(violation.index_select(0, indices))].item())
    return tuple(
        float(violation[step].item()) if step >= worst and bool(applicable[step]) else None
        for step in range(int(violation.numel()))
    )


def _v017_unplanned_contact_transitions(
    expected: torch.Tensor,
    actual: torch.Tensor,
    valid: torch.Tensor,
    *,
    tolerance: int,
) -> torch.Tensor:
    """Mark actual support transitions lacking a nearby Clean-plan transition."""

    steps = int(expected.shape[0])
    expected_transition = torch.zeros(steps, dtype=torch.bool)
    actual_transition = torch.zeros(steps, dtype=torch.bool)
    if steps > 1:
        expected_transition[1:] = torch.any(expected[1:] != expected[:-1], dim=-1)
        actual_transition[1:] = torch.any(actual[1:] != actual[:-1], dim=-1)
    result = torch.zeros(steps, dtype=torch.bool)
    for step in torch.nonzero(actual_transition & valid, as_tuple=False).flatten().tolist():
        lo = max(0, int(step) - int(tolerance))
        hi = min(steps, int(step) + int(tolerance) + 1)
        result[int(step)] = not bool(expected_transition[lo:hi].any())
    return result


def _v017_lateral_roll(quat: torch.Tensor) -> torch.Tensor:
    """Return world-frame root roll for evaluation-only sustained-lean diagnostics."""

    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))


@dataclass(frozen=True)
class _V015ZMPRoleProjection:
    margins: tuple[tuple[float | None, ...], ...]
    violations: tuple[tuple[float | None, ...], ...]
    argmax_frames: tuple[int | None, ...]
    max_violations: tuple[float | None, ...]
    recovery_trajectories: tuple[tuple[float | None, ...], ...]


def _project_v015_zmp_role(
    *,
    margins: torch.Tensor,
    applicable: torch.Tensor,
    step_violation: torch.Tensor,
    pair_valid: torch.Tensor,
    horizons: torch.Tensor,
) -> _V015ZMPRoleProjection:
    """Project exact v004 ZMP intermediates into row-major JSON-safe diagnostics."""

    if not (
        margins.ndim == 2
        and tuple(margins.shape) == tuple(applicable.shape)
        and tuple(margins.shape) == tuple(step_violation.shape)
        and tuple(margins.shape) == tuple(pair_valid.shape)
        and horizons.ndim == 1
        and int(horizons.numel()) == int(margins.shape[1])
    ):
        raise ValueError("v015 ZMP diagnostic projection requires aligned [K,B] tensors and [B] horizons")
    margins_cpu = margins.detach().to(device="cpu", dtype=torch.float32)
    applicable_cpu = applicable.detach().to(device="cpu", dtype=torch.bool)
    violation_cpu = step_violation.detach().to(device="cpu", dtype=torch.float32)
    valid_cpu = pair_valid.detach().to(device="cpu", dtype=torch.bool)
    horizons_cpu = horizons.detach().to(device="cpu", dtype=torch.long)

    margin_rows: list[tuple[float | None, ...]] = []
    violation_rows: list[tuple[float | None, ...]] = []
    argmax_frames: list[int | None] = []
    max_violations: list[float | None] = []
    recovery_rows: list[tuple[float | None, ...]] = []
    for row, horizon_value in enumerate(horizons_cpu.tolist()):
        horizon = int(horizon_value)
        if horizon <= 0 or horizon > int(margins_cpu.shape[0]):
            raise ValueError("v015 ZMP diagnostic horizon exceeds sealed K evidence")
        row_margins: list[float | None] = []
        row_violations: list[float | None] = []
        applicable_indices: list[int] = []
        for step in range(horizon):
            is_valid = bool(valid_cpu[step, row])
            is_applicable = bool(applicable_cpu[step, row])
            margin = float(margins_cpu[step, row])
            violation = float(violation_cpu[step, row])
            if is_applicable and not math.isfinite(margin):
                raise ValueError("v015 ZMP diagnostic applicable margin must be finite")
            if is_applicable and (not is_valid or not math.isfinite(violation) or violation < 0.0):
                raise ValueError("v015 ZMP diagnostic applicable violation must be finite and unsaturated")
            row_margins.append(margin if is_valid and math.isfinite(margin) else None)
            row_violations.append(violation if is_applicable else None)
            if is_applicable:
                applicable_indices.append(step)
        if applicable_indices:
            argmax = max(applicable_indices, key=lambda step: float(violation_cpu[step, row]))
            max_violation = float(violation_cpu[argmax, row])
            recovery = tuple(row_violations[argmax:])
        else:
            argmax = None
            max_violation = None
            recovery = ()
        margin_rows.append(tuple(row_margins))
        violation_rows.append(tuple(row_violations))
        argmax_frames.append(argmax)
        max_violations.append(max_violation)
        recovery_rows.append(recovery)
    return _V015ZMPRoleProjection(
        margins=tuple(margin_rows),
        violations=tuple(violation_rows),
        argmax_frames=tuple(argmax_frames),
        max_violations=tuple(max_violations),
        recovery_trajectories=tuple(recovery_rows),
    )


@dataclass(frozen=True)
class FrontRESV015LocalEvaluationReport:
    """Read-only v015 local-K diagnostic projection of sealed candidate evidence.

    Status: active read-only projection for candidate and formal v015 routes.
    Upstream: Step 3B `FrontRESV015GainConsumerEvidence` after its v003 return
    carrier has been sealed.
    Downstream: terminal/diagnostic review only, never sampler, PPO, optimizer,
    checkpoint, or formal evaluator state.
    Evidence: deterministic S1/S2 Physics and transaction contracts.
    Gap: real simulator Physics values remain a bounded live gate.
    """

    transaction_id: str
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    horizon_k: tuple[int, ...]
    policy_actions: tuple[tuple[float, ...], ...]
    valid_policy_row_mask: tuple[bool, ...]
    intent_gain: tuple[float, ...]
    physics_gain: tuple[float, ...]
    repair_cost: tuple[float, ...]
    gain_total: tuple[float, ...]
    policy_values: tuple[float, ...]
    returns: tuple[float, ...]
    raw_advantages: tuple[float, ...]
    contact_constraint: tuple[float, ...]
    zmp_constraint: tuple[float, ...]
    survival_constraint: tuple[float, ...]
    contact_constraint_advantage: tuple[float, ...]
    zmp_constraint_advantage: tuple[float, ...]
    survival_constraint_advantage: tuple[float, ...]
    zmp_applicable_repaired: tuple[bool, ...]
    zmp_applicable_noisy: tuple[bool, ...]
    zmp_constraint_applicable: tuple[bool, ...]
    constraint_advantage_state: str
    repaired_success: tuple[float, ...]
    noisy_success: tuple[float, ...]
    repaired_survival: tuple[float, ...]
    noisy_survival: tuple[float, ...]
    physics_survival_quality_repaired: tuple[float, ...]
    physics_survival_quality_noisy: tuple[float, ...]
    repaired_zmp_margin: tuple[float, ...]
    noisy_zmp_margin: tuple[float, ...]
    repaired_contact: tuple[float, ...]
    noisy_contact: tuple[float, ...]
    physics_success_gain: tuple[float, ...]
    physics_survival_gain: tuple[float, ...]
    physics_zmp_gain: tuple[float, ...]
    physics_contact_gain: tuple[float, ...]
    intent_quality_repaired: tuple[float, ...]
    intent_quality_noisy: tuple[float, ...]
    physics_admissible_repaired: tuple[float, ...]
    physics_admissible_noisy: tuple[float, ...]
    physics_deficit_repaired: tuple[float, ...]
    physics_deficit_noisy: tuple[float, ...]
    utility_repaired: tuple[float, ...]
    utility_noisy: tuple[float, ...]
    repair_penalty: tuple[float, ...]
    expected_support_steps: tuple[tuple[tuple[float, float], ...], ...]
    actual_contact_repaired_steps: tuple[tuple[tuple[float, float], ...], ...]
    actual_contact_noisy_steps: tuple[tuple[tuple[float, float], ...], ...]
    zmp_margin_repaired_steps: tuple[tuple[float | None, ...], ...]
    zmp_margin_noisy_steps: tuple[tuple[float | None, ...], ...]
    zmp_applicable_steps: tuple[tuple[bool, ...], ...]
    zmp_applicable_noisy_steps: tuple[tuple[bool, ...], ...]
    support_transition_steps: tuple[tuple[bool, ...], ...]
    zmp_step_violation_repaired: tuple[tuple[float | None, ...], ...]
    zmp_step_violation_noisy: tuple[tuple[float | None, ...], ...]
    zmp_argmax_frame_repaired: tuple[int | None, ...]
    zmp_argmax_frame_noisy: tuple[int | None, ...]
    zmp_max_violation_repaired: tuple[float | None, ...]
    zmp_max_violation_noisy: tuple[float | None, ...]
    zmp_recovery_trajectory_repaired: tuple[tuple[float | None, ...], ...]
    zmp_recovery_trajectory_noisy: tuple[tuple[float | None, ...], ...]
    physics_valid_step_count: tuple[int, ...]
    policy_row_count: int
    valid_policy_row_count: int
    intent_q29_provenance: str
    intent_q29_source: str
    intent_gain_mean: float
    physics_gain_mean: float
    repair_cost_mean: float
    gain_total_mean: float
    gain_total_pos_frac: float
    gain_total_neg_frac: float
    evaluation_kind: str = _V015_LOCAL_EVALUATION_KIND
    gain_source: str = _V015_GAIN_SOURCE
    return_feedback: bool = False
    priority_feedback: bool = False
    ppo_feedback: bool = False

    def validate(self) -> None:
        """Reject non-v003, partial, or feedback-bearing local diagnostic reports."""

        count = int(self.policy_row_count)
        components = (self.intent_gain, self.physics_gain, self.repair_cost, self.gain_total)
        row_diagnostics = (
            self.policy_values,
            self.returns,
            self.raw_advantages,
            self.contact_constraint,
            self.zmp_constraint,
            self.survival_constraint,
            self.repaired_success,
            self.noisy_success,
            self.repaired_survival,
            self.noisy_survival,
            self.physics_survival_quality_repaired,
            self.physics_survival_quality_noisy,
            self.repaired_zmp_margin,
            self.noisy_zmp_margin,
            self.repaired_contact,
            self.noisy_contact,
            self.physics_success_gain,
            self.physics_survival_gain,
            self.physics_zmp_gain,
            self.physics_contact_gain,
            self.intent_quality_repaired,
            self.intent_quality_noisy,
            self.physics_admissible_repaired,
            self.physics_admissible_noisy,
            self.physics_deficit_repaired,
            self.physics_deficit_noisy,
            self.utility_repaired,
            self.utility_noisy,
            self.repair_penalty,
        )
        if (
            not self.transaction_id
            or count <= 0
            or len(self.scenario_ids) != count
            or len(self.noisy_segment_hashes) != count
            or len(self.x_t_identities) != count
            or len(self.horizon_k) != count
            or len(self.policy_actions) != count
            or any(len(row) != 6 for row in self.policy_actions)
            or len(self.valid_policy_row_mask) != count
            or len(self.zmp_applicable_repaired) != count
            or len(self.zmp_applicable_noisy) != count
            or len(self.zmp_constraint_applicable) != count
            or any(len(values) != count for values in components)
            or any(len(values) != count for values in row_diagnostics)
            or any(
                len(values) != count
                for values in (
                    self.contact_constraint_advantage,
                    self.zmp_constraint_advantage,
                    self.survival_constraint_advantage,
                )
            )
            or len(self.physics_valid_step_count) != count
            or len(self.expected_support_steps) != count
            or len(self.actual_contact_repaired_steps) != count
            or len(self.actual_contact_noisy_steps) != count
            or any(
                len(values) != count
                for values in (
                    self.zmp_margin_repaired_steps,
                    self.zmp_margin_noisy_steps,
                    self.zmp_applicable_steps,
                    self.zmp_applicable_noisy_steps,
                    self.support_transition_steps,
                    self.zmp_step_violation_repaired,
                    self.zmp_step_violation_noisy,
                    self.zmp_argmax_frame_repaired,
                    self.zmp_argmax_frame_noisy,
                    self.zmp_max_violation_repaired,
                    self.zmp_max_violation_noisy,
                    self.zmp_recovery_trajectory_repaired,
                    self.zmp_recovery_trajectory_noisy,
                )
            )
            or any(value < 0 or value > self.horizon_k[index] for index, value in enumerate(self.physics_valid_step_count))
            or any(not str(value) for value in self.scenario_ids)
            or any(not str(value) for value in self.noisy_segment_hashes)
            or any(not str(value) for value in self.x_t_identities)
            or any(int(value) <= 0 for value in self.horizon_k)
            or self.valid_policy_row_count < 0
            or self.valid_policy_row_count > count
            or self.valid_policy_row_count != sum(bool(value) for value in self.valid_policy_row_mask)
            or self.evaluation_kind != _V015_LOCAL_EVALUATION_KIND
            or self.gain_source != _V015_GAIN_SOURCE
            or self.intent_q29_provenance != "deployment_noisy_q29"
            or self.return_feedback
            or self.priority_feedback
            or self.ppo_feedback
        ):
            raise ValueError("v015 local evaluation report has invalid identity, Gain source, or feedback boundary")
        if not all(math.isfinite(float(value)) for row in self.policy_actions for value in row):
            raise ValueError("v015 local evaluation report requires finite sealed policy actions [B,6]")
        for row, row_valid in enumerate(self.valid_policy_row_mask):
            required_diagnostics = tuple(
                values for values in row_diagnostics
                if all(
                    values is not optional
                    for optional in (self.repaired_zmp_margin, self.noisy_zmp_margin, self.physics_zmp_gain)
                )
            )
            row_values = tuple(float(values[row]) for values in (*components, *required_diagnostics))
            if row_valid and not all(math.isfinite(value) for value in row_values):
                raise ValueError("v015 local evaluation report requires finite v003 components on valid policy rows")
            all_row_values = tuple(float(values[row]) for values in (*components, *row_diagnostics))
            if not row_valid and not all(math.isnan(value) for value in all_row_values):
                raise ValueError("v015 local evaluation report keeps invalid-row diagnostics UNCONFIRMED, never zero-filled")
            if self.zmp_constraint_applicable[row] != self.zmp_applicable_repaired[row]:
                raise ValueError("v015 local evaluation PPO ZMP applicability must alias the Repair role")
            repaired_finite = math.isfinite(self.repaired_zmp_margin[row])
            noisy_finite = math.isfinite(self.noisy_zmp_margin[row])
            paired_finite = math.isfinite(self.physics_zmp_gain[row])
            if repaired_finite != bool(row_valid and self.zmp_applicable_repaired[row]):
                raise ValueError("v015 local evaluation Repair ZMP must follow explicit applicability")
            if noisy_finite != bool(row_valid and self.zmp_applicable_noisy[row]):
                raise ValueError("v015 local evaluation Noisy ZMP must follow explicit applicability")
            if paired_finite != bool(
                row_valid and self.zmp_applicable_repaired[row] and self.zmp_applicable_noisy[row]
            ):
                raise ValueError("v015 local evaluation paired ZMP gain requires both role applicability masks")
        constraint_advantages = (
            self.contact_constraint_advantage,
            self.zmp_constraint_advantage,
            self.survival_constraint_advantage,
        )
        if self.constraint_advantage_state == "sealed":
            if any(not all(math.isfinite(float(value)) for value in values) for values in constraint_advantages):
                raise ValueError("v015 sealed constraint diagnostics require finite centered advantages")
        elif self.constraint_advantage_state == "unsealed":
            if any(any(math.isfinite(float(value)) for value in values) for values in constraint_advantages):
                raise ValueError("v015 unsealed constraint diagnostics must remain UNCONFIRMED")
        else:
            raise ValueError("v015 rejects unknown constraint advantage lifecycle state")
        source = self.intent_q29_source.lower()
        if not source or any(token in source for token in ("clean", "root", "global")):
            raise ValueError("v015 local evaluation report rejects non-deployment q29 provenance")
        for row, horizon in enumerate(self.horizon_k):
            step_rows = (
                self.zmp_margin_repaired_steps[row],
                self.zmp_margin_noisy_steps[row],
                self.zmp_applicable_steps[row],
                self.zmp_applicable_noisy_steps[row],
                self.support_transition_steps[row],
                self.zmp_step_violation_repaired[row],
                self.zmp_step_violation_noisy[row],
            )
            if any(len(values) != horizon for values in step_rows):
                raise ValueError("v015 ZMP diagnostics must preserve each row's exact K-step order")
            if self.zmp_applicable_repaired[row] != any(self.zmp_applicable_steps[row]):
                raise ValueError("v015 Repair ZMP aggregate applicability disagrees with its K-step evidence")
            if self.zmp_applicable_noisy[row] != any(self.zmp_applicable_noisy_steps[row]):
                raise ValueError("v015 Noisy ZMP aggregate applicability disagrees with its K-step evidence")
            for role_name in ("repaired", "noisy"):
                violations = getattr(self, f"zmp_step_violation_{role_name}")[row]
                argmax = getattr(self, f"zmp_argmax_frame_{role_name}")[row]
                maximum = getattr(self, f"zmp_max_violation_{role_name}")[row]
                recovery = getattr(self, f"zmp_recovery_trajectory_{role_name}")[row]
                role_applicability = (
                    self.zmp_applicable_steps[row]
                    if role_name == "repaired"
                    else self.zmp_applicable_noisy_steps[row]
                )
                applicable_indices = [index for index, flag in enumerate(role_applicability) if flag]
                if applicable_indices:
                    if argmax not in applicable_indices or maximum is None:
                        raise ValueError("v015 ZMP diagnostics require an applicable argmax and maximum")
                    finite_values = [float(violations[index]) for index in applicable_indices if violations[index] is not None]
                    if len(finite_values) != len(applicable_indices) or not math.isclose(
                        float(maximum), max(finite_values), rel_tol=0.0, abs_tol=1.0e-6
                    ):
                        raise ValueError("v015 ZMP diagnostic maximum disagrees with its step trajectory")
                    if tuple(recovery) != tuple(violations[int(argmax) :]):
                        raise ValueError("v015 ZMP recovery trajectory must start at the first worst frame")
                elif argmax is not None or maximum is not None or recovery:
                    raise ValueError("v015 ZMP N/A rows cannot invent argmax or recovery evidence")
        metrics = (
            self.intent_gain_mean,
            self.physics_gain_mean,
            self.repair_cost_mean,
            self.gain_total_mean,
            self.gain_total_pos_frac,
            self.gain_total_neg_frac,
        )
        if self.valid_policy_row_count > 0:
            if not all(math.isfinite(float(value)) for value in metrics):
                raise ValueError("v015 local evaluation report requires finite v003 diagnostics on valid rows")
            if (
                not 0.0 <= self.gain_total_pos_frac <= 1.0
                or not 0.0 <= self.gain_total_neg_frac <= 1.0
                or self.gain_total_pos_frac + self.gain_total_neg_frac > 1.0 + 1.0e-7
            ):
                raise ValueError("v015 local evaluation report has invalid sign-preserving Gain fractions")
        elif not all(math.isnan(float(value)) for value in metrics):
            raise ValueError("v015 local evaluation report keeps missing diagnostics UNCONFIRMED, never zero-filled")


@dataclass(frozen=True)
class FrontRESV015CompositionEvaluationProtocol:
    """Separate deployment-composition protocol with no local-training feedback channel.

    Status: protocol-only/candidate-only.
    Upstream: an explicitly named deployment reference stream, not local return
    evidence.
    Downstream: a later dedicated sequence evaluator only.
    Evidence: deterministic S1 isolation contract.
    Gap: no sequence simulator execution or composition metric is claimed here.
    """

    reference_stream_id: str
    reference_provenance: str
    frame_count: int
    femr_action_count: int
    evaluation_kind: str = _V015_COMPOSITION_EVALUATION_KIND
    return_feedback: bool = False
    priority_feedback: bool = False
    ppo_feedback: bool = False

    def validate(self) -> None:
        """Reject local-return reuse and invalid deployment-composition protocol facts."""

        if (
            not self.reference_stream_id
            or self.reference_provenance != "deployment_reference_stream"
            or self.frame_count <= 0
            or self.femr_action_count < 0
            or self.femr_action_count > self.frame_count
            or self.evaluation_kind != _V015_COMPOSITION_EVALUATION_KIND
            or self.return_feedback
            or self.priority_feedback
            or self.ppo_feedback
        ):
            raise ValueError("v015 composition protocol has invalid deployment identity or local-training feedback")


def build_frontres_v015_local_evaluation_report(
    candidate_evidence: Any,
    *,
    transaction_id: str,
) -> FrontRESV015LocalEvaluationReport:
    """Project one sealed v003 local candidate carrier into read-only diagnostic facts.

    函数名说明:
        `build_frontres_v015_local_evaluation_report` 是 local-K diagnostic
        projection owner. 它不重算 Gain, 不读取 Clean global metric, 不写 return,
        priority, PPO 或 sampler state.

    主链路:
        上游: Step 3B sealed candidate evidence.
        下游: local evaluator formatter 与 deterministic review contract.

    语义:
        每个 scalar 仅在 valid Repair policy rows 上聚合. 没有 valid row 时保留
        NaN/UNCONFIRMED, 绝不以 0 伪造诊断.
    """

    validate = getattr(candidate_evidence, "validate", None)
    if not callable(validate):
        raise TypeError("v015 local evaluation requires a validated Step 3B candidate carrier")
    validate()
    return_evidence = getattr(candidate_evidence, "return_evidence", None)
    one_action = getattr(candidate_evidence, "one_action", None)
    validate_return = getattr(return_evidence, "validate", None)
    if not callable(validate_return):
        raise TypeError("v015 local evaluation requires validated v003 return evidence")
    validate_return()
    validate_one_action = getattr(one_action, "validate", None)
    if not callable(validate_one_action):
        raise TypeError("v015 local evaluation requires immutable one-action-K Physics evidence")
    validate_one_action()
    if getattr(return_evidence, "gain_source", None) != _V015_GAIN_SOURCE:
        raise ValueError("v015 local evaluation rejects legacy or unspecified Gain source")
    if not str(transaction_id):
        raise ValueError("v015 local evaluation requires sealed transaction identity")

    # B1: 读取 sealed one-row policy metadata, 不读取 mutable sampler state.
    valid_source = getattr(return_evidence, "policy_row_valid", None)
    policy_actions_source = getattr(return_evidence, "policy_actions", None)
    if not isinstance(valid_source, torch.Tensor):
        raise TypeError("v015 local evaluation requires sealed policy_row_valid")
    valid = valid_source.detach().bool().reshape(-1)
    count = int(valid.numel())
    if count <= 0:
        raise ValueError("v015 local evaluation requires at least one policy row")
    if not isinstance(policy_actions_source, torch.Tensor) or tuple(policy_actions_source.shape) != (count, 6):
        raise ValueError("v015 local evaluation requires sealed policy_actions [B,6]")
    components = (
        ("intent_gain", getattr(return_evidence, "intent_gain", None)),
        ("physics_gain", getattr(return_evidence, "physics_gain", None)),
        ("repair_cost", getattr(return_evidence, "repair_cost", None)),
        ("gain_total", getattr(return_evidence, "gain_total", None)),
    )
    if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != (count,) for _, value in components):
        raise ValueError("v015 local evaluation requires row-aligned v003 decomposition tensors")
    diagnostic_names = (
        "policy_values",
        "return_k",
        "advantage_k",
        "repaired_success",
        "noisy_success",
        "repaired_survival",
        "noisy_survival",
        "physics_survival_quality_repaired",
        "physics_survival_quality_noisy",
        "repaired_zmp_margin",
        "noisy_zmp_margin",
        "repaired_contact",
        "noisy_contact",
        "physics_success_gain",
        "physics_survival_gain",
        "physics_zmp_gain",
        "physics_contact_gain",
        "intent_quality_repaired",
        "intent_quality_noisy",
        "physics_admissible_repaired",
        "physics_admissible_noisy",
        "physics_deficit_repaired",
        "physics_deficit_noisy",
        "utility_repaired",
        "utility_noisy",
        "repair_penalty",
        "physics_valid_step_count",
        "contact_constraint",
        "zmp_constraint",
        "survival_constraint",
        "contact_constraint_advantage",
        "zmp_constraint_advantage",
        "survival_constraint_advantage",
    )
    diagnostic_tensors = {name: getattr(return_evidence, name, None) for name in diagnostic_names}
    if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != (count,) for value in diagnostic_tensors.values()):
        raise ValueError("v015 local evaluation requires complete row-aligned Physics/critic diagnostics")

    # B2: 仅在 valid rows 聚合 v003 component, invalid rows 保持 UNCONFIRMED.
    component_mean = {name: _v015_masked_mean(value, valid) for name, value in components}
    gain_total_pos_frac = _v015_masked_sign_fraction(components[3][1], valid, positive=True)
    gain_total_neg_frac = _v015_masked_sign_fraction(components[3][1], valid, positive=False)
    horizon = return_evidence.horizon_k.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if int(horizon.numel()) != count:
        raise ValueError("v015 local evaluation horizon_k must align with policy rows")

    from rsl_rl.frontres.frontres_gain_legacy import evaluate_phase_conditioned_physics

    # B3: 复用 historical v004 phase evaluator, 只暴露其逐帧 ZMP 中间量.
    expected_support = one_action.physics_expected_support_steps.detach()
    pair_valid = one_action.physics_pair_valid_mask.detach().bool()
    policy_valid = valid.to(device=pair_valid.device).unsqueeze(0)
    horizon_mask = torch.arange(pair_valid.shape[0], device=pair_valid.device).unsqueeze(1) < return_evidence.horizon_k.to(
        device=pair_valid.device
    ).reshape(1, -1)
    diagnostic_valid = pair_valid & policy_valid & horizon_mask
    phase_evaluator = evaluate_phase_conditioned_physics
    repaired_phase = phase_evaluator(
        expected_support,
        one_action.physics_contact_repaired_steps.detach(),
        one_action.physics_zmp_repaired_steps.detach(),
        diagnostic_valid,
    )
    noisy_phase = phase_evaluator(
        expected_support,
        one_action.physics_contact_noisy_steps.detach(),
        one_action.physics_zmp_noisy_steps.detach(),
        diagnostic_valid,
    )
    if not torch.equal(repaired_phase["support_transition_steps"], noisy_phase["support_transition_steps"]):
        raise ValueError("v015 ZMP diagnostic roles disagree on sealed support transitions")
    repaired_zmp = _project_v015_zmp_role(
        margins=one_action.physics_zmp_repaired_steps,
        applicable=repaired_phase["zmp_applicable_steps"],
        step_violation=repaired_phase["zmp_step_violation"],
        pair_valid=diagnostic_valid,
        horizons=return_evidence.horizon_k,
    )
    noisy_zmp = _project_v015_zmp_role(
        margins=one_action.physics_zmp_noisy_steps,
        applicable=noisy_phase["zmp_applicable_steps"],
        step_violation=noisy_phase["zmp_step_violation"],
        pair_valid=diagnostic_valid,
        horizons=return_evidence.horizon_k,
    )
    applicable_rows = tuple(
        tuple(bool(value) for value in row[: int(horizon[index])])
        for index, row in enumerate(repaired_phase["zmp_applicable_steps"].detach().permute(1, 0).cpu().tolist())
    )
    applicable_noisy_rows = tuple(
        tuple(bool(value) for value in row[: int(horizon[index])])
        for index, row in enumerate(noisy_phase["zmp_applicable_steps"].detach().permute(1, 0).cpu().tolist())
    )
    aggregate_repaired = repaired_phase["zmp_applicable_steps"].any(dim=0) & valid.to(
        device=repaired_phase["zmp_applicable_steps"].device
    )
    aggregate_noisy = noisy_phase["zmp_applicable_steps"].any(dim=0) & valid.to(
        device=noisy_phase["zmp_applicable_steps"].device
    )
    if not torch.equal(return_evidence.zmp_applicable_repaired.to(aggregate_repaired.device), aggregate_repaired):
        raise ValueError("v015 ReturnEvidence lost Repair ZMP applicability identity")
    if not torch.equal(return_evidence.zmp_applicable_noisy.to(aggregate_noisy.device), aggregate_noisy):
        raise ValueError("v015 ReturnEvidence lost Noisy ZMP applicability identity")
    transition_rows = tuple(
        tuple(bool(value) for value in row[: int(horizon[index])])
        for index, row in enumerate(repaired_phase["support_transition_steps"].detach().permute(1, 0).cpu().tolist())
    )

    # B4: 构造 immutable report, 明确声明 evaluation 不反馈训练状态.
    report = FrontRESV015LocalEvaluationReport(
        transaction_id=str(transaction_id),
        scenario_ids=tuple(str(value) for value in return_evidence.scenario_ids),
        noisy_segment_hashes=tuple(str(value) for value in return_evidence.noisy_segment_hashes),
        x_t_identities=tuple(str(value) for value in return_evidence.x_t_identities),
        horizon_k=tuple(int(value) for value in horizon.tolist()),
        policy_actions=tuple(
            tuple(float(value) for value in row)
            for row in policy_actions_source.detach().to(device="cpu", dtype=torch.float32).tolist()
        ),
        valid_policy_row_mask=tuple(bool(value) for value in valid.to(device="cpu").tolist()),
        intent_gain=tuple(float(value) for value in components[0][1].detach().to(device="cpu").tolist()),
        physics_gain=tuple(float(value) for value in components[1][1].detach().to(device="cpu").tolist()),
        repair_cost=tuple(float(value) for value in components[2][1].detach().to(device="cpu").tolist()),
        gain_total=tuple(float(value) for value in components[3][1].detach().to(device="cpu").tolist()),
        policy_values=tuple(float(value) for value in diagnostic_tensors["policy_values"].detach().cpu().tolist()),
        returns=tuple(float(value) for value in diagnostic_tensors["return_k"].detach().cpu().tolist()),
        raw_advantages=tuple(float(value) for value in diagnostic_tensors["advantage_k"].detach().cpu().tolist()),
        contact_constraint=tuple(float(value) for value in diagnostic_tensors["contact_constraint"].detach().cpu().tolist()),
        zmp_constraint=tuple(float(value) for value in diagnostic_tensors["zmp_constraint"].detach().cpu().tolist()),
        survival_constraint=tuple(float(value) for value in diagnostic_tensors["survival_constraint"].detach().cpu().tolist()),
        contact_constraint_advantage=tuple(float(value) for value in diagnostic_tensors["contact_constraint_advantage"].detach().cpu().tolist()),
        zmp_constraint_advantage=tuple(float(value) for value in diagnostic_tensors["zmp_constraint_advantage"].detach().cpu().tolist()),
        survival_constraint_advantage=tuple(float(value) for value in diagnostic_tensors["survival_constraint_advantage"].detach().cpu().tolist()),
        zmp_applicable_repaired=tuple(bool(value) for value in aggregate_repaired.detach().cpu().tolist()),
        zmp_applicable_noisy=tuple(bool(value) for value in aggregate_noisy.detach().cpu().tolist()),
        zmp_constraint_applicable=tuple(bool(value) for value in return_evidence.zmp_constraint_applicable.detach().cpu().tolist()),
        constraint_advantage_state=str(return_evidence.constraint_advantage_state),
        repaired_success=tuple(float(value) for value in diagnostic_tensors["repaired_success"].detach().cpu().tolist()),
        noisy_success=tuple(float(value) for value in diagnostic_tensors["noisy_success"].detach().cpu().tolist()),
        repaired_survival=tuple(float(value) for value in diagnostic_tensors["repaired_survival"].detach().cpu().tolist()),
        noisy_survival=tuple(float(value) for value in diagnostic_tensors["noisy_survival"].detach().cpu().tolist()),
        physics_survival_quality_repaired=tuple(
            float(value) for value in diagnostic_tensors["physics_survival_quality_repaired"].detach().cpu().tolist()
        ),
        physics_survival_quality_noisy=tuple(
            float(value) for value in diagnostic_tensors["physics_survival_quality_noisy"].detach().cpu().tolist()
        ),
        repaired_zmp_margin=tuple(float(value) for value in diagnostic_tensors["repaired_zmp_margin"].detach().cpu().tolist()),
        noisy_zmp_margin=tuple(float(value) for value in diagnostic_tensors["noisy_zmp_margin"].detach().cpu().tolist()),
        repaired_contact=tuple(float(value) for value in diagnostic_tensors["repaired_contact"].detach().cpu().tolist()),
        noisy_contact=tuple(float(value) for value in diagnostic_tensors["noisy_contact"].detach().cpu().tolist()),
        physics_success_gain=tuple(float(value) for value in diagnostic_tensors["physics_success_gain"].detach().cpu().tolist()),
        physics_survival_gain=tuple(float(value) for value in diagnostic_tensors["physics_survival_gain"].detach().cpu().tolist()),
        physics_zmp_gain=tuple(float(value) for value in diagnostic_tensors["physics_zmp_gain"].detach().cpu().tolist()),
        physics_contact_gain=tuple(float(value) for value in diagnostic_tensors["physics_contact_gain"].detach().cpu().tolist()),
        intent_quality_repaired=tuple(float(value) for value in diagnostic_tensors["intent_quality_repaired"].detach().cpu().tolist()),
        intent_quality_noisy=tuple(float(value) for value in diagnostic_tensors["intent_quality_noisy"].detach().cpu().tolist()),
        physics_admissible_repaired=tuple(float(value) for value in diagnostic_tensors["physics_admissible_repaired"].detach().cpu().tolist()),
        physics_admissible_noisy=tuple(float(value) for value in diagnostic_tensors["physics_admissible_noisy"].detach().cpu().tolist()),
        physics_deficit_repaired=tuple(float(value) for value in diagnostic_tensors["physics_deficit_repaired"].detach().cpu().tolist()),
        physics_deficit_noisy=tuple(float(value) for value in diagnostic_tensors["physics_deficit_noisy"].detach().cpu().tolist()),
        utility_repaired=tuple(float(value) for value in diagnostic_tensors["utility_repaired"].detach().cpu().tolist()),
        utility_noisy=tuple(float(value) for value in diagnostic_tensors["utility_noisy"].detach().cpu().tolist()),
        repair_penalty=tuple(float(value) for value in diagnostic_tensors["repair_penalty"].detach().cpu().tolist()),
        expected_support_steps=tuple(
            tuple(tuple(float(value) for value in step) for step in row)
            for row in one_action.physics_expected_support_steps.detach().permute(1, 0, 2).cpu().tolist()
        ),
        actual_contact_repaired_steps=tuple(
            tuple(tuple(float(value) for value in step) for step in row)
            for row in one_action.physics_contact_repaired_steps.detach().permute(1, 0, 2).cpu().tolist()
        ),
        actual_contact_noisy_steps=tuple(
            tuple(tuple(float(value) for value in step) for step in row)
            for row in one_action.physics_contact_noisy_steps.detach().permute(1, 0, 2).cpu().tolist()
        ),
        zmp_margin_repaired_steps=repaired_zmp.margins,
        zmp_margin_noisy_steps=noisy_zmp.margins,
        zmp_applicable_steps=applicable_rows,
        zmp_applicable_noisy_steps=applicable_noisy_rows,
        support_transition_steps=transition_rows,
        zmp_step_violation_repaired=repaired_zmp.violations,
        zmp_step_violation_noisy=noisy_zmp.violations,
        zmp_argmax_frame_repaired=repaired_zmp.argmax_frames,
        zmp_argmax_frame_noisy=noisy_zmp.argmax_frames,
        zmp_max_violation_repaired=repaired_zmp.max_violations,
        zmp_max_violation_noisy=noisy_zmp.max_violations,
        zmp_recovery_trajectory_repaired=repaired_zmp.recovery_trajectories,
        zmp_recovery_trajectory_noisy=noisy_zmp.recovery_trajectories,
        physics_valid_step_count=tuple(
            int(value) for value in diagnostic_tensors["physics_valid_step_count"].detach().cpu().tolist()
        ),
        policy_row_count=count,
        valid_policy_row_count=int(valid.sum().item()),
        intent_q29_provenance=str(return_evidence.intent_q29_provenance),
        intent_q29_source=str(return_evidence.intent_q29_source),
        intent_gain_mean=component_mean["intent_gain"],
        physics_gain_mean=component_mean["physics_gain"],
        repair_cost_mean=component_mean["repair_cost"],
        gain_total_mean=component_mean["gain_total"],
        gain_total_pos_frac=gain_total_pos_frac,
        gain_total_neg_frac=gain_total_neg_frac,
    )
    report.validate()
    return report


def format_frontres_v015_local_evaluation_report(report: FrontRESV015LocalEvaluationReport) -> str:
    """Format a v015 local-K report without legacy Style or Clean-global fields."""

    report.validate()
    return "\n".join(
        (
            "[FrontRES v015 Local-K Evaluation]",
            (
                "  identity: "
                f"transaction={report.transaction_id} scenarios={report.scenario_ids} "
                f"hashes={report.noisy_segment_hashes} "
                f"x_t={report.x_t_identities} K={report.horizon_k}"
            ),
            (
                "  intent: "
                f"provenance={report.intent_q29_provenance} source={report.intent_q29_source} "
                f"gain={_fmt_v015_eval_scalar(report.intent_gain_mean)}"
            ),
            (
                "  physics: "
                f"gain={_fmt_v015_eval_scalar(report.physics_gain_mean)} "
                f"success_gain={report.physics_success_gain} "
                f"survival=(repair={report.repaired_survival},noisy={report.noisy_survival},"
                f"quality_repair={report.physics_survival_quality_repaired},"
                f"quality_noisy={report.physics_survival_quality_noisy},gain={report.physics_survival_gain}) "
                f"zmp=(repair={report.repaired_zmp_margin},noisy={report.noisy_zmp_margin},gain={report.physics_zmp_gain}) "
                f"contact=(repair={report.repaired_contact},noisy={report.noisy_contact},gain={report.physics_contact_gain})"
            ),
            (
                "  credit: "
                f"value={report.policy_values} return={report.returns} raw_advantage={report.raw_advantages}"
            ),
            (
                "  constraints: "
                f"contact={report.contact_constraint} zmp={report.zmp_constraint} "
                f"survival={report.survival_constraint} zmp_applicable={report.zmp_constraint_applicable}"
            ),
            f"  repair: cost={_fmt_v015_eval_scalar(report.repair_cost_mean)}",
            (
                "  total: "
                f"gain={_fmt_v015_eval_scalar(report.gain_total_mean)} "
                f"positive={_fmt_v015_eval_percent(report.gain_total_pos_frac)} "
                f"negative={_fmt_v015_eval_percent(report.gain_total_neg_frac)} "
                f"valid_rows={report.valid_policy_row_count}/{report.policy_row_count}"
            ),
            "  boundary: candidate_only=1 return_feedback=0 priority_feedback=0 ppo_feedback=0",
        )
    )


def build_frontres_v015_composition_evaluation_protocol(
    *,
    reference_stream_id: str,
    frame_count: int,
    femr_action_count: int,
    reference_provenance: str = "deployment_reference_stream",
) -> FrontRESV015CompositionEvaluationProtocol:
    """Declare a separate deployment-composition evaluation without local feedback reuse."""

    protocol = FrontRESV015CompositionEvaluationProtocol(
        reference_stream_id=str(reference_stream_id),
        reference_provenance=str(reference_provenance),
        frame_count=int(frame_count),
        femr_action_count=int(femr_action_count),
    )
    protocol.validate()
    return protocol


def format_frontres_v015_composition_evaluation_protocol(
    protocol: FrontRESV015CompositionEvaluationProtocol,
) -> str:
    """Format the protocol boundary; no sequence execution is implied."""

    protocol.validate()
    return "\n".join(
        (
            "[FrontRES v015 Deployment Composition Protocol]",
            f"  reference: id={protocol.reference_stream_id} provenance={protocol.reference_provenance}",
            f"  execution: frames={protocol.frame_count} femr_actions={protocol.femr_action_count}",
            "  boundary: local_return_feedback=0 replay_priority_feedback=0 ppo_feedback=0",
        )
    )


def _v015_masked_mean(value: torch.Tensor, valid: torch.Tensor) -> float:
    data = value.detach().float().reshape(-1)
    if tuple(data.shape) != tuple(valid.shape):
        raise ValueError("v015 diagnostic component mask must align with values")
    selected = data[valid]
    if selected.numel() == 0:
        return float("nan")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("v015 diagnostic component is nonfinite on a valid policy row")
    return float(selected.mean().cpu().item())


def _v015_masked_sign_fraction(value: torch.Tensor, valid: torch.Tensor, *, positive: bool) -> float:
    data = value.detach().float().reshape(-1)
    if tuple(data.shape) != tuple(valid.shape):
        raise ValueError("v015 diagnostic positivity mask must align with values")
    selected = data[valid]
    if selected.numel() == 0:
        return float("nan")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("v015 diagnostic total Gain is nonfinite on a valid policy row")
    sign_mask = selected > 0.0 if positive else selected < 0.0
    return float(sign_mask.float().mean().cpu().item())


def _fmt_v015_eval_scalar(value: float) -> str:
    return f"{float(value):.6f}" if math.isfinite(float(value)) else "UNCONFIRMED"


def _fmt_v015_eval_percent(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%" if math.isfinite(float(value)) else "UNCONFIRMED"
