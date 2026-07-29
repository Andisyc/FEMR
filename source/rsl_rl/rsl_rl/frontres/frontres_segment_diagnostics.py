from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Callable

import torch

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_diagnostics",
    Path(__file__).resolve().with_name("frontres_formal_runtime_probe.py"),
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


FORBIDDEN_ACCEPTANCE_KEYS = {
    "acceptance_gt",
    "acceptance_mask",
    "acceptance_margin",
    "acceptance_prob",
}

_V004_PROJECTION_STATUSES = {
    "INTENT_FEASIBLE",
    "PROJECTED_INTENT",
    "CONSTRAINT_RECOVERY",
    "NO_EMPIRICAL_DIRECTION",
    "NO_COMMON_FIRST_ORDER_DESCENT",
}
_V004_CONSTRAINT_FAMILIES = {"contact", "zmp", "survival"}


@dataclass(frozen=True)
class FrontRESV004ActualUpdateTelemetry:
    projection_status: str
    actual_projection_status: str
    active_families: tuple[str, ...]
    directional_derivatives: dict[str, float]
    kkt_max_violation: float
    gradient_kkt_max_violation: float
    optimizer_candidate_actor_delta_l2: float
    committed_actor_delta_l2: float
    actor_optimizer_state_preserved: bool
    actor_loss_weight: float


def validate_frontres_v004_actual_update_telemetry(
    diagnostics: Mapping[str, Any],
    *,
    tolerance: float,
) -> FrontRESV004ActualUpdateTelemetry:
    """Validate the final post-optimizer Actor authority before serialization."""

    def finite(name: str) -> float:
        if name not in diagnostics:
            raise RuntimeError(f"v015 formal result is missing {name} telemetry")
        value = float(diagnostics[name])
        if not math.isfinite(value):
            raise RuntimeError(f"v015 formal result has non-finite {name} telemetry")
        return value

    projection_status = str(diagnostics.get("constraint_projection_status", ""))
    if projection_status not in _V004_PROJECTION_STATUSES:
        raise RuntimeError(f"v015 formal result has invalid constraint projection status: {projection_status!r}")
    actual_status = str(diagnostics.get("actual_update_projection_status", ""))
    if actual_status != projection_status:
        raise RuntimeError(
            "v015 formal result actual update status disagrees with its gradient projection: "
            f"gradient={projection_status!r} actual={actual_status!r}"
        )
    active_families = tuple(str(value) for value in diagnostics.get("constraint_active_families", ()))
    if len(set(active_families)) != len(active_families) or not set(active_families) <= _V004_CONSTRAINT_FAMILIES:
        raise RuntimeError(f"v015 formal result has invalid active constraint families: {active_families}")
    raw_derivatives = diagnostics.get("constraint_directional_derivatives")
    if not isinstance(raw_derivatives, Mapping):
        raise RuntimeError("v015 formal result is missing constraint_directional_derivatives telemetry")
    derivatives = {str(key): float(value) for key, value in raw_derivatives.items()}
    if not set(derivatives) <= _V004_CONSTRAINT_FAMILIES or not all(
        math.isfinite(value) for value in derivatives.values()
    ):
        raise RuntimeError("v015 formal result has invalid constraint_directional_derivatives telemetry")

    kkt = finite("constraint_kkt_max_violation")
    if not 0.0 <= kkt <= float(tolerance):
        raise RuntimeError(
            "v015 formal result exceeds the checkpoint-v6 constraint projection tolerance: "
            f"kkt={kkt:.9g} tolerance={float(tolerance):.9g}"
        )
    observed_kkt = max((max(0.0, value) for value in derivatives.values()), default=0.0)
    if abs(observed_kkt - kkt) > float(tolerance):
        raise RuntimeError(
            "v015 formal result has inconsistent constraint KKT telemetry: "
            f"reported={kkt:.9g} observed={observed_kkt:.9g}"
        )
    gradient_kkt = finite("gradient_projection_kkt_max_violation")
    if not 0.0 <= gradient_kkt <= float(tolerance):
        raise RuntimeError("v015 formal result has invalid pre-optimizer projection KKT telemetry")
    candidate_l2 = finite("optimizer_candidate_actor_delta_l2")
    committed_l2 = finite("committed_actor_delta_l2")
    if candidate_l2 < 0.0 or committed_l2 < 0.0:
        raise RuntimeError("v015 formal result has negative Actor delta norm telemetry")
    state_preserved = diagnostics.get("actor_optimizer_state_restored")
    if not isinstance(state_preserved, bool):
        raise RuntimeError("v015 formal result is missing Actor optimizer-state authority telemetry")
    actor_loss_weight = finite("actor_loss_weight")
    must_preserve = actor_loss_weight == 0.0 or projection_status in {
        "NO_EMPIRICAL_DIRECTION",
        "NO_COMMON_FIRST_ORDER_DESCENT",
    }
    if state_preserved != must_preserve:
        raise RuntimeError("v015 formal result Actor optimizer-state preservation disagrees with projection authority")
    if must_preserve and committed_l2 != 0.0:
        raise RuntimeError("v015 formal result committed an Actor delta while Actor authority was frozen")
    if active_families and not must_preserve and committed_l2 == 0.0:
        raise RuntimeError("v015 formal result lost a permitted nonzero Actor update")

    return FrontRESV004ActualUpdateTelemetry(
        projection_status=projection_status,
        actual_projection_status=actual_status,
        active_families=active_families,
        directional_derivatives=derivatives,
        kkt_max_violation=kkt,
        gradient_kkt_max_violation=gradient_kkt,
        optimizer_candidate_actor_delta_l2=candidate_l2,
        committed_actor_delta_l2=committed_l2,
        actor_optimizer_state_preserved=state_preserved,
        actor_loss_weight=actor_loss_weight,
    )


@dataclass(frozen=True)
class FrontRESSegmentReplaySummary:
    scalars: dict[str, float]
    stage: str
    objective: str


_V015_GAIN_SOURCE = "FRS-GAIN-v006-loaded-support-zmp-applicability"
_V015_LOCAL_EVALUATION_KIND = "local_k_candidate_only"
_V015_COMPOSITION_EVALUATION_KIND = "deployment_composition_protocol"


def _v004_phase_evaluator() -> Callable[..., dict[str, torch.Tensor]]:
    """Load the active v004 evaluator so diagnostics reuse, never copy, its masks."""

    try:
        from rsl_rl.frontres.frontres_gain import evaluate_phase_conditioned_physics

        return evaluate_phase_conditioned_physics
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(
            "frontres_gain_diagnostics_runtime", Path(__file__).resolve().with_name("frontres_gain.py")
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load FRS-GAIN-v006 phase evaluator for diagnostics")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.evaluate_phase_conditioned_physics


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

    # B3: 复用 active v004 phase evaluator, 只暴露其逐帧 ZMP 中间量.
    expected_support = one_action.physics_expected_support_steps.detach()
    pair_valid = one_action.physics_pair_valid_mask.detach().bool()
    policy_valid = valid.to(device=pair_valid.device).unsqueeze(0)
    horizon_mask = torch.arange(pair_valid.shape[0], device=pair_valid.device).unsqueeze(1) < return_evidence.horizon_k.to(
        device=pair_valid.device
    ).reshape(1, -1)
    diagnostic_valid = pair_valid & policy_valid & horizon_mask
    phase_evaluator = _v004_phase_evaluator()
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


def summarize_segment_batch(
    sample: Any,
    reward_result: Any,
    reset_result: Any,
    action_stats: Any,
    sampler_stats: Any | None = None,
    stage: str = "stage3_segment_hrl",
    objective: str = "segment_replay_hrl",
) -> FrontRESSegmentReplaySummary:
    scalars: dict[str, float] = {}
    sources = tuple(getattr(sample, "source", ()))
    total = max(1, len(sources))
    scalars["segment/global_frac"] = sources.count("global") / total
    scalars["segment/replay_frac"] = sources.count("replay") / total
    scalars["segment/review_frac"] = sources.count("review") / total
    priority = getattr(sample, "priority", torch.zeros(0))
    scalars["segment/priority_mean"] = _mean(priority)
    scalars["segment/priority_p90"] = _quantile(priority, 0.9)
    if sampler_stats is not None:
        scalars["segment/replay_pool_size"] = float(getattr(sampler_stats, "replay_pool_size", 0))
    else:
        scalars["segment/replay_pool_size"] = float((priority > 0.0).sum().item()) if isinstance(priority, torch.Tensor) else 0.0

    solved = getattr(reward_result, "solved_mask", torch.zeros(0, dtype=torch.bool))
    hopeless = getattr(reward_result, "hopeless_mask", torch.zeros_like(solved))
    valid = getattr(reward_result, "valid_mask", torch.ones_like(solved))
    scalars["segment/solved_frac"] = _bool_mean(solved)
    scalars["segment/hopeless_frac"] = _bool_mean(hopeless)
    scalars["segment/active_frac"] = _bool_mean(valid & (~solved.bool()) & (~hopeless.bool())) if isinstance(valid, torch.Tensor) else 0.0
    scalars["segment/reset_success_frac"] = _bool_mean(getattr(reset_result, "success_mask", torch.zeros(0, dtype=torch.bool)))
    scalars["segment/preroll_frac"] = _bool_mean(getattr(reset_result, "preroll_mask", torch.zeros(0, dtype=torch.bool)))
    horizon = getattr(sample, "horizon_k", None)
    if horizon is None:
        horizon = getattr(reward_result, "horizon_k", None)
    scalars["segment/k"] = _mean(horizon) if isinstance(horizon, torch.Tensor) else float(horizon or 0.0)
    scalars["segment/score_noisy"] = _mean(getattr(reward_result, "score_noisy", torch.zeros(0)))
    scalars["segment/score_repaired"] = _mean(getattr(reward_result, "score_repaired", torch.zeros(0)))
    scalars["segment/score_clean"] = _mean(getattr(reward_result, "score_clean", torch.zeros(0)))
    scalars["segment/gain_over_noisy"] = _mean(getattr(reward_result, "gain_over_noisy", torch.zeros(0)))
    scalars["segment/fall_frac"] = _bool_mean(getattr(reward_result, "fall_flag", torch.zeros(0, dtype=torch.bool)))
    scalars["segment/contact_consistency"] = _mean(getattr(reward_result, "contact_consistency", torch.zeros(0)))
    scalars["segment/action_norm"] = float(getattr(action_stats, "action_norm_mean", 0.0))
    per_dim = getattr(action_stats, "per_dim_norm", torch.zeros(6))
    per_dim = per_dim.detach().flatten().float().cpu() if isinstance(per_dim, torch.Tensor) else torch.zeros(6)
    labels = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
    for i, label in enumerate(labels):
        scalars[f"segment/action_norm_{label}"] = float(per_dim[i].item()) if i < per_dim.numel() else 0.0
    # B2: Remove retired acceptance fields before terminal/logger formatting.
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    return FrontRESSegmentReplaySummary(scalars=scalars, stage=stage, objective=objective)


def format_segment_replay_log(summary: FrontRESSegmentReplaySummary) -> str:
    scalars = summary.scalars
    return (
        f"FrontRES Segment HRL active: stage={summary.stage} objective={summary.objective} "
        f"k={scalars.get('segment/k', 0.0):.0f} "
        f"mix=global:{scalars.get('segment/global_frac', 0.0):.2f}/"
        f"replay:{scalars.get('segment/replay_frac', 0.0):.2f}/"
        f"review:{scalars.get('segment/review_frac', 0.0):.2f} "
        f"gain={scalars.get('segment/gain_over_noisy', 0.0):.4f} "
        f"reset={scalars.get('segment/reset_success_frac', 0.0):.2f}"
    )


def repair_effect_summary_to_scalars(summary: dict[str, Any]) -> dict[str, float]:
    """把 canonical Gain summary 转换为 train-effect scalars.

    函数名说明:
        `repair_effect_summary_to_scalars` 是 diagnostic projection owner, 只选择和
        命名正式 Gain/sampler 字段; 它不是 Gain 公式, 也不回写训练状态.

    主链路:
        上游: live probe 和 update-loop aggregation 产生 canonical summary.
        下游: terminal/logger formatter 消费扁平 scalars.

    语义:
        诊断必须读取 Style/Physics/Repair Cost/Total Gain 的正式 owner 字段.
        缺失 motion/physics evidence 保持 UNCONFIRMED, legacy acceptance score 被忽略.
    """
    # B1: 从 final live summary 读取 canonical Gain 和 sampler fields.
    scalars = {
        "segment/train_effect_gain_style": _optional_float(summary.get("gain_style_mean")),
        "segment/train_effect_gain_physics": _optional_float(summary.get("gain_physics_mean")),
        "segment/train_effect_repair_cost": _optional_float(summary.get("gain_repair_cost_mean")),
        "segment/train_effect_gain_total": _optional_float(summary.get("gain_total_mean")),
        "segment/train_effect_gain_pos_frac": _optional_float(summary.get("gain_total_pos_frac")),
        "segment/train_effect_fall_rate": _optional_float(summary.get("done_frac")),
        "segment/train_effect_valid_frac": _optional_float(summary.get("storage_valid_frac")),
        "segment/train_effect_replay_candidates": _summary_float(
            summary,
            "sampler_update_replay_candidate_count",
            "sampler_replay_candidates",
        ),
        "segment/train_effect_replay_pool_size": _float(summary.get("sampler_replay_pool_size")),
    }
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    # B3: AUDIT-DIAG-01 截获 terminal/logger formatter 实际消费的 scalars 和 transaction 聚合状态.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-DIAG-01",
        scalars=scalars,
        audit_identity_mode=summary.get("audit_identity_mode", "UNCONFIRMED"),
        audit_transaction_count=summary.get("audit_transaction_count", 0),
        audit_transaction_ids=summary.get("audit_transaction_ids", ()),
        audit_batch_signature_count=summary.get("audit_batch_signature_count", 0),
        audit_batch_signatures=summary.get("audit_batch_signatures", ()),
        audit_same_transaction=summary.get("audit_same_transaction", False),
    )
    return scalars


def format_segment_train_effect_log(summary: dict[str, Any]) -> str:
    scalars = repair_effect_summary_to_scalars(summary)
    return "\n".join(
        (
            "[FrontRES Segment Train Effect]",
            (
                "  audit: "
                f"mode={summary.get('audit_identity_mode', 'UNCONFIRMED')} "
                f"transactions={summary.get('audit_transaction_count', 0)} "
                f"batches={summary.get('audit_batch_signature_count', 0)} "
                f"same_transaction={summary.get('audit_same_transaction', False)} "
                f"transaction_ids={summary.get('audit_transaction_ids', ())} "
                f"batch_signatures={summary.get('audit_batch_signatures', ())}"
            ),
            (
                "  gain: "
                f"style={_fmt_motion_scalar(scalars, 'segment/train_effect_gain_style')} "
                f"physics={_fmt_motion_scalar(scalars, 'segment/train_effect_gain_physics')} "
                f"repair_cost={_fmt_motion_scalar(scalars, 'segment/train_effect_repair_cost')} "
                f"total={_fmt_motion_scalar(scalars, 'segment/train_effect_gain_total')} "
                f"gain_pos={_fmt_motion_percent(scalars, 'segment/train_effect_gain_pos_frac')}"
            ),
            (
                "  data: "
                f"fall={_fmt_motion_percent(scalars, 'segment/train_effect_fall_rate')} "
                f"valid={_fmt_motion_percent(scalars, 'segment/train_effect_valid_frac')}"
            ),
            (
                "  replay: "
                f"candidates={scalars['segment/train_effect_replay_candidates']:.0f} "
                f"pool={scalars['segment/train_effect_replay_pool_size']:.0f}"
            ),
        )
    )


def motion_quality_summary_to_scalars(
    *,
    clean_positions: torch.Tensor | None = None,
    repaired_positions: torch.Tensor | None = None,
    noisy_positions: torch.Tensor | None = None,
    delta_se: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    temporal_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    valid = valid_mask.bool() if isinstance(valid_mask, torch.Tensor) else None
    temporal = temporal_mask.bool() if isinstance(temporal_mask, torch.Tensor) else None
    return {
        "segment/motion_mpjpe_repaired_clean": _mpjpe(repaired_positions, clean_positions, valid, temporal),
        "segment/motion_mpjpe_noisy_clean": _mpjpe(noisy_positions, clean_positions, valid, temporal),
        "segment/motion_vel_error_repaired_clean": _diff_mpjpe(repaired_positions, clean_positions, valid, temporal, order=1),
        "segment/motion_acc_error_repaired_clean": _diff_mpjpe(repaired_positions, clean_positions, valid, temporal, order=2),
        "segment/motion_delta_se_norm": _delta_se_norm(delta_se, None),
        "segment/motion_delta_z_up_frac": _delta_z_up_frac(delta_se, None),
    }


def format_segment_motion_quality_log(scalars: dict[str, float]) -> str:
    return "\n".join(
        (
            "[FrontRES Segment Motion Quality]",
            (
                "  pose: "
                f"mpjpe_repaired={_fmt_motion_scalar(scalars, 'segment/motion_mpjpe_repaired_clean')} "
                f"mpjpe_noisy={_fmt_motion_scalar(scalars, 'segment/motion_mpjpe_noisy_clean')}"
            ),
            (
                "  dynamics: "
                f"vel_err={_fmt_motion_scalar(scalars, 'segment/motion_vel_error_repaired_clean')} "
                f"acc_err={_fmt_motion_scalar(scalars, 'segment/motion_acc_error_repaired_clean')}"
            ),
            (
                "  action: "
                f"delta_se_norm={_fmt_motion_scalar(scalars, 'segment/motion_delta_se_norm')} "
                f"dz_up={_fmt_motion_percent(scalars, 'segment/motion_delta_z_up_frac')}"
            ),
        )
    )


def periodic_eval_summary_to_scalars(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "segment/eval_episode_length": _float(summary.get("episode_length")),
        "segment/eval_success_rate": _float(summary.get("success_rate")),
        "segment/eval_fall_rate": _float(summary.get("fall_rate")),
        "segment/eval_mean_survival_steps": _float(summary.get("mean_survival_steps")),
        "segment/eval_gain_style": _optional_float(summary.get("gain_style_mean")),
        "segment/eval_gain_physics": _optional_float(summary.get("gain_physics_mean")),
        "segment/eval_gain_repair_cost": _optional_float(summary.get("gain_repair_cost_mean")),
        "segment/eval_gain_total": _optional_float(summary.get("gain_total_mean")),
        "segment/eval_gain_total_pos_frac": _optional_float(summary.get("gain_total_pos_frac")),
        "segment/eval_motion_mpjpe_repaired_clean": _optional_float(summary.get("segment/motion_mpjpe_repaired_clean")),
        "segment/eval_motion_mpjpe_noisy_clean": _optional_float(summary.get("segment/motion_mpjpe_noisy_clean")),
        "segment/eval_motion_vel_error_repaired_clean": _optional_float(summary.get("segment/motion_vel_error_repaired_clean")),
        "segment/eval_motion_acc_error_repaired_clean": _optional_float(summary.get("segment/motion_acc_error_repaired_clean")),
        "segment/eval_motion_delta_se_norm": _optional_float(summary.get("segment/motion_delta_se_norm")),
        "segment/eval_motion_delta_z_up_frac": _optional_float(summary.get("segment/motion_delta_z_up_frac")),
    }


def action_distribution_health_summary(
    *,
    means: torch.Tensor | None = None,
    sigmas: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    supervised_target: torch.Tensor | None = None,
    raw_saturation_warn: float = 2.0,
    raw_saturation_bad: float = 20.0,
) -> dict[str, float | str | bool]:
    """Summarize whether a task-space policy distribution is numerically usable."""
    summary: dict[str, float | str | bool] = {
        "available": isinstance(means, torch.Tensor) and means.numel() > 0,
        "status": "UNCONFIRMED",
        "raw_mean_abs_max": _unconfirmed(),
        "raw_mean_abs_mean": _unconfirmed(),
        "raw_saturated_frac_abs_gt_2": _unconfirmed(),
        "raw_saturated_frac_abs_gt_20": _unconfirmed(),
        "sigma_min": _unconfirmed(),
        "sigma_mean": _unconfirmed(),
        "sigma_max": _unconfirmed(),
        "action_norm_mean": _unconfirmed(),
        "target_norm_mean": _unconfirmed(),
        "action_norm_over_target_norm": _unconfirmed(),
    }
    if not isinstance(means, torch.Tensor) or means.numel() == 0:
        return summary

    raw = means.detach().float()
    finite = torch.isfinite(raw)
    if not bool(finite.all().item()):
        summary["status"] = "BAD_NONFINITE_RAW_MEAN"
        return summary

    abs_raw = raw.abs()
    raw_abs_max = float(abs_raw.max().cpu().item())
    raw_abs_mean = float(abs_raw.mean().cpu().item())
    sat_warn = float((abs_raw > float(raw_saturation_warn)).float().mean().cpu().item())
    sat_bad = float((abs_raw > float(raw_saturation_bad)).float().mean().cpu().item())
    summary.update(
        {
            "raw_mean_abs_max": raw_abs_max,
            "raw_mean_abs_mean": raw_abs_mean,
            "raw_saturated_frac_abs_gt_2": sat_warn,
            "raw_saturated_frac_abs_gt_20": sat_bad,
        }
    )

    if isinstance(sigmas, torch.Tensor) and sigmas.numel() > 0:
        sigma = sigmas.detach().float()
        if not bool(torch.isfinite(sigma).all().item()):
            summary["status"] = "BAD_NONFINITE_SIGMA"
            return summary
        summary.update(
            {
                "sigma_min": float(sigma.min().cpu().item()),
                "sigma_mean": float(sigma.mean().cpu().item()),
                "sigma_max": float(sigma.max().cpu().item()),
            }
        )

    action_norm = _row_norm_mean(actions)
    target_norm = _row_norm_mean(supervised_target)
    summary["action_norm_mean"] = action_norm
    summary["target_norm_mean"] = target_norm
    if math.isfinite(action_norm) and math.isfinite(target_norm) and target_norm > 1e-8:
        summary["action_norm_over_target_norm"] = action_norm / target_norm

    if raw_abs_max >= float(raw_saturation_bad) or sat_bad > 0.0:
        summary["status"] = "BAD_RAW_MEAN_SATURATED"
    elif sat_warn >= 0.50:
        summary["status"] = "WARN_RAW_MEAN_SATURATING"
    else:
        summary["status"] = "OK"
    return summary


def format_segment_periodic_eval_log(summary: dict[str, Any]) -> str:
    scalars = periodic_eval_summary_to_scalars(summary)
    families = dict(summary.get("perturbation_family_counts", {}) or {})
    return "\n".join(
        (
            "[FrontRES Segment Periodic Eval]",
            (
                "  batch: "
                f"source={summary.get('eval_batch_source', 'UNCONFIRMED')} "
                f"reset={bool(summary.get('eval_reset_applied', False))} "
                f"motion_ids={tuple(summary.get('motion_ids', ()) or ())} "
                f"start_frames={tuple(summary.get('start_frames', ()) or ())}"
            ),
            (
                "  perturbation: "
                f"families={families} "
                f"strength_min={float(summary.get('perturbation_strength_min', 0.0)):.6f} "
                f"strength_mean={float(summary.get('perturbation_strength_mean', 0.0)):.6f} "
                f"strength_max={float(summary.get('perturbation_strength_max', 0.0)):.6f}"
            ),
            (
                "  rollout: "
                f"episode_length={scalars['segment/eval_episode_length']:.1f} "
                f"survival={scalars['segment/eval_mean_survival_steps']:.1f}"
            ),
            (
                "  result: "
                f"success={scalars['segment/eval_success_rate'] * 100.0:.1f}% "
                f"fall={scalars['segment/eval_fall_rate'] * 100.0:.1f}%"
            ),
            (
                "  gain: "
                f"source={summary.get('gain_source', 'UNCONFIRMED')} "
                f"style={_fmt_eval_scalar(scalars, 'segment/eval_gain_style')} "
                f"physics={_fmt_eval_scalar(scalars, 'segment/eval_gain_physics')} "
                f"repair_cost={_fmt_eval_scalar(scalars, 'segment/eval_gain_repair_cost')} "
                f"total={_fmt_eval_scalar(scalars, 'segment/eval_gain_total')} "
                f"positive={_fmt_eval_percent(scalars, 'segment/eval_gain_total_pos_frac')}"
            ),
            (
                "  motion: "
                f"mpjpe_repaired={_fmt_eval_scalar(scalars, 'segment/eval_motion_mpjpe_repaired_clean')} "
                f"mpjpe_noisy={_fmt_eval_scalar(scalars, 'segment/eval_motion_mpjpe_noisy_clean')} "
                f"vel_err={_fmt_eval_scalar(scalars, 'segment/eval_motion_vel_error_repaired_clean')} "
                f"acc_err={_fmt_eval_scalar(scalars, 'segment/eval_motion_acc_error_repaired_clean')}"
            ),
            (
                "  action: "
                f"delta_se_norm={_fmt_eval_scalar(scalars, 'segment/eval_motion_delta_se_norm')} "
                f"dz_up={_fmt_eval_percent(scalars, 'segment/eval_motion_delta_z_up_frac')}"
            ),
        )
    )


def segment_summary_to_scalars(summary: FrontRESSegmentReplaySummary) -> dict[str, float]:
    scalars = dict(summary.scalars)
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    return scalars


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())
    if value is None:
        return 0.0
    return float(value)


def _optional_float(value: Any) -> float:
    if value is None:
        return _unconfirmed()
    return _float(value)


def _summary_float(summary: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return _float(value)
    return 0.0


def _unconfirmed() -> float:
    return float("nan")


def _fmt_motion_scalar(scalars: dict[str, float], key: str) -> str:
    value = float(scalars.get(key, _unconfirmed()))
    if not math.isfinite(value):
        return "UNCONFIRMED"
    return f"{value:.6f}"


def _fmt_motion_percent(scalars: dict[str, float], key: str) -> str:
    value = float(scalars.get(key, _unconfirmed()))
    if not math.isfinite(value):
        return "UNCONFIRMED"
    return f"{value * 100.0:.1f}%"


def _fmt_eval_scalar(scalars: dict[str, float], key: str) -> str:
    return _fmt_motion_scalar(scalars, key)


def _fmt_eval_percent(scalars: dict[str, float], key: str) -> str:
    return _fmt_motion_percent(scalars, key)


def _mpjpe(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    valid: torch.Tensor | None,
    temporal: torch.Tensor | None,
) -> float:
    if not _same_position_shape(a, b):
        return _unconfirmed()
    diff = torch.linalg.norm(a.float() - b.float(), dim=-1)
    return _masked_batch_mean(diff, valid, temporal)


def _diff_mpjpe(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    valid: torch.Tensor | None,
    temporal: torch.Tensor | None,
    *,
    order: int,
) -> float:
    if not _same_position_shape(a, b) or a.shape[1] <= order:
        return _unconfirmed()
    da = torch.diff(a.float(), n=order, dim=1)
    db = torch.diff(b.float(), n=order, dim=1)
    diff_temporal = temporal[:, order:] if temporal is not None and temporal.ndim == 2 else None
    return _masked_batch_mean(torch.linalg.norm(da - db, dim=-1), valid, diff_temporal)


def _same_position_shape(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and a.ndim >= 3


def _masked_batch_mean(
    value: torch.Tensor,
    valid: torch.Tensor | None,
    temporal: torch.Tensor | None = None,
) -> float:
    if value.numel() == 0:
        return 0.0
    if valid is not None and valid.shape[0] == value.shape[0]:
        value = value[valid]
        if temporal is not None and temporal.shape[0] == valid.shape[0]:
            temporal = temporal[valid]
        if value.numel() == 0:
            return 0.0
    if temporal is not None and value.ndim >= 2 and tuple(temporal.shape) == tuple(value.shape[:2]):
        value = value[temporal]
        if value.numel() == 0:
            return 0.0
    return float(value.mean().item())


def _delta_se_norm(delta_se: torch.Tensor | None, valid: torch.Tensor | None) -> float:
    if not isinstance(delta_se, torch.Tensor) or delta_se.numel() == 0:
        return 0.0
    value = torch.linalg.norm(delta_se.float(), dim=-1)
    if valid is not None and valid.shape[0] == value.shape[0]:
        value = value[valid]
        if value.numel() == 0:
            return 0.0
    return float(value.mean().item())


def _delta_z_up_frac(delta_se: torch.Tensor | None, valid: torch.Tensor | None) -> float:
    if not isinstance(delta_se, torch.Tensor) or delta_se.ndim < 2 or delta_se.shape[-1] < 3:
        return 0.0
    value = delta_se[..., 2] > 0.0
    if valid is not None and valid.shape[0] == value.shape[0]:
        value = value[valid]
        if value.numel() == 0:
            return 0.0
    return float(value.float().mean().item())


def _row_norm_mean(value: torch.Tensor | None) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return _unconfirmed()
    data = value.detach().float()
    if data.ndim == 1:
        data = data.view(1, -1)
    if not bool(torch.isfinite(data).all().item()):
        return _unconfirmed()
    return float(torch.linalg.norm(data, dim=-1).mean().cpu().item())


def _mean(value: torch.Tensor | None) -> float:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.float().mean().item())


def _quantile(value: torch.Tensor | None, q: float) -> float:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(torch.quantile(value.float().flatten(), q).item())


def _bool_mean(value: torch.Tensor | None) -> float:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.bool().float().mean().item())
