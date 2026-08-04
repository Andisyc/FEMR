"""Sequence-evaluation schemas, v015 composition, and legacy planning helpers.

Status: E-FI-46 connects ordinary NPZ plus fixed protocol to one deterministic
carrier. E-FI-28--E-FI-30 connect its NPZ/protocol identity through the
command-owned deployment carrier, per-frame FEMR, frozen GMT, and the immutable
no-feedback report. The older plan/reset helpers below remain legacy v002 and
are rejected by the v015 runner boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rsl_rl.runners.frontres_deployment_carrier import (
    FrontRESV015DeploymentCarrier,
    FrontRESV015DeploymentCarrierLifecycle,
    FrontRESV015DeploymentCompositionConfig,
    FrontRESV015DeploymentCompositionRequest,
    FrontRESV015DeploymentCompositionRunConfig,
    FrontRESV015NoTrainingFeedback,
    FrontRESV015PersistentCorruptionProtocol,
    FRONTRES_V015_DEPLOYMENT_COMPOSITION_KIND,
    FRONTRES_V015_REQUIRED_NPZ_ARRAYS,
    build_frontres_v015_persistent_corruption_protocol,
    load_frontres_v015_deployment_composition_request,
    load_frontres_v015_reference_arrays,
    materialize_frontres_v015_deployment_carrier,
)

from rsl_rl.runners.frontres_evaluation_reporting import write_frontres_atomic_json


@dataclass(frozen=True)
class FrontRESV015DeploymentBranchReport:
    """One Baseline or Repair branch captured from the same route-start state."""

    role: str
    route_start_state_hash: str
    per_frame_femr_action_used: tuple[bool, ...]
    per_frame_intent_q29_error: tuple[float, ...]
    per_frame_physics_success: tuple[bool, ...]
    per_frame_fall: tuple[bool, ...]
    per_frame_zmp_margin: tuple[float | None, ...]
    per_frame_contact_consistency: tuple[float, ...]
    per_frame_policy_actions: tuple[tuple[tuple[float, ...], ...], ...]
    actual_contact_steps: tuple[tuple[tuple[bool, bool], ...], ...]
    contact_mismatch_steps: tuple[tuple[tuple[bool, bool], ...], ...]
    phase_zmp_applicable_steps: tuple[tuple[bool, ...], ...]
    phase_zmp_violation_steps: tuple[tuple[float | None, ...], ...]
    phase_zmp_recovery_steps: tuple[tuple[bool, ...], ...]
    survival_steps: tuple[tuple[bool, ...], ...]
    lateral_roll_rad_steps: tuple[tuple[float, ...], ...]
    lateral_roll_cumulative_mean_rad_steps: tuple[tuple[float, ...], ...]
    unplanned_contact_steps: tuple[tuple[bool, ...], ...]

    def validate(self, *, frame_count: int, row_count: int, role: str) -> None:
        # B1: 校验 branch role, trajectory lengths, row shapes 与 finite evidence.
        if (
            self.role != role
            or len(self.route_start_state_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in self.route_start_state_hash)
        ):
            raise ValueError(f"v015 deployment branch requires valid {role} identity")
        trajectories = (
            self.per_frame_femr_action_used,
            self.per_frame_intent_q29_error,
            self.per_frame_physics_success,
            self.per_frame_fall,
            self.per_frame_zmp_margin,
            self.per_frame_contact_consistency,
            self.per_frame_policy_actions,
            self.actual_contact_steps,
            self.contact_mismatch_steps,
            self.phase_zmp_applicable_steps,
            self.phase_zmp_violation_steps,
            self.phase_zmp_recovery_steps,
            self.survival_steps,
            self.lateral_roll_rad_steps,
            self.lateral_roll_cumulative_mean_rad_steps,
            self.unplanned_contact_steps,
        )
        if any(not isinstance(values, tuple) or len(values) != frame_count for values in trajectories):
            raise ValueError(f"v015 {role} trajectories must align with every evaluated frame")
        row_trajectories = trajectories[6:]
        if any(len(frame) != row_count for values in row_trajectories for frame in values):
            raise ValueError(f"v015 {role} trajectories lost row alignment")
        if any(len(action) != 6 for frame in self.per_frame_policy_actions for action in frame):
            raise ValueError(f"v015 {role} policy actions must be [T,B,6]")
        if any(
            len(contact) != 2
            for values in (self.actual_contact_steps, self.contact_mismatch_steps)
            for frame in values
            for contact in frame
        ):
            raise ValueError(f"v015 {role} Contact trajectories must be [T,B,2]")
        if role == "baseline":
            if any(self.per_frame_femr_action_used):
                raise ValueError("v015 Baseline must never invoke FEMR")
            if any(
                float(value) != 0.0
                for frame in self.per_frame_policy_actions
                for row in frame
                for value in row
            ):
                raise ValueError("v015 Baseline policy action must be exact zero")


@dataclass(frozen=True)
class FrontRESV015DeploymentCompositionReport(FrontRESV015NoTrainingFeedback):
    """Per-frame deployment-only metrics, separate from local K Gain and training."""

    request: FrontRESV015DeploymentCompositionRequest
    baseline: FrontRESV015DeploymentBranchReport
    repair: FrontRESV015DeploymentBranchReport
    route_start_state_hash: str
    expected_contact_steps: tuple[tuple[tuple[bool, bool], ...], ...]
    evaluation_kind: str = FRONTRES_V015_DEPLOYMENT_COMPOSITION_KIND

    @property
    def frame_count(self) -> int:
        return self.request.frame_count - max(self.request.future_offsets)

    @property
    def reference_frame_count(self) -> int:
        return self.request.frame_count

    @property
    def femr_action_count(self) -> int:
        return sum(bool(value) for value in self.repair.per_frame_femr_action_used)

    @property
    def accumulated_failure_count(self) -> int:
        return sum(not bool(value) for value in self.repair.per_frame_physics_success)

    @property
    def baseline_accumulated_failure_count(self) -> int:
        return sum(not bool(value) for value in self.baseline.per_frame_physics_success)

    @property
    def paired_intent_error_improvement(self) -> float:
        baseline = sum(self.baseline.per_frame_intent_q29_error) / max(self.frame_count, 1)
        return baseline - self.mean_intent_q29_error

    @property
    def mean_intent_q29_error(self) -> float:
        return sum(self.repair.per_frame_intent_q29_error) / max(self.frame_count, 1)

    @property
    def contact_preservation_fraction(self) -> float:
        total = self.frame_count * len(self.repair.contact_mismatch_steps[0])
        return 1.0 - sum(
            any(foot) for frame in self.repair.contact_mismatch_steps for foot in frame
        ) / max(total, 1)

    @property
    def phase_zmp_applicable_count(self) -> int:
        return sum(bool(value) for frame in self.repair.phase_zmp_applicable_steps for value in frame)

    @property
    def phase_zmp_violation_count(self) -> int:
        return sum(
            value is not None and float(value) > 0.0
            for frame in self.repair.phase_zmp_violation_steps
            for value in frame
        )

    @property
    def survival_fraction(self) -> float:
        total = self.frame_count * len(self.repair.survival_steps[0])
        return sum(bool(value) for frame in self.repair.survival_steps for value in frame) / max(total, 1)

    @property
    def max_abs_cumulative_lateral_roll_rad(self) -> float:
        return max(
            abs(float(value))
            for frame in self.repair.lateral_roll_cumulative_mean_rad_steps
            for value in frame
        )

    @property
    def unplanned_contact_event_count(self) -> int:
        return sum(bool(value) for frame in self.repair.unplanned_contact_steps for value in frame)

    def validate(self) -> None:
        # B1: 校验 request/carrier, paired route identity, metrics 与 no-feedback facts.
        self.request.validate()
        if self.evaluation_kind != FRONTRES_V015_DEPLOYMENT_COMPOSITION_KIND:
            raise ValueError("v015 deployment report has an invalid evaluation kind")
        if (
            not isinstance(self.baseline, FrontRESV015DeploymentBranchReport)
            or not isinstance(self.repair, FrontRESV015DeploymentBranchReport)
            or self.baseline.role != "baseline"
            or self.repair.role != "repair"
            or len(self.route_start_state_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in self.route_start_state_hash)
            or self.baseline.route_start_state_hash != self.route_start_state_hash
            or self.repair.route_start_state_hash != self.route_start_state_hash
        ):
            raise ValueError("v015 deployment report requires same-state Baseline and Repair branches")
        rows = (
            self.repair.per_frame_femr_action_used,
            self.repair.per_frame_intent_q29_error,
            self.repair.per_frame_physics_success,
            self.repair.per_frame_fall,
            self.repair.per_frame_zmp_margin,
            self.repair.per_frame_contact_consistency,
        )
        if any(not isinstance(values, tuple) or len(values) != self.frame_count for values in rows):
            raise ValueError("v015 deployment report per-frame length must equal its unclamped evaluated frame count")
        rich_rows = (
            self.repair.per_frame_policy_actions,
            self.expected_contact_steps,
            self.repair.actual_contact_steps,
            self.repair.contact_mismatch_steps,
            self.repair.phase_zmp_applicable_steps,
            self.repair.phase_zmp_violation_steps,
            self.repair.phase_zmp_recovery_steps,
            self.repair.survival_steps,
            self.repair.lateral_roll_rad_steps,
            self.repair.lateral_roll_cumulative_mean_rad_steps,
            self.repair.unplanned_contact_steps,
        )
        if any(not isinstance(values, tuple) or len(values) != self.frame_count for values in rich_rows):
            raise ValueError("v015 deployment quality trajectories must align with every evaluated frame")
        batch_sizes = {len(values[0]) for values in rich_rows if values}
        if len(batch_sizes) != 1 or next(iter(batch_sizes), 0) <= 0:
            raise ValueError("v015 deployment quality trajectories must share one positive row count")
        batch_size = next(iter(batch_sizes))
        self.baseline.validate(frame_count=self.frame_count, row_count=batch_size, role="baseline")
        self.repair.validate(frame_count=self.frame_count, row_count=batch_size, role="repair")
        if any(len(frame) != batch_size for values in rich_rows for frame in values):
            raise ValueError("v015 deployment quality trajectories lost row alignment")
        if any(len(action) != 6 for frame in self.repair.per_frame_policy_actions for action in frame):
            raise ValueError("v015 deployment policy actions must be [T,B,6]")
        contact_rows = (
            self.expected_contact_steps,
            self.repair.actual_contact_steps,
            self.repair.contact_mismatch_steps,
        )
        if any(len(contact) != 2 for values in contact_rows for frame in values for contact in frame):
            raise ValueError("v015 deployment Contact trajectories must be [T,B,2]")
        boolean_rows = (
            self.expected_contact_steps,
            self.repair.actual_contact_steps,
            self.repair.contact_mismatch_steps,
            self.repair.phase_zmp_applicable_steps,
            self.repair.phase_zmp_recovery_steps,
            self.repair.survival_steps,
            self.repair.unplanned_contact_steps,
        )
        if any(
            type(value) is not bool
            for rows_ in boolean_rows
            for frame in rows_
            for row in frame
            for value in (row if isinstance(row, tuple) else (row,))
        ):
            raise ValueError("v015 deployment Contact/ZMP/survival trajectories must contain bool values")
        if any(type(value) is not bool for value in self.repair.per_frame_femr_action_used):
            raise ValueError("v015 per-frame FEMR action flags must be bool")
        if any(type(value) is not bool for value in self.repair.per_frame_physics_success + self.repair.per_frame_fall):
            raise ValueError("v015 per-frame physics success/fall flags must be bool")
        if any(
            success and fall
            for success, fall in zip(
                self.repair.per_frame_physics_success,
                self.repair.per_frame_fall,
                strict=True,
            )
        ):
            raise ValueError("v015 composition frame cannot report physics success and fall together")
        numeric_rows = (self.repair.per_frame_intent_q29_error, self.repair.per_frame_contact_consistency)
        if any(not math.isfinite(float(value)) for values in numeric_rows for value in values):
            raise ValueError("v015 deployment report metrics must be finite")
        if any(value is not None and not math.isfinite(float(value)) for value in self.repair.per_frame_zmp_margin):
            raise ValueError("v015 deployment ZMP margins must be finite or explicit N/A")
        if any(float(value) < 0.0 for value in self.repair.per_frame_intent_q29_error):
            raise ValueError("v015 per-frame q29 intent error must be nonnegative")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.repair.per_frame_contact_consistency):
            raise ValueError("v015 per-frame contact consistency must be in [0,1]")
        finite_values = (
            value
            for rows_ in (
                self.repair.per_frame_policy_actions,
                self.repair.lateral_roll_rad_steps,
                self.repair.lateral_roll_cumulative_mean_rad_steps,
            )
            for frame in rows_
            for row in frame
            for value in (row if isinstance(row, tuple) else (row,))
        )
        if any(not math.isfinite(float(value)) for value in finite_values):
            raise ValueError("v015 deployment action/lean trajectories must be finite")
        if any(
            value is not None and (not math.isfinite(float(value)) or float(value) < 0.0)
            for frame in self.repair.phase_zmp_violation_steps
            for value in frame
        ):
            raise ValueError("v015 phase-ZMP violation must be nonnegative or explicit N/A")
        if any(
            bool(applicable) != (value is not None)
            for app_frame, value_frame in zip(
                self.repair.phase_zmp_applicable_steps,
                self.repair.phase_zmp_violation_steps,
                strict=True,
            )
            for applicable, value in zip(app_frame, value_frame, strict=True)
        ):
            raise ValueError("v015 phase-ZMP violation N/A must exactly match applicability")


def _collect_frontres_v015_deployment_branch(
    gateway: Any,
    *,
    config: FrontRESV015DeploymentCompositionRunConfig,
    use_femr: bool,
    route_start_state_hash: str,
) -> FrontRESV015DeploymentBranchReport:
    """Execute one isolated deployment branch and return immutable evidence.

    Status: S2B formal offline connector. This path performs deterministic
    FEMR inference and frozen-GMT execution but has no training-feedback call.
    """

    # B1: 验证 request 与 Clean expected-Physics, 产出一条已安装的 deployment branch.
    if not isinstance(config, FrontRESV015DeploymentCompositionRunConfig):
        raise TypeError("v015 composition executor requires its dedicated run config")
    config.validate()
    request = load_frontres_v015_deployment_composition_request(config.request_config)
    training_before = gateway.training_state_fingerprint()
    # B2: 分配逐帧 trajectory buffers, 产出本 branch 唯一的证据收集容器.
    femr_used: list[bool] = []
    intent_error: list[float] = []
    physics_success: list[bool] = []
    fall: list[bool] = []
    policy_actions: list[torch.Tensor] = []
    actual_contacts: list[torch.Tensor] = []
    zmp_margins: list[torch.Tensor] = []
    survival_rows: list[torch.Tensor] = []
    lateral_roll_rows: list[torch.Tensor] = []
    evaluated_frames = request.frame_count - max(request.future_offsets)
    arrays = load_frontres_v015_reference_arrays(Path(request.source_reference_path))
    expected_sequence, envelope_sequence = gateway.expected_physics(
        request,
        clean_body_pos=torch.as_tensor(arrays["body_pos_w"], device=gateway.device, dtype=torch.float32),
        clean_body_quat=torch.as_tensor(arrays["body_quat_w"], device=gateway.device, dtype=torch.float32),
    )
    gateway.prepare_metrics()
    alive = torch.ones(gateway.row_count, device=gateway.device, dtype=torch.bool)
    gateway.set_sequence(request)
    try:
        # B3: 按 frame 执行 zero/Repair correction -> frozen GMT -> simulator, 产出逐帧 Intent/Physics 证据.
        with gateway.inference_mode(), torch.inference_mode():
            for frame_index in range(evaluated_frames):
                snapshot = gateway.read_context()
                cursors = snapshot.frame_indices
                if not torch.equal(cursors, torch.full_like(cursors, frame_index)):
                    raise RuntimeError(
                        "v015 composition frame/cursor identity diverged: "
                        f"expected={frame_index} got={cursors.detach().cpu().tolist()}"
                    )
                raw_obs = gateway.read_policy_observation()
                actor_obs = gateway.normalize_observation(gateway.build_observation(raw_obs, snapshot=snapshot))
                correction = gateway.correction(actor_obs, use_femr=use_femr)
                gateway.apply_correction(correction)

                corrected_raw = gateway.read_policy_observation()
                corrected_obs = gateway.normalize_observation(
                    gateway.build_observation(corrected_raw, snapshot=snapshot)
                )
                motor_action = gateway.gmt_action(corrected_obs, correction)
                dones, infos = gateway.step(motor_action)
                frame_metrics = gateway.frame_metrics(
                    frame_index=frame_index,
                    dones=dones,
                    infos=infos,
                    expected_support=expected_sequence[frame_index].unsqueeze(0).expand(int(dones.numel()), -1),
                    expected_support_envelope=envelope_sequence[frame_index].unsqueeze(0).expand(
                        int(dones.numel()), -1
                    ),
                )
                executed_q29 = gateway.executed_q29()
                q29_error = (executed_q29 - snapshot.intent_q29[:, 0]).abs().mean()

                femr_used.append(bool(use_femr))
                intent_error.append(float(q29_error.item()))
                fall.append(bool(frame_metrics.fall.any().item()))
                policy_actions.append(correction.detach().clone())
                actual_contacts.append(frame_metrics.actual_contact.detach().clone())
                zmp_margins.append(frame_metrics.zmp_margin.detach().clone())
                alive = alive & ~frame_metrics.fall
                survival_rows.append(alive.detach().clone())
                lateral_roll_rows.append(frame_metrics.lateral_roll_rad.detach().clone())
                if frame_index + 1 < evaluated_frames:
                    gateway.advance_sequence()
    finally:
        gateway.clear_sequence()

    training_after = gateway.training_state_fingerprint()
    if training_after != training_before:
        changed = tuple(name for name in training_before if training_before[name] != training_after.get(name))
        raise RuntimeError(f"v015 composition mutated forbidden training state: {changed}")
    # B4: 按 expected Contact phase 归约逐帧证据, 产出 Contact/ZMP/survival trajectory facts.
    expected_steps = expected_sequence[:evaluated_frames].unsqueeze(1).expand(-1, gateway.row_count, -1)
    actual_steps = torch.stack(actual_contacts, dim=0)
    margin_steps = torch.stack(zmp_margins, dim=0)
    valid_steps = torch.ones_like(margin_steps, dtype=torch.bool)
    phase = gateway.evaluate_phase(
        expected_steps,
        actual_steps,
        margin_steps,
        valid_steps,
        dt=1.0 / float(request.fps),
    )
    mismatch = phase["contact_mismatch_steps"].bool()
    applicable = phase["zmp_applicable_steps"].bool()
    violation = phase["zmp_step_violation"].float()
    survival = torch.stack(survival_rows, dim=0)
    roll = torch.stack(lateral_roll_rows, dim=0)
    roll_cumulative = roll.cumsum(dim=0) / torch.arange(
        1, evaluated_frames + 1, device=roll.device, dtype=roll.dtype
    ).unsqueeze(1)
    recovery = expected_steps.any(dim=-1) & actual_steps.any(dim=-1) & ~applicable
    unplanned = gateway.unplanned_contact_steps(expected_steps, actual_steps)
    row_success = survival & ~mismatch.any(dim=-1) & ~(applicable & (violation > 0.0))
    physics_success.extend(bool(value) for value in row_success.all(dim=1).detach().cpu().tolist())
    contact_consistency = (1.0 - mismatch.float().mean(dim=(1, 2))).detach().cpu().tolist()
    zmp_margin: list[float | None] = []
    for frame_margin, frame_applicable in zip(margin_steps, applicable, strict=True):
        values = frame_margin[frame_applicable]
        zmp_margin.append(None if int(values.numel()) == 0 else float(values.mean().item()))

    def nested(value: torch.Tensor) -> tuple:
        return tuple(_frontres_v015_tuple_tree(item) for item in value.detach().cpu().tolist())

    zmp_violation_steps = tuple(
        tuple(float(violation[t, b].item()) if bool(applicable[t, b]) else None for b in range(int(applicable.shape[1])))
        for t in range(evaluated_frames)
    )
    # B5: 封装并校验 branch identity 与 trajectory, 产出无训练反馈的 immutable branch report.
    branch = FrontRESV015DeploymentBranchReport(
        role="repair" if use_femr else "baseline",
        route_start_state_hash=route_start_state_hash,
        per_frame_femr_action_used=tuple(femr_used),
        per_frame_intent_q29_error=tuple(intent_error),
        per_frame_physics_success=tuple(physics_success),
        per_frame_fall=tuple(fall),
        per_frame_zmp_margin=tuple(zmp_margin),
        per_frame_contact_consistency=tuple(contact_consistency),
        per_frame_policy_actions=nested(torch.stack(policy_actions, dim=0)),
        actual_contact_steps=nested(actual_steps.bool()),
        contact_mismatch_steps=nested(mismatch),
        phase_zmp_applicable_steps=nested(applicable),
        phase_zmp_violation_steps=zmp_violation_steps,
        phase_zmp_recovery_steps=nested(recovery),
        survival_steps=nested(survival),
        lateral_roll_rad_steps=nested(roll),
        lateral_roll_cumulative_mean_rad_steps=nested(roll_cumulative),
        unplanned_contact_steps=nested(unplanned),
    )
    return branch


def run_frontres_v015_deployment_composition_eval(
    runner: Any,
    *,
    config: FrontRESV015DeploymentCompositionRunConfig,
) -> FrontRESV015DeploymentCompositionReport:
    """Run same-state Baseline and Repair branches, then atomically write one paired report."""

    from rsl_rl.runners.frontres_runtime import FrontRESV015DeploymentRuntimeGateway

    # B1: 解析 request 并捕获 canonical route-start, 产出 Baseline/Repair 共享的比较起点.
    config.validate()
    request = load_frontres_v015_deployment_composition_request(config.request_config)
    gateway = FrontRESV015DeploymentRuntimeGateway.from_runner(runner)
    row_count = gateway.row_count
    comparison_signature = hashlib.sha256(
        f"paired-composition:{request.reference_file_hash}:{request.corruption_protocol.protocol_hash}".encode("ascii")
    ).hexdigest()
    route_start = gateway.capture_route_start(comparison_signature=comparison_signature)
    branches: dict[str, FrontRESV015DeploymentBranchReport] = {}
    try:
        # B2: 每条 route 恢复同一起点并执行一次 branch collector, 产出 paired branch evidence.
        for role, use_femr in (("baseline", False), ("repair", True)):
            restored_hash = gateway.restore_route_start(route_start, comparison_signature=comparison_signature)
            if restored_hash != route_start.initial_state_hash:
                raise RuntimeError(f"v015 composition {role} route-start state identity drifted")
            branches[role] = _collect_frontres_v015_deployment_branch(
                gateway,
                config=config,
                use_femr=use_femr,
                route_start_state_hash=route_start.initial_state_hash,
            )
    finally:
        gateway.clear_sequence()
        gateway.restore_route_start(route_start, comparison_signature=comparison_signature)

    baseline = branches["baseline"]
    repair = branches["repair"]
    arrays = load_frontres_v015_reference_arrays(Path(request.source_reference_path))
    expected_sequence, _envelope = gateway.expected_physics(
        request,
        clean_body_pos=torch.as_tensor(arrays["body_pos_w"], device=gateway.device, dtype=torch.float32),
        clean_body_quat=torch.as_tensor(arrays["body_quat_w"], device=gateway.device, dtype=torch.float32),
    )
    expected_steps = expected_sequence[: len(repair.per_frame_femr_action_used)].unsqueeze(1).expand(
        -1, row_count, -1
    )

    def nested(value: torch.Tensor) -> tuple:
        return tuple(_frontres_v015_tuple_tree(item) for item in value.detach().cpu().tolist())

    # B3: 合并 paired identities 与 Repair 轨迹, 校验后原子写入 composition report.
    report = FrontRESV015DeploymentCompositionReport(
        request=request,
        baseline=baseline,
        repair=repair,
        route_start_state_hash=route_start.initial_state_hash,
        expected_contact_steps=nested(expected_steps.bool()),
    )
    report.validate()
    _write_frontres_v015_deployment_composition_report(report, Path(config.report_path))
    return report


def _frontres_v015_tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_frontres_v015_tuple_tree(item) for item in value)
    return value


def _write_frontres_v015_deployment_composition_report(
    report: FrontRESV015DeploymentCompositionReport,
    path: Path,
) -> None:
    """Serialize one validated paired composition report through an atomic file boundary."""

    # B1: 验证 report 并定义 branch serializer, 产出稳定的 JSON branch schema.
    report.validate()

    def branch_payload(branch: FrontRESV015DeploymentBranchReport) -> dict[str, Any]:
        # B1: 将 immutable branch record 投影为 JSON-compatible trajectory payload.
        return {
            "role": branch.role,
            "route_start_state_hash": branch.route_start_state_hash,
            "per_frame_femr_action_used": list(branch.per_frame_femr_action_used),
            "per_frame_intent_q29_error": list(branch.per_frame_intent_q29_error),
            "per_frame_physics_success": list(branch.per_frame_physics_success),
            "per_frame_fall": list(branch.per_frame_fall),
            "per_frame_zmp_margin": list(branch.per_frame_zmp_margin),
            "per_frame_contact_consistency": list(branch.per_frame_contact_consistency),
            "per_frame_policy_actions": branch.per_frame_policy_actions,
            "actual_contact_steps": branch.actual_contact_steps,
            "contact_mismatch_steps": branch.contact_mismatch_steps,
            "phase_zmp_applicable_steps": branch.phase_zmp_applicable_steps,
            "phase_zmp_violation_steps": branch.phase_zmp_violation_steps,
            "phase_zmp_recovery_steps": branch.phase_zmp_recovery_steps,
            "survival_steps": branch.survival_steps,
            "evaluation_only_sustained_lean": {
                "lateral_roll_rad": branch.lateral_roll_rad_steps,
                "cumulative_mean_rad": branch.lateral_roll_cumulative_mean_rad_steps,
            },
            "unplanned_contact_steps": branch.unplanned_contact_steps,
        }

    # B2: 组装 paired routes 与兼容字段, 产出最终 versioned report payload.
    payload = {
        "evaluation_kind": report.evaluation_kind,
        "source_reference_path": report.request.source_reference_path,
        "source_reference_file_hash": report.request.source_reference_file_hash,
        "reference_path": report.request.reference_path,
        "reference_stream_id": report.request.reference_stream_id,
        "reference_file_hash": report.request.reference_file_hash,
        "reference_provenance": report.request.reference_provenance,
        "reference_frame_count": report.reference_frame_count,
        "evaluated_frame_count": report.frame_count,
        "future_offsets": list(report.request.future_offsets),
        "corruption_id": report.request.corruption_protocol.corruption_id,
        "corruption_family": report.request.corruption_protocol.family,
        "corruption_seed": report.request.corruption_protocol.seed,
        "corruption_parameters": dict(report.request.corruption_protocol.parameters),
        "corruption_temporal_mode": report.request.corruption_protocol.temporal_mode,
        "corruption_protocol_hash": report.request.corruption_protocol.protocol_hash,
        "route_start_state_hash": report.route_start_state_hash,
        "routes": {
            "baseline": branch_payload(report.baseline),
            "repair": branch_payload(report.repair),
        },
        "paired": {
            "intent_q29_error_improvement": report.paired_intent_error_improvement,
            "failure_count_improvement": (
                report.baseline_accumulated_failure_count - report.accumulated_failure_count
            ),
        },
        "femr_action_count": report.femr_action_count,
        "accumulated_failure_count": report.accumulated_failure_count,
        "per_frame_femr_action_used": list(report.repair.per_frame_femr_action_used),
        "per_frame_intent_q29_error": list(report.repair.per_frame_intent_q29_error),
        "per_frame_physics_success": list(report.repair.per_frame_physics_success),
        "per_frame_fall": list(report.repair.per_frame_fall),
        "per_frame_zmp_margin": list(report.repair.per_frame_zmp_margin),
        "per_frame_contact_consistency": list(report.repair.per_frame_contact_consistency),
        "per_frame_policy_actions": report.repair.per_frame_policy_actions,
        "expected_contact_steps": report.expected_contact_steps,
        "actual_contact_steps": report.repair.actual_contact_steps,
        "contact_mismatch_steps": report.repair.contact_mismatch_steps,
        "phase_zmp_applicable_steps": report.repair.phase_zmp_applicable_steps,
        "phase_zmp_violation_steps": report.repair.phase_zmp_violation_steps,
        "phase_zmp_recovery_steps": report.repair.phase_zmp_recovery_steps,
        "survival_steps": report.repair.survival_steps,
        "evaluation_only_sustained_lean": {
            "lateral_roll_rad": report.repair.lateral_roll_rad_steps,
            "cumulative_mean_rad": report.repair.lateral_roll_cumulative_mean_rad_steps,
        },
        "unplanned_contact_steps": report.repair.unplanned_contact_steps,
        "summary": {
            "mean_intent_q29_error": report.mean_intent_q29_error,
            "contact_preservation_fraction": report.contact_preservation_fraction,
            "phase_zmp_applicable_count": report.phase_zmp_applicable_count,
            "phase_zmp_violation_count": report.phase_zmp_violation_count,
            "survival_fraction": report.survival_fraction,
            "max_abs_cumulative_lateral_roll_rad": report.max_abs_cumulative_lateral_roll_rad,
            "unplanned_contact_event_count": report.unplanned_contact_event_count,
        },
        "return_feedback": False,
        "priority_feedback": False,
        "ppo_feedback": False,
        "sampler_feedback": False,
        "optimizer_feedback": False,
    }
    # B3: 拒绝覆盖旧结果, 通过 temporary -> replace 原子提交唯一 report.
    write_frontres_atomic_json(path, payload)
