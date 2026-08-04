"""Active FRS-GAIN-v007 Clean-anchored Recovery-Aware scalar owner."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

@dataclass(frozen=True)
class FrontRESRecoveryAwareGainConfig:
    """Fixed semantic units for the active FRS-GAIN-v007 scalar owner."""

    beta: float = 0.02
    root_orientation_scale: float = 0.087
    joint_pose_scale: float = 0.087
    key_body_pose_scale: float = 0.10
    root_linear_velocity_scale: float = 0.75
    root_angular_velocity_scale: float = 2.0
    root_height_scale: float = 0.05
    contact_phase_scale: float = 0.10
    support_foot_drift_scale: float = 0.03
    phase_zmp_scale: float = 0.02
    survival_scale: float = 0.10
    translation_repair_scale: float = 0.10
    rotation_repair_scale: float = math.radians(5.0)
    contact_timing_tolerance: int = 1

    def validate(self) -> None:
        scales = (
            self.root_orientation_scale,
            self.joint_pose_scale,
            self.key_body_pose_scale,
            self.root_linear_velocity_scale,
            self.root_angular_velocity_scale,
            self.root_height_scale,
            self.contact_phase_scale,
            self.support_foot_drift_scale,
            self.phase_zmp_scale,
            self.survival_scale,
            self.translation_repair_scale,
            self.rotation_repair_scale,
        )
        if not math.isfinite(float(self.beta)) or float(self.beta) < 0.0:
            raise ValueError("FRS-GAIN-v007 beta must be finite and non-negative")
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in scales):
            raise ValueError("FRS-GAIN-v007 semantic scales must be finite and positive")
        if int(self.contact_timing_tolerance) < 0:
            raise ValueError("FRS-GAIN-v007 contact timing tolerance must be non-negative")


@dataclass(frozen=True)
class FrontRESRecoveryAwareGainInput:
    """Executed Clean/Noisy/Repair K evidence for one row per Repair attempt."""

    clean_joint_pos: torch.Tensor
    noisy_joint_pos: torch.Tensor
    repaired_joint_pos: torch.Tensor
    clean_root_pos: torch.Tensor
    noisy_root_pos: torch.Tensor
    repaired_root_pos: torch.Tensor
    clean_root_quat: torch.Tensor
    noisy_root_quat: torch.Tensor
    repaired_root_quat: torch.Tensor
    clean_key_body_pos: torch.Tensor
    noisy_key_body_pos: torch.Tensor
    repaired_key_body_pos: torch.Tensor
    clean_root_lin_vel: torch.Tensor
    noisy_root_lin_vel: torch.Tensor
    repaired_root_lin_vel: torch.Tensor
    clean_root_ang_vel: torch.Tensor
    noisy_root_ang_vel: torch.Tensor
    repaired_root_ang_vel: torch.Tensor
    clean_foot_pos: torch.Tensor
    noisy_foot_pos: torch.Tensor
    repaired_foot_pos: torch.Tensor
    expected_support: torch.Tensor
    clean_contact: torch.Tensor
    noisy_contact: torch.Tensor
    repaired_contact: torch.Tensor
    clean_zmp_margin: torch.Tensor
    noisy_zmp_margin: torch.Tensor
    repaired_zmp_margin: torch.Tensor
    clean_survival: torch.Tensor
    noisy_survival: torch.Tensor
    repaired_survival: torch.Tensor
    clean_valid_mask: torch.Tensor
    noisy_valid_mask: torch.Tensor
    repaired_valid_mask: torch.Tensor
    repair_actions: torch.Tensor


@dataclass(frozen=True)
class FrontRESRecoveryAwareGainResult:
    """Complete scalar ordering and falsifiable per-family diagnostics."""

    intent_remaining_noisy: torch.Tensor
    intent_remaining_repaired: torch.Tensor
    physics_remaining_noisy: torch.Tensor
    physics_remaining_repaired: torch.Tensor
    intent_channel_noisy: torch.Tensor
    intent_channel_repaired: torch.Tensor
    physics_channel_noisy: torch.Tensor
    physics_channel_repaired: torch.Tensor
    support_foot_drift_noisy: torch.Tensor
    support_foot_drift_repaired: torch.Tensor
    intent_gain: torch.Tensor
    physics_gain: torch.Tensor
    recovery_pressure: torch.Tensor
    weighted_physics_gain: torch.Tensor
    repair_cost: torch.Tensor
    repair_penalty: torch.Tensor
    cost_free_score: torch.Tensor
    gain_total: torch.Tensor
    intent_scales: tuple[float, ...]
    physics_scales: tuple[float, ...]
    translation_repair_scale: float
    rotation_repair_scale: float
    beta: float
    scalar_target_id: str = "clean-anchored-recovery-aware-gain-v1"
    physics_schema_id: str = "clean-anchored-contact-zmp-survival-v1"

    @property
    def available(self) -> torch.Tensor:
        return torch.isfinite(self.gain_total)


def compute_recovery_aware_gain(
    evidence: FrontRESRecoveryAwareGainInput,
    *,
    config: FrontRESRecoveryAwareGainConfig,
) -> FrontRESRecoveryAwareGainResult:
    """Compute the unique FRS-GAIN-v007 Clean-anchored scalar.

    B1 validates the complete executed evidence. B2 constructs normalized
    channel and family remaining problems. B3 applies the accepted signed
    Noisy-to-Repair ordering and full-6D intervention cost.
    """

    # B1: Clean/Noisy/Repair 必须共享 K x B identity; 缺失或非有限的必需证据直接失败.
    config.validate()
    k_steps, batch_size = _validate_recovery_aware_gain_input(evidence)
    clean_valid = evidence.clean_valid_mask.bool()
    noisy_valid = evidence.noisy_valid_mask.bool()
    repaired_valid = evidence.repaired_valid_mask.bool()

    # B2: 先在固定物理单位内计算每个剩余问题, 再用同一 smooth-worst owner 聚合家族.
    intent_noisy = _recovery_intent_channels(
        evidence,
        role="noisy",
        clean_valid=clean_valid,
        role_valid=noisy_valid,
        config=config,
    )
    intent_repaired = _recovery_intent_channels(
        evidence,
        role="repaired",
        clean_valid=clean_valid,
        role_valid=repaired_valid,
        config=config,
    )
    physics_noisy = _recovery_physics_channels(
        evidence,
        role="noisy",
        clean_valid=clean_valid,
        role_valid=noisy_valid,
        config=config,
    )
    physics_repaired = _recovery_physics_channels(
        evidence,
        role="repaired",
        clean_valid=clean_valid,
        role_valid=repaired_valid,
        config=config,
    )
    support_foot_drift_noisy = physics_noisy[:, 1] * float(config.support_foot_drift_scale)
    support_foot_drift_repaired = physics_repaired[:, 1] * float(config.support_foot_drift_scale)
    intent_remaining_noisy = _smooth_worst_rows(intent_noisy, family="Intent")
    intent_remaining_repaired = _smooth_worst_rows(intent_repaired, family="Intent")
    physics_remaining_noisy = _smooth_worst_rows(physics_noisy, family="Physics")
    physics_remaining_repaired = _smooth_worst_rows(physics_repaired, family="Physics")

    # B3: Noisy 定义零点, 平均剩余 Physics 压力调节 Physics 改善, cost 只惩罚干预大小.
    intent_gain = intent_remaining_noisy - intent_remaining_repaired
    physics_gain = physics_remaining_noisy - physics_remaining_repaired
    recovery_pressure = 0.5 * (physics_remaining_noisy + physics_remaining_repaired)
    weighted_physics_gain = recovery_pressure * physics_gain
    repair_cost = _full6_repair_cost(evidence.repair_actions, config=config)
    repair_penalty = float(config.beta) * repair_cost
    cost_free_score = intent_gain + weighted_physics_gain
    gain_total = cost_free_score - repair_penalty
    if not bool(torch.isfinite(gain_total).all()):
        raise ValueError("FRS-GAIN-v007 produced a non-finite required scalar")
    return FrontRESRecoveryAwareGainResult(
        intent_remaining_noisy=intent_remaining_noisy,
        intent_remaining_repaired=intent_remaining_repaired,
        physics_remaining_noisy=physics_remaining_noisy,
        physics_remaining_repaired=physics_remaining_repaired,
        intent_channel_noisy=intent_noisy,
        intent_channel_repaired=intent_repaired,
        physics_channel_noisy=physics_noisy,
        physics_channel_repaired=physics_repaired,
        support_foot_drift_noisy=support_foot_drift_noisy,
        support_foot_drift_repaired=support_foot_drift_repaired,
        intent_gain=intent_gain,
        physics_gain=physics_gain,
        recovery_pressure=recovery_pressure,
        weighted_physics_gain=weighted_physics_gain,
        repair_cost=repair_cost,
        repair_penalty=repair_penalty,
        cost_free_score=cost_free_score,
        gain_total=gain_total,
        intent_scales=(
            float(config.root_orientation_scale),
            float(config.joint_pose_scale),
            float(config.key_body_pose_scale),
            float(config.root_linear_velocity_scale),
            float(config.root_angular_velocity_scale),
            float(config.root_height_scale),
        ),
        physics_scales=(
            float(config.contact_phase_scale),
            float(config.support_foot_drift_scale),
            float(config.phase_zmp_scale),
            float(config.survival_scale),
        ),
        translation_repair_scale=float(config.translation_repair_scale),
        rotation_repair_scale=float(config.rotation_repair_scale),
        beta=float(config.beta),
    )


def _validate_recovery_aware_gain_input(evidence: FrontRESRecoveryAwareGainInput) -> tuple[int, int]:
    if not isinstance(evidence, FrontRESRecoveryAwareGainInput):
        raise TypeError("FRS-GAIN-v007 requires FrontRESRecoveryAwareGainInput")
    anchor = evidence.clean_joint_pos
    if anchor.ndim != 3 or int(anchor.shape[-1]) != 29 or int(anchor.shape[0]) <= 0 or int(anchor.shape[1]) <= 0:
        raise ValueError("FRS-GAIN-v007 joint trajectories must start with [K,B,29]")
    k_steps, batch_size = int(anchor.shape[0]), int(anchor.shape[1])
    shapes = {
        "noisy_joint_pos": (k_steps, batch_size, 29),
        "repaired_joint_pos": (k_steps, batch_size, 29),
        "clean_root_pos": (k_steps, batch_size, 3),
        "noisy_root_pos": (k_steps, batch_size, 3),
        "repaired_root_pos": (k_steps, batch_size, 3),
        "clean_root_quat": (k_steps, batch_size, 4),
        "noisy_root_quat": (k_steps, batch_size, 4),
        "repaired_root_quat": (k_steps, batch_size, 4),
        "clean_root_lin_vel": (k_steps, batch_size, 3),
        "noisy_root_lin_vel": (k_steps, batch_size, 3),
        "repaired_root_lin_vel": (k_steps, batch_size, 3),
        "clean_root_ang_vel": (k_steps, batch_size, 3),
        "noisy_root_ang_vel": (k_steps, batch_size, 3),
        "repaired_root_ang_vel": (k_steps, batch_size, 3),
        "clean_foot_pos": (k_steps, batch_size, 2, 3),
        "noisy_foot_pos": (k_steps, batch_size, 2, 3),
        "repaired_foot_pos": (k_steps, batch_size, 2, 3),
        "expected_support": (k_steps, batch_size, 2),
        "clean_contact": (k_steps, batch_size, 2),
        "noisy_contact": (k_steps, batch_size, 2),
        "repaired_contact": (k_steps, batch_size, 2),
        "clean_zmp_margin": (k_steps, batch_size),
        "noisy_zmp_margin": (k_steps, batch_size),
        "repaired_zmp_margin": (k_steps, batch_size),
        "clean_survival": (k_steps, batch_size),
        "noisy_survival": (k_steps, batch_size),
        "repaired_survival": (k_steps, batch_size),
        "clean_valid_mask": (k_steps, batch_size),
        "noisy_valid_mask": (k_steps, batch_size),
        "repaired_valid_mask": (k_steps, batch_size),
        "repair_actions": (batch_size, 6),
    }
    for name, shape in shapes.items():
        value = getattr(evidence, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"FRS-GAIN-v007 {name} must have shape {shape}, got {getattr(value, 'shape', None)}")
    key_shape = tuple(evidence.clean_key_body_pos.shape)
    if (
        len(key_shape) != 4
        or key_shape[:2] != (k_steps, batch_size)
        or key_shape[-1] != 3
        or int(key_shape[2]) <= 0
        or tuple(evidence.noisy_key_body_pos.shape) != key_shape
        or tuple(evidence.repaired_key_body_pos.shape) != key_shape
    ):
        raise ValueError("FRS-GAIN-v007 key-body trajectories must share [K,B,J,3]")
    masks = (
        evidence.clean_valid_mask.bool(),
        evidence.noisy_valid_mask.bool(),
        evidence.repaired_valid_mask.bool(),
    )
    if any(not bool(mask.any(dim=0).all()) for mask in masks):
        raise ValueError("FRS-GAIN-v007 requires at least one observed step per role and Repair row")
    binary = (
        evidence.expected_support,
        evidence.clean_contact,
        evidence.noisy_contact,
        evidence.repaired_contact,
        evidence.clean_survival,
        evidence.noisy_survival,
        evidence.repaired_survival,
    )
    if any(bool(((value != 0) & (value != 1)).any()) for value in binary):
        raise ValueError("FRS-GAIN-v007 Contact/support/survival evidence must be binary")
    zmp_roles = (
        ("clean", evidence.clean_zmp_margin, evidence.clean_contact, evidence.clean_valid_mask),
        ("noisy", evidence.noisy_zmp_margin, evidence.noisy_contact, evidence.noisy_valid_mask),
        ("repaired", evidence.repaired_zmp_margin, evidence.repaired_contact, evidence.repaired_valid_mask),
    )
    for name, zmp, contact, valid in zmp_roles:
        applicable = valid.bool() & evidence.expected_support.bool().any(dim=-1) & contact.bool().any(dim=-1)
        finite = torch.isfinite(zmp.float())
        if not bool(finite[applicable].all()) or bool(finite[~applicable].any()):
            raise ValueError(f"FRS-GAIN-v007 {name} ZMP must be finite exactly on loaded-support valid steps")
    required_finite = (
        evidence.clean_joint_pos,
        evidence.noisy_joint_pos,
        evidence.repaired_joint_pos,
        evidence.clean_root_pos,
        evidence.noisy_root_pos,
        evidence.repaired_root_pos,
        evidence.clean_root_quat,
        evidence.noisy_root_quat,
        evidence.repaired_root_quat,
        evidence.clean_key_body_pos,
        evidence.noisy_key_body_pos,
        evidence.repaired_key_body_pos,
        evidence.clean_root_lin_vel,
        evidence.noisy_root_lin_vel,
        evidence.repaired_root_lin_vel,
        evidence.clean_root_ang_vel,
        evidence.noisy_root_ang_vel,
        evidence.repaired_root_ang_vel,
        evidence.clean_foot_pos,
        evidence.noisy_foot_pos,
        evidence.repaired_foot_pos,
        evidence.repair_actions,
    )
    if any(not bool(torch.isfinite(value.float()).all()) for value in required_finite):
        raise ValueError("FRS-GAIN-v007 required execution evidence must be finite")
    return k_steps, batch_size


def _recovery_intent_channels(
    evidence: FrontRESRecoveryAwareGainInput,
    *,
    role: str,
    clean_valid: torch.Tensor,
    role_valid: torch.Tensor,
    config: FrontRESRecoveryAwareGainConfig,
) -> torch.Tensor:
    role_joint = getattr(evidence, f"{role}_joint_pos")
    role_root_pos = getattr(evidence, f"{role}_root_pos")
    role_root_quat = getattr(evidence, f"{role}_root_quat")
    role_body = getattr(evidence, f"{role}_key_body_pos")
    role_lin = getattr(evidence, f"{role}_root_lin_vel")
    role_ang = getattr(evidence, f"{role}_root_ang_vel")
    paired_valid = clean_valid & role_valid
    root_orientation = _late_weighted_mean(
        _quat_geodesic(evidence.clean_root_quat.float(), role_root_quat.float()),
        paired_valid,
        hold_after_invalid=True,
    )
    joint_pose = _late_weighted_mean(
        torch.sqrt(torch.mean((role_joint.float() - evidence.clean_joint_pos.float()).square(), dim=-1)),
        paired_valid,
        hold_after_invalid=True,
    )
    horizontal_shift = (role_root_pos[..., :2] - evidence.clean_root_pos[..., :2]).unsqueeze(-2)
    body_delta = role_body.float() - evidence.clean_key_body_pos.float()
    body_delta = body_delta.clone()
    body_delta[..., :2] -= horizontal_shift
    key_body_pose = _late_weighted_mean(
        torch.linalg.vector_norm(body_delta, dim=-1).mean(dim=-1),
        paired_valid,
        hold_after_invalid=True,
    )
    clean_lin_local = _quat_rotate_inverse(evidence.clean_root_quat, evidence.clean_root_lin_vel)
    role_lin_local = _quat_rotate_inverse(role_root_quat, role_lin)
    linear_velocity = _late_weighted_mean(
        torch.linalg.vector_norm(role_lin_local - clean_lin_local, dim=-1),
        paired_valid,
        hold_after_invalid=True,
    )
    clean_ang_local = _quat_rotate_inverse(evidence.clean_root_quat, evidence.clean_root_ang_vel)
    role_ang_local = _quat_rotate_inverse(role_root_quat, role_ang)
    angular_velocity = _late_weighted_mean(
        torch.linalg.vector_norm(role_ang_local - clean_ang_local, dim=-1),
        paired_valid,
        hold_after_invalid=True,
    )
    root_height = _late_weighted_mean(
        (role_root_pos[..., 2] - evidence.clean_root_pos[..., 2]).abs(),
        paired_valid,
        hold_after_invalid=True,
    )
    return torch.stack(
        (
            root_orientation / float(config.root_orientation_scale),
            joint_pose / float(config.joint_pose_scale),
            key_body_pose / float(config.key_body_pose_scale),
            linear_velocity / float(config.root_linear_velocity_scale),
            angular_velocity / float(config.root_angular_velocity_scale),
            root_height / float(config.root_height_scale),
        ),
        dim=-1,
    )


def _recovery_physics_channels(
    evidence: FrontRESRecoveryAwareGainInput,
    *,
    role: str,
    clean_valid: torch.Tensor,
    role_valid: torch.Tensor,
    config: FrontRESRecoveryAwareGainConfig,
) -> torch.Tensor:
    contact = getattr(evidence, f"{role}_contact").bool()
    foot_pos = getattr(evidence, f"{role}_foot_pos")
    zmp = getattr(evidence, f"{role}_zmp_margin")
    survival = getattr(evidence, f"{role}_survival").bool()
    valid = clean_valid & role_valid
    contact_mismatch = _contact_mismatch_with_tolerance(
        evidence.expected_support.bool(),
        contact,
        valid,
        tolerance=int(config.contact_timing_tolerance),
    )
    valid_foot_exposure = valid.unsqueeze(-1).expand_as(contact_mismatch)
    contact_den = valid_foot_exposure.float().sum(dim=(0, 2))
    if bool((contact_den <= 0).any()):
        raise ValueError("FRS-GAIN-v007 Contact exposure cannot be empty")
    contact_problem = contact_mismatch.float().sum(dim=(0, 2)) / contact_den

    expected_loaded = evidence.expected_support.bool() & valid.unsqueeze(-1)
    foot_error = torch.linalg.vector_norm(foot_pos.float() - evidence.clean_foot_pos.float(), dim=-1)
    support_drift = _late_weighted_mean(
        _masked_foot_mean(foot_error, expected_loaded),
        expected_loaded.any(dim=-1),
        hold_after_invalid=True,
    )

    zmp_applicable = valid & evidence.expected_support.bool().any(dim=-1) & contact.any(dim=-1)
    zmp_problem = _late_weighted_optional_mean(torch.relu(-zmp.float()), zmp_applicable)
    survival_den = clean_valid.float().sum(dim=0)
    if bool((survival_den <= 0).any()):
        raise ValueError("FRS-GAIN-v007 survival exposure cannot be empty")
    survived = (survival & clean_valid).float().sum(dim=0)
    survival_problem = 1.0 - survived / survival_den
    return torch.stack(
        (
            contact_problem / float(config.contact_phase_scale),
            support_drift / float(config.support_foot_drift_scale),
            zmp_problem / float(config.phase_zmp_scale),
            survival_problem / float(config.survival_scale),
        ),
        dim=-1,
    )


def _late_weighted_mean(values: torch.Tensor, valid: torch.Tensor, *, hold_after_invalid: bool) -> torch.Tensor:
    if tuple(values.shape) != tuple(valid.shape) or values.ndim != 2:
        raise ValueError("FRS-GAIN-v007 continuous channel requires aligned [K,B] values/mask")
    work = values.float()
    mask = valid.bool()
    if hold_after_invalid:
        held: list[torch.Tensor] = []
        held_mask: list[torch.Tensor] = []
        last = torch.zeros(work.shape[1], device=work.device, dtype=work.dtype)
        seen = torch.zeros(work.shape[1], device=work.device, dtype=torch.bool)
        for step in range(int(work.shape[0])):
            current_valid = mask[step]
            last = torch.where(current_valid, work[step], last)
            seen |= current_valid
            held.append(last.clone())
            held_mask.append(seen.clone())
        if not bool(seen.all()):
            raise ValueError("FRS-GAIN-v007 continuous channel has no valid observation")
        work = torch.stack(held, dim=0)
        mask = torch.stack(held_mask, dim=0)
    tau = torch.arange(1, work.shape[0] + 1, device=work.device, dtype=work.dtype).unsqueeze(1)
    tau = tau / float(work.shape[0])
    weights = tau * mask.to(dtype=work.dtype)
    denominator = weights.sum(dim=0)
    if bool((denominator <= 0).any()):
        raise ValueError("FRS-GAIN-v007 continuous channel has empty weighted exposure")
    return (work * weights).sum(dim=0) / denominator


def _late_weighted_optional_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if tuple(values.shape) != tuple(valid.shape):
        raise ValueError("FRS-GAIN-v007 optional channel requires aligned values/mask")
    tau = torch.arange(1, values.shape[0] + 1, device=values.device, dtype=values.dtype).unsqueeze(1)
    tau = tau / float(values.shape[0])
    weights = tau * valid.to(dtype=values.dtype)
    denominator = weights.sum(dim=0)
    numerator = torch.where(valid, values, torch.zeros_like(values)).mul(weights).sum(dim=0)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(values.dtype).eps),
        torch.full_like(denominator, float("nan")),
    )


def _masked_foot_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.float().sum(dim=-1)
    return torch.where(
        count > 0,
        torch.where(mask, values, torch.zeros_like(values)).sum(dim=-1) / count.clamp_min(1.0),
        torch.zeros_like(count),
    )


def _contact_mismatch_with_tolerance(
    expected: torch.Tensor,
    actual: torch.Tensor,
    valid: torch.Tensor,
    *,
    tolerance: int,
) -> torch.Tensor:
    if tuple(expected.shape) != tuple(actual.shape) or tuple(expected.shape[:2]) != tuple(valid.shape):
        raise ValueError("FRS-GAIN-v007 Contact phase requires aligned [K,B,2] evidence")
    k_steps = int(expected.shape[0])
    aligned = torch.zeros_like(expected, dtype=torch.bool)
    for delta in range(-int(tolerance), int(tolerance) + 1):
        source = torch.arange(k_steps, device=expected.device) + delta
        in_range = (source >= 0) & (source < k_steps)
        source_index = source.clamp(0, max(k_steps - 1, 0))
        aligned |= (actual == expected.index_select(0, source_index)) & in_range.view(-1, 1, 1)
    return valid.unsqueeze(-1) & ~aligned


def _smooth_worst_rows(channels: torch.Tensor, *, family: str) -> torch.Tensor:
    if channels.ndim != 2 or int(channels.shape[1]) <= 0:
        raise ValueError(f"FRS-GAIN-v007 {family} channels must be [B,J]")
    finite = torch.isfinite(channels)
    if not bool(finite.any(dim=1).all()):
        raise ValueError(f"FRS-GAIN-v007 {family} family has no applicable channel")
    masked = torch.where(finite, channels, torch.full_like(channels, float("-inf")))
    count = finite.sum(dim=1).to(dtype=channels.dtype)
    return torch.logsumexp(masked, dim=1) - torch.log(count)


def _full6_repair_cost(actions: torch.Tensor, *, config: FrontRESRecoveryAwareGainConfig) -> torch.Tensor:
    translation = torch.linalg.vector_norm(actions[:, :3].float(), dim=-1) / float(config.translation_repair_scale)
    rotation = torch.linalg.vector_norm(actions[:, 3:6].float(), dim=-1) / float(config.rotation_repair_scale)
    return torch.sqrt(translation.square() + rotation.square())


def _quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q = quaternion.float()
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    w = q[..., :1]
    xyz = -q[..., 1:]
    return vector.float() + 2.0 * (
        xyz.cross(xyz.cross(vector.float(), dim=-1) + w * vector.float(), dim=-1)
    )


def _quat_geodesic(reference: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    reference = reference / reference.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    observed = observed / observed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ref_inv = reference.clone()
    ref_inv[..., 1:] = -ref_inv[..., 1:]
    w1, x1, y1, z1 = ref_inv.unbind(dim=-1)
    w2, x2, y2, z2 = observed.unbind(dim=-1)
    relative_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    relative_xyz = torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )
    return 2.0 * torch.atan2(relative_xyz.norm(dim=-1), relative_w.abs().clamp_min(1e-8))
