"""Shared paired Style/Physics/Repair gain calculations for Segment Replay.

This module is deliberately independent from the environment reward.  It owns
only the repair-specific quantities that may enter Segment PPO returns,
sampler evidence, and evaluation summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Mapping

import torch

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_gain",
    Path(__file__).resolve().with_name("frontres_formal_runtime_probe.py"),
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


@dataclass(frozen=True)
class FrontRESSegmentGainConfig:
    """Named scales and weights for the accepted paired gain contract."""

    style_weight: float = 1.0
    physics_weight: float = 1.0
    repair_weight: float = 0.15
    mpjpe_scale: float = 0.10
    velocity_scale: float = 1.0
    acceleration_scale: float = 1.0
    root_orientation_scale: float = 1.0
    repair_norm_scale: float = 1.0
    repair_temporal_scale: float = 1.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | Any | None) -> "FrontRESSegmentGainConfig":
        if isinstance(cfg, Mapping):
            values = dict(cfg)
        else:
            values = {
                name: getattr(cfg, name)
                for name in (
                    "frontres_gain_style_weight",
                    "frontres_gain_physics_weight",
                    "frontres_gain_repair_weight",
                    "frontres_gain_mpjpe_scale",
                    "frontres_gain_velocity_scale",
                    "frontres_gain_acceleration_scale",
                    "frontres_gain_root_orientation_scale",
                    "frontres_gain_repair_norm_scale",
                    "frontres_gain_repair_temporal_scale",
                )
                if cfg is not None and hasattr(cfg, name)
            }
        return cls(
            style_weight=float(values.get("frontres_gain_style_weight", cls.style_weight)),
            physics_weight=float(values.get("frontres_gain_physics_weight", cls.physics_weight)),
            repair_weight=float(values.get("frontres_gain_repair_weight", cls.repair_weight)),
            mpjpe_scale=float(values.get("frontres_gain_mpjpe_scale", cls.mpjpe_scale)),
            velocity_scale=float(values.get("frontres_gain_velocity_scale", cls.velocity_scale)),
            acceleration_scale=float(values.get("frontres_gain_acceleration_scale", cls.acceleration_scale)),
            root_orientation_scale=float(values.get("frontres_gain_root_orientation_scale", cls.root_orientation_scale)),
            repair_norm_scale=float(values.get("frontres_gain_repair_norm_scale", cls.repair_norm_scale)),
            repair_temporal_scale=float(values.get("frontres_gain_repair_temporal_scale", cls.repair_temporal_scale)),
        )


@dataclass(frozen=True)
class FrontRESSegmentGainResult:
    """Per-row paired gain and component diagnostics.

    NaN means the component was not observable from the supplied capture.  It
    is intentionally not converted to zero.
    """

    style_gain: torch.Tensor
    physics_gain: torch.Tensor
    repair_cost: torch.Tensor
    gain_total: torch.Tensor
    style_mpjpe_gain: torch.Tensor
    style_velocity_gain: torch.Tensor
    style_acceleration_gain: torch.Tensor
    style_root_orientation_gain: torch.Tensor
    physics_success_gain: torch.Tensor
    physics_survival_quality_repaired: torch.Tensor
    physics_survival_quality_noisy: torch.Tensor
    physics_survival_gain: torch.Tensor
    physics_zmp_gain: torch.Tensor
    physics_contact_gain: torch.Tensor
    repair_norm: torch.Tensor
    repair_temporal_change: torch.Tensor
    repair_clean_norm: torch.Tensor
    repair_clean_temporal_change: torch.Tensor
    repair_clean_cost: torch.Tensor

    @property
    def available(self) -> torch.Tensor:
        return torch.isfinite(self.gain_total)


def compute_paired_style_gain(
    clean_positions: torch.Tensor | None,
    repaired_positions: torch.Tensor | None,
    noisy_positions: torch.Tensor | None,
    *,
    config: FrontRESSegmentGainConfig,
    clean_root_quaternions: torch.Tensor | None = None,
    repaired_root_quaternions: torch.Tensor | None = None,
    noisy_root_quaternions: torch.Tensor | None = None,
    temporal_mask: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return normalized Noisy->Repaired style gains against immutable Clean."""

    if not _same_shape(clean_positions, repaired_positions, noisy_positions):
        return _unconfirmed_components(clean_positions, repaired_positions, noisy_positions)

    clean = clean_positions.float()
    repaired = repaired_positions.float()
    noisy = noisy_positions.float()
    result = {
        "mpjpe": _error_gain(clean, repaired, noisy, config.mpjpe_scale, temporal_mask, valid_mask),
        "velocity": _error_gain(
            clean,
            repaired,
            noisy,
            config.velocity_scale,
            temporal_mask,
            valid_mask,
            order=1,
        ),
        "acceleration": _error_gain(
            clean,
            repaired,
            noisy,
            config.acceleration_scale,
            temporal_mask,
            valid_mask,
            order=2,
        ),
    }
    result["root_orientation"] = _quaternion_error_gain(
        clean_root_quaternions,
        repaired_root_quaternions,
        noisy_root_quaternions,
        config.root_orientation_scale,
        valid_mask,
        temporal_mask=temporal_mask,
    )
    result["style"] = _available_mean(tuple(result.values()))
    return result


def compute_paired_physics_gain(
    repaired_success: torch.Tensor | None,
    noisy_success: torch.Tensor | None,
    repaired_survival: torch.Tensor | None,
    noisy_survival: torch.Tensor | None,
    *,
    config: FrontRESSegmentGainConfig,
    effective_horizon_k: torch.Tensor | float | int | None,
    repaired_zmp_margin: torch.Tensor | None = None,
    noisy_zmp_margin: torch.Tensor | None = None,
    repaired_contact: torch.Tensor | None = None,
    noisy_contact: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return paired frozen-GMT physics gains with K-normalized survival quality.

    `survival_steps` remains a raw rollout diagnostic. It enters this owner only
    after conversion to `survival_steps / effective_horizon_k`; missing K is
    unconfirmed and therefore cannot silently become a raw-step Gain.

    Status: active, FRS-GAIN-v002 owner.
    Upstream: paired frozen-GMT capture with raw survival steps and per-row K.
    Downstream: final Gain composition, per-step Gain, and evaluation summaries.
    Evidence: S1/S2 contract-confirmed; S4 live mixed-K population remains open.
    """

    like = _first_tensor(
        repaired_success,
        noisy_success,
        repaired_survival,
        noisy_survival,
        repaired_zmp_margin,
        noisy_zmp_margin,
        repaired_contact,
        noisy_contact,
    )
    if like is None:
        empty = torch.empty(0)
        return {
            key: empty
            for key in (
                "success",
                "survival_quality_repaired",
                "survival_quality_noisy",
                "survival",
                "zmp",
                "contact",
                "physics",
            )
        }

    repaired_survival_quality = _survival_quality(
        repaired_survival,
        effective_horizon_k,
        like,
    )
    noisy_survival_quality = _survival_quality(
        noisy_survival,
        effective_horizon_k,
        like,
    )

    result = {
        "success": _pair_difference(repaired_success, noisy_success, like),
        "survival_quality_repaired": repaired_survival_quality,
        "survival_quality_noisy": noisy_survival_quality,
        "survival": _pair_difference(repaired_survival_quality, noisy_survival_quality, like),
        "zmp": _pair_difference(repaired_zmp_margin, noisy_zmp_margin, like),
        "contact": _pair_difference(repaired_contact, noisy_contact, like),
    }
    # B2: quality 的 repaired/noisy 两侧只用于诊断, 不能再次作为 Physics
    # component 参与平均; physics 只聚合四个 paired difference.
    result["physics"] = _available_mean(
        (result["success"], result["survival"], result["zmp"], result["contact"])
    )
    return result


def compute_repair_cost(
    action_steps: torch.Tensor | None,
    *,
    config: FrontRESSegmentGainConfig,
    valid_steps: torch.Tensor | None = None,
    clean_action_steps: torch.Tensor | None = None,
    clean_valid_steps: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return executed full-6D cost with per-row K/done and Clean diagnostics.

    ``valid_steps`` is true for actions executed before the row had already
    terminated and inside its effective K.  The action that produces ``done``
    remains valid because it was executed by the environment.
    """

    if not isinstance(action_steps, torch.Tensor) or action_steps.numel() == 0:
        empty = torch.empty(0)
        return {
            "norm": empty,
            "temporal": empty,
            "cost": empty,
            "clean_norm": empty,
            "clean_temporal": empty,
            "clean_cost": empty,
        }
    result = _repair_cost_components(action_steps, valid_steps, config=config)
    clean = _repair_cost_components(clean_action_steps, clean_valid_steps, config=config)
    result.update(
        {
            "clean_norm": clean["norm"],
            "clean_temporal": clean["temporal"],
            "clean_cost": clean["cost"],
        }
    )
    return result


def _repair_cost_components(
    action_steps: torch.Tensor | None,
    valid_steps: torch.Tensor | None,
    *,
    config: FrontRESSegmentGainConfig,
) -> dict[str, torch.Tensor]:
    if not isinstance(action_steps, torch.Tensor) or action_steps.numel() == 0:
        empty = torch.empty(0)
        return {"norm": empty, "temporal": empty, "cost": empty}
    actions = action_steps.float()
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if actions.ndim != 3 or actions.shape[-1] != 6:
        raise ValueError(f"repair action steps must have shape [T,B,6] or [B,6], got {tuple(actions.shape)}")
    if valid_steps is None:
        valid = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
    else:
        valid = valid_steps.to(device=actions.device, dtype=torch.bool)
        if valid.ndim == 1 and actions.shape[0] == 1:
            valid = valid.unsqueeze(0)
        if tuple(valid.shape) != tuple(actions.shape[:2]):
            raise ValueError(
                "repair action validity must have shape [T,B] matching action steps, "
                f"got {tuple(valid.shape)} for {tuple(actions.shape)}"
            )

    norm_values = torch.linalg.norm(actions, dim=-1) / max(float(config.repair_norm_scale), 1e-8)
    norm = _masked_step_mean(norm_values, valid)
    if actions.shape[0] >= 2:
        temporal_values = torch.linalg.norm(torch.diff(actions, dim=0), dim=-1)
        temporal_values = temporal_values / max(float(config.repair_temporal_scale), 1e-8)
        temporal_valid = valid[:-1] & valid[1:]
        temporal = _masked_step_mean(temporal_values, temporal_valid)
    else:
        temporal = torch.full_like(norm, float("nan"))
    return {"norm": norm, "temporal": temporal, "cost": _available_mean((norm, temporal))}


def _masked_step_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=0)
    summed = torch.where(valid, value, torch.zeros_like(value)).sum(dim=0)
    return torch.where(count > 0, summed / count.clamp_min(1).to(value.dtype), torch.full_like(summed, float("nan")))


def compute_segment_gain_step(
    *,
    clean_position: torch.Tensor | None,
    repaired_position: torch.Tensor | None,
    noisy_position: torch.Tensor | None,
    previous_clean_position: torch.Tensor | None,
    previous_repaired_position: torch.Tensor | None,
    previous_noisy_position: torch.Tensor | None,
    previous_previous_clean_position: torch.Tensor | None,
    previous_previous_repaired_position: torch.Tensor | None,
    previous_previous_noisy_position: torch.Tensor | None,
    clean_root_quaternion: torch.Tensor | None,
    repaired_root_quaternion: torch.Tensor | None,
    noisy_root_quaternion: torch.Tensor | None,
    repaired_success: torch.Tensor | None,
    noisy_success: torch.Tensor | None,
    repaired_survival: torch.Tensor | None,
    noisy_survival: torch.Tensor | None,
    action: torch.Tensor | None,
    previous_action: torch.Tensor | None,
    config: FrontRESSegmentGainConfig,
    effective_horizon_k: torch.Tensor | float | int | None,
    repaired_zmp_margin: torch.Tensor | None = None,
    noisy_zmp_margin: torch.Tensor | None = None,
    repaired_contact: torch.Tensor | None = None,
    noisy_contact: torch.Tensor | None = None,
) -> FrontRESSegmentGainResult:
    """Compute one K-step reward using the same components as final Gain.

    The survival inputs are the current alive increments, not cumulative steps;
    the caller supplies the same effective K used by the final paired owner.
    """

    style = _step_style_gain(
        clean_position,
        repaired_position,
        noisy_position,
        previous_clean_position,
        previous_repaired_position,
        previous_noisy_position,
        previous_previous_clean_position,
        previous_previous_repaired_position,
        previous_previous_noisy_position,
        clean_root_quaternion,
        repaired_root_quaternion,
        noisy_root_quaternion,
        config,
    )
    physics = compute_paired_physics_gain(
        repaired_success,
        noisy_success,
        repaired_survival,
        noisy_survival,
        config=config,
        effective_horizon_k=effective_horizon_k,
        repaired_zmp_margin=repaired_zmp_margin,
        noisy_zmp_margin=noisy_zmp_margin,
        repaired_contact=repaired_contact,
        noisy_contact=noisy_contact,
    )
    repair = _step_repair_cost(action, previous_action, config)
    like = _first_tensor(style["style"], physics["physics"], repair["cost"])
    if like is None:
        like = torch.empty(0)
    # B2: 只用已确认可用的 canonical components 和 accepted weights 合成 Gain.
    total = (
        float(config.style_weight) * _match(style["style"], like)
        + float(config.physics_weight) * _match(physics["physics"], like)
        - float(config.repair_weight) * _match(repair["cost"], like)
    )
    total = torch.where(
        torch.isfinite(style["style"]) & torch.isfinite(physics["physics"]) & torch.isfinite(repair["cost"]),
        total,
        torch.full_like(total, float("nan")),
    )
    return FrontRESSegmentGainResult(
        style_gain=style["style"],
        physics_gain=physics["physics"],
        repair_cost=repair["cost"],
        gain_total=total,
        style_mpjpe_gain=style["mpjpe"],
        style_velocity_gain=style["velocity"],
        style_acceleration_gain=style["acceleration"],
        style_root_orientation_gain=style["root_orientation"],
        physics_success_gain=physics["success"],
        physics_survival_quality_repaired=physics["survival_quality_repaired"],
        physics_survival_quality_noisy=physics["survival_quality_noisy"],
        physics_survival_gain=physics["survival"],
        physics_zmp_gain=physics["zmp"],
        physics_contact_gain=physics["contact"],
        repair_norm=repair["norm"],
        repair_temporal_change=repair["temporal"],
        repair_clean_norm=repair["clean_norm"],
        repair_clean_temporal_change=repair["clean_temporal"],
        repair_clean_cost=repair["clean_cost"],
    )


def compute_segment_gain(
    *,
    clean_positions: torch.Tensor | None,
    repaired_positions: torch.Tensor | None,
    noisy_positions: torch.Tensor | None,
    repaired_success: torch.Tensor | None,
    noisy_success: torch.Tensor | None,
    repaired_survival: torch.Tensor | None,
    noisy_survival: torch.Tensor | None,
    action_steps: torch.Tensor | None,
    config: FrontRESSegmentGainConfig,
    effective_horizon_k: torch.Tensor | float | int | None,
    repaired_zmp_margin: torch.Tensor | None = None,
    noisy_zmp_margin: torch.Tensor | None = None,
    repaired_contact: torch.Tensor | None = None,
    noisy_contact: torch.Tensor | None = None,
    action_step_mask: torch.Tensor | None = None,
    clean_action_steps: torch.Tensor | None = None,
    clean_action_step_mask: torch.Tensor | None = None,
    clean_root_quaternions: torch.Tensor | None = None,
    repaired_root_quaternions: torch.Tensor | None = None,
    noisy_root_quaternions: torch.Tensor | None = None,
    temporal_mask: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> FrontRESSegmentGainResult:
    """计算一个 paired capture 的 canonical Segment Gain.

    函数名说明:
        `compute_segment_gain` 是正式 Gain composition owner, 组合 Style Gain,
        Physics Gain 和 Repair Cost; 它不是 environment reward 或 sampler priority.

    主链路:
        上游: paired capture 提供 matching Clean/Repaired/Noisy execution evidence.
        下游: `gain_total` 写入 storage/PPO return, 同源 component diagnostics 提供给
        sampler 和 terminal/logger.

    语义:
        `gain_total = w_style * style + w_physics * physics - w_repair * cost`.
        缺失 component 必须显式记录 availability, 不得用无关 env reward 替代.
    """

    # B1: 构造 paired Style, Physics 和 executed Repair Cost components.
    style = compute_paired_style_gain(
        clean_positions,
        repaired_positions,
        noisy_positions,
        config=config,
        clean_root_quaternions=clean_root_quaternions,
        repaired_root_quaternions=repaired_root_quaternions,
        noisy_root_quaternions=noisy_root_quaternions,
        temporal_mask=temporal_mask,
        valid_mask=valid_mask,
    )
    physics = compute_paired_physics_gain(
        repaired_success,
        noisy_success,
        repaired_survival,
        noisy_survival,
        config=config,
        effective_horizon_k=effective_horizon_k,
        repaired_zmp_margin=repaired_zmp_margin,
        noisy_zmp_margin=noisy_zmp_margin,
        repaired_contact=repaired_contact,
        noisy_contact=noisy_contact,
    )
    repair = compute_repair_cost(
        action_steps,
        config=config,
        valid_steps=action_step_mask,
        clean_action_steps=clean_action_steps,
        clean_valid_steps=clean_action_step_mask,
    )
    like = _first_tensor(style["style"], physics["physics"], repair["cost"])
    if like is None:
        like = torch.empty(0)
    total = (
        float(config.style_weight) * _match(style["style"], like)
        + float(config.physics_weight) * _match(physics["physics"], like)
        - float(config.repair_weight) * _match(repair["cost"], like)
    )
    total = torch.where(
        torch.isfinite(style["style"]) & torch.isfinite(physics["physics"]) & torch.isfinite(repair["cost"]),
        total,
        torch.full_like(total, float("nan")),
    )
    result = FrontRESSegmentGainResult(
        style_gain=style["style"],
        physics_gain=physics["physics"],
        repair_cost=repair["cost"],
        gain_total=total,
        style_mpjpe_gain=style["mpjpe"],
        style_velocity_gain=style["velocity"],
        style_acceleration_gain=style["acceleration"],
        style_root_orientation_gain=style["root_orientation"],
        physics_success_gain=physics["success"],
        physics_survival_quality_repaired=physics["survival_quality_repaired"],
        physics_survival_quality_noisy=physics["survival_quality_noisy"],
        physics_survival_gain=physics["survival"],
        physics_zmp_gain=physics["zmp"],
        physics_contact_gain=physics["contact"],
        repair_norm=repair["norm"],
        repair_temporal_change=repair["temporal"],
        repair_clean_norm=repair["clean_norm"],
        repair_clean_temporal_change=repair["clean_temporal"],
        repair_clean_cost=repair["clean_cost"],
    )
    # B3: AUDIT-GAIN-01 截获 storage/sampler 消费前的 canonical Gain result.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-GAIN-01",
        style_gain=result.style_gain,
        physics_gain=result.physics_gain,
        repair_cost=result.repair_cost,
        gain_total=result.gain_total,
        available=result.available,
    )
    return result


def _error_gain(
    clean: torch.Tensor,
    repaired: torch.Tensor,
    noisy: torch.Tensor,
    scale: float,
    temporal_mask: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    *,
    order: int = 0,
) -> torch.Tensor:
    if clean.shape[1] <= order:
        return torch.full((clean.shape[0],), float("nan"), device=clean.device, dtype=clean.dtype)
    if order:
        clean = torch.diff(clean, n=order, dim=1)
        repaired = torch.diff(repaired, n=order, dim=1)
        noisy = torch.diff(noisy, n=order, dim=1)
        if temporal_mask is not None and temporal_mask.ndim == 2:
            temporal_mask = temporal_mask[:, order:]
    repaired_err = torch.linalg.norm(repaired - clean, dim=-1)
    noisy_err = torch.linalg.norm(noisy - clean, dim=-1)
    repaired_err = _masked_temporal_mean(repaired_err, temporal_mask)
    noisy_err = _masked_temporal_mean(noisy_err, temporal_mask)
    gain = (noisy_err - repaired_err) / max(float(scale), 1e-8)
    if valid_mask is not None and valid_mask.shape[0] == gain.shape[0]:
        gain = torch.where(valid_mask.bool(), gain, torch.full_like(gain, float("nan")))
    return gain


def _step_style_gain(
    clean: torch.Tensor | None,
    repaired: torch.Tensor | None,
    noisy: torch.Tensor | None,
    previous_clean: torch.Tensor | None,
    previous_repaired: torch.Tensor | None,
    previous_noisy: torch.Tensor | None,
    previous_previous_clean: torch.Tensor | None,
    previous_previous_repaired: torch.Tensor | None,
    previous_previous_noisy: torch.Tensor | None,
    clean_root_quaternion: torch.Tensor | None,
    repaired_root_quaternion: torch.Tensor | None,
    noisy_root_quaternion: torch.Tensor | None,
    config: FrontRESSegmentGainConfig,
) -> dict[str, torch.Tensor]:
    if not _same_shape(clean, repaired, noisy):
        return _unconfirmed_components(clean, repaired, noisy)
    result = {
        "mpjpe": _step_error_gain(clean, repaired, noisy, config.mpjpe_scale),
        "velocity": _step_error_gain(
            _difference(clean, previous_clean),
            _difference(repaired, previous_repaired),
            _difference(noisy, previous_noisy),
            config.velocity_scale,
        ),
        "root_orientation": _quaternion_error_gain(
            clean_root_quaternion,
            repaired_root_quaternion,
            noisy_root_quaternion,
            config.root_orientation_scale,
            None,
        ),
        "acceleration": _step_error_gain(
            _second_difference(clean, previous_clean, previous_previous_clean),
            _second_difference(repaired, previous_repaired, previous_previous_repaired),
            _second_difference(noisy, previous_noisy, previous_previous_noisy),
            config.acceleration_scale,
        ),
    }
    result["style"] = _available_mean(tuple(result.values()))
    return result


def _step_repair_cost(
    action: torch.Tensor | None,
    previous_action: torch.Tensor | None,
    config: FrontRESSegmentGainConfig,
) -> dict[str, torch.Tensor]:
    if not isinstance(action, torch.Tensor) or action.ndim != 2 or action.shape[-1] != 6:
        empty = torch.empty(0)
        return {
            "norm": empty,
            "temporal": empty,
            "cost": empty,
            "clean_norm": empty,
            "clean_temporal": empty,
            "clean_cost": empty,
        }
    norm = torch.linalg.norm(action.float(), dim=-1) / max(float(config.repair_norm_scale), 1e-8)
    if isinstance(previous_action, torch.Tensor) and previous_action.shape == action.shape:
        temporal = torch.linalg.norm(action.float() - previous_action.float(), dim=-1)
        temporal = temporal / max(float(config.repair_temporal_scale), 1e-8)
    else:
        temporal = torch.full_like(norm, float("nan"))
    empty = torch.empty(0, device=norm.device, dtype=norm.dtype)
    return {
        "norm": norm,
        "temporal": temporal,
        "cost": _available_mean((norm, temporal)),
        "clean_norm": empty,
        "clean_temporal": empty,
        "clean_cost": empty,
    }


def _step_error_gain(
    clean: torch.Tensor | None,
    repaired: torch.Tensor | None,
    noisy: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    if not _same_shape(clean, repaired, noisy):
        like = _first_tensor(clean, repaired, noisy)
        return torch.full_like(like, float("nan")) if like is not None else torch.empty(0)
    clean_err = torch.linalg.norm(clean.float() - clean.float(), dim=-1).mean(dim=tuple(range(1, clean.ndim - 1)))
    repaired_err = torch.linalg.norm(repaired.float() - clean.float(), dim=-1).mean(dim=tuple(range(1, repaired.ndim - 1)))
    noisy_err = torch.linalg.norm(noisy.float() - clean.float(), dim=-1).mean(dim=tuple(range(1, noisy.ndim - 1)))
    del clean_err
    return (noisy_err - repaired_err) / max(float(scale), 1e-8)


def _quaternion_error_gain(
    clean: torch.Tensor | None,
    repaired: torch.Tensor | None,
    noisy: torch.Tensor | None,
    scale: float,
    valid_mask: torch.Tensor | None,
    *,
    temporal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _same_shape(clean, repaired, noisy) or clean.shape[-1] != 4:
        like = _first_tensor(clean, repaired, noisy)
        return torch.full((like.shape[0],), float("nan"), device=like.device) if like is not None else torch.empty(0)
    repaired_err = _quat_geodesic(clean.float(), repaired.float())
    noisy_err = _quat_geodesic(clean.float(), noisy.float())
    if (
        repaired_err.ndim >= 2
        and isinstance(temporal_mask, torch.Tensor)
        and tuple(temporal_mask.shape) == tuple(repaired_err.shape[:2])
    ):
        repaired_err = _masked_temporal_mean(repaired_err, temporal_mask)
        noisy_err = _masked_temporal_mean(noisy_err, temporal_mask)
    elif repaired_err.ndim > 1:
        repaired_err = repaired_err.mean(dim=tuple(range(1, repaired_err.ndim)))
        noisy_err = noisy_err.mean(dim=tuple(range(1, noisy_err.ndim)))
    gain = (noisy_err - repaired_err) / max(float(scale), 1e-8)
    if valid_mask is not None and valid_mask.shape[0] == gain.shape[0]:
        gain = torch.where(valid_mask.bool(), gain, torch.full_like(gain, float("nan")))
    return gain


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


def _difference(current: torch.Tensor | None, previous: torch.Tensor | None) -> torch.Tensor | None:
    if not _same_shape(current, previous):
        return None
    return current - previous


def _second_difference(
    current: torch.Tensor | None,
    previous: torch.Tensor | None,
    previous_previous: torch.Tensor | None,
) -> torch.Tensor | None:
    if not _same_shape(current, previous, previous_previous):
        return None
    return current - 2.0 * previous + previous_previous


def _masked_temporal_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None or mask.ndim != 2 or tuple(mask.shape) != tuple(value.shape[:2]):
        return value.mean(dim=tuple(range(1, value.ndim)))
    flat = value.reshape(value.shape[0], value.shape[1], -1).mean(dim=-1)
    mask = mask.bool()
    count = mask.sum(dim=1)
    result = (flat * mask.to(flat.dtype)).sum(dim=1) / count.clamp_min(1).to(flat.dtype)
    return torch.where(count > 0, result, torch.full_like(result, float("nan")))


def _available_mean(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not values:
        return torch.empty(0)
    like = _first_tensor(*values)
    if like is None:
        return torch.empty(0)
    stack = torch.stack([_match(value, like) for value in values], dim=0)
    valid = torch.isfinite(stack)
    count = valid.sum(dim=0)
    summed = torch.where(valid, stack, torch.zeros_like(stack)).sum(dim=0)
    return torch.where(count > 0, summed / count.clamp_min(1), torch.full_like(summed, float("nan")))


def _pair_difference(repaired: torch.Tensor | None, noisy: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if not isinstance(repaired, torch.Tensor) or not isinstance(noisy, torch.Tensor):
        return torch.full_like(like, float("nan"))
    if repaired.shape != noisy.shape:
        return torch.full_like(like, float("nan"))
    return repaired.to(device=like.device, dtype=like.dtype).reshape(-1) - noisy.to(device=like.device, dtype=like.dtype).reshape(-1)


def _survival_quality(
    survival_steps: torch.Tensor | None,
    effective_horizon_k: torch.Tensor | float | int | None,
    like: torch.Tensor,
) -> torch.Tensor:
    """将 raw survival steps 转成当前 segment 的 K-normalized quality."""

    if not isinstance(survival_steps, torch.Tensor):
        return torch.full_like(like, float("nan"))
    survival = survival_steps.to(device=like.device, dtype=like.dtype).reshape(-1)
    if survival.numel() != like.numel():
        return torch.full_like(like, float("nan"))
    if isinstance(effective_horizon_k, torch.Tensor):
        horizon = effective_horizon_k.to(device=like.device, dtype=like.dtype).reshape(-1)
    elif effective_horizon_k is None:
        return torch.full_like(like, float("nan"))
    else:
        horizon = torch.full_like(like, float(effective_horizon_k))
    if horizon.numel() == 1 and like.numel() != 1:
        horizon = horizon.expand_as(like)
    if horizon.numel() != like.numel():
        return torch.full_like(like, float("nan"))
    valid = torch.isfinite(survival) & torch.isfinite(horizon) & horizon.gt(0.0)
    quality = survival / horizon.clamp_min(1.0)
    return torch.where(valid, quality, torch.full_like(quality, float("nan")))


def _match(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return torch.full_like(like, float("nan"))
    return value.to(device=like.device, dtype=like.dtype).reshape(-1)


def _first_tensor(*values: torch.Tensor | None) -> torch.Tensor | None:
    for value in values:
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            return value.reshape(-1).float()
    return None


def _same_shape(*values: torch.Tensor | None) -> bool:
    return bool(values) and all(isinstance(value, torch.Tensor) for value in values) and len({tuple(value.shape) for value in values if isinstance(value, torch.Tensor)}) == 1


def _unconfirmed_components(*values: torch.Tensor | None) -> dict[str, torch.Tensor]:
    like = _first_tensor(*values)
    if like is None:
        empty = torch.empty(0)
        return {key: empty for key in ("mpjpe", "velocity", "acceleration", "root_orientation", "style")}
    missing = torch.full_like(like, float("nan"))
    return {key: missing.clone() for key in ("mpjpe", "velocity", "acceleration", "root_orientation", "style")}
