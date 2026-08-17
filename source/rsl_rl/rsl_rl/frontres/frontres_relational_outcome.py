"""Pure K-step trajectory to FRS-GAIN-v009 Outcome materialization."""

from __future__ import annotations

from typing import Any

import torch

from rsl_rl.frontres.frontres_hierarchical_gain_candidate import Outcome, Thresholds


def _require_trajectory_pair(clean: Any, repair: Any, expected_support: torch.Tensor) -> torch.Tensor:
    for value in (clean, repair):
        validate = getattr(value, "validate", None)
        if not callable(validate):
            raise TypeError("relational Outcome requires validated Clean/Repair trajectories")
        validate()
    if tuple(clean.joint_pos.shape) != tuple(repair.joint_pos.shape):
        raise ValueError("relational Outcome requires shape-aligned Clean/Repair trajectories")
    k_steps = int(repair.joint_pos.shape[0])
    if int(repair.joint_pos.shape[1]) != 1 or tuple(expected_support.shape) != (k_steps, 1, 2):
        raise ValueError("relational Outcome requires one Scenario row and expected support [K,1,2]")
    valid = clean.valid_mask[:, 0].bool() & repair.valid_mask[:, 0].bool()
    if not bool(valid.any().item()):
        raise ValueError("relational Outcome requires at least one paired valid K-step")
    return valid


def _local_root_xy(trajectory: Any) -> torch.Tensor:
    origin = getattr(trajectory, "env_origin", None)
    if not isinstance(origin, torch.Tensor) or tuple(origin.shape) != tuple(trajectory.root_pos.shape):
        raise ValueError("relational Outcome requires row-aligned environment origins")
    if not bool(torch.isfinite(origin.float()).all().item()):
        raise ValueError("relational Outcome requires finite environment origins")
    return trajectory.root_pos[:, 0, :2].float() - origin[:, 0, :2].float()


def _capture_margin(trajectory: Any, valid: torch.Tensor) -> torch.Tensor:
    """Root capture-point margin in the current two-foot support proxy."""

    root_xy = _local_root_xy(trajectory)
    root_z = trajectory.root_pos[:, 0, 2].float() - trajectory.env_origin[:, 0, 2].float()
    velocity_xy = trajectory.root_lin_vel[:, 0, :2].float()
    feet_xy = trajectory.foot_pos[:, 0, :, :2].float()
    contact = trajectory.contact[:, 0].bool()
    if not bool(torch.isfinite(root_z[valid]).all().item()) or bool((root_z[valid] <= 0.0).any().item()):
        raise ValueError("relational Outcome requires positive finite root height")
    # Planned flight frames use both feet as the next-support proxy. Loaded
    # support failures are classified as L1 before recovery quality is used.
    active = torch.where(contact.any(dim=-1, keepdim=True), contact, torch.ones_like(contact))
    inf = torch.finfo(feet_xy.dtype).max
    support_min = torch.where(active.unsqueeze(-1), feet_xy, torch.full_like(feet_xy, inf)).amin(dim=1) - 0.04
    support_max = torch.where(active.unsqueeze(-1), feet_xy, torch.full_like(feet_xy, -inf)).amax(dim=1) + 0.04
    omega = torch.sqrt(torch.tensor(9.81, device=root_z.device, dtype=root_z.dtype) / root_z.clamp_min(1.0e-6))
    capture_xy = root_xy + velocity_xy / omega.unsqueeze(-1)
    margin = torch.minimum(capture_xy - support_min, support_max - capture_xy).amin(dim=-1)
    if not bool(torch.isfinite(margin[valid]).all().item()):
        raise ValueError("relational Outcome produced a non-finite capture margin")
    return margin


def _quat_geodesic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = torch.nn.functional.normalize(left.float(), dim=-1)
    right = torch.nn.functional.normalize(right.float(), dim=-1)
    dot = (left * right).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _paired_step_metrics(clean: Any, repair: Any) -> tuple[torch.Tensor, ...]:
    linear = torch.linalg.vector_norm(
        repair.root_lin_vel[:, 0].float() - clean.root_lin_vel[:, 0].float(), dim=-1
    )
    angular = torch.linalg.vector_norm(
        repair.root_ang_vel[:, 0].float() - clean.root_ang_vel[:, 0].float(), dim=-1
    )
    support_drift = torch.linalg.vector_norm(
        repair.foot_pos[:, 0].float() - clean.foot_pos[:, 0].float(), dim=-1
    ).amax(dim=-1)
    joint = torch.sqrt(torch.mean(
        (repair.joint_pos[:, 0].float() - clean.joint_pos[:, 0].float()).square(), dim=-1
    ))
    orientation = _quat_geodesic(repair.root_quat[:, 0], clean.root_quat[:, 0])
    horizontal_shift = (
        repair.root_pos[:, 0, :2].float() - clean.root_pos[:, 0, :2].float()
    ).unsqueeze(1)
    body_delta = repair.key_body_pos[:, 0].float() - clean.key_body_pos[:, 0].float()
    body_delta = body_delta.clone()
    body_delta[..., :2] -= horizontal_shift
    key_body = torch.linalg.vector_norm(body_delta, dim=-1).mean(dim=-1)
    return linear, angular, support_drift, joint, orientation, key_body


def _mean(values: torch.Tensor, mask: torch.Tensor, name: str) -> float:
    selected = values[mask]
    if int(selected.numel()) == 0 or not bool(torch.isfinite(selected).all().item()):
        raise ValueError(f"relational Outcome requires finite {name} evidence")
    return float(selected.mean().detach().cpu().item())


def build_frontres_relational_outcome(
    *,
    clean: Any,
    repair: Any,
    expected_support: torch.Tensor,
    repair_action: torch.Tensor,
    thresholds: Thresholds = Thresholds(),
) -> Outcome:
    """Materialize one relation-ready Outcome without scalar Gain or reward."""

    valid = _require_trajectory_pair(clean, repair, expected_support)
    expected = expected_support[:, 0].bool()
    actual = repair.contact[:, 0].bool()
    survival = repair.survival[:, 0].bool()
    survival_fail = valid & ~survival
    no_load = valid.unsqueeze(-1) & expected & ~actual
    illegal = valid.unsqueeze(-1) & ~expected & actual
    if int(valid.sum().item()) > 1:
        expected_switch = (expected[1:] != expected[:-1]).any(dim=-1)
        actual_switch = (actual[1:] != actual[:-1]).any(dim=-1)
        unplanned_switch = valid[1:] & valid[:-1] & actual_switch & ~expected_switch
    else:
        unplanned_switch = torch.zeros(0, device=valid.device, dtype=torch.bool)

    capture = _capture_margin(repair, valid)
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    capture_trend = float((capture[valid_indices[-1]] - capture[valid_indices[0]]).detach().cpu().item())
    linear, angular, support_drift, joint, orientation, key_body = _paired_step_metrics(clean, repair)
    expected_loaded = expected.any(dim=-1)
    zmp_mask = valid & expected_loaded & actual.any(dim=-1)
    zmp_values = repair.zmp_margin[:, 0].float()
    zmp_applicable = bool(zmp_mask.any().item())
    if zmp_applicable:
        if not bool(torch.isfinite(zmp_values[zmp_mask]).all().item()):
            raise ValueError("relational Outcome requires finite applicable phase-ZMP margins")
        zmp_margin: float | None = float(zmp_values[zmp_mask].amin().detach().cpu().item())
    else:
        zmp_margin = None

    step_stable = (
        valid
        & (capture >= float(thresholds.capture_margin_min))
        & (linear <= float(thresholds.linear_momentum_error_max))
        & (angular <= float(thresholds.angular_momentum_error_max))
        & (support_drift <= float(thresholds.support_drift_max))
        & ~survival_fail
        & ~no_load.any(dim=-1)
        & ~illegal.any(dim=-1)
    )
    if zmp_applicable:
        step_stable = step_stable & (~zmp_mask | (zmp_values >= float(thresholds.zmp_margin_min)))
    stable_hold_steps = 0
    for value in reversed(step_stable.tolist()):
        if not value:
            break
        stable_hold_steps += 1

    if not isinstance(repair_action, torch.Tensor) or tuple(repair_action.shape) != (6,):
        raise ValueError("relational Outcome requires one full-6D repair action")
    action = repair_action.detach().float()
    if not bool(torch.isfinite(action).all().item()):
        raise ValueError("relational Outcome requires a finite repair action")
    return Outcome(
        survival_ok=not bool(survival_fail.any().item()),
        survival_failure_duration=float(survival_fail.sum().item()),
        expected_support_no_load=float(no_load.sum().item()),
        unplanned_support_switch=float(unplanned_switch.sum().item()),
        illegal_contact_duration=float(illegal.sum().item()),
        capture_margin=float(capture[valid].amin().detach().cpu().item()),
        capture_margin_trend=capture_trend,
        zmp_applicable=zmp_applicable,
        zmp_margin=zmp_margin,
        linear_momentum_error=_mean(linear, valid, "linear dynamic error"),
        angular_momentum_error=_mean(angular, valid, "angular dynamic error"),
        support_drift=_mean(support_drift, valid, "support drift"),
        stable_hold_steps=stable_hold_steps,
        intent_error=(
            _mean(orientation, valid, "root orientation Intent error"),
            _mean(joint, valid, "joint Intent error"),
            _mean(key_body, valid, "key-body Intent error"),
        ),
        repair_cost=float(torch.linalg.vector_norm(action).detach().cpu().item()),
    )


__all__ = ("build_frontres_relational_outcome",)
