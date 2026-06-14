"""Structured-rho carrier construction for FrontRES PPO updates.

The structured-rho branch uses the acceptance target/mask tensors as a carrier
for PPO advantages.  This module owns that carrier construction so the runner
does not also own the rho credit-assignment math.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrontRESStructuredRhoCarrier:
    accept_target: torch.Tensor
    accept_mask: torch.Tensor
    target_exec: torch.Tensor
    mask_exec: torch.Tensor
    adv_mean: float
    weight_mean: float
    retention_mean: float
    floor_mean: float
    full_bonus_mean: float
    direction_mean: float
    centered_mean: float
    drive_mean: float


def _mean_active(values: torch.Tensor, active: torch.Tensor) -> float:
    if values.numel() == 0 or not bool(active.any().detach().item()):
        return 0.0
    return float(values[active].mean().detach().item())


def _slice_or_pad_alpha(
    alpha_source: torch.Tensor | None,
    fallback: torch.Tensor,
    *,
    n_exec: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(alpha_source, torch.Tensor) and alpha_source.numel() > 0:
        alpha = alpha_source.to(device=device, dtype=dtype).view(-1)
        if alpha.numel() < n_exec:
            alpha = torch.nn.functional.pad(alpha, (0, n_exec - alpha.numel()), value=0.0)
        return alpha[:n_exec].detach().clamp(0.0, 1.0)
    return fallback[:n_exec].to(device=device, dtype=dtype).view(-1).detach().clamp(0.0, 1.0)


def _center_drive(rho_current: torch.Tensor, *, rho_center: float, deadzone: float) -> tuple[torch.Tensor, torch.Tensor]:
    centered = (2.0 * (rho_current.detach() - float(rho_center))).clamp(-1.0, 1.0)
    if deadzone > 1.0e-6:
        drive = torch.where(
            centered.abs() >= deadzone,
            torch.sign(centered),
            centered / deadzone,
        )
    else:
        drive = torch.sign(centered)
    return centered, drive


def _live_route_direction(
    candidate_score: torch.Tensor,
    projected_score: torch.Tensor,
    fallback_score: torch.Tensor,
    *,
    pref_margin: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    candidate = candidate_score.detach().to(dtype)
    projected = projected_score.detach().to(dtype)
    fallback = fallback_score.detach().to(dtype)
    candidate_regret = torch.relu(candidate - projected - pref_margin)
    fallback_regret = torch.relu(fallback - projected - pref_margin)
    scale = (candidate - fallback).abs() + pref_margin + 1.0e-6
    return ((candidate_regret - fallback_regret) / scale).clamp(-1.0, 1.0)


def _feasible_component(
    feasible_components: Mapping[str, torch.Tensor],
    name: str,
    fallback: torch.Tensor,
    *,
    n_exec: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    component = feasible_components.get(name, fallback)
    return component[:n_exec].to(device=device, dtype=dtype)


def _tri_anchor_direction(
    *,
    rho_space: str,
    grouped_targets_enabled: bool,
    n_exec: int,
    dtype: torch.dtype,
    device: torch.device,
    fallback_alpha: torch.Tensor,
    exec_feasible: torch.Tensor,
    feasible_components: Mapping[str, torch.Tensor] | None,
    candidate_planar: torch.Tensor | None,
    candidate_rp: torch.Tensor | None,
    candidate_z: torch.Tensor | None,
    projected_planar: torch.Tensor | None,
    projected_rp: torch.Tensor | None,
    projected_z: torch.Tensor | None,
    base_planar: torch.Tensor | None,
    base_rp: torch.Tensor | None,
    base_z: torch.Tensor | None,
    pref_margin: float,
) -> torch.Tensor | None:
    if (
        rho_space not in ("tri_anchor", "tri-anchor", "tri")
        or not grouped_targets_enabled
        or not isinstance(candidate_planar, torch.Tensor)
        or not isinstance(candidate_rp, torch.Tensor)
        or not isinstance(candidate_z, torch.Tensor)
        or not isinstance(projected_planar, torch.Tensor)
        or not isinstance(projected_rp, torch.Tensor)
        or not isinstance(projected_z, torch.Tensor)
        or not isinstance(base_planar, torch.Tensor)
        or not isinstance(base_rp, torch.Tensor)
        or not isinstance(base_z, torch.Tensor)
        or not isinstance(feasible_components, Mapping)
    ):
        return None

    feasible_xy = _feasible_component(
        feasible_components, "xy", exec_feasible, n_exec=n_exec, device=device, dtype=dtype
    )
    feasible_yaw = _feasible_component(
        feasible_components, "yaw", exec_feasible, n_exec=n_exec, device=device, dtype=dtype
    )
    feasible_planar = 0.5 * (feasible_xy + feasible_yaw)
    feasible_rp = _feasible_component(
        feasible_components, "rp", exec_feasible, n_exec=n_exec, device=device, dtype=dtype
    )
    feasible_z = _feasible_component(
        feasible_components, "z", exec_feasible, n_exec=n_exec, device=device, dtype=dtype
    )

    fallback_planar = (1.0 - fallback_alpha) * base_planar.detach().to(dtype) + fallback_alpha * feasible_planar.detach()
    fallback_rp = (1.0 - fallback_alpha) * base_rp.detach().to(dtype) + fallback_alpha * feasible_rp.detach()
    fallback_z = (1.0 - fallback_alpha) * base_z.detach().to(dtype) + fallback_alpha * feasible_z.detach()

    direction_planar = _live_route_direction(
        candidate_planar,
        projected_planar,
        fallback_planar,
        pref_margin=pref_margin,
        dtype=dtype,
    )
    direction_rp = _live_route_direction(
        candidate_rp,
        projected_rp,
        fallback_rp,
        pref_margin=pref_margin,
        dtype=dtype,
    )
    direction_z = _live_route_direction(
        candidate_z,
        projected_z,
        fallback_z,
        pref_margin=pref_margin,
        dtype=dtype,
    )
    return torch.stack(
        [direction_planar, direction_planar, direction_z, direction_rp, direction_rp, direction_planar],
        dim=-1,
    ).detach()


def build_structured_rho_carrier(
    *,
    num_envs: int,
    n_exec: int,
    rho_current: torch.Tensor,
    rho_dim_weight: torch.Tensor,
    actor_gate: torch.Tensor,
    exec_perturbed: torch.Tensor,
    exec_feasible: torch.Tensor,
    exec_frontres: torch.Tensor,
    exec_candidate: torch.Tensor,
    state_alpha_target: torch.Tensor,
    live_alpha: torch.Tensor | None,
    rho_space: str,
    grouped_targets_enabled: bool,
    feasible_components: Mapping[str, torch.Tensor] | None,
    candidate_planar: torch.Tensor | None,
    candidate_rp: torch.Tensor | None,
    candidate_z: torch.Tensor | None,
    projected_planar: torch.Tensor | None,
    projected_rp: torch.Tensor | None,
    projected_z: torch.Tensor | None,
    base_planar: torch.Tensor | None,
    base_rp: torch.Tensor | None,
    base_z: torch.Tensor | None,
    pref_margin: float,
    rho_floor: float,
    directional_weight: float,
    rho_center: float,
    center_drive_deadzone: float,
    retention_weight: float,
    floor_penalty_weight: float,
    full_bonus_weight: float,
    joint_weight_floor: float,
    use_actor_gate_weight: bool,
    device: torch.device,
) -> FrontRESStructuredRhoCarrier:
    """Build the structured-rho PPO advantage carrier and diagnostics."""
    accept_target = torch.zeros(num_envs, 6, device=device, dtype=rho_current.dtype)
    accept_mask = torch.zeros(num_envs, 6, device=device, dtype=rho_current.dtype)
    if n_exec <= 0:
        zero = torch.zeros(0, 6, device=device, dtype=rho_current.dtype)
        return FrontRESStructuredRhoCarrier(
            accept_target, accept_mask, zero, zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    n = min(int(n_exec), int(rho_current.shape[0]))
    rho = rho_current[:n]
    dim_weight = rho_dim_weight[:n].detach().clamp(min=0.0)
    dim_active = dim_weight > 1.0e-6

    floor_violation = torch.relu(float(rho_floor) - exec_frontres[:n].detach())
    full_repair_ok = (exec_candidate[:n].detach() >= float(rho_floor)).to(rho.dtype)
    rho_centered, rho_drive = _center_drive(
        rho,
        rho_center=max(0.0, min(1.0, float(rho_center))),
        deadzone=max(0.0, float(center_drive_deadzone)),
    )

    fallback_alpha = _slice_or_pad_alpha(
        live_alpha,
        state_alpha_target[:n, 0],
        n_exec=n,
        device=device,
        dtype=rho.dtype,
    )
    fallback_exec = (1.0 - fallback_alpha) * exec_perturbed[:n].detach() + fallback_alpha * exec_feasible[:n].detach()

    route_direction = _tri_anchor_direction(
        rho_space=rho_space,
        grouped_targets_enabled=grouped_targets_enabled,
        n_exec=n,
        dtype=rho.dtype,
        device=device,
        fallback_alpha=fallback_alpha,
        exec_feasible=exec_feasible,
        feasible_components=feasible_components,
        candidate_planar=candidate_planar,
        candidate_rp=candidate_rp,
        candidate_z=candidate_z,
        projected_planar=projected_planar,
        projected_rp=projected_rp,
        projected_z=projected_z,
        base_planar=base_planar,
        base_rp=base_rp,
        base_z=base_z,
        pref_margin=float(pref_margin),
    )

    candidate_regret = torch.relu(exec_candidate[:n].detach() - exec_frontres[:n].detach() - pref_margin)
    fallback_regret = torch.relu(fallback_exec - exec_frontres[:n].detach() - pref_margin)
    direction_scale = (exec_candidate[:n].detach() - fallback_exec).abs() + pref_margin + 1.0e-6
    rho_direction = ((candidate_regret - fallback_regret) / direction_scale).clamp(-1.0, 1.0)
    if isinstance(route_direction, torch.Tensor) and route_direction.shape == rho.shape:
        rho_direction_dim = route_direction.to(device=device, dtype=rho.dtype).detach()
    else:
        rho_direction_dim = rho_direction.view(-1, 1).expand_as(rho).detach()

    directional_adv = float(directional_weight) * rho_direction_dim * rho_drive
    retention_term = float(retention_weight) * rho_drive
    floor_direction = torch.where(full_repair_ok > 0.5, torch.ones_like(rho_direction), -torch.ones_like(rho_direction))
    floor_term = (
        float(floor_penalty_weight)
        * floor_violation.view(-1, 1)
        * floor_direction.view(-1, 1)
        * rho_drive
    )
    full_bonus = float(full_bonus_weight) * full_repair_ok.view(-1, 1) * rho_drive
    adv_dim = directional_adv + retention_term + floor_term + full_bonus

    weight_floor = max(0.0, min(1.0, float(joint_weight_floor)))
    if use_actor_gate_weight:
        sample_weight = (weight_floor + (1.0 - weight_floor) * actor_gate[:n]).detach().clamp(0.0, 1.0)
    else:
        sample_weight = torch.ones((n,), device=device, dtype=rho.dtype)
    weight_dim = (sample_weight.view(-1, 1) * dim_weight).detach().clamp(0.0, 1.0)

    accept_target[:n, :6] = adv_dim.detach()
    accept_mask[:n, :6] = weight_dim
    return FrontRESStructuredRhoCarrier(
        accept_target=accept_target,
        accept_mask=accept_mask,
        target_exec=adv_dim,
        mask_exec=weight_dim,
        adv_mean=_mean_active(adv_dim, dim_active),
        weight_mean=_mean_active(weight_dim, dim_active),
        retention_mean=_mean_active(retention_term, dim_active),
        floor_mean=_mean_active(floor_term, dim_active),
        full_bonus_mean=_mean_active(full_bonus, dim_active),
        direction_mean=_mean_active(rho_direction_dim, dim_active),
        centered_mean=_mean_active(rho_centered, dim_active),
        drive_mean=_mean_active(rho_drive, dim_active),
    )
