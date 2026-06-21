"""Structured-rho advantage construction for FrontRES PPO updates.

The structured-rho branch is Advantage Learning, not supervised rho regression:
rollout preference builds a signed rho advantage, and rho-update validity
weights decide how strongly that advantage should enter PPO.  The legacy
storage tensors are still named ``acceptance_target`` / ``acceptance_mask``
elsewhere; inside this module we use the live concept names instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrontRESStructuredRhoCarrier:
    rho_advantage: torch.Tensor
    rho_validity_weight: torch.Tensor
    rho_advantage_exec: torch.Tensor
    rho_validity_weight_exec: torch.Tensor
    adv_mean: float
    weight_mean: float
    retention_mean: float
    floor_mean: float
    full_bonus_mean: float
    direction_mean: float
    underwrite_mean: float
    accept_mean: float
    raw_direction_mean: float
    centered_mean: float
    drive_mean: float

    @property
    def rho_weight(self) -> torch.Tensor:
        """Legacy alias for rho_validity_weight."""
        return self.rho_validity_weight

    @property
    def rho_weight_exec(self) -> torch.Tensor:
        """Legacy alias for rho_validity_weight_exec."""
        return self.rho_validity_weight_exec


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
    rho_update_weight: torch.Tensor,
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
    underwrite_weight: float,
    joint_weight_floor: float,
    use_rho_update_weight: bool,
    device: torch.device,
) -> FrontRESStructuredRhoCarrier:
    """Build rho PPO advantages, validity weights, and diagnostics."""
    rho_advantage = torch.zeros(num_envs, 6, device=device, dtype=rho_current.dtype)
    rho_validity_weight = torch.zeros(num_envs, 6, device=device, dtype=rho_current.dtype)
    if n_exec <= 0:
        zero = torch.zeros(0, 6, device=device, dtype=rho_current.dtype)
        return FrontRESStructuredRhoCarrier(
            rho_advantage=rho_advantage,
            rho_validity_weight=rho_validity_weight,
            rho_advantage_exec=zero,
            rho_validity_weight_exec=zero,
            adv_mean=0.0,
            weight_mean=0.0,
            retention_mean=0.0,
            floor_mean=0.0,
            full_bonus_mean=0.0,
            direction_mean=0.0,
            underwrite_mean=0.0,
            accept_mean=0.0,
            raw_direction_mean=0.0,
            centered_mean=0.0,
            drive_mean=0.0,
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

    rollout_gain = exec_frontres[:n].detach() - exec_perturbed[:n].detach()
    evidence_scale = (
        exec_candidate[:n].detach() - exec_perturbed[:n].detach()
    ).abs() + float(pref_margin) + 1.0e-6
    rho_direction = (rollout_gain / evidence_scale).clamp(-1.0, 1.0)
    rho_direction = torch.where(
        rollout_gain.abs() > float(pref_margin),
        rho_direction,
        torch.zeros_like(rho_direction),
    )
    raw_rho_direction = rho_direction
    underwrite_weight = max(0.0, float(underwrite_weight))
    accept_from_projected = (rollout_gain > float(pref_margin)).to(rho.dtype)
    underwrite = torch.relu(exec_candidate[:n].detach() - exec_frontres[:n].detach() - float(pref_margin))
    underwrite_direction = (underwrite / evidence_scale).clamp(0.0, 1.0)
    if underwrite_weight > 0.0:
        rho_direction = (
            rho_direction + underwrite_weight * accept_from_projected * underwrite_direction
        ).clamp(-1.0, 1.0)
    rho_direction_dim = rho_direction.view(-1, 1).expand_as(rho).detach()
    raw_direction_dim = raw_rho_direction.view(-1, 1).expand_as(rho).detach()
    underwrite_dim = underwrite_direction.view(-1, 1).expand_as(rho).detach()
    accept_dim = accept_from_projected.view(-1, 1).expand_as(rho).detach()

    # Formal rho advantage must match the debug contract: rollout evidence
    # owns PPO advantage; boundary priors are handled by a separate loss.
    adv_dim = float(directional_weight) * rho_direction_dim
    retention_term = torch.zeros_like(adv_dim)
    floor_term = torch.zeros_like(adv_dim)
    full_bonus = torch.zeros_like(adv_dim)

    # The validity weight is now only the rho/action-cone loss mask.  Sample
    # selection and boundary priors must not be hidden inside this mask.
    weight_dim = dim_weight.detach().clamp(0.0, 1.0)

    rho_advantage[:n, :6] = adv_dim.detach()
    rho_validity_weight[:n, :6] = weight_dim
    return FrontRESStructuredRhoCarrier(
        rho_advantage=rho_advantage,
        rho_validity_weight=rho_validity_weight,
        rho_advantage_exec=adv_dim,
        rho_validity_weight_exec=weight_dim,
        adv_mean=_mean_active(adv_dim, dim_active),
        weight_mean=_mean_active(weight_dim, dim_active),
        retention_mean=_mean_active(retention_term, dim_active),
        floor_mean=_mean_active(floor_term, dim_active),
        full_bonus_mean=_mean_active(full_bonus, dim_active),
        direction_mean=_mean_active(rho_direction_dim, dim_active),
        underwrite_mean=_mean_active(underwrite_dim, dim_active),
        accept_mean=_mean_active(accept_dim, dim_active),
        raw_direction_mean=_mean_active(raw_direction_dim, dim_active),
        centered_mean=_mean_active(rho_centered, dim_active),
        drive_mean=_mean_active(rho_drive, dim_active),
    )
