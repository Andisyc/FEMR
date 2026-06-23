# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous 6D authority parameterization for FrontRES stage-2 control."""

from __future__ import annotations

from dataclasses import dataclass

import torch


FRONTRES_AUTHORITY_DIM_NAMES: tuple[str, ...] = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
FRONTRES_AUTHORITY_MIN = 0.0
FRONTRES_AUTHORITY_MAX = 1.0
FRONTRES_TASK_SPACE_DIM = 6


@dataclass(frozen=True)
class FrontRESAuthorityStats:
    """Small detached diagnostics for continuous rho distributions."""

    mean: torch.Tensor
    std: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor
    near_zero_frac: torch.Tensor
    near_one_frac: torch.Tensor


def _validate_authority_shape(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.shape[-1] != FRONTRES_TASK_SPACE_DIM:
        raise ValueError(
            f"{name} must have last dimension {FRONTRES_TASK_SPACE_DIM} "
            f"for {FRONTRES_AUTHORITY_DIM_NAMES}, got shape {tuple(tensor.shape)}."
        )


def _broadcast_active_task_dims(
    active_task_dims: torch.Tensor | list[float] | tuple[float, ...] | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if active_task_dims is None:
        return None
    mask = torch.as_tensor(active_task_dims, device=reference.device, dtype=reference.dtype)
    if mask.shape[-1] != FRONTRES_TASK_SPACE_DIM:
        raise ValueError(
            f"active_task_dims must have last dimension {FRONTRES_TASK_SPACE_DIM}, "
            f"got shape {tuple(mask.shape)}."
        )
    return mask


def bound_authority_rho(raw_authority: torch.Tensor, *, bound: str = "sigmoid") -> torch.Tensor:
    """Map raw Stage-2 output to continuous rho in ``[0, 1]``."""

    _validate_authority_shape(raw_authority, name="raw_authority")
    if bound == "sigmoid":
        return torch.sigmoid(raw_authority)
    if bound == "clamp":
        return raw_authority.clamp(FRONTRES_AUTHORITY_MIN, FRONTRES_AUTHORITY_MAX)
    raise ValueError(f"Unsupported authority bound '{bound}'. Expected 'sigmoid' or 'clamp'.")


def apply_authority_active_mask(
    authority_rho: torch.Tensor,
    active_task_dims: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Zero forbidden task-space dimensions without changing allowed rho values."""

    _validate_authority_shape(authority_rho, name="authority_rho")
    active_mask = _broadcast_active_task_dims(active_task_dims, authority_rho)
    if active_mask is None:
        return authority_rho
    return authority_rho * active_mask


def raw_authority_to_rho(
    raw_authority: torch.Tensor,
    active_task_dims: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    *,
    bound: str = "sigmoid",
) -> torch.Tensor:
    """Convert raw Stage-2 output to masked continuous 6D rho."""

    return apply_authority_active_mask(bound_authority_rho(raw_authority, bound=bound), active_task_dims)


def apply_authority_to_delta_se(
    proposal_delta_se: torch.Tensor,
    authority_rho: torch.Tensor,
    active_task_dims: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Apply continuous rho to the detached Stage-1 ``Delta SE`` proposal."""

    _validate_authority_shape(proposal_delta_se, name="proposal_delta_se")
    _validate_authority_shape(authority_rho, name="authority_rho")
    masked_rho = apply_authority_active_mask(authority_rho, active_task_dims)
    return masked_rho * proposal_delta_se


def authority_rho_stats(
    authority_rho: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
    *,
    near_zero: float = 0.05,
    near_one: float = 0.95,
) -> FrontRESAuthorityStats:
    """Summarize continuous rho by dimension with detached CPU-friendly tensors."""

    _validate_authority_shape(authority_rho, name="authority_rho")
    rho = authority_rho.detach().reshape(-1, FRONTRES_TASK_SPACE_DIM)
    if sample_mask is not None:
        mask = sample_mask.detach().reshape(-1).to(device=rho.device, dtype=torch.bool)
        if mask.numel() != rho.shape[0]:
            raise ValueError(f"sample_mask must flatten to {rho.shape[0]} entries, got {mask.numel()}.")
        rho = rho[mask]

    if rho.shape[0] == 0:
        zeros = torch.zeros(FRONTRES_TASK_SPACE_DIM, device=authority_rho.device, dtype=authority_rho.dtype)
        return FrontRESAuthorityStats(
            mean=zeros,
            std=zeros,
            min=zeros,
            max=zeros,
            near_zero_frac=zeros,
            near_one_frac=zeros,
        )

    return FrontRESAuthorityStats(
        mean=rho.mean(dim=0),
        std=rho.std(dim=0, unbiased=False),
        min=rho.min(dim=0).values,
        max=rho.max(dim=0).values,
        near_zero_frac=(rho <= near_zero).to(dtype=torch.float32).mean(dim=0),
        near_one_frac=(rho >= near_one).to(dtype=torch.float32).mean(dim=0),
    )
