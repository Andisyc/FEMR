# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event masks for FrontRES authority learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrontRESAuthorityEvents:
    """Authority event schedule over a time-major perturbation mask."""

    event_start: torch.Tensor
    event_active: torch.Tensor
    event_id: torch.Tensor
    event_step: torch.Tensor
    query_mask: torch.Tensor


def _positive_int(name: str, value: int | None, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if value is None:
        raise ValueError(f"{name} must be provided.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def build_frontres_authority_events(
    perturbation_active: torch.Tensor,
    *,
    mode: str,
    burst_length: int = 1,
    refresh_interval: int | None = None,
) -> FrontRESAuthorityEvents:
    """Build event/query masks for single, burst, or persistent perturbations.

    Args:
        perturbation_active: Boolean-like tensor with shape ``(T, ...)``.
        mode: One of ``"single"``, ``"burst"``, or ``"persistent"``.
        burst_length: Maximum authority-hold length for ``mode="burst"``.
        refresh_interval: Optional re-query interval for persistent mode.  When
            omitted, one continuous active segment owns one authority decision.
    """

    if perturbation_active.ndim < 1:
        raise ValueError("perturbation_active must have a leading time dimension.")
    mode = str(mode).lower()
    if mode not in {"single", "burst", "persistent"}:
        raise ValueError(f"Unsupported authority event mode '{mode}'.")

    active = perturbation_active.to(dtype=torch.bool)
    original_shape = active.shape
    time_steps = int(original_shape[0])
    flat_active = active.reshape(time_steps, -1)
    num_tracks = int(flat_active.shape[1])

    if mode == "burst":
        burst_length = int(_positive_int("burst_length", burst_length))
    else:
        burst_length = 1
    refresh_interval = _positive_int("refresh_interval", refresh_interval, allow_none=True)

    event_start = torch.zeros_like(flat_active, dtype=torch.bool)
    event_active = torch.zeros_like(flat_active, dtype=torch.bool)
    event_id = torch.full(flat_active.shape, -1, device=active.device, dtype=torch.long)
    event_step = torch.zeros(flat_active.shape, device=active.device, dtype=torch.long)

    next_event_id = torch.zeros(num_tracks, device=active.device, dtype=torch.long)
    current_event_id = torch.full((num_tracks,), -1, device=active.device, dtype=torch.long)
    current_step = torch.zeros(num_tracks, device=active.device, dtype=torch.long)
    was_active = torch.zeros(num_tracks, device=active.device, dtype=torch.bool)

    for t in range(time_steps):
        now_active = flat_active[t]
        if mode == "single":
            start = now_active
        elif mode == "burst":
            start = now_active & ((~was_active) | current_step.ge(burst_length))
        else:
            if refresh_interval is None:
                start = now_active & (~was_active)
            else:
                start = now_active & ((~was_active) | current_step.ge(refresh_interval))

        if start.any():
            current_event_id = torch.where(start, next_event_id, current_event_id)
            next_event_id = torch.where(start, next_event_id + 1, next_event_id)
            current_step = torch.where(start, torch.zeros_like(current_step), current_step)

        event_start[t] = start
        event_active[t] = now_active
        event_id[t] = torch.where(now_active, current_event_id, torch.full_like(current_event_id, -1))
        event_step[t] = torch.where(now_active, current_step, torch.zeros_like(current_step))

        current_step = torch.where(now_active, current_step + 1, torch.zeros_like(current_step))
        was_active = now_active

    event_start = event_start.reshape(original_shape)
    event_active = event_active.reshape(original_shape)
    event_id = event_id.reshape(original_shape)
    event_step = event_step.reshape(original_shape)
    return FrontRESAuthorityEvents(
        event_start=event_start,
        event_active=event_active,
        event_id=event_id,
        event_step=event_step,
        query_mask=event_start,
    )
