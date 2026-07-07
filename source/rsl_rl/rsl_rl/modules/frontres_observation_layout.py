"""FrontRES policy-observation layout helpers."""

from __future__ import annotations

import torch


def split_frontres_policy_obs(obs: torch.Tensor, gmt_dim: int | None) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Split FrontRES-only prefix from the GMT-compatible suffix."""
    if gmt_dim is None or obs.shape[-1] <= int(gmt_dim):
        return None, obs
    gmt_dim = int(gmt_dim)
    num_extra = obs.shape[-1] - gmt_dim
    return obs[..., :num_extra], obs[..., num_extra:]


def extract_frontres_extra_norm_stats(
    state: dict,
    obs_dim: int | None,
    gmt_dim: int | None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return FrontRES-only prefix mean/std from a combined obs normalizer state.

    B1: Current checkpoints may have [100D FrontRES prefix | 770D GMT suffix].
    B2: Legacy checkpoints may have [30D anchor prefix | 770D GMT suffix].
    B3: Missing newly-added prefix stats use identity normalization.
    """
    if obs_dim is None or gmt_dim is None:
        return None
    obs_dim = int(obs_dim)
    gmt_dim = int(gmt_dim)
    num_extra = obs_dim - gmt_dim
    mean = state.get("_mean")
    std = state.get("_std")
    if num_extra <= 0 or not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        return None
    state_dim = int(mean.shape[-1])
    if state_dim < gmt_dim or int(std.shape[-1]) != state_dim:
        return None

    if state_dim >= obs_dim:
        extra_mean = mean[..., :num_extra]
        extra_std = std[..., :num_extra]
    else:
        legacy_extra = state_dim - gmt_dim
        if legacy_extra <= 0:
            return None
        legacy_extra = min(legacy_extra, num_extra)
        extra_shape = (*mean.shape[:-1], num_extra)
        extra_mean = torch.zeros(extra_shape, device=mean.device, dtype=mean.dtype)
        extra_std = torch.ones(extra_shape, device=std.device, dtype=std.dtype)
        extra_mean[..., :legacy_extra] = mean[..., :legacy_extra]
        extra_std[..., :legacy_extra] = std[..., :legacy_extra]

    if device is not None:
        extra_mean = extra_mean.to(device)
        extra_std = extra_std.to(device)
    return extra_mean, extra_std


def compose_frontres_obs_norm_state(
    gmt_state: dict,
    extra_mean: torch.Tensor | None,
    extra_std: torch.Tensor | None,
) -> dict:
    """Save a combined [extra | GMT] normalizer state when extra stats exist."""
    if extra_mean is None or extra_std is None:
        return gmt_state
    mean = gmt_state.get("_mean")
    std = gmt_state.get("_std")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        return gmt_state
    result = dict(gmt_state)
    extra_mean = extra_mean.to(device=mean.device, dtype=mean.dtype)
    extra_std = extra_std.to(device=std.device, dtype=std.dtype)
    result["_mean"] = torch.cat([extra_mean, mean], dim=-1)
    result["_std"] = torch.cat([extra_std, std], dim=-1)
    var = gmt_state.get("_var")
    if isinstance(var, torch.Tensor):
        result["_var"] = torch.cat([extra_std.to(device=var.device, dtype=var.dtype).square(), var], dim=-1)
    return result
