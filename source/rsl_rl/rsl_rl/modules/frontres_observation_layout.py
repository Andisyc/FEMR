"""FrontRES policy-observation layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch


FRONTRES_FUTURE_INTENT_LAYOUT_VERSION = "frontres-v015-future-intent-q29-v1"
FRONTRES_FUTURE_INTENT_DIM = 29


@dataclass(frozen=True)
class FrontRESFutureIntentLayout:
    """Versioned actor-only layout for future deployment-provenance q29 intent."""

    version: str
    future_offsets: tuple[int, ...]

    @property
    def intent_frame_count(self) -> int:
        return max(self.future_offsets) + 1

    @property
    def actor_tail_dim(self) -> int:
        return len(self.future_offsets) * FRONTRES_FUTURE_INTENT_DIM

    def validate(self) -> None:
        if self.version != FRONTRES_FUTURE_INTENT_LAYOUT_VERSION:
            raise ValueError(
                "FrontRES future-intent layout version must be "
                f"{FRONTRES_FUTURE_INTENT_LAYOUT_VERSION!r}, got {self.version!r}"
            )
        if not self.future_offsets:
            raise ValueError("FrontRES future-intent layout requires nonempty future offsets")
        if any(int(offset) <= 0 for offset in self.future_offsets):
            raise ValueError(f"FrontRES future-intent offsets must be positive, got {self.future_offsets}")
        if tuple(sorted(set(int(offset) for offset in self.future_offsets))) != tuple(self.future_offsets):
            raise ValueError(
                "FrontRES future-intent offsets must be strictly ordered and unique, "
                f"got {self.future_offsets}"
            )


def resolve_frontres_future_intent_layout(
    future_offsets: Iterable[int] | torch.Tensor,
    layout_version: str,
) -> FrontRESFutureIntentLayout:
    """Freeze the only accepted v015 actor-H tensor layout."""

    if isinstance(future_offsets, torch.Tensor):
        future_offsets = future_offsets.detach().cpu().tolist()
    try:
        offsets = tuple(int(value) for value in future_offsets)
    except TypeError as exc:
        raise ValueError("FrontRES future-intent offsets must be an iterable of integers") from exc
    layout = FrontRESFutureIntentLayout(version=str(layout_version), future_offsets=offsets)
    layout.validate()
    return layout


def build_frontres_future_intent_tail(
    intent_q29: torch.Tensor,
    *,
    layout: FrontRESFutureIntentLayout,
    provenance: Iterable[Mapping[str, Any]],
) -> torch.Tensor:
    """Select ordered future q29 offsets after rejecting non-deployment provenance."""

    layout.validate()
    if not isinstance(intent_q29, torch.Tensor) or intent_q29.ndim != 3:
        raise ValueError(
            "FrontRES future intent must have shape [B,H_max+1,29], "
            f"got {getattr(intent_q29, 'shape', None)}"
        )
    if intent_q29.requires_grad or not torch.is_floating_point(intent_q29):
        raise ValueError("FrontRES future intent must be detached floating-point scenario data")
    if not bool(torch.isfinite(intent_q29).all().item()):
        raise ValueError("FrontRES future intent contains non-finite values")
    expected_shape = (layout.intent_frame_count, FRONTRES_FUTURE_INTENT_DIM)
    if tuple(intent_q29.shape[1:]) != expected_shape:
        raise ValueError(
            "FrontRES future intent must have shape [B,H_max+1,29] with "
            f"H_max+1={expected_shape[0]}, got {tuple(intent_q29.shape)}"
        )
    _validate_frontres_future_intent_provenance(provenance, batch_size=int(intent_q29.shape[0]))
    return intent_q29[:, layout.future_offsets, :].reshape(intent_q29.shape[0], layout.actor_tail_dim).detach().clone()


def _validate_frontres_future_intent_provenance(
    provenance: Iterable[Mapping[str, Any]],
    *,
    batch_size: int,
) -> None:
    try:
        rows = tuple(provenance)
    except TypeError as exc:
        raise ValueError("FrontRES future intent provenance must be row-aligned") from exc
    if len(rows) != batch_size:
        raise ValueError(
            "FrontRES future intent provenance must have one mapping per actor row, "
            f"got {len(rows)} for B={batch_size}"
        )
    for row, value in enumerate(rows):
        if not isinstance(value, Mapping):
            raise ValueError(f"FrontRES future intent provenance row {row} must be a mapping")
        if value.get("intent_q29_provenance") != "deployment_noisy_q29":
            raise ValueError(
                "FrontRES actor future intent requires intent_q29_provenance="
                f"'deployment_noisy_q29', row {row} has {value.get('intent_q29_provenance')!r}"
            )
        source = str(value.get("intent_q29_source", "")).lower()
        if not source or "root" in source or "global" in source or "clean" in source:
            raise ValueError(
                "FrontRES actor future intent source must exclude root/global/Clean fields, "
                f"row {row} has {value.get('intent_q29_source')!r}"
            )
        if value.get("clean_continuation_provenance") != "clean_gmt_only":
            raise ValueError("FrontRES local scenario must retain a GMT-only Clean continuation")


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


def frontres_extra_norm_stats_for_save(
    extra_mean: torch.Tensor | None,
    extra_std: torch.Tensor | None,
    extra_normalizer: object | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return persisted FrontRES prefix stats from fixed tensors or a live normalizer."""
    if isinstance(extra_mean, torch.Tensor) and isinstance(extra_std, torch.Tensor):
        return extra_mean.detach().clone(), extra_std.detach().clone()
    if extra_normalizer is None:
        return None, None
    mean = getattr(extra_normalizer, "_mean", None)
    std = getattr(extra_normalizer, "_std", None)
    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        return mean.detach().clone(), std.detach().clone()
    return None, None
