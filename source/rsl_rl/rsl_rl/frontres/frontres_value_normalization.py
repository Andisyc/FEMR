"""Pure output-preserving Critic target-scale state for active FrontRES PPO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch

from rsl_rl.frontres.frontres_interfaces import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FRONTRES_VALUE_NORMALIZER_DECAY,
    FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
)



@dataclass(frozen=True)
class FrontRESValueNormalizerState:
    """Immutable scalar target moments committed once per formal transaction."""

    mean: float = 0.0
    second_moment: float = 1.0
    update_count: int = 0

    def validate(self) -> "FrontRESValueNormalizerState":
        values = (float(self.mean), float(self.second_moment))
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("FRS-PPO-v009 value-normalizer state must be finite")
        tolerance = 1.0e-12 * max(1.0, abs(self.second_moment), self.mean**2)
        if self.second_moment < 0.0 or self.second_moment + tolerance < self.mean**2:
            raise ValueError("FRS-PPO-v009 value-normalizer second moment is inconsistent with its mean")
        if not isinstance(self.update_count, int) or isinstance(self.update_count, bool) or self.update_count < 0:
            raise ValueError("FRS-PPO-v009 value-normalizer update_count must be a non-negative integer")
        return self

    def state_dict(self) -> dict[str, float | int | str]:
        self.validate()
        return {
            "normalization_id": FRONTRES_VALUE_NORMALIZATION_ID,
            "mean": float(self.mean),
            "second_moment": float(self.second_moment),
            "update_count": int(self.update_count),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "FrontRESValueNormalizerState":
        if not isinstance(payload, Mapping) or set(payload) != {
            "normalization_id",
            "mean",
            "second_moment",
            "update_count",
        }:
            raise ValueError("FRS-PPO-v009 value-normalizer state has an incompatible schema")
        if payload.get("normalization_id") != FRONTRES_VALUE_NORMALIZATION_ID:
            raise ValueError("FRS-PPO-v009 value-normalizer identity mismatch")
        try:
            state = cls(
                mean=float(payload["mean"]),
                second_moment=float(payload["second_moment"]),
                update_count=payload["update_count"],
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("FRS-PPO-v009 value-normalizer state contains invalid values") from exc
        return state.validate()


@dataclass(frozen=True)
class FrontRESValueNormalizerUpdate:
    previous: FrontRESValueNormalizerState
    candidate: FrontRESValueNormalizerState
    batch_mean: float
    batch_second_moment: float
    scale: float


def preview_frontres_v007_value_normalization(
    segment_targets: torch.Tensor,
    state: FrontRESValueNormalizerState,
    *,
    decay: float,
    scale_floor: float,
) -> FrontRESValueNormalizerUpdate:
    """Preview one non-amplifying EMA scale without mutating training state."""

    if not isinstance(segment_targets, torch.Tensor) or segment_targets.ndim != 1 or segment_targets.numel() != 8:
        raise ValueError("FRS-PPO-v011 value normalization requires exactly eight Scenario targets")
    if not bool(torch.isfinite(segment_targets).all().item()):
        raise FloatingPointError("FRS-PPO-v009 value normalization requires finite Segment targets")
    if not isinstance(state, FrontRESValueNormalizerState):
        raise TypeError("FRS-PPO-v009 value normalization requires an immutable normalizer state")
    state.validate()
    decay = float(decay)
    scale_floor = float(scale_floor)
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("FRS-PPO-v009 value-normalizer decay must be finite in [0,1)")
    if not math.isfinite(scale_floor) or scale_floor < 1.0:
        raise ValueError("FRS-PPO-v009 value-normalizer scale floor must be finite and at least one")

    values = segment_targets.detach().to(device="cpu", dtype=torch.float64)
    batch_mean = float(values.mean().item())
    batch_second_moment = float(values.square().mean().item())
    update_weight = 1.0 - decay
    candidate_mean = decay * state.mean + update_weight * batch_mean
    candidate_second_moment = decay * state.second_moment + update_weight * batch_second_moment
    variance = max(0.0, candidate_second_moment - candidate_mean**2)
    scale = max(scale_floor, math.sqrt(variance))
    candidate = FrontRESValueNormalizerState(
        mean=candidate_mean,
        second_moment=candidate_second_moment,
        update_count=state.update_count + 1,
    ).validate()
    if not math.isfinite(scale) or scale < 1.0:
        raise FloatingPointError("FRS-PPO-v009 produced an invalid Critic target scale")
    return FrontRESValueNormalizerUpdate(
        previous=state,
        candidate=candidate,
        batch_mean=batch_mean,
        batch_second_moment=batch_second_moment,
        scale=scale,
    )
