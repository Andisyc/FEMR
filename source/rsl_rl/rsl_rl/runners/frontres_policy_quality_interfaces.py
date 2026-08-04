"""Stable ports shared by policy-quality evaluation and runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from rsl_rl.frontres.frontres_policy_quality_manifest import (
    FrontRESPolicyQualityManifest,
    FrontRESPolicyQualityRouteIdentity,
)


@dataclass(frozen=True)
class FrontRESPolicyQualityObservationIdentity:
    expected_obs_dim: int
    actor_input_dim: int
    normalizer_identity: str

    def __post_init__(self) -> None:
        if self.expected_obs_dim <= 0 or not 0 < self.actor_input_dim <= self.expected_obs_dim:
            raise ValueError("quality observation dimensions must satisfy 0 < actor_input_dim <= expected_obs_dim")
        if not self.normalizer_identity.strip():
            raise ValueError("normalizer_identity must be explicit")


@dataclass(frozen=True)
class FrontRESPolicyQualityRouteResult:
    identity: FrontRESPolicyQualityRouteIdentity
    observation_identity: FrontRESPolicyQualityObservationIdentity
    actions: torch.Tensor
    gain: Any
    execution: Any


@dataclass(frozen=True)
class FrontRESPolicyQualityRouteHooks:
    observe: Callable[[], torch.Tensor]
    apply_action: Callable[[torch.Tensor], Any]
    step: Callable[[], Any]
    compute_gain: Callable[[], Any]
    capture_execution: Callable[[], Any]
    begin_route: Callable[[str], None] | None = None
    set_audit_identity: Callable[[Mapping[str, str]], None] | None = None


@dataclass(frozen=True)
class FrontRESPolicyQualityEvalRequest:
    manifest_path: str
    hsl_checkpoint_path: str
    policy_checkpoint_path: str
    result_path: str
    manifest: FrontRESPolicyQualityManifest


@dataclass(frozen=True)
class FrontRESPolicyQualityFormalOwnerBundle:
    owner_identity: tuple[tuple[str, str], ...]
    prepare_item: Callable[[Any, Any, FrontRESPolicyQualityEvalRequest], tuple[Any, Any, Any]]
    isolation_state: Callable[[Any], str]
    serialize_result: Callable[[Any, tuple[FrontRESPolicyQualityRouteResult, ...]], Mapping[str, Any]]

    def __post_init__(self) -> None:
        owners = dict(self.owner_identity)
        required = {"reset", "observation", "action", "rollout", "gain", "execution"}
        if set(owners) != required or any(not str(value).strip() for value in owners.values()):
            raise ValueError(f"quality formal owner bundle must name exactly {sorted(required)}")


__all__ = [
    "FrontRESPolicyQualityEvalRequest",
    "FrontRESPolicyQualityFormalOwnerBundle",
    "FrontRESPolicyQualityObservationIdentity",
    "FrontRESPolicyQualityRouteHooks",
    "FrontRESPolicyQualityRouteResult",
]
