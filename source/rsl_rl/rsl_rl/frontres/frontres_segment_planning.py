from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import torch

from rsl_rl.frontres.frontres_formal_runtime_probe import emit_formal_runtime_probe


class FrontRESSegmentState(IntEnum):
    """Segment replay budget state owned by the sampler."""

    UNKNOWN = 0
    PROMISING = 1
    FRONTIER = 2
    DELAYED_REGRET = 3
    SOLVED = 4
    HOPELESS = 5


SEGMENT_STATE_NAMES = tuple(state.name.lower() for state in FrontRESSegmentState)


@dataclass(frozen=True)
class FrontRESSegmentSample:
    segment_ids: torch.Tensor
    source: tuple[str, ...]
    priority: torch.Tensor
    staleness: torch.Tensor
    valid_mask: torch.Tensor
    segment_state: torch.Tensor | None = None
    rollout_trial_count: torch.Tensor | None = None
    horizon_k: torch.Tensor | None = None
    budget_reason: tuple[str, ...] = ()
    trial_role: tuple[str, ...] = ()
    source_index: torch.Tensor | None = None
    trial_index: torch.Tensor | None = None


@dataclass(frozen=True)
class FrontRESSegmentRolloutEvidence:
    segment_ids: torch.Tensor
    reset_success: torch.Tensor
    score_noisy: torch.Tensor
    score_repaired: torch.Tensor
    score_clean: torch.Tensor
    gain_over_noisy: torch.Tensor
    fall_repaired: torch.Tensor
    contact_consistency: torch.Tensor
    action_norm: torch.Tensor
    valid_reward: torch.Tensor
    horizon_k: torch.Tensor
    gain_total: torch.Tensor | None = None
    gain_style: torch.Tensor | None = None
    gain_physics: torch.Tensor | None = None
    repair_cost: torch.Tensor | None = None
    gain_source: str = "legacy"


@dataclass(frozen=True)
class FrontRESV015PriorityEvidence:
    """Scenario-keyed v003 replay-priority evidence, 不是 sampler-state update.

    状态: candidate-only.
    上游: one-row v003 return evidence.
    下游: 后续携带 stable segment/trial identities 的 transaction owner.
    该对象没有 actor loss, optimizer, 或 legacy sampler mutation path.
    """

    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    horizon_k: torch.Tensor
    gain_total: torch.Tensor
    intent_gain: torch.Tensor
    physics_gain: torch.Tensor
    repair_cost: torch.Tensor
    valid_mask: torch.Tensor
    intent_q29_provenance: str
    intent_q29_source: str
    gain_source: str = "FRS-GAIN-v006-loaded-support-zmp-applicability"

    def validate(self) -> None:
        count = int(self.gain_total.numel())
        if count <= 0:
            raise ValueError("v015 priority evidence requires at least one policy row")
        for name, value in (
            ("horizon_k", self.horizon_k),
            ("gain_total", self.gain_total),
            ("intent_gain", self.intent_gain),
            ("physics_gain", self.physics_gain),
            ("repair_cost", self.repair_cost),
            ("valid_mask", self.valid_mask),
        ):
            if value.ndim != 1 or int(value.numel()) != count:
                raise ValueError(f"v015 priority evidence {name} must be [B]")
        if (
            len(self.scenario_ids) != count
            or len(self.noisy_segment_hashes) != count
            or len(self.x_t_identities) != count
            or not bool((self.horizon_k > 0).all())
            or self.intent_q29_provenance != "deployment_noisy_q29"
            or self.gain_source != "FRS-GAIN-v006-loaded-support-zmp-applicability"
        ):
            raise ValueError("v015 priority evidence has invalid identity, provenance, or Gain source")
        source = self.intent_q29_source.lower()
        if not source or any(token in source for token in ("clean", "root", "global")):
            raise ValueError("v015 priority evidence rejects non-deployment q29 provenance")
        valid = self.valid_mask.bool()
        for name, value in (
            ("gain_total", self.gain_total),
            ("intent_gain", self.intent_gain),
            ("physics_gain", self.physics_gain),
            ("repair_cost", self.repair_cost),
        ):
            finite = torch.isfinite(value)
            if not bool(finite[valid].all()) or bool(finite[~valid].any()):
                raise ValueError(f"v015 priority evidence {name} must be finite exactly on valid rows")

    @property
    def priority_signal(self) -> torch.Tensor:
        """返回 raw canonical Gain evidence, 不施加 loss mass 或 sampler update."""

        return self.gain_total


def build_frontres_v015_priority_evidence(return_evidence: Any) -> FrontRESV015PriorityEvidence:
    """将 v003 decomposition 复制到 scenario-keyed priority-evidence carrier."""

    validate = getattr(return_evidence, "validate", None)
    if not callable(validate):
        raise TypeError("v015 priority evidence requires a validated v003 return carrier")
    validate()
    if getattr(return_evidence, "gain_source", None) != "FRS-GAIN-v006-loaded-support-zmp-applicability":
        raise ValueError("v015 priority evidence rejects legacy or unspecified Gain source")
    result = FrontRESV015PriorityEvidence(
        scenario_ids=tuple(str(value) for value in return_evidence.scenario_ids),
        noisy_segment_hashes=tuple(str(value) for value in return_evidence.noisy_segment_hashes),
        x_t_identities=tuple(str(value) for value in return_evidence.x_t_identities),
        horizon_k=return_evidence.horizon_k.detach().clone(),
        gain_total=return_evidence.gain_total.detach().clone(),
        intent_gain=return_evidence.intent_gain.detach().clone(),
        physics_gain=return_evidence.physics_gain.detach().clone(),
        repair_cost=return_evidence.repair_cost.detach().clone(),
        valid_mask=return_evidence.policy_row_valid.detach().clone(),
        intent_q29_provenance=str(return_evidence.intent_q29_provenance),
        intent_q29_source=str(return_evidence.intent_q29_source),
        gain_source=str(return_evidence.gain_source),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class FrontRESSegmentTrialEvidence:
    segment_ids: torch.Tensor
    trial_count: torch.Tensor
    valid_trial_count: torch.Tensor
    policy_gain: torch.Tensor
    best_gain: torch.Tensor
    mean_gain: torch.Tensor
    success_frac: torch.Tensor
    fall_frac: torch.Tensor
    oracle_gap: torch.Tensor
    confidence: torch.Tensor
    score_noisy: torch.Tensor
    score_repaired: torch.Tensor
    horizon_k: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class FrontRESSegmentRolloutBudget:
    segment_ids: torch.Tensor
    trial_count: torch.Tensor
    horizon_k: torch.Tensor
    segment_state: torch.Tensor
    reason: tuple[str, ...]


@dataclass(frozen=True)
class FrontRESSegmentTrialPlan:
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor
    trial_role: tuple[str, ...]
    base_segment_ids: torch.Tensor
    base_trial_count: torch.Tensor


@dataclass(frozen=True)
class FrontRESFrozenPolicyTransactionPlan:
    """Pure row layout for a future frozen-policy Double Segment transaction.

    The caller owns snapshot capture.  This object only preserves the supplied
    snapshot identity and proves that every scheduled attempt is an ordinary
    policy sample before runner/storage wiring is added.
    """

    transaction_id: str
    policy_snapshot_id: str
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor
    trial_role: tuple[str, ...]
    base_segment_ids: torch.Tensor
    base_trial_count: torch.Tensor
    base_horizon_k: torch.Tensor
    minimum_policy_attempts: int
    exact_policy_attempts: int | None = None

    def validate(self) -> None:
        if not self.transaction_id:
            raise ValueError("transaction_id must be non-empty")
        if not self.policy_snapshot_id:
            raise ValueError("policy_snapshot_id must be non-empty")
        if int(self.minimum_policy_attempts) < 2:
            raise ValueError("minimum_policy_attempts must be at least two")
        if self.exact_policy_attempts is not None and int(self.exact_policy_attempts) < 2:
            raise ValueError("exact_policy_attempts must be at least two")

        source_count = int(self.base_segment_ids.numel())
        if source_count < 2:
            raise ValueError("frozen policy transaction requires at least two selected segments")
        if self.base_segment_ids.ndim != 1:
            raise ValueError("base_segment_ids must be a one-dimensional tensor")
        if self.base_trial_count.ndim != 1 or int(self.base_trial_count.numel()) != source_count:
            raise ValueError("base_trial_count must be source-aligned [S] data")
        if self.base_horizon_k.ndim != 1 or int(self.base_horizon_k.numel()) != source_count:
            raise ValueError("base_horizon_k must be source-aligned [S] data")
        if int(torch.unique(self.base_segment_ids).numel()) != source_count:
            raise ValueError("frozen policy transaction selected duplicate segment groups")
        if bool((self.base_trial_count < int(self.minimum_policy_attempts)).any()):
            raise ValueError("every selected segment requires the configured minimum policy attempts")
        if self.exact_policy_attempts is not None and bool(
            (self.base_trial_count != int(self.exact_policy_attempts)).any()
        ):
            raise ValueError("every selected segment requires the configured exact policy attempts")
        if bool((self.base_horizon_k <= 0).any()):
            raise ValueError("base_horizon_k must be positive")

        row_count = int(self.segment_ids.numel())
        expected_rows = int(self.base_trial_count.sum().item())
        if row_count != expected_rows:
            raise ValueError(f"transaction row count {row_count} does not match planned attempts {expected_rows}")
        if any(tensor.ndim != 1 or int(tensor.numel()) != row_count for tensor in (
            self.source_index,
            self.trial_index,
            self.horizon_k,
        )):
            raise ValueError("transaction row tensors must be source-expanded [sum(M_s)] data")
        if len(self.trial_role) != row_count:
            raise ValueError("trial_role must have one entry per transaction row")
        if any(role != "policy" for role in self.trial_role):
            raise ValueError("frozen policy transaction may contain only policy-sampled attempts")
        if bool((self.horizon_k <= 0).any()):
            raise ValueError("transaction horizon_k must be positive")

        expected_source_index = torch.repeat_interleave(
            torch.arange(source_count, dtype=torch.long, device=self.source_index.device),
            self.base_trial_count.to(device=self.source_index.device, dtype=torch.long),
        )
        expected_segment_ids = torch.repeat_interleave(
            self.base_segment_ids.to(device=self.segment_ids.device, dtype=torch.long),
            self.base_trial_count.to(device=self.segment_ids.device, dtype=torch.long),
        )
        expected_horizon_k = torch.repeat_interleave(
            self.base_horizon_k.to(device=self.horizon_k.device, dtype=torch.long),
            self.base_trial_count.to(device=self.horizon_k.device, dtype=torch.long),
        )
        expected_trial_index = torch.cat(
            [
                torch.arange(int(count), dtype=torch.long, device=self.trial_index.device)
                for count in self.base_trial_count.detach().cpu().tolist()
            ],
            dim=0,
        )
        if not torch.equal(self.source_index, expected_source_index):
            raise ValueError("transaction rows must be source-major")
        if not torch.equal(self.segment_ids, expected_segment_ids):
            raise ValueError("transaction rows must preserve each selected segment identity")
        if not torch.equal(self.horizon_k, expected_horizon_k):
            raise ValueError("transaction rows must preserve one K value per selected segment")
        if not torch.equal(self.trial_index, expected_trial_index):
            raise ValueError("transaction rows must be trial-major inside each segment group")


