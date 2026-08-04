from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable

import torch

from rsl_rl.frontres.frontres_formal_runtime_probe import emit_formal_runtime_probe


class FrontRESV015RejectedTransactionEvidence(RuntimeError):
    """A complete v015 transaction must be discarded before any update."""


@dataclass(frozen=True)
class FrontRESSegmentTransition:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    valid_mask: torch.Tensor
    reset_mask: torch.Tensor
    segment_ids: torch.Tensor
    segment_source: tuple[str, ...] | None = None
    privileged_observations: torch.Tensor | None = None
    old_means: torch.Tensor | None = None
    old_sigmas: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    priority_evidence: Any | None = None
    audit_transaction_id: str | None = None
    audit_batch_signature: str | None = None
    audit_identity_state: str = "UNCONFIRMED"
    transaction_metadata: Any | None = None
    transaction_row_indices: torch.Tensor | None = None


_V015_GROUPED_CANDIDATE_LAYOUT = "frontres-v015-local-scenario-v1"


def _immutable_v015_candidate_vector(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"v015 candidate metadata {name} must be rank-1")
    if value.requires_grad:
        raise ValueError(f"v015 candidate metadata {name} must be detached")
    return value.detach().to(device="cpu", dtype=torch.long).clone().contiguous()


@dataclass(frozen=True)
class FrontRESV015GroupedCandidateMetadata:
    """Immutable one-row v015 transaction metadata for the grouped candidate adapter.

    函数名说明:
        该对象是 storage 的 local-scenario schema, 不是 formal runner transaction
        lifecycle. 它只绑定已封存的 policy row 与 local scenario identity.

    主链路:
        上游: Step 3B sealed Gain consumer evidence.
        下游: `to_grouped_ppo_candidate_batch()` 和 grouped v003 loss.

    语义:
        `noisy_segment_hash` 的含义是 root artifact, deployable q29 intent,
        Clean continuation, x_t, K 的 local scenario identity. 它不表示整段 K
        reference 都是 Noisy, 也不向 actor 或 PPO 送入 Clean reference.
    """

    transaction_id: str
    policy_snapshot_id: str
    motion_ids: tuple[str, ...]
    start_frames: torch.Tensor
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor
    evidence_valid_step_count: torch.Tensor
    trial_role: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    intent_q29_provenance: str
    intent_q29_source: str
    layout_version: str = _V015_GROUPED_CANDIDATE_LAYOUT
    _integrity_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transaction_id", str(self.transaction_id))
        object.__setattr__(self, "policy_snapshot_id", str(self.policy_snapshot_id))
        object.__setattr__(self, "motion_ids", tuple(str(value) for value in self.motion_ids))
        object.__setattr__(self, "trial_role", tuple(str(value) for value in self.trial_role))
        object.__setattr__(self, "noisy_segment_hashes", tuple(str(value) for value in self.noisy_segment_hashes))
        object.__setattr__(self, "scenario_ids", tuple(str(value) for value in self.scenario_ids))
        object.__setattr__(self, "x_t_identities", tuple(str(value) for value in self.x_t_identities))
        object.__setattr__(self, "intent_q29_provenance", str(self.intent_q29_provenance))
        object.__setattr__(self, "intent_q29_source", str(self.intent_q29_source))
        object.__setattr__(self, "layout_version", str(self.layout_version))
        for name in (
            "start_frames",
            "segment_ids",
            "source_index",
            "trial_index",
            "horizon_k",
            "evidence_valid_step_count",
        ):
            object.__setattr__(self, name, _immutable_v015_candidate_vector(name, getattr(self, name)))
        object.__setattr__(self, "_integrity_hash", _v015_grouped_candidate_metadata_hash(self))
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.segment_ids.numel())

    def validate(self) -> None:
        """Reject partial, mixed, privileged, or mutated v015 candidate metadata."""

        count = self.batch_size
        if (
            count <= 0
            or not self.transaction_id
            or not self.policy_snapshot_id
            or self.layout_version != _V015_GROUPED_CANDIDATE_LAYOUT
        ):
            raise ValueError("v015 grouped candidate metadata has invalid transaction or layout identity")
        for name, value in (
            ("motion_ids", self.motion_ids),
            ("start_frames", self.start_frames),
            ("source_index", self.source_index),
            ("trial_index", self.trial_index),
            ("horizon_k", self.horizon_k),
            ("evidence_valid_step_count", self.evidence_valid_step_count),
            ("trial_role", self.trial_role),
            ("noisy_segment_hashes", self.noisy_segment_hashes),
            ("scenario_ids", self.scenario_ids),
            ("x_t_identities", self.x_t_identities),
        ):
            row_count = len(value) if isinstance(value, tuple) else int(value.numel())
            if row_count != count:
                raise ValueError(f"v015 grouped candidate metadata {name} must have {count} rows")
        if (
            any(not value for value in self.motion_ids)
            or any(not value for value in self.noisy_segment_hashes)
            or any(not value for value in self.scenario_ids)
            or any(not value for value in self.x_t_identities)
            or any(role != "policy" for role in self.trial_role)
            or bool((self.start_frames < 0).any())
            or bool((self.segment_ids < 0).any())
            or bool((self.source_index < 0).any())
            or bool((self.trial_index < 0).any())
            or bool((self.horizon_k <= 0).any())
            or bool((self.evidence_valid_step_count < 0).any())
            or bool((self.evidence_valid_step_count > self.horizon_k).any())
        ):
            raise ValueError("v015 grouped candidate metadata has invalid row, policy-role, or K evidence values")
        source = self.intent_q29_source.lower()
        if (
            self.intent_q29_provenance != "deployment_noisy_q29"
            or not source
            or any(token in source for token in ("clean", "root", "global"))
        ):
            raise ValueError("v015 grouped candidate metadata rejects non-deployment q29 provenance")
        by_source: dict[int, tuple[str, int, int, str, str, str, int]] = {}
        seen_attempts: set[tuple[int, int]] = set()
        for row in range(count):
            source_index = int(self.source_index[row].item())
            trial_index = int(self.trial_index[row].item())
            attempt_key = (source_index, trial_index)
            if attempt_key in seen_attempts:
                raise ValueError("v015 grouped candidate metadata has duplicate source/trial policy rows")
            seen_attempts.add(attempt_key)
            identity = (
                self.motion_ids[row],
                int(self.start_frames[row].item()),
                int(self.segment_ids[row].item()),
                self.scenario_ids[row],
                self.noisy_segment_hashes[row],
                self.x_t_identities[row],
                int(self.horizon_k[row].item()),
            )
            previous = by_source.setdefault(source_index, identity)
            if previous != identity:
                raise ValueError("v015 grouped candidate metadata mixes local scenario identity within one source")
        if self._integrity_hash != _v015_grouped_candidate_metadata_hash(self):
            raise RuntimeError("v015 grouped candidate metadata was mutated after sealing")


def _v015_grouped_candidate_metadata_hash(metadata: FrontRESV015GroupedCandidateMetadata) -> str:
    digest = hashlib.sha256()
    for value in (
        metadata.transaction_id,
        metadata.policy_snapshot_id,
        metadata.intent_q29_provenance,
        metadata.intent_q29_source,
        metadata.layout_version,
        repr(metadata.motion_ids),
        repr(metadata.trial_role),
        repr(metadata.noisy_segment_hashes),
        repr(metadata.scenario_ids),
        repr(metadata.x_t_identities),
    ):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    for value in (
        metadata.start_frames,
        metadata.segment_ids,
        metadata.source_index,
        metadata.trial_index,
        metadata.horizon_k,
        metadata.evidence_valid_step_count,
    ):
        digest.update(repr(value.detach().to(device="cpu", dtype=torch.long).tolist()).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class FrontRESSegmentStorageBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    # Diagnostic owner copy of the rollout reward. PPO still consumes returns
    # and advantages below; retaining reward here makes that transformation auditable.
    rewards: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    valid_mask: torch.Tensor
    segment_ids: torch.Tensor
    privileged_observations: torch.Tensor | None = None
    old_means: torch.Tensor | None = None
    old_sigmas: torch.Tensor | None = None
    audit_transaction_id: str | None = None
    audit_batch_signature: str | None = None
    audit_identity_state: str = "UNCONFIRMED"
    transaction_metadata: Any | None = None
    transaction_row_indices: torch.Tensor | None = None

    def to_ppo_batch(self, batch_cls: Callable[..., Any]) -> Any:
        """把 finalized Segment storage 转换为 legacy PPO batch contract.

        函数名说明:
            `to_ppo_batch` 是 legacy storage-to-algorithm adapter, 只搬运已冻结
            字段; 它不重算 action, log_prob, return 或 advantage.

        主链路:
            上游: K-aware return computation 完成 storage tuple.
            下游: `compute_frontres_segment_ppo_loss` 消费同一 row order 的 batch.

        语义:
            actions, old log-prob, old mean/sigma 和 advantage 必须来自同一个
            rollout tuple. 转换时不得改变 action representation 或 row identity.
            此 legacy adapter 不携带 Step 3 transaction metadata.
        """
        if isinstance(self.transaction_metadata, FrontRESV015GroupedCandidateMetadata):
            raise ValueError("v015 grouped candidate metadata must not enter legacy to_ppo_batch()")
        # B1: 读取 K-aware return 已完成的 finalized storage tuple.
        # B2: 保持旧 PPO adapter 的字段边界, 不把 candidate metadata 送入 runner.
        batch = self._build_ppo_batch(batch_cls, include_transaction_metadata=False)
        # B3: AUDIT-RETURN-01 截获 PPO 实际消费的 return/advantage tuple.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-RETURN-01",
            returns=self.returns,
            advantages=self.advantages,
            valid_mask=self.valid_mask,
            segment_ids=self.segment_ids,
        )
        return batch

    def to_grouped_ppo_candidate_batch(self, batch_cls: Callable[..., Any]) -> Any:
        """Build the v015 metadata-bearing candidate batch without activating the legacy runner."""

        # B1: 读取与 legacy PPO 同源的 finalized storage tuple.
        # B2: 仅在 candidate adapter 中保留 sealed v015 metadata 与完整 row mapping.
        if not isinstance(self.transaction_metadata, FrontRESV015GroupedCandidateMetadata):
            raise ValueError("v015 grouped candidate adapter requires FrontRESV015GroupedCandidateMetadata")
        self.transaction_metadata.validate()
        row_indices = _storage_batch_transaction_row_indices(self)
        expected_rows = torch.arange(self.transaction_metadata.batch_size, device=self.segment_ids.device, dtype=torch.long)
        if (
            row_indices is None
            or int(self.segment_ids.numel()) != self.transaction_metadata.batch_size
            or not torch.equal(torch.sort(row_indices).values, expected_rows)
        ):
            raise ValueError("v015 grouped candidate adapter requires one complete sealed transaction")
        return self._build_ppo_batch(batch_cls, include_transaction_metadata=True)

    def _build_ppo_batch(self, batch_cls: Callable[..., Any], *, include_transaction_metadata: bool) -> Any:
        payload: dict[str, Any] = {
            "observations": self.observations,
            "actions": self.actions,
            "old_log_probs": self.old_log_probs,
            "old_values": self.old_values,
            "returns": self.returns,
            "advantages": self.advantages,
            "valid_mask": self.valid_mask,
            "segment_ids": self.segment_ids,
            "old_means": self.old_means,
            "old_sigmas": self.old_sigmas,
        }
        if include_transaction_metadata:
            payload["privileged_observations"] = self.privileged_observations
            payload["transaction_metadata"] = self.transaction_metadata
            payload["transaction_row_indices"] = _storage_batch_transaction_row_indices(self)
        return batch_cls(**payload)


def _storage_batch_transaction_row_indices(batch: FrontRESSegmentStorageBatch) -> torch.Tensor | None:
    """Resolve sealed transaction rows at the immutable batch boundary."""

    metadata = batch.transaction_metadata
    raw_indices = batch.transaction_row_indices
    if metadata is None:
        if raw_indices is not None:
            raise ValueError("transaction_row_indices require transaction_metadata")
        return None
    metadata_segment_ids = getattr(metadata, "segment_ids", None)
    metadata_size = int(getattr(metadata, "batch_size", 0) or 0)
    if not isinstance(metadata_segment_ids, torch.Tensor):
        raise ValueError("transaction metadata requires segment_ids")
    if metadata_size <= 0:
        metadata_size = int(metadata_segment_ids.numel())
    batch_size = int(batch.segment_ids.numel())
    if raw_indices is None:
        if metadata_size != batch_size:
            raise ValueError(
                "transaction metadata requires explicit transaction_row_indices when storage rows are a subset"
            )
        row_indices = torch.arange(batch_size, device=batch.segment_ids.device, dtype=torch.long)
    else:
        _require_vector("transaction_row_indices", raw_indices, batch_size)
        if raw_indices.requires_grad:
            raise ValueError("transaction_row_indices must be detached")
        row_indices = raw_indices.detach().to(device=batch.segment_ids.device, dtype=torch.long)
        if bool((row_indices < 0).any()) or bool((row_indices >= metadata_size).any()):
            raise ValueError("transaction_row_indices contain an out-of-range metadata row")
    selected_segment_ids = metadata_segment_ids.detach().to(
        device=batch.segment_ids.device,
        dtype=torch.long,
    )[row_indices]
    if not torch.equal(selected_segment_ids, batch.segment_ids.detach().to(dtype=torch.long)):
        raise ValueError("transaction metadata segment_ids do not match the storage batch rows")
    return row_indices


def _require_vector(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [B], got {tuple(tensor.shape)}")
