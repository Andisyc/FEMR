from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_storage",
    Path(__file__).resolve().with_name("frontres_formal_runtime_probe.py"),
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


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
            payload["transaction_metadata"] = self.transaction_metadata
            payload["transaction_row_indices"] = _storage_batch_transaction_row_indices(self)
        return batch_cls(**payload)


@dataclass(frozen=True)
class FrontRESV015OneActionKEvidence:
    """Candidate-only evidence for one Repair tuple and its frozen-GMT K consequence.

    This is deliberately not a PPO batch: it has no reward, return, advantage,
    optimizer, or legacy ``to_ppo_batch`` path. Step 4A may convert it into a
    sealed candidate-only metadata batch; formal learning remains a later gate.
    """

    policy_observations: torch.Tensor
    policy_actions: torch.Tensor
    policy_log_probs: torch.Tensor
    policy_values: torch.Tensor
    policy_means: torch.Tensor
    policy_sigmas: torch.Tensor
    policy_row_indices: torch.Tensor
    t_env_actions: torch.Tensor
    continuation: torch.Tensor
    continuation_valid_mask: torch.Tensor
    frozen_gmt_env_actions: torch.Tensor
    actor_forward_count: int
    later_femr_action_count: int
    horizon_k: torch.Tensor
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    roles: tuple[str, ...]
    intent_q29: torch.Tensor
    intent_q29_provenance: tuple[str, ...]
    intent_q29_source: tuple[str, ...]
    executed_q29_t: torch.Tensor
    executed_q29_t_valid_mask: torch.Tensor
    done_any: torch.Tensor
    survival_steps: torch.Tensor

    def validate(self) -> None:
        """Fail closed unless the evidence encodes exactly one Repair policy row per scenario."""

        policy_count = int(self.policy_actions.shape[0])
        role_count = int(self.t_env_actions.shape[0])
        if int(self.actor_forward_count) != 1 or int(self.later_femr_action_count) != 0:
            raise ValueError("v015 one-action evidence requires exactly one actor forward and zero later FEMR actions")
        if policy_count <= 0 or role_count != 2 * policy_count:
            raise ValueError("v015 one-action evidence requires equal Repair/Noisy roles and one Repair policy row per scenario")
        if self.policy_actions.ndim != 2 or int(self.policy_actions.shape[1]) != 6:
            raise ValueError("v015 one-action evidence requires policy_actions [B,6]")
        if self.policy_observations.ndim != 2 or int(self.policy_observations.shape[0]) != policy_count:
            raise ValueError("v015 one-action evidence policy observations must align with Repair rows")
        vector_fields = {
            "policy_log_probs": self.policy_log_probs,
            "policy_values": self.policy_values,
            "policy_row_indices": self.policy_row_indices,
        }
        for name, value in vector_fields.items():
            if value.ndim != 1 or int(value.numel()) != policy_count:
                raise ValueError(f"v015 one-action evidence {name} must be [B]")
        for name, value in (("policy_means", self.policy_means), ("policy_sigmas", self.policy_sigmas)):
            if tuple(value.shape) != tuple(self.policy_actions.shape):
                raise ValueError(f"v015 one-action evidence {name} must be [B,6]")
        if (
            self.continuation.ndim != 3
            or int(self.continuation.shape[1]) != role_count
            or int(self.continuation.shape[2]) != 65
            or tuple(self.continuation_valid_mask.shape) != tuple(self.continuation.shape[:2])
            or self.frozen_gmt_env_actions.ndim != 3
            or tuple(self.frozen_gmt_env_actions.shape[:2]) != tuple(self.continuation.shape[:2])
        ):
            raise ValueError("v015 one-action evidence requires [K,N,65] C, [K,N] masks, and [K,N,A] frozen GMT actions")
        if tuple(self.horizon_k.shape) != (role_count,) or bool((self.horizon_k <= 0).any()):
            raise ValueError("v015 one-action evidence horizon_k must be positive [N]")
        if int(self.continuation.shape[0]) != int(self.horizon_k.max().item()):
            raise ValueError("v015 one-action evidence K dimension must equal max per-row horizon_k")
        expected_valid = torch.arange(
            int(self.continuation.shape[0]), device=self.horizon_k.device, dtype=torch.long
        ).unsqueeze(1) < self.horizon_k.unsqueeze(0)
        if not torch.equal(self.continuation_valid_mask.to(device=expected_valid.device, dtype=torch.bool), expected_valid):
            raise ValueError("v015 one-action evidence valid mask must exactly encode each K horizon")
        metadata = (self.scenario_ids, self.noisy_segment_hashes, self.x_t_identities, self.roles)
        if any(len(value) != role_count for value in metadata):
            raise ValueError("v015 one-action evidence metadata must cover every Repair/Noisy role row")
        if any(role not in {"repair", "noisy"} for role in self.roles):
            raise ValueError("v015 one-action evidence rejects Clean and legacy quartet roles")
        if (
            self.intent_q29.ndim != 3
            or int(self.intent_q29.shape[0]) != role_count
            or int(self.intent_q29.shape[1]) < 2
            or int(self.intent_q29.shape[2]) != 29
        ):
            raise ValueError("v015 one-action evidence requires intent_q29 [N,H+1,29] with H>=1")
        if tuple(self.executed_q29_t.shape) != (role_count, 29):
            raise ValueError("v015 one-action evidence requires post-t executed_q29_t [N,29]")
        for name, value in (
            ("executed_q29_t_valid_mask", self.executed_q29_t_valid_mask),
            ("done_any", self.done_any),
            ("survival_steps", self.survival_steps),
        ):
            if value.ndim != 1 or int(value.numel()) != role_count:
                raise ValueError(f"v015 one-action evidence {name} must be [N]")
        if (
            len(self.intent_q29_provenance) != role_count
            or len(self.intent_q29_source) != role_count
            or not bool(torch.isfinite(self.survival_steps.float()).all())
            or bool((self.survival_steps.float() < 0.0).any())
            or bool((self.survival_steps.float() > self.horizon_k.float()).any())
        ):
            raise ValueError("v015 one-action evidence has invalid q29 provenance or K survival evidence")
        repair_rows = torch.tensor(
            [index for index, role in enumerate(self.roles) if role == "repair"],
            device=self.policy_row_indices.device,
            dtype=torch.long,
        )
        if not torch.equal(self.policy_row_indices.to(dtype=torch.long), repair_rows):
            raise ValueError("v015 one-action evidence policy rows must be exactly the Repair role rows")
        rows_by_scenario: dict[str, list[int]] = {}
        for row, scenario_id in enumerate(self.scenario_ids):
            rows_by_scenario.setdefault(str(scenario_id), []).append(row)
        if len(rows_by_scenario) != policy_count:
            raise ValueError("v015 one-action evidence requires one scenario identity per Repair policy row")
        for scenario_id, rows in rows_by_scenario.items():
            if len(rows) != 2 or {self.roles[row] for row in rows} != {"repair", "noisy"}:
                raise ValueError(f"v015 one-action evidence scenario={scenario_id!r} must have one Repair and one Noisy row")
            left, right = rows
            if (
                self.noisy_segment_hashes[left] != self.noisy_segment_hashes[right]
                or self.x_t_identities[left] != self.x_t_identities[right]
                or int(self.horizon_k[left].item()) != int(self.horizon_k[right].item())
                or not torch.equal(self.continuation[:, left], self.continuation[:, right])
                or not torch.equal(self.intent_q29[left], self.intent_q29[right])
                or self.intent_q29_provenance[left] != self.intent_q29_provenance[right]
                or self.intent_q29_source[left] != self.intent_q29_source[right]
            ):
                raise ValueError(f"v015 one-action evidence scenario={scenario_id!r} mixes immutable local artifacts")
            provenance = self.intent_q29_provenance[left]
            source = self.intent_q29_source[left].lower()
            if provenance != "deployment_noisy_q29" or not source or any(
                token in source for token in ("clean", "root", "global")
            ):
                raise ValueError("v015 one-action evidence q29 target must retain deployment/Noisy provenance")
        tensors = (
            self.policy_observations,
            self.policy_actions,
            self.policy_log_probs,
            self.policy_values,
            self.policy_means,
            self.policy_sigmas,
            self.t_env_actions,
            self.continuation,
            self.continuation_valid_mask,
            self.frozen_gmt_env_actions,
            self.horizon_k,
            self.intent_q29,
            self.executed_q29_t,
            self.executed_q29_t_valid_mask,
            self.done_any,
            self.survival_steps,
        )
        if any(value.requires_grad for value in tensors):
            raise ValueError("v015 one-action evidence must be immutable detached capture data")


@dataclass(frozen=True)
class FrontRESV015PairedGainFacts:
    """将一个 Repair policy row 与其 Noisy baseline 配对给 v003 owner.

    状态: candidate-only local storage adapter.
    上游: immutable one-action capture.
    下游: FRS-GAIN-v003 input 和 one return/advantage carrier.
    证据: deterministic fake connectivity only, no formal storage write.
    """

    policy_observations: torch.Tensor
    policy_actions: torch.Tensor
    policy_log_probs: torch.Tensor
    policy_values: torch.Tensor
    policy_means: torch.Tensor
    policy_sigmas: torch.Tensor
    intent_q29: torch.Tensor
    repaired_q29: torch.Tensor
    noisy_q29: torch.Tensor
    intent_valid_mask: torch.Tensor
    repaired_success: torch.Tensor
    noisy_success: torch.Tensor
    repaired_survival: torch.Tensor
    noisy_survival: torch.Tensor
    horizon_k: torch.Tensor
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    intent_q29_provenance: str
    intent_q29_source: str

    def validate(self) -> None:
        count = int(self.policy_actions.shape[0])
        if count <= 0 or self.policy_actions.ndim != 2 or int(self.policy_actions.shape[1]) != 6:
            raise ValueError("v015 paired gain facts require policy_actions [B,6]")
        if self.policy_observations.ndim != 2 or int(self.policy_observations.shape[0]) != count:
            raise ValueError("v015 paired gain facts observations must align with policy rows")
        for name, value in (
            ("policy_log_probs", self.policy_log_probs),
            ("policy_values", self.policy_values),
            ("intent_valid_mask", self.intent_valid_mask),
            ("repaired_success", self.repaired_success),
            ("noisy_success", self.noisy_success),
            ("repaired_survival", self.repaired_survival),
            ("noisy_survival", self.noisy_survival),
            ("horizon_k", self.horizon_k),
        ):
            if value.ndim != 1 or int(value.numel()) != count:
                raise ValueError(f"v015 paired gain facts {name} must be [B]")
        for name, value in (
            ("policy_means", self.policy_means),
            ("policy_sigmas", self.policy_sigmas),
            ("intent_q29", self.intent_q29),
            ("repaired_q29", self.repaired_q29),
            ("noisy_q29", self.noisy_q29),
        ):
            expected = (count, 6) if name.startswith("policy_") else (count, 29)
            if tuple(value.shape) != expected:
                raise ValueError(f"v015 paired gain facts {name} must be {expected}")
        if (
            len(self.scenario_ids) != count
            or len(self.noisy_segment_hashes) != count
            or len(self.x_t_identities) != count
            or not bool((self.horizon_k > 0).all())
            or not bool(torch.isfinite(self.repaired_survival.float()).all())
            or not bool(torch.isfinite(self.noisy_survival.float()).all())
        ):
            raise ValueError("v015 paired gain facts have invalid identity or Physics evidence")
        source = self.intent_q29_source.lower()
        if (
            self.intent_q29_provenance != "deployment_noisy_q29"
            or not source
            or any(token in source for token in ("clean", "root", "global"))
        ):
            raise ValueError("v015 paired gain facts reject non-deployment q29 provenance")


@dataclass(frozen=True)
class FrontRESV015GainReturnEvidence:
    """从唯一 v003 Gain owner 构造 candidate-only one-row return evidence.

    这不是 legacy PPO batch, 不能传入 to_ppo_batch.
    Step 4A 可将其写入 candidate-only storage/grouped batch, 但 formal update
    仍属于后续 gate.
    """

    policy_observations: torch.Tensor
    policy_actions: torch.Tensor
    policy_log_probs: torch.Tensor
    policy_values: torch.Tensor
    policy_means: torch.Tensor
    policy_sigmas: torch.Tensor
    gain_total: torch.Tensor
    intent_gain: torch.Tensor
    physics_gain: torch.Tensor
    repair_cost: torch.Tensor
    return_k: torch.Tensor
    advantage_k: torch.Tensor
    policy_row_valid: torch.Tensor
    horizon_k: torch.Tensor
    evidence_valid_step_count: torch.Tensor
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    intent_q29_provenance: str
    intent_q29_source: str
    gain_source: str = "FRS-GAIN-v003-intent-physics-local-repair"

    def validate(self) -> None:
        count = int(self.policy_actions.shape[0])
        if count <= 0 or tuple(self.policy_actions.shape[1:]) != (6,):
            raise ValueError("v015 return evidence requires policy_actions [B,6]")
        for name, value in (
            ("policy_log_probs", self.policy_log_probs),
            ("policy_values", self.policy_values),
            ("gain_total", self.gain_total),
            ("intent_gain", self.intent_gain),
            ("physics_gain", self.physics_gain),
            ("repair_cost", self.repair_cost),
            ("return_k", self.return_k),
            ("advantage_k", self.advantage_k),
            ("policy_row_valid", self.policy_row_valid),
            ("horizon_k", self.horizon_k),
            ("evidence_valid_step_count", self.evidence_valid_step_count),
        ):
            if value.ndim != 1 or int(value.numel()) != count:
                raise ValueError(f"v015 return evidence {name} must be [B]")
        if (
            self.policy_observations.ndim != 2
            or int(self.policy_observations.shape[0]) != count
            or tuple(self.policy_means.shape) != (count, 6)
            or tuple(self.policy_sigmas.shape) != (count, 6)
            or len(self.scenario_ids) != count
            or len(self.noisy_segment_hashes) != count
            or len(self.x_t_identities) != count
            or self.gain_source != "FRS-GAIN-v003-intent-physics-local-repair"
            or bool((self.horizon_k <= 0).any())
            or bool((self.evidence_valid_step_count < 0).any())
            or bool((self.evidence_valid_step_count > self.horizon_k).any())
        ):
            raise ValueError("v015 return evidence has invalid policy tuple or Gain source")
        valid = self.policy_row_valid.bool()
        for name, value in (
            ("gain_total", self.gain_total),
            ("intent_gain", self.intent_gain),
            ("physics_gain", self.physics_gain),
            ("repair_cost", self.repair_cost),
            ("return_k", self.return_k),
            ("advantage_k", self.advantage_k),
        ):
            finite = torch.isfinite(value)
            if not bool(finite[valid].all()) or bool(finite[~valid].any()):
                raise ValueError(f"v015 return evidence {name} must be finite exactly on valid rows")
        source = self.intent_q29_source.lower()
        if (
            self.intent_q29_provenance != "deployment_noisy_q29"
            or not source
            or any(token in source for token in ("clean", "root", "global"))
        ):
            raise ValueError("v015 return evidence rejects non-deployment q29 provenance")


def pair_frontres_v015_gain_facts(evidence: FrontRESV015OneActionKEvidence) -> FrontRESV015PairedGainFacts:
    """提取对齐的 Repair/Noisy current-q29 facts, 不把 Clean C 用作 intent."""

    evidence.validate()
    repair_rows = evidence.policy_row_indices.to(dtype=torch.long)
    noisy_rows: list[int] = []
    for repair_row in repair_rows.tolist():
        scenario_id = evidence.scenario_ids[int(repair_row)]
        matches = [
            row
            for row, candidate_id in enumerate(evidence.scenario_ids)
            if candidate_id == scenario_id and evidence.roles[row] == "noisy"
        ]
        if len(matches) != 1:
            raise ValueError(f"v015 paired gain facts require one Noisy row for scenario={scenario_id!r}")
        noisy_rows.append(matches[0])
    noisy_index = torch.tensor(noisy_rows, device=repair_rows.device, dtype=torch.long)
    provenance = tuple(evidence.intent_q29_provenance[int(row)] for row in repair_rows.tolist())
    source = tuple(evidence.intent_q29_source[int(row)] for row in repair_rows.tolist())
    if len(set(provenance)) != 1 or len(set(source)) != 1:
        raise ValueError("v015 paired gain facts require one q29 provenance/source across the candidate batch")
    facts = FrontRESV015PairedGainFacts(
        policy_observations=evidence.policy_observations.detach().clone(),
        policy_actions=evidence.policy_actions.detach().clone(),
        policy_log_probs=evidence.policy_log_probs.detach().clone(),
        policy_values=evidence.policy_values.detach().clone(),
        policy_means=evidence.policy_means.detach().clone(),
        policy_sigmas=evidence.policy_sigmas.detach().clone(),
        intent_q29=evidence.intent_q29.index_select(0, repair_rows)[:, 0].detach().clone(),
        repaired_q29=evidence.executed_q29_t.index_select(0, repair_rows).detach().clone(),
        noisy_q29=evidence.executed_q29_t.index_select(0, noisy_index).detach().clone(),
        intent_valid_mask=(
            evidence.executed_q29_t_valid_mask.index_select(0, repair_rows).bool()
            & evidence.executed_q29_t_valid_mask.index_select(0, noisy_index).bool()
        ).detach().clone(),
        repaired_success=(~evidence.done_any.index_select(0, repair_rows).bool()).detach().clone(),
        noisy_success=(~evidence.done_any.index_select(0, noisy_index).bool()).detach().clone(),
        repaired_survival=evidence.survival_steps.index_select(0, repair_rows).detach().clone(),
        noisy_survival=evidence.survival_steps.index_select(0, noisy_index).detach().clone(),
        horizon_k=evidence.horizon_k.index_select(0, repair_rows).detach().clone(),
        scenario_ids=tuple(evidence.scenario_ids[int(row)] for row in repair_rows.tolist()),
        noisy_segment_hashes=tuple(evidence.noisy_segment_hashes[int(row)] for row in repair_rows.tolist()),
        x_t_identities=tuple(evidence.x_t_identities[int(row)] for row in repair_rows.tolist()),
        intent_q29_provenance=provenance[0],
        intent_q29_source=source[0],
    )
    facts.validate()
    return facts


def build_frontres_v015_gain_return_evidence(
    facts: FrontRESV015PairedGainFacts,
    gain_result: Any,
) -> FrontRESV015GainReturnEvidence:
    """只从 v003 为每个 Repair policy row 构造一个 return/advantage carrier."""

    facts.validate()
    count = int(facts.policy_actions.shape[0])
    components: dict[str, torch.Tensor] = {}
    for name in ("gain_total", "intent_gain", "physics_gain", "repair_cost"):
        value = getattr(gain_result, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or int(value.numel()) != count:
            raise ValueError(f"v015 return evidence requires FRS-GAIN-v003 {name} [B]")
        components[name] = value.detach().to(device=facts.policy_values.device, dtype=torch.float32).clone()
    if (
        getattr(gain_result, "intent_q29_provenance", None) != facts.intent_q29_provenance
        or getattr(gain_result, "intent_q29_source", None) != facts.intent_q29_source
    ):
        raise ValueError("v015 return evidence rejects a Gain result with mismatched q29 provenance")
    valid = facts.intent_valid_mask.bool()
    for value in components.values():
        valid = valid & torch.isfinite(value)
    nan = torch.full_like(components["gain_total"], float("nan"))
    masked_components = {
        name: torch.where(valid, value, nan)
        for name, value in components.items()
    }
    return_k = masked_components["gain_total"]
    advantage_k = torch.where(valid, return_k - facts.policy_values.detach().float(), nan)
    survival = facts.repaired_survival.detach().to(device=facts.policy_values.device, dtype=torch.float32)
    if not bool(torch.isfinite(survival).all()) or bool((survival < 0.0).any()):
        raise ValueError("v015 return evidence requires finite non-negative K survival evidence")
    evidence_valid_step_count = survival.to(dtype=torch.long)
    if not torch.equal(survival, evidence_valid_step_count.to(dtype=survival.dtype)):
        raise ValueError("v015 return evidence requires integer K survival-step counts")
    if bool((evidence_valid_step_count > facts.horizon_k).any()):
        raise ValueError("v015 return evidence survival-step count exceeds horizon_k")
    result = FrontRESV015GainReturnEvidence(
        policy_observations=facts.policy_observations.detach().clone(),
        policy_actions=facts.policy_actions.detach().clone(),
        policy_log_probs=facts.policy_log_probs.detach().clone(),
        policy_values=facts.policy_values.detach().clone(),
        policy_means=facts.policy_means.detach().clone(),
        policy_sigmas=facts.policy_sigmas.detach().clone(),
        gain_total=masked_components["gain_total"],
        intent_gain=masked_components["intent_gain"],
        physics_gain=masked_components["physics_gain"],
        repair_cost=masked_components["repair_cost"],
        return_k=return_k,
        advantage_k=advantage_k,
        policy_row_valid=valid,
        horizon_k=facts.horizon_k.detach().clone(),
        evidence_valid_step_count=evidence_valid_step_count.detach().clone(),
        scenario_ids=facts.scenario_ids,
        noisy_segment_hashes=facts.noisy_segment_hashes,
        x_t_identities=facts.x_t_identities,
        intent_q29_provenance=facts.intent_q29_provenance,
        intent_q29_source=facts.intent_q29_source,
    )
    result.validate()
    return result


def build_frontres_v015_grouped_candidate_storage(
    candidate_evidence: Any,
    *,
    transaction_id: str,
    policy_snapshot_id: str,
    motion_ids: tuple[str, ...],
    start_frames: torch.Tensor,
    segment_ids: torch.Tensor,
    source_index: torch.Tensor,
    trial_index: torch.Tensor,
) -> FrontRESSegmentStorageBatch:
    """Bind sealed v015 candidate evidence to one complete metadata-bearing storage batch.

    函数名说明:
        `build_frontres_v015_grouped_candidate_storage` 是 candidate-only storage
        adapter. 它不收集 rollout, 不选择 best-of-M, 不读取 priority 数值, 不执行
        PPO 或 optimizer.

    主链路:
        上游: Step 3B `FrontRESV015GainConsumerEvidence`.
        下游: `to_grouped_ppo_candidate_batch()` 的完整 v015 transaction batch.

    语义:
        每个 ordinary Repair attempt 只得到一个 `[B,6]` policy row. K 的实际
        survival/evidence count 作为 metadata 保留, 不能复制 row 或改变 actor mass.
    """

    # B1: 验证 sealed Gain carrier 与 Repair row 仍是同一 local scenario.
    validate = getattr(candidate_evidence, "validate", None)
    if not callable(validate):
        raise TypeError("v015 grouped candidate adapter requires validated Step 3B evidence")
    validate()
    return_evidence = getattr(candidate_evidence, "return_evidence", None)
    one_action = getattr(candidate_evidence, "one_action", None)
    validate_return = getattr(return_evidence, "validate", None)
    validate_one_action = getattr(one_action, "validate", None)
    if not callable(validate_return) or not callable(validate_one_action):
        raise TypeError("v015 grouped candidate adapter requires return and one-action evidence")
    validate_return()
    validate_one_action()
    repair_rows = getattr(one_action, "policy_row_indices", None)
    if not isinstance(repair_rows, torch.Tensor) or repair_rows.ndim != 1:
        raise ValueError("v015 grouped candidate adapter requires one Repair-row index per policy tuple")
    count = int(return_evidence.policy_actions.shape[0])
    if int(repair_rows.numel()) != count:
        raise ValueError("v015 grouped candidate adapter Repair-row count disagrees with return evidence")
    expected_horizon = one_action.horizon_k.index_select(0, repair_rows.to(dtype=torch.long))
    expected_survival = one_action.survival_steps.index_select(0, repair_rows.to(dtype=torch.long)).to(
        device=return_evidence.policy_actions.device,
        dtype=torch.float32,
    )
    expected_steps = expected_survival.to(dtype=torch.long)
    if (
        not torch.equal(expected_survival, expected_steps.to(dtype=expected_survival.dtype))
        or not torch.equal(expected_horizon.to(device=return_evidence.horizon_k.device, dtype=torch.long), return_evidence.horizon_k)
        or not torch.equal(return_evidence.evidence_valid_step_count, expected_steps)
        or tuple(one_action.scenario_ids[int(row)] for row in repair_rows.tolist()) != return_evidence.scenario_ids
        or tuple(one_action.noisy_segment_hashes[int(row)] for row in repair_rows.tolist()) != return_evidence.noisy_segment_hashes
        or tuple(one_action.x_t_identities[int(row)] for row in repair_rows.tolist()) != return_evidence.x_t_identities
    ):
        raise ValueError("v015 grouped candidate adapter lost one-action local scenario identity or K evidence")

    # B2: 封存 transaction/motion/Segment/trial 与 local scenario row metadata.
    metadata = FrontRESV015GroupedCandidateMetadata(
        transaction_id=transaction_id,
        policy_snapshot_id=policy_snapshot_id,
        motion_ids=motion_ids,
        start_frames=start_frames,
        segment_ids=segment_ids,
        source_index=source_index,
        trial_index=trial_index,
        horizon_k=return_evidence.horizon_k,
        evidence_valid_step_count=return_evidence.evidence_valid_step_count,
        trial_role=("policy",) * count,
        noisy_segment_hashes=return_evidence.noisy_segment_hashes,
        scenario_ids=return_evidence.scenario_ids,
        x_t_identities=return_evidence.x_t_identities,
        intent_q29_provenance=return_evidence.intent_q29_provenance,
        intent_q29_source=return_evidence.intent_q29_source,
    )
    metadata.validate()

    # B3: 构造 one-row storage batch, metadata 只供 grouped candidate path 消费.
    device = return_evidence.policy_actions.device
    storage_batch = FrontRESSegmentStorageBatch(
        observations=return_evidence.policy_observations.detach().clone(),
        actions=return_evidence.policy_actions.detach().clone(),
        old_log_probs=return_evidence.policy_log_probs.detach().clone(),
        old_values=return_evidence.policy_values.detach().clone(),
        rewards=return_evidence.gain_total.detach().clone(),
        returns=return_evidence.return_k.detach().clone(),
        advantages=return_evidence.advantage_k.detach().clone(),
        valid_mask=return_evidence.policy_row_valid.detach().clone(),
        segment_ids=metadata.segment_ids.to(device=device, dtype=torch.long).detach().clone(),
        old_means=return_evidence.policy_means.detach().clone(),
        old_sigmas=return_evidence.policy_sigmas.detach().clone(),
        transaction_metadata=metadata,
        transaction_row_indices=torch.arange(count, device=device, dtype=torch.long),
    )
    return storage_batch


@dataclass(frozen=True)
class FrontRESSegmentStorageStats:
    size: int
    capacity: int
    valid_frac: float
    reset_success_frac: float
    reward_mean: float
    advantage_mean: float


class FrontRESSegmentRolloutStorage:
    """Independent Stage 3 storage for Segment Replay HRL.

    Segment rewards are already K-step rollout outcomes, so returns default to
    reward and advantages default to reward minus stored value.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Iterable[int] | torch.Size,
        action_dim: int = 6,
        privileged_obs_shape: Iterable[int] | torch.Size | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if action_dim != 6:
            raise ValueError(f"Segment Replay HRL action_dim must be 6, got {action_dim}")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs_shape = tuple(obs_shape)
        self.privileged_obs_shape = tuple(privileged_obs_shape) if privileged_obs_shape is not None else None
        self.action_dim = int(action_dim)
        self.step = 0
        self.segment_source: list[str] = []
        self.priority_evidence: list[Any] = []
        self.audit_transaction_id: str | None = None
        self.audit_batch_signature: str | None = None
        self.audit_identity_state = "UNCONFIRMED"
        self.transaction_metadata: Any | None = None

        self.observations = torch.zeros(self.capacity, *self.obs_shape, device=self.device)
        self.privileged_observations = (
            torch.zeros(self.capacity, *self.privileged_obs_shape, device=self.device)
            if self.privileged_obs_shape is not None
            else None
        )
        self.actions = torch.zeros(self.capacity, 6, device=self.device)
        self.old_log_probs = torch.zeros(self.capacity, device=self.device)
        self.old_values = torch.zeros(self.capacity, device=self.device)
        self.old_means = torch.zeros(self.capacity, 6, device=self.device)
        self.old_sigmas = torch.zeros(self.capacity, 6, device=self.device)
        self.rewards = torch.zeros(self.capacity, device=self.device)
        self.returns = torch.zeros(self.capacity, device=self.device)
        self.advantages = torch.zeros(self.capacity, device=self.device)
        self.valid_mask = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.reset_mask = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.segment_ids = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.transaction_row_indices = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)

    def add_transition(self, transition: FrontRESSegmentTransition) -> None:
        transition = self._normalize_transition(transition)
        batch_size = int(transition.actions.shape[0])
        transaction_row_indices = _resolve_transaction_row_indices(transition, batch_size=batch_size)
        identity = (
            transition.audit_transaction_id,
            transition.audit_batch_signature,
            transition.audit_identity_state,
        )
        if self.step == 0:
            self.audit_transaction_id, self.audit_batch_signature, self.audit_identity_state = identity
        elif identity != (self.audit_transaction_id, self.audit_batch_signature, self.audit_identity_state):
            raise ValueError(
                "Segment storage received rows from different rollout transactions: "
                f"existing={(self.audit_transaction_id, self.audit_batch_signature, self.audit_identity_state)!r} "
                f"incoming={identity!r}"
            )
        if self.step == 0:
            self.transaction_metadata = transition.transaction_metadata
        elif transition.transaction_metadata is not self.transaction_metadata:
            raise ValueError("Segment storage requires one identical frozen transaction metadata carrier")
        if self.step + batch_size > self.capacity:
            raise OverflowError("FrontRESSegmentRolloutStorage overflow; call clear() before adding more transitions")
        sl = slice(self.step, self.step + batch_size)
        self.observations[sl].copy_(transition.observations)
        if self.privileged_observations is not None:
            if transition.privileged_observations is None:
                raise ValueError("privileged_observations are required by this storage")
            self.privileged_observations[sl].copy_(transition.privileged_observations)
        self.actions[sl].copy_(transition.actions)
        self.old_log_probs[sl].copy_(transition.old_log_probs)
        self.old_values[sl].copy_(transition.values)
        self.rewards[sl].copy_(transition.rewards)
        returns = transition.returns if transition.returns is not None else transition.rewards
        advantages = transition.advantages if transition.advantages is not None else returns - transition.values
        self.returns[sl].copy_(returns)
        self.advantages[sl].copy_(advantages)
        self.valid_mask[sl].copy_(transition.valid_mask & transition.reset_mask)
        self.reset_mask[sl].copy_(transition.reset_mask)
        self.segment_ids[sl].copy_(transition.segment_ids)
        if transaction_row_indices is not None:
            self.transaction_row_indices[sl].copy_(transaction_row_indices)
        if transition.old_means is not None:
            self.old_means[sl].copy_(transition.old_means)
        if transition.old_sigmas is not None:
            self.old_sigmas[sl].copy_(transition.old_sigmas)
        self.segment_source.extend(transition.segment_source or ("unknown",) * batch_size)
        if transition.priority_evidence is not None:
            self.priority_evidence.append(_detach_evidence(transition.priority_evidence))
        self.step += batch_size

    def write(self, **payload: Any) -> None:
        self.add_transition(self.transition_from_connector_payload(payload))

    def transition_from_connector_payload(self, payload: dict[str, Any]) -> FrontRESSegmentTransition:
        batch = payload["batch"]
        repair_action = payload["repair_action"]
        reward_result = payload["reward_result"]
        reset_result = payload["reset_result"]
        sample = payload.get("sample")
        policy_output = payload.get("policy_output") or payload.get("raw_action")
        observations = _required_attr_or_key(policy_output, "observations")
        old_log_probs = _required_attr_or_key(policy_output, "log_prob")
        values = _required_attr_or_key(policy_output, "value")
        old_means = _optional_attr_or_key(policy_output, "mean")
        old_sigmas = _optional_attr_or_key(policy_output, "sigma")
        return FrontRESSegmentTransition(
            observations=observations,
            actions=repair_action.projected_delta_se,
            old_log_probs=old_log_probs,
            values=values,
            rewards=reward_result.reward,
            valid_mask=reward_result.valid_mask,
            reset_mask=_required_reset_mask(reset_result),
            segment_ids=batch.segment_ids,
            segment_source=getattr(sample, "source", None),
            privileged_observations=_optional_attr_or_key(policy_output, "privileged_observations"),
            old_means=old_means,
            old_sigmas=old_sigmas,
            priority_evidence=payload.get("priority_evidence"),
            audit_transaction_id=payload.get("audit_transaction_id"),
            audit_batch_signature=payload.get("audit_batch_signature"),
            audit_identity_state=str(payload.get("audit_identity_state", "UNCONFIRMED")),
            transaction_metadata=payload.get("transaction_metadata"),
            transaction_row_indices=payload.get("transaction_row_indices"),
        )

    def compute_returns_and_advantages(
        self,
        *,
        reward_steps: torch.Tensor | None = None,
        done_steps: torch.Tensor | None = None,
        horizon: int | torch.Tensor | None = None,
        gamma: float = 1.0,
    ) -> None:
        # QUALITY-CREDIT-01: 检查 canonical Gain steps -> returns/advantages -> PPO batch.
        # Result: PENDING_Q_EVIDENCE.
        # B1: policy-row effective K 与 Gain steps 在 return aggregation 前冻结.
        # B2: sign-preserving credit 与 valid mask 决定每行 advantage.
        # B3: to_ppo_batch 前统计 sign、bucket contribution 与 dominance.
        """按每行 K 和 done 边界累计 Segment return/advantage.

        函数名说明:
            `compute_returns_and_advantages` 是 Segment temporal-credit owner, 从
            canonical per-step Gain 构造 PPO 学习信号; 它不是 Gain 公式或 PPO loss.

        主链路:
            上游: rollout capture 提供 `[T, B]` Gain, done 和 per-row horizon K.
            下游: finalized returns/advantages 经 `to_ppo_batch` 进入 PPO.

        语义:
            每行只累计 alive 且 `offset < K` 的 paired Gain. Advantage 保持
            `return - old_value`, 不混入 full environment reward.
        """
        # B1: 读取 canonical per-step Gain 和每行 effective K.
        storage_slice = slice(0, self.step)
        if reward_steps is None:
            self.returns[storage_slice].copy_(self.rewards[storage_slice])
            self.advantages[storage_slice].copy_(self.returns[storage_slice] - self.old_values[storage_slice])
            return

        if reward_steps.ndim != 2:
            raise ValueError(f"reward_steps must be rank-2 [T, B], got {tuple(reward_steps.shape)}")
        if int(reward_steps.shape[1]) < self.step:
            raise ValueError(f"reward_steps must have at least {self.step} batch entries, got {int(reward_steps.shape[1])}")
        if done_steps is not None and tuple(done_steps.shape) != tuple(reward_steps.shape):
            raise ValueError(f"done_steps shape {tuple(done_steps.shape)} must match reward_steps {tuple(reward_steps.shape)}")

        step_count = int(reward_steps.shape[0])
        if isinstance(horizon, torch.Tensor):
            horizon_k = horizon.to(device=self.device, dtype=torch.long).reshape(-1)
            if int(horizon_k.numel()) != self.step:
                raise ValueError(f"horizon must have {self.step} rows, got {int(horizon_k.numel())}")
            horizon_k = horizon_k.clamp(min=1, max=step_count)
        else:
            scalar_horizon = min(step_count, max(1, int(horizon if horizon is not None else step_count)))
            horizon_k = torch.full((self.step,), scalar_horizon, dtype=torch.long, device=self.device)
        return_horizon = int(horizon_k.max().item())
        rewards = reward_steps[:return_horizon, : self.step].to(device=self.device, dtype=torch.float32)
        if done_steps is None:
            dones = torch.zeros_like(rewards, dtype=torch.bool)
        else:
            dones = done_steps[:return_horizon, : self.step].to(device=self.device).bool()

        # B2: 仅在每行仍 alive 且位于 K 内时累计 discounted Gain.
        returns = torch.zeros(self.step, device=self.device, dtype=torch.float32)
        alive = torch.ones(self.step, device=self.device, dtype=torch.float32)
        discount = 1.0
        gamma_value = float(gamma)
        for offset in range(return_horizon):
            horizon_active = offset < horizon_k
            returns = returns + (discount * alive * horizon_active.float() * rewards[offset])
            alive = alive * (~(dones[offset] & horizon_active)).float()
            discount *= gamma_value

        self.returns[storage_slice].copy_(returns)
        self.advantages[storage_slice].copy_(self.returns[storage_slice] - self.old_values[storage_slice])
        # B3: finalized returns/advantages 已准备好进入 PPO batch conversion.

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 1,
        shuffle: bool = True,
    ):
        if self.step == 0:
            raise RuntimeError("cannot generate mini-batches from empty segment storage")
        if num_mini_batches <= 0:
            raise ValueError(f"num_mini_batches must be positive, got {num_mini_batches}")
        total = self.step
        mini_batch_size = max(1, (total + num_mini_batches - 1) // num_mini_batches)
        for _ in range(num_epochs):
            indices = torch.randperm(total, device=self.device) if shuffle else torch.arange(total, device=self.device)
            for start in range(0, total, mini_batch_size):
                idx = indices[start : min(start + mini_batch_size, total)]
                yield self._batch(idx)

    def full_batch(self) -> FrontRESSegmentStorageBatch:
        return self._batch(torch.arange(self.step, device=self.device))

    def stats(self) -> FrontRESSegmentStorageStats:
        if self.step == 0:
            return FrontRESSegmentStorageStats(0, self.capacity, 0.0, 0.0, 0.0, 0.0)
        active = slice(0, self.step)
        return FrontRESSegmentStorageStats(
            size=self.step,
            capacity=self.capacity,
            valid_frac=float(self.valid_mask[active].float().mean().item()),
            reset_success_frac=float(self.reset_mask[active].float().mean().item()),
            reward_mean=float(self.rewards[active].mean().item()),
            advantage_mean=float(self.advantages[active].mean().item()),
        )

    def clear(self) -> None:
        self.step = 0
        self.segment_source.clear()
        self.priority_evidence.clear()
        self.audit_transaction_id = None
        self.audit_batch_signature = None
        self.audit_identity_state = "UNCONFIRMED"
        self.transaction_metadata = None
        self.transaction_row_indices.fill_(-1)

    def state_dict(self) -> dict[str, Any]:
        active = slice(0, self.step)
        return {
            "step": self.step,
            "observations": self.observations[active].detach().cpu(),
            "privileged_observations": self.privileged_observations[active].detach().cpu()
            if self.privileged_observations is not None
            else None,
            "actions": self.actions[active].detach().cpu(),
            "old_log_probs": self.old_log_probs[active].detach().cpu(),
            "old_values": self.old_values[active].detach().cpu(),
            "old_means": self.old_means[active].detach().cpu(),
            "old_sigmas": self.old_sigmas[active].detach().cpu(),
            "rewards": self.rewards[active].detach().cpu(),
            "returns": self.returns[active].detach().cpu(),
            "advantages": self.advantages[active].detach().cpu(),
            "valid_mask": self.valid_mask[active].detach().cpu(),
            "reset_mask": self.reset_mask[active].detach().cpu(),
            "segment_ids": self.segment_ids[active].detach().cpu(),
            "segment_source": tuple(self.segment_source),
            "priority_evidence": tuple(self.priority_evidence),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        step = int(state["step"])
        if step > self.capacity:
            raise ValueError(f"stored step {step} exceeds capacity {self.capacity}")
        self.clear()
        self.step = step
        active = slice(0, self.step)
        for name in (
            "observations",
            "actions",
            "old_log_probs",
            "old_values",
            "old_means",
            "old_sigmas",
            "rewards",
            "returns",
            "advantages",
            "valid_mask",
            "reset_mask",
            "segment_ids",
        ):
            getattr(self, name)[active].copy_(state[name].to(self.device))
        if self.privileged_observations is not None and state.get("privileged_observations") is not None:
            self.privileged_observations[active].copy_(state["privileged_observations"].to(self.device))
        self.segment_source = list(state.get("segment_source", ("unknown",) * self.step))
        self.priority_evidence = list(state.get("priority_evidence", ()))

    def _batch(self, idx: torch.Tensor) -> FrontRESSegmentStorageBatch:
        privileged = self.privileged_observations[idx] if self.privileged_observations is not None else None
        return FrontRESSegmentStorageBatch(
            observations=self.observations[idx],
            privileged_observations=privileged,
            actions=self.actions[idx],
            old_log_probs=self.old_log_probs[idx],
            old_values=self.old_values[idx],
            old_means=self.old_means[idx],
            old_sigmas=self.old_sigmas[idx],
            rewards=self.rewards[idx],
            returns=self.returns[idx],
            advantages=self.advantages[idx],
            valid_mask=self.valid_mask[idx],
            segment_ids=self.segment_ids[idx],
            audit_transaction_id=self.audit_transaction_id,
            audit_batch_signature=self.audit_batch_signature,
            audit_identity_state=self.audit_identity_state,
            transaction_metadata=self.transaction_metadata,
            transaction_row_indices=(self.transaction_row_indices[idx] if self.transaction_metadata is not None else None),
        )

    def _normalize_transition(self, transition: FrontRESSegmentTransition) -> FrontRESSegmentTransition:
        if transition.actions.ndim != 2 or transition.actions.shape[-1] != 6:
            raise ValueError(f"actions must have shape [B, 6], got {tuple(transition.actions.shape)}")
        batch_size = transition.actions.shape[0]
        _require_batch("observations", transition.observations, batch_size)
        for name in ("old_log_probs", "values", "rewards", "valid_mask", "reset_mask", "segment_ids"):
            _require_vector(name, getattr(transition, name), batch_size)
        if transition.segment_source is not None and len(transition.segment_source) != batch_size:
            raise ValueError("segment_source length must match batch size")
        if transition.privileged_observations is not None:
            _require_batch("privileged_observations", transition.privileged_observations, batch_size)
        for name in ("old_means", "old_sigmas"):
            value = getattr(transition, name)
            if value is not None and tuple(value.shape) != (batch_size, 6):
                raise ValueError(f"{name} must have shape [B, 6], got {tuple(value.shape)}")
        for name in ("returns", "advantages"):
            value = getattr(transition, name)
            if value is not None:
                _require_vector(name, value, batch_size)
        return FrontRESSegmentTransition(
            observations=transition.observations.to(self.device).detach(),
            actions=transition.actions.to(self.device).detach(),
            old_log_probs=transition.old_log_probs.to(self.device).detach(),
            values=transition.values.to(self.device).detach(),
            rewards=transition.rewards.to(self.device).detach(),
            valid_mask=transition.valid_mask.to(self.device).bool().detach(),
            reset_mask=transition.reset_mask.to(self.device).bool().detach(),
            segment_ids=transition.segment_ids.to(self.device, dtype=torch.long).detach(),
            segment_source=transition.segment_source,
            privileged_observations=transition.privileged_observations.to(self.device).detach()
            if transition.privileged_observations is not None
            else None,
            old_means=transition.old_means.to(self.device).detach() if transition.old_means is not None else None,
            old_sigmas=transition.old_sigmas.to(self.device).detach() if transition.old_sigmas is not None else None,
            returns=transition.returns.to(self.device).detach() if transition.returns is not None else None,
            advantages=transition.advantages.to(self.device).detach() if transition.advantages is not None else None,
            priority_evidence=transition.priority_evidence,
            audit_transaction_id=transition.audit_transaction_id,
            audit_batch_signature=transition.audit_batch_signature,
            audit_identity_state=transition.audit_identity_state,
            transaction_metadata=transition.transaction_metadata,
            transaction_row_indices=(
                transition.transaction_row_indices.to(self.device, dtype=torch.long).detach()
                if transition.transaction_row_indices is not None
                else None
            ),
        )


def _resolve_transaction_row_indices(
    transition: FrontRESSegmentTransition,
    *,
    batch_size: int,
) -> torch.Tensor | None:
    """Return the sealed metadata row for every storage row without rebuilding identity."""

    metadata = transition.transaction_metadata
    raw_indices = transition.transaction_row_indices
    if metadata is None:
        if raw_indices is not None:
            raise ValueError("transaction_row_indices require transaction_metadata")
        return None
    validate = getattr(metadata, "validate", None)
    if not callable(validate):
        raise ValueError("transaction metadata requires validate()")
    validate()
    metadata_segment_ids = getattr(metadata, "segment_ids", None)
    metadata_size = int(getattr(metadata, "batch_size", 0) or 0)
    if not isinstance(metadata_segment_ids, torch.Tensor):
        raise ValueError("transaction metadata requires segment_ids")
    if metadata_size <= 0:
        metadata_size = int(metadata_segment_ids.numel())
    if metadata_segment_ids.ndim != 1 or int(metadata_segment_ids.numel()) != metadata_size:
        raise ValueError("transaction metadata segment_ids must be a rank-1 metadata vector")
    if raw_indices is None:
        if metadata_size != batch_size:
            raise ValueError(
                "transaction metadata requires explicit transaction_row_indices when storage rows are a subset"
            )
        row_indices = torch.arange(batch_size, device=transition.segment_ids.device, dtype=torch.long)
    else:
        _require_vector("transaction_row_indices", raw_indices, batch_size)
        if raw_indices.requires_grad:
            raise ValueError("transaction_row_indices must be detached")
        row_indices = raw_indices.detach().to(device=transition.segment_ids.device, dtype=torch.long)
        if bool((row_indices < 0).any()) or bool((row_indices >= metadata_size).any()):
            raise ValueError("transaction_row_indices contain an out-of-range metadata row")
    selected_segment_ids = metadata_segment_ids.detach().to(
        device=transition.segment_ids.device,
        dtype=torch.long,
    )[row_indices]
    if not torch.equal(selected_segment_ids, transition.segment_ids):
        raise ValueError("transaction metadata segment_ids do not match the storage rows")
    return row_indices


def _storage_batch_transaction_row_indices(batch: FrontRESSegmentStorageBatch) -> torch.Tensor | None:
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


def _require_batch(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{name} batch dimension must be {batch_size}, got {tensor.shape[0]}")


def _require_vector(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [B], got {tuple(tensor.shape)}")


def _required_attr_or_key(obj: Any, name: str) -> torch.Tensor:
    value = _optional_attr_or_key(obj, name)
    if value is None:
        raise ValueError(f"connector payload must provide policy {name}")
    return value


def _required_reset_mask(reset_result: Any) -> torch.Tensor:
    value = _optional_attr_or_key(reset_result, "success_mask")
    if value is None:
        value = _optional_attr_or_key(reset_result, "valid_mask")
    if value is None:
        raise ValueError("connector payload reset_result must provide success_mask or valid_mask")
    return value


def _optional_attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _detach_evidence(evidence: Any) -> Any:
    if isinstance(evidence, torch.Tensor):
        return evidence.detach().cpu()
    if isinstance(evidence, dict):
        return {key: _detach_evidence(value) for key, value in evidence.items()}
    if hasattr(evidence, "__dict__"):
        return {
            key: _detach_evidence(value)
            for key, value in vars(evidence).items()
            if not key.startswith("_")
        }
    return evidence
