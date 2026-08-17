"""Frozen-policy and exact-M transaction aggregate for FrontRES Segment Replay.

This module owns immutable transaction identity plus collection/seal lifecycle.
It does not sample scenarios, reset the simulator, compute Gain/PPO, or step an
optimizer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable

import torch

from rsl_rl.frontres.frontres_segment_planning import FrontRESFrozenPolicyTransactionPlan
from rsl_rl.frontres.frontres_segment_warmup import FRONTRES_V011_SELECTED_SEGMENT_COUNT


def _frontres_policy_state_fingerprint(policy: Any) -> tuple[str, int, int]:
    state_dict = getattr(policy, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("frozen policy snapshot requires runner.alg.policy.state_dict()")
    state = state_dict()
    if not hasattr(state, "items"):
        raise TypeError("policy.state_dict() must return a mapping")
    digest = hashlib.sha256()
    tensor_count = 0
    tensor_numel = 0
    for raw_name, value in sorted(state.items(), key=lambda item: str(item[0])):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"policy.state_dict()[{raw_name!r}] must be a tensor")
        if value.layout != torch.strided:
            raise TypeError(f"policy.state_dict()[{raw_name!r}] must use strided layout")
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(str(raw_name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        tensor_count += 1
        tensor_numel += int(tensor.numel())
    if tensor_count <= 0:
        raise ValueError("frozen policy snapshot requires at least one state tensor")
    return digest.hexdigest(), tensor_count, tensor_numel


@dataclass(frozen=True)
class FrontRESFrozenPolicySnapshot:
    """Identity-only proof that the pre-collection policy state stays unchanged."""

    transaction_id: str
    policy_snapshot_id: str
    policy_state_hash: str
    policy_state_tensor_count: int
    policy_state_numel: int

    def validate(self) -> None:
        if not self.transaction_id:
            raise ValueError("frozen policy snapshot transaction_id must be non-empty")
        if not self.policy_snapshot_id:
            raise ValueError("frozen policy snapshot policy_snapshot_id must be non-empty")
        if len(self.policy_state_hash) != 64 or any(char not in "0123456789abcdef" for char in self.policy_state_hash):
            raise ValueError("frozen policy snapshot must carry a sha256 policy_state_hash")
        if int(self.policy_state_tensor_count) <= 0 or int(self.policy_state_numel) < 0:
            raise ValueError("frozen policy snapshot has invalid state tensor metadata")
        expected_id = f"{self.transaction_id}:pi-{self.policy_state_hash[:16]}"
        if self.policy_snapshot_id != expected_id:
            raise ValueError("policy_snapshot_id must be derived from the actual policy state hash")

    def verify_policy(self, policy: Any) -> None:
        self.validate()
        observed_hash, observed_count, observed_numel = _frontres_policy_state_fingerprint(policy)
        observed = (observed_hash, observed_count, observed_numel)
        expected = (self.policy_state_hash, int(self.policy_state_tensor_count), int(self.policy_state_numel))
        if observed != expected:
            raise RuntimeError(
                "frozen policy snapshot mismatch: "
                f"expected={expected!r} observed={observed!r}; collection must stop before storage"
            )


def capture_frontres_frozen_policy_snapshot(
    runner: Any,
    *,
    transaction_id: str,
) -> FrontRESFrozenPolicySnapshot:
    """Capture an identity-only old-policy fingerprint without cloning or mutating the actor."""

    transaction_id = str(transaction_id)
    if not transaction_id:
        raise ValueError("transaction_id must be non-empty")
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    state_hash, tensor_count, tensor_numel = _frontres_policy_state_fingerprint(policy)
    snapshot = FrontRESFrozenPolicySnapshot(
        transaction_id=transaction_id,
        policy_snapshot_id=f"{transaction_id}:pi-{state_hash[:16]}",
        policy_state_hash=state_hash,
        policy_state_tensor_count=tensor_count,
        policy_state_numel=tensor_numel,
    )
    snapshot.validate()
    return snapshot


def _immutable_v015_formal_transaction_vector(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got {tuple(value.shape)}")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached immutable transaction metadata")
    return value.detach().to(device="cpu", dtype=torch.long).clone().contiguous()


@dataclass(frozen=True)
class FrontRESFormalTransactionPlan:
    """Immutable expected-row contract for one v015 multi-Segment x M update.

    这个对象只描述已经选定的 local scenario rows. 它不 materialize/reset
    scenario, 不采样动作, 不写 storage, 不调用 simulator 或 optimizer.
    """

    snapshot: FrontRESFrozenPolicySnapshot
    motion_ids: tuple[str, ...]
    start_frames: torch.Tensor
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    intent_q29_provenance: str
    intent_q29_source: str
    _integrity_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, FrontRESFrozenPolicySnapshot):
            raise TypeError("v015 formal transaction plan requires FrontRESFrozenPolicySnapshot")
        object.__setattr__(self, "motion_ids", tuple(str(value) for value in self.motion_ids))
        object.__setattr__(self, "scenario_ids", tuple(str(value) for value in self.scenario_ids))
        object.__setattr__(self, "noisy_segment_hashes", tuple(str(value) for value in self.noisy_segment_hashes))
        object.__setattr__(self, "x_t_identities", tuple(str(value) for value in self.x_t_identities))
        object.__setattr__(self, "intent_q29_provenance", str(self.intent_q29_provenance))
        object.__setattr__(self, "intent_q29_source", str(self.intent_q29_source))
        for name in ("start_frames", "segment_ids", "source_index", "trial_index", "horizon_k"):
            object.__setattr__(self, name, _immutable_v015_formal_transaction_vector(name, getattr(self, name)))
        object.__setattr__(self, "_integrity_hash", _v015_formal_transaction_plan_hash(self))
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.segment_ids.numel())

    @property
    def transaction_id(self) -> str:
        return self.snapshot.transaction_id

    @property
    def policy_snapshot_id(self) -> str:
        return self.snapshot.policy_snapshot_id

    @property
    def selected_segment_count(self) -> int:
        return int(torch.unique(self.source_index).numel())

    @property
    def active_m(self) -> int:
        counts = torch.bincount(self.source_index, minlength=self.selected_segment_count)
        if counts.numel() == 0 or int(torch.unique(counts).numel()) != 1:
            raise ValueError("FRS-TRAIN-v024 formal transaction requires one exact M across all Scenarios")
        return int(counts[0].item())

    def _row_identity(self, row: int) -> tuple[str, int, int, int, str, str, str]:
        return (
            self.motion_ids[row],
            int(self.start_frames[row].item()),
            int(self.segment_ids[row].item()),
            int(self.horizon_k[row].item()),
            self.scenario_ids[row],
            self.noisy_segment_hashes[row],
            self.x_t_identities[row],
        )

    def validate(self) -> None:
        self.snapshot.validate()
        count = self.batch_size
        if count <= 0:
            raise ValueError("v015 formal transaction plan requires at least one expected policy row")
        for name, value in (
            ("motion_ids", self.motion_ids),
            ("start_frames", self.start_frames),
            ("segment_ids", self.segment_ids),
            ("source_index", self.source_index),
            ("trial_index", self.trial_index),
            ("horizon_k", self.horizon_k),
            ("scenario_ids", self.scenario_ids),
            ("noisy_segment_hashes", self.noisy_segment_hashes),
            ("x_t_identities", self.x_t_identities),
        ):
            row_count = len(value) if isinstance(value, tuple) else int(value.numel())
            if row_count != count:
                raise ValueError(f"v015 formal transaction plan {name} must have {count} rows")
        source_name = self.intent_q29_source.lower()
        if (
            self.intent_q29_provenance != "deployment_noisy_q29"
            or not source_name
            or any(token in source_name for token in ("clean", "root", "global"))
        ):
            raise ValueError("v015 formal transaction plan rejects non-deployment q29 provenance")
        if (
            any(not value for value in self.motion_ids)
            or any(not value for value in self.scenario_ids)
            or any(not value for value in self.noisy_segment_hashes)
            or any(not value for value in self.x_t_identities)
            or bool((self.start_frames < 0).any())
            or bool((self.segment_ids < 0).any())
            or bool((self.source_index < 0).any())
            or bool((self.trial_index < 0).any())
            or bool((self.horizon_k <= 0).any())
        ):
            raise ValueError("v015 formal transaction plan has invalid row identity")
        source_rows: dict[int, list[int]] = {}
        seen_attempts: set[tuple[int, int]] = set()
        for row in range(count):
            source = int(self.source_index[row].item())
            trial = int(self.trial_index[row].item())
            key = (source, trial)
            if key in seen_attempts:
                raise ValueError("v015 formal transaction plan has duplicate source/trial attempts")
            seen_attempts.add(key)
            source_rows.setdefault(source, []).append(row)
        expected_sources = list(range(FRONTRES_V011_SELECTED_SEGMENT_COUNT))
        if len(source_rows) != FRONTRES_V011_SELECTED_SEGMENT_COUNT or sorted(source_rows) != expected_sources:
            raise ValueError("FRS-TRAIN-v024 formal transaction plan requires source_index exactly {0,...,7}")
        attempt_counts = {len(rows) for rows in source_rows.values()}
        if len(attempt_counts) != 1 or next(iter(attempt_counts)) < 2:
            raise ValueError("FRS-TRAIN-v024 formal transaction plan requires one exact M>=2 across all Scenarios")
        for source, rows in source_rows.items():
            expected_trials = list(range(len(rows)))
            observed_trials = sorted(int(self.trial_index[row].item()) for row in rows)
            if observed_trials != expected_trials:
                raise ValueError(f"v015 source_index={source} trial_index must be contiguous from zero")
            identities = {self._row_identity(row) for row in rows}
            if len(identities) != 1:
                raise ValueError(f"v015 source_index={source} mixes local scenario identity across attempts")
        if self._integrity_hash != _v015_formal_transaction_plan_hash(self):
            raise RuntimeError("v015 formal transaction plan was mutated after sealing")

    def verify_policy(self, policy: Any) -> None:
        self.validate()
        self.snapshot.verify_policy(policy)

    def validate_candidate_batch(self, batch: Any) -> tuple[tuple[int, int], ...]:
        """Validate one candidate-adapter shard as a subset of this sealed plan."""

        self.validate()
        metadata = getattr(batch, "transaction_metadata", None)
        validate = getattr(metadata, "validate", None)
        if not callable(validate):
            raise TypeError("v015 formal transaction requires grouped candidate metadata")
        validate()
        if (
            str(getattr(metadata, "transaction_id", "")) != self.transaction_id
            or str(getattr(metadata, "policy_snapshot_id", "")) != self.policy_snapshot_id
        ):
            raise ValueError("v015 formal transaction candidate has foreign transaction or snapshot identity")
        row_count = int(getattr(metadata, "batch_size", -1))
        actions = getattr(batch, "actions", None)
        if not isinstance(actions, torch.Tensor) or actions.ndim != 2 or tuple(actions.shape[1:]) != (6,):
            raise ValueError("v015 formal transaction candidate requires actions [B,6]")
        if row_count <= 0 or int(actions.shape[0]) != row_count:
            raise ValueError("v015 formal transaction candidate action rows disagree with metadata")
        row_indices = getattr(batch, "transaction_row_indices", None)
        expected_indices = torch.arange(row_count, dtype=torch.long)
        if (
            not isinstance(row_indices, torch.Tensor)
            or row_indices.ndim != 1
            or not torch.equal(row_indices.detach().to(device="cpu", dtype=torch.long), expected_indices)
        ):
            raise ValueError("v015 formal transaction requires the grouped candidate adapter row indices")
        if (
            str(getattr(metadata, "intent_q29_provenance", "")) != self.intent_q29_provenance
            or str(getattr(metadata, "intent_q29_source", "")) != self.intent_q29_source
        ):
            raise ValueError("v015 formal transaction candidate lost q29 provenance")
        expected_by_attempt = {
            (int(self.source_index[row].item()), int(self.trial_index[row].item())): self._row_identity(row)
            for row in range(self.batch_size)
        }
        for name in ("start_frames", "segment_ids", "source_index", "trial_index", "horizon_k"):
            value = getattr(metadata, name, None)
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or int(value.numel()) != row_count:
                raise ValueError(f"v015 formal transaction candidate {name} must be rank-1 [B]")
        for name in ("motion_ids", "scenario_ids", "noisy_segment_hashes", "x_t_identities", "trial_role"):
            value = getattr(metadata, name, None)
            if not isinstance(value, tuple) or len(value) != row_count:
                raise ValueError(f"v015 formal transaction candidate {name} must be row-aligned")
        candidate_attempts: list[tuple[int, int]] = []
        for row in range(row_count):
            source = int(metadata.source_index[row].item())
            trial = int(metadata.trial_index[row].item())
            key = (source, trial)
            expected = expected_by_attempt.get(key)
            observed = (
                str(metadata.motion_ids[row]),
                int(metadata.start_frames[row].item()),
                int(metadata.segment_ids[row].item()),
                int(metadata.horizon_k[row].item()),
                str(metadata.scenario_ids[row]),
                str(metadata.noisy_segment_hashes[row]),
                str(metadata.x_t_identities[row]),
            )
            if expected is None or observed != expected or str(metadata.trial_role[row]) != "policy":
                raise ValueError("v015 formal transaction candidate is partial, mixed, or not a planned policy attempt")
            candidate_attempts.append(key)
        if len(set(candidate_attempts)) != len(candidate_attempts):
            raise ValueError("v015 formal transaction candidate repeats a source/trial attempt")
        return tuple(candidate_attempts)


def _v015_formal_transaction_plan_hash(plan: FrontRESFormalTransactionPlan) -> str:
    digest = hashlib.sha256()
    for value in (
        plan.snapshot.transaction_id,
        plan.snapshot.policy_snapshot_id,
        plan.snapshot.policy_state_hash,
        str(int(plan.snapshot.policy_state_tensor_count)),
        str(int(plan.snapshot.policy_state_numel)),
        plan.intent_q29_provenance,
        plan.intent_q29_source,
        repr(plan.motion_ids),
        repr(plan.scenario_ids),
        repr(plan.noisy_segment_hashes),
        repr(plan.x_t_identities),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for value in (plan.start_frames, plan.segment_ids, plan.source_index, plan.trial_index, plan.horizon_k):
        digest.update(value.detach().to(device="cpu", dtype=torch.long).contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


class FrontRESFormalTransactionAccumulator:
    """Collect immutable candidate-adapter shards and seal exactly the planned rows.

    该 accumulator 只处理 metadata 和 batch 拼接. optimizer 是下游 formal-update
    owner 的唯一职责; collection 期间任一步都会立即失败.
    """

    def __init__(self, plan: FrontRESFormalTransactionPlan, *, optimizer_step_count: Any) -> None:
        if not callable(optimizer_step_count):
            raise TypeError("optimizer_step_count must be callable")
        plan.validate()
        self._plan = plan
        self._optimizer_step_count = optimizer_step_count
        self._step_at_open = self._read_step_count()
        self._batches: list[Any] = []
        self._attempts: set[tuple[int, int]] = set()
        self._state = "collecting"

    @property
    def state(self) -> str:
        return self._state

    @property
    def collected_attempt_count(self) -> int:
        return len(self._attempts)

    def _read_step_count(self) -> int:
        value = int(self._optimizer_step_count())
        if value < 0:
            raise ValueError("optimizer_step_count must be non-negative")
        return value

    def _require_no_step(self) -> None:
        current = self._read_step_count()
        if current != self._step_at_open:
            self._state = "failed"
            raise RuntimeError(
                "optimizer step occurred during v015 formal transaction collection: "
                f"opened={self._step_at_open} current={current}"
            )

    def append_candidate_batch(self, batch: Any) -> None:
        if self._state != "collecting":
            raise RuntimeError(f"v015 formal transaction is not collecting; state={self._state}")
        self._require_no_step()
        attempts = self._plan.validate_candidate_batch(batch)
        overlap = self._attempts.intersection(attempts)
        if overlap:
            raise ValueError(f"v015 formal transaction repeats planned attempts: {sorted(overlap)}")
        self._attempts.update(attempts)
        self._batches.append(batch)
        self._require_no_step()

    def seal(self) -> Any:
        if self._state != "collecting":
            raise RuntimeError(f"v015 formal transaction cannot seal; state={self._state}")
        self._require_no_step()
        expected = {
            (int(self._plan.source_index[row].item()), int(self._plan.trial_index[row].item()))
            for row in range(self._plan.batch_size)
        }
        if self._attempts != expected:
            missing = sorted(expected.difference(self._attempts))
            unexpected = sorted(self._attempts.difference(expected))
            self._state = "failed"
            raise RuntimeError(
                "v015 formal transaction is partial or mixed before update: "
                f"missing={missing} unexpected={unexpected}"
            )
        result = self._merge_candidate_batches()
        self._plan.validate_candidate_batch(result)
        self._require_no_step()
        self._state = "sealed"
        return result

    def _merge_candidate_batches(self) -> Any:
        if not self._batches:
            raise RuntimeError("v015 formal transaction has no candidate batches")
        first = self._batches[0]
        first_metadata = first.transaction_metadata

        def cat_batch_tensor(name: str) -> torch.Tensor:
            values = [getattr(batch, name, None) for batch in self._batches]
            if any(not isinstance(value, torch.Tensor) for value in values):
                raise ValueError(f"v015 formal transaction batch {name} must be a tensor")
            return torch.cat(tuple(values), dim=0)

        def cat_optional_batch_tensor(name: str) -> torch.Tensor | None:
            values = [getattr(batch, name, None) for batch in self._batches]
            if all(value is None for value in values):
                return None
            if any(not isinstance(value, torch.Tensor) for value in values):
                raise ValueError(f"v015 formal transaction optional batch {name} is mixed")
            return torch.cat(tuple(values), dim=0)

        def cat_metadata_tensor(name: str) -> torch.Tensor:
            return torch.cat(tuple(getattr(batch.transaction_metadata, name) for batch in self._batches), dim=0)

        def cat_metadata_tuple(name: str) -> tuple[str, ...]:
            return tuple(value for batch in self._batches for value in getattr(batch.transaction_metadata, name))

        source_index = cat_metadata_tensor("source_index")
        trial_index = cat_metadata_tensor("trial_index")
        order_values = sorted(
            range(int(source_index.numel())),
            key=lambda row: (int(source_index[row].item()), int(trial_index[row].item())),
        )
        order = torch.tensor(order_values, device=source_index.device, dtype=torch.long)
        inverse_order = {old_row: new_row for new_row, old_row in enumerate(order_values)}
        preference_edges: list[tuple[int, int]] = []
        row_offset = 0
        for batch in self._batches:
            batch_rows = int(batch.observations.shape[0])
            for winner, loser in tuple(getattr(batch, "preference_edges", ()) or ()):
                if not (0 <= int(winner) < batch_rows) or not (0 <= int(loser) < batch_rows):
                    raise ValueError("formal transaction preference edge is outside its candidate shard")
                preference_edges.append(
                    (inverse_order[row_offset + int(winner)], inverse_order[row_offset + int(loser)])
                )
            row_offset += batch_rows

        def reorder_tensor(value: torch.Tensor) -> torch.Tensor:
            return value.index_select(0, order.to(device=value.device))

        def reorder_tuple(value: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(value[row] for row in order_values)

        metadata_cls = type(first_metadata)
        metadata = metadata_cls(
            transaction_id=self._plan.transaction_id,
            policy_snapshot_id=self._plan.policy_snapshot_id,
            motion_ids=reorder_tuple(cat_metadata_tuple("motion_ids")),
            start_frames=reorder_tensor(cat_metadata_tensor("start_frames")),
            segment_ids=reorder_tensor(cat_metadata_tensor("segment_ids")),
            source_index=reorder_tensor(source_index),
            trial_index=reorder_tensor(trial_index),
            horizon_k=reorder_tensor(cat_metadata_tensor("horizon_k")),
            evidence_valid_step_count=reorder_tensor(cat_metadata_tensor("evidence_valid_step_count")),
            trial_role=reorder_tuple(cat_metadata_tuple("trial_role")),
            noisy_segment_hashes=reorder_tuple(cat_metadata_tuple("noisy_segment_hashes")),
            scenario_ids=reorder_tuple(cat_metadata_tuple("scenario_ids")),
            x_t_identities=reorder_tuple(cat_metadata_tuple("x_t_identities")),
            intent_q29_provenance=self._plan.intent_q29_provenance,
            intent_q29_source=self._plan.intent_q29_source,
            layout_version=str(getattr(first_metadata, "layout_version", "")),
        )
        batch_cls = type(first)
        total_rows = int(metadata.batch_size)
        valid_mask = reorder_tensor(cat_batch_tensor("valid_mask")).bool()

        common_payload = dict(
            observations=reorder_tensor(cat_batch_tensor("observations")),
            actions=reorder_tensor(cat_batch_tensor("actions")),
            old_log_probs=reorder_tensor(cat_batch_tensor("old_log_probs")),
            valid_mask=valid_mask,
            segment_ids=reorder_tensor(cat_batch_tensor("segment_ids")),
            old_means=(
                None
                if cat_optional_batch_tensor("old_means") is None
                else reorder_tensor(cat_optional_batch_tensor("old_means"))
            ),
            old_sigmas=(
                None
                if cat_optional_batch_tensor("old_sigmas") is None
                else reorder_tensor(cat_optional_batch_tensor("old_sigmas"))
            ),
            privileged_observations=(
                None
                if cat_optional_batch_tensor("privileged_observations") is None
                else reorder_tensor(cat_optional_batch_tensor("privileged_observations"))
            ),
            transaction_metadata=metadata,
            transaction_row_indices=torch.arange(total_rows, dtype=torch.long),
            preference_edges=tuple(preference_edges),
        )
        if hasattr(first, "returns"):
            common_payload.update(
                old_values=reorder_tensor(cat_batch_tensor("old_values")),
                returns=reorder_tensor(cat_batch_tensor("returns")),
                advantages=reorder_tensor(cat_batch_tensor("advantages")),
            )
        return batch_cls(**common_payload)


def _immutable_frozen_transaction_vector(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got {tuple(value.shape)}")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached metadata")
    return value.detach().to(device="cpu", dtype=torch.long).clone().contiguous()


@dataclass(frozen=True)
class FrontRESFrozenPolicyTransactionMetadata:
    """One sealed identity carrier for a planned all-policy transaction."""

    transaction_id: str
    policy_snapshot_id: str
    policy_state_hash: str
    policy_state_tensor_count: int
    policy_state_numel: int
    motion_ids: tuple[str, ...]
    start_frames: torch.Tensor
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor
    trial_role: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    _integrity_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transaction_id", str(self.transaction_id))
        object.__setattr__(self, "policy_snapshot_id", str(self.policy_snapshot_id))
        object.__setattr__(self, "policy_state_hash", str(self.policy_state_hash))
        object.__setattr__(self, "motion_ids", tuple(str(value) for value in self.motion_ids))
        object.__setattr__(self, "trial_role", tuple(str(value) for value in self.trial_role))
        object.__setattr__(self, "noisy_segment_hashes", tuple(str(value) for value in self.noisy_segment_hashes))
        object.__setattr__(self, "scenario_ids", tuple(str(value) for value in self.scenario_ids))
        for name in ("start_frames", "segment_ids", "source_index", "trial_index", "horizon_k"):
            object.__setattr__(self, name, _immutable_frozen_transaction_vector(name, getattr(self, name)))
        object.__setattr__(self, "_integrity_hash", _frozen_transaction_metadata_hash(self))
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.segment_ids.numel())

    def verify_policy(self, policy: Any) -> None:
        FrontRESFrozenPolicySnapshot(
            transaction_id=self.transaction_id,
            policy_snapshot_id=self.policy_snapshot_id,
            policy_state_hash=self.policy_state_hash,
            policy_state_tensor_count=int(self.policy_state_tensor_count),
            policy_state_numel=int(self.policy_state_numel),
        ).verify_policy(policy)

    def validate(self) -> None:
        FrontRESFrozenPolicySnapshot(
            transaction_id=self.transaction_id,
            policy_snapshot_id=self.policy_snapshot_id,
            policy_state_hash=self.policy_state_hash,
            policy_state_tensor_count=int(self.policy_state_tensor_count),
            policy_state_numel=int(self.policy_state_numel),
        ).validate()
        count = self.batch_size
        if count <= 0:
            raise ValueError("frozen transaction metadata must have at least one row")
        for name, value in (
            ("motion_ids", self.motion_ids),
            ("start_frames", self.start_frames),
            ("source_index", self.source_index),
            ("trial_index", self.trial_index),
            ("horizon_k", self.horizon_k),
            ("trial_role", self.trial_role),
            ("noisy_segment_hashes", self.noisy_segment_hashes),
            ("scenario_ids", self.scenario_ids),
        ):
            row_count = len(value) if isinstance(value, tuple) else int(value.numel())
            if row_count != count:
                raise ValueError(f"{name} must have {count} transaction rows")
        if any(not motion_id for motion_id in self.motion_ids):
            raise ValueError("transaction metadata motion_ids must be non-empty")
        if any(role != "policy" for role in self.trial_role):
            raise ValueError("frozen transaction metadata may contain only policy attempts")
        if bool((self.source_index < 0).any()) or bool((self.trial_index < 0).any()) or bool((self.horizon_k <= 0).any()):
            raise ValueError("transaction metadata has invalid source/trial/K rows")
        if any(not value for value in self.noisy_segment_hashes) or any(not value for value in self.scenario_ids):
            raise ValueError("transaction metadata requires one sealed Noisy identity per row")
        by_source: dict[int, tuple[str, int, int, int, str, str]] = {}
        for row in range(count):
            source = int(self.source_index[row].item())
            identity = (
                self.motion_ids[row],
                int(self.start_frames[row].item()),
                int(self.segment_ids[row].item()),
                int(self.horizon_k[row].item()),
                self.noisy_segment_hashes[row],
                self.scenario_ids[row],
            )
            previous = by_source.setdefault(source, identity)
            if previous != identity:
                raise ValueError(f"source_index={source} maps to mixed transaction/reference identities")
        if self._integrity_hash != _frozen_transaction_metadata_hash(self):
            raise RuntimeError("frozen transaction metadata was mutated after sealing")


def _frozen_transaction_metadata_hash(metadata: FrontRESFrozenPolicyTransactionMetadata) -> str:
    digest = hashlib.sha256()
    for value in (
        metadata.transaction_id,
        metadata.policy_snapshot_id,
        metadata.policy_state_hash,
        str(int(metadata.policy_state_tensor_count)),
        str(int(metadata.policy_state_numel)),
        repr(metadata.motion_ids),
        repr(metadata.trial_role),
        repr(metadata.noisy_segment_hashes),
        repr(metadata.scenario_ids),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for value in (metadata.start_frames, metadata.segment_ids, metadata.source_index, metadata.trial_index, metadata.horizon_k):
        digest.update(value.detach().to(device="cpu", dtype=torch.long).contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_frontres_frozen_policy_transaction_plan(plan: Any) -> None:
    validate = getattr(plan, "validate", None)
    if not callable(validate):
        raise TypeError("frozen transaction binding requires an S1a plan with validate()")
    validate()
    for name in (
        "transaction_id",
        "policy_snapshot_id",
        "segment_ids",
        "source_index",
        "trial_index",
        "horizon_k",
        "trial_role",
    ):
        if not hasattr(plan, name):
            raise TypeError(f"frozen transaction plan is missing {name}")


def _equal_frozen_transaction_rows(left: Any, right: Any) -> bool:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    return torch.equal(
        left.detach().to(device="cpu", dtype=torch.long).reshape(-1),
        right.detach().to(device="cpu", dtype=torch.long).reshape(-1),
    )


def _bound_transaction_trial_rows(batch: Any) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    roles = getattr(batch, "frontres_segment_trial_role", None)
    source_index = getattr(batch, "frontres_segment_source_index", None)
    trial_index = getattr(batch, "frontres_segment_trial_index", None)
    horizon_k = getattr(batch, "frontres_segment_budget_horizon_k", None)
    if not isinstance(roles, tuple) or not all(isinstance(role, str) for role in roles):
        raise ValueError("frozen transaction batch requires tuple frontres_segment_trial_role")
    if not all(isinstance(value, torch.Tensor) for value in (source_index, trial_index, horizon_k)):
        raise ValueError("frozen transaction batch requires source/trial/K tensors")
    return roles, source_index, trial_index, horizon_k


def bind_frontres_frozen_policy_transaction(
    runner: Any,
    batch: Any,
    *,
    plan: Any,
    snapshot: FrontRESFrozenPolicySnapshot,
) -> Any:
    """Pre-bind one S1a layout to a real unchanged policy before tape materialization."""

    validate_frontres_frozen_policy_transaction_plan(plan)
    if not isinstance(snapshot, FrontRESFrozenPolicySnapshot):
        raise TypeError("frozen transaction binding requires a FrontRESFrozenPolicySnapshot")
    snapshot.validate()
    if str(plan.transaction_id) != snapshot.transaction_id:
        raise ValueError("plan transaction_id must match the captured frozen policy snapshot")
    if str(plan.policy_snapshot_id) != snapshot.policy_snapshot_id:
        raise ValueError("plan policy_snapshot_id must match the captured frozen policy snapshot")
    batch_segment_ids = getattr(batch, "segment_ids", None)
    if not _equal_frozen_transaction_rows(batch_segment_ids, plan.segment_ids):
        raise ValueError("selected batch segment_ids do not match the frozen transaction plan")
    roles, source_index, trial_index, horizon_k = _bound_transaction_trial_rows(batch)
    if tuple(roles) != tuple(str(value) for value in plan.trial_role):
        raise ValueError("selected batch trial roles do not match the frozen transaction plan")
    for name, actual, expected in (
        ("source_index", source_index, plan.source_index),
        ("trial_index", trial_index, plan.trial_index),
        ("horizon_k", horizon_k, plan.horizon_k),
    ):
        if not _equal_frozen_transaction_rows(actual, expected):
            raise ValueError(f"selected batch {name} does not match the frozen transaction plan")
    snapshot.verify_policy(getattr(getattr(runner, "alg", None), "policy", None))
    existing = getattr(batch, "frontres_segment_transaction_metadata", None)
    if existing is not None:
        raise RuntimeError("frozen transaction metadata is already sealed on this batch")
    for name, value in (
        ("frontres_segment_transaction_id", snapshot.transaction_id),
        ("frontres_segment_policy_snapshot_id", snapshot.policy_snapshot_id),
        ("frontres_segment_policy_state_hash", snapshot.policy_state_hash),
        ("frontres_segment_frozen_policy_snapshot", snapshot),
        ("frontres_segment_transaction_plan", plan),
    ):
        previous = getattr(batch, name, None)
        conflict = previous is not value if name == "frontres_segment_transaction_plan" else previous != value
        if previous is not None and conflict:
            raise RuntimeError(f"frozen transaction binding conflicts with existing {name}")
        object.__setattr__(batch, name, value)
    return batch


def finalize_frontres_frozen_policy_transaction_metadata(
    runner: Any,
    batch: Any,
) -> FrontRESFrozenPolicyTransactionMetadata:
    """Seal batch/reset/storage provenance after the one fixed Noisy tape is attached."""

    existing = getattr(batch, "frontres_segment_transaction_metadata", None)
    if isinstance(existing, FrontRESFrozenPolicyTransactionMetadata):
        existing.validate()
        return existing
    if existing is not None:
        raise TypeError("frozen transaction metadata carrier must not be replaced after binding")
    plan = getattr(batch, "frontres_segment_transaction_plan", None)
    snapshot = getattr(batch, "frontres_segment_frozen_policy_snapshot", None)
    validate_frontres_frozen_policy_transaction_plan(plan)
    if not isinstance(snapshot, FrontRESFrozenPolicySnapshot):
        raise TypeError("frozen transaction metadata requires the captured policy snapshot")
    snapshot.verify_policy(getattr(getattr(runner, "alg", None), "policy", None))
    if getattr(batch, "frontres_segment_transaction_id", None) != snapshot.transaction_id:
        raise ValueError("batch transaction_id does not match the frozen policy snapshot")
    if getattr(batch, "frontres_segment_policy_snapshot_id", None) != snapshot.policy_snapshot_id:
        raise ValueError("batch policy_snapshot_id does not match the frozen policy snapshot")
    if getattr(batch, "frontres_fixed_noisy_transaction_id", None) != snapshot.transaction_id:
        raise ValueError("fixed Noisy tape must use the frozen transaction_id")
    if str(plan.transaction_id) != snapshot.transaction_id or str(plan.policy_snapshot_id) != snapshot.policy_snapshot_id:
        raise ValueError("plan identity does not match the frozen policy snapshot")
    batch_size = int(getattr(batch, "segment_ids").numel())
    roles, source_index, trial_index, horizon_k = _bound_transaction_trial_rows(batch)
    if not _equal_frozen_transaction_rows(getattr(batch, "segment_ids"), plan.segment_ids):
        raise ValueError("sealed batch segment_ids do not match the frozen transaction plan")
    if tuple(roles) != tuple(str(value) for value in plan.trial_role):
        raise ValueError("sealed batch trial roles do not match the frozen transaction plan")
    for name, actual, expected in (
        ("source_index", source_index, plan.source_index),
        ("trial_index", trial_index, plan.trial_index),
        ("horizon_k", horizon_k, plan.horizon_k),
    ):
        if not _equal_frozen_transaction_rows(actual, expected):
            raise ValueError(f"sealed batch {name} does not match the frozen transaction plan")
    specs = tuple(getattr(batch, "specs", ()) or ())
    if len(specs) != batch_size:
        raise ValueError("frozen transaction metadata requires one index spec per batch row")
    motion_ids = tuple(str(getattr(spec, "motion_id", "")) for spec in specs)
    start_frames = torch.tensor(
        [int(getattr(spec, "start_frame", -1)) for spec in specs],
        dtype=torch.long,
        device=getattr(batch, "segment_ids").device,
    )
    noisy_segment_hashes = tuple(str(value) for value in (getattr(batch, "frontres_fixed_noisy_segment_hashes", ()) or ()))
    scenario_ids = tuple(str(value) for value in (getattr(batch, "frontres_fixed_noisy_scenario_ids", ()) or ()))
    if len(noisy_segment_hashes) != batch_size or len(scenario_ids) != batch_size:
        raise ValueError("frozen transaction metadata requires sealed Noisy scenario/hash rows")
    scenario_rows = getattr(batch, "frontres_fixed_noisy_scenario_rows", None)
    if scenario_rows is None:
        raise ValueError("frozen transaction metadata requires fixed Noisy lifecycle rows")
    if tuple(scenario_rows.noisy_segment_hashes) != noisy_segment_hashes or tuple(scenario_rows.scenario_ids) != scenario_ids:
        raise ValueError("fixed Noisy lifecycle rows disagree with the batch scenario/hash identity")
    metadata = FrontRESFrozenPolicyTransactionMetadata(
        transaction_id=snapshot.transaction_id,
        policy_snapshot_id=snapshot.policy_snapshot_id,
        policy_state_hash=snapshot.policy_state_hash,
        policy_state_tensor_count=snapshot.policy_state_tensor_count,
        policy_state_numel=snapshot.policy_state_numel,
        motion_ids=motion_ids,
        start_frames=start_frames,
        segment_ids=getattr(batch, "segment_ids"),
        source_index=source_index,
        trial_index=trial_index,
        horizon_k=horizon_k,
        trial_role=roles,
        noisy_segment_hashes=noisy_segment_hashes,
        scenario_ids=scenario_ids,
    )
    metadata.validate()
    for name, value in (
        ("frontres_segment_transaction_metadata", metadata),
        ("frontres_segment_motion_ids", metadata.motion_ids),
        ("frontres_segment_start_frames", metadata.start_frames),
        ("frontres_segment_noisy_segment_hashes", metadata.noisy_segment_hashes),
    ):
        object.__setattr__(batch, name, value)
    return metadata
