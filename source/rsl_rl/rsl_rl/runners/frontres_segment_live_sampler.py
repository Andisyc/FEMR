from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import torch

_FORMAL_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_audit_sampler", Path(__file__).resolve().with_name("frontres_formal_runtime_audit.py")
)
_FORMAL_AUDIT_MODULE = importlib.util.module_from_spec(_FORMAL_AUDIT_SPEC)
assert _FORMAL_AUDIT_SPEC.loader is not None
_FORMAL_AUDIT_SPEC.loader.exec_module(_FORMAL_AUDIT_MODULE)
print_sampler_audit = _FORMAL_AUDIT_MODULE.print_sampler_audit

_SAMPLER_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_segment_sampler.py"
_SAMPLER_SPEC = importlib.util.spec_from_file_location(
    "frontres_segment_sampler_live_module",
    _SAMPLER_PATH,
)
if _SAMPLER_SPEC is None or _SAMPLER_SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES Segment sampler from {_SAMPLER_PATH}.")
_SAMPLER_MODULE = importlib.util.module_from_spec(_SAMPLER_SPEC)
sys.modules[_SAMPLER_SPEC.name] = _SAMPLER_MODULE
_SAMPLER_SPEC.loader.exec_module(_SAMPLER_MODULE)

_DATASET_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_segment_dataset.py"
_DATASET_SPEC = importlib.util.spec_from_file_location(
    "frontres_segment_dataset_live_module",
    _DATASET_PATH,
)
if _DATASET_SPEC is None or _DATASET_SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES Segment dataset from {_DATASET_PATH}.")
_DATASET_MODULE = importlib.util.module_from_spec(_DATASET_SPEC)
sys.modules[_DATASET_SPEC.name] = _DATASET_MODULE
_DATASET_SPEC.loader.exec_module(_DATASET_MODULE)

_CURRICULUM_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_dr_curriculum.py"
_CURRICULUM_SPEC = importlib.util.spec_from_file_location(
    "frontres_dr_curriculum_live_sampler_module",
    _CURRICULUM_PATH,
)
if _CURRICULUM_SPEC is None or _CURRICULUM_SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES DR curriculum from {_CURRICULUM_PATH}.")
_CURRICULUM_MODULE = importlib.util.module_from_spec(_CURRICULUM_SPEC)
sys.modules[_CURRICULUM_SPEC.name] = _CURRICULUM_MODULE
_CURRICULUM_SPEC.loader.exec_module(_CURRICULUM_MODULE)

FrontRESSegmentRolloutEvidence = _SAMPLER_MODULE.FrontRESSegmentRolloutEvidence
FrontRESSegmentSample = _SAMPLER_MODULE.FrontRESSegmentSample
FrontRESSegmentSampler = _SAMPLER_MODULE.FrontRESSegmentSampler
FrontRESFrozenPolicyTransactionPlan = _SAMPLER_MODULE.FrontRESFrozenPolicyTransactionPlan
FrontRESFixedNoisyScenarioLifecycle = _SAMPLER_MODULE.FrontRESFixedNoisyScenarioLifecycle
FrontRESNoisyReferenceMaterialization = _SAMPLER_MODULE.FrontRESNoisyReferenceMaterialization
FrontRESLocalScenarioLifecycle = _SAMPLER_MODULE.FrontRESLocalScenarioLifecycle
FrontRESLocalScenarioMaterialization = _SAMPLER_MODULE.FrontRESLocalScenarioMaterialization
load_stage1_cache_dataset = _DATASET_MODULE.load_stage1_cache_dataset
sample_per_env_dr_strength = _CURRICULUM_MODULE.sample_per_env_dr_strength
sample_perturbation_mix = _CURRICULUM_MODULE.sample_perturbation_mix

_VERBOSE_PROBE_BATCH_LIMIT = 16
_LOG_SEPARATOR = "-" * 80


def _log_block(*lines: str) -> str:
    return "\n".join(("", _LOG_SEPARATOR, "", *lines))


def _kv_lines(prefix: str, values: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"  {prefix}.{key}: {value}" for key, value in values.items())


def _fmt_num(value: Any) -> str:
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    abs_value = abs(value)
    if abs_value != 0.0 and (abs_value >= 10000.0 or abs_value < 0.001):
        return f"{value:.3e}"
    return f"{value:.6f}"


def _fmt_pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


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
class FrontRESV015FormalTransactionPlan:
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
        if len(source_rows) < 2:
            raise ValueError("v015 formal transaction plan requires at least two selected Segment sources")
        for source, rows in source_rows.items():
            if len(rows) < 2:
                raise ValueError(f"v015 source_index={source} requires at least two policy attempts")
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


def _v015_formal_transaction_plan_hash(plan: FrontRESV015FormalTransactionPlan) -> str:
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


class FrontRESV015FormalTransactionAccumulator:
    """Collect immutable candidate-adapter shards and seal exactly the planned rows.

    该 accumulator 只处理 metadata 和 batch 拼接. optimizer 是下游 formal-update
    owner 的唯一职责; collection 期间任一步都会立即失败.
    """

    def __init__(self, plan: FrontRESV015FormalTransactionPlan, *, optimizer_step_count: Any) -> None:
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
        return batch_cls(
            observations=reorder_tensor(cat_batch_tensor("observations")),
            actions=reorder_tensor(cat_batch_tensor("actions")),
            old_log_probs=reorder_tensor(cat_batch_tensor("old_log_probs")),
            old_values=reorder_tensor(cat_batch_tensor("old_values")),
            returns=reorder_tensor(cat_batch_tensor("returns")),
            advantages=reorder_tensor(cat_batch_tensor("advantages")),
            valid_mask=reorder_tensor(cat_batch_tensor("valid_mask")),
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
        )


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


def _validate_frozen_policy_transaction_plan(plan: Any) -> None:
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

    _validate_frozen_policy_transaction_plan(plan)
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
    _validate_frozen_policy_transaction_plan(plan)
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


def initialize_frontres_segment_live_sampler(runner: Any) -> None:
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    if not bool(getattr(boundary, "requested", False) and getattr(boundary, "live_runner_enabled", False)):
        return
    if getattr(runner, "_frontres_segment_sampler", None) is not None:
        return
    _ensure_stage1_cache_dataset(runner)
    _ensure_stage1_index_reset_hook(runner)
    num_segments = _resolve_num_segments(runner)
    runner._frontres_segment_sampler = FrontRESSegmentSampler(
        num_segments=num_segments,
        global_frac=float(getattr(runner.alg, "frontres_segment_sampler_global_frac", 0.4)),
        replay_frac=float(getattr(runner.alg, "frontres_segment_sampler_replay_frac", 0.5)),
        review_frac=float(getattr(runner.alg, "frontres_segment_sampler_review_frac", 0.1)),
        seed=int(getattr(runner, "seed", 0) or 0),
        device=getattr(runner, "device", "cpu"),
    )
    print(
        _log_block(
            "[FrontRES Segment Sampler Ready]",
            "  config: "
            f"num_segments={num_segments} "
            f"global_frac={runner._frontres_segment_sampler.global_frac:.3f} "
            f"replay_frac={runner._frontres_segment_sampler.replay_frac:.3f} "
            f"review_frac={runner._frontres_segment_sampler.review_frac:.3f}",
        ),
        flush=True,
    )


def ensure_frontres_policy_quality_reset_support(runner: Any) -> None:
    """Install only cache-backed index reset support; never create or sample a replay sampler."""
    sampler_before = getattr(runner, "_frontres_segment_sampler", None)
    _ensure_stage1_cache_dataset(runner)
    _ensure_stage1_index_reset_hook(runner)
    if getattr(runner, "_frontres_segment_sampler", None) is not sampler_before:
        raise RuntimeError("policy-quality reset support must not create or replace the Segment sampler")


def _ensure_stage1_index_reset_hook(runner: Any) -> None:
    dataset = getattr(runner, "_frontres_segment_dataset", None)
    metadata = dataset.cache_metadata() if dataset is not None and hasattr(dataset, "cache_metadata") else None
    if not isinstance(metadata, dict) or not bool(metadata.get("index_only", False)):
        return
    amass_root = str(metadata.get("amass_root", "") or "")
    if not amass_root:
        raise ValueError("index-only Stage 1 dataset metadata is missing amass_root")
    from rsl_rl.frontres.frontres_segment_stage1_env_hooks import ensure_frontres_segment_index_reset_hook

    adapter = ensure_frontres_segment_index_reset_hook(
        runner.env,
        amass_root=amass_root,
        robot_name=str(getattr(runner.alg, "frontres_segment_reset_robot_name", "robot")),
        trace=bool(getattr(runner.alg, "frontres_segment_reset_trace", True)),
    )
    probe = adapter.frontres_motion_loader_probe()
    filter_probe = None
    if hasattr(dataset, "filter_to_loaded_motion_paths"):
        filter_probe = dataset.filter_to_loaded_motion_paths(
            adapter.frontres_loaded_motion_paths(),
            amass_root=amass_root,
        )
    print(
        _log_block(
            "[FrontRES Segment Index Reset Hook Ready]",
            "  loader: "
            f"amass_root={amass_root} "
            f"loaded_motion_count={probe.get('loaded_motion_count')} "
            f"all_motion_count={probe.get('all_motion_count')} "
            f"first_loaded_motion={probe.get('first_loaded_motion')}",
            "  index_filter: "
            f"{filter_probe if filter_probe is not None else 'not_applied'}",
        ),
        flush=True,
    )


def _ensure_stage1_cache_dataset(runner: Any) -> None:
    if getattr(runner, "_frontres_segment_dataset", None) is not None:
        return
    alg = getattr(runner, "alg", None)
    cache_dir = str(getattr(alg, "frontres_segment_cache_dir", "") or "")
    if not cache_dir:
        print(
            _log_block(
                "[FrontRES Segment Dataset]",
                "  cache_load: skipped reason=no_cache_dir",
            ),
            flush=True,
        )
        return
    include_boundary = bool(getattr(alg, "frontres_segment_include_boundary_diagnostic", False))
    shard_cache_size = max(1, int(getattr(alg, "frontres_segment_shard_cache_size", 8)))
    dataset = load_stage1_cache_dataset(
        cache_dir,
        device=getattr(runner, "device", "cpu"),
        include_boundary_diagnostic=include_boundary,
        shard_cache_size=shard_cache_size,
    )
    runner._frontres_segment_dataset = dataset
    metadata = dataset.cache_metadata() if hasattr(dataset, "cache_metadata") else None
    print(
            _log_block(
                "[FrontRES Segment Dataset Ready]",
                *_kv_lines(
                    "cache",
                    {
                        "cache_dir": cache_dir,
                        "num_segments": dataset.num_segments(),
                        "include_boundary_diagnostic": include_boundary,
                        "shard_cache_size": shard_cache_size,
                    },
                ),
                f"  metadata: {metadata}",
            ),
        flush=True,
    )


def run_frontres_segment_sampler_step(
    runner: Any,
    *,
    init_at_random_ep_len: bool,
    update_step: int,
) -> dict[str, object]:
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    if sampler is None:
        return runner.run_frontres_segment_live_probe(init_at_random_ep_len=init_at_random_ep_len)

    sample = _sample_live_segment_rows(runner, sampler)
    detail_log = _live_detail_log_enabled(runner)
    verbose_probe = _verbose_probe_enabled(runner, sample)
    if detail_log:
        _print_sample_probe(update_step, sample, verbose=verbose_probe)
    batch = _build_current_segment_batch(runner, sample, update_step=update_step, print_probe=detail_log)
    runner._frontres_segment_live_current_sample = sample
    runner._frontres_segment_live_current_batch = batch
    runner._frontres_segment_live_detail_log_enabled = detail_log
    adapter = getattr(getattr(runner, "env", None), "_frontres_segment_index_reset_adapter", None)
    old_adapter_trace = getattr(adapter, "trace", None)
    if adapter is not None and old_adapter_trace is not None:
        adapter.trace = bool(detail_log)
    reset_result = None
    try:
        try:
            summary = runner.run_frontres_segment_live_probe(init_at_random_ep_len=init_at_random_ep_len)
            reset_result = getattr(runner, "_frontres_segment_live_current_reset_result", None)
        finally:
            if adapter is not None and old_adapter_trace is not None:
                adapter.trace = old_adapter_trace
            runner._frontres_segment_live_current_sample = None
            runner._frontres_segment_live_current_batch = None
            runner._frontres_segment_live_current_reset_request = None
            runner._frontres_segment_live_current_reset_result = None
            runner._frontres_segment_live_detail_log_enabled = True

        evidence = build_live_sampler_evidence(
            sample,
            summary,
            horizon_k=sample.horizon_k if isinstance(sample.horizon_k, torch.Tensor) else int(getattr(runner.alg, "frontres_segment_k", 1)),
            reset_result=reset_result,
            print_probe=detail_log,
        )
    finally:
        _close_fixed_noisy_scenarios(batch)
    update_probe = sampler.update_with_probe(evidence)
    sampler_summary = summarize_sampler_step(sampler, sample)
    sampler_summary.update(
        {
            "sampler_update_valid_count": update_probe.valid_count,
            "sampler_update_fall_count": update_probe.fall_count,
            "sampler_update_gain_mean": update_probe.gain_mean,
            "sampler_update_gain_pos_frac": update_probe.gain_pos_frac,
            "sampler_update_useful_mean": update_probe.useful_mean,
            "sampler_update_useful_max": update_probe.useful_max,
            "sampler_update_priority_before_mean": update_probe.priority_before_mean,
            "sampler_update_priority_after_mean": update_probe.priority_after_mean,
            "sampler_update_priority_after_max": update_probe.priority_after_max,
            "sampler_update_replay_candidate_count": update_probe.replay_candidate_count,
            "sampler_update_hopeless_count": update_probe.hopeless_count,
            "sampler_update_delayed_regret_count": update_probe.delayed_regret_count,
            "sampler_update_segment_count": update_probe.segment_count,
            "sampler_update_trial_count": update_probe.trial_count,
            "sampler_update_oracle_gap_mean": update_probe.oracle_gap_mean,
            "sampler_update_confidence_mean": update_probe.confidence_mean,
        }
    )
    summary.update(sampler_summary)
    # AUDIT-SAMPLER-01: 检查 Segment Replay 与 per-row K, 位于 rollout evidence -> sampler summary.
    # Result: PENDING_LIVE.
    print_sampler_audit(runner, update_step=update_step, sample=sample, batch=batch, summary=summary)
    if detail_log:
        _print_sampler_summary(update_step, sampler_summary)
    return summary


def _cfg_get(owner: Any, key: str, default: Any) -> Any:
    if owner is None:
        return default
    if isinstance(owner, dict):
        return owner.get(key, default)
    return getattr(owner, key, default)


def _runner_cfg_get(runner: Any, key: str, default: Any) -> Any:
    for owner in (
        getattr(runner, "alg", None),
        getattr(runner, "cfg", None),
        getattr(runner, "alg_cfg", None),
    ):
        value = _cfg_get(owner, key, None)
        if value is not None:
            return value
    return default


def _stage3_index_frontier_scale(runner: Any) -> float:
    value = getattr(runner, "_dr_scale", None)
    if value is not None:
        return float(value)
    return float(_runner_cfg_get(runner, "frontres_dr_scale", _runner_cfg_get(runner, "dr_scale_init", 1.0)))


def _stage3_index_progress(runner: Any, update_step: int) -> float:
    if getattr(runner, "_frontres_segment_sequence_eval_seed", None) is not None:
        # Evaluation compares policies under one fixed, fully materialized
        # perturbation curriculum rather than their training-time progress.
        return 1.0
    current_iter = int(getattr(runner, "current_learning_iteration", 0) or 0)
    max_iter = max(1, int(_runner_cfg_get(runner, "max_iterations", 1)))
    return max(0.0, min(1.0, (current_iter + int(update_step)) / float(max_iter)))


def _index_only_segment_batch(batch: Any) -> bool:
    families = tuple(getattr(batch, "perturbation_family", ()) or ())
    if families:
        return all(str(family) == "index_only" for family in families)
    specs = tuple(getattr(batch, "specs", ()) or ())
    return bool(specs) and all(str(getattr(spec, "perturbation_family", "")) == "index_only" for spec in specs)


def _build_stage3_index_perturbation_plan(runner: Any, batch: Any, *, update_step: int) -> Any | None:
    if not _index_only_segment_batch(batch):
        return None
    n = int(getattr(batch, "batch_size", int(batch.segment_ids.numel())))
    source_index = _source_index_for_batch(batch, n=n, device=batch.segment_ids.device)
    source_ids, source_inverse = torch.unique(source_index, sorted=True, return_inverse=True)
    source_count = int(source_ids.numel())
    cfg = getattr(runner, "cfg", None) or getattr(runner, "alg_cfg", None) or {}
    eval_seed = getattr(runner, "_frontres_segment_sequence_eval_seed", None)
    if eval_seed is None:
        seq_idx = int(getattr(runner, "current_learning_iteration", 0) or 0) * 100000 + int(update_step)
    else:
        # Sequence eval must not change its corruption when comparing
        # checkpoints saved at different training iterations.
        seq_idx = int(eval_seed) * 100000 + int(update_step)
    progress = _stage3_index_progress(runner, update_step)
    mix_plan = sample_perturbation_mix(cfg, None, progress, seq_idx, source_count, is_frontres=True)
    frontier_scale = _stage3_index_frontier_scale(runner)
    dr_min = float(_runner_cfg_get(runner, "dr_min_scale", 0.0))
    dr_max = float(_runner_cfg_get(runner, "dr_max_scale", max(4.0, frontier_scale)))
    strength_plan = sample_per_env_dr_strength(
        cfg,
        frontier_scale,
        True,
        seq_idx,
        n_train=source_count,
        n_candidate=0,
        n_base=0,
        num_envs=source_count,
        dr_min=dr_min,
        dr_max=dr_max,
    )
    if strength_plan.scale_vector is None:
        source_strengths = [float(strength_plan.effective_scale)] * source_count
    else:
        source_strengths = [float(v) for v in strength_plan.scale_vector[:source_count]]
    source_family = tuple("+".join(group) for group in mix_plan.groups[:source_count])
    source_strength = torch.tensor(source_strengths, dtype=batch.perturbation_strength.dtype, device=batch.segment_ids.device)
    perturbation_strength = source_strength.index_select(0, source_inverse)
    perturbation_family = tuple(source_family[int(row)] for row in source_inverse.detach().cpu().tolist())
    return SimpleNamespace(
        perturbation_family=perturbation_family,
        perturbation_strength=perturbation_strength,
        source_index=source_index.detach().clone(),
        source_ids=source_ids.detach().clone(),
        source_perturbation_family=source_family,
        source_perturbation_strength=source_strength.detach().clone(),
        active_modes=tuple(mix_plan.active_modes),
        complexity=str(mix_plan.complexity),
        mix_mode=str(strength_plan.mix_mode),
        mix_diag=dict(strength_plan.diag),
        progress=float(progress),
        seq_idx=int(seq_idx),
    )


def _attach_stage3_index_perturbation_plan(batch: Any, plan: Any | None) -> Any:
    if plan is None:
        return batch
    object.__setattr__(batch, "perturbation_strength", plan.perturbation_strength)
    object.__setattr__(batch, "stage3_index_perturbation_family", plan.perturbation_family)
    object.__setattr__(batch, "stage3_index_perturbation_strength", plan.perturbation_strength)
    object.__setattr__(batch, "stage3_index_perturbation_plan", plan)
    return batch


def _source_index_for_batch(batch: Any, *, n: int, device: torch.device | str) -> torch.Tensor:
    value = getattr(batch, "frontres_segment_source_index", None)
    if value is None:
        return torch.arange(n, dtype=torch.long, device=device)
    source_index = torch.as_tensor(value, dtype=torch.long, device=device).flatten()
    if int(source_index.numel()) != int(n) or bool((source_index < 0).any()):
        raise ValueError("frontres_segment_source_index must be a nonnegative [B] tensor")
    return source_index.detach().clone()


def _attach_frontres_segment_trial_plan(batch: Any, sample: FrontRESSegmentSample) -> Any:
    # QUALITY-DATA-01: 检查 sampled segment -> policy/search role -> gradient-bearing rows.
    # Result: PENDING_Q_EVIDENCE.
    # B1: sample source/global-replay-review identity 在 trial expansion 前可见.
    # B2: policy/search role 与 K/difficulty 在这里首次绑定到 batch rows.
    # B3: storage/PPO 消费前可统计 unique/repeat/staleness 与 valid policy rows.
    roles = tuple(getattr(sample, "trial_role", ()) or ())
    if roles and len(roles) == int(sample.segment_ids.numel()):
        object.__setattr__(batch, "frontres_segment_trial_role", roles)
    source_index = getattr(sample, "source_index", None)
    if isinstance(source_index, torch.Tensor) and int(source_index.numel()) == int(sample.segment_ids.numel()):
        object.__setattr__(batch, "frontres_segment_source_index", source_index.detach().clone())
    trial_index = getattr(sample, "trial_index", None)
    if isinstance(trial_index, torch.Tensor) and int(trial_index.numel()) == int(sample.segment_ids.numel()):
        object.__setattr__(batch, "frontres_segment_trial_index", trial_index.detach().clone())
    horizon_k = getattr(sample, "horizon_k", None)
    if isinstance(horizon_k, torch.Tensor) and int(horizon_k.numel()) == int(sample.segment_ids.numel()):
        object.__setattr__(batch, "frontres_segment_budget_horizon_k", horizon_k.detach().clone())
    return batch


def _require_frontres_future_offsets(runner: Any) -> tuple[int, ...]:
    raw = _runner_cfg_get(runner, "frontres_future_offsets", None)
    if raw is None or isinstance(raw, (str, bytes)):
        raise ValueError(
            "fixed Noisy Segment Replay requires explicit nonempty frontres_future_offsets; no legacy default is allowed"
        )
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    try:
        offsets = tuple(int(value) for value in raw)
    except TypeError as exc:
        raise ValueError("frontres_future_offsets must be an ordered sequence of positive integers") from exc
    if not offsets or any(offset <= 0 for offset in offsets) or tuple(sorted(set(offsets))) != offsets:
        raise ValueError(
            f"frontres_future_offsets must be nonempty, positive, ordered, and unique; got {offsets}"
        )
    return offsets


def _fixed_noisy_materializer_adapter(runner: Any) -> Any:
    env = getattr(runner, "env", None)
    candidates = (env, getattr(env, "unwrapped", None))
    for owner in candidates:
        adapter = getattr(owner, "_frontres_segment_index_reset_adapter", None)
        if callable(getattr(adapter, "materialize_frontres_fixed_noisy_tape", None)):
            return adapter
    raise RuntimeError(
        "fixed Noisy Segment Replay requires the Stage 1 index-reset adapter with a command-owned tape materializer"
    )


def _local_scenario_materializer_adapter(runner: Any) -> Any:
    env = getattr(runner, "env", None)
    candidates = (env, getattr(env, "unwrapped", None))
    for owner in candidates:
        adapter = getattr(owner, "_frontres_segment_index_reset_adapter", None)
        if callable(getattr(adapter, "materialize_frontres_local_scenario", None)):
            return adapter
    raise RuntimeError(
        "v015 local scenario requires the Stage 1 index-reset adapter with "
        "MultiMotionCommand.materialize_frontres_local_scenario()"
    )


def _local_scenario_transaction_id(runner: Any, *, update_step: int) -> str:
    sequence = int(getattr(runner, "_frontres_local_scenario_transaction_sequence", 0)) + 1
    runner._frontres_local_scenario_transaction_sequence = sequence
    iteration = int(getattr(runner, "current_learning_iteration", 0) or 0)
    return f"frontres-local-scenario:i{iteration}:u{int(update_step)}:n{sequence}"


def _attach_frontres_local_scenarios(
    runner: Any,
    batch: Any,
    sample: FrontRESSegmentSample,
    *,
    update_step: int,
    transaction_id: str | None = None,
) -> Any:
    """Attach separated v015 local-scenario carriers once per selected source.

    This function is the selection/materialization boundary. The explicit
    Step 5A-S0 sentinel may consume its sealed carrier through later owners;
    this function itself does not reset, sample an action, execute GMT, or
    update PPO.
    """

    if not _index_only_segment_batch(batch):
        return batch
    if getattr(batch, "frontres_fixed_noisy_tape", None) is not None:
        raise RuntimeError("v015 local scenario cannot mix with a legacy complete fixed-Noisy tape")
    if getattr(batch, "frontres_segment_transaction_id", None) is not None:
        raise RuntimeError(
            "v015 local scenario collection cannot enter the frozen-policy transaction route before Step 2"
        )
    future_offsets = _require_frontres_future_offsets(runner)
    adapter = _local_scenario_materializer_adapter(runner)
    batch_size = int(batch.segment_ids.numel())
    source_index = _source_index_for_batch(batch, n=batch_size, device=batch.segment_ids.device)
    horizon_k = getattr(batch, "frontres_segment_budget_horizon_k", None)
    if not isinstance(horizon_k, torch.Tensor):
        raise ValueError("v015 local scenario requires source-aligned frontres_segment_budget_horizon_k")
    horizon_k = horizon_k.detach().to(device=batch.segment_ids.device, dtype=torch.long).flatten()
    if int(horizon_k.numel()) != batch_size or bool((horizon_k <= 0).any()):
        raise ValueError("frontres_segment_budget_horizon_k must be positive [B] data")
    families = tuple(
        str(value)
        for value in (
            getattr(batch, "stage3_index_perturbation_family", ()) or getattr(batch, "perturbation_family", ()) or ()
        )
    )
    strengths = getattr(batch, "stage3_index_perturbation_strength", getattr(batch, "perturbation_strength", None))
    if len(families) != batch_size or not isinstance(strengths, torch.Tensor):
        raise ValueError("v015 local scenario requires physical family and strength for every batch row")
    strengths = strengths.detach().to(device=batch.segment_ids.device, dtype=torch.float32).flatten()
    if int(strengths.numel()) != batch_size:
        raise ValueError("v015 local scenario strength must be [B]")
    specs = tuple(getattr(batch, "specs", ()) or ())
    if len(specs) != batch_size:
        raise ValueError("v015 local scenario requires one Stage 1 index spec per batch row")

    source_rows: dict[int, int] = {}
    source_reference: dict[int, tuple[str, int, int, str, float, str]] = {}
    for row, source in enumerate(source_index.detach().cpu().tolist()):
        spec = specs[row]
        motion_id = str(getattr(spec, "motion_id", ""))
        start_frame = getattr(spec, "start_frame", None)
        if not motion_id or start_frame is None:
            raise ValueError("v015 local scenario specs require motion_id and start_frame")
        segment_id = int(batch.segment_ids[row].item())
        source_identity = f"motion={motion_id}|frame={int(start_frame)}|segment={segment_id}"
        reference = (
            motion_id,
            int(start_frame),
            int(horizon_k[row].item()),
            families[row],
            float(strengths[row].item()),
            source_identity,
        )
        previous = source_reference.setdefault(int(source), reference)
        if previous != reference:
            raise ValueError(
                f"source_index={source} maps to multiple local scenario inputs: first={previous}, row_{row}={reference}"
            )
        source_rows.setdefault(int(source), row)

    if transaction_id is None:
        transaction_id = _local_scenario_transaction_id(runner, update_step=update_step)
    transaction_id = str(transaction_id)
    if not transaction_id:
        raise ValueError("v015 local scenario requires a non-empty immutable transaction id")
    x_t_identity_by_source = {source: reference[-1] for source, reference in source_reference.items()}
    intent_horizon = max(future_offsets)

    def materialize(request: Any) -> Any:
        row = source_rows.get(int(request.source_index))
        if row is None:
            raise RuntimeError(f"missing selected source row for local scenario {request.source_index}")
        motion_id, start_frame, horizon, family, strength, _x_t_identity = source_reference[int(request.source_index)]
        if int(request.horizon_k) != horizon:
            raise RuntimeError("local scenario lifecycle changed the selected K horizon")
        payload = adapter.materialize_frontres_local_scenario(
            motion_id=motion_id,
            start_frame=start_frame,
            horizon_k=horizon,
            intent_horizon=intent_horizon,
            perturbation_family=family,
            perturbation_strength=strength,
        )
        if not isinstance(payload, dict):
            raise TypeError("local scenario adapter must return a dict payload")
        return FrontRESLocalScenarioMaterialization(
            current_root_artifact_t=payload.get("current_root_artifact_t"),
            intent_q29=payload.get("intent_q29"),
            clean_continuation=payload.get("clean_continuation"),
            provenance=payload.get("provenance"),
        )

    lifecycle = FrontRESLocalScenarioLifecycle(
        transaction_id=transaction_id,
        future_offsets=future_offsets,
        x_t_identity_by_source=x_t_identity_by_source,
        materialize_scenario=materialize,
    )
    rows = lifecycle.bind_rows(sample)
    artifacts = torch.stack([scenario.current_root_artifact_t for scenario in rows.scenarios], dim=0).to(
        batch.segment_ids.device
    )
    intent_q29 = torch.stack([scenario.intent_q29 for scenario in rows.scenarios], dim=0).to(batch.segment_ids.device)
    max_horizon = int(horizon_k.max().item())
    clean_continuation = torch.zeros(
        (batch_size, max_horizon, 65),
        dtype=artifacts.dtype,
        device=batch.segment_ids.device,
    )
    clean_continuation_mask = torch.zeros(
        (batch_size, max_horizon),
        dtype=torch.bool,
        device=batch.segment_ids.device,
    )
    for row, scenario in enumerate(rows.scenarios):
        continuation = scenario.clean_continuation.to(batch.segment_ids.device)
        length = int(continuation.shape[0])
        clean_continuation[row, :length] = continuation
        clean_continuation_mask[row, :length] = True
    if tuple(artifacts.shape) != (batch_size, 7):
        raise RuntimeError(f"local scenario current-root carrier must be [B,7], got {tuple(artifacts.shape)}")
    if tuple(intent_q29.shape) != (batch_size, intent_horizon + 1, 29):
        raise RuntimeError(
            "local scenario intent carrier must be [B,H_max+1,29], "
            f"got {tuple(intent_q29.shape)}"
        )
    if tuple(clean_continuation.shape) != (batch_size, max_horizon, 65):
        raise RuntimeError(
            "local scenario Clean continuation carrier must be [B,K_max,65], "
            f"got {tuple(clean_continuation.shape)}"
        )
    object.__setattr__(batch, "frontres_local_scenario_rows", rows)
    object.__setattr__(batch, "frontres_local_scenario_lifecycle", lifecycle)
    object.__setattr__(batch, "frontres_local_scenario_transaction_id", transaction_id)
    object.__setattr__(batch, "frontres_local_scenario_ids", rows.scenario_ids)
    object.__setattr__(batch, "frontres_local_scenario_hashes", rows.noisy_segment_hashes)
    object.__setattr__(batch, "frontres_local_scenario_x_t_identities", tuple(s.request.x_t_identity for s in rows.scenarios))
    object.__setattr__(batch, "frontres_local_scenario_current_root_artifact_t", artifacts.detach().clone())
    object.__setattr__(batch, "frontres_local_scenario_intent_q29", intent_q29.detach().clone())
    object.__setattr__(batch, "frontres_local_scenario_clean_continuation", clean_continuation.detach().clone())
    object.__setattr__(batch, "frontres_local_scenario_clean_continuation_lengths", rows.continuation_lengths)
    object.__setattr__(batch, "frontres_local_scenario_clean_continuation_mask", clean_continuation_mask.detach().clone())
    object.__setattr__(batch, "frontres_local_scenario_provenance", tuple(s.provenance for s in rows.scenarios))
    object.__setattr__(batch, "frontres_future_offsets", future_offsets)
    return batch


def _close_frontres_local_scenarios(batch: Any) -> None:
    lifecycle = getattr(batch, "frontres_local_scenario_lifecycle", None)
    rows = getattr(batch, "frontres_local_scenario_rows", None)
    if lifecycle is None or rows is None:
        return
    closed: list[str] = []
    for scenario_id in dict.fromkeys(rows.scenario_ids):
        lifecycle.close_scenario(scenario_id)
        closed.append(str(scenario_id))
    object.__setattr__(batch, "frontres_local_scenario_closed_ids", tuple(closed))


def _fixed_noisy_transaction_id(runner: Any, *, update_step: int) -> str:
    sequence = int(getattr(runner, "_frontres_fixed_noisy_transaction_sequence", 0)) + 1
    runner._frontres_fixed_noisy_transaction_sequence = sequence
    iteration = int(getattr(runner, "current_learning_iteration", 0) or 0)
    return f"frontres-fixed-noisy:i{iteration}:u{int(update_step)}:n{sequence}"


def _attach_fixed_noisy_scenarios(
    runner: Any,
    batch: Any,
    sample: FrontRESSegmentSample,
    *,
    update_step: int,
) -> Any:
    """Bind one command-materialized Noisy tape per source before any reset/rollout."""

    if not _index_only_segment_batch(batch):
        return batch
    future_offsets = _require_frontres_future_offsets(runner)
    adapter = _fixed_noisy_materializer_adapter(runner)
    batch_size = int(batch.segment_ids.numel())
    source_index = _source_index_for_batch(batch, n=batch_size, device=batch.segment_ids.device)
    horizon_k = getattr(batch, "frontres_segment_budget_horizon_k", None)
    if not isinstance(horizon_k, torch.Tensor):
        raise ValueError("fixed Noisy Segment Replay requires source-aligned frontres_segment_budget_horizon_k")
    horizon_k = horizon_k.detach().to(device=batch.segment_ids.device, dtype=torch.long).flatten()
    if int(horizon_k.numel()) != batch_size or bool((horizon_k <= 0).any()):
        raise ValueError("frontres_segment_budget_horizon_k must be positive [B] data")
    common_frame_count = int(horizon_k.max().item()) + max(future_offsets)
    families = tuple(
        str(value)
        for value in (
            getattr(batch, "stage3_index_perturbation_family", ()) or getattr(batch, "perturbation_family", ()) or ()
        )
    )
    strengths = getattr(batch, "stage3_index_perturbation_strength", getattr(batch, "perturbation_strength", None))
    if len(families) != batch_size or not isinstance(strengths, torch.Tensor):
        raise ValueError("fixed Noisy Segment Replay requires physical family and strength for every batch row")
    strengths = strengths.detach().to(device=batch.segment_ids.device, dtype=torch.float32).flatten()
    if int(strengths.numel()) != batch_size:
        raise ValueError("fixed Noisy Segment Replay strength must be [B]")
    specs = tuple(getattr(batch, "specs", ()) or ())
    if len(specs) != batch_size:
        raise ValueError("fixed Noisy Segment Replay requires one Stage 1 index spec per batch row")

    source_rows: dict[int, int] = {}
    source_reference: dict[int, tuple[str, int, str, float]] = {}
    for row, source in enumerate(source_index.detach().cpu().tolist()):
        spec = specs[row]
        motion_id = str(getattr(spec, "motion_id", ""))
        start_frame = getattr(spec, "start_frame", None)
        if not motion_id or start_frame is None:
            raise ValueError("fixed Noisy Segment Replay specs require motion_id and start_frame")
        reference = (motion_id, int(start_frame), families[row], float(strengths[row].item()))
        previous = source_reference.setdefault(int(source), reference)
        if previous != reference:
            raise ValueError(
                f"source_index={source} maps to multiple materialization inputs: first={previous}, row_{row}={reference}"
            )
        source_rows.setdefault(int(source), row)

    bound_transaction_id = getattr(batch, "frontres_segment_transaction_id", None)
    if bound_transaction_id is None:
        transaction_id = _fixed_noisy_transaction_id(runner, update_step=update_step)
    else:
        transaction_id = str(bound_transaction_id)
        plan = getattr(batch, "frontres_segment_transaction_plan", None)
        snapshot = getattr(batch, "frontres_segment_frozen_policy_snapshot", None)
        _validate_frozen_policy_transaction_plan(plan)
        if not isinstance(snapshot, FrontRESFrozenPolicySnapshot):
            raise TypeError("pre-bound fixed Noisy transaction requires a captured frozen policy snapshot")
        if snapshot.transaction_id != transaction_id or str(plan.transaction_id) != transaction_id:
            raise ValueError("pre-bound fixed Noisy transaction identity is inconsistent")
        snapshot.verify_policy(getattr(getattr(runner, "alg", None), "policy", None))

    def materialize(request: Any) -> Any:
        row = source_rows.get(int(request.source_index))
        if row is None:
            raise RuntimeError(f"missing selected source row for fixed Noisy scenario {request.source_index}")
        motion_id, start_frame, family, strength = source_reference[int(request.source_index)]
        tape = adapter.materialize_frontres_fixed_noisy_tape(
            motion_id=motion_id,
            start_frame=start_frame,
            frame_count=common_frame_count,
            perturbation_family=family,
            perturbation_strength=strength,
        )
        return FrontRESNoisyReferenceMaterialization(
            reference_sequence=tape,
            provenance={
                "materializer_owner": "MultiMotionCommand",
                "motion_id": motion_id,
                "start_frame": start_frame,
                "perturbation_family": family,
                "perturbation_strength": strength,
                "carrier_feature_dim": int(tape.shape[-1]),
            },
        )

    lifecycle = FrontRESFixedNoisyScenarioLifecycle(
        transaction_id=transaction_id,
        future_offsets=future_offsets,
        materialize_reference=materialize,
    )
    rows = lifecycle.bind_rows(sample)
    tape = torch.stack([scenario.reference_sequence for scenario in rows.scenarios], dim=0).to(batch.segment_ids.device)
    if tuple(tape.shape) != (batch_size, common_frame_count, 65):
        raise RuntimeError(f"fixed Noisy tape must be [B,{common_frame_count},65], got {tuple(tape.shape)}")
    object.__setattr__(batch, "frontres_fixed_noisy_scenario_rows", rows)
    object.__setattr__(batch, "frontres_fixed_noisy_lifecycle", lifecycle)
    object.__setattr__(batch, "frontres_fixed_noisy_transaction_id", transaction_id)
    object.__setattr__(batch, "frontres_fixed_noisy_tape", tape.detach().clone())
    object.__setattr__(
        batch,
        "frontres_fixed_noisy_tape_lengths",
        torch.full((batch_size,), common_frame_count, dtype=torch.long, device=batch.segment_ids.device),
    )
    object.__setattr__(batch, "frontres_fixed_noisy_scenario_ids", rows.scenario_ids)
    object.__setattr__(batch, "frontres_fixed_noisy_segment_hashes", rows.noisy_segment_hashes)
    object.__setattr__(batch, "frontres_future_offsets", future_offsets)
    if bound_transaction_id is not None:
        finalize_frontres_frozen_policy_transaction_metadata(runner, batch)
    return batch


def _close_fixed_noisy_scenarios(batch: Any) -> None:
    lifecycle = getattr(batch, "frontres_fixed_noisy_lifecycle", None)
    rows = getattr(batch, "frontres_fixed_noisy_scenario_rows", None)
    if lifecycle is None or rows is None:
        return
    closed: list[str] = []
    for scenario_id in dict.fromkeys(rows.scenario_ids):
        lifecycle.close_scenario(scenario_id)
        closed.append(str(scenario_id))
    object.__setattr__(batch, "frontres_fixed_noisy_closed_scenario_ids", tuple(closed))


def _tensor_nonzero_frac(value: object) -> float:
    if not isinstance(value, torch.Tensor) or int(value.numel()) <= 0:
        return 0.0
    data = value.detach().reshape(-1)
    return float((data != 0).float().mean().cpu().item())


def _build_current_segment_batch(
    runner: Any,
    sample: FrontRESSegmentSample,
    *,
    update_step: int,
    print_probe: bool = True,
    v015_local_scenario_transaction_id: str | None = None,
) -> Any | None:
    dataset = getattr(runner, "_frontres_segment_dataset", None)
    if dataset is None or not hasattr(dataset, "get_segments"):
        if print_probe:
            alg = getattr(runner, "alg", None)
            cache_dir = str(getattr(alg, "frontres_segment_cache_dir", "") or "")
            sampler = getattr(runner, "_frontres_segment_sampler", None)
            sampler_segments = getattr(sampler, "num_segments", "n/a")
            print(
                _log_block(
                    "[FrontRES Segment Batch]",
                    *_kv_lines(
                        "skipped",
                        {
                            "reason": "no_dataset",
                            "cache_dir": cache_dir or "<empty>",
                            "has_dataset": dataset is not None,
                            "dataset_has_get_segments": hasattr(dataset, "get_segments"),
                            "sampler_segments": sampler_segments,
                        },
                    ),
                ),
                flush=True,
            )
        return None
    batch = dataset.get_segments(sample.segment_ids)
    _attach_frontres_segment_trial_plan(batch, sample)
    validation = dataset.validate_batch(batch) if hasattr(dataset, "validate_batch") else None
    valid_count = (
        int(validation.valid_mask.bool().sum().detach().cpu().item())
        if validation is not None and hasattr(validation, "valid_mask")
        else int(sample.segment_ids.numel())
    )
    dynamic_plan = _build_stage3_index_perturbation_plan(runner, batch, update_step=update_step)
    batch = _attach_stage3_index_perturbation_plan(batch, dynamic_plan)
    if v015_local_scenario_transaction_id is not None:
        batch = _attach_frontres_local_scenarios(
            runner,
            batch,
            sample,
            update_step=update_step,
            transaction_id=v015_local_scenario_transaction_id,
        )
    else:
        batch = _attach_fixed_noisy_scenarios(runner, batch, sample, update_step=update_step)
    roles = tuple(getattr(batch, "perturbation_role", ()))
    strength = getattr(batch, "perturbation_strength", None)
    dynamic_family = tuple(getattr(batch, "stage3_index_perturbation_family", ()) or ())
    verbose_probe = _verbose_probe_enabled(runner, sample)
    if print_probe:
        print(
            _log_block(
                "[FrontRES Segment Batch]",
                *_kv_lines(
                    "batch",
                    {
                        "update_step": update_step,
                        "ids": _id_summary(sample.segment_ids),
                        "valid_count": valid_count,
                        "role_counts": _count_summary(roles),
                        "trial_role_counts": _count_summary(tuple(getattr(batch, "frontres_segment_trial_role", ()) or ())),
                        "strength": _tensor_value_summary("strength", strength),
                        "budget_horizon": _tensor_value_summary(
                            "budget_horizon",
                            getattr(batch, "frontres_segment_budget_horizon_k", None),
                        ),
                        "strength_nonzero_frac": _fmt_pct(_tensor_nonzero_frac(strength)),
                        "dynamic_family_counts": _count_summary(dynamic_family),
                    },
                ),
                *_verbose_batch_lines(sample, roles=roles, strength=strength, verbose=verbose_probe),
            ),
            flush=True,
        )
    return batch


def _sample_frontres_v015_transaction_sources(
    sampler: Any,
    *,
    repair_rows: int,
    max_horizon_k: int,
) -> FrontRESSegmentSample:
    """Select distinct sources whose unchanged M budgets exactly fill Repair rows."""

    if repair_rows < 4 or repair_rows % 2 != 0:
        raise RuntimeError("v015 formal transaction requires an even Repair-row budget of at least four")
    max_draws = max(64, repair_rows * 4)
    num_segments = int(getattr(sampler, "num_segments", 0) or 0)
    if 0 < num_segments < 2:
        raise RuntimeError("v015 formal transaction requires at least two valid Segment sources")

    candidates: list[FrontRESSegmentSample] = []
    candidate_ids: set[int] = set()
    reachable = [False] * (repair_rows + 1)
    previous_sum = [-1] * (repair_rows + 1)
    previous_candidate = [-1] * (repair_rows + 1)
    reachable[0] = True

    drawn = 0
    draw_batch_size = max(8, repair_rows * 2)
    while drawn < max_draws and not reachable[repair_rows]:
        requested = min(draw_batch_size, max_draws - drawn)
        sampled = sampler.sample(requested, max_horizon_k=max_horizon_k)
        sampled_count = int(sampled.segment_ids.numel())
        if sampled_count <= 0:
            raise RuntimeError("v015 formal transaction sampler returned no candidate Segment")
        drawn += sampled_count
        for row in range(sampled_count):
            segment_id = int(sampled.segment_ids[row].item())
            if segment_id in candidate_ids:
                continue
            attempts = max(2, int(sampled.rollout_trial_count[row].item()))
            if attempts > repair_rows:
                continue
            candidate_ids.add(segment_id)
            candidate = FrontRESSegmentSample(
                segment_ids=sampled.segment_ids[row : row + 1],
                source=(str(sampled.source[row]),),
                priority=sampled.priority[row : row + 1],
                staleness=sampled.staleness[row : row + 1],
                valid_mask=sampled.valid_mask[row : row + 1],
                segment_state=(
                    sampled.segment_state[row : row + 1]
                    if isinstance(sampled.segment_state, torch.Tensor)
                    else None
                ),
                rollout_trial_count=sampled.rollout_trial_count[row : row + 1],
                horizon_k=sampled.horizon_k[row : row + 1],
                budget_reason=(str(sampled.budget_reason[row]),),
                trial_role=("policy",),
                source_index=torch.zeros(1, dtype=torch.long, device=sampled.segment_ids.device),
                trial_index=torch.zeros(1, dtype=torch.long, device=sampled.segment_ids.device),
            )
            candidate_index = len(candidates)
            candidates.append(candidate)
            for current in range(repair_rows - attempts, -1, -1):
                target = current + attempts
                if reachable[current] and not reachable[target]:
                    reachable[target] = True
                    previous_sum[target] = current
                    previous_candidate[target] = candidate_index
            if reachable[repair_rows]:
                break

    if not reachable[repair_rows]:
        raise RuntimeError(
            "v015 formal transaction could not select a complete multi-Segment x M layout "
            f"for repair_rows={repair_rows} after {len(candidates)} distinct candidates"
        )

    selected_indices: list[int] = []
    cursor = repair_rows
    while cursor > 0:
        candidate_index = previous_candidate[cursor]
        if candidate_index < 0:
            raise RuntimeError("v015 transaction source selection lost its exact-row predecessor")
        selected_indices.append(candidate_index)
        cursor = previous_sum[cursor]
    selected_indices.reverse()
    selected = [candidates[index] for index in selected_indices]
    if len(selected) < 2:
        raise RuntimeError("v015 formal transaction requires at least two selected Segment sources")

    device = selected[0].segment_ids.device
    segment_ids = torch.cat([sample.segment_ids for sample in selected], dim=0)
    segment_state_values = [sample.segment_state for sample in selected]
    segment_state = (
        torch.cat(segment_state_values, dim=0)
        if all(isinstance(value, torch.Tensor) for value in segment_state_values)
        else None
    )
    return FrontRESSegmentSample(
        segment_ids=segment_ids,
        source=tuple(str(sample.source[0]) for sample in selected),
        priority=torch.cat([sample.priority for sample in selected], dim=0),
        staleness=torch.cat([sample.staleness for sample in selected], dim=0),
        valid_mask=torch.cat([sample.valid_mask for sample in selected], dim=0),
        segment_state=segment_state,
        rollout_trial_count=torch.cat([sample.rollout_trial_count for sample in selected], dim=0),
        horizon_k=torch.cat([sample.horizon_k for sample in selected], dim=0),
        budget_reason=tuple(str(sample.budget_reason[0]) for sample in selected),
        trial_role=("policy",) * len(selected),
        source_index=torch.arange(len(selected), dtype=torch.long, device=device),
        trial_index=torch.zeros(len(selected), dtype=torch.long, device=device),
    )


def _prepare_frontres_v015_local_transaction_batch(
    runner: Any,
    *,
    route: str,
) -> SimpleNamespace:
    """Select and seal one complete v015 local transaction before reset.

    Status: active v015 selection owner for the bounded sentinel and ordinary
    formal Stage-3 route. Downstream is one local-scenario batch plus its frozen
    policy/expected-row plan. It selects no legacy fixed tape and does not reset
    the environment, sample an action, update priority, or step an optimizer.
    """

    alg = getattr(runner, "alg", None)
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    if alg is None or sampler is None:
        raise RuntimeError("v015 local transaction requires initialized algorithm and segment sampler owners")
    if route not in {"sentinel", "training"}:
        raise ValueError(f"unknown v015 local transaction route={route!r}")
    sentinel_only = bool(getattr(alg, "frontres_v015_local_sentinel_only", False))
    live_train_enabled = bool(getattr(alg, "frontres_segment_live_train_enabled", False))
    if route == "sentinel" and not sentinel_only:
        raise RuntimeError("v015 local sentinel batch requires its explicit config flag")
    if route == "training" and (sentinel_only or not live_train_enabled):
        raise RuntimeError("v015 formal training batch requires ordinary live training and rejects sentinel mode")
    env_count = int(getattr(getattr(runner, "env", None), "num_envs", 0) or 0)
    if env_count <= 0 or env_count % 4 != 0:
        raise RuntimeError("v015 local transaction requires num_envs divisible by four for complete Repair/Noisy attempts")
    repair_rows = env_count // 2
    if repair_rows < 4:
        raise RuntimeError("v015 local transaction requires at least four Repair rows for 2 Segments x 2 attempts")
    max_horizon = max(1, int(getattr(alg, "frontres_segment_max_horizon_k", 1) or 1))
    iteration = int(getattr(runner, "current_learning_iteration", 0) or 0)
    sequence_attr = f"_frontres_v015_local_{route}_sequence"
    sequence = int(getattr(runner, sequence_attr, 0) or 0) + 1
    setattr(runner, sequence_attr, sequence)
    transaction_id = f"frontres-v015-local-{route}:i{iteration}:n{sequence}"
    base_sample = _sample_frontres_v015_transaction_sources(
        sampler,
        repair_rows=repair_rows,
        max_horizon_k=max_horizon,
    )
    base_ids = base_sample.segment_ids.detach().to(dtype=torch.long).clone()
    if int(base_ids.numel()) < 2 or int(torch.unique(base_ids).numel()) != int(base_ids.numel()):
        raise RuntimeError("v015 local transaction requires distinct selected Segment sources")
    snapshot = capture_frontres_frozen_policy_snapshot(runner, transaction_id=transaction_id)
    frozen_plan = sampler.plan_frozen_policy_transaction(
        base_ids,
        transaction_id=transaction_id,
        policy_snapshot_id=snapshot.policy_snapshot_id,
        max_horizon_k=max_horizon,
        minimum_policy_attempts=2,
    )
    if int(frozen_plan.segment_ids.numel()) != repair_rows:
        raise RuntimeError(
            "v015 local transaction requires environment Repair rows to equal its complete selected transaction: "
            f"repair_rows={repair_rows} planned_attempts={int(frozen_plan.segment_ids.numel())}"
        )
    source_index = frozen_plan.source_index.detach().to(device=base_ids.device, dtype=torch.long)
    expanded_sample = FrontRESSegmentSample(
        segment_ids=frozen_plan.segment_ids.detach().clone(),
        source=tuple(str(base_sample.source[int(index)]) for index in source_index.tolist()),
        priority=base_sample.priority.index_select(0, source_index).detach().clone(),
        staleness=base_sample.staleness.index_select(0, source_index).detach().clone(),
        valid_mask=base_sample.valid_mask.index_select(0, source_index).detach().clone(),
        segment_state=(
            base_sample.segment_state.index_select(0, source_index).detach().clone()
            if isinstance(base_sample.segment_state, torch.Tensor)
            else None
        ),
        rollout_trial_count=frozen_plan.base_trial_count.index_select(0, source_index).detach().clone(),
        horizon_k=frozen_plan.horizon_k.detach().clone(),
        budget_reason=tuple(str(base_sample.budget_reason[int(index)]) for index in source_index.tolist()),
        trial_role=tuple(frozen_plan.trial_role),
        source_index=source_index.detach().clone(),
        trial_index=frozen_plan.trial_index.detach().clone(),
    )
    batch = _build_current_segment_batch(
        runner,
        expanded_sample,
        update_step=sequence,
        print_probe=True,
        v015_local_scenario_transaction_id=transaction_id,
    )
    if batch is None or getattr(batch, "frontres_local_scenario_rows", None) is None:
        raise RuntimeError("v015 local transaction failed to materialize a sealed local scenario batch")
    scenario_ids = tuple(str(value) for value in getattr(batch, "frontres_local_scenario_ids", ()) or ())
    hashes = tuple(str(value) for value in getattr(batch, "frontres_local_scenario_hashes", ()) or ())
    x_t_identities = tuple(str(value) for value in getattr(batch, "frontres_local_scenario_x_t_identities", ()) or ())
    provenance = tuple(getattr(batch, "frontres_local_scenario_provenance", ()) or ())
    specs = tuple(getattr(batch, "specs", ()) or ())
    row_count = int(expanded_sample.segment_ids.numel())
    if len(specs) != row_count or len(scenario_ids) != row_count or len(hashes) != row_count or len(x_t_identities) != row_count:
        raise RuntimeError("v015 local transaction batch lost source-aligned local scenario identities")
    if not provenance or any(not isinstance(value, Mapping) for value in provenance):
        raise RuntimeError("v015 local transaction batch lost local scenario provenance")
    intent_provenance = {str(value.get("intent_q29_provenance", "")) for value in provenance}
    intent_source = {str(value.get("intent_q29_source", "")) for value in provenance}
    if len(intent_provenance) != 1 or len(intent_source) != 1:
        raise RuntimeError("v015 local transaction requires one q29 provenance/source semantic owner per transaction")
    plan = FrontRESV015FormalTransactionPlan(
        snapshot=snapshot,
        motion_ids=tuple(str(getattr(spec, "motion_id", "")) for spec in specs),
        start_frames=torch.tensor(
            [int(getattr(spec, "start_frame", -1)) for spec in specs],
            dtype=torch.long,
            device=expanded_sample.segment_ids.device,
        ),
        segment_ids=expanded_sample.segment_ids,
        source_index=expanded_sample.source_index,
        trial_index=expanded_sample.trial_index,
        horizon_k=expanded_sample.horizon_k,
        scenario_ids=scenario_ids,
        noisy_segment_hashes=hashes,
        x_t_identities=x_t_identities,
        intent_q29_provenance=next(iter(intent_provenance)),
        intent_q29_source=next(iter(intent_source)),
    )
    plan.validate()
    return SimpleNamespace(sample=expanded_sample, batch=batch, plan=plan)


def prepare_frontres_v015_local_sentinel_batch(runner: Any) -> SimpleNamespace:
    """Prepare the explicit bounded sentinel transaction."""

    return _prepare_frontres_v015_local_transaction_batch(runner, route="sentinel")


def prepare_frontres_v015_formal_training_batch(runner: Any) -> SimpleNamespace:
    """Prepare one complete ordinary Stage-3 transaction without legacy rows."""

    return _prepare_frontres_v015_local_transaction_batch(runner, route="training")


def prepare_frontres_v015_policy_quality_item_batch(runner: Any, item: Any) -> SimpleNamespace:
    """Materialize one fixed manifest item as an immutable two-role local batch.

    The manifest selects an existing Stage-1 index row by motion/frame identity.
    Its effective K is the executable-evidence budget, not the cache index
    window used to identify x_t. All Repair attempts use one source identity so
    zero/HSL/policy resets can reuse the same sealed scenario without invoking
    the training sampler or curriculum.
    """

    dataset = getattr(runner, "_frontres_segment_dataset", None)
    specs = tuple(getattr(dataset, "_specs", ()) or ())
    if dataset is None or not callable(getattr(dataset, "get_segments", None)) or not specs:
        raise RuntimeError("v015 quality manifest requires the initialized Stage-1 index dataset")
    motion_id = str(getattr(item, "motion_id", "")).lstrip("./")
    start_frame = int(getattr(item, "start_frame", -1))
    horizon_k = int(getattr(item, "effective_horizon_k", 0))
    matches = tuple(
        spec
        for spec in specs
        if str(getattr(spec, "motion_id", "")).lstrip("./") == motion_id
        and int(getattr(spec, "start_frame", -1)) == start_frame
    )
    if len(matches) != 1:
        cache_horizons = tuple(sorted({int(getattr(spec, "horizon_k", -1)) for spec in matches}))
        raise RuntimeError(
            "v015 quality manifest must resolve motion/start to exactly one loaded Segment identity: "
            f"motion={motion_id!r} frame={start_frame} execution_K={horizon_k} "
            f"matches={len(matches)} cache_horizons={cache_horizons}"
        )
    params = dict(getattr(item, "perturbation_parameters", ()) or ())
    strength_values = [params[name] for name in ("strength", "dr_scale", "scale") if name in params]
    if len(strength_values) != 1:
        raise ValueError("v015 quality manifest requires exactly one strength/dr_scale/scale parameter")
    strength = float(strength_values[0])
    family = str(getattr(item, "perturbation_family", ""))
    if not family or not math.isfinite(strength) or strength < 0.0:
        raise ValueError("v015 quality manifest has invalid perturbation family or strength")
    env_count = int(getattr(getattr(runner, "env", None), "num_envs", 0) or 0)
    if env_count != 8:
        raise RuntimeError("v015 bounded held-out quality requires exactly 8 envs (4 Repair + 4 Noisy)")
    repair_rows = env_count // 2
    device = torch.device(getattr(runner, "device", "cpu"))
    segment_id = int(matches[0].segment_id)
    segment_ids = torch.full((repair_rows,), segment_id, dtype=torch.long, device=device)
    source_index = torch.zeros(repair_rows, dtype=torch.long, device=device)
    sample = FrontRESSegmentSample(
        segment_ids=segment_ids,
        source=("heldout",) * repair_rows,
        priority=torch.ones(repair_rows, dtype=torch.float32, device=device),
        staleness=torch.zeros(repair_rows, dtype=torch.float32, device=device),
        valid_mask=torch.ones(repair_rows, dtype=torch.bool, device=device),
        segment_state=None,
        rollout_trial_count=torch.zeros(repair_rows, dtype=torch.long, device=device),
        horizon_k=torch.full((repair_rows,), horizon_k, dtype=torch.long, device=device),
        budget_reason=("heldout_manifest",) * repair_rows,
        trial_role=("policy",) * repair_rows,
        source_index=source_index,
        trial_index=torch.arange(repair_rows, dtype=torch.long, device=device),
    )
    batch = dataset.get_segments(segment_ids)
    _attach_frontres_segment_trial_plan(batch, sample)
    fixed_plan = SimpleNamespace(
        perturbation_family=(family,) * repair_rows,
        perturbation_strength=torch.full(
            (repair_rows,), strength, dtype=batch.perturbation_strength.dtype, device=device
        ),
        source_index=source_index,
        source_ids=torch.zeros(1, dtype=torch.long, device=device),
        source_perturbation_family=(family,),
        source_perturbation_strength=torch.tensor([strength], dtype=torch.float32, device=device),
        active_modes=(family,),
        complexity="heldout_fixed",
        mix_mode="heldout_fixed",
        mix_diag={"seed": int(getattr(item, "seed", -1))},
        progress=1.0,
        seq_idx=int(getattr(item, "seed", -1)),
    )
    batch = _attach_stage3_index_perturbation_plan(batch, fixed_plan)

    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        torch.manual_seed(int(getattr(item, "seed", -1)))
        batch = _attach_frontres_local_scenarios(
            runner,
            batch,
            sample,
            update_step=0,
            transaction_id=f"frontres-v015-quality:{item.comparison_signature}",
        )
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
    scenario_ids = tuple(getattr(batch, "frontres_local_scenario_ids", ()) or ())
    hashes = tuple(getattr(batch, "frontres_local_scenario_hashes", ()) or ())
    x_t = tuple(getattr(batch, "frontres_local_scenario_x_t_identities", ()) or ())
    if (
        len(scenario_ids) != repair_rows
        or len(set(scenario_ids)) != 1
        or len(set(hashes)) != 1
        or len(set(x_t)) != 1
    ):
        raise RuntimeError("v015 held-out manifest failed to seal one shared scenario/hash/x_t identity")
    return SimpleNamespace(sample=sample, batch=batch)


def build_live_sampler_evidence(
    sample: FrontRESSegmentSample,
    summary: dict[str, object],
    *,
    horizon_k: int,
    reset_result: Any | None = None,
    print_probe: bool = True,
) -> FrontRESSegmentRolloutEvidence:
    """Construct sampler evidence from the formal paired Gain summary.

    Status: active formal evidence boundary; legacy score fields are compatibility
    payload only and cannot affect active sampler decisions.
    Upstream: live probe summary. Downstream: Segment sampler evidence update.
    Evidence: contract-confirmed by frontres_segment_live_sampler_contract.py.
    Gap: real simulator Gain population remains an S4 boundary.
    """
    ids = sample.segment_ids.detach().clone().long()
    row_count = _summary_int(summary, "evidence_row_count", int(ids.numel()))
    if 0 < row_count < int(ids.numel()):
        ids = ids[:row_count]
    n = int(ids.numel())
    device = ids.device
    horizon = _horizon_vector(horizon_k, n=n, device=device)
    reset_success = _reset_success_for_sample(reset_result, n=n, device=device)
    reward = _summary_vector(
        summary,
        keys=("evidence_reward_per_sample", "storage_reward_per_sample", "reward_per_sample"),
        n=n,
        device=device,
        default=_summary_float(summary, "storage_reward_mean", _summary_float(summary, "reward_mean", 0.0)),
    ).float()
    rollout_valid = _summary_bool_vector(
        summary,
        keys=("evidence_valid_mask_per_sample", "storage_valid_mask_per_sample"),
        n=n,
        device=device,
        default=bool(_summary_int(summary, "ppo_valid_count", 0) > 0 and _summary_float(summary, "storage_valid_frac", 0.0) > 0.0),
    )
    fall = _summary_bool_vector(
        summary,
        keys=("evidence_done_any_per_sample", "done_any_per_sample"),
        n=n,
        device=device,
        default=bool(_summary_float(summary, "done_frac", 0.0) >= 0.5),
    )
    score_noisy = _summary_vector(
        summary,
        keys=("score_noisy_per_sample", "noisy_score_per_sample", "baseline_score_per_sample"),
        n=n,
        device=device,
        default=float("nan"),
    ).float()
    score_repaired = _summary_vector(
        summary,
        keys=("score_repaired_per_sample", "repaired_score_per_sample"),
        n=n,
        device=device,
        default=float("nan"),
    ).float()
    gain_source = str(summary.get("gain_source", ""))
    if gain_source != "FRS-GAIN-v002":
        raise ValueError(
            "sampler evidence requires gain_source=FRS-GAIN-v002; "
            f"got {gain_source or 'UNCONFIRMED'}"
        )
    formal_gain = _required_gain_vector(summary, "gain_total_per_sample", n=n, device=device)
    gain_style = _required_gain_vector(summary, "gain_style_per_sample", n=n, device=device)
    gain_physics = _required_gain_vector(summary, "gain_physics_per_sample", n=n, device=device)
    repair_cost = _required_gain_vector(summary, "gain_repair_cost_per_sample", n=n, device=device)
    has_real_scores = torch.isfinite(score_noisy).all() and torch.isfinite(score_repaired).all()
    if has_real_scores:
        score_noisy = score_noisy.clamp(0.0, 1.0)
        score_repaired = score_repaired.clamp(0.0, 1.0)
    gain = formal_gain
    valid_reward = rollout_valid & reset_success
    if print_probe:
        _print_evidence_probe(
            ids,
            reward,
            reset_success,
            rollout_valid,
            valid_reward,
            fall,
            gain,
            score_noisy=score_noisy,
            score_repaired=score_repaired,
            evidence_source=gain_source,
        )
    return FrontRESSegmentRolloutEvidence(
        segment_ids=ids,
        reset_success=reset_success,
        score_noisy=score_noisy,
        score_repaired=score_repaired,
        score_clean=torch.ones(n, dtype=torch.float32, device=device),
        gain_over_noisy=gain,
        fall_repaired=fall,
        contact_consistency=torch.ones(n, dtype=torch.float32, device=device),
        action_norm=torch.ones(n, dtype=torch.float32, device=device),
        valid_reward=valid_reward,
        horizon_k=horizon,
        gain_total=formal_gain,
        gain_style=gain_style,
        gain_physics=gain_physics,
        repair_cost=repair_cost,
        gain_source=gain_source,
    )


def _reset_success_for_sample(reset_result: Any | None, *, n: int, device: torch.device) -> torch.Tensor:
    if reset_result is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    success = getattr(reset_result, "success_mask", None)
    if success is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    success = success.to(device=device).bool().reshape(-1)
    if int(success.numel()) < n:
        raise ValueError(f"reset_success must have at least {n} rows, got {int(success.numel())}")
    if int(success.numel()) > n:
        success = success[:n]
    return success.detach()


def summarize_sampler_step(sampler: FrontRESSegmentSampler, sample: FrontRESSegmentSample) -> dict[str, object]:
    stats = sampler.stats()
    counts = Counter(sample.source)
    trial_counts = Counter(sample.trial_role)
    stale_review_count = int(((sampler.staleness > 0.0) & sampler.solved & (~sampler.invalid)).sum().item())
    return {
        "sampler_update": True,
        "sampler_batch_size": int(sample.segment_ids.numel()),
        "sampler_source_global_count": int(counts.get("global", 0)),
        "sampler_source_replay_count": int(counts.get("replay", 0)),
        "sampler_source_review_count": int(counts.get("review", 0)),
        "sampler_trial_policy_count": int(trial_counts.get("policy", 0)),
        "sampler_trial_search_count": int(trial_counts.get("search", 0)),
        "sampler_budget_trial_count_mean": float(sample.rollout_trial_count.float().mean().item())
        if isinstance(sample.rollout_trial_count, torch.Tensor) and sample.rollout_trial_count.numel() > 0
        else 0.0,
        "sampler_budget_horizon_mean": float(sample.horizon_k.float().mean().item())
        if isinstance(sample.horizon_k, torch.Tensor) and sample.horizon_k.numel() > 0
        else 0.0,
        "sampler_replay_pool_size": int(stats.replay_pool_size),
        "sampler_review_pool_size": int(stats.review_pool_size),
        "sampler_priority_mean": float(stats.priority_mean),
        "sampler_priority_p90": float(stats.priority_p90),
        "sampler_solved_frac": float(stats.solved_frac),
        "sampler_hopeless_frac": float(stats.hopeless_frac),
        "sampler_stale_review_count": stale_review_count,
    }


def _resolve_num_segments(runner: Any) -> int:
    dataset = getattr(runner, "_frontres_segment_dataset", None)
    if dataset is not None and hasattr(dataset, "num_segments"):
        num_segments = dataset.num_segments()
        return max(1, int(num_segments))
    env = getattr(runner, "env", None)
    return max(1, int(getattr(env, "num_envs", 1) or 1))


def _resolve_live_batch_size(runner: Any) -> int:
    env = getattr(runner, "env", None)
    return max(1, int(getattr(env, "num_envs", 1) or 1))


def _resolve_live_scorable_row_budget(runner: Any) -> int:
    """Return the FrontRES repair rows that can receive paired rollout scores."""
    batch_size = _resolve_live_batch_size(runner)
    cfg_present = getattr(runner, "cfg", None) is not None or getattr(runner, "alg_cfg", None) is not None
    if not cfg_present:
        return batch_size
    use_quartet_reward = bool(_runner_cfg_get(runner, "frontres_candidate_rollout_enabled", False))
    divisor = 4 if use_quartet_reward else 3
    return max(1, batch_size // divisor)


def _resolve_live_max_horizon_k(runner: Any) -> int:
    alg = getattr(runner, "alg", None)
    return max(1, int(getattr(alg, "frontres_segment_max_horizon_k", getattr(alg, "frontres_segment_k", 1))))


def _sample_live_segment_rows(runner: Any, sampler: FrontRESSegmentSampler) -> FrontRESSegmentSample:
    row_budget = _resolve_live_scorable_row_budget(runner)
    max_horizon_k = _resolve_live_max_horizon_k(runner)
    if hasattr(sampler, "sample_rollout_rows"):
        return sampler.sample_rollout_rows(row_budget, max_horizon_k=max_horizon_k)
    return sampler.sample(row_budget, max_horizon_k=max_horizon_k)


def _summary_float(summary: dict[str, object], key: str, default: float) -> float:
    try:
        return float(summary.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _summary_int(summary: dict[str, object], key: str, default: int) -> int:
    try:
        return int(summary.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _horizon_vector(horizon_k: int | torch.Tensor | list[int] | tuple[int, ...], *, n: int, device: torch.device) -> torch.Tensor:
    if isinstance(horizon_k, torch.Tensor):
        horizon = horizon_k.to(device=device, dtype=torch.long).reshape(-1)
    elif isinstance(horizon_k, (list, tuple)):
        horizon = torch.tensor(list(horizon_k), dtype=torch.long, device=device).reshape(-1)
    else:
        return torch.full((n,), max(1, int(horizon_k)), dtype=torch.long, device=device)
    if int(horizon.numel()) < n:
        raise ValueError(f"horizon_k must have at least {n} rows, got {int(horizon.numel())}")
    return horizon[:n].clamp_min(1).detach().clone()


def _summary_vector(
    summary: dict[str, object],
    *,
    keys: tuple[str, ...],
    n: int,
    device: torch.device,
    default: float,
) -> torch.Tensor:
    for key in keys:
        if key not in summary:
            continue
        value = summary.get(key)
        tensor = _as_float_tensor(value, device=device)
        if tensor is None or int(tensor.numel()) == 0:
            continue
        if int(tensor.numel()) != n:
            raise ValueError(f"{key} must have {n} rows, got {int(tensor.numel())}")
        return tensor.reshape(-1).detach()
    return torch.full((n,), float(default), dtype=torch.float32, device=device)


def _summary_bool_vector(
    summary: dict[str, object],
    *,
    keys: tuple[str, ...],
    n: int,
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    for key in keys:
        if key not in summary:
            continue
        value = summary.get(key)
        tensor = _as_bool_tensor(value, device=device)
        if tensor is None or int(tensor.numel()) == 0:
            continue
        if int(tensor.numel()) != n:
            raise ValueError(f"{key} must have {n} rows, got {int(tensor.numel())}")
        return tensor.reshape(-1).detach()
    return torch.full((n,), bool(default), dtype=torch.bool, device=device)


def _as_float_tensor(value: object, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).reshape(-1)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.float32, device=device).reshape(-1)
    return None


def _as_bool_tensor(value: object, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device).bool().reshape(-1)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.bool, device=device).reshape(-1)
    return None


def _required_gain_vector(
    summary: dict[str, object],
    key: str,
    *,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Read one finite FRS-GAIN-v002 vector; never synthesize missing evidence."""

    if key not in summary:
        raise ValueError(f"sampler evidence requires {key}")
    value = _as_float_tensor(summary.get(key), device=device)
    if value is None or int(value.numel()) != n:
        got = 0 if value is None else int(value.numel())
        raise ValueError(f"{key} must have {n} rows, got {got}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{key} contains non-finite values")
    return value.detach()


def _print_evidence_probe(
    ids: torch.Tensor,
    reward: torch.Tensor,
    reset_success: torch.Tensor,
    rollout_valid: torch.Tensor,
    valid_reward: torch.Tensor,
    fall: torch.Tensor,
    gain: torch.Tensor,
    *,
    score_noisy: torch.Tensor,
    score_repaired: torch.Tensor,
    evidence_source: str,
) -> None:
    print(
        _log_block(
            "[FrontRES Segment Evidence]",
            *_kv_lines(
                "evidence",
                {
                    "ids": _id_summary(ids),
                    "source": evidence_source,
                    "reset_valid": int(reset_success.bool().sum().detach().cpu().item()),
                    "rollout_valid": int(rollout_valid.bool().sum().detach().cpu().item()),
                    "valid_reward": int(valid_reward.bool().sum().detach().cpu().item()),
                    "fall_count": int(fall.bool().sum().detach().cpu().item()),
                },
            ),
            *_kv_lines(
                "score",
                {
                    "reward_min": _fmt_num(float(reward.min().detach().cpu().item()) if reward.numel() else 0.0),
                    "reward_max": _fmt_num(float(reward.max().detach().cpu().item()) if reward.numel() else 0.0),
                    "noisy": _fmt_num(float(score_noisy.mean().detach().cpu().item()) if score_noisy.numel() else 0.0),
                    "repaired": _fmt_num(
                        float(score_repaired.mean().detach().cpu().item()) if score_repaired.numel() else 0.0
                    ),
                    "gain": _fmt_num(float(gain.mean().detach().cpu().item()) if gain.numel() else 0.0),
                },
            ),
        ),
        flush=True,
    )


def _verbose_probe_enabled(runner: Any, sample: FrontRESSegmentSample | None = None) -> bool:
    alg = getattr(runner, "alg", None)
    if bool(getattr(alg, "frontres_segment_verbose_probe", False)):
        return True
    if sample is None:
        return False
    return int(sample.segment_ids.numel()) <= _VERBOSE_PROBE_BATCH_LIMIT


def _live_detail_log_enabled(runner: Any) -> bool:
    alg = getattr(runner, "alg", None)
    if bool(getattr(alg, "frontres_segment_verbose_probe", False)):
        return True
    count = int(getattr(runner, "_frontres_segment_live_detail_log_count", 0)) + 1
    runner._frontres_segment_live_detail_log_count = count
    warmup = max(0, int(getattr(alg, "frontres_segment_live_log_warmup", 3)))
    interval = max(1, int(getattr(alg, "frontres_segment_live_log_interval", 10)))
    return count <= warmup or count % interval == 0


def _id_summary(ids: torch.Tensor) -> str:
    ids = ids.detach().long().reshape(-1).cpu()
    count = int(ids.numel())
    if count == 0:
        return "count=0 id_min=-1 id_max=-1"
    return f"count={count} id_min={int(ids.min().item())} id_max={int(ids.max().item())}"


def _count_summary(items: tuple[str, ...] | list[str]) -> dict[str, int]:
    return dict(Counter(str(item) for item in items))


def _tensor_value_summary(name: str, value: object) -> str:
    if not isinstance(value, torch.Tensor):
        return f"{name}_count=0 {name}_min=0.000000 {name}_max=0.000000"
    tensor = value.detach().float().reshape(-1).cpu()
    if int(tensor.numel()) == 0:
        return f"{name}_count=0 {name}_min=0.000000 {name}_max=0.000000"
    return (
        f"{name}_count={int(tensor.numel())} "
        f"{name}_min={float(tensor.min().item()):.6f} "
        f"{name}_max={float(tensor.max().item()):.6f}"
    )


def _verbose_sample_lines(sample: FrontRESSegmentSample, *, verbose: bool) -> tuple[str, ...]:
    if not verbose:
        return ()
    horizon = sample.horizon_k.detach().cpu().tolist() if isinstance(sample.horizon_k, torch.Tensor) else []
    trial_index = sample.trial_index.detach().cpu().tolist() if isinstance(sample.trial_index, torch.Tensor) else []
    return (
        f"  sample.segment_ids: {sample.segment_ids.detach().cpu().tolist()}",
        f"  sample.sources: {list(sample.source)}",
        f"  sample.trial_roles: {list(sample.trial_role)}",
        f"  sample.trial_index: {trial_index}",
        f"  sample.budget_horizon: {horizon}",
    )


def _verbose_batch_lines(
    sample: FrontRESSegmentSample,
    *,
    roles: tuple[str, ...],
    strength: object,
    verbose: bool,
) -> tuple[str, ...]:
    if not verbose:
        return ()
    strength_list = strength.detach().cpu().tolist() if isinstance(strength, torch.Tensor) else []
    return (
        f"  batch.segment_ids: {sample.segment_ids.detach().cpu().tolist()}",
        f"  batch.roles: {roles}",
        f"  batch.trial_roles: {list(sample.trial_role)}",
        f"  batch.strength: {strength_list}",
    )


def _print_sample_probe(update_step: int, sample: FrontRESSegmentSample, *, verbose: bool = False) -> None:
    print(
            _log_block(
                "[FrontRES Segment Sample]",
                *_kv_lines(
                    "sample",
                    {
                        "update_step": update_step,
                        "ids": _id_summary(sample.segment_ids),
                        "source_counts": _count_summary(list(sample.source)),
                        "priority": _fmt_num(sample.priority.float().mean().detach().cpu()),
                        "staleness": _fmt_num(sample.staleness.float().mean().detach().cpu()),
                        "valid_count": int(sample.valid_mask.bool().sum().detach().cpu().item()),
                        "trial_role_counts": _count_summary(list(sample.trial_role)),
                        "budget_horizon": _tensor_value_summary("budget_horizon", sample.horizon_k),
                    },
                ),
                *_verbose_sample_lines(sample, verbose=verbose),
            ),
        flush=True,
    )


def _print_sampler_summary(update_step: int, summary: dict[str, object]) -> None:
    print(
            _log_block(
                "[FrontRES Segment Sampler]",
                *_kv_lines(
                    "sampler",
                    {
                        "update_step": update_step,
                        "src": (
                            f"global:{int(summary['sampler_source_global_count'])},"
                            f"replay:{int(summary['sampler_source_replay_count'])},"
                            f"review:{int(summary['sampler_source_review_count'])}"
                        ),
                        "pool": (
                            f"replay:{int(summary['sampler_replay_pool_size'])},"
                            f"review:{int(summary['sampler_review_pool_size'])}"
                        ),
                        "trial": (
                            f"policy:{int(summary.get('sampler_trial_policy_count', 0))},"
                            f"search:{int(summary.get('sampler_trial_search_count', 0))},"
                            f"budget_mean:{_fmt_num(summary.get('sampler_budget_trial_count_mean', 0.0))},"
                            f"horizon_mean:{_fmt_num(summary.get('sampler_budget_horizon_mean', 0.0))}"
                        ),
                        "priority": _fmt_num(summary["sampler_priority_mean"]),
                        "useful": (
                            f"mean:{_fmt_num(summary.get('sampler_update_useful_mean', 0.0))},"
                            f"max:{_fmt_num(summary.get('sampler_update_useful_max', 0.0))}"
                        ),
                        "priority_flow": (
                            f"before:{_fmt_num(summary.get('sampler_update_priority_before_mean', 0.0))},"
                            f"after:{_fmt_num(summary.get('sampler_update_priority_after_mean', 0.0))},"
                            f"max:{_fmt_num(summary.get('sampler_update_priority_after_max', 0.0))}"
                        ),
                        "gain": (
                            f"mean:{_fmt_num(summary.get('sampler_update_gain_mean', 0.0))},"
                            f"pos:{_fmt_pct(summary.get('sampler_update_gain_pos_frac', 0.0))}"
                        ),
                        "oracle": (
                            f"gap:{_fmt_num(summary.get('sampler_update_oracle_gap_mean', 0.0))},"
                            f"confidence:{_fmt_num(summary.get('sampler_update_confidence_mean', 0.0))},"
                            f"delayed:{int(summary.get('sampler_update_delayed_regret_count', 0))}"
                        ),
                        "update": (
                            f"valid:{int(summary.get('sampler_update_valid_count', 0))},"
                            f"fall:{int(summary.get('sampler_update_fall_count', 0))},"
                            f"hopeless:{int(summary.get('sampler_update_hopeless_count', 0))},"
                            f"segments:{int(summary.get('sampler_update_segment_count', 0))},"
                            f"trials:{int(summary.get('sampler_update_trial_count', 0))},"
                            f"replay_candidates:{int(summary.get('sampler_update_replay_candidate_count', 0))}"
                        ),
                        "solved": _fmt_pct(summary["sampler_solved_frac"]),
                        "hopeless": _fmt_pct(summary["sampler_hopeless_frac"]),
                        "stale_review": int(summary["sampler_stale_review_count"]),
                    },
                ),
            ),
        flush=True,
    )
