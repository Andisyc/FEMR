from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import importlib.util
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import torch

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_sampler",
    Path(__file__).resolve().with_name("frontres_formal_runtime_probe.py"),
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


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
    gain_source: str = "FRS-GAIN-v005-vector-physics-constraints"

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
            or self.gain_source != "FRS-GAIN-v005-vector-physics-constraints"
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
    if getattr(return_evidence, "gain_source", None) != "FRS-GAIN-v005-vector-physics-constraints":
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

    def validate(self) -> None:
        if not self.transaction_id:
            raise ValueError("transaction_id must be non-empty")
        if not self.policy_snapshot_id:
            raise ValueError("policy_snapshot_id must be non-empty")
        if int(self.minimum_policy_attempts) < 2:
            raise ValueError("minimum_policy_attempts must be at least two")

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


@dataclass(frozen=True)
class FrontRESFixedNoisyScenarioRequest:
    """Describe one selection-time Noisy reference scenario."""

    transaction_id: str
    scenario_id: str
    segment_id: int
    source_index: int
    horizon_k: int
    future_offsets: tuple[int, ...]

    @property
    def required_frame_count(self) -> int:
        """Return the tape length needed by K rollout steps and every H offset."""
        return int(self.horizon_k) + max(self.future_offsets)

    def validate(self) -> None:
        if not self.transaction_id:
            raise ValueError("transaction_id must be non-empty")
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if int(self.segment_id) < 0:
            raise ValueError(f"segment_id must be non-negative, got {self.segment_id}")
        if int(self.source_index) < 0:
            raise ValueError(f"source_index must be non-negative, got {self.source_index}")
        if int(self.horizon_k) <= 0:
            raise ValueError(f"horizon_k must be positive, got {self.horizon_k}")
        if not self.future_offsets:
            raise ValueError("future_offsets must be non-empty")
        if any(int(offset) <= 0 for offset in self.future_offsets):
            raise ValueError(f"future_offsets must be positive, got {self.future_offsets}")
        if tuple(sorted(set(int(offset) for offset in self.future_offsets))) != tuple(self.future_offsets):
            raise ValueError(f"future_offsets must be strictly ordered and unique, got {self.future_offsets}")


@dataclass(frozen=True)
class FrontRESNoisyReferenceMaterialization:
    """Return the deployable Noisy sequence produced by one selected scenario."""

    reference_sequence: torch.Tensor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class FrontRESFixedNoisyScenario:
    """Seal one Noisy reference tape so retries cannot alter its content."""

    request: FrontRESFixedNoisyScenarioRequest
    noisy_segment_hash: str
    _reference_sequence: torch.Tensor
    provenance: Mapping[str, str | int | float | bool]

    @classmethod
    def from_materialization(
        cls,
        request: FrontRESFixedNoisyScenarioRequest,
        materialization: FrontRESNoisyReferenceMaterialization,
    ) -> "FrontRESFixedNoisyScenario":
        request.validate()
        if not isinstance(materialization, FrontRESNoisyReferenceMaterialization):
            raise TypeError(
                "materialize_reference must return FrontRESNoisyReferenceMaterialization, "
                f"got {type(materialization)!r}"
            )
        sequence = _validate_fixed_noisy_sequence(materialization.reference_sequence, request=request)
        return cls(
            request=request,
            noisy_segment_hash=_fixed_noisy_sequence_hash(sequence),
            _reference_sequence=sequence,
            provenance=_freeze_noisy_provenance(materialization.provenance),
        )

    def __post_init__(self) -> None:
        self.request.validate()
        sequence = _validate_fixed_noisy_sequence(self._reference_sequence, request=self.request)
        observed_hash = _fixed_noisy_sequence_hash(sequence)
        if self.noisy_segment_hash != observed_hash:
            raise ValueError(
                "noisy_segment_hash does not match the immutable reference sequence: "
                f"expected {observed_hash}, got {self.noisy_segment_hash}"
            )
        object.__setattr__(self, "_reference_sequence", sequence)
        object.__setattr__(self, "provenance", _freeze_noisy_provenance(self.provenance))

    @property
    def scenario_id(self) -> str:
        return self.request.scenario_id

    @property
    def required_frame_count(self) -> int:
        return self.request.required_frame_count

    @property
    def reference_sequence(self) -> torch.Tensor:
        """Return a copy so external retry code cannot mutate the sealed tape."""
        return self._reference_sequence.detach().clone()

    def probe(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "segment_id": int(self.request.segment_id),
            "source_index": int(self.request.source_index),
            "horizon_k": int(self.request.horizon_k),
            "future_offsets": tuple(self.request.future_offsets),
            "required_frame_count": self.required_frame_count,
            "reference_shape": tuple(self._reference_sequence.shape),
            "reference_dtype": str(self._reference_sequence.dtype),
            "reference_device": str(self._reference_sequence.device),
            "reference_finite": bool(torch.isfinite(self._reference_sequence).all().item()),
            "noisy_segment_hash": self.noisy_segment_hash,
        }


@dataclass(frozen=True)
class FrontRESFixedNoisyScenarioRows:
    """Attach one immutable selected scenario to every expanded trial row."""

    scenarios: tuple[FrontRESFixedNoisyScenario, ...]
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_ids", _immutable_row_tensor("segment_ids", self.segment_ids))
        object.__setattr__(self, "source_index", _immutable_row_tensor("source_index", self.source_index))
        object.__setattr__(self, "trial_index", _immutable_row_tensor("trial_index", self.trial_index))
        object.__setattr__(self, "horizon_k", _immutable_row_tensor("horizon_k", self.horizon_k))
        self.validate()

    @property
    def batch_size(self) -> int:
        return len(self.scenarios)

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(scenario.scenario_id for scenario in self.scenarios)

    @property
    def noisy_segment_hashes(self) -> tuple[str, ...]:
        return tuple(scenario.noisy_segment_hash for scenario in self.scenarios)

    @property
    def required_frame_counts(self) -> torch.Tensor:
        return torch.tensor(
            [scenario.required_frame_count for scenario in self.scenarios],
            dtype=torch.long,
            device=self.segment_ids.device,
        )

    def scenario_for_row(self, row_index: int) -> FrontRESFixedNoisyScenario:
        if not 0 <= int(row_index) < self.batch_size:
            raise IndexError(f"row_index {row_index} is outside batch size {self.batch_size}")
        return self.scenarios[int(row_index)]

    def validate(self) -> None:
        count = self.batch_size
        for name, value in (
            ("segment_ids", self.segment_ids),
            ("source_index", self.source_index),
            ("trial_index", self.trial_index),
            ("horizon_k", self.horizon_k),
        ):
            if int(value.numel()) != count:
                raise ValueError(f"{name} must have {count} rows, got {int(value.numel())}")
        by_source: dict[int, tuple[str, str]] = {}
        for row, scenario in enumerate(self.scenarios):
            scenario.request.validate()
            source_index = int(self.source_index[row].item())
            segment_id = int(self.segment_ids[row].item())
            horizon_k = int(self.horizon_k[row].item())
            if source_index != int(scenario.request.source_index):
                raise ValueError(f"row {row} source_index does not match scenario identity")
            if segment_id != int(scenario.request.segment_id):
                raise ValueError(f"row {row} segment_id does not match scenario identity")
            if horizon_k != int(scenario.request.horizon_k):
                raise ValueError(f"row {row} horizon_k does not match scenario identity")
            identity = (scenario.scenario_id, scenario.noisy_segment_hash)
            previous = by_source.setdefault(source_index, identity)
            if previous != identity:
                raise ValueError(f"source_index={source_index} maps to multiple Noisy scenario identities")

    def probe(self) -> dict[str, Any]:
        self.validate()
        return {
            "row_count": self.batch_size,
            "scenario_count": len(set(self.scenario_ids)),
            "source_index": self.source_index.detach().cpu().tolist(),
            "trial_index": self.trial_index.detach().cpu().tolist(),
            "scenario_ids": self.scenario_ids,
            "noisy_segment_hashes": self.noisy_segment_hashes,
            "required_frame_counts": self.required_frame_counts.detach().cpu().tolist(),
        }


class FrontRESFixedNoisyScenarioLifecycle:
    """Own selection-time Noisy scenario creation and closing for one transaction."""

    def __init__(
        self,
        *,
        transaction_id: str,
        future_offsets: Iterable[int],
        materialize_reference: Callable[[FrontRESFixedNoisyScenarioRequest], FrontRESNoisyReferenceMaterialization],
    ) -> None:
        self.transaction_id = str(transaction_id)
        self.future_offsets = tuple(int(offset) for offset in future_offsets)
        if not callable(materialize_reference):
            raise TypeError("materialize_reference must be callable")
        FrontRESFixedNoisyScenarioRequest(
            transaction_id=self.transaction_id,
            scenario_id="validation",
            segment_id=0,
            source_index=0,
            horizon_k=1,
            future_offsets=self.future_offsets,
        ).validate()
        self._materialize_reference = materialize_reference
        self._open_scenarios: dict[str, FrontRESFixedNoisyScenario] = {}
        self._closed_scenarios: dict[str, FrontRESFixedNoisyScenario] = {}

    def bind_rows(self, sample: FrontRESSegmentSample) -> FrontRESFixedNoisyScenarioRows:
        """为每个 selected base source 绑定一次 Noisy scenario, 并复用于全部展开行.

        函数名说明:
            这是 selection-time lifecycle owner. 它不执行 reset, 不写 command,
            不改变 PPO role 或 optimizer.

        主链路:
            上游: FrontRESSegmentSampler 的 expanded source/trial rows.
            下游: 后续 reset/command connector 消费按行对齐的 immutable scenario.

        语义:
            同一 source_index 必须有同一 segment/K/scenario/hash. 关闭后的 identity
            只能保留证据, 不得在本 transaction 下重新 materialize.

        Status: S1 lifecycle is consumed by the S2 command/reset/actor connector.
        Evidence: S1 frontres_fixed_noisy_segment_lifecycle_contract.py; S2 offline
        command/reset/actor-context contracts.
        Boundary: this owner remains selection-only; it does not reset command state,
        evaluate the actor, or update PPO.
        """
        # B1: 读取并验证 expanded rows 的 source/trial/K identity.
        segment_ids, source_index, trial_index, horizon_k = _scenario_row_fields(sample)
        source_specs: dict[int, tuple[int, int]] = {}
        for row, (segment_id, source, horizon) in enumerate(
            zip(segment_ids.tolist(), source_index.tolist(), horizon_k.tolist(), strict=True)
        ):
            if int(source) < 0:
                raise ValueError(f"row {row} has negative source_index={source}")
            spec = (int(segment_id), int(horizon))
            previous = source_specs.setdefault(int(source), spec)
            if previous != spec:
                raise ValueError(
                    f"source_index={source} has inconsistent Segment/K identity: "
                    f"first={previous}, row_{row}={spec}"
                )

        # B2: 每个 source 只调用一次 materializer, 并封存 sequence/hash.
        scenario_by_source: dict[int, FrontRESFixedNoisyScenario] = {}
        for source, (segment_id, horizon) in source_specs.items():
            scenario_id = self._scenario_id(source_index=source, segment_id=segment_id)
            if scenario_id in self._closed_scenarios:
                raise RuntimeError(
                    f"closed Noisy scenario {scenario_id} cannot be rematerialized under the same identity"
                )
            scenario = self._open_scenarios.get(scenario_id)
            if scenario is None:
                request = FrontRESFixedNoisyScenarioRequest(
                    transaction_id=self.transaction_id,
                    scenario_id=scenario_id,
                    segment_id=segment_id,
                    source_index=source,
                    horizon_k=horizon,
                    future_offsets=self.future_offsets,
                )
                materialization = self._materialize_reference(request)
                scenario = FrontRESFixedNoisyScenario.from_materialization(request, materialization)
                self._open_scenarios[scenario_id] = scenario
            scenario_by_source[source] = scenario

        # B3: 以原始行顺序返回同一 scenario 的 M 次复用视图.
        rows = FrontRESFixedNoisyScenarioRows(
            scenarios=tuple(scenario_by_source[int(source)] for source in source_index.tolist()),
            segment_ids=segment_ids,
            source_index=source_index,
            trial_index=trial_index,
            horizon_k=horizon_k,
        )
        rows.validate()
        return rows

    def close_scenario(self, scenario_id: str) -> FrontRESFixedNoisyScenario:
        """Close the semantic binding while retaining immutable evidence."""
        scenario = self._open_scenarios.pop(str(scenario_id), None)
        if scenario is None:
            if str(scenario_id) in self._closed_scenarios:
                raise RuntimeError(f"Noisy scenario {scenario_id} is already closed")
            raise KeyError(f"unknown open Noisy scenario {scenario_id}")
        self._closed_scenarios[scenario.scenario_id] = scenario
        return scenario

    def closed_scenario(self, scenario_id: str) -> FrontRESFixedNoisyScenario | None:
        return self._closed_scenarios.get(str(scenario_id))

    def _scenario_id(self, *, source_index: int, segment_id: int) -> str:
        return f"{self.transaction_id}:source-{int(source_index)}:segment-{int(segment_id)}"


def _scenario_row_fields(sample: FrontRESSegmentSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_ids = _immutable_row_tensor("segment_ids", sample.segment_ids)
    source_index = getattr(sample, "source_index", None)
    trial_index = getattr(sample, "trial_index", None)
    horizon_k = getattr(sample, "horizon_k", None)
    if not isinstance(source_index, torch.Tensor):
        raise ValueError("fixed Noisy scenario lifecycle requires sample.source_index")
    if not isinstance(trial_index, torch.Tensor):
        raise ValueError("fixed Noisy scenario lifecycle requires sample.trial_index")
    if not isinstance(horizon_k, torch.Tensor):
        raise ValueError("fixed Noisy scenario lifecycle requires sample.horizon_k")
    source_index = _immutable_row_tensor("source_index", source_index)
    trial_index = _immutable_row_tensor("trial_index", trial_index)
    horizon_k = _immutable_row_tensor("horizon_k", horizon_k)
    count = int(segment_ids.numel())
    for name, value in (("source_index", source_index), ("trial_index", trial_index), ("horizon_k", horizon_k)):
        if int(value.numel()) != count:
            raise ValueError(f"sample.{name} must have {count} rows, got {int(value.numel())}")
    if bool((horizon_k <= 0).any().item()):
        raise ValueError("sample.horizon_k must be positive")
    return segment_ids, source_index, trial_index, horizon_k


def _immutable_row_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {tuple(value.shape)}")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached lifecycle metadata")
    return value.detach().to(dtype=torch.long).clone()


def _validate_fixed_noisy_sequence(
    reference_sequence: torch.Tensor,
    *,
    request: FrontRESFixedNoisyScenarioRequest,
) -> torch.Tensor:
    if not isinstance(reference_sequence, torch.Tensor):
        raise TypeError("reference_sequence must be a torch.Tensor")
    if reference_sequence.ndim != 2:
        raise ValueError(
            "reference_sequence must have shape [frames, deployable_features], "
            f"got {tuple(reference_sequence.shape)}"
        )
    if reference_sequence.requires_grad:
        raise ValueError("reference_sequence must be detached scenario data")
    if not torch.is_floating_point(reference_sequence):
        raise TypeError(f"reference_sequence must be floating point, got {reference_sequence.dtype}")
    if not bool(torch.isfinite(reference_sequence).all().item()):
        raise ValueError("reference_sequence contains non-finite values")
    if int(reference_sequence.shape[0]) < request.required_frame_count:
        raise ValueError(
            "reference_sequence coverage is shorter than K + H_max: "
            f"got {int(reference_sequence.shape[0])}, required {request.required_frame_count}"
        )
    if int(reference_sequence.shape[1]) <= 0:
        raise ValueError("reference_sequence must expose at least one deployable feature")
    return reference_sequence.detach().clone().contiguous()


def _fixed_noisy_sequence_hash(reference_sequence: torch.Tensor) -> str:
    value = reference_sequence.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _freeze_noisy_provenance(provenance: Mapping[str, Any]) -> Mapping[str, str | int | float | bool]:
    if not isinstance(provenance, Mapping):
        raise TypeError("Noisy scenario provenance must be a mapping")
    frozen: dict[str, str | int | float | bool] = {}
    for key, value in provenance.items():
        name = str(key)
        if not name:
            raise ValueError("Noisy scenario provenance keys must be non-empty")
        if "clean" in name.lower():
            raise ValueError(f"Noisy scenario provenance must not carry Clean reference data: {name}")
        if isinstance(value, torch.Tensor):
            raise ValueError(f"Noisy scenario provenance must not carry tensor payloads: {name}")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Noisy scenario provenance value {name} must be scalar, got {type(value)!r}")
        frozen[name] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class FrontRESLocalScenarioRequest:
    """Selection-time identity for one v015 local reference scenario."""

    transaction_id: str
    scenario_id: str
    segment_id: int
    source_index: int
    x_t_identity: str
    horizon_k: int
    future_offsets: tuple[int, ...]

    @property
    def intent_frame_count(self) -> int:
        """Dense q29 carrier covers current t through the largest H offset."""
        return max(self.future_offsets) + 1

    def validate(self) -> None:
        if not isinstance(self.transaction_id, str) or not self.transaction_id:
            raise ValueError("transaction_id must be non-empty")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not isinstance(self.x_t_identity, str) or not self.x_t_identity:
            raise ValueError("x_t_identity must be non-empty")
        if int(self.segment_id) < 0:
            raise ValueError(f"segment_id must be non-negative, got {self.segment_id}")
        if int(self.source_index) < 0:
            raise ValueError(f"source_index must be non-negative, got {self.source_index}")
        if int(self.horizon_k) <= 0:
            raise ValueError(f"horizon_k must be positive, got {self.horizon_k}")
        if not self.future_offsets:
            raise ValueError("future_offsets must be non-empty")
        if any(int(offset) <= 0 for offset in self.future_offsets):
            raise ValueError(f"future_offsets must be positive, got {self.future_offsets}")
        if tuple(sorted(set(int(offset) for offset in self.future_offsets))) != tuple(self.future_offsets):
            raise ValueError(f"future_offsets must be strictly ordered and unique, got {self.future_offsets}")


@dataclass(frozen=True)
class FrontRESLocalScenarioMaterialization:
    """Unsealed command-owner output with explicit H/K provenance separation."""

    current_root_artifact_t: torch.Tensor
    intent_q29: torch.Tensor
    clean_continuation: torch.Tensor
    expected_support: torch.Tensor
    expected_support_envelope: torch.Tensor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class FrontRESLocalScenario:
    """Seal one local scenario so M attempts cannot mutate or resample it."""

    request: FrontRESLocalScenarioRequest
    noisy_segment_hash: str
    _current_root_artifact_t: torch.Tensor
    _intent_q29: torch.Tensor
    _clean_continuation: torch.Tensor
    _expected_support: torch.Tensor
    _expected_support_envelope: torch.Tensor
    provenance: Mapping[str, str | int | float | bool]

    @classmethod
    def from_materialization(
        cls,
        request: FrontRESLocalScenarioRequest,
        materialization: FrontRESLocalScenarioMaterialization,
    ) -> "FrontRESLocalScenario":
        request.validate()
        if not isinstance(materialization, FrontRESLocalScenarioMaterialization):
            raise TypeError(
                "materialize_scenario must return FrontRESLocalScenarioMaterialization, "
                f"got {type(materialization)!r}"
            )
        artifact, intent, continuation, expected_support, expected_support_envelope = _validate_local_scenario_payload(
            materialization.current_root_artifact_t,
            materialization.intent_q29,
            materialization.clean_continuation,
            materialization.expected_support,
            materialization.expected_support_envelope,
            request=request,
        )
        provenance = _freeze_local_scenario_provenance(materialization.provenance)
        return cls(
            request=request,
            noisy_segment_hash=_local_scenario_hash(
                request, artifact, intent, continuation, expected_support, expected_support_envelope, provenance
            ),
            _current_root_artifact_t=artifact,
            _intent_q29=intent,
            _clean_continuation=continuation,
            _expected_support=expected_support,
            _expected_support_envelope=expected_support_envelope,
            provenance=provenance,
        )

    def __post_init__(self) -> None:
        self.request.validate()
        artifact, intent, continuation, expected_support, expected_support_envelope = _validate_local_scenario_payload(
            self._current_root_artifact_t,
            self._intent_q29,
            self._clean_continuation,
            self._expected_support,
            self._expected_support_envelope,
            request=self.request,
        )
        provenance = _freeze_local_scenario_provenance(self.provenance)
        observed_hash = _local_scenario_hash(
            self.request, artifact, intent, continuation, expected_support, expected_support_envelope, provenance
        )
        if self.noisy_segment_hash != observed_hash:
            raise ValueError(
                "noisy_segment_hash does not match the immutable local scenario: "
                f"expected {observed_hash}, got {self.noisy_segment_hash}"
            )
        object.__setattr__(self, "_current_root_artifact_t", artifact)
        object.__setattr__(self, "_intent_q29", intent)
        object.__setattr__(self, "_clean_continuation", continuation)
        object.__setattr__(self, "_expected_support", expected_support)
        object.__setattr__(self, "_expected_support_envelope", expected_support_envelope)
        object.__setattr__(self, "provenance", provenance)

    @property
    def scenario_id(self) -> str:
        return self.request.scenario_id

    @property
    def current_root_artifact_t(self) -> torch.Tensor:
        return self._current_root_artifact_t.detach().clone()

    @property
    def intent_q29(self) -> torch.Tensor:
        return self._intent_q29.detach().clone()

    @property
    def clean_continuation(self) -> torch.Tensor:
        return self._clean_continuation.detach().clone()

    @property
    def expected_support(self) -> torch.Tensor:
        return self._expected_support.detach().clone()

    @property
    def expected_support_envelope(self) -> torch.Tensor:
        return self._expected_support_envelope.detach().clone()

    def probe(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "segment_id": int(self.request.segment_id),
            "source_index": int(self.request.source_index),
            "x_t_identity": self.request.x_t_identity,
            "horizon_k": int(self.request.horizon_k),
            "future_offsets": tuple(self.request.future_offsets),
            "current_root_artifact_shape": tuple(self._current_root_artifact_t.shape),
            "intent_q29_shape": tuple(self._intent_q29.shape),
            "clean_continuation_shape": tuple(self._clean_continuation.shape),
            "expected_support_shape": tuple(self._expected_support.shape),
            "expected_support_envelope_shape": tuple(self._expected_support_envelope.shape),
            "intent_q29_provenance": self.provenance["intent_q29_provenance"],
            "clean_continuation_provenance": self.provenance["clean_continuation_provenance"],
            "noisy_segment_hash": self.noisy_segment_hash,
        }


@dataclass(frozen=True)
class FrontRESLocalScenarioRows:
    """Row-aligned immutable local scenarios for M attempts over selected sources."""

    scenarios: tuple[FrontRESLocalScenario, ...]
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_ids", _immutable_row_tensor("segment_ids", self.segment_ids))
        object.__setattr__(self, "source_index", _immutable_row_tensor("source_index", self.source_index))
        object.__setattr__(self, "trial_index", _immutable_row_tensor("trial_index", self.trial_index))
        object.__setattr__(self, "horizon_k", _immutable_row_tensor("horizon_k", self.horizon_k))
        self.validate()

    @property
    def batch_size(self) -> int:
        return len(self.scenarios)

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(scenario.scenario_id for scenario in self.scenarios)

    @property
    def noisy_segment_hashes(self) -> tuple[str, ...]:
        return tuple(scenario.noisy_segment_hash for scenario in self.scenarios)

    @property
    def intent_frame_counts(self) -> torch.Tensor:
        return torch.tensor(
            [scenario.request.intent_frame_count for scenario in self.scenarios],
            dtype=torch.long,
            device=self.segment_ids.device,
        )

    @property
    def continuation_lengths(self) -> torch.Tensor:
        return self.horizon_k.detach().clone()

    def scenario_for_row(self, row_index: int) -> FrontRESLocalScenario:
        if not 0 <= int(row_index) < self.batch_size:
            raise IndexError(f"row_index {row_index} is outside batch size {self.batch_size}")
        return self.scenarios[int(row_index)]

    def validate(self) -> None:
        count = self.batch_size
        for name, value in (
            ("segment_ids", self.segment_ids),
            ("source_index", self.source_index),
            ("trial_index", self.trial_index),
            ("horizon_k", self.horizon_k),
        ):
            if int(value.numel()) != count:
                raise ValueError(f"{name} must have {count} rows, got {int(value.numel())}")
        by_source: dict[int, tuple[str, str, str]] = {}
        for row, scenario in enumerate(self.scenarios):
            scenario.request.validate()
            source = int(self.source_index[row].item())
            if source != int(scenario.request.source_index):
                raise ValueError(f"row {row} source_index does not match local scenario identity")
            if int(self.segment_ids[row].item()) != int(scenario.request.segment_id):
                raise ValueError(f"row {row} segment_id does not match local scenario identity")
            if int(self.horizon_k[row].item()) != int(scenario.request.horizon_k):
                raise ValueError(f"row {row} horizon_k does not match local scenario identity")
            identity = (scenario.scenario_id, scenario.noisy_segment_hash, scenario.request.x_t_identity)
            previous = by_source.setdefault(source, identity)
            if previous != identity:
                raise ValueError(f"source_index={source} maps to multiple local scenario identities")

    def probe(self) -> dict[str, Any]:
        self.validate()
        return {
            "row_count": self.batch_size,
            "scenario_count": len(set(self.scenario_ids)),
            "scenario_ids": self.scenario_ids,
            "noisy_segment_hashes": self.noisy_segment_hashes,
            "intent_frame_counts": self.intent_frame_counts.detach().cpu().tolist(),
            "continuation_lengths": self.continuation_lengths.detach().cpu().tolist(),
        }


class FrontRESLocalScenarioLifecycle:
    """Selection-time owner for immutable v015 local scenarios only."""

    def __init__(
        self,
        *,
        transaction_id: str,
        future_offsets: Iterable[int],
        x_t_identity_by_source: Mapping[int, str],
        materialize_scenario: Callable[[FrontRESLocalScenarioRequest], FrontRESLocalScenarioMaterialization],
    ) -> None:
        self.transaction_id = str(transaction_id)
        self.future_offsets = tuple(int(offset) for offset in future_offsets)
        if not callable(materialize_scenario):
            raise TypeError("materialize_scenario must be callable")
        if not isinstance(x_t_identity_by_source, Mapping):
            raise TypeError("x_t_identity_by_source must be a mapping")
        if any(not isinstance(identity, str) or not identity for identity in x_t_identity_by_source.values()):
            raise ValueError("x_t_identity_by_source values must be nonempty strings")
        self._x_t_identity_by_source = {int(source): identity for source, identity in x_t_identity_by_source.items()}
        FrontRESLocalScenarioRequest(
            transaction_id=self.transaction_id,
            scenario_id="validation",
            segment_id=0,
            source_index=0,
            x_t_identity=self._x_t_identity_by_source.get(0, "validation-x_t"),
            horizon_k=1,
            future_offsets=self.future_offsets,
        ).validate()
        if any(source < 0 or not identity for source, identity in self._x_t_identity_by_source.items()):
            raise ValueError("x_t_identity_by_source must use nonnegative sources and nonempty identities")
        self._materialize_scenario = materialize_scenario
        self._open_scenarios: dict[str, FrontRESLocalScenario] = {}
        self._closed_scenarios: dict[str, FrontRESLocalScenario] = {}

    def bind_rows(self, sample: FrontRESSegmentSample) -> FrontRESLocalScenarioRows:
        """Materialize once per selected source and reuse the sealed local scenario for M rows."""

        segment_ids, source_index, trial_index, horizon_k = _scenario_row_fields(sample)
        source_specs: dict[int, tuple[int, int]] = {}
        for row, (segment_id, source, horizon) in enumerate(
            zip(segment_ids.tolist(), source_index.tolist(), horizon_k.tolist(), strict=True)
        ):
            if int(source) < 0:
                raise ValueError(f"row {row} has negative source_index={source}")
            spec = (int(segment_id), int(horizon))
            previous = source_specs.setdefault(int(source), spec)
            if previous != spec:
                raise ValueError(
                    f"source_index={source} has inconsistent Segment/K identity: "
                    f"first={previous}, row_{row}={spec}"
                )

        scenario_by_source: dict[int, FrontRESLocalScenario] = {}
        for source, (segment_id, horizon) in source_specs.items():
            scenario_id = self._scenario_id(source_index=source, segment_id=segment_id)
            if scenario_id in self._closed_scenarios:
                raise RuntimeError(
                    f"closed local scenario {scenario_id} cannot be rematerialized under the same identity"
                )
            scenario = self._open_scenarios.get(scenario_id)
            if scenario is None:
                x_t_identity = self._x_t_identity_by_source.get(source)
                if not x_t_identity:
                    raise ValueError(f"missing x_t_identity for selected source_index={source}")
                request = FrontRESLocalScenarioRequest(
                    transaction_id=self.transaction_id,
                    scenario_id=scenario_id,
                    segment_id=segment_id,
                    source_index=source,
                    x_t_identity=x_t_identity,
                    horizon_k=horizon,
                    future_offsets=self.future_offsets,
                )
                materialization = self._materialize_scenario(request)
                scenario = FrontRESLocalScenario.from_materialization(request, materialization)
                self._open_scenarios[scenario_id] = scenario
            scenario_by_source[source] = scenario

        rows = FrontRESLocalScenarioRows(
            scenarios=tuple(scenario_by_source[int(source)] for source in source_index.tolist()),
            segment_ids=segment_ids,
            source_index=source_index,
            trial_index=trial_index,
            horizon_k=horizon_k,
        )
        rows.validate()
        return rows

    def close_scenario(self, scenario_id: str) -> FrontRESLocalScenario:
        scenario = self._open_scenarios.pop(str(scenario_id), None)
        if scenario is None:
            if str(scenario_id) in self._closed_scenarios:
                raise RuntimeError(f"local scenario {scenario_id} is already closed")
            raise KeyError(f"unknown open local scenario {scenario_id}")
        self._closed_scenarios[scenario.scenario_id] = scenario
        return scenario

    def closed_scenario(self, scenario_id: str) -> FrontRESLocalScenario | None:
        return self._closed_scenarios.get(str(scenario_id))

    def _scenario_id(self, *, source_index: int, segment_id: int) -> str:
        return f"{self.transaction_id}:source-{int(source_index)}:segment-{int(segment_id)}"


def _validate_local_scenario_payload(
    current_root_artifact_t: torch.Tensor,
    intent_q29: torch.Tensor,
    clean_continuation: torch.Tensor,
    expected_support: torch.Tensor,
    expected_support_envelope: torch.Tensor,
    *,
    request: FrontRESLocalScenarioRequest,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def freeze(name: str, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.requires_grad:
            raise ValueError(f"{name} must be detached scenario data")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be floating point, got {value.dtype}")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains non-finite values")
        return value.detach().clone().contiguous()

    artifact = freeze("current_root_artifact_t", current_root_artifact_t)
    intent = freeze("intent_q29", intent_q29)
    continuation = freeze("clean_continuation", clean_continuation)
    support = freeze("expected_support", expected_support)
    envelope = freeze("expected_support_envelope", expected_support_envelope)
    if artifact.ndim != 1 or tuple(artifact.shape) != (7,):
        raise ValueError(f"current_root_artifact_t must have shape [7], got {tuple(artifact.shape)}")
    if intent.ndim != 2 or tuple(intent.shape) != (request.intent_frame_count, 29):
        raise ValueError(
            "intent_q29 must have shape "
            f"[{request.intent_frame_count},29], got {tuple(intent.shape)}"
        )
    if continuation.ndim != 2 or tuple(continuation.shape) != (int(request.horizon_k), 65):
        raise ValueError(
            "clean_continuation must have shape "
            f"[{int(request.horizon_k)},65], got {tuple(continuation.shape)}"
        )
    if tuple(support.shape) != (int(request.horizon_k), 2):
        raise ValueError(f"expected_support must have shape [{int(request.horizon_k)},2], got {tuple(support.shape)}")
    if bool(((support != 0.0) & (support != 1.0)).any()):
        raise ValueError("expected_support must contain only binary left/right support states")
    if tuple(envelope.shape) != (int(request.horizon_k), 6):
        raise ValueError(
            f"expected_support_envelope must have shape [{int(request.horizon_k)},6], got {tuple(envelope.shape)}"
        )
    if bool((envelope[:, 4:6] <= 0.0).any()):
        raise ValueError("expected_support_envelope half extents must be positive")
    return artifact, intent, continuation, support, envelope


def _freeze_local_scenario_provenance(provenance: Mapping[str, Any]) -> Mapping[str, str | int | float | bool]:
    if not isinstance(provenance, Mapping):
        raise TypeError("local scenario provenance must be a mapping")
    frozen: dict[str, str | int | float | bool] = {}
    for key, value in provenance.items():
        name = str(key)
        if not name:
            raise ValueError("local scenario provenance keys must be non-empty")
        if isinstance(value, torch.Tensor):
            raise ValueError(f"local scenario provenance must not carry tensor payloads: {name}")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"local scenario provenance value {name} must be scalar, got {type(value)!r}")
        frozen[name] = value
    required = {
        "current_root_artifact_provenance": "noisy_root_artifact_t",
        "intent_q29_provenance": "deployment_noisy_q29",
        "clean_continuation_provenance": "clean_gmt_only",
        "expected_support_provenance": "clean_gmt_physics_only",
        "expected_support_envelope_provenance": "clean_gmt_physics_only",
        "intent_q29_source": None,
    }
    for key, expected in required.items():
        if key not in frozen:
            raise ValueError(f"local scenario provenance is missing {key}")
        if expected is not None and frozen[key] != expected:
            raise ValueError(f"local scenario {key} must be {expected!r}, got {frozen[key]!r}")
    intent_source = str(frozen["intent_q29_source"]).lower()
    if "root" in intent_source or "global" in intent_source or "clean" in intent_source:
        raise ValueError("intent_q29_source must exclude root/global/Clean actor-reference fields")
    return MappingProxyType(frozen)


def _local_scenario_hash(
    request: FrontRESLocalScenarioRequest,
    current_root_artifact_t: torch.Tensor,
    intent_q29: torch.Tensor,
    clean_continuation: torch.Tensor,
    expected_support: torch.Tensor,
    expected_support_envelope: torch.Tensor,
    provenance: Mapping[str, str | int | float | bool],
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("transaction_id", request.transaction_id),
        ("scenario_id", request.scenario_id),
        ("segment_id", int(request.segment_id)),
        ("source_index", int(request.source_index)),
        ("x_t_identity", request.x_t_identity),
        ("horizon_k", int(request.horizon_k)),
        ("future_offsets", tuple(int(offset) for offset in request.future_offsets)),
    ):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    for name, tensor in (
        ("current_root_artifact_t", current_root_artifact_t),
        ("intent_q29", intent_q29),
        ("clean_continuation", clean_continuation),
        ("expected_support", expected_support),
        ("expected_support_envelope", expected_support_envelope),
    ):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    for key, value in sorted(provenance.items(), key=lambda item: str(item[0])):
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class FrontRESSegmentSamplerStats:
    replay_pool_size: int
    review_pool_size: int
    invalid_count: int
    seen_count: int
    priority_mean: float
    priority_p90: float
    solved_frac: float
    hopeless_frac: float
    unknown_count: int = 0
    promising_count: int = 0
    frontier_count: int = 0
    delayed_regret_count: int = 0
    solved_count: int = 0
    hopeless_count: int = 0
    mean_trial_count: float = 0.0
    oracle_gap_mean: float = 0.0
    confidence_mean: float = 0.0


@dataclass(frozen=True)
class FrontRESSegmentSamplerUpdateProbe:
    count: int
    valid_count: int
    fall_count: int
    gain_mean: float
    gain_pos_frac: float
    useful_mean: float
    useful_max: float
    priority_before_mean: float
    priority_after_mean: float
    priority_after_max: float
    replay_candidate_count: int
    hopeless_count: int
    delayed_regret_count: int = 0
    segment_count: int = 0
    trial_count: int = 0
    oracle_gap_mean: float = 0.0
    confidence_mean: float = 0.0


class FrontRESSegmentSampler:
    """Prioritized sampler where each level is a motion segment."""

    def __init__(
        self,
        num_segments: int,
        global_frac: float = 0.4,
        replay_frac: float = 0.5,
        review_frac: float = 0.1,
        priority_mode: str = "learning_value",
        staleness_weight: float = 0.1,
        min_replay_score: float = 0.05,
        max_hopeless_replay_frac: float = 0.1,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_segments <= 0:
            raise ValueError(f"num_segments must be positive, got {num_segments}")
        if priority_mode != "learning_value":
            raise ValueError(f"unsupported priority_mode: {priority_mode}")
        if min(global_frac, replay_frac, review_frac) < 0.0:
            raise ValueError("sampling fractions must be non-negative")
        total = global_frac + replay_frac + review_frac
        if total <= 0.0:
            raise ValueError("at least one sampling fraction must be positive")
        self.num_segments = int(num_segments)
        self.global_frac = float(global_frac) / total
        self.replay_frac = float(replay_frac) / total
        self.review_frac = float(review_frac) / total
        self.priority_mode = priority_mode
        self.staleness_weight = float(staleness_weight)
        self.min_replay_score = float(min_replay_score)
        self.max_hopeless_replay_frac = float(max_hopeless_replay_frac)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

        self.priority = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.staleness = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.seen = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
        self.solved = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
        self.hopeless = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
        self.invalid = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
        self.invalid_reasons: dict[int, str] = {}
        self.segment_state = torch.full(
            (self.num_segments,),
            int(FrontRESSegmentState.UNKNOWN),
            dtype=torch.long,
            device=self.device,
        )
        self.evidence_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.valid_evidence_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.success_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.fall_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.best_gain = torch.full((self.num_segments,), -float("inf"), dtype=torch.float32, device=self.device)
        self.best_short_gain = torch.full((self.num_segments,), -float("inf"), dtype=torch.float32, device=self.device)
        self.best_long_gain = torch.full((self.num_segments,), -float("inf"), dtype=torch.float32, device=self.device)
        self.last_horizon_k = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.last_trial_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.last_policy_gain = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_mean_gain = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_success_frac = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_fall_frac = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_oracle_gap = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_confidence = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)

    def reset_for_deterministic_eval(self, *, seed: int) -> None:
        """Reset replay history before a cross-checkpoint sequence evaluation.

        Checkpoints persist replay frontier state because it is part of training
        resume semantics.  That state must not choose different evaluation
        motions for two checkpoints being compared.  Rebuild only this sampler
        with the same configuration and seed; the policy, normalizer, and
        environment state remain owned by the runner and are untouched.
        """
        fresh = type(self)(
            num_segments=self.num_segments,
            global_frac=self.global_frac,
            replay_frac=self.replay_frac,
            review_frac=self.review_frac,
            priority_mode=self.priority_mode,
            staleness_weight=self.staleness_weight,
            min_replay_score=self.min_replay_score,
            max_hopeless_replay_frac=self.max_hopeless_replay_frac,
            seed=int(seed),
            device=self.device,
        )
        self.__dict__.update(fresh.__dict__)

    def sample(self, batch_size: int, *, max_horizon_k: int = 8) -> FrontRESSegmentSample:
        """按 replay mixture 选择 base segments 并附加 rollout budget.

        函数名说明:
            `sample` 是 base-segment selection owner, 选择 segment 和来源; 它不
            展开多 trial 行, 正式 live row expansion 由 `sample_rollout_rows` 完成.

        主链路:
            上游: runner 给出 base batch size 和最大 K.
            下游: 返回 segment id, source, priority, state 和初始 rollout budget.

        语义:
            sampling source 决定 global/replay/review 混合, segment state 决定后续
            K/trial budget. 两者不能被 PPO post-update diagnostics 污染.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        valid_ids = self._valid_ids()
        if valid_ids.numel() == 0:
            raise RuntimeError("no valid segments are available")

        ids: list[int] = []
        sources: list[str] = []
        for _ in range(batch_size):
            source = self._choose_source()
            segment_id, effective_source = self._sample_one(source, valid_ids)
            ids.append(segment_id)
            sources.append(effective_source)
            self.seen[segment_id] = True

        segment_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.staleness += 1.0
        self.staleness[segment_ids] = 0.0
        budget = self.plan_rollout_budget(segment_ids, max_horizon_k=max_horizon_k)
        return FrontRESSegmentSample(
            segment_ids=segment_ids,
            source=tuple(sources),
            priority=self.priority[segment_ids].clone(),
            staleness=self.staleness[segment_ids].clone(),
            valid_mask=~self.invalid[segment_ids],
            segment_state=self.segment_state[segment_ids].clone(),
            rollout_trial_count=budget.trial_count.clone(),
            horizon_k=budget.horizon_k.clone(),
            budget_reason=budget.reason,
            trial_role=tuple("policy" for _ in ids),
            source_index=torch.arange(int(segment_ids.numel()), dtype=torch.long, device=self.device),
            trial_index=torch.zeros(int(segment_ids.numel()), dtype=torch.long, device=self.device),
        )

    def sample_rollout_rows(self, row_budget: int, *, max_horizon_k: int = 8) -> FrontRESSegmentSample:
        """展开 per-segment trial budget, 生成正式 live rollout rows.

        函数名说明:
            `sample_rollout_rows` 是 live row sampler, 把 base segment 变成固定行数
            的 policy-first trials; 它不是 env reset 或 PPO batch builder.

        主链路:
            上游: live sampler helper 给出 split-env 可用 repair row budget.
            下游: batch builder 按 `source_index/trial_index/trial_role` 构造 reset 和
            rollout metadata.

        语义:
            返回行数服从 env row budget, 每行仍保留原 segment, K 和 trial 身份,
            因而多个 trial 不得被误当成多个独立 segment.
        """
        # B1: 选择 base segments, 直到计划 trial rows 覆盖 live row budget.
        if row_budget <= 0:
            raise ValueError(f"row_budget must be positive, got {row_budget}")
        valid_ids = self._valid_ids()
        if valid_ids.numel() == 0:
            raise RuntimeError("no valid segments are available")

        base_ids: list[int] = []
        base_sources: list[str] = []
        planned_rows = 0
        while planned_rows < row_budget:
            source = self._choose_source()
            segment_id, effective_source = self._sample_one(source, valid_ids)
            base_ids.append(segment_id)
            base_sources.append(effective_source)
            self.seen[segment_id] = True
            budget = self.plan_rollout_budget([segment_id], max_horizon_k=max_horizon_k)
            planned_rows += max(1, int(budget.trial_count[0].item()))

        base_segment_ids = torch.tensor(base_ids, dtype=torch.long, device=self.device)
        plan = self.expand_rollout_trials(base_segment_ids, max_horizon_k=max_horizon_k)
        keep = min(int(row_budget), int(plan.segment_ids.numel()))
        source_index = plan.source_index[:keep].to(device=self.device, dtype=torch.long)
        row_ids = plan.segment_ids[:keep].to(device=self.device, dtype=torch.long)
        self.staleness += 1.0
        self.staleness[torch.unique(row_ids)] = 0.0
        base_budget = self.plan_rollout_budget(base_segment_ids, max_horizon_k=max_horizon_k)
        source_rows = source_index.detach().cpu().tolist()
        # B2: 物化带 source, K, role 和 trial identity 的 row-level sample.
        sample = FrontRESSegmentSample(
            segment_ids=row_ids.detach().clone(),
            source=tuple(str(base_sources[int(row)]) for row in source_rows),
            priority=self.priority[row_ids].detach().clone(),
            staleness=self.staleness[row_ids].detach().clone(),
            valid_mask=~self.invalid[row_ids],
            segment_state=self.segment_state[row_ids].detach().clone(),
            rollout_trial_count=base_budget.trial_count[source_index].detach().clone(),
            horizon_k=plan.horizon_k[:keep].detach().clone(),
            budget_reason=tuple(str(base_budget.reason[int(row)]) for row in source_rows),
            trial_role=tuple(plan.trial_role[:keep]),
            source_index=source_index.detach().clone(),
            trial_index=plan.trial_index[:keep].to(device=self.device, dtype=torch.long).detach().clone(),
        )
        # B3: AUDIT-SAMPLER-01 截获 live batch builder 实际消费的 sample.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-SAMPLER-01",
            segment_ids=sample.segment_ids,
            source=sample.source,
            horizon_k=sample.horizon_k,
            trial_role=sample.trial_role,
        )
        return sample

    def update(self, evidence: FrontRESSegmentRolloutEvidence) -> None:
        self.update_with_probe(evidence)

    def update_with_probe(self, evidence: FrontRESSegmentRolloutEvidence) -> FrontRESSegmentSamplerUpdateProbe:
        """用 rollout-time evidence 更新 segment replay state 和 priority.

        函数名说明:
            `update_with_probe` 是 sampler state transaction owner, 同时返回可读的
            update probe; 它不是 PPO update, 也不读取 post-update KL 或梯度.

        主链路:
            上游: live probe 提交带 segment/trial identity 的 paired rollout evidence.
            下游: 更新 priority, solved/hopeless/state 和 curriculum history, 供下一次
            sample/K planning 使用.

        语义:
            更新依据必须来自 policy update 前的 rollout evidence. 多 trial 先按
            segment 聚合, 再改变持久 replay state.
        """
        # B1: 改变 replay state 前, 先按 segment 聚合 rollout-time evidence.
        row_ids = evidence.segment_ids.to(device=self.device, dtype=torch.long).flatten()
        self._validate_ids(row_ids)
        trial = self.aggregate_trial_evidence(evidence)
        ids = trial.segment_ids
        useful_rows = self._learning_value(evidence)
        useful = self._mean_by_ids(row_ids, useful_rows, ids)
        valid = trial.valid_mask
        fall_count = torch.round(trial.fall_frac * trial.trial_count.float()).long()

        current = self.priority[ids]
        self.priority[ids] = torch.where(valid, 0.8 * current + 0.2 * useful, current)
        self.seen[ids] = True
        self._update_segment_state_from_trials(trial)
        self.priority[ids] = torch.where(self.solved[ids] | self.hopeless[ids], self.priority[ids] * 0.25, self.priority[ids])
        priority_after = self.priority[ids]
        replay_candidates = (~self.invalid[ids]) & (~self.solved[ids]) & (~self.hopeless[ids]) & (priority_after >= self.min_replay_score)
        # B2: 将 rollout-time evidence 写入 priority 和持久 segment state.
        update_probe = FrontRESSegmentSamplerUpdateProbe(
            count=int(row_ids.numel()),
            valid_count=int(trial.valid_trial_count.sum().item()),
            fall_count=int(fall_count.sum().item()),
            gain_mean=float(self._active_gain(evidence).mean().item()) if row_ids.numel() > 0 else 0.0,
            gain_pos_frac=float((self._active_gain(evidence) > 0.0).float().mean().item()) if row_ids.numel() > 0 else 0.0,
            useful_mean=float(useful.mean().item()) if useful.numel() > 0 else 0.0,
            useful_max=float(useful.max().item()) if useful.numel() > 0 else 0.0,
            priority_before_mean=float(current.mean().item()) if current.numel() > 0 else 0.0,
            priority_after_mean=float(priority_after.mean().item()) if priority_after.numel() > 0 else 0.0,
            priority_after_max=float(priority_after.max().item()) if priority_after.numel() > 0 else 0.0,
            replay_candidate_count=int(replay_candidates.sum().item()),
            hopeless_count=int(self.hopeless[ids].sum().item()),
            delayed_regret_count=int((self.segment_state[ids] == int(FrontRESSegmentState.DELAYED_REGRET)).sum().item()),
            segment_count=int(ids.numel()),
            trial_count=int(trial.trial_count.sum().item()),
            oracle_gap_mean=float(trial.oracle_gap.mean().item()) if trial.oracle_gap.numel() > 0 else 0.0,
            confidence_mean=float(trial.confidence.mean().item()) if trial.confidence.numel() > 0 else 0.0,
        )
        # B3: AUDIT-SAMPLER-01 同步截获该 transaction 完成后的 priority state.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-SAMPLER-01",
            priority_before=update_probe.priority_before_mean,
            priority_after=update_probe.priority_after_mean,
            valid_count=update_probe.valid_count,
            trial_count=update_probe.trial_count,
        )
        return update_probe

    def aggregate_trial_evidence(self, evidence: FrontRESSegmentRolloutEvidence) -> FrontRESSegmentTrialEvidence:
        ids = evidence.segment_ids.to(device=self.device, dtype=torch.long).flatten()
        self._validate_ids(ids)
        unique_ids = torch.unique(ids, sorted=True)
        gain = self._active_gain(evidence)
        valid = evidence.reset_success.to(self.device).bool().flatten() & evidence.valid_reward.to(self.device).bool().flatten()
        fall = evidence.fall_repaired.to(self.device).bool().flatten() | (~valid)
        horizon = evidence.horizon_k.to(self.device).long().flatten()

        trial_count: list[int] = []
        valid_trial_count: list[int] = []
        policy_gain: list[float] = []
        best_gain: list[float] = []
        mean_gain: list[float] = []
        success_frac: list[float] = []
        fall_frac: list[float] = []
        oracle_gap: list[float] = []
        confidence: list[float] = []
        horizon_k: list[int] = []
        valid_mask: list[bool] = []

        for segment_id in unique_ids.tolist():
            mask = ids == int(segment_id)
            trial_n = int(mask.sum().item())
            row_gain = gain[mask]
            row_valid = valid[mask]
            row_fall = fall[mask]
            row_horizon = horizon[mask]
            valid_gain = row_gain[row_valid]
            valid_n = int(row_valid.sum().item())
            policy = float(row_gain[0].item()) if trial_n else 0.0
            best = float(valid_gain.max().item()) if valid_n else 0.0
            mean = float(valid_gain.mean().item()) if valid_n else 0.0
            success = (row_valid & (~row_fall) & (row_gain > self.min_replay_score)).float()
            fall_or_invalid = row_fall.float()
            gap = max(0.0, best - policy)
            fall_rate = float(fall_or_invalid.mean().item()) if trial_n else 0.0
            conf = min(1.0, float(valid_n) / 3.0) * max(0.0, 1.0 - fall_rate)

            trial_count.append(trial_n)
            valid_trial_count.append(valid_n)
            policy_gain.append(policy)
            best_gain.append(best)
            mean_gain.append(mean)
            success_frac.append(float(success.mean().item()) if trial_n else 0.0)
            fall_frac.append(fall_rate)
            oracle_gap.append(gap)
            confidence.append(conf)
            horizon_k.append(int(row_horizon.max().item()) if trial_n else 0)
            valid_mask.append(valid_n > 0)

        return FrontRESSegmentTrialEvidence(
            segment_ids=unique_ids,
            trial_count=torch.tensor(trial_count, dtype=torch.long, device=self.device),
            valid_trial_count=torch.tensor(valid_trial_count, dtype=torch.long, device=self.device),
            policy_gain=torch.tensor(policy_gain, dtype=torch.float32, device=self.device),
            best_gain=torch.tensor(best_gain, dtype=torch.float32, device=self.device),
            mean_gain=torch.tensor(mean_gain, dtype=torch.float32, device=self.device),
            success_frac=torch.tensor(success_frac, dtype=torch.float32, device=self.device),
            fall_frac=torch.tensor(fall_frac, dtype=torch.float32, device=self.device),
            oracle_gap=torch.tensor(oracle_gap, dtype=torch.float32, device=self.device),
            confidence=torch.tensor(confidence, dtype=torch.float32, device=self.device),
            score_noisy=torch.full((len(trial_count),), float("nan"), dtype=torch.float32, device=self.device),
            score_repaired=torch.full((len(trial_count),), float("nan"), dtype=torch.float32, device=self.device),
            horizon_k=torch.tensor(horizon_k, dtype=torch.long, device=self.device),
            valid_mask=torch.tensor(valid_mask, dtype=torch.bool, device=self.device),
        )

    def plan_rollout_budget(
        self,
        segment_ids: Iterable[int] | torch.Tensor,
        *,
        max_horizon_k: int = 8,
    ) -> FrontRESSegmentRolloutBudget:
        """把 segment state 映射为纯 K-step rollout budget.

        函数名说明:
            `plan_rollout_budget` 是 K curriculum 的 pure planner, 只计算 horizon K,
            trial count 和 reason; 它不触碰 env, storage 或 PPO.

        主链路:
            上游: sampler 提供选中 segment 及其持久 state/history.
            下游: `expand_rollout_trials` 和 live batch builder 消费不可变 budget.

        语义:
            K 表示本次修复证据需要持续观察的时间窗. state 越接近 delayed regret,
            越需要更长 horizon 或更多 trials, 但不得超过正式 max_horizon_k.
        """
        # B1: 读取拥有 curriculum progression 的持久 segment state.
        ids = self._ids_tensor(segment_ids)
        max_horizon = int(max_horizon_k)
        if max_horizon <= 0:
            raise ValueError(f"max_horizon_k must be positive, got {max_horizon_k}")
        states = self.segment_state[ids].clone()
        trial_count = torch.ones_like(ids, dtype=torch.long, device=self.device)
        horizon_k = torch.empty_like(ids, dtype=torch.long, device=self.device)
        reasons: list[str] = []

        for row, segment_id in enumerate(ids.tolist()):
            state = FrontRESSegmentState(int(states[row].item()))
            trial_n = 1
            preferred_horizon = 8
            reason = "unknown_probe"
            if state == FrontRESSegmentState.PROMISING:
                trial_n = 3
                preferred_horizon = 16
                reason = "promising_local_trials"
            elif state == FrontRESSegmentState.FRONTIER:
                trial_n = 6
                use_long = (
                    float(self.last_success_frac[segment_id].item()) < 0.75
                    or int(self.last_trial_count[segment_id].item()) >= 2
                )
                preferred_horizon = 32 if use_long else 16
                reason = "frontier_multi_trial"
            elif state == FrontRESSegmentState.DELAYED_REGRET:
                trial_n = 6
                preferred_horizon = 64 if max_horizon >= 64 else 32
                reason = "delayed_regret_long_check"
            elif state == FrontRESSegmentState.SOLVED:
                trial_n = 1
                preferred_horizon = 64
                reason = "solved_review"
            elif state == FrontRESSegmentState.HOPELESS:
                trial_n = 1
                preferred_horizon = 8
                reason = "hopeless_recheck"

            trial_count[row] = int(trial_n)
            horizon_k[row] = self._bounded_horizon(preferred_horizon, max_horizon)
            reasons.append(reason)

        # B2: 物化每个 segment 的不可变 curriculum budget.
        budget = FrontRESSegmentRolloutBudget(
            segment_ids=ids.clone(),
            trial_count=trial_count,
            horizon_k=horizon_k,
            segment_state=states,
            reason=tuple(reasons),
        )
        # B3: AUDIT-KPLAN-01 截获 row expansion 前的 per-segment K 和 trial budget.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-KPLAN-01",
            segment_ids=budget.segment_ids,
            segment_state=budget.segment_state,
            trial_count=budget.trial_count,
            horizon_k=budget.horizon_k,
            reason=budget.reason,
        )
        return budget

    def expand_rollout_trials(
        self,
        segment_ids: Iterable[int] | torch.Tensor,
        *,
        max_horizon_k: int = 8,
    ) -> FrontRESSegmentTrialPlan:
        """Expand per-segment budget into policy-first trial rows for future live wiring."""
        """把 per-segment budget 展开为 policy-first trial rows.

        函数名说明:
            `expand_rollout_trials` 是 K plan 到 row layout 的转换 owner; 它不重新
            规划 K, 也不改变 sampler priority.

        主链路:
            上游: `plan_rollout_budget` 提供 segment-level horizon 和 trial count.
            下游: live sampler/reset 通过 source/trial index 消费 expanded rows.

        语义:
            每个 segment 的第 0 行必须是 policy trial. 后续 probe rows 共享同一 K
            和 source segment, 使短窗和长窗 evidence 能按 trial identity 聚合.
        """
        # B1: 不改变 K, 将一个 budget row 展开为 policy-first trial rows.
        budget = self.plan_rollout_budget(segment_ids, max_horizon_k=max_horizon_k)
        expanded_ids: list[int] = []
        source_index: list[int] = []
        trial_index: list[int] = []
        horizon: list[int] = []
        roles: list[str] = []
        for source_row, segment_id in enumerate(budget.segment_ids.tolist()):
            count = int(budget.trial_count[source_row].item())
            horizon_value = int(budget.horizon_k[source_row].item())
            for trial_row in range(count):
                expanded_ids.append(int(segment_id))
                source_index.append(source_row)
                trial_index.append(trial_row)
                horizon.append(horizon_value)
                roles.append("policy" if trial_row == 0 else "search")
        # B2: 保留 source/trial indexes, 物化 policy-first rows.
        plan = FrontRESSegmentTrialPlan(
            segment_ids=torch.tensor(expanded_ids, dtype=torch.long, device=self.device),
            source_index=torch.tensor(source_index, dtype=torch.long, device=self.device),
            trial_index=torch.tensor(trial_index, dtype=torch.long, device=self.device),
            horizon_k=torch.tensor(horizon, dtype=torch.long, device=self.device),
            trial_role=tuple(roles),
            base_segment_ids=budget.segment_ids.clone(),
            base_trial_count=budget.trial_count.clone(),
        )
        # B3: AUDIT-KROLLOUT-01 截获 reset/rollout 实际消费的 expanded rows.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-KROLLOUT-01",
            segment_ids=plan.segment_ids,
            source_index=plan.source_index,
            trial_index=plan.trial_index,
            horizon_k=plan.horizon_k,
            trial_role=plan.trial_role,
        )
        return plan

    def plan_frozen_policy_transaction(
        self,
        segment_ids: Iterable[int] | torch.Tensor,
        *,
        transaction_id: str,
        policy_snapshot_id: str,
        max_horizon_k: int = 8,
        minimum_policy_attempts: int = 2,
        active_horizon_k: int | None = None,
    ) -> FrontRESFrozenPolicyTransactionPlan:
        """Plan a complete all-policy attempt layout without mutating sampler state.

        This is deliberately not a policy snapshot implementation: the runner
        must later capture and verify the supplied ``policy_snapshot_id`` before
        executing these rows.  The sampler owns only row grouping, M and K.
        """

        ids = self._ids_tensor(segment_ids)
        self._validate_ids(ids)
        if int(ids.numel()) < 2:
            raise ValueError("frozen policy transaction requires at least two selected segments")
        if int(torch.unique(ids).numel()) != int(ids.numel()):
            raise ValueError("frozen policy transaction selected duplicate segment groups")
        if not str(transaction_id):
            raise ValueError("transaction_id must be non-empty")
        if not str(policy_snapshot_id):
            raise ValueError("policy_snapshot_id must be non-empty")
        if int(minimum_policy_attempts) < 2:
            raise ValueError("minimum_policy_attempts must be at least two")

        budget = self.plan_rollout_budget(ids, max_horizon_k=max_horizon_k)
        if active_horizon_k is not None:
            active_horizon = int(active_horizon_k)
            if active_horizon <= 0 or active_horizon > int(max_horizon_k):
                raise ValueError("active_horizon_k must be positive and no larger than max_horizon_k")
            budget = FrontRESSegmentRolloutBudget(
                segment_ids=budget.segment_ids,
                trial_count=budget.trial_count,
                horizon_k=torch.full_like(budget.horizon_k, active_horizon),
                segment_state=budget.segment_state,
                reason=tuple(f"v009_global_k_{active_horizon}" for _ in budget.reason),
            )
        trial_count = torch.clamp_min(
            budget.trial_count.to(device=self.device, dtype=torch.long),
            int(minimum_policy_attempts),
        )
        expanded_ids: list[int] = []
        source_index: list[int] = []
        trial_index: list[int] = []
        horizon: list[int] = []
        for source_row, segment_id in enumerate(budget.segment_ids.tolist()):
            count = int(trial_count[source_row].item())
            horizon_value = int(budget.horizon_k[source_row].item())
            for attempt in range(count):
                expanded_ids.append(int(segment_id))
                source_index.append(source_row)
                trial_index.append(attempt)
                horizon.append(horizon_value)

        plan = FrontRESFrozenPolicyTransactionPlan(
            transaction_id=str(transaction_id),
            policy_snapshot_id=str(policy_snapshot_id),
            segment_ids=torch.tensor(expanded_ids, dtype=torch.long, device=self.device),
            source_index=torch.tensor(source_index, dtype=torch.long, device=self.device),
            trial_index=torch.tensor(trial_index, dtype=torch.long, device=self.device),
            horizon_k=torch.tensor(horizon, dtype=torch.long, device=self.device),
            trial_role=("policy",) * len(expanded_ids),
            base_segment_ids=budget.segment_ids.detach().clone(),
            base_trial_count=trial_count.detach().clone(),
            base_horizon_k=budget.horizon_k.detach().clone(),
            minimum_policy_attempts=int(minimum_policy_attempts),
        )
        plan.validate()
        return plan

    def mark_invalid(self, segment_ids: Iterable[int] | torch.Tensor, reason: str) -> None:
        ids = self._ids_tensor(segment_ids)
        self.invalid[ids] = True
        for segment_id in ids.tolist():
            self.invalid_reasons[int(segment_id)] = reason

    def stats(self) -> FrontRESSegmentSamplerStats:
        valid = ~self.invalid
        valid_count = max(1, int(valid.sum().item()))
        replay_pool = valid & (~self.solved) & (~self.hopeless) & (self.priority >= self.min_replay_score)
        review_pool = valid & self.solved
        priority_valid = self.priority[valid]
        p90 = float(torch.quantile(priority_valid, 0.9).item()) if priority_valid.numel() > 0 else 0.0
        return FrontRESSegmentSamplerStats(
            replay_pool_size=int(replay_pool.sum().item()),
            review_pool_size=int(review_pool.sum().item()),
            invalid_count=int(self.invalid.sum().item()),
            seen_count=int(self.seen.sum().item()),
            priority_mean=float(priority_valid.mean().item()) if priority_valid.numel() > 0 else 0.0,
            priority_p90=p90,
            solved_frac=float((self.solved & valid).sum().item()) / valid_count,
            hopeless_frac=float((self.hopeless & valid).sum().item()) / valid_count,
            unknown_count=self._state_count(FrontRESSegmentState.UNKNOWN, valid),
            promising_count=self._state_count(FrontRESSegmentState.PROMISING, valid),
            frontier_count=self._state_count(FrontRESSegmentState.FRONTIER, valid),
            delayed_regret_count=self._state_count(FrontRESSegmentState.DELAYED_REGRET, valid),
            solved_count=self._state_count(FrontRESSegmentState.SOLVED, valid),
            hopeless_count=self._state_count(FrontRESSegmentState.HOPELESS, valid),
            mean_trial_count=float(self.last_trial_count[valid].float().mean().item()) if valid.any() else 0.0,
            oracle_gap_mean=float(self.last_oracle_gap[valid].mean().item()) if valid.any() else 0.0,
            confidence_mean=float(self.last_confidence[valid].mean().item()) if valid.any() else 0.0,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority.cpu(),
            "staleness": self.staleness.cpu(),
            "seen": self.seen.cpu(),
            "solved": self.solved.cpu(),
            "hopeless": self.hopeless.cpu(),
            "invalid": self.invalid.cpu(),
            "segment_state": self.segment_state.cpu(),
            "evidence_count": self.evidence_count.cpu(),
            "valid_evidence_count": self.valid_evidence_count.cpu(),
            "success_count": self.success_count.cpu(),
            "fall_count": self.fall_count.cpu(),
            "best_gain": self.best_gain.cpu(),
            "best_short_gain": self.best_short_gain.cpu(),
            "best_long_gain": self.best_long_gain.cpu(),
            "last_horizon_k": self.last_horizon_k.cpu(),
            "last_trial_count": self.last_trial_count.cpu(),
            "last_policy_gain": self.last_policy_gain.cpu(),
            "last_mean_gain": self.last_mean_gain.cpu(),
            "last_success_frac": self.last_success_frac.cpu(),
            "last_fall_frac": self.last_fall_frac.cpu(),
            "last_oracle_gap": self.last_oracle_gap.cpu(),
            "last_confidence": self.last_confidence.cpu(),
            "invalid_reasons": dict(self.invalid_reasons),
            "fractions": (self.global_frac, self.replay_frac, self.review_frac),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.priority = self._load_state_tensor(state, "priority", self.priority)
        self.staleness = self._load_state_tensor(state, "staleness", self.staleness)
        self.seen = self._load_state_tensor(state, "seen", self.seen)
        self.solved = self._load_state_tensor(state, "solved", self.solved)
        self.hopeless = self._load_state_tensor(state, "hopeless", self.hopeless)
        self.invalid = self._load_state_tensor(state, "invalid", self.invalid)
        self.evidence_count = self._load_state_tensor(state, "evidence_count", self.seen.long())
        self.valid_evidence_count = self._load_state_tensor(state, "valid_evidence_count", self.valid_evidence_count)
        self.success_count = self._load_state_tensor(state, "success_count", self.success_count)
        self.fall_count = self._load_state_tensor(state, "fall_count", self.fall_count)
        self.best_gain = self._load_state_tensor(state, "best_gain", self.best_gain)
        self.best_short_gain = self._load_state_tensor(state, "best_short_gain", self.best_short_gain)
        self.best_long_gain = self._load_state_tensor(state, "best_long_gain", self.best_long_gain)
        self.last_horizon_k = self._load_state_tensor(state, "last_horizon_k", self.last_horizon_k)
        self.last_trial_count = self._load_state_tensor(state, "last_trial_count", self.last_trial_count)
        self.last_policy_gain = self._load_state_tensor(state, "last_policy_gain", self.last_policy_gain)
        self.last_mean_gain = self._load_state_tensor(state, "last_mean_gain", self.last_mean_gain)
        self.last_success_frac = self._load_state_tensor(state, "last_success_frac", self.last_success_frac)
        self.last_fall_frac = self._load_state_tensor(state, "last_fall_frac", self.last_fall_frac)
        self.last_oracle_gap = self._load_state_tensor(state, "last_oracle_gap", self.last_oracle_gap)
        self.last_confidence = self._load_state_tensor(state, "last_confidence", self.last_confidence)
        if "segment_state" in state:
            self.segment_state = self._load_state_tensor(state, "segment_state", self.segment_state)
            self._validate_segment_state()
            self._sync_terminal_flags_from_state()
        else:
            self._derive_segment_state_from_legacy_flags()
        self.invalid_reasons = {int(k): str(v) for k, v in state.get("invalid_reasons", {}).items()}

    def _choose_source(self) -> str:
        draw = float(torch.rand((), generator=self.generator, device=self.device).item())
        if draw < self.global_frac:
            return "global"
        if draw < self.global_frac + self.replay_frac:
            return "replay"
        return "review"

    def _sample_one(self, source: str, valid_ids: torch.Tensor) -> tuple[int, str]:
        if source == "replay":
            pool = self._replay_ids()
            if pool.numel() > 0:
                weights = self._sample_weights(pool)
                segment_id = int(pool[torch.multinomial(weights, 1, generator=self.generator).item()].item())
                return segment_id, "replay"
            source = "global"
        if source == "review":
            pool = torch.nonzero((~self.invalid) & self.solved, as_tuple=False).flatten()
            if pool.numel() > 0:
                weights = self._sample_weights(pool)
                segment_id = int(pool[torch.multinomial(weights, 1, generator=self.generator).item()].item())
                return segment_id, "review"
            source = "global"
        unseen = valid_ids[~self.seen[valid_ids]]
        pool = unseen if unseen.numel() > 0 else valid_ids
        index = torch.randint(0, pool.numel(), (1,), generator=self.generator, device=self.device)
        return int(pool[index].item()), source

    def _sample_weights(self, ids: torch.Tensor) -> torch.Tensor:
        weights = self.priority[ids].clamp_min(0.0) + self.staleness_weight * self.staleness[ids].clamp_min(0.0)
        if torch.sum(weights) <= 0.0:
            weights = torch.ones_like(weights)
        return weights / torch.sum(weights)

    @staticmethod
    def _bounded_horizon(preferred_horizon: int, max_horizon: int) -> int:
        target = min(int(preferred_horizon), int(max_horizon))
        if target < 8:
            return max(1, target)
        allowed = [8, 16, 32, 64]
        return max(horizon for horizon in allowed if horizon <= target)

    def _replay_ids(self) -> torch.Tensor:
        base = (~self.invalid) & (~self.solved) & (self.priority >= self.min_replay_score)
        normal = torch.nonzero(base & (~self.hopeless), as_tuple=False).flatten()
        hopeless = torch.nonzero(base & self.hopeless, as_tuple=False).flatten()
        if hopeless.numel() == 0:
            return normal
        max_hopeless = int(max(0, round(self.max_hopeless_replay_frac * max(1, normal.numel()))))
        if max_hopeless <= 0:
            return normal
        return torch.cat([normal, hopeless[:max_hopeless]], dim=0)

    def _learning_value(self, evidence: FrontRESSegmentRolloutEvidence) -> torch.Tensor:
        gain = self._active_gain(evidence)
        reset = evidence.reset_success.to(self.device).float()
        valid = evidence.valid_reward.to(self.device).float()
        contact = evidence.contact_consistency.to(self.device).float().clamp(0.0, 1.0)
        fall = evidence.fall_repaired.to(self.device).float()
        improvement = gain.clamp_min(0.0)
        return reset * valid * contact * (1.0 - fall) * improvement

    def _active_gain(self, evidence: FrontRESSegmentRolloutEvidence) -> torch.Tensor:
        """Return finite canonical Gain consumed by sampler decisions."""

        gain = evidence.gain_total
        if not isinstance(gain, torch.Tensor):
            raise ValueError("sampler priority/state requires canonical gain_total evidence")
        gain = gain.to(self.device).float().flatten()
        expected = int(evidence.segment_ids.numel())
        if int(gain.numel()) != expected:
            raise ValueError(f"gain_total must have {expected} rows, got {int(gain.numel())}")
        if not bool(torch.isfinite(gain).all().item()):
            raise ValueError("sampler priority/state requires finite gain_total evidence")
        return gain

    def _mean_by_ids(self, ids: torch.Tensor, values: torch.Tensor, unique_ids: torch.Tensor) -> torch.Tensor:
        means = []
        values = values.to(self.device).float().flatten()
        for segment_id in unique_ids.tolist():
            mask = ids == int(segment_id)
            means.append(float(values[mask].mean().item()) if bool(mask.any()) else 0.0)
        return torch.tensor(means, dtype=torch.float32, device=self.device)

    def _update_segment_state_from_trials(self, trial: FrontRESSegmentTrialEvidence) -> None:
        ids = trial.segment_ids
        self.evidence_count[ids] += trial.trial_count
        self.valid_evidence_count[ids] += trial.valid_trial_count
        self.success_count[ids] += torch.round(trial.success_frac * trial.trial_count.float()).long()
        self.fall_count[ids] += torch.round(trial.fall_frac * trial.trial_count.float()).long()
        self.last_horizon_k[ids] = trial.horizon_k
        self.last_trial_count[ids] = trial.trial_count
        self.last_policy_gain[ids] = trial.policy_gain
        self.last_mean_gain[ids] = trial.mean_gain
        self.last_success_frac[ids] = trial.success_frac
        self.last_fall_frac[ids] = trial.fall_frac
        self.last_oracle_gap[ids] = trial.oracle_gap
        self.last_confidence[ids] = trial.confidence

        neg_inf = torch.full_like(trial.best_gain, -float("inf"))
        self._scatter_max(self.best_gain, ids, torch.where(trial.valid_mask, trial.best_gain, neg_inf))
        short_horizon = trial.horizon_k <= 8
        long_horizon = trial.horizon_k >= 16
        self._scatter_max(self.best_short_gain, ids, torch.where(trial.valid_mask & short_horizon, trial.best_gain, neg_inf))
        self._scatter_max(self.best_long_gain, ids, torch.where(trial.valid_mask & long_horizon, trial.best_gain, neg_inf))

        solved = trial.valid_mask & (trial.fall_frac <= 0.0) & (trial.mean_gain.abs() < self.min_replay_score)
        short_positive = self.best_short_gain[ids] > self.min_replay_score
        long_regret = long_horizon & short_positive & ((trial.mean_gain < 0.0) | (trial.fall_frac > 0.0) | (~trial.valid_mask))
        hopeless = (~trial.valid_mask) | (
            (trial.fall_frac >= 0.5) & (trial.best_gain <= 0.0)
        )
        positive = trial.valid_mask & (trial.best_gain > self.min_replay_score)
        frontier = positive & (
            (self.evidence_count[ids] >= 2)
            | ((trial.trial_count >= 2) & (trial.success_frac < 0.75))
        )
        promising = positive | ((self.segment_state[ids] == int(FrontRESSegmentState.PROMISING)) & (~frontier))

        state = self.segment_state[ids].clone()
        state = torch.where(promising, torch.full_like(state, int(FrontRESSegmentState.PROMISING)), state)
        state = torch.where(frontier, torch.full_like(state, int(FrontRESSegmentState.FRONTIER)), state)
        state = torch.where(solved, torch.full_like(state, int(FrontRESSegmentState.SOLVED)), state)
        state = torch.where(hopeless & (~long_regret), torch.full_like(state, int(FrontRESSegmentState.HOPELESS)), state)
        state = torch.where(long_regret, torch.full_like(state, int(FrontRESSegmentState.DELAYED_REGRET)), state)
        self.segment_state[ids] = state
        self._sync_terminal_flags_for_ids(ids)

    def _update_segment_state(
        self,
        ids: torch.Tensor,
        *,
        valid: torch.Tensor,
        fall: torch.Tensor,
        gain: torch.Tensor,
        horizon: torch.Tensor,
    ) -> None:
        if horizon.numel() != ids.numel():
            raise ValueError(f"horizon_k must match segment_ids, got {horizon.numel()} and {ids.numel()}")
        ones = torch.ones_like(ids, dtype=torch.long, device=self.device)
        valid_long = valid.to(self.device).long()
        success = valid & (~fall) & (gain > self.min_replay_score)
        self.evidence_count.scatter_add_(0, ids, ones)
        self.valid_evidence_count.scatter_add_(0, ids, valid_long)
        self.success_count.scatter_add_(0, ids, success.long())
        self.fall_count.scatter_add_(0, ids, fall.long())
        self.last_horizon_k[ids] = horizon

        neg_inf = torch.full_like(gain, -float("inf"))
        self._scatter_max(self.best_gain, ids, torch.where(valid, gain, neg_inf))
        short_horizon = horizon <= 8
        long_horizon = horizon >= 16
        self._scatter_max(self.best_short_gain, ids, torch.where(valid & short_horizon, gain, neg_inf))
        self._scatter_max(self.best_long_gain, ids, torch.where(valid & long_horizon, gain, neg_inf))

        solved = valid & (~fall) & (gain.abs() < self.min_replay_score)
        short_positive = self.best_short_gain[ids] > self.min_replay_score
        long_regret = long_horizon & short_positive & ((gain < 0.0) | fall | (~valid))
        hopeless = (~valid) | (fall & (gain <= 0.0))
        positive = valid & (~fall) & (gain > self.min_replay_score)
        frontier = positive & (self.evidence_count[ids] >= 2)
        promising = positive | ((self.segment_state[ids] == int(FrontRESSegmentState.PROMISING)) & (~frontier))

        state = self.segment_state[ids].clone()
        state = torch.where(promising, torch.full_like(state, int(FrontRESSegmentState.PROMISING)), state)
        state = torch.where(frontier, torch.full_like(state, int(FrontRESSegmentState.FRONTIER)), state)
        state = torch.where(solved, torch.full_like(state, int(FrontRESSegmentState.SOLVED)), state)
        state = torch.where(hopeless & (~long_regret), torch.full_like(state, int(FrontRESSegmentState.HOPELESS)), state)
        state = torch.where(long_regret, torch.full_like(state, int(FrontRESSegmentState.DELAYED_REGRET)), state)
        self.segment_state[ids] = state
        self._sync_terminal_flags_for_ids(ids)

    def _scatter_max(self, target: torch.Tensor, ids: torch.Tensor, values: torch.Tensor) -> None:
        if hasattr(target, "scatter_reduce_"):
            target.scatter_reduce_(0, ids, values.to(target.dtype), reduce="amax", include_self=True)
            return
        for segment_id, value in zip(ids.tolist(), values.tolist()):
            if value > float(target[int(segment_id)].item()):
                target[int(segment_id)] = float(value)

    def _state_count(self, state: FrontRESSegmentState, valid: torch.Tensor) -> int:
        return int(((self.segment_state == int(state)) & valid).sum().item())

    def _load_state_tensor(self, state: dict[str, Any], name: str, default: torch.Tensor) -> torch.Tensor:
        value = state.get(name)
        if value is None:
            return default.clone()
        value = value.to(device=self.device, dtype=default.dtype).flatten()
        if value.numel() != self.num_segments:
            raise ValueError(f"{name} size mismatch: {value.numel()} != {self.num_segments}")
        return value.clone()

    def _validate_segment_state(self) -> None:
        min_state = int(self.segment_state.min().item()) if self.segment_state.numel() else 0
        max_state = int(self.segment_state.max().item()) if self.segment_state.numel() else 0
        if min_state < int(FrontRESSegmentState.UNKNOWN) or max_state > int(FrontRESSegmentState.HOPELESS):
            raise ValueError(f"segment_state contains unsupported ids: min={min_state} max={max_state}")

    def _derive_segment_state_from_legacy_flags(self) -> None:
        self.segment_state = torch.full(
            (self.num_segments,),
            int(FrontRESSegmentState.UNKNOWN),
            dtype=torch.long,
            device=self.device,
        )
        self.segment_state[self.solved] = int(FrontRESSegmentState.SOLVED)
        self.segment_state[self.hopeless] = int(FrontRESSegmentState.HOPELESS)

    def _sync_terminal_flags_for_ids(self, ids: torch.Tensor) -> None:
        self.solved[ids] = self.segment_state[ids] == int(FrontRESSegmentState.SOLVED)
        self.hopeless[ids] = self.segment_state[ids] == int(FrontRESSegmentState.HOPELESS)

    def _sync_terminal_flags_from_state(self) -> None:
        self.solved = self.segment_state == int(FrontRESSegmentState.SOLVED)
        self.hopeless = self.segment_state == int(FrontRESSegmentState.HOPELESS)

    def _valid_ids(self) -> torch.Tensor:
        return torch.nonzero(~self.invalid, as_tuple=False).flatten()

    def _ids_tensor(self, segment_ids: Iterable[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(segment_ids, torch.Tensor):
            ids = segment_ids.to(device=self.device, dtype=torch.long).flatten()
        else:
            ids = torch.tensor(list(segment_ids), dtype=torch.long, device=self.device)
        self._validate_ids(ids)
        return ids

    def _validate_ids(self, ids: torch.Tensor) -> None:
        if torch.any(ids < 0) or torch.any(ids >= self.num_segments):
            raise KeyError(f"segment ids out of range: {ids.tolist()}")
