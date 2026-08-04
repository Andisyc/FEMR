"""Legacy fixed-Noisy tape lifecycle retained outside the active v015 owner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

from rsl_rl.frontres.frontres_segment_planning import FrontRESSegmentSample
from rsl_rl.frontres.frontres_scenario_rows import (
    immutable_row_tensor as _immutable_row_tensor,
    scenario_row_fields as _scenario_row_fields,
)


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

