"""Immutable v015 local-scenario aggregate and lifecycle."""

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
    clean_reference_t: torch.Tensor
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
    _clean_reference_t: torch.Tensor
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
        artifact, clean_reference, intent, continuation, expected_support, expected_support_envelope = _validate_local_scenario_payload(
            materialization.current_root_artifact_t,
            materialization.clean_reference_t,
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
                request, artifact, clean_reference, intent, continuation, expected_support, expected_support_envelope, provenance
            ),
            _current_root_artifact_t=artifact,
            _clean_reference_t=clean_reference,
            _intent_q29=intent,
            _clean_continuation=continuation,
            _expected_support=expected_support,
            _expected_support_envelope=expected_support_envelope,
            provenance=provenance,
        )

    def __post_init__(self) -> None:
        self.request.validate()
        artifact, clean_reference, intent, continuation, expected_support, expected_support_envelope = _validate_local_scenario_payload(
            self._current_root_artifact_t,
            self._clean_reference_t,
            self._intent_q29,
            self._clean_continuation,
            self._expected_support,
            self._expected_support_envelope,
            request=self.request,
        )
        provenance = _freeze_local_scenario_provenance(self.provenance)
        observed_hash = _local_scenario_hash(
            self.request, artifact, clean_reference, intent, continuation, expected_support, expected_support_envelope, provenance
        )
        if self.noisy_segment_hash != observed_hash:
            raise ValueError(
                "noisy_segment_hash does not match the immutable local scenario: "
                f"expected {observed_hash}, got {self.noisy_segment_hash}"
            )
        object.__setattr__(self, "_current_root_artifact_t", artifact)
        object.__setattr__(self, "_clean_reference_t", clean_reference)
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
    def clean_reference_t(self) -> torch.Tensor:
        return self._clean_reference_t.detach().clone()

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
            "clean_reference_t_shape": tuple(self._clean_reference_t.shape),
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
    clean_reference_t: torch.Tensor,
    intent_q29: torch.Tensor,
    clean_continuation: torch.Tensor,
    expected_support: torch.Tensor,
    expected_support_envelope: torch.Tensor,
    *,
    request: FrontRESLocalScenarioRequest,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    clean_reference = freeze("clean_reference_t", clean_reference_t)
    intent = freeze("intent_q29", intent_q29)
    continuation = freeze("clean_continuation", clean_continuation)
    support = freeze("expected_support", expected_support)
    envelope = freeze("expected_support_envelope", expected_support_envelope)
    if artifact.ndim != 1 or tuple(artifact.shape) != (7,):
        raise ValueError(f"current_root_artifact_t must have shape [7], got {tuple(artifact.shape)}")
    if clean_reference.ndim != 1 or tuple(clean_reference.shape) != (65,):
        raise ValueError(f"clean_reference_t must have shape [65], got {tuple(clean_reference.shape)}")
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
    return artifact, clean_reference, intent, continuation, support, envelope


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
        "clean_reference_t_provenance": "clean_gmt_physics_only",
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
    clean_reference_t: torch.Tensor,
    intent_q29: torch.Tensor,
    clean_continuation: torch.Tensor,
    expected_support: torch.Tensor,
    expected_support_envelope: torch.Tensor,
    provenance: Mapping[str, str | int | float | bool],
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("segment_id", int(request.segment_id)),
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
        ("clean_reference_t", clean_reference_t),
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
