"""Read-only Clean-relative calibration primitives for the FRS-GAIN-v010 proposal."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence

from rsl_rl.frontres.frontres_hierarchical_gain_candidate import Outcome


CLEAN_CALIBRATION_FIELD_UNITS: tuple[tuple[str, str], ...] = (
    ("capture_margin", "m"),
    ("capture_margin_trend", "m_per_s"),
    ("zmp_margin", "m"),
    ("linear_momentum_error", "m_per_s"),
    ("angular_momentum_error", "rad_per_s"),
    ("support_drift", "m"),
)
_CALIBRATION_ARTIFACT_SCHEMA = "frontres-clean-calibration-artifact-v1"


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _centered_bin(value: float, resolution: float) -> int:
    # The calibrated normal deviation occupies the center bin; all comparisons
    # retain one immutable global anchor at the matched Clean reference.
    scaled = value / (2.0 * resolution)
    if -0.5 <= scaled <= 0.5:
        return 0
    if scaled > 0.5:
        return math.floor(scaled + 0.5)
    return math.ceil(scaled - 0.5)


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nearest_rank(values: Sequence[float], coverage: float, *, name: str) -> float:
    if not values:
        raise ValueError(f"{name} requires repeated pair differences")
    ordered = sorted(abs(_finite(value, name=name)) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(coverage * len(ordered)) - 1))
    result = ordered[index]
    if result <= 0.0:
        raise ValueError(f"{name} resolution must be positive")
    return result


@dataclass(frozen=True)
class CleanCalibrationObservation:
    """One window-level Clean execution used only for offline calibration."""

    domain_id: str
    scenario_id: str
    repeat_id: str
    capture_margin: float
    capture_margin_trend: float
    zmp_applicable: bool
    zmp_margin: float | None
    linear_momentum_error: float
    angular_momentum_error: float
    support_drift: float

    def validate(self) -> None:
        if not self.domain_id or not self.scenario_id or not self.repeat_id:
            raise ValueError("Clean calibration observation requires complete identity")
        _finite(self.capture_margin, name="observation capture_margin")
        _finite(self.capture_margin_trend, name="observation capture_margin_trend")
        if not isinstance(self.zmp_applicable, bool):
            raise ValueError("observation zmp_applicable must be boolean")
        if self.zmp_applicable:
            _finite(self.zmp_margin, name="observation zmp_margin")
        elif self.zmp_margin is not None:
            raise ValueError("non-applicable observation ZMP must be None")
        for name in (
            "linear_momentum_error",
            "angular_momentum_error",
            "support_drift",
        ):
            if _finite(getattr(self, name), name=f"observation {name}") < 0.0:
                raise ValueError(f"observation {name} must be non-negative")

    def canonical_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "domain_id": self.domain_id,
            "scenario_id": self.scenario_id,
            "repeat_id": self.repeat_id,
            "capture_margin": float(self.capture_margin),
            "capture_margin_trend": float(self.capture_margin_trend),
            "zmp_applicable": self.zmp_applicable,
            "zmp_margin": None if self.zmp_margin is None else float(self.zmp_margin),
            "linear_momentum_error": float(self.linear_momentum_error),
            "angular_momentum_error": float(self.angular_momentum_error),
            "support_drift": float(self.support_drift),
        }


@dataclass(frozen=True)
class CleanCalibration:
    calibration_id: str
    artifact_hash: str
    domain_id: str
    field_schema_id: str
    horizon_k: int
    timestep_seconds: float
    coverage: float
    repeated_sample_count: int
    repeated_pair_count: int
    source_scenario_ids: tuple[str, ...]
    source_observation_hash: str
    field_units: tuple[tuple[str, str], ...]
    capture_margin_resolution: float
    capture_trend_resolution: float
    zmp_margin_resolution: float
    linear_momentum_resolution: float
    angular_momentum_resolution: float
    support_drift_resolution: float
    request_hash: str = ""

    def validate(self) -> None:
        if not self.calibration_id or not self.domain_id or not self.field_schema_id:
            raise ValueError("Clean calibration requires non-empty identity")
        if (
            len(self.artifact_hash) != 64
            or any(value not in "0123456789abcdef" for value in self.artifact_hash.lower())
        ):
            raise ValueError("Clean calibration requires a SHA-256 artifact hash")
        if (
            len(self.source_observation_hash) != 64
            or any(value not in "0123456789abcdef" for value in self.source_observation_hash.lower())
        ):
            raise ValueError("Clean calibration requires a SHA-256 source observation hash")
        if isinstance(self.horizon_k, bool) or not isinstance(self.horizon_k, int) or self.horizon_k <= 0:
            raise ValueError("Clean calibration horizon_k must be a positive integer")
        if _finite(self.timestep_seconds, name="timestep_seconds") <= 0.0:
            raise ValueError("Clean calibration timestep_seconds must be positive")
        coverage = _finite(self.coverage, name="coverage")
        if not 0.0 < coverage < 1.0:
            raise ValueError("Clean calibration coverage must be in (0, 1)")
        if isinstance(self.repeated_sample_count, bool) or not isinstance(self.repeated_sample_count, int):
            raise ValueError("Clean calibration repeated_sample_count must be an integer")
        if self.repeated_sample_count < 2:
            raise ValueError("Clean calibration requires at least two repeated samples")
        if isinstance(self.repeated_pair_count, bool) or not isinstance(self.repeated_pair_count, int):
            raise ValueError("Clean calibration repeated_pair_count must be an integer")
        if self.repeated_pair_count < 1:
            raise ValueError("Clean calibration requires at least one repeated pair")
        if (
            not isinstance(self.source_scenario_ids, tuple)
            or not self.source_scenario_ids
            or any(not isinstance(value, str) or not value for value in self.source_scenario_ids)
            or tuple(sorted(set(self.source_scenario_ids))) != self.source_scenario_ids
        ):
            raise ValueError("Clean calibration source Scenario identities must be sorted and unique")
        if self.field_units != CLEAN_CALIBRATION_FIELD_UNITS:
            raise ValueError("Clean calibration field units do not match the field schema")
        for name in (
            "capture_margin_resolution",
            "capture_trend_resolution",
            "zmp_margin_resolution",
            "linear_momentum_resolution",
            "angular_momentum_resolution",
            "support_drift_resolution",
        ):
            if _finite(getattr(self, name), name=name) <= 0.0:
                raise ValueError(f"{name} resolution must be positive")
        if self.artifact_hash != _sha256(self.hash_payload()):
            raise ValueError("Clean calibration artifact hash mismatch")
        if self.request_hash and not _is_sha256(self.request_hash):
            raise ValueError("Clean calibration request_hash must be SHA-256 when present")

    def hash_payload(self) -> dict[str, object]:
        return {
            "artifact_schema": _CALIBRATION_ARTIFACT_SCHEMA,
            "calibration_id": self.calibration_id,
            "domain_id": self.domain_id,
            "field_schema_id": self.field_schema_id,
            "horizon_k": self.horizon_k,
            "timestep_seconds": self.timestep_seconds,
            "coverage": self.coverage,
            "repeated_sample_count": self.repeated_sample_count,
            "repeated_pair_count": self.repeated_pair_count,
            "source_scenario_ids": self.source_scenario_ids,
            "source_observation_hash": self.source_observation_hash,
            "field_units": self.field_units,
            "capture_margin_resolution": self.capture_margin_resolution,
            "capture_trend_resolution": self.capture_trend_resolution,
            "zmp_margin_resolution": self.zmp_margin_resolution,
            "linear_momentum_resolution": self.linear_momentum_resolution,
            "angular_momentum_resolution": self.angular_momentum_resolution,
            "support_drift_resolution": self.support_drift_resolution,
            "request_hash": self.request_hash,
        }


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


@dataclass(frozen=True)
class CleanCalibrationCollectionIdentity:
    """Same-Scenario identity shared by every repeated Clean window."""

    domain_id: str
    scenario_id: str
    segment_identity: str
    clean_artifact_hash: str
    cache_artifact_hash: str
    expected_support_hash: str
    gmt_checkpoint_hash: str
    gmt_normalizer_hash: str
    field_schema_id: str
    horizon_k: int
    timestep_seconds: float
    seed_protocol_id: str
    preroll_steps: int = 0

    def validate(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.domain_id,
                self.scenario_id,
                self.segment_identity,
                self.field_schema_id,
                self.seed_protocol_id,
            )
        ):
            raise ValueError("Clean collection identity requires complete non-hash identity")
        for name in (
            "clean_artifact_hash",
            "cache_artifact_hash",
            "expected_support_hash",
            "gmt_checkpoint_hash",
            "gmt_normalizer_hash",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"Clean collection identity {name} must be SHA-256")
        if isinstance(self.horizon_k, bool) or not isinstance(self.horizon_k, int) or self.horizon_k <= 0:
            raise ValueError("Clean collection identity horizon_k must be positive")
        if (
            isinstance(self.preroll_steps, bool)
            or not isinstance(self.preroll_steps, int)
            or self.preroll_steps < 0
        ):
            raise ValueError("Clean collection identity preroll_steps must be a non-negative integer")
        if _finite(self.timestep_seconds, name="Clean collection identity timestep_seconds") <= 0.0:
            raise ValueError("Clean collection identity timestep_seconds must be positive")

    def canonical_payload(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "identity_schema": (
                "frontres-readonly-clean-collection-identity-v1"
                if self.preroll_steps == 0
                else "frontres-readonly-clean-collection-identity-v2"
            ),
            "domain_id": self.domain_id,
            "scenario_id": self.scenario_id,
            "segment_identity": self.segment_identity,
            "clean_artifact_hash": self.clean_artifact_hash,
            "cache_artifact_hash": self.cache_artifact_hash,
            "expected_support_hash": self.expected_support_hash,
            "gmt_checkpoint_hash": self.gmt_checkpoint_hash,
            "gmt_normalizer_hash": self.gmt_normalizer_hash,
            "field_schema_id": self.field_schema_id,
            "horizon_k": self.horizon_k,
            "timestep_seconds": float(self.timestep_seconds),
            "seed_protocol_id": self.seed_protocol_id,
        }
        if self.preroll_steps > 0:
            payload["preroll_steps"] = self.preroll_steps
        return payload


@dataclass(frozen=True)
class CleanCalibrationRepeatSpec:
    repeat_id: str
    seed: int

    def validate(self) -> None:
        if not isinstance(self.repeat_id, str) or not self.repeat_id:
            raise ValueError("Clean repeat requires a non-empty repeat_id")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("Clean repeat seed must be an integer")


@dataclass(frozen=True)
class CleanHardEventEvidence:
    """Raw hard-event labels retained alongside each Clean window."""

    survival_ok: bool
    survival_failure_duration: float
    expected_support_no_load: float
    unplanned_support_switch: float
    illegal_contact_duration: float
    valid_step_count: int
    zmp_applicable_step_count: int

    def validate(self) -> None:
        if not isinstance(self.survival_ok, bool):
            raise ValueError("Clean hard-event survival_ok must be boolean")
        for name in (
            "survival_failure_duration",
            "expected_support_no_load",
            "unplanned_support_switch",
            "illegal_contact_duration",
        ):
            value = _finite(getattr(self, name), name=f"Clean hard-event {name}")
            if value < 0.0:
                raise ValueError(f"Clean hard-event {name} must be non-negative")
        for name in ("valid_step_count", "zmp_applicable_step_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Clean hard-event {name} must be a non-negative integer")
        if self.zmp_applicable_step_count > self.valid_step_count:
            raise ValueError("Clean hard-event ZMP applicability exceeds valid steps")

    def canonical_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "survival_ok": self.survival_ok,
            "survival_failure_duration": float(self.survival_failure_duration),
            "expected_support_no_load": float(self.expected_support_no_load),
            "unplanned_support_switch": float(self.unplanned_support_switch),
            "illegal_contact_duration": float(self.illegal_contact_duration),
            "valid_step_count": int(self.valid_step_count),
            "zmp_applicable_step_count": int(self.zmp_applicable_step_count),
        }


@dataclass(frozen=True)
class CleanCalibrationCollectionRequest:
    """Typed request consumed by a future official read-only gateway."""

    calibration_id: str
    identity: CleanCalibrationCollectionIdentity
    repeats: tuple[CleanCalibrationRepeatSpec, ...]
    coverage: float

    def validate(self) -> None:
        if not isinstance(self.calibration_id, str) or not self.calibration_id:
            raise ValueError("Clean collection request requires calibration_id")
        self.identity.validate()
        if (
            not isinstance(self.repeats, tuple)
            or len(self.repeats) < 2
            or any(not isinstance(value, CleanCalibrationRepeatSpec) for value in self.repeats)
            or tuple(value.repeat_id for value in self.repeats) != tuple(
                sorted({value.repeat_id for value in self.repeats})
            )
            or len({value.repeat_id for value in self.repeats}) != len(self.repeats)
        ):
            raise ValueError("Clean collection request requires sorted unique repeats with count >= 2")
        for repeat in self.repeats:
            repeat.validate()
        coverage = _finite(self.coverage, name="Clean collection coverage")
        if not 0.0 < coverage < 1.0:
            raise ValueError("Clean collection coverage must be in (0, 1)")

    def canonical_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "request_schema": "frontres-readonly-clean-collection-request-v2",
            "calibration_id": self.calibration_id,
            "identity": self.identity.canonical_payload(),
            "repeats": tuple({"repeat_id": item.repeat_id, "seed": item.seed} for item in self.repeats),
            "coverage": float(self.coverage),
        }


@dataclass(frozen=True)
class ReadOnlyCleanWindow:
    """One completed Clean window returned by a collection gateway."""

    observation: CleanCalibrationObservation
    identity: CleanCalibrationCollectionIdentity
    repeat_seed: int
    repeat_seed_hash: str
    training_state_hash: str
    rng_restore_hash: str
    collector_id: str
    collector_version: str
    hard_events: CleanHardEventEvidence

    def validate(self, request: CleanCalibrationCollectionRequest, repeat: CleanCalibrationRepeatSpec) -> None:
        self.observation.validate()
        if not isinstance(self.hard_events, CleanHardEventEvidence):
            raise TypeError("Clean window requires typed hard-event evidence")
        self.hard_events.validate()
        self.identity.validate()
        if self.identity != request.identity:
            raise ValueError("Clean window identity differs from the request")
        if self.observation.domain_id != request.identity.domain_id or self.observation.scenario_id != request.identity.scenario_id:
            raise ValueError("Clean window observation has mismatched domain/Scenario identity")
        if self.observation.repeat_id != repeat.repeat_id or self.repeat_seed != repeat.seed:
            raise ValueError("Clean window repeat identity or seed differs from the request")
        expected_seed_hash = _sha256(
            {
                "seed_protocol_id": request.identity.seed_protocol_id,
                "repeat_id": repeat.repeat_id,
                "seed": repeat.seed,
            }
        )
        if self.repeat_seed_hash != expected_seed_hash:
            raise ValueError("Clean window repeat seed provenance differs from the request")
        for name, value in (
            ("training_state_hash", self.training_state_hash),
            ("rng_restore_hash", self.rng_restore_hash),
        ):
            if not _is_sha256(value):
                raise ValueError(f"Clean window {name} must be SHA-256")
        if not self.collector_id or not self.collector_version:
            raise ValueError("Clean window requires collector identity/version")


@dataclass(frozen=True)
class ReadOnlyCleanCollection:
    """Complete gateway output; partial collections are not admissible."""

    windows: tuple[ReadOnlyCleanWindow, ...]
    training_state_before_hash: str
    training_state_after_hash: str
    rng_state_before_hash: str
    rng_state_after_hash: str
    closed_repeat_ids: tuple[str, ...]
    collector_id: str
    collector_version: str

    def validate(self, request: CleanCalibrationCollectionRequest) -> None:
        request.validate()
        expected_repeats = tuple(value.repeat_id for value in request.repeats)
        if (
            not isinstance(self.windows, tuple)
            or tuple(window.observation.repeat_id for window in self.windows) != expected_repeats
            or self.closed_repeat_ids != expected_repeats
            or len(self.windows) != len(request.repeats)
            or any(not isinstance(window, ReadOnlyCleanWindow) for window in self.windows)
        ):
            raise ValueError("Clean collection must contain exactly one complete window per requested repeat")
        for window, repeat in zip(self.windows, request.repeats, strict=True):
            window.validate(request, repeat)
        for name, value in (
            ("training_state_before_hash", self.training_state_before_hash),
            ("training_state_after_hash", self.training_state_after_hash),
            ("rng_state_before_hash", self.rng_state_before_hash),
            ("rng_state_after_hash", self.rng_state_after_hash),
        ):
            if not _is_sha256(value):
                raise ValueError(f"Clean collection {name} must be SHA-256")
        if self.training_state_before_hash != self.training_state_after_hash:
            raise ValueError("read-only Clean collection changed training state")
        if self.rng_state_before_hash != self.rng_state_after_hash:
            raise ValueError("read-only Clean collection failed to restore RNG state")
        if self.collector_id != self.windows[0].collector_id or self.collector_version != self.windows[0].collector_version:
            raise ValueError("Clean collection collector identity is not constant")
        if any(
            window.training_state_hash != self.training_state_before_hash
            or window.rng_restore_hash != self.rng_state_before_hash
            for window in self.windows
        ):
            raise ValueError("Clean collection repeat changed training/RNG identity")


@dataclass(frozen=True)
class ReadOnlyCleanCollectionReceipt:
    """Immutable adapter receipt; it is not simulator or formal-route proof."""

    request_hash: str
    collection_hash: str
    training_state_before_hash: str
    training_state_after_hash: str
    rng_state_before_hash: str
    rng_state_after_hash: str
    repeat_ids: tuple[str, ...]
    collected_count: int
    collector_id: str
    collector_version: str
    calibration: CleanCalibration
    path_class: str = "ALTERNATE-PATH/READ-ONLY-COLLECTION-ADAPTER"

    def validate(self) -> None:
        for name, value in (
            ("request_hash", self.request_hash),
            ("collection_hash", self.collection_hash),
            ("training_state_before_hash", self.training_state_before_hash),
            ("training_state_after_hash", self.training_state_after_hash),
            ("rng_state_before_hash", self.rng_state_before_hash),
            ("rng_state_after_hash", self.rng_state_after_hash),
        ):
            if not _is_sha256(value):
                raise ValueError(f"Clean collection receipt {name} must be SHA-256")
        if (
            tuple(sorted(set(self.repeat_ids))) != self.repeat_ids
            or isinstance(self.collected_count, bool)
            or not isinstance(self.collected_count, int)
            or self.collected_count != len(self.repeat_ids)
            or self.collected_count < 2
            or not self.collector_id
            or not self.collector_version
        ):
            raise ValueError("Clean collection receipt has invalid repeat or collector identity")
        if self.training_state_before_hash != self.training_state_after_hash:
            raise ValueError("read-only Clean collection changed training state")
        if self.rng_state_before_hash != self.rng_state_after_hash:
            raise ValueError("read-only Clean collection failed to restore RNG state")
        if self.path_class != "ALTERNATE-PATH/READ-ONLY-COLLECTION-ADAPTER":
            raise ValueError("Clean collection receipt has an invalid path classification")
        self.calibration.validate()

    def validate_for_request(self, request: CleanCalibrationCollectionRequest) -> None:
        """Re-check receipt/calibration binding at the next consumer boundary."""

        request.validate()
        self.validate()
        expected_request_hash = _sha256(request.canonical_payload())
        if self.request_hash != expected_request_hash or self.calibration.request_hash != expected_request_hash:
            raise ValueError("Clean collection receipt/calibration request identity mismatch")
        identity = request.identity
        if (
            self.calibration.calibration_id != request.calibration_id
            or self.calibration.domain_id != identity.domain_id
            or self.calibration.field_schema_id != identity.field_schema_id
            or self.calibration.horizon_k != identity.horizon_k
            or self.calibration.timestep_seconds != identity.timestep_seconds
            or self.calibration.coverage != request.coverage
        ):
            raise ValueError("Clean calibration metadata differs from its collection request")


def adapt_read_only_clean_collection(
    request: CleanCalibrationCollectionRequest,
    collection: ReadOnlyCleanCollection,
) -> ReadOnlyCleanCollectionReceipt:
    """Validate a gateway result and build the immutable calibration artifact.

    This is intentionally a pure adapter boundary.  A future official gateway
    must own reset/materialization, the existing read-only scope, full RNG
    restoration, and cleanup before handing its typed result here.  No generic
    callback, simulator object, training runner, optimizer, Replay, or
    checkpoint is accepted by this domain owner.
    """

    collection.validate(request)
    identity = request.identity
    calibration = build_clean_calibration(
        tuple(window.observation for window in collection.windows),
        calibration_id=request.calibration_id,
        domain_id=identity.domain_id,
        field_schema_id=identity.field_schema_id,
        horizon_k=identity.horizon_k,
        timestep_seconds=identity.timestep_seconds,
        coverage=request.coverage,
    )
    request_hash = _sha256(request.canonical_payload())
    calibration = replace(calibration, request_hash=request_hash)
    calibration = replace(calibration, artifact_hash=_sha256(calibration.hash_payload()))
    calibration.validate()
    collection_hash = _sha256(
        {
            "collection_schema": "frontres-readonly-clean-collection-v2",
            "request_hash": request_hash,
            "windows": tuple(
                {
                    "repeat_id": window.observation.repeat_id,
                    "observation": window.observation.canonical_payload(),
                    "repeat_seed": window.repeat_seed,
                    "repeat_seed_hash": window.repeat_seed_hash,
                    "training_state_hash": window.training_state_hash,
                    "rng_restore_hash": window.rng_restore_hash,
                    "collector_id": window.collector_id,
                    "collector_version": window.collector_version,
                    "hard_events": window.hard_events.canonical_payload(),
                }
                for window in collection.windows
            ),
            "closed_repeat_ids": collection.closed_repeat_ids,
            "training_state_before_hash": collection.training_state_before_hash,
            "training_state_after_hash": collection.training_state_after_hash,
            "rng_state_before_hash": collection.rng_state_before_hash,
            "rng_state_after_hash": collection.rng_state_after_hash,
        }
    )
    receipt = ReadOnlyCleanCollectionReceipt(
        request_hash=request_hash,
        collection_hash=collection_hash,
        training_state_before_hash=collection.training_state_before_hash,
        training_state_after_hash=collection.training_state_after_hash,
        rng_state_before_hash=collection.rng_state_before_hash,
        rng_state_after_hash=collection.rng_state_after_hash,
        repeat_ids=tuple(window.observation.repeat_id for window in collection.windows),
        collected_count=len(collection.windows),
        collector_id=collection.collector_id,
        collector_version=collection.collector_version,
        calibration=calibration,
    )
    receipt.validate_for_request(request)
    return receipt


def build_clean_calibration(
    observations: Sequence[CleanCalibrationObservation],
    *,
    calibration_id: str,
    domain_id: str,
    field_schema_id: str,
    horizon_k: int,
    timestep_seconds: float,
    coverage: float,
) -> CleanCalibration:
    """Build one immutable calibration from repeated same-Scenario windows."""

    if not calibration_id or not domain_id or not field_schema_id:
        raise ValueError("Clean calibration producer requires complete identity")
    if isinstance(horizon_k, bool) or not isinstance(horizon_k, int) or horizon_k <= 0:
        raise ValueError("Clean calibration horizon_k must be a positive integer")
    if _finite(timestep_seconds, name="timestep_seconds") <= 0.0:
        raise ValueError("Clean calibration timestep_seconds must be positive")
    coverage_value = _finite(coverage, name="coverage")
    if not 0.0 < coverage_value < 1.0:
        raise ValueError("Clean calibration coverage must be in (0, 1)")
    if not isinstance(observations, Sequence) or len(observations) < 2:
        raise ValueError("Clean calibration requires at least two observations")

    canonical_rows: list[dict[str, object]] = []
    grouped: dict[str, list[CleanCalibrationObservation]] = defaultdict(list)
    seen_repeats: set[tuple[str, str]] = set()
    for observation in observations:
        if not isinstance(observation, CleanCalibrationObservation):
            raise ValueError("Clean calibration observations have an invalid type")
        observation.validate()
        if observation.domain_id != domain_id:
            raise ValueError("Clean calibration observation domain mismatch")
        repeat_key = (observation.scenario_id, observation.repeat_id)
        if repeat_key in seen_repeats:
            raise ValueError("Clean calibration repeat identity must be unique per Scenario")
        seen_repeats.add(repeat_key)
        grouped[observation.scenario_id].append(observation)
        canonical_rows.append(observation.canonical_payload())

    field_differences: dict[str, list[float]] = {
        "capture_margin": [],
        "capture_margin_trend": [],
        "zmp_margin": [],
        "linear_momentum_error": [],
        "angular_momentum_error": [],
        "support_drift": [],
    }
    repeated_pair_count = 0
    for scenario_id, rows in grouped.items():
        if len(rows) < 2:
            raise ValueError(f"Clean calibration Scenario {scenario_id!r} requires repeated observations")
        applicability = {row.zmp_applicable for row in rows}
        if len(applicability) != 1:
            raise ValueError("Clean calibration ZMP applicability must match within a Scenario")
        ordered_rows = sorted(rows, key=lambda row: row.repeat_id)
        for left_index, left in enumerate(ordered_rows):
            for right in ordered_rows[left_index + 1 :]:
                repeated_pair_count += 1
                for name in (
                    "capture_margin",
                    "capture_margin_trend",
                    "linear_momentum_error",
                    "angular_momentum_error",
                    "support_drift",
                ):
                    field_differences[name].append(
                        float(getattr(left, name)) - float(getattr(right, name))
                    )
                if left.zmp_applicable:
                    assert left.zmp_margin is not None and right.zmp_margin is not None
                    field_differences["zmp_margin"].append(left.zmp_margin - right.zmp_margin)

    resolutions = {
        name: _nearest_rank(values, coverage_value, name=name)
        for name, values in field_differences.items()
    }
    sorted_rows = sorted(
        canonical_rows,
        key=lambda row: (str(row["scenario_id"]), str(row["repeat_id"])),
    )
    source_observation_hash = _sha256(
        {
            "domain_id": domain_id,
            "field_schema_id": field_schema_id,
            "horizon_k": horizon_k,
            "timestep_seconds": timestep_seconds,
            "field_units": CLEAN_CALIBRATION_FIELD_UNITS,
            "observations": sorted_rows,
        }
    )
    common = {
        "calibration_id": calibration_id,
        "domain_id": domain_id,
        "field_schema_id": field_schema_id,
        "horizon_k": horizon_k,
        "timestep_seconds": float(timestep_seconds),
        "coverage": coverage_value,
        "repeated_sample_count": len(observations),
        "repeated_pair_count": repeated_pair_count,
        "source_scenario_ids": tuple(sorted(grouped)),
        "source_observation_hash": source_observation_hash,
        "field_units": CLEAN_CALIBRATION_FIELD_UNITS,
        "capture_margin_resolution": resolutions["capture_margin"],
        "capture_trend_resolution": resolutions["capture_margin_trend"],
        "zmp_margin_resolution": resolutions["zmp_margin"],
        "linear_momentum_resolution": resolutions["linear_momentum_error"],
        "angular_momentum_resolution": resolutions["angular_momentum_error"],
        "support_drift_resolution": resolutions["support_drift"],
    }
    provisional = CleanCalibration(artifact_hash="0" * 64, **common)
    result = CleanCalibration(artifact_hash=_sha256(provisional.hash_payload()), **common)
    result.validate()
    return result


@dataclass(frozen=True)
class CleanReference:
    domain_id: str
    scenario_id: str
    capture_margin: float
    capture_margin_trend: float
    zmp_applicable: bool
    zmp_margin: float | None

    def validate(self) -> None:
        if not self.domain_id or not self.scenario_id:
            raise ValueError("Clean reference requires domain and Scenario identity")
        _finite(self.capture_margin, name="Clean capture_margin")
        _finite(self.capture_margin_trend, name="Clean capture_margin_trend")
        if not isinstance(self.zmp_applicable, bool):
            raise ValueError("Clean zmp_applicable must be boolean")
        if self.zmp_applicable:
            _finite(self.zmp_margin, name="Clean zmp_margin")
        elif self.zmp_margin is not None:
            raise ValueError("non-applicable Clean ZMP must be None")


@dataclass(frozen=True)
class CleanRelativeResult:
    recovery_bins: tuple[int, ...]
    absolute_physics_valid: bool
    inside_clean_domain: bool


def apply_clean_relative_calibration(
    outcome: Outcome,
    clean: CleanReference,
    calibration: CleanCalibration,
    *,
    outcome_scenario_id: str,
) -> CleanRelativeResult:
    """Apply an immutable calibration without changing it or ranking outcomes."""

    calibration.validate()
    clean.validate()
    if clean.domain_id != calibration.domain_id:
        raise ValueError("Clean calibration/reference domain mismatch")
    if not outcome_scenario_id or outcome_scenario_id != clean.scenario_id:
        raise ValueError("Clean/Repair Scenario identity mismatch")
    if not isinstance(outcome.zmp_applicable, bool):
        raise ValueError("Repair zmp_applicable must be boolean")
    if outcome.zmp_applicable != clean.zmp_applicable:
        raise ValueError("Clean/Repair ZMP applicability mismatch")

    capture = _finite(outcome.capture_margin, name="Repair capture_margin")
    trend = _finite(outcome.capture_margin_trend, name="Repair capture_margin_trend")
    linear = _finite(outcome.linear_momentum_error, name="Repair linear_momentum_error")
    angular = _finite(outcome.angular_momentum_error, name="Repair angular_momentum_error")
    drift = _finite(outcome.support_drift, name="Repair support_drift")
    if min(linear, angular, drift) < 0.0:
        raise ValueError("Clean-relative error evidence must be non-negative")

    zmp_bins: tuple[int, ...] = ()
    zmp_inside = True
    zmp_absolute_valid = True
    if outcome.zmp_applicable:
        repair_zmp = _finite(outcome.zmp_margin, name="Repair zmp_margin")
        clean_zmp = _finite(clean.zmp_margin, name="Clean zmp_margin")
        zmp_bins = (
            _centered_bin(
                repair_zmp - clean_zmp,
                calibration.zmp_margin_resolution,
            ),
        )
        zmp_absolute_valid = repair_zmp >= 0.0
        zmp_inside = repair_zmp >= clean_zmp - calibration.zmp_margin_resolution
    elif outcome.zmp_margin is not None:
        raise ValueError("non-applicable Repair ZMP must be None")

    severe = tuple(
        _finite(getattr(outcome, name), name=name)
        for name in (
            "expected_support_no_load",
            "unplanned_support_switch",
            "illegal_contact_duration",
        )
    )
    if min(severe) < 0.0:
        raise ValueError("severe Physics evidence must be non-negative")
    if not isinstance(outcome.survival_ok, bool):
        raise ValueError("survival_ok must be boolean")

    recovery_bins = (
        _centered_bin(capture - clean.capture_margin, calibration.capture_margin_resolution),
        _centered_bin(trend - clean.capture_margin_trend, calibration.capture_trend_resolution),
        *zmp_bins,
        _centered_bin(-linear, calibration.linear_momentum_resolution),
        _centered_bin(-angular, calibration.angular_momentum_resolution),
        _centered_bin(-drift, calibration.support_drift_resolution),
    )
    absolute_physics_valid = (
        outcome.survival_ok
        and not any(value > 0.0 for value in severe)
        and capture >= 0.0
        and zmp_absolute_valid
    )
    inside_clean_domain = (
        absolute_physics_valid
        and capture >= clean.capture_margin - calibration.capture_margin_resolution
        and trend >= clean.capture_margin_trend - calibration.capture_trend_resolution
        and zmp_inside
        and linear <= calibration.linear_momentum_resolution
        and angular <= calibration.angular_momentum_resolution
        and drift <= calibration.support_drift_resolution
    )
    return CleanRelativeResult(
        recovery_bins=recovery_bins,
        absolute_physics_valid=absolute_physics_valid,
        inside_clean_domain=inside_clean_domain,
    )


__all__ = (
    "CLEAN_CALIBRATION_FIELD_UNITS",
    "CleanCalibration",
    "CleanCalibrationCollectionIdentity",
    "CleanCalibrationCollectionRequest",
    "CleanCalibrationObservation",
    "CleanHardEventEvidence",
    "CleanCalibrationRepeatSpec",
    "CleanReference",
    "CleanRelativeResult",
    "ReadOnlyCleanCollection",
    "ReadOnlyCleanCollectionReceipt",
    "ReadOnlyCleanWindow",
    "adapt_read_only_clean_collection",
    "apply_clean_relative_calibration",
    "build_clean_calibration",
)
