"""Immutable comparison identity for the isolated FrontRES quality evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal


_SCHEMA_VERSION = "frontres_policy_quality_manifest_v1"
_ROUTES = frozenset(("zero", "hsl", "policy"))
_Scalar = bool | int | float | str


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    value = _require_nonnegative_int(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_hash(value: object, *, name: str) -> str:
    value = _require_text(value, name=name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _strict_fields(payload: object, *, required: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be an object")
    keys = frozenset(payload)
    missing = sorted(required - keys)
    unexpected = sorted(keys - required)
    if missing:
        raise ValueError(f"{name} missing fields: {missing}")
    if unexpected:
        raise ValueError(f"{name} has unexpected fields: {unexpected}")
    return payload


def _canonical_parameters(value: object) -> tuple[tuple[str, _Scalar], ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("perturbation_parameters must be an immutable key/value sequence")
    result: list[tuple[str, _Scalar]] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("each perturbation parameter must contain exactly key and value")
        key = _require_text(entry[0], name="perturbation parameter key")
        scalar = entry[1]
        if isinstance(scalar, float) and not math.isfinite(scalar):
            raise ValueError(f"perturbation parameter {key!r} must be finite")
        if not isinstance(scalar, (bool, int, float, str)):
            raise ValueError(f"perturbation parameter {key!r} must be a JSON scalar")
        if key in seen:
            raise ValueError(f"duplicate perturbation parameter: {key}")
        seen.add(key)
        result.append((key, scalar))
    return tuple(sorted(result, key=lambda pair: pair[0]))


@dataclass(frozen=True)
class FrontRESPolicyQualityManifestItem:
    """One checkpoint-independent counterfactual comparison question."""

    item_id: str
    motion_id: str
    start_frame: int
    perturbation_family: str
    perturbation_parameters: tuple[tuple[str, _Scalar], ...]
    effective_horizon_k: int
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _require_text(self.item_id, name="item_id"))
        object.__setattr__(self, "motion_id", _require_text(self.motion_id, name="motion_id"))
        object.__setattr__(self, "start_frame", _require_nonnegative_int(self.start_frame, name="start_frame"))
        object.__setattr__(
            self,
            "perturbation_family",
            _require_text(self.perturbation_family, name="perturbation_family"),
        )
        object.__setattr__(self, "perturbation_parameters", _canonical_parameters(self.perturbation_parameters))
        object.__setattr__(
            self,
            "effective_horizon_k",
            _require_positive_int(self.effective_horizon_k, name="effective_horizon_k"),
        )
        object.__setattr__(self, "seed", _require_nonnegative_int(self.seed, name="seed"))

    @property
    def comparison_payload(self) -> dict[str, Any]:
        return {
            "motion_id": self.motion_id,
            "start_frame": self.start_frame,
            "perturbation_family": self.perturbation_family,
            "perturbation_parameters": [[key, value] for key, value in self.perturbation_parameters],
            "effective_horizon_k": self.effective_horizon_k,
            "seed": self.seed,
        }

    @property
    def comparison_signature(self) -> str:
        return _sha256(self.comparison_payload)

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, **self.comparison_payload}

    @classmethod
    def from_dict(cls, payload: object) -> FrontRESPolicyQualityManifestItem:
        values = _strict_fields(
            payload,
            required=frozenset(
                (
                    "item_id",
                    "motion_id",
                    "start_frame",
                    "perturbation_family",
                    "perturbation_parameters",
                    "effective_horizon_k",
                    "seed",
                )
            ),
            name="manifest item",
        )
        return cls(**values)


@dataclass(frozen=True)
class FrontRESPolicyQualityManifest:
    """Canonical checkpoint-independent bank shared by every evaluated route."""

    environment_revision: str
    config_revision: str
    evaluator_version: str
    items: tuple[FrontRESPolicyQualityManifestItem, ...]
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "environment_revision", _require_text(self.environment_revision, name="environment_revision")
        )
        object.__setattr__(self, "config_revision", _require_text(self.config_revision, name="config_revision"))
        object.__setattr__(self, "evaluator_version", _require_text(self.evaluator_version, name="evaluator_version"))
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.items, tuple) or not self.items:
            raise ValueError("items must be a non-empty immutable tuple")
        if not all(isinstance(item, FrontRESPolicyQualityManifestItem) for item in self.items):
            raise ValueError("items must contain FrontRESPolicyQualityManifestItem values")
        ids = [item.item_id for item in self.items]
        signatures = [item.comparison_signature for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate item_id in policy-quality manifest")
        if len(set(signatures)) != len(signatures):
            raise ValueError("duplicate comparison identity in policy-quality manifest")

    @property
    def comparison_signature(self) -> str:
        # QUALITY-ID-01: 在 route/checkpoint metadata 进入前冻结评估问题身份.
        # Result: Q-E1 OFFLINE PASS; identity 对行顺序稳定, 对控制变量敏感,
        # checkpoint/sampler state 被 schema 排除. 动态 state equality 留给 Q1-B.
        return _sha256(
            {
                "schema_version": self.schema_version,
                "environment_revision": self.environment_revision,
                "config_revision": self.config_revision,
                "evaluator_version": self.evaluator_version,
                "items": sorted((item.comparison_payload for item in self.items), key=_canonical_json),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment_revision": self.environment_revision,
            "config_revision": self.config_revision,
            "evaluator_version": self.evaluator_version,
            "items": [item.to_dict() for item in self.items],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> FrontRESPolicyQualityManifest:
        values = _strict_fields(
            payload,
            required=frozenset(
                ("schema_version", "environment_revision", "config_revision", "evaluator_version", "items")
            ),
            name="manifest",
        )
        if not isinstance(values["items"], list):
            raise ValueError("manifest items must be a list on disk")
        return cls(
            schema_version=values["schema_version"],
            environment_revision=values["environment_revision"],
            config_revision=values["config_revision"],
            evaluator_version=values["evaluator_version"],
            items=tuple(FrontRESPolicyQualityManifestItem.from_dict(item) for item in values["items"]),
        )

    @classmethod
    def from_json(cls, text: str) -> FrontRESPolicyQualityManifest:
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("manifest must be valid JSON") from exc
        return cls.from_dict(payload)


@dataclass(frozen=True)
class FrontRESPolicyQualityStateIdentity:
    """Dynamic scoring-start identity, populated by the Q1-B state owner."""

    comparison_signature: str
    initial_state_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "comparison_signature", _require_hash(self.comparison_signature, name="comparison_signature")
        )
        object.__setattr__(self, "initial_state_hash", _require_hash(self.initial_state_hash, name="initial_state_hash"))


@dataclass(frozen=True)
class FrontRESPolicyQualityRouteIdentity:
    """Route/checkpoint metadata that cannot alter the comparison question."""

    route: Literal["zero", "hsl", "policy"]
    checkpoint_identity: str
    state: FrontRESPolicyQualityStateIdentity

    def __post_init__(self) -> None:
        if self.route not in _ROUTES:
            raise ValueError(f"route must be one of {sorted(_ROUTES)}, got {self.route!r}")
        object.__setattr__(
            self, "checkpoint_identity", _require_text(self.checkpoint_identity, name="checkpoint_identity")
        )
        if not isinstance(self.state, FrontRESPolicyQualityStateIdentity):
            raise ValueError("state must be FrontRESPolicyQualityStateIdentity")

    @property
    def comparison_signature(self) -> str:
        return self.state.comparison_signature

    @property
    def route_signature(self) -> str:
        return _sha256(
            {
                "comparison_signature": self.comparison_signature,
                "initial_state_hash": self.state.initial_state_hash,
                "route": self.route,
                "checkpoint_identity": self.checkpoint_identity,
            }
        )
