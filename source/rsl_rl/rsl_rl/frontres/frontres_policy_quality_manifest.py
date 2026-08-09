"""Immutable comparison identity for the isolated FrontRES quality evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal


_SCHEMA_VERSION = "frontres_policy_quality_manifest_v1"
_V015_SCHEMA_VERSION = "frontres-v015-policy-quality-manifest-v1"
_V018_SCHEMA_VERSION = "frontres-v018-policy-quality-manifest-v1"
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
class FrontRESV015PolicyQualityManifest:
    """Strict held-out identity for the v015/v003 one-action-K quality route."""

    environment_revision: str
    config_revision: str
    evaluator_version: str
    items: tuple[FrontRESPolicyQualityManifestItem, ...]
    schema_version: str = _V015_SCHEMA_VERSION
    method_contract_id: str = "FRS-METHOD-v016"
    training_contract_id: str = "FRS-TRAIN-v011"
    gain_contract_id: str = "FRS-GAIN-v006"
    ppo_contract_id: str = "FRS-PPO-v004"
    future_intent_layout_version: str = "frontres-v015-future-intent-q29-v1"
    future_offsets: tuple[int, ...] = (1, 2)
    raw_observation_dim: int = 870
    combined_observation_dim: int = 928
    actor_input_dim: int = 158
    gmt_suffix_dim: int = 770
    action_kind: str = "delta_se3"
    action_dim: int = 6

    def __post_init__(self) -> None:
        exact_identity = (
            self.schema_version == _V015_SCHEMA_VERSION
            and self.method_contract_id == "FRS-METHOD-v016"
            and self.training_contract_id == "FRS-TRAIN-v011"
            and self.gain_contract_id == "FRS-GAIN-v006"
            and self.ppo_contract_id == "FRS-PPO-v004"
            and self.future_intent_layout_version == "frontres-v015-future-intent-q29-v1"
            and tuple(self.future_offsets) == (1, 2)
            and self.raw_observation_dim == 870
            and self.combined_observation_dim == 928
            and self.actor_input_dim == 158
            and self.gmt_suffix_dim == 770
            and self.action_kind == "delta_se3"
            and self.action_dim == 6
        )
        if not exact_identity:
            raise ValueError("v015 quality manifest has an incompatible schema, contract, layout, or action identity")
        object.__setattr__(self, "environment_revision", _require_text(self.environment_revision, name="environment_revision"))
        object.__setattr__(self, "config_revision", _require_text(self.config_revision, name="config_revision"))
        object.__setattr__(self, "evaluator_version", _require_text(self.evaluator_version, name="evaluator_version"))
        object.__setattr__(self, "future_offsets", tuple(int(value) for value in self.future_offsets))
        if not isinstance(self.items, tuple) or not self.items:
            raise ValueError("v015 quality manifest items must be a non-empty immutable tuple")
        if not all(isinstance(item, FrontRESPolicyQualityManifestItem) for item in self.items):
            raise ValueError("v015 quality manifest items have an invalid owner")
        ids = tuple(item.item_id for item in self.items)
        signatures = tuple(item.comparison_signature for item in self.items)
        if len(set(ids)) != len(ids) or len(set(signatures)) != len(signatures):
            raise ValueError("v015 quality manifest rejects duplicate item or comparison identity")

    @property
    def comparison_signature(self) -> str:
        return _sha256(
            {
                "schema_version": self.schema_version,
                "method_contract_id": self.method_contract_id,
                "training_contract_id": self.training_contract_id,
                "gain_contract_id": self.gain_contract_id,
                "ppo_contract_id": self.ppo_contract_id,
                "future_intent_layout_version": self.future_intent_layout_version,
                "future_offsets": list(self.future_offsets),
                "raw_observation_dim": self.raw_observation_dim,
                "combined_observation_dim": self.combined_observation_dim,
                "actor_input_dim": self.actor_input_dim,
                "gmt_suffix_dim": self.gmt_suffix_dim,
                "action_kind": self.action_kind,
                "action_dim": self.action_dim,
                "environment_revision": self.environment_revision,
                "config_revision": self.config_revision,
                "evaluator_version": self.evaluator_version,
                "items": sorted((item.comparison_payload for item in self.items), key=_canonical_json),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_contract_id": self.method_contract_id,
            "training_contract_id": self.training_contract_id,
            "gain_contract_id": self.gain_contract_id,
            "ppo_contract_id": self.ppo_contract_id,
            "future_intent_layout_version": self.future_intent_layout_version,
            "future_offsets": list(self.future_offsets),
            "raw_observation_dim": self.raw_observation_dim,
            "combined_observation_dim": self.combined_observation_dim,
            "actor_input_dim": self.actor_input_dim,
            "gmt_suffix_dim": self.gmt_suffix_dim,
            "action_kind": self.action_kind,
            "action_dim": self.action_dim,
            "environment_revision": self.environment_revision,
            "config_revision": self.config_revision,
            "evaluator_version": self.evaluator_version,
            "items": [item.to_dict() for item in self.items],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> FrontRESV015PolicyQualityManifest:
        required = frozenset(
            (
                "schema_version",
                "method_contract_id",
                "training_contract_id",
                "gain_contract_id",
                "ppo_contract_id",
                "future_intent_layout_version",
                "future_offsets",
                "raw_observation_dim",
                "combined_observation_dim",
                "actor_input_dim",
                "gmt_suffix_dim",
                "action_kind",
                "action_dim",
                "environment_revision",
                "config_revision",
                "evaluator_version",
                "items",
            )
        )
        values = _strict_fields(payload, required=required, name="v015 quality manifest")
        if not isinstance(values["items"], list) or not isinstance(values["future_offsets"], list):
            raise ValueError("v015 quality manifest items and future_offsets must be lists on disk")
        values = dict(values)
        values["items"] = tuple(FrontRESPolicyQualityManifestItem.from_dict(item) for item in values["items"])
        values["future_offsets"] = tuple(values["future_offsets"])
        return cls(**values)

    @classmethod
    def from_json(cls, text: str) -> FrontRESV015PolicyQualityManifest:
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("v015 quality manifest must be valid JSON") from exc
        return cls.from_dict(payload)


@dataclass(frozen=True)
class FrontRESV018PolicyQualityManifest:
    """Immutable EVAL-v004 held-out bank for checkpoint-v14 K16/M4 evaluation."""

    environment_revision: str
    config_revision: str
    evaluator_version: str
    items: tuple[FrontRESPolicyQualityManifestItem, ...]
    schema_version: str = _V018_SCHEMA_VERSION
    method_contract_id: str = "FRS-METHOD-v020"
    training_contract_id: str = "FRS-TRAIN-v019"
    gain_contract_id: str = "FRS-GAIN-v008"
    ppo_contract_id: str = "FRS-PPO-v008"
    evaluation_contract_id: str = "FRS-EVAL-v004"
    checkpoint_format: str = "frontres-v019-checkpoint-v14"
    hsl_checkpoint_format: str = "frontres-v017-hsl-proposal-v2"
    hsl_method_contract_id: str = "FRS-METHOD-v017"
    hsl_training_contract_id: str = "FRS-TRAIN-v014"
    future_intent_layout_version: str = "frontres-v015-future-intent-q29-v1"
    future_offsets: tuple[int, ...] = (1, 2)
    raw_observation_dim: int = 870
    combined_observation_dim: int = 928
    actor_input_dim: int = 158
    gmt_suffix_dim: int = 770
    action_kind: str = "delta_se3"
    action_semantics: str = "direct-world-full6-v1"
    action_dim: int = 6
    critic_input_dim: int = 449
    critic_value_kind: str = "state_value"
    critic_action_conditioned: bool = False
    critic_target_id: str = "segment-exact-m-mean-symlog-v1"
    critic_support_context_id: str = "action-pre-support-plan-kmax32-v1"
    critic_value_normalization_id: str = "ema-target-std-nonamplifying-v1"
    horizon_k: int = 16
    attempts_per_segment: int = 4
    segments_per_transaction: int = 2

    def __post_init__(self) -> None:
        # B1: 固定 active Contract, layout, Critic 和 K16/M4 identity, 拒绝旧 evaluator payload.
        exact_identity = (
            self.schema_version == _V018_SCHEMA_VERSION
            and self.method_contract_id == "FRS-METHOD-v020"
            and self.training_contract_id == "FRS-TRAIN-v019"
            and self.gain_contract_id == "FRS-GAIN-v008"
            and self.ppo_contract_id == "FRS-PPO-v008"
            and self.evaluation_contract_id == "FRS-EVAL-v004"
            and self.checkpoint_format == "frontres-v019-checkpoint-v14"
            and self.hsl_checkpoint_format == "frontres-v017-hsl-proposal-v2"
            and self.hsl_method_contract_id == "FRS-METHOD-v017"
            and self.hsl_training_contract_id == "FRS-TRAIN-v014"
            and self.future_intent_layout_version == "frontres-v015-future-intent-q29-v1"
            and tuple(self.future_offsets) == (1, 2)
            and self.raw_observation_dim == 870
            and self.combined_observation_dim == 928
            and self.actor_input_dim == 158
            and self.gmt_suffix_dim == 770
            and self.action_kind == "delta_se3"
            and self.action_semantics == "direct-world-full6-v1"
            and self.action_dim == 6
            and self.critic_input_dim == 449
            and self.critic_value_kind == "state_value"
            and self.critic_action_conditioned is False
            and self.critic_target_id == "segment-exact-m-mean-symlog-v1"
            and self.critic_support_context_id == "action-pre-support-plan-kmax32-v1"
            and self.critic_value_normalization_id == "ema-target-std-nonamplifying-v1"
            and self.horizon_k == 16
            and self.attempts_per_segment == 4
            and self.segments_per_transaction == 2
        )
        if not exact_identity:
            raise ValueError(
                "v018 policy-quality manifest has incompatible contract, layout, Critic, action, or K16/M4 identity"
            )
        object.__setattr__(self, "environment_revision", _require_text(self.environment_revision, name="environment_revision"))
        object.__setattr__(self, "config_revision", _require_text(self.config_revision, name="config_revision"))
        object.__setattr__(self, "evaluator_version", _require_text(self.evaluator_version, name="evaluator_version"))
        object.__setattr__(self, "future_offsets", tuple(int(value) for value in self.future_offsets))
        if not isinstance(self.items, tuple) or len(self.items) < 2 or len(self.items) % 2 != 0:
            raise ValueError("v018 policy-quality manifest requires an even non-zero number of held-out Segments")
        if not all(isinstance(item, FrontRESPolicyQualityManifestItem) for item in self.items):
            raise ValueError("v018 policy-quality manifest items have an invalid owner")
        if any(int(item.effective_horizon_k) != self.horizon_k for item in self.items):
            raise ValueError("v018 policy-quality manifest requires homogeneous K16 items")
        identities = tuple((item.motion_id, item.start_frame) for item in self.items)
        if len(set(identities)) != len(identities):
            raise ValueError("v018 policy-quality manifest requires distinct Segment identities")
        ids = tuple(item.item_id for item in self.items)
        signatures = tuple(item.comparison_signature for item in self.items)
        if len(set(ids)) != len(ids) or len(set(signatures)) != len(signatures):
            raise ValueError("v018 policy-quality manifest rejects duplicate item or comparison identity")

    @property
    def comparison_signature(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_contract_id": self.method_contract_id,
            "training_contract_id": self.training_contract_id,
            "gain_contract_id": self.gain_contract_id,
            "ppo_contract_id": self.ppo_contract_id,
            "evaluation_contract_id": self.evaluation_contract_id,
            "checkpoint_format": self.checkpoint_format,
            "hsl_checkpoint_format": self.hsl_checkpoint_format,
            "hsl_method_contract_id": self.hsl_method_contract_id,
            "hsl_training_contract_id": self.hsl_training_contract_id,
            "future_intent_layout_version": self.future_intent_layout_version,
            "future_offsets": list(self.future_offsets),
            "raw_observation_dim": self.raw_observation_dim,
            "combined_observation_dim": self.combined_observation_dim,
            "actor_input_dim": self.actor_input_dim,
            "gmt_suffix_dim": self.gmt_suffix_dim,
            "action_kind": self.action_kind,
            "action_semantics": self.action_semantics,
            "action_dim": self.action_dim,
            "critic_input_dim": self.critic_input_dim,
            "critic_value_kind": self.critic_value_kind,
            "critic_action_conditioned": self.critic_action_conditioned,
            "critic_target_id": self.critic_target_id,
            "critic_support_context_id": self.critic_support_context_id,
            "critic_value_normalization_id": self.critic_value_normalization_id,
            "horizon_k": self.horizon_k,
            "attempts_per_segment": self.attempts_per_segment,
            "segments_per_transaction": self.segments_per_transaction,
            "environment_revision": self.environment_revision,
            "config_revision": self.config_revision,
            "evaluator_version": self.evaluator_version,
            "items": [item.to_dict() for item in self.items],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> FrontRESV018PolicyQualityManifest:
        required = frozenset(
            (
                "schema_version", "method_contract_id", "training_contract_id", "gain_contract_id",
                "ppo_contract_id", "evaluation_contract_id", "checkpoint_format",
                "hsl_checkpoint_format", "hsl_method_contract_id", "hsl_training_contract_id",
                "future_intent_layout_version", "future_offsets", "raw_observation_dim",
                "combined_observation_dim", "actor_input_dim", "gmt_suffix_dim", "action_kind",
                "action_semantics", "action_dim", "critic_input_dim", "critic_value_kind",
                "critic_action_conditioned", "critic_target_id", "critic_support_context_id",
                "critic_value_normalization_id", "horizon_k", "attempts_per_segment",
                "segments_per_transaction", "environment_revision", "config_revision",
                "evaluator_version", "items",
            )
        )
        values = _strict_fields(payload, required=required, name="v018 policy-quality manifest")
        if not isinstance(values["items"], list) or not isinstance(values["future_offsets"], list):
            raise ValueError("v018 policy-quality manifest items and future_offsets must be lists on disk")
        values = dict(values)
        values["items"] = tuple(FrontRESPolicyQualityManifestItem.from_dict(item) for item in values["items"])
        values["future_offsets"] = tuple(values["future_offsets"])
        return cls(**values)

    @classmethod
    def from_json(cls, text: str) -> FrontRESV018PolicyQualityManifest:
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("v018 policy-quality manifest must be valid JSON") from exc
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
