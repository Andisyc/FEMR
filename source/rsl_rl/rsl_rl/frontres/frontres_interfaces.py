"""Public FEMR/FrontRES interfaces for the frozen MOSAIC host.

This module contains contracts only. It owns no simulator, policy, Gain, PPO,
checkpoint, or logging implementation. MOSAIC objects are consumed through
narrow adapters so the formal FrontRES route does not depend on their concrete
classes or private layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from rsl_rl.frontres.frontres_return_utility import (
    FRONTRES_RETURN_UTILITY_ID,
    FRONTRES_RETURN_UTILITY_SCALE,
)

FRONTRES_METHOD_CONTRACT_ID = "FRS-METHOD-v024"
FRONTRES_GAIN_CONTRACT_ID = "FRS-GAIN-v008"
FRONTRES_OPTIMIZATION_CONTRACT_ID = "FRS-PPO-v011"
FRONTRES_TRAINING_CONTRACT_ID = "FRS-TRAIN-v023"
FRONTRES_SCALAR_TARGET_ID = "symmetric-log-recovery-aware-utility-v1"
FRONTRES_PHYSICS_SCHEMA_ID = "clean-anchored-contact-zmp-survival-v1"
FRONTRES_GROUPED_SCHEMA_ID = "grouped-all-attempt-scalar-v1"
FRONTRES_CHECKPOINT_FORMAT = "frontres-v023-checkpoint-v18"
FRONTRES_DR_CURRICULUM_SCHEMA_ID = "nested-k-dr-four-class-v1"
FRONTRES_CRITIC_VALUE_KIND = "state_value"
FRONTRES_CRITIC_INPUT_DIM = 449
FRONTRES_CRITIC_ACTION_CONDITIONED = False
FRONTRES_CRITIC_TARGET_ID = "scenario-compatible-robust-mean-symlog-v1"
FRONTRES_CRITIC_SUPPORT_CONTEXT_ID = "action-pre-support-plan-kmax32-v1"
FRONTRES_GRADIENT_CLIP_ID = "separate-actor-critic-v1"
FRONTRES_GRADIENT_CLIP_MAX_NORM = 0.5
FRONTRES_VALUE_NORMALIZATION_ID = "ema-target-std-nonamplifying-v1"
FRONTRES_VALUE_NORMALIZER_DECAY = 0.9
FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR = 1.0


class FrontRESActiveRunMode(str, Enum):
    """One explicit FrontRES route selected by the frozen MOSAIC entrypoint."""

    DISABLED = "disabled"
    UNCONFIGURED = "unconfigured"
    HSL = "hsl"
    FORMAL_TRAIN = "formal_train"
    LOCAL_SENTINEL = "local_sentinel"
    LIVE_PROBE = "live_probe"
    POLICY_QUALITY = "policy_quality"
    DEPLOYMENT = "deployment"


@dataclass(frozen=True)
class FrontRESActiveContractIdentity:
    """Version identity shared by transactions, telemetry, and checkpoints."""

    method: str = FRONTRES_METHOD_CONTRACT_ID
    gain: str = FRONTRES_GAIN_CONTRACT_ID
    optimization: str = FRONTRES_OPTIMIZATION_CONTRACT_ID
    training: str = FRONTRES_TRAINING_CONTRACT_ID
    scalar_target: str = FRONTRES_SCALAR_TARGET_ID
    physics_schema: str = FRONTRES_PHYSICS_SCHEMA_ID
    grouped_schema: str = FRONTRES_GROUPED_SCHEMA_ID
    checkpoint_format: str = FRONTRES_CHECKPOINT_FORMAT
    critic_value_kind: str = FRONTRES_CRITIC_VALUE_KIND
    critic_input_dim: int = FRONTRES_CRITIC_INPUT_DIM
    critic_action_conditioned: bool = FRONTRES_CRITIC_ACTION_CONDITIONED
    critic_target: str = FRONTRES_CRITIC_TARGET_ID
    return_utility: str = FRONTRES_RETURN_UTILITY_ID
    return_utility_scale: float = FRONTRES_RETURN_UTILITY_SCALE
    critic_support_context: str = FRONTRES_CRITIC_SUPPORT_CONTEXT_ID
    gradient_clip: str = FRONTRES_GRADIENT_CLIP_ID

    def validate(self) -> None:
        expected = FrontRESActiveContractIdentity()
        if self != expected:
            raise ValueError(f"FrontRES v017 contract identity drifted: expected={expected!r} actual={self!r}")


@dataclass(frozen=True)
class FrontRESActiveObservationAuthority:
    """Shape authority for the deployable FEMR prefix and frozen GMT suffix."""

    environment_dim: int = 870
    future_intent_dim: int = 58
    combined_dim: int = 928
    frontres_prefix_dim: int = 158
    gmt_suffix_dim: int = 770
    action_dim: int = 6

    def validate(self) -> None:
        if (
            self.environment_dim != 870
            or self.future_intent_dim != 58
            or self.combined_dim != self.environment_dim + self.future_intent_dim
            or self.frontres_prefix_dim != 158
            or self.gmt_suffix_dim != 770
            or self.frontres_prefix_dim + self.gmt_suffix_dim != self.combined_dim
            or self.action_dim != 6
        ):
            raise ValueError(f"FrontRES v017 observation authority drifted: {self!r}")


@dataclass(frozen=True)
class FrontRESActiveTransactionShape:
    """Exact B8 x M4 policy/role layout for one sealed transaction."""

    active_k: int
    active_m: int
    selected_segment_count: int
    policy_row_count: int
    role_row_count: int

    def validate(self) -> None:
        expected_m_by_k = {8: 4, 16: 4, 32: 4}
        if isinstance(self.active_k, bool) or int(self.active_k) not in expected_m_by_k:
            raise ValueError(f"FrontRES transaction has an invalid TRAIN-v013 K: {self.active_k}")
        if isinstance(self.active_m, bool) or int(self.active_m) != expected_m_by_k[int(self.active_k)]:
            raise ValueError(
                "FrontRES transaction active M does not match TRAIN-v013: "
                f"K={self.active_k} expected_M={expected_m_by_k[int(self.active_k)]} actual_M={self.active_m}"
            )
        if int(self.selected_segment_count) != 8:
            raise ValueError("FrontRES transaction requires exactly eight Scenario sources")
        expected_policy_rows = 8 * int(self.active_m)
        if int(self.policy_row_count) != expected_policy_rows:
            raise ValueError(
                "FrontRES transaction policy rows must equal eight Scenario sources x active M: "
                f"expected={expected_policy_rows} actual={self.policy_row_count}"
            )
        if int(self.role_row_count) != 2 * expected_policy_rows:
            raise ValueError(
                "FrontRES transaction role rows must equal Repair/Noisy x policy rows: "
                f"expected={2 * expected_policy_rows} actual={self.role_row_count}"
            )


@dataclass(frozen=True)
class FrontRESActiveTransactionRequestView:
    """Immutable consumer view of one sealed formal Stage-3 request."""

    identity: FrontRESActiveContractIdentity
    transaction_id: str
    policy_snapshot_id: str
    shape: FrontRESActiveTransactionShape
    curriculum_fingerprint: str
    k_stage_index: int
    k_stage_iteration: int
    training_iteration: int
    warmup_phase_name: str
    warmup_actor_loss_weight: float
    warmup_actor_learning_rate: float
    dr_stage_fingerprint: str
    dr_progress: float
    d_cap: float

    def validate(self) -> None:
        self.identity.validate()
        self.shape.validate()
        if not self.transaction_id or not self.policy_snapshot_id:
            raise ValueError("FrontRES request requires transaction and frozen-policy identity")
        fingerprint = self.curriculum_fingerprint
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise ValueError("FrontRES request requires a lowercase SHA-256 curriculum fingerprint")
        for name, value in (
            ("k_stage_index", self.k_stage_index),
            ("k_stage_iteration", self.k_stage_iteration),
            ("training_iteration", self.training_iteration),
        ):
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"FrontRES request {name} must be a nonnegative integer")
        if self.warmup_phase_name not in {"low_dr_joint_init", "coupled_ramp", "joint"}:
            raise ValueError("FrontRES request has an invalid TRAIN-v021 phase")
        if float(self.warmup_actor_loss_weight) != 1.0:
            raise ValueError("FrontRES request actor-loss weight must remain one")
        if not math.isfinite(float(self.warmup_actor_learning_rate)) or not 3.0e-7 <= float(
            self.warmup_actor_learning_rate
        ) <= 1.0e-6:
            raise ValueError("FrontRES request Actor LR must be finite in [3e-7,1e-6]")
        if len(self.dr_stage_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.dr_stage_fingerprint
        ):
            raise ValueError("FrontRES request requires a sealed TRAIN-v013 DR-stage fingerprint")
        if not math.isfinite(float(self.dr_progress)) or not 0.0 <= float(self.dr_progress) <= 1.0:
            raise ValueError("FrontRES request DR progress must be finite in [0,1]")
        if not math.isfinite(float(self.d_cap)) or float(self.d_cap) <= 0.0 or float(self.d_cap) > 2.381:
            raise ValueError("FrontRES request d_cap must be finite in (0,2.381]")


@runtime_checkable
class FrontRESStage3Request(Protocol):
    """Consumer-shaped request port required by the Stage-3 engine."""

    def frontres_stage3_request_view(self) -> FrontRESActiveTransactionRequestView: ...


@dataclass(frozen=True)
class FrontRESActiveCommittedUpdateView:
    """Typed, read-only projection of the existing formal update result."""

    transaction_id: str
    policy_snapshot_id: str
    segment_count: int
    policy_attempt_count: int
    valid_row_count: int
    optimizer_step_before: int
    optimizer_step_after: int
    optimizer_step_delta: int
    update_invocation_count: int

    @classmethod
    def from_result(cls, result: object) -> "FrontRESActiveCommittedUpdateView":
        names = (
            "transaction_id",
            "policy_snapshot_id",
            "segment_count",
            "policy_attempt_count",
            "valid_row_count",
            "optimizer_step_before",
            "optimizer_step_after",
            "optimizer_step_delta",
            "update_invocation_count",
        )
        missing = tuple(name for name in names if not hasattr(result, name))
        if missing:
            raise TypeError(f"formal FrontRES update result is missing typed fields: {missing}")
        view = cls(
            transaction_id=str(getattr(result, "transaction_id")),
            policy_snapshot_id=str(getattr(result, "policy_snapshot_id")),
            segment_count=int(getattr(result, "segment_count")),
            policy_attempt_count=int(getattr(result, "policy_attempt_count")),
            valid_row_count=int(getattr(result, "valid_row_count")),
            optimizer_step_before=int(getattr(result, "optimizer_step_before")),
            optimizer_step_after=int(getattr(result, "optimizer_step_after")),
            optimizer_step_delta=int(getattr(result, "optimizer_step_delta")),
            update_invocation_count=int(getattr(result, "update_invocation_count")),
        )
        view.validate()
        return view

    def validate(self, *, expected_request: FrontRESActiveTransactionRequestView | None = None) -> None:
        if not self.transaction_id or not self.policy_snapshot_id:
            raise ValueError("committed FrontRES update requires transaction and frozen-policy identity")
        if self.segment_count != 8:
            raise ValueError("committed FrontRES update requires exactly eight Scenario sources")
        if (
            self.policy_attempt_count < 4
            or self.valid_row_count <= 0
            or self.valid_row_count > self.policy_attempt_count
        ):
            raise ValueError("committed FrontRES update has an invalid attempt/valid-row count")
        if self.optimizer_step_delta != 1 or self.update_invocation_count != 1:
            raise ValueError("committed FrontRES update requires exactly one optimizer invocation")
        if self.optimizer_step_after - self.optimizer_step_before != 1:
            raise ValueError("committed FrontRES optimizer receipt is inconsistent")
        if expected_request is not None:
            expected_request.validate()
            if (
                self.transaction_id != expected_request.transaction_id
                or self.policy_snapshot_id != expected_request.policy_snapshot_id
                or self.segment_count != expected_request.shape.selected_segment_count
                or self.policy_attempt_count != expected_request.shape.policy_row_count
            ):
                raise ValueError("committed FrontRES receipt does not match its sealed request identity and shape")


@dataclass(frozen=True)
class FrontRESActiveTelemetryView:
    """Required final-consumer identity and exact-one fields from live telemetry."""

    identity: FrontRESActiveContractIdentity
    transaction_id: str
    shape: FrontRESActiveTransactionShape
    optimizer_step_delta: int
    update_count: int
    actor_learning_rate: float
    critic_learning_rate: float
    actor_observation_dim: int
    gmt_observation_dim: int
    gradient_clip_max_norm: float
    actor_gradient_post_clip_norm: float
    critic_gradient_post_clip_norm: float
    critic_value_normalization_id: str
    critic_value_scale: float
    critic_value_normalizer_decay: float
    critic_value_normalizer_scale_floor: float
    critic_value_normalizer_update_count_before: int
    critic_value_normalizer_update_count_after: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FrontRESActiveTelemetryView":
        required = {
            "method_contract_id",
            "gain_contract_id",
            "optimization_contract_id",
            "training_contract_id",
            "scalar_target_id",
            "physics_schema_id",
            "grouped_schema_id",
            "checkpoint_format",
            "critic_value_kind",
            "critic_input_dim",
            "critic_action_conditioned",
            "critic_target_id",
            "critic_support_context_id",
            "gradient_clip_identity",
            "actor_observation_dim",
            "gmt_observation_dim",
            "gradient_clip_max_norm",
            "actor_gradient_post_clip_norm",
            "critic_gradient_post_clip_norm",
            "critic_value_normalization_id",
            "critic_value_scale",
            "critic_value_normalizer_decay",
            "critic_value_normalizer_scale_floor",
            "critic_value_normalizer_update_count_before",
            "critic_value_normalizer_update_count_after",
            "transaction_id",
            "active_k",
            "active_m",
            "selected_segment_count",
            "policy_row_count",
            "role_row_count",
            "optimizer_step_delta",
            "update_count",
            "actor_learning_rate",
            "critic_learning_rate",
        }
        missing = tuple(sorted(required.difference(values)))
        if missing:
            raise ValueError(f"FrontRES telemetry is missing required final fields: {missing}")
        view = cls(
            identity=FrontRESActiveContractIdentity(
                method=str(values["method_contract_id"]),
                gain=str(values["gain_contract_id"]),
                optimization=str(values["optimization_contract_id"]),
                training=str(values["training_contract_id"]),
                scalar_target=str(values["scalar_target_id"]),
                physics_schema=str(values["physics_schema_id"]),
                grouped_schema=str(values["grouped_schema_id"]),
                checkpoint_format=str(values["checkpoint_format"]),
                critic_value_kind=str(values["critic_value_kind"]),
                critic_input_dim=int(values["critic_input_dim"]),
                critic_action_conditioned=bool(values["critic_action_conditioned"]),
                critic_target=str(values["critic_target_id"]),
                critic_support_context=str(values["critic_support_context_id"]),
                gradient_clip=str(values["gradient_clip_identity"]),
            ),
            transaction_id=str(values["transaction_id"]),
            shape=FrontRESActiveTransactionShape(
                active_k=int(values["active_k"]),
                active_m=int(values["active_m"]),
                selected_segment_count=int(values["selected_segment_count"]),
                policy_row_count=int(values["policy_row_count"]),
                role_row_count=int(values["role_row_count"]),
            ),
            optimizer_step_delta=int(values["optimizer_step_delta"]),
            update_count=int(values["update_count"]),
            actor_learning_rate=float(values["actor_learning_rate"]),
            critic_learning_rate=float(values["critic_learning_rate"]),
            actor_observation_dim=int(values["actor_observation_dim"]),
            gmt_observation_dim=int(values["gmt_observation_dim"]),
            gradient_clip_max_norm=float(values["gradient_clip_max_norm"]),
            actor_gradient_post_clip_norm=float(values["actor_gradient_post_clip_norm"]),
            critic_gradient_post_clip_norm=float(values["critic_gradient_post_clip_norm"]),
            critic_value_normalization_id=str(values["critic_value_normalization_id"]),
            critic_value_scale=float(values["critic_value_scale"]),
            critic_value_normalizer_decay=float(values["critic_value_normalizer_decay"]),
            critic_value_normalizer_scale_floor=float(values["critic_value_normalizer_scale_floor"]),
            critic_value_normalizer_update_count_before=int(
                values["critic_value_normalizer_update_count_before"]
            ),
            critic_value_normalizer_update_count_after=int(
                values["critic_value_normalizer_update_count_after"]
            ),
        )
        view.validate()
        return view

    def validate(self) -> None:
        self.identity.validate()
        self.shape.validate()
        if not self.transaction_id:
            raise ValueError("FrontRES telemetry requires transaction identity")
        if self.optimizer_step_delta != 1 or self.update_count != 1:
            raise ValueError("FrontRES telemetry requires exact-one update identity")
        if not 3.0e-7 <= self.actor_learning_rate <= 1.0e-6 or self.critic_learning_rate != 1.0e-5:
            raise ValueError("FRS-TRAIN-v023 telemetry requires Actor LR in [3e-7,1e-6] and Critic LR=1e-5")
        if self.actor_observation_dim != 158 or self.gmt_observation_dim != 770:
            raise ValueError("FRS-TRAIN-v023 telemetry requires Actor/GMT dimensions 158/770")
        gradient_values = (
            self.gradient_clip_max_norm,
            self.actor_gradient_post_clip_norm,
            self.critic_gradient_post_clip_norm,
        )
        if (
            self.gradient_clip_max_norm != FRONTRES_GRADIENT_CLIP_MAX_NORM
            or not all(math.isfinite(value) and value >= 0.0 for value in gradient_values)
            or self.actor_gradient_post_clip_norm > self.gradient_clip_max_norm + 1.0e-6
            or self.critic_gradient_post_clip_norm > self.gradient_clip_max_norm + 1.0e-6
        ):
            raise ValueError("FRS-PPO-v011 telemetry has invalid separate gradient clipping facts")
        if (
            self.critic_value_normalization_id != FRONTRES_VALUE_NORMALIZATION_ID
            or self.critic_value_normalizer_decay != FRONTRES_VALUE_NORMALIZER_DECAY
            or self.critic_value_normalizer_scale_floor != FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR
            or not math.isfinite(self.critic_value_scale)
            or self.critic_value_scale < 1.0
            or self.critic_value_normalizer_update_count_after
            != self.critic_value_normalizer_update_count_before + 1
        ):
            raise ValueError("FRS-PPO-v009 telemetry has invalid Critic value-normalizer facts")


class FrontRESTransactionLifecyclePort(Protocol):
    """Open, collect, abort, and close one sealed transaction lifecycle."""

    def open_transaction_barrier(self) -> None: ...

    def build_training_request(self, *, init_at_random_ep_len: bool) -> FrontRESStage3Request: ...

    def abort_training_collection(self) -> None: ...

    def close_training_request(self) -> None: ...

    def is_rejected_evidence(self, error: BaseException) -> bool: ...


class FrontRESExactUpdatePort(Protocol):
    """Expose optimizer identity and commit one complete grouped request."""

    def optimizer_step_count(self) -> int: ...

    def commit_transaction(self, request: FrontRESStage3Request) -> object: ...


class FrontRESModePort(Protocol):
    """Resolve one explicit route without leaking config boolean combinations."""

    def formal_transaction_enabled(self) -> bool: ...

    def formal_training_enabled(self) -> bool: ...

    def sentinel_only(self) -> bool: ...


@runtime_checkable
class FrontRESStage3Backend(
    FrontRESTransactionLifecyclePort,
    FrontRESExactUpdatePort,
    FrontRESModePort,
    Protocol,
):
    """Complete narrow backend required by the FEMR Stage-3 engine."""


FrontRESTransactionProvider = Callable[[], FrontRESStage3Request]


__all__ = [
    "FRONTRES_CHECKPOINT_FORMAT",
    "FRONTRES_CRITIC_ACTION_CONDITIONED",
    "FRONTRES_CRITIC_INPUT_DIM",
    "FRONTRES_CRITIC_TARGET_ID",
    "FRONTRES_CRITIC_SUPPORT_CONTEXT_ID",
    "FRONTRES_CRITIC_VALUE_KIND",
    "FRONTRES_GRADIENT_CLIP_ID",
    "FRONTRES_GRADIENT_CLIP_MAX_NORM",
    "FRONTRES_GROUPED_SCHEMA_ID",
    "FRONTRES_GAIN_CONTRACT_ID",
    "FRONTRES_METHOD_CONTRACT_ID",
    "FRONTRES_OPTIMIZATION_CONTRACT_ID",
    "FRONTRES_PHYSICS_SCHEMA_ID",
    "FRONTRES_SCALAR_TARGET_ID",
    "FRONTRES_TRAINING_CONTRACT_ID",
    "FrontRESExactUpdatePort",
    "FrontRESModePort",
    "FrontRESStage3Request",
    "FrontRESStage3Backend",
    "FrontRESTransactionLifecyclePort",
    "FrontRESTransactionProvider",
    "FrontRESActiveCommittedUpdateView",
    "FrontRESActiveContractIdentity",
    "FrontRESActiveObservationAuthority",
    "FrontRESActiveRunMode",
    "FrontRESActiveTelemetryView",
    "FrontRESActiveTransactionRequestView",
    "FrontRESActiveTransactionShape",
]
