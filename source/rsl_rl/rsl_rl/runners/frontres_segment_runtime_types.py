"""Immutable records and transaction lifecycle state for FrontRES Segment execution."""





from __future__ import annotations





from collections.abc import Mapping
from dataclasses import dataclass


import hashlib
import math


from typing import TYPE_CHECKING, Any, Callable


import torch

from rsl_rl.runners.frontres_stage3_engine import frontres_stage3_transaction_aggregate


from rsl_rl.frontres.frontres_interfaces import FrontRESActiveContractIdentity, FrontRESActiveTransactionRequestView, FrontRESActiveTransactionShape


from rsl_rl.frontres.frontres_local_evaluation import (
    FrontRESV017LocalEvaluationReport,
)


from rsl_rl.runners.frontres_segment_transaction import FrontRESFormalTransactionPlan

if TYPE_CHECKING:
    from rsl_rl.frontres.frontres_segment_evidence_legacy import (
        FrontRESV015GainReturnEvidence,
        FrontRESV015OneActionKEvidence,
    )





@dataclass
class FrontRESSegmentLiveObservations:
    obs: torch.Tensor
    privileged_obs: torch.Tensor
    teacher_obs: torch.Tensor
    ref_vel_estimator_obs: torch.Tensor | None


@dataclass(frozen=True)
class FrontRESV015GainConsumerEvidence:
    """Candidate-only v003 consumer chain, 与 formal PPO 和 sampler state 隔离."""

    one_action: FrontRESV015OneActionKEvidence
    return_evidence: FrontRESV015GainReturnEvidence
    priority_evidence: Any

    def validate(self) -> None:
        self.one_action.validate()
        self.return_evidence.validate()
        validate_priority = getattr(self.priority_evidence, "validate", None)
        if not callable(validate_priority):
            raise TypeError("v015 Gain consumer chain requires validated priority evidence")
        validate_priority()
        same_gain = (
            tuple(self.return_evidence.gain_total.shape) == tuple(self.priority_evidence.gain_total.shape)
            and torch.equal(
                torch.isnan(self.return_evidence.gain_total),
                torch.isnan(self.priority_evidence.gain_total),
            )
            and torch.equal(
                torch.nan_to_num(self.return_evidence.gain_total, nan=0.0),
                torch.nan_to_num(self.priority_evidence.gain_total, nan=0.0),
            )
        )
        if (
            not same_gain
            or self.return_evidence.scenario_ids != self.priority_evidence.scenario_ids
            or self.return_evidence.noisy_segment_hashes != self.priority_evidence.noisy_segment_hashes
        ):
            raise ValueError("v015 Gain consumer chain lost the shared Gain decomposition or scenario identity")
        repair_rows = self.one_action.policy_row_indices.detach().to(dtype=torch.long)
        if repair_rows.ndim != 1 or int(repair_rows.numel()) != int(self.return_evidence.policy_actions.shape[0]):
            raise ValueError("v015 Gain consumer chain has misaligned Repair policy rows")
        for name in (
            "policy_observations",
            "policy_actions",
            "policy_log_probs",
            "policy_values",
            "policy_means",
            "policy_sigmas",
        ):
            expected = getattr(self.one_action, name).detach()
            actual = getattr(self.return_evidence, name).detach()
            if not torch.equal(actual.to(device="cpu"), expected.to(device="cpu")):
                raise ValueError(f"v015 Gain consumer chain lost the one-action {name} tuple")
        expected_horizon = self.one_action.horizon_k.index_select(0, repair_rows)
        expected_survival = self.one_action.survival_steps.index_select(0, repair_rows).detach().float()
        expected_steps = expected_survival.to(dtype=torch.long)
        if (
            not torch.equal(expected_survival, expected_steps.to(dtype=expected_survival.dtype))
            or not torch.equal(expected_horizon.to(device="cpu", dtype=torch.long), self.return_evidence.horizon_k.detach().to(device="cpu", dtype=torch.long))
            or not torch.equal(
                expected_steps.to(device="cpu"),
                self.return_evidence.evidence_valid_step_count.detach().to(device="cpu", dtype=torch.long),
            )
            or tuple(self.one_action.scenario_ids[int(row)] for row in repair_rows.tolist()) != self.return_evidence.scenario_ids
            or tuple(self.one_action.noisy_segment_hashes[int(row)] for row in repair_rows.tolist())
            != self.return_evidence.noisy_segment_hashes
            or tuple(self.one_action.x_t_identities[int(row)] for row in repair_rows.tolist())
            != self.return_evidence.x_t_identities
        ):
            raise ValueError("v015 Gain consumer chain lost the one-action local scenario or K evidence identity")


@dataclass(frozen=True)
class FrontRESFormalTransactionRequest:
    """Concrete sealed request behind the public Stage-3 request port.

    Status: active for deterministic fixtures, the bounded sentinel and ordinary
    Stage-3 training. The engine sees only `frontres_stage3_request_view()`;
    tensors and simulator-specific collaborators remain inside this backend.
    """

    plan: FrontRESFormalTransactionPlan
    candidate_batches: tuple[Any, ...]
    diagnostic_reports: tuple[FrontRESV017LocalEvaluationReport, ...]
    curriculum_fingerprint: str
    k_stage_index: int
    active_k: int
    active_m: int
    k_stage_iteration: int
    training_iteration: int
    warmup_phase_name: str
    warmup_actor_loss_weight: float
    dr_stage_fingerprint: str
    dr_progress: float
    d_cap: float
    dr_class_by_segment: tuple[str, ...]
    dr_strength_by_segment: tuple[float, ...]
    policy_evaluator: Any | None = None
    # Optional compatibility cross-check only. It cannot replace the critic
    # rows carried and reordered by the sealed candidate transaction.
    privileged_observations: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_batches", tuple(self.candidate_batches))
        object.__setattr__(self, "diagnostic_reports", tuple(self.diagnostic_reports))
        if not isinstance(self.plan, FrontRESFormalTransactionPlan):
            raise TypeError("v015 formal transaction request requires FrontRESFormalTransactionPlan")
        self.plan.validate()
        if not self.candidate_batches:
            raise ValueError("v015 formal transaction request requires candidate batches")
        if len(self.diagnostic_reports) != len(self.candidate_batches):
            raise ValueError("v015 formal transaction requires one immutable diagnostic projection per candidate batch")
        for candidate_batch, report in zip(self.candidate_batches, self.diagnostic_reports, strict=True):
            if not isinstance(report, FrontRESV017LocalEvaluationReport):
                raise TypeError("v015 formal transaction diagnostic projection has an invalid owner")
            report.validate()
            metadata = getattr(candidate_batch, "transaction_metadata", None)
            if (
                report.transaction_id != self.plan.transaction_id
                or metadata is None
                or report.policy_row_count != int(candidate_batch.observations.shape[0])
                or report.scenario_ids != tuple(metadata.scenario_ids)
                or report.noisy_segment_hashes != tuple(metadata.noisy_segment_hashes)
            ):
                raise ValueError("v015 formal transaction diagnostic projection lost transaction or scenario identity")
        if self.privileged_observations is not None and not isinstance(self.privileged_observations, torch.Tensor):
            raise TypeError("v015 formal transaction privileged_observations must be a tensor or None")
        if not isinstance(self.curriculum_fingerprint, str) or len(self.curriculum_fingerprint) != 64:
            raise ValueError("v015 formal transaction requires a SHA-256 curriculum fingerprint")
        if any(
            isinstance(value, bool) or int(value) < 0
            for value in (self.k_stage_index, self.k_stage_iteration, self.training_iteration)
        ):
            raise ValueError("v015 formal transaction K-stage indexes must be nonnegative integers")
        if isinstance(self.active_k, bool) or int(self.active_k) <= 0:
            raise ValueError("v015 formal transaction active_k must be positive")
        if isinstance(self.active_m, bool) or int(self.active_m) < 2:
            raise ValueError("FRS-TRAIN-v019 formal transaction active_m must be at least two")
        if self.plan.active_m != int(self.active_m) or self.plan.selected_segment_count != 2:
            raise ValueError("FRS-TRAIN-v019 formal transaction plan does not match exact two-Segment x M identity")
        if self.warmup_phase_name not in {"critic_only", "actor_ramp", "joint"}:
            raise ValueError("v015 formal transaction has an invalid warmup phase")
        if not 0.0 <= float(self.warmup_actor_loss_weight) <= 1.0:
            raise ValueError("v015 formal transaction actor loss weight must be in [0,1]")
        if len(self.dr_stage_fingerprint) != 64:
            raise ValueError("FRS-TRAIN-v019 formal transaction requires a sealed DR-stage fingerprint")
        if not 0.0 <= float(self.dr_progress) <= 1.0 or not 0.0 < float(self.d_cap) <= 2.381:
            raise ValueError("FRS-TRAIN-v019 formal transaction has invalid DR progress or d_cap")
        if len(self.dr_class_by_segment) != 2 or len(self.dr_strength_by_segment) != 2:
            raise ValueError("FRS-TRAIN-v019 transaction requires one sealed DR class/strength per Segment")
        if any(name not in {"easy", "medium", "hard", "broken"} for name in self.dr_class_by_segment):
            raise ValueError("FRS-TRAIN-v019 transaction has an invalid DR class")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in self.dr_strength_by_segment):
            raise ValueError("FRS-TRAIN-v019 transaction has an invalid DR strength")
        horizon = self.plan.horizon_k.detach().to(device="cpu", dtype=torch.long)
        if not bool((horizon == int(self.active_k)).all().item()):
            raise ValueError("v015 formal transaction rejects mixed-K or active-K-mismatched plan rows")

    def frontres_stage3_request_view(self) -> FrontRESActiveTransactionRequestView:
        """Return the immutable identity/shape projection consumed by the engine."""

        view = FrontRESActiveTransactionRequestView(
            identity=FrontRESActiveContractIdentity(),
            transaction_id=str(self.plan.transaction_id),
            policy_snapshot_id=str(self.plan.policy_snapshot_id),
            shape=FrontRESActiveTransactionShape(
                active_k=int(self.active_k),
                active_m=int(self.active_m),
                selected_segment_count=int(self.plan.selected_segment_count),
                policy_row_count=int(self.plan.batch_size),
                role_row_count=int(2 * self.plan.batch_size),
            ),
            curriculum_fingerprint=str(self.curriculum_fingerprint),
            k_stage_index=int(self.k_stage_index),
            k_stage_iteration=int(self.k_stage_iteration),
            training_iteration=int(self.training_iteration),
            warmup_phase_name=str(self.warmup_phase_name),
            warmup_actor_loss_weight=float(self.warmup_actor_loss_weight),
            dr_stage_fingerprint=str(self.dr_stage_fingerprint),
            dr_progress=float(self.dr_progress),
            d_cap=float(self.d_cap),
        )
        view.validate()
        return view


_V015_CHECKPOINT_TRANSACTION_STATE_ATTR = "_frontres_checkpoint_transaction_state"


def _v015_checkpoint_plan_hash(plan: FrontRESFormalTransactionPlan, *, scenario_only: bool) -> str:
    """只 hash immutable identity field, 不把 raw scenario reference 写入 receipt."""

    digest = hashlib.sha256()
    values = (
        plan.scenario_ids,
        plan.noisy_segment_hashes,
        plan.x_t_identities,
        tuple(int(value) for value in plan.horizon_k.tolist()),
    )
    if not scenario_only:
        values = (
            plan.transaction_id,
            plan.policy_snapshot_id,
            plan.intent_q29_provenance,
            plan.intent_q29_source,
            plan.motion_ids,
            tuple(int(value) for value in plan.start_frames.tolist()),
            tuple(int(value) for value in plan.segment_ids.tolist()),
            tuple(int(value) for value in plan.source_index.tolist()),
            tuple(int(value) for value in plan.trial_index.tolist()),
            *values,
        )
    for value in values:
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def open_frontres_checkpoint_transaction_barrier(runner: Any) -> None:
    """在 injected provider 收集 candidate evidence 前打开 save barrier."""

    aggregate = frontres_stage3_transaction_aggregate(runner)
    if aggregate.execution_phase == "collecting" and aggregate.as_dict() == {
        "state": "collecting",
        "phase": "provider",
    }:
        return
    aggregate.begin_collection()


def _bind_frontres_checkpoint_transaction_plan(
    runner: Any,
    plan: FrontRESFormalTransactionPlan,
) -> None:
    """在 collection 仍禁止 checkpoint 时发布 immutable transaction identity."""

    plan.validate()
    aggregate = frontres_stage3_transaction_aggregate(runner)
    aggregate.bind_plan(
        {
            "transaction_id": plan.transaction_id,
            "policy_snapshot_id": plan.policy_snapshot_id,
            "plan_identity_hash": _v015_checkpoint_plan_hash(plan, scenario_only=False),
            "expected_policy_row_count": plan.batch_size,
        }
    )


def _seal_frontres_checkpoint_transaction_plan(runner: Any, plan: FrontRESFormalTransactionPlan) -> None:
    """标记全部 expected attempt 已到齐, 但 step 前仍禁止 persistence."""

    aggregate = frontres_stage3_transaction_aggregate(runner)
    aggregate.seal(collected_policy_attempt_count=plan.batch_size)


def begin_frontres_v015_checkpoint_transaction_commit(runner: Any) -> None:
    """Close collection before any grouped loss can mutate optimizer state."""

    frontres_stage3_transaction_aggregate(runner).begin_commit()


def _commit_frontres_checkpoint_transaction(
    runner: Any,
    *,
    plan: FrontRESFormalTransactionPlan,
    valid_policy_row_count: int,
    optimizer_step_before: int,
    optimizer_step_after: int,
    curriculum: Any,
) -> None:
    """在唯一允许的 optimizer step 后发布 metadata-only receipt."""

    aggregate = frontres_stage3_transaction_aggregate(runner)
    state = aggregate.as_dict()
    if state.get("state") != "sealed":
        raise RuntimeError("v015 formal transaction commit requires a sealed checkpoint barrier")
    receipt = {
        "method_contract_id": "FRS-METHOD-v020",
        "gain_contract_id": "FRS-GAIN-v008",
        "optimization_contract_id": "FRS-PPO-v008",
        "training_contract_id": "FRS-TRAIN-v019",
        "scalar_target_id": "symmetric-log-recovery-aware-utility-v1",
        "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
        "grouped_schema_id": "grouped-all-attempt-scalar-v1",
        "transaction_id": plan.transaction_id,
        "policy_snapshot_id": plan.policy_snapshot_id,
        "plan_identity_hash": _v015_checkpoint_plan_hash(plan, scenario_only=False),
        "scenario_identity_hash": _v015_checkpoint_plan_hash(plan, scenario_only=True),
        "expected_policy_row_count": int(plan.batch_size),
        "collected_policy_attempt_count": int(state["collected_policy_attempt_count"]),
        "valid_policy_row_count": int(valid_policy_row_count),
        "optimizer_step_before": int(optimizer_step_before),
        "optimizer_step_after": int(optimizer_step_after),
        "optimizer_step_delta": int(optimizer_step_after - optimizer_step_before),
        "curriculum_fingerprint": str(curriculum.schedule_fingerprint),
        "k_stage_index": int(curriculum.stage_index),
        "active_k": int(curriculum.active_k),
        "active_m": int(curriculum.active_m),
        "selected_segment_count": int(plan.selected_segment_count),
        "policy_row_count": int(plan.batch_size),
        "role_row_count": int(2 * plan.batch_size),
        "k_stage_iteration": int(curriculum.stage_iteration),
        "training_iteration": int(curriculum.absolute_iteration),
        "dr_stage_fingerprint": str(curriculum.dr_stage_fingerprint),
        "dr_progress": float(curriculum.dr_progress),
        "d_cap": float(curriculum.d_cap),
    }
    aggregate.commit(receipt)


def abort_frontres_v015_checkpoint_transaction(runner: Any) -> None:
    """Return the single transaction Aggregate to its persistable idle state."""

    frontres_stage3_transaction_aggregate(runner).abort()


def bind_frontres_collection_context(runner: Any, *, route: str, sample: Any, batch: Any) -> None:
    frontres_stage3_transaction_aggregate(runner).bind_collection_context(
        route=route,
        sample=sample,
        batch=batch,
    )


def frontres_collection_batch(runner: Any) -> Any | None:
    batch = frontres_stage3_transaction_aggregate(runner).collection_batch
    if batch is not None:
        return batch
    return getattr(runner, "_frontres_segment_live_current_batch", None)


def clear_frontres_collection_context(runner: Any) -> None:
    frontres_stage3_transaction_aggregate(runner).clear_collection_context()


def update_frontres_observation_trace(runner: Any, **values: Any) -> None:
    frontres_stage3_transaction_aggregate(runner).update_observation_trace(**values)


def frontres_observation_trace(runner: Any) -> Mapping[str, Any]:
    return frontres_stage3_transaction_aggregate(runner).observation_trace()


def publish_frontres_preupdate_diagnostics(runner: Any, values: Mapping[str, Any]) -> None:
    frontres_stage3_transaction_aggregate(runner).publish_preupdate_diagnostics(values)


def frontres_preupdate_diagnostics(runner: Any) -> Mapping[str, Any]:
    return frontres_stage3_transaction_aggregate(runner).preupdate_diagnostics()


@dataclass(frozen=True)
class FrontRESFormalTransactionUpdateResult:
    """One formal grouped update and its committed transaction diagnostics."""

    transaction_id: str
    policy_snapshot_id: str
    segment_count: int
    source_count: int
    policy_attempt_count: int
    valid_row_count: int
    optimizer_step_before: int
    optimizer_step_after: int
    optimizer_step_delta: int
    update_invocation_count: int
    ppo_result: Any
    diagnostics: dict[str, Any]


@dataclass
class FrontRESSegmentLiveRolloutCapture:
    rollout_k: int
    reward_mean: float
    done_frac: float
    last_obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...] | None
    env_action_shape: tuple[int, ...] | None
    transition_obs: torch.Tensor | None
    transition_privileged_obs: torch.Tensor | None
    transition_actions: torch.Tensor | None
    transition_log_probs: torch.Tensor | None
    transition_values: torch.Tensor | None
    transition_means: torch.Tensor | None
    transition_sigmas: torch.Tensor | None
    reward_accum: torch.Tensor | None
    done_any: torch.Tensor | None
    reward_steps: torch.Tensor | None = None
    done_steps: torch.Tensor | None = None
    horizon_k: torch.Tensor | None = None
    actor_update_mask: torch.Tensor | None = None
    n_train: int = 0
    n_candidate: int = 0
    n_base: int = 0
    n_clean: int = 0
    survival_steps: torch.Tensor | None = None
    motion_clean_body_pos: torch.Tensor | None = None
    motion_repaired_body_pos: torch.Tensor | None = None
    motion_noisy_body_pos: torch.Tensor | None = None
    motion_clean_root_quat: torch.Tensor | None = None
    motion_repaired_root_quat: torch.Tensor | None = None
    motion_noisy_root_quat: torch.Tensor | None = None
    physics_zmp_repaired_steps: torch.Tensor | None = None
    physics_zmp_noisy_steps: torch.Tensor | None = None
    physics_contact_repaired_steps: torch.Tensor | None = None
    physics_contact_noisy_steps: torch.Tensor | None = None
    env_actions: torch.Tensor | None = None
    transition_perturbation_rp: torch.Tensor | None = None
    transition_supervised_target: torch.Tensor | None = None
    repair_score_accum: torch.Tensor | None = None
    repair_score_steps: torch.Tensor | None = None
    audit_transaction_id: str | None = None
    audit_batch_signature: str | None = None
    audit_role_signature: str | None = None
    audit_k_signature: str | None = None
    audit_segment_signature: str | None = None
    audit_row_count: int = 0
    audit_identity_state: str = "UNCONFIRMED"
    transition_action_steps: torch.Tensor | None = None
    gain_steps: torch.Tensor | None = None
    survival_gain_steps: torch.Tensor | None = None
    gain_config: Any | None = None


@dataclass(frozen=True)
class FrontRESFrozenPolicyTransactionResult:
    """Offline S2 proof emitted after one complete transaction reaches one update callback."""

    transaction_id: str
    policy_snapshot_id: str
    segment_count: int
    source_count: int
    policy_attempt_count: int
    valid_row_count: int
    optimizer_step_before: int
    optimizer_step_after: int
    optimizer_step_delta: int
    update_invocation_count: int
    update_result: Any


class FrontRESFrozenPolicyTransactionAccumulator:
    """Gate one complete S1b storage batch before exactly one injected update.

    Status: candidate-only/offline.
    Upstream: S1b sealed storage batch and frozen transaction metadata.
    Downstream: one future transaction-aware update callback.
    Evidence: deterministic S2 contract only; no live runner is wired here.
    Gap: grouped PPO and formal route integration remain separate steps.
    """

    def __init__(self, runner: Any, *, optimizer_step_count: Callable[[], int]) -> None:
        if not callable(optimizer_step_count):
            raise TypeError("optimizer_step_count must be callable")
        self._runner = runner
        self._optimizer_step_count = optimizer_step_count
        self._optimizer_step_at_open = self._read_optimizer_step_count()
        self._storage_batch: Any | None = None
        self._metadata: Any | None = None
        self._state = "collecting"
        self._update_invocation_count = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def transaction_id(self) -> str:
        if self._metadata is None:
            raise RuntimeError("frozen transaction has no sealed S1b metadata")
        return str(self._metadata.transaction_id)

    @property
    def update_invocation_count(self) -> int:
        return int(self._update_invocation_count)

    def _read_optimizer_step_count(self) -> int:
        try:
            count = int(self._optimizer_step_count())
        except Exception as exc:
            raise RuntimeError("optimizer_step_count must return an integer") from exc
        if count < 0:
            raise ValueError("optimizer_step_count must be non-negative")
        return count

    def _require_no_optimizer_step_during_collection(self) -> int:
        current = self._read_optimizer_step_count()
        if current != self._optimizer_step_at_open:
            self._state = "failed"
            raise RuntimeError(
                "optimizer step occurred during frozen-policy transaction collection: "
                f"opened={self._optimizer_step_at_open} current={current}"
            )
        return current

    @staticmethod
    def _metadata_row_tensor(metadata: Any, name: str, *, batch_size: int) -> torch.Tensor:
        value = getattr(metadata, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or int(value.numel()) != int(batch_size):
            raise ValueError(f"frozen transaction metadata {name} must be rank-1 [B]")
        return value.detach().to(device="cpu", dtype=torch.long).contiguous()

    def _validate_complete_s1b_storage_batch(self, storage_batch: Any) -> tuple[Any, int, int, int, int]:
        metadata = getattr(storage_batch, "transaction_metadata", None)
        if metadata is None:
            raise ValueError("frozen transaction accumulator requires S1b transaction_metadata")
        validate = getattr(metadata, "validate", None)
        verify_policy = getattr(metadata, "verify_policy", None)
        if not callable(validate) or not callable(verify_policy):
            raise TypeError("S1b transaction_metadata must provide validate() and verify_policy()")
        validate()
        transaction_id = str(getattr(metadata, "transaction_id", ""))
        policy_snapshot_id = str(getattr(metadata, "policy_snapshot_id", ""))
        if not transaction_id or not policy_snapshot_id:
            raise ValueError("S1b transaction metadata requires transaction_id and policy_snapshot_id")
        segment_ids = getattr(storage_batch, "segment_ids", None)
        valid_mask = getattr(storage_batch, "valid_mask", None)
        if not isinstance(segment_ids, torch.Tensor) or segment_ids.ndim != 1:
            raise ValueError("frozen transaction storage requires rank-1 segment_ids")
        batch_size = int(segment_ids.numel())
        if batch_size <= 0:
            raise ValueError("frozen transaction storage must contain at least one row")
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.ndim != 1 or int(valid_mask.numel()) != batch_size:
            raise ValueError("frozen transaction storage valid_mask must be rank-1 [B]")
        metadata_batch_size = int(getattr(metadata, "batch_size", -1))
        if metadata_batch_size != batch_size:
            raise ValueError("S1b transaction metadata row count must equal storage rows")
        metadata_segment_ids = self._metadata_row_tensor(metadata, "segment_ids", batch_size=batch_size)
        if not torch.equal(segment_ids.detach().to(device="cpu", dtype=torch.long), metadata_segment_ids):
            raise ValueError("storage segment_ids disagree with sealed S1b transaction metadata")
        source_index = self._metadata_row_tensor(metadata, "source_index", batch_size=batch_size)
        trial_index = self._metadata_row_tensor(metadata, "trial_index", batch_size=batch_size)
        self._metadata_row_tensor(metadata, "horizon_k", batch_size=batch_size)
        roles = tuple(str(value) for value in getattr(metadata, "trial_role", ()))
        noisy_hashes = tuple(str(value) for value in getattr(metadata, "noisy_segment_hashes", ()))
        scenario_ids = tuple(str(value) for value in getattr(metadata, "scenario_ids", ()))
        if len(roles) != batch_size or len(noisy_hashes) != batch_size or len(scenario_ids) != batch_size:
            raise ValueError("S1b transaction metadata role/hash/scenario rows must equal storage rows")
        if any(role != "policy" for role in roles):
            raise ValueError("frozen-policy transaction may contain only policy attempts")
        unique_segments = torch.unique(metadata_segment_ids, sorted=True)
        unique_sources = torch.unique(source_index, sorted=True)
        if int(unique_segments.numel()) < 2 or int(unique_sources.numel()) < 2:
            raise ValueError("frozen-policy transaction requires at least two distinct Segment sources")
        for source in unique_sources.tolist():
            rows = torch.nonzero(source_index == int(source), as_tuple=False).reshape(-1)
            if int(rows.numel()) < 2:
                raise ValueError(f"source_index={source} requires at least two policy attempts")
            source_trials = sorted(int(value) for value in trial_index[rows].tolist())
            if source_trials != list(range(len(source_trials))):
                raise ValueError(f"source_index={source} trial_index must be contiguous from zero")
            source_hashes = {noisy_hashes[int(row)] for row in rows.tolist()}
            source_scenarios = {scenario_ids[int(row)] for row in rows.tolist()}
            if len(source_hashes) != 1 or len(source_scenarios) != 1:
                raise ValueError(f"source_index={source} has mixed fixed Noisy identity")
        policy = getattr(getattr(self._runner, "alg", None), "policy", None)
        verify_policy(policy)
        return (
            metadata,
            int(unique_segments.numel()),
            int(unique_sources.numel()),
            batch_size,
            int(valid_mask.detach().bool().sum().item()),
        )

    def append_storage_batch(self, storage_batch: Any) -> None:
        if self._state != "collecting":
            raise RuntimeError(f"frozen transaction is not collecting; state={self._state}")
        if self._storage_batch is not None:
            raise RuntimeError("frozen transaction already has its one complete S1b storage batch")
        self._require_no_optimizer_step_during_collection()
        metadata, _, _, _, _ = self._validate_complete_s1b_storage_batch(storage_batch)
        self._require_no_optimizer_step_during_collection()
        self._storage_batch = storage_batch
        self._metadata = metadata

    def finalize_one_update(self, update_callback: Callable[[Any], Any]) -> FrontRESFrozenPolicyTransactionResult:
        if not callable(update_callback):
            raise TypeError("update_callback must be callable")
        if self._state == "finalized":
            raise RuntimeError("frozen transaction is already finalized")
        if self._state != "collecting" or self._storage_batch is None or self._metadata is None:
            raise RuntimeError(f"frozen transaction is not ready to finalize; state={self._state}")
        before_update = self._require_no_optimizer_step_during_collection()
        metadata, segment_count, source_count, policy_attempt_count, valid_row_count = self._validate_complete_s1b_storage_batch(
            self._storage_batch
        )
        self._state = "finalizing"
        self._update_invocation_count += 1
        try:
            update_result = update_callback(self._storage_batch)
        except Exception:
            self._state = "failed"
            raise
        after_update = self._read_optimizer_step_count()
        step_delta = after_update - before_update
        if step_delta != 1:
            self._state = "failed"
            raise RuntimeError(
                "frozen transaction finalization requires exactly one optimizer step: "
                f"before={before_update} after={after_update} delta={step_delta}"
            )
        self._state = "finalized"
        return FrontRESFrozenPolicyTransactionResult(
            transaction_id=str(metadata.transaction_id),
            policy_snapshot_id=str(metadata.policy_snapshot_id),
            segment_count=segment_count,
            source_count=source_count,
            policy_attempt_count=policy_attempt_count,
            valid_row_count=valid_row_count,
            optimizer_step_before=before_update,
            optimizer_step_after=after_update,
            optimizer_step_delta=step_delta,
            update_invocation_count=self._update_invocation_count,
            update_result=update_result,
        )


# Public owner surface. Historical underscore names remain facade-only aliases.
FRONTRES_V015_CHECKPOINT_TRANSACTION_STATE_ATTR = _V015_CHECKPOINT_TRANSACTION_STATE_ATTR
bind_frontres_checkpoint_transaction_plan = _bind_frontres_checkpoint_transaction_plan
seal_frontres_checkpoint_transaction_plan = _seal_frontres_checkpoint_transaction_plan
start_frontres_checkpoint_transaction_commit = begin_frontres_v015_checkpoint_transaction_commit
commit_frontres_checkpoint_transaction = _commit_frontres_checkpoint_transaction
reset_frontres_checkpoint_transaction = abort_frontres_v015_checkpoint_transaction
