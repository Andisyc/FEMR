from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Callable

import torch

from rsl_rl.algorithms import FrontRESUnified
from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
    install_frontres_v004_projected_gradients,
)
from rsl_rl.frontres.frontres_segment_storage import (
    FrontRESV015RejectedTransactionEvidence,
    FrontRESSegmentRolloutStorage,
    FrontRESSegmentTransition,
    FrontRESV015GainReturnEvidence,
    FrontRESV015OneActionKEvidence,
    build_frontres_v015_grouped_candidate_storage,
    build_frontres_v015_gain_return_evidence,
    pair_frontres_v015_gain_facts,
)
try:
    from rsl_rl.frontres.frontres_segment_diagnostics import (
        FrontRESV015LocalEvaluationReport,
        build_frontres_v015_local_evaluation_report,
    )
except (ModuleNotFoundError, ImportError):
    _V015_DIAGNOSTICS_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_segment_diagnostics.py"
    _V015_DIAGNOSTICS_SPEC = importlib.util.spec_from_file_location(
        "frontres_segment_diagnostics_runtime",
        _V015_DIAGNOSTICS_PATH,
    )
    if _V015_DIAGNOSTICS_SPEC is None or _V015_DIAGNOSTICS_SPEC.loader is None:
        raise RuntimeError(f"Could not load v015 transaction diagnostics owner from {_V015_DIAGNOSTICS_PATH}.")
    _V015_DIAGNOSTICS_MODULE = importlib.util.module_from_spec(_V015_DIAGNOSTICS_SPEC)
    sys.modules[_V015_DIAGNOSTICS_SPEC.name] = _V015_DIAGNOSTICS_MODULE
    _V015_DIAGNOSTICS_SPEC.loader.exec_module(_V015_DIAGNOSTICS_MODULE)
    FrontRESV015LocalEvaluationReport = _V015_DIAGNOSTICS_MODULE.FrontRESV015LocalEvaluationReport
    build_frontres_v015_local_evaluation_report = _V015_DIAGNOSTICS_MODULE.build_frontres_v015_local_evaluation_report
from rsl_rl.frontres.frontres_segment_reset import (
    FrontRESSegmentResetAdapter,
    FrontRESSegmentResetResult,
    ensure_frontres_segment_live_reset_hook,
)
try:
    from rsl_rl.frontres.frontres_segment_warmup import (
        frontres_segment_warmup_phase,
        resolve_frontres_k_stage_identity,
    )
except (ModuleNotFoundError, ImportError):
    _WARMUP_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_segment_warmup.py"
    _WARMUP_SPEC = importlib.util.spec_from_file_location("frontres_segment_warmup_runtime", _WARMUP_PATH)
    if _WARMUP_SPEC is None or _WARMUP_SPEC.loader is None:
        raise RuntimeError(f"Could not load Segment warmup owner from {_WARMUP_PATH}.")
    _WARMUP_MODULE = importlib.util.module_from_spec(_WARMUP_SPEC)
    sys.modules[_WARMUP_SPEC.name] = _WARMUP_MODULE
    _WARMUP_SPEC.loader.exec_module(_WARMUP_MODULE)
    frontres_segment_warmup_phase = _WARMUP_MODULE.frontres_segment_warmup_phase
    resolve_frontres_k_stage_identity = _WARMUP_MODULE.resolve_frontres_k_stage_identity
from rsl_rl.frontres.training_schedule import resolve_frontres_mode_state
from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout
try:
    from rsl_rl.runners.frontres_segment_live_sampler import (
        _close_frontres_local_scenarios,
        FrontRESV015FormalTransactionAccumulator,
        FrontRESV015FormalTransactionPlan,
        capture_frontres_frozen_policy_snapshot,
        prepare_frontres_v015_formal_training_batch,
        prepare_frontres_v015_local_sentinel_batch,
    )
except ModuleNotFoundError:
    # 契约测试会按文件加载 probe; 此处保持与常规 package import 相同的 owner.
    _V015_LIVE_SAMPLER_PATH = Path(__file__).resolve().with_name("frontres_segment_live_sampler.py")
    _V015_LIVE_SAMPLER_SPEC = importlib.util.spec_from_file_location(
        "rsl_rl.runners.frontres_segment_live_sampler",
        _V015_LIVE_SAMPLER_PATH,
    )
    if _V015_LIVE_SAMPLER_SPEC is None or _V015_LIVE_SAMPLER_SPEC.loader is None:
        raise RuntimeError(f"Could not load v015 formal transaction sampler from {_V015_LIVE_SAMPLER_PATH}.")
    _V015_LIVE_SAMPLER_MODULE = importlib.util.module_from_spec(_V015_LIVE_SAMPLER_SPEC)
    sys.modules[_V015_LIVE_SAMPLER_SPEC.name] = _V015_LIVE_SAMPLER_MODULE
    _V015_LIVE_SAMPLER_SPEC.loader.exec_module(_V015_LIVE_SAMPLER_MODULE)
    FrontRESV015FormalTransactionAccumulator = _V015_LIVE_SAMPLER_MODULE.FrontRESV015FormalTransactionAccumulator
    FrontRESV015FormalTransactionPlan = _V015_LIVE_SAMPLER_MODULE.FrontRESV015FormalTransactionPlan
    capture_frontres_frozen_policy_snapshot = _V015_LIVE_SAMPLER_MODULE.capture_frontres_frozen_policy_snapshot
    _close_frontres_local_scenarios = _V015_LIVE_SAMPLER_MODULE._close_frontres_local_scenarios
    prepare_frontres_v015_formal_training_batch = _V015_LIVE_SAMPLER_MODULE.prepare_frontres_v015_formal_training_batch
    prepare_frontres_v015_local_sentinel_batch = _V015_LIVE_SAMPLER_MODULE.prepare_frontres_v015_local_sentinel_batch
from rsl_rl.runners.frontres_rollout_step import (
    _append_future_intent_actor_context,
    _frontres_motion_command,
    prepare_frontres_rollout_step,
    prepare_frontres_v015_frozen_gmt_step,
    prepare_frontres_v015_one_action_at_t,
)
_FORMAL_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_audit_probe", Path(__file__).resolve().with_name("frontres_formal_runtime_audit.py")
)
_FORMAL_AUDIT_MODULE = importlib.util.module_from_spec(_FORMAL_AUDIT_SPEC)
assert _FORMAL_AUDIT_SPEC.loader is not None
_FORMAL_AUDIT_SPEC.loader.exec_module(_FORMAL_AUDIT_MODULE)
print_ppo_audit = _FORMAL_AUDIT_MODULE.print_ppo_audit
print_rollout_storage_audit = _FORMAL_AUDIT_MODULE.print_rollout_storage_audit
emit_formal_runtime_probe = _FORMAL_AUDIT_MODULE.emit_formal_runtime_probe
print_reset_lifecycle_audit = _FORMAL_AUDIT_MODULE.print_reset_lifecycle_audit
snapshot_reset_pair_state = _FORMAL_AUDIT_MODULE.snapshot_reset_pair_state
snapshot_termination_terms = _FORMAL_AUDIT_MODULE.snapshot_termination_terms


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


def _fmt_vec(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return "UNCONFIRMED"
    return "[" + ", ".join(_fmt_num(item) for item in value) + "]"


def _mean_sequence(value: Any, default: float = 0.0) -> float:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return float(default)
    return float(sum(float(item) for item in value) / len(value))


def _positive_fraction(value: Any) -> float:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return 0.0
    return sum(1 for item in value if float(item) > 0.0) / float(len(value))


def _finite_mean(value: torch.Tensor | None) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return float("nan")
    flat = value.detach().float().reshape(-1)
    finite = torch.isfinite(flat)
    if not bool(finite.any().item()):
        return float("nan")
    return float(flat[finite].mean().cpu().item())


def _fmt_metric(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "UNCONFIRMED"
    return "UNCONFIRMED" if not math.isfinite(value) else _fmt_num(value)


def _shape_last_dim(shape: tuple[int, ...] | None) -> int | None:
    if shape is None or len(shape) == 0:
        return None
    return int(shape[-1])


def _delta_se_norm(actions: torch.Tensor | None) -> float:
    if not isinstance(actions, torch.Tensor) or actions.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(actions.detach().float(), dim=-1).mean().cpu().item())


def _delta_z_up_frac(actions: torch.Tensor | None) -> float:
    if not isinstance(actions, torch.Tensor) or actions.ndim < 2 or actions.shape[-1] < 3:
        return 0.0
    return float((actions.detach()[..., 2] > 0.0).float().mean().cpu().item())


def _probe_status(summary: dict[str, object]) -> str:
    if int(summary.get("ppo_trust_region_rejected_count", 0) or 0) > 0:
        return "WARN_TRUST_REGION_REJECTED"
    total_loss = float(summary.get("ppo_total_loss", 0.0))
    actor_loss = float(summary.get("ppo_actor_loss", 0.0))
    approx_kl = float(summary.get("ppo_approx_kl", 0.0))
    clip_frac = float(summary.get("ppo_clip_frac", 0.0))
    if not all(math.isfinite(v) for v in (total_loss, actor_loss, approx_kl, clip_frac)):
        return "BAD_NONFINITE"
    if abs(actor_loss) >= 1000.0 or abs(total_loss) >= 1000.0:
        return "BAD_LOSS_EXPLOSION"
    if clip_frac >= 0.3:
        return "WARN_HIGH_CLIP"
    if approx_kl < -0.001:
        return "WARN_NEG_KL"
    return "OK"


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
class FrontRESV015FormalTransactionRequest:
    """Fake-S2-only request injected after all v015 candidate attempts are collected.

    该 request 不会从 sampler/live runner 自动构造. 只有测试注入 owner 能提供
    已封存的 plan 和 candidate shards, 因而本步骤不会意外启动真实训练.
    """

    plan: FrontRESV015FormalTransactionPlan
    candidate_batches: tuple[Any, ...]
    diagnostic_reports: tuple[FrontRESV015LocalEvaluationReport, ...]
    curriculum_fingerprint: str
    k_stage_index: int
    active_k: int
    k_stage_iteration: int
    training_iteration: int
    warmup_phase_name: str
    warmup_actor_loss_weight: float
    policy_evaluator: Any | None = None
    # Optional compatibility cross-check only. It cannot replace the critic
    # rows carried and reordered by the sealed candidate transaction.
    privileged_observations: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_batches", tuple(self.candidate_batches))
        object.__setattr__(self, "diagnostic_reports", tuple(self.diagnostic_reports))
        if not isinstance(self.plan, FrontRESV015FormalTransactionPlan):
            raise TypeError("v015 formal transaction request requires FrontRESV015FormalTransactionPlan")
        self.plan.validate()
        if not self.candidate_batches:
            raise ValueError("v015 formal transaction request requires candidate batches")
        if len(self.diagnostic_reports) != len(self.candidate_batches):
            raise ValueError("v015 formal transaction requires one immutable diagnostic projection per candidate batch")
        for candidate_batch, report in zip(self.candidate_batches, self.diagnostic_reports, strict=True):
            if not isinstance(report, FrontRESV015LocalEvaluationReport):
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
        if self.warmup_phase_name not in {"critic_only", "actor_warmup", "joint"}:
            raise ValueError("v015 formal transaction has an invalid warmup phase")
        if not 0.0 <= float(self.warmup_actor_loss_weight) <= 1.0:
            raise ValueError("v015 formal transaction actor loss weight must be in [0,1]")
        horizon = self.plan.horizon_k.detach().to(device="cpu", dtype=torch.long)
        if not bool((horizon == int(self.active_k)).all().item()):
            raise ValueError("v015 formal transaction rejects mixed-K or active-K-mismatched plan rows")


_V015_CHECKPOINT_TRANSACTION_STATE_ATTR = "_frontres_v015_checkpoint_transaction_state"


def _v015_checkpoint_plan_hash(plan: FrontRESV015FormalTransactionPlan, *, scenario_only: bool) -> str:
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


def open_frontres_v015_checkpoint_transaction_barrier(runner: Any) -> None:
    """在 injected provider 收集 candidate evidence 前打开 save barrier."""

    existing = getattr(runner, _V015_CHECKPOINT_TRANSACTION_STATE_ATTR, None)
    phase = str(existing.get("state", "")) if isinstance(existing, dict) else ""
    if phase in {"collecting", "sealed"}:
        raise RuntimeError(f"v015 formal transaction checkpoint barrier is already active; state={phase}")
    setattr(runner, _V015_CHECKPOINT_TRANSACTION_STATE_ATTR, {"state": "collecting", "phase": "provider"})


def _bind_frontres_v015_checkpoint_transaction_plan(
    runner: Any,
    plan: FrontRESV015FormalTransactionPlan,
) -> None:
    """在 collection 仍禁止 checkpoint 时发布 immutable transaction identity."""

    plan.validate()
    existing = getattr(runner, _V015_CHECKPOINT_TRANSACTION_STATE_ATTR, None)
    phase = str(existing.get("state", "")) if isinstance(existing, dict) else ""
    if phase not in {"", "idle", "collecting", "committed"}:
        raise RuntimeError(f"v015 formal transaction cannot bind checkpoint state={phase}")
    setattr(
        runner,
        _V015_CHECKPOINT_TRANSACTION_STATE_ATTR,
        {
            "state": "collecting",
            "transaction_id": plan.transaction_id,
            "policy_snapshot_id": plan.policy_snapshot_id,
            "plan_identity_hash": _v015_checkpoint_plan_hash(plan, scenario_only=False),
            "expected_policy_row_count": plan.batch_size,
        },
    )


def _seal_frontres_v015_checkpoint_transaction_plan(runner: Any, plan: FrontRESV015FormalTransactionPlan) -> None:
    """标记全部 expected attempt 已到齐, 但 step 前仍禁止 persistence."""

    state = getattr(runner, _V015_CHECKPOINT_TRANSACTION_STATE_ATTR, None)
    if not isinstance(state, dict) or state.get("state") != "collecting":
        raise RuntimeError("v015 formal transaction seal requires an active collecting checkpoint barrier")
    sealed = dict(state)
    sealed["state"] = "sealed"
    sealed["collected_policy_attempt_count"] = plan.batch_size
    setattr(runner, _V015_CHECKPOINT_TRANSACTION_STATE_ATTR, sealed)


def _commit_frontres_v015_checkpoint_transaction(
    runner: Any,
    *,
    plan: FrontRESV015FormalTransactionPlan,
    valid_policy_row_count: int,
    optimizer_step_before: int,
    optimizer_step_after: int,
    curriculum: Any,
) -> None:
    """在唯一允许的 optimizer step 后发布 metadata-only receipt."""

    state = getattr(runner, _V015_CHECKPOINT_TRANSACTION_STATE_ATTR, None)
    if not isinstance(state, dict) or state.get("state") != "sealed":
        raise RuntimeError("v015 formal transaction commit requires a sealed checkpoint barrier")
    receipt = {
        "method_contract_id": "FRS-METHOD-v016",
        "gain_contract_id": "FRS-GAIN-v006",
        "optimization_contract_id": "FRS-PPO-v004",
        "training_contract_id": "FRS-TRAIN-v010",
        "scalar_target_id": "paired-intent-minus-repair-v1",
        "constraint_schema_id": "contact-loaded-phase_zmp-survival-physical-v2",
        "projection_schema_id": "grouped-first-order-constraint-projection-v1",
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
        "k_stage_iteration": int(curriculum.stage_iteration),
        "training_iteration": int(curriculum.absolute_iteration),
    }
    setattr(runner, _V015_CHECKPOINT_TRANSACTION_STATE_ATTR, {"state": "committed", "receipt": receipt})


@dataclass(frozen=True)
class FrontRESV015FormalTransactionUpdateResult:
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
    max_delta_rpy: float | None = None
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


def _gain_module() -> Any | None:
    try:
        from rsl_rl.frontres import frontres_gain
    except (ImportError, ModuleNotFoundError):
        return None
    return frontres_gain


def _verbose_probe_enabled(runner: Any, items: Any) -> bool:
    if bool(getattr(getattr(runner, "alg", object()), "frontres_segment_verbose_probe", False)):
        return True
    if isinstance(items, torch.Tensor):
        count = int(items.numel())
    else:
        try:
            count = len(items)
        except TypeError:
            count = int(items)
    return count <= _VERBOSE_PROBE_BATCH_LIMIT


def _id_summary(segment_ids: torch.Tensor) -> str:
    ids = segment_ids.detach().long().reshape(-1).cpu()
    count = int(ids.numel())
    if count == 0:
        return "count=0 id_min=None id_max=None"
    return f"count={count} id_min={int(ids.min().item())} id_max={int(ids.max().item())}"


_AUDIT_IDENTITY_KEYS = (
    "audit_transaction_id",
    "audit_batch_signature",
    "audit_role_signature",
    "audit_k_signature",
    "audit_segment_signature",
    "audit_row_count",
    "audit_identity_state",
)


def _audit_identity_kwargs(identity: dict[str, Any] | None) -> dict[str, Any]:
    """Return the compact identity fields shared by cards 15-17."""

    if not isinstance(identity, dict):
        return {
            "audit_transaction_id": "UNCONFIRMED",
            "audit_batch_signature": "UNCONFIRMED",
            "audit_role_signature": "UNCONFIRMED",
            "audit_k_signature": "UNCONFIRMED",
            "audit_segment_signature": "UNCONFIRMED",
            "audit_row_count": 0,
            "audit_identity_state": "UNCONFIRMED",
        }
    return {key: identity.get(key, "UNCONFIRMED") for key in _AUDIT_IDENTITY_KEYS}


def _capture_audit_identity_kwargs(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, Any]:
    return _audit_identity_kwargs(
        {
            "audit_transaction_id": capture.audit_transaction_id,
            "audit_batch_signature": capture.audit_batch_signature,
            "audit_role_signature": capture.audit_role_signature,
            "audit_k_signature": capture.audit_k_signature,
            "audit_segment_signature": capture.audit_segment_signature,
            "audit_row_count": capture.audit_row_count,
            "audit_identity_state": capture.audit_identity_state,
        }
    )


def _audit_identity_tuple(value: Any, batch_size: int, default: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().reshape(-1).cpu().tolist()
    try:
        items = tuple(value)
    except TypeError:
        items = ()
    if len(items) == batch_size:
        return items
    if len(items) > 0 and batch_size % len(items) == 0:
        return items * (batch_size // len(items))
    return (default,) * batch_size


def _new_live_audit_identity(
    runner: Any,
    *,
    pair_layout: Any,
    batch_size: int,
    horizon_k: torch.Tensor,
) -> dict[str, Any]:
    """Create one stable row identity for the current rollout capture.

    Status: active evidence identity owner.
    Upstream: current segment batch/reset request and rollout horizon.
    Downstream: paired Gain, Segment storage/returns, and diagnostics.
    Evidence: offline identity contract; live equality remains to be observed.
    """

    counter = int(getattr(runner, "_frontres_segment_audit_transaction_counter", 0)) + 1
    runner._frontres_segment_audit_transaction_counter = counter
    iteration = int(getattr(runner, "current_learning_iteration", 0))
    transaction_id = f"iter{iteration}:capture{counter}"
    current_batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    sample = getattr(runner, "_frontres_segment_live_current_sample", None)
    raw_segment_ids = getattr(current_batch, "segment_ids", None)
    if raw_segment_ids is None:
        raw_segment_ids = getattr(sample, "segment_ids", None)
    segment_ids = _audit_identity_tuple(raw_segment_ids, batch_size, -1)
    raw_roles = getattr(current_batch, "frontres_segment_trial_role", None)
    if raw_roles is None:
        raw_roles = getattr(sample, "trial_role", None)
    roles = tuple(str(item) for item in _audit_identity_tuple(raw_roles, batch_size, "UNCONFIRMED"))
    request = getattr(runner, "_frontres_segment_live_current_reset_request", None)
    motion_ids = tuple(
        str(item)
        for item in _audit_identity_tuple(getattr(request, "motion_ids", None), batch_size, "UNCONFIRMED")
    )
    start_frames = tuple(
        int(item)
        for item in _audit_identity_tuple(getattr(request, "start_frames", None), batch_size, -1)
    )
    horizon = tuple(int(item) for item in horizon_k.detach().long().reshape(-1).cpu().tolist())
    if len(horizon) != batch_size:
        horizon = (int(max(1, int(getattr(pair_layout, "rollout_k", 1)))),) * batch_size
    rows = tuple(zip(segment_ids, roles, motion_ids, start_frames, horizon))
    batch_signature = hashlib.sha1(repr(rows).encode("utf-8")).hexdigest()[:16]
    identity_state = (
        "complete"
        if all(item != "UNCONFIRMED" for item in motion_ids)
        and all(item >= 0 for item in start_frames)
        and all(item != "UNCONFIRMED" for item in roles)
        else "partial"
    )
    identity = {
        "audit_transaction_id": transaction_id,
        "audit_batch_signature": batch_signature,
        "audit_role_signature": "|".join(roles),
        "audit_k_signature": ",".join(str(item) for item in horizon),
        "audit_segment_signature": ",".join(str(item) for item in segment_ids),
        "audit_row_count": batch_size,
        "audit_identity_state": identity_state,
    }
    runner._frontres_segment_live_audit_identity = identity
    return identity


def _tensor_range_summary(name: str, value: torch.Tensor) -> str:
    data = value.detach().long().reshape(-1).cpu()
    count = int(data.numel())
    if count == 0:
        return f"{name}_count=0 {name}_min=None {name}_max=None"
    return f"{name}_count={count} {name}_min={int(data.min().item())} {name}_max={int(data.max().item())}"


def _tensor_nonzero_frac(value: torch.Tensor) -> float:
    data = value.detach().reshape(-1)
    if int(data.numel()) <= 0:
        return 0.0
    return float((data != 0).float().mean().cpu().item())


def _safe_getattr(owner: Any, name: str) -> Any:
    try:
        return getattr(owner, name)
    except Exception as exc:  # pragma: no cover - diagnostic-only best effort.
        return f"<error {type(exc).__name__}: {exc}>"


def _tensor_debug_summary(name: str, value: Any, *, limit: int = _VERBOSE_PROBE_BATCH_LIMIT) -> str:
    if value is None:
        return f"  {name}: None"
    if not isinstance(value, torch.Tensor):
        return f"  {name}: {value}"
    data = value.detach()
    flat = data.reshape(-1)
    result: dict[str, Any] = {
        "shape": tuple(data.shape),
        "device": str(data.device),
        "dtype": str(data.dtype),
    }
    if int(flat.numel()) <= 0:
        result.update({"numel": 0, "finite": True, "nonzero_frac": "0.0%"})
        return f"  {name}: {result}"
    numeric = flat.float()
    result.update(
        {
            "numel": int(flat.numel()),
            "finite": bool(torch.isfinite(numeric).all().cpu().item()),
            "min": _fmt_num(numeric.min().cpu().item()),
            "max": _fmt_num(numeric.max().cpu().item()),
            "mean": _fmt_num(numeric.mean().cpu().item()),
            "abs_max": _fmt_num(numeric.abs().max().cpu().item()),
            "nonzero_frac": _fmt_pct((flat != 0).float().mean().cpu().item()),
        }
    )
    if int(flat.numel()) <= int(limit):
        result["values"] = flat.cpu().tolist()
    return f"  {name}: {result}"


def _optimizer_parameter_snapshots(policy: Any, optimizer: Any) -> tuple[tuple[str, torch.Tensor], dict[int, torch.Tensor]]:
    names = {id(param): name for name, param in policy.named_parameters()} if hasattr(policy, "named_parameters") else {}
    params: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()
    for group in getattr(optimizer, "param_groups", ()):
        for param in group.get("params", ()):
            if not isinstance(param, torch.Tensor) or id(param) in seen:
                continue
            seen.add(id(param))
            params.append((names.get(id(param), f"param_{len(params)}"), param))
    snapshots = {id(param): param.detach().clone() for _, param in params}
    return tuple(params), snapshots


def _parameter_delta_stats(
    params: tuple[tuple[str, torch.Tensor], ...],
    snapshots: dict[int, torch.Tensor],
) -> dict[str, Any]:
    total = len(params)
    changed = 0
    max_abs = 0.0
    l2_sq = 0.0
    first_changed = ""
    for name, param in params:
        before = snapshots.get(id(param))
        if before is None:
            continue
        delta = (param.detach() - before).float().reshape(-1)
        if int(delta.numel()) <= 0:
            continue
        param_max = float(delta.abs().max().cpu().item())
        if param_max > 0.0:
            changed += 1
            if not first_changed:
                first_changed = name
        max_abs = max(max_abs, param_max)
        l2_sq += float(delta.pow(2).sum().cpu().item())
    return {
        "param_delta_max_abs": max_abs,
        "param_delta_l2": math.sqrt(l2_sq),
        "param_delta_changed": changed,
        "param_delta_total": total,
        "param_delta_first_changed": first_changed,
    }


def _restore_optimizer_parameters(
    params: tuple[tuple[str, torch.Tensor], ...],
    snapshots: dict[int, torch.Tensor],
) -> None:
    for _, param in params:
        before = snapshots.get(id(param))
        if before is not None:
            param.data.copy_(before)


def _clear_noncritic_grads(policy: Any, optimizer_params: tuple[tuple[str, torch.Tensor], ...]) -> None:
    """Hold the full-6D actor and its std fixed during DP-09 critic-only warmup."""
    critic = getattr(policy, "critic", None)
    critic_ids = {id(param) for param in critic.parameters()} if critic is not None else set()
    if not critic_ids:
        raise RuntimeError("DP-09 critic-only warmup requires policy.critic parameters.")
    for _, param in optimizer_params:
        if id(param) not in critic_ids:
            param.grad = None


def _set_segment_optimizer_lr(alg: Any, lr: float) -> None:
    optimizer = getattr(alg, "optimizer", None)
    for group in getattr(optimizer, "param_groups", ()) or ():
        group["lr"] = float(lr)
    object.__setattr__(alg, "learning_rate", float(lr))


def _attach_ppo_update_diagnostics(result: Any, diagnostics: dict[str, Any]) -> None:
    for key, value in diagnostics.items():
        object.__setattr__(result, key, value)


def _post_update_segment_ppo_diagnostics(
    policy_adapter: Any,
    ppo_batch: FrontRESSegmentPPOBatch,
    ppo_cfg: FrontRESSegmentPPOConfig,
) -> dict[str, Any]:
    """Re-forward the same batch after optimizer.step and rename diagnostics as post-update.

    Status: active diagnostic boundary, not an optimizer or loss owner.
    Upstream: run_frontres_segment_single_update calls this after optimizer.step.
    Downstream: trust-region rollback, live summary, and PPO probe text consume these fields.
    Evidence: contract-confirmed by frontres_segment_live_single_update_contract.py.
    Gap: only proves same-batch post-step diagnostics, not long-horizon training quality.
    """
    with torch.no_grad():
        post_result = compute_frontres_segment_ppo_loss(policy_adapter, ppo_batch, ppo_cfg)
    post_kl = (
        float(post_result.distribution_kl_mean)
        if bool(post_result.distribution_kl_available)
        else float(post_result.logprob_approx_kl)
    )
    # compute_frontres_segment_ppo_loss names values by local forward timing.
    # Here that local "pre_update" means "before any further update", i.e. the
    # post-step distribution produced by the just-finished optimizer.step.
    post_raw_log_ratio_mean = float(post_result.pre_update_raw_log_ratio_mean)
    post_raw_log_ratio_min = float(post_result.pre_update_raw_log_ratio_min)
    post_raw_log_ratio_max = float(post_result.pre_update_raw_log_ratio_max)
    post_clamped_ratio_mean = float(post_result.pre_update_clamped_ratio_mean)
    post_clamped_ratio_max = float(post_result.pre_update_clamped_ratio_max)
    return {
        "post_update_distribution_kl_mean": float(post_result.distribution_kl_mean),
        "post_update_distribution_kl_available": bool(post_result.distribution_kl_available),
        "post_update_logprob_approx_kl": float(post_result.logprob_approx_kl),
        "post_update_raw_log_ratio_mean": post_raw_log_ratio_mean,
        "post_update_raw_log_ratio_min": post_raw_log_ratio_min,
        "post_update_raw_log_ratio_max": post_raw_log_ratio_max,
        "post_update_clamped_ratio_mean": post_clamped_ratio_mean,
        "post_update_clamped_ratio_max": post_clamped_ratio_max,
        "post_update_ratio_mean": post_clamped_ratio_mean,
        "post_update_ratio_max": post_clamped_ratio_max,
        "post_update_clip_frac": float(post_result.clip_frac),
        "post_update_approx_kl": post_kl,
        "post_update_mean_delta_l2_mean": float(post_result.distribution_mean_delta_l2_mean),
        "post_update_mean_delta_max_abs": float(post_result.distribution_mean_delta_max_abs),
        "post_update_old_sigma_min": float(post_result.old_sigma_min),
        "post_update_sigma_min": float(post_result.sigma_min),
        "post_update_raw_action_old_mean_l2_mean": float(post_result.raw_action_old_mean_l2_mean),
        "post_update_raw_action_old_mean_abs_max": float(post_result.raw_action_old_mean_abs_max),
        "post_update_raw_action_old_mean_abs_dim_mean": tuple(post_result.raw_action_old_mean_abs_dim_mean),
        "post_update_raw_action_old_mean_abs_dim_max": tuple(post_result.raw_action_old_mean_abs_dim_max),
        "post_update_old_sigma_dim_mean": tuple(post_result.old_sigma_dim_mean),
        "post_update_sigma_dim_mean": tuple(post_result.sigma_dim_mean),
        "post_update_distribution_mean_delta_dim_mean": tuple(post_result.distribution_mean_delta_dim_mean),
        "post_update_distribution_mean_delta_abs_dim_max": tuple(
            post_result.distribution_mean_delta_abs_dim_max
        ),
        "post_update_log_ratio_contrib_dim_mean": tuple(post_result.log_ratio_contrib_dim_mean),
        "post_update_log_ratio_contrib_abs_dim_max": tuple(post_result.log_ratio_contrib_abs_dim_max),
        "post_update_log_jacobian_dim_mean": tuple(post_result.log_jacobian_dim_mean),
        "post_update_log_jacobian_abs_dim_max": tuple(post_result.log_jacobian_abs_dim_max),
    }


def _apply_segment_adaptive_learning_rate(
    alg: Any,
    ppo_result: Any,
    *,
    kl_mean: float | None = None,
    allow_increase: bool = True,
) -> dict[str, Any]:
    optimizer = getattr(alg, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    desired_kl = getattr(alg, "desired_kl", None)
    schedule = str(getattr(alg, "schedule", "fixed")).lower()
    min_lr = float(getattr(alg, "frontres_segment_min_learning_rate", 1e-7))
    max_lr = float(getattr(alg, "frontres_segment_max_learning_rate", 1e-2))
    if not param_groups:
        return {
            "adaptive_lr_applied": 0,
            "adaptive_lr_before": 0.0,
            "adaptive_lr_after": 0.0,
            "adaptive_lr_desired_kl": float(desired_kl) if desired_kl is not None else 0.0,
            "adaptive_lr_schedule": schedule,
            "adaptive_lr_allow_increase": int(bool(allow_increase)),
        }
    lr_before = float(getattr(alg, "learning_rate", param_groups[0].get("lr", 0.0)))
    lr_after = lr_before
    if kl_mean is not None:
        kl_mean = float(kl_mean)
    elif bool(getattr(ppo_result, "distribution_kl_available", False)):
        kl_mean = float(getattr(ppo_result, "distribution_kl_mean", 0.0))
    else:
        kl_mean = float(getattr(ppo_result, "approx_kl", 0.0))
    applied = 0
    if desired_kl is not None and schedule == "adaptive" and math.isfinite(kl_mean):
        desired = float(desired_kl)
        if kl_mean > desired * 2.0:
            excess = kl_mean / max(desired * 2.0, 1e-12)
            lr_after = min(max_lr, max(min_lr, lr_before / max(1.5, math.sqrt(excess))))
        elif allow_increase and kl_mean < desired / 2.0 and kl_mean > 0.0:
            lr_after = min(max_lr, lr_before * 1.5)
        applied = int(lr_after != lr_before)
        _set_segment_optimizer_lr(alg, lr_after)
    return {
        "adaptive_lr_applied": applied,
        "adaptive_lr_before": lr_before,
        "adaptive_lr_after": lr_after,
        "adaptive_lr_kl_mean": kl_mean,
        "adaptive_lr_desired_kl": float(desired_kl) if desired_kl is not None else 0.0,
        "adaptive_lr_schedule": schedule,
        "adaptive_lr_min": min_lr,
        "adaptive_lr_max": max_lr,
        "adaptive_lr_allow_increase": int(bool(allow_increase)),
    }


def _family_mask_debug_lines(masks: Any) -> tuple[str, ...]:
    if not isinstance(masks, dict):
        return (f"  dr.family_masks: {masks}",)
    counts: dict[str, int] = {}
    values: dict[str, Any] = {}
    for family, mask in masks.items():
        if isinstance(mask, torch.Tensor):
            bool_mask = mask.detach().bool().reshape(-1)
            counts[str(family)] = int(bool_mask.sum().cpu().item())
            if int(bool_mask.numel()) <= _VERBOSE_PROBE_BATCH_LIMIT:
                values[str(family)] = bool_mask.cpu().tolist()
        else:
            values[str(family)] = mask
    return (
        f"  dr.family_mask_counts: {counts}",
        f"  dr.family_mask_values: {values}",
    )


def _perturber_debug_lines(runner: Any, *, rollout_step: int | None = None) -> tuple[str, ...]:
    command = _motion_command_for_runner(runner)
    if command is None:
        return ("  dr.motion_command: missing",)
    perturber = getattr(command, "perturber", None)
    if perturber is None:
        return ("  dr.perturber: missing",)
    cfg = getattr(perturber, "cfg", None)
    cfg_names = (
        "enable",
        "root_tilt_prob",
        "root_tilt_max_rad",
        "iid_prob_rp",
        "iid_std_rp",
        "iid_prob_xy",
        "iid_std_xy",
        "iid_prob_ya",
        "iid_std_ya",
        "iid_prob_z",
        "iid_std_z",
        "local_root_artifact_prob",
        "local_root_artifact_xy_std",
        "local_root_artifact_yaw_std",
        "iid_temporal_mode",
        "iid_burst_min_steps",
        "iid_burst_max_steps",
    )
    cfg_values = {name: getattr(cfg, name, None) for name in cfg_names} if cfg is not None else None
    lines = [
        f"  dr.rollout_step: {rollout_step if rollout_step is not None else 'n/a'}",
        f"  dr.cfg: {cfg_values}",
        f"  dr.scale_scalar: {_safe_getattr(perturber, '_dr_scale')}",
        _tensor_debug_summary("dr.scale_env", _safe_getattr(perturber, "_dr_scale_env")),
        *_family_mask_debug_lines(_safe_getattr(perturber, "_family_masks")),
    ]
    for name in (
        "_roll_state",
        "_pitch_state",
        "_iid_event_rp",
        "_iid_event_yaw",
        "_iid_event_xy",
        "_iid_event_z",
        "_iid_event_active",
        "_iid_event_start",
        "_artifact_yaw",
        "_artifact_xy",
        "_artifact_steps",
    ):
        lines.append(_tensor_debug_summary(f"dr.{name}", _safe_getattr(perturber, name)))
    for name in (
        "_cached_perturbed_pos",
        "_cached_perturbed_quat",
        "anchor_dr_delta_pos",
        "anchor_dr_delta_quat_correction",
        "_dr_supervised_target",
        "jump_degree",
    ):
        lines.append(_tensor_debug_summary(f"cmd.{name}", _safe_getattr(command, name)))
    return tuple(lines)


def _print_frontres_dr_runtime_probe(runner: Any, *, label: str, rollout_step: int | None = None) -> None:
    return
    # DR runtime diagnostic dump; uncomment when tracing live perturbation state.
    # if not _live_detail_log_enabled(runner):
    #     return
    # print(
    #     _log_block(
    #         "[FrontRES DR Runtime Probe]",
    #         f"  dr.label: {label}",
    #         *_perturber_debug_lines(runner, rollout_step=rollout_step),
    #     ),
    #     flush=True,
    # )


def _count_summary(values: tuple[Any, ...]) -> dict[str, int]:
    return dict(Counter(str(item) for item in values))


def _motion_summary(motion_ids: tuple[str, ...]) -> str:
    if not motion_ids:
        return "motion_count=0 unique_motion_count=0 first_motion=None"
    return (
        f"motion_count={len(motion_ids)} "
        f"unique_motion_count={len(set(motion_ids))} "
        f"first_motion={motion_ids[0]}"
    )


def _sequence_summary(values: Any, *, limit: int = _VERBOSE_PROBE_BATCH_LIMIT) -> Any:
    try:
        count = len(values)
    except TypeError:
        return values
    if count <= limit:
        return list(values)
    first = values[0] if count else None
    last = values[-1] if count else None
    result = {"count": count, "first": first, "last": last}
    if all(isinstance(item, int) for item in values):
        result.update({"min": min(values), "max": max(values)})
    else:
        result["unique_count"] = len(set(values))
    return result


def _verbose_reset_lines(request: Any, *, verbose: bool) -> tuple[str, ...]:
    if not verbose:
        return ()
    segment_ids = request.segment_ids.detach().long().reshape(-1).cpu().tolist()
    return (
        f"  reset.segment_ids: {_sequence_summary(segment_ids)}",
        f"  reset.mode: {_sequence_summary(tuple(request.mode))}",
    )


def _verbose_index_reset_lines(request: Any, *, verbose: bool) -> tuple[str, ...]:
    if not verbose:
        return ()
    segment_ids = request.segment_ids.detach().long().reshape(-1).cpu().tolist()
    start_frames = request.start_frames.detach().long().reshape(-1).cpu().tolist()
    horizon_k = request.horizon_k.detach().long().reshape(-1).cpu().tolist()
    strength = getattr(request, "perturbation_strength", None)
    strength_values = strength.detach().float().reshape(-1).cpu().tolist() if isinstance(strength, torch.Tensor) else ()
    return (
        f"  reset.segment_ids: {_sequence_summary(segment_ids)}",
        f"  reset.motion_ids: {_sequence_summary(tuple(request.motion_ids))}",
        f"  reset.start_frames: {_sequence_summary(start_frames)}",
        f"  reset.horizon_k: {_sequence_summary(horizon_k)}",
        f"  reset.perturbation_family: {_sequence_summary(tuple(getattr(request, 'perturbation_family', ())))}",
        f"  reset.perturbation_strength: {_sequence_summary(strength_values)}",
    )


def _should_print_once_or_verbose(owner: Any, flag_name: str) -> bool:
    if bool(getattr(owner, "frontres_segment_verbose_probe", False)):
        return True
    if bool(getattr(owner, flag_name, False)):
        return False
    setattr(owner, flag_name, True)
    return True


def _live_detail_log_enabled(runner: Any) -> bool:
    alg = getattr(runner, "alg", None)
    if bool(getattr(alg, "frontres_segment_verbose_probe", False)):
        return True
    return bool(getattr(runner, "_frontres_segment_live_detail_log_enabled", True))


class FrontRESSegmentLivePolicyAdapter:
    def __init__(self, alg: FrontRESUnified, privileged_observations: torch.Tensor | None):
        self.alg = alg
        self.privileged_observations = privileged_observations

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if bool(getattr(self.alg, "use_estimate_ref_vel", False)):
            raise NotImplementedError(
                "FrontRES Segment single-update sentinel does not yet store ref_vel_estimator observations."
            )
        self.alg.policy.act(observations)
        value_obs = self.privileged_observations if self.privileged_observations is not None else observations
        if actions.ndim != 2 or actions.shape[-1] != 6:
            raise ValueError(f"Segment PPO policy evaluation requires 6D Delta SE actions, got {tuple(actions.shape)}")
        action_mean = getattr(self.alg.policy, "action_mean", None)
        action_std = getattr(self.alg.policy, "action_std", None)
        mean_6d = None
        std_6d = None
        raw_actions = None
        log_jacobian_contrib = None
        if action_mean is not None and action_mean.ndim == 2 and action_mean.shape[-1] >= 6:
            mean_6d = action_mean[:, :6]
        if action_std is not None and action_std.ndim == 2 and action_std.shape[-1] >= 6:
            std_6d = action_std[:, :6]
        distribution = getattr(self.alg.policy, "distribution", None)
        if (
            distribution is not None
            and hasattr(distribution, "mean")
            and distribution.mean.ndim == 2
            and distribution.mean.shape[-1] >= 6
        ):
            logprob_parts = _segment_delta_se_log_prob_parts(
                self.alg.policy,
                actions,
                distribution.mean,
                distribution.stddev,
            )
            log_prob = logprob_parts["log_prob"]
            raw_actions = logprob_parts["raw_actions"]
            log_jacobian_contrib = logprob_parts["log_jacobian_contrib"]
        else:
            log_prob = _evaluate_segment_delta_se_log_prob(self.alg.policy, actions, alg=self.alg)
        entropy = getattr(self.alg.policy, "entropy", None)
        if callable(entropy):
            entropy = entropy()
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.reshape(-1)
            if entropy.numel() == 1 and actions.shape[0] != 1:
                entropy = entropy.expand(actions.shape[0])
        if _should_print_once_or_verbose(self.alg, "_frontres_segment_ppo_eval_trace_printed"):
            print(
                "[FrontRES Segment PPO Eval Trace] "
                f"batch_action_shape={tuple(actions.shape)} "
                f"policy_action_mean_shape={tuple(action_mean.shape) if action_mean is not None else None} "
                f"eval_mean_shape={tuple(mean_6d.shape) if mean_6d is not None else None} "
                f"log_prob_shape={tuple(log_prob.shape)} "
                f"actor_obs_shape={tuple(observations.shape)} "
                f"critic_obs_shape={tuple(value_obs.shape)} "
                "semantic=ppo_eval_uses_6d_delta_se_with_separate_critic_obs",
                flush=True,
            )
        return {
            "log_prob": log_prob,
            "value": self.alg.policy.evaluate(value_obs).reshape(-1),
            "entropy": entropy if isinstance(entropy, torch.Tensor) else None,
            "mean": mean_6d,
            "sigma": std_6d,
            "raw_actions": raw_actions,
            "log_jacobian_contrib": log_jacobian_contrib,
        }


def _evaluate_segment_delta_se_log_prob(policy: Any, actions: torch.Tensor, *, alg: Any | None = None) -> torch.Tensor:
    distribution = getattr(policy, "distribution", None)
    if (
        distribution is not None
        and hasattr(distribution, "mean")
        and distribution.mean.ndim == 2
        and distribution.mean.shape[-1] >= 6
    ):
        return _evaluate_segment_delta_se_log_prob_from_stats(policy, actions, distribution.mean, distribution.stddev)
    if alg is not None and hasattr(alg, "_get_actor_log_prob"):
        return alg._get_actor_log_prob(actions).reshape(-1)
    if hasattr(policy, "get_actions_log_prob"):
        return policy.get_actions_log_prob(actions).reshape(-1)
    raise TypeError("policy must expose distribution or get_actions_log_prob for Segment PPO evaluation")


def _evaluate_segment_delta_se_log_prob_from_stats(
    policy: Any,
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return _segment_delta_se_log_prob_parts(policy, actions, mean, std)["log_prob"]


def _segment_delta_se_log_prob_parts(
    policy: Any,
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mean_6d = mean[:, :6].to(device=actions.device, dtype=actions.dtype)
    std_6d = std[:, :6].to(device=actions.device, dtype=actions.dtype)
    if int(getattr(policy, "num_task_corrections", 0)) > 0:
        max_delta_pos = float(getattr(policy, "max_delta_pos", 1.0))
        max_delta_rpy = float(getattr(policy, "max_delta_rpy", 1.0))
        max_d = torch.cat(
            [
                torch.full((3,), max_delta_pos, device=actions.device, dtype=actions.dtype),
                torch.full((3,), max_delta_rpy, device=actions.device, dtype=actions.dtype),
            ],
            dim=-1,
        )
        normalized = (actions / max_d).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(normalized)
        log_prob_dim = torch.distributions.Normal(mean_6d, std_6d).log_prob(raw)
        log_j_dim = torch.log(max_d) + torch.log(1.0 - normalized.pow(2) + 1e-6)
        return {
            "log_prob": log_prob_dim.sum(dim=-1) - log_j_dim.sum(dim=-1),
            "raw_actions": raw,
            "log_jacobian_contrib": log_j_dim,
        }
    log_prob_dim = torch.distributions.Normal(mean_6d, std_6d).log_prob(actions)
    return {
        "log_prob": log_prob_dim.sum(dim=-1),
        "raw_actions": actions,
        "log_jacobian_contrib": torch.zeros_like(actions),
    }


def run_frontres_segment_live_probe(runner: Any, init_at_random_ep_len: bool = True) -> dict[str, object]:
    single_update, storage_write = _resolve_probe_modes(runner)
    episode_before = runner.env.episode_length_buf.detach().clone()
    if init_at_random_ep_len:
        runner.env.episode_length_buf = torch.randint_like(
            runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
        )
    episode_randomized = runner.env.episode_length_buf.detach().clone()

    frontres_mode = resolve_frontres_mode_state(runner, FrontRESActorCritic)
    pair_layout = configure_frontres_pair_layout(runner, is_frontres=frontres_mode.is_frontres)
    reset_result = _apply_current_segment_reset(runner, pair_layout=pair_layout)
    episode_after_reset = runner.env.episode_length_buf.detach().clone()
    reset_skip_reason = str(getattr(runner, "_frontres_segment_live_current_reset_skip_reason", "") or "")
    _print_frontres_dr_runtime_probe(runner, label="after_current_segment_reset")
    observations = _read_live_observations(runner)
    runner.eval_mode()
    capture = _run_live_rollout_capture(
        runner,
        observations,
        reset_lifecycle={
            "episode_before": episode_before,
            "episode_randomized": episode_randomized,
            "episode_after_reset": episode_after_reset,
        },
        pair_layout=pair_layout,
    )
    summary = _initial_live_probe_summary(capture, storage_write=storage_write, single_update=single_update)
    _update_trial_metadata_summary(summary, runner, batch_size=_capture_batch_size(capture))
    _update_reset_summary(
        summary,
        reset_result,
        request=getattr(runner, "_frontres_segment_live_current_reset_request", None),
        skip_reason=reset_skip_reason,
    )

    storage_batch = None
    if storage_write:
        segment_storage = build_live_segment_storage(runner, capture)
        storage_stats = segment_storage.stats()
        storage_batch = segment_storage.full_batch()
        _update_ppo_boundary_summary(summary, storage_batch.valid_mask)
        train_reward_mean = _valid_reward_mean(storage_batch.returns, storage_batch.valid_mask)
        summary.update(
            {
                "storage_size": storage_stats.size,
                "storage_valid_frac": storage_stats.valid_frac,
                "storage_reward_mean": storage_stats.reward_mean,
                "train_reward_mean": train_reward_mean,
                "storage_reward_per_sample": _float_list(storage_batch.returns),
                "storage_valid_mask_per_sample": _bool_list(storage_batch.valid_mask),
                "storage_segment_ids": _long_list(storage_batch.segment_ids),
            }
        )
        if single_update:
            ppo_result = run_frontres_segment_single_update(runner, storage_batch)
            summary.update(
                {
                    "ppo_update": bool(ppo_result.should_step),
                    "ppo_total_loss": float(ppo_result.total_loss.detach().cpu().item()),
                    "ppo_actor_loss": float(ppo_result.actor_loss.detach().cpu().item()),
                    "ppo_value_loss": float(ppo_result.value_loss.detach().cpu().item()),
                    "ppo_valid_count": int(ppo_result.valid_count),
                    "ppo_approx_kl": float(ppo_result.approx_kl),
                    "ppo_clip_frac": float(ppo_result.clip_frac),
                    "ppo_ratio_mean": float(ppo_result.ratio_mean),
                    "ppo_ratio_max": float(ppo_result.ratio_max),
                    "ppo_old_log_prob_mean": float(ppo_result.old_log_prob_mean),
                    "ppo_new_log_prob_mean": float(ppo_result.new_log_prob_mean),
                    "ppo_raw_log_ratio_mean": float(ppo_result.raw_log_ratio_mean),
                    "ppo_raw_log_ratio_min": float(ppo_result.raw_log_ratio_min),
                    "ppo_raw_log_ratio_max": float(ppo_result.raw_log_ratio_max),
                    "ppo_pre_update_raw_log_ratio_mean": float(
                        ppo_result.pre_update_raw_log_ratio_mean
                    ),
                    "ppo_pre_update_raw_log_ratio_min": float(
                        ppo_result.pre_update_raw_log_ratio_min
                    ),
                    "ppo_pre_update_raw_log_ratio_max": float(
                        ppo_result.pre_update_raw_log_ratio_max
                    ),
                    "ppo_pre_update_clamped_ratio_mean": float(
                        ppo_result.pre_update_clamped_ratio_mean
                    ),
                    "ppo_pre_update_clamped_ratio_max": float(
                        ppo_result.pre_update_clamped_ratio_max
                    ),
                    "ppo_pre_distribution_kl_mean": float(getattr(ppo_result, "distribution_kl_mean", 0.0)),
                    "ppo_pre_logprob_approx_kl": float(getattr(ppo_result, "logprob_approx_kl", 0.0)),
                    "ppo_distribution_kl_available": bool(
                        getattr(ppo_result, "distribution_kl_available", False)
                    ),
                    "ppo_post_update_distribution_kl_mean": float(
                        getattr(ppo_result, "post_update_distribution_kl_mean", 0.0)
                    ),
                    "ppo_post_update_logprob_approx_kl": float(
                        getattr(ppo_result, "post_update_logprob_approx_kl", 0.0)
                    ),
                    "ppo_post_update_ratio_mean": float(
                        getattr(ppo_result, "post_update_ratio_mean", 0.0)
                    ),
                    "ppo_post_update_ratio_max": float(
                        getattr(ppo_result, "post_update_ratio_max", 0.0)
                    ),
                    "ppo_post_update_raw_log_ratio_mean": float(
                        getattr(ppo_result, "post_update_raw_log_ratio_mean", 0.0)
                    ),
                    "ppo_post_update_raw_log_ratio_min": float(
                        getattr(ppo_result, "post_update_raw_log_ratio_min", 0.0)
                    ),
                    "ppo_post_update_raw_log_ratio_max": float(
                        getattr(ppo_result, "post_update_raw_log_ratio_max", 0.0)
                    ),
                    "ppo_post_update_clamped_ratio_mean": float(
                        getattr(ppo_result, "post_update_clamped_ratio_mean", 0.0)
                    ),
                    "ppo_post_update_clamped_ratio_max": float(
                        getattr(ppo_result, "post_update_clamped_ratio_max", 0.0)
                    ),
                    "ppo_post_update_clip_frac": float(
                        getattr(ppo_result, "post_update_clip_frac", 0.0)
                    ),
                    "ppo_advantage_mean": float(ppo_result.advantage_mean),
                    "ppo_advantage_min": float(ppo_result.advantage_min),
                    "ppo_advantage_max": float(ppo_result.advantage_max),
                    "ppo_advantage_abs_mean": float(ppo_result.advantage_abs_mean),
                    "ppo_advantage_abs_max": float(ppo_result.advantage_abs_max),
                    "ppo_advantage_abs_top1_frac": float(ppo_result.advantage_abs_top1_frac),
                    "ppo_distribution_mean_delta_l2_mean": float(
                        ppo_result.distribution_mean_delta_l2_mean
                    ),
                    "ppo_distribution_mean_delta_max_abs": float(
                        ppo_result.distribution_mean_delta_max_abs
                    ),
                    "ppo_old_sigma_min": float(ppo_result.old_sigma_min),
                    "ppo_sigma_min": float(ppo_result.sigma_min),
                    "ppo_param_delta_max_abs": float(getattr(ppo_result, "param_delta_max_abs", 0.0)),
                    "ppo_param_delta_l2": float(getattr(ppo_result, "param_delta_l2", 0.0)),
                    "ppo_param_delta_changed": int(getattr(ppo_result, "param_delta_changed", 0)),
                    "ppo_param_delta_total": int(getattr(ppo_result, "param_delta_total", 0)),
                    "ppo_param_delta_first_changed": str(getattr(ppo_result, "param_delta_first_changed", "")),
                    "ppo_param_grad_norm": float(getattr(ppo_result, "param_grad_norm", 0.0)),
                    "ppo_warmup_phase": str(getattr(ppo_result, "warmup_phase", "joint")),
                    "ppo_warmup_phase_iteration": int(getattr(ppo_result, "warmup_phase_iteration", 0)),
                    "ppo_actor_loss_weight": float(getattr(ppo_result, "actor_loss_weight", 1.0)),
                    "ppo_trust_region_rejected_count": int(
                        getattr(ppo_result, "trust_region_rejected_count", 0)
                    ),
                    "ppo_trust_region_accepted": int(getattr(ppo_result, "trust_region_accepted", 1)),
                    "ppo_trust_region_rollback_enabled": int(
                        getattr(ppo_result, "trust_region_rollback_enabled", 0)
                    ),
                    "ppo_trust_region_max_retries": int(
                        getattr(ppo_result, "trust_region_max_retries", 0)
                    ),
                    "ppo_trust_region_schedule": str(
                        getattr(ppo_result, "trust_region_schedule", "unknown")
                    ),
                    "ppo_trust_region_schedule_adaptive": int(
                        getattr(ppo_result, "trust_region_schedule_adaptive", 0)
                    ),
                    "ppo_adaptive_lr_before": float(getattr(ppo_result, "adaptive_lr_before", 0.0)),
                    "ppo_adaptive_lr_after": float(getattr(ppo_result, "adaptive_lr_after", 0.0)),
                    "ppo_adaptive_lr_kl_mean": float(getattr(ppo_result, "adaptive_lr_kl_mean", 0.0)),
                    "ppo_adaptive_lr_desired_kl": float(getattr(ppo_result, "adaptive_lr_desired_kl", 0.0)),
                    "ppo_mosaic_pre_step_adaptive_lr_before": float(
                        getattr(ppo_result, "mosaic_pre_step_adaptive_lr_before", 0.0)
                    ),
                    "ppo_mosaic_pre_step_adaptive_lr_after": float(
                        getattr(ppo_result, "mosaic_pre_step_adaptive_lr_after", 0.0)
                    ),
                    "ppo_mosaic_pre_step_adaptive_lr_kl_mean": float(
                        getattr(ppo_result, "mosaic_pre_step_adaptive_lr_kl_mean", 0.0)
                    ),
                    "ppo_segment_reject_adaptive_lr_after": float(
                        getattr(ppo_result, "segment_reject_adaptive_lr_after", 0.0)
                    ),
                    "ppo_post_update_mean_delta_l2_mean": float(
                        getattr(ppo_result, "post_update_mean_delta_l2_mean", 0.0)
                    ),
                    "ppo_post_update_mean_delta_max_abs": float(
                        getattr(ppo_result, "post_update_mean_delta_max_abs", 0.0)
                    ),
                    "ppo_post_update_old_sigma_min": float(
                        getattr(ppo_result, "post_update_old_sigma_min", 0.0)
                    ),
                    "ppo_post_update_sigma_min": float(
                        getattr(ppo_result, "post_update_sigma_min", 0.0)
                    ),
                    "ppo_post_update_raw_action_old_mean_l2_mean": float(
                        getattr(ppo_result, "post_update_raw_action_old_mean_l2_mean", 0.0)
                    ),
                    "ppo_post_update_raw_action_old_mean_abs_max": float(
                        getattr(ppo_result, "post_update_raw_action_old_mean_abs_max", 0.0)
                    ),
                    "ppo_post_update_raw_action_old_mean_abs_dim_mean": tuple(
                        getattr(ppo_result, "post_update_raw_action_old_mean_abs_dim_mean", ())
                    ),
                    "ppo_post_update_raw_action_old_mean_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_raw_action_old_mean_abs_dim_max", ())
                    ),
                    "ppo_post_update_old_sigma_dim_mean": tuple(
                        getattr(ppo_result, "post_update_old_sigma_dim_mean", ())
                    ),
                    "ppo_post_update_sigma_dim_mean": tuple(
                        getattr(ppo_result, "post_update_sigma_dim_mean", ())
                    ),
                    "ppo_post_update_distribution_mean_delta_dim_mean": tuple(
                        getattr(ppo_result, "post_update_distribution_mean_delta_dim_mean", ())
                    ),
                    "ppo_post_update_distribution_mean_delta_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_distribution_mean_delta_abs_dim_max", ())
                    ),
                    "ppo_post_update_log_ratio_contrib_dim_mean": tuple(
                        getattr(ppo_result, "post_update_log_ratio_contrib_dim_mean", ())
                    ),
                    "ppo_post_update_log_ratio_contrib_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_log_ratio_contrib_abs_dim_max", ())
                    ),
                    "ppo_post_update_log_jacobian_dim_mean": tuple(
                        getattr(ppo_result, "post_update_log_jacobian_dim_mean", ())
                    ),
                    "ppo_post_update_log_jacobian_abs_dim_max": tuple(
                        getattr(ppo_result, "post_update_log_jacobian_abs_dim_max", ())
                    ),
                }
            )
    # AUDIT-PERTURB-02..AUDIT-RETURN-01: 检查 perturb/obs/action/GMT/pair/Gain/return owner 边界.
    # Result: PENDING_LIVE.
    print_rollout_storage_audit(runner, capture=capture, summary=summary, storage_batch=storage_batch)
    _print_live_probe_summary(runner, capture, summary)
    return summary


def _apply_current_segment_reset(
    runner: Any,
    *,
    pair_layout: Any | None = None,
) -> FrontRESSegmentResetResult | None:
    # FRS3-EVAL-013: apply the current index-only reset batch to the live env.
    batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    if batch is None:
        runner._frontres_segment_live_current_reset_skip_reason = "no_current_segment_batch"
        return None
    if _is_index_only_segment_batch(batch):
        return _apply_index_only_segment_reset(runner, batch, pair_layout=pair_layout)
    adapter = getattr(runner, "_frontres_segment_reset_adapter", None)
    if adapter is None:
        adapter = FrontRESSegmentResetAdapter(
            default_preroll_steps=int(getattr(runner.alg, "frontres_segment_preroll_steps", 0)),
            velocity_mismatch_tolerance=float(getattr(runner.alg, "frontres_segment_reset_velocity_tolerance", 1e-3)),
        )
        runner._frontres_segment_reset_adapter = adapter
    reset_mode = str(
        getattr(
            runner.alg,
            "frontres_segment_reset_mode",
            getattr(runner._frontres_segment_replay_boundary, "reset_mode", "auto"),
        )
    ).lower()
    request = adapter.build_request(batch, mode=reset_mode)
    trial_metadata = _current_trial_metadata(
        runner,
        batch_size=int(request.segment_ids.numel()),
        device=request.segment_ids.device,
    )
    _attach_trial_metadata_to_request(request, trial_metadata)
    _attach_frozen_transaction_metadata_to_request(
        request,
        runner=runner,
        batch=batch,
        trial_metadata=trial_metadata,
    )
    if not _env_has_segment_reset_hook(runner.env):
        ensure_frontres_segment_live_reset_hook(
            runner.env,
            robot_name=str(getattr(runner.alg, "frontres_segment_reset_robot_name", "robot")),
            trace=bool(getattr(runner.alg, "frontres_segment_reset_trace", True)),
        )
    result = adapter.apply(runner.env, request)
    runner._frontres_segment_live_current_reset_request = request
    runner._frontres_segment_live_current_reset_result = result
    runner._frontres_segment_live_current_reset_skip_reason = ""
    verbose = _verbose_probe_enabled(runner, request.segment_ids)
    if _live_detail_log_enabled(runner):
        print(
            _log_block(
                "[FrontRES Segment Reset]",
                *_kv_lines(
                    "reset",
                    {
                        "ids": _id_summary(request.segment_ids),
                        "mode_counts": _count_summary(tuple(request.mode)),
                        "valid_count": int(request.valid_mask.detach().bool().sum().cpu().item()),
                        "success_frac": f"{float(result.success_mask.float().mean().detach().cpu().item()):.4f}",
                        "direct_frac": f"{float(result.direct_reset_mask.float().mean().detach().cpu().item()):.4f}",
                        "preroll_frac": f"{float(result.preroll_mask.float().mean().detach().cpu().item()):.4f}",
                        "velocity_mismatch_mean": f"{float(result.velocity_mismatch.float().mean().detach().cpu().item()):.6f}",
                    },
                ),
                *_verbose_reset_lines(request, verbose=verbose),
            ),
            flush=True,
        )
    return result


def _is_index_only_segment_batch(batch: Any) -> bool:
    families = tuple(getattr(batch, "perturbation_family", ()) or ())
    if families:
        return all(str(family) == "index_only" for family in families)
    specs = tuple(getattr(batch, "specs", ()) or ())
    return bool(specs) and all(str(getattr(spec, "perturbation_family", "")) == "index_only" for spec in specs)


def _apply_index_only_segment_reset(
    runner: Any,
    batch: Any,
    *,
    pair_layout: Any | None = None,
) -> FrontRESSegmentResetResult | None:
    specs = tuple(getattr(batch, "specs", ()) or ())
    motion_ids = tuple(str(getattr(spec, "motion_id", "")) for spec in specs)
    start_frames = torch.tensor(
        [int(getattr(spec, "start_frame", 0) or 0) for spec in specs],
        dtype=torch.long,
        device=batch.segment_ids.device,
    )
    horizon_k = torch.tensor(
        [int(getattr(spec, "horizon_k", 1) or 1) for spec in specs],
        dtype=torch.long,
        device=batch.segment_ids.device,
    )
    trial_metadata = _current_trial_metadata(
        runner,
        batch_size=int(batch.segment_ids.numel()),
        device=batch.segment_ids.device,
        default_horizon_k=horizon_k,
    )
    horizon_k = trial_metadata.horizon_k
    perturbation_family = tuple(
        getattr(batch, "stage3_index_perturbation_family", ())
        or getattr(batch, "perturbation_family", ())
        or ()
    )
    perturbation_strength = getattr(
        batch,
        "stage3_index_perturbation_strength",
        getattr(batch, "perturbation_strength", None),
    )
    if not isinstance(perturbation_strength, torch.Tensor):
        perturbation_strength = torch.zeros_like(batch.segment_ids, dtype=torch.float32)
    perturbation_strength = perturbation_strength.to(device=batch.segment_ids.device, dtype=torch.float32).reshape(-1)
    request = SimpleNamespace(
        segment_ids=batch.segment_ids,
        motion_ids=motion_ids,
        start_frames=start_frames,
        horizon_k=horizon_k,
        perturbation_family=perturbation_family,
        perturbation_strength=perturbation_strength,
        valid_mask=torch.ones_like(batch.segment_ids, dtype=torch.bool),
    )
    v015_local_scenario = getattr(batch, "frontres_local_scenario_rows", None) is not None
    if v015_local_scenario:
        _attach_frontres_local_scenario_to_index_request(request, batch)
    else:
        _attach_fixed_noisy_tape_to_index_request(request, batch)
    if pair_layout is not None:
        request.frontres_role_env_ids = _frontres_reset_role_env_ids(
            pair_layout,
            source_count=int(batch.segment_ids.numel()),
            device=batch.segment_ids.device,
            v015_local=v015_local_scenario,
        )
    _attach_trial_metadata_to_request(request, trial_metadata)
    _attach_frozen_transaction_metadata_to_request(
        request,
        runner=runner,
        batch=batch,
        trial_metadata=trial_metadata,
    )
    hook = _index_segment_reset_hook(runner.env)
    if hook is None:
        runner._frontres_segment_live_current_reset_request = None
        runner._frontres_segment_live_current_reset_result = None
        runner._frontres_segment_live_current_reset_skip_reason = "index_only_segment_index"
        verbose = _verbose_probe_enabled(runner, batch.segment_ids)
        if _live_detail_log_enabled(runner):
            print(
                _log_block(
                    "[FrontRES Segment Reset]",
                    *_kv_lines(
                        "reset",
                        {
                            "skip_reason": "index_only_segment_index",
                            "ids": _id_summary(batch.segment_ids),
                            "motion": _motion_summary(motion_ids),
                            "start": _tensor_range_summary("start", start_frames),
                            "perturbation_family_counts": _count_summary(perturbation_family),
                            "request_strength_nonzero_frac": _fmt_pct(_tensor_nonzero_frac(perturbation_strength)),
                        },
                    ),
                    *_verbose_index_reset_lines(request, verbose=verbose),
                ),
                flush=True,
            )
        return None

    raw_result = hook(request)
    result = _index_reset_result_from_mapping(raw_result, request)
    runner._frontres_segment_live_current_reset_request = request
    runner._frontres_segment_live_current_reset_result = result
    runner._frontres_segment_live_current_reset_skip_reason = ""
    verbose = _verbose_probe_enabled(runner, request.segment_ids)
    if _live_detail_log_enabled(runner):
        print(
            _log_block(
                "[FrontRES Segment Reset]",
                *_kv_lines(
                    "reset",
                    {
                        "mode": "index_only",
                        "ids": _id_summary(request.segment_ids),
                        "motion": _motion_summary(motion_ids),
                        "start": _tensor_range_summary("start", request.start_frames),
                        "horizon": _tensor_range_summary("horizon", request.horizon_k),
                        "perturbation_family_counts": _count_summary(request.perturbation_family),
                        "request_strength_nonzero_frac": _fmt_pct(_tensor_nonzero_frac(request.perturbation_strength)),
                        "success_frac": f"{float(result.success_mask.float().mean().detach().cpu().item()):.4f}",
                    },
                ),
                *_verbose_index_reset_lines(request, verbose=verbose),
            ),
            flush=True,
        )
    return result


def _attach_fixed_noisy_tape_to_index_request(request: Any, batch: Any) -> None:
    tape = getattr(batch, "frontres_fixed_noisy_tape", None)
    if tape is None:
        return
    if not isinstance(tape, torch.Tensor) or tape.ndim != 3:
        raise ValueError(f"frontres_fixed_noisy_tape must be [B,L,65], got {getattr(tape, 'shape', None)}")
    batch_size = int(request.segment_ids.numel())
    if int(tape.shape[0]) != batch_size or int(tape.shape[-1]) != 65:
        raise ValueError("frontres_fixed_noisy_tape must align with reset rows and use the 65D carrier")
    offsets = tuple(int(value) for value in (getattr(batch, "frontres_future_offsets", ()) or ()))
    if not offsets or any(value <= 0 for value in offsets) or tuple(sorted(set(offsets))) != offsets:
        raise ValueError("frontres_future_offsets must be nonempty positive ordered offsets for fixed Noisy reset")
    lengths = getattr(batch, "frontres_fixed_noisy_tape_lengths", None)
    scenario_ids = tuple(str(value) for value in (getattr(batch, "frontres_fixed_noisy_scenario_ids", ()) or ()))
    hashes = tuple(str(value) for value in (getattr(batch, "frontres_fixed_noisy_segment_hashes", ()) or ()))
    if (
        not isinstance(lengths, torch.Tensor)
        or int(lengths.numel()) != batch_size
        or len(scenario_ids) != batch_size
        or len(hashes) != batch_size
    ):
        raise ValueError("fixed Noisy reset requires source-aligned tape lengths, scenario ids, and hashes")
    request.frontres_fixed_noisy_tape = tape.detach()
    request.frontres_fixed_noisy_tape_lengths = lengths.detach()
    request.frontres_fixed_noisy_scenario_ids = scenario_ids
    request.frontres_fixed_noisy_segment_hashes = hashes
    request.frontres_future_offsets = offsets


def _attach_frontres_local_scenario_to_index_request(request: Any, batch: Any) -> None:
    """Attach only the v015 split local carrier to an index-reset request.

    The request owns no actor-side Clean reference: q29 intent and the full 65D
    continuation remain separate fields for the command/reset owner to route to
    the actor and frozen GMT consumers respectively.
    """

    if getattr(batch, "frontres_fixed_noisy_tape", None) is not None:
        raise ValueError("v015 local reset request cannot mix a sealed local scenario with a legacy fixed Noisy tape")
    rows = getattr(batch, "frontres_local_scenario_rows", None)
    artifact = getattr(batch, "frontres_local_scenario_current_root_artifact_t", None)
    intent = getattr(batch, "frontres_local_scenario_intent_q29", None)
    continuation = getattr(batch, "frontres_local_scenario_clean_continuation", None)
    expected_support = getattr(batch, "frontres_local_scenario_expected_support", None)
    expected_support_envelope = getattr(batch, "frontres_local_scenario_expected_support_envelope", None)
    lengths = getattr(batch, "frontres_local_scenario_clean_continuation_lengths", None)
    mask = getattr(batch, "frontres_local_scenario_clean_continuation_mask", None)
    scenario_ids = tuple(str(value) for value in (getattr(batch, "frontres_local_scenario_ids", ()) or ()))
    hashes = tuple(str(value) for value in (getattr(batch, "frontres_local_scenario_hashes", ()) or ()))
    x_t_identities = tuple(str(value) for value in (getattr(batch, "frontres_local_scenario_x_t_identities", ()) or ()))
    provenance = tuple(getattr(batch, "frontres_local_scenario_provenance", ()) or ())
    offsets = tuple(int(value) for value in (getattr(batch, "frontres_future_offsets", ()) or ()))
    batch_size = int(request.segment_ids.numel())
    if (
        rows is None
        or not isinstance(artifact, torch.Tensor)
        or not isinstance(intent, torch.Tensor)
        or not isinstance(continuation, torch.Tensor)
        or not isinstance(expected_support, torch.Tensor)
        or not isinstance(expected_support_envelope, torch.Tensor)
        or not isinstance(lengths, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or not offsets
        or any(value <= 0 for value in offsets)
        or tuple(sorted(set(offsets))) != offsets
        or tuple(artifact.shape) != (batch_size, 7)
        or tuple(intent.shape) != (batch_size, max(offsets) + 1, 29)
        or continuation.ndim != 3
        or tuple(continuation.shape[:1]) != (batch_size,)
        or int(continuation.shape[-1]) != 65
        or tuple(expected_support.shape) != tuple(continuation.shape[:2]) + (2,)
        or tuple(expected_support_envelope.shape) != tuple(continuation.shape[:2]) + (6,)
        or tuple(lengths.shape) != (batch_size,)
        or tuple(mask.shape) != tuple(continuation.shape[:2])
        or len(scenario_ids) != batch_size
        or len(hashes) != batch_size
        or len(x_t_identities) != batch_size
        or len(provenance) != batch_size
    ):
        raise ValueError("v015 local reset request requires one aligned sealed artifact, q29 intent, Clean continuation, identity, and provenance row")
    if any(not isinstance(value, Mapping) for value in provenance):
        raise ValueError("v015 local reset request requires mapping provenance for every local scenario row")
    request.frontres_local_scenario_rows = rows
    request.frontres_local_scenario_current_root_artifact_t = artifact.detach().clone()
    request.frontres_local_scenario_intent_q29 = intent.detach().clone()
    request.frontres_local_scenario_clean_continuation = continuation.detach().clone()
    request.frontres_local_scenario_expected_support = expected_support.detach().clone()
    request.frontres_local_scenario_expected_support_envelope = expected_support_envelope.detach().clone()
    request.frontres_local_scenario_clean_continuation_lengths = lengths.detach().clone()
    request.frontres_local_scenario_clean_continuation_mask = mask.detach().clone()
    request.frontres_local_scenario_ids = scenario_ids
    request.frontres_local_scenario_hashes = hashes
    request.frontres_local_scenario_x_t_identities = x_t_identities
    request.frontres_local_scenario_provenance = tuple(dict(value) for value in provenance)
    request.frontres_future_offsets = offsets


def _frontres_reset_role_env_ids(
    pair_layout: Any,
    *,
    source_count: int,
    device: torch.device,
    v015_local: bool = False,
) -> dict[str, torch.Tensor]:
    """将 sampled policy rows 映射到配对的 split-env role rows."""
    source_count = int(source_count)
    counts = (
        (("repair", int(getattr(pair_layout, "n_train", 0))), ("noisy", int(getattr(pair_layout, "n_base", 0))) )
        if v015_local
        else (
            ("policy", int(getattr(pair_layout, "n_train", 0))),
            ("candidate", int(getattr(pair_layout, "n_candidate", 0))),
            ("noisy", int(getattr(pair_layout, "n_base", 0))),
            ("clean", int(getattr(pair_layout, "n_clean", 0))),
        )
    )
    active_counts = [count for _, count in counts if count > 0]
    if not active_counts or any(count != source_count for count in active_counts):
        raise ValueError(
            "Segment index reset requires one split-env row per sampled source and active role; "
            f"source_count={source_count} role_counts={dict(counts)}"
        )
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for role, count in counts:
        if count > 0:
            result[role] = torch.arange(offset, offset + count, dtype=torch.long, device=device)
        offset += count
    return result


def _index_segment_reset_hook(env: Any) -> Any | None:
    for name in ("apply_frontres_segment_index_reset", "reset_to_frontres_segment_index", "set_frontres_segment_index"):
        if hasattr(env, name):
            return getattr(env, name)
    return None


def _index_reset_result_from_mapping(mapping: Any, request: Any) -> FrontRESSegmentResetResult:
    if isinstance(mapping, FrontRESSegmentResetResult):
        return mapping
    if mapping is None:
        mapping = {}
    count = int(request.segment_ids.numel())
    device = request.segment_ids.device
    success = _mapping_bool(mapping, ("success_mask", "reset_success", "valid_mask"), count, device, True)
    fall = _mapping_bool(mapping, ("fall_at_reset_mask", "fall_at_reset", "fall"), count, device, False)
    contact = _mapping_bool(mapping, ("contact_mismatch_mask", "contact_mismatch"), count, device, False)
    velocity = _mapping_float(mapping, ("velocity_mismatch",), count, device, 0.0)
    success = success & (~fall) & (~contact)
    zero = torch.zeros(count, dtype=torch.bool, device=device)
    diagnostics = {
        "reset_success_frac": float(success.float().mean().item()) if count else 0.0,
        "direct_frac": 0.0,
        "preroll_frac": 0.0,
        "invalid_static_frac": 0.0,
        "fall_at_reset_frac": float(fall.float().mean().item()) if count else 0.0,
        "contact_mismatch_frac": float(contact.float().mean().item()) if count else 0.0,
        "velocity_mismatch_mean": float(velocity.float().mean().item()) if count else 0.0,
        "reference_window_applied_frac": 0.0,
    }
    return FrontRESSegmentResetResult(
        success_mask=success,
        direct_reset_mask=zero,
        preroll_mask=zero,
        invalid_static_reset_mask=zero,
        fall_at_reset_mask=fall,
        contact_mismatch_mask=contact,
        velocity_mismatch=velocity,
        diagnostics=diagnostics,
    )


def _mapping_bool(mapping: dict[str, Any], names: tuple[str, ...], count: int, device: torch.device, default: bool) -> torch.Tensor:
    for name in names:
        if name in mapping:
            return mapping[name].to(device=device).bool().flatten()
    return torch.full((count,), default, dtype=torch.bool, device=device)


def _mapping_float(mapping: dict[str, Any], names: tuple[str, ...], count: int, device: torch.device, default: float) -> torch.Tensor:
    for name in names:
        if name in mapping:
            return mapping[name].to(device=device).float().flatten()
    return torch.full((count,), default, dtype=torch.float32, device=device)


def _env_has_segment_reset_hook(env: Any) -> bool:
    return any(hasattr(env, name) for name in ("apply_frontres_segment_reset", "reset_to_segment", "set_segment_state"))


def _update_reset_summary(
    summary: dict[str, object],
    result: FrontRESSegmentResetResult | None,
    *,
    request: Any | None = None,
    skip_reason: str = "",
) -> None:
    # B1: Read the perturbation identity consumed by the reset owner.
    families = tuple(str(item) for item in (getattr(request, "perturbation_family", ()) or ()))
    strength = getattr(request, "perturbation_strength", None)
    strength_values = _float_list(strength) if isinstance(strength, torch.Tensor) else []
    # B2: Preserve distribution facts for diagnostics without changing reset behavior.
    summary.update(
        {
            "perturbation_family_counts": _count_summary(families),
            "perturbation_strength_min": min(strength_values) if strength_values else 0.0,
            "perturbation_strength_mean": (
                sum(strength_values) / float(len(strength_values)) if strength_values else 0.0
            ),
            "perturbation_strength_max": max(strength_values) if strength_values else 0.0,
        }
    )
    if result is None:
        summary.update(
            {
                "segment_reset": False,
                "segment_reset_skip_reason": skip_reason or "not_requested",
                "segment_reset_success_frac": 0.0,
                "segment_reset_direct_frac": 0.0,
                "segment_reset_preroll_frac": 0.0,
                "segment_reset_invalid_static_frac": 0.0,
                "segment_reset_fall_frac": 0.0,
                "segment_reset_contact_mismatch_frac": 0.0,
                "segment_reset_velocity_mismatch_mean": 0.0,
                "segment_reference_window_applied_frac": 0.0,
            }
        )
        return
    diagnostics = result.diagnostics
    summary.update(
        {
            "segment_reset": True,
            "segment_reset_skip_reason": "",
            "segment_reset_success_frac": float(diagnostics.get("reset_success_frac", 0.0)),
            "segment_reset_direct_frac": float(diagnostics.get("direct_frac", 0.0)),
            "segment_reset_preroll_frac": float(diagnostics.get("preroll_frac", 0.0)),
            "segment_reset_invalid_static_frac": float(diagnostics.get("invalid_static_frac", 0.0)),
            "segment_reset_fall_frac": float(diagnostics.get("fall_at_reset_frac", 0.0)),
            "segment_reset_contact_mismatch_frac": float(diagnostics.get("contact_mismatch_frac", 0.0)),
            "segment_reset_velocity_mismatch_mean": float(diagnostics.get("velocity_mismatch_mean", 0.0)),
            "segment_reference_window_applied_frac": float(diagnostics.get("reference_window_applied_frac", 0.0)),
        }
    )


def _capture_batch_size(capture: FrontRESSegmentLiveRolloutCapture) -> int:
    for value in (capture.transition_actions, capture.reward_accum, capture.done_any):
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            return int(value.shape[0])
    return 0


def _current_trial_metadata(
    runner: Any,
    *,
    batch_size: int,
    device: torch.device | str,
    default_horizon_k: torch.Tensor | None = None,
) -> SimpleNamespace:
    batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    roles = getattr(batch, "frontres_segment_trial_role", None) if batch is not None else None
    if roles is None:
        trial_role = ("policy",) * int(batch_size)
    else:
        trial_role = tuple(str(item) for item in roles)
    if len(trial_role) < int(batch_size):
        trial_role = trial_role + ("baseline",) * (int(batch_size) - len(trial_role))
    if len(trial_role) != int(batch_size):
        raise ValueError(f"frontres_segment_trial_role must have {batch_size} rows, got {len(trial_role)}")

    default_source_index = torch.arange(batch_size, dtype=torch.long, device=device)
    default_trial_index = torch.zeros(batch_size, dtype=torch.long, device=device)
    if default_horizon_k is None:
        alg = getattr(runner, "alg", None)
        default_horizon = int(getattr(alg, "frontres_segment_k", 1) or 1)
        default_horizon_k = torch.full((batch_size,), default_horizon, dtype=torch.long, device=device)

    return SimpleNamespace(
        trial_role=trial_role,
        source_index=_trial_long_vector(
            getattr(batch, "frontres_segment_source_index", None) if batch is not None else None,
            name="frontres_segment_source_index",
            batch_size=batch_size,
            device=device,
            default=default_source_index,
        ),
        trial_index=_trial_long_vector(
            getattr(batch, "frontres_segment_trial_index", None) if batch is not None else None,
            name="frontres_segment_trial_index",
            batch_size=batch_size,
            device=device,
            default=default_trial_index,
        ),
        horizon_k=_trial_horizon_vector(
            getattr(batch, "frontres_segment_budget_horizon_k", None) if batch is not None else None,
            name="frontres_segment_budget_horizon_k",
            batch_size=batch_size,
            device=device,
            default=default_horizon_k,
        ),
    )


def _trial_long_vector(
    value: Any,
    *,
    name: str,
    batch_size: int,
    device: torch.device | str,
    default: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        tensor = default
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.tensor(list(value), dtype=torch.long)
    tensor = tensor.to(device=device, dtype=torch.long).reshape(-1)
    if int(tensor.numel()) < int(batch_size):
        expanded = default.to(device=device, dtype=torch.long).reshape(-1).detach().clone()
        if int(expanded.numel()) != int(batch_size):
            raise ValueError(f"{name} default must have {batch_size} rows, got {int(expanded.numel())}")
        expanded[: int(tensor.numel())] = tensor
        tensor = expanded
    if int(tensor.numel()) != int(batch_size):
        raise ValueError(f"{name} must have {batch_size} rows, got {int(tensor.numel())}")
    return tensor.detach()


def _trial_horizon_vector(
    value: Any,
    *,
    name: str,
    batch_size: int,
    device: torch.device | str,
    default: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return default.to(device=device, dtype=torch.long).reshape(-1).detach()
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(list(value), dtype=torch.long)
    tensor = tensor.to(device=device, dtype=torch.long).reshape(-1)
    if int(tensor.numel()) == int(batch_size):
        return tensor.detach()
    if int(tensor.numel()) > 0 and int(batch_size) % int(tensor.numel()) == 0:
        return tensor.repeat(int(batch_size) // int(tensor.numel())).detach()
    return _trial_long_vector(
        tensor,
        name=name,
        batch_size=batch_size,
        device=device,
        default=default,
    )


def _attach_trial_metadata_to_request(request: Any, metadata: SimpleNamespace) -> None:
    object.__setattr__(request, "trial_role", metadata.trial_role)
    object.__setattr__(request, "source_index", metadata.source_index)
    object.__setattr__(request, "trial_index", metadata.trial_index)
    object.__setattr__(request, "budget_horizon_k", metadata.horizon_k)


def _frozen_transaction_vector_has_rows(value: Any, *, batch_size: int) -> bool:
    return isinstance(value, torch.Tensor) and value.ndim == 1 and int(value.numel()) == int(batch_size)


def _same_frozen_transaction_vector(left: Any, right: Any, *, batch_size: int) -> bool:
    if not _frozen_transaction_vector_has_rows(left, batch_size=batch_size) or not _frozen_transaction_vector_has_rows(
        right,
        batch_size=batch_size,
    ):
        return False
    return torch.equal(
        left.detach().to(device="cpu", dtype=torch.long).reshape(-1),
        right.detach().to(device="cpu", dtype=torch.long).reshape(-1),
    ) and int(left.numel()) == int(batch_size)


def _current_frozen_transaction_metadata(
    runner: Any,
    *,
    batch_size: int,
    trial_metadata: SimpleNamespace,
) -> Any | None:
    """Fail closed when a sealed S1b transaction carrier disagrees with the selected batch."""

    batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    metadata = getattr(batch, "frontres_segment_transaction_metadata", None) if batch is not None else None
    if metadata is None:
        return None
    validate = getattr(metadata, "validate", None)
    verify_policy = getattr(metadata, "verify_policy", None)
    if not callable(validate) or not callable(verify_policy):
        raise TypeError("frozen transaction metadata must provide validate() and verify_policy()")
    validate()
    for name in (
        "transaction_id",
        "policy_snapshot_id",
        "policy_state_hash",
        "motion_ids",
        "start_frames",
        "segment_ids",
        "source_index",
        "trial_index",
        "horizon_k",
        "trial_role",
        "noisy_segment_hashes",
    ):
        if not hasattr(metadata, name):
            raise TypeError(f"frozen transaction metadata is missing {name}")
    if not str(metadata.transaction_id) or not str(metadata.policy_snapshot_id) or not str(metadata.policy_state_hash):
        raise ValueError("frozen transaction metadata identity must be non-empty")
    if len(tuple(metadata.motion_ids)) != int(batch_size) or len(tuple(metadata.trial_role)) != int(batch_size):
        raise ValueError("frozen transaction metadata row count does not match the reset/storage batch")
    if len(tuple(metadata.noisy_segment_hashes)) != int(batch_size):
        raise ValueError("frozen transaction metadata requires one Noisy hash per batch row")
    for name in ("start_frames", "segment_ids", "source_index", "trial_index", "horizon_k"):
        if not _frozen_transaction_vector_has_rows(getattr(metadata, name), batch_size=batch_size):
            raise ValueError(f"frozen transaction metadata {name} must be [B]")
    for name, expected, actual in (
        ("segment_ids", metadata.segment_ids, getattr(batch, "segment_ids", None)),
        ("source_index", metadata.source_index, trial_metadata.source_index),
        ("trial_index", metadata.trial_index, trial_metadata.trial_index),
        ("horizon_k", metadata.horizon_k, trial_metadata.horizon_k),
    ):
        if not _same_frozen_transaction_vector(expected, actual, batch_size=batch_size):
            raise ValueError(f"frozen transaction metadata {name} disagrees with the selected batch")
    if tuple(str(value) for value in metadata.trial_role) != tuple(trial_metadata.trial_role):
        raise ValueError("frozen transaction metadata trial_role disagrees with the selected batch")
    if getattr(batch, "frontres_segment_transaction_id", None) != metadata.transaction_id:
        raise ValueError("batch transaction_id disagrees with frozen transaction metadata")
    if getattr(batch, "frontres_segment_policy_snapshot_id", None) != metadata.policy_snapshot_id:
        raise ValueError("batch policy_snapshot_id disagrees with frozen transaction metadata")
    if getattr(batch, "frontres_segment_policy_state_hash", None) != metadata.policy_state_hash:
        raise ValueError("batch policy_state_hash disagrees with frozen transaction metadata")
    if getattr(batch, "frontres_fixed_noisy_transaction_id", None) != metadata.transaction_id:
        raise ValueError("fixed Noisy tape transaction_id disagrees with frozen transaction metadata")
    batch_hashes = tuple(str(value) for value in (getattr(batch, "frontres_fixed_noisy_segment_hashes", ()) or ()))
    if batch_hashes != tuple(str(value) for value in metadata.noisy_segment_hashes):
        raise ValueError("fixed Noisy tape hash rows disagree with frozen transaction metadata")
    return metadata


def _attach_frozen_transaction_metadata_to_request(
    request: Any,
    *,
    runner: Any,
    batch: Any,
    trial_metadata: SimpleNamespace,
) -> Any | None:
    metadata = _current_frozen_transaction_metadata(
        runner,
        batch_size=int(request.segment_ids.numel()),
        trial_metadata=trial_metadata,
    )
    if metadata is None:
        return None
    if not _same_frozen_transaction_vector(metadata.segment_ids, request.segment_ids, batch_size=int(request.segment_ids.numel())):
        raise ValueError("frozen transaction metadata segment_ids disagree with reset request")
    request_motion_ids = getattr(request, "motion_ids", None)
    if request_motion_ids is not None and tuple(str(value) for value in request_motion_ids) != tuple(metadata.motion_ids):
        raise ValueError("frozen transaction metadata motion_ids disagree with reset request")
    request_start_frames = getattr(request, "start_frames", None)
    if request_start_frames is not None and not _same_frozen_transaction_vector(
        metadata.start_frames,
        request_start_frames,
        batch_size=int(request.segment_ids.numel()),
    ):
        raise ValueError("frozen transaction metadata start_frames disagree with reset request")
    for name, value in (
        ("frontres_segment_transaction_metadata", metadata),
        ("frontres_segment_transaction_id", metadata.transaction_id),
        ("frontres_segment_policy_snapshot_id", metadata.policy_snapshot_id),
        ("frontres_segment_policy_state_hash", metadata.policy_state_hash),
        ("frontres_segment_motion_ids", metadata.motion_ids),
        ("frontres_segment_start_frames", metadata.start_frames),
        ("frontres_segment_segment_ids", metadata.segment_ids),
        ("frontres_segment_source_index", metadata.source_index),
        ("frontres_segment_trial_index", metadata.trial_index),
        ("frontres_segment_budget_horizon_k", metadata.horizon_k),
        ("frontres_segment_trial_role", metadata.trial_role),
        ("frontres_segment_noisy_segment_hashes", metadata.noisy_segment_hashes),
    ):
        object.__setattr__(request, name, value)
    return metadata


def _update_trial_metadata_summary(
    summary: dict[str, object],
    runner: Any,
    *,
    batch_size: int,
) -> None:
    metadata = _current_trial_metadata(runner, batch_size=batch_size, device=getattr(runner, "device", "cpu"))
    role_counts = dict(Counter(metadata.trial_role))
    policy_count = int(role_counts.get("policy", 0))
    search_count = int(role_counts.get("search", 0))
    evidence_count = policy_count + search_count
    summary.update(
        {
            "trial_role_per_sample": list(metadata.trial_role),
            "trial_source_index_per_sample": _long_list(metadata.source_index),
            "trial_index_per_sample": _long_list(metadata.trial_index),
            "trial_horizon_k_per_sample": _long_list(metadata.horizon_k),
            "trial_role_counts": role_counts,
            "trial_policy_count": policy_count,
            "trial_search_count": search_count,
            "trial_horizon_summary": _tensor_range_summary("horizon", metadata.horizon_k),
            "ppo_boundary_evidence_rows": evidence_count,
            "ppo_boundary_policy_rows": policy_count,
            "ppo_boundary_search_rows": search_count,
            "ppo_boundary_eligible_rows": 0,
            "ppo_boundary_search_evidence_only_rows": search_count,
            "ppo_boundary_policy_invalid_rows": policy_count,
            "ppo_boundary_valid_policy_frac": 0.0,
            "ppo_boundary_valid_evidence_frac": 0.0,
        }
    )


def _update_ppo_boundary_summary(summary: dict[str, object], valid_mask: torch.Tensor) -> None:
    roles = tuple(str(item) for item in summary.get("trial_role_per_sample", ()))
    valid = valid_mask.detach().bool().reshape(-1).cpu()
    if not roles or len(roles) != int(valid.numel()):
        roles = ("policy",) * int(valid.numel())
    policy_mask = torch.tensor([role == "policy" for role in roles], dtype=torch.bool)
    search_mask = torch.tensor([role == "search" for role in roles], dtype=torch.bool)
    evidence_mask = policy_mask | search_mask
    policy_rows = int(policy_mask.sum().item())
    search_rows = int(search_mask.sum().item())
    eligible_rows = int(valid.sum().item())
    policy_invalid_rows = int((policy_mask & ~valid).sum().item())
    evidence_rows = int(evidence_mask.sum().item())
    summary.update(
        {
            "ppo_boundary_evidence_rows": evidence_rows,
            "ppo_boundary_policy_rows": policy_rows,
            "ppo_boundary_search_rows": search_rows,
            "ppo_boundary_eligible_rows": eligible_rows,
            "ppo_boundary_search_evidence_only_rows": search_rows,
            "ppo_boundary_policy_invalid_rows": policy_invalid_rows,
            "ppo_boundary_valid_policy_frac": float(eligible_rows / max(1, policy_rows)),
            "ppo_boundary_valid_evidence_frac": float(eligible_rows / max(1, evidence_rows)),
        }
    )


def _trial_metadata_priority_evidence(runner: Any, *, batch_size: int, device: torch.device | str) -> dict[str, Any]:
    metadata = _current_trial_metadata(runner, batch_size=batch_size, device=device)
    return {
        "trial_role": metadata.trial_role,
        "source_index": metadata.source_index,
        "trial_index": metadata.trial_index,
        "horizon_k": metadata.horizon_k,
    }


def _trial_metadata_ppo_update_mask(runner: Any, *, batch_size: int, device: torch.device | str) -> torch.Tensor:
    metadata = _current_trial_metadata(runner, batch_size=batch_size, device=device)
    return torch.tensor(
        [role == "policy" for role in metadata.trial_role],
        dtype=torch.bool,
        device=device,
    )


def build_live_segment_storage(runner: Any, capture: FrontRESSegmentLiveRolloutCapture) -> FrontRESSegmentRolloutStorage:
    if (
        capture.transition_obs is None
        or capture.transition_privileged_obs is None
        or capture.transition_actions is None
        or capture.transition_log_probs is None
        or capture.transition_values is None
        or capture.reward_accum is None
        or capture.done_any is None
    ):
        raise RuntimeError("FrontRES Segment live storage probe did not capture a valid first-step PPO tuple.")
    if capture.transition_actions.ndim != 2 or capture.transition_actions.shape[-1] != 6:
        raise ValueError(f"live storage probe requires 6D actions, got {tuple(capture.transition_actions.shape)}")

    batch_size = int(capture.transition_actions.shape[0])
    sample = getattr(runner, "_frontres_segment_live_current_sample", None)
    current_batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    sample_ids = getattr(sample, "segment_ids", None)
    sample_source = getattr(sample, "source", None)
    batch_ids = getattr(current_batch, "segment_ids", None)
    if sample_ids is not None:
        segment_ids = _expand_short_counterfactual_vector(
            sample_ids.to(device=runner.device, dtype=torch.long).reshape(-1),
            name="segment ids",
            batch_size=batch_size,
        )
    elif batch_ids is not None:
        segment_ids = _expand_short_counterfactual_vector(
            batch_ids.to(device=runner.device, dtype=torch.long).reshape(-1),
            name="segment ids",
            batch_size=batch_size,
        )
    else:
        segment_ids = torch.arange(batch_size, device=runner.device, dtype=torch.long)
    if sample_source is not None:
        segment_source = _expand_short_counterfactual_tuple(
            sample_source,
            name="segment source",
            batch_size=batch_size,
        )
    else:
        segment_source = ("live_storage_probe",) * batch_size
    reset_mask = _current_reset_success_mask(runner, batch_size=batch_size, device=runner.device)
    rollout_valid_mask = ~capture.done_any.reshape(-1).bool().to(device=runner.device)
    if capture.actor_update_mask is not None:
        actor_update_mask = capture.actor_update_mask.reshape(-1).bool().to(device=runner.device)
        if int(actor_update_mask.numel()) != batch_size:
            raise ValueError(
                f"actor_update_mask must have {batch_size} rows, got {int(actor_update_mask.numel())}"
            )
    else:
        actor_update_mask = torch.ones(batch_size, device=runner.device, dtype=torch.bool)
    trial_metadata = _current_trial_metadata(runner, batch_size=batch_size, device=runner.device)
    frozen_transaction_metadata = _current_frozen_transaction_metadata(
        runner,
        batch_size=batch_size,
        trial_metadata=trial_metadata,
    )
    if frozen_transaction_metadata is not None:
        policy = getattr(getattr(runner, "alg", None), "policy", None)
        frozen_transaction_metadata.verify_policy(policy)
    ppo_update_mask = _trial_metadata_ppo_update_mask(runner, batch_size=batch_size, device=runner.device)
    valid_mask = rollout_valid_mask & reset_mask & actor_update_mask & ppo_update_mask
    rewards = _segment_storage_rewards(capture, batch_size=batch_size, device=runner.device)
    segment_storage = FrontRESSegmentRolloutStorage(
        capacity=batch_size,
        obs_shape=capture.transition_obs.shape[1:],
        action_dim=6,
        privileged_obs_shape=capture.transition_privileged_obs.shape[1:],
        device=runner.device,
    )
    segment_storage.add_transition(
        FrontRESSegmentTransition(
            observations=capture.transition_obs,
            privileged_observations=capture.transition_privileged_obs,
            actions=capture.transition_actions,
            old_log_probs=capture.transition_log_probs,
            values=capture.transition_values,
            rewards=rewards,
            valid_mask=valid_mask,
            reset_mask=reset_mask,
            segment_ids=segment_ids,
            segment_source=segment_source,
            old_means=capture.transition_means,
            old_sigmas=capture.transition_sigmas,
            audit_transaction_id=capture.audit_transaction_id,
            audit_batch_signature=capture.audit_batch_signature,
            audit_identity_state=capture.audit_identity_state,
            priority_evidence=_trial_metadata_priority_evidence(
                runner,
                batch_size=batch_size,
                device=runner.device,
            ),
            transaction_metadata=frozen_transaction_metadata,
        )
    )
    reward_steps = _segment_storage_reward_steps(capture, batch_size=batch_size, device=runner.device)
    done_steps = _segment_storage_done_steps(capture, batch_size=batch_size, device=runner.device)
    if reward_steps is not None:
        alg = getattr(runner, "alg", None)
        segment_storage.compute_returns_and_advantages(
            reward_steps=reward_steps,
            done_steps=done_steps,
            horizon=capture.horizon_k
            if isinstance(capture.horizon_k, torch.Tensor)
            else max(1, int(getattr(alg, "frontres_segment_k", capture.rollout_k))),
            gamma=float(getattr(alg, "gamma", 1.0)),
        )
    return segment_storage


def _capture_averaged_rewards(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    reward = capture.reward_accum.reshape(-1).detach().float()
    if device is not None:
        reward = reward.to(device=device)
    if isinstance(capture.horizon_k, torch.Tensor):
        horizon = capture.horizon_k.to(device=reward.device, dtype=torch.float32).reshape(-1)
        if int(horizon.numel()) != int(reward.numel()):
            raise ValueError(f"capture horizon must have {int(reward.numel())} rows, got {int(horizon.numel())}")
        return reward / horizon.clamp_min(1.0)
    return reward / float(max(1, int(capture.rollout_k)))


def _capture_averaged_repair_scores(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if capture.repair_score_accum is None:
        raise RuntimeError(
            "paired Segment Replay gain requires repair-specific executability scores; "
            "generic env reward is not a valid fallback"
        )
    score = capture.repair_score_accum.reshape(-1).detach().float()
    if device is not None:
        score = score.to(device=device)
    if isinstance(capture.horizon_k, torch.Tensor):
        horizon = capture.horizon_k.to(device=score.device, dtype=torch.float32).reshape(-1)
        if int(horizon.numel()) != int(score.numel()):
            raise ValueError(f"capture horizon must have {int(score.numel())} rows, got {int(horizon.numel())}")
        return score / horizon.clamp_min(1.0)
    return score / float(max(1, int(capture.rollout_k)))


def _capture_paired_gain(capture: FrontRESSegmentLiveRolloutCapture) -> Any | None:
    n_train = max(0, int(capture.n_train))
    n_base = max(0, int(capture.n_base))
    n_candidate = max(0, int(capture.n_candidate))
    n = min(n_train, n_base)
    gain_module = _gain_module()
    if n <= 0 or capture.gain_config is None or gain_module is None:
        return None
    if capture.done_any is None or capture.survival_steps is None:
        raise RuntimeError("paired Gain requires done_any and survival_steps")
    if capture.transition_action_steps is None:
        raise RuntimeError("paired Gain requires full-6D action steps")
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    horizon = capture.horizon_k[:n].to(dtype=torch.float32) if isinstance(capture.horizon_k, torch.Tensor) else None
    action_valid_steps = _capture_action_valid_steps(capture)
    clean_action_steps = None
    clean_action_step_mask = None
    if capture.n_clean >= n and int(capture.transition_action_steps.shape[1]) >= clean_start + n:
        clean_action_steps = capture.transition_action_steps[:, clean_start : clean_start + n]
        if action_valid_steps is not None:
            clean_action_step_mask = action_valid_steps[:, clean_start : clean_start + n]
    temporal_mask = None
    repaired_zmp = _average_physics_steps(capture.physics_zmp_repaired_steps, horizon)
    noisy_zmp = _average_physics_steps(capture.physics_zmp_noisy_steps, horizon)
    repaired_contact = _average_physics_steps(capture.physics_contact_repaired_steps, horizon)
    noisy_contact = _average_physics_steps(capture.physics_contact_noisy_steps, horizon)
    if action_valid_steps is not None and capture.motion_clean_body_pos is not None:
        # Style owns the executed trajectory prefix. A terminal fall truncates
        # later frames, but it must not erase the finite pre-fall evidence.
        temporal_mask = action_valid_steps[:, :n].transpose(0, 1)
        expected_shape = tuple(capture.motion_clean_body_pos[:n].shape[:2])
        if tuple(temporal_mask.shape) != expected_shape:
            raise ValueError(
                "paired Style validity must match captured [B,T] motion evidence, "
                f"got {tuple(temporal_mask.shape)} for {expected_shape}"
            )
    elif horizon is not None and capture.motion_clean_body_pos is not None:
        temporal_mask = torch.arange(
            capture.motion_clean_body_pos.shape[1],
            device=capture.motion_clean_body_pos.device,
        ).view(1, -1) < horizon.to(capture.motion_clean_body_pos.device).view(-1, 1)
    return gain_module.compute_segment_gain(
        clean_positions=capture.motion_clean_body_pos[:n] if capture.motion_clean_body_pos is not None else None,
        repaired_positions=capture.motion_repaired_body_pos[:n] if capture.motion_repaired_body_pos is not None else None,
        noisy_positions=capture.motion_noisy_body_pos[:n] if capture.motion_noisy_body_pos is not None else None,
        clean_root_quaternions=capture.motion_clean_root_quat[:n] if capture.motion_clean_root_quat is not None else None,
        repaired_root_quaternions=capture.motion_repaired_root_quat[:n] if capture.motion_repaired_root_quat is not None else None,
        noisy_root_quaternions=capture.motion_noisy_root_quat[:n] if capture.motion_noisy_root_quat is not None else None,
        repaired_success=(~capture.done_any[:n]).reshape(-1),
        noisy_success=(~capture.done_any[base_start : base_start + n]).reshape(-1),
        repaired_survival=capture.survival_steps[:n].reshape(-1),
        noisy_survival=capture.survival_steps[base_start : base_start + n].reshape(-1),
        effective_horizon_k=horizon,
        repaired_zmp_margin=repaired_zmp,
        noisy_zmp_margin=noisy_zmp,
        repaired_contact=repaired_contact,
        noisy_contact=noisy_contact,
        action_steps=capture.transition_action_steps[:, :n],
        config=capture.gain_config,
        audit_transaction_id=capture.audit_transaction_id,
        audit_batch_signature=capture.audit_batch_signature,
        audit_identity_state=capture.audit_identity_state,
        action_step_mask=action_valid_steps[:, :n] if action_valid_steps is not None else None,
        clean_action_steps=clean_action_steps,
        clean_action_step_mask=clean_action_step_mask,
        temporal_mask=temporal_mask,
        # PPO row eligibility still excludes terminal rows in storage. Gain
        # retains their paired pre-fall evidence for diagnostics and replay.
        valid_mask=None,
    )


def _average_physics_steps(
    values: torch.Tensor | None,
    horizon: torch.Tensor | None,
) -> torch.Tensor | None:
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        return None
    mask = torch.isfinite(values)
    if horizon is not None and horizon.numel() == values.shape[0]:
        time = torch.arange(values.shape[1], device=values.device).view(1, -1)
        mask = mask & (time < horizon.to(values.device).view(-1, 1))
    count = mask.sum(dim=1)
    summed = torch.where(mask, values.float(), torch.zeros_like(values.float())).sum(dim=1)
    return torch.where(count > 0, summed / count.clamp_min(1), torch.full_like(summed, float("nan")))


def _capture_action_valid_steps(capture: FrontRESSegmentLiveRolloutCapture) -> torch.Tensor | None:
    """Build the executed-action mask from horizon and done-before-step state.

    Status: active, paired repair-cost boundary.
    Upstream: captured action steps, per-row horizon, and raw done trace.
    Downstream: `frontres_gain.compute_repair_cost`.
    Evidence: offline mixed-K/done contract; live population uses the same
    rollout trace but still requires S4 confirmation.
    Gap: none for the captured tensor schema; missing traces return None.
    """
    actions = capture.transition_action_steps
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        return None
    steps, batch_size = int(actions.shape[0]), int(actions.shape[1])
    if not isinstance(capture.horizon_k, torch.Tensor) or int(capture.horizon_k.numel()) != batch_size:
        return None
    time = torch.arange(steps, device=actions.device).view(-1, 1)
    valid = time < capture.horizon_k.to(device=actions.device, dtype=torch.long).reshape(1, -1)
    if isinstance(capture.done_steps, torch.Tensor):
        done_steps = capture.done_steps.to(device=actions.device, dtype=torch.bool)
        if tuple(done_steps.shape) != (steps, batch_size):
            raise ValueError(
                "segment done_steps must match captured action steps, "
                f"got {tuple(done_steps.shape)} for {(steps, batch_size)}"
            )
        done_before = torch.zeros_like(done_steps)
        if steps > 1:
            done_before[1:] = done_steps[:-1].cumsum(dim=0).bool()
        valid = valid & ~done_before
    return valid


def _segment_storage_rewards(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """选择正式 policy-row reward, 不把 legacy score 当作 Gain.

    Status: active.
    Upstream: paired live capture and FRS-GAIN-v002 component owner.
    Downstream: FrontRESSegmentRolloutStorage.rewards.
    Evidence: contract-confirmed by the formal Gain connectivity test.
    Gap: real rollout population remains live-only.
    """
    reward = _capture_averaged_rewards(capture, device=device)
    n_train = max(0, int(capture.n_train))
    n_candidate = max(0, int(capture.n_candidate))
    n_base = max(0, int(capture.n_base))
    base_start = n_train + n_candidate
    if n_train > 0 and n_base >= n_train and int(reward.numel()) >= base_start + n_train and batch_size == int(reward.numel()):
        paired_gain = _capture_paired_gain(capture)
        if paired_gain is not None:
            if int(paired_gain.gain_total.numel()) != n_train or not bool(torch.isfinite(paired_gain.gain_total).all().item()):
                raise RuntimeError("paired Gain has missing/non-finite training rows; inspect component evidence before PPO")
            reward = reward.clone()
            reward[:n_train] = paired_gain.gain_total.to(device=device)
            return reward
        if capture.gain_config is not None:
            raise RuntimeError(
                "FRS-GAIN formal policy-row reward evidence is unavailable; "
                "refusing legacy repair_score fallback"
            )
        repair_score = _capture_averaged_repair_scores(capture, device=device)
        if int(repair_score.numel()) != batch_size:
            raise ValueError(f"segment repair scores must have {batch_size} rows, got {int(repair_score.numel())}")
        reward = repair_score.clone()
        reward[:n_train] = repair_score[:n_train] - repair_score[base_start : base_start + n_train]
    if int(reward.numel()) != batch_size:
        raise ValueError(f"segment rewards must have {batch_size} rows, got {int(reward.numel())}")
    return reward


def _segment_storage_reward_steps(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """选择进入 K-step return 的 policy-row Gain trace.

    Status: active.
    Upstream: per-step paired Gain capture.
    Downstream: storage.compute_returns_and_advantages.
    Evidence: contract-confirmed by the storage and formal-route tests.
    Gap: live finite-value diversity remains unconfirmed.
    """
    if capture.reward_steps is None:
        return None
    reward_steps = capture.reward_steps.to(device=device, dtype=torch.float32)
    if reward_steps.ndim != 2:
        raise ValueError(f"segment reward_steps must be rank-2 [T, B], got {tuple(reward_steps.shape)}")
    if int(reward_steps.shape[1]) != batch_size:
        raise ValueError(f"segment reward_steps must have {batch_size} batch entries, got {int(reward_steps.shape[1])}")

    n_train = max(0, int(capture.n_train))
    n_candidate = max(0, int(capture.n_candidate))
    n_base = max(0, int(capture.n_base))
    base_start = n_train + n_candidate
    if n_train > 0 and n_base >= n_train and batch_size >= base_start + n_train:
        if capture.gain_config is not None:
            if capture.gain_steps is None:
                raise RuntimeError("paired Gain returns require per-step Gain evidence")
            gain_steps = capture.gain_steps.to(device=device, dtype=torch.float32)
            if gain_steps.ndim != 2 or int(gain_steps.shape[1]) != batch_size:
                raise ValueError(f"segment gain_steps must have shape [T, {batch_size}], got {tuple(gain_steps.shape)}")
            if not bool(torch.isfinite(gain_steps[:, :n_train]).all().item()):
                raise RuntimeError("paired Gain step evidence contains missing/non-finite training rows")
            reward_steps = reward_steps.clone()
            reward_steps[:, :n_train] = gain_steps[:, :n_train]
            return reward_steps
        if capture.repair_score_steps is None:
            raise RuntimeError(
                "paired Segment PPO returns require repair-specific executability steps; "
                "generic env reward is not a valid fallback"
            )
        reward_steps = capture.repair_score_steps.to(device=device, dtype=torch.float32)
        if reward_steps.ndim != 2 or int(reward_steps.shape[1]) != batch_size:
            raise ValueError(
                f"segment repair_score_steps must have shape [T, {batch_size}], got {tuple(reward_steps.shape)}"
            )
        reward_steps = reward_steps.clone()
        reward_steps[:, :n_train] = reward_steps[:, :n_train] - reward_steps[:, base_start : base_start + n_train]
    return reward_steps


def _segment_storage_done_steps(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if capture.done_steps is None:
        return None
    done_steps = capture.done_steps.to(device=device).bool()
    if done_steps.ndim != 2:
        raise ValueError(f"segment done_steps must be rank-2 [T, B], got {tuple(done_steps.shape)}")
    if int(done_steps.shape[1]) != batch_size:
        raise ValueError(f"segment done_steps must have {batch_size} batch entries, got {int(done_steps.shape[1])}")
    return done_steps


def _select_segment_transition_actions(
    runner: Any,
    *,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 2:
        raise ValueError(f"live segment transition actions must be rank-2, got {tuple(actions.shape)}")
    if actions.shape[-1] == 6:
        log_probs = runner.alg.transition.actions_log_prob.detach().clone().reshape(-1)
        if _should_print_once_or_verbose(runner.alg, "_frontres_segment_live_probe_trace_printed"):
            print(
                "[FrontRES Segment Live Probe Trace] "
                f"raw_action_shape={tuple(actions.shape)} "
                f"segment_action_shape={tuple(actions.shape)} "
                f"log_prob_shape={tuple(log_probs.shape)} "
                "semantic=storage_uses_native_6d_delta_se_policy",
                flush=True,
            )
        return actions, log_probs
    if actions.shape[-1] < 6:
        raise ValueError(f"live segment transition actions must expose at least 6 Delta SE dims, got {tuple(actions.shape)}")

    segment_actions = actions[:, :6]
    action_mean = getattr(runner.alg.transition, "action_mean", None)
    action_sigma = getattr(runner.alg.transition, "action_sigma", None)
    if action_mean is not None and action_sigma is not None:
        log_probs = _evaluate_segment_delta_se_log_prob_from_stats(
            runner.alg.policy,
            segment_actions,
            action_mean,
            action_sigma,
        ).detach().clone().reshape(-1)
    elif hasattr(runner.alg.policy, "get_actions_log_prob_selected"):
        log_probs = runner.alg.policy.get_actions_log_prob_selected(actions, list(range(6))).detach().clone().reshape(-1)
    else:
        raise ValueError("12D live segment actions require action_mean/action_sigma to rebuild 6D log_prob.")
    if _should_print_once_or_verbose(runner.alg, "_frontres_segment_live_probe_trace_printed"):
        print(
            "[FrontRES Segment Live Probe Trace] "
            f"raw_action_shape={tuple(actions.shape)} "
            f"segment_action_shape={tuple(segment_actions.shape)} "
            f"log_prob_shape={tuple(log_probs.shape)} "
            "semantic=storage_uses_first_6_delta_se_dims",
            flush=True,
        )
    return segment_actions, log_probs


def _select_executed_segment_actions(
    runner: Any,
    *,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Return the full-6D action actually stored after baseline overrides.

    This is intentionally separate from `_select_segment_transition_actions`:
    the latter reconstructs old log-probabilities from raw policy statistics,
    while Repair Cost must observe the executed transition tuple. Candidate,
    baseline, and Clean rows are therefore zero after the baseline override.
    """
    transition_actions = getattr(getattr(runner, "alg", None), "transition", None)
    transition_actions = getattr(transition_actions, "actions", None)
    if isinstance(transition_actions, torch.Tensor) and transition_actions.shape == actions.shape:
        selected = transition_actions
    else:
        selected, _ = _select_segment_transition_actions(runner, actions=actions)
    if selected.ndim != 2 or selected.shape[-1] < 6:
        raise ValueError(f"executed Segment action must expose full 6D Delta SE, got {tuple(selected.shape)}")
    return selected[:, :6].detach().clone()


def _motion_perturber_from_runner(runner: Any) -> Any | None:
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    command_manager = getattr(env_raw, "command_manager", None)
    terms = getattr(command_manager, "_terms", {}) if command_manager is not None else {}
    motion_command = terms.get("motion") if hasattr(terms, "get") else None
    if motion_command is None:
        motion_command = getattr(env_raw, "command", None)
    return getattr(motion_command, "perturber", None)


def _snapshot_frontres_perturbation_rp(runner: Any, *, num_envs: int) -> torch.Tensor | None:
    perturber = _motion_perturber_from_runner(runner)
    roll_state = getattr(perturber, "_roll_state", None)
    pitch_state = getattr(perturber, "_pitch_state", None)
    if not isinstance(roll_state, torch.Tensor) or not isinstance(pitch_state, torch.Tensor):
        return None
    count = max(0, min(int(num_envs), int(roll_state.numel()), int(pitch_state.numel())))
    if count <= 0:
        return None
    rp = torch.stack(
        (
            roll_state[:count].detach().float(),
            pitch_state[:count].detach().float(),
        ),
        dim=-1,
    )
    iid_event_rp = getattr(perturber, "_iid_event_rp", None)
    if isinstance(iid_event_rp, torch.Tensor) and iid_event_rp.ndim == 2 and int(iid_event_rp.shape[0]) >= count:
        rp = rp + iid_event_rp[:count, :2].detach().float()
    family_masks = getattr(perturber, "_family_masks", None)
    if isinstance(family_masks, dict) and isinstance(family_masks.get("local_rp"), torch.Tensor):
        mask = family_masks["local_rp"][:count].to(device=rp.device, dtype=torch.bool)
        rp = rp * mask.to(dtype=rp.dtype).view(-1, 1)
    baseline_mask = getattr(perturber, "_baseline_mask", None)
    if isinstance(baseline_mask, torch.Tensor) and int(baseline_mask.numel()) >= count:
        mask = ~baseline_mask[:count].to(device=rp.device, dtype=torch.bool)
        rp = rp * mask.to(dtype=rp.dtype).view(-1, 1)
    return rp.detach().clone()


def _expand_short_counterfactual_vector(
    tensor: torch.Tensor,
    *,
    name: str,
    batch_size: int,
) -> torch.Tensor:
    rows = int(tensor.numel())
    if rows == int(batch_size):
        return tensor
    if rows > 0 and int(batch_size) % rows == 0:
        return tensor.repeat(int(batch_size) // rows)
    raise ValueError(f"{name} must have {batch_size} rows, got {rows}")


def _expand_short_counterfactual_tuple(value: Any, *, name: str, batch_size: int) -> tuple[str, ...]:
    items = tuple(str(item) for item in value)
    rows = len(items)
    if rows == int(batch_size):
        return items
    if rows > 0 and int(batch_size) % rows == 0:
        return items * (int(batch_size) // rows)
    raise ValueError(f"{name} must have {batch_size} rows, got {rows}")


def _current_reset_success_mask(runner: Any, *, batch_size: int, device: torch.device | str) -> torch.Tensor:
    result = getattr(runner, "_frontres_segment_live_current_reset_result", None)
    if result is None:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    success_mask = getattr(result, "success_mask", None)
    if success_mask is None:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    success_mask = success_mask.to(device=device).bool().reshape(-1)
    success_mask = _expand_short_counterfactual_vector(
        success_mask,
        name="segment reset success mask",
        batch_size=batch_size,
    )
    return success_mask.detach()


def run_frontres_segment_single_update(runner: Any, storage_batch: Any) -> object:
    # QUALITY-UPDATE-01: 检查 advantage/log-prob -> optimizer step -> accepted policy delta.
    # Result: PENDING_Q_EVIDENCE.
    # B1: step 前冻结 old/new distribution、advantage sign 与 held-out identity.
    # B2: backward/optimizer/trust 顺序记录 parameter 与 per-dim mean delta.
    # B3: accepted/rollback 后比较正负 advantage log-prob 方向.
    """Run one Stage 3 Segment PPO update on the isolated live Segment path.

    Status: active Segment Replay update boundary.
    Upstream: live probe/update loop passes storage_batch from rollout evidence.
    Downstream: FrontRESSegmentPPOBatch -> compute_frontres_segment_ppo_loss -> optimizer.step -> post diagnostics.
    Evidence: contract-confirmed by frontres_segment_live_single_update_contract.py.
    Gap: one fake/live-boundary update does not prove long live training quality.
    """
    runner.train_mode()
    # B1: Convert storage evidence into the algorithm-owned batch contract.
    ppo_batch = storage_batch.to_ppo_batch(FrontRESSegmentPPOBatch)
    policy_adapter = FrontRESSegmentLivePolicyAdapter(
        runner.alg,
        privileged_observations=storage_batch.privileged_observations,
    )
    warmup_phase = frontres_segment_warmup_phase(
        iteration=int(getattr(runner, "current_learning_iteration", 0)),
        critic_warmup_iterations=int(getattr(runner.alg, "frontres_segment_critic_warmup_iterations", 0)),
        actor_warmup_iterations=int(getattr(runner.alg, "frontres_segment_actor_warmup_iterations", 0)),
    )
    ppo_cfg = FrontRESSegmentPPOConfig(
        clip_param=float(getattr(runner.alg, "clip_param", 0.2)),
        value_clip_param=float(getattr(runner.alg, "clip_param", 0.2)),
        value_loss_coef=float(getattr(runner.alg, "value_loss_coef", 1.0)),
        entropy_coef=float(getattr(runner.alg, "entropy_coef", 0.0)),
        use_clipped_value_loss=bool(getattr(runner.alg, "use_clipped_value_loss", True)),
        advantage_normalization=str(getattr(runner.alg, "frontres_segment_advantage_normalization", "scale_only")),
        actor_loss_weight=warmup_phase.actor_loss_weight,
    )
    # B2: First forward is the pre-step loss and MOSAIC-style old/new KL source.
    ppo_result = compute_frontres_segment_ppo_loss(policy_adapter, ppo_batch, ppo_cfg)
    credit_result_path = str(getattr(runner, "_frontres_policy_quality_q2d_credit_result", "") or "")
    if credit_result_path and not Path(credit_result_path).exists():
        from rsl_rl.frontres.frontres_policy_quality_q2d import write_q2d_credit_tuple

        if ppo_batch.old_means is None or ppo_batch.old_sigmas is None:
            raise ValueError("Q2-D credit capture requires rollout old_means and old_sigmas")
        raw_actions = _segment_delta_se_log_prob_parts(
            runner.alg.policy,
            ppo_batch.actions,
            ppo_batch.old_means,
            ppo_batch.old_sigmas,
        )["raw_actions"]
        # QUALITY-CREDIT-01: capture the finalized Gain -> return -> advantage tuple
        # at the last read-only boundary before the official PPO optimizer step.
        write_q2d_credit_tuple(
            result_path=credit_result_path,
            raw_actions=raw_actions,
            bounded_actions=ppo_batch.actions,
            old_means=ppo_batch.old_means,
            old_sigmas=ppo_batch.old_sigmas,
            gains=storage_batch.rewards,
            returns=ppo_batch.returns,
            advantages=ppo_batch.advantages,
            valid_mask=ppo_batch.valid_mask,
            segment_ids=ppo_batch.segment_ids,
            audit_transaction_id=storage_batch.audit_transaction_id,
            audit_batch_signature=storage_batch.audit_batch_signature,
            audit_identity_state=storage_batch.audit_identity_state,
        )
    pre_step_lr_diagnostics = _apply_segment_adaptive_learning_rate(
        runner.alg,
        ppo_result,
        allow_increase=False,
    )
    optimizer_params, param_snapshots = _optimizer_parameter_snapshots(runner.alg.policy, runner.alg.optimizer)
    optimizer_state_snapshot = copy.deepcopy(runner.alg.optimizer.state_dict())
    grad_norm = 0.0
    post_update_diagnostics: dict[str, Any] = {}
    rejected_lr_diagnostics: dict[str, Any] = {}
    rejected_count = 0
    accepted = True
    max_retries = max(0, int(getattr(runner.alg, "frontres_segment_trust_region_max_retries", 2)))
    rollback_enabled = bool(getattr(runner.alg, "frontres_segment_trust_region_rollback", True))
    schedule = str(getattr(runner.alg, "schedule", "fixed")).lower()
    if ppo_result.should_step:
        # B3: The optimizer step is accepted only after a post-step same-batch
        # diagnostic pass says the policy distribution stayed inside the trust region.
        for attempt in range(max_retries + 1):
            if attempt > 0:
                ppo_result = compute_frontres_segment_ppo_loss(policy_adapter, ppo_batch, ppo_cfg)
            runner.alg.optimizer.zero_grad()
            ppo_result.total_loss.backward()
            if warmup_phase.name == "critic_only":
                _clear_noncritic_grads(runner.alg.policy, optimizer_params)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                (param for _, param in optimizer_params),
                float(getattr(runner.alg, "max_grad_norm", 1.0)),
            )
            grad_norm = float(grad_norm_tensor.detach().cpu().item())
            runner.alg.optimizer.step()
            post_update_diagnostics = _post_update_segment_ppo_diagnostics(policy_adapter, ppo_batch, ppo_cfg)
            post_kl = float(post_update_diagnostics["post_update_approx_kl"])
            desired_kl = getattr(runner.alg, "desired_kl", None)
            reject = (
                rollback_enabled
                and desired_kl is not None
                and schedule == "adaptive"
                and math.isfinite(post_kl)
                and post_kl > float(desired_kl) * 2.0
            )
            if reject:
                _restore_optimizer_parameters(optimizer_params, param_snapshots)
                runner.alg.optimizer.load_state_dict(optimizer_state_snapshot)
                rejected_count += 1
                rejected_lr_diagnostics = _apply_segment_adaptive_learning_rate(
                    runner.alg,
                    ppo_result,
                    kl_mean=post_kl,
                )
                if attempt < max_retries:
                    continue
                accepted = False
            # B4: Keep legacy ratio_mean/ratio_max as post-step aliases for
            # existing logs, while explicit pre/post fields carry the white-box timing.
            object.__setattr__(ppo_result, "approx_kl", post_kl)
            object.__setattr__(ppo_result, "clip_frac", float(post_update_diagnostics["post_update_clip_frac"]))
            object.__setattr__(
                ppo_result,
                "ratio_mean",
                float(post_update_diagnostics["post_update_clamped_ratio_mean"]),
            )
            object.__setattr__(
                ppo_result,
                "ratio_max",
                float(post_update_diagnostics["post_update_clamped_ratio_max"]),
            )
            break
    if rejected_lr_diagnostics and not accepted:
        lr_diagnostics = rejected_lr_diagnostics
    else:
        lr_diagnostics = pre_step_lr_diagnostics
    diagnostics = _parameter_delta_stats(optimizer_params, param_snapshots)
    diagnostics["param_grad_norm"] = grad_norm
    diagnostics["trust_region_rejected_count"] = rejected_count
    diagnostics["trust_region_accepted"] = int(bool(accepted))
    diagnostics["trust_region_rollback_enabled"] = int(bool(rollback_enabled))
    diagnostics["trust_region_max_retries"] = max_retries
    diagnostics["trust_region_schedule_adaptive"] = int(schedule == "adaptive")
    diagnostics["trust_region_schedule"] = schedule
    diagnostics["warmup_phase"] = warmup_phase.name
    diagnostics["warmup_phase_iteration"] = warmup_phase.phase_iteration
    diagnostics["actor_loss_weight"] = warmup_phase.actor_loss_weight
    for key, value in pre_step_lr_diagnostics.items():
        diagnostics[f"mosaic_pre_step_{key}"] = value
    for key, value in rejected_lr_diagnostics.items():
        diagnostics[f"segment_reject_{key}"] = value
    diagnostics.update(post_update_diagnostics)
    diagnostics.update(lr_diagnostics)
    _attach_ppo_update_diagnostics(ppo_result, diagnostics)
    # AUDIT-PPO-01: 检查 warmup/PPO/KL/Frozen GMT, 位于 optimizer diagnostics -> live summary.
    # Result: E68 LIVE OBSERVED: actor_warmup weight=0.002..0.040, accepted
    # updates, finite post-update KL, and frozen GMT ownership.
    print_ppo_audit(runner, result=ppo_result)
    runner.eval_mode()
    return ppo_result


def _resolve_probe_modes(runner: Any) -> tuple[bool, bool]:
    single_update = bool(
        runner._frontres_segment_replay_boundary.live_single_update_only
        or runner._frontres_segment_replay_boundary.live_update_loop_only
        or runner._frontres_segment_replay_boundary.live_train_enabled
    )
    storage_write = bool(runner._frontres_segment_replay_boundary.live_storage_write_only or single_update)
    if not (runner._frontres_segment_replay_boundary.live_probe_only or storage_write):
        raise ValueError(
            "FrontRES Segment live probe requires frontres_segment_live_probe_only=True "
            "or frontres_segment_live_storage_write_only=True "
            "or frontres_segment_live_single_update_only=True "
            "or frontres_segment_live_update_loop_only=True."
        )
    return single_update, storage_write


def _append_fixed_noisy_actor_context(runner: Any, obs: torch.Tensor) -> torch.Tensor:
    append = getattr(runner, "_append_frontres_fixed_noisy_future_context", None)
    if callable(append):
        return append(obs)
    batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    if isinstance(getattr(batch, "frontres_fixed_noisy_tape", None), torch.Tensor):
        raise RuntimeError("fixed Noisy Segment Replay requires runner actor-context connectivity")
    return obs


def _uses_v015_future_intent_route_local(runner: Any) -> bool:
    """本地判定避免 legacy probe stub 依赖新的 rollout-step symbol."""

    alg = getattr(runner, "alg", None)
    offsets = tuple(int(value) for value in (getattr(alg, "frontres_future_offsets", ()) or ()))
    return bool(offsets) or getattr(runner, "_frontres_future_intent_layout", None) is not None


def _read_live_observations(runner: Any) -> FrontRESSegmentLiveObservations:
    """Read env-owned observations and apply the active actor-context/normalizer route.

    Status: R5 offline S2 contract-confirmed; simulator/live timing remains unconfirmed.
    """

    obs, extras = runner.env.get_observations()
    obs_dict = extras.get("observations", {})
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type]
    privileged_obs = obs_dict.get(runner.privileged_obs_type, obs)
    teacher_obs = obs_dict.get(runner.teacher_obs_type)
    if teacher_obs is None:
        teacher_obs = privileged_obs
    ref_vel_estimator_obs = obs_dict.get(runner.ref_vel_estimator_obs_type)

    obs = obs.to(runner.device)
    raw_obs_dim = int(obs.shape[-1])
    uses_v015_future_intent = _uses_v015_future_intent_route_local(runner)
    # v015 的 actor 只能看到 deployment-q29 intent. 旧 65D fixed tape 只保留给
    # 历史路径, 两者不能在同一个 observation 中拼接.
    if uses_v015_future_intent:
        obs = _append_future_intent_actor_context(runner, obs)
    else:
        obs = _append_fixed_noisy_actor_context(runner, obs)
    combined_obs_dim = int(obs.shape[-1])
    obs = runner._apply_obs_normalizer(obs)
    if uses_v015_future_intent:
        policy = getattr(getattr(runner, "alg", None), "policy", None)
        runner._frontres_v015_observation_route_trace = {
            "role_row_count": int(obs.shape[0]),
            "current_command_dim": 0,
            "raw_observation_dim": raw_obs_dim,
            "q29_tail_dim": combined_obs_dim - raw_obs_dim,
            "combined_observation_dim": combined_obs_dim,
            "normalized_observation_dim": int(obs.shape[-1]),
            "femr_visible_dim": int(getattr(policy, "num_frontres_obs", 0) or 0),
            "gmt_suffix_dim": int(getattr(runner, "_frontres_gmt_obs_dim", 0) or 0),
            "gmt_input_dim": 0,
            "post_advance_gmt_read_count": 0,
        }
    privileged_obs = runner.privileged_obs_normalizer(privileged_obs.to(runner.device))
    teacher_obs = runner.teacher_obs_normalizer(teacher_obs.to(runner.device))
    if ref_vel_estimator_obs is not None:
        ref_vel_estimator_obs = ref_vel_estimator_obs.to(runner.device)
    return FrontRESSegmentLiveObservations(
        obs=obs,
        privileged_obs=privileged_obs,
        teacher_obs=teacher_obs,
        ref_vel_estimator_obs=ref_vel_estimator_obs,
    )


def _read_v015_frozen_gmt_observations(
    runner: Any,
    obs: torch.Tensor,
    infos: dict[str, Any],
    *,
    frozen_frontres_prefix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prepare a GMT execution observation without reopening actor-only q29 context."""

    obs_dict = infos.get("observations", {}) if isinstance(infos, dict) else {}
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type].to(runner.device)
    else:
        obs = obs.to(runner.device)
    # R5 exact route: t 已经验证并归一化 158D FEMR prefix. K 内 actor 冻结, 因此
    # 只允许重新读取/归一化 fresh 770D GMT suffix, 不得重新打开 q29 actor snapshot.
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    frontres_dim = int(getattr(policy, "num_frontres_obs", 0) or 0)
    gmt_dim = int(getattr(runner, "_frontres_gmt_obs_dim", 0) or 0)
    if isinstance(frozen_frontres_prefix, torch.Tensor) and frontres_dim > 0 and gmt_dim > 0:
        prefix = frozen_frontres_prefix.to(device=runner.device)
        if tuple(prefix.shape) != (int(obs.shape[0]), frontres_dim):
            raise RuntimeError(
                "v015 frozen-GMT route requires the t-time normalized FEMR prefix "
                f"[{int(obs.shape[0])},{frontres_dim}], got {tuple(prefix.shape)}"
            )
        if int(obs.shape[-1]) < gmt_dim:
            raise RuntimeError("v015 frozen-GMT raw observation is smaller than the frozen GMT suffix")
        gmt_raw = obs[..., -gmt_dim:]
        normalize_gmt = getattr(runner, "obs_normalizer", None)
        if not callable(normalize_gmt):
            raise RuntimeError("v015 frozen-GMT route requires the frozen GMT normalizer")
        gmt_obs = normalize_gmt(gmt_raw)
        combined = torch.cat([prefix.to(dtype=gmt_obs.dtype), gmt_obs], dim=-1)
        if int(combined.shape[-1]) != int(getattr(policy, "num_actor_obs", 0) or 0):
            raise RuntimeError("v015 frozen-GMT route lost the exact FEMR/GMT observation authority")
        trace = dict(getattr(runner, "_frontres_v015_observation_route_trace", {}) or {})
        trace["gmt_input_dim"] = int(gmt_obs.shape[-1])
        trace["post_advance_gmt_read_count"] = int(trace.get("post_advance_gmt_read_count", 0)) + 1
        runner._frontres_v015_observation_route_trace = trace
        return combined

    if bool(getattr(getattr(runner, "alg", None), "frontres_v015_formal_transaction_enabled", False)):
        raise RuntimeError("v015 formal frozen-GMT route requires the exact R3 158D/770D authority")
    # Candidate-only legacy fixtures without R3 dimensions retain their local test contract.
    obs = _append_future_intent_actor_context(runner, obs)
    return runner._apply_obs_normalizer(obs)


def _require_v015_one_action_k_layout(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    pair_layout: Any,
) -> tuple[Any, dict[str, object], torch.Tensor]:
    """Fail closed unless the candidate collector sees exactly the sealed two-role layout."""

    n_repair = int(getattr(pair_layout, "n_train", 0))
    n_noisy = int(getattr(pair_layout, "n_base", 0))
    total = int(observations.obs.shape[0])
    if (
        n_repair <= 0
        or n_noisy != n_repair
        or int(getattr(pair_layout, "n_candidate", 0)) != 0
        or int(getattr(pair_layout, "n_clean", 0)) != 0
        or total != n_repair + n_noisy
    ):
        raise RuntimeError("v015 one-action K collector requires only equal Repair/Noisy role rows")
    command = _frontres_motion_command(runner)
    snapshot = command.frontres_local_scenario_snapshot(
        torch.arange(total, device=observations.obs.device, dtype=torch.long)
    )
    roles = tuple(snapshot["roles"])
    expected_roles = ("repair",) * n_repair + ("noisy",) * n_noisy
    if roles != expected_roles:
        raise RuntimeError(
            "v015 one-action K collector requires Repair rows followed by Noisy rows; "
            f"got roles={roles}"
        )
    repair_rows = torch.arange(n_repair, device=observations.obs.device, dtype=torch.long)
    return command, snapshot, repair_rows


def _capture_v015_post_t_executed_q29(command: Any, *, role_count: int, device: torch.device) -> torch.Tensor:
    """读取 post-action robot articulation state, 绝不读取 command/reference q29."""

    executed_q29 = getattr(command, "robot_joint_pos", None)
    if not isinstance(executed_q29, torch.Tensor):
        raise RuntimeError("v015 Gain capture requires command.robot_joint_pos after the t action")
    executed_q29 = executed_q29.detach().to(device=device, dtype=torch.float32).clone()
    if tuple(executed_q29.shape) != (role_count, 29):
        raise RuntimeError(
            "v015 Gain capture requires post-t robot joint state [N,29], "
            f"got {tuple(executed_q29.shape)}"
        )
    return executed_q29


def _v015_intent_provenance_rows(snapshot: dict[str, object], *, role_count: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """只从 sealed command snapshot 提取 deployment-q29 provenance."""

    rows = snapshot.get("provenance")
    if not isinstance(rows, tuple) or len(rows) != role_count:
        raise RuntimeError("v015 Gain capture requires one sealed q29 provenance row per scored role")
    provenance: list[str] = []
    source_rows: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("v015 Gain capture requires mapping-like q29 provenance")
        value = str(row.get("intent_q29_provenance", ""))
        source = str(row.get("intent_q29_source", ""))
        lowered = source.lower()
        if value != "deployment_noisy_q29" or not source or any(
            token in lowered for token in ("clean", "root", "global")
        ):
            raise RuntimeError("v015 Gain capture rejects Clean/root/global q29 provenance")
        provenance.append(value)
        source_rows.append(source)
    return tuple(provenance), tuple(source_rows)


def collect_frontres_v015_one_action_k_evidence(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    *,
    pair_layout: Any,
) -> FrontRESV015OneActionKEvidence:
    """Capture one action followed by frozen-GMT K evidence for v015.

    This bypasses the legacy repeated-action loop. It is consumed by the
    explicit pre-live sentinel, formal v015 held-out evaluator, and CPU
    contracts; generic legacy storage/update paths remain isolated.
    """

    command, snapshot, repair_rows = _require_v015_one_action_k_layout(runner, observations, pair_layout)
    n_repair = int(repair_rows.numel())
    frontres_dim = int(getattr(getattr(runner.alg, "policy", None), "num_frontres_obs", 0) or 0)
    # B1: Sentinel 和 ordinary training 从 command owner 读取当前 GMT command
    # 维度. Held-out quality 不消费 training observation trace, 不应触发该读取.
    trace_current_command = bool(
        getattr(runner.alg, "frontres_v015_local_sentinel_only", False)
        or getattr(runner.alg, "frontres_segment_live_train_enabled", False)
    )
    if trace_current_command:
        current_command = command.command
        if not isinstance(current_command, torch.Tensor) or current_command.ndim != 2:
            raise RuntimeError("v015 formal one-action-K requires a rank-2 current GMT command")
        trace = dict(getattr(runner, "_frontres_v015_observation_route_trace", {}) or {})
        trace["current_command_dim"] = int(current_command.shape[-1])
        runner._frontres_v015_observation_route_trace = trace
    frozen_frontres_prefix = (
        observations.obs[:, :frontres_dim].detach().clone()
        if frontres_dim > 0
        else None
    )
    execution_started = False
    actor_forward_count = 0
    later_femr_action_count = 0
    try:
        t_plan = prepare_frontres_v015_one_action_at_t(
            runner,
            obs=observations.obs,
            privileged_obs=observations.privileged_obs,
            teacher_obs=observations.teacher_obs,
            ref_vel_estimator_obs=observations.ref_vel_estimator_obs,
            iteration=int(getattr(runner, "current_learning_iteration", 0)),
            n_repair=n_repair,
            n_noisy=int(getattr(pair_layout, "n_base", 0)),
        )
        actor_forward_count += 1
        transition = getattr(runner.alg, "transition", None)
        required = (
            "observations",
            "privileged_observations",
            "actions_log_prob",
            "values",
            "action_mean",
            "action_sigma",
        )
        if transition is None or any(not isinstance(getattr(transition, name, None), torch.Tensor) for name in required):
            raise RuntimeError("v015 one-action K collector requires the t policy tuple before frozen GMT execution")
        policy_actions = t_plan.actions.index_select(0, repair_rows).detach().clone()
        policy_observations = transition.observations.index_select(0, repair_rows).detach().clone()
        policy_privileged_observations = (
            transition.privileged_observations.index_select(0, repair_rows).detach().clone()
        )
        policy_log_probs = transition.actions_log_prob.index_select(0, repair_rows).detach().clone().reshape(-1)
        policy_values = transition.values.index_select(0, repair_rows).detach().clone().reshape(-1)
        policy_means = transition.action_mean.index_select(0, repair_rows).detach().clone()[:, :6]
        policy_sigmas = transition.action_sigma.index_select(0, repair_rows).detach().clone()[:, :6]
        if tuple(policy_actions.shape) != tuple(policy_means.shape) or tuple(policy_actions.shape) != tuple(policy_sigmas.shape):
            raise RuntimeError("v015 one-action K collector requires aligned full-6D old policy statistics")
        if (
            policy_privileged_observations.ndim != 2
            or int(policy_privileged_observations.shape[0]) != n_repair
            or int(policy_privileged_observations.shape[1]) <= 0
        ):
            raise RuntimeError("v015 one-action K collector requires one non-empty t critic observation per Repair row")
        trace = dict(getattr(runner, "_frontres_v015_observation_route_trace", {}) or {})
        trace["critic_observation_dim"] = int(policy_privileged_observations.shape[-1])
        runner._frontres_v015_observation_route_trace = trace

        _raw_obs, _rewards, t_dones, _infos = runner.env.step(t_plan.env_actions.to(runner.env.device))
        t_dones = t_dones.to(runner.device).detach().bool().reshape(-1)
        if int(t_dones.numel()) != int(t_plan.env_actions.shape[0]):
            raise RuntimeError("v015 one-action K collector requires one t done flag per scored role")
        executed_q29_t = _capture_v015_post_t_executed_q29(
            command,
            role_count=int(t_plan.env_actions.shape[0]),
            device=runner.device,
        )
        survival_steps = torch.zeros_like(t_dones, dtype=torch.float32)
        done_any = t_dones.detach().clone()
        begin = getattr(command, "begin_frontres_local_scenario_k_execution", None)
        if not callable(begin):
            raise RuntimeError("v015 one-action K collector requires command Clean-continuation lifecycle ownership")
        begin()
        execution_started = True

        continuation_frames: list[torch.Tensor] = []
        valid_frames: list[torch.Tensor] = []
        gmt_action_frames: list[torch.Tensor] = []
        zmp_repaired_frames: list[torch.Tensor] = []
        zmp_noisy_frames: list[torch.Tensor] = []
        contact_repaired_frames: list[torch.Tensor] = []
        contact_noisy_frames: list[torch.Tensor] = []
        expected_support_frames: list[torch.Tensor] = []
        physics_pair_valid_frames: list[torch.Tensor] = []
        survival_repaired_frames: list[torch.Tensor] = []
        survival_noisy_frames: list[torch.Tensor] = []
        lean_repaired_frames: list[torch.Tensor] = []
        lean_noisy_frames: list[torch.Tensor] = []
        quality_trace_enabled = hasattr(runner, "_frontres_v015_quality_action_route")
        horizon_k = snapshot["horizon_k"].detach().long().clone()
        for _offset in range(int(horizon_k.max().item())):
            if getattr(runner, "_frontres_v015_one_action_k_phase", None) != "frozen":
                raise RuntimeError("v015 one-action K collector lost its frozen-FEMR phase before GMT continuation")

            def post_advance_gmt_observation() -> torch.Tensor:
                fresh_obs, fresh_infos = runner.env.get_observations()
                return _read_v015_frozen_gmt_observations(
                    runner,
                    fresh_obs,
                    fresh_infos,
                    frozen_frontres_prefix=frozen_frontres_prefix,
                )

            frozen_plan = prepare_frontres_v015_frozen_gmt_step(
                runner,
                gmt_observation_provider=post_advance_gmt_observation,
            )
            continuation_frames.append(frozen_plan.continuation)
            valid_frames.append(frozen_plan.valid_mask)
            gmt_action_frames.append(frozen_plan.env_actions)
            _raw_obs, _rewards, frozen_dones, _infos = runner.env.step(frozen_plan.env_actions.to(runner.env.device))
            frozen_dones = frozen_dones.to(runner.device).detach().bool().reshape(-1)
            valid = frozen_plan.valid_mask.to(device=runner.device, dtype=torch.bool).reshape(-1)
            if int(frozen_dones.numel()) != int(valid.numel()):
                raise RuntimeError("v015 one-action K collector requires one frozen-GMT done flag per role")
            alive = valid & (~done_any)
            physics_frame = _capture_physics_frame(runner, pair_layout)
            if physics_frame is None:
                raise RuntimeError(
                    "v015 one-action K collector requires paired ZMP/contact evidence on every executable K step"
                )
            pair_valid = alive[:n_repair] & alive[n_repair : 2 * n_repair]
            survival_repaired_frames.append(alive[:n_repair].detach().clone())
            survival_noisy_frames.append(alive[n_repair : 2 * n_repair].detach().clone())
            nan = torch.full((n_repair,), float("nan"), device=runner.device, dtype=torch.float32)
            expected_support, contact_repaired, contact_noisy = physics_frame[2:]
            for name, frame, destination in (
                ("expected_support", expected_support, expected_support_frames),
                ("contact_repaired", contact_repaired, contact_repaired_frames),
                ("contact_noisy", contact_noisy, contact_noisy_frames),
            ):
                frame = frame.detach().to(device=runner.device, dtype=torch.float32)
                if tuple(frame.shape) != (n_repair, 2):
                    raise RuntimeError(f"v015 one-action K collector received invalid {name} shape {tuple(frame.shape)}")
                destination.append(frame.clone())
            expected_loaded = expected_support.bool().any(dim=-1)
            frame_names = (
                ("zmp_repaired", physics_frame[0], contact_repaired, zmp_repaired_frames),
                ("zmp_noisy", physics_frame[1], contact_noisy, zmp_noisy_frames),
            )
            for name, frame, actual_contact, destination in frame_names:
                frame = frame.detach().to(device=runner.device, dtype=torch.float32).reshape(-1)
                applicable = pair_valid & expected_loaded & actual_contact.bool().any(dim=-1)
                finite = torch.isfinite(frame)
                invalid_physical = pair_valid & ~applicable
                if int(frame.numel()) != n_repair or not bool(finite[applicable].all()) or bool(finite[invalid_physical].any()):
                    raise RuntimeError(
                        f"v015 one-action K collector received invalid loaded-support applicability for {name}"
                    )
                destination.append(torch.where(applicable, frame, nan).detach().clone())
            physics_pair_valid_frames.append(pair_valid.detach().clone())
            if quality_trace_enabled:
                lean_frame = _capture_v015_quality_lateral_lean_frame(runner, pair_layout)
                if lean_frame is None:
                    raise RuntimeError("v015 quality requires evaluation-only paired lateral-lean evidence")
                for frame, destination in zip(lean_frame, (lean_repaired_frames, lean_noisy_frames), strict=True):
                    frame = frame.detach().to(device=runner.device, dtype=torch.float32).reshape(-1)
                    if int(frame.numel()) != n_repair or not bool(torch.isfinite(frame[pair_valid]).all()):
                        raise RuntimeError("v015 quality received invalid lateral-lean evidence")
                    destination.append(torch.where(pair_valid, frame, nan).detach().clone())
            survival_steps = survival_steps + alive.to(dtype=survival_steps.dtype)
            done_any = done_any | (frozen_dones & alive)

        intent_q29_provenance, intent_q29_source = _v015_intent_provenance_rows(
            snapshot,
            role_count=int(t_plan.env_actions.shape[0]),
        )

        evidence = FrontRESV015OneActionKEvidence(
            policy_observations=policy_observations,
            policy_privileged_observations=policy_privileged_observations,
            policy_actions=policy_actions,
            policy_log_probs=policy_log_probs,
            policy_values=policy_values,
            policy_means=policy_means,
            policy_sigmas=policy_sigmas,
            policy_row_indices=repair_rows.detach().clone(),
            t_env_actions=t_plan.env_actions.detach().clone(),
            continuation=torch.stack(continuation_frames, dim=0),
            continuation_valid_mask=torch.stack(valid_frames, dim=0),
            frozen_gmt_env_actions=torch.stack(gmt_action_frames, dim=0),
            actor_forward_count=actor_forward_count,
            later_femr_action_count=later_femr_action_count,
            horizon_k=horizon_k,
            scenario_ids=tuple(snapshot["scenario_ids"]),
            noisy_segment_hashes=tuple(snapshot["noisy_segment_hashes"]),
            x_t_identities=tuple(snapshot["x_t_identities"]),
            roles=tuple(snapshot["roles"]),
            intent_q29=snapshot["intent_q29"].detach().clone(),
            intent_q29_provenance=intent_q29_provenance,
            intent_q29_source=intent_q29_source,
            executed_q29_t=executed_q29_t,
            executed_q29_t_valid_mask=(~t_dones).detach().clone(),
            done_any=done_any.detach().clone(),
            survival_steps=survival_steps.detach().clone(),
            physics_expected_support_steps=torch.stack(expected_support_frames, dim=0),
            physics_zmp_repaired_steps=torch.stack(zmp_repaired_frames, dim=0),
            physics_zmp_noisy_steps=torch.stack(zmp_noisy_frames, dim=0),
            physics_contact_repaired_steps=torch.stack(contact_repaired_frames, dim=0),
            physics_contact_noisy_steps=torch.stack(contact_noisy_frames, dim=0),
            physics_pair_valid_mask=torch.stack(physics_pair_valid_frames, dim=0),
            physics_survival_repaired_steps=torch.stack(survival_repaired_frames, dim=0),
            physics_survival_noisy_steps=torch.stack(survival_noisy_frames, dim=0),
            evaluation_only_lateral_lean_repaired_steps=(
                torch.stack(lean_repaired_frames, dim=0) if quality_trace_enabled else None
            ),
            evaluation_only_lateral_lean_noisy_steps=(
                torch.stack(lean_noisy_frames, dim=0) if quality_trace_enabled else None
            ),
        )
        evidence.validate()
        return evidence
    finally:
        if execution_started:
            end = getattr(command, "end_frontres_local_scenario_k_execution", None)
            if not callable(end):
                raise RuntimeError("v015 one-action K collector requires command Clean-continuation close ownership")
            end()
        if hasattr(runner, "_frontres_v015_one_action_k_phase"):
            delattr(runner, "_frontres_v015_one_action_k_phase")


def collect_frontres_v015_gain_return_priority_evidence(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    *,
    pair_layout: Any,
    gain_config: Any | None = None,
) -> FrontRESV015GainConsumerEvidence:
    """Run the v015 capture -> v003 Gain -> return/priority chain.

    It consumes a sealed Repair/Noisy reset and one t policy action. The
    explicit pre-live sentinel may pass its immutable result to the candidate
    adapter; it does not mutate sampler state or invoke legacy Gain/storage.
    """

    one_action = collect_frontres_v015_one_action_k_evidence(
        runner,
        observations,
        pair_layout=pair_layout,
    )
    facts = pair_frontres_v015_gain_facts(one_action)
    gain_module = _gain_module()
    if gain_module is None:
        raise RuntimeError("v015 Gain consumer chain requires the frontres_gain owner")
    config_cls = getattr(gain_module, "FrontRESIntentPhysicsGainConfig", None)
    input_cls = getattr(gain_module, "FrontRESIntentPhysicsGainInput", None)
    compute = getattr(gain_module, "compute_intent_physics_local_repair_gain", None)
    if not callable(config_cls) or not callable(input_cls) or not callable(compute):
        raise RuntimeError("v015 Gain consumer chain rejects the legacy Clean-global Gain owner")
    config = config_cls() if gain_config is None else gain_config
    gain_input = input_cls(
        intent_q29=facts.intent_q29,
        repaired_q29=facts.repaired_q29,
        noisy_q29=facts.noisy_q29,
        intent_q29_provenance=facts.intent_q29_provenance,
        intent_q29_source=facts.intent_q29_source,
        repair_action_steps=facts.policy_actions,
        intent_valid_mask=facts.intent_valid_mask,
        repaired_success=facts.repaired_success,
        noisy_success=facts.noisy_success,
        repaired_survival=facts.repaired_survival,
        noisy_survival=facts.noisy_survival,
        effective_horizon_k=facts.horizon_k,
        repaired_zmp_margin=facts.repaired_zmp_margin,
        noisy_zmp_margin=facts.noisy_zmp_margin,
        repaired_contact=facts.repaired_contact,
        noisy_contact=facts.noisy_contact,
        repaired_contact_violation=facts.repaired_contact_violation,
        noisy_contact_violation=facts.noisy_contact_violation,
        repaired_zmp_violation=facts.repaired_zmp_violation,
        noisy_zmp_violation=facts.noisy_zmp_violation,
        expected_support_steps=facts.expected_support_steps,
        repaired_contact_steps=facts.repaired_contact_steps,
        noisy_contact_steps=facts.noisy_contact_steps,
        repaired_zmp_margin_steps=facts.repaired_zmp_margin_steps,
        noisy_zmp_margin_steps=facts.noisy_zmp_margin_steps,
        physics_pair_valid_mask=facts.physics_pair_valid_mask,
    )
    gain_result = compute(gain_input, config=config)
    return_evidence = build_frontres_v015_gain_return_evidence(facts, gain_result)
    try:
        from rsl_rl.frontres.frontres_segment_sampler import build_frontres_v015_priority_evidence
    except ModuleNotFoundError as exc:
        raise RuntimeError("v015 Gain consumer chain requires the sampler priority-evidence owner") from exc
    priority_evidence = build_frontres_v015_priority_evidence(return_evidence)
    result = FrontRESV015GainConsumerEvidence(
        one_action=one_action,
        return_evidence=return_evidence,
        priority_evidence=priority_evidence,
    )
    result.validate()
    return result


def build_frontres_v015_grouped_candidate_batch(
    candidate_evidence: FrontRESV015GainConsumerEvidence,
    *,
    transaction_id: str,
    policy_snapshot_id: str,
    motion_ids: tuple[str, ...],
    start_frames: torch.Tensor,
    segment_ids: torch.Tensor,
    source_index: torch.Tensor,
    trial_index: torch.Tensor,
) -> FrontRESSegmentPPOBatch:
    """Connect sealed v015 candidate evidence to a grouped PPO batch.

    函数名说明:
        `build_frontres_v015_grouped_candidate_batch` 是 Step 4A candidate connector.
        它不创建 frozen snapshot, 不调用 runner storage, 不执行 loss/backward/step,
        也不触碰 priority 或 sampler state.

    主链路:
        上游: Step 3B v003 Gain return evidence 与显式 transaction row identity.
        下游: explicit pre-live sentinel or CPU formal transaction provider.

    语义:
        每个 Repair policy attempt 只映射到一个 PPO row. `evidence_valid_step_count`
        是 K-step executability metadata, 不参与 actor mass 或 grouped formula.
    """

    # B1: storage owner validates sealed local scenario and one-row policy tuple.
    storage_batch = build_frontres_v015_grouped_candidate_storage(
        candidate_evidence,
        transaction_id=transaction_id,
        policy_snapshot_id=policy_snapshot_id,
        motion_ids=motion_ids,
        start_frames=start_frames,
        segment_ids=segment_ids,
        source_index=source_index,
        trial_index=trial_index,
    )

    # B2: only the explicit grouped candidate adapter may retain v015 metadata.
    candidate_batch = storage_batch.to_grouped_ppo_candidate_batch(FrontRESSegmentPPOBatch)
    metadata = candidate_batch.transaction_metadata
    validate_metadata = getattr(metadata, "validate", None)
    if not callable(validate_metadata):
        raise TypeError("v015 grouped candidate connector lost sealed transaction metadata")
    validate_metadata()

    # B3: return a detached batch; the explicit transaction owner alone may evaluate loss or step.
    return candidate_batch


def _v015_formal_optimizer_step_count(optimizer: Any) -> int:
    """Require an explicit step counter; unknown optimizer state is not evidence."""

    for name in ("frontres_v015_step_count", "step_count"):
        value = getattr(optimizer, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return int(value)
    raise RuntimeError(
        "v015 formal transaction fake S2 requires an explicit non-negative optimizer "
        "frontres_v015_step_count or step_count"
    )


def _require_v015_formal_transaction_config(runner: Any) -> Any:
    """Freeze the v015 isolation boundary before any batch, loss, or step."""

    alg = getattr(runner, "alg", None)
    if alg is None or not bool(getattr(alg, "frontres_v015_formal_transaction_enabled", False)):
        raise RuntimeError("v015 formal transaction route requires frontres_v015_formal_transaction_enabled=True")
    if str(getattr(alg, "frontres_segment_advantage_normalization", "")).lower() != "grouped_scale_only":
        raise RuntimeError("v015 formal transaction route requires grouped_scale_only normalization")
    if any(
        float(getattr(alg, name, 0.0) or 0.0) != 0.0
        for name in ("lambda_supervised", "lambda_supervised_min")
    ):
        raise RuntimeError("v015 formal transaction route rejects nonzero Stage-3 supervised loss")
    if any(
        bool(getattr(alg, name, False))
        for name in ("frontres_hsl_init_enabled", "frontres_hsl_rollout_label_enabled")
    ):
        raise RuntimeError("v015 formal transaction route rejects implicit HSL initialization or rollout labels")
    schedule = tuple(getattr(alg, "frontres_segment_k_curriculum", ()) or ())
    if not schedule:
        raise RuntimeError("FRS-TRAIN-v010 formal transaction requires an explicit K-stage curriculum")
    if any(
        bool(getattr(alg, name, False))
        for name in (
            "frontres_segment_live_update_loop_only",
            "frontres_segment_live_single_update_only",
        )
    ):
        raise RuntimeError("v015 formal transaction rejects legacy immediate-update dispatch flags")
    if bool(getattr(alg, "frontres_segment_live_train_enabled", False)) and int(
        getattr(alg, "frontres_segment_live_update_steps", 0) or 0
    ) != 1:
        raise RuntimeError("v015 formal training requires one complete transaction and one update per iteration")
    offsets = tuple(int(value) for value in (getattr(alg, "frontres_future_offsets", ()) or ()))
    if not offsets or any(value <= 0 for value in offsets):
        raise RuntimeError("v015 formal transaction route requires positive deployment-q29 future offsets")
    if str(getattr(alg, "frontres_future_intent_layout_version", "")) != "frontres-v015-future-intent-q29-v1":
        raise RuntimeError("v015 formal transaction route requires the v015 q29 actor layout")
    required_identity = {
        "frontres_method_contract_id": "FRS-METHOD-v016",
        "frontres_gain_contract_id": "FRS-GAIN-v006",
        "frontres_optimization_contract_id": "FRS-PPO-v004",
        "frontres_training_contract_id": "FRS-TRAIN-v010",
        "frontres_scalar_target_id": "paired-intent-minus-repair-v1",
        "frontres_constraint_schema_id": "contact-loaded-phase_zmp-survival-physical-v2",
        "frontres_projection_schema_id": "grouped-first-order-constraint-projection-v1",
    }
    for name, expected in required_identity.items():
        if str(getattr(alg, name, "")) != expected:
            raise RuntimeError(f"v015 formal transaction requires {name}={expected}")
    return alg


def _v015_resolve_curriculum_identity(runner: Any, alg: Any | None = None) -> Any:
    """Resolve the sole K/phase identity allowed for the next transaction."""

    alg = _require_v015_formal_transaction_config(runner) if alg is None else alg
    identity = resolve_frontres_k_stage_identity(
        schedule=tuple(getattr(alg, "frontres_segment_k_curriculum", ()) or ()),
        committed_update_iteration=int(getattr(runner, "current_learning_iteration", 0)),
        max_horizon_k=int(getattr(alg, "frontres_segment_max_horizon_k", 0)),
    )
    configured_fingerprint = str(getattr(alg, "frontres_segment_k_curriculum_fingerprint", "") or "")
    if configured_fingerprint and configured_fingerprint != identity.schedule_fingerprint:
        raise RuntimeError("FRS-TRAIN-v010 runtime curriculum fingerprint drifted after config resolution")
    return identity


def _v015_formal_ppo_config(alg: Any, *, actor_loss_weight: float) -> FrontRESSegmentPPOConfig:
    """复用 v003 公式参数, 仅选择已确认的 grouped reduction mode."""

    return FrontRESSegmentPPOConfig(
        clip_param=float(getattr(alg, "clip_param", 0.2)),
        value_clip_param=float(getattr(alg, "clip_param", 0.2)),
        value_loss_coef=float(getattr(alg, "value_loss_coef", 1.0)),
        entropy_coef=float(getattr(alg, "entropy_coef", 0.0)),
        use_clipped_value_loss=bool(getattr(alg, "use_clipped_value_loss", True)),
        normalize_advantages=False,
        advantage_normalization="grouped_scale_only",
        actor_loss_weight=float(actor_loss_weight),
    )


def _v015_formal_policy_evaluator(
    request: FrontRESV015FormalTransactionRequest,
    alg: Any,
    ppo_batch: Any,
) -> Any:
    evaluator = request.policy_evaluator
    if evaluator is not None:
        if not callable(getattr(evaluator, "evaluate_segment_actions", None)):
            raise TypeError("v015 formal transaction policy_evaluator must expose evaluate_segment_actions")
        return evaluator
    privileged_observations = getattr(ppo_batch, "privileged_observations", None)
    request_privileged = request.privileged_observations
    if not isinstance(privileged_observations, torch.Tensor):
        raise RuntimeError("v015 formal transaction requires sealed t critic observations; actor-observation fallback is forbidden")
    if request_privileged is not None:
        if (
            tuple(request_privileged.shape) != tuple(privileged_observations.shape)
            or not torch.equal(
                request_privileged.to(device=privileged_observations.device),
                privileged_observations,
            )
        ):
            raise ValueError("v015 formal transaction request critic observations disagree with sealed candidate rows")
    if (
        privileged_observations.ndim != 2
        or int(privileged_observations.shape[0]) != int(ppo_batch.observations.shape[0])
        or int(privileged_observations.shape[1]) <= 0
    ):
        raise ValueError("v015 formal transaction critic observations must be non-empty [policy_row, critic_feature]")
    return FrontRESSegmentLivePolicyAdapter(alg, privileged_observations)


def run_frontres_v015_formal_transaction_update(
    runner: Any,
    request: FrontRESV015FormalTransactionRequest,
) -> FrontRESV015FormalTransactionUpdateResult:
    """Execute one sealed v015 offline-S2 grouped PPO update after all M attempts.

    此函数是 Step 4B 的唯一 update owner, 也是 Step 4C 的 committed receipt
    publisher. 它不调用 legacy `to_ppo_batch`, `run_frontres_segment_single_update`,
    sampler state, checkpoint save/load, simulator 或 live loop.

    Status: active exact-one update owner for offline contracts, the bounded
    sentinel, and ordinary v015 Stage-3 dispatch. Simulator and policy-quality
    evidence remain separate live gates.
    """

    if not isinstance(request, FrontRESV015FormalTransactionRequest):
        raise TypeError("v015 formal transaction update requires FrontRESV015FormalTransactionRequest")
    request.__post_init__()
    _bind_frontres_v015_checkpoint_transaction_plan(runner, request.plan)
    alg = _require_v015_formal_transaction_config(runner)
    policy = getattr(alg, "policy", None)
    optimizer = getattr(alg, "optimizer", None)
    if policy is None or optimizer is None:
        raise RuntimeError("v015 formal transaction update requires runner.alg policy and optimizer")
    optimizer_step_before = _v015_formal_optimizer_step_count(optimizer)
    curriculum = _v015_resolve_curriculum_identity(runner, alg)
    iteration = curriculum.absolute_iteration
    warmup_phase = curriculum.phase
    if (
        request.training_iteration != iteration
        or request.curriculum_fingerprint != curriculum.schedule_fingerprint
        or request.k_stage_index != curriculum.stage_index
        or request.active_k != curriculum.active_k
        or request.k_stage_iteration != curriculum.stage_iteration
        or request.warmup_phase_name != warmup_phase.name
        or not math.isclose(float(request.warmup_actor_loss_weight), warmup_phase.actor_loss_weight, abs_tol=1e-12)
    ):
        raise RuntimeError("v015 transaction crossed or changed its sealed FRS-TRAIN-v010 K-stage identity")
    if not bool((request.plan.horizon_k.detach().to(dtype=torch.long) == curriculum.active_k).all().item()):
        raise RuntimeError("FRS-TRAIN-v010 formal update rejects mixed-K transaction rows")
    request.plan.verify_policy(policy)
    accumulator = FrontRESV015FormalTransactionAccumulator(
        request.plan,
        optimizer_step_count=lambda: _v015_formal_optimizer_step_count(optimizer),
    )
    for candidate_batch in request.candidate_batches:
        accumulator.append_candidate_batch(candidate_batch)
    ppo_batch = accumulator.seal()
    _seal_frontres_v015_checkpoint_transaction_plan(runner, request.plan)
    request.plan.verify_policy(policy)
    policy_evaluator = _v015_formal_policy_evaluator(request, alg, ppo_batch)
    ppo_cfg = _v015_formal_ppo_config(alg, actor_loss_weight=warmup_phase.actor_loss_weight)
    ppo_result = compute_frontres_segment_ppo_loss(
        policy_evaluator,
        ppo_batch,
        ppo_cfg,
    )
    if not ppo_result.should_step:
        raise RuntimeError("v015 formal transaction has no valid grouped PPO rows; refusing optimizer step")
    zero_grad = getattr(optimizer, "zero_grad", None)
    step = getattr(optimizer, "step", None)
    if not callable(zero_grad) or not callable(step):
        raise RuntimeError("v015 formal transaction optimizer must expose zero_grad() and step()")
    try:
        zero_grad(set_to_none=True)
    except TypeError:
        zero_grad()
    optimizer_params, parameter_snapshots = _optimizer_parameter_snapshots(policy, optimizer)
    parameters = [
        parameter
        for group in getattr(optimizer, "param_groups", ())
        for parameter in group.get("params", ())
        if isinstance(parameter, torch.Tensor) and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("v015 formal transaction optimizer has no trainable parameters")
    projection = install_frontres_v004_projected_gradients(
        policy,
        ppo_result,
        ppo_cfg,
        tuple(parameters),
    )
    gradient_pre_clip_norm = float(
        torch.nn.utils.clip_grad_norm_(parameters, float(getattr(alg, "max_grad_norm", 1.0)))
    )
    gradient_tensors = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
    gradient_post_clip_norm = math.sqrt(
        sum(float(gradient.pow(2).sum().cpu().item()) for gradient in gradient_tensors)
    )
    gradient_nonzero_parameter_count = sum(
        int(bool((gradient != 0.0).any().cpu().item())) for gradient in gradient_tensors
    )
    if not math.isfinite(gradient_pre_clip_norm) or not math.isfinite(gradient_post_clip_norm):
        raise FloatingPointError(
            "v015 formal transaction produced non-finite gradients: "
            f"pre_clip={gradient_pre_clip_norm} post_clip={gradient_post_clip_norm}"
        )
    diagnostic_valid = (
        ppo_batch.valid_mask.detach().bool()
        & torch.isfinite(ppo_batch.returns.detach())
        & torch.isfinite(ppo_batch.advantages.detach())
    )
    if int(diagnostic_valid.sum().item()) != int(ppo_result.valid_count):
        raise RuntimeError(
            "v015 formal transaction return/advantage telemetry disagrees with PPO valid rows: "
            f"telemetry={int(diagnostic_valid.sum().item())} ppo={int(ppo_result.valid_count)}"
        )
    valid_returns = ppo_batch.returns.detach().float()[diagnostic_valid]
    step()
    parameter_delta = _parameter_delta_stats(optimizer_params, parameter_snapshots)
    critic = getattr(policy, "critic", None)
    critic_ids = {id(parameter) for parameter in critic.parameters()} if critic is not None else set()
    critic_params = tuple((name, parameter) for name, parameter in optimizer_params if id(parameter) in critic_ids)
    noncritic_params = tuple((name, parameter) for name, parameter in optimizer_params if id(parameter) not in critic_ids)
    critic_delta = _parameter_delta_stats(critic_params, parameter_snapshots)
    noncritic_delta = _parameter_delta_stats(noncritic_params, parameter_snapshots)
    if warmup_phase.name == "critic_only" and noncritic_delta["param_delta_max_abs"] != 0.0:
        raise RuntimeError("FRS-TRAIN-v010 critic-only update mutated actor or distribution parameters")
    optimizer_step_after = _v015_formal_optimizer_step_count(optimizer)
    optimizer_step_delta = optimizer_step_after - optimizer_step_before
    if optimizer_step_delta != 1:
        raise RuntimeError(
            "v015 formal transaction requires exactly one optimizer step: "
            f"before={optimizer_step_before} after={optimizer_step_after} delta={optimizer_step_delta}"
        )
    _commit_frontres_v015_checkpoint_transaction(
        runner,
        plan=request.plan,
        valid_policy_row_count=int(ppo_result.valid_count),
        optimizer_step_before=optimizer_step_before,
        optimizer_step_after=optimizer_step_after,
        curriculum=curriculum,
    )
    metadata = ppo_batch.transaction_metadata
    source_count = int(torch.unique(metadata.source_index.detach().to(dtype=torch.long)).numel())
    segment_count = int(torch.unique(metadata.segment_ids.detach().to(dtype=torch.long)).numel())
    flat_report_row_by_attempt: dict[tuple[int, int], int] = {}
    flat_report_row = 0
    for candidate_batch, report in zip(request.candidate_batches, request.diagnostic_reports, strict=True):
        candidate_metadata = candidate_batch.transaction_metadata
        if len(report.policy_actions) != int(candidate_metadata.batch_size):
            raise RuntimeError("v015 formal diagnostics disagree with their candidate batch row count")
        for row in range(int(candidate_metadata.batch_size)):
            key = (
                int(candidate_metadata.source_index[row].item()),
                int(candidate_metadata.trial_index[row].item()),
            )
            if key in flat_report_row_by_attempt:
                raise RuntimeError(f"v015 formal diagnostics repeat attempt identity {key}")
            flat_report_row_by_attempt[key] = flat_report_row
            flat_report_row += 1
    diagnostic_report_row_order = tuple(
        flat_report_row_by_attempt[(int(metadata.source_index[row].item()), int(metadata.trial_index[row].item()))]
        for row in range(int(metadata.batch_size))
    )
    diagnostics = {
        "transaction_id": request.plan.transaction_id,
        "policy_snapshot_id": request.plan.policy_snapshot_id,
        "motion_ids": tuple(metadata.motion_ids),
        "segment_ids": tuple(int(value) for value in metadata.segment_ids.tolist()),
        "source_index": tuple(int(value) for value in metadata.source_index.tolist()),
        "trial_index": tuple(int(value) for value in metadata.trial_index.tolist()),
        "horizon_k": tuple(int(value) for value in metadata.horizon_k.tolist()),
        "evidence_valid_step_count": tuple(int(value) for value in metadata.evidence_valid_step_count.tolist()),
        "noisy_segment_hashes": tuple(metadata.noisy_segment_hashes),
        "intent_q29_provenance": str(metadata.intent_q29_provenance),
        "intent_q29_source": str(metadata.intent_q29_source),
        "grouped_motion_mass_shares": tuple(ppo_result.grouped_motion_mass_shares),
        "grouped_segment_mass_shares": tuple(ppo_result.grouped_segment_mass_shares),
        "grouped_attempt_mass_shares": tuple(ppo_result.grouped_attempt_mass_shares),
        "return_mean": float(valid_returns.mean().cpu().item()),
        "return_min": float(valid_returns.min().cpu().item()),
        "return_max": float(valid_returns.max().cpu().item()),
        "return_abs_mean": float(valid_returns.abs().mean().cpu().item()),
        "gradient_pre_clip_norm": gradient_pre_clip_norm,
        "gradient_post_clip_norm": gradient_post_clip_norm,
        "gradient_parameter_count": len(parameters),
        "gradient_nonzero_parameter_count": gradient_nonzero_parameter_count,
        "optimizer_step_delta": int(optimizer_step_delta),
        "method_contract_id": "FRS-METHOD-v016",
        "training_contract_id": "FRS-TRAIN-v010",
        "gain_contract_id": "FRS-GAIN-v006",
        "optimization_contract_id": "FRS-PPO-v004",
        "scalar_target_id": "paired-intent-minus-repair-v1",
        "constraint_schema_id": "contact-loaded-phase_zmp-survival-physical-v2",
        "projection_schema_id": "grouped-first-order-constraint-projection-v1",
        "constraint_projection_status": projection.status,
        "constraint_active_families": projection.active_families,
        "constraint_gradient_norms": projection.gradient_norms,
        "constraint_directional_derivatives": projection.directional_derivatives,
        "constraint_levels": dict(ppo_result.constraint_levels or {}),
        "constraint_dual_coefficients": projection.dual_coefficients,
        "constraint_gram": projection.constraint_gram,
        "constraint_intent_directional_derivatives": projection.intent_directional_derivatives,
        "constraint_kkt_max_violation": projection.kkt_max_violation,
        "contact_constraint": tuple(float(value) for value in ppo_batch.contact_constraint.detach().cpu().tolist()),
        "zmp_constraint": tuple(float(value) for value in ppo_batch.zmp_constraint.detach().cpu().tolist()),
        "survival_constraint": tuple(float(value) for value in ppo_batch.survival_constraint.detach().cpu().tolist()),
        "contact_constraint_advantage": tuple(float(value) for value in ppo_batch.contact_constraint_advantage.detach().cpu().tolist()),
        "zmp_constraint_advantage": tuple(float(value) for value in ppo_batch.zmp_constraint_advantage.detach().cpu().tolist()),
        "survival_constraint_advantage": tuple(float(value) for value in ppo_batch.survival_constraint_advantage.detach().cpu().tolist()),
        "zmp_constraint_applicable": tuple(bool(value) for value in ppo_batch.zmp_constraint_applicable.detach().cpu().tolist()),
        "training_iteration": iteration,
        "curriculum_fingerprint": curriculum.schedule_fingerprint,
        "k_stage_index": curriculum.stage_index,
        "active_k": curriculum.active_k,
        "k_stage_iteration": curriculum.stage_iteration,
        "warmup_phase": warmup_phase.name,
        "warmup_phase_iteration": warmup_phase.phase_iteration,
        "actor_loss_weight": warmup_phase.actor_loss_weight,
        "parameter_delta": parameter_delta,
        "critic_parameter_delta": critic_delta,
        "actor_std_parameter_delta": noncritic_delta,
        "v006_action_constraint_reports": request.diagnostic_reports,
        "v006_diagnostic_report_row_order": diagnostic_report_row_order,
    }
    print(
        "[FrontRES v015 Formal Transaction] "
        f"transaction={request.plan.transaction_id} sources={source_count} "
        f"attempts={accumulator.collected_attempt_count} valid={ppo_result.valid_count} "
        f"step_delta={optimizer_step_delta}",
        flush=True,
    )
    return FrontRESV015FormalTransactionUpdateResult(
        transaction_id=request.plan.transaction_id,
        policy_snapshot_id=request.plan.policy_snapshot_id,
        segment_count=segment_count,
        source_count=source_count,
        policy_attempt_count=accumulator.collected_attempt_count,
        valid_row_count=int(ppo_result.valid_count),
        optimizer_step_before=optimizer_step_before,
        optimizer_step_after=optimizer_step_after,
        optimizer_step_delta=optimizer_step_delta,
        update_invocation_count=1,
        ppo_result=ppo_result,
        diagnostics=diagnostics,
    )


def _build_frontres_v015_local_transaction_request(
    runner: Any,
    *,
    init_at_random_ep_len: bool,
    route: str,
) -> FrontRESV015FormalTransactionRequest:
    """Build a real local-scenario request for one explicit v015 route.

    Keeping the request builder separate makes the collecting-barrier order
    testable: no reset or policy attempt may occur before the formal update loop
    has opened it.
    """

    del init_at_random_ep_len  # x_t reset owns the local dynamic start.
    alg = _require_v015_formal_transaction_config(runner)
    _prepare_frontres_raw_contact_views(runner)
    sealed_iteration = int(getattr(runner, "current_learning_iteration", 0))
    sealed_curriculum = _v015_resolve_curriculum_identity(runner, alg)
    if route == "sentinel":
        prepared = prepare_frontres_v015_local_sentinel_batch(runner)
        label = "local sentinel"
        diagnostics_attr = "_frontres_v015_local_sentinel_preupdate_diagnostics"
        batch_attr = "_frontres_v015_local_sentinel_batch"
    elif route == "training":
        prepared = prepare_frontres_v015_formal_training_batch(runner)
        label = "formal training"
        diagnostics_attr = "_frontres_v015_formal_training_preupdate_diagnostics"
        batch_attr = "_frontres_v015_formal_training_batch"
    else:
        raise ValueError(f"unknown v015 request route={route!r}")
    batch = prepared.batch
    plan = prepared.plan
    runner._frontres_segment_live_current_sample = prepared.sample
    runner._frontres_segment_live_current_batch = batch
    try:
        frontres_mode = resolve_frontres_mode_state(runner, FrontRESActorCritic)
        pair_layout = configure_frontres_pair_layout(runner, is_frontres=frontres_mode.is_frontres)
        policy_row_count = int(plan.batch_size)
        if (
            int(getattr(pair_layout, "n_train", 0)) != policy_row_count
            or int(getattr(pair_layout, "n_base", 0)) != policy_row_count
            or int(getattr(pair_layout, "n_candidate", 0)) != 0
            or int(getattr(pair_layout, "n_clean", 0)) != 0
        ):
            raise RuntimeError(f"v015 {label} requires an exact Repair/Noisy two-role layout for every planned policy row")
        reset_result = _apply_current_segment_reset(runner, pair_layout=pair_layout)
        success_mask = getattr(reset_result, "success_mask", None)
        if (
            reset_result is None
            or not isinstance(success_mask, torch.Tensor)
            or int(success_mask.numel()) != policy_row_count
            or not bool(success_mask.detach().bool().all())
        ):
            raise RuntimeError(f"v015 {label} requires every selected local scenario reset to succeed before actor evaluation")
        observations = _read_live_observations(runner)
        candidate_evidence = collect_frontres_v015_gain_return_priority_evidence(
            runner,
            observations,
            pair_layout=pair_layout,
        )
        candidate_batch = build_frontres_v015_grouped_candidate_batch(
            candidate_evidence,
            transaction_id=plan.transaction_id,
            policy_snapshot_id=plan.policy_snapshot_id,
            motion_ids=plan.motion_ids,
            start_frames=plan.start_frames,
            segment_ids=plan.segment_ids,
            source_index=plan.source_index,
            trial_index=plan.trial_index,
        )
        diagnostic_report = build_frontres_v015_local_evaluation_report(
            candidate_evidence,
            transaction_id=plan.transaction_id,
        )
        artifact = getattr(batch, "frontres_local_scenario_current_root_artifact_t", None)
        continuation_lengths = getattr(batch, "frontres_local_scenario_clean_continuation_lengths", None)
        if not isinstance(artifact, torch.Tensor) or not isinstance(continuation_lengths, torch.Tensor):
            raise RuntimeError(f"v015 {label} lost sealed root-artifact or Clean-continuation identity before storage")
        observation_trace = dict(getattr(runner, "_frontres_v015_observation_route_trace", {}) or {})
        expected_trace = {
            "role_row_count": 2 * policy_row_count,
            "current_command_dim": 58,
            "raw_observation_dim": 870,
            "q29_tail_dim": 58,
            "combined_observation_dim": 928,
            "normalized_observation_dim": 928,
            "femr_visible_dim": 158,
            "gmt_suffix_dim": 770,
            "gmt_input_dim": 770,
            "critic_observation_dim": 289,
        }
        mismatched_trace = {
            key: (observation_trace.get(key), expected)
            for key, expected in expected_trace.items()
            if observation_trace.get(key) != expected
        }
        if mismatched_trace or int(observation_trace.get("post_advance_gmt_read_count", 0)) <= 0:
            raise RuntimeError(
                f"v015 {label} observation trace is incomplete or violates the frozen authority: "
                f"mismatched={mismatched_trace}, trace={observation_trace}"
            )
        setattr(runner, diagnostics_attr, {
            "transaction_id": plan.transaction_id,
            "policy_snapshot_id": plan.policy_snapshot_id,
            "x_t_identities": tuple(plan.x_t_identities),
            "scenario_ids": tuple(plan.scenario_ids),
            "noisy_segment_hashes": tuple(plan.noisy_segment_hashes),
            "root_artifact_l2": tuple(float(value) for value in artifact.norm(dim=1).detach().cpu().tolist()),
            "intent_q29_provenance": plan.intent_q29_provenance,
            "intent_q29_source": plan.intent_q29_source,
            "clean_continuation_lengths": tuple(int(value) for value in continuation_lengths.detach().cpu().tolist()),
            "roles": ("repair",) * policy_row_count + ("noisy",) * policy_row_count,
            "policy_row_count": policy_row_count,
            "actor_forward_count": int(candidate_evidence.one_action.actor_forward_count),
            "later_femr_action_count": int(candidate_evidence.one_action.later_femr_action_count),
            "horizon_k": tuple(int(value) for value in plan.horizon_k.tolist()),
            "observation_route": observation_trace,
        })
        setattr(runner, batch_attr, batch)
        if int(getattr(runner, "current_learning_iteration", 0)) != sealed_iteration:
            raise RuntimeError("v015 formal transaction changed persisted iteration while collecting attempts")
        return FrontRESV015FormalTransactionRequest(
            plan=plan,
            candidate_batches=(candidate_batch,),
            diagnostic_reports=(diagnostic_report,),
            curriculum_fingerprint=sealed_curriculum.schedule_fingerprint,
            k_stage_index=sealed_curriculum.stage_index,
            active_k=sealed_curriculum.active_k,
            k_stage_iteration=sealed_curriculum.stage_iteration,
            training_iteration=sealed_iteration,
            warmup_phase_name=sealed_curriculum.phase.name,
            warmup_actor_loss_weight=sealed_curriculum.phase.actor_loss_weight,
        )
    except Exception:
        abort_frontres_v015_formal_training_collection(runner, batch=batch)
        raise


def _build_frontres_v015_local_identity_sentinel_request(
    runner: Any,
    *,
    init_at_random_ep_len: bool,
) -> FrontRESV015FormalTransactionRequest:
    """Build the dedicated bounded sentinel request."""

    return _build_frontres_v015_local_transaction_request(
        runner,
        init_at_random_ep_len=init_at_random_ep_len,
        route="sentinel",
    )


def build_frontres_v015_formal_training_request(
    runner: Any,
    *,
    init_at_random_ep_len: bool,
) -> FrontRESV015FormalTransactionRequest:
    """Build one complete ordinary Stage-3 request without legacy storage."""

    return _build_frontres_v015_local_transaction_request(
        runner,
        init_at_random_ep_len=init_at_random_ep_len,
        route="training",
    )


def close_frontres_v015_formal_training_request(runner: Any) -> None:
    """Release command and sampler carriers owned by one completed request."""

    batch = getattr(runner, "_frontres_v015_formal_training_batch", None)
    try:
        if batch is not None:
            command = _motion_command_for_runner(runner)
            clear = getattr(command, "clear_frontres_local_scenario", None)
            try:
                if not callable(clear):
                    raise RuntimeError(
                        "v015 formal training close requires command-owned local-scenario lifecycle"
                    )
                # Command rows own the live active bit; release them before the
                # immutable materializer identities are closed.
                clear()
            finally:
                _close_frontres_local_scenarios(batch)
    finally:
        if hasattr(runner, "_frontres_v015_formal_training_batch"):
            delattr(runner, "_frontres_v015_formal_training_batch")
        runner._frontres_segment_live_current_sample = None
        runner._frontres_segment_live_current_batch = None


def abort_frontres_v015_formal_training_collection(runner: Any, *, batch: Any | None = None) -> None:
    """Idempotently discard one rejected provider collection without an update.

    The command carrier is released before the materializer lifecycle.  The
    checkpoint barrier returns to the only persistable no-transaction state,
    so a later provider may open a fresh transaction at the same absolute
    training iteration.
    """

    if batch is None:
        batch = getattr(runner, "_frontres_v015_formal_training_batch", None)
    if batch is None:
        batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    command = None
    try:
        command = _motion_command_for_runner(runner)
    except (AttributeError, KeyError, RuntimeError):
        command = None
    if command is not None:
        active = getattr(command, "_frontres_local_scenario_active", None)
        clear = getattr(command, "clear_frontres_local_scenario", None)
        if isinstance(active, torch.Tensor) and bool(active.any()):
            if not callable(clear):
                raise RuntimeError("v015 rejected transaction requires command-owned scenario cleanup")
            clear()
    if batch is not None and not tuple(getattr(batch, "frontres_local_scenario_closed_ids", ()) or ()):
        _close_frontres_local_scenarios(batch)
    setattr(runner, _V015_CHECKPOINT_TRANSACTION_STATE_ATTR, {"state": "idle"})
    for name in (
        "_frontres_v015_formal_training_batch",
        "_frontres_v015_formal_training_preupdate_diagnostics",
    ):
        if hasattr(runner, name):
            delattr(runner, name)
    runner._frontres_segment_live_current_sample = None
    runner._frontres_segment_live_current_batch = None


def run_frontres_v015_local_identity_sentinel(
    runner: Any,
    *,
    init_at_random_ep_len: bool = True,
) -> FrontRESV015FormalTransactionUpdateResult:
    """Run one explicit v015 sentinel request through the existing exact-one update owner.

    Status: R6-S0 live-ready connector with fail-closed structured telemetry.
    The provider is invoked only after the formal collecting barrier opens;
    legacy probe/storage/update loops are not called. S4 live evidence remains open.
    """

    alg = _require_v015_formal_transaction_config(runner)
    if not bool(getattr(alg, "frontres_v015_local_sentinel_only", False)):
        raise RuntimeError("v015 local identity sentinel requires its explicit config flag")
    from rsl_rl.runners.frontres_segment_live_update_loop import run_frontres_v015_formal_transaction_update_loop

    def provider() -> FrontRESV015FormalTransactionRequest:
        return _build_frontres_v015_local_identity_sentinel_request(
            runner,
            init_at_random_ep_len=init_at_random_ep_len,
        )

    if hasattr(runner, "_frontres_v015_formal_transaction_provider"):
        raise RuntimeError("v015 local identity sentinel refuses an existing transaction provider")
    runner._frontres_v015_formal_transaction_provider = provider
    try:
        result = run_frontres_v015_formal_transaction_update_loop(runner)
        preupdate = getattr(runner, "_frontres_v015_local_sentinel_preupdate_diagnostics", None)
        if not isinstance(preupdate, dict):
            raise RuntimeError("v015 local sentinel requires a complete pre-update identity/observation snapshot")
        telemetry = dict(preupdate)
        result_diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        # B1: Reuse the ordinary Stage-3 final serializer so the sentinel cannot
        # silently drop sealed per-step Contact/ZMP evidence at its last adapter.
        from rsl_rl.runners.frontres_segment_live_training import _v015_sealed_transaction_telemetry

        telemetry["sealed_transaction_evidence"] = _v015_sealed_transaction_telemetry(
            result,
            ppo=result.ppo_result,
        )
        result_diagnostics.pop("v006_action_constraint_reports", None)
        telemetry.update(result_diagnostics)
        telemetry["optimizer_step_delta"] = int(getattr(result, "optimizer_step_delta", -1))
        telemetry["exact_one_update"] = telemetry["optimizer_step_delta"] == 1
        runner._frontres_v015_local_sentinel_telemetry = telemetry
        print(
            "[FrontRES v015 Local Sentinel] "
            f"transaction={telemetry['transaction_id']} "
            f"scenario_hashes={telemetry['noisy_segment_hashes']} "
            f"x_t={telemetry['x_t_identities']} "
            f"roles={telemetry['roles']} "
            f"actor_forwards={telemetry['actor_forward_count']} "
            f"later_femr_actions={telemetry['later_femr_action_count']} "
            f"K={telemetry['horizon_k']} "
            f"group_mass={telemetry.get('grouped_attempt_mass_shares', ())} "
            f"step_delta={telemetry['optimizer_step_delta']}",
            flush=True,
        )
        print(
            "[FrontRES v015 Live Snapshot] "
            + json.dumps(telemetry, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return result
    finally:
        if hasattr(runner, "_frontres_v015_formal_transaction_provider"):
            delattr(runner, "_frontres_v015_formal_transaction_provider")
        batch = getattr(runner, "_frontres_v015_local_sentinel_batch", None)
        if batch is not None:
            _close_frontres_local_scenarios(batch)
            delattr(runner, "_frontres_v015_local_sentinel_batch")
        runner._frontres_segment_live_current_sample = None
        runner._frontres_segment_live_current_batch = None


def _segment_repair_executability_scores(
    runner: Any,
    pair_layout: Any,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Return family-matched repair scores without generic env/task reward."""
    scorer = getattr(runner, "_frontres_executability", None)
    if scorer is None:
        raise RuntimeError("Segment Replay gain requires runner._frontres_executability")
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    command_manager = getattr(env, "command_manager", None)
    command = getattr(command_manager, "_terms", {}).get("motion") if command_manager is not None else None
    if command is None:
        raise RuntimeError("Segment Replay gain requires the motion command executability source")

    _, components = scorer.exec_score(command, return_components=True)
    role_counts = (
        int(pair_layout.n_train),
        int(pair_layout.n_candidate),
        int(pair_layout.n_base),
        int(pair_layout.n_clean),
    )
    if sum(role_counts) != int(batch_size):
        raise ValueError(
            "Segment Replay executability requires an exact quartet row layout; "
            f"counts={role_counts} batch_size={batch_size}"
        )

    cfg = getattr(runner, "cfg", {}) or {}
    specialist = str(cfg.get("frontres_specialist_mode", "") if hasattr(cfg, "get") else "").lower()
    active_modes = tuple(getattr(runner, "_frontres_curriculum_active_modes", ()))
    if specialist in ("rp", "local_rp", "rp_only", "strong_rp"):
        fallback_modes = ("local_rp",)
    elif active_modes:
        fallback_modes = active_modes
    else:
        raise RuntimeError("Segment Replay gain requires an explicit perturbation family")

    max_count = max(role_counts, default=0)
    mode_groups = list(getattr(runner, "_frontres_curriculum_env_mode_groups", ()))[:max_count]
    if len(mode_groups) < max_count:
        mode_groups.extend([fallback_modes] * (max_count - len(mode_groups)))

    scores = torch.empty(batch_size, device=runner.device, dtype=components["rp"].dtype)
    start = 0
    for count in role_counts:
        if count > 0:
            scores[start : start + count] = scorer.exec_score_for_modes(
                components,
                start,
                count,
                mode_groups=mode_groups[:count],
                active_modes=active_modes,
                include_task=False,
            )
        start += count
    return scores.detach()


def _run_live_rollout_capture(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    *,
    rollout_steps: int | None = None,
    capture_motion_quality: bool = True,
    zero_segment_action: bool = False,
    reset_lifecycle: dict[str, torch.Tensor] | None = None,
    pair_layout: Any | None = None,
) -> FrontRESSegmentLiveRolloutCapture:
    # FRS3-EVAL-014: step the live env and optionally capture motion-quality frames.
    try:
        v015_command = _frontres_motion_command(runner)
    except (RuntimeError, AttributeError):
        v015_command = None
    v015_local_active = getattr(v015_command, "_frontres_local_scenario_active", None)
    if isinstance(v015_local_active, torch.Tensor) and bool(v015_local_active.any()):
        raise RuntimeError(
            "v015 local scenarios are forbidden on the legacy repeated-actor live rollout; "
            "use collect_frontres_v015_one_action_k_evidence() until the formal-route gate is authorized"
        )
    frontres_mode = resolve_frontres_mode_state(runner, FrontRESActorCritic)
    if pair_layout is None:
        pair_layout = configure_frontres_pair_layout(runner, is_frontres=frontres_mode.is_frontres)
    batch_size = int(observations.obs.shape[0])
    # B1: reset 完成后比较四类 role 的 episode_length_buf, 确认生命周期是否只重置了 policy rows.
    # B2: rollout 前比较 policy/candidate/noisy/clean 的 root 与 joint dynamic state, 定位 quartet 配对断点.
    # B3: 每次 env.step 后按 role 分解 done/timeout/physical termination/alive/survival 与 first-done step.
    # AUDIT-RESET-LIFECYCLE-01: 检查 index reset -> quartet dynamic state -> K-step termination 生命周期.
    # Result: quartet reset is live-aligned; anchor_pos alone terminates all 32 rows at step 0, run=E33.
    if reset_lifecycle is not None:
        print_reset_lifecycle_audit(
            runner,
            pair_layout=pair_layout,
            phase="reset",
            pair_state=snapshot_reset_pair_state(runner, pair_layout),
            **reset_lifecycle,
        )
    if rollout_steps is not None:
        rollout_k = max(1, int(rollout_steps))
        horizon_k = torch.full((batch_size,), rollout_k, dtype=torch.long, device=runner.device)
    else:
        metadata = _current_trial_metadata(runner, batch_size=batch_size, device=runner.device)
        horizon_k = metadata.horizon_k.clamp_min(1)
        rollout_k = int(horizon_k.max().item())
    audit_identity = _new_live_audit_identity(
        runner,
        pair_layout=pair_layout,
        batch_size=batch_size,
        horizon_k=horizon_k,
    )
    vel_est_error_buffer = deque(maxlen=1)
    reward_accum = None
    repair_score_accum = None
    done_any = None
    reward_frames = []
    repair_score_frames = []
    gain_step_frames = []
    survival_gain_step_frames = []
    action_step_frames = []
    done_frames = []
    survival_steps = None
    first_done_step = torch.full((batch_size,), -1, dtype=torch.long, device=runner.device)
    actor_update_mask = None
    transition_obs = None
    transition_privileged_obs = None
    transition_actions = None
    transition_log_probs = None
    transition_values = None
    transition_means = None
    transition_sigmas = None
    transition_env_actions = None
    transition_perturbation_rp = None
    transition_supervised_target = None
    action_shape = None
    env_action_shape = None
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    max_delta_rpy = float(getattr(policy, "max_delta_rpy", 0.0)) if policy is not None else None
    clean_body_frames = []
    repaired_body_frames = []
    noisy_body_frames = []
    clean_root_quat_frames = []
    repaired_root_quat_frames = []
    noisy_root_quat_frames = []
    zmp_repaired_frames = []
    zmp_noisy_frames = []
    contact_repaired_frames = []
    contact_noisy_frames = []
    previous_clean_body = None
    previous_repaired_body = None
    previous_noisy_body = None
    previous_clean_root_quat = None
    previous_repaired_root_quat = None
    previous_noisy_root_quat = None
    previous_previous_clean_body = None
    previous_previous_repaired_body = None
    previous_previous_noisy_body = None
    previous_action = None
    gain_module = _gain_module()
    gain_config = (
        gain_module.FrontRESSegmentGainConfig.from_mapping(getattr(runner, "cfg", None))
        if gain_module is not None
        else None
    )
    obs = observations.obs
    privileged_obs = observations.privileged_obs
    teacher_obs = observations.teacher_obs
    ref_vel_estimator_obs = observations.ref_vel_estimator_obs
    last_obs_shape = tuple(obs.shape)

    with torch.inference_mode():
        for rollout_step in range(rollout_k):
            step_plan = prepare_frontres_rollout_step(
                runner,
                obs=obs,
                privileged_obs=privileged_obs,
                teacher_obs=teacher_obs,
                ref_vel_estimator_obs=ref_vel_estimator_obs,
                obs_raw_for_gmt=None,
                vel_est_error_buffer=vel_est_error_buffer,
                iteration=runner.current_learning_iteration,
                rollout_step=rollout_step,
                is_frontres=frontres_mode.is_frontres,
                is_task_space_mode=frontres_mode.is_task_space_mode,
                n_train=pair_layout.n_train,
                n_candidate=pair_layout.n_candidate,
                n_base=pair_layout.n_base,
                n_clean=pair_layout.n_clean,
            )
            actions = step_plan.actions
            env_actions = step_plan.env_actions
            if bool(zero_segment_action) and actions is not None and frontres_mode.is_task_space_mode:
                actions = actions.detach().clone()
                actions[: max(0, min(int(pair_layout.n_train), int(actions.shape[0])))] = 0.0
                runner.alg.transition.actions = actions.detach()
                env_actions = _zero_segment_env_actions(
                    runner,
                    obs=obs,
                    actions=actions,
                    is_frontres=frontres_mode.is_frontres,
                    is_task_space_mode=frontres_mode.is_task_space_mode,
                    n_train=pair_layout.n_train,
                    n_candidate=pair_layout.n_candidate,
                )
            action_shape = tuple(actions.shape) if actions is not None else None
            env_action_shape = tuple(env_actions.shape)
            if rollout_step == 0 and actions is not None:
                transition_obs = runner.alg.transition.observations.detach().clone()
                transition_privileged_obs = runner.alg.transition.privileged_observations.detach().clone()
                transition_env_actions = env_actions.detach().clone()
                transition_perturbation_rp = _snapshot_frontres_perturbation_rp(
                    runner,
                    num_envs=int(actions.shape[0]),
                )
                supervised_target = getattr(runner.alg.transition, "supervised_target", None)
                if supervised_target is not None and supervised_target.ndim == 2 and supervised_target.shape[-1] >= 6:
                    transition_supervised_target = supervised_target.detach().clone()
                selected_actions, selected_log_probs = _select_segment_transition_actions(runner, actions=actions)
                transition_actions = _select_executed_segment_actions(runner, actions=actions)
                transition_log_probs = selected_log_probs.detach().clone().reshape(-1)
                transition_values = runner.alg.transition.values.detach().clone().reshape(-1)
                action_mean = getattr(runner.alg.transition, "action_mean", None)
                action_sigma = getattr(runner.alg.transition, "action_sigma", None)
                if action_mean is not None and action_mean.ndim == 2 and action_mean.shape[-1] >= 6:
                    transition_means = action_mean[:, :6].detach().clone()
                if action_sigma is not None and action_sigma.ndim == 2 and action_sigma.shape[-1] >= 6:
                    transition_sigmas = action_sigma[:, :6].detach().clone()
                actor_update_mask = torch.zeros(actions.shape[0], device=runner.device, dtype=torch.bool)
                actor_update_mask[: max(0, min(int(pair_layout.n_train), actions.shape[0]))] = True

            selected_actions, _ = _select_segment_transition_actions(runner, actions=actions)
            executed_actions = _select_executed_segment_actions(runner, actions=actions)
            action_step_frames.append(executed_actions)

            obs, rewards, dones, infos = runner.env.step(env_actions.to(runner.env.device))
            _print_frontres_dr_runtime_probe(runner, label="after_env_step", rollout_step=rollout_step)
            rewards = rewards.to(runner.device)
            dones = dones.to(runner.device)
            paired_repair_evidence = (
                int(pair_layout.n_train) > 0
                and int(pair_layout.n_base) >= int(pair_layout.n_train)
            )
            repair_scores = (
                _segment_repair_executability_scores(
                    runner,
                    pair_layout,
                    batch_size=batch_size,
                )
                if paired_repair_evidence
                else None
            )
            horizon_active = rollout_step < horizon_k
            alive_before_step = torch.ones_like(horizon_active) if done_any is None else ~done_any
            score_active = horizon_active & alive_before_step
            scored_rewards = rewards.detach() * score_active.to(dtype=rewards.dtype)
            scored_repair = (
                repair_scores * score_active.to(dtype=repair_scores.dtype)
                if repair_scores is not None
                else None
            )
            scored_dones = dones.detach().bool() & score_active
            reward_accum = scored_rewards.clone() if reward_accum is None else reward_accum + scored_rewards
            if scored_repair is not None:
                repair_score_accum = (
                    scored_repair.clone()
                    if repair_score_accum is None
                    else repair_score_accum + scored_repair
                )
            reward_frames.append(rewards.detach().clone())
            if repair_scores is not None:
                repair_score_frames.append(repair_scores.detach().clone())
            done_frames.append(dones.detach().bool().clone())
            if done_any is None:
                done_any = torch.zeros_like(dones.detach(), dtype=torch.bool)
                survival_steps = torch.zeros_like(rewards.detach(), dtype=torch.float32)
            survival_steps = survival_steps + score_active.float()
            newly_done = scored_dones & first_done_step.lt(0)
            first_done_step[newly_done] = int(rollout_step)
            done_any = done_any | scored_dones
            time_outs = infos.get("time_outs") if isinstance(infos, dict) else None
            if isinstance(time_outs, torch.Tensor):
                time_outs = time_outs.to(runner.device).detach().bool()
                terminated = dones.detach().bool() & ~time_outs
            else:
                terminated = None
            print_reset_lifecycle_audit(
                runner,
                pair_layout=pair_layout,
                phase="step",
                rollout_step=rollout_step,
                dones=dones.detach().bool(),
                time_outs=time_outs,
                terminated=terminated,
                alive=~done_any,
                survival_steps=survival_steps,
                termination_terms=snapshot_termination_terms(
                    runner,
                    pair_layout,
                    batch_size=batch_size,
                ),
            )
            if capture_motion_quality:
                clean_body, repaired_body, noisy_body = _capture_motion_quality_frame(runner, pair_layout)
                clean_root_quat, repaired_root_quat, noisy_root_quat = _capture_root_orientation_frame(runner, pair_layout)
                physics_frame = _capture_physics_frame(runner, pair_layout)
                if clean_body is not None and repaired_body is not None and noisy_body is not None:
                    clean_body_frames.append(clean_body)
                    repaired_body_frames.append(repaired_body)
                    noisy_body_frames.append(noisy_body)
                    if clean_root_quat is not None and repaired_root_quat is not None and noisy_root_quat is not None:
                        clean_root_quat_frames.append(clean_root_quat)
                        repaired_root_quat_frames.append(repaired_root_quat)
                        noisy_root_quat_frames.append(noisy_root_quat)
                    if physics_frame is not None:
                        zmp_repaired, zmp_noisy, contact_repaired, contact_noisy = physics_frame
                        zmp_repaired_frames.append(zmp_repaired)
                        zmp_noisy_frames.append(zmp_noisy)
                        contact_repaired_frames.append(contact_repaired)
                        contact_noisy_frames.append(contact_noisy)
                    n_pair = min(int(pair_layout.n_train), int(pair_layout.n_base))
                    if n_pair > 0 and gain_module is not None and gain_config is not None:
                        train_success = (~done_any[:n_pair]).detach()
                        base_start = int(pair_layout.n_train) + int(pair_layout.n_candidate)
                        base_success = (~done_any[base_start : base_start + n_pair]).detach()
                        # B4: 逐步路径传入本步 alive increment, 由 Gain owner 用每行
                        # effective K 转成 survival quality increment. 累计这些增量后,
                        # 才与最终 raw survival_steps / K 的 Segment Gain 同源.
                        train_survival = score_active[:n_pair].float()
                        base_survival = score_active[base_start : base_start + n_pair].float()
                        step_horizon = (
                            horizon_k[:n_pair]
                            if isinstance(horizon_k, torch.Tensor)
                            else None
                        )
                        step_result = gain_module.compute_segment_gain_step(
                            clean_position=clean_body[:n_pair],
                            repaired_position=repaired_body[:n_pair],
                            noisy_position=noisy_body[:n_pair],
                            previous_clean_position=previous_clean_body,
                            previous_repaired_position=previous_repaired_body,
                            previous_noisy_position=previous_noisy_body,
                            previous_previous_clean_position=previous_previous_clean_body,
                            previous_previous_repaired_position=previous_previous_repaired_body,
                            previous_previous_noisy_position=previous_previous_noisy_body,
                            clean_root_quaternion=clean_root_quat,
                            repaired_root_quaternion=repaired_root_quat,
                            noisy_root_quaternion=noisy_root_quat,
                            repaired_zmp_margin=physics_frame[0] if physics_frame is not None else None,
                            noisy_zmp_margin=physics_frame[1] if physics_frame is not None else None,
                            repaired_contact=physics_frame[2] if physics_frame is not None else None,
                            noisy_contact=physics_frame[3] if physics_frame is not None else None,
                            repaired_success=train_success,
                            noisy_success=base_success,
                            repaired_survival=train_survival,
                            noisy_survival=base_survival,
                            effective_horizon_k=step_horizon,
                            action=executed_actions[:n_pair],
                            previous_action=previous_action,
                            config=gain_config,
                        )
                        full_step_gain = torch.full(
                            (batch_size,),
                            float("nan"),
                            device=runner.device,
                            dtype=step_result.gain_total.dtype,
                        )
                        full_step_gain[:n_pair] = step_result.gain_total
                        gain_step_frames.append(full_step_gain)
                        full_step_survival_gain = torch.full(
                            (batch_size,),
                            float("nan"),
                            device=runner.device,
                            dtype=step_result.physics_survival_gain.dtype,
                        )
                        full_step_survival_gain[:n_pair] = step_result.physics_survival_gain
                        survival_gain_step_frames.append(full_step_survival_gain)
                    previous_previous_clean_body = previous_clean_body
                    previous_previous_repaired_body = previous_repaired_body
                    previous_previous_noisy_body = previous_noisy_body
                    previous_clean_body = clean_body
                    previous_repaired_body = repaired_body
                    previous_noisy_body = noisy_body
                    previous_clean_root_quat = clean_root_quat
                    previous_repaired_root_quat = repaired_root_quat
                    previous_noisy_root_quat = noisy_root_quat
            elif int(pair_layout.n_train) > 0:
                gain_step_frames.append(torch.full((batch_size,), float("nan"), device=runner.device))
                survival_gain_step_frames.append(torch.full((batch_size,), float("nan"), device=runner.device))
            previous_action = executed_actions

            obs, privileged_obs, teacher_obs, ref_vel_estimator_obs = _read_step_observations(runner, obs, infos)
            last_obs_shape = tuple(obs.shape)

    print_reset_lifecycle_audit(
        runner,
        pair_layout=pair_layout,
        phase="final",
        first_done_step=first_done_step,
    )

    return FrontRESSegmentLiveRolloutCapture(
        rollout_k=rollout_k,
        reward_mean=float((reward_accum / horizon_k.to(dtype=reward_accum.dtype)).mean().detach().cpu()),
        done_frac=float(done_any.float().mean().detach().cpu()),
        last_obs_shape=last_obs_shape,
        action_shape=action_shape,
        env_action_shape=env_action_shape,
        transition_obs=transition_obs,
        transition_privileged_obs=transition_privileged_obs,
        transition_actions=transition_actions,
        transition_log_probs=transition_log_probs,
        transition_values=transition_values,
        transition_means=transition_means,
        transition_sigmas=transition_sigmas,
        transition_action_steps=torch.stack(action_step_frames, dim=0) if action_step_frames else None,
        reward_accum=reward_accum,
        done_any=done_any,
        reward_steps=torch.stack(reward_frames, dim=0) if reward_frames else None,
        repair_score_accum=repair_score_accum,
        repair_score_steps=torch.stack(repair_score_frames, dim=0) if repair_score_frames else None,
        gain_steps=torch.stack(gain_step_frames, dim=0) if gain_step_frames else None,
        survival_gain_steps=(
            torch.stack(survival_gain_step_frames, dim=0)
            if survival_gain_step_frames
            else None
        ),
        gain_config=gain_config,
        done_steps=torch.stack(done_frames, dim=0) if done_frames else None,
        horizon_k=horizon_k.detach().clone(),
        actor_update_mask=actor_update_mask,
        n_train=int(pair_layout.n_train),
        n_candidate=int(pair_layout.n_candidate),
        n_base=int(pair_layout.n_base),
        n_clean=int(pair_layout.n_clean),
        survival_steps=survival_steps,
        motion_clean_body_pos=_stack_motion_quality_frames(clean_body_frames),
        motion_repaired_body_pos=_stack_motion_quality_frames(repaired_body_frames),
        motion_noisy_body_pos=_stack_motion_quality_frames(noisy_body_frames),
        motion_clean_root_quat=_stack_motion_quality_frames(clean_root_quat_frames),
        motion_repaired_root_quat=_stack_motion_quality_frames(repaired_root_quat_frames),
        motion_noisy_root_quat=_stack_motion_quality_frames(noisy_root_quat_frames),
        physics_zmp_repaired_steps=_stack_motion_quality_frames(zmp_repaired_frames),
        physics_zmp_noisy_steps=_stack_motion_quality_frames(zmp_noisy_frames),
        physics_contact_repaired_steps=_stack_motion_quality_frames(contact_repaired_frames),
        physics_contact_noisy_steps=_stack_motion_quality_frames(contact_noisy_frames),
        env_actions=transition_env_actions,
        transition_perturbation_rp=transition_perturbation_rp,
        transition_supervised_target=transition_supervised_target,
        max_delta_rpy=max_delta_rpy,
        audit_transaction_id=audit_identity["audit_transaction_id"],
        audit_batch_signature=audit_identity["audit_batch_signature"],
        audit_role_signature=audit_identity["audit_role_signature"],
        audit_k_signature=audit_identity["audit_k_signature"],
        audit_segment_signature=audit_identity["audit_segment_signature"],
        audit_row_count=audit_identity["audit_row_count"],
        audit_identity_state=audit_identity["audit_identity_state"],
    )


def _zero_segment_env_actions(
    runner: Any,
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    is_frontres: bool,
    is_task_space_mode: bool,
    n_train: int,
    n_candidate: int,
) -> torch.Tensor:
    if is_task_space_mode:
        runner._apply_frontres_task_corrections(
            actions,
            n_train,
            allow_oracle=True,
            n_candidate=n_candidate if is_frontres else 0,
        )
        obs_corr, extras_corr = runner.env.get_observations()
        obs_corr_dict = extras_corr.get("observations", {})
        if runner.policy_obs_type is not None and runner.policy_obs_type in obs_corr_dict:
            obs_corr = obs_corr_dict[runner.policy_obs_type]
        obs_corr = _append_fixed_noisy_actor_context(runner, obs_corr.to(runner.device))
        obs_corr = runner._apply_obs_normalizer(obs_corr)
        return runner.alg.policy.get_env_action(obs_corr, actions)
    if hasattr(runner.alg.policy, "get_env_action"):
        return runner.alg.policy.get_env_action(obs, actions)
    return actions


def _capture_motion_quality_frame(
    runner: Any,
    pair_layout: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    # QUALITY-EXEC-01: 检查 applied repair -> frozen-GMT physical execution evidence.
    # Result: PENDING_Q_EVIDENCE; Q-E3 only proves execution callback connectivity.
    # B1: env.step 后 role states 尚在时捕获 success/fall/survival 与 action identity.
    # B2: 同帧记录 ZMP/contact/MPJPE/velocity/acceleration evidence.
    # B3: Gain/sequence aggregator 前保留 short-K 与 long-sequence metric boundary.
    """截获同一 quartet frame 的 Clean/Repaired/Noisy Style evidence.

    函数名说明:
        `_capture_motion_quality_frame` 是 paired Style capture adapter, 只对齐并
        返回 root-relative body positions; 它不是 MPJPE 聚合器或 Gain 公式.

    主链路:
        上游: env.step 后的 motion command 和 split-env pair layout.
        下游: `compute_segment_gain` 的 Style component 比较 matching motion/frame.

    语义:
        三个分支必须来自同一 motion/frame. 任一字段缺失时返回 None, diagnostics
        应标记 UNCONFIRMED, 不得静默写成 0.
    """
    # B1: 从一个 quartet frame 读取 matching Clean, Repaired 和 Noisy rows.
    command = _motion_command_for_runner(runner)
    if command is None:
        return None, None, None
    clean_ref = getattr(command, "body_pos_w", None)
    if not isinstance(clean_ref, torch.Tensor):
        clean_ref = getattr(command, "body_pos_relative_w", None)
    robot_pos = getattr(command, "robot_body_pos_w", None)
    if not isinstance(clean_ref, torch.Tensor) or not isinstance(robot_pos, torch.Tensor):
        return None, None, None
    n_train = max(0, int(pair_layout.n_train))
    n_candidate = max(0, int(pair_layout.n_candidate))
    n_base = max(0, int(pair_layout.n_base))
    n_clean = max(0, int(pair_layout.n_clean))
    n = min(n_train, n_base, n_clean)
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    if n <= 0 or int(robot_pos.shape[0]) < base_start + n or int(clean_ref.shape[0]) < clean_start + n:
        return None, None, None
    # B2: 按 role 对齐 root-relative body positions, 不跨 motion 聚合.
    frame = (
        _root_relative_body_pos(clean_ref[clean_start : clean_start + n]),
        _root_relative_body_pos(robot_pos[:n]),
        _root_relative_body_pos(robot_pos[base_start : base_start + n]),
    )
    # AUDIT-PAIR-EVIDENCE-01: Record style evidence before canonical Gain consumes it.
    # Result: E67 LIVE PASS for one capture; style evidence shares the
    # canonical transaction/batch identity.
    emit_formal_runtime_probe(
        "AUDIT-PAIR-EVIDENCE-01",
        clean_positions=frame[0],
        repaired_positions=frame[1],
        noisy_positions=frame[2],
        **_audit_identity_kwargs(getattr(runner, "_frontres_segment_live_audit_identity", None)),
    )
    return frame


def _capture_root_orientation_frame(
    runner: Any,
    pair_layout: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Capture Clean-target and executed root quaternions for one quartet.

    Status: active Style capture boundary.
    Upstream: motion command quartet and robot anchor state after env.step.
    Downstream: frontres_gain geodesic Style component.
    Evidence: source-confirmed fields; runtime availability still requires S4.
    Gap: absent anchor quaternions remain UNCONFIRMED rather than zero.
    """
    command = _motion_command_for_runner(runner)
    if command is None:
        return None, None, None
    clean_ref = getattr(command, "anchor_quat_w_original", None)
    robot_quat = getattr(command, "robot_anchor_quat_w", None)
    if not isinstance(clean_ref, torch.Tensor) or not isinstance(robot_quat, torch.Tensor):
        return None, None, None
    n_train = max(0, int(pair_layout.n_train))
    n_candidate = max(0, int(pair_layout.n_candidate))
    n_base = max(0, int(pair_layout.n_base))
    n_clean = max(0, int(pair_layout.n_clean))
    n = min(n_train, n_base, n_clean)
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    if n <= 0 or clean_ref.shape[-1] != 4 or robot_quat.shape[-1] != 4:
        return None, None, None
    if int(clean_ref.shape[0]) < clean_start + n or int(robot_quat.shape[0]) < base_start + n:
        return None, None, None
    return (
        clean_ref[clean_start : clean_start + n].detach().clone(),
        robot_quat[:n].detach().clone(),
        robot_quat[base_start : base_start + n].detach().clone(),
    )


def _capture_physics_frame(
    runner: Any,
    pair_layout: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Capture paired ZMP plus sealed expected and sensor-authoritative Contact.

    函数名说明:
        `_capture_physics_frame` 是 paired Physics capture adapter, 读取 frozen-GMT
        执行结果; 它不是 environment reward, 也不构造 Style Gain.

    主链路:
        上游: env.step 后的 robot state, motion command 和 paired role layout.
        下游: `compute_paired_physics_gain` 比较 Repaired/Noisy executability.

    语义:
        ZMP/support 必须按同一 Repair/Noisy frame 配对. Actual Contact 只来自
        已配置的 contact_forces ContactSensor；缺失时 fail closed.
    """
    # B1: 读取同一 quartet frame 的 paired frozen-GMT execution state.
    command = _motion_command_for_runner(runner)
    if command is None:
        return None
    n_train = max(0, int(pair_layout.n_train))
    n_candidate = max(0, int(pair_layout.n_candidate))
    n_base = max(0, int(pair_layout.n_base))
    n = min(n_train, n_base)
    if n <= 0:
        return None
    contact = _contact_sensor_pair(runner, command, pair_layout, n)
    if contact is None:
        return None
    expected_support, contact_repaired, contact_noisy = contact
    zmp_pair = _contact_wrench_zmp_pair(
        runner,
        command,
        pair_layout,
        expected_support,
        contact_repaired,
        contact_noisy,
        n,
    )
    if zmp_pair is None:
        return None
    zmp_repaired, zmp_noisy = zmp_pair
    # B2: 对齐 Repaired/Noisy ZMP 和 contact evidence, 产出 canonical Physics 输入.
    frame = (zmp_repaired, zmp_noisy, expected_support, contact_repaired, contact_noisy)
    # AUDIT-PAIR-EVIDENCE-01: Record physics evidence beside style evidence.
    # Result: E67 LIVE PASS for one capture; physics evidence shares the
    # canonical transaction/batch identity.
    emit_formal_runtime_probe(
        "AUDIT-PAIR-EVIDENCE-01",
        zmp_repaired=frame[0],
        zmp_noisy=frame[1],
        expected_support=frame[2],
        contact_repaired=frame[3],
        contact_noisy=frame[4],
        **_audit_identity_kwargs(getattr(runner, "_frontres_segment_live_audit_identity", None)),
    )
    return frame


def _capture_v015_quality_lateral_lean_frame(
    runner: Any,
    pair_layout: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read paired robot root roll for evaluation only; never expose it to training."""

    command = _motion_command_for_runner(runner)
    robot_quat = getattr(command, "robot_anchor_quat_w", None) if command is not None else None
    n = min(max(0, int(pair_layout.n_train)), max(0, int(pair_layout.n_base)))
    base_start = int(pair_layout.n_train) + int(pair_layout.n_candidate)
    if (
        not isinstance(robot_quat, torch.Tensor)
        or n <= 0
        or robot_quat.ndim != 2
        or int(robot_quat.shape[1]) != 4
        or int(robot_quat.shape[0]) < base_start + n
    ):
        return None

    def roll_wxyz(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(dim=-1)
        return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))

    return (
        roll_wxyz(robot_quat[:n]).detach().clone(),
        roll_wxyz(robot_quat[base_start : base_start + n]).detach().clone(),
    )


def _contact_sensor_pair(
    runner: Any,
    command: Any,
    pair_layout: Any,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    snapshot = getattr(command, "frontres_local_scenario_k_execution_snapshot", None)
    if not callable(snapshot):
        return None
    sealed = snapshot()
    support_rows = sealed.get("expected_support") if isinstance(sealed, Mapping) else None
    if not isinstance(support_rows, torch.Tensor) or tuple(support_rows.shape) != (int(command.num_envs), 2):
        return None
    n_train = int(pair_layout.n_train)
    n_candidate = int(pair_layout.n_candidate)
    n_base = int(pair_layout.n_base)
    base_start = n_train + n_candidate
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    # B1: Actual support 与 raw ZMP 必须服从同一 foot-to-ground filtered view.
    # 未过滤 net_forces_w 会包含足部与机器人/其他物体的接触, 不能定义地面支撑.
    actual_feet: list[torch.Tensor] = []
    sensors = getattr(scene, "sensors", None)
    for name in ("frontres_left_foot_contacts", "frontres_right_foot_contacts"):
        try:
            sensor = scene[name]
        except (KeyError, TypeError, AssertionError):
            sensor = sensors.get(name) if isinstance(sensors, Mapping) else None
        if sensor is None:
            return None
        force_matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
        if not isinstance(force_matrix, torch.Tensor):
            return None
        force_matrix = force_matrix.to(device=runner.device, dtype=torch.float32)
        expected_shape = (int(command.num_envs), 1)
        if (
            force_matrix.ndim != 4
            or tuple(force_matrix.shape[:2]) != expected_shape
            or int(force_matrix.shape[2]) <= 0
            or int(force_matrix.shape[3]) != 3
        ):
            raise RuntimeError(
                f"{name} filtered force matrix must be [N,1,F,3], got {tuple(force_matrix.shape)}"
            )
        if not bool(torch.isfinite(force_matrix).all()):
            raise RuntimeError(f"{name} filtered force matrix must be finite")
        threshold_value = getattr(getattr(sensor, "cfg", None), "force_threshold", None)
        if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
            raise RuntimeError(f"{name} requires an explicit numeric force threshold")
        threshold = float(threshold_value)
        if threshold <= 0.0:
            raise RuntimeError(f"{name} requires a positive force threshold")
        vertical_ground_load = force_matrix[..., 2].sum(dim=(1, 2)).abs()
        actual_feet.append(vertical_ground_load >= threshold)
    actual = torch.stack(actual_feet, dim=-1)
    expected_repair = support_rows[:n].bool()
    expected_noisy = support_rows[base_start : base_start + n].bool()
    if not torch.equal(expected_repair, expected_noisy):
        raise RuntimeError("FRS-GAIN-v006 paired roles do not share sealed expected support identity")
    return expected_repair, actual[:n], actual[base_start : base_start + n]


def _ensure_frontres_raw_contact_view(sensor: Any, *, num_envs: int) -> Any:
    """Install a raw-capable PhysX view for legacy IsaacLab ContactSensor."""

    existing = getattr(sensor, "contact_physx_view", None)
    if int(getattr(sensor, "_frontres_raw_contact_capacity", 0)) > 0:
        return existing
    cfg = getattr(sensor, "cfg", None)
    if int(getattr(cfg, "max_contact_data_count", 0)) > 0:
        return existing
    physics_view = getattr(sensor, "_physics_sim_view", None)
    create_view = getattr(physics_view, "create_rigid_contact_view", None)
    body_names = getattr(sensor, "body_names", None)
    prim_path = getattr(cfg, "prim_path", None)
    filter_expr = getattr(cfg, "filter_prim_paths_expr", None)
    if not callable(create_view) or not isinstance(body_names, (list, tuple)) or not body_names:
        return existing
    if not isinstance(prim_path, str) or not isinstance(filter_expr, (list, tuple)) or not filter_expr:
        return existing

    # Legacy IsaacLab creates ContactSensor views with the PhysX default capacity 0.
    # Reuse its resolved body identity and provision enough headroom for complex
    # foot-mesh/terrain contacts. The reader still rejects an exactly saturated
    # buffer because PhysX cannot prove that the returned payload is complete.
    parent = prim_path.rsplit("/", 1)[0]
    body_regex = r"(" + "|".join(re.escape(str(name)) for name in body_names) + r")"
    body_glob = f"{parent}/{body_regex}".replace(".*", "*")
    filter_glob = [str(expr).replace(".*", "*") for expr in filter_expr]
    raw_contacts_per_foot_env = 256
    capacity = max(raw_contacts_per_foot_env, int(num_envs) * raw_contacts_per_foot_env)
    raw_view = create_view(
        body_glob,
        filter_patterns=filter_glob,
        max_contact_data_count=capacity,
    )
    if int(getattr(raw_view, "count", int(num_envs))) != int(getattr(existing, "count", int(num_envs))):
        raise RuntimeError("raw contact view changed the ContactSensor body/env identity")
    if int(getattr(raw_view, "filter_count", len(filter_glob))) != int(
        getattr(existing, "filter_count", len(filter_glob))
    ):
        raise RuntimeError("raw contact view changed the ContactSensor filter identity")
    sensor._contact_physx_view = raw_view
    sensor._frontres_raw_contact_capacity = capacity
    return raw_view


def _prepare_frontres_raw_contact_views(runner: Any) -> None:
    """Install both raw views before reset/step can produce scored Physics evidence."""

    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    scene = getattr(env, "scene", None)
    if scene is None:
        raise RuntimeError("v015 Physics preparation requires the formal IsaacLab scene")
    num_envs = int(getattr(env, "num_envs", 0))
    if num_envs <= 0:
        raise RuntimeError("v015 Physics preparation requires a positive env count")
    for name in ("frontres_left_foot_contacts", "frontres_right_foot_contacts"):
        try:
            sensor = scene[name]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"v015 Physics preparation is missing scene sensor {name}") from exc
        view = _ensure_frontres_raw_contact_view(sensor, num_envs=num_envs)
        if int(getattr(sensor, "_frontres_raw_contact_capacity", 0)) <= 0:
            raise RuntimeError(f"v015 Physics preparation could not provision raw contact capacity for {name}")
        if view is not getattr(sensor, "_contact_physx_view", None):
            raise RuntimeError(f"v015 Physics preparation did not install the authoritative view for {name}")


def _raw_filtered_contact_rows(sensor: Any, *, num_envs: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Unpack one single-foot filtered ContactSensor into padded raw contacts."""

    view = getattr(sensor, "contact_physx_view", None)
    if int(getattr(sensor, "_frontres_raw_contact_capacity", 0)) <= 0:
        raise RuntimeError("contact-wrench ZMP requires a raw view installed before the scored physics step")
    get_contact_data = getattr(view, "get_contact_data", None)
    if not callable(get_contact_data):
        raise RuntimeError("contact-wrench ZMP requires ContactSensor.contact_physx_view.get_contact_data")
    dt = float(getattr(sensor, "_sim_physics_dt", 0.0))
    if dt <= 0.0:
        raise RuntimeError("contact-wrench ZMP requires a positive ContactSensor physics dt")
    payload = get_contact_data(dt=dt)
    if not isinstance(payload, tuple) or len(payload) != 6:
        raise RuntimeError("unexpected IsaacLab raw contact-data payload")
    normal_force, points_w, normals_w, _distance, counts, starts = payload
    tensors = (normal_force, points_w, normals_w, counts, starts)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise RuntimeError("raw contact-data payload must contain tensors")
    counts = counts.to(device=device, dtype=torch.long).reshape(-1)
    starts = starts.to(device=device, dtype=torch.long).reshape(-1)
    if int(counts.numel()) != int(num_envs) or int(starts.numel()) != int(num_envs):
        raise RuntimeError("each v015 foot sensor must resolve exactly one body and one ground filter per env")
    capacity = int(getattr(sensor, "_frontres_raw_contact_capacity", 0))
    if capacity > 0 and int(counts.sum().item()) >= capacity:
        raise RuntimeError("contact-wrench ZMP raw contact buffer reached capacity; evidence may be truncated")
    max_contacts = max(1, int(counts.max().item()) if int(counts.numel()) else 0)
    points = torch.zeros(num_envs, 1, max_contacts, 3, device=device, dtype=torch.float32)
    normals = torch.zeros_like(points)
    forces = torch.zeros(num_envs, 1, max_contacts, device=device, dtype=torch.float32)
    valid = torch.zeros(num_envs, 1, max_contacts, device=device, dtype=torch.bool)
    normal_force = normal_force.to(device=device, dtype=torch.float32).reshape(-1)
    points_w = points_w.to(device=device, dtype=torch.float32).reshape(-1, 3)
    normals_w = normals_w.to(device=device, dtype=torch.float32).reshape(-1, 3)
    for env_id in range(num_envs):
        count = int(counts[env_id].item())
        start = int(starts[env_id].item())
        if count <= 0:
            continue
        stop = start + count
        if stop > int(normal_force.numel()) or stop > int(points_w.shape[0]) or stop > int(normals_w.shape[0]):
            raise RuntimeError("raw contact-data count/start exceeds the PhysX contact buffer")
        forces[env_id, 0, :count] = normal_force[start:stop].abs()
        points[env_id, 0, :count] = points_w[start:stop]
        normals[env_id, 0, :count] = normals_w[start:stop]
        valid[env_id, 0, :count] = True
    return points, forces, normals, valid


def _pad_raw_contact_slots(raw: tuple[torch.Tensor, ...], *, contact_slots: int) -> tuple[torch.Tensor, ...]:
    """Right-pad one foot's raw contacts so both feet share a contact axis."""

    points, forces, normals, valid = raw
    current = int(points.shape[2])
    if current > int(contact_slots) or int(contact_slots) <= 0:
        raise RuntimeError("raw contact-slot padding requires target C >= current C > 0")
    if current == int(contact_slots):
        return raw
    batch, feet = int(points.shape[0]), int(points.shape[1])
    padded_points = torch.zeros(batch, feet, contact_slots, 3, device=points.device, dtype=points.dtype)
    padded_forces = torch.zeros(batch, feet, contact_slots, device=forces.device, dtype=forces.dtype)
    padded_normals = torch.zeros(batch, feet, contact_slots, 3, device=normals.device, dtype=normals.dtype)
    padded_valid = torch.zeros(batch, feet, contact_slots, device=valid.device, dtype=torch.bool)
    padded_points[:, :, :current] = points
    padded_forces[:, :, :current] = forces
    padded_normals[:, :, :current] = normals
    padded_valid[:, :, :current] = valid.bool()
    return padded_points, padded_forces, padded_normals, padded_valid


def _contact_wrench_zmp_pair(
    runner: Any,
    command: Any,
    pair_layout: Any,
    expected_support: torch.Tensor,
    contact_repaired: torch.Tensor,
    contact_noisy: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Produce paired true contact-wrench ZMP margins; no proxy fallback exists."""

    snapshot = getattr(command, "frontres_local_scenario_k_execution_snapshot", None)
    if not callable(snapshot):
        raise RuntimeError("contact-wrench ZMP requires the sealed local-scenario K snapshot")
    sealed = snapshot()
    envelope = sealed.get("expected_support_envelope") if isinstance(sealed, Mapping) else None
    if not isinstance(envelope, torch.Tensor) or tuple(envelope.shape) != (int(command.num_envs), 6):
        shape = tuple(envelope.shape) if isinstance(envelope, torch.Tensor) else None
        raise RuntimeError(
            "contact-wrench ZMP requires sealed expected_support_envelope "
            f"[{int(command.num_envs)},6], got {shape}"
        )
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    scene = getattr(env, "scene", None)
    if scene is None:
        raise RuntimeError("contact-wrench ZMP requires the formal IsaacLab scene")
    try:
        left_sensor = scene["frontres_left_foot_contacts"]
        right_sensor = scene["frontres_right_foot_contacts"]
        from rsl_rl.frontres.frontres_balance import contact_wrench_zmp_xy, expected_support_envelope_margin

        raw_left = _raw_filtered_contact_rows(left_sensor, num_envs=int(command.num_envs), device=runner.device)
        raw_right = _raw_filtered_contact_rows(right_sensor, num_envs=int(command.num_envs), device=runner.device)
        contact_slots = max(int(raw_left[0].shape[2]), int(raw_right[0].shape[2]))
        raw_left = _pad_raw_contact_slots(raw_left, contact_slots=contact_slots)
        raw_right = _pad_raw_contact_slots(raw_right, contact_slots=contact_slots)
        points = torch.cat((raw_left[0], raw_right[0]), dim=1)
        forces = torch.cat((raw_left[1], raw_right[1]), dim=1)
        normals = torch.cat((raw_left[2], raw_right[2]), dim=1)
        valid = torch.cat((raw_left[3], raw_right[3]), dim=1)
        zmp_xy, zmp_valid = contact_wrench_zmp_xy(points, forces, normals, valid)
        origins_xy = getattr(scene, "env_origins", None)
        if not isinstance(origins_xy, torch.Tensor):
            raise RuntimeError("contact-wrench ZMP requires scene.env_origins")
        support_all = sealed.get("expected_support")
        if not isinstance(support_all, torch.Tensor) or tuple(support_all.shape) != (int(command.num_envs), 2):
            shape = tuple(support_all.shape) if isinstance(support_all, torch.Tensor) else None
            raise RuntimeError(
                f"contact-wrench ZMP requires sealed expected_support [{int(command.num_envs)},2], got {shape}"
            )
        margin = expected_support_envelope_margin(
            zmp_xy,
            envelope.to(device=runner.device),
            support_all.to(device=runner.device),
            env_origins_xy=origins_xy[:, :2].to(device=runner.device),
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"contact-wrench ZMP capture failed at {type(exc).__name__}: {exc}"
        ) from exc
    base_start = int(pair_layout.n_train) + int(pair_layout.n_candidate)
    expected_loaded = expected_support.to(device=runner.device, dtype=torch.bool).any(dim=-1)
    branch_rows = (
        ("Repair", slice(0, n), contact_repaired),
        ("Noisy", slice(base_start, base_start + n), contact_noisy),
    )
    outputs: list[torch.Tensor] = []
    for role, rows, actual_contact in branch_rows:
        actual_loaded = actual_contact.to(device=runner.device, dtype=torch.bool).any(dim=-1)
        required = expected_loaded & actual_loaded
        branch_valid = zmp_valid[rows]
        if bool((required & ~branch_valid).any()):
            missing_rows = torch.nonzero(required & ~branch_valid, as_tuple=False).reshape(-1).tolist()
            raise RuntimeError(
                f"{role} loaded support is missing a finite raw contact-wrench resultant; "
                f"branch_rows={missing_rows}"
            )
        branch_margin = margin[rows]
        outputs.append(
            torch.where(required, branch_margin, torch.full_like(branch_margin, float("nan"))).detach()
        )
    return outputs[0], outputs[1]


def _root_relative_body_pos(body_pos: torch.Tensor) -> torch.Tensor:
    if body_pos.ndim < 3 or int(body_pos.shape[-2]) <= 0:
        return body_pos.detach().clone()
    return (body_pos - body_pos[..., :1, :]).detach().clone()


def _motion_command_for_runner(runner: Any) -> Any | None:
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return None
    if hasattr(manager, "get_term"):
        try:
            return manager.get_term("motion")
        except Exception:
            return None
    terms = getattr(manager, "_terms", {})
    return terms.get("motion") if isinstance(terms, dict) else None


def _stack_motion_quality_frames(frames: list[torch.Tensor]) -> torch.Tensor | None:
    if not frames:
        return None
    return torch.stack(frames, dim=1)


def _read_step_observations(runner: Any, obs: torch.Tensor, infos: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    obs_dict = infos.get("observations", {})
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type].to(runner.device)
    else:
        obs = obs.to(runner.device)
    obs = _append_fixed_noisy_actor_context(runner, obs)
    obs = runner._apply_obs_normalizer(obs)
    if runner.privileged_obs_type is not None and runner.privileged_obs_type in obs_dict:
        privileged_obs = runner.privileged_obs_normalizer(obs_dict[runner.privileged_obs_type].to(runner.device))
    else:
        privileged_obs = obs
    if runner.teacher_obs_type is not None and runner.teacher_obs_type in obs_dict:
        teacher_obs = runner.teacher_obs_normalizer(obs_dict[runner.teacher_obs_type].to(runner.device))
    else:
        teacher_obs = privileged_obs
    if runner.ref_vel_estimator_obs_type is not None and runner.ref_vel_estimator_obs_type in obs_dict:
        ref_vel_estimator_obs = obs_dict[runner.ref_vel_estimator_obs_type].to(runner.device)
    else:
        ref_vel_estimator_obs = None
    return obs, privileged_obs, teacher_obs, ref_vel_estimator_obs


def _initial_live_probe_summary(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    storage_write: bool,
    single_update: bool,
) -> dict[str, object]:
    """Build the active live summary from canonical paired Gain evidence.

    Status: active diagnostic boundary.
    Upstream: captured paired rollout and ``_capture_paired_gain``.
    Downstream: sampler evidence, update-loop aggregation, and train logs.
    Evidence: Step 7 implementation path; legacy per-row score fields remain
    only for sampler compatibility and are not active train diagnostics.
    Gap: real simulator component population remains an S4 boundary.
    """
    legacy_score_compatibility = _paired_score_summary(capture)
    paired_gain = _capture_paired_gain(capture)
    gain_summary = _paired_gain_summary(capture)
    gain_total_pos_frac = (
        _positive_fraction(_float_list(paired_gain.gain_total))
        if paired_gain is not None
        else float("nan")
    )
    summary = {
        **_capture_audit_identity_kwargs(capture),
        "rollout_k": capture.rollout_k,
        "rollout_horizon_summary": _tensor_range_summary("horizon", capture.horizon_k)
        if isinstance(capture.horizon_k, torch.Tensor)
        else f"horizon_count=0 horizon_min={capture.rollout_k} horizon_max={capture.rollout_k}",
        "reward_mean": capture.reward_mean,
        "env_reward_mean": capture.reward_mean,
        "train_reward_mean": capture.reward_mean,
        "gain_total_pos_frac": gain_total_pos_frac,
        "motion_delta_se_norm": _delta_se_norm(capture.transition_actions),
        "motion_delta_z_up_frac": _delta_z_up_frac(capture.transition_actions),
        "done_frac": capture.done_frac,
        "valid_mask_frac": 1.0 - capture.done_frac,
        "reward_per_sample": _rollout_reward_per_sample(capture),
        "done_any_per_sample": _rollout_done_per_sample(capture),
        "storage_write": storage_write,
        "storage_size": 0,
        "storage_valid_frac": 0.0,
        "storage_reward_mean": 0.0,
        "storage_reward_per_sample": [],
        "storage_valid_mask_per_sample": [],
        "storage_segment_ids": [],
        "single_update": single_update,
        "ppo_update": False,
        "ppo_valid_count": 0,
        "ppo_total_loss": 0.0,
        "ppo_actor_loss": 0.0,
        "ppo_value_loss": 0.0,
        "ppo_approx_kl": 0.0,
        "ppo_clip_frac": 0.0,
        "ppo_pre_update_raw_log_ratio_mean": 0.0,
        "ppo_pre_update_raw_log_ratio_min": 0.0,
        "ppo_pre_update_raw_log_ratio_max": 0.0,
        "ppo_pre_update_clamped_ratio_mean": 0.0,
        "ppo_pre_update_clamped_ratio_max": 0.0,
        "ppo_pre_distribution_kl_mean": 0.0,
        "ppo_pre_logprob_approx_kl": 0.0,
        "ppo_distribution_kl_available": False,
        "ppo_post_update_distribution_kl_mean": 0.0,
        "ppo_post_update_logprob_approx_kl": 0.0,
        "ppo_post_update_ratio_mean": 0.0,
        "ppo_post_update_ratio_max": 0.0,
        "ppo_post_update_raw_log_ratio_mean": 0.0,
        "ppo_post_update_raw_log_ratio_min": 0.0,
        "ppo_post_update_raw_log_ratio_max": 0.0,
        "ppo_post_update_clamped_ratio_mean": 0.0,
        "ppo_post_update_clamped_ratio_max": 0.0,
        "ppo_post_update_clip_frac": 0.0,
        "ppo_param_delta_max_abs": 0.0,
        "ppo_param_delta_l2": 0.0,
        "ppo_param_delta_changed": 0,
        "ppo_param_delta_total": 0,
        "ppo_param_delta_first_changed": "",
        "ppo_param_grad_norm": 0.0,
        "ppo_trust_region_rejected_count": 0,
        "ppo_trust_region_accepted": 1,
    }
    # Compatibility vectors are retained for sampler evidence serialization;
    # no legacy scalar is used by active diagnostics or training aggregation.
    summary.update(legacy_score_compatibility)
    summary.update(gain_summary)
    summary.update(_motion_quality_summary(capture))
    return summary


def _motion_quality_summary(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, float]:
    try:
        from rsl_rl.frontres.frontres_segment_diagnostics import motion_quality_summary_to_scalars
    except ModuleNotFoundError:
        return {}
    positions = capture.motion_repaired_body_pos
    temporal_mask = None
    valid_mask = capture.actor_update_mask
    if isinstance(positions, torch.Tensor):
        batch_size, time_steps = int(positions.shape[0]), int(positions.shape[1])
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask[:batch_size]
        if isinstance(capture.horizon_k, torch.Tensor):
            horizon = capture.horizon_k[:batch_size].to(device=positions.device, dtype=torch.long)
            temporal_mask = torch.arange(time_steps, device=positions.device).view(1, -1) < horizon.view(-1, 1)
    return motion_quality_summary_to_scalars(
        clean_positions=capture.motion_clean_body_pos,
        repaired_positions=capture.motion_repaired_body_pos,
        noisy_positions=capture.motion_noisy_body_pos,
        delta_se=capture.transition_actions,
        valid_mask=valid_mask,
        temporal_mask=temporal_mask,
    )


def _paired_score_summary(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, object]:
    """Return legacy executable-score vectors for compatibility evidence only.

    Status: legacy compatibility boundary, not an active training diagnostic.
    Upstream: paired rollout capture. Downstream: sampler evidence compatibility
    fields and migration tests only. Evidence: Step 6C/7 audit.
    Gap: the active route must use ``_paired_gain_summary`` instead.
    """
    if capture.done_any is None:
        return {}
    n_train = max(0, int(capture.n_train))
    n_candidate = max(0, int(capture.n_candidate))
    n_base = max(0, int(capture.n_base))
    n_clean = max(0, int(capture.n_clean))
    n = min(n_train, n_base)
    if n <= 0:
        return {}
    score = _capture_averaged_repair_scores(capture)
    done = capture.done_any.reshape(-1).detach().bool()
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    if int(score.numel()) < base_start + n:
        return {}
    clean = score[clean_start : clean_start + n] if n_clean >= n and int(score.numel()) >= clean_start + n else torch.ones(n, device=score.device)
    noisy = score[base_start : base_start + n]
    repaired = score[:n]
    return {
        "evidence_row_count": n,
        "evidence_reward_per_sample": _float_list(repaired),
        "evidence_done_any_per_sample": _bool_list(done[:n]),
        "evidence_valid_mask_per_sample": _bool_list(~done[:n]),
        "score_repaired_per_sample": _float_list(repaired),
        "score_noisy_per_sample": _float_list(noisy),
        "gain_over_noisy_per_sample": _float_list(repaired - noisy),
        "score_clean_per_sample": _float_list(clean),
        "score_source": "repair_executability",
    }


def _paired_gain_summary(capture: FrontRESSegmentLiveRolloutCapture) -> dict[str, object]:
    result = _capture_paired_gain(capture)
    if result is None:
        return {**_capture_audit_identity_kwargs(capture), "gain_source": "UNCONFIRMED"}
    return {
        **_capture_audit_identity_kwargs(capture),
        "gain_source": "FRS-GAIN-v002",
        "gain_style_per_sample": _float_list(result.style_gain),
        "gain_physics_per_sample": _float_list(result.physics_gain),
        "gain_repair_cost_per_sample": _float_list(result.repair_cost),
        "gain_total_per_sample": _float_list(result.gain_total),
        "gain_style_mean": _finite_mean(result.style_gain),
        "gain_physics_mean": _finite_mean(result.physics_gain),
        "gain_repair_cost_mean": _finite_mean(result.repair_cost),
        "gain_total_mean": _finite_mean(result.gain_total),
        "gain_style_mpjpe_mean": _finite_mean(result.style_mpjpe_gain),
        "gain_style_velocity_mean": _finite_mean(result.style_velocity_gain),
        "gain_style_acceleration_mean": _finite_mean(result.style_acceleration_gain),
        "gain_style_root_orientation_mean": _finite_mean(result.style_root_orientation_gain),
        "gain_physics_success_mean": _finite_mean(result.physics_success_gain),
        "gain_physics_survival_quality_repaired_per_sample": _float_list(result.physics_survival_quality_repaired),
        "gain_physics_survival_quality_noisy_per_sample": _float_list(result.physics_survival_quality_noisy),
        "gain_physics_survival_per_sample": _float_list(result.physics_survival_gain),
        "gain_physics_survival_quality_repaired_mean": _finite_mean(result.physics_survival_quality_repaired),
        "gain_physics_survival_quality_noisy_mean": _finite_mean(result.physics_survival_quality_noisy),
        "gain_physics_survival_mean": _finite_mean(result.physics_survival_gain),
        "gain_physics_zmp_mean": _finite_mean(result.physics_zmp_gain),
        "gain_physics_contact_mean": _finite_mean(result.physics_contact_gain),
        "gain_repair_norm_mean": _finite_mean(result.repair_norm),
        "gain_repair_temporal_mean": _finite_mean(result.repair_temporal_change),
        "gain_repair_clean_cost_per_sample": _float_list(result.repair_clean_cost),
        "gain_repair_clean_cost_mean": _finite_mean(result.repair_clean_cost),
    }


def _rollout_reward_per_sample(capture: FrontRESSegmentLiveRolloutCapture) -> list[float]:
    if capture.reward_accum is None:
        return []
    reward = _capture_averaged_rewards(capture)
    return _float_list(reward)


def _rollout_done_per_sample(capture: FrontRESSegmentLiveRolloutCapture) -> list[bool]:
    if capture.done_any is None:
        return []
    return _bool_list(capture.done_any.reshape(-1))


def _valid_reward_mean(rewards: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid = valid_mask.detach().bool().reshape(-1)
    reward = rewards.detach().float().reshape(-1)
    if int(valid.numel()) != int(reward.numel()):
        raise ValueError(f"valid_mask must have {int(reward.numel())} rows, got {int(valid.numel())}")
    if not bool(valid.any().item()):
        return 0.0
    return float(reward[valid].mean().cpu().item())


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().reshape(-1).cpu().tolist()]


def _bool_list(value: torch.Tensor) -> list[bool]:
    return [bool(item) for item in value.detach().bool().reshape(-1).cpu().tolist()]


def _long_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().long().reshape(-1).cpu().tolist()]


def _print_live_probe_summary(
    runner: Any,
    capture: FrontRESSegmentLiveRolloutCapture,
    summary: dict[str, object],
) -> None:
    """Print the human-facing live probe blocks without changing training state.

    Status: active diagnostic formatter.
    Upstream: run_frontres_segment_live_probe builds summary from rollout, storage, and PPO result.
    Downstream: terminal/log review only; no sampler, loss, optimizer, or checkpoint side effect.
    Evidence: contract-confirmed by frontres_segment_live_probe_contract.py.
    Gap: text presence does not prove live physics quality.
    """
    if not _live_detail_log_enabled(runner):
        return
    segment_action_shape = (
        tuple(capture.transition_actions.shape) if capture.transition_actions is not None else None
    )
    segment_delta_se_6d = bool(_shape_last_dim(segment_action_shape) == 6)
    print(
        _log_block(
            "[FrontRES Segment Live Probe]",
            *_kv_lines(
                "route",
                {
                    "objective": getattr(runner.alg, "frontres_training_objective", "n/a"),
                    "segment_id": "live_env_current",
                    "reset_mode": runner._frontres_segment_replay_boundary.reset_mode,
                },
            ),
            *_kv_lines(
                "reset",
                {
                    "enabled": bool(summary["segment_reset"]),
                    "reason": summary.get("segment_reset_skip_reason", "") or "applied",
                    "ok": _fmt_pct(summary["segment_reset_success_frac"]),
                    "direct": _fmt_pct(summary["segment_reset_direct_frac"]),
                    "preroll": _fmt_pct(summary["segment_reset_preroll_frac"]),
                    "vel_mismatch": _fmt_num(summary["segment_reset_velocity_mismatch_mean"]),
                    "ref_window": _fmt_pct(summary["segment_reference_window_applied_frac"]),
                },
            ),
            *_kv_lines(
                "rollout",
                {
                    "obs": capture.last_obs_shape,
                    "policy_action": capture.action_shape,
                    "policy_dim": _shape_last_dim(capture.action_shape),
                    "segment_action": segment_action_shape,
                    "segment_delta_se_6d": segment_delta_se_6d,
                    "env_action": capture.env_action_shape,
                    "env_dim": _shape_last_dim(capture.env_action_shape),
                    "k": capture.rollout_k,
                    "horizon": summary.get("rollout_horizon_summary", "unavailable"),
                    "env_reward": _fmt_num(summary.get("env_reward_mean", summary["reward_mean"])),
                    "done": _fmt_pct(summary["done_frac"]),
                },
            ),
            *_kv_lines(
                "trial",
                {
                    "roles": summary.get("trial_role_counts", {}),
                    "policy": int(summary.get("trial_policy_count", 0) or 0),
                    "search": int(summary.get("trial_search_count", 0) or 0),
                    "horizon": summary.get("trial_horizon_summary", "horizon_count=0 horizon_min=None horizon_max=None"),
                },
            ),
            *_kv_lines(
                "ppo_boundary",
                {
                    "evidence": int(summary.get("ppo_boundary_evidence_rows", 0) or 0),
                    "policy": int(summary.get("ppo_boundary_policy_rows", 0) or 0),
                    "search": int(summary.get("ppo_boundary_search_rows", 0) or 0),
                    "ppo_valid": int(summary.get("ppo_boundary_eligible_rows", summary.get("ppo_valid_count", 0)) or 0),
                    "search_evidence_only": int(summary.get("ppo_boundary_search_evidence_only_rows", 0) or 0),
                    "policy_invalid": int(summary.get("ppo_boundary_policy_invalid_rows", 0) or 0),
                    "valid_policy": _fmt_pct(summary.get("ppo_boundary_valid_policy_frac", 0.0)),
                    "valid_evidence": _fmt_pct(summary.get("ppo_boundary_valid_evidence_frac", 0.0)),
                },
            ),
            *_kv_lines(
                "gain",
                {
                    "source": summary.get("gain_source", "UNCONFIRMED"),
                    "style": _fmt_metric(summary.get("gain_style_mean")),
                    "physics": _fmt_metric(summary.get("gain_physics_mean")),
                    "repair_cost": _fmt_metric(summary.get("gain_repair_cost_mean")),
                    "total": _fmt_metric(summary.get("gain_total_mean")),
                    "mpjpe": _fmt_metric(summary.get("gain_style_mpjpe_mean")),
                    "velocity": _fmt_metric(summary.get("gain_style_velocity_mean")),
                    "acceleration": _fmt_metric(summary.get("gain_style_acceleration_mean")),
                    "root_orientation": _fmt_metric(summary.get("gain_style_root_orientation_mean")),
                    "success": _fmt_metric(summary.get("gain_physics_success_mean")),
                    "survival_quality_repaired": _fmt_metric(summary.get("gain_physics_survival_quality_repaired_mean")),
                    "survival_quality_noisy": _fmt_metric(summary.get("gain_physics_survival_quality_noisy_mean")),
                    "survival_quality": _fmt_metric(summary.get("gain_physics_survival_mean")),
                    "zmp": _fmt_metric(summary.get("gain_physics_zmp_mean")),
                    "contact": _fmt_metric(summary.get("gain_physics_contact_mean")),
                    "repair_norm": _fmt_metric(summary.get("gain_repair_norm_mean")),
                    "repair_temporal": _fmt_metric(summary.get("gain_repair_temporal_mean")),
                },
            ),
            *_kv_lines(
                "storage",
                {
                    "write": bool(summary["storage_write"]),
                    "size": int(summary["storage_size"]),
                    "mask_valid": _fmt_pct(summary["valid_mask_frac"]),
                    "valid_frac": _fmt_pct(summary["storage_valid_frac"]),
                    "train_reward": _fmt_num(summary.get("train_reward_mean", summary["storage_reward_mean"])),
                    "all_reward": _fmt_num(summary["storage_reward_mean"]),
                },
            ),
            *_kv_lines(
                "ppo",
                {
                    "single_update": bool(summary["single_update"]),
                    "update": bool(summary["ppo_update"]),
                    "valid": int(summary["ppo_valid_count"]),
                    "loss_total": _fmt_num(summary["ppo_total_loss"]),
                    "actor": _fmt_num(summary["ppo_actor_loss"]),
                    "value": _fmt_num(summary["ppo_value_loss"]),
                    "kl": _fmt_num(summary["ppo_approx_kl"]),
                    "clip": _fmt_pct(summary["ppo_clip_frac"]),
                    "status": _probe_status(summary),
                },
            ),
        ),
        flush=True,
    )
    if bool(summary.get("ppo_update", False)):
        # B1: Separate the same-batch PPO evidence by time. pre_* comes from
        # the loss forward before optimizer.step; post_* comes from the second
        # forward after optimizer.step on the same stored batch.
        print(
            _log_block(
                "[FrontRES Segment PPO Probe]",
                *_kv_lines(
                    "log_prob",
                    {
                        "old": _fmt_num(summary.get("ppo_old_log_prob_mean", 0.0)),
                        "new": _fmt_num(summary.get("ppo_new_log_prob_mean", 0.0)),
                    },
                ),
                *_kv_lines(
                    "pre_log_ratio",
                    {
                        "mean": _fmt_num(summary.get("ppo_pre_update_raw_log_ratio_mean", 0.0)),
                        "min": _fmt_num(summary.get("ppo_pre_update_raw_log_ratio_min", 0.0)),
                        "max": _fmt_num(summary.get("ppo_pre_update_raw_log_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "pre_ratio",
                    {
                        "clamped_mean": _fmt_num(summary.get("ppo_pre_update_clamped_ratio_mean", 0.0)),
                        "clamped_max": _fmt_num(summary.get("ppo_pre_update_clamped_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "kl",
                    {
                        "pre_distribution": _fmt_num(summary.get("ppo_pre_distribution_kl_mean", 0.0)),
                        "pre_logprob": _fmt_num(summary.get("ppo_pre_logprob_approx_kl", 0.0)),
                        "post_distribution": _fmt_num(
                            summary.get("ppo_post_update_distribution_kl_mean", 0.0)
                        ),
                        "post_logprob": _fmt_num(summary.get("ppo_post_update_logprob_approx_kl", 0.0)),
                        "distribution_available": bool(summary.get("ppo_distribution_kl_available", False)),
                    },
                ),
                *_kv_lines(
                    "post_log_ratio",
                    {
                        "mean": _fmt_num(summary.get("ppo_post_update_raw_log_ratio_mean", 0.0)),
                        "min": _fmt_num(summary.get("ppo_post_update_raw_log_ratio_min", 0.0)),
                        "max": _fmt_num(summary.get("ppo_post_update_raw_log_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "post_ratio",
                    {
                        "clamped_mean": _fmt_num(summary.get("ppo_post_update_clamped_ratio_mean", 0.0)),
                        "clamped_max": _fmt_num(summary.get("ppo_post_update_clamped_ratio_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "ratio_source",
                    {
                        "raw_action_old_mean_l2": _fmt_num(
                            summary.get("ppo_post_update_raw_action_old_mean_l2_mean", 0.0)
                        ),
                        "raw_action_old_mean_abs_max": _fmt_num(
                            summary.get("ppo_post_update_raw_action_old_mean_abs_max", 0.0)
                        ),
                        "raw_action_old_mean_abs_dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_raw_action_old_mean_abs_dim_mean", ())
                        ),
                        "raw_action_old_mean_abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_raw_action_old_mean_abs_dim_max", ())
                        ),
                    },
                ),
                *_kv_lines(
                    "ratio_sigma",
                    {
                        "old_dim_mean": _fmt_vec(summary.get("ppo_post_update_old_sigma_dim_mean", ())),
                        "new_dim_mean": _fmt_vec(summary.get("ppo_post_update_sigma_dim_mean", ())),
                    },
                ),
                *_kv_lines(
                    "ratio_mean_delta",
                    {
                        "dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_distribution_mean_delta_dim_mean", ())
                        ),
                        "abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_distribution_mean_delta_abs_dim_max", ())
                        ),
                    },
                ),
                *_kv_lines(
                    "ratio_contrib",
                    {
                        "log_ratio_dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_log_ratio_contrib_dim_mean", ())
                        ),
                        "log_ratio_abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_log_ratio_contrib_abs_dim_max", ())
                        ),
                        "log_jacobian_dim_mean": _fmt_vec(
                            summary.get("ppo_post_update_log_jacobian_dim_mean", ())
                        ),
                        "log_jacobian_abs_dim_max": _fmt_vec(
                            summary.get("ppo_post_update_log_jacobian_abs_dim_max", ())
                        ),
                    },
                ),
                *_kv_lines(
                    "trust",
                    {
                        "accepted": bool(summary.get("ppo_trust_region_accepted", 1)),
                        "rejected": int(summary.get("ppo_trust_region_rejected_count", 0)),
                        "lr_before": _fmt_num(summary.get("ppo_adaptive_lr_before", 0.0)),
                        "lr_after": _fmt_num(summary.get("ppo_adaptive_lr_after", 0.0)),
                        "desired_kl": _fmt_num(summary.get("ppo_adaptive_lr_desired_kl", 0.0)),
                        "schedule": str(summary.get("ppo_trust_region_schedule", "unknown")),
                        "rollback": bool(summary.get("ppo_trust_region_rollback_enabled", 0)),
                        "max_retries": int(summary.get("ppo_trust_region_max_retries", 0)),
                    },
                ),
                *_kv_lines(
                    "advantage",
                    {
                        "mean": _fmt_num(summary.get("ppo_advantage_mean", 0.0)),
                        "min": _fmt_num(summary.get("ppo_advantage_min", 0.0)),
                        "max": _fmt_num(summary.get("ppo_advantage_max", 0.0)),
                    },
                ),
                *_kv_lines(
                    "param_delta",
                    {
                        "max_abs": _fmt_num(summary.get("ppo_param_delta_max_abs", 0.0)),
                        "l2": _fmt_num(summary.get("ppo_param_delta_l2", 0.0)),
                        "changed": (
                            f"{int(summary.get('ppo_param_delta_changed', 0))}/"
                            f"{int(summary.get('ppo_param_delta_total', 0))}"
                        ),
                        "first": summary.get("ppo_param_delta_first_changed", ""),
                        "grad_norm": _fmt_num(summary.get("ppo_param_grad_norm", 0.0)),
                    },
                ),
            ),
            flush=True,
        )
