"""Read-only telemetry projection for committed recovery-aware transactions."""

from __future__ import annotations

import math
from dataclasses import asdict
from collections.abc import Mapping
from typing import Any

from rsl_rl.frontres.frontres_interfaces import FRONTRES_CHECKPOINT_FORMAT, FrontRESActiveTelemetryView
from rsl_rl.frontres.frontres_local_evaluation import FrontRESV017LocalEvaluationReport
from rsl_rl.frontres.frontres_relational_evaluation import FrontRESRelationalEvaluationReport
from rsl_rl.frontres.frontres_value_normalization import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FRONTRES_VALUE_NORMALIZER_DECAY,
    FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
)


def _finite(diagnostics: Mapping[str, Any], name: str) -> float:
    value = float(diagnostics.get(name, float("nan")))
    if not math.isfinite(value):
        raise RuntimeError(f"v017 formal result is missing finite {name}")
    return value


def build_frontres_formal_update_summary(result: Any) -> dict[str, Any]:
    """Project one committed grouped scalar update without legacy sampler facts."""

    ppo = getattr(result, "ppo_result", None)
    diagnostics = getattr(result, "diagnostics", None)
    if ppo is None or not isinstance(diagnostics, Mapping):
        raise TypeError("v017 formal summary requires PPO result and immutable diagnostics")
    relational = str(diagnostics.get("scalar_target_id", "")) == "none"
    summary = {
        "transaction_id": str(getattr(result, "transaction_id", "")),
        "policy_snapshot_id": str(getattr(result, "policy_snapshot_id", "")),
        "update_steps": 1,
        "update_count": int(getattr(result, "update_invocation_count", 0)),
        "ppo_valid_count": int(getattr(result, "valid_row_count", 0)),
        "policy_attempt_count": int(getattr(result, "policy_attempt_count", 0)),
        "segment_count": int(getattr(result, "segment_count", 0)),
        "source_count": int(getattr(result, "source_count", 0)),
        "optimizer_step_before": int(getattr(result, "optimizer_step_before", -1)),
        "optimizer_step_after": int(getattr(result, "optimizer_step_after", -1)),
        "optimizer_step_delta": int(getattr(result, "optimizer_step_delta", -1)),
        "training_contract_id": str(diagnostics.get("training_contract_id", "")),
        "gain_contract_id": str(diagnostics.get("gain_contract_id", "")),
        "method_contract_id": str(diagnostics.get("method_contract_id", "")),
        "optimization_contract_id": str(diagnostics.get("optimization_contract_id", "")),
        "scalar_target_id": str(diagnostics.get("scalar_target_id", "")),
        "physics_schema_id": str(diagnostics.get("physics_schema_id", "")),
        "grouped_schema_id": str(diagnostics.get("grouped_schema_id", "")),
        "training_iteration": int(diagnostics.get("training_iteration", -1)),
        "curriculum_fingerprint": str(diagnostics.get("curriculum_fingerprint", "")),
        "k_stage_index": int(diagnostics.get("k_stage_index", -1)),
        "k_stage_iteration": int(diagnostics.get("k_stage_iteration", -1)),
        "warmup_phase": str(diagnostics.get("warmup_phase", "")),
        "warmup_phase_iteration": int(diagnostics.get("warmup_phase_iteration", -1)),
        "actor_loss_weight": float(diagnostics.get("actor_loss_weight", float("nan"))),
        "dr_stage_fingerprint": str(diagnostics.get("dr_stage_fingerprint", "")),
        "dr_progress": float(diagnostics.get("dr_progress", float("nan"))),
        "d_cap": float(diagnostics.get("d_cap", float("nan"))),
        "dr_class_by_segment": tuple(diagnostics.get("dr_class_by_segment", ())),
        "dr_strength_by_segment": tuple(float(value) for value in diagnostics.get("dr_strength_by_segment", ())),
        "active_k": int(diagnostics.get("active_k", -1)),
        "active_m": int(diagnostics.get("active_m", -1)),
        "warmup_phase": str(diagnostics.get("warmup_phase", "")),
        "actor_loss_weight": float(diagnostics.get("actor_loss_weight", 1.0)),
        "actor_learning_rate": _finite(diagnostics, "actor_learning_rate"),
        "critic_learning_rate": _finite(diagnostics, "critic_learning_rate"),
        "ppo_total_loss_mean": float(ppo.total_loss.detach().cpu().item()),
        "ppo_actor_loss_mean": float(ppo.actor_loss.detach().cpu().item()),
        "relational": relational,
    }
    if relational:
        summary.update(
            {
                "relational_edge_count": int(getattr(ppo, "edge_count", 0)),
                "relational_status": str(getattr(ppo, "status", "")),
                "relational_entropy_mean": float(ppo.entropy.detach().cpu().item()),
            }
        )
    else:
        summary.update(
            {
                "ppo_value_loss_mean": float(ppo.value_loss.detach().cpu().item()),
                "ppo_approx_kl_mean": float(ppo.approx_kl),
                "ppo_clip_frac_mean": float(ppo.clip_frac),
                "grouped_motion_count": int(ppo.grouped_motion_count),
                "grouped_segment_count": int(ppo.grouped_segment_count),
                "grouped_attempt_count": int(ppo.grouped_attempt_count),
                "grouped_valid_step_count": int(ppo.grouped_valid_step_count),
            }
        )
    summary["frontres_transaction_telemetry"] = build_frontres_transaction_telemetry(result, ppo=ppo)
    return summary


def _build_relational_transaction_telemetry(result: Any, *, ppo: Any, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    reports = diagnostics.get("v009_relational_reports")
    if not isinstance(reports, tuple) or not reports:
        raise RuntimeError("FRS-TRAIN-v025 telemetry requires immutable relational reports")
    scenario_ids: list[str] = []
    noisy_hashes: list[str] = []
    comparable_counts: list[int] = []
    outcomes: list[dict[str, Any]] = []
    edges: tuple[tuple[int, int], ...] | None = None
    transaction_id = str(getattr(result, "transaction_id", ""))
    for report in reports:
        if not isinstance(report, FrontRESRelationalEvaluationReport):
            raise TypeError("FRS-TRAIN-v025 telemetry rejects scalar local reports")
        report.validate()
        if report.transaction_id != transaction_id:
            raise RuntimeError("FRS-TRAIN-v025 telemetry reports mix transaction identity")
        scenario_ids.extend(report.scenario_ids)
        noisy_hashes.extend(report.noisy_segment_hashes)
        comparable_counts.extend(report.comparable_pair_count_by_row)
        report_outcomes = getattr(report, "outcomes", ())
        if not isinstance(report_outcomes, tuple) or len(report_outcomes) != int(report.policy_row_count):
            raise RuntimeError("FRS-TRAIN-v025 telemetry requires row-aligned Outcome evidence")
        outcomes.extend(asdict(value) for value in report_outcomes)
        if edges is None:
            edges = tuple(report.preference_edges)
        elif tuple(report.preference_edges) != edges:
            raise RuntimeError("FRS-TRAIN-v025 telemetry reports mix preference edges")
    row_count = int(getattr(result, "policy_attempt_count", -1))
    if len(scenario_ids) != row_count or len(noisy_hashes) != row_count:
        raise RuntimeError("FRS-TRAIN-v025 telemetry lost relational row identity")
    edges = tuple(edges or ())
    if edges != tuple(diagnostics.get("preference_edges", ())):
        raise RuntimeError("FRS-TRAIN-v025 telemetry edges disagree with committed diagnostics")
    actor_credit = tuple(float(value) for value in ppo.actor_credit.detach().cpu().tolist())
    if len(actor_credit) != row_count or not all(math.isfinite(value) for value in actor_credit):
        raise RuntimeError("FRS-TRAIN-v025 telemetry requires finite row-aligned Actor credit")
    telemetry = {
        "transaction_id": transaction_id,
        "policy_snapshot_id": str(getattr(result, "policy_snapshot_id", "")),
        "method_contract_id": "FRS-METHOD-v026",
        "gain_contract_id": "FRS-GAIN-v009",
        "optimization_contract_id": "FRS-PPO-v013",
        "training_contract_id": "FRS-TRAIN-v025",
        "scalar_target_id": "none",
        "physics_schema_id": "hierarchical-relational-evidence-v1",
        "grouped_schema_id": "relational-preference-edge-v1",
        "checkpoint_format": "frontres-v025-checkpoint-v20",
        "critic_value_kind": "inert-legacy-compat",
        "critic_input_dim": 449,
        "critic_action_conditioned": False,
        "critic_target_id": "none",
        "critic_support_context_id": "none",
        "return_utility_id": "none",
        "return_utility_scale": 0.0,
        "gradient_clip_identity": "actor-only-relational-v1",
        "gradient_clip_max_norm": _finite(diagnostics, "gradient_clip_max_norm"),
        "actor_observation_dim": 158,
        "gmt_observation_dim": 770,
        "critic_value_normalization_id": "none",
        "critic_value_scale": 0.0,
        "critic_value_normalizer_decay": 0.0,
        "critic_value_normalizer_scale_floor": 0.0,
        "critic_value_normalizer_update_count_before": 0,
        "critic_value_normalizer_update_count_after": 0,
        "actor_gradient_pre_clip_norm": _finite(diagnostics, "actor_gradient_pre_clip_norm"),
        "actor_gradient_post_clip_norm": _finite(diagnostics, "actor_gradient_post_clip_norm"),
        "actor_gradient_clip_coefficient": _finite(diagnostics, "actor_gradient_clip_coefficient"),
        "actor_gradient_nonzero_parameter_count": int(
            diagnostics.get("actor_gradient_nonzero_parameter_count", 0)
        ),
        "actor_parameter_delta_l2": _finite(diagnostics, "actor_parameter_delta_l2"),
        "action_l2_mean": _finite(diagnostics, "action_l2_mean"),
        "action_l2_max": _finite(diagnostics, "action_l2_max"),
        "action_nonzero_fraction": _finite(diagnostics, "action_nonzero_fraction"),
        "critic_gradient_post_clip_norm": 0.0,
        "active_k": int(diagnostics.get("active_k", -1)),
        "active_m": int(diagnostics.get("active_m", -1)),
        "selected_segment_count": int(diagnostics.get("selected_segment_count", -1)),
        "policy_row_count": row_count,
        "role_row_count": int(diagnostics.get("role_row_count", 2 * row_count)),
        "optimizer_step_delta": int(getattr(result, "optimizer_step_delta", -1)),
        "update_count": int(getattr(result, "update_invocation_count", 0)),
        "actor_learning_rate": _finite(diagnostics, "actor_learning_rate"),
        "critic_learning_rate": 0.0,
        "training_iteration": int(diagnostics.get("training_iteration", -1)),
        "curriculum_fingerprint": str(diagnostics.get("curriculum_fingerprint", "")),
        "k_stage_index": int(diagnostics.get("k_stage_index", -1)),
        "k_stage_iteration": int(diagnostics.get("k_stage_iteration", -1)),
        "warmup_phase": str(diagnostics.get("warmup_phase", "")),
        "warmup_phase_iteration": int(diagnostics.get("warmup_phase_iteration", -1)),
        "actor_loss_weight": float(diagnostics.get("actor_loss_weight", 0.0)),
        "dr_stage_fingerprint": str(diagnostics.get("dr_stage_fingerprint", "")),
        "dr_progress": float(diagnostics.get("dr_progress", 0.0)),
        "d_cap": float(diagnostics.get("d_cap", 0.0)),
        "dr_class_by_segment": tuple(diagnostics.get("dr_class_by_segment", ())),
        "dr_strength_by_segment": tuple(diagnostics.get("dr_strength_by_segment", ())),
        "edge_count": int(diagnostics.get("edge_count", getattr(ppo, "edge_count", -1))),
        "valid_count": int(getattr(ppo, "valid_count", -1)),
        "status": str(getattr(ppo, "status", "")),
        "preference_edges": edges,
        "actor_credit": actor_credit,
        "scenario_ids": tuple(scenario_ids),
        "noisy_segment_hashes": tuple(noisy_hashes),
        "comparable_pair_count_by_row": tuple(comparable_counts),
        "outcome_schema_id": "frs-gain-v009-outcome-v1",
        "relational_outcomes": tuple(outcomes),
        "outer_replay": diagnostics.get("outer_replay"),
        "return_feedback": False,
        "priority_feedback": False,
        "ppo_feedback": False,
    }
    FrontRESActiveTelemetryView.from_mapping(telemetry)
    return telemetry


def build_frontres_transaction_telemetry(result: Any, *, ppo: Any) -> dict[str, Any]:
    """Serialize v007 owner output and lifecycle facts without recomputation."""

    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("v017 telemetry requires sealed transaction diagnostics")
    if str(diagnostics.get("scalar_target_id", "")) == "none":
        return _build_relational_transaction_telemetry(result, ppo=ppo, diagnostics=diagnostics)
    reports = diagnostics.get("v007_recovery_aware_reports")
    if not isinstance(reports, tuple) or not reports:
        raise RuntimeError("v017 telemetry requires immutable recovery-aware reports")
    transaction_id = str(getattr(result, "transaction_id", ""))
    fields = {
        name: []
        for name in (
            "scenario_ids",
            "noisy_segment_hashes",
            "policy_actions",
            "valid_policy_row_mask",
            "intent_remaining_noisy",
            "intent_remaining_repaired",
            "physics_remaining_noisy",
            "physics_remaining_repaired",
            "intent_channel_noisy",
            "intent_channel_repaired",
            "physics_channel_noisy",
            "physics_channel_repaired",
            "support_foot_drift_noisy",
            "support_foot_drift_repaired",
            "intent_gain",
            "physics_gain",
            "recovery_pressure",
            "weighted_physics_gain",
            "repair_cost",
            "repair_penalty",
            "cost_free_score",
            "gain_total",
            "policy_values",
            "raw_advantages",
            "expected_support_steps",
            "contact_clean_steps",
            "contact_noisy_steps",
            "contact_repair_steps",
            "zmp_clean_steps",
            "zmp_noisy_steps",
            "zmp_repair_steps",
            "survival_clean_steps",
            "survival_noisy_steps",
            "survival_repair_steps",
            "contact_violation_repair_steps",
            "zmp_applicable_repair_steps",
            "zmp_violation_repair_steps",
            "zmp_recovery_repair_steps",
            "unplanned_contact_repair_steps",
            "lateral_roll_repair_steps",
            "lateral_roll_cumulative_mean_repair_steps",
            "sustained_lean_repair",
        )
    }
    clean_counts: list[int] = []
    noisy_counts: list[int] = []
    gain_identity: tuple[Any, ...] | None = None
    for report in reports:
        if not isinstance(report, FrontRESV017LocalEvaluationReport):
            raise TypeError("v017 telemetry rejects legacy local reports")
        report.validate()
        if report.transaction_id != transaction_id:
            raise RuntimeError("v017 telemetry reports mix transaction identity")
        report_identity = (
            report.intent_scales,
            report.physics_scales,
            report.translation_repair_scale,
            report.rotation_repair_scale,
            report.beta,
            report.gain_contract_id,
        )
        if gain_identity is None:
            gain_identity = report_identity
        elif report_identity != gain_identity:
            raise RuntimeError("v017 telemetry reports mix Gain scale/beta identity")
        for name in fields:
            fields[name].extend(getattr(report, name))
        clean_counts.extend(report.clean_execution_count)
        noisy_counts.extend(report.noisy_execution_count)

    row_order = tuple(int(value) for value in diagnostics.get("v007_diagnostic_report_row_order", ()))
    row_count = len(fields["policy_actions"])
    if sorted(row_order) != list(range(row_count)):
        raise RuntimeError("v017 telemetry requires an exact report-to-PPO row permutation")
    fields = {name: tuple(values[index] for index in row_order) for name, values in fields.items()}
    if row_count != int(getattr(result, "policy_attempt_count", -1)) or not all(fields["valid_policy_row_mask"]):
        raise RuntimeError("v017 telemetry requires every sealed Repair policy row")
    if any(value != 1 for value in clean_counts + noisy_counts):
        raise RuntimeError("v017 telemetry requires one Clean and one Noisy execution per Segment")

    expected_ids = {
        "method_contract_id": "FRS-METHOD-v025",
        "gain_contract_id": "FRS-GAIN-v008",
        "optimization_contract_id": "FRS-PPO-v012",
        "training_contract_id": "FRS-TRAIN-v024",
        "scalar_target_id": "symmetric-log-recovery-aware-utility-v1",
        "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
        "grouped_schema_id": "grouped-all-attempt-scalar-v1",
    }
    for name, expected in expected_ids.items():
        if str(diagnostics.get(name, "")) != expected:
            raise RuntimeError(f"v017 telemetry requires {name}={expected}")

    prepared = tuple(float(value) for value in getattr(ppo, "prepared_advantages", ()))
    if len(prepared) != row_count or not all(math.isfinite(value) for value in prepared):
        raise RuntimeError("v017 telemetry requires one finite scaled advantage per Repair row")
    actions = fields["policy_actions"]
    action_l2 = [math.sqrt(sum(float(value) ** 2 for value in row)) for row in actions]
    gains = tuple(float(value) for value in fields["gain_total"])
    if not all(math.isfinite(value) for value in gains):
        raise RuntimeError("v017 telemetry requires finite G_total rows")
    active_m = int(diagnostics.get("active_m", -1))
    selected_segments = int(diagnostics.get("selected_segment_count", -1))
    role_rows = int(diagnostics.get("role_row_count", -1))
    if selected_segments != 8 or row_count != 8 * active_m or role_rows != 16 * active_m:
        raise RuntimeError("v022 telemetry lost exact eight-Segment x M layout")
    critic_targets = tuple(float(value) for value in diagnostics.get("critic_value_targets", ()))
    segment_targets = tuple(float(value) for value in diagnostics.get("critic_segment_target_means", ()))
    actor_advantages = tuple(float(value) for value in diagnostics.get("actor_advantages", ()))
    raw_returns = tuple(float(value) for value in diagnostics.get("raw_returns", ()))
    utility_returns = tuple(float(value) for value in diagnostics.get("utility_returns", ()))
    return_utility_id = str(diagnostics.get("return_utility_id", ""))
    return_utility_scale = _finite(diagnostics, "return_utility_scale")
    if (
        len(critic_targets) != row_count
        or len(actor_advantages) != row_count
        or len(segment_targets) != selected_segments
        or len(raw_returns) != row_count
        or len(utility_returns) != row_count
        or not all(
            math.isfinite(value)
            for value in critic_targets + segment_targets + actor_advantages + raw_returns + utility_returns
        )
        or return_utility_id != "symmetric-log-gain-g0-1-v1"
        or return_utility_scale != 1.0
        or raw_returns != tuple(float(value) for value in getattr(ppo, "raw_returns", ()))
        or utility_returns != tuple(float(value) for value in getattr(ppo, "utility_returns", ()))
        or critic_targets != tuple(float(value) for value in getattr(ppo, "critic_value_targets", ()))
        or actor_advantages != tuple(float(value) for value in getattr(ppo, "actor_advantages", ()))
    ):
        raise RuntimeError("TRAIN-v024 telemetry has malformed raw/utility/target/advantage rows")
    required_v016_fields = {
        "actor_observation_dim",
        "critic_observation_dim",
        "gmt_observation_dim",
        "critic_value_kind",
        "critic_action_conditioned",
        "critic_target_id",
        "return_utility_id",
        "return_utility_scale",
        "critic_support_context_id",
        "gradient_clip_identity",
        "gradient_clip_max_norm",
        "critic_raw_value_loss",
        "critic_value_normalization_id",
        "critic_value_scale",
        "critic_value_normalizer_decay",
        "critic_value_normalizer_scale_floor",
        "critic_value_normalizer_mean_before",
        "critic_value_normalizer_mean_after",
        "critic_value_normalizer_second_moment_before",
        "critic_value_normalizer_second_moment_after",
        "critic_value_normalizer_update_count_before",
        "critic_value_normalizer_update_count_after",
    }
    missing_v016_fields = tuple(sorted(required_v016_fields.difference(diagnostics)))
    if missing_v016_fields:
        raise RuntimeError(f"TRAIN-v021 telemetry is missing required fields: {missing_v016_fields}")
    if (
        int(diagnostics.get("actor_observation_dim", -1)) != 158
        or int(diagnostics.get("critic_observation_dim", -1)) != 449
        or int(diagnostics.get("gmt_observation_dim", -1)) != 770
        or str(diagnostics.get("critic_value_kind", "")) != "state_value"
        or diagnostics.get("critic_action_conditioned") is not False
        or str(diagnostics.get("critic_target_id", ""))
        != "scenario-current-exact-m4-mean-symlog-v1"
        or str(diagnostics.get("return_utility_id", "")) != "symmetric-log-gain-g0-1-v1"
        or _finite(diagnostics, "return_utility_scale") != 1.0
        or str(diagnostics.get("critic_support_context_id", "")) != "action-pre-support-plan-kmax32-v1"
        or str(diagnostics.get("gradient_clip_identity", "")) != "separate-actor-critic-v1"
        or _finite(diagnostics, "gradient_clip_max_norm") != 0.5
    ):
        raise RuntimeError("TRAIN-v021 telemetry lost state-value utility or gradient identity")
    value_normalization_id = str(diagnostics["critic_value_normalization_id"])
    value_scale = _finite(diagnostics, "critic_value_scale")
    value_decay = _finite(diagnostics, "critic_value_normalizer_decay")
    value_scale_floor = _finite(diagnostics, "critic_value_normalizer_scale_floor")
    value_mean_before = _finite(diagnostics, "critic_value_normalizer_mean_before")
    value_mean_after = _finite(diagnostics, "critic_value_normalizer_mean_after")
    value_second_before = _finite(diagnostics, "critic_value_normalizer_second_moment_before")
    value_second_after = _finite(diagnostics, "critic_value_normalizer_second_moment_after")
    value_count_before = int(diagnostics["critic_value_normalizer_update_count_before"])
    value_count_after = int(diagnostics["critic_value_normalizer_update_count_after"])
    raw_value_loss = _finite(diagnostics, "critic_raw_value_loss")
    scaled_value_loss = float(ppo.value_loss.detach().cpu().item())
    if (
        value_normalization_id != FRONTRES_VALUE_NORMALIZATION_ID
        or value_decay != FRONTRES_VALUE_NORMALIZER_DECAY
        or value_scale_floor != FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR
        or value_scale < 1.0
        or value_count_before < 0
        or value_count_after != value_count_before + 1
        or value_second_before + 1.0e-12 < value_mean_before**2
        or value_second_after + 1.0e-12 < value_mean_after**2
        or raw_value_loss < 0.0
        or scaled_value_loss < 0.0
        or not math.isclose(
            scaled_value_loss,
            raw_value_loss / value_scale**2,
            rel_tol=1.0e-5,
            abs_tol=1.0e-7,
        )
    ):
        raise RuntimeError("TRAIN-v021 telemetry has invalid Critic value-normalizer transition")

    outer_replay = diagnostics.get("outer_replay")
    outer_sources = tuple(str(value) for value in diagnostics.get("outer_replay_sources", ()))
    outer_key_digests = tuple(str(value) for value in diagnostics.get("outer_replay_scenario_key_digests", ()))
    outer_seeds = tuple(int(value) for value in diagnostics.get("outer_replay_perturbation_seeds", ()))
    outer_utility_means = tuple(float(value) for value in diagnostics.get("outer_replay_utility_means", ()))
    outer_old_value_means = tuple(float(value) for value in diagnostics.get("outer_replay_old_value_means", ()))
    if not isinstance(outer_replay, Mapping):
        raise RuntimeError("TRAIN-v021 telemetry requires committed outer replay evidence")
    outer_score_kind = str(outer_replay.get("score_kind", ""))
    outer_critic_calibration_values = tuple(
        float(value) for value in outer_replay.get("critic_calibration_values", ())
    )
    outer_repair_spread_values = tuple(float(value) for value in outer_replay.get("repair_spread_values", ()))
    outer_priority_scores = tuple(float(value) for value in outer_replay.get("priority_scores", ()))
    outer_target_means = tuple(float(value) for value in outer_replay.get("critic_target_means", ()))
    outer_current_means = tuple(float(value) for value in outer_replay.get("current_utility_means", ()))
    outer_outcome_variances = tuple(float(value) for value in outer_replay.get("outcome_variances", ()))
    outer_standard_errors = tuple(float(value) for value in outer_replay.get("standard_errors", ()))
    outer_confidence_half_widths = tuple(
        float(value) for value in outer_replay.get("confidence_half_widths", ())
    )
    outer_current_sample_counts = tuple(
        int(value) for value in outer_replay.get("current_sample_counts", ())
    )
    outer_visit_counts = tuple(int(value) for value in outer_replay.get("visit_counts", ()))
    outer_staleness = tuple(int(value) for value in outer_replay.get("staleness", ()))
    if (
        int(outer_replay.get("state_delta", -1)) != 1
        or len(outer_sources) != selected_segments
        or any(source not in {"global", "replay", "review"} for source in outer_sources)
        or len(outer_key_digests) != selected_segments
        or len(set(outer_key_digests)) != selected_segments
        or any(len(value) != 64 for value in outer_key_digests)
        or len(outer_seeds) != selected_segments
        or any(value < 0 for value in outer_seeds)
        or any(
            len(values) != selected_segments or not all(math.isfinite(value) for value in values)
            for values in (
                outer_utility_means,
                outer_old_value_means,
                outer_critic_calibration_values,
                outer_repair_spread_values,
                outer_priority_scores,
                outer_target_means,
                outer_current_means,
            )
        )
        or any(
            len(values) != selected_segments
            or not all(math.isfinite(value) and value >= 0.0 for value in values)
            for values in (
                outer_outcome_variances,
                outer_standard_errors,
                outer_confidence_half_widths,
            )
        )
        or outer_current_sample_counts != (active_m,) * selected_segments
        or outer_score_kind not in {"critic_calibration", "repair_spread"}
        or len(outer_visit_counts) != selected_segments
        or any(value <= 0 for value in outer_visit_counts)
        or len(outer_staleness) != selected_segments
        or any(value != 0 for value in outer_staleness)
    ):
        raise RuntimeError("TRAIN-v024 telemetry has malformed outer replay commit evidence")
    source_index = tuple(int(value) for value in diagnostics.get("source_index", ()))
    for source in range(8):
        source_advantages = tuple(
            actor_advantages[row]
            for row, source_value in enumerate(source_index)
            if source_value == source
        )
        advantage_mean = sum(source_advantages) / active_m
        expected_calibration = max(
            abs(outer_old_value_means[source] - outer_target_means[source])
            - outer_confidence_half_widths[source],
            0.0,
        )
        expected_spread = sum(abs(value - advantage_mean) for value in source_advantages) / active_m
        if not math.isclose(
            outer_target_means[source], segment_targets[source], rel_tol=1.0e-6, abs_tol=1.0e-6
        ):
            raise RuntimeError("TRAIN-v024 Replay target differs from the PPO Critic target")
        if not math.isclose(
            outer_current_means[source], outer_utility_means[source], rel_tol=1.0e-6, abs_tol=1.0e-6
        ):
            raise RuntimeError("TRAIN-v024 Replay current utility mean differs from the current M4 evidence")
        if not math.isclose(
            outer_target_means[source], outer_current_means[source], rel_tol=1.0e-6, abs_tol=1.0e-6
        ):
            raise RuntimeError("TRAIN-v024 Critic target must equal the current M4 mean")
        if not math.isclose(
            outer_critic_calibration_values[source], expected_calibration, rel_tol=1.0e-6, abs_tol=1.0e-6
        ):
            raise RuntimeError("TRAIN-v024 Replay calibration score differs from current-M4 excess error")
        if not math.isclose(
            outer_repair_spread_values[source], expected_spread, rel_tol=1.0e-6, abs_tol=1.0e-6
        ):
            raise RuntimeError("TRAIN-v021 outer replay spread score differs from centered Repair advantage spread")

    telemetry = {
        "transaction_id": transaction_id,
        "policy_snapshot_id": str(getattr(result, "policy_snapshot_id", "")),
        "source_index": tuple(int(value) for value in diagnostics.get("source_index", ())),
        "trial_index": tuple(int(value) for value in diagnostics.get("trial_index", ())),
        **fields,
        "intent_scales": gain_identity[0],
        "physics_scales": gain_identity[1],
        "translation_repair_scale": gain_identity[2],
        "rotation_repair_scale": gain_identity[3],
        "beta": gain_identity[4],
        "scaled_advantages": prepared,
        "clean_execution_count": tuple(clean_counts),
        "noisy_execution_count": tuple(noisy_counts),
        "active_k": int(diagnostics.get("active_k", -1)),
        "active_m": active_m,
        "selected_segment_count": selected_segments,
        "policy_row_count": row_count,
        "role_row_count": role_rows,
        "positive_gain_fraction": sum(value > 0.0 for value in gains) / len(gains),
        "negative_gain_fraction": sum(value < 0.0 for value in gains) / len(gains),
        "action_l2_mean": sum(action_l2) / len(action_l2),
        "return_mean": _finite(diagnostics, "return_mean"),
        "return_min": _finite(diagnostics, "return_min"),
        "return_max": _finite(diagnostics, "return_max"),
        "return_abs_mean": _finite(diagnostics, "return_abs_mean"),
        "advantage_mean": float(ppo.advantage_mean),
        "advantage_min": float(ppo.advantage_min),
        "advantage_max": float(ppo.advantage_max),
        "advantage_scale": float(ppo.advantage_scale),
        "advantage_sign_flip_count": int(ppo.advantage_sign_flip_count),
        "grouped_reduction_active": bool(ppo.grouped_reduction_active),
        "grouped_motion_mass_shares": tuple(ppo.grouped_motion_mass_shares),
        "grouped_segment_mass_shares": tuple(ppo.grouped_segment_mass_shares),
        "grouped_attempt_mass_shares": tuple(ppo.grouped_attempt_mass_shares),
        "actor_observation_dim": int(diagnostics["actor_observation_dim"]),
        "critic_observation_dim": int(diagnostics["critic_observation_dim"]),
        "critic_input_dim": int(diagnostics["critic_observation_dim"]),
        "gmt_observation_dim": int(diagnostics["gmt_observation_dim"]),
        "critic_value_kind": str(diagnostics["critic_value_kind"]),
        "critic_action_conditioned": bool(diagnostics["critic_action_conditioned"]),
        "critic_target_id": str(diagnostics["critic_target_id"]),
        "return_utility_id": return_utility_id,
        "return_utility_scale": return_utility_scale,
        "raw_returns": raw_returns,
        "utility_returns": utility_returns,
        "critic_support_context_id": str(diagnostics["critic_support_context_id"]),
        "critic_value_targets": critic_targets,
        "critic_segment_target_means": segment_targets,
        "actor_advantages": actor_advantages,
        "critic_raw_value_loss": raw_value_loss,
        "critic_scaled_value_loss": scaled_value_loss,
        "critic_value_normalization_id": value_normalization_id,
        "critic_value_scale": value_scale,
        "critic_value_normalizer_decay": value_decay,
        "critic_value_normalizer_scale_floor": value_scale_floor,
        "critic_value_normalizer_mean_before": value_mean_before,
        "critic_value_normalizer_mean_after": value_mean_after,
        "critic_value_normalizer_second_moment_before": value_second_before,
        "critic_value_normalizer_second_moment_after": value_second_after,
        "critic_value_normalizer_update_count_before": value_count_before,
        "critic_value_normalizer_update_count_after": value_count_after,
        "gradient_clip_identity": str(diagnostics["gradient_clip_identity"]),
        "gradient_clip_max_norm": _finite(diagnostics, "gradient_clip_max_norm"),
        "actor_gradient_pre_clip_norm": _finite(diagnostics, "actor_gradient_pre_clip_norm"),
        "actor_gradient_post_clip_norm": _finite(diagnostics, "actor_gradient_post_clip_norm"),
        "actor_gradient_clip_coefficient": _finite(diagnostics, "actor_gradient_clip_coefficient"),
        "critic_gradient_pre_clip_norm": _finite(diagnostics, "critic_gradient_pre_clip_norm"),
        "critic_gradient_post_clip_norm": _finite(diagnostics, "critic_gradient_post_clip_norm"),
        "critic_gradient_clip_coefficient": _finite(diagnostics, "critic_gradient_clip_coefficient"),
        "actor_gradient_nonzero_parameter_count": int(
            diagnostics.get("actor_gradient_nonzero_parameter_count", -1)
        ),
        "critic_gradient_nonzero_parameter_count": int(
            diagnostics.get("critic_gradient_nonzero_parameter_count", -1)
        ),
        "gradient_pre_clip_norm": _finite(diagnostics, "gradient_pre_clip_norm"),
        "gradient_post_clip_norm": _finite(diagnostics, "gradient_post_clip_norm"),
        "gradient_parameter_count": int(diagnostics.get("gradient_parameter_count", -1)),
        "gradient_nonzero_parameter_count": int(diagnostics.get("gradient_nonzero_parameter_count", -1)),
        "optimizer_candidate_actor_delta_l2": float(diagnostics.get("optimizer_candidate_actor_delta_l2", 0.0)),
        "committed_actor_delta_l2": float(diagnostics.get("committed_actor_delta_l2", 0.0)),
        "actor_optimizer_state_restored": bool(diagnostics.get("actor_optimizer_state_restored", False)),
        "update_count": int(getattr(result, "update_invocation_count", 0)),
        "optimizer_step_delta": int(getattr(result, "optimizer_step_delta", -1)),
        "checkpoint_format": FRONTRES_CHECKPOINT_FORMAT,
        "training_iteration": int(diagnostics.get("training_iteration", -1)),
        "curriculum_fingerprint": str(diagnostics.get("curriculum_fingerprint", "")),
        "k_stage_index": int(diagnostics.get("k_stage_index", -1)),
        "k_stage_iteration": int(diagnostics.get("k_stage_iteration", -1)),
        "warmup_phase": str(diagnostics.get("warmup_phase", "")),
        "warmup_phase_iteration": int(diagnostics.get("warmup_phase_iteration", -1)),
        "actor_loss_weight": float(diagnostics.get("actor_loss_weight", float("nan"))),
        "actor_learning_rate": _finite(diagnostics, "actor_learning_rate"),
        "critic_learning_rate": _finite(diagnostics, "critic_learning_rate"),
        "dr_stage_fingerprint": str(diagnostics.get("dr_stage_fingerprint", "")),
        "dr_progress": float(diagnostics.get("dr_progress", float("nan"))),
        "d_cap": float(diagnostics.get("d_cap", float("nan"))),
        "dr_class_by_segment": tuple(diagnostics.get("dr_class_by_segment", ())),
        "dr_strength_by_segment": tuple(float(value) for value in diagnostics.get("dr_strength_by_segment", ())),
        "critic_parameter_delta": dict(diagnostics.get("critic_parameter_delta", {})),
        "actor_std_parameter_delta": dict(diagnostics.get("actor_std_parameter_delta", {})),
        "outer_replay_state_delta": int(outer_replay["state_delta"]),
        "outer_replay_sources": outer_sources,
        "outer_replay_scenario_key_digests": outer_key_digests,
        "outer_replay_perturbation_seeds": outer_seeds,
        "outer_replay_utility_means": outer_utility_means,
        "outer_replay_old_value_means": outer_old_value_means,
        "outer_replay_score_kind": outer_score_kind,
        "outer_replay_critic_calibration_values": outer_critic_calibration_values,
        "outer_replay_repair_spread_values": outer_repair_spread_values,
        "outer_replay_priority_scores": outer_priority_scores,
        "outer_replay_critic_target_means": outer_target_means,
        "outer_replay_current_utility_means": outer_current_means,
        "outer_replay_outcome_variances": outer_outcome_variances,
        "outer_replay_standard_errors": outer_standard_errors,
        "outer_replay_confidence_half_widths": outer_confidence_half_widths,
        "outer_replay_current_sample_counts": outer_current_sample_counts,
        "outer_replay_visit_counts": outer_visit_counts,
        "outer_replay_staleness": outer_staleness,
        "outer_replay_record_count": int(outer_replay.get("record_count", -1)),
        "outer_replay_pool_sizes": (
            int(outer_replay.get("replay_pool_size", -1)),
            int(outer_replay.get("review_pool_size", -1)),
        ),
        "return_feedback": False,
        "priority_feedback": False,
        "ppo_feedback": False,
        **expected_ids,
    }
    if len(telemetry["dr_class_by_segment"]) != 8 or len(telemetry["dr_strength_by_segment"]) != 8:
        raise RuntimeError("FRS-TRAIN-v024 telemetry requires eight sealed Segment DR class/strength values")
    FrontRESActiveTelemetryView.from_mapping(telemetry)
    return telemetry


_v015_sealed_transaction_telemetry = build_frontres_transaction_telemetry


def require_frontres_committed_result(runner: Any, result: Any) -> dict[str, Any]:
    """Require one complete receipt before iteration advance or checkpoint save."""

    summary = build_frontres_formal_update_summary(result)
    if (
        summary["update_count"] != 1
        or summary["optimizer_step_delta"] != 1
        or summary["optimizer_step_after"] != summary["optimizer_step_before"] + 1
        or (
            summary["relational"]
            and summary["relational_edge_count"] <= 0
        )
        or (
            not summary["relational"]
            and summary["grouped_attempt_count"] != summary["policy_attempt_count"]
        )
    ):
        raise RuntimeError("v017 formal result is not one complete grouped update")
    state = getattr(runner, "_frontres_checkpoint_transaction_state", None)
    receipt = state.get("receipt") if isinstance(state, Mapping) and state.get("state") == "committed" else None
    if not isinstance(receipt, Mapping):
        raise RuntimeError("v017 formal result requires a committed checkpoint receipt")
    telemetry = summary["frontres_transaction_telemetry"]
    if (
        receipt.get("transaction_id") != summary["transaction_id"]
        or int(receipt.get("optimizer_step_delta", -1)) != 1
        or int(receipt.get("collected_policy_attempt_count", -1)) != summary["policy_attempt_count"]
    ):
        raise RuntimeError("v017 committed receipt disagrees with the grouped update")
    for name in (
        "active_k", "active_m", "training_iteration", "curriculum_fingerprint",
        "dr_stage_fingerprint", "dr_progress", "d_cap",
    ):
        if receipt.get(name) != telemetry.get(name):
            raise RuntimeError(f"v017 receipt/telemetry mismatch for {name}")
    if not summary["relational"] and telemetry["warmup_phase"] == "low_dr_joint_init" and telemetry["k_stage_iteration"] == 0:
        actor_delta = telemetry["actor_std_parameter_delta"]
        critic_delta = telemetry["critic_parameter_delta"]
        if (
            not float(actor_delta.get("param_delta_max_abs", 0.0)) > 0.0
            or not float(critic_delta.get("param_delta_max_abs", 0.0)) > 0.0
        ):
            raise RuntimeError("FRS-TRAIN-v024 first coupled commit requires updated Actor/std and Critic")
    # AUDIT-B02/B05/B06/B07: 最终 serializer 只读审计, 不反馈训练状态.
    from rsl_rl.runners.frontres_formal_runtime_audit import print_phase_b_telemetry_audit

    print_phase_b_telemetry_audit(runner, telemetry=telemetry)
    return summary
