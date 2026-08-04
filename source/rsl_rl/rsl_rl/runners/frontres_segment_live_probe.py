"""Compatibility facade for the interface-oriented FrontRES Segment runtime.

Production owners live in responsibility-specific modules. New consumers
must import those public owners directly; this module only preserves the
historical import surface while legacy callers retire."""

from __future__ import annotations

# Historical imported-name surface retained for external contracts. New code
# imports the responsibility-specific owners below instead of this facade.
from rsl_rl.algorithms import FrontRESUnified
from rsl_rl.algorithms.frontres_constraint_projection import (
    install_frontres_v004_projected_gradients,
    step_frontres_v004_optimizer_with_actor_authority,
)
from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
)
from rsl_rl.frontres.frontres_local_evaluation import (
    FrontRESV015LocalEvaluationReport,
    build_frontres_v015_local_evaluation_report,
)
from rsl_rl.frontres.frontres_segment_evidence_legacy import (
    FrontRESV015GainReturnEvidence,
    FrontRESV015OneActionKEvidence,
    build_frontres_v015_gain_return_evidence,
    pair_frontres_v015_gain_facts,
)
from rsl_rl.frontres.frontres_segment_grouped_adapter import (
    build_frontres_v015_grouped_candidate_storage,
)
from rsl_rl.frontres.frontres_segment_rollout_storage import FrontRESSegmentRolloutStorage
from rsl_rl.frontres.frontres_segment_storage_records import (
    FrontRESV015RejectedTransactionEvidence,
    FrontRESSegmentTransition,
)

from rsl_rl.runners.frontres_segment_runtime_types import (
    FrontRESFrozenPolicyTransactionAccumulator,
    FrontRESFrozenPolicyTransactionResult,
    FrontRESSegmentLiveObservations,
    FrontRESSegmentLiveRolloutCapture,
    FrontRESFormalTransactionRequest,
    FrontRESFormalTransactionUpdateResult,
    FrontRESV015GainConsumerEvidence,
    _V015_CHECKPOINT_TRANSACTION_STATE_ATTR,
    _bind_frontres_checkpoint_transaction_plan,
    _commit_frontres_checkpoint_transaction,
    _seal_frontres_checkpoint_transaction_plan,
    _v015_checkpoint_plan_hash,
    open_frontres_checkpoint_transaction_barrier,
)
from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout
from rsl_rl.runners.frontres_segment_live_sampler import prepare_frontres_v015_local_sentinel_batch
from rsl_rl.runners.frontres_segment_transaction import FrontRESFormalTransactionAccumulator

from rsl_rl.runners.frontres_segment_probe_logging import (
    _AUDIT_IDENTITY_KEYS,
    _LOG_SEPARATOR,
    _VERBOSE_PROBE_BATCH_LIMIT,
    _audit_identity_kwargs,
    _audit_identity_tuple,
    _bool_list,
    _capture_audit_identity_kwargs,
    _count_summary,
    _delta_se_norm,
    _delta_z_up_frac,
    _family_mask_debug_lines,
    _finite_mean,
    _float_list,
    _fmt_metric,
    _fmt_num,
    _fmt_pct,
    _fmt_vec,
    _id_summary,
    _kv_lines,
    _live_detail_log_enabled,
    _log_block,
    _long_list,
    _mean_sequence,
    _motion_command_for_runner,
    _motion_summary,
    _new_live_audit_identity,
    _perturber_debug_lines,
    _positive_fraction,
    _print_frontres_dr_runtime_probe,
    _probe_status,
    _safe_getattr,
    _sequence_summary,
    _shape_last_dim,
    _should_print_once_or_verbose,
    _tensor_debug_summary,
    _tensor_nonzero_frac,
    _tensor_range_summary,
    _verbose_index_reset_lines,
    _verbose_probe_enabled,
    _verbose_reset_lines,
)

from rsl_rl.runners.frontres_segment_live_policy import (
    FrontRESSegmentLivePolicyAdapter,
    _apply_segment_adaptive_learning_rate,
    _attach_ppo_update_diagnostics,
    _clear_noncritic_grads,
    _evaluate_segment_delta_se_log_prob,
    _evaluate_segment_delta_se_log_prob_from_stats,
    _optimizer_parameter_snapshots,
    _parameter_delta_stats,
    _post_update_segment_ppo_diagnostics,
    _restore_optimizer_parameters,
    _segment_delta_se_log_prob_parts,
    _set_segment_optimizer_lr,
    run_frontres_segment_single_update,
)

from rsl_rl.runners.frontres_segment_live_reset import (
    _apply_current_segment_reset,
    _apply_index_only_segment_reset,
    _attach_fixed_noisy_tape_to_index_request,
    _attach_frontres_local_scenario_to_index_request,
    _attach_frozen_transaction_metadata_to_request,
    _attach_trial_metadata_to_request,
    _capture_batch_size,
    _current_frozen_transaction_metadata,
    _current_trial_metadata,
    _env_has_segment_reset_hook,
    _frontres_reset_role_env_ids,
    _frozen_transaction_vector_has_rows,
    _index_reset_result_from_mapping,
    _index_segment_reset_hook,
    _is_index_only_segment_batch,
    _mapping_bool,
    _mapping_float,
    _same_frozen_transaction_vector,
    _trial_horizon_vector,
    _trial_long_vector,
    _trial_metadata_ppo_update_mask,
    _trial_metadata_priority_evidence,
    _update_ppo_boundary_summary,
    _update_reset_summary,
    _update_trial_metadata_summary,
    apply_frontres_current_segment_reset,
)

from rsl_rl.runners.frontres_segment_live_storage import (
    _average_physics_steps,
    _capture_action_valid_steps,
    _capture_averaged_repair_scores,
    _capture_averaged_rewards,
    _capture_paired_gain,
    _current_reset_success_mask,
    _expand_short_counterfactual_tuple,
    _expand_short_counterfactual_vector,
    _gain_module,
    _motion_perturber_from_runner,
    _segment_storage_done_steps,
    _segment_storage_reward_steps,
    _segment_storage_rewards,
    _select_executed_segment_actions,
    _select_segment_transition_actions,
    _snapshot_frontres_perturbation_rp,
    build_live_segment_storage,
    capture_frontres_paired_gain,
)

from rsl_rl.runners.frontres_segment_one_action_k import (
    _append_fixed_noisy_actor_context,
    _capture_v015_post_t_executed_q29,
    _read_live_observations,
    _read_v015_frozen_gmt_observations,
    _require_v015_one_action_k_layout,
    _resolve_probe_modes,
    _uses_v015_future_intent_route_local,
    _v015_intent_provenance_rows,
    build_frontres_v015_grouped_candidate_batch,
    collect_frontres_v015_gain_return_priority_evidence,
    collect_frontres_v015_one_action_k_evidence,
    read_frontres_live_observations,
)

from rsl_rl.runners.frontres_segment_formal_transaction import (
    _build_frontres_v015_local_identity_sentinel_request,
    _build_frontres_v015_local_transaction_request,
    _require_v015_formal_transaction_config,
    _v015_formal_optimizer_step_count,
    _v015_formal_policy_evaluator,
    _v015_formal_ppo_config,
    _v015_resolve_curriculum_identity,
    abort_frontres_formal_training_collection,
    build_frontres_formal_training_request,
    close_frontres_formal_training_request,
    run_frontres_formal_transaction_update,
    run_frontres_local_identity_sentinel,
)

from rsl_rl.runners.frontres_segment_physics import (
    _capture_motion_quality_frame,
    _capture_physics_frame,
    _capture_root_orientation_frame,
    _capture_v015_quality_lateral_lean_frame,
    _contact_sensor_pair,
    _contact_wrench_zmp_pair,
    _ensure_frontres_raw_contact_view,
    _pad_raw_contact_slots,
    _prepare_frontres_raw_contact_views,
    _raw_filtered_contact_rows,
    _root_relative_body_pos,
    _stack_motion_quality_frames,
    prepare_frontres_raw_contact_views,
)

from rsl_rl.runners.frontres_segment_live_rollout import (
    _read_step_observations,
    _run_live_rollout_capture,
    _segment_repair_executability_scores,
    _zero_segment_env_actions,
    run_frontres_live_rollout_capture,
)

from rsl_rl.runners.frontres_segment_probe_reporting import (
    _initial_live_probe_summary,
    _motion_quality_summary,
    _paired_gain_summary,
    _paired_score_summary,
    _print_live_probe_summary,
    _rollout_done_per_sample,
    _rollout_reward_per_sample,
    _valid_reward_mean,
)

from rsl_rl.runners.frontres_segment_legacy_probe import (
    run_frontres_segment_live_probe,
)
