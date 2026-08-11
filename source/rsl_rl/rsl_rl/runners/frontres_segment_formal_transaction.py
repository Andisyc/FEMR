"""Formal v017 transaction request assembly and commit adapter."""





from __future__ import annotations





from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import json


import math


from typing import Any


import torch


from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
    install_frontres_v006_scalar_gradients,
    step_frontres_v005_scalar_optimizer,
)
from rsl_rl.frontres.frontres_value_normalization import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FRONTRES_VALUE_NORMALIZER_DECAY,
    FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
    FrontRESValueNormalizerState,
)


from rsl_rl.frontres.frontres_balance import prepare_frontres_raw_contact_views


from rsl_rl.frontres.frontres_gain import FrontRESRecoveryAwareGainConfig, compute_recovery_aware_gain
from rsl_rl.frontres.frontres_local_evaluation import build_frontres_v017_local_evaluation_report
from rsl_rl.frontres.frontres_outer_scenario_replay import FrontRESOuterScenarioReplay
from rsl_rl.frontres.frontres_segment_evidence import (
    FrontRESSegmentBaselineEvidence,
    FrontRESSealedRecoveryAwareGainBatch,
)
from rsl_rl.frontres.frontres_segment_grouped_adapter import build_frontres_v017_grouped_candidate_storage


from rsl_rl.frontres.frontres_segment_warmup import require_frontres_v013_campaign_schedule, resolve_frontres_k_stage_identity


from rsl_rl.frontres.training_schedule import resolve_frontres_mode_state


from rsl_rl.modules import FrontRESActorCritic


from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout


from rsl_rl.runners.frontres_segment_transaction import FrontRESFormalTransactionAccumulator
from rsl_rl.runners.frontres_segment_live_sampler import close_frontres_local_scenarios, prepare_frontres_v015_formal_training_batch, prepare_frontres_v015_local_sentinel_batch





from rsl_rl.runners.frontres_segment_runtime_types import (
    FrontRESFormalTransactionRequest,
    FrontRESFormalTransactionUpdateResult,
    FrontRESSegmentLiveObservations,
    bind_frontres_collection_context,
    bind_frontres_checkpoint_transaction_plan as _bind_frontres_checkpoint_transaction_plan,
    clear_frontres_collection_context,
    commit_frontres_checkpoint_transaction as _commit_frontres_checkpoint_transaction,
    frontres_collection_batch,
    frontres_observation_trace,
    frontres_preupdate_diagnostics,
    frontres_stage3_transaction_aggregate,
    publish_frontres_preupdate_diagnostics,
    reset_frontres_checkpoint_transaction as _reset_frontres_checkpoint_transaction,
    seal_frontres_checkpoint_transaction_plan as _seal_frontres_checkpoint_transaction_plan,
    start_frontres_checkpoint_transaction_commit as _start_frontres_checkpoint_transaction_commit,
)


from rsl_rl.runners.frontres_segment_probe_logging import (
    try_frontres_motion_command as _motion_command_for_runner,
)


from rsl_rl.runners.frontres_segment_live_policy import (
    FrontRESSegmentLivePolicyAdapter,
    snapshot_optimizer_parameters as _optimizer_parameter_snapshots,
    summarize_parameter_deltas as _parameter_delta_stats,
)
from rsl_rl.runners.frontres_formal_runtime_audit import print_segment_replay_transaction_audit


from rsl_rl.runners.frontres_segment_live_reset import (
    apply_frontres_current_segment_reset as _apply_current_segment_reset,
)


from rsl_rl.runners.frontres_segment_one_action_k import (
    read_frontres_live_observations as _read_live_observations,
    collect_frontres_v017_no_actor_baseline,
    collect_frontres_v017_repair_attempts,
    select_frontres_v017_trajectory_rows,
)





def _require_frontres_v016_observation_trace(
    observation_trace: Mapping[str, object],
    *,
    policy_row_count: int,
    label: str,
) -> None:
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
        "critic_current_observation_dim": 289,
        "critic_future_intent_dim": 58,
        "critic_support_context_dim": 102,
        "critic_observation_dim": 449,
    }
    mismatched_trace = {
        key: (observation_trace.get(key), expected)
        for key, expected in expected_trace.items()
        if observation_trace.get(key) != expected
    }
    shared_state_mismatch = {
        key: observation_trace.get(key)
        for key in ("actor_segment_state_max_abs_diff", "critic_segment_state_max_abs_diff")
        if observation_trace.get(key) != 0.0
    }
    raw_state_diff = {
        key: float(observation_trace.get(key, float("nan")))
        for key in ("actor_raw_observation_max_abs_diff", "critic_raw_observation_max_abs_diff")
    }
    if (
        mismatched_trace
        or shared_state_mismatch
        or any(not math.isfinite(value) or value < 0.0 for value in raw_state_diff.values())
        or int(observation_trace.get("post_advance_gmt_read_count", 0)) <= 0
    ):
        raise RuntimeError(
            f"v017 {label} observation trace is incomplete or violates the frozen authority: "
            f"mismatched={mismatched_trace}, shared_state={shared_state_mismatch}, "
            f"raw_state_diff={raw_state_diff}, trace={observation_trace}"
        )


def _v015_formal_optimizer_step_count(optimizer: Any) -> int:
    """Require an explicit step counter; unknown optimizer state is not evidence."""

    for name in ("frontres_step_count", "step_count"):
        value = getattr(optimizer, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return int(value)
    raise RuntimeError(
        "v017 formal transaction fake S2 requires an explicit non-negative optimizer "
        "frontres_step_count or step_count"
    )


def _require_v015_formal_transaction_config(runner: Any) -> Any:
    """Freeze the v015 isolation boundary before any batch, loss, or step."""

    alg = getattr(runner, "alg", None)
    if alg is None or not bool(getattr(alg, "frontres_formal_transaction_enabled", False)):
        raise RuntimeError("v017 formal transaction route requires frontres_formal_transaction_enabled=True")
    if str(getattr(alg, "frontres_segment_advantage_normalization", "")).lower() != "grouped_scale_only":
        raise RuntimeError("v017 formal transaction route requires grouped_scale_only normalization")
    if any(
        float(getattr(alg, name, 0.0) or 0.0) != 0.0
        for name in ("lambda_supervised", "lambda_supervised_min")
    ):
        raise RuntimeError("v017 formal transaction route rejects nonzero Stage-3 supervised loss")
    if any(
        bool(getattr(alg, name, False))
        for name in ("frontres_hsl_init_enabled", "frontres_hsl_rollout_label_enabled")
    ):
        raise RuntimeError("v017 formal transaction route rejects implicit HSL initialization or rollout labels")
    schedule = tuple(getattr(alg, "frontres_segment_k_curriculum", ()) or ())
    if not schedule:
        raise RuntimeError("FRS-TRAIN-v021 formal transaction requires an explicit K x M x DR curriculum")
    require_frontres_v013_campaign_schedule(schedule)
    if any(
        bool(getattr(alg, name, False))
        for name in (
            "frontres_segment_live_update_loop_only",
            "frontres_segment_live_single_update_only",
        )
    ):
        raise RuntimeError("v017 formal transaction rejects legacy immediate-update dispatch flags")
    if bool(getattr(alg, "frontres_segment_live_train_enabled", False)) and int(
        getattr(alg, "frontres_segment_live_update_steps", 0) or 0
    ) != 1:
        raise RuntimeError("v017 formal training requires one complete transaction and one update per iteration")
    offsets = tuple(int(value) for value in (getattr(alg, "frontres_future_offsets", ()) or ()))
    layout_version = str(getattr(alg, "frontres_future_intent_layout_version", ""))
    if offsets != (1, 2) or layout_version != "frontres-v015-future-intent-q29-v1":
        raise RuntimeError(
            "v017 formal transaction route requires exact deployment-q29 offsets (1, 2) "
            "and layout frontres-v015-future-intent-q29-v1"
        )
    required_identity = {
        "frontres_method_contract_id": "FRS-METHOD-v023",
        "frontres_gain_contract_id": "FRS-GAIN-v008",
        "frontres_optimization_contract_id": "FRS-PPO-v010",
        "frontres_training_contract_id": "FRS-TRAIN-v022",
        "frontres_scalar_target_id": "symmetric-log-recovery-aware-utility-v1",
        "frontres_return_utility_id": "symmetric-log-gain-g0-1-v1",
        "frontres_physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
        "frontres_grouped_schema_id": "grouped-all-attempt-scalar-v1",
        "frontres_critic_support_context_id": "action-pre-support-plan-kmax32-v1",
    }
    for name, expected in required_identity.items():
        if str(getattr(alg, name, "")) != expected:
            raise RuntimeError(f"v017 formal transaction requires {name}={expected}")
    if float(getattr(alg, "frontres_gain_beta", float("nan"))) != 0.02:
        raise RuntimeError("FRS-GAIN-v008 formal transaction requires frozen beta_init=0.02")
    if float(getattr(alg, "frontres_return_utility_scale", float("nan"))) != 1.0:
        raise RuntimeError("FRS-PPO-v009 formal transaction requires fixed utility G0=1")
    if float(getattr(alg, "frontres_segment_actor_joint_lr", float("nan"))) != 1.0e-6:
        raise RuntimeError("FRS-TRAIN-v022 formal transaction requires Actor joint LR=1e-6")
    if float(getattr(alg, "critic_learning_rate", float("nan"))) != 1.0e-5:
        raise RuntimeError("FRS-TRAIN-v022 formal transaction requires Critic LR=1e-5")
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
        raise RuntimeError("FRS-TRAIN-v021 runtime curriculum fingerprint drifted after config resolution")
    return identity


def _v015_formal_ppo_config(alg: Any, *, actor_loss_weight: float) -> FrontRESSegmentPPOConfig:
    """复用 v003 公式参数, 仅选择已确认的 grouped reduction mode."""

    normalizer_state = getattr(alg, "frontres_critic_value_normalizer_state", None)
    if not isinstance(normalizer_state, FrontRESValueNormalizerState):
        raise RuntimeError("FRS-TRAIN-v021 requires one immutable Critic value-normalizer state")
    normalization_id = str(getattr(alg, "frontres_critic_value_normalization", "")).lower()
    decay = float(getattr(alg, "frontres_critic_value_normalizer_decay", float("nan")))
    scale_floor = float(getattr(alg, "frontres_critic_value_normalizer_scale_floor", float("nan")))
    if (
        normalization_id != FRONTRES_VALUE_NORMALIZATION_ID
        or decay != FRONTRES_VALUE_NORMALIZER_DECAY
        or scale_floor != FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR
    ):
        raise RuntimeError("FRS-TRAIN-v021 formal transaction requires the fixed Critic value-normalizer identity")
    return FrontRESSegmentPPOConfig(
        clip_param=float(getattr(alg, "clip_param", 0.2)),
        value_clip_param=float(getattr(alg, "clip_param", 0.2)),
        value_loss_coef=float(getattr(alg, "value_loss_coef", 1.0)),
        entropy_coef=float(getattr(alg, "entropy_coef", 0.0)),
        use_clipped_value_loss=bool(getattr(alg, "use_clipped_value_loss", True)),
        normalize_advantages=False,
        advantage_normalization="grouped_scale_only",
        actor_loss_weight=float(actor_loss_weight),
        critic_target_id="segment-exact-m-mean-symlog-v1",
        critic_value_normalization=normalization_id,
        critic_value_normalizer_state=normalizer_state,
        critic_value_normalizer_decay=decay,
        critic_value_normalizer_scale_floor=scale_floor,
    )


def _v015_formal_policy_evaluator(
    request: FrontRESFormalTransactionRequest,
    alg: Any,
    ppo_batch: Any,
) -> Any:
    evaluator = request.policy_evaluator
    if evaluator is not None:
        if not callable(getattr(evaluator, "evaluate_segment_actions", None)):
            raise TypeError("v017 formal transaction policy_evaluator must expose evaluate_segment_actions")
        return evaluator
    privileged_observations = getattr(ppo_batch, "privileged_observations", None)
    request_privileged = request.privileged_observations
    if not isinstance(privileged_observations, torch.Tensor):
        raise RuntimeError("v017 formal transaction requires sealed t critic observations; actor-observation fallback is forbidden")
    if request_privileged is not None:
        if (
            tuple(request_privileged.shape) != tuple(privileged_observations.shape)
            or not torch.equal(
                request_privileged.to(device=privileged_observations.device),
                privileged_observations,
            )
        ):
            raise ValueError("v017 formal transaction request critic observations disagree with sealed candidate rows")
    if (
        privileged_observations.ndim != 2
        or int(privileged_observations.shape[0]) != int(ppo_batch.observations.shape[0])
        or int(privileged_observations.shape[1]) <= 0
    ):
        raise ValueError("v017 formal transaction critic observations must be non-empty [policy_row, critic_feature]")
    return FrontRESSegmentLivePolicyAdapter(alg, privileged_observations)


def run_frontres_formal_transaction_update(
    runner: Any,
    request: FrontRESFormalTransactionRequest,
) -> FrontRESFormalTransactionUpdateResult:
    """Execute one sealed v015 offline-S2 grouped PPO update after all M attempts.

    此函数是 Step 4B 的唯一 update owner, 也是 Step 4C 的 committed receipt
    publisher. 它不调用 legacy `to_ppo_batch`, `run_frontres_segment_single_update`,
    sampler state, checkpoint save/load, simulator 或 live loop.

    Status: active exact-one update owner for offline contracts, the bounded
    sentinel, and ordinary v015 Stage-3 dispatch. Simulator and policy-quality
    evidence remain separate live gates.
    """

    if not isinstance(request, FrontRESFormalTransactionRequest):
        raise TypeError("v017 formal transaction update requires FrontRESFormalTransactionRequest")
    request.__post_init__()
    _bind_frontres_checkpoint_transaction_plan(runner, request.plan)
    alg = _require_v015_formal_transaction_config(runner)
    policy = getattr(alg, "policy", None)
    optimizer = getattr(alg, "optimizer", None)
    outer_replay = getattr(runner, "_frontres_outer_scenario_replay", None)
    if policy is None or optimizer is None:
        raise RuntimeError("v017 formal transaction update requires runner.alg policy and optimizer")
    if (
        not isinstance(outer_replay, FrontRESOuterScenarioReplay)
        or request.outer_replay_plan is None
        or len(request.outer_replay_scenario_keys) != 8
    ):
        raise RuntimeError("FRS-TRAIN-v022 formal transaction requires the outer Scenario replay owner and eight keys")
    optimizer_groups = tuple(getattr(optimizer, "param_groups", ()))
    optimizer_lr_by_role = {
        str(group.get("frontres_role", "")): float(group.get("lr", float("nan")))
        for group in optimizer_groups
    }
    if (
        len(optimizer_groups) != 2
        or set(optimizer_lr_by_role) != {"actor", "critic"}
        or not all(math.isfinite(value) and value > 0.0 for value in optimizer_lr_by_role.values())
        or optimizer_lr_by_role["critic"] != float(getattr(alg, "critic_learning_rate", float("nan")))
    ):
        raise RuntimeError("FRS-TRAIN-v022 formal transaction requires the exact named split-LR optimizer identity")
    optimizer_step_before = _v015_formal_optimizer_step_count(optimizer)
    curriculum = _v015_resolve_curriculum_identity(runner, alg)
    iteration = curriculum.absolute_iteration
    value_normalizer_state = getattr(alg, "frontres_critic_value_normalizer_state", None)
    if (
        not isinstance(value_normalizer_state, FrontRESValueNormalizerState)
        or value_normalizer_state.update_count != iteration
    ):
        raise RuntimeError(
            "FRS-TRAIN-v021 requires Critic value-normalizer count to equal the committed iteration"
        )
    warmup_phase = curriculum.phase
    if (
        request.training_iteration != iteration
        or request.curriculum_fingerprint != curriculum.schedule_fingerprint
        or request.k_stage_index != curriculum.stage_index
        or request.active_k != curriculum.active_k
        or request.active_m != curriculum.active_m
        or request.k_stage_iteration != curriculum.stage_iteration
        or request.warmup_phase_name != warmup_phase.name
        or not math.isclose(float(request.warmup_actor_loss_weight), warmup_phase.actor_loss_weight, abs_tol=1e-12)
        or not math.isclose(float(request.warmup_actor_learning_rate), warmup_phase.actor_learning_rate, abs_tol=1e-15)
        or request.dr_stage_fingerprint != curriculum.dr_stage_fingerprint
        or not math.isclose(float(request.dr_progress), curriculum.dr_progress, abs_tol=1e-12)
        or not math.isclose(float(request.d_cap), curriculum.d_cap, abs_tol=1e-12)
    ):
        raise RuntimeError("v017 transaction crossed or changed its sealed FRS-TRAIN-v022 K x M x DR x LR identity")
    if not bool((request.plan.horizon_k.detach().to(dtype=torch.long) == curriculum.active_k).all().item()):
        raise RuntimeError("FRS-TRAIN-v021 formal update rejects mixed-K transaction rows")
    if request.plan.active_m != curriculum.active_m or request.plan.selected_segment_count != 8:
        raise RuntimeError("FRS-TRAIN-v022 formal update rejects mixed-M or non-B8 transactions")
    request.plan.verify_policy(policy)
    accumulator = FrontRESFormalTransactionAccumulator(
        request.plan,
        optimizer_step_count=lambda: _v015_formal_optimizer_step_count(optimizer),
    )
    for candidate_batch in request.candidate_batches:
        accumulator.append_candidate_batch(candidate_batch)
    ppo_batch = accumulator.seal()
    complete_rows = (
        ppo_batch.valid_mask.detach().bool()
        & torch.isfinite(ppo_batch.returns.detach())
        & torch.isfinite(ppo_batch.advantages.detach())
    )
    if int(complete_rows.numel()) != int(request.plan.batch_size) or not bool(complete_rows.all()):
        raise RuntimeError(
            "FRS-TRAIN-v022 requires every B8 x exact-M Repair row before optimizer update"
        )
    _seal_frontres_checkpoint_transaction_plan(runner, request.plan)
    _start_frontres_checkpoint_transaction_commit(runner)
    request.plan.verify_policy(policy)
    policy_evaluator = _v015_formal_policy_evaluator(request, alg, ppo_batch)
    ppo_cfg = _v015_formal_ppo_config(alg, actor_loss_weight=warmup_phase.actor_loss_weight)
    ppo_result = compute_frontres_segment_ppo_loss(
        policy_evaluator,
        ppo_batch,
        ppo_cfg,
    )
    if not ppo_result.should_step:
        raise RuntimeError("v017 formal transaction has no valid grouped PPO rows; refusing optimizer step")
    candidate_normalizer_state = ppo_result.critic_value_normalizer_candidate_state
    if (
        ppo_result.critic_value_normalization_id != FRONTRES_VALUE_NORMALIZATION_ID
        or ppo_result.critic_value_normalizer_previous_state != alg.frontres_critic_value_normalizer_state
        or not isinstance(candidate_normalizer_state, FrontRESValueNormalizerState)
        or candidate_normalizer_state.update_count != alg.frontres_critic_value_normalizer_state.update_count + 1
    ):
        raise RuntimeError("FRS-PPO-v009 transaction produced an invalid Critic value-normalizer transition")
    outer_replay_candidate = outer_replay.stage(
        request.outer_replay_plan,
        keys=request.outer_replay_scenario_keys,
        actor_advantages=torch.tensor(ppo_result.actor_advantages, dtype=torch.float32),
        source_index=ppo_batch.transaction_metadata.source_index,
        policy_snapshot_id=request.plan.policy_snapshot_id,
        active_m=request.active_m,
    )
    zero_grad = getattr(optimizer, "zero_grad", None)
    step = getattr(optimizer, "step", None)
    if not callable(zero_grad) or not callable(step):
        raise RuntimeError("v017 formal transaction optimizer must expose zero_grad() and step()")
    try:
        zero_grad(set_to_none=True)
    except TypeError:
        zero_grad()
    optimizer_params, parameter_snapshots = _optimizer_parameter_snapshots(policy, optimizer)
    critic = getattr(policy, "critic", None)
    critic_ids = {id(parameter) for parameter in critic.parameters()} if critic is not None else set()
    critic_params = tuple((name, parameter) for name, parameter in optimizer_params if id(parameter) in critic_ids)
    noncritic_params = tuple((name, parameter) for name, parameter in optimizer_params if id(parameter) not in critic_ids)
    parameters = [
        parameter
        for group in getattr(optimizer, "param_groups", ())
        for parameter in group.get("params", ())
        if isinstance(parameter, torch.Tensor) and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("v017 formal transaction optimizer has no trainable parameters")
    max_grad_norm = float(getattr(alg, "max_grad_norm", float("nan")))
    if max_grad_norm != 0.5:
        raise RuntimeError("FRS-PPO-v009 formal transaction requires max_grad_norm=0.5")
    gradient_install = install_frontres_v006_scalar_gradients(
        policy,
        ppo_result,
        ppo_cfg,
        tuple(parameters),
        max_grad_norm=max_grad_norm,
    )
    actor_parameters = gradient_install.actor_parameters
    gradient_pre_clip_norm = math.sqrt(
        gradient_install.actor_pre_clip_norm**2 + gradient_install.critic_pre_clip_norm**2
    )
    gradient_post_clip_norm = math.sqrt(
        gradient_install.actor_post_clip_norm**2 + gradient_install.critic_post_clip_norm**2
    )
    gradient_nonzero_parameter_count = (
        gradient_install.actor_nonzero_parameter_count
        + gradient_install.critic_nonzero_parameter_count
    )
    if not math.isfinite(gradient_pre_clip_norm) or not math.isfinite(gradient_post_clip_norm):
        raise FloatingPointError(
            "v017 formal transaction produced non-finite gradients: "
            f"pre_clip={gradient_pre_clip_norm} post_clip={gradient_post_clip_norm}"
        )
    diagnostic_valid = (
        ppo_batch.valid_mask.detach().bool()
        & torch.isfinite(ppo_batch.returns.detach())
        & torch.isfinite(ppo_batch.advantages.detach())
    )
    if int(diagnostic_valid.sum().item()) != int(ppo_result.valid_count):
        raise RuntimeError(
            "v017 formal transaction return/advantage telemetry disagrees with PPO valid rows: "
            f"telemetry={int(diagnostic_valid.sum().item())} ppo={int(ppo_result.valid_count)}"
        )
    valid_returns = ppo_batch.returns.detach().float()[diagnostic_valid]
    actor_group = next(group for group in optimizer_groups if group.get("frontres_role") == "actor")
    previous_actor_lr = float(actor_group["lr"])
    actor_group["lr"] = float(warmup_phase.actor_learning_rate)
    alg.learning_rate = float(warmup_phase.actor_learning_rate)
    alg.actor_learning_rate = float(warmup_phase.actor_learning_rate)
    try:
        actual_commit = step_frontres_v005_scalar_optimizer(
            optimizer,
            actor_parameters,
            parameter_snapshots,
            actor_loss_weight=1.0,
        )
    except Exception:
        actor_group["lr"] = previous_actor_lr
        alg.learning_rate = previous_actor_lr
        alg.actor_learning_rate = previous_actor_lr
        raise
    actor_state_restored = actual_commit.actor_optimizer_state_preserved
    parameter_delta = _parameter_delta_stats(optimizer_params, parameter_snapshots)
    critic_delta = _parameter_delta_stats(critic_params, parameter_snapshots)
    noncritic_delta = _parameter_delta_stats(noncritic_params, parameter_snapshots)
    if (
        warmup_phase.name == "low_dr_joint_init"
        and request.k_stage_iteration == 0
        and noncritic_delta["param_delta_max_abs"] <= 0.0
    ):
        raise RuntimeError("FRS-TRAIN-v021 first coupled update did not change actor or distribution parameters")
    optimizer_step_after = _v015_formal_optimizer_step_count(optimizer)
    optimizer_step_delta = optimizer_step_after - optimizer_step_before
    if optimizer_step_delta != 1:
        raise RuntimeError(
            "v017 formal transaction requires exactly one optimizer step: "
            f"before={optimizer_step_before} after={optimizer_step_after} delta={optimizer_step_delta}"
        )
    _commit_frontres_checkpoint_transaction(
        runner,
        plan=request.plan,
        valid_policy_row_count=int(ppo_result.valid_count),
        optimizer_step_before=optimizer_step_before,
        optimizer_step_after=optimizer_step_after,
        curriculum=curriculum,
    )
    committed_state = frontres_stage3_transaction_aggregate(runner).as_dict()
    receipt = committed_state.get("receipt")
    if committed_state.get("state") != "committed" or not isinstance(receipt, Mapping):
        raise RuntimeError("FRS-TRAIN-v021 optimizer commit did not publish a transaction receipt")
    outer_replay_telemetry = outer_replay.commit(outer_replay_candidate, receipt=receipt)
    alg.frontres_critic_value_normalizer_state = candidate_normalizer_state
    metadata = ppo_batch.transaction_metadata
    outer_sources = metadata.source_index.detach().to(device="cpu", dtype=torch.long)
    outer_utilities = torch.tensor(ppo_result.utility_returns, dtype=torch.float32)
    outer_old_values = ppo_batch.old_values.detach().to(device="cpu", dtype=torch.float32)
    outer_utility_means = tuple(
        float(outer_utilities[outer_sources == source].mean().item()) for source in range(8)
    )
    outer_old_value_means = tuple(
        float(outer_old_values[outer_sources == source].mean().item()) for source in range(8)
    )
    source_count = int(torch.unique(metadata.source_index.detach().to(dtype=torch.long)).numel())
    segment_count = int(torch.unique(metadata.segment_ids.detach().to(dtype=torch.long)).numel())
    flat_report_row_by_attempt: dict[tuple[int, int], int] = {}
    flat_report_row = 0
    for candidate_batch, report in zip(request.candidate_batches, request.diagnostic_reports, strict=True):
        candidate_metadata = candidate_batch.transaction_metadata
        if len(report.policy_actions) != int(candidate_metadata.batch_size):
            raise RuntimeError("v017 formal diagnostics disagree with their candidate batch row count")
        for row in range(int(candidate_metadata.batch_size)):
            key = (
                int(candidate_metadata.source_index[row].item()),
                int(candidate_metadata.trial_index[row].item()),
            )
            if key in flat_report_row_by_attempt:
                raise RuntimeError(f"v017 formal diagnostics repeat attempt identity {key}")
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
        "actor_learning_rate": float(actor_group["lr"]),
        "critic_learning_rate": optimizer_lr_by_role["critic"],
        "method_contract_id": "FRS-METHOD-v023",
        "training_contract_id": "FRS-TRAIN-v022",
        "gain_contract_id": "FRS-GAIN-v008",
        "optimization_contract_id": "FRS-PPO-v010",
        "scalar_target_id": "symmetric-log-recovery-aware-utility-v1",
        "return_utility_id": ppo_result.return_utility_id,
        "return_utility_scale": float(ppo_result.return_utility_scale),
        "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
        "grouped_schema_id": "grouped-all-attempt-scalar-v1",
        "actor_observation_dim": 158,
        "critic_observation_dim": int(ppo_batch.privileged_observations.shape[-1]),
        "gmt_observation_dim": 770,
        "critic_value_kind": "state_value",
        "critic_action_conditioned": False,
        "critic_target_id": "segment-exact-m-mean-symlog-v1",
        "critic_support_context_id": "action-pre-support-plan-kmax32-v1",
        "critic_value_targets": tuple(ppo_result.critic_value_targets),
        "critic_segment_target_means": tuple(ppo_result.critic_segment_target_means),
        "raw_returns": tuple(ppo_result.raw_returns),
        "utility_returns": tuple(ppo_result.utility_returns),
        "critic_raw_value_loss": float(ppo_result.critic_raw_value_loss.detach().cpu().item()),
        "critic_value_normalization_id": ppo_result.critic_value_normalization_id,
        "critic_value_scale": ppo_result.critic_value_scale,
        "critic_value_normalizer_decay": float(alg.frontres_critic_value_normalizer_decay),
        "critic_value_normalizer_scale_floor": float(alg.frontres_critic_value_normalizer_scale_floor),
        "critic_value_normalizer_mean_before": ppo_result.critic_value_normalizer_previous_state.mean,
        "critic_value_normalizer_mean_after": candidate_normalizer_state.mean,
        "critic_value_normalizer_second_moment_before": (
            ppo_result.critic_value_normalizer_previous_state.second_moment
        ),
        "critic_value_normalizer_second_moment_after": candidate_normalizer_state.second_moment,
        "critic_value_normalizer_update_count_before": (
            ppo_result.critic_value_normalizer_previous_state.update_count
        ),
        "critic_value_normalizer_update_count_after": candidate_normalizer_state.update_count,
        "actor_advantages": tuple(ppo_result.actor_advantages),
        "gradient_clip_identity": "separate-actor-critic-v1",
        "gradient_clip_max_norm": max_grad_norm,
        "actor_gradient_pre_clip_norm": gradient_install.actor_pre_clip_norm,
        "actor_gradient_post_clip_norm": gradient_install.actor_post_clip_norm,
        "actor_gradient_clip_coefficient": gradient_install.actor_clip_coefficient,
        "critic_gradient_pre_clip_norm": gradient_install.critic_pre_clip_norm,
        "critic_gradient_post_clip_norm": gradient_install.critic_post_clip_norm,
        "critic_gradient_clip_coefficient": gradient_install.critic_clip_coefficient,
        "actor_gradient_nonzero_parameter_count": gradient_install.actor_nonzero_parameter_count,
        "critic_gradient_nonzero_parameter_count": gradient_install.critic_nonzero_parameter_count,
        "optimizer_candidate_actor_delta_l2": actual_commit.optimizer_candidate_actor_delta_l2,
        "committed_actor_delta_l2": actual_commit.committed_actor_delta_l2,
        "actor_optimizer_state_restored": bool(actor_state_restored),
        "gain_beta": float(getattr(alg, "frontres_gain_beta", 0.02)),
        "training_iteration": iteration,
        "curriculum_fingerprint": curriculum.schedule_fingerprint,
        "k_stage_index": curriculum.stage_index,
        "active_k": curriculum.active_k,
        "active_m": curriculum.active_m,
        "selected_segment_count": request.plan.selected_segment_count,
        "policy_row_count": request.plan.batch_size,
        "role_row_count": 2 * request.plan.batch_size,
        "k_stage_iteration": curriculum.stage_iteration,
        "warmup_phase": warmup_phase.name,
        "warmup_phase_iteration": warmup_phase.phase_iteration,
        "actor_loss_weight": warmup_phase.actor_loss_weight,
        "actor_learning_rate_schedule": warmup_phase.actor_learning_rate,
        "dr_stage_fingerprint": curriculum.dr_stage_fingerprint,
        "dr_progress": curriculum.dr_progress,
        "d_cap": curriculum.d_cap,
        "dr_class_by_segment": tuple(request.dr_class_by_segment),
        "dr_strength_by_segment": tuple(request.dr_strength_by_segment),
        "parameter_delta": parameter_delta,
        "critic_parameter_delta": critic_delta,
        "actor_std_parameter_delta": noncritic_delta,
        "outer_replay": outer_replay_telemetry,
        "outer_replay_utility_means": outer_utility_means,
        "outer_replay_old_value_means": outer_old_value_means,
        "outer_replay_sources": tuple(selection.source for selection in request.outer_replay_plan.selections),
        "outer_replay_scenario_key_digests": tuple(key.digest for key in request.outer_replay_scenario_keys),
        "outer_replay_perturbation_seeds": tuple(
            selection.perturbation_seed for selection in request.outer_replay_plan.selections
        ),
        "v007_recovery_aware_reports": request.diagnostic_reports,
        "v007_diagnostic_report_row_order": diagnostic_report_row_order,
    }
    print(
        "[FrontRES v017 Formal Transaction] "
        f"transaction={request.plan.transaction_id} sources={source_count} "
        f"attempts={accumulator.collected_attempt_count} valid={ppo_result.valid_count} "
        f"step_delta={optimizer_step_delta}",
        flush=True,
    )
    result = FrontRESFormalTransactionUpdateResult(
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
    print_segment_replay_transaction_audit(runner, result=result)
    return result


def _reset_frontres_v017_phase(
    runner: Any,
    *,
    pair_layout: Any,
    mode: str,
    policy_row_count: int,
    label: str,
) -> None:
    """Install the sealed scenario, select one phase, then validate the reset."""

    # B1: command 尚无 carrier 时不得预先选 mode；reset hook 拥有 install -> mode -> refresh 顺序。
    reset_result = _apply_current_segment_reset(
        runner,
        pair_layout=pair_layout,
        local_scenario_execution_mode=mode,
    )
    success_mask = getattr(reset_result, "success_mask", None)
    if (
        reset_result is None
        or not isinstance(success_mask, torch.Tensor)
        or int(success_mask.numel()) != int(policy_row_count)
        or not bool(success_mask.detach().bool().all())
    ):
        raise RuntimeError(
            f"v017 {label} requires every selected local scenario reset to succeed in phase={mode}"
        )


@dataclass(frozen=True)
class FrontRESRecoveryAwareCollection:
    """One read-only Clean/Noisy/Repair collection shared by training and EVAL-v004."""

    evidence: FrontRESSealedRecoveryAwareGainBatch
    gain: Any
    report: Any
    pair_layout: Any
    observation_trace: dict[str, Any]
    policy_observations: FrontRESSegmentLiveObservations | None


def _clone_frontres_policy_observations(
    observations: FrontRESSegmentLiveObservations,
    *,
    expected_rows: int,
    device: torch.device,
) -> FrontRESSegmentLiveObservations:
    """Detach one complete policy input without retaining simulator-owned storage."""

    if not isinstance(observations, FrontRESSegmentLiveObservations):
        raise TypeError("policy-quality repeat requires typed live observations")

    def clone_field(name: str, *, optional: bool = False) -> torch.Tensor | None:
        value = getattr(observations, name)
        if optional and value is None:
            return None
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or int(value.shape[0]) != int(expected_rows)
            or value.requires_grad
            or not bool(torch.isfinite(value).all().item())
        ):
            raise RuntimeError(
                f"policy-quality repeat requires detached finite {name} [{expected_rows},D]"
            )
        return value.detach().to(device=device).clone()

    return FrontRESSegmentLiveObservations(
        obs=clone_field("obs"),
        privileged_obs=clone_field("privileged_obs"),
        teacher_obs=clone_field("teacher_obs"),
        ref_vel_estimator_obs=clone_field("ref_vel_estimator_obs", optional=True),
    )


def _resolve_frontres_repeat_policy_observations(
    live: FrontRESSegmentLiveObservations,
    *,
    frozen: FrontRESSegmentLiveObservations | None,
    route: str,
    expected_rows: int,
    device: torch.device,
) -> tuple[FrontRESSegmentLiveObservations, float, float]:
    """Use first-repeat policy inputs while retaining live-history drift as diagnostics."""

    live_copy = _clone_frontres_policy_observations(
        live,
        expected_rows=expected_rows,
        device=device,
    )
    if frozen is None:
        return live_copy, 0.0, 0.0
    if route != "policy_quality":
        raise RuntimeError("frozen repeat policy inputs are restricted to the policy-quality route")
    frozen_copy = _clone_frontres_policy_observations(
        frozen,
        expected_rows=expected_rows,
        device=device,
    )
    for name in ("obs", "privileged_obs", "teacher_obs", "ref_vel_estimator_obs"):
        live_value = getattr(live_copy, name)
        frozen_value = getattr(frozen_copy, name)
        if (live_value is None) != (frozen_value is None) or (
            isinstance(live_value, torch.Tensor)
            and isinstance(frozen_value, torch.Tensor)
            and tuple(live_value.shape) != tuple(frozen_value.shape)
        ):
            raise RuntimeError(f"policy-quality repeat changed the {name} input shape")
    actor_drift = float((live_copy.obs - frozen_copy.obs).abs().max().detach().cpu().item())
    critic_drift = float(
        (live_copy.privileged_obs - frozen_copy.privileged_obs).abs().max().detach().cpu().item()
    )
    return frozen_copy, actor_drift, critic_drift


@contextmanager
def frontres_readonly_collection_scope(runner: Any):
    """Own one EVAL-v004 collection lifecycle without mutating training state."""

    aggregate = frontres_stage3_transaction_aggregate(runner)
    aggregate.begin_readonly_collection()
    try:
        yield
    finally:
        # B1: 先释放 command/scenario carrier, 再关闭 read-only aggregate lifecycle.
        try:
            close_frontres_formal_training_request(runner)
        finally:
            aggregate.finish_readonly_collection()


def collect_frontres_recovery_aware_evaluation(
    runner: Any,
    prepared: Any,
    *,
    route: str,
    label: str,
    beta: float,
    policy_observations: FrontRESSegmentLiveObservations | None = None,
) -> FrontRESRecoveryAwareCollection:
    """Execute Clean once, Noisy once and exact-M Repairs without an optimizer update."""

    if policy_observations is not None and route != "policy_quality":
        raise RuntimeError("frozen repeat policy inputs are restricted to the policy-quality route")
    # B1: 安装 prepared transaction 并验证 two-role/exact-M identity, 产出 reset-ready layout.
    batch = getattr(prepared, "batch", None)
    sample = getattr(prepared, "sample", None)
    plan = getattr(prepared, "plan", None)
    if batch is None or sample is None or plan is None or not callable(getattr(plan, "validate", None)):
        raise TypeError("recovery-aware collector requires prepared sample, batch, and plan")
    plan.validate()
    prepare_frontres_raw_contact_views(runner)
    bind_frontres_collection_context(runner, route=route, sample=sample, batch=batch)
    frontres_mode = resolve_frontres_mode_state(runner, FrontRESActorCritic)
    pair_layout = configure_frontres_pair_layout(runner, is_frontres=frontres_mode.is_frontres)
    policy_row_count = int(plan.batch_size)
    expected_scenarios = 2 if route == "policy_quality" else 8
    if (
        int(getattr(pair_layout, "n_train", 0)) != policy_row_count
        or int(getattr(pair_layout, "n_base", 0)) != policy_row_count
        or int(getattr(pair_layout, "n_candidate", 0)) != 0
        or int(getattr(pair_layout, "n_clean", 0)) != 0
        or plan.selected_segment_count != expected_scenarios
        or plan.active_m < 2
    ):
        raise RuntimeError(f"{label} requires B{expected_scenarios} x exact-M Repair/Noisy rows")

    def reset_phase(mode: str) -> None:
        _reset_frontres_v017_phase(
            runner,
            pair_layout=pair_layout,
            mode=mode,
            policy_row_count=policy_row_count,
            label=label,
        )

    representatives: list[int] = []
    for source in sorted(set(int(value) for value in plan.source_index.tolist())):
        rows = torch.nonzero(plan.source_index == source, as_tuple=False).reshape(-1)
        if int(rows.numel()) != int(plan.active_m):
            raise RuntimeError("v017 baseline phase requires exact M policy rows per Segment")
        representatives.append(int(rows[0].item()))
    representative_rows = torch.tensor(representatives, device=runner.device, dtype=torch.long)
    active_k_values = torch.unique(plan.horizon_k.detach().to(dtype=torch.long))
    if int(active_k_values.numel()) != 1:
        raise RuntimeError("recovery-aware collector rejects mixed-K rows")
    active_k = int(active_k_values[0].item())

    # B2: 每个 Segment 执行一条 authoritative Clean/Noisy, 产出 shared baseline evidence.
    reset_phase("clean_baseline")
    clean_all, clean_support_all = collect_frontres_v017_no_actor_baseline(
        runner, horizon_k=active_k, authoritative_rows=representative_rows
    )
    reset_phase("noisy_baseline")
    noisy_all, noisy_support_all = collect_frontres_v017_no_actor_baseline(
        runner, horizon_k=active_k, authoritative_rows=representative_rows
    )
    if not torch.equal(clean_support_all, noisy_support_all):
        raise RuntimeError("v017 Clean/Noisy baselines lost expected-support identity")
    baselines: list[FrontRESSegmentBaselineEvidence] = []
    for baseline_row, representative in enumerate(representatives):
        baseline = FrontRESSegmentBaselineEvidence(
            transaction_id=plan.transaction_id,
            policy_snapshot_id=plan.policy_snapshot_id,
            scenario_id=plan.scenario_ids[representative],
            noisy_segment_hash=plan.noisy_segment_hashes[representative],
            x_t_identity=plan.x_t_identities[representative],
            source_index=int(plan.source_index[representative].item()),
            segment_id=int(plan.segment_ids[representative].item()),
            horizon_k=int(plan.horizon_k[representative].item()),
            expected_support=clean_support_all[:, baseline_row : baseline_row + 1].detach().clone(),
            clean=select_frontres_v017_trajectory_rows(
                clean_all, torch.tensor([baseline_row], device=runner.device)
            ),
            noisy=select_frontres_v017_trajectory_rows(
                noisy_all, torch.tensor([baseline_row], device=runner.device)
            ),
        )
        baseline.validate()
        baselines.append(baseline)

    # B3: 从同一 sealed scenario 收集 exact-M one-action-K Repairs, 产出唯一 GAIN-v007 report.
    reset_phase("repair_attempts")
    live_policy_observations = _read_live_observations(runner)
    if route == "policy_quality":
        used_policy_observations, live_actor_drift, live_critic_drift = (
            _resolve_frontres_repeat_policy_observations(
                live_policy_observations,
                frozen=policy_observations,
                route=route,
                expected_rows=2 * policy_row_count,
                device=runner.device,
            )
        )
    else:
        used_policy_observations = live_policy_observations
        live_actor_drift = 0.0
        live_critic_drift = 0.0
    attempts = collect_frontres_v017_repair_attempts(
        runner,
        used_policy_observations,
        pair_layout=pair_layout,
        transaction_id=plan.transaction_id,
        policy_snapshot_id=plan.policy_snapshot_id,
        source_index=plan.source_index,
        segment_ids=plan.segment_ids,
        trial_index=plan.trial_index,
    )
    evidence = FrontRESSealedRecoveryAwareGainBatch(
        baselines=tuple(baselines), attempts=attempts, active_m=int(plan.active_m)
    )
    evidence.validate()
    gain = compute_recovery_aware_gain(
        evidence.to_gain_input(),
        config=FrontRESRecoveryAwareGainConfig(beta=float(beta)),
    )
    report = build_frontres_v017_local_evaluation_report(evidence, gain)
    observation_trace = dict(frontres_observation_trace(runner))
    if route == "policy_quality":
        observation_trace.update({
            "repeat_policy_input_source": (
                "first-repeat-frozen" if policy_observations is not None else "live-first-repeat"
            ),
            "repeat_live_actor_input_max_abs_diff": live_actor_drift,
            "repeat_live_critic_input_max_abs_diff": live_critic_drift,
        })
    return FrontRESRecoveryAwareCollection(
        evidence=evidence,
        gain=gain,
        report=report,
        pair_layout=pair_layout,
        observation_trace=observation_trace,
        policy_observations=(
            _clone_frontres_policy_observations(
                used_policy_observations,
                expected_rows=2 * policy_row_count,
                device=runner.device,
            )
            if route == "policy_quality"
            else None
        ),
    )


def _build_frontres_v015_local_transaction_request(
    runner: Any,
    *,
    init_at_random_ep_len: bool,
    route: str,
) -> FrontRESFormalTransactionRequest:
    """Build a real local-scenario request for one explicit v015 route.

    Keeping the request builder separate makes the collecting-barrier order
    testable: no reset or policy attempt may occur before the formal update loop
    has opened it.
    """

    # B1: 解析并封存唯一 transaction 的 K/M, scenario, old-policy 和 reset lifecycle.
    del init_at_random_ep_len  # x_t reset owns the local dynamic start.
    alg = _require_v015_formal_transaction_config(runner)
    prepare_frontres_raw_contact_views(runner)
    sealed_iteration = int(getattr(runner, "current_learning_iteration", 0))
    sealed_curriculum = _v015_resolve_curriculum_identity(runner, alg)
    if route == "sentinel":
        prepared = prepare_frontres_v015_local_sentinel_batch(runner)
        label = "local sentinel"
    elif route == "training":
        prepared = prepare_frontres_v015_formal_training_batch(runner)
        label = "formal training"
    else:
        raise ValueError(f"unknown v015 request route={route!r}")
    batch = prepared.batch
    plan = prepared.plan
    try:
        policy_row_count = int(plan.batch_size)
        gain_beta = getattr(alg, "frontres_gain_beta", None)
        if gain_beta is None:
            raise RuntimeError("FRS-GAIN-v008 formal transaction requires an explicit repair-cost beta")
        collection = collect_frontres_recovery_aware_evaluation(
            runner,
            prepared,
            route=route,
            label=label,
            beta=float(gain_beta),
        )
        sealed_evidence = collection.evidence
        gain_result = collection.gain
        baselines = collection.evidence.baselines
        # B3: 唯一 Gain owner 产出 scalar rows, 再交给 grouped adapter 和 read-only local report.
        candidate_storage = build_frontres_v017_grouped_candidate_storage(
            sealed_evidence,
            gain_result,
            motion_ids=plan.motion_ids,
            start_frames=plan.start_frames,
            intent_q29_provenance=plan.intent_q29_provenance,
            intent_q29_source=plan.intent_q29_source,
        )
        candidate_batch = candidate_storage.to_grouped_ppo_candidate_batch(FrontRESSegmentPPOBatch)
        diagnostic_report = collection.report
        artifact = getattr(batch, "frontres_local_scenario_current_root_artifact_t", None)
        continuation_lengths = getattr(batch, "frontres_local_scenario_clean_continuation_lengths", None)
        if not isinstance(artifact, torch.Tensor) or not isinstance(continuation_lengths, torch.Tensor):
            raise RuntimeError(f"v015 {label} lost sealed root-artifact or Clean-continuation identity before storage")
        observation_trace = collection.observation_trace
        _require_frontres_v016_observation_trace(
            observation_trace,
            policy_row_count=policy_row_count,
            label=label,
        )
        publish_frontres_preupdate_diagnostics(runner, {
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
            "actor_forward_count": 1,
            "later_femr_action_count": 0,
            "clean_execution_count": tuple(value.clean_execution_count for value in baselines),
            "noisy_execution_count": tuple(value.noisy_execution_count for value in baselines),
            "horizon_k": tuple(int(value) for value in plan.horizon_k.tolist()),
            "observation_route": observation_trace,
        })
        if int(getattr(runner, "current_learning_iteration", 0)) != sealed_iteration:
            raise RuntimeError("v017 formal transaction changed persisted iteration while collecting attempts")
        dr_plan = getattr(batch, "stage3_index_perturbation_plan", None)
        dr_classes = tuple(getattr(dr_plan, "source_dr_class", ()) or ())
        dr_strength_tensor = getattr(dr_plan, "source_perturbation_strength", None)
        if len(dr_classes) != 8 or not isinstance(dr_strength_tensor, torch.Tensor) or int(dr_strength_tensor.numel()) != 8:
            raise RuntimeError("FRS-TRAIN-v022 formal request requires eight sealed Scenario DR class/strength rows")
        return FrontRESFormalTransactionRequest(
            plan=plan,
            candidate_batches=(candidate_batch,),
            diagnostic_reports=(diagnostic_report,),
            curriculum_fingerprint=sealed_curriculum.schedule_fingerprint,
            k_stage_index=sealed_curriculum.stage_index,
            active_k=sealed_curriculum.active_k,
            active_m=sealed_curriculum.active_m,
            k_stage_iteration=sealed_curriculum.stage_iteration,
            training_iteration=sealed_iteration,
            warmup_phase_name=sealed_curriculum.phase.name,
            warmup_actor_loss_weight=sealed_curriculum.phase.actor_loss_weight,
            warmup_actor_learning_rate=sealed_curriculum.phase.actor_learning_rate,
            dr_stage_fingerprint=sealed_curriculum.dr_stage_fingerprint,
            dr_progress=sealed_curriculum.dr_progress,
            d_cap=sealed_curriculum.d_cap,
            dr_class_by_segment=dr_classes,
            dr_strength_by_segment=tuple(float(value) for value in dr_strength_tensor.detach().cpu().tolist()),
            outer_replay_plan=prepared.outer_replay_plan,
            outer_replay_scenario_keys=tuple(prepared.outer_replay_scenario_keys),
        )
    except Exception:
        abort_frontres_formal_training_collection(runner, batch=batch)
        raise


def _build_frontres_v015_local_identity_sentinel_request(
    runner: Any,
    *,
    init_at_random_ep_len: bool,
) -> FrontRESFormalTransactionRequest:
    """Build the dedicated bounded sentinel request."""

    return _build_frontres_v015_local_transaction_request(
        runner,
        init_at_random_ep_len=init_at_random_ep_len,
        route="sentinel",
    )


def build_frontres_formal_training_request(
    runner: Any,
    *,
    init_at_random_ep_len: bool,
) -> FrontRESFormalTransactionRequest:
    """Build one complete ordinary Stage-3 request without legacy storage."""

    return _build_frontres_v015_local_transaction_request(
        runner,
        init_at_random_ep_len=init_at_random_ep_len,
        route="training",
    )


def close_frontres_formal_training_request(runner: Any) -> None:
    """Release command and sampler carriers owned by one completed request."""

    batch = frontres_collection_batch(runner)
    try:
        if batch is not None:
            command = _motion_command_for_runner(runner)
            clear = getattr(command, "clear_frontres_local_scenario", None)
            try:
                if not callable(clear):
                    raise RuntimeError(
                        "v017 formal training close requires command-owned local-scenario lifecycle"
                    )
                # Command rows own the live active bit; release them before the
                # immutable materializer identities are closed.
                clear()
            finally:
                close_frontres_local_scenarios(batch)
    finally:
        clear_frontres_collection_context(runner)


def abort_frontres_formal_training_collection(runner: Any, *, batch: Any | None = None) -> None:
    """Idempotently discard one rejected provider collection without an update.

    The command carrier is released before the materializer lifecycle.  The
    checkpoint barrier returns to the only persistable no-transaction state,
    so a later provider may open a fresh transaction at the same absolute
    training iteration.
    """

    if batch is None:
        batch = frontres_collection_batch(runner)
    command = _motion_command_for_runner(runner)
    clear = getattr(command, "clear_frontres_local_scenario", None)
    if not callable(clear):
        raise RuntimeError("v015 rejected transaction requires command-owned scenario cleanup")
    clear()
    if batch is not None and not tuple(getattr(batch, "frontres_local_scenario_closed_ids", ()) or ()):
        close_frontres_local_scenarios(batch)
    _reset_frontres_checkpoint_transaction(runner)


def run_frontres_local_identity_sentinel(
    runner: Any,
    *,
    init_at_random_ep_len: bool = True,
) -> FrontRESFormalTransactionUpdateResult:
    """Run one explicit v015 sentinel request through the existing exact-one update owner.

    Status: R6-S0 live-ready connector with fail-closed structured telemetry.
    The provider is invoked only after the formal collecting barrier opens;
    legacy probe/storage/update loops are not called. S4 live evidence remains open.
    """

    alg = _require_v015_formal_transaction_config(runner)
    if not bool(getattr(alg, "frontres_local_sentinel_only", False)):
        raise RuntimeError("v015 local identity sentinel requires its explicit config flag")
    def provider() -> FrontRESFormalTransactionRequest:
        return _build_frontres_v015_local_identity_sentinel_request(
            runner,
            init_at_random_ep_len=init_at_random_ep_len,
        )

    dispatch = getattr(runner, "run_frontres_formal_transaction", None)
    if not callable(dispatch):
        raise RuntimeError("v015 local identity sentinel requires the runner formal-transaction port")
    try:
        result = dispatch(provider)
        preupdate = frontres_preupdate_diagnostics(runner)
        if not preupdate:
            raise RuntimeError("v015 local sentinel requires a complete pre-update identity/observation snapshot")
        telemetry = dict(preupdate)
        result_diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        # B1: Reuse the ordinary Stage-3 final serializer so the sentinel cannot
        # silently drop sealed per-step Contact/ZMP evidence at its last adapter.
        from rsl_rl.runners.frontres_segment_training_telemetry import build_frontres_transaction_telemetry

        telemetry["sealed_transaction_evidence"] = build_frontres_transaction_telemetry(
            result,
            ppo=result.ppo_result,
        )
        telemetry.update(result_diagnostics)
        telemetry["optimizer_step_delta"] = int(getattr(result, "optimizer_step_delta", -1))
        telemetry["exact_one_update"] = telemetry["optimizer_step_delta"] == 1
        runner._frontres_local_sentinel_telemetry = telemetry
        print(
            "[FrontRES v017 Local Sentinel] "
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
            "[FrontRES v017 Live Snapshot] "
            + json.dumps(telemetry, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return result
    finally:
        batch = frontres_collection_batch(runner)
        if batch is not None:
            command = _motion_command_for_runner(runner)
            clear = getattr(command, "clear_frontres_local_scenario", None)
            if not callable(clear):
                raise RuntimeError("v015 local sentinel requires command-owned scenario cleanup")
            clear()
            close_frontres_local_scenarios(batch)
        clear_frontres_collection_context(runner)
