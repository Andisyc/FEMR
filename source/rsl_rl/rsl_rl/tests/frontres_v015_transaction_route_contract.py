#!/usr/bin/env python3
"""Deterministic S2 formal Unit-of-Work contract for v018/v006/v016."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
RSL_ROOT = SOURCE_ROOT / "rsl_rl"
runners_package = types.ModuleType("rsl_rl.runners")
runners_package.__path__ = [str(RSL_ROOT / "runners")]
sys.modules.setdefault("rsl_rl.runners", runners_package)
algorithms_package = types.ModuleType("rsl_rl.algorithms")
algorithms_package.__path__ = [str(RSL_ROOT / "algorithms")]
algorithms_package.FrontRESUnified = object
sys.modules.setdefault("rsl_rl.algorithms", algorithms_package)
modules_package = types.ModuleType("rsl_rl.modules")
modules_package.__path__ = [str(RSL_ROOT / "modules")]
modules_package.FrontRESActorCritic = object
sys.modules.setdefault("rsl_rl.modules", modules_package)

from rsl_rl.algorithms.frontres_segment_ppo import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FrontRESSegmentPPOBatch,
    FrontRESValueNormalizerState,
)
from rsl_rl.frontres.frontres_local_evaluation import FrontRESV017LocalEvaluationReport
from rsl_rl.frontres.frontres_outer_scenario_replay import (
    FrontRESOuterScenarioReplay,
    FrontRESScenarioKey,
)
from rsl_rl.frontres.frontres_return_utility import (
    FRONTRES_RETURN_UTILITY_ID,
    FRONTRES_RETURN_UTILITY_SCALE,
    frontres_symmetric_log_utility,
)
from rsl_rl.frontres.frontres_segment_storage_records import FrontRESV015GroupedCandidateMetadata
from rsl_rl.frontres.frontres_segment_warmup import resolve_frontres_k_stage_identity
from rsl_rl.runners import frontres_segment_formal_transaction as formal_transaction
from rsl_rl.runners.frontres_segment_formal_transaction import run_frontres_formal_transaction_update
from rsl_rl.runners.frontres_segment_runtime_types import open_frontres_checkpoint_transaction_barrier
from rsl_rl.runners.frontres_segment_runtime_types import reset_frontres_checkpoint_transaction
from rsl_rl.runners.frontres_segment_runtime_types import FrontRESFormalTransactionRequest
from rsl_rl.runners.frontres_segment_training_telemetry import build_frontres_transaction_telemetry
from rsl_rl.runners.frontres_segment_transaction import (
    FrontRESFormalTransactionPlan,
    capture_frontres_frozen_policy_snapshot,
)


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, 1, bias=False)
        self.critic = torch.nn.Linear(449, 1, bias=False)
        self.log_std = torch.nn.Parameter(torch.tensor(-0.4))


class _PolicyEvaluator:
    def __init__(self, policy: _Policy, critic_observations: torch.Tensor) -> None:
        self.policy = policy
        self.critic_observations = critic_observations

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        return {
            "log_prob": self.policy.actor(observations).reshape(-1) + self.policy.log_std,
            "value": self.policy.critic(self.critic_observations).reshape(-1),
            "entropy": self.policy.log_std.expand(observations.shape[0]),
        }


class _TrackingAdam(torch.optim.Adam):
    def __init__(self, params) -> None:
        super().__init__(params, lr=1.0e-3)
        self.frontres_step_count = 0

    def step(self, closure=None):
        self.frontres_step_count += 1
        return super().step(closure=closure)


SCHEDULE = (
    (8, 4, 200, 500, 1300, "lower-k8", 0.5, "linear-coupled-v1", 700, 2.381),
    (16, 4, 300, 300, 900, "lower-k16", 0.6, "linear-coupled-v1", 600, 2.381),
    (32, 4, 400, 300, 625, "lower-k32", 0.7, "linear-coupled-v1", 700, 2.381),
)


def test_formal_request_owns_the_grouped_ppo_batch_dependency() -> None:
    """Reject a request builder that reaches grouped storage with an undefined batch type."""

    assert formal_transaction.FrontRESSegmentPPOBatch is FrontRESSegmentPPOBatch


def _alg(policy: _Policy, optimizer: _TrackingAdam) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        optimizer=optimizer,
        actor_learning_rate=3.0e-7,
        critic_learning_rate=1.0e-5,
        frontres_segment_actor_joint_lr=1.0e-6,
        clip_param=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        use_clipped_value_loss=True,
        max_grad_norm=0.5,
        lambda_supervised=0.0,
        lambda_supervised_min=0.0,
        frontres_formal_transaction_enabled=True,
        frontres_segment_advantage_normalization="grouped_scale_only",
        frontres_critic_value_normalization=FRONTRES_VALUE_NORMALIZATION_ID,
        frontres_critic_value_normalizer_decay=0.9,
        frontres_critic_value_normalizer_scale_floor=1.0,
        frontres_critic_value_normalizer_state=FrontRESValueNormalizerState(),
        frontres_hsl_init_enabled=False,
        frontres_hsl_rollout_label_enabled=False,
        frontres_segment_k_curriculum=SCHEDULE,
        frontres_segment_k_curriculum_fingerprint="",
        frontres_segment_max_horizon_k=32,
        frontres_future_offsets=(1, 2),
        frontres_future_intent_layout_version="frontres-v015-future-intent-q29-v1",
        frontres_segment_live_train_enabled=False,
        frontres_segment_live_update_loop_only=False,
        frontres_segment_live_single_update_only=False,
        frontres_method_contract_id="FRS-METHOD-v024",
        frontres_gain_contract_id="FRS-GAIN-v008",
        frontres_optimization_contract_id="FRS-PPO-v011",
        frontres_training_contract_id="FRS-TRAIN-v023",
        frontres_scalar_target_id="symmetric-log-recovery-aware-utility-v1",
        frontres_physics_schema_id="clean-anchored-contact-zmp-survival-v1",
        frontres_grouped_schema_id="grouped-all-attempt-scalar-v1",
        frontres_critic_support_context_id="action-pre-support-plan-kmax32-v1",
        frontres_return_utility_id=FRONTRES_RETURN_UTILITY_ID,
        frontres_return_utility_scale=FRONTRES_RETURN_UTILITY_SCALE,
        frontres_gain_beta=0.02,
    )


def _report(transaction_id: str, *, count: int = 8, horizon_k: int = 8) -> FrontRESV017LocalEvaluationReport:
    if count % 8 != 0:
        raise ValueError("v022 test report requires eight exact-M Scenario groups")
    active_m = count // 8
    scalar = tuple(round(0.2 - 0.1 * row, 10) for row in range(count))
    intent_noisy = tuple(round(1.1 + 0.1 * row, 10) for row in range(count))
    physics_noisy = tuple(round(2.1 + 0.1 * row, 10) for row in range(count))
    contact = tuple(tuple((1.0, 1.0) for _ in range(horizon_k)) for _ in range(count))
    zmp = tuple(tuple(0.01 for _ in range(horizon_k)) for _ in range(count))
    survival = tuple(tuple(1.0 for _ in range(horizon_k)) for _ in range(count))
    false_steps = tuple(tuple(False for _ in range(horizon_k)) for _ in range(count))
    true_steps = tuple(tuple(True for _ in range(horizon_k)) for _ in range(count))
    zero_steps = tuple(tuple(0.0 for _ in range(horizon_k)) for _ in range(count))
    scenario_ids = tuple(f"scenario-{row // active_m}" for row in range(count))
    noisy_hashes = tuple(f"hash-{row // active_m}" for row in range(count))
    report = FrontRESV017LocalEvaluationReport(
        transaction_id=transaction_id,
        scenario_ids=scenario_ids,
        noisy_segment_hashes=noisy_hashes,
        policy_actions=tuple((0.01, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(count)),
        valid_policy_row_mask=(True,) * count,
        intent_remaining_noisy=intent_noisy,
        intent_remaining_repaired=tuple(
            round(value - gain, 10) for value, gain in zip(intent_noisy, scalar, strict=True)
        ),
        physics_remaining_noisy=physics_noisy,
        physics_remaining_repaired=tuple(
            round(value - gain, 10) for value, gain in zip(physics_noisy, scalar, strict=True)
        ),
        intent_channel_noisy=tuple(tuple(float(10 * row + col) for col in range(6)) for row in range(count)),
        intent_channel_repaired=tuple(tuple(float(20 * row + col) for col in range(6)) for row in range(count)),
        physics_channel_noisy=tuple(tuple(float(30 * row + col) for col in range(4)) for row in range(count)),
        physics_channel_repaired=tuple(tuple(float(40 * row + col) for col in range(4)) for row in range(count)),
        intent_gain=scalar,
        physics_gain=scalar,
        recovery_pressure=(1.0,) * count,
        weighted_physics_gain=scalar,
        support_foot_drift_noisy=tuple(round(0.03 * (row + 1), 10) for row in range(count)),
        support_foot_drift_repaired=tuple(round(0.01 * (row + 1), 10) for row in range(count)),
        repair_cost=(0.1,) * count,
        repair_penalty=(0.002,) * count,
        cost_free_score=tuple(round(value + 0.002, 10) for value in scalar),
        intent_scales=(0.087, 0.087, 0.10, 0.75, 2.0, 0.05),
        physics_scales=(0.10, 0.03, 0.02, 0.10),
        translation_repair_scale=0.10,
        rotation_repair_scale=0.08726646259971647,
        beta=0.02,
        gain_total=scalar,
        policy_values=(0.0,) * count,
        raw_advantages=scalar,
        clean_execution_count=(1,) * 8,
        noisy_execution_count=(1,) * 8,
        expected_support_steps=contact,
        contact_clean_steps=contact,
        contact_noisy_steps=contact,
        contact_repair_steps=contact,
        zmp_clean_steps=zmp,
        zmp_noisy_steps=zmp,
        zmp_repair_steps=zmp,
        survival_clean_steps=survival,
        survival_noisy_steps=survival,
        survival_repair_steps=survival,
        contact_violation_repair_steps=false_steps,
        zmp_applicable_repair_steps=true_steps,
        zmp_violation_repair_steps=zero_steps,
        zmp_recovery_repair_steps=zero_steps,
        unplanned_contact_repair_steps=false_steps,
        lateral_roll_repair_steps=zero_steps,
        lateral_roll_cumulative_mean_repair_steps=zero_steps,
        sustained_lean_repair=(False,) * count,
    )
    report.validate()
    return report


def _request(
    *,
    iteration: int = 0,
    runner: SimpleNamespace | None = None,
) -> tuple[SimpleNamespace, FrontRESFormalTransactionRequest, _Policy]:
    if runner is None:
        policy = _Policy()
        optimizer = _TrackingAdam(
            [
                {
                    "params": (*tuple(policy.actor.parameters()), policy.log_std),
                    "lr": 3.0e-7,
                    "frontres_role": "actor",
                },
                {
                    "params": tuple(policy.critic.parameters()),
                    "lr": 1.0e-5,
                    "frontres_role": "critic",
                },
            ]
        )
        runner = SimpleNamespace(alg=_alg(policy, optimizer), current_learning_iteration=iteration)
    else:
        policy = runner.alg.policy
        runner.current_learning_iteration = iteration
    runner.alg.frontres_critic_value_normalizer_state = replace(
        runner.alg.frontres_critic_value_normalizer_state,
        update_count=iteration,
    )
    identity = resolve_frontres_k_stage_identity(
        schedule=SCHEDULE,
        committed_update_iteration=iteration,
        max_horizon_k=32,
    )
    count = 8 * identity.active_m
    transaction_id = f"tx-v017-formal-{iteration}"
    snapshot = capture_frontres_frozen_policy_snapshot(runner, transaction_id=transaction_id)
    outer_replay = getattr(runner, "_frontres_outer_scenario_replay", None)
    if outer_replay is None:
        outer_replay = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=7)
        runner._frontres_outer_scenario_replay = outer_replay
    outer_plan = outer_replay.plan(
        transaction_id=transaction_id,
        curriculum=identity,
        num_segments=32,
        eligible=lambda _segment_id: True,
        global_family=lambda _segment_id: "local_rp",
    )
    source = torch.arange(8).repeat_interleave(identity.active_m)
    trial = torch.arange(identity.active_m).repeat(8)
    selected_segments = tuple(selection.segment_id for selection in outer_plan.selections)
    segment = torch.tensor(
        [segment_id for segment_id in selected_segments for _ in range(identity.active_m)]
    )
    motion_by_source = tuple(f"motion-{index}" for index in range(8))
    frame_by_source = tuple(4 * (index + 1) for index in range(8))
    scenario_by_source = tuple(f"scenario-{index}" for index in range(8))
    noisy_hash_by_source = tuple(f"hash-{index}" for index in range(8))
    x_t_by_source = tuple(f"x-{index}" for index in range(8))
    motion_ids = tuple(motion_by_source[int(row)] for row in source.tolist())
    start_frames = torch.tensor([frame_by_source[int(row)] for row in source.tolist()])
    scenario_ids = tuple(scenario_by_source[int(row)] for row in source.tolist())
    noisy_hashes = tuple(noisy_hash_by_source[int(row)] for row in source.tolist())
    x_t_ids = tuple(x_t_by_source[int(row)] for row in source.tolist())
    plan = FrontRESFormalTransactionPlan(
        snapshot=snapshot,
        motion_ids=motion_ids,
        start_frames=start_frames,
        segment_ids=segment,
        source_index=source,
        trial_index=trial,
        horizon_k=torch.full((count,), identity.active_k),
        scenario_ids=scenario_ids,
        noisy_segment_hashes=noisy_hashes,
        x_t_identities=x_t_ids,
        intent_q29_provenance="deployment_noisy_q29",
        intent_q29_source="motion_internal_q29",
    )
    metadata = FrontRESV015GroupedCandidateMetadata(
        transaction_id=plan.transaction_id,
        policy_snapshot_id=plan.policy_snapshot_id,
        motion_ids=plan.motion_ids,
        start_frames=plan.start_frames,
        segment_ids=segment,
        source_index=source,
        trial_index=trial,
        horizon_k=plan.horizon_k,
        evidence_valid_step_count=torch.full((count,), identity.active_k),
        trial_role=("policy",) * count,
        noisy_segment_hashes=plan.noisy_segment_hashes,
        scenario_ids=plan.scenario_ids,
        x_t_identities=plan.x_t_identities,
        intent_q29_provenance=plan.intent_q29_provenance,
        intent_q29_source=plan.intent_q29_source,
    )
    obs = torch.tensor(
        [[float(source_id + 1), float(trial_id + 1)] for source_id in range(8) for trial_id in range(identity.active_m)]
    )
    critic_obs = torch.zeros(count, 449)
    critic_obs[:, 0] = source.to(dtype=torch.float32) + 1.0
    returns = torch.tensor(_report(transaction_id, count=count, horizon_k=identity.active_k).gain_total)
    utility_returns = frontres_symmetric_log_utility(returns)
    batch = FrontRESSegmentPPOBatch(
        observations=obs,
        privileged_observations=critic_obs,
        actions=torch.zeros(count, 6),
        old_log_probs=torch.zeros(count),
        old_values=torch.zeros(count),
        returns=returns,
        advantages=utility_returns,
        valid_mask=torch.ones(count, dtype=torch.bool),
        segment_ids=segment,
        old_means=torch.zeros(count, 6),
        old_sigmas=torch.ones(count, 6),
        transaction_metadata=metadata,
        transaction_row_indices=torch.arange(count),
    )
    request = FrontRESFormalTransactionRequest(
        plan=plan,
        candidate_batches=(batch,),
        diagnostic_reports=(_report(plan.transaction_id, count=count, horizon_k=identity.active_k),),
        curriculum_fingerprint=identity.schedule_fingerprint,
        k_stage_index=identity.stage_index,
        active_k=identity.active_k,
        active_m=identity.active_m,
        k_stage_iteration=identity.stage_iteration,
        training_iteration=identity.absolute_iteration,
        warmup_phase_name=identity.phase.name,
        warmup_actor_loss_weight=identity.phase.actor_loss_weight,
        warmup_actor_learning_rate=identity.phase.actor_learning_rate,
        dr_stage_fingerprint=identity.dr_stage_fingerprint,
        dr_progress=identity.dr_progress,
        d_cap=identity.d_cap,
        dr_class_by_segment=tuple(selection.dr_class for selection in outer_plan.selections),
        dr_strength_by_segment=tuple(selection.perturbation_strength for selection in outer_plan.selections),
        outer_replay_plan=outer_plan,
        outer_replay_scenario_keys=tuple(
            FrontRESScenarioKey(
                motion_id=motion_by_source[index],
                start_frame=frame_by_source[index],
                segment_id=selection.segment_id,
                x_t_identity=x_t_by_source[index],
                perturbation_family=selection.perturbation_family,
                perturbation_strength=selection.perturbation_strength,
                perturbation_seed=selection.perturbation_seed,
                noisy_segment_hash=noisy_hash_by_source[index],
                horizon_k=identity.active_k,
                future_intent_identity=f"future-{selection.segment_id}",
                planned_support_identity=f"support-{selection.segment_id}",
            )
            for index, selection in enumerate(outer_plan.selections)
        ),
        policy_evaluator=_PolicyEvaluator(policy, critic_obs),
    )
    return runner, request, policy


def _clone_optimizer_state(optimizer: torch.optim.Optimizer, parameters: tuple[torch.nn.Parameter, ...]):
    return {
        id(parameter): {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for name, value in optimizer.state.get(parameter, {}).items()
        }
        for parameter in parameters
    }


def _assert_optimizer_state_equal(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    expected: dict[int, dict[str, object]],
) -> None:
    for parameter in parameters:
        actual_state = optimizer.state.get(parameter, {})
        expected_state = expected[id(parameter)]
        assert actual_state.keys() == expected_state.keys()
        for name, expected_value in expected_state.items():
            actual_value = actual_state[name]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def _prime_actor_optimizer_history(runner: SimpleNamespace) -> tuple[torch.nn.Parameter, ...]:
    policy = runner.alg.policy
    actor_parameters = (*tuple(policy.actor.parameters()), policy.log_std)
    runner.alg.optimizer.zero_grad(set_to_none=True)
    for parameter in actor_parameters:
        parameter.grad = torch.ones_like(parameter)
    runner.alg.optimizer.step()
    runner.alg.optimizer.zero_grad(set_to_none=True)
    return actor_parameters


def test_exact_one_scalar_commit_updates_actor_and_critic_from_first_transaction() -> None:
    runner, request, policy = _request()
    open_frontres_checkpoint_transaction_barrier(runner)
    actor_before = {name: value.detach().clone() for name, value in policy.actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in policy.critic.state_dict().items()}
    result = run_frontres_formal_transaction_update(runner, request)
    assert result.optimizer_step_delta == 1 and result.update_invocation_count == 1
    assert len(runner._frontres_outer_scenario_replay.records) == 8
    assert result.diagnostics["outer_replay"]["state_delta"] == 1
    assert runner.alg.frontres_critic_value_normalizer_state.update_count == 1
    assert result.valid_row_count == 32 and result.policy_attempt_count == 32
    assert request.warmup_phase_name == "low_dr_joint_init"
    assert request.warmup_actor_loss_weight > 0.0
    assert any(not torch.equal(value, actor_before[name]) for name, value in policy.actor.state_dict().items())
    assert any(not torch.equal(value, critic_before[name]) for name, value in policy.critic.state_dict().items())
    assert result.diagnostics["gain_contract_id"] == "FRS-GAIN-v008"
    assert result.diagnostics["optimization_contract_id"] == "FRS-PPO-v011"
    assert "constraint_kkt_max_violation" not in result.diagnostics
    telemetry = build_frontres_transaction_telemetry(result, ppo=result.ppo_result)
    assert telemetry["clean_execution_count"] == (1,) * 8
    assert telemetry["noisy_execution_count"] == (1,) * 8
    assert telemetry["policy_row_count"] == 32
    assert telemetry["optimizer_step_delta"] == 1
    assert telemetry["update_count"] == 1
    assert telemetry["gain_contract_id"] == "FRS-GAIN-v008"
    assert telemetry["intent_gain"] == tuple(round(0.2 - 0.1 * row, 10) for row in range(32))
    assert telemetry["intent_remaining_noisy"] == tuple(round(1.1 + 0.1 * row, 10) for row in range(32))
    assert telemetry["intent_channel_noisy"][2] == (20.0, 21.0, 22.0, 23.0, 24.0, 25.0)
    assert telemetry["physics_channel_repaired"][3] == (120.0, 121.0, 122.0, 123.0)
    assert telemetry["recovery_pressure"] == (1.0,) * 32
    assert telemetry["intent_scales"] == (0.087, 0.087, 0.10, 0.75, 2.0, 0.05)
    assert telemetry["physics_scales"] == (0.10, 0.03, 0.02, 0.10)
    assert telemetry["beta"] == 0.02
    assert telemetry["contact_violation_repair_steps"] == tuple((False,) * 8 for _ in range(32))
    assert telemetry["zmp_applicable_repair_steps"] == tuple((True,) * 8 for _ in range(32))
    assert telemetry["zmp_violation_repair_steps"] == tuple((0.0,) * 8 for _ in range(32))
    assert telemetry["sustained_lean_repair"] == (False,) * 32
    assert telemetry["grouped_attempt_mass_shares"] == (0.03125,) * 32
    assert "constraint_projection_status" not in telemetry

    before = repr(result.diagnostics)
    permuted_diagnostics = dict(result.diagnostics)
    permutation = tuple(reversed(range(32)))
    permuted_diagnostics["v007_diagnostic_report_row_order"] = permutation
    permuted = build_frontres_transaction_telemetry(
        replace(result, diagnostics=permuted_diagnostics),
        ppo=result.ppo_result,
    )
    assert permuted["scenario_ids"] == tuple(telemetry["scenario_ids"][row] for row in permutation)
    assert permuted["gain_total"] == tuple(telemetry["gain_total"][row] for row in permutation)
    assert repr(result.diagnostics) == before


def test_k_transitions_keep_one_critic_and_restart_nonzero_joint_adaptation() -> None:
    runner, _unused, policy = _request(iteration=0)
    actor_parameters = _prime_actor_optimizer_history(runner)
    actor_state = {id(parameter): parameter.detach().clone() for parameter in actor_parameters}
    actor_optimizer_state = _clone_optimizer_state(runner.alg.optimizer, actor_parameters)
    critic_identity = id(policy.critic)

    for iteration, expected_k, expected_m in ((2000, 16, 4), (3500, 32, 4)):
        runner, request, current_policy = _request(iteration=iteration, runner=runner)
        assert id(current_policy.critic) == critic_identity
        assert request.warmup_phase_name == "low_dr_joint_init"
        assert request.warmup_actor_loss_weight > 0.0
        assert request.active_k == expected_k and request.active_m == expected_m
        critic_before = {
            name: value.detach().clone() for name, value in current_policy.critic.state_dict().items()
        }
        open_frontres_checkpoint_transaction_barrier(runner)
        result = run_frontres_formal_transaction_update(runner, request)
        assert result.optimizer_step_delta == 1
        assert result.valid_row_count == 8 * expected_m
        assert id(current_policy.critic) == critic_identity
        assert any(
            not torch.equal(value, critic_before[name])
            for name, value in current_policy.critic.state_dict().items()
        )
        assert any(not torch.equal(parameter, actor_state[id(parameter)]) for parameter in actor_parameters)
        reset_frontres_checkpoint_transaction(runner)


def test_coupled_ramp_identity_reaches_transaction_and_telemetry() -> None:
    runner, request, policy = _request(iteration=203)
    assert request.warmup_phase_name == "coupled_ramp"
    assert request.warmup_actor_loss_weight == 1.0
    assert 3.0e-7 < request.warmup_actor_learning_rate < 1.0e-6
    actor_before = {
        name: value.detach().clone() for name, value in policy.actor.state_dict().items()
    }
    open_frontres_checkpoint_transaction_barrier(runner)
    result = run_frontres_formal_transaction_update(runner, request)
    assert result.diagnostics["warmup_phase"] == "coupled_ramp"
    assert result.diagnostics["actor_loss_weight"] == request.warmup_actor_loss_weight
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in policy.actor.state_dict().items()
    )
    telemetry = build_frontres_transaction_telemetry(result, ppo=result.ppo_result)
    assert telemetry["warmup_phase"] == "coupled_ramp"
    assert telemetry["actor_loss_weight"] == request.warmup_actor_loss_weight


def test_partial_transaction_rejects_before_update() -> None:
    runner, request, _policy = _request()
    normalizer_before = runner.alg.frontres_critic_value_normalizer_state
    open_frontres_checkpoint_transaction_barrier(runner)
    batch = request.candidate_batches[0]
    partial = replace(batch, valid_mask=torch.tensor([True] * 31 + [False]))
    rejected = replace(request, candidate_batches=(partial,))
    try:
        run_frontres_formal_transaction_update(runner, rejected)
    except (ValueError, RuntimeError):
        assert runner.alg.optimizer.frontres_step_count == 0
        assert runner.alg.frontres_critic_value_normalizer_state == normalizer_before
    else:
        raise AssertionError("partial exact-M transaction must fail before optimizer update")


def test_value_normalizer_iteration_mismatch_rejects_before_update() -> None:
    runner, request, _policy = _request()
    runner.alg.frontres_critic_value_normalizer_state = replace(
        runner.alg.frontres_critic_value_normalizer_state,
        update_count=1,
    )
    open_frontres_checkpoint_transaction_barrier(runner)
    try:
        run_frontres_formal_transaction_update(runner, request)
    except RuntimeError as exc:
        assert "normalizer count" in str(exc)
        assert runner.alg.optimizer.frontres_step_count == 0
    else:
        raise AssertionError("mismatched value-normalizer iteration must fail before optimizer update")


def test_phase_reset_routes_mode_through_sealed_reset_owner() -> None:
    calls: list[tuple[object, str]] = []
    original = formal_transaction._apply_current_segment_reset

    def reset(_runner, *, pair_layout, local_scenario_execution_mode):
        calls.append((pair_layout, local_scenario_execution_mode))
        return SimpleNamespace(success_mask=torch.ones(32, dtype=torch.bool))

    formal_transaction._apply_current_segment_reset = reset
    layout = object()
    try:
        for mode in ("clean_baseline", "noisy_baseline", "repair_attempts"):
            formal_transaction._reset_frontres_v017_phase(
                object(),
                pair_layout=layout,
                mode=mode,
                policy_row_count=32,
                label="contract",
            )
    finally:
        formal_transaction._apply_current_segment_reset = original
    assert calls == [
        (layout, "clean_baseline"),
        (layout, "noisy_baseline"),
        (layout, "repair_attempts"),
    ]


def main() -> None:
    torch.manual_seed(0)
    test_formal_request_owns_the_grouped_ppo_batch_dependency()
    test_exact_one_scalar_commit_updates_actor_and_critic_from_first_transaction()
    test_k_transitions_keep_one_critic_and_restart_nonzero_joint_adaptation()
    test_coupled_ramp_identity_reaches_transaction_and_telemetry()
    test_partial_transaction_rejects_before_update()
    test_value_normalizer_iteration_mismatch_rejects_before_update()
    test_phase_reset_routes_mode_through_sealed_reset_owner()
    print("frontres_v015_transaction_route_contract: v023 robust outer replay exact-one ok", flush=True)


if __name__ == "__main__":
    main()
