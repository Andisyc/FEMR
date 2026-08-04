#!/usr/bin/env python3
"""Deterministic S2 formal Unit-of-Work contract for v017/v005/v012."""

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

from rsl_rl.algorithms.frontres_segment_ppo import FrontRESSegmentPPOBatch
from rsl_rl.frontres.frontres_local_evaluation import FrontRESV017LocalEvaluationReport
from rsl_rl.frontres.frontres_segment_storage_records import FrontRESV015GroupedCandidateMetadata
from rsl_rl.frontres.frontres_segment_warmup import resolve_frontres_k_stage_identity
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
        self.critic = torch.nn.Linear(2, 1, bias=False)
        self.log_std = torch.nn.Parameter(torch.tensor(-0.4))

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        return {
            "log_prob": self.actor(observations).reshape(-1) + self.log_std,
            "value": self.critic(observations).reshape(-1),
            "entropy": self.log_std.expand(observations.shape[0]),
        }


class _TrackingAdam(torch.optim.Adam):
    def __init__(self, params) -> None:
        super().__init__(params, lr=1.0e-3)
        self.frontres_step_count = 0

    def step(self, closure=None):
        self.frontres_step_count += 1
        return super().step(closure=closure)


SCHEDULE = (
    (8, 2, 200, 500, 1300, "lower-k8", 0.5, "linear-joint-v1", 1300, 2.381),
    (16, 3, 300, 300, 900, "lower-k16", 0.6, "linear-joint-v1", 900, 2.381),
    (32, 4, 400, 300, 625, "lower-k32", 0.7, "linear-joint-v1", 625, 2.381),
)


def _alg(policy: _Policy, optimizer: _TrackingAdam) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        optimizer=optimizer,
        clip_param=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        use_clipped_value_loss=True,
        max_grad_norm=1.0,
        lambda_supervised=0.0,
        lambda_supervised_min=0.0,
        frontres_formal_transaction_enabled=True,
        frontres_segment_advantage_normalization="grouped_scale_only",
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
        frontres_method_contract_id="FRS-METHOD-v017",
        frontres_gain_contract_id="FRS-GAIN-v007",
        frontres_optimization_contract_id="FRS-PPO-v005",
        frontres_training_contract_id="FRS-TRAIN-v014",
        frontres_scalar_target_id="clean-anchored-recovery-aware-gain-v1",
        frontres_physics_schema_id="clean-anchored-contact-zmp-survival-v1",
        frontres_grouped_schema_id="grouped-all-attempt-scalar-v1",
        frontres_gain_beta=0.02,
    )


def _report(transaction_id: str, *, count: int = 4, horizon_k: int = 8) -> FrontRESV017LocalEvaluationReport:
    split = count // 2
    scalar = tuple(round(0.2 - 0.1 * row, 10) for row in range(split)) + tuple(
        round(-0.1 - 0.1 * row, 10) for row in range(split)
    )
    intent_noisy = tuple(round(1.1 + 0.1 * row, 10) for row in range(count))
    physics_noisy = tuple(round(2.1 + 0.1 * row, 10) for row in range(count))
    contact = tuple(tuple((1.0, 1.0) for _ in range(horizon_k)) for _ in range(count))
    zmp = tuple(tuple(0.01 for _ in range(horizon_k)) for _ in range(count))
    survival = tuple(tuple(1.0 for _ in range(horizon_k)) for _ in range(count))
    false_steps = tuple(tuple(False for _ in range(horizon_k)) for _ in range(count))
    true_steps = tuple(tuple(True for _ in range(horizon_k)) for _ in range(count))
    zero_steps = tuple(tuple(0.0 for _ in range(horizon_k)) for _ in range(count))
    scenario_ids = tuple("sa" if row < split else "sb" for row in range(count))
    noisy_hashes = tuple("ha" if row < split else "hb" for row in range(count))
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
        clean_execution_count=(1, 1),
        noisy_execution_count=(1, 1),
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
        optimizer = _TrackingAdam(policy.parameters())
        runner = SimpleNamespace(alg=_alg(policy, optimizer), current_learning_iteration=iteration)
    else:
        policy = runner.alg.policy
        runner.current_learning_iteration = iteration
    identity = resolve_frontres_k_stage_identity(
        schedule=SCHEDULE,
        committed_update_iteration=iteration,
        max_horizon_k=32,
    )
    count = 2 * identity.active_m
    transaction_id = f"tx-v017-formal-{iteration}"
    snapshot = capture_frontres_frozen_policy_snapshot(runner, transaction_id=transaction_id)
    source = torch.tensor([0] * identity.active_m + [1] * identity.active_m)
    trial = torch.tensor(list(range(identity.active_m)) * 2)
    segment = torch.tensor([10] * identity.active_m + [11] * identity.active_m)
    motion_ids = tuple("motion-a" if row < identity.active_m else "motion-b" for row in range(count))
    start_frames = torch.tensor([4] * identity.active_m + [8] * identity.active_m)
    scenario_ids = tuple("sa" if row < identity.active_m else "sb" for row in range(count))
    noisy_hashes = tuple("ha" if row < identity.active_m else "hb" for row in range(count))
    x_t_ids = tuple("xa" if row < identity.active_m else "xb" for row in range(count))
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
        [[float(row + 1), 0.0] for row in range(identity.active_m)]
        + [[0.0, float(row + 1)] for row in range(identity.active_m)]
    )
    returns = torch.tensor(_report(transaction_id, count=count, horizon_k=identity.active_k).gain_total)
    batch = FrontRESSegmentPPOBatch(
        observations=obs,
        privileged_observations=obs.clone(),
        actions=torch.zeros(count, 6),
        old_log_probs=torch.zeros(count),
        old_values=torch.zeros(count),
        returns=returns,
        advantages=returns.clone(),
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
        dr_stage_fingerprint=identity.dr_stage_fingerprint,
        dr_progress=identity.dr_progress,
        d_cap=identity.d_cap,
        dr_class_by_segment=("easy", "hard"),
        dr_strength_by_segment=(0.1, 0.4),
        policy_evaluator=policy,
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


def test_exact_one_scalar_commit_and_critic_only() -> None:
    runner, request, policy = _request()
    open_frontres_checkpoint_transaction_barrier(runner)
    actor_before = {name: value.detach().clone() for name, value in policy.actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in policy.critic.state_dict().items()}
    result = run_frontres_formal_transaction_update(runner, request)
    assert result.optimizer_step_delta == 1 and result.update_invocation_count == 1
    assert result.valid_row_count == 4 and result.policy_attempt_count == 4
    assert all(torch.equal(value, actor_before[name]) for name, value in policy.actor.state_dict().items())
    assert any(not torch.equal(value, critic_before[name]) for name, value in policy.critic.state_dict().items())
    assert result.diagnostics["gain_contract_id"] == "FRS-GAIN-v007"
    assert result.diagnostics["optimization_contract_id"] == "FRS-PPO-v005"
    assert "constraint_kkt_max_violation" not in result.diagnostics
    telemetry = build_frontres_transaction_telemetry(result, ppo=result.ppo_result)
    assert telemetry["clean_execution_count"] == (1, 1)
    assert telemetry["noisy_execution_count"] == (1, 1)
    assert telemetry["policy_row_count"] == 4
    assert telemetry["optimizer_step_delta"] == 1
    assert telemetry["update_count"] == 1
    assert telemetry["gain_contract_id"] == "FRS-GAIN-v007"
    assert telemetry["intent_gain"] == (0.2, 0.1, -0.1, -0.2)
    assert telemetry["intent_remaining_noisy"] == (1.1, 1.2, 1.3, 1.4)
    assert telemetry["physics_remaining_repaired"] == (1.9, 2.1, 2.4, 2.6)
    assert telemetry["intent_channel_noisy"][2] == (20.0, 21.0, 22.0, 23.0, 24.0, 25.0)
    assert telemetry["physics_channel_repaired"][3] == (120.0, 121.0, 122.0, 123.0)
    assert telemetry["recovery_pressure"] == (1.0, 1.0, 1.0, 1.0)
    assert telemetry["support_foot_drift_noisy"] == (0.03, 0.06, 0.09, 0.12)
    assert telemetry["cost_free_score"] == (0.202, 0.102, -0.098, -0.198)
    assert telemetry["intent_scales"] == (0.087, 0.087, 0.10, 0.75, 2.0, 0.05)
    assert telemetry["physics_scales"] == (0.10, 0.03, 0.02, 0.10)
    assert telemetry["beta"] == 0.02
    assert telemetry["contact_violation_repair_steps"] == tuple((False,) * 8 for _ in range(4))
    assert telemetry["zmp_applicable_repair_steps"] == tuple((True,) * 8 for _ in range(4))
    assert telemetry["zmp_violation_repair_steps"] == tuple((0.0,) * 8 for _ in range(4))
    assert telemetry["sustained_lean_repair"] == (False,) * 4
    assert telemetry["grouped_attempt_mass_shares"] == (0.25, 0.25, 0.25, 0.25)
    assert "constraint_projection_status" not in telemetry

    before = repr(result.diagnostics)
    permuted_diagnostics = dict(result.diagnostics)
    permuted_diagnostics["v007_diagnostic_report_row_order"] = (3, 1, 0, 2)
    permuted = build_frontres_transaction_telemetry(
        replace(result, diagnostics=permuted_diagnostics),
        ppo=result.ppo_result,
    )
    assert permuted["scenario_ids"] == ("sb", "sa", "sa", "sb")
    assert permuted["gain_total"] == (-0.2, 0.1, 0.2, -0.1)
    assert repr(result.diagnostics) == before


def test_k_transitions_keep_one_critic_and_preserve_frozen_actor_optimizer_state() -> None:
    runner, _unused, policy = _request(iteration=0)
    actor_parameters = _prime_actor_optimizer_history(runner)
    actor_state = {id(parameter): parameter.detach().clone() for parameter in actor_parameters}
    actor_optimizer_state = _clone_optimizer_state(runner.alg.optimizer, actor_parameters)
    critic_identity = id(policy.critic)

    for iteration, expected_k, expected_m in ((2000, 16, 3), (3500, 32, 4)):
        runner, request, current_policy = _request(iteration=iteration, runner=runner)
        assert id(current_policy.critic) == critic_identity
        assert request.warmup_phase_name == "critic_only"
        assert request.warmup_actor_loss_weight == 0.0
        assert request.active_k == expected_k and request.active_m == expected_m
        critic_before = {
            name: value.detach().clone() for name, value in current_policy.critic.state_dict().items()
        }
        open_frontres_checkpoint_transaction_barrier(runner)
        result = run_frontres_formal_transaction_update(runner, request)
        assert result.optimizer_step_delta == 1
        assert result.valid_row_count == 2 * expected_m
        assert id(current_policy.critic) == critic_identity
        assert any(
            not torch.equal(value, critic_before[name])
            for name, value in current_policy.critic.state_dict().items()
        )
        for parameter in actor_parameters:
            assert torch.equal(parameter, actor_state[id(parameter)])
        _assert_optimizer_state_equal(runner.alg.optimizer, actor_parameters, actor_optimizer_state)
        reset_frontres_checkpoint_transaction(runner)


def test_actor_ramp_identity_reaches_transaction_and_telemetry() -> None:
    runner, request, policy = _request(iteration=203)
    assert request.warmup_phase_name == "actor_ramp"
    assert 0.0 < request.warmup_actor_loss_weight < 1.0
    actor_before = {
        name: value.detach().clone() for name, value in policy.actor.state_dict().items()
    }
    open_frontres_checkpoint_transaction_barrier(runner)
    result = run_frontres_formal_transaction_update(runner, request)
    assert result.diagnostics["warmup_phase"] == "actor_ramp"
    assert result.diagnostics["actor_loss_weight"] == request.warmup_actor_loss_weight
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in policy.actor.state_dict().items()
    )
    telemetry = build_frontres_transaction_telemetry(result, ppo=result.ppo_result)
    assert telemetry["warmup_phase"] == "actor_ramp"
    assert telemetry["actor_loss_weight"] == request.warmup_actor_loss_weight


def test_partial_transaction_rejects_before_update() -> None:
    runner, request, _policy = _request()
    batch = request.candidate_batches[0]
    partial = replace(batch, valid_mask=torch.tensor([True, True, True, False]))
    rejected = replace(request, candidate_batches=(partial,))
    try:
        run_frontres_formal_transaction_update(runner, rejected)
    except (ValueError, RuntimeError):
        assert runner.alg.optimizer.frontres_step_count == 0
    else:
        raise AssertionError("partial exact-M transaction must fail before optimizer update")


def main() -> None:
    test_exact_one_scalar_commit_and_critic_only()
    test_k_transitions_keep_one_critic_and_preserve_frozen_actor_optimizer_state()
    test_actor_ramp_identity_reaches_transaction_and_telemetry()
    test_partial_transaction_rejects_before_update()
    print("frontres_v015_transaction_route_contract: v017 scalar exact-one ok", flush=True)


if __name__ == "__main__":
    main()
