#!/usr/bin/env python3
"""CPU-only Step 4B-S2 contract for the v015 formal transaction connector."""
from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
CANDIDATE_TEST = RSL_ROOT / "tests" / "frontres_v015_grouped_candidate_adapter_contract.py"
LIVE_SAMPLER_PATH = RSL_ROOT / "runners" / "frontres_segment_live_sampler.py"
LIVE_UPDATE_LOOP_PATH = RSL_ROOT / "runners" / "frontres_segment_live_update_loop.py"
LIVE_TRAINING_PATH = RSL_ROOT / "runners" / "frontres_segment_live_training.py"
ON_POLICY_RUNNER_PATH = RSL_ROOT / "runners" / "on_policy_runner.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_runtime_error(fn) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


class _TrackingSGD(torch.optim.SGD):
    """Test-only explicit optimizer event counter required by fake S2."""

    def __init__(self, params) -> None:
        super().__init__(params, lr=0.05)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure=closure)


def _load_owners():
    candidate_contract = _load("frontres_v015_transaction_candidate_helper", CANDIDATE_TEST)
    owners = candidate_contract._load_owners()
    live_probe = owners[6]
    ppo = owners[8]
    live_probe.FrontRESSegmentPPOConfig = ppo.FrontRESSegmentPPOConfig
    live_probe.compute_frontres_segment_ppo_loss = ppo.compute_frontres_segment_ppo_loss
    live_probe.install_frontres_v004_projected_gradients = ppo.install_frontres_v004_projected_gradients
    live_sampler = sys.modules.get("rsl_rl.runners.frontres_segment_live_sampler")
    if live_sampler is None:
        live_sampler = _load("rsl_rl.runners.frontres_segment_live_sampler", LIVE_SAMPLER_PATH)
    live_update_loop = _load("frontres_v015_transaction_live_update_loop", LIVE_UPDATE_LOOP_PATH)
    return candidate_contract, owners, live_sampler, live_update_loop


def _formal_alg(policy: torch.nn.Module, optimizer: _TrackingSGD) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        optimizer=optimizer,
        clip_param=0.2,
        value_loss_coef=0.0,
        entropy_coef=0.0,
        use_clipped_value_loss=True,
        max_grad_norm=1.0,
        lambda_supervised=0.0,
        lambda_supervised_min=0.0,
        frontres_v015_formal_transaction_enabled=True,
        frontres_training_objective="segment_replay_hrl",
        frontres_segment_replay_enabled=True,
        frontres_segment_advantage_normalization="grouped_scale_only",
        frontres_hsl_init_enabled=False,
        frontres_hsl_rollout_label_enabled=False,
        frontres_segment_critic_warmup_iterations=1,
        frontres_segment_actor_warmup_iterations=1,
        frontres_segment_k_curriculum=((8, 2, 200, 500, 1300), (16, 3, 300, 300, 900), (32, 4, 400, 300, 625)),
        frontres_segment_k_curriculum_fingerprint="",
        frontres_segment_max_horizon_k=32,
        frontres_future_offsets=(1, 2),
        frontres_future_intent_layout_version="frontres-v015-future-intent-q29-v1",
        frontres_segment_live_train_enabled=False,
        frontres_segment_live_update_loop_only=False,
        frontres_segment_live_single_update_only=False,
        frontres_method_contract_id="FRS-METHOD-v016",
        frontres_gain_contract_id="FRS-GAIN-v006",
        frontres_optimization_contract_id="FRS-PPO-v004",
        frontres_training_contract_id="FRS-TRAIN-v011",
        frontres_scalar_target_id="paired-intent-minus-repair-v1",
        frontres_constraint_schema_id="contact-loaded-phase_zmp-survival-physical-v2",
        frontres_projection_schema_id="grouped-first-order-constraint-projection-v1",
    )


def _build_request(candidate_contract, owners, live_sampler):
    gain_contract, one_action, helper, commands, hooks, setup, live_probe, storage, ppo = owners
    captured, _kwargs, _batch = candidate_contract._capture_and_build(
        gain_contract,
        one_action,
        helper,
        commands,
        hooks,
        setup,
        live_probe,
        ppo,
    )
    policy = candidate_contract._ZeroRatioPolicy()
    optimizer = _TrackingSGD(policy.parameters())
    # Default semantic transaction is the second K8 actor-ramp update. Tests
    # that exercise critic-only behavior explicitly reset this to iteration 0.
    runner = SimpleNamespace(alg=_formal_alg(policy, optimizer), current_learning_iteration=202)
    snapshot = live_sampler.capture_frontres_frozen_policy_snapshot(runner, transaction_id="tx-v015-formal-s2")
    common = {
        "transaction_id": snapshot.transaction_id,
        "policy_snapshot_id": snapshot.policy_snapshot_id,
        "motion_ids": ("motion-a", "motion-b"),
        "start_frames": torch.tensor([12, 24], dtype=torch.long),
        "segment_ids": torch.tensor([101, 202], dtype=torch.long),
        "source_index": torch.tensor([0, 1], dtype=torch.long),
    }
    attempt_zero = live_probe.build_frontres_v015_grouped_candidate_batch(
        captured.result,
        **common,
        trial_index=torch.tensor([0, 0], dtype=torch.long),
    )
    attempt_one = live_probe.build_frontres_v015_grouped_candidate_batch(
        captured.result,
        **common,
        trial_index=torch.tensor([1, 1], dtype=torch.long),
    )
    critic_dim = int(attempt_zero.privileged_observations.shape[1])
    attempt_zero = replace(
        attempt_zero,
        privileged_observations=torch.tensor([0.0, 10.0]).unsqueeze(1).expand(2, critic_dim).clone(),
    )
    attempt_one = replace(
        attempt_one,
        privileged_observations=torch.tensor([1.0, 11.0]).unsqueeze(1).expand(2, critic_dim).clone(),
    )
    attempt_zero = replace(
        attempt_zero,
        transaction_metadata=replace(
            attempt_zero.transaction_metadata,
            horizon_k=torch.full_like(attempt_zero.transaction_metadata.horizon_k, 8),
        ),
    )
    attempt_one = replace(
        attempt_one,
        transaction_metadata=replace(
            attempt_one.transaction_metadata,
            horizon_k=torch.full_like(attempt_one.transaction_metadata.horizon_k, 8),
        ),
    )
    metadata = attempt_zero.transaction_metadata
    plan = live_sampler.FrontRESV015FormalTransactionPlan(
        snapshot=snapshot,
        motion_ids=metadata.motion_ids + metadata.motion_ids,
        start_frames=torch.cat((metadata.start_frames, metadata.start_frames), dim=0),
        segment_ids=torch.cat((metadata.segment_ids, metadata.segment_ids), dim=0),
        source_index=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        trial_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        horizon_k=torch.cat((metadata.horizon_k, metadata.horizon_k), dim=0),
        scenario_ids=metadata.scenario_ids + metadata.scenario_ids,
        noisy_segment_hashes=metadata.noisy_segment_hashes + metadata.noisy_segment_hashes,
        x_t_identities=metadata.x_t_identities + metadata.x_t_identities,
        intent_q29_provenance=metadata.intent_q29_provenance,
        intent_q29_source=metadata.intent_q29_source,
    )
    request = live_probe.FrontRESV015FormalTransactionRequest(
        plan=plan,
        candidate_batches=(attempt_zero, attempt_one),
        diagnostic_reports=(
            live_probe.build_frontres_v015_local_evaluation_report(
                captured.result,
                transaction_id=plan.transaction_id,
            ),
            live_probe.build_frontres_v015_local_evaluation_report(
                captured.result,
                transaction_id=plan.transaction_id,
            ),
        ),
        curriculum_fingerprint=live_probe._v015_resolve_curriculum_identity(runner).schedule_fingerprint,
        k_stage_index=live_probe._v015_resolve_curriculum_identity(runner).stage_index,
        active_k=live_probe._v015_resolve_curriculum_identity(runner).active_k,
        active_m=live_probe._v015_resolve_curriculum_identity(runner).active_m,
        k_stage_iteration=live_probe._v015_resolve_curriculum_identity(runner).stage_iteration,
        training_iteration=live_probe._v015_resolve_curriculum_identity(runner).absolute_iteration,
        warmup_phase_name=live_probe._v015_resolve_curriculum_identity(runner).phase.name,
        warmup_actor_loss_weight=live_probe._v015_resolve_curriculum_identity(runner).phase.actor_loss_weight,
        policy_evaluator=policy,
    )
    return SimpleNamespace(
        runner=runner,
        optimizer=optimizer,
        policy=policy,
        request=request,
        storage=storage,
    )


def test_t_connect_order_exact_one_and_diagnostics(candidate_contract, owners, live_sampler, live_update_loop) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    order_accumulator = live_sampler.FrontRESV015FormalTransactionAccumulator(
        fixture.request.plan,
        optimizer_step_count=lambda: fixture.optimizer.step_count,
    )
    order_accumulator.append_candidate_batch(fixture.request.candidate_batches[1])
    order_accumulator.append_candidate_batch(fixture.request.candidate_batches[0])
    ordered = order_accumulator.seal()
    torch.testing.assert_close(
        ordered.privileged_observations[:, 0],
        torch.tensor([0.0, 1.0, 10.0, 11.0]),
    )
    provider_steps: list[int] = []

    def provider():
        provider_steps.append(fixture.optimizer.step_count)
        assert fixture.runner._frontres_v015_checkpoint_transaction_state == {
            "state": "collecting",
            "phase": "provider",
        }
        return fixture.request

    fixture.runner._frontres_v015_formal_transaction_provider = provider
    original_legacy = fixture.storage.FrontRESSegmentStorageBatch.to_ppo_batch

    def legacy_forbidden(*_args, **_kwargs):
        raise AssertionError("v015 formal transaction must not call legacy to_ppo_batch")

    fixture.storage.FrontRESSegmentStorageBatch.to_ppo_batch = legacy_forbidden
    try:
        result = live_update_loop.run_frontres_v015_formal_transaction_update_loop(fixture.runner)
    finally:
        fixture.storage.FrontRESSegmentStorageBatch.to_ppo_batch = original_legacy

    assert provider_steps == [0]
    assert fixture.optimizer.step_count == 1
    assert result.optimizer_step_before == 0
    assert result.optimizer_step_after == 1
    assert result.optimizer_step_delta == 1
    assert result.update_invocation_count == 1
    assert result.segment_count == 2
    assert result.source_count == 2
    assert result.policy_attempt_count == 4
    assert result.ppo_result.grouped_reduction_active
    assert result.ppo_result.grouped_motion_count == 2
    assert result.ppo_result.grouped_segment_count == 2
    assert result.ppo_result.grouped_attempt_count == 4
    torch.testing.assert_close(
        torch.tensor(result.ppo_result.grouped_attempt_mass_shares),
        torch.full((4,), 0.25),
    )
    assert result.diagnostics["intent_q29_provenance"] == "deployment_noisy_q29"
    assert result.diagnostics["optimizer_step_delta"] == 1
    assert result.diagnostics["method_contract_id"] == "FRS-METHOD-v016"
    assert result.diagnostics["gain_contract_id"] == "FRS-GAIN-v006"
    assert result.diagnostics["optimization_contract_id"] == "FRS-PPO-v004"
    assert result.diagnostics["training_contract_id"] == "FRS-TRAIN-v011"
    assert result.diagnostics["scalar_target_id"] == "paired-intent-minus-repair-v1"
    assert result.diagnostics["constraint_schema_id"] == "contact-loaded-phase_zmp-survival-physical-v2"
    assert result.diagnostics["projection_schema_id"] == "grouped-first-order-constraint-projection-v1"
    assert result.diagnostics["constraint_projection_status"] in {
        "INTENT_FEASIBLE", "PROJECTED_INTENT", "CONSTRAINT_RECOVERY",
        "NO_EMPIRICAL_DIRECTION", "NO_COMMON_FIRST_ORDER_DESCENT",
    }
    for name in (
        "contact_constraint_advantage",
        "zmp_constraint_advantage",
        "survival_constraint_advantage",
    ):
        values = torch.tensor(result.diagnostics[name])
        assert bool(torch.isfinite(values).all())
        torch.testing.assert_close(values.reshape(2, 2).sum(dim=1), torch.zeros(2), atol=1.0e-7, rtol=0.0)
    for name in (
        "return_mean",
        "return_min",
        "return_max",
        "return_abs_mean",
        "gradient_pre_clip_norm",
        "gradient_post_clip_norm",
    ):
        assert math.isfinite(float(result.diagnostics[name]))
    assert result.diagnostics["gradient_pre_clip_norm"] > 0.0
    assert result.diagnostics["gradient_post_clip_norm"] > 0.0
    assert result.diagnostics["gradient_parameter_count"] > 0
    assert result.diagnostics["gradient_nonzero_parameter_count"] > 0
    quality = result.diagnostics["v006_action_constraint_reports"]
    assert isinstance(quality, tuple) and len(quality) == 2
    assert all(report.transaction_id == result.transaction_id for report in quality)
    assert all(len(report.policy_actions) == 2 for report in quality)
    assert all(all(len(row) == 6 for row in report.policy_actions) for report in quality)
    assert all(report.valid_policy_row_mask == (True, True) for report in quality)
    assert all(report.scenario_ids == ("scenario-a", "scenario-b") for report in quality)
    assert all(report.noisy_segment_hashes == ("hash-a", "hash-b") for report in quality)
    assert all(report.gain_total_pos_frac + report.gain_total_neg_frac <= 1.0 for report in quality)
    assert all(len(report.contact_constraint) == 2 for report in quality)
    assert all(len(report.zmp_constraint) == 2 for report in quality)
    assert all(len(report.survival_constraint) == 2 for report in quality)
    assert fixture.optimizer.step_count == 1
    checkpoint_state = fixture.runner._frontres_v015_checkpoint_transaction_state
    assert checkpoint_state["state"] == "committed"
    assert checkpoint_state["receipt"]["optimizer_step_delta"] == 1
    assert checkpoint_state["receipt"]["collected_policy_attempt_count"] == 4
    assert "clean_continuation" not in repr(checkpoint_state["receipt"])
    runner_source = ON_POLICY_RUNNER_PATH.read_text(encoding="utf-8")
    assert "def run_frontres_v015_formal_transaction(self)" in runner_source
    assert "run_frontres_v015_formal_transaction_update_loop_helper(self)" in runner_source
    print(
        "[T-connect/T-order/T-exact-one-update/T-no-legacy-route/T-diagnostic/T-checkpoint-barrier] "
        "provider barrier seals a 2x2 transaction, then one update yields a metadata-only receipt",
        flush=True,
    )


def test_t_partial_hsl_and_legacy_config_fail_before_step(candidate_contract, owners, live_sampler, live_update_loop) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    live_probe = owners[6]
    partial = replace(
        fixture.request,
        candidate_batches=(fixture.request.candidate_batches[0],),
        diagnostic_reports=(fixture.request.diagnostic_reports[0],),
    )
    _expect_runtime_error(lambda: live_probe.run_frontres_v015_formal_transaction_update(fixture.runner, partial))
    assert fixture.optimizer.step_count == 0

    fixture = _build_request(candidate_contract, owners, live_sampler)
    fixture.runner.alg.frontres_hsl_init_enabled = True
    _expect_runtime_error(
        lambda: live_probe.run_frontres_v015_formal_transaction_update(fixture.runner, fixture.request)
    )
    assert fixture.optimizer.step_count == 0

    fixture = _build_request(candidate_contract, owners, live_sampler)
    fixture.runner.alg.frontres_segment_advantage_normalization = "scale_only"
    _expect_runtime_error(
        lambda: live_probe.run_frontres_v015_formal_transaction_update(fixture.runner, fixture.request)
    )
    assert fixture.optimizer.step_count == 0
    print(
        "[T-partial/T-warmup-isolation/T-fail-closed] partial transaction, HSL, and legacy normalization reject before step",
        flush=True,
    )


def test_t_ordinary_training_provider_uses_exact_one_owner(candidate_contract, owners, live_sampler, live_update_loop) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    fixture.runner.alg.frontres_segment_live_train_enabled = True
    fixture.runner.alg.frontres_segment_live_update_steps = 1
    fixture.runner.alg.frontres_v015_local_sentinel_only = False
    live_probe = owners[6]
    calls: list[str] = []
    original_build = live_probe.build_frontres_v015_formal_training_request
    original_close = live_probe.close_frontres_v015_formal_training_request

    def build(runner, *, init_at_random_ep_len):
        assert init_at_random_ep_len
        assert runner._frontres_v015_checkpoint_transaction_state == {"state": "collecting", "phase": "provider"}
        calls.append("provider")
        return fixture.request

    def close(runner):
        calls.append("close")
        runner._frontres_segment_live_current_sample = None
        runner._frontres_segment_live_current_batch = None

    live_probe.build_frontres_v015_formal_training_request = build
    live_probe.close_frontres_v015_formal_training_request = close
    try:
        result = live_update_loop.run_frontres_v015_formal_training_update_loop(
            fixture.runner,
            init_at_random_ep_len=True,
        )
    finally:
        live_probe.build_frontres_v015_formal_training_request = original_build
        live_probe.close_frontres_v015_formal_training_request = original_close

    assert calls == ["provider", "close"]
    assert result.optimizer_step_delta == 1
    assert fixture.optimizer.step_count == 1
    assert fixture.runner._frontres_v015_checkpoint_transaction_state["state"] == "committed"
    assert not hasattr(fixture.runner, "_frontres_v015_formal_transaction_provider")
    print(
        "[T-provider/T-complete-transaction/T-exact-one-update] ordinary provider closes only after committed exact-one update",
        flush=True,
    )


def test_t_rejected_collection_reopens_barrier_without_update(
    candidate_contract, owners, live_sampler, live_update_loop
) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    fixture.runner.alg.frontres_segment_live_train_enabled = True
    fixture.runner.alg.frontres_segment_live_update_steps = 1
    fixture.runner.alg.frontres_v015_local_sentinel_only = False
    live_probe = owners[6]
    calls: list[str] = []
    original_build = live_probe.build_frontres_v015_formal_training_request
    original_close = live_probe.close_frontres_v015_formal_training_request

    def build(runner, *, init_at_random_ep_len):
        assert init_at_random_ep_len
        assert runner._frontres_v015_checkpoint_transaction_state == {"state": "collecting", "phase": "provider"}
        calls.append("provider")
        if calls.count("provider") == 1:
            raise live_probe.FrontRESV015RejectedTransactionEvidence("invalid M-attempt evidence")
        return fixture.request

    def close(runner):
        calls.append("close")
        runner._frontres_segment_live_current_sample = None
        runner._frontres_segment_live_current_batch = None

    live_probe.build_frontres_v015_formal_training_request = build
    live_probe.close_frontres_v015_formal_training_request = close
    try:
        result = live_update_loop.run_frontres_v015_formal_training_update_loop(
            fixture.runner,
            init_at_random_ep_len=True,
        )
    finally:
        live_probe.build_frontres_v015_formal_training_request = original_build
        live_probe.close_frontres_v015_formal_training_request = original_close

    assert calls == ["provider", "provider", "close"]
    assert result.optimizer_step_delta == 1
    assert fixture.optimizer.step_count == 1
    assert fixture.runner._frontres_v015_checkpoint_transaction_state["state"] == "committed"
    print(
        "[T-reject/T-recollect/T-exact-one] invalid evidence returns the barrier to idle; accepted transaction steps once",
        flush=True,
    )


def test_t_rejected_collection_budget_fails_closed(
    candidate_contract, owners, live_sampler, live_update_loop
) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    fixture.runner.alg.frontres_segment_live_train_enabled = True
    fixture.runner.alg.frontres_segment_live_update_steps = 1
    fixture.runner.alg.frontres_v015_local_sentinel_only = False
    live_probe = owners[6]
    calls: list[str] = []
    original_build = live_probe.build_frontres_v015_formal_training_request
    original_close = live_probe.close_frontres_v015_formal_training_request
    original_budget = live_update_loop._V015_MAX_REJECTED_COLLECTIONS

    def build(_runner, *, init_at_random_ep_len):
        assert init_at_random_ep_len
        calls.append("provider")
        raise live_probe.FrontRESV015RejectedTransactionEvidence("always invalid")

    def close(runner):
        calls.append("close")
        runner._frontres_segment_live_current_sample = None
        runner._frontres_segment_live_current_batch = None

    live_probe.build_frontres_v015_formal_training_request = build
    live_probe.close_frontres_v015_formal_training_request = close
    live_update_loop._V015_MAX_REJECTED_COLLECTIONS = 1
    try:
        _expect_runtime_error(
            lambda: live_update_loop.run_frontres_v015_formal_training_update_loop(
                fixture.runner,
                init_at_random_ep_len=True,
            )
        )
    finally:
        live_update_loop._V015_MAX_REJECTED_COLLECTIONS = original_budget
        live_probe.build_frontres_v015_formal_training_request = original_build
        live_probe.close_frontres_v015_formal_training_request = original_close

    assert calls == ["provider", "provider", "close"]
    assert fixture.optimizer.step_count == 0
    assert fixture.runner._frontres_v015_checkpoint_transaction_state == {"state": "idle"}
    print("[T-reject-budget/T-zero-update] repeated invalid evidence stops bounded and persistably idle", flush=True)


def test_t_formal_training_close_releases_command_before_sampler_lifecycle(
    _candidate_contract, owners, _live_sampler, _live_update_loop
) -> None:
    live_probe = owners[6]
    events: list[str] = []
    command = SimpleNamespace(
        active=True,
        clear_frontres_local_scenario=lambda: (
            events.append("command"),
            setattr(command, "active", False),
        ),
    )
    runner = SimpleNamespace(
        env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command)),
        _frontres_v015_formal_training_batch=object(),
        _frontres_segment_live_current_sample=object(),
        _frontres_segment_live_current_batch=object(),
    )
    original_close = live_probe._close_frontres_local_scenarios
    live_probe._close_frontres_local_scenarios = lambda batch: events.append("sampler")
    try:
        live_probe.close_frontres_v015_formal_training_request(runner)
    finally:
        live_probe._close_frontres_local_scenarios = original_close

    assert events == ["command", "sampler"]
    assert command.active is False
    assert not hasattr(runner, "_frontres_v015_formal_training_batch")
    assert runner._frontres_segment_live_current_sample is None
    assert runner._frontres_segment_live_current_batch is None
    print("[T-command-close/T-next-transaction] completed request releases the sealed command carrier", flush=True)


def test_t_rejected_collection_cleanup_is_idempotent(_candidate_contract, owners, _live_sampler, _live_update_loop) -> None:
    live_probe = owners[6]
    events: list[str] = []
    command = SimpleNamespace(_frontres_local_scenario_active=torch.ones(2, dtype=torch.bool))

    def clear() -> None:
        events.append("command")
        command._frontres_local_scenario_active[:] = False

    command.clear_frontres_local_scenario = clear
    batch = SimpleNamespace(frontres_local_scenario_closed_ids=())
    runner = SimpleNamespace(
        env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command)),
        _frontres_segment_live_current_sample=object(),
        _frontres_segment_live_current_batch=batch,
        _frontres_v015_checkpoint_transaction_state={"state": "collecting", "phase": "provider"},
    )
    original_close = live_probe._close_frontres_local_scenarios

    def close_lifecycle(current_batch) -> None:
        events.append("sampler")
        current_batch.frontres_local_scenario_closed_ids = ("closed",)

    live_probe._close_frontres_local_scenarios = close_lifecycle
    try:
        live_probe.abort_frontres_v015_formal_training_collection(runner, batch=batch)
        live_probe.abort_frontres_v015_formal_training_collection(runner, batch=batch)
    finally:
        live_probe._close_frontres_local_scenarios = original_close

    assert events == ["command", "sampler"]
    assert runner._frontres_v015_checkpoint_transaction_state == {"state": "idle"}
    assert runner._frontres_segment_live_current_sample is None
    assert runner._frontres_segment_live_current_batch is None
    print("[T-reject-cleanup/T-idempotent] command, sampler, and barrier lifecycle close exactly once", flush=True)


def test_t_v011_critic_only_formal_update(candidate_contract, owners, live_sampler) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    fixture.runner.current_learning_iteration = 0
    fixture.runner.alg.value_loss_coef = 1.0
    curriculum = owners[6]._v015_resolve_curriculum_identity(fixture.runner)
    request = replace(
        fixture.request,
        curriculum_fingerprint=curriculum.schedule_fingerprint,
        k_stage_index=curriculum.stage_index,
        active_k=curriculum.active_k,
        active_m=curriculum.active_m,
        k_stage_iteration=curriculum.stage_iteration,
        training_iteration=curriculum.absolute_iteration,
        warmup_phase_name=curriculum.phase.name,
        warmup_actor_loss_weight=curriculum.phase.actor_loss_weight,
    )
    result = owners[6].run_frontres_v015_formal_transaction_update(fixture.runner, request)
    diagnostics = result.diagnostics
    assert diagnostics["training_contract_id"] == "FRS-TRAIN-v011"
    assert diagnostics["gain_contract_id"] == "FRS-GAIN-v006"
    assert diagnostics["warmup_phase"] == "critic_only"
    assert diagnostics["actor_loss_weight"] == 0.0
    assert diagnostics["actor_std_parameter_delta"]["param_delta_max_abs"] == 0.0
    assert diagnostics["critic_parameter_delta"]["param_delta_max_abs"] > 0.0
    assert result.optimizer_step_delta == 1
    print("[T-v011-critic-only] scalar Intent Critic updates exactly once and freezes actor/std", flush=True)


def test_t_v011_mixed_k_rejects(candidate_contract, owners, live_sampler) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    try:
        mixed_plan = replace(
            fixture.request.plan,
            horizon_k=torch.tensor([8, 16, 8, 8], dtype=torch.long),
        )
        replace(fixture.request, plan=mixed_plan)
    except ValueError as exc:
        assert "mixes local scenario identity" in str(exc) or "mixed-K" in str(exc)
    else:
        raise AssertionError("v011 request must reject mixed-K before update")


def test_t_checkpoint_trigger_requires_matching_commit(candidate_contract, owners, live_sampler, _live_update_loop) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    live_probe = owners[6]
    result = live_probe.run_frontres_v015_formal_transaction_update(fixture.runner, fixture.request)
    training = _load("frontres_v015_formal_live_training_contract", LIVE_TRAINING_PATH)
    saves: list[str] = []
    runner = fixture.runner
    runner.log_dir = "/tmp"
    runner.current_learning_iteration = 1
    runner.save = lambda path: saves.append(path)
    runner._record_frontres_checkpoint_probe = lambda _summary, _path: None
    summary = training._require_v015_committed_result(runner, result)
    telemetry = summary["v015_transaction_telemetry"]
    assert telemetry["policy_row_count"] == 4
    assert len(telemetry["policy_actions"]) == 4
    assert all(len(row) == 6 for row in telemetry["policy_actions"])
    assert telemetry["valid_policy_row_mask"] == (True, True, True, True)
    assert len(telemetry["intent_gain"]) == 4
    assert len(telemetry["physics_gain"]) == 4
    assert len(telemetry["repair_cost"]) == 4
    assert len(telemetry["gain_total"]) == 4
    for name in (
        "policy_values",
        "returns",
        "raw_advantages",
        "scaled_advantages",
        "repaired_survival",
        "noisy_survival",
        "physics_survival_quality_repaired",
        "physics_survival_quality_noisy",
        "repaired_zmp_margin",
        "noisy_zmp_margin",
        "repaired_contact",
        "noisy_contact",
        "physics_success_gain",
        "physics_survival_gain",
        "physics_zmp_gain",
        "physics_contact_gain",
        "physics_valid_step_count",
    ):
        assert len(telemetry[name]) == 4, name
    assert all(
        math.isclose(raw, ret - value, rel_tol=1e-6, abs_tol=1e-6)
        for raw, ret, value in zip(telemetry["raw_advantages"], telemetry["returns"], telemetry["policy_values"])
    )
    assert all(
        math.copysign(1.0, raw) == math.copysign(1.0, scaled)
        for raw, scaled in zip(telemetry["raw_advantages"], telemetry["scaled_advantages"])
        if raw != 0.0 and scaled != 0.0
    )
    assert telemetry["scenario_ids"] == (
        "scenario-a",
        "scenario-a",
        "scenario-b",
        "scenario-b",
    )
    assert telemetry["noisy_segment_hashes"] == (
        "hash-a",
        "hash-a",
        "hash-b",
        "hash-b",
    )
    assert telemetry["grouped_attempt_mass_shares"] == (0.25, 0.25, 0.25, 0.25)
    for name in (
        "return_mean",
        "return_min",
        "return_max",
        "return_abs_mean",
        "advantage_mean",
        "advantage_min",
        "advantage_max",
        "advantage_abs_mean",
        "advantage_abs_max",
        "advantage_abs_top1_frac",
        "advantage_scale",
        "grouped_transaction_advantage_rms",
        "gradient_pre_clip_norm",
        "gradient_post_clip_norm",
        "action_abs_mean",
        "action_abs_max",
        "action_l2_mean",
    ):
        assert math.isfinite(float(telemetry[name])), name
    assert telemetry["gradient_pre_clip_norm"] > 0.0
    assert telemetry["gradient_post_clip_norm"] > 0.0
    assert telemetry["gradient_parameter_count"] > 0
    assert telemetry["gradient_nonzero_parameter_count"] > 0
    assert telemetry["grouped_reduction_active"] is True
    assert telemetry["advantage_sign_flip_count"] == 0
    assert telemetry["update_count"] == 1
    assert telemetry["optimizer_step_delta"] == 1
    missing_reports = replace(result, diagnostics={**result.diagnostics, "v006_action_constraint_reports": ()})
    _expect_runtime_error(lambda: training._v015_formal_update_summary(missing_reports))
    feedback_report = replace(result.diagnostics["v006_action_constraint_reports"][0], ppo_feedback=True)
    feedback_result = replace(
        result,
        diagnostics={
            **result.diagnostics,
            "v006_action_constraint_reports": (
                feedback_report,
                *result.diagnostics["v006_action_constraint_reports"][1:],
            ),
        },
    )
    try:
        training._v015_formal_update_summary(feedback_result)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("feedback-bearing telemetry must fail closed")
    assert training._save_live_checkpoint(
        runner,
        checkpoint_path="/tmp/v015-committed.pt",
        summary=summary,
        required=True,
        expected_v015_transaction_id=result.transaction_id,
    )
    assert saves == ["/tmp/v015-committed.pt"]

    runner._frontres_v015_checkpoint_transaction_state = {"state": "collecting"}
    _expect_runtime_error(
        lambda: training._save_live_checkpoint(
            runner,
            checkpoint_path="/tmp/v015-partial.pt",
            summary=summary,
            required=True,
            expected_v015_transaction_id=result.transaction_id,
        )
    )
    assert saves == ["/tmp/v015-committed.pt"]
    print("[T-commit/T-save] checkpoint trigger accepts the matching receipt and rejects partial state", flush=True)

def test_t_formal_training_loop_never_calls_legacy_and_saves_after_commit(
    candidate_contract,
    owners,
    live_sampler,
    _live_update_loop,
) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    live_probe = owners[6]
    result = live_probe.run_frontres_v015_formal_transaction_update(fixture.runner, fixture.request)
    training = _load("frontres_v015_formal_training_loop_contract", LIVE_TRAINING_PATH)
    runner = fixture.runner
    missing_projection = dict(result.diagnostics)
    missing_projection.pop("constraint_projection_status")
    _expect_runtime_error(
        lambda: training._v015_sealed_transaction_telemetry(
            replace(result, diagnostics=missing_projection),
            ppo=runner.alg,
        )
    )
    nonfinite_kkt = dict(result.diagnostics)
    nonfinite_kkt["constraint_kkt_max_violation"] = float("nan")
    _expect_runtime_error(
        lambda: training._v015_sealed_transaction_telemetry(
            replace(result, diagnostics=nonfinite_kkt),
            ppo=runner.alg,
        )
    )
    infeasible_kkt = dict(result.diagnostics)
    infeasible_kkt["constraint_kkt_max_violation"] = 1.0e-4
    try:
        training._v015_sealed_transaction_telemetry(
            replace(result, diagnostics=infeasible_kkt),
            ppo=runner.alg,
        )
    except RuntimeError as exc:
        assert "exceeds the checkpoint-v6 constraint projection tolerance" in str(exc)
    else:
        raise AssertionError("formal telemetry must reject a postscale KKT violation")
    inconsistent_kkt = dict(result.diagnostics)
    inconsistent_kkt["constraint_kkt_max_violation"] = 0.0
    inconsistent_kkt["constraint_directional_derivatives"] = {"contact": 1.0e-4}
    try:
        training._v015_sealed_transaction_telemetry(
            replace(result, diagnostics=inconsistent_kkt),
            ppo=runner.alg,
        )
    except RuntimeError as exc:
        assert "inconsistent constraint KKT telemetry" in str(exc)
    else:
        raise AssertionError("formal telemetry must reject an inconsistent KKT projection report")
    runner.alg.frontres_segment_live_train_enabled = True
    runner._frontres_segment_replay_boundary = SimpleNamespace(
        live_train_enabled=True,
        periodic_eval_enabled=False,
    )
    runner.current_learning_iteration = 0
    runner.log_dir = "/tmp"
    runner.disable_logs = False
    runner.save_interval = 1
    calls: list[str] = []

    def formal_transaction(*, init_at_random_ep_len):
        assert init_at_random_ep_len
        calls.append("formal")
        return result

    def legacy_forbidden(*_args, **_kwargs):
        raise AssertionError("ordinary v015 training must not call the legacy update loop")

    def save(path):
        assert runner._frontres_v015_checkpoint_transaction_state["state"] == "committed"
        calls.append(f"save:{path}")

    runner.run_frontres_v015_formal_training_transaction = formal_transaction
    runner.run_frontres_segment_live_update_loop = legacy_forbidden
    runner.save = save
    runner._record_frontres_checkpoint_probe = lambda _summary, _path: None
    training.print_formal_route_audit = lambda *_args, **_kwargs: None
    output = io.StringIO()
    with redirect_stdout(output):
        training.run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    assert runner.current_learning_iteration == 1
    assert calls == ["formal", "save:/tmp/model_1.pt"]
    telemetry_line = next(
        line for line in output.getvalue().splitlines()
        if line.startswith("[FrontRES v015 Transaction Telemetry] ")
    )
    logged = json.loads(telemetry_line.split("] ", 1)[1])
    assert logged["policy_row_count"] == 4
    assert logged["optimizer_step_delta"] == 1
    assert logged["gain_source"] == "FRS-GAIN-v006-loaded-support-zmp-applicability"
    assert logged["method_contract_id"] == "FRS-METHOD-v016"
    assert logged["optimization_contract_id"] == "FRS-PPO-v004"
    assert logged["scalar_target_id"] == "paired-intent-minus-repair-v1"
    assert logged["constraint_schema_id"] == "contact-loaded-phase_zmp-survival-physical-v2"
    assert logged["projection_schema_id"] == "grouped-first-order-constraint-projection-v1"
    assert logged["constraint_projection_status"] in {
        "INTENT_FEASIBLE",
        "PROJECTED_INTENT",
        "CONSTRAINT_RECOVERY",
        "NO_EMPIRICAL_DIRECTION",
        "NO_COMMON_FIRST_ORDER_DESCENT",
    }
    assert set(logged["constraint_levels"]) == {"contact", "zmp", "survival"}
    assert set(logged["constraint_gradient_norms"]) == {"contact", "zmp", "survival"}
    assert set(logged["constraint_directional_derivatives"]) <= {"contact", "zmp", "survival"}
    assert set(logged["constraint_intent_directional_derivatives"]) <= {"contact", "zmp", "survival"}
    assert logged["constraint_kkt_max_violation"] >= 0.0
    assert len(logged["contact_constraint_advantage"]) == 4
    assert len(logged["zmp_constraint_advantage"]) == 4
    assert len(logged["survival_constraint_advantage"]) == 4
    assert len(logged["zmp_margin_repaired_steps"]) == 4
    assert len(logged["zmp_margin_noisy_steps"]) == 4
    assert len(logged["zmp_applicable_steps"]) == 4
    assert len(logged["zmp_applicable_noisy_steps"]) == 4
    assert len(logged["support_transition_steps"]) == 4
    assert len(logged["zmp_step_violation_repaired"]) == 4
    assert len(logged["zmp_step_violation_noisy"]) == 4
    assert len(logged["zmp_argmax_frame_repaired"]) == 4
    assert len(logged["zmp_argmax_frame_noisy"]) == 4
    assert len(logged["zmp_max_violation_repaired"]) == 4
    assert len(logged["zmp_max_violation_noisy"]) == 4
    assert len(logged["zmp_recovery_trajectory_repaired"]) == 4
    assert len(logged["zmp_recovery_trajectory_noisy"]) == 4
    assert logged["return_feedback"] is False
    assert logged["priority_feedback"] is False
    assert logged["ppo_feedback"] is False
    print("[T-formal-dispatch/T-legacy-isolation/T-commit-before-save] one ordinary iteration uses only formal update and saves once", flush=True)


def test_t_q29_actor_route_before_normalizer(_candidate_contract, owners, _live_sampler, _live_update_loop) -> None:
    live_probe = owners[6]
    trace: list[str] = []
    obs = torch.zeros(2, 5)
    env = SimpleNamespace(get_observations=lambda: (obs, {"observations": {}}))

    def fixed(value: torch.Tensor) -> torch.Tensor:
        trace.append("fixed")
        return value + 1.0

    def q29(value: torch.Tensor) -> torch.Tensor:
        trace.append("q29")
        return value + 2.0

    def normalize(value: torch.Tensor) -> torch.Tensor:
        trace.append("normalizer")
        return value

    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        alg=SimpleNamespace(frontres_future_offsets=(1, 2)),
        policy_obs_type=None,
        privileged_obs_type=None,
        teacher_obs_type=None,
        ref_vel_estimator_obs_type=None,
        _append_frontres_fixed_noisy_future_context=fixed,
        _append_frontres_future_intent_context=q29,
        _apply_obs_normalizer=normalize,
        privileged_obs_normalizer=lambda value: value,
        teacher_obs_normalizer=lambda value: value,
    )
    observations = live_probe._read_live_observations(runner)
    assert trace == ["q29", "normalizer"]
    torch.testing.assert_close(observations.obs, torch.full((2, 5), 2.0))
    print(
        "[T-q29-route] v015 actor observation rejects legacy fixed tail and appends q29 before normalizer",
        flush=True,
    )


def main() -> None:
    candidate_contract, owners, live_sampler, live_update_loop = _load_owners()
    test_t_connect_order_exact_one_and_diagnostics(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_partial_hsl_and_legacy_config_fail_before_step(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_ordinary_training_provider_uses_exact_one_owner(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_rejected_collection_reopens_barrier_without_update(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_rejected_collection_budget_fails_closed(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_formal_training_close_releases_command_before_sampler_lifecycle(
        candidate_contract, owners, live_sampler, live_update_loop
    )
    test_t_rejected_collection_cleanup_is_idempotent(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_v011_critic_only_formal_update(candidate_contract, owners, live_sampler)
    test_t_v011_mixed_k_rejects(candidate_contract, owners, live_sampler)
    test_t_checkpoint_trigger_requires_matching_commit(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_formal_training_loop_never_calls_legacy_and_saves_after_commit(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_q29_actor_route_before_normalizer(candidate_contract, owners, live_sampler, live_update_loop)
    print("frontres_v015_transaction_route_contract: ok", flush=True)


if __name__ == "__main__":
    main()
