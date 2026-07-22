#!/usr/bin/env python3
"""CPU-only S2 contract for the v015 local-scenario sentinel connector."""

from __future__ import annotations

import importlib.util
from dataclasses import fields, replace
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
FORMAL_TEST = RSL_ROOT / "tests" / "frontres_v015_transaction_route_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_owners():
    formal = _load("frontres_v015_local_sentinel_formal_helper", FORMAL_TEST)
    candidate_contract, owners, live_sampler, live_update_loop = formal._load_owners()
    live_probe = owners[6]
    sys.modules["rsl_rl.runners.frontres_segment_live_update_loop"] = live_update_loop
    return formal, candidate_contract, owners, live_sampler, live_probe


def _kernel_provenance(live_sampler):
    return live_sampler._SAMPLER_MODULE._freeze_local_scenario_provenance(
        {
            "current_root_artifact_provenance": "noisy_root_artifact_t",
            "intent_q29_provenance": "deployment_noisy_q29",
            "intent_q29_source": "motion_internal_q29",
            "clean_continuation_provenance": "clean_gmt_only",
            "expected_support_provenance": "clean_gmt_physics_only",
        }
    )


def _local_batch(live_sampler) -> SimpleNamespace:
    lengths = torch.tensor([3, 2], dtype=torch.long)
    continuation = torch.arange(2 * 3 * 65, dtype=torch.float32).reshape(2, 3, 65)
    return SimpleNamespace(
        frontres_local_scenario_rows=object(),
        frontres_local_scenario_current_root_artifact_t=torch.ones(2, 7),
        frontres_local_scenario_intent_q29=torch.arange(2 * 3 * 29, dtype=torch.float32).reshape(2, 3, 29),
        frontres_local_scenario_clean_continuation=continuation,
        frontres_local_scenario_expected_support=torch.ones(2, 3, 2),
        frontres_local_scenario_clean_continuation_lengths=lengths,
        frontres_local_scenario_clean_continuation_mask=(
            torch.arange(3, dtype=torch.long).unsqueeze(0) < lengths.unsqueeze(1)
        ),
        frontres_local_scenario_ids=("scenario-0", "scenario-1"),
        frontres_local_scenario_hashes=("hash-0", "hash-1"),
        frontres_local_scenario_x_t_identities=("x-0", "x-1"),
        frontres_local_scenario_provenance=(
            _kernel_provenance(live_sampler),
            _kernel_provenance(live_sampler),
        ),
        frontres_future_offsets=(1, 2),
    )


def test_t_local_carrier_replaces_fixed_tape_on_reset_request() -> None:
    _formal, _candidate, owners, live_sampler, live_probe = _load_owners()
    request = SimpleNamespace(segment_ids=torch.tensor([101, 202], dtype=torch.long))
    live_probe._attach_frontres_local_scenario_to_index_request(request, _local_batch(live_sampler))

    assert request.frontres_local_scenario_rows is not None
    assert not hasattr(request, "frontres_fixed_noisy_tape")
    assert tuple(request.frontres_local_scenario_current_root_artifact_t.shape) == (2, 7)
    assert tuple(request.frontres_local_scenario_intent_q29.shape) == (2, 3, 29)
    assert tuple(request.frontres_local_scenario_clean_continuation.shape) == (2, 3, 65)
    assert request.frontres_local_scenario_ids == ("scenario-0", "scenario-1")
    assert request.frontres_local_scenario_hashes == ("hash-0", "hash-1")
    assert request.frontres_local_scenario_provenance[0]["intent_q29_provenance"] == "deployment_noisy_q29"
    assert request.frontres_local_scenario_provenance[0]["clean_continuation_provenance"] == "clean_gmt_only"
    assert request.frontres_future_offsets == (1, 2)
    print("[T-reset/T-provenance/T-no-tape] sealed local artifact/I/C reaches the reset request without a 65D tape", flush=True)


def test_t_sentinel_batch_materializes_local_scenario_not_legacy_tape() -> None:
    _formal, _candidate, _owners, live_sampler, _live_probe = _load_owners()
    batch = SimpleNamespace(
        segment_ids=torch.tensor([101, 202], dtype=torch.long),
        specs=(SimpleNamespace(motion_id="motion-a", start_frame=12), SimpleNamespace(motion_id="motion-b", start_frame=24)),
        perturbation_role=("policy", "policy"),
        perturbation_strength=torch.ones(2),
        stage3_index_perturbation_family=("local_rp", "local_rp"),
        frontres_segment_budget_horizon_k=torch.tensor([3, 3], dtype=torch.long),
    )
    runner = SimpleNamespace(
        _frontres_segment_dataset=SimpleNamespace(get_segments=lambda _ids: batch),
    )
    sample = SimpleNamespace(segment_ids=batch.segment_ids)
    calls: list[tuple[str, str | None]] = []
    originals = {
        name: getattr(live_sampler, name)
        for name in (
            "_attach_frontres_segment_trial_plan",
            "_build_stage3_index_perturbation_plan",
            "_attach_stage3_index_perturbation_plan",
            "_attach_fixed_noisy_scenarios",
            "_attach_frontres_local_scenarios",
        )
    }

    live_sampler._attach_frontres_segment_trial_plan = lambda current, _sample: current
    live_sampler._build_stage3_index_perturbation_plan = lambda *_args, **_kwargs: object()
    live_sampler._attach_stage3_index_perturbation_plan = lambda current, _plan: current

    def legacy_forbidden(*_args, **_kwargs):
        raise AssertionError("v015 sentinel batch must not materialize the legacy fixed tape")

    def attach_local(_runner, current, _sample, *, update_step, transaction_id=None):
        calls.append((str(update_step), transaction_id))
        current.frontres_local_scenario_rows = object()
        return current

    live_sampler._attach_fixed_noisy_scenarios = legacy_forbidden
    live_sampler._attach_frontres_local_scenarios = attach_local
    try:
        returned = live_sampler._build_current_segment_batch(
            runner,
            sample,
            update_step=7,
            print_probe=False,
            v015_local_scenario_transaction_id="tx-local-sentinel",
        )
    finally:
        for name, original in originals.items():
            setattr(live_sampler, name, original)

    assert returned is batch
    assert calls == [("7", "tx-local-sentinel")]
    assert returned.frontres_local_scenario_rows is not None
    assert not hasattr(returned, "frontres_fixed_noisy_tape")
    print("[T-materialize/T-no-tape] v015 sentinel batch selects the local-scenario materializer with its frozen transaction id", flush=True)


def test_t_sentinel_prepare_accepts_kernel_immutable_provenance() -> None:
    _formal, _candidate, _owners, live_sampler, _live_probe = _load_owners()
    policy = torch.nn.Linear(2, 1)
    base_sample = SimpleNamespace(
        segment_ids=torch.tensor([101, 202], dtype=torch.long),
        source=("global", "global"),
        priority=torch.ones(2),
        staleness=torch.zeros(2),
        valid_mask=torch.ones(2, dtype=torch.bool),
        segment_state=None,
        rollout_trial_count=torch.tensor([2, 2], dtype=torch.long),
        budget_reason=("cold_start", "cold_start"),
    )
    frozen_plan = SimpleNamespace(
        segment_ids=torch.tensor([101, 101, 202, 202], dtype=torch.long),
        source_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        trial_index=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        horizon_k=torch.tensor([3, 3, 3, 3], dtype=torch.long),
        base_trial_count=torch.tensor([2, 2], dtype=torch.long),
        trial_role=("policy", "policy", "policy", "policy"),
    )
    draw_count = 0

    def sample_one(*_args, **_kwargs):
        nonlocal draw_count
        row = draw_count % 2
        draw_count += 1
        return live_sampler.FrontRESSegmentSample(
            segment_ids=base_sample.segment_ids[row : row + 1],
            source=(base_sample.source[row],),
            priority=base_sample.priority[row : row + 1],
            staleness=base_sample.staleness[row : row + 1],
            valid_mask=base_sample.valid_mask[row : row + 1],
            segment_state=None,
            rollout_trial_count=base_sample.rollout_trial_count[row : row + 1],
            horizon_k=torch.tensor([3], dtype=torch.long),
            budget_reason=(base_sample.budget_reason[row],),
            trial_role=("policy",),
            source_index=torch.zeros(1, dtype=torch.long),
            trial_index=torch.zeros(1, dtype=torch.long),
        )

    sampler = SimpleNamespace(
        sample=sample_one,
        plan_frozen_policy_transaction=lambda *_args, **_kwargs: frozen_plan,
        num_segments=2,
    )
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            policy=policy,
            frontres_v015_local_sentinel_only=True,
            frontres_segment_max_horizon_k=3,
        ),
        env=SimpleNamespace(num_envs=8),
        _frontres_segment_sampler=sampler,
        current_learning_iteration=0,
    )
    local_batch = _local_batch(live_sampler)
    for name in (
        "frontres_local_scenario_current_root_artifact_t",
        "frontres_local_scenario_intent_q29",
        "frontres_local_scenario_clean_continuation",
        "frontres_local_scenario_clean_continuation_lengths",
        "frontres_local_scenario_clean_continuation_mask",
    ):
        value = getattr(local_batch, name)
        setattr(local_batch, name, value.repeat_interleave(2, dim=0))
    local_batch.frontres_local_scenario_ids = ("scenario-0", "scenario-0", "scenario-1", "scenario-1")
    local_batch.frontres_local_scenario_hashes = ("hash-0", "hash-0", "hash-1", "hash-1")
    local_batch.frontres_local_scenario_x_t_identities = ("x-0", "x-0", "x-1", "x-1")
    local_batch.frontres_local_scenario_provenance = (
        local_batch.frontres_local_scenario_provenance[0],
        local_batch.frontres_local_scenario_provenance[0],
        local_batch.frontres_local_scenario_provenance[1],
        local_batch.frontres_local_scenario_provenance[1],
    )
    local_batch.specs = (
        SimpleNamespace(motion_id="motion-a", start_frame=12),
        SimpleNamespace(motion_id="motion-a", start_frame=12),
        SimpleNamespace(motion_id="motion-b", start_frame=24),
        SimpleNamespace(motion_id="motion-b", start_frame=24),
    )
    original = live_sampler._build_current_segment_batch
    live_sampler._build_current_segment_batch = lambda *_args, **_kwargs: local_batch
    try:
        prepared = live_sampler.prepare_frontres_v015_local_sentinel_batch(runner)
        runner.alg.frontres_v015_local_sentinel_only = False
        runner.alg.frontres_segment_live_train_enabled = True
        formal_prepared = live_sampler.prepare_frontres_v015_formal_training_batch(runner)
    finally:
        live_sampler._build_current_segment_batch = original

    assert prepared.batch is local_batch
    assert prepared.plan.intent_q29_provenance == "deployment_noisy_q29"
    assert prepared.plan.intent_q29_source == "motion_internal_q29"
    assert formal_prepared.plan.transaction_id.startswith("frontres-v015-local-training:")
    assert formal_prepared.plan.noisy_segment_hashes == prepared.plan.noisy_segment_hashes
    assert all(isinstance(value, MappingProxyType) for value in local_batch.frontres_local_scenario_provenance)
    print("[T-immutable-provenance/T-prepare/T-formal-owner] sentinel and ordinary routes share the sealed local transaction owner", flush=True)


def test_t_formal_source_selection_fills_rows_without_partial_attempts() -> None:
    _formal, _candidate, _owners, live_sampler, _live_probe = _load_owners()
    attempts = (3, 3, 2, 6)

    def sample_batch(batch_size, **_kwargs):
        assert batch_size >= len(attempts)
        return live_sampler.FrontRESSegmentSample(
            segment_ids=torch.arange(100, 100 + len(attempts), dtype=torch.long),
            source=("global",) * len(attempts),
            priority=torch.ones(len(attempts)),
            staleness=torch.zeros(len(attempts)),
            valid_mask=torch.ones(len(attempts), dtype=torch.bool),
            segment_state=None,
            rollout_trial_count=torch.tensor(attempts, dtype=torch.long),
            horizon_k=torch.arange(4, 4 + len(attempts), dtype=torch.long),
            budget_reason=("fixture",) * len(attempts),
            trial_role=("policy",) * len(attempts),
            source_index=torch.arange(len(attempts), dtype=torch.long),
            trial_index=torch.zeros(len(attempts), dtype=torch.long),
        )

    selected = live_sampler._sample_frontres_v015_transaction_sources(
        SimpleNamespace(sample=sample_batch, num_segments=len(attempts)),
        repair_rows=8,
        max_horizon_k=8,
    )
    assert tuple(selected.segment_ids.tolist()) == (100, 101, 102)
    assert tuple(selected.rollout_trial_count.tolist()) == (3, 3, 2)
    assert int(selected.rollout_trial_count.sum().item()) == 8
    print("[T-complete-transaction/T-no-partial] exact source selection preserves whole M budgets for all Repair rows", flush=True)


def test_t_real_builder_orders_local_reset_capture_and_candidate_adapter() -> None:
    formal, candidate_contract, owners, live_sampler, live_probe = _load_owners()
    fixture = formal._build_request(candidate_contract, owners, live_sampler)
    fixture.runner.alg.frontres_v015_local_sentinel_only = True
    accumulator = live_probe.FrontRESV015FormalTransactionAccumulator(
        fixture.request.plan,
        optimizer_step_count=lambda: fixture.optimizer.step_count,
    )
    for candidate_batch in fixture.request.candidate_batches:
        accumulator.append_candidate_batch(candidate_batch)
    complete_candidate = accumulator.seal()
    local_batch = _local_batch(live_sampler)
    local_batch.frontres_local_scenario_current_root_artifact_t = local_batch.frontres_local_scenario_current_root_artifact_t.repeat_interleave(2, dim=0)
    local_batch.frontres_local_scenario_intent_q29 = local_batch.frontres_local_scenario_intent_q29.repeat_interleave(2, dim=0)
    local_batch.frontres_local_scenario_clean_continuation = local_batch.frontres_local_scenario_clean_continuation.repeat_interleave(2, dim=0)
    local_batch.frontres_local_scenario_clean_continuation_lengths = local_batch.frontres_local_scenario_clean_continuation_lengths.repeat_interleave(2)
    local_batch.frontres_local_scenario_clean_continuation_mask = local_batch.frontres_local_scenario_clean_continuation_mask.repeat_interleave(2, dim=0)
    local_batch.frontres_local_scenario_ids = ("scenario-0", "scenario-0", "scenario-1", "scenario-1")
    local_batch.frontres_local_scenario_hashes = ("hash-0", "hash-0", "hash-1", "hash-1")
    local_batch.frontres_local_scenario_x_t_identities = ("x-0", "x-0", "x-1", "x-1")
    local_batch.frontres_local_scenario_provenance = (
        local_batch.frontres_local_scenario_provenance[0],
        local_batch.frontres_local_scenario_provenance[0],
        local_batch.frontres_local_scenario_provenance[1],
        local_batch.frontres_local_scenario_provenance[1],
    )
    trace: list[str] = []
    originals = {
        "prepare": live_probe.prepare_frontres_v015_local_sentinel_batch,
        "layout": live_probe.configure_frontres_pair_layout,
        "reset": live_probe._apply_current_segment_reset,
        "observations": live_probe._read_live_observations,
        "capture": live_probe.collect_frontres_v015_gain_return_priority_evidence,
        "candidate": live_probe.build_frontres_v015_grouped_candidate_batch,
        "diagnostics": live_probe.build_frontres_v015_local_evaluation_report,
    }

    def prepare(_runner):
        trace.append("prepare")
        return SimpleNamespace(sample=SimpleNamespace(), batch=local_batch, plan=fixture.request.plan)

    def reset(runner, *, pair_layout):
        trace.append("reset")
        assert runner._frontres_segment_live_current_batch is local_batch
        assert pair_layout.n_train == pair_layout.n_base == 4
        assert local_batch.frontres_local_scenario_rows is not None
        return SimpleNamespace(success_mask=torch.ones(4, dtype=torch.bool))

    def observations(_runner):
        trace.append("observations")
        _runner._frontres_v015_observation_route_trace = {
            "role_row_count": 8,
            "current_command_dim": 58,
            "raw_observation_dim": 870,
            "q29_tail_dim": 58,
            "combined_observation_dim": 928,
            "normalized_observation_dim": 928,
            "femr_visible_dim": 158,
            "gmt_suffix_dim": 770,
            "gmt_input_dim": 770,
            "critic_observation_dim": 289,
            "post_advance_gmt_read_count": 2,
        }
        return object()

    def capture(_runner, _observations, *, pair_layout):
        trace.append("capture")
        assert pair_layout.n_train == pair_layout.n_base == 4
        return SimpleNamespace(one_action=SimpleNamespace(actor_forward_count=1, later_femr_action_count=0))

    def candidate(_evidence, **kwargs):
        trace.append("candidate")
        assert kwargs["transaction_id"] == fixture.request.plan.transaction_id
        assert kwargs["policy_snapshot_id"] == fixture.request.plan.policy_snapshot_id
        return complete_candidate

    live_probe.prepare_frontres_v015_local_sentinel_batch = prepare
    live_probe.configure_frontres_pair_layout = lambda _runner, **_kwargs: SimpleNamespace(
        n_train=4, n_base=4, n_candidate=0, n_clean=0
    )
    live_probe._apply_current_segment_reset = reset
    live_probe._read_live_observations = observations
    live_probe.collect_frontres_v015_gain_return_priority_evidence = capture
    live_probe.build_frontres_v015_grouped_candidate_batch = candidate
    report_a, report_b = fixture.request.diagnostic_reports
    row_updates = {
        field.name: getattr(report_a, field.name) + getattr(report_b, field.name)
        for field in fields(report_a)
        if isinstance(getattr(report_a, field.name), tuple)
        and len(getattr(report_a, field.name)) == report_a.policy_row_count
        and field.name not in {"scenario_ids", "noisy_segment_hashes"}
    }
    combined_gain = report_a.gain_total + report_b.gain_total
    combined_report = replace(
        report_a,
        **row_updates,
        scenario_ids=tuple(complete_candidate.transaction_metadata.scenario_ids),
        noisy_segment_hashes=tuple(complete_candidate.transaction_metadata.noisy_segment_hashes),
        policy_row_count=report_a.policy_row_count + report_b.policy_row_count,
        valid_policy_row_count=report_a.valid_policy_row_count + report_b.valid_policy_row_count,
        intent_gain_mean=sum(report_a.intent_gain + report_b.intent_gain) / 4,
        physics_gain_mean=sum(report_a.physics_gain + report_b.physics_gain) / 4,
        repair_cost_mean=sum(report_a.repair_cost + report_b.repair_cost) / 4,
        gain_total_mean=sum(combined_gain) / 4,
        gain_total_pos_frac=sum(value > 0 for value in combined_gain) / 4,
        gain_total_neg_frac=sum(value < 0 for value in combined_gain) / 4,
    )
    live_probe.build_frontres_v015_local_evaluation_report = lambda _evidence, *, transaction_id: combined_report
    try:
        request = live_probe._build_frontres_v015_local_identity_sentinel_request(
            fixture.runner,
            init_at_random_ep_len=True,
        )
    finally:
        live_probe.prepare_frontres_v015_local_sentinel_batch = originals["prepare"]
        live_probe.configure_frontres_pair_layout = originals["layout"]
        live_probe._apply_current_segment_reset = originals["reset"]
        live_probe._read_live_observations = originals["observations"]
        live_probe.collect_frontres_v015_gain_return_priority_evidence = originals["capture"]
        live_probe.build_frontres_v015_grouped_candidate_batch = originals["candidate"]
        live_probe.build_frontres_v015_local_evaluation_report = originals["diagnostics"]

    assert trace == ["prepare", "reset", "observations", "capture", "candidate"]
    assert request.plan is fixture.request.plan
    assert request.candidate_batches == (complete_candidate,)
    assert fixture.optimizer.step_count == 0
    print("[T-builder/T-reset/T-capture/T-candidate] real builder orders local setup before candidate collection and performs no optimizer step", flush=True)


def test_t_sentinel_provider_is_collected_before_one_grouped_update() -> None:
    formal, candidate_contract, owners, _sampler, live_probe = _load_owners()
    fixture = formal._build_request(candidate_contract, owners, sys.modules["rsl_rl.runners.frontres_segment_live_sampler"])
    fixture.runner.alg.frontres_v015_local_sentinel_only = True
    calls: list[str] = []

    def provider_builder(runner, *, init_at_random_ep_len):
        assert init_at_random_ep_len
        assert runner._frontres_v015_checkpoint_transaction_state == {"state": "collecting", "phase": "provider"}
        calls.append("provider")
        plan = fixture.request.plan
        runner._frontres_v015_local_sentinel_preupdate_diagnostics = {
            "transaction_id": plan.transaction_id,
            "policy_snapshot_id": plan.policy_snapshot_id,
            "x_t_identities": tuple(plan.x_t_identities),
            "scenario_ids": tuple(plan.scenario_ids),
            "noisy_segment_hashes": tuple(plan.noisy_segment_hashes),
            "root_artifact_l2": (0.1, 0.1, 0.2, 0.2),
            "intent_q29_provenance": plan.intent_q29_provenance,
            "intent_q29_source": plan.intent_q29_source,
            "clean_continuation_lengths": tuple(int(value) for value in plan.horizon_k.tolist()),
            "roles": ("repair",) * 4 + ("noisy",) * 4,
            "policy_row_count": 4,
            "actor_forward_count": 1,
            "later_femr_action_count": 0,
            "horizon_k": tuple(int(value) for value in plan.horizon_k.tolist()),
            "observation_route": {
                "role_row_count": 8,
                "current_command_dim": 58,
                "raw_observation_dim": 870,
                "q29_tail_dim": 58,
                "combined_observation_dim": 928,
                "normalized_observation_dim": 928,
                "femr_visible_dim": 158,
                "gmt_suffix_dim": 770,
                "gmt_input_dim": 770,
                "post_advance_gmt_read_count": 2,
            },
        }
        return fixture.request

    original = live_probe._build_frontres_v015_local_identity_sentinel_request
    live_probe._build_frontres_v015_local_identity_sentinel_request = provider_builder
    original_legacy = fixture.storage.FrontRESSegmentStorageBatch.to_ppo_batch

    def legacy_forbidden(*_args, **_kwargs):
        raise AssertionError("v015 local sentinel must not call legacy to_ppo_batch")

    fixture.storage.FrontRESSegmentStorageBatch.to_ppo_batch = legacy_forbidden
    try:
        result = live_probe.run_frontres_v015_local_identity_sentinel(
            fixture.runner,
            init_at_random_ep_len=True,
        )
    finally:
        live_probe._build_frontres_v015_local_identity_sentinel_request = original
        fixture.storage.FrontRESSegmentStorageBatch.to_ppo_batch = original_legacy

    assert calls == ["provider"]
    assert fixture.optimizer.step_count == 1
    assert result.optimizer_step_delta == 1
    assert result.policy_attempt_count == 4
    assert result.diagnostics["grouped_attempt_mass_shares"] == (0.25, 0.25, 0.25, 0.25)
    telemetry = fixture.runner._frontres_v015_local_sentinel_telemetry
    assert telemetry["observation_route"]["raw_observation_dim"] == 870
    assert telemetry["observation_route"]["femr_visible_dim"] == 158
    assert telemetry["observation_route"]["gmt_input_dim"] == 770
    assert telemetry["exact_one_update"] is True
    assert not hasattr(fixture.runner, "_frontres_v015_formal_transaction_provider")
    print("[T-connect/T-order/T-live-snapshot/T-mass/T-exact-one-update] provider emits the complete 870/58/928/158/770 snapshot before one grouped v003 update", flush=True)


def main() -> None:
    test_t_local_carrier_replaces_fixed_tape_on_reset_request()
    test_t_sentinel_batch_materializes_local_scenario_not_legacy_tape()
    test_t_sentinel_prepare_accepts_kernel_immutable_provenance()
    test_t_formal_source_selection_fills_rows_without_partial_attempts()
    test_t_real_builder_orders_local_reset_capture_and_candidate_adapter()
    test_t_sentinel_provider_is_collected_before_one_grouped_update()
    print("frontres_v015_local_sentinel_connectivity_contract: ok", flush=True)


if __name__ == "__main__":
    main()
