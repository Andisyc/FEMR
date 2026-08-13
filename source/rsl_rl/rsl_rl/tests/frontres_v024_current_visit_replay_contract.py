#!/usr/bin/env python3
"""TEST-25A/B/C production-boundary contracts for replay-v5."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.frontres.frontres_outer_scenario_replay import (
    FRONTRES_OUTER_REPLAY_SCHEMA,
    FrontRESOuterScenarioReplay,
    FrontRESScenarioKey,
)
from rsl_rl.frontres.frontres_segment_warmup import (
    FrontRESKStageSpec,
    resolve_frontres_k_stage_identity,
)


def _identity(*, joint: bool = False):
    schedule = (
        FrontRESKStageSpec(8, 4, 2, 2, 2, "low-k8", 0.5, "linear-coupled-v1", 4, 1.1),
    )
    return resolve_frontres_k_stage_identity(
        schedule=schedule,
        committed_update_iteration=4 if joint else 0,
    )


def _plan(owner: FrontRESOuterScenarioReplay, transaction_id: str, *, joint: bool = False):
    return owner.plan(
        transaction_id=transaction_id,
        curriculum=_identity(joint=joint),
        num_segments=256,
        eligible=lambda _segment_id: True,
        global_family=lambda _segment_id: "local_rp",
    )


def _keys(plan) -> tuple[FrontRESScenarioKey, ...]:
    return tuple(
        FrontRESScenarioKey(
            motion_id=f"motion-{selection.segment_id}",
            start_frame=selection.segment_id + 10,
            segment_id=selection.segment_id,
            x_t_identity=f"x-{selection.segment_id}",
            perturbation_family=selection.perturbation_family,
            perturbation_strength=selection.perturbation_strength,
            perturbation_seed=selection.perturbation_seed,
            noisy_segment_hash=f"noisy-{selection.perturbation_seed}",
            horizon_k=8,
            future_intent_identity=f"intent-{selection.segment_id}",
            planned_support_identity=f"support-{selection.segment_id}",
        )
        for selection in plan.selections
    )


def _stage(owner, plan, utilities, old_values, *, source_index=None):
    sources = torch.arange(8).repeat_interleave(4) if source_index is None else source_index
    return owner.stage(
        plan,
        keys=_keys(plan),
        utilities=utilities,
        old_values=old_values,
        source_index=sources,
        policy_snapshot_id=f"pi-{plan.transaction_id}",
        active_m=4,
    )


def _receipt(candidate) -> dict[str, object]:
    return {
        "method_contract_id": "FRS-METHOD-v025",
        "training_contract_id": "FRS-TRAIN-v024",
        "transaction_id": candidate.transaction_id,
        "policy_snapshot_id": candidate.policy_snapshot_id,
        "optimizer_step_delta": 1,
    }


def _assert_no_historical_outcomes(value: object) -> None:
    forbidden = ("utility_window", "utility_visits", "policy_anchor", "policy_kl", "window_reset")
    if isinstance(value, dict):
        for key, child in value.items():
            assert not any(token in str(key) for token in forbidden), key
            _assert_no_historical_outcomes(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_no_historical_outcomes(child)


def _assert_state_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_state_equal(actual_value, expected_value)
    else:
        assert actual == expected


def test_replayed_scenario_target_uses_only_the_fresh_current_m4() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=0.0, replay_frac=1.0, review_frac=0.0, seed=5)
    first_plan = _plan(owner, "first")
    first = _stage(owner, first_plan, torch.zeros(32), torch.zeros(32))
    owner.commit(first, receipt=_receipt(first))

    second_plan = _plan(owner, "second")
    assert any(selection.replay_key_digest is not None for selection in second_plan.selections)
    second = _stage(owner, second_plan, torch.full((32,), 9.0), torch.zeros(32))
    assert second.critic_target_means == (9.0,) * 8
    assert second.current_utility_means == second.critic_target_means


def test_current_m4_uncertainty_changes_priority_but_not_target() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=7)
    plan = _plan(owner, "priority")
    utilities = torch.tensor(
        (-10.0, -10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, *((0.0,) * 24)),
        dtype=torch.float32,
    )
    candidate = _stage(owner, plan, utilities, torch.zeros(32))
    assert candidate.critic_target_means[:2] == (0.0, 1.0)
    assert candidate.outcome_variances[0] > candidate.outcome_variances[1]
    assert candidate.critic_calibration_values[0] == 0.0
    assert candidate.critic_calibration_values[1] == 1.0
    assert candidate.current_sample_counts == (4,) * 8


def test_replay_v5_persists_selection_metadata_without_outcome_history() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=11)
    plan = _plan(owner, "persist")
    candidate = _stage(owner, plan, torch.arange(32, dtype=torch.float32), torch.zeros(32))
    before = owner.state_dict()
    try:
        owner.commit(candidate, receipt={**_receipt(candidate), "policy_snapshot_id": "wrong"})
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched receipt must fail closed")
    _assert_state_equal(owner.state_dict(), before)

    owner.commit(candidate, receipt=_receipt(candidate))
    state = owner.state_dict()
    assert state["schema"] == "frontres-outer-scenario-replay-v5" == FRONTRES_OUTER_REPLAY_SCHEMA
    _assert_no_historical_outcomes(state)

    restored = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=11)
    restored.load_state_dict(state)
    assert restored.state_dict()["records"] == state["records"]
    legacy = dict(state)
    legacy["schema"] = "frontres-outer-scenario-replay-v4"
    snapshot = restored.state_dict()
    try:
        restored.load_state_dict(legacy)
    except ValueError:
        pass
    else:
        raise AssertionError("replay-v4 must reject before mutation")
    _assert_state_equal(restored.state_dict(), snapshot)


def test_capacity_maturity_uses_committed_visit_count() -> None:
    owner = FrontRESOuterScenarioReplay(
        global_frac=1.0,
        replay_frac=0.0,
        review_frac=0.0,
        capacity_ladder=(8, 16),
        minimum_visits_before_expand=4,
        seed=13,
    )
    plan = _plan(owner, "seed-capacity")
    candidate = _stage(owner, plan, torch.zeros(32), torch.zeros(32))
    owner.commit(candidate, receipt=_receipt(candidate))
    state = owner.state_dict()
    records = []
    for record in state["records"]:
        updated = dict(record)
        updated["visit_count"] = 4
        records.append(updated)
    state["records"] = tuple(records)

    restored = FrontRESOuterScenarioReplay(
        global_frac=1.0,
        replay_frac=0.0,
        review_frac=0.0,
        capacity_ladder=(8, 16),
        minimum_visits_before_expand=4,
        seed=13,
    )
    restored.load_state_dict(state)
    next_plan = _plan(restored, "expand", joint=True)
    assert next_plan.active_capacity_before == 8
    assert next_plan.active_capacity_after == 16
    assert restored.stats(active_k=8)["minimum_active_visits"] == 4


def test_full_pool_seals_one_replacement_before_current_evidence() -> None:
    owner = FrontRESOuterScenarioReplay(
        capacity_ladder=(8, 16),
        minimum_visits_before_expand=1,
        seed=19,
    )
    transaction = 0
    while int(owner.stats(active_k=8)["active_count"]) < 16:
        plan = _plan(owner, f"fill-{transaction}", joint=True)
        candidate = _stage(owner, plan, torch.zeros(32), torch.zeros(32))
        owner.commit(candidate, receipt=_receipt(candidate))
        transaction += 1
        assert transaction <= 9

    plan = _plan(owner, "replace-full", joint=True)
    assert plan.active_capacity_before == plan.active_capacity_after == 16
    assert plan.replacement_digest is not None
    replacement = next(record for record in owner.records if record.key.digest == plan.replacement_digest)
    assert replacement.key.segment_id not in {selection.segment_id for selection in plan.selections}

    candidate = _stage(owner, plan, torch.arange(32, dtype=torch.float32), torch.zeros(32))
    active = dict(candidate.active_digests_by_k)[8]
    assert len(active) == 16
    assert plan.replacement_digest not in active
    assert plan.replacement_digest in {record.key.digest for record in candidate.records}
    owner.commit(candidate, receipt=_receipt(candidate))
    assert int(owner.stats(active_k=8)["active_count"]) == 16


def test_row_permutation_preserves_scenario_targets() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=17)
    plan = _plan(owner, "ordered")
    utilities = torch.arange(32, dtype=torch.float32)
    old_values = torch.arange(8, dtype=torch.float32).repeat_interleave(4)
    ordered = _stage(owner, plan, utilities, old_values)

    permutation = torch.tensor((7, 0, 21, 15, 31, 2, 12, 24, 5, 18, 9, 29, 1, 23, 16, 10,
                                6, 27, 3, 20, 14, 30, 8, 25, 4, 19, 11, 28, 13, 22, 17, 26))
    sources = torch.arange(8).repeat_interleave(4)
    permuted = _stage(
        owner,
        plan,
        utilities[permutation],
        old_values[permutation],
        source_index=sources[permutation],
    )
    assert permuted.critic_target_means == ordered.critic_target_means


def main() -> None:
    test_replayed_scenario_target_uses_only_the_fresh_current_m4()
    test_current_m4_uncertainty_changes_priority_but_not_target()
    test_replay_v5_persists_selection_metadata_without_outcome_history()
    test_capacity_maturity_uses_committed_visit_count()
    test_full_pool_seals_one_replacement_before_current_evidence()
    test_row_permutation_preserves_scenario_targets()
    print("frontres_v024_current_visit_replay_contract: PASS", flush=True)


if __name__ == "__main__":
    main()
