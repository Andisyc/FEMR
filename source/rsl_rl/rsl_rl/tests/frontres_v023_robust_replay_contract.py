#!/usr/bin/env python3
"""TEST-24A/B/C production-boundary contracts for replay-v4."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.frontres.frontres_outer_scenario_replay import (
    FRONTRES_REPLAY_POLICY_SYMMETRIC_KL_LIMIT,
    FrontRESOuterScenarioReplay,
    FrontRESScenarioKey,
    FrontRESScenarioUtilityWindow,
)
from rsl_rl.frontres.frontres_segment_warmup import (
    FrontRESKStageSpec,
    resolve_frontres_k_stage_identity,
)


def _identity():
    schedule = (
        FrontRESKStageSpec(8, 4, 2, 2, 2, "low-k8", 0.5, "linear-coupled-v1", 4, 1.1),
    )
    return resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=0)


def _joint_identity():
    schedule = (
        FrontRESKStageSpec(8, 4, 2, 2, 2, "low-k8", 0.5, "linear-coupled-v1", 4, 1.1),
    )
    return resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=4)


def _plan(owner: FrontRESOuterScenarioReplay, transaction_id: str):
    return owner.plan(
        transaction_id=transaction_id,
        curriculum=_identity(),
        num_segments=128,
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


def _rows(first: tuple[float, ...], *, fill: float = 0.0) -> torch.Tensor:
    return torch.tensor((*first, *((fill,) * 4 * 7)), dtype=torch.float32)


def _stage(owner, plan, keys, utilities, old_values, means=None):
    means = torch.zeros(32, 6) if means is None else means
    return owner.stage(
        plan,
        keys=keys,
        utilities=utilities,
        old_values=old_values,
        policy_means=means,
        policy_sigmas=torch.ones(32, 6),
        source_index=torch.arange(8).repeat_interleave(4),
        policy_snapshot_id=f"pi-{plan.transaction_id}",
        active_m=4,
    )


def _receipt(candidate) -> dict[str, object]:
    return {
        "method_contract_id": "FRS-METHOD-v024",
        "training_contract_id": "FRS-TRAIN-v023",
        "transaction_id": candidate.transaction_id,
        "policy_snapshot_id": candidate.policy_snapshot_id,
        "optimizer_step_delta": 1,
    }


def test_first_visit_and_compatible_winsorized_mean() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=5)
    first_plan = _plan(owner, "first")
    keys = _keys(first_plan)
    first = _stage(owner, first_plan, keys, _rows((1.0, 2.0, 3.0, 4.0)), torch.zeros(32))
    assert first.critic_target_means[0] == 2.5
    before = owner.state_dict()
    try:
        owner.commit(first, receipt={**_receipt(first), "policy_snapshot_id": "wrong"})
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched receipt must not commit Replay")
    assert owner.state_dict()["records"] == before["records"]
    owner.commit(first, receipt=_receipt(first))
    committed_state = owner.state_dict()
    try:
        _plan(owner, "first")
    except RuntimeError as exc:
        assert "previously committed" in str(exc)
    else:
        raise AssertionError("duplicate transaction ID must reject before a new Replay plan")
    assert owner.state_dict()["records"] == committed_state["records"]

    first_window = next(record.utility_window for record in owner.records if record.key.digest == keys[0].digest)
    second_window, kl, reset = first_window.preview_visit(
        utilities=(5.0, 6.0, 7.0, 100.0),
        policy_mean=(0.0,) * 6,
        policy_sigma=(1.0,) * 6,
    )
    assert kl == 0.0 and not reset
    assert second_window.robust_mean == 4.5
    assert second_window.sample_count == 8
    assert second_window.compatible_visit_count == 2


def test_calibration_priority_removes_sampling_uncertainty() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=7)
    plan = _plan(owner, "priority")
    utilities = torch.tensor(
        (-10.0, -10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, *((0.0,) * 24)),
        dtype=torch.float32,
    )
    old_values = torch.tensor((*((0.0,) * 4), *((0.0,) * 4), *((0.0,) * 24)))
    candidate = _stage(owner, plan, _keys(plan), utilities, old_values)
    assert candidate.critic_target_means[0] == 0.0
    assert candidate.outcome_variances[0] > candidate.outcome_variances[1]
    assert candidate.critic_calibration_values[0] == 0.0
    assert candidate.critic_calibration_values[1] > 0.0


def test_policy_reset_bounded_window_and_v3_rejection() -> None:
    window = FrontRESScenarioUtilityWindow.from_visit(
        utilities=(0.0, 1.0, 2.0, 3.0),
        policy_mean=(0.0,) * 6,
        policy_sigma=(1.0,) * 6,
    )
    for visit in range(1, 33):
        window, kl, reset = window.preview_visit(
            utilities=(float(visit),) * 4,
            policy_mean=(0.0,) * 6,
            policy_sigma=(1.0,) * 6,
        )
        assert kl == 0.0 and not reset
    assert window.compatible_visit_count == 32
    assert window.utility_visits[0] == (1.0,) * 4

    reset_window, kl, reset = window.preview_visit(
        utilities=(9.0,) * 4,
        policy_mean=(0.2,) * 6,
        policy_sigma=(1.0,) * 6,
    )
    assert kl > FRONTRES_REPLAY_POLICY_SYMMETRIC_KL_LIMIT and reset
    assert reset_window.compatible_visit_count == 1
    assert reset_window.reset_count == 1
    assert reset_window.policy_anchor_mean == (0.2,) * 6

    owner = FrontRESOuterScenarioReplay(seed=1)
    legacy = owner.state_dict()
    legacy["schema"] = "frontres-outer-scenario-replay-v3"
    try:
        owner.load_state_dict(legacy)
    except ValueError:
        pass
    else:
        raise AssertionError("replay-v3 must reject before mutation")
    assert owner.records == ()


def test_capacity_requires_current_compatible_window_maturity() -> None:
    owner = FrontRESOuterScenarioReplay(
        global_frac=1.0,
        replay_frac=0.0,
        review_frac=0.0,
        capacity_ladder=(8, 16),
        minimum_visits_before_expand=4,
        seed=11,
    )
    first_plan = owner.plan(
        transaction_id="seed-capacity",
        curriculum=_identity(),
        num_segments=128,
        eligible=lambda _segment_id: True,
        global_family=lambda _segment_id: "local_rp",
    )
    first = _stage(owner, first_plan, _keys(first_plan), torch.zeros(32), torch.zeros(32))
    owner.commit(first, receipt=_receipt(first))

    state = owner.state_dict()
    records = list(state["records"])
    for index, record in enumerate(records):
        record = dict(record)
        record["visit_count"] = 4
        window = dict(record["utility_window"])
        window["utility_visits"] = ((0.0, 0.0, 0.0, 0.0),) * (1 if index == 0 else 4)
        window["reset_count"] = 1 if index == 0 else 0
        record["utility_window"] = window
        records[index] = record
    state["records"] = tuple(records)

    restored = FrontRESOuterScenarioReplay(
        global_frac=1.0,
        replay_frac=0.0,
        review_frac=0.0,
        capacity_ladder=(8, 16),
        minimum_visits_before_expand=4,
        seed=11,
    )
    restored.load_state_dict(state)
    plan = restored.plan(
        transaction_id="after-policy-reset",
        curriculum=_joint_identity(),
        num_segments=128,
        eligible=lambda _segment_id: True,
        global_family=lambda _segment_id: "local_rp",
    )
    assert plan.active_capacity_before == 8
    assert plan.active_capacity_after == 8
    assert restored.stats(active_k=8)["minimum_active_visits"] == 1


def main() -> None:
    test_first_visit_and_compatible_winsorized_mean()
    test_calibration_priority_removes_sampling_uncertainty()
    test_policy_reset_bounded_window_and_v3_rejection()
    test_capacity_requires_current_compatible_window_maturity()
    print("frontres_v023_robust_replay_contract: PASS", flush=True)


if __name__ == "__main__":
    main()
