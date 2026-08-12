#!/usr/bin/env python3
"""TEST-22B/C semantic pseudo-samples for TRAIN-v022 outer replay."""

from __future__ import annotations

# Historical replay-v3 evidence. Active replay-v5 coverage lives in
# frontres_v023_robust_replay_contract.py.
__test__ = False

from dataclasses import replace
from pathlib import Path
import sys

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.frontres.frontres_outer_scenario_replay import (
    FrontRESOuterScenarioReplay,
    FrontRESScenarioKey,
    FrontRESScenarioReplayRecord,
)
from rsl_rl.frontres.frontres_segment_warmup import (
    FrontRESKStageSpec,
    frontres_v021_dr_strength_in_class,
    resolve_frontres_k_stage_identity,
)


def _identity(iteration: int):
    schedule = (
        FrontRESKStageSpec(
            8,
            4,
            2,
            2,
            2,
            "low-k8",
            0.5,
            "linear-coupled-v1",
            4,
            1.1,
        ),
    )
    return resolve_frontres_k_stage_identity(
        schedule=schedule,
        committed_update_iteration=iteration,
    )


def _key(selection, *, suffix: str) -> FrontRESScenarioKey:
    return FrontRESScenarioKey(
        motion_id=f"motion-{selection.segment_id}",
        start_frame=10 + selection.segment_id,
        segment_id=selection.segment_id,
        x_t_identity=f"x-{selection.segment_id}",
        perturbation_family=selection.perturbation_family,
        perturbation_strength=selection.perturbation_strength,
        perturbation_seed=selection.perturbation_seed,
        noisy_segment_hash=f"noisy-{selection.perturbation_seed}-{suffix}",
        horizon_k=8,
        future_intent_identity=f"intent-{selection.segment_id}",
        planned_support_identity=f"support-{selection.segment_id}",
    )


def _plan(owner: FrontRESOuterScenarioReplay, transaction_id: str, *, iteration: int = 0):
    identity = _identity(iteration)
    return owner.plan(
        transaction_id=transaction_id,
        curriculum=identity,
        num_segments=64,
        eligible=lambda _segment_id: True,
        global_family=lambda _segment_id: "local_rp",
    )


def _receipt(plan, *, policy_snapshot_id: str) -> dict[str, object]:
    return {
        "method_contract_id": "FRS-METHOD-v023",
        "training_contract_id": "FRS-TRAIN-v022",
        "transaction_id": plan.transaction_id,
        "policy_snapshot_id": policy_snapshot_id,
        "optimizer_step_delta": 1,
    }


def test_phase_scores_are_independent_hand_computed_values() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=5)
    plan = _plan(owner, "tx-dual-score")
    assert plan.phase_name == "low_dr_joint_init"
    assert plan.score_kind == "critic_calibration"
    keys = tuple(_key(selection, suffix=str(index)) for index, selection in enumerate(plan.selections))
    advantages = torch.tensor(
        [3.0, 1.0, -1.0, -3.0, 5.0, 5.0, 5.0, 5.0]
        + [float(source) for source in range(2, 8) for _ in range(4)]
    )
    candidate = owner.stage(
        plan,
        keys=keys,
        actor_advantages=advantages,
        source_index=torch.arange(8).repeat_interleave(4),
        policy_snapshot_id="pi-dual-score",
        active_m=4,
    )
    assert candidate.critic_calibration_values == (0.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    assert candidate.repair_spread_values == (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    telemetry = owner.commit(candidate, receipt=_receipt(plan, policy_snapshot_id="pi-dual-score"))
    assert telemetry["critic_calibration_values"] == candidate.critic_calibration_values
    assert telemetry["repair_spread_values"] == candidate.repair_spread_values
    assert len(owner.records) == 8

    joint_plan = _plan(owner, "tx-joint-score", iteration=4)
    assert joint_plan.phase_name == "joint"
    assert joint_plan.score_kind == "repair_spread"


def test_current_dr_interval_overrides_stored_class_label() -> None:
    identity = _identity(0)
    assert not frontres_v021_dr_strength_in_class(identity, class_name="easy", strength=0.6)
    key = FrontRESScenarioKey(
        motion_id="foreign-strength",
        start_frame=1,
        segment_id=1,
        x_t_identity="x-foreign",
        perturbation_family="local_rp",
        perturbation_strength=0.6,
        perturbation_seed=7,
        noisy_segment_hash="noisy-foreign",
        horizon_k=8,
        future_intent_identity="intent-foreign",
        planned_support_identity="support-foreign",
    )
    record = FrontRESScenarioReplayRecord(
        key=key,
        dr_class="easy",
        critic_calibration_score_by_k=((8, 4.0),),
        repair_spread_score_by_k=((8, 3.0),),
        staleness=9,
        visit_count=1,
        last_transaction_id="tx-old",
    )
    owner = FrontRESOuterScenarioReplay(global_frac=0.0, replay_frac=1.0, review_frac=0.0, seed=3)
    state = owner.state_dict()
    state["records"] = (record.to_state(),)
    owner.load_state_dict(state)
    plan = _plan(owner, "tx-compatible-only")
    assert all(selection.source == "global" for selection in plan.selections)
    assert all(
        frontres_v021_dr_strength_in_class(
            identity,
            class_name=selection.dr_class,
            strength=selection.perturbation_strength,
        )
        for selection in plan.selections
    )


def test_replay_v3_roundtrip_rejects_legacy_schema() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=17)
    plan = _plan(owner, "tx-roundtrip")
    keys = tuple(_key(selection, suffix=str(index)) for index, selection in enumerate(plan.selections))
    candidate = owner.stage(
        plan,
        keys=keys,
        actor_advantages=torch.tensor(
            [1.0, 2.0, 3.0, 4.0, -4.0, -2.0, 0.0, 2.0]
            + [float(source) for source in range(2, 8) for _ in range(4)]
        ),
        source_index=torch.arange(8).repeat_interleave(4),
        policy_snapshot_id="pi-roundtrip",
        active_m=4,
    )
    owner.commit(candidate, receipt=_receipt(plan, policy_snapshot_id="pi-roundtrip"))
    state = owner.state_dict()
    restored = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=99)
    restored.load_state_dict(state)
    assert restored.state_dict()["schema"] == "frontres-outer-scenario-replay-v3"
    assert tuple(record.to_state() for record in restored.records) == state["records"]

    legacy = dict(state)
    legacy["schema"] = "frontres-outer-scenario-replay-v2"
    legacy_record = dict(legacy["records"][0])
    legacy_record["score_by_k"] = legacy_record.pop("critic_calibration_score_by_k")
    legacy_record.pop("repair_spread_score_by_k")
    legacy["records"] = (legacy_record, *legacy["records"][1:])
    before = tuple(record.to_state() for record in restored.records)
    try:
        restored.load_state_dict(legacy)
    except ValueError:
        pass
    else:
        raise AssertionError("single-score replay-v1 state must reject")
    assert tuple(record.to_state() for record in restored.records) == before


def main() -> None:
    test_phase_scores_are_independent_hand_computed_values()
    test_current_dr_interval_overrides_stored_class_label()
    test_replay_v3_roundtrip_rejects_legacy_schema()
    print("frontres_v021_coupled_replay_contract: PASS")


if __name__ == "__main__":
    main()
