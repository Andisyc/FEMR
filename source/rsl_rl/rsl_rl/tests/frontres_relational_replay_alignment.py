#!/usr/bin/env python3
"""Deterministic S1 contract for relation-only Scenario Replay."""

from __future__ import annotations

import torch

from rsl_rl.frontres.frontres_outer_scenario_replay import FrontRESScenarioKey
from rsl_rl.frontres.frontres_relational_scenario_replay import (
    FRONTRES_RELATIONAL_REPLAY_SCHEMA,
    FrontRESRelationalScenarioReplay,
)
from rsl_rl.frontres.frontres_segment_warmup import FrontRESKStageSpec, resolve_frontres_k_stage_identity


def _curriculum():
    schedule = (
        FrontRESKStageSpec(8, 4, 2, 2, 2, "low-k8", 0.5, "linear-coupled-v1", 4, 1.1),
    )
    return resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=0)


def _keys(plan):
    return tuple(
        FrontRESScenarioKey(
            motion_id=f"motion-{value.segment_id}",
            start_frame=10 + value.segment_id,
            segment_id=value.segment_id,
            x_t_identity=f"x-{value.segment_id}",
            perturbation_family=value.perturbation_family,
            perturbation_strength=value.perturbation_strength,
            perturbation_seed=value.perturbation_seed,
            noisy_segment_hash=f"noisy-{value.perturbation_seed}",
            horizon_k=8,
            future_intent_identity=f"intent-{value.segment_id}",
            planned_support_identity=f"support-{value.segment_id}",
        )
        for value in plan.selections
    )


def main() -> None:
    owner = FrontRESRelationalScenarioReplay(seed=19)
    plan = owner.plan(
        transaction_id="tx-rel-1",
        curriculum=_curriculum(),
        num_segments=256,
        eligible=lambda _value: True,
        global_family=lambda _value: "local_rp",
    )
    assert tuple(value.purpose for value in plan.selections) == (
        "admission", "edge_density", "edge_density", "edge_density",
        "edge_density", "edge_density", "edge_density", "stale_review",
    )
    source_index = torch.arange(8).repeat_interleave(4)
    edges = tuple((source * 4, source * 4 + 1) for source in range(8))
    candidate = owner.stage(
        plan,
        keys=_keys(plan),
        preference_edges=edges,
        source_index=source_index,
        policy_snapshot_id="pi-old",
        active_m=4,
    )
    assert candidate.edge_counts == (1,) * 8
    assert candidate.edge_densities == (1.0 / 6.0,) * 8
    telemetry = owner.commit(candidate, receipt={
        "method_contract_id": "FRS-METHOD-v026",
        "training_contract_id": "FRS-TRAIN-v025",
        "transaction_id": "tx-rel-1",
        "policy_snapshot_id": "pi-old",
        "optimizer_step_delta": 1,
    })
    assert telemetry["schema"] == FRONTRES_RELATIONAL_REPLAY_SCHEMA
    try:
        owner.commit(candidate, receipt={
            "method_contract_id": "FRS-METHOD-v026",
            "training_contract_id": "FRS-TRAIN-v025",
            "transaction_id": "tx-rel-1",
            "policy_snapshot_id": "pi-old",
            "optimizer_step_delta": 1,
        })
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate relational Replay commit did not fail closed")
    state = owner.state_dict()
    restored = FrontRESRelationalScenarioReplay(seed=0)
    restored.load_state_dict(state)
    assert restored.state_dict()["schema"] == FRONTRES_RELATIONAL_REPLAY_SCHEMA
    cross_owner = FrontRESRelationalScenarioReplay(seed=23)
    cross_plan = cross_owner.plan(
        transaction_id="tx-rel-cross",
        curriculum=_curriculum(),
        num_segments=256,
        eligible=lambda _value: True,
        global_family=lambda _value: "local_rp",
    )
    try:
        cross_owner.stage(
            cross_plan,
            keys=_keys(cross_plan),
            preference_edges=((0, 4),),
            source_index=source_index,
            policy_snapshot_id="pi-old-2",
            active_m=4,
        )
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("cross-Scenario preference edge did not fail closed")
    invalid_owner = FrontRESRelationalScenarioReplay(seed=29)
    invalid_plan = invalid_owner.plan(
        transaction_id="tx-rel-invalid",
        curriculum=_curriculum(),
        num_segments=256,
        eligible=lambda _value: True,
        global_family=lambda _value: "local_rp",
    )
    try:
        invalid_owner.stage(
            invalid_plan,
            keys=_keys(invalid_plan),
            preference_edges=((0, 32),),
            source_index=source_index,
            policy_snapshot_id="pi-old-3",
            active_m=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range relational Replay edge did not fail closed")
    print("frontres_relational_replay_alignment: OBJECTIVE-ALIGNED", flush=True)


if __name__ == "__main__":
    main()
