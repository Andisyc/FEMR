#!/usr/bin/env python3
"""TEST-21A-D semantic contracts for committed outer Scenario replay."""

from __future__ import annotations

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
    isolated_frontres_perturbation_rng,
)


def _descriptor(segment_id: int, seed: int) -> tuple[str, float, str]:
    return "local_rp", 0.5 + 0.01 * segment_id, f"class-{seed % 3}"


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


def _receipt(plan, *, policy_snapshot_id: str) -> dict[str, object]:
    return {
        "method_contract_id": "FRS-METHOD-v021",
        "training_contract_id": "FRS-TRAIN-v020",
        "transaction_id": plan.transaction_id,
        "policy_snapshot_id": policy_snapshot_id,
        "optimizer_step_delta": 1,
    }


def test_seeded_identity_and_rng_isolation() -> None:
    torch.manual_seed(91)
    state_before = torch.random.get_rng_state().clone()
    with isolated_frontres_perturbation_rng(7, device="cpu"):
        first = torch.randn(6)
    torch.testing.assert_close(torch.random.get_rng_state(), state_before, rtol=0.0, atol=0.0)
    with isolated_frontres_perturbation_rng(7, device="cpu"):
        second = torch.randn(6)
    with isolated_frontres_perturbation_rng(8, device="cpu"):
        different = torch.randn(6)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.equal(first, different)

    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=3)
    plan = owner.plan(
        transaction_id="tx-identity",
        active_k=8,
        num_segments=20,
        eligible=lambda segment_id: segment_id % 2 == 0,
        global_descriptor=_descriptor,
    )
    key = _key(plan.selections[0], suffix="same")
    assert key.digest == FrontRESScenarioKey.from_state(key.to_state()).digest
    assert key.digest != replace(key, perturbation_seed=key.perturbation_seed + 1).digest


def test_negative_advantage_priority_and_committed_only_mutation() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=5)
    plan = owner.plan(
        transaction_id="tx-negative",
        active_k=8,
        num_segments=32,
        eligible=lambda _segment_id: True,
        global_descriptor=_descriptor,
    )
    keys = tuple(_key(selection, suffix=str(index)) for index, selection in enumerate(plan.selections))
    advantages = torch.tensor([-4.0, -2.0, -1.0, -3.0, -8.0, -6.0, -2.0, -4.0])
    source_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    candidate = owner.stage(
        plan,
        keys=keys,
        actor_advantages=advantages,
        source_index=source_index,
        policy_snapshot_id="pi-negative",
        active_m=4,
    )
    assert candidate.learning_values == (2.5, 5.0)
    before = owner.state_dict()
    try:
        owner.commit(candidate, receipt={**_receipt(plan, policy_snapshot_id="wrong"), "policy_snapshot_id": "wrong"})
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched receipt must fail")
    assert owner.records == ()
    torch.testing.assert_close(
        owner.state_dict()["generator_state"], before["generator_state"], rtol=0.0, atol=0.0
    )

    telemetry = owner.commit(candidate, receipt=_receipt(plan, policy_snapshot_id="pi-negative"))
    assert telemetry["state_delta"] == 1
    assert tuple(record.score_for_k(8) for record in owner.records) == (5.0, 2.5) or tuple(
        record.score_for_k(8) for record in owner.records
    ) == (2.5, 5.0)
    committed = owner.state_dict()
    try:
        owner.commit(candidate, receipt=_receipt(plan, policy_snapshot_id="pi-negative"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate receipt must fail")
    assert tuple(record.to_state() for record in owner.records) == committed["records"]
    torch.testing.assert_close(
        owner.state_dict()["generator_state"], committed["generator_state"], rtol=0.0, atol=0.0
    )


def test_replay_selection_k_isolation_and_key_counterexample() -> None:
    seed_owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=11)
    seed_plan = seed_owner.plan(
        transaction_id="tx-seed",
        active_k=8,
        num_segments=64,
        eligible=lambda _segment_id: True,
        global_descriptor=_descriptor,
    )
    seed_keys = tuple(_key(selection, suffix=str(index)) for index, selection in enumerate(seed_plan.selections))
    seed_candidate = seed_owner.stage(
        seed_plan,
        keys=seed_keys,
        actor_advantages=torch.tensor([3.0] * 4 + [1.0] * 4),
        source_index=torch.tensor([0] * 4 + [1] * 4),
        policy_snapshot_id="pi-seed",
        active_m=4,
    )
    seed_owner.commit(seed_candidate, receipt=_receipt(seed_plan, policy_snapshot_id="pi-seed"))

    state = seed_owner.state_dict()
    replay_owner = FrontRESOuterScenarioReplay(global_frac=0.0, replay_frac=1.0, review_frac=0.0, seed=0)
    replay_state = dict(state)
    replay_state["fractions"] = (0.0, 1.0, 0.0)
    replay_owner.load_state_dict(replay_state)
    replay_plan = replay_owner.plan(
        transaction_id="tx-replay",
        active_k=8,
        num_segments=64,
        eligible=lambda _segment_id: True,
        global_descriptor=_descriptor,
    )
    assert tuple(selection.source for selection in replay_plan.selections) == ("replay", "replay")
    replay_keys = tuple(
        next(record.key for record in replay_owner.records if record.key.digest == selection.replay_key_digest)
        for selection in replay_plan.selections
    )
    wrong = (replace(replay_keys[0], noisy_segment_hash="wrong"), replay_keys[1])
    try:
        replay_owner.stage(
            replay_plan,
            keys=wrong,
            actor_advantages=torch.ones(8),
            source_index=torch.tensor([0] * 4 + [1] * 4),
            policy_snapshot_id="pi-replay",
            active_m=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("changed replay key must fail")

    k16_plan = replay_owner.plan(
        transaction_id="tx-k16",
        active_k=16,
        num_segments=64,
        eligible=lambda _segment_id: True,
        global_descriptor=_descriptor,
    )
    assert tuple(selection.source for selection in k16_plan.selections) == ("global", "global")


def test_strict_persistence_roundtrip() -> None:
    owner = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=17)
    plan = owner.plan(
        transaction_id="tx-persist",
        active_k=8,
        num_segments=40,
        eligible=lambda _segment_id: True,
        global_descriptor=_descriptor,
    )
    keys = tuple(_key(selection, suffix=str(index)) for index, selection in enumerate(plan.selections))
    candidate = owner.stage(
        plan,
        keys=keys,
        actor_advantages=torch.arange(1.0, 9.0),
        source_index=torch.tensor([0] * 4 + [1] * 4),
        policy_snapshot_id="pi-persist",
        active_m=4,
    )
    owner.commit(candidate, receipt=_receipt(plan, policy_snapshot_id="pi-persist"))
    state = owner.state_dict()
    restored = FrontRESOuterScenarioReplay(global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=99)
    restored.load_state_dict(state)
    assert tuple(record.to_state() for record in restored.records) == state["records"]
    torch.testing.assert_close(
        restored.state_dict()["generator_state"], state["generator_state"], rtol=0.0, atol=0.0
    )
    malformed = dict(state)
    malformed.pop("records")
    try:
        restored.load_state_dict(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("missing persistence field must fail")


def main() -> None:
    test_seeded_identity_and_rng_isolation()
    test_negative_advantage_priority_and_committed_only_mutation()
    test_replay_selection_k_isolation_and_key_counterexample()
    test_strict_persistence_roundtrip()
    print("frontres_v020_outer_scenario_replay_contract: PASS")


if __name__ == "__main__":
    main()
