from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_sampler.py"
spec = importlib.util.spec_from_file_location("frontres_segment_sampler", MODULE_PATH)
sampler_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sampler_module
spec.loader.exec_module(sampler_module)
FrontRESSegmentRolloutEvidence = sampler_module.FrontRESSegmentRolloutEvidence
FrontRESSegmentRolloutBudget = sampler_module.FrontRESSegmentRolloutBudget
FrontRESSegmentSampler = sampler_module.FrontRESSegmentSampler
FrontRESSegmentState = sampler_module.FrontRESSegmentState
FrontRESSegmentTrialPlan = sampler_module.FrontRESSegmentTrialPlan
FrontRESFrozenPolicyTransactionPlan = sampler_module.FrontRESFrozenPolicyTransactionPlan
FrontRESSegmentTrialEvidence = sampler_module.FrontRESSegmentTrialEvidence


def _evidence(
    segment_ids: list[int],
    gain: list[float],
    repaired: list[float],
    noisy: list[float],
    *,
    fall: list[bool] | None = None,
    valid: list[bool] | None = None,
    horizon_k: list[int] | int = 4,
) -> FrontRESSegmentRolloutEvidence:
    n = len(segment_ids)
    if isinstance(horizon_k, int):
        horizon = torch.ones(n, dtype=torch.long) * int(horizon_k)
    else:
        horizon = torch.tensor(horizon_k, dtype=torch.long)
    return FrontRESSegmentRolloutEvidence(
        segment_ids=torch.tensor(segment_ids, dtype=torch.long),
        reset_success=torch.ones(n, dtype=torch.bool),
        score_noisy=torch.tensor(noisy, dtype=torch.float32),
        score_repaired=torch.tensor(repaired, dtype=torch.float32),
        score_clean=torch.ones(n, dtype=torch.float32),
        gain_over_noisy=torch.tensor(gain, dtype=torch.float32),
        fall_repaired=torch.tensor(fall if fall is not None else [False] * n, dtype=torch.bool),
        contact_consistency=torch.ones(n, dtype=torch.float32),
        action_norm=torch.ones(n, dtype=torch.float32),
        valid_reward=torch.tensor(valid if valid is not None else [True] * n, dtype=torch.bool),
        horizon_k=horizon,
        gain_total=torch.tensor(gain, dtype=torch.float32),
        gain_style=torch.tensor(gain, dtype=torch.float32),
        gain_physics=torch.zeros(n, dtype=torch.float32),
        repair_cost=torch.zeros(n, dtype=torch.float32),
        gain_source="FRS-GAIN-v002",
    )


def test_sampler_global_sampling_visits_unseen_segments() -> None:
    sampler = FrontRESSegmentSampler(5, global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=3)
    sample = sampler.sample(5)
    assert set(sample.segment_ids.tolist()) == {0, 1, 2, 3, 4}
    assert sample.source == ("global", "global", "global", "global", "global")
    assert sampler.stats().seen_count == 5


def test_sampler_replays_useful_unsolved_segments() -> None:
    sampler = FrontRESSegmentSampler(4, global_frac=0.0, replay_frac=1.0, review_frac=0.0, seed=7)
    sampler.update(_evidence([0, 1, 2, 3], gain=[0.5, 0.02, -0.2, 0.1], repaired=[0.6, 0.98, 0.1, 0.4], noisy=[0.2, 0.96, 0.2, 0.3], fall=[False, False, True, False]))

    assert sampler.priority[0] > sampler.priority[1]
    assert sampler.solved[1].item()
    assert sampler.hopeless[2].item()

    sample = sampler.sample(12)
    assert 0 in sample.segment_ids.tolist()
    assert 2 not in sample.segment_ids.tolist()


def test_sampler_reports_effective_source_after_fallback() -> None:
    sampler = FrontRESSegmentSampler(4, global_frac=0.0, replay_frac=1.0, review_frac=0.0, seed=17)
    sample = sampler.sample(4)
    assert sample.source == ("global", "global", "global", "global")
    sampler = FrontRESSegmentSampler(4, global_frac=0.0, replay_frac=0.0, review_frac=1.0, seed=19)
    sample = sampler.sample(4)
    assert sample.source == ("global", "global", "global", "global")


def test_sampler_deterministic_eval_reset_ignores_checkpoint_frontier() -> None:
    sampler = FrontRESSegmentSampler(5, seed=7)
    sampler.priority.fill_(3.0)
    sampler.solved[0] = True
    sampler.invalid[1] = True
    sampler.reset_for_deterministic_eval(seed=19)

    fresh = FrontRESSegmentSampler(5, seed=19)
    sample = sampler.sample(10)
    fresh_sample = fresh.sample(10)

    assert sample.segment_ids.tolist() == fresh_sample.segment_ids.tolist()
    assert sample.source == fresh_sample.source
    assert int(sampler.stats().seen_count) == int(fresh.stats().seen_count)


def test_sampler_update_probe_exposes_priority_boundary() -> None:
    sampler = FrontRESSegmentSampler(4, global_frac=0.0, replay_frac=1.0, review_frac=0.0, seed=23)
    probe = sampler.update_with_probe(
        _evidence([0, 1, 2, 3], gain=[0.5, -0.1, 0.0, 0.2], repaired=[0.6, 0.2, 0.3, 0.7], noisy=[0.2, 0.3, 0.3, 0.4], fall=[False, True, False, False])
    )
    print(
        "[probe sampler_update] "
        f"valid={probe.valid_count} fall={probe.fall_count} "
        f"useful_mean={probe.useful_mean:.6f} useful_max={probe.useful_max:.6f} "
        f"priority_before={probe.priority_before_mean:.6f} priority_after={probe.priority_after_mean:.6f} "
        f"replay_candidates={probe.replay_candidate_count} hopeless={probe.hopeless_count}",
        flush=True,
    )
    assert probe.valid_count == 4
    assert probe.fall_count == 1
    assert probe.useful_max > 0.0
    assert probe.priority_after_mean > probe.priority_before_mean
    assert probe.replay_candidate_count > 0
    assert probe.hopeless_count == 1


def test_sampler_state_model_tracks_learning_frontier() -> None:
    sampler = FrontRESSegmentSampler(4, global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=29)

    assert sampler.segment_state.tolist() == [FrontRESSegmentState.UNKNOWN] * 4

    sampler.update(_evidence([0], gain=[0.30], repaired=[0.55], noisy=[0.20], horizon_k=8))
    assert int(sampler.segment_state[0].item()) == FrontRESSegmentState.PROMISING

    sampler.update(_evidence([0], gain=[0.25], repaired=[0.60], noisy=[0.20], horizon_k=8))
    assert int(sampler.segment_state[0].item()) == FrontRESSegmentState.FRONTIER

    sampler.update(_evidence([0], gain=[-0.10], repaired=[0.35], noisy=[0.45], fall=[False], horizon_k=32))
    assert int(sampler.segment_state[0].item()) == FrontRESSegmentState.DELAYED_REGRET
    assert not sampler.hopeless[0].item()

    sampler.update(_evidence([1], gain=[0.01], repaired=[0.95], noisy=[0.94], horizon_k=8))
    sampler.update(_evidence([2], gain=[-0.20], repaired=[0.10], noisy=[0.10], fall=[True], horizon_k=8))
    stats = sampler.stats()
    print(
        "[probe sampler_state] "
        f"states={sampler.segment_state.tolist()} "
        f"unknown={stats.unknown_count} promising={stats.promising_count} "
        f"frontier={stats.frontier_count} delayed={stats.delayed_regret_count} "
        f"solved={stats.solved_count} hopeless={stats.hopeless_count}",
        flush=True,
    )
    assert int(sampler.segment_state[1].item()) == FrontRESSegmentState.SOLVED
    assert int(sampler.segment_state[2].item()) == FrontRESSegmentState.HOPELESS
    assert stats.delayed_regret_count == 1
    assert stats.solved_count == 1
    assert stats.hopeless_count == 1


def test_sampler_priority_and_state_ignore_legacy_scores() -> None:
    base = _evidence(
        [0, 1],
        gain=[0.20, -0.20],
        repaired=[0.00, 0.00],
        noisy=[0.00, 0.00],
        fall=[False, True],
    )
    poisoned = _evidence(
        [0, 1],
        gain=[0.20, -0.20],
        repaired=[1.00, 1.00],
        noisy=[1.00, 1.00],
        fall=[False, True],
    )
    clean_sampler = FrontRESSegmentSampler(2, seed=53)
    poisoned_sampler = FrontRESSegmentSampler(2, seed=53)
    clean_update = clean_sampler.update_with_probe(base)
    poisoned_update = poisoned_sampler.update_with_probe(poisoned)
    print(
        "[probe sampler_gain_only] "
        f"clean_priority={clean_sampler.priority.tolist()} "
        f"poisoned_priority={poisoned_sampler.priority.tolist()} "
        f"clean_state={clean_sampler.segment_state.tolist()} "
        f"poisoned_state={poisoned_sampler.segment_state.tolist()}",
        flush=True,
    )
    torch.testing.assert_close(clean_sampler.priority, poisoned_sampler.priority)
    assert clean_sampler.segment_state.tolist() == poisoned_sampler.segment_state.tolist()
    assert clean_update.priority_after_mean == poisoned_update.priority_after_mean
    assert clean_sampler.segment_state.tolist() == [FrontRESSegmentState.PROMISING, FrontRESSegmentState.HOPELESS]


def test_sampler_state_model_round_trips_and_migrates_legacy_state() -> None:
    sampler = FrontRESSegmentSampler(3, seed=31)
    sampler.update(_evidence([0], gain=[0.20], repaired=[0.50], noisy=[0.20], horizon_k=8))
    sampler.update(_evidence([0], gain=[-0.10], repaired=[0.35], noisy=[0.45], horizon_k=32))
    sampler.update(_evidence([1], gain=[0.01], repaired=[0.95], noisy=[0.94], horizon_k=8))

    restored = FrontRESSegmentSampler(3, seed=31)
    restored.load_state_dict(sampler.state_dict())
    assert restored.segment_state.tolist() == sampler.segment_state.tolist()
    torch.testing.assert_close(restored.best_short_gain, sampler.best_short_gain)
    assert restored.evidence_count.tolist() == sampler.evidence_count.tolist()

    legacy = {
        "priority": torch.zeros(3),
        "staleness": torch.zeros(3),
        "seen": torch.tensor([True, True, True]),
        "solved": torch.tensor([False, True, False]),
        "hopeless": torch.tensor([False, False, True]),
        "invalid": torch.zeros(3, dtype=torch.bool),
        "invalid_reasons": {},
    }
    migrated = FrontRESSegmentSampler(3, seed=31)
    migrated.load_state_dict(legacy)
    print(
        "[probe sampler_state_migration] "
        f"legacy_solved={legacy['solved'].tolist()} "
        f"legacy_hopeless={legacy['hopeless'].tolist()} "
        f"migrated_state={migrated.segment_state.tolist()}",
        flush=True,
    )
    assert int(migrated.segment_state[0].item()) == FrontRESSegmentState.UNKNOWN
    assert int(migrated.segment_state[1].item()) == FrontRESSegmentState.SOLVED
    assert int(migrated.segment_state[2].item()) == FrontRESSegmentState.HOPELESS


def test_sampler_multi_trial_aggregates_fixed_policy_visit() -> None:
    sampler = FrontRESSegmentSampler(3, seed=37)
    trial = sampler.aggregate_trial_evidence(
        _evidence(
            [0, 0, 0, 1, 1],
            gain=[0.10, 0.60, -0.20, 0.20, 0.25],
            repaired=[0.30, 0.80, 0.10, 0.50, 0.55],
            noisy=[0.20, 0.20, 0.20, 0.30, 0.30],
            fall=[False, False, True, False, False],
            horizon_k=8,
        )
    )
    print(
        "[probe sampler_multi_trial] "
        f"ids={trial.segment_ids.tolist()} "
        f"trial_count={trial.trial_count.tolist()} "
        f"policy_gain={trial.policy_gain.tolist()} "
        f"best_gain={trial.best_gain.tolist()} "
        f"mean_gain={trial.mean_gain.tolist()} "
        f"success_frac={trial.success_frac.tolist()} "
        f"fall_frac={trial.fall_frac.tolist()} "
        f"oracle_gap={trial.oracle_gap.tolist()} "
        f"confidence={trial.confidence.tolist()}",
        flush=True,
    )
    assert isinstance(trial, FrontRESSegmentTrialEvidence)
    assert trial.segment_ids.tolist() == [0, 1]
    assert trial.trial_count.tolist() == [3, 2]
    torch.testing.assert_close(trial.policy_gain, torch.tensor([0.10, 0.20]))
    torch.testing.assert_close(trial.best_gain, torch.tensor([0.60, 0.25]))
    torch.testing.assert_close(trial.mean_gain, torch.tensor([0.16666667, 0.225]))
    torch.testing.assert_close(trial.success_frac, torch.tensor([2.0 / 3.0, 1.0]))
    torch.testing.assert_close(trial.fall_frac, torch.tensor([1.0 / 3.0, 0.0]))
    torch.testing.assert_close(trial.oracle_gap, torch.tensor([0.50, 0.05]))
    torch.testing.assert_close(trial.confidence, torch.tensor([2.0 / 3.0, 2.0 / 3.0]))


def test_sampler_multi_trial_update_records_oracle_gap_without_policy_update() -> None:
    sampler = FrontRESSegmentSampler(3, seed=41)
    probe = sampler.update_with_probe(
        _evidence(
            [0, 0, 0],
            gain=[0.10, 0.60, -0.20],
            repaired=[0.30, 0.80, 0.10],
            noisy=[0.20, 0.20, 0.20],
            fall=[False, False, True],
            horizon_k=8,
        )
    )
    print(
        "[probe sampler_multi_trial_update] "
        f"segment_count={probe.segment_count} trial_count={probe.trial_count} "
        f"oracle_gap_mean={probe.oracle_gap_mean:.6f} "
        f"last_trial_count={sampler.last_trial_count.tolist()} "
        f"last_oracle_gap={sampler.last_oracle_gap.tolist()} "
        f"states={sampler.segment_state.tolist()}",
        flush=True,
    )
    assert probe.segment_count == 1
    assert probe.trial_count == 3
    assert probe.valid_count == 3
    assert probe.fall_count == 1
    assert probe.oracle_gap_mean > 0.49
    assert int(sampler.last_trial_count[0].item()) == 3
    torch.testing.assert_close(sampler.last_oracle_gap[0], torch.tensor(0.50))
    assert int(sampler.segment_state[0].item()) == FrontRESSegmentState.FRONTIER


def _sampler_with_all_budget_states() -> FrontRESSegmentSampler:
    sampler = FrontRESSegmentSampler(6, seed=43)
    sampler.update(_evidence([1], gain=[0.30], repaired=[0.55], noisy=[0.20], horizon_k=8))
    sampler.update(
        _evidence(
            [2, 2, 2],
            gain=[0.10, 0.60, -0.20],
            repaired=[0.30, 0.80, 0.10],
            noisy=[0.20, 0.20, 0.20],
            fall=[False, False, True],
            horizon_k=8,
        )
    )
    sampler.update(_evidence([3], gain=[0.30], repaired=[0.55], noisy=[0.20], horizon_k=8))
    sampler.update(_evidence([3], gain=[-0.10], repaired=[0.35], noisy=[0.45], fall=[False], horizon_k=32))
    sampler.update(_evidence([4], gain=[0.01], repaired=[0.95], noisy=[0.94], horizon_k=8))
    sampler.update(_evidence([5], gain=[-0.20], repaired=[0.10], noisy=[0.10], fall=[True], horizon_k=8))
    return sampler


def test_sampler_rollout_budget_allocates_trials_by_state_and_horizon_unlock() -> None:
    sampler = _sampler_with_all_budget_states()
    budget = sampler.plan_rollout_budget(torch.tensor([0, 1, 2, 3, 4, 5]), max_horizon_k=32)
    print(
        "[probe sampler_budget] "
        f"states={budget.segment_state.tolist()} "
        f"trials={budget.trial_count.tolist()} "
        f"horizon={budget.horizon_k.tolist()} "
        f"reason={budget.reason}",
        flush=True,
    )
    assert isinstance(budget, FrontRESSegmentRolloutBudget)
    assert budget.trial_count.tolist() == [1, 3, 6, 6, 1, 1]
    assert budget.horizon_k.tolist() == [8, 16, 32, 32, 32, 8]
    assert budget.reason == (
        "unknown_probe",
        "promising_local_trials",
        "frontier_multi_trial",
        "delayed_regret_long_check",
        "solved_review",
        "hopeless_recheck",
    )

    short_budget = sampler.plan_rollout_budget(torch.tensor([2, 3, 4]), max_horizon_k=8)
    assert short_budget.horizon_k.tolist() == [8, 8, 8]

    full_budget = sampler.plan_rollout_budget(torch.tensor([0, 1, 2, 3, 4, 5]), max_horizon_k=64)
    assert full_budget.horizon_k.tolist() == [8, 16, 32, 64, 64, 8]


def test_sampler_trial_plan_expands_policy_first_roles() -> None:
    sampler = _sampler_with_all_budget_states()
    plan = sampler.expand_rollout_trials(torch.tensor([1, 2]), max_horizon_k=32)
    print(
        "[probe sampler_trial_plan] "
        f"segment_ids={plan.segment_ids.tolist()} "
        f"source_index={plan.source_index.tolist()} "
        f"trial_index={plan.trial_index.tolist()} "
        f"roles={plan.trial_role} "
        f"horizon={plan.horizon_k.tolist()}",
        flush=True,
    )
    assert isinstance(plan, FrontRESSegmentTrialPlan)
    assert plan.segment_ids.tolist() == [1, 1, 1, 2, 2, 2, 2, 2, 2]
    assert plan.source_index.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert plan.trial_index.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5]
    assert plan.trial_role == (
        "policy",
        "search",
        "search",
        "policy",
        "search",
        "search",
        "search",
        "search",
        "search",
    )
    assert plan.horizon_k.tolist() == [16, 16, 16, 32, 32, 32, 32, 32, 32]


def test_sampler_frozen_policy_transaction_plan_keeps_all_attempts_policy_sampled() -> None:
    sampler = _sampler_with_all_budget_states()
    priority_before = sampler.priority.clone()
    staleness_before = sampler.staleness.clone()
    seen_before = sampler.seen.clone()

    plan = sampler.plan_frozen_policy_transaction(
        torch.tensor([1, 2]),
        transaction_id="txn-41",
        policy_snapshot_id="snapshot-17",
        max_horizon_k=32,
    )
    print(
        "[probe frozen_policy_transaction_plan] "
        f"transaction={plan.transaction_id} snapshot={plan.policy_snapshot_id} "
        f"segment_ids={plan.segment_ids.tolist()} "
        f"source_index={plan.source_index.tolist()} "
        f"trial_index={plan.trial_index.tolist()} "
        f"roles={plan.trial_role} horizon={plan.horizon_k.tolist()} "
        f"counts={plan.base_trial_count.tolist()}",
        flush=True,
    )

    assert isinstance(plan, FrontRESFrozenPolicyTransactionPlan)
    assert plan.transaction_id == "txn-41"
    assert plan.policy_snapshot_id == "snapshot-17"
    assert plan.base_segment_ids.tolist() == [1, 2]
    assert plan.base_trial_count.tolist() == [3, 6]
    assert plan.segment_ids.tolist() == [1, 1, 1, 2, 2, 2, 2, 2, 2]
    assert plan.source_index.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert plan.trial_index.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5]
    assert plan.horizon_k.tolist() == [16, 16, 16, 32, 32, 32, 32, 32, 32]
    assert plan.trial_role == ("policy",) * 9
    assert torch.equal(sampler.priority, priority_before)
    assert torch.equal(sampler.staleness, staleness_before)
    assert torch.equal(sampler.seen, seen_before)


def test_v009_frozen_transaction_overrides_legacy_per_segment_k() -> None:
    sampler = _sampler_with_all_budget_states()
    plan = sampler.plan_frozen_policy_transaction(
        torch.tensor([1, 2]),
        transaction_id="txn-v009-global-k",
        policy_snapshot_id="snapshot-v009-global-k",
        max_horizon_k=32,
        active_horizon_k=24,
    )
    assert plan.base_horizon_k.tolist() == [24, 24]
    assert plan.horizon_k.tolist() == [24] * int(plan.horizon_k.numel())
    for invalid in (0, 33):
        try:
            sampler.plan_frozen_policy_transaction(
                torch.tensor([1, 2]),
                transaction_id=f"txn-v009-invalid-{invalid}",
                policy_snapshot_id="snapshot-v009-invalid",
                max_horizon_k=32,
                active_horizon_k=invalid,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("v009 active K outside the formal horizon must reject")


def test_sampler_frozen_policy_transaction_plan_enforces_multiple_segments_and_attempts() -> None:
    sampler = _sampler_with_all_budget_states()
    plan = sampler.plan_frozen_policy_transaction(
        torch.tensor([0, 4]),
        transaction_id="txn-min-attempts",
        policy_snapshot_id="snapshot-min-attempts",
        max_horizon_k=32,
    )
    assert plan.base_trial_count.tolist() == [2, 2]
    assert plan.segment_ids.tolist() == [0, 0, 4, 4]
    assert plan.trial_index.tolist() == [0, 1, 0, 1]
    assert plan.trial_role == ("policy",) * 4

    for selected, transaction_id, snapshot_id in (
        (torch.tensor([0]), "txn-one", "snapshot-one"),
        (torch.tensor([0, 0]), "txn-duplicate", "snapshot-duplicate"),
        (torch.tensor([0, 1]), "", "snapshot-missing-txn"),
        (torch.tensor([0, 1]), "txn-missing-snapshot", ""),
    ):
        try:
            sampler.plan_frozen_policy_transaction(
                selected,
                transaction_id=transaction_id,
                policy_snapshot_id=snapshot_id,
                max_horizon_k=32,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("frozen transaction planner must reject malformed transaction identity or segment groups")


def test_sampler_rollout_row_sampling_respects_fixed_env_budget() -> None:
    sampler = FrontRESSegmentSampler(2, global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=47)
    sampler.mark_invalid([1], "holdout")
    sampler.segment_state[0] = int(FrontRESSegmentState.FRONTIER)
    sampler.last_trial_count[0] = 3
    sampler.last_oracle_gap[0] = 0.5
    sampler.last_success_frac[0] = 0.5

    sample = sampler.sample_rollout_rows(2, max_horizon_k=32)
    print(
        "[probe sampler_rollout_rows] "
        f"segment_ids={sample.segment_ids.tolist()} "
        f"roles={sample.trial_role} "
        f"trial_index={sample.trial_index.tolist()} "
        f"horizon={sample.horizon_k.tolist()} "
        f"seen={sampler.seen.tolist()}",
        flush=True,
    )
    assert int(sample.segment_ids.numel()) == 2
    assert sample.segment_ids.tolist() == [0, 0]
    assert sample.trial_role == ("policy", "search")
    assert sample.trial_index.tolist() == [0, 1]
    assert sample.horizon_k.tolist() == [32, 32]
    assert sampler.seen.tolist() == [True, False]


def test_sampler_review_and_staleness_keep_coverage() -> None:
    sampler = FrontRESSegmentSampler(3, global_frac=0.0, replay_frac=0.0, review_frac=1.0, seed=11)
    sampler.update(_evidence([0, 1], gain=[0.01, 0.4], repaired=[0.95, 0.5], noisy=[0.94, 0.1]))

    review = sampler.sample(4)
    assert set(review.segment_ids.tolist()) == {0}

    sampler.staleness[1] = 100.0
    sampler.global_frac = 0.0
    sampler.replay_frac = 1.0
    sampler.review_frac = 0.0
    replay = sampler.sample(8)
    assert 1 in replay.segment_ids.tolist()


def test_sampler_invalid_and_state_dict_restore() -> None:
    sampler = FrontRESSegmentSampler(4, global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=5)
    sampler.mark_invalid([0, 1], "bad reset")
    sample = sampler.sample(12)
    assert not ({0, 1} & set(sample.segment_ids.tolist()))

    restored = FrontRESSegmentSampler(4, seed=5)
    restored.load_state_dict(sampler.state_dict())
    assert restored.invalid.tolist() == sampler.invalid.tolist()
    assert restored.invalid_reasons[0] == "bad reset"


def main() -> None:
    test_sampler_global_sampling_visits_unseen_segments()
    test_sampler_replays_useful_unsolved_segments()
    test_sampler_reports_effective_source_after_fallback()
    test_sampler_update_probe_exposes_priority_boundary()
    test_sampler_state_model_tracks_learning_frontier()
    test_sampler_priority_and_state_ignore_legacy_scores()
    test_sampler_state_model_round_trips_and_migrates_legacy_state()
    test_sampler_multi_trial_aggregates_fixed_policy_visit()
    test_sampler_multi_trial_update_records_oracle_gap_without_policy_update()
    test_sampler_rollout_budget_allocates_trials_by_state_and_horizon_unlock()
    test_sampler_trial_plan_expands_policy_first_roles()
    test_sampler_frozen_policy_transaction_plan_keeps_all_attempts_policy_sampled()
    test_v009_frozen_transaction_overrides_legacy_per_segment_k()
    test_sampler_frozen_policy_transaction_plan_enforces_multiple_segments_and_attempts()
    test_sampler_rollout_row_sampling_respects_fixed_env_budget()
    test_sampler_review_and_staleness_keep_coverage()
    test_sampler_invalid_and_state_dict_restore()
    print("result: PASS")


if __name__ == "__main__":
    main()
