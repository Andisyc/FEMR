"""Offline alignment tests for the candidate relational training adapter."""

from dataclasses import replace

from rsl_rl.frontres.frontres_hierarchical_gain_candidate import (
    Outcome,
    build_relational_training_batch,
)


def main() -> None:
    stable = Outcome()
    unsettled = replace(stable, capture_margin=0.0, stable_hold_steps=0)
    severe = replace(stable, expected_support_no_load=0.2)
    failed = replace(stable, survival_ok=False, survival_failure_duration=0.2)
    incomparable = replace(stable, intent_error=(0.2, 0.05))

    # Hand oracle: L3 > L2 > L1 > L0, so exact M4 edge credits are
    # (+3, +1, -1, -3). This is independent of the implementation.
    outcomes = (stable, unsettled, severe, failed)
    batch = build_relational_training_batch(outcomes)
    assert batch.status == "READY"
    assert batch.pair_relations[0][1] == "BETTER"
    assert batch.pair_relations[1][2] == "BETTER"
    assert batch.pair_relations[2][3] == "BETTER"
    assert batch.dominance_credit == (3.0, 1.0, -1.0, -3.0)
    assert batch.preference_edges == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    assert batch.comparable_pair_count == (3, 3, 3, 3)
    assert all(batch.actor_credit_mask)
    assert sum(batch.dominance_credit) == 0.0
    assert tuple(-value for value in batch.dominance_credit) != batch.dominance_credit

    # Row order carries no semantics: reversing M4 reverses the aligned output.
    permuted = build_relational_training_batch(tuple(reversed(outcomes)))
    assert permuted.dominance_credit == tuple(reversed(batch.dominance_credit))
    assert permuted.comparable_pair_count == tuple(reversed(batch.comparable_pair_count))
    assert permuted.actor_credit_mask == tuple(reversed(batch.actor_credit_mask))
    assert permuted.dominance_credit != batch.dominance_credit

    no_direction = build_relational_training_batch((stable, incomparable))
    assert no_direction.status == "NO_COMPARABLE_PAIRS"
    assert no_direction.dominance_credit == (None, None)
    assert no_direction.actor_credit_mask == (False, False)
    assert no_direction.preference_edges == ()

    # Controlled counterexample: zero-filling undefined credit would fabricate
    # two Critic targets even though the human oracle provides no ordering.
    zero_filled_mutant = tuple(
        0.0 if value is None else value for value in no_direction.dominance_credit
    )
    assert zero_filled_mutant == (0.0, 0.0)
    assert tuple(True for _ in zero_filled_mutant) != no_direction.actor_credit_mask

    invalid = build_relational_training_batch((stable, replace(stable, zmp_margin=float("nan"))))
    assert invalid.status == "INVALID"
    assert invalid.actor_credit_mask == (False, False)
    assert tuple(True for _ in invalid.actor_credit_mask) != invalid.actor_credit_mask

    print("frontres_relational_training_adapter_alignment: OBJECTIVE-ALIGNED")


if __name__ == "__main__":
    main()
