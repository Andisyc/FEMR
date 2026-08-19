from __future__ import annotations

import math

from rsl_rl.frontres.frontres_segment_warmup import (
    FRONTRES_V022_ACTOR_LR_INIT,
    FRONTRES_V022_ACTOR_LR_JOINT,
    frontres_actor_only_learning_rate_phase,
)


INIT_TRANSACTIONS = 100
RAMP_TRANSACTIONS = 50


def _expected_lr(global_transaction: int) -> float:
    if global_transaction < INIT_TRANSACTIONS:
        return FRONTRES_V022_ACTOR_LR_INIT
    ramp_iteration = global_transaction - INIT_TRANSACTIONS
    if ramp_iteration >= RAMP_TRANSACTIONS:
        return FRONTRES_V022_ACTOR_LR_JOINT
    progress = ramp_iteration / (RAMP_TRANSACTIONS - 1)
    return FRONTRES_V022_ACTOR_LR_INIT + (
        FRONTRES_V022_ACTOR_LR_JOINT - FRONTRES_V022_ACTOR_LR_INIT
    ) * progress


def _phase(global_transaction: int):
    return frontres_actor_only_learning_rate_phase(
        committed_transaction_iteration=global_transaction,
        init_transactions=INIT_TRANSACTIONS,
        ramp_transactions=RAMP_TRANSACTIONS,
    )


def main() -> None:
    # C1/C2: exact initial, ramp, endpoint, and stable-region values.
    for transaction in (0, 99, 100, 101, 125, 149, 150, 4000):
        observed = _phase(transaction)
        expected = _expected_lr(transaction)
        assert math.isclose(observed.actor_learning_rate, expected, rel_tol=0.0, abs_tol=1e-15)
        assert observed.critic_update_enabled is False

    assert _phase(99).name == "actor_only_init"
    assert _phase(100).name == "actor_only_ramp"
    assert _phase(149).name == "actor_only_ramp"
    assert _phase(150).name == "actor_only_stable"

    # C4 sensitivity: a K-local counter reset would incorrectly return 3e-7.
    global_progress = 150
    k_local_progress = 0
    correct_after_k_transition = _phase(global_progress).actor_learning_rate
    reset_mutant = _phase(k_local_progress).actor_learning_rate
    assert correct_after_k_transition == FRONTRES_V022_ACTOR_LR_JOINT
    assert reset_mutant == FRONTRES_V022_ACTOR_LR_INIT
    assert correct_after_k_transition != reset_mutant

    # C3: invalid progress and ambiguous curriculum durations fail closed.
    invalid_calls = (
        dict(committed_transaction_iteration=-1, init_transactions=100, ramp_transactions=50),
        dict(committed_transaction_iteration=True, init_transactions=100, ramp_transactions=50),
        dict(committed_transaction_iteration=0, init_transactions=49, ramp_transactions=50),
        dict(committed_transaction_iteration=0, init_transactions=101, ramp_transactions=50),
        dict(committed_transaction_iteration=0, init_transactions=100, ramp_transactions=1),
    )
    for kwargs in invalid_calls:
        try:
            frontres_actor_only_learning_rate_phase(**kwargs)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"candidate LR curriculum accepted invalid input: {kwargs}")

    print(
        "frontres_actor_only_lr_curriculum_alignment: MODULE-CORRECT "
        "cases=C1,C2,C3,C4 sensitivity=k_local_reset"
    )


if __name__ == "__main__":
    main()
