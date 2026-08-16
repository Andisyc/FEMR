#!/usr/bin/env python3
"""Independent T-preference test for the proposed hierarchical Gain.

Candidate under test:
  P_X_new = (S_X, H_X, C_X), H_X = L_X + W_X
  survival change -> +/-3
  severe-contact change -> signed magnitude in (1, 2)
  ordinary Intent/Physics/cost -> tanh range (-1, 1)

The oracle is only the human-confirmed partial order.  This file does not touch
or import the active Gain implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CARD_PATH = PROJECT_ROOT / "note/testing/manifests/frontres_pr_behavioral_ordering_v1.json"
EPSILON = 1.0e-12
NA = "N/A"
ALLOWED_STATUS = {
    "OBJECTIVE-ALIGNED",
    "OBJECTIVE-VIOLATION",
    "INCONCLUSIVE",
    "TELEMETRY-GAP",
}


class CandidateInputError(ValueError):
    """Missing or invalid candidate evidence fails closed."""


@dataclass(frozen=True)
class Outcome:
    survival: object
    no_load: object
    unplanned_switch: object
    ordinary_contact: object = 0.0
    intent_error: object = 0.0
    drift: object = 0.0
    zmp: object = 0.0
    repair_cost: object = 0.0


def _problem(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateInputError(f"{field} must be present as a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CandidateInputError(f"{field} must be finite and non-negative")
    return result


def _ordinary_vector(outcome: Outcome) -> tuple[float, ...]:
    values: list[float] = []
    for field in ("ordinary_contact", "intent_error", "drift", "zmp", "repair_cost"):
        value = getattr(outcome, field)
        if value == NA:
            if field != "zmp":
                raise CandidateInputError(f"{field} cannot be N/A")
            continue
        values.append(_problem(value, field=field))
    if not values:
        raise CandidateInputError("ordinary evidence has no applicable item")
    return tuple(values)


def candidate_gain(noisy: Outcome, repaired: Outcome) -> tuple[float, str, dict[str, float]]:
    """Public boundary for the isolated candidate, not an oracle."""

    noisy_s = _problem(noisy.survival, field="noisy.survival")
    repaired_s = _problem(repaired.survival, field="repaired.survival")
    noisy_l = _problem(noisy.no_load, field="noisy.no_load")
    repaired_l = _problem(repaired.no_load, field="repaired.no_load")
    noisy_w = _problem(noisy.unplanned_switch, field="noisy.unplanned_switch")
    repaired_w = _problem(repaired.unplanned_switch, field="repaired.unplanned_switch")
    survival_change = noisy_s - repaired_s
    noisy_h = noisy_l + noisy_w
    repaired_h = repaired_l + repaired_w
    hard_change = noisy_h - repaired_h

    if abs(survival_change) > EPSILON:
        value = math.copysign(3.0, survival_change)
        return value, "survival", {"delta_s": survival_change, "delta_h": hard_change}
    if abs(hard_change) > EPSILON:
        magnitude = 1.0 + abs(hard_change) / (1.0 + abs(hard_change))
        value = math.copysign(magnitude, hard_change)
        return value, "severe_contact", {"delta_s": survival_change, "delta_h": hard_change}

    noisy_ordinary = _ordinary_vector(noisy)
    repaired_ordinary = _ordinary_vector(repaired)
    if len(noisy_ordinary) != len(repaired_ordinary):
        raise CandidateInputError("Noisy and Repair ordinary applicability must match")
    ordinary_change = sum(noisy_ordinary) - sum(repaired_ordinary)
    return math.tanh(ordinary_change), "ordinary", {
        "delta_s": survival_change,
        "delta_h": hard_change,
    }


def _relation(left: float, right: float) -> str:
    if left > right + EPSILON:
        return "left_better"
    if right > left + EPSILON:
        return "right_better"
    return "tied"


def _case(
    *,
    case_id: str,
    noisy: Outcome,
    left: Outcome,
    right: Outcome,
    expected: str,
    rationale: str,
) -> dict[str, Any]:
    left_gain, left_branch, left_parts = candidate_gain(noisy, left)
    right_gain, right_branch, right_parts = candidate_gain(noisy, right)
    observed = _relation(left_gain, right_gain)
    if expected == "incomparable":
        aligned = observed == "tied"
    else:
        aligned = observed == expected
    return {
        "id": case_id,
        "expected_relation": expected,
        "rationale": rationale,
        "left": {"gain": left_gain, "branch": left_branch, **left_parts},
        "right": {"gain": right_gain, "branch": right_branch, **right_parts},
        "observed_relation": observed,
        "status": "OBJECTIVE-ALIGNED" if aligned else "OBJECTIVE-VIOLATION",
    }


def _fail_closed_cases() -> list[dict[str, Any]]:
    base = Outcome(0.0, 0.0, 0.0)
    cases: list[dict[str, Any]] = []
    for case_id, bad in (
        ("missing-severe-evidence", Outcome(None, 0.0, 0.0)),
        ("invalid-severe-evidence", Outcome(float("nan"), 0.0, 0.0)),
        ("missing-ordinary-evidence", Outcome(0.0, 0.0, 0.0, drift=None)),
    ):
        try:
            candidate_gain(base, bad)
        except CandidateInputError as error:
            cases.append({
                "id": case_id,
                "expected_relation": "fail_closed",
                "observed": str(error),
                "status": "OBJECTIVE-ALIGNED",
            })
        else:
            cases.append({
                "id": case_id,
                "expected_relation": "fail_closed",
                "observed": "accepted",
                "status": "OBJECTIVE-VIOLATION",
            })

    na_noisy = Outcome(0.0, 0.0, 0.0, zmp=NA, intent_error=1.0)
    na_repaired = Outcome(0.0, 0.0, 0.0, zmp=NA, intent_error=0.0)
    value, branch, _ = candidate_gain(na_noisy, na_repaired)
    cases.append({
        "id": "phase-zmp-semantic-na",
        "expected_relation": "applicable ordinary evidence preserved while ZMP is N/A",
        "observed": {"gain": value, "branch": branch},
        "status": (
            "OBJECTIVE-ALIGNED"
            if branch == "ordinary" and 0.0 < value < 1.0
            else "OBJECTIVE-VIOLATION"
        ),
    })
    return cases


def run_alignment() -> dict[str, Any]:
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    if card.get("status") != "CONFIRMED" or card.get("ordering_kind") != "partial":
        raise AssertionError("candidate test requires the confirmed partial-order card")
    if card.get("oracle", {}).get("second_scoring_formula_forbidden") is not True:
        raise AssertionError("candidate test requires the no-second-score oracle boundary")

    zero = Outcome(0.0, 0.0, 0.0)
    cases = [
        _case(
            case_id="no-severe-must-beat-remaining-no-load",
            noisy=Outcome(1.0, 1.0, 0.0),
            left=zero,
            right=Outcome(0.0, 1.0, 0.0),
            expected="left_better",
            rationale="left clears all severe violations; right retains expected-support no-load",
        ),
        _case(
            case_id="no-severe-must-beat-remaining-survival-loss",
            noisy=Outcome(1.0, 0.0, 0.0),
            left=zero,
            right=Outcome(0.5, 0.0, 0.0),
            expected="left_better",
            rationale="left clears Survival loss; right retains half of the lost horizon",
        ),
        _case(
            case_id="no-severe-must-beat-remaining-unplanned-switch",
            noisy=Outcome(0.0, 0.0, 1.0),
            left=zero,
            right=Outcome(0.0, 0.0, 0.5),
            expected="left_better",
            rationale="left clears the switch violation; right retains half of it",
        ),
        _case(
            case_id="same-family-survival-pareto-improvement",
            noisy=Outcome(1.0, 0.0, 0.0),
            left=Outcome(0.0, 0.0, 0.0),
            right=Outcome(0.5, 0.0, 0.0),
            expected="left_better",
            rationale="left loses less survival horizon and is equal on all other severe items",
        ),
        _case(
            case_id="same-family-no-load-pareto-improvement",
            noisy=Outcome(0.0, 1.0, 0.0),
            left=Outcome(0.0, 0.0, 0.0),
            right=Outcome(0.0, 0.5, 0.0),
            expected="left_better",
            rationale="left has less no-load exposure and is equal on other severe items",
        ),
        _case(
            case_id="survival-improves-but-no-load-introduced",
            noisy=Outcome(1.0, 0.0, 0.0),
            left=Outcome(0.0, 1.0, 0.0),
            right=Outcome(1.0, 0.0, 0.0),
            expected="incomparable",
            rationale="left improves Survival but introduces no-load",
        ),
        _case(
            case_id="no-load-improves-but-switch-introduced",
            noisy=Outcome(0.0, 1.0, 0.0, intent_error=1.0),
            left=Outcome(0.0, 0.0, 1.0, intent_error=0.0),
            right=Outcome(0.0, 1.0, 0.0, intent_error=1.0),
            expected="incomparable",
            rationale="L decreases while W increases; H=L+W cancels and exposes ordinary score",
        ),
        _case(
            case_id="l-and-w-opposite-directions",
            noisy=Outcome(0.0, 0.6, 0.4, drift=1.0),
            left=Outcome(0.0, 0.2, 0.8, drift=0.0),
            right=Outcome(0.0, 0.8, 0.2, drift=1.0),
            expected="incomparable",
            rationale="both H values equal one while L and W trade in opposite directions",
        ),
        _case(
            case_id="ordinary-pareto-after-severe-clear",
            noisy=Outcome(0.0, 0.0, 0.0, ordinary_contact=1.0, intent_error=1.0, drift=1.0, zmp=1.0, repair_cost=1.0),
            left=Outcome(0.0, 0.0, 0.0, ordinary_contact=0.2, intent_error=0.2, drift=0.2, zmp=0.2, repair_cost=0.2),
            right=Outcome(0.0, 0.0, 0.0, ordinary_contact=0.5, intent_error=0.5, drift=0.5, zmp=0.5, repair_cost=0.5),
            expected="left_better",
            rationale="both are severe-clear and left Pareto-dominates every ordinary item",
        ),
        _case(
            case_id="ordinary-tradeoff-remains-incomparable",
            noisy=Outcome(0.0, 0.0, 0.0, intent_error=1.0, drift=1.0),
            left=Outcome(0.0, 0.0, 0.0, intent_error=0.0, drift=1.4),
            right=Outcome(0.0, 0.0, 0.0, intent_error=1.0, drift=0.8),
            expected="incomparable",
            rationale="left improves Intent and worsens drift; no ordinary tradeoff weight is confirmed",
        ),
    ]
    cases.extend(_fail_closed_cases())

    violations = [case for case in cases if case["status"] == "OBJECTIVE-VIOLATION"]
    overall_status = "OBJECTIVE-VIOLATION" if violations else "OBJECTIVE-ALIGNED"
    if overall_status not in ALLOWED_STATUS:
        raise AssertionError("unexpected objective status")
    if not violations:
        raise AssertionError("candidate fixture no longer exposes its partial-order defect")
    return {
        "schema": "frontres-hierarchical-gain-partial-order-alignment/v1",
        "candidate": {
            "physics_tuple": "P_X_new=(S_X,H_X,C_X)",
            "hard_contact": "H_X=L_X+W_X",
            "survival_interval": "{-3,+3}",
            "severe_contact_interval": "(-2,-1) union (1,2)",
            "ordinary_interval": "(-1,1)",
        },
        "card_id": card["card_id"],
        "oracle": "human-confirmed partial order only",
        "second_scoring_formula_used": False,
        "sct": ["S1", "C1", "C2", "C3", "C4", "T-preference", "T-value", "T-mask"],
        "cases": cases,
        "status": overall_status,
        "first_violation": violations[0],
        "sensitivity": (
            "the current isolated candidate itself supplies the observed failing behavior; "
            "a checker that accepted the tied first pair would fail the confirmed strict preference"
        ),
        "active_boundary_modified": False,
    }


def main() -> None:
    print(json.dumps(run_alignment(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
