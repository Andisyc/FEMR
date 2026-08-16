#!/usr/bin/env python3
"""S1 alignment test for the isolated candidate relational Gain boundary."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "rsl_rl"))

from rsl_rl.frontres.frontres_hierarchical_gain_candidate import (  # noqa: E402
    Outcome,
    Thresholds,
    compare,
)


def _case(case_id: str, left: Outcome, right: Outcome, expected: str) -> dict[str, object]:
    observed = compare(left, right)
    inverse = compare(right, left)
    inverse_expected = {
        "BETTER": "WORSE",
        "WORSE": "BETTER",
        "SAME": "SAME",
        "INCOMPARABLE": "INCOMPARABLE",
        "INVALID": "INVALID",
    }[expected]
    return {
        "id": case_id,
        "expected": expected,
        "observed": asdict(observed),
        "inverse": asdict(inverse),
        "status": (
            "OBJECTIVE-ALIGNED"
            if observed.relation == expected and inverse.relation == inverse_expected
            else "OBJECTIVE-VIOLATION"
        ),
    }


def run_alignment() -> dict[str, object]:
    thresholds = Thresholds()
    stable = Outcome()
    l2 = replace(stable, stable_hold_steps=2)
    l1_no_load = replace(stable, expected_support_no_load=0.4)
    l1_switch = replace(stable, unplanned_support_switch=0.4)
    l0 = replace(stable, survival_ok=False, survival_failure_duration=0.5)
    cases = [
        _case("level-l3-over-l2", stable, l2, "BETTER"),
        _case("level-l2-over-l1", l2, l1_no_load, "BETTER"),
        _case("level-l1-over-l0", l1_no_load, l0, "BETTER"),
        _case("l1-pareto", replace(stable, expected_support_no_load=0.2), l1_no_load, "BETTER"),
        _case("l1-tradeoff", replace(stable, expected_support_no_load=0.2, unplanned_support_switch=0.8), replace(stable, expected_support_no_load=0.8, unplanned_support_switch=0.2), "INCOMPARABLE"),
        _case("l2-pareto", replace(l2, capture_margin=0.03, support_drift=0.02), replace(l2, capture_margin=0.02, support_drift=0.04), "BETTER"),
        _case("l2-tradeoff", replace(l2, capture_margin=0.03, linear_momentum_error=0.12), replace(l2, capture_margin=0.02, linear_momentum_error=0.08), "INCOMPARABLE"),
        _case("l0-internal-unconfirmed", replace(l0, survival_failure_duration=0.2), l0, "INCOMPARABLE"),
        _case("l3-intent", replace(stable, intent_error=(0.05, 0.05), repair_cost=0.3), stable, "BETTER"),
        _case("l3-cost-tiebreak", replace(stable, repair_cost=0.05), replace(stable, repair_cost=0.2), "BETTER"),
        _case("zmp-legitimate-na", replace(stable, zmp_applicable=False, zmp_margin=None, repair_cost=0.05), replace(stable, zmp_applicable=False, zmp_margin=None, repair_cost=0.2), "BETTER"),
        _case("zmp-domain-mismatch", stable, replace(stable, zmp_applicable=False, zmp_margin=None), "INVALID"),
        _case("missing-evidence", replace(stable, capture_margin=None), stable, "INVALID"),
        _case("invalid-evidence", replace(stable, angular_momentum_error=math.nan), stable, "INVALID"),
        _case("identity", stable, stable, "SAME"),
    ]
    base = -0.02
    step = 2.0 * thresholds.comparison_resolution
    a = replace(l2, capture_margin=base + 2.0 * step, capture_margin_trend=base)
    b = replace(l2, capture_margin=base + step, capture_margin_trend=base)
    c = replace(l2, capture_margin=base, capture_margin_trend=base)
    chain = {
        "A>B": compare(a, b).relation,
        "B>C": compare(b, c).relation,
        "A>C": compare(a, c).relation,
    }
    cases.append({
        "id": "pareto-transitivity",
        "expected": {"A>B": "BETTER", "B>C": "BETTER", "A>C": "BETTER"},
        "observed": chain,
        "status": "OBJECTIVE-ALIGNED" if set(chain.values()) == {"BETTER"} else "OBJECTIVE-VIOLATION",
    })
    violations = [case for case in cases if case["status"] == "OBJECTIVE-VIOLATION"]
    return {
        "schema": "frontres-hierarchical-gain-v2-public-alignment/v1",
        "card": "frontres-hierarchical-gain-v2-ordering-v1",
        "production_boundary": "rsl_rl.frontres.frontres_hierarchical_gain_candidate.compare",
        "oracle": "human-confirmed relations; no second score",
        "cases": cases,
        "status": "OBJECTIVE-VIOLATION" if violations else "OBJECTIVE-ALIGNED",
        "first_counterexample": violations[0] if violations else None,
        "active_gain_modified": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_alignment(), indent=2, sort_keys=True))
