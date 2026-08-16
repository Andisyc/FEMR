#!/usr/bin/env python3
"""Independent pseudo-data test for the proposed relational FrontRES Gain.

This file is intentionally outside the active Gain and training path.  It tests
one candidate interface:

    evidence -> L0/L1/L2/L3 -> BETTER/WORSE/SAME/INCOMPARABLE

Human-authored expected relations are the oracle.  They are not produced by a
second score.  Missing or invalid evidence fails the comparison closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RERUN2_PATH = (
    PROJECT_ROOT
    / "FRS_EVAL_V006_V024_ACTION_GAIN_DIRECTION_MODEL200_RERUN2_20260814.json"
)

ALLOWED_OBJECTIVE_STATUS = {
    "OBJECTIVE-ALIGNED",
    "OBJECTIVE-VIOLATION",
    "INCONCLUSIVE",
    "TELEMETRY-GAP",
}
INVERSE_RELATION = {
    "BETTER": "WORSE",
    "WORSE": "BETTER",
    "SAME": "SAME",
    "INCOMPARABLE": "INCOMPARABLE",
    "INVALID": "INVALID",
}


class EvidenceError(ValueError):
    """The relational Gain cannot make a valid comparison."""


class PhysicsLevel(IntEnum):
    L0_PHYSICS_FAILED = 0
    L1_CONTACT_INVALID = 1
    L2_ADMISSIBLE_UNSETTLED = 2
    L3_ADMISSIBLE_STABLE = 3


@dataclass(frozen=True)
class Thresholds:
    capture_margin_min: float = 0.02
    capture_margin_trend_min: float = 0.0
    zmp_margin_min: float = 0.01
    linear_momentum_error_max: float = 0.10
    angular_momentum_error_max: float = 0.10
    support_drift_max: float = 0.05
    stable_hold_steps_required: int = 4
    comparison_resolution: float = 1.0e-6


@dataclass(frozen=True)
class Outcome:
    """Minimal proposed telemetry for one K-step outcome.

    Errors and violation amounts are lower-is-better.  Margins and margin trend
    are higher-is-better.  ``zmp_margin=None`` is valid only when ZMP is
    explicitly not applicable for the planned support phase.
    """

    survival_ok: object = True
    survival_failure_duration: object = 0.0
    expected_support_no_load: object = 0.0
    unplanned_support_switch: object = 0.0
    illegal_contact_duration: object = 0.0
    capture_margin: object = 0.04
    capture_margin_trend: object = 0.01
    zmp_applicable: object = True
    zmp_margin: object = 0.03
    linear_momentum_error: object = 0.04
    angular_momentum_error: object = 0.04
    support_drift: object = 0.02
    stable_hold_steps: object = 4
    intent_error: object = (0.10, 0.10)
    repair_cost: object = 0.10


@dataclass(frozen=True)
class ClassifiedOutcome:
    level: PhysicsLevel
    severe_vector: tuple[float, float, float]
    recovery_vector: tuple[float, ...] | None
    intent_vector: tuple[float, ...] | None
    repair_cost: float | None
    zmp_applicable: bool | None
    survival_failure_duration: float


@dataclass(frozen=True)
class Comparison:
    relation: str
    left_level: str | None
    right_level: str | None
    reason: str


def _real(value: object, *, name: str, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{name} must be finite")
    if non_negative and result < 0.0:
        raise EvidenceError(f"{name} must be non-negative")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise EvidenceError(f"{name} must be boolean")
    return value


def _steps(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{name} must be a non-negative integer")
    return value


def _intent_vector(value: object) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise EvidenceError("intent_error must be a non-empty sequence")
    return tuple(-_real(item, name=f"intent_error[{index}]", non_negative=True) for index, item in enumerate(value))


def _pareto_higher_better(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    resolution: float,
) -> str:
    if len(left) != len(right) or not left:
        raise EvidenceError("Pareto vectors must have the same non-zero dimension")
    left_bins = tuple(_quantize(value, resolution) for value in left)
    right_bins = tuple(_quantize(value, resolution) for value in right)
    if left_bins == right_bins:
        return "SAME"
    if all(left_value >= right_value for left_value, right_value in zip(left_bins, right_bins, strict=True)):
        return "BETTER"
    if all(right_value >= left_value for left_value, right_value in zip(left_bins, right_bins, strict=True)):
        return "WORSE"
    return "INCOMPARABLE"


def _quantize(value: float, resolution: float) -> int:
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise EvidenceError("comparison_resolution must be finite and positive")
    return math.floor(value / resolution + 0.5)


def _at_least(value: float, threshold: float, resolution: float) -> bool:
    return _quantize(value, resolution) >= _quantize(threshold, resolution)


def _at_most(value: float, threshold: float, resolution: float) -> bool:
    return _quantize(value, resolution) <= _quantize(threshold, resolution)


def classify(outcome: Outcome, thresholds: Thresholds) -> ClassifiedOutcome:
    """Map raw evidence to the proposed Physics recovery level."""

    survival_ok = _boolean(outcome.survival_ok, name="survival_ok")
    failure_duration = _real(
        outcome.survival_failure_duration,
        name="survival_failure_duration",
        non_negative=True,
    )
    resolution = thresholds.comparison_resolution
    failure_bin = _quantize(failure_duration, resolution)
    if survival_ok and failure_bin != 0:
        raise EvidenceError("survival_ok conflicts with survival_failure_duration")
    if not survival_ok and failure_bin <= 0:
        raise EvidenceError("failed survival requires positive failure duration")

    no_load = _real(
        outcome.expected_support_no_load,
        name="expected_support_no_load",
        non_negative=True,
    )
    switch = _real(
        outcome.unplanned_support_switch,
        name="unplanned_support_switch",
        non_negative=True,
    )
    illegal_contact = _real(
        outcome.illegal_contact_duration,
        name="illegal_contact_duration",
        non_negative=True,
    )
    severe = (no_load, switch, illegal_contact)

    if not survival_ok:
        return ClassifiedOutcome(
            level=PhysicsLevel.L0_PHYSICS_FAILED,
            severe_vector=severe,
            recovery_vector=None,
            intent_vector=None,
            repair_cost=None,
            zmp_applicable=None,
            survival_failure_duration=failure_duration,
        )

    if any(_quantize(value, resolution) > 0 for value in severe):
        return ClassifiedOutcome(
            level=PhysicsLevel.L1_CONTACT_INVALID,
            severe_vector=severe,
            recovery_vector=None,
            intent_vector=None,
            repair_cost=None,
            zmp_applicable=None,
            survival_failure_duration=failure_duration,
        )

    capture_margin = _real(outcome.capture_margin, name="capture_margin")
    capture_trend = _real(outcome.capture_margin_trend, name="capture_margin_trend")
    zmp_applicable = _boolean(outcome.zmp_applicable, name="zmp_applicable")
    if zmp_applicable:
        zmp_margin = _real(outcome.zmp_margin, name="zmp_margin")
    else:
        if outcome.zmp_margin is not None:
            raise EvidenceError("planned ZMP N/A must be represented by zmp_margin=None")
        zmp_margin = None
    linear_error = _real(
        outcome.linear_momentum_error,
        name="linear_momentum_error",
        non_negative=True,
    )
    angular_error = _real(
        outcome.angular_momentum_error,
        name="angular_momentum_error",
        non_negative=True,
    )
    drift = _real(outcome.support_drift, name="support_drift", non_negative=True)
    hold_steps = _steps(outcome.stable_hold_steps, name="stable_hold_steps")
    intent = _intent_vector(outcome.intent_error)
    repair_cost = _real(outcome.repair_cost, name="repair_cost", non_negative=True)

    stable = (
        _at_least(capture_margin, thresholds.capture_margin_min, resolution)
        and _at_least(capture_trend, thresholds.capture_margin_trend_min, resolution)
        and (
            not zmp_applicable
            or (
                zmp_margin is not None
                and _at_least(zmp_margin, thresholds.zmp_margin_min, resolution)
            )
        )
        and _at_most(linear_error, thresholds.linear_momentum_error_max, resolution)
        and _at_most(angular_error, thresholds.angular_momentum_error_max, resolution)
        and _at_most(drift, thresholds.support_drift_max, resolution)
        and hold_steps >= thresholds.stable_hold_steps_required
    )
    recovery = (
        capture_margin,
        capture_trend,
        *(() if zmp_margin is None else (zmp_margin,)),
        -linear_error,
        -angular_error,
        -drift,
    )
    return ClassifiedOutcome(
        level=(
            PhysicsLevel.L3_ADMISSIBLE_STABLE
            if stable
            else PhysicsLevel.L2_ADMISSIBLE_UNSETTLED
        ),
        severe_vector=severe,
        recovery_vector=recovery,
        intent_vector=intent,
        repair_cost=repair_cost,
        zmp_applicable=zmp_applicable,
        survival_failure_duration=failure_duration,
    )


def compare(
    left: Outcome,
    right: Outcome,
    thresholds: Thresholds = Thresholds(),
) -> Comparison:
    """Compare two Repairs without collapsing evidence into a scalar."""

    try:
        left_classified = classify(left, thresholds)
        right_classified = classify(right, thresholds)
    except EvidenceError as error:
        return Comparison("INVALID", None, None, str(error))

    left_level = left_classified.level
    right_level = right_classified.level
    if (
        left_level >= PhysicsLevel.L2_ADMISSIBLE_UNSETTLED
        and right_level >= PhysicsLevel.L2_ADMISSIBLE_UNSETTLED
        and left_classified.zmp_applicable != right_classified.zmp_applicable
    ):
        return Comparison(
            "INVALID",
            left_level.name,
            right_level.name,
            "same-Scenario outcomes have different planned ZMP applicability",
        )
    if left_level > right_level:
        return Comparison("BETTER", left_level.name, right_level.name, "higher Physics recovery level")
    if left_level < right_level:
        return Comparison("WORSE", left_level.name, right_level.name, "lower Physics recovery level")

    resolution = thresholds.comparison_resolution
    if left_level == PhysicsLevel.L0_PHYSICS_FAILED:
        same_failure_bin = _quantize(
            left_classified.survival_failure_duration, resolution
        ) == _quantize(right_classified.survival_failure_duration, resolution)
        relation = "SAME" if same_failure_bin else "INCOMPARABLE"
        return Comparison(
            relation,
            left_level.name,
            right_level.name,
            "L0 internal severity ordering is not yet human-confirmed",
        )

    if left_level == PhysicsLevel.L1_CONTACT_INVALID:
        relation = _pareto_higher_better(
            tuple(-value for value in left_classified.severe_vector),
            tuple(-value for value in right_classified.severe_vector),
            resolution=resolution,
        )
        return Comparison(relation, left_level.name, right_level.name, "severe-contact Pareto relation")

    if left_level == PhysicsLevel.L2_ADMISSIBLE_UNSETTLED:
        assert left_classified.recovery_vector is not None
        assert right_classified.recovery_vector is not None
        relation = _pareto_higher_better(
            left_classified.recovery_vector,
            right_classified.recovery_vector,
            resolution=resolution,
        )
        return Comparison(relation, left_level.name, right_level.name, "unsettled recovery Pareto relation")

    assert left_classified.intent_vector is not None
    assert right_classified.intent_vector is not None
    intent_relation = _pareto_higher_better(
        left_classified.intent_vector,
        right_classified.intent_vector,
        resolution=resolution,
    )
    if intent_relation != "SAME":
        return Comparison(intent_relation, left_level.name, right_level.name, "stable-domain Intent Pareto relation")
    assert left_classified.repair_cost is not None
    assert right_classified.repair_cost is not None
    cost_relation = _pareto_higher_better(
        (-left_classified.repair_cost,),
        (-right_classified.repair_cost,),
        resolution=resolution,
    )
    return Comparison(cost_relation, left_level.name, right_level.name, "Intent SAME; repair-cost tie break")


def _replace(outcome: Outcome, **changes: object) -> Outcome:
    values = asdict(outcome)
    values.update(changes)
    return Outcome(**values)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _test_case(
    case_id: str,
    left: Outcome,
    right: Outcome,
    expected: str,
    rationale: str,
    thresholds: Thresholds,
) -> dict[str, Any]:
    observed = compare(left, right, thresholds)
    inverse = compare(right, left, thresholds)
    relation_aligned = observed.relation == expected
    inverse_aligned = inverse.relation == INVERSE_RELATION[expected]
    return {
        "id": case_id,
        "rationale": rationale,
        "expected_relation": expected,
        "observed": asdict(observed),
        "inverse_observed": asdict(inverse),
        "left": _json_safe(asdict(left)),
        "right": _json_safe(asdict(right)),
        "status": (
            "OBJECTIVE-ALIGNED"
            if relation_aligned and inverse_aligned
            else "OBJECTIVE-VIOLATION"
        ),
    }


def _mutant_collapsed_l1(left: Outcome, right: Outcome) -> str:
    """Deliberately wrong sensitivity mutant: collapse L/W/contact by summing."""

    left_total = sum(
        float(value)
        for value in (
            left.expected_support_no_load,
            left.unplanned_support_switch,
            left.illegal_contact_duration,
        )
    )
    right_total = sum(
        float(value)
        for value in (
            right.expected_support_no_load,
            right.unplanned_support_switch,
            right.illegal_contact_duration,
        )
    )
    if left_total < right_total:
        return "BETTER"
    if right_total < left_total:
        return "WORSE"
    return "SAME"


def _mutant_pairwise_tolerance(
    left: tuple[float, ...],
    right: tuple[float, ...],
    tolerance: float,
) -> str:
    """Deliberately wrong mutant that can make BETTER non-transitive."""

    left_strict = any(
        left_value > right_value + tolerance
        for left_value, right_value in zip(left, right, strict=True)
    )
    right_strict = any(
        right_value > left_value + tolerance
        for left_value, right_value in zip(left, right, strict=True)
    )
    if left_strict and right_strict:
        return "INCOMPARABLE"
    if left_strict:
        return "BETTER"
    if right_strict:
        return "WORSE"
    return "SAME"


def _pseudo_cases(thresholds: Thresholds) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stable = Outcome()
    l0_failed = _replace(stable, survival_ok=False, survival_failure_duration=0.5)
    l1_no_load = _replace(stable, expected_support_no_load=0.4)
    l1_less_no_load = _replace(stable, expected_support_no_load=0.2)
    l1_switch = _replace(stable, unplanned_support_switch=0.4)
    l1_mixed_left = _replace(stable, expected_support_no_load=0.2, unplanned_support_switch=0.8)
    l1_mixed_right = _replace(stable, expected_support_no_load=0.8, unplanned_support_switch=0.2)
    l2_base = _replace(stable, stable_hold_steps=2)

    cases = [
        _test_case(
            "unsettled-no-severe-vs-severe",
            l2_base,
            l1_no_load,
            "BETTER",
            "clearing severe violations reaches L2 even before stable recovery",
            thresholds,
        ),
        _test_case(
            "no-severe-vs-any-severe",
            stable,
            l1_no_load,
            "BETTER",
            "an admissible Repair must beat one with expected-support no-load",
            thresholds,
        ),
        _test_case(
            "same-severe-family-pareto",
            l1_less_no_load,
            l1_no_load,
            "BETTER",
            "both are L1 and the left has strictly less no-load with no new violation",
            thresholds,
        ),
        _test_case(
            "survival-improves-but-introduces-no-load",
            l1_no_load,
            l0_failed,
            "BETTER",
            "surviving with a contact violation is L1 and outranks survival failure L0",
            thresholds,
        ),
        _test_case(
            "l0-internal-severity-unconfirmed",
            _replace(stable, survival_ok=False, survival_failure_duration=0.25),
            l0_failed,
            "INCOMPARABLE",
            "the approved hierarchy orders L1 above L0 but does not rank two L0 failures",
            thresholds,
        ),
        _test_case(
            "no-load-improves-but-switch-introduced",
            l1_switch,
            l1_no_load,
            "INCOMPARABLE",
            "different severe-contact violations cannot compensate each other",
            thresholds,
        ),
        _test_case(
            "L-and-W-opposite-directions",
            l1_mixed_left,
            l1_mixed_right,
            "INCOMPARABLE",
            "lower L and higher W is a Pareto tradeoff, not a scalar tie",
            thresholds,
        ),
        _test_case(
            "unsettled-physics-pareto",
            _replace(
                l2_base,
                capture_margin=0.015,
                capture_margin_trend=-0.01,
                linear_momentum_error=0.12,
                angular_momentum_error=0.12,
                support_drift=0.07,
            ),
            _replace(
                l2_base,
                capture_margin=-0.01,
                capture_margin_trend=-0.03,
                linear_momentum_error=0.20,
                angular_momentum_error=0.20,
                support_drift=0.10,
            ),
            "BETTER",
            "after severe violations clear, every recovery coordinate improves",
            thresholds,
        ),
        _test_case(
            "unsettled-ordinary-tradeoff",
            _replace(l2_base, capture_margin=-0.01, linear_momentum_error=0.12),
            _replace(l2_base, capture_margin=-0.02, linear_momentum_error=0.08),
            "INCOMPARABLE",
            "capture margin improves while momentum error worsens",
            thresholds,
        ),
        _test_case(
            "stable-intent-pareto",
            _replace(stable, intent_error=(0.05, 0.05), repair_cost=0.30),
            _replace(stable, intent_error=(0.10, 0.10), repair_cost=0.01),
            "BETTER",
            "inside L3, Pareto-better Intent outranks cost",
            thresholds,
        ),
        _test_case(
            "stable-physics-surplus-does-not-rank",
            _replace(stable, capture_margin=0.50, zmp_margin=0.30),
            stable,
            "SAME",
            "once both outcomes are L3, unconfirmed surplus stability is not a hidden score",
            thresholds,
        ),
        _test_case(
            "stable-intent-beats-physics-surplus",
            _replace(stable, capture_margin=0.04, intent_error=(0.05, 0.05)),
            _replace(stable, capture_margin=0.50, zmp_margin=0.30, intent_error=(0.10, 0.10)),
            "BETTER",
            "inside L3, Intent orders outcomes instead of surplus Physics margin",
            thresholds,
        ),
        _test_case(
            "stable-intent-tradeoff",
            _replace(stable, intent_error=(0.05, 0.20)),
            _replace(stable, intent_error=(0.10, 0.10)),
            "INCOMPARABLE",
            "unconfirmed exchanges between Intent coordinates remain incomparable",
            thresholds,
        ),
        _test_case(
            "repair-cost-only-after-intent-same",
            _replace(stable, intent_error=(0.10, 0.10), repair_cost=0.05),
            _replace(stable, intent_error=(0.10, 0.10), repair_cost=0.20),
            "BETTER",
            "cost breaks a tie only after stable-domain Intent is SAME",
            thresholds,
        ),
        _test_case(
            "planned-zmp-na",
            _replace(stable, zmp_applicable=False, zmp_margin=None, repair_cost=0.05),
            _replace(stable, zmp_applicable=False, zmp_margin=None, repair_cost=0.20),
            "BETTER",
            "planned ZMP N/A is valid evidence and cost resolves equal L3 Intent",
            thresholds,
        ),
        _test_case(
            "zmp-applicability-domain-mismatch",
            stable,
            _replace(stable, zmp_applicable=False, zmp_margin=None),
            "INVALID",
            "same-Scenario Repairs must share the Clean-planned ZMP evidence domain",
            thresholds,
        ),
        _test_case(
            "zmp-applicability-domain-mismatch-across-levels",
            stable,
            _replace(l2_base, zmp_applicable=False, zmp_margin=None),
            "INVALID",
            "a level difference cannot hide a same-Scenario ZMP evidence-domain mismatch",
            thresholds,
        ),
        _test_case(
            "identity",
            stable,
            stable,
            "SAME",
            "identical valid evidence must compare SAME",
            thresholds,
        ),
        _test_case(
            "same-quantization-bin",
            _replace(stable, repair_cost=0.10),
            _replace(
                stable,
                repair_cost=0.10 + thresholds.comparison_resolution / 4.0,
            ),
            "SAME",
            "evidence in the same global resolution bin cannot create a preference",
            thresholds,
        ),
        _test_case(
            "missing-capture-evidence",
            _replace(stable, capture_margin=None),
            stable,
            "INVALID",
            "missing evidence fails closed",
            thresholds,
        ),
        _test_case(
            "invalid-nan-evidence",
            _replace(stable, angular_momentum_error=float("nan")),
            stable,
            "INVALID",
            "non-finite evidence fails closed",
            thresholds,
        ),
        _test_case(
            "invalid-applicable-zmp-na",
            _replace(stable, zmp_applicable=True, zmp_margin=None),
            stable,
            "INVALID",
            "applicable ZMP cannot be silently replaced by N/A",
            thresholds,
        ),
    ]

    boundary_delta = 2.0 * thresholds.comparison_resolution
    boundary_cases = (
        (
            "capture-margin-boundary",
            {"capture_margin": thresholds.capture_margin_min},
            {"capture_margin": thresholds.capture_margin_min - boundary_delta},
        ),
        (
            "capture-trend-boundary",
            {"capture_margin_trend": thresholds.capture_margin_trend_min},
            {"capture_margin_trend": thresholds.capture_margin_trend_min - boundary_delta},
        ),
        (
            "zmp-margin-boundary",
            {"zmp_margin": thresholds.zmp_margin_min},
            {"zmp_margin": thresholds.zmp_margin_min - boundary_delta},
        ),
        (
            "linear-momentum-boundary",
            {"linear_momentum_error": thresholds.linear_momentum_error_max},
            {"linear_momentum_error": thresholds.linear_momentum_error_max + boundary_delta},
        ),
        (
            "angular-momentum-boundary",
            {"angular_momentum_error": thresholds.angular_momentum_error_max},
            {"angular_momentum_error": thresholds.angular_momentum_error_max + boundary_delta},
        ),
        (
            "support-drift-boundary",
            {"support_drift": thresholds.support_drift_max},
            {"support_drift": thresholds.support_drift_max + boundary_delta},
        ),
        (
            "stable-hold-boundary",
            {"stable_hold_steps": thresholds.stable_hold_steps_required},
            {"stable_hold_steps": thresholds.stable_hold_steps_required - 1},
        ),
    )
    for case_id, passing_change, failing_change in boundary_cases:
        cases.append(
            _test_case(
                case_id,
                _replace(stable, **passing_change),
                _replace(stable, **failing_change),
                "BETTER",
                "the exact inclusive boundary is L3 and the isolated failing side is L2",
                thresholds,
            )
        )

    chain_base = -0.02
    chain_step = 2.0 * thresholds.comparison_resolution
    chain_a = _replace(
        l2_base,
        capture_margin=chain_base + 2.0 * chain_step,
        capture_margin_trend=chain_base,
    )
    chain_b = _replace(
        l2_base,
        capture_margin=chain_base + chain_step,
        capture_margin_trend=chain_base,
    )
    chain_c = _replace(
        l2_base,
        capture_margin=chain_base,
        capture_margin_trend=chain_base,
    )
    chain_relations = {
        "A>B": compare(chain_a, chain_b, thresholds).relation,
        "B>C": compare(chain_b, chain_c, thresholds).relation,
        "A>C": compare(chain_a, chain_c, thresholds).relation,
    }
    cases.append(
        {
            "id": "pareto-transitivity",
            "rationale": "global evidence bins followed by exact Pareto must preserve a strict chain",
            "expected_relation": {"A>B": "BETTER", "B>C": "BETTER", "A>C": "BETTER"},
            "observed": chain_relations,
            "status": (
                "OBJECTIVE-ALIGNED"
                if set(chain_relations.values()) == {"BETTER"}
                else "OBJECTIVE-VIOLATION"
            ),
        }
    )

    mutant_observed = _mutant_collapsed_l1(l1_mixed_left, l1_mixed_right)
    resolution = thresholds.comparison_resolution
    tolerance_a = (1.1 * resolution, -0.6 * resolution)
    tolerance_b = (0.0, 0.0)
    tolerance_c = (-1.1 * resolution, 0.6 * resolution)
    tolerance_mutant = {
        "A>B": _mutant_pairwise_tolerance(tolerance_a, tolerance_b, resolution),
        "B>C": _mutant_pairwise_tolerance(tolerance_b, tolerance_c, resolution),
        "A>C": _mutant_pairwise_tolerance(tolerance_a, tolerance_c, resolution),
    }
    sum_mutant_status = (
        "OBJECTIVE-ALIGNED"
        if mutant_observed != "INCOMPARABLE"
        else "OBJECTIVE-VIOLATION"
    )
    tolerance_mutant_status = (
        "OBJECTIVE-ALIGNED"
        if tolerance_mutant == {
            "A>B": "BETTER",
            "B>C": "BETTER",
            "A>C": "INCOMPARABLE",
        }
        else "OBJECTIVE-VIOLATION"
    )
    sensitivity = {
        "sum_collapse": {
            "mutation": "collapse L/W/illegal-contact into one sum",
            "expected_relation": "INCOMPARABLE",
            "mutant_relation": mutant_observed,
            "left_severe_values": [0.2, 0.8, 0.0],
            "right_severe_values": [0.8, 0.2, 0.0],
            "status": sum_mutant_status,
        },
        "pairwise_tolerance": {
            "mutation": "compare raw floats with a pair-specific tolerance",
            "mutant_relations": tolerance_mutant,
            "defect": "A>B and B>C but A and C incomparable",
            "status": tolerance_mutant_status,
        },
        "status": (
            "OBJECTIVE-ALIGNED"
            if sum_mutant_status == tolerance_mutant_status == "OBJECTIVE-ALIGNED"
            else "OBJECTIVE-VIOLATION"
        ),
    }
    return cases, sensitivity


def _rerun2_telemetry_check(path: Path = RERUN2_PATH) -> dict[str, Any]:
    required = {
        "survival_ok",
        "survival_failure_duration",
        "expected_support_no_load",
        "unplanned_support_switch",
        "illegal_contact_duration",
        "capture_margin",
        "capture_margin_trend",
        "zmp_applicable",
        "zmp_margin",
        "linear_momentum_error",
        "angular_momentum_error",
        "support_drift",
        "stable_hold_steps",
        "intent_error",
        "repair_cost",
    }
    if not path.exists():
        return {
            "artifact": str(path),
            "status": "TELEMETRY-GAP",
            "reason": "RERUN2 artifact is not present",
            "missing_fields": sorted(required),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [row for scenario in payload["scenarios"] for row in scenario["rows"]]
        component_keys = set(rows[0]["components"]) if rows else set()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "artifact": str(path),
            "status": "TELEMETRY-GAP",
            "reason": f"malformed or unsupported RERUN2 schema: {error}",
            "missing_fields": sorted(required),
        }
    missing = sorted(required - component_keys)
    return {
        "artifact": str(path),
        "artifact_schema": payload.get("schema"),
        "row_count": len(rows),
        "available_component_fields": sorted(component_keys),
        "missing_fields": missing,
        "status": "TELEMETRY-GAP" if missing else "INCONCLUSIVE",
        "reason": (
            "existing aggregate Physics channels cannot be inverted into the new raw level evidence"
            if missing
            else "raw fields exist, but this pseudo-data test does not establish runtime provenance"
        ),
    }


def run_alignment() -> dict[str, Any]:
    thresholds = Thresholds()
    cases, sensitivity = _pseudo_cases(thresholds)
    violations = [case for case in cases if case["status"] == "OBJECTIVE-VIOLATION"]
    if sensitivity["status"] == "OBJECTIVE-VIOLATION":
        violations.append(sensitivity)
    status = "OBJECTIVE-VIOLATION" if violations else "OBJECTIVE-ALIGNED"
    if status not in ALLOWED_OBJECTIVE_STATUS:
        raise AssertionError(f"unsupported objective status: {status}")
    report = {
        "schema": "frontres-hierarchical-relational-gain-offline/v2",
        "test_boundary": "independent pseudo-data; no active Gain, Contract, or training import",
        "candidate_interface": "evidence -> L0/L1/L2/L3 -> relation",
        "relations": ["BETTER", "WORSE", "SAME", "INCOMPARABLE"],
        "invalid_evidence_behavior": "fail closed as INVALID",
        "oracle": "human-authored expected relations; no second score",
        "candidate_assumptions": {
            "threshold_values": "provisional pseudo-data values; not calibrated",
            "l0_internal_order": "unconfirmed differences remain INCOMPARABLE",
            "zmp_na": "valid only when Clean-planned applicability is explicitly false",
            "numeric_uncertainty": "global quantization before exact Pareto",
        },
        "thresholds": asdict(thresholds),
        "cases": cases,
        "sensitivity": sensitivity,
        "status": status,
        "first_counterexample": violations[0] if violations else None,
        "rerun2_adapter": _rerun2_telemetry_check(),
        "active_boundary_modified": False,
    }
    return report


def main() -> None:
    report = run_alignment()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["status"] == "OBJECTIVE-VIOLATION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
