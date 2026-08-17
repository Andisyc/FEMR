"""Candidate relational Gain boundary; not imported by active training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Sequence


class EvidenceError(ValueError):
    """Evidence cannot support a fail-open comparison."""


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
class _Classified:
    level: PhysicsLevel
    severe: tuple[float, float, float]
    recovery: tuple[float, ...] | None
    intent: tuple[float, ...] | None
    cost: float | None
    zmp_applicable: bool | None
    failure_duration: float


@dataclass(frozen=True)
class Comparison:
    relation: str
    left_level: str | None
    right_level: str | None
    reason: str


@dataclass(frozen=True)
class RelationalTrainingBatch:
    """Candidate Actor pairwise-credit carrier; not an active PPO target."""

    status: str
    pair_relations: tuple[tuple[str, ...], ...]
    dominance_credit: tuple[float | None, ...]
    comparable_pair_count: tuple[int, ...]
    actor_credit_mask: tuple[bool, ...]
    reason: str
    preference_edges: tuple[tuple[int, int], ...] = ()


def _real(value: object, name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (non_negative and result < 0.0):
        raise EvidenceError(f"{name} is invalid")
    return result


def _quantize(value: float, resolution: float) -> int:
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise EvidenceError("comparison_resolution must be positive")
    return math.floor(value / resolution + 0.5)


def _pareto(left: tuple[float, ...], right: tuple[float, ...], resolution: float) -> str:
    if len(left) != len(right) or not left:
        raise EvidenceError("Pareto dimensions must match")
    left_q = tuple(_quantize(value, resolution) for value in left)
    right_q = tuple(_quantize(value, resolution) for value in right)
    if left_q == right_q:
        return "SAME"
    if all(a >= b for a, b in zip(left_q, right_q, strict=True)):
        return "BETTER"
    if all(b >= a for a, b in zip(left_q, right_q, strict=True)):
        return "WORSE"
    return "INCOMPARABLE"


def _intent(value: object) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise EvidenceError("intent_error must be non-empty")
    return tuple(-_real(item, f"intent_error[{i}]", non_negative=True) for i, item in enumerate(value))


def _classify(outcome: Outcome, thresholds: Thresholds) -> _Classified:
    resolution = thresholds.comparison_resolution
    if not isinstance(outcome.survival_ok, bool):
        raise EvidenceError("survival_ok must be boolean")
    failure = _real(outcome.survival_failure_duration, "survival_failure_duration", non_negative=True)
    failure_bin = _quantize(failure, resolution)
    if outcome.survival_ok != (failure_bin == 0):
        raise EvidenceError("survival evidence is inconsistent")
    severe = tuple(
        _real(getattr(outcome, name), name, non_negative=True)
        for name in (
            "expected_support_no_load",
            "unplanned_support_switch",
            "illegal_contact_duration",
        )
    )
    if not outcome.survival_ok:
        return _Classified(PhysicsLevel.L0_PHYSICS_FAILED, severe, None, None, None, None, failure)
    if any(_quantize(value, resolution) > 0 for value in severe):
        return _Classified(PhysicsLevel.L1_CONTACT_INVALID, severe, None, None, None, None, failure)

    capture = _real(outcome.capture_margin, "capture_margin")
    trend = _real(outcome.capture_margin_trend, "capture_margin_trend")
    if not isinstance(outcome.zmp_applicable, bool):
        raise EvidenceError("zmp_applicable must be boolean")
    if outcome.zmp_applicable:
        zmp = _real(outcome.zmp_margin, "zmp_margin")
    else:
        if outcome.zmp_margin is not None:
            raise EvidenceError("non-applicable ZMP must be None")
        zmp = None
    linear = _real(outcome.linear_momentum_error, "linear_momentum_error", non_negative=True)
    angular = _real(outcome.angular_momentum_error, "angular_momentum_error", non_negative=True)
    drift = _real(outcome.support_drift, "support_drift", non_negative=True)
    hold = outcome.stable_hold_steps
    if isinstance(hold, bool) or not isinstance(hold, int) or hold < 0:
        raise EvidenceError("stable_hold_steps is invalid")
    intent = _intent(outcome.intent_error)
    cost = _real(outcome.repair_cost, "repair_cost", non_negative=True)
    stable = (
        _quantize(capture, resolution) >= _quantize(thresholds.capture_margin_min, resolution)
        and _quantize(trend, resolution) >= _quantize(thresholds.capture_margin_trend_min, resolution)
        and (zmp is None or _quantize(zmp, resolution) >= _quantize(thresholds.zmp_margin_min, resolution))
        and _quantize(linear, resolution) <= _quantize(thresholds.linear_momentum_error_max, resolution)
        and _quantize(angular, resolution) <= _quantize(thresholds.angular_momentum_error_max, resolution)
        and _quantize(drift, resolution) <= _quantize(thresholds.support_drift_max, resolution)
        and hold >= thresholds.stable_hold_steps_required
    )
    recovery = (capture, trend, *(() if zmp is None else (zmp,)), -linear, -angular, -drift)
    level = PhysicsLevel.L3_ADMISSIBLE_STABLE if stable else PhysicsLevel.L2_ADMISSIBLE_UNSETTLED
    return _Classified(level, severe, recovery, intent, cost, outcome.zmp_applicable, failure)


def compare(left: Outcome, right: Outcome, thresholds: Thresholds = Thresholds()) -> Comparison:
    """Public candidate boundary: evidence to a relational outcome."""
    try:
        left_c = _classify(left, thresholds)
        right_c = _classify(right, thresholds)
    except EvidenceError as error:
        return Comparison("INVALID", None, None, str(error))
    if (
        left_c.level >= PhysicsLevel.L2_ADMISSIBLE_UNSETTLED
        and right_c.level >= PhysicsLevel.L2_ADMISSIBLE_UNSETTLED
        and left_c.zmp_applicable != right_c.zmp_applicable
    ):
        return Comparison("INVALID", left_c.level.name, right_c.level.name, "ZMP applicability mismatch")
    if left_c.level > right_c.level:
        return Comparison("BETTER", left_c.level.name, right_c.level.name, "higher Physics level")
    if left_c.level < right_c.level:
        return Comparison("WORSE", left_c.level.name, right_c.level.name, "lower Physics level")
    resolution = thresholds.comparison_resolution
    if left_c.level == PhysicsLevel.L0_PHYSICS_FAILED:
        relation = "SAME" if _quantize(left_c.failure_duration, resolution) == _quantize(right_c.failure_duration, resolution) else "INCOMPARABLE"
        return Comparison(relation, left_c.level.name, right_c.level.name, "L0 internal order unconfirmed")
    if left_c.level == PhysicsLevel.L1_CONTACT_INVALID:
        relation = _pareto(tuple(-value for value in left_c.severe), tuple(-value for value in right_c.severe), resolution)
        return Comparison(relation, left_c.level.name, right_c.level.name, "severe-contact Pareto")
    if left_c.level == PhysicsLevel.L2_ADMISSIBLE_UNSETTLED:
        assert left_c.recovery is not None and right_c.recovery is not None
        relation = _pareto(left_c.recovery, right_c.recovery, resolution)
        return Comparison(relation, left_c.level.name, right_c.level.name, "Recovery Pareto")
    assert left_c.intent is not None and right_c.intent is not None
    relation = _pareto(left_c.intent, right_c.intent, resolution)
    if relation != "SAME":
        return Comparison(relation, left_c.level.name, right_c.level.name, "stable Intent Pareto")
    assert left_c.cost is not None and right_c.cost is not None
    relation = _pareto((-left_c.cost,), (-right_c.cost,), resolution)
    return Comparison(relation, left_c.level.name, right_c.level.name, "Intent SAME; cost tie-break")


def build_relational_training_batch(
    outcomes: Sequence[Outcome], thresholds: Thresholds = Thresholds()
) -> RelationalTrainingBatch:
    """Build pairwise credit without scalarizing incomparable outcomes.

    BETTER/WORSE pairs contribute +/-1. SAME and INCOMPARABLE pairs contribute
    no direction and therefore do not create Actor credit by themselves. This
    adapter intentionally does not define a Critic target: a state-value target
    cannot be inferred from a partial order without another semantic decision.
    This is a candidate adapter for offline tests, not the active PPO contract.
    """

    if len(outcomes) < 2:
        return RelationalTrainingBatch("INVALID", (), (), (), (), "at least two outcomes are required")
    matrix: list[list[str]] = [["SAME" for _ in outcomes] for _ in outcomes]
    wins = [0 for _ in outcomes]
    losses = [0 for _ in outcomes]
    comparable = [0 for _ in outcomes]
    preference_edges: list[tuple[int, int]] = []
    for left_index in range(len(outcomes)):
        for right_index in range(left_index + 1, len(outcomes)):
            result = compare(outcomes[left_index], outcomes[right_index], thresholds)
            matrix[left_index][right_index] = result.relation
            reverse = {
                "BETTER": "WORSE",
                "WORSE": "BETTER",
                "SAME": "SAME",
                "INCOMPARABLE": "INCOMPARABLE",
                "INVALID": "INVALID",
            }[result.relation]
            matrix[right_index][left_index] = reverse
            if result.relation == "INVALID":
                return RelationalTrainingBatch(
                    "INVALID",
                    tuple(tuple(row) for row in matrix),
                    tuple(None for _ in outcomes),
                    tuple(comparable),
                    tuple(False for _ in outcomes),
                    "invalid evidence cannot enter relational training",
                )
            if result.relation == "BETTER":
                preference_edges.append((left_index, right_index))
                wins[left_index] += 1
                losses[right_index] += 1
                comparable[left_index] += 1
                comparable[right_index] += 1
            elif result.relation == "WORSE":
                preference_edges.append((right_index, left_index))
                losses[left_index] += 1
                wins[right_index] += 1
                comparable[left_index] += 1
                comparable[right_index] += 1
    # Keep edge incidence integer-valued. The PPO owner normalizes the sum of
    # edge losses by the number of valid preference edges; normalizing each row
    # here would give the two ends of one edge different mass.
    credits = tuple(
        None if comparable[index] == 0 else float(wins[index] - losses[index])
        for index in range(len(outcomes))
    )
    masks = tuple(value is not None for value in credits)
    return RelationalTrainingBatch(
        "READY" if any(masks) else "NO_COMPARABLE_PAIRS",
        tuple(tuple(row) for row in matrix),
        credits,
        tuple(comparable),
        masks,
        "pairwise dominance credit; SAME/INCOMPARABLE are not scalarized",
        tuple(preference_edges),
    )


__all__ = (
    "Comparison",
    "EvidenceError",
    "Outcome",
    "PhysicsLevel",
    "RelationalTrainingBatch",
    "Thresholds",
    "build_relational_training_batch",
    "compare",
)
