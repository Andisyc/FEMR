#!/usr/bin/env python3
"""Offline alignment and real-log comparison for a hierarchical Gain candidate.

This file is deliberately disconnected from the active Gain owner.  It tests a
candidate ordering and replays exported diagnostics without changing training,
contracts, checkpoints, or simulator state.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "frontres-action-gain-direction-v2"
ACTION_DIM = 6
HARD_EPSILON = 1.0e-12


class CandidateInputError(ValueError):
    """Raised when candidate evidence is missing or semantically invalid."""


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateInputError(f"{field} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateInputError(f"{field} must be finite")
    return result


def _problem(value: object, *, field: str) -> float:
    result = _number(value, field=field)
    if result < 0.0:
        raise CandidateInputError(f"{field} must be non-negative")
    return result


def _symlog(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _bounded_problem(value: float) -> float:
    return value / (1.0 + value)


def _continuous_problem(channels: Sequence[object], *, field: str) -> float:
    if len(channels) != 2:
        raise CandidateInputError(f"{field} must contain drift and phase-ZMP")
    applicable = [
        _bounded_problem(_problem(value, field=f"{field}[{index}]"))
        for index, value in enumerate(channels)
        if value is not None
    ]
    if not applicable:
        raise CandidateInputError(f"{field} has no applicable continuous Physics evidence")
    return sum(applicable) / len(applicable)


def hierarchical_gain(
    *,
    noisy_channels: Sequence[object],
    repaired_channels: Sequence[object],
    intent_gain: object,
    negative_repair_cost: object,
) -> tuple[float, str]:
    """Return the scalar adapter and the lexicographic branch that owned it.

    Exported channel order is [contact, support drift, phase-ZMP, survival].
    Contact is a proxy for the proposed no-load/unplanned-switch severity because
    the current diagnostic does not export those event identities separately.
    """

    if len(noisy_channels) != 4 or len(repaired_channels) != 4:
        raise CandidateInputError("Physics channel rows must contain exactly four values")
    survival_delta = _problem(noisy_channels[3], field="noisy survival") - _problem(
        repaired_channels[3], field="repaired survival"
    )
    if abs(survival_delta) > HARD_EPSILON:
        return 3.0 * math.copysign(1.0, survival_delta), "survival"

    contact_delta = _problem(noisy_channels[0], field="noisy contact") - _problem(
        repaired_channels[0], field="repaired contact"
    )
    if abs(contact_delta) > HARD_EPSILON:
        magnitude = 1.0 + _bounded_problem(abs(contact_delta))
        return math.copysign(magnitude, contact_delta), "severe_contact_proxy"

    noisy_continuous = _continuous_problem(noisy_channels[1:3], field="noisy continuous")
    repaired_continuous = _continuous_problem(
        repaired_channels[1:3], field="repaired continuous"
    )
    soft_score = (
        _number(intent_gain, field="intent_gain")
        + noisy_continuous
        - repaired_continuous
        + _number(negative_repair_cost, field="negative_repair_cost")
    )
    return math.tanh(soft_score), "soft"


def _run_semantic_cases() -> dict[str, object]:
    cases: dict[str, bool] = {}

    soft, branch = hierarchical_gain(
        noisy_channels=[0.0, 1.0, 0.0, 0.0],
        repaired_channels=[0.0, 0.25, 0.0, 0.0],
        intent_gain=0.0,
        negative_repair_cost=0.0,
    )
    cases["ordinary_soft_improvement"] = branch == "soft" and 0.0 < soft < 1.0

    hard_worse, branch = hierarchical_gain(
        noisy_channels=[0.0, 1.0, 0.0, 0.0],
        repaired_channels=[0.2, 0.0, 0.0, 0.0],
        intent_gain=100.0,
        negative_repair_cost=0.0,
    )
    cases["contact_is_not_compensated"] = branch == "severe_contact_proxy" and hard_worse < -1.0

    survival_better, branch = hierarchical_gain(
        noisy_channels=[0.0, 0.0, 0.0, 1.0],
        repaired_channels=[10.0, 10.0, 10.0, 0.0],
        intent_gain=-100.0,
        negative_repair_cost=-100.0,
    )
    cases["survival_has_first_priority"] = branch == "survival" and survival_better == 3.0

    forward, _ = hierarchical_gain(
        noisy_channels=[0.4, 1.0, None, 0.0],
        repaired_channels=[0.1, 2.0, None, 0.0],
        intent_gain=0.0,
        negative_repair_cost=0.0,
    )
    reverse, _ = hierarchical_gain(
        noisy_channels=[0.1, 2.0, None, 0.0],
        repaired_channels=[0.4, 1.0, None, 0.0],
        intent_gain=0.0,
        negative_repair_cost=0.0,
    )
    cases["noisy_repair_role_swap_flips_sign"] = math.isclose(
        forward, -reverse, rel_tol=0.0, abs_tol=1.0e-12
    )

    try:
        hierarchical_gain(
            noisy_channels=[0.0, None, None, 0.0],
            repaired_channels=[0.0, None, None, 0.0],
            intent_gain=0.0,
            negative_repair_cost=0.0,
        )
    except CandidateInputError:
        cases["missing_continuous_evidence_fails_closed"] = True
    else:
        cases["missing_continuous_evidence_fails_closed"] = False

    wrong_additive_score = 100.0 + (0.0 - 0.2)
    sensitivity = hard_worse < 0.0 < wrong_additive_score
    if not all(cases.values()) or not sensitivity:
        failed = [name for name, passed in cases.items() if not passed]
        raise AssertionError(f"candidate semantic cases failed: {failed}; sensitivity={sensitivity}")
    return {
        "status": "PASS",
        "cases": cases,
        "sensitivity": {
            "controlled_wrong_rule": "additive contact plus soft benefit",
            "wrong_rule_prefers_contact_worsening": sensitivity,
        },
    }


def _direction(
    actions: Sequence[Sequence[float]],
    values: Sequence[float],
    mean: Sequence[float],
    sigma: Sequence[float],
    indices: Sequence[int],
) -> list[float]:
    value_mean = sum(values[index] for index in indices) / len(indices)
    result = [0.0] * ACTION_DIM
    for index in indices:
        for axis in range(ACTION_DIM):
            score = (actions[index][axis] - mean[axis]) / (sigma[axis] * sigma[axis])
            result[axis] += score * (values[index] - value_mean)
    return [value / len(indices) for value in result]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1.0e-15 or right_norm <= 1.0e-15:
        return None
    value = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def _kendall_sign_agreement(left: Sequence[float], right: Sequence[float]) -> float | None:
    concordant = 0
    discordant = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if abs(left_delta) <= HARD_EPSILON or abs(right_delta) <= HARD_EPSILON:
                continue
            if left_delta * right_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    compared = concordant + discordant
    return (concordant - discordant) / compared if compared else None


def _sign(value: float) -> int:
    return int(value > HARD_EPSILON) - int(value < -HARD_EPSILON)


def _analyze_scenario(scenario: Mapping[str, object]) -> dict[str, object]:
    scenario_id = str(scenario.get("scenario_id"))
    raw_rows = scenario.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 32:
        raise CandidateInputError(f"Scenario {scenario_id!r} must contain 32 rows")
    mean = [_number(value, field="actor_mean") for value in scenario.get("actor_mean", [])]
    sigma = [_number(value, field="actor_sigma") for value in scenario.get("actor_sigma", [])]
    if len(mean) != ACTION_DIM or len(sigma) != ACTION_DIM or any(value <= 0.0 for value in sigma):
        raise CandidateInputError(f"Scenario {scenario_id!r} has invalid actor distribution")

    if any(not isinstance(row, Mapping) for row in raw_rows):
        raise CandidateInputError(f"Scenario {scenario_id!r} rows must be objects")
    rows = sorted(raw_rows, key=lambda row: int(row["repair_index"]))
    old_gains: list[float] = []
    old_utilities: list[float] = []
    candidate_gains: list[float] = []
    candidate_utilities: list[float] = []
    actions: list[list[float]] = []
    branches: Counter[str] = Counter()
    max_total_error = 0.0
    max_utility_error = 0.0
    hard_worsen_old_positive = 0
    hard_improve_old_negative = 0
    noisy_contact_by_visit: dict[int, set[float]] = {visit: set() for visit in range(8)}

    for row_index, row in enumerate(rows):
        components = row.get("components")
        if not isinstance(components, Mapping):
            raise CandidateInputError(f"Scenario {scenario_id!r} row {row_index} lacks components")
        noisy = components.get("physics_channel_noisy")
        repaired = components.get("physics_channel_repaired")
        if not isinstance(noisy, list) or not isinstance(repaired, list):
            raise CandidateInputError(f"Scenario {scenario_id!r} row {row_index} lacks Physics channels")
        intent = _number(components.get("intent_gain"), field="intent_gain")
        weighted_physics = _number(
            components.get("weighted_physics_gain"), field="weighted_physics_gain"
        )
        negative_cost = _number(
            components.get("negative_repair_cost"), field="negative_repair_cost"
        )
        old_gain = _number(components.get("gain_total"), field="gain_total")
        old_utility = _number(components.get("utility"), field="utility")
        max_total_error = max(max_total_error, abs(old_gain - (intent + weighted_physics + negative_cost)))
        max_utility_error = max(max_utility_error, abs(old_utility - _symlog(old_gain)))
        visit_index = row.get("visit_index")
        if isinstance(visit_index, bool) or not isinstance(visit_index, int) or visit_index not in range(8):
            raise CandidateInputError(f"Scenario {scenario_id!r} row {row_index} has invalid visit_index")
        noisy_contact_by_visit[visit_index].add(_problem(noisy[0], field="noisy contact"))

        candidate_gain, branch = hierarchical_gain(
            noisy_channels=noisy,
            repaired_channels=repaired,
            intent_gain=intent,
            negative_repair_cost=negative_cost,
        )
        branches[branch] += 1
        contact_delta = _problem(noisy[0], field="noisy contact") - _problem(
            repaired[0], field="repaired contact"
        )
        if contact_delta < -HARD_EPSILON and old_gain > HARD_EPSILON:
            hard_worsen_old_positive += 1
        if contact_delta > HARD_EPSILON and old_gain < -HARD_EPSILON:
            hard_improve_old_negative += 1

        raw_action = row.get("action")
        if not isinstance(raw_action, list) or len(raw_action) != ACTION_DIM:
            raise CandidateInputError(f"Scenario {scenario_id!r} row {row_index} has invalid action")
        actions.append([_number(value, field="action") for value in raw_action])
        old_gains.append(old_gain)
        old_utilities.append(old_utility)
        candidate_gains.append(candidate_gain)
        candidate_utilities.append(_symlog(candidate_gain))

    if max_total_error > 1.0e-6 or max_utility_error > 1.0e-6:
        raise CandidateInputError(
            f"Scenario {scenario_id!r} does not reproduce the exported old Gain/utility identities"
        )
    if any(len(values) != 1 for values in noisy_contact_by_visit.values()):
        raise CandidateInputError(
            f"Scenario {scenario_id!r} has inconsistent Noisy Contact within one M4 visit"
        )
    visit_noisy_contact = [next(iter(noisy_contact_by_visit[visit])) for visit in range(8)]

    left = list(range(16))
    right = list(range(16, 32))
    old_cosine = _cosine(
        _direction(actions, old_utilities, mean, sigma, left),
        _direction(actions, old_utilities, mean, sigma, right),
    )
    candidate_cosine = _cosine(
        _direction(actions, candidate_utilities, mean, sigma, left),
        _direction(actions, candidate_utilities, mean, sigma, right),
    )
    changed_m4_winners = 0
    for visit in range(8):
        indices = list(range(visit * 4, visit * 4 + 4))
        old_winner = max(indices, key=old_gains.__getitem__)
        candidate_winner = max(indices, key=candidate_gains.__getitem__)
        changed_m4_winners += old_winner != candidate_winner

    return {
        "scenario_id": scenario_id,
        "segment_id": scenario.get("segment_id"),
        "old_identity_max_abs_error": {
            "gain_total": max_total_error,
            "utility": max_utility_error,
        },
        "candidate_branch_counts": dict(sorted(branches.items())),
        "noisy_contact_baseline": {
            "per_visit": visit_noisy_contact,
            "unique_across_visits": len(set(visit_noisy_contact)),
            "stable_across_visits": len(set(visit_noisy_contact)) == 1,
        },
        "old_candidate_sign_flip_count": sum(
            _sign(old) != _sign(candidate)
            for old, candidate in zip(old_gains, candidate_gains, strict=True)
        ),
        "changed_m4_winner_count": changed_m4_winners,
        "old_candidate_kendall_sign_agreement": _kendall_sign_agreement(
            old_gains, candidate_gains
        ),
        "old_formula_priority_counterexamples": {
            "contact_worsened_but_old_gain_positive": hard_worsen_old_positive,
            "contact_improved_but_old_gain_negative": hard_improve_old_negative,
        },
        "primary_disjoint_m16_policy_score_cosine": {
            "old_utility": old_cosine,
            "candidate_utility": candidate_cosine,
        },
        "candidate_gain_range": [min(candidate_gains), max(candidate_gains)],
    }


def analyze_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise CandidateInputError(f"input schema must be {SCHEMA!r}")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise CandidateInputError("input must contain Scenarios")
    semantic_cases = _run_semantic_cases()
    scenarios = [_analyze_scenario(scenario) for scenario in raw_scenarios]
    return {
        "schema": "frontres-hierarchical-gain-candidate-offline-v1",
        "evidence_label": "MISSING-CASE",
        "reason": (
            "candidate ordering and real-log proxy comparison passed offline, but no production public "
            "boundary exists and the log does not separate no-load from unplanned-switch events"
        ),
        "preserved_boundary": "offline-only; active FRS-GAIN-v008 and training are unchanged",
        "log_adapter": {
            "contact": "exported normalized contact mismatch exposure used only as severe-contact proxy",
            "continuous": "exported support-foot drift and phase-ZMP channels",
            "survival": "exported normalized lost-horizon channel",
            "soft_scale": "existing dimensionless intent_gain and weighted negative repair cost; unit scale",
        },
        "semantic_pseudo_samples": semantic_cases,
        "scenarios": scenarios,
        "aggregate": {
            "scenario_count": len(scenarios),
            "row_count": 32 * len(scenarios),
            "sign_flip_count": sum(row["old_candidate_sign_flip_count"] for row in scenarios),
            "changed_m4_winner_count": sum(row["changed_m4_winner_count"] for row in scenarios),
            "old_priority_counterexample_count": sum(
                sum(row["old_formula_priority_counterexamples"].values()) for row in scenarios
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(analyze_payload(payload), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
