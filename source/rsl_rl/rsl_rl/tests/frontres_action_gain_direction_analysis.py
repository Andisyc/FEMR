#!/usr/bin/env python3
"""Offline action-to-Gain direction analysis for fixed-Scenario M32 evidence.

This is a pure numerical diagnostic.  It neither imports the FrontRES runtime
nor updates Actor, Critic, optimizer, normalizer, or Replay state.  Every local
direction is the centered action/scalar cross-moment computed within the named
Scenario and subset under analysis.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
import random
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


SCHEMA = "frontres-action-gain-direction-v1"
ACTION_DIM = 6
ROWS_PER_SCENARIO = 32
COMPONENTS = (
    "utility",
    "gain_total",
    "intent_gain",
    "weighted_physics_gain",
    "negative_repair_cost",
)
EVIDENCE_COMPONENTS = (
    "utility",
    "raw_return",
    "gain_total",
    "intent_gain",
    "weighted_physics_gain",
    "repair_penalty",
    "negative_repair_cost",
)
SUBSET_SIZES = (4, 8, 16, 32)
RUNTIME_ACTIVE_M = 4
_NORM_EPSILON = 1.0e-15


class DirectionAnalysisInputError(ValueError):
    """Raised when diagnostic evidence is incomplete or semantically invalid."""


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectionAnalysisInputError(f"{field} must be a real scalar")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DirectionAnalysisInputError(f"{field} must be finite")
    return numeric


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-7, abs_tol=1.0e-9)


def _symlog(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _fixed_policy_vector(value: object, *, field: str, positive: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != ACTION_DIM:
        raise DirectionAnalysisInputError(f"{field} must have {ACTION_DIM} values")
    result = [_finite_number(item, field=f"{field}[{axis}]") for axis, item in enumerate(value)]
    if positive and any(item <= 0.0 for item in result):
        raise DirectionAnalysisInputError(f"{field} values must be strictly positive")
    return result


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise DirectionAnalysisInputError(f"{field} must be lowercase SHA-256")
    return value


def _validate_scenario(
    scenario: object,
) -> tuple[str, list[float], list[float], list[list[float]], dict[str, list[float]]]:
    if not isinstance(scenario, Mapping):
        raise DirectionAnalysisInputError("each Scenario must be an object")
    scenario_id = scenario.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise DirectionAnalysisInputError("scenario_id must be a non-empty string")
    active_m = scenario.get("active_m")
    if active_m != RUNTIME_ACTIVE_M:
        raise DirectionAnalysisInputError(
            f"Scenario {scenario_id!r} active_m must record source transaction M={RUNTIME_ACTIVE_M}"
        )
    scenario_checkpoint_hash = _sha256(
        scenario.get("checkpoint_file_sha256"),
        field=f"Scenario {scenario_id!r} checkpoint_file_sha256",
    )
    scenario_manifest_hash = _sha256(
        scenario.get("manifest_file_sha256"),
        field=f"Scenario {scenario_id!r} manifest_file_sha256",
    )
    raw_visits = scenario.get("visits")
    if not isinstance(raw_visits, list) or len(raw_visits) != 8:
        raise DirectionAnalysisInputError(f"Scenario {scenario_id!r} must contain exactly eight visits")
    visit_seeds: dict[int, int] = {}
    for position, visit in enumerate(raw_visits):
        if not isinstance(visit, Mapping):
            raise DirectionAnalysisInputError(f"Scenario {scenario_id!r} visit {position} must be an object")
        visit_index = visit.get("visit_index")
        action_seed = visit.get("action_seed")
        actor_drift = visit.get("actor_input_max_abs_diff")
        critic_drift = visit.get("critic_input_max_abs_diff")
        if (
            isinstance(visit_index, bool)
            or not isinstance(visit_index, int)
            or visit_index not in range(8)
            or visit_index in visit_seeds
            or isinstance(action_seed, bool)
            or not isinstance(action_seed, int)
            or action_seed < 0
            or _finite_number(actor_drift, field=f"Scenario {scenario_id!r} visit actor drift") != 0.0
            or _finite_number(critic_drift, field=f"Scenario {scenario_id!r} visit critic drift") != 0.0
        ):
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} visit {position} has invalid identity, seed, or policy-input drift"
            )
        visit_seeds[visit_index] = action_seed
    if set(visit_seeds) != set(range(8)):
        raise DirectionAnalysisInputError(f"Scenario {scenario_id!r} visit_index coverage must be 0..7")
    actor_mean = _fixed_policy_vector(
        scenario.get("actor_mean"),
        field=f"Scenario {scenario_id!r} actor_mean",
    )
    actor_sigma = _fixed_policy_vector(
        scenario.get("actor_sigma"),
        field=f"Scenario {scenario_id!r} actor_sigma",
        positive=True,
    )
    rows = scenario.get("rows")
    if not isinstance(rows, list) or len(rows) != ROWS_PER_SCENARIO:
        raise DirectionAnalysisInputError(
            f"Scenario {scenario_id!r} must contain exactly {ROWS_PER_SCENARIO} rows"
        )

    by_index: dict[int, tuple[list[float], dict[str, float]]] = {}
    visit_actions: dict[int, list[list[float]]] = {visit: [] for visit in range(8)}
    checkpoint_hashes: set[str] = set()
    manifest_hashes: set[str] = set()
    row_visit_seeds: dict[int, set[int]] = {visit: set() for visit in range(8)}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DirectionAnalysisInputError(f"Scenario {scenario_id!r} row {position} must be an object")
        repair_index = row.get("repair_index")
        if isinstance(repair_index, bool) or not isinstance(repair_index, int):
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} row {position} repair_index must be an integer"
            )
        if repair_index < 0 or repair_index >= ROWS_PER_SCENARIO or repair_index in by_index:
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} has invalid or duplicate repair_index {repair_index!r}"
            )
        visit_index = row.get("visit_index")
        attempt_index = row.get("attempt_index")
        action_seed = row.get("action_seed")
        if (
            isinstance(visit_index, bool)
            or not isinstance(visit_index, int)
            or isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or isinstance(action_seed, bool)
            or not isinstance(action_seed, int)
            or visit_index not in range(8)
            or attempt_index not in range(4)
            or repair_index != visit_index * 4 + attempt_index
            or action_seed < 0
        ):
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} repair {repair_index} has invalid visit/attempt/action-seed provenance"
            )
        row_visit_seeds[visit_index].add(action_seed)
        for field, destination in (
            ("checkpoint_file_sha256", checkpoint_hashes),
            ("manifest_file_sha256", manifest_hashes),
        ):
            fingerprint = _sha256(
                row.get(field), field=f"Scenario {scenario_id!r} repair {repair_index} {field}"
            )
            destination.add(fingerprint)
        action_value = row.get("action")
        if not isinstance(action_value, list) or len(action_value) != ACTION_DIM:
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} repair {repair_index} action must have {ACTION_DIM} values"
            )
        action = [
            _finite_number(value, field=f"Scenario {scenario_id!r} repair {repair_index} action[{axis}]")
            for axis, value in enumerate(action_value)
        ]
        visit_actions[visit_index].append(action)
        raw_components = row.get("components")
        if not isinstance(raw_components, Mapping):
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} repair {repair_index} components must be an object"
            )
        missing = [name for name in EVIDENCE_COMPONENTS if name not in raw_components]
        extra = [name for name in raw_components if name not in EVIDENCE_COMPONENTS]
        if missing or extra:
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} repair {repair_index} component identity mismatch; "
                f"missing={missing}, extra={extra}"
            )
        components = {
            name: _finite_number(
                raw_components[name],
                field=f"Scenario {scenario_id!r} repair {repair_index} component {name}",
            )
            for name in EVIDENCE_COMPONENTS
        }
        reconstructed_total = (
            components["intent_gain"]
            + components["weighted_physics_gain"]
            + components["negative_repair_cost"]
        )
        if not _close(components["gain_total"], reconstructed_total):
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} repair {repair_index} Gain components do not sum to gain_total"
            )
        if components["negative_repair_cost"] > 1.0e-12:
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} repair {repair_index} negative_repair_cost must be non-positive"
            )
        if components["repair_penalty"] < -1.0e-12 or not _close(
            components["negative_repair_cost"], -components["repair_penalty"]
        ):
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} repair {repair_index} negative_repair_cost "
                "must equal -repair_penalty"
            )
        if not _close(components["raw_return"], components["gain_total"]):
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} repair {repair_index} raw_return must equal gain_total"
            )
        if not _close(components["utility"], _symlog(components["gain_total"])):
            raise DirectionAnalysisInputError(
                f"Scenario {scenario_id!r} repair {repair_index} utility is not symlog(gain_total)"
            )
        by_index[repair_index] = (action, components)

    if checkpoint_hashes != {scenario_checkpoint_hash} or manifest_hashes != {scenario_manifest_hash}:
        raise DirectionAnalysisInputError(
            f"Scenario {scenario_id!r} rows disagree with Scenario checkpoint or manifest provenance"
        )
    if any(len(actions) != 4 for actions in visit_actions.values()):
        raise DirectionAnalysisInputError(f"Scenario {scenario_id!r} must contain exact M4 rows for visits 0..7")
    if any(row_visit_seeds[visit] != {visit_seeds[visit]} for visit in range(8)):
        raise DirectionAnalysisInputError(
            f"Scenario {scenario_id!r} row action_seed values disagree with visit provenance"
        )
    expected_indices = set(range(ROWS_PER_SCENARIO))
    if set(by_index) != expected_indices:
        raise DirectionAnalysisInputError(f"Scenario {scenario_id!r} repair_index coverage must be 0..31")
    actions = [by_index[index][0] for index in range(ROWS_PER_SCENARIO)]
    components = {
        name: [by_index[index][1][name] for index in range(ROWS_PER_SCENARIO)]
        for name in COMPONENTS
    }
    return scenario_id, actor_mean, actor_sigma, actions, components


def centered_cross_moment(
    actions: Sequence[Sequence[float]],
    values: Sequence[float],
    indices: Sequence[int] | None = None,
) -> list[float]:
    """Return mean((action-action_mean) * (value-value_mean)) on one subset."""

    if len(actions) != len(values):
        raise DirectionAnalysisInputError("actions and values must contain the same row count")
    selected = list(range(len(actions))) if indices is None else list(indices)
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise DirectionAnalysisInputError("a direction subset needs at least two unique row indices")
    if any(index < 0 or index >= len(actions) for index in selected):
        raise DirectionAnalysisInputError("direction subset index is out of range")

    selected_actions: list[list[float]] = []
    selected_values: list[float] = []
    for index in selected:
        action = actions[index]
        if len(action) != ACTION_DIM:
            raise DirectionAnalysisInputError(f"action row {index} must have {ACTION_DIM} axes")
        selected_actions.append(
            [_finite_number(value, field=f"action[{index}][{axis}]") for axis, value in enumerate(action)]
        )
        selected_values.append(_finite_number(values[index], field=f"value[{index}]"))

    count = float(len(selected))
    action_mean = [sum(row[axis] for row in selected_actions) / count for axis in range(ACTION_DIM)]
    value_mean = sum(selected_values) / count
    return [
        sum(
            (action[axis] - action_mean[axis]) * (value - value_mean)
            for action, value in zip(selected_actions, selected_values, strict=True)
        )
        / count
        for axis in range(ACTION_DIM)
    ]


def policy_score_direction(
    actions: Sequence[Sequence[float]],
    values: Sequence[float],
    actor_mean: Sequence[float],
    actor_sigma: Sequence[float],
    indices: Sequence[int] | None = None,
) -> list[float]:
    """Return the Gaussian policy-score/scalar cross-moment on one subset.

    The scalar is centered within the estimator subset.  ``actor_mean`` and
    ``actor_sigma`` are the frozen policy-distribution identity for the named
    Scenario; sigma is never estimated from the 32 diagnostic actions.
    """

    if len(actions) != len(values):
        raise DirectionAnalysisInputError("actions and values must contain the same row count")
    mean = _fixed_policy_vector(list(actor_mean), field="actor_mean")
    sigma = _fixed_policy_vector(list(actor_sigma), field="actor_sigma", positive=True)
    selected = list(range(len(actions))) if indices is None else list(indices)
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise DirectionAnalysisInputError("a direction subset needs at least two unique row indices")
    if any(index < 0 or index >= len(actions) for index in selected):
        raise DirectionAnalysisInputError("direction subset index is out of range")

    selected_values = [_finite_number(values[index], field=f"value[{index}]") for index in selected]
    scalar_mean = sum(selected_values) / len(selected_values)
    direction = [0.0] * ACTION_DIM
    for index, scalar in zip(selected, selected_values, strict=True):
        action = actions[index]
        if len(action) != ACTION_DIM:
            raise DirectionAnalysisInputError(f"action row {index} must have {ACTION_DIM} axes")
        for axis, action_value in enumerate(action):
            finite_action = _finite_number(action_value, field=f"action[{index}][{axis}]")
            score = (finite_action - mean[axis]) / (sigma[axis] * sigma[axis])
            direction[axis] += score * (scalar - scalar_mean)
    return [value / len(selected) for value in direction]


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def direction_cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != ACTION_DIM or len(right) != ACTION_DIM:
        raise DirectionAnalysisInputError(f"direction vectors must have {ACTION_DIM} axes")
    left_norm = vector_norm(left)
    right_norm = vector_norm(right)
    if left_norm <= _NORM_EPSILON or right_norm <= _NORM_EPSILON:
        return None
    value = sum(float(a) * float(b) for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def _derived_seed(seed: int, *identities: object) -> int:
    material = "|".join((str(seed), *(str(identity) for identity in identities))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="big", signed=False)


def _partition_pairs(row_count: int, subset_size: int, count: int, *, seed: int) -> list[tuple[list[int], list[int]]]:
    if count < 1:
        raise DirectionAnalysisInputError("partition_count must be positive")
    if subset_size < 2 or subset_size > row_count - 2:
        raise DirectionAnalysisInputError("partition subset must leave at least two held-out rows")
    universe = tuple(range(row_count))
    pairs: list[tuple[list[int], list[int]]] = []
    seen: set[tuple[int, ...]] = set()

    def add(selected: Iterable[int]) -> None:
        left = tuple(sorted(selected))
        if subset_size * 2 == row_count:
            complement = tuple(index for index in universe if index not in set(left))
            identity = min(left, complement)
        else:
            identity = left
        if identity in seen:
            return
        seen.add(identity)
        left_set = set(left)
        pairs.append((list(left), [index for index in universe if index not in left_set]))

    add(range(subset_size))
    rng = random.Random(seed)
    attempts = 0
    while len(pairs) < count and attempts < count * 100:
        add(rng.sample(universe, subset_size))
        attempts += 1
    if len(pairs) != count:
        raise DirectionAnalysisInputError(
            f"could not construct {count} distinct M{subset_size} held-out partitions"
        )
    return pairs


def _nested_fixed_reference_subsets(
    subset_size: int,
    count: int,
    *,
    seed: int,
) -> list[list[int]]:
    """Use complete M4 visit clusters from visits 0..3 against visits 4..7."""

    if subset_size not in (4, 8, 16):
        raise DirectionAnalysisInputError("fixed-reference subset size must be 4, 8, or 16")
    if count < 1:
        raise DirectionAnalysisInputError("partition_count must be positive")
    del count, seed
    visit_count = subset_size // RUNTIME_ACTIVE_M
    return [
        [row for visit in selected_visits for row in range(visit * 4, visit * 4 + 4)]
        for selected_visits in combinations(range(4), visit_count)
    ]


def _quantile(sorted_values: Sequence[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_optional(values: Sequence[float | None]) -> dict[str, object]:
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    return {
        "attempted_count": len(values),
        "defined_count": len(finite),
        "mean": (sum(finite) / len(finite)) if finite else None,
        "median": _quantile(finite, 0.5),
        "q05": _quantile(finite, 0.05),
        "q95": _quantile(finite, 0.95),
        "values": finite,
    }


def _direction_record(direction: Sequence[float], *, estimator_sample_count: int) -> dict[str, object]:
    return {
        "estimator_sample_count": estimator_sample_count,
        "direction": [float(value) for value in direction],
        "norm": vector_norm(direction),
    }


def _estimator_analysis(
    actions: Sequence[Sequence[float]],
    values: Sequence[float],
    partitions: Mapping[int, Sequence[tuple[list[int], list[int]]]],
    *,
    direction_estimator: Callable[
        [Sequence[Sequence[float]], Sequence[float], Sequence[int] | None], list[float]
    ],
    fixed_reference_subsets: Mapping[int, Sequence[Sequence[int]]],
    permutation_count: int,
    permutation_seed: int,
) -> dict[str, object]:
    full_direction = direction_estimator(actions, values, None)
    # Primary independence unit is one committed/fresh M4 visit.  The two M16
    # estimates therefore contain four complete, disjoint visits each.
    primary_left = list(range(16))
    primary_right = list(range(16, 32))
    primary_left_direction = direction_estimator(actions, values, primary_left)
    primary_right_direction = direction_estimator(actions, values, primary_right)
    primary_cosine = direction_cosine(primary_left_direction, primary_right_direction)

    fixed_reference_direction = primary_right_direction
    fixed_reference_progression: dict[str, object] = {}
    for subset_size in (4, 8, 16):
        subset_cosines: list[float | None] = []
        for subset in fixed_reference_subsets[subset_size]:
            subset_direction = direction_estimator(actions, values, subset)
            subset_cosines.append(direction_cosine(subset_direction, fixed_reference_direction))
        first_subset = list(fixed_reference_subsets[subset_size][0])
        first_direction = direction_estimator(actions, values, first_subset)
        fixed_reference_progression[f"m{subset_size}_vs_fixed_m16"] = {
            "candidate_pool_indices": list(range(16)),
            "fixed_reference_indices": primary_right,
            "first_subset_indices": first_subset,
            "first_subset": _direction_record(first_direction, estimator_sample_count=subset_size),
            "fixed_reference": _direction_record(fixed_reference_direction, estimator_sample_count=16),
            "cosine_summary": summarize_optional(subset_cosines),
        }

    held_out: dict[str, object] = {}
    primary_subset_directions: dict[str, object] = {}
    for subset_size in (4, 8, 16):
        cosines = []
        for left, right in partitions[subset_size]:
            left_direction = direction_estimator(actions, values, left)
            right_direction = direction_estimator(actions, values, right)
            cosines.append(direction_cosine(left_direction, right_direction))
        key = f"m{subset_size}_vs_m{ROWS_PER_SCENARIO - subset_size}"
        held_out[key] = summarize_optional(cosines)
        first_left, first_right = partitions[subset_size][0]
        first_left_direction = direction_estimator(actions, values, first_left)
        first_right_direction = direction_estimator(actions, values, first_right)
        primary_subset_directions[key] = {
            "subset_indices": first_left,
            "held_out_indices": first_right,
            "subset": _direction_record(first_left_direction, estimator_sample_count=subset_size),
            "held_out": _direction_record(
                first_right_direction,
                estimator_sample_count=ROWS_PER_SCENARIO - subset_size,
            ),
            "cosine": direction_cosine(first_left_direction, first_right_direction),
        }

    if permutation_count < 1:
        raise DirectionAnalysisInputError("permutation_count must be positive")
    permutation_rng = random.Random(permutation_seed)
    null_cosines: list[float | None] = []
    for _ in range(permutation_count):
        visit_order = list(range(8))
        permutation_rng.shuffle(visit_order)
        permuted = [
            value
            for source_visit in visit_order
            for value in values[source_visit * 4 : source_visit * 4 + 4]
        ]
        left_direction = direction_estimator(actions, permuted, primary_left)
        right_direction = direction_estimator(actions, permuted, primary_right)
        null_cosines.append(direction_cosine(left_direction, right_direction))
    null_summary = summarize_optional(null_cosines)
    null_defined = [float(value) for value in null_cosines if value is not None]
    upper_tail_probability = None
    if primary_cosine is not None and null_defined:
        upper_tail_probability = (1 + sum(value >= primary_cosine for value in null_defined)) / (
            len(null_defined) + 1
        )

    return {
        "m32": _direction_record(full_direction, estimator_sample_count=32),
        "primary_disjoint_m16": {
            "scientific_unit": "four-complete-m4-visits-per-half",
            "left_visit_indices": [0, 1, 2, 3],
            "right_visit_indices": [4, 5, 6, 7],
            "left_indices": primary_left,
            "right_indices": primary_right,
            "left": _direction_record(primary_left_direction, estimator_sample_count=16),
            "right": _direction_record(primary_right_direction, estimator_sample_count=16),
            "cosine": primary_cosine,
            "permutation_upper_tail_probability": upper_tail_probability,
        },
        "fixed_reference_progression": fixed_reference_progression,
        "primary_subset_directions": primary_subset_directions,
        "held_out_complement": {
            "evidence_role": "row-level-auxiliary; may split M4 visits and is not the primary verdict",
            **held_out,
        },
        "permutation_null_primary_m16": {
            "permutation_unit": "complete-m4-visit-block",
            **null_summary,
        },
    }


def analyze_scenario(
    scenario: object,
    *,
    partition_count: int = 128,
    permutation_count: int = 512,
    seed: int = 20260813,
) -> dict[str, object]:
    scenario_id, actor_mean, actor_sigma, actions, component_values = _validate_scenario(scenario)
    partitions = {
        subset_size: _partition_pairs(
            ROWS_PER_SCENARIO,
            subset_size,
            partition_count,
            seed=_derived_seed(seed, scenario_id, "partition", subset_size),
        )
        for subset_size in (4, 8, 16)
    }
    fixed_reference_subsets = {
        subset_size: _nested_fixed_reference_subsets(
            subset_size,
            partition_count,
            seed=_derived_seed(seed, scenario_id, "fixed-reference", subset_size),
        )
        for subset_size in (4, 8, 16)
    }
    analyses = {
        component: {
            "primary_estimator": "policy_score",
            "policy_score": _estimator_analysis(
                actions,
                component_values[component],
                partitions,
                direction_estimator=lambda observed_actions, observed_values, indices: policy_score_direction(
                    observed_actions,
                    observed_values,
                    actor_mean,
                    actor_sigma,
                    indices,
                ),
                fixed_reference_subsets=fixed_reference_subsets,
                permutation_count=permutation_count,
                permutation_seed=_derived_seed(seed, scenario_id, component, "permutation"),
            ),
            "raw_centered_covariance": _estimator_analysis(
                actions,
                component_values[component],
                partitions,
                direction_estimator=centered_cross_moment,
                fixed_reference_subsets=fixed_reference_subsets,
                permutation_count=permutation_count,
                permutation_seed=_derived_seed(seed, scenario_id, component, "permutation"),
            ),
        }
        for component in COMPONENTS
    }
    return {
        "scenario_id": scenario_id,
        "source_transaction_identity": {"active_m": RUNTIME_ACTIVE_M},
        "frozen_actor_distribution": {"mean": actor_mean, "sigma": actor_sigma},
        "row_count": ROWS_PER_SCENARIO,
        "component_analysis": analyses,
    }


def _equal_scenario_summary(scenarios: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for component in COMPONENTS:
        estimator_summaries: dict[str, object] = {}
        for estimator_name in ("policy_score", "raw_centered_covariance"):
            component_rows = [
                scenario["component_analysis"][component][estimator_name]  # type: ignore[index]
                for scenario in scenarios
            ]
            primary = [row["primary_disjoint_m16"]["cosine"] for row in component_rows]  # type: ignore[index]
            summary = {"primary_disjoint_m16_cosine": summarize_optional(primary)}
            for subset_size in (4, 8, 16):
                key = f"m{subset_size}_vs_m{ROWS_PER_SCENARIO - subset_size}"
                per_scenario_medians = [
                    row["held_out_complement"][key]["median"]  # type: ignore[index]
                    for row in component_rows
                ]
                summary[f"held_out_{key}_median"] = summarize_optional(per_scenario_medians)
                fixed_key = f"m{subset_size}_vs_fixed_m16"
                fixed_reference_medians = [
                    row["fixed_reference_progression"][fixed_key]["cosine_summary"]["median"]  # type: ignore[index]
                    for row in component_rows
                ]
                summary[f"fixed_reference_{fixed_key}_median"] = summarize_optional(
                    fixed_reference_medians
                )
            null_medians = [
                row["permutation_null_primary_m16"]["median"]  # type: ignore[index]
                for row in component_rows
            ]
            null_q95 = [
                row["permutation_null_primary_m16"]["q95"]  # type: ignore[index]
                for row in component_rows
            ]
            summary["permutation_null_primary_m16_median"] = summarize_optional(null_medians)
            summary["permutation_null_primary_m16_q95"] = summarize_optional(null_q95)
            estimator_summaries[estimator_name] = summary
        result[component] = {
            "primary_estimator": "policy_score",
            "estimator_summaries": estimator_summaries,
        }
    return result


def analyze_payload(
    payload: object,
    *,
    partition_count: int = 128,
    permutation_count: int = 512,
    seed: int = 20260813,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise DirectionAnalysisInputError("input must be a JSON object")
    if payload.get("schema") != SCHEMA:
        raise DirectionAnalysisInputError(f"input schema must be exactly {SCHEMA!r}")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise DirectionAnalysisInputError("input must contain at least one Scenario")
    scenarios = [
        analyze_scenario(
            scenario,
            partition_count=partition_count,
            permutation_count=permutation_count,
            seed=seed,
        )
        for scenario in raw_scenarios
    ]
    scenario_ids = [str(scenario["scenario_id"]) for scenario in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise DirectionAnalysisInputError("scenario_id values must be unique")
    return {
        "schema": SCHEMA,
        "analysis_identity": {
            "primary_direction": "frozen-gaussian-policy-score-times-subset-centered-scalar",
            "auxiliary_direction": "subset-centered-action-scalar-cross-moment",
            "action_dim": ACTION_DIM,
            "rows_per_scenario": ROWS_PER_SCENARIO,
            "source_transaction_active_m": RUNTIME_ACTIVE_M,
            "estimator_sample_counts": list(SUBSET_SIZES),
            "comparison_reference": (
                "primary M16 compares four complete M4 visits against four complete M4 visits; "
                "row-level held-out complements are auxiliary; inclusive-m32 is not a stability verdict"
            ),
            "partition_count": partition_count,
            "permutation_count": permutation_count,
            "seed": seed,
            "scenario_weighting": "equal",
        },
        "scenarios": scenarios,
        "equal_scenario_summary": _equal_scenario_summary(scenarios),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=f"JSON evidence using schema {SCHEMA}")
    parser.add_argument("--partitions", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyze_payload(
        payload,
        partition_count=args.partitions,
        permutation_count=args.permutations,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
