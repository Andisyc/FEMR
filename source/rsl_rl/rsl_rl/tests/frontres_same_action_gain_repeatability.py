"""Pure final-snapshot validation and same-action Gain comparison.

Status: test-only functional core.
Upstream: the independent repeatability probe final-log parser.
Downstream: descriptive JSON-ready comparison data only.
Evidence: offline contract-confirmed.
Gap: no simulator, formal-route or policy-quality claim.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Mapping, Sequence


SNAPSHOT_PREFIX = "[FrontRES v017 Live Snapshot] "
REPEAT_COUNT = 4
ROW_COUNT = 32
SOURCE_COUNT = 8
ATTEMPTS_PER_SOURCE = 4
ACTION_DIM = 6
EXPECTED_CONTRACTS = {
    "method_contract_id": "FRS-METHOD-v025",
    "gain_contract_id": "FRS-GAIN-v008",
    "optimization_contract_id": "FRS-PPO-v012",
    "training_contract_id": "FRS-TRAIN-v024",
    "checkpoint_format": "frontres-v024-checkpoint-v19",
}
IDENTITY_FIELDS = (
    "scenario_ids",
    "noisy_segment_hashes",
    "x_t_identities",
    "source_index",
    "trial_index",
    "valid_policy_row_mask",
)
COMPONENT_FIELDS = (
    "intent_gain",
    "physics_gain",
    "recovery_pressure",
    "weighted_physics_gain",
    "repair_cost",
    "gain_total",
    "raw_returns",
    "utility_returns",
)
RANK_FIELDS = ("gain_total", "utility_returns")


class ProbeInputError(ValueError):
    """Reject incomplete or identity-ambiguous probe evidence."""


def _require_sequence(value: Any, *, name: str, length: int = ROW_COUNT) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ProbeInputError(f"{name} must contain exactly {length} rows")
    return value


def _string_rows(value: Any, *, name: str) -> tuple[str, ...]:
    rows = _require_sequence(value, name=name)
    if any(not isinstance(item, str) or not item for item in rows):
        raise ProbeInputError(f"{name} must contain non-empty string identities")
    return tuple(rows)


def _integer_rows(value: Any, *, name: str) -> tuple[int, ...]:
    rows = _require_sequence(value, name=name)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in rows):
        raise ProbeInputError(f"{name} must contain integer rows")
    return tuple(int(item) for item in rows)


def _boolean_rows(value: Any, *, name: str) -> tuple[bool, ...]:
    rows = _require_sequence(value, name=name)
    if any(not isinstance(item, bool) for item in rows):
        raise ProbeInputError(f"{name} must contain boolean rows")
    return tuple(rows)


def _finite_rows(value: Any, *, name: str) -> tuple[float, ...]:
    raw_rows = _require_sequence(value, name=name)
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw_rows):
        raise ProbeInputError(f"{name} must contain numeric rows")
    rows = tuple(float(item) for item in raw_rows)
    if not all(math.isfinite(item) for item in rows):
        raise ProbeInputError(f"{name} contains a non-finite row")
    return rows


def _action_rows(value: Any) -> tuple[tuple[float, ...], ...]:
    rows = _require_sequence(value, name="policy_actions")
    actions: list[tuple[float, ...]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != ACTION_DIM:
            raise ProbeInputError(f"policy_actions[{index}] must be one finite 6D action")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in row):
            raise ProbeInputError(f"policy_actions[{index}] must contain numeric values")
        action = tuple(float(item) for item in row)
        if not all(math.isfinite(item) for item in action):
            raise ProbeInputError(f"policy_actions[{index}] contains a non-finite value")
        actions.append(action)
    return tuple(actions)


def parse_live_snapshot(log_text: str, *, source: str) -> dict[str, Any]:
    """Parse exactly one production serializer record into a closed row schema."""

    # B1: 定位唯一最终 serializer 记录, 产出原始 JSON object.
    encoded = [line[len(SNAPSHOT_PREFIX) :] for line in log_text.splitlines() if line.startswith(SNAPSHOT_PREFIX)]
    if len(encoded) != 1:
        raise ProbeInputError(f"{source} must contain exactly one {SNAPSHOT_PREFIX.strip()} record")
    try:
        snapshot = json.loads(encoded[0])
    except json.JSONDecodeError as exc:
        raise ProbeInputError(f"{source} has malformed snapshot JSON: {exc}") from exc
    if not isinstance(snapshot, Mapping):
        raise ProbeInputError(f"{source} snapshot must be a JSON object")
    sealed = snapshot.get("sealed_transaction_evidence")
    if not isinstance(sealed, Mapping):
        raise ProbeInputError(f"{source} is missing sealed_transaction_evidence")

    # B2: 校验 Contract, B8/M4, exact-one 和行身份, 产出不可歧义 identity.
    for name, expected in EXPECTED_CONTRACTS.items():
        if sealed.get(name) != expected:
            raise ProbeInputError(f"{source} requires {name}={expected}")
    if (
        sealed.get("active_k") != 8
        or sealed.get("active_m") != ATTEMPTS_PER_SOURCE
        or sealed.get("selected_segment_count") != SOURCE_COUNT
        or sealed.get("policy_row_count") != ROW_COUNT
        or sealed.get("optimizer_step_delta") != 1
        or snapshot.get("optimizer_step_delta") != 1
        or snapshot.get("exact_one_update") is not True
    ):
        raise ProbeInputError(f"{source} is not one exact K8/B8/M4 update")
    scenario_ids = _string_rows(sealed.get("scenario_ids"), name="scenario_ids")
    noisy_hashes = _string_rows(sealed.get("noisy_segment_hashes"), name="noisy_segment_hashes")
    x_t_identities = _string_rows(snapshot.get("x_t_identities"), name="x_t_identities")
    source_index = _integer_rows(sealed.get("source_index"), name="source_index")
    trial_index = _integer_rows(sealed.get("trial_index"), name="trial_index")
    valid_mask = _boolean_rows(sealed.get("valid_policy_row_mask"), name="valid_policy_row_mask")
    if tuple(snapshot.get("scenario_ids", ())) != scenario_ids:
        raise ProbeInputError(f"{source} mixes pre-update and sealed scenario_ids")
    if tuple(snapshot.get("noisy_segment_hashes", ())) != noisy_hashes:
        raise ProbeInputError(f"{source} mixes pre-update and sealed noisy_segment_hashes")
    if not all(valid_mask):
        raise ProbeInputError(f"{source} contains an invalid Repair row")
    if set(zip(source_index, trial_index, strict=True)) != {
        (source_row, trial_row)
        for source_row in range(SOURCE_COUNT)
        for trial_row in range(ATTEMPTS_PER_SOURCE)
    }:
        raise ProbeInputError(f"{source} lost the exact B8/M4 source/trial identity")
    for source_row in range(SOURCE_COUNT):
        indices = tuple(index for index, value in enumerate(source_index) if value == source_row)
        if any(
            len({rows[index] for index in indices}) != 1
            for rows in (scenario_ids, noisy_hashes, x_t_identities)
        ):
            raise ProbeInputError(f"{source} mixes Scenario identity within source {source_row}")

    # B3: 读取已封存 action 和 Gain 字段, 不重算任何训练语义.
    policy_actions = _action_rows(sealed.get("policy_actions"))
    components = {name: _finite_rows(sealed.get(name), name=name) for name in COMPONENT_FIELDS}
    return {
        "source": source,
        "identity": {
            "scenario_ids": scenario_ids,
            "noisy_segment_hashes": noisy_hashes,
            "x_t_identities": x_t_identities,
            "source_index": source_index,
            "trial_index": trial_index,
            "valid_policy_row_mask": valid_mask,
        },
        "policy_actions": policy_actions,
        "components": components,
        "contracts": dict(EXPECTED_CONTRACTS),
    }


def _field_summary(repeats: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    row_values = tuple(zip(*(repeat["components"][field] for repeat in repeats), strict=True))
    ranges = tuple(max(values) - min(values) for values in row_values)
    deviations = tuple(statistics.pstdev(values) for values in row_values)
    sign_flip_rows = tuple(
        index for index, values in enumerate(row_values) if min(values) < 0.0 < max(values)
    )
    return {
        "max_absolute_range": max(ranges),
        "max_population_std": max(deviations),
        "sign_flip_row_count": len(sign_flip_rows),
        "sign_flip_rows": sign_flip_rows,
    }


def _scenario_mean_spread(repeats: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
    source_index = repeats[0]["identity"]["source_index"]
    result: dict[str, float] = {}
    for source in range(SOURCE_COUNT):
        indices = tuple(index for index, value in enumerate(source_index) if value == source)
        means = tuple(
            sum(repeat["components"][field][index] for index in indices) / ATTEMPTS_PER_SOURCE
            for repeat in repeats
        )
        result[str(source)] = max(means) - min(means)
    return result


def _ranking_changes(repeats: Sequence[dict[str, Any]], field: str) -> list[int]:
    source_index = repeats[0]["identity"]["source_index"]
    trial_index = repeats[0]["identity"]["trial_index"]
    changed: list[int] = []
    for source in range(SOURCE_COUNT):
        indices = tuple(index for index, value in enumerate(source_index) if value == source)
        rankings = tuple(
            tuple(
                trial_index[index]
                for index in sorted(
                    indices,
                    key=lambda row: (-repeat["components"][field][row], trial_index[row]),
                )
            )
            for repeat in repeats
        )
        if any(ranking != rankings[0] for ranking in rankings[1:]):
            changed.append(source)
    return changed


def compare_repeats(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply exact identity/action gates before descriptive Gain comparison."""

    if len(repeats) != REPEAT_COUNT:
        raise ProbeInputError(f"repeatability comparison requires exactly {REPEAT_COUNT} snapshots")
    baseline = repeats[0]
    identity_mismatches = [
        {"repeat": index + 1, "field": field}
        for index, repeat in enumerate(repeats[1:], start=1)
        for field in IDENTITY_FIELDS
        if repeat["identity"][field] != baseline["identity"][field]
    ]
    if identity_mismatches:
        return {
            "status": "INVALID_IDENTITY",
            "conclusion_authorized": False,
            "identity_mismatches": identity_mismatches,
        }
    action_mismatches = [
        index + 1
        for index, repeat in enumerate(repeats[1:], start=1)
        if repeat["policy_actions"] != baseline["policy_actions"]
    ]
    if action_mismatches:
        return {
            "status": "INVALID_ACTION",
            "conclusion_authorized": False,
            "action_mismatch_repeats": action_mismatches,
        }

    comparison = {
        "field_summaries": {field: _field_summary(repeats, field) for field in COMPONENT_FIELDS},
        "scenario_m4_mean_spread": {
            field: _scenario_mean_spread(repeats, field) for field in ("gain_total", "utility_returns")
        },
        "within_m4_rank_changes": {field: _ranking_changes(repeats, field) for field in RANK_FIELDS},
    }
    serialized_repeats = [
        {
            "source": repeat["source"],
            "components": repeat["components"],
        }
        for repeat in repeats
    ]
    return {
        "status": "DESCRIPTIVE_COMPLETE",
        "conclusion_authorized": True,
        "identity_gate": {
            "identity_exact": True,
            "policy_actions_exact": True,
            "canonical_identity": baseline["identity"],
            "policy_actions": baseline["policy_actions"],
        },
        "repeats": serialized_repeats,
        "comparison": comparison,
    }
