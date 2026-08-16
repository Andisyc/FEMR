#!/usr/bin/env python3
"""T-preference alignment for production FRS-GAIN-v008 Repair Physics P_R.

The oracle is the confirmed partial order in
frontres_pr_behavioral_ordering_v1.json.  No alternate score is computed.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import math
from pathlib import Path
from typing import Any

import torch

from frontres_gain_v008_numeric_alignment import _semantic_fixture
from rsl_rl.frontres.frontres_gain import (
    FrontRESRecoveryAwareGainConfig,
    FrontRESRecoveryAwareGainInput,
    compute_recovery_aware_gain,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CARD_PATH = PROJECT_ROOT / "note/testing/manifests/frontres_pr_behavioral_ordering_v1.json"
EPSILON = 1.0e-7


def _load_ordering_card() -> dict[str, Any]:
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    if card.get("schema") != "reward-ordering-card/v1":
        raise AssertionError("P_R ordering card schema is stale")
    if card.get("status") != "CONFIRMED" or card.get("ordering_kind") != "partial":
        raise AssertionError("P_R ordering card must be a confirmed partial order")
    if card.get("oracle", {}).get("second_scoring_formula_forbidden") is not True:
        raise AssertionError("P_R ordering card must forbid a second scoring formula")
    return card


def _eight_clean_rows() -> FrontRESRecoveryAwareGainInput:
    """Reuse the established semantic fixture setup, selecting its clean row."""

    base = _semantic_fixture()
    indices = torch.full((8,), 3, dtype=torch.long)
    values: dict[str, torch.Tensor] = {}
    for field in fields(base):
        value = getattr(base, field.name)
        batch_axis = 0 if field.name == "repair_actions" else 1
        values[field.name] = value.index_select(batch_axis, indices).clone()
    values["repair_actions"].zero_()
    return type(base)(**values)


def _preference_fixture() -> FrontRESRecoveryAwareGainInput:
    """Create four named pairs without copying the production P_R formula."""

    evidence = _eight_clean_rows()
    repaired_foot = evidence.repaired_foot_pos.clone()
    expected_support = evidence.expected_support.clone()
    clean_contact = evidence.clean_contact.clone()
    noisy_contact = evidence.noisy_contact.clone()
    repaired_contact = evidence.repaired_contact.clone()
    repaired_zmp = evidence.repaired_zmp_margin.clone()
    repaired_survival = evidence.repaired_survival.clone()

    # Pair 0/1: row 0 has only ordinary 6 cm support drift. Row 1 has one
    # expected-support no-load frame and no ordinary residual. Human rule: 0 < 1.
    repaired_foot[:, 0, :, 0] += 0.06
    for row in (1, 5):
        expected_support[:, row, 1] = 0.0
        clean_contact[:, row, 1] = 0.0
        noisy_contact[:, row, 1] = 0.0
        repaired_contact[:, row, 1] = 0.0
        repaired_contact[0, row, 0] = 0.0
        repaired_zmp[0, row] = float("nan")

    # Pair 2/3: both fail survival; row 2 loses less horizon and Pareto-dominates.
    repaired_survival[-1, 2] = 0.0
    repaired_survival[-2:, 3] = 0.0

    # Pair 4/5: survival loss versus expected-support no-load. The confirmed
    # partial order deliberately supplies no expected relation for this exchange.
    repaired_survival[-2:, 4] = 0.0

    # Pair 6/7: severe-clear ordinary drift dominance.
    repaired_foot[:, 6, :, 0] += 0.03
    repaired_foot[:, 7, :, 0] += 0.06

    return replace(
        evidence,
        repaired_foot_pos=repaired_foot,
        expected_support=expected_support,
        clean_contact=clean_contact,
        noisy_contact=noisy_contact,
        repaired_contact=repaired_contact,
        repaired_zmp_margin=repaired_zmp,
        repaired_survival=repaired_survival,
    )


def _strictly_less(left: float, right: float) -> bool:
    return left < right - EPSILON


def run_semantic_pseudo_samples() -> dict[str, Any]:
    card = _load_ordering_card()
    result = compute_recovery_aware_gain(
        _preference_fixture(),
        config=FrontRESRecoveryAwareGainConfig(beta=0.02, contact_timing_tolerance=0),
    )
    p_repaired = [float(value) for value in result.physics_remaining_repaired]
    cases = [
        {
            "id": "no-severe-dominates-expected-support-no-load",
            "case_family": "C3",
            "expected_relation": "row_0_better_than_row_1",
            "observed": {"row_0_pr": p_repaired[0], "row_1_pr": p_repaired[1]},
            "passed": _strictly_less(p_repaired[0], p_repaired[1]),
        },
        {
            "id": "severe-pareto-dominance",
            "case_family": "C1",
            "expected_relation": "row_2_better_than_row_3",
            "observed": {"row_2_pr": p_repaired[2], "row_3_pr": p_repaired[3]},
            "passed": _strictly_less(p_repaired[2], p_repaired[3]),
        },
        {
            "id": "severe-exchange-remains-incomparable",
            "case_family": "C4",
            "expected_relation": "incomparable",
            "observed": {
                "survival_loss_pr": p_repaired[4],
                "expected_support_no_load_pr": p_repaired[5],
                "production_scalar_relation": (
                    "survival_loss_lower"
                    if _strictly_less(p_repaired[4], p_repaired[5])
                    else "no_load_lower"
                    if _strictly_less(p_repaired[5], p_repaired[4])
                    else "equal_within_tolerance"
                ),
            },
            "passed": None,
        },
        {
            "id": "ordinary-pareto-dominance-after-severe-clear",
            "case_family": "C1",
            "expected_relation": "row_6_better_than_row_7",
            "observed": {"row_6_pr": p_repaired[6], "row_7_pr": p_repaired[7]},
            "passed": _strictly_less(p_repaired[6], p_repaired[7]),
        },
    ]
    violations = [case["id"] for case in cases if case["passed"] is False]
    if "no-severe-dominates-expected-support-no-load" not in violations:
        raise AssertionError("fixture no longer exposes the confirmed P_R compensation defect")
    if any(case["passed"] is False for case in cases[1:2] + cases[3:]):
        raise AssertionError("production P_R lost a confirmed Pareto monotonicity relation")
    return {
        "card_id": card["card_id"],
        "public_boundary": "compute_recovery_aware_gain(...).physics_remaining_repaired",
        "sct": ["S1", "C1", "C3", "C4", "T-preference", "T-role", "T-mask"],
        "second_scoring_formula_used": False,
        "cases": cases,
        "objective_status": "OBJECTIVE-VIOLATION" if violations else "OBJECTIVE-ALIGNED",
        "first_violation": violations[0] if violations else None,
        "sensitivity": "observed current production behavior violates the confirmed non-compensation case",
    }


def _finite_vector(value: Any, *, size: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != size:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    result = [float(item) for item in value]
    return result if all(math.isfinite(item) for item in result) else None


def replay_exported_transaction(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "frontres-action-gain-direction-v2":
        raise AssertionError("real transaction schema is not frontres-action-gain-direction-v2")

    pareto_pairs = 0
    pareto_inversions = 0
    ordinary_pairs = 0
    ordinary_inversions = 0
    survival_rule_pairs = 0
    survival_rule_violations = 0
    row_count = 0
    for scenario in payload.get("scenarios", []):
        rows = scenario.get("rows", [])
        valid_rows: list[tuple[list[float], float]] = []
        for row in rows:
            components = row.get("components", {})
            channels = _finite_vector(components.get("physics_channel_repaired"), size=4)
            p_repaired = components.get("physics_remaining_repaired")
            if channels is None or isinstance(p_repaired, bool) or not isinstance(p_repaired, (int, float)):
                continue
            if not math.isfinite(float(p_repaired)):
                continue
            valid_rows.append((channels, float(p_repaired)))
        row_count += len(valid_rows)
        for left_index, left in enumerate(valid_rows):
            for right in valid_rows[left_index + 1 :]:
                for better, worse in ((left, right), (right, left)):
                    better_channels, better_pr = better
                    worse_channels, worse_pr = worse
                    if all(a <= b + EPSILON for a, b in zip(better_channels, worse_channels, strict=True)) and any(
                        a < b - EPSILON for a, b in zip(better_channels, worse_channels, strict=True)
                    ):
                        pareto_pairs += 1
                        pareto_inversions += not _strictly_less(better_pr, worse_pr)
                    if better_channels[0] <= EPSILON and better_channels[3] <= EPSILON and worse_channels[3] > EPSILON:
                        survival_rule_pairs += 1
                        survival_rule_violations += not _strictly_less(better_pr, worse_pr)
                    if (
                        better_channels[0] <= EPSILON
                        and better_channels[3] <= EPSILON
                        and worse_channels[0] <= EPSILON
                        and worse_channels[3] <= EPSILON
                        and better_channels[1] <= worse_channels[1] + EPSILON
                        and better_channels[2] <= worse_channels[2] + EPSILON
                        and (
                            better_channels[1] < worse_channels[1] - EPSILON
                            or better_channels[2] < worse_channels[2] - EPSILON
                        )
                    ):
                        ordinary_pairs += 1
                        ordinary_inversions += not _strictly_less(better_pr, worse_pr)

    return {
        "fixture_kind": "real_transaction",
        "path": str(path),
        "row_count": row_count,
        "exported_pr_pareto": {"pair_count": pareto_pairs, "inversion_count": pareto_inversions},
        "confirmed_survival_non_compensation": {
            "pair_count": survival_rule_pairs,
            "violation_count": survival_rule_violations,
        },
        "severe_clear_ordinary_pareto": {
            "pair_count": ordinary_pairs,
            "inversion_count": ordinary_inversions,
        },
        "production_boundary_replayed": False,
        "objective_status": "TELEMETRY-GAP",
        "missing_fields": [
            "expected_support",
            "actual_loaded_support",
            "clean_planned_support_transition",
            "actual_support_transition",
            "raw K-step trajectories required by compute_recovery_aware_gain",
        ],
        "interpretation": (
            "exported Contact exposure conflates ordinary mismatch, expected-support no-load, "
            "and unplanned support switching; it cannot decide the severe partial order"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transaction", type=Path, nargs="?")
    args = parser.parse_args()
    report: dict[str, Any] = {"semantic_pseudo_samples": run_semantic_pseudo_samples()}
    if args.transaction is not None:
        report["real_transaction_replay"] = replay_exported_transaction(args.transaction)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
