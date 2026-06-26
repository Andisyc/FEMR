#!/usr/bin/env python3
"""TEST ONLY: FEMR note contract must point to HSL+HRL acceptance.

This test prevents the active method notes from drifting back to the retired
Authority Actor-Critic mainline while the code cleanup is in progress.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NOTE = ROOT / "note"


def _read(name: str) -> str:
    path = NOTE / name
    if not path.exists():
        raise AssertionError(f"missing note file: {path}")
    return path.read_text(encoding="utf-8")


def _assert_contains(text: str, needle: str, name: str) -> None:
    if needle not in text:
        raise AssertionError(f"{name} missing required text: {needle!r}")


def run_design_contract_sentinel() -> None:
    design = _read("FrontRES Design Contract.md")
    plan = _read("FrontRES Engineering Plan.md")
    checklist = _read("FrontRES Modification Checklist.md")

    _assert_contains(design, "2026-06-25 FEMR HSL+HRL Acceptance Contract", "design")
    _assert_contains(design, "HSL owns continuous proposal magnitude", "design")
    _assert_contains(design, "HRL owns admissibility", "design")
    _assert_contains(design, "Authority Actor-Critic Contract (RETIRED MAINLINE)", "design")

    active_start = design.index("2026-06-25 FEMR HSL+HRL Acceptance Contract")
    retired_start = design.index("Authority Actor-Critic Contract (RETIRED MAINLINE)")
    if retired_start < active_start:
        raise AssertionError("retired authority contract appears before active HSL+HRL contract")

    _assert_contains(plan, "Implement the simplified FEMR HSL+HRL acceptance design", "plan")
    _assert_contains(plan, "Forbidden active path", "plan")
    _assert_contains(plan, "frontres_config_hsl_acceptance.py", "plan")

    _assert_contains(checklist, "2026-06-25 FEMR HSL+HRL Acceptance Refactor", "checklist")
    _assert_contains(checklist, "Step 2 config cleanup", "checklist")
    _assert_contains(checklist, "Step 11 legacy active-path audit", "checklist")

    forbidden_active_phrases = [
        "Implement the FrontRES Authority Actor-Critic design",
        "The active method is event-level, not per-frame rho",
    ]
    active_plan_prefix = plan.split("## 6. Test Discipline", 1)[0]
    for phrase in forbidden_active_phrases:
        if phrase in active_plan_prefix:
            raise AssertionError(f"plan still presents retired authority path as active: {phrase}")

    print("FrontRES design contract sentinel: PASS")


if __name__ == "__main__":
    run_design_contract_sentinel()
