#!/usr/bin/env python3
"""TEST ONLY: the FrontRES contract registry must select full-6D direct repair."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NOTE = ROOT / "note"


def _read(relative_path: str) -> str:
    path = NOTE / relative_path
    if not path.exists():
        raise AssertionError(f"missing note file: {path}")
    return path.read_text(encoding="utf-8")


def _assert_contains(text: str, needle: str, name: str) -> None:
    if needle not in text:
        raise AssertionError(f"{name} missing required text: {needle!r}")


def run_design_contract_sentinel() -> None:
    registry = _read("frontres_core/contracts/README.md")
    design = _read(
        "frontres_core/contracts/active/method/"
        "FRS-METHOD-v016-physics-constrained-intent-replay.md"
    )
    compatibility_entry = _read("frontres_core/contracts/design_contract.md")
    training = _read(
        "frontres_core/contracts/active/training/"
        "FRS-TRAIN-v010-intent-critic-k-curriculum.md"
    )
    optimization = _read(
        "frontres_core/contracts/active/optimization/"
        "FRS-PPO-v004-grouped-constraint-gradient-projection.md"
    )
    reward = _read(
        "frontres_core/contracts/active/reward/"
        "FRS-GAIN-v006-loaded-support-zmp-applicability.md"
    )
    evaluation = _read(
        "frontres_core/contracts/active/evaluation/"
        "FRS-EVAL-v003-local-repair-composition-evaluation.md"
    )
    historical_reward = _read(
        "frontres_core/contracts/history/reward/"
        "FRS-GAIN-v005-vector-physics-constraints.md"
    )

    _assert_contains(registry, "FRS-METHOD-v016-physics-constrained-intent-replay.md", "registry")
    _assert_contains(registry, "FRS-TRAIN-v010-intent-critic-k-curriculum.md", "registry")
    _assert_contains(
        registry,
        "FRS-PPO-v004-grouped-constraint-gradient-projection.md",
        "registry",
    )
    _assert_contains(registry, "FRS-GAIN-v006-loaded-support-zmp-applicability.md", "registry")
    _assert_contains(registry, "FRS-EVAL-v003-local-repair-composition-evaluation.md", "registry")
    _assert_contains(registry, "Do not scan `history/`", "registry")
    _assert_contains(design, "contract_id: FRS-METHOD-v016", "design")
    _assert_contains(design, "status: active", "design")
    _assert_contains(design, "one full-6D `Delta SE(3)` action", "design")
    _assert_contains(design, "paired Intent improvement - full-6D repair cost", "design")
    _assert_contains(design, "FRS-GAIN-v006", "design")
    _assert_contains(design, "one grouped constraint projection", "design")
    _assert_contains(compatibility_entry, "contracts/README.md", "compatibility entry")
    _assert_contains(training, "contract_id: FRS-TRAIN-v010", "training")
    _assert_contains(training, "gain_contract_id = FRS-GAIN-v006", "training")
    _assert_contains(training, "checkpoint_schema = frontres-v015-checkpoint-v5", "training")
    _assert_contains(optimization, "contract_id: FRS-PPO-v004", "optimization")
    _assert_contains(optimization, "FRS-GAIN-v006", "optimization")
    _assert_contains(reward, "contract_id: FRS-GAIN-v006", "reward")
    _assert_contains(reward, "actual_loaded_support", "reward")
    _assert_contains(reward, "expected-supported/actual-unloaded", "reward")
    _assert_contains(evaluation, "contract_id: FRS-EVAL-v003", "evaluation")
    _assert_contains(historical_reward, "contract_id: FRS-GAIN-v005", "historical reward")
    _assert_contains(historical_reward, "status: superseded", "historical reward")
    _assert_contains(historical_reward, "Superseded by `FRS-GAIN-v006`", "historical reward")

    retired_segment_notes = NOTE / "frontres_segment_replay"
    if retired_segment_notes.exists():
        raise AssertionError(
            f"retired standalone Segment Replay notes still exist: {retired_segment_notes}"
        )

    forbidden_active_phrases = (
        "Authority Actor-Critic Contract (RETIRED MAINLINE)",
        "HSL owns continuous proposal magnitude",
        "HRL owns admissibility, not full continuous rho authority",
        "contact-phase_zmp-survival-physical-v1",
    )
    active_contracts = "\n".join((design, training, optimization, reward, evaluation))
    for phrase in forbidden_active_phrases:
        if phrase in active_contracts:
            raise AssertionError(f"active contract contains historical method text: {phrase}")

    print("FrontRES active v016/v010/v006/v004/v003 design contract sentinel: PASS")


if __name__ == "__main__":
    run_design_contract_sentinel()
