#!/usr/bin/env python3
"""Static sentinel for the active current-visit FrontRES Contract set."""

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
        "FRS-METHOD-v025-current-visit-scenario-replay.md"
    )
    compatibility_entry = _read("frontres_core/contracts/design_contract.md")
    training = _read(
        "frontres_core/contracts/active/training/"
        "FRS-TRAIN-v024-current-visit-target-cold-start.md"
    )
    optimization = _read(
        "frontres_core/contracts/active/optimization/"
        "FRS-PPO-v012-current-m4-mean-target.md"
    )
    reward = _read(
        "frontres_core/contracts/active/reward/"
        "FRS-GAIN-v008-recovery-aware-raw-evidence-utility-boundary.md"
    )
    evaluation = _read(
        "frontres_core/contracts/active/evaluation/"
        "FRS-EVAL-v006-current-visit-policy-quality.md"
    )
    historical_optimization = _read(
        "frontres_core/contracts/history/optimization/"
        "FRS-PPO-v006-state-value-segment-mean-update.md"
    )
    historical_training = _read(
        "frontres_core/contracts/history/training/"
        "FRS-TRAIN-v017-adaptive-critic-value-scale-curriculum.md"
    )

    _assert_contains(registry, "FRS-METHOD-v025-current-visit-scenario-replay.md", "registry")
    _assert_contains(registry, "FRS-TRAIN-v024-current-visit-target-cold-start.md", "registry")
    _assert_contains(registry, "FRS-PPO-v012-current-m4-mean-target.md", "registry")
    _assert_contains(registry, "FRS-GAIN-v008-recovery-aware-raw-evidence-utility-boundary.md", "registry")
    _assert_contains(registry, "FRS-EVAL-v006-current-visit-policy-quality.md", "registry")
    _assert_contains(registry, "Do not scan `history/`", "registry")
    _assert_contains(design, "contract_id: FRS-METHOD-v025", "design")
    _assert_contains(design, "status: active", "design")
    _assert_contains(design, "Replay is not Experience Replay", "design")
    _assert_contains(design, "target_s    = mean_m=1..4(u_sm)", "design")
    _assert_contains(design, "No M16 collection", "design")
    _assert_contains(design, "FRS-GAIN-v008", "design")
    _assert_contains(compatibility_entry, "contracts/README.md", "compatibility entry")
    _assert_contains(training, "contract_id: FRS-TRAIN-v024", "training")
    _assert_contains(training, "frontres-v024-checkpoint-v19", "training")
    _assert_contains(training, "Critic LR                   = 1e-5 throughout", "training")
    _assert_contains(optimization, "contract_id: FRS-PPO-v012", "optimization")
    _assert_contains(optimization, "value_target_s = mean_m=1..4(utility_sm)", "optimization")
    _assert_contains(optimization, "exactly one", "optimization")
    _assert_contains(reward, "contract_id: FRS-GAIN-v008", "reward")
    _assert_contains(reward, "G_total", "reward")
    _assert_contains(evaluation, "contract_id: FRS-EVAL-v006", "evaluation")
    _assert_contains(historical_optimization, "contract_id: FRS-PPO-v006", "historical optimization")
    _assert_contains(historical_optimization, "status: superseded", "historical optimization")
    _assert_contains(historical_training, "contract_id: FRS-TRAIN-v017", "historical training")
    _assert_contains(historical_training, "status: superseded", "historical training")

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

    print("FrontRES active v025/v024/v012/v008/v006 design contract sentinel: PASS")


if __name__ == "__main__":
    run_design_contract_sentinel()
