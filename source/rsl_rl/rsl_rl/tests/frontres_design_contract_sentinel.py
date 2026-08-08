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
        "FRS-METHOD-v018-future-conditioned-state-value-segment-replay.md"
    )
    compatibility_entry = _read("frontres_core/contracts/design_contract.md")
    training = _read(
        "frontres_core/contracts/active/training/"
        "FRS-TRAIN-v017-adaptive-critic-value-scale-curriculum.md"
    )
    optimization = _read(
        "frontres_core/contracts/active/optimization/"
        "FRS-PPO-v007-output-preserving-adaptive-value-scale.md"
    )
    reward = _read(
        "frontres_core/contracts/active/reward/"
        "FRS-GAIN-v007-clean-anchored-recovery-aware-ranking.md"
    )
    evaluation = _read(
        "frontres_core/contracts/active/evaluation/"
        "FRS-EVAL-v004-clean-anchored-local-and-composition-evaluation.md"
    )
    historical_optimization = _read(
        "frontres_core/contracts/history/optimization/"
        "FRS-PPO-v006-state-value-segment-mean-update.md"
    )
    historical_training = _read(
        "frontres_core/contracts/history/training/"
        "FRS-TRAIN-v016-future-conditioned-state-value-curriculum.md"
    )

    _assert_contains(registry, "FRS-METHOD-v018-future-conditioned-state-value-segment-replay.md", "registry")
    _assert_contains(registry, "FRS-TRAIN-v017-adaptive-critic-value-scale-curriculum.md", "registry")
    _assert_contains(registry, "FRS-PPO-v007-output-preserving-adaptive-value-scale.md", "registry")
    _assert_contains(registry, "FRS-GAIN-v007-clean-anchored-recovery-aware-ranking.md", "registry")
    _assert_contains(registry, "FRS-EVAL-v004-clean-anchored-local-and-composition-evaluation.md", "registry")
    _assert_contains(registry, "Do not scan `history/`", "registry")
    _assert_contains(design, "contract_id: FRS-METHOD-v018", "design")
    _assert_contains(design, "status: active", "design")
    _assert_contains(design, "one full-6D current-frame repair", "design")
    _assert_contains(design, "FRS-GAIN-v007", "design")
    _assert_contains(design, "FRS-PPO-v007", "design")
    _assert_contains(compatibility_entry, "contracts/README.md", "compatibility entry")
    _assert_contains(training, "contract_id: FRS-TRAIN-v017", "training")
    _assert_contains(training, "checkpoint_schema = frontres-v017-checkpoint-v12", "training")
    _assert_contains(training, "critic_value_normalization_id = ema-target-std-nonamplifying-v1", "training")
    _assert_contains(optimization, "contract_id: FRS-PPO-v007", "optimization")
    _assert_contains(optimization, "L_value_scaled = L_value_raw / sigma^2", "optimization")
    _assert_contains(optimization, "`G_total` is not transformed", "optimization")
    _assert_contains(reward, "contract_id: FRS-GAIN-v007", "reward")
    _assert_contains(reward, "G_total", "reward")
    _assert_contains(evaluation, "contract_id: FRS-EVAL-v004", "evaluation")
    _assert_contains(historical_optimization, "contract_id: FRS-PPO-v006", "historical optimization")
    _assert_contains(historical_optimization, "status: superseded", "historical optimization")
    _assert_contains(historical_training, "contract_id: FRS-TRAIN-v016", "historical training")
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

    print("FrontRES active v018/v017/v007/v007/v004 design contract sentinel: PASS")


if __name__ == "__main__":
    run_design_contract_sentinel()
