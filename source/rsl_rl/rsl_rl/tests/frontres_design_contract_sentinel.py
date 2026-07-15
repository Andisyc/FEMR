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
        "FRS-METHOD-v010-segment-replay.md"
    )
    compatibility_entry = _read("frontres_core/contracts/design_contract.md")
    history_index = _read("frontres_core/contracts/history/method/README.md")
    training = _read(
        "frontres_core/contracts/active/training/"
        "FRS-TRAIN-v001-segment-replay.md"
    )
    optimization = _read(
        "frontres_core/contracts/active/optimization/"
        "FRS-PPO-v001-sign-preserving-advantage-scaling.md"
    )
    evaluation = _read(
        "frontres_core/contracts/active/evaluation/"
        "FRS-EVAL-v001-segment-evaluation.md"
    )

    _assert_contains(registry, "FRS-METHOD-v010-segment-replay.md", "registry")
    _assert_contains(registry, "FRS-TRAIN-v001-segment-replay.md", "registry")
    _assert_contains(
        registry,
        "FRS-PPO-v001-sign-preserving-advantage-scaling.md",
        "registry",
    )
    _assert_contains(registry, "FRS-EVAL-v001-segment-evaluation.md", "registry")
    _assert_contains(registry, "Do not scan `history/`", "registry")
    _assert_contains(design, "contract_id: FRS-METHOD-v010", "design")
    _assert_contains(design, "status: active", "design")
    _assert_contains(design, "The executable action remains full 6D", "design")
    _assert_contains(design, "The method-version delta is Segment Replay", "design")
    _assert_contains(design, "Dynamic Reset Boundary", "design")
    _assert_contains(design, "K-Step Curriculum", "design")
    _assert_contains(design, "implemented-only, not integrated", "design")
    _assert_contains(design, "[dx, dy, dz, droll, dpitch, dyaw]", "design")
    _assert_contains(
        design,
        "Perturbation family describes the corruption source. It must not narrow",
        "design",
    )
    _assert_contains(design, "local_rp", "design")
    _assert_contains(compatibility_entry, "contracts/README.md", "compatibility entry")
    _assert_contains(training, "contract_id: FRS-TRAIN-v001", "training")
    _assert_contains(training, "Implementation gate", "training")
    _assert_contains(training, "Integration gate", "training")
    _assert_contains(training, "implemented-not-integrated", "training")
    _assert_contains(optimization, "contract_id: FRS-PPO-v001", "optimization")
    _assert_contains(optimization, "advantage_normalization = scale_only", "optimization")
    _assert_contains(evaluation, "contract_id: FRS-EVAL-v001", "evaluation")
    _assert_contains(evaluation, "Implementation gate", "evaluation")
    _assert_contains(evaluation, "Integration gate", "evaluation")

    retired_segment_notes = NOTE / "frontres_segment_replay"
    if retired_segment_notes.exists():
        raise AssertionError(
            f"retired standalone Segment Replay notes still exist: {retired_segment_notes}"
        )

    forbidden_active_phrases = (
        "Authority Actor-Critic Contract (RETIRED MAINLINE)",
        "HSL owns continuous proposal magnitude",
        "HRL owns admissibility, not full continuous rho authority",
    )
    for phrase in forbidden_active_phrases:
        if phrase in design:
            raise AssertionError(f"active contract contains historical method text: {phrase}")

    historical_versions = (
        ("v001", "hsl-rho-acceptance", "superseded", "v002"),
        ("v002", "stable-to-repair", "superseded", "v003"),
        ("v003", "tri-anchor-projection", "superseded", "v004"),
        ("v004", "structured-joint-alpha-rho", "rejected", "v005"),
        ("v005", "executable-floor-retention", "superseded", "v006"),
        ("v006", "conditional-repair-authority", "superseded", "v007"),
        ("v007", "proposal-conditioned-acceptance", "superseded", "v008"),
        ("v008", "authority-actor-critic", "stopped", "v009"),
        ("v009", "hsl-binary-acceptance", "superseded", "v010"),
    )
    for version, slug, status, successor in historical_versions:
        relative_path = (
            "frontres_core/contracts/history/method/"
            f"FRS-METHOD-{version}-{slug}.md"
        )
        historical = _read(relative_path)
        contract_id = f"FRS-METHOD-{version}"
        _assert_contains(historical, f"contract_id: {contract_id}", contract_id)
        _assert_contains(historical, f"status: {status}", contract_id)
        _assert_contains(
            historical,
            f"superseded_by: FRS-METHOD-{successor}",
            contract_id,
        )
        _assert_contains(history_index, f"`{contract_id}`", "history index")

    _assert_contains(history_index, "`FRS-METHOD-v010`", "history index")
    _assert_contains(design, "supersedes: FRS-METHOD-v009", "design")

    print("FrontRES design contract sentinel: PASS")


if __name__ == "__main__":
    run_design_contract_sentinel()
