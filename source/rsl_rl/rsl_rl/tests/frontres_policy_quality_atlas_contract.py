#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ATLAS_PATH = ROOT / "note" / "architecture" / "runtime" / "05_policy_quality_audit.data.json"

EXPECTED_IDS = (
    "QUALITY-ID-01",
    "QUALITY-DATA-01",
    "QUALITY-ACTION-01",
    "QUALITY-GAIN-01",
    "QUALITY-CREDIT-01",
    "QUALITY-UPDATE-01",
    "QUALITY-EXEC-01",
    "QUALITY-TRAJECTORY-01",
)


def main() -> None:
    atlas = json.loads(ATLAS_PATH.read_text())
    cards = [card for system in atlas["systems"] for card in system.get("modules", [])]

    assert atlas["layout"] == "repository_reading_atlas"
    assert tuple(atlas["runtimeOrder"]) == EXPECTED_IDS
    assert tuple(card["id"] for card in cards) == EXPECTED_IDS

    for card in cards:
        assert card["cardKind"] == "quality_probe"
        assert card["parentDesignPoint"]
        assert card["question"]
        assert card["failureOwner"]
        expected_blocks = 4 if card["id"] == "QUALITY-ID-01" else 3
        assert len(card["files"]) == expected_blocks
        assert len(card["probeSteps"]) == expected_blocks
        assert len(card["mainRoute"]) == expected_blocks
        for index, (file_block, probe_step) in enumerate(zip(card["files"], card["probeSteps"]), start=1):
            assert file_block["id"] == f"{card['id']}-B{index}"
            assert file_block["sourceLine"] == probe_step["sourceLine"]
            assert file_block["sourceHref"] == probe_step["sourceHref"]
            source_path = ROOT / file_block["path"]
            source_line = source_path.read_text().splitlines()[file_block["sourceLine"] - 1]
            assert f"# B{index}:" in source_line

    print("PASS: policy-quality eight-owner Atlas is source-linked and governance-readable.")


if __name__ == "__main__":
    main()
