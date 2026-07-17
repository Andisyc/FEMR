#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = ROOT / "note/testing/manifests/frontres_policy_quality_q1f_single_v1.json"
FREEZE_PATH = ROOT / "note/testing/frontres_policy_quality_q1f_input_freeze.md"
MODULE_PATH = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_policy_quality_manifest.py"

spec = importlib.util.spec_from_file_location("frontres_policy_quality_q1f_manifest", MODULE_PATH)
manifest_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = manifest_module
spec.loader.exec_module(manifest_module)


def main() -> None:
    manifest = manifest_module.FrontRESPolicyQualityManifest.from_json(MANIFEST_PATH.read_text())
    assert len(manifest.items) == 1
    item = manifest.items[0]
    assert item.motion_id == "KIT/572/amass_g1_wave_right02_poses_reflect.npz"
    assert item.start_frame == 163
    assert item.perturbation_family == "local_rp"
    assert dict(item.perturbation_parameters) == {"dr_scale": 1.25}
    assert item.effective_horizon_k == 8
    assert item.seed == 42
    assert len(item.comparison_signature) == 64
    assert len(manifest.comparison_signature) == 64

    freeze = FREEZE_PATH.read_text()
    assert "model_200.pt" in freeze and "model_701.pt" in freeze
    assert "SHA-256 hashes" in freeze and "UNCONFIRMED" in freeze
    assert "Q1-F remains blocked" in freeze
    print(
        "PASS: Q1-F single-item inputs are immutable and reviewable; "
        f"item_signature={item.comparison_signature} "
        f"manifest_signature={manifest.comparison_signature}"
    )


if __name__ == "__main__":
    main()
