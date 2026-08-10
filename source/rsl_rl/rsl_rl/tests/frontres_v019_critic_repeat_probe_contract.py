from __future__ import annotations

import copy
import math
from pathlib import Path

from frontres_contract_imports import install_frontres_contract_packages


install_frontres_contract_packages()

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "note/testing/manifests/frontres_v019_critic_repeat_k8_m4_v1.json"

from rsl_rl.frontres.frontres_policy_quality_manifest import (  # noqa: E402
    FrontRESV018PolicyQualityManifest,
)
from rsl_rl.runners import frontres_policy_quality_eval as quality  # noqa: E402


def _transaction(repeat_index: int, target_a: float, target_b: float, action_offset: float) -> dict:
    actions = [
        [action_offset + 0.01 * row + 0.001 * axis for axis in range(6)]
        for row in range(8)
    ]
    return {
        "repeat_index": repeat_index,
        "item_ids": ["fixed-a", "fixed-b"],
        "item_comparison_signatures": ["a" * 64, "b" * 64],
        "x_t_identities": ["x-t-a"] * 4 + ["x-t-b"] * 4,
        "critic_input_rows": [[0.0] * 449, [1.0] * 449],
        "critic_calibration": [
            {
                "source_index": 0,
                "segment_id": 10,
                "scenario_id": "scenario-a",
                "noisy_segment_hash": "hash-a",
                "attempt_count": 4,
                "policy_value": 0.25,
                "raw_target_mean": 10.0 * target_a,
                "target_mean": target_a,
                "value_error": 0.25 - target_a,
            },
            {
                "source_index": 1,
                "segment_id": 20,
                "scenario_id": "scenario-b",
                "noisy_segment_hash": "hash-b",
                "attempt_count": 4,
                "policy_value": -0.1,
                "raw_target_mean": 10.0 * target_b,
                "target_mean": target_b,
                "value_error": -0.1 - target_b,
            },
        ],
        "report": {"policy_actions": actions},
    }


def test_repeat_diagnostics_preserve_fixed_segment_identity_and_hand_statistics() -> None:
    transactions = [
        _transaction(0, 0.0, -1.0, 0.0),
        _transaction(1, 1.0, 0.0, 1.0),
        _transaction(2, 2.0, 1.0, 2.0),
    ]
    result = quality.build_frontres_v019_critic_repeat_diagnostics(transactions, repeat_count=3)

    assert result["schema_version"] == "frontres-v019-critic-repeat-diagnostics-v1"
    assert (result["repeat_count"], result["fixed_segment_count"]) == (3, 2)
    first, second = result["segments"]
    assert first["item_id"] == "fixed-a" and second["item_id"] == "fixed-b"
    assert first["repeat_target_means"] == [0.0, 1.0, 2.0]
    assert first["target_mean"] == 1.0
    assert math.isclose(first["target_std"], math.sqrt(2.0 / 3.0), rel_tol=0.0, abs_tol=1e-12)
    assert (first["target_min"], first["target_max"]) == (0.0, 2.0)
    assert first["critic_policy_value"] == 0.25
    assert first["critic_error_to_repeat_mean"] == -0.75
    assert first["critic_input_max_abs_diff"] == 0.0
    assert len(set(first["action_fingerprints"])) == 3


def test_repeat_diagnostics_tolerate_roundoff_but_reject_critic_state_drift() -> None:
    roundoff = [_transaction(0, 0.0, -1.0, 0.0), _transaction(1, 1.0, 0.0, 1.0)]
    roundoff[1]["critic_input_rows"][0][17] = 1.0e-4
    result = quality.build_frontres_v019_critic_repeat_diagnostics(roundoff, repeat_count=2)
    assert math.isclose(result["segments"][0]["critic_input_max_abs_diff"], 1.0e-4)

    drifted = copy.deepcopy(roundoff)
    drifted[1]["critic_input_rows"][0][17] = 1.0e-2
    try:
        quality.build_frontres_v019_critic_repeat_diagnostics(drifted, repeat_count=2)
    except RuntimeError as exc:
        assert "Critic input drift" in str(exc), str(exc)
    else:
        raise AssertionError("repeat diagnostics must reject material Critic-input drift")


def test_repeat_diagnostics_fail_closed_on_identity_or_action_collapse() -> None:
    base = [_transaction(0, 0.0, -1.0, 0.0), _transaction(1, 1.0, 0.0, 1.0)]

    drifted = copy.deepcopy(base)
    drifted[1]["critic_calibration"][0]["noisy_segment_hash"] = "changed"
    try:
        quality.build_frontres_v019_critic_repeat_diagnostics(drifted, repeat_count=2)
    except RuntimeError as exc:
        assert "fixed Segment identity" in str(exc), str(exc)
    else:
        raise AssertionError("repeat diagnostics must reject scenario/hash drift")

    collapsed = copy.deepcopy(base)
    collapsed[1]["report"]["policy_actions"] = copy.deepcopy(collapsed[0]["report"]["policy_actions"])
    try:
        quality.build_frontres_v019_critic_repeat_diagnostics(collapsed, repeat_count=2)
    except RuntimeError as exc:
        assert "distinct Repair actions" in str(exc), str(exc)
    else:
        raise AssertionError("repeat diagnostics must reject identical Repair-action groups")


def test_k8_repeat_manifest_is_strict_and_minimal() -> None:
    manifest = FrontRESV018PolicyQualityManifest.from_json(MANIFEST.read_text(encoding="utf-8"))
    assert manifest.horizon_k == 8
    assert manifest.attempts_per_segment == 4
    assert manifest.segments_per_transaction == 2
    assert len(manifest.items) == 2
    assert all(item.effective_horizon_k == 8 for item in manifest.items)


if __name__ == "__main__":
    test_repeat_diagnostics_preserve_fixed_segment_identity_and_hand_statistics()
    test_repeat_diagnostics_tolerate_roundoff_but_reject_critic_state_drift()
    test_repeat_diagnostics_fail_closed_on_identity_or_action_collapse()
    test_k8_repeat_manifest_is_strict_and_minimal()
    print("frontres_v019_critic_repeat_probe_contract: ok")
