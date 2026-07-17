from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[4]
FRONTRES_DIR = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres"


def _load(name: str, filename: str):
    package_name = "q2_contract_frontres"
    if package_name not in sys.modules:
        package = importlib.util.module_from_spec(importlib.util.spec_from_loader(package_name, loader=None))
        package.__path__ = [str(FRONTRES_DIR)]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", FRONTRES_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifest_module = _load("frontres_policy_quality_manifest", "frontres_policy_quality_manifest.py")
report_module = _load("frontres_policy_quality_q2_report", "frontres_policy_quality_q2_report.py")
Manifest = manifest_module.FrontRESPolicyQualityManifest
ManifestItem = manifest_module.FrontRESPolicyQualityManifestItem
build_report = report_module.build_frontres_policy_quality_q2_report


def _manifest() -> object:
    items = []
    for motion_index in range(8):
        for seed in (42, 43):
            items.append(
                ManifestItem(
                    item_id=f"motion-{motion_index}-seed-{seed}",
                    motion_id=f"motion-{motion_index}.npz",
                    start_frame=10 + motion_index,
                    perturbation_family="local_rp",
                    perturbation_parameters=(("dr_scale", 1.25),),
                    effective_horizon_k=8,
                    seed=seed,
                )
            )
    return Manifest(
        environment_revision="env-v1",
        config_revision="config-v1",
        evaluator_version="quality-v1",
        items=tuple(items),
    )


def _role_identity() -> dict[str, object]:
    return {
        "policy_noisy": {
            "joint_pos_max_abs": 0.0,
            "joint_vel_max_abs": 0.0,
            "local_root_pos_max_abs": 1.0e-7,
            "root_quat_max_abs": 0.0,
            "root_lin_vel_max_abs": 0.0,
            "root_ang_vel_max_abs": 0.0,
            "cached_perturbed_pos_max_abs": 0.0,
            "cached_perturbed_quat_max_abs": 0.0,
            "env_origin_max_abs": 40.0,
            "world_root_pos_max_abs": 40.0,
        },
        "corruption_present": {
            "policy_clean_cached_pos_max_abs": 0.0,
            "policy_clean_cached_quat_max_abs": 0.06,
        },
        "role_rows": {"policy": [0], "candidate": [1], "noisy": [2], "clean": [3]},
    }


def _result(manifest: object) -> dict[str, object]:
    rows = []
    for index, item in enumerate(manifest.items):
        zero = 0.01 + index * 0.0001
        hsl = zero + 0.03
        policy = hsl + (0.02 if index < 10 else -0.002)
        state_hash = f"{index + 1:064x}"
        routes = {}
        for route, checkpoint, gain in (
            ("zero", "zero:no-checkpoint", zero),
            ("hsl", "model_200:hash", hsl),
            ("policy", "model_701:hash", policy),
        ):
            routes[route] = {
                "checkpoint_identity": checkpoint,
                "initial_state_hash": state_hash,
                "gain": {"gain_total": [gain]},
                "actions": [],
                "execution": {},
            }
        rows.append(
            {
                "comparison_signature": item.comparison_signature,
                "item": item.to_dict(),
                "role_identity": _role_identity(),
                "routes": routes,
            }
        )
    return {
        "schema_version": "frontres_policy_quality_result_v1",
        "comparison_signature": manifest.comparison_signature,
        "owner_identity": {},
        "rows": rows,
    }


def _must_fail(manifest: object, result: object, needle: str) -> None:
    try:
        build_report(manifest, result)
    except ValueError as exc:
        assert needle in str(exc), str(exc)
    else:
        raise AssertionError(f"expected Q2 report failure containing {needle!r}")


def test_per_item_noise_floor_delta_and_permutation() -> None:
    manifest = _manifest()
    result = _result(manifest)
    report = build_report(manifest, result)
    assert report["technical_pass"] is True
    assert len(report["items"]) == 16 and len(report["motions"]) == 8
    first = report["items"][0]
    assert abs(first["zero_noise_floor"] - 0.01) < 1.0e-12
    assert abs(first["delta"]["hsl_zero"] - 0.03) < 1.0e-12
    assert abs(first["delta"]["policy_hsl"] - 0.02) < 1.0e-12
    assert first["classification"]["policy_hsl"] == "positive"

    shuffled = deepcopy(result)
    random.Random(7).shuffle(shuffled["rows"])
    assert build_report(manifest, shuffled) == report


def test_negative_scientific_outcome_is_a_report_not_an_exception() -> None:
    manifest = _manifest()
    result = _result(manifest)
    for row in result["rows"]:
        zero = row["routes"]["zero"]["gain"]["gain_total"][0]
        row["routes"]["hsl"]["gain"]["gain_total"] = [zero - 0.03]
        row["routes"]["policy"]["gain"]["gain_total"] = [zero - 0.04]
    report = build_report(manifest, result)
    assert report["technical_pass"] is True
    assert report["verdict"]["oracle_valid"] is False
    assert report["verdict"]["method_review_required"] is True
    assert report["verdict"]["ppo_regression_supported"] is False


def test_identity_schema_gain_and_role_fail_closed() -> None:
    manifest = _manifest()

    missing = _result(manifest)
    missing["rows"].pop()
    _must_fail(manifest, missing, "missing manifest items")

    signature = _result(manifest)
    signature["comparison_signature"] = "0" * 64
    _must_fail(manifest, signature, "signatures differ")

    nonscalar = _result(manifest)
    nonscalar["rows"][0]["routes"]["zero"]["gain"]["gain_total"] = [0.0, 1.0]
    _must_fail(manifest, nonscalar, "exactly one scalar")

    nonfinite = _result(manifest)
    nonfinite["rows"][0]["routes"]["zero"]["gain"]["gain_total"] = [float("nan")]
    _must_fail(manifest, nonfinite, "finite scalar")

    role = _result(manifest)
    role["rows"][0]["role_identity"]["policy_noisy"]["joint_pos_max_abs"] = 0.1
    _must_fail(manifest, role, "matched-role tolerance")

    state = _result(manifest)
    state["rows"][0]["routes"]["policy"]["initial_state_hash"] = "f" * 64
    _must_fail(manifest, state, "initial_state_hash")


if __name__ == "__main__":
    test_per_item_noise_floor_delta_and_permutation()
    test_negative_scientific_outcome_is_a_report_not_an_exception()
    test_identity_schema_gain_and_role_fail_closed()
    print("PASS: Q2 report preserves per-item noise floors and separates technical validity from science.")
