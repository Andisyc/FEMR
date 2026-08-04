from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
install_frontres_contract_packages(SOURCE_ROOT)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load(
    "rsl_rl.frontres.frontres_policy_quality_manifest",
    SOURCE_ROOT / "frontres" / "frontres_policy_quality_manifest.py",
)
quality = _load(
    "rsl_rl.runners.frontres_policy_quality_eval",
    SOURCE_ROOT / "runners" / "frontres_policy_quality_eval.py",
)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "frontres_policy_quality_manifest_v1",
        "environment_revision": "env",
        "config_revision": "cfg",
        "evaluator_version": "quality-v1",
        "items": [
            {
                "item_id": "a",
                "motion_id": "KIT/a.npz",
                "start_frame": 3,
                "perturbation_family": "local_rp",
                "perturbation_parameters": [["roll_rad", 0.1]],
                "effective_horizon_k": 8,
                "seed": 1,
            },
            {
                "item_id": "b",
                "motion_id": "CMU/b.npz",
                "start_frame": 9,
                "perturbation_family": "local_rp",
                "perturbation_parameters": [["pitch_rad", -0.1]],
                "effective_horizon_k": 16,
                "seed": 2,
            },
        ],
    }


def test_manifest_executor_uses_all_named_owners_and_writes_atomic_result() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path = root / "manifest.json"
        hsl_path = root / "hsl.pt"
        policy_path = root / "policy.pt"
        result_path = root / "result.json"
        manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
        hsl_path.write_bytes(b"hsl")
        policy_path.write_bytes(b"policy")
        runner = SimpleNamespace(training_state={"optimizer": 1, "sampler": 2, "warmup": 3})
        prepared: list[tuple[str, int]] = []
        routed: list[tuple[str, int]] = []

        def prepare(_runner, item, _request):
            prepared.append((item.item_id, item.effective_horizon_k))
            return f"snapshot:{item.item_id}", ("zero", "hsl", "policy"), f"hooks:{item.item_id}"

        original = quality.run_frontres_policy_quality_counterfactuals

        def fake_counterfactuals(_runner, *, snapshot, comparison_signature, adapters, hooks, horizon_k, isolation_state):
            assert adapters == ("zero", "hsl", "policy")
            assert isolation_state() == repr(runner.training_state)
            routed.append((snapshot, horizon_k))
            return (SimpleNamespace(route="zero"), SimpleNamespace(route="hsl"), SimpleNamespace(route="policy"))

        quality.run_frontres_policy_quality_counterfactuals = fake_counterfactuals
        try:
            owners = quality.FrontRESPolicyQualityFormalOwnerBundle(
                owner_identity=(
                    ("reset", "frontres_segment_reset.FrontRESSegmentResetAdapter"),
                    ("observation", "OnPolicyRunner._apply_obs_normalizer"),
                    ("action", "task_space_correction.apply_frontres_task_corrections"),
                    ("rollout", "policy_quality_eval.run_frontres_policy_quality_counterfactuals"),
                    ("gain", "frontres_gain.compute_segment_gain"),
                    ("execution", "policy_quality_eval.FrontRESPolicyQualityRouteResult"),
                ),
                prepare_item=prepare,
                isolation_state=lambda current: repr(current.training_state),
                serialize_result=lambda item, results: {
                    "item_id": item.item_id,
                    "routes": [result.route for result in results],
                },
            )
            quality.install_frontres_policy_quality_manifest_executor(runner, owners)
            payload = quality.run_frontres_legacy_policy_quality_eval(
                runner,
                manifest_path=str(manifest_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(result_path),
            )
        finally:
            quality.run_frontres_policy_quality_counterfactuals = original

        assert prepared == [("a", 8), ("b", 16)]
        assert routed == [("snapshot:a", 8), ("snapshot:b", 16)]
        assert payload == json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["rows"][0]["routes"] == ["zero", "hsl", "policy"]
        assert not result_path.with_suffix(".json.tmp").exists()
        assert runner.training_state == {"optimizer": 1, "sampler": 2, "warmup": 3}


def test_owner_bundle_and_double_install_fail_closed() -> None:
    try:
        quality.FrontRESPolicyQualityFormalOwnerBundle(
            owner_identity=(("reset", "owner"),),
            prepare_item=lambda *_args: None,
            isolation_state=lambda _runner: "fixed",
            serialize_result=lambda *_args: {},
        )
    except ValueError as exc:
        assert "must name exactly" in str(exc)
    else:
        raise AssertionError("incomplete formal owner bundle must fail closed")


if __name__ == "__main__":
    test_manifest_executor_uses_all_named_owners_and_writes_atomic_result()
    test_owner_bundle_and_double_install_fail_closed()
    print("PASS: policy-quality formal manifest executor preflight is closed offline.")
