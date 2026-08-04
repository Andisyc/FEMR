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


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "frontres_policy_quality_manifest_v1",
        "environment_revision": "env-rev",
        "config_revision": "config-rev",
        "evaluator_version": "quality-v1",
        "items": [
            {
                "item_id": "item-0",
                "motion_id": "KIT/example.npz",
                "start_frame": 12,
                "perturbation_family": "local_rp",
                "perturbation_parameters": [["roll_rad", 0.1]],
                "effective_horizon_k": 8,
                "seed": 7,
            }
        ],
    }


def test_request_validation_and_single_owner_dispatch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "manifest.json"
        hsl = root / "model_200.pt"
        policy = root / "model_701.pt"
        result = root / "quality.json"
        manifest.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
        hsl.write_bytes(b"hsl-checkpoint-placeholder")
        policy.write_bytes(b"policy-checkpoint-placeholder")
        calls: list[object] = []
        runner = SimpleNamespace(
            _frontres_policy_quality_manifest_executor=lambda request: calls.append(request) or {"status": "ok"}
        )
        observed = quality.run_frontres_legacy_policy_quality_eval(
            runner,
            manifest_path=str(manifest),
            hsl_checkpoint_path=str(hsl),
            policy_checkpoint_path=str(policy),
            result_path=str(result),
        )
        assert observed == {"status": "ok"}
        assert len(calls) == 1
        request = calls[0]
        assert request.manifest.items[0].effective_horizon_k == 8
        assert request.hsl_checkpoint_path != request.policy_checkpoint_path

        source = (SOURCE_ROOT / "runners" / "frontres_policy_quality_eval.py").read_text(encoding="utf-8")
        assert "build_frontres_policy_quality_formal_owner_bundle(runner, request)" in source
        assert "install_frontres_policy_quality_manifest_executor(runner, owners)" in source


def test_active_entry_rejects_implicit_legacy_fallback() -> None:
    runner = SimpleNamespace(alg=SimpleNamespace(frontres_formal_transaction_enabled=False))
    try:
        quality.run_frontres_policy_quality_eval(
            runner,
            manifest_path="unused.json",
            hsl_checkpoint_path="unused-hsl.pt",
            policy_checkpoint_path="unused-policy.pt",
            result_path="unused-result.json",
        )
    except RuntimeError as exc:
        assert "run_frontres_legacy_policy_quality_eval explicitly" in str(exc)
    else:
        raise AssertionError("active policy-quality entry silently selected legacy semantics")


def test_cli_and_runner_are_dedicated_and_lazy() -> None:
    train = (ROOT / "scripts" / "rsl_rl" / "train.py").read_text(encoding="utf-8")
    runner = (SOURCE_ROOT / "runners" / "on_policy_runner.py").read_text(encoding="utf-8")
    shell = (ROOT / "run" / "run_frontres_stage3_segment_hrl.sh").read_text(encoding="utf-8")

    for flag in (
        "--frontres_policy_quality_eval_only",
        "--frontres_policy_quality_manifest",
        "--frontres_policy_quality_hsl_checkpoint",
        "--frontres_policy_quality_policy_checkpoint",
        "--frontres_policy_quality_result",
    ):
        assert flag in train
    quality_dispatch = train.index('if bool(getattr(args_cli, "frontres_policy_quality_eval_only", False)):')
    live_train_dispatch = train.index("runner.learn_frontres_segment_live(", quality_dispatch)
    assert quality_dispatch < live_train_dispatch
    quality_block = train[quality_dispatch:live_train_dispatch]
    assert quality_block.count("runner.run_frontres_policy_quality_eval(") == 1
    assert "run_frontres_segment_offline_eval" not in quality_block
    assert "run_frontres_segment_sequence_offline_eval" not in quality_block
    assert "learn_frontres_segment_live" not in quality_block
    assert "conflicting_modes" in quality_block

    method_at = runner.index("def run_frontres_policy_quality_eval(")
    lazy_import_at = runner.index(
        "from rsl_rl.runners.frontres_policy_quality_eval import run_frontres_policy_quality_eval",
        method_at,
    )
    assert lazy_import_at > method_at
    assert runner[:method_at].count("frontres_policy_quality_eval") == 0
    assert "policy_quality_eval)" not in shell
    assert "Evaluation is launched independently" in shell
    assert '--frontres_v015_hsl_initializer_checkpoint "${HSL_CHECKPOINT}"' in shell
    assert "STAGE3_IS_FULL_RESUME" not in shell
    assert '--resume_student_checkpoint "${HSL_CHECKPOINT}"' not in shell


if __name__ == "__main__":
    test_request_validation_and_single_owner_dispatch()
    test_active_entry_rejects_implicit_legacy_fallback()
    test_cli_and_runner_are_dedicated_and_lazy()
    print("PASS: dedicated policy-quality entrypoint and old-mode isolation are closed offline.")
