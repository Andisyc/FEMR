"""Static route contract for the v010 raw Clean producer.

This is intentionally R0 route evidence.  It does not claim an IsaacLab
composition-root transaction when the local runtime has no Torch/Isaac scene.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
GATEWAY = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_clean_calibration_gateway.py"
RUNNER = ROOT / "source/rsl_rl/rsl_rl/runners/on_policy_runner.py"
TRAIN = ROOT / "scripts/rsl_rl/train.py"


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"missing function {name} in {path}")


def main() -> None:
    raw_gateway = _function_source(GATEWAY, "collect_frontres_clean_calibration_raw_gateway")
    raw_producer = _function_source(GATEWAY, "_collect_raw_clean_collection")
    manifest_connector = _function_source(GATEWAY, "collect_frontres_clean_calibration_from_manifest")
    raw_gateway += raw_producer
    for required in (
        "frontres_readonly_collection_scope",
        "collect_frontres_v017_no_actor_baseline",
        "clean_baseline",
        "adapt_read_only_clean_collection",
    ):
        assert required in raw_gateway, f"raw gateway lost required owner {required}"
    for forbidden in (
        "collect_frontres_v017_repair_attempts",
        "compute_recovery_aware_gain",
        "optimizer.step",
        "noisy_baseline",
    ):
        assert forbidden not in raw_gateway, f"raw gateway must not invoke {forbidden}"

    runner_source = RUNNER.read_text(encoding="utf-8")
    train_source = TRAIN.read_text(encoding="utf-8")
    assert "run_frontres_clean_calibration_collect_typed" in runner_source
    assert "prepare_frontres_fixed_k_m4_evaluation_batch" in manifest_connector
    assert "typed_connector" in manifest_connector
    assert "typed_collection_artifact" not in manifest_connector
    assert "--frontres_clean_calibration_collect_only" in train_source
    assert (
        '_set_if_present(alg_cfg, "frontres_policy_quality_eval_only", policy_quality_eval_arg)'
        in train_source
    ), "clean route must remain separate from policy-quality"
    print("frontres_clean_calibration_raw_gateway_route_contract: R0_ROUTE_STATIC_PASS")


if __name__ == "__main__":
    main()
