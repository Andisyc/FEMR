#!/usr/bin/env python3
"""Step 12 readiness suite for active FEMR HSL + acceptance.

This is the single pre-training sentinel.  It runs the contract tests created
across Steps 1-11 in order, so a future change cannot silently break config,
policy, rollout labels, storage, loss, runner path, diagnostics, entrypoints,
full toy chain, or legacy-branch retirement.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TEST_DIR = ROOT / "source/rsl_rl/rsl_rl/tests"

READINESS_TESTS = [
    "frontres_design_contract_sentinel.py",
    "frontres_config_hsl_acceptance.py",
    "frontres_hsl_acceptance_policy.py",
    "frontres_acceptance_label_from_rollout.py",
    "frontres_acceptance_storage_contract.py",
    "frontres_hsl_acceptance_algorithm_loss.py",
    "frontres_runner_hsl_acceptance_path.py",
    "frontres_hsl_acceptance_diagnostics.py",
    "frontres_stage_entrypoint_contract.py",
    "frontres_stage2_two_head_checkpoint_migration.py",
    "frontres_hsl_acceptance_full_toy_chain.py",
    "frontres_no_legacy_active_path.py",
]


def _run_test(name: str) -> None:
    path = TEST_DIR / name
    if not path.exists():
        raise AssertionError(f"missing readiness test: {path}")
    print(f"[FrontRES readiness] running {name}", flush=True)
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    for name in READINESS_TESTS:
        _run_test(name)
    print("PASS: active FEMR HSL+acceptance is ready for a short live training smoke test.")


if __name__ == "__main__":
    main()
