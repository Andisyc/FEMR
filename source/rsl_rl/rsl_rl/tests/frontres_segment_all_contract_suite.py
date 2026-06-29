#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CONTRACT_TIMEOUT_SEC = float(os.environ.get("FRONTRES_SEGMENT_CONTRACT_TIMEOUT_SEC", "60"))


@dataclass(frozen=True)
class ContractTarget:
    name: str
    path: str
    expected_marker: str
    path_class: str
    requires_torch: bool = True


CONTRACTS = (
    ContractTarget(
        name="cache_schema",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_cache_schema_contract.py",
        expected_marker="PASS: FrontRES Segment cache schema validates ids and rollout state tensors.",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="cache_indexer",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_cache_indexer_contract.py",
        expected_marker="PASS: FrontRES AMASS indexer builds segment index from motion paths and frame counts.",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="cache_io",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_cache_io_contract.py",
        expected_marker="PASS: FrontRES Segment cache IO round-trips clean states and noisy variants.",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="cache_validator",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_cache_validator_contract.py",
        expected_marker="PASS: FrontRES Stage 1 cache validator reads back metadata and clean/noisy shards.",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="cache_extractor",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_cache_extractor_contract.py",
        expected_marker="PASS: FrontRES clean state extractor captures detached robot rollout state.",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="cache_perturbation",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_cache_perturbation_contract.py",
        expected_marker="PASS: FrontRES perturbation curriculum descriptors are reproducible and indexed.",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="cache_curriculum",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_cache_curriculum_contract.py",
        expected_marker="PASS: FrontRES Stage 1 curriculum bank derives cache levels from HRL perturbation curriculum.",
        path_class="core_param_path",
        requires_torch=False,
    ),
    ContractTarget(
        name="cache_noisy_capture",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_cache_noisy_capture_contract.py",
        expected_marker="PASS: FrontRES noisy capture interface builds noisy variants through reset and perturbation hooks.",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="cache_builder",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_cache_builder_contract.py",
        expected_marker="PASS: FrontRES Stage 1 cache builder orchestrates index, clean, perturbation, noisy, and IO.",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="stage1_env_hooks",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_stage1_env_hooks_contract.py",
        expected_marker="PASS: FrontRES Stage 1 env adapter hooks trace motion, clean reset, perturbation, and baseline rollout.",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="dataset",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_dataset_contract.py",
        expected_marker="result: PASS",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="sampler",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_sampler_contract.py",
        expected_marker="result: PASS",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="hrl_action",
        path="source/rsl_rl/rsl_rl/tests/frontres_hrl_action_contract.py",
        expected_marker="result: PASS",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="reward",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_reward_contract.py",
        expected_marker="result: PASS",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="reset",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_reset_contract.py",
        expected_marker="result: PASS",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="diagnostics",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_diagnostics_contract.py",
        expected_marker="result: PASS",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="runner_boundary",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_runner_boundary_contract.py",
        expected_marker="result: PASS",
        path_class="live_sentinel_path",
    ),
    ContractTarget(
        name="runner_toy_chain",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_replay_toy_chain.py",
        expected_marker="result: PASS",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="storage",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py",
        expected_marker="result: PASS",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="algorithm",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_algorithm_contract.py",
        expected_marker="result: PASS",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="checkpoint",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py",
        expected_marker="result: PASS",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="runner_lifecycle",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_runner_lifecycle_contract.py",
        expected_marker="result: PASS",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="live_sentinel",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_sentinel_contract.py",
        expected_marker="result: PASS",
        path_class="live_sentinel_path",
    ),
    ContractTarget(
        name="stage3_pseudo_suite",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_pseudo_suite.py",
        expected_marker="frontres_segment_stage3_pseudo_suite: ok",
        path_class="secondary_contract_path",
        requires_torch=False,
    ),
    ContractTarget(
        name="stage3_contract_preflight",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_contract_preflight_contract.py",
        expected_marker="frontres_segment_stage3_contract_preflight_contract: ok",
        path_class="live_sentinel_path",
        requires_torch=False,
    ),
    ContractTarget(
        name="stage3_contract_failure_preflight",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_contract_failure_preflight_contract.py",
        expected_marker="frontres_segment_stage3_contract_failure_preflight_contract: ok",
        path_class="live_sentinel_path",
        requires_torch=False,
    ),
    ContractTarget(
        name="stage3_live_resume",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_resume_pseudo_contract.py",
        expected_marker="frontres_segment_live_resume_pseudo_contract: ok",
        path_class="core_param_path",
        requires_torch=False,
    ),
    ContractTarget(
        name="stage3_live_sampler",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py",
        expected_marker="frontres_segment_live_sampler_contract: ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="stage3_per_sample_evidence",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py",
        expected_marker="[probe step14]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="stage3_live_reset_hook",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_reset_hook_contract.py",
        expected_marker="frontres_segment_live_reset_hook_contract: ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="stage3_reference_window_hook",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_reset_hook_contract.py",
        expected_marker="[probe step15]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="stage3_motion_command_reference",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_motion_command_reference_contract.py",
        expected_marker="frontres_segment_motion_command_reference_contract: ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="stage3_local_closed_loop",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_closed_loop_contract.py",
        expected_marker="frontres_segment_live_closed_loop_contract: ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
)


def _can_import_torch(python_path: str) -> bool:
    result = subprocess.run(
        [python_path, "-c", "import torch"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0


@lru_cache(maxsize=1)
def _torch_python() -> str:
    candidates = [
        str(ROOT / "frontres" / "bin" / "python"),
        str(ROOT.parent / "MOSAIC" / "frontres" / "bin" / "python"),
        sys.executable,
    ]
    for candidate in candidates:
        if Path(candidate).exists() and _can_import_torch(candidate):
            return candidate
    raise RuntimeError("No torch-capable Python found for Segment Replay contract suite.")


def _python_for_contract(contract: ContractTarget) -> str:
    return _torch_python() if contract.requires_torch else sys.executable


def _run_contract(contract: ContractTarget) -> tuple[int, bool, int]:
    python_path = _python_for_contract(contract)
    try:
        result = subprocess.run(
            [python_path, contract.path],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=CONTRACT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        print(
            f"[probe step9] {contract.name}: "
            f"class={contract.path_class} "
            f"python={python_path} "
            f"timeout_sec={CONTRACT_TIMEOUT_SEC} "
            "observed_marker=False",
            flush=True,
        )
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        print(f"[probe step9] {contract.name}: stdout_begin", flush=True)
        print(stdout, flush=True)
        print(f"[probe step9] {contract.name}: stderr_begin", flush=True)
        print(stderr, flush=True)
        return 124, False, 0
    combined = result.stdout + result.stderr
    marker_count = combined.count(contract.expected_marker)
    observed_marker = marker_count > 0
    print(
        f"[probe step9] {contract.name}: "
        f"class={contract.path_class} "
        f"python={python_path} "
        f"returncode={result.returncode} "
        f"expected_marker={contract.expected_marker!r} "
        f"marker_count={marker_count} "
        f"observed_marker={observed_marker}",
        flush=True,
    )
    if result.returncode != 0 or not observed_marker:
        print(f"[probe step9] {contract.name}: stdout_begin", flush=True)
        print(result.stdout, flush=True)
        print(f"[probe step9] {contract.name}: stderr_begin", flush=True)
        print(result.stderr, flush=True)
    return result.returncode, observed_marker, marker_count


def main() -> None:
    failed = []
    total_marker_count = 0
    for contract in CONTRACTS:
        returncode, observed_marker, marker_count = _run_contract(contract)
        total_marker_count += marker_count
        if returncode != 0 or not observed_marker:
            failed.append(contract.name)

    print(
        f"[probe step9] suite_summary: "
        f"contract_count={len(CONTRACTS)} "
        f"failed_count={len(failed)} "
        f"total_marker_count={total_marker_count}",
        flush=True,
    )
    assert not failed, f"Segment Replay contract suite failed: {failed}"
    assert total_marker_count >= len(CONTRACTS)
    print("frontres_segment_all_contract_suite: ok")


if __name__ == "__main__":
    main()
