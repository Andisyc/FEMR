#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PseudoContract:
    name: str
    path: str
    expected_probe: str
    path_class: str
    requires_torch: bool = False


CONTRACTS = (
    PseudoContract(
        name="step1_live_storage_write",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py",
        expected_probe="[probe step1]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step2_storage_to_ppo",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_ppo_contract.py",
        expected_probe="[probe step2]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step3_single_update",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_single_update_contract.py",
        expected_probe="[probe step3]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step4_update_loop",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_update_loop_contract.py",
        expected_probe="[FrontRES Segment Live Update Loop]",
        path_class="secondary_contract_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step5_live_training_loop",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_training_pseudo_contract.py",
        expected_probe="[probe step5]",
        path_class="secondary_contract_path",
    ),
    PseudoContract(
        name="step21_live_training_resume",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_resume_pseudo_contract.py",
        expected_probe="[probe step21]",
        path_class="core_param_path",
    ),
    PseudoContract(
        name="step22_live_sampler",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py",
        expected_probe="[probe step22]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step14_per_sample_evidence",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py",
        expected_probe="[probe step14]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step13_live_reset_hook",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_reset_hook_contract.py",
        expected_probe="[probe step13]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step15_reference_window_hook",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_reset_hook_contract.py",
        expected_probe="[probe step15]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step16_motion_command_reference",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_motion_command_reference_contract.py",
        expected_probe="[probe step16]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step17_local_closed_loop",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_closed_loop_contract.py",
        expected_probe="[probe step17]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    PseudoContract(
        name="step6_stage3_entrypoint",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_entrypoint_pseudo_contract.py",
        expected_probe="[probe step6]",
        path_class="secondary_contract_path",
    ),
    PseudoContract(
        name="step7_launch_command",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_launch_command_contract.py",
        expected_probe="[probe step7]",
        path_class="live_sentinel_path",
    ),
    PseudoContract(
        name="stage_entrypoint_guard",
        path="source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py",
        expected_probe="PASS: FrontRES Stage",
        path_class="secondary_contract_path",
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
    raise RuntimeError(
        "No torch-capable Python found for core-param Stage 3 contracts. "
        "Expected FEMR frontres/bin/python to import torch."
    )


def _python_for_contract(contract: PseudoContract) -> str:
    return _torch_python() if contract.requires_torch else sys.executable


def _run_contract(contract: PseudoContract) -> tuple[int, bool, int]:
    python_path = _python_for_contract(contract)
    result = subprocess.run(
        [python_path, contract.path],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    combined = result.stdout + result.stderr
    probe_count = combined.count(contract.expected_probe)
    observed_probe = probe_count > 0
    print(
        f"[probe step8] {contract.name}: "
        f"class={contract.path_class} "
        f"python={python_path} "
        f"returncode={result.returncode} "
        f"expected_probe={contract.expected_probe!r} "
        f"probe_count={probe_count} "
        f"observed_probe={observed_probe}",
        flush=True,
    )
    if result.returncode != 0 or not observed_probe:
        print(f"[probe step8] {contract.name}: stdout_begin", flush=True)
        print(result.stdout, flush=True)
        print(f"[probe step8] {contract.name}: stderr_begin", flush=True)
        print(result.stderr, flush=True)
    return result.returncode, observed_probe, probe_count


def main() -> None:
    failed = []
    total_probe_count = 0
    for contract in CONTRACTS:
        returncode, observed_probe, probe_count = _run_contract(contract)
        total_probe_count += probe_count
        if returncode != 0 or not observed_probe:
            failed.append(contract.name)

    print(
        f"[probe step8] suite_summary: "
        f"contract_count={len(CONTRACTS)} "
        f"failed_count={len(failed)} "
        f"total_probe_count={total_probe_count}",
        flush=True,
    )
    assert not failed, f"Stage 3 pseudo suite failed: {failed}"
    assert total_probe_count >= len(CONTRACTS)
    print("frontres_segment_stage3_pseudo_suite: ok")


if __name__ == "__main__":
    main()
