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
        requires_torch=True,
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
        name="fixed_noisy_segment_lifecycle",
        path="source/rsl_rl/rsl_rl/tests/frontres_fixed_noisy_segment_lifecycle_contract.py",
        expected_marker="frontres_fixed_noisy_segment_lifecycle_contract: ok",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="full6_no_active_mask",
        path="source/rsl_rl/rsl_rl/tests/frontres_full6_no_active_mask_contract.py",
        expected_marker="frontres_full6_no_active_mask_contract: ok",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="task_space_contact_correction",
        path="source/rsl_rl/rsl_rl/tests/frontres_task_space_correction_contract.py",
        expected_marker="frontres_task_space_correction_contract: ok",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="reward",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_reward_contract.py",
        expected_marker="result: PASS",
        path_class="secondary_contract_path",
    ),
    ContractTarget(
        name="total_reward_scale",
        path="source/rsl_rl/rsl_rl/tests/frontres_total_reward_scale_contract.py",
        expected_marker="frontres_total_reward_scale_contract: ok",
        path_class="secondary_contract_path",
        requires_torch=False,
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
        name="interface_refactor",
        path="source/rsl_rl/rsl_rl/tests/frontres_interface_refactor_contract.py",
        expected_marker="FrontRES formal Stage-3 route is isolated behind typed fail-closed ports",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="rollout_step_action_stats",
        path="source/rsl_rl/rsl_rl/tests/frontres_rollout_step_action_stats_contract.py",
        expected_marker="frontres_rollout_step_action_stats_contract: ok",
        path_class="core_param_path",
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
        name="grouped_ppo",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_grouped_ppo_contract.py",
            expected_marker="frontres_segment_grouped_ppo_contract: scalar v005 ok",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="actual_policy_distribution",
        path="source/rsl_rl/rsl_rl/tests/frontres_actual_policy_distribution_contract.py",
        expected_marker="frontres_actual_policy_distribution_contract: ok",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="proposal_only_task_space_policy",
        path="source/rsl_rl/rsl_rl/tests/frontres_task_space_proposal_only_contract.py",
        expected_marker="frontres_task_space_proposal_only_contract: ok",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="observation_layout",
        path="source/rsl_rl/rsl_rl/tests/frontres_observation_layout_contract.py",
        expected_marker="frontres_observation_layout_contract: ok",
        path_class="core_param_path",
    ),
    ContractTarget(
        name="balance_obs_cfg",
        path="source/rsl_rl/rsl_rl/tests/frontres_balance_obs_cfg_contract.py",
        expected_marker="frontres_balance_obs_cfg_contract: ok",
        path_class="secondary_contract_path",
        requires_torch=False,
    ),
    ContractTarget(
        name="balance_offline_connectivity",
        path="source/rsl_rl/rsl_rl/tests/frontres_balance_offline_connectivity_contract.py",
        expected_marker="frontres_balance_offline_connectivity_contract: ok",
        path_class="core_param_path",
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
        requires_torch=True,
    ),
    ContractTarget(
        name="stage3_actor_critic_warmup",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_warmup_contract.py",
        expected_marker="frontres_segment_warmup_contract: ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="frozen_gmt_gradient_boundary",
        path="source/rsl_rl/rsl_rl/tests/frontres_frozen_gmt_contract.py",
        expected_marker="frontres_frozen_gmt_contract: ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v016_runtime_telemetry",
        path="source/rsl_rl/rsl_rl/tests/frontres_v016_runtime_telemetry_contract.py",
        expected_marker="frontres_v016_runtime_telemetry_contract: final serialized fields exact",
        path_class="secondary_contract_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="gain_v007",
        path="source/rsl_rl/rsl_rl/tests/frontres_gain_v007_contract.py",
        expected_marker="[T-v007-gain]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v017_step1_connectivity",
        path="source/rsl_rl/rsl_rl/tests/frontres_v017_step1_contract.py",
        expected_marker="[T-v017-step1]",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v018_policy_quality_eval",
        path="source/rsl_rl/rsl_rl/tests/frontres_v018_policy_quality_eval_contract.py",
        expected_marker="frontres_v018_policy_quality_eval_contract: ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v018_policy_quality_compatibility",
        path="source/rsl_rl/rsl_rl/tests/frontres_v018_policy_quality_compatibility_contract.py",
        expected_marker="frontres_v018_policy_quality_compatibility_contract: ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v018_policy_quality_batch_adapter",
        path="source/rsl_rl/rsl_rl/tests/frontres_local_scenario_kernel_contract.py",
        expected_marker="[T-v018-heldout] two Segment x M4, K16 and source-shared scenario identities are sealed",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v017_transaction_route",
        path="source/rsl_rl/rsl_rl/tests/frontres_v015_transaction_route_contract.py",
        expected_marker="frontres_v015_transaction_route_contract: v019 symlog state-value exact-one ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v018_support_conditioned_observation",
        path="source/rsl_rl/rsl_rl/tests/frontres_v018_support_conditioned_observation_contract.py",
        expected_marker="frontres_v018_support_conditioned_observation_contract: 289+58+102=449 exact",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v018_m4_schedule",
        path="source/rsl_rl/rsl_rl/tests/frontres_v018_m4_schedule_contract.py",
        expected_marker="frontres_v018_m4_schedule_contract: K8/K16/K32 all M4",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v016_state_value_ppo",
        path="source/rsl_rl/rsl_rl/tests/frontres_v016_state_value_ppo_contract.py",
        expected_marker="frontres_v016_state_value_ppo_contract: segment mean + separate clip exact",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="v016_checkpoint_resume",
        path="source/rsl_rl/rsl_rl/tests/frontres_v016_checkpoint_contract.py",
        expected_marker="frontres_v016_checkpoint_contract: v14 utility round-trip and v13/v10 reject",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="hsl_rollout_target_audit",
        path="source/rsl_rl/rsl_rl/tests/frontres_hsl_rollout_target_contract.py",
        expected_marker="frontres_hsl_rollout_target_contract: v007 reject ok",
        path_class="core_param_path",
        requires_torch=True,
    ),
    ContractTarget(
        name="stage3_live_probe_ppo_boundary",
        path="source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_ppo_contract.py",
        expected_marker="frontres_segment_live_probe_ppo_contract: ok",
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
        name="fixed_noisy_actor_context",
        path="source/rsl_rl/rsl_rl/tests/frontres_fixed_noisy_actor_context_contract.py",
        expected_marker="frontres_fixed_noisy_actor_context_contract: ok",
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
    env = os.environ.copy()
    source_root = str(ROOT / "source" / "rsl_rl")
    prior_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = source_root if not prior_pythonpath else os.pathsep.join((source_root, prior_pythonpath))
    try:
        result = subprocess.run(
            [python_path, contract.path],
            cwd=ROOT,
            env=env,
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
