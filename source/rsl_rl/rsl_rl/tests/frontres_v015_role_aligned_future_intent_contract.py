#!/usr/bin/env python3
"""Deterministic S1 contract for the role-aligned v015 q29 H bridge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
TEST_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl" / "tests"
RESET_HELPER_PATH = TEST_ROOT / "frontres_v015_two_role_reset_contract.py"
CURRENT_HELPER_PATH = TEST_ROOT / "frontres_v015_current_gmt_command_contract.py"
ACTOR_HELPER_PATH = TEST_ROOT / "frontres_future_intent_actor_context_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_error(exc_type, callback, contains: str) -> None:
    try:
        callback()
    except exc_type as exc:
        assert contains in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {exc_type.__name__} containing {contains!r}")


def _owners():
    reset_helper = _load("frontres_v015_role_intent_reset_helper", RESET_HELPER_PATH)
    current_helper = _load("frontres_v015_role_intent_current_helper", CURRENT_HELPER_PATH)
    commands, hooks, setup = reset_helper._load_owners()
    command, _request = current_helper._sealed_role_command(reset_helper, commands, hooks, setup)
    env = reset_helper._FakeEnv(command, command.robot, num_envs=8)

    actor_helper = _load("frontres_v015_role_intent_actor_helper", ACTOR_HELPER_PATH)
    layout_module, runtime = actor_helper._load_modules()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    poison_batch = SimpleNamespace(
        frontres_local_scenario_intent_q29=torch.full((4, 3, 29), -7777.0),
        frontres_local_scenario_provenance=actor_helper._provenance(4),
        frontres_future_offsets=(1, 2),
    )
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        alg=SimpleNamespace(policy=SimpleNamespace(num_actor_obs=63, num_frontres_obs=60)),
        _frontres_gmt_obs_dim=3,
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_layout_version=layout.version,
        _frontres_future_intent_actor_context_dim=layout.actor_tail_dim,
        _frontres_segment_live_current_batch=poison_batch,
    )
    return reset_helper, commands, command, runtime, layout, runner


def test_t_role_aligned_offset_and_policy_batch_isolation() -> None:
    _helper, _commands, command, runtime, layout, runner = _owners()
    snapshot = command.frontres_local_scenario_intent_snapshot()
    assert set(snapshot) == {
        "intent_q29",
        "scenario_ids",
        "noisy_segment_hashes",
        "x_t_identities",
        "roles",
        "provenance",
    }
    assert tuple(snapshot["intent_q29"].shape) == (8, 3, 29)
    assert snapshot["roles"] == ("repair", "repair", "repair", "repair", "noisy", "noisy", "noisy", "noisy")

    raw_obs = torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5)
    augmented = runtime.append_frontres_future_intent_context(runner, raw_obs)
    expected_tail = torch.cat([snapshot["intent_q29"][:, 1], snapshot["intent_q29"][:, 2]], dim=-1)
    assert tuple(augmented.shape) == (8, 63)
    torch.testing.assert_close(augmented[:, : layout.actor_tail_dim], expected_tail)
    torch.testing.assert_close(augmented[:, layout.actor_tail_dim :], raw_obs)
    torch.testing.assert_close(augmented[:4, : layout.actor_tail_dim], augmented[4:, : layout.actor_tail_dim])
    assert not bool((augmented[:, : layout.actor_tail_dim] == -7777.0).any())
    print(
        "[T-role-expand/T-offset/T-policy-batch-isolation] "
        "command_intent=(8,3,29) offsets=(1,2) tail=(8,58) policy_batch_B4_ignored=true",
        flush=True,
    )


def test_t_snapshot_read_only_and_no_forbidden_fields() -> None:
    _helper, _commands, command, runtime, layout, runner = _owners()
    before = command.frontres_local_scenario_intent_snapshot()
    before["intent_q29"].fill_(-9999.0)
    before["provenance"][0]["intent_q29_source"] = "mutated"
    after = command.frontres_local_scenario_intent_snapshot()
    assert not bool((after["intent_q29"] == -9999.0).any())
    assert after["provenance"][0]["intent_q29_source"] == "motion_internal_q29"

    raw_obs = torch.zeros(8, 5)
    baseline = runtime.append_frontres_future_intent_context(runner, raw_obs)
    command._frontres_local_scenario_current_root_artifact_t = torch.full_like(
        command._frontres_local_scenario_current_root_artifact_t, 123456.0
    )
    command._frontres_local_scenario_clean_continuation = torch.full_like(
        command._frontres_local_scenario_clean_continuation, -123456.0
    )
    command.future_root_global = torch.full((8, 3, 7), 8888.0)
    poisoned = runtime.append_frontres_future_intent_context(runner, raw_obs)
    torch.testing.assert_close(baseline[:, : layout.actor_tail_dim], poisoned[:, : layout.actor_tail_dim])
    print("[T-read-only/T-no-root/T-no-Clean/T-no-C] snapshot mutation and forbidden carriers cannot alter H tail", flush=True)


def test_t_row_permutation_identity_and_provenance() -> None:
    helper, commands, command, runtime, layout, _runner = _owners()
    sealed = command.frontres_local_scenario_intent_snapshot()
    permutation = torch.tensor([4, 0, 5, 1, 6, 2, 7, 3], dtype=torch.long)
    permuted = helper._make_command(commands, helper._FakeRobot(num_envs=8), num_envs=8)
    permuted.set_frontres_local_scenario(
        current_root_artifact_t=command._frontres_local_scenario_current_root_artifact_t.index_select(0, permutation),
        intent_q29=sealed["intent_q29"].index_select(0, permutation),
        clean_continuation=command._frontres_local_scenario_clean_continuation.index_select(0, permutation),
        expected_support=command._frontres_local_scenario_expected_support.index_select(0, permutation),
        horizon_k=command._frontres_local_scenario_horizon_k.index_select(0, permutation),
        continuation_lengths=command._frontres_local_scenario_continuation_lengths.index_select(0, permutation),
        scenario_ids=tuple(sealed["scenario_ids"][index] for index in permutation.tolist()),
        noisy_segment_hashes=tuple(sealed["noisy_segment_hashes"][index] for index in permutation.tolist()),
        x_t_identities=tuple(sealed["x_t_identities"][index] for index in permutation.tolist()),
        provenance=tuple(sealed["provenance"][index] for index in permutation.tolist()),
        roles=tuple(sealed["roles"][index] for index in permutation.tolist()),
        env_ids=torch.arange(8),
    )
    permuted.refresh_frontres_reference_cache_current_frame()
    env = helper._FakeEnv(permuted, permuted.robot, num_envs=8)
    runner = SimpleNamespace(
        env=env,
        alg=SimpleNamespace(policy=SimpleNamespace(num_actor_obs=63, num_frontres_obs=60)),
        _frontres_gmt_obs_dim=3,
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_layout_version=layout.version,
        _frontres_future_intent_actor_context_dim=layout.actor_tail_dim,
        _frontres_segment_live_current_batch=SimpleNamespace(
            frontres_local_scenario_intent_q29=torch.full((4, 3, 29), -7777.0)
        ),
    )
    output = runtime.append_frontres_future_intent_context(runner, torch.zeros(8, 5))
    original_tail = torch.cat([sealed["intent_q29"][:, 1], sealed["intent_q29"][:, 2]], dim=-1)
    torch.testing.assert_close(output[:, : layout.actor_tail_dim], original_tail.index_select(0, permutation))

    bad = dict(permuted._frontres_local_scenario_provenance[0])
    bad["intent_q29_provenance"] = "clean_q29"
    permuted._frontres_local_scenario_provenance[0] = bad
    _expect_error(
        RuntimeError,
        lambda: runtime.append_frontres_future_intent_context(runner, torch.zeros(8, 5)),
        "deployment_noisy_q29",
    )
    print("[T-permute/T-scenario-identity/T-provenance] row order follows command identities and Clean-labelled q29 rejects", flush=True)


def main() -> None:
    test_t_role_aligned_offset_and_policy_batch_isolation()
    test_t_snapshot_read_only_and_no_forbidden_fields()
    test_t_row_permutation_identity_and_provenance()
    print("frontres_v015_role_aligned_future_intent_contract: ok", flush=True)


if __name__ == "__main__":
    main()
