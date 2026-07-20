#!/usr/bin/env python3
"""Deterministic S1 contract for the v015 pre-action GMT command."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
HELPER_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "tests" / "frontres_v015_two_role_reset_contract.py"


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


def _sealed_role_command(helper, commands, hooks, setup):
    robot = helper._FakeRobot(num_envs=8)
    command = helper._make_command(commands, robot, num_envs=8)
    env = helper._FakeEnv(command, robot, num_envs=8)
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        cfg={"frontres_candidate_rollout_enabled": True},
        alg=SimpleNamespace(frontres_future_offsets=(1, 2)),
        _frontres_future_intent_layout=SimpleNamespace(version="frontres-v015-future-intent-q29-v1"),
    )
    setup.configure_frontres_pair_layout(runner, is_frontres=True)
    request = helper._parallel_attempt_request(runner._frontres_v015_two_role_env_ids)
    source_frames = request.start_frames.to(torch.float32).unsqueeze(1)
    request.frontres_local_scenario_intent_q29[:, 0] = source_frames + torch.arange(
        29, dtype=torch.float32
    ).unsqueeze(0)
    adapter = hooks.FrontRESStage1EnvAdapter(env=env, amass_root="/tmp", trace=False)
    result = adapter.apply_frontres_segment_index_reset(request)
    assert result["reset_success"].tolist() == [True, True, True, True]
    return command, request


def test_t_current_command_shape_provenance_role_identity(helper, commands, hooks, setup) -> None:
    command, request = _sealed_role_command(helper, commands, hooks, setup)
    command.cfg = SimpleNamespace(motion_horizon=1, command_velocity=True)

    current = command.command
    assert tuple(current.shape) == (8, 58)
    frames = torch.tensor([2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0]).unsqueeze(1)
    expected_q = frames + torch.arange(29, dtype=torch.float32).unsqueeze(0)
    expected_dq = 100.0 + expected_q
    expected = torch.cat([expected_q, expected_dq], dim=-1)
    torch.testing.assert_close(current, expected)

    # Repair/Noisy rows share the selected deployment motion/frame identity.
    torch.testing.assert_close(current[:4], current[4:])
    assert not torch.equal(current[:, :29], request.frontres_local_scenario_intent_q29.repeat(2, 1, 1)[:, 1])
    assert not bool(torch.isin(current, torch.tensor([300.0, 400.0])).any())
    print(
        "[T-current-command/T-shape/T-provenance/T-role-identity] "
        "deployment_q29_dq29=true shape=(8,58) repair_noisy_equal=true",
        flush=True,
    )


def test_t_current_only_and_command_velocity(helper, commands, hooks, setup) -> None:
    command, _request = _sealed_role_command(helper, commands, hooks, setup)
    command.cfg = SimpleNamespace(motion_horizon=2, command_velocity=True)
    _expect_error(RuntimeError, lambda: command.command, "motion_horizon=1")
    command.cfg = SimpleNamespace(motion_horizon=1, command_velocity=False)
    _expect_error(RuntimeError, lambda: command.command, "command_velocity=True")
    print("[T-current-only] local pre-action command rejects future horizon and q-only layout", flush=True)


def test_t_continuation_and_mixed_route_isolation(helper, commands, hooks, setup) -> None:
    command, request = _sealed_role_command(helper, commands, hooks, setup)
    command.cfg = SimpleNamespace(motion_horizon=1, command_velocity=True)
    command.begin_frontres_local_scenario_k_execution()
    step = command.advance_frontres_local_scenario_k_execution()
    during_k = command.command
    expected_c = request.frontres_local_scenario_clean_continuation.repeat(2, 1, 1)[:, 0, :58]
    torch.testing.assert_close(during_k, expected_c)
    torch.testing.assert_close(step["continuation"][:, :58], expected_c)
    command.end_frontres_local_scenario_k_execution()

    mixed = helper._make_command(commands, helper._FakeRobot(num_envs=2), num_envs=2)
    mixed.cfg = SimpleNamespace(motion_horizon=1, command_velocity=True)
    mixed._frontres_local_scenario_active[0] = True
    _expect_error(RuntimeError, lambda: mixed.command, "cannot mix")
    print("[T-continuation-isolation/T-legacy-reject] Clean C remains K-only and mixed rows reject", flush=True)


def test_t_local_and_legacy_command_clock_ownership(helper, commands, hooks, setup) -> None:
    update_source = inspect.getsource(commands.MultiMotionCommand._update_command)
    assert update_source.count("self._advance_frontres_command_clock()") == 1
    assert "self.time_steps += 1" not in update_source

    command, _request = _sealed_role_command(helper, commands, hooks, setup)
    command.cfg = SimpleNamespace(motion_horizon=1, command_velocity=True)
    command._global_sim_step = 0
    time_t = command.time_steps.detach().clone()
    artifact_t = command._cached_perturbed_pos.detach().clone()
    current_t = command.command.detach().clone()

    # A direct second install is still invalid, but IsaacLab's regular command
    # compute must hold the explicit local reference instead of calling it.
    _expect_error(RuntimeError, command.refresh_frontres_reference_cache_current_frame, "Step 2B")
    branch = command._advance_frontres_command_clock()
    assert branch == "local_current_hold"
    assert command._global_sim_step == 1
    torch.testing.assert_close(command.time_steps, time_t)
    torch.testing.assert_close(command._cached_perturbed_pos, artifact_t)
    torch.testing.assert_close(command.command, current_t)
    assert bool((command._frontres_local_scenario_k_execution_cursor == -1).all())

    command.begin_frontres_local_scenario_k_execution()
    first_c = command.advance_frontres_local_scenario_k_execution()
    cursor = command._frontres_local_scenario_k_execution_cursor.detach().clone()
    cache_c = command._cached_perturbed_pos.detach().clone()
    branch = command._advance_frontres_command_clock()
    assert branch == "local_k_hold"
    assert command._global_sim_step == 2
    torch.testing.assert_close(command.time_steps, time_t)
    torch.testing.assert_close(command._frontres_local_scenario_k_execution_cursor, cursor)
    torch.testing.assert_close(command._cached_perturbed_pos, cache_c)
    torch.testing.assert_close(command.command, first_c["continuation"][:, :58])
    _expect_error(RuntimeError, command.refresh_frontres_reference_cache_current_frame, "Step 2B")
    command.end_frontres_local_scenario_k_execution()

    legacy_calls: list[str] = []
    legacy = SimpleNamespace(
        _global_sim_step=9,
        time_steps=torch.tensor([3, 5], dtype=torch.long),
        _frontres_local_scenario_active=torch.zeros(2, dtype=torch.bool),
        _frontres_local_scenario_current_frame_ready=torch.zeros(2, dtype=torch.bool),
        _frontres_local_scenario_k_execution_active=torch.zeros(2, dtype=torch.bool),
        _advance_frontres_reference_window=lambda: legacy_calls.append("window"),
        _advance_frontres_fixed_noisy_tape=lambda: legacy_calls.append("tape"),
        refresh_frontres_reference_cache_current_frame=lambda: legacy_calls.append("refresh"),
    )
    branch = commands.MultiMotionCommand._advance_frontres_command_clock(legacy)
    assert branch == "legacy_advance"
    assert legacy._global_sim_step == 10
    assert legacy.time_steps.tolist() == [4, 6]
    assert legacy_calls == ["window", "tape", "refresh"]
    print(
        "[T-t-clock-hold/T-K-clock-hold/T-legacy-clock/T-duplicate-refresh-reject] "
        "local reference clock is explicit while legacy clock still advances",
        flush=True,
    )


def main() -> None:
    helper = _load("frontres_v015_current_gmt_command_helper", HELPER_PATH)
    commands, hooks, setup = helper._load_owners()
    test_t_current_command_shape_provenance_role_identity(helper, commands, hooks, setup)
    test_t_current_only_and_command_velocity(helper, commands, hooks, setup)
    test_t_continuation_and_mixed_route_isolation(helper, commands, hooks, setup)
    test_t_local_and_legacy_command_clock_ownership(helper, commands, hooks, setup)
    print("frontres_v015_current_gmt_command_contract: ok", flush=True)


if __name__ == "__main__":
    main()
