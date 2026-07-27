#!/usr/bin/env python3
"""Deterministic candidate-only S1 contract for v006 Gain consumers."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
ONE_ACTION_TEST = RSL_ROOT / "tests" / "frontres_v015_one_action_k_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_owners():
    one_action = _load("frontres_v015_gain_consumer_one_action_helper", ONE_ACTION_TEST)
    helper, commands, hooks, setup, live_probe = one_action._load_owners()
    frontres_pkg = sys.modules["rsl_rl.frontres"]
    gain = _load(
        "rsl_rl.frontres.frontres_gain",
        RSL_ROOT / "frontres" / "frontres_gain.py",
    )
    sampler = _load(
        "rsl_rl.frontres.frontres_segment_sampler",
        RSL_ROOT / "frontres" / "frontres_segment_sampler.py",
    )
    frontres_pkg.frontres_gain = gain
    frontres_pkg.frontres_segment_sampler = sampler
    return one_action, helper, commands, hooks, setup, live_probe, gain, sampler


def _capture_consumer(one_action, helper, commands, hooks, setup, live_probe, *, t_done: bool = False):
    env, command, pair_layout, request = one_action._configure_fake_env(
        helper,
        commands,
        hooks,
        setup,
        horizons=(3, 2),
    )
    policy = one_action._FakePolicy(command)
    alg = one_action._FakeAlg(policy)
    apply_calls: list[torch.Tensor] = []

    def apply_frontres_task_corrections(actions, n_train, **_kwargs):
        del _kwargs
        apply_calls.append(actions.detach().clone())
        command._frontres_pos_correction.zero_()
        command._frontres_pos_correction[:n_train] = actions[:n_train, :3]
        command._frontres_quat_correction.zero_()
        command._frontres_quat_correction[:, 0] = 1.0

    def step(actions: torch.Tensor):
        env.env_action_history.append(actions.detach().clone())
        if len(env.env_action_history) == 1:
            target = command.joint_pos.detach().clone()
            command.robot.data.joint_pos.copy_(target)
            command.robot.data.joint_pos[:2, 0] += 0.1
            command.robot.data.joint_pos[2:, 0] += 0.4
        env._obs = env._obs + 1.0
        dones = torch.zeros(4, dtype=torch.bool)
        if t_done and len(env.env_action_history) == 1:
            dones[0] = True
        return (
            env._obs.detach().clone(),
            torch.zeros(4, dtype=torch.float32),
            dones,
            {"observations": {}},
        )

    env.step = step
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        alg=alg,
        training_type="frontres",
        cfg={},
        current_learning_iteration=0,
        policy_obs_type=None,
        privileged_obs_type=None,
        teacher_obs_type=None,
        ref_vel_estimator_obs_type=None,
        _frontres_future_intent_layout=SimpleNamespace(version="frontres-v015-future-intent-q29-v1"),
        _append_frontres_future_intent_context=lambda obs: obs,
        _apply_obs_normalizer=lambda obs: obs,
        privileged_obs_normalizer=lambda obs: obs,
        teacher_obs_normalizer=lambda obs: obs,
        _apply_frontres_task_corrections=apply_frontres_task_corrections,
    )
    observations = live_probe.FrontRESSegmentLiveObservations(
        obs=env._obs.detach().clone(),
        privileged_obs=env._obs[:, :5].detach().clone(),
        teacher_obs=env._obs[:, :5].detach().clone(),
        ref_vel_estimator_obs=None,
    )
    physics_offset = 0
    original_physics = live_probe._capture_physics_frame

    def capture_physics(_runner, _layout):
        nonlocal physics_offset
        frame = one_action._fake_physics_frame(physics_offset, mode="unequal")
        physics_offset += 1
        return frame

    live_probe._capture_physics_frame = capture_physics
    try:
        result = live_probe.collect_frontres_v015_gain_return_priority_evidence(
            runner,
            observations,
            pair_layout=pair_layout,
        )
    finally:
        live_probe._capture_physics_frame = original_physics
    return SimpleNamespace(
        result=result,
        runner=runner,
        command=command,
        policy=policy,
        alg=alg,
        apply_calls=apply_calls,
        request=request,
    )


def test_t_provenance_and_consumer_value(one_action, helper, commands, hooks, setup, live_probe, gain, sampler) -> None:
    legacy_calls: list[str] = []
    original_segment_gain = gain.compute_segment_gain
    original_capture_gain = live_probe._capture_paired_gain

    def legacy_forbidden(*_args, **_kwargs):
        legacy_calls.append("legacy")
        raise AssertionError("v015 candidate consumer must not call Clean-global v002 Gain")

    gain.compute_segment_gain = legacy_forbidden
    live_probe._capture_paired_gain = legacy_forbidden
    try:
        captured = _capture_consumer(one_action, helper, commands, hooks, setup, live_probe)
    finally:
        gain.compute_segment_gain = original_segment_gain
        live_probe._capture_paired_gain = original_capture_gain

    result = captured.result
    result.validate()
    assert not legacy_calls
    one = result.one_action
    returned = result.return_evidence
    priority = result.priority_evidence
    assert one.intent_q29_provenance == ("deployment_noisy_q29",) * 4
    assert one.intent_q29_source == ("motion_internal_q29",) * 4
    torch.testing.assert_close(one.executed_q29_t[:2, 0], one.intent_q29[:2, 0, 0] + 0.1)
    torch.testing.assert_close(one.executed_q29_t[2:, 0], one.intent_q29[2:, 0, 0] + 0.4)
    assert bool((returned.intent_gain > 0.0).all())
    assert bool((returned.physics_zmp_gain >= 0.0).all())
    assert bool((returned.physics_contact_gain >= 0.0).all())
    assert bool((returned.physics_valid_step_count > 0).all())
    torch.testing.assert_close(returned.return_k, returned.gain_total)
    torch.testing.assert_close(returned.advantage_k, returned.return_k - returned.policy_values)
    expected_evidence_steps = one.survival_steps.index_select(0, one.policy_row_indices).to(dtype=torch.long)
    torch.testing.assert_close(returned.evidence_valid_step_count, expected_evidence_steps)
    torch.testing.assert_close(priority.priority_signal, returned.gain_total)
    assert priority.scenario_ids == returned.scenario_ids
    assert priority.noisy_segment_hashes == returned.noisy_segment_hashes
    assert priority.intent_q29_provenance == "deployment_noisy_q29"
    assert not hasattr(priority, "advantages")
    print(
        "[T-provenance/T-consumer/T-no-v002-fallback] I[t] and post-t robot q29 reach v006 only; return=scalar Intent target",
        flush=True,
    )


def test_t_invalid_q29_fails_closed_and_priority_isolated(
    one_action,
    helper,
    commands,
    hooks,
    setup,
    live_probe,
    gain,
    sampler,
) -> None:
    captured = _capture_consumer(one_action, helper, commands, hooks, setup, live_probe, t_done=True)
    result = captured.result
    returned = result.return_evidence
    priority = result.priority_evidence
    assert not bool(returned.policy_row_valid[0])
    assert torch.isnan(returned.gain_total[0])
    assert torch.isnan(returned.intent_gain[0])
    assert torch.isnan(returned.physics_gain[0])
    assert torch.isnan(returned.repair_cost[0])
    assert torch.isnan(returned.return_k[0])
    assert torch.isnan(returned.advantage_k[0])
    assert not bool(priority.valid_mask[0])
    assert torch.isnan(priority.physics_gain[0])

    priority_before = returned.gain_total.detach().clone()
    priority.priority_signal.add_(17.0)
    torch.testing.assert_close(
        torch.nan_to_num(returned.gain_total, nan=0.0),
        torch.nan_to_num(priority_before, nan=0.0),
    )
    assert torch.equal(torch.isnan(returned.gain_total), torch.isnan(priority_before))

    bad = replace(
        result.one_action,
        intent_q29_provenance=("clean_q29",) * len(result.one_action.roles),
    )
    try:
        result.one_action.__class__.validate(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("v015 consumer accepted Clean q29 provenance")
    bad_return = replace(returned, intent_q29_source="clean_q29")
    try:
        bad_return.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("v015 return carrier accepted Clean q29 source")
    print(
        "[T-priority-isolation/T-invalid] invalid q29 rows stay NaN; priority evidence cannot mutate return/loss carrier",
        flush=True,
    )


def main() -> None:
    one_action, helper, commands, hooks, setup, live_probe, gain, sampler = _load_owners()
    test_t_provenance_and_consumer_value(one_action, helper, commands, hooks, setup, live_probe, gain, sampler)
    test_t_invalid_q29_fails_closed_and_priority_isolated(
        one_action,
        helper,
        commands,
        hooks,
        setup,
        live_probe,
        gain,
        sampler,
    )
    print("frontres_v015_gain_consumer_contract: ok", flush=True)


if __name__ == "__main__":
    main()
