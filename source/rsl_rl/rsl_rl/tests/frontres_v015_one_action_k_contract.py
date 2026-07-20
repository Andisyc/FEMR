#!/usr/bin/env python3
"""Deterministic S1/S2 contract for the v015 one-action frozen-GMT K collector."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
HELPER_PATH = RSL_ROOT / "tests" / "frontres_v015_two_role_reset_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str) -> types.ModuleType:
    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load_owners():
    helper = _load("frontres_v015_one_action_k_reset_helper", HELPER_PATH)
    commands, hooks, setup = helper._load_owners()

    rsl_rl_pkg = sys.modules["rsl_rl"]
    frontres_pkg = sys.modules["rsl_rl.frontres"]
    runners_pkg = _package("rsl_rl.runners")
    algorithms_pkg = _package("rsl_rl.algorithms")
    modules_pkg = _package("rsl_rl.modules")
    rsl_rl_pkg.runners = runners_pkg
    rsl_rl_pkg.algorithms = algorithms_pkg
    rsl_rl_pkg.modules = modules_pkg
    algorithms_pkg.FrontRESUnified = object
    modules_pkg.FrontRESActorCritic = object

    ppo_module = types.ModuleType("rsl_rl.algorithms.frontres_segment_ppo")
    ppo_module.FrontRESSegmentPPOBatch = object
    ppo_module.FrontRESSegmentPPOConfig = object
    ppo_module.compute_frontres_segment_ppo_loss = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("Step 2B must not enter PPO")
    )
    sys.modules[ppo_module.__name__] = ppo_module
    algorithms_pkg.frontres_segment_ppo = ppo_module

    schedule = sys.modules["rsl_rl.frontres.training_schedule"]
    schedule.resolve_frontres_mode_state = lambda *_args, **_kwargs: SimpleNamespace(
        is_frontres=True,
        is_task_space_mode=True,
    )
    warmup = types.ModuleType("rsl_rl.frontres.frontres_segment_warmup")
    warmup.frontres_segment_warmup_phase = lambda *_args, **_kwargs: "disabled"
    sys.modules[warmup.__name__] = warmup
    frontres_pkg.frontres_segment_warmup = warmup

    training_setup = types.ModuleType("rsl_rl.runners.frontres_training_setup")
    training_setup.configure_frontres_pair_layout = lambda runner, **_kwargs: getattr(
        runner, "_frontres_test_pair_layout", None
    )
    sys.modules[training_setup.__name__] = training_setup
    runners_pkg.frontres_training_setup = training_setup

    storage = _load(
        "rsl_rl.frontres.frontres_segment_storage",
        RSL_ROOT / "frontres" / "frontres_segment_storage.py",
    )
    frontres_pkg.frontres_segment_storage = storage
    reset = _load(
        "rsl_rl.frontres.frontres_segment_reset",
        RSL_ROOT / "frontres" / "frontres_segment_reset.py",
    )
    frontres_pkg.frontres_segment_reset = reset
    rollout = _load(
        "rsl_rl.runners.frontres_rollout_step",
        RSL_ROOT / "runners" / "frontres_rollout_step.py",
    )
    runners_pkg.frontres_rollout_step = rollout
    live_probe = _load(
        "rsl_rl.runners.frontres_segment_live_probe",
        RSL_ROOT / "runners" / "frontres_segment_live_probe.py",
    )
    runners_pkg.frontres_segment_live_probe = live_probe
    return helper, commands, hooks, setup, live_probe


class _FakePolicy:
    num_task_corrections = 6
    max_delta_pos = 1.0
    max_delta_rpy = 1.0

    def __init__(self, command) -> None:
        self.command = command
        self.env_action_calls: list[dict[str, torch.Tensor]] = []
        self.gmt_direct_calls: list[dict[str, torch.Tensor]] = []

    def get_actions_log_prob_per_dim_from_stats(self, actions, mean, sigma, dims):
        del mean, sigma, dims
        return torch.zeros_like(actions)

    def get_env_action(self, observations: torch.Tensor, delta_se: torch.Tensor) -> torch.Tensor:
        self.env_action_calls.append(
            {
                "observations": observations.detach().clone(),
                "delta_se": delta_se.detach().clone(),
                "joint_pos": self.command.joint_pos.detach().clone(),
                "joint_vel": self.command.joint_vel.detach().clone(),
                "anchor_pos": self.command.anchor_pos_w.detach().clone(),
                "correction": self.command._frontres_pos_correction.detach().clone(),
            }
        )
        return torch.cat([delta_se[:, :1], self.command.joint_pos[:, :2]], dim=-1)

    def _parse_observations(self, observations: torch.Tensor):
        return observations, None, None

    def _run_gmt_direct(self, _policy_obs, _ref_vel, _ref_vel_estimator_obs) -> torch.Tensor:
        self.gmt_direct_calls.append(
            {
                "joint_pos": self.command.joint_pos.detach().clone(),
                "joint_vel": self.command.joint_vel.detach().clone(),
                "anchor_pos": self.command.anchor_pos_w.detach().clone(),
                "correction": self.command._frontres_pos_correction.detach().clone(),
            }
        )
        return self.command.joint_pos[:, :3].detach().clone()


class _FakeAlg:
    def __init__(self, policy: _FakePolicy) -> None:
        self.policy = policy
        self.transition = SimpleNamespace()
        self.frontres_future_offsets = (1, 2)
        self.lambda_supervised = 0.0
        self.act_calls = 0

    def act(self, obs, privileged_obs, **_kwargs):
        self.act_calls += 1
        batch = int(obs.shape[0])
        actions = torch.arange(batch * 6, dtype=torch.float32).reshape(batch, 6) / 10.0
        self.transition.observations = obs.detach().clone()
        self.transition.privileged_observations = privileged_obs.detach().clone()
        self.transition.actions = actions.detach().clone()
        self.transition.actions_log_prob = torch.zeros(batch, dtype=torch.float32)
        self.transition.values = torch.arange(batch, dtype=torch.float32)
        self.transition.action_mean = actions.detach().clone()
        self.transition.action_sigma = torch.full_like(actions, 0.25)
        return actions

    def _get_actor_log_prob(self, actions):
        return torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)


def _configure_fake_env(helper, commands, hooks, setup, *, horizons: tuple[int, int]):
    robot = helper._FakeRobot(num_envs=4)
    command = helper._make_command(commands, robot, num_envs=4)
    env = helper._FakeEnv(command, robot, num_envs=4)
    env.command_manager._terms = {"motion": command}
    env.device = torch.device("cpu")
    command._env = env
    command.cfg = SimpleNamespace(motion_horizon=2, command_velocity=True)
    env._obs = torch.arange(4 * 11, dtype=torch.float32).reshape(4, 11)
    env.env_action_history: list[torch.Tensor] = []

    def get_observations():
        return env._obs.detach().clone(), {"observations": {}}

    def step(actions: torch.Tensor):
        env.env_action_history.append(actions.detach().clone())
        env._obs = env._obs + 1.0
        return (
            env._obs.detach().clone(),
            torch.zeros(4, dtype=torch.float32),
            torch.zeros(4, dtype=torch.bool),
            {"observations": {}},
        )

    env.get_observations = get_observations
    env.step = step
    reset_runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        cfg={"frontres_candidate_rollout_enabled": True},
        alg=SimpleNamespace(frontres_future_offsets=(1, 2)),
        _frontres_future_intent_layout=SimpleNamespace(version="frontres-v015-future-intent-q29-v1"),
    )
    pair_layout = setup.configure_frontres_pair_layout(reset_runner, is_frontres=True)
    request = helper._local_request(reset_runner._frontres_v015_two_role_env_ids)
    request.horizon_k = torch.tensor(horizons, dtype=torch.long)
    request.frontres_local_scenario_clean_continuation_lengths = request.horizon_k.clone()
    request.frontres_local_scenario_clean_continuation_mask = (
        torch.arange(request.frontres_local_scenario_clean_continuation.shape[1]).unsqueeze(0)
        < request.horizon_k.unsqueeze(1)
    )
    for source in range(2):
        for offset in range(request.frontres_local_scenario_clean_continuation.shape[1]):
            row = request.frontres_local_scenario_clean_continuation[source, offset]
            value = 1000.0 * (source + 1) + 100.0 * offset
            row[:29].fill_(value)
            row[29:58].fill_(value + 29.0)
            row[58:61].fill_(value + 58.0)
            row[61:65].fill_(value + 61.0)
    adapter = hooks.FrontRESStage1EnvAdapter(env=env, amass_root="/tmp", trace=False)
    reset_result = adapter.apply_frontres_segment_index_reset(request)
    assert reset_result["reset_success"].tolist() == [True, True]
    return env, command, pair_layout, request


def _capture(live_probe, helper, commands, hooks, setup, *, horizons: tuple[int, int]):
    env, command, pair_layout, request = _configure_fake_env(helper, commands, hooks, setup, horizons=horizons)
    policy = _FakePolicy(command)
    alg = _FakeAlg(policy)
    apply_calls: list[torch.Tensor] = []

    def apply_frontres_task_corrections(actions, n_train, **_kwargs):
        apply_calls.append(actions.detach().clone())
        command._frontres_pos_correction.zero_()
        command._frontres_pos_correction[:n_train] = actions[:n_train, :3]
        command._frontres_quat_correction.zero_()
        command._frontres_quat_correction[:, 0] = 1.0

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
    evidence = live_probe.collect_frontres_v015_one_action_k_evidence(
        runner,
        observations,
        pair_layout=pair_layout,
    )
    return SimpleNamespace(
        evidence=evidence,
        env=env,
        command=command,
        policy=policy,
        alg=alg,
        apply_calls=apply_calls,
        request=request,
        runner=runner,
    )


def test_t_action_count_and_frozen(live_probe, helper, commands, hooks, setup) -> None:
    result = _capture(live_probe, helper, commands, hooks, setup, horizons=(3, 2))
    assert result.alg.act_calls == 1
    assert len(result.apply_calls) == 1
    assert len(result.env.env_action_history) == 1 + 3
    assert len(result.policy.env_action_calls) == 1
    assert len(result.policy.gmt_direct_calls) == 3
    assert torch.count_nonzero(result.policy.env_action_calls[0]["correction"][:2]) > 0
    for call in result.policy.gmt_direct_calls:
        assert torch.count_nonzero(call["correction"]) == 0
    assert not hasattr(result.runner, "_frontres_v015_one_action_k_phase")
    assert not bool(result.command._frontres_local_scenario_k_execution_active.any())
    assert not bool(result.command._frontres_local_scenario_current_frame_ready.any())
    sealed = result.command.frontres_local_scenario_snapshot(torch.arange(4))
    replay = result.command.set_frontres_local_scenario(
        current_root_artifact_t=sealed["current_root_artifact_t"],
        intent_q29=sealed["intent_q29"],
        clean_continuation=sealed["clean_continuation"],
        horizon_k=sealed["horizon_k"],
        continuation_lengths=sealed["continuation_lengths"],
        scenario_ids=sealed["scenario_ids"],
        noisy_segment_hashes=sealed["noisy_segment_hashes"],
        x_t_identities=sealed["x_t_identities"],
        provenance=sealed["provenance"],
        roles=sealed["roles"],
        env_ids=torch.arange(4),
    )
    assert bool(replay.all())
    result.command.refresh_frontres_reference_cache_current_frame()
    torch.testing.assert_close(result.command._cached_perturbed_pos, sealed["current_root_artifact_t"][:, :3])
    assert result.command.frontres_local_scenario_snapshot(torch.arange(4))["noisy_segment_hashes"] == sealed[
        "noisy_segment_hashes"
    ]
    print(
        "[T-action-count/T-frozen] actor=1 correction-write=1 later-actor=0 later-repair-write=0 M-rearm=immutable",
        flush=True,
    )


def test_t_continuation_and_row(live_probe, helper, commands, hooks, setup) -> None:
    result = _capture(live_probe, helper, commands, hooks, setup, horizons=(3, 2))
    evidence = result.evidence
    evidence.validate()
    assert evidence.actor_forward_count == 1
    assert evidence.later_femr_action_count == 0
    assert tuple(evidence.policy_actions.shape) == (2, 6)
    assert evidence.policy_row_indices.tolist() == [0, 1]
    assert evidence.roles == ("repair", "repair", "noisy", "noisy")
    assert evidence.scenario_ids == ("scenario-a", "scenario-b", "scenario-a", "scenario-b")
    assert evidence.noisy_segment_hashes == ("hash-a", "hash-b", "hash-a", "hash-b")
    expected_c = torch.tensor(
        [
            [1000.0, 2000.0, 1000.0, 2000.0],
            [1100.0, 2100.0, 1100.0, 2100.0],
            [1200.0, 2100.0, 1200.0, 2100.0],
        ]
    )
    torch.testing.assert_close(evidence.continuation[:, :, 0], expected_c)
    torch.testing.assert_close(
        evidence.continuation_valid_mask,
        torch.tensor([[True, True, True, True], [True, True, True, True], [True, False, True, False]]),
    )
    # The action at t reads deployment q29 intent, while every later frozen-GMT
    # call reads the command-owned Clean continuation C.
    first_q = result.policy.env_action_calls[0]["joint_pos"][:, 0]
    assert first_q.tolist() == [0.0, 1000.0, 0.0, 1000.0]
    for offset, call in enumerate(result.policy.gmt_direct_calls):
        torch.testing.assert_close(call["joint_pos"][:, 0], expected_c[offset])
        torch.testing.assert_close(call["joint_vel"][:, 0], expected_c[offset] + 29.0)
        torch.testing.assert_close(call["anchor_pos"][:, 0], expected_c[offset] + 58.0)
    print("[T-continuation/T-row] one-policy-row-per-Repair; GMT=C[q29,dq29,root] only", flush=True)


def test_t_k_metamorphic_and_legacy_reject(live_probe, helper, commands, hooks, setup) -> None:
    long_result = _capture(live_probe, helper, commands, hooks, setup, horizons=(3, 2))
    short_result = _capture(live_probe, helper, commands, hooks, setup, horizons=(1, 1))
    assert int(long_result.evidence.policy_actions.shape[0]) == int(short_result.evidence.policy_actions.shape[0]) == 2
    assert int(long_result.evidence.continuation.shape[0]) == 3
    assert int(short_result.evidence.continuation.shape[0]) == 1
    assert long_result.alg.act_calls == short_result.alg.act_calls == 1
    try:
        live_probe._require_v015_one_action_k_layout(
            long_result.runner,
            live_probe.FrontRESSegmentLiveObservations(
                obs=torch.zeros(4, 11),
                privileged_obs=torch.zeros(4, 5),
                teacher_obs=torch.zeros(4, 5),
                ref_vel_estimator_obs=None,
            ),
            SimpleNamespace(n_train=1, n_candidate=1, n_base=1, n_clean=1),
        )
    except RuntimeError as exc:
        assert "Repair/Noisy" in str(exc)
    else:
        raise AssertionError("legacy quartet layout unexpectedly entered v015 one-action collector")
    try:
        live_probe._run_live_rollout_capture(
            long_result.runner,
            live_probe.FrontRESSegmentLiveObservations(
                obs=torch.zeros(4, 11),
                privileged_obs=torch.zeros(4, 5),
                teacher_obs=torch.zeros(4, 5),
                ref_vel_estimator_obs=None,
            ),
            rollout_steps=1,
            pair_layout=SimpleNamespace(n_train=2, n_candidate=0, n_base=2, n_clean=0),
        )
    except RuntimeError as exc:
        assert "legacy repeated-actor" in str(exc)
    else:
        raise AssertionError("legacy repeated-actor collector accepted an active v015 local scenario")
    print(
        "[T-K-metamorphic/T-legacy-reject] K changes evidence length, never policy-row count; quartet and legacy loop rejected",
        flush=True,
    )


def main() -> None:
    helper, commands, hooks, setup, live_probe = _load_owners()
    test_t_action_count_and_frozen(live_probe, helper, commands, hooks, setup)
    test_t_continuation_and_row(live_probe, helper, commands, hooks, setup)
    test_t_k_metamorphic_and_legacy_reject(live_probe, helper, commands, hooks, setup)
    print("frontres_v015_one_action_k_contract: ok", flush=True)


if __name__ == "__main__":
    main()
