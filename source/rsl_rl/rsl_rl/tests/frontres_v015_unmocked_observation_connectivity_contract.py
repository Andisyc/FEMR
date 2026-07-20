#!/usr/bin/env python3
"""CPU-only S2 contract for the unmocked v015 observation-to-update route."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_SOURCE = ROOT / "source" / "rsl_rl"
TEST_ROOT = RSL_SOURCE / "rsl_rl" / "tests"
if str(RSL_SOURCE) not in sys.path:
    sys.path.insert(0, str(RSL_SOURCE))

# Capture the real visibility/normalizer functions before the contract helpers
# install their lightweight package stubs.
from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic


ACTOR_PARSE = FrontRESActorCritic._parse_observations
GMT_DIRECT = FrontRESActorCritic._run_gmt_direct
TRANSACTION_TEST = TEST_ROOT / "frontres_v015_transaction_route_contract.py"
CURRENT_COMMAND_TEST = TEST_ROOT / "frontres_v015_current_gmt_command_contract.py"
ACTOR_CONTEXT_TEST = TEST_ROOT / "frontres_future_intent_actor_context_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ACTOR_CONTEXT_HELPER = _load("frontres_v015_r5_actor_context_helper", ACTOR_CONTEXT_TEST)
_LAYOUT_MODULE, _RUNTIME_MODULE = _ACTOR_CONTEXT_HELPER._load_modules()
resolve_frontres_future_intent_layout = _LAYOUT_MODULE.resolve_frontres_future_intent_layout
apply_obs_normalizer = _RUNTIME_MODULE.apply_obs_normalizer
append_frontres_future_intent_context = _RUNTIME_MODULE.append_frontres_future_intent_context


class _TrackingGMTNormalizer:
    def __init__(self) -> None:
        self._mean = torch.zeros(1, 770)
        self._std = torch.ones(1, 770)
        self.calls: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(value.detach().clone())
        return value / 10.0


class _TrackingGMTPolicy:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def act_inference(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value.detach().clone())
        return value[:, :3]


class _SemanticPolicy(torch.nn.Module):
    num_actor_obs = 928
    num_frontres_obs = 158
    num_task_corrections = 6
    max_delta_pos = 1.0
    max_delta_rpy = 1.0
    gmt_policy_obs_dim = 770
    gmt_actor_input_dim = 770

    def __init__(self, command) -> None:
        super().__init__()
        self.log_prob_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.value_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.command = command
        self.gmt_normalizer = SimpleNamespace(_mean=torch.zeros(1, 770))
        self.gmt_policy = _TrackingGMTPolicy()
        self.ref_vel_estimator = None
        self.actor_inputs: list[torch.Tensor] = []
        self.env_action_inputs: list[torch.Tensor] = []

    def _parse_observations(self, observations: torch.Tensor):
        return ACTOR_PARSE(self, observations)

    def _run_gmt_direct(self, policy_obs, ref_vel, ref_vel_estimator_obs):
        return GMT_DIRECT(self, policy_obs, ref_vel, ref_vel_estimator_obs)

    def _pad_observations_for_gmt(self, observations: torch.Tensor) -> torch.Tensor:
        assert tuple(observations.shape[1:]) == (770,)
        return observations

    def actor_forward(self, observations: torch.Tensor) -> torch.Tensor:
        prefix, _ref_vel, _estimator = self._parse_observations(observations)
        self.actor_inputs.append(prefix.detach().clone())
        assert tuple(prefix.shape[1:]) == (158,)
        return torch.zeros(observations.shape[0], 6, dtype=observations.dtype)

    def get_actions_log_prob_per_dim_from_stats(self, actions, mean, sigma, dims):
        del mean, sigma, dims
        return torch.zeros_like(actions)

    def get_env_action(self, observations: torch.Tensor, delta_se: torch.Tensor) -> torch.Tensor:
        self.env_action_inputs.append(observations.detach().clone())
        return torch.cat([delta_se[:, :1], self.command.robot_joint_pos[:, :2]], dim=-1)

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        del actions
        return {
            "log_prob": self.log_prob_scale * observations[:, 0],
            "value": self.value_scale * observations[:, 1],
            "entropy": torch.zeros_like(observations[:, 0]),
        }


class _SemanticAlg:
    def __init__(self, policy: _SemanticPolicy, optimizer) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.transition = SimpleNamespace()
        self.act_calls = 0
        self.frontres_future_offsets = (1, 2)
        self.frontres_future_intent_layout_version = "frontres-v015-future-intent-q29-v1"
        self.frontres_v015_formal_transaction_enabled = True
        self.frontres_v015_local_sentinel_only = True
        self.frontres_training_objective = "segment_replay_hrl"
        self.frontres_segment_replay_enabled = True
        self.frontres_segment_advantage_normalization = "grouped_scale_only"
        self.frontres_hsl_init_enabled = False
        self.frontres_hsl_rollout_label_enabled = False
        self.frontres_segment_live_train_enabled = False
        self.frontres_segment_live_update_loop_only = False
        self.frontres_segment_live_single_update_only = False
        self.frontres_segment_critic_warmup_iterations = 0
        self.frontres_segment_actor_warmup_iterations = 0
        self.lambda_supervised = 0.0
        self.lambda_supervised_min = 0.0
        self.clip_param = 0.2
        self.value_loss_coef = 0.0
        self.entropy_coef = 0.0
        self.use_clipped_value_loss = True
        self.max_grad_norm = 1.0

    def act(self, obs, privileged_obs, **_kwargs):
        self.act_calls += 1
        actions = self.policy.actor_forward(obs)
        batch = int(obs.shape[0])
        actions[: batch // 2, 0] = torch.linspace(0.05, 0.20, batch // 2)
        self.transition.observations = obs.detach().clone()
        self.transition.privileged_observations = privileged_obs.detach().clone()
        self.transition.actions = actions.detach().clone()
        self.transition.actions_log_prob = torch.zeros(batch, dtype=obs.dtype)
        self.transition.values = torch.zeros(batch, dtype=obs.dtype)
        self.transition.action_mean = actions.detach().clone()
        self.transition.action_sigma = torch.full_like(actions, 0.25)
        return actions

    def _get_actor_log_prob(self, actions):
        return torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)


class _SemanticEnv:
    def __init__(self, base_env, command) -> None:
        self.unwrapped = self
        self.num_envs = 8
        self.device = torch.device("cpu")
        self.command_manager = base_env.command_manager
        self.command_manager._terms = {"motion": command}
        self.scene = base_env.scene
        self.episode_length_buf = base_env.episode_length_buf
        self.command = command
        self.step_count = 0
        self.raw_observations: list[torch.Tensor] = []
        self.current_commands: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

    def _raw_observation(self) -> torch.Tensor:
        current_command = self.command.command.detach().clone()
        artifact = self.command._frontres_local_scenario_current_root_artifact_t.detach().clone()
        prefix = torch.zeros(self.num_envs, 100)
        prefix[:, :7] = artifact
        prefix[:, 7:] = torch.arange(93, dtype=torch.float32).unsqueeze(0) / 100.0
        suffix = torch.zeros(self.num_envs, 770)
        suffix[:, :290] = current_command.repeat(1, 5)
        suffix[:, 290:319] = self.command.robot_joint_pos.detach().clone()
        suffix[:, 319:] = torch.arange(451, dtype=torch.float32).unsqueeze(0) / 10.0
        raw = torch.cat([prefix, suffix], dim=-1)
        self.current_commands.append(current_command)
        self.raw_observations.append(raw.detach().clone())
        return raw

    def get_observations(self):
        raw = self._raw_observation()
        return raw, {"observations": {}}

    def step(self, actions: torch.Tensor):
        self.actions.append(actions.detach().clone())
        self.step_count += 1
        if self.step_count == 1:
            intent_t = self.command.frontres_local_scenario_intent_snapshot()["intent_q29"][:, 0]
            self.scene.robot.data.joint_pos[:4] = intent_t[:4]
            self.scene.robot.data.joint_pos[4:] = intent_t[4:] + 0.5
        raw = self._raw_observation()
        return (
            raw,
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {"observations": {}},
        )


def _build_fixture():
    formal = _load("frontres_v015_r5_transaction_helper", TRANSACTION_TEST)
    candidate_contract, owners, live_sampler, _live_update_loop = formal._load_owners()
    _gain_contract, _one_action, helper, commands, hooks, setup, live_probe, _storage, ppo = owners
    current_helper = _load("frontres_v015_r5_current_command_helper", CURRENT_COMMAND_TEST)
    command, request = current_helper._sealed_role_command(helper, commands, hooks, setup)
    command.cfg = SimpleNamespace(motion_horizon=1, command_velocity=True)
    base_env = helper._FakeEnv(command, command.robot, num_envs=8)
    env = _SemanticEnv(base_env, command)
    command._env = env

    layout = resolve_frontres_future_intent_layout((1, 2), "frontres-v015-future-intent-q29-v1")
    policy = _SemanticPolicy(command)
    optimizer = formal._TrackingSGD(policy.parameters())
    alg = _SemanticAlg(policy, optimizer)
    prefix_mean = torch.zeros(1, 158)
    prefix_std = torch.full((1, 158), 2.0)
    gmt_normalizer = _TrackingGMTNormalizer()
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
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_layout_version=layout.version,
        _frontres_future_intent_actor_context_dim=58,
        _frontres_gmt_obs_dim=770,
        _frontres_extra_mean=prefix_mean,
        _frontres_extra_std=prefix_std,
        _frontres_extra_stats_layout_version=layout.version,
        _frontres_extra_normalizer=None,
        _frontres_segment_live_current_batch=SimpleNamespace(
            frontres_local_scenario_intent_q29=torch.full((4, 3, 29), -7777.0)
        ),
        obs_normalizer=gmt_normalizer,
        privileged_obs_normalizer=lambda value: value,
        teacher_obs_normalizer=lambda value: value,
    )
    runner._append_frontres_future_intent_context = lambda obs: append_frontres_future_intent_context(runner, obs)
    runner._apply_obs_normalizer = lambda obs: apply_obs_normalizer(runner, obs)

    def apply_corrections(actions, n_train, **_kwargs):
        command._frontres_pos_correction.zero_()
        command._frontres_pos_correction[:n_train] = actions[:n_train, :3]
        command._frontres_quat_correction.zero_()
        command._frontres_quat_correction[:, 0] = 1.0

    runner._apply_frontres_task_corrections = apply_corrections
    pair_layout = SimpleNamespace(n_train=4, n_base=4, n_candidate=0, n_clean=0)
    live_probe.FrontRESSegmentPPOBatch = ppo.FrontRESSegmentPPOBatch
    return SimpleNamespace(
        formal=formal,
        live_probe=live_probe,
        live_sampler=live_sampler,
        runner=runner,
        command=command,
        request=request,
        pair_layout=pair_layout,
        policy=policy,
        optimizer=optimizer,
        env=env,
    )


def test_t_unmocked_observation_to_exact_one_update() -> None:
    fixture = _build_fixture()
    live_probe = fixture.live_probe
    runner = fixture.runner
    snapshot = fixture.live_sampler.capture_frontres_frozen_policy_snapshot(
        runner, transaction_id="tx-v015-r5-s2"
    )

    observations = live_probe._read_live_observations(runner)
    assert tuple(observations.obs.shape) == (8, 928)
    raw = fixture.env.raw_observations[0]
    current = fixture.env.current_commands[0]
    assert tuple(raw.shape) == (8, 870)
    torch.testing.assert_close(raw[:, 100:390], current.repeat(1, 5))
    intent = fixture.command.frontres_local_scenario_intent_snapshot()["intent_q29"]
    expected_tail = torch.cat([intent[:, 1], intent[:, 2]], dim=-1) / 2.0
    torch.testing.assert_close(observations.obs[:, :58], expected_tail)
    assert runner.obs_normalizer.calls and tuple(runner.obs_normalizer.calls[0].shape) == (8, 770)
    assert runner._frontres_v015_observation_route_trace == {
        "role_row_count": 8,
        "current_command_dim": 0,
        "raw_observation_dim": 870,
        "q29_tail_dim": 58,
        "combined_observation_dim": 928,
        "normalized_observation_dim": 928,
        "femr_visible_dim": 158,
        "gmt_suffix_dim": 770,
        "gmt_input_dim": 0,
        "post_advance_gmt_read_count": 0,
    }

    evidence = live_probe.collect_frontres_v015_gain_return_priority_evidence(
        runner,
        observations,
        pair_layout=fixture.pair_layout,
    )
    assert fixture.policy.actor_inputs and tuple(fixture.policy.actor_inputs[0].shape) == (8, 158)
    assert evidence.one_action.actor_forward_count == 1
    assert evidence.one_action.later_femr_action_count == 0
    assert runner.alg.act_calls == 1
    assert runner._frontres_v015_observation_route_trace["current_command_dim"] == 58
    assert runner._frontres_v015_observation_route_trace["gmt_input_dim"] == 770

    continuation = fixture.command._frontres_local_scenario_clean_continuation.detach().clone()
    horizon = fixture.command._frontres_local_scenario_horizon_k.detach().clone()
    assert len(fixture.policy.gmt_policy.inputs) == int(horizon.max().item())
    assert runner._frontres_v015_observation_route_trace["post_advance_gmt_read_count"] == int(horizon.max().item())
    for offset, gmt_input in enumerate(fixture.policy.gmt_policy.inputs):
        valid = horizon > offset
        expected_command = continuation[:, offset, :58] / 10.0
        torch.testing.assert_close(gmt_input[valid, :58], expected_command[valid])

    metadata_kwargs = {
        "transaction_id": snapshot.transaction_id,
        "policy_snapshot_id": snapshot.policy_snapshot_id,
        "motion_ids": ("motion-a", "motion-a", "motion-b", "motion-b"),
        "start_frames": torch.tensor([12, 12, 24, 24], dtype=torch.long),
        "segment_ids": torch.tensor([101, 101, 202, 202], dtype=torch.long),
        "source_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "trial_index": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }
    candidate = live_probe.build_frontres_v015_grouped_candidate_batch(evidence, **metadata_kwargs)
    metadata = candidate.transaction_metadata
    plan = fixture.live_sampler.FrontRESV015FormalTransactionPlan(
        snapshot=snapshot,
        motion_ids=metadata.motion_ids,
        start_frames=metadata.start_frames,
        segment_ids=metadata.segment_ids,
        source_index=metadata.source_index,
        trial_index=metadata.trial_index,
        horizon_k=metadata.horizon_k,
        scenario_ids=metadata.scenario_ids,
        noisy_segment_hashes=metadata.noisy_segment_hashes,
        x_t_identities=metadata.x_t_identities,
        intent_q29_provenance=metadata.intent_q29_provenance,
        intent_q29_source=metadata.intent_q29_source,
    )
    request = live_probe.FrontRESV015FormalTransactionRequest(
        plan=plan,
        candidate_batches=(candidate,),
        policy_evaluator=fixture.policy,
    )
    result = live_probe.run_frontres_v015_formal_transaction_update(runner, request)
    assert fixture.optimizer.step_count == 1
    assert result.optimizer_step_delta == 1
    assert result.update_invocation_count == 1
    assert result.policy_attempt_count == 4
    assert result.ppo_result.grouped_attempt_mass_shares == (0.25, 0.25, 0.25, 0.25)
    print(
        "[T-command-connect/T-history-layout/T-role-tail/T-normalizer/T-consumer/"
        "T-one-action/T-clean-C-order/T-exact-one-update] "
        "58/290/870 + 58 -> 928 -> 158/770; K uses current C; attempts=4; step_delta=1",
        flush=True,
    )


if __name__ == "__main__":
    test_t_unmocked_observation_to_exact_one_update()
    print("frontres_v015_unmocked_observation_connectivity_contract: ok", flush=True)
