#!/usr/bin/env python3
"""CPU-only S2 contract for the unmocked v015 observation-to-policy route."""

from __future__ import annotations

import importlib.util
import sys
import types
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
CURRENT_COMMAND_TEST = TEST_ROOT / "frontres_v015_current_gmt_command_contract.py"
TWO_ROLE_TEST = TEST_ROOT / "frontres_v015_two_role_reset_contract.py"
ACTOR_CONTEXT_TEST = TEST_ROOT / "frontres_future_intent_actor_context_contract.py"
ONE_ACTION_OWNER = RSL_SOURCE / "rsl_rl" / "runners" / "frontres_segment_one_action_k.py"


def _expect_error(error_type, fn) -> None:
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


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
    gmt_policy_obs_dim = 770
    gmt_actor_input_dim = 770

    def __init__(self, command) -> None:
        super().__init__()
        self.log_prob_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.critic = torch.nn.Linear(289, 1, bias=False)
        self.command = command
        self.gmt_normalizer = SimpleNamespace(_mean=torch.zeros(1, 770))
        self.gmt_policy = _TrackingGMTPolicy()
        self.ref_vel_estimator = None
        self.actor_inputs: list[torch.Tensor] = []
        self.critic_inputs: list[torch.Tensor] = []
        self.env_action_inputs: list[torch.Tensor] = []
        self.action_mean = None
        self.action_std = None
        self.distribution = None

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

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        self.actor_forward(observations)
        mean = self.log_prob_scale.expand(observations.shape[0], 6)
        std = torch.full_like(mean, 0.25)
        self.action_mean = mean
        self.action_std = std
        self.distribution = torch.distributions.Normal(mean, std)
        return mean

    def evaluate(self, privileged_observations: torch.Tensor) -> torch.Tensor:
        self.critic_inputs.append(privileged_observations.detach().clone())
        return self.critic(privileged_observations)

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
        self.frontres_formal_transaction_enabled = True
        self.frontres_local_sentinel_only = True
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
        self.frontres_segment_k_curriculum = ((8, 2, 200, 500, 1300), (16, 3, 300, 300, 900), (32, 4, 400, 300, 625))
        self.frontres_segment_k_curriculum_fingerprint = ""
        self.frontres_segment_max_horizon_k = 32
        self.frontres_method_contract_id = "FRS-METHOD-v016"
        self.frontres_gain_contract_id = "FRS-GAIN-v006"
        self.frontres_optimization_contract_id = "FRS-PPO-v004"
        self.frontres_training_contract_id = "FRS-TRAIN-v011"
        self.frontres_scalar_target_id = "paired-intent-minus-repair-v1"
        self.frontres_constraint_schema_id = "contact-loaded-phase_zmp-survival-physical-v2"
        self.frontres_projection_schema_id = "grouped-first-order-constraint-projection-v1"
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
        critic = torch.arange(self.num_envs * 289, dtype=raw.dtype).reshape(self.num_envs, 289) / 1000.0
        return raw, {"observations": {"critic": critic}}

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
    current_helper = _load("frontres_v015_r5_current_command_helper", CURRENT_COMMAND_TEST)
    helper = _load("frontres_v015_r5_two_role_helper", TWO_ROLE_TEST)
    commands, hooks, setup = helper._load_owners()
    command, request = current_helper._sealed_role_command(helper, commands, hooks, setup)
    command.cfg = SimpleNamespace(motion_horizon=1, command_velocity=True)
    env_ids = torch.arange(8, dtype=torch.long)
    sealed = command.frontres_local_scenario_snapshot(env_ids)
    def repeat_to_k(value: torch.Tensor, horizon_k: int = 8) -> torch.Tensor:
        repeats = (horizon_k + int(value.shape[1]) - 1) // int(value.shape[1])
        return value.repeat((1, repeats) + (1,) * (value.ndim - 2))[:, :horizon_k].clone()

    command.clear_frontres_local_scenario()
    command.set_frontres_local_scenario(
        current_root_artifact_t=sealed["current_root_artifact_t"],
        clean_reference_t=sealed["clean_reference_t"],
        intent_q29=sealed["intent_q29"],
        clean_continuation=repeat_to_k(sealed["clean_continuation"]),
        expected_support=repeat_to_k(sealed["expected_support"]),
        expected_support_envelope=repeat_to_k(sealed["expected_support_envelope"]),
        horizon_k=torch.full((8,), 8, dtype=torch.long),
        continuation_lengths=torch.full((8,), 8, dtype=torch.long),
        scenario_ids=sealed["scenario_ids"],
        noisy_segment_hashes=sealed["noisy_segment_hashes"],
        x_t_identities=sealed["x_t_identities"],
        provenance=sealed["provenance"],
        roles=sealed["roles"],
        env_ids=env_ids,
    )
    command.refresh_frontres_reference_cache_current_frame()
    base_env = helper._FakeEnv(command, command.robot, num_envs=8)
    env = _SemanticEnv(base_env, command)
    command._env = env

    layout = resolve_frontres_future_intent_layout((1, 2), "frontres-v015-future-intent-q29-v1")
    policy = _SemanticPolicy(command)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0e-3)
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
        privileged_obs_type="critic",
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
    rsl_rl = sys.modules["rsl_rl"]
    runners = types.ModuleType("rsl_rl.runners")
    runners.__path__ = [str(RSL_SOURCE / "rsl_rl" / "runners")]
    sys.modules["rsl_rl.runners"] = runners
    rsl_rl.runners = runners
    algorithms = types.ModuleType("rsl_rl.algorithms")
    algorithms.__path__ = [str(RSL_SOURCE / "rsl_rl" / "algorithms")]
    sys.modules["rsl_rl.algorithms"] = algorithms
    rsl_rl.algorithms = algorithms
    one_action = _load("rsl_rl.runners.frontres_segment_one_action_k", ONE_ACTION_OWNER)
    runtime_types = sys.modules["rsl_rl.runners.frontres_segment_runtime_types"]
    runtime_types.open_frontres_checkpoint_transaction_barrier(runner)
    runtime_types.bind_frontres_collection_context(
        runner,
        route="training",
        sample=SimpleNamespace(identity="unmocked-observation"),
        batch=runner._frontres_segment_live_current_batch,
    )
    return SimpleNamespace(
        one_action=one_action,
        runtime_types=runtime_types,
        runner=runner,
        command=command,
        request=request,
        policy=policy,
        optimizer=optimizer,
        env=env,
    )


def test_t_unmocked_observation_to_policy_authority() -> None:
    fixture = _build_fixture()
    one_action = fixture.one_action
    runner = fixture.runner

    observations = one_action._read_live_observations(runner)
    assert tuple(observations.obs.shape) == (8, 928)
    raw = fixture.env.raw_observations[0]
    current = fixture.env.current_commands[0]
    assert tuple(raw.shape) == (8, 870)
    torch.testing.assert_close(raw[:, 100:390], current.repeat(1, 5))
    intent = fixture.command.frontres_local_scenario_intent_snapshot()["intent_q29"]
    expected_tail = torch.cat([intent[:, 1], intent[:, 2]], dim=-1) / 2.0
    torch.testing.assert_close(observations.obs[:, :58], expected_tail)
    assert runner.obs_normalizer.calls and tuple(runner.obs_normalizer.calls[0].shape) == (8, 770)
    assert dict(fixture.runtime_types.frontres_observation_trace(runner)) == {
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

    actions = runner.alg.act(observations.obs, observations.privileged_obs)
    assert tuple(actions.shape) == (8, 6)
    assert runner.alg.act_calls == 1
    assert fixture.policy.actor_inputs and tuple(fixture.policy.actor_inputs[-1].shape) == (8, 158)
    parsed = fixture.policy._parse_observations(observations.obs)
    gmt_action = fixture.policy._run_gmt_direct(*parsed)
    assert tuple(gmt_action.shape) == (8, 3)
    assert fixture.policy.gmt_policy.inputs and tuple(fixture.policy.gmt_policy.inputs[-1].shape) == (8, 770)
    print(
        "[T-command-connect/T-role-tail/T-normalizer/T-consumer/T-one-action] "
        "58/290/870 + 58 -> 928 -> FEMR 158 / frozen GMT 770; actor_calls=1",
        flush=True,
    )


if __name__ == "__main__":
    test_t_unmocked_observation_to_policy_authority()
    print("frontres_v015_unmocked_observation_connectivity_contract: ok", flush=True)
