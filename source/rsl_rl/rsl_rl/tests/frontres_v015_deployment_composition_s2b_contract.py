#!/usr/bin/env python3
"""Semantic CPU S2 contract for the v015 formal composition executor."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[4]
TEST_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl" / "tests"
S2A_HELPER_PATH = TEST_ROOT / "frontres_v015_deployment_carrier_s2a_contract.py"
RUNNER_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "on_policy_runner.py"


def _load_s2a_helper():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("frontres_v015_deployment_s2b_s2a_helper", S2A_HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FrozenGMT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.inputs: list[torch.Tensor] = []
        self.eval()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        assert tuple(obs.shape[1:]) == (770,)
        self.inputs.append(obs.detach().clone())
        return obs[:, :3].detach().clone()


class _DeploymentPolicy:
    num_actor_obs = 928
    num_frontres_obs = 158
    num_task_corrections = 6
    gmt_policy_obs_dim = 770

    def __init__(self) -> None:
        self.gmt_policy = _FrozenGMT()
        self.actor_inputs: list[torch.Tensor] = []
        self.gmt_inputs: list[torch.Tensor] = []
        self.corrections: list[torch.Tensor] = []

    def get_task_correction_inference(self, obs: torch.Tensor) -> torch.Tensor:
        assert tuple(obs.shape[1:]) == (928,)
        self.actor_inputs.append(obs.detach().clone())
        correction = torch.zeros(obs.shape[0], 6, dtype=obs.dtype)
        correction[:, 0] = 0.01 * (len(self.actor_inputs))
        self.corrections.append(correction.detach().clone())
        return correction

    def get_env_action(self, obs: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        assert tuple(obs.shape[1:]) == (928,)
        assert tuple(correction.shape[1:]) == (6,)
        gmt_obs = obs[:, -770:]
        self.gmt_inputs.append(gmt_obs.detach().clone())
        return self.gmt_policy.act_inference(gmt_obs)


class _FrozenState:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tensor = torch.tensor([1.0, 2.0])
        self.write_calls = 0

    def state_dict(self):
        return {"name": self.name, "tensor": self.tensor.detach().clone(), "writes": self.write_calls}

    def step(self, *_args, **_kwargs):
        self.write_calls += 1
        raise AssertionError(f"{self.name} mutation is forbidden during composition evaluation")

    update = step
    update_with_probe = step
    sample = step


class _DeploymentEnv:
    def __init__(self, base_env, command) -> None:
        self.unwrapped = self
        self.device = torch.device("cpu")
        self.num_envs = command.num_envs
        self.command_manager = base_env.command_manager
        self.command_manager._terms = {"motion": command}
        self.scene = base_env.scene
        self.command = command
        self.raw_reads: list[torch.Tensor] = []
        self.current_commands: list[torch.Tensor] = []
        self.motor_actions: list[torch.Tensor] = []
        self.clock_modes: list[str] = []

    def _raw(self) -> torch.Tensor:
        current = self.command.command.detach().clone()
        assert tuple(current.shape) == (self.num_envs, 58)
        prefix = torch.zeros(self.num_envs, 100)
        prefix[:, :3] = self.command.anchor_pos_w
        prefix[:, 3:7] = self.command.anchor_quat_w
        prefix[:, 7:] = torch.arange(93, dtype=torch.float32).unsqueeze(0) / 100.0
        suffix = torch.zeros(self.num_envs, 770)
        suffix[:, :290] = current.repeat(1, 5)
        suffix[:, 290:319] = self.command.robot.data.joint_pos
        suffix[:, 319:] = torch.arange(451, dtype=torch.float32).unsqueeze(0) / 10.0
        raw = torch.cat([prefix, suffix], dim=-1)
        self.raw_reads.append(raw.detach().clone())
        self.current_commands.append(current)
        return raw

    def get_observations(self):
        return self._raw(), {"observations": {}}

    def step(self, motor_actions: torch.Tensor):
        self.motor_actions.append(motor_actions.detach().clone())
        cursor = int(self.command._frontres_v015_deployment_sequence_cursor[0].item())
        snapshot = self.command.frontres_v015_deployment_sequence_snapshot()
        intent_t = snapshot["intent_q29"][:, 0]
        self.command.robot.data.joint_pos.copy_(intent_t + 0.01 * cursor)
        self.clock_modes.append(self.command._advance_frontres_command_clock())
        dones = torch.full((self.num_envs,), cursor == 2, dtype=torch.bool)
        return self._raw(), torch.zeros(self.num_envs), dones, {"observations": {}}


def _runner(helper, command, runtime, layout):
    base_env = helper._FakeEnv(command, command.robot, command.num_envs)
    env = _DeploymentEnv(base_env, command)
    policy = _DeploymentPolicy()
    optimizer = _FrozenState("optimizer")
    sampler = _FrozenState("sampler")
    normalizer_calls: list[torch.Tensor] = []

    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        alg=SimpleNamespace(policy=policy, optimizer=optimizer, transition=SimpleNamespace(marker="unchanged")),
        cfg={"empirical_normalization": True},
        policy_obs_type=None,
        _frontres_gmt_obs_dim=770,
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_layout_version=layout.version,
        _frontres_future_intent_actor_context_dim=layout.actor_tail_dim,
        _frontres_segment_sampler=sampler,
    )

    def normalize(obs: torch.Tensor) -> torch.Tensor:
        normalizer_calls.append(obs.detach().clone())
        return obs

    def apply_correction(correction: torch.Tensor, n_train: int, **_kwargs):
        assert n_train == command.num_envs
        command._frontres_pos_correction.copy_(correction[:, :3])
        return correction

    def metrics(*, frame_index: int, dones: torch.Tensor, **_kwargs):
        assert frame_index == int(command._frontres_v015_deployment_sequence_cursor[0].item())
        return {
            "physics_success": ~dones,
            "fall": dones,
            "zmp_margin": torch.full((command.num_envs,), 0.20 - 0.01 * frame_index),
            "contact_consistency": torch.full((command.num_envs,), 1.0 - 0.10 * frame_index),
        }

    runner._apply_obs_normalizer = normalize
    runner._apply_frontres_task_corrections = apply_correction
    runner._frontres_v015_deployment_metric_provider = metrics
    runner._read_frontres_v015_deployment_context = lambda env_ids=None: runtime.read_frontres_v015_deployment_context(
        runner, env_ids
    )
    runner._build_frontres_v015_deployment_observation = (
        lambda obs, snapshot=None: runtime.build_frontres_v015_deployment_observation(
            runner, obs, snapshot=snapshot
        )
    )
    runner._normalizer_calls = normalizer_calls
    return runner


def test_t_formal_composition_connectivity() -> None:
    helper = _load_s2a_helper()
    reset_helper, commands, runtime, s1_helper, owner = helper._owners()
    layout = runtime.FrontRESFutureIntentLayout(
        version=runtime.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION,
        future_offsets=(1, 2),
    )
    layout.validate()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reference, _request = helper._request(s1_helper, owner, root, frame_count=6)
        protocol = owner.build_frontres_v015_persistent_corruption_protocol(
            corruption_id="persistent-rp-s2b",
            family="local_rp",
            seed=17,
            parameters={"source": "pre_materialized_deployment_npz"},
        )
        request_config = owner.FrontRESV015DeploymentCompositionConfig(
            enabled=True,
            reference_path=str(reference),
            future_offsets=(1, 2),
            corruption_protocol=protocol,
        )
        report_path = root / "composition_report.json"
        run_config = owner.FrontRESV015DeploymentCompositionRunConfig(
            request_config=request_config,
            report_path=str(report_path),
        )

        command = helper._command(reset_helper, commands, num_envs=2)
        command.cfg = SimpleNamespace(motion_horizon=1, command_velocity=True, body_names=("pelvis",))
        command._global_sim_step = 0
        runner = _runner(reset_helper, command, runtime, layout)
        report = owner.run_frontres_v015_deployment_composition_eval(runner, config=run_config)

        assert report.reference_frame_count == 6
        assert report.frame_count == 4
        assert report.femr_action_count == 4
        assert report.accumulated_failure_count == 1
        assert report.per_frame_femr_action_used == (True, True, True, True)
        assert report.per_frame_physics_success == (True, True, False, True)
        assert report.per_frame_fall == (False, False, True, False)
        assert all(
            math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
            for actual, expected in zip(
                report.per_frame_intent_q29_error,
                (0.0, 0.01, 0.02, 0.03),
                strict=True,
            )
        )
        assert len(runner.alg.policy.actor_inputs) == 4
        assert len(runner.alg.policy.gmt_policy.inputs) == 4
        assert len(runner.env.motor_actions) == 4
        assert runner.env.clock_modes == ["deployment_current_hold"] * 4
        assert all(tuple(value.shape) == (2, 928) for value in runner.alg.policy.actor_inputs)
        assert all(tuple(value.shape) == (2, 770) for value in runner.alg.policy.gmt_policy.inputs)
        assert len(runner._normalizer_calls) == 8
        assert runner.alg.optimizer.write_calls == 0
        assert runner._frontres_segment_sampler.write_calls == 0
        assert runner.alg.transition.marker == "unchanged"
        assert not bool(command._frontres_v015_deployment_sequence_active.any())

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["evaluation_kind"] == "deployment_composition_v015"
        assert payload["reference_stream_id"] == report.request.reference_stream_id
        assert payload["corruption_protocol_hash"] == protocol.protocol_hash
        assert payload["reference_frame_count"] == 6
        assert payload["evaluated_frame_count"] == 4
        assert payload["femr_action_count"] == 4
    print(
        "[T-connect/T-per-frame/T-frozen-GMT/T-report/T-zero-write] "
        "T=6 Hmax=2 -> 4 actions, 4 GMT reads, immutable JSON, optimizer/sampler writes=0",
        flush=True,
    )


def test_t_formal_entry_and_legacy_isolation() -> None:
    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "def run_frontres_v015_deployment_composition_eval(" in runner_text
    block = runner_text.split("def run_frontres_v015_deployment_composition_eval(", 1)[1].split("\n    def ", 1)[0]
    assert "run_frontres_segment_sequence_offline_eval" not in block
    forbidden = ("to_ppo_batch", "optimizer.step", "update_with_probe", "storage", "priority")
    assert all(value not in block for value in forbidden)
    print(
        "[T-formal-entry/T-legacy-isolation] dedicated runner method does not call v002 sequence or training paths",
        flush=True,
    )


def main() -> None:
    test_t_formal_composition_connectivity()
    test_t_formal_entry_and_legacy_isolation()
    print("frontres_v015_deployment_composition_s2b_contract: ok", flush=True)


if __name__ == "__main__":
    main()
