#!/usr/bin/env python3
"""Deterministic S1 contract for the v015 two-role local-scenario reset."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl"
COMMANDS_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "commands.py"
)
HOOKS_PATH = RSL_ROOT / "rsl_rl" / "frontres" / "frontres_segment_stage1_env_hooks.py"
SETUP_PATH = RSL_ROOT / "rsl_rl" / "runners" / "frontres_training_setup.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _install_isaac_stubs() -> None:
    isaaclab = _package("isaaclab")
    assets = _package("isaaclab.assets")
    managers = _package("isaaclab.managers")
    markers = _package("isaaclab.markers")
    markers_config = _package("isaaclab.markers.config")
    utils = _package("isaaclab.utils")
    math_mod = _package("isaaclab.utils.math")

    class _Dummy:
        def __init__(self, *_args, **_kwargs) -> None:
            self.markers = {"frame": SimpleNamespace(scale=None)}

        def replace(self, **_kwargs):
            return self

        def copy(self):
            return self

    assets.Articulation = _Dummy
    managers.CommandTerm = _Dummy
    managers.CommandTermCfg = _Dummy
    markers.VisualizationMarkers = _Dummy
    markers.VisualizationMarkersCfg = _Dummy
    markers_config.FRAME_MARKER_CFG = _Dummy()
    utils.configclass = lambda cls: cls
    math_mod.euler_xyz_from_quat = lambda value: (value[..., 0], value[..., 0], value[..., 0])
    math_mod.quat_apply = lambda _quat, value: value
    math_mod.quat_error_magnitude = lambda left, _right: torch.zeros(left.shape[0], device=left.device)
    math_mod.quat_from_euler_xyz = lambda x, _y, _z: torch.stack(
        [torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)], dim=-1
    )
    math_mod.quat_inv = lambda value: value
    math_mod.quat_mul = lambda _left, right: right
    math_mod.sample_uniform = lambda _low, _high, shape, device=None: torch.zeros(shape, device=device)
    math_mod.yaw_quat = lambda value: value

    isaaclab.assets = assets
    isaaclab.managers = managers
    isaaclab.markers = markers
    isaaclab.utils = utils

    _package("whole_body_tracking")
    _package("whole_body_tracking.whole_body_tracking")
    _package("whole_body_tracking.whole_body_tracking.tasks")
    _package("whole_body_tracking.whole_body_tracking.tasks.tracking")
    mdp = _package("whole_body_tracking.whole_body_tracking.tasks.tracking.mdp")
    perturbations = types.ModuleType(
        "whole_body_tracking.whole_body_tracking.tasks.tracking.mdp.motion_perturbations"
    )
    perturbations.MotionPerturber = _Dummy
    sys.modules[perturbations.__name__] = perturbations
    mdp.motion_perturbations = perturbations


def _install_setup_stubs() -> None:
    rsl_rl = _package("rsl_rl")
    frontres = _package("rsl_rl.frontres")
    rsl_rl.__path__ = [str(RSL_ROOT / "rsl_rl")]
    frontres.__path__ = [str(RSL_ROOT / "rsl_rl" / "frontres")]
    rsl_rl.frontres = frontres

    perturbation_runtime = types.ModuleType("rsl_rl.frontres.perturbation_runtime")
    for name in (
        "apply_frontres_dr_scale",
        "apply_frontres_dr_scale_env",
        "apply_frontres_family_env_masks",
        "snapshot_frontres_perturbation_target",
    ):
        setattr(perturbation_runtime, name, lambda *_args, **_kwargs: None)
    sys.modules[perturbation_runtime.__name__] = perturbation_runtime
    frontres.perturbation_runtime = perturbation_runtime

    schedule = types.ModuleType("rsl_rl.frontres.training_schedule")

    @dataclass(frozen=True)
    class FrontRESPairLayout:
        use_quartet_reward: bool
        n_train: int
        n_candidate: int
        n_base: int
        n_clean: int
        cur_reward_sum_gmt: torch.Tensor | None

    schedule.FrontRESPairLayout = FrontRESPairLayout
    schedule.FrontRESDRIterationPlan = type("FrontRESDRIterationPlan", (), {})
    schedule.FrontRESDRScaleEnvPlan = type("FrontRESDRScaleEnvPlan", (), {})
    schedule.FrontRESDRSetup = type("FrontRESDRSetup", (), {})
    schedule.GMTFrontierState = type("GMTFrontierState", (), {})
    for name in (
        "frontres_curriculum_allowed_bases",
        "frontres_curriculum_choices",
        "frontres_curriculum_hash",
        "frontres_mixed_dr_scale",
        "frontres_mixed_dr_scale_env",
        "mode_complexity",
        "sample_perturbation_mix",
        "score_gmt_frontier",
        "update_boundary_ema",
        "update_gmt_frontier_state",
        "_frontres_boundary_scale_step",
        "_frontres_curriculum_progress",
        "_frontres_pi_scale_step",
    ):
        setattr(schedule, name, lambda *_args, **_kwargs: None)
    sys.modules[schedule.__name__] = schedule
    frontres.training_schedule = schedule


def _load_owners():
    _install_isaac_stubs()
    _install_setup_stubs()
    commands = _load(
        "whole_body_tracking.whole_body_tracking.tasks.tracking.mdp.commands",
        COMMANDS_PATH,
    )
    hooks = _load("frontres_v015_two_role_hooks_contract", HOOKS_PATH)
    setup = _load("frontres_v015_two_role_setup_contract", SETUP_PATH)
    return commands, hooks, setup


class _FakeRobotData:
    def __init__(self, num_envs: int) -> None:
        self.root_pos_w = torch.zeros(num_envs, 3)
        self.root_quat_w = torch.zeros(num_envs, 4)
        self.root_quat_w[:, 0] = 1.0
        self.root_lin_vel_w = torch.zeros(num_envs, 3)
        self.root_ang_vel_w = torch.zeros(num_envs, 3)
        self.joint_pos = torch.zeros(num_envs, 29)
        self.joint_vel = torch.zeros(num_envs, 29)


class _FakeRobot:
    def __init__(self, num_envs: int) -> None:
        self.data = _FakeRobotData(num_envs)

    def write_root_state_to_sim(self, state: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        self.data.root_pos_w[env_ids] = state[:, :3]
        self.data.root_quat_w[env_ids] = state[:, 3:7]
        self.data.root_lin_vel_w[env_ids] = state[:, 7:10]
        self.data.root_ang_vel_w[env_ids] = state[:, 10:13]

    def write_joint_state_to_sim(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        self.data.joint_pos[env_ids] = joint_pos
        self.data.joint_vel[env_ids] = joint_vel


class _FakeMotionLoader:
    def __init__(self) -> None:
        self.motion_paths = ["/tmp/motion_a.npz"]
        self.motion_paths_all = list(self.motion_paths)
        self.motion_lengths = torch.tensor([32], dtype=torch.long)
        self.motion_to_group = {0: "default"}
        self.joint_pos = torch.zeros(1, 29)
        self.joint_vel = torch.zeros(1, 29)

    def gather(self, getter: str, motion_ids: torch.Tensor, frame_ids: torch.Tensor, out_device) -> torch.Tensor:
        del motion_ids
        frame = frame_ids.to(torch.float32).reshape(-1, 1)
        if getter == "joint_pos":
            return (frame + torch.arange(29, dtype=torch.float32).reshape(1, 29)).to(out_device)
        if getter == "joint_vel":
            return (100.0 + frame + torch.arange(29, dtype=torch.float32).reshape(1, 29)).to(out_device)
        if getter == "body_pos_w":
            return torch.stack([frame[:, 0], frame[:, 0] + 1.0, frame[:, 0] + 2.0], dim=-1).unsqueeze(1).to(out_device)
        if getter == "body_quat_w":
            value = torch.zeros(frame.shape[0], 1, 4)
            value[..., 0] = 1.0
            return value.to(out_device)
        if getter == "body_lin_vel_w":
            return torch.full((frame.shape[0], 1, 3), 0.25, dtype=torch.float32, device=out_device)
        if getter == "body_ang_vel_w":
            return torch.full((frame.shape[0], 1, 3), 0.50, dtype=torch.float32, device=out_device)
        raise KeyError(getter)


class _RejectPerturber:
    def __init__(self) -> None:
        self.calls = 0

    def _reject(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("Step 2A local reset must not resample or apply a perturber")

    reset_envs = _reject
    apply_perturbations = _reject
    apply_quat_perturbation = _reject
    apply_joint_perturbation = _reject


class _FakeCommandManager:
    def __init__(self, command) -> None:
        self._command = command

    def get_term(self, name: str):
        assert name == "motion"
        return self._command


class _FakeScene:
    def __init__(self, robot: _FakeRobot, num_envs: int) -> None:
        self.robot = robot
        self.env_origins = torch.zeros(num_envs, 3)
        self.write_data_calls = 0

    def __getitem__(self, name: str):
        assert name == "robot"
        return self.robot

    def write_data_to_sim(self) -> None:
        self.write_data_calls += 1


class _FakeSim:
    def __init__(self) -> None:
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeObservationManager:
    def __init__(self, num_envs: int) -> None:
        self.history = torch.ones(num_envs, 5)
        self.reset_calls: list[torch.Tensor] = []

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        ids = torch.as_tensor(env_ids, dtype=torch.long).clone()
        self.history[ids] = 0.0
        self.reset_calls.append(ids)
        return {}


class _FakeEnv:
    def __init__(self, command, robot: _FakeRobot, num_envs: int) -> None:
        self.unwrapped = self
        self.num_envs = num_envs
        self.command_manager = _FakeCommandManager(command)
        self.scene = _FakeScene(robot, num_envs)
        self.sim = _FakeSim()
        self.observation_manager = _FakeObservationManager(num_envs)
        self.episode_length_buf = torch.full((num_envs,), 9, dtype=torch.long)


def _make_command(commands, robot: _FakeRobot, num_envs: int):
    command = object.__new__(commands.MultiMotionCommand)
    command.num_envs = num_envs
    command.device = torch.device("cpu")
    command.robot = robot
    command.motion_dir_loader = _FakeMotionLoader()
    command.motion_lengths = command.motion_dir_loader.motion_lengths
    command.env_motion_indices = torch.zeros(num_envs, dtype=torch.long)
    command.env_motion_groups = torch.full((num_envs,), -1, dtype=torch.long)
    command.group_name_to_idx = {"default": 0}
    command.time_steps = torch.zeros(num_envs, dtype=torch.long)
    command.motion_end_buf = torch.zeros(num_envs, dtype=torch.bool)
    command.motion_anchor_body_index = 0
    command.left_foot_idx = 0
    command.right_foot_idx = 0
    command._env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)))
    command._frontres_pos_correction = torch.ones(num_envs, 3)
    command._frontres_quat_correction = torch.zeros(num_envs, 4)
    command._frontres_quat_correction[:, 0] = 1.0
    command._cached_perturbed_pos = torch.zeros(num_envs, 3)
    command._cached_perturbed_quat = torch.zeros(num_envs, 4)
    command._cached_perturbed_quat[:, 0] = 1.0
    command._dr_supervised_target = torch.zeros(num_envs, 6)
    command._frontres_pair_train_ids = None
    command._frontres_pair_candidate_ids = None
    command._frontres_pair_base_ids = None
    command._frontres_pair_clean_ids = None
    command.perturber = _RejectPerturber()
    command._init_frontres_reference_window_buffers()
    return command


def _local_request(role_env_ids: dict[str, torch.Tensor]) -> SimpleNamespace:
    artifacts = torch.tensor(
        [[10.0, 11.0, 12.0, 1.0, 0.1, 0.2, 0.3], [20.0, 21.0, 22.0, 1.0, 0.4, 0.5, 0.6]]
    )
    intent = torch.stack(
        [torch.arange(87, dtype=torch.float32).reshape(3, 29), 1000.0 + torch.arange(87, dtype=torch.float32).reshape(3, 29)],
        dim=0,
    )
    continuation = torch.stack(
        [torch.full((3, 65), 300.0), torch.full((3, 65), 400.0)],
        dim=0,
    )
    clean_reference_t = torch.stack(
        [torch.arange(65, dtype=torch.float32), 100.0 + torch.arange(65, dtype=torch.float32)],
        dim=0,
    )
    return SimpleNamespace(
        segment_ids=torch.tensor([7, 8], dtype=torch.long),
        motion_ids=("motion_a.npz", "motion_a.npz"),
        start_frames=torch.tensor([2, 4], dtype=torch.long),
        horizon_k=torch.tensor([3, 2], dtype=torch.long),
        frontres_future_offsets=(1, 2),
        frontres_local_scenario_rows=object(),
        frontres_local_scenario_current_root_artifact_t=artifacts,
        frontres_local_scenario_clean_reference_t=clean_reference_t,
        frontres_local_scenario_intent_q29=intent,
        frontres_local_scenario_clean_continuation=continuation,
        frontres_local_scenario_expected_support=torch.ones(2, 3, 2),
        frontres_local_scenario_expected_support_envelope=torch.tensor(
            [[[0.0, 0.0, 1.0, 0.0, 0.1, 0.05]] * 3, [[0.0, 0.0, 1.0, 0.0, 0.1, 0.05]] * 3]
        ),
        frontres_local_scenario_clean_continuation_lengths=torch.tensor([3, 2], dtype=torch.long),
        frontres_local_scenario_clean_continuation_mask=torch.tensor([[True, True, True], [True, True, False]]),
        frontres_local_scenario_ids=("scenario-a", "scenario-b"),
        frontres_local_scenario_hashes=("hash-a", "hash-b"),
        frontres_local_scenario_x_t_identities=("x_t-a", "x_t-b"),
        frontres_local_scenario_provenance=(
            {
                "current_root_artifact_provenance": "noisy_root_artifact_t",
                "clean_reference_t_provenance": "clean_gmt_physics_only",
                "intent_q29_provenance": "deployment_noisy_q29",
                "intent_q29_source": "motion_internal_q29",
                "clean_continuation_provenance": "clean_gmt_only",
                "expected_support_provenance": "clean_gmt_physics_only",
                "expected_support_envelope_provenance": "clean_gmt_physics_only",
            },
            {
                "current_root_artifact_provenance": "noisy_root_artifact_t",
                "clean_reference_t_provenance": "clean_gmt_physics_only",
                "intent_q29_provenance": "deployment_noisy_q29",
                "intent_q29_source": "motion_internal_q29",
                "clean_continuation_provenance": "clean_gmt_only",
                "expected_support_provenance": "clean_gmt_physics_only",
                "expected_support_envelope_provenance": "clean_gmt_physics_only",
            },
        ),
        frontres_segment_source_index=torch.arange(2, dtype=torch.long),
        frontres_role_env_ids=role_env_ids,
    )


def _parallel_attempt_request(role_env_ids: dict[str, torch.Tensor]) -> SimpleNamespace:
    request = _local_request(role_env_ids)
    for name in (
        "segment_ids",
        "start_frames",
        "horizon_k",
        "frontres_local_scenario_current_root_artifact_t",
        "frontres_local_scenario_clean_reference_t",
        "frontres_local_scenario_intent_q29",
        "frontres_local_scenario_clean_continuation",
        "frontres_local_scenario_expected_support",
        "frontres_local_scenario_expected_support_envelope",
        "frontres_local_scenario_clean_continuation_lengths",
        "frontres_local_scenario_clean_continuation_mask",
    ):
        value = getattr(request, name)
        setattr(request, name, value.repeat_interleave(2, dim=0))
    request.motion_ids = ("motion_a.npz", "motion_a.npz", "motion_a.npz", "motion_a.npz")
    request.frontres_local_scenario_ids = ("scenario-a", "scenario-a", "scenario-b", "scenario-b")
    request.frontres_local_scenario_hashes = ("hash-a", "hash-a", "hash-b", "hash-b")
    request.frontres_local_scenario_x_t_identities = ("x_t-a", "x_t-a", "x_t-b", "x_t-b")
    request.frontres_local_scenario_provenance = (
        request.frontres_local_scenario_provenance[0],
        request.frontres_local_scenario_provenance[0],
        request.frontres_local_scenario_provenance[1],
        request.frontres_local_scenario_provenance[1],
    )
    request.frontres_segment_source_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return request


def _expect_error(exc_type, callback, contains: str) -> None:
    try:
        callback()
    except exc_type as exc:
        assert contains in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {exc_type.__name__} containing {contains!r}")


def test_t_role_layout(commands, setup) -> tuple[_FakeEnv, object, dict[str, torch.Tensor]]:
    robot = _FakeRobot(num_envs=4)
    command = _make_command(commands, robot, num_envs=4)
    env = _FakeEnv(command, robot, num_envs=4)
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        cfg={"frontres_candidate_rollout_enabled": True},
        alg=SimpleNamespace(frontres_future_offsets=(1, 2)),
        _frontres_future_intent_layout=SimpleNamespace(version="frontres-v015-future-intent-q29-v1"),
    )
    layout = setup.configure_frontres_pair_layout(runner, is_frontres=True)
    role_env_ids = runner._frontres_v015_two_role_env_ids
    assert layout.use_quartet_reward is False
    assert (layout.n_train, layout.n_candidate, layout.n_base, layout.n_clean) == (2, 0, 2, 0)
    assert tuple(role_env_ids) == ("repair", "noisy")
    assert role_env_ids["repair"].tolist() == [0, 1]
    assert role_env_ids["noisy"].tolist() == [2, 3]
    assert command._frontres_pair_train_ids.tolist() == [0, 1]
    assert command._frontres_pair_base_ids.tolist() == [2, 3]
    assert command._frontres_pair_candidate_ids is None
    assert command._frontres_pair_clean_ids is None
    print("[T-2A-role] repair=2 noisy=2 candidate=0 clean=0 v015_layout=true", flush=True)
    return env, command, role_env_ids


def test_t_state_and_identity(hooks, env, command, role_env_ids) -> None:
    adapter = hooks.FrontRESStage1EnvAdapter(env=env, amass_root="/tmp", trace=False)
    request = _local_request(role_env_ids)
    request.frontres_local_scenario_execution_mode = "clean_baseline"
    source_artifacts = request.frontres_local_scenario_current_root_artifact_t.clone()
    source_clean = request.frontres_local_scenario_clean_reference_t.clone()
    source_intent = request.frontres_local_scenario_intent_q29.clone()
    source_continuation = request.frontres_local_scenario_clean_continuation.clone()
    result = adapter.apply_frontres_segment_index_reset(request)
    assert result["reset_success"].tolist() == [True, True]
    snapshot = command.frontres_local_scenario_snapshot(torch.arange(4))
    expected_rows = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    torch.testing.assert_close(snapshot["current_root_artifact_t"], source_artifacts.index_select(0, expected_rows))
    torch.testing.assert_close(snapshot["intent_q29"], source_intent.index_select(0, expected_rows))
    torch.testing.assert_close(snapshot["clean_continuation"], source_continuation.index_select(0, expected_rows))
    assert snapshot["horizon_k"].tolist() == [3, 2, 3, 2]
    assert snapshot["scenario_ids"] == ("scenario-a", "scenario-b", "scenario-a", "scenario-b")
    assert snapshot["noisy_segment_hashes"] == ("hash-a", "hash-b", "hash-a", "hash-b")
    assert snapshot["x_t_identities"] == ("x_t-a", "x_t-b", "x_t-a", "x_t-b")
    assert snapshot["roles"] == ("repair", "repair", "noisy", "noisy")
    torch.testing.assert_close(command._cached_perturbed_pos, source_clean.index_select(0, expected_rows)[:, 58:61])
    torch.testing.assert_close(command._cached_perturbed_quat, source_clean.index_select(0, expected_rows)[:, 61:65])
    assert command._frontres_local_scenario_execution_mode == "clean_baseline"
    assert not bool(command._frontres_reference_window_active.any())
    assert command.perturber.calls == 0

    expected_clean_x_t = torch.tensor([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    torch.testing.assert_close(env.scene.robot.data.root_pos_w, expected_clean_x_t)
    assert not torch.equal(env.scene.robot.data.root_pos_w, command._cached_perturbed_pos)
    assert env.scene.write_data_calls == 1
    assert env.sim.forward_calls == 1
    assert len(env.observation_manager.reset_calls) == 1
    assert env.observation_manager.reset_calls[0].tolist() == [0, 1, 2, 3]
    assert not bool(env.observation_manager.history.any())
    assert result["source_state_max_abs_diff"].tolist() == [0.0, 0.0]

    # M attempts are new physical resets over the exact same sealed command carrier.
    retry_request = _local_request(role_env_ids)
    retry_request.frontres_local_scenario_execution_mode = "repair_attempts"
    retry_result = adapter.apply_frontres_segment_index_reset(retry_request)
    assert retry_result["reset_success"].tolist() == [True, True]
    retry_snapshot = command.frontres_local_scenario_snapshot(torch.arange(4))
    torch.testing.assert_close(retry_snapshot["current_root_artifact_t"], source_artifacts.index_select(0, expected_rows))
    torch.testing.assert_close(retry_snapshot["intent_q29"], source_intent.index_select(0, expected_rows))
    torch.testing.assert_close(retry_snapshot["clean_continuation"], source_continuation.index_select(0, expected_rows))
    assert retry_snapshot["noisy_segment_hashes"] == ("hash-a", "hash-b", "hash-a", "hash-b")
    assert command._frontres_local_scenario_execution_mode == "repair_attempts"
    torch.testing.assert_close(command._cached_perturbed_pos, source_artifacts.index_select(0, expected_rows)[:, :3])
    torch.testing.assert_close(command._cached_perturbed_quat, source_artifacts.index_select(0, expected_rows)[:, 3:])
    assert command.perturber.calls == 0
    assert env.scene.write_data_calls == 2
    assert env.sim.forward_calls == 2
    assert len(env.observation_manager.reset_calls) == 2
    assert retry_result["source_state_max_abs_diff"].tolist() == [0.0, 0.0]
    print("[T-2A-scenario-identity] retry reuses x_t/artifact/q29/C/K/hash without resampling", flush=True)

    mutated_active_artifact = retry_snapshot["current_root_artifact_t"].clone()
    mutated_active_artifact[0, 0] += 1.0
    mutated_active_artifact[2, 0] += 1.0
    _expect_error(
        RuntimeError,
        lambda: command.set_frontres_local_scenario(
            current_root_artifact_t=mutated_active_artifact,
            clean_reference_t=retry_snapshot["clean_reference_t"],
            intent_q29=retry_snapshot["intent_q29"],
            clean_continuation=retry_snapshot["clean_continuation"],
            expected_support=retry_snapshot["expected_support"],
            expected_support_envelope=retry_snapshot["expected_support_envelope"],
            horizon_k=retry_snapshot["horizon_k"],
            continuation_lengths=retry_snapshot["continuation_lengths"],
            scenario_ids=retry_snapshot["scenario_ids"],
            noisy_segment_hashes=retry_snapshot["noisy_segment_hashes"],
            x_t_identities=retry_snapshot["x_t_identities"],
            provenance=retry_snapshot["provenance"],
            roles=retry_snapshot["roles"],
            env_ids=torch.arange(4),
        ),
        "mutate",
    )

    request.frontres_local_scenario_current_root_artifact_t.fill_(-999.0)
    request.frontres_local_scenario_intent_q29.fill_(-999.0)
    request.frontres_local_scenario_clean_continuation.fill_(-999.0)
    snapshot_after_mutation = command.frontres_local_scenario_snapshot(torch.arange(4))
    torch.testing.assert_close(snapshot_after_mutation["current_root_artifact_t"], source_artifacts.index_select(0, expected_rows))
    torch.testing.assert_close(snapshot_after_mutation["intent_q29"], source_intent.index_select(0, expected_rows))
    torch.testing.assert_close(snapshot_after_mutation["clean_continuation"], source_continuation.index_select(0, expected_rows))
    command.cfg = SimpleNamespace(motion_horizon=1, command_velocity=False)
    _expect_error(RuntimeError, lambda: command.command, "command_velocity=True")
    _expect_error(RuntimeError, command.refresh_frontres_reference_cache_current_frame, "Step 2B")
    _ = command.joint_pos
    assert command.perturber.calls == 0
    print(
        "[T-2A-state] clean_x_t_reset=true artifact/intention/C/K/hash immutable across repair/noisy, "
        "clean_C_current_reference=false",
        flush=True,
    )


def test_t_parallel_m_attempt_role_balance(commands, hooks, setup) -> None:
    robot = _FakeRobot(num_envs=8)
    command = _make_command(commands, robot, num_envs=8)
    env = _FakeEnv(command, robot, num_envs=8)
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        cfg={"frontres_candidate_rollout_enabled": True},
        alg=SimpleNamespace(frontres_future_offsets=(1, 2)),
        _frontres_future_intent_layout=SimpleNamespace(version="frontres-v015-future-intent-q29-v1"),
    )
    layout = setup.configure_frontres_pair_layout(runner, is_frontres=True)
    role_env_ids = runner._frontres_v015_two_role_env_ids
    assert (layout.n_train, layout.n_base) == (4, 4)
    assert role_env_ids["repair"].tolist() == [0, 1, 2, 3]
    assert role_env_ids["noisy"].tolist() == [4, 5, 6, 7]

    adapter = hooks.FrontRESStage1EnvAdapter(env=env, amass_root="/tmp", trace=False)
    request = _parallel_attempt_request(role_env_ids)
    result = adapter.apply_frontres_segment_index_reset(request)
    assert result["reset_success"].tolist() == [True, True, True, True]
    snapshot = command.frontres_local_scenario_snapshot(torch.arange(8))
    assert snapshot["scenario_ids"] == (
        "scenario-a",
        "scenario-a",
        "scenario-b",
        "scenario-b",
        "scenario-a",
        "scenario-a",
        "scenario-b",
        "scenario-b",
    )
    assert snapshot["roles"] == ("repair", "repair", "repair", "repair", "noisy", "noisy", "noisy", "noisy")
    expected_rows = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    torch.testing.assert_close(
        snapshot["current_root_artifact_t"],
        request.frontres_local_scenario_current_root_artifact_t.index_select(0, expected_rows),
    )
    torch.testing.assert_close(
        snapshot["intent_q29"],
        request.frontres_local_scenario_intent_q29.index_select(0, expected_rows),
    )
    torch.testing.assert_close(
        snapshot["clean_continuation"],
        request.frontres_local_scenario_clean_continuation.index_select(0, expected_rows),
    )
    command.robot.data.joint_pos[1, 0] += 0.1
    _expect_error(
        RuntimeError,
        lambda: adapter._require_frontres_source_shared_robot_state(
            request,
            repair_env_ids=role_env_ids["repair"],
        ),
        "not one physical Segment state",
    )

    def install(candidate_command, *, intent_q29, roles):
        return candidate_command.set_frontres_local_scenario(
            current_root_artifact_t=snapshot["current_root_artifact_t"],
            clean_reference_t=snapshot["clean_reference_t"],
            intent_q29=intent_q29,
            clean_continuation=snapshot["clean_continuation"],
            expected_support=snapshot["expected_support"],
            expected_support_envelope=snapshot["expected_support_envelope"],
            horizon_k=snapshot["horizon_k"],
            continuation_lengths=snapshot["continuation_lengths"],
            scenario_ids=snapshot["scenario_ids"],
            noisy_segment_hashes=snapshot["noisy_segment_hashes"],
            x_t_identities=snapshot["x_t_identities"],
            provenance=snapshot["provenance"],
            roles=roles,
            env_ids=torch.arange(8),
        )

    unbalanced_roles = list(snapshot["roles"])
    unbalanced_roles[5] = "repair"
    _expect_error(
        ValueError,
        lambda: install(
            _make_command(commands, _FakeRobot(num_envs=8), num_envs=8),
            intent_q29=snapshot["intent_q29"],
            roles=tuple(unbalanced_roles),
        ),
        "balanced",
    )
    mutated_intent = snapshot["intent_q29"].clone()
    mutated_intent[5, 0, 0] += 1.0
    _expect_error(
        ValueError,
        lambda: install(
            _make_command(commands, _FakeRobot(num_envs=8), num_envs=8),
            intent_q29=mutated_intent,
            roles=snapshot["roles"],
        ),
        "immutable local scenario",
    )
    print("[T-2A-parallel-M] two scenarios x M=2 install balanced immutable Repair/Noisy rows", flush=True)


def test_t_legacy_reject(hooks, env, role_env_ids) -> None:
    adapter = hooks.FrontRESStage1EnvAdapter(env=env, amass_root="/tmp", trace=False)
    legacy_roles = {"policy": role_env_ids["repair"], "noisy": role_env_ids["noisy"]}
    request = _local_request(legacy_roles)
    _expect_error(ValueError, lambda: adapter.apply_frontres_segment_index_reset(request), "repair/noisy")
    print("[T-2A-legacy-reject] policy/candidate/clean role names cannot enter v015 local reset", flush=True)


def main() -> None:
    commands, hooks, setup = _load_owners()
    env, command, role_env_ids = test_t_role_layout(commands, setup)
    test_t_state_and_identity(hooks, env, command, role_env_ids)
    test_t_parallel_m_attempt_role_balance(commands, hooks, setup)
    test_t_legacy_reject(hooks, env, role_env_ids)
    print("frontres_v015_two_role_reset_contract: ok", flush=True)


if __name__ == "__main__":
    main()
