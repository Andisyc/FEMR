#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
import importlib.util
import sys
import tempfile
import types

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
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
SOURCE_ROOT = ROOT / "source" / "rsl_rl"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hooks = _load_module(
    "frontres_segment_stage1_env_hooks",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_stage1_env_hooks.py",
)
builder = _load_module(
    "frontres_segment_cache_builder_for_stage1_hooks",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_builder.py",
)
cache_io = _load_module(
    "frontres_segment_cache_io_for_stage1_hooks",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py",
)
schema = _load_module(
    "frontres_segment_cache_schema_for_stage1_hooks",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_schema.py",
)

FrontRESStage1EnvAdapter = hooks.FrontRESStage1EnvAdapter
FrontRESStage1CacheBuilderConfig = builder.FrontRESStage1CacheBuilderConfig
build_stage1_segment_cache = builder.build_stage1_segment_cache
FrontRESPerturbationDescriptor = schema.FrontRESPerturbationDescriptor
FrontRESRobotRolloutState = schema.FrontRESRobotRolloutState
FrontRESSegmentIndex = schema.FrontRESSegmentIndex


class FakeRobotData:
    def __init__(self, num_envs: int = 1, dofs: int = 29, bodies: int = 30) -> None:
        self.root_pos_w = torch.zeros(num_envs, 3)
        self.root_quat_w = torch.zeros(num_envs, 4)
        self.root_quat_w[:, 0] = 1.0
        self.root_lin_vel_w = torch.zeros(num_envs, 3)
        self.root_ang_vel_w = torch.zeros(num_envs, 3)
        self.joint_pos = torch.zeros(num_envs, dofs)
        self.joint_vel = torch.zeros(num_envs, dofs)
        self.body_pos_w = torch.zeros(num_envs, bodies, 3)
        self.body_quat_w = torch.zeros(num_envs, bodies, 4)
        self.body_quat_w[:, :, 0] = 1.0
        self.body_lin_vel_w = torch.zeros(num_envs, bodies, 3)
        self.body_ang_vel_w = torch.zeros(num_envs, bodies, 3)


class FakeRobot:
    def __init__(self, num_envs: int = 1) -> None:
        self.data = FakeRobotData(num_envs=num_envs)
        self.root_writes: list[torch.Tensor] = []
        self.joint_writes: list[torch.Tensor] = []

    def write_root_state_to_sim(self, root_state: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        self.root_writes.append(root_state.detach().clone())
        self.data.root_pos_w[env_ids] = root_state[:, 0:3]
        self.data.root_quat_w[env_ids] = root_state[:, 3:7]
        self.data.root_lin_vel_w[env_ids] = root_state[:, 7:10]
        self.data.root_ang_vel_w[env_ids] = root_state[:, 10:13]
        self.data.body_pos_w[env_ids, 0] = root_state[:, 0:3]
        self.data.body_quat_w[env_ids, 0] = root_state[:, 3:7]
        self.data.body_lin_vel_w[env_ids, 0] = root_state[:, 7:10]
        self.data.body_ang_vel_w[env_ids, 0] = root_state[:, 10:13]

    def write_joint_state_to_sim(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        self.joint_writes.append(joint_pos.detach().clone())
        self.data.joint_pos[env_ids] = joint_pos
        self.data.joint_vel[env_ids] = joint_vel


class FakeMotionLoader:
    def __init__(self, root: Path) -> None:
        self.motion_paths = [str(root / "KIT" / "359" / "motion_a.npz")]
        self.motion_paths_all = list(self.motion_paths)
        self.shard_info = {
            "selected_motions": len(self.motion_paths),
            "total_motions": len(self.motion_paths_all),
        }
        self.motion_lengths = torch.tensor([8], dtype=torch.long)
        self.motion_fps = torch.tensor([30.0], dtype=torch.float32)
        self.motion_to_group = {0: "default"}

    def gather(self, attr: str, motion_indices: torch.Tensor, frame_indices: torch.Tensor, out_device) -> torch.Tensor:
        batch = motion_indices.numel()
        frames = frame_indices.to(torch.float32).view(batch, 1)
        if attr == "joint_pos":
            return frames + torch.arange(29, dtype=torch.float32).view(1, 29)
        if attr == "joint_vel":
            return 0.01 * torch.arange(29, dtype=torch.float32).view(1, 29)
        if attr == "body_pos_w":
            body = torch.zeros(batch, 30, 3)
            body[:, :, 0] = frames
            body[:, :, 1] = torch.arange(30, dtype=torch.float32).view(1, 30)
            body[:, :, 2] = 1.0
            return body
        if attr == "body_quat_w":
            quat = torch.zeros(batch, 30, 4)
            quat[:, :, 0] = 1.0
            return quat
        if attr == "body_lin_vel_w":
            vel = torch.zeros(batch, 30, 3)
            vel[:, :, 0] = 0.2
            return vel
        if attr == "body_ang_vel_w":
            vel = torch.zeros(batch, 30, 3)
            vel[:, :, 2] = 0.3
            return vel
        raise KeyError(attr)


class FakePerturber:
    def __init__(self) -> None:
        self.reset_calls: list[list[int]] = []
        self.dr_scale_env: torch.Tensor | None = None
        self.family_masks: dict[str, torch.Tensor] | None = None

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        self.reset_calls.append(env_ids.detach().cpu().tolist())

    def set_dr_scale_env(self, scales: torch.Tensor | None) -> None:
        self.dr_scale_env = None if scales is None else scales.detach().clone()

    def set_family_env_masks(self, masks: dict[str, torch.Tensor] | None) -> None:
        self.family_masks = None if masks is None else {key: value.detach().clone() for key, value in masks.items()}


class FakeCommand:
    def __init__(self, root: Path, robot: FakeRobot, num_envs: int = 1) -> None:
        self.device = torch.device("cpu")
        self.num_envs = int(num_envs)
        self.robot = robot
        self.cfg = types.SimpleNamespace(
            motion_dataset_load_cap=1,
            motion_dataset_shard_across_gpus=False,
        )
        self.motion_dir_loader = FakeMotionLoader(root)
        self.motion_lengths = self.motion_dir_loader.motion_lengths
        self.env_motion_indices = torch.zeros(self.num_envs, dtype=torch.long)
        self.env_motion_groups = torch.full((self.num_envs,), -1, dtype=torch.long)
        self.group_name_to_idx = {"default": 0}
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.motion_end_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self._frontres_pos_correction = torch.ones(self.num_envs, 3)
        self._frontres_quat_correction = torch.zeros(self.num_envs, 4)
        self._cached_perturbed_pos = torch.zeros(self.num_envs, 3)
        self._cached_perturbed_quat = torch.zeros(self.num_envs, 4)
        self._cached_perturbed_quat[:, 0] = 1.0
        self._dr_supervised_target = torch.zeros(self.num_envs, 6)
        self.cache_refresh_calls = 0
        self.perturber = FakePerturber()
        self.fixed_tape_install_calls: list[dict[str, object]] = []
        self.fixed_tape: torch.Tensor | None = None
        self.fixed_tape_execution_mask = torch.zeros(self.num_envs, dtype=torch.bool)
        self.fixed_tape_hashes: tuple[str, ...] = ()
        self.materialize_calls: list[dict[str, object]] = []
        self.metrics = {
            "error_anchor_pos": torch.zeros(1),
            "error_anchor_rot": torch.zeros(1),
        }

    def _gather_by_motion_for_envs(self, getter: str, env_ids: torch.Tensor) -> torch.Tensor:
        return self.motion_dir_loader.gather(
            getter,
            self.env_motion_indices[env_ids],
            self.time_steps[env_ids],
            out_device=self.device,
        )

    def _update_metrics(self) -> None:
        self.metrics["error_anchor_pos"][:] = torch.norm(self.robot.data.root_pos_w[:, :2], dim=-1)
        self.metrics["error_anchor_rot"][:] = 0.25

    def refresh_frontres_reference_cache_current_frame(self) -> None:
        """Model the command-owned current-frame cache boundary for connector tests."""

        self.cache_refresh_calls += 1
        body_pos = self._gather_by_motion_for_envs("body_pos_w", torch.arange(self.num_envs))
        body_quat = self._gather_by_motion_for_envs("body_quat_w", torch.arange(self.num_envs))
        self._cached_perturbed_pos.copy_(body_pos[:, 0])
        self._cached_perturbed_quat.copy_(body_quat[:, 0])
        self._dr_supervised_target.zero_()
        if self.fixed_tape is not None:
            execution_ids = torch.nonzero(self.fixed_tape_execution_mask, as_tuple=False).flatten()
            if int(execution_ids.numel()) > 0:
                self._cached_perturbed_pos[execution_ids] = self.fixed_tape[execution_ids, 0, 58:61]
                self._cached_perturbed_quat[execution_ids] = self.fixed_tape[execution_ids, 0, 61:65]

    def set_frontres_fixed_noisy_tape(
        self,
        tape: torch.Tensor,
        *,
        tape_lengths: torch.Tensor,
        scenario_ids: tuple[str, ...],
        noisy_segment_hashes: tuple[str, ...],
        execution_mask: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.fixed_tape_install_calls.append(
            {
                "tape": tape.detach().clone(),
                "tape_lengths": tape_lengths.detach().clone(),
                "scenario_ids": tuple(scenario_ids),
                "noisy_segment_hashes": tuple(noisy_segment_hashes),
                "execution_mask": execution_mask.detach().clone(),
                "env_ids": env_ids.detach().clone(),
            }
        )
        self.fixed_tape = torch.zeros(self.num_envs, tape.shape[1], tape.shape[2])
        self.fixed_tape[env_ids] = tape
        self.fixed_tape_execution_mask[env_ids] = execution_mask
        self.fixed_tape_hashes = tuple(noisy_segment_hashes)
        return torch.ones(env_ids.numel(), dtype=torch.bool)

    def clear_frontres_fixed_noisy_tape(self, env_ids: torch.Tensor) -> None:
        if self.fixed_tape is not None:
            self.fixed_tape[env_ids] = 0.0
        self.fixed_tape_execution_mask[env_ids] = False

    def materialize_frontres_fixed_noisy_tape(self, **kwargs) -> torch.Tensor:
        self.materialize_calls.append(dict(kwargs))
        frame_count = int(kwargs["frame_count"])
        start_frame = float(kwargs["start_frame"])
        strength = float(kwargs["perturbation_strength"])
        return torch.full((frame_count, 65), start_frame + strength, dtype=torch.float32)

    def materialize_frontres_local_scenario(self, **kwargs) -> dict[str, object]:
        """Provide a hand-constructed Clean/deployment carrier at the command seam."""

        self.materialize_calls.append(dict(kwargs))
        start = int(kwargs["start_frame"])
        horizon_k = int(kwargs["horizon_k"])
        intent_horizon = int(kwargs["intent_horizon"])

        def command_at(frame: int) -> torch.Tensor:
            frame_ids = torch.tensor([frame], dtype=torch.long)
            motion_ids = torch.zeros(1, dtype=torch.long)
            q = self.motion_dir_loader.gather("joint_pos", motion_ids, frame_ids, self.device)[0]
            dq = self.motion_dir_loader.gather("joint_vel", motion_ids, frame_ids, self.device)[0]
            root_pos = self.motion_dir_loader.gather("body_pos_w", motion_ids, frame_ids, self.device)[0, 0]
            root_quat = self.motion_dir_loader.gather("body_quat_w", motion_ids, frame_ids, self.device)[0, 0]
            return torch.cat((q, dq, root_pos, root_quat))

        clean_t = command_at(start)
        return {
            "current_root_artifact_t": clean_t[58:65].clone(),
            "clean_reference_t": clean_t.clone(),
            "intent_q29": torch.stack([command_at(start + offset)[:29] for offset in range(intent_horizon + 1)]),
            "clean_continuation": torch.stack([command_at(start + offset) for offset in range(1, horizon_k + 1)]),
            "expected_support": torch.ones(horizon_k, 2),
            "expected_support_envelope": torch.zeros(horizon_k, 6),
            "provenance": {"owner": "FakeCommand", "source": "hand_constructed_clean_motion"},
        }


class FakeCommandManager:
    def __init__(self, command: FakeCommand) -> None:
        self.command = command

    def get_term(self, name: str):
        assert name == "motion"
        return self.command


class FakeScene:
    def __init__(self, robot: FakeRobot, num_envs: int = 1) -> None:
        self.robot = robot
        self.env_origins = torch.zeros(num_envs, 3)

    def __getitem__(self, name: str):
        assert name == "robot"
        return self.robot


class FakeBaseEnv:
    def __init__(self, root: Path, num_envs: int = 1) -> None:
        self.num_envs = int(num_envs)
        self.num_actions = 29
        self.robot = FakeRobot(num_envs=self.num_envs)
        self.scene = FakeScene(self.robot, num_envs=self.num_envs)
        self.command = FakeCommand(root, self.robot, num_envs=self.num_envs)
        self.command_manager = FakeCommandManager(self.command)
        self.episode_length_buf = torch.full((self.num_envs,), 99, dtype=torch.long)


class FakeGymEnv:
    def __init__(self, root: Path, num_envs: int = 1) -> None:
        self.unwrapped = FakeBaseEnv(root, num_envs=num_envs)
        self.step_actions: list[torch.Tensor] = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        return None, {}

    def step(self, actions: torch.Tensor):
        if self.reset_count <= 0:
            raise RuntimeError("Cannot call env.step() before calling env.reset()")
        self.step_actions.append(actions.detach().clone())
        rewards = torch.full((self.unwrapped.num_envs,), 0.75, dtype=torch.float32)
        dones = torch.zeros(self.unwrapped.num_envs, dtype=torch.bool)
        return None, rewards, dones, {}


def _make_stage1_fake_state_inference(env: FakeGymEnv) -> None:
    robot_data = env.unwrapped.robot.data
    command = env.unwrapped.command
    with torch.inference_mode():
        for name in (
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            setattr(robot_data, name, getattr(robot_data, name).clone())
        command.env_motion_indices = command.env_motion_indices.clone()
        command.time_steps = command.time_steps.clone()
        command.motion_end_buf = command.motion_end_buf.clone()
        command._frontres_pos_correction = command._frontres_pos_correction.clone()
        command._frontres_quat_correction = command._frontres_quat_correction.clone()


def _write_fake_amass(path: Path, frames: int = 8) -> None:
    dofs = 29
    bodies = 30
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fps=np.array([30], dtype=np.int64),
        joint_pos=np.zeros((frames, dofs), dtype=np.float32),
        joint_vel=np.zeros((frames, dofs), dtype=np.float32),
        body_pos_w=np.zeros((frames, bodies, 3), dtype=np.float32),
        body_quat_w=np.zeros((frames, bodies, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
    )


def _segment() -> FrontRESSegmentIndex:
    return FrontRESSegmentIndex(
        segment_id=5,
        motion_rel_path="KIT/359/motion_a.npz",
        motion_num_frames=8,
        fps=30.0,
        start_frame=3,
        horizon_k=2,
    )


def _clean_state() -> FrontRESRobotRolloutState:
    batch = 1
    dofs = 29
    bodies = 30
    return FrontRESRobotRolloutState(
        root_pos=torch.tensor([[9.0, 8.0, 1.5]], dtype=torch.float32),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        root_lin_vel=torch.zeros(batch, 3),
        root_ang_vel=torch.zeros(batch, 3),
        joint_pos=torch.full((batch, dofs), 0.4),
        joint_vel=torch.full((batch, dofs), 0.05),
        body_pos_w=torch.zeros(batch, bodies, 3),
        body_quat_w=torch.zeros(batch, bodies, 4).index_fill(2, torch.tensor([0]), 1.0),
        body_lin_vel_w=torch.zeros(batch, bodies, 3),
        body_ang_vel_w=torch.zeros(batch, bodies, 3),
    )


def _descriptor() -> FrontRESPerturbationDescriptor:
    return FrontRESPerturbationDescriptor(
        perturbation_id=11,
        segment_id=5,
        strength=0.5,
        seed=123,
        family="external_push",
        start_step=0,
        duration=2,
        target="root",
        frame="world",
        params={"axis": [1.0, 0.0, 0.0], "signed_magnitude": 0.5},
    )


def test_stage1_hook_trace_summarizes_large_sequences() -> None:
    adapter = FrontRESStage1EnvAdapter(FakeGymEnv(Path("/tmp/fake_amass")), amass_root="/tmp/fake_amass", trace=False)
    int_summary = adapter._format_trace_value(list(range(12)))
    str_summary = adapter._format_trace_value([f"motion_{idx}" for idx in range(12)])
    print(
        "[stage1_hooks trace] large_sequence_summary "
        f"int_summary={int_summary} "
        f"str_summary={str_summary}",
        flush=True,
    )
    assert int_summary == {"count": 12, "first": 0, "last": 11, "min": 0, "max": 11}
    assert str_summary["count"] == 12
    assert str_summary["first"] == "motion_0"
    assert str_summary["last"] == "motion_11"
    assert str_summary["unique_count"] == 12


def test_stage1_env_adapter_hooks_trace_real_boundary_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        _write_fake_amass(root / "KIT" / "359" / "motion_a.npz")
        env = FakeGymEnv(root)
        adapter = FrontRESStage1EnvAdapter(env, amass_root=str(root), trace=True)
        loaded_paths = adapter.frontres_loaded_motion_paths()
        loader_probe = adapter.frontres_motion_loader_probe()
        print(f"[stage1_hooks trace] loaded_motion_paths={loaded_paths} loader_probe={loader_probe}")
        assert loaded_paths == [str(root / "KIT" / "359" / "motion_a.npz")]
        assert loader_probe["loaded_motion_count"] == 1
        assert loader_probe["all_motion_count"] == 1
        assert loader_probe["cfg_motion_dataset_load_cap"] == 1
        assert loader_probe["cfg_motion_dataset_shard_across_gpus"] is False
        env_ids = torch.tensor([0], dtype=torch.long)

        prepare = adapter.prepare_frontres_clean_segment(segment=_segment(), env_ids=env_ids)
        print(
            "[stage1_hooks trace] after_prepare "
            f"success={prepare['success'].tolist()} "
            f"motion_idx={env.unwrapped.command.env_motion_indices.tolist()} "
            f"time_steps={env.unwrapped.command.time_steps.tolist()} "
            f"root_pos={env.unwrapped.robot.data.root_pos_w.tolist()} "
            f"joint_mean={float(env.unwrapped.robot.data.joint_pos.mean().item()):.4f}"
        )
        assert prepare["success"].tolist() == [True]
        assert env.unwrapped.command.env_motion_indices.tolist() == [0]
        assert env.unwrapped.command.time_steps.tolist() == [3]
        torch.testing.assert_close(env.unwrapped.robot.data.root_pos_w, torch.tensor([[3.0, 0.0, 1.0]]))
        assert env.unwrapped.command.perturber.reset_calls == [[0]]

        index_reset = adapter.apply_frontres_segment_index_reset(
            types.SimpleNamespace(
                segment_ids=torch.tensor([5], dtype=torch.long),
                motion_ids=("KIT/359/motion_a.npz",),
                start_frames=torch.tensor([4], dtype=torch.long),
                horizon_k=torch.tensor([2], dtype=torch.long),
            )
        )
        print(
            "[stage1_hooks trace] index_reset "
            f"success={index_reset['reset_success'].tolist()} "
            f"motion_idx={env.unwrapped.command.env_motion_indices.tolist()} "
            f"time_steps={env.unwrapped.command.time_steps.tolist()} "
            f"root_pos={env.unwrapped.robot.data.root_pos_w.tolist()} "
            f"joint_head={env.unwrapped.robot.data.joint_pos[:, :3].tolist()}"
        )
        assert index_reset["reset_success"].tolist() == [True]
        assert env.unwrapped.command.env_motion_indices.tolist() == [0]
        assert env.unwrapped.command.time_steps.tolist() == [4]
        torch.testing.assert_close(env.unwrapped.robot.data.root_pos_w, torch.tensor([[4.0, 0.0, 1.0]]))
        torch.testing.assert_close(env.unwrapped.robot.data.joint_pos[:, :3], torch.tensor([[4.0, 5.0, 6.0]]))

        reset = adapter.set_frontres_rollout_state(clean_state=_clean_state(), env_ids=env_ids)
        print(
            "[stage1_hooks trace] after_reset "
            f"success={reset['success'].tolist()} "
            f"root_pos={env.unwrapped.robot.data.root_pos_w.tolist()} "
            f"joint_mean={float(env.unwrapped.robot.data.joint_pos.mean().item()):.4f}"
        )
        assert reset["success"].tolist() == [True]
        torch.testing.assert_close(env.unwrapped.robot.data.root_pos_w, torch.tensor([[9.0, 8.0, 1.5]]))
        torch.testing.assert_close(env.unwrapped.robot.data.joint_pos.mean(), torch.tensor(0.4))

        perturb = adapter.apply_frontres_segment_perturbation(descriptor=_descriptor(), env_ids=env_ids)
        print(
            "[stage1_hooks trace] after_perturb "
            f"success={perturb['success'].tolist()} "
            f"root_pos={env.unwrapped.robot.data.root_pos_w.tolist()} "
            f"root_lin_vel={env.unwrapped.robot.data.root_lin_vel_w.tolist()}"
        )
        assert perturb["success"].tolist() == [True]
        torch.testing.assert_close(env.unwrapped.robot.data.root_pos_w, torch.tensor([[9.5, 8.0, 1.5]]))
        torch.testing.assert_close(env.unwrapped.robot.data.root_lin_vel_w, torch.tensor([[0.05, 0.0, 0.0]]))

        baseline = adapter.rollout_frontres_noisy_baseline(segment=_segment(), descriptor=_descriptor(), env_ids=env_ids)
        print(
            "[stage1_hooks trace] baseline "
            f"actions={len(env.step_actions)} "
            f"action_shape={tuple(env.step_actions[0].shape)} "
            f"reset_count={env.reset_count} "
            f"score={baseline['score'].tolist()} "
            f"fall={baseline['fall'].tolist()} "
            f"rollout_len={baseline['rollout_len'].tolist()}"
        )
        assert env.reset_count == 1
        assert len(env.step_actions) == 2
        assert tuple(env.step_actions[0].shape) == (1, 29)
        torch.testing.assert_close(baseline["score"], torch.tensor([0.75]))
        torch.testing.assert_close(baseline["fall"], torch.tensor([0.0]))
        torch.testing.assert_close(baseline["rollout_len"], torch.tensor([2.0]))

        cache_dir = Path(tmp) / "cache"
        connected_env = FakeGymEnv(root)
        connected_adapter = FrontRESStage1EnvAdapter(connected_env, amass_root=str(root), trace=True)
        result = build_stage1_segment_cache(
            connected_adapter,
            FrontRESStage1CacheBuilderConfig(
                amass_root=str(root),
                cache_dir=str(cache_dir),
                horizon_k=2,
                frame_stride=1,
                max_motions=1,
                max_segments=1,
                strengths=(0.0, 0.5),
                variants_per_strength=1,
                perturbation_curriculum_mode="discrete_bank",
                base_seed=123,
                env_id=0,
            ),
        )
        clean_entries = cache_io.read_clean_state_shard(result.clean_shard_path)
        noisy_zero = cache_io.read_noisy_variant_shard(result.noisy_shard_paths[0.0])
        noisy_half = cache_io.read_noisy_variant_shard(result.noisy_shard_paths[0.5])
        print(
            "[stage1_hooks trace] builder_adapter_connector "
            f"segment_count={result.segment_count} "
            f"clean_count={result.clean_count} "
            f"noisy_count={result.noisy_count} "
            f"reset_count={connected_env.reset_count} "
            f"clean_ids={[entry.segment_id for entry in clean_entries]} "
            f"zero_ids={[(item.segment_id, item.perturbation_id) for item in noisy_zero]} "
            f"half_ids={[(item.segment_id, item.perturbation_id) for item in noisy_half]} "
            f"half_root_pos={[item.noisy_state.root_pos.flatten().tolist() for item in noisy_half]}"
        )
        assert result.segment_count == 1
        assert result.clean_count == 1
        assert result.noisy_count == 2
        assert connected_env.reset_count == 1
        assert [entry.segment_id for entry in clean_entries] == [0]
        assert [(item.segment_id, item.perturbation_id) for item in noisy_zero] == [(0, 0)]
        assert [(item.segment_id, item.perturbation_id) for item in noisy_half] == [(0, 1)]
        assert connected_env.step_actions and tuple(connected_env.step_actions[0].shape) == (1, 29)


def test_stage1_env_adapter_writes_inference_tensors_under_inference_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        _write_fake_amass(root / "KIT" / "359" / "motion_a.npz")
        env = FakeGymEnv(root)
        _make_stage1_fake_state_inference(env)
        adapter = FrontRESStage1EnvAdapter(env, amass_root=str(root), trace=False)
        env_ids = torch.tensor([0], dtype=torch.long)

        prepare = adapter.prepare_frontres_clean_segment(segment=_segment(), env_ids=env_ids)
        index_reset = adapter.apply_frontres_segment_index_reset(
            types.SimpleNamespace(
                segment_ids=torch.tensor([5], dtype=torch.long),
                motion_ids=("KIT/359/motion_a.npz",),
                start_frames=torch.tensor([4], dtype=torch.long),
                horizon_k=torch.tensor([2], dtype=torch.long),
            )
        )
        reset = adapter.set_frontres_rollout_state(clean_state=_clean_state(), env_ids=env_ids)
        perturb = adapter.apply_frontres_segment_perturbation(descriptor=_descriptor(), env_ids=env_ids)
        print(
            "[stage1_hooks trace] inference_tensor_write_boundary "
            f"prepare={prepare['success'].tolist()} "
            f"index={index_reset['reset_success'].tolist()} "
            f"reset={reset['success'].tolist()} "
            f"perturb={perturb['success'].tolist()} "
            f"root_pos={env.unwrapped.robot.data.root_pos_w.tolist()} "
            f"quat_corr={env.unwrapped.command._frontres_quat_correction.tolist()}",
            flush=True,
        )
        assert prepare["success"].tolist() == [True]
        assert index_reset["reset_success"].tolist() == [True]
        assert reset["success"].tolist() == [True]
        assert perturb["success"].tolist() == [True]
        torch.testing.assert_close(env.unwrapped.robot.data.root_pos_w, torch.tensor([[9.5, 8.0, 1.5]]))
        torch.testing.assert_close(
            env.unwrapped.command._frontres_quat_correction,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )


def test_stage1_index_reset_applies_dynamic_motion_perturbation_request() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        _write_fake_amass(root / "KIT" / "359" / "motion_a.npz")
        env = FakeGymEnv(root)
        adapter = FrontRESStage1EnvAdapter(env, amass_root=str(root), trace=False)

        result = adapter.apply_frontres_segment_index_reset(
            types.SimpleNamespace(
                segment_ids=torch.tensor([5], dtype=torch.long),
                motion_ids=("KIT/359/motion_a.npz",),
                start_frames=torch.tensor([4], dtype=torch.long),
                horizon_k=torch.tensor([2], dtype=torch.long),
                perturbation_family=("planar+yaw",),
                perturbation_strength=torch.tensor([2.0], dtype=torch.float32),
            )
        )
        perturber = env.unwrapped.command.perturber
        assert result["reset_success"].tolist() == [True]
        assert perturber.dr_scale_env is not None
        torch.testing.assert_close(perturber.dr_scale_env, torch.tensor([2.0]))
        assert perturber.family_masks is not None
        assert perturber.family_masks["planar"].tolist() == [True]
        assert perturber.family_masks["yaw"].tolist() == [True]
        assert perturber.family_masks["global_z"].tolist() == [False]
        assert perturber.family_masks["local_rp"].tolist() == [False]
        print(
            "[probe step4] index_reset_applies_dynamic_perturbation "
            f"family=('planar+yaw',) "
            f"strength={perturber.dr_scale_env.tolist()} "
            f"planar_mask={perturber.family_masks['planar'].tolist()} "
            f"yaw_mask={perturber.family_masks['yaw'].tolist()}",
            flush=True,
        )


def test_stage1_index_reset_applies_local_rp_only_perturbation_request() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        _write_fake_amass(root / "KIT" / "359" / "motion_a.npz")
        env = FakeGymEnv(root)
        adapter = FrontRESStage1EnvAdapter(env, amass_root=str(root), trace=False)

        result = adapter.apply_frontres_segment_index_reset(
            types.SimpleNamespace(
                segment_ids=torch.tensor([5], dtype=torch.long),
                motion_ids=("KIT/359/motion_a.npz",),
                start_frames=torch.tensor([4], dtype=torch.long),
                horizon_k=torch.tensor([2], dtype=torch.long),
                perturbation_family=("local_rp",),
                perturbation_strength=torch.tensor([1.25], dtype=torch.float32),
            )
        )
        perturber = env.unwrapped.command.perturber
        assert result["reset_success"].tolist() == [True]
        assert perturber.dr_scale_env is not None
        torch.testing.assert_close(perturber.dr_scale_env, torch.tensor([1.25]))
        assert perturber.family_masks is not None
        assert perturber.family_masks["planar"].tolist() == [False]
        assert perturber.family_masks["yaw"].tolist() == [False]
        assert perturber.family_masks["global_z"].tolist() == [False]
        assert perturber.family_masks["local_rp"].tolist() == [True]
        print(
            "[probe step4] index_reset_applies_local_rp_only_perturbation "
            f"family=('local_rp',) "
            f"strength={perturber.dr_scale_env.tolist()} "
            f"planar_mask={perturber.family_masks['planar'].tolist()} "
            f"yaw_mask={perturber.family_masks['yaw'].tolist()} "
            f"global_z_mask={perturber.family_masks['global_z'].tolist()} "
            f"local_rp_mask={perturber.family_masks['local_rp'].tolist()}",
            flush=True,
        )


def test_index_reset_expands_sampled_rows_to_full_quartet_dynamic_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        _write_fake_amass(root / "KIT" / "359" / "motion_a.npz")
        env = FakeGymEnv(root, num_envs=8)
        env.unwrapped.scene.env_origins[:, 0] = torch.arange(8, dtype=torch.float32) * 10.0
        env.unwrapped.robot.data.joint_pos[:] = torch.arange(8, dtype=torch.float32).view(8, 1) * 100.0
        adapter = FrontRESStage1EnvAdapter(env, amass_root=str(root), trace=False)
        role_env_ids = {
            "policy": torch.tensor([0, 1], dtype=torch.long),
            "candidate": torch.tensor([2, 3], dtype=torch.long),
            "noisy": torch.tensor([4, 5], dtype=torch.long),
            "clean": torch.tensor([6, 7], dtype=torch.long),
        }

        result = adapter.apply_frontres_segment_index_reset(
            types.SimpleNamespace(
                segment_ids=torch.tensor([5, 6], dtype=torch.long),
                motion_ids=("KIT/359/motion_a.npz", "KIT/359/motion_a.npz"),
                start_frames=torch.tensor([3, 4], dtype=torch.long),
                horizon_k=torch.tensor([2, 2], dtype=torch.long),
                perturbation_family=("local_rp", "local_rp"),
                perturbation_strength=torch.tensor([0.5, 1.0], dtype=torch.float32),
                frontres_role_env_ids=role_env_ids,
            )
        )

        expected_frames = torch.tensor([3, 4, 3, 4, 3, 4, 3, 4], dtype=torch.long)
        expected_joint_head = torch.tensor([3, 4, 3, 4, 3, 4, 3, 4], dtype=torch.float32)
        local_root = env.unwrapped.robot.data.root_pos_w - env.unwrapped.scene.env_origins
        assert result["reset_success"].tolist() == [True, True]
        torch.testing.assert_close(env.unwrapped.command.time_steps, expected_frames)
        assert env.unwrapped.command.cache_refresh_calls == 1
        torch.testing.assert_close(
            env.unwrapped.command._cached_perturbed_pos[:, 2],
            torch.ones(8, dtype=torch.float32),
        )
        assert env.unwrapped.command.env_motion_groups.tolist() == [0] * 8
        torch.testing.assert_close(env.unwrapped.robot.data.joint_pos[:, 0], expected_joint_head)
        torch.testing.assert_close(local_root[:, 0], expected_joint_head)
        assert env.unwrapped.episode_length_buf.tolist() == [0] * 8
        torch.testing.assert_close(
            env.unwrapped.command.perturber.dr_scale_env,
            torch.tensor([0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        assert env.unwrapped.command.perturber.family_masks["local_rp"].tolist() == [
            True, True, False, False, False, False, False, False
        ]
        print(
            "[probe quartet_reset] "
            f"frames={env.unwrapped.command.time_steps.tolist()} "
            f"joint0={env.unwrapped.robot.data.joint_pos[:, 0].tolist()} "
            f"local_root_x={local_root[:, 0].tolist()} "
            f"episode={env.unwrapped.episode_length_buf.tolist()} "
            f"cache_z={env.unwrapped.command._cached_perturbed_pos[:, 2].tolist()} "
            f"cache_refresh_calls={env.unwrapped.command.cache_refresh_calls} "
            f"dr_scale={env.unwrapped.command.perturber.dr_scale_env.tolist()}",
            flush=True,
        )


def test_index_reset_installs_fixed_noisy_tape_for_all_roles_without_resampling() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        _write_fake_amass(root / "KIT" / "359" / "motion_a.npz")
        env = FakeGymEnv(root, num_envs=8)
        adapter = FrontRESStage1EnvAdapter(env, amass_root=str(root), trace=False)
        role_env_ids = {
            "policy": torch.tensor([0, 1], dtype=torch.long),
            "candidate": torch.tensor([2, 3], dtype=torch.long),
            "noisy": torch.tensor([4, 5], dtype=torch.long),
            "clean": torch.tensor([6, 7], dtype=torch.long),
        }
        source_tape = torch.zeros(2, 4, 65, dtype=torch.float32)
        source_tape[0, :, :58] = 10.0
        source_tape[1, :, :58] = 20.0
        source_tape[0, :, 58:61] = torch.tensor([101.0, 102.0, 103.0])
        source_tape[1, :, 58:61] = torch.tensor([201.0, 202.0, 203.0])
        source_tape[:, :, 61] = 1.0

        result = adapter.apply_frontres_segment_index_reset(
            types.SimpleNamespace(
                segment_ids=torch.tensor([5, 6], dtype=torch.long),
                motion_ids=("KIT/359/motion_a.npz", "KIT/359/motion_a.npz"),
                start_frames=torch.tensor([3, 4], dtype=torch.long),
                horizon_k=torch.tensor([2, 2], dtype=torch.long),
                frontres_fixed_noisy_tape=source_tape,
                frontres_fixed_noisy_tape_lengths=torch.tensor([4, 4], dtype=torch.long),
                frontres_fixed_noisy_scenario_ids=("scenario-a", "scenario-b"),
                frontres_fixed_noisy_segment_hashes=("hash-a", "hash-b"),
                frontres_future_offsets=(1, 2),
                frontres_role_env_ids=role_env_ids,
            )
        )

        command = env.unwrapped.command
        assert result["reset_success"].tolist() == [True, True]
        assert len(command.fixed_tape_install_calls) == 1
        installed = command.fixed_tape_install_calls[0]
        assert tuple(installed["tape"].shape) == (8, 4, 65)
        assert installed["execution_mask"].tolist() == [True, True, True, True, True, True, False, False]
        assert installed["noisy_segment_hashes"] == (
            "hash-a", "hash-b", "hash-a", "hash-b", "hash-a", "hash-b", "hash-a", "hash-b"
        )
        assert env.unwrapped.command.perturber.reset_calls == []
        torch.testing.assert_close(command._cached_perturbed_pos[:6, 0], torch.tensor([101.0, 201.0, 101.0, 201.0, 101.0, 201.0]))
        torch.testing.assert_close(command._cached_perturbed_pos[6:, 0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(env.unwrapped.robot.data.joint_pos[:, 0], torch.tensor([3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]))
        print(
            "[probe fixed_tape_reset] "
            f"installed_shape={tuple(installed['tape'].shape)} "
            f"execution_mask={installed['execution_mask'].tolist()} "
            f"hashes={installed['noisy_segment_hashes']} "
            f"perturber_reset_calls={env.unwrapped.command.perturber.reset_calls}",
            flush=True,
        )


def test_adapter_routes_selection_time_materialization_to_command_owner() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        _write_fake_amass(root / "KIT" / "359" / "motion_a.npz")
        env = FakeGymEnv(root)
        adapter = FrontRESStage1EnvAdapter(env, amass_root=str(root), trace=False)

        tape = adapter.materialize_frontres_fixed_noisy_tape(
            motion_id="KIT/359/motion_a.npz",
            start_frame=3,
            frame_count=4,
            perturbation_family="local_rp",
            perturbation_strength=0.5,
        )

        assert tuple(tape.shape) == (4, 65)
        torch.testing.assert_close(tape, torch.full((4, 65), 3.5))
        assert env.unwrapped.command.materialize_calls == [
            {
                "motion_index": 0,
                "start_frame": 3,
                "frame_count": 4,
                "perturbation_family": "local_rp",
                "perturbation_strength": 0.5,
            }
        ]


def test_production_cache_refresh_owner_does_not_advance_frame() -> None:
    source = COMMANDS_PATH.read_text()
    module = ast.parse(source)
    multi_motion = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MultiMotionCommand"
    )
    functions = {
        node.name: node
        for node in multi_motion.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    refresh = ast.get_source_segment(source, functions["refresh_frontres_reference_cache_current_frame"])
    clock = ast.get_source_segment(source, functions["_advance_frontres_command_clock"])
    update = ast.get_source_segment(source, functions["_update_command"])
    sync_pairs = ast.get_source_segment(source, functions["_sync_frontres_pairs"])
    assert refresh is not None and clock is not None and update is not None and sync_pairs is not None
    assert "time_steps +=" not in refresh
    assert refresh.count("apply_perturbations(") == 1
    assert refresh.count("apply_quat_perturbation(") == 1
    assert refresh.count("_sync_frontres_pairs(sync_perturbation=True)") == 1
    assert "self._cached_perturbed_pos[base_ids] = self._cached_perturbed_pos[train_ids]" in sync_pairs
    assert "self._cached_perturbed_quat[base_ids] = self._cached_perturbed_quat[train_ids]" in sync_pairs
    assert "self._dr_supervised_target[base_ids] = self._dr_supervised_target[train_ids]" in sync_pairs
    assert "self._cached_perturbed_pos[clean_ids] = pos_data" in sync_pairs
    assert "self._dr_supervised_target[clean_ids] = 0.0" in sync_pairs
    assert update.count("self._advance_frontres_command_clock()") == 1
    assert "self.time_steps += 1" not in update
    assert "refresh_frontres_reference_cache_current_frame()" not in update
    assert clock.count("refresh_frontres_reference_cache_current_frame()") == 1
    assert clock.index("self.time_steps += 1") < clock.index("refresh_frontres_reference_cache_current_frame()")
    assert '"local_current_hold"' in clock and '"local_k_hold"' in clock
    assert 'return "legacy_advance"' in clock
    print(
        "[probe quartet_cache_owner] frame_advance_in_refresh=0 "
        "position_draws=1 quaternion_draws=1 pair_sync=1 clock_owner_calls=1",
        flush=True,
    )


if __name__ == "__main__":
    test_stage1_hook_trace_summarizes_large_sequences()
    test_stage1_env_adapter_hooks_trace_real_boundary_contract()
    test_stage1_env_adapter_writes_inference_tensors_under_inference_mode()
    test_stage1_index_reset_applies_dynamic_motion_perturbation_request()
    test_stage1_index_reset_applies_local_rp_only_perturbation_request()
    test_index_reset_expands_sampled_rows_to_full_quartet_dynamic_state()
    test_index_reset_installs_fixed_noisy_tape_for_all_roles_without_resampling()
    test_adapter_routes_selection_time_materialization_to_command_owner()
    test_production_cache_refresh_owner_does_not_advance_frame()
    print("PASS: FrontRES Stage 1 env adapter hooks trace motion, clean reset, perturbation, and baseline rollout.")
