#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile
import types

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
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
    def __init__(self) -> None:
        self.data = FakeRobotData()
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

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        self.reset_calls.append(env_ids.detach().cpu().tolist())


class FakeCommand:
    def __init__(self, root: Path, robot: FakeRobot) -> None:
        self.device = torch.device("cpu")
        self.num_envs = 1
        self.robot = robot
        self.cfg = types.SimpleNamespace(
            motion_dataset_load_cap=1,
            motion_dataset_shard_across_gpus=False,
        )
        self.motion_dir_loader = FakeMotionLoader(root)
        self.motion_lengths = self.motion_dir_loader.motion_lengths
        self.env_motion_indices = torch.zeros(1, dtype=torch.long)
        self.time_steps = torch.zeros(1, dtype=torch.long)
        self.motion_end_buf = torch.zeros(1, dtype=torch.bool)
        self._frontres_pos_correction = torch.ones(1, 3)
        self._frontres_quat_correction = torch.zeros(1, 4)
        self.perturber = FakePerturber()
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


class FakeCommandManager:
    def __init__(self, command: FakeCommand) -> None:
        self.command = command

    def get_term(self, name: str):
        assert name == "motion"
        return self.command


class FakeScene:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot
        self.env_origins = torch.zeros(1, 3)

    def __getitem__(self, name: str):
        assert name == "robot"
        return self.robot


class FakeBaseEnv:
    def __init__(self, root: Path) -> None:
        self.num_envs = 1
        self.num_actions = 29
        self.robot = FakeRobot()
        self.scene = FakeScene(self.robot)
        self.command = FakeCommand(root, self.robot)
        self.command_manager = FakeCommandManager(self.command)


class FakeGymEnv:
    def __init__(self, root: Path) -> None:
        self.unwrapped = FakeBaseEnv(root)
        self.step_actions: list[torch.Tensor] = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        return None, {}

    def step(self, actions: torch.Tensor):
        if self.reset_count <= 0:
            raise RuntimeError("Cannot call env.step() before calling env.reset()")
        self.step_actions.append(actions.detach().clone())
        rewards = torch.full((1,), 0.75, dtype=torch.float32)
        dones = torch.zeros(1, dtype=torch.bool)
        return None, rewards, dones, {}


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


if __name__ == "__main__":
    test_stage1_env_adapter_hooks_trace_real_boundary_contract()
    print("PASS: FrontRES Stage 1 env adapter hooks trace motion, clean reset, perturbation, and baseline rollout.")
