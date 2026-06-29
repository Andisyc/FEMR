#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module(
    "frontres_segment_cache_builder",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_builder.py",
)
cache_io = _load_module(
    "frontres_segment_cache_io_for_builder_contract",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py",
)

FrontRESStage1CacheBuilderConfig = builder.FrontRESStage1CacheBuilderConfig
build_stage1_segment_cache = builder.build_stage1_segment_cache


class FakeRobotData:
    def __init__(self, num_envs: int = 1, dofs: int = 29, bodies: int = 30) -> None:
        self.num_envs = num_envs
        self.dofs = dofs
        self.bodies = bodies
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


class FakeScene:
    def __init__(self) -> None:
        self.robot = FakeRobot()

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot


class FakeStage1Env:
    def __init__(self, loaded_motion_paths: list[str] | None = None) -> None:
        self.scene = FakeScene()
        self.unwrapped = self
        self.loaded_motion_paths = list(loaded_motion_paths or [])
        self.prepare_calls: list[tuple[int, int, list[int]]] = []
        self.reset_calls: list[list[int]] = []
        self.perturb_calls: list[tuple[int, int, float, list[int]]] = []
        self.baseline_calls: list[tuple[int, int, list[int]]] = []

    @property
    def data(self) -> FakeRobotData:
        return self.scene.robot.data

    def frontres_loaded_motion_paths(self) -> list[str]:
        return list(self.loaded_motion_paths)

    def prepare_frontres_clean_segment(self, *, segment, env_ids: torch.Tensor):
        ids = env_ids.detach().cpu().tolist()
        self.prepare_calls.append((int(segment.segment_id), int(segment.start_frame), ids))
        for env_id in ids:
            clean_root = torch.tensor(
                [
                    float(segment.segment_id),
                    float(segment.start_frame),
                    float(segment.horizon_k),
                ],
                dtype=torch.float32,
            )
            self.data.root_pos_w[env_id] = clean_root
            self.data.root_lin_vel_w[env_id] = torch.tensor(
                [
                    0.1 * float(segment.segment_id + 1),
                    0.01 * float(segment.start_frame + 1),
                    0.0,
                ],
                dtype=torch.float32,
            )
            self.data.root_ang_vel_w[env_id] = 0.05 * float(segment.segment_id + 1)
            self.data.joint_pos[env_id] = torch.arange(self.data.dofs, dtype=torch.float32) + float(segment.start_frame)
            self.data.joint_vel[env_id] = 0.01 * torch.arange(self.data.dofs, dtype=torch.float32)
            self.data.body_pos_w[env_id] = clean_root.view(1, 3) + 0.001 * torch.arange(
                self.data.bodies, dtype=torch.float32
            ).view(-1, 1)
            self.data.body_lin_vel_w[env_id] = self.data.root_lin_vel_w[env_id].view(1, 3)
            self.data.body_ang_vel_w[env_id] = self.data.root_ang_vel_w[env_id].view(1, 3)
        return {"success": torch.ones(env_ids.numel(), dtype=torch.bool)}

    def set_frontres_rollout_state(self, *, clean_state, env_ids: torch.Tensor):
        ids = env_ids.detach().cpu().tolist()
        self.reset_calls.append(ids)
        for local_i, env_id in enumerate(ids):
            self.data.root_pos_w[env_id] = clean_state.root_pos[local_i]
            self.data.root_quat_w[env_id] = clean_state.root_quat[local_i]
            self.data.root_lin_vel_w[env_id] = clean_state.root_lin_vel[local_i]
            self.data.root_ang_vel_w[env_id] = clean_state.root_ang_vel[local_i]
            self.data.joint_pos[env_id] = clean_state.joint_pos[local_i]
            self.data.joint_vel[env_id] = clean_state.joint_vel[local_i]
            self.data.body_pos_w[env_id] = clean_state.body_pos_w[local_i]
            self.data.body_quat_w[env_id] = clean_state.body_quat_w[local_i]
            self.data.body_lin_vel_w[env_id] = clean_state.body_lin_vel_w[local_i]
            self.data.body_ang_vel_w[env_id] = clean_state.body_ang_vel_w[local_i]
        return {"success": torch.ones(env_ids.numel(), dtype=torch.bool)}

    def apply_frontres_segment_perturbation(self, *, descriptor, env_ids: torch.Tensor):
        ids = env_ids.detach().cpu().tolist()
        axis = torch.tensor(descriptor.params["axis"], dtype=torch.float32)
        signed_magnitude = float(descriptor.params["signed_magnitude"])
        delta = axis * signed_magnitude
        self.perturb_calls.append(
            (int(descriptor.segment_id), int(descriptor.perturbation_id), float(descriptor.strength), ids)
        )
        for env_id in ids:
            self.data.root_pos_w[env_id] += delta
            self.data.root_lin_vel_w[env_id] += 0.1 * delta
            self.data.body_pos_w[env_id, 0] += delta
        return torch.ones(env_ids.numel(), dtype=torch.bool)

    def rollout_frontres_noisy_baseline(self, *, segment, descriptor, env_ids: torch.Tensor):
        ids = env_ids.detach().cpu().tolist()
        self.baseline_calls.append((int(segment.segment_id), int(descriptor.perturbation_id), ids))
        score = torch.empty(env_ids.numel(), dtype=torch.float32)
        fall = torch.zeros(env_ids.numel(), dtype=torch.float32)
        rollout_len = torch.full((env_ids.numel(),), float(segment.horizon_k), dtype=torch.float32)
        for local_i, env_id in enumerate(ids):
            score[local_i] = 1.0 - self.data.root_lin_vel_w[env_id].norm()
            fall[local_i] = float(self.data.root_pos_w[env_id, 2] < 0.2)
        return {"score": score, "fall": fall, "rollout_len": rollout_len}


def _write_fake_motion(path: Path, frames: int = 5) -> None:
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


def test_stage1_builder_orchestrates_cache_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        amass_root = tmp_path / "AMASS_G1NPZ_Final"
        cache_dir = tmp_path / "frontres_stage1_cache"
        _write_fake_motion(amass_root / "KIT" / "359" / "motion_a.npz", frames=5)

        env = FakeStage1Env()
        cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=2,
            frame_stride=2,
            max_motions=1,
            max_segments=2,
            strengths=(0.0, 0.5),
            variants_per_strength=1,
            base_seed=123,
            env_id=0,
        )

        result = build_stage1_segment_cache(env, cfg)
        print(
            "[cache_builder trace] result "
            f"segment_count={result.segment_count} "
            f"clean_count={result.clean_count} "
            f"noisy_count={result.noisy_count} "
            f"strength_counts={result.strength_counts} "
            f"segment_index_exists={Path(result.segment_index_path).exists()} "
            f"clean_shard_exists={Path(result.clean_shard_path).exists()} "
            f"noisy_shard_paths={result.noisy_shard_paths}"
        )

        assert result.segment_count == 2
        assert result.clean_count == 2
        assert result.noisy_count == 4
        assert result.strength_counts == {0.0: 2, 0.5: 2}
        assert Path(result.segment_index_path).exists()
        assert Path(result.clean_shard_path).exists()
        assert set(result.noisy_shard_paths) == {0.0, 0.5}
        assert all(Path(path).exists() for path in result.noisy_shard_paths.values())
        assert Path(result.clean_shard_path).relative_to(cache_dir).as_posix() == "manifests/clean_states/shard_000000.pt"
        assert (
            cache_dir / "KIT" / "359" / "motion_a" / "segment_00000000_start_00000000_k_0002" / "clean.pt"
        ).exists()
        assert (
            cache_dir
            / "KIT"
            / "359"
            / "motion_a"
            / "segment_00000000_start_00000000_k_0002"
            / "noisy_variants"
            / "strength_0p5"
            / "perturbation_00000001.pt"
        ).exists()

        print(
            "[cache_builder trace] hooks "
            f"prepare_calls={env.prepare_calls} "
            f"reset_calls={env.reset_calls} "
            f"perturb_calls={env.perturb_calls} "
            f"baseline_calls={env.baseline_calls}"
        )
        assert env.prepare_calls == [(0, 0, [0]), (1, 2, [0])]
        assert len(env.reset_calls) == 4
        assert len(env.perturb_calls) == 4
        assert len(env.baseline_calls) == 4
        assert [call[0] for call in env.baseline_calls] == [0, 0, 1, 1]

        metadata = json.loads(Path(result.metadata_path).read_text())
        clean_entries = cache_io.read_clean_state_shard(result.clean_shard_path)
        noisy_zero = cache_io.read_noisy_variant_shard(result.noisy_shard_paths[0.0])
        noisy_half = cache_io.read_noisy_variant_shard(result.noisy_shard_paths[0.5])

        print(
            "[cache_builder trace] readback "
            f"metadata_format={metadata['format']} "
            f"clean_ids={[entry.segment_id for entry in clean_entries]} "
            f"clean_root_pos={[entry.clean_state.root_pos.flatten().tolist() for entry in clean_entries]} "
            f"zero_ids={[(item.segment_id, item.perturbation_id) for item in noisy_zero]} "
            f"half_ids={[(item.segment_id, item.perturbation_id) for item in noisy_half]} "
            f"half_root_pos={[item.noisy_state.root_pos.flatten().tolist() for item in noisy_half]} "
            f"half_score={[float(item.noisy_baseline_score.item()) for item in noisy_half]}"
        )

        assert metadata["format"] == "frontres_segment_cache_v1"
        assert metadata["stage"] == "stage1_segment_cache"
        assert metadata["segment_count"] == 2
        assert metadata["noisy_count"] == 4
        assert [entry.segment_id for entry in clean_entries] == [0, 1]
        torch.testing.assert_close(clean_entries[0].clean_state.root_pos, torch.tensor([[0.0, 0.0, 2.0]]))
        torch.testing.assert_close(clean_entries[1].clean_state.root_pos, torch.tensor([[1.0, 2.0, 2.0]]))
        assert [(item.segment_id, item.perturbation_id) for item in noisy_zero] == [(0, 0), (1, 2)]
        assert [(item.segment_id, item.perturbation_id) for item in noisy_half] == [(0, 1), (1, 3)]
        for item in noisy_zero + noisy_half:
            probe = item.probe()
            assert probe["noisy_state.finite"] is True
            assert probe["noisy_state.requires_grad"] is False
            assert probe["noisy_state.root_pos_shape"] == (1, 3)
            assert probe["noisy_state.body_pos_shape"] == (1, 30, 3)
            assert item.noisy_baseline_score.requires_grad is False


def test_stage1_builder_uses_loaded_motion_paths_before_disk_scan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        amass_root = tmp_path / "AMASS_G1NPZ_Final"
        cache_dir = tmp_path / "frontres_stage1_cache"
        motion_a = amass_root / "AAA" / "motion_a.npz"
        motion_b = amass_root / "ZZZ" / "motion_b.npz"
        _write_fake_motion(motion_a, frames=6)
        _write_fake_motion(motion_b, frames=7)

        env = FakeStage1Env(loaded_motion_paths=[str(motion_b)])
        cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=2,
            frame_stride=2,
            max_motions=1,
            max_segments=2,
            strengths=(0.0,),
            variants_per_strength=1,
            base_seed=123,
            env_id=0,
        )

        result = build_stage1_segment_cache(env, cfg)
        clean_entries = cache_io.read_clean_state_shard(result.clean_shard_path)
        print(
            "[cache_builder trace] loaded_motion_paths_override "
            f"loaded={motion_b.relative_to(amass_root)} "
            f"clean_paths={[entry.segment.motion_rel_path for entry in clean_entries]} "
            f"prepare_calls={env.prepare_calls}"
        )
        assert result.segment_count == 2
        assert [entry.segment.motion_rel_path for entry in clean_entries] == ["ZZZ/motion_b.npz", "ZZZ/motion_b.npz"]
        assert env.prepare_calls == [(0, 0, [0]), (1, 2, [0])]


if __name__ == "__main__":
    test_stage1_builder_orchestrates_cache_pipeline()
    test_stage1_builder_uses_loaded_motion_paths_before_disk_scan()
    print("PASS: FrontRES Stage 1 cache builder orchestrates index, clean, perturbation, noisy, and IO.")
