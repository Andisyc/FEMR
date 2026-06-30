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


class FakeStage1CrashOnPrepareEnv(FakeStage1Env):
    def __init__(self, *, crash_at_segment_id: int) -> None:
        super().__init__()
        self.crash_at_segment_id = int(crash_at_segment_id)

    def prepare_frontres_clean_segment(self, *, segment, env_ids: torch.Tensor):
        if int(segment.segment_id) >= self.crash_at_segment_id:
            raise RuntimeError(f"intentional crash before segment {int(segment.segment_id)}")
        return super().prepare_frontres_clean_segment(segment=segment, env_ids=env_ids)


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


def _write_bad_motion_shape(path: Path) -> None:
    dofs = 29
    bodies = 30
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fps=np.array([30], dtype=np.int64),
        joint_pos=np.zeros((5, dofs), dtype=np.float32),
        joint_vel=np.zeros((4, dofs), dtype=np.float32),
        body_pos_w=np.zeros((5, bodies, 3), dtype=np.float32),
        body_quat_w=np.zeros((5, bodies, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((5, bodies, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((5, bodies, 3), dtype=np.float32),
    )


def _read_segment_index_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
            cache_chunk_size=2,
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
        assert (cache_dir / "shards" / "clean_states" / "shard_000000.pt").exists()
        assert (
            cache_dir
            / "shards"
            / "noisy_variants"
            / "strength_0p5"
            / "shard_000000.pt"
        ).exists()
        status = json.loads((cache_dir / "build_status.json").read_text())
        progress = [json.loads(line) for line in (cache_dir / "progress.jsonl").read_text().splitlines()]
        progress_events = [item["event"] for item in progress]
        complete_index = progress_events.index("complete")
        clean_flush_events = [
            (index, item)
            for index, item in enumerate(progress)
            if item["event"] == "clean_done" and item.get("flushed_shard_path")
        ]
        noisy_flush_events = [
            (index, item)
            for index, item in enumerate(progress)
            if item["event"] == "noisy_done" and item.get("flushed_shard_path")
        ]
        clean_flush_paths = [Path(item["flushed_shard_path"]) for _, item in clean_flush_events]
        noisy_flush_paths = [Path(item["flushed_shard_path"]) for _, item in noisy_flush_events]
        print(
            "[cache_builder trace] progress "
            f"status={status} "
            f"events={progress_events} "
            f"clean_flush_paths={[path.relative_to(cache_dir).as_posix() for path in clean_flush_paths]} "
            f"noisy_flush_paths={[path.relative_to(cache_dir).as_posix() for path in noisy_flush_paths]} "
            f"complete_index={complete_index}"
        )
        assert status["status"] == "complete"
        assert status["clean_written"] == 2
        assert status["noisy_written"] == 4
        assert progress_events[0] == "started"
        assert "indexed" in progress_events
        assert progress_events.count("clean_done") == 2
        assert progress_events.count("noisy_done") == 4
        assert clean_flush_events
        assert noisy_flush_events
        assert all(index < complete_index for index, _ in clean_flush_events)
        assert all(index < complete_index for index, _ in noisy_flush_events)
        assert all(path.exists() for path in clean_flush_paths)
        assert all(path.exists() for path in noisy_flush_paths)
        assert {path.relative_to(cache_dir).as_posix() for path in clean_flush_paths} == {
            "shards/clean_states/shard_000000.pt"
        }
        assert {path.relative_to(cache_dir).as_posix() for path in noisy_flush_paths} == {
            "shards/noisy_variants/strength_0/shard_000000.pt",
            "shards/noisy_variants/strength_0p5/shard_000000.pt",
        }
        assert Path(status["clean_shard_path"]).relative_to(cache_dir).as_posix() == "manifests/clean_states/shard_000000.pt"
        assert Path(status["metadata_path"]).exists()
        assert progress_events[-1] == "complete"

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
        assert metadata["perturbation_curriculum_mode"] == "discrete_bank"
        assert metadata["perturbation_levels"] == [
            {"level_index": 0, "level_name": "level_00", "strength": 0.0},
            {"level_index": 1, "level_name": "level_01", "strength": 0.5},
        ]
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


def test_stage1_builder_streaming_path_completes_without_eager_max_segments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        amass_root = tmp_path / "AMASS_G1NPZ_Final"
        cache_dir = tmp_path / "frontres_stage1_cache"
        motion_a = amass_root / "AAA" / "motion_a.npz"
        motion_b = amass_root / "BBB" / "motion_b.npz"
        _write_fake_motion(motion_a, frames=5)
        _write_fake_motion(motion_b, frames=5)

        env = FakeStage1Env(loaded_motion_paths=[str(motion_a), str(motion_b)])
        cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=2,
            frame_stride=2,
            max_segments=None,
            strengths=(0.0,),
            variants_per_strength=1,
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
            cache_chunk_size=2,
        )

        result = build_stage1_segment_cache(env, cfg)
        status = json.loads((cache_dir / "build_status.json").read_text())
        progress = [json.loads(line) for line in (cache_dir / "progress.jsonl").read_text().splitlines()]
        segment_records = _read_segment_index_records(cache_dir / "segment_index.jsonl")
        scan = cache_io.scan_stage1_cache_resume_state(cache_dir)
        print(
            "[cache_builder streaming trace] complete "
            f"result={result.probe()} "
            f"status={status} "
            f"index_ids={[item['segment_id'] for item in segment_records]} "
            f"events={[item['event'] for item in progress]} "
            f"resume_probe={scan.probe()} "
            f"prepare_calls={env.prepare_calls}"
        )

        assert result.segment_count == 4
        assert result.clean_count == 4
        assert result.noisy_count == 4
        assert result.strength_counts == {0.0: 4}
        assert status["indexing_mode"] == "streaming_chunk"
        assert [item["segment_id"] for item in segment_records] == [0, 1, 2, 3]
        assert [item["event"] for item in progress].count("index_chunk") == 2
        assert [item["event"] for item in progress].count("clean_done") == 4
        assert [item["event"] for item in progress].count("noisy_done") == 4
        assert (cache_dir / "shards" / "clean_states" / "shard_000000.pt").exists()
        assert (cache_dir / "shards" / "clean_states" / "shard_000001.pt").exists()
        assert (cache_dir / "shards" / "noisy_variants" / "strength_0" / "shard_000000.pt").exists()
        assert (cache_dir / "shards" / "noisy_variants" / "strength_0" / "shard_000001.pt").exists()
        assert scan.probe()["completed_clean"] == 4
        assert scan.probe()["completed_noisy"] == 4
        assert env.prepare_calls == [(0, 0, [0]), (1, 2, [0]), (2, 0, [0]), (3, 2, [0])]


def test_stage1_builder_streaming_commits_first_chunk_before_later_motion_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        amass_root = tmp_path / "AMASS_G1NPZ_Final"
        cache_dir = tmp_path / "frontres_stage1_cache"
        good = amass_root / "AAA" / "good_motion.npz"
        bad = amass_root / "BBB" / "bad_motion.npz"
        _write_fake_motion(good, frames=5)
        _write_bad_motion_shape(bad)

        env = FakeStage1Env(loaded_motion_paths=[str(good), str(bad)])
        cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=2,
            frame_stride=2,
            max_segments=None,
            strengths=(0.0,),
            variants_per_strength=1,
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
            cache_chunk_size=2,
        )

        try:
            build_stage1_segment_cache(env, cfg)
        except ValueError as exc:
            assert "joint_vel shape" in str(exc)
        else:
            raise AssertionError("expected later bad motion to fail after first chunk")

        progress = [json.loads(line) for line in (cache_dir / "progress.jsonl").read_text().splitlines()]
        segment_records = _read_segment_index_records(cache_dir / "segment_index.jsonl")
        scan = cache_io.scan_stage1_cache_resume_state(cache_dir)
        print(
            "[cache_builder streaming trace] partial_commit "
            f"index_ids={[item['segment_id'] for item in segment_records]} "
            f"events={[item['event'] for item in progress]} "
            f"resume_probe={scan.probe()} "
            f"prepare_calls={env.prepare_calls} "
            f"baseline_calls={env.baseline_calls}"
        )
        assert [item["segment_id"] for item in segment_records] == [0, 1]
        assert any(item["event"] == "index_chunk" for item in progress)
        assert [item["event"] for item in progress].count("clean_done") == 2
        assert [item["event"] for item in progress].count("noisy_done") == 2
        assert not any(item["event"] == "complete" for item in progress)
        assert (cache_dir / "shards" / "clean_states" / "shard_000000.pt").exists()
        assert (cache_dir / "shards" / "noisy_variants" / "strength_0" / "shard_000000.pt").exists()
        assert scan.probe()["completed_clean"] == 2
        assert scan.probe()["completed_noisy"] == 2
        assert env.prepare_calls == [(0, 0, [0]), (1, 2, [0])]
        assert [call[0] for call in env.baseline_calls] == [0, 1]


def test_stage1_builder_streaming_resume_skips_committed_chunk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        amass_root = tmp_path / "AMASS_G1NPZ_Final"
        cache_dir = tmp_path / "frontres_stage1_cache"
        good = amass_root / "AAA" / "good_motion.npz"
        bad = amass_root / "BBB" / "bad_motion.npz"
        _write_fake_motion(good, frames=5)
        _write_bad_motion_shape(bad)

        cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=2,
            frame_stride=2,
            max_segments=None,
            strengths=(0.0,),
            variants_per_strength=1,
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
            cache_chunk_size=2,
        )

        first_env = FakeStage1Env(loaded_motion_paths=[str(good), str(bad)])
        try:
            build_stage1_segment_cache(first_env, cfg)
        except ValueError as exc:
            assert "joint_vel shape" in str(exc)
        else:
            raise AssertionError("expected first run to fail on later bad motion")

        _write_fake_motion(bad, frames=5)
        resume_env = FakeStage1Env(loaded_motion_paths=[str(good), str(bad)])
        result = build_stage1_segment_cache(resume_env, cfg)
        progress = [json.loads(line) for line in (cache_dir / "progress.jsonl").read_text().splitlines()]
        segment_records = _read_segment_index_records(cache_dir / "segment_index.jsonl")
        scan = cache_io.scan_stage1_cache_resume_state(cache_dir)
        clean_entries = cache_io.read_clean_state_shard(result.clean_shard_path)
        noisy_zero = cache_io.read_noisy_variant_shard(result.noisy_shard_paths[0.0])
        resume_events = [item for item in progress if item["event"] in {"resume_skip_segment", "resume_reuse_clean"}]
        print(
            "[cache_builder streaming resume trace] "
            f"result={result.probe()} "
            f"index_ids={[item['segment_id'] for item in segment_records]} "
            f"resume_probe={scan.probe()} "
            f"resume_prepare_calls={resume_env.prepare_calls} "
            f"resume_baseline_calls={resume_env.baseline_calls} "
            f"clean_ids={[entry.segment_id for entry in clean_entries]} "
            f"noisy_ids={[(item.segment_id, item.perturbation_id) for item in noisy_zero]} "
            f"resume_events={resume_events}"
        )

        assert result.segment_count == 4
        assert result.clean_count == 4
        assert result.noisy_count == 4
        assert [item["segment_id"] for item in segment_records] == [0, 1, 2, 3]
        assert scan.probe()["completed_clean"] == 4
        assert scan.probe()["completed_noisy"] == 4
        assert resume_env.prepare_calls == [(2, 0, [0]), (3, 2, [0])]
        assert [call[0] for call in resume_env.baseline_calls] == [2, 3]
        assert [entry.segment_id for entry in clean_entries] == [0, 1, 2, 3]
        assert [(item.segment_id, item.perturbation_id) for item in noisy_zero] == [(0, 0), (1, 1), (2, 2), (3, 3)]
        assert [item for item in resume_events if item["event"] == "resume_skip_segment"]


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
            perturbation_curriculum_mode="discrete_bank",
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
        assert env.prepare_calls == [(0, 0, [0]), (1, 4, [0])]


def test_stage1_builder_resume_skips_committed_segments_and_preserves_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        amass_root = tmp_path / "AMASS_G1NPZ_Final"
        cache_dir = tmp_path / "frontres_stage1_cache"
        _write_fake_motion(amass_root / "KIT" / "359" / "motion_a.npz", frames=5)

        first_env = FakeStage1Env()
        first_cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=2,
            frame_stride=2,
            max_motions=1,
            max_segments=1,
            strengths=(0.0, 0.5),
            variants_per_strength=1,
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
            cache_chunk_size=1,
        )
        first = build_stage1_segment_cache(first_env, first_cfg)
        first_scan = cache_io.scan_stage1_cache_resume_state(cache_dir)
        assert first.clean_count == 1
        assert first.noisy_count == 2
        assert first_scan.probe()["completed_clean"] == 1
        assert first_scan.probe()["completed_noisy"] == 2

        second_env = FakeStage1Env()
        second_cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=2,
            frame_stride=2,
            max_motions=1,
            max_segments=2,
            strengths=(0.0, 0.5),
            variants_per_strength=1,
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
            cache_chunk_size=1,
        )
        second = build_stage1_segment_cache(second_env, second_cfg)
        clean_entries = cache_io.read_clean_state_shard(second.clean_shard_path)
        noisy_zero = cache_io.read_noisy_variant_shard(second.noisy_shard_paths[0.0])
        noisy_half = cache_io.read_noisy_variant_shard(second.noisy_shard_paths[0.5])
        second_scan = cache_io.scan_stage1_cache_resume_state(cache_dir)
        progress = [json.loads(line) for line in (cache_dir / "progress.jsonl").read_text().splitlines()]
        resume_events = [item for item in progress if item["event"] in {"resume_scan", "resume_skip_segment"}]

        print(
            "[cache_builder resume trace] "
            f"first_probe={first_scan.probe()} "
            f"second_probe={second_scan.probe()} "
            f"prepare_calls={second_env.prepare_calls} "
            f"reset_calls={second_env.reset_calls} "
            f"perturb_calls={second_env.perturb_calls} "
            f"baseline_calls={second_env.baseline_calls} "
            f"clean_ids={[entry.segment_id for entry in clean_entries]} "
            f"zero_ids={[(item.segment_id, item.perturbation_id) for item in noisy_zero]} "
            f"half_ids={[(item.segment_id, item.perturbation_id) for item in noisy_half]} "
            f"resume_events={resume_events[-3:]}"
        )

        assert second.segment_count == 2
        assert second.clean_count == 2
        assert second.noisy_count == 4
        assert second.strength_counts == {0.0: 2, 0.5: 2}
        assert second_env.prepare_calls == [(1, 2, [0])]
        assert len(second_env.reset_calls) == 2
        assert [call[0] for call in second_env.perturb_calls] == [1, 1]
        assert [call[0] for call in second_env.baseline_calls] == [1, 1]
        assert [entry.segment_id for entry in clean_entries] == [0, 1]
        assert [(item.segment_id, item.perturbation_id) for item in noisy_zero] == [(0, 0), (1, 2)]
        assert [(item.segment_id, item.perturbation_id) for item in noisy_half] == [(0, 1), (1, 3)]
        assert second_scan.probe()["completed_clean"] == 2
        assert second_scan.probe()["completed_noisy"] == 4
        assert second_scan.probe()["corrupt_count"] == 0
        assert any(item["event"] == "resume_skip_segment" and item["segment_id"] == 0 for item in progress)


def test_stage1_builder_resume_after_crash_uses_flush_committed_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        amass_root = tmp_path / "AMASS_G1NPZ_Final"
        cache_dir = tmp_path / "frontres_stage1_cache"
        _write_fake_motion(amass_root / "KIT" / "359" / "motion_a.npz", frames=5)

        crash_env = FakeStage1CrashOnPrepareEnv(crash_at_segment_id=1)
        crash_cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=2,
            frame_stride=2,
            max_motions=1,
            max_segments=2,
            strengths=(0.0, 0.5),
            variants_per_strength=1,
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
            cache_chunk_size=1,
        )

        try:
            build_stage1_segment_cache(crash_env, crash_cfg)
        except RuntimeError as exc:
            assert "intentional crash before segment 1" in str(exc)
        else:
            raise AssertionError("expected fake crash before segment 1")

        crash_scan = cache_io.scan_stage1_cache_resume_state(cache_dir)
        print(
            "[cache_builder crash_resume trace] after_crash "
            f"probe={crash_scan.probe()} "
            f"prepare_calls={crash_env.prepare_calls} "
            f"baseline_calls={crash_env.baseline_calls}"
        )
        assert crash_scan.probe()["completed_clean"] == 1
        assert crash_scan.probe()["completed_noisy"] == 2
        assert crash_scan.probe()["corrupt_count"] == 0
        assert crash_scan.probe()["clean_manifest_count"] == 1
        assert crash_scan.probe()["noisy_manifest_count"] == 2

        resume_env = FakeStage1Env()
        resumed = build_stage1_segment_cache(resume_env, crash_cfg)
        resumed_scan = cache_io.scan_stage1_cache_resume_state(cache_dir)
        clean_entries = cache_io.read_clean_state_shard(resumed.clean_shard_path)
        noisy_zero = cache_io.read_noisy_variant_shard(resumed.noisy_shard_paths[0.0])
        noisy_half = cache_io.read_noisy_variant_shard(resumed.noisy_shard_paths[0.5])
        progress = [json.loads(line) for line in (cache_dir / "progress.jsonl").read_text().splitlines()]

        print(
            "[cache_builder crash_resume trace] after_rerun "
            f"probe={resumed_scan.probe()} "
            f"prepare_calls={resume_env.prepare_calls} "
            f"reset_calls={resume_env.reset_calls} "
            f"baseline_calls={resume_env.baseline_calls} "
            f"clean_ids={[entry.segment_id for entry in clean_entries]} "
            f"zero_ids={[(item.segment_id, item.perturbation_id) for item in noisy_zero]} "
            f"half_ids={[(item.segment_id, item.perturbation_id) for item in noisy_half]}"
        )
        assert resumed.clean_count == 2
        assert resumed.noisy_count == 4
        assert resumed_scan.probe()["completed_clean"] == 2
        assert resumed_scan.probe()["completed_noisy"] == 4
        assert resume_env.prepare_calls == [(1, 2, [0])]
        assert len(resume_env.reset_calls) == 2
        assert [call[0] for call in resume_env.baseline_calls] == [1, 1]
        assert [entry.segment_id for entry in clean_entries] == [0, 1]
        assert [(item.segment_id, item.perturbation_id) for item in noisy_zero] == [(0, 0), (1, 2)]
        assert [(item.segment_id, item.perturbation_id) for item in noisy_half] == [(0, 1), (1, 3)]
        assert any(item["event"] == "resume_skip_segment" and item["segment_id"] == 0 for item in progress)


def test_stage1_builder_rejects_resume_signature_mismatch_before_env_work() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        amass_root = tmp_path / "AMASS_G1NPZ_Final"
        cache_dir = tmp_path / "frontres_stage1_cache"
        _write_fake_motion(amass_root / "KIT" / "359" / "motion_a.npz", frames=6)

        first_cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=2,
            frame_stride=2,
            max_motions=1,
            max_segments=1,
            strengths=(0.0, 0.5),
            variants_per_strength=1,
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
            cache_chunk_size=1,
        )
        build_stage1_segment_cache(FakeStage1Env(), first_cfg)
        original_index_text = (cache_dir / "segment_index.jsonl").read_text()

        mismatch_env = FakeStage1Env()
        mismatch_cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(cache_dir),
            horizon_k=3,
            frame_stride=2,
            max_motions=1,
            max_segments=1,
            strengths=(0.0, 0.5),
            variants_per_strength=1,
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
            cache_chunk_size=1,
        )

        try:
            build_stage1_segment_cache(mismatch_env, mismatch_cfg)
        except ValueError as exc:
            assert "signature mismatch" in str(exc)
        else:
            raise AssertionError("expected signature mismatch")

        progress = [json.loads(line) for line in (cache_dir / "progress.jsonl").read_text().splitlines()]
        status = json.loads((cache_dir / "build_status.json").read_text())
        print(
            "[cache_builder signature trace] "
            f"status={status['status']} "
            f"existing_hash={status['build_signature']['hash']} "
            f"current_hash={status['current_build_signature']['hash']} "
            f"prepare_calls={mismatch_env.prepare_calls} "
            f"events={[item['event'] for item in progress[-3:]]}"
        )
        assert mismatch_env.prepare_calls == []
        assert any(item["event"] == "resume_signature_mismatch" for item in progress)
        assert status["status"] == "signature_mismatch"
        assert status["build_signature"]["hash"] != status["current_build_signature"]["hash"]
        assert (cache_dir / "segment_index.jsonl").read_text() == original_index_text


def test_stage1_build_signature_summarizes_loaded_motion_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        amass_root = tmp_path / "AMASS_G1NPZ_Final"
        motion_a = amass_root / "AAA" / "motion_a.npz"
        motion_b = amass_root / "BBB" / "motion_b.npz"
        cfg = FrontRESStage1CacheBuilderConfig(
            amass_root=str(amass_root),
            cache_dir=str(tmp_path / "frontres_stage1_cache"),
            horizon_k=2,
            frame_stride=1,
            max_motions=2,
            max_segments=2,
            strengths=(0.0,),
            variants_per_strength=1,
            perturbation_curriculum_mode="discrete_bank",
            base_seed=123,
            env_id=0,
        )
        index_summary = builder.FrontRESAMASSIndexSummary(
            amass_root=str(amass_root),
            motion_count=2,
            segment_count=2,
            horizon_k=2,
            frame_stride=1,
            skipped_short_motions=0,
        )
        perturbation_metadata = {
            "perturbation_curriculum_mode": "discrete_bank",
            "perturbation_levels": [{"level_index": 0, "level_name": "level_00", "strength": 0.0}],
        }

        signature = builder._stage1_build_signature(
            cfg,
            loaded_motion_paths=[str(motion_a), str(motion_b)],
            index_summary=index_summary,
            perturbation_metadata=perturbation_metadata,
        )
        same_signature = builder._stage1_build_signature(
            cfg,
            loaded_motion_paths=[str(motion_a), str(motion_b)],
            index_summary=index_summary,
            perturbation_metadata=perturbation_metadata,
        )
        changed_signature = builder._stage1_build_signature(
            cfg,
            loaded_motion_paths=[str(motion_b), str(motion_a)],
            index_summary=index_summary,
            perturbation_metadata=perturbation_metadata,
        )
        payload = signature["payload"]
        print(
            "[cache_builder signature summary trace] "
            f"loaded_motion_count={payload['loaded_motion_count']} "
            f"first_loaded_motion={Path(payload['first_loaded_motion']).name} "
            f"last_loaded_motion={Path(payload['last_loaded_motion']).name} "
            f"path_hash={payload['loaded_motion_paths_hash'][:12]} "
            f"same_hash={same_signature['hash'] == signature['hash']} "
            f"changed_hash={changed_signature['hash'] != signature['hash']}"
        )
        assert "loaded_motion_paths" not in payload
        assert payload["loaded_motion_count"] == 2
        assert payload["first_loaded_motion"] == str(motion_a.resolve())
        assert payload["last_loaded_motion"] == str(motion_b.resolve())
        assert payload["loaded_motion_paths_hash"]
        assert same_signature["hash"] == signature["hash"]
        assert changed_signature["hash"] != signature["hash"]


def test_stage1_builder_derives_noisy_descriptors_from_hrl_curriculum_bank() -> None:
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
            max_segments=1,
            strengths=(0.0, 0.5),
            variants_per_strength=1,
            curriculum_bank_size=16,
            curriculum_active_dims=(0, 1, 3, 4, 5),
            base_seed=321,
            env_id=0,
        )

        result = build_stage1_segment_cache(env, cfg)
        metadata = json.loads(Path(result.metadata_path).read_text())
        all_noisy = []
        for path in result.noisy_shard_paths.values():
            all_noisy.extend(cache_io.read_noisy_variant_shard(path))
        params = [dict(item.descriptor.params) for item in all_noisy]
        print(
            "[cache_builder trace] hrl_curriculum_bank "
            f"mode={metadata['perturbation_curriculum_mode']} "
            f"segment_count={result.segment_count} "
            f"noisy_count={result.noisy_count} "
            f"strength_counts={result.strength_counts} "
            f"bank_record_count={metadata['curriculum_bank_record_count']} "
            f"allowed_bases={metadata['curriculum_allowed_bases']} "
            f"mix_classes={[item['mix_class'] for item in params]} "
            f"roles={[item['perturbation_role'] for item in params]} "
            f"family_groups={[item['family_group'] for item in params]} "
            f"actual_dr_scales={[item['actual_dr_scale'] for item in params]}"
        )
        assert result.segment_count == 1
        assert result.noisy_count == 16
        assert metadata["perturbation_curriculum_mode"] == "hrl_curriculum_bank"
        assert metadata["legacy_strengths"] == [0.0, 0.5]
        assert metadata["curriculum_bank_record_count"] == 16
        assert metadata["curriculum_allowed_bases"] == ["planar", "yaw", "local_rp"]
        assert set(metadata["strengths"]).issubset({1.5, 2.0, 2.16})
        assert len(metadata["perturbation_levels"]) == 16
        assert len(all_noisy) == 16
        assert {item["curriculum_mode"] for item in params} == {"hrl_curriculum_bank"}
        assert {tuple(item["family_group"]) for item in params}.issubset(
            {
                ("planar", "yaw"),
                ("planar", "local_rp"),
                ("yaw", "local_rp"),
                ("planar",),
                ("yaw",),
                ("local_rp",),
            }
        )
        assert {item["mix_class"] for item in params}.issubset({"easy", "frontier", "hard"})
        assert "hard" in {item["mix_class"] for item in params}
        for item in params:
            if item["mix_class"] == "hard":
                assert item["perturbation_role"] == "boundary_diagnostic"
            else:
                assert item["perturbation_role"] == "train"
            assert item["temporal_mode"] == "single"
            assert item["burst_min_steps"] == 4
            assert item["burst_max_steps"] == 8


if __name__ == "__main__":
    test_stage1_builder_orchestrates_cache_pipeline()
    test_stage1_builder_streaming_path_completes_without_eager_max_segments()
    test_stage1_builder_streaming_commits_first_chunk_before_later_motion_failure()
    test_stage1_builder_streaming_resume_skips_committed_chunk()
    test_stage1_builder_uses_loaded_motion_paths_before_disk_scan()
    test_stage1_builder_resume_skips_committed_segments_and_preserves_manifest()
    test_stage1_builder_resume_after_crash_uses_flush_committed_manifest()
    test_stage1_builder_rejects_resume_signature_mismatch_before_env_work()
    test_stage1_build_signature_summarizes_loaded_motion_paths()
    test_stage1_builder_derives_noisy_descriptors_from_hrl_curriculum_bank()
    print("PASS: FrontRES Stage 1 cache builder orchestrates index, clean, perturbation, noisy, and IO.")
