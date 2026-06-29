#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
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


schema = _load_module(
    "frontres_segment_cache_schema_for_completion_contract",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_schema.py",
)
indexer = _load_module(
    "frontres_segment_cache_indexer_for_completion_contract",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_indexer.py",
)
cache_io = _load_module(
    "frontres_segment_cache_io_for_completion_contract",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py",
)
completion = _load_module(
    "frontres_stage1_cache_completion_script_for_contract",
    "scripts/rsl_rl/check_frontres_stage1_segment_cache_completion.py",
)

FrontRESAMASSIndexSummary = indexer.FrontRESAMASSIndexSummary
FrontRESCleanStateEntry = cache_io.FrontRESCleanStateEntry
FrontRESNoisyVariant = cache_io.FrontRESNoisyVariant
FrontRESPerturbationDescriptor = schema.FrontRESPerturbationDescriptor
FrontRESRobotRolloutState = schema.FrontRESRobotRolloutState


def _write_fake_amass_npz(path: Path, *, frames: int, fps: int = 30, dofs: int = 29, bodies: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.arange(frames, dtype=np.float32)
    np.savez(
        path,
        fps=np.array([fps], dtype=np.int64),
        joint_pos=np.tile(base[:, None], (1, dofs)).astype(np.float32),
        joint_vel=np.tile((base * 0.1)[:, None], (1, dofs)).astype(np.float32),
        body_pos_w=np.ones((frames, bodies, 3), dtype=np.float32),
        body_quat_w=np.concatenate(
            [np.ones((frames, bodies, 1), dtype=np.float32), np.zeros((frames, bodies, 3), dtype=np.float32)],
            axis=-1,
        ),
        body_lin_vel_w=np.full((frames, bodies, 3), 0.2, dtype=np.float32),
        body_ang_vel_w=np.full((frames, bodies, 3), 0.3, dtype=np.float32),
    )


def _state(offset: float) -> FrontRESRobotRolloutState:
    batch = 1
    dofs = 29
    bodies = 30
    return FrontRESRobotRolloutState(
        root_pos=torch.full((batch, 3), offset),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        root_lin_vel=torch.full((batch, 3), offset + 0.1),
        root_ang_vel=torch.full((batch, 3), offset + 0.2),
        joint_pos=torch.arange(dofs, dtype=torch.float32).view(batch, dofs) + offset,
        joint_vel=torch.arange(dofs, dtype=torch.float32).view(batch, dofs) * 0.01 + offset,
        body_pos_w=torch.full((batch, bodies, 3), offset + 1.0),
        body_quat_w=torch.zeros(batch, bodies, 4).index_fill(2, torch.tensor([0]), 1.0),
        body_lin_vel_w=torch.full((batch, bodies, 3), offset + 0.3),
        body_ang_vel_w=torch.full((batch, bodies, 3), offset + 0.4),
        contact_state=torch.ones(batch, 4),
        action_history=torch.zeros(batch, 2, dofs),
    )


def _descriptor(segment_id: int, perturbation_id: int, strength: float) -> FrontRESPerturbationDescriptor:
    return FrontRESPerturbationDescriptor(
        perturbation_id=perturbation_id,
        segment_id=segment_id,
        strength=strength,
        seed=1000 + perturbation_id,
        family="hrl_curriculum_bank",
        start_step=0,
        duration=4,
        target="torso_link",
        frame="world",
        params={
            "curriculum_mode": "hrl_curriculum_bank",
            "family_group": ("planar", "yaw"),
            "mix_class": "frontier",
            "mix_class_index": 1,
            "frontier_scale": 2.0,
            "dr_factor": 1.0,
            "actual_dr_scale": strength,
            "perturbation_role": "train",
            "temporal_mode": "single",
            "burst_min_steps": 4,
            "burst_max_steps": 8,
        },
    )


def _write_stage1_cache(cache_dir: Path, segments: list, *, drop_last_segment: bool = False) -> None:
    written_segments = segments[:-1] if drop_last_segment else segments
    summary = FrontRESAMASSIndexSummary(
        amass_root="/tmp/fake_amass",
        motion_count=2,
        segment_count=len(written_segments),
        horizon_k=4,
        frame_stride=2,
        skipped_short_motions=0,
    )
    indexer.write_amass_segment_index(cache_dir, written_segments, summary)
    cache_io.write_clean_state_shard(
        cache_dir,
        [FrontRESCleanStateEntry(segment=segment, clean_state=_state(0.0)) for segment in written_segments],
        shard_id=0,
    )
    variants = []
    for idx, segment in enumerate(written_segments):
        variants.append(
            FrontRESNoisyVariant(
                segment=segment,
                descriptor=_descriptor(segment.segment_id, idx, 1.5),
                noisy_state=_state(1.5),
                noisy_baseline_score=torch.tensor([0.1], dtype=torch.float32),
                noisy_fall=torch.tensor([0.0], dtype=torch.float32),
                noisy_rollout_len=torch.tensor([4.0], dtype=torch.float32),
            )
        )
    cache_io.write_noisy_variant_shard(cache_dir, variants, strength=1.5, shard_id=0)
    cache_io.write_cache_metadata(
        cache_dir,
        {
            "stage": "stage1_segment_cache",
            "amass_root": "/tmp/fake_amass",
            "segment_count": len(written_segments),
            "clean_count": len(written_segments),
            "noisy_count": len(written_segments),
            "horizon_k": 4,
            "frame_stride": 2,
            "strengths": [1.5],
            "perturbation_curriculum_mode": "hrl_curriculum_bank",
            "perturbation_levels": [],
            "curriculum_bank_record_count": 1,
            "variants_per_strength": 1,
            "clean_shard_id": 0,
            "noisy_shard_id": 0,
        },
    )


def test_stage1_completion_script_reports_pass_for_complete_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        amass_root = Path(tmp) / "AMASS_G1NPZ_Final"
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_fake_amass_npz(amass_root / "AAA" / "motion_a.npz", frames=9)
        _write_fake_amass_npz(amass_root / "BBB" / "motion_b.npz", frames=8)
        segments, _ = indexer.build_amass_segment_index(amass_root, horizon_k=4, frame_stride=2)
        _write_stage1_cache(cache_dir, segments)

        probe = completion.check_stage1_segment_cache_completion(
            amass_root,
            cache_dir,
            horizon_k=4,
            frame_stride=2,
            max_motions=None,
            max_segments=None,
            expect_mode="hrl_curriculum_bank",
            deep_read_shards=True,
            show_missing_limit=3,
        )
        print(
            "[stage1_completion trace] pass_case "
            f"status={probe.status} expected_segments={probe.expected_segment_count} "
            f"cached_segments={probe.cached_segment_count} noisy={probe.metadata_noisy_count}"
        )
        assert probe.status == "PASS"
        assert probe.expected_segment_count == probe.cached_segment_count
        assert probe.deep_validation_ok


def test_stage1_completion_script_reports_incomplete_for_missing_segment() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        amass_root = Path(tmp) / "AMASS_G1NPZ_Final"
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_fake_amass_npz(amass_root / "AAA" / "motion_a.npz", frames=9)
        _write_fake_amass_npz(amass_root / "BBB" / "motion_b.npz", frames=8)
        segments, _ = indexer.build_amass_segment_index(amass_root, horizon_k=4, frame_stride=2)
        _write_stage1_cache(cache_dir, segments, drop_last_segment=True)

        probe = completion.check_stage1_segment_cache_completion(
            amass_root,
            cache_dir,
            horizon_k=4,
            frame_stride=2,
            max_motions=None,
            max_segments=None,
            expect_mode="hrl_curriculum_bank",
            deep_read_shards=False,
            show_missing_limit=3,
        )
        print(
            "[stage1_completion trace] incomplete_case "
            f"status={probe.status} missing={probe.missing_segment_count} "
            f"expected_segments={probe.expected_segment_count} cached_segments={probe.cached_segment_count}"
        )
        assert probe.status == "INCOMPLETE"
        assert probe.missing_segment_count == 1


if __name__ == "__main__":
    test_stage1_completion_script_reports_pass_for_complete_cache()
    test_stage1_completion_script_reports_incomplete_for_missing_segment()
    print("PASS: FrontRES Stage 1 completion checker compares AMASS and Segment cache.")
