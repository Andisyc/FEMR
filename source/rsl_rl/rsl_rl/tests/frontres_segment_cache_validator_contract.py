#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile

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
    "frontres_segment_cache_schema_for_validator_contract",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_schema.py",
)
indexer = _load_module(
    "frontres_segment_cache_indexer_for_validator_contract",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_indexer.py",
)
cache_io = _load_module(
    "frontres_segment_cache_io_for_validator_contract",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py",
)
validator = _load_module(
    "frontres_segment_cache_validator_for_contract",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_validator.py",
)

FrontRESAMASSIndexSummary = indexer.FrontRESAMASSIndexSummary
FrontRESCleanStateEntry = cache_io.FrontRESCleanStateEntry
FrontRESNoisyVariant = cache_io.FrontRESNoisyVariant
FrontRESPerturbationDescriptor = schema.FrontRESPerturbationDescriptor
FrontRESRobotRolloutState = schema.FrontRESRobotRolloutState
FrontRESSegmentIndex = schema.FrontRESSegmentIndex


def _segment() -> FrontRESSegmentIndex:
    return FrontRESSegmentIndex(
        segment_id=0,
        motion_rel_path="KIT/359/motion_a.npz",
        motion_num_frames=8,
        fps=30.0,
        start_frame=2,
        horizon_k=4,
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


def _descriptor(perturbation_id: int, strength: float, role: str, mode: str = "hrl_curriculum_bank"):
    return FrontRESPerturbationDescriptor(
        perturbation_id=perturbation_id,
        segment_id=0,
        strength=strength,
        seed=900 + perturbation_id,
        family="hrl_curriculum_bank",
        start_step=0,
        duration=4,
        target="torso_link",
        frame="world",
        params={
            "curriculum_mode": mode,
            "family_group": ("planar", "yaw"),
            "mix_class": "hard" if role == "boundary_diagnostic" else "frontier",
            "mix_class_index": 2 if role == "boundary_diagnostic" else 1,
            "frontier_scale": 2.0,
            "dr_factor": 1.08 if role == "boundary_diagnostic" else 1.0,
            "actual_dr_scale": strength,
            "perturbation_role": role,
            "temporal_mode": "single",
            "burst_min_steps": 4,
            "burst_max_steps": 8,
        },
    )


def _write_minimal_cache(cache_dir: Path, *, wrong_mode: bool = False) -> None:
    segment = _segment()
    summary = FrontRESAMASSIndexSummary(
        amass_root="/tmp/fake_amass",
        motion_count=1,
        segment_count=1,
        horizon_k=4,
        frame_stride=1,
        skipped_short_motions=0,
    )
    indexer.write_amass_segment_index(cache_dir, [segment], summary)
    cache_io.write_clean_state_shard(
        cache_dir,
        [FrontRESCleanStateEntry(segment=segment, clean_state=_state(0.0))],
        shard_id=0,
    )
    for perturbation_id, strength, role in (
        (0, 1.5, "train"),
        (1, 2.16, "boundary_diagnostic"),
    ):
        mode = "discrete_bank" if wrong_mode and perturbation_id == 1 else "hrl_curriculum_bank"
        cache_io.write_noisy_variant_shard(
            cache_dir,
            [
                FrontRESNoisyVariant(
                    segment=segment,
                    descriptor=_descriptor(perturbation_id, strength, role, mode=mode),
                    noisy_state=_state(strength),
                    noisy_baseline_score=torch.tensor([0.4 + strength], dtype=torch.float32),
                    noisy_fall=torch.tensor([1.0 if role == "boundary_diagnostic" else 0.0], dtype=torch.float32),
                    noisy_rollout_len=torch.tensor([4.0], dtype=torch.float32),
                )
            ],
            strength=strength,
            shard_id=0,
        )
    cache_io.write_cache_metadata(
        cache_dir,
        {
            "stage": "stage1_segment_cache",
            "segment_count": 1,
            "clean_count": 1,
            "noisy_count": 2,
            "horizon_k": 4,
            "frame_stride": 1,
            "strengths": [1.5, 2.16],
            "perturbation_curriculum_mode": "hrl_curriculum_bank",
            "perturbation_levels": [],
            "curriculum_bank_record_count": 2,
            "variants_per_strength": 1,
            "clean_shard_id": 0,
            "noisy_shard_id": 0,
        },
    )


def test_stage1_cache_validator_reads_back_metadata_and_shards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_minimal_cache(cache_dir)
        result = validator.validate_stage1_cache_artifacts(cache_dir)
        probe = result.probe()
        print(
            "[cache_validator trace] readback "
            f"stage={probe['metadata_stage']} "
            f"mode={probe['perturbation_curriculum_mode']} "
            f"segment_count={probe['segment_count']} "
            f"clean_count={probe['clean_count']} "
            f"noisy_count={probe['noisy_count']} "
            f"strength_counts={probe['strength_counts']} "
            f"train_count={probe['train_count']} "
            f"boundary_count={probe['boundary_diagnostic_count']} "
            f"clean_shard={Path(probe['clean_shard_path']).relative_to(cache_dir)}"
        )
        assert probe["metadata_stage"] == "stage1_segment_cache"
        assert probe["perturbation_curriculum_mode"] == "hrl_curriculum_bank"
        assert probe["segment_count"] == 1
        assert probe["clean_count"] == 1
        assert probe["noisy_count"] == 2
        assert probe["strength_counts"] == {1.5: 1, 2.16: 1}
        assert probe["train_count"] == 1
        assert probe["boundary_diagnostic_count"] == 1


def test_stage1_cache_validator_rejects_descriptor_mode_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_minimal_cache(cache_dir, wrong_mode=True)
        try:
            validator.validate_stage1_cache_artifacts(cache_dir)
        except ValueError as exc:
            print(f"[cache_validator trace] rejected_mode_drift={exc}")
            assert "curriculum mode does not match metadata" in str(exc)
            return
        raise AssertionError("validator should reject descriptor mode drift")


def test_stage1_cache_validator_cli_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_minimal_cache(cache_dir)
        rc = validator.main(
            [
                str(cache_dir),
                "--expect-mode",
                "hrl_curriculum_bank",
                "--min-segments",
                "1",
                "--min-noisy",
                "2",
                "--require-boundary-diagnostic",
            ]
        )
        print(f"[cache_validator trace] cli_success_returncode={rc}")
        assert rc == 0
        try:
            validator.main([str(cache_dir), "--expect-mode", "discrete_bank"])
        except ValueError as exc:
            print(f"[cache_validator trace] cli_rejected_wrong_mode={exc}")
            assert "unexpected perturbation mode" in str(exc)
            return
        raise AssertionError("validator CLI should reject the wrong expected mode")


if __name__ == "__main__":
    test_stage1_cache_validator_reads_back_metadata_and_shards()
    test_stage1_cache_validator_rejects_descriptor_mode_drift()
    test_stage1_cache_validator_cli_contract()
    print("PASS: FrontRES Stage 1 cache validator reads back metadata and clean/noisy shards.")
