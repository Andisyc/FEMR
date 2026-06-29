#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_segment_cache_io.py"
spec = importlib.util.spec_from_file_location("frontres_segment_cache_io", MODULE_PATH)
cache_io = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = cache_io
spec.loader.exec_module(cache_io)

FrontRESCleanStateEntry = cache_io.FrontRESCleanStateEntry
FrontRESNoisyVariant = cache_io.FrontRESNoisyVariant
FrontRESPerturbationDescriptor = cache_io.FrontRESPerturbationDescriptor
FrontRESRobotRolloutState = cache_io.FrontRESRobotRolloutState
FrontRESSegmentIndex = cache_io.FrontRESSegmentIndex


def _segment(segment_id: int, start_frame: int) -> FrontRESSegmentIndex:
    return FrontRESSegmentIndex(
        segment_id=segment_id,
        motion_rel_path=f"KIT/359/motion_{segment_id}.npz",
        motion_num_frames=20,
        fps=30.0,
        start_frame=start_frame,
        horizon_k=4,
    )


def _state(offset: float) -> FrontRESRobotRolloutState:
    batch = 1
    dofs = 29
    bodies = 30
    return FrontRESRobotRolloutState(
        root_pos=torch.full((batch, 3), 1.0 + offset),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        root_lin_vel=torch.full((batch, 3), 0.1 + offset),
        root_ang_vel=torch.full((batch, 3), 0.2 + offset),
        joint_pos=torch.arange(dofs, dtype=torch.float32).view(batch, dofs) + offset,
        joint_vel=torch.arange(dofs, dtype=torch.float32).view(batch, dofs) * 0.01 + offset,
        body_pos_w=torch.full((batch, bodies, 3), 2.0 + offset),
        body_quat_w=torch.zeros(batch, bodies, 4).index_fill(2, torch.tensor([0]), 1.0),
        body_lin_vel_w=torch.full((batch, bodies, 3), 0.3 + offset),
        body_ang_vel_w=torch.full((batch, bodies, 3), 0.4 + offset),
        contact_state=torch.full((batch, 4), offset),
        action_history=torch.full((batch, 2, dofs), offset),
    )


def _descriptor(segment_id: int, perturbation_id: int, strength: float) -> FrontRESPerturbationDescriptor:
    return FrontRESPerturbationDescriptor(
        perturbation_id=perturbation_id,
        segment_id=segment_id,
        strength=strength,
        seed=1000 + perturbation_id,
        family="external_push",
        start_step=0,
        duration=2,
        target="torso_link",
        frame="world",
        params={
            "curriculum_mode": "discrete_bank",
            "family": "external_push",
            "level_index": 1,
            "level_name": "level_01",
            "level_strength": strength,
            "variant_index": 0,
            "axis": [1.0, 0.0, 0.0],
            "signed_magnitude": strength,
            "frame": "world",
        },
    )


def _variant(segment_id: int, perturbation_id: int, strength: float, offset: float) -> FrontRESNoisyVariant:
    segment = _segment(segment_id, start_frame=segment_id)
    return FrontRESNoisyVariant(
        segment=segment,
        descriptor=_descriptor(segment.segment_id, perturbation_id, strength),
        noisy_state=_state(offset),
        noisy_baseline_score=torch.tensor([0.5 + offset], dtype=torch.float32),
        noisy_fall=torch.tensor([0.0], dtype=torch.float32),
        noisy_rollout_len=torch.tensor([4.0], dtype=torch.float32),
    )


def test_cache_io_round_trips_clean_and_noisy_shards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "frontres_segment_cache"
        clean_entries = [
            FrontRESCleanStateEntry(segment=_segment(7, 3), clean_state=_state(0.0)),
            FrontRESCleanStateEntry(segment=_segment(8, 4), clean_state=_state(1.0)),
        ]
        noisy_variants = [
            _variant(7, 3, 0.5, 0.5),
            _variant(8, 4, 0.5, 1.5),
        ]

        metadata_path = cache_io.write_cache_metadata(
            cache_dir,
            {"stage": "stage1_segment_cache", "segment_count": 2, "noisy_variant_count": 2},
        )
        clean_path = cache_io.write_clean_state_shard(cache_dir, clean_entries, shard_id=0)
        noisy_path = cache_io.write_noisy_variant_shard(cache_dir, noisy_variants, strength=0.5, shard_id=0)

        print(
            "[cache_io trace] write "
            f"metadata={metadata_path.relative_to(cache_dir)} "
            f"clean={clean_path.relative_to(cache_dir)} "
            f"noisy={noisy_path.relative_to(cache_dir)} "
            f"clean_ids={[entry.segment_id for entry in clean_entries]} "
            f"noisy_ids={[(variant.segment_id, variant.perturbation_id) for variant in noisy_variants]}"
        )
        clean_payload_path = (
            cache_dir / "KIT" / "359" / "motion_7" / "segment_00000007_start_00000003_k_0004" / "clean.pt"
        )
        noisy_payload_path = (
            cache_dir
            / "KIT"
            / "359"
            / "motion_7"
            / "segment_00000007_start_00000007_k_0004"
            / "noisy_variants"
            / "strength_0p5"
            / "perturbation_00000003.pt"
        )
        print(
            "[cache_io trace] mirror_paths "
            f"clean={clean_payload_path.relative_to(cache_dir)} exists={clean_payload_path.exists()} "
            f"noisy={noisy_payload_path.relative_to(cache_dir)} exists={noisy_payload_path.exists()}"
        )

        metadata = cache_io.read_cache_metadata(cache_dir)
        loaded_clean = cache_io.read_clean_state_shard(clean_path)
        loaded_noisy = cache_io.read_noisy_variant_shard(noisy_path)

        print(
            "[cache_io trace] read "
            f"format={metadata['format']} "
            f"clean_ids={[entry.segment_id for entry in loaded_clean]} "
            f"noisy_ids={[(variant.segment_id, variant.perturbation_id) for variant in loaded_noisy]} "
            f"clean_shape={loaded_clean[0].clean_state.body_pos_w.shape} "
            f"noisy_shape={loaded_noisy[0].noisy_state.body_pos_w.shape}"
        )

        assert metadata["format"] == "frontres_segment_cache_v1"
        assert metadata["segment_count"] == 2
        assert clean_payload_path.exists()
        assert noisy_payload_path.exists()
        assert clean_path.relative_to(cache_dir).as_posix() == "manifests/clean_states/shard_000000.pt"
        assert noisy_path.relative_to(cache_dir).as_posix() == "manifests/noisy_variants/strength_0p5/shard_000000.pt"
        assert [entry.segment_id for entry in loaded_clean] == [7, 8]
        assert [(variant.segment_id, variant.perturbation_id) for variant in loaded_noisy] == [(7, 3), (8, 4)]
        for before, after in zip(clean_entries, loaded_clean):
            assert before.segment.motion_rel_path == after.segment.motion_rel_path
            torch.testing.assert_close(before.clean_state.root_pos, after.clean_state.root_pos)
            torch.testing.assert_close(before.clean_state.joint_pos, after.clean_state.joint_pos)
            torch.testing.assert_close(before.clean_state.body_pos_w, after.clean_state.body_pos_w)
            assert after.clean_state.root_pos.requires_grad is False
        for before, after in zip(noisy_variants, loaded_noisy):
            assert before.descriptor.seed == after.descriptor.seed
            assert before.descriptor.params == after.descriptor.params
            assert after.descriptor.params["curriculum_mode"] == "discrete_bank"
            assert after.descriptor.params["level_index"] == 1
            assert after.descriptor.params["level_name"] == "level_01"
            torch.testing.assert_close(before.noisy_state.root_lin_vel, after.noisy_state.root_lin_vel)
            torch.testing.assert_close(before.noisy_baseline_score, after.noisy_baseline_score)
            assert after.noisy_state.root_pos.requires_grad is False


def test_cache_io_rejects_id_drift_before_write() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "frontres_segment_cache"
        segment = _segment(7, 3)
        bad = FrontRESNoisyVariant(
            segment=segment,
            descriptor=_descriptor(segment_id=8, perturbation_id=3, strength=0.5),
            noisy_state=_state(0.5),
            noisy_baseline_score=torch.tensor([0.5], dtype=torch.float32),
            noisy_fall=torch.tensor([0.0], dtype=torch.float32),
            noisy_rollout_len=torch.tensor([4.0], dtype=torch.float32),
        )
        try:
            cache_io.write_noisy_variant_shard(cache_dir, [bad], strength=0.5, shard_id=0)
        except ValueError as exc:
            print(f"[cache_io trace] rejected_id_drift_before_write={exc}")
            assert "does not match" in str(exc)
            return
        raise AssertionError("writer should reject noisy variant id drift")


if __name__ == "__main__":
    test_cache_io_round_trips_clean_and_noisy_shards()
    test_cache_io_rejects_id_drift_before_write()
    print("PASS: FrontRES Segment cache IO round-trips clean states and noisy variants.")
