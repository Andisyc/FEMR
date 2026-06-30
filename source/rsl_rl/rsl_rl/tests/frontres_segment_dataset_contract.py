from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_dataset.py"
spec = importlib.util.spec_from_file_location("frontres_segment_dataset", MODULE_PATH)
dataset_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = dataset_module
spec.loader.exec_module(dataset_module)
FrontRESSegmentDataset = dataset_module.FrontRESSegmentDataset
build_stage1_cache_motion_source = dataset_module.build_stage1_cache_motion_source
build_stage1_cache_lazy_records = dataset_module.build_stage1_cache_lazy_records
load_stage1_cache_dataset = dataset_module.load_stage1_cache_dataset


def _load_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


schema = _load_module("frontres_segment_cache_schema_for_dataset_contract", ROOT / "rsl_rl" / "frontres" / "frontres_segment_cache_schema.py")
indexer = _load_module("frontres_segment_cache_indexer_for_dataset_contract", ROOT / "rsl_rl" / "frontres" / "frontres_segment_cache_indexer.py")
cache_io = _load_module("frontres_segment_cache_io_for_dataset_contract", ROOT / "rsl_rl" / "frontres" / "frontres_segment_cache_io.py")


def _fake_motion(motion_id: int, frames: int = 6, dofs: int = 4) -> dict[str, torch.Tensor | int | float | str]:
    base = torch.arange(frames, dtype=torch.float32).unsqueeze(-1)
    dof_base = torch.arange(frames * dofs, dtype=torch.float32).reshape(frames, dofs)
    return {
        "motion_id": motion_id,
        "root_pos": torch.cat([base, base + 1.0, base + 2.0], dim=-1),
        "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(frames, 1),
        "root_lin_vel": torch.ones(frames, 3) * (motion_id + 1),
        "root_ang_vel": torch.ones(frames, 3) * (motion_id + 2),
        "dof_pos": dof_base + motion_id,
        "dof_vel": torch.ones(frames, dofs) * (motion_id + 3),
        "reference": torch.arange(frames * 2, dtype=torch.float32).reshape(frames, 2) + motion_id,
        "perturbation_family": "yaw",
        "perturbation_strength": 0.2,
    }


def _cache_segment(
    *,
    segment_id: int = 0,
    motion_rel_path: str = "KIT/359/motion_a.npz",
    start_frame: int = 2,
):
    return schema.FrontRESSegmentIndex(
        segment_id=segment_id,
        motion_rel_path=motion_rel_path,
        motion_num_frames=8,
        fps=30.0,
        start_frame=start_frame,
        horizon_k=4,
    )


def _cache_state(offset: float):
    batch = 1
    dofs = 29
    bodies = 30
    return schema.FrontRESRobotRolloutState(
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


def _cache_descriptor(perturbation_id: int, strength: float, role: str, *, segment_id: int = 0):
    return schema.FrontRESPerturbationDescriptor(
        perturbation_id=perturbation_id,
        segment_id=segment_id,
        strength=strength,
        seed=900 + perturbation_id,
        family="hrl_curriculum_bank",
        start_step=0,
        duration=4,
        target="torso_link",
        frame="world",
        params={
            "curriculum_mode": "hrl_curriculum_bank",
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


def _write_stage1_cache(cache_dir: Path) -> None:
    segment = _cache_segment()
    summary = indexer.FrontRESAMASSIndexSummary(
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
        [cache_io.FrontRESCleanStateEntry(segment=segment, clean_state=_cache_state(0.0))],
        shard_id=0,
    )
    for perturbation_id, strength, role in (
        (0, 1.5, "train"),
        (1, 2.16, "boundary_diagnostic"),
    ):
        cache_io.write_noisy_variant_shard(
            cache_dir,
            [
                schema.FrontRESNoisyVariant(
                    segment=segment,
                    descriptor=_cache_descriptor(perturbation_id, strength, role),
                    noisy_state=_cache_state(strength),
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
            "clean_shard_id": 0,
            "noisy_shard_id": 0,
        },
    )


def _write_stage1_multirow_cache(cache_dir: Path) -> None:
    segments = [
        _cache_segment(segment_id=0, motion_rel_path="KIT/359/motion_a.npz", start_frame=2),
        _cache_segment(segment_id=1, motion_rel_path="KIT/359/motion_b.npz", start_frame=3),
    ]
    summary = indexer.FrontRESAMASSIndexSummary(
        amass_root="/tmp/fake_amass",
        motion_count=2,
        segment_count=2,
        horizon_k=4,
        frame_stride=1,
        skipped_short_motions=0,
    )
    indexer.write_amass_segment_index(cache_dir, segments, summary)
    cache_io.write_clean_state_shard(
        cache_dir,
        [
            cache_io.FrontRESCleanStateEntry(segment=segments[0], clean_state=_cache_state(10.0)),
            cache_io.FrontRESCleanStateEntry(segment=segments[1], clean_state=_cache_state(20.0)),
        ],
        shard_id=0,
    )
    variants = [
        schema.FrontRESNoisyVariant(
            segment=segments[0],
            descriptor=_cache_descriptor(perturbation_id=0, strength=1.5, role="train", segment_id=0),
            noisy_state=_cache_state(101.0),
            noisy_baseline_score=torch.tensor([1.1], dtype=torch.float32),
            noisy_fall=torch.tensor([0.0], dtype=torch.float32),
            noisy_rollout_len=torch.tensor([4.0], dtype=torch.float32),
        ),
        schema.FrontRESNoisyVariant(
            segment=segments[1],
            descriptor=_cache_descriptor(perturbation_id=1, strength=1.5, role="train", segment_id=1),
            noisy_state=_cache_state(202.0),
            noisy_baseline_score=torch.tensor([2.2], dtype=torch.float32),
            noisy_fall=torch.tensor([0.0], dtype=torch.float32),
            noisy_rollout_len=torch.tensor([4.0], dtype=torch.float32),
        ),
    ]
    cache_io.write_noisy_variant_shard(cache_dir, variants, strength=1.5, shard_id=0)
    cache_io.write_cache_metadata(
        cache_dir,
        {
            "stage": "stage1_segment_cache",
            "segment_count": 2,
            "clean_count": 2,
            "noisy_count": 2,
            "horizon_k": 4,
            "frame_stride": 1,
            "strengths": [1.5],
            "perturbation_curriculum_mode": "hrl_curriculum_bank",
            "clean_shard_id": 0,
            "noisy_shard_id": 0,
        },
    )


def _write_stage1_multishard_cache(cache_dir: Path) -> None:
    segment = _cache_segment(segment_id=0, motion_rel_path="KIT/359/motion_a.npz", start_frame=2)
    summary = indexer.FrontRESAMASSIndexSummary(
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
        [cache_io.FrontRESCleanStateEntry(segment=segment, clean_state=_cache_state(10.0))],
        shard_id=0,
    )
    strengths = [1.5, 2.0, 2.5]
    for perturbation_id, strength in enumerate(strengths):
        cache_io.write_noisy_variant_shard(
            cache_dir,
            [
                schema.FrontRESNoisyVariant(
                    segment=segment,
                    descriptor=_cache_descriptor(
                        perturbation_id=perturbation_id,
                        strength=strength,
                        role="train",
                        segment_id=0,
                    ),
                    noisy_state=_cache_state(100.0 + 10.0 * perturbation_id),
                    noisy_baseline_score=torch.tensor([1.0 + perturbation_id], dtype=torch.float32),
                    noisy_fall=torch.tensor([0.0], dtype=torch.float32),
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
            "noisy_count": 3,
            "horizon_k": 4,
            "frame_stride": 1,
            "strengths": strengths,
            "perturbation_curriculum_mode": "hrl_curriculum_bank",
            "clean_shard_id": 0,
            "noisy_shard_id": 0,
        },
    )


def test_dataset_samples_stable_dynamic_segments() -> None:
    dataset = FrontRESSegmentDataset(
        motion_source=[_fake_motion(10), _fake_motion(11)],
        dt=0.02,
        default_horizon_k=2,
        device="cpu",
    )
    assert dataset.num_segments() == 8

    batch = dataset.get_segments(torch.tensor([0, 4]))
    assert batch.segment_ids.tolist() == [0, 4]
    assert batch.clean_state.root_pos.shape == (2, 3)
    assert batch.clean_state.root_lin_vel.shape == (2, 3)
    assert batch.clean_state.root_ang_vel.shape == (2, 3)
    assert batch.clean_state.dof_pos.shape == (2, 4)
    assert batch.clean_state.dof_vel.shape == (2, 4)
    assert batch.reference_window.shape == (2, 3, 2)
    assert batch.horizon_k.tolist() == [2, 2]
    assert batch.perturbation_family == ("yaw", "yaw")

    repeat = dataset.get_segments([0, 4])
    torch.testing.assert_close(repeat.clean_state.root_pos, batch.clean_state.root_pos)
    torch.testing.assert_close(repeat.clean_state.dof_vel, batch.clean_state.dof_vel)


def test_dataset_global_sampling_excludes_invalid_segments() -> None:
    dataset = FrontRESSegmentDataset([_fake_motion(0)], dt=0.02, default_horizon_k=2, device="cpu")
    dataset.update_validity([0, 1, 2], [False, False, False], reason="bad reset")

    generator = torch.Generator().manual_seed(0)
    batch = dataset.sample_global(16, generator=generator)
    assert all(segment_id == 3 for segment_id in batch.segment_ids.tolist())

    validation = dataset.validate_batch(dataset.get_segments([0, 3]))
    assert validation.valid_mask.tolist() == [False, True]
    assert validation.reasons[0] == "bad reset"


def test_dataset_state_dict_restores_invalidity_and_baseline() -> None:
    dataset = FrontRESSegmentDataset([_fake_motion(0)], dt=0.02, default_horizon_k=2, device="cpu")
    dataset.update_validity([1], [False], reason="unstable")
    dataset.write_noisy_baseline([2], {"score": 0.5})

    restored = FrontRESSegmentDataset([_fake_motion(0)], dt=0.02, default_horizon_k=2, device="cpu")
    restored.load_state_dict(dataset.state_dict())

    validation = restored.validate_batch(restored.get_segments([1, 2]))
    assert validation.valid_mask.tolist() == [False, True]
    assert restored.read_noisy_baseline([2])[2]["score"] == 0.5


def test_dataset_loads_stage1_cache_and_excludes_boundary_diagnostics_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_stage1_cache(cache_dir)
        motion_source, summary = build_stage1_cache_motion_source(cache_dir)
        print(
            "[dataset cache trace] motion_source "
            f"loaded={summary.loaded_motion_count} "
            f"skipped_boundary={summary.skipped_boundary_diagnostic_count} "
            f"role_counts={summary.role_counts} "
            f"roles={[motion['perturbation_role'] for motion in motion_source]} "
            f"strengths={[motion['perturbation_strength'] for motion in motion_source]}"
        )
        assert summary.loaded_motion_count == 1
        assert summary.skipped_boundary_diagnostic_count == 1
        assert summary.role_counts == {"train": 1, "boundary_diagnostic": 1}
        assert [motion["perturbation_role"] for motion in motion_source] == ["train"]

        dataset = load_stage1_cache_dataset(cache_dir, device="cpu")
        assert dataset.num_segments() == 1
        batch = dataset.sample_global(1, generator=torch.Generator().manual_seed(0))
        print(
            "[dataset cache trace] batch "
            f"ids={batch.segment_ids.tolist()} "
            f"roles={batch.perturbation_role} "
            f"families={batch.perturbation_family} "
            f"strength={batch.perturbation_strength.tolist()} "
            f"metadata={dataset.cache_metadata()}"
        )
        assert batch.perturbation_role == ("train",)
        assert batch.perturbation_strength.tolist() == [1.5]
        assert dataset.cache_metadata()["skipped_boundary_diagnostic_count"] == 1


def test_dataset_lazy_cache_uses_lru_shard_reads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_stage1_cache(cache_dir)
        records, summary = build_stage1_cache_lazy_records(cache_dir)
        print(
            "[dataset cache trace] lazy_records "
            f"loaded={summary.loaded_motion_count} "
            f"record_paths={[record.manifest_record['path'] for record in records]} "
            f"record_rows={[record.manifest_record.get('row') for record in records]}"
        )
        assert summary.loaded_motion_count == 1
        assert records[0].manifest_record["path"].startswith("shards/noisy_variants/")

        dataset = load_stage1_cache_dataset(cache_dir, device="cpu", lazy=True, shard_cache_size=1)
        before = dataset.cache_metadata()["shard_cache"]
        batch = dataset.get_segments([0])
        after_first = dataset.cache_metadata()["shard_cache"]
        repeat = dataset.get_segments([0])
        after_second = dataset.cache_metadata()["shard_cache"]
        print(
            "[dataset cache trace] lazy_lru "
            f"before={before} "
            f"after_first={after_first} "
            f"after_second={after_second} "
            f"batch_root={batch.clean_state.root_pos.tolist()} "
            f"repeat_root={repeat.clean_state.root_pos.tolist()}"
        )
        assert before["load_count"] == 0
        assert after_first["load_count"] == 1
        assert after_second["load_count"] == 1
        assert after_second["hit_count"] >= 1
        torch.testing.assert_close(batch.clean_state.root_pos, repeat.clean_state.root_pos)


def test_dataset_lazy_read_maps_segment_id_to_manifest_row_and_batch_values() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_stage1_multirow_cache(cache_dir)

        records, summary = build_stage1_cache_lazy_records(cache_dir)
        manifest_rows = [int(record.manifest_record["row"]) for record in records]
        source_segment_ids = [int(record.source_segment_id) for record in records]
        source_perturbation_ids = [int(record.source_perturbation_id) for record in records]
        print(
            "[dataset cache trace] lazy_row_index "
            f"loaded={summary.loaded_motion_count} "
            f"record_segment_ids={[record.segment_id for record in records]} "
            f"source_segment_ids={source_segment_ids} "
            f"source_perturbation_ids={source_perturbation_ids} "
            f"manifest_paths={[record.manifest_record['path'] for record in records]} "
            f"manifest_rows={manifest_rows}"
        )
        assert summary.loaded_motion_count == 2
        assert [record.segment_id for record in records] == [0, 1]
        assert source_segment_ids == [0, 1]
        assert source_perturbation_ids == [0, 1]
        assert manifest_rows == [0, 1]
        assert records[0].manifest_record["path"] == records[1].manifest_record["path"]

        dataset = load_stage1_cache_dataset(cache_dir, device="cpu", lazy=True, shard_cache_size=1)
        requested_ids = [1, 0]
        batch = dataset.get_segments(requested_ids)
        metadata = dataset.cache_metadata()
        print(
            "[dataset cache trace] lazy_row_batch "
            f"requested_ids={requested_ids} "
            f"batch_ids={batch.segment_ids.tolist()} "
            f"spec_motion_ids={[spec.motion_id for spec in batch.specs]} "
            f"root_pos={batch.clean_state.root_pos.tolist()} "
            f"joint_pos_head={batch.clean_state.dof_pos[:, :3].tolist()} "
            f"reference_head={batch.reference_window[:, 0, :3].tolist()} "
            f"shard_cache={metadata['shard_cache']}"
        )
        assert batch.segment_ids.tolist() == requested_ids
        assert [spec.motion_id for spec in batch.specs] == ["1:1", "0:0"]
        torch.testing.assert_close(batch.clean_state.root_pos, torch.tensor([[202.0, 202.0, 202.0], [101.0, 101.0, 101.0]]))
        torch.testing.assert_close(batch.clean_state.dof_pos[:, :3], torch.tensor([[202.0, 203.0, 204.0], [101.0, 102.0, 103.0]]))
        torch.testing.assert_close(batch.reference_window[:, 0, :3], torch.tensor([[202.0, 203.0, 204.0], [101.0, 102.0, 103.0]]))
        assert metadata["shard_cache"]["load_count"] == 1
        assert metadata["shard_cache"]["hit_count"] >= 1


def test_dataset_lazy_cache_lru_bounds_multishard_reads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_stage1_multishard_cache(cache_dir)

        records, summary = build_stage1_cache_lazy_records(cache_dir)
        print(
            "[dataset cache trace] lazy_multishard_index "
            f"loaded={summary.loaded_motion_count} "
            f"record_segment_ids={[record.segment_id for record in records]} "
            f"strengths={[record.perturbation_strength for record in records]} "
            f"paths={[record.manifest_record['path'] for record in records]} "
            f"rows={[record.manifest_record['row'] for record in records]}"
        )
        assert summary.loaded_motion_count == 3
        assert [record.segment_id for record in records] == [0, 1, 2]
        assert [round(record.perturbation_strength, 1) for record in records] == [1.5, 2.0, 2.5]
        assert len({record.manifest_record["path"] for record in records}) == 3

        dataset = load_stage1_cache_dataset(cache_dir, device="cpu", lazy=True, shard_cache_size=2)
        access_order = [0, 1, 2, 0]
        probes = []
        roots = []
        for segment_id in access_order:
            batch = dataset.get_segments([segment_id])
            probes.append(dataset.cache_metadata()["shard_cache"])
            roots.append(float(batch.clean_state.root_pos[0, 0].item()))

        print(
            "[dataset cache trace] lazy_multishard_lru "
            f"access_order={access_order} "
            f"roots={roots} "
            f"probes={probes}"
        )
        assert roots == [100.0, 110.0, 120.0, 100.0]
        assert max(item["resident_shards"] for item in probes) <= 2
        assert probes[0]["load_count"] == 1
        assert probes[1]["load_count"] == 2
        assert probes[2]["load_count"] == 3
        assert probes[3]["load_count"] == 4


def test_dataset_lazy_global_sampling_excludes_runtime_invalid_segments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_stage1_cache(cache_dir)
        dataset = load_stage1_cache_dataset(cache_dir, device="cpu", lazy=True, shard_cache_size=1)
        dataset.update_validity([0], [False], reason="runtime fall")

        validation = dataset.validate_batch(dataset.get_segments([0]))
        print(
            "[dataset cache trace] lazy_invalid_sampling "
            f"valid_mask={validation.valid_mask.tolist()} "
            f"reasons={validation.reasons} "
            f"metadata={dataset.cache_metadata()}"
        )
        assert validation.valid_mask.tolist() == [False]
        assert validation.reasons == ("runtime fall",)
        try:
            dataset.sample_global(1, generator=torch.Generator().manual_seed(0))
        except RuntimeError as exc:
            assert "no valid FrontRES segments" in str(exc)
        else:
            raise AssertionError("lazy sample_global should exclude runtime-invalid segments")


def test_dataset_can_include_boundary_diagnostics_as_invalid_samples() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_stage1_cache(cache_dir)
        motion_source, summary = build_stage1_cache_motion_source(cache_dir, include_boundary_diagnostic=True)
        dataset = FrontRESSegmentDataset(motion_source, dt=1.0 / 30.0, default_horizon_k=4, device="cpu")
        batch = dataset.get_segments([0, 1])
        validation = dataset.validate_batch(batch)
        print(
            "[dataset cache trace] include_boundary "
            f"loaded={summary.loaded_motion_count} "
            f"included_boundary={summary.included_boundary_diagnostic_count} "
            f"roles={batch.perturbation_role} "
            f"valid_mask={validation.valid_mask.tolist()} "
            f"reasons={validation.reasons}"
        )
        assert summary.loaded_motion_count == 2
        assert summary.included_boundary_diagnostic_count == 1
        assert batch.perturbation_role == ("train", "boundary_diagnostic")
        assert validation.valid_mask.tolist() == [True, False]
        assert validation.reasons[1] == "marked invalid"


def main() -> None:
    test_dataset_samples_stable_dynamic_segments()
    test_dataset_global_sampling_excludes_invalid_segments()
    test_dataset_state_dict_restores_invalidity_and_baseline()
    test_dataset_loads_stage1_cache_and_excludes_boundary_diagnostics_by_default()
    test_dataset_lazy_cache_uses_lru_shard_reads()
    test_dataset_lazy_read_maps_segment_id_to_manifest_row_and_batch_values()
    test_dataset_lazy_cache_lru_bounds_multishard_reads()
    test_dataset_lazy_global_sampling_excludes_runtime_invalid_segments()
    test_dataset_can_include_boundary_diagnostics_as_invalid_samples()
    print("result: PASS")


if __name__ == "__main__":
    main()
