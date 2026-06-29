#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_segment_cache_indexer.py"
spec = importlib.util.spec_from_file_location("frontres_segment_cache_indexer", MODULE_PATH)
indexer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = indexer
spec.loader.exec_module(indexer)

build_amass_segment_index = indexer.build_amass_segment_index
build_amass_segment_index_from_paths = indexer.build_amass_segment_index_from_paths
read_amass_motion_info = indexer.read_amass_motion_info
read_amass_segment_index = indexer.read_amass_segment_index
write_amass_segment_index = indexer.write_amass_segment_index


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
            [
                np.ones((frames, bodies, 1), dtype=np.float32),
                np.zeros((frames, bodies, 3), dtype=np.float32),
            ],
            axis=-1,
        ),
        body_lin_vel_w=np.full((frames, bodies, 3), 0.2, dtype=np.float32),
        body_ang_vel_w=np.full((frames, bodies, 3), 0.3, dtype=np.float32),
    )


def test_indexer_builds_semantic_segments_and_writes_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        _write_fake_amass_npz(root / "KIT" / "359" / "motion_a.npz", frames=6, fps=30)
        _write_fake_amass_npz(root / "CMU" / "001" / "motion_b.npz", frames=4, fps=60)

        info = read_amass_motion_info(root, root / "KIT" / "359" / "motion_a.npz")
        print(
            "[cache_indexer trace] motion_info "
            f"rel_path={info.rel_path} frames={info.num_frames} fps={info.fps} "
            f"dofs={info.num_dofs} bodies={info.num_bodies}"
        )
        assert info.rel_path == "KIT/359/motion_a.npz"
        assert info.num_frames == 6
        assert info.fps == 30.0
        assert info.num_dofs == 29
        assert info.num_bodies == 30

        segments, summary = build_amass_segment_index(root, horizon_k=2, frame_stride=2)
        starts_by_motion: dict[str, list[int]] = {}
        for segment in segments:
            starts_by_motion.setdefault(segment.motion_rel_path, []).append(segment.start_frame)
        print(
            "[cache_indexer trace] built_index "
            f"motion_count={summary.motion_count} segment_count={summary.segment_count} "
            f"starts={starts_by_motion} ids={[segment.segment_id for segment in segments]}"
        )
        assert summary.motion_count == 2
        assert summary.segment_count == 3
        assert starts_by_motion["CMU/001/motion_b.npz"] == [0]
        assert starts_by_motion["KIT/359/motion_a.npz"] == [0, 2]
        assert [segment.segment_id for segment in segments] == list(range(3))
        assert all(segment.start_frame + segment.horizon_k < segment.motion_num_frames for segment in segments)

        out_dir = Path(tmp) / "cache_index"
        write_amass_segment_index(out_dir, segments, summary)
        loaded = read_amass_segment_index(out_dir / "segment_index.jsonl")
        with (out_dir / "metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(
            "[cache_indexer trace] reload "
            f"loaded_count={len(loaded)} first={loaded[0].motion_rel_path}:{loaded[0].start_frame} "
            f"metadata_format={metadata['format']}"
        )
        assert len(loaded) == len(segments)
        assert [item.segment_id for item in loaded] == [item.segment_id for item in segments]
        assert [item.motion_rel_path for item in loaded] == [item.motion_rel_path for item in segments]
        assert [item.start_frame for item in loaded] == [item.start_frame for item in segments]
        assert metadata["format"] == "frontres_segment_cache_index_v1"
        assert metadata["segment_count"] == 3


def test_indexer_respects_max_segments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        _write_fake_amass_npz(root / "KIT" / "359" / "motion_a.npz", frames=8)
        segments, summary = build_amass_segment_index(root, horizon_k=2, frame_stride=1, max_segments=3)
        print(
            "[cache_indexer trace] max_segments "
            f"segment_count={summary.segment_count} starts={[segment.start_frame for segment in segments]}"
        )
        assert len(segments) == 3
        assert summary.segment_count == 3
        assert [segment.start_frame for segment in segments] == [0, 1, 2]


def test_indexer_can_follow_loaded_motion_paths_instead_of_disk_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        motion_a = root / "AAA" / "motion_a.npz"
        motion_b = root / "ZZZ" / "motion_b.npz"
        _write_fake_amass_npz(motion_a, frames=8)
        _write_fake_amass_npz(motion_b, frames=7)

        segments, summary = build_amass_segment_index_from_paths(
            root,
            [motion_b],
            horizon_k=2,
            frame_stride=2,
            max_segments=2,
        )
        print(
            "[cache_indexer trace] loaded_paths_override "
            f"loaded={[str(motion_b.relative_to(root))]} "
            f"indexed={[segment.motion_rel_path for segment in segments]} "
            f"starts={[segment.start_frame for segment in segments]}"
        )
        assert summary.motion_count == 1
        assert summary.segment_count == 2
        assert [segment.motion_rel_path for segment in segments] == ["ZZZ/motion_b.npz", "ZZZ/motion_b.npz"]
        assert [segment.start_frame for segment in segments] == [0, 2]


def test_indexer_rejects_bad_motion_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "AMASS_G1NPZ_Final"
        bad = root / "KIT" / "359" / "bad_motion.npz"
        bad.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            bad,
            fps=np.array([30], dtype=np.int64),
            joint_pos=np.zeros((5, 29), dtype=np.float32),
            joint_vel=np.zeros((4, 29), dtype=np.float32),
            body_pos_w=np.zeros((5, 30, 3), dtype=np.float32),
            body_quat_w=np.zeros((5, 30, 4), dtype=np.float32),
            body_lin_vel_w=np.zeros((5, 30, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((5, 30, 3), dtype=np.float32),
        )
        try:
            read_amass_motion_info(root, bad)
        except ValueError as exc:
            print(f"[cache_indexer trace] rejected_bad_shape={exc}")
            assert "joint_vel shape" in str(exc)
            return
        raise AssertionError("bad AMASS motion shape should be rejected")


if __name__ == "__main__":
    test_indexer_builds_semantic_segments_and_writes_jsonl()
    test_indexer_respects_max_segments()
    test_indexer_can_follow_loaded_motion_paths_instead_of_disk_order()
    test_indexer_rejects_bad_motion_shape()
    print("PASS: FrontRES AMASS indexer builds segment index from motion paths and frame counts.")
