from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

try:
    from rsl_rl.frontres.frontres_segment_cache_schema import FrontRESSegmentIndex
except ModuleNotFoundError:
    _SCHEMA_PATH = Path(__file__).with_name("frontres_segment_cache_schema.py")
    _SCHEMA_SPEC = importlib.util.spec_from_file_location("frontres_segment_cache_schema", _SCHEMA_PATH)
    if _SCHEMA_SPEC is None or _SCHEMA_SPEC.loader is None:
        raise
    _SCHEMA_MODULE = importlib.util.module_from_spec(_SCHEMA_SPEC)
    sys.modules[_SCHEMA_SPEC.name] = _SCHEMA_MODULE
    _SCHEMA_SPEC.loader.exec_module(_SCHEMA_MODULE)
    FrontRESSegmentIndex = _SCHEMA_MODULE.FrontRESSegmentIndex


AMASS_G1_REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class FrontRESAMASSMotionInfo:
    rel_path: str
    num_frames: int
    fps: float
    num_dofs: int
    num_bodies: int

    def validate(self) -> None:
        if not self.rel_path:
            raise ValueError("rel_path must be non-empty")
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.num_dofs <= 0:
            raise ValueError(f"num_dofs must be positive, got {self.num_dofs}")
        if self.num_bodies <= 0:
            raise ValueError(f"num_bodies must be positive, got {self.num_bodies}")


@dataclass(frozen=True)
class FrontRESAMASSIndexSummary:
    amass_root: str
    motion_count: int
    segment_count: int
    horizon_k: int
    frame_stride: int
    skipped_short_motions: int

    def probe(self) -> dict[str, Any]:
        return {
            "amass_root": self.amass_root,
            "motion_count": self.motion_count,
            "segment_count": self.segment_count,
            "horizon_k": self.horizon_k,
            "frame_stride": self.frame_stride,
            "skipped_short_motions": self.skipped_short_motions,
        }


@dataclass(frozen=True)
class FrontRESAMASSSegmentIndexChunk:
    chunk_id: int
    segments: tuple[FrontRESSegmentIndex, ...]
    motion_count: int
    segment_count: int
    skipped_short_motions: int

    @property
    def chunk_segment_count(self) -> int:
        return len(self.segments)

    def probe(self) -> dict[str, Any]:
        segment_ids = [int(segment.segment_id) for segment in self.segments]
        return {
            "chunk_id": int(self.chunk_id),
            "chunk_segment_count": int(self.chunk_segment_count),
            "motion_count": int(self.motion_count),
            "segment_count": int(self.segment_count),
            "skipped_short_motions": int(self.skipped_short_motions),
            "segment_id_min": min(segment_ids) if segment_ids else None,
            "segment_id_max": max(segment_ids) if segment_ids else None,
        }


def discover_amass_npz_files(amass_root: str | Path, *, max_motions: int | None = None) -> list[Path]:
    root = Path(amass_root)
    if not root.is_dir():
        raise FileNotFoundError(f"AMASS root does not exist or is not a directory: {root}")
    files = sorted(path for path in root.rglob("*.npz") if path.is_file())
    if max_motions is not None:
        files = files[: max(0, int(max_motions))]
    if not files:
        raise ValueError(f"no .npz motion files found under {root}")
    return files


def read_amass_motion_info(amass_root: str | Path, motion_path: str | Path) -> FrontRESAMASSMotionInfo:
    root = Path(amass_root).resolve()
    path = Path(motion_path).resolve()
    rel_path = path.relative_to(root).as_posix()
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in AMASS_G1_REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"{rel_path} is missing AMASS G1 keys: {missing}")
        joint_pos_shape = tuple(data["joint_pos"].shape)
        joint_vel_shape = tuple(data["joint_vel"].shape)
        body_pos_shape = tuple(data["body_pos_w"].shape)
        body_quat_shape = tuple(data["body_quat_w"].shape)
        body_lin_vel_shape = tuple(data["body_lin_vel_w"].shape)
        body_ang_vel_shape = tuple(data["body_ang_vel_w"].shape)
        _validate_motion_shapes(
            rel_path,
            joint_pos_shape=joint_pos_shape,
            joint_vel_shape=joint_vel_shape,
            body_pos_shape=body_pos_shape,
            body_quat_shape=body_quat_shape,
            body_lin_vel_shape=body_lin_vel_shape,
            body_ang_vel_shape=body_ang_vel_shape,
        )
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    info = FrontRESAMASSMotionInfo(
        rel_path=rel_path,
        num_frames=int(joint_pos_shape[0]),
        fps=fps,
        num_dofs=int(joint_pos_shape[1]),
        num_bodies=int(body_pos_shape[1]),
    )
    info.validate()
    return info


def build_amass_segment_index(
    amass_root: str | Path,
    *,
    horizon_k: int,
    frame_stride: int = 1,
    max_motions: int | None = None,
    max_segments: int | None = None,
) -> tuple[list[FrontRESSegmentIndex], FrontRESAMASSIndexSummary]:
    return build_amass_segment_index_from_paths(
        amass_root,
        discover_amass_npz_files(amass_root, max_motions=max_motions),
        horizon_k=horizon_k,
        frame_stride=frame_stride,
        max_segments=max_segments,
    )


def iter_amass_segment_index_chunks_from_paths(
    amass_root: str | Path,
    motion_paths: Iterable[str | Path],
    *,
    horizon_k: int,
    frame_stride: int = 1,
    chunk_size: int = 128,
    max_segments: int | None = None,
) -> Iterable[FrontRESAMASSSegmentIndexChunk]:
    if horizon_k <= 0:
        raise ValueError(f"horizon_k must be positive, got {horizon_k}")
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if max_segments is not None and int(max_segments) <= 0:
        raise ValueError(f"max_segments must be positive when provided, got {max_segments}")
    root = Path(amass_root).resolve()
    chunk: list[FrontRESSegmentIndex] = []
    chunk_id = 0
    motion_count = 0
    skipped_short = 0
    segment_id = 0
    saw_path = False
    for motion_path in motion_paths:
        saw_path = True
        info = read_amass_motion_info(root, Path(motion_path).expanduser().resolve())
        motion_count += 1
        if info.num_frames <= horizon_k:
            skipped_short += 1
            continue
        for start_frame in range(0, info.num_frames - horizon_k, frame_stride):
            if max_segments is not None and segment_id >= int(max_segments):
                if chunk:
                    yield FrontRESAMASSSegmentIndexChunk(
                        chunk_id=int(chunk_id),
                        segments=tuple(chunk),
                        motion_count=int(motion_count),
                        segment_count=int(segment_id),
                        skipped_short_motions=int(skipped_short),
                    )
                return
            segment = FrontRESSegmentIndex(
                segment_id=int(segment_id),
                motion_rel_path=info.rel_path,
                motion_num_frames=info.num_frames,
                fps=info.fps,
                start_frame=start_frame,
                horizon_k=int(horizon_k),
            )
            segment.validate()
            chunk.append(segment)
            segment_id += 1
            if len(chunk) >= int(chunk_size):
                yield FrontRESAMASSSegmentIndexChunk(
                    chunk_id=int(chunk_id),
                    segments=tuple(chunk),
                    motion_count=int(motion_count),
                    segment_count=int(segment_id),
                    skipped_short_motions=int(skipped_short),
                )
                chunk.clear()
                chunk_id += 1
    if not saw_path:
        raise ValueError("motion_paths must contain at least one loaded AMASS motion")
    if chunk:
        yield FrontRESAMASSSegmentIndexChunk(
            chunk_id=int(chunk_id),
            segments=tuple(chunk),
            motion_count=int(motion_count),
            segment_count=int(segment_id),
            skipped_short_motions=int(skipped_short),
        )
    elif segment_id == 0:
        raise ValueError(f"no valid segments built from {root} with horizon_k={horizon_k}")


def build_amass_segment_index_from_paths(
    amass_root: str | Path,
    motion_paths: Iterable[str | Path],
    *,
    horizon_k: int,
    frame_stride: int = 1,
    max_segments: int | None = None,
) -> tuple[list[FrontRESSegmentIndex], FrontRESAMASSIndexSummary]:
    if horizon_k <= 0:
        raise ValueError(f"horizon_k must be positive, got {horizon_k}")
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    root = Path(amass_root).resolve()
    paths = [Path(path).expanduser().resolve() for path in motion_paths]
    if not paths:
        raise ValueError("motion_paths must contain at least one loaded AMASS motion")
    segments_by_motion: list[list[FrontRESSegmentIndex]] = []
    skipped_short = 0
    motion_count = 0
    for path in paths:
        info = read_amass_motion_info(root, path)
        motion_count += 1
        if info.num_frames <= horizon_k:
            skipped_short += 1
            continue
        motion_segments: list[FrontRESSegmentIndex] = []
        for start_frame in range(0, info.num_frames - horizon_k, frame_stride):
            segment = FrontRESSegmentIndex(
                segment_id=0,
                motion_rel_path=info.rel_path,
                motion_num_frames=info.num_frames,
                fps=info.fps,
                start_frame=start_frame,
                horizon_k=int(horizon_k),
            )
            segment.validate()
            motion_segments.append(segment)
        if motion_segments:
            segments_by_motion.append(motion_segments)
    if max_segments is None:
        selected = [segment for motion_segments in segments_by_motion for segment in motion_segments]
    else:
        selected = _select_segments_round_robin(segments_by_motion, max_segments=int(max_segments))
    segments = _renumber_segments(selected)
    if not segments:
        raise ValueError(f"no valid segments built from {root} with horizon_k={horizon_k}")
    summary = FrontRESAMASSIndexSummary(
        amass_root=str(root),
        motion_count=motion_count,
        segment_count=len(segments),
        horizon_k=int(horizon_k),
        frame_stride=int(frame_stride),
        skipped_short_motions=skipped_short,
    )
    return segments, summary


def _select_segments_round_robin(
    segments_by_motion: list[list[FrontRESSegmentIndex]], *, max_segments: int
) -> list[FrontRESSegmentIndex]:
    if max_segments <= 0:
        return []
    quotas = _round_robin_motion_quotas(segments_by_motion, max_segments=max_segments)
    sampled_by_motion = [
        _select_segments_uniformly(motion_segments, count=quota)
        for motion_segments, quota in zip(segments_by_motion, quotas, strict=False)
    ]
    selected: list[FrontRESSegmentIndex] = []
    cursor = 0
    while len(selected) < max_segments:
        added = False
        for motion_segments in sampled_by_motion:
            if cursor < len(motion_segments):
                selected.append(motion_segments[cursor])
                added = True
                if len(selected) >= max_segments:
                    break
        if not added:
            break
        cursor += 1
    return selected


def _round_robin_motion_quotas(segments_by_motion: list[list[FrontRESSegmentIndex]], *, max_segments: int) -> list[int]:
    quotas = [0 for _ in segments_by_motion]
    if max_segments <= 0:
        return quotas
    selected = 0
    while selected < max_segments:
        added = False
        for idx, motion_segments in enumerate(segments_by_motion):
            if quotas[idx] < len(motion_segments):
                quotas[idx] += 1
                selected += 1
                added = True
                if selected >= max_segments:
                    break
        if not added:
            break
    return quotas


def _select_segments_uniformly(
    motion_segments: list[FrontRESSegmentIndex], *, count: int
) -> list[FrontRESSegmentIndex]:
    if count <= 0:
        return []
    if count >= len(motion_segments):
        return list(motion_segments)
    if count == 1:
        return [motion_segments[0]]
    last = len(motion_segments) - 1
    indices = [round(i * last / (count - 1)) for i in range(count)]
    return [motion_segments[int(index)] for index in indices]


def _renumber_segments(segments: Iterable[FrontRESSegmentIndex]) -> list[FrontRESSegmentIndex]:
    result: list[FrontRESSegmentIndex] = []
    for idx, segment in enumerate(segments):
        updated = FrontRESSegmentIndex(
            segment_id=int(idx),
            motion_rel_path=segment.motion_rel_path,
            motion_num_frames=segment.motion_num_frames,
            fps=segment.fps,
            start_frame=segment.start_frame,
            horizon_k=segment.horizon_k,
        )
        updated.validate()
        result.append(updated)
    return result


def write_amass_segment_index(
    output_dir: str | Path,
    segments: Iterable[FrontRESSegmentIndex],
    summary: FrontRESAMASSIndexSummary,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    segment_path = path / "segment_index.jsonl"
    metadata_path = path / "metadata.json"
    segment_list = list(segments)
    with segment_path.open("w", encoding="utf-8") as f:
        for segment in segment_list:
            segment.validate()
            f.write(json.dumps(segment_to_record(segment), sort_keys=True) + "\n")
    metadata = {
        "format": "frontres_segment_cache_index_v1",
        **summary.probe(),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def append_amass_segment_index(
    output_dir: str | Path,
    segments: Iterable[FrontRESSegmentIndex],
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    segment_path = path / "segment_index.jsonl"
    segment_list = list(segments)
    if not segment_list:
        raise ValueError("segments must be non-empty when appending segment_index")
    with segment_path.open("a", encoding="utf-8") as f:
        for segment in segment_list:
            segment.validate()
            f.write(json.dumps(segment_to_record(segment), sort_keys=True) + "\n")
    return segment_path


def read_amass_segment_index(index_path: str | Path) -> list[FrontRESSegmentIndex]:
    path = Path(index_path)
    result: list[FrontRESSegmentIndex] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segment = segment_from_record(json.loads(line))
            segment.validate()
            result.append(segment)
    if not result:
        raise ValueError(f"segment index is empty: {path}")
    return result


def segment_to_record(segment: FrontRESSegmentIndex) -> dict[str, Any]:
    segment.validate()
    return {
        "segment_id": int(segment.segment_id),
        "motion_rel_path": str(segment.motion_rel_path),
        "motion_num_frames": int(segment.motion_num_frames),
        "fps": float(segment.fps),
        "start_frame": int(segment.start_frame),
        "horizon_k": int(segment.horizon_k),
    }


def segment_from_record(record: dict[str, Any]) -> FrontRESSegmentIndex:
    segment = FrontRESSegmentIndex(
        segment_id=int(record["segment_id"]),
        motion_rel_path=str(record["motion_rel_path"]),
        motion_num_frames=int(record["motion_num_frames"]),
        fps=float(record["fps"]),
        start_frame=int(record["start_frame"]),
        horizon_k=int(record["horizon_k"]),
    )
    segment.validate()
    return segment


def _validate_motion_shapes(
    rel_path: str,
    *,
    joint_pos_shape: tuple[int, ...],
    joint_vel_shape: tuple[int, ...],
    body_pos_shape: tuple[int, ...],
    body_quat_shape: tuple[int, ...],
    body_lin_vel_shape: tuple[int, ...],
    body_ang_vel_shape: tuple[int, ...],
) -> None:
    if len(joint_pos_shape) != 2:
        raise ValueError(f"{rel_path}: joint_pos must have shape [T, D], got {joint_pos_shape}")
    if joint_vel_shape != joint_pos_shape:
        raise ValueError(f"{rel_path}: joint_vel shape {joint_vel_shape} must match joint_pos {joint_pos_shape}")
    if len(body_pos_shape) != 3 or body_pos_shape[-1] != 3:
        raise ValueError(f"{rel_path}: body_pos_w must have shape [T, B, 3], got {body_pos_shape}")
    expected_body_quat = (body_pos_shape[0], body_pos_shape[1], 4)
    expected_body_vec = (body_pos_shape[0], body_pos_shape[1], 3)
    if body_quat_shape != expected_body_quat:
        raise ValueError(f"{rel_path}: body_quat_w shape {body_quat_shape} must be {expected_body_quat}")
    if body_lin_vel_shape != expected_body_vec:
        raise ValueError(f"{rel_path}: body_lin_vel_w shape {body_lin_vel_shape} must be {expected_body_vec}")
    if body_ang_vel_shape != expected_body_vec:
        raise ValueError(f"{rel_path}: body_ang_vel_w shape {body_ang_vel_shape} must be {expected_body_vec}")
    if joint_pos_shape[0] != body_pos_shape[0]:
        raise ValueError(f"{rel_path}: joint and body frame counts differ")
