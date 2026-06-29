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
    segments: list[FrontRESSegmentIndex] = []
    skipped_short = 0
    motion_count = 0
    for path in paths:
        info = read_amass_motion_info(root, path)
        motion_count += 1
        if info.num_frames <= horizon_k:
            skipped_short += 1
            continue
        for start_frame in range(0, info.num_frames - horizon_k, frame_stride):
            segment = FrontRESSegmentIndex(
                segment_id=len(segments),
                motion_rel_path=info.rel_path,
                motion_num_frames=info.num_frames,
                fps=info.fps,
                start_frame=start_frame,
                horizon_k=int(horizon_k),
            )
            segment.validate()
            segments.append(segment)
            if max_segments is not None and len(segments) >= int(max_segments):
                summary = FrontRESAMASSIndexSummary(
                    amass_root=str(root),
                    motion_count=motion_count,
                    segment_count=len(segments),
                    horizon_k=int(horizon_k),
                    frame_stride=int(frame_stride),
                    skipped_short_motions=skipped_short,
                )
                return segments, summary
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
