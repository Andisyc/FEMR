#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FRONTRES_SOURCE = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres"
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
class Stage1CompletionProbe:
    amass_root: str
    cache_dir: str
    expected_motion_count: int
    expected_segment_count: int
    cached_segment_count: int
    metadata_segment_count: int
    metadata_clean_count: int
    metadata_noisy_count: int
    expected_noisy_count: int | None
    missing_segment_count: int
    extra_segment_count: int
    clean_shard_exists: bool
    noisy_shard_count: int
    expected_noisy_shard_count: int
    deep_validation_ok: bool
    deep_validation_status: str
    status: str

    def format_lines(self) -> list[str]:
        return [
            "[FrontRES Stage1 Completion] "
            f"amass_root={self.amass_root} cache_dir={self.cache_dir}",
            "[FrontRES Stage1 Completion] "
            f"expected_motion_count={self.expected_motion_count} "
            f"expected_segment_count={self.expected_segment_count} "
            f"cached_segment_count={self.cached_segment_count}",
            "[FrontRES Stage1 Completion] "
            f"metadata_segment_count={self.metadata_segment_count} "
            f"metadata_clean_count={self.metadata_clean_count} "
            f"metadata_noisy_count={self.metadata_noisy_count} "
            f"expected_noisy_count={self.expected_noisy_count}",
            "[FrontRES Stage1 Completion] "
            f"missing_segment_count={self.missing_segment_count} "
            f"extra_segment_count={self.extra_segment_count} "
            f"clean_shard_exists={self.clean_shard_exists} "
            f"noisy_shard_count={self.noisy_shard_count}/{self.expected_noisy_shard_count} "
            f"deep_validation_status={self.deep_validation_status}",
            f"{self.status}: FrontRES Stage 1 segment cache completion check",
        ]


def check_stage1_segment_cache_completion(
    amass_root: str | Path,
    cache_dir: str | Path,
    *,
    horizon_k: int,
    frame_stride: int,
    max_motions: int | None,
    max_segments: int | None,
    expect_mode: str | None,
    deep_read_shards: bool | str,
    show_missing_limit: int,
) -> Stage1CompletionProbe:
    root = Path(amass_root).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    metadata = _read_metadata(cache)
    cached_keys = set(_read_segment_index_keys(cache / "segment_index.jsonl"))
    expected_keys, expected_motion_count = _build_expected_segment_keys(
        root,
        horizon_k=int(horizon_k),
        frame_stride=int(frame_stride),
        max_motions=max_motions,
        max_segments=max_segments,
    )
    missing = sorted(expected_keys - cached_keys)
    extra = sorted(cached_keys - expected_keys)

    clean_shard_id = int(metadata.get("clean_shard_id", 0))
    noisy_shard_id = int(metadata.get("noisy_shard_id", 0))
    strengths = [float(item) for item in metadata.get("strengths", [])]
    clean_path = cache / "manifests" / "clean_states" / f"shard_{clean_shard_id:06d}.pt"
    noisy_paths = [
        cache / "manifests" / "noisy_variants" / _strength_dir(strength) / f"shard_{noisy_shard_id:06d}.pt"
        for strength in strengths
    ]
    expected_noisy_count = _expected_noisy_count(metadata, len(expected_keys))
    deep_mode = _parse_deep_mode(deep_read_shards)
    deep_validation_ok, deep_validation_status = _run_deep_validation(cache, expect_mode=expect_mode, mode=deep_mode)

    failures: list[str] = []
    _require(metadata.get("stage") == "stage1_segment_cache", "metadata.stage is not stage1_segment_cache", failures)
    if expect_mode is not None:
        _require(
            metadata.get("perturbation_curriculum_mode") == expect_mode,
            f"metadata perturbation mode mismatch: expected={expect_mode} observed={metadata.get('perturbation_curriculum_mode')}",
            failures,
        )
    _require(len(missing) == 0, _format_segment_delta("missing", missing, show_missing_limit), failures)
    _require(len(extra) == 0, _format_segment_delta("extra", extra, show_missing_limit), failures)
    _require(int(metadata.get("segment_count", -1)) == len(cached_keys), "metadata segment_count != segment_index count", failures)
    _require(int(metadata.get("clean_count", -1)) == len(cached_keys), "metadata clean_count != segment_index count", failures)
    if expected_noisy_count is not None:
        _require(
            int(metadata.get("noisy_count", -1)) == expected_noisy_count,
            f"metadata noisy_count != expected noisy count {expected_noisy_count}",
            failures,
        )
    _require(clean_path.is_file(), f"missing clean shard: {clean_path}", failures)
    _require(all(path.is_file() for path in noisy_paths), "one or more noisy shards are missing", failures)
    _require(deep_validation_ok, "deep shard validation failed", failures)

    status = "PASS" if not failures else "INCOMPLETE"
    probe = Stage1CompletionProbe(
        amass_root=str(root),
        cache_dir=str(cache),
        expected_motion_count=int(expected_motion_count),
        expected_segment_count=len(expected_keys),
        cached_segment_count=len(cached_keys),
        metadata_segment_count=int(metadata.get("segment_count", -1)),
        metadata_clean_count=int(metadata.get("clean_count", -1)),
        metadata_noisy_count=int(metadata.get("noisy_count", -1)),
        expected_noisy_count=expected_noisy_count,
        missing_segment_count=len(missing),
        extra_segment_count=len(extra),
        clean_shard_exists=clean_path.is_file(),
        noisy_shard_count=sum(1 for path in noisy_paths if path.is_file()),
        expected_noisy_shard_count=len(noisy_paths),
        deep_validation_ok=deep_validation_ok,
        deep_validation_status=deep_validation_status,
        status=status,
    )
    for failure in failures:
        print(f"[FrontRES Stage1 Completion] failure={failure}", flush=True)
    return probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare AMASS_G1NPZ_Final with AMASS_G1Segment and report whether Stage 1 cache finished."
    )
    parser.add_argument("--amass-root", default="/hdd1/cyx/AMASS_G1NPZ_Final")
    parser.add_argument("--cache-dir", default="/hdd1/cyx/AMASS_G1Segment")
    parser.add_argument("--horizon-k", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-motions", default="all")
    parser.add_argument("--max-segments", default="all")
    parser.add_argument("--expect-mode", default="hrl_curriculum_bank")
    parser.add_argument(
        "--deep-shard-read",
        choices=("auto", "always", "never"),
        default="auto",
        help="Read .pt clean/noisy shards through the torch validator when available.",
    )
    parser.add_argument("--skip-deep-shard-read", action="store_true", default=False)
    parser.add_argument("--show-missing-limit", type=int, default=8)
    args = parser.parse_args(argv)
    probe = check_stage1_segment_cache_completion(
        args.amass_root,
        args.cache_dir,
        horizon_k=args.horizon_k,
        frame_stride=args.frame_stride,
        max_motions=_parse_limit(args.max_motions),
        max_segments=_parse_limit(args.max_segments),
        expect_mode=args.expect_mode or None,
        deep_read_shards="never" if args.skip_deep_shard_read else args.deep_shard_read,
        show_missing_limit=max(0, int(args.show_missing_limit)),
    )
    for line in probe.format_lines():
        print(line, flush=True)
    return 0 if probe.status == "PASS" else 2


def _read_metadata(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "metadata.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"metadata must be a JSON object: {path}")
    return data


def _read_segment_index_keys(index_path: Path) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            result.append((str(record["motion_rel_path"]), int(record["start_frame"]), int(record["horizon_k"])))
    if not result:
        raise ValueError(f"segment index is empty: {index_path}")
    return result


def _build_expected_segment_keys(
    amass_root: Path,
    *,
    horizon_k: int,
    frame_stride: int,
    max_motions: int | None,
    max_segments: int | None,
) -> tuple[set[tuple[str, int, int]], int]:
    if horizon_k <= 0:
        raise ValueError(f"horizon_k must be positive, got {horizon_k}")
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    motion_paths = _discover_npz_files(amass_root, max_motions=max_motions)
    segments_by_motion: list[list[tuple[str, int, int]]] = []
    for path in motion_paths:
        rel_path, num_frames = _read_motion_frame_count(amass_root, path)
        if num_frames <= horizon_k:
            continue
        segments_by_motion.append(
            [(rel_path, start_frame, int(horizon_k)) for start_frame in range(0, num_frames - horizon_k, frame_stride)]
        )
    if max_segments is None:
        selected = [segment for motion_segments in segments_by_motion for segment in motion_segments]
    else:
        selected = _select_segments_round_robin(segments_by_motion, max_segments=max_segments)
    if not selected:
        raise ValueError(f"no valid expected segments built from {amass_root} with horizon_k={horizon_k}")
    return set(selected), len(motion_paths)


def _discover_npz_files(amass_root: Path, *, max_motions: int | None) -> list[Path]:
    if not amass_root.is_dir():
        raise FileNotFoundError(f"AMASS root does not exist or is not a directory: {amass_root}")
    files = sorted(path for path in amass_root.rglob("*.npz") if path.is_file())
    if max_motions is not None:
        files = files[: max(0, int(max_motions))]
    if not files:
        raise ValueError(f"no .npz motion files found under {amass_root}")
    return files


def _read_motion_frame_count(amass_root: Path, motion_path: Path) -> tuple[str, int]:
    rel_path = motion_path.resolve().relative_to(amass_root.resolve()).as_posix()
    with np.load(motion_path, allow_pickle=False) as data:
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
    return rel_path, int(joint_pos_shape[0])


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
    if body_quat_shape != (body_pos_shape[0], body_pos_shape[1], 4):
        raise ValueError(f"{rel_path}: body_quat_w shape {body_quat_shape} must match body_pos frame/body dims")
    if body_lin_vel_shape != (body_pos_shape[0], body_pos_shape[1], 3):
        raise ValueError(f"{rel_path}: body_lin_vel_w shape {body_lin_vel_shape} must match body_pos frame/body dims")
    if body_ang_vel_shape != (body_pos_shape[0], body_pos_shape[1], 3):
        raise ValueError(f"{rel_path}: body_ang_vel_w shape {body_ang_vel_shape} must match body_pos frame/body dims")
    if joint_pos_shape[0] != body_pos_shape[0]:
        raise ValueError(f"{rel_path}: joint and body frame counts differ")


def _select_segments_round_robin(
    segments_by_motion: list[list[tuple[str, int, int]]], *, max_segments: int
) -> list[tuple[str, int, int]]:
    if max_segments <= 0:
        return []
    quotas = _round_robin_motion_quotas(segments_by_motion, max_segments=max_segments)
    sampled_by_motion = [
        _select_segments_uniformly(motion_segments, count=quota)
        for motion_segments, quota in zip(segments_by_motion, quotas, strict=False)
    ]
    selected: list[tuple[str, int, int]] = []
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


def _round_robin_motion_quotas(segments_by_motion: list[list[tuple[str, int, int]]], *, max_segments: int) -> list[int]:
    quotas = [0 for _ in segments_by_motion]
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
    motion_segments: list[tuple[str, int, int]], *, count: int
) -> list[tuple[str, int, int]]:
    if count <= 0:
        return []
    if count >= len(motion_segments):
        return list(motion_segments)
    if count == 1:
        return [motion_segments[0]]
    last = len(motion_segments) - 1
    indices = [round(i * last / (count - 1)) for i in range(count)]
    return [motion_segments[int(index)] for index in indices]


def _run_deep_validation(cache_dir: Path, *, expect_mode: str | None, mode: str) -> tuple[bool, str]:
    if mode == "never":
        return True, "skipped"
    try:
        validator = _load_frontres_module("frontres_segment_cache_validator")
        result = validator.validate_stage1_cache_artifacts(cache_dir)
        if expect_mode is not None and result.perturbation_curriculum_mode != expect_mode:
            return False, "failed_mode_mismatch"
        return True, "passed"
    except Exception as exc:
        print(f"[FrontRES Stage1 Completion] deep_validation_error={exc}", flush=True)
        if mode == "auto":
            return True, "skipped_unavailable"
        return False, "failed"


def _load_frontres_module(module_name: str):
    path = FRONTRES_SOURCE / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expected_noisy_count(metadata: dict[str, Any], segment_count: int) -> int | None:
    variants_per_strength = int(metadata.get("variants_per_strength", 1))
    mode = str(metadata.get("perturbation_curriculum_mode", ""))
    if mode == "hrl_curriculum_bank":
        record_count = metadata.get("curriculum_bank_record_count")
        if record_count is None:
            return None
        return int(segment_count) * int(record_count) * variants_per_strength
    strengths = metadata.get("strengths")
    if strengths is None:
        return None
    return int(segment_count) * len(list(strengths)) * variants_per_strength


def _format_segment_delta(name: str, delta: list[tuple[str, int, int]], limit: int) -> str:
    if not delta:
        return ""
    sample = delta[:limit]
    suffix = "" if len(delta) <= limit else f"...(+{len(delta) - limit})"
    return f"{name}_segments={sample}{suffix}"


def _parse_limit(value: str | int | None) -> int | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"", "all", "auto", "full", "none"}:
        return None
    limit = int(raw)
    return limit if limit > 0 else None


def _parse_deep_mode(value: bool | str) -> str:
    if isinstance(value, bool):
        return "always" if value else "never"
    mode = str(value).strip().lower()
    if mode in {"1", "true", "yes"}:
        return "always"
    if mode in {"0", "false", "no"}:
        return "never"
    if mode not in {"auto", "always", "never"}:
        raise ValueError(f"deep_read_shards must be auto, always, or never; got {value!r}")
    return mode


def _strength_dir(strength: float) -> str:
    text = f"{float(strength):.6f}".rstrip("0").rstrip(".")
    return "strength_" + text.replace("-", "neg_").replace(".", "p")


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


if __name__ == "__main__":
    raise SystemExit(main())
