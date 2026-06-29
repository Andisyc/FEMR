from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch


def _load_same_dir(module_name: str):
    path = Path(__file__).with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    from rsl_rl.frontres.frontres_segment_cache_extractor import extract_robot_rollout_state
    from rsl_rl.frontres.frontres_segment_cache_indexer import (
        build_amass_segment_index,
        build_amass_segment_index_from_paths,
        write_amass_segment_index,
    )
    from rsl_rl.frontres.frontres_segment_cache_io import (
        FrontRESCleanStateEntry,
        write_cache_metadata,
        write_clean_state_shard,
        write_noisy_variant_shard,
    )
    from rsl_rl.frontres.frontres_segment_cache_noisy_capture import capture_noisy_variant
    from rsl_rl.frontres.frontres_segment_cache_perturbation import (
        FrontRESPerturbationCurriculumConfig,
        build_perturbation_descriptors,
    )
    from rsl_rl.frontres.frontres_segment_cache_schema import FrontRESSegmentIndex
except ModuleNotFoundError:
    _extractor = _load_same_dir("frontres_segment_cache_extractor")
    _indexer = _load_same_dir("frontres_segment_cache_indexer")
    _cache_io = _load_same_dir("frontres_segment_cache_io")
    _noisy_capture = _load_same_dir("frontres_segment_cache_noisy_capture")
    _perturbation = _load_same_dir("frontres_segment_cache_perturbation")
    _schema = _load_same_dir("frontres_segment_cache_schema")
    extract_robot_rollout_state = _extractor.extract_robot_rollout_state
    build_amass_segment_index = _indexer.build_amass_segment_index
    build_amass_segment_index_from_paths = _indexer.build_amass_segment_index_from_paths
    write_amass_segment_index = _indexer.write_amass_segment_index
    FrontRESCleanStateEntry = _cache_io.FrontRESCleanStateEntry
    write_cache_metadata = _cache_io.write_cache_metadata
    write_clean_state_shard = _cache_io.write_clean_state_shard
    write_noisy_variant_shard = _cache_io.write_noisy_variant_shard
    capture_noisy_variant = _noisy_capture.capture_noisy_variant
    FrontRESPerturbationCurriculumConfig = _perturbation.FrontRESPerturbationCurriculumConfig
    build_perturbation_descriptors = _perturbation.build_perturbation_descriptors
    FrontRESSegmentIndex = _schema.FrontRESSegmentIndex


@dataclass(frozen=True)
class FrontRESStage1CacheBuilderConfig:
    amass_root: str
    cache_dir: str
    horizon_k: int
    frame_stride: int = 1
    max_motions: int | None = None
    max_segments: int | None = None
    strengths: tuple[float, ...] = (0.0, 0.5)
    variants_per_strength: int = 1
    base_seed: int = 0
    env_id: int = 0
    robot_name: str = "robot"
    clean_shard_id: int = 0
    noisy_shard_id: int = 0

    def validate(self) -> None:
        if int(self.horizon_k) <= 0:
            raise ValueError(f"horizon_k must be positive, got {self.horizon_k}")
        if int(self.frame_stride) <= 0:
            raise ValueError(f"frame_stride must be positive, got {self.frame_stride}")
        if not self.strengths:
            raise ValueError("strengths must be non-empty")
        if any(float(strength) < 0.0 for strength in self.strengths):
            raise ValueError(f"strengths must be non-negative, got {self.strengths}")
        if int(self.variants_per_strength) <= 0:
            raise ValueError(f"variants_per_strength must be positive, got {self.variants_per_strength}")
        if int(self.env_id) < 0:
            raise ValueError(f"env_id must be non-negative, got {self.env_id}")


@dataclass(frozen=True)
class FrontRESStage1CacheBuildResult:
    cache_dir: str
    segment_count: int
    clean_count: int
    noisy_count: int
    strength_counts: dict[float, int]
    segment_index_path: str
    clean_shard_path: str
    noisy_shard_paths: dict[float, str]
    metadata_path: str

    def probe(self) -> dict[str, Any]:
        return {
            "cache_dir": self.cache_dir,
            "segment_count": self.segment_count,
            "clean_count": self.clean_count,
            "noisy_count": self.noisy_count,
            "strength_counts": dict(self.strength_counts),
            "segment_index_path": self.segment_index_path,
            "clean_shard_path": self.clean_shard_path,
            "noisy_shard_paths": dict(self.noisy_shard_paths),
            "metadata_path": self.metadata_path,
        }


def build_stage1_segment_cache(env: Any, cfg: FrontRESStage1CacheBuilderConfig) -> FrontRESStage1CacheBuildResult:
    cfg.validate()
    cache_dir = Path(cfg.cache_dir)
    loaded_motion_paths = _frontres_loaded_motion_paths(env)
    if loaded_motion_paths:
        segments, index_summary = build_amass_segment_index_from_paths(
            cfg.amass_root,
            loaded_motion_paths,
            horizon_k=int(cfg.horizon_k),
            frame_stride=int(cfg.frame_stride),
            max_segments=cfg.max_segments,
        )
    else:
        segments, index_summary = build_amass_segment_index(
            cfg.amass_root,
            horizon_k=int(cfg.horizon_k),
            frame_stride=int(cfg.frame_stride),
            max_motions=cfg.max_motions,
            max_segments=cfg.max_segments,
        )
    print(
        "[FrontRES Stage1 Segment Cache] index_source "
        f"loaded_motion_count={len(loaded_motion_paths)} "
        f"indexed_motion_count={index_summary.motion_count} "
        f"segment_count={len(segments)} "
        f"first_loaded_motion={str(loaded_motion_paths[0]) if loaded_motion_paths else 'disk_scan'} "
        f"first_segment_motion={segments[0].motion_rel_path if segments else 'none'}",
        flush=True,
    )
    env_ids = torch.tensor([int(cfg.env_id)], dtype=torch.long)
    _ensure_frontres_env_reset(env)
    clean_entries = []
    clean_by_segment: dict[int, Any] = {}
    for segment in segments:
        prepare_clean_segment(env, segment=segment, env_ids=env_ids)
        clean_state = extract_robot_rollout_state(env, env_ids=env_ids, robot_name=cfg.robot_name)
        entry = FrontRESCleanStateEntry(segment=segment, clean_state=clean_state)
        entry.validate()
        clean_entries.append(entry)
        clean_by_segment[int(segment.segment_id)] = clean_state

    perturb_cfg = FrontRESPerturbationCurriculumConfig(
        strengths=tuple(float(item) for item in cfg.strengths),
        variants_per_strength=int(cfg.variants_per_strength),
        base_seed=int(cfg.base_seed),
        duration=int(cfg.horizon_k),
    )
    descriptors = build_perturbation_descriptors(segments, perturb_cfg)
    noisy_by_strength: dict[float, list[Any]] = {float(strength): [] for strength in cfg.strengths}
    segment_by_id = {int(segment.segment_id): segment for segment in segments}
    for descriptor in descriptors:
        segment = segment_by_id[int(descriptor.segment_id)]
        clean_state = clean_by_segment[int(descriptor.segment_id)]
        capture = capture_noisy_variant(
            env,
            segment=segment,
            clean_state=clean_state,
            descriptor=descriptor,
            env_ids=env_ids,
            robot_name=cfg.robot_name,
        )
        noisy_by_strength[float(descriptor.strength)].append(capture.variant)

    write_amass_segment_index(cache_dir, segments, index_summary)
    clean_shard_path = write_clean_state_shard(cache_dir, clean_entries, shard_id=int(cfg.clean_shard_id))
    noisy_shard_paths: dict[float, str] = {}
    for strength, variants in noisy_by_strength.items():
        noisy_path = write_noisy_variant_shard(
            cache_dir,
            variants,
            strength=float(strength),
            shard_id=int(cfg.noisy_shard_id),
        )
        noisy_shard_paths[float(strength)] = str(noisy_path)
    metadata_path = write_cache_metadata(
        cache_dir,
        {
            "stage": "stage1_segment_cache",
            "amass_root": str(Path(cfg.amass_root).resolve()),
            "segment_count": len(segments),
            "clean_count": len(clean_entries),
            "noisy_count": sum(len(items) for items in noisy_by_strength.values()),
            "horizon_k": int(cfg.horizon_k),
            "frame_stride": int(cfg.frame_stride),
            "strengths": [float(item) for item in cfg.strengths],
            "variants_per_strength": int(cfg.variants_per_strength),
            "base_seed": int(cfg.base_seed),
        },
    )
    return FrontRESStage1CacheBuildResult(
        cache_dir=str(cache_dir),
        segment_count=len(segments),
        clean_count=len(clean_entries),
        noisy_count=sum(len(items) for items in noisy_by_strength.values()),
        strength_counts={float(strength): len(items) for strength, items in noisy_by_strength.items()},
        segment_index_path=str(cache_dir / "segment_index.jsonl"),
        clean_shard_path=str(clean_shard_path),
        noisy_shard_paths=noisy_shard_paths,
        metadata_path=str(metadata_path),
    )


def prepare_clean_segment(env: Any, *, segment: FrontRESSegmentIndex, env_ids: torch.Tensor) -> torch.Tensor:
    segment.validate()
    for name in ("prepare_frontres_clean_segment", "rollout_frontres_clean_to_segment"):
        if hasattr(env, name):
            result = getattr(env, name)(segment=segment, env_ids=env_ids)
            return _success_tensor(result, env_ids)
    raise AttributeError("env must define prepare_frontres_clean_segment or rollout_frontres_clean_to_segment")


def _success_tensor(result: Any, env_ids: torch.Tensor) -> torch.Tensor:
    if result is None:
        return torch.ones(env_ids.numel(), dtype=torch.bool, device=env_ids.device)
    if isinstance(result, torch.Tensor):
        return result.to(device=env_ids.device).bool().flatten()
    if isinstance(result, dict):
        for name in ("success", "success_mask", "clean_success"):
            if name in result:
                return result[name].to(device=env_ids.device).bool().flatten()
    raise TypeError(f"unsupported clean prepare hook result type: {type(result)!r}")


def _frontres_loaded_motion_paths(env: Any) -> list[str]:
    for owner in (env, getattr(env, "unwrapped", None), getattr(env, "base_env", None)):
        if owner is None:
            continue
        for name in ("frontres_loaded_motion_paths", "get_frontres_loaded_motion_paths"):
            fn = getattr(owner, name, None)
            if callable(fn):
                paths = [str(path) for path in fn()]
                if paths:
                    return paths
        command = getattr(owner, "command", None)
        loader = getattr(command, "motion_dir_loader", None)
        paths = list(getattr(loader, "motion_paths", []) or [])
        if paths:
            return [str(path) for path in paths]
    return []


def _ensure_frontres_env_reset(env: Any) -> None:
    for owner in (env, getattr(env, "unwrapped", None), getattr(env, "base_env", None)):
        if owner is None:
            continue
        fn = getattr(owner, "ensure_frontres_env_reset", None)
        if callable(fn):
            fn()
            return
