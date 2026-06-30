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
        append_stage1_cache_progress,
        write_cache_metadata,
        write_clean_state_chunked_shard,
        write_clean_state_manifest_records,
        write_noisy_variant_chunked_shard,
        write_noisy_variant_manifest_records,
        write_stage1_cache_status,
    )
    from rsl_rl.frontres.frontres_segment_cache_noisy_capture import capture_noisy_variant
    from rsl_rl.frontres.frontres_segment_cache_perturbation import (
        FrontRESBankDescriptorConfig,
        FrontRESPerturbationCurriculumConfig,
        build_perturbation_descriptors,
        build_perturbation_descriptors_from_curriculum_bank,
        descriptor_probe,
    )
    from rsl_rl.frontres.frontres_segment_cache_schema import FrontRESSegmentIndex
    from rsl_rl.frontres.frontres_segment_cache_curriculum import (
        FrontRESStage1CurriculumBankConfig,
        build_stage1_curriculum_bank,
        stage1_curriculum_bank_probe,
    )
    from rsl_rl.frontres.frontres_segment_cache_validator import validate_stage1_cache_artifacts
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
    append_stage1_cache_progress = _cache_io.append_stage1_cache_progress
    write_cache_metadata = _cache_io.write_cache_metadata
    write_clean_state_chunked_shard = _cache_io.write_clean_state_chunked_shard
    write_clean_state_manifest_records = _cache_io.write_clean_state_manifest_records
    write_noisy_variant_chunked_shard = _cache_io.write_noisy_variant_chunked_shard
    write_noisy_variant_manifest_records = _cache_io.write_noisy_variant_manifest_records
    write_stage1_cache_status = _cache_io.write_stage1_cache_status
    capture_noisy_variant = _noisy_capture.capture_noisy_variant
    FrontRESBankDescriptorConfig = _perturbation.FrontRESBankDescriptorConfig
    FrontRESPerturbationCurriculumConfig = _perturbation.FrontRESPerturbationCurriculumConfig
    build_perturbation_descriptors = _perturbation.build_perturbation_descriptors
    build_perturbation_descriptors_from_curriculum_bank = (
        _perturbation.build_perturbation_descriptors_from_curriculum_bank
    )
    descriptor_probe = _perturbation.descriptor_probe
    FrontRESSegmentIndex = _schema.FrontRESSegmentIndex
    _cache_curriculum = _load_same_dir("frontres_segment_cache_curriculum")
    FrontRESStage1CurriculumBankConfig = _cache_curriculum.FrontRESStage1CurriculumBankConfig
    build_stage1_curriculum_bank = _cache_curriculum.build_stage1_curriculum_bank
    stage1_curriculum_bank_probe = _cache_curriculum.stage1_curriculum_bank_probe
    _cache_validator = _load_same_dir("frontres_segment_cache_validator")
    validate_stage1_cache_artifacts = _cache_validator.validate_stage1_cache_artifacts


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
    perturbation_curriculum_mode: str = "hrl_curriculum_bank"
    curriculum_bank_size: int = 16
    curriculum_frontier_scale: float = 2.0
    curriculum_dr_min: float = 1.25
    curriculum_dr_max: float = 4.5
    curriculum_progress: float = 0.8
    curriculum_seq_idx: int = 17
    curriculum_active_dims: tuple[int, ...] | None = (0, 1, 2, 3, 4, 5)
    curriculum_include_hard_as_train: bool = False
    curriculum_temporal_mode: str = "single"
    curriculum_burst_min_steps: int = 4
    curriculum_burst_max_steps: int = 8
    base_seed: int = 0
    env_id: int = 0
    robot_name: str = "robot"
    clean_shard_id: int = 0
    noisy_shard_id: int = 0
    cache_chunk_size: int = 128

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
        if self.perturbation_curriculum_mode not in {"discrete_bank", "hrl_curriculum_bank"}:
            raise ValueError(
                "perturbation_curriculum_mode must be discrete_bank or hrl_curriculum_bank, "
                f"got {self.perturbation_curriculum_mode}"
            )
        if int(self.curriculum_bank_size) <= 0:
            raise ValueError(f"curriculum_bank_size must be positive, got {self.curriculum_bank_size}")
        if float(self.curriculum_frontier_scale) < 0.0:
            raise ValueError(
                f"curriculum_frontier_scale must be non-negative, got {self.curriculum_frontier_scale}"
            )
        if float(self.curriculum_dr_min) < 0.0:
            raise ValueError(f"curriculum_dr_min must be non-negative, got {self.curriculum_dr_min}")
        if float(self.curriculum_dr_max) < float(self.curriculum_dr_min):
            raise ValueError(
                f"curriculum_dr_max must be >= curriculum_dr_min, got {self.curriculum_dr_max} < {self.curriculum_dr_min}"
            )
        if not (0.0 <= float(self.curriculum_progress) <= 1.0):
            raise ValueError(f"curriculum_progress must be in [0, 1], got {self.curriculum_progress}")
        if int(self.curriculum_burst_min_steps) <= 0:
            raise ValueError(f"curriculum_burst_min_steps must be positive, got {self.curriculum_burst_min_steps}")
        if int(self.curriculum_burst_max_steps) < int(self.curriculum_burst_min_steps):
            raise ValueError(
                "curriculum_burst_max_steps must be >= curriculum_burst_min_steps, "
                f"got {self.curriculum_burst_max_steps} < {self.curriculum_burst_min_steps}"
            )
        if int(self.env_id) < 0:
            raise ValueError(f"env_id must be non-negative, got {self.env_id}")
        if int(self.cache_chunk_size) <= 0:
            raise ValueError(f"cache_chunk_size must be positive, got {self.cache_chunk_size}")


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
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_stage1_cache_status(
        cache_dir,
        {
            "stage": "stage1_segment_cache",
            "status": "started",
            "clean_written": 0,
            "noisy_written": 0,
            "cache_dir": str(cache_dir),
            "amass_root": str(Path(cfg.amass_root).resolve()),
            "horizon_k": int(cfg.horizon_k),
            "frame_stride": int(cfg.frame_stride),
        },
    )
    append_stage1_cache_progress(
        cache_dir,
        {
            "event": "started",
            "clean_written": 0,
            "noisy_written": 0,
            "horizon_k": int(cfg.horizon_k),
            "frame_stride": int(cfg.frame_stride),
        },
    )
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
    write_amass_segment_index(cache_dir, segments, index_summary)
    write_stage1_cache_status(
        cache_dir,
        {
            "stage": "stage1_segment_cache",
            "status": "indexed",
            "segment_count": len(segments),
            "clean_written": 0,
            "noisy_written": 0,
            "cache_dir": str(cache_dir),
            "segment_index_path": str(cache_dir / "segment_index.jsonl"),
        },
    )
    append_stage1_cache_progress(
        cache_dir,
        {
            "event": "indexed",
            "segment_count": len(segments),
            "indexed_motion_count": int(index_summary.motion_count),
            "segment_index_path": str(cache_dir / "segment_index.jsonl"),
        },
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
    descriptors, perturbation_metadata = build_stage1_perturbation_plan(segments, cfg)
    descriptor_trace = descriptor_probe(descriptors)
    print(
        "[FrontRES Stage1 Segment Cache] perturbation_plan "
        f"mode={perturbation_metadata['perturbation_curriculum_mode']} "
        f"descriptor_count={descriptor_trace['count']} "
        f"families={descriptor_trace['families'][:8]} "
        f"mix_classes={descriptor_trace['mix_classes'][:8]} "
        f"roles={descriptor_trace['roles'][:8]} "
        f"strengths={descriptor_trace['strengths'][:8]}",
        flush=True,
    )
    clean_records: list[dict[str, Any]] = []
    noisy_records_by_strength: dict[float, list[dict[str, Any]]] = {
        float(strength): [] for strength in perturbation_metadata["strengths"]
    }
    descriptors_by_segment: dict[int, list[Any]] = {}
    for descriptor in descriptors:
        descriptors_by_segment.setdefault(int(descriptor.segment_id), []).append(descriptor)

    clean_written = 0
    noisy_written = 0
    strength_counts = {float(strength): 0 for strength in perturbation_metadata["strengths"]}
    chunk_size = int(cfg.cache_chunk_size)
    clean_buffer: list[FrontRESCleanStateEntry] = []
    clean_shard_id = int(cfg.clean_shard_id)
    noisy_buffers_by_strength: dict[float, list[Any]] = {
        float(strength): [] for strength in perturbation_metadata["strengths"]
    }
    noisy_shard_ids_by_strength: dict[float, int] = {
        float(strength): int(cfg.noisy_shard_id) for strength in perturbation_metadata["strengths"]
    }

    def flush_clean_buffer() -> str | None:
        nonlocal clean_shard_id
        if not clean_buffer:
            return None
        shard_path, records = write_clean_state_chunked_shard(
            cache_dir,
            clean_buffer,
            shard_id=clean_shard_id,
        )
        clean_records.extend(records)
        clean_buffer.clear()
        clean_shard_id += 1
        return str(shard_path)

    def flush_noisy_buffer(strength: float) -> str | None:
        buffer = noisy_buffers_by_strength.setdefault(float(strength), [])
        if not buffer:
            return None
        shard_id = noisy_shard_ids_by_strength.get(float(strength), int(cfg.noisy_shard_id))
        shard_path, records = write_noisy_variant_chunked_shard(
            cache_dir,
            buffer,
            strength=float(strength),
            shard_id=shard_id,
        )
        noisy_records_by_strength.setdefault(float(strength), []).extend(records)
        buffer.clear()
        noisy_shard_ids_by_strength[float(strength)] = shard_id + 1
        return str(shard_path)

    for segment in segments:
        prepare_clean_segment(env, segment=segment, env_ids=env_ids)
        clean_state = extract_robot_rollout_state(env, env_ids=env_ids, robot_name=cfg.robot_name)
        entry = FrontRESCleanStateEntry(segment=segment, clean_state=clean_state)
        entry.validate()
        clean_buffer.append(entry)
        clean_written += 1
        clean_shard_path = None
        if len(clean_buffer) >= chunk_size:
            clean_shard_path = flush_clean_buffer()
        write_stage1_cache_status(
            cache_dir,
            {
                "stage": "stage1_segment_cache",
                "status": "clean_capture",
                "segment_count": len(segments),
                "clean_written": clean_written,
                "noisy_written": noisy_written,
                "last_segment_id": int(segment.segment_id),
                "last_clean_shard_path": clean_shard_path,
                "clean_buffer_count": len(clean_buffer),
            },
        )
        append_stage1_cache_progress(
            cache_dir,
            {
                "event": "clean_done",
                "segment_id": int(segment.segment_id),
                "clean_written": clean_written,
                "segment_count": len(segments),
                "flushed_shard_path": clean_shard_path,
                "buffer_count": len(clean_buffer),
            },
        )

        for descriptor in descriptors_by_segment.get(int(segment.segment_id), []):
            capture = capture_noisy_variant(
                env,
                segment=segment,
                clean_state=clean_state,
                descriptor=descriptor,
                env_ids=env_ids,
                robot_name=cfg.robot_name,
            )
            strength = float(descriptor.strength)
            noisy_buffers_by_strength.setdefault(strength, []).append(capture.variant)
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
            noisy_written += 1
            noisy_shard_path = None
            if len(noisy_buffers_by_strength[strength]) >= chunk_size:
                noisy_shard_path = flush_noisy_buffer(strength)
            write_stage1_cache_status(
                cache_dir,
                {
                    "stage": "stage1_segment_cache",
                    "status": "noisy_capture",
                    "segment_count": len(segments),
                    "clean_written": clean_written,
                    "noisy_written": noisy_written,
                    "descriptor_count": len(descriptors),
                    "last_segment_id": int(descriptor.segment_id),
                    "last_perturbation_id": int(descriptor.perturbation_id),
                    "last_noisy_shard_path": noisy_shard_path,
                    "last_strength": strength,
                    "noisy_buffer_count": len(noisy_buffers_by_strength[strength]),
                },
            )
            append_stage1_cache_progress(
                cache_dir,
                {
                    "event": "noisy_done",
                    "segment_id": int(descriptor.segment_id),
                    "perturbation_id": int(descriptor.perturbation_id),
                    "strength": strength,
                    "noisy_written": noisy_written,
                    "descriptor_count": len(descriptors),
                    "flushed_shard_path": noisy_shard_path,
                    "buffer_count": len(noisy_buffers_by_strength[strength]),
                },
            )
            del capture
        del clean_state
        del entry

    flush_clean_buffer()
    for strength in list(noisy_buffers_by_strength):
        flush_noisy_buffer(float(strength))

    clean_shard_path = write_clean_state_manifest_records(cache_dir, clean_records, shard_id=int(cfg.clean_shard_id))
    noisy_shard_paths: dict[float, str] = {}
    for strength, records in noisy_records_by_strength.items():
        noisy_path = write_noisy_variant_manifest_records(
            cache_dir,
            records,
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
            "clean_count": clean_written,
            "noisy_count": noisy_written,
            "horizon_k": int(cfg.horizon_k),
            "frame_stride": int(cfg.frame_stride),
            **perturbation_metadata,
            "variants_per_strength": int(cfg.variants_per_strength),
            "base_seed": int(cfg.base_seed),
            "clean_shard_id": int(cfg.clean_shard_id),
            "noisy_shard_id": int(cfg.noisy_shard_id),
            "cache_storage_backend": "torch_chunked_shard",
            "cache_chunk_size": int(cfg.cache_chunk_size),
        },
    )
    validation = validate_stage1_cache_artifacts(cache_dir)
    validation_probe = validation.probe()
    print(
        "[FrontRES Stage1 Segment Cache] cache_readback "
        f"stage={validation_probe['metadata_stage']} "
        f"mode={validation_probe['perturbation_curriculum_mode']} "
        f"segment_count={validation_probe['segment_count']} "
        f"clean_count={validation_probe['clean_count']} "
        f"noisy_count={validation_probe['noisy_count']} "
        f"strength_counts={validation_probe['strength_counts']} "
        f"train_count={validation_probe['train_count']} "
        f"boundary_diagnostic_count={validation_probe['boundary_diagnostic_count']} "
        f"clean_shard_path={validation_probe['clean_shard_path']}",
        flush=True,
    )
    write_stage1_cache_status(
        cache_dir,
        {
            "stage": "stage1_segment_cache",
            "status": "complete",
            "segment_count": len(segments),
            "clean_written": clean_written,
            "noisy_written": noisy_written,
            "descriptor_count": len(descriptors),
            "segment_index_path": str(cache_dir / "segment_index.jsonl"),
            "clean_shard_path": str(clean_shard_path),
            "noisy_shard_paths": noisy_shard_paths,
            "metadata_path": str(metadata_path),
        },
    )
    append_stage1_cache_progress(
        cache_dir,
        {
            "event": "complete",
            "segment_count": len(segments),
            "clean_written": clean_written,
            "noisy_written": noisy_written,
            "metadata_path": str(metadata_path),
        },
    )
    return FrontRESStage1CacheBuildResult(
        cache_dir=str(cache_dir),
        segment_count=len(segments),
        clean_count=clean_written,
        noisy_count=noisy_written,
        strength_counts={float(strength): int(count) for strength, count in strength_counts.items()},
        segment_index_path=str(cache_dir / "segment_index.jsonl"),
        clean_shard_path=str(clean_shard_path),
        noisy_shard_paths=noisy_shard_paths,
        metadata_path=str(metadata_path),
    )


def build_stage1_perturbation_plan(
    segments: list[FrontRESSegmentIndex],
    cfg: FrontRESStage1CacheBuilderConfig,
) -> tuple[list[Any], dict[str, Any]]:
    if cfg.perturbation_curriculum_mode == "discrete_bank":
        perturb_cfg = FrontRESPerturbationCurriculumConfig(
            strengths=tuple(float(item) for item in cfg.strengths),
            variants_per_strength=int(cfg.variants_per_strength),
            base_seed=int(cfg.base_seed),
            duration=int(cfg.horizon_k),
        )
        descriptors = build_perturbation_descriptors(segments, perturb_cfg)
        metadata = {
            "strengths": [float(item) for item in cfg.strengths],
            "legacy_strengths": [float(item) for item in cfg.strengths],
            "perturbation_curriculum_mode": "discrete_bank",
            "perturbation_levels": [
                {"level_index": idx, "level_name": f"level_{idx:02d}", "strength": float(strength)}
                for idx, strength in enumerate(cfg.strengths)
            ],
        }
        return descriptors, metadata

    bank = build_stage1_curriculum_bank(
        _stage1_hrl_curriculum_cfg(),
        FrontRESStage1CurriculumBankConfig(
            frontier_scale=float(cfg.curriculum_frontier_scale),
            dr_min=float(cfg.curriculum_dr_min),
            dr_max=float(cfg.curriculum_dr_max),
            n_train=int(cfg.curriculum_bank_size),
            progress=float(cfg.curriculum_progress),
            seq_idx=int(cfg.curriculum_seq_idx),
            active_dims=cfg.curriculum_active_dims,
            boundary_stats=_stage1_boundary_stats(),
            include_hard_as_train=bool(cfg.curriculum_include_hard_as_train),
        ),
    )
    bank_probe = stage1_curriculum_bank_probe(bank)
    print(
        "[FrontRES Stage1 Segment Cache] curriculum_bank "
        f"record_count={bank_probe['record_count']} "
        f"allowed_bases={bank_probe['allowed_bases']} "
        f"mix_classes={bank_probe['mix_classes'][:8]} "
        f"dr_factors={bank_probe['dr_factors'][:8]} "
        f"roles={bank_probe['roles'][:8]}",
        flush=True,
    )
    descriptors = build_perturbation_descriptors_from_curriculum_bank(
        segments,
        bank,
        FrontRESBankDescriptorConfig(
            variants_per_record=int(cfg.variants_per_strength),
            base_seed=int(cfg.base_seed),
            duration=int(cfg.horizon_k),
            temporal_mode=str(cfg.curriculum_temporal_mode),
            burst_min_steps=int(cfg.curriculum_burst_min_steps),
            burst_max_steps=int(cfg.curriculum_burst_max_steps),
        ),
    )
    levels = [
        {
            "level_index": int(record.bank_id),
            "level_name": f"level_{int(record.bank_id):02d}",
            "strength": float(record.actual_dr_scale),
            "family_group": list(record.family_group),
            "mix_class": str(record.mix_class),
            "frontier_scale": float(record.frontier_scale),
            "dr_factor": float(record.dr_factor),
            "actual_dr_scale": float(record.actual_dr_scale),
            "role": str(record.role),
        }
        for record in bank.records
    ]
    metadata = {
        "strengths": sorted({float(record.actual_dr_scale) for record in bank.records}),
        "legacy_strengths": [float(item) for item in cfg.strengths],
        "perturbation_curriculum_mode": "hrl_curriculum_bank",
        "perturbation_levels": levels,
        "curriculum_bank_record_count": len(bank.records),
        "curriculum_allowed_bases": list(bank.allowed_bases),
        "curriculum_active_modes": list(bank.active_modes),
        "curriculum_complexity": bank.complexity,
        "curriculum_mix_diag": dict(bank.mix_diag),
        "curriculum_frontier_scale": float(cfg.curriculum_frontier_scale),
        "curriculum_dr_min": float(cfg.curriculum_dr_min),
        "curriculum_dr_max": float(cfg.curriculum_dr_max),
        "curriculum_include_hard_as_train": bool(cfg.curriculum_include_hard_as_train),
        "curriculum_temporal_mode": str(cfg.curriculum_temporal_mode),
        "curriculum_burst_min_steps": int(cfg.curriculum_burst_min_steps),
        "curriculum_burst_max_steps": int(cfg.curriculum_burst_max_steps),
    }
    return descriptors, metadata


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


def _stage1_hrl_curriculum_cfg() -> dict[str, Any]:
    return {
        "frontres_perturbation_curriculum_enabled": True,
        "frontres_adaptive_perturb_curriculum_enabled": True,
        "frontres_mixed_dr_strength_per_env": True,
        "frontres_mixed_dr_easy_weight": 0.45,
        "frontres_mixed_dr_frontier_weight": 0.40,
        "frontres_mixed_dr_hard_weight": 0.15,
        "frontres_mixed_dr_easy_factor": 0.75,
        "frontres_mixed_dr_frontier_factor": 1.0,
        "frontres_mixed_dr_hard_factor": 1.08,
        "frontres_boundary_safe_high": 0.45,
        "frontres_boundary_repair_low": 0.45,
        "frontres_boundary_repair_high": 0.70,
        "frontres_boundary_broken_target": 0.25,
        "frontres_boundary_broken_high": 0.35,
        "frontres_boundary_positive_gain_low": 0.45,
        "frontres_boundary_positive_gain_high": 0.55,
        "frontres_curriculum_full_prob": 0.05,
        "frontres_curriculum_three_prob": 0.10,
        "frontres_curriculum_two_late_prob": 0.40,
    }


def _stage1_boundary_stats() -> dict[str, float]:
    return {"safe": 0.2, "repair": 0.55, "broken": 0.1, "positive_gain": 0.7}
