from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
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
        FrontRESAMASSIndexSummary,
        append_amass_segment_index,
        build_amass_segment_index,
        build_amass_segment_index_from_paths,
        discover_amass_npz_files,
        iter_amass_segment_index_chunks_from_paths,
        read_amass_segment_index,
        write_amass_segment_index,
    )
    from rsl_rl.frontres.frontres_segment_cache_io import (
        FrontRESCleanStateEntry,
        append_stage1_cache_progress,
        clean_resume_key,
        noisy_resume_key,
        read_clean_state_manifest_records,
        read_clean_state_record,
        read_noisy_variant_manifest_records,
        scan_stage1_cache_resume_state,
        write_cache_metadata,
        write_clean_state_chunked_shard_atomic,
        write_clean_state_manifest_records,
        write_noisy_variant_chunked_shard_atomic,
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
    FrontRESAMASSIndexSummary = _indexer.FrontRESAMASSIndexSummary
    append_amass_segment_index = _indexer.append_amass_segment_index
    build_amass_segment_index = _indexer.build_amass_segment_index
    build_amass_segment_index_from_paths = _indexer.build_amass_segment_index_from_paths
    discover_amass_npz_files = _indexer.discover_amass_npz_files
    iter_amass_segment_index_chunks_from_paths = _indexer.iter_amass_segment_index_chunks_from_paths
    read_amass_segment_index = _indexer.read_amass_segment_index
    write_amass_segment_index = _indexer.write_amass_segment_index
    FrontRESCleanStateEntry = _cache_io.FrontRESCleanStateEntry
    append_stage1_cache_progress = _cache_io.append_stage1_cache_progress
    clean_resume_key = _cache_io.clean_resume_key
    noisy_resume_key = _cache_io.noisy_resume_key
    read_clean_state_manifest_records = _cache_io.read_clean_state_manifest_records
    read_clean_state_record = _cache_io.read_clean_state_record
    read_noisy_variant_manifest_records = _cache_io.read_noisy_variant_manifest_records
    scan_stage1_cache_resume_state = _cache_io.scan_stage1_cache_resume_state
    write_cache_metadata = _cache_io.write_cache_metadata
    write_clean_state_chunked_shard_atomic = _cache_io.write_clean_state_chunked_shard_atomic
    write_clean_state_manifest_records = _cache_io.write_clean_state_manifest_records
    write_noisy_variant_chunked_shard_atomic = _cache_io.write_noisy_variant_chunked_shard_atomic
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
    previous_status = _read_stage1_cache_status(cache_dir)
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
    if cfg.max_segments is None:
        return _build_stage1_segment_cache_streaming(
            env,
            cfg,
            cache_dir=cache_dir,
            previous_status=previous_status,
            loaded_motion_paths=loaded_motion_paths,
        )
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
    descriptors, perturbation_metadata = build_stage1_perturbation_plan(segments, cfg)
    descriptor_trace = descriptor_probe(descriptors)
    build_signature = _stage1_build_signature(
        cfg,
        loaded_motion_paths=loaded_motion_paths,
        index_summary=index_summary,
        perturbation_metadata=perturbation_metadata,
    )
    existing_signature = _existing_stage1_build_signature(cache_dir, previous_status)
    signature_match = existing_signature is None or existing_signature.get("hash") == build_signature["hash"]
    if not signature_match:
        print(
            "[FrontRES Stage1 Resume Probe] "
            f"signature_match=False existing_hash={existing_signature.get('hash')} "
            f"current_hash={build_signature['hash']} resume_enabled=False force_rebuild=False",
            flush=True,
        )
        append_stage1_cache_progress(
            cache_dir,
            {
                "event": "resume_signature_mismatch",
                "existing_hash": existing_signature.get("hash"),
                "current_hash": build_signature["hash"],
            },
        )
        write_stage1_cache_status(
            cache_dir,
            {
                "stage": "stage1_segment_cache",
                "status": "signature_mismatch",
                "cache_dir": str(cache_dir),
                "build_signature": existing_signature,
                "current_build_signature": build_signature,
            },
        )
        raise ValueError(
            "Stage 1 cache build signature mismatch; use an empty cache_dir or rebuild the cache explicitly"
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
            "build_signature": build_signature,
        },
    )
    append_stage1_cache_progress(
        cache_dir,
        {
            "event": "indexed",
            "segment_count": len(segments),
            "indexed_motion_count": int(index_summary.motion_count),
            "segment_index_path": str(cache_dir / "segment_index.jsonl"),
            "signature_hash": build_signature["hash"],
        },
    )
    _ensure_frontres_env_reset(env)
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
    descriptors_by_segment: dict[int, list[Any]] = {}
    for descriptor in descriptors:
        descriptors_by_segment.setdefault(int(descriptor.segment_id), []).append(descriptor)

    expected_clean_keys = {clean_resume_key(segment) for segment in segments}
    segment_by_id = {int(segment.segment_id): segment for segment in segments}
    expected_noisy_keys = {
        noisy_resume_key(segment_by_id[int(descriptor.segment_id)], descriptor)
        for descriptor in descriptors
        if int(descriptor.segment_id) in segment_by_id
    }
    resume_scan = scan_stage1_cache_resume_state(cache_dir)
    clean_record_by_key = _load_committed_clean_records_by_key(
        cache_dir,
        expected_clean_keys=expected_clean_keys,
        completed_clean_keys=resume_scan.completed_clean_keys,
    )
    noisy_records_by_key = _load_committed_noisy_records_by_key(
        cache_dir,
        expected_noisy_keys=expected_noisy_keys,
        completed_noisy_keys=resume_scan.completed_noisy_keys,
    )
    completed_clean_keys = set(clean_record_by_key)
    completed_noisy_keys = set(noisy_records_by_key)
    pending_clean_count = len(expected_clean_keys - completed_clean_keys)
    pending_noisy_count = len(expected_noisy_keys - completed_noisy_keys)
    print(
        "[FrontRES Stage1 Resume Probe] "
        f"signature_match=True "
        f"completed_clean={len(completed_clean_keys)} "
        f"completed_noisy={len(completed_noisy_keys)} "
        f"pending_clean={pending_clean_count} "
        f"pending_noisy={pending_noisy_count} "
        f"ignored_tmp={len(resume_scan.ignored_tmp_paths)} "
        f"corrupt_count={len(resume_scan.corrupt_records)} "
        f"resume_enabled=True "
        f"force_rebuild=False",
        flush=True,
    )
    append_stage1_cache_progress(
        cache_dir,
        {
            "event": "resume_scan",
            "completed_clean": len(completed_clean_keys),
            "completed_noisy": len(completed_noisy_keys),
            "pending_clean": pending_clean_count,
            "pending_noisy": pending_noisy_count,
            "ignored_tmp": len(resume_scan.ignored_tmp_paths),
            "corrupt_count": len(resume_scan.corrupt_records),
            "signature_hash": build_signature["hash"],
        },
    )

    clean_records: list[dict[str, Any]] = list(clean_record_by_key.values())
    noisy_records_by_strength: dict[float, list[dict[str, Any]]] = {
        float(strength): [] for strength in perturbation_metadata["strengths"]
    }
    for record in noisy_records_by_key.values():
        strength = float(record["descriptor"]["strength"])
        noisy_records_by_strength.setdefault(strength, []).append(record)

    clean_written = len(clean_records)
    noisy_written = sum(len(records) for records in noisy_records_by_strength.values())
    strength_counts = {
        float(strength): len(noisy_records_by_strength.get(float(strength), []))
        for strength in perturbation_metadata["strengths"]
    }
    chunk_size = int(cfg.cache_chunk_size)
    clean_buffer: list[FrontRESCleanStateEntry] = []
    clean_shard_id = max(
        int(cfg.clean_shard_id),
        _next_shard_id_from_records(clean_records, default=int(cfg.clean_shard_id)),
    )
    noisy_buffers_by_strength: dict[float, list[Any]] = {
        float(strength): [] for strength in perturbation_metadata["strengths"]
    }
    noisy_shard_ids_by_strength: dict[float, int] = {
        float(strength): max(
            int(cfg.noisy_shard_id),
            _next_shard_id_from_records(
                noisy_records_by_strength.get(float(strength), []),
                default=int(cfg.noisy_shard_id),
            ),
        )
        for strength in perturbation_metadata["strengths"]
    }

    def flush_clean_buffer() -> str | None:
        nonlocal clean_shard_id
        if not clean_buffer:
            return None
        shard_path, records = write_clean_state_chunked_shard_atomic(
            cache_dir,
            clean_buffer,
            shard_id=clean_shard_id,
        )
        clean_records.extend(records)
        manifest_path = write_clean_state_manifest_records(cache_dir, clean_records, shard_id=int(cfg.clean_shard_id))
        print(
            "[FrontRES Stage1 Shard Commit] "
            f"kind=clean shard_path={shard_path} manifest_path={manifest_path} "
            f"row_count={len(records)} committed_total={len(clean_records)}",
            flush=True,
        )
        clean_buffer.clear()
        clean_shard_id += 1
        return str(shard_path)

    def flush_noisy_buffer(strength: float) -> str | None:
        buffer = noisy_buffers_by_strength.setdefault(float(strength), [])
        if not buffer:
            return None
        shard_id = noisy_shard_ids_by_strength.get(float(strength), int(cfg.noisy_shard_id))
        shard_path, records = write_noisy_variant_chunked_shard_atomic(
            cache_dir,
            buffer,
            strength=float(strength),
            shard_id=shard_id,
        )
        strength_records = noisy_records_by_strength.setdefault(float(strength), [])
        strength_records.extend(records)
        manifest_path = write_noisy_variant_manifest_records(
            cache_dir,
            strength_records,
            strength=float(strength),
            shard_id=int(cfg.noisy_shard_id),
        )
        print(
            "[FrontRES Stage1 Shard Commit] "
            f"kind=noisy strength={float(strength)} shard_path={shard_path} manifest_path={manifest_path} "
            f"row_count={len(records)} committed_total={len(strength_records)}",
            flush=True,
        )
        buffer.clear()
        noisy_shard_ids_by_strength[float(strength)] = shard_id + 1
        return str(shard_path)

    for segment in segments:
        segment_clean_key = clean_resume_key(segment)
        segment_descriptors = descriptors_by_segment.get(int(segment.segment_id), [])
        pending_descriptors = [
            descriptor
            for descriptor in segment_descriptors
            if noisy_resume_key(segment, descriptor) not in completed_noisy_keys
        ]
        if segment_clean_key in completed_clean_keys:
            clean_state = read_clean_state_record(cache_dir, clean_record_by_key[segment_clean_key]).clean_state
            clean_shard_path = None
            clean_reused = True
        else:
            prepare_clean_segment(env, segment=segment, env_ids=env_ids)
            clean_state = extract_robot_rollout_state(env, env_ids=env_ids, robot_name=cfg.robot_name)
            entry = FrontRESCleanStateEntry(segment=segment, clean_state=clean_state)
            entry.validate()
            clean_buffer.append(entry)
            clean_written += 1
            completed_clean_keys.add(segment_clean_key)
            clean_shard_path = None
            if len(clean_buffer) >= chunk_size:
                clean_shard_path = flush_clean_buffer()
            clean_reused = False
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
            del entry

        if clean_reused and not pending_descriptors:
            append_stage1_cache_progress(
                cache_dir,
                {
                    "event": "resume_skip_segment",
                    "segment_id": int(segment.segment_id),
                    "clean_reused": True,
                    "pending_noisy": 0,
                },
            )
            del clean_state
            continue

        if clean_reused:
            append_stage1_cache_progress(
                cache_dir,
                {
                    "event": "resume_reuse_clean",
                    "segment_id": int(segment.segment_id),
                    "pending_noisy": len(pending_descriptors),
                },
            )

        for descriptor in pending_descriptors:
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
            "build_signature": build_signature,
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
            "build_signature": build_signature,
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


def _build_stage1_segment_cache_streaming(
    env: Any,
    cfg: FrontRESStage1CacheBuilderConfig,
    *,
    cache_dir: Path,
    previous_status: dict[str, Any],
    loaded_motion_paths: list[str],
) -> FrontRESStage1CacheBuildResult:
    existing_signature = _existing_stage1_build_signature(cache_dir, previous_status)
    if existing_signature is not None:
        print(
            "[FrontRES Stage1 Resume Probe] "
            "signature_match=unknown resume_enabled=False streaming_resume=False",
            flush=True,
        )
        raise ValueError("Stage 1 streaming resume is not enabled until Step 5")

    motion_paths = (
        [Path(path).expanduser().resolve() for path in loaded_motion_paths]
        if loaded_motion_paths
        else discover_amass_npz_files(cfg.amass_root, max_motions=cfg.max_motions)
    )
    env_ids = torch.tensor([int(cfg.env_id)], dtype=torch.long)
    _ensure_frontres_env_reset(env)

    perturbation_metadata: dict[str, Any] | None = None
    clean_records: list[dict[str, Any]] = []
    noisy_records_by_strength: dict[float, list[dict[str, Any]]] = {}
    strength_counts: dict[float, int] = {}
    clean_buffer: list[FrontRESCleanStateEntry] = []
    noisy_buffers_by_strength: dict[float, list[Any]] = {}
    clean_shard_id = int(cfg.clean_shard_id)
    noisy_shard_ids_by_strength: dict[float, int] = {}
    clean_written = 0
    noisy_written = 0
    descriptor_count = 0
    next_perturbation_id = 0
    segment_count = 0
    motion_count = 0
    skipped_short = 0
    last_clean_manifest_path: Path | None = None
    noisy_shard_paths: dict[float, str] = {}
    indexed_segment_keys = _load_existing_segment_index_keys(cache_dir)
    clean_record_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    noisy_records_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}

    def flush_clean_buffer() -> str | None:
        nonlocal clean_shard_id, last_clean_manifest_path
        if not clean_buffer:
            return None
        shard_path, records = write_clean_state_chunked_shard_atomic(
            cache_dir,
            clean_buffer,
            shard_id=clean_shard_id,
        )
        clean_records.extend(records)
        last_clean_manifest_path = write_clean_state_manifest_records(
            cache_dir,
            clean_records,
            shard_id=int(cfg.clean_shard_id),
        )
        print(
            "[FrontRES Stage1 Shard Commit] "
            f"kind=clean shard_path={shard_path} manifest_path={last_clean_manifest_path} "
            f"row_count={len(records)} committed_total={len(clean_records)}",
            flush=True,
        )
        clean_buffer.clear()
        clean_shard_id += 1
        return str(shard_path)

    def flush_noisy_buffer(strength: float) -> str | None:
        buffer = noisy_buffers_by_strength.setdefault(float(strength), [])
        if not buffer:
            return None
        shard_id = noisy_shard_ids_by_strength.get(float(strength), int(cfg.noisy_shard_id))
        shard_path, records = write_noisy_variant_chunked_shard_atomic(
            cache_dir,
            buffer,
            strength=float(strength),
            shard_id=shard_id,
        )
        strength_records = noisy_records_by_strength.setdefault(float(strength), [])
        strength_records.extend(records)
        manifest_path = write_noisy_variant_manifest_records(
            cache_dir,
            strength_records,
            strength=float(strength),
            shard_id=int(cfg.noisy_shard_id),
        )
        noisy_shard_paths[float(strength)] = str(manifest_path)
        print(
            "[FrontRES Stage1 Shard Commit] "
            f"kind=noisy strength={float(strength)} shard_path={shard_path} manifest_path={manifest_path} "
            f"row_count={len(records)} committed_total={len(strength_records)}",
            flush=True,
        )
        buffer.clear()
        noisy_shard_ids_by_strength[float(strength)] = shard_id + 1
        return str(shard_path)

    for chunk in iter_amass_segment_index_chunks_from_paths(
        cfg.amass_root,
        motion_paths,
        horizon_k=int(cfg.horizon_k),
        frame_stride=int(cfg.frame_stride),
        chunk_size=int(cfg.cache_chunk_size),
    ):
        chunk_segments = list(chunk.segments)
        if not chunk_segments:
            continue
        new_index_segments = [segment for segment in chunk_segments if clean_resume_key(segment) not in indexed_segment_keys]
        if new_index_segments:
            append_amass_segment_index(cache_dir, new_index_segments)
            indexed_segment_keys.update(clean_resume_key(segment) for segment in new_index_segments)
        chunk_probe = chunk.probe()
        segment_count = int(chunk.segment_count)
        motion_count = int(chunk.motion_count)
        skipped_short = int(chunk.skipped_short_motions)
        print(
            "[FrontRES Stage1 Index Chunk] "
            f"chunk_id={chunk_probe['chunk_id']} "
            f"motion_count={chunk_probe['motion_count']} "
            f"segment_count={chunk_probe['chunk_segment_count']} "
            f"segment_id_min={chunk_probe['segment_id_min']} "
            f"segment_id_max={chunk_probe['segment_id_max']}",
            flush=True,
        )
        append_stage1_cache_progress(
            cache_dir,
            {
                "event": "index_chunk",
                "chunk_id": int(chunk.chunk_id),
                "segment_count": int(chunk.segment_count),
                "chunk_segment_count": len(chunk_segments),
                "segment_id_min": int(chunk_segments[0].segment_id),
                "segment_id_max": int(chunk_segments[-1].segment_id),
                "new_index_count": len(new_index_segments),
                "segment_index_path": str(cache_dir / "segment_index.jsonl"),
            },
        )
        descriptors, chunk_metadata = build_stage1_perturbation_plan(
            chunk_segments,
            cfg,
            start_perturbation_id=next_perturbation_id,
        )
        if perturbation_metadata is None:
            perturbation_metadata = chunk_metadata
            noisy_records_by_strength = {float(strength): [] for strength in perturbation_metadata["strengths"]}
            strength_counts = {float(strength): 0 for strength in perturbation_metadata["strengths"]}
            noisy_buffers_by_strength = {float(strength): [] for strength in perturbation_metadata["strengths"]}
            noisy_shard_ids_by_strength = {
                float(strength): int(cfg.noisy_shard_id) for strength in perturbation_metadata["strengths"]
            }
        descriptor_count += len(descriptors)
        next_perturbation_id += len(descriptors)
        descriptor_trace = descriptor_probe(descriptors)
        resume_scan = scan_stage1_cache_resume_state(cache_dir)
        expected_clean_keys = {clean_resume_key(segment) for segment in chunk_segments}
        segment_by_id = {int(segment.segment_id): segment for segment in chunk_segments}
        expected_noisy_keys = {
            noisy_resume_key(segment_by_id[int(descriptor.segment_id)], descriptor)
            for descriptor in descriptors
            if int(descriptor.segment_id) in segment_by_id
        }
        chunk_clean_records = _load_committed_clean_records_by_key(
            cache_dir,
            expected_clean_keys=expected_clean_keys,
            completed_clean_keys=resume_scan.completed_clean_keys,
        )
        chunk_noisy_records = _load_committed_noisy_records_by_key(
            cache_dir,
            expected_noisy_keys=expected_noisy_keys,
            completed_noisy_keys=resume_scan.completed_noisy_keys,
        )
        for key, record in chunk_clean_records.items():
            if key not in clean_record_by_key:
                clean_record_by_key[key] = record
                clean_records.append(record)
        for key, record in chunk_noisy_records.items():
            if key not in noisy_records_by_key:
                noisy_records_by_key[key] = record
                strength = float(record["descriptor"]["strength"])
                noisy_records_by_strength.setdefault(strength, []).append(record)
        clean_written = len(clean_records)
        noisy_written = sum(len(records) for records in noisy_records_by_strength.values())
        strength_counts = {
            float(strength): len(noisy_records_by_strength.get(float(strength), []))
            for strength in (perturbation_metadata or chunk_metadata)["strengths"]
        }
        clean_shard_id = max(
            clean_shard_id,
            _next_shard_id_from_records(clean_records, default=int(cfg.clean_shard_id)),
        )
        for strength, records in noisy_records_by_strength.items():
            noisy_shard_ids_by_strength[float(strength)] = max(
                noisy_shard_ids_by_strength.get(float(strength), int(cfg.noisy_shard_id)),
                _next_shard_id_from_records(records, default=int(cfg.noisy_shard_id)),
            )
        completed_clean_keys = set(clean_record_by_key)
        completed_noisy_keys = set(noisy_records_by_key)
        pending_clean_count = len(expected_clean_keys - completed_clean_keys)
        pending_noisy_count = len(expected_noisy_keys - completed_noisy_keys)
        print(
            "[FrontRES Stage1 Chunk Resume Probe] "
            f"chunk_id={int(chunk.chunk_id)} "
            f"completed_clean={len(completed_clean_keys)} "
            f"completed_noisy={len(completed_noisy_keys)} "
            f"pending_clean={pending_clean_count} "
            f"pending_noisy={pending_noisy_count} "
            f"corrupt_count={len(resume_scan.corrupt_records)}",
            flush=True,
        )
        write_stage1_cache_status(
            cache_dir,
            {
                "stage": "stage1_segment_cache",
                "status": "index_chunk",
                "segment_count": segment_count,
                "chunk_id": int(chunk.chunk_id),
                "chunk_segment_count": len(chunk_segments),
                "descriptor_count": descriptor_count,
                "clean_written": clean_written,
                "noisy_written": noisy_written,
                "cache_dir": str(cache_dir),
                "segment_index_path": str(cache_dir / "segment_index.jsonl"),
            },
        )
        print(
            "[FrontRES Stage1 Segment Cache] perturbation_chunk "
            f"chunk_id={int(chunk.chunk_id)} "
            f"mode={chunk_metadata['perturbation_curriculum_mode']} "
            f"descriptor_count={descriptor_trace['count']} "
            f"strengths={descriptor_trace['strengths'][:8]}",
            flush=True,
        )

        descriptors_by_segment: dict[int, list[Any]] = {}
        for descriptor in descriptors:
            descriptors_by_segment.setdefault(int(descriptor.segment_id), []).append(descriptor)

        for segment in chunk_segments:
            segment_clean_key = clean_resume_key(segment)
            segment_descriptors = descriptors_by_segment.get(int(segment.segment_id), [])
            pending_descriptors = [
                descriptor
                for descriptor in segment_descriptors
                if noisy_resume_key(segment, descriptor) not in completed_noisy_keys
            ]
            if segment_clean_key in completed_clean_keys:
                clean_state = read_clean_state_record(cache_dir, clean_record_by_key[segment_clean_key]).clean_state
                clean_reused = True
            else:
                prepare_clean_segment(env, segment=segment, env_ids=env_ids)
                clean_state = extract_robot_rollout_state(env, env_ids=env_ids, robot_name=cfg.robot_name)
                entry = FrontRESCleanStateEntry(segment=segment, clean_state=clean_state)
                entry.validate()
                clean_buffer.append(entry)
                clean_written += 1
                clean_shard_path = None
                if len(clean_buffer) >= int(cfg.cache_chunk_size):
                    clean_shard_path = flush_clean_buffer()
                clean_reused = False
                append_stage1_cache_progress(
                    cache_dir,
                    {
                        "event": "clean_done",
                        "segment_id": int(segment.segment_id),
                        "clean_written": clean_written,
                        "segment_count": segment_count,
                        "flushed_shard_path": clean_shard_path,
                        "buffer_count": len(clean_buffer),
                    },
                )
                del entry

            if clean_reused and not pending_descriptors:
                append_stage1_cache_progress(
                    cache_dir,
                    {
                        "event": "resume_skip_segment",
                        "segment_id": int(segment.segment_id),
                        "clean_reused": True,
                        "pending_noisy": 0,
                    },
                )
                del clean_state
                continue
            if clean_reused:
                append_stage1_cache_progress(
                    cache_dir,
                    {
                        "event": "resume_reuse_clean",
                        "segment_id": int(segment.segment_id),
                        "pending_noisy": len(pending_descriptors),
                    },
                )

            for descriptor in pending_descriptors:
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
                if len(noisy_buffers_by_strength[strength]) >= int(cfg.cache_chunk_size):
                    noisy_shard_path = flush_noisy_buffer(strength)
                append_stage1_cache_progress(
                    cache_dir,
                    {
                        "event": "noisy_done",
                        "segment_id": int(descriptor.segment_id),
                        "perturbation_id": int(descriptor.perturbation_id),
                        "strength": strength,
                        "noisy_written": noisy_written,
                        "descriptor_count": descriptor_count,
                        "flushed_shard_path": noisy_shard_path,
                        "buffer_count": len(noisy_buffers_by_strength[strength]),
                    },
                )
                del capture
            del clean_state

        flush_clean_buffer()
        for strength in list(noisy_buffers_by_strength):
            flush_noisy_buffer(float(strength))

    if perturbation_metadata is None:
        raise ValueError(f"no valid segments built from {cfg.amass_root} with horizon_k={cfg.horizon_k}")

    clean_shard_path = write_clean_state_manifest_records(
        cache_dir,
        clean_records,
        shard_id=int(cfg.clean_shard_id),
    )
    noisy_shard_paths = {}
    for strength, records in noisy_records_by_strength.items():
        noisy_path = write_noisy_variant_manifest_records(
            cache_dir,
            records,
            strength=float(strength),
            shard_id=int(cfg.noisy_shard_id),
        )
        noisy_shard_paths[float(strength)] = str(noisy_path)
    index_summary = FrontRESAMASSIndexSummary(
        amass_root=str(Path(cfg.amass_root).resolve()),
        motion_count=motion_count,
        segment_count=segment_count,
        horizon_k=int(cfg.horizon_k),
        frame_stride=int(cfg.frame_stride),
        skipped_short_motions=skipped_short,
    )
    build_signature = _stage1_build_signature(
        cfg,
        loaded_motion_paths=loaded_motion_paths,
        index_summary=index_summary,
        perturbation_metadata=perturbation_metadata,
    )
    metadata_path = write_cache_metadata(
        cache_dir,
        {
            "stage": "stage1_segment_cache",
            "amass_root": str(Path(cfg.amass_root).resolve()),
            "segment_count": segment_count,
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
            "build_signature": build_signature,
            "indexing_mode": "streaming_chunk",
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
            "segment_count": segment_count,
            "clean_written": clean_written,
            "noisy_written": noisy_written,
            "descriptor_count": descriptor_count,
            "segment_index_path": str(cache_dir / "segment_index.jsonl"),
            "clean_shard_path": str(clean_shard_path),
            "noisy_shard_paths": noisy_shard_paths,
            "metadata_path": str(metadata_path),
            "build_signature": build_signature,
            "indexing_mode": "streaming_chunk",
        },
    )
    append_stage1_cache_progress(
        cache_dir,
        {
            "event": "complete",
            "segment_count": segment_count,
            "clean_written": clean_written,
            "noisy_written": noisy_written,
            "metadata_path": str(metadata_path),
        },
    )
    return FrontRESStage1CacheBuildResult(
        cache_dir=str(cache_dir),
        segment_count=segment_count,
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
    *,
    start_perturbation_id: int = 0,
) -> tuple[list[Any], dict[str, Any]]:
    if cfg.perturbation_curriculum_mode == "discrete_bank":
        perturb_cfg = FrontRESPerturbationCurriculumConfig(
            strengths=tuple(float(item) for item in cfg.strengths),
            variants_per_strength=int(cfg.variants_per_strength),
            base_seed=int(cfg.base_seed),
            duration=int(cfg.horizon_k),
        )
        descriptors = build_perturbation_descriptors(
            segments,
            perturb_cfg,
            start_perturbation_id=int(start_perturbation_id),
        )
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
        start_perturbation_id=int(start_perturbation_id),
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


def _load_committed_clean_records_by_key(
    cache_dir: Path,
    *,
    expected_clean_keys: set[tuple[Any, ...]],
    completed_clean_keys: frozenset[tuple[Any, ...]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    usable_keys = set(expected_clean_keys).intersection(set(completed_clean_keys))
    for manifest_path in sorted((cache_dir / "manifests" / "clean_states").glob("*.pt")):
        try:
            _, records = read_clean_state_manifest_records(manifest_path)
        except Exception:
            continue
        for record in records:
            try:
                key = clean_resume_key(record)
            except Exception:
                continue
            if key in usable_keys:
                result[key] = record
    return result


def _load_existing_segment_index_keys(cache_dir: Path) -> set[tuple[Any, ...]]:
    index_path = cache_dir / "segment_index.jsonl"
    if not index_path.is_file():
        return set()
    try:
        return {clean_resume_key(segment) for segment in read_amass_segment_index(index_path)}
    except Exception:
        return set()


def _load_committed_noisy_records_by_key(
    cache_dir: Path,
    *,
    expected_noisy_keys: set[tuple[Any, ...]],
    completed_noisy_keys: frozenset[tuple[Any, ...]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    usable_keys = set(expected_noisy_keys).intersection(set(completed_noisy_keys))
    for manifest_path in sorted((cache_dir / "manifests" / "noisy_variants").glob("**/*.pt")):
        try:
            _, records = read_noisy_variant_manifest_records(manifest_path)
        except Exception:
            continue
        for record in records:
            try:
                key = noisy_resume_key(record)
            except Exception:
                continue
            if key in usable_keys:
                result[key] = record
    return result


def _next_shard_id_from_records(records: list[dict[str, Any]], *, default: int) -> int:
    max_id = int(default) - 1
    for record in records:
        path = Path(str(record.get("path", "")))
        stem = path.stem
        if not stem.startswith("shard_"):
            continue
        try:
            max_id = max(max_id, int(stem.removeprefix("shard_")))
        except ValueError:
            continue
    return max_id + 1


def _read_stage1_cache_status(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "build_status.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _existing_stage1_build_signature(cache_dir: Path, previous_status: dict[str, Any]) -> dict[str, Any] | None:
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            signature = metadata.get("build_signature")
            if isinstance(signature, dict) and signature.get("hash"):
                return signature
        except Exception:
            pass
    signature = previous_status.get("build_signature")
    if isinstance(signature, dict) and signature.get("hash"):
        return signature
    return None


def _stage1_build_signature(
    cfg: FrontRESStage1CacheBuilderConfig,
    *,
    loaded_motion_paths: list[str],
    index_summary: Any,
    perturbation_metadata: dict[str, Any],
) -> dict[str, Any]:
    loaded_motion_summary = _stage1_loaded_motion_path_summary(loaded_motion_paths)
    payload = {
        "schema": "frontres_stage1_cache_build_signature_v1",
        "amass_root": str(Path(cfg.amass_root).resolve()),
        "motion_source": "loaded_motion_paths" if loaded_motion_paths else "disk_scan",
        **loaded_motion_summary,
        "horizon_k": int(cfg.horizon_k),
        "frame_stride": int(cfg.frame_stride),
        "variants_per_strength": int(cfg.variants_per_strength),
        "base_seed": int(cfg.base_seed),
        "perturbation_curriculum_mode": str(perturbation_metadata["perturbation_curriculum_mode"]),
        "perturbation_levels": perturbation_metadata["perturbation_levels"],
        "curriculum_temporal_mode": str(cfg.curriculum_temporal_mode),
        "curriculum_active_dims": None
        if cfg.curriculum_active_dims is None
        else [int(item) for item in cfg.curriculum_active_dims],
        "indexed_motion_count": int(index_summary.motion_count),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return {"hash": hashlib.sha256(text.encode("utf-8")).hexdigest(), "payload": payload}


def _stage1_loaded_motion_path_summary(loaded_motion_paths: list[str]) -> dict[str, Any]:
    resolved_paths = [str(Path(path).resolve()) for path in loaded_motion_paths]
    path_text = json.dumps(resolved_paths, separators=(",", ":"), ensure_ascii=True)
    return {
        "loaded_motion_count": len(resolved_paths),
        "first_loaded_motion": resolved_paths[0] if resolved_paths else None,
        "last_loaded_motion": resolved_paths[-1] if resolved_paths else None,
        "loaded_motion_paths_hash": hashlib.sha256(path_text.encode("utf-8")).hexdigest(),
    }


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
