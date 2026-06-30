from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

try:
    from rsl_rl.frontres.frontres_segment_cache_schema import (
        FrontRESNoisyVariant,
        FrontRESPerturbationDescriptor,
        FrontRESRobotRolloutState,
        FrontRESSegmentIndex,
    )
except ModuleNotFoundError:
    _SCHEMA_PATH = Path(__file__).with_name("frontres_segment_cache_schema.py")
    _SCHEMA_SPEC = importlib.util.spec_from_file_location("frontres_segment_cache_schema", _SCHEMA_PATH)
    if _SCHEMA_SPEC is None or _SCHEMA_SPEC.loader is None:
        raise
    _SCHEMA_MODULE = importlib.util.module_from_spec(_SCHEMA_SPEC)
    sys.modules[_SCHEMA_SPEC.name] = _SCHEMA_MODULE
    _SCHEMA_SPEC.loader.exec_module(_SCHEMA_MODULE)
    FrontRESNoisyVariant = _SCHEMA_MODULE.FrontRESNoisyVariant
    FrontRESPerturbationDescriptor = _SCHEMA_MODULE.FrontRESPerturbationDescriptor
    FrontRESRobotRolloutState = _SCHEMA_MODULE.FrontRESRobotRolloutState
    FrontRESSegmentIndex = _SCHEMA_MODULE.FrontRESSegmentIndex


@dataclass(frozen=True)
class FrontRESCleanStateEntry:
    segment: FrontRESSegmentIndex
    clean_state: FrontRESRobotRolloutState

    @property
    def segment_id(self) -> int:
        return int(self.segment.segment_id)

    def validate(self) -> None:
        self.segment.validate()
        self.clean_state.validate(name="clean_state")

    def probe(self) -> dict[str, Any]:
        self.validate()
        result = {
            "segment_id": self.segment_id,
            "motion_rel_path": self.segment.motion_rel_path,
            "start_frame": int(self.segment.start_frame),
            "horizon_k": int(self.segment.horizon_k),
        }
        result.update(self.clean_state.probe(prefix="clean_state"))
        return result


class FrontRESSegmentShardLRU:
    def __init__(self, *, max_shards: int = 8, map_location: str | torch.device = "cpu") -> None:
        if int(max_shards) <= 0:
            raise ValueError(f"max_shards must be positive, got {max_shards}")
        self.max_shards = int(max_shards)
        self.map_location = map_location
        self._payloads: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.load_count = 0
        self.hit_count = 0

    def load(self, path: str | Path) -> dict[str, Any]:
        key = str(Path(path))
        if key in self._payloads:
            payload = self._payloads.pop(key)
            self._payloads[key] = payload
            self.hit_count += 1
            return payload
        payload = torch.load(Path(path), map_location=self.map_location, weights_only=False)
        self._payloads[key] = payload
        self.load_count += 1
        while len(self._payloads) > self.max_shards:
            self._payloads.popitem(last=False)
        return payload

    def probe(self) -> dict[str, int]:
        return {
            "max_shards": int(self.max_shards),
            "resident_shards": len(self._payloads),
            "load_count": int(self.load_count),
            "hit_count": int(self.hit_count),
        }


@dataclass(frozen=True)
class FrontRESStage1CacheResumeScan:
    completed_clean_keys: frozenset[tuple[Any, ...]]
    completed_noisy_keys: frozenset[tuple[Any, ...]]
    ignored_tmp_paths: tuple[str, ...]
    corrupt_records: tuple[dict[str, Any], ...]
    clean_manifest_count: int
    noisy_manifest_count: int

    def probe(self) -> dict[str, Any]:
        return {
            "completed_clean": len(self.completed_clean_keys),
            "completed_noisy": len(self.completed_noisy_keys),
            "ignored_tmp": len(self.ignored_tmp_paths),
            "corrupt_count": len(self.corrupt_records),
            "clean_manifest_count": int(self.clean_manifest_count),
            "noisy_manifest_count": int(self.noisy_manifest_count),
        }


def write_cache_metadata(cache_dir: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "metadata.json"
    payload = {"format": "frontres_segment_cache_v1", **metadata}
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return metadata_path


def write_stage1_cache_status(cache_dir: str | Path, status: dict[str, Any]) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    status_path = path / "build_status.json"
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, sort_keys=True)
    return status_path


def append_stage1_cache_progress(cache_dir: str | Path, event: dict[str, Any]) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    progress_path = path / "progress.jsonl"
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
    return progress_path


def read_cache_metadata(cache_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(cache_dir) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "frontres_segment_cache_v1":
        raise ValueError(f"unsupported FrontRES segment cache format: {payload.get('format')}")
    return payload


def write_clean_state_entry_file(cache_dir: str | Path, entry: FrontRESCleanStateEntry) -> Path:
    entry.validate()
    entry_path = _clean_segment_path(cache_dir, entry.segment)
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "frontres_clean_state_file_v1",
            "entry": clean_entry_to_record(entry),
        },
        entry_path,
    )
    return entry_path


def write_clean_state_chunked_shard(
    cache_dir: str | Path,
    entries: Iterable[FrontRESCleanStateEntry],
    *,
    shard_id: int = 0,
) -> tuple[Path, list[dict[str, Any]]]:
    entry_list = list(entries)
    for entry in entry_list:
        entry.validate()
    shard_path = _clean_chunked_shard_path(cache_dir, shard_id)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "frontres_clean_state_chunked_shard_v1",
            "entries": [clean_entry_to_record(entry) for entry in entry_list],
        },
        shard_path,
    )
    manifest_records = [
        {
            "path": _relative_posix(shard_path, cache_dir),
            "row": row,
            "segment": segment_to_record(entry.segment),
        }
        for row, entry in enumerate(entry_list)
    ]
    return shard_path, manifest_records


def write_clean_state_chunked_shard_atomic(
    cache_dir: str | Path,
    entries: Iterable[FrontRESCleanStateEntry],
    *,
    shard_id: int = 0,
) -> tuple[Path, list[dict[str, Any]]]:
    entry_list = list(entries)
    for entry in entry_list:
        entry.validate()
    shard_path = _clean_chunked_shard_path(cache_dir, shard_id)
    payload = {
        "format": "frontres_clean_state_chunked_shard_v1",
        "entries": [clean_entry_to_record(entry) for entry in entry_list],
    }
    _torch_save_atomic(payload, shard_path)
    manifest_records = [
        {
            "path": _relative_posix(shard_path, cache_dir),
            "row": row,
            "segment": segment_to_record(entry.segment),
        }
        for row, entry in enumerate(entry_list)
    ]
    return shard_path, manifest_records


def clean_state_manifest_record(cache_dir: str | Path, entry: FrontRESCleanStateEntry) -> dict[str, Any]:
    entry.validate()
    entry_path = _clean_segment_path(cache_dir, entry.segment)
    if not entry_path.is_file():
        raise FileNotFoundError(f"clean state payload missing before manifest record: {entry_path}")
    return {
        "path": _relative_posix(entry_path, cache_dir),
        "segment": segment_to_record(entry.segment),
    }


def write_clean_state_manifest_records(
    cache_dir: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    shard_id: int = 0,
) -> Path:
    record_list = list(records)
    manifest_path = _clean_manifest_path(cache_dir, shard_id)
    _torch_save_atomic(
        {
            "format": "frontres_clean_state_tree_manifest_v1",
            "entries": record_list,
        },
        manifest_path,
    )
    return manifest_path


def write_clean_state_manifest(
    cache_dir: str | Path,
    entries: Iterable[FrontRESCleanStateEntry],
    *,
    shard_id: int = 0,
) -> Path:
    entry_list = list(entries)
    for entry in entry_list:
        entry.validate()
    manifest_path = _clean_manifest_path(cache_dir, shard_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = [clean_state_manifest_record(cache_dir, entry) for entry in entry_list]
    return write_clean_state_manifest_records(cache_dir, records, shard_id=shard_id)


def write_clean_state_shard(
    cache_dir: str | Path,
    entries: Iterable[FrontRESCleanStateEntry],
    *,
    shard_id: int = 0,
) -> Path:
    entry_list = list(entries)
    for entry in entry_list:
        entry.validate()
    _, records = write_clean_state_chunked_shard(cache_dir, entry_list, shard_id=shard_id)
    return write_clean_state_manifest_records(cache_dir, records, shard_id=shard_id)


def read_clean_state_shard(path: str | Path) -> list[FrontRESCleanStateEntry]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") == "frontres_clean_state_shard_v1":
        entries = [clean_entry_from_record(record) for record in payload["entries"]]
    elif payload.get("format") == "frontres_clean_state_tree_manifest_v1":
        base_dir = Path(path).parent.parent.parent
        entries = []
        for item in payload["entries"]:
            entry_payload = torch.load(base_dir / item["path"], map_location="cpu", weights_only=False)
            if entry_payload.get("format") != "frontres_clean_state_file_v1":
                if entry_payload.get("format") != "frontres_clean_state_chunked_shard_v1":
                    raise ValueError(f"unsupported clean state file format: {entry_payload.get('format')}")
                entries.append(clean_entry_from_record(entry_payload["entries"][int(item["row"])]))
            else:
                entries.append(clean_entry_from_record(entry_payload["entry"]))
    else:
        raise ValueError(f"unsupported clean state shard format: {payload.get('format')}")
    for entry in entries:
        entry.validate()
    return entries


def read_clean_state_manifest_records(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = Path(path)
    payload = torch.load(manifest_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "frontres_clean_state_tree_manifest_v1":
        raise ValueError(f"unsupported clean state manifest format: {payload.get('format')}")
    base_dir = manifest_path.parent.parent.parent
    return base_dir, list(payload["entries"])


def read_clean_state_record(
    cache_dir: str | Path,
    record: dict[str, Any],
    *,
    shard_cache: FrontRESSegmentShardLRU | None = None,
) -> FrontRESCleanStateEntry:
    payload_path = Path(cache_dir) / str(record["path"])
    payload = shard_cache.load(payload_path) if shard_cache is not None else torch.load(
        payload_path,
        map_location="cpu",
        weights_only=False,
    )
    fmt = payload.get("format")
    if fmt == "frontres_clean_state_file_v1":
        entry = clean_entry_from_record(payload["entry"])
    elif fmt == "frontres_clean_state_chunked_shard_v1":
        entry = clean_entry_from_record(payload["entries"][int(record["row"])])
    else:
        raise ValueError(f"unsupported clean state payload format: {fmt}")
    entry.validate()
    return entry


def write_noisy_variant_file(cache_dir: str | Path, variant: FrontRESNoisyVariant) -> Path:
    variant.validate()
    variant_path = _noisy_variant_path(cache_dir, variant)
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "frontres_noisy_variant_file_v1",
            "variant": noisy_variant_to_record(variant),
        },
        variant_path,
    )
    return variant_path


def write_noisy_variant_chunked_shard(
    cache_dir: str | Path,
    variants: Iterable[FrontRESNoisyVariant],
    *,
    strength: float,
    shard_id: int = 0,
) -> tuple[Path, list[dict[str, Any]]]:
    variant_list = list(variants)
    for variant in variant_list:
        variant.validate()
    shard_path = _noisy_chunked_shard_path(cache_dir, strength=strength, shard_id=shard_id)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "frontres_noisy_variant_chunked_shard_v1",
            "strength": float(strength),
            "variants": [noisy_variant_to_record(variant) for variant in variant_list],
        },
        shard_path,
    )
    manifest_records = [
        {
            "path": _relative_posix(shard_path, cache_dir),
            "row": row,
            "segment": segment_to_record(variant.segment),
            "descriptor": perturbation_to_record(variant.descriptor),
        }
        for row, variant in enumerate(variant_list)
    ]
    return shard_path, manifest_records


def write_noisy_variant_chunked_shard_atomic(
    cache_dir: str | Path,
    variants: Iterable[FrontRESNoisyVariant],
    *,
    strength: float,
    shard_id: int = 0,
) -> tuple[Path, list[dict[str, Any]]]:
    variant_list = list(variants)
    for variant in variant_list:
        variant.validate()
    shard_path = _noisy_chunked_shard_path(cache_dir, strength=strength, shard_id=shard_id)
    payload = {
        "format": "frontres_noisy_variant_chunked_shard_v1",
        "strength": float(strength),
        "variants": [noisy_variant_to_record(variant) for variant in variant_list],
    }
    _torch_save_atomic(payload, shard_path)
    manifest_records = [
        {
            "path": _relative_posix(shard_path, cache_dir),
            "row": row,
            "segment": segment_to_record(variant.segment),
            "descriptor": perturbation_to_record(variant.descriptor),
        }
        for row, variant in enumerate(variant_list)
    ]
    return shard_path, manifest_records


def noisy_variant_manifest_record(cache_dir: str | Path, variant: FrontRESNoisyVariant) -> dict[str, Any]:
    variant.validate()
    variant_path = _noisy_variant_path(cache_dir, variant)
    if not variant_path.is_file():
        raise FileNotFoundError(f"noisy variant payload missing before manifest record: {variant_path}")
    return {
        "path": _relative_posix(variant_path, cache_dir),
        "segment": segment_to_record(variant.segment),
        "descriptor": perturbation_to_record(variant.descriptor),
    }


def write_noisy_variant_manifest_records(
    cache_dir: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    strength: float,
    shard_id: int = 0,
) -> Path:
    record_list = list(records)
    manifest_path = _noisy_manifest_path(cache_dir, strength=strength, shard_id=shard_id)
    _torch_save_atomic(
        {
            "format": "frontres_noisy_variant_tree_manifest_v1",
            "strength": float(strength),
            "variants": record_list,
        },
        manifest_path,
    )
    return manifest_path


def write_noisy_variant_manifest(
    cache_dir: str | Path,
    variants: Iterable[FrontRESNoisyVariant],
    *,
    strength: float,
    shard_id: int = 0,
) -> Path:
    variant_list = list(variants)
    for variant in variant_list:
        variant.validate()
    manifest_path = _noisy_manifest_path(cache_dir, strength=strength, shard_id=shard_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = [noisy_variant_manifest_record(cache_dir, variant) for variant in variant_list]
    return write_noisy_variant_manifest_records(cache_dir, records, strength=strength, shard_id=shard_id)


def write_noisy_variant_shard(
    cache_dir: str | Path,
    variants: Iterable[FrontRESNoisyVariant],
    *,
    strength: float,
    shard_id: int = 0,
) -> Path:
    variant_list = list(variants)
    for variant in variant_list:
        variant.validate()
    _, records = write_noisy_variant_chunked_shard(
        cache_dir,
        variant_list,
        strength=strength,
        shard_id=shard_id,
    )
    return write_noisy_variant_manifest_records(cache_dir, records, strength=strength, shard_id=shard_id)


def read_noisy_variant_shard(path: str | Path) -> list[FrontRESNoisyVariant]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") == "frontres_noisy_variant_shard_v1":
        variants = [noisy_variant_from_record(record) for record in payload["variants"]]
    elif payload.get("format") == "frontres_noisy_variant_tree_manifest_v1":
        base_dir = Path(path).parent.parent.parent.parent
        variants = []
        for item in payload["variants"]:
            variant_payload = torch.load(base_dir / item["path"], map_location="cpu", weights_only=False)
            if variant_payload.get("format") != "frontres_noisy_variant_file_v1":
                if variant_payload.get("format") != "frontres_noisy_variant_chunked_shard_v1":
                    raise ValueError(f"unsupported noisy variant file format: {variant_payload.get('format')}")
                variants.append(noisy_variant_from_record(variant_payload["variants"][int(item["row"])]))
            else:
                variants.append(noisy_variant_from_record(variant_payload["variant"]))
    else:
        raise ValueError(f"unsupported noisy variant shard format: {payload.get('format')}")
    for variant in variants:
        variant.validate()
    return variants


def read_noisy_variant_manifest_records(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = Path(path)
    payload = torch.load(manifest_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "frontres_noisy_variant_tree_manifest_v1":
        raise ValueError(f"unsupported noisy variant manifest format: {payload.get('format')}")
    base_dir = manifest_path.parent.parent.parent.parent
    return base_dir, list(payload["variants"])


def read_noisy_variant_record(
    cache_dir: str | Path,
    record: dict[str, Any],
    *,
    shard_cache: FrontRESSegmentShardLRU | None = None,
) -> FrontRESNoisyVariant:
    payload_path = Path(cache_dir) / str(record["path"])
    payload = shard_cache.load(payload_path) if shard_cache is not None else torch.load(
        payload_path,
        map_location="cpu",
        weights_only=False,
    )
    fmt = payload.get("format")
    if fmt == "frontres_noisy_variant_file_v1":
        variant = noisy_variant_from_record(payload["variant"])
    elif fmt == "frontres_noisy_variant_chunked_shard_v1":
        variant = noisy_variant_from_record(payload["variants"][int(record["row"])])
    else:
        raise ValueError(f"unsupported noisy variant payload format: {fmt}")
    variant.validate()
    return variant


def read_noisy_variants_from_records(
    cache_dir: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    shard_cache: FrontRESSegmentShardLRU | None = None,
) -> list[FrontRESNoisyVariant]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        grouped.setdefault(str(record["path"]), []).append(record)
    variants: list[FrontRESNoisyVariant] = []
    for _, group in grouped.items():
        for record in group:
            variants.append(read_noisy_variant_record(cache_dir, record, shard_cache=shard_cache))
    return variants


def clean_resume_key(segment_or_record: FrontRESSegmentIndex | dict[str, Any]) -> tuple[Any, ...]:
    record = _segment_record(segment_or_record)
    return (
        str(record["motion_rel_path"]),
        int(record["start_frame"]),
        int(record["horizon_k"]),
    )


def noisy_resume_key(
    segment_or_record: FrontRESSegmentIndex | dict[str, Any],
    descriptor: FrontRESPerturbationDescriptor | dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    if descriptor is None:
        if not isinstance(segment_or_record, dict) or "descriptor" not in segment_or_record:
            raise ValueError("descriptor is required unless a noisy manifest record is provided")
        descriptor_record = dict(segment_or_record["descriptor"])
    else:
        descriptor_record = _descriptor_record(descriptor)
    return (
        *clean_resume_key(segment_or_record),
        int(descriptor_record["perturbation_id"]),
        float(descriptor_record["strength"]),
        int(descriptor_record["seed"]),
        str(descriptor_record["family"]),
        int(descriptor_record["start_step"]),
        int(descriptor_record["duration"]),
        str(descriptor_record["target"]),
        str(descriptor_record["frame"]),
        json.dumps(dict(descriptor_record.get("params", {})), sort_keys=True, default=str),
    )


def scan_stage1_cache_resume_state(cache_dir: str | Path) -> FrontRESStage1CacheResumeScan:
    root = Path(cache_dir)
    ignored_tmp_paths = tuple(
        sorted(_relative_posix(path, root) for path in root.glob("shards/**/*.tmp") if path.is_file())
    )
    completed_clean: set[tuple[Any, ...]] = set()
    completed_noisy: set[tuple[Any, ...]] = set()
    corrupt_records: list[dict[str, Any]] = []
    clean_manifest_count = 0
    noisy_manifest_count = 0

    for manifest_path in sorted((root / "manifests" / "clean_states").glob("*.pt")):
        clean_manifest_count += 1
        try:
            payload = torch.load(manifest_path, map_location="cpu", weights_only=False)
            if payload.get("format") != "frontres_clean_state_tree_manifest_v1":
                raise ValueError(f"unsupported clean manifest format: {payload.get('format')}")
            for row_idx, record in enumerate(payload.get("entries", [])):
                row_state = _resume_record_state(root, record, kind="clean")
                if row_state == "complete":
                    try:
                        completed_clean.add(clean_resume_key(record))
                    except Exception as exc:
                        corrupt_records.append(
                            _corrupt_record("clean", manifest_path, row_idx, record, f"bad_key:{exc}")
                        )
                elif row_state == "tmp":
                    continue
                else:
                    corrupt_records.append(
                        _corrupt_record("clean", manifest_path, row_idx, record, row_state)
                    )
        except Exception as exc:
            corrupt_records.append(
                {
                    "kind": "clean_manifest",
                    "manifest_path": _relative_posix(manifest_path, root),
                    "error": str(exc),
                }
            )

    for manifest_path in sorted((root / "manifests" / "noisy_variants").glob("**/*.pt")):
        noisy_manifest_count += 1
        try:
            payload = torch.load(manifest_path, map_location="cpu", weights_only=False)
            if payload.get("format") != "frontres_noisy_variant_tree_manifest_v1":
                raise ValueError(f"unsupported noisy manifest format: {payload.get('format')}")
            for row_idx, record in enumerate(payload.get("variants", [])):
                row_state = _resume_record_state(root, record, kind="noisy")
                if row_state == "complete":
                    try:
                        completed_noisy.add(noisy_resume_key(record))
                    except Exception as exc:
                        corrupt_records.append(
                            _corrupt_record("noisy", manifest_path, row_idx, record, f"bad_key:{exc}")
                        )
                elif row_state == "tmp":
                    continue
                else:
                    corrupt_records.append(
                        _corrupt_record("noisy", manifest_path, row_idx, record, row_state)
                    )
        except Exception as exc:
            corrupt_records.append(
                {
                    "kind": "noisy_manifest",
                    "manifest_path": _relative_posix(manifest_path, root),
                    "error": str(exc),
                }
            )

    return FrontRESStage1CacheResumeScan(
        completed_clean_keys=frozenset(completed_clean),
        completed_noisy_keys=frozenset(completed_noisy),
        ignored_tmp_paths=ignored_tmp_paths,
        corrupt_records=tuple(corrupt_records),
        clean_manifest_count=clean_manifest_count,
        noisy_manifest_count=noisy_manifest_count,
    )


def clean_entry_to_record(entry: FrontRESCleanStateEntry) -> dict[str, Any]:
    entry.validate()
    return {
        "segment": segment_to_record(entry.segment),
        "clean_state": rollout_state_to_record(entry.clean_state),
    }


def clean_entry_from_record(record: dict[str, Any]) -> FrontRESCleanStateEntry:
    entry = FrontRESCleanStateEntry(
        segment=segment_from_record(record["segment"]),
        clean_state=rollout_state_from_record(record["clean_state"]),
    )
    entry.validate()
    return entry


def noisy_variant_to_record(variant: FrontRESNoisyVariant) -> dict[str, Any]:
    variant.validate()
    return {
        "segment": segment_to_record(variant.segment),
        "descriptor": perturbation_to_record(variant.descriptor),
        "noisy_state": rollout_state_to_record(variant.noisy_state),
        "noisy_baseline_score": variant.noisy_baseline_score.detach().cpu(),
        "noisy_fall": variant.noisy_fall.detach().cpu(),
        "noisy_rollout_len": variant.noisy_rollout_len.detach().cpu(),
    }


def noisy_variant_from_record(record: dict[str, Any]) -> FrontRESNoisyVariant:
    variant = FrontRESNoisyVariant(
        segment=segment_from_record(record["segment"]),
        descriptor=perturbation_from_record(record["descriptor"]),
        noisy_state=rollout_state_from_record(record["noisy_state"]),
        noisy_baseline_score=record["noisy_baseline_score"].detach().cpu(),
        noisy_fall=record["noisy_fall"].detach().cpu(),
        noisy_rollout_len=record["noisy_rollout_len"].detach().cpu(),
    )
    variant.validate()
    return variant


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


def perturbation_to_record(descriptor: FrontRESPerturbationDescriptor) -> dict[str, Any]:
    descriptor.validate()
    return {
        "perturbation_id": int(descriptor.perturbation_id),
        "segment_id": int(descriptor.segment_id),
        "strength": float(descriptor.strength),
        "seed": int(descriptor.seed),
        "family": str(descriptor.family),
        "start_step": int(descriptor.start_step),
        "duration": int(descriptor.duration),
        "target": str(descriptor.target),
        "frame": str(descriptor.frame),
        "params": dict(descriptor.params),
    }


def perturbation_from_record(record: dict[str, Any]) -> FrontRESPerturbationDescriptor:
    descriptor = FrontRESPerturbationDescriptor(
        perturbation_id=int(record["perturbation_id"]),
        segment_id=int(record["segment_id"]),
        strength=float(record["strength"]),
        seed=int(record["seed"]),
        family=str(record["family"]),
        start_step=int(record["start_step"]),
        duration=int(record["duration"]),
        target=str(record["target"]),
        frame=str(record["frame"]),
        params=dict(record.get("params", {})),
    )
    descriptor.validate()
    return descriptor


def _segment_record(segment_or_record: FrontRESSegmentIndex | dict[str, Any]) -> dict[str, Any]:
    if isinstance(segment_or_record, FrontRESSegmentIndex):
        return segment_to_record(segment_or_record)
    if all(hasattr(segment_or_record, name) for name in ("motion_rel_path", "start_frame", "horizon_k")):
        segment_or_record.validate()
        return {
            "segment_id": int(segment_or_record.segment_id),
            "motion_rel_path": str(segment_or_record.motion_rel_path),
            "motion_num_frames": int(segment_or_record.motion_num_frames),
            "fps": float(segment_or_record.fps),
            "start_frame": int(segment_or_record.start_frame),
            "horizon_k": int(segment_or_record.horizon_k),
        }
    record = dict(segment_or_record)
    if "segment" in record:
        record = dict(record["segment"])
    for name in ("motion_rel_path", "start_frame", "horizon_k"):
        if name not in record:
            raise ValueError(f"segment resume key record missing {name!r}")
    return record


def _descriptor_record(descriptor: FrontRESPerturbationDescriptor | dict[str, Any]) -> dict[str, Any]:
    if isinstance(descriptor, FrontRESPerturbationDescriptor):
        return perturbation_to_record(descriptor)
    if all(
        hasattr(descriptor, name)
        for name in ("perturbation_id", "segment_id", "strength", "seed", "family", "start_step", "duration")
    ):
        descriptor.validate()
        return {
            "perturbation_id": int(descriptor.perturbation_id),
            "segment_id": int(descriptor.segment_id),
            "strength": float(descriptor.strength),
            "seed": int(descriptor.seed),
            "family": str(descriptor.family),
            "start_step": int(descriptor.start_step),
            "duration": int(descriptor.duration),
            "target": str(descriptor.target),
            "frame": str(descriptor.frame),
            "params": dict(descriptor.params),
        }
    record = dict(descriptor)
    for name in (
        "perturbation_id",
        "strength",
        "seed",
        "family",
        "start_step",
        "duration",
        "target",
        "frame",
    ):
        if name not in record:
            raise ValueError(f"noisy resume key record missing {name!r}")
    return record


def _resume_record_state(cache_dir: Path, record: dict[str, Any], *, kind: str) -> str:
    raw_path = str(record.get("path", ""))
    if not raw_path:
        return "missing_path"
    if raw_path.endswith(".tmp"):
        return "tmp"
    payload_path = cache_dir / raw_path
    if not payload_path.is_file():
        return "missing_shard"
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        row = int(record.get("row", 0))
        if kind == "clean":
            return _clean_payload_row_state(payload, row)
        if kind == "noisy":
            return _noisy_payload_row_state(payload, row)
        return "unknown_kind"
    except Exception as exc:
        return f"unreadable:{exc}"


def _clean_payload_row_state(payload: dict[str, Any], row: int) -> str:
    fmt = payload.get("format")
    if fmt == "frontres_clean_state_file_v1":
        clean_entry_from_record(payload["entry"])
        return "complete"
    if fmt != "frontres_clean_state_chunked_shard_v1":
        return f"bad_format:{fmt}"
    entries = list(payload.get("entries", []))
    if row < 0 or row >= len(entries):
        return "row_out_of_range"
    clean_entry_from_record(entries[row])
    return "complete"


def _noisy_payload_row_state(payload: dict[str, Any], row: int) -> str:
    fmt = payload.get("format")
    if fmt == "frontres_noisy_variant_file_v1":
        noisy_variant_from_record(payload["variant"])
        return "complete"
    if fmt != "frontres_noisy_variant_chunked_shard_v1":
        return f"bad_format:{fmt}"
    variants = list(payload.get("variants", []))
    if row < 0 or row >= len(variants):
        return "row_out_of_range"
    noisy_variant_from_record(variants[row])
    return "complete"


def _corrupt_record(kind: str, manifest_path: Path, row_idx: int, record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "manifest_path": str(manifest_path),
        "row_idx": int(row_idx),
        "path": str(record.get("path", "")),
        "reason": str(reason),
    }


def _torch_save_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def rollout_state_to_record(state: FrontRESRobotRolloutState) -> dict[str, torch.Tensor | None]:
    state.validate()
    return {
        "root_pos": state.root_pos.detach().cpu(),
        "root_quat": state.root_quat.detach().cpu(),
        "root_lin_vel": state.root_lin_vel.detach().cpu(),
        "root_ang_vel": state.root_ang_vel.detach().cpu(),
        "joint_pos": state.joint_pos.detach().cpu(),
        "joint_vel": state.joint_vel.detach().cpu(),
        "body_pos_w": state.body_pos_w.detach().cpu(),
        "body_quat_w": state.body_quat_w.detach().cpu(),
        "body_lin_vel_w": state.body_lin_vel_w.detach().cpu(),
        "body_ang_vel_w": state.body_ang_vel_w.detach().cpu(),
        "contact_state": None if state.contact_state is None else state.contact_state.detach().cpu(),
        "action_history": None if state.action_history is None else state.action_history.detach().cpu(),
    }


def rollout_state_from_record(record: dict[str, torch.Tensor | None]) -> FrontRESRobotRolloutState:
    state = FrontRESRobotRolloutState(
        root_pos=_tensor(record["root_pos"]),
        root_quat=_tensor(record["root_quat"]),
        root_lin_vel=_tensor(record["root_lin_vel"]),
        root_ang_vel=_tensor(record["root_ang_vel"]),
        joint_pos=_tensor(record["joint_pos"]),
        joint_vel=_tensor(record["joint_vel"]),
        body_pos_w=_tensor(record["body_pos_w"]),
        body_quat_w=_tensor(record["body_quat_w"]),
        body_lin_vel_w=_tensor(record["body_lin_vel_w"]),
        body_ang_vel_w=_tensor(record["body_ang_vel_w"]),
        contact_state=None if record.get("contact_state") is None else _tensor(record["contact_state"]),
        action_history=None if record.get("action_history") is None else _tensor(record["action_history"]),
    )
    state.validate()
    return state


def _tensor(value: torch.Tensor | None) -> torch.Tensor:
    if value is None:
        raise ValueError("required tensor field is missing")
    return value.detach().cpu()


def _clean_manifest_path(cache_dir: str | Path, shard_id: int) -> Path:
    return Path(cache_dir) / "manifests" / "clean_states" / f"shard_{int(shard_id):06d}.pt"


def _noisy_manifest_path(cache_dir: str | Path, *, strength: float, shard_id: int) -> Path:
    return Path(cache_dir) / "manifests" / "noisy_variants" / _strength_dir(strength) / f"shard_{int(shard_id):06d}.pt"


def _clean_chunked_shard_path(cache_dir: str | Path, shard_id: int) -> Path:
    return Path(cache_dir) / "shards" / "clean_states" / f"shard_{int(shard_id):06d}.pt"


def _noisy_chunked_shard_path(cache_dir: str | Path, *, strength: float, shard_id: int) -> Path:
    return Path(cache_dir) / "shards" / "noisy_variants" / _strength_dir(strength) / f"shard_{int(shard_id):06d}.pt"


def _clean_segment_path(cache_dir: str | Path, segment: FrontRESSegmentIndex) -> Path:
    return _segment_dir(cache_dir, segment) / "clean.pt"


def _noisy_variant_path(cache_dir: str | Path, variant: FrontRESNoisyVariant) -> Path:
    return (
        _segment_dir(cache_dir, variant.segment)
        / "noisy_variants"
        / _strength_dir(float(variant.descriptor.strength))
        / f"perturbation_{int(variant.perturbation_id):08d}.pt"
    )


def _segment_dir(cache_dir: str | Path, segment: FrontRESSegmentIndex) -> Path:
    segment.validate()
    motion_rel = _safe_motion_rel_path(segment.motion_rel_path)
    motion_dir = motion_rel.with_suffix("")
    return (
        Path(cache_dir)
        / motion_dir
        / f"segment_{int(segment.segment_id):08d}_start_{int(segment.start_frame):08d}_k_{int(segment.horizon_k):04d}"
    )


def _safe_motion_rel_path(motion_rel_path: str) -> Path:
    raw = str(motion_rel_path).replace("\\", "/")
    rel = Path(raw)
    if rel.is_absolute() or any(part in {"", ".."} for part in rel.parts):
        raise ValueError(f"motion_rel_path must be a safe relative path, got {motion_rel_path!r}")
    return rel


def _relative_posix(path: Path, root: str | Path) -> str:
    return path.relative_to(Path(root)).as_posix()


def _strength_dir(strength: float) -> str:
    text = f"{float(strength):.6f}".rstrip("0").rstrip(".")
    return "strength_" + text.replace("-", "neg_").replace(".", "p")
