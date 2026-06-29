from __future__ import annotations

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


def write_cache_metadata(cache_dir: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "metadata.json"
    payload = {"format": "frontres_segment_cache_v1", **metadata}
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return metadata_path


def read_cache_metadata(cache_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(cache_dir) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "frontres_segment_cache_v1":
        raise ValueError(f"unsupported FrontRES segment cache format: {payload.get('format')}")
    return payload


def write_clean_state_shard(
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
    records = []
    for entry in entry_list:
        entry_path = _clean_segment_path(cache_dir, entry.segment)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "frontres_clean_state_file_v1",
                "entry": clean_entry_to_record(entry),
            },
            entry_path,
        )
        records.append(
            {
                "path": _relative_posix(entry_path, cache_dir),
                "segment": segment_to_record(entry.segment),
            }
        )
    torch.save(
        {
            "format": "frontres_clean_state_tree_manifest_v1",
            "entries": records,
        },
        manifest_path,
    )
    return manifest_path


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
                raise ValueError(f"unsupported clean state file format: {entry_payload.get('format')}")
            entries.append(clean_entry_from_record(entry_payload["entry"]))
    else:
        raise ValueError(f"unsupported clean state shard format: {payload.get('format')}")
    for entry in entries:
        entry.validate()
    return entries


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
    manifest_path = _noisy_manifest_path(cache_dir, strength=strength, shard_id=shard_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for variant in variant_list:
        variant_path = _noisy_variant_path(cache_dir, variant)
        variant_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "frontres_noisy_variant_file_v1",
                "variant": noisy_variant_to_record(variant),
            },
            variant_path,
        )
        records.append(
            {
                "path": _relative_posix(variant_path, cache_dir),
                "segment": segment_to_record(variant.segment),
                "descriptor": perturbation_to_record(variant.descriptor),
            }
        )
    torch.save(
        {
            "format": "frontres_noisy_variant_tree_manifest_v1",
            "strength": float(strength),
            "variants": records,
        },
        manifest_path,
    )
    return manifest_path


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
                raise ValueError(f"unsupported noisy variant file format: {variant_payload.get('format')}")
            variants.append(noisy_variant_from_record(variant_payload["variant"]))
    else:
        raise ValueError(f"unsupported noisy variant shard format: {payload.get('format')}")
    for variant in variants:
        variant.validate()
    return variants


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
