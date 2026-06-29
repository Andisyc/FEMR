from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

try:
    from rsl_rl.frontres.frontres_segment_cache_schema import (
        FrontRESPerturbationDescriptor,
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
    FrontRESPerturbationDescriptor = _SCHEMA_MODULE.FrontRESPerturbationDescriptor
    FrontRESSegmentIndex = _SCHEMA_MODULE.FrontRESSegmentIndex


@dataclass(frozen=True)
class FrontRESPerturbationCurriculumConfig:
    strengths: tuple[float, ...]
    variants_per_strength: int = 1
    base_seed: int = 0
    mode: str = "discrete_bank"
    family: str = "external_push"
    target: str = "torso_link"
    frame: str = "world"
    start_step: int = 0
    duration: int = 2

    def validate(self) -> None:
        if not self.strengths:
            raise ValueError("strengths must be non-empty")
        if any(float(strength) < 0.0 for strength in self.strengths):
            raise ValueError(f"strengths must be non-negative, got {self.strengths}")
        if int(self.variants_per_strength) <= 0:
            raise ValueError(f"variants_per_strength must be positive, got {self.variants_per_strength}")
        if self.mode not in {"discrete_bank"}:
            raise ValueError(f"mode must be discrete_bank, got {self.mode}")
        if not self.family:
            raise ValueError("family must be non-empty")
        if not self.target:
            raise ValueError("target must be non-empty")
        if self.frame not in {"world", "local", "joint"}:
            raise ValueError(f"frame must be world, local, or joint, got {self.frame}")
        if int(self.start_step) < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")
        if int(self.duration) <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")


@dataclass(frozen=True)
class FrontRESBankDescriptorConfig:
    variants_per_record: int = 1
    base_seed: int = 0
    target: str = "torso_link"
    frame: str = "world"
    start_step: int = 0
    duration: int = 2
    temporal_mode: str = "single"
    burst_min_steps: int = 4
    burst_max_steps: int = 8

    def validate(self) -> None:
        if int(self.variants_per_record) <= 0:
            raise ValueError(f"variants_per_record must be positive, got {self.variants_per_record}")
        if not self.target:
            raise ValueError("target must be non-empty")
        if self.frame not in {"world", "local", "joint"}:
            raise ValueError(f"frame must be world, local, or joint, got {self.frame}")
        if int(self.start_step) < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")
        if int(self.duration) <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if not self.temporal_mode:
            raise ValueError("temporal_mode must be non-empty")
        if int(self.burst_min_steps) <= 0:
            raise ValueError(f"burst_min_steps must be positive, got {self.burst_min_steps}")
        if int(self.burst_max_steps) < int(self.burst_min_steps):
            raise ValueError(
                f"burst_max_steps must be >= burst_min_steps, got {self.burst_max_steps} < {self.burst_min_steps}"
            )


def parse_strengths(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        strengths = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    else:
        strengths = tuple(float(item) for item in value)
    cfg = FrontRESPerturbationCurriculumConfig(strengths=strengths)
    cfg.validate()
    return cfg.strengths


def build_perturbation_descriptors(
    segments: Iterable[FrontRESSegmentIndex],
    cfg: FrontRESPerturbationCurriculumConfig,
    *,
    start_perturbation_id: int = 0,
) -> list[FrontRESPerturbationDescriptor]:
    cfg.validate()
    next_id = int(start_perturbation_id)
    descriptors: list[FrontRESPerturbationDescriptor] = []
    for segment in segments:
        segment.validate()
        for strength_index, strength in enumerate(cfg.strengths):
            for variant_index in range(int(cfg.variants_per_strength)):
                seed = descriptor_seed(
                    cfg.base_seed,
                    segment_id=int(segment.segment_id),
                    strength_index=strength_index,
                    variant_index=variant_index,
                )
                descriptor = FrontRESPerturbationDescriptor(
                    perturbation_id=next_id,
                    segment_id=int(segment.segment_id),
                    strength=float(strength),
                    seed=seed,
                    family=cfg.family,
                    start_step=int(cfg.start_step),
                    duration=int(cfg.duration),
                    target=cfg.target,
                    frame=cfg.frame,
                    params=_sample_params(
                        seed=seed,
                        strength=float(strength),
                        frame=cfg.frame,
                        mode=cfg.mode,
                        family=cfg.family,
                        level_index=strength_index,
                        variant_index=variant_index,
                    ),
                )
                descriptor.validate()
                descriptors.append(descriptor)
                next_id += 1
    return descriptors


def build_perturbation_descriptors_from_curriculum_bank(
    segments: Iterable[FrontRESSegmentIndex],
    bank: Any,
    cfg: FrontRESBankDescriptorConfig,
    *,
    start_perturbation_id: int = 0,
) -> list[FrontRESPerturbationDescriptor]:
    cfg.validate()
    if hasattr(bank, "validate"):
        bank.validate()
    records = tuple(getattr(bank, "records", ()))
    if not records:
        raise ValueError("curriculum bank must contain records")
    next_id = int(start_perturbation_id)
    descriptors: list[FrontRESPerturbationDescriptor] = []
    for segment in segments:
        segment.validate()
        for record in records:
            if hasattr(record, "validate"):
                record.validate()
            for variant_index in range(int(cfg.variants_per_record)):
                seed = descriptor_seed(
                    cfg.base_seed,
                    segment_id=int(segment.segment_id),
                    strength_index=int(record.bank_id),
                    variant_index=variant_index,
                )
                family_group = tuple(str(item) for item in record.family_group)
                descriptor = FrontRESPerturbationDescriptor(
                    perturbation_id=next_id,
                    segment_id=int(segment.segment_id),
                    strength=float(record.actual_dr_scale),
                    seed=seed,
                    family="+".join(family_group),
                    start_step=int(cfg.start_step),
                    duration=int(cfg.duration),
                    target=cfg.target,
                    frame=cfg.frame,
                    params=_sample_bank_record_params(
                        seed=seed,
                        record=record,
                        family_group=family_group,
                        frame=cfg.frame,
                        variant_index=variant_index,
                        temporal_mode=cfg.temporal_mode,
                        burst_min_steps=int(cfg.burst_min_steps),
                        burst_max_steps=int(cfg.burst_max_steps),
                    ),
                )
                descriptor.validate()
                descriptors.append(descriptor)
                next_id += 1
    return descriptors


def descriptor_seed(base_seed: int, *, segment_id: int, strength_index: int, variant_index: int) -> int:
    return (
        int(base_seed)
        + int(segment_id) * 1_000_003
        + int(strength_index) * 10_007
        + int(variant_index) * 101
    )


def descriptor_probe(descriptors: Sequence[FrontRESPerturbationDescriptor]) -> dict[str, Any]:
    for descriptor in descriptors:
        descriptor.validate()
    perturbation_ids = [int(item.perturbation_id) for item in descriptors]
    segment_ids = [int(item.segment_id) for item in descriptors]
    strengths = [float(item.strength) for item in descriptors]
    return {
        "count": len(descriptors),
        "perturbation_ids": perturbation_ids,
        "segment_ids": segment_ids,
        "strengths": strengths,
        "levels": [dict(item.params).get("level_name") for item in descriptors],
        "families": [item.family for item in descriptors],
        "family_groups": [tuple(dict(item.params).get("family_group", ())) for item in descriptors],
        "mix_classes": [dict(item.params).get("mix_class") for item in descriptors],
        "actual_dr_scales": [dict(item.params).get("actual_dr_scale") for item in descriptors],
        "roles": [dict(item.params).get("perturbation_role") for item in descriptors],
        "unique_perturbation_ids": len(set(perturbation_ids)),
        "first_seed": None if not descriptors else int(descriptors[0].seed),
        "first_params": None if not descriptors else dict(descriptors[0].params),
    }


def descriptor_signature(descriptor: FrontRESPerturbationDescriptor) -> tuple[Any, ...]:
    descriptor.validate()
    return (
        int(descriptor.perturbation_id),
        int(descriptor.segment_id),
        float(descriptor.strength),
        int(descriptor.seed),
        descriptor.family,
        int(descriptor.start_step),
        int(descriptor.duration),
        descriptor.target,
        descriptor.frame,
        descriptor.params.get("curriculum_mode", ""),
        int(descriptor.params.get("level_index", -1)),
        descriptor.params.get("level_name", ""),
        int(descriptor.params.get("variant_index", -1)),
        tuple(round(float(item), 8) for item in descriptor.params.get("axis", ())),
        round(float(descriptor.params.get("signed_magnitude", 0.0)), 8),
        tuple(descriptor.params.get("family_group", ())),
        descriptor.params.get("mix_class", ""),
        int(descriptor.params.get("mix_class_index", -99)),
        round(float(descriptor.params.get("frontier_scale", -1.0)), 8),
        round(float(descriptor.params.get("dr_factor", -1.0)), 8),
        round(float(descriptor.params.get("actual_dr_scale", -1.0)), 8),
        descriptor.params.get("perturbation_role", ""),
        descriptor.params.get("temporal_mode", ""),
        int(descriptor.params.get("burst_min_steps", -1)),
        int(descriptor.params.get("burst_max_steps", -1)),
    )


def _sample_params(
    *,
    seed: int,
    strength: float,
    frame: str,
    mode: str,
    family: str,
    level_index: int,
    variant_index: int,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    raw = [rng.uniform(-1.0, 1.0) for _ in range(3)]
    norm = sum(item * item for item in raw) ** 0.5
    if norm <= 1.0e-8 or float(strength) == 0.0:
        axis = [0.0, 0.0, 0.0]
    else:
        axis = [float(item / norm) for item in raw]
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return {
        "curriculum_mode": mode,
        "family": family,
        "level_index": int(level_index),
        "level_name": f"level_{int(level_index):02d}",
        "level_strength": float(strength),
        "variant_index": int(variant_index),
        "axis": axis,
        "signed_magnitude": float(sign * float(strength)),
        "frame": frame,
    }


def _sample_bank_record_params(
    *,
    seed: int,
    record: Any,
    family_group: tuple[str, ...],
    frame: str,
    variant_index: int,
    temporal_mode: str,
    burst_min_steps: int,
    burst_max_steps: int,
) -> dict[str, Any]:
    params = _sample_params(
        seed=seed,
        strength=float(record.actual_dr_scale),
        frame=frame,
        mode="hrl_curriculum_bank",
        family="+".join(family_group),
        level_index=int(record.bank_id),
        variant_index=int(variant_index),
    )
    params.update(
        {
            "descriptor_schema_version": 2,
            "bank_id": int(record.bank_id),
            "family_group": family_group,
            "mix_class": str(record.mix_class),
            "mix_class_index": int(record.mix_class_index),
            "frontier_scale": float(record.frontier_scale),
            "dr_factor": float(record.dr_factor),
            "actual_dr_scale": float(record.actual_dr_scale),
            "perturbation_role": str(record.role),
            "seq_idx": int(record.seq_idx),
            "env_slot": int(record.env_slot),
            "temporal_mode": str(temporal_mode),
            "burst_min_steps": int(burst_min_steps),
            "burst_max_steps": int(burst_max_steps),
        }
    )
    return params
