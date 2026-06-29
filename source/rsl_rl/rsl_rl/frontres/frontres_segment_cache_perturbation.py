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
                    params=_sample_params(seed=seed, strength=float(strength), frame=cfg.frame),
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
        tuple(round(float(item), 8) for item in descriptor.params.get("axis", ())),
        round(float(descriptor.params.get("signed_magnitude", 0.0)), 8),
    )


def _sample_params(*, seed: int, strength: float, frame: str) -> dict[str, Any]:
    rng = random.Random(int(seed))
    raw = [rng.uniform(-1.0, 1.0) for _ in range(3)]
    norm = sum(item * item for item in raw) ** 0.5
    if norm <= 1.0e-8 or float(strength) == 0.0:
        axis = [0.0, 0.0, 0.0]
    else:
        axis = [float(item / norm) for item in raw]
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return {
        "axis": axis,
        "signed_magnitude": float(sign * float(strength)),
        "frame": frame,
    }
