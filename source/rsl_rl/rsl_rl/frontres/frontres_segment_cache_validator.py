from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any


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
    from rsl_rl.frontres.frontres_segment_cache_indexer import read_amass_segment_index
    from rsl_rl.frontres.frontres_segment_cache_io import (
        read_cache_metadata,
        read_clean_state_shard,
        read_noisy_variant_shard,
    )
except ModuleNotFoundError:
    _indexer = _load_same_dir("frontres_segment_cache_indexer")
    _cache_io = _load_same_dir("frontres_segment_cache_io")
    read_amass_segment_index = _indexer.read_amass_segment_index
    read_cache_metadata = _cache_io.read_cache_metadata
    read_clean_state_shard = _cache_io.read_clean_state_shard
    read_noisy_variant_shard = _cache_io.read_noisy_variant_shard


@dataclass(frozen=True)
class FrontRESStage1CacheValidationResult:
    cache_dir: str
    metadata_stage: str
    perturbation_curriculum_mode: str
    segment_count: int
    clean_count: int
    noisy_count: int
    strength_counts: dict[float, int]
    descriptor_count: int
    train_count: int
    boundary_diagnostic_count: int
    clean_shard_path: str
    noisy_shard_paths: dict[float, str]

    def probe(self) -> dict[str, Any]:
        return {
            "cache_dir": self.cache_dir,
            "metadata_stage": self.metadata_stage,
            "perturbation_curriculum_mode": self.perturbation_curriculum_mode,
            "segment_count": self.segment_count,
            "clean_count": self.clean_count,
            "noisy_count": self.noisy_count,
            "strength_counts": dict(self.strength_counts),
            "descriptor_count": self.descriptor_count,
            "train_count": self.train_count,
            "boundary_diagnostic_count": self.boundary_diagnostic_count,
            "clean_shard_path": self.clean_shard_path,
            "noisy_shard_paths": dict(self.noisy_shard_paths),
        }


def validate_stage1_cache_artifacts(cache_dir: str | Path) -> FrontRESStage1CacheValidationResult:
    root = Path(cache_dir)
    metadata = read_cache_metadata(root)
    segments = read_amass_segment_index(root / "segment_index.jsonl")
    clean_shard_id = int(metadata.get("clean_shard_id", 0))
    noisy_shard_id = int(metadata.get("noisy_shard_id", 0))
    clean_path = root / "manifests" / "clean_states" / f"shard_{clean_shard_id:06d}.pt"
    clean_entries = read_clean_state_shard(clean_path)
    strengths = [float(item) for item in metadata.get("strengths", [])]
    noisy_shard_paths: dict[float, str] = {}
    strength_counts: dict[float, int] = {}
    descriptor_count = 0
    train_count = 0
    boundary_count = 0
    mode = str(metadata.get("perturbation_curriculum_mode", ""))
    for strength in strengths:
        noisy_path = (
            root
            / "manifests"
            / "noisy_variants"
            / _strength_dir(float(strength))
            / f"shard_{noisy_shard_id:06d}.pt"
        )
        variants = read_noisy_variant_shard(noisy_path)
        noisy_shard_paths[float(strength)] = str(noisy_path)
        strength_counts[float(strength)] = len(variants)
        descriptor_count += len(variants)
        for variant in variants:
            params = dict(variant.descriptor.params)
            if mode and params.get("curriculum_mode") != mode:
                raise ValueError(
                    "noisy descriptor curriculum mode does not match metadata: "
                    f"descriptor={params.get('curriculum_mode')} metadata={mode}"
                )
            role = str(params.get("perturbation_role", "train"))
            if role == "boundary_diagnostic":
                boundary_count += 1
            elif role == "train":
                train_count += 1
            else:
                raise ValueError(f"unsupported perturbation_role in cache readback: {role}")
    _assert_equal("metadata segment_count", int(metadata.get("segment_count", -1)), len(segments))
    _assert_equal("metadata clean_count", int(metadata.get("clean_count", -1)), len(clean_entries))
    _assert_equal("metadata noisy_count", int(metadata.get("noisy_count", -1)), descriptor_count)
    if int(metadata.get("segment_count", -1)) != int(metadata.get("clean_count", -2)):
        raise ValueError(
            "Stage 1 cache must have one clean state per segment: "
            f"segment_count={metadata.get('segment_count')} clean_count={metadata.get('clean_count')}"
        )
    segment_ids = {int(segment.segment_id) for segment in segments}
    clean_ids = {int(entry.segment_id) for entry in clean_entries}
    if segment_ids != clean_ids:
        raise ValueError(f"clean segment ids do not match segment index: segments={segment_ids} clean={clean_ids}")
    return FrontRESStage1CacheValidationResult(
        cache_dir=str(root),
        metadata_stage=str(metadata.get("stage", "")),
        perturbation_curriculum_mode=mode,
        segment_count=len(segments),
        clean_count=len(clean_entries),
        noisy_count=descriptor_count,
        strength_counts=strength_counts,
        descriptor_count=descriptor_count,
        train_count=train_count,
        boundary_diagnostic_count=boundary_count,
        clean_shard_path=str(clean_path),
        noisy_shard_paths=noisy_shard_paths,
    )


def format_stage1_cache_validation_probe(result: FrontRESStage1CacheValidationResult) -> str:
    probe = result.probe()
    return (
        "[FrontRES Stage1 Segment Cache] cache_validate "
        f"stage={probe['metadata_stage']} "
        f"mode={probe['perturbation_curriculum_mode']} "
        f"segment_count={probe['segment_count']} "
        f"clean_count={probe['clean_count']} "
        f"noisy_count={probe['noisy_count']} "
        f"strength_counts={probe['strength_counts']} "
        f"train_count={probe['train_count']} "
        f"boundary_diagnostic_count={probe['boundary_diagnostic_count']} "
        f"clean_shard_path={probe['clean_shard_path']} "
        f"noisy_strength_count={len(probe['noisy_shard_paths'])}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a FrontRES Stage 1 Segment cache without IsaacLab.")
    parser.add_argument("cache_dir", type=str, help="Stage 1 cache directory, e.g. /hdd1/cyx/AMASS_G1Segment")
    parser.add_argument("--expect-mode", type=str, default=None, help="Expected perturbation_curriculum_mode.")
    parser.add_argument("--min-segments", type=int, default=1, help="Minimum segment count required.")
    parser.add_argument("--min-noisy", type=int, default=1, help="Minimum noisy variant count required.")
    parser.add_argument(
        "--require-boundary-diagnostic",
        action="store_true",
        default=False,
        help="Require at least one boundary_diagnostic perturbation.",
    )
    args = parser.parse_args(argv)
    result = validate_stage1_cache_artifacts(args.cache_dir)
    print(format_stage1_cache_validation_probe(result), flush=True)
    if args.expect_mode is not None and result.perturbation_curriculum_mode != args.expect_mode:
        raise ValueError(
            f"unexpected perturbation mode: expected={args.expect_mode} observed={result.perturbation_curriculum_mode}"
        )
    if result.segment_count < int(args.min_segments):
        raise ValueError(f"too few segments: min={args.min_segments} observed={result.segment_count}")
    if result.noisy_count < int(args.min_noisy):
        raise ValueError(f"too few noisy variants: min={args.min_noisy} observed={result.noisy_count}")
    if args.require_boundary_diagnostic and result.boundary_diagnostic_count <= 0:
        raise ValueError("boundary_diagnostic perturbations are required but none were found")
    return 0


def _assert_equal(name: str, expected: int, observed: int) -> None:
    if int(expected) != int(observed):
        raise ValueError(f"{name} mismatch: metadata={expected} readback={observed}")


def _strength_dir(strength: float) -> str:
    text = f"{float(strength):.6f}".rstrip("0").rstrip(".")
    return "strength_" + text.replace("-", "neg_").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
