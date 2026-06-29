#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_segment_cache_perturbation.py"
spec = importlib.util.spec_from_file_location("frontres_segment_cache_perturbation", MODULE_PATH)
perturb = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = perturb
spec.loader.exec_module(perturb)

FrontRESSegmentIndex = perturb.FrontRESSegmentIndex
FrontRESPerturbationCurriculumConfig = perturb.FrontRESPerturbationCurriculumConfig


def _segments() -> list[FrontRESSegmentIndex]:
    return [
        FrontRESSegmentIndex(
            segment_id=7,
            motion_rel_path="KIT/359/motion_a.npz",
            motion_num_frames=20,
            fps=30.0,
            start_frame=3,
            horizon_k=4,
        ),
        FrontRESSegmentIndex(
            segment_id=8,
            motion_rel_path="CMU/001/motion_b.npz",
            motion_num_frames=30,
            fps=60.0,
            start_frame=5,
            horizon_k=4,
        ),
    ]


def test_curriculum_descriptor_is_reproducible_and_unique() -> None:
    cfg = FrontRESPerturbationCurriculumConfig(
        strengths=perturb.parse_strengths("0.0,0.5"),
        variants_per_strength=2,
        base_seed=123,
        duration=3,
    )
    descriptors_a = perturb.build_perturbation_descriptors(_segments(), cfg)
    descriptors_b = perturb.build_perturbation_descriptors(_segments(), cfg)
    probe = perturb.descriptor_probe(descriptors_a)
    print(
        "[cache_perturbation trace] build "
        f"count={probe['count']} "
        f"ids={probe['perturbation_ids']} "
        f"segment_ids={probe['segment_ids']} "
        f"strengths={probe['strengths']} "
        f"first_seed={probe['first_seed']} "
        f"first_params={probe['first_params']}"
    )
    assert probe["count"] == 8
    assert probe["unique_perturbation_ids"] == 8
    assert probe["perturbation_ids"] == list(range(8))
    assert probe["segment_ids"] == [7, 7, 7, 7, 8, 8, 8, 8]
    assert probe["strengths"] == [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5]
    assert [perturb.descriptor_signature(item) for item in descriptors_a] == [
        perturb.descriptor_signature(item) for item in descriptors_b
    ]
    assert all(item.duration == 3 for item in descriptors_a)
    assert all(item.target == "torso_link" for item in descriptors_a)


def test_curriculum_seed_changes_nonzero_descriptor_params() -> None:
    cfg_a = FrontRESPerturbationCurriculumConfig(strengths=(0.5,), variants_per_strength=1, base_seed=123)
    cfg_b = FrontRESPerturbationCurriculumConfig(strengths=(0.5,), variants_per_strength=1, base_seed=124)
    desc_a = perturb.build_perturbation_descriptors(_segments()[:1], cfg_a)[0]
    desc_b = perturb.build_perturbation_descriptors(_segments()[:1], cfg_b)[0]
    print(
        "[cache_perturbation trace] seed_change "
        f"seed_a={desc_a.seed} params_a={desc_a.params} "
        f"seed_b={desc_b.seed} params_b={desc_b.params}"
    )
    assert desc_a.seed != desc_b.seed
    assert perturb.descriptor_signature(desc_a) != perturb.descriptor_signature(desc_b)
    assert desc_a.params != desc_b.params


def test_curriculum_rejects_invalid_strengths() -> None:
    try:
        perturb.parse_strengths("0.0,-0.5")
    except ValueError as exc:
        print(f"[cache_perturbation trace] rejected_strengths={exc}")
        assert "non-negative" in str(exc)
        return
    raise AssertionError("negative perturbation strength should be rejected")


if __name__ == "__main__":
    test_curriculum_descriptor_is_reproducible_and_unique()
    test_curriculum_seed_changes_nonzero_descriptor_params()
    test_curriculum_rejects_invalid_strengths()
    print("PASS: FrontRES perturbation curriculum descriptors are reproducible and indexed.")
