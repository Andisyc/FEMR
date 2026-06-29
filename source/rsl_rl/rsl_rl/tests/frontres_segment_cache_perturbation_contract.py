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
FrontRESBankDescriptorConfig = perturb.FrontRESBankDescriptorConfig


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
        f"levels={probe['levels']} "
        f"first_seed={probe['first_seed']} "
        f"first_params={probe['first_params']}"
    )
    assert probe["count"] == 8
    assert probe["unique_perturbation_ids"] == 8
    assert probe["perturbation_ids"] == list(range(8))
    assert probe["segment_ids"] == [7, 7, 7, 7, 8, 8, 8, 8]
    assert probe["strengths"] == [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5]
    assert probe["levels"] == [
        "level_00",
        "level_00",
        "level_01",
        "level_01",
        "level_00",
        "level_00",
        "level_01",
        "level_01",
    ]
    assert probe["first_params"]["curriculum_mode"] == "discrete_bank"
    assert probe["first_params"]["level_index"] == 0
    assert probe["first_params"]["level_name"] == "level_00"
    assert probe["first_params"]["variant_index"] == 0
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


class _Bank:
    def __init__(self, records):
        self.records = tuple(records)

    def validate(self) -> None:
        assert self.records
        for record in self.records:
            record.validate()


class _BankRecord:
    def __init__(
        self,
        *,
        bank_id: int,
        family_group: tuple[str, ...],
        mix_class: str,
        mix_class_index: int,
        frontier_scale: float,
        dr_factor: float,
        actual_dr_scale: float,
        role: str,
        seq_idx: int,
        env_slot: int,
    ):
        self.bank_id = bank_id
        self.family_group = family_group
        self.mix_class = mix_class
        self.mix_class_index = mix_class_index
        self.frontier_scale = frontier_scale
        self.dr_factor = dr_factor
        self.actual_dr_scale = actual_dr_scale
        self.role = role
        self.seq_idx = seq_idx
        self.env_slot = env_slot

    def validate(self) -> None:
        assert self.bank_id >= 0
        assert self.family_group
        assert self.mix_class in {"easy", "frontier", "hard"}
        assert self.mix_class_index in {0, 1, 2}
        assert self.frontier_scale >= 0.0
        assert self.dr_factor >= 0.0
        assert self.actual_dr_scale >= 0.0
        assert self.role in {"train", "boundary_diagnostic"}


def test_hrl_curriculum_bank_records_are_written_into_descriptor_params() -> None:
    bank = _Bank(
        [
            _BankRecord(
                bank_id=0,
                family_group=("planar", "yaw"),
                mix_class="frontier",
                mix_class_index=1,
                frontier_scale=2.0,
                dr_factor=1.0,
                actual_dr_scale=2.0,
                role="train",
                seq_idx=17,
                env_slot=0,
            ),
            _BankRecord(
                bank_id=1,
                family_group=("local_rp",),
                mix_class="hard",
                mix_class_index=2,
                frontier_scale=2.0,
                dr_factor=1.08,
                actual_dr_scale=2.16,
                role="boundary_diagnostic",
                seq_idx=17,
                env_slot=1,
            ),
        ]
    )
    cfg = FrontRESBankDescriptorConfig(
        variants_per_record=2,
        base_seed=555,
        duration=4,
        temporal_mode="single",
        burst_min_steps=4,
        burst_max_steps=8,
    )
    descriptors = perturb.build_perturbation_descriptors_from_curriculum_bank(_segments()[:1], bank, cfg)
    probe = perturb.descriptor_probe(descriptors)
    print(
        "[cache_perturbation trace] hrl_bank "
        f"count={probe['count']} "
        f"families={probe['families']} "
        f"family_groups={probe['family_groups']} "
        f"mix_classes={probe['mix_classes']} "
        f"actual_dr_scales={probe['actual_dr_scales']} "
        f"roles={probe['roles']} "
        f"first_params={probe['first_params']}"
    )
    assert probe["count"] == 4
    assert probe["segment_ids"] == [7, 7, 7, 7]
    assert probe["families"] == ["planar+yaw", "planar+yaw", "local_rp", "local_rp"]
    assert probe["family_groups"] == [
        ("planar", "yaw"),
        ("planar", "yaw"),
        ("local_rp",),
        ("local_rp",),
    ]
    assert probe["mix_classes"] == ["frontier", "frontier", "hard", "hard"]
    assert probe["actual_dr_scales"] == [2.0, 2.0, 2.16, 2.16]
    assert probe["roles"] == ["train", "train", "boundary_diagnostic", "boundary_diagnostic"]
    assert probe["first_params"]["curriculum_mode"] == "hrl_curriculum_bank"
    assert probe["first_params"]["frontier_scale"] == 2.0
    assert probe["first_params"]["dr_factor"] == 1.0
    assert probe["first_params"]["temporal_mode"] == "single"
    assert probe["first_params"]["burst_min_steps"] == 4
    assert probe["first_params"]["burst_max_steps"] == 8
    assert len({perturb.descriptor_signature(item) for item in descriptors}) == 4


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
    test_hrl_curriculum_bank_records_are_written_into_descriptor_params()
    test_curriculum_rejects_invalid_strengths()
    print("PASS: FrontRES perturbation curriculum descriptors are reproducible and indexed.")
