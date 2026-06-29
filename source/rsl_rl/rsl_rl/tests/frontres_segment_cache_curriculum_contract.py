#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = (
    ROOT
    / "source"
    / "rsl_rl"
    / "rsl_rl"
    / "frontres"
    / "frontres_segment_cache_curriculum.py"
)
spec = importlib.util.spec_from_file_location("frontres_segment_cache_curriculum", MODULE_PATH)
cache_curriculum = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = cache_curriculum
spec.loader.exec_module(cache_curriculum)


def _cfg() -> dict[str, float | bool]:
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
    }


def test_stage1_bank_traces_hrl_curriculum_chain() -> None:
    bank_cfg = cache_curriculum.FrontRESStage1CurriculumBankConfig(
        frontier_scale=2.0,
        dr_min=1.25,
        dr_max=4.5,
        n_train=16,
        progress=0.8,
        seq_idx=17,
        active_dims=[0, 1, 3, 4, 5],
        boundary_stats={"safe": 0.2, "repair": 0.55, "broken": 0.1, "positive_gain": 0.7},
    )
    bank = cache_curriculum.build_stage1_curriculum_bank(_cfg(), bank_cfg)
    probe = cache_curriculum.stage1_curriculum_bank_probe(bank)
    print(
        "[cache_curriculum trace] bank "
        f"record_count={probe['record_count']} "
        f"allowed_bases={probe['allowed_bases']} "
        f"active_modes={probe['active_modes']} "
        f"complexity={probe['complexity']} "
        f"mix_classes={probe['mix_classes']} "
        f"dr_factors={probe['dr_factors']} "
        f"roles={probe['roles']} "
        f"mix_diag={probe['mix_diag']}"
    )
    assert probe["record_count"] == 16
    assert probe["allowed_bases"] == ("planar", "yaw", "local_rp")
    assert set(probe["active_modes"]).issubset(set(probe["allowed_bases"]))
    assert set(probe["mix_classes"]).issubset({"easy", "frontier", "hard"})
    assert set(probe["dr_factors"]).issubset({0.75, 1.0, 1.08})
    assert "hard" in probe["mix_classes"]
    for mix_class, role in zip(probe["mix_classes"], probe["roles"]):
        if mix_class == "hard":
            assert role == "boundary_diagnostic"
        else:
            assert role == "train"
    assert abs(sum(probe["mix_diag"][key] for key in ("easy", "frontier", "hard")) - 1.0) < 1e-6


def test_stage1_bank_respects_active_dims_and_hard_training_override() -> None:
    bank_cfg = cache_curriculum.FrontRESStage1CurriculumBankConfig(
        frontier_scale=2.0,
        dr_min=1.25,
        dr_max=4.5,
        n_train=16,
        progress=0.8,
        seq_idx=17,
        active_dims=[3, 4],
        boundary_stats={"safe": 0.2, "repair": 0.55, "broken": 0.1, "positive_gain": 0.7},
        include_hard_as_train=True,
    )
    bank = cache_curriculum.build_stage1_curriculum_bank(_cfg(), bank_cfg)
    probe = cache_curriculum.stage1_curriculum_bank_probe(bank)
    print(
        "[cache_curriculum trace] active_dims "
        f"allowed_bases={probe['allowed_bases']} "
        f"family_groups={probe['family_groups']} "
        f"mix_classes={probe['mix_classes']} "
        f"roles={probe['roles']}"
    )
    assert probe["allowed_bases"] == ("local_rp",)
    assert set(probe["family_groups"]) == {("local_rp",)}
    assert set(probe["roles"]) == {"train"}


def test_stage1_bank_rejects_invalid_contract_config() -> None:
    try:
        cache_curriculum.FrontRESStage1CurriculumBankConfig(
            frontier_scale=2.0,
            dr_min=2.0,
            dr_max=1.0,
            n_train=1,
            progress=0.0,
            seq_idx=0,
        ).validate()
    except ValueError as exc:
        print(f"[cache_curriculum trace] rejected_config={exc}")
        assert "dr_max" in str(exc)
        return
    raise AssertionError("invalid dr range should be rejected")


if __name__ == "__main__":
    test_stage1_bank_traces_hrl_curriculum_chain()
    test_stage1_bank_respects_active_dims_and_hard_training_override()
    test_stage1_bank_rejects_invalid_contract_config()
    print("PASS: FrontRES Stage 1 curriculum bank derives cache levels from HRL perturbation curriculum.")
