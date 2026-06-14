from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "rsl_rl"
    / "rsl_rl"
    / "runners"
    / "frontres_dr_curriculum.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("frontres_dr_curriculum", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_allowed_bases_respect_active_dims():
    drc = load_module()

    assert drc.allowed_perturbation_bases([0, 1, 5]) == ("planar", "yaw")
    assert drc.allowed_perturbation_bases([3, 4]) == ("local_rp",)
    assert drc.allowed_perturbation_bases([6, 7]) == (
        "planar",
        "yaw",
        "global_z",
        "local_rp",
    )


def test_specialist_mix_plan_uses_local_rp_only():
    drc = load_module()
    cfg = {
        "frontres_specialist_mode": "rp",
        "frontres_perturbation_curriculum_enabled": True,
    }

    plan = drc.sample_perturbation_mix(
        cfg,
        active_dims=[3, 4],
        progress=0.5,
        seq_idx=7,
        n_train=5,
        boundary_stats={"safe": 0.2, "repair": 0.5, "broken": 0.1, "positive_gain": 0.7},
    )

    assert plan.active_modes == ("local_rp",)
    assert plan.complexity == "single"
    assert plan.groups == [("local_rp",)] * 5


def test_per_env_strength_preserves_branch_pairing_and_clean_branch():
    drc = load_module()
    cfg = {
        "frontres_mixed_dr_strength_per_env": True,
        "frontres_mixed_dr_easy_weight": 0.5,
        "frontres_mixed_dr_frontier_weight": 0.4,
        "frontres_mixed_dr_hard_weight": 0.1,
        "frontres_mixed_dr_easy_factor": 0.75,
        "frontres_mixed_dr_frontier_factor": 1.0,
        "frontres_mixed_dr_hard_factor": 1.05,
    }

    plan = drc.sample_per_env_dr_strength(
        cfg,
        frontier_scale=2.0,
        enabled=True,
        seq_idx=17,
        n_train=4,
        n_candidate=4,
        n_base=4,
        num_envs=14,
        dr_min=0.0,
        dr_max=4.0,
    )

    assert plan.scale_vector is not None
    assert plan.mix_class is not None
    assert len(plan.scale_vector) == 14
    assert len(plan.mix_class) == 4
    assert plan.scale_vector[0:4] == plan.scale_vector[4:8]
    assert plan.scale_vector[0:4] == plan.scale_vector[8:12]
    assert plan.scale_vector[12:14] == [0.0, 0.0]
    assert abs(plan.diag["easy"] + plan.diag["frontier"] + plan.diag["hard"] - 1.0) < 1e-6


def test_gmt_frontier_safe_and_broken_updates_bracket():
    drc = load_module()
    cfg = {
        "frontres_gmt_frontier_safe_threshold": 0.85,
        "frontres_gmt_frontier_broken_threshold": 0.65,
        "frontres_gmt_frontier_growth_factor": 1.12,
        "frontres_gmt_frontier_conservative_frac": 0.0,
    }

    state = drc.GMTFrontierState(safe_low=1.25, broken_high=None, probe_scale=1.5)
    safe_update = drc.update_gmt_frontier_state(
        cfg,
        state,
        score=0.95,
        samples=100,
        dr_scale=1.5,
        dr_scale_init=1.25,
        dr_min=0.0,
        dr_max=4.0,
    )
    assert safe_update.state.decision == "safe"
    assert safe_update.state.safe_low == 1.5
    assert safe_update.next_dr_scale > 1.5

    broken_state = drc.GMTFrontierState(
        safe_low=safe_update.state.safe_low,
        broken_high=None,
        probe_scale=safe_update.next_dr_scale,
    )
    broken_update = drc.update_gmt_frontier_state(
        cfg,
        broken_state,
        score=0.2,
        samples=100,
        dr_scale=safe_update.next_dr_scale,
        dr_scale_init=1.25,
        dr_min=0.0,
        dr_max=4.0,
    )
    assert broken_update.state.decision == "broken"
    assert broken_update.state.broken_high is not None
    assert broken_update.state.safe_low <= broken_update.state.broken_high


if __name__ == "__main__":
    test_allowed_bases_respect_active_dims()
    test_specialist_mix_plan_uses_local_rp_only()
    test_per_env_strength_preserves_branch_pairing_and_clean_branch()
    test_gmt_frontier_safe_and_broken_updates_bracket()
