"""TEST ONLY: Stage-1 FrontRES reward diagnostics without acceptance payload."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_DIAGNOSTICS_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_reward_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location("frontres_reward_diagnostics_under_test", _DIAGNOSTICS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load reward diagnostics module from {_DIAGNOSTICS_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
accumulate_frontres_reward_diagnostics = _MODULE.accumulate_frontres_reward_diagnostics
initialize_frontres_reward_diagnostic_sums = _MODULE.initialize_frontres_reward_diagnostic_sums


def test_stage1_reward_diagnostics_acceptance_payload_optional() -> None:
    n = 4
    zeros = torch.zeros(n)
    ones = torch.ones(n)
    locs = {
        "_n_exec": n,
        "N_train": n,
        "_r_z": zeros,
        "_r_xy": zeros,
        "_r_rp": zeros,
        "_r_ya": zeros,
        "_r_rescue_log": 0.0,
        "_r_exec": zeros,
        "_r_step": zeros,
        "_intervention_cost": zeros,
        "_clean_bound_cost": zeros,
        "_side_cost": zeros,
        "_over_cost": zeros,
        "_under_repair_penalty": zeros,
        "_r_frontres_log": 0.0,
        "_r_clean_log": 0.0,
        "_exec_candidate": zeros,
        "_r_oracle_log": 0.0,
        "_exec_planar_log": 0.0,
        "_exec_vertical_log": 0.0,
        "_exec_task_log": 0.0,
        "_damage_gap": zeros,
        "_oracle_clean_gap": zeros,
        "_oracle_trust": zeros,
        "_repair_gain": zeros,
        "_candidate_gain": zeros,
        "_projection_gain": zeros,
        "_under_write": zeros,
        "_repair_ratio": zeros,
        "_exec_weight": ones,
        "_cost_exec": zeros,
        "r_delta": zeros,
        "_reward_progress": 0.0,
        "_constraint_progress": 0.0,
        "_effective_gain_bonus": zeros,
        "_safe_gate": zeros,
        "_repair_gate": ones,
        "_broken_gate": zeros,
        "_actor_gate": ones,
        "_safe_gap": 0.1,
        "_broken_gap": 0.9,
        "_window_mu": ones,
        "_harm_mag": zeros,
        "_effective_gain_bonus_exec": zeros,
        "_harm_penalty_exec": zeros,
        "_cost_gate": zeros,
        "_e_raw": ones,
        "_e_fr": zeros,
        "_rot_raw_to_fr": torch.zeros(n, 3),
        "_safe_frac": 0.0,
        "_repair_frac": 1.0,
        "_broken_frac": 0.0,
        "_candidate_floor_margin": zeros,
        "_candidate_floor_pass_frac": 0.0,
        "_oracle_ub_gain": zeros,
        "_oracle_ub_pass": zeros,
        "_oracle_ub_projected_win": zeros,
        "_oracle_ub_candidate_win": zeros,
        "_oracle_ub_feasible_win": zeros,
        "_oracle_ub_noisy_win": zeros,
        "_exec_gate": ones,
        "_dr_z_abs_log": 0.0,
        "_dr_xy_abs_log": 0.0,
        "_dr_rp_abs_log": 0.0,
        "_dr_yaw_abs_log": 0.0,
        "_corr_z_abs_log": 0.0,
        "_corr_xy_abs_log": 0.0,
        "_corr_rp_abs_log": 0.0,
        "_corr_yaw_abs_log": 0.0,
        "_is_task_space_mode": False,
    }
    runner = SimpleNamespace(cfg={}, alg=SimpleNamespace(policy=SimpleNamespace()))
    sums = initialize_frontres_reward_diagnostic_sums()

    accumulate_frontres_reward_diagnostics(runner, sums, locs)

    if sums["reward_diag_steps"] != 1:
        raise AssertionError("reward diagnostic step was not accumulated.")
    if sums["accept_pref_mask"] != 0.0:
        raise AssertionError("missing Stage-1 acceptance payload should accumulate zero acceptance mask.")

    print("=== FrontRES Stage-1 Reward Diagnostics TEST ONLY ===")
    print("checks=acceptance payload optional in proposal-only Stage 1")
    print("result: PASS")


if __name__ == "__main__":
    test_stage1_reward_diagnostics_acceptance_payload_optional()
