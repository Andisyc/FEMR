#!/usr/bin/env python3
"""Step 8 sentinel for active FEMR HSL acceptance diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3]
DIAG = SOURCE_ROOT / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_diagnostics.py"
LOGGING = SOURCE_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "frontres_runner_logging.py"
CHECKLIST = SOURCE_ROOT.parents[0] / "note" / "FrontRES Modification Checklist.md"


def _load_diag():
    spec = importlib.util.spec_from_file_location("frontres_diagnostics_under_test", DIAG)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_active_hsl_console_uses_acceptance_language() -> None:
    diag = _load_diag()
    cfg = {
        "frontres_training_objective": "hsl_hybrid",
        "frontres_authority_actor_critic_enabled": False,
        "frontres_structured_joint_rl_enabled": False,
        "frontres_structured_joint_rl_weight": 0.0,
    }
    loss = {
        "acceptance_preference_loss": 0.123,
        "lambda_acceptance_preference": 1.0,
        "hsl_acceptance_loss_enabled": 1.0,
        "hsl_acceptance_mask_frac": 0.75,
        "hsl_acceptance_gt_mean": 0.60,
        "hsl_acceptance_prob_mean": 0.55,
        "hsl_acceptance_abs_err": 0.20,
        "frontres_accept_pos_mean": 0.52,
        "frontres_accept_rpy_mean": 0.58,
        "frontres_accept_active_frac": 0.47,
        "frontres_proposal_ratio": 0.91,
        "frontres_axis_leakage": 0.03,
    }
    locs = {
        "frontres_accept_pref_mask_mean": 0.75,
        "frontres_accept_pref_full_mean": 0.60,
        "frontres_accept_pref_noop_mean": 0.30,
        "frontres_accept_pref_ignore_mean": 0.10,
        "frontres_accept_pref_margin_mean": 0.0123,
    }
    text = diag.format_frontres_hsl_acceptance_diagnostics(locs, loss, cfg, pad=28)
    assert "acceptance loss" in text
    assert "accept labels" in text
    assert "accept prob" in text
    assert "proposal ratio" in text
    forbidden = ("authority", "rho", "alpha", "structured", "accept pref")
    assert not any(word in text.lower() for word in forbidden), text


def test_active_hsl_suppresses_old_route_and_preference_formatters() -> None:
    diag = _load_diag()
    cfg = {
        "frontres_training_objective": "hsl_hybrid",
        "frontres_authority_actor_critic_enabled": False,
        "frontres_structured_joint_rl_enabled": False,
        "frontres_structured_joint_rl_weight": 0.0,
    }
    loss = {
        "hsl_acceptance_loss_enabled": 1.0,
        "lambda_acceptance_preference": 1.0,
        "structured_joint_rl_enabled": 0.0,
        "lambda_structured_joint_rl": 0.0,
        "authority_actor_critic_enabled": 0.0,
    }
    locs = {
        "frontres_structured_joint_rho_adv_mean": 0.0,
        "frontres_accept_pref_mask_mean": 0.5,
        "frontres_candidate_floor_margin_mean": 0.1,
    }
    assert diag.format_frontres_route_rho_diagnostics(locs, cfg, pad=28) == ""
    assert diag.format_frontres_preference_diagnostics(locs, loss, cfg, pad=28) == ""


def test_runner_logging_contains_step8_guards() -> None:
    text = LOGGING.read_text()
    assert "format_frontres_hsl_acceptance_diagnostics" in text
    assert "_active_hsl_acceptance_log_mode" in text
    assert "FrontRES/Acceptance/loss" in text
    assert "FrontRES/Acceptance/lambda" in text
    assert 'key.startswith("acceptance_preference_")' in text
    assert "if not _active_hsl_acceptance_log:" in text
    assert "HSL ΔSE proposal + masked acceptance" in text
    assert "accept pref loss" in text
    assert "_active_hsl_acceptance_log" in text


if __name__ == "__main__":
    test_active_hsl_console_uses_acceptance_language()
    test_active_hsl_suppresses_old_route_and_preference_formatters()
    test_runner_logging_contains_step8_guards()
    print("frontres_hsl_acceptance_diagnostics: PASS")
