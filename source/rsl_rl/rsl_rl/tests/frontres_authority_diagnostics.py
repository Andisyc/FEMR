"""TEST ONLY: FrontRES authority actor-critic console diagnostics.

This test exercises the formatting-only diagnostics used by the live runner log.
It prevents stale structured-rho diagnostics or local variable leaks from
breaking short-run inspection.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_DIAGNOSTICS_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location("frontres_diagnostics_under_test", _DIAGNOSTICS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load diagnostics module from {_DIAGNOSTICS_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
format_frontres_optimization_diagnostics = _MODULE.format_frontres_optimization_diagnostics
format_frontres_route_rho_diagnostics = _MODULE.format_frontres_route_rho_diagnostics


def test_authority_diagnostics_format() -> None:
    loss_dict = {
        "authority_actor_critic_enabled": 1.0,
        "authority_loss": 0.42,
        "lambda_authority_actor": 1.0,
        "lambda_authority_actor_effective": 0.5,
        "lambda_authority_critic": 1.0,
        "authority_actor_loss": -0.31,
        "authority_critic_loss": 0.73,
        "authority_actor_phase_weight": 0.5,
        "authority_actor_warmup_active": 0.0,
        "authority_actor_ramp_active": 1.0,
        "authority_active_frac": 0.64,
        "authority_return_mean": 0.12,
        "authority_q_behavior_mean": 0.10,
        "authority_q_actor_mean": 0.18,
        "authority_rho_mean": 0.41,
        "authority_rho_std": 0.09,
        "authority_rho_min": 0.02,
        "authority_rho_max": 0.94,
        "authority_rho_near_zero_frac": 0.11,
        "authority_rho_near_one_frac": 0.07,
        "authority_rho_dx_mean": 0.20,
        "authority_rho_dy_mean": 0.30,
        "authority_rho_dz_mean": 0.00,
        "authority_rho_roll_mean": 0.52,
        "authority_rho_pitch_mean": 0.55,
        "authority_rho_yaw_mean": 0.65,
        "authority_return_low_rho_mean": -0.10,
        "authority_return_mid_rho_mean": 0.12,
        "authority_return_high_rho_mean": 0.25,
        "authority_q_actor_low_rho_mean": -0.08,
        "authority_q_actor_mid_rho_mean": 0.10,
        "authority_q_actor_high_rho_mean": 0.28,
        "authority_proposal_abs_low_rho_mean": 0.01,
        "authority_proposal_abs_mid_rho_mean": 0.04,
        "authority_proposal_abs_high_rho_mean": 0.08,
        "authority_k_horizon": 1.0,
        "ppo_actor_weight": 0.0,
        "raw_ppo_actor_weight": 1.0,
        "structured_joint_rl_loss": 9.0,
        "lambda_structured_joint_rl": 0.0,
    }

    text = format_frontres_optimization_diagnostics(loss_dict, pad=32)
    required = (
        "authority AC loss:",
        "authority takeover:",
        "actor_phase=0.500",
        "authority return/Q:",
        "authority rho",
        "authority rho dims:",
        "authority buckets L/M/H:",
        "authority temporal:",
        "K=1",
        "generic PPO=0.000",
    )
    for marker in required:
        if marker not in text:
            raise AssertionError(f"missing authority diagnostic marker: {marker}\n{text}")
    if "joint rl loss:" in text or "rho region loss:" in text:
        raise AssertionError(f"old structured-rho diagnostics should be hidden when lambda is zero:\n{text}")

    disabled_text = format_frontres_optimization_diagnostics(
        {
            "authority_actor_critic_enabled": 0.0,
            "authority_loss": 0.0,
            "lambda_authority_actor": 1.0,
            "lambda_authority_critic": 1.0,
        },
        pad=32,
    )
    if "authority AC loss:" in disabled_text:
        raise AssertionError("authority diagnostics printed while authority actor-critic was disabled.")

    route_text = format_frontres_route_rho_diagnostics(
        {
            "frontres_structured_joint_rho_adv_mean": 1.0,
            "frontres_candidate_floor_margin_mean": 1.0,
            "frontres_rho_target_planar_mean": 1.0,
        },
        {
            "frontres_authority_actor_critic_enabled": True,
            "frontres_structured_joint_rl_enabled": False,
            "frontres_structured_joint_rl_weight": 0.0,
        },
        pad=32,
    )
    if route_text:
        raise AssertionError(f"legacy route-rho diagnostics should be hidden in authority mode:\n{route_text}")

    print("=== FrontRES Authority Diagnostics TEST ONLY ===")
    print(text.rstrip())
    print("checks=authority console sentinel, old structured-rho hidden when inactive/authority-active, no formatter scope leak")
    print("result: PASS")


if __name__ == "__main__":
    test_authority_diagnostics_format()
