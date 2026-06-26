#!/usr/bin/env python3
"""Step 11 sentinel: old authority/rho/alpha branches do not run in active FEMR."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified

ROOT = Path(__file__).resolve().parents[4]
LOGGING_PATH = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_runner_logging.py"
TRANSITION_PAYLOAD = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_transition_payload.py"
RUNNER = ROOT / "source/rsl_rl/rsl_rl/runners/on_policy_runner.py"
ROLLOUT_STEP = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_rollout_step.py"
DIAG = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_diagnostics.py"
TRAIN = ROOT / "scripts/rsl_rl/train.py"


class FakePolicy:
    task_conf_dim = 6
    num_task_corrections = 6


class FakeAlgorithm:
    def __init__(self):
        self.device = "cpu"
        self.policy = FakePolicy()
        self.frontres_training_objective = "hsl_hybrid"
        self.frontres_acceptance_preference_weight = 1.0
        self.frontres_structured_joint_rl_enabled = False
        self.frontres_structured_joint_rl_weight = 0.0
        self.frontres_structured_joint_rl_keep_legacy_bce = False
        self.frontres_authority_actor_critic_enabled = False
        self.frontres_authority_actor_loss_weight = 0.0
        self.frontres_authority_critic_loss_weight = 0.0
        self.ppo_actor_weight = 1.0

    def _ppo_acceptance_only_mode(self) -> bool:
        return FrontRESUnified._ppo_acceptance_only_mode(self)

    def _structured_joint_rl_enabled(self) -> bool:
        return FrontRESUnified._structured_joint_rl_enabled(self)

    def _authority_actor_critic_enabled(self) -> bool:
        return FrontRESUnified._authority_actor_critic_enabled(self)

    def _active_hsl_acceptance_loss_enabled(self) -> bool:
        return FrontRESUnified._active_hsl_acceptance_loss_enabled(self)



def _active_locs() -> dict:
    return {
        "loss_dict": {
            "hsl_acceptance_loss_enabled": 1.0,
            "lambda_acceptance_preference": 1.0,
            "acceptance_preference_loss": 0.25,
            "structured_joint_rl_enabled": 0.0,
            "lambda_structured_joint_rl": 0.0,
            "authority_actor_critic_enabled": 0.0,
            "lambda_authority_actor": 0.0,
            "lambda_authority_critic": 0.0,
            "state_alpha_loss": 0.0,
            "lambda_state_alpha": 0.0,
        }
    }


def test_algorithm_guards() -> None:
    alg = FakeAlgorithm()
    if not alg._active_hsl_acceptance_loss_enabled():
        raise AssertionError("active HSL acceptance should be enabled in the main FEMR config")
    alg.frontres_structured_joint_rl_enabled = True
    alg.frontres_structured_joint_rl_weight = 1.0
    if alg._active_hsl_acceptance_loss_enabled():
        raise AssertionError("structured-rho active branch must disable active HSL acceptance")
    alg.frontres_structured_joint_rl_enabled = False
    alg.frontres_structured_joint_rl_weight = 0.0
    alg.frontres_authority_actor_critic_enabled = True
    alg.frontres_authority_actor_loss_weight = 1.0
    if alg._active_hsl_acceptance_loss_enabled():
        raise AssertionError("authority actor-critic branch must disable active HSL acceptance")


def test_runner_and_payload_guards_are_present() -> None:
    runner_text = RUNNER.read_text()
    payload_text = TRANSITION_PAYLOAD.read_text()
    rollout_text = ROLLOUT_STEP.read_text()
    train_text = TRAIN.read_text()
    required = [
        "and _structured_rho_active",
        "and not _authority_actor_critic_active",
        "_is_active_hsl_acceptance_payload_mode",
        "if not active_hsl_acceptance_payload:",
        "transition.acceptance_gt = acceptance_gt",
        "transition.acceptance_target = accept_target",
        "transition.rho_prior_authority = rho_prior_authority",
        "def _frontres_authority_enabled",
        "if not _frontres_authority_enabled",
        'choices=("stage1_hsl", "stage2_acceptance")',
        'elif stage == "stage2_acceptance":',
    ]
    combined = "\n".join([runner_text, payload_text, rollout_text, train_text])
    for needle in required:
        if needle not in combined:
            raise AssertionError(f"missing legacy guard evidence: {needle}")
    if "stage2_authority" in train_text:
        raise AssertionError("active train.py should not expose stage2_authority")


def test_active_logging_suppresses_legacy_names() -> None:
    text = LOGGING_PATH.read_text()
    required = [
        "def _active_hsl_acceptance_log_mode",
        "_active_hsl_acceptance_loss_enabled",
        "hsl_acceptance_path_enabled",
        "key.startswith(_legacy_active_log_prefixes)",
        "FrontRES/Acceptance/loss",
        "FrontRES/Acceptance/lambda",
    ]
    for needle in required:
        if needle not in text:
            raise AssertionError(f"runner logging guard missing: {needle}")
    legacy_guard = text.split("_legacy_pref_disabled =", 1)[1].split("if not _legacy_pref_disabled", 1)[0]
    if "_active_hsl_acceptance_log" not in legacy_guard:
        raise AssertionError("console legacy accept-pref line is not suppressed in active HSL mode")


def test_diagnostics_suppress_legacy_sections() -> None:
    diag_text = DIAG.read_text()
    for needle in (
        "def _active_hsl_acceptance_enabled",
        'return ""', 
        "format_frontres_hsl_acceptance_diagnostics",
    ):
        if needle not in diag_text:
            raise AssertionError(f"missing diagnostics guard: {needle}")


def main() -> None:
    test_algorithm_guards()
    test_runner_and_payload_guards_are_present()
    test_active_logging_suppresses_legacy_names()
    test_diagnostics_suppress_legacy_sections()
    print("PASS: active FEMR path hard-gates old authority/rho/alpha branches.")


if __name__ == "__main__":
    main()
