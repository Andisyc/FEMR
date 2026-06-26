#!/usr/bin/env python3
"""Step 7 sentinel: active FEMR HSL acceptance must not enter old live branches.

This is a lightweight source-level test.  It protects the runner live path from
silently reactivating authority actor-critic, structured-rho, or alpha-rho
storage writes in the active `hsl_hybrid` FEMR route.
"""

from __future__ import annotations

from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3]
RUNNER = SOURCE_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "on_policy_runner.py"
PAYLOAD = SOURCE_ROOT / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_transition_payload.py"
ROLLOUT_STEP = SOURCE_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "frontres_rollout_step.py"
POST_STEP = SOURCE_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "frontres_post_step_connector.py"


def _read(path: Path) -> str:
    return path.read_text()


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


def test_runner_old_alpha_rho_branch_is_structured_only() -> None:
    text = _read(RUNNER)
    block = _between(text, "rho_advantage = None", "# 计算∆_reward并累积日志诊断")
    assert "_authority_actor_critic_active" in block
    assert "_structured_rho_active" in block
    assert "and _structured_rho_active" in block
    assert "and not _authority_actor_critic_active" in block
    assert "write_rho_update_weight(" in block
    assert "write_alpha_groundtruth(" in block
    assert "write_rho_advantage(" in block


def test_active_hsl_payload_uses_algorithm_enable_gates() -> None:
    text = _read(PAYLOAD)
    active_gate = _between(text, "def _frontres_authority_actor_critic_active", "def _write_active_hsl_acceptance_payload")
    assert "_authority_actor_critic_enabled" in active_gate
    assert "not _frontres_authority_actor_critic_active(runner)" in active_gate
    assert "not bool(structured_joint_requested)" in active_gate


def test_runner_syncs_algorithm_config_before_rollout_gates() -> None:
    text = _read(RUNNER)
    init_block = _between(text, "self.alg: PPO | Distillation | MOSAIC | FrontRESUnified", "self._frontres_alpha_rho_bridge")
    assert "self._sync_frontres_algorithm_state_to_runner_cfg()" in init_block
    assert "self._validate_frontres_active_hsl_acceptance_path()" in init_block

    sync_block = _between(text, "def _sync_frontres_algorithm_state_to_runner_cfg", "def _validate_frontres_active_hsl_acceptance_path")
    assert '"frontres_training_objective"' in sync_block
    assert '"frontres_structured_joint_rl_enabled"' in sync_block
    assert '"frontres_structured_joint_rl_weight"' in sync_block
    assert "self.cfg[key] = getattr(self.alg, key)" in sync_block
    assert "self.alg_cfg[key] = getattr(self.alg, key)" in sync_block

    validate_block = _between(text, "def _validate_frontres_active_hsl_acceptance_path", "def evaluate_frontres_dr_sweep")
    assert "_active_hsl_acceptance_loss_enabled" in validate_block
    assert "FrontRES active HSL acceptance path verified" in validate_block
    assert "Invalid FEMR Stage 2 config" in validate_block


def test_active_hsl_payload_does_not_allocate_or_write_structured_rho_prior_storage() -> None:
    text = _read(PAYLOAD)
    init_block = _between(text, "rho_prior_authority = None", "pref_inertial_penalty_rho_mean")
    assert "if not active_hsl_acceptance_payload:" in init_block
    assert "rho_prior_authority = torch.zeros" in init_block
    assert "rho_prior_target = torch.zeros" in init_block

    tail = _between(text, "runner.alg.transition.acceptance_target = accept_target", "return accept_payload")
    assert "runner.alg.transition.acceptance_gt" in tail
    assert "runner.alg.transition.acceptance_margin" in tail
    assert "if not active_hsl_acceptance_payload:" in tail
    guarded = tail[tail.index("if not active_hsl_acceptance_payload:") :]
    assert "runner.alg.transition.rho_prior_authority = rho_prior_authority" in guarded
    assert "runner.alg.transition.rho_prior_target = rho_prior_target" in guarded


def test_authority_live_path_has_early_disable_guards() -> None:
    rollout_text = _read(ROLLOUT_STEP)
    assert "def _frontres_authority_enabled" in rollout_text
    assert "if not _frontres_authority_enabled(" in rollout_text
    assert "return actions" in rollout_text

    post_text = _read(POST_STEP)
    write_return = _between(post_text, "def _write_frontres_authority_return", "def finalize_frontres_authority_k_step_returns")
    finalize = _between(post_text, "def finalize_frontres_authority_k_step_returns", "def apply_frontres_post_step_reward_connector")
    assert "if not _frontres_authority_enabled(runner):" in write_return
    assert "return" in write_return
    assert "if not _frontres_authority_enabled(runner):" in finalize
    assert "return" in finalize


if __name__ == "__main__":
    test_runner_old_alpha_rho_branch_is_structured_only()
    test_active_hsl_payload_uses_algorithm_enable_gates()
    test_runner_syncs_algorithm_config_before_rollout_gates()
    test_active_hsl_payload_does_not_allocate_or_write_structured_rho_prior_storage()
    test_authority_live_path_has_early_disable_guards()
    print("frontres_runner_hsl_acceptance_path: PASS")
