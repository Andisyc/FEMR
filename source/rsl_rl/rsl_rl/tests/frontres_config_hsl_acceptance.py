#!/usr/bin/env python3
"""TEST ONLY: active FEMR config must default to HSL+HRL acceptance.

The test uses static AST parsing so it does not import IsaacLab/Hydra modules.
It checks config intent only; later steps test policy, storage, runner, and loss.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
G1_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py"
WBT_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py"
RSL_CFG = ROOT / "source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py"


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _class_assignments(path: Path, class_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out: dict[str, Any] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    value = _literal(stmt.value)
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            out[target.id] = value
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    out[stmt.target.id] = _literal(stmt.value) if stmt.value is not None else None
            return out
    raise AssertionError(f"class {class_name} not found in {path}")


def _constructor_kwargs(path: Path, constructor_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == constructor_name:
            matches.append({kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg is not None})
    if not matches:
        raise AssertionError(f"constructor {constructor_name} not found in {path}")
    return matches[-1]


def _assert_eq(name: str, got: Any, expected: Any) -> None:
    if got != expected:
        raise AssertionError(f"{name}: got {got!r}, expected {expected!r}")


def _assert_positive(name: str, got: Any) -> None:
    if not isinstance(got, (int, float)) or got <= 0.0:
        raise AssertionError(f"{name}: expected positive numeric value, got {got!r}")


def run_config_hsl_acceptance_check() -> None:
    policy = _constructor_kwargs(G1_CFG, "RslRlFrontResidualActorCriticCfg")
    alg = _constructor_kwargs(G1_CFG, "RslRlFrontRESUnifiedAlgorithmCfg")
    runner = _class_assignments(G1_CFG, "G1FlatFrontRESUnifiedRunnerCfg")

    _assert_eq("policy.frontres_authority_actor_critic", policy.get("frontres_authority_actor_critic"), False)
    _assert_eq("algorithm.frontres_authority_actor_critic_enabled", alg.get("frontres_authority_actor_critic_enabled"), False)
    _assert_eq("algorithm.frontres_authority_actor_loss_weight", alg.get("frontres_authority_actor_loss_weight"), 0.0)
    _assert_eq("algorithm.frontres_authority_critic_loss_weight", alg.get("frontres_authority_critic_loss_weight"), 0.0)
    _assert_eq("algorithm.frontres_authority_actor_warmup_iterations", alg.get("frontres_authority_actor_warmup_iterations"), 0)
    _assert_eq("algorithm.frontres_authority_actor_ramp_iterations", alg.get("frontres_authority_actor_ramp_iterations"), 0)
    _assert_eq("algorithm.frontres_authority_return_horizon", alg.get("frontres_authority_return_horizon"), 1)

    _assert_eq("algorithm.frontres_training_objective", alg.get("frontres_training_objective"), "hsl_hybrid")
    _assert_positive("algorithm.frontres_acceptance_preference_weight", alg.get("frontres_acceptance_preference_weight"))

    _assert_eq("algorithm.frontres_structured_joint_rl_enabled", alg.get("frontres_structured_joint_rl_enabled"), False)
    _assert_eq("algorithm.frontres_structured_joint_rl_weight", alg.get("frontres_structured_joint_rl_weight"), 0.0)
    _assert_eq("algorithm.frontres_structured_joint_rl_keep_legacy_bce", alg.get("frontres_structured_joint_rl_keep_legacy_bce"), False)
    _assert_eq("algorithm.frontres_structured_joint_show_legacy_rho_diag", alg.get("frontres_structured_joint_show_legacy_rho_diag"), False)

    _assert_eq("algorithm.frontres_state_alpha_weight", alg.get("frontres_state_alpha_weight"), 0.0)
    _assert_eq("runner.frontres_state_alpha_enabled", runner.get("frontres_state_alpha_enabled"), False)
    _assert_eq("runner.frontres_state_alpha_route_enabled", runner.get("frontres_state_alpha_route_enabled"), False)
    _assert_eq("runner.frontres_authority_actor_critic_enabled", runner.get("frontres_authority_actor_critic_enabled"), False)
    _assert_eq("runner.frontres_perturbation_temporal_mode", runner.get("frontres_perturbation_temporal_mode"), "single")

    for cfg_path in (WBT_CFG, RSL_CFG):
        cfg = _class_assignments(cfg_path, "RslRlFrontRESUnifiedAlgorithmCfg")
        _assert_eq(f"{cfg_path.name}.frontres_training_objective", cfg.get("frontres_training_objective"), "hsl_hybrid")
        _assert_eq(f"{cfg_path.name}.frontres_state_alpha_enabled", cfg.get("frontres_state_alpha_enabled"), False)
        _assert_eq(f"{cfg_path.name}.frontres_state_alpha_route_enabled", cfg.get("frontres_state_alpha_route_enabled"), False)
        _assert_eq(f"{cfg_path.name}.frontres_authority_actor_critic_enabled", cfg.get("frontres_authority_actor_critic_enabled"), False)
        _assert_eq(f"{cfg_path.name}.frontres_authority_actor_loss_weight", cfg.get("frontres_authority_actor_loss_weight"), 0.0)
        _assert_eq(f"{cfg_path.name}.frontres_authority_critic_loss_weight", cfg.get("frontres_authority_critic_loss_weight"), 0.0)

    print("FrontRES HSL+acceptance config sentinel: PASS")


if __name__ == "__main__":
    run_config_hsl_acceptance_check()
