#!/usr/bin/env python3
"""TEST ONLY: active FEMR policy surface is HSL proposal + acceptance.

This avoids loading GMT/IsaacLab by testing the relevant FrontRESActorCritic
methods on a minimal object.  It proves the Stage-2 acceptance head sees
full_obs + detached Stage-1 proposal, and that acceptance loss does not update
Stage 1 proposal parameters.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
RSL_SOURCE = ROOT / "source/rsl_rl"
if str(RSL_SOURCE) not in sys.path:
    sys.path.insert(0, str(RSL_SOURCE))
MODULE_PATH = ROOT / "source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py"
G1_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py"


def _load_policy_class():
    spec = importlib.util.spec_from_file_location("front_residual_actor_critic_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FrontRESActorCritic


class ProposalOnlyActor(nn.Module):
    def __init__(self, raw: torch.Tensor):
        super().__init__()
        self.raw = nn.Parameter(raw.clone())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.raw.expand(obs.shape[0], -1)


class RecordingAcceptanceActor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.ones_(self.linear.weight)
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        return self.linear(x)


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _constructor_kwargs(path: Path, constructor_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == constructor_name:
            matches.append({kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg is not None})
    if not matches:
        raise AssertionError(f"constructor {constructor_name} not found")
    return matches[-1]


def _build_minimal_policy():
    FrontRESActorCritic = _load_policy_class()
    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    nn.Module.__init__(policy)
    policy.num_task_corrections = 6
    policy.task_conf_dim = 6
    policy.max_delta_pos = 0.3
    policy.max_delta_rpy = 0.4
    policy.frontres_split_acceptance_head = True
    policy.frontres_authority_actor_critic = False
    policy.frontres_state_router_enabled = False
    raw = torch.tensor([[0.20, -0.10, 0.05, 0.30, -0.20, 0.10]], dtype=torch.float32)
    policy.residual_actor = ProposalOnlyActor(raw)
    policy.acceptance_actor = RecordingAcceptanceActor(input_dim=10 + 6, output_dim=6)
    policy.authority_actor = None
    policy.authority_critic = None
    policy.state_router_head = None
    policy._cached_full_policy_obs = torch.arange(20, dtype=torch.float32).view(2, 10) / 10.0
    return policy


def test_config_policy_surface() -> None:
    policy_cfg = _constructor_kwargs(G1_CFG, "RslRlFrontResidualActorCriticCfg")
    if policy_cfg.get("frontres_split_acceptance_head") is not True:
        raise AssertionError("active G1 policy must enable split acceptance head")
    if policy_cfg.get("frontres_authority_actor_critic") is not False:
        raise AssertionError("active G1 policy must not enable authority actor-critic")
    if policy_cfg.get("frontres_state_router_enabled") is not False:
        raise AssertionError("active G1 policy must not enable state-router alpha head")


def test_split_acceptance_gradient_boundary() -> None:
    policy = _build_minimal_policy()
    policy_obs = torch.zeros(2, 10)
    raw = policy._frontres_raw_task_output(policy_obs)
    if tuple(raw.shape) != (2, 12):
        raise AssertionError(f"raw output shape mismatch: {tuple(raw.shape)}")

    acceptance_actor = policy.acceptance_actor
    assert isinstance(acceptance_actor, RecordingAcceptanceActor)
    if acceptance_actor.last_input is None:
        raise AssertionError("acceptance actor was not called")
    full_obs = acceptance_actor.last_input[:, :10]
    proposal_seen = acceptance_actor.last_input[:, 10:]
    expected_full_obs = policy._cached_full_policy_obs
    expected_proposal = policy._frontres_bounded_proposal(policy.residual_actor(policy_obs)).detach()
    if not torch.allclose(full_obs, expected_full_obs):
        raise AssertionError("acceptance head did not receive full current-state observation")
    if not torch.allclose(proposal_seen, expected_proposal):
        raise AssertionError("acceptance head did not receive detached bounded proposal")
    if proposal_seen.requires_grad:
        raise AssertionError("acceptance head input keeps proposal gradient alive")

    acceptance_loss = raw[:, 6:].sum()
    acceptance_loss.backward()
    proposal_grad = policy.residual_actor.raw.grad
    if proposal_grad is not None and float(proposal_grad.abs().sum().item()) != 0.0:
        raise AssertionError("acceptance loss leaked gradient into Stage-1 proposal actor")
    accept_grad = policy.acceptance_actor.linear.weight.grad
    if accept_grad is None or float(accept_grad.abs().sum().item()) <= 0.0:
        raise AssertionError("acceptance loss did not train Stage-2 acceptance head")

    policy = _build_minimal_policy()
    raw = policy._frontres_raw_task_output(policy_obs)
    proposal_loss = raw[:, :6].sum()
    proposal_loss.backward()
    proposal_grad = policy.residual_actor.raw.grad
    if proposal_grad is None or float(proposal_grad.abs().sum().item()) <= 0.0:
        raise AssertionError("proposal loss did not train Stage-1 proposal actor")


def main() -> None:
    test_config_policy_surface()
    test_split_acceptance_gradient_boundary()
    print("FrontRES HSL+acceptance policy surface: PASS")


if __name__ == "__main__":
    main()
