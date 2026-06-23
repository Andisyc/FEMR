# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: FrontRES authority actor-critic network surface contract."""

from __future__ import annotations

import sys
import tempfile
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic


def _load_checkpoint_helpers():
    helper_path = Path(__file__).resolve().parents[1] / "runners" / "frontres_checkpointing.py"
    spec = importlib.util.spec_from_file_location("frontres_checkpointing_test_only", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load checkpoint helper from {helper_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.save_runner, module.load_runner


save_runner, load_runner = _load_checkpoint_helpers()


class ProposalActor(nn.Module):
    """Small Stage-1 proposal actor returning raw [Delta pos, Delta rpy, coeff]."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, 12, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            for row in range(6):
                self.linear.weight[row, row % obs_dim] = 0.2 + 0.05 * row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RecordingAuthorityActor(nn.Module):
    """Authority actor that records state/proposal input."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.last_input: torch.Tensor | None = None
        with torch.no_grad():
            self.linear.weight.fill_(0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        return self.linear(x)


class RecordingAuthorityCritic(nn.Module):
    """Authority critic that records state/proposal/rho input."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.last_input: torch.Tensor | None = None
        with torch.no_grad():
            self.linear.weight.fill_(0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        return self.linear(x)


def _build_minimal_policy(*, full_obs_dim: int, proposal_dim: int = 6) -> FrontRESActorCritic:
    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    nn.Module.__init__(policy)
    policy.num_actor_obs = full_obs_dim
    policy.gmt_actor_input_dim = full_obs_dim
    policy.num_frontres_obs = 0
    policy.num_task_corrections = proposal_dim
    policy.task_conf_dim = proposal_dim
    policy.max_delta_pos = 0.30
    policy.max_delta_rpy = 0.40
    policy.frontres_split_acceptance_head = False
    policy.acceptance_actor = None
    policy.residual_actor = ProposalActor(full_obs_dim)
    policy.critic = nn.Linear(full_obs_dim, 1, bias=False)
    policy.authority_actor = RecordingAuthorityActor(full_obs_dim + proposal_dim, proposal_dim)
    policy.authority_critic = RecordingAuthorityCritic(full_obs_dim + 2 * proposal_dim)
    return policy


def _expected_proposal(policy: FrontRESActorCritic, obs: torch.Tensor) -> torch.Tensor:
    raw = policy.residual_actor(obs)
    return torch.cat(
        [
            torch.tanh(raw[:, :3]) * policy.max_delta_pos,
            torch.tanh(raw[:, 3:6]) * policy.max_delta_rpy,
        ],
        dim=-1,
    )


def test_authority_actor_sees_full_obs_and_detached_proposal() -> None:
    torch.manual_seed(11)
    batch = 4
    full_obs_dim = 10
    policy = _build_minimal_policy(full_obs_dim=full_obs_dim)
    obs = torch.randn(batch, full_obs_dim)
    active_task_dims = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])

    rho = FrontRESActorCritic.get_authority_rho(policy, obs, active_task_dims=active_task_dims)
    authority_actor = policy.authority_actor
    assert isinstance(authority_actor, RecordingAuthorityActor)
    assert authority_actor.last_input is not None

    if rho.shape != (batch, 6):
        raise AssertionError(f"rho shape mismatch: {tuple(rho.shape)}")
    if float(rho.min().item()) < 0.0 or float(rho.max().item()) > 1.0:
        raise AssertionError("authority rho must stay in [0, 1].")
    torch.testing.assert_close(rho[:, 1], torch.zeros(batch))
    torch.testing.assert_close(rho[:, 5], torch.zeros(batch))

    received_full_obs = authority_actor.last_input[:, :full_obs_dim]
    received_proposal = authority_actor.last_input[:, full_obs_dim:]
    torch.testing.assert_close(received_full_obs, obs)
    torch.testing.assert_close(received_proposal, _expected_proposal(policy, obs))
    if received_proposal.requires_grad:
        raise AssertionError("Authority actor proposal feature must be detached from Stage-1 proposal graph.")

    policy.zero_grad(set_to_none=True)
    rho.sum().backward()
    proposal_grad = policy.residual_actor.linear.weight.grad
    authority_grad = policy.authority_actor.linear.weight.grad
    if proposal_grad is not None and float(proposal_grad.abs().max().item()) > 0.0:
        raise AssertionError("Authority actor loss leaked gradient into Stage-1 proposal actor.")
    if authority_grad is None or float(authority_grad.abs().sum().item()) <= 0.0:
        raise AssertionError("Authority actor did not receive gradients.")


def test_authority_critic_sees_state_proposal_and_rho() -> None:
    torch.manual_seed(13)
    batch = 3
    full_obs_dim = 9
    policy = _build_minimal_policy(full_obs_dim=full_obs_dim)
    obs = torch.randn(batch, full_obs_dim)
    proposal = _expected_proposal(policy, obs)
    rho = FrontRESActorCritic.get_authority_rho(policy, obs, proposal_delta_se=proposal)

    q = FrontRESActorCritic.evaluate_authority_q(policy, obs, proposal, rho)
    authority_critic = policy.authority_critic
    assert isinstance(authority_critic, RecordingAuthorityCritic)
    assert authority_critic.last_input is not None
    if q.shape != (batch, 1):
        raise AssertionError(f"authority Q shape mismatch: {tuple(q.shape)}")

    critic_input = authority_critic.last_input
    torch.testing.assert_close(critic_input[:, :full_obs_dim], obs)
    torch.testing.assert_close(critic_input[:, full_obs_dim:full_obs_dim + 6], proposal.detach())
    torch.testing.assert_close(critic_input[:, full_obs_dim + 6:], rho)

    policy.zero_grad(set_to_none=True)
    q.sum().backward()
    proposal_grad = policy.residual_actor.linear.weight.grad
    actor_grad = policy.authority_actor.linear.weight.grad
    critic_grad = policy.authority_critic.linear.weight.grad
    if proposal_grad is not None and float(proposal_grad.abs().max().item()) > 0.0:
        raise AssertionError("Authority critic loss leaked gradient into Stage-1 proposal actor.")
    if actor_grad is None or float(actor_grad.abs().sum().item()) <= 0.0:
        raise AssertionError("Authority critic did not propagate actor-gradient through rho.")
    if critic_grad is None or float(critic_grad.abs().sum().item()) <= 0.0:
        raise AssertionError("Authority critic did not receive gradients.")


def test_authority_inference_uses_authority_actor_not_legacy_coeff_head() -> None:
    torch.manual_seed(15)
    batch = 2
    full_obs_dim = 8
    policy = _build_minimal_policy(full_obs_dim=full_obs_dim)
    obs = torch.ones(batch, full_obs_dim)

    with torch.no_grad():
        # Make the legacy residual coefficient head strongly disagree with the
        # authority actor.  In authority mode, inference must use authority_actor.
        policy.residual_actor.linear.weight[6:12].fill_(-10.0)
        policy.authority_actor.linear.weight.zero_()
        policy.authority_actor.linear.weight[:, 0].fill_(1.0)

    correction = FrontRESActorCritic.get_task_correction_inference(policy, obs)
    expected_proposal = _expected_proposal(policy, obs)
    expected_rho = FrontRESActorCritic.get_authority_rho(policy, obs, proposal_delta_se=expected_proposal)
    legacy_raw = policy.residual_actor(obs)
    legacy_rho = torch.sigmoid(legacy_raw[:, 6:12])

    torch.testing.assert_close(correction[:, :6], expected_proposal)
    torch.testing.assert_close(correction[:, 6:12], expected_rho)
    if torch.allclose(correction[:, 6:12], legacy_rho):
        raise AssertionError("authority inference still appears to use the legacy residual coefficient head.")


def test_authority_checkpoint_round_trip() -> None:
    torch.manual_seed(17)
    policy = _build_minimal_policy(full_obs_dim=8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=optimizer, rnd=None),
        current_learning_iteration=42,
        empirical_normalization=False,
        logger_type="none",
        disable_logs=True,
        training_type="frontres",
        cfg={"is_full_resume": True},
        alg_cfg={},
        policy_cfg={},
        device=torch.device("cpu"),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = str(Path(tmp_dir) / "frontres_authority_round_trip.pt")
        save_runner(runner, checkpoint_path, infos={"test": "authority"})
        saved = torch.load(checkpoint_path, weights_only=False)
        model_state = saved["model_state_dict"]
        for key in ("authority_actor", "authority_critic"):
            if key not in model_state:
                raise AssertionError(f"{key} missing from FrontRES checkpoint.")

        with torch.no_grad():
            policy.authority_actor.linear.weight.zero_()
            policy.authority_critic.linear.weight.zero_()

        infos = load_runner(runner, checkpoint_path, load_optimizer=True, load_critic=True)
        if infos != {"test": "authority"}:
            raise AssertionError("Checkpoint load did not return saved infos.")
        torch.testing.assert_close(
            policy.authority_actor.linear.weight,
            model_state["authority_actor"]["linear.weight"],
        )
        torch.testing.assert_close(
            policy.authority_critic.linear.weight,
            model_state["authority_critic"]["linear.weight"],
        )


def main() -> None:
    test_authority_actor_sees_full_obs_and_detached_proposal()
    test_authority_critic_sees_state_proposal_and_rho()
    test_authority_inference_uses_authority_actor_not_legacy_coeff_head()
    test_authority_checkpoint_round_trip()
    print("=== FrontRES Authority Network TEST ONLY ===")
    print("checks=state+proposal input, bounded rho, Q(state,proposal,rho), inference authority rho, gradient boundary, checkpoint round-trip")
    print("result: PASS")


if __name__ == "__main__":
    main()
