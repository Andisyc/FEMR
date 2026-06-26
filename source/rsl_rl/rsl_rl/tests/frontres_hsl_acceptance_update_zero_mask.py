#!/usr/bin/env python3
"""Regression: active Stage 2 update tolerates mini-batches with no acceptance labels."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified
from rsl_rl.storage.rollout_storage import RolloutStorage


class TinySplitPolicy(nn.Module):
    is_recurrent = False
    num_task_corrections = 6
    task_conf_dim = 6

    def __init__(self, obs_dim: int, critic_obs_dim: int):
        super().__init__()
        self.residual_actor = nn.Sequential(nn.Linear(obs_dim, 16), nn.Tanh(), nn.Linear(16, 6))
        self.acceptance_actor = nn.Sequential(nn.Linear(obs_dim + 6, 16), nn.Tanh(), nn.Linear(16, 6))
        self.critic = nn.Sequential(nn.Linear(critic_obs_dim, 16), nn.Tanh(), nn.Linear(16, 1))
        self.std = nn.Parameter(torch.full((12,), 0.05))
        self.max_delta_pos = 0.2
        self.max_delta_rpy = 0.2
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        self.distribution: Normal | None = None
        self.entropy: torch.Tensor | None = None

    def update_distribution(self, observations: torch.Tensor) -> None:
        proposal = self.residual_actor(observations)
        bounded = torch.cat(
            [
                torch.tanh(proposal[:, :3]) * self.max_delta_pos,
                torch.tanh(proposal[:, 3:6]) * self.max_delta_rpy,
            ],
            dim=-1,
        )
        acceptance = self.acceptance_actor(torch.cat([observations, bounded.detach()], dim=-1))
        self.action_mean = torch.cat([proposal, acceptance], dim=-1)
        self.action_std = self.std.clamp(min=0.01).expand_as(self.action_mean)
        self.distribution = Normal(self.action_mean, self.action_std)
        self.entropy = self.distribution.entropy().sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def reset(self, _dones=None) -> None:
        return None

    def get_actions_log_prob_selected(self, actions: torch.Tensor, dims: list[int]) -> torch.Tensor:
        assert self.distribution is not None
        dim_tensor = torch.as_tensor(dims, device=actions.device, dtype=torch.long)
        return self.distribution.log_prob(actions)[:, dim_tensor].sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)


def _fill_zero_acceptance_storage(
    storage: RolloutStorage,
    policy: TinySplitPolicy,
    *,
    obs_dim: int,
    critic_obs_dim: int,
) -> None:
    n = storage.num_envs
    obs = torch.randn(n, obs_dim)
    critic_obs = torch.randn(n, critic_obs_dim)
    with torch.no_grad():
        policy.update_distribution(obs)
        raw = policy.action_mean
        actions = raw.clone()
        values = policy.evaluate(critic_obs)
        log_prob = policy.get_actions_log_prob(actions)

    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.privileged_observations = critic_obs
    transition.actions = actions
    transition.rewards = torch.linspace(0.0, 1.0, n)
    transition.dones = torch.zeros(n)
    transition.values = values
    transition.actions_log_prob = log_prob
    transition.action_mean = raw
    transition.action_sigma = torch.full_like(raw, 0.05)
    transition.frontres_mask = torch.ones(n, 1)
    transition.frontres_actor_gate = torch.ones(n, 1)
    transition.supervised_target = torch.zeros(n, 6)
    transition.supervised_weight = torch.ones(n, 1)
    transition.supervised_harm_weight = torch.zeros(n, 1)
    transition.acceptance_target = torch.ones(n, 6)
    transition.acceptance_mask = torch.zeros(n, 6)
    transition.acceptance_margin = torch.zeros(n)
    transition.state_alpha_target = torch.zeros(n, 1)
    transition.state_alpha_mask = torch.zeros(n, 1)
    transition.hidden_states = None
    storage.add_transitions(transition)
    storage.returns.copy_(transition.values.view(1, n, 1))
    storage.advantages.zero_()


def main() -> None:
    torch.manual_seed(11)
    obs_dim = 8
    critic_obs_dim = 5
    policy = TinySplitPolicy(obs_dim, critic_obs_dim)
    alg = FrontRESUnified(
        policy,
        num_learning_epochs=1,
        num_mini_batches=2,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        device="cpu",
        lambda_supervised=0.25,
        diagnose_gradient_conflict=False,
        frontres_training_objective="hsl_hybrid",
        frontres_acceptance_preference_weight=1.0,
        frontres_acceptance_preference_focal_gamma=0.0,
        frontres_acceptance_preference_balance_min=1.0,
        frontres_acceptance_preference_balance_max=1.0,
        frontres_state_alpha_weight=0.0,
        frontres_structured_joint_rl_enabled=False,
        frontres_structured_joint_rl_weight=0.0,
        frontres_authority_actor_critic_enabled=False,
        frontres_authority_actor_loss_weight=0.0,
        frontres_authority_critic_loss_weight=0.0,
    )
    storage = RolloutStorage(
        training_type="frontres",
        num_envs=4,
        num_transitions_per_env=1,
        obs_shape=(obs_dim,),
        privileged_obs_shape=(critic_obs_dim,),
        actions_shape=(12,),
        device="cpu",
    )
    alg.storage = storage
    _fill_zero_acceptance_storage(storage, policy, obs_dim=obs_dim, critic_obs_dim=critic_obs_dim)

    loss_dict = alg.update()
    if storage.step != 0:
        raise AssertionError("storage was not cleared after update")
    if loss_dict["acceptance_preference_loss"] != 0.0:
        raise AssertionError("zero-mask acceptance batch should report zero acceptance loss")
    if loss_dict["hsl_acceptance_loss_enabled"] != 0.0:
        raise AssertionError("zero-mask acceptance batch should report HSL acceptance disabled for that batch")
    print("PASS: active Stage 2 update skips zero-mask acceptance loss without backward error.")


if __name__ == "__main__":
    main()
