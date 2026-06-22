# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: execute the active FrontRES region-direct update path.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_region_direct_update_path.py

This is not an IsaacLab run.  It builds a tiny policy and a tiny RolloutStorage,
then calls ``FrontRESUnified.update()`` once.  The goal is to catch ordinary
integration bugs in the memory-safe region-direct branch before spending GPU
time on a live smoke test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified
from rsl_rl.storage.rollout_storage import RolloutStorage


class TinyTwoHeadActor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.proposal_head = nn.Linear(hidden_dim, 6)
        self.acceptance_head = nn.Linear(hidden_dim, 6)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.trunk(obs)
        return torch.cat([self.proposal_head(z), self.acceptance_head(z)], dim=-1)


class TinyFrontRESPolicy(nn.Module):
    is_recurrent = False
    num_task_corrections = 6
    task_conf_dim = 6

    def __init__(self, obs_dim: int, critic_obs_dim: int):
        super().__init__()
        self.residual_actor = TinyTwoHeadActor(obs_dim)
        self.critic = nn.Sequential(nn.Linear(critic_obs_dim, 16), nn.Tanh(), nn.Linear(16, 1))
        self.std = nn.Parameter(torch.full((12,), 0.05))
        self.max_delta_pos = 0.2
        self.max_delta_rpy = 0.2
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        self.distribution: Normal | None = None
        self.entropy: torch.Tensor | None = None

    def update_distribution(self, observations: torch.Tensor) -> None:
        self.action_mean = self.residual_actor(observations)
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


def _fill_storage(storage: RolloutStorage, policy: TinyFrontRESPolicy, *, obs_dim: int, critic_obs_dim: int) -> None:
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
    rho_adv = torch.tensor([1.0, -1.0, 0.8, -0.8], dtype=torch.float32).view(n, 1).expand(n, 6)
    transition.acceptance_target = rho_adv
    transition.acceptance_mask = torch.ones(n, 6)
    transition.rho_prior_authority = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32).view(n, 1)
    transition.rho_prior_target = torch.zeros(n, 6)
    transition.state_alpha_target = torch.zeros(n, 1)
    transition.state_alpha_mask = torch.zeros(n, 1)
    transition.hidden_states = None
    storage.add_transitions(transition)
    storage.returns.copy_(transition.values.view(1, n, 1))
    storage.advantages.zero_()


def run_region_direct_update_path_check() -> None:
    torch.manual_seed(7)
    obs_dim = 8
    critic_obs_dim = 5
    policy = TinyFrontRESPolicy(obs_dim, critic_obs_dim)
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
        frontres_acceptance_preference_weight=0.0,
        frontres_state_alpha_weight=0.0,
        frontres_structured_joint_rl_enabled=True,
        frontres_structured_joint_rl_weight=1.0,
        frontres_structured_joint_rl_loss_mode="region_direct",
        frontres_structured_joint_rl_disable_generic_ppo=True,
        frontres_structured_joint_rl_keep_legacy_bce=False,
        frontres_structured_joint_prior_loss_weight=1.0,
        frontres_structured_joint_repair_loss_kind="bce_logit",
        frontres_structured_joint_repair_loss_scale=1.0,
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
    _fill_storage(storage, policy, obs_dim=obs_dim, critic_obs_dim=critic_obs_dim)

    loss_dict = alg.update()
    required = {
        "value_function",
        "supervised_loss",
        "structured_joint_rl_loss",
        "structured_joint_rl_mode_region_direct",
        "structured_joint_rl_repair_loss_is_bce",
        "ppo_actor_weight",
        "raw_ppo_actor_weight",
    }
    missing = sorted(required.difference(loss_dict))
    if missing:
        raise AssertionError(f"missing loss_dict keys: {missing}")
    if loss_dict["ppo_actor_weight"] != 0.0:
        raise AssertionError("generic PPO should be disabled in region-direct mode")
    if loss_dict["structured_joint_rl_mode_region_direct"] != 1.0:
        raise AssertionError("region-direct mode was not reported active")
    if loss_dict["structured_joint_rl_repair_loss_is_bce"] != 1.0:
        raise AssertionError("repair BCE mode was not reported active")
    if storage.step != 0:
        raise AssertionError("storage was not cleared after update")

    print("=== FrontRES Region-Direct Update Path TEST ONLY ===")
    print(
        "update path: PASS "
        f"value={loss_dict['value_function']:.4f}, "
        f"sup={loss_dict['supervised_loss']:.4f}, "
        f"rho={loss_dict['structured_joint_rl_loss']:.4f}, "
        f"generic={loss_dict['ppo_actor_weight']:.1f}, "
        f"raw={loss_dict['raw_ppo_actor_weight']:.1f}"
    )
    print("meaning: CPU update reached the memory-safe region-direct branch and returned diagnostics.")


if __name__ == "__main__":
    run_region_direct_update_path_check()
