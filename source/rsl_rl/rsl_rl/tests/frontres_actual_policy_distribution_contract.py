#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch import nn
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic


def _probe_policy() -> FrontRESActorCritic:
    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    nn.Module.__init__(policy)
    policy.num_task_corrections = 6
    policy.noise_std_type = "scalar"
    policy.std = nn.Parameter(torch.tensor(0.01))
    raw_mean = torch.tensor([0.2, -0.1, 0.3, -0.2, 0.1, -0.3])
    policy._parse_observations = lambda observations: (observations, None, None)
    policy._frontres_raw_task_output = lambda observations: raw_mean.to(
        device=observations.device,
        dtype=observations.dtype,
    ).expand(observations.shape[0], -1)
    return policy


def test_actual_policy_uses_one_direct_full6_gaussian() -> None:
    policy = _probe_policy()
    observations = torch.zeros(4, 8)
    torch.manual_seed(7)
    actions = policy.act(observations)

    expected_mean = torch.tensor([0.2, -0.1, 0.3, -0.2, 0.1, -0.3]).expand(4, -1)
    torch.testing.assert_close(policy.action_mean, expected_mean)
    assert actions.shape == (4, 6)
    manual = torch.distributions.Normal(policy.action_mean, policy.action_std).log_prob(actions)
    expected_log_prob = manual.sum(dim=-1)
    torch.testing.assert_close(policy.get_actions_log_prob(actions), expected_log_prob)

    per_dim = policy.get_actions_log_prob_per_dim(actions, torch.arange(6))
    torch.testing.assert_close(per_dim.sum(dim=-1), expected_log_prob)


if __name__ == "__main__":
    test_actual_policy_uses_one_direct_full6_gaussian()
    print("frontres_actual_policy_distribution_contract: ok")
