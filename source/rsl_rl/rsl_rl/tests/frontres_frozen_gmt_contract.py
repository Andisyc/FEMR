#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "source" / "rsl_rl"))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified
from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic


def _toy_frontres_policy() -> FrontRESActorCritic:
    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    nn.Module.__init__(policy)
    policy.gmt_policy = nn.Linear(3, 3)
    policy.gmt_normalizer = nn.BatchNorm1d(3)
    policy.gmt_normalizer.until = 10
    policy.ref_vel_estimator = nn.Linear(3, 2)
    policy.residual_actor = nn.Linear(3, 6)
    policy.critic = nn.Linear(3, 1)
    policy.log_std = nn.Parameter(torch.zeros(6))
    policy.std = None
    policy.enforce_frozen_gmt_inference()
    return policy


def test_parent_train_keeps_frozen_gmt_family_in_inference_mode() -> None:
    policy = _toy_frontres_policy()
    policy.eval()
    policy.train()

    assert policy.residual_actor.training
    assert policy.critic.training
    for module in (policy.gmt_policy, policy.gmt_normalizer, policy.ref_vel_estimator):
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())
    assert policy.gmt_normalizer.until == 0


def test_frozen_gmt_is_excluded_from_optimizer_and_unchanged_after_update() -> None:
    torch.manual_seed(7)
    policy = _toy_frontres_policy()
    trainable = list(FrontRESUnified._collect_trainable_params(policy))
    trainable_ids = {id(param) for param in trainable}
    gmt_params = tuple(policy.gmt_policy.parameters())

    assert gmt_params
    assert all(not param.requires_grad for param in gmt_params)
    assert all(id(param) not in trainable_ids for param in gmt_params)
    assert all(id(param) in trainable_ids for param in policy.residual_actor.parameters())
    assert all(id(param) in trainable_ids for param in policy.critic.parameters())
    assert id(policy.log_std) in trainable_ids

    optimizer = torch.optim.Adam(trainable, lr=1.0e-2)
    before_gmt = tuple(param.detach().clone() for param in gmt_params)
    before_actor = tuple(param.detach().clone() for param in policy.residual_actor.parameters())
    before_critic = tuple(param.detach().clone() for param in policy.critic.parameters())
    obs = torch.randn(8, 3)
    loss = policy.residual_actor(obs).square().mean() + policy.critic(obs).square().mean() + policy.log_std.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert all(param.grad is None for param in gmt_params)
    assert all(torch.equal(param, before) for param, before in zip(gmt_params, before_gmt))
    assert any(not torch.equal(param, before) for param, before in zip(policy.residual_actor.parameters(), before_actor))
    assert any(not torch.equal(param, before) for param, before in zip(policy.critic.parameters(), before_critic))


if __name__ == "__main__":
    test_parent_train_keeps_frozen_gmt_family_in_inference_mode()
    test_frozen_gmt_is_excluded_from_optimizer_and_unchanged_after_update()
    print("frontres_frozen_gmt_contract: ok")
