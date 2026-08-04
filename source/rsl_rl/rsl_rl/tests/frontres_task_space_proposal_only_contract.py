#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[4]
RSL_SOURCE = ROOT / "source/rsl_rl"
if str(RSL_SOURCE) not in sys.path:
    sys.path.insert(0, str(RSL_SOURCE))

from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic


def _build_policy() -> FrontRESActorCritic:
    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    nn.Module.__init__(policy)
    policy.num_task_corrections = 6
    policy.total_output_dim = 6
    policy.num_frontres_obs = 158
    policy.noise_std_type = "scalar"
    policy.residual_actor = nn.Linear(158, 6, bias=False)
    policy.register_buffer("std", torch.full((6,), 0.05))
    policy._parse_observations = lambda observations: (observations, None, None)
    with torch.no_grad():
        policy.residual_actor.weight.zero_()
        policy.residual_actor.weight[:, :4].copy_(torch.eye(6, 4))
    return policy


def test_proposal_only_task_space_policy_is_6d() -> None:
    policy = _build_policy()
    observations = torch.zeros(2, 158, dtype=torch.float32)
    observations[:, :4] = torch.tensor(
        [[1.10, -1.20, 1.30, -1.40], [-1.30, 1.20, -1.10, 1.40]],
        dtype=torch.float32,
    )
    policy.update_distribution(observations)
    actions = policy.act(observations)
    log_prob = policy.get_actions_log_prob(actions)
    selected = policy.get_actions_log_prob_selected(actions, [3, 4])
    per_dim = policy.get_actions_log_prob_per_dim(actions, [3, 4])
    correction = policy.get_task_correction_inference(observations)

    print(
        "[FrontRES Proposal-Only Task-Space Contract] "
        f"mean_shape={tuple(policy.distribution.mean.shape)} "
        f"std_shape={tuple(policy.distribution.stddev.shape)} "
        f"action_shape={tuple(actions.shape)} "
        f"log_prob_shape={tuple(log_prob.shape)} "
        f"selected_shape={tuple(selected.shape)} "
        f"per_dim_shape={tuple(per_dim.shape)} "
        f"correction_shape={tuple(correction.shape)}",
        flush=True,
    )

    assert tuple(policy.distribution.mean.shape) == (2, 6)
    assert tuple(policy.distribution.stddev.shape) == (2, 6)
    assert tuple(actions.shape) == (2, 6)
    assert tuple(log_prob.shape) == (2,)
    assert tuple(selected.shape) == (2,)
    assert tuple(per_dim.shape) == (2, 2)
    assert tuple(correction.shape) == (2, 6)
    expected_correction = policy.residual_actor(observations)
    torch.testing.assert_close(correction, expected_correction)
    assert float(correction.abs().max()) > 1.0


if __name__ == "__main__":
    test_proposal_only_task_space_policy_is_6d()
    print("frontres_task_space_proposal_only_contract: ok")
