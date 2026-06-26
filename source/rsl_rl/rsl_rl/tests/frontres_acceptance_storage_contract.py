#!/usr/bin/env python3
"""TEST ONLY: FEMR HSL+HRL acceptance storage contract.

This checks storage plumbing only.  It proves the active contract can carry
proposal, acceptance action/logit/prob, rollout-built GT, mask, and margin
without requiring authority actor-critic fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.storage.rollout_storage import RolloutStorage


ACCEPTANCE_START = 36
BATCH_INDEX = 41


def _make_transition(num_envs: int) -> RolloutStorage.Transition:
    transition = RolloutStorage.Transition()
    transition.observations = torch.arange(num_envs * 4, dtype=torch.float32).view(num_envs, 4)
    transition.actions = torch.zeros(num_envs, 12)
    transition.actions[:, :6] = torch.arange(num_envs, dtype=torch.float32).view(-1, 1) + torch.arange(6).view(1, -1) * 0.01
    transition.actions[:, 6:12] = torch.tensor([
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        [0.90, 0.80, 0.70, 0.60, 0.50, 0.40],
        [-0.10, 1.10, 0.25, 0.75, 0.00, 1.00],
    ])[:num_envs]
    transition.rewards = torch.zeros(num_envs)
    transition.dones = torch.zeros(num_envs)
    transition.values = torch.zeros(num_envs, 1)
    transition.actions_log_prob = torch.zeros(num_envs)
    transition.action_mean = torch.zeros(num_envs, 12)
    logits = torch.tensor([
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
        [0.5, -0.5, 1.5, -1.5, 2.5, -2.5],
    ])[:num_envs]
    transition.action_mean[:, 6:12] = logits
    transition.action_sigma = torch.ones(num_envs, 12)
    transition.frontres_mask = torch.ones(num_envs, 1)
    transition.frontres_actor_gate = torch.ones(num_envs, 1)
    transition.supervised_target = torch.zeros(num_envs, 6)
    transition.supervised_weight = torch.ones(num_envs, 1)
    transition.supervised_harm_weight = torch.zeros(num_envs, 1)
    transition.acceptance_gt = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])[:num_envs]
    transition.acceptance_mask = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])[:num_envs]
    transition.acceptance_margin = torch.tensor([0.20, -0.30, 0.002])[:num_envs].view(-1, 1)
    transition.state_alpha_target = torch.zeros(num_envs, 1)
    transition.state_alpha_mask = torch.zeros(num_envs, 1)
    return transition


def test_acceptance_storage_round_trip() -> None:
    num_envs = 3
    storage = RolloutStorage(
        training_type="frontres",
        num_envs=num_envs,
        num_transitions_per_env=1,
        obs_shape=(4,),
        privileged_obs_shape=None,
        actions_shape=(12,),
        device="cpu",
    )
    storage.yield_batch_indices = True
    transition = _make_transition(num_envs)
    storage.add_transitions(transition)

    # The active fields are explicit; the old target field is only a mirror.
    torch.testing.assert_close(storage.proposal_delta_se[0], transition.actions[:, :6])
    torch.testing.assert_close(storage.acceptance_action[0], transition.actions[:, 6:12].clamp(0.0, 1.0))
    torch.testing.assert_close(storage.acceptance_logit[0], transition.action_mean[:, 6:12])
    torch.testing.assert_close(storage.acceptance_prob[0], torch.sigmoid(transition.action_mean[:, 6:12]))
    torch.testing.assert_close(storage.acceptance_gt[0], transition.acceptance_gt)
    torch.testing.assert_close(storage.acceptance_mask[0], transition.acceptance_mask)
    torch.testing.assert_close(storage.acceptance_margin[0], transition.acceptance_margin)
    torch.testing.assert_close(storage.acceptance_target[0], transition.acceptance_gt)

    # No authority actor-critic fields are required for the acceptance contract.
    if float(storage.authority_mask.abs().sum().item()) != 0.0:
        raise AssertionError("authority_mask should stay zero when Step-5 acceptance storage is used alone")
    if float(storage.authority_action.abs().sum().item()) != 0.0:
        raise AssertionError("authority_action should stay zero when Step-5 acceptance storage is used alone")

    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    if len(batch) != 42:
        raise AssertionError(f"FrontRES acceptance batch should have 42 entries, got {len(batch)}")
    idx = batch[BATCH_INDEX]
    torch.testing.assert_close(batch[28], storage.proposal_delta_se.flatten(0, 1)[idx])
    torch.testing.assert_close(batch[ACCEPTANCE_START + 0], storage.acceptance_action.flatten(0, 1)[idx])
    torch.testing.assert_close(batch[ACCEPTANCE_START + 1], storage.acceptance_logit.flatten(0, 1)[idx])
    torch.testing.assert_close(batch[ACCEPTANCE_START + 2], storage.acceptance_prob.flatten(0, 1)[idx])
    torch.testing.assert_close(batch[ACCEPTANCE_START + 3], storage.acceptance_gt.flatten(0, 1)[idx])
    torch.testing.assert_close(batch[ACCEPTANCE_START + 4], storage.acceptance_margin.flatten(0, 1)[idx])


def main() -> None:
    test_acceptance_storage_round_trip()
    print("=== FrontRES Acceptance Storage Contract TEST ONLY ===")
    print("fields=proposal_delta_se, acceptance_action/logit/prob, acceptance_gt/mask/margin")
    print("authority_required=0")
    print("result: PASS")


if __name__ == "__main__":
    main()
