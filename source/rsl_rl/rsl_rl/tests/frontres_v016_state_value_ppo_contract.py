#!/usr/bin/env python3
"""Deterministic TEST-15 contracts for FRS-PPO-v009."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESScalarGradientInstallResult,
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    FrontRESSegmentPPOResult,
    compute_frontres_segment_ppo_loss,
    install_frontres_v006_scalar_gradients,
)
from rsl_rl.frontres.frontres_segment_storage_records import FrontRESV015GroupedCandidateMetadata


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, 1, bias=False)
        self.critic = torch.nn.Linear(2, 1, bias=False)

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        return {
            "log_prob": self.actor(observations).reshape(-1),
            "value": self.critic(observations).reshape(-1),
            "entropy": torch.zeros(observations.shape[0]),
        }


def _metadata(
    order: torch.Tensor,
    *,
    shared_local_segment_id: bool = False,
) -> FrontRESV015GroupedCandidateMetadata:
    segment_ids = [10, 10, 10, 10, 10, 10] if shared_local_segment_id else [10, 10, 10, 11, 11, 11]
    values = {
        "motion_ids": ("motion-a",) * 3 + ("motion-b",) * 3,
        "segment_ids": torch.tensor(segment_ids),
        "source_index": torch.tensor([0, 0, 0, 1, 1, 1]),
        "trial_index": torch.tensor([0, 1, 2, 0, 1, 2]),
        "noisy_segment_hashes": ("ha",) * 3 + ("hb",) * 3,
        "scenario_ids": ("sa",) * 3 + ("sb",) * 3,
        "x_t_identities": ("xa",) * 3 + ("xb",) * 3,
    }
    indices = [int(value) for value in order.tolist()]
    return FrontRESV015GroupedCandidateMetadata(
        transaction_id="tx-v006",
        policy_snapshot_id="pi-old-v006",
        motion_ids=tuple(values["motion_ids"][index] for index in indices),
        start_frames=torch.tensor([4, 4, 4, 8, 8, 8])[order],
        segment_ids=values["segment_ids"][order],
        source_index=values["source_index"][order],
        trial_index=values["trial_index"][order],
        horizon_k=torch.full((6,), 16)[order],
        evidence_valid_step_count=torch.full((6,), 16)[order],
        trial_role=("policy",) * 6,
        noisy_segment_hashes=tuple(values["noisy_segment_hashes"][index] for index in indices),
        scenario_ids=tuple(values["scenario_ids"][index] for index in indices),
        x_t_identities=tuple(values["x_t_identities"][index] for index in indices),
        intent_q29_provenance="deployment_noisy_q29",
        intent_q29_source="motion_internal_q29",
    )


def _batch(
    order: torch.Tensor | None = None,
    *,
    shared_local_segment_id: bool = False,
) -> FrontRESSegmentPPOBatch:
    order = torch.arange(6) if order is None else order
    observations = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    privileged = torch.zeros(6, 449)
    privileged[:3, 0] = 1.0
    privileged[3:, 1] = 1.0
    returns = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    utility = torch.sign(returns) * torch.log1p(torch.abs(returns))
    old_values = torch.tensor([0.5, 0.5, 0.5, -0.5, -0.5, -0.5])
    segment_ids = torch.tensor(
        [10, 10, 10, 10, 10, 10] if shared_local_segment_id else [10, 10, 10, 11, 11, 11]
    )
    return FrontRESSegmentPPOBatch(
        observations=observations[order],
        privileged_observations=privileged[order],
        actions=torch.zeros(6, 6)[order],
        old_log_probs=torch.zeros(6)[order],
        old_values=old_values[order],
        returns=returns[order],
        advantages=(utility - old_values)[order],
        valid_mask=torch.ones(6, dtype=torch.bool)[order],
        segment_ids=segment_ids[order],
        old_means=torch.zeros(6, 6)[order],
        old_sigmas=torch.ones(6, 6)[order],
        transaction_metadata=_metadata(order, shared_local_segment_id=shared_local_segment_id),
        transaction_row_indices=torch.arange(6),
    )


def _cfg(weight: float = 1.0) -> FrontRESSegmentPPOConfig:
    return FrontRESSegmentPPOConfig(
        normalize_advantages=False,
        advantage_normalization="grouped_scale_only",
        actor_loss_weight=weight,
        critic_target_id="segment-exact-m-mean-symlog-v1",
    )


def _expect_error(call, text: str) -> None:
    try:
        call()
    except (RuntimeError, ValueError) as exc:
        assert text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("expected PPO contract rejection")


def test_segment_mean_target_and_permutation() -> None:
    policy = _Policy()
    result = compute_frontres_segment_ppo_loss(policy, _batch(), _cfg())
    utility = torch.log1p(torch.tensor([1.0, 2.0, 3.0]))
    target = float(utility.mean())
    expected_targets = (target, target, target, -target, -target, -target)
    torch.testing.assert_close(torch.tensor(result.critic_value_targets), torch.tensor(expected_targets))
    torch.testing.assert_close(torch.tensor(result.critic_segment_target_means), torch.tensor((target, -target)))
    expected_advantages = torch.tensor(
        [utility[0] - 0.5, utility[1] - 0.5, utility[2] - 0.5,
         -utility[0] + 0.5, -utility[1] + 0.5, -utility[2] + 0.5]
    )
    torch.testing.assert_close(torch.tensor(result.actor_advantages), expected_advantages)

    order = torch.tensor([4, 0, 5, 2, 3, 1])
    permuted = compute_frontres_segment_ppo_loss(policy, _batch(order), _cfg())
    expected = torch.tensor(result.critic_value_targets)[order]
    torch.testing.assert_close(torch.tensor(permuted.critic_value_targets), expected)
    torch.testing.assert_close(result.total_loss, permuted.total_loss)

    shared_local_id = compute_frontres_segment_ppo_loss(
        policy,
        _batch(shared_local_segment_id=True),
        _cfg(),
    )
    torch.testing.assert_close(
        torch.tensor(shared_local_id.critic_value_targets), torch.tensor(result.critic_value_targets)
    )
    torch.testing.assert_close(
        torch.tensor(shared_local_id.critic_segment_target_means),
        torch.tensor(result.critic_segment_target_means),
    )

    aliased = _batch()
    privileged = aliased.privileged_observations.clone()
    privileged[1, 0] += 1.0
    _expect_error(
        lambda: compute_frontres_segment_ppo_loss(policy, replace(aliased, privileged_observations=privileged), _cfg()),
        "identical critic state",
    )
    old_values = aliased.old_values.clone()
    old_values[2] += 0.25
    adjusted_advantages = (
        torch.sign(aliased.returns) * torch.log1p(torch.abs(aliased.returns)) - old_values
    )
    _expect_error(
        lambda: compute_frontres_segment_ppo_loss(
            policy,
            replace(aliased, old_values=old_values, advantages=adjusted_advantages),
            _cfg(),
        ),
        "shared old value",
    )


def test_separate_gradient_clip() -> None:
    policy = _Policy()
    actor_loss = 100.0 * policy.actor.weight.sum()
    value_loss = 0.01 * policy.critic.weight.sum()
    zero = actor_loss * 0.0
    result = FrontRESSegmentPPOResult(
        total_loss=actor_loss + value_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        entropy=zero,
        valid_count=1,
        valid_frac=1.0,
        clip_frac=0.0,
        approx_kl=0.0,
        ratio_mean=1.0,
    )
    installed = install_frontres_v006_scalar_gradients(
        policy,
        result,
        _cfg(),
        tuple(policy.parameters()),
        max_grad_norm=0.5,
    )
    assert isinstance(installed, FrontRESScalarGradientInstallResult)
    assert installed.actor_pre_clip_norm > 0.5
    assert installed.actor_post_clip_norm <= 0.500001
    assert installed.critic_pre_clip_norm < 0.5
    assert installed.critic_post_clip_norm == installed.critic_pre_clip_norm
    assert installed.actor_clip_coefficient < 1.0
    assert installed.critic_clip_coefficient == 1.0

    frozen_policy = _Policy()
    frozen_value_loss = frozen_policy.critic.weight.square().sum()
    frozen = FrontRESSegmentPPOResult(
        total_loss=frozen_value_loss,
        actor_loss=frozen_policy.actor.weight.sum(),
        value_loss=frozen_value_loss,
        entropy=frozen_value_loss * 0.0,
        valid_count=1,
        valid_frac=1.0,
        clip_frac=0.0,
        approx_kl=0.0,
        ratio_mean=1.0,
    )
    frozen_install = install_frontres_v006_scalar_gradients(
        frozen_policy,
        frozen,
        _cfg(weight=0.0),
        tuple(frozen_policy.parameters()),
        max_grad_norm=0.5,
    )
    assert all(parameter.grad is None for parameter in frozen_install.actor_parameters)
    assert frozen_install.actor_pre_clip_norm == 0.0
    assert frozen_install.critic_post_clip_norm <= 0.500001


def main() -> None:
    test_segment_mean_target_and_permutation()
    test_separate_gradient_clip()
    print("frontres_v016_state_value_ppo_contract: segment mean + separate clip exact", flush=True)


if __name__ == "__main__":
    main()
