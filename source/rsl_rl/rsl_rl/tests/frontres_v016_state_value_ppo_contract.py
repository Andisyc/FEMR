#!/usr/bin/env python3
"""Deterministic TEST-15/25D contracts for FRS-PPO-v012 B8/M4 state values."""

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
from rsl_rl.frontres.frontres_return_utility import frontres_symmetric_log_utility
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
    source_index = torch.arange(8).repeat_interleave(4)
    trial_index = torch.arange(4).repeat(8)
    segment_ids = torch.full((32,), 10) if shared_local_segment_id else 10 + source_index
    values = {
        "motion_ids": tuple(f"motion-{source}" for source in source_index.tolist()),
        "segment_ids": segment_ids,
        "source_index": source_index,
        "trial_index": trial_index,
        "noisy_segment_hashes": tuple(f"hash-{source}" for source in source_index.tolist()),
        "scenario_ids": tuple(f"scenario-{source}" for source in source_index.tolist()),
        "x_t_identities": tuple(f"x-{source}" for source in source_index.tolist()),
    }
    indices = [int(value) for value in order.tolist()]
    return FrontRESV015GroupedCandidateMetadata(
        transaction_id="tx-v006",
        policy_snapshot_id="pi-old-v006",
        motion_ids=tuple(values["motion_ids"][index] for index in indices),
        start_frames=(4 * (source_index + 1))[order],
        segment_ids=values["segment_ids"][order],
        source_index=values["source_index"][order],
        trial_index=values["trial_index"][order],
        horizon_k=torch.full((32,), 16)[order],
        evidence_valid_step_count=torch.full((32,), 16)[order],
        trial_role=("policy",) * 32,
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
    order = torch.arange(32) if order is None else order
    source_index = torch.arange(8).repeat_interleave(4)
    observations = torch.stack((source_index.float() + 1.0, torch.ones(32)), dim=-1)
    privileged = torch.zeros(32, 449)
    privileged[torch.arange(32), source_index] = 1.0
    base_returns = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat(8)
    signs = torch.where(source_index % 2 == 0, 1.0, -1.0)
    returns = signs * base_returns
    utility = torch.sign(returns) * torch.log1p(torch.abs(returns))
    old_values = 0.5 * signs
    segment_ids = torch.full((32,), 10) if shared_local_segment_id else 10 + source_index
    return FrontRESSegmentPPOBatch(
        observations=observations[order],
        privileged_observations=privileged[order],
        actions=torch.zeros(32, 6)[order],
        old_log_probs=torch.zeros(32)[order],
        old_values=old_values[order],
        returns=returns[order],
        advantages=(utility - old_values)[order],
        valid_mask=torch.ones(32, dtype=torch.bool)[order],
        segment_ids=segment_ids[order],
        old_means=torch.zeros(32, 6)[order],
        old_sigmas=torch.ones(32, 6)[order],
        transaction_metadata=_metadata(order, shared_local_segment_id=shared_local_segment_id),
        transaction_row_indices=torch.arange(32),
    )


def _cfg(weight: float = 1.0) -> FrontRESSegmentPPOConfig:
    return FrontRESSegmentPPOConfig(
        normalize_advantages=False,
        advantage_normalization="grouped_scale_only",
        actor_loss_weight=weight,
        critic_target_id="segment-exact-m-mean-symlog-v1",
    )


def _current_target_cfg(targets: tuple[float, ...]) -> FrontRESSegmentPPOConfig:
    return FrontRESSegmentPPOConfig(
        normalize_advantages=False,
        advantage_normalization="grouped_scale_only",
        actor_loss_weight=1.0,
        critic_target_id="scenario-current-exact-m4-mean-symlog-v1",
        critic_target_means_by_source=targets,
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
    utility = torch.log1p(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    target = float(utility.mean())
    expected_targets = tuple(
        target if source % 2 == 0 else -target
        for source in range(8)
        for _ in range(4)
    )
    torch.testing.assert_close(torch.tensor(result.critic_value_targets), torch.tensor(expected_targets))
    expected_segment_targets = torch.tensor([target if source % 2 == 0 else -target for source in range(8)])
    torch.testing.assert_close(torch.tensor(result.critic_segment_target_means), expected_segment_targets)
    expected_advantages = torch.cat(
        tuple((utility - 0.5) if source % 2 == 0 else (-utility + 0.5) for source in range(8))
    )
    torch.testing.assert_close(torch.tensor(result.actor_advantages), expected_advantages)

    order = torch.arange(31, -1, -1)
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


def test_current_m4_targets_do_not_change_actor_advantages() -> None:
    policy = _Policy()
    batch = _batch()
    utility = frontres_symmetric_log_utility(batch.returns)
    targets = tuple(float(utility[source * 4 : (source + 1) * 4].mean()) for source in range(8))
    result = compute_frontres_segment_ppo_loss(policy, batch, _current_target_cfg(targets))
    expected_rows = tuple(target for target in targets for _ in range(4))
    torch.testing.assert_close(torch.tensor(result.critic_value_targets), torch.tensor(expected_rows))
    assert result.critic_segment_target_means == targets
    expected_actor = frontres_symmetric_log_utility(batch.returns) - batch.old_values
    torch.testing.assert_close(torch.tensor(result.actor_advantages), expected_actor)

    order = torch.arange(31, -1, -1)
    permuted = compute_frontres_segment_ppo_loss(policy, _batch(order), _current_target_cfg(targets))
    torch.testing.assert_close(
        torch.tensor(permuted.critic_value_targets),
        torch.tensor(expected_rows)[order],
    )
    torch.testing.assert_close(result.total_loss, permuted.total_loss)

    _expect_error(
        lambda: compute_frontres_segment_ppo_loss(policy, batch, _current_target_cfg(targets[:-1])),
        "exactly eight",
    )
    stale_targets = tuple(value + 0.5 for value in targets)
    _expect_error(
        lambda: compute_frontres_segment_ppo_loss(policy, batch, _current_target_cfg(stale_targets)),
        "current exact-m4 mean",
    )
    m2_mask = torch.tensor([True, True, False, False]).repeat(8)
    m2_targets = tuple(
        float(utility[source * 4 : source * 4 + 2].mean()) for source in range(8)
    )
    _expect_error(
        lambda: compute_frontres_segment_ppo_loss(
            policy,
            replace(batch, valid_mask=m2_mask),
            _current_target_cfg(m2_targets),
        ),
        "exact m4",
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
    test_current_m4_targets_do_not_change_actor_advantages()
    test_separate_gradient_clip()
    print("frontres_v016_state_value_ppo_contract: segment mean + separate clip exact", flush=True)


if __name__ == "__main__":
    main()
