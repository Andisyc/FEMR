#!/usr/bin/env python3
"""TEST-14/15 contracts for the TRAIN-v019 symmetric-log utility."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
)
from rsl_rl.frontres.frontres_return_utility import (
    FRONTRES_RETURN_UTILITY_ID,
    FRONTRES_RETURN_UTILITY_SCALE,
    frontres_symmetric_log_utility,
)
from rsl_rl.frontres.frontres_segment_rollout_storage import FrontRESSegmentRolloutStorage
from rsl_rl.frontres.frontres_segment_storage_records import (
    FrontRESSegmentTransition,
    FrontRESV015GroupedCandidateMetadata,
)


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, 1, bias=False)
        self.critic = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.critic.weight)

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        return {
            "log_prob": self.actor(observations).reshape(-1),
            "value": self.critic(observations).reshape(-1),
            "entropy": torch.zeros(observations.shape[0]),
        }


def _utility(raw: torch.Tensor) -> torch.Tensor:
    return torch.sign(raw) * torch.log1p(torch.abs(raw))


def _metadata() -> FrontRESV015GroupedCandidateMetadata:
    return FrontRESV015GroupedCandidateMetadata(
        transaction_id="tx-v019-utility",
        policy_snapshot_id="pi-old-v019",
        motion_ids=("motion-a",) * 4 + ("motion-b",) * 4,
        start_frames=torch.tensor([4] * 4 + [8] * 4),
        segment_ids=torch.tensor([10] * 4 + [11] * 4),
        source_index=torch.tensor([0] * 4 + [1] * 4),
        trial_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        horizon_k=torch.full((8,), 8),
        evidence_valid_step_count=torch.full((8,), 8),
        trial_role=("policy",) * 8,
        noisy_segment_hashes=("ha",) * 4 + ("hb",) * 4,
        scenario_ids=("sa",) * 4 + ("sb",) * 4,
        x_t_identities=("xa",) * 4 + ("xb",) * 4,
        intent_q29_provenance="deployment_noisy_q29",
        intent_q29_source="motion_internal_q29",
    )


def _batch() -> tuple[FrontRESSegmentPPOBatch, torch.Tensor, torch.Tensor]:
    raw = torch.tensor([100.0, -1.0, -1.0, -1.0, -100.0, 1.0, 1.0, 1.0])
    old_value = torch.tensor([0.25] * 4 + [-0.5] * 4)
    utility = _utility(raw)
    privileged = torch.zeros(8, 449)
    privileged[:4, 0] = 1.0
    privileged[4:, 1] = 1.0
    batch = FrontRESSegmentPPOBatch(
        observations=torch.tensor([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4),
        privileged_observations=privileged,
        actions=torch.zeros(8, 6),
        old_log_probs=torch.zeros(8),
        old_values=old_value,
        returns=raw,
        advantages=utility - old_value,
        valid_mask=torch.ones(8, dtype=torch.bool),
        segment_ids=torch.tensor([10] * 4 + [11] * 4),
        old_means=torch.zeros(8, 6),
        old_sigmas=torch.ones(8, 6),
        transaction_metadata=_metadata(),
        transaction_row_indices=torch.arange(8),
    )
    return batch, raw, utility


def _cfg() -> FrontRESSegmentPPOConfig:
    return FrontRESSegmentPPOConfig(
        normalize_advantages=False,
        advantage_normalization="grouped_scale_only",
        critic_target_id="segment-exact-m-mean-symlog-v1",
    )


def _expect_error(call, text: str) -> None:
    try:
        call()
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        assert text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("expected TRAIN-v019 utility rejection")


def test_fixed_utility_owner() -> None:
    assert FRONTRES_RETURN_UTILITY_ID == "symmetric-log-gain-g0-1-v1"
    assert FRONTRES_RETURN_UTILITY_SCALE == 1.0
    raw = torch.tensor([-1087.0, -10.0, -0.16, 0.0, 0.16, 10.0, 105.0])
    actual = frontres_symmetric_log_utility(raw)
    torch.testing.assert_close(actual, _utility(raw))
    assert torch.equal(torch.sign(actual), torch.sign(raw))
    assert bool(torch.all(actual[1:] > actual[:-1]))
    assert float(actual[-1]) < float(raw[-1])
    assert not actual.requires_grad

    _expect_error(lambda: frontres_symmetric_log_utility(torch.tensor([float("nan")])), "finite")
    _expect_error(lambda: frontres_symmetric_log_utility(torch.tensor([1.0], requires_grad=True)), "detached")


def test_storage_retains_raw_return_and_builds_utility_advantage() -> None:
    raw = torch.tensor([100.0, -1.0, 0.0, 3.0])
    old_value = torch.tensor([0.25, 0.25, -0.5, -0.5])
    storage = FrontRESSegmentRolloutStorage(capacity=4, obs_shape=(2,))
    storage.add_transition(
        FrontRESSegmentTransition(
            observations=torch.zeros(4, 2),
            actions=torch.zeros(4, 6),
            old_log_probs=torch.zeros(4),
            values=old_value,
            rewards=raw,
            valid_mask=torch.ones(4, dtype=torch.bool),
            reset_mask=torch.ones(4, dtype=torch.bool),
            segment_ids=torch.tensor([10, 10, 11, 11]),
        )
    )
    storage.compute_returns_and_advantages()
    batch = storage.full_batch()
    torch.testing.assert_close(batch.returns, raw)
    torch.testing.assert_close(batch.advantages, _utility(raw) - old_value)


def test_ppo_transforms_before_m4_mean_and_shares_utility() -> None:
    policy = _Policy()
    batch, raw, utility = _batch()
    result = compute_frontres_segment_ppo_loss(policy, batch, _cfg())
    expected_means = (float(utility[:4].mean()), float(utility[4:].mean()))
    expected_targets = torch.tensor([expected_means[0]] * 4 + [expected_means[1]] * 4)
    torch.testing.assert_close(torch.tensor(result.critic_value_targets), expected_targets)
    torch.testing.assert_close(torch.tensor(result.actor_advantages), utility - batch.old_values)
    assert result.critic_segment_target_means == expected_means

    wrong_after_mean = torch.stack((_utility(raw[:4].mean()), _utility(raw[4:].mean())))
    assert not torch.allclose(torch.tensor(expected_means), wrong_after_mean)

    wrong_advantage = replace(batch, advantages=raw - batch.old_values)
    _expect_error(lambda: compute_frontres_segment_ppo_loss(policy, wrong_advantage, _cfg()), "utility advantage")


def test_legacy_raw_segment_target_remains_explicitly_separate() -> None:
    policy = _Policy()
    batch, raw, _utility_returns = _batch()
    legacy_batch = replace(batch, advantages=raw - batch.old_values)
    legacy_cfg = replace(_cfg(), critic_target_id="segment-exact-m-mean-v1")
    result = compute_frontres_segment_ppo_loss(policy, legacy_batch, legacy_cfg)
    expected = torch.tensor([float(raw[:4].mean())] * 4 + [float(raw[4:].mean())] * 4)
    torch.testing.assert_close(torch.tensor(result.critic_value_targets), expected)
    assert result.return_utility_id == "none"


def main() -> None:
    test_fixed_utility_owner()
    test_storage_retains_raw_return_and_builds_utility_advantage()
    test_ppo_transforms_before_m4_mean_and_shares_utility()
    test_legacy_raw_segment_target_remains_explicitly_separate()
    print("frontres_v019_symmetric_log_utility_contract: exact", flush=True)


if __name__ == "__main__":
    main()
