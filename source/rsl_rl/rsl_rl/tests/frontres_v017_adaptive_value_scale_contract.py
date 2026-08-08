#!/usr/bin/env python3
"""Deterministic TEST-15 contracts for output-preserving Critic value scaling."""

from __future__ import annotations

from dataclasses import replace

import torch

from frontres_v016_state_value_ppo_contract import _Policy, _batch, _cfg
from rsl_rl.algorithms.frontres_segment_ppo import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FrontRESValueNormalizerState,
    compute_frontres_segment_ppo_loss,
    preview_frontres_v007_value_normalization,
)


def _normalized_cfg(state: FrontRESValueNormalizerState):
    return replace(
        _cfg(),
        critic_value_normalization=FRONTRES_VALUE_NORMALIZATION_ID,
        critic_value_normalizer_state=state,
        critic_value_normalizer_decay=0.9,
        critic_value_normalizer_scale_floor=1.0,
    )


def _expect_error(call, text: str) -> None:
    try:
        call()
    except (TypeError, ValueError, FloatingPointError) as exc:
        assert text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("expected value-normalizer rejection")


def test_hand_calculated_moments_and_nonamplifying_scale() -> None:
    state = FrontRESValueNormalizerState()
    update = preview_frontres_v007_value_normalization(
        torch.tensor([2.0, -2.0]),
        state,
        decay=0.9,
        scale_floor=1.0,
    )
    assert update.candidate.update_count == 1
    assert abs(update.candidate.mean) < 1.0e-12
    assert abs(update.candidate.second_moment - 1.3) < 1.0e-6
    assert abs(update.scale - (1.3**0.5)) < 1.0e-6
    assert update.scale >= 1.0

    extreme = preview_frontres_v007_value_normalization(
        torch.tensor([1578.0, 0.5]),
        state,
        decay=0.9,
        scale_floor=1.0,
    )
    assert extreme.scale > 300.0
    assert extreme.scale < 500.0
    permuted = preview_frontres_v007_value_normalization(
        torch.tensor([0.5, 1578.0]),
        state,
        decay=0.9,
        scale_floor=1.0,
    )
    assert permuted == extreme


def test_raw_actor_and_value_outputs_remain_unchanged() -> None:
    torch.manual_seed(7)
    policy = _Policy()
    batch = _batch()
    actor_before = {name: value.detach().clone() for name, value in policy.actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in policy.critic.state_dict().items()}
    raw = compute_frontres_segment_ppo_loss(policy, batch, _cfg())
    normalized = compute_frontres_segment_ppo_loss(
        policy,
        batch,
        _normalized_cfg(FrontRESValueNormalizerState()),
    )
    assert normalized.critic_value_targets == raw.critic_value_targets
    assert normalized.actor_advantages == raw.actor_advantages
    torch.testing.assert_close(normalized.actor_loss, raw.actor_loss)
    assert normalized.critic_raw_value_loss > 0.0
    torch.testing.assert_close(
        normalized.value_loss,
        normalized.critic_raw_value_loss / normalized.critic_value_scale**2,
    )
    assert normalized.critic_value_normalizer_candidate_state.update_count == 1
    for name, value in actor_before.items():
        torch.testing.assert_close(policy.actor.state_dict()[name], value)
    for name, value in critic_before.items():
        torch.testing.assert_close(policy.critic.state_dict()[name], value)


def test_floor_zero_variance_and_invalid_state() -> None:
    update = preview_frontres_v007_value_normalization(
        torch.tensor([0.0, 0.0]),
        FrontRESValueNormalizerState(mean=0.0, second_moment=0.0, update_count=4),
        decay=0.9,
        scale_floor=1.0,
    )
    assert update.scale == 1.0
    _expect_error(
        lambda: preview_frontres_v007_value_normalization(
            torch.tensor([1.0, float("nan")]),
            FrontRESValueNormalizerState(),
            decay=0.9,
            scale_floor=1.0,
        ),
        "finite",
    )
    _expect_error(
        lambda: FrontRESValueNormalizerState(mean=2.0, second_moment=1.0, update_count=1).validate(),
        "second moment",
    )


def main() -> None:
    test_hand_calculated_moments_and_nonamplifying_scale()
    test_raw_actor_and_value_outputs_remain_unchanged()
    test_floor_zero_variance_and_invalid_state()
    print("frontres_v017_adaptive_value_scale_contract: output-preserving scale exact", flush=True)


if __name__ == "__main__":
    main()
