#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


segment_ppo = _load("frontres_segment_ppo", ROOT / "rsl_rl" / "algorithms" / "frontres_segment_ppo.py")

FrontRESSegmentPPOBatch = segment_ppo.FrontRESSegmentPPOBatch
FrontRESSegmentPPOConfig = segment_ppo.FrontRESSegmentPPOConfig
FrontRESSegmentPolicyEval = segment_ppo.FrontRESSegmentPolicyEval
compute_frontres_segment_ppo_loss = segment_ppo.compute_frontres_segment_ppo_loss


class FakeSegmentPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(4, 6, bias=False)
        self.critic = torch.nn.Linear(4, 1, bias=False)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.critic.weight)
        self.acceptance_called = False

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> FrontRESSegmentPolicyEval:
        mean = self.actor(observations)
        value = self.critic(observations).squeeze(-1)
        log_prob = -0.5 * (actions - mean).square().sum(dim=-1)
        entropy = torch.ones_like(log_prob) * 0.5
        return FrontRESSegmentPolicyEval(log_prob=log_prob, value=value, entropy=entropy, mean=mean)

    def acceptance_loss(self, *args, **kwargs):
        self.acceptance_called = True
        raise AssertionError("old acceptance path must not be used by segment PPO")


def _batch(invalid_action: float = 20.0, invalid_advantage: float = 1000.0) -> FrontRESSegmentPPOBatch:
    return FrontRESSegmentPPOBatch(
        observations=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        actions=torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [invalid_action, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        old_log_probs=torch.zeros(2),
        old_values=torch.zeros(2),
        returns=torch.tensor([1.0, 999.0]),
        advantages=torch.tensor([1.0, invalid_advantage]),
        valid_mask=torch.tensor([True, False]),
        segment_ids=torch.tensor([7, 8]),
        action_mask=torch.ones(2, 6),
    )


def test_fake_batch_updates_actor_on_valid_segments() -> None:
    policy = FakeSegmentPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.actor.weight.detach().clone()
    result = compute_frontres_segment_ppo_loss(policy, _batch(), FrontRESSegmentPPOConfig(entropy_coef=0.0))
    assert result.should_step
    assert result.valid_count == 1
    assert result.valid_frac == 0.5
    optimizer.zero_grad(set_to_none=True)
    result.total_loss.backward()
    optimizer.step()
    assert not torch.allclose(policy.actor.weight.detach(), before)
    assert not policy.acceptance_called


def test_invalid_samples_do_not_contribute_to_loss() -> None:
    policy = FakeSegmentPolicy()
    clean_invalid = compute_frontres_segment_ppo_loss(policy, _batch(invalid_action=1.0, invalid_advantage=1.0))
    extreme_invalid = compute_frontres_segment_ppo_loss(policy, _batch(invalid_action=1e6, invalid_advantage=1e6))
    torch.testing.assert_close(clean_invalid.actor_loss, extreme_invalid.actor_loss)
    torch.testing.assert_close(clean_invalid.value_loss, extreme_invalid.value_loss)
    torch.testing.assert_close(clean_invalid.total_loss, extreme_invalid.total_loss)


def test_nonfinite_valid_rows_are_masked_before_loss() -> None:
    policy = FakeSegmentPolicy()
    batch = _batch()
    batch = FrontRESSegmentPPOBatch(
        observations=batch.observations,
        actions=batch.actions,
        old_log_probs=torch.tensor([0.0, float("nan")]),
        old_values=batch.old_values,
        returns=batch.returns,
        advantages=batch.advantages,
        valid_mask=torch.tensor([True, True]),
        segment_ids=batch.segment_ids,
        action_mask=batch.action_mask,
    )
    result = compute_frontres_segment_ppo_loss(policy, batch)
    print(
        "[probe nonfinite_mask] "
        f"valid_count={result.valid_count} "
        f"total_loss_finite={torch.isfinite(result.total_loss).item()}",
        flush=True,
    )
    assert result.valid_count == 1
    assert torch.isfinite(result.total_loss)


def test_extreme_log_ratio_does_not_overflow_loss() -> None:
    policy = FakeSegmentPolicy()
    batch = _batch(invalid_action=1.0, invalid_advantage=1.0)
    batch = FrontRESSegmentPPOBatch(
        observations=batch.observations[:1],
        actions=torch.zeros(1, 6),
        old_log_probs=torch.tensor([-1000.0]),
        old_values=torch.zeros(1),
        returns=torch.zeros(1),
        advantages=torch.tensor([-1.0]),
        valid_mask=torch.tensor([True]),
        segment_ids=torch.tensor([7]),
        action_mask=torch.ones(1, 6),
    )
    result = compute_frontres_segment_ppo_loss(policy, batch)
    print(
        "[probe log_ratio_layer] "
        f"valid_count={result.valid_count} "
        f"old_logp_mean={result.old_log_prob_mean:.6f} "
        f"new_logp_mean={result.new_log_prob_mean:.6f} "
        f"raw_log_ratio_max={result.raw_log_ratio_max:.6f} "
        f"ratio_mean={result.ratio_mean:.6e} "
        f"ratio_max={result.ratio_max:.6e} "
        f"advantage_min={result.advantage_min:.6f} "
        f"actor_loss_finite={torch.isfinite(result.actor_loss).item()} "
        f"total_loss_finite={torch.isfinite(result.total_loss).item()}",
        flush=True,
    )
    assert result.valid_count == 1
    assert torch.isfinite(result.actor_loss)
    assert torch.isfinite(result.total_loss)
    assert result.raw_log_ratio_max >= 999.0
    assert result.ratio_mean > 1e8
    assert result.clip_frac == 1.0


def test_ppo_tuple_requires_6d_action_and_vector_fields() -> None:
    policy = FakeSegmentPolicy()
    bad_action = _batch()
    bad_action = FrontRESSegmentPPOBatch(
        observations=bad_action.observations,
        actions=torch.zeros(2, 5),
        old_log_probs=bad_action.old_log_probs,
        old_values=bad_action.old_values,
        returns=bad_action.returns,
        advantages=bad_action.advantages,
        valid_mask=bad_action.valid_mask,
    )
    try:
        compute_frontres_segment_ppo_loss(policy, bad_action)
    except ValueError as exc:
        assert "actions must have shape [B, 6]" in str(exc)
    else:
        raise AssertionError("5D action should be rejected")

    bad_log_prob = _batch()
    bad_log_prob = FrontRESSegmentPPOBatch(
        observations=bad_log_prob.observations,
        actions=bad_log_prob.actions,
        old_log_probs=torch.zeros(2, 1),
        old_values=bad_log_prob.old_values,
        returns=bad_log_prob.returns,
        advantages=bad_log_prob.advantages,
        valid_mask=bad_log_prob.valid_mask,
    )
    try:
        compute_frontres_segment_ppo_loss(policy, bad_log_prob)
    except ValueError as exc:
        assert "old_log_probs must have shape [B]" in str(exc)
    else:
        raise AssertionError("non-vector old log-prob should be rejected")


def test_all_invalid_batch_has_zero_loss_and_no_step() -> None:
    policy = FakeSegmentPolicy()
    batch = _batch()
    batch = FrontRESSegmentPPOBatch(
        observations=batch.observations,
        actions=batch.actions,
        old_log_probs=batch.old_log_probs,
        old_values=batch.old_values,
        returns=batch.returns,
        advantages=batch.advantages,
        valid_mask=torch.tensor([False, False]),
    )
    result = compute_frontres_segment_ppo_loss(policy, batch)
    assert not result.should_step
    assert result.valid_count == 0
    assert result.total_loss.item() == 0.0
    result.total_loss.backward()
    assert policy.actor.weight.grad is not None
    assert torch.count_nonzero(policy.actor.weight.grad) == 0


def main() -> None:
    test_fake_batch_updates_actor_on_valid_segments()
    test_invalid_samples_do_not_contribute_to_loss()
    test_nonfinite_valid_rows_are_masked_before_loss()
    test_extreme_log_ratio_does_not_overflow_loss()
    test_ppo_tuple_requires_6d_action_and_vector_fields()
    test_all_invalid_batch_has_zero_loss_and_no_step()
    print("result: PASS")


if __name__ == "__main__":
    main()
