#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


storage_module = _load("frontres_segment_storage", ROOT / "rsl_rl" / "frontres" / "frontres_segment_storage.py")
ppo_module = _load("frontres_segment_ppo", ROOT / "rsl_rl" / "algorithms" / "frontres_segment_ppo.py")

FrontRESSegmentRolloutStorage = storage_module.FrontRESSegmentRolloutStorage
FrontRESSegmentTransition = storage_module.FrontRESSegmentTransition
FrontRESSegmentPPOBatch = ppo_module.FrontRESSegmentPPOBatch
compute_frontres_segment_ppo_loss = ppo_module.compute_frontres_segment_ppo_loss


class FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(4, 6, bias=False)
        self.critic = torch.nn.Linear(4, 1, bias=False)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.critic.weight)

    def evaluate_segment_actions(self, observations, actions):
        mean = self.actor(observations)
        value = self.critic(observations).squeeze(-1)
        return {
            "log_prob": -0.5 * (actions - mean).square().sum(dim=-1),
            "value": value,
            "entropy": torch.ones(actions.shape[0]) * 0.5,
            "mean": mean,
        }


@dataclass
class FakeEvidence:
    segment_ids: torch.Tensor
    priority: torch.Tensor


def _transition() -> FrontRESSegmentTransition:
    return FrontRESSegmentTransition(
        observations=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        actions=torch.tensor(
            [
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            ]
        ),
        old_log_probs=torch.zeros(3),
        values=torch.tensor([0.1, 0.2, 10.0]),
        rewards=torch.tensor([1.0, 0.5, 100.0]),
        valid_mask=torch.tensor([True, True, False]),
        reset_mask=torch.tensor([True, False, True]),
        segment_ids=torch.tensor([4, 5, 6]),
        segment_source=("global", "replay", "review"),
        old_means=torch.zeros(3, 6),
        old_sigmas=torch.ones(3, 6),
        action_mask=torch.ones(3, 6),
        priority_evidence=FakeEvidence(segment_ids=torch.tensor([4, 5, 6]), priority=torch.tensor([0.1, 0.2, 0.3])),
    )


def test_segment_storage_writes_clean_6d_ppo_tuple() -> None:
    storage = FrontRESSegmentRolloutStorage(capacity=4, obs_shape=(4,))
    storage.add_transition(_transition())
    batch = storage.full_batch()
    assert batch.actions.shape == (3, 6)
    assert batch.old_log_probs.shape == (3,)
    assert batch.old_values.shape == (3,)
    assert batch.returns.shape == (3,)
    assert batch.advantages.shape == (3,)
    assert batch.segment_ids.tolist() == [4, 5, 6]
    assert batch.valid_mask.tolist() == [True, False, False]
    torch.testing.assert_close(batch.returns, torch.tensor([1.0, 0.5, 100.0]))
    torch.testing.assert_close(batch.advantages, torch.tensor([0.9, 0.3, 90.0]))


def test_segment_storage_converts_to_algorithm_batch_and_masks_invalid_samples() -> None:
    storage = FrontRESSegmentRolloutStorage(capacity=4, obs_shape=(4,))
    storage.add_transition(_transition())
    ppo_batch = storage.full_batch().to_ppo_batch(FrontRESSegmentPPOBatch)
    policy = FakePolicy()
    result = compute_frontres_segment_ppo_loss(policy, ppo_batch)
    assert result.valid_count == 1
    result.total_loss.backward()
    assert policy.actor.weight.grad is not None
    assert torch.count_nonzero(policy.actor.weight.grad[:, 0]) > 0
    assert torch.count_nonzero(policy.actor.weight.grad[:, 1:]) == 0


def test_segment_storage_minibatches_and_round_trip_state() -> None:
    storage = FrontRESSegmentRolloutStorage(capacity=4, obs_shape=(4,))
    storage.add_transition(_transition())
    batches = list(storage.mini_batch_generator(num_mini_batches=2, num_epochs=1, shuffle=False))
    assert len(batches) == 2
    assert batches[0].actions.shape == (2, 6)
    state = storage.state_dict()
    restored = FrontRESSegmentRolloutStorage(capacity=4, obs_shape=(4,))
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.full_batch().actions, storage.full_batch().actions)
    assert restored.segment_source == ["global", "replay", "review"]
    torch.testing.assert_close(restored.priority_evidence[0]["priority"], torch.tensor([0.1, 0.2, 0.3]))


def test_segment_storage_rejects_non_6d_actions_and_overflow() -> None:
    storage = FrontRESSegmentRolloutStorage(capacity=3, obs_shape=(4,))
    bad = _transition()
    bad = FrontRESSegmentTransition(
        observations=bad.observations,
        actions=torch.zeros(3, 5),
        old_log_probs=bad.old_log_probs,
        values=bad.values,
        rewards=bad.rewards,
        valid_mask=bad.valid_mask,
        reset_mask=bad.reset_mask,
        segment_ids=bad.segment_ids,
    )
    try:
        storage.add_transition(bad)
    except ValueError as exc:
        assert "actions must have shape [B, 6]" in str(exc)
    else:
        raise AssertionError("5D segment action should be rejected")
    storage.add_transition(_transition())
    try:
        storage.add_transition(_transition())
    except OverflowError:
        pass
    else:
        raise AssertionError("storage overflow should be explicit")


def test_connector_writer_requires_policy_log_prob_and_value() -> None:
    storage = FrontRESSegmentRolloutStorage(capacity=2, obs_shape=(4,))
    payload = {
        "sample": SimpleNamespace(source=("global", "replay")),
        "batch": SimpleNamespace(segment_ids=torch.tensor([0, 1])),
        "repair_action": SimpleNamespace(projected_delta_se=torch.zeros(2, 6), active_mask=torch.ones(2, 6)),
        "reward_result": SimpleNamespace(
            reward=torch.ones(2),
            valid_mask=torch.tensor([True, True]),
        ),
        "reset_result": SimpleNamespace(success_mask=torch.tensor([True, True])),
        "raw_action": torch.zeros(2, 6),
    }
    try:
        storage.write(**payload)
    except ValueError as exc:
        assert "policy observations" in str(exc)
    else:
        raise AssertionError("connector payload without policy PPO fields should be rejected")


def main() -> None:
    test_segment_storage_writes_clean_6d_ppo_tuple()
    test_segment_storage_converts_to_algorithm_batch_and_masks_invalid_samples()
    test_segment_storage_minibatches_and_round_trip_state()
    test_segment_storage_rejects_non_6d_actions_and_overflow()
    test_connector_writer_requires_policy_log_prob_and_value()
    print("result: PASS")


if __name__ == "__main__":
    main()
