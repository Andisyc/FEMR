"""Offline contract tests for the relational, Actor-only PPO adapter.

The tests use a tiny differentiable policy and a hand-authored preference edge
set.  They do not instantiate the FrontRES runner or touch the active scalar
Gain path.
"""

import torch

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESRelationalPPOBatch,
    FrontRESRelationalPPOConfig,
    compute_frontres_relational_actor_loss,
)


class _Policy:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        log_prob = (self.weight * actions[:, 0]).reshape(-1)
        return {
            "log_prob": log_prob,
            "value": torch.zeros_like(log_prob),
            "entropy": torch.zeros_like(log_prob),
        }


def _batch() -> FrontRESRelationalPPOBatch:
    rows = 4
    return FrontRESRelationalPPOBatch(
        observations=torch.zeros(rows, 2),
        actions=torch.tensor([[1.0, 0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0],
                              [-0.5, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0]]),
        old_log_probs=torch.zeros(rows),
        valid_mask=torch.ones(rows, dtype=torch.bool),
    )


def main() -> None:
    policy = _Policy()
    result = compute_frontres_relational_actor_loss(
        policy,
        _batch(),
        ((0, 1), (1, 2), (2, 3)),
        FrontRESRelationalPPOConfig(),
    )
    assert result.status == "READY"
    assert result.edge_count == 3
    assert result.actor_credit.tolist() == [1.0, 0.0, 0.0, -1.0]
    assert result.should_step
    result.total_loss.backward()
    assert policy.weight.grad is not None
    assert torch.isfinite(policy.weight.grad)

    order = torch.tensor([3, 2, 1, 0])
    base = _batch()
    permuted_batch = FrontRESRelationalPPOBatch(
        **{
            **base.__dict__,
            "observations": base.observations[order],
            "actions": base.actions[order],
            "old_log_probs": base.old_log_probs[order],
            "valid_mask": base.valid_mask[order],
        }
    )
    permuted = compute_frontres_relational_actor_loss(
        policy,
        permuted_batch,
        ((3, 2), (2, 1), (1, 0)),
        FrontRESRelationalPPOConfig(),
    )
    assert permuted.actor_credit.tolist() == list(reversed(result.actor_credit.tolist()))

    no_edges = compute_frontres_relational_actor_loss(policy, _batch(), (), FrontRESRelationalPPOConfig())
    assert no_edges.status == "NO_COMPARABLE_PAIRS"
    assert not no_edges.should_step
    assert no_edges.actor_credit.tolist() == [0.0] * 4

    invalid = _batch()
    invalid = FrontRESRelationalPPOBatch(**{**invalid.__dict__, "valid_mask": torch.tensor([True, True, True, False])})
    try:
        compute_frontres_relational_actor_loss(policy, invalid, ((0, 3),), FrontRESRelationalPPOConfig())
    except ValueError:
        pass
    else:
        raise AssertionError("invalid preference endpoints must fail closed")

    try:
        compute_frontres_relational_actor_loss(
            policy,
            _batch(),
            ((0, 1), (1, 2), (2, 0)),
            FrontRESRelationalPPOConfig(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("cyclic preference edges must fail closed")

    print("frontres_relational_ppo_alignment: OBJECTIVE-ALIGNED")


if __name__ == "__main__":
    main()
