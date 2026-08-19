"""S1 pseudo-data contract for the FRS-PPO-v014 preference loss candidate.

This test is intentionally independent of the active v013 transaction.  It
checks only the v014 logistic pairwise preference objective.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESRelationalPPOBatch,
    compute_frontres_relational_preference_loss,
)


class _TinyPreferencePolicy:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.log_std = torch.nn.Parameter(torch.tensor(-0.1))

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        mean = self.weight * observations[:, 0]
        scale = torch.exp(self.log_std)
        log_prob = -0.5 * ((actions[:, 0] - mean) / scale).square() - self.log_std
        return {"log_prob": log_prob, "value": torch.zeros_like(log_prob), "entropy": None}


class _NonFiniteUnreferencedPolicy(_TinyPreferencePolicy):
    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        output = super().evaluate_segment_actions(observations, actions)
        log_prob = output["log_prob"].clone()
        log_prob[-1] = float("nan")
        output["log_prob"] = log_prob
        return output


def _batch(*, old_log_prob: float = 0.0) -> FrontRESRelationalPPOBatch:
    actions = torch.tensor(
        [
            [1.0, 0, 0, 0, 0, 0],
            [0.5, 0, 0, 0, 0, 0],
            [-0.5, 0, 0, 0, 0, 0],
            [-1.0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    return FrontRESRelationalPPOBatch(
        observations=torch.ones(4, 2),
        actions=actions,
        old_log_probs=torch.full((4,), old_log_prob),
        valid_mask=torch.ones(4, dtype=torch.bool),
    )


def _run(*, old_log_prob: float = 0.0, edges=((0, 1), (2, 3))):
    policy = _TinyPreferencePolicy()
    result = compute_frontres_relational_preference_loss(policy, _batch(old_log_prob=old_log_prob), edges)
    result.total_loss.backward()
    return result, policy


def _permuted_batch() -> FrontRESRelationalPPOBatch:
    batch = _batch()
    order = torch.tensor([2, 3, 0, 1])
    return FrontRESRelationalPPOBatch(
        observations=batch.observations.index_select(0, order),
        actions=batch.actions.index_select(0, order),
        old_log_probs=batch.old_log_probs.index_select(0, order),
        valid_mask=batch.valid_mask.index_select(0, order),
    )


def _independent_log_prob(policy: _TinyPreferencePolicy, batch: FrontRESRelationalPPOBatch) -> list[float]:
    """Hand oracle for the tiny Gaussian fixture; does not call production code."""

    weight = float(policy.weight.detach())
    log_std = float(policy.log_std.detach())
    scale = math.exp(log_std)
    return [
        -0.5 * ((float(action[0]) - weight * float(observation[0])) / scale) ** 2 - log_std
        for observation, action in zip(batch.observations, batch.actions)
    ]


def main() -> None:
    result, policy = _run()
    assert result.status == "READY"
    assert result.contract_id == "FRS-PPO-v014"
    assert result.edge_count == 2
    assert result.actor_credit.tolist() == [1.0, -1.0, 1.0, -1.0]
    assert math.isfinite(float(result.total_loss.detach()))
    assert policy.weight.grad is not None and math.isfinite(float(policy.weight.grad))
    assert policy.log_std.grad is not None and math.isfinite(float(policy.log_std.grad))
    assert policy.weight.grad.item() < 0.0
    fixture_log_probs = _independent_log_prob(policy, _batch())
    expected_loss = 0.5 * sum(
        math.log1p(math.exp(-(fixture_log_probs[winner] - fixture_log_probs[loser])))
        for winner, loser in ((0, 1), (2, 3))
    )
    assert math.isclose(float(result.total_loss.detach()), expected_loss, rel_tol=1e-6, abs_tol=1e-6)
    wrong_sign_loss = 0.5 * sum(
        math.log1p(math.exp(-(fixture_log_probs[loser] - fixture_log_probs[winner])))
        for winner, loser in ((0, 1), (2, 3))
    )
    assert not math.isclose(float(result.total_loss.detach()), wrong_sign_loss, rel_tol=1e-6, abs_tol=1e-6)

    # A malformed row outside the selected edges still invalidates the whole
    # sealed policy evaluation; no partial evidence may update the Actor.
    try:
        compute_frontres_relational_preference_loss(_NonFiniteUnreferencedPolicy(), _batch(), ((0, 1),))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite unreferenced policy row must fail closed")

    # The candidate is a direct preference objective: changing stale PPO
    # log-probabilities must not change its loss or gradient.
    changed, changed_policy = _run(old_log_prob=-1_000_000.0)
    assert math.isclose(
        float(changed.total_loss.detach()), float(result.total_loss.detach()), rel_tol=1e-6, abs_tol=1e-6
    )
    assert math.isclose(
        float(changed_policy.weight.grad), float(policy.weight.grad), rel_tol=1e-6, abs_tol=1e-6
    )

    # Row order is not semantic when the corresponding edge indices move with
    # the rows.
    permuted_policy = _TinyPreferencePolicy()
    permuted = compute_frontres_relational_preference_loss(
        permuted_policy,
        _permuted_batch(),
        ((2, 3), (0, 1)),
    )
    assert math.isclose(
        float(permuted.total_loss.detach()), float(result.total_loss.detach()), rel_tol=1e-6, abs_tol=1e-6
    )

    # Reversing one directed relation reverses only that relation's credit.
    reversed_result, _ = _run(edges=((1, 0), (2, 3)))
    assert reversed_result.actor_credit.tolist() == [-1.0, 1.0, 1.0, -1.0]

    # SAME/INCOMPARABLE are represented by no edge and must be zero-write.
    no_edge, no_edge_policy = _run(edges=())
    assert no_edge.status == "NO_COMPARABLE_PAIRS"
    assert no_edge.edge_count == 0
    assert no_edge_policy.weight.grad is None or float(no_edge_policy.weight.grad) == 0.0

    # Invalid evidence must not silently become a training direction.
    invalid_edges = (
        ((0, 0),),
        ((0, 1), (0, 1)),
        ((0, 1), (1, 0)),
        ((0, 4),),
    )
    for edges in invalid_edges:
        try:
            compute_frontres_relational_preference_loss(_TinyPreferencePolicy(), _batch(), edges)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid preference edges must fail closed: {edges!r}")

    invalid_rows = _batch()
    invalid_rows.valid_mask[1] = False
    try:
        compute_frontres_relational_preference_loss(_TinyPreferencePolicy(), invalid_rows, ((0, 1),))
    except ValueError:
        pass
    else:
        raise AssertionError("preference edges cannot reference invalid rows")

    # A small fixed-edge pseudo transaction exercises repeated optimizer
    # updates and distribution-state finiteness without changing production
    # gradient clipping or optimizer policy.
    rollout_policy = _TinyPreferencePolicy()
    optimizer = torch.optim.Adam([rollout_policy.weight, rollout_policy.log_std], lr=0.01)
    losses: list[float] = []
    for _ in range(8):
        optimizer.zero_grad()
        step = compute_frontres_relational_preference_loss(rollout_policy, _batch(), ((0, 1), (2, 3)))
        step.total_loss.backward()
        optimizer.step()
        losses.append(float(step.total_loss.detach()))
    assert all(math.isfinite(value) for value in losses)
    assert math.isfinite(float(rollout_policy.log_std.detach()))
    assert abs(float(rollout_policy.log_std.detach())) < 5.0

    print("frontres_relational_preference_loss_alignment: MODULE-CORRECT")
    print(f"loss={float(result.total_loss.detach()):.6f}")
    print(f"weight_gradient={float(policy.weight.grad):.6f}")
    print(f"log_std_gradient={float(policy.log_std.grad):.6f}")


if __name__ == "__main__":
    main()
