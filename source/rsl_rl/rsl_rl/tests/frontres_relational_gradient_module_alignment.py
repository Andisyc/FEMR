"""S1 pseudo-samples for the relational Actor loss and gradient boundary.

This test uses the production relational PPO public function with a tiny
differentiable policy. It does not instantiate Isaac, the runner, or the
active Gain path.
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
    FrontRESRelationalPPOConfig,
    compute_frontres_relational_actor_loss,
)


class _TinyPolicy:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        log_prob = self.weight * observations[:, 0] * actions[:, 0]
        return {
            "log_prob": log_prob,
            "value": torch.zeros_like(log_prob),
            "entropy": torch.zeros_like(log_prob),
        }


def _batch(*, observation_scale: float = 1.0, old_log_prob: float = 0.0) -> FrontRESRelationalPPOBatch:
    actions = torch.tensor(
        [[1.0, 0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0], [-0.5, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0]]
    )
    return FrontRESRelationalPPOBatch(
        observations=torch.ones(4, 2) * observation_scale,
        actions=actions,
        old_log_probs=torch.full((4,), old_log_prob),
        valid_mask=torch.ones(4, dtype=torch.bool),
    )


def _gradient(*, observation_scale: float = 1.0, old_log_prob: float = 0.0, edges=((0, 1), (1, 2), (2, 3))):
    policy = _TinyPolicy()
    result = compute_frontres_relational_actor_loss(
        policy,
        _batch(observation_scale=observation_scale, old_log_prob=old_log_prob),
        edges,
        FrontRESRelationalPPOConfig(),
    )
    result.total_loss.backward()
    gradient = float(policy.weight.grad.detach().abs().item())
    return result, gradient


def main() -> None:
    baseline, baseline_gradient = _gradient()
    assert baseline.actor_credit.tolist() == [1.0, 0.0, 0.0, -1.0]
    assert math.isfinite(baseline_gradient) and baseline_gradient > 0.0

    # Independent credit oracle: a star has degree three and must change the
    # loss gradient relative to the chain's degree-one endpoints.
    _, star_gradient = _gradient(edges=((0, 1), (0, 2), (0, 3)))
    assert star_gradient > baseline_gradient

    # In the unclipped local region, doubling the feature scale doubles the
    # analytic score derivative up to the small ratio curvature.
    _, small_gradient = _gradient(observation_scale=0.01)
    _, double_gradient = _gradient(observation_scale=0.02)
    assert 1.8 < double_gradient / small_gradient < 2.2

    # Extreme old/new log-prob mismatch is a sensitivity counterexample for
    # the production ratio clamp/surrogate boundary.
    _, extreme_gradient = _gradient(old_log_prob=-10.0)
    assert extreme_gradient > baseline_gradient * 100.0

    # Row permutation with correspondingly permuted edges preserves the local
    # semantic gradient; changing only the edge graph must not be invariant.
    _, permuted_gradient = _gradient(edges=((3, 2), (2, 1), (1, 0)))
    assert math.isclose(permuted_gradient, baseline_gradient, rel_tol=1e-6)
    assert not math.isclose(star_gradient, baseline_gradient, rel_tol=1e-6)

    # The runtime boundary clips only the resulting global gradient norm.
    assert baseline_gradient > 0.5
    clip_coefficient = 0.5 / baseline_gradient
    assert 0.0 < clip_coefficient < 1.0
    assert math.isclose(baseline_gradient * clip_coefficient, 0.5, rel_tol=1e-6)

    print("frontres_relational_gradient_module_alignment: MODULE-CORRECT")
    print(f"baseline_gradient={baseline_gradient:.6f}")
    print(f"star_credit_gradient={star_gradient:.6f}")
    print(f"scale_ratio={double_gradient / small_gradient:.6f}")
    print(f"extreme_ratio_gradient={extreme_gradient:.6f}")
    print(f"norm_clip_coefficient={clip_coefficient:.6f}")


if __name__ == "__main__":
    main()
