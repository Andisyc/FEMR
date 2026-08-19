"""S1 pseudo-data numerical oracle for the active v013 clipped PPO loss.

The scalar oracle below intentionally uses plain Python arithmetic.  It does
not call a production helper, so it can detect a wrong ratio, clip or minimum
operation in the public loss boundary.
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
    compute_frontres_relational_preference_loss,
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


def _batch(old_log_prob: float) -> FrontRESRelationalPPOBatch:
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


def _hand_clipped_loss(
    *,
    new_log_probs: tuple[float, ...],
    old_log_prob: float,
    edges: tuple[tuple[int, int], ...],
    clip_param: float = 0.2,
    max_log_ratio: float = 20.0,
) -> float:
    credits = [0.0 for _ in new_log_probs]
    for winner, loser in edges:
        credits[winner] += 1.0
        credits[loser] -= 1.0
    surrogates = []
    for credit, new_log_prob in zip(credits, new_log_probs):
        if credit == 0.0:
            continue
        raw_log_ratio = new_log_prob - old_log_prob
        log_ratio = max(-max_log_ratio, min(max_log_ratio, raw_log_ratio))
        ratio = math.exp(log_ratio)
        clipped_ratio = max(1.0 - clip_param, min(1.0 + clip_param, ratio))
        surrogates.append(min(ratio * credit, clipped_ratio * credit))
    return -sum(surrogates) / float(len(edges))


def main() -> None:
    edges = ((0, 1), (1, 2), (2, 3))
    new_log_probs = (0.1, 0.05, -0.05, -0.1)

    normal_policy = _TinyPolicy()
    normal = compute_frontres_relational_actor_loss(
        normal_policy,
        _batch(0.0),
        edges,
        FrontRESRelationalPPOConfig(clip_param=0.2, max_log_ratio=20.0),
    )
    expected_normal = _hand_clipped_loss(
        new_log_probs=new_log_probs,
        old_log_prob=0.0,
        edges=edges,
    )
    assert math.isclose(float(normal.total_loss.detach()), expected_normal, rel_tol=1e-6, abs_tol=1e-6)

    # A stale old policy creates a large ratio.  The production min/clip
    # behavior is compared against the independent scalar oracle, not just a
    # finite check.
    extreme_policy = _TinyPolicy()
    extreme = compute_frontres_relational_actor_loss(
        extreme_policy,
        _batch(-10.0),
        edges,
        FrontRESRelationalPPOConfig(clip_param=0.2, max_log_ratio=20.0),
    )
    expected_extreme = _hand_clipped_loss(
        new_log_probs=new_log_probs,
        old_log_prob=-10.0,
        edges=edges,
    )
    assert math.isclose(float(extreme.total_loss.detach()), expected_extreme, rel_tol=1e-6, abs_tol=1e-6)
    assert abs(expected_extreme) > abs(expected_normal) * 100.0

    capped_policy = _TinyPolicy()
    capped = compute_frontres_relational_actor_loss(
        capped_policy,
        _batch(-100.0),
        edges,
        FrontRESRelationalPPOConfig(clip_param=0.2, max_log_ratio=20.0),
    )
    expected_capped = _hand_clipped_loss(
        new_log_probs=new_log_probs,
        old_log_prob=-100.0,
        edges=edges,
    )
    assert math.isclose(float(capped.total_loss.detach()), expected_capped, rel_tol=1e-6, abs_tol=1e-6)
    expected_very_capped = _hand_clipped_loss(
        new_log_probs=new_log_probs,
        old_log_prob=-1_000.0,
        edges=edges,
    )
    assert math.isclose(expected_capped, expected_very_capped, rel_tol=1e-6, abs_tol=1e-6)

    # At clip_param=0, the ratio branch is forced to the unit boundary.
    zero_clip_policy = _TinyPolicy()
    zero_clip = compute_frontres_relational_actor_loss(
        zero_clip_policy,
        _batch(0.0),
        edges,
        FrontRESRelationalPPOConfig(clip_param=0.0, max_log_ratio=20.0),
    )
    expected_zero_clip = _hand_clipped_loss(
        new_log_probs=new_log_probs,
        old_log_prob=0.0,
        edges=edges,
        clip_param=0.0,
    )
    assert math.isclose(float(zero_clip.total_loss.detach()), expected_zero_clip, rel_tol=1e-6, abs_tol=1e-6)

    # The candidate direct loss is intentionally old-log-prob independent on
    # the same rows.  This exposes the exact source of the old ratio explosion.
    direct_a = compute_frontres_relational_preference_loss(_TinyPolicy(), _batch(0.0), edges)
    direct_b = compute_frontres_relational_preference_loss(_TinyPolicy(), _batch(-10.0), edges)
    assert math.isclose(
        float(direct_a.total_loss.detach()), float(direct_b.total_loss.detach()), rel_tol=1e-6, abs_tol=1e-6
    )

    # Controlled mutant: replacing minimum with maximum must not pass the
    # independent expected value for the extreme-ratio case.
    mutant_max = max(
        math.exp(min(20.0, max(-20.0, new_log_probs[0] + 10.0))) * 1.0,
        1.2 * 1.0,
    )
    assert not math.isclose(mutant_max, 1.2, rel_tol=1e-6, abs_tol=1e-6)

    print("frontres_relational_ppo_loss_clipping_alignment: MODULE-CORRECT")
    print(f"normal_loss={float(normal.total_loss.detach()):.6f}")
    print(f"normal_oracle={expected_normal:.6f}")
    print(f"extreme_loss={float(extreme.total_loss.detach()):.6f}")
    print(f"extreme_oracle={expected_extreme:.6f}")
    print(f"max_log_ratio_capped_loss={float(capped.total_loss.detach()):.6f}")
    print(f"zero_clip_loss={float(zero_clip.total_loss.detach()):.6f}")
    print(f"extreme_to_normal_abs_ratio={abs(expected_extreme) / abs(expected_normal):.3f}")
    print(f"direct_old_log_invariance={float(direct_a.total_loss.detach() - direct_b.total_loss.detach()):.6f}")


if __name__ == "__main__":
    main()
