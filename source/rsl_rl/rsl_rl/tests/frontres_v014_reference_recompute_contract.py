"""Regression for v014 reference-local Fisher evaluation at formal commit."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.distributions import Normal

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESRelationalPPOBatch,
    compute_frontres_relational_reference_fisher_loss,
)
from rsl_rl.frontres.frontres_policy_evaluation import FrontRESSegmentLivePolicyAdapter


class _RecomputeDriftPolicy:
    """Represent the same Actor snapshot with bounded forward-roundoff drift."""

    def __init__(self, reference_mean: torch.Tensor, *, drift: float) -> None:
        self.mean = torch.nn.Parameter(reference_mean.detach().clone())
        self.sigma = torch.ones_like(reference_mean)
        self.drift = float(drift)

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        del observations
        mean = self.mean + self.drift
        distribution = Normal(mean, self.sigma)
        return {
            "log_prob": distribution.log_prob(actions).sum(dim=-1),
            "value": torch.zeros(actions.shape[0], dtype=actions.dtype),
            "entropy": distribution.entropy().sum(dim=-1),
            "mean": mean,
            "sigma": self.sigma,
        }


class _NoSamplePolicy:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean
        self.std = torch.full_like(mean, 0.5)
        self.distribution: Normal | None = None
        self.act_calls = 0
        self.update_calls = 0

    def act(self, observations: torch.Tensor):
        del observations
        self.act_calls += 1
        raise AssertionError("formal Loss evaluation must not sample through policy.act")

    def update_distribution(self, observations: torch.Tensor) -> None:
        self.update_calls += 1
        self.distribution = Normal(self.mean.expand(observations.shape[0], -1), self.std.expand(observations.shape[0], -1))

    @property
    def action_mean(self) -> torch.Tensor | None:
        return None if self.distribution is None else self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor | None:
        return None if self.distribution is None else self.distribution.stddev


class _SamplingOnlyPolicy:
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.zeros(observations.shape[0], 6)


def _batch() -> FrontRESRelationalPPOBatch:
    observations = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    actions = torch.tensor(
        [[0.4, -0.1, 0.2, 0.0, 0.3, -0.2], [-0.2, 0.3, -0.1, 0.2, -0.4, 0.1]],
        dtype=torch.float32,
    )
    old_means = torch.zeros_like(actions)
    old_sigmas = torch.ones_like(actions)
    old_log_probs = Normal(old_means, old_sigmas).log_prob(actions).sum(dim=-1)
    return FrontRESRelationalPPOBatch(
        observations=observations,
        actions=actions,
        old_log_probs=old_log_probs,
        valid_mask=torch.ones(2, dtype=torch.bool),
        segment_ids=torch.tensor([0, 0], dtype=torch.long),
        old_means=old_means,
        old_sigmas=old_sigmas,
        transaction_metadata=SimpleNamespace(scenario_ids=("scenario-a", "scenario-a")),
    )


def test_same_snapshot_recompute_drift_keeps_reference_local_gradient() -> None:
    batch = _batch()
    exact = _RecomputeDriftPolicy(batch.old_means, drift=0.0)
    drifted = _RecomputeDriftPolicy(batch.old_means, drift=2.0e-6)

    exact_result = compute_frontres_relational_reference_fisher_loss(exact, batch, ((0, 1),))
    drifted_result = compute_frontres_relational_reference_fisher_loss(drifted, batch, ((0, 1),))
    exact_result.total_loss.backward()
    drifted_result.total_loss.backward()

    assert exact_result.status == drifted_result.status == "READY"
    torch.testing.assert_close(exact_result.total_loss, drifted_result.total_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(exact.mean.grad, drifted.mean.grad, rtol=1.0e-6, atol=1.0e-7)
    assert bool(torch.isfinite(drifted.mean.grad).all())
    assert bool(torch.any(drifted.mean.grad != 0.0))


def test_formal_policy_evaluation_recomputes_distribution_without_sampling() -> None:
    policy = _NoSamplePolicy(torch.zeros(1, 6))
    adapter = FrontRESSegmentLivePolicyAdapter(
        SimpleNamespace(policy=policy),
        privileged_observations=None,
        actor_only=True,
    )
    observations = torch.ones(2, 3)
    actions = torch.zeros(2, 6)
    rng_before = torch.random.get_rng_state().clone()

    result = adapter.evaluate_segment_actions(observations, actions)

    assert policy.act_calls == 0
    assert policy.update_calls == 1
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert tuple(result["mean"].shape) == (2, 6)


def test_formal_policy_evaluation_rejects_sampling_only_policy() -> None:
    adapter = FrontRESSegmentLivePolicyAdapter(
        SimpleNamespace(policy=_SamplingOnlyPolicy()),
        privileged_observations=None,
        actor_only=True,
    )
    try:
        adapter.evaluate_segment_actions(torch.ones(2, 3), torch.zeros(2, 6))
    except TypeError as exc:
        assert "update_distribution" in str(exc)
    else:
        raise AssertionError("formal Loss evaluator accepted a sampling-only policy")


def main() -> None:
    test_same_snapshot_recompute_drift_keeps_reference_local_gradient()
    test_formal_policy_evaluation_recomputes_distribution_without_sampling()
    test_formal_policy_evaluation_rejects_sampling_only_policy()
    print("frontres_v014_reference_recompute_contract: PASS")


if __name__ == "__main__":
    main()
