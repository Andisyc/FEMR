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
        return FrontRESSegmentPolicyEval(log_prob=log_prob, value=value, entropy=entropy, mean=mean, sigma=torch.ones_like(mean))

    def acceptance_loss(self, *args, **kwargs):
        self.acceptance_called = True
        raise AssertionError("old acceptance path must not be used by segment PPO")


class StaticEvalPolicy:
    def __init__(
        self,
        *,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        mean: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
        raw_actions: torch.Tensor | None = None,
        log_jacobian_contrib: torch.Tensor | None = None,
    ) -> None:
        self.log_prob = log_prob
        self.value = value
        self.mean = mean
        self.sigma = sigma
        self.entropy = entropy
        self.raw_actions = raw_actions
        self.log_jacobian_contrib = log_jacobian_contrib

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> FrontRESSegmentPolicyEval:
        return FrontRESSegmentPolicyEval(
            log_prob=self.log_prob,
            value=self.value,
            entropy=self.entropy,
            mean=self.mean,
            sigma=self.sigma,
            raw_actions=self.raw_actions,
            log_jacobian_contrib=self.log_jacobian_contrib,
        )


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


def _permute_batch(batch: FrontRESSegmentPPOBatch, order: torch.Tensor) -> FrontRESSegmentPPOBatch:
    return FrontRESSegmentPPOBatch(
        observations=batch.observations[order],
        actions=batch.actions[order],
        old_log_probs=batch.old_log_probs[order],
        old_values=batch.old_values[order],
        returns=batch.returns[order],
        advantages=batch.advantages[order],
        valid_mask=batch.valid_mask[order],
        segment_ids=batch.segment_ids[order] if batch.segment_ids is not None else None,
        action_mask=batch.action_mask[order] if batch.action_mask is not None else None,
        old_means=batch.old_means[order] if batch.old_means is not None else None,
        old_sigmas=batch.old_sigmas[order] if batch.old_sigmas is not None else None,
    )


def _full_delta_repair_batch(action_mask: torch.Tensor) -> FrontRESSegmentPPOBatch:
    action = torch.tensor([[0.20, -0.15, 0.10, 0.30, -0.25, 0.05]])
    return FrontRESSegmentPPOBatch(
        observations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        actions=action,
        old_log_probs=-0.5 * action.square().sum(dim=-1),
        old_values=torch.zeros(1),
        returns=torch.ones(1),
        advantages=torch.ones(1),
        valid_mask=torch.tensor([True]),
        segment_ids=torch.tensor([61]),
        action_mask=action_mask,
        old_means=torch.zeros(1, 6),
        old_sigmas=torch.ones(1, 6),
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


def test_positive_advantage_moves_mean_toward_stored_delta_se_action() -> None:
    policy = FakeSegmentPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    obs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    large_rp_action = torch.tensor([[0.0, 0.0, 0.0, 0.4, 0.4, 0.0]])
    old_log_prob = -0.5 * large_rp_action.square().sum(dim=-1)
    batch = FrontRESSegmentPPOBatch(
        observations=obs,
        actions=large_rp_action,
        old_log_probs=old_log_prob,
        old_values=torch.zeros(1),
        returns=torch.ones(1),
        advantages=torch.ones(1),
        valid_mask=torch.tensor([True]),
        segment_ids=torch.tensor([11]),
        action_mask=torch.ones(1, 6),
    )

    before_mean = policy.evaluate_segment_actions(obs, large_rp_action).mean.detach().clone()
    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    optimizer.zero_grad(set_to_none=True)
    result.total_loss.backward()
    optimizer.step()
    after_mean = policy.evaluate_segment_actions(obs, large_rp_action).mean.detach()

    print(
        "[probe ppo_advantage_large_action] "
        f"before_rp={before_mean[0, 3:5].tolist()} "
        f"after_rp={after_mean[0, 3:5].tolist()} "
        f"stored_action_rp={large_rp_action[0, 3:5].tolist()} "
        f"advantage={batch.advantages.item():.6f}",
        flush=True,
    )
    assert result.should_step
    assert after_mean[0, 3] > before_mean[0, 3]
    assert after_mean[0, 4] > before_mean[0, 4]
    assert after_mean[0, 3] < large_rp_action[0, 3]
    assert after_mean[0, 4] < large_rp_action[0, 4]


def test_old_distribution_stats_drive_mosaic_style_kl() -> None:
    policy = FakeSegmentPolicy()
    obs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    actions = torch.zeros(1, 6)
    batch = FrontRESSegmentPPOBatch(
        observations=obs,
        actions=actions,
        old_log_probs=torch.zeros(1),
        old_values=torch.zeros(1),
        returns=torch.ones(1),
        advantages=torch.ones(1),
        valid_mask=torch.tensor([True]),
        segment_ids=torch.tensor([12]),
        action_mask=torch.ones(1, 6),
        old_means=torch.full((1, 6), 0.5),
        old_sigmas=torch.ones(1, 6),
    )

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    print(
        "[probe old_stats_kl] "
        f"logprob_approx_kl={result.logprob_approx_kl:.6f} "
        f"distribution_kl_mean={result.distribution_kl_mean:.6f} "
        f"distribution_kl_available={result.distribution_kl_available} "
        f"approx_kl={result.approx_kl:.6f}",
        flush=True,
    )
    assert result.distribution_kl_available
    assert abs(result.logprob_approx_kl) < 1e-6
    assert result.distribution_kl_mean > 0.7
    assert abs(result.approx_kl - result.distribution_kl_mean) < 1e-6


def test_distribution_kl_matches_old_new_stats_exactly() -> None:
    obs = torch.zeros(2, 4)
    actions = torch.zeros(2, 6)
    old_means = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    old_sigmas = torch.tensor(
        [
            [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    new_means = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.0, 0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    new_sigmas = torch.ones(2, 6)
    policy = StaticEvalPolicy(
        log_prob=torch.zeros(2, requires_grad=True),
        value=torch.zeros(2, requires_grad=True),
        mean=new_means,
        sigma=new_sigmas,
        entropy=torch.zeros(2),
    )
    batch = FrontRESSegmentPPOBatch(
        observations=obs,
        actions=actions,
        old_log_probs=torch.zeros(2),
        old_values=torch.zeros(2),
        returns=torch.zeros(2),
        advantages=torch.ones(2),
        valid_mask=torch.tensor([True, True]),
        segment_ids=torch.tensor([21, 22]),
        action_mask=torch.ones(2, 6),
        old_means=old_means,
        old_sigmas=old_sigmas,
    )

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    expected = torch.sum(
        torch.log(new_sigmas / old_sigmas + 1.0e-5)
        + (old_sigmas.square() + (old_means - new_means.detach()).square()) / (2.0 * new_sigmas.square())
        - 0.5,
        dim=-1,
    ).mean()
    print(
        "[probe ppo_distribution_exact] "
        f"observed={result.distribution_kl_mean:.9f} "
        f"expected={float(expected.item()):.9f} "
        f"available={result.distribution_kl_available}",
        flush=True,
    )
    assert result.distribution_kl_available
    assert abs(result.distribution_kl_mean - float(expected.item())) < 1e-8
    assert abs(result.approx_kl - result.distribution_kl_mean) < 1e-8


def test_clipped_surrogate_matches_hand_computed_ratio_cases() -> None:
    log_ratio = torch.log(torch.tensor([1.5, 0.5], dtype=torch.float32))
    policy = StaticEvalPolicy(
        log_prob=log_ratio.clone().detach().requires_grad_(True),
        value=torch.zeros(2, requires_grad=True),
        entropy=torch.zeros(2),
    )
    batch = FrontRESSegmentPPOBatch(
        observations=torch.zeros(2, 4),
        actions=torch.zeros(2, 6),
        old_log_probs=torch.zeros(2),
        old_values=torch.zeros(2),
        returns=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]),
        valid_mask=torch.tensor([True, True]),
        segment_ids=torch.tensor([31, 32]),
        action_mask=torch.ones(2, 6),
    )

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    expected_actor_loss = -torch.tensor([1.2, -0.8]).mean()
    print(
        "[probe ppo_clip_exact] "
        f"actor_loss={result.actor_loss.detach().item():.6f} "
        f"expected={expected_actor_loss.item():.6f} "
        f"ratio_mean={result.ratio_mean:.6f} "
        f"clip_frac={result.clip_frac:.6f}",
        flush=True,
    )
    torch.testing.assert_close(result.actor_loss.detach(), expected_actor_loss)
    assert abs(result.ratio_mean - 1.0) < 1e-6
    assert result.clip_frac == 1.0


def test_advantage_dominance_diagnostic_exposes_top_sample_control() -> None:
    policy = StaticEvalPolicy(
        log_prob=torch.zeros(4, requires_grad=True),
        value=torch.zeros(4, requires_grad=True),
        mean=torch.zeros(4, 6, requires_grad=True),
        sigma=torch.ones(4, 6),
        entropy=torch.zeros(4),
    )
    batch = FrontRESSegmentPPOBatch(
        observations=torch.zeros(4, 4),
        actions=torch.zeros(4, 6),
        old_log_probs=torch.zeros(4),
        old_values=torch.zeros(4),
        returns=torch.zeros(4),
        advantages=torch.tensor([1000.0, 1.0, -1.0, 1.0]),
        valid_mask=torch.tensor([True, True, True, True]),
        segment_ids=torch.tensor([33, 34, 35, 36]),
        action_mask=torch.ones(4, 6),
        old_means=torch.zeros(4, 6),
        old_sigmas=torch.ones(4, 6),
    )

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    expected_top1 = 1000.0 / 1003.0
    print(
        "[probe ppo_advantage_dominance] "
        f"adv_mean={result.advantage_mean:.6f} "
        f"adv_min={result.advantage_min:.6f} "
        f"adv_max={result.advantage_max:.6f} "
        f"adv_abs_mean={result.advantage_abs_mean:.6f} "
        f"adv_abs_top1_frac={result.advantage_abs_top1_frac:.6f}",
        flush=True,
    )

    assert result.valid_count == 4
    assert abs(result.advantage_abs_top1_frac - expected_top1) < 1e-6
    assert result.advantage_abs_top1_frac > 0.99


def test_scale_only_advantage_normalization_preserves_no_regret_sign() -> None:
    advantages = torch.tensor([0.01, 0.03, 0.06])
    policy = StaticEvalPolicy(
        log_prob=torch.zeros(3, requires_grad=True),
        value=torch.zeros(3, requires_grad=True),
        mean=torch.zeros(3, 6, requires_grad=True),
        sigma=torch.ones(3, 6),
        entropy=torch.zeros(3),
    )
    batch = FrontRESSegmentPPOBatch(
        observations=torch.zeros(3, 4),
        actions=torch.zeros(3, 6),
        old_log_probs=torch.zeros(3),
        old_values=torch.zeros(3),
        returns=torch.zeros(3),
        advantages=advantages,
        valid_mask=torch.tensor([True, True, True]),
        segment_ids=torch.tensor([34, 35, 36]),
        action_mask=torch.ones(3, 6),
        old_means=torch.zeros(3, 6),
        old_sigmas=torch.ones(3, 6),
    )

    scale_only = compute_frontres_segment_ppo_loss(
        policy,
        batch,
        FrontRESSegmentPPOConfig(entropy_coef=0.0, advantage_normalization="scale_only"),
    )
    standard = compute_frontres_segment_ppo_loss(
        policy,
        batch,
        FrontRESSegmentPPOConfig(entropy_coef=0.0, advantage_normalization="standard"),
    )
    expected_scale = torch.sqrt(advantages.square().mean()).item()
    print(
        "[probe ppo_advantage_scale_only] "
        f"scale={scale_only.advantage_scale:.9f} expected_scale={expected_scale:.9f} "
        f"scale_min={scale_only.advantage_min:.6f} scale_max={scale_only.advantage_max:.6f} "
        f"scale_sign_flips={scale_only.advantage_sign_flip_count} "
        f"standard_min={standard.advantage_min:.6f} standard_sign_flips={standard.advantage_sign_flip_count}",
        flush=True,
    )

    assert abs(scale_only.advantage_scale - expected_scale) < 1e-8
    assert scale_only.advantage_min > 0.0
    assert scale_only.advantage_sign_flip_count == 0
    assert standard.advantage_min < 0.0
    assert standard.advantage_sign_flip_count > 0


def test_small_sigma_kl_sensitivity_matches_exact_formula() -> None:
    obs = torch.zeros(1, 4)
    actions = torch.zeros(1, 6)
    old_means = torch.zeros(1, 6)
    new_means = torch.full((1, 6), 0.02, requires_grad=True)

    def _result_for_sigma(sigma_value: float):
        sigma = torch.full((1, 6), float(sigma_value))
        policy = StaticEvalPolicy(
            log_prob=torch.zeros(1, requires_grad=True),
            value=torch.zeros(1, requires_grad=True),
            mean=new_means,
            sigma=sigma,
            entropy=torch.zeros(1),
        )
        batch = FrontRESSegmentPPOBatch(
            observations=obs,
            actions=actions,
            old_log_probs=torch.zeros(1),
            old_values=torch.zeros(1),
            returns=torch.zeros(1),
            advantages=torch.ones(1),
            valid_mask=torch.tensor([True]),
            segment_ids=torch.tensor([37]),
            action_mask=torch.ones(1, 6),
            old_means=old_means,
            old_sigmas=sigma,
        )
        return compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))

    small = _result_for_sigma(0.01)
    normal = _result_for_sigma(1.0)
    expected_small = torch.sum(
        torch.log(torch.ones(1, 6) + 1.0e-5)
        + new_means.detach().square() / (2.0 * (0.01**2))
    ).item()
    print(
        "[probe ppo_small_sigma_kl_sensitivity] "
        f"sigma_small={small.sigma_min:.6f} "
        f"kl_small={small.distribution_kl_mean:.6f} "
        f"sigma_normal={normal.sigma_min:.6f} "
        f"kl_normal={normal.distribution_kl_mean:.9f} "
        f"mean_delta_l2={small.distribution_mean_delta_l2_mean:.6f}",
        flush=True,
    )

    assert abs(small.distribution_kl_mean - expected_small) < 1e-5
    assert small.distribution_kl_mean > normal.distribution_kl_mean * 1000.0
    assert abs(small.old_sigma_min - 0.01) < 1e-8
    assert abs(small.sigma_min - 0.01) < 1e-8


def test_ratio_source_decomposition_identifies_tail_sigma_and_dim_contribution() -> None:
    raw_actions = torch.tensor([[0.03, -0.01, 0.00, 0.06, -0.02, 0.01]])
    old_means = torch.zeros(1, 6)
    old_sigmas = torch.full((1, 6), 0.01)
    new_means = torch.tensor([[0.00, -0.01, 0.00, 0.03, -0.02, 0.01]], requires_grad=True)
    new_sigmas = torch.full((1, 6), 0.01)
    old_logprob_dim = torch.distributions.Normal(old_means, old_sigmas).log_prob(raw_actions)
    new_logprob_dim = torch.distributions.Normal(new_means, new_sigmas).log_prob(raw_actions)
    log_ratio_dim = new_logprob_dim - old_logprob_dim
    policy = StaticEvalPolicy(
        log_prob=new_logprob_dim.sum(dim=-1).detach().clone().requires_grad_(True),
        value=torch.zeros(1, requires_grad=True),
        mean=new_means,
        sigma=new_sigmas,
        entropy=torch.zeros(1),
        raw_actions=raw_actions,
        log_jacobian_contrib=torch.tensor([[-1.61, -1.61, -1.61, -0.94, -0.94, -0.94]]),
    )
    batch = FrontRESSegmentPPOBatch(
        observations=torch.zeros(1, 4),
        actions=torch.zeros(1, 6),
        old_log_probs=old_logprob_dim.sum(dim=-1),
        old_values=torch.zeros(1),
        returns=torch.zeros(1),
        advantages=torch.ones(1),
        valid_mask=torch.tensor([True]),
        segment_ids=torch.tensor([38]),
        action_mask=torch.ones(1, 6),
        old_means=old_means,
        old_sigmas=old_sigmas,
    )

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    print(
        "[probe ppo_ratio_source_decomposition] "
        f"raw_old_l2={result.raw_action_old_mean_l2_mean:.9f} "
        f"raw_abs_dim_mean={result.raw_action_old_mean_abs_dim_mean} "
        f"sigma_dim={result.sigma_dim_mean} "
        f"mean_delta_dim={result.distribution_mean_delta_dim_mean} "
        f"log_ratio_dim={result.log_ratio_contrib_dim_mean} "
        f"log_jacobian_dim={result.log_jacobian_dim_mean}",
        flush=True,
    )

    assert abs(result.raw_action_old_mean_l2_mean - raw_actions.norm(dim=-1).item()) < 1e-8
    torch.testing.assert_close(
        torch.tensor(result.raw_action_old_mean_abs_dim_mean),
        raw_actions.abs()[0],
    )
    torch.testing.assert_close(torch.tensor(result.sigma_dim_mean), new_sigmas[0])
    torch.testing.assert_close(torch.tensor(result.distribution_mean_delta_dim_mean), new_means.detach()[0])
    torch.testing.assert_close(torch.tensor(result.log_ratio_contrib_dim_mean), log_ratio_dim[0])
    torch.testing.assert_close(torch.tensor(result.log_ratio_contrib_abs_dim_max), log_ratio_dim.abs()[0])
    torch.testing.assert_close(
        torch.tensor(result.log_jacobian_dim_mean),
        torch.tensor([-1.61, -1.61, -1.61, -0.94, -0.94, -0.94]),
    )


def test_execution_mask_prevents_inactive_dim_ratio_spike_from_unprojected_old_mean() -> None:
    raw_actions = torch.zeros(1, 6)
    old_means = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, -1.946]])
    new_means = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, -1.945]], requires_grad=True)
    sigma = torch.full((1, 6), 0.01)
    old_logprob_dim = torch.distributions.Normal(old_means, sigma).log_prob(raw_actions)
    new_logprob_dim = torch.distributions.Normal(new_means, sigma).log_prob(raw_actions)
    policy = StaticEvalPolicy(
        log_prob=new_logprob_dim.sum(dim=-1).detach().clone().requires_grad_(True),
        value=torch.zeros(1, requires_grad=True),
        mean=new_means,
        sigma=sigma,
        entropy=torch.zeros(1),
        raw_actions=raw_actions,
        log_jacobian_contrib=torch.zeros(1, 6),
    )
    batch = FrontRESSegmentPPOBatch(
        observations=torch.zeros(1, 4),
        actions=torch.zeros(1, 6),
        old_log_probs=old_logprob_dim.sum(dim=-1),
        old_values=torch.zeros(1),
        returns=torch.zeros(1),
        advantages=torch.ones(1),
        valid_mask=torch.tensor([True]),
        segment_ids=torch.tensor([39]),
        action_mask=torch.tensor([[False, False, False, True, True, False]]),
        old_means=old_means,
        old_sigmas=sigma,
    )

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    print(
        "[probe ppo_masked_action_unmasked_old_mean_guard] "
        f"raw_abs_dim_mean={result.raw_action_old_mean_abs_dim_mean} "
        f"mean_delta_dim={result.distribution_mean_delta_dim_mean} "
        f"sigma_dim={result.sigma_dim_mean} "
        f"log_ratio_dim={result.log_ratio_contrib_dim_mean}",
        flush=True,
    )

    assert result.raw_action_old_mean_abs_dim_mean[5] > 1.9
    assert abs(result.distribution_mean_delta_dim_mean[5]) < 1e-8
    assert abs(result.sigma_dim_mean[5] - 0.01) < 1e-8
    assert abs(result.log_ratio_contrib_dim_mean[5]) < 1e-8
    assert max(abs(v) for v in result.log_ratio_contrib_dim_mean[:5]) < 1e-6


def test_old_policy_tensors_are_detached_from_segment_ppo_loss() -> None:
    policy = FakeSegmentPolicy()
    old_log_probs = torch.zeros(2, requires_grad=True)
    old_values = torch.zeros(2, requires_grad=True)
    returns = torch.ones(2, requires_grad=True)
    advantages = torch.ones(2, requires_grad=True)
    old_means = torch.zeros(2, 6, requires_grad=True)
    old_sigmas = torch.ones(2, 6, requires_grad=True)
    batch = FrontRESSegmentPPOBatch(
        observations=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        actions=torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        old_log_probs=old_log_probs,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        valid_mask=torch.tensor([True, True]),
        segment_ids=torch.tensor([41, 42]),
        action_mask=torch.ones(2, 6),
        old_means=old_means,
        old_sigmas=old_sigmas,
    )

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    result.total_loss.backward()
    print(
        "[probe ppo_old_tensor_detach] "
        f"old_log_probs_grad={old_log_probs.grad} "
        f"old_values_grad={old_values.grad} "
        f"returns_grad={returns.grad} "
        f"advantages_grad={advantages.grad} "
        f"old_means_grad={old_means.grad} "
        f"old_sigmas_grad={old_sigmas.grad}",
        flush=True,
    )
    assert old_log_probs.grad is None
    assert old_values.grad is None
    assert returns.grad is None
    assert advantages.grad is None
    assert old_means.grad is None
    assert old_sigmas.grad is None
    assert policy.actor.weight.grad is not None


def test_row_permutation_does_not_change_segment_ppo_loss_or_diagnostics() -> None:
    policy = FakeSegmentPolicy()
    batch = FrontRESSegmentPPOBatch(
        observations=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        actions=torch.tensor(
            [
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.2, 0.1, 0.0, 0.0, 0.0],
            ]
        ),
        old_log_probs=torch.tensor([-0.02, -0.01, -0.03]),
        old_values=torch.tensor([0.1, -0.1, 0.0]),
        returns=torch.tensor([0.5, -0.2, 0.3]),
        advantages=torch.tensor([1.0, -0.5, 0.25]),
        valid_mask=torch.tensor([True, True, True]),
        segment_ids=torch.tensor([51, 52, 53]),
        action_mask=torch.ones(3, 6),
        old_means=torch.zeros(3, 6),
        old_sigmas=torch.ones(3, 6),
    )
    permuted = _permute_batch(batch, torch.tensor([2, 0, 1]))

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    permuted_result = compute_frontres_segment_ppo_loss(
        policy,
        permuted,
        FrontRESSegmentPPOConfig(entropy_coef=0.0),
    )
    print(
        "[probe ppo_row_permutation] "
        f"loss={result.total_loss.detach().item():.9f} "
        f"permuted_loss={permuted_result.total_loss.detach().item():.9f} "
        f"kl={result.approx_kl:.9f} "
        f"permuted_kl={permuted_result.approx_kl:.9f}",
        flush=True,
    )
    torch.testing.assert_close(result.total_loss.detach(), permuted_result.total_loss.detach())
    torch.testing.assert_close(result.actor_loss.detach(), permuted_result.actor_loss.detach())
    torch.testing.assert_close(result.value_loss.detach(), permuted_result.value_loss.detach())
    assert abs(result.approx_kl - permuted_result.approx_kl) < 1e-8
    assert abs(result.ratio_mean - permuted_result.ratio_mean) < 1e-8


def test_action_mask_does_not_reduce_direct_delta_se_ppo_support() -> None:
    full_mask = torch.ones(1, 6, dtype=torch.bool)
    perturbation_rp_metadata = torch.tensor([[False, False, False, True, True, False]])
    full_policy = FakeSegmentPolicy()
    rp_policy = FakeSegmentPolicy()
    full_batch = _full_delta_repair_batch(full_mask)
    rp_batch = _full_delta_repair_batch(full_mask)

    full_result = compute_frontres_segment_ppo_loss(
        full_policy,
        full_batch,
        FrontRESSegmentPPOConfig(entropy_coef=0.0),
    )
    rp_result = compute_frontres_segment_ppo_loss(
        rp_policy,
        rp_batch,
        FrontRESSegmentPPOConfig(entropy_coef=0.0),
    )
    full_result.total_loss.backward()
    rp_result.total_loss.backward()
    full_grad = full_policy.actor.weight.grad.detach().clone()
    rp_grad = rp_policy.actor.weight.grad.detach().clone()
    rp_grad_by_dim = rp_grad[:, 0].detach().abs()
    print(
        "[probe ppo_cone_full_support] "
        f"full_loss={full_result.total_loss.detach().item():.9f} "
        f"rp_mask_loss={rp_result.total_loss.detach().item():.9f} "
        f"perturbation_rp_metadata={perturbation_rp_metadata[0].int().tolist()} "
        f"execution_mask={full_mask[0].int().tolist()} "
        f"grad_by_dim={rp_grad_by_dim.tolist()}",
        flush=True,
    )

    torch.testing.assert_close(full_result.actor_loss.detach(), rp_result.actor_loss.detach())
    torch.testing.assert_close(full_result.total_loss.detach(), rp_result.total_loss.detach())
    torch.testing.assert_close(full_grad, rp_grad)
    assert int((rp_grad_by_dim > 0.0).sum().item()) == 6


def test_local_rp_metadata_can_train_full_6d_repair_action() -> None:
    perturbation_rp_metadata = torch.tensor([[False, False, False, True, True, False]])
    execution_mask = torch.ones(1, 6, dtype=torch.bool)
    policy = FakeSegmentPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.25)
    batch = _full_delta_repair_batch(execution_mask)
    before_mean = policy.evaluate_segment_actions(batch.observations, batch.actions).mean.detach().clone()
    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    optimizer.zero_grad(set_to_none=True)
    result.total_loss.backward()
    optimizer.step()
    after_mean = policy.evaluate_segment_actions(batch.observations, batch.actions).mean.detach()
    print(
        "[probe ppo_local_rp_full_6d_repair] "
        f"perturbation_rp_metadata={perturbation_rp_metadata[0].int().tolist()} "
        f"execution_mask={execution_mask[0].int().tolist()} "
        f"before={before_mean[0].tolist()} "
        f"after={after_mean[0].tolist()} "
        f"action={batch.actions[0].tolist()}",
        flush=True,
    )

    assert result.should_step
    assert torch.all(after_mean[0].abs() > before_mean[0].abs())
    assert torch.all(torch.sign(after_mean[0]) == torch.sign(batch.actions[0]))
    assert torch.all(after_mean[0].abs() < batch.actions[0].abs())


def test_distribution_kl_remains_full_6d_under_rp_only_action_mask() -> None:
    perturbation_rp_metadata = torch.tensor([[False, False, False, True, True, False]])
    execution_mask = torch.ones(1, 6, dtype=torch.bool)
    old_means = torch.zeros(1, 6)
    old_sigmas = torch.ones(1, 6)
    new_means = torch.tensor([[0.20, -0.15, 0.10, 0.30, -0.25, 0.05]], requires_grad=True)
    new_sigmas = torch.ones(1, 6)
    policy = StaticEvalPolicy(
        log_prob=torch.zeros(1, requires_grad=True),
        value=torch.zeros(1, requires_grad=True),
        mean=new_means,
        sigma=new_sigmas,
        entropy=torch.zeros(1),
    )
    batch = FrontRESSegmentPPOBatch(
        observations=torch.zeros(1, 4),
        actions=torch.zeros(1, 6),
        old_log_probs=torch.zeros(1),
        old_values=torch.zeros(1),
        returns=torch.zeros(1),
        advantages=torch.ones(1),
        valid_mask=torch.tensor([True]),
        segment_ids=torch.tensor([71]),
        action_mask=execution_mask,
        old_means=old_means,
        old_sigmas=old_sigmas,
    )

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    full_kl = torch.sum(
        torch.log(new_sigmas / old_sigmas + 1.0e-5)
        + (old_sigmas.square() + (old_means - new_means.detach()).square()) / (2.0 * new_sigmas.square())
        - 0.5,
        dim=-1,
    ).mean()
    rp_dims = torch.tensor([3, 4])
    rp_only_kl = torch.sum(0.5 * new_means.detach()[:, rp_dims].square(), dim=-1).mean()
    print(
        "[probe ppo_cone_full_kl] "
        f"observed={result.distribution_kl_mean:.9f} "
        f"full_expected={float(full_kl.item()):.9f} "
        f"rp_only_expected_without_non_rp={float(rp_only_kl.item()):.9f} "
        f"perturbation_rp_metadata={perturbation_rp_metadata[0].int().tolist()} "
        f"execution_mask={execution_mask[0].int().tolist()}",
        flush=True,
    )

    assert result.distribution_kl_available
    assert abs(result.distribution_kl_mean - float(full_kl.item())) < 1e-8
    assert result.distribution_kl_mean > float(rp_only_kl.item()) + 1e-3


def test_execution_mask_projects_current_policy_eval_for_inactive_dims() -> None:
    execution_mask = torch.tensor([[True, True, True, True, True, False]])
    raw_actions = torch.zeros(1, 6)
    old_means = torch.zeros(1, 6)
    old_sigmas = torch.full((1, 6), 0.01)
    new_means = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, -1.946]], requires_grad=True)
    new_sigmas = torch.full((1, 6), 0.01)
    log_j = torch.zeros(1, 6)
    old_log_probs = torch.distributions.Normal(old_means, old_sigmas).log_prob(raw_actions).sum(dim=-1)
    policy = StaticEvalPolicy(
        log_prob=torch.tensor([-18904.886719], requires_grad=True),
        value=torch.zeros(1, requires_grad=True),
        mean=new_means,
        sigma=new_sigmas,
        entropy=torch.zeros(1),
        raw_actions=raw_actions,
        log_jacobian_contrib=log_j,
    )
    batch = FrontRESSegmentPPOBatch(
        observations=torch.zeros(1, 4),
        actions=torch.zeros(1, 6),
        old_log_probs=old_log_probs,
        old_values=torch.zeros(1),
        returns=torch.zeros(1),
        advantages=torch.ones(1),
        valid_mask=torch.tensor([True]),
        segment_ids=torch.tensor([72]),
        action_mask=execution_mask,
        old_means=old_means,
        old_sigmas=old_sigmas,
    )

    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))
    print(
        "[probe ppo_execution_mask_projected_eval] "
        f"execution_mask={execution_mask[0].int().tolist()} "
        f"raw_action_old_mean_dim={result.raw_action_old_mean_abs_dim_mean} "
        f"mean_delta_dim={result.distribution_mean_delta_dim_mean} "
        f"log_ratio_dim={result.log_ratio_contrib_dim_mean} "
        f"ratio_mean={result.ratio_mean:.9f} "
        f"clip_frac={result.clip_frac:.9f}",
        flush=True,
    )

    assert result.distribution_kl_available
    assert abs(result.raw_action_old_mean_abs_dim_mean[5]) < 1e-8
    assert abs(result.distribution_mean_delta_dim_mean[5]) < 1e-8
    assert abs(result.log_ratio_contrib_dim_mean[5]) < 1e-8
    assert abs(result.ratio_mean - 1.0) < 1e-6
    assert result.clip_frac == 0.0


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
    test_positive_advantage_moves_mean_toward_stored_delta_se_action()
    test_old_distribution_stats_drive_mosaic_style_kl()
    test_distribution_kl_matches_old_new_stats_exactly()
    test_clipped_surrogate_matches_hand_computed_ratio_cases()
    test_advantage_dominance_diagnostic_exposes_top_sample_control()
    test_scale_only_advantage_normalization_preserves_no_regret_sign()
    test_small_sigma_kl_sensitivity_matches_exact_formula()
    test_ratio_source_decomposition_identifies_tail_sigma_and_dim_contribution()
    test_execution_mask_prevents_inactive_dim_ratio_spike_from_unprojected_old_mean()
    test_old_policy_tensors_are_detached_from_segment_ppo_loss()
    test_row_permutation_does_not_change_segment_ppo_loss_or_diagnostics()
    test_action_mask_does_not_reduce_direct_delta_se_ppo_support()
    test_local_rp_metadata_can_train_full_6d_repair_action()
    test_distribution_kl_remains_full_6d_under_rp_only_action_mask()
    test_execution_mask_projects_current_policy_eval_for_inactive_dims()
    test_invalid_samples_do_not_contribute_to_loss()
    test_nonfinite_valid_rows_are_masked_before_loss()
    test_extreme_log_ratio_does_not_overflow_loss()
    test_ppo_tuple_requires_6d_action_and_vector_fields()
    test_all_invalid_batch_has_zero_loss_and_no_step()
    print("result: PASS")


if __name__ == "__main__":
    main()
