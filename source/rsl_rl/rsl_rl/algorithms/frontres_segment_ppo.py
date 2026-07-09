from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FrontRESSegmentPPOConfig:
    clip_param: float = 0.2
    value_clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    use_clipped_value_loss: bool = True
    normalize_advantages: bool = False
    advantage_normalization: str = "none"
    advantage_scale_epsilon: float = 1.0e-8
    max_log_ratio: float = 20.0


@dataclass(frozen=True)
class FrontRESSegmentPPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    valid_mask: torch.Tensor
    segment_ids: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    old_means: torch.Tensor | None = None
    old_sigmas: torch.Tensor | None = None


@dataclass(frozen=True)
class FrontRESSegmentPolicyEval:
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor | None = None
    mean: torch.Tensor | None = None
    sigma: torch.Tensor | None = None
    raw_actions: torch.Tensor | None = None
    log_jacobian_contrib: torch.Tensor | None = None


@dataclass(frozen=True)
class FrontRESSegmentPPOResult:
    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    valid_count: int
    valid_frac: float
    clip_frac: float
    approx_kl: float
    ratio_mean: float
    ratio_max: float = 0.0
    old_log_prob_mean: float = 0.0
    new_log_prob_mean: float = 0.0
    raw_log_ratio_mean: float = 0.0
    raw_log_ratio_min: float = 0.0
    raw_log_ratio_max: float = 0.0
    pre_update_raw_log_ratio_mean: float = 0.0
    pre_update_raw_log_ratio_min: float = 0.0
    pre_update_raw_log_ratio_max: float = 0.0
    pre_update_clamped_ratio_mean: float = 0.0
    pre_update_clamped_ratio_max: float = 0.0
    advantage_mean: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    advantage_abs_mean: float = 0.0
    advantage_abs_max: float = 0.0
    advantage_abs_top1_frac: float = 0.0
    advantage_scale: float = 1.0
    advantage_sign_flip_count: int = 0
    logprob_approx_kl: float = 0.0
    distribution_kl_mean: float = 0.0
    distribution_kl_available: bool = False
    distribution_mean_delta_l2_mean: float = 0.0
    distribution_mean_delta_max_abs: float = 0.0
    old_sigma_min: float = 0.0
    sigma_min: float = 0.0
    post_update_distribution_kl_mean: float = 0.0
    post_update_distribution_kl_available: bool = False
    post_update_logprob_approx_kl: float = 0.0
    post_update_ratio_mean: float = 0.0
    post_update_ratio_max: float = 0.0
    post_update_raw_log_ratio_mean: float = 0.0
    post_update_raw_log_ratio_min: float = 0.0
    post_update_raw_log_ratio_max: float = 0.0
    post_update_clamped_ratio_mean: float = 0.0
    post_update_clamped_ratio_max: float = 0.0
    post_update_clip_frac: float = 0.0
    raw_action_old_mean_l2_mean: float = 0.0
    raw_action_old_mean_abs_max: float = 0.0
    raw_action_old_mean_abs_dim_mean: tuple[float, ...] = ()
    raw_action_old_mean_abs_dim_max: tuple[float, ...] = ()
    old_sigma_dim_mean: tuple[float, ...] = ()
    sigma_dim_mean: tuple[float, ...] = ()
    distribution_mean_delta_dim_mean: tuple[float, ...] = ()
    distribution_mean_delta_abs_dim_max: tuple[float, ...] = ()
    log_ratio_contrib_dim_mean: tuple[float, ...] = ()
    log_ratio_contrib_abs_dim_max: tuple[float, ...] = ()
    log_jacobian_dim_mean: tuple[float, ...] = ()
    log_jacobian_abs_dim_max: tuple[float, ...] = ()
    post_update_raw_action_old_mean_l2_mean: float = 0.0
    post_update_raw_action_old_mean_abs_max: float = 0.0
    post_update_raw_action_old_mean_abs_dim_mean: tuple[float, ...] = ()
    post_update_raw_action_old_mean_abs_dim_max: tuple[float, ...] = ()
    post_update_old_sigma_dim_mean: tuple[float, ...] = ()
    post_update_sigma_dim_mean: tuple[float, ...] = ()
    post_update_distribution_mean_delta_dim_mean: tuple[float, ...] = ()
    post_update_distribution_mean_delta_abs_dim_max: tuple[float, ...] = ()
    post_update_log_ratio_contrib_dim_mean: tuple[float, ...] = ()
    post_update_log_ratio_contrib_abs_dim_max: tuple[float, ...] = ()
    post_update_log_jacobian_dim_mean: tuple[float, ...] = ()
    post_update_log_jacobian_abs_dim_max: tuple[float, ...] = ()

    @property
    def should_step(self) -> bool:
        return self.valid_count > 0

    def diagnostics(self) -> dict[str, float]:
        return {
            "segment/ppo_total_loss": float(self.total_loss.detach().cpu().item()),
            "segment/ppo_actor_loss": float(self.actor_loss.detach().cpu().item()),
            "segment/ppo_value_loss": float(self.value_loss.detach().cpu().item()),
            "segment/ppo_entropy": float(self.entropy.detach().cpu().item()),
            "segment/ppo_valid_frac": self.valid_frac,
            "segment/ppo_clip_frac": self.clip_frac,
            "segment/ppo_approx_kl": self.approx_kl,
            "segment/ppo_ratio_mean": self.ratio_mean,
            "segment/ppo_ratio_max": self.ratio_max,
            "segment/ppo_old_log_prob_mean": self.old_log_prob_mean,
            "segment/ppo_new_log_prob_mean": self.new_log_prob_mean,
            "segment/ppo_raw_log_ratio_mean": self.raw_log_ratio_mean,
            "segment/ppo_raw_log_ratio_min": self.raw_log_ratio_min,
            "segment/ppo_raw_log_ratio_max": self.raw_log_ratio_max,
            "segment/ppo_pre_update_raw_log_ratio_mean": self.pre_update_raw_log_ratio_mean,
            "segment/ppo_pre_update_raw_log_ratio_min": self.pre_update_raw_log_ratio_min,
            "segment/ppo_pre_update_raw_log_ratio_max": self.pre_update_raw_log_ratio_max,
            "segment/ppo_pre_update_clamped_ratio_mean": self.pre_update_clamped_ratio_mean,
            "segment/ppo_pre_update_clamped_ratio_max": self.pre_update_clamped_ratio_max,
            "segment/ppo_advantage_mean": self.advantage_mean,
            "segment/ppo_advantage_min": self.advantage_min,
            "segment/ppo_advantage_max": self.advantage_max,
            "segment/ppo_advantage_abs_mean": self.advantage_abs_mean,
            "segment/ppo_advantage_abs_max": self.advantage_abs_max,
            "segment/ppo_advantage_abs_top1_frac": self.advantage_abs_top1_frac,
            "segment/ppo_advantage_scale": self.advantage_scale,
            "segment/ppo_advantage_sign_flip_count": float(self.advantage_sign_flip_count),
            "segment/ppo_logprob_approx_kl": self.logprob_approx_kl,
            "segment/ppo_distribution_kl_mean": self.distribution_kl_mean,
            "segment/ppo_distribution_kl_available": float(self.distribution_kl_available),
            "segment/ppo_distribution_mean_delta_l2_mean": self.distribution_mean_delta_l2_mean,
            "segment/ppo_distribution_mean_delta_max_abs": self.distribution_mean_delta_max_abs,
            "segment/ppo_old_sigma_min": self.old_sigma_min,
            "segment/ppo_sigma_min": self.sigma_min,
            "segment/ppo_post_update_distribution_kl_mean": self.post_update_distribution_kl_mean,
            "segment/ppo_post_update_distribution_kl_available": float(
                self.post_update_distribution_kl_available
            ),
            "segment/ppo_post_update_logprob_approx_kl": self.post_update_logprob_approx_kl,
            "segment/ppo_post_update_ratio_mean": self.post_update_ratio_mean,
            "segment/ppo_post_update_ratio_max": self.post_update_ratio_max,
            "segment/ppo_post_update_raw_log_ratio_mean": self.post_update_raw_log_ratio_mean,
            "segment/ppo_post_update_raw_log_ratio_min": self.post_update_raw_log_ratio_min,
            "segment/ppo_post_update_raw_log_ratio_max": self.post_update_raw_log_ratio_max,
            "segment/ppo_post_update_clamped_ratio_mean": self.post_update_clamped_ratio_mean,
            "segment/ppo_post_update_clamped_ratio_max": self.post_update_clamped_ratio_max,
            "segment/ppo_post_update_clip_frac": self.post_update_clip_frac,
        }


def compute_frontres_segment_ppo_loss(
    policy: Any,
    batch: FrontRESSegmentPPOBatch,
    cfg: FrontRESSegmentPPOConfig | None = None,
) -> FrontRESSegmentPPOResult:
    """Compute one direct Delta SE PPO loss on an already sampled Segment batch.

    Status: active Segment Replay algorithm boundary.
    Upstream: live runner/storage converts rollout evidence into FrontRESSegmentPPOBatch.
    Downstream: runner uses total_loss for backward/step and diagnostics for KL/trust logs.
    Evidence: contract-confirmed by frontres_segment_algorithm_contract.py and live single-update tests.
    Gap: this pure loss does not prove IsaacLab live rollout quality.
    """
    cfg = FrontRESSegmentPPOConfig() if cfg is None else cfg
    _validate_batch(batch)
    policy_eval = _evaluate_policy(policy, batch)
    policy_eval = _project_policy_eval_to_action_mask(policy_eval, batch)
    _validate_policy_eval(policy_eval, batch)

    # B1: Build the valid training rows. Old policy tensors are detached below,
    # so gradients only flow through the current policy evaluation.
    finite = (
        torch.isfinite(policy_eval.log_prob)
        & torch.isfinite(policy_eval.value)
        & torch.isfinite(batch.old_log_probs)
        & torch.isfinite(batch.old_values)
        & torch.isfinite(batch.returns)
        & torch.isfinite(batch.advantages)
    )
    if policy_eval.entropy is not None:
        finite = finite & torch.isfinite(policy_eval.entropy)
    has_distribution_stats = (
        batch.old_means is not None
        and batch.old_sigmas is not None
        and policy_eval.mean is not None
        and policy_eval.sigma is not None
    )
    if has_distribution_stats:
        finite = (
            finite
            & torch.isfinite(batch.old_means).all(dim=-1)
            & torch.isfinite(batch.old_sigmas).all(dim=-1)
            & (batch.old_sigmas > 0.0).all(dim=-1)
            & torch.isfinite(policy_eval.mean).all(dim=-1)
            & torch.isfinite(policy_eval.sigma).all(dim=-1)
            & (policy_eval.sigma > 0.0).all(dim=-1)
        )
    valid = batch.valid_mask.bool() & finite
    valid_count = int(valid.sum().item())
    valid_frac = float(valid.float().mean().item()) if valid.numel() else 0.0
    if valid_count == 0:
        zero = (policy_eval.log_prob.sum() + policy_eval.value.sum()) * 0.0
        entropy_zero = zero.detach()
        return FrontRESSegmentPPOResult(
            total_loss=zero,
            actor_loss=zero,
            value_loss=zero,
            entropy=entropy_zero,
            valid_count=0,
            valid_frac=valid_frac,
            clip_frac=0.0,
            approx_kl=0.0,
            ratio_mean=0.0,
        )

    log_prob = policy_eval.log_prob[valid]
    value = policy_eval.value[valid]
    old_log_prob = batch.old_log_probs[valid].detach()
    old_value = batch.old_values[valid].detach()
    returns = batch.returns[valid].detach()
    advantages = batch.advantages[valid].detach()
    advantages, advantage_scale, advantage_sign_flip_count = _prepare_advantages(advantages, cfg)

    # B2: PPO ratio path. raw_log_ratio is kept for diagnosis; log_ratio is
    # clamped only before exp() to keep the surrogate finite.
    raw_log_ratio = log_prob - old_log_prob
    log_ratio = raw_log_ratio.clamp(-abs(float(cfg.max_log_ratio)), abs(float(cfg.max_log_ratio)))
    ratio = torch.exp(log_ratio)
    surrogate = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
    clipped_surrogate = clipped_ratio * advantages
    actor_loss = -torch.min(surrogate, clipped_surrogate).mean()

    if cfg.use_clipped_value_loss:
        value_clipped = old_value + (value - old_value).clamp(-cfg.value_clip_param, cfg.value_clip_param)
        value_loss = 0.5 * torch.max((value - returns).square(), (value_clipped - returns).square()).mean()
    else:
        value_loss = 0.5 * (value - returns).square().mean()

    entropy = _masked_entropy(policy_eval.entropy, valid, log_prob)
    total_loss = actor_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy
    with torch.no_grad():
        # B3: Diagnostics are measured on this exact forward pass. The runner
        # later reuses these fields as pre-update values, and runs a second
        # forward after optimizer.step for post-update values.
        clip_frac = ((ratio - 1.0).abs() > cfg.clip_param).float().mean().item()
        logprob_approx_kl = (old_log_prob - log_prob).mean().item()
        distribution_kl_mean = (
            _distribution_kl_mean(policy_eval, batch, valid).item() if has_distribution_stats else 0.0
        )
        approx_kl = distribution_kl_mean if has_distribution_stats else logprob_approx_kl
        ratio_mean = ratio.mean().item()
        ratio_max = ratio.max().item()
        old_log_prob_mean = old_log_prob.mean().item()
        new_log_prob_mean = log_prob.mean().item()
        raw_log_ratio_mean = raw_log_ratio.mean().item()
        raw_log_ratio_min = raw_log_ratio.min().item()
        raw_log_ratio_max = raw_log_ratio.max().item()
        advantage_mean = advantages.mean().item()
        advantage_min = advantages.min().item()
        advantage_max = advantages.max().item()
        advantage_abs = advantages.abs()
        advantage_abs_mean = advantage_abs.mean().item()
        advantage_abs_max = advantage_abs.max().item()
        advantage_abs_sum = advantage_abs.sum().item()
        advantage_abs_top1_frac = advantage_abs_max / advantage_abs_sum if advantage_abs_sum > 0.0 else 0.0
        distribution_mean_delta_l2_mean = 0.0
        distribution_mean_delta_max_abs = 0.0
        old_sigma_min = 0.0
        sigma_min = 0.0
        raw_action_old_mean_l2_mean = 0.0
        raw_action_old_mean_abs_max = 0.0
        raw_action_old_mean_abs_dim_mean: tuple[float, ...] = ()
        raw_action_old_mean_abs_dim_max: tuple[float, ...] = ()
        old_sigma_dim_mean: tuple[float, ...] = ()
        sigma_dim_mean: tuple[float, ...] = ()
        distribution_mean_delta_dim_mean: tuple[float, ...] = ()
        distribution_mean_delta_abs_dim_max: tuple[float, ...] = ()
        log_ratio_contrib_dim_mean: tuple[float, ...] = ()
        log_ratio_contrib_abs_dim_max: tuple[float, ...] = ()
        log_jacobian_dim_mean: tuple[float, ...] = ()
        log_jacobian_abs_dim_max: tuple[float, ...] = ()
        if has_distribution_stats:
            old_mean = batch.old_means[valid].detach()
            old_sigma = batch.old_sigmas[valid].detach()
            mean = policy_eval.mean[valid].detach()
            sigma = policy_eval.sigma[valid].detach()
            mean_delta = mean - old_mean
            distribution_mean_delta_l2_mean = mean_delta.norm(dim=-1).mean().item()
            distribution_mean_delta_max_abs = mean_delta.abs().max().item()
            old_sigma_min = old_sigma.min().item()
            sigma_min = sigma.min().item()
            old_sigma_dim_mean = _dim_mean_tuple(old_sigma)
            sigma_dim_mean = _dim_mean_tuple(sigma)
            distribution_mean_delta_dim_mean = _dim_mean_tuple(mean_delta)
            distribution_mean_delta_abs_dim_max = _dim_max_tuple(mean_delta.abs())
            if policy_eval.raw_actions is not None:
                raw_actions = policy_eval.raw_actions[valid].detach()
                raw_action_old_mean = raw_actions - old_mean
                raw_action_old_mean_l2_mean = raw_action_old_mean.norm(dim=-1).mean().item()
                raw_action_old_mean_abs_max = raw_action_old_mean.abs().max().item()
                raw_action_old_mean_abs_dim_mean = _dim_mean_tuple(raw_action_old_mean.abs())
                raw_action_old_mean_abs_dim_max = _dim_max_tuple(raw_action_old_mean.abs())
                old_logprob_dim = torch.distributions.Normal(old_mean, old_sigma).log_prob(raw_actions)
                new_logprob_dim = torch.distributions.Normal(mean, sigma).log_prob(raw_actions)
                log_ratio_contrib = new_logprob_dim - old_logprob_dim
                log_ratio_contrib_dim_mean = _dim_mean_tuple(log_ratio_contrib)
                log_ratio_contrib_abs_dim_max = _dim_max_tuple(log_ratio_contrib.abs())
            if policy_eval.log_jacobian_contrib is not None:
                log_jacobian = policy_eval.log_jacobian_contrib[valid].detach()
                log_jacobian_dim_mean = _dim_mean_tuple(log_jacobian)
                log_jacobian_abs_dim_max = _dim_max_tuple(log_jacobian.abs())

    return FrontRESSegmentPPOResult(
        total_loss=total_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        entropy=entropy,
        valid_count=valid_count,
        valid_frac=valid_frac,
        clip_frac=float(clip_frac),
        approx_kl=float(approx_kl),
        ratio_mean=float(ratio_mean),
        ratio_max=float(ratio_max),
        old_log_prob_mean=float(old_log_prob_mean),
        new_log_prob_mean=float(new_log_prob_mean),
        raw_log_ratio_mean=float(raw_log_ratio_mean),
        raw_log_ratio_min=float(raw_log_ratio_min),
        raw_log_ratio_max=float(raw_log_ratio_max),
        pre_update_raw_log_ratio_mean=float(raw_log_ratio_mean),
        pre_update_raw_log_ratio_min=float(raw_log_ratio_min),
        pre_update_raw_log_ratio_max=float(raw_log_ratio_max),
        pre_update_clamped_ratio_mean=float(ratio_mean),
        pre_update_clamped_ratio_max=float(ratio_max),
        advantage_mean=float(advantage_mean),
        advantage_min=float(advantage_min),
        advantage_max=float(advantage_max),
        advantage_abs_mean=float(advantage_abs_mean),
        advantage_abs_max=float(advantage_abs_max),
        advantage_abs_top1_frac=float(advantage_abs_top1_frac),
        advantage_scale=float(advantage_scale),
        advantage_sign_flip_count=int(advantage_sign_flip_count),
        logprob_approx_kl=float(logprob_approx_kl),
        distribution_kl_mean=float(distribution_kl_mean),
        distribution_kl_available=bool(has_distribution_stats),
        distribution_mean_delta_l2_mean=float(distribution_mean_delta_l2_mean),
        distribution_mean_delta_max_abs=float(distribution_mean_delta_max_abs),
        old_sigma_min=float(old_sigma_min),
        sigma_min=float(sigma_min),
        raw_action_old_mean_l2_mean=float(raw_action_old_mean_l2_mean),
        raw_action_old_mean_abs_max=float(raw_action_old_mean_abs_max),
        raw_action_old_mean_abs_dim_mean=raw_action_old_mean_abs_dim_mean,
        raw_action_old_mean_abs_dim_max=raw_action_old_mean_abs_dim_max,
        old_sigma_dim_mean=old_sigma_dim_mean,
        sigma_dim_mean=sigma_dim_mean,
        distribution_mean_delta_dim_mean=distribution_mean_delta_dim_mean,
        distribution_mean_delta_abs_dim_max=distribution_mean_delta_abs_dim_max,
        log_ratio_contrib_dim_mean=log_ratio_contrib_dim_mean,
        log_ratio_contrib_abs_dim_max=log_ratio_contrib_abs_dim_max,
        log_jacobian_dim_mean=log_jacobian_dim_mean,
        log_jacobian_abs_dim_max=log_jacobian_abs_dim_max,
    )


def _dim_mean_tuple(tensor: torch.Tensor) -> tuple[float, ...]:
    if tensor.ndim != 2 or tensor.shape[-1] == 0:
        return ()
    return tuple(float(item) for item in tensor.mean(dim=0).detach().cpu().tolist())


def _dim_max_tuple(tensor: torch.Tensor) -> tuple[float, ...]:
    if tensor.ndim != 2 or tensor.shape[-1] == 0:
        return ()
    return tuple(float(item) for item in tensor.max(dim=0).values.detach().cpu().tolist())


def _prepare_advantages(
    advantages: torch.Tensor,
    cfg: FrontRESSegmentPPOConfig,
) -> tuple[torch.Tensor, float, int]:
    mode = str(cfg.advantage_normalization).lower()
    if cfg.normalize_advantages and mode == "none":
        mode = "standard"
    if mode not in ("none", "standard", "scale_only"):
        raise ValueError(
            "advantage_normalization must be one of none, standard, or scale_only; "
            f"got {cfg.advantage_normalization!r}"
        )
    if advantages.numel() <= 1 or mode == "none":
        return advantages, 1.0, 0
    original = advantages
    eps = abs(float(cfg.advantage_scale_epsilon))
    if mode == "standard":
        scale = advantages.std(unbiased=False) + eps
        scaled = (advantages - advantages.mean()) / scale
    else:
        scale = torch.sqrt(advantages.square().mean()).clamp_min(eps)
        scaled = advantages / scale
    sign_rows = (original != 0.0) & (scaled != 0.0)
    sign_flip_count = int((torch.sign(original[sign_rows]) != torch.sign(scaled[sign_rows])).sum().item())
    return scaled, float(scale.detach().cpu().item()), sign_flip_count


def _evaluate_policy(policy: Any, batch: FrontRESSegmentPPOBatch) -> FrontRESSegmentPolicyEval:
    if hasattr(policy, "evaluate_segment_actions"):
        value = policy.evaluate_segment_actions(batch.observations, batch.actions)
    elif callable(policy):
        value = policy(batch.observations, batch.actions)
    else:
        raise TypeError("policy must define evaluate_segment_actions(observations, actions) or be callable")
    if isinstance(value, FrontRESSegmentPolicyEval):
        return value
    if isinstance(value, dict):
        return FrontRESSegmentPolicyEval(
            log_prob=value["log_prob"],
            value=value["value"],
            entropy=value.get("entropy"),
            mean=value.get("mean"),
            sigma=value.get("sigma"),
            raw_actions=value.get("raw_actions"),
            log_jacobian_contrib=value.get("log_jacobian_contrib"),
        )
    raise TypeError(f"unsupported policy evaluation output: {type(value)!r}")


def _project_policy_eval_to_action_mask(
    policy_eval: FrontRESSegmentPolicyEval,
    batch: FrontRESSegmentPPOBatch,
) -> FrontRESSegmentPolicyEval:
    """Align current policy stats with the executed Delta SE action cone."""

    if (
        batch.action_mask is None
        or batch.old_means is None
        or batch.old_sigmas is None
        or policy_eval.mean is None
        or policy_eval.sigma is None
        or policy_eval.raw_actions is None
        or policy_eval.log_jacobian_contrib is None
    ):
        return policy_eval
    mask = batch.action_mask.to(device=policy_eval.mean.device, dtype=torch.bool)
    if bool(mask.all().item()):
        return policy_eval
    old_mean = batch.old_means.to(device=policy_eval.mean.device, dtype=policy_eval.mean.dtype).detach()
    old_sigma = batch.old_sigmas.to(device=policy_eval.sigma.device, dtype=policy_eval.sigma.dtype).detach()
    mean = torch.where(mask, policy_eval.mean, old_mean)
    sigma = torch.where(mask, policy_eval.sigma, old_sigma)
    raw_actions = policy_eval.raw_actions.to(device=mean.device, dtype=mean.dtype).detach()
    log_j = policy_eval.log_jacobian_contrib.to(device=mean.device, dtype=mean.dtype).detach()
    log_prob = (torch.distributions.Normal(mean, sigma).log_prob(raw_actions) - log_j).sum(dim=-1)
    return FrontRESSegmentPolicyEval(
        log_prob=log_prob,
        value=policy_eval.value,
        entropy=policy_eval.entropy,
        mean=mean,
        sigma=sigma,
        raw_actions=policy_eval.raw_actions,
        log_jacobian_contrib=policy_eval.log_jacobian_contrib,
    )


def _masked_entropy(entropy: torch.Tensor | None, valid: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if entropy is None:
        return like.new_zeros(())
    return entropy[valid].mean()


def _distribution_kl_mean(
    policy_eval: FrontRESSegmentPolicyEval,
    batch: FrontRESSegmentPPOBatch,
    valid: torch.Tensor,
) -> torch.Tensor:
    if batch.old_means is None or batch.old_sigmas is None or policy_eval.mean is None or policy_eval.sigma is None:
        return batch.actions.new_zeros(())
    old_mean = batch.old_means[valid].detach()
    old_sigma = batch.old_sigmas[valid].detach()
    mean = policy_eval.mean[valid].detach()
    sigma = policy_eval.sigma[valid].detach()
    kl = torch.sum(
        torch.log(sigma / old_sigma + 1.0e-5)
        + (old_sigma.square() + (old_mean - mean).square()) / (2.0 * sigma.square())
        - 0.5,
        dim=-1,
    )
    return kl.mean()


def _validate_batch(batch: FrontRESSegmentPPOBatch) -> None:
    if batch.actions.ndim != 2 or batch.actions.shape[-1] != 6:
        raise ValueError(f"actions must have shape [B, 6], got {tuple(batch.actions.shape)}")
    batch_size = batch.actions.shape[0]
    if batch.observations.ndim < 2 or batch.observations.shape[0] != batch_size:
        raise ValueError("observations must have batch dimension B matching actions")
    for name in ("old_log_probs", "old_values", "returns", "advantages", "valid_mask"):
        _require_vector(name, getattr(batch, name), batch_size)
    if batch.segment_ids is not None:
        _require_vector("segment_ids", batch.segment_ids, batch_size)
    if batch.action_mask is not None and tuple(batch.action_mask.shape) != (batch_size, 6):
        raise ValueError(f"action_mask must have shape [B, 6], got {tuple(batch.action_mask.shape)}")
    if (batch.old_means is None) != (batch.old_sigmas is None):
        raise ValueError("old_means and old_sigmas must be provided together")
    if batch.old_means is not None and tuple(batch.old_means.shape) != (batch_size, 6):
        raise ValueError(f"old_means must have shape [B, 6], got {tuple(batch.old_means.shape)}")
    if batch.old_sigmas is not None and tuple(batch.old_sigmas.shape) != (batch_size, 6):
        raise ValueError(f"old_sigmas must have shape [B, 6], got {tuple(batch.old_sigmas.shape)}")


def _validate_policy_eval(policy_eval: FrontRESSegmentPolicyEval, batch: FrontRESSegmentPPOBatch) -> None:
    batch_size = batch.actions.shape[0]
    _require_vector("policy log_prob", policy_eval.log_prob, batch_size)
    _require_vector("policy value", policy_eval.value, batch_size)
    if policy_eval.entropy is not None:
        _require_vector("policy entropy", policy_eval.entropy, batch_size)
    if policy_eval.mean is not None and tuple(policy_eval.mean.shape) != (batch_size, 6):
        raise ValueError(f"policy mean must have shape [B, 6], got {tuple(policy_eval.mean.shape)}")
    if policy_eval.sigma is not None and tuple(policy_eval.sigma.shape) != (batch_size, 6):
        raise ValueError(f"policy sigma must have shape [B, 6], got {tuple(policy_eval.sigma.shape)}")
    if policy_eval.raw_actions is not None and tuple(policy_eval.raw_actions.shape) != (batch_size, 6):
        raise ValueError(f"policy raw_actions must have shape [B, 6], got {tuple(policy_eval.raw_actions.shape)}")
    if policy_eval.log_jacobian_contrib is not None and tuple(policy_eval.log_jacobian_contrib.shape) != (batch_size, 6):
        raise ValueError(
            "policy log_jacobian_contrib must have shape [B, 6], "
            f"got {tuple(policy_eval.log_jacobian_contrib.shape)}"
        )


def _require_vector(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [B], got {tuple(tensor.shape)}")
