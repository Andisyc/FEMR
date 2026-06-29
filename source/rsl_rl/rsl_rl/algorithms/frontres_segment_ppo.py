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


@dataclass(frozen=True)
class FrontRESSegmentPolicyEval:
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor | None = None
    mean: torch.Tensor | None = None
    sigma: torch.Tensor | None = None


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
        }


def compute_frontres_segment_ppo_loss(
    policy: Any,
    batch: FrontRESSegmentPPOBatch,
    cfg: FrontRESSegmentPPOConfig | None = None,
) -> FrontRESSegmentPPOResult:
    cfg = FrontRESSegmentPPOConfig() if cfg is None else cfg
    _validate_batch(batch)
    policy_eval = _evaluate_policy(policy, batch)
    _validate_policy_eval(policy_eval, batch)

    valid = batch.valid_mask.bool()
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
    if cfg.normalize_advantages and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    ratio = torch.exp(log_prob - old_log_prob)
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
        clip_frac = ((ratio - 1.0).abs() > cfg.clip_param).float().mean().item()
        approx_kl = (old_log_prob - log_prob).mean().item()
        ratio_mean = ratio.mean().item()

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
    )


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
        )
    raise TypeError(f"unsupported policy evaluation output: {type(value)!r}")


def _masked_entropy(entropy: torch.Tensor | None, valid: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if entropy is None:
        return like.new_zeros(())
    return entropy[valid].mean()


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


def _require_vector(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [B], got {tuple(tensor.shape)}")
