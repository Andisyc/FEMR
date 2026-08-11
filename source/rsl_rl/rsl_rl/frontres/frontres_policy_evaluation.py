"""Framework-free FrontRES policy-evaluation owner shared by train and audit.

This module must stay importable without ``rsl_rl.runners``, IsaacLab, or Omni.
It owns only the production 6D policy-evaluation projection consumed by PPO;
simulator lifecycle and optimizer/update orchestration remain outside.
"""

from __future__ import annotations

from typing import Any

import torch


def _should_print_once_or_verbose(owner: Any, flag_name: str) -> bool:
    if bool(getattr(owner, "frontres_segment_verbose_probe", False)):
        return True
    if bool(getattr(owner, flag_name, False)):
        return False
    setattr(owner, flag_name, True)
    return True


class FrontRESSegmentLivePolicyAdapter:
    """Project the production FrontRES policy into the Segment-PPO interface."""

    def __init__(self, alg: Any, privileged_observations: torch.Tensor | None):
        self.alg = alg
        self.privileged_observations = privileged_observations

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if bool(getattr(self.alg, "use_estimate_ref_vel", False)):
            raise NotImplementedError(
                "FrontRES Segment single-update sentinel does not yet store ref_vel_estimator observations."
            )
        self.alg.policy.act(observations)
        value_obs = self.privileged_observations if self.privileged_observations is not None else observations
        if actions.ndim != 2 or actions.shape[-1] != 6:
            raise ValueError(f"Segment PPO policy evaluation requires 6D Delta SE actions, got {tuple(actions.shape)}")
        action_mean = getattr(self.alg.policy, "action_mean", None)
        action_std = getattr(self.alg.policy, "action_std", None)
        mean_6d = None
        std_6d = None
        raw_actions = None
        log_jacobian_contrib = None
        if action_mean is not None and tuple(action_mean.shape) == tuple(actions.shape):
            mean_6d = action_mean
        if action_std is not None and tuple(action_std.shape) == tuple(actions.shape):
            std_6d = action_std
        distribution = getattr(self.alg.policy, "distribution", None)
        if (
            distribution is not None
            and hasattr(distribution, "mean")
            and distribution.mean.ndim == 2
            and tuple(distribution.mean.shape) == tuple(actions.shape)
        ):
            logprob_parts = _segment_delta_se_log_prob_parts(
                self.alg.policy,
                actions,
                distribution.mean,
                distribution.stddev,
            )
            log_prob = logprob_parts["log_prob"]
            raw_actions = logprob_parts["raw_actions"]
            log_jacobian_contrib = logprob_parts["log_jacobian_contrib"]
        else:
            log_prob = _evaluate_segment_delta_se_log_prob(self.alg.policy, actions, alg=self.alg)
        entropy = getattr(self.alg.policy, "entropy", None)
        if callable(entropy):
            entropy = entropy()
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.reshape(-1)
            if entropy.numel() == 1 and actions.shape[0] != 1:
                entropy = entropy.expand(actions.shape[0])
        if _should_print_once_or_verbose(self.alg, "_frontres_segment_ppo_eval_trace_printed"):
            print(
                "[FrontRES Segment PPO Eval Trace] "
                f"batch_action_shape={tuple(actions.shape)} "
                f"policy_action_mean_shape={tuple(action_mean.shape) if action_mean is not None else None} "
                f"eval_mean_shape={tuple(mean_6d.shape) if mean_6d is not None else None} "
                f"log_prob_shape={tuple(log_prob.shape)} "
                f"actor_obs_shape={tuple(observations.shape)} "
                f"critic_obs_shape={tuple(value_obs.shape)} "
                "semantic=ppo_eval_uses_6d_delta_se_with_separate_critic_obs",
                flush=True,
            )
        return {
            "log_prob": log_prob,
            "value": self.alg.policy.evaluate(value_obs).reshape(-1),
            "entropy": entropy if isinstance(entropy, torch.Tensor) else None,
            "mean": mean_6d,
            "sigma": std_6d,
            "raw_actions": raw_actions,
            "log_jacobian_contrib": log_jacobian_contrib,
        }


def _evaluate_segment_delta_se_log_prob(
    policy: Any,
    actions: torch.Tensor,
    *,
    alg: Any | None = None,
) -> torch.Tensor:
    distribution = getattr(policy, "distribution", None)
    if (
        distribution is not None
        and hasattr(distribution, "mean")
        and distribution.mean.ndim == 2
        and distribution.mean.shape[-1] >= 6
    ):
        return _evaluate_segment_delta_se_log_prob_from_stats(policy, actions, distribution.mean, distribution.stddev)
    if alg is not None and hasattr(alg, "_get_actor_log_prob"):
        return alg._get_actor_log_prob(actions).reshape(-1)
    if hasattr(policy, "get_actions_log_prob"):
        return policy.get_actions_log_prob(actions).reshape(-1)
    raise TypeError("policy must expose distribution or get_actions_log_prob for Segment PPO evaluation")


def _evaluate_segment_delta_se_log_prob_from_stats(
    policy: Any,
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return _segment_delta_se_log_prob_parts(policy, actions, mean, std)["log_prob"]


def _segment_delta_se_log_prob_parts(
    policy: Any,
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if actions.ndim != 2 or tuple(mean.shape) != tuple(actions.shape) or tuple(std.shape) != tuple(actions.shape):
        raise ValueError(
            "direct Delta SE log-prob requires matching [B,6] action/mean/std tensors, "
            f"got action={tuple(actions.shape)} mean={tuple(mean.shape)} std={tuple(std.shape)}"
        )
    if int(actions.shape[-1]) != 6:
        raise ValueError(f"direct Delta SE log-prob requires exactly 6 dimensions, got {tuple(actions.shape)}")
    mean_6d = mean.to(device=actions.device, dtype=actions.dtype)
    std_6d = std.to(device=actions.device, dtype=actions.dtype)
    log_prob_dim = torch.distributions.Normal(mean_6d, std_6d).log_prob(actions)
    return {
        "log_prob": log_prob_dim.sum(dim=-1),
        "raw_actions": actions,
        "log_jacobian_contrib": torch.zeros_like(actions),
    }


evaluate_segment_delta_se_log_prob_from_stats = _evaluate_segment_delta_se_log_prob_from_stats


__all__ = [
    "FrontRESSegmentLivePolicyAdapter",
    "evaluate_segment_delta_se_log_prob_from_stats",
]
