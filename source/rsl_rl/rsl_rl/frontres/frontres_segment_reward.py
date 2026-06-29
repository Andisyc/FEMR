from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class FrontRESSegmentScoreWindow:
    score: torch.Tensor
    fall_flag: torch.Tensor | None = None
    contact_consistency: torch.Tensor | None = None
    full_env_reward: torch.Tensor | None = None


@dataclass(frozen=True)
class FrontRESSegmentRewardResult:
    reward: torch.Tensor
    score_noisy: torch.Tensor
    score_repaired: torch.Tensor
    score_clean: torch.Tensor
    gain_over_noisy: torch.Tensor
    clean_gap: torch.Tensor
    fall_flag: torch.Tensor
    contact_consistency: torch.Tensor
    valid_mask: torch.Tensor
    solved_mask: torch.Tensor
    hopeless_mask: torch.Tensor
    diagnostics: dict[str, float]


class FrontRESSegmentReward:
    """Noisy-relative K-step recovery reward for Segment Replay HRL."""

    def __init__(
        self,
        score_fn: Callable[[Any, str], FrontRESSegmentScoreWindow | torch.Tensor] | None = None,
        gain_weight: float = 1.0,
        fall_penalty: float = 1.0,
        contact_weight: float = 0.1,
        valid_score_bounds: tuple[float, float] = (-1.0, 1.0),
        solved_score: float = 0.9,
        hopeless_score: float = 0.2,
        use_full_env_reward: bool = False,
        full_env_weight: float = 1.0,
        evidence_type: Callable[..., Any] | None = None,
    ) -> None:
        self.score_fn = score_fn
        self.gain_weight = float(gain_weight)
        self.fall_penalty = float(fall_penalty)
        self.contact_weight = float(contact_weight)
        self.valid_score_bounds = valid_score_bounds
        self.solved_score = float(solved_score)
        self.hopeless_score = float(hopeless_score)
        self.use_full_env_reward = bool(use_full_env_reward)
        self.full_env_weight = float(full_env_weight)
        self.evidence_type = evidence_type

    def score_window(self, rollout: Any, role: str) -> FrontRESSegmentScoreWindow:
        if self.score_fn is not None:
            return self._as_window(self.score_fn(rollout, role))
        if isinstance(rollout, FrontRESSegmentScoreWindow):
            return rollout
        if isinstance(rollout, torch.Tensor):
            return FrontRESSegmentScoreWindow(score=rollout)
        if isinstance(rollout, dict):
            score = rollout.get(f"score_{role}", rollout.get("score"))
            if score is None:
                raise KeyError(f"rollout dict must contain score_{role} or score")
            return FrontRESSegmentScoreWindow(
                score=score,
                fall_flag=rollout.get(f"fall_{role}", rollout.get("fall_flag")),
                contact_consistency=rollout.get(f"contact_{role}", rollout.get("contact_consistency")),
                full_env_reward=rollout.get(f"full_env_reward_{role}", rollout.get("full_env_reward")),
            )
        raise TypeError(f"unsupported rollout type for score_window: {type(rollout)!r}")

    def compute(
        self,
        noisy: FrontRESSegmentScoreWindow | torch.Tensor | dict[str, torch.Tensor],
        repaired: FrontRESSegmentScoreWindow | torch.Tensor | dict[str, torch.Tensor],
        clean: FrontRESSegmentScoreWindow | torch.Tensor | dict[str, torch.Tensor],
        reset_result: Any | None = None,
    ) -> FrontRESSegmentRewardResult:
        noisy_w = self.score_window(noisy, "noisy")
        repaired_w = self.score_window(repaired, "repaired")
        clean_w = self.score_window(clean, "clean")
        score_noisy = noisy_w.score.float()
        score_repaired = repaired_w.score.to(score_noisy.device).float()
        score_clean = clean_w.score.to(score_noisy.device).float()
        gain = score_repaired - score_noisy
        clean_gap = score_clean - score_repaired
        contact = self._optional_float(repaired_w.contact_consistency, score_noisy, default=1.0).clamp(0.0, 1.0)
        fall = self._optional_bool(repaired_w.fall_flag, score_noisy, default=False)
        valid = self._score_valid(score_noisy) & self._score_valid(score_repaired) & self._score_valid(score_clean)
        if reset_result is not None:
            reset_mask = getattr(reset_result, "success_mask", getattr(reset_result, "valid_mask", None))
            if reset_mask is not None:
                valid = valid & reset_mask.to(score_noisy.device).bool()
        reward = self.gain_weight * gain
        reward = reward + self.contact_weight * (contact - 1.0)
        reward = reward - self.fall_penalty * fall.float()
        if self.use_full_env_reward and repaired_w.full_env_reward is not None:
            reward = reward + self.full_env_weight * repaired_w.full_env_reward.to(score_noisy.device).float()
        reward = torch.where(valid, reward, torch.zeros_like(reward))
        solved = valid & (score_repaired >= self.solved_score) & (gain.abs() <= 0.05)
        hopeless = (~valid) | (fall & (gain <= 0.0)) | (
            (score_noisy <= self.hopeless_score) & (score_repaired <= self.hopeless_score) & (gain <= 0.0)
        )
        diagnostics = {
            "reward_mean": float(reward.mean().item()) if reward.numel() else 0.0,
            "gain_mean": float(gain.mean().item()) if gain.numel() else 0.0,
            "valid_frac": float(valid.float().mean().item()) if valid.numel() else 0.0,
            "solved_frac": float(solved.float().mean().item()) if solved.numel() else 0.0,
            "hopeless_frac": float(hopeless.float().mean().item()) if hopeless.numel() else 0.0,
            "learning_value_mean": float(self._learning_value(gain, score_noisy, score_repaired, contact, fall, valid).mean().item())
            if gain.numel()
            else 0.0,
        }
        return FrontRESSegmentRewardResult(
            reward=reward,
            score_noisy=score_noisy,
            score_repaired=score_repaired,
            score_clean=score_clean,
            gain_over_noisy=gain,
            clean_gap=clean_gap,
            fall_flag=fall,
            contact_consistency=contact,
            valid_mask=valid,
            solved_mask=solved,
            hopeless_mask=hopeless,
            diagnostics=diagnostics,
        )

    def priority_evidence(
        self,
        result: FrontRESSegmentRewardResult,
        segment_ids: torch.Tensor,
        horizon_k: torch.Tensor | int,
        evidence_type: Callable[..., Any] | None = None,
    ) -> Any:
        evidence_cls = evidence_type or self.evidence_type
        if evidence_cls is None:
            raise ValueError("priority_evidence requires evidence_type or constructor evidence_type")
        if isinstance(horizon_k, int):
            horizon = torch.full_like(segment_ids, int(horizon_k), dtype=torch.long)
        else:
            horizon = horizon_k.to(segment_ids.device, dtype=torch.long)
        return evidence_cls(
            segment_ids=segment_ids,
            reset_success=result.valid_mask,
            score_noisy=result.score_noisy,
            score_repaired=result.score_repaired,
            score_clean=result.score_clean,
            gain_over_noisy=result.gain_over_noisy,
            fall_repaired=result.fall_flag,
            contact_consistency=result.contact_consistency,
            action_norm=torch.zeros_like(result.score_noisy),
            valid_reward=result.valid_mask,
            horizon_k=horizon,
        )

    def _as_window(self, value: FrontRESSegmentScoreWindow | torch.Tensor) -> FrontRESSegmentScoreWindow:
        if isinstance(value, FrontRESSegmentScoreWindow):
            return value
        if isinstance(value, torch.Tensor):
            return FrontRESSegmentScoreWindow(score=value)
        raise TypeError(f"score_fn returned unsupported type: {type(value)!r}")

    def _score_valid(self, score: torch.Tensor) -> torch.Tensor:
        low, high = self.valid_score_bounds
        return torch.isfinite(score) & (score >= float(low)) & (score <= float(high))

    def _optional_float(self, value: torch.Tensor | None, like: torch.Tensor, default: float) -> torch.Tensor:
        if value is None:
            return torch.full_like(like, float(default))
        return value.to(like.device).float()

    def _optional_bool(self, value: torch.Tensor | None, like: torch.Tensor, default: bool) -> torch.Tensor:
        if value is None:
            return torch.full(like.shape, bool(default), device=like.device, dtype=torch.bool)
        return value.to(like.device).bool()

    def _learning_value(
        self,
        gain: torch.Tensor,
        score_noisy: torch.Tensor,
        score_repaired: torch.Tensor,
        contact: torch.Tensor,
        fall: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        need = (1.0 - score_noisy).clamp(0.0, 1.0)
        unsolved = (1.0 - score_repaired).clamp(0.0, 1.0)
        solved = (score_repaired >= self.solved_score) & (gain.abs() <= 0.05)
        hopeless = (score_noisy <= self.hopeless_score) & (score_repaired <= self.hopeless_score) & (gain <= 0.0)
        useful = valid & (~solved) & (~hopeless)
        return useful.float() * contact * (1.0 - fall.float()) * (gain.clamp_min(0.0) + 0.25 * need * unsolved)
