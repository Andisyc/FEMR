from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class FrontRESSegmentSample:
    segment_ids: torch.Tensor
    source: tuple[str, ...]
    priority: torch.Tensor
    staleness: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class FrontRESSegmentRolloutEvidence:
    segment_ids: torch.Tensor
    reset_success: torch.Tensor
    score_noisy: torch.Tensor
    score_repaired: torch.Tensor
    score_clean: torch.Tensor
    gain_over_noisy: torch.Tensor
    fall_repaired: torch.Tensor
    contact_consistency: torch.Tensor
    action_norm: torch.Tensor
    valid_reward: torch.Tensor
    horizon_k: torch.Tensor


@dataclass(frozen=True)
class FrontRESSegmentSamplerStats:
    replay_pool_size: int
    review_pool_size: int
    invalid_count: int
    seen_count: int
    priority_mean: float
    priority_p90: float
    solved_frac: float
    hopeless_frac: float


class FrontRESSegmentSampler:
    """Prioritized sampler where each level is a motion segment."""

    def __init__(
        self,
        num_segments: int,
        global_frac: float = 0.4,
        replay_frac: float = 0.5,
        review_frac: float = 0.1,
        priority_mode: str = "learning_value",
        staleness_weight: float = 0.1,
        min_replay_score: float = 0.05,
        max_hopeless_replay_frac: float = 0.1,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_segments <= 0:
            raise ValueError(f"num_segments must be positive, got {num_segments}")
        if priority_mode != "learning_value":
            raise ValueError(f"unsupported priority_mode: {priority_mode}")
        if min(global_frac, replay_frac, review_frac) < 0.0:
            raise ValueError("sampling fractions must be non-negative")
        total = global_frac + replay_frac + review_frac
        if total <= 0.0:
            raise ValueError("at least one sampling fraction must be positive")
        self.num_segments = int(num_segments)
        self.global_frac = float(global_frac) / total
        self.replay_frac = float(replay_frac) / total
        self.review_frac = float(review_frac) / total
        self.priority_mode = priority_mode
        self.staleness_weight = float(staleness_weight)
        self.min_replay_score = float(min_replay_score)
        self.max_hopeless_replay_frac = float(max_hopeless_replay_frac)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

        self.priority = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.staleness = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.seen = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
        self.solved = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
        self.hopeless = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
        self.invalid = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
        self.invalid_reasons: dict[int, str] = {}

    def sample(self, batch_size: int) -> FrontRESSegmentSample:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        valid_ids = self._valid_ids()
        if valid_ids.numel() == 0:
            raise RuntimeError("no valid segments are available")

        ids: list[int] = []
        sources: list[str] = []
        for _ in range(batch_size):
            source = self._choose_source()
            segment_id = self._sample_one(source, valid_ids)
            ids.append(segment_id)
            sources.append(source)
            self.seen[segment_id] = True

        segment_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.staleness += 1.0
        self.staleness[segment_ids] = 0.0
        return FrontRESSegmentSample(
            segment_ids=segment_ids,
            source=tuple(sources),
            priority=self.priority[segment_ids].clone(),
            staleness=self.staleness[segment_ids].clone(),
            valid_mask=~self.invalid[segment_ids],
        )

    def update(self, evidence: FrontRESSegmentRolloutEvidence) -> None:
        ids = evidence.segment_ids.to(device=self.device, dtype=torch.long).flatten()
        self._validate_ids(ids)
        useful = self._learning_value(evidence)
        valid = evidence.reset_success.to(self.device).bool() & evidence.valid_reward.to(self.device).bool()
        fall = evidence.fall_repaired.to(self.device).bool()
        gain = evidence.gain_over_noisy.to(self.device).float()
        repaired = evidence.score_repaired.to(self.device).float()
        noisy = evidence.score_noisy.to(self.device).float()

        current = self.priority[ids]
        self.priority[ids] = torch.where(valid, 0.8 * current + 0.2 * useful, current)
        self.seen[ids] = True
        self.solved[ids] = valid & (repaired >= 0.9) & (gain.abs() < self.min_replay_score)
        self.hopeless[ids] = (~valid) | (fall & (gain <= 0.0)) | ((noisy < 0.2) & (repaired < 0.2) & (gain <= 0.0))
        self.priority[ids] = torch.where(self.solved[ids] | self.hopeless[ids], self.priority[ids] * 0.25, self.priority[ids])

    def mark_invalid(self, segment_ids: Iterable[int] | torch.Tensor, reason: str) -> None:
        ids = self._ids_tensor(segment_ids)
        self.invalid[ids] = True
        for segment_id in ids.tolist():
            self.invalid_reasons[int(segment_id)] = reason

    def stats(self) -> FrontRESSegmentSamplerStats:
        valid = ~self.invalid
        valid_count = max(1, int(valid.sum().item()))
        replay_pool = valid & (~self.solved) & (~self.hopeless) & (self.priority >= self.min_replay_score)
        review_pool = valid & self.solved
        priority_valid = self.priority[valid]
        p90 = float(torch.quantile(priority_valid, 0.9).item()) if priority_valid.numel() > 0 else 0.0
        return FrontRESSegmentSamplerStats(
            replay_pool_size=int(replay_pool.sum().item()),
            review_pool_size=int(review_pool.sum().item()),
            invalid_count=int(self.invalid.sum().item()),
            seen_count=int(self.seen.sum().item()),
            priority_mean=float(priority_valid.mean().item()) if priority_valid.numel() > 0 else 0.0,
            priority_p90=p90,
            solved_frac=float((self.solved & valid).sum().item()) / valid_count,
            hopeless_frac=float((self.hopeless & valid).sum().item()) / valid_count,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority.cpu(),
            "staleness": self.staleness.cpu(),
            "seen": self.seen.cpu(),
            "solved": self.solved.cpu(),
            "hopeless": self.hopeless.cpu(),
            "invalid": self.invalid.cpu(),
            "invalid_reasons": dict(self.invalid_reasons),
            "fractions": (self.global_frac, self.replay_frac, self.review_frac),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("priority", "staleness", "seen", "solved", "hopeless", "invalid"):
            value = state[name].to(self.device)
            if value.numel() != self.num_segments:
                raise ValueError(f"{name} size mismatch: {value.numel()} != {self.num_segments}")
            setattr(self, name, value.clone())
        self.invalid_reasons = {int(k): str(v) for k, v in state.get("invalid_reasons", {}).items()}

    def _choose_source(self) -> str:
        draw = float(torch.rand((), generator=self.generator, device=self.device).item())
        if draw < self.global_frac:
            return "global"
        if draw < self.global_frac + self.replay_frac:
            return "replay"
        return "review"

    def _sample_one(self, source: str, valid_ids: torch.Tensor) -> int:
        if source == "replay":
            pool = self._replay_ids()
            if pool.numel() > 0:
                weights = self._sample_weights(pool)
                return int(pool[torch.multinomial(weights, 1, generator=self.generator).item()].item())
            source = "global"
        if source == "review":
            pool = torch.nonzero((~self.invalid) & self.solved, as_tuple=False).flatten()
            if pool.numel() > 0:
                weights = self._sample_weights(pool)
                return int(pool[torch.multinomial(weights, 1, generator=self.generator).item()].item())
            source = "global"
        unseen = valid_ids[~self.seen[valid_ids]]
        pool = unseen if unseen.numel() > 0 else valid_ids
        index = torch.randint(0, pool.numel(), (1,), generator=self.generator, device=self.device)
        return int(pool[index].item())

    def _sample_weights(self, ids: torch.Tensor) -> torch.Tensor:
        weights = self.priority[ids].clamp_min(0.0) + self.staleness_weight * self.staleness[ids].clamp_min(0.0)
        if torch.sum(weights) <= 0.0:
            weights = torch.ones_like(weights)
        return weights / torch.sum(weights)

    def _replay_ids(self) -> torch.Tensor:
        base = (~self.invalid) & (~self.solved) & (self.priority >= self.min_replay_score)
        normal = torch.nonzero(base & (~self.hopeless), as_tuple=False).flatten()
        hopeless = torch.nonzero(base & self.hopeless, as_tuple=False).flatten()
        if hopeless.numel() == 0:
            return normal
        max_hopeless = int(max(0, round(self.max_hopeless_replay_frac * max(1, normal.numel()))))
        if max_hopeless <= 0:
            return normal
        return torch.cat([normal, hopeless[:max_hopeless]], dim=0)

    def _learning_value(self, evidence: FrontRESSegmentRolloutEvidence) -> torch.Tensor:
        gain = evidence.gain_over_noisy.to(self.device).float()
        reset = evidence.reset_success.to(self.device).float()
        valid = evidence.valid_reward.to(self.device).float()
        contact = evidence.contact_consistency.to(self.device).float().clamp(0.0, 1.0)
        fall = evidence.fall_repaired.to(self.device).float()
        repaired = evidence.score_repaired.to(self.device).float()
        noisy = evidence.score_noisy.to(self.device).float()
        need = (1.0 - noisy).clamp(0.0, 1.0)
        unsolved = (1.0 - repaired).clamp(0.0, 1.0)
        improvement = gain.clamp_min(0.0)
        return reset * valid * contact * (1.0 - fall) * (improvement + 0.25 * need * unsolved)

    def _valid_ids(self) -> torch.Tensor:
        return torch.nonzero(~self.invalid, as_tuple=False).flatten()

    def _ids_tensor(self, segment_ids: Iterable[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(segment_ids, torch.Tensor):
            ids = segment_ids.to(device=self.device, dtype=torch.long).flatten()
        else:
            ids = torch.tensor(list(segment_ids), dtype=torch.long, device=self.device)
        self._validate_ids(ids)
        return ids

    def _validate_ids(self, ids: torch.Tensor) -> None:
        if torch.any(ids < 0) or torch.any(ids >= self.num_segments):
            raise KeyError(f"segment ids out of range: {ids.tolist()}")
