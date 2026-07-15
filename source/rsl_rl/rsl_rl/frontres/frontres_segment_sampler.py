from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import importlib.util
from pathlib import Path
from typing import Any, Iterable

import torch

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_sampler",
    Path(__file__).resolve().with_name("frontres_formal_runtime_probe.py"),
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


class FrontRESSegmentState(IntEnum):
    """Segment replay budget state owned by the sampler."""

    UNKNOWN = 0
    PROMISING = 1
    FRONTIER = 2
    DELAYED_REGRET = 3
    SOLVED = 4
    HOPELESS = 5


SEGMENT_STATE_NAMES = tuple(state.name.lower() for state in FrontRESSegmentState)


@dataclass(frozen=True)
class FrontRESSegmentSample:
    segment_ids: torch.Tensor
    source: tuple[str, ...]
    priority: torch.Tensor
    staleness: torch.Tensor
    valid_mask: torch.Tensor
    segment_state: torch.Tensor | None = None
    rollout_trial_count: torch.Tensor | None = None
    horizon_k: torch.Tensor | None = None
    budget_reason: tuple[str, ...] = ()
    trial_role: tuple[str, ...] = ()
    source_index: torch.Tensor | None = None
    trial_index: torch.Tensor | None = None


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
    gain_total: torch.Tensor | None = None
    gain_style: torch.Tensor | None = None
    gain_physics: torch.Tensor | None = None
    repair_cost: torch.Tensor | None = None
    gain_source: str = "legacy"


@dataclass(frozen=True)
class FrontRESSegmentTrialEvidence:
    segment_ids: torch.Tensor
    trial_count: torch.Tensor
    valid_trial_count: torch.Tensor
    policy_gain: torch.Tensor
    best_gain: torch.Tensor
    mean_gain: torch.Tensor
    success_frac: torch.Tensor
    fall_frac: torch.Tensor
    oracle_gap: torch.Tensor
    confidence: torch.Tensor
    score_noisy: torch.Tensor
    score_repaired: torch.Tensor
    horizon_k: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class FrontRESSegmentRolloutBudget:
    segment_ids: torch.Tensor
    trial_count: torch.Tensor
    horizon_k: torch.Tensor
    segment_state: torch.Tensor
    reason: tuple[str, ...]


@dataclass(frozen=True)
class FrontRESSegmentTrialPlan:
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor
    trial_role: tuple[str, ...]
    base_segment_ids: torch.Tensor
    base_trial_count: torch.Tensor


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
    unknown_count: int = 0
    promising_count: int = 0
    frontier_count: int = 0
    delayed_regret_count: int = 0
    solved_count: int = 0
    hopeless_count: int = 0
    mean_trial_count: float = 0.0
    oracle_gap_mean: float = 0.0
    confidence_mean: float = 0.0


@dataclass(frozen=True)
class FrontRESSegmentSamplerUpdateProbe:
    count: int
    valid_count: int
    fall_count: int
    gain_mean: float
    gain_pos_frac: float
    useful_mean: float
    useful_max: float
    priority_before_mean: float
    priority_after_mean: float
    priority_after_max: float
    replay_candidate_count: int
    hopeless_count: int
    delayed_regret_count: int = 0
    segment_count: int = 0
    trial_count: int = 0
    oracle_gap_mean: float = 0.0
    confidence_mean: float = 0.0


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
        self.segment_state = torch.full(
            (self.num_segments,),
            int(FrontRESSegmentState.UNKNOWN),
            dtype=torch.long,
            device=self.device,
        )
        self.evidence_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.valid_evidence_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.success_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.fall_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.best_gain = torch.full((self.num_segments,), -float("inf"), dtype=torch.float32, device=self.device)
        self.best_short_gain = torch.full((self.num_segments,), -float("inf"), dtype=torch.float32, device=self.device)
        self.best_long_gain = torch.full((self.num_segments,), -float("inf"), dtype=torch.float32, device=self.device)
        self.last_horizon_k = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.last_trial_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.last_policy_gain = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_mean_gain = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_success_frac = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_fall_frac = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_oracle_gap = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)
        self.last_confidence = torch.zeros(self.num_segments, dtype=torch.float32, device=self.device)

    def reset_for_deterministic_eval(self, *, seed: int) -> None:
        """Reset replay history before a cross-checkpoint sequence evaluation.

        Checkpoints persist replay frontier state because it is part of training
        resume semantics.  That state must not choose different evaluation
        motions for two checkpoints being compared.  Rebuild only this sampler
        with the same configuration and seed; the policy, normalizer, and
        environment state remain owned by the runner and are untouched.
        """
        fresh = type(self)(
            num_segments=self.num_segments,
            global_frac=self.global_frac,
            replay_frac=self.replay_frac,
            review_frac=self.review_frac,
            priority_mode=self.priority_mode,
            staleness_weight=self.staleness_weight,
            min_replay_score=self.min_replay_score,
            max_hopeless_replay_frac=self.max_hopeless_replay_frac,
            seed=int(seed),
            device=self.device,
        )
        self.__dict__.update(fresh.__dict__)

    def sample(self, batch_size: int, *, max_horizon_k: int = 8) -> FrontRESSegmentSample:
        """按 replay mixture 选择 base segments 并附加 rollout budget.

        函数名说明:
            `sample` 是 base-segment selection owner, 选择 segment 和来源; 它不
            展开多 trial 行, 正式 live row expansion 由 `sample_rollout_rows` 完成.

        主链路:
            上游: runner 给出 base batch size 和最大 K.
            下游: 返回 segment id, source, priority, state 和初始 rollout budget.

        语义:
            sampling source 决定 global/replay/review 混合, segment state 决定后续
            K/trial budget. 两者不能被 PPO post-update diagnostics 污染.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        valid_ids = self._valid_ids()
        if valid_ids.numel() == 0:
            raise RuntimeError("no valid segments are available")

        ids: list[int] = []
        sources: list[str] = []
        for _ in range(batch_size):
            source = self._choose_source()
            segment_id, effective_source = self._sample_one(source, valid_ids)
            ids.append(segment_id)
            sources.append(effective_source)
            self.seen[segment_id] = True

        segment_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.staleness += 1.0
        self.staleness[segment_ids] = 0.0
        budget = self.plan_rollout_budget(segment_ids, max_horizon_k=max_horizon_k)
        return FrontRESSegmentSample(
            segment_ids=segment_ids,
            source=tuple(sources),
            priority=self.priority[segment_ids].clone(),
            staleness=self.staleness[segment_ids].clone(),
            valid_mask=~self.invalid[segment_ids],
            segment_state=self.segment_state[segment_ids].clone(),
            rollout_trial_count=budget.trial_count.clone(),
            horizon_k=budget.horizon_k.clone(),
            budget_reason=budget.reason,
            trial_role=tuple("policy" for _ in ids),
            source_index=torch.arange(int(segment_ids.numel()), dtype=torch.long, device=self.device),
            trial_index=torch.zeros(int(segment_ids.numel()), dtype=torch.long, device=self.device),
        )

    def sample_rollout_rows(self, row_budget: int, *, max_horizon_k: int = 8) -> FrontRESSegmentSample:
        """展开 per-segment trial budget, 生成正式 live rollout rows.

        函数名说明:
            `sample_rollout_rows` 是 live row sampler, 把 base segment 变成固定行数
            的 policy-first trials; 它不是 env reset 或 PPO batch builder.

        主链路:
            上游: live sampler helper 给出 split-env 可用 repair row budget.
            下游: batch builder 按 `source_index/trial_index/trial_role` 构造 reset 和
            rollout metadata.

        语义:
            返回行数服从 env row budget, 每行仍保留原 segment, K 和 trial 身份,
            因而多个 trial 不得被误当成多个独立 segment.
        """
        # B1: 选择 base segments, 直到计划 trial rows 覆盖 live row budget.
        if row_budget <= 0:
            raise ValueError(f"row_budget must be positive, got {row_budget}")
        valid_ids = self._valid_ids()
        if valid_ids.numel() == 0:
            raise RuntimeError("no valid segments are available")

        base_ids: list[int] = []
        base_sources: list[str] = []
        planned_rows = 0
        while planned_rows < row_budget:
            source = self._choose_source()
            segment_id, effective_source = self._sample_one(source, valid_ids)
            base_ids.append(segment_id)
            base_sources.append(effective_source)
            self.seen[segment_id] = True
            budget = self.plan_rollout_budget([segment_id], max_horizon_k=max_horizon_k)
            planned_rows += max(1, int(budget.trial_count[0].item()))

        base_segment_ids = torch.tensor(base_ids, dtype=torch.long, device=self.device)
        plan = self.expand_rollout_trials(base_segment_ids, max_horizon_k=max_horizon_k)
        keep = min(int(row_budget), int(plan.segment_ids.numel()))
        source_index = plan.source_index[:keep].to(device=self.device, dtype=torch.long)
        row_ids = plan.segment_ids[:keep].to(device=self.device, dtype=torch.long)
        self.staleness += 1.0
        self.staleness[torch.unique(row_ids)] = 0.0
        base_budget = self.plan_rollout_budget(base_segment_ids, max_horizon_k=max_horizon_k)
        source_rows = source_index.detach().cpu().tolist()
        # B2: 物化带 source, K, role 和 trial identity 的 row-level sample.
        sample = FrontRESSegmentSample(
            segment_ids=row_ids.detach().clone(),
            source=tuple(str(base_sources[int(row)]) for row in source_rows),
            priority=self.priority[row_ids].detach().clone(),
            staleness=self.staleness[row_ids].detach().clone(),
            valid_mask=~self.invalid[row_ids],
            segment_state=self.segment_state[row_ids].detach().clone(),
            rollout_trial_count=base_budget.trial_count[source_index].detach().clone(),
            horizon_k=plan.horizon_k[:keep].detach().clone(),
            budget_reason=tuple(str(base_budget.reason[int(row)]) for row in source_rows),
            trial_role=tuple(plan.trial_role[:keep]),
            source_index=source_index.detach().clone(),
            trial_index=plan.trial_index[:keep].to(device=self.device, dtype=torch.long).detach().clone(),
        )
        # B3: AUDIT-SAMPLER-01 截获 live batch builder 实际消费的 sample.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-SAMPLER-01",
            segment_ids=sample.segment_ids,
            source=sample.source,
            horizon_k=sample.horizon_k,
            trial_role=sample.trial_role,
        )
        return sample

    def update(self, evidence: FrontRESSegmentRolloutEvidence) -> None:
        self.update_with_probe(evidence)

    def update_with_probe(self, evidence: FrontRESSegmentRolloutEvidence) -> FrontRESSegmentSamplerUpdateProbe:
        """用 rollout-time evidence 更新 segment replay state 和 priority.

        函数名说明:
            `update_with_probe` 是 sampler state transaction owner, 同时返回可读的
            update probe; 它不是 PPO update, 也不读取 post-update KL 或梯度.

        主链路:
            上游: live probe 提交带 segment/trial identity 的 paired rollout evidence.
            下游: 更新 priority, solved/hopeless/state 和 curriculum history, 供下一次
            sample/K planning 使用.

        语义:
            更新依据必须来自 policy update 前的 rollout evidence. 多 trial 先按
            segment 聚合, 再改变持久 replay state.
        """
        # B1: 改变 replay state 前, 先按 segment 聚合 rollout-time evidence.
        row_ids = evidence.segment_ids.to(device=self.device, dtype=torch.long).flatten()
        self._validate_ids(row_ids)
        trial = self.aggregate_trial_evidence(evidence)
        ids = trial.segment_ids
        useful_rows = self._learning_value(evidence)
        useful = self._mean_by_ids(row_ids, useful_rows, ids)
        valid = trial.valid_mask
        fall_count = torch.round(trial.fall_frac * trial.trial_count.float()).long()

        current = self.priority[ids]
        self.priority[ids] = torch.where(valid, 0.8 * current + 0.2 * useful, current)
        self.seen[ids] = True
        self._update_segment_state_from_trials(trial)
        self.priority[ids] = torch.where(self.solved[ids] | self.hopeless[ids], self.priority[ids] * 0.25, self.priority[ids])
        priority_after = self.priority[ids]
        replay_candidates = (~self.invalid[ids]) & (~self.solved[ids]) & (~self.hopeless[ids]) & (priority_after >= self.min_replay_score)
        # B2: 将 rollout-time evidence 写入 priority 和持久 segment state.
        update_probe = FrontRESSegmentSamplerUpdateProbe(
            count=int(row_ids.numel()),
            valid_count=int(trial.valid_trial_count.sum().item()),
            fall_count=int(fall_count.sum().item()),
            gain_mean=float(self._active_gain(evidence).mean().item()) if row_ids.numel() > 0 else 0.0,
            gain_pos_frac=float((self._active_gain(evidence) > 0.0).float().mean().item()) if row_ids.numel() > 0 else 0.0,
            useful_mean=float(useful.mean().item()) if useful.numel() > 0 else 0.0,
            useful_max=float(useful.max().item()) if useful.numel() > 0 else 0.0,
            priority_before_mean=float(current.mean().item()) if current.numel() > 0 else 0.0,
            priority_after_mean=float(priority_after.mean().item()) if priority_after.numel() > 0 else 0.0,
            priority_after_max=float(priority_after.max().item()) if priority_after.numel() > 0 else 0.0,
            replay_candidate_count=int(replay_candidates.sum().item()),
            hopeless_count=int(self.hopeless[ids].sum().item()),
            delayed_regret_count=int((self.segment_state[ids] == int(FrontRESSegmentState.DELAYED_REGRET)).sum().item()),
            segment_count=int(ids.numel()),
            trial_count=int(trial.trial_count.sum().item()),
            oracle_gap_mean=float(trial.oracle_gap.mean().item()) if trial.oracle_gap.numel() > 0 else 0.0,
            confidence_mean=float(trial.confidence.mean().item()) if trial.confidence.numel() > 0 else 0.0,
        )
        # B3: AUDIT-SAMPLER-01 同步截获该 transaction 完成后的 priority state.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-SAMPLER-01",
            priority_before=update_probe.priority_before_mean,
            priority_after=update_probe.priority_after_mean,
            valid_count=update_probe.valid_count,
            trial_count=update_probe.trial_count,
        )
        return update_probe

    def aggregate_trial_evidence(self, evidence: FrontRESSegmentRolloutEvidence) -> FrontRESSegmentTrialEvidence:
        ids = evidence.segment_ids.to(device=self.device, dtype=torch.long).flatten()
        self._validate_ids(ids)
        unique_ids = torch.unique(ids, sorted=True)
        gain = self._active_gain(evidence)
        valid = evidence.reset_success.to(self.device).bool().flatten() & evidence.valid_reward.to(self.device).bool().flatten()
        fall = evidence.fall_repaired.to(self.device).bool().flatten() | (~valid)
        horizon = evidence.horizon_k.to(self.device).long().flatten()

        trial_count: list[int] = []
        valid_trial_count: list[int] = []
        policy_gain: list[float] = []
        best_gain: list[float] = []
        mean_gain: list[float] = []
        success_frac: list[float] = []
        fall_frac: list[float] = []
        oracle_gap: list[float] = []
        confidence: list[float] = []
        horizon_k: list[int] = []
        valid_mask: list[bool] = []

        for segment_id in unique_ids.tolist():
            mask = ids == int(segment_id)
            trial_n = int(mask.sum().item())
            row_gain = gain[mask]
            row_valid = valid[mask]
            row_fall = fall[mask]
            row_horizon = horizon[mask]
            valid_gain = row_gain[row_valid]
            valid_n = int(row_valid.sum().item())
            policy = float(row_gain[0].item()) if trial_n else 0.0
            best = float(valid_gain.max().item()) if valid_n else 0.0
            mean = float(valid_gain.mean().item()) if valid_n else 0.0
            success = (row_valid & (~row_fall) & (row_gain > self.min_replay_score)).float()
            fall_or_invalid = row_fall.float()
            gap = max(0.0, best - policy)
            fall_rate = float(fall_or_invalid.mean().item()) if trial_n else 0.0
            conf = min(1.0, float(valid_n) / 3.0) * max(0.0, 1.0 - fall_rate)

            trial_count.append(trial_n)
            valid_trial_count.append(valid_n)
            policy_gain.append(policy)
            best_gain.append(best)
            mean_gain.append(mean)
            success_frac.append(float(success.mean().item()) if trial_n else 0.0)
            fall_frac.append(fall_rate)
            oracle_gap.append(gap)
            confidence.append(conf)
            horizon_k.append(int(row_horizon.max().item()) if trial_n else 0)
            valid_mask.append(valid_n > 0)

        return FrontRESSegmentTrialEvidence(
            segment_ids=unique_ids,
            trial_count=torch.tensor(trial_count, dtype=torch.long, device=self.device),
            valid_trial_count=torch.tensor(valid_trial_count, dtype=torch.long, device=self.device),
            policy_gain=torch.tensor(policy_gain, dtype=torch.float32, device=self.device),
            best_gain=torch.tensor(best_gain, dtype=torch.float32, device=self.device),
            mean_gain=torch.tensor(mean_gain, dtype=torch.float32, device=self.device),
            success_frac=torch.tensor(success_frac, dtype=torch.float32, device=self.device),
            fall_frac=torch.tensor(fall_frac, dtype=torch.float32, device=self.device),
            oracle_gap=torch.tensor(oracle_gap, dtype=torch.float32, device=self.device),
            confidence=torch.tensor(confidence, dtype=torch.float32, device=self.device),
            score_noisy=torch.full((len(trial_count),), float("nan"), dtype=torch.float32, device=self.device),
            score_repaired=torch.full((len(trial_count),), float("nan"), dtype=torch.float32, device=self.device),
            horizon_k=torch.tensor(horizon_k, dtype=torch.long, device=self.device),
            valid_mask=torch.tensor(valid_mask, dtype=torch.bool, device=self.device),
        )

    def plan_rollout_budget(
        self,
        segment_ids: Iterable[int] | torch.Tensor,
        *,
        max_horizon_k: int = 8,
    ) -> FrontRESSegmentRolloutBudget:
        """把 segment state 映射为纯 K-step rollout budget.

        函数名说明:
            `plan_rollout_budget` 是 K curriculum 的 pure planner, 只计算 horizon K,
            trial count 和 reason; 它不触碰 env, storage 或 PPO.

        主链路:
            上游: sampler 提供选中 segment 及其持久 state/history.
            下游: `expand_rollout_trials` 和 live batch builder 消费不可变 budget.

        语义:
            K 表示本次修复证据需要持续观察的时间窗. state 越接近 delayed regret,
            越需要更长 horizon 或更多 trials, 但不得超过正式 max_horizon_k.
        """
        # B1: 读取拥有 curriculum progression 的持久 segment state.
        ids = self._ids_tensor(segment_ids)
        max_horizon = int(max_horizon_k)
        if max_horizon <= 0:
            raise ValueError(f"max_horizon_k must be positive, got {max_horizon_k}")
        states = self.segment_state[ids].clone()
        trial_count = torch.ones_like(ids, dtype=torch.long, device=self.device)
        horizon_k = torch.empty_like(ids, dtype=torch.long, device=self.device)
        reasons: list[str] = []

        for row, segment_id in enumerate(ids.tolist()):
            state = FrontRESSegmentState(int(states[row].item()))
            trial_n = 1
            preferred_horizon = 8
            reason = "unknown_probe"
            if state == FrontRESSegmentState.PROMISING:
                trial_n = 3
                preferred_horizon = 16
                reason = "promising_local_trials"
            elif state == FrontRESSegmentState.FRONTIER:
                trial_n = 6
                use_long = (
                    float(self.last_success_frac[segment_id].item()) < 0.75
                    or int(self.last_trial_count[segment_id].item()) >= 2
                )
                preferred_horizon = 32 if use_long else 16
                reason = "frontier_multi_trial"
            elif state == FrontRESSegmentState.DELAYED_REGRET:
                trial_n = 6
                preferred_horizon = 64 if max_horizon >= 64 else 32
                reason = "delayed_regret_long_check"
            elif state == FrontRESSegmentState.SOLVED:
                trial_n = 1
                preferred_horizon = 64
                reason = "solved_review"
            elif state == FrontRESSegmentState.HOPELESS:
                trial_n = 1
                preferred_horizon = 8
                reason = "hopeless_recheck"

            trial_count[row] = int(trial_n)
            horizon_k[row] = self._bounded_horizon(preferred_horizon, max_horizon)
            reasons.append(reason)

        # B2: 物化每个 segment 的不可变 curriculum budget.
        budget = FrontRESSegmentRolloutBudget(
            segment_ids=ids.clone(),
            trial_count=trial_count,
            horizon_k=horizon_k,
            segment_state=states,
            reason=tuple(reasons),
        )
        # B3: AUDIT-KPLAN-01 截获 row expansion 前的 per-segment K 和 trial budget.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-KPLAN-01",
            segment_ids=budget.segment_ids,
            segment_state=budget.segment_state,
            trial_count=budget.trial_count,
            horizon_k=budget.horizon_k,
            reason=budget.reason,
        )
        return budget

    def expand_rollout_trials(
        self,
        segment_ids: Iterable[int] | torch.Tensor,
        *,
        max_horizon_k: int = 8,
    ) -> FrontRESSegmentTrialPlan:
        """Expand per-segment budget into policy-first trial rows for future live wiring."""
        """把 per-segment budget 展开为 policy-first trial rows.

        函数名说明:
            `expand_rollout_trials` 是 K plan 到 row layout 的转换 owner; 它不重新
            规划 K, 也不改变 sampler priority.

        主链路:
            上游: `plan_rollout_budget` 提供 segment-level horizon 和 trial count.
            下游: live sampler/reset 通过 source/trial index 消费 expanded rows.

        语义:
            每个 segment 的第 0 行必须是 policy trial. 后续 probe rows 共享同一 K
            和 source segment, 使短窗和长窗 evidence 能按 trial identity 聚合.
        """
        # B1: 不改变 K, 将一个 budget row 展开为 policy-first trial rows.
        budget = self.plan_rollout_budget(segment_ids, max_horizon_k=max_horizon_k)
        expanded_ids: list[int] = []
        source_index: list[int] = []
        trial_index: list[int] = []
        horizon: list[int] = []
        roles: list[str] = []
        for source_row, segment_id in enumerate(budget.segment_ids.tolist()):
            count = int(budget.trial_count[source_row].item())
            horizon_value = int(budget.horizon_k[source_row].item())
            for trial_row in range(count):
                expanded_ids.append(int(segment_id))
                source_index.append(source_row)
                trial_index.append(trial_row)
                horizon.append(horizon_value)
                roles.append("policy" if trial_row == 0 else "search")
        # B2: 保留 source/trial indexes, 物化 policy-first rows.
        plan = FrontRESSegmentTrialPlan(
            segment_ids=torch.tensor(expanded_ids, dtype=torch.long, device=self.device),
            source_index=torch.tensor(source_index, dtype=torch.long, device=self.device),
            trial_index=torch.tensor(trial_index, dtype=torch.long, device=self.device),
            horizon_k=torch.tensor(horizon, dtype=torch.long, device=self.device),
            trial_role=tuple(roles),
            base_segment_ids=budget.segment_ids.clone(),
            base_trial_count=budget.trial_count.clone(),
        )
        # B3: AUDIT-KROLLOUT-01 截获 reset/rollout 实际消费的 expanded rows.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-KROLLOUT-01",
            segment_ids=plan.segment_ids,
            source_index=plan.source_index,
            trial_index=plan.trial_index,
            horizon_k=plan.horizon_k,
            trial_role=plan.trial_role,
        )
        return plan

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
            unknown_count=self._state_count(FrontRESSegmentState.UNKNOWN, valid),
            promising_count=self._state_count(FrontRESSegmentState.PROMISING, valid),
            frontier_count=self._state_count(FrontRESSegmentState.FRONTIER, valid),
            delayed_regret_count=self._state_count(FrontRESSegmentState.DELAYED_REGRET, valid),
            solved_count=self._state_count(FrontRESSegmentState.SOLVED, valid),
            hopeless_count=self._state_count(FrontRESSegmentState.HOPELESS, valid),
            mean_trial_count=float(self.last_trial_count[valid].float().mean().item()) if valid.any() else 0.0,
            oracle_gap_mean=float(self.last_oracle_gap[valid].mean().item()) if valid.any() else 0.0,
            confidence_mean=float(self.last_confidence[valid].mean().item()) if valid.any() else 0.0,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority.cpu(),
            "staleness": self.staleness.cpu(),
            "seen": self.seen.cpu(),
            "solved": self.solved.cpu(),
            "hopeless": self.hopeless.cpu(),
            "invalid": self.invalid.cpu(),
            "segment_state": self.segment_state.cpu(),
            "evidence_count": self.evidence_count.cpu(),
            "valid_evidence_count": self.valid_evidence_count.cpu(),
            "success_count": self.success_count.cpu(),
            "fall_count": self.fall_count.cpu(),
            "best_gain": self.best_gain.cpu(),
            "best_short_gain": self.best_short_gain.cpu(),
            "best_long_gain": self.best_long_gain.cpu(),
            "last_horizon_k": self.last_horizon_k.cpu(),
            "last_trial_count": self.last_trial_count.cpu(),
            "last_policy_gain": self.last_policy_gain.cpu(),
            "last_mean_gain": self.last_mean_gain.cpu(),
            "last_success_frac": self.last_success_frac.cpu(),
            "last_fall_frac": self.last_fall_frac.cpu(),
            "last_oracle_gap": self.last_oracle_gap.cpu(),
            "last_confidence": self.last_confidence.cpu(),
            "invalid_reasons": dict(self.invalid_reasons),
            "fractions": (self.global_frac, self.replay_frac, self.review_frac),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.priority = self._load_state_tensor(state, "priority", self.priority)
        self.staleness = self._load_state_tensor(state, "staleness", self.staleness)
        self.seen = self._load_state_tensor(state, "seen", self.seen)
        self.solved = self._load_state_tensor(state, "solved", self.solved)
        self.hopeless = self._load_state_tensor(state, "hopeless", self.hopeless)
        self.invalid = self._load_state_tensor(state, "invalid", self.invalid)
        self.evidence_count = self._load_state_tensor(state, "evidence_count", self.seen.long())
        self.valid_evidence_count = self._load_state_tensor(state, "valid_evidence_count", self.valid_evidence_count)
        self.success_count = self._load_state_tensor(state, "success_count", self.success_count)
        self.fall_count = self._load_state_tensor(state, "fall_count", self.fall_count)
        self.best_gain = self._load_state_tensor(state, "best_gain", self.best_gain)
        self.best_short_gain = self._load_state_tensor(state, "best_short_gain", self.best_short_gain)
        self.best_long_gain = self._load_state_tensor(state, "best_long_gain", self.best_long_gain)
        self.last_horizon_k = self._load_state_tensor(state, "last_horizon_k", self.last_horizon_k)
        self.last_trial_count = self._load_state_tensor(state, "last_trial_count", self.last_trial_count)
        self.last_policy_gain = self._load_state_tensor(state, "last_policy_gain", self.last_policy_gain)
        self.last_mean_gain = self._load_state_tensor(state, "last_mean_gain", self.last_mean_gain)
        self.last_success_frac = self._load_state_tensor(state, "last_success_frac", self.last_success_frac)
        self.last_fall_frac = self._load_state_tensor(state, "last_fall_frac", self.last_fall_frac)
        self.last_oracle_gap = self._load_state_tensor(state, "last_oracle_gap", self.last_oracle_gap)
        self.last_confidence = self._load_state_tensor(state, "last_confidence", self.last_confidence)
        if "segment_state" in state:
            self.segment_state = self._load_state_tensor(state, "segment_state", self.segment_state)
            self._validate_segment_state()
            self._sync_terminal_flags_from_state()
        else:
            self._derive_segment_state_from_legacy_flags()
        self.invalid_reasons = {int(k): str(v) for k, v in state.get("invalid_reasons", {}).items()}

    def _choose_source(self) -> str:
        draw = float(torch.rand((), generator=self.generator, device=self.device).item())
        if draw < self.global_frac:
            return "global"
        if draw < self.global_frac + self.replay_frac:
            return "replay"
        return "review"

    def _sample_one(self, source: str, valid_ids: torch.Tensor) -> tuple[int, str]:
        if source == "replay":
            pool = self._replay_ids()
            if pool.numel() > 0:
                weights = self._sample_weights(pool)
                segment_id = int(pool[torch.multinomial(weights, 1, generator=self.generator).item()].item())
                return segment_id, "replay"
            source = "global"
        if source == "review":
            pool = torch.nonzero((~self.invalid) & self.solved, as_tuple=False).flatten()
            if pool.numel() > 0:
                weights = self._sample_weights(pool)
                segment_id = int(pool[torch.multinomial(weights, 1, generator=self.generator).item()].item())
                return segment_id, "review"
            source = "global"
        unseen = valid_ids[~self.seen[valid_ids]]
        pool = unseen if unseen.numel() > 0 else valid_ids
        index = torch.randint(0, pool.numel(), (1,), generator=self.generator, device=self.device)
        return int(pool[index].item()), source

    def _sample_weights(self, ids: torch.Tensor) -> torch.Tensor:
        weights = self.priority[ids].clamp_min(0.0) + self.staleness_weight * self.staleness[ids].clamp_min(0.0)
        if torch.sum(weights) <= 0.0:
            weights = torch.ones_like(weights)
        return weights / torch.sum(weights)

    @staticmethod
    def _bounded_horizon(preferred_horizon: int, max_horizon: int) -> int:
        target = min(int(preferred_horizon), int(max_horizon))
        if target < 8:
            return max(1, target)
        allowed = [8, 16, 32, 64]
        return max(horizon for horizon in allowed if horizon <= target)

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
        gain = self._active_gain(evidence)
        reset = evidence.reset_success.to(self.device).float()
        valid = evidence.valid_reward.to(self.device).float()
        contact = evidence.contact_consistency.to(self.device).float().clamp(0.0, 1.0)
        fall = evidence.fall_repaired.to(self.device).float()
        improvement = gain.clamp_min(0.0)
        return reset * valid * contact * (1.0 - fall) * improvement

    def _active_gain(self, evidence: FrontRESSegmentRolloutEvidence) -> torch.Tensor:
        """Return finite canonical Gain consumed by sampler decisions."""

        gain = evidence.gain_total
        if not isinstance(gain, torch.Tensor):
            raise ValueError("sampler priority/state requires canonical gain_total evidence")
        gain = gain.to(self.device).float().flatten()
        expected = int(evidence.segment_ids.numel())
        if int(gain.numel()) != expected:
            raise ValueError(f"gain_total must have {expected} rows, got {int(gain.numel())}")
        if not bool(torch.isfinite(gain).all().item()):
            raise ValueError("sampler priority/state requires finite gain_total evidence")
        return gain

    def _mean_by_ids(self, ids: torch.Tensor, values: torch.Tensor, unique_ids: torch.Tensor) -> torch.Tensor:
        means = []
        values = values.to(self.device).float().flatten()
        for segment_id in unique_ids.tolist():
            mask = ids == int(segment_id)
            means.append(float(values[mask].mean().item()) if bool(mask.any()) else 0.0)
        return torch.tensor(means, dtype=torch.float32, device=self.device)

    def _update_segment_state_from_trials(self, trial: FrontRESSegmentTrialEvidence) -> None:
        ids = trial.segment_ids
        self.evidence_count[ids] += trial.trial_count
        self.valid_evidence_count[ids] += trial.valid_trial_count
        self.success_count[ids] += torch.round(trial.success_frac * trial.trial_count.float()).long()
        self.fall_count[ids] += torch.round(trial.fall_frac * trial.trial_count.float()).long()
        self.last_horizon_k[ids] = trial.horizon_k
        self.last_trial_count[ids] = trial.trial_count
        self.last_policy_gain[ids] = trial.policy_gain
        self.last_mean_gain[ids] = trial.mean_gain
        self.last_success_frac[ids] = trial.success_frac
        self.last_fall_frac[ids] = trial.fall_frac
        self.last_oracle_gap[ids] = trial.oracle_gap
        self.last_confidence[ids] = trial.confidence

        neg_inf = torch.full_like(trial.best_gain, -float("inf"))
        self._scatter_max(self.best_gain, ids, torch.where(trial.valid_mask, trial.best_gain, neg_inf))
        short_horizon = trial.horizon_k <= 8
        long_horizon = trial.horizon_k >= 16
        self._scatter_max(self.best_short_gain, ids, torch.where(trial.valid_mask & short_horizon, trial.best_gain, neg_inf))
        self._scatter_max(self.best_long_gain, ids, torch.where(trial.valid_mask & long_horizon, trial.best_gain, neg_inf))

        solved = trial.valid_mask & (trial.fall_frac <= 0.0) & (trial.mean_gain.abs() < self.min_replay_score)
        short_positive = self.best_short_gain[ids] > self.min_replay_score
        long_regret = long_horizon & short_positive & ((trial.mean_gain < 0.0) | (trial.fall_frac > 0.0) | (~trial.valid_mask))
        hopeless = (~trial.valid_mask) | (
            (trial.fall_frac >= 0.5) & (trial.best_gain <= 0.0)
        )
        positive = trial.valid_mask & (trial.best_gain > self.min_replay_score)
        frontier = positive & (
            (self.evidence_count[ids] >= 2)
            | ((trial.trial_count >= 2) & (trial.success_frac < 0.75))
        )
        promising = positive | ((self.segment_state[ids] == int(FrontRESSegmentState.PROMISING)) & (~frontier))

        state = self.segment_state[ids].clone()
        state = torch.where(promising, torch.full_like(state, int(FrontRESSegmentState.PROMISING)), state)
        state = torch.where(frontier, torch.full_like(state, int(FrontRESSegmentState.FRONTIER)), state)
        state = torch.where(solved, torch.full_like(state, int(FrontRESSegmentState.SOLVED)), state)
        state = torch.where(hopeless & (~long_regret), torch.full_like(state, int(FrontRESSegmentState.HOPELESS)), state)
        state = torch.where(long_regret, torch.full_like(state, int(FrontRESSegmentState.DELAYED_REGRET)), state)
        self.segment_state[ids] = state
        self._sync_terminal_flags_for_ids(ids)

    def _update_segment_state(
        self,
        ids: torch.Tensor,
        *,
        valid: torch.Tensor,
        fall: torch.Tensor,
        gain: torch.Tensor,
        horizon: torch.Tensor,
    ) -> None:
        if horizon.numel() != ids.numel():
            raise ValueError(f"horizon_k must match segment_ids, got {horizon.numel()} and {ids.numel()}")
        ones = torch.ones_like(ids, dtype=torch.long, device=self.device)
        valid_long = valid.to(self.device).long()
        success = valid & (~fall) & (gain > self.min_replay_score)
        self.evidence_count.scatter_add_(0, ids, ones)
        self.valid_evidence_count.scatter_add_(0, ids, valid_long)
        self.success_count.scatter_add_(0, ids, success.long())
        self.fall_count.scatter_add_(0, ids, fall.long())
        self.last_horizon_k[ids] = horizon

        neg_inf = torch.full_like(gain, -float("inf"))
        self._scatter_max(self.best_gain, ids, torch.where(valid, gain, neg_inf))
        short_horizon = horizon <= 8
        long_horizon = horizon >= 16
        self._scatter_max(self.best_short_gain, ids, torch.where(valid & short_horizon, gain, neg_inf))
        self._scatter_max(self.best_long_gain, ids, torch.where(valid & long_horizon, gain, neg_inf))

        solved = valid & (~fall) & (gain.abs() < self.min_replay_score)
        short_positive = self.best_short_gain[ids] > self.min_replay_score
        long_regret = long_horizon & short_positive & ((gain < 0.0) | fall | (~valid))
        hopeless = (~valid) | (fall & (gain <= 0.0))
        positive = valid & (~fall) & (gain > self.min_replay_score)
        frontier = positive & (self.evidence_count[ids] >= 2)
        promising = positive | ((self.segment_state[ids] == int(FrontRESSegmentState.PROMISING)) & (~frontier))

        state = self.segment_state[ids].clone()
        state = torch.where(promising, torch.full_like(state, int(FrontRESSegmentState.PROMISING)), state)
        state = torch.where(frontier, torch.full_like(state, int(FrontRESSegmentState.FRONTIER)), state)
        state = torch.where(solved, torch.full_like(state, int(FrontRESSegmentState.SOLVED)), state)
        state = torch.where(hopeless & (~long_regret), torch.full_like(state, int(FrontRESSegmentState.HOPELESS)), state)
        state = torch.where(long_regret, torch.full_like(state, int(FrontRESSegmentState.DELAYED_REGRET)), state)
        self.segment_state[ids] = state
        self._sync_terminal_flags_for_ids(ids)

    def _scatter_max(self, target: torch.Tensor, ids: torch.Tensor, values: torch.Tensor) -> None:
        if hasattr(target, "scatter_reduce_"):
            target.scatter_reduce_(0, ids, values.to(target.dtype), reduce="amax", include_self=True)
            return
        for segment_id, value in zip(ids.tolist(), values.tolist()):
            if value > float(target[int(segment_id)].item()):
                target[int(segment_id)] = float(value)

    def _state_count(self, state: FrontRESSegmentState, valid: torch.Tensor) -> int:
        return int(((self.segment_state == int(state)) & valid).sum().item())

    def _load_state_tensor(self, state: dict[str, Any], name: str, default: torch.Tensor) -> torch.Tensor:
        value = state.get(name)
        if value is None:
            return default.clone()
        value = value.to(device=self.device, dtype=default.dtype).flatten()
        if value.numel() != self.num_segments:
            raise ValueError(f"{name} size mismatch: {value.numel()} != {self.num_segments}")
        return value.clone()

    def _validate_segment_state(self) -> None:
        min_state = int(self.segment_state.min().item()) if self.segment_state.numel() else 0
        max_state = int(self.segment_state.max().item()) if self.segment_state.numel() else 0
        if min_state < int(FrontRESSegmentState.UNKNOWN) or max_state > int(FrontRESSegmentState.HOPELESS):
            raise ValueError(f"segment_state contains unsupported ids: min={min_state} max={max_state}")

    def _derive_segment_state_from_legacy_flags(self) -> None:
        self.segment_state = torch.full(
            (self.num_segments,),
            int(FrontRESSegmentState.UNKNOWN),
            dtype=torch.long,
            device=self.device,
        )
        self.segment_state[self.solved] = int(FrontRESSegmentState.SOLVED)
        self.segment_state[self.hopeless] = int(FrontRESSegmentState.HOPELESS)

    def _sync_terminal_flags_for_ids(self, ids: torch.Tensor) -> None:
        self.solved[ids] = self.segment_state[ids] == int(FrontRESSegmentState.SOLVED)
        self.hopeless[ids] = self.segment_state[ids] == int(FrontRESSegmentState.HOPELESS)

    def _sync_terminal_flags_from_state(self) -> None:
        self.solved = self.segment_state == int(FrontRESSegmentState.SOLVED)
        self.hopeless = self.segment_state == int(FrontRESSegmentState.HOPELESS)

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
