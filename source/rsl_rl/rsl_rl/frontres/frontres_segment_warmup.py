from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Iterable

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_warmup",
    Path(__file__).resolve().with_name("frontres_formal_runtime_probe.py"),
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


FRONTRES_V011_SELECTED_SEGMENT_COUNT = 2
FRONTRES_V011_MAX_ABSOLUTE_ITERATION = 8000
FRONTRES_V011_REVIEW_BOUNDARIES = (2000, 3500, 4825, 6500, 8000)
FRONTRES_V011_K_M_SCHEDULE = (
    (8, 2, 200, 500, 1300),
    (16, 3, 300, 300, 900),
    (32, 4, 400, 300, 625),
)


@dataclass(frozen=True)
class FrontRESSegmentWarmupPhase:
    """Describe the direct Segment PPO optimization phase for one iteration."""

    name: str
    phase_iteration: int
    actor_loss_weight: float
    critic_update_enabled: bool = True


@dataclass(frozen=True)
class FrontRESKStageSpec:
    """One immutable FRS-TRAIN-v011 coordinated K x exact-M stage."""

    horizon_k: int
    attempts_m: int
    critic_only_iterations: int
    actor_warmup_iterations: int
    joint_iterations: int


@dataclass(frozen=True)
class FrontRESKStageIdentity:
    """Resolved v011 K x M identity for one committed-update iteration."""

    schedule_fingerprint: str
    stage_index: int
    active_k: int
    active_m: int
    stage_iteration: int
    absolute_iteration: int
    phase: FrontRESSegmentWarmupPhase


def normalize_frontres_k_stage_schedule(
    schedule: Iterable[FrontRESKStageSpec | Iterable[int]],
    *,
    max_horizon_k: int | None = None,
) -> tuple[FrontRESKStageSpec, ...]:
    """Validate and freeze the explicit global K-stage schedule."""

    normalized: list[FrontRESKStageSpec] = []
    for row, raw in enumerate(tuple(schedule)):
        if isinstance(raw, FrontRESKStageSpec):
            spec = raw
            values = (
                spec.horizon_k,
                spec.attempts_m,
                spec.critic_only_iterations,
                spec.actor_warmup_iterations,
                spec.joint_iterations,
            )
        else:
            values = tuple(raw)
            if len(values) != 5:
                raise ValueError(f"K x M stage row {row} must contain (K,M,N_c,N_a,N_joint)")
            spec = FrontRESKStageSpec(*values)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError(f"K-stage row {row} requires integer durations and K")
        if spec.horizon_k <= 0:
            raise ValueError(f"K-stage row {row} horizon_k must be positive")
        if spec.attempts_m < 2:
            raise ValueError(f"K x M stage row {row} attempts_m must be at least two")
        if spec.critic_only_iterations <= 0 or spec.actor_warmup_iterations <= 0:
            raise ValueError(f"K-stage row {row} requires positive critic-only and actor-warmup durations")
        if spec.joint_iterations < 0:
            raise ValueError(f"K-stage row {row} joint duration must be nonnegative")
        if row > 0 and spec.horizon_k <= normalized[-1].horizon_k:
            raise ValueError("K-stage horizons must be strictly increasing")
        if row > 0 and spec.attempts_m < normalized[-1].attempts_m:
            raise ValueError("K x M stage attempts must be non-decreasing")
        if max_horizon_k is not None and spec.horizon_k > int(max_horizon_k):
            raise ValueError(f"K-stage horizon {spec.horizon_k} exceeds max_horizon_k={int(max_horizon_k)}")
        normalized.append(spec)
    if not normalized:
        raise ValueError("FRS-TRAIN-v011 requires a nonempty explicit K x M schedule")
    for row, spec in enumerate(normalized[:-1]):
        if spec.joint_iterations <= 0:
            raise ValueError(f"non-final K-stage row {row} requires a positive joint duration")
    return tuple(normalized)


def parse_frontres_k_stage_schedule(
    value: str,
    *,
    max_horizon_k: int | None = None,
) -> tuple[FrontRESKStageSpec, ...]:
    """Parse ``K:M:N_c:N_a:N_joint`` comma-separated CLI syntax."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("FRS-TRAIN-v011 requires an explicit K x M schedule")
    rows: list[tuple[int, int, int, int, int]] = []
    for row, token in enumerate(value.split(",")):
        parts = tuple(part.strip() for part in token.split(":"))
        if len(parts) != 5 or any(not part for part in parts):
            raise ValueError(f"K x M stage token {row} must use K:M:N_c:N_a:N_joint")
        try:
            rows.append(tuple(int(part) for part in parts))
        except ValueError as exc:
            raise ValueError(f"K-stage token {row} contains a non-integer value") from exc
    return normalize_frontres_k_stage_schedule(rows, max_horizon_k=max_horizon_k)


def frontres_k_stage_schedule_tuple(
    schedule: Iterable[FrontRESKStageSpec | Iterable[int]],
) -> tuple[tuple[int, int, int, int, int], ...]:
    normalized = normalize_frontres_k_stage_schedule(schedule)
    return tuple(
        (
            spec.horizon_k,
            spec.attempts_m,
            spec.critic_only_iterations,
            spec.actor_warmup_iterations,
            spec.joint_iterations,
        )
        for spec in normalized
    )


def frontres_k_stage_schedule_fingerprint(
    schedule: Iterable[FrontRESKStageSpec | Iterable[int]],
) -> str:
    payload = json.dumps(frontres_k_stage_schedule_tuple(schedule), separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def require_frontres_v011_campaign_schedule(
    schedule: Iterable[FrontRESKStageSpec | Iterable[int]],
) -> tuple[FrontRESKStageSpec, ...]:
    """Return the one immutable TRAIN-v011 campaign schedule or fail closed."""

    normalized = normalize_frontres_k_stage_schedule(schedule, max_horizon_k=32)
    if frontres_k_stage_schedule_tuple(normalized) != FRONTRES_V011_K_M_SCHEDULE:
        raise ValueError(
            "FRS-TRAIN-v011 requires the frozen K8/M2 -> K16/M3 -> K32/M4 campaign schedule"
        )
    return normalized


def resolve_frontres_k_stage_identity(
    *,
    schedule: Iterable[FrontRESKStageSpec | Iterable[int]],
    committed_update_iteration: int,
    max_horizon_k: int | None = None,
) -> FrontRESKStageIdentity:
    """Resolve one global K and stage-local phase from committed updates."""

    if (
        not isinstance(committed_update_iteration, int)
        or isinstance(committed_update_iteration, bool)
        or committed_update_iteration < 0
    ):
        raise ValueError("committed_update_iteration must be a nonnegative integer")
    absolute_iteration = committed_update_iteration
    normalized = normalize_frontres_k_stage_schedule(schedule, max_horizon_k=max_horizon_k)
    remaining = absolute_iteration
    stage_index = len(normalized) - 1
    stage_iteration = remaining
    for row, spec in enumerate(normalized[:-1]):
        duration = spec.critic_only_iterations + spec.actor_warmup_iterations + spec.joint_iterations
        if remaining < duration:
            stage_index = row
            stage_iteration = remaining
            break
        remaining -= duration
    else:
        stage_iteration = remaining
    spec = normalized[stage_index]
    phase = frontres_segment_warmup_phase(
        iteration=stage_iteration,
        critic_warmup_iterations=spec.critic_only_iterations,
        actor_warmup_iterations=spec.actor_warmup_iterations,
    )
    identity = FrontRESKStageIdentity(
        schedule_fingerprint=frontres_k_stage_schedule_fingerprint(normalized),
        stage_index=stage_index,
        active_k=spec.horizon_k,
        active_m=spec.attempts_m,
        stage_iteration=stage_iteration,
        absolute_iteration=absolute_iteration,
        phase=phase,
    )
    emit_formal_runtime_probe(
        "AUDIT-KPLAN-01",
        schedule_fingerprint=identity.schedule_fingerprint,
        stage_index=identity.stage_index,
        active_k=identity.active_k,
        active_m=identity.active_m,
        stage_iteration=identity.stage_iteration,
        absolute_iteration=identity.absolute_iteration,
        phase=identity.phase.name,
        actor_loss_weight=identity.phase.actor_loss_weight,
    )
    return identity


def frontres_segment_warmup_phase(
    *,
    iteration: int,
    critic_warmup_iterations: int,
    actor_warmup_iterations: int,
) -> FrontRESSegmentWarmupPhase:
    """把持久 Stage 3 iteration 映射为 critic/actor warmup phase.

    函数名说明:
        `frontres_segment_warmup_phase` 是纯 phase scheduler, 只计算 actor loss
        weight 和 phase identity; 它不是 acceptance/rho authority ramp.

    主链路:
        上游: runner 提供 persisted iteration 和 immutable warmup boundaries.
        下游: PPO update 使用 `actor_loss_weight`, critic 始终保持可训练.

    语义:
        critic-only 先建立 value baseline, actor warmup 再平滑开放 policy gradient,
        最后进入 joint PPO, 防止初始梯度冲毁 HSL actor.
    """

    iteration = max(0, int(iteration))
    critic_warmup_iterations = max(0, int(critic_warmup_iterations))
    actor_warmup_iterations = max(0, int(actor_warmup_iterations))

    # B1: 读取 persisted Stage 3 iteration 和 immutable warmup boundaries.
    # B2: 唯一选择 critic-only, actor-warmup 或 joint phase.
    if iteration < critic_warmup_iterations:
        phase = FrontRESSegmentWarmupPhase(
            name="critic_only",
            phase_iteration=iteration,
            actor_loss_weight=0.0,
        )
    else:
        actor_iteration = iteration - critic_warmup_iterations
        if actor_iteration < actor_warmup_iterations:
            weight = float(actor_iteration + 1) / float(actor_warmup_iterations)
            phase = FrontRESSegmentWarmupPhase(
                name="actor_warmup",
                phase_iteration=actor_iteration,
                actor_loss_weight=max(0.0, min(1.0, weight)),
            )
        else:
            phase = FrontRESSegmentWarmupPhase(
                name="joint",
                phase_iteration=max(0, actor_iteration - actor_warmup_iterations),
                actor_loss_weight=1.0,
            )
    # B3: AUDIT-WARMUP-01 截获 PPO loss weighting 实际消费的 phase.
    # Result: E68/E69 LIVE PASS. resume 后 absolute iter 220 对应 actor_warmup
    # phase_iter=20, actor_weight=0.042, 没有重启 warmup schedule.
    emit_formal_runtime_probe(
        "AUDIT-WARMUP-01",
        iteration=iteration,
        critic_warmup_iterations=critic_warmup_iterations,
        actor_warmup_iterations=actor_warmup_iterations,
        phase=phase.name,
        phase_iteration=phase.phase_iteration,
        actor_loss_weight=phase.actor_loss_weight,
    )
    return phase


__all__ = [
    "FRONTRES_V011_K_M_SCHEDULE",
    "FRONTRES_V011_MAX_ABSOLUTE_ITERATION",
    "FRONTRES_V011_REVIEW_BOUNDARIES",
    "FRONTRES_V011_SELECTED_SEGMENT_COUNT",
    "FrontRESKStageIdentity",
    "FrontRESKStageSpec",
    "FrontRESSegmentWarmupPhase",
    "frontres_k_stage_schedule_fingerprint",
    "frontres_k_stage_schedule_tuple",
    "frontres_segment_warmup_phase",
    "normalize_frontres_k_stage_schedule",
    "parse_frontres_k_stage_schedule",
    "require_frontres_v011_campaign_schedule",
    "resolve_frontres_k_stage_identity",
]
