from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

from rsl_rl.frontres.frontres_formal_runtime_probe import emit_formal_runtime_probe


FRONTRES_V011_SELECTED_SEGMENT_COUNT = 2
FRONTRES_V011_MAX_ABSOLUTE_ITERATION = 8000
FRONTRES_V011_REVIEW_BOUNDARIES = (2000, 3500, 4825, 6500, 8000)
FRONTRES_V011_K_M_SCHEDULE = (
    (8, 2, 200, 500, 1300),
    (16, 3, 300, 300, 900),
    (32, 4, 400, 300, 625),
)
FRONTRES_V013_DR_CURRICULUM_SCHEMA_ID = "nested-k-dr-four-class-v1"
FRONTRES_V013_DR_CLASS_WEIGHTS = (0.20, 0.30, 0.40, 0.10)
FRONTRES_V013_DR_CLASS_BOUNDARIES = (0.25, 0.70, 1.00, 1.10)
FRONTRES_V013_DR_ADVANCE_RULE_ID = "linear-joint-v1"


@dataclass(frozen=True)
class FrontRESSegmentWarmupPhase:
    """Describe the direct Segment PPO optimization phase for one iteration."""

    name: str
    phase_iteration: int
    actor_loss_weight: float
    critic_update_enabled: bool = True


@dataclass(frozen=True)
class FrontRESKStageSpec:
    """One immutable coordinated K x exact-M stage and optional v013 DR spec."""

    horizon_k: int
    attempts_m: int
    critic_only_iterations: int
    actor_warmup_iterations: int
    joint_iterations: int
    dr_start_distribution_id: str | None = None
    dr_start_cap: float | None = None
    dr_advance_rule_id: str | None = None
    dr_advance_updates: int | None = None
    dr_reference_ceiling: float | None = None


@dataclass(frozen=True)
class FrontRESKStageIdentity:
    """Resolved K x M x DR identity for one committed-update iteration."""

    schedule_fingerprint: str
    stage_index: int
    active_k: int
    active_m: int
    stage_iteration: int
    absolute_iteration: int
    phase: FrontRESSegmentWarmupPhase
    dr_stage_fingerprint: str = ""
    dr_progress: float = 0.0
    d_cap: float = 0.0
    dr_reference_ceiling: float = 0.0


@dataclass(frozen=True)
class FrontRESDRStrengthSample:
    """One deterministic class/strength draw from a sealed v013 DR stage."""

    class_name: str
    class_index: int
    strength: float
    d_cap: float
    dr_progress: float
    dr_stage_fingerprint: str


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
            core_values = (
                spec.horizon_k,
                spec.attempts_m,
                spec.critic_only_iterations,
                spec.actor_warmup_iterations,
                spec.joint_iterations,
            )
            values = core_values
        else:
            values = tuple(raw)
            if len(values) not in {5, 10}:
                raise ValueError(
                    f"K x M stage row {row} must contain five legacy fields or ten explicit TRAIN-v013 fields"
                )
            spec = FrontRESKStageSpec(*values)
            core_values = values[:5]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in core_values):
            raise ValueError(f"K-stage row {row} requires integer durations and K")
        if spec.horizon_k <= 0:
            raise ValueError(f"K-stage row {row} horizon_k must be positive")
        if spec.attempts_m < 2:
            raise ValueError(f"K x M stage row {row} attempts_m must be at least two")
        if spec.critic_only_iterations <= 0 or spec.actor_warmup_iterations <= 0:
            raise ValueError(f"K-stage row {row} requires positive critic-only and actor-ramp durations")
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
    """Parse legacy five-field or explicit v013 ten-field stage rows."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("FRS-TRAIN-v011 requires an explicit K x M schedule")
    rows: list[FrontRESKStageSpec] = []
    for row, token in enumerate(value.split(",")):
        parts = tuple(part.strip() for part in token.split(":"))
        if len(parts) not in {5, 10} or any(not part for part in parts):
            raise ValueError(f"K x M stage token {row} must use five legacy or ten explicit v013 fields")
        try:
            core = tuple(int(part) for part in parts[:5])
            if len(parts) == 5:
                rows.append(FrontRESKStageSpec(*core))
            else:
                rows.append(
                    FrontRESKStageSpec(
                        *core,
                        dr_start_distribution_id=parts[5],
                        dr_start_cap=float(parts[6]),
                        dr_advance_rule_id=parts[7],
                        dr_advance_updates=int(parts[8]),
                        dr_reference_ceiling=float(parts[9]),
                    )
                )
        except ValueError as exc:
            raise ValueError(f"K-stage token {row} contains an invalid typed value") from exc
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
            *(
                ()
                if spec.dr_start_distribution_id is None
                else (
                    spec.dr_start_distribution_id,
                    float(spec.dr_start_cap),
                    spec.dr_advance_rule_id,
                    int(spec.dr_advance_updates),
                    float(spec.dr_reference_ceiling),
                )
            ),
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


def _frontres_v013_dr_stage_payload(spec: FrontRESKStageSpec) -> tuple[object, ...]:
    return (
        int(spec.horizon_k),
        str(spec.dr_start_distribution_id),
        float(spec.dr_start_cap),
        str(spec.dr_advance_rule_id),
        int(spec.dr_advance_updates),
        float(spec.dr_reference_ceiling),
        FRONTRES_V013_DR_CLASS_BOUNDARIES,
        FRONTRES_V013_DR_CLASS_WEIGHTS,
    )


def frontres_v013_dr_stage_fingerprint(spec: FrontRESKStageSpec) -> str:
    payload = json.dumps(_frontres_v013_dr_stage_payload(spec), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_frontres_v013_campaign_schedule(
    schedule: Iterable[FrontRESKStageSpec | Iterable[object]],
) -> tuple[FrontRESKStageSpec, ...]:
    """Require explicit TRAIN-v013 K/M and per-K DR identities without defaults."""

    normalized = normalize_frontres_k_stage_schedule(schedule, max_horizon_k=32)
    if tuple((s.horizon_k, s.attempts_m, s.critic_only_iterations, s.actor_warmup_iterations, s.joint_iterations) for s in normalized) != FRONTRES_V011_K_M_SCHEDULE:
        raise ValueError("FRS-TRAIN-v014 requires the frozen K8/M2 -> K16/M3 -> K32/M4 schedule")
    for row, spec in enumerate(normalized):
        if not isinstance(spec.dr_start_distribution_id, str) or not spec.dr_start_distribution_id.strip():
            raise ValueError(f"TRAIN-v013 stage {row} requires an explicit start_distribution_id")
        if spec.dr_advance_rule_id != FRONTRES_V013_DR_ADVANCE_RULE_ID:
            raise ValueError(f"TRAIN-v013 stage {row} requires advance_rule_id={FRONTRES_V013_DR_ADVANCE_RULE_ID}")
        if spec.dr_start_cap is None or not math.isfinite(float(spec.dr_start_cap)) or float(spec.dr_start_cap) <= 0.0:
            raise ValueError(f"TRAIN-v013 stage {row} requires a positive finite dr_start_cap")
        if spec.dr_reference_ceiling is None or not math.isfinite(float(spec.dr_reference_ceiling)):
            raise ValueError(f"TRAIN-v013 stage {row} requires an explicit finite reference ceiling")
        terminal_hard_cap = float(spec.dr_reference_ceiling) / FRONTRES_V013_DR_CLASS_BOUNDARIES[3]
        if float(spec.dr_start_cap) >= terminal_hard_cap or float(spec.dr_reference_ceiling) > 2.381:
            raise ValueError(
                f"TRAIN-v013 stage {row} requires 0 < start_cap < reference_ceiling/1.10 and reference_ceiling <= 2.381"
            )
        if isinstance(spec.dr_advance_updates, bool) or not isinstance(spec.dr_advance_updates, int) or spec.dr_advance_updates <= 0:
            raise ValueError(f"TRAIN-v013 stage {row} requires positive explicit advance_updates")
    return normalized


def _frontres_v013_dr_progress(spec: FrontRESKStageSpec, stage_iteration: int) -> float:
    joint_progress = max(
        0,
        int(stage_iteration) - int(spec.critic_only_iterations) - int(spec.actor_warmup_iterations),
    )
    return min(1.0, joint_progress / float(spec.dr_advance_updates))


def sample_frontres_v013_dr_strength(
    identity: FrontRESKStageIdentity,
    *,
    sample_key: int,
) -> FrontRESDRStrengthSample:
    """Draw one stable four-class sample from immutable identity plus caller key."""

    if not identity.dr_stage_fingerprint or identity.d_cap <= 0.0:
        raise ValueError("TRAIN-v013 strength sampling requires a resolved DR identity")
    if isinstance(sample_key, bool) or not isinstance(sample_key, int) or sample_key < 0:
        raise ValueError("TRAIN-v013 sample_key must be a nonnegative integer")
    digest = hashlib.sha256(f"{identity.dr_stage_fingerprint}:{sample_key}".encode("ascii")).digest()
    class_u = int.from_bytes(digest[:8], "big") / float(2**64)
    within_u = int.from_bytes(digest[8:16], "big") / float(2**64)
    cumulative = (0.20, 0.50, 0.90, 1.00)
    class_index = next(index for index, upper in enumerate(cumulative) if class_u < upper)
    names = ("easy", "medium", "hard", "broken")
    cap = float(identity.d_cap)
    ceiling = float(identity.dr_reference_ceiling)
    bounds = ((0.0, 0.25 * cap), (0.25 * cap, 0.70 * cap), (0.70 * cap, cap), (cap, min(1.10 * cap, ceiling)))
    low, high = bounds[class_index]
    if high <= low:
        raise RuntimeError("TRAIN-v013 broken-tail support collapsed at the configured reference ceiling")
    strength = low + within_u * (high - low)
    if class_index == 3 and strength <= cap:
        strength = math.nextafter(cap, high)
    return FrontRESDRStrengthSample(
        class_name=names[class_index],
        class_index=class_index,
        strength=float(strength),
        d_cap=cap,
        dr_progress=float(identity.dr_progress),
        dr_stage_fingerprint=str(identity.dr_stage_fingerprint),
    )


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
        dr_stage_fingerprint=(frontres_v013_dr_stage_fingerprint(spec) if spec.dr_start_distribution_id else ""),
        dr_progress=(_frontres_v013_dr_progress(spec, stage_iteration) if spec.dr_start_distribution_id else 0.0),
        d_cap=(
            float(spec.dr_start_cap)
            + (float(spec.dr_reference_ceiling) / FRONTRES_V013_DR_CLASS_BOUNDARIES[3] - float(spec.dr_start_cap))
            * _frontres_v013_dr_progress(spec, stage_iteration)
            if spec.dr_start_distribution_id
            else 0.0
        ),
        dr_reference_ceiling=float(spec.dr_reference_ceiling or 0.0),
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
        d_cap=identity.d_cap,
        dr_progress=identity.dr_progress,
    )
    return identity


def resolve_frontres_k_stage_transition(
    *,
    schedule: Iterable[FrontRESKStageSpec | Iterable[int]],
    committed_update_iteration: int,
    max_horizon_k: int | None = None,
) -> FrontRESKStageIdentity | None:
    """Return the new K-stage identity only at an exact committed boundary."""

    if (
        not isinstance(committed_update_iteration, int)
        or isinstance(committed_update_iteration, bool)
        or committed_update_iteration <= 0
    ):
        return None
    normalized = normalize_frontres_k_stage_schedule(schedule, max_horizon_k=max_horizon_k)

    # B1: 只累计完整 stage 长度, 定位 committed K-stage 边界.
    boundary = 0
    for spec in normalized[:-1]:
        boundary += spec.critic_only_iterations + spec.actor_warmup_iterations + spec.joint_iterations
        if committed_update_iteration == boundary:
            return resolve_frontres_k_stage_identity(
                schedule=normalized,
                committed_update_iteration=committed_update_iteration,
                max_horizon_k=max_horizon_k,
            )
        if committed_update_iteration < boundary:
            return None
    return None


def frontres_segment_warmup_phase(
    *,
    iteration: int,
    critic_warmup_iterations: int,
    actor_warmup_iterations: int,
) -> FrontRESSegmentWarmupPhase:
    """把持久 Stage 3 iteration 映射为 critic-only/actor-ramp/joint phase.

    函数名说明:
        `frontres_segment_warmup_phase` 是纯 phase scheduler, 只计算 actor loss
        weight 和 phase identity; 它不是 acceptance/rho authority ramp.

    主链路:
        上游: runner 提供 persisted iteration 和 immutable warmup boundaries.
        下游: PPO update 使用 `actor_loss_weight`, critic 始终保持可训练.

    语义:
        critic-only 先建立 value baseline, actor ramp 再平滑开放 policy gradient,
        最后进入 joint PPO, 防止初始梯度冲毁 HSL actor.
    """

    iteration = max(0, int(iteration))
    critic_warmup_iterations = max(0, int(critic_warmup_iterations))
    actor_warmup_iterations = max(0, int(actor_warmup_iterations))

    # B1: 读取 persisted Stage 3 iteration 和 immutable warmup boundaries.
    # B2: 唯一选择 critic-only, actor-ramp 或 joint phase.
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
                name="actor_ramp",
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
    # Historical result: E68/E69 used the retired actor_warmup label. The active
    # TRAIN-v014 identity is actor_ramp with the same persisted schedule weight.
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
    "FrontRESDRStrengthSample",
    "FrontRESSegmentWarmupPhase",
    "frontres_k_stage_schedule_fingerprint",
    "frontres_k_stage_schedule_tuple",
    "frontres_segment_warmup_phase",
    "normalize_frontres_k_stage_schedule",
    "parse_frontres_k_stage_schedule",
    "require_frontres_v011_campaign_schedule",
    "require_frontres_v013_campaign_schedule",
    "sample_frontres_v013_dr_strength",
    "frontres_v013_dr_stage_fingerprint",
    "resolve_frontres_k_stage_identity",
    "resolve_frontres_k_stage_transition",
]
