from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PERTURBATION_BASES = ("planar", "yaw", "global_z", "local_rp")


@dataclass(frozen=True)
class PerturbationMixPlan:
    groups: list[tuple[str, ...]]
    active_modes: tuple[str, ...]
    complexity: str


@dataclass(frozen=True)
class DRStrengthPlan:
    scale_vector: list[float] | None
    mix_class: list[int] | None
    mix_mode: str
    diag: dict[str, float]
    effective_scale: float


@dataclass(frozen=True)
class GMTFrontierState:
    safe_low: float
    broken_high: float | None
    probe_scale: float
    probe_score: float | None = None
    decision: str = "init"
    confirmed: float | None = None


@dataclass(frozen=True)
class GMTFrontierUpdate:
    state: GMTFrontierState
    next_dr_scale: float


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_bool(cfg: Any, key: str, default: bool) -> bool:
    return bool(_cfg_get(cfg, key, default))


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    return float(_cfg_get(cfg, key, default))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def choice_hash(seq_idx: int) -> int:
    value = (int(seq_idx) + 1) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def allowed_perturbation_bases(active_dims: Any = None) -> tuple[str, ...]:
    """Map active FrontRES task dimensions to repairable perturbation families."""
    if active_dims is None:
        return PERTURBATION_BASES

    dims = {int(idx) for idx in active_dims}
    bases: list[str] = []
    if 0 in dims or 1 in dims:
        bases.append("planar")
    if 5 in dims:
        bases.append("yaw")
    if 2 in dims:
        bases.append("global_z")
    if 3 in dims or 4 in dims:
        bases.append("local_rp")
    return tuple(bases) if bases else PERTURBATION_BASES


def mode_complexity(modes: tuple[str, ...], fallback: str | None = None) -> str:
    if len(modes) == 1:
        return "single"
    if len(modes) == 2:
        return "two"
    if len(modes) == 3:
        return "three"
    return fallback or "full"


def _canonical_groups(bases: tuple[str, ...]) -> tuple[
    list[tuple[str, ...]], list[tuple[str, ...]], list[tuple[str, ...]], list[tuple[str, ...]]
]:
    single = [(mode,) for mode in bases]
    canonical_two = [
        ("planar", "yaw"),
        ("planar", "local_rp"),
        ("yaw", "local_rp"),
        ("global_z", "local_rp"),
        ("planar", "global_z"),
        ("yaw", "global_z"),
    ]
    canonical_three = [
        ("planar", "yaw", "local_rp"),
        ("planar", "global_z", "local_rp"),
        ("yaw", "global_z", "local_rp"),
        ("planar", "yaw", "global_z"),
    ]
    base_set = set(bases)
    two = [modes for modes in canonical_two if set(modes).issubset(base_set)]
    three = [modes for modes in canonical_three if set(modes).issubset(base_set)]
    return single, two, three, [tuple(bases)]


def choose_perturbation_choices(
    cfg: Any,
    active_dims: Any,
    progress: float,
    seq_idx: int,
    *,
    boundary_stats: dict[str, float] | None = None,
    is_frontres: bool = True,
) -> tuple[list[tuple[str, ...]], str]:
    bases = allowed_perturbation_bases(active_dims)
    specialist_mode = str(_cfg_get(cfg, "frontres_specialist_mode", "") or "").lower()
    if specialist_mode in ("rp", "local_rp", "rp_only", "strong_rp"):
        return [("local_rp",)], "single"
    if specialist_mode in ("rp_z", "z_rp", "vertical_contact"):
        return [("global_z", "local_rp")], "two"
    if not (is_frontres and _cfg_bool(cfg, "frontres_perturbation_curriculum_enabled", True)):
        return [tuple(bases)], "full"

    progress = _clamp(progress, 0.0, 1.0)
    single, two, three, full = _canonical_groups(bases)
    single_until = _cfg_float(cfg, "frontres_curriculum_single_until", 0.30)
    two_until = _cfg_float(cfg, "frontres_curriculum_two_until", 0.70)
    full_prob = _cfg_float(cfg, "frontres_curriculum_full_prob", 0.05)
    three_prob = _cfg_float(cfg, "frontres_curriculum_three_prob", 0.10)
    two_mid_prob = _cfg_float(cfg, "frontres_curriculum_two_mid_prob", 0.35)
    two_late_prob = _cfg_float(cfg, "frontres_curriculum_two_late_prob", 0.40)
    bucket = (int(seq_idx) * 37) % 1000 / 1000.0

    if _cfg_bool(cfg, "frontres_adaptive_perturb_curriculum_enabled", True):
        if boundary_stats is None:
            return single, "single"
        safe = float(boundary_stats.get("safe", 0.0))
        repair = float(boundary_stats.get("repair", boundary_stats.get("fragile", 0.0)))
        broken = float(boundary_stats.get("broken", 0.0))
        gainpos = float(boundary_stats.get("positive_gain", 0.5))
        safe_hi = _cfg_float(cfg, "frontres_boundary_safe_high", 0.45)
        broken_hi = _cfg_float(cfg, "frontres_boundary_broken_high", 0.35)
        broken_target = _cfg_float(cfg, "frontres_boundary_broken_target", 0.25)
        repair_lo = _cfg_float(
            cfg,
            "frontres_boundary_repair_low",
            _cfg_float(cfg, "frontres_boundary_fragile_low", 0.45),
        )
        repair_hi = _cfg_float(
            cfg,
            "frontres_boundary_repair_high",
            _cfg_float(cfg, "frontres_boundary_fragile_high", 0.70),
        )
        gain_hi = _cfg_float(cfg, "frontres_boundary_positive_gain_high", 0.55)
        gain_lo = _cfg_float(cfg, "frontres_boundary_positive_gain_low", 0.45)

        if broken > broken_hi or (gainpos < gain_lo and broken > broken_target):
            return single, "single"
        if safe > safe_hi and broken < broken_target and two:
            return (two, "two") if bucket < 0.65 else (single, "single")
        if repair_lo <= repair <= repair_hi and gainpos > gain_hi:
            if bucket < full_prob and len(bases) > 1:
                return full, "full"
            if bucket < full_prob + max(three_prob, 0.15) and three:
                return three, "three"
            if bucket < full_prob + max(three_prob, 0.15) + 0.55 and two:
                return two, "two"
            return single, "single"
        return (two, "two") if bucket < 0.30 and two else (single, "single")

    if progress < single_until:
        return single, "single"
    if progress < two_until:
        return (two, "two") if bucket < two_mid_prob and two else (single, "single")
    if bucket < full_prob and len(bases) > 1:
        return full, "full"
    if bucket < full_prob + three_prob and three:
        return three, "three"
    if bucket < full_prob + three_prob + two_late_prob and two:
        return two, "two"
    return single, "single"


def sample_perturbation_mix(
    cfg: Any,
    active_dims: Any,
    progress: float,
    seq_idx: int,
    n_train: int,
    *,
    boundary_stats: dict[str, float] | None = None,
    is_frontres: bool = True,
) -> PerturbationMixPlan:
    choices, phase_complexity = choose_perturbation_choices(
        cfg,
        active_dims,
        progress,
        seq_idx,
        boundary_stats=boundary_stats,
        is_frontres=is_frontres,
    )
    groups = [
        tuple(choices[choice_hash(seq_idx * 1009 + env_i) % len(choices)])
        for env_i in range(max(int(n_train), 0))
    ]
    if not groups:
        groups = [tuple(allowed_perturbation_bases(active_dims))]
    active_modes = tuple(sorted({mode for group in groups for mode in group}))
    complexities = {mode_complexity(group) for group in groups}
    complexity = next(iter(complexities)) if len(complexities) == 1 else "mixed"
    if complexity == "full" and phase_complexity != "full":
        complexity = phase_complexity
    return PerturbationMixPlan(groups=groups, active_modes=active_modes, complexity=complexity)


def warmup_perturbation_mode_groups(
    cfg: Any,
    active_dims: Any,
    seq_idx: int,
    *,
    current_active_modes: tuple[str, ...] | None = None,
) -> list[tuple[str, ...]]:
    bases = allowed_perturbation_bases(active_dims)
    mode = str(_cfg_get(
        cfg,
        "frontres_warmup_perturbation_schedule",
        _cfg_get(cfg, "supervised_warmup_perturbation_schedule", "mixed_single"),
    ))
    specialist_mode = str(_cfg_get(cfg, "frontres_specialist_mode", "") or "").lower()
    if specialist_mode in ("rp", "local_rp", "rp_only", "strong_rp"):
        return [("local_rp",)]
    if specialist_mode in ("rp_z", "z_rp", "vertical_contact"):
        return [("global_z", "local_rp")]
    if mode == "rl_curriculum":
        active = tuple(current_active_modes or bases)
        return [active] if active else [tuple(bases)]
    if mode == "full":
        return [tuple(bases)]
    if mode in ("mixed_pair", "balanced_pair"):
        _, pairs, _, _ = _canonical_groups(bases)
        if pairs:
            if mode == "balanced_pair":
                return [pairs[choice_hash(seq_idx) % len(pairs)]]
            return pairs
    if mode == "single":
        return [(bases[choice_hash(seq_idx) % len(bases)],)]
    return [(base,) for base in bases]


def sample_scalar_dr_strength(
    cfg: Any,
    frontier_scale: float,
    enabled: bool,
    seq_idx: int,
    *,
    dr_min: float,
    dr_max: float,
) -> tuple[float, str]:
    effective_scale = float(frontier_scale)
    mix_enabled = bool(enabled) and _cfg_bool(cfg, "frontres_mixed_dr_strength_enabled", True)
    mix_mode = "frontier" if mix_enabled else "fixed"
    if mix_enabled:
        easy_w = max(0.0, _cfg_float(cfg, "frontres_mixed_dr_easy_weight", 0.5))
        frontier_w = max(0.0, _cfg_float(cfg, "frontres_mixed_dr_frontier_weight", 0.4))
        hard_w = max(0.0, _cfg_float(cfg, "frontres_mixed_dr_hard_weight", 0.1))
        weight_sum = max(easy_w + frontier_w + hard_w, 1e-6)
        easy_w /= weight_sum
        frontier_w /= weight_sum
        bucket = (int(seq_idx) * 2654435761 % 1000) / 1000.0
        if bucket < easy_w:
            mix_mode = "easy"
            factor = _cfg_float(cfg, "frontres_mixed_dr_easy_factor", 0.75)
        elif bucket < easy_w + frontier_w:
            mix_mode = "frontier"
            factor = _cfg_float(cfg, "frontres_mixed_dr_frontier_factor", 1.0)
        else:
            mix_mode = "hard"
            factor = _cfg_float(cfg, "frontres_mixed_dr_hard_factor", 1.05)
        effective_scale = float(frontier_scale) * max(0.0, factor)
    return _clamp(effective_scale, dr_min, dr_max), mix_mode


def sample_per_env_dr_strength(
    cfg: Any,
    frontier_scale: float,
    enabled: bool,
    seq_idx: int,
    *,
    n_train: int,
    n_candidate: int,
    n_base: int,
    num_envs: int,
    dr_min: float,
    dr_max: float,
) -> DRStrengthPlan:
    fixed_diag = {"easy": 0.0, "frontier": 1.0, "hard": 0.0, "mean": float(frontier_scale)}
    mix_enabled = bool(enabled) and _cfg_bool(cfg, "frontres_mixed_dr_strength_per_env", True)
    if not mix_enabled or n_train <= 0:
        return DRStrengthPlan(None, None, "fixed", fixed_diag, float(frontier_scale))

    easy_w = max(0.0, _cfg_float(cfg, "frontres_mixed_dr_easy_weight", 0.5))
    frontier_w = max(0.0, _cfg_float(cfg, "frontres_mixed_dr_frontier_weight", 0.4))
    hard_w = max(0.0, _cfg_float(cfg, "frontres_mixed_dr_hard_weight", 0.1))
    weight_sum = max(easy_w + frontier_w + hard_w, 1e-6)
    easy_w /= weight_sum
    frontier_w /= weight_sum

    factors = (
        max(0.0, _cfg_float(cfg, "frontres_mixed_dr_easy_factor", 0.75)),
        max(0.0, _cfg_float(cfg, "frontres_mixed_dr_frontier_factor", 1.0)),
        max(0.0, _cfg_float(cfg, "frontres_mixed_dr_hard_factor", 1.05)),
    )
    frontier_scale = _clamp(frontier_scale, dr_min, dr_max)
    train_scales: list[float] = []
    mix_class: list[int] = []
    for env_i in range(int(n_train)):
        bucket = (choice_hash(seq_idx * 9176 + env_i * 131 + 17) % 1000) / 1000.0
        if bucket < easy_w:
            cls = 0
        elif bucket < easy_w + frontier_w:
            cls = 1
        else:
            cls = 2
        mix_class.append(cls)
        train_scales.append(_clamp(frontier_scale * factors[cls], dr_min, dr_max))

    scale_vector = [0.0 for _ in range(int(num_envs))]
    for env_i, scale in enumerate(train_scales[:num_envs]):
        scale_vector[env_i] = scale
    candidate_start = int(n_train)
    base_start = int(n_train) + int(n_candidate)
    for env_i, scale in enumerate(train_scales[: max(0, int(n_candidate))]):
        dst = candidate_start + env_i
        if dst < len(scale_vector):
            scale_vector[dst] = scale
    for env_i, scale in enumerate(train_scales[: max(0, int(n_base))]):
        dst = base_start + env_i
        if dst < len(scale_vector):
            scale_vector[dst] = scale

    denom = max(1, len(mix_class))
    diag = {
        "easy": sum(1 for cls in mix_class if cls == 0) / denom,
        "frontier": sum(1 for cls in mix_class if cls == 1) / denom,
        "hard": sum(1 for cls in mix_class if cls == 2) / denom,
        "mean": sum(train_scales) / max(1, len(train_scales)),
    }
    return DRStrengthPlan(scale_vector, mix_class, "per_env", diag, diag["mean"])


def update_boundary_ema(
    cfg: Any,
    previous: dict[str, float] | None,
    current: dict[str, float],
) -> dict[str, float]:
    alpha = _clamp(_cfg_float(cfg, "frontres_boundary_dr_ema_alpha", 0.90), 0.0, 0.999)
    if previous is None:
        return dict(current)
    updated = dict(previous)
    for key, value in current.items():
        updated[key] = alpha * float(updated.get(key, value)) + (1.0 - alpha) * float(value)
    return updated


def score_gmt_frontier(lengths: list[float] | tuple[float, ...], ref_episode_len: float) -> float | None:
    if not lengths:
        return None
    ref = max(1e-6, float(ref_episode_len))
    return _clamp(sum(float(v) for v in lengths) / len(lengths) / ref, 0.0, 1.5)


def update_gmt_frontier_state(
    cfg: Any,
    state: GMTFrontierState,
    *,
    score: float | None,
    samples: int,
    dr_scale: float,
    dr_scale_init: float,
    dr_min: float,
    dr_max: float,
) -> GMTFrontierUpdate:
    probe_scale = _clamp(state.probe_scale, dr_min, dr_max)
    safe_low = _clamp(state.safe_low, dr_min, dr_max)
    broken_high = state.broken_high
    if broken_high is not None:
        broken_high = max(safe_low, min(dr_max, float(broken_high)))

    if samples <= 0 or score is None:
        confirmed = state.confirmed if state.confirmed is not None else dr_scale
        new_state = GMTFrontierState(
            safe_low=safe_low,
            broken_high=broken_high,
            probe_scale=probe_scale,
            probe_score=score,
            decision="waiting",
            confirmed=_clamp(confirmed, dr_min, dr_max),
        )
        return GMTFrontierUpdate(new_state, _clamp(dr_scale, dr_min, dr_max))

    safe_thr = _cfg_float(cfg, "frontres_gmt_frontier_safe_threshold", 0.85)
    broken_thr = _cfg_float(cfg, "frontres_gmt_frontier_broken_threshold", 0.65)
    growth = max(1.001, _cfg_float(cfg, "frontres_gmt_frontier_growth_factor", 1.12))

    if score >= safe_thr:
        safe_low = max(safe_low, probe_scale)
        decision = "safe"
        if broken_high is None:
            next_probe = min(dr_max, max(probe_scale * growth, safe_low + 1e-6))
        else:
            next_probe = 0.5 * (safe_low + float(broken_high))
    elif score <= broken_thr:
        broken_high = probe_scale if broken_high is None else min(float(broken_high), probe_scale)
        if probe_scale <= safe_low + 1e-6:
            retreat = max(0.1, min(0.99, _cfg_float(cfg, "frontres_gmt_frontier_retreat_factor", 0.85)))
            safe_low = max(dr_min, min(safe_low * retreat, probe_scale - 1e-6))
        decision = "broken"
        next_probe = 0.5 * (safe_low + float(broken_high))
    else:
        decision = "frontier"
        if broken_high is None:
            broken_high = min(dr_max, max(probe_scale * growth, probe_scale + 1e-6))
        next_probe = probe_scale

    safe_low = _clamp(safe_low, dr_min, dr_max)
    if broken_high is not None:
        broken_high = max(safe_low, min(dr_max, float(broken_high)))
    conservative = _clamp(_cfg_float(cfg, "frontres_gmt_frontier_conservative_frac", 0.0), 0.0, 1.0)
    if broken_high is not None:
        confirmed = safe_low + conservative * (float(broken_high) - safe_low)
    else:
        confirmed = safe_low
    new_state = GMTFrontierState(
        safe_low=safe_low,
        broken_high=broken_high,
        probe_scale=_clamp(next_probe, dr_min, dr_max),
        probe_score=score,
        decision=decision,
        confirmed=_clamp(confirmed, dr_min, dr_max),
    )
    return GMTFrontierUpdate(new_state, new_state.probe_scale)
