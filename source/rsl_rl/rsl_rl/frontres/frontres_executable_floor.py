"""Executable-floor calibration shared by FrontRES alpha, rho, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ExecutableFloorState:
    """Running score evidence for the GMT executable boundary."""

    safe_score_ema: float | None = None
    broken_score_ema: float | None = None
    safe_count: float = 0.0
    broken_count: float = 0.0


@dataclass(frozen=True)
class ExecutableFloorValues:
    """Resolved score-space floor used by every FrontRES consumer."""

    floor: float
    safe_floor: float
    source: str
    adaptive: float
    safe_count: float
    broken_count: float


def _cfg_get(cfg: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def resolve_executable_floor(
    cfg: Mapping[str, Any] | Any,
    state: ExecutableFloorState,
) -> ExecutableFloorValues:
    """Resolve the live executable floor from fixed fallback plus adaptive evidence.

    The floor is intentionally a score-space quantity. GMT frontier search finds
    the DR boundary; this module converts accumulated safe/broken scores near
    that boundary into the threshold shared by candidate diagnostics, alpha
    labels, and rho floor penalties.
    """

    fixed_floor = float(
        _cfg_get(
            cfg,
            "frontres_executable_floor_score",
            _cfg_get(cfg, "frontres_state_alpha_exec_floor", 0.0),
        )
    )
    safe_default = float(
        _cfg_get(
            cfg,
            "frontres_state_alpha_safe_exec_floor",
            fixed_floor + 0.05,
        )
    )
    safe_margin = float(
        _cfg_get(
            cfg,
            "frontres_executable_floor_safe_margin",
            max(0.0, safe_default - fixed_floor),
        )
    )
    adaptive_enabled = bool(_cfg_get(cfg, "frontres_executable_floor_adaptive_enabled", True))
    min_count = float(_cfg_get(cfg, "frontres_executable_floor_min_samples", 32.0))

    if (
        adaptive_enabled
        and state.safe_score_ema is not None
        and state.broken_score_ema is not None
        and state.safe_count >= min_count
        and state.broken_count >= min_count
    ):
        floor = 0.5 * (float(state.safe_score_ema) + float(state.broken_score_ema))
        source = "adaptive"
    else:
        floor = fixed_floor
        source = "fixed"

    floor = float(floor)
    return ExecutableFloorValues(
        floor=floor,
        safe_floor=floor + max(0.0, safe_margin),
        source=source,
        adaptive=1.0 if source == "adaptive" else 0.0,
        safe_count=float(state.safe_count),
        broken_count=float(state.broken_count),
    )


def update_executable_floor_stats(
    cfg: Mapping[str, Any] | Any,
    state: ExecutableFloorState,
    exec_score: "torch.Tensor",
    *,
    done: "torch.Tensor | None" = None,
    timeout: "torch.Tensor | None" = None,
    mix_class: "torch.Tensor | None" = None,
    frontier_decision: str = "",
) -> tuple[ExecutableFloorState, ExecutableFloorValues]:
    """Update running GMT safe/broken score evidence and resolve the floor."""

    import torch

    values = resolve_executable_floor(cfg, state)
    if not bool(_cfg_get(cfg, "frontres_executable_floor_adaptive_enabled", True)):
        return state, values
    if exec_score.numel() == 0:
        return state, values

    score = exec_score.detach().view(-1)
    n = score.numel()
    valid = torch.ones(n, device=score.device, dtype=torch.bool)
    if timeout is not None:
        timeout = timeout.to(score.device).view(-1)[:n].bool()
        valid = valid & (~timeout)
    fall = torch.zeros(n, device=score.device, dtype=torch.bool)
    if done is not None:
        done = done.to(score.device).view(-1)[:n].bool()
        fall = done & valid

    class_mask = torch.ones(n, device=score.device, dtype=torch.bool)
    if mix_class is not None:
        mix_class = mix_class.to(score.device).view(-1)
        if mix_class.numel() >= n:
            # Only the frontier bucket calibrates the boundary. Easy/hard
            # buckets are coverage samples, not direct floor evidence.
            class_mask = mix_class[:n].long().eq(1)

    decision = str(frontier_decision or "").lower()
    safe_mask = torch.zeros(n, device=score.device, dtype=torch.bool)
    broken_mask = fall & class_mask
    if decision == "safe":
        safe_mask = class_mask & valid & (~fall)
    elif decision == "broken":
        broken_mask = class_mask & valid
    elif decision == "frontier":
        safe_mask = class_mask & valid & (~fall)
        broken_mask = class_mask & valid & fall

    ema_alpha = float(_cfg_get(cfg, "frontres_executable_floor_ema_alpha", 0.95))
    ema_alpha = max(0.0, min(0.999, ema_alpha))

    def _update(prev: float | None, count: float, values: "torch.Tensor") -> tuple[float | None, float]:
        if values.numel() == 0:
            return prev, count
        batch_mean = float(values.mean().item())
        next_ema = batch_mean if prev is None else ema_alpha * float(prev) + (1.0 - ema_alpha) * batch_mean
        return next_ema, float(count) + float(values.numel())

    safe_score_ema, safe_count = _update(
        state.safe_score_ema,
        state.safe_count,
        score[safe_mask],
    )
    broken_score_ema, broken_count = _update(
        state.broken_score_ema,
        state.broken_count,
        score[broken_mask],
    )
    next_state = ExecutableFloorState(
        safe_score_ema=safe_score_ema,
        broken_score_ema=broken_score_ema,
        safe_count=safe_count,
        broken_count=broken_count,
    )
    return next_state, resolve_executable_floor(cfg, next_state)


def _runner_floor_state(runner: Any) -> ExecutableFloorState:
    return ExecutableFloorState(
        safe_score_ema=getattr(runner, "_frontres_exec_floor_safe_score_ema", None),
        broken_score_ema=getattr(runner, "_frontres_exec_floor_broken_score_ema", None),
        safe_count=float(getattr(runner, "_frontres_exec_floor_safe_count", 0.0)),
        broken_count=float(getattr(runner, "_frontres_exec_floor_broken_count", 0.0)),
    )


def _write_runner_floor_values(runner: Any, values: ExecutableFloorValues) -> None:
    runner._frontres_exec_floor_value_last = values.floor
    runner._frontres_exec_floor_safe_last = values.safe_floor
    runner._frontres_exec_floor_source_last = values.source
    runner._frontres_exec_floor_adaptive_last = values.adaptive
    runner._frontres_exec_floor_safe_count_last = values.safe_count
    runner._frontres_exec_floor_broken_count_last = values.broken_count


def resolve_runner_executable_floor(runner: Any) -> tuple[float, float, str]:
    """Return the runner's live executable floor and update last-value diagnostics."""

    values = resolve_executable_floor(runner.cfg, _runner_floor_state(runner))
    _write_runner_floor_values(runner, values)
    return values.floor, values.safe_floor, values.source


def update_runner_executable_floor_stats(
    runner: Any,
    exec_score: "torch.Tensor",
    *,
    done: "torch.Tensor | None" = None,
    timeout: "torch.Tensor | None" = None,
    mix_class: "torch.Tensor | None" = None,
) -> tuple[float, float, str]:
    """Update runner GMT-score floor evidence and return the active floor."""

    state, values = update_executable_floor_stats(
        runner.cfg,
        _runner_floor_state(runner),
        exec_score,
        done=done,
        timeout=timeout,
        mix_class=mix_class,
        frontier_decision=getattr(runner, "_frontres_gmt_frontier_decision", ""),
    )
    for name, value in (
        ("safe_score_ema", state.safe_score_ema),
        ("broken_score_ema", state.broken_score_ema),
        ("safe_count", state.safe_count),
        ("broken_count", state.broken_count),
    ):
        attr = f"_frontres_exec_floor_{name}"
        if value is None:
            if hasattr(runner, attr):
                delattr(runner, attr)
        else:
            setattr(runner, attr, float(value))
    _write_runner_floor_values(runner, values)
    return values.floor, values.safe_floor, values.source
