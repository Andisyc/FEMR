from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_warmup",
    Path(__file__).resolve().with_name("frontres_formal_runtime_probe.py"),
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


@dataclass(frozen=True)
class FrontRESSegmentWarmupPhase:
    """Describe the direct Segment PPO optimization phase for one iteration."""

    name: str
    phase_iteration: int
    actor_loss_weight: float
    critic_update_enabled: bool = True


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


__all__ = ["FrontRESSegmentWarmupPhase", "frontres_segment_warmup_phase"]
