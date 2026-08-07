from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rsl_rl.frontres.frontres_interfaces import FrontRESActiveRunMode


def frontres_runner_cfg_get(runner: Any, key: str, default: Any) -> Any:
    """Read one FrontRES setting through the formal runner ownership order."""

    for owner in (
        getattr(runner, "alg", None),
        getattr(runner, "cfg", None),
        getattr(runner, "alg_cfg", None),
    ):
        if owner is None:
            continue
        value = owner.get(key) if isinstance(owner, dict) else getattr(owner, key, None)
        if value is not None:
            return value
    return default


class FrontRESStartupLifecycle:
    """Observable Composition-Root seam for one FrontRES process launch."""

    def __init__(self, *, event_sink: Callable[[str], None] | None = None) -> None:
        self._event_sink = event_sink
        self._events: list[str] = []
        self._phase = "configured"
        self._load_kind: str | None = None
        self._dispatch_started = False
        self._record("config")

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(self._events)

    @property
    def load_kind(self) -> str | None:
        return self._load_kind

    def resolve_layout(self, operation: Callable[[], Any]) -> Any:
        if self._phase != "configured":
            raise RuntimeError(f"FrontRES startup layout is out of order: phase={self._phase}")
        return self._run_transition("layout", "layout_ready", operation)

    def load(self, kind: str, operation: Callable[[], Any]) -> Any:
        normalized = str(kind).strip().lower()
        if normalized not in {"hsl", "resume"}:
            raise ValueError(f"FrontRES startup load kind must be hsl or resume, got {kind!r}")
        if self._phase != "layout_ready" or self._load_kind is not None:
            raise RuntimeError(
                "FrontRES startup HSL/resume load must occur exactly once after layout resolution"
            )
        result = self._run_transition(normalized, "load_ready", operation)
        self._load_kind = normalized
        return result

    def dispatch_once(self, mode: str, operation: Callable[[], Any]) -> Any:
        normalized = str(mode).strip().lower()
        if self._dispatch_started:
            raise RuntimeError("FrontRES startup request has already been dispatched")
        if self._phase not in {"layout_ready", "load_ready"}:
            raise RuntimeError(f"FrontRES startup dispatch is out of order: phase={self._phase}")
        if normalized == "formal_train" and self._load_kind not in {"hsl", "resume"}:
            raise RuntimeError("formal FrontRES training requires exactly one HSL or resume load")
        self._dispatch_started = True
        try:
            result = operation()
        except BaseException:
            self._phase = "failed"
            self._record(f"{normalized}:failed")
            raise
        self._phase = "dispatched"
        self._record(f"dispatch:{normalized}")
        return result

    def _run_transition(self, event: str, next_phase: str, operation: Callable[[], Any]) -> Any:
        try:
            result = operation()
        except BaseException:
            self._phase = "failed"
            self._record(f"{event}:failed")
            raise
        self._phase = next_phase
        self._record(event)
        return result

    def _record(self, event: str) -> None:
        self._events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)


@dataclass(frozen=True)
class FrontRESSegmentRunnerBoundary:
    requested: bool
    live_runner_enabled: bool
    live_sentinel_only: bool
    local_sentinel_only: bool
    live_probe_only: bool
    live_storage_write_only: bool
    live_single_update_only: bool
    live_update_loop_only: bool
    live_train_enabled: bool
    live_update_steps: int
    objective: str
    segment_k: int
    max_horizon_k: int
    reset_mode: str
    future_offsets: tuple[int, ...]
    future_intent_layout_version: str

    @classmethod
    def from_train_cfg(cls, train_cfg: dict[str, Any]) -> "FrontRESSegmentRunnerBoundary":
        alg_cfg = train_cfg.get("algorithm", {})
        objective = str(alg_cfg.get("frontres_training_objective", "")).lower()
        evaluation_only = bool(alg_cfg.get("frontres_policy_quality_eval_only", False))
        if evaluation_only:
            evaluation_conflicts = tuple(
                name
                for name in (
                    "frontres_segment_replay_enabled",
                    "frontres_segment_live_runner_enabled",
                    "frontres_segment_live_sentinel_only",
                    "frontres_local_sentinel_only",
                    "frontres_segment_live_probe_only",
                    "frontres_segment_live_storage_write_only",
                    "frontres_segment_live_single_update_only",
                    "frontres_segment_live_update_loop_only",
                    "frontres_segment_live_train_enabled",
                )
                if bool(alg_cfg.get(name, False))
            )
            if evaluation_conflicts:
                raise ValueError(
                    "policy-quality evaluation cannot enable Segment Replay/live training flags: "
                    f"{evaluation_conflicts}"
                )
        requested = not evaluation_only and (
            bool(alg_cfg.get("frontres_segment_replay_enabled", False))
            or objective == "segment_replay_hrl"
        )
        return cls(
            requested=requested,
            live_runner_enabled=bool(alg_cfg.get("frontres_segment_live_runner_enabled", False)),
            live_sentinel_only=bool(alg_cfg.get("frontres_segment_live_sentinel_only", False)),
            local_sentinel_only=bool(alg_cfg.get("frontres_local_sentinel_only", False)),
            live_probe_only=bool(alg_cfg.get("frontres_segment_live_probe_only", False)),
            live_storage_write_only=bool(alg_cfg.get("frontres_segment_live_storage_write_only", False)),
            live_single_update_only=bool(alg_cfg.get("frontres_segment_live_single_update_only", False)),
            live_update_loop_only=bool(alg_cfg.get("frontres_segment_live_update_loop_only", False)),
            live_train_enabled=bool(alg_cfg.get("frontres_segment_live_train_enabled", False)),
            live_update_steps=max(1, int(alg_cfg.get("frontres_segment_live_update_steps", 4))),
            objective=objective,
            segment_k=max(1, int(alg_cfg.get("frontres_segment_k", 1))),
            max_horizon_k=max(
                max(1, int(alg_cfg.get("frontres_segment_k", 1))),
                int(alg_cfg.get("frontres_segment_max_horizon_k", 64)),
            ),
            reset_mode=str(alg_cfg.get("frontres_segment_reset_mode", "auto")).lower(),
            future_offsets=tuple(int(value) for value in (alg_cfg.get("frontres_future_offsets", ()) or ())),
            future_intent_layout_version=str(alg_cfg.get("frontres_future_intent_layout_version", "") or ""),
        )

    @property
    def run_mode(self) -> FrontRESActiveRunMode:
        """Return one explicit FrontRES mode instead of exposing boolean mixing."""

        if not self.requested:
            return FrontRESActiveRunMode.DISABLED
        if self.local_sentinel_only and self.live_sentinel_only:
            raise ValueError("active FrontRES local sentinel cannot mix with the legacy live sentinel")
        enabled_modes = (
            ("local_sentinel", self.local_sentinel_only),
            ("local_sentinel", self.live_sentinel_only),
            ("live_probe", self.live_probe_only),
            ("live_probe", self.live_storage_write_only),
            ("live_probe", self.live_single_update_only),
            ("live_probe", self.live_update_loop_only),
            ("formal_train", self.live_train_enabled),
        )
        active = tuple(name for name, enabled in enabled_modes if enabled)
        if not active:
            return FrontRESActiveRunMode.UNCONFIGURED
        if len(active) != 1:
            raise ValueError(f"FrontRES requires exactly one explicit run mode; active={active}")
        return FrontRESActiveRunMode(active[0])

    def assert_live_runner_ready(self) -> None:
        if not self.requested:
            return
        if not self.live_runner_enabled:
            raise NotImplementedError(
                "Stage 3 Segment Replay HRL is recognized, but live runner integration is disabled. "
                "Use frontres_segment_replay_toy_chain.py and boundary tests until PPO/live rollout wiring is implemented."
            )
        _ = self.run_mode
        if self.local_sentinel_only:
            if (
                not self.future_offsets
                or any(value <= 0 for value in self.future_offsets)
                or tuple(sorted(set(self.future_offsets))) != self.future_offsets
                or self.future_intent_layout_version != "frontres-v015-future-intent-q29-v1"
            ):
                raise ValueError("v015 local sentinel requires explicit ordered q29 future offsets and the v015 layout")
            if any(
                (
                    self.live_sentinel_only,
                    self.live_probe_only,
                    self.live_storage_write_only,
                    self.live_single_update_only,
                    self.live_update_loop_only,
                    self.live_train_enabled,
                )
            ):
                raise ValueError("v015 local sentinel rejects legacy Stage 3 live mode mixing")
            return
        if (
            self.live_sentinel_only
            or self.live_probe_only
            or self.live_storage_write_only
            or self.live_single_update_only
            or self.live_update_loop_only
            or self.live_train_enabled
        ):
            return
        raise NotImplementedError(
            "Stage 3 live runner flag is enabled, but PPO/live rollout wiring is still not implemented."
        )

    def sentinel_log(self) -> str | None:
        if not (self.requested and self.live_runner_enabled and self.live_sentinel_only):
            return None
        return (
            "[FrontRES Segment Live Sentinel] "
            f"objective={self.objective} "
            f"segment_k={self.segment_k} "
            f"max_horizon_k={self.max_horizon_k} "
            f"reset_mode={self.reset_mode} "
            "live_runner=True "
            "sentinel_only=True "
            "storage=independent "
            "ppo_action=delta_se3_6d "
            "training_update=disabled"
        )

    def local_sentinel_log(self) -> str | None:
        if not (self.requested and self.live_runner_enabled and self.local_sentinel_only):
            return None
        return (
            "[FrontRES v016 Local Sentinel] "
            f"objective={self.objective} "
            f"segment_k={self.segment_k} "
            f"max_horizon_k={self.max_horizon_k} "
            "frontres_local_sentinel=True "
            f"future_offsets={self.future_offsets} "
            "evaluation=external"
        )

    def probe_log(self) -> str | None:
        if not (
            self.requested
            and self.live_runner_enabled
            and (
                self.live_probe_only
                or self.live_storage_write_only
                or self.live_single_update_only
                or self.live_update_loop_only
            )
        ):
            return None
        storage_write = (
            "True"
            if self.live_storage_write_only or self.live_single_update_only or self.live_update_loop_only
            else "False"
        )
        ppo_update = "True" if self.live_single_update_only or self.live_update_loop_only else "False"
        return (
            "[FrontRES Segment Live Probe Ready] "
            f"objective={self.objective} "
            f"segment_k={self.segment_k} "
            f"max_horizon_k={self.max_horizon_k} "
            f"update_steps={self.live_update_steps} "
            f"reset_mode={self.reset_mode} "
            "live_runner=True "
            "mode=probe "
            "probe_only=True "
            f"storage_write={storage_write} "
            f"ppo_update={ppo_update}"
        )

    def train_log(self) -> str | None:
        if not (self.requested and self.live_runner_enabled and self.live_train_enabled):
            return None
        return (
            "[FrontRES Segment Live Train Ready] "
            f"objective={self.objective} "
            f"segment_k={self.segment_k} "
            f"max_horizon_k={self.max_horizon_k} "
            f"update_steps={self.live_update_steps} "
            f"reset_mode={self.reset_mode} "
            "live_runner=True "
            "runner_learn=True "
            "storage=independent "
            "ppo_action=delta_se3_6d "
            "evaluation=external"
        )
