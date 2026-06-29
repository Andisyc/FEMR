from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FrontRESSegmentRunnerBoundary:
    requested: bool
    live_runner_enabled: bool
    live_sentinel_only: bool
    live_probe_only: bool
    live_storage_write_only: bool
    live_single_update_only: bool
    live_update_loop_only: bool
    live_train_enabled: bool
    live_update_steps: int
    objective: str
    segment_k: int
    reset_mode: str

    @classmethod
    def from_train_cfg(cls, train_cfg: dict[str, Any]) -> "FrontRESSegmentRunnerBoundary":
        alg_cfg = train_cfg.get("algorithm", {})
        objective = str(alg_cfg.get("frontres_training_objective", "")).lower()
        requested = bool(alg_cfg.get("frontres_segment_replay_enabled", False)) or objective == "segment_replay_hrl"
        return cls(
            requested=requested,
            live_runner_enabled=bool(alg_cfg.get("frontres_segment_live_runner_enabled", False)),
            live_sentinel_only=bool(alg_cfg.get("frontres_segment_live_sentinel_only", False)),
            live_probe_only=bool(alg_cfg.get("frontres_segment_live_probe_only", False)),
            live_storage_write_only=bool(alg_cfg.get("frontres_segment_live_storage_write_only", False)),
            live_single_update_only=bool(alg_cfg.get("frontres_segment_live_single_update_only", False)),
            live_update_loop_only=bool(alg_cfg.get("frontres_segment_live_update_loop_only", False)),
            live_train_enabled=bool(alg_cfg.get("frontres_segment_live_train_enabled", False)),
            live_update_steps=max(1, int(alg_cfg.get("frontres_segment_live_update_steps", 4))),
            objective=objective,
            segment_k=max(1, int(alg_cfg.get("frontres_segment_k", 1))),
            reset_mode=str(alg_cfg.get("frontres_segment_reset_mode", "auto")).lower(),
        )

    def assert_live_runner_ready(self) -> None:
        if not self.requested:
            return
        if not self.live_runner_enabled:
            raise NotImplementedError(
                "Stage 3 Segment Replay HRL is recognized, but live runner integration is disabled. "
                "Use frontres_segment_replay_toy_chain.py and boundary tests until PPO/live rollout wiring is implemented."
            )
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
            f"reset_mode={self.reset_mode} "
            "live_runner=True "
            "sentinel_only=True "
            "storage=independent "
            "ppo_action=delta_se3_6d "
            "training_update=disabled"
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
            f"update_steps={self.live_update_steps} "
            f"reset_mode={self.reset_mode} "
            "live_runner=True "
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
            f"update_steps={self.live_update_steps} "
            f"reset_mode={self.reset_mode} "
            "live_runner=True "
            "runner_learn=True "
            "storage=independent "
            "ppo_action=delta_se3_6d"
        )

    def build_fake_connector(
        self,
        *,
        dataset: Any,
        sampler: Any,
        reset_adapter: Any,
        action_projector: Any,
        reward: Any,
        rollout_fn: Callable[..., Any],
        transition_writer: Any | None = None,
        diagnostics_fn: Callable[..., Any] | None = None,
        log_formatter: Callable[[Any], str] | None = None,
        connector_cls: type | None = None,
    ) -> Any:
        if not self.requested:
            raise ValueError("fake Segment Replay connector requires frontres_segment_replay_enabled or segment_replay_hrl")
        if connector_cls is None:
            from rsl_rl.runners.frontres_segment_replay import FrontRESSegmentReplayConnector

            connector_cls = FrontRESSegmentReplayConnector
        return connector_cls(
            dataset=dataset,
            sampler=sampler,
            reset_adapter=reset_adapter,
            action_projector=action_projector,
            reward=reward,
            rollout_fn=rollout_fn,
            transition_writer=transition_writer,
            diagnostics_fn=diagnostics_fn,
            log_formatter=log_formatter,
            reset_mode=self.reset_mode,
            stage="stage3_segment_hrl",
            objective="segment_replay_hrl",
        )
