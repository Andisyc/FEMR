from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FrontRESSegmentRunnerBoundary:
    requested: bool
    live_runner_enabled: bool
    live_sentinel_only: bool
    v015_local_sentinel_only: bool
    live_probe_only: bool
    live_storage_write_only: bool
    live_single_update_only: bool
    live_update_loop_only: bool
    offline_eval_only: bool
    sequence_offline_eval_only: bool
    live_train_enabled: bool
    live_update_steps: int
    periodic_eval_enabled: bool
    periodic_eval_interval: int
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
        requested = bool(alg_cfg.get("frontres_segment_replay_enabled", False)) or objective == "segment_replay_hrl"
        return cls(
            requested=requested,
            live_runner_enabled=bool(alg_cfg.get("frontres_segment_live_runner_enabled", False)),
            live_sentinel_only=bool(alg_cfg.get("frontres_segment_live_sentinel_only", False)),
            v015_local_sentinel_only=bool(alg_cfg.get("frontres_v015_local_sentinel_only", False)),
            live_probe_only=bool(alg_cfg.get("frontres_segment_live_probe_only", False)),
            live_storage_write_only=bool(alg_cfg.get("frontres_segment_live_storage_write_only", False)),
            live_single_update_only=bool(alg_cfg.get("frontres_segment_live_single_update_only", False)),
            live_update_loop_only=bool(alg_cfg.get("frontres_segment_live_update_loop_only", False)),
            offline_eval_only=bool(alg_cfg.get("frontres_segment_offline_eval_only", False)),
            sequence_offline_eval_only=bool(alg_cfg.get("frontres_segment_sequence_offline_eval_only", False)),
            live_train_enabled=bool(alg_cfg.get("frontres_segment_live_train_enabled", False)),
            live_update_steps=max(1, int(alg_cfg.get("frontres_segment_live_update_steps", 4))),
            periodic_eval_enabled=bool(alg_cfg.get("frontres_segment_periodic_eval_enabled", False)),
            periodic_eval_interval=max(1, int(alg_cfg.get("frontres_segment_periodic_eval_interval", 100))),
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

    def assert_live_runner_ready(self) -> None:
        if not self.requested:
            return
        if not self.live_runner_enabled:
            raise NotImplementedError(
                "Stage 3 Segment Replay HRL is recognized, but live runner integration is disabled. "
                "Use frontres_segment_replay_toy_chain.py and boundary tests until PPO/live rollout wiring is implemented."
            )
        if self.v015_local_sentinel_only:
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
                    self.offline_eval_only,
                    self.sequence_offline_eval_only,
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
            or self.offline_eval_only
            or self.sequence_offline_eval_only
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

    def v015_sentinel_log(self) -> str | None:
        if not (self.requested and self.live_runner_enabled and self.v015_local_sentinel_only):
            return None
        return (
            "[FrontRES v015 Local Sentinel] "
            f"objective={self.objective} "
            f"segment_k={self.segment_k} "
            f"max_horizon_k={self.max_horizon_k} "
            "v015_local_sentinel=True "
            f"future_offsets={self.future_offsets} "
            "legacy_modes=False"
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
                or self.offline_eval_only
                or self.sequence_offline_eval_only
            )
        ):
            return None
        storage_write = (
            "True"
            if self.live_storage_write_only or self.live_single_update_only or self.live_update_loop_only
            else "False"
        )
        ppo_update = "True" if self.live_single_update_only or self.live_update_loop_only else "False"
        mode = "sequence_eval" if self.sequence_offline_eval_only else ("offline_eval" if self.offline_eval_only else "probe")
        return (
            "[FrontRES Segment Live Probe Ready] "
            f"objective={self.objective} "
            f"segment_k={self.segment_k} "
            f"max_horizon_k={self.max_horizon_k} "
            f"update_steps={self.live_update_steps} "
            f"reset_mode={self.reset_mode} "
            "live_runner=True "
            f"mode={mode} "
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
            f"periodic_eval={self.periodic_eval_enabled} "
            f"eval_interval={self.periodic_eval_interval}"
        )
