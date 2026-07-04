from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Sequence


@dataclass(frozen=True)
class FrontRESSegmentSequenceEvalItem:
    segment_id: int
    motion_id: str
    reset_frame: int
    preroll_steps: int
    eval_start_frame: int
    eval_rollout_steps: int
    segment_horizon_k: int

    @property
    def eval_end_frame(self) -> int:
        return self.eval_start_frame + self.eval_rollout_steps


@dataclass(frozen=True)
class FrontRESSegmentSequenceEvalPlan:
    items: tuple[FrontRESSegmentSequenceEvalItem, ...]
    requested_sequences: int
    available_envs: int
    paired_envs_per_sequence: int
    chunk_capacity: int
    max_preroll_steps: int | None = None

    @property
    def sequence_count(self) -> int:
        return len(self.items)

    @property
    def chunk_count(self) -> int:
        return (self.sequence_count + self.chunk_capacity - 1) // self.chunk_capacity

    @property
    def motion_ids(self) -> tuple[str, ...]:
        return tuple(item.motion_id for item in self.items)


def build_frontres_sequence_eval_plan(
    specs: Sequence[Any],
    *,
    requested_sequences: int = 10,
    available_envs: int = 0,
    paired_envs_per_sequence: int = 4,
    eval_rollout_steps: int | None = None,
    max_preroll_steps: int | None = None,
) -> FrontRESSegmentSequenceEvalPlan:
    if requested_sequences <= 0:
        raise ValueError("requested_sequences must be positive")
    if paired_envs_per_sequence <= 0:
        raise ValueError("paired_envs_per_sequence must be positive")

    preroll_cap = None if max_preroll_steps is None or int(max_preroll_steps) <= 0 else int(max_preroll_steps)

    # FRS3-EVAL-004: choose unique motion sequences and derive frame0->segment eval windows.
    items: list[FrontRESSegmentSequenceEvalItem] = []
    seen: set[str] = set()
    for index, spec in enumerate(specs):
        motion_id = str(getattr(spec, "motion_id", ""))
        if not motion_id:
            raise ValueError("sequence eval specs must expose motion_id")
        start_frame = _required_nonnegative_int(spec, "start_frame")
        if preroll_cap is not None and start_frame > preroll_cap:
            continue
        if motion_id in seen:
            continue
        seen.add(motion_id)
        horizon_k = _positive_int(getattr(spec, "horizon_k", 1), "horizon_k")
        rollout_steps = _positive_int(eval_rollout_steps if eval_rollout_steps is not None else horizon_k, "eval_rollout_steps")
        items.append(
            FrontRESSegmentSequenceEvalItem(
                segment_id=int(getattr(spec, "segment_id", index)),
                motion_id=motion_id,
                reset_frame=0,
                preroll_steps=start_frame,
                eval_start_frame=start_frame,
                eval_rollout_steps=rollout_steps,
                segment_horizon_k=horizon_k,
            )
        )
        if len(items) >= requested_sequences:
            break

    if len(items) < requested_sequences:
        cap_note = "" if preroll_cap is None else f" with max_preroll_steps<={preroll_cap}"
        raise ValueError(
            f"sequence eval requires {requested_sequences} unique motion ids{cap_note}, got {len(items)}"
        )

    envs = max(0, int(available_envs))
    chunk_capacity = len(items) if envs <= 0 else max(1, envs // int(paired_envs_per_sequence))
    return FrontRESSegmentSequenceEvalPlan(
        items=tuple(items),
        requested_sequences=int(requested_sequences),
        available_envs=envs,
        paired_envs_per_sequence=int(paired_envs_per_sequence),
        chunk_capacity=chunk_capacity,
        max_preroll_steps=preroll_cap,
    )


def segment_ids_for_sequence_eval_item(
    item: FrontRESSegmentSequenceEvalItem,
    *,
    env_count: int,
) -> tuple[int, ...]:
    # FRS3-EVAL-005: repeat one segment across the full B1 role layout.
    count = _positive_int(env_count, "env_count")
    return tuple(int(item.segment_id) for _ in range(count))


def build_frontres_sequence_eval_reset_batch(
    batch: Any,
    item: FrontRESSegmentSequenceEvalItem,
) -> Any:
    # FRS3-EVAL-006: rewrite reset specs to motion frame 0 before preroll.
    specs = tuple(getattr(batch, "specs", ()) or ())
    if not specs:
        raise ValueError("sequence eval reset batch requires specs")
    reset_specs = tuple(_replace_spec_start_frame(spec, item.reset_frame) for spec in specs)
    if is_dataclass(batch):
        reset_batch = replace(batch, specs=reset_specs)
        _copy_sequence_eval_dynamic_attrs(batch, reset_batch)
        return reset_batch
    values = dict(vars(batch))
    values["specs"] = reset_specs
    return SimpleNamespace(**values)


def _copy_sequence_eval_dynamic_attrs(src: Any, dst: Any) -> None:
    for name in (
        "stage3_index_perturbation_family",
        "stage3_index_perturbation_strength",
        "stage3_index_perturbation_plan",
    ):
        if hasattr(src, name):
            object.__setattr__(dst, name, getattr(src, name))


def _replace_spec_start_frame(spec: Any, start_frame: int) -> Any:
    if is_dataclass(spec):
        changes: dict[str, Any] = {"start_frame": int(start_frame)}
        if hasattr(spec, "start_time"):
            changes["start_time"] = 0.0
        if hasattr(spec, "phase"):
            changes["phase"] = 0.0
        return replace(spec, **changes)
    values = dict(vars(spec))
    values["start_frame"] = int(start_frame)
    if "start_time" in values:
        values["start_time"] = 0.0
    if "phase" in values:
        values["phase"] = 0.0
    return SimpleNamespace(**values)


def _required_nonnegative_int(spec: Any, name: str) -> int:
    value = getattr(spec, name, None)
    if value is None:
        raise ValueError(f"sequence eval spec requires {name}")
    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{name} must be non-negative")
    return value_int


def _positive_int(value: Any, name: str) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{name} must be positive")
    return value_int
