#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sequence_eval_module = _load(
    "frontres_segment_sequence_eval",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_sequence_eval.py",
)

build_frontres_sequence_eval_plan = sequence_eval_module.build_frontres_sequence_eval_plan
build_frontres_sequence_eval_reset_batch = sequence_eval_module.build_frontres_sequence_eval_reset_batch
segment_ids_for_sequence_eval_item = sequence_eval_module.segment_ids_for_sequence_eval_item

live_training_module = _load(
    "frontres_segment_live_training",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_training.py",
)
run_frontres_segment_sequence_offline_eval = live_training_module.run_frontres_segment_sequence_offline_eval


@dataclass(frozen=True)
class FakeSpec:
    segment_id: int
    motion_id: str
    start_frame: int
    horizon_k: int


def test_sequence_eval_prerolls_from_motion_start() -> None:
    plan = build_frontres_sequence_eval_plan(
        (
            FakeSpec(1, "walk_a", 12, 4),
            FakeSpec(2, "walk_a", 99, 4),
            FakeSpec(3, "walk_b", 0, 5),
            FakeSpec(4, "walk_c", 25, 6),
        ),
        requested_sequences=3,
        available_envs=8,
        eval_rollout_steps=50,
    )
    assert plan.motion_ids == ("walk_a", "walk_b", "walk_c")
    assert plan.sequence_count == 3
    assert plan.chunk_capacity == 2
    assert plan.chunk_count == 2
    assert tuple(item.reset_frame for item in plan.items) == (0, 0, 0)
    assert tuple(item.preroll_steps for item in plan.items) == (12, 0, 25)
    assert tuple(item.eval_start_frame for item in plan.items) == (12, 0, 25)
    assert tuple(item.eval_end_frame for item in plan.items) == (62, 50, 75)


def test_sequence_count_is_not_env_count() -> None:
    specs = tuple(FakeSpec(i, f"motion_{i}", i + 1, 4) for i in range(10))
    plan = build_frontres_sequence_eval_plan(specs, requested_sequences=10, available_envs=8)
    assert plan.sequence_count == 10
    assert plan.chunk_capacity == 2
    assert plan.chunk_count == 5


def test_sequence_eval_requires_unique_motion_ids() -> None:
    try:
        build_frontres_sequence_eval_plan(
            (
                FakeSpec(1, "same_motion", 10, 4),
                FakeSpec(2, "same_motion", 20, 4),
            ),
            requested_sequences=2,
        )
    except ValueError as exc:
        assert "unique motion ids" in str(exc)
    else:
        raise AssertionError("duplicate-only specs must not satisfy sequence eval")


def test_sequence_eval_reset_batch_rewrites_start_only() -> None:
    item = build_frontres_sequence_eval_plan(
        (FakeSpec(7, "walk_reset", 31, 4),),
        requested_sequences=1,
    ).items[0]
    batch = SimpleNamespace(specs=(FakeSpec(7, "walk_reset", 31, 4),))
    reset_batch = build_frontres_sequence_eval_reset_batch(batch, item)
    assert reset_batch.specs[0].start_frame == 0
    assert batch.specs[0].start_frame == 31
    assert segment_ids_for_sequence_eval_item(item, env_count=8) == (7,) * 8


class FakeSampler:
    def __init__(self):
        self.calls: list[int] = []

    def sample(self, batch_size: int):
        self.calls.append(int(batch_size))
        return SimpleNamespace(segment_ids=torch.arange(batch_size, dtype=torch.long))


class FakeRunner:
    def __init__(self):
        self.env = SimpleNamespace(num_envs=8)
        self.device = torch.device("cpu")
        self._frontres_segment_sampler = FakeSampler()
        self._frontres_segment_live_current_sample = None
        self._frontres_segment_live_current_batch = None
        self._frontres_segment_live_current_reset_request = None
        self._frontres_segment_live_current_reset_result = None
        self._frontres_segment_live_detail_log_enabled = True
        self.events: list[tuple[str, object]] = []

    def eval_mode(self):
        self.events.append(("eval_mode", None))


class FakeCapture:
    rollout_k = 50
    reward_mean = 0.0
    done_frac = 0.0
    last_obs_shape = (8, 29)
    action_shape = (8, 6)
    env_action_shape = (8, 12)
    transition_obs = None
    transition_privileged_obs = None
    transition_actions = torch.zeros((8, 6))
    transition_log_probs = None
    transition_values = None
    transition_means = None
    transition_sigmas = None
    reward_accum = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.2, 0.2, 0.6, 0.6])
    done_any = torch.zeros(8, dtype=torch.bool)
    actor_update_mask = None
    n_train = 2
    n_candidate = 2
    n_base = 2
    n_clean = 2
    survival_steps = torch.full((8,), 50.0)
    motion_clean_body_pos = None
    motion_repaired_body_pos = None
    motion_noisy_body_pos = None


def test_sequence_offline_eval_owner_orders_reset_preroll_eval() -> None:
    runner = FakeRunner()

    def fake_build_current_segment_batch(runner_arg, sample, *, update_step: int, print_probe: bool):
        specs = tuple(
            FakeSpec(
                int(segment_id),
                f"motion_{int(segment_id)}",
                int(segment_id) + 3,
                4,
            )
            for segment_id in sample.segment_ids.detach().cpu().tolist()
        )
        return SimpleNamespace(segment_ids=sample.segment_ids, specs=specs)

    def fake_apply_current_segment_reset(runner_arg):
        starts = tuple(spec.start_frame for spec in runner_arg._frontres_segment_live_current_batch.specs)
        runner_arg.events.append(("reset", starts))

    def fake_read_live_observations(runner_arg):
        runner_arg.events.append(("read_obs", None))
        return "obs"

    def fake_run_live_rollout_capture(runner_arg, observations, *, rollout_steps: int):
        starts = tuple(spec.start_frame for spec in runner_arg._frontres_segment_live_current_batch.specs)
        runner_arg.events.append(("rollout", int(rollout_steps), starts))
        return FakeCapture()

    live_training_module._build_current_segment_batch = fake_build_current_segment_batch
    live_training_module.build_frontres_sequence_eval_plan = build_frontres_sequence_eval_plan
    live_training_module.build_frontres_sequence_eval_reset_batch = build_frontres_sequence_eval_reset_batch
    live_training_module.segment_ids_for_sequence_eval_item = segment_ids_for_sequence_eval_item
    live_training_module._apply_current_segment_reset = fake_apply_current_segment_reset
    live_training_module._read_live_observations = fake_read_live_observations
    live_training_module._run_live_rollout_capture = fake_run_live_rollout_capture

    summary = run_frontres_segment_sequence_offline_eval(
        runner,
        num_eval_sequences=2,
        rollout_steps=50,
    )
    reset_events = [event for event in runner.events if event[0] == "reset"]
    rollout_events = [event for event in runner.events if event[0] == "rollout"]
    assert len(reset_events) == 2
    assert all(set(event[1]) == {0} for event in reset_events)
    assert rollout_events[0][1:] == (3, (0, 0, 0, 0, 0, 0, 0, 0))
    assert rollout_events[1][1:] == (50, (3, 3, 3, 3, 3, 3, 3, 3))
    assert rollout_events[2][1:] == (4, (0, 0, 0, 0, 0, 0, 0, 0))
    assert rollout_events[3][1:] == (50, (4, 4, 4, 4, 4, 4, 4, 4))
    assert int(summary["sequence_count"]) == 2
    assert summary["motion_ids"] == ("motion_0", "motion_1")


def main() -> None:
    test_sequence_eval_prerolls_from_motion_start()
    test_sequence_count_is_not_env_count()
    test_sequence_eval_requires_unique_motion_ids()
    test_sequence_eval_reset_batch_rewrites_start_only()
    test_sequence_offline_eval_owner_orders_reset_preroll_eval()
    print(
        "[probe step23] sequence_eval_contract "
        "reset_frame=0 preroll_to_segment_start=True requested_sequences_not_env_count=True",
        flush=True,
    )
    print(
        "[probe step24] sequence_eval_live_owner "
        "reset_before_preroll=True preroll_before_eval=True role_envs_repeated=True",
        flush=True,
    )
    print("frontres_segment_sequence_eval_contract: ok")


if __name__ == "__main__":
    main()
