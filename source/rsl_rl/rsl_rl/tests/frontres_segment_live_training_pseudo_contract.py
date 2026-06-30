#!/usr/bin/env python3
"""Pseudo-parameter contract for Stage 3 Segment Replay live training.

This test uses fake runner arguments instead of IsaacLab so interface mistakes
are caught before the first real server run.
"""
from __future__ import annotations

import contextlib
import io
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


live_training_module = _load(
    "frontres_segment_live_training",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_training.py",
)

run_frontres_segment_live_training_loop = live_training_module.run_frontres_segment_live_training_loop


def _probe_summary(name: str, summary: dict) -> None:
    print(
        f"[probe step5] {name}: "
        f"update_steps={summary.get('update_steps')} "
        f"update_count={summary.get('update_count')} "
        f"ppo_valid_count={summary.get('ppo_valid_count')} "
        f"reward_mean={summary.get('reward_mean')} "
        f"storage_valid_frac={summary.get('storage_valid_frac')} "
        f"ppo_total_loss_mean={summary.get('ppo_total_loss_mean')} "
        f"ppo_actor_loss_mean={summary.get('ppo_actor_loss_mean')} "
        f"ppo_value_loss_mean={summary.get('ppo_value_loss_mean')} "
        f"ppo_approx_kl_mean={summary.get('ppo_approx_kl_mean')} "
        f"ppo_clip_frac_mean={summary.get('ppo_clip_frac_mean')}",
        flush=True,
    )


def _probe_runner_state(name: str, runner: "FakeRunner") -> None:
    print(
        f"[probe step5] {name}: "
        f"current_learning_iteration={runner.current_learning_iteration} "
        f"update_calls={runner.update_calls} "
        f"saved_paths={runner.saved_paths} "
        f"probe_record_count={len(runner.probe_records)} "
        f"probe_checkpoint_paths={[path for _, path in runner.probe_records]}",
        flush=True,
    )


def _probe_exception(name: str, exc: Exception) -> None:
    print(f"[probe step5] {name}: exception={type(exc).__name__} message={exc}", flush=True)


class FakeBoundary:
    def __init__(self, live_train_enabled: bool = True):
        self.live_train_enabled = live_train_enabled


class FakeAlg:
    def __init__(
        self,
        *,
        fail_on_invalid_update: bool = True,
        min_valid_count: int = 1,
        fail_on_nonfinite: bool = True,
    ):
        self.frontres_segment_live_fail_on_invalid_update = fail_on_invalid_update
        self.frontres_segment_live_min_valid_count = min_valid_count
        self.frontres_segment_live_fail_on_nonfinite = fail_on_nonfinite


def _full_summary(**overrides) -> dict:
    summary = {
        "update_steps": 4,
        "update_count": 4,
        "ppo_valid_count": 8,
        "reward_mean": 0.25,
        "storage_valid_frac": 1.0,
        "ppo_total_loss_mean": 0.5,
        "ppo_actor_loss_mean": 0.1,
        "ppo_value_loss_mean": 0.2,
        "ppo_approx_kl_mean": 0.01,
        "ppo_clip_frac_mean": 0.0,
    }
    summary.update(overrides)
    return summary


class FakeRunner:
    def __init__(
        self,
        *,
        log_dir: str | None = "/tmp/frontres-pseudo",
        live_train_enabled: bool = True,
        fail_on_invalid_update: bool = True,
        min_valid_count: int = 1,
        fail_on_nonfinite: bool = True,
    ):
        self._frontres_segment_replay_boundary = FakeBoundary(live_train_enabled=live_train_enabled)
        self.alg = FakeAlg(
            fail_on_invalid_update=fail_on_invalid_update,
            min_valid_count=min_valid_count,
            fail_on_nonfinite=fail_on_nonfinite,
        )
        self.current_learning_iteration = 0
        self.log_dir = log_dir
        self.disable_logs = False
        self.save_interval = 1
        self.update_calls: list[tuple[bool, bool]] = []
        self.saved_paths: list[str] = []
        self.probe_records: list[tuple[dict, str]] = []

    def run_frontres_segment_live_update_loop(self, *, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        self.update_calls.append((init_at_random_ep_len, runner_learn))
        call_id = len(self.update_calls)
        summary = _full_summary(
            ppo_valid_count=8 * call_id,
            reward_mean=0.25 * call_id,
            ppo_total_loss_mean=0.5 * call_id,
            ppo_actor_loss_mean=0.1 * call_id,
            ppo_value_loss_mean=0.2 * call_id,
            ppo_approx_kl_mean=0.01 * call_id,
        )
        _probe_summary(f"update_loop_summary_{call_id}", summary)
        return summary

    def save(self, path: str) -> None:
        self.saved_paths.append(path)

    def _record_frontres_checkpoint_probe(self, locs: dict, checkpoint_path: str) -> None:
        self.probe_records.append((locs, checkpoint_path))


def test_pseudo_live_training_runs_two_iterations_and_saves_checkpoints() -> None:
    runner = FakeRunner()
    run_frontres_segment_live_training_loop(
        runner,
        num_learning_iterations=2,
        init_at_random_ep_len=True,
    )
    _probe_runner_state("after_two_iteration_training", runner)
    assert runner.current_learning_iteration == 2
    assert runner.update_calls == [(True, True), (False, True)]
    assert runner.saved_paths[0].endswith("model_1.pt")
    assert runner.saved_paths[1].endswith("model_2.pt")
    assert runner.saved_paths[-1].endswith("model_2.pt")
    assert len(runner.probe_records) == 2
    assert runner.probe_records[0][0]["update_steps"] == 4
    assert runner.probe_records[1][0]["ppo_valid_count"] == 16


def test_pseudo_live_training_zero_iterations_does_not_touch_update_loop() -> None:
    runner = FakeRunner()
    run_frontres_segment_live_training_loop(
        runner,
        num_learning_iterations=0,
        init_at_random_ep_len=True,
    )
    _probe_runner_state("after_zero_iteration_training", runner)
    assert runner.current_learning_iteration == 0
    assert runner.update_calls == []
    assert runner.saved_paths == []
    assert runner.probe_records == []


def test_pseudo_live_training_requires_train_flag() -> None:
    runner = FakeRunner(live_train_enabled=False)
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except ValueError as exc:
        _probe_exception("requires_train_flag", exc)
        assert "frontres_segment_live_train_enabled=True" in str(exc)
    else:
        raise AssertionError("live training must reject fake runners without the train flag")


def test_pseudo_live_training_rejects_incomplete_summary() -> None:
    runner = FakeRunner()

    def bad_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = {"update_steps": 4}
        _probe_summary("incomplete_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = bad_update_loop
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except KeyError as exc:
        _probe_exception("rejects_incomplete_summary", exc)
        assert "missing keys" in str(exc)
        assert "ppo_valid_count" in str(exc)
    else:
        raise AssertionError("live training must reject incomplete update summaries")


def test_pseudo_live_training_rejects_nonfinite_summary() -> None:
    runner = FakeRunner()

    def bad_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = _full_summary(ppo_total_loss_mean=float("nan"))
        _probe_summary("nonfinite_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = bad_update_loop
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except FloatingPointError as exc:
        _probe_exception("rejects_nonfinite_summary", exc)
        assert "non-finite" in str(exc)
        assert "ppo_total_loss_mean" in str(exc)
    else:
        raise AssertionError("live training must reject non-finite update summaries")


def test_pseudo_live_training_rejects_zero_update_count() -> None:
    runner = FakeRunner()

    def bad_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = _full_summary(update_count=0, ppo_valid_count=8)
        _probe_summary("zero_update_count_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = bad_update_loop
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except RuntimeError as exc:
        _probe_exception("rejects_zero_update_count", exc)
        assert "update_count=0" in str(exc)
    else:
        raise AssertionError("live training must reject empty update iterations")


def test_pseudo_live_training_rejects_too_few_valid_samples() -> None:
    runner = FakeRunner(min_valid_count=2)

    def bad_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = _full_summary(update_count=1, ppo_valid_count=1)
        _probe_summary("too_few_valid_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = bad_update_loop
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except RuntimeError as exc:
        _probe_exception("rejects_too_few_valid_samples", exc)
        assert "too few valid PPO samples" in str(exc)
        assert "ppo_valid_count=1" in str(exc)
    else:
        raise AssertionError("live training must reject iterations with too few valid PPO samples")


def test_pseudo_live_training_can_disable_fail_fast_guards() -> None:
    runner = FakeRunner(
        fail_on_invalid_update=False,
        fail_on_nonfinite=False,
    )

    def unchecked_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = _full_summary(update_count=0, ppo_valid_count=0, ppo_total_loss_mean=float("nan"))
        _probe_summary("unchecked_invalid_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = unchecked_update_loop
    run_frontres_segment_live_training_loop(
        runner,
        num_learning_iterations=1,
        init_at_random_ep_len=True,
    )
    _probe_runner_state("after_disabled_fail_fast_training", runner)
    assert runner.current_learning_iteration == 1
    assert runner.update_calls == [(True, True)]


def test_pseudo_live_training_log_formats_large_loss_readably() -> None:
    runner = FakeRunner(fail_on_nonfinite=True)

    def large_loss_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        return _full_summary(
            ppo_total_loss_mean=1.5157918219343223e23,
            ppo_actor_loss_mean=1.5157918219343223e23,
            ppo_value_loss_mean=0.00114,
            ppo_approx_kl_mean=-0.004483,
            ppo_clip_frac_mean=0.376726,
        )

    runner.run_frontres_segment_live_update_loop = large_loss_update_loop
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=False,
        )
    output = buffer.getvalue()
    print(f"[probe readable_log] live_train_block={output.strip().splitlines()[:4]}", flush=True)

    assert "[FrontRES Segment Live Train]" in output
    assert "  progress: iter=1/1 updates=4/4 runner_learn=True" in output
    assert "  data: valid=8 valid_frac=100.0% reward=0.250000" in output
    assert "  ppo: loss_total=1.516e+23" in output
    assert "loss_total=1.516e+23" in output
    assert "actor=1.516e+23" in output
    assert "clip=37.7%" in output
    assert "status=BAD_LOSS_EXPLOSION" in output
    assert "151579182193432229576704.000000" not in output


def main() -> None:
    test_pseudo_live_training_runs_two_iterations_and_saves_checkpoints()
    test_pseudo_live_training_zero_iterations_does_not_touch_update_loop()
    test_pseudo_live_training_requires_train_flag()
    test_pseudo_live_training_rejects_incomplete_summary()
    test_pseudo_live_training_rejects_nonfinite_summary()
    test_pseudo_live_training_rejects_zero_update_count()
    test_pseudo_live_training_rejects_too_few_valid_samples()
    test_pseudo_live_training_can_disable_fail_fast_guards()
    test_pseudo_live_training_log_formats_large_loss_readably()
    print("result: PASS")


if __name__ == "__main__":
    main()
