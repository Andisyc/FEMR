#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


update_loop_module = _load(
    "frontres_segment_live_update_loop",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_update_loop.py",
)
run_frontres_segment_live_update_loop = update_loop_module.run_frontres_segment_live_update_loop


class FakeBoundary:
    def __init__(
        self,
        *,
        live_update_loop_only: bool = True,
        live_train_enabled: bool = False,
        live_update_steps: int = 3,
    ) -> None:
        self.live_update_loop_only = live_update_loop_only
        self.live_train_enabled = live_train_enabled
        self.live_update_steps = live_update_steps


class FakeRunner:
    def __init__(
        self,
        summaries: list[dict],
        *,
        boundary: FakeBoundary | None = None,
        alg_update_steps: int | None = None,
    ) -> None:
        self._frontres_segment_replay_boundary = boundary or FakeBoundary(live_update_steps=len(summaries))
        self.alg = SimpleNamespace(frontres_training_objective="stage3_segment_hrl")
        if alg_update_steps is not None:
            self.alg.frontres_segment_live_update_steps = alg_update_steps
        self.summaries = summaries
        self.probe_init_flags: list[bool] = []

    def run_frontres_segment_live_probe(self, *, init_at_random_ep_len: bool) -> dict:
        self.probe_init_flags.append(init_at_random_ep_len)
        return self.summaries[len(self.probe_init_flags) - 1]


def _summary(
    *,
    ppo_update: bool,
    ppo_valid_count: int,
    reward_mean: float,
    storage_valid_frac: float,
    ppo_total_loss: float,
    ppo_actor_loss: float,
    ppo_value_loss: float,
    ppo_approx_kl: float,
    ppo_clip_frac: float,
) -> dict:
    return {
        "ppo_update": ppo_update,
        "ppo_valid_count": ppo_valid_count,
        "reward_mean": reward_mean,
        "storage_valid_frac": storage_valid_frac,
        "ppo_total_loss": ppo_total_loss,
        "ppo_actor_loss": ppo_actor_loss,
        "ppo_value_loss": ppo_value_loss,
        "ppo_approx_kl": ppo_approx_kl,
        "ppo_clip_frac": ppo_clip_frac,
    }


def test_live_update_loop_aggregates_probe_metrics_and_init_flag() -> None:
    runner = FakeRunner(
        [
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=1.0,
                storage_valid_frac=0.50,
                ppo_total_loss=10.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=2.0,
                ppo_approx_kl=0.01,
                ppo_clip_frac=0.10,
            ),
            _summary(
                ppo_update=False,
                ppo_valid_count=0,
                reward_mean=2.0,
                storage_valid_frac=0.25,
                ppo_total_loss=20.0,
                ppo_actor_loss=3.0,
                ppo_value_loss=4.0,
                ppo_approx_kl=0.02,
                ppo_clip_frac=0.20,
            ),
            _summary(
                ppo_update=True,
                ppo_valid_count=4,
                reward_mean=3.0,
                storage_valid_frac=1.00,
                ppo_total_loss=30.0,
                ppo_actor_loss=5.0,
                ppo_value_loss=6.0,
                ppo_approx_kl=0.03,
                ppo_clip_frac=0.30,
            ),
        ]
    )

    result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=True, runner_learn=True)

    assert runner.probe_init_flags == [True, False, False]
    assert result["update_steps"] == 3
    assert result["update_count"] == 2
    assert result["ppo_valid_count"] == 6
    assert result["reward_mean"] == 2.0
    assert result["storage_valid_frac"] == (0.50 + 0.25 + 1.00) / 3.0
    assert result["ppo_total_loss_mean"] == 20.0
    assert result["ppo_actor_loss_mean"] == 3.0
    assert result["ppo_value_loss_mean"] == 4.0
    assert abs(result["ppo_approx_kl_mean"] - 0.02) < 1e-8
    assert abs(result["ppo_clip_frac_mean"] - 0.20) < 1e-8


def test_live_update_loop_uses_algorithm_update_steps_override() -> None:
    runner = FakeRunner(
        [
            _summary(
                ppo_update=True,
                ppo_valid_count=1,
                reward_mean=1.0,
                storage_valid_frac=1.0,
                ppo_total_loss=1.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=1.0,
                ppo_approx_kl=0.0,
                ppo_clip_frac=0.0,
            ),
            _summary(
                ppo_update=True,
                ppo_valid_count=1,
                reward_mean=3.0,
                storage_valid_frac=1.0,
                ppo_total_loss=3.0,
                ppo_actor_loss=3.0,
                ppo_value_loss=3.0,
                ppo_approx_kl=0.0,
                ppo_clip_frac=0.0,
            ),
        ],
        boundary=FakeBoundary(live_update_steps=5),
        alg_update_steps=2,
    )

    result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=False)

    assert runner.probe_init_flags == [False, False]
    assert result["update_steps"] == 2
    assert result["reward_mean"] == 2.0


def test_live_update_loop_requires_enabled_boundary() -> None:
    runner = FakeRunner(
        [],
        boundary=FakeBoundary(live_update_loop_only=False, live_train_enabled=False, live_update_steps=1),
    )

    try:
        run_frontres_segment_live_update_loop(runner)
    except ValueError as exc:
        assert "frontres_segment_live_update_loop_only=True" in str(exc)
    else:
        raise AssertionError("update loop must reject disabled live runner boundary")


def test_live_update_loop_summary_print_rate_default_and_verbose() -> None:
    summaries = [
        _summary(
            ppo_update=True,
            ppo_valid_count=1,
            reward_mean=1.0,
            storage_valid_frac=1.0,
            ppo_total_loss=1.0,
            ppo_actor_loss=1.0,
            ppo_value_loss=1.0,
            ppo_approx_kl=0.0,
            ppo_clip_frac=0.0,
        )
        for _ in range(12)
    ]
    runner = FakeRunner(summaries, boundary=FakeBoundary(live_update_steps=1))

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for _ in range(12):
            run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=False)
    output = buffer.getvalue()
    default_count = output.count("[FrontRES Segment Live Update Loop]")
    print(
        "[probe step5] update_loop_log_rate: "
        f"default_count={default_count} "
        f"call_count={runner._frontres_segment_live_update_loop_summary_count}",
        flush=True,
    )

    assert default_count == 4
    assert runner._frontres_segment_live_update_loop_summary_count == 12

    verbose_runner = FakeRunner(summaries[:4], boundary=FakeBoundary(live_update_steps=1))
    verbose_runner.alg.frontres_segment_verbose_probe = True
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for _ in range(4):
            run_frontres_segment_live_update_loop(verbose_runner, init_at_random_ep_len=False)
    verbose_count = buffer.getvalue().count("[FrontRES Segment Live Update Loop]")
    print(
        "[probe step5] update_loop_log_verbose_rate: "
        f"verbose_count={verbose_count} "
        f"verbose={verbose_runner.alg.frontres_segment_verbose_probe}",
        flush=True,
    )

    assert verbose_count == 4


if __name__ == "__main__":
    test_live_update_loop_aggregates_probe_metrics_and_init_flag()
    test_live_update_loop_uses_algorithm_update_steps_override()
    test_live_update_loop_requires_enabled_boundary()
    test_live_update_loop_summary_print_rate_default_and_verbose()
    print("frontres_segment_live_update_loop_contract: ok")
