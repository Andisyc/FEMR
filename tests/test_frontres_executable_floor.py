from __future__ import annotations

import pathlib
import sys
import importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "frontres_executable_floor.py"

spec = importlib.util.spec_from_file_location("frontres_executable_floor", MODULE_PATH)
frontres_executable_floor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frontres_executable_floor
spec.loader.exec_module(frontres_executable_floor)

ExecutableFloorState = frontres_executable_floor.ExecutableFloorState
resolve_executable_floor = frontres_executable_floor.resolve_executable_floor
update_executable_floor_stats = frontres_executable_floor.update_executable_floor_stats


def _cfg(**overrides):
    cfg = {
        "frontres_executable_floor_score": 0.0,
        "frontres_executable_floor_safe_margin": 0.05,
        "frontres_executable_floor_adaptive_enabled": True,
        "frontres_executable_floor_min_samples": 2.0,
        "frontres_executable_floor_ema_alpha": 0.5,
    }
    cfg.update(overrides)
    return cfg


def test_fixed_floor_until_both_sides_are_mature():
    values = resolve_executable_floor(
        _cfg(),
        ExecutableFloorState(safe_score_ema=0.20, broken_score_ema=-0.10, safe_count=2.0, broken_count=1.0),
    )

    assert values.floor == 0.0
    assert values.safe_floor == 0.05
    assert values.source == "fixed"
    assert values.adaptive == 0.0


def test_adaptive_floor_uses_safe_broken_midpoint():
    values = resolve_executable_floor(
        _cfg(),
        ExecutableFloorState(safe_score_ema=0.20, broken_score_ema=-0.10, safe_count=2.0, broken_count=2.0),
    )

    assert abs(values.floor - 0.05) < 1e-6
    assert abs(values.safe_floor - 0.10) < 1e-6
    assert values.source == "adaptive"
    assert values.adaptive == 1.0


def test_frontier_update_uses_only_frontier_bucket_and_ignores_timeouts():
    try:
        import torch
    except ModuleNotFoundError:
        return

    exec_score = torch.tensor([0.30, -0.20, 0.90, -0.80, 0.40])
    done = torch.tensor([False, True, False, True, False])
    timeout = torch.tensor([False, False, False, True, False])
    mix_class = torch.tensor([1, 1, 0, 1, 1])

    state, values = update_executable_floor_stats(
        _cfg(frontres_executable_floor_min_samples=1.0),
        ExecutableFloorState(),
        exec_score,
        done=done,
        timeout=timeout,
        mix_class=mix_class,
        frontier_decision="frontier",
    )

    assert abs(float(state.safe_score_ema) - 0.35) < 1e-6
    assert abs(float(state.broken_score_ema) - (-0.20)) < 1e-6
    assert state.safe_count == 2.0
    assert state.broken_count == 1.0
    assert abs(values.floor - 0.075) < 1e-6
    assert values.source == "adaptive"


if __name__ == "__main__":
    test_fixed_floor_until_both_sides_are_mature()
    test_adaptive_floor_uses_safe_broken_midpoint()
    test_frontier_update_uses_only_frontier_bucket_and_ignores_timeouts()
