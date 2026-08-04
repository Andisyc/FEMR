from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


boundary_module = _load(
    "frontres_segment_runner_boundary",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_runner_boundary.py",
)
FrontRESSegmentRunnerBoundary = boundary_module.FrontRESSegmentRunnerBoundary
FrontRESStartupLifecycle = boundary_module.FrontRESStartupLifecycle


def _stage3_cfg(
    live: bool = False,
    sentinel: bool = False,
    probe: bool = False,
    storage: bool = False,
    single_update: bool = False,
    update_loop: bool = False,
    train: bool = False,
) -> dict:
    return {
        "algorithm": {
            "frontres_training_objective": "segment_replay_hrl",
            "frontres_segment_replay_enabled": True,
            "frontres_segment_live_runner_enabled": live,
            "frontres_segment_live_sentinel_only": sentinel,
            "frontres_segment_live_probe_only": probe,
            "frontres_segment_live_storage_write_only": storage,
            "frontres_segment_live_single_update_only": single_update,
            "frontres_segment_live_update_loop_only": update_loop,
            "frontres_segment_live_train_enabled": train,
            "frontres_segment_live_update_steps": 4,
            "frontres_segment_k": 8,
            "frontres_segment_max_horizon_k": 64,
            "frontres_segment_reset_mode": "auto",
        }
    }


def test_stage3_boundary_rejects_live_runner_by_default() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=False))
    assert boundary.requested
    assert boundary.segment_k == 8
    assert boundary.max_horizon_k == 64
    try:
        boundary.assert_live_runner_ready()
    except NotImplementedError as exc:
        assert "live runner integration is disabled" in str(exc)
    else:
        raise AssertionError("Stage 3 live runner must fail fast while integration is not wired")


def test_stage3_boundary_rejects_live_flag_until_ppo_wiring_exists() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=True))
    try:
        boundary.assert_live_runner_ready()
    except NotImplementedError as exc:
        assert "PPO/live rollout wiring is still not implemented" in str(exc)
    else:
        raise AssertionError("Stage 3 live runner flag must still fail before PPO wiring")


def test_stage3_boundary_allows_live_sentinel_only() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=True, sentinel=True))
    boundary.assert_live_runner_ready()
    log = boundary.sentinel_log()
    assert log is not None
    assert "FrontRES Segment Live Sentinel" in log
    assert "objective=segment_replay_hrl" in log
    assert "segment_k=8" in log
    assert "max_horizon_k=64" in log
    assert "reset_mode=auto" in log
    assert "live_runner=True" in log
    assert "sentinel_only=True" in log
    assert "storage=independent" in log
    assert "ppo_action=delta_se3_6d" in log
    assert "training_update=disabled" in log


def test_stage3_boundary_allows_live_probe_only() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=True, probe=True))
    boundary.assert_live_runner_ready()
    log = boundary.probe_log()
    assert log is not None
    assert "FrontRES Segment Live Probe Ready" in log
    assert "objective=segment_replay_hrl" in log
    assert "segment_k=8" in log
    assert "max_horizon_k=64" in log
    assert "reset_mode=auto" in log
    assert "live_runner=True" in log
    assert "probe_only=True" in log
    assert "storage_write=False" in log
    assert "ppo_update=False" in log


def test_stage3_boundary_allows_live_storage_write_only() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=True, storage=True))
    boundary.assert_live_runner_ready()
    log = boundary.probe_log()
    assert log is not None
    assert "FrontRES Segment Live Probe Ready" in log
    assert "objective=segment_replay_hrl" in log
    assert "segment_k=8" in log
    assert "max_horizon_k=64" in log
    assert "reset_mode=auto" in log
    assert "live_runner=True" in log
    assert "probe_only=True" in log
    assert "storage_write=True" in log
    assert "ppo_update=False" in log


def test_stage3_boundary_allows_live_single_update_only() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=True, single_update=True))
    boundary.assert_live_runner_ready()
    log = boundary.probe_log()
    assert log is not None
    assert "FrontRES Segment Live Probe Ready" in log
    assert "objective=segment_replay_hrl" in log
    assert "segment_k=8" in log
    assert "max_horizon_k=64" in log
    assert "reset_mode=auto" in log
    assert "live_runner=True" in log
    assert "probe_only=True" in log
    assert "storage_write=True" in log
    assert "ppo_update=True" in log


def test_stage3_boundary_allows_live_update_loop_only() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=True, update_loop=True))
    boundary.assert_live_runner_ready()
    log = boundary.probe_log()
    assert log is not None
    assert "FrontRES Segment Live Probe Ready" in log
    assert "objective=segment_replay_hrl" in log
    assert "segment_k=8" in log
    assert "max_horizon_k=64" in log
    assert "update_steps=4" in log
    assert "reset_mode=auto" in log
    assert "live_runner=True" in log
    assert "probe_only=True" in log
    assert "storage_write=True" in log
    assert "ppo_update=True" in log


def test_stage3_boundary_allows_live_train_enabled() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=True, train=True))
    boundary.assert_live_runner_ready()
    log = boundary.train_log()
    assert log is not None
    assert "FrontRES Segment Live Train Ready" in log
    assert "objective=segment_replay_hrl" in log
    assert "segment_k=8" in log
    assert "max_horizon_k=64" in log
    assert "update_steps=4" in log
    assert "reset_mode=auto" in log
    assert "live_runner=True" in log
    assert "runner_learn=True" in log
    assert "storage=independent" in log
    assert "ppo_action=delta_se3_6d" in log


def test_on_policy_runner_calls_stage3_boundary() -> None:
    runner_text = (ROOT / "rsl_rl" / "runners" / "on_policy_runner.py").read_text()
    assert "FrontRESSegmentRunnerBoundary" in runner_text
    assert "from_train_cfg(self.cfg)" in runner_text
    assert "assert_live_runner_ready()" in runner_text
    assert "sentinel_log()" in runner_text
    assert "probe_log()" in runner_text
    assert "train_log()" in runner_text
    assert "run_frontres_segment_live_probe" in runner_text
    assert "_run_frontres_segment_single_update" in runner_text
    assert "run_frontres_segment_live_update_loop" in runner_text
    assert "run_frontres_segment_sequence_offline_eval" not in runner_text
    assert "learn_frontres_segment_live" in runner_text


def test_startup_lifecycle_records_fresh_resume_and_evaluation_exactly_once() -> None:
    recorded: list[str] = []
    calls: list[str] = []
    fresh = FrontRESStartupLifecycle(event_sink=recorded.append)
    fresh.resolve_layout(lambda: calls.append("layout") or "928/158/770")
    fresh.load("hsl", lambda: calls.append("hsl") or "loaded")
    result = fresh.dispatch_once("formal_train", lambda: calls.append("dispatch") or "ok")
    assert result == "ok"
    assert fresh.events == ("config", "layout", "hsl", "dispatch:formal_train")
    assert recorded == list(fresh.events)
    assert calls == ["layout", "hsl", "dispatch"]
    try:
        fresh.dispatch_once("formal_train", lambda: calls.append("duplicate"))
    except RuntimeError as exc:
        assert "already been dispatched" in str(exc)
    else:
        raise AssertionError("one launch request must not dispatch twice")
    assert "duplicate" not in calls

    resume = FrontRESStartupLifecycle()
    resume.resolve_layout(lambda: None)
    resume.load("resume", lambda: None)
    resume.dispatch_once("formal_train", lambda: None)
    assert resume.events == ("config", "layout", "resume", "dispatch:formal_train")
    try:
        resume.load("hsl", lambda: calls.append("mixed-hsl"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("resume must reject a later HSL initializer")
    assert "mixed-hsl" not in calls

    optimizer_updates = 0
    evaluation = FrontRESStartupLifecycle()
    evaluation.resolve_layout(lambda: None)
    assert evaluation.dispatch_once("policy_quality", lambda: "atomic-report") == "atomic-report"
    assert optimizer_updates == 0


def test_startup_layout_failure_stops_load_and_dispatch() -> None:
    lifecycle = FrontRESStartupLifecycle()
    downstream: list[str] = []

    def fail_layout():
        raise ValueError("bad layout")

    try:
        lifecycle.resolve_layout(fail_layout)
    except ValueError as exc:
        assert str(exc) == "bad layout"
    else:
        raise AssertionError("layout failure must propagate")
    for operation in (
        lambda: lifecycle.load("hsl", lambda: downstream.append("load")),
        lambda: lifecycle.dispatch_once("formal_train", lambda: downstream.append("dispatch")),
    ):
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed startup must not continue")
    assert downstream == []
    assert lifecycle.events == ("config", "layout:failed")


def main() -> None:
    test_stage3_boundary_rejects_live_runner_by_default()
    test_stage3_boundary_rejects_live_flag_until_ppo_wiring_exists()
    test_stage3_boundary_allows_live_sentinel_only()
    test_stage3_boundary_allows_live_probe_only()
    test_stage3_boundary_allows_live_storage_write_only()
    test_stage3_boundary_allows_live_single_update_only()
    test_stage3_boundary_allows_live_update_loop_only()
    test_stage3_boundary_allows_live_train_enabled()
    test_on_policy_runner_calls_stage3_boundary()
    test_startup_lifecycle_records_fresh_resume_and_evaluation_exactly_once()
    test_startup_layout_failure_stops_load_and_dispatch()
    print("result: PASS")


if __name__ == "__main__":
    main()
