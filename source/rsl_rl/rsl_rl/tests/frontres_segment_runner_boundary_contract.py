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
connector_module = _load(
    "frontres_segment_replay",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_replay.py",
)

FrontRESSegmentRunnerBoundary = boundary_module.FrontRESSegmentRunnerBoundary
FrontRESSegmentReplayConnector = connector_module.FrontRESSegmentReplayConnector


class FakeConnectorDep:
    pass


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
            "frontres_segment_k": 4,
            "frontres_segment_reset_mode": "auto",
        }
    }


def test_stage3_boundary_rejects_live_runner_by_default() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=False))
    assert boundary.requested
    assert boundary.segment_k == 4
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
    assert "segment_k=4" in log
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
    assert "segment_k=4" in log
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
    assert "segment_k=4" in log
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
    assert "segment_k=4" in log
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
    assert "segment_k=4" in log
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
    assert "segment_k=4" in log
    assert "update_steps=4" in log
    assert "reset_mode=auto" in log
    assert "live_runner=True" in log
    assert "runner_learn=True" in log
    assert "storage=independent" in log
    assert "ppo_action=delta_se3_6d" in log


def test_stage3_boundary_builds_fake_connector() -> None:
    boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(_stage3_cfg(live=False))
    connector = boundary.build_fake_connector(
        dataset=FakeConnectorDep(),
        sampler=FakeConnectorDep(),
        reset_adapter=FakeConnectorDep(),
        action_projector=FakeConnectorDep(),
        reward=FakeConnectorDep(),
        rollout_fn=lambda **kwargs: kwargs,
        connector_cls=FrontRESSegmentReplayConnector,
    )
    assert connector.stage == "stage3_segment_hrl"
    assert connector.objective == "segment_replay_hrl"
    assert connector.reset_mode == "auto"


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
    assert "learn_frontres_segment_live" in runner_text


def main() -> None:
    test_stage3_boundary_rejects_live_runner_by_default()
    test_stage3_boundary_rejects_live_flag_until_ppo_wiring_exists()
    test_stage3_boundary_allows_live_sentinel_only()
    test_stage3_boundary_allows_live_probe_only()
    test_stage3_boundary_allows_live_storage_write_only()
    test_stage3_boundary_allows_live_single_update_only()
    test_stage3_boundary_allows_live_update_loop_only()
    test_stage3_boundary_allows_live_train_enabled()
    test_stage3_boundary_builds_fake_connector()
    test_on_policy_runner_calls_stage3_boundary()
    print("result: PASS")


if __name__ == "__main__":
    main()
