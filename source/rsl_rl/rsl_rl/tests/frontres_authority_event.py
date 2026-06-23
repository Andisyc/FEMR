"""FrontRES authority event scheduler toy checks.

TEST ONLY: this file does not touch the live training path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_authority_event.py"
SPEC = importlib.util.spec_from_file_location("frontres_authority_event_test_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES authority event module from {MODULE_PATH}.")
authority_event = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = authority_event
SPEC.loader.exec_module(authority_event)

build_frontres_authority_events = authority_event.build_frontres_authority_events


def _assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        raise AssertionError(f"{name} mismatch:\nactual={actual}\nexpected={expected}")


def test_single_mode_every_active_frame_queries() -> None:
    active = torch.tensor(
        [
            [False, True],
            [True, True],
            [False, False],
            [True, False],
        ]
    )
    events = build_frontres_authority_events(active, mode="single")

    expected_id = torch.tensor(
        [
            [-1, 0],
            [0, 1],
            [-1, -1],
            [1, -1],
        ]
    )
    expected_step = torch.zeros_like(expected_id)

    _assert_equal("single start", events.event_start, active)
    _assert_equal("single active", events.event_active, active)
    _assert_equal("single id", events.event_id, expected_id)
    _assert_equal("single step", events.event_step, expected_step)
    _assert_equal("single query", events.query_mask, events.event_start)


def test_burst_mode_splits_long_active_segments() -> None:
    active = torch.tensor(
        [
            [True, False],
            [True, True],
            [True, True],
            [True, False],
            [True, True],
        ]
    )
    events = build_frontres_authority_events(active, mode="burst", burst_length=3)

    expected_start = torch.tensor(
        [
            [True, False],
            [False, True],
            [False, False],
            [True, False],
            [False, True],
        ]
    )
    expected_id = torch.tensor(
        [
            [0, -1],
            [0, 0],
            [0, 0],
            [1, -1],
            [1, 1],
        ]
    )
    expected_step = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [2, 1],
            [0, 0],
            [1, 0],
        ]
    )

    _assert_equal("burst start", events.event_start, expected_start)
    _assert_equal("burst active", events.event_active, active)
    _assert_equal("burst id", events.event_id, expected_id)
    _assert_equal("burst step", events.event_step, expected_step)


def test_persistent_mode_holds_one_event_per_active_segment() -> None:
    active = torch.tensor([[True], [True], [True], [False], [True], [True]])
    events = build_frontres_authority_events(active, mode="persistent")

    expected_start = torch.tensor([[True], [False], [False], [False], [True], [False]])
    expected_id = torch.tensor([[0], [0], [0], [-1], [1], [1]])
    expected_step = torch.tensor([[0], [1], [2], [0], [0], [1]])

    _assert_equal("persistent start", events.event_start, expected_start)
    _assert_equal("persistent id", events.event_id, expected_id)
    _assert_equal("persistent step", events.event_step, expected_step)


def test_persistent_refresh_interval_requeries_inside_segment() -> None:
    active = torch.tensor([[True], [True], [True], [True], [True]])
    events = build_frontres_authority_events(active, mode="persistent", refresh_interval=2)

    expected_start = torch.tensor([[True], [False], [True], [False], [True]])
    expected_id = torch.tensor([[0], [0], [1], [1], [2]])
    expected_step = torch.tensor([[0], [1], [0], [1], [0]])

    _assert_equal("persistent refresh start", events.event_start, expected_start)
    _assert_equal("persistent refresh id", events.event_id, expected_id)
    _assert_equal("persistent refresh step", events.event_step, expected_step)


def test_inactive_frames_never_own_events() -> None:
    active = torch.zeros(3, 2, dtype=torch.bool)
    events = build_frontres_authority_events(active, mode="burst", burst_length=2)

    _assert_equal("inactive start", events.event_start, active)
    _assert_equal("inactive active", events.event_active, active)
    _assert_equal("inactive ids", events.event_id, torch.full((3, 2), -1, dtype=torch.long))
    _assert_equal("inactive steps", events.event_step, torch.zeros(3, 2, dtype=torch.long))


def main() -> None:
    test_single_mode_every_active_frame_queries()
    test_burst_mode_splits_long_active_segments()
    test_persistent_mode_holds_one_event_per_active_segment()
    test_persistent_refresh_interval_requeries_inside_segment()
    test_inactive_frames_never_own_events()
    print("=== FrontRES Authority Event Scheduler TEST ONLY ===")
    print("checks=single, burst, persistent, persistent_refresh, inactive")
    print("result: PASS")


if __name__ == "__main__":
    main()
