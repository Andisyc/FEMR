"""TEST ONLY: FrontRES authority event mask semantics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1] / "frontres"

EVENT_SPEC = importlib.util.spec_from_file_location("frontres_authority_event_test_module", ROOT / "frontres_authority_event.py")
RETURN_SPEC = importlib.util.spec_from_file_location(
    "frontres_authority_return_test_module", ROOT / "frontres_authority_return.py"
)
if EVENT_SPEC is None or EVENT_SPEC.loader is None or RETURN_SPEC is None or RETURN_SPEC.loader is None:
    raise RuntimeError("Could not load FrontRES authority event/return modules.")
event_module = importlib.util.module_from_spec(EVENT_SPEC)
return_module = importlib.util.module_from_spec(RETURN_SPEC)
sys.modules[EVENT_SPEC.name] = event_module
sys.modules[RETURN_SPEC.name] = return_module
EVENT_SPEC.loader.exec_module(event_module)
RETURN_SPEC.loader.exec_module(return_module)

build_frontres_authority_events = event_module.build_frontres_authority_events
compute_frontres_authority_k_step_return = return_module.compute_frontres_authority_k_step_return


def test_burst_trains_query_frame_but_returns_cover_window() -> None:
    active = torch.ones(6, 1, dtype=torch.bool)
    events = build_frontres_authority_events(active, mode="burst", burst_length=6)
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    dones = torch.zeros_like(rewards, dtype=torch.bool)

    result = compute_frontres_authority_k_step_return(
        rewards,
        dones,
        events.event_start,
        horizon=6,
        gamma=1.0,
    )

    expected_start = torch.tensor([[True], [False], [False], [False], [False], [False]])
    expected_return = torch.tensor([[21.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    if not torch.equal(events.event_start, expected_start):
        raise AssertionError(f"event_start mismatch:\n{events.event_start}\n!=\n{expected_start}")
    torch.testing.assert_close(result.returns, expected_return)
    if int(result.valid_mask.sum().item()) != 1:
        raise AssertionError("burst authority should train exactly one query frame for one held event.")
    if int(result.steps[0, 0].item()) != 6:
        raise AssertionError("query-frame authority return should cover the full burst window.")


def main() -> None:
    test_burst_trains_query_frame_but_returns_cover_window()
    print("=== FrontRES Authority Event Mask Semantics TEST ONLY ===")
    print("checks=burst has one authority query while K-step return covers event window")
    print("result: PASS")


if __name__ == "__main__":
    main()

