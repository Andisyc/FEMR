"""TEST ONLY: FrontRES authority endpoint target construction.

This test checks the scalar target semantics used by the authority critic:

rho=0        -> Noisy-relative return, which is zero.
rho=1        -> Candidate/full-write gain over Noisy.
behavior rho -> executed FrontRES gain over Noisy.

It is intentionally pure and does not start IsaacLab.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODULE_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_authority_return.py"
SPEC = importlib.util.spec_from_file_location("frontres_authority_return_test_module", MODULE_PATH)
authority_return = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(authority_return)
compute_frontres_authority_k_step_return = authority_return.compute_frontres_authority_k_step_return


def test_endpoint_targets_are_noisy_relative_k_step_returns() -> None:
    noisy = torch.tensor(
        [
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
            [2.5, 3.5],
        ]
    )
    projected = torch.tensor(
        [
            [1.2, 1.8],
            [1.8, 2.1],
            [2.4, 2.7],
            [2.9, 3.2],
        ]
    )
    candidate = torch.tensor(
        [
            [1.6, 2.4],
            [2.0, 2.9],
            [2.7, 3.5],
            [3.0, 3.9],
        ]
    )
    dones = torch.zeros_like(noisy, dtype=torch.bool)
    event_start = torch.zeros_like(noisy, dtype=torch.bool)
    event_start[0, 0] = True
    event_start[1, 1] = True

    behavior_delta = projected - noisy
    zero_delta = torch.zeros_like(noisy)
    one_delta = candidate - noisy

    behavior_return = compute_frontres_authority_k_step_return(
        behavior_delta,
        dones,
        event_start,
        horizon=3,
        gamma=0.9,
    )
    zero_return = compute_frontres_authority_k_step_return(
        zero_delta,
        dones,
        event_start,
        horizon=3,
        gamma=0.9,
    )
    one_return = compute_frontres_authority_k_step_return(
        one_delta,
        dones,
        event_start,
        horizon=3,
        gamma=0.9,
    )

    expected_behavior_env0 = 0.2 + 0.9 * 0.3 + 0.9 * 0.9 * 0.4
    expected_behavior_env1 = -0.4 + 0.9 * -0.3 + 0.9 * 0.9 * -0.3
    expected_one_env0 = 0.6 + 0.9 * 0.5 + 0.9 * 0.9 * 0.7
    expected_one_env1 = 0.4 + 0.9 * 0.5 + 0.9 * 0.9 * 0.4

    torch.testing.assert_close(behavior_return.returns[0, 0], torch.tensor(expected_behavior_env0))
    torch.testing.assert_close(behavior_return.returns[1, 1], torch.tensor(expected_behavior_env1))
    torch.testing.assert_close(zero_return.returns, torch.zeros_like(zero_return.returns))
    torch.testing.assert_close(one_return.returns[0, 0], torch.tensor(expected_one_env0))
    torch.testing.assert_close(one_return.returns[1, 1], torch.tensor(expected_one_env1))
    torch.testing.assert_close(behavior_return.valid_mask, event_start)
    torch.testing.assert_close(one_return.valid_mask, event_start)


def main() -> None:
    test_endpoint_targets_are_noisy_relative_k_step_returns()
    print("=== FrontRES Authority Endpoint Targets TEST ONLY ===")
    print("checks=rho0 zero baseline, rho1 Candidate-Noisy gain, behavior Projected-Noisy gain, K-step event masks")
    print("result: PASS")


if __name__ == "__main__":
    main()
