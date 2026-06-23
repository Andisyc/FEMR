"""FrontRES K-step authority return toy checks.

TEST ONLY: this file does not touch the live training path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_authority_return.py"
SPEC = importlib.util.spec_from_file_location("frontres_authority_return_test_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES authority return module from {MODULE_PATH}.")
authority_return = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = authority_return
SPEC.loader.exec_module(authority_return)

compute_frontres_authority_k_step_return = authority_return.compute_frontres_authority_k_step_return


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=1e-5, rtol=0.0):
        raise AssertionError(f"{name} mismatch:\nactual={actual}\nexpected={expected}")


def test_plain_k_step_and_event_mask() -> None:
    rewards = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    dones = torch.zeros_like(rewards, dtype=torch.bool)
    event_mask = torch.tensor(
        [
            [True, False],
            [True, True],
            [False, False],
            [False, False],
        ]
    )
    result = compute_frontres_authority_k_step_return(
        rewards,
        dones,
        event_mask,
        horizon=3,
        gamma=0.9,
    )

    expected_returns = torch.tensor(
        [
            [1.0 + 0.9 * 2.0 + 0.81 * 3.0, 0.0],
            [2.0 + 0.9 * 3.0 + 0.81 * 4.0, 20.0 + 0.9 * 30.0 + 0.81 * 40.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    expected_steps = torch.tensor(
        [
            [3, 0],
            [3, 3],
            [0, 0],
            [0, 0],
        ]
    )
    _assert_close("plain returns", result.returns, expected_returns)
    if not torch.equal(result.steps, expected_steps):
        raise AssertionError(f"plain steps mismatch:\nactual={result.steps}\nexpected={expected_steps}")
    if not torch.equal(result.valid_mask, event_mask):
        raise AssertionError("event valid mask mismatch.")


def test_done_truncates_after_including_done_reward() -> None:
    rewards = torch.tensor([[1.0], [2.0], [100.0], [1000.0]])
    dones = torch.tensor([[False], [True], [False], [False]])
    event_mask = torch.tensor([[True], [False], [False], [False]])
    result = compute_frontres_authority_k_step_return(
        rewards,
        dones,
        event_mask,
        horizon=4,
        gamma=0.5,
    )

    expected_returns = torch.tensor([[1.0 + 0.5 * 2.0], [0.0], [0.0], [0.0]])
    expected_steps = torch.tensor([[2], [0], [0], [0]])
    _assert_close("done truncated return", result.returns, expected_returns)
    if not torch.equal(result.steps, expected_steps):
        raise AssertionError(f"done steps mismatch:\nactual={result.steps}\nexpected={expected_steps}")


def test_bootstrap_when_horizon_reaches_valid_state() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    dones = torch.zeros_like(rewards, dtype=torch.bool)
    event_mask = torch.tensor([[True], [True], [False]])
    bootstrap = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    result = compute_frontres_authority_k_step_return(
        rewards,
        dones,
        event_mask,
        horizon=2,
        gamma=0.9,
        bootstrap_values=bootstrap,
    )

    expected_returns = torch.tensor(
        [
            [1.0 + 0.9 * 2.0 + 0.81 * 30.0],
            [2.0 + 0.9 * 3.0 + 0.81 * 40.0],
            [0.0],
        ]
    )
    expected_bootstrap_mask = torch.tensor([[True], [True], [False]])
    _assert_close("bootstrap return", result.returns, expected_returns)
    if not torch.equal(result.bootstrap_mask, expected_bootstrap_mask):
        raise AssertionError(
            f"bootstrap mask mismatch:\nactual={result.bootstrap_mask}\nexpected={expected_bootstrap_mask}"
        )


def test_done_blocks_bootstrap() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    dones = torch.tensor([[False], [True], [False]])
    event_mask = torch.tensor([[True], [False], [False]])
    bootstrap = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    result = compute_frontres_authority_k_step_return(
        rewards,
        dones,
        event_mask,
        horizon=2,
        gamma=0.9,
        bootstrap_values=bootstrap,
    )

    expected_returns = torch.tensor([[1.0 + 0.9 * 2.0], [0.0], [0.0]])
    _assert_close("done blocks bootstrap", result.returns, expected_returns)
    if bool(result.bootstrap_mask[0, 0].item()):
        raise AssertionError("bootstrap should be blocked after done.")


def test_bootstrap_is_detached() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    dones = torch.zeros_like(rewards, dtype=torch.bool)
    event_mask = torch.tensor([[True], [False], [False]])
    bootstrap = torch.tensor([[10.0], [20.0], [30.0], [40.0]], requires_grad=True)
    result = compute_frontres_authority_k_step_return(
        rewards,
        dones,
        event_mask,
        horizon=2,
        gamma=0.9,
        bootstrap_values=bootstrap,
    )
    result.returns.sum().backward()

    if rewards.grad is None or float(rewards.grad.abs().sum().item()) <= 0.0:
        raise AssertionError("reward gradients should exist in this detach test.")
    if bootstrap.grad is not None:
        raise AssertionError("bootstrap_values must be detached from the target.")


def main() -> None:
    test_plain_k_step_and_event_mask()
    test_done_truncates_after_including_done_reward()
    test_bootstrap_when_horizon_reaches_valid_state()
    test_done_blocks_bootstrap()
    test_bootstrap_is_detached()
    print("=== FrontRES Authority K-Step Return TEST ONLY ===")
    print("checks=event_mask, done_truncation, bootstrap, bootstrap_detach")
    print("result: PASS")


if __name__ == "__main__":
    main()
