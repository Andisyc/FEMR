"""FrontRES continuous authority space toy checks.

TEST ONLY: this file does not touch the live training path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_authority_space.py"
SPEC = importlib.util.spec_from_file_location("frontres_authority_space_test_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES authority space module from {MODULE_PATH}.")
authority_space = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = authority_space
SPEC.loader.exec_module(authority_space)

FRONTRES_AUTHORITY_DIM_NAMES = authority_space.FRONTRES_AUTHORITY_DIM_NAMES
apply_authority_active_mask = authority_space.apply_authority_active_mask
apply_authority_to_delta_se = authority_space.apply_authority_to_delta_se
authority_rho_stats = authority_space.authority_rho_stats
bound_authority_rho = authority_space.bound_authority_rho
raw_authority_to_rho = authority_space.raw_authority_to_rho


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=1e-6, rtol=0.0):
        raise AssertionError(f"{name} mismatch:\nactual={actual}\nexpected={expected}")


def test_bound_and_active_mask() -> None:
    raw = torch.tensor(
        [
            [-10.0, 0.0, 10.0, 1.0, -1.0, 2.0],
            [0.5, -0.5, 0.0, 3.0, -3.0, 0.25],
        ]
    )
    active_task_dims = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    rho = raw_authority_to_rho(raw, active_task_dims)

    if rho.shape != (2, 6):
        raise AssertionError(f"rho shape mismatch: {tuple(rho.shape)}")
    if float(rho.min().item()) < 0.0 or float(rho.max().item()) > 1.0:
        raise AssertionError(f"rho out of [0, 1]: min={rho.min().item()}, max={rho.max().item()}")
    _assert_close("masked dy", rho[:, 1], torch.zeros(2))
    _assert_close("masked dyaw", rho[:, 5], torch.zeros(2))


def test_gradient_through_bounded_rho() -> None:
    raw = torch.zeros(1, 6, requires_grad=True)
    active_task_dims = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    rho = raw_authority_to_rho(raw, active_task_dims)
    loss = rho.sum()
    loss.backward()

    expected_grad = torch.tensor([[0.25, 0.0, 0.25, 0.25, 0.25, 0.0]])
    _assert_close("masked sigmoid gradient", raw.grad, expected_grad)


def test_apply_authority_to_delta_se() -> None:
    proposal = torch.tensor([[1.0, -2.0, 3.0, 0.4, -0.5, 0.6]])
    rho = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    active_task_dims = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    expected = torch.tensor([[0.2, 0.0, 1.2, 0.2, -0.3, 0.0]])
    _assert_close("executed Delta SE", apply_authority_to_delta_se(proposal, rho, active_task_dims), expected)


def test_rho_stats() -> None:
    rho = torch.tensor(
        [
            [0.00, 0.10, 0.20, 0.30, 0.95, 1.00],
            [0.10, 0.20, 0.30, 0.40, 0.90, 0.95],
            [0.20, 0.30, 0.40, 0.50, 0.85, 0.90],
        ]
    )
    stats = authority_rho_stats(rho)
    _assert_close("rho mean", stats.mean, torch.tensor([0.10, 0.20, 0.30, 0.40, 0.90, 0.95]))
    _assert_close("near zero frac", stats.near_zero_frac, torch.tensor([1.0 / 3.0, 0, 0, 0, 0, 0]))
    _assert_close("near one frac", stats.near_one_frac, torch.tensor([0, 0, 0, 0, 1.0 / 3.0, 2.0 / 3.0]))

    masked = authority_rho_stats(rho, sample_mask=torch.tensor([True, False, False]))
    _assert_close("masked mean", masked.mean, rho[0])


def main() -> None:
    test_bound_and_active_mask()
    test_gradient_through_bounded_rho()
    test_apply_authority_to_delta_se()
    test_rho_stats()
    print("=== FrontRES Continuous Authority Space TEST ONLY ===")
    print(f"dims={FRONTRES_AUTHORITY_DIM_NAMES}")
    print("checks=bound, active_mask, gradient, execution, diagnostics")
    print("result: PASS")


if __name__ == "__main__":
    main()
