"""TEST ONLY: FrontRES authority target conflict arbitration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_authority_targets.py"
SPEC = importlib.util.spec_from_file_location("frontres_authority_targets_test_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES authority targets module from {MODULE_PATH}.")
authority_targets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = authority_targets
SPEC.loader.exec_module(authority_targets)

resolve_frontres_authority_targets = authority_targets.resolve_frontres_authority_targets


def test_harmful_high_rho_overrides_optimistic_endpoint() -> None:
    behavior = torch.tensor([[-2.0], [1.0], [-0.1]])
    zero = torch.tensor([[0.0], [0.0], [0.0]])
    one = torch.tensor([[4.0], [3.0], [2.0]])
    rho = torch.tensor(
        [
            [0.95, 0.90, 1.00, 0.95, 0.90, 1.00],
            [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
            [0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
        ]
    )
    mask = torch.ones(3, 1)

    resolved = resolve_frontres_authority_targets(
        behavior_return=behavior,
        zero_return=zero,
        one_return=one,
        behavior_rho=rho,
        mask=mask,
    )

    torch.testing.assert_close(resolved.one_return[0], behavior[0])
    torch.testing.assert_close(resolved.one_return[1], one[1])
    torch.testing.assert_close(resolved.one_return[2], one[2])
    torch.testing.assert_close(resolved.harmful_full_write_mask, torch.tensor([[1.0], [0.0], [0.0]]))
    torch.testing.assert_close(resolved.conflict_mask, torch.tensor([[1.0], [0.0], [0.0]]))


def main() -> None:
    test_harmful_high_rho_overrides_optimistic_endpoint()
    print("=== FrontRES Authority Target Conflict TEST ONLY ===")
    print("checks=near-full harmful behavior overrides optimistic rho=1 endpoint target")
    print("result: PASS")


if __name__ == "__main__":
    main()

