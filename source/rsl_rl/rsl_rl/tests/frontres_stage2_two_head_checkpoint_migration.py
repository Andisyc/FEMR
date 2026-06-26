#!/usr/bin/env python3
"""Regression: Stage 2 can load a Stage 1 two-head proposal checkpoint."""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHECKPOINTING_PATH = Path(__file__).resolve().parents[1] / "runners" / "frontres_checkpointing.py"
spec = importlib.util.spec_from_file_location("frontres_checkpointing_under_test", CHECKPOINTING_PATH)
checkpointing = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(checkpointing)
_load_split_proposal_from_two_head_residual = checkpointing._load_split_proposal_from_two_head_residual


def _proposal_actor() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(10, 8),
        nn.ELU(),
        nn.Linear(8, 6),
    )


def _two_head_state() -> dict[str, torch.Tensor]:
    return {
        "trunk.0.weight": torch.randn(8, 10),
        "trunk.0.bias": torch.randn(8),
        "proposal_head.weight": torch.randn(6, 8),
        "proposal_head.bias": torch.randn(6),
        "acceptance_head.weight": torch.randn(6, 8),
        "acceptance_head.bias": torch.randn(6),
    }


def main() -> None:
    actor = _proposal_actor()
    state = _two_head_state()
    migrated = _load_split_proposal_from_two_head_residual(actor, state)
    if not migrated:
        raise AssertionError("Stage 1 two-head residual_actor did not migrate into Stage 2 proposal actor.")

    loaded = actor.state_dict()
    torch.testing.assert_close(loaded["0.weight"], state["trunk.0.weight"])
    torch.testing.assert_close(loaded["0.bias"], state["trunk.0.bias"])
    torch.testing.assert_close(loaded["2.weight"], state["proposal_head.weight"])
    torch.testing.assert_close(loaded["2.bias"], state["proposal_head.bias"])
    print("PASS: Stage 2 maps Stage 1 two-head residual_actor into proposal-only actor.")


if __name__ == "__main__":
    main()
