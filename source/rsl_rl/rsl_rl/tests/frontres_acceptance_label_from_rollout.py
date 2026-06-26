#!/usr/bin/env python3
"""TEST ONLY: acceptance labels come from Noisy-vs-Candidate rollout evidence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_acceptance_labels.py"


def _load_acceptance_label_module():
    spec = importlib.util.spec_from_file_location("frontres_acceptance_labels_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_label_module = _load_acceptance_label_module()
build_frontres_acceptance_labels = _label_module.build_frontres_acceptance_labels
expand_acceptance_labels_to_task_dims = _label_module.expand_acceptance_labels_to_task_dims


def _assert_close(name: str, got: torch.Tensor, expected: list[float]) -> None:
    exp = torch.tensor(expected, device=got.device, dtype=got.dtype).view_as(got)
    if not torch.allclose(got, exp, atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError(f"{name}: got {got.flatten().tolist()}, expected {exp.flatten().tolist()}")


def run_acceptance_label_check() -> None:
    names = ["good", "harmful", "margin", "large_good", "large_bad"]
    noisy = torch.tensor([0.20, 0.20, 0.20, -0.10, -0.10])
    candidate = torch.tensor([0.30, 0.05, 0.203, 0.40, -0.35])
    labels = build_frontres_acceptance_labels(
        candidate_score=candidate,
        noisy_score=noisy,
        positive_margin=0.01,
        negative_margin=0.01,
    )

    _assert_close("accept_gt", labels.accept_gt, [1.0, 0.0, 0.0, 1.0, 0.0])
    _assert_close("accept_mask", labels.accept_mask, [1.0, 1.0, 0.0, 1.0, 1.0])
    _assert_close("margin", labels.margin, [0.10, -0.15, 0.003, 0.50, -0.25])

    target, mask = expand_acceptance_labels_to_task_dims(labels, task_dim=6)
    if tuple(target.shape) != (5, 6) or tuple(mask.shape) != (5, 6):
        raise AssertionError("expanded task-dim label shape mismatch")
    for row, name in enumerate(names):
        expected_target = labels.accept_gt[row].expand(6)
        expected_mask = labels.accept_mask[row].expand(6)
        if not torch.allclose(target[row], expected_target):
            raise AssertionError(f"{name}: expanded target mismatch")
        if not torch.allclose(mask[row], expected_mask):
            raise AssertionError(f"{name}: expanded mask mismatch")

    dim_mask = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)
    _, masked = expand_acceptance_labels_to_task_dims(labels, task_dim=6, dim_mask=dim_mask)
    if not torch.allclose(masked[0], dim_mask):
        raise AssertionError("active dim mask did not apply to accepted sample")
    if float(masked[2].sum().item()) != 0.0:
        raise AssertionError("margin sample should remain inactive after dim masking")

    print("case        noisy candidate margin accept_gt mask")
    for i, name in enumerate(names):
        print(
            f"{name:10s} {float(noisy[i]):+.3f} {float(candidate[i]):+.3f} "
            f"{float(labels.margin[i]):+.3f} {float(labels.accept_gt[i]):.0f} {float(labels.accept_mask[i]):.0f}"
        )
    print("FrontRES acceptance label from rollout: PASS")


if __name__ == "__main__":
    run_acceptance_label_check()
