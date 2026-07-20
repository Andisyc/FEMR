#!/usr/bin/env python3
"""Regression contract: v007 rejects the retired Stage-3 HSL rollout label."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "frontres_hsl_rollout_target.py"


def _load_owner():
    spec = importlib.util.spec_from_file_location("frontres_hsl_rollout_target_contract_owner", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v007_rejects_legacy_rollout_label_before_transition_write() -> None:
    owner = _load_owner()
    transition = SimpleNamespace(sentinel="unchanged")
    runner = SimpleNamespace(alg=SimpleNamespace(transition=transition))
    try:
        owner.build_frontres_hsl_rollout_target(
            runner,
            command=None,
            actions=None,
            dones=None,
            current_pos_correction=None,
            current_quat_correction=None,
            n_train=0,
            n_candidate=0,
            n_base=0,
            n_clean=0,
            quat_to_rotvec_wxyz=None,
        )
    except RuntimeError as exc:
        assert "FRS-TRAIN-v007" in str(exc)
    else:
        raise AssertionError("legacy Stage-3 HSL rollout label was not rejected")
    assert vars(transition) == {"sentinel": "unchanged"}


if __name__ == "__main__":
    test_v007_rejects_legacy_rollout_label_before_transition_write()
    print("frontres_hsl_rollout_target_contract: v007 reject ok")
