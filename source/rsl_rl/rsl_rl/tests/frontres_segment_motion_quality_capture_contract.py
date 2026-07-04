#!/usr/bin/env python3
"""Contract tests for Segment Replay motion-quality capture."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_live_probe_module():
    class _Dummy:
        pass

    sys.modules.setdefault("rsl_rl", types.ModuleType("rsl_rl"))
    _stub_module("rsl_rl.algorithms", FrontRESUnified=_Dummy)
    _stub_module(
        "rsl_rl.algorithms.frontres_segment_ppo",
        FrontRESSegmentPPOBatch=_Dummy,
        FrontRESSegmentPPOConfig=_Dummy,
        compute_frontres_segment_ppo_loss=lambda *args, **kwargs: None,
    )
    _stub_module(
        "rsl_rl.frontres.frontres_segment_storage",
        FrontRESSegmentRolloutStorage=_Dummy,
        FrontRESSegmentTransition=_Dummy,
    )
    _stub_module(
        "rsl_rl.frontres.frontres_segment_reset",
        FrontRESSegmentResetAdapter=_Dummy,
        FrontRESSegmentResetResult=_Dummy,
        ensure_frontres_segment_live_reset_hook=lambda *args, **kwargs: None,
    )
    _stub_module("rsl_rl.frontres.training_schedule", resolve_frontres_mode_state=lambda *args, **kwargs: None)
    _stub_module("rsl_rl.modules", FrontRESActorCritic=_Dummy)
    _stub_module("rsl_rl.runners.frontres_training_setup", configure_frontres_pair_layout=lambda *args, **kwargs: None)
    _stub_module("rsl_rl.runners.frontres_rollout_step", prepare_frontres_rollout_step=lambda *args, **kwargs: None)

    spec = importlib.util.spec_from_file_location(
        "frontres_segment_live_probe_contract_target",
        ROOT / "rsl_rl" / "runners" / "frontres_segment_live_probe.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _PairLayout:
    n_train = 1
    n_candidate = 1
    n_base = 1
    n_clean = 1


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name: str):
        assert name == "motion"
        return self.command


class _Runner:
    def __init__(self, command):
        env = types.SimpleNamespace(command_manager=_CommandManager(command))
        self.env = types.SimpleNamespace(unwrapped=env)


def test_motion_quality_capture_removes_role_origin_offsets() -> None:
    module = _load_live_probe_module()
    local_body = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.1, 0.0],
            [-0.2, 0.4, 0.2],
        ]
    )
    origins = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
        ]
    )
    world_body = local_body.unsqueeze(0) + origins.unsqueeze(1)
    command = types.SimpleNamespace(
        body_pos_w=world_body,
        body_pos_relative_w=world_body + 1000.0,
        robot_body_pos_w=world_body.clone(),
    )

    clean, repaired, noisy = module._capture_motion_quality_frame(_Runner(command), _PairLayout())

    print(
        "[probe motion_quality_capture] "
        f"clean_max={float(clean.abs().max())} "
        f"repaired_mpjpe={float(torch.linalg.norm(repaired - clean, dim=-1).mean())} "
        f"noisy_mpjpe={float(torch.linalg.norm(noisy - clean, dim=-1).mean())}",
        flush=True,
    )
    assert torch.allclose(clean, repaired)
    assert torch.allclose(clean, noisy)


def main() -> None:
    test_motion_quality_capture_removes_role_origin_offsets()
    print("result: PASS")


if __name__ == "__main__":
    main()
