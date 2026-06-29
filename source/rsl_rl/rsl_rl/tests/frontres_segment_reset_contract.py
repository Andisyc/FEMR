from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_dataset.py"
dataset_spec = importlib.util.spec_from_file_location("frontres_segment_dataset", DATASET_PATH)
dataset_module = importlib.util.module_from_spec(dataset_spec)
assert dataset_spec.loader is not None
sys.modules[dataset_spec.name] = dataset_module
dataset_spec.loader.exec_module(dataset_module)
FrontRESSegmentBatch = dataset_module.FrontRESSegmentBatch
FrontRESSegmentSpec = dataset_module.FrontRESSegmentSpec
FrontRESSegmentState = dataset_module.FrontRESSegmentState

RESET_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_reset.py"
reset_spec = importlib.util.spec_from_file_location("frontres_segment_reset", RESET_PATH)
reset_module = importlib.util.module_from_spec(reset_spec)
assert reset_spec.loader is not None
sys.modules[reset_spec.name] = reset_module
reset_spec.loader.exec_module(reset_module)
FrontRESSegmentResetAdapter = reset_module.FrontRESSegmentResetAdapter


class FakeEnv:
    def __init__(self) -> None:
        self.last_request = None

    def apply_frontres_segment_reset(self, request):
        self.last_request = request
        return {
            "reset_success": torch.tensor([True, False]),
            "fall_at_reset": torch.tensor([False, True]),
            "velocity_mismatch": torch.tensor([0.0, 0.0]),
        }


def _batch(static: bool = False) -> FrontRESSegmentBatch:
    vel = torch.zeros(2, 3) if static else torch.ones(2, 3)
    dof_vel = torch.zeros(2, 4) if static else torch.ones(2, 4)
    state = FrontRESSegmentState(
        root_pos=torch.zeros(2, 3),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        root_lin_vel=vel,
        root_ang_vel=vel,
        dof_pos=torch.zeros(2, 4),
        dof_vel=dof_vel,
    )
    return FrontRESSegmentBatch(
        segment_ids=torch.tensor([0, 1]),
        specs=(
            FrontRESSegmentSpec(segment_id=0, motion_id=0, start_frame=0, phase=0.2, reset_mode_hint="direct"),
            FrontRESSegmentSpec(segment_id=1, motion_id=0, start_frame=1, phase=0.5, reset_mode_hint="preroll"),
        ),
        clean_state=state,
        reference_window=torch.zeros(2, 3, 2),
        phase=torch.tensor([0.2, 0.5]),
        horizon_k=torch.tensor([2, 2]),
        perturbation_family=("yaw", "planar"),
        perturbation_strength=torch.tensor([0.1, 0.2]),
    )


def test_reset_request_contains_velocity_fields_and_auto_preroll() -> None:
    adapter = FrontRESSegmentResetAdapter(default_preroll_steps=3)
    request = adapter.build_request(_batch(), mode="auto")
    assert request.root_lin_vel.shape == (2, 3)
    assert request.root_ang_vel.shape == (2, 3)
    assert request.dof_vel.shape == (2, 4)
    assert request.mode == ("direct", "preroll")
    assert request.preroll_steps.tolist() == [0, 3]


def test_static_direct_reset_is_rejected_by_result_mask() -> None:
    adapter = FrontRESSegmentResetAdapter(default_preroll_steps=3)
    request = adapter.build_request(_batch(static=True), mode="direct")
    result = adapter.validate_after_reset(None, {"reset_success": torch.tensor([True, True])}, request)
    assert result.invalid_static_reset_mask.tolist() == [True, True]
    assert result.success_mask.tolist() == [False, False]


def test_apply_returns_partial_failure_masks_without_exception() -> None:
    adapter = FrontRESSegmentResetAdapter(default_preroll_steps=3)
    request = adapter.build_request(_batch(), mode="auto")
    env = FakeEnv()
    result = adapter.apply(env, request)
    assert env.last_request is request
    assert result.success_mask.tolist() == [True, False]
    assert result.fall_at_reset_mask.tolist() == [False, True]
    assert result.diagnostics["direct_frac"] == 0.5
    assert result.diagnostics["preroll_frac"] == 0.5


def test_validate_after_reset_detects_velocity_mismatch() -> None:
    adapter = FrontRESSegmentResetAdapter(velocity_mismatch_tolerance=0.1)
    request = adapter.build_request(_batch(), mode="auto")
    result = adapter.validate_after_reset(None, {"root_lin_vel": torch.zeros(2, 3)}, request)
    assert result.velocity_mismatch[0] > 0.1
    assert result.success_mask.tolist() == [False, False]


def main() -> None:
    test_reset_request_contains_velocity_fields_and_auto_preroll()
    test_static_direct_reset_is_rejected_by_result_mask()
    test_apply_returns_partial_failure_masks_without_exception()
    test_validate_after_reset_detects_velocity_mismatch()
    print("result: PASS")


if __name__ == "__main__":
    main()
