"""TEST ONLY: MotionPerturber burst temporal events for FrontRES."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4] / "source"
MODULE_PATH = (
    ROOT
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "motion_perturbations.py"
)


isaaclab = types.ModuleType("isaaclab")
isaaclab_utils = types.ModuleType("isaaclab.utils")
isaaclab_utils.configclass = lambda cls: cls
sys.modules.setdefault("isaaclab", isaaclab)
sys.modules.setdefault("isaaclab.utils", isaaclab_utils)

spec = importlib.util.spec_from_file_location("frontres_motion_perturbations_test_module", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}.")
motion_perturbations = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = motion_perturbations
spec.loader.exec_module(motion_perturbations)

MotionPerturbationCfg = motion_perturbations.MotionPerturbationCfg
MotionPerturber = motion_perturbations.MotionPerturber


def _make_perturber() -> MotionPerturber:
    cfg = MotionPerturbationCfg()
    cfg.iid_temporal_mode = "burst"
    cfg.iid_burst_min_steps = 3
    cfg.iid_burst_max_steps = 3
    cfg.iid_prob_xy = 1.0
    cfg.iid_std_xy = 0.10
    cfg.iid_prob_z = 0.0
    cfg.iid_prob_rp = 0.0
    cfg.iid_prob_ya = 0.0
    cfg.local_root_artifact_prob = 0.0
    perturber = MotionPerturber(cfg, num_envs=2, device="cpu")
    perturber._dr_scale = 1.0
    return perturber


def _make_local_artifact_perturber() -> MotionPerturber:
    cfg = MotionPerturbationCfg()
    cfg.iid_temporal_mode = "burst"
    cfg.iid_prob_xy = 0.0
    cfg.iid_prob_z = 0.0
    cfg.iid_prob_rp = 0.0
    cfg.iid_prob_ya = 0.0
    cfg.local_root_artifact_prob = 1.0
    cfg.local_root_artifact_min_steps = 2
    cfg.local_root_artifact_max_steps = 2
    cfg.local_root_artifact_xy_std = 0.10
    cfg.local_root_artifact_yaw_std = 0.10
    perturber = MotionPerturber(cfg, num_envs=2, device="cpu")
    perturber._dr_scale = 1.0
    return perturber


def test_burst_iid_xy_is_held_for_event_duration() -> None:
    torch.manual_seed(7)
    perturber = _make_perturber()
    root = torch.zeros(2, 3)
    feet = torch.zeros(2, 3)

    first = perturber.apply_perturbations(root, feet, feet)
    first_state = perturber.frontres_authority_event_state()
    second = perturber.apply_perturbations(root, feet, feet)
    second_state = perturber.frontres_authority_event_state()
    third = perturber.apply_perturbations(root, feet, feet)
    third_state = perturber.frontres_authority_event_state()

    torch.testing.assert_close(first[:, :2], second[:, :2])
    torch.testing.assert_close(second[:, :2], third[:, :2])
    assert bool(first_state["event_start"].all().item())
    assert not bool(second_state["event_start"].any().item())
    assert not bool(third_state["event_start"].any().item())
    torch.testing.assert_close(first_state["event_step"].float(), torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(second_state["event_step"].float(), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(third_state["event_step"].float(), torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(first_state["event_duration"].float(), torch.tensor([3.0, 3.0]))


def test_local_root_artifact_is_exposed_as_authority_event() -> None:
    torch.manual_seed(13)
    perturber = _make_local_artifact_perturber()
    root = torch.zeros(2, 3)
    feet = torch.zeros(2, 3)

    _ = perturber.apply_perturbations(root, feet, feet)
    first_state = perturber.frontres_authority_event_state()
    _ = perturber.apply_perturbations(root, feet, feet)
    second_state = perturber.frontres_authority_event_state()

    assert bool(first_state["event_start"].all().item())
    assert bool(first_state["event_active"].all().item())
    assert not bool(second_state["event_start"].any().item())
    assert bool(second_state["event_active"].all().item())
    torch.testing.assert_close(first_state["event_step"].float(), torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(second_state["event_step"].float(), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(first_state["event_duration"].float(), torch.tensor([2.0, 2.0]))


def test_reset_clears_burst_event_state() -> None:
    torch.manual_seed(11)
    perturber = _make_perturber()
    root = torch.zeros(2, 3)
    feet = torch.zeros(2, 3)
    _ = perturber.apply_perturbations(root, feet, feet)
    perturber.reset_envs(torch.tensor([0, 1], dtype=torch.long))
    state = perturber.frontres_authority_event_state()
    assert not bool(state["event_start"].any().item())
    assert not bool(state["event_active"].any().item())
    torch.testing.assert_close(state["event_duration"].float(), torch.zeros(2))


def main() -> None:
    test_burst_iid_xy_is_held_for_event_duration()
    test_local_root_artifact_is_exposed_as_authority_event()
    test_reset_clears_burst_event_state()
    print("=== FrontRES Burst Perturbation TEST ONLY ===")
    print("checks=held IID burst values, local artifact event exposure, event metadata, reset clear")
    print("result: PASS")


if __name__ == "__main__":
    main()
