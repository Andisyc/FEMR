from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_hrl_action.py"
spec = importlib.util.spec_from_file_location("frontres_hrl_action", MODULE_PATH)
action_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = action_module
spec.loader.exec_module(action_module)
FrontRESHRLActionProjector = action_module.FrontRESHRLActionProjector


def test_projector_respects_active_mode_and_dz_rule() -> None:
    projector = FrontRESHRLActionProjector(
        active_task_dims=[0, 1, 2, 3, 4, 5],
        action_scale=[1.0, 1.0, 1.0, 0.5, 0.5, 1.0],
        upward_dz_rule="nonpositive",
    )
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    action = projector.project(raw, mode_groups=(("planar", "yaw"), ("global_z", "local_rp")))

    expected = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 6.0], [0.0, 0.0, 0.0, 2.0, 2.5, 0.0]])
    torch.testing.assert_close(action.projected_delta_se, expected)
    assert action.delta_se.shape == (2, 6)
    assert action.active_mask.shape == (2, 6)
    assert not hasattr(action, "acceptance_prob")
    assert not hasattr(action, "acceptance_logit")


def test_projector_applies_to_reference_without_acceptance_fields() -> None:
    projector = FrontRESHRLActionProjector(active_task_dims=[0, 1, 5], upward_dz_rule="allow")
    action = projector.project(torch.ones(2, 6))
    command = projector.apply_to_reference({}, action)
    assert set(command.keys()) == {"frontres_delta_se", "frontres_pos_correction", "frontres_rpy_correction"}
    torch.testing.assert_close(command["frontres_delta_se"][:, 2:5], torch.zeros(2, 3))


def test_hsl_initialization_copies_matching_repair_actor_weights() -> None:
    hsl_actor = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 6))
    repair_actor = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 6))
    with torch.no_grad():
        for param in hsl_actor.parameters():
            param.fill_(0.25)
        for param in repair_actor.parameters():
            param.fill_(0.0)

    copied = FrontRESHRLActionProjector.initialize_repair_actor_from_hsl(repair_actor, hsl_actor)
    assert copied
    for repair_param, hsl_param in zip(repair_actor.parameters(), hsl_actor.parameters()):
        torch.testing.assert_close(repair_param, hsl_param)


def main() -> None:
    test_projector_respects_active_mode_and_dz_rule()
    test_projector_applies_to_reference_without_acceptance_fields()
    test_hsl_initialization_copies_matching_repair_actor_weights()
    print("result: PASS")


if __name__ == "__main__":
    main()
