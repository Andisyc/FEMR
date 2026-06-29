from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_dataset.py"
spec = importlib.util.spec_from_file_location("frontres_segment_dataset", MODULE_PATH)
dataset_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = dataset_module
spec.loader.exec_module(dataset_module)
FrontRESSegmentDataset = dataset_module.FrontRESSegmentDataset


def _fake_motion(motion_id: int, frames: int = 6, dofs: int = 4) -> dict[str, torch.Tensor | int | float | str]:
    base = torch.arange(frames, dtype=torch.float32).unsqueeze(-1)
    dof_base = torch.arange(frames * dofs, dtype=torch.float32).reshape(frames, dofs)
    return {
        "motion_id": motion_id,
        "root_pos": torch.cat([base, base + 1.0, base + 2.0], dim=-1),
        "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(frames, 1),
        "root_lin_vel": torch.ones(frames, 3) * (motion_id + 1),
        "root_ang_vel": torch.ones(frames, 3) * (motion_id + 2),
        "dof_pos": dof_base + motion_id,
        "dof_vel": torch.ones(frames, dofs) * (motion_id + 3),
        "reference": torch.arange(frames * 2, dtype=torch.float32).reshape(frames, 2) + motion_id,
        "perturbation_family": "yaw",
        "perturbation_strength": 0.2,
    }


def test_dataset_samples_stable_dynamic_segments() -> None:
    dataset = FrontRESSegmentDataset(
        motion_source=[_fake_motion(10), _fake_motion(11)],
        dt=0.02,
        default_horizon_k=2,
        device="cpu",
    )
    assert dataset.num_segments() == 8

    batch = dataset.get_segments(torch.tensor([0, 4]))
    assert batch.segment_ids.tolist() == [0, 4]
    assert batch.clean_state.root_pos.shape == (2, 3)
    assert batch.clean_state.root_lin_vel.shape == (2, 3)
    assert batch.clean_state.root_ang_vel.shape == (2, 3)
    assert batch.clean_state.dof_pos.shape == (2, 4)
    assert batch.clean_state.dof_vel.shape == (2, 4)
    assert batch.reference_window.shape == (2, 3, 2)
    assert batch.horizon_k.tolist() == [2, 2]
    assert batch.perturbation_family == ("yaw", "yaw")

    repeat = dataset.get_segments([0, 4])
    torch.testing.assert_close(repeat.clean_state.root_pos, batch.clean_state.root_pos)
    torch.testing.assert_close(repeat.clean_state.dof_vel, batch.clean_state.dof_vel)


def test_dataset_global_sampling_excludes_invalid_segments() -> None:
    dataset = FrontRESSegmentDataset([_fake_motion(0)], dt=0.02, default_horizon_k=2, device="cpu")
    dataset.update_validity([0, 1, 2], [False, False, False], reason="bad reset")

    generator = torch.Generator().manual_seed(0)
    batch = dataset.sample_global(16, generator=generator)
    assert all(segment_id == 3 for segment_id in batch.segment_ids.tolist())

    validation = dataset.validate_batch(dataset.get_segments([0, 3]))
    assert validation.valid_mask.tolist() == [False, True]
    assert validation.reasons[0] == "bad reset"


def test_dataset_state_dict_restores_invalidity_and_baseline() -> None:
    dataset = FrontRESSegmentDataset([_fake_motion(0)], dt=0.02, default_horizon_k=2, device="cpu")
    dataset.update_validity([1], [False], reason="unstable")
    dataset.write_noisy_baseline([2], {"score": 0.5})

    restored = FrontRESSegmentDataset([_fake_motion(0)], dt=0.02, default_horizon_k=2, device="cpu")
    restored.load_state_dict(dataset.state_dict())

    validation = restored.validate_batch(restored.get_segments([1, 2]))
    assert validation.valid_mask.tolist() == [False, True]
    assert restored.read_noisy_baseline([2])[2]["score"] == 0.5


def main() -> None:
    test_dataset_samples_stable_dynamic_segments()
    test_dataset_global_sampling_excludes_invalid_segments()
    test_dataset_state_dict_restores_invalidity_and_baseline()
    print("result: PASS")


if __name__ == "__main__":
    main()
