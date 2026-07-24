from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl"
manifest_path = SOURCE_ROOT / "rsl_rl" / "frontres" / "frontres_policy_quality_manifest.py"
manifest_spec = importlib.util.spec_from_file_location(
    "rsl_rl.frontres.frontres_policy_quality_manifest", manifest_path
)
manifest_module = importlib.util.module_from_spec(manifest_spec)
assert manifest_spec.loader is not None
sys.modules.setdefault("rsl_rl", ModuleType("rsl_rl"))
sys.modules.setdefault("rsl_rl.frontres", ModuleType("rsl_rl.frontres"))
sys.modules[manifest_spec.name] = manifest_module
manifest_spec.loader.exec_module(manifest_module)

MODULE_PATH = SOURCE_ROOT / "rsl_rl" / "runners" / "frontres_policy_quality_eval.py"
spec = importlib.util.spec_from_file_location("frontres_policy_quality_eval", MODULE_PATH)
state_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = state_module
spec.loader.exec_module(state_module)


class _FakeRobot:
    def __init__(self, rows: int) -> None:
        self.data = SimpleNamespace(
            root_state_w=torch.arange(rows * 13, dtype=torch.float32).reshape(rows, 13),
            joint_pos=torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4),
            joint_vel=torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) * 0.1,
        )

    def write_root_state_to_sim(self, value: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        self.data.root_state_w.index_copy_(0, env_ids, value)

    def write_joint_state_to_sim(
        self, positions: torch.Tensor, velocities: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.data.joint_pos.index_copy_(0, env_ids, positions)
        self.data.joint_vel.index_copy_(0, env_ids, velocities)


class _FakeScene(dict):
    def __init__(self, robot: _FakeRobot, rows: int) -> None:
        super().__init__(robot=robot)
        self.env_origins = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)


class _FakeCommand:
    def __init__(self, rows: int) -> None:
        self.time_steps = torch.arange(rows, dtype=torch.long) + 40
        self.env_motion_indices = torch.arange(rows, dtype=torch.long) + 3
        self._cached_perturbed_pos = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) * 0.01
        self._cached_perturbed_quat = torch.zeros(rows, 4)
        self._cached_perturbed_quat[:, 0] = 1.0
        self._frontres_pos_correction = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) * 0.001
        self._frontres_quat_correction = torch.zeros(rows, 4)
        self._frontres_quat_correction[:, 0] = 1.0
        self.perturber = SimpleNamespace(
            _x_state=torch.arange(rows, dtype=torch.float32) * 0.01,
            _roll_state=torch.arange(rows, dtype=torch.float32) * 0.02,
            _iid_event_steps_remaining=torch.arange(rows, dtype=torch.long) + 2,
            _iid_event_rp=torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) * 0.03,
            static_scalar=0.95,
        )


def _runner(rows: int = 4) -> SimpleNamespace:
    robot = _FakeRobot(rows)
    command = _FakeCommand(rows)
    raw = SimpleNamespace(
        scene=_FakeScene(robot, rows),
        command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None),
        episode_length_buf=torch.arange(rows, dtype=torch.long) + 10,
    )
    env = SimpleNamespace(unwrapped=raw, episode_length_buf=raw.episode_length_buf)
    return SimpleNamespace(env=env)


def test_zero_preroll_has_no_policy_route() -> None:
    observed: list[torch.Tensor] = []
    state_module.run_zero_frontres_preroll(
        lambda actions: observed.append(actions), num_envs=3, steps=4, device="cpu"
    )
    assert len(observed) == 4
    assert all(tuple(actions.shape) == (3, 6) and bool((actions == 0).all()) for actions in observed)


def test_complete_scoring_state_round_trip_restores_hash_and_rng() -> None:
    runner = _runner()
    signature = "a" * 64
    ids = (0, 2)
    random.seed(7)
    np.random.seed(11)
    torch.manual_seed(13)
    snapshot = state_module.capture_frontres_policy_quality_state(
        runner, env_ids=ids, comparison_signature=signature, role_layout=("policy", "clean")
    )
    expected_hash = snapshot.initial_state_hash
    different_roles = state_module.capture_frontres_policy_quality_state(
        runner, env_ids=ids, comparison_signature=signature, role_layout=("noisy", "clean")
    )
    assert different_roles.initial_state_hash != expected_hash

    robot = runner.env.unwrapped.scene["robot"]
    command = runner.env.unwrapped.command_manager.get_term("motion")
    robot.data.root_state_w[list(ids)] += 100
    robot.data.joint_pos[list(ids)] -= 100
    robot.data.joint_vel[list(ids)] += 10
    runner.env.unwrapped.scene.env_origins[list(ids)] += 20
    runner.env.episode_length_buf[list(ids)] += 30
    for name, _ in snapshot.command_state:
        getattr(command, name)[list(ids)] += 1
    for qualified_name, _ in snapshot.perturber_state:
        getattr(command.perturber, qualified_name.removeprefix("perturber."))[list(ids)] += 1
    random.random()
    np.random.random()
    torch.rand(3)

    identity = state_module.restore_frontres_policy_quality_state(
        runner, snapshot, comparison_signature=signature
    )
    assert identity.initial_state_hash == expected_hash
    recaptured = state_module.capture_frontres_policy_quality_state(
        runner, env_ids=ids, comparison_signature=signature, role_layout=("policy", "clean")
    )
    assert recaptured == snapshot
    print(f"[quality state trace] env_ids={ids} initial_state_hash={expected_hash}")


def test_restore_rejects_comparison_mismatch_and_missing_cache() -> None:
    runner = _runner()
    snapshot = state_module.capture_frontres_policy_quality_state(
        runner,
        env_ids=(1, 3),
        comparison_signature="b" * 64,
        role_layout=("policy", "noisy"),
    )
    try:
        state_module.restore_frontres_policy_quality_state(
            runner, snapshot, comparison_signature="c" * 64
        )
    except ValueError as exc:
        assert "signature mismatch" in str(exc)
    else:
        raise AssertionError("comparison mismatch must fail closed")

    del runner.env.unwrapped.command_manager.get_term("motion")._cached_perturbed_pos
    try:
        state_module.capture_frontres_policy_quality_state(
            runner,
            env_ids=(1, 3),
            comparison_signature="b" * 64,
            role_layout=("policy", "noisy"),
        )
    except AttributeError as exc:
        assert "_cached_perturbed_pos" in str(exc)
    else:
        raise AssertionError("missing command cache must fail closed")


def test_restore_rows_supports_isaac_inference_tensors() -> None:
    with torch.inference_mode():
        target = torch.zeros((4, 2), dtype=torch.float32)
    image = state_module._TensorImage.capture(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    state_module._restore_rows(target, torch.tensor([1, 3]), image)

    assert target.tolist() == [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]


def test_local_scenario_route_start_restores_command_lifecycle() -> None:
    runner = _runner()
    command = runner.env.unwrapped.command_manager.get_term("motion")
    rows = int(command.time_steps.numel())
    command._frontres_local_scenario_active = torch.ones(rows, dtype=torch.bool)
    command._frontres_local_scenario_current_frame_ready = torch.ones(rows, dtype=torch.bool)
    command._frontres_local_scenario_k_execution_active = torch.zeros(rows, dtype=torch.bool)
    command._frontres_local_scenario_k_execution_cursor = torch.full((rows,), -1, dtype=torch.long)
    signature = "d" * 64
    snapshot = state_module.capture_frontres_policy_quality_state(
        runner,
        env_ids=tuple(range(rows)),
        comparison_signature=signature,
        role_layout=("repair", "repair", "noisy", "noisy"),
    )

    # Simulate one-action-K completion before the next counterfactual route.
    command._frontres_local_scenario_current_frame_ready.zero_()
    command._frontres_local_scenario_k_execution_active.zero_()
    command._frontres_local_scenario_k_execution_cursor.fill_(-1)
    state_module.restore_frontres_policy_quality_state(
        runner,
        snapshot,
        comparison_signature=signature,
    )

    assert bool(command._frontres_local_scenario_active.all())
    assert bool(command._frontres_local_scenario_current_frame_ready.all())
    assert not bool(command._frontres_local_scenario_k_execution_active.any())
    assert bool((command._frontres_local_scenario_k_execution_cursor == -1).all())


def test_active_local_scenario_rejects_missing_lifecycle_state() -> None:
    runner = _runner()
    command = runner.env.unwrapped.command_manager.get_term("motion")
    command._frontres_local_scenario_active = torch.ones(command.time_steps.numel(), dtype=torch.bool)
    try:
        state_module.capture_frontres_policy_quality_state(
            runner,
            env_ids=(0, 1),
            comparison_signature="e" * 64,
            role_layout=("repair", "noisy"),
        )
    except RuntimeError as exc:
        assert "lifecycle field is invalid" in str(exc)
    else:
        raise AssertionError("active local scenario must reject an incomplete route-start lifecycle snapshot")


if __name__ == "__main__":
    test_zero_preroll_has_no_policy_route()
    test_complete_scoring_state_round_trip_restores_hash_and_rng()
    test_restore_rejects_comparison_mismatch_and_missing_cache()
    test_restore_rows_supports_isaac_inference_tensors()
    test_local_scenario_route_start_restores_command_lifecycle()
    test_active_local_scenario_rejects_missing_lifecycle_state()
    print("PASS: policy-quality scoring state capture and restore are closed offline.")
