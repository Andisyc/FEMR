#!/usr/bin/env python3
"""TEST-05 contract for TRAIN-v018 action-pre support conditioning."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from frontres_contract_imports import install_frontres_contract_packages


install_frontres_contract_packages()

from rsl_rl.modules.frontres_observation_layout import (
    FRONTRES_V018_CRITIC_DIM,
    FRONTRES_V018_CRITIC_SUPPORT_CONTEXT_DIM,
    compose_frontres_v018_critic_observation,
    compose_frontres_v018_critic_support_context,
)
from rsl_rl.runners import frontres_segment_physics as physics_owner


def _expect_error(call: Any, text: str) -> None:
    try:
        call()
    except (RuntimeError, TypeError, ValueError) as exc:
        assert text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("expected TRAIN-v018 support observation rejection")


def test_pure_support_layout_and_mask() -> None:
    contact = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    load = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    applicable = torch.tensor([1.0, 0.0])
    margin = torch.tensor([0.03, 0.0])
    planned = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
        ]
    )
    horizon = torch.tensor([3, 2])
    support = compose_frontres_v018_critic_support_context(
        current_contact=contact,
        vertical_load_fraction=load,
        zmp_applicable=applicable,
        zmp_margin=margin,
        planned_support=planned,
        planned_horizon=horizon,
    )
    assert tuple(support.shape) == (2, FRONTRES_V018_CRITIC_SUPPORT_CONTEXT_DIM) == (2, 102)
    torch.testing.assert_close(support[:, :2], contact)
    torch.testing.assert_close(support[:, 2:4], load)
    torch.testing.assert_close(support[:, 4], applicable)
    torch.testing.assert_close(support[:, 5], margin)
    planned_flat = support[:, 6:70].reshape(2, 32, 2)
    valid = support[:, 70:102]
    torch.testing.assert_close(planned_flat[0, :3], planned[0])
    torch.testing.assert_close(planned_flat[1, :2], planned[1, :2])
    assert not bool(planned_flat[1, 2].any())
    assert not bool(planned_flat[:, 3:].any())
    torch.testing.assert_close(valid[0, :3], torch.ones(3))
    torch.testing.assert_close(valid[1, :2], torch.ones(2))
    assert not bool(valid[0, 3:].any()) and not bool(valid[1, 2:].any())

    current = torch.arange(2 * 289, dtype=torch.float32).reshape(2, 289)
    tail = torch.arange(2 * 58, dtype=torch.float32).reshape(2, 58)
    critic = compose_frontres_v018_critic_observation(current, tail, support)
    assert tuple(critic.shape) == (2, FRONTRES_V018_CRITIC_DIM) == (2, 449)
    torch.testing.assert_close(critic[:, :289], current)
    torch.testing.assert_close(critic[:, 289:347], tail)
    torch.testing.assert_close(critic[:, 347:], support)
    order = torch.tensor([1, 0])
    torch.testing.assert_close(
        compose_frontres_v018_critic_observation(current[order], tail[order], support[order]),
        critic[order],
    )

    _expect_error(
        lambda: compose_frontres_v018_critic_support_context(
            current_contact=contact,
            vertical_load_fraction=load,
            zmp_applicable=applicable,
            zmp_margin=torch.tensor([0.03, 1.0]),
            planned_support=planned,
            planned_horizon=horizon,
        ),
        "inapplicable",
    )
    _expect_error(
        lambda: compose_frontres_v018_critic_observation(
            current, tail, support.clone().requires_grad_(True)
        ),
        "detached",
    )
    _expect_error(
        lambda: compose_frontres_v018_critic_support_context(
            current_contact=contact,
            vertical_load_fraction=load,
            zmp_applicable=applicable,
            zmp_margin=margin,
            planned_support=planned,
            planned_horizon=torch.tensor([True, True]),
        ),
        "non-boolean integer",
    )


def _gateway_fixture() -> tuple[Any, Any]:
    num_envs = 2
    foot_pos = torch.tensor(
        [
            [[-0.10, 0.08, 0.02], [0.10, -0.08, 0.02]],
            [[-0.08, 0.07, 0.02], [0.12, -0.07, 0.02]],
        ]
    )
    body_pos = torch.zeros(num_envs, 3, 3)
    body_pos[:, 1:] = foot_pos
    body_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num_envs, 3, 1)

    class _Scene(dict):
        pass

    def _sensor(side: int, loads: tuple[float, float]) -> Any:
        force = torch.zeros(num_envs, 1, 1, 3)
        force[:, 0, 0, 2] = torch.tensor(loads)
        return SimpleNamespace(
            side=side,
            data=SimpleNamespace(force_matrix_w=force),
            cfg=SimpleNamespace(force_threshold=10.0),
        )

    scene = _Scene(
        frontres_left_foot_contacts=_sensor(0, (75.0, 25.0)),
        frontres_right_foot_contacts=_sensor(1, (25.0, 75.0)),
    )
    scene.env_origins = torch.zeros(num_envs, 3)
    command = SimpleNamespace(
        num_envs=num_envs,
        left_foot_idx=1,
        right_foot_idx=2,
        robot_body_pos_w=body_pos,
        robot_body_quat_w=body_quat,
        frontres_local_scenario_intent_snapshot=lambda: {"timing": "action_pre"},
        frontres_local_scenario_snapshot=lambda _ids: {
            "expected_support": torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                ]
            ),
            "horizon_k": torch.tensor([3, 2]),
        },
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        env=SimpleNamespace(scene=scene),
        cfg={
            "frontres_expected_foot_half_length": 0.10,
            "frontres_expected_foot_half_width": 0.05,
        },
    )

    def _raw(sensor: Any, *, num_envs: int, device: torch.device):
        points = foot_pos[:, sensor.side : sensor.side + 1, None, :].to(device)
        force = sensor.data.force_matrix_w[..., 2].to(device)
        normals = torch.zeros_like(points)
        normals[..., 2] = 1.0
        return points, force, normals, torch.ones_like(force, dtype=torch.bool)

    return runner, (command, _raw)


def test_action_pre_gateway_uses_current_support_and_sealed_plan() -> None:
    runner, (command, raw_reader) = _gateway_fixture()
    original_command = physics_owner._motion_command_for_runner
    original_raw = physics_owner.read_frontres_raw_filtered_contact_rows
    try:
        physics_owner._motion_command_for_runner = lambda _runner: command
        physics_owner.read_frontres_raw_filtered_contact_rows = raw_reader
        context = physics_owner.capture_frontres_v018_critic_support_context(runner)
    finally:
        physics_owner._motion_command_for_runner = original_command
        physics_owner.read_frontres_raw_filtered_contact_rows = original_raw
    assert tuple(context.shape) == (2, 102)
    torch.testing.assert_close(context[:, :2], torch.ones(2, 2))
    torch.testing.assert_close(context[:, 2:4], torch.tensor([[0.75, 0.25], [0.25, 0.75]]))
    torch.testing.assert_close(context[:, 4], torch.ones(2))
    assert bool(torch.isfinite(context[:, 5]).all())
    torch.testing.assert_close(context[:, 70:73], torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))


def main() -> None:
    test_pure_support_layout_and_mask()
    test_action_pre_gateway_uses_current_support_and_sealed_plan()
    print("frontres_v018_support_conditioned_observation_contract: 289+58+102=449 exact", flush=True)


if __name__ == "__main__":
    main()
