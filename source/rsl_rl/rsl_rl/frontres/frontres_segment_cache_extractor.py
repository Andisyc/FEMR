from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

import torch

from rsl_rl.frontres.frontres_segment_cache_schema import FrontRESRobotRolloutState


def extract_robot_rollout_state(
    env: Any,
    env_ids: Iterable[int] | torch.Tensor | None = None,
    *,
    robot_name: str = "robot",
    contact_state: torch.Tensor | None = None,
    action_history: torch.Tensor | None = None,
) -> FrontRESRobotRolloutState:
    base = getattr(env, "unwrapped", env)
    robot = resolve_robot(base, robot_name=robot_name)
    data = getattr(robot, "data", robot)
    ids = _normalize_env_ids(env_ids, data.joint_pos)
    if action_history is None:
        action_history = _capture_action_history(base)
    if contact_state is None:
        contact_state = _capture_contact_state(base)
    state = FrontRESRobotRolloutState(
        root_pos=_select_detached(data.root_pos_w, ids),
        root_quat=_select_detached(data.root_quat_w, ids),
        root_lin_vel=_select_detached(data.root_lin_vel_w, ids),
        root_ang_vel=_select_detached(data.root_ang_vel_w, ids),
        joint_pos=_select_detached(data.joint_pos, ids),
        joint_vel=_select_detached(data.joint_vel, ids),
        body_pos_w=_select_detached(data.body_pos_w, ids),
        body_quat_w=_select_detached(data.body_quat_w, ids),
        body_lin_vel_w=_select_detached(data.body_lin_vel_w, ids),
        body_ang_vel_w=_select_detached(data.body_ang_vel_w, ids),
        contact_state=None if contact_state is None else _select_detached(contact_state, ids),
        action_history=None if action_history is None else _select_detached(action_history, ids),
    )
    state.validate(name="extracted_state")
    return state


def _capture_action_history(env: Any) -> torch.Tensor | None:
    """Capture the public ActionManager Markov state as [previous, current]."""

    manager = getattr(env, "action_manager", None)
    if manager is None:
        return None
    previous = getattr(manager, "prev_action", None)
    current = getattr(manager, "action", None)
    if not isinstance(previous, torch.Tensor) or not isinstance(current, torch.Tensor):
        raise RuntimeError("FrontRES cache requires ActionManager.prev_action and action tensors")
    if previous.ndim != 2 or current.shape != previous.shape:
        raise RuntimeError(
            "FrontRES cache action history requires matching rank-2 previous/current actions"
        )
    if not bool(torch.isfinite(previous).all().item()) or not bool(torch.isfinite(current).all().item()):
        raise RuntimeError("FrontRES cache action history must be finite")
    return torch.stack((previous, current), dim=1)


def _capture_contact_state(env: Any) -> torch.Tensor | None:
    """Capture binary left/right support from the official foot sensors."""

    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    names = ("frontres_left_foot_contacts", "frontres_right_foot_contacts")
    sensors: list[Any] = []
    scene_sensors = getattr(scene, "sensors", None)
    for name in names:
        sensor = scene_sensors.get(name) if isinstance(scene_sensors, Mapping) else getattr(scene, name, None)
        if sensor is not None:
            sensors.append(sensor)
    if not sensors:
        return None
    if len(sensors) != len(names):
        raise RuntimeError("FrontRES cache requires both left and right foot contact sensors")
    loaded: list[torch.Tensor] = []
    for name, sensor in zip(names, sensors, strict=True):
        force = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
        threshold = float(getattr(getattr(sensor, "cfg", None), "force_threshold", 0.0))
        if not isinstance(force, torch.Tensor) or force.ndim < 2 or int(force.shape[-1]) != 3:
            raise RuntimeError(f"FrontRES cache sensor {name} has no valid force_matrix_w")
        if threshold <= 0.0 or not bool(torch.isfinite(force).all().item()):
            raise RuntimeError(f"FrontRES cache sensor {name} has invalid force evidence")
        vertical = force[..., 2].abs().reshape(int(force.shape[0]), -1).sum(dim=-1)
        loaded.append((vertical >= threshold).to(dtype=torch.float32))
    return torch.stack(loaded, dim=-1)


def resolve_robot(env: Any, *, robot_name: str = "robot") -> Any:
    base = getattr(env, "unwrapped", env)
    scene = getattr(base, "scene", None)
    if scene is not None:
        try:
            return scene[robot_name]
        except (KeyError, TypeError):
            pass
        articulations = getattr(scene, "articulations", None)
        if isinstance(articulations, dict) and robot_name in articulations:
            return articulations[robot_name]
    if hasattr(base, robot_name):
        return getattr(base, robot_name)
    if hasattr(base, "robot"):
        return getattr(base, "robot")
    raise AttributeError(f"could not resolve robot '{robot_name}' from env")


def robot_state_probe(state: FrontRESRobotRolloutState, *, prefix: str = "extracted_state") -> dict[str, Any]:
    return state.probe(prefix=prefix)


def _normalize_env_ids(env_ids: Iterable[int] | torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    if env_ids is None:
        return None
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=reference.device, dtype=torch.long).flatten()
    return torch.tensor(list(env_ids), dtype=torch.long, device=reference.device)


def _select_detached(tensor: torch.Tensor, env_ids: torch.Tensor | None) -> torch.Tensor:
    value = tensor if env_ids is None else tensor.index_select(0, env_ids)
    return value.detach().clone()
