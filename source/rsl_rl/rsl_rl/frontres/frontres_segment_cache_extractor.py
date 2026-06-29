from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

try:
    from rsl_rl.frontres.frontres_segment_cache_schema import FrontRESRobotRolloutState
except ModuleNotFoundError:
    _SCHEMA_PATH = Path(__file__).with_name("frontres_segment_cache_schema.py")
    _SCHEMA_SPEC = importlib.util.spec_from_file_location("frontres_segment_cache_schema", _SCHEMA_PATH)
    if _SCHEMA_SPEC is None or _SCHEMA_SPEC.loader is None:
        raise
    _SCHEMA_MODULE = importlib.util.module_from_spec(_SCHEMA_SPEC)
    sys.modules[_SCHEMA_SPEC.name] = _SCHEMA_MODULE
    _SCHEMA_SPEC.loader.exec_module(_SCHEMA_MODULE)
    FrontRESRobotRolloutState = _SCHEMA_MODULE.FrontRESRobotRolloutState


def extract_robot_rollout_state(
    env: Any,
    env_ids: Iterable[int] | torch.Tensor | None = None,
    *,
    robot_name: str = "robot",
    contact_state: torch.Tensor | None = None,
    action_history: torch.Tensor | None = None,
) -> FrontRESRobotRolloutState:
    robot = resolve_robot(env, robot_name=robot_name)
    data = getattr(robot, "data", robot)
    ids = _normalize_env_ids(env_ids, data.joint_pos)
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
