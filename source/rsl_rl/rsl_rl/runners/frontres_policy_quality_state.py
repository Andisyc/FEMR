"""Capture and restore the immutable policy-quality route-start state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pickle
import random
from typing import Any

import numpy as np
import torch

from rsl_rl.frontres.frontres_policy_quality_manifest import FrontRESPolicyQualityStateIdentity


_COMMAND_STATE_FIELDS = (
    "time_steps",
    "env_motion_indices",
    "_cached_perturbed_pos",
    "_cached_perturbed_quat",
    "_frontres_pos_correction",
    "_frontres_quat_correction",
)

_LOCAL_SCENARIO_LIFECYCLE_FIELDS = (
    "_frontres_local_scenario_active",
    "_frontres_local_scenario_current_frame_ready",
    "_frontres_local_scenario_k_execution_active",
    "_frontres_local_scenario_k_execution_cursor",
)


def _policy_quality_command_state_fields(command: Any) -> tuple[str, ...]:
    active = getattr(command, "_frontres_local_scenario_active", None)
    if not isinstance(active, torch.Tensor) or not bool(active.any().item()):
        return _COMMAND_STATE_FIELDS
    if active.ndim != 1 or not bool(active.all().item()):
        raise RuntimeError("policy-quality state requires one transaction-wide active local scenario")
    for name in _LOCAL_SCENARIO_LIFECYCLE_FIELDS:
        value = getattr(command, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or int(value.shape[0]) != int(active.shape[0]):
            raise RuntimeError(f"policy-quality local-scenario lifecycle field is invalid: {name}")
    return (*_COMMAND_STATE_FIELDS, *_LOCAL_SCENARIO_LIFECYCLE_FIELDS)


@dataclass(frozen=True)
class FrontRESPolicyQualityTensorImage:
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> FrontRESPolicyQualityTensorImage:
        value = tensor.detach().contiguous().cpu()
        return cls(dtype=str(value.dtype), shape=tuple(value.shape), data=value.numpy().tobytes(order="C"))

    def restore(self, *, device: torch.device | str) -> torch.Tensor:
        dtype = getattr(torch, self.dtype.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported snapshot dtype: {self.dtype}")
        value = torch.frombuffer(bytearray(self.data), dtype=dtype).clone()
        return value.reshape(self.shape).to(device=device)

    def update_hash(self, digest: Any, *, name: str) -> None:
        digest.update(name.encode("utf-8"))
        digest.update(self.dtype.encode("ascii"))
        digest.update(repr(self.shape).encode("ascii"))
        digest.update(self.data)


@dataclass(frozen=True)
class FrontRESPolicyQualityScoringState:
    comparison_signature: str
    env_ids: tuple[int, ...]
    role_layout: tuple[str, ...]
    root_state_w: FrontRESPolicyQualityTensorImage
    joint_pos: FrontRESPolicyQualityTensorImage
    joint_vel: FrontRESPolicyQualityTensorImage
    env_origins: FrontRESPolicyQualityTensorImage
    episode_length: FrontRESPolicyQualityTensorImage
    command_state: tuple[tuple[str, FrontRESPolicyQualityTensorImage], ...]
    perturber_state: tuple[tuple[str, FrontRESPolicyQualityTensorImage], ...]
    python_rng_state: bytes
    numpy_rng_state: bytes
    torch_rng_state: FrontRESPolicyQualityTensorImage
    cuda_rng_state: tuple[FrontRESPolicyQualityTensorImage, ...]

    @property
    def initial_state_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.comparison_signature.encode("ascii"))
        digest.update(repr(self.env_ids).encode("ascii"))
        digest.update(repr(self.role_layout).encode("utf-8"))
        for name, image in (
            ("root_state_w", self.root_state_w),
            ("joint_pos", self.joint_pos),
            ("joint_vel", self.joint_vel),
            ("env_origins", self.env_origins),
            ("episode_length", self.episode_length),
            *self.command_state,
            *self.perturber_state,
            ("torch_rng_state", self.torch_rng_state),
        ):
            image.update_hash(digest, name=name)
        for index, image in enumerate(self.cuda_rng_state):
            image.update_hash(digest, name=f"cuda_rng_state[{index}]")
        digest.update(self.python_rng_state)
        digest.update(self.numpy_rng_state)
        return digest.hexdigest()

    @property
    def state_identity(self) -> FrontRESPolicyQualityStateIdentity:
        return FrontRESPolicyQualityStateIdentity(
            comparison_signature=self.comparison_signature,
            initial_state_hash=self.initial_state_hash,
        )


def capture_frontres_policy_quality_state(
    runner: Any,
    *,
    env_ids: torch.Tensor | tuple[int, ...] | list[int],
    comparison_signature: str,
    role_layout: tuple[str, ...] | list[str],
) -> FrontRESPolicyQualityScoringState:
    ids = _normalize_env_ids(env_ids)
    roles = _normalize_role_layout(role_layout, count=int(ids.numel()))
    env, raw_env = resolve_frontres_policy_quality_envs(runner)
    robot = _resolve_robot(raw_env)
    command = resolve_frontres_policy_quality_command(raw_env)
    origins = _require_tensor(getattr(getattr(raw_env, "scene", None), "env_origins", None), "env_origins")
    episode = _require_tensor(
        getattr(env, "episode_length_buf", getattr(raw_env, "episode_length_buf", None)),
        "episode_length_buf",
    )
    command_state = tuple(
        (name, _capture_rows(_require_tensor(getattr(command, name, None), f"command.{name}"), ids))
        for name in _policy_quality_command_state_fields(command)
    )
    perturber = getattr(command, "perturber", None)
    if perturber is None:
        raise AttributeError("policy-quality state capture requires command.perturber")
    perturber_state = tuple(
        (f"perturber.{name}", _capture_rows(value, ids))
        for name, value in sorted(vars(perturber).items())
        if isinstance(value, torch.Tensor)
        and value.ndim > 0
        and int(value.shape[0]) > int(ids.max().item())
    )
    if not perturber_state:
        raise AttributeError("policy-quality state capture found no per-env perturber tensors")
    cuda_rng = tuple(FrontRESPolicyQualityTensorImage.capture(state) for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return FrontRESPolicyQualityScoringState(
        comparison_signature=comparison_signature,
        env_ids=tuple(ids.tolist()),
        role_layout=roles,
        root_state_w=_capture_rows(_require_tensor(robot.data.root_state_w, "robot.root_state_w"), ids),
        joint_pos=_capture_rows(_require_tensor(robot.data.joint_pos, "robot.joint_pos"), ids),
        joint_vel=_capture_rows(_require_tensor(robot.data.joint_vel, "robot.joint_vel"), ids),
        env_origins=_capture_rows(origins, ids),
        episode_length=_capture_rows(episode, ids),
        command_state=command_state,
        perturber_state=perturber_state,
        python_rng_state=pickle.dumps(random.getstate(), protocol=5),
        numpy_rng_state=pickle.dumps(np.random.get_state(), protocol=5),
        torch_rng_state=FrontRESPolicyQualityTensorImage.capture(torch.random.get_rng_state()),
        cuda_rng_state=cuda_rng,
    )


def restore_frontres_policy_quality_state(
    runner: Any,
    snapshot: FrontRESPolicyQualityScoringState,
    *,
    comparison_signature: str,
) -> FrontRESPolicyQualityStateIdentity:
    if comparison_signature != snapshot.comparison_signature:
        raise ValueError("comparison signature mismatch during policy-quality state restore")
    env, raw_env = resolve_frontres_policy_quality_envs(runner)
    robot = _resolve_robot(raw_env)
    command = resolve_frontres_policy_quality_command(raw_env)
    ids = torch.tensor(snapshot.env_ids, dtype=torch.long)
    root_target = _require_tensor(robot.data.root_state_w, "robot.root_state_w")
    joint_pos_target = _require_tensor(robot.data.joint_pos, "robot.joint_pos")
    joint_vel_target = _require_tensor(robot.data.joint_vel, "robot.joint_vel")
    sim_ids = ids.to(root_target.device)
    robot.write_root_state_to_sim(snapshot.root_state_w.restore(device=root_target.device), env_ids=sim_ids)
    robot.write_joint_state_to_sim(
        snapshot.joint_pos.restore(device=joint_pos_target.device),
        snapshot.joint_vel.restore(device=joint_vel_target.device),
        env_ids=sim_ids,
    )
    origins = _require_tensor(getattr(getattr(raw_env, "scene", None), "env_origins", None), "env_origins")
    _restore_rows(origins, ids, snapshot.env_origins)
    episode = _require_tensor(
        getattr(env, "episode_length_buf", getattr(raw_env, "episode_length_buf", None)),
        "episode_length_buf",
    )
    _restore_rows(episode, ids, snapshot.episode_length)
    for name, image in snapshot.command_state:
        _restore_rows(_require_tensor(getattr(command, name, None), f"command.{name}"), ids, image)
    perturber = getattr(command, "perturber", None)
    if perturber is None:
        raise AttributeError("policy-quality state restore requires command.perturber")
    for qualified_name, image in snapshot.perturber_state:
        _restore_rows(
            _require_tensor(getattr(perturber, qualified_name.removeprefix("perturber."), None), qualified_name),
            ids,
            image,
        )
    random.setstate(pickle.loads(snapshot.python_rng_state))
    np.random.set_state(pickle.loads(snapshot.numpy_rng_state))
    torch.random.set_rng_state(snapshot.torch_rng_state.restore(device="cpu"))
    if snapshot.cuda_rng_state:
        if not torch.cuda.is_available() or len(snapshot.cuda_rng_state) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG topology differs from captured policy-quality state")
        torch.cuda.set_rng_state_all([image.restore(device="cpu") for image in snapshot.cuda_rng_state])
    restored = capture_frontres_policy_quality_state(
        runner,
        env_ids=snapshot.env_ids,
        comparison_signature=comparison_signature,
        role_layout=snapshot.role_layout,
    ).state_identity
    if restored.initial_state_hash != snapshot.initial_state_hash:
        raise RuntimeError(
            "policy-quality scoring state restore mismatch: "
            f"expected={snapshot.initial_state_hash} observed={restored.initial_state_hash}"
        )
    return restored


def resolve_frontres_policy_quality_envs(runner: Any) -> tuple[Any, Any]:
    env = getattr(runner, "env", None)
    if env is None:
        raise AttributeError("policy-quality state capture requires runner.env")
    return env, getattr(env, "unwrapped", env)


def resolve_frontres_policy_quality_command(raw_env: Any) -> Any:
    manager = getattr(raw_env, "command_manager", None)
    if manager is None or not hasattr(manager, "get_term"):
        raise AttributeError("policy-quality state capture requires command_manager.get_term('motion')")
    return manager.get_term("motion")


def _normalize_env_ids(env_ids: torch.Tensor | tuple[int, ...] | list[int]) -> torch.Tensor:
    ids = env_ids.detach().to(device="cpu", dtype=torch.long).flatten() if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, dtype=torch.long)
    if ids.numel() == 0 or bool((ids < 0).any()) or int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError("env_ids must be non-empty, non-negative, and unique")
    return ids


def _normalize_role_layout(role_layout: tuple[str, ...] | list[str], *, count: int) -> tuple[str, ...]:
    if not isinstance(role_layout, (tuple, list)) or len(role_layout) != count:
        raise ValueError(f"role_layout must contain exactly {count} entries")
    roles = tuple(str(role).strip() for role in role_layout)
    if any(not role for role in roles):
        raise ValueError("role_layout entries must be non-empty")
    return roles


def _resolve_robot(raw_env: Any) -> Any:
    scene = getattr(raw_env, "scene", None)
    try:
        robot = scene["robot"]
    except (KeyError, TypeError):
        robot = getattr(scene, "robot", None)
    if robot is None or not hasattr(robot, "data"):
        raise AttributeError("policy-quality state capture requires scene['robot']")
    return robot


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise AttributeError(f"policy-quality state requires tensor {name}")
    return value


def _capture_rows(tensor: torch.Tensor, ids: torch.Tensor) -> FrontRESPolicyQualityTensorImage:
    if tensor.ndim == 0 or int(tensor.shape[0]) <= int(ids.max().item()):
        raise ValueError(f"state tensor shape {tuple(tensor.shape)} cannot select env_ids={ids.tolist()}")
    return FrontRESPolicyQualityTensorImage.capture(tensor.index_select(0, ids.to(tensor.device)))


def _restore_rows(target: torch.Tensor, ids: torch.Tensor, image: FrontRESPolicyQualityTensorImage) -> None:
    values = image.restore(device=target.device)
    target_ids = ids.to(target.device)
    expected = (int(target_ids.numel()), *tuple(target.shape[1:]))
    if tuple(values.shape) != expected:
        raise ValueError(f"snapshot shape {tuple(values.shape)} does not match restore target {expected}")
    with torch.inference_mode():
        target.index_copy_(0, target_ids, values.to(dtype=target.dtype))


__all__ = [
    "FrontRESPolicyQualityScoringState",
    "capture_frontres_policy_quality_state",
    "resolve_frontres_policy_quality_command",
    "resolve_frontres_policy_quality_envs",
    "restore_frontres_policy_quality_state",
]
