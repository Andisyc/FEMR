from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import torch


def _load_same_dir(module_name: str):
    path = Path(__file__).with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    from rsl_rl.frontres.frontres_segment_cache_noisy_capture import FrontRESNoisyBaselineResult
    from rsl_rl.frontres.frontres_segment_cache_schema import (
        FrontRESPerturbationDescriptor,
        FrontRESRobotRolloutState,
        FrontRESSegmentIndex,
    )
except ModuleNotFoundError:
    _noisy_capture = _load_same_dir("frontres_segment_cache_noisy_capture")
    _schema = _load_same_dir("frontres_segment_cache_schema")
    FrontRESNoisyBaselineResult = _noisy_capture.FrontRESNoisyBaselineResult
    FrontRESPerturbationDescriptor = _schema.FrontRESPerturbationDescriptor
    FrontRESRobotRolloutState = _schema.FrontRESRobotRolloutState
    FrontRESSegmentIndex = _schema.FrontRESSegmentIndex


@dataclass
class FrontRESStage1EnvAdapter:
    env: Any
    amass_root: str
    robot_name: str = "robot"
    trace: bool = True
    baseline_rollout_steps: int | None = None
    trace_preview_count: int = 4

    def __post_init__(self) -> None:
        self.base_env = getattr(self.env, "unwrapped", self.env)
        self.command = self._resolve_motion_command()
        self.robot = getattr(self.command, "robot", None) or self._resolve_robot()
        self.motion_path_to_index = self._build_motion_path_index()

    @property
    def unwrapped(self) -> Any:
        return self.base_env

    @property
    def scene(self) -> Any:
        return self.base_env.scene

    def frontres_loaded_motion_paths(self) -> list[str]:
        return [str(path) for path in getattr(self.command.motion_dir_loader, "motion_paths", [])]

    def frontres_motion_loader_probe(self) -> dict[str, Any]:
        loader = getattr(self.command, "motion_dir_loader", None)
        cfg = getattr(self.command, "cfg", None)
        loaded_paths = list(getattr(loader, "motion_paths", []) or [])
        all_paths = list(getattr(loader, "motion_paths_all", []) or [])
        shard_info = dict(getattr(loader, "shard_info", {}) or {})
        return {
            "loaded_motion_count": len(loaded_paths),
            "all_motion_count": len(all_paths),
            "cfg_motion_dataset_load_cap": getattr(cfg, "motion_dataset_load_cap", None),
            "cfg_motion_dataset_shard_across_gpus": getattr(cfg, "motion_dataset_shard_across_gpus", None),
            "shard_selected_motions": shard_info.get("selected_motions"),
            "shard_total_motions": shard_info.get("total_motions"),
            "first_loaded_motion": str(loaded_paths[0]) if loaded_paths else "none",
        }

    def ensure_frontres_env_reset(self) -> dict[str, bool]:
        if bool(getattr(self, "_frontres_env_reset_done", False)):
            return {"reset_called": False, "already_reset": True}
        reset_fn = getattr(self.env, "reset", None)
        if not callable(reset_fn):
            self._frontres_env_reset_done = True
            self._trace("env_reset", reset_called=False, already_reset=False)
            return {"reset_called": False, "already_reset": False}
        result = reset_fn()
        self._frontres_env_reset_done = True
        self._trace("env_reset", reset_called=True, already_reset=False, result_type=type(result).__name__)
        return {"reset_called": True, "already_reset": False}

    def prepare_frontres_clean_segment(self, *, segment: FrontRESSegmentIndex, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        segment.validate()
        ids = self._normalize_env_ids(env_ids)
        motion_index = self._motion_index_for_segment(segment)
        frame_index = self._frame_index_for_segment(segment, motion_index)
        self.command.env_motion_indices[ids] = int(motion_index)
        self.command.time_steps[ids] = int(frame_index)
        if hasattr(self.command, "motion_end_buf"):
            self.command.motion_end_buf[ids] = False
        self._reset_frontres_command_state(ids)
        self._write_command_reference_to_robot(ids)
        self._trace(
            "prepare_clean",
            segment_id=int(segment.segment_id),
            motion_index=int(motion_index),
            frame_index=int(frame_index),
            env_ids=ids.detach().cpu().tolist(),
            root_pos=self.robot.data.root_pos_w.index_select(0, ids),
            joint_pos=self.robot.data.joint_pos.index_select(0, ids),
        )
        return {"success": torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)}

    def apply_frontres_segment_index_reset(self, request: Any) -> dict[str, torch.Tensor]:
        segment_ids = getattr(request, "segment_ids")
        count = int(segment_ids.numel())
        ids = self._normalize_env_ids(range(count))
        num_envs = int(getattr(self.base_env, "num_envs", getattr(self.command, "num_envs", count)) or count)
        if count > num_envs:
            raise ValueError(f"index reset request has {count} rows but env exposes only {num_envs} envs")
        motion_ids = tuple(str(item) for item in getattr(request, "motion_ids"))
        start_frames = getattr(request, "start_frames").to(device=ids.device, dtype=torch.long).flatten()
        if len(motion_ids) != count or int(start_frames.numel()) != count:
            raise ValueError("motion_ids and start_frames must match segment_ids count")
        motion_indices = torch.tensor(
            [self._motion_index_for_key(motion_id) for motion_id in motion_ids],
            dtype=torch.long,
            device=ids.device,
        )
        frame_indices = torch.tensor(
            [
                self._frame_index_for_values(int(frame.item()), int(motion_index.item()))
                for frame, motion_index in zip(start_frames, motion_indices, strict=True)
            ],
            dtype=torch.long,
            device=ids.device,
        )
        self.command.env_motion_indices[ids] = motion_indices
        self.command.time_steps[ids] = frame_indices
        if hasattr(self.command, "motion_end_buf"):
            self.command.motion_end_buf[ids] = False
        self._reset_frontres_command_state(ids)
        self._write_command_reference_to_robot(ids)
        success = torch.ones(count, dtype=torch.bool, device=segment_ids.device)
        velocity = torch.zeros(count, dtype=torch.float32, device=segment_ids.device)
        self._trace(
            "index_reset",
            segment_ids=segment_ids,
            motion_ids=motion_ids,
            motion_indices=motion_indices,
            start_frames=start_frames,
            frame_indices=frame_indices,
            env_ids=ids.detach().cpu().tolist(),
            root_pos=self.robot.data.root_pos_w.index_select(0, ids),
            joint_pos=self.robot.data.joint_pos.index_select(0, ids),
        )
        return {"reset_success": success, "velocity_mismatch": velocity}

    def set_frontres_rollout_state(
        self, *, clean_state: FrontRESRobotRolloutState, env_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        clean_state.validate(name="clean_state")
        ids = self._normalize_env_ids(env_ids)
        if clean_state.batch_size != ids.numel():
            raise ValueError(f"clean_state batch {clean_state.batch_size} does not match env_ids {ids.numel()}")
        root_state = torch.cat(
            [
                clean_state.root_pos.to(ids.device),
                clean_state.root_quat.to(ids.device),
                clean_state.root_lin_vel.to(ids.device),
                clean_state.root_ang_vel.to(ids.device),
            ],
            dim=-1,
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=ids)
        self.robot.write_joint_state_to_sim(
            clean_state.joint_pos.to(ids.device),
            clean_state.joint_vel.to(ids.device),
            env_ids=ids,
        )
        self._trace(
            "reset_clean_state",
            env_ids=ids.detach().cpu().tolist(),
            root_pos=clean_state.root_pos,
            joint_pos=clean_state.joint_pos,
        )
        return {"success": torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)}

    def apply_frontres_segment_perturbation(
        self, *, descriptor: FrontRESPerturbationDescriptor, env_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        descriptor.validate()
        ids = self._normalize_env_ids(env_ids)
        axis = torch.as_tensor(descriptor.params.get("axis", [0.0, 0.0, 0.0]), dtype=torch.float32, device=ids.device)
        if axis.numel() != 3:
            raise ValueError(f"descriptor axis must have 3 values, got {axis.numel()}")
        signed_magnitude = float(
            descriptor.params.get("signed_magnitude", descriptor.params.get("magnitude", descriptor.strength))
        )
        delta = axis.reshape(1, 3) * float(signed_magnitude)
        root_pos = self.robot.data.root_pos_w.index_select(0, ids).clone()
        root_quat = self.robot.data.root_quat_w.index_select(0, ids).clone()
        root_lin_vel = self.robot.data.root_lin_vel_w.index_select(0, ids).clone()
        root_ang_vel = self.robot.data.root_ang_vel_w.index_select(0, ids).clone()
        before_root_pos = root_pos.clone()
        root_pos = root_pos + delta.to(root_pos.device, root_pos.dtype)
        root_lin_vel = root_lin_vel + 0.1 * delta.to(root_lin_vel.device, root_lin_vel.dtype)
        root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=ids)
        self._trace(
            "apply_perturbation",
            segment_id=int(descriptor.segment_id),
            perturbation_id=int(descriptor.perturbation_id),
            strength=float(descriptor.strength),
            env_ids=ids.detach().cpu().tolist(),
            delta=delta,
            root_pos_before=before_root_pos,
            root_pos_after=root_pos,
        )
        return {"success": torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)}

    def rollout_frontres_noisy_baseline(
        self, *, segment: FrontRESSegmentIndex, descriptor: FrontRESPerturbationDescriptor, env_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.ensure_frontres_env_reset()
        ids = self._normalize_env_ids(env_ids)
        steps = self._baseline_steps(descriptor)
        fall = torch.zeros(ids.numel(), dtype=torch.float32, device=ids.device)
        rollout_len = torch.zeros(ids.numel(), dtype=torch.float32, device=ids.device)
        rewards = []
        for _ in range(steps):
            action = self._zero_action()
            step_result = self.env.step(action)
            reward, done = self._parse_step_result(step_result)
            if reward is not None:
                rewards.append(reward.index_select(0, ids).detach().float())
            if done is not None:
                fall = torch.maximum(fall, done.index_select(0, ids).detach().float())
            rollout_len += 1.0
        score = self._baseline_score(ids, rewards)
        self._trace(
            "baseline_rollout",
            segment_id=int(segment.segment_id),
            perturbation_id=int(descriptor.perturbation_id),
            steps=int(steps),
            env_ids=ids.detach().cpu().tolist(),
            score=score,
            fall=fall,
            rollout_len=rollout_len,
        )
        baseline = FrontRESNoisyBaselineResult(score=score.detach(), fall=fall.detach(), rollout_len=rollout_len.detach())
        baseline.validate(ids.numel())
        return {"score": baseline.score, "fall": baseline.fall, "rollout_len": baseline.rollout_len}

    def _resolve_motion_command(self) -> Any:
        manager = getattr(self.base_env, "command_manager", None)
        if manager is None or not hasattr(manager, "get_term"):
            raise AttributeError("Stage 1 cache requires base_env.command_manager.get_term('motion').")
        return manager.get_term("motion")

    def _resolve_robot(self) -> Any:
        scene = getattr(self.base_env, "scene", None)
        if scene is None:
            raise AttributeError("Stage 1 cache requires base_env.scene.")
        try:
            return scene[self.robot_name]
        except (KeyError, TypeError):
            pass
        if hasattr(scene, self.robot_name):
            return getattr(scene, self.robot_name)
        raise AttributeError(f"could not resolve robot {self.robot_name!r} from env scene")

    def _build_motion_path_index(self) -> dict[str, int]:
        paths = list(getattr(self.command.motion_dir_loader, "motion_paths", []))
        root = Path(self.amass_root).expanduser().resolve()
        mapping: dict[str, int] = {}
        for idx, value in enumerate(paths):
            path = Path(value).expanduser().resolve()
            mapping[str(path)] = int(idx)
            try:
                mapping[path.relative_to(root).as_posix()] = int(idx)
            except ValueError:
                pass
            mapping[path.name] = int(idx)
        if len(mapping) == 0:
            raise ValueError("motion command has no loaded motion paths")
        return mapping

    def _motion_index_for_segment(self, segment: FrontRESSegmentIndex) -> int:
        return self._motion_index_for_key(str(segment.motion_rel_path))

    def _motion_index_for_key(self, key: str) -> int:
        if key in self.motion_path_to_index:
            return self.motion_path_to_index[key]
        suffix_hits = [
            idx for path_key, idx in self.motion_path_to_index.items() if path_key.endswith(key) or key.endswith(path_key)
        ]
        if len(suffix_hits) == 1:
            return int(suffix_hits[0])
        raise KeyError(f"segment motion path {key!r} is not loaded by the motion command")

    def _frame_index_for_segment(self, segment: FrontRESSegmentIndex, motion_index: int) -> int:
        return self._frame_index_for_values(int(segment.start_frame), motion_index)

    def _frame_index_for_values(self, start_frame: int, motion_index: int) -> int:
        motion_lengths = getattr(self.command, "motion_lengths", None)
        if motion_lengths is None:
            return int(start_frame)
        max_frame = int(motion_lengths[int(motion_index)].item()) - 1
        return min(max(int(start_frame), 0), max(max_frame, 0))

    def _write_command_reference_to_robot(self, env_ids: torch.Tensor) -> None:
        body_pos = self.command._gather_by_motion_for_envs("body_pos_w", env_ids)
        body_quat = self.command._gather_by_motion_for_envs("body_quat_w", env_ids)
        body_lin = self.command._gather_by_motion_for_envs("body_lin_vel_w", env_ids)
        body_ang = self.command._gather_by_motion_for_envs("body_ang_vel_w", env_ids)
        joint_pos = self.command._gather_by_motion_for_envs("joint_pos", env_ids)
        joint_vel = self.command._gather_by_motion_for_envs("joint_vel", env_ids)
        root_pos = body_pos[:, 0] + self.base_env.scene.env_origins[env_ids]
        root_state = torch.cat([root_pos, body_quat[:, 0], body_lin[:, 0], body_ang[:, 0]], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _reset_frontres_command_state(self, env_ids: torch.Tensor) -> None:
        if hasattr(self.command, "_frontres_pos_correction"):
            self.command._frontres_pos_correction[env_ids] = 0.0
        if hasattr(self.command, "_frontres_quat_correction"):
            self.command._frontres_quat_correction[env_ids] = 0.0
            self.command._frontres_quat_correction[env_ids, 0] = 1.0
        if hasattr(self.command, "perturber") and hasattr(self.command.perturber, "reset_envs"):
            self.command.perturber.reset_envs(env_ids)

    def _baseline_steps(self, descriptor: FrontRESPerturbationDescriptor) -> int:
        if self.baseline_rollout_steps is not None:
            return max(int(self.baseline_rollout_steps), 0)
        return max(int(descriptor.duration), 0)

    def _zero_action(self) -> torch.Tensor:
        num_envs = int(getattr(self.base_env, "num_envs", getattr(self.command, "num_envs", 1)))
        num_actions = int(getattr(self.base_env, "num_actions", 0) or 0)
        if num_actions <= 0:
            action_space = getattr(self.env, "action_space", None)
            shape = getattr(action_space, "shape", None)
            if shape is None or len(shape) == 0:
                raise AttributeError("cannot infer zero action shape for Stage 1 baseline rollout")
            num_actions = int(shape[-1])
        device = torch.device(getattr(self.command, "device", "cpu"))
        return torch.zeros(num_envs, num_actions, dtype=torch.float32, device=device)

    def _parse_step_result(self, step_result: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not isinstance(step_result, tuple):
            return None, None
        if len(step_result) == 4:
            _, rewards, dones, _ = step_result
            return rewards.detach().float(), dones.detach().bool()
        if len(step_result) == 5:
            _, rewards, terminated, truncated, _ = step_result
            dones = torch.logical_or(terminated.bool(), truncated.bool())
            return rewards.detach().float(), dones
        return None, None

    def _baseline_score(self, env_ids: torch.Tensor, rewards: list[torch.Tensor]) -> torch.Tensor:
        if rewards:
            return torch.stack(rewards, dim=0).mean(dim=0).detach()
        if hasattr(self.command, "_update_metrics"):
            self.command._update_metrics()
        anchor_pos = self._metric_value("error_anchor_pos", env_ids)
        anchor_rot = self._metric_value("error_anchor_rot", env_ids)
        return -(anchor_pos + anchor_rot).detach().float()

    def _metric_value(self, name: str, env_ids: torch.Tensor) -> torch.Tensor:
        value = getattr(self.command, "metrics", {}).get(name)
        if value is None:
            return torch.zeros(env_ids.numel(), dtype=torch.float32, device=env_ids.device)
        return value.index_select(0, env_ids).detach().float()

    def _normalize_env_ids(self, env_ids: Iterable[int] | torch.Tensor) -> torch.Tensor:
        device = torch.device(getattr(self.command, "device", "cpu"))
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=device, dtype=torch.long).flatten()
        else:
            ids = torch.tensor(list(env_ids), dtype=torch.long, device=device)
        if ids.numel() == 0:
            raise ValueError("env_ids must be non-empty")
        return ids

    def _trace(self, label: str, **items: Any) -> None:
        if not self.trace:
            return
        parts = [f"[frontres_stage1_hook trace] {label}"]
        for key, value in items.items():
            parts.append(f"{key}={self._format_trace_value(value)}")
        print(" ".join(parts), flush=True)

    def _format_trace_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            t = value.detach()
            if t.numel() == 0:
                return {"shape": tuple(t.shape), "numel": 0}
            if torch.is_floating_point(t):
                finite = bool(torch.isfinite(t).all().item())
                return {
                    "shape": tuple(t.shape),
                    "device": str(t.device),
                    "finite": finite,
                    "min": float(t.min().item()),
                    "max": float(t.max().item()),
                    "mean": float(t.float().mean().item()),
                    "requires_grad": bool(t.requires_grad),
                }
            return {"shape": tuple(t.shape), "device": str(t.device), "min": int(t.min().item()), "max": int(t.max().item())}
        if isinstance(value, (list, tuple)):
            return self._format_sequence_trace(value)
        return value

    def _format_sequence_trace(self, value: list[Any] | tuple[Any, ...]) -> Any:
        count = len(value)
        if count <= self.trace_preview_count:
            return list(value)
        preview = list(value[: self.trace_preview_count])
        tail = list(value[-self.trace_preview_count :])
        result = {"count": count, "first": preview}
        if all(isinstance(item, int) for item in value):
            result.update({"last": tail, "min": min(value), "max": max(value)})
        else:
            result["last"] = tail
            result["unique_count"] = len(set(value))
        return result


def ensure_frontres_segment_index_reset_hook(
    env: Any,
    *,
    amass_root: str,
    robot_name: str = "robot",
    trace: bool = True,
) -> FrontRESStage1EnvAdapter:
    existing = getattr(env, "_frontres_segment_index_reset_adapter", None)
    if isinstance(existing, FrontRESStage1EnvAdapter):
        return existing
    adapter = FrontRESStage1EnvAdapter(env, amass_root=amass_root, robot_name=robot_name, trace=trace)
    setattr(env, "_frontres_segment_index_reset_adapter", adapter)
    setattr(env, "apply_frontres_segment_index_reset", adapter.apply_frontres_segment_index_reset)
    base_env = getattr(adapter, "base_env", None)
    if base_env is not None and base_env is not env:
        setattr(base_env, "_frontres_segment_index_reset_adapter", adapter)
        setattr(base_env, "apply_frontres_segment_index_reset", adapter.apply_frontres_segment_index_reset)
    return adapter
