from __future__ import annotations

import hashlib
import math
import numpy as np
import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .motion_perturbations import MotionPerturber


def _quat_to_rotvec_wxyz(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map wxyz unit quaternions to shortest-path rotation vectors."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    xyz_norm = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, q[..., :1].clamp(min=eps))
    scale = torch.where(xyz_norm > eps, angle / xyz_norm.clamp(min=eps), 2.0 * torch.ones_like(xyz_norm))
    return xyz * scale


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _frontres_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_rank_world_size(shard_by: str = "global") -> tuple[int, int]:
    """Best-effort (rank, world_size) for multi-GPU dataset sharding.

    Priority:
    1) torch.distributed (if initialized)
    2) environment variables (torchrun/accelerate/slurm compatible)
    """

    shard_by = str(shard_by).lower()
    if shard_by not in {"global", "local"}:
        raise ValueError(f"Invalid shard_by={shard_by!r}. Expected 'global' or 'local'.")

    # torch.distributed path
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
            return rank, world_size
    except Exception:
        pass

    # env var path
    def _get_int_env(name: str, default: int) -> int:
        val = os.getenv(name, "")
        if val == "":
            return default
        try:
            return int(val)
        except Exception:
            return default

    if shard_by == "local":
        rank = _get_int_env("LOCAL_RANK", 0)
        world_size = _get_int_env("LOCAL_WORLD_SIZE", _get_int_env("WORLD_SIZE", 1))
    else:
        rank = _get_int_env("RANK", _get_int_env("LOCAL_RANK", 0))
        world_size = _get_int_env("WORLD_SIZE", _get_int_env("LOCAL_WORLD_SIZE", 1))

    world_size = max(int(world_size), 1)
    rank = int(rank) % world_size
    return rank, world_size


def _select_motion_paths_for_rank(
    motion_paths: list[str],
    *,
    max_motions: int | None,
    shard_across_gpus: bool,
    shard_by: str,
    shard_seed: int,
    shard_strategy: str = "chunk",
) -> tuple[list[str], dict[str, int | str | bool]]:
    """Select a (possibly sharded) subset of motion_paths for the current process.

    Design goals:
    - When motions are plentiful and max_motions is small (e.g. <= num_envs), enable
      deterministic *disjoint* subsets across GPUs (when possible), to reduce duplicates.
    - Keep default behavior unchanged unless shard_across_gpus or max_motions is provided.
    """

    total = len(motion_paths)
    info: dict[str, int | str | bool] = {
        "total_motions": total,
        "selected_motions": total,
        "shard_across_gpus": bool(shard_across_gpus),
        "shard_by": str(shard_by),
        "shard_seed": int(shard_seed),
        "shard_strategy": str(shard_strategy),
        "rank": 0,
        "world_size": 1,
        "max_motions": int(max_motions) if max_motions is not None else -1,
    }

    if total == 0:
        return motion_paths, info

    if max_motions is None:
        # No selection requested.
        return motion_paths, info

    max_motions = int(max_motions)
    if max_motions <= 0:
        raise ValueError("max_motions must be a positive integer when provided.")

    rank, world_size = _get_rank_world_size(shard_by=shard_by)
    info["rank"] = rank
    info["world_size"] = world_size

    # Deterministic shuffle of paths to avoid correlated filesystem ordering.
    indices = list(range(total))
    rng = random.Random(int(shard_seed))
    rng.shuffle(indices)

    # If not sharding, just take the first max_motions after shuffle.
    if (not shard_across_gpus) or (world_size <= 1):
        selected_idx = indices[: min(max_motions, total)]
        selected = [motion_paths[i] for i in selected_idx]
        info["selected_motions"] = len(selected)
        return selected, info

    # Sharded selection:
    # If dataset is large enough, create disjoint fixed-size shards (best case).
    if total >= world_size * max_motions:
        start = rank * max_motions
        selected_idx = indices[start : start + max_motions]
        selected = [motion_paths[i] for i in selected_idx]
        info["selected_motions"] = len(selected)
        return selected, info

    # Otherwise, fall back to disjoint partitioning then (optionally) cap.
    shard_strategy = str(shard_strategy).lower()
    if shard_strategy not in {"chunk", "stride"}:
        raise ValueError(f"Invalid shard_strategy={shard_strategy!r}. Expected 'chunk' or 'stride'.")

    if shard_strategy == "stride":
        shard_idx = indices[rank::world_size]
    else:
        # chunk: contiguous chunks after shuffle
        chunk_size = int(math.ceil(total / float(world_size)))
        start = rank * chunk_size
        shard_idx = indices[start : start + chunk_size]

    if len(shard_idx) > max_motions:
        shard_idx = shard_idx[:max_motions]

    selected = [motion_paths[i] for i in shard_idx]
    info["selected_motions"] = len(selected)
    return selected, info


def _maybe_log_motion_shard_to_wandb_summary(
    shard_info: dict[str, int | str | bool], cfg: "MultiMotionCommandCfg"
) -> None:
    """Best-effort: log one-time shard info to Weights&Biases summary.

    Intended behavior:
    - If torch.distributed is initialized, gather per-rank loaded counts and write summary from rank0 only.
    - If wandb is not available or not initialized, do nothing.
    """

    if not getattr(cfg, "motion_dataset_log_wandb_summary", True):
        return

    # Import wandb lazily and safely.
    try:
        import wandb  # type: ignore

        run = getattr(wandb, "run", None)
        if run is None:
            return
    except Exception:
        return

    rank = int(shard_info.get("rank", 0)) if isinstance(shard_info.get("rank", 0), (int, float)) else 0
    world_size = (
        int(shard_info.get("world_size", 1)) if isinstance(shard_info.get("world_size", 1), (int, float)) else 1
    )
    total = (
        int(shard_info.get("total_motions", 0)) if isinstance(shard_info.get("total_motions", 0), (int, float)) else 0
    )
    loaded = (
        int(shard_info.get("selected_motions", 0))
        if isinstance(shard_info.get("selected_motions", 0), (int, float))
        else 0
    )

    # Try distributed gather for per-rank reporting.
    loaded_list: list[int] | None = None
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t = torch.tensor([loaded], dtype=torch.long, device=device)
            out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(out, t)
            loaded_list = [int(x.item()) for x in out]
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
    except Exception:
        loaded_list = None

    # Only rank0 writes summary to avoid collisions.
    if rank != 0:
        return

    # Stable keys for easy browsing.
    run.summary["motion_dataset/world_size"] = int(world_size)
    run.summary["motion_dataset/total_motions_seen_by_rank0"] = int(total)
    run.summary["motion_dataset/shard_enabled"] = bool(getattr(cfg, "motion_dataset_shard_across_gpus", False))
    run.summary["motion_dataset/shard_by"] = str(getattr(cfg, "motion_dataset_shard_by", "global"))
    run.summary["motion_dataset/shard_strategy"] = str(getattr(cfg, "motion_dataset_shard_strategy", "chunk"))
    run.summary["motion_dataset/shard_seed"] = int(getattr(cfg, "motion_dataset_shard_seed", 0))
    run.summary["motion_dataset/load_cap"] = (
        int(getattr(cfg, "motion_dataset_load_cap", -1))
        if getattr(cfg, "motion_dataset_load_cap", None) is not None
        else None
    )

    if loaded_list is not None:
        # Store as a compact string to avoid schema issues.
        run.summary["motion_dataset/loaded_motions_per_rank"] = str(loaded_list)
        run.summary["motion_dataset/loaded_motions_sum"] = int(sum(loaded_list))
        run.summary["motion_dataset/loaded_motions_min"] = int(min(loaded_list)) if len(loaded_list) > 0 else 0
        run.summary["motion_dataset/loaded_motions_max"] = int(max(loaded_list)) if len(loaded_list) > 0 else 0
    else:
        run.summary["motion_dataset/loaded_motions_rank0"] = int(loaded)


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        # FrontRES task-space anchor corrections (zeroed = identity, no correction)
        self._frontres_pos_correction = torch.zeros(self.num_envs, 3, device=self.device)
        self._frontres_quat_correction = torch.zeros(self.num_envs, 4, device=self.device)
        self._frontres_quat_correction[:, 0] = 1.0  # identity quaternion (w=1)
        self._frontres_pair_train_ids: torch.Tensor | None = None
        self._frontres_pair_candidate_ids: torch.Tensor | None = None
        self._frontres_pair_base_ids: torch.Tensor | None = None
        self._frontres_pair_clean_ids: torch.Tensor | None = None

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
                + self._env.scene.env_origins + self._frontres_pos_correction)

    @property
    def anchor_penetration_depth(self) -> torch.Tensor:
        """Conservative penetration depth for non-perturbed commands.

        The plain motion command has no degradation cache, so there is no
        FrontRES-correctable penetration artifact to expose.
        """
        return torch.zeros(self.num_envs, device=self.device)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        base_quat = self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]
        # Right-multiply: apply correction in anchor's local frame.
        # q_anchor_new = q_anchor * q_correction, so q_correction = q_anchor^{-1} * q_robot = q_rel
        # which is exactly what anchor_root_rpy_error_w computes → Stage1 target consistent.
        return quat_mul(base_quat, self._frontres_quat_correction)

    @property
    def anchor_dr_delta_pos(self) -> torch.Tensor:
        """DR-induced anchor position delta (perturbed - original). Zero when no perturber."""
        return torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def anchor_dr_delta_quat_correction(self) -> torch.Tensor:
        """The quaternion correction that undoes DR tilt (identity when no perturber).

        Computed as quat_inv(perturbed) * original → right-multiply this onto
        perturbed_quat to recover the original anchor orientation.
        Stored as (w,x,y,z).
        """
        identity = torch.zeros(self.num_envs, 4, device=self.device)
        identity[:, 0] = 1.0  # w = 1
        return identity

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.start_from_beginning:
            start_frame = max(int(self.cfg.start_frame), 0)
            start_frame = min(start_frame, max(self.motion.time_step_total - 1, 0))
            self.time_steps[env_ids] = start_frame
        else:
            self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        # Reset FrontRES anchor corrections and OU perturbation states for resampled envs
        self._frontres_pos_correction[env_ids] = 0.0
        self._frontres_quat_correction[env_ids] = 0.0
        self._frontres_quat_correction[env_ids, 0] = 1.0
        self.perturber.reset_envs(env_ids)

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    start_from_beginning: bool = False
    start_frame: int = 0

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class MultiMotionLoader:
    """Preload motion files into contiguous tensors for fast batched sampling."""

    def __init__(
        self,
        motion_dir: str,                                    # 动作序列所在的根目录路径
        body_indexes: Sequence[int],                        # 身体部位的索引序列（如 [0, 1, 2, 3]）
        device: str | torch.device = "cpu",                 # 设置计算设备
        file_glob: str = "*.npz",                           # 文件匹配模式，默认匹配所有.npz文件
        storage_device: str | torch.device | None = None,   # 存储设备
        *,                                                  # * 之后的参数必须用关键字传递，不能用位置参数
        max_motions: int | None = None,                     # 最大加载的动作数量 (用于限制内存使用)
        shard_across_gpus: bool = False,                    # 是否跨GPU分片数据 (分布式训练)
        shard_by: str = "global",                           # 分片依据："global" 或其他策略
        shard_seed: int = 0,                                # 分片的随机种子，保证可复现性
        shard_strategy: str = "chunk",                      # 分片策略: "chunk" (连续块) 或其他
        motion_groups: dict[str, list[str]] | None = None,  # 动作分组配置
    ):
        # --- 路径处理部分 ---

        # Path(): 将字符串转换为路径对象
        # .expanduser(): 展开 ~ 成绝对路径
        # .resolve(): . 和 .. 转换为绝对路径
        motion_dir_path = Path(motion_dir).expanduser().resolve()

        # assert + is_dir()判断路径是否存在
        assert motion_dir_path.is_dir(), f"Invalid directory path: {motion_dir}"

        # rglob(file_glob): 递归搜索所有子目录中匹配模式的文件
        # sorted(): 对可迭代对象进行排序, 返回列表(保证顺序一致性)
        all_motion_paths = sorted(str(path) for path in motion_dir_path.rglob(file_glob) if path.is_file())
        
        assert len(all_motion_paths) > 0, f"No motion files matched in: {motion_dir} with pattern: {file_glob}"
        
        # --- 分组感知分片逻辑 ---

        # 获取当前进程的rank (编号) 和总进程数world_size (用于分布式训练)
        # Group-aware sharding: keep every GPU seeing every group when possible.
        rank, world_size = _get_rank_world_size(shard_by=shard_by)

        def _assign_group_name(motion_path: str) -> str:
            # 嵌套函数: 为每个动作文件分配所属组名
            if motion_groups is None:
                return "default"
            
            # .items(): 返回字典的键值对 (key, value)
            for group_name, folder_patterns in motion_groups.items():
                # 字符串包含检查: pattern是否是motion_path的子串
                for pattern in folder_patterns:
                    if pattern in motion_path:
                        return group_name
            return "default"

        # 判断是否使用分组分片
        use_group_sharding = motion_groups is not None and (shard_across_gpus or max_motions is not None)
        
        if use_group_sharding:
            # --- 按组组织文件路径 ---

            # 类型注解: 显式声明变量类型
            group_to_paths: dict[str, list[str]] = {}
            for motion_path in all_motion_paths:
                group_name = _assign_group_name(motion_path)
                # 确保每个组都有一个列表，然后追加路径
                group_to_paths.setdefault(group_name, []).append(motion_path)
                # dict.setdefault(key, default): 
                #   - 如果key存在, 返回对应的value
                #   - 如果key不存在, 插入key:default并返回default

            # --- 验证配置的组是否都存在 ---

            # Validate configured groups exist in the dataset.
            missing_groups = []
            for group_name in motion_groups.keys():
                # dict.get(key, default): 安全获取, 
                # key不存在时返回default而非抛出KeyError
                if len(group_to_paths.get(group_name, [])) == 0:
                    missing_groups.append(group_name)
            if missing_groups:
                raise ValueError(
                    "No motions matched for motion_groups: "
                    f"{missing_groups}. Check motion_groups patterns or dataset layout.")
            
            # 过滤得到非空组
            nonempty_groups = [g for g, paths in group_to_paths.items() if len(paths) > 0]
            num_groups = len(nonempty_groups)

            # --- 计算每组的最大加载数量 ---

            if max_motions is not None:
                max_motions = int(max_motions)
                if max_motions < num_groups:
                    raise ValueError(
                        f"motion_dataset_load_cap={max_motions} is smaller than the number of "
                        f"non-empty groups ({num_groups}). Increase the cap to ensure every group is loaded.")
                    
                # 计算每组的路径数并求和
                total_paths = sum(len(group_to_paths[g]) for g in nonempty_groups)

                # 每组至少保证1个
                group_caps = {g: 1 for g in nonempty_groups}
                remaining = max_motions - num_groups # 剩余可分配额度
                if remaining > 0 and total_paths > 0:
                    # 按比例分配剩余额度
                    extras = {}
                    for g in nonempty_groups:
                        extras[g] = int(math.floor(remaining * len(group_to_paths[g]) / total_paths))
                    
                    used = sum(extras.values())

                    # 由于取整产生的剩余
                    leftover = remaining - used

                    # 将剩余额度逐个分配给还有空间的组
                    for g in nonempty_groups:
                        if leftover <= 0:
                            break
                        extras[g] += 1
                        leftover -= 1
                    
                    # 合并基础配额和额外配额
                    for g in nonempty_groups:
                        # min(a, b): 返回较小值, 确保不超过实际拥有的文件数
                        group_caps[g] = min(len(group_to_paths[g]), group_caps[g] + extras[g])
            else:
                group_caps = {g: None for g in nonempty_groups}

            selected_paths = []
            shard_info = {
                "total_motions": len(all_motion_paths),
                "selected_motions": 0,
                "shard_across_gpus": bool(shard_across_gpus),
                "shard_by": str(shard_by),
                "shard_seed": int(shard_seed),
                "shard_strategy": str(shard_strategy),
                "rank": int(rank),
                "world_size": int(world_size),
                "max_motions": int(max_motions) if max_motions is not None else -1,
                "group_mode": "per_group",
                "group_shards": {},}

            for idx, group_name in enumerate(nonempty_groups):
                group_paths = group_to_paths[group_name]
                group_cap = group_caps[group_name]

                # 如果组太小，禁用该组的分片 (避免某些GPU分到空数据)
                group_shard_across = shard_across_gpus
                if group_shard_across and world_size > 1 and len(group_paths) < world_size:
                    group_shard_across = False

                group_selected, group_info = _select_motion_paths_for_rank(
                    group_paths,
                    max_motions=group_cap,
                    shard_across_gpus=group_shard_across,
                    shard_by=shard_by,
                    shard_seed=int(shard_seed) + (idx + 1) * 10007, # 每组用不同种子
                    shard_strategy=shard_strategy,)
                
                shard_info["group_shards"][group_name] = group_info
                selected_paths.extend(group_selected)

            shard_info["selected_motions"] = len(selected_paths)
        else:
            selected_paths, shard_info = _select_motion_paths_for_rank(
                all_motion_paths,
                max_motions=max_motions,
                shard_across_gpus=shard_across_gpus,
                shard_by=shard_by,
                shard_seed=shard_seed,
                shard_strategy=shard_strategy,)
        
        # --- 暴露调试信息 ---

        # Expose for debugging/analysis
        self.motion_paths_all = all_motion_paths # 所有找到的路径
        self.motion_paths = selected_paths       # 当前rank实际加载的路径
        self.shard_info = shard_info             # 分片详细信息

        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device is not None else self.device

        # --- 身体索引处理 ---

        body_idx_tensor = torch.as_tensor(body_indexes, dtype=torch.long, device="cpu")
        if body_idx_tensor.ndim != 1:
            raise ValueError("body_indexes must be a 1D sequence of indices.")
        body_idx_np = body_idx_tensor.cpu().numpy()

        # --- 数据加载循环 ---

        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_list: list[torch.Tensor] = []
        body_quat_list: list[torch.Tensor] = []
        body_lin_vel_list: list[torch.Tensor] = []
        body_ang_vel_list: list[torch.Tensor] = []
        lengths: list[int] = []
        fps_list: list[float] = []

        for motion_path in self.motion_paths:
            with np.load(motion_path) as data: # with上下文管理器, 确保资源正确释放
                fps_value = float(np.asarray(data["fps"]).reshape(-1)[0])
                fps_list.append(fps_value)

                joint_pos_tensor = torch.from_numpy(np.asarray(data["joint_pos"], dtype=np.float32)).to(
                    self.storage_device)
                joint_vel_tensor = torch.from_numpy(np.asarray(data["joint_vel"], dtype=np.float32)).to(
                    self.storage_device)
                
                body_pos_tensor = torch.from_numpy(
                    np.asarray(data["body_pos_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)
                body_quat_tensor = torch.from_numpy(
                    np.asarray(data["body_quat_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)
                body_lin_vel_tensor = torch.from_numpy(
                    np.asarray(data["body_lin_vel_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)
                body_ang_vel_tensor = torch.from_numpy(
                    np.asarray(data["body_ang_vel_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)

                joint_pos_list.append(joint_pos_tensor)
                joint_vel_list.append(joint_vel_tensor)
                body_pos_list.append(body_pos_tensor)
                body_quat_list.append(body_quat_tensor)
                body_lin_vel_list.append(body_lin_vel_tensor)
                body_ang_vel_list.append(body_ang_vel_tensor)
                lengths.append(joint_pos_tensor.shape[0])

        # --- 拼接所有数据 ---

        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self.body_pos_w = torch.cat(body_pos_list, dim=0)
        self.body_quat_w = torch.cat(body_quat_list, dim=0)
        self.body_lin_vel_w = torch.cat(body_lin_vel_list, dim=0)
        self.body_ang_vel_w = torch.cat(body_ang_vel_list, dim=0)

        # --- 内存优化：固定内存 ---

        # .pin_memory(): 将CPU张量分配到页锁定内存, 加速CPU→GPU传输
        if self.storage_device.type == "cpu":
            self.joint_pos = self.joint_pos.pin_memory()
            self.joint_vel = self.joint_vel.pin_memory()
            self.body_pos_w = self.body_pos_w.pin_memory()
            self.body_quat_w = self.body_quat_w.pin_memory()
            self.body_lin_vel_w = self.body_lin_vel_w.pin_memory()
            self.body_ang_vel_w = self.body_ang_vel_w.pin_memory()

        # --- 构建索引结构 ---

        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_lengths = lengths_tensor

        # torch.cumsum(): 累积求和[a,b,c] → [a, a+b, a+b+c], 再减去自身得到起始偏移[0, a, a+b]
        self.motion_offsets = torch.cumsum(lengths_tensor, dim=0) - lengths_tensor
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        self.total_frames = int(lengths_tensor.sum().item())

        # --- 构建动作到组的映射 ---

        # Build motion-to-group mapping for multi-teacher support
        self.motion_to_group: dict[int, str] = {}

        if motion_groups is not None:
            # Map each motion to its group based on path patterns
            for motion_idx, motion_path in enumerate(self.motion_paths):
                group_assigned = False
                # Check each group's folder patterns
                for group_name, folder_patterns in motion_groups.items():
                    for pattern in folder_patterns:
                        if pattern in motion_path:
                            self.motion_to_group[motion_idx] = group_name
                            group_assigned = True
                            break
                    if group_assigned:
                        break

                # If no match, assign to default group
                if not group_assigned:
                    self.motion_to_group[motion_idx] = "default"
        else:
            # If no groups specified, all motions belong to default group
            for motion_idx in range(len(self.motion_paths)):
                self.motion_to_group[motion_idx] = "default"

        # --- 构建动作到组的映射 ---

        # Print motion group distribution
        from collections import Counter # Counter统计各元素出现次数
        group_counts = Counter(self.motion_to_group.values())
        print(f"[MultiMotionLoader] Motion group distribution:")
        for group_name in sorted(group_counts.keys()):
            count = group_counts[group_name]
            print(f"  - {group_name}: {count} motions")

    def __len__(self) -> int:
        return len(self.motion_paths)

    def motion_length(self, motion_index: int) -> int:
        return int(self.motion_lengths[motion_index].item())

    # 计算全局索引:将[动作索引, 帧索引]映射到拼接后的大张量中的位置
    def compute_global_indices(self, motion_indices: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        lengths = self.motion_lengths[motion_indices] # 获取这些动作的长度
        max_valid = torch.clamp(lengths - 1, min=0) # 最大有效帧索引 (防止越界)
        clamped = torch.minimum(frame_indices, max_valid) # 确保不越界
        offsets = self.motion_offsets[motion_indices] # 获取各动作的起始偏移
        return offsets + clamped # 全局位置 = 起始偏移 + 帧内偏移

    def gather_from_global(
        self, attr: str, global_indices: torch.Tensor, out_device: torch.device | str
    ) -> torch.Tensor:
        # 从全局张量中按索引收集数据
        # getattr(对象, 属性名): 获取对象某属性值
        source_tensor = getattr(self, attr)
        if not isinstance(out_device, torch.device):
            out_device = torch.device(out_device)
        
        # 处理设备不一致的情况
        if global_indices.device != source_tensor.device:
            local_indices = global_indices.to(source_tensor.device)
        else:
            local_indices = global_indices
        gathered = source_tensor.index_select(0, local_indices)
        if gathered.device != out_device:
            # non_blocking=True: 异步传输，不阻塞CPU (配合pin_memory使用)
            gathered = gathered.to(out_device, non_blocking=True)
        return gathered

    def gather(
        self,
        attr: str,
        motion_indices: torch.Tensor,
        frame_indices: torch.Tensor,
        out_device: torch.device | str,
    ) -> torch.Tensor:
        global_indices = self.compute_global_indices(motion_indices, frame_indices)
        return self.gather_from_global(attr, global_indices, out_device)


class MultiMotionCommand(CommandTerm):
    """Command term that supports loading and training with multiple motions from a folder.
    
    - Samples motions across files according to difficulty-based and novelty-based sampling.
    - Within each motion, time-step sampling is adaptive based on failure counts (same as single-motion logic).
    - Periodically remaps environment ids to a fresh set of motions so all motions get chances to be sampled.
    """

    cfg: "MultiMotionCommandCfg"

    def __init__(self, cfg: "MultiMotionCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        body_index_array = self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0]
        body_index_tensor = torch.as_tensor(body_index_array, dtype=torch.long)
        self.body_indexes = body_index_tensor.to(self.device)
        body_index_list = body_index_tensor.cpu().tolist()

        # get foot indices for perturbation
        self.left_foot_idx = self.cfg.body_names.index("left_ankle_roll_link")
        self.right_foot_idx = self.cfg.body_names.index("right_ankle_roll_link")

        preload_device = (
            torch.device(self.cfg.motion_preload_device)
            if self.cfg.motion_preload_device is not None
            else torch.device(self.device)
        )

        keys = ["x","y","z","roll","pitch","yaw"]
        self._pose_ranges = torch.tensor([self.cfg.pose_range.get(k,(0.,0.)) for k in keys],
                                        device=self.device, dtype=torch.float32)
        self._vel_ranges  = torch.tensor([self.cfg.velocity_range.get(k,(0.,0.)) for k in keys],
                                        device=self.device, dtype=torch.float32)

        # Optional: cap number of motions loaded per process (useful with large datasets).
        # Auto cap is only enabled when sharding is enabled and user doesn't specify a cap.
        load_cap = self.cfg.motion_dataset_load_cap
        if load_cap is None and self.cfg.motion_dataset_shard_across_gpus:
            candidates = [int(self.num_envs)]
            k_cfg = getattr(self.cfg, "max_active_motions", None)
            if k_cfg is not None:
                candidates.append(int(k_cfg))
            load_cap = int(min(candidates)) if len(candidates) > 0 else None

        self.motion_dir_loader = MultiMotionLoader(
            self.cfg.motion,
            body_index_list,
            device=self.device,
            file_glob=self.cfg.file_glob,
            storage_device=preload_device,
            max_motions=load_cap,
            shard_across_gpus=self.cfg.motion_dataset_shard_across_gpus,
            shard_by=self.cfg.motion_dataset_shard_by,
            shard_seed=self.cfg.motion_dataset_shard_seed,
            shard_strategy=self.cfg.motion_dataset_shard_strategy,
            motion_groups=self.cfg.motion_groups,
        )

        self.num_motions_total = len(self.motion_dir_loader)

        # ---- dataset shard one-time logging ----
        self._motion_dataset_shard_info = getattr(self.motion_dir_loader, "shard_info", {}) or {}
        if getattr(self.cfg, "motion_dataset_log_shard_info", False):
            print(f"[MultiMotionLoader] shard_info={self._motion_dataset_shard_info}")
        _maybe_log_motion_shard_to_wandb_summary(self._motion_dataset_shard_info, self.cfg)

        self.sim_dt = env.cfg.decimation * env.cfg.sim.dt
        self.frames_per_bin = max(1, int(round(1.0 / self.sim_dt)))

        self.motion_lengths = self.motion_dir_loader.motion_lengths.to(self.device)
        self.motion_lengths_minus_one = (self.motion_lengths - 1).clamp(min=0)
        self.motion_length_denominator = self.motion_lengths.clamp(min=1)

        # ── Jump-degree soft gate (parabola-fit vertical acceleration) ─────────
        # Fits z(t) = A*t² + B*t + C over a [-N, +K] frame window; a_z = 2A*fps².
        # jump_degree = exp(-(a_z + g)² / 2σ²) ≈ 1 only during free-flight (a_z≈-g).
        # Applied by the runner as: Δpos *= (1 - jump_degree)  [orientation left free]
        import numpy as _np
        _N, _K = 10, 10          # past / future frames in window
        _W = _N + _K + 1         # = 21 total points
        _t = _np.arange(-_N, _K + 1, dtype=_np.float64)          # frame-index units
        _X = _np.column_stack([_t ** 2, _t, _np.ones(_W)])        # (W, 3) design matrix
        _M = _np.linalg.lstsq(_X, _np.eye(_W), rcond=None)[0]    # pseudoinverse (3, W)
        self._jump_n_past   = _N
        self._jump_k_future = _K
        self._jump_w_A      = torch.tensor(_M[0], dtype=torch.float32, device=self.device)
        self._jump_sigma    = 2.0    # m/s² — Gaussian bandwidth around free-fall
        self._jump_g        = 9.81   # m/s²
        self.jump_degree    = torch.zeros(self.num_envs, device=self.device)
        # Pre-extract anchor-body z from the full body_pos_w tensor (total_frames,).
        # Avoids gathering (N_envs*W, num_bodies, 3) each step — 42× cheaper.
        _anchor_idx = self.cfg.body_names.index(self.cfg.anchor_body_name)
        _raw_body_pos = self.motion_dir_loader.body_pos_w  # (total_frames, num_bodies, 3)
        self._jump_anchor_z = _raw_body_pos[:, _anchor_idx, 2].contiguous().to(self.device)
        # ── end jump-degree init ───────────────────────────────────────────────

        self.motion_bin_counts = (self.motion_lengths // self.frames_per_bin) + 1
        self.motion_bin_counts_float = self.motion_bin_counts.to(torch.float32)
        self.max_bin_count = int(self.motion_bin_counts.max().item())
        self.bin_index_range = torch.arange(self.max_bin_count, device=self.device)
        self.motion_bin_mask = self.bin_index_range.unsqueeze(0) < self.motion_bin_counts.unsqueeze(1)
        self.motion_end_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.bin_failed_count = torch.zeros(
            self.num_motions_total, self.max_bin_count, dtype=torch.float32, device=self.device
        )
        self.current_bin_failed = torch.zeros_like(self.bin_failed_count)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Track motion groups for multi-teacher support
        self.env_motion_groups = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Create group name to index mappings
        unique_groups = set(self.motion_dir_loader.motion_to_group.values())
        self.group_name_to_idx: dict[str, int] = {}
        self.idx_to_group_name: dict[int, str] = {}
        for idx, group_name in enumerate(sorted(unique_groups)):
            self.group_name_to_idx[group_name] = idx
            self.idx_to_group_name[idx] = group_name

        print(f"[MultiMotionCommand] Registered motion groups: {list(self.group_name_to_idx.keys())}")

        # Build reverse mapping: group_name -> list of motion indices
        self.group_to_motions: dict[str, list[int]] = {group_name: [] for group_name in self.group_name_to_idx.keys()}
        for motion_idx, group_name in self.motion_dir_loader.motion_to_group.items():
            self.group_to_motions[group_name].append(motion_idx)

        # Convert to tensors for efficient sampling
        self.group_to_motions_tensor: dict[str, torch.Tensor] = {}
        for group_name, motion_list in self.group_to_motions.items():
            self.group_to_motions_tensor[group_name] = torch.tensor(motion_list, dtype=torch.long, device=self.device)
            print(f"[MultiMotionCommand]   - Group '{group_name}': {len(motion_list)} motions")

        # Pre-allocate relative pose buffers
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Shared kernel for adaptive sampling
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        # Sampling cadence for motion-to-env remap
        if self.cfg.resample_motions_every_s <= 0:
            self._resample_motions_every_steps = 0
        else:
            steps = max(1, int(round(self.cfg.resample_motions_every_s / self.sim_dt)))
            self._resample_motions_every_steps = steps
        self._global_sim_step = 0

        self._remap_version = 0
        self._env_remap_version = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Metrics
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_std"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_min"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_entropy"] = torch.zeros(self.num_envs, device=self.device)

        prob_init = 1.0 / max(self.num_motions_total, 1)
        self.motion_sampling_probs = torch.full(
            (self.num_motions_total,), prob_init, dtype=torch.float32, device=self.device
        )
        self.motion_sample_counts = torch.zeros(
            self.num_motions_total, dtype=torch.float32, device=self.device
        )
        self.motion_assigned_counts = torch.zeros(
            self.num_motions_total, dtype=torch.float32, device=self.device
        )
        self.motion_fail_counts = torch.zeros(
            self.num_motions_total, dtype=torch.float32, device=self.device
        )

        # instantiate motion perturber
        self.perturber = MotionPerturber(env.cfg.motion_perturbations, self.num_envs, self.device)

        # Initial assignment of motions to envs and initial resample of time
        all_envs = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._assign_motions(all_envs)
        # Do not resample here: termination manager may not be ready during managers' construction.
        # Time steps start at zero; sampling and writes happen in _update_command().

        # FrontRES task-space anchor corrections (zeroed = identity, no correction)
        self._frontres_pos_correction = torch.zeros(self.num_envs, 3, device=self.device)
        self._frontres_quat_correction = torch.zeros(self.num_envs, 4, device=self.device)
        self._frontres_quat_correction[:, 0] = 1.0  # identity quaternion (w=1)
        self._frontres_pair_train_ids: torch.Tensor | None = None
        self._frontres_pair_candidate_ids: torch.Tensor | None = None
        self._frontres_pair_base_ids: torch.Tensor | None = None
        self._frontres_pair_clean_ids: torch.Tensor | None = None
        self._init_frontres_reference_window_buffers()

        # Per-step perturbation cache: computed once in _update_command() so that all
        # properties (anchor_pos_w, anchor_quat_w, anchor_dr_delta_*) share the SAME
        # random draw and the runner can read a consistent supervised_target.
        self._cached_perturbed_pos  = torch.zeros(self.num_envs, 3, device=self.device)
        self._cached_perturbed_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._cached_perturbed_quat[:, 0] = 1.0  # identity
        # ΔSE3 supervised target: correction to UNDO current DR perturbation [Δpos(3), Δrpy(3)]
        self._dr_supervised_target  = torch.zeros(self.num_envs, 6, device=self.device)

    def set_frontres_paired_baseline(self, n_train: int) -> None:
        """Pair env i with env i+n_train for B1 FrontRES-vs-GMT comparisons."""
        n_train = int(n_train)
        n_base = int(self.num_envs - n_train)
        n_pair = max(0, min(n_train, n_base))
        if n_pair <= 0:
            self._frontres_pair_train_ids = None
            self._frontres_pair_candidate_ids = None
            self._frontres_pair_base_ids = None
            return

        train_ids = torch.arange(n_pair, device=self.device, dtype=torch.long)
        base_ids = torch.arange(n_train, n_train + n_pair, device=self.device, dtype=torch.long)
        self._frontres_pair_train_ids = train_ids
        self._frontres_pair_candidate_ids = None
        self._frontres_pair_base_ids = base_ids
        self._frontres_pair_clean_ids = None
        self._sync_frontres_pairs(sync_perturbation=False)

    def set_frontres_v015_two_role_baseline(self, *, n_repair: int, n_noisy: int) -> None:
        """Install the only v015 reset layout: Repair rows paired with Noisy rows."""

        n_repair = int(n_repair)
        n_noisy = int(n_noisy)
        if (
            n_repair <= 0
            or n_noisy <= 0
            or n_repair != n_noisy
            or n_repair + n_noisy != int(self.num_envs)
        ):
            raise ValueError(
                "v015 two-role baseline requires equal positive Repair/Noisy counts covering all env rows, "
                f"got repair={n_repair}, noisy={n_noisy}, num_envs={int(self.num_envs)}"
            )
        self._frontres_pair_train_ids = torch.arange(n_repair, device=self.device, dtype=torch.long)
        self._frontres_pair_candidate_ids = None
        self._frontres_pair_base_ids = torch.arange(n_repair, n_repair + n_noisy, device=self.device, dtype=torch.long)
        self._frontres_pair_clean_ids = None
        self._frontres_v015_two_role_layout_active = True
        self._sync_frontres_pairs(sync_perturbation=False)

    def set_frontres_quartet_baseline(self, n_projected: int, n_candidate: int, n_base: int, n_clean: int) -> None:
        """Synchronize Projected, Candidate, Noisy, and Clean FrontRES env groups.

        Layout:
            [0:n_projected)                                      Projected write
            [n_projected:n_projected+n_candidate)                Candidate/full HSL write
            [n_projected+n_candidate:...+n_base)                 Noisy no-write
            [...:...+n_clean)                                    Clean reference
        """
        n_projected = int(n_projected)
        n_candidate = int(n_candidate)
        n_base = int(n_base)
        n_clean = int(n_clean)
        n_pair = max(0, min(n_projected, n_candidate, n_base, n_clean))
        if n_pair <= 0:
            self._frontres_pair_train_ids = None
            self._frontres_pair_candidate_ids = None
            self._frontres_pair_base_ids = None
            self._frontres_pair_clean_ids = None
            return

        train_ids = torch.arange(n_pair, device=self.device, dtype=torch.long)
        candidate_start = n_projected
        base_start = n_projected + n_candidate
        clean_start = base_start + n_base
        candidate_ids = torch.arange(candidate_start, candidate_start + n_pair, device=self.device, dtype=torch.long)
        base_ids = torch.arange(base_start, base_start + n_pair, device=self.device, dtype=torch.long)
        clean_ids = torch.arange(clean_start, clean_start + n_pair, device=self.device, dtype=torch.long)
        self._frontres_pair_train_ids = train_ids
        self._frontres_pair_candidate_ids = candidate_ids
        self._frontres_pair_base_ids = base_ids
        self._frontres_pair_clean_ids = clean_ids
        self._sync_frontres_pairs(sync_perturbation=False)
        if hasattr(self.perturber, "set_baseline_envs"):
            self.perturber.set_baseline_envs(clean_ids)

    def set_frontres_triplet_baseline(self, n_train: int, n_base: int, n_clean: int) -> None:
        """Synchronize FrontRES, noisy GMT, and clean GMT env triplets.

        Layout:
            [0:n_train)                         FrontRES correction on noisy reference
            [n_train:n_train+n_base)            GMT on the same noisy reference
            [n_train+n_base:n_train+n_base+n_clean) GMT on the clean reference
        """
        n_train = int(n_train)
        n_base = int(n_base)
        n_clean = int(n_clean)
        n_pair = max(0, min(n_train, n_base, n_clean))
        if n_pair <= 0:
            self._frontres_pair_train_ids = None
            self._frontres_pair_candidate_ids = None
            self._frontres_pair_base_ids = None
            self._frontres_pair_clean_ids = None
            return

        train_ids = torch.arange(n_pair, device=self.device, dtype=torch.long)
        base_ids = torch.arange(n_train, n_train + n_pair, device=self.device, dtype=torch.long)
        clean_start = n_train + n_base
        clean_ids = torch.arange(clean_start, clean_start + n_pair, device=self.device, dtype=torch.long)
        self._frontres_pair_train_ids = train_ids
        self._frontres_pair_candidate_ids = None
        self._frontres_pair_base_ids = base_ids
        self._frontres_pair_clean_ids = clean_ids
        self._sync_frontres_pairs(sync_perturbation=False)
        if hasattr(self.perturber, "set_baseline_envs"):
            self.perturber.set_baseline_envs(clean_ids)

    def _sync_frontres_pairs(self, sync_perturbation: bool = True) -> None:
        train_ids = getattr(self, '_frontres_pair_train_ids', None)
        candidate_ids = getattr(self, '_frontres_pair_candidate_ids', None)
        base_ids  = getattr(self, '_frontres_pair_base_ids',  None)
        clean_ids = getattr(self, '_frontres_pair_clean_ids', None)
        if train_ids is None or base_ids is None:
            return

        if candidate_ids is not None:
            self.env_motion_indices[candidate_ids] = self.env_motion_indices[train_ids]
            self.env_motion_groups[candidate_ids] = self.env_motion_groups[train_ids]
            self.time_steps[candidate_ids] = self.time_steps[train_ids]
        self.env_motion_indices[base_ids] = self.env_motion_indices[train_ids]
        self.env_motion_groups[base_ids] = self.env_motion_groups[train_ids]
        self.time_steps[base_ids] = self.time_steps[train_ids]
        if clean_ids is not None:
            self.env_motion_indices[clean_ids] = self.env_motion_indices[train_ids]
            self.env_motion_groups[clean_ids] = self.env_motion_groups[train_ids]
            self.time_steps[clean_ids] = self.time_steps[train_ids]

        if sync_perturbation:
            if candidate_ids is not None:
                self._cached_perturbed_pos[candidate_ids] = self._cached_perturbed_pos[train_ids]
                self._cached_perturbed_quat[candidate_ids] = self._cached_perturbed_quat[train_ids]
                self._dr_supervised_target[candidate_ids] = self._dr_supervised_target[train_ids]
                self.perturber._z_state[candidate_ids] = self.perturber._z_state[train_ids]
                self.perturber._x_state[candidate_ids] = self.perturber._x_state[train_ids]
                self.perturber._y_state[candidate_ids] = self.perturber._y_state[train_ids]
                self.perturber._roll_state[candidate_ids] = self.perturber._roll_state[train_ids]
                self.perturber._pitch_state[candidate_ids] = self.perturber._pitch_state[train_ids]
                if hasattr(self.perturber, "_artifact_steps"):
                    self.perturber._artifact_steps[candidate_ids] = self.perturber._artifact_steps[train_ids]
                    self.perturber._artifact_duration[candidate_ids] = self.perturber._artifact_duration[train_ids]
                    self.perturber._artifact_start[candidate_ids] = self.perturber._artifact_start[train_ids]
                    self.perturber._artifact_xy[candidate_ids] = self.perturber._artifact_xy[train_ids]
                    self.perturber._artifact_yaw[candidate_ids] = self.perturber._artifact_yaw[train_ids]
                if hasattr(self.perturber, "_iid_event_steps_remaining"):
                    self.perturber._iid_event_steps_remaining[candidate_ids] = self.perturber._iid_event_steps_remaining[train_ids]
                    self.perturber._iid_event_duration[candidate_ids] = self.perturber._iid_event_duration[train_ids]
                    self.perturber._iid_event_step[candidate_ids] = self.perturber._iid_event_step[train_ids]
                    self.perturber._iid_event_start[candidate_ids] = self.perturber._iid_event_start[train_ids]
                    self.perturber._iid_event_active[candidate_ids] = self.perturber._iid_event_active[train_ids]
                    self.perturber._iid_event_xy[candidate_ids] = self.perturber._iid_event_xy[train_ids]
                    self.perturber._iid_event_z[candidate_ids] = self.perturber._iid_event_z[train_ids]
                    self.perturber._iid_event_rp[candidate_ids] = self.perturber._iid_event_rp[train_ids]
                    self.perturber._iid_event_yaw[candidate_ids] = self.perturber._iid_event_yaw[train_ids]
                if self.perturber._joint_state is not None:
                    self.perturber._joint_state[candidate_ids] = self.perturber._joint_state[train_ids]
            self._cached_perturbed_pos[base_ids] = self._cached_perturbed_pos[train_ids]
            self._cached_perturbed_quat[base_ids] = self._cached_perturbed_quat[train_ids]
            self._dr_supervised_target[base_ids] = self._dr_supervised_target[train_ids]
            self.perturber._z_state[base_ids] = self.perturber._z_state[train_ids]
            self.perturber._x_state[base_ids] = self.perturber._x_state[train_ids]
            self.perturber._y_state[base_ids] = self.perturber._y_state[train_ids]
            self.perturber._roll_state[base_ids] = self.perturber._roll_state[train_ids]
            self.perturber._pitch_state[base_ids] = self.perturber._pitch_state[train_ids]
            if hasattr(self.perturber, "_artifact_steps"):
                self.perturber._artifact_steps[base_ids] = self.perturber._artifact_steps[train_ids]
                self.perturber._artifact_duration[base_ids] = self.perturber._artifact_duration[train_ids]
                self.perturber._artifact_start[base_ids] = self.perturber._artifact_start[train_ids]
                self.perturber._artifact_xy[base_ids] = self.perturber._artifact_xy[train_ids]
                self.perturber._artifact_yaw[base_ids] = self.perturber._artifact_yaw[train_ids]
            if hasattr(self.perturber, "_iid_event_steps_remaining"):
                self.perturber._iid_event_steps_remaining[base_ids] = self.perturber._iid_event_steps_remaining[train_ids]
                self.perturber._iid_event_duration[base_ids] = self.perturber._iid_event_duration[train_ids]
                self.perturber._iid_event_step[base_ids] = self.perturber._iid_event_step[train_ids]
                self.perturber._iid_event_start[base_ids] = self.perturber._iid_event_start[train_ids]
                self.perturber._iid_event_active[base_ids] = self.perturber._iid_event_active[train_ids]
                self.perturber._iid_event_xy[base_ids] = self.perturber._iid_event_xy[train_ids]
                self.perturber._iid_event_z[base_ids] = self.perturber._iid_event_z[train_ids]
                self.perturber._iid_event_rp[base_ids] = self.perturber._iid_event_rp[train_ids]
                self.perturber._iid_event_yaw[base_ids] = self.perturber._iid_event_yaw[train_ids]
            if self.perturber._joint_state is not None:
                self.perturber._joint_state[base_ids] = self.perturber._joint_state[train_ids]
            if clean_ids is not None:
                pos_data = self._gather_by_motion("body_pos_w")
                quat_data = self._gather_by_motion("body_quat_w")
                self._cached_perturbed_pos[clean_ids] = pos_data[clean_ids, self.motion_anchor_body_index]
                self._cached_perturbed_quat[clean_ids] = quat_data[clean_ids, self.motion_anchor_body_index]
                self._dr_supervised_target[clean_ids] = 0.0
                self.perturber._z_state[clean_ids] = 0.0
                self.perturber._x_state[clean_ids] = 0.0
                self.perturber._y_state[clean_ids] = 0.0
                self.perturber._roll_state[clean_ids] = 0.0
                self.perturber._pitch_state[clean_ids] = 0.0
                if hasattr(self.perturber, "_artifact_steps"):
                    self.perturber._artifact_steps[clean_ids] = 0
                    self.perturber._artifact_duration[clean_ids] = 0
                    self.perturber._artifact_start[clean_ids] = False
                    self.perturber._artifact_xy[clean_ids] = 0.0
                    self.perturber._artifact_yaw[clean_ids] = 0.0
                if hasattr(self.perturber, "_iid_event_steps_remaining"):
                    self.perturber._iid_event_steps_remaining[clean_ids] = 0
                    self.perturber._iid_event_duration[clean_ids] = 0
                    self.perturber._iid_event_step[clean_ids] = 0
                    self.perturber._iid_event_start[clean_ids] = False
                    self.perturber._iid_event_active[clean_ids] = False
                    self.perturber._iid_event_xy[clean_ids] = 0.0
                    self.perturber._iid_event_z[clean_ids] = 0.0
                    self.perturber._iid_event_rp[clean_ids] = 0.0
                    self.perturber._iid_event_yaw[clean_ids] = 0.0
                if self.perturber._joint_state is not None:
                    self.perturber._joint_state[clean_ids] = 0.0

        if candidate_ids is not None:
            self._frontres_pos_correction[candidate_ids] = self._frontres_pos_correction[train_ids]
            self._frontres_quat_correction[candidate_ids] = self._frontres_quat_correction[train_ids]
        self._frontres_pos_correction[base_ids] = 0.0
        self._frontres_quat_correction[base_ids] = 0.0
        self._frontres_quat_correction[base_ids, 0] = 1.0
        if clean_ids is not None:
            self._frontres_pos_correction[clean_ids] = 0.0
            self._frontres_quat_correction[clean_ids] = 0.0
            self._frontres_quat_correction[clean_ids, 0] = 1.0

    def _expand_frontres_pair_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        # set_frontres_paired_baseline() is called by the runner AFTER env init,
        # so these attributes may not exist yet during the first reset().
        train_ids = getattr(self, '_frontres_pair_train_ids', None)
        candidate_ids = getattr(self, '_frontres_pair_candidate_ids', None)
        base_ids  = getattr(self, '_frontres_pair_base_ids',  None)
        clean_ids = getattr(self, '_frontres_pair_clean_ids', None)
        if train_ids is None or base_ids is None or env_ids.numel() == 0:
            return env_ids

        pair_count = train_ids.numel()
        train_mask = env_ids < pair_count
        candidate_mask = None
        if candidate_ids is not None:
            candidate_mask = (env_ids >= candidate_ids[0]) & (env_ids < candidate_ids[0] + pair_count)
        base_mask = (env_ids >= base_ids[0]) & (env_ids < base_ids[0] + pair_count)
        clean_mask = None
        if clean_ids is not None:
            clean_mask = (env_ids >= clean_ids[0]) & (env_ids < clean_ids[0] + pair_count)
        paired = [env_ids]
        if train_mask.any():
            if candidate_ids is not None:
                paired.append(env_ids[train_mask] + candidate_ids[0])
            paired.append(env_ids[train_mask] + base_ids[0])
            if clean_ids is not None:
                paired.append(env_ids[train_mask] + clean_ids[0])
        if candidate_mask is not None and candidate_mask.any():
            paired.append(env_ids[candidate_mask] - candidate_ids[0])
            paired.append(env_ids[candidate_mask] - candidate_ids[0] + base_ids[0])
            if clean_ids is not None:
                paired.append(env_ids[candidate_mask] - candidate_ids[0] + clean_ids[0])
        if base_mask.any():
            if candidate_ids is not None:
                paired.append(env_ids[base_mask] - base_ids[0] + candidate_ids[0])
            paired.append(env_ids[base_mask] - base_ids[0])
            if clean_ids is not None:
                paired.append(env_ids[base_mask] - base_ids[0] + clean_ids[0])
        if clean_mask is not None and clean_mask.any():
            paired.append(env_ids[clean_mask] - clean_ids[0])
            if candidate_ids is not None:
                paired.append(env_ids[clean_mask] - clean_ids[0] + candidate_ids[0])
            paired.append(env_ids[clean_mask] - clean_ids[0] + base_ids[0])
        return torch.unique(torch.cat(paired), sorted=False)

    def _init_frontres_reference_window_buffers(self) -> None:
        self._frontres_reference_window: torch.Tensor | None = None
        self._frontres_reference_window_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._frontres_reference_window_cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # v013 fixed-Noisy carrier.  The legacy reference window is joint-only;
        # this buffer is the command-owned carrier for q/dq plus the raw anchor
        # pose that local-rp perturbs.
        self._frontres_fixed_noisy_tape: torch.Tensor | None = None
        self._frontres_fixed_noisy_tape_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._frontres_fixed_noisy_tape_cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._frontres_fixed_noisy_tape_context_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._frontres_fixed_noisy_tape_execution_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._frontres_fixed_noisy_tape_scenario_ids: list[str | None] = [None] * self.num_envs
        self._frontres_fixed_noisy_tape_hashes: list[str | None] = [None] * self.num_envs

        # v015 local carrier.  Its three artifacts have separate authorities:
        # current root artifact -> current reference cache; q29 -> current
        # deployment-command identity plus later actor context; Clean
        # continuation -> later GMT K executor.
        self._frontres_local_scenario_current_root_artifact_t: torch.Tensor | None = None
        self._frontres_local_scenario_clean_reference_t: torch.Tensor | None = None
        self._frontres_local_scenario_intent_q29: torch.Tensor | None = None
        self._frontres_local_scenario_current_command_q29_dq29: torch.Tensor | None = None
        self._frontres_local_scenario_clean_continuation: torch.Tensor | None = None
        self._frontres_local_scenario_expected_support: torch.Tensor | None = None
        self._frontres_local_scenario_expected_support_envelope: torch.Tensor | None = None
        self._frontres_local_scenario_horizon_k = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._frontres_local_scenario_continuation_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._frontres_local_scenario_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._frontres_local_scenario_current_frame_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Step 2B owns this explicit, candidate-only cursor.  It is inactive
        # during the current Noisy actor frame and never advances through the
        # generic command update path; the K collector advances it only after
        # the one repair action has been executed.
        self._frontres_local_scenario_k_execution_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._frontres_local_scenario_k_execution_cursor = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._frontres_local_scenario_ids: list[str | None] = [None] * self.num_envs
        self._frontres_local_scenario_hashes: list[str | None] = [None] * self.num_envs
        self._frontres_local_scenario_x_t_identities: list[str | None] = [None] * self.num_envs
        self._frontres_local_scenario_roles: list[str | None] = [None] * self.num_envs
        self._frontres_local_scenario_provenance: list[dict[str, object] | None] = [None] * self.num_envs
        self._frontres_local_scenario_execution_mode = "repair_attempts"

        # v015 deployment-composition carrier. This is independent of the
        # local Segment carrier: it owns one immutable .npz q29/dq29 sequence
        # plus a transaction-wide read cursor, and has no Clean continuation.
        self._frontres_v015_deployment_sequence_q29: torch.Tensor | None = None
        self._frontres_v015_deployment_sequence_dq29: torch.Tensor | None = None
        self._frontres_v015_deployment_sequence_body_pos_w: torch.Tensor | None = None
        self._frontres_v015_deployment_sequence_body_quat_w: torch.Tensor | None = None
        self._frontres_v015_deployment_sequence_body_lin_vel_w: torch.Tensor | None = None
        self._frontres_v015_deployment_sequence_body_ang_vel_w: torch.Tensor | None = None
        self._frontres_v015_deployment_sequence_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._frontres_v015_deployment_sequence_cursor = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._frontres_v015_deployment_sequence_frame_count = 0
        self._frontres_v015_deployment_sequence_future_offsets: tuple[int, ...] = ()
        self._frontres_v015_deployment_sequence_reference_path: str | None = None
        self._frontres_v015_deployment_sequence_reference_stream_id: str | None = None
        self._frontres_v015_deployment_sequence_reference_file_hash: str | None = None
        self._frontres_v015_deployment_sequence_corruption_id: str | None = None
        self._frontres_v015_deployment_sequence_protocol_hash: str | None = None
        self._frontres_v015_deployment_sequence_corruption_family: str | None = None
        self._frontres_v015_deployment_sequence_temporal_mode: str | None = None
        self._frontres_v015_deployment_sequence_evaluation_kind: str | None = None

    def set_frontres_v015_deployment_sequence(self, request: object) -> None:
        """Install one E-FI-28 deployment request as an immutable q29/dq29 carrier.

        Status: Step 5B-S2A command owner only. The sequence is not connected
        to the actor, GMT, command clock, metrics, or training state here.
        """

        # B1: 校验并安装 immutable deployment arrays/identity, 初始化 command-owned cursor.

        validate = getattr(request, "validate", None)
        if not callable(validate):
            raise TypeError("v015 deployment sequence requires a validated E-FI-28 request")
        validate()
        if bool(self._frontres_v015_deployment_sequence_active.any()):
            raise RuntimeError("v015 deployment sequence is already active and cannot be reinstalled")
        if bool(self._frontres_local_scenario_active.any()):
            raise RuntimeError("v015 deployment sequence cannot mix with a local scenario")
        if bool(self._frontres_fixed_noisy_tape_context_active.any()):
            raise RuntimeError("v015 deployment sequence cannot mix with a legacy fixed Noisy tape")
        if bool(self._frontres_reference_window_active.any()):
            raise RuntimeError("v015 deployment sequence cannot mix with a legacy reference window")

        reference_path = Path(str(getattr(request, "reference_path", ""))).expanduser().resolve(strict=True)
        requested_path = str(getattr(request, "reference_path", ""))
        if str(reference_path) != requested_path or reference_path.suffix.lower() != ".npz":
            raise ValueError("v015 deployment request must retain one absolute .npz reference path")
        reference_file_hash = str(getattr(request, "reference_file_hash", ""))
        reference_stream_id = str(getattr(request, "reference_stream_id", ""))
        reference_provenance = str(getattr(request, "reference_provenance", ""))
        evaluation_kind = str(getattr(request, "evaluation_kind", ""))
        frame_count = int(getattr(request, "frame_count", 0))
        joint_dof = int(getattr(request, "joint_dof", 0))
        future_offsets = tuple(int(value) for value in (getattr(request, "future_offsets", ()) or ()))
        protocol = getattr(request, "corruption_protocol", None)
        protocol_validate = getattr(protocol, "validate", None)
        if not callable(protocol_validate):
            raise TypeError("v015 deployment sequence requires an immutable corruption protocol")
        protocol_validate()
        corruption_id = str(getattr(protocol, "corruption_id", ""))
        protocol_hash = str(getattr(protocol, "protocol_hash", ""))
        corruption_family = str(getattr(protocol, "family", ""))
        temporal_mode = str(getattr(protocol, "temporal_mode", ""))
        if (
            reference_stream_id != f"deployment-npz:{reference_file_hash}"
            or reference_provenance != "deployment_reference_stream"
            or evaluation_kind != "deployment_composition_v015"
            or frame_count <= 0
            or joint_dof != 29
            or not future_offsets
            or tuple(sorted(set(future_offsets))) != future_offsets
            or any(value <= 0 for value in future_offsets)
            or not corruption_id
            or not corruption_family
            or temporal_mode != "persistent_full_sequence"
            or len(protocol_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in protocol_hash)
        ):
            raise ValueError("v015 deployment request identity or q29/H schema is invalid")

        hash_before = _frontres_sha256_file(reference_path)
        if hash_before != reference_file_hash:
            raise RuntimeError("v015 deployment reference file hash changed after request sealing")
        try:
            with np.load(reference_path, allow_pickle=False) as data:
                q29_np = np.asarray(data["joint_pos"])
                dq29_np = np.asarray(data["joint_vel"])
                body_pos_np = np.asarray(data["body_pos_w"])
                body_quat_np = np.asarray(data["body_quat_w"])
                body_lin_np = np.asarray(data["body_lin_vel_w"])
                body_ang_np = np.asarray(data["body_ang_vel_w"])
                q29 = torch.as_tensor(q29_np.copy(), device=self.device, dtype=torch.float32)
                dq29 = torch.as_tensor(dq29_np.copy(), device=self.device, dtype=torch.float32)
                body_pos = torch.as_tensor(body_pos_np.copy(), device=self.device, dtype=torch.float32)
                body_quat = torch.as_tensor(body_quat_np.copy(), device=self.device, dtype=torch.float32)
                body_lin = torch.as_tensor(body_lin_np.copy(), device=self.device, dtype=torch.float32)
                body_ang = torch.as_tensor(body_ang_np.copy(), device=self.device, dtype=torch.float32)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError("v015 deployment sequence cannot read sealed q29/dq29 arrays") from exc
        hash_after = _frontres_sha256_file(reference_path)
        if hash_after != reference_file_hash:
            raise RuntimeError("v015 deployment reference file hash changed during carrier installation")
        if (
            tuple(q29.shape) != (frame_count, 29)
            or tuple(dq29.shape) != (frame_count, 29)
            or tuple(body_pos.shape) != (frame_count, int(getattr(request, "body_count", 0)), 3)
            or tuple(body_quat.shape) != (frame_count, int(getattr(request, "body_count", 0)), 4)
            or tuple(body_lin.shape) != tuple(body_pos.shape)
            or tuple(body_ang.shape) != tuple(body_pos.shape)
            or not bool(torch.isfinite(q29).all().item())
            or not bool(torch.isfinite(dq29).all().item())
            or not bool(torch.isfinite(body_pos).all().item())
            or not bool(torch.isfinite(body_quat).all().item())
            or not bool(torch.isfinite(body_lin).all().item())
            or not bool(torch.isfinite(body_ang).all().item())
        ):
            raise ValueError("v015 deployment sequence must retain finite q29/dq29 and body reference arrays")
        max_offset = max(future_offsets)
        if frame_count <= max_offset:
            raise ValueError("v015 deployment sequence is too short for its H offsets")

        # Sequence values and identities are copied once. Only the explicit
        # cursor may change after installation; no reset or command callback
        # can resample or overwrite the carrier.
        self._frontres_v015_deployment_sequence_q29 = q29.detach().clone().contiguous()
        self._frontres_v015_deployment_sequence_dq29 = dq29.detach().clone().contiguous()
        self._frontres_v015_deployment_sequence_body_pos_w = body_pos.detach().clone().contiguous()
        self._frontres_v015_deployment_sequence_body_quat_w = body_quat.detach().clone().contiguous()
        self._frontres_v015_deployment_sequence_body_lin_vel_w = body_lin.detach().clone().contiguous()
        self._frontres_v015_deployment_sequence_body_ang_vel_w = body_ang.detach().clone().contiguous()
        self._frontres_v015_deployment_sequence_cursor.zero_()
        self._frontres_v015_deployment_sequence_frame_count = frame_count
        self._frontres_v015_deployment_sequence_future_offsets = future_offsets
        self._frontres_v015_deployment_sequence_reference_path = str(reference_path)
        self._frontres_v015_deployment_sequence_reference_stream_id = reference_stream_id
        self._frontres_v015_deployment_sequence_reference_file_hash = reference_file_hash
        self._frontres_v015_deployment_sequence_corruption_id = corruption_id
        self._frontres_v015_deployment_sequence_protocol_hash = protocol_hash
        self._frontres_v015_deployment_sequence_corruption_family = corruption_family
        self._frontres_v015_deployment_sequence_temporal_mode = temporal_mode
        self._frontres_v015_deployment_sequence_evaluation_kind = evaluation_kind
        self._frontres_v015_deployment_sequence_active[:] = True
        self._install_frontres_v015_deployment_current_frame()

    def _frontres_v015_deployment_current_rows(self, getter: str) -> torch.Tensor:
        """Return one row-aligned current reference field from the sealed sequence."""

        if not bool(self._frontres_v015_deployment_sequence_active.all()):
            raise RuntimeError("v015 deployment current reference requires one active sequence")
        fields = {
            "joint_pos": self._frontres_v015_deployment_sequence_q29,
            "joint_vel": self._frontres_v015_deployment_sequence_dq29,
            "body_pos_w": self._frontres_v015_deployment_sequence_body_pos_w,
            "body_quat_w": self._frontres_v015_deployment_sequence_body_quat_w,
            "body_lin_vel_w": self._frontres_v015_deployment_sequence_body_lin_vel_w,
            "body_ang_vel_w": self._frontres_v015_deployment_sequence_body_ang_vel_w,
        }
        if getter not in fields or fields[getter] is None:
            raise RuntimeError(f"v015 deployment current reference has no field {getter!r}")
        rows = fields[getter].index_select(0, self._frontres_v015_deployment_sequence_cursor)
        return rows.detach().clone()

    def _install_frontres_v015_deployment_current_frame(self) -> None:
        """Install the current deployment root artifact and clear the prior repair."""

        body_pos = self._frontres_v015_deployment_current_rows("body_pos_w")
        body_quat = self._frontres_v015_deployment_current_rows("body_quat_w")
        anchor = int(self.motion_anchor_body_index)
        if anchor < 0 or anchor >= int(body_pos.shape[1]):
            raise RuntimeError("v015 deployment sequence does not contain the command anchor body")
        self._cached_perturbed_pos.copy_(body_pos[:, anchor])
        self._cached_perturbed_quat.copy_(body_quat[:, anchor])
        self._frontres_pos_correction.zero_()
        self._frontres_quat_correction.zero_()
        self._frontres_quat_correction[:, 0] = 1.0
        self._dr_supervised_target.zero_()

    def frontres_v015_deployment_sequence_snapshot(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Return current q29/dq29 and dense H intent without advancing the cursor."""

        # B1: 按 env roles 与 cursor 读取 current/H q29 rows, 产出只读 deployment snapshot.

        if not bool(self._frontres_v015_deployment_sequence_active.all()):
            raise RuntimeError("v015 deployment sequence snapshot requires one transaction-wide active carrier")
        q29 = self._frontres_v015_deployment_sequence_q29
        dq29 = self._frontres_v015_deployment_sequence_dq29
        if q29 is None or dq29 is None:
            raise RuntimeError("active v015 deployment sequence is missing q29/dq29 data")
        if env_ids is None:
            ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if (
            int(ids.numel()) == 0
            or int(torch.unique(ids).numel()) != int(ids.numel())
            or bool((ids < 0).any())
            or bool((ids >= int(self.num_envs)).any())
        ):
            raise ValueError("v015 deployment sequence snapshot requires unique in-range command rows")

        cursors = self._frontres_v015_deployment_sequence_cursor.index_select(0, ids)
        max_offset = max(self._frontres_v015_deployment_sequence_future_offsets)
        dense_offsets = torch.arange(max_offset + 1, dtype=torch.long, device=self.device)
        frames = cursors.unsqueeze(1) + dense_offsets.unsqueeze(0)
        if bool((frames >= self._frontres_v015_deployment_sequence_frame_count).any()):
            raise RuntimeError("v015 deployment sequence cannot clamp an out-of-range H snapshot")
        intent_q29 = q29[frames]
        current_q29 = q29.index_select(0, cursors)
        current_dq29 = dq29.index_select(0, cursors)
        batch_size = int(ids.numel())
        provenance = {
            "reference_provenance": "deployment_reference_stream",
            "current_command_provenance": "deployment_q29_dq29",
            "intent_q29_provenance": "deployment_noisy_q29",
            "intent_q29_source": "deployment_npz_joint_pos",
        }

        def repeat(value: str | None) -> tuple[str, ...]:
            if value is None or not value:
                raise RuntimeError("active v015 deployment sequence is missing immutable identity metadata")
            return (value,) * batch_size

        return {
            "env_ids": ids.detach().clone(),
            "frame_indices": cursors.detach().clone(),
            "current_q29_dq29": torch.cat([current_q29, current_dq29], dim=-1).detach().clone(),
            "intent_q29": intent_q29.detach().clone(),
            "future_offsets": tuple(self._frontres_v015_deployment_sequence_future_offsets),
            "reference_paths": repeat(self._frontres_v015_deployment_sequence_reference_path),
            "reference_stream_ids": repeat(self._frontres_v015_deployment_sequence_reference_stream_id),
            "reference_file_hashes": repeat(self._frontres_v015_deployment_sequence_reference_file_hash),
            "corruption_ids": repeat(self._frontres_v015_deployment_sequence_corruption_id),
            "corruption_protocol_hashes": repeat(self._frontres_v015_deployment_sequence_protocol_hash),
            "corruption_families": repeat(self._frontres_v015_deployment_sequence_corruption_family),
            "corruption_temporal_modes": repeat(self._frontres_v015_deployment_sequence_temporal_mode),
            "evaluation_kinds": repeat(self._frontres_v015_deployment_sequence_evaluation_kind),
            "provenance": tuple(dict(provenance) for _ in range(batch_size)),
        }

    def advance_frontres_v015_deployment_sequence(self) -> None:
        """Advance every command row by one frame, rejecting before H would clamp."""

        # B1: 严格推进 command-owned frame cursor, 拒绝越过 sealed sequence boundary.

        if not bool(self._frontres_v015_deployment_sequence_active.all()):
            raise RuntimeError("v015 deployment sequence advance requires one transaction-wide active carrier")
        next_cursor = self._frontres_v015_deployment_sequence_cursor + 1
        max_offset = max(self._frontres_v015_deployment_sequence_future_offsets)
        if bool((next_cursor + max_offset >= self._frontres_v015_deployment_sequence_frame_count).any()):
            raise RuntimeError("v015 deployment sequence cannot clamp past the final valid H frame")
        self._frontres_v015_deployment_sequence_cursor.copy_(next_cursor)
        self._install_frontres_v015_deployment_current_frame()

    def clear_frontres_v015_deployment_sequence(self) -> None:
        """Close the deployment carrier as a whole without retaining mutable rows."""

        self._frontres_v015_deployment_sequence_active[:] = False
        self._frontres_v015_deployment_sequence_cursor.zero_()
        self._frontres_v015_deployment_sequence_q29 = None
        self._frontres_v015_deployment_sequence_dq29 = None
        self._frontres_v015_deployment_sequence_body_pos_w = None
        self._frontres_v015_deployment_sequence_body_quat_w = None
        self._frontres_v015_deployment_sequence_body_lin_vel_w = None
        self._frontres_v015_deployment_sequence_body_ang_vel_w = None
        self._frontres_v015_deployment_sequence_frame_count = 0
        self._frontres_v015_deployment_sequence_future_offsets = ()
        self._frontres_v015_deployment_sequence_reference_path = None
        self._frontres_v015_deployment_sequence_reference_stream_id = None
        self._frontres_v015_deployment_sequence_reference_file_hash = None
        self._frontres_v015_deployment_sequence_corruption_id = None
        self._frontres_v015_deployment_sequence_protocol_hash = None
        self._frontres_v015_deployment_sequence_corruption_family = None
        self._frontres_v015_deployment_sequence_temporal_mode = None
        self._frontres_v015_deployment_sequence_evaluation_kind = None

    def set_frontres_local_scenario(
        self,
        *,
        current_root_artifact_t: torch.Tensor,
        clean_reference_t: torch.Tensor,
        intent_q29: torch.Tensor,
        clean_continuation: torch.Tensor,
        expected_support: torch.Tensor,
        expected_support_envelope: torch.Tensor,
        horizon_k: torch.Tensor,
        continuation_lengths: torch.Tensor,
        scenario_ids: Sequence[str],
        noisy_segment_hashes: Sequence[str],
        x_t_identities: Sequence[str],
        provenance: Sequence[dict[str, object]],
        roles: Sequence[str],
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Seal each local v015 scenario across balanced Repair/Noisy attempt rows."""

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        batch_size = int(env_ids.numel())
        expected_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if (
            batch_size != int(self.num_envs)
            or int(torch.unique(env_ids).numel()) != batch_size
            or not torch.equal(torch.sort(env_ids).values, expected_ids)
        ):
            raise ValueError("v015 local scenario install must cover every command row exactly once")
        payloads = {
            "current_root_artifact_t": current_root_artifact_t,
            "clean_reference_t": clean_reference_t,
            "intent_q29": intent_q29,
            "clean_continuation": clean_continuation,
            "expected_support": expected_support,
            "expected_support_envelope": expected_support_envelope,
        }
        for name, value in payloads.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.requires_grad
                or not torch.is_floating_point(value)
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"v015 {name} must be detached finite floating-point data")
        if tuple(current_root_artifact_t.shape) != (batch_size, 7):
            raise ValueError(
                "v015 current_root_artifact_t must have shape "
                f"[{batch_size},7], got {tuple(current_root_artifact_t.shape)}"
            )
        if tuple(clean_reference_t.shape) != (batch_size, 65):
            raise ValueError(
                "v017 clean_reference_t must have shape "
                f"[{batch_size},65], got {tuple(clean_reference_t.shape)}"
            )
        if intent_q29.ndim != 3 or tuple(intent_q29.shape[:1]) != (batch_size,) or int(intent_q29.shape[1]) < 2 or int(intent_q29.shape[2]) != 29:
            raise ValueError(
                "v015 intent_q29 must have shape [B,H+1,29] with H>=1, "
                f"got {tuple(intent_q29.shape)}"
            )
        if clean_continuation.ndim != 3 or tuple(clean_continuation.shape[:1]) != (batch_size,) or int(clean_continuation.shape[1]) <= 0 or int(clean_continuation.shape[2]) != 65:
            raise ValueError(
                "v015 clean_continuation must have shape [B,K_max,65], "
                f"got {tuple(clean_continuation.shape)}"
            )
        if tuple(expected_support.shape) != (batch_size, int(clean_continuation.shape[1]), 2):
            raise ValueError("v015 expected_support must have shape [B,K_max,2]")
        if tuple(expected_support_envelope.shape) != (batch_size, int(clean_continuation.shape[1]), 6):
            raise ValueError("v015 expected_support_envelope must have shape [B,K_max,6]")
        if bool(((expected_support != 0.0) & (expected_support != 1.0)).any()):
            raise ValueError("v015 expected_support must contain binary left/right support states")
        horizon = torch.as_tensor(horizon_k, device=self.device, dtype=torch.long).flatten()
        lengths = torch.as_tensor(continuation_lengths, device=self.device, dtype=torch.long).flatten()
        if (
            int(horizon.numel()) != batch_size
            or int(lengths.numel()) != batch_size
            or bool((horizon <= 0).any())
            or not torch.equal(horizon, lengths)
            or bool((lengths > int(clean_continuation.shape[1])).any())
        ):
            raise ValueError("v015 horizon_k and continuation_lengths must be equal positive [B] values within K_max")
        metadata = (scenario_ids, noisy_segment_hashes, x_t_identities, provenance, roles)
        if any(len(value) != batch_size for value in metadata):
            raise ValueError("v015 local scenario metadata must have one row per command env")
        scenario_ids = tuple(str(value) for value in scenario_ids)
        noisy_segment_hashes = tuple(str(value) for value in noisy_segment_hashes)
        x_t_identities = tuple(str(value) for value in x_t_identities)
        roles = tuple(str(value) for value in roles)
        if (
            any(not value for value in scenario_ids)
            or any(not value for value in noisy_segment_hashes)
            or any(not value for value in x_t_identities)
            or any(value not in {"repair", "noisy"} for value in roles)
        ):
            raise ValueError("v015 local scenario requires nonempty identity metadata and only repair/noisy roles")
        provenance_rows: tuple[dict[str, object], ...] = tuple(dict(value) for value in provenance)
        for row, value in enumerate(provenance_rows):
            if (
                value.get("current_root_artifact_provenance") != "noisy_root_artifact_t"
                or value.get("clean_reference_t_provenance") != "clean_gmt_physics_only"
                or value.get("intent_q29_provenance") != "deployment_noisy_q29"
                or value.get("clean_continuation_provenance") != "clean_gmt_only"
                or value.get("expected_support_provenance") != "clean_gmt_physics_only"
                or value.get("expected_support_envelope_provenance") != "clean_gmt_physics_only"
            ):
                raise ValueError(
                    "v015 local scenario provenance must keep Noisy current/q29 and GMT-only Clean continuation, "
                    f"row {row} is invalid"
                )
            intent_source = str(value.get("intent_q29_source", "")).lower()
            if not intent_source or "root" in intent_source or "global" in intent_source or "clean" in intent_source:
                raise ValueError(
                    "v015 q29 intent provenance must exclude Clean/root/global actor input, "
                    f"row {row} source={value.get('intent_q29_source')!r}"
                )

        rows_by_scenario: dict[str, list[int]] = {}
        for row, scenario_id in enumerate(scenario_ids):
            rows_by_scenario.setdefault(scenario_id, []).append(row)
        for scenario_id, rows in rows_by_scenario.items():
            repair_rows = [row for row in rows if roles[row] == "repair"]
            noisy_rows = [row for row in rows if roles[row] == "noisy"]
            if not repair_rows or len(repair_rows) != len(noisy_rows):
                raise ValueError(
                    "each v015 local scenario must occupy balanced nonempty repair/noisy attempt rows, "
                    f"scenario={scenario_id!r}, roles={[roles[row] for row in rows]}"
                )
            anchor = rows[0]
            for row in rows[1:]:
                if (
                    noisy_segment_hashes[anchor] != noisy_segment_hashes[row]
                    or x_t_identities[anchor] != x_t_identities[row]
                    or not torch.equal(current_root_artifact_t[anchor], current_root_artifact_t[row])
                    or not torch.equal(clean_reference_t[anchor], clean_reference_t[row])
                    or not torch.equal(intent_q29[anchor], intent_q29[row])
                    or not torch.equal(clean_continuation[anchor], clean_continuation[row])
                    or not torch.equal(expected_support[anchor], expected_support[row])
                    or not torch.equal(expected_support_envelope[anchor], expected_support_envelope[row])
                    or int(horizon[anchor].item()) != int(horizon[row].item())
                    or provenance_rows[anchor] != provenance_rows[row]
                ):
                    raise ValueError(
                        "v015 Repair/Noisy attempts must reuse one immutable local scenario without mixed artifacts, "
                        f"scenario={scenario_id!r}"
                    )

        active = self._frontres_local_scenario_active
        if bool(self._frontres_fixed_noisy_tape_context_active.any()):
            raise RuntimeError("v015 local scenario cannot mix with a legacy fixed Noisy tape")
        if bool(self._frontres_reference_window_active.any()):
            raise RuntimeError("v015 local scenario cannot mix with a legacy reference window")
        value_artifact = current_root_artifact_t.detach().to(device=self.device, dtype=torch.float32).contiguous()
        value_clean_reference = clean_reference_t.detach().to(device=self.device, dtype=torch.float32).contiguous()
        value_intent = intent_q29.detach().to(device=self.device, dtype=torch.float32).contiguous()
        value_continuation = clean_continuation.detach().to(device=self.device, dtype=torch.float32).contiguous()
        value_support = expected_support.detach().to(device=self.device, dtype=torch.float32).contiguous()
        value_envelope = expected_support_envelope.detach().to(device=self.device, dtype=torch.float32).contiguous()
        if bool(active.any()):
            if not bool(active.all()):
                raise RuntimeError("v015 local scenario cannot replace a partially active command carrier")
            if bool(self._frontres_local_scenario_k_execution_active.any()):
                raise RuntimeError("v015 local scenario cannot reset while its K-step Clean continuation is executing")
            if self._frontres_local_scenario_current_command_q29_dq29 is None:
                raise RuntimeError("active v015 local scenario lost its sealed current q29+dq29 command")
            existing = self.frontres_local_scenario_snapshot(env_ids)
            if not (
                torch.equal(existing["current_root_artifact_t"], value_artifact)
                and torch.equal(existing["clean_reference_t"], value_clean_reference)
                and torch.equal(existing["intent_q29"], value_intent)
                and torch.equal(existing["clean_continuation"], value_continuation)
                and torch.equal(existing["expected_support"], value_support)
                and torch.equal(existing["expected_support_envelope"], value_envelope)
                and torch.equal(existing["horizon_k"], horizon)
                and torch.equal(existing["continuation_lengths"], lengths)
                and existing["scenario_ids"] == scenario_ids
                and existing["noisy_segment_hashes"] == noisy_segment_hashes
                and existing["x_t_identities"] == x_t_identities
                and existing["roles"] == roles
                and existing["provenance"] == provenance_rows
            ):
                raise RuntimeError("v015 local scenario reset attempted to mutate an active sealed scenario")
            self._frontres_local_scenario_current_frame_ready[env_ids] = False
            self._frontres_local_scenario_k_execution_cursor[env_ids] = -1
            return torch.ones(batch_size, dtype=torch.bool, device=self.device)

        current_dq29 = self._gather_by_motion("joint_vel")
        if (
            tuple(current_dq29.shape) != (self.num_envs, 29)
            or not bool(torch.isfinite(current_dq29).all())
        ):
            raise RuntimeError(
                "v015 local scenario install requires finite selected deployment dq29 "
                "before sealing the current q29+dq29 command"
            )
        value_current_command = torch.cat(
            [
                value_intent[:, 0],
                current_dq29.index_select(0, env_ids).to(device=self.device, dtype=torch.float32),
            ],
            dim=-1,
        ).contiguous()

        self._frontres_local_scenario_current_root_artifact_t = torch.empty(
            self.num_envs, 7, dtype=torch.float32, device=self.device
        )
        self._frontres_local_scenario_clean_reference_t = torch.empty(
            self.num_envs, 65, dtype=torch.float32, device=self.device
        )
        self._frontres_local_scenario_intent_q29 = torch.empty(
            self.num_envs, int(value_intent.shape[1]), 29, dtype=torch.float32, device=self.device
        )
        self._frontres_local_scenario_clean_continuation = torch.empty(
            self.num_envs, int(value_continuation.shape[1]), 65, dtype=torch.float32, device=self.device
        )
        self._frontres_local_scenario_expected_support = torch.empty(
            self.num_envs, int(value_support.shape[1]), 2, dtype=torch.float32, device=self.device
        )
        self._frontres_local_scenario_expected_support_envelope = torch.empty(
            self.num_envs, int(value_envelope.shape[1]), 6, dtype=torch.float32, device=self.device
        )
        self._frontres_local_scenario_current_root_artifact_t[env_ids] = value_artifact.clone()
        self._frontres_local_scenario_clean_reference_t[env_ids] = value_clean_reference.clone()
        self._frontres_local_scenario_intent_q29[env_ids] = value_intent.clone()
        self._frontres_local_scenario_current_command_q29_dq29 = torch.empty(
            self.num_envs, 58, dtype=torch.float32, device=self.device
        )
        self._frontres_local_scenario_current_command_q29_dq29[env_ids] = value_current_command.detach().clone()
        self._frontres_local_scenario_clean_continuation[env_ids] = value_continuation.clone()
        self._frontres_local_scenario_expected_support[env_ids] = value_support.clone()
        self._frontres_local_scenario_expected_support_envelope[env_ids] = value_envelope.clone()
        self._frontres_local_scenario_horizon_k[env_ids] = horizon
        self._frontres_local_scenario_continuation_lengths[env_ids] = lengths
        self._frontres_local_scenario_active[env_ids] = True
        self._frontres_local_scenario_current_frame_ready[env_ids] = False
        self._frontres_local_scenario_k_execution_active[env_ids] = False
        self._frontres_local_scenario_k_execution_cursor[env_ids] = -1
        for row, env_id in enumerate(env_ids.detach().cpu().tolist()):
            self._frontres_local_scenario_ids[int(env_id)] = scenario_ids[row]
            self._frontres_local_scenario_hashes[int(env_id)] = noisy_segment_hashes[row]
            self._frontres_local_scenario_x_t_identities[int(env_id)] = x_t_identities[row]
            self._frontres_local_scenario_roles[int(env_id)] = roles[row]
            self._frontres_local_scenario_provenance[int(env_id)] = dict(provenance_rows[row])
        return torch.ones(batch_size, dtype=torch.bool, device=self.device)

    def clear_frontres_local_scenario(self, env_ids: torch.Tensor | None = None) -> None:
        """Close a local carrier only as a whole transaction, never row by row."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        expected_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if int(ids.numel()) == 0:
            return
        if int(ids.numel()) != int(self.num_envs) or not torch.equal(torch.sort(ids).values, expected_ids):
            raise ValueError("v015 local scenario close must cover every command row")
        self._frontres_local_scenario_active[ids] = False
        self._frontres_local_scenario_current_frame_ready[ids] = False
        self._frontres_local_scenario_k_execution_active[ids] = False
        self._frontres_local_scenario_k_execution_cursor[ids] = -1
        self._frontres_local_scenario_horizon_k[ids] = 0
        self._frontres_local_scenario_continuation_lengths[ids] = 0
        self._frontres_local_scenario_current_command_q29_dq29 = None
        self._frontres_local_scenario_execution_mode = "repair_attempts"
        for env_id in ids.detach().cpu().tolist():
            self._frontres_local_scenario_ids[int(env_id)] = None
            self._frontres_local_scenario_hashes[int(env_id)] = None
            self._frontres_local_scenario_x_t_identities[int(env_id)] = None
            self._frontres_local_scenario_roles[int(env_id)] = None
            self._frontres_local_scenario_provenance[int(env_id)] = None

    def frontres_local_scenario_snapshot(self, env_ids: torch.Tensor) -> dict[str, object]:
        """Return cloned local-carrier evidence; this is not an actor or GMT route."""

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if (
            int(ids.numel()) == 0
            or not bool(self._frontres_local_scenario_active[ids].all())
            or self._frontres_local_scenario_current_root_artifact_t is None
            or self._frontres_local_scenario_clean_reference_t is None
            or self._frontres_local_scenario_intent_q29 is None
            or self._frontres_local_scenario_clean_continuation is None
            or self._frontres_local_scenario_expected_support is None
            or self._frontres_local_scenario_expected_support_envelope is None
        ):
            raise RuntimeError("v015 local scenario snapshot requires active command rows")
        scenario_ids = tuple(self._frontres_local_scenario_ids[int(env_id)] for env_id in ids.detach().cpu().tolist())
        hashes = tuple(self._frontres_local_scenario_hashes[int(env_id)] for env_id in ids.detach().cpu().tolist())
        x_t_identities = tuple(
            self._frontres_local_scenario_x_t_identities[int(env_id)] for env_id in ids.detach().cpu().tolist()
        )
        roles = tuple(self._frontres_local_scenario_roles[int(env_id)] for env_id in ids.detach().cpu().tolist())
        provenance = tuple(self._frontres_local_scenario_provenance[int(env_id)] for env_id in ids.detach().cpu().tolist())
        if any(value is None for value in scenario_ids + hashes + x_t_identities + roles + provenance):
            raise RuntimeError("active v015 local scenario is missing identity or provenance metadata")
        return {
            "current_root_artifact_t": self._frontres_local_scenario_current_root_artifact_t.index_select(0, ids).detach().clone(),
            "clean_reference_t": self._frontres_local_scenario_clean_reference_t.index_select(0, ids).detach().clone(),
            "intent_q29": self._frontres_local_scenario_intent_q29.index_select(0, ids).detach().clone(),
            "clean_continuation": self._frontres_local_scenario_clean_continuation.index_select(0, ids).detach().clone(),
            "expected_support": self._frontres_local_scenario_expected_support.index_select(0, ids).detach().clone(),
            "expected_support_envelope": self._frontres_local_scenario_expected_support_envelope.index_select(0, ids).detach().clone(),
            "horizon_k": self._frontres_local_scenario_horizon_k.index_select(0, ids).detach().clone(),
            "continuation_lengths": self._frontres_local_scenario_continuation_lengths.index_select(0, ids).detach().clone(),
            "scenario_ids": tuple(str(value) for value in scenario_ids),
            "noisy_segment_hashes": tuple(str(value) for value in hashes),
            "x_t_identities": tuple(str(value) for value in x_t_identities),
            "roles": tuple(str(value) for value in roles),
            "provenance": tuple(dict(value) for value in provenance if value is not None),
        }

    def frontres_local_scenario_intent_snapshot(self) -> dict[str, object]:
        """Return a read-only role-aligned view of the deployable q29 actor carrier."""

        if (
            not bool(self._frontres_local_scenario_active.all())
            or self._frontres_local_scenario_execution_mode == "clean_baseline"
            or not bool(self._frontres_local_scenario_current_frame_ready.all())
            or bool(self._frontres_local_scenario_k_execution_active.any())
            or self._frontres_local_scenario_intent_q29 is None
            or self._frontres_local_scenario_current_command_q29_dq29 is None
        ):
            raise RuntimeError(
                "v015 actor intent snapshot requires one transaction-wide current-frame-ready local scenario "
                "before the Clean-C K executor opens"
            )
        scenario_ids = tuple(self._frontres_local_scenario_ids)
        hashes = tuple(self._frontres_local_scenario_hashes)
        x_t_identities = tuple(self._frontres_local_scenario_x_t_identities)
        roles = tuple(self._frontres_local_scenario_roles)
        provenance = tuple(self._frontres_local_scenario_provenance)
        if any(value is None for value in scenario_ids + hashes + x_t_identities + roles + provenance):
            raise RuntimeError("v015 actor intent snapshot requires complete role identity and provenance")
        return {
            "intent_q29": self._frontres_local_scenario_intent_q29.detach().clone(),
            "scenario_ids": tuple(str(value) for value in scenario_ids),
            "noisy_segment_hashes": tuple(str(value) for value in hashes),
            "x_t_identities": tuple(str(value) for value in x_t_identities),
            "roles": tuple(str(value) for value in roles),
            "provenance": tuple(dict(value) for value in provenance if value is not None),
        }

    def set_frontres_local_scenario_execution_mode(self, mode: str) -> None:
        """Select the current GMT-only baseline or Repair phase without changing scenario identity."""

        value = str(mode)
        if value not in {"clean_baseline", "noisy_baseline", "repair_attempts"}:
            raise ValueError(f"unknown v017 local-scenario execution mode={value!r}")
        if not bool(self._frontres_local_scenario_active.all()):
            raise RuntimeError("v017 execution mode requires one active sealed local scenario transaction")
        if bool(self._frontres_local_scenario_k_execution_active.any()):
            raise RuntimeError("v017 execution mode cannot change during K-step execution")
        self._frontres_local_scenario_execution_mode = value
        self._frontres_local_scenario_current_frame_ready[:] = False

    @property
    def frontres_local_scenario_execution_mode(self) -> str:
        return str(self._frontres_local_scenario_execution_mode)

    def begin_frontres_local_scenario_k_execution(self) -> None:
        """Open the explicit GMT-only Clean-continuation phase after the one actor action.

        This does not mutate the current Noisy root cache or the applied repair.
        The candidate collector calls ``advance_frontres_local_scenario_k_execution``
        only after its t-step environment transition, so the first C frame is
        never visible to the actor action at t.
        """

        active = self._frontres_local_scenario_active
        ready = self._frontres_local_scenario_current_frame_ready
        execution = self._frontres_local_scenario_k_execution_active
        if (
            not bool(active.all())
            or not bool(ready.all())
            or self._frontres_local_scenario_clean_continuation is None
        ):
            raise RuntimeError("v015 K-step execution requires one active, current-frame-ready local scenario")
        if bool(execution.any()):
            raise RuntimeError("v015 K-step Clean continuation is already executing")
        self._frontres_local_scenario_k_execution_cursor.fill_(-1)
        execution[:] = True

    def _frontres_local_scenario_continuation_rows(self, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only Clean C rows for the active frozen-GMT executor."""

        if horizon <= 0:
            raise ValueError("v015 Clean continuation horizon must be positive")
        active = self._frontres_local_scenario_active
        execution = self._frontres_local_scenario_k_execution_active
        continuation = self._frontres_local_scenario_clean_continuation
        if (
            not bool(active.all())
            or not bool(execution.all())
            or continuation is None
            or not bool(self._frontres_local_scenario_current_frame_ready.all())
        ):
            raise RuntimeError("v015 Clean continuation rows require the explicit frozen-GMT execution phase")
        cursor = self._frontres_local_scenario_k_execution_cursor
        if bool((cursor < 0).any()):
            raise RuntimeError("v015 Clean continuation has not advanced past the one actor action")
        offsets = torch.arange(int(horizon), device=self.device, dtype=torch.long).view(1, -1)
        requested = cursor.unsqueeze(1) + offsets
        lengths = self._frontres_local_scenario_continuation_lengths.unsqueeze(1)
        valid = requested < lengths
        frame_ids = torch.minimum(requested, torch.clamp(lengths - 1, min=0))
        rows = continuation[
            torch.arange(self.num_envs, device=self.device, dtype=torch.long).unsqueeze(1),
            frame_ids,
        ]
        return rows.detach().clone(), valid.detach().clone()

    def advance_frontres_local_scenario_k_execution(self) -> dict[str, torch.Tensor]:
        """Advance one C offset, clear repair, and expose the GMT-only 65D reference.

        No actor, perturbation owner, or sampling path is reachable here.  Rows
        past their per-scenario K remain clamped for environment shape safety but
        are explicitly invalid in the returned mask.
        """

        active = self._frontres_local_scenario_active
        execution = self._frontres_local_scenario_k_execution_active
        if not bool(active.all()) or not bool(execution.all()):
            raise RuntimeError("v015 Clean continuation advance requires every local command row in execution")
        self._frontres_local_scenario_k_execution_cursor.add_(1)
        rows, valid = self._frontres_local_scenario_continuation_rows(1)
        current = rows[:, 0]
        self._cached_perturbed_pos.copy_(current[:, 58:61])
        self._cached_perturbed_quat.copy_(current[:, 61:65])
        self._frontres_pos_correction.zero_()
        self._frontres_quat_correction.zero_()
        self._frontres_quat_correction[:, 0] = 1.0
        self._dr_supervised_target.zero_()
        return {
            "continuation": current.detach().clone(),
            "valid_mask": valid[:, 0].detach().clone(),
            "cursor": self._frontres_local_scenario_k_execution_cursor.detach().clone(),
        }

    def end_frontres_local_scenario_k_execution(self) -> None:
        """Close only the K executor so the sealed scenario can service the next M attempt.

        The artifact, q29 intent, Clean continuation, identities, and hash stay
        installed.  The following attempt must still perform its Clean x_t reset
        and reinstall the same immutable carrier before another actor action.
        """

        active = self._frontres_local_scenario_active
        execution = self._frontres_local_scenario_k_execution_active
        if not bool(active.all()) or not bool(execution.all()):
            raise RuntimeError("v015 K-step execution close requires one active execution across every command row")
        execution[:] = False
        self._frontres_local_scenario_k_execution_cursor.fill_(-1)
        self._frontres_local_scenario_current_frame_ready[:] = False

    def frontres_local_scenario_k_execution_snapshot(self) -> dict[str, torch.Tensor]:
        """Return frozen-GMT C evidence without opening an actor route."""

        rows, valid = self._frontres_local_scenario_continuation_rows(1)
        cursor = self._frontres_local_scenario_k_execution_cursor
        if self._frontres_local_scenario_expected_support is None:
            raise RuntimeError("v015 K execution is missing sealed expected support evidence")
        if self._frontres_local_scenario_expected_support_envelope is None:
            raise RuntimeError("v015 K execution is missing sealed expected support envelope")
        support = self._frontres_local_scenario_expected_support[
            torch.arange(self.num_envs, device=self.device, dtype=torch.long), cursor
        ]
        envelope = self._frontres_local_scenario_expected_support_envelope[
            torch.arange(self.num_envs, device=self.device, dtype=torch.long), cursor
        ]
        return {
            "continuation": rows[:, 0].detach().clone(),
            "valid_mask": valid[:, 0].detach().clone(),
            "cursor": self._frontres_local_scenario_k_execution_cursor.detach().clone(),
            "expected_support": support.detach().clone(),
            "expected_support_envelope": envelope.detach().clone(),
        }

    def _frontres_fixed_noisy_tape_feature_dim(self) -> int:
        dof = int(self.motion_dir_loader.joint_pos.shape[-1])
        return 2 * dof + 7

    def _extract_frontres_noisy_intent_q29_rows(
        self,
        motion_indices: torch.Tensor,
        start_frames: torch.Tensor,
        intent_horizon: int,
    ) -> torch.Tensor:
        """Read dense deployment q29 windows for row-aligned motion/frame pairs."""

        motion_ids = torch.as_tensor(motion_indices, device=self.device, dtype=torch.long).flatten()
        frames_t = torch.as_tensor(start_frames, device=self.device, dtype=torch.long).flatten()
        horizon = int(intent_horizon)
        if int(motion_ids.numel()) == 0 or tuple(motion_ids.shape) != tuple(frames_t.shape):
            raise ValueError("q29 intent rows require nonempty aligned motion/frame ids")
        if bool((motion_ids < 0).any()) or bool((motion_ids >= int(self.motion_lengths_minus_one.numel())).any()):
            raise ValueError("q29 intent motion index is outside the loaded motion range")
        if bool((frames_t < 0).any()) or horizon <= 0:
            raise ValueError("q29 intent start frames must be nonnegative and H must be positive")
        dof = int(self.motion_dir_loader.joint_pos.shape[-1])
        if dof != 29:
            raise RuntimeError(
                "v015 Future Motion Context requires exactly q29 deployment intent; "
                f"command motion carrier has {dof} DoF"
            )
        max_frames = self.motion_lengths_minus_one.index_select(0, motion_ids)
        invalid = frames_t + horizon > max_frames
        if bool(invalid.any()):
            row = int(torch.nonzero(invalid, as_tuple=False)[0].item())
            raise ValueError(
                "q29 intent window cannot clamp future deployment frames: "
                f"row={row}, start={int(frames_t[row].item())}, H={horizon}, "
                f"max_frame={int(max_frames[row].item())}"
            )

        # B1: 将每个 role row 展开为稠密 t:t+H frame identity, 产出对齐帧索引.
        offsets = torch.arange(horizon + 1, dtype=torch.long, device=self.device)
        dense_frames = frames_t.unsqueeze(1) + offsets.unsqueeze(0)
        dense_motions = motion_ids.unsqueeze(1).expand_as(dense_frames)

        # B2: 复用 deployment joint_pos owner, 产出不含 root/global 与 Clean 字段的 q29 intent.
        with torch.no_grad():
            intent = self.motion_dir_loader.gather(
                "joint_pos",
                dense_motions.reshape(-1),
                dense_frames.reshape(-1),
                out_device=self.device,
            ).reshape(int(motion_ids.numel()), horizon + 1, 29)
        if (
            intent.requires_grad
            or not torch.is_floating_point(intent)
            or not bool(torch.isfinite(intent).all().item())
        ):
            raise RuntimeError("Noisy q29 extractor must return detached finite [B,H+1,29] deployment intent")
        return intent.detach().to(device=self.device, dtype=torch.float32).clone().contiguous()

    def extract_frontres_noisy_intent_q29(
        self,
        *,
        motion_index: int,
        start_frame: int,
        intent_horizon: int,
    ) -> torch.Tensor:
        """从 deployment/Noisy carrier 提取 actor-only q29 future intent.

        函数名说明:
            这是 v015 Future Motion Context 的唯一 q29 extraction owner. ``Noisy``
            表示 deployment carrier 的 provenance, 不表示向 articulated joints
            注入物理扰动.

        主链路:
            上游: ``materialize_frontres_local_scenario`` 为 sealed local scenario
            指定 motion, t 和 H.
            下游: local scenario 的 ``intent_q29`` 由 actor H-window 消费.

        语义:
            返回 detached finite ``[H+1,29]`` internal-motion window. 该 owner
            只读取 deployment carrier 的 ``joint_pos``; 不读取 root/global,
            不读取 Clean reference, 不重采样或变异 perturbation.
        """

        rows = self._extract_frontres_noisy_intent_q29_rows(
            torch.tensor([int(motion_index)], dtype=torch.long, device=self.device),
            torch.tensor([int(start_frame)], dtype=torch.long, device=self.device),
            int(intent_horizon),
        )
        return rows[0].detach().clone()

    def frontres_hsl_proposal_intent_snapshot(
        self,
        future_offsets: Sequence[int],
    ) -> dict[str, object]:
        """Return an immutable Stage-1 artifact/q29 proposal context.

        Status: Stage-1 carrier only. This read-only snapshot reuses the command's
        deployment q29 owner and excludes every Segment/K/C object.
        """

        offsets = tuple(int(value) for value in future_offsets)
        if not offsets or any(value <= 0 for value in offsets) or tuple(sorted(set(offsets))) != offsets:
            raise ValueError("HSL proposal future offsets must be ordered unique positive integers")
        if (
            bool(self._frontres_local_scenario_active.any())
            or bool(self._frontres_fixed_noisy_tape_context_active.any())
            or bool(self._frontres_v015_deployment_sequence_active.any())
        ):
            raise RuntimeError("HSL proposal carrier cannot mix with Segment, fixed-tape, or deployment-eval state")

        motion_ids = self.env_motion_indices.detach().to(device=self.device, dtype=torch.long).clone()
        frame_ids = self.time_steps.detach().to(device=self.device, dtype=torch.long).clone()
        intent = self._extract_frontres_noisy_intent_q29_rows(motion_ids, frame_ids, max(offsets))
        artifact_pos = self.anchor_dr_delta_pos.detach().to(device="cpu", dtype=torch.float32).contiguous()
        artifact_quat = self.anchor_dr_delta_quat_correction.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        intent_cpu = intent.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if (
            tuple(artifact_pos.shape) != (self.num_envs, 3)
            or tuple(artifact_quat.shape) != (self.num_envs, 4)
            or not bool(torch.isfinite(artifact_pos).all().item())
            or not bool(torch.isfinite(artifact_quat).all().item())
        ):
            raise RuntimeError("HSL proposal carrier requires finite current root-artifact [B,3]+[B,4]")

        artifact_ids: list[str] = []
        context_ids: list[str] = []
        for row in range(self.num_envs):
            artifact_digest = hashlib.sha256()
            artifact_digest.update(artifact_pos[row].numpy().tobytes())
            artifact_digest.update(artifact_quat[row].numpy().tobytes())
            artifact_id = artifact_digest.hexdigest()
            context_digest = hashlib.sha256()
            context_digest.update(artifact_id.encode("ascii"))
            context_digest.update(str(int(motion_ids[row].item())).encode("ascii"))
            context_digest.update(str(int(frame_ids[row].item())).encode("ascii"))
            context_digest.update(repr(offsets).encode("ascii"))
            context_digest.update(intent_cpu[row].numpy().tobytes())
            artifact_ids.append(artifact_id)
            context_ids.append(context_digest.hexdigest())

        provenance = tuple(
            {
                "carrier_kind": "hsl_proposal",
                "current_root_artifact_provenance": "noisy_root_artifact_t",
                "intent_q29_provenance": "deployment_noisy_q29",
                "intent_q29_source": "motion_internal_q29",
            }
            for _ in range(self.num_envs)
        )
        return {
            "intent_q29": intent.detach().clone(),
            "proposal_context_ids": tuple(context_ids),
            "current_root_artifact_ids": tuple(artifact_ids),
            "motion_indices": tuple(int(value) for value in motion_ids.detach().cpu().tolist()),
            "frame_indices": tuple(int(value) for value in frame_ids.detach().cpu().tolist()),
            "future_offsets": offsets,
            "provenance": provenance,
        }

    def materialize_frontres_local_scenario(
        self,
        *,
        motion_index: int,
        start_frame: int,
        horizon_k: int,
        intent_horizon: int,
        perturbation_family: str,
        perturbation_strength: float,
    ) -> dict[str, object]:
        """Materialize one v015 local scenario without constructing a shared 65D tape.

        The only Noisy reference payload is the current root artifact.  The q29
        intent carrier is deliberately read from the deployment motion carrier
        without the joint-perturbation owner, while the full Clean continuation
        is reserved for a later GMT-only K-step executor.
        """

        motion_index = int(motion_index)
        start_frame = int(start_frame)
        horizon_k = int(horizon_k)
        intent_horizon = int(intent_horizon)
        strength = float(perturbation_strength)
        if motion_index < 0 or motion_index >= int(self.motion_lengths_minus_one.numel()):
            raise ValueError(f"motion_index={motion_index} is outside the loaded motion range")
        if start_frame < 0:
            raise ValueError(f"start_frame must be nonnegative, got {start_frame}")
        if horizon_k <= 0:
            raise ValueError(f"horizon_k must be positive, got {horizon_k}")
        if intent_horizon <= 0:
            raise ValueError(f"intent_horizon must be positive, got {intent_horizon}")
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError(f"perturbation_strength must be finite and nonnegative, got {perturbation_strength}")
        family_parts = {part.strip() for part in str(perturbation_family).split("+") if part.strip()}
        supported_families = ("planar", "yaw", "global_z", "local_rp")
        if not family_parts or not family_parts.issubset(set(supported_families)):
            raise ValueError(
                "local scenario requires one or more physical perturbation families "
                f"from {supported_families}, got {perturbation_family!r}"
            )
        dof = int(self.motion_dir_loader.joint_pos.shape[-1])
        if dof != 29:
            raise RuntimeError(
                "v015 local scenario requires exactly q29 deployment intent; "
                f"command motion carrier has {dof} DoF"
            )
        max_frame = int(self.motion_lengths_minus_one[motion_index].item())
        last_required_frame = start_frame + max(horizon_k, intent_horizon)
        if last_required_frame > max_frame:
            raise ValueError(
                "local scenario cannot clamp future intent or Clean continuation: "
                f"start={start_frame}, H={intent_horizon}, K={horizon_k}, max_frame={max_frame}"
            )
        perturber_cfg = getattr(self.perturber, "cfg", None)
        if perturber_cfg is None:
            raise RuntimeError("local scenario materialization requires MotionPerturber.cfg")
        isolated_perturber = type(self.perturber)(perturber_cfg, 1, self.device)
        set_scale = getattr(isolated_perturber, "set_dr_scale_env", None)
        set_masks = getattr(isolated_perturber, "set_family_env_masks", None)
        if not callable(set_scale) or not callable(set_masks):
            raise RuntimeError("local scenario materialization requires MotionPerturber scale and family-mask setters")
        set_scale(torch.tensor([strength], dtype=torch.float32, device=self.device))
        set_masks(
            {
                name: torch.tensor([name in family_parts], dtype=torch.bool, device=self.device)
                for name in supported_families
            }
        )

        def gather(frame: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            motion_ids = torch.tensor([motion_index], dtype=torch.long, device=self.device)
            frame_ids = torch.tensor([int(frame)], dtype=torch.long, device=self.device)
            joint_pos = self.motion_dir_loader.gather("joint_pos", motion_ids, frame_ids, out_device=self.device)
            joint_vel = self.motion_dir_loader.gather("joint_vel", motion_ids, frame_ids, out_device=self.device)
            body_pos = self.motion_dir_loader.gather("body_pos_w", motion_ids, frame_ids, out_device=self.device)
            body_quat = self.motion_dir_loader.gather("body_quat_w", motion_ids, frame_ids, out_device=self.device)
            if tuple(joint_pos.shape) != (1, 29) or tuple(joint_vel.shape) != (1, 29):
                raise RuntimeError(
                    "local scenario q29 materializer received invalid joint payloads: "
                    f"joint_pos={tuple(joint_pos.shape)}, joint_vel={tuple(joint_vel.shape)}"
                )
            return joint_pos, joint_vel, body_pos, body_quat

        with torch.no_grad():
            joint_pos_t, joint_vel_t, body_pos_t, body_quat_t = gather(start_frame)
            root_pos_t = body_pos_t[:, self.motion_anchor_body_index]
            root_quat_t = body_quat_t[:, self.motion_anchor_body_index]
            clean_reference_t = torch.cat(
                [joint_pos_t[0], joint_vel_t[0], root_pos_t[0], root_quat_t[0]], dim=0
            )
            noisy_root_pos_t = isolated_perturber.apply_perturbations(
                root_pos_t,
                body_pos_t[:, self.left_foot_idx],
                body_pos_t[:, self.right_foot_idx],
            )
            noisy_root_quat_t = isolated_perturber.apply_quat_perturbation(root_quat_t)
            current_root_artifact_t = torch.cat([noisy_root_pos_t[0], noisy_root_quat_t[0]], dim=0)

            intent_q29 = self.extract_frontres_noisy_intent_q29(
                motion_index=motion_index,
                start_frame=start_frame,
                intent_horizon=intent_horizon,
            )
            continuation_frames = [gather(start_frame + offset) for offset in range(1, horizon_k + 1)]
            clean_continuation = torch.stack(
                [torch.cat(
                        [
                            joint_pos[0],
                            joint_vel[0],
                            body_pos[:, self.motion_anchor_body_index][0],
                            body_quat[:, self.motion_anchor_body_index][0],
                        ],
                        dim=0,
                    ) for joint_pos, joint_vel, body_pos, body_quat in continuation_frames],
                dim=0,
            )
            support_height = float(getattr(getattr(self, "cfg", None), "frontres_expected_contact_height", 0.08))
            expected_support = torch.stack(
                [
                    torch.stack(
                        (body_pos[0, self.left_foot_idx, 2], body_pos[0, self.right_foot_idx, 2])
                    ) <= support_height
                    for _joint_pos, _joint_vel, body_pos, _body_quat in continuation_frames
                ],
                dim=0,
            ).to(dtype=torch.float32)
            foot_half_length = float(
                getattr(getattr(self, "cfg", None), "frontres_expected_foot_half_length", 0.10)
            )
            foot_half_width = float(
                getattr(getattr(self, "cfg", None), "frontres_expected_foot_half_width", 0.05)
            )
            if foot_half_length <= 0.0 or foot_half_width <= 0.0:
                raise RuntimeError("expected support-envelope foot extents must be positive")

            # Physics-only Clean carrier: oriented support box per K frame.
            # Layout is center_xy, cos(yaw), sin(yaw), half_x, half_y.
            envelope_rows: list[torch.Tensor] = []
            for frame_index, (_joint_pos, _joint_vel, body_pos, body_quat) in enumerate(continuation_frames):
                feet_xy = torch.stack(
                    (body_pos[0, self.left_foot_idx, :2], body_pos[0, self.right_foot_idx, :2]), dim=0
                )
                feet_quat = torch.stack(
                    (body_quat[0, self.left_foot_idx], body_quat[0, self.right_foot_idx]), dim=0
                )
                w, x, y, z = feet_quat.unbind(dim=-1)
                yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
                active = expected_support[frame_index].bool()
                if not bool(active.any()):
                    active = torch.ones_like(active)
                mean_cos = torch.cos(yaw[active]).mean()
                mean_sin = torch.sin(yaw[active]).mean()
                norm = torch.sqrt(mean_cos.square() + mean_sin.square()).clamp_min(1.0e-8)
                cos_ref, sin_ref = mean_cos / norm, mean_sin / norm
                centers = feet_xy[active]
                center_x = cos_ref * centers[:, 0] + sin_ref * centers[:, 1]
                center_y = -sin_ref * centers[:, 0] + cos_ref * centers[:, 1]
                delta_yaw = yaw[active] - torch.atan2(sin_ref, cos_ref)
                projected_half_x = (
                    torch.cos(delta_yaw).abs() * foot_half_length
                    + torch.sin(delta_yaw).abs() * foot_half_width
                )
                projected_half_y = (
                    torch.sin(delta_yaw).abs() * foot_half_length
                    + torch.cos(delta_yaw).abs() * foot_half_width
                )
                lower_x = (center_x - projected_half_x).min()
                upper_x = (center_x + projected_half_x).max()
                lower_y = (center_y - projected_half_y).min()
                upper_y = (center_y + projected_half_y).max()
                box_center_x = 0.5 * (lower_x + upper_x)
                box_center_y = 0.5 * (lower_y + upper_y)
                world_center = torch.stack(
                    (
                        cos_ref * box_center_x - sin_ref * box_center_y,
                        sin_ref * box_center_x + cos_ref * box_center_y,
                    )
                )
                envelope_rows.append(
                    torch.cat(
                        (
                            world_center,
                            torch.stack((cos_ref, sin_ref)),
                            torch.stack((0.5 * (upper_x - lower_x), 0.5 * (upper_y - lower_y))),
                        )
                    )
                )
            expected_support_envelope = torch.stack(envelope_rows, dim=0).to(dtype=torch.float32)
        if tuple(current_root_artifact_t.shape) != (7,):
            raise RuntimeError(
                "local scenario current root artifact must be [7], got "
                f"{tuple(current_root_artifact_t.shape)}"
            )
        if tuple(clean_reference_t.shape) != (65,):
            raise RuntimeError(f"local scenario clean_reference_t has invalid shape {tuple(clean_reference_t.shape)}")
        if tuple(intent_q29.shape) != (intent_horizon + 1, 29):
            raise RuntimeError(f"local scenario intent_q29 has invalid shape {tuple(intent_q29.shape)}")
        if tuple(clean_continuation.shape) != (horizon_k, 65):
            raise RuntimeError(f"local scenario clean_continuation has invalid shape {tuple(clean_continuation.shape)}")
        if tuple(expected_support.shape) != (horizon_k, 2):
            raise RuntimeError(f"local scenario expected_support has invalid shape {tuple(expected_support.shape)}")
        if tuple(expected_support_envelope.shape) != (horizon_k, 6):
            raise RuntimeError(
                "local scenario expected_support_envelope has invalid shape "
                f"{tuple(expected_support_envelope.shape)}"
            )
        return {
            "current_root_artifact_t": current_root_artifact_t.detach().to(dtype=torch.float32).contiguous(),
            "clean_reference_t": clean_reference_t.detach().to(dtype=torch.float32).contiguous(),
            "intent_q29": intent_q29.detach().to(dtype=torch.float32).contiguous(),
            "clean_continuation": clean_continuation.detach().to(dtype=torch.float32).contiguous(),
            "expected_support": expected_support.detach().to(dtype=torch.float32).contiguous(),
            "expected_support_envelope": expected_support_envelope.detach().to(dtype=torch.float32).contiguous(),
            "provenance": {
                "materializer_owner": "MultiMotionCommand",
                "current_root_artifact_provenance": "noisy_root_artifact_t",
                "clean_reference_t_provenance": "clean_gmt_physics_only",
                "intent_q29_provenance": "deployment_noisy_q29",
                "intent_q29_source": "motion_internal_q29",
                "clean_continuation_provenance": "clean_gmt_only",
                "expected_support_provenance": "clean_gmt_physics_only",
                "expected_support_envelope_provenance": "clean_gmt_physics_only",
                "expected_support_envelope_schema": "clean-foot-pose-oriented-box-v1",
                "expected_contact_height": support_height,
                "motion_index": motion_index,
                "start_frame": start_frame,
                "intent_horizon": intent_horizon,
                "horizon_k": horizon_k,
                "perturbation_family": str(perturbation_family),
                "perturbation_strength": strength,
            },
        }

    def materialize_frontres_fixed_noisy_tape(
        self,
        *,
        motion_index: int,
        start_frame: int,
        frame_count: int,
        perturbation_family: str,
        perturbation_strength: float,
    ) -> torch.Tensor:
        """Materialize one sealed ``[L, q+dq+anchor_pos+anchor_quat]`` scenario tape.

        This is selection-time reference construction, not a simulator reset.  It
        uses an isolated perturbation state so it cannot mutate the live command
        or resample an already-installed scenario during retry/reset.
        """

        motion_index = int(motion_index)
        start_frame = int(start_frame)
        frame_count = int(frame_count)
        strength = float(perturbation_strength)
        if motion_index < 0 or motion_index >= int(self.motion_lengths_minus_one.numel()):
            raise ValueError(f"motion_index={motion_index} is outside the loaded motion range")
        if start_frame < 0:
            raise ValueError(f"start_frame must be nonnegative, got {start_frame}")
        if frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {frame_count}")
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError(f"perturbation_strength must be finite and nonnegative, got {perturbation_strength}")
        family_parts = {part.strip() for part in str(perturbation_family).split("+") if part.strip()}
        supported_families = ("planar", "yaw", "global_z", "local_rp")
        if not family_parts or not family_parts.issubset(set(supported_families)):
            raise ValueError(
                "fixed Noisy tape requires one or more physical perturbation families "
                f"from {supported_families}, got {perturbation_family!r}"
            )
        perturber_cfg = getattr(self.perturber, "cfg", None)
        if perturber_cfg is None:
            raise RuntimeError("fixed Noisy tape materialization requires MotionPerturber.cfg")
        isolated_perturber = type(self.perturber)(perturber_cfg, 1, self.device)
        set_scale = getattr(isolated_perturber, "set_dr_scale_env", None)
        set_masks = getattr(isolated_perturber, "set_family_env_masks", None)
        if not callable(set_scale) or not callable(set_masks):
            raise RuntimeError("fixed Noisy tape materialization requires MotionPerturber scale and family-mask setters")
        set_scale(torch.tensor([strength], dtype=torch.float32, device=self.device))
        set_masks(
            {
                name: torch.tensor([name in family_parts], dtype=torch.bool, device=self.device)
                for name in supported_families
            }
        )

        max_frame = int(self.motion_lengths_minus_one[motion_index].item())
        dof = int(self.motion_dir_loader.joint_pos.shape[-1])
        rows: list[torch.Tensor] = []
        with torch.no_grad():
            for offset in range(frame_count):
                frame = min(start_frame + offset, max_frame)
                motion_ids = torch.tensor([motion_index], dtype=torch.long, device=self.device)
                frame_ids = torch.tensor([frame], dtype=torch.long, device=self.device)
                joint_pos = self.motion_dir_loader.gather("joint_pos", motion_ids, frame_ids, out_device=self.device)
                joint_vel = self.motion_dir_loader.gather("joint_vel", motion_ids, frame_ids, out_device=self.device)
                body_pos = self.motion_dir_loader.gather("body_pos_w", motion_ids, frame_ids, out_device=self.device)
                body_quat = self.motion_dir_loader.gather("body_quat_w", motion_ids, frame_ids, out_device=self.device)
                root_pos = body_pos[:, self.motion_anchor_body_index]
                root_quat = body_quat[:, self.motion_anchor_body_index]
                noisy_pos = isolated_perturber.apply_perturbations(
                    root_pos,
                    body_pos[:, self.left_foot_idx],
                    body_pos[:, self.right_foot_idx],
                )
                noisy_quat = isolated_perturber.apply_quat_perturbation(root_quat)
                noisy_joint_pos = isolated_perturber.apply_joint_perturbation(joint_pos)
                row = torch.cat(
                    [
                        noisy_joint_pos.reshape(1, dof),
                        joint_vel.reshape(1, dof),
                        noisy_pos.reshape(1, 3),
                        noisy_quat.reshape(1, 4),
                    ],
                    dim=-1,
                )
                rows.append(row[0])
        return torch.stack(rows, dim=0).detach().to(dtype=torch.float32).contiguous()

    def set_frontres_fixed_noisy_tape(
        self,
        tape: torch.Tensor,
        *,
        tape_lengths: torch.Tensor,
        scenario_ids: Sequence[str],
        noisy_segment_hashes: Sequence[str],
        execution_mask: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Install immutable v013 Noisy reference tape rows without sampling.

        Every installed row exposes the tape to actor H context. ``execution_mask``
        separately controls whether GMT reads the Noisy current/K command; paired
        Clean evidence rows therefore never become actor-visible Clean future.
        """

        if bool(self._frontres_local_scenario_active.any()):
            raise RuntimeError("legacy fixed Noisy tape cannot mix with an active v015 local scenario")
        if not isinstance(tape, torch.Tensor) or tape.ndim != 3:
            raise ValueError(f"fixed Noisy tape must have shape [B, L, F], got {getattr(tape, 'shape', None)}")
        if tape.requires_grad or not torch.is_floating_point(tape) or not bool(torch.isfinite(tape).all().item()):
            raise ValueError("fixed Noisy tape must be detached, finite floating-point data")
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        batch_size = int(env_ids.numel())
        if int(tape.shape[0]) != batch_size or int(tape.shape[1]) <= 0:
            raise ValueError("fixed Noisy tape batch/length must match nonempty env_ids")
        expected_dim = self._frontres_fixed_noisy_tape_feature_dim()
        if int(tape.shape[-1]) != expected_dim:
            raise ValueError(
                f"fixed Noisy tape feature dim must be {expected_dim} [q,dq,anchor_pos,anchor_quat], "
                f"got {int(tape.shape[-1])}"
            )
        lengths = torch.as_tensor(tape_lengths, device=self.device, dtype=torch.long).flatten()
        if int(lengths.numel()) != batch_size or bool((lengths <= 0).any()) or bool((lengths > int(tape.shape[1])).any()):
            raise ValueError("fixed Noisy tape lengths must be [B] with values in [1, L]")
        execute = torch.as_tensor(execution_mask, device=self.device, dtype=torch.bool).flatten()
        if int(execute.numel()) != batch_size:
            raise ValueError("fixed Noisy tape execution_mask must have B rows")
        if len(scenario_ids) != batch_size or len(noisy_segment_hashes) != batch_size:
            raise ValueError("fixed Noisy tape identity metadata must have B rows")
        if any(not str(value) for value in scenario_ids) or any(not str(value) for value in noisy_segment_hashes):
            raise ValueError("fixed Noisy tape scenario/hash metadata must be nonempty")

        value = tape.detach().to(device=self.device, dtype=torch.float32).contiguous()
        if self._frontres_fixed_noisy_tape is None:
            self._frontres_fixed_noisy_tape = torch.zeros(
                self.num_envs, int(value.shape[1]), int(value.shape[2]), dtype=value.dtype, device=self.device
            )
        elif tuple(self._frontres_fixed_noisy_tape.shape[1:]) != tuple(value.shape[1:]):
            outside = self._frontres_fixed_noisy_tape_context_active.clone()
            outside[env_ids] = False
            if bool(outside.any()):
                raise RuntimeError("cannot resize fixed Noisy tape while another active scenario is installed")
            self._frontres_fixed_noisy_tape = torch.zeros(
                self.num_envs, int(value.shape[1]), int(value.shape[2]), dtype=value.dtype, device=self.device
            )

        self._frontres_fixed_noisy_tape[env_ids] = value
        self._frontres_fixed_noisy_tape_lengths[env_ids] = lengths
        self._frontres_fixed_noisy_tape_cursor[env_ids] = 0
        self._frontres_fixed_noisy_tape_context_active[env_ids] = True
        self._frontres_fixed_noisy_tape_execution_active[env_ids] = execute
        for row, env_id in enumerate(env_ids.detach().cpu().tolist()):
            self._frontres_fixed_noisy_tape_scenario_ids[int(env_id)] = str(scenario_ids[row])
            self._frontres_fixed_noisy_tape_hashes[int(env_id)] = str(noisy_segment_hashes[row])
        # A joint-only override may not coexist with the complete fixed carrier.
        self.clear_frontres_reference_window(env_ids)
        return torch.ones(batch_size, dtype=torch.bool, device=self.device)

    def clear_frontres_fixed_noisy_tape(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if int(ids.numel()) == 0:
            return
        self._frontres_fixed_noisy_tape_lengths[ids] = 0
        self._frontres_fixed_noisy_tape_cursor[ids] = 0
        self._frontres_fixed_noisy_tape_context_active[ids] = False
        self._frontres_fixed_noisy_tape_execution_active[ids] = False
        for env_id in ids.detach().cpu().tolist():
            self._frontres_fixed_noisy_tape_scenario_ids[int(env_id)] = None
            self._frontres_fixed_noisy_tape_hashes[int(env_id)] = None

    def frontres_fixed_noisy_tape_hashes(self, env_ids: torch.Tensor) -> tuple[str, ...]:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if not bool(self._frontres_fixed_noisy_tape_context_active[ids].all()):
            raise RuntimeError("fixed Noisy tape hash requested for an inactive context row")
        values = tuple(self._frontres_fixed_noisy_tape_hashes[int(env_id)] for env_id in ids.detach().cpu().tolist())
        if any(value is None for value in values):
            raise RuntimeError("active fixed Noisy tape row is missing its hash")
        return tuple(str(value) for value in values)

    def _frontres_fixed_noisy_tape_rows(
        self,
        env_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self._frontres_fixed_noisy_tape is None:
            raise RuntimeError("active fixed Noisy tape has no stored values")
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        offsets = torch.as_tensor(offsets, device=self.device, dtype=torch.long).flatten()
        if int(ids.numel()) == 0 or int(offsets.numel()) == 0 or bool((offsets < 0).any()):
            raise ValueError("fixed Noisy tape reads require nonempty nonnegative offsets")
        frame_ids = self._frontres_fixed_noisy_tape_cursor[ids].unsqueeze(1) + offsets.unsqueeze(0)
        max_frame = (self._frontres_fixed_noisy_tape_lengths[ids] - 1).clamp_min(0).unsqueeze(1)
        frame_ids = torch.minimum(frame_ids, max_frame)
        return self._frontres_fixed_noisy_tape[ids.unsqueeze(1), frame_ids]

    def _advance_frontres_fixed_noisy_tape(self) -> None:
        active = self._frontres_fixed_noisy_tape_context_active
        if not bool(active.any()):
            return
        self._frontres_fixed_noisy_tape_cursor[active] += 1
        max_frame = (self._frontres_fixed_noisy_tape_lengths - 1).clamp_min(0)
        self._frontres_fixed_noisy_tape_cursor[active] = torch.minimum(
            self._frontres_fixed_noisy_tape_cursor[active], max_frame[active]
        )

    def _frontres_fixed_noisy_tape_for(self, getter: str, horizon: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._frontres_fixed_noisy_tape is None:
            return None
        if getter not in {"joint_pos", "joint_vel"}:
            return None
        env_ids = torch.nonzero(self._frontres_fixed_noisy_tape_execution_active, as_tuple=False).flatten()
        if int(env_ids.numel()) == 0:
            return None
        offsets = torch.arange(int(horizon), device=self.device, dtype=torch.long)
        rows = self._frontres_fixed_noisy_tape_rows(env_ids, offsets)
        dof = int(self.motion_dir_loader.joint_pos.shape[-1])
        if getter == "joint_pos":
            rows = rows[..., :dof]
        else:
            rows = rows[..., dof : 2 * dof]
        return env_ids, rows

    def _apply_frontres_fixed_noisy_tape(self, getter: str, gathered: torch.Tensor, horizon: int) -> torch.Tensor:
        override = self._frontres_fixed_noisy_tape_for(getter, horizon)
        if override is None:
            return gathered
        env_ids, rows = override
        if gathered.ndim == rows.ndim - 1:
            if int(horizon) != 1:
                raise RuntimeError("single-frame fixed Noisy tape override requires horizon=1")
            rows = rows[:, 0]
        elif gathered.ndim != rows.ndim:
            raise RuntimeError(
                f"fixed Noisy tape rank mismatch: gathered={tuple(gathered.shape)} rows={tuple(rows.shape)}"
            )
        output = gathered.clone()
        output[env_ids] = rows.to(output.device, dtype=output.dtype)
        return output

    def frontres_fixed_noisy_future_context(self, future_offsets: Sequence[int]) -> torch.Tensor:
        """Return actor-only ordered H reads without advancing the K cursor."""

        offsets = tuple(int(offset) for offset in future_offsets)
        if not offsets or any(offset <= 0 for offset in offsets) or tuple(sorted(set(offsets))) != offsets:
            raise ValueError(f"future offsets must be nonempty, positive, ordered, unique; got {offsets}")
        if not bool(self._frontres_fixed_noisy_tape_context_active.all()):
            raise RuntimeError("fixed Noisy actor context requires an installed tape for every actor row")
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        rows = self._frontres_fixed_noisy_tape_rows(
            env_ids,
            torch.tensor(offsets, dtype=torch.long, device=self.device),
        )
        return rows.reshape(self.num_envs, -1)

    def set_frontres_reference_window(self, reference_window: torch.Tensor, *, env_ids: torch.Tensor) -> torch.Tensor:
        """Override GMT command future joint reference for Segment Replay env rows."""
        if bool(self._frontres_local_scenario_active.any()):
            raise RuntimeError("legacy reference window cannot mix with an active v015 local scenario")
        if not isinstance(reference_window, torch.Tensor):
            raise TypeError("reference_window must be a torch.Tensor")
        if reference_window.ndim != 3:
            raise ValueError(f"reference_window must have shape [B, W, F], got {tuple(reference_window.shape)}")
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if int(reference_window.shape[0]) != int(env_ids.numel()):
            raise ValueError(
                "reference_window first dimension must match env_ids, "
                f"got {int(reference_window.shape[0])} and {int(env_ids.numel())}"
            )
        dof = int(self.motion_dir_loader.joint_pos.shape[-1])
        feature_dim = int(reference_window.shape[-1])
        if feature_dim not in (dof, 2 * dof):
            raise ValueError(f"reference_window feature dim must be {dof} or {2 * dof}, got {feature_dim}")
        value = reference_window.to(device=self.device, dtype=torch.float32).detach()
        if (
            self._frontres_reference_window is None
            or tuple(self._frontres_reference_window.shape[1:]) != tuple(value.shape[1:])
        ):
            self._frontres_reference_window = torch.zeros(
                self.num_envs,
                int(value.shape[1]),
                int(value.shape[2]),
                dtype=value.dtype,
                device=self.device,
            )
        self._frontres_reference_window[env_ids] = value
        self._frontres_reference_window_active[env_ids] = True
        self._frontres_reference_window_cursor[env_ids] = 0
        return torch.ones(int(env_ids.numel()), dtype=torch.bool, device=self.device)

    def clear_frontres_reference_window(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._frontres_reference_window_active[:] = False
            self._frontres_reference_window_cursor[:] = 0
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if int(ids.numel()) == 0:
            return
        self._frontres_reference_window_active[ids] = False
        self._frontres_reference_window_cursor[ids] = 0

    def _advance_frontres_reference_window(self) -> None:
        active = self._frontres_reference_window_active
        if not bool(active.any()):
            return
        self._frontres_reference_window_cursor[active] += 1
        if self._frontres_reference_window is None:
            self.clear_frontres_reference_window(torch.nonzero(active, as_tuple=False).flatten())
            return
        window_len = int(self._frontres_reference_window.shape[1])
        expired = active & (self._frontres_reference_window_cursor >= window_len)
        if bool(expired.any()):
            self.clear_frontres_reference_window(torch.nonzero(expired, as_tuple=False).flatten())

    def _frontres_reference_window_for(self, getter: str, horizon: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._frontres_reference_window is None or not bool(self._frontres_reference_window_active.any()):
            return None
        if getter not in {"joint_pos", "joint_vel"}:
            return None
        dof = int(self.motion_dir_loader.joint_pos.shape[-1])
        feature_dim = int(self._frontres_reference_window.shape[-1])
        if getter == "joint_vel" and feature_dim != 2 * dof:
            return None
        env_ids = torch.nonzero(self._frontres_reference_window_active, as_tuple=False).flatten()
        if int(env_ids.numel()) == 0:
            return None
        cursor = self._frontres_reference_window_cursor[env_ids].unsqueeze(1)
        offsets = torch.arange(int(horizon), device=self.device, dtype=torch.long).view(1, -1)
        window_len = int(self._frontres_reference_window.shape[1])
        frame_ids = torch.clamp(cursor + offsets, max=max(window_len - 1, 0))
        rows = self._frontres_reference_window[env_ids.unsqueeze(1), frame_ids]
        if getter == "joint_pos":
            rows = rows[..., :dof]
        else:
            rows = rows[..., dof : 2 * dof]
        return env_ids, rows

    def _apply_frontres_reference_window(self, getter: str, gathered: torch.Tensor, horizon: int) -> torch.Tensor:
        override = self._frontres_reference_window_for(getter, horizon)
        if override is None:
            return gathered
        env_ids, rows = override
        output = gathered.clone()
        output[env_ids] = rows.to(output.device, dtype=output.dtype)
        return output

    # ------------- properties (gathered across envs/motions) -------------
    def _frontres_local_scenario_current_command_rows(self, getter: str, horizon: int) -> torch.Tensor:
        """Return only the current deployment q29/dq29 command before the FEMR action."""

        if horizon != 1:
            raise RuntimeError(
                "v015 local pre-action GMT command requires motion_horizon=1; "
                f"got horizon={horizon}"
            )
        if getter not in {"joint_pos", "joint_vel"}:
            raise RuntimeError(
                "v015 local pre-action GMT command exposes only current q29/dq29; "
                f"getter={getter!r}"
            )
        if (
            not bool(self._frontres_local_scenario_active.all())
            or not bool(self._frontres_local_scenario_current_frame_ready.all())
            or bool(self._frontres_local_scenario_k_execution_active.any())
            or self._frontres_local_scenario_intent_q29 is None
        ):
            raise RuntimeError(
                "v015 local pre-action GMT command requires one transaction-wide current-frame-ready scenario "
                "before the Clean-C K executor opens"
            )

        current_command = self._frontres_local_scenario_current_command_q29_dq29
        if self._frontres_local_scenario_execution_mode == "clean_baseline":
            if self._frontres_local_scenario_clean_reference_t is None:
                raise RuntimeError("v017 Clean baseline lost its sealed current reference")
            current_command = self._frontres_local_scenario_clean_reference_t[:, :58]
        current = current_command[:, :29] if getter == "joint_pos" else current_command[:, 29:]
        expected_shape = (self.num_envs, 29)
        if tuple(current.shape) != expected_shape or not bool(torch.isfinite(current).all()):
            raise RuntimeError(
                "v015 local pre-action GMT command requires finite deployment q29/dq29 rows; "
                f"getter={getter!r} shape={tuple(current.shape)} expected={expected_shape}"
            )
        if getter == "joint_pos" and self._frontres_local_scenario_execution_mode != "clean_baseline":
            sealed_current_q29 = self._frontres_local_scenario_intent_q29[:, 0]
            if not torch.equal(
                current.to(device=sealed_current_q29.device, dtype=sealed_current_q29.dtype),
                sealed_current_q29,
            ):
                raise RuntimeError(
                    "v015 local pre-action GMT command lost the sealed deployment q29 identity at t"
                )
        return current.unsqueeze(1)

    def _gather_future_by_motion(self, getter: str, horizon: int) -> torch.Tensor:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        deployment_active = getattr(
            self,
            "_frontres_v015_deployment_sequence_active",
            torch.zeros_like(self._frontres_local_scenario_active),
        )
        if bool(deployment_active.any()):
            if not bool(deployment_active.all()) or horizon != 1 or getter not in {"joint_pos", "joint_vel"}:
                raise RuntimeError(
                    "v015 deployment GMT command requires all rows, motion_horizon=1, and current q29/dq29 only"
                )
            return self._frontres_v015_deployment_current_rows(getter).unsqueeze(1)
        local_active = self._frontres_local_scenario_active
        if bool(local_active.any()):
            if not bool(local_active.all()):
                raise RuntimeError("v015 local scenario command rows cannot mix with legacy future references")
            if bool(self._frontres_local_scenario_k_execution_active.all()):
                continuation, _valid = self._frontres_local_scenario_continuation_rows(horizon)
                if getter == "joint_pos":
                    return continuation[..., :29]
                if getter == "joint_vel":
                    return continuation[..., 29:58]
                raise RuntimeError(
                    "v015 frozen-GMT continuation exposes only the q29/dq29 command reference; "
                    f"getter={getter!r} is not a Clean-C command field"
                )
            return self._frontres_local_scenario_current_command_rows(getter, horizon)
        motion_indices = self.env_motion_indices
        base_indices = self.time_steps.unsqueeze(1)
        offsets = torch.arange(horizon, device=self.device, dtype=torch.long).view(1, -1)
        frame_indices = base_indices + offsets
        max_valid = self.motion_lengths_minus_one[motion_indices].unsqueeze(1)
        frame_indices = torch.minimum(frame_indices, max_valid)

        flat_motion = motion_indices.unsqueeze(1).expand_as(frame_indices).reshape(-1)
        flat_frames = frame_indices.reshape(-1)
        gathered = self.motion_dir_loader.gather(getter, flat_motion, flat_frames, out_device=self.device)
        new_shape = (self.num_envs, horizon) + gathered.shape[1:]
        gathered = gathered.view(new_shape)
        gathered = self._apply_frontres_reference_window(getter, gathered, horizon)
        return self._apply_frontres_fixed_noisy_tape(getter, gathered, horizon)

    @property
    def command(self) -> torch.Tensor:
        horizon = self.cfg.motion_horizon
        local_pre_action = bool(self._frontres_local_scenario_active.any()) and not bool(
            self._frontres_local_scenario_k_execution_active.any()
        )
        if local_pre_action and not bool(self.cfg.command_velocity):
            raise RuntimeError("v015 local pre-action GMT command requires command_velocity=True for q29+dq29")
        joint_pos_seq = self._gather_future_by_motion("joint_pos", horizon)
        if self.cfg.command_velocity:
            joint_vel_seq = self._gather_future_by_motion("joint_vel", horizon)
            command_seq = torch.cat([joint_pos_seq, joint_vel_seq], dim=-1)
        else:
            command_seq = joint_pos_seq
        return command_seq.reshape(self.num_envs, -1)

    def _gather_by_motion(self, getter: str) -> torch.Tensor:
        if bool(self._frontres_v015_deployment_sequence_active.any()):
            if not bool(self._frontres_v015_deployment_sequence_active.all()):
                raise RuntimeError("v015 deployment reference rows cannot mix with legacy motion rows")
            return self._frontres_v015_deployment_current_rows(getter)
        return self.motion_dir_loader.gather(
            getter, self.env_motion_indices, self.time_steps, out_device=self.device
        )
    
    def _gather_by_motion_for_envs(self, getter: str, env_ids: torch.Tensor) -> torch.Tensor:
        motion_idx = self.env_motion_indices[env_ids]
        frame_idx = self.time_steps[env_ids]
        return self.motion_dir_loader.gather(getter, motion_idx, frame_idx, out_device=self.device)

    def _update_sampling_prob_metrics(self):
        probs = self._compute_motion_sampling_probs()
        if probs.numel() == 0:
            zero = 0.0
            self.metrics["motion_sampling_prob_mean"].fill_(zero)
            self.metrics["motion_sampling_prob_std"].fill_(zero)
            self.metrics["motion_sampling_prob_min"].fill_(zero)
            self.metrics["motion_sampling_prob_max"].fill_(zero)
            self.metrics["motion_sampling_prob_entropy"].fill_(zero)
            return

        mean_val = probs.mean().item()
        std_val = probs.std(unbiased=False).item()
        min_val = probs.min().item()
        max_val = probs.max().item()
        entropy = -(probs * (probs + 1e-12).log()).sum().item()
        norm_entropy = entropy / max(math.log(max(probs.numel(), 1)), 1e-12)

        self.metrics["motion_sampling_prob_mean"].fill_(mean_val)
        self.metrics["motion_sampling_prob_std"].fill_(std_val)
        self.metrics["motion_sampling_prob_min"].fill_(min_val)
        self.metrics["motion_sampling_prob_max"].fill_(max_val)
        self.metrics["motion_sampling_prob_entropy"].fill_(norm_entropy)

    def _motion_sampling_progress(self) -> float:
        """Return ramp progress in [0, 1] for motion-level sampling weights.

        Motivation: early in training, fail statistics are noisy (often everything fails), so we
        start from uniform sampling and gradually increase the configured weights.
        """

        warmup_s = float(getattr(self.cfg, "motion_sampling_warmup_s", 0.0))
        ramp_s = float(getattr(self.cfg, "motion_sampling_ramp_s", 0.0))
        warmup_steps = max(0, int(round(warmup_s / max(self.sim_dt, 1e-12))))
        ramp_steps = max(0, int(round(ramp_s / max(self.sim_dt, 1e-12))))

        if self._global_sim_step <= warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return 1.0

        x = (float(self._global_sim_step - warmup_steps)) / float(ramp_steps)
        x = max(0.0, min(1.0, x))

        schedule = str(getattr(self.cfg, "motion_sampling_schedule", "linear")).lower()
        if schedule == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * x)
        # default: linear
        return x

    def _in_motion_sampling_warmup(self) -> bool:
        """True if we are still in motion-level sampling warmup window."""
        warmup_s = float(getattr(self.cfg, "motion_sampling_warmup_s", 0.0))
        warmup_steps = max(0, int(round(warmup_s / max(self.sim_dt, 1e-12))))
        return self._global_sim_step <= warmup_steps

    def _compute_motion_sampling_probs(self) -> torch.Tensor:
        num_bins = self.num_motions_total
        if num_bins == 0:
            self.motion_sampling_probs = torch.empty(0, device=self.device)
            return self.motion_sampling_probs

        sample_counts = self.motion_sample_counts
        assigned_counts = self.motion_assigned_counts
        fail_counts = self.motion_fail_counts
        
        # fail-based difficulty
        fail_rates = torch.zeros_like(sample_counts)
        valid_mask = sample_counts > 0
        if valid_mask.any():
            fail_rates[valid_mask] = fail_counts[valid_mask] / sample_counts[valid_mask].clamp(min=1e-6)

        mean_fail = fail_rates.mean()
        beta_cap = self.cfg.cap_beta * mean_fail
        if beta_cap > 0:
            capped_rates = torch.minimum(fail_rates, beta_cap)
        else:
            capped_rates = torch.zeros_like(fail_rates)

        capped_sum = capped_rates.sum()
        if capped_sum > 0:
            prob_fail = capped_rates / capped_sum
        else:
            prob_fail = torch.zeros_like(capped_rates)

        # novelty term: prefer less-sampled motions
        novelty = 1.0 / torch.sqrt(assigned_counts + 1.0)
        if novelty.sum() > 0:
            prob_novel = novelty / novelty.sum()
        else:
            prob_novel = torch.zeros_like(novelty)
        
        # uniform term: prefer more uniform sampling
        prob_uniform = torch.full_like(prob_fail, 1.0 / max(num_bins, 1))

        # mix the terms
        progress = self._motion_sampling_progress()
        w_fail_target = float(self.cfg.weight_fail)
        w_novel_target = float(self.cfg.weight_novel)
        w_fail = progress * w_fail_target
        w_novel = progress * w_novel_target

        # keep weights well-formed
        w_sum = w_fail + w_novel
        if w_sum > 1.0:
            w_fail = w_fail / w_sum
            w_novel = w_novel / w_sum
            w_uniform = 0.0
        else:
            w_uniform = max(0.0, 1.0 - w_fail - w_novel)

        probs = w_fail * prob_fail + w_novel * prob_novel + w_uniform * prob_uniform

        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = prob_uniform
        else:
            probs = probs / probs_sum

        self.motion_sampling_probs = probs
        return probs

    @property
    def joint_pos(self) -> torch.Tensor:
        raw = self._gather_by_motion("joint_pos")
        if bool(self._frontres_v015_deployment_sequence_active.all()):
            return raw
        local_active = self._frontres_local_scenario_active
        if bool(local_active.any()):
            if not bool(local_active.all()) or self._frontres_local_scenario_intent_q29 is None:
                raise RuntimeError("v015 local scenario command rows cannot mix with legacy joint references")
            if bool(self._frontres_local_scenario_k_execution_active.all()):
                continuation, _valid = self._frontres_local_scenario_continuation_rows(1)
                return continuation[:, 0, :29]
            # q29 is a deployment-provenance carrier; its numeric calibration may
            # equal Clean motion but it is not a Clean actor/reference window.
            return self._frontres_local_scenario_intent_q29[:, 0].detach().clone()
        if bool(self._frontres_fixed_noisy_tape_context_active.any()):
            if not bool(self._frontres_fixed_noisy_tape_context_active.all()):
                raise RuntimeError("fixed Noisy command rows cannot be mixed with random-perturbation rows")
            return self._apply_frontres_fixed_noisy_tape("joint_pos", raw, horizon=1)
        perturbed = self.perturber.apply_joint_perturbation(raw)
        train_ids = getattr(self, '_frontres_pair_train_ids', None)
        base_ids  = getattr(self, '_frontres_pair_base_ids',  None)
        if train_ids is not None and base_ids is not None:
            perturbed[base_ids] = perturbed[train_ids]
        return perturbed

    @property
    def joint_vel(self) -> torch.Tensor:
        raw = self._gather_by_motion("joint_vel")
        if bool(self._frontres_v015_deployment_sequence_active.all()):
            return raw
        local_active = self._frontres_local_scenario_active
        if bool(local_active.any()) and not bool(local_active.all()):
            raise RuntimeError("v015 local scenario command rows cannot mix with legacy joint references")
        if bool(local_active.all()) and bool(self._frontres_local_scenario_k_execution_active.all()):
            continuation, _valid = self._frontres_local_scenario_continuation_rows(1)
            return continuation[:, 0, 29:58]
        if bool(local_active.all()):
            current_command = self._frontres_local_scenario_current_command_q29_dq29
            if current_command is None:
                raise RuntimeError("v015 local current dq29 requires the sealed current-command carrier")
            return current_command[:, 29:].detach().clone()
        if bool(self._frontres_fixed_noisy_tape_context_active.any()):
            if not bool(self._frontres_fixed_noisy_tape_context_active.all()):
                raise RuntimeError("fixed Noisy command rows cannot be mixed with random-perturbation rows")
            return self._apply_frontres_fixed_noisy_tape("joint_vel", raw, horizon=1)
        return raw

    @property
    def body_pos_w(self) -> torch.Tensor:
        body_pos = self._gather_by_motion("body_pos_w") + self._env.scene.env_origins[:, None, :]
        if bool(self._frontres_local_scenario_active.all()) and bool(
            self._frontres_local_scenario_k_execution_active.all()
        ):
            continuation, _valid = self._frontres_local_scenario_continuation_rows(1)
            body_pos = body_pos.clone()
            body_pos[:, self.motion_anchor_body_index] = (
                continuation[:, 0, 58:61] + self._env.scene.env_origins
            )
        return body_pos

    @property
    def body_quat_w(self) -> torch.Tensor:
        body_quat = self._gather_by_motion("body_quat_w")
        if bool(self._frontres_local_scenario_active.all()) and bool(
            self._frontres_local_scenario_k_execution_active.all()
        ):
            continuation, _valid = self._frontres_local_scenario_continuation_rows(1)
            body_quat = body_quat.clone()
            body_quat[:, self.motion_anchor_body_index] = continuation[:, 0, 61:65]
        return body_quat

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_lin_vel_w")

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_ang_vel_w")

    @property
    def anchor_dr_delta_pos(self) -> torch.Tensor:
        """DR-induced anchor position delta: perturbed_pos - clean_pos (world frame)."""
        root_pos_ref = self._gather_by_motion("body_pos_w")[:, self.motion_anchor_body_index]
        return self._cached_perturbed_pos - root_pos_ref

    @property
    def anchor_dr_delta_quat_correction(self) -> torch.Tensor:
        """Quaternion correction that undoes DR tilt (anchor local frame, wxyz).

        perturbed_quat = tilt * clean_quat   (left-multiply, world-frame tilt)
        We want:  quat_mul(perturbed_quat, correction) = clean_quat
        → correction = quat_mul(quat_inv(perturbed_quat), clean_quat)
        """
        root_quat = self._gather_by_motion("body_quat_w")[:, self.motion_anchor_body_index]
        return quat_mul(quat_inv(self._cached_perturbed_quat), root_quat)

    @property
    def anchor_pos_w_original(self) -> torch.Tensor:
        """Uncorrected motion-carrier anchor; deployment mode is not a Clean oracle."""
        pos = self._gather_by_motion("body_pos_w")
        return pos[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_pos_w_raw(self) -> torch.Tensor:
        """Perturbed anchor position (DR applied, no FrontRES correction)."""
        return self._cached_perturbed_pos + self._env.scene.env_origins

    @property
    def anchor_penetration_depth(self) -> torch.Tensor:
        """Depth below the local ground/contact boundary for the perturbed anchor.

        Positive values mean the degraded reference anchor is below the local
        ground plane.  This is a conservative upper bound for upward FrontRES
        correction: it can remove penetration, but cannot reconstruct missing
        flight height.
        """
        ground_z = self._env.scene.env_origins[:, 2]
        raw_z = self.anchor_pos_w_raw[:, 2]
        return torch.clamp(ground_z - raw_z, min=0.0)

    @property
    def anchor_quat_w_original(self) -> torch.Tensor:
        """Uncorrected motion-carrier quaternion; deployment mode is not a Clean oracle."""
        quat = self._gather_by_motion("body_quat_w")
        return quat[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w_raw(self) -> torch.Tensor:
        """Perturbed anchor quaternion (DR applied, no FrontRES correction)."""
        return self._cached_perturbed_quat

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._cached_perturbed_pos + self._env.scene.env_origins + self._frontres_pos_correction

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        # Right-multiply: apply correction in anchor's local frame.
        return quat_mul(self._cached_perturbed_quat, self._frontres_quat_correction)

    @property
    def supervised_target(self) -> torch.Tensor:
        """ΔSE3 = [Δpos(3), Δrpy(3)] that UNDOES the current DR perturbation.

        Cached once per step in _update_command(). All zeros when perturber is disabled.
        Used by on_policy_runner to store supervised training targets for FrontRES.
        """
        return self._dr_supervised_target

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        vel = self._gather_by_motion("body_lin_vel_w")
        return vel[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        vel = self._gather_by_motion("body_ang_vel_w")
        return vel[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _assign_motions(self, env_ids: torch.Tensor):
        """Assign motions to given env ids using difficulty-based and novelty-based sampling."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Check if motion group sampling ratios are configured
        if self.cfg.motion_group_sampling_ratios is not None:
            # Use ratio-based sampling: split environments by group and sample from each group's pool
            self._assign_motions_with_ratios(env_ids)
            return

        probs_motion = self._compute_motion_sampling_probs()

        k_cfg = getattr(self.cfg, "max_active_motions", None)
        use_active_pool = (k_cfg is not None) and (self.num_motions_total > k_cfg)

        if use_active_pool:
            # sample a active set from the motion database (no replacement)
            K = min(k_cfg, self.num_motions_total, n_envs)
            active_motions = torch.multinomial(probs_motion, K, replacement=False)

            # using these motions to fill the envs
            # make every motion get about n_envs / K envs
            reps = n_envs // K
            rem = n_envs % K

            base = active_motions.repeat_interleave(reps)
            if rem > 0:
                extra = active_motions[torch.randperm(K, device=self.device)[:rem]]
                sampled = torch.cat([base, extra], dim=0)
            else:
                sampled = base

            # randomize the order of the envs
            perm = torch.randperm(n_envs, device=self.device)
            self.env_motion_indices[env_ids] = sampled[perm]

            # Update motion groups for multi-teacher support
            for i, env_id in enumerate(env_ids):
                motion_idx = self.env_motion_indices[env_id].item()
                group_name = self.motion_dir_loader.motion_to_group.get(motion_idx, "default")
                group_idx = self.group_name_to_idx[group_name]
                self.env_motion_groups[env_id] = group_idx

            # update the motion sample counts
            unique_motions, counts = torch.unique(sampled, return_counts=True)
            self.motion_assigned_counts.index_add_(
                0, unique_motions, counts.to(self.motion_assigned_counts.dtype)
            )

        else:
            if self.cfg.unique_per_batch and self.num_motions_total >= n_envs:
                sampled = torch.multinomial(probs_motion, n_envs, replacement=False)
            else:
                sampled = torch.multinomial(probs_motion, n_envs, replacement=True)

            self.env_motion_indices[env_ids] = sampled

            # Update motion groups for multi-teacher support
            for i, env_id in enumerate(env_ids):
                motion_idx = self.env_motion_indices[env_id].item()
                group_name = self.motion_dir_loader.motion_to_group.get(motion_idx, "default")
                group_idx = self.group_name_to_idx[group_name]
                self.env_motion_groups[env_id] = group_idx

            unique_motions, counts = torch.unique(sampled, return_counts=True)
            self.motion_assigned_counts.index_add_(
                0, unique_motions, counts.to(self.motion_assigned_counts.dtype)
            )

        self._update_sampling_prob_metrics()

    def _assign_motions_with_ratios(self, env_ids: torch.Tensor):
        """Assign motions to environments using motion group sampling ratios.

        This method splits environments according to configured sampling ratios,
        then samples motions from each group's pool separately while maintaining
        difficulty-based and novelty-based sampling within each group.

        Args:
            env_ids: Tensor of environment IDs to assign motions to
        """
        n_envs = len(env_ids)
        ratios = self.cfg.motion_group_sampling_ratios

        # Validate ratios
        ratio_sum = sum(ratios.values())
        if abs(ratio_sum - 1.0) > 1e-5:
            print(f"[MultiMotionCommand] Warning: motion_group_sampling_ratios sum to {ratio_sum:.4f}, not 1.0. Normalizing...")
            ratios = {k: v / ratio_sum for k, v in ratios.items()}

        # Check that all ratio groups exist in motion groups
        for group_name in ratios.keys():
            if group_name not in self.group_name_to_idx:
                raise ValueError(f"Sampling ratio specified for group '{group_name}' but this group doesn't exist. "
                               f"Available groups: {list(self.group_name_to_idx.keys())}")

        # Compute global motion sampling probabilities (for difficulty/novelty weighting)
        probs_motion_global = self._compute_motion_sampling_probs()

        # Split environments by group according to ratios
        group_env_splits = {}
        start_idx = 0
        for group_name, ratio in sorted(ratios.items()):
            n_envs_group = int(round(n_envs * ratio))
            # Ensure we don't exceed total environments
            if group_name == list(sorted(ratios.keys()))[-1]:  # Last group gets remainder
                n_envs_group = n_envs - start_idx

            if n_envs_group > 0:
                group_env_splits[group_name] = env_ids[start_idx:start_idx + n_envs_group]
                start_idx += n_envs_group

        print(f"[MultiMotionCommand] Assigning {n_envs} environments with ratios: "
              f"{', '.join([f'{k}={len(v)}/{n_envs}' for k, v in group_env_splits.items()])}")

        # Sample motions for each group
        for group_name, group_env_ids in group_env_splits.items():
            self._assign_motions_for_group(group_name, group_env_ids, probs_motion_global)

        self._update_sampling_prob_metrics()

    def _assign_motions_for_group(self, group_name: str, env_ids: torch.Tensor, probs_motion_global: torch.Tensor):
        """Assign motions from a specific group to given environments.

        Args:
            group_name: Name of the motion group to sample from
            env_ids: Tensor of environment IDs to assign motions to
            probs_motion_global: Global motion sampling probabilities (for all motions)
        """
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Get motion indices for this group
        group_motion_indices = self.group_to_motions_tensor[group_name]
        n_motions_in_group = len(group_motion_indices)

        if n_motions_in_group == 0:
            raise ValueError(f"Group '{group_name}' has no motions!")

        # Extract and renormalize probabilities for this group's motions
        probs_group = probs_motion_global[group_motion_indices]
        probs_group = probs_group / probs_group.sum()

        # Sample motions from this group
        k_cfg = getattr(self.cfg, "max_active_motions", None)
        use_active_pool = (k_cfg is not None) and (self.num_motions_total > k_cfg)

        if use_active_pool:
            # Sample active motions from this group's pool
            K = min(k_cfg, n_motions_in_group, n_envs)
            active_motion_indices_in_group = torch.multinomial(probs_group, K, replacement=False)
            active_motions = group_motion_indices[active_motion_indices_in_group]

            # Distribute these motions across environments
            reps = n_envs // K
            rem = n_envs % K

            base = active_motions.repeat_interleave(reps)
            if rem > 0:
                extra = active_motions[torch.randperm(K, device=self.device)[:rem]]
                sampled = torch.cat([base, extra], dim=0)
            else:
                sampled = base

            # Randomize order
            perm = torch.randperm(n_envs, device=self.device)
            self.env_motion_indices[env_ids] = sampled[perm]

        else:
            # Direct sampling from group
            if self.cfg.unique_per_batch and n_motions_in_group >= n_envs:
                sampled_indices_in_group = torch.multinomial(probs_group, n_envs, replacement=False)
            else:
                sampled_indices_in_group = torch.multinomial(probs_group, n_envs, replacement=True)

            sampled = group_motion_indices[sampled_indices_in_group]
            self.env_motion_indices[env_ids] = sampled

        # Update motion groups (all environments in this batch belong to the same group)
        group_idx = self.group_name_to_idx[group_name]
        self.env_motion_groups[env_ids] = group_idx

        # Update motion sample counts
        unique_motions, counts = torch.unique(sampled, return_counts=True)
        self.motion_assigned_counts.index_add_(
            0, unique_motions, counts.to(self.motion_assigned_counts.dtype)
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Within-motion sampling for the provided environment indices."""
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        motion_indices = self.env_motion_indices[env_ids]

        lengths = self.motion_lengths[motion_indices]
        lengths_minus_one = self.motion_lengths_minus_one[motion_indices]
        denominators = self.motion_length_denominator[motion_indices]
        bin_counts = self.motion_bin_counts[motion_indices]
        bin_counts_float = self.motion_bin_counts_float[motion_indices]
        bin_mask = self.motion_bin_mask[motion_indices]

        # During warmup, we intentionally do NOT update failure statistics to avoid
        # cold-start bias (everything fails early, which makes difficulty estimates noisy).
        if not self._in_motion_sampling_warmup():
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if torch.any(episode_failed):
                fail_envs = env_ids[episode_failed]
                fail_motion_idx = motion_indices[episode_failed]
                fail_bin_counts = bin_counts[episode_failed]
                fail_denominators = denominators[episode_failed]
                fail_bins = torch.clamp(
                    (self.time_steps[fail_envs] * fail_bin_counts) // fail_denominators,
                    max=fail_bin_counts - 1,
                )
                linear_indices = fail_motion_idx * self.max_bin_count + fail_bins
                self.current_bin_failed.view(-1).index_add_(
                    0,
                    linear_indices,
                    torch.ones_like(fail_bins, dtype=self.current_bin_failed.dtype),
                )
                self.motion_fail_counts.index_add_(
                    0,
                    fail_motion_idx,
                    torch.ones_like(fail_motion_idx, dtype=self.motion_fail_counts.dtype),
                )

        prob = self.bin_failed_count[motion_indices]
        uniform_term = (self.cfg.adaptive_uniform_ratio / bin_counts_float).unsqueeze(1)
        prob = (prob + uniform_term) * bin_mask

        kernel_tail = self.kernel.numel() - 1
        if kernel_tail > 0:
            prob = F.conv1d(
                F.pad(prob.unsqueeze(1), (0, kernel_tail), mode="replicate"),
                self.kernel.view(1, 1, -1),
            ).squeeze(1)
        else:
            prob = prob.clone()

        prob = prob * bin_mask
        prob_sum = prob.sum(dim=1, keepdim=True)
        zero_rows = prob_sum <= 0
        if torch.any(zero_rows):
            prob[zero_rows] = bin_mask[zero_rows].float()
            prob_sum = prob.sum(dim=1, keepdim=True)
        prob = prob / prob_sum

        sampled_bins = torch.multinomial(prob, 1).squeeze(1)
        rand_offset = torch.rand(len(env_ids), device=self.device)

        lengths_minus_one_float = lengths_minus_one.to(torch.float32)
        time_steps = torch.where(
            lengths_minus_one == 0,
            torch.zeros_like(lengths_minus_one),
            (
                (sampled_bins.to(torch.float32) + rand_offset)
                / torch.clamp(bin_counts_float, min=1.0)
                * lengths_minus_one_float
            ).long(),
        )
        time_steps = torch.clamp(time_steps, max=lengths_minus_one)
        self.time_steps[env_ids] = time_steps

        entropy = -(prob * (prob + 1e-12).log()).sum(dim=1)
        log_bins = torch.log(torch.clamp(bin_counts_float, min=1.0))
        entropy_norm = torch.where(log_bins > 0, entropy / log_bins, torch.zeros_like(entropy))
        self.metrics["sampling_entropy"][env_ids] = entropy_norm

        pmax, imax = prob.max(dim=1)
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.to(torch.float32) / torch.clamp(bin_counts_float, min=1.0)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = self._expand_frontres_pair_env_ids(env_ids)
        if self.cfg.start_from_beginning:
            start_frame = max(int(self.cfg.start_frame), 0)
            lengths_minus_one = self.motion_lengths[self.env_motion_indices[env_ids]] - 1
            lengths_minus_one = torch.clamp(lengths_minus_one, min=0)
            start_frame_tensor = torch.full_like(lengths_minus_one, start_frame)
            self.time_steps[env_ids] = torch.minimum(start_frame_tensor, lengths_minus_one)
        else:
            self._adaptive_sampling(env_ids)

        self._sync_frontres_pairs(sync_perturbation=False)

        motion_indices = self.env_motion_indices[env_ids]
        self.motion_sample_counts.index_add_(
            0,
            motion_indices,
            torch.ones_like(motion_indices, dtype=self.motion_sample_counts.dtype),
        )

        # Gather current sampled states
        body_pos = self._gather_by_motion_for_envs("body_pos_w", env_ids)
        body_quat = self._gather_by_motion_for_envs("body_quat_w", env_ids)
        body_lin = self._gather_by_motion_for_envs("body_lin_vel_w", env_ids)
        body_ang = self._gather_by_motion_for_envs("body_ang_vel_w", env_ids)
        jpos = self._gather_by_motion_for_envs("joint_pos", env_ids)
        jvel = self._gather_by_motion_for_envs("joint_vel", env_ids)

        root_pos = (body_pos[:, 0] + self._env.scene.env_origins[env_ids])
        root_ori = body_quat[:, 0]
        root_lin_vel = body_lin[:, 0].clone()
        root_ang_vel = body_ang[:, 0].clone()

        # Random pose/velocity deltas around sampled states
        rand_samples = sample_uniform(self._pose_ranges[:, 0], self._pose_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)

        rand_samples = sample_uniform(self._vel_ranges[:, 0], self._vel_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = jpos.clone()
        joint_vel = jvel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )
        # Reset FrontRES anchor corrections and OU perturbation states for resampled envs
        self._frontres_pos_correction[env_ids] = 0.0
        self._frontres_quat_correction[env_ids] = 0.0
        self._frontres_quat_correction[env_ids, 0] = 1.0
        self.clear_frontres_reference_window(env_ids)
        self.perturber.reset_envs(env_ids)

    def _update_metrics(self):
        self.metrics["error_anchor_pos"].copy_(torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1))
        self.metrics["error_anchor_rot"].copy_(quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w))
        self.metrics["error_anchor_lin_vel"].copy_(torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1))
        self.metrics["error_anchor_ang_vel"].copy_(torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1))

        self.metrics["error_body_pos"].copy_(torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1))
        self.metrics["error_body_rot"].copy_(quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1))

        self.metrics["error_body_lin_vel"].copy_(torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1))
        self.metrics["error_body_ang_vel"].copy_(torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1))

        self.metrics["error_joint_pos"].copy_(torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1))
        self.metrics["error_joint_vel"].copy_(torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1))
        self._update_sampling_prob_metrics()

    def _compute_jump_degree(self):
        """Per-step update of jump_degree via parabola fit on anchor-body root_z.

        Algorithm
        ---------
        1. Gather root_z for frames [t-N, ..., t, ..., t+K] from the reference motion.
        2. Fit z(t) = A*t² + B*t + C using precomputed least-squares weights w_A.
        3. Convert A (frame-index units) to physical acceleration: a_z = 2*A*fps².
        4. jump_degree = exp(-(a_z + g)² / 2σ²)
           ≈ 1 only when a_z ≈ -g (free flight); ≈ 0 on ground or float artifact.

        The runner multiplies Δpos by (1 - jump_degree) before applying to the
        command term, suppressing height correction during genuine airborne phases.
        """
        N, K = self._jump_n_past, self._jump_k_future
        W    = N + K + 1

        # Build (N_envs, W) frame-index tensor clamped to valid motion range
        offsets      = torch.arange(-N, K + 1, device=self.device)            # (W,)
        frame_idx    = self.time_steps.unsqueeze(1) + offsets.unsqueeze(0)    # (N_envs, W)
        max_frames   = self.motion_lengths_minus_one[self.env_motion_indices] \
                           .unsqueeze(1).expand_as(frame_idx)
        frame_idx    = frame_idx.clamp(min=0)
        frame_idx    = torch.min(frame_idx, max_frames)

        # Convert (env, window_offset) frame indices to global flat indices.
        # Uses precomputed per-motion offsets to avoid the full body_pos_w gather.
        motion_offsets = self.motion_dir_loader.motion_offsets[self.env_motion_indices]  # (N_envs,)
        global_idx = (motion_offsets.unsqueeze(1) + frame_idx).reshape(-1)  # (N_envs*W,)

        # Anchor body z-coordinate → (N_envs, W)  [scalar gather, 42× cheaper]
        anchor_z = self._jump_anchor_z[global_idx].view(self.num_envs, W)

        # A coefficient: a_z_frame² = w_A · Z  →  (N_envs,)
        A = (anchor_z * self._jump_w_A.unsqueeze(0)).sum(dim=1)

        # fps per env → physical a_z = 2A * fps²
        fps  = self.motion_dir_loader.motion_fps[self.env_motion_indices]  # (N_envs,)
        a_z  = 2.0 * A * fps.pow(2)

        # Gaussian kernel centred at -g
        self.jump_degree = torch.exp(
            -(a_z + self._jump_g).pow(2) / (2.0 * self._jump_sigma * self._jump_sigma)
        )  # (N_envs,) ∈ [0, 1]

    def refresh_frontres_reference_cache_current_frame(self) -> None:
        """Refresh FrontRES reference caches without advancing the motion frame.

        Status: active command-owned reset boundary.
        Upstream: Segment index reset after motion/frame and perturbation setup.
        Downstream: anchor properties, termination, observations, and GMT rollout.
        Evidence: runtime-confirmed by E37; current-frame cache is aligned before first termination.
        """

        local_active = self._frontres_local_scenario_active
        if bool(local_active.any()):
            if (
                not bool(local_active.all())
                or self._frontres_local_scenario_current_root_artifact_t is None
            ):
                raise RuntimeError("v015 local scenario current cache cannot mix with legacy command rows")
            ready = self._frontres_local_scenario_current_frame_ready
            if bool(ready.any()):
                if not bool(ready.all()):
                    raise RuntimeError("v015 local scenario current cache readiness must be transaction-wide")
                raise RuntimeError(
                    "v015 local scenario current cache was already installed; Step 2B owns all subsequent K-step command advancement"
                )
            if self._frontres_local_scenario_execution_mode == "clean_baseline":
                if self._frontres_local_scenario_clean_reference_t is None:
                    raise RuntimeError("v017 Clean baseline lost its sealed current reference")
                artifact = self._frontres_local_scenario_clean_reference_t[:, 58:65]
            else:
                artifact = self._frontres_local_scenario_current_root_artifact_t
            self._cached_perturbed_pos.copy_(artifact[:, :3])
            self._cached_perturbed_quat.copy_(artifact[:, 3:])
            self._dr_supervised_target.zero_()
            ready[:] = True
            return

        fixed_context = self._frontres_fixed_noisy_tape_context_active
        if bool(fixed_context.any()):
            if not bool(fixed_context.all()):
                raise RuntimeError("fixed Noisy command rows cannot be mixed with random-perturbation rows")

            # The carrier already contains the sealed noisy current anchor.  Do
            # not touch MotionPerturber here: every replay attempt must reread
            # the same materialized scenario rather than draw a new perturbation.
            _pos_data = self._gather_by_motion("body_pos_w")
            _root_pos_ref = _pos_data[:, self.motion_anchor_body_index]
            _quat_data = self._gather_by_motion("body_quat_w")
            _root_quat_ref = _quat_data[:, self.motion_anchor_body_index]
            execution_ids = torch.nonzero(self._frontres_fixed_noisy_tape_execution_active, as_tuple=False).flatten()
            clean_ids = torch.nonzero(~self._frontres_fixed_noisy_tape_execution_active, as_tuple=False).flatten()
            if int(execution_ids.numel()) > 0:
                current_rows = self._frontres_fixed_noisy_tape_rows(
                    execution_ids, torch.zeros(1, dtype=torch.long, device=self.device)
                )[:, 0]
                dof = int(self.motion_dir_loader.joint_pos.shape[-1])
                self._cached_perturbed_pos[execution_ids] = current_rows[:, 2 * dof : 2 * dof + 3]
                self._cached_perturbed_quat[execution_ids] = current_rows[:, 2 * dof + 3 : 2 * dof + 7]
            if int(clean_ids.numel()) > 0:
                self._cached_perturbed_pos[clean_ids] = _root_pos_ref[clean_ids]
                self._cached_perturbed_quat[clean_ids] = _root_quat_ref[clean_ids]

            self._compute_jump_degree()
            self._dr_supervised_target.zero_()
            if int(execution_ids.numel()) > 0:
                self._dr_supervised_target[execution_ids, :3] = (
                    _root_pos_ref[execution_ids] - self._cached_perturbed_pos[execution_ids]
                )
                z_upper = self.jump_degree[execution_ids] * self.anchor_penetration_depth[execution_ids]
                self._dr_supervised_target[execution_ids, 2] = torch.minimum(
                    self._dr_supervised_target[execution_ids, 2], z_upper
                )
                corr_quat = quat_mul(
                    quat_inv(self._cached_perturbed_quat[execution_ids]), _root_quat_ref[execution_ids]
                )
                self._dr_supervised_target[execution_ids, 3:6] = _quat_to_rotvec_wxyz(corr_quat)
            return

        # B1: 从当前 time_steps 读取一次 sampled-frame reference, 不推进 frame.
        _pos_data = self._gather_by_motion("body_pos_w")
        _root_pos_ref = _pos_data[:, self.motion_anchor_body_index]
        self._cached_perturbed_pos = self.perturber.apply_perturbations(
            _root_pos_ref,
            _pos_data[:, self.left_foot_idx],
            _pos_data[:, self.right_foot_idx],
        )
        _quat_data = self._gather_by_motion("body_quat_w")
        _root_quat_ref = _quat_data[:, self.motion_anchor_body_index]
        self._cached_perturbed_quat = self.perturber.apply_quat_perturbation(_root_quat_ref)

        # B2: 使用同一次 perturbation draw 构造监督 target 与 vertical feasibility.
        self._compute_jump_degree()
        self._dr_supervised_target[:, :3] = _root_pos_ref - self._cached_perturbed_pos
        z_upper = self.jump_degree * self.anchor_penetration_depth
        self._dr_supervised_target[:, 2] = torch.minimum(self._dr_supervised_target[:, 2], z_upper)
        _corr_quat = quat_mul(quat_inv(self._cached_perturbed_quat), _root_quat_ref)
        _corr_rotvec = _quat_to_rotvec_wxyz(_corr_quat)
        self._dr_supervised_target[:, 3:6] = _corr_rotvec

        # B3: 将同一当前帧 cache 同步到 quartet, Clean 保持 clean/no-op 语义.
        self._sync_frontres_pairs(sync_perturbation=True)

    def _advance_frontres_command_clock(self) -> str:
        """Advance the legacy reference clock or hold the explicit v015 clock.

        Status: active R6-F1 command-clock owner. Local scenarios keep the
        sealed current/C reference; only the Step 2B cursor may advance them.
        """

        # B1: 根据 deployment/local-scenario owner 选择 command clock, 保持 current reference identity.

        self._global_sim_step += 1
        deployment_active = getattr(
            self,
            "_frontres_v015_deployment_sequence_active",
            torch.zeros_like(self._frontres_local_scenario_active),
        )
        if bool(deployment_active.any()):
            if not bool(deployment_active.all()):
                raise RuntimeError("v015 deployment command clock cannot mix with legacy rows")
            # The composition executor advances this cursor only after it has
            # captured the current frame metrics. IsaacLab command compute must hold.
            return "deployment_current_hold"
        local_active = self._frontres_local_scenario_active
        if bool(local_active.any()):
            ready = self._frontres_local_scenario_current_frame_ready
            execution = self._frontres_local_scenario_k_execution_active
            if not bool(local_active.all()) or not bool(ready.all()):
                raise RuntimeError(
                    "v015 local command clock requires one transaction-wide active, current-frame-ready scenario"
                )
            if bool(execution.any()) and not bool(execution.all()):
                raise RuntimeError("v015 local command clock cannot mix current and Clean-C execution rows")
            # IsaacLab 在每次 env.step 后调用 command compute. 对 local scenario,
            # t reference 和每个 C[offset] 都已由显式 owner 安装, 此处只能 hold.
            return "local_k_hold" if bool(execution.all()) else "local_current_hold"

        self.time_steps += 1
        self._advance_frontres_reference_window()
        self._advance_frontres_fixed_noisy_tape()

        # Each command step advances first, then draws exactly one perturbation
        # sample for the new frame. Index reset calls the same cache owner for
        # its explicitly selected current frame before the first termination.
        self.refresh_frontres_reference_cache_current_frame()
        return "legacy_advance"

    def _update_command(self):
        self._advance_frontres_command_clock()
        # ─────────────────────────────────────────────────────────────────────

        # Per-motion episode end detection and resampling
        motion_lengths = self.motion_lengths[self.env_motion_indices]
        ended = self.time_steps >= motion_lengths                                  # (num_envs,)
        self.motion_end_buf[:] = ended
        # envs_to_resample = torch.where(self.time_steps >= motion_lengths)[0]
        # if envs_to_resample.numel() > 0:
        #     self._resample_command(envs_to_resample)

        # Compute relative body poses vs anchor
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # Update per-motion failure statistics using EMA (skip during warmup)
        if not self._in_motion_sampling_warmup():
            mask = self.motion_bin_mask.float()
            if self.current_bin_failed.any():
                self.bin_failed_count = (
                    self.cfg.adaptive_alpha * self.current_bin_failed
                    + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
                ) * mask
        self.current_bin_failed.zero_()

        # Periodic motion remap
        if self._resample_motions_every_steps > 0 and \
            (self._global_sim_step % self._resample_motions_every_steps) == 0:
            self._remap_version += 1

        # jump_degree was updated before supervised_target construction above.

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        need = self._env_remap_version[env_ids] < self._remap_version
        envs_need_remap = env_ids[need]
        if envs_need_remap.numel() > 0:
            self._assign_motions(envs_need_remap)
            self._env_remap_version[envs_need_remap] = self._remap_version

        self.motion_end_buf[env_ids] = False
        
        return super().reset(env_ids=env_ids)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name))
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name))
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the multi-motion command."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING

    motion: str = MISSING
    file_glob: str = "*.npz"
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    motion_preload_device: str | None = 'cuda'

    motion_horizon: int = 1

    # ---- multi-teacher support: motion groups ----
    # Motion groups configuration for multi-teacher support.
    # Maps group names to folder patterns for categorizing motions.
    # Example: {"lafan": ["lafan_npz_10s_without_fall_and_getup"], "fld": ["motions_fld_test"]}
    # If None, all motions belong to a single "default" group.
    motion_groups: dict[str, list[str]] | None = None

    # Motion group sampling ratios - controls proportion of environments assigned to each group.
    # Maps group names to sampling ratios (should sum to 1.0).
    # Example: {"lafan": 0.7, "fld": 0.3} means 70% of envs use lafan motions, 30% use fld motions.
    # If None, environments are assigned uniformly across all available motions (ignoring groups).
    motion_group_sampling_ratios: dict[str, float] | None = None

    # ---- dataset slicing / sharding across GPUs ----
    # If True, each GPU process loads a disjoint subset of motions (when possible),
    # reducing duplicates during multi-GPU training. This is especially useful when
    # total motions >> (num_envs or max_active_motions).
    motion_dataset_shard_across_gpus: bool = True
    # Use 'global' rank/world_size (RANK/WORLD_SIZE) or 'local' (LOCAL_RANK/LOCAL_WORLD_SIZE).
    motion_dataset_shard_by: str = "global"
    # Deterministic shuffle seed before slicing.
    motion_dataset_shard_seed: int = 0
    # Sharding strategy when total < world_size * load_cap: 'chunk' (contiguous) or 'stride' (round-robin).
    motion_dataset_shard_strategy: str = "chunk"
    # Cap number of motions loaded per process. If None and sharding enabled, auto-caps to min(num_envs, max_active_motions).
    motion_dataset_load_cap: int | None = None
    # If True, print shard info once at startup (useful for sanity checks).
    motion_dataset_log_shard_info: bool = False
    # If True, write shard info once into wandb run.summary (rank0 only). No-op if wandb isn't used.
    motion_dataset_log_wandb_summary: bool = True

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    start_from_beginning: bool = False
    start_frame: int = 0

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    weight_fail: float = 0.5
    weight_novel: float = 0.3
    cap_beta: float = 2.0

    command_velocity: bool = True

    # ---- motion-level sampling weight schedule (uniform -> ramp to weights above) ----
    # insight for the values:
    # - warmup_s: make sure every motion is sampled at least once. (10-20 times of resample_motions_every_s)
    # - ramp_s: make sure the weights are not too small. (20-50 times of resample_motions_every_s)
    # - schedule: "linear" or "cosine" is the schedule type for the ramp. (cosine is better)
    # Warmup: keep motion sampling uniform for this duration (seconds).
    motion_sampling_warmup_s: float = 1000000000.0
    # Ramp: linearly/cosine ramp fail/novel weights from 0 -> target over this duration (seconds).
    motion_sampling_ramp_s: float = 1000000000.0
    # Schedule type for ramp: "linear" or "cosine".
    motion_sampling_schedule: str = "linear"

    # Resampling cadence (seconds) for motion-to-env reassignment (set to 0 or 1e9 to disable)
    resample_motions_every_s: float = 1000000000.0
    # Whether to sample motions without replacement per remap batch when possible
    unique_per_batch: bool = True

    max_active_motions: int | None = 10000

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
