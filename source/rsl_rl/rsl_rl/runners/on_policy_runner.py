# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import json
import os
import shutil
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation, MOSAIC, FrontRESUnified
from whole_body_tracking.utils.supervise import SuperviseTrainer
from rsl_rl.modules.supervise_learning import SuperviseLearning
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticFSQ,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
    ActorCriticTransformer,
    ActorCriticVQ,
    ActorCriticAttention,
    ResidualActorCritic,
    FrontRESActorCritic, # 引入第二阶段模型
)
from rsl_rl.utils import store_code_state
from isaaclab.utils.math import (
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_rotate_inverse,
    euler_xyz_from_quat,
    quat_mul,
    quat_inv,
    quat_apply,
    yaw_quat,
)


def _quat_to_rotvec_wxyz(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map wxyz unit quaternions to shortest-path rotation vectors."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    xyz_norm = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, q[..., :1].clamp(min=eps))
    scale = torch.where(xyz_norm > eps, angle / xyz_norm.clamp(min=eps), 2.0 * torch.ones_like(xyz_norm))
    return xyz * scale


def _rotvec_to_quat_wxyz(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map local rotation vectors to wxyz unit quaternions."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    xyz_scale = torch.where(
        angle > eps,
        torch.sin(half) / angle.clamp(min=eps),
        0.5 * torch.ones_like(angle),
    )
    quat = torch.cat([torch.cos(half), rotvec * xyz_scale], dim=-1)
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=eps)


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def _apply_frontres_specialist_mode(self) -> None:
        """Apply narrow FrontRES demo-specialist presets before policy/algorithm construction."""
        if self.training_type != "frontres":
            return
        mode = str(self.cfg.get("frontres_specialist_mode", "") or "").lower()
        if mode in ("rp", "local_rp", "rp_only", "strong_rp"):
            task_conf_dim = int(self.policy_cfg.get("task_conf_dim", 2))
            active_dims = [0, 1, 2, 3, 4, 5, 6] if task_conf_dim == 1 else (
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if task_conf_dim == 6 else [3, 4, 7]
            )
            self.cfg["frontres_specialist_mode"] = "rp"
            self.cfg["frontres_active_task_dims"] = active_dims
            self.cfg["frontres_perturbation_channels"] = "rp"
            self.cfg["frontres_exec_task_weight"] = 0.0
            self.cfg["frontres_exec_cone_task_weight"] = 0.0
            self.alg_cfg["frontres_active_task_dims"] = active_dims
            print(
                "[Runner] FrontRES specialist mode enabled: rp "
                f"(local_rp only; active dims={active_dims})",
                flush=True,
            )
            return
        if mode not in ("rp_z", "z_rp", "vertical_contact"):
            return

        task_conf_dim = int(self.policy_cfg.get("task_conf_dim", 2))
        active_dims = [0, 1, 2, 3, 4, 5, 6] if task_conf_dim == 1 else (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if task_conf_dim == 6 else [2, 3, 4, 6, 7]
        )
        self.cfg["frontres_specialist_mode"] = "rp_z"
        self.cfg["frontres_active_task_dims"] = active_dims
        self.cfg["frontres_perturbation_channels"] = "rp_z"
        self.cfg["frontres_exec_task_weight"] = 0.0
        self.cfg["frontres_exec_cone_task_weight"] = 0.0
        self.alg_cfg["frontres_active_task_dims"] = active_dims
        print(
            "[Runner] FrontRES specialist mode enabled: rp_z "
            f"(global_z + local_rp; active dims={active_dims})",
            flush=True,
        )

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm 训练算法
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "MOSAIC":
            self.training_type = "mosaic"  # MOSAIC has its own training type with teacher action storage
        elif self.alg_cfg["class_name"] == "FrontRESUnified":
            self.training_type = "frontres"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        elif self.alg_cfg["class_name"] == "SuperviseTrainer":
            self.training_type = "supervise"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")
        self._apply_frontres_specialist_mode()

        # resolve dimensions of observations 观测量维度
        obs, extras = self.env.get_observations()
        obs_dict = extras.get("observations", {})
        if "policy" in obs_dict:
            self.policy_obs_type = "policy"
            obs = obs_dict["policy"]
        else:
            self.policy_obs_type = None
        if "teacher" in obs_dict:
            self.teacher_obs_type = "teacher"
        else:
            self.teacher_obs_type = None
        num_obs = obs.shape[1]

        # resolve type of privileged observations 特权信息
        if self.training_type == "rl":
            if "critic" in obs_dict:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learning, e.g., PPO
            else:
                self.privileged_obs_type = None
        elif self.training_type in ("mosaic", "frontres"):
            # MOSAIC uses critic observations for value function when available.
            # Teacher observations are handled separately for teacher BC.
            has_teacher_obs = "teacher" in obs_dict
            has_critic_obs = "critic" in obs_dict
            if has_critic_obs:
                self.privileged_obs_type = "critic"
                print(f"[{self.alg_cfg['class_name']}] Using 'critic' observations for value estimation.")
            elif has_teacher_obs:
                self.privileged_obs_type = "teacher"
                print(f"[{self.alg_cfg['class_name']}] Using 'teacher' observations for value estimation (no critic obs available).")
            else:
                self.privileged_obs_type = None
        elif self.training_type == "distillation":
            if "teacher" in obs_dict:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None
        elif self.training_type == "supervise":
            if "target" in obs_dict:
                self.privileged_obs_type = "target"
            else:
                self.privileged_obs_type = None

        # resolve type of ref_vel_estimator observations (for MOSAIC with velocity estimator) 速度估计器
        if "ref_vel_estimator" in obs_dict:
            self.ref_vel_estimator_obs_type = "ref_vel_estimator"
            num_ref_vel_estimator_obs = obs_dict["ref_vel_estimator"].shape[1]
            print(f"[Runner] Found 'ref_vel_estimator' observations for velocity estimation (dim={num_ref_vel_estimator_obs}).")
        else:
            self.ref_vel_estimator_obs_type = None

        # resolve dimensions of privileged observations 特权信息维度
        if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
            num_privileged_obs = obs_dict[self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs
        if self.teacher_obs_type is not None and self.teacher_obs_type in obs_dict:
            num_teacher_obs = obs_dict[self.teacher_obs_type].shape[1]
        else:
            num_teacher_obs = None

        # Adjust actor input dimension if using velocity estimator (MOSAIC with estimated ref vel)
        # The actor will receive obs_augmented = [obs, estimated_ref_vel] where estimated_ref_vel is 3D
        # IMPORTANT: Keep num_obs unchanged for normalizer initialization!
        # IMPORTANT: For ResidualActorCritic, do NOT adjust num_actor_obs (it handles estimator internally)
        num_actor_obs = num_obs  # Start with policy obs dimension 动作维度

        # evaluate the policy class (非常危险的做法)
        # eval会将字符串直接作为python代码执行, class_name="ResidualActorCritic"
        # eval会直接将字符串"ResidualActorCritic"变为ResidualActorCritic类的实例
        policy_class = eval(self.policy_cfg.pop("class_name"))

        # Check if using ResidualActorCritic (special handling for estimator dimension)
        is_residual_policy = policy_class in [ResidualActorCritic, FrontRESActorCritic]

        if self.training_type in ("mosaic", "frontres") and self.alg_cfg.get("use_estimate_ref_vel", False):
            if not is_residual_policy:
                # For normal ActorCritic: adjust input dimension to include estimated ref_vel
                num_actor_obs += 3  # Add 3 dimensions for estimated reference velocity (x, y, z)
                print(f"[Runner] Velocity estimator enabled: actor input dimension adjusted to {num_actor_obs} (policy obs {num_obs} + 3D velocity)")
            else:
                # For ResidualActorCritic: keep num_actor_obs unchanged (770)
                # ResidualActorCritic handles estimator internally:
                # - residual_actor uses num_actor_obs (770)
                # - GMT policy uses num_actor_obs + 3 (773)
                print(f"[Runner] Velocity estimator enabled for ResidualActorCritic: residual_actor uses {num_actor_obs} dims, GMT uses {num_actor_obs + 3} dims")
        
        # 选择网络架构 (Actor-Critic是网络架构, PPO是更新算法, AMP是Loss) (Actor-Critic与Teacher-Student可叠加)
        # 无记忆Actor-Critic, 有记忆的Actor-Critic, 无记忆Teacher-Student, 有记忆Teacher-Student
        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_actor_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg).to(self.device)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm 实例化训练方式
        alg_class_name = self.alg_cfg.pop("class_name")
        alg_class = eval(alg_class_name)
        self.alg: PPO | Distillation | MOSAIC | FrontRESUnified = alg_class(
            policy,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        # Track whether task-space FrontRES needs partial obs normalization.
        # Runtime layout for task-space FrontRES is:
        #   [0:num_extra] = anchor-error extras, [num_extra:] = GMT obs.
        # When set, the trailing _frontres_gmt_obs_dim dims are GMT-normalized;
        # leading anchor-error extras optionally use Stage-1 empirical stats.
        self._frontres_gmt_obs_dim: int | None = None
        self._frontres_extra_mean: torch.Tensor | None = None  # (1, K) Stage-1 mean for extra dims
        self._frontres_extra_std:  torch.Tensor | None = None  # (1, K) Stage-1 std  for extra dims

        # Check if using ResidualActorCritic (special handling for GMT normalizer)
        if isinstance(policy, (ResidualActorCritic, FrontRESActorCritic)):
            # Use GMT's frozen normalizer for observations
            if policy.gmt_normalizer is not None:
                self.obs_normalizer = policy.gmt_normalizer
                print(f"[Runner] Using GMT's frozen normalizer for {type(policy).__name__}")
                # Task-space mode: student obs may have extra anchor-error dims beyond
                # what the GMT normalizer expects.  Detect and store the split point.
                if (isinstance(policy, FrontRESActorCritic)
                        and getattr(policy, 'num_task_corrections', 0) > 0):
                    _gmt_mean = getattr(policy.gmt_normalizer, '_mean', None)
                    gmt_norm_dim = _gmt_mean.shape[-1] if _gmt_mean is not None else num_obs
                    if num_obs > gmt_norm_dim:
                        self._frontres_gmt_obs_dim = gmt_norm_dim
                        print(f"[Runner] FrontRES task-space obs layout: first "
                              f"{num_obs - gmt_norm_dim} anchor-error dims pass-through; "
                              f"last {gmt_norm_dim} GMT dims normalized")
            else:
                print("[Runner] WARNING: ResidualActorCritic has no GMT normalizer, using Identity")
                self.obs_normalizer = torch.nn.Identity().to(self.device)

            # Create privileged obs normalizer (for critic)
            if self.empirical_normalization:
                self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], 
                                                                        until=1.0e8).to(self.device)
            else:
                self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

            # Teacher obs normalizer (not used for residual learning)
            self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
        elif self.training_type == "supervise":
            # Student obs: empirical normalization is fine for MLP inputs
            if self.empirical_normalization:
                self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            else:
                self.obs_normalizer = torch.nn.Identity().to(self.device)
            # Target Δq must NOT be normalized: it is in physical units (radians) and will be
            # added directly to q_ref in Stage 2. Normalizing would corrupt the scale.
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)
            self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
        elif self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], 
                                                                    until=1.0e8).to(self.device)
            if num_teacher_obs is not None:
                self.teacher_obs_normalizer = EmpiricalNormalization(shape=[num_teacher_obs], 
                                                                     until=1.0e8).to(self.device)
            else:
                self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # For MOSAIC, use teacher normalizer from checkpoint and freeze it. 教师观测量归一器
        # IMPORTANT: In multi-teacher mode, skip runner-level normalization
        # because each teacher will use its own normalizer in MOSAIC.update()
        if (alg_class_name == "MOSAIC" and self.teacher_obs_type == "teacher"):
            # Check for multi-teacher mode
            if hasattr(self.alg, "teacher_normalizers") and self.alg.teacher_normalizers is not None:
                # Multi-teacher: skip runner-level normalization
                self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
                print("[Runner] Multi-teacher mode: skipping runner-level teacher_obs normalization (each teacher uses its own normalizer)")
            elif hasattr(self.alg, "teacher_normalizer") and self.alg.teacher_normalizer is not None:
                # Single teacher: use teacher's normalizer
                self.teacher_obs_normalizer = self.alg.teacher_normalizer
                self.teacher_obs_normalizer.eval()  # Freeze teacher normalizer
                print("[Runner] Using teacher observation normalizer from checkpoint (frozen)")
            # else: keep the EmpiricalNormalization created above

        # MOSAIC needs these for teacher BC; FrontRESUnified uses them for its
        # own supervised auxiliary loss and checkpoint resume path.
        if alg_class_name in ("MOSAIC", "FrontRESUnified"):
            self.alg.obs_normalizer = self.obs_normalizer
            self.alg.privileged_obs_normalizer = self.privileged_obs_normalizer
            print(f"[Runner] Passed obs_normalizer and privileged_obs_normalizer to {alg_class_name}")

            # Pass environment's group mapping for multi-teacher consistency in MOSAIC
            # and paired motion bookkeeping in FrontRESUnified.
            env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                motion_command = env.command_manager._terms['motion']
                if hasattr(motion_command, 'group_name_to_idx'):
                    self.alg.env_group_name_to_idx = motion_command.group_name_to_idx
                    print(f"[Runner] Passed environment's group mapping to {alg_class_name}: {self.alg.env_group_name_to_idx}")

            # If MOSAIC loaded a teacher critic normalizer, use it for privileged obs
            if hasattr(self.alg, "teacher_critic_normalizer") and self.alg.teacher_critic_normalizer is not None:
                self.privileged_obs_normalizer = self.alg.teacher_critic_normalizer
                print("[Runner] Using teacher critic normalizer for privileged observations")

        # init storage and model
        if self.training_type in ("mosaic", "frontres"):
            # For FrontRESActorCritic in task-space mode, the "action" stored in the
            # rollout buffer is the residual correction [Δpos, Δrpy, c_pos, c_rpy],
            # NOT 29-dim robot joints.
            _mosaic_action_dim = getattr(policy, 'total_output_dim', None) or self.env.num_actions
            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [_mosaic_action_dim],
                teacher_obs_shape=(
                    [num_teacher_obs]
                    if self.training_type == "mosaic" and num_teacher_obs is not None
                    else None
                ),
                ref_vel_estimator_obs_shape=[num_ref_vel_estimator_obs] if self.ref_vel_estimator_obs_type is not None else None,)
        elif self.training_type == "supervise":
            # action_shape must match the supervision target dim (num_privileged_obs),
            # NOT self.env.num_actions (robot DOFs). The target is [Δq(29), Δz(1)] = 30 dims.
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [num_privileged_obs],)
        else:
            # For FrontRESActorCritic in task-space mode, the "policy action" stored in the
            # rollout buffer is the residual correction [Δpos, Δrpy, c_pos, c_rpy],
            # NOT the 29-dim robot joint targets produced by GMT. Use total_output_dim
            # when available.
            _policy_action_dim = getattr(policy, 'total_output_dim', None) or self.env.num_actions
            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [_policy_action_dim],)

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _frontres_exec_score(self, command, return_components: bool = False):
        """Continuous executability score for the frozen GMT tracker.

        This is intentionally narrower than the environment reward. It avoids
        teleop-style terms and actuator penalties, and only measures whether the
        current reference is physically executable while still preserving motion.
        """
        n_envs = command.anchor_pos_w.shape[0]
        dtype = command.anchor_pos_w.dtype

        def cfg_float(name: str, default: float) -> float:
            return float(self.cfg.get(name, default))

        def quat_to_yaw_wxyz(q: torch.Tensor) -> torch.Tensor:
            _, _, yaw = euler_xyz_from_quat(q)
            return yaw

        def wrap_pi(a: torch.Tensor) -> torch.Tensor:
            return torch.atan2(torch.sin(a), torch.cos(a))

        anchor_xy_th = cfg_float("frontres_exec_anchor_xy_threshold", 0.35)
        anchor_yaw_th = cfg_float("frontres_exec_anchor_yaw_threshold", 0.45)
        anchor_xy_vel_std = cfg_float("frontres_exec_anchor_xy_vel_std", 1.0)
        anchor_yaw_rate_std = cfg_float("frontres_exec_anchor_yaw_rate_std", 1.0)

        anchor_xy_err = torch.norm(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
        anchor_xy_score = (1.0 - anchor_xy_err / max(anchor_xy_th, 1e-6)).clamp(-1.0, 1.0)
        anchor_yaw_err = wrap_pi(quat_to_yaw_wxyz(command.anchor_quat_w) - quat_to_yaw_wxyz(command.robot_anchor_quat_w)).abs()
        anchor_yaw_score = (1.0 - anchor_yaw_err / max(anchor_yaw_th, 1e-6)).clamp(-1.0, 1.0)
        anchor_xy_vel_err = torch.square(
            command.anchor_lin_vel_w[:, :2] - command.robot_anchor_lin_vel_w[:, :2]
        ).sum(dim=-1)
        anchor_yaw_rate_err = torch.square(command.anchor_ang_vel_w[:, 2] - command.robot_anchor_ang_vel_w[:, 2])
        anchor_xy_vel_score = torch.exp(
            (-anchor_xy_vel_err / max(anchor_xy_vel_std * anchor_xy_vel_std, 1e-6)).clamp(min=-50.0)
        )
        anchor_yaw_rate_score = torch.exp(
            (-anchor_yaw_rate_err / max(anchor_yaw_rate_std * anchor_yaw_rate_std, 1e-6)).clamp(min=-50.0)
        )

        body_names = list(getattr(command.cfg, "body_names", []))
        foot_names = self.cfg.get(
            "frontres_exec_foot_body_names",
            ["left_ankle_roll_link", "right_ankle_roll_link"],
        )
        foot_idx = [i for i, name in enumerate(body_names) if name in foot_names]
        if len(foot_idx) == 0:
            foot_idx = list(range(command.body_pos_relative_w.shape[1]))
        foot_xy_err = torch.norm(
            command.body_pos_relative_w[:, foot_idx, :2] - command.robot_body_pos_w[:, foot_idx, :2],
            dim=-1,
        )
        foot_z_err = (
            command.body_pos_relative_w[:, foot_idx, 2] - command.robot_body_pos_w[:, foot_idx, 2]
        ).abs()
        foot_z_th = cfg_float("frontres_exec_foot_phase_z_threshold", 0.12)
        foot_gate_temp = cfg_float("frontres_exec_foot_phase_gate_temp", 0.03)
        foot_xy_th = cfg_float("frontres_exec_foot_phase_xy_threshold", 0.25)
        foot_gate = torch.sigmoid((foot_z_th - foot_z_err) / max(foot_gate_temp, 1e-6))
        foot_phase_score_each = (1.0 - foot_xy_err / max(foot_xy_th, 1e-6)).clamp(-1.0, 1.0)
        foot_gate_den = foot_gate.sum(dim=-1).clamp(min=1e-6)
        foot_phase_score = (foot_gate * foot_phase_score_each).sum(dim=-1) / foot_gate_den

        w_xy = cfg_float("frontres_exec_anchor_xy_weight", 1.0)
        w_yaw = cfg_float("frontres_exec_anchor_yaw_weight", 1.0)
        w_xy_vel = cfg_float("frontres_exec_anchor_xy_vel_weight", 0.5)
        w_yaw_rate = cfg_float("frontres_exec_anchor_yaw_rate_weight", 0.5)
        w_foot_phase = cfg_float("frontres_exec_foot_phase_weight", 0.5)
        w_xy_sum = max(w_xy + w_xy_vel + w_foot_phase, 1e-6)
        xy_score = (
            w_xy * anchor_xy_score
            + w_xy_vel * anchor_xy_vel_score
            + w_foot_phase * foot_phase_score
        ) / w_xy_sum
        w_yaw_sum = max(w_yaw + w_yaw_rate, 1e-6)
        yaw_score = (w_yaw * anchor_yaw_score + w_yaw_rate * anchor_yaw_rate_score) / w_yaw_sum
        planar_score = 0.5 * (xy_score + yaw_score)

        anchor_z_th = cfg_float("frontres_exec_anchor_z_threshold", 0.25)
        anchor_ori_th = cfg_float("frontres_exec_anchor_ori_threshold", 0.20)
        ee_z_th = cfg_float("frontres_exec_ee_z_threshold", 0.25)
        anchor_z_err = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]).abs()
        anchor_z_score = (1.0 - anchor_z_err / max(anchor_z_th, 1e-6)).clamp(-1.0, 1.0)

        gravity = getattr(command.robot.data, "GRAVITY_VEC_W", None)
        if gravity is None:
            gravity = torch.zeros(n_envs, 3, device=self.device, dtype=dtype)
            gravity[:, 2] = -1.0
        # Roll/pitch executability must be tied to the reference the frozen GMT
        # is asked to track.  A pure robot-upright margin is mostly independent
        # of the current Δr/Δp action and can give stale or wrong credit during
        # Actor takeover.
        rp_error_rotvec = _quat_to_rotvec_wxyz(
            quat_mul(quat_inv(command.robot_anchor_quat_w), command.anchor_quat_w)
        )
        anchor_rp_err = torch.norm(rp_error_rotvec[:, :2], dim=-1)
        anchor_rp_score = (1.0 - anchor_rp_err / max(anchor_ori_th, 1e-6)).clamp(-1.0, 1.0)

        ee_names = self.cfg.get(
            "frontres_exec_ee_body_names",
            [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        )
        ee_idx = [i for i, name in enumerate(body_names) if name in ee_names]
        if len(ee_idx) == 0:
            ee_idx = list(range(command.body_pos_relative_w.shape[1]))
        ee_z_err = (
            command.body_pos_relative_w[:, ee_idx, 2] - command.robot_body_pos_w[:, ee_idx, 2]
        ).abs().amax(dim=-1)
        ee_z_score = (1.0 - ee_z_err / max(ee_z_th, 1e-6)).clamp(-1.0, 1.0)

        w_z = cfg_float("frontres_exec_anchor_z_weight", 1.0)
        w_ori = cfg_float("frontres_exec_anchor_ori_weight", 1.0)
        w_ee = cfg_float("frontres_exec_ee_z_weight", 1.0)
        w_z_sum = max(w_z + w_ee, 1e-6)
        z_score = (w_z * anchor_z_score + w_ee * ee_z_score) / w_z_sum
        rp_score = anchor_rp_score
        w_stab_sum = max(w_z + w_ori + w_ee, 1e-6)
        vertical_score = (w_z * anchor_z_score + w_ori * rp_score + w_ee * ee_z_score) / w_stab_sum

        vel_body_names = self.cfg.get("frontres_exec_velocity_body_names", None)
        if vel_body_names is None:
            vel_idx = list(range(command.body_lin_vel_w.shape[1]))
        else:
            vel_idx = [i for i, name in enumerate(body_names) if name in vel_body_names]
            if len(vel_idx) == 0:
                vel_idx = list(range(command.body_lin_vel_w.shape[1]))
        lin_std = cfg_float("frontres_exec_body_lin_vel_std", 1.0)
        ang_std = cfg_float("frontres_exec_body_ang_vel_std", 3.14)
        anchor_lin_std = cfg_float("frontres_exec_anchor_lin_vel_std", 1.0)
        lin_err = torch.square(command.body_lin_vel_w[:, vel_idx] - command.robot_body_lin_vel_w[:, vel_idx]).sum(
            dim=-1
        ).mean(dim=-1)
        ang_err = torch.square(command.body_ang_vel_w[:, vel_idx] - command.robot_body_ang_vel_w[:, vel_idx]).sum(
            dim=-1
        ).mean(dim=-1)
        anchor_lin_err = torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w).sum(dim=-1)
        lin_score = torch.exp((-lin_err / max(lin_std * lin_std, 1e-6)).clamp(min=-50.0))
        ang_score = torch.exp((-ang_err / max(ang_std * ang_std, 1e-6)).clamp(min=-50.0))
        anchor_lin_score = torch.exp((-anchor_lin_err / max(anchor_lin_std * anchor_lin_std, 1e-6)).clamp(min=-50.0))
        task_score = (lin_score + ang_score + anchor_lin_score) / 3.0

        planar_weight = cfg_float("frontres_exec_planar_weight", 1.0)
        vertical_weight = cfg_float("frontres_exec_vertical_weight", 0.25)
        task_weight = cfg_float("frontres_exec_task_weight", 0.25)
        score = planar_weight * planar_score + vertical_weight * vertical_score + task_weight * task_score
        score = torch.nan_to_num(score, nan=-1.0, posinf=1.0, neginf=-1.0)
        if return_components:
            return score, {
                "planar": torch.nan_to_num(planar_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "vertical": torch.nan_to_num(vertical_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "xy": torch.nan_to_num(xy_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "yaw": torch.nan_to_num(yaw_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "z": torch.nan_to_num(z_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "rp": torch.nan_to_num(rp_score, nan=-1.0, posinf=1.0, neginf=-1.0),
                "task": torch.nan_to_num(task_score, nan=0.0, posinf=1.0, neginf=0.0),
            }
        return score

    def _frontres_feasible_oracle_exec_score(
        self,
        command,
        start: int,
        count: int,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Executability score after the best correction allowed by the active action cone.

        This is the reward-side oracle, not an action target for the policy.  It
        projects the clean-reference correction into the same feasible cone used
        when applying FrontRES actions.  The cone is determined by
        frontres_active_task_dims:

        * x/y/yaw are repairable only when their action dims are active.
        * roll/pitch are repairable only when their action dims are active.
        * z can move downward freely when active, but upward z is limited to
          jump-time penetration removal to avoid artificial root lift.
        """
        if count <= 0:
            return torch.empty(0, device=self.device)

        needed = (
            "anchor_pos_w_original",
            "anchor_pos_w_raw",
            "anchor_quat_w_original",
            "anchor_quat_w_raw",
            "_frontres_pos_correction",
            "_frontres_quat_correction",
        )
        if not all(hasattr(command, name) for name in needed):
            score = self._frontres_exec_score(command, return_components=return_components)
            if return_components:
                full_score, components = score
                return full_score[start:start + count], {
                    key: value[start:start + count] for key, value in components.items()
                }
            return score[start:start + count]

        env_slice = slice(start, start + count)
        raw_pos = command.anchor_pos_w_raw[env_slice]
        clean_pos = command.anchor_pos_w_original[env_slice]
        raw_quat = command.anchor_quat_w_raw[env_slice]
        clean_quat = command.anchor_quat_w_original[env_slice]

        max_delta_pos = float(getattr(getattr(self.alg, "policy", None), "max_delta_pos", 0.3))
        max_delta_rpy = float(getattr(getattr(self.alg, "policy", None), "max_delta_rpy", 0.1))

        active_dims = getattr(self.alg, "frontres_active_task_dims", self.cfg.get("frontres_active_task_dims", None))
        active_set = None if active_dims is None else {int(dim) for dim in active_dims}

        def _active(dim: int) -> bool:
            return active_set is None or dim in active_set

        oracle_pos = torch.zeros_like(raw_pos)
        dpos_clean = clean_pos - raw_pos
        if _active(0):
            oracle_pos[:, 0] = dpos_clean[:, 0].clamp(-max_delta_pos, max_delta_pos)
        if _active(1):
            oracle_pos[:, 1] = dpos_clean[:, 1].clamp(-max_delta_pos, max_delta_pos)
        dz_clean = clean_pos[:, 2] - raw_pos[:, 2]
        z_upper = torch.zeros_like(dz_clean)
        if hasattr(command, "jump_degree") and hasattr(command, "anchor_penetration_depth"):
            jump_degree = command.jump_degree[env_slice].to(raw_pos.device).to(raw_pos.dtype).clamp(0.0, 1.0)
            penetration = command.anchor_penetration_depth[env_slice].to(raw_pos.device).to(raw_pos.dtype)
            z_upper = (jump_degree * penetration).clamp(max=max_delta_pos)
        if _active(2):
            z_lower = torch.full_like(dz_clean, -max_delta_pos)
            oracle_pos[:, 2] = torch.minimum(torch.maximum(dz_clean, z_lower), z_upper)

        correction_quat = quat_mul(quat_inv(raw_quat), clean_quat)
        correction_rotvec = _quat_to_rotvec_wxyz(correction_quat)
        oracle_rotvec = torch.zeros_like(correction_rotvec)
        for dim in (3, 4, 5):
            if _active(dim):
                axis = dim - 3
                oracle_rotvec[:, axis] = correction_rotvec[:, axis].clamp(-max_delta_rpy, max_delta_rpy)
        oracle_quat = _rotvec_to_quat_wxyz(oracle_rotvec)

        saved_pos = command._frontres_pos_correction[env_slice].clone()
        saved_quat = command._frontres_quat_correction[env_slice].clone()
        saved_body_pos = None
        saved_body_quat = None
        try:
            command._frontres_pos_correction[env_slice].copy_(oracle_pos)
            command._frontres_quat_correction[env_slice].copy_(oracle_quat)
            if hasattr(command, "body_pos_relative_w") and hasattr(command, "body_quat_relative_w"):
                saved_body_pos = command.body_pos_relative_w[env_slice].clone()
                saved_body_quat = command.body_quat_relative_w[env_slice].clone()

                body_count = len(getattr(command.cfg, "body_names", []))
                if body_count > 0:
                    anchor_pos = command.anchor_pos_w[env_slice]
                    anchor_quat = command.anchor_quat_w[env_slice]
                    robot_anchor_pos = command.robot_anchor_pos_w[env_slice]
                    robot_anchor_quat = command.robot_anchor_quat_w[env_slice]
                    body_pos = command.body_pos_w[env_slice]
                    body_quat = command.body_quat_w[env_slice]

                    anchor_pos_repeat = anchor_pos[:, None, :].repeat(1, body_count, 1)
                    anchor_quat_repeat = anchor_quat[:, None, :].repeat(1, body_count, 1)
                    robot_anchor_pos_repeat = robot_anchor_pos[:, None, :].repeat(1, body_count, 1)
                    robot_anchor_quat_repeat = robot_anchor_quat[:, None, :].repeat(1, body_count, 1)

                    delta_pos = robot_anchor_pos_repeat.clone()
                    delta_pos[..., 2] = anchor_pos_repeat[..., 2]
                    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_repeat, quat_inv(anchor_quat_repeat)))
                    command.body_quat_relative_w[env_slice].copy_(quat_mul(delta_ori, body_quat))
                    command.body_pos_relative_w[env_slice].copy_(
                        delta_pos + quat_apply(delta_ori, body_pos - anchor_pos_repeat)
                    )
            if return_components:
                feasible_score_all, feasible_components_all = self._frontres_exec_score(command, return_components=True)
                feasible_score = feasible_score_all[env_slice].clone()
                feasible_components = {
                    key: value[env_slice].clone() for key, value in feasible_components_all.items()
                }
            else:
                feasible_score = self._frontres_exec_score(command)[env_slice].clone()
        finally:
            command._frontres_pos_correction[env_slice].copy_(saved_pos)
            command._frontres_quat_correction[env_slice].copy_(saved_quat)
            if saved_body_pos is not None:
                command.body_pos_relative_w[env_slice].copy_(saved_body_pos)
            if saved_body_quat is not None:
                command.body_quat_relative_w[env_slice].copy_(saved_body_quat)
        if return_components:
            return feasible_score, feasible_components
        return feasible_score

    def _frontres_exec_score_for_modes(
        self,
        components: dict[str, torch.Tensor],
        start: int,
        count: int,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...] | None = None,
    ) -> torch.Tensor:
        """Select executable score components that match each sample's repair cone."""
        if count <= 0:
            return torch.empty(0, device=self.device)
        if mode_groups is None:
            active_modes = tuple(getattr(self, "_frontres_curriculum_active_modes", ()))
            if not active_modes:
                active_dims = getattr(self.alg, "frontres_active_task_dims", self.cfg.get("frontres_active_task_dims", None))
                if active_dims is None:
                    active_modes = ("planar", "yaw", "global_z", "local_rp")
                else:
                    dims = {int(dim) for dim in active_dims}
                    inferred = []
                    if 0 in dims or 1 in dims:
                        inferred.append("planar")
                    if 5 in dims:
                        inferred.append("yaw")
                    if 2 in dims:
                        inferred.append("global_z")
                    if 3 in dims or 4 in dims:
                        inferred.append("local_rp")
                    active_modes = tuple(inferred) if inferred else ("planar", "yaw", "global_z", "local_rp")
            mode_groups = [tuple(active_modes)] * count

        xy = components.get("xy", components["planar"])[start:start + count]
        yaw = components.get("yaw", components["planar"])[start:start + count]
        z = components.get("z", components["vertical"])[start:start + count]
        rp = components.get("rp", components["vertical"])[start:start + count]
        task = components["task"][start:start + count]
        score = torch.zeros(count, device=xy.device, dtype=xy.dtype)
        denom = torch.zeros_like(score)

        planar_weight = float(self.cfg.get("frontres_exec_cone_planar_weight", 1.0))
        yaw_weight = float(self.cfg.get("frontres_exec_cone_yaw_weight", planar_weight))
        vertical_weight = float(self.cfg.get("frontres_exec_cone_vertical_weight", 1.0))
        rp_weight = float(self.cfg.get("frontres_exec_cone_rp_weight", vertical_weight))
        task_weight = float(self.cfg.get("frontres_exec_cone_task_weight", 0.0))
        for idx, modes in enumerate(mode_groups[:count]):
            mode_set = set(modes)
            if "planar" in mode_set:
                score[idx] += planar_weight * xy[idx]
                denom[idx] += planar_weight
            if "yaw" in mode_set:
                score[idx] += yaw_weight * yaw[idx]
                denom[idx] += yaw_weight
            if "global_z" in mode_set:
                score[idx] += vertical_weight * z[idx]
                denom[idx] += vertical_weight
            if "local_rp" in mode_set:
                score[idx] += rp_weight * rp[idx]
                denom[idx] += rp_weight
            if task_weight > 0.0:
                score[idx] += task_weight * task[idx]
                denom[idx] += task_weight
        fallback = 0.25 * (xy + yaw + z + rp)
        score = torch.where(denom > 0.0, score / denom.clamp(min=1e-6), fallback)
        return torch.nan_to_num(score, nan=-1.0, posinf=1.0, neginf=-1.0)

    def _frontres_project_task_target_to_action_cone(self, command, target: torch.Tensor) -> torch.Tensor:
        """Project supervised ΔSE3 targets into the same cone as applied actions.

        The supervised target is the anti-perturbation, but not every
        anti-perturbation is dynamically admissible.  In particular, ordinary
        root sink would require upward Δz, which the runtime projection blocks
        to avoid artificial lift.  Training the actor on that unexecutable
        target creates an actor/reward mismatch, so the warmup and online
        supervised anchor use this projected target instead.
        """
        if target.numel() == 0 or target.shape[-1] < 6:
            return target

        projected = target.clone()
        n = min(projected.shape[0], command._frontres_pos_correction.shape[0])
        if n <= 0:
            return projected

        max_delta_pos = float(getattr(getattr(self.alg, "policy", None), "max_delta_pos", 0.3))
        max_delta_rpy = float(getattr(getattr(self.alg, "policy", None), "max_delta_rpy", 0.1))
        projected[:n, :3] = projected[:n, :3].clamp(-max_delta_pos, max_delta_pos)
        projected[:n, 3:6] = projected[:n, 3:6].clamp(-max_delta_rpy, max_delta_rpy)

        active_dims = getattr(self.alg, "frontres_active_task_dims", self.cfg.get("frontres_active_task_dims", None))
        if active_dims is not None:
            mask = torch.zeros(6, device=projected.device, dtype=projected.dtype)
            for dim in active_dims:
                dim = int(dim)
                if 0 <= dim < 6:
                    mask[dim] = 1.0
            projected[:n, :6] = projected[:n, :6] * mask.view(1, 6)

        z_upper = torch.zeros(n, device=projected.device, dtype=projected.dtype)
        if hasattr(command, "jump_degree") and hasattr(command, "anchor_penetration_depth"):
            jump_degree = command.jump_degree[:n].to(projected.device).to(projected.dtype).clamp(0.0, 1.0)
            penetration = command.anchor_penetration_depth[:n].to(projected.device).to(projected.dtype)
            z_upper = (jump_degree * penetration).clamp(max=max_delta_pos)
        z_lower = torch.full_like(z_upper, -max_delta_pos)
        projected[:n, 2] = torch.minimum(torch.maximum(projected[:n, 2], z_lower), z_upper)
        return projected

    def _frontres_mode_dim_mask(
        self,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a per-env ΔSE3 mask from perturbation families.

        The global action mask says what FrontRES is allowed to output.  This
        per-mode mask says what the current sample actually asks the supervised
        anchor to explain, avoiding planar rollouts supervising z/rp/yaw targets
        that are unrelated to the active perturbation.
        """
        mask = torch.zeros(count, 6, device=device, dtype=dtype)
        for env_i, modes in enumerate(list(mode_groups)[:count]):
            mode_set = set(modes)
            if "planar" in mode_set:
                mask[env_i, 0] = 1.0
                mask[env_i, 1] = 1.0
            if "global_z" in mode_set:
                mask[env_i, 2] = 1.0
            if "local_rp" in mode_set:
                mask[env_i, 3] = 1.0
                mask[env_i, 4] = 1.0
            if "yaw" in mode_set:
                mask[env_i, 5] = 1.0
        return mask

    def _frontres_apply_per_mode_supervised_mask(
        self,
        target: torch.Tensor,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
        count: int,
    ) -> torch.Tensor:
        if target.numel() == 0 or target.shape[-1] < 6 or count <= 0:
            return target
        masked = target.clone()
        n = min(count, masked.shape[0])
        mode_mask = self._frontres_mode_dim_mask(mode_groups, n, masked.device, masked.dtype)
        masked[:n, :6] = masked[:n, :6] * mode_mask
        return masked

    def _maybe_update_frontres_hsl_rollout_target(
        self,
        command,
        actions: torch.Tensor | None,
        dones: torch.Tensor | None,
        current_pos_correction: torch.Tensor | None,
        current_quat_correction: torch.Tensor | None,
        n_train: int,
        n_candidate: int,
        n_base: int,
        n_clean: int,
    ) -> None:
        """Build HSL supervised labels from Clean/Noisy/FEMR rollout states.

        Geometry gives the initial in-cone correction, but the label here is
        corrected by the one-step simulation residual between the Clean rollout
        and the FEMR rollout.  The simulator is still treated as label data:
        gradients flow through the FEMR prediction loss, not through physics.
        """
        if not bool(self.cfg.get("frontres_hsl_rollout_label_enabled", False)):
            return
        if (
            command is None
            or actions is None
            or current_pos_correction is None
            or current_quat_correction is None
            or n_train <= 0
            or n_base <= 0
            or n_clean <= 0
        ):
            return
        if not hasattr(command, "robot") or not hasattr(command.robot, "data"):
            return
        data = command.robot.data
        if not (hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w")):
            return

        n = min(int(n_train), int(n_base), int(n_clean), actions.shape[0])
        if n <= 0:
            return

        base_start = int(n_train + max(0, int(n_candidate)))
        clean_start = int(base_start + n_base)
        device = self.device
        dtype = actions.dtype
        valid_rollout = torch.ones(n, device=device, dtype=dtype)
        if dones is not None:
            done_flat = dones.to(device).view(-1)
            if done_flat.numel() >= clean_start + n:
                done_any = (
                    done_flat[:n].bool()
                    | done_flat[base_start:base_start + n].bool()
                    | done_flat[clean_start:clean_start + n].bool()
                )
                valid_rollout = (~done_any).to(dtype)

        root_pos = data.root_pos_w.to(device=device, dtype=dtype)
        root_quat = data.root_quat_w.to(device=device, dtype=dtype)
        env_raw = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        origins = getattr(getattr(env_raw, "scene", None), "env_origins", None)
        if origins is None:
            origins = torch.zeros_like(root_pos)
        else:
            origins = origins.to(device=device, dtype=dtype)

        fr_pos = root_pos[:n] - origins[:n]
        noisy_pos = root_pos[base_start:base_start + n] - origins[base_start:base_start + n]
        clean_pos = root_pos[clean_start:clean_start + n] - origins[clean_start:clean_start + n]
        fr_quat = root_quat[:n]
        noisy_quat = root_quat[base_start:base_start + n]
        clean_quat = root_quat[clean_start:clean_start + n]

        front_to_clean_r = _quat_to_rotvec_wxyz(quat_mul(quat_inv(fr_quat), clean_quat))
        noisy_to_clean_r = _quat_to_rotvec_wxyz(quat_mul(quat_inv(noisy_quat), clean_quat))
        sim_residual = torch.cat([clean_pos - fr_pos, front_to_clean_r], dim=-1)

        current = torch.zeros(n, 6, device=device, dtype=dtype)
        current[:, :3] = current_pos_correction[:n].to(device=device, dtype=dtype)
        current[:, 3:6] = _quat_to_rotvec_wxyz(
            current_quat_correction[:n].to(device=device, dtype=dtype)
        )

        eta = float(self.cfg.get("frontres_hsl_rollout_eta", 1.0))
        label = current + eta * sim_residual
        label = self._frontres_project_task_target_to_action_cone(command, label)
        if bool(self.cfg.get("frontres_per_mode_supervised_mask", True)):
            mode_groups = list(getattr(
                self,
                "_frontres_curriculum_env_mode_groups",
                [tuple(getattr(self, "_frontres_curriculum_active_modes", ()))] * n,
            ))[:n]
            label = self._frontres_apply_per_mode_supervised_mask(label, mode_groups, n)

        rot_scale = float(self.cfg.get("frontres_hsl_rot_error_scale", 0.25))
        noisy_err = (noisy_pos - clean_pos).norm(dim=-1) + rot_scale * noisy_to_clean_r.norm(dim=-1)
        front_err = (fr_pos - clean_pos).norm(dim=-1) + rot_scale * front_to_clean_r.norm(dim=-1)
        safe_th = float(self.cfg.get("frontres_hsl_safe_threshold", 0.03))
        broken_th = float(self.cfg.get("frontres_hsl_broken_threshold", 0.35))
        safe_tau = max(float(self.cfg.get("frontres_hsl_safe_temperature", 0.01)), 1e-6)
        broken_tau = max(float(self.cfg.get("frontres_hsl_broken_temperature", 0.05)), 1e-6)
        harm_tau = max(float(self.cfg.get("frontres_hsl_harm_temperature", 0.02)), 1e-6)
        safe_w = torch.sigmoid((safe_th - noisy_err) / safe_tau)
        broken_w = torch.sigmoid((noisy_err - broken_th) / broken_tau)
        repair_w = torch.sigmoid((noisy_err - safe_th) / safe_tau) * torch.sigmoid((broken_th - noisy_err) / broken_tau)
        harm_w = torch.sigmoid((front_err - noisy_err) / harm_tau)

        safe_scale = float(self.cfg.get("frontres_hsl_safe_noop_weight", 1.0))
        broken_scale = float(self.cfg.get("frontres_hsl_broken_noop_weight", 1.0))
        harm_scale = float(self.cfg.get("frontres_hsl_harm_noop_weight", 2.0))
        noop_w = safe_scale * safe_w + broken_scale * broken_w + harm_scale * harm_w
        denom = (repair_w + noop_w).clamp(min=1e-6)

        objective = str(getattr(self.alg, "frontres_training_objective", "")).lower()
        task_conf_dim = int(getattr(getattr(self.alg, "policy", None), "task_conf_dim", 2))
        acceptance_hybrid = objective == "hsl_hybrid" and task_conf_dim == 6

        target_full = torch.zeros(self.env.num_envs, 6, device=device, dtype=dtype)
        weight_full = torch.zeros(self.env.num_envs, 1, device=device, dtype=dtype)
        harm_weight_full = torch.zeros(self.env.num_envs, 1, device=device, dtype=dtype)
        max_weight = float(self.cfg.get("frontres_hsl_max_sample_weight", 4.0))
        if acceptance_hybrid:
            # HSL owns the repair direction.  The continuous sample classifier
            # trains the proposal on repairable samples and the harmful loss
            # suppresses unsafe proposals.  The six-dimensional acceptance
            # vector is left to PPO, so the label is not pre-shrunk here.
            target_full[:n] = label
            weight_full[:n, 0] = repair_w.clamp(max=max_weight) * valid_rollout
            harm_weight_full[:n, 0] = noop_w.clamp(max=max_weight) * valid_rollout
        else:
            mixed_label = label * (repair_w / denom).unsqueeze(-1)
            target_full[:n] = mixed_label
            weight_full[:n, 0] = denom.clamp(max=max_weight) * valid_rollout
            harm_weight_full[:n, 0] = harm_w * valid_rollout
        self.alg.transition.supervised_target = target_full
        self.alg.transition.supervised_weight = weight_full
        self.alg.transition.supervised_harm_weight = harm_weight_full

    def _frontres_family_gain_std(
        self,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
        gain: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample gain std from per-family EMA stats, then update stats."""
        if gain.numel() == 0:
            return torch.empty_like(gain)
        init_std = float(self.cfg.get("frontres_family_gain_initial_std", 0.01))
        min_std = float(self.cfg.get("frontres_family_gain_min_std", 0.002))
        alpha = float(self.cfg.get("frontres_family_gain_ema_alpha", 0.05))
        alpha = max(0.0, min(1.0, alpha))
        stats = getattr(self, "_frontres_family_gain_stats", None)
        if stats is None:
            stats = {}
            self._frontres_family_gain_stats = stats

        mode_groups_list = list(mode_groups)[: gain.shape[0]]
        if len(mode_groups_list) < gain.shape[0]:
            fallback_modes = ("planar", "yaw", "global_z", "local_rp")
            mode_groups_list.extend([fallback_modes] * (gain.shape[0] - len(mode_groups_list)))
        std = torch.full_like(gain, max(init_std, min_std))
        for idx, modes in enumerate(mode_groups_list):
            families = tuple(m for m in modes if m in ("planar", "yaw", "global_z", "local_rp"))
            if not families:
                families = ("all",)
            vals = []
            for family in families:
                entry = stats.get(family)
                if entry is None:
                    vals.append(max(init_std, min_std))
                else:
                    vals.append(max(float(entry.get("std", init_std)), min_std))
            std[idx] = sum(vals) / float(len(vals))

        with torch.no_grad():
            gain_detached = gain.detach()
            for family in ("planar", "yaw", "global_z", "local_rp"):
                mask_vals = [
                    family in set(modes)
                    for modes in mode_groups_list
                ]
                if not any(mask_vals):
                    continue
                mask = torch.tensor(mask_vals, device=gain.device, dtype=torch.bool)
                values = gain_detached[mask]
                if values.numel() == 0:
                    continue
                batch_mean = values.mean().item()
                batch_var = values.var(unbiased=False).item() if values.numel() > 1 else 0.0
                entry = stats.get(family)
                if entry is None:
                    mean = batch_mean
                    var = max(batch_var, init_std * init_std)
                else:
                    old_mean = float(entry.get("mean", 0.0))
                    old_var = float(entry.get("var", init_std * init_std))
                    mean = (1.0 - alpha) * old_mean + alpha * batch_mean
                    var = (1.0 - alpha) * old_var + alpha * batch_var
                stats[family] = {
                    "mean": mean,
                    "var": max(var, min_std * min_std),
                    "std": max(math.sqrt(max(var, 0.0)), min_std),
                }
        return std.clamp(min=min_std)

    def _frontres_update_supervised_controller(
        self,
        *,
        loss_dict: dict,
        positive_gain_frac: float | None,
        harm_rate: float | None,
    ) -> None:
        """Decay supervised learning into a one-way anchor once PPO is learnable."""
        if not bool(self.cfg.get("frontres_state_supervised_controller_enabled", True)):
            return
        if not hasattr(self.alg, "lambda_supervised"):
            return
        self.alg.state_supervised_controller_enabled = True
        lam = float(getattr(self.alg, "lambda_supervised", 0.0))
        if lam <= 0.0:
            return

        anchor = float(self.cfg.get(
            "frontres_supervised_anchor_weight",
            self.cfg.get("lambda_supervised_min", 0.02),
        ))
        hold_iters = int(self.cfg.get("frontres_supervised_min_hold_iters", 5))
        seen = int(getattr(self, "_frontres_supervised_controller_seen", 0)) + 1
        self._frontres_supervised_controller_seen = seen
        if seen < max(0, hold_iters):
            return

        pos_trigger = float(self.cfg.get("frontres_supervised_positive_gain_trigger", 0.52))
        harm_limit = float(self.cfg.get("frontres_supervised_harm_limit", 0.06))
        grad_low = float(self.cfg.get("frontres_supervised_grad_cos_low", 0.03))
        decay_good = float(self.cfg.get("frontres_supervised_decay_good", 0.985))
        decay_conflict = float(self.cfg.get("frontres_supervised_decay_conflict", 0.97))
        grad_cos = float(loss_dict.get("grad_cos_ppo_supervised", 0.0))

        learnable = (
            positive_gain_frac is not None
            and harm_rate is not None
            and float(positive_gain_frac) >= pos_trigger
            and float(harm_rate) <= harm_limit
        )
        factor = 1.0
        if learnable:
            factor = min(factor, decay_good)
        if grad_cos < grad_low and positive_gain_frac is not None and float(positive_gain_frac) >= 0.50:
            factor = min(factor, decay_conflict)
        if factor < 1.0:
            setattr(self.alg, "lambda_supervised", max(anchor, lam * factor))


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        print("[Runner] learn() entered — initializing logger...", flush=True)
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        print("[Runner] Logger initialized — starting training setup...", flush=True)
        # Pass writer and log_interval to algorithm for logging (needed by MOSAIC)
        if hasattr(self, 'writer') and self.writer is not None:
            self.alg.writer = self.writer

        self.alg.log_interval = 1 # Default log interval

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # For MOSAIC multi-teacher: ensure env_group_name_to_idx is set before training 整理动作序号
        if self.training_type == "mosaic" and hasattr(self.alg, 'use_multi_teacher') and self.alg.use_multi_teacher:
            if self.alg.env_group_name_to_idx is None:
                print("[Runner] Retrieving environment's group mapping...")
                env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                    motion_command = env.command_manager._terms['motion']
                    if hasattr(motion_command, 'group_name_to_idx'):
                        self.alg.env_group_name_to_idx = motion_command.group_name_to_idx
                        print(f"[Runner] Successfully retrieved environment's group mapping: {self.alg.env_group_name_to_idx}")
                    else:
                        raise RuntimeError(
                            "[Runner] FATAL: motion_command does not have 'group_name_to_idx' attribute!\n"
                            "Multi-teacher training cannot proceed without environment's group mapping.")
                else:
                    raise RuntimeError(
                        "[Runner] FATAL: Cannot retrieve environment's group mapping!\n"
                        f"Environment type: {type(env)}\n"
                        f"Has command_manager: {hasattr(env, 'command_manager')}\n"
                        "Multi-teacher training cannot proceed.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # start learning
        print("[Runner] Getting initial observations...", flush=True)
        obs, extras = self.env.get_observations() # obs.shape=[num_env, 770], 770=[t, t-1, t-2, t-3, t-4]
        print("[Runner] Observations received.", flush=True)
        obs_dict = extras.get("observations", {})
        if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
            obs = obs_dict[self.policy_obs_type]

        # 获取特权信息 & 教师观测
        privileged_obs = obs_dict.get(self.privileged_obs_type, obs) 
        teacher_obs = obs_dict.get(self.teacher_obs_type) # 
        obs = obs.to(self.device)
        privileged_obs = privileged_obs.to(self.device)
        if teacher_obs is not None:
            teacher_obs = teacher_obs.to(self.device)
        else:
            teacher_obs = privileged_obs
        print(
            f"[Runner] Initial obs moved to device: obs={tuple(obs.shape)} {obs.device}, "
            f"priv={tuple(privileged_obs.shape)} {privileged_obs.device}, "
            f"teacher={tuple(teacher_obs.shape)} {teacher_obs.device}",
            flush=True,
        )
        
        # Initialize ref_vel_estimator observations (NO normalization!) 速度估计器
        ref_vel_estimator_obs = obs_dict.get(self.ref_vel_estimator_obs_type)
        if ref_vel_estimator_obs is not None:
            ref_vel_estimator_obs = ref_vel_estimator_obs.to(self.device)
            print(
                f"[Runner] ref_vel_estimator_obs moved: "
                f"{tuple(ref_vel_estimator_obs.shape)} {ref_vel_estimator_obs.device}",
                flush=True,
            )

        # For Stage 1: save raw obs BEFORE obs_normalizer for GMT ONNX input.
        # The exported ONNX includes the normalizer in the computation graph, so
        # get_gmt_action() must receive raw (unnormalized) observations.
        if self.training_type == "supervise":
            obs_raw_for_gmt = obs.clone()

        # Normalize initial observations (same as in training loop) 观测归一器
        print("[Runner] Applying obs normalizer...", flush=True)
        obs = self._apply_obs_normalizer(obs) # 三种观测量分别使用不同观测归一器
        print("[Runner] Policy obs normalized.", flush=True)

        # 使用观测量归一化器对观测量进行处理
        print("[Runner] Applying privileged/teacher normalizers...", flush=True)
        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        print("[Runner] Privileged/teacher obs normalized.", flush=True)

        print("[Runner] Switching modules to train mode...", flush=True)
        self.train_mode() # switch to train mode (for dropout for example)
        print("[Runner] Train mode set.", flush=True)

        # Book keeping
        print("[Runner] Initializing bookkeeping buffers...", flush=True)
        ep_infos = []
        rewbuffer = deque(maxlen=100)  # FrontRES envs: r_delta per episode; others: raw reward
        lenbuffer = deque(maxlen=100)  # FrontRES training envs episode lengths
        # B1: separate GMT baseline reward buffer (only populated when _is_frontres)
        rewbuffer_gmt    = deque(maxlen=100)  # GMT-only envs: raw GMT reward per episode
        lenbuffer_gmt    = deque(maxlen=100)  # GMT-only envs: episode lengths (key diagnostic)
        lenbuffer_gmt_base = deque(maxlen=100)  # Noisy/GMT baseline only
        # In per-env mixed DR mode, frontier search must remain a single-scale
        # probe.  This buffer keeps only the Noisy/GMT baseline samples whose
        # paired training sample was assigned to the frontier class.
        lenbuffer_gmt_base_frontier = deque(maxlen=100)

        # self.env.num_envs: 仿真中同时运行的机器人数量
        # cur_reward_sum & cur_episode_length: 每个机器人的总得分与存活时间
        print("[Runner] Allocating episode reward/length tensors...", flush=True)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        print("[Runner] Episode tensors allocated.", flush=True)

        # create buffers for logging extrinsic and intrinsic rewards
        print("[Runner] Checking RND buffers...", flush=True)
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            print("[Runner] Allocating RND reward buffers...", flush=True)
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            print("[Runner] RND buffers allocated.", flush=True)

        # Velocity estimator error tracking
        vel_est_error_buffer = deque(maxlen=100)
        print("[Runner] Bookkeeping buffers ready.", flush=True)

        # Ensure all parameters are in-synced
        print(f"[Runner] Distributed enabled: {self.is_distributed}", flush=True)
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            print(f"[Runner] Parameter synchronization complete for rank {self.gpu_global_rank}.", flush=True)
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        print("[Runner] Preparing iteration counters...", flush=True)
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        print(f"[Runner] Iteration counters ready: start={start_iter}, total={tot_iter}", flush=True)

        _is_frontres = isinstance(self.alg.policy, FrontRESActorCritic)
        _frontres_training_objective = str(getattr(
            self.alg,
            "frontres_training_objective",
            self.cfg.get("frontres_training_objective", "ppo_hrl"),
        )).lower()
        _frontres_supervised_restore = (
            _is_frontres
            and _frontres_training_objective in ("supervised_restore", "basis_restore")
        )
        _frontres_hsl_restore = (
            _is_frontres
            and _frontres_training_objective in ("supervised_restore", "basis_restore", "hsl_hybrid")
        )

        # Critic warmup: freeze Actor for the first N iterations so the Critic
        # can converge before Actor weights (pretrained from Stage 1) are updated.
        # Only applied to FrontRESActorCritic; other policy types are unaffected.
        if _is_frontres and bool(self.cfg.get("frontres_debug_training", False)):
            def _cfg_set(key, value):
                self.cfg[key] = value

            def _debug_value(debug_key, default):
                # In debug mode we want a genuinely shortened feedback loop.
                # Some Hydra/ConfigClass paths may drop debug_* class defaults
                # from agent_cfg.to_dict(); falling back to formal-training
                # values would silently turn debug into the slow schedule.
                return self.cfg.get(debug_key, default)

            _debug_overrides = {
                "supervised_warmup_iterations": int(_debug_value("debug_supervised_warmup_iterations", 200)),
                "supervised_warmup_diag_interval": int(_debug_value("debug_supervised_warmup_diag_interval", 40)),
                "critic_warmup_iterations": int(_debug_value("debug_critic_warmup_iterations", 50)),
                "dr_scale_init": float(_debug_value("debug_dr_scale_init", 0.5)),
                "dr_min_scale": float(_debug_value("debug_dr_min_scale", 0.3)),
                "dr_ema_alpha": float(_debug_value("debug_dr_ema_alpha", 0.90)),
                "dr_p_gain": float(_debug_value("debug_dr_p_gain", 0.20)),
                "dr_i_gain": float(_debug_value("debug_dr_i_gain", 0.03)),
                "dr_start_ppo_actor_weight": float(_debug_value("debug_dr_start_ppo_actor_weight", 1.0)),
                "frontres_safe_gap_per_step": float(_debug_value("debug_frontres_safe_gap_per_step", 0.003)),
                "frontres_broken_gap_per_step": float(_debug_value("debug_frontres_broken_gap_per_step", 0.08)),
                "frontres_gap_gate_temp": float(_debug_value("debug_frontres_gap_gate_temp", 0.005)),
            }
            for _key, _value in _debug_overrides.items():
                _cfg_set(_key, _value)

            _actor_warmup_debug = int(_debug_value(
                "debug_ppo_actor_warmup_iterations",
                50,
            ))
            _actor_ramp_debug = int(_debug_value(
                "debug_ppo_actor_ramp_iterations",
                200,
            ))
            self.cfg["ppo_actor_warmup_iterations"] = _actor_warmup_debug
            self.cfg["ppo_actor_ramp_iterations"] = _actor_ramp_debug
            self.alg_cfg["ppo_actor_warmup_iterations"] = _actor_warmup_debug
            self.alg_cfg["ppo_actor_ramp_iterations"] = _actor_ramp_debug

            print(
                "[Runner] === FrontRES DEBUG TRAINING enabled ===\n"
                f"[Runner]   supervised_warmup_iterations={self.cfg['supervised_warmup_iterations']}, "
                f"critic_warmup_iterations={self.cfg['critic_warmup_iterations']}\n"
                f"[Runner]   ppo_actor_warmup_iterations={_actor_warmup_debug}, "
                f"ppo_actor_ramp_iterations={_actor_ramp_debug}\n"
                f"[Runner]   dr_scale_init={self.cfg['dr_scale_init']}, "
                f"dr_min_scale={self.cfg['dr_min_scale']}, "
                f"dr_p_gain={self.cfg['dr_p_gain']}, dr_i_gain={self.cfg['dr_i_gain']}, "
                f"dr_start_ppo_actor_weight={self.cfg['dr_start_ppo_actor_weight']}\n"
                f"[Runner]   frontres_safe_gap_per_step={self.cfg['frontres_safe_gap_per_step']}, "
                f"frontres_broken_gap_per_step={self.cfg['frontres_broken_gap_per_step']}, "
                f"frontres_gap_gate_temp={self.cfg['frontres_gap_gate_temp']}",
                flush=True,
            )

        critic_warmup_iters = self.cfg.get("critic_warmup_iterations", 0)

        print("[Runner] Checking FrontRES mode...", flush=True)
        _is_task_space_mode = _is_frontres and getattr(self.alg.policy, 'num_task_corrections', 0) > 0
        print(
            f"[Runner] FrontRES mode: is_frontres={_is_frontres}, "
            f"task_space={_is_task_space_mode}",
            flush=True,
        )
        if _is_frontres and critic_warmup_iters > 0:
            print(f"[Runner] Critic warmup (fixed DR scale, Actor may be held): {critic_warmup_iters} iters")

        def _frontres_ppo_actor_weight_for_iter(iteration: int) -> float:
            """Linear supervised-to-PPO takeover schedule for the current iteration."""
            if not (_is_frontres and hasattr(self.alg, "ppo_actor_weight")):
                return 1.0
            if _frontres_supervised_restore:
                return 0.0
            _actor_warmup = int(self.alg_cfg.get(
                "ppo_actor_warmup_iterations",
                self.cfg.get("ppo_actor_warmup_iterations", 0)))
            _actor_ramp = int(self.alg_cfg.get(
                "ppo_actor_ramp_iterations",
                self.cfg.get("ppo_actor_ramp_iterations", 0)))
            # Use absolute training iteration.  On full resume, start_iter is
            # already the checkpoint iteration; subtracting it would replay the
            # actor warmup/ramp schedule from zero.
            _phase_iter = max(0, iteration)
            if _phase_iter < _actor_warmup:
                return 0.0
            if _actor_ramp > 0 and _phase_iter < _actor_warmup + _actor_ramp:
                _weight = (_phase_iter - _actor_warmup + 1) / float(_actor_ramp)
                return max(0.0, min(1.0, _weight))
            return 1.0

        # B1 quartet ranking reward:
        #   [0:N_train)                         Projected FrontRES on noisy reference
        #   [N_train:N_train+N_candidate)        Candidate/full HSL write
        #   [N_train+N_candidate:N_train+N_candidate+N_base) Noisy GMT
        #   remaining envs                       Clean GMT
        # This exposes the feasible executable gap:
        #   R_feasible_oracle - R_noisy
        # and the repair gain:
        #   R_frontres - R_noisy
        if _is_frontres:
            self.alg.state_supervised_controller_enabled = bool(
                self.cfg.get("frontres_state_supervised_controller_enabled", True)
            )
            _use_quartet_reward = bool(self.cfg.get("frontres_candidate_rollout_enabled", False))
            N_pair = self.env.num_envs // (4 if _use_quartet_reward else 3)
            N_train = N_pair
            if _use_quartet_reward:
                N_candidate = N_pair
                N_base = N_pair
                N_clean = self.env.num_envs - N_train - N_candidate - N_base
                print(
                    f"[Runner] FrontRES B1 quartet reward: "
                    f"{N_train} projected envs + {N_candidate} candidate envs + "
                    f"{N_base} noisy-GMT envs + {N_clean} clean-GMT envs",
                    flush=True,
                )
            else:
                N_candidate = 0
                N_base  = N_pair
                N_clean = self.env.num_envs - N_train - N_base
                print(f"[Runner] FrontRES B1 triplet reward: "
                      f"{N_train} FrontRES envs + {N_base} noisy-GMT envs + {N_clean} clean-GMT envs",
                      flush=True)
            _env_pair = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(_env_pair, 'command_manager') and 'motion' in _env_pair.command_manager._terms:
                _mcmd_pair = _env_pair.command_manager._terms['motion']
                if _use_quartet_reward and hasattr(_mcmd_pair, 'set_frontres_quartet_baseline'):
                    _mcmd_pair.set_frontres_quartet_baseline(N_train, N_candidate, N_base, N_clean)
                    print(
                        "[Runner] FrontRES B1 quartet baseline enabled: "
                        "projected/candidate/noisy-GMT/clean-GMT share motion/frame; clean-GMT has zero perturbation",
                        flush=True,
                    )
                elif hasattr(_mcmd_pair, 'set_frontres_triplet_baseline'):
                    _mcmd_pair.set_frontres_triplet_baseline(N_train, N_base, N_clean)
                    N_candidate = 0
                    _use_quartet_reward = False
                    print("[Runner] FrontRES B1 triplet baseline enabled: "
                          "FrontRES/noisy-GMT/clean-GMT share motion/frame; clean-GMT has zero perturbation",
                          flush=True)
                elif hasattr(_mcmd_pair, 'set_frontres_paired_baseline'):
                    _mcmd_pair.set_frontres_paired_baseline(N_train)
                    print("[Runner] FrontRES B1 paired baseline enabled (legacy two-way fallback): "
                          "env i and env i+N_train share motion/frame/perturbation",
                          flush=True)
            # Separate cumulative reward tracker for GMT envs (raw GMT reward, for logging only).
            # We zero GMT rewards in the PPO storage so V(s) only learns from FrontRES r_delta.
            # GMT raw rewards must be tracked separately before they are zeroed.
            cur_reward_sum_gmt = torch.zeros(N_candidate + N_base + N_clean, dtype=torch.float, device=self.device)

            # B1 triplet baseline:
            #   FrontRES envs: noisy reference + FrontRES correction.
            #   Noisy-GMT envs: same noisy reference, no correction.
            #   Clean-GMT envs: same motion/frame with perturbation disabled.
            # r_delta uses FrontRES minus noisy-GMT; clean-GMT is only a diagnostic
            # and for feasible-gap calibration.  Do not zero perturbations for the
            # noisy-GMT block or the PPO signal degenerates into reward(noisy+Δ)-reward(clean).

        _frontres_task_action_mask = None
        if _is_task_space_mode:
            _active_dims = self.cfg.get("frontres_active_task_dims", None)
            if _active_dims is not None:
                _task_action_dim = int(getattr(self.alg.policy, "total_output_dim", 8))
                _frontres_task_action_mask = torch.zeros(_task_action_dim, device=self.device)
                for _idx in _active_dims:
                    _idx = int(_idx)
                    if not 0 <= _idx < _task_action_dim:
                        raise ValueError(
                            "frontres_active_task_dims contains an index outside the "
                            f"current FrontRES action dim {_task_action_dim}."
                        )
                    _frontres_task_action_mask[_idx] = 1.0
                print(
                    "[Runner] FrontRES task-space action mask enabled: "
                    f"dim={_task_action_dim} mask="
                    f"{_frontres_task_action_mask.detach().cpu().tolist()}",
                    flush=True,
                )

        def _mask_frontres_task_actions(_actions: torch.Tensor) -> torch.Tensor:
            return self._mask_frontres_task_actions(_actions)

        # ── Adaptive DR controller ───────────────────────────────────────────
        # Primary path: boundary sampler keeps perturbations near the repairable
        # GMT boundary using safe/repair/broken/positive-gain diagnostics.
        # Fallback path: legacy r_delta PI before boundary diagnostics exist.
        _dr_max         = float(self.cfg.get("dr_max_scale",    4.0))
        _dr_min         = float(self.cfg.get("dr_min_scale",    0.0))
        _dr_ema_alpha   = float(self.cfg.get("dr_ema_alpha",    0.95))

        # Restore dr_scale from checkpoint (set by load()); start at 0 for fresh runs.
        _dr_scale_init = float(self.cfg.get("dr_scale_init", 0.3))
        _dr_scale     = float(getattr(self, '_dr_scale', _dr_scale_init))
        _dr_scale     = max(_dr_scale, _dr_scale_init)  # enforce floor on resume
        _r_delta_ema  = 0.1  # optimistic: push DR up from dr_init=1.0
        if _is_frontres:
            _frontres_gmt_frontier_safe_low = float(
                getattr(self, "_frontres_gmt_frontier_safe_low", _dr_scale_init)
            )
            _frontres_gmt_frontier_safe_low = max(_dr_min, min(_dr_max, _frontres_gmt_frontier_safe_low))
            self._frontres_gmt_frontier_safe_low = _frontres_gmt_frontier_safe_low
            _frontres_gmt_frontier_broken_high = getattr(self, "_frontres_gmt_frontier_broken_high", None)
            if _frontres_gmt_frontier_broken_high is not None:
                _frontres_gmt_frontier_broken_high = max(
                    _frontres_gmt_frontier_safe_low,
                    min(_dr_max, float(_frontres_gmt_frontier_broken_high)),
                )
            self._frontres_gmt_frontier_broken_high = _frontres_gmt_frontier_broken_high
            self._frontres_gmt_frontier_probe_score = getattr(
                self, "_frontres_gmt_frontier_probe_score", None
            )
            self._frontres_gmt_frontier_decision = getattr(
                self, "_frontres_gmt_frontier_decision", "init"
            )
            self._frontres_gmt_frontier_probe_scale = float(
                getattr(self, "_frontres_gmt_frontier_probe_scale", _frontres_gmt_frontier_safe_low)
            )

        # Read base perturbation values from env config (scaled by dr_scale each iteration).
        # Only ratio/magnitude fields are scaled; prob fields remain constant.
        _perturb_target = None
        if _is_frontres:
            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(_env_raw, 'cfg') and hasattr(_env_raw.cfg, 'motion_perturbations'):
                from types import SimpleNamespace as _NS
                _pt = _env_raw.cfg.motion_perturbations
                _perturb_target = _NS(
                    float_prob           = float(_pt.float_prob),
                    float_ratio          = float(_pt.float_ratio),
                    sink_prob            = float(_pt.sink_prob),
                    sink_ratio           = float(_pt.sink_ratio),
                    foot_slip_prob       = float(_pt.foot_slip_prob),
                    foot_slip_ratio      = float(_pt.foot_slip_ratio),
                    lateral_drift_prob   = float(getattr(_pt, 'lateral_drift_prob',   0.0)),
                    lateral_drift_std    = float(getattr(_pt, 'lateral_drift_std',    0.0)),
                    root_tilt_prob       = float(getattr(_pt, 'root_tilt_prob',       0.0)),
                    root_tilt_max_rad    = float(getattr(_pt, 'root_tilt_max_rad',    0.0)),
                    joint_noise_prob     = float(getattr(_pt, 'joint_noise_prob',     0.0)),
                    joint_noise_std      = float(getattr(_pt, 'joint_noise_std',      0.0)),
                    # IID step-jump base values
                    iid_prob_z           = float(getattr(_pt, 'iid_prob_z',           0.0)),
                    iid_std_z            = float(getattr(_pt, 'iid_std_z',            0.0)),
                    iid_prob_xy          = float(getattr(_pt, 'iid_prob_xy',          0.0)),
                    iid_std_xy           = float(getattr(_pt, 'iid_std_xy',           0.0)),
                    iid_prob_rp          = float(getattr(_pt, 'iid_prob_rp',          0.0)),
                    iid_std_rp           = float(getattr(_pt, 'iid_std_rp',           0.0)),
                    iid_prob_ya          = float(getattr(_pt, 'iid_prob_ya',          0.0)),
                    iid_std_ya           = float(getattr(_pt, 'iid_std_ya',           0.0)),
                    local_root_artifact_prob = float(getattr(_pt, 'local_root_artifact_prob', 0.0)),
                    local_root_artifact_xy_std = float(getattr(_pt, 'local_root_artifact_xy_std', 0.0)),
                    local_root_artifact_yaw_std = float(getattr(_pt, 'local_root_artifact_yaw_std', 0.0)),
                )
                print(
                    f"[Runner] Adaptive DR controller: "
                    f"boundary_enabled={self.cfg.get('frontres_boundary_dr_enabled', True)}, "
                    f"boundary_takeover={self.cfg.get('frontres_boundary_dr_during_actor_takeover', False)}, "
                    f"fallback_PI=(Kp={self.cfg.get('dr_p_gain', 0.10)}, "
                    f"Ki={self.cfg.get('dr_i_gain', 0.01)}, "
                    f"target={self.cfg.get('dr_target_r_delta', 0.01)}), "
                    f"max_scale={_dr_max}, ema_alpha={_dr_ema_alpha}, "
                    f"start_actor_weight={self.cfg.get('dr_start_ppo_actor_weight', 1.0)}, "
                    f"resume dr_scale={_dr_scale:.3f}"
                )
            else:
                print("[Runner] WARNING: FrontRES DR enabled but env.cfg.motion_perturbations not found")

        def _frontres_curriculum_allowed_bases() -> tuple[str, ...]:
            """Map the active FrontRES output dimensions to repairable perturbation families."""
            _all_bases = ("planar", "yaw", "global_z", "local_rp")
            _active_dims = self.cfg.get("frontres_active_task_dims", None)
            if _active_dims is None:
                return _all_bases

            _dims = {int(_idx) for _idx in _active_dims}
            _bases: list[str] = []
            if 0 in _dims or 1 in _dims:
                _bases.append("planar")
            if 5 in _dims:
                _bases.append("yaw")
            if 2 in _dims:
                _bases.append("global_z")
            if 3 in _dims or 4 in _dims:
                _bases.append("local_rp")
            # If only confidence heads are active, keep the full disturbance set
            # so the confidence channels still see meaningful artifact states.
            return tuple(_bases) if _bases else _all_bases

        def _set_frontres_perturbation_curriculum(progress: float, seq_idx: int) -> None:
            """Select which perturbation bases are active for the next rollout.

            Warmup still calls this directly to force one clean mode group at a
            time. PPO rollouts call _sample_frontres_rollout_perturbation_mix()
            so different train envs see different groups in the same iteration.
            """
            _choices, _complexity = _frontres_curriculum_choices(progress, seq_idx)
            _choice = _choices[_choice_hash(seq_idx) % len(_choices)]
            self._frontres_curriculum_active_modes = tuple(_choice)
            self._frontres_curriculum_complexity = _complexity_for_modes(_choice, _complexity)

        def _complexity_for_modes(modes: tuple[str, ...], fallback: str | None = None) -> str:
            if len(modes) == 1:
                return "single"
            if len(modes) == 2:
                return "two"
            if len(modes) == 3:
                return "three"
            return fallback or "full"

        def _frontres_curriculum_choices(progress: float, seq_idx: int) -> tuple[list[tuple[str, ...]], str]:
            _bases = _frontres_curriculum_allowed_bases()
            _specialist_mode = str(self.cfg.get("frontres_specialist_mode", "") or "").lower()
            if _specialist_mode in ("rp", "local_rp", "rp_only", "strong_rp"):
                return [("local_rp",)], "single"
            if _specialist_mode in ("rp_z", "z_rp", "vertical_contact"):
                return [("global_z", "local_rp")], "two"
            if not (_is_frontres and bool(self.cfg.get("frontres_perturbation_curriculum_enabled", True))):
                return [tuple(_bases)], "full"

            _progress = max(0.0, min(1.0, float(progress)))
            _single = [(m,) for m in _bases]
            _canonical_two = [
                ("planar", "yaw"),
                ("planar", "local_rp"),
                ("yaw", "local_rp"),
                ("global_z", "local_rp"),
                ("planar", "global_z"),
                ("yaw", "global_z"),
            ]
            _canonical_three = [
                ("planar", "yaw", "local_rp"),
                ("planar", "global_z", "local_rp"),
                ("yaw", "global_z", "local_rp"),
                ("planar", "yaw", "global_z"),
            ]
            _base_set = set(_bases)
            _two = [m for m in _canonical_two if set(m).issubset(_base_set)]
            _three = [m for m in _canonical_three if set(m).issubset(_base_set)]
            _full = [tuple(_bases)]

            _single_until = float(self.cfg.get("frontres_curriculum_single_until", 0.30))
            _two_until = float(self.cfg.get("frontres_curriculum_two_until", 0.70))
            _full_prob = float(self.cfg.get("frontres_curriculum_full_prob", 0.05))
            _three_prob = float(self.cfg.get("frontres_curriculum_three_prob", 0.10))
            _two_mid_prob = float(self.cfg.get("frontres_curriculum_two_mid_prob", 0.35))
            _two_late_prob = float(self.cfg.get("frontres_curriculum_two_late_prob", 0.40))

            _bucket = (int(seq_idx) * 37) % 1000 / 1000.0
            if bool(self.cfg.get("frontres_adaptive_perturb_curriculum_enabled", True)):
                _stats = getattr(self, "_frontres_boundary_ema", None)
                if _stats is None:
                    _stats = getattr(self, "_last_frontres_boundary_stats", None)
                if _stats is None:
                    return _single, "single"
                _safe = float(_stats.get("safe", 0.0))
                _repair = float(_stats.get("repair", _stats.get("fragile", 0.0)))
                _broken = float(_stats.get("broken", 0.0))
                _gainpos = float(_stats.get("positive_gain", 0.5))
                _safe_hi = float(self.cfg.get("frontres_boundary_safe_high", 0.45))
                _broken_hi = float(self.cfg.get("frontres_boundary_broken_high", 0.35))
                _broken_target = float(self.cfg.get("frontres_boundary_broken_target", 0.25))
                _repair_lo = float(self.cfg.get(
                    "frontres_boundary_repair_low",
                    self.cfg.get("frontres_boundary_fragile_low", 0.45),
                ))
                _repair_hi = float(self.cfg.get(
                    "frontres_boundary_repair_high",
                    self.cfg.get("frontres_boundary_fragile_high", 0.70),
                ))
                _gain_hi = float(self.cfg.get("frontres_boundary_positive_gain_high", 0.55))
                _gain_lo = float(self.cfg.get("frontres_boundary_positive_gain_low", 0.45))

                # State-based curriculum:
                # * broken-heavy or low-gain batches retreat to clean single-family credit assignment;
                # * safe-heavy batches add pairs to reach the repair frontier;
                # * learnable boundary batches add pairs/occasional composites;
                # * otherwise keep mostly single with a little pair exposure.
                if _broken > _broken_hi or (_gainpos < _gain_lo and _broken > _broken_target):
                    _complexity, _choices = "single", _single
                elif _safe > _safe_hi and _broken < _broken_target and _two:
                    if _bucket < 0.65:
                        _complexity, _choices = "two", _two
                    else:
                        _complexity, _choices = "single", _single
                elif (_repair_lo <= _repair <= _repair_hi) and _gainpos > _gain_hi:
                    if _bucket < _full_prob and len(_bases) > 1:
                        _complexity, _choices = "full", _full
                    elif _bucket < _full_prob + max(_three_prob, 0.15) and _three:
                        _complexity, _choices = "three", _three
                    elif _bucket < _full_prob + max(_three_prob, 0.15) + 0.55 and _two:
                        _complexity, _choices = "two", _two
                    else:
                        _complexity, _choices = "single", _single
                else:
                    if _bucket < 0.30 and _two:
                        _complexity, _choices = "two", _two
                    else:
                        _complexity, _choices = "single", _single
                return _choices, _complexity

            if _progress < _single_until:
                _complexity, _choices = "single", _single
            elif _progress < _two_until:
                if _bucket < _two_mid_prob and _two:
                    _complexity, _choices = "two", _two
                else:
                    _complexity, _choices = "single", _single
            else:
                if _bucket < _full_prob and len(_bases) > 1:
                    _complexity, _choices = "full", _full
                elif _bucket < _full_prob + _three_prob and _three:
                    _complexity, _choices = "three", _three
                elif _bucket < _full_prob + _three_prob + _two_late_prob and _two:
                    _complexity, _choices = "two", _two
                else:
                    _complexity, _choices = "single", _single

            return _choices, _complexity

        def _sample_frontres_rollout_perturbation_mix(progress: float, seq_idx: int) -> None:
            """Assign perturbation mode groups across env triplets for one PPO rollout."""
            _choices, _phase_complexity = _frontres_curriculum_choices(progress, seq_idx)
            try:
                _n_train = int(N_train)
                _n_candidate = int(N_candidate)
                _n_base = int(N_base)
                _n_clean = int(N_clean)
            except NameError:
                _n_train = self.env.num_envs
                _n_candidate = 0
                _n_base = 0
                _n_clean = 0
            _groups: list[tuple[str, ...]] = []
            for _env_i in range(max(_n_train, 0)):
                _groups.append(tuple(_choices[_choice_hash(seq_idx * 1009 + _env_i) % len(_choices)]))
            if not _groups:
                _groups = [tuple(_frontres_curriculum_allowed_bases())]
            _active_union = tuple(sorted({mode for group in _groups for mode in group}))
            self._frontres_curriculum_active_modes = _active_union
            _unique_complexities = {_complexity_for_modes(group) for group in _groups}
            if len(_unique_complexities) == 1:
                self._frontres_curriculum_complexity = next(iter(_unique_complexities))
            else:
                self._frontres_curriculum_complexity = "mixed"
            self._frontres_curriculum_env_mode_groups = _groups

            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if not (hasattr(_env_raw, 'command_manager') and 'motion' in _env_raw.command_manager._terms):
                return
            _mcmd = _env_raw.command_manager._terms['motion']
            if not hasattr(_mcmd, "perturber") or not hasattr(_mcmd.perturber, "set_family_env_masks"):
                return

            _family_masks = {
                "planar": torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
                "yaw": torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
                "global_z": torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
                "local_rp": torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
            }
            _candidate_start = _n_train
            _base_start = _n_train + _n_candidate
            for _env_i, _group in enumerate(_groups[:_n_train]):
                for _mode in _group:
                    if _mode in _family_masks:
                        _family_masks[_mode][_env_i] = True
                        if _env_i < _n_candidate:
                            _family_masks[_mode][_candidate_start + _env_i] = True
                        if _env_i < _n_base:
                            _family_masks[_mode][_base_start + _env_i] = True
            # Clean-GMT envs intentionally remain all False; baseline
            # masking in MotionPerturber also keeps them clean.
            _mcmd.perturber.set_family_env_masks(_family_masks)

        if _is_frontres and bool(self.cfg.get("frontres_perturbation_curriculum_enabled", True)):
            _allowed_bases = ",".join(_frontres_curriculum_allowed_bases())
            print(
                "[Runner] Perturbation curriculum enabled: "
                f"bases=[{_allowed_bases}], "
                f"adaptive={self.cfg.get('frontres_adaptive_perturb_curriculum_enabled', True)}, "
                f"single_until={self.cfg.get('frontres_curriculum_single_until', 0.30)}, "
                f"two_until={self.cfg.get('frontres_curriculum_two_until', 0.70)}, "
                f"full_prob={self.cfg.get('frontres_curriculum_full_prob', 0.05)}",
                flush=True,
            )

        def _choice_hash(seq_idx: int) -> int:
            _hash = (int(seq_idx) + 1) & 0xFFFFFFFF
            _hash ^= (_hash >> 16)
            _hash = (_hash * 0x7FEB352D) & 0xFFFFFFFF
            _hash ^= (_hash >> 15)
            _hash = (_hash * 0x846CA68B) & 0xFFFFFFFF
            _hash ^= (_hash >> 16)
            return _hash

        def _set_frontres_curriculum_modes(modes: tuple[str, ...]) -> None:
            self._frontres_curriculum_active_modes = tuple(modes)
            if len(modes) <= 1:
                self._frontres_curriculum_complexity = "single"
            elif len(modes) == 2:
                self._frontres_curriculum_complexity = "two"
            elif len(modes) == 3:
                self._frontres_curriculum_complexity = "three"
            else:
                self._frontres_curriculum_complexity = "full"

        def _frontres_warmup_perturbation_mode_groups(seq_idx: int) -> list[tuple[str, ...]]:
            """Return perturbation families to mix inside one warmup update.

            Warmup should fit all active repair directions together.  Cycling
            single-family samples inside each update avoids serial forgetting
            while keeping the supervised target cleaner than a full composite.
            """
            _bases = _frontres_curriculum_allowed_bases()
            _mode = str(self.cfg.get(
                "frontres_warmup_perturbation_schedule",
                self.cfg.get("supervised_warmup_perturbation_schedule", "mixed_single"),
            ))
            _specialist_mode = str(self.cfg.get("frontres_specialist_mode", "") or "").lower()
            if _specialist_mode in ("rp", "local_rp", "rp_only", "strong_rp"):
                return [("local_rp",)]
            if _specialist_mode in ("rp_z", "z_rp", "vertical_contact"):
                return [("global_z", "local_rp")]
            if _mode == "rl_curriculum":
                _active = tuple(getattr(self, "_frontres_curriculum_active_modes", tuple(_bases)))
                return [_active] if _active else [tuple(_bases)]
            if _mode == "full":
                return [tuple(_bases)]
            if _mode in ("mixed_pair", "balanced_pair"):
                _canonical_two = [
                    ("planar", "yaw"),
                    ("planar", "local_rp"),
                    ("yaw", "local_rp"),
                    ("global_z", "local_rp"),
                    ("planar", "global_z"),
                    ("yaw", "global_z"),
                ]
                _base_set = set(_bases)
                _pairs = [m for m in _canonical_two if set(m).issubset(_base_set)]
                if _pairs:
                    if _mode == "balanced_pair":
                        return [_pairs[_choice_hash(seq_idx) % len(_pairs)]]
                    return _pairs
            if _mode == "single":
                return [(_bases[_choice_hash(seq_idx) % len(_bases)],)]
            if _mode in ("balanced_single", "mixed_single"):
                # Every warmup SGD update contains each active single
                # perturbation family. Each rollout step still has a clean
                # single-family target, but the optimizer update is balanced
                # across all currently repairable directions.
                return [(base,) for base in _bases]
            # Conservative fallback for unknown schedule names: use balanced
            # single-family warmup instead of silently returning a composite.
            return [(base,) for base in _bases]

        def _apply_frontres_dr_scale(scale: float) -> None:
            if not (_is_frontres and _perturb_target is not None):
                return
            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if not (hasattr(_env_raw, 'command_manager') and 'motion' in _env_raw.command_manager._terms):
                return
            _mcmd = _env_raw.command_manager._terms['motion']
            if not hasattr(_mcmd, 'perturber'):
                return

            def _pt(attr, default=0.0):
                return getattr(_perturb_target, attr, default)

            _modes = set(getattr(self, "_frontres_curriculum_active_modes",
                                 ("planar", "yaw", "global_z", "local_rp")))
            _planar = "planar" in _modes
            _yaw = "yaw" in _modes
            _global_z = "global_z" in _modes
            _local_rp = "local_rp" in _modes

            _mcmd.perturber.cfg.float_prob          = _pt('float_prob')          if _global_z else 0.0
            _mcmd.perturber.cfg.float_ratio         = _pt('float_ratio')         * scale
            _mcmd.perturber.cfg.sink_prob           = _pt('sink_prob')           if _global_z else 0.0
            _mcmd.perturber.cfg.sink_ratio          = _pt('sink_ratio')          * scale
            _mcmd.perturber.cfg.foot_slip_prob      = _pt('foot_slip_prob')      if _planar else 0.0
            _mcmd.perturber.cfg.foot_slip_ratio     = _pt('foot_slip_ratio')     * scale
            _mcmd.perturber.cfg.lateral_drift_prob  = _pt('lateral_drift_prob')  if _planar else 0.0
            _mcmd.perturber.cfg.lateral_drift_std   = _pt('lateral_drift_std')   * scale
            _mcmd.perturber.cfg.root_tilt_prob      = _pt('root_tilt_prob')      if _local_rp else 0.0
            _mcmd.perturber.cfg.root_tilt_max_rad   = _pt('root_tilt_max_rad')   * scale
            _mcmd.perturber.cfg.joint_noise_prob    = _pt('joint_noise_prob')
            _mcmd.perturber.cfg.joint_noise_std     = _pt('joint_noise_std')     * scale
            _mcmd.perturber.cfg.iid_prob_z          = _pt('iid_prob_z')          if _global_z else 0.0
            _mcmd.perturber.cfg.iid_std_z           = _pt('iid_std_z')           * scale
            _mcmd.perturber.cfg.iid_prob_xy         = _pt('iid_prob_xy')         if _planar else 0.0
            _mcmd.perturber.cfg.iid_std_xy          = _pt('iid_std_xy')          * scale
            _mcmd.perturber.cfg.iid_prob_rp         = _pt('iid_prob_rp')         if _local_rp else 0.0
            _mcmd.perturber.cfg.iid_std_rp          = _pt('iid_std_rp')          * scale
            _mcmd.perturber.cfg.iid_prob_ya         = _pt('iid_prob_ya')         if _yaw else 0.0
            _mcmd.perturber.cfg.iid_std_ya          = _pt('iid_std_ya')          * scale
            _mcmd.perturber.cfg.local_root_artifact_prob = (
                _pt('local_root_artifact_prob') if (_planar or _yaw) else 0.0
            )
            # Local artifact magnitudes are multiplied by perturber._dr_scale
            # at burst sampling time, so keep cfg as the unscaled base value.
            _mcmd.perturber.cfg.local_root_artifact_xy_std = (
                _pt('local_root_artifact_xy_std') if _planar else 0.0
            )
            _mcmd.perturber.cfg.local_root_artifact_yaw_std = (
                _pt('local_root_artifact_yaw_std') if _yaw else 0.0
            )
            _mcmd.perturber._dr_scale = float(scale)
            if hasattr(_mcmd.perturber, "set_dr_scale_env"):
                _mcmd.perturber.set_dr_scale_env(None)

        def _frontres_mixed_dr_scale(frontier_scale: float, enabled: bool, seq_idx: int) -> tuple[float, str]:
            """Sample an easy/frontier/hard perturbation magnitude around the current frontier."""
            _effective_scale = float(frontier_scale)
            _mix_enabled = bool(enabled) and bool(self.cfg.get("frontres_mixed_dr_strength_enabled", True))
            _mix_mode = "frontier" if _mix_enabled else "fixed"
            if _mix_enabled:
                _easy_w = max(0.0, float(self.cfg.get("frontres_mixed_dr_easy_weight", 0.5)))
                _frontier_w = max(0.0, float(self.cfg.get("frontres_mixed_dr_frontier_weight", 0.4)))
                _hard_w = max(0.0, float(self.cfg.get("frontres_mixed_dr_hard_weight", 0.1)))
                _w_sum = max(_easy_w + _frontier_w + _hard_w, 1e-6)
                _easy_w = _easy_w / _w_sum
                _frontier_w = _frontier_w / _w_sum
                _bucket = (int(seq_idx) * 2654435761 % 1000) / 1000.0
                if _bucket < _easy_w:
                    _mix_mode = "easy"
                    _factor = float(self.cfg.get("frontres_mixed_dr_easy_factor", 0.75))
                elif _bucket < _easy_w + _frontier_w:
                    _mix_mode = "frontier"
                    _factor = float(self.cfg.get("frontres_mixed_dr_frontier_factor", 1.0))
                else:
                    _mix_mode = "hard"
                    _factor = float(self.cfg.get("frontres_mixed_dr_hard_factor", 1.05))
                _effective_scale = float(frontier_scale) * max(0.0, _factor)
            return max(_dr_min, min(_dr_max, _effective_scale)), _mix_mode

        def _frontres_mixed_dr_scale_env(
            frontier_scale: float,
            enabled: bool,
            seq_idx: int,
        ) -> tuple[torch.Tensor | None, str, dict[str, float]]:
            """Sample per-env easy/frontier/hard DR strength for paired rollout branches."""
            _mix_enabled = bool(enabled) and bool(
                self.cfg.get("frontres_mixed_dr_strength_per_env", True)
            )
            if not _mix_enabled:
                return None, "fixed", {"easy": 0.0, "frontier": 1.0, "hard": 0.0, "mean": float(frontier_scale)}
            try:
                _n_train = int(N_train)
                _n_candidate = int(N_candidate)
                _n_base = int(N_base)
            except NameError:
                _n_train = self.env.num_envs
                _n_candidate = 0
                _n_base = 0
            if _n_train <= 0:
                return None, "fixed", {"easy": 0.0, "frontier": 1.0, "hard": 0.0, "mean": float(frontier_scale)}

            _easy_w = max(0.0, float(self.cfg.get("frontres_mixed_dr_easy_weight", 0.5)))
            _frontier_w = max(0.0, float(self.cfg.get("frontres_mixed_dr_frontier_weight", 0.4)))
            _hard_w = max(0.0, float(self.cfg.get("frontres_mixed_dr_hard_weight", 0.1)))
            _w_sum = max(_easy_w + _frontier_w + _hard_w, 1e-6)
            _easy_w /= _w_sum
            _frontier_w /= _w_sum

            _easy_factor = max(0.0, float(self.cfg.get("frontres_mixed_dr_easy_factor", 0.75)))
            _frontier_factor = max(0.0, float(self.cfg.get("frontres_mixed_dr_frontier_factor", 1.0)))
            _hard_factor = max(0.0, float(self.cfg.get("frontres_mixed_dr_hard_factor", 1.05)))
            _frontier_scale = max(_dr_min, min(_dr_max, float(frontier_scale)))
            _train_scales = torch.empty(_n_train, device=self.device, dtype=torch.float32)
            _class = torch.empty(_n_train, device=self.device, dtype=torch.long)
            for _env_i in range(_n_train):
                _bucket = (_choice_hash(seq_idx * 9176 + _env_i * 131 + 17) % 1000) / 1000.0
                if _bucket < _easy_w:
                    _factor = _easy_factor
                    _class[_env_i] = 0
                elif _bucket < _easy_w + _frontier_w:
                    _factor = _frontier_factor
                    _class[_env_i] = 1
                else:
                    _factor = _hard_factor
                    _class[_env_i] = 2
                _train_scales[_env_i] = max(_dr_min, min(_dr_max, _frontier_scale * _factor))

            _scales = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float32)
            _scales[:_n_train] = _train_scales
            _candidate_start = _n_train
            _base_start = _n_train + _n_candidate
            _copy_candidate = min(_n_candidate, _n_train)
            if _copy_candidate > 0:
                _scales[_candidate_start:_candidate_start + _copy_candidate] = _train_scales[:_copy_candidate]
            _copy_base = min(_n_base, _n_train)
            if _copy_base > 0:
                _scales[_base_start:_base_start + _copy_base] = _train_scales[:_copy_base]

            _diag = {
                "easy": float((_class == 0).float().mean().item()),
                "frontier": float((_class == 1).float().mean().item()),
                "hard": float((_class == 2).float().mean().item()),
                "mean": float(_train_scales.mean().item()),
            }
            self._frontres_dr_mix_class_train = _class
            return _scales, "per_env", _diag

        def _apply_frontres_dr_scale_env(scale_vec: torch.Tensor, scalar_fallback: float) -> None:
            if not (_is_frontres and _perturb_target is not None):
                return
            # Write unscaled base magnitudes to cfg; MotionPerturber multiplies
            # them by the per-env scale vector. This avoids double-scaling OU
            # paths while keeping scalar fallback behavior untouched.
            _apply_frontres_dr_scale(1.0)
            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if not (hasattr(_env_raw, 'command_manager') and 'motion' in _env_raw.command_manager._terms):
                return
            _mcmd = _env_raw.command_manager._terms['motion']
            if hasattr(_mcmd, 'perturber') and hasattr(_mcmd.perturber, "set_dr_scale_env"):
                _mcmd.perturber.set_dr_scale_env(scale_vec)

        # ── FrontRES supervised warmup (Stage 1 → Stage 2 merge) ──────────────────
        # Runs BEFORE the PPO loop. Teaches FrontRES to detect perturbations via
        # supervised learning on the anti-DR target:  target = -(perturbed - original).
        # After warmup, PPO fine-tunes with r_delta reward (same architecture, same obs).
        _warmup_iters = int(self.cfg.get("supervised_warmup_iterations", 0))
        if start_iter > 0 and _warmup_iters > 0:
            print(f"[Runner] Resuming from iter {start_iter} — skipping supervised warmup", flush=True)
            _warmup_iters = 0
        if getattr(self, "_frontres_warmup_complete", False) and _warmup_iters > 0:
            print("[Runner] Loaded a completed FrontRES warmup checkpoint — skipping supervised warmup",
                  flush=True)
            _warmup_iters = 0
        if _is_frontres and _warmup_iters > 0:
            _warmup_dr_scale_end = float(self.cfg.get("supervised_warmup_dr_scale", _dr_scale_init))
            _warmup_dr_scale_start = float(self.cfg.get(
                "supervised_warmup_dr_scale_start",
                self.cfg.get("supervised_warmup_dr_scale_min", _warmup_dr_scale_end),
            ))
            _warmup_dr_scale_start = max(0.0, _warmup_dr_scale_start)
            _warmup_dr_scale_end = max(0.0, _warmup_dr_scale_end)
            _warmup_lr     = float(self.cfg.get("supervised_warmup_lr", 1e-4))
            _warmup_epochs = int(self.cfg.get("supervised_warmup_epochs", 5))
            _warmup_steps  = int(self.cfg.get("supervised_warmup_steps_per_iter", self.num_steps_per_env))
            _warmup_steps  = max(1, min(_warmup_steps, self.num_steps_per_env))
            _warmup_max_envs = int(self.cfg.get("supervised_warmup_max_envs_per_step", self.env.num_envs))
            _warmup_max_envs = max(1, min(_warmup_max_envs, self.env.num_envs))
            _warmup_valid_w = float(getattr(self.alg, "supervised_valid_loss_weight", 4.0))
            _warmup_dir_w = float(getattr(self.alg, "supervised_direction_loss_weight", 0.1))
            _warmup_energy_w = float(self.cfg.get("frontres_warmup_energy_loss_weight", 1.0))
            _warmup_diag_interval = int(self.cfg.get(
                "supervised_warmup_diag_interval", max(1, _warmup_iters // 5)))
            _warmup_diag_interval = max(1, _warmup_diag_interval)
            _warmup_opt = torch.optim.Adam(
                list(self.alg.policy.residual_actor.parameters())
                + list(self.alg.policy.critic.parameters()),
                lr=_warmup_lr,
            )

            # Import once to avoid per-step overhead
            from whole_body_tracking.tasks.tracking.mdp.observations import \
                get_supervision_target_task_space as _get_warmup_target

            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            _nfo = self.alg.policy.num_frontres_obs
            if _nfo <= 0:
                _nfo = self.alg.policy.num_actor_obs  # use full obs when no subset configured

            _warmup_dr_desc = (
                f"{_warmup_dr_scale_start}->{_warmup_dr_scale_end}"
                if abs(_warmup_dr_scale_end - _warmup_dr_scale_start) > 1e-8
                else f"{_warmup_dr_scale_end}"
            )
            print(f"[Runner] === Joint warmup: {_warmup_iters} iters "
                  f"(dr_scale={_warmup_dr_desc}, lr={_warmup_lr}, epochs={_warmup_epochs}, "
                  f"steps_per_iter={_warmup_steps}, "
                  f"max_envs_per_step={_warmup_max_envs}, "
                  f"frontres_input={_nfo} dims, energy_w={_warmup_energy_w}, "
                  f"perturb_schedule={self.cfg.get('supervised_warmup_perturbation_schedule', self.cfg.get('frontres_warmup_perturbation_schedule', 'mixed_single'))}) ===",
                  flush=True)

            for _wu in range(_warmup_iters):
                if _warmup_iters > 1:
                    _warmup_frac = _wu / float(_warmup_iters - 1)
                else:
                    _warmup_frac = 1.0
                # Smooth curriculum: spend early warmup on learnable direction,
                # then expose the critic/actor to near-boundary perturbations.
                _warmup_frac = _warmup_frac * _warmup_frac * (3.0 - 2.0 * _warmup_frac)
                _warmup_dr_scale = (
                    _warmup_dr_scale_start
                    + (_warmup_dr_scale_end - _warmup_dr_scale_start) * _warmup_frac
                )
                _set_frontres_perturbation_curriculum(_warmup_frac, _wu)
                _warmup_mode_groups = _frontres_warmup_perturbation_mode_groups(_wu)
                if not _warmup_mode_groups:
                    _warmup_mode_groups = [tuple(_frontres_curriculum_allowed_bases())]

                _wo_list: list[torch.Tensor] = []
                _wt_list: list[torch.Tensor] = []
                _wc_list: list[torch.Tensor] = []
                _we_list: list[torch.Tensor] = []

                # Use no_grad rather than inference_mode: warmup samples are
                # later fed back through trainable actor/critic networks, and
                # some PyTorch versions reject inference tensors in backward.
                with torch.no_grad():
                    for _step in range(_warmup_steps):
                        _mode_group = _warmup_mode_groups[
                            (_wu * max(_warmup_steps, 1) + _step) % len(_warmup_mode_groups)
                        ]
                        _set_frontres_curriculum_modes(tuple(_mode_group))
                        _apply_frontres_dr_scale(_warmup_dr_scale)
                        obs, extras = self.env.get_observations()
                        obs_dict = extras.get("observations", {})
                        _p_obs_raw = obs_dict.get(self.policy_obs_type, obs).to(self.device)
                        _p_obs = self._apply_obs_normalizer(_p_obs_raw)

                        # GMT-only actions (FrontRES correction = zero during warmup)
                        env_actions = self.alg.policy.get_env_action(
                            _p_obs,
                            torch.zeros(_p_obs.shape[0], self.alg.policy.total_output_dim,
                                        device=self.device))

                        obs, rewards_wu, dones, extras = self.env.step(env_actions.to(self.env.device))
                        obs_dict = extras.get("observations", {})
                        _p_obs_raw = obs_dict.get(self.policy_obs_type, obs).to(self.device)
                        _p_obs = self._apply_obs_normalizer(_p_obs_raw)
                        if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
                            _c_obs = self.privileged_obs_normalizer(
                                obs_dict[self.privileged_obs_type].to(self.device)
                            )
                        else:
                            _c_obs = _p_obs
                        _target = _get_warmup_target(_env_raw, "motion").to(self.device)
                        _mcmd_wu = _env_raw.command_manager._terms.get("motion")
                        if _mcmd_wu is not None:
                            _target = self._frontres_project_task_target_to_action_cone(_mcmd_wu, _target)
                        if "N_train" in locals() and N_train > 0 and N_base > 0 and N_clean > 0:
                            _n_energy = min(N_train, N_base, N_clean)
                            if _mcmd_wu is not None:
                                _, _exec_wu_components = self._frontres_exec_score(_mcmd_wu, return_components=True)
                                _wu_modes = [
                                    tuple(getattr(self, "_frontres_curriculum_active_modes", ()))
                                ] * _n_energy
                                _r_perturbed_wu = self._frontres_exec_score_for_modes(
                                    _exec_wu_components, N_train, _n_energy, _wu_modes
                                ).view(-1)
                                _, _feasible_wu_components = self._frontres_feasible_oracle_exec_score(
                                    _mcmd_wu, N_train, _n_energy, return_components=True
                                )
                                _r_feasible_wu = self._frontres_exec_score_for_modes(
                                    _feasible_wu_components, 0, _n_energy, _wu_modes
                                ).to(self.device).view(-1)
                                _energy_target = (_r_feasible_wu - _r_perturbed_wu).clamp(min=0.0).unsqueeze(-1)
                            else:
                                _energy_target = torch.zeros(_n_energy, 1, device=self.device)
                            _p_obs = _p_obs[:_n_energy]
                            _c_obs = _c_obs[:_n_energy]
                            _target = _target[:_n_energy]
                        else:
                            _energy_target = torch.zeros(_p_obs.shape[0], 1, device=self.device)

                        if _warmup_max_envs < _p_obs.shape[0]:
                            _sample_ids = torch.randperm(
                                _p_obs.shape[0], device=self.device)[:_warmup_max_envs]
                            _p_obs = _p_obs[_sample_ids]
                            _c_obs = _c_obs[_sample_ids]
                            _target = _target[_sample_ids]
                            _energy_target = _energy_target[_sample_ids]

                        # Both obs and target reflect the perturbation applied this step
                        _wo_list.append(_p_obs[:, :_nfo])
                        _wt_list.append(_target)
                        _wc_list.append(_c_obs)
                        _we_list.append(_energy_target)

                # Joint SGD over collected rollout data:
                #   Actor:  Δ ≈ -noise
                #   Critic: E(s_perturbed) ≈ max(R_exec_feasible_oracle - R_exec_perturbed, 0)
                _all_obs = torch.cat(_wo_list, dim=0)      # (S*E, _nfo)
                _all_tgt = torch.cat(_wt_list, dim=0)      # (S*E, 6)
                _all_critic_obs = torch.cat(_wc_list, dim=0)
                _all_energy = torch.cat(_we_list, dim=0)
                _N = _all_obs.shape[0]
                _last_actor_loss = torch.tensor(0.0, device=self.device)
                _last_energy_loss = torch.tensor(0.0, device=self.device)
                _sup_mask = None
                _active_sup_dims = getattr(self.alg, "frontres_active_task_dims", None)
                if _active_sup_dims is not None:
                    _sup_mask = torch.zeros(_all_tgt.shape[-1], device=self.device, dtype=_all_tgt.dtype)
                    for _dim in _active_sup_dims:
                        _dim = int(_dim)
                        if 0 <= _dim < _sup_mask.numel():
                            _sup_mask[_dim] = 1.0
                    if _wu == 0:
                        print(
                            "[Runner] Joint warmup supervised active mask: "
                            f"{[float(x) for x in _sup_mask.detach().cpu().tolist()]} "
                            "(dx,dy,dz,droll,dpitch,dyaw)",
                            flush=True,
                        )

                for epoch in range(_warmup_epochs):
                    perm = torch.randperm(_N, device=self.device)
                    for i in range(0, _N, 4096):
                        idx = perm[i:i + 4096]
                        pred = self.alg.policy.residual_actor(_all_obs[idx])
                        # pred: [N, 8] (Δpos_raw, Δrpy_raw, conf_raw)
                        # tgt:  [N, 6] (Δpos, Δrpy) — compare pos/rpy only
                        if getattr(self.alg.policy, 'num_task_corrections', 0) > 0:
                            pred_sup = torch.cat([
                                torch.tanh(pred[:, :3]) * self.alg.policy.max_delta_pos,
                                torch.tanh(pred[:, 3:6]) * self.alg.policy.max_delta_rpy,
                            ], dim=-1)
                            target_sup = torch.cat([
                                _all_tgt[idx, :3].clamp(
                                    -self.alg.policy.max_delta_pos, self.alg.policy.max_delta_pos),
                                _all_tgt[idx, 3:].clamp(
                                    -self.alg.policy.max_delta_rpy, self.alg.policy.max_delta_rpy),
                            ], dim=-1)
                        else:
                            pred_sup = pred[:, :_all_tgt.shape[-1]]
                            target_sup = _all_tgt[idx]
                        if _sup_mask is not None:
                            pred_sup = pred_sup * _sup_mask.view(1, -1)
                            target_sup = target_sup * _sup_mask.view(1, -1)

                        target_norm = target_sup.norm(dim=-1)
                        valid = target_norm > 1e-4
                        pos_valid = target_sup[:, :3].norm(dim=-1) > 1e-4
                        rpy_valid = target_sup[:, 3:].norm(dim=-1) > 1e-4
                        pos_weight = torch.ones_like(target_norm)
                        rpy_weight = torch.ones_like(target_norm)
                        if pos_valid.any():
                            pos_weight[pos_valid] = _warmup_valid_w
                        if rpy_valid.any():
                            rpy_weight[rpy_valid] = _warmup_valid_w
                        pos_weight = pos_weight / pos_weight.mean().clamp(min=1e-6)
                        rpy_weight = rpy_weight / rpy_weight.mean().clamp(min=1e-6)

                        pos_err = torch.nn.functional.huber_loss(
                            pred_sup[:, :3], target_sup[:, :3].detach(), reduction="none").mean(dim=-1)
                        rpy_err = torch.nn.functional.huber_loss(
                            pred_sup[:, 3:], target_sup[:, 3:].detach(), reduction="none").mean(dim=-1)
                        _rpy_w = float(getattr(self.alg, 'supervised_rpy_loss_weight', 1.0))
                        loss = (pos_err * pos_weight).mean() + _rpy_w * (rpy_err * rpy_weight).mean()
                        if _warmup_dir_w > 0.0:
                            direction_loss = torch.zeros((), device=self.device)
                            if pos_valid.any():
                                direction_loss = direction_loss + (
                                    1.0 - torch.nn.functional.cosine_similarity(
                                        pred_sup[pos_valid, :3],
                                        target_sup[pos_valid, :3].detach(),
                                        dim=-1,
                                    ).mean()
                                )
                            if rpy_valid.any():
                                direction_loss = direction_loss + (
                                    1.0 - torch.nn.functional.cosine_similarity(
                                        pred_sup[rpy_valid, 3:],
                                        target_sup[rpy_valid, 3:].detach(),
                                        dim=-1,
                                    ).mean()
                                )
                            loss = loss + _warmup_dir_w * direction_loss
                        _conf_w = float(getattr(self.alg, 'supervised_conf_loss_weight', 0.0))
                        if (
                            getattr(self.alg.policy, 'num_task_corrections', 0) > 0
                            and pred.shape[-1] >= 8
                            and _conf_w > 0
                            and int(getattr(self.alg.policy, "task_conf_dim", 2)) == 2
                        ):
                            target_conf = valid.view(-1, 1).to(pred.dtype)
                            conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                pred[:, 6:8], target_conf.expand(-1, 2))
                            loss = loss + _conf_w * conf_loss
                        actor_loss = loss
                        value_pred = self.alg.policy.evaluate(_all_critic_obs[idx])
                        energy_loss = torch.nn.functional.huber_loss(
                            value_pred, _all_energy[idx].detach(), reduction="mean"
                        )
                        loss = actor_loss + _warmup_energy_w * energy_loss
                        _warmup_opt.zero_grad()
                        loss.backward()
                        _warmup_opt.step()
                        _last_actor_loss = actor_loss.detach()
                        _last_energy_loss = energy_loss.detach()

                if (_wu + 1) % _warmup_diag_interval == 0 or (_wu + 1) == _warmup_iters:
                    with torch.inference_mode():
                        _valid_all = _all_tgt.norm(dim=-1) > 1e-4
                        _pred_all_raw = self.alg.policy.residual_actor(_all_obs[:, :_nfo])
                        if getattr(self.alg.policy, 'num_task_corrections', 0) > 0:
                            _pred_all = torch.cat([
                                torch.tanh(_pred_all_raw[:, :3]) * self.alg.policy.max_delta_pos,
                                torch.tanh(_pred_all_raw[:, 3:6]) * self.alg.policy.max_delta_rpy,
                            ], dim=-1)
                            _target_all = torch.cat([
                                _all_tgt[:, :3].clamp(
                                    -self.alg.policy.max_delta_pos, self.alg.policy.max_delta_pos),
                                _all_tgt[:, 3:].clamp(
                                    -self.alg.policy.max_delta_rpy, self.alg.policy.max_delta_rpy),
                            ], dim=-1)
                        else:
                            _pred_all = _pred_all_raw[:, :_all_tgt.shape[-1]]
                            _target_all = _all_tgt
                        if _sup_mask is not None:
                            _pred_all = _pred_all * _sup_mask.view(1, -1)
                            _target_all = _target_all * _sup_mask.view(1, -1)

                        _valid_all = _target_all.norm(dim=-1) > 1e-4
                        _valid_pos = _target_all[:, :3].norm(dim=-1) > 1e-4
                        _valid_rpy = _target_all[:, 3:].norm(dim=-1) > 1e-4

                        def _masked_cos(a, b, mask):
                            if mask.any():
                                return torch.nn.functional.cosine_similarity(
                                    a[mask], b[mask], dim=-1).mean().item()
                            return 0.0

                        def _masked_mae(a, b, mask):
                            if mask.any():
                                return (a[mask] - b[mask]).abs().mean().item()
                            return 0.0

                        def _masked_norm(a, mask):
                            if mask.any():
                                return a[mask].norm(dim=-1).mean().item()
                            return 0.0

                        def _masked_abs_mean(a, mask):
                            if mask.any():
                                return a[mask].abs().mean().item()
                            return 0.0

                        def _sign_agreement(a, b, mask):
                            if mask.any():
                                return ((a[mask] * b[mask]) > 0.0).float().mean().item()
                            return 0.0

                        if _valid_all.any():
                            _warmup_cos = torch.nn.functional.cosine_similarity(
                                _pred_all[_valid_all], _target_all[_valid_all], dim=-1).mean().item()
                        else:
                            _warmup_cos = 0.0
                        _valid_frac = _valid_all.float().mean().item()
                        _valid_pos_frac = _valid_pos.float().mean().item()
                        _valid_rpy_frac = _valid_rpy.float().mean().item()
                        _cos_pos = _masked_cos(_pred_all[:, :3], _target_all[:, :3], _valid_pos)
                        _cos_rpy = _masked_cos(_pred_all[:, 3:], _target_all[:, 3:], _valid_rpy)
                        _valid_roll = _target_all[:, 3].abs() > 1e-4
                        _valid_pitch = _target_all[:, 4].abs() > 1e-4
                        _valid_yaw = _target_all[:, 5].abs() > 1e-4
                        _sign_roll = _sign_agreement(_pred_all[:, 3], _target_all[:, 3], _valid_roll)
                        _sign_pitch = _sign_agreement(_pred_all[:, 4], _target_all[:, 4], _valid_pitch)
                        _sign_yaw = _sign_agreement(_pred_all[:, 5], _target_all[:, 5], _valid_yaw)
                        _abs_tgt_roll = _masked_abs_mean(_target_all[:, 3], _valid_roll)
                        _abs_tgt_pitch = _masked_abs_mean(_target_all[:, 4], _valid_pitch)
                        _abs_tgt_yaw = _masked_abs_mean(_target_all[:, 5], _valid_yaw)
                        _abs_pred_roll = _masked_abs_mean(_pred_all[:, 3], _valid_roll)
                        _abs_pred_pitch = _masked_abs_mean(_pred_all[:, 4], _valid_pitch)
                        _abs_pred_yaw = _masked_abs_mean(_pred_all[:, 5], _valid_yaw)
                        _valid_roll_frac = _valid_roll.float().mean().item()
                        _valid_pitch_frac = _valid_pitch.float().mean().item()
                        _valid_yaw_frac = _valid_yaw.float().mean().item()
                        _valid_x = _target_all[:, 0].abs() > 1e-4
                        _valid_y = _target_all[:, 1].abs() > 1e-4
                        _valid_z = _target_all[:, 2].abs() > 1e-4
                        _valid_x_frac = _valid_x.float().mean().item()
                        _valid_y_frac = _valid_y.float().mean().item()
                        _valid_z_frac = _valid_z.float().mean().item()
                        _mae_pos = _masked_mae(_pred_all[:, :3], _target_all[:, :3], _valid_pos)
                        _mae_rpy = _masked_mae(_pred_all[:, 3:], _target_all[:, 3:], _valid_rpy)
                        _pred_pos_norm = _masked_norm(_pred_all[:, :3], _valid_pos)
                        _tgt_pos_norm = _masked_norm(_target_all[:, :3], _valid_pos)
                        _pred_rpy_norm = _masked_norm(_pred_all[:, 3:], _valid_rpy)
                        _tgt_rpy_norm = _masked_norm(_target_all[:, 3:], _valid_rpy)
                        _obs_pos_best_cos = 0.0
                        _obs_rpy_best_cos = 0.0
                        _obs_rpy_best_neg_cos = 0.0
                        _obs_rpy_best_norm = 0.0
                        _obs_z_best_sign = 0.0
                        _obs_roll_best_sign = 0.0
                        _obs_pitch_best_sign = 0.0
                        _obs_z_best_corr = 0.0
                        _obs_roll_best_corr = 0.0
                        _obs_pitch_best_corr = 0.0
                        # FrontRES-only anchor error observations occupy the first 30 dims
                        # after _apply_obs_normalizer(): [extra | normalized_gmt].  IsaacLab
                        # history flattening can be either term-blocked
                        #   [pos_hist(5*3), rpy_hist(5*3)]
                        # or frame-interleaved
                        #   [[pos(3), rpy(3)] * 5],
                        # depending on the obs manager version/config.  Check both layouts so
                        # the diagnostic itself does not become another moving target.
                        if _all_obs.shape[-1] >= 30:
                            _extra = _all_obs[:, :30]
                            _target_pos = _target_all[:, :3]
                            _target_rpy = _target_all[:, 3:]

                            def _scalar_corr(a, b, mask):
                                if mask.any():
                                    a_m = a[mask] - a[mask].mean()
                                    b_m = b[mask] - b[mask].mean()
                                    return (a_m * b_m).mean() / (
                                        a_m.std(unbiased=False) * b_m.std(unbiased=False)
                                    ).clamp(min=1e-6)
                                return torch.tensor(0.0, device=self.device)

                            def _scalar_sign(a, b, mask):
                                if mask.any():
                                    return ((a[mask] * b[mask]) > 0.0).float().mean()
                                return torch.tensor(0.0, device=self.device)

                            def _score_extra_layout(_pos_frames, _rpy_frames):
                                _pos_cos_vals = []
                                _rpy_cos_vals = []
                                _rpy_neg_cos_vals = []
                                _rpy_norm_vals = []
                                _z_sign_vals = []
                                _roll_sign_vals = []
                                _pitch_sign_vals = []
                                _z_corr_vals = []
                                _roll_corr_vals = []
                                _pitch_corr_vals = []
                                for _hist_i in range(_pos_frames.shape[1]):
                                    _pos_mask_i = _valid_pos & (_pos_frames[:, _hist_i].norm(dim=-1) > 1e-4)
                                    _rpy_mask_i = _valid_rpy & (_rpy_frames[:, _hist_i].norm(dim=-1) > 1e-4)
                                    if _pos_mask_i.any():
                                        _pos_cos_vals.append(torch.nn.functional.cosine_similarity(
                                            _pos_frames[_pos_mask_i, _hist_i],
                                            _target_pos[_pos_mask_i],
                                            dim=-1,
                                        ).mean())
                                    if _rpy_mask_i.any():
                                        _obs_rpy_i = _rpy_frames[_rpy_mask_i, _hist_i]
                                        _target_rpy_i = _target_rpy[_rpy_mask_i]
                                        _rpy_cos_vals.append(torch.nn.functional.cosine_similarity(
                                            _obs_rpy_i,
                                            _target_rpy_i,
                                            dim=-1,
                                        ).mean())
                                        _rpy_neg_cos_vals.append(torch.nn.functional.cosine_similarity(
                                            -_obs_rpy_i,
                                            _target_rpy_i,
                                            dim=-1,
                                        ).mean())
                                        _rpy_norm_vals.append(_obs_rpy_i.norm(dim=-1).mean())
                                    _z_mask_i = _target_pos[:, 2].abs() > 1e-4
                                    _roll_mask_i = _target_rpy[:, 0].abs() > 1e-4
                                    _pitch_mask_i = _target_rpy[:, 1].abs() > 1e-4
                                    _z_sign_vals.append(_scalar_sign(
                                        _pos_frames[:, _hist_i, 2], _target_pos[:, 2], _z_mask_i))
                                    _roll_sign_vals.append(_scalar_sign(
                                        _rpy_frames[:, _hist_i, 0], _target_rpy[:, 0], _roll_mask_i))
                                    _pitch_sign_vals.append(_scalar_sign(
                                        _rpy_frames[:, _hist_i, 1], _target_rpy[:, 1], _pitch_mask_i))
                                    _z_corr_vals.append(_scalar_corr(
                                        _pos_frames[:, _hist_i, 2], _target_pos[:, 2], _z_mask_i))
                                    _roll_corr_vals.append(_scalar_corr(
                                        _rpy_frames[:, _hist_i, 0], _target_rpy[:, 0], _roll_mask_i))
                                    _pitch_corr_vals.append(_scalar_corr(
                                        _rpy_frames[:, _hist_i, 1], _target_rpy[:, 1], _pitch_mask_i))
                                _pos_cos = torch.stack(_pos_cos_vals).max() if _pos_cos_vals else torch.tensor(0.0, device=self.device)
                                _rpy_cos = torch.stack(_rpy_cos_vals).max() if _rpy_cos_vals else torch.tensor(0.0, device=self.device)
                                _rpy_neg_cos = (
                                    torch.stack(_rpy_neg_cos_vals).max()
                                    if _rpy_neg_cos_vals else torch.tensor(0.0, device=self.device)
                                )
                                _rpy_norm = (
                                    torch.stack(_rpy_norm_vals).max()
                                    if _rpy_norm_vals else torch.tensor(0.0, device=self.device)
                                )
                                _z_sign = torch.stack(_z_sign_vals).max()
                                _roll_sign = torch.stack(_roll_sign_vals).max()
                                _pitch_sign = torch.stack(_pitch_sign_vals).max()
                                _z_corr = torch.stack(_z_corr_vals).max()
                                _roll_corr = torch.stack(_roll_corr_vals).max()
                                _pitch_corr = torch.stack(_pitch_corr_vals).max()
                                return (
                                    _pos_cos, _rpy_cos, _rpy_neg_cos, _rpy_norm,
                                    _z_sign, _roll_sign, _pitch_sign,
                                    _z_corr, _roll_corr, _pitch_corr,
                                )

                            _frame_extra = _extra.reshape(_all_obs.shape[0], 5, 6)
                            _frame_scores = _score_extra_layout(
                                _frame_extra[:, :, :3],
                                _frame_extra[:, :, 3:],
                            )
                            _term_scores = _score_extra_layout(
                                _extra[:, :15].reshape(_all_obs.shape[0], 5, 3),
                                _extra[:, 15:30].reshape(_all_obs.shape[0], 5, 3),
                            )
                            _best_scores = _frame_scores
                            if _term_scores[0] > _frame_scores[0]:
                                _best_scores = _term_scores
                            _obs_pos_best_cos = _best_scores[0].item()
                            _obs_rpy_best_cos = _best_scores[1].item()
                            _obs_rpy_best_neg_cos = _best_scores[2].item()
                            _obs_rpy_best_norm = _best_scores[3].item()
                            _obs_z_best_sign = _best_scores[4].item()
                            _obs_roll_best_sign = _best_scores[5].item()
                            _obs_pitch_best_sign = _best_scores[6].item()
                            _obs_z_best_corr = _best_scores[7].item()
                            _obs_roll_best_corr = _best_scores[8].item()
                            _obs_pitch_best_corr = _best_scores[9].item()
                        _energy_pred_all = self.alg.policy.evaluate(_all_critic_obs)
                        _energy_loss_all = torch.nn.functional.huber_loss(
                            _energy_pred_all, _all_energy, reduction="mean").item()
                        _energy_mae = (_energy_pred_all - _all_energy).abs().mean().item()
                        _energy_target_mean = _all_energy.mean().item()
                        _energy_pred_mean = _energy_pred_all.mean().item()
                        _energy_target_std = _all_energy.std(unbiased=False).item()
                        _energy_pred_std = _energy_pred_all.std(unbiased=False).item()
                        _energy_cov = (
                            (_energy_pred_all - _energy_pred_all.mean())
                            * (_all_energy - _all_energy.mean())
                        ).mean()
                        _energy_corr = (
                            _energy_cov
                            / (_energy_pred_all.std(unbiased=False) * _all_energy.std(unbiased=False)).clamp(min=1e-6)
                        ).item()
                        _safe_gap_diag = float(self.cfg.get("frontres_safe_gap_per_step", 0.003))
                        _broken_gap_diag = float(self.cfg.get("frontres_broken_gap_per_step", 0.08))
                        _energy_damage_frac = (_all_energy.view(-1) > _safe_gap_diag).float().mean().item()
                        _energy_broken_frac = (_all_energy.view(-1) > _broken_gap_diag).float().mean().item()
                    print(f"[Runner]   warmup {_wu + 1}/{_warmup_iters}: "
                          f"dr_scale={_warmup_dr_scale:.3f}, "
                          f"mode_mix={tuple(_warmup_mode_groups)}, "
                          f"loss={loss.item():.6f}, actor={_last_actor_loss.item():.6f}, "
                          f"energy={_last_energy_loss.item():.6f}, cos={_warmup_cos:.4f}, "
                          f"valid={_valid_frac:.3f}",
                          flush=True)
                    print(f"[Runner]      diag: "
                          f"cos_pos={_cos_pos:+.4f}, cos_rpy={_cos_rpy:+.4f}, "
                          f"valid_pos={_valid_pos_frac:.3f}, valid_rpy={_valid_rpy_frac:.3f}",
                          flush=True)
                    print(f"[Runner]      diag_valid_axes: "
                          f"x/y/z={_valid_x_frac:.3f}/{_valid_y_frac:.3f}/{_valid_z_frac:.3f}, "
                          f"r/p/yaw={_valid_roll_frac:.3f}/{_valid_pitch_frac:.3f}/{_valid_yaw_frac:.3f}",
                          flush=True)
                    print(f"[Runner]      diag: "
                          f"mae_pos={_mae_pos:.5f}m, mae_rpy={_mae_rpy:.5f}rad, "
                          f"|pred_pos|/|tgt_pos|={_pred_pos_norm:.5f}/{_tgt_pos_norm:.5f}, "
                          f"|pred_rpy|/|tgt_rpy|={_pred_rpy_norm:.5f}/{_tgt_rpy_norm:.5f}",
                          flush=True)
                    print(f"[Runner]      diag_rpy: "
                          f"sign_r/p/y={_sign_roll:.3f}/{_sign_pitch:.3f}/{_sign_yaw:.3f}, "
                          f"valid_r/p/y={_valid_roll_frac:.3f}/{_valid_pitch_frac:.3f}/{_valid_yaw_frac:.3f}, "
                          f"|pred_r/p/y|={_abs_pred_roll:.5f}/{_abs_pred_pitch:.5f}/{_abs_pred_yaw:.5f}, "
                          f"|tgt_r/p/y|={_abs_tgt_roll:.5f}/{_abs_tgt_pitch:.5f}/{_abs_tgt_yaw:.5f}",
                          flush=True)
                    print(f"[Runner]      diag_obs_target: "
                          f"best_obs_pos_cos={_obs_pos_best_cos:+.4f}, "
                          f"best_obs_rpy_cos={_obs_rpy_best_cos:+.4f}, "
                          f"best_neg_obs_rpy_cos={_obs_rpy_best_neg_cos:+.4f}, "
                          f"best_obs_rpy_norm={_obs_rpy_best_norm:.5f}",
                          flush=True)
                    print(f"[Runner]      diag_obs_target_axis: "
                          f"sign_z/r/p={_obs_z_best_sign:.3f}/{_obs_roll_best_sign:.3f}/{_obs_pitch_best_sign:.3f}, "
                          f"corr_z/r/p={_obs_z_best_corr:+.3f}/{_obs_roll_best_corr:+.3f}/{_obs_pitch_best_corr:+.3f}",
                          flush=True)
                    print(f"[Runner]      energy: "
                          f"loss={_energy_loss_all:.6f}, mae={_energy_mae:.6f}, "
                          f"pred/target={_energy_pred_mean:.6f}/{_energy_target_mean:.6f}",
                          flush=True)
                    print(f"[Runner]      energy: "
                          f"corr={_energy_corr:+.4f}, std_pred/target={_energy_pred_std:.6f}/{_energy_target_std:.6f}, "
                          f"damage_frac={_energy_damage_frac:.3f}, broken_frac={_energy_broken_frac:.3f}",
                          flush=True)

            print(f"[Runner] === Joint warmup complete (final loss={loss.item():.6f}) ===",
                  flush=True)
            # Save warmup checkpoint so subsequent runs can skip this phase via --resume
            if self.log_dir is not None:
                self._frontres_warmup_complete = True
                self._dr_scale = _dr_scale
                warmup_path = os.path.join(self.log_dir, f"model_warmup.pt")
                self.save(warmup_path)
                print(f"[Runner] Warmup checkpoint saved to {warmup_path}", flush=True)
            _apply_frontres_dr_scale(_dr_scale)
        # ── END supervised warmup ─────────────────────────────────────────────────

        print(
            f"[Runner] Entering PPO loop: start_iter={start_iter}, "
            f"tot_iter={tot_iter}, steps_per_env={self.num_steps_per_env}",
            flush=True,
        )
        for it in range(start_iter, tot_iter):
            start = time.time()
            _ppo_actor_weight_current = _frontres_ppo_actor_weight_for_iter(it)
            if _is_frontres and hasattr(self.alg, "ppo_actor_weight"):
                # Set before collection as well as before update so diagnostics,
                # curriculum gates, and PPO loss all see the same phase.
                self.alg.ppo_actor_weight = _ppo_actor_weight_current

            # Critic learns V(s) under real low perturbations. PPO actor may be
            # frozen separately by ppo_actor_warmup_iterations.
            _critic_warmup = (isinstance(self.alg.policy, FrontRESActorCritic)
                              and critic_warmup_iters > 0
                              and it < critic_warmup_iters)
            _dr_start_actor_weight = float(self.cfg.get("dr_start_ppo_actor_weight", 1.0))
            _actor_takeover_active = (
                _is_frontres
                and (not _critic_warmup)
                and _ppo_actor_weight_current < _dr_start_actor_weight
            )

            _boundary_enabled_for_iter = bool(self.cfg.get("frontres_boundary_dr_enabled", True))
            _boundary_takeover_for_iter = bool(self.cfg.get("frontres_boundary_dr_during_actor_takeover", False))
            _hsl_boundary_available = (
                _frontres_hsl_restore
                and _perturb_target is not None
                and _boundary_enabled_for_iter
                and getattr(self, "_last_frontres_boundary_stats", None) is not None
                and (not _critic_warmup)
                and ((not _actor_takeover_active) or _boundary_takeover_for_iter)
            )

            if _frontres_hsl_restore and _perturb_target is not None and not _hsl_boundary_available:
                _sup_dr_start = float(self.cfg.get("frontres_supervised_dr_scale_start", _dr_scale_init))
                _sup_dr_end = float(self.cfg.get("frontres_supervised_dr_scale_end", _sup_dr_start))
                _sup_dr_ramp = max(1, int(self.cfg.get("frontres_supervised_dr_ramp_iters", 500)))
                _sup_dr_delay = max(0, int(self.cfg.get("frontres_supervised_dr_delay_iters", 0)))
                # Use absolute iteration, with an optional post-warmup delay, so
                # a full-resume keeps the supervised perturbation curriculum
                # aligned while still letting Critic/Actor settle before harder
                # disturbances arrive.
                _sup_dr_phase = max(0, it - _sup_dr_delay)
                _sup_dr_frac = min(1.0, max(0.0, _sup_dr_phase / float(_sup_dr_ramp)))
                _dr_scale = _sup_dr_start + (_sup_dr_end - _sup_dr_start) * _sup_dr_frac
                if _critic_warmup or _actor_takeover_active:
                    _dr_scale = _dr_scale_init
                _dr_scale = max(_dr_min, min(_dr_max, _dr_scale))
                self._dr_scale = _dr_scale
                _curriculum_iters = int(self.cfg.get("frontres_curriculum_total_iterations", 1500))
                _curriculum_iters = max(1, _curriculum_iters)
                _curriculum_progress = min(1.0, max(0.0, it / float(_curriculum_iters)))
                if _critic_warmup or _actor_takeover_active:
                    _curriculum_progress = 0.0
                _sample_frontres_rollout_perturbation_mix(_curriculum_progress, it)
                _mix_strength_enabled = not (_critic_warmup or _actor_takeover_active)
                _scale_env, _frontres_dr_mix_mode, _mix_diag = _frontres_mixed_dr_scale_env(
                    _dr_scale, _mix_strength_enabled, it
                )
                if _scale_env is None:
                    _effective_dr_scale, _frontres_dr_mix_mode = _frontres_mixed_dr_scale(
                        _dr_scale, _mix_strength_enabled, it
                    )
                    _mix_diag = {"easy": 0.0, "frontier": 1.0, "hard": 0.0, "mean": _effective_dr_scale}
                    self._frontres_dr_mix_class_train = None
                    _apply_frontres_dr_scale(_effective_dr_scale)
                else:
                    _effective_dr_scale = float(_mix_diag["mean"])
                    _apply_frontres_dr_scale_env(_scale_env, _dr_scale)
                self._frontres_effective_dr_scale = _effective_dr_scale
                self._frontres_dr_mix_mode = _frontres_dr_mix_mode
                self._frontres_dr_frontier_scale = _dr_scale
                self._frontres_dr_mix_easy_frac = float(_mix_diag["easy"])
                self._frontres_dr_mix_frontier_frac = float(_mix_diag["frontier"])
                self._frontres_dr_mix_hard_frac = float(_mix_diag["hard"])
                self._frontres_dr_mix_mean_scale = float(_mix_diag["mean"])
                _skip_frontres_dr_controller = True
            else:
                _skip_frontres_dr_controller = False

            # --- DR controller -------------------------------------------------
            # Prefer boundary sampling when FrontRES gap diagnostics are available:
            # keep perturbations near GMT's repairable boundary instead of pushing
            # blindly on reward scale.  Falls back to the older r_delta PI before
            # the first diagnostic batch is available.
            if _is_frontres and _perturb_target is not None and not _skip_frontres_dr_controller:
                # Update r_delta EMA from last iteration's mean
                _r_delta_ema = (_dr_ema_alpha * _r_delta_ema
                                + (1.0 - _dr_ema_alpha) * getattr(self, '_last_r_delta_mean', 0.0))
                _boundary_enabled = bool(self.cfg.get("frontres_boundary_dr_enabled", True))
                _boundary_takeover = bool(self.cfg.get("frontres_boundary_dr_during_actor_takeover", False))
                _boundary_stats = getattr(self, "_last_frontres_boundary_stats", None)
                _use_boundary = (
                    _boundary_enabled
                    and _boundary_stats is not None
                    and (not _critic_warmup)
                    and ((not _actor_takeover_active) or _boundary_takeover)
                )

                if _critic_warmup or (_actor_takeover_active and not _boundary_takeover):
                    # Hold at dr_scale_init during Critic warmup.
                    # Also hold during Actor takeover.  Otherwise p(s_noisy) and
                    # p(s_noisy + Δ_actor) drift at the same time, which makes
                    # Critic targets hard to interpret and PPO updates unstable.
                    _dr_scale = _dr_scale_init
                    self._dr_hold_just_ended = True
                elif _use_boundary:
                    if getattr(self, '_dr_hold_just_ended', False):
                        if hasattr(self, "_dr_prev_error"):
                            delattr(self, "_dr_prev_error")
                        self._dr_hold_just_ended = False

                    _ema_alpha = float(self.cfg.get("frontres_boundary_dr_ema_alpha", 0.90))
                    _ema_alpha = max(0.0, min(0.999, _ema_alpha))
                    _ema = getattr(self, "_frontres_boundary_ema", None)
                    if _ema is None:
                        _ema = dict(_boundary_stats)
                    else:
                        for _key, _value in _boundary_stats.items():
                            _ema[_key] = _ema_alpha * float(_ema.get(_key, _value)) + (1.0 - _ema_alpha) * float(_value)
                    self._frontres_boundary_ema = _ema

                    _safe = float(_ema.get("safe", 0.0))
                    _repair = float(_ema.get("repair", _ema.get("fragile", 0.0)))
                    _broken = float(_ema.get("broken", 0.0))
                    _gainpos = float(_ema.get("positive_gain", 0.5))
                    _gmt_frontier_enabled = bool(
                        self.cfg.get("frontres_gmt_frontier_probe_enabled", True)
                    )
                    _gmt_frontier_decision = "disabled"
                    _gmt_frontier_score = getattr(self, "_frontres_gmt_frontier_probe_score", None)

                    _safe_hi = float(self.cfg.get("frontres_boundary_safe_high", 0.45))
                    _broken_hi = float(self.cfg.get("frontres_boundary_broken_high", 0.35))
                    _broken_target = float(self.cfg.get("frontres_boundary_broken_target", 0.25))
                    _repair_lo = float(self.cfg.get(
                        "frontres_boundary_repair_low",
                        self.cfg.get("frontres_boundary_fragile_low", 0.45),
                    ))
                    _repair_hi = float(self.cfg.get(
                        "frontres_boundary_repair_high",
                        self.cfg.get("frontres_boundary_fragile_high", 0.70),
                    ))
                    _gain_hi = float(self.cfg.get("frontres_boundary_positive_gain_high", 0.55))
                    _gain_lo = float(self.cfg.get("frontres_boundary_positive_gain_low", 0.45))
                    _step = float(self.cfg.get("frontres_boundary_dr_step", 0.03))

                    if _gmt_frontier_enabled:
                        _probe_scale = float(
                            getattr(self, "_frontres_gmt_frontier_probe_scale", _dr_scale)
                        )
                        _probe_scale = max(_dr_min, min(_dr_max, _probe_scale))
                        _safe_low = float(
                            getattr(self, "_frontres_gmt_frontier_safe_low", _dr_scale_init)
                        )
                        _safe_low = max(_dr_min, min(_dr_max, _safe_low))
                        _broken_high = getattr(self, "_frontres_gmt_frontier_broken_high", None)
                        _score_ref = max(
                            1e-6,
                            float(self.cfg.get("frontres_gmt_frontier_ref_episode_len", 0.0) or 0.0),
                        )
                        if _score_ref <= 1e-6:
                            _score_ref = max(1e-6, float(getattr(self, "max_episode_length", 1.0)))
                        _per_env_mix_active = (
                            getattr(self, "_frontres_dr_mix_mode", None) == "per_env"
                            and getattr(self, "_frontres_dr_mix_class_train", None) is not None
                        )
                        _frontier_len_source = (
                            lenbuffer_gmt_base_frontier if _per_env_mix_active else lenbuffer_gmt_base
                        )
                        self._frontres_gmt_frontier_probe_source = (
                            "frontier-class" if _per_env_mix_active else "all-base"
                        )
                        self._frontres_gmt_frontier_probe_samples = len(_frontier_len_source)
                        if len(_frontier_len_source) > 0:
                            _gmt_ep_len = float(statistics.mean(_frontier_len_source))
                            _gmt_frontier_score = max(0.0, min(1.5, _gmt_ep_len / _score_ref))
                            _safe_thr = float(
                                self.cfg.get("frontres_gmt_frontier_safe_threshold", 0.85)
                            )
                            _broken_thr = float(
                                self.cfg.get("frontres_gmt_frontier_broken_threshold", 0.65)
                            )
                            _growth = max(
                                1.001,
                                float(self.cfg.get("frontres_gmt_frontier_growth_factor", 1.12)),
                            )
                            if _gmt_frontier_score >= _safe_thr:
                                _safe_low = max(_safe_low, _probe_scale)
                                _gmt_frontier_decision = "safe"
                                if _broken_high is None:
                                    _next_probe = min(_dr_max, max(_probe_scale * _growth, _safe_low + 1e-6))
                                else:
                                    _next_probe = 0.5 * (_safe_low + float(_broken_high))
                            elif _gmt_frontier_score <= _broken_thr:
                                _broken_high = (
                                    _probe_scale
                                    if _broken_high is None
                                    else min(float(_broken_high), _probe_scale)
                                )
                                if _probe_scale <= _safe_low + 1e-6:
                                    _retreat = max(
                                        0.1,
                                        min(0.99, float(self.cfg.get("frontres_gmt_frontier_retreat_factor", 0.85))),
                                    )
                                    _safe_low = max(_dr_min, min(_safe_low * _retreat, _probe_scale - 1e-6))
                                _gmt_frontier_decision = "broken"
                                _next_probe = 0.5 * (_safe_low + float(_broken_high))
                            else:
                                _gmt_frontier_decision = "frontier"
                                if _broken_high is None:
                                    _broken_high = min(_dr_max, max(_probe_scale * _growth, _probe_scale + 1e-6))
                                _next_probe = _probe_scale
                            _safe_low = max(_dr_min, min(_dr_max, _safe_low))
                            if _broken_high is not None:
                                _broken_high = max(_safe_low, min(_dr_max, float(_broken_high)))
                            _conservative = max(
                                0.0,
                                min(1.0, float(self.cfg.get("frontres_gmt_frontier_conservative_frac", 0.0))),
                            )
                            if _broken_high is not None:
                                _frontier = _safe_low + _conservative * (float(_broken_high) - _safe_low)
                            else:
                                _frontier = _safe_low
                            self._frontres_gmt_frontier_safe_low = _safe_low
                            self._frontres_gmt_frontier_broken_high = _broken_high
                            self._frontres_gmt_frontier_confirmed = max(_dr_min, min(_dr_max, _frontier))
                            self._frontres_gmt_frontier_probe_scale = max(_dr_min, min(_dr_max, _next_probe))
                            # The next rollout uses the probe scale.  The confirmed
                            # safe frontier is logged separately so we can see the
                            # bracket converge instead of mistaking a probe for a
                            # validated boundary.
                            _dr_scale = self._frontres_gmt_frontier_probe_scale
                        else:
                            _gmt_frontier_decision = "waiting"
                            self._frontres_gmt_frontier_probe_scale = _probe_scale
                            self._frontres_gmt_frontier_confirmed = float(
                                getattr(self, "_frontres_gmt_frontier_confirmed", _dr_scale)
                            )
                        self._frontres_gmt_frontier_probe_score = _gmt_frontier_score
                        self._frontres_gmt_frontier_decision = _gmt_frontier_decision
                    else:
                        # Multiplicative controller.  Broken samples dominate because
                        # they pollute the actor with unrepairable targets.  Safe-heavy
                        # batches are too easy.  Fragile + positive gain means the
                        # current boundary is learnable and can be nudged outward.
                        _factor = 1.0
                        if _broken > _broken_hi:
                            _factor = 1.0 - _step * min(3.0, 1.0 + (_broken - _broken_hi) / max(1.0 - _broken_hi, 1e-6))
                        elif _safe > _safe_hi and _broken < _broken_target:
                            _factor = 1.0 + _step
                        elif (_repair_lo <= _repair <= _repair_hi) and _gainpos > _gain_hi and _broken < _broken_hi:
                            _factor = 1.0 + 0.5 * _step
                        elif _gainpos < _gain_lo and _broken > _broken_target:
                            _factor = 1.0 - 0.5 * _step
                        _factor = max(0.80, min(1.10, _factor))
                        _dr_scale = max(_dr_min, min(_dr_max, _dr_scale * _factor))
                else:
                    if getattr(self, '_dr_hold_just_ended', False):
                        _dr_scale = _dr_scale_init
                        # The hold phase also runs at dr_scale_init, so the EMA is
                        # real signal.  Keep it; only reset the PI previous error
                        # so the first adaptive step is not a stale jump.
                        if hasattr(self, "_dr_prev_error"):
                            delattr(self, "_dr_prev_error")
                        self._dr_hold_just_ended = False

                    _kp        = float(self.cfg.get("dr_p_gain",          0.10))
                    _ki        = float(self.cfg.get("dr_i_gain",          0.01))
                    _dr_target = float(self.cfg.get("dr_target_r_delta",  0.01))

                    error      = _r_delta_ema - _dr_target
                    _prev_err  = getattr(self, '_dr_prev_error', error)
                    # Velocity-form PI: Δu = Kp×Δe + Ki×e
                    # dr_scale += Δu  (stays constant when error = 0)
                    # No double-integration: P reacts to error *change*, I to error *level*.
                    _delta = _kp * (error - _prev_err) + _ki * error
                    self._dr_prev_error = error

                    _dr_scale = max(_dr_min, min(_dr_max, _dr_scale + _delta))

                # Persist for resume
                self._dr_scale = _dr_scale
                # Apply dr_scale and active perturbation family to the perturber.
                _curriculum_iters = int(self.cfg.get("frontres_curriculum_total_iterations", 1500))
                _curriculum_iters = max(1, _curriculum_iters)
                # Use absolute iteration so a true full-resume does not silently
                # restart the perturbation curriculum from easy single modes.
                _curriculum_progress = min(1.0, max(0.0, it / float(_curriculum_iters)))
                _sample_frontres_rollout_perturbation_mix(_curriculum_progress, it)
                _scale_env, _frontres_dr_mix_mode, _mix_diag = _frontres_mixed_dr_scale_env(_dr_scale, _use_boundary, it)
                if _scale_env is None:
                    _effective_dr_scale, _frontres_dr_mix_mode = _frontres_mixed_dr_scale(_dr_scale, _use_boundary, it)
                    _mix_diag = {"easy": 0.0, "frontier": 1.0, "hard": 0.0, "mean": _effective_dr_scale}
                    self._frontres_dr_mix_class_train = None
                    _apply_frontres_dr_scale(_effective_dr_scale)
                else:
                    _effective_dr_scale = float(_mix_diag["mean"])
                    _apply_frontres_dr_scale_env(_scale_env, _dr_scale)
                self._frontres_effective_dr_scale = _effective_dr_scale
                self._frontres_dr_mix_mode = _frontres_dr_mix_mode
                self._frontres_dr_frontier_scale = _dr_scale
                self._frontres_dr_mix_easy_frac = float(_mix_diag["easy"])
                self._frontres_dr_mix_frontier_frac = float(_mix_diag["frontier"])
                self._frontres_dr_mix_hard_frac = float(_mix_diag["hard"])
                self._frontres_dr_mix_mean_scale = float(_mix_diag["mean"])

            # FrontRES reward-shaping state: reset at the start of each rollout.
            _frontres_prev_delta_q: torch.Tensor | None = None  # [N, A] residual action for smoothness penalty
            # Accumulators for wandb logging (per iteration, divided by shaping steps)
            _frontres_rdelta_sum:         float = 0.0
            _frontres_baseline_sum:       float = 0.0
            _frontres_r_z_sum:            float = 0.0
            _frontres_r_xy_sum:           float = 0.0
            _frontres_r_rp_sum:           float = 0.0
            _frontres_r_yaw_sum:          float = 0.0
            _frontres_r_rescue_sum:       float = 0.0
            _frontres_r_exec_sum:         float = 0.0
            _frontres_r_geom_sum:         float = 0.0
            _frontres_intervention_cost_sum: float = 0.0
            _frontres_clean_bound_cost_sum: float = 0.0
            _frontres_clean_bound_side_cost_sum: float = 0.0
            _frontres_over_cost_sum: float = 0.0
            _frontres_under_repair_cost_sum: float = 0.0
            _frontres_reward_frontres_sum: float = 0.0
            _frontres_reward_clean_sum:   float = 0.0
            _frontres_reward_candidate_sum: float = 0.0
            _frontres_reward_oracle_sum:  float = 0.0
            _frontres_exec_planar_sum:    float = 0.0
            _frontres_exec_vertical_sum:  float = 0.0
            _frontres_exec_task_sum:      float = 0.0
            _frontres_damage_gap_sum:     float = 0.0
            _frontres_oracle_clean_gap_sum: float = 0.0
            _frontres_oracle_trust_sum:   float = 0.0
            _frontres_repair_gain_sum:    float = 0.0
            _frontres_candidate_gain_sum: float = 0.0
            _frontres_projection_gain_sum: float = 0.0
            _frontres_underwrite_sum: float = 0.0
            _frontres_accept_pref_mask_sum: float = 0.0
            _frontres_accept_pref_full_sum: float = 0.0
            _frontres_accept_pref_noop_sum: float = 0.0
            _frontres_accept_pref_keep_sum: float = 0.0
            _frontres_accept_pref_ignore_sum: float = 0.0
            _frontres_accept_pref_margin_sum: float = 0.0
            _frontres_accept_pref_need_sum: float = 0.0
            _frontres_accept_pref_admiss_sum: float = 0.0
            _frontres_accept_pref_target_sum: float = 0.0
            _frontres_inertial_pref_penalty_rho_sum: float = 0.0
            _frontres_inertial_pref_penalty_one_sum: float = 0.0
            _frontres_positive_gain_frac_sum: float = 0.0
            _frontres_repair_ratio_sum:   float = 0.0
            _frontres_exec_signal_sum:     float = 0.0
            _frontres_weighted_exec_signal_sum: float = 0.0
            _frontres_train_reward_sum:    float = 0.0
            _frontres_effective_gain_bonus_sum: float = 0.0
            _frontres_safe_cost_sum:        float = 0.0
            _frontres_repair_cost_sum:      float = 0.0
            _frontres_broken_cost_sum:      float = 0.0
            _frontres_behavior_fit_sum:    float = 0.0
            _frontres_repair_fit_rate_sum: float = 0.0
            _frontres_repair_fit_gain_sum: float = 0.0
            _frontres_restore_ratio_rp_sum: float = 0.0
            _frontres_residual_rp_abs_sum: float = 0.0
            _frontres_corr_roll_bias_sum: float = 0.0
            _frontres_corr_pitch_bias_sum: float = 0.0
            _frontres_harm_rate_sum:       float = 0.0
            _frontres_harm_mag_sum:        float = 0.0
            _frontres_safe_harm_rate_sum:  float = 0.0
            _frontres_broken_harm_rate_sum: float = 0.0
            _frontres_safe_abstain_cost_sum: float = 0.0
            _frontres_broken_abstain_cost_sum: float = 0.0
            _frontres_window_mu_sum:      float = 0.0
            _frontres_safe_frac_sum:      float = 0.0
            _frontres_repair_frac_sum:    float = 0.0
            _frontres_broken_frac_sum:    float = 0.0
            _frontres_candidate_floor_margin_sum: float = 0.0
            _frontres_candidate_floor_pass_sum: float = 0.0
            _frontres_stable_route_frac_sum: float = 0.0
            _frontres_stable_endpoint_frac_sum: float = 0.0
            _frontres_tri_weight_repair_sum: float = 0.0
            _frontres_tri_weight_noisy_sum: float = 0.0
            _frontres_tri_weight_stable_sum: float = 0.0
            _frontres_state_alpha_pred_sum: float = 0.0
            _frontres_state_alpha_target_sum: float = 0.0
            _frontres_state_alpha_mask_sum: float = 0.0
            _frontres_state_alpha_route_sum: float = 0.0
            _frontres_actor_gate_sum:     float = 0.0
            _frontres_exec_gate_sum:      float = 0.0
            _frontres_cost_gate_sum:      float = 0.0
            _frontres_dr_z_abs_sum:       float = 0.0
            _frontres_dr_xy_abs_sum:      float = 0.0
            _frontres_dr_rp_abs_sum:      float = 0.0
            _frontres_dr_yaw_abs_sum:     float = 0.0
            _frontres_corr_z_abs_sum:     float = 0.0
            _frontres_corr_xy_abs_sum:    float = 0.0
            _frontres_corr_rp_abs_sum:    float = 0.0
            _frontres_corr_yaw_abs_sum:   float = 0.0
            _frontres_smooth_penalty_sum: float = 0.0
            _frontres_reg_penalty_sum:    float = 0.0
            _frontres_delta_pos_abs_sum:  float = 0.0   # mean |Δpos[:3]| per step (task-space mode)
            _frontres_delta_rpy_abs_sum:  float = 0.0   # mean |Δrpy[3:]| per step (task-space mode)
            _frontres_delta_z_abs_sum:    float = 0.0   # mean |Δz| per step (joint-space mode only)
            _frontres_jump_degree_sum:    float = 0.0   # mean jump_degree (gate activation monitor)
            _frontres_shaping_steps:      int   = 0
            _frontres_reward_diag_steps:  int   = 0
            _frontres_reward_progress_sum: float = 0.0
            _frontres_constraint_progress_sum: float = 0.0
            # Termination tracking for training envs (used to compute survival_rate this rollout)
            _frontres_term_count: int = 0
            _frontres_step_count: int = 0
            # reg_penalty activates once dr_scale ≥ 1.0 (base values fully applied).
            # Before that, reg pushing corrections→0 reinforces the no-op shortcut trap.
            _lambda_reg = getattr(self.alg, 'lambda_reg_current', 0.0) if _is_frontres else 0.0
            _dr_done    = _is_frontres and (_dr_scale >= 1.0)

            def _frontres_state_alpha_policy_obs(_obs, _ref_vel_obs):
                if (
                    getattr(self.alg, "use_estimate_ref_vel", False)
                    and getattr(self.alg, "ref_vel_estimator", None) is not None
                ):
                    _estimator_input = _ref_vel_obs if _ref_vel_obs is not None else _obs
                    _estimated = self.alg.ref_vel_estimator(_estimator_input)
                    return torch.cat([_obs, _estimated], dim=-1)
                return _obs

            # Rollout: 训练首先需要积攒数据, 等数据攒够才能调用self.alg.update()更新权重
            with torch.inference_mode(): # 关闭计算图的梯度追踪, 只进行推理
                for _rollout_step in range(self.num_steps_per_env):
                    # Sample actions
                    if self.training_type in ("mosaic", "frontres"):
                        # Extract motion groups for MOSAIC multi-teacher support.
                        # FrontRESUnified ignores this argument; paired-baseline bookkeeping
                        # is handled directly by the motion command.
                        motion_groups = None

                        # CRITICAL FIX: Use unwrapped env to access command_manager
                        # 对仿真环境进行unwrapped, 直接调用底层指令管理器得到正在运行的动作标签
                        # 因为仿真环境知道正在运行的动作序列, 但算法不知道, 需要把动作序号传递给算法
                        env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                        if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                            motion_command = env.command_manager._terms['motion']
                            if hasattr(motion_command, 'env_motion_groups'):
                                motion_groups = motion_command.env_motion_groups.clone()

                        # 前向传播获得动作
                        actions = self.alg.act(obs,
                                               privileged_obs,
                                               teacher_obs=teacher_obs if self.training_type == "mosaic" else None,
                                               ref_vel_estimator_obs=ref_vel_estimator_obs,
                                               motion_groups=motion_groups)
                        if _is_task_space_mode:
                            actions = _mask_frontres_task_actions(actions)
                            self.alg.transition.actions = actions.detach()
                            if hasattr(self.alg, "_get_actor_log_prob"):
                                self.alg.transition.actions_log_prob = \
                                    self.alg._get_actor_log_prob(actions).detach()
                            else:
                                self.alg.transition.actions_log_prob = \
                                    self.alg.policy.get_actions_log_prob(actions).detach()

                        if (
                            _is_frontres
                            and _is_task_space_mode
                            and bool(self.cfg.get("frontres_state_alpha_enabled", True))
                            and hasattr(self.alg.policy, "get_state_router_alpha")
                        ):
                            _n_alpha_route = min(N_train, N_base)
                            if _n_alpha_route > 0:
                                _alpha_obs = _frontres_state_alpha_policy_obs(obs, ref_vel_estimator_obs)
                                _alpha_prob = self.alg.policy.get_state_router_alpha(
                                    _alpha_obs[:_n_alpha_route]
                                ).view(-1).detach()
                                _alpha_thr = float(self.cfg.get("frontres_state_alpha_route_threshold", 0.70))
                                _alpha_route = _alpha_prob >= _alpha_thr
                                # During the first random iterations, avoid route
                                # takeover before the auxiliary head has labels.
                                _min_iter = int(self.cfg.get("frontres_state_alpha_route_min_iteration", 0))
                                if int(it) < _min_iter:
                                    _alpha_route = torch.zeros_like(_alpha_route, dtype=torch.bool)
                                if not bool(self.cfg.get("frontres_state_alpha_route_enabled", True)):
                                    _alpha_route = torch.zeros_like(_alpha_route, dtype=torch.bool)
                                self._frontres_stable_route_next_mask = _alpha_route.detach()
                                self._frontres_state_alpha_prob_next = _alpha_prob.detach()
                                self._frontres_state_alpha_pred_last = float(_alpha_prob.mean().item())
                                self._frontres_state_alpha_route_last = float(_alpha_route.float().mean().item())
                            else:
                                self._frontres_stable_route_next_mask = torch.zeros(0, device=self.device, dtype=torch.bool)
                                self._frontres_state_alpha_prob_next = torch.zeros(0, device=self.device)
                                self._frontres_state_alpha_pred_last = 0.0
                                self._frontres_state_alpha_route_last = 0.0
                        elif _is_frontres and _is_task_space_mode:
                            self._frontres_stable_route_next_mask = torch.zeros(N_train, device=self.device, dtype=torch.bool)
                            self._frontres_state_alpha_prob_next = torch.zeros(N_train, device=self.device)
                            self._frontres_state_alpha_pred_last = 0.0
                            self._frontres_state_alpha_route_last = 0.0

                        # Track velocity estimator error if available 仿真器有速度真值, 但学生模型只能瞎猜
                        if hasattr(self.alg, 'last_estimated_ref_vel') and self.alg.last_estimated_ref_vel is not None:
                            # Get ground truth ref_anchor_lin_vel_b from environment using existing mdp function
                            # This uses anchor body coordinate system to match offline training
                            from whole_body_tracking.tasks.tracking.mdp import observations as mdp
                            gt_ref_vel_b = mdp.ref_base_lin_vel_b(self.env.unwrapped, "motion")  # [N, 3]

                            # Compute MAE (same metric as training validation) 计算两者均方差
                            # MAE per environment (averaging across 3 velocity dimensions)
                            vel_error = (self.alg.last_estimated_ref_vel - gt_ref_vel_b).abs().mean(dim=-1)  # [N]
                            vel_est_error_buffer.extend(vel_error.cpu().numpy().tolist()) # 送入buffer

                        # B1 split-env: override stored actions/log_probs for GMT baseline envs
                        # and set frontres_mask so the critic update is masked correctly.
                        # Task-space mode applies ΔSE3 before building env_actions below,
                        # so GMT sees the corrected current reference, not only next obs/reward.
                        if _is_frontres:
                            _zeros_gmt = torch.zeros(N_candidate + N_base + N_clean, self.alg.transition.actions.shape[-1],
                                                     device=self.device)
                            self.alg.transition.actions[N_train:] = _zeros_gmt
                            if hasattr(self.alg, "_get_actor_log_prob"):
                                _logp_all = self.alg._get_actor_log_prob(self.alg.transition.actions)
                                _logp_zeros = _logp_all[N_train:]
                            else:
                                _mean_gmt = self.alg.policy.action_mean[N_train:].clone()
                                _std_gmt  = self.alg.policy.action_std[N_train:]
                                _logp_zeros = torch.distributions.Normal(_mean_gmt, _std_gmt) \
                                                  .log_prob(_zeros_gmt).sum(dim=-1)
                                # TanhNormal Jacobian correction for zero action:
                                # bounded=0 → raw=atanh(0)=0, Jacobian = Σlog(max_d)
                                # so TanhNormal.log_prob(0) = Normal.log_prob(0) - Σlog(max_d)
                                # This ensures IS ratio = 1 for GMT envs (zero correction taken).
                                if _is_task_space_mode:
                                    _logp_zeros = _logp_zeros - (
                                        3 * math.log(self.alg.policy.max_delta_pos)
                                        + 3 * math.log(self.alg.policy.max_delta_rpy))
                            self.alg.transition.actions_log_prob[N_train:] = _logp_zeros
                            _frontres_mask = torch.zeros(self.env.num_envs, 1, device=self.device)
                            _frontres_mask[:N_train] = 1.0
                            self.alg.transition.frontres_mask = _frontres_mask

                        if _is_task_space_mode:
                            self._apply_frontres_task_corrections(
                                actions,
                                N_train,
                                allow_oracle=True,
                                n_candidate=N_candidate if _is_frontres else 0,
                            )
                            _obs_corr, _extras_corr = self.env.get_observations()
                            _obs_corr_dict = _extras_corr.get("observations", {})
                            if self.policy_obs_type is not None and self.policy_obs_type in _obs_corr_dict:
                                _obs_corr = _obs_corr_dict[self.policy_obs_type]
                            _obs_corr = self._apply_obs_normalizer(_obs_corr.to(self.device))
                            env_actions = self.alg.policy.get_env_action(_obs_corr, actions)
                        elif hasattr(self.alg.policy, 'get_env_action'):
                            env_actions = self.alg.policy.get_env_action(obs, actions)
                        else:
                            env_actions = actions

                    elif self.training_type == "supervise":
                        # Stage 1 Supervised Learning rollout:
                        #   - GMT drives the environment (produces joint position targets)
                        #   - Student only records (obs, target_delta_q) for offline training
                        #
                        # GMT ONNX includes the normalizer in the computation graph, so it
                        # expects RAW (unnormalized) observations.  `obs_raw_for_gmt` holds
                        # the un-normalized policy obs from the previous step.
                        env_actions = self.alg.policy.get_gmt_action(obs_raw_for_gmt)

                        # Record current (obs_t, target_delta_q_t) transition for training.
                        # privileged_obs == obs_dict["target"] == Δq_gt (Identity normalizer, raw units).
                        # The return value (student's Δq_pred) is discarded here; prediction is
                        # recomputed with gradients inside SuperviseTrainer.update().
                        _ = self.alg.act(obs, privileged_obs)

                    else:
                        actions = self.alg.act(obs, privileged_obs)
                        if _is_task_space_mode:
                            actions = _mask_frontres_task_actions(actions)
                            self.alg.transition.actions = actions.detach()
                            if hasattr(self.alg, "_get_actor_log_prob"):
                                self.alg.transition.actions_log_prob = \
                                    self.alg._get_actor_log_prob(actions).detach()
                            else:
                                self.alg.transition.actions_log_prob = \
                                    self.alg.policy.get_actions_log_prob(actions).detach()

                        if _is_frontres:
                            # B1 split: FrontRES envs [0:N_train] use policy delta_q,
                            #           GMT envs [N_train:] receive zero delta_q (pure GMT).
                            # Override the transition's stored action/log_prob for GMT envs
                            # BEFORE process_env_step() calls add_transitions() — this ensures
                            # the IS ratio used in PPO is correct (zero was the actual env action).
                            _zeros_gmt = torch.zeros_like(actions[N_train:])
                            self.alg.transition.actions[N_train:] = _zeros_gmt
                            if hasattr(self.alg, "_get_actor_log_prob"):
                                _logp_all = self.alg._get_actor_log_prob(self.alg.transition.actions)
                                _logp_zeros = _logp_all[N_train:]
                            else:
                                _mean_gmt   = self.alg.policy.action_mean[N_train:].clone()
                                _std_gmt    = self.alg.policy.action_std[N_train:]  # [N_base, A]
                                _logp_zeros = torch.distributions.Normal(_mean_gmt, _std_gmt) \
                                                  .log_prob(_zeros_gmt).sum(dim=-1)
                                if _is_task_space_mode:
                                    _logp_zeros = _logp_zeros - (
                                        3 * math.log(self.alg.policy.max_delta_pos)
                                        + 3 * math.log(self.alg.policy.max_delta_rpy))
                            self.alg.transition.actions_log_prob[N_train:] = _logp_zeros
                            # Mark which envs are FrontRES (1) vs GMT baseline (0) for critic masking.
                            _frontres_mask = torch.zeros(self.env.num_envs, 1, device=self.device)
                            _frontres_mask[:N_train] = 1.0
                            self.alg.transition.frontres_mask = _frontres_mask

                        if _is_frontres and _is_task_space_mode:
                            self._apply_frontres_task_corrections(
                                actions,
                                N_train,
                                allow_oracle=True,
                                n_candidate=N_candidate,
                            )
                            _obs_corr, _extras_corr = self.env.get_observations()
                            _obs_corr_dict = _extras_corr.get("observations", {})
                            if self.policy_obs_type is not None and self.policy_obs_type in _obs_corr_dict:
                                _obs_corr = _obs_corr_dict[self.policy_obs_type]
                            _obs_corr = self._apply_obs_normalizer(_obs_corr.to(self.device))
                            env_actions = self.alg.policy.get_env_action(_obs_corr, self.alg.transition.actions)
                        elif hasattr(self.alg.policy, 'get_env_action'):
                            env_actions = self.alg.policy.get_env_action(obs, actions)
                        else:
                            env_actions = actions

                    # Read supervised target BEFORE env.step: the command term's cache holds
                    # the perturbation that generated the CURRENT obs (used by FrontRES this step).
                    # After env.step, _update_command() overwrites the cache for the next step.
                    if _is_task_space_mode and getattr(self.alg, 'lambda_supervised', 0.0) > 0:
                        # Use a local env reference to avoid relying on _env_raw which is only
                        # assigned inside the _task_corr is not None branch above.
                        _env_for_sup = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                        for _cmd_sup in _env_for_sup.command_manager._terms.values():
                            if hasattr(_cmd_sup, 'supervised_target'):
                                _sup_target = _cmd_sup.supervised_target.clone().to(self.device)
                                _sup_target = self._frontres_project_task_target_to_action_cone(
                                    _cmd_sup, _sup_target
                                )
                                if bool(self.cfg.get("frontres_per_mode_supervised_mask", True)):
                                    _sup_mode_groups = list(getattr(
                                        self,
                                        "_frontres_curriculum_env_mode_groups",
                                        [tuple(getattr(self, "_frontres_curriculum_active_modes", ()))] * N_train,
                                    ))[:N_train]
                                    _sup_target = self._frontres_apply_per_mode_supervised_mask(
                                        _sup_target, _sup_mode_groups, N_train
                                    )
                                self.alg.transition.supervised_target = _sup_target
                                self._maybe_print_frontres_restore_debug(
                                    it,
                                    _rollout_step,
                                    actions,
                                    _sup_target,
                                    N_train,
                                )
                                break

                    _hsl_pos_snapshot = None
                    _hsl_quat_snapshot = None
                    if (
                        _is_frontres
                        and _is_task_space_mode
                        and bool(self.cfg.get("frontres_hsl_rollout_label_enabled", False))
                    ):
                        _env_for_hsl_pre = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
                        if hasattr(_env_for_hsl_pre, "command_manager"):
                            for _term_hsl_pre in _env_for_hsl_pre.command_manager._terms.values():
                                if (
                                    hasattr(_term_hsl_pre, "_frontres_pos_correction")
                                    and hasattr(_term_hsl_pre, "_frontres_quat_correction")
                                ):
                                    _hsl_pos_snapshot = _term_hsl_pre._frontres_pos_correction[:N_train].clone()
                                    _hsl_quat_snapshot = _term_hsl_pre._frontres_quat_correction[:N_train].clone()
                                    break

                    # Step the environment 仿真环境更新观测量/动作评分/序列结束与否/监控数据
                    # NOTE: This is where the environment computes the *next* observation internally.
                    # The result is returned here and then used in the next loop iteration.
                    obs, rewards, dones, infos = self.env.step(env_actions.to(self.env.device))

                    # Move to device
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    if _is_frontres and _is_task_space_mode:
                        self._frontres_invalidate_temporal_reference_cache(dones)

                    if (
                        _is_frontres
                        and _is_task_space_mode
                        and bool(self.cfg.get("frontres_hsl_rollout_label_enabled", False))
                    ):
                        _env_for_hsl = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
                        _cmd_hsl = None
                        if hasattr(_env_for_hsl, "command_manager"):
                            for _term_hsl in _env_for_hsl.command_manager._terms.values():
                                if (
                                    hasattr(_term_hsl, "_frontres_pos_correction")
                                    and hasattr(_term_hsl, "_frontres_quat_correction")
                                ):
                                    _cmd_hsl = _term_hsl
                                    break
                        if _cmd_hsl is not None:
                            self._maybe_update_frontres_hsl_rollout_target(
                                _cmd_hsl,
                                actions,
                                dones,
                                _hsl_pos_snapshot,
                                _hsl_quat_snapshot,
                                N_train,
                                N_candidate,
                                N_base,
                                N_clean,
                            )

                    # ── GMT baselines ────────────────────────────────────────────────
                    # Noisy-GMT envs share the FrontRES perturbation and receive no
                    # correction. Clean-GMT envs are registered via set_baseline_envs()
                    # so the perturber keeps only the clean diagnostic block unperturbed.

                    # ── FrontRES B1 delta-reward ────────────────────────────────────────
                    # Noisy-GMT envs run the same perturbed reference with zero residual
                    # correction, so their reward is the GMT-only baseline.
                    # r_delta isolates FrontRES contribution to anchor tracking only.
                    # GMT env rewards are zeroed → returns ≈ 0 → advantage ≈ 0 → no policy gradient.
                    #
                    # Fix 1: anchor-only r_delta (HRL intrinsic reward)
                    # Global reward (joint_torque, contact, body vel) conflates FrontRES with GMT:
                    #   - FrontRES corrects correctly but causes torque spike → penalises correct action
                    #   - FrontRES correction has no effect → should give 0, not negative
                    # Anchor error is the ONLY thing FrontRES directly controls. Using it as the
                    # intrinsic reward decouples FrontRES credit from GMT behaviour.
                    if _is_frontres:
                        _candidate_start = N_train
                        _candidate_end = _candidate_start + N_candidate
                        _base_start = _candidate_end
                        _base_end = _base_start + N_base
                        _clean_start = _base_end
                        _clean_end = _clean_start + N_clean
                        r_raw_gmt = rewards[_base_start:_base_end].view(-1).clone()
                        r_clean_gmt = rewards[_clean_start:_clean_end].view(-1).clone()
                        r_candidate_gmt = rewards[_candidate_start:_candidate_end].view(-1).clone() if N_candidate > 0 else None
                        r_total = rewards[:N_train].view(-1).clone()

                        # ── Anchor-only r_delta (GMT-free) ────────────────────────────
                        # Compares corrected anchor against ORIGINAL motion data,
                        # NOT against robot position.  GMT tracking noise ~28cm
                        # cannot drown a 2cm FrontRES correction anymore.
                        _env_for_rdelta = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                        _mcmd_rdelta = _env_for_rdelta.command_manager._terms.get('motion')
                        _use_clean = (
                            _mcmd_rdelta is not None
                            and hasattr(_mcmd_rdelta, 'anchor_pos_w_original')
                            and hasattr(_mcmd_rdelta, 'anchor_quat_w_original')
                        )
                        if _use_clean:
                            # ── Per-axis r_step ───────────────────────────
                            # r_axis[i] = |DR_i| - |DR_i + correction_i|
                            # Keep this per-env; batch-averaging here destroys PPO
                            # credit assignment because every action receives the same reward.
                            _a_w   = _mcmd_rdelta.anchor_pos_w_original
                            _a_raw = _mcmd_rdelta.anchor_pos_w_raw
                            _a_fr  = _mcmd_rdelta.anchor_pos_w
                            _q_w   = _mcmd_rdelta.anchor_quat_w_original
                            _q_raw = _mcmd_rdelta.anchor_quat_w_raw
                            _q_fr  = _mcmd_rdelta.anchor_quat_w

                            def _r_axis(dr, corr):
                                return dr.abs() - (dr + corr).abs()

                            def _r_vec(dr_vec, corr_vec):
                                return dr_vec.norm(dim=-1) - (dr_vec + corr_vec).norm(dim=-1)

                            # Z
                            _dr_z_fr   = _a_raw[:N_train, 2] - _a_w[:N_train, 2]
                            _corr_z_fr = _a_fr[:N_train,  2] - _a_raw[:N_train, 2]
                            _r_z = _r_axis(_dr_z_fr, _corr_z_fr)
                            # XY
                            _dr_xy_fr   = _a_raw[:N_train, :2] - _a_w[:N_train, :2]
                            _corr_xy_fr = _a_fr[:N_train,  :2] - _a_raw[:N_train, :2]
                            _r_xy = _r_vec(_dr_xy_fr, _corr_xy_fr)
                            def _wrap_pi(a: torch.Tensor):
                                return torch.atan2(torch.sin(a), torch.cos(a))

                            # Roll/Pitch diagnostics live in the same local tangent
                            # space as the FrontRES action:
                            #   q_frontres = q_raw * exp(delta_rotvec)
                            # Therefore the target error is log(inv(q_raw)*q_clean)
                            # and the residual error is log(inv(q_frontres)*q_clean).
                            _rot_raw_to_clean = _quat_to_rotvec_wxyz(
                                quat_mul(quat_inv(_q_raw[:N_train]), _q_w[:N_train])
                            )
                            _rot_fr_to_clean = _quat_to_rotvec_wxyz(
                                quat_mul(quat_inv(_q_fr[:N_train]), _q_w[:N_train])
                            )
                            _rot_raw_to_fr = _quat_to_rotvec_wxyz(
                                quat_mul(quat_inv(_q_raw[:N_train]), _q_fr[:N_train])
                            )
                            _rp_raw = _rot_raw_to_clean[:, :2]
                            _rp_fr = _rot_fr_to_clean[:, :2]
                            _e_raw = _rp_raw.norm(dim=-1)
                            _e_fr = _rp_fr.norm(dim=-1)
                            _r_rp = _e_raw - _e_fr
                            # Yaw
                            _roll_raw, _pitch_raw, _yaw_raw = euler_xyz_from_quat(_q_raw[:N_train])
                            _roll_fr,  _pitch_fr,  _yaw_fr  = euler_xyz_from_quat(_q_fr[:N_train])
                            _roll_w,   _pitch_w,   _yaw_w   = euler_xyz_from_quat(_q_w[:N_train])
                            _yaw_err_raw = _wrap_pi(_yaw_raw - _yaw_w)
                            _yaw_corr = _wrap_pi(_yaw_fr - _yaw_raw)
                            _r_ya = _r_axis(_yaw_err_raw, _yaw_corr)

                            _restore_z_weight = float(self.cfg.get("frontres_restore_z_weight", 0.3))
                            _restore_xy_weight = float(self.cfg.get("frontres_restore_xy_weight", 0.3))
                            _restore_rp_weight = float(self.cfg.get("frontres_restore_rp_weight", 0.15))
                            _restore_yaw_weight = float(self.cfg.get("frontres_restore_yaw_weight", 0.02))
                            _r_step = (
                                _restore_z_weight * _r_z
                                + _restore_xy_weight * _r_xy
                                + _restore_rp_weight * _r_rp
                                + _restore_yaw_weight * _r_ya
                            )
                            _dr_z_abs_log = _dr_z_fr.abs().mean()
                            _dr_xy_abs_log = _dr_xy_fr.norm(dim=-1).mean()
                            _dr_rp_abs_log = _e_raw.mean()
                            _dr_yaw_abs_log = _yaw_err_raw.abs().mean()
                            _corr_z_abs_log = _corr_z_fr.abs().mean()
                            _corr_xy_abs_log = _corr_xy_fr.norm(dim=-1).mean()
                            _corr_rp_abs_log = _rot_raw_to_fr[:, :2].norm(dim=-1).mean()
                            _corr_yaw_abs_log = _yaw_corr.abs().mean()

                            # ── r_rescue ─────────────────────────────────
                            _n_pair = min(N_train, N_candidate if N_candidate > 0 else N_train, N_base, N_clean)
                            _fell_base = dones[_base_start:_base_start+_n_pair].view(-1) > 0
                            _fell_fr   = dones[:_n_pair].view(-1) > 0
                            _r_rescue = torch.zeros(N_train, device=self.device)
                            _r_rescue_pair = torch.zeros(_n_pair, device=self.device)
                            # ±0.5 matches r_step magnitude (~0.02-0.05/step).
                            # ±10 was 200× larger, making Value learn fall-probability
                            # instead of correction quality, drowning the r_step signal.
                            _RESCUE_MAG = float(self.cfg.get("r_rescue_magnitude", 0.5))
                            _r_rescue_pair[_fell_base & ~_fell_fr] =  _RESCUE_MAG   # rescued
                            _r_rescue_pair[_fell_base &  _fell_fr] = -0.1 * _RESCUE_MAG   # both failed
                            _r_rescue_pair[~_fell_base & _fell_fr] = -_RESCUE_MAG   # FrontRES caused fall
                            _r_rescue[:_n_pair] = _r_rescue_pair

                            # ── Execution advantage (main HRL signal) ───────────────
                            # FrontRES should optimize the frozen tracker's
                            # executability, not the full environment reward.  The
                            # full reward contains teleop terms and low-level action
                            # penalties that are not aligned with reference-frame
                            # repair, so we build a dedicated continuous execution
                            # score here:
                            #   - stability margins: anchor z, anchor orientation,
                            #     and key end-effector z tracking margins
                            #   - weak velocity tracking: preserves motion semantics
                            #     so "be stable by doing nothing" is not a loophole
                            _r_exec = torch.zeros(N_train, device=self.device)
                            _n_exec = min(N_train, N_candidate if N_candidate > 0 else N_train, N_base, N_clean)

                            _exec_score_all, _exec_components = self._frontres_exec_score(
                                _mcmd_rdelta, return_components=True
                            )
                            _mode_groups = list(getattr(
                                self,
                                "_frontres_curriculum_env_mode_groups",
                                [tuple(getattr(self, "_frontres_curriculum_active_modes", ()))] * _n_exec,
                            ))[:_n_exec]
                            _exec_frontres = self._frontres_exec_score_for_modes(
                                _exec_components, 0, _n_exec, _mode_groups
                            )
                            if N_candidate > 0:
                                _exec_candidate = self._frontres_exec_score_for_modes(
                                    _exec_components, _candidate_start, _n_exec, _mode_groups
                                )
                            else:
                                _exec_candidate = _exec_frontres.detach()
                            _exec_perturbed = self._frontres_exec_score_for_modes(
                                _exec_components, _base_start, _n_exec, _mode_groups
                            )
                            _exec_clean = self._frontres_exec_score_for_modes(
                                _exec_components, _clean_start, _n_exec, _mode_groups
                            )
                            _, _feasible_components = self._frontres_feasible_oracle_exec_score(
                                _mcmd_rdelta, _base_start, _n_exec, return_components=True
                            )
                            _exec_feasible = self._frontres_exec_score_for_modes(
                                _feasible_components, 0, _n_exec, _mode_groups
                            ).to(self.device).view(-1)
                            _exec_planar_log = _exec_components["planar"][:_n_exec].mean()
                            _exec_vertical_log = _exec_components["vertical"][:_n_exec].mean()
                            _exec_task_log = _exec_components["task"][:_n_exec].mean()

                            # ── Clean-bounded intervention costs ───────────────────
                            # A plain ||delta|| penalty suppresses necessary repairs.
                            # The default regularizer is therefore target-relative:
                            #   - side cost: correction away from the Clean direction;
                            #   - over cost: correction past the Clean target.
                            # The legacy magnitude cost remains configurable, but its
                            # default weights are zero for demo-oriented training.
                            _intervention_cost = torch.zeros(N_train, device=self.device)
                            _clean_bound_cost = torch.zeros(N_train, device=self.device)
                            _side_cost = torch.zeros(N_train, device=self.device)
                            _over_cost = torch.zeros(N_train, device=self.device)
                            _under_repair_penalty = torch.zeros(N_train, device=self.device)
                            _action_activity = torch.zeros(N_train, device=self.device)
                            if _is_task_space_mode and actions.shape[-1] >= 6:
                                _delta = actions[:N_train, :6]
                                _max_delta = torch.tensor(
                                    [
                                        self.alg.policy.max_delta_pos,
                                        self.alg.policy.max_delta_pos,
                                        self.alg.policy.max_delta_pos,
                                        self.alg.policy.max_delta_rpy,
                                        self.alg.policy.max_delta_rpy,
                                        self.alg.policy.max_delta_rpy,
                                    ],
                                    device=self.device,
                                    dtype=_delta.dtype,
                                ).clamp(min=1e-6)
                                _weights = torch.tensor(
                                    self.cfg.get(
                                        "frontres_intervention_cost_weights",
                                        [0.02, 0.02, 0.05, 0.30, 0.30, 0.10],
                                    ),
                                    device=self.device,
                                    dtype=_delta.dtype,
                                )
                                _intervention_cost = (_weights * (_delta / _max_delta).pow(2)).sum(dim=-1)
                                _active_dims_cfg = self.cfg.get("frontres_active_task_dims", None)
                                if _active_dims_cfg is not None:
                                    _active_delta_dims = [
                                        int(_idx) for _idx in _active_dims_cfg
                                        if 0 <= int(_idx) < min(6, _delta.shape[-1])
                                    ]
                                else:
                                    _active_delta_dims = list(range(min(6, _delta.shape[-1])))
                                if _active_delta_dims:
                                    _active_idx = torch.tensor(_active_delta_dims, device=self.device, dtype=torch.long)
                                    _action_activity = (_delta[:, _active_idx] / _max_delta[_active_idx]).pow(2).mean(dim=-1)
                                    _target_delta = torch.cat(
                                        [
                                            (_a_w[:N_train] - _a_raw[:N_train]),
                                            _rot_raw_to_clean,
                                        ],
                                        dim=-1,
                                    )
                                    _corr_delta = torch.cat(
                                        [
                                            (_a_fr[:N_train] - _a_raw[:N_train]),
                                            _rot_raw_to_fr,
                                        ],
                                        dim=-1,
                                    )
                                    _target_active = _target_delta[:, _active_idx] / _max_delta[_active_idx]
                                    _corr_active = _corr_delta[:, _active_idx] / _max_delta[_active_idx]
                                    _target_norm = _target_active.norm(dim=-1, keepdim=True)
                                    _target_dir = _target_active / _target_norm.clamp(min=1e-6)
                                    _parallel_scalar = (_corr_active * _target_dir).sum(dim=-1, keepdim=True)
                                    _parallel = _parallel_scalar * _target_dir
                                    _side = _corr_active - _parallel

                                    _side_weight = float(self.cfg.get("frontres_clean_bound_side_weight", 0.0))
                                    _side_cost = max(_side_weight, 0.0) * _side.pow(2).sum(dim=-1)

                                    _over_margin = float(self.cfg.get("frontres_overcorrection_margin", 0.0))
                                    _over_weight = float(self.cfg.get("frontres_overcorrection_weight", 0.0))
                                    _over = torch.relu(
                                        _parallel_scalar.squeeze(-1)
                                        - _target_norm.squeeze(-1)
                                        - max(_over_margin, 0.0)
                                    )
                                    _over_cost = max(_over_weight, 0.0) * _over.pow(2)
                                    _clean_bound_cost = _side_cost + _over_cost
                            _overcorrection_cost = _clean_bound_cost

                            _w_exec = float(self.cfg.get("frontres_exec_reward_weight", 1.0))
                            _repair_scale = float(self.cfg.get("frontres_repair_reward_scale", 1.0))
                            _w_geom = float(self.cfg.get("frontres_geometry_reward_weight", 0.05))
                            _w_rescue = float(self.cfg.get("frontres_rescue_reward_weight", 1.0))
                            _w_exec_harm = float(self.cfg.get("frontres_executable_harm_weight", 1.0))
                            _reward_dr_ref = float(self.cfg.get(
                                "frontres_reward_scale_dr_reference",
                                self.cfg.get("supervised_warmup_dr_scale", self.cfg.get("dr_scale_init", 1.0)),
                            ))
                            _reward_dr_ref = max(_reward_dr_ref, 1e-6)
                            _reward_dr_progress = max(0.0, min(1.0, float(_dr_scale) / _reward_dr_ref))
                            _reward_actor_progress = max(0.0, min(1.0, float(_ppo_actor_weight_current)))
                            _reward_progress = _reward_dr_progress * _reward_actor_progress
                            _reward_progress = max(
                                float(self.cfg.get("frontres_reward_progress_min", 0.0)),
                                min(1.0, _reward_progress),
                            )
                            _constraint_exp = float(self.cfg.get("frontres_constraint_progress_exponent", 2.0))
                            _constraint_exp = max(1.0, _constraint_exp)
                            _constraint_progress = _reward_progress ** _constraint_exp
                            # Executable diagnostics and sample gates:
                            #   damage_gap  = R_clean_exec - R_perturbed_exec
                            #   repair_gain = R_frontres_exec - R_perturbed_exec
                            #   repair_ratio = repair_gain / damage_gap
                            #
                            # Clean is the behavior target.  The feasible oracle is
                            # only a trust diagnostic: if its executable score falls
                            # below Clean, it must not become a false repair ceiling.
                            #
                            # Safe/no-op and deeply broken samples should not
                            # drive the repair reward.  A double-sigmoid window
                            # gives one smooth repairability weight:
                            #   mu ~= 0 below safe_gap
                            #   mu ~= 1 between safe_gap and broken_gap
                            #   mu ~= 0 above broken_gap
                            # In selective mode this becomes a three-way objective:
                            #   safe:       abstain (action cost)
                            #   repairable: repair decisively (gain + margin bonus)
                            #   broken:     abstain/conservative repair (cost + harm)
                            _gap_raw = _exec_clean - _exec_perturbed
                            _damage_gap = _gap_raw.clamp(min=0.0)
                            _oracle_clean_gap = (_exec_clean - _exec_feasible).clamp(min=0.0)
                            _oracle_trust_tau = float(self.cfg.get("frontres_oracle_clean_gap_tau", 0.0))
                            if _oracle_trust_tau > 0.0:
                                _oracle_trust = torch.exp(-_oracle_clean_gap / max(_oracle_trust_tau, 1e-6))
                            else:
                                _oracle_trust_threshold = float(
                                    self.cfg.get("frontres_oracle_clean_gap_threshold", 1e9)
                                )
                                _oracle_trust = (_oracle_clean_gap <= _oracle_trust_threshold).to(_damage_gap.dtype)
                            _repair_gain = _exec_frontres - _exec_perturbed
                            _candidate_gain = _exec_candidate - _exec_perturbed
                            _projection_gain = _exec_frontres - _exec_candidate
                            _gap_floor = float(self.cfg.get("frontres_gap_floor_per_step", 0.005))
                            _repair_ratio = (_repair_gain / _damage_gap.clamp(min=_gap_floor)).clamp(-1.0, 1.0)
                            _reward_signal_mode = str(self.cfg.get("frontres_exec_reward_signal", "gain")).lower()
                            if _reward_signal_mode in ("family_preference", "preference", "ranking"):
                                _gain_std = self._frontres_family_gain_std(_mode_groups, _repair_gain.detach())
                                _tau = max(float(self.cfg.get("frontres_family_preference_tau", 1.0)), 1e-6)
                                _pref = torch.tanh((_repair_gain / _gain_std) / _tau)
                                _alpha = float(self.cfg.get("frontres_family_preference_alpha", 0.7))
                                _alpha = max(0.0, min(1.0, _alpha))
                                _scale = float(self.cfg.get("frontres_family_preference_scale", 0.02))
                                _exec_signal = _scale * (_alpha * _pref + (1.0 - _alpha) * _repair_ratio)
                            elif _reward_signal_mode == "ratio":
                                _exec_signal = _repair_ratio
                            else:
                                _exec_signal = _repair_gain
                            _r_exec[:_n_exec] = _exec_signal

                            _safe_gap = float(self.cfg.get("frontres_safe_gap_per_step", 0.003))
                            _broken_gap = float(self.cfg.get("frontres_broken_gap_per_step", 0.08))
                            _broken_gap = max(_broken_gap, _safe_gap + 1e-6)
                            _gate_temp = float(self.cfg.get("frontres_gap_gate_temp", 0.005))
                            _gate_temp = max(_gate_temp, 1e-6)
                            _enter_window = torch.sigmoid((_damage_gap - _safe_gap) / _gate_temp)
                            _exit_window = torch.sigmoid((_broken_gap - _damage_gap) / _gate_temp)
                            _window_mu_raw = _enter_window * _exit_window
                            _gap_mid = 0.5 * (_safe_gap + _broken_gap)
                            _peak_enter_arg = max(-60.0, min(60.0, (_gap_mid - _safe_gap) / _gate_temp))
                            _peak_exit_arg = max(-60.0, min(60.0, (_broken_gap - _gap_mid) / _gate_temp))
                            _window_peak = (
                                1.0 / (1.0 + math.exp(-_peak_enter_arg))
                                * 1.0 / (1.0 + math.exp(-_peak_exit_arg))
                            )
                            _window_mu = (_window_mu_raw / max(_window_peak, 1e-6)).clamp(0.0, 1.0)
                            _safe_gate = (1.0 - _enter_window).clamp(0.0, 1.0)
                            _repair_gate = _window_mu
                            _broken_gate = (1.0 - _exit_window).clamp(0.0, 1.0)

                            _safe_frac = (_damage_gap < _safe_gap).float().mean()
                            _repair_frac = ((_damage_gap >= _safe_gap) & (_damage_gap <= _broken_gap)).float().mean()
                            _broken_frac = (_damage_gap > _broken_gap).float().mean()
                            # Executable floor: the Candidate branch is judged
                            # against the GMT recoverability envelope, not
                            # against Noisy.  Clean is the center; broken_gap is
                            # the allowed distance from that center.
                            _candidate_floor_margin = (
                                _broken_gap - (_exec_clean - _exec_candidate)
                            )
                            _candidate_floor_pass = (_candidate_floor_margin >= 0.0).to(_damage_gap.dtype)
                            _candidate_floor_pass_frac = _candidate_floor_pass.mean()
                            _stable_route_next = getattr(
                                self,
                                "_frontres_stable_route_next_mask",
                                torch.zeros_like(_candidate_floor_margin, dtype=torch.bool),
                            )
                            _stable_route_next = _stable_route_next.to(self.device).view(-1).bool()
                            if _stable_route_next.numel() < _n_exec:
                                _stable_route_next = torch.nn.functional.pad(
                                    _stable_route_next,
                                    (0, _n_exec - _stable_route_next.numel()),
                                    value=False,
                                )
                            _stable_route_next = _stable_route_next[:_n_exec]
                            if _n_exec > 0:
                                _alive_next = ~(dones[:_n_exec].view(-1) > 0)
                                _stable_route_next = _stable_route_next & _alive_next
                            self._frontres_stable_route_next_mask = _stable_route_next.detach()
                            self._frontres_candidate_floor_margin_last = float(
                                _candidate_floor_margin.mean().detach().item()
                            )
                            self._frontres_candidate_floor_pass_last = float(
                                _candidate_floor_pass_frac.detach().item()
                            )
                            self._frontres_stable_route_frac_last = float(
                                _stable_route_next.to(_damage_gap.dtype).mean().detach().item()
                            )
                            _stable_route_active = getattr(self, "_frontres_stable_route_active_mask", None)
                            if _stable_route_active is None:
                                _stable_route_active = torch.zeros(_n_exec, device=self.device, dtype=torch.bool)
                            else:
                                _stable_route_active = _stable_route_active.to(self.device).view(-1).bool()
                                if _stable_route_active.numel() < _n_exec:
                                    _stable_route_active = torch.nn.functional.pad(
                                        _stable_route_active,
                                        (0, _n_exec - _stable_route_active.numel()),
                                        value=False,
                                    )
                                _stable_route_active = _stable_route_active[:_n_exec]
                            _learnable_route_mask = (~_stable_route_active).to(_damage_gap.dtype)

                            _restore_min_ratio = float(self.cfg.get("frontres_min_restore_ratio", 0.0))
                            _restore_under_weight = float(self.cfg.get("frontres_under_repair_weight", 0.0))
                            if _restore_min_ratio > 0.0 and _restore_under_weight > 0.0:
                                _restore_ratio = ((_e_raw - _e_fr) / _e_raw.clamp(min=1e-6)).clamp(-1.0, 1.0)
                                _under = torch.relu(_restore_min_ratio - _restore_ratio)
                                _under_repair_penalty[:_n_exec] = (
                                    _restore_under_weight
                                    * _repair_gate
                                    * _under[:_n_exec].pow(2)
                                )

                            _selective_reward = bool(self.cfg.get("frontres_selective_reward_enabled", True))
                            if _selective_reward:
                                _exec_gate = _repair_gate
                                _safe_cost_weight = float(self.cfg.get("frontres_safe_cost_weight", 1.0))
                                _repair_cost_weight = float(self.cfg.get("frontres_repair_cost_weight", 0.15))
                                _broken_cost_weight = float(self.cfg.get("frontres_broken_cost_weight", 1.0))
                                _cost_gate = (
                                    _safe_cost_weight * _safe_gate
                                    + _repair_cost_weight * _repair_gate
                                    + _broken_cost_weight * _broken_gate
                                ).clamp(min=0.0)
                            else:
                                _exec_gate = _window_mu
                                _cost_gate = (1.0 - _window_mu).clamp(0.0, 1.0)
                            _harm_eps = float(self.cfg.get("frontres_harm_epsilon", 0.001))
                            _harm_weight_cfg = float(self.cfg.get("frontres_harm_penalty_weight", 0.25))
                            _side_harm_weight = float(self.cfg.get("frontres_side_harm_weight", 0.0))
                            _side_harm_weight = max(0.0, min(1.0, _side_harm_weight))
                            _harm_mag_raw = torch.relu(-_repair_gain - max(_harm_eps, 0.0))
                            # A negative FrontRES-vs-perturbed executable delta is
                            # only a harmful repair if FrontRES actually changed the
                            # reference.  Otherwise small branch/evaluation noise can
                            # mark no-op safe/broken samples as "harmful" and dominate
                            # the reward.
                            _cost_exec = _intervention_cost[:_n_exec]
                            _harm_action_floor = float(self.cfg.get("frontres_harm_action_cost_floor", 0.001))
                            _harm_action_ref = float(self.cfg.get("frontres_harm_action_cost_ref", 0.01))
                            _harm_action_ref = max(_harm_action_ref, _harm_action_floor + 1e-6)
                            _harm_action_measure = _action_activity[:_n_exec]
                            _harm_action_gate = (
                                (_harm_action_measure - _harm_action_floor) / (_harm_action_ref - _harm_action_floor)
                            ).clamp(0.0, 1.0)
                            _harm_mag = _harm_mag_raw * _harm_action_gate
                            if _selective_reward:
                                _broken_harm_weight = float(self.cfg.get("frontres_broken_harm_weight", 1.0))
                                _harm_weight = (
                                    _repair_gate
                                    + _broken_harm_weight * _broken_gate
                                    + _side_harm_weight * _safe_gate
                                ).clamp(0.0, 1.0)
                            else:
                                _harm_weight = (
                                    _window_mu + _side_harm_weight * (1.0 - _window_mu)
                                ).clamp(0.0, 1.0)
                            _harm_penalty_exec = _harm_weight_cfg * _harm_weight * _harm_mag

                            _side_actor_weight = float(self.cfg.get("frontres_side_actor_gate_weight", 0.05))
                            _side_actor_weight = max(0.0, min(1.0, _side_actor_weight))
                            if _selective_reward:
                                _actor_gate = (
                                    _oracle_trust * _repair_gate
                                    + _side_actor_weight * (_safe_gate + _broken_gate)
                                ).clamp(0.0, 1.0)
                            else:
                                _actor_gate = (
                                    _oracle_trust * _window_mu + _side_actor_weight * (1.0 - _window_mu)
                                ).clamp(0.0, 1.0)
                            _actor_gate = _actor_gate * _learnable_route_mask
                            _exec_weight = torch.zeros(N_train, device=self.device)
                            _cost_weight = torch.ones(N_train, device=self.device)
                            _exec_weight[:_n_exec] = _exec_gate
                            _cost_weight[:_n_exec] = _cost_gate
                            _frontres_actor_gate = torch.zeros(self.env.num_envs, 1, device=self.device)
                            _frontres_actor_gate[:_n_exec, 0] = _actor_gate
                            self.alg.transition.frontres_actor_gate = _frontres_actor_gate
                            _state_alpha_target = torch.zeros(self.env.num_envs, 1, device=self.device)
                            _state_alpha_mask = torch.zeros(self.env.num_envs, 1, device=self.device)
                            self._frontres_state_alpha_target_last = 0.0
                            self._frontres_state_alpha_mask_last = 0.0
                            if (
                                bool(self.cfg.get("frontres_state_alpha_enabled", True))
                                and _n_exec > 0
                            ):
                                _base_done = dones[_base_start:_base_start + _n_exec].view(-1) > 0
                                _timeout = infos.get("time_outs", None)
                                if _timeout is not None:
                                    _timeout = _timeout.to(self.device).view(-1)
                                    _base_timeout = _timeout[_base_start:_base_start + _n_exec] > 0
                                else:
                                    _base_timeout = torch.zeros(_n_exec, device=self.device, dtype=torch.bool)
                                _fall_now = _base_done & (~_base_timeout)
                                _alpha_exec_floor = float(
                                    self.cfg.get("frontres_state_alpha_exec_floor", 0.0)
                                )
                                _alpha_safe_floor = float(
                                    self.cfg.get(
                                        "frontres_state_alpha_safe_exec_floor",
                                        max(_alpha_exec_floor + 0.05, _alpha_exec_floor),
                                    )
                                )
                                _alpha_temp = max(
                                    1e-6,
                                    float(self.cfg.get("frontres_state_alpha_temp", 0.08)),
                                )
                                _floor_mid = 0.5 * (_alpha_exec_floor + _alpha_safe_floor)
                                _soft_label = torch.sigmoid(
                                    (_floor_mid - _exec_perturbed) / _alpha_temp
                                )
                                _soft_label = torch.where(
                                    _fall_now,
                                    torch.ones_like(_soft_label),
                                    _soft_label,
                                )
                                _safe_signal = (~_base_done) & (_exec_perturbed >= _alpha_safe_floor)
                                _broken_signal = _fall_now | (_exec_perturbed <= _alpha_exec_floor)
                                _label_active = (_broken_signal | _safe_signal) & (~_base_timeout)
                                _state_alpha_target[:_n_exec, 0] = _soft_label.detach()
                                _state_alpha_mask[:_n_exec, 0] = _label_active.to(_state_alpha_mask.dtype).detach()
                                self._frontres_state_alpha_target_last = float(
                                    _state_alpha_target[:_n_exec, 0][_label_active].mean().item()
                                ) if bool(_label_active.any().item()) else 0.0
                                self._frontres_state_alpha_mask_last = float(
                                    _label_active.float().mean().item()
                                )
                            self.alg.transition.state_alpha_target = _state_alpha_target
                            self.alg.transition.state_alpha_mask = _state_alpha_mask
                            _harm_penalty = torch.zeros(N_train, device=self.device)
                            _harm_penalty[:_n_exec] = _harm_penalty_exec
                            _effective_gain_bonus_exec = torch.zeros(_n_exec, device=self.device)
                            if _selective_reward:
                                _min_effective_gain = float(self.cfg.get("frontres_min_effective_gain", 0.006))
                                _bonus_weight = float(self.cfg.get("frontres_effective_gain_bonus_weight", 0.5))
                                _effective_gain_bonus_exec = (
                                    _bonus_weight
                                    * _repair_gate
                                    * torch.relu(_repair_gain - _min_effective_gain)
                                )
                            _effective_gain_bonus = torch.zeros(N_train, device=self.device)
                            _effective_gain_bonus[:_n_exec] = _effective_gain_bonus_exec
                            _accept_pref_target = torch.zeros(self.env.num_envs, 6, device=self.device)
                            _accept_pref_mask = torch.zeros(self.env.num_envs, 6, device=self.device)
                            _pref_full_frac = torch.tensor(0.0, device=self.device)
                            _pref_noop_frac = torch.tensor(0.0, device=self.device)
                            _pref_keep_frac = torch.tensor(0.0, device=self.device)
                            _pref_ignore_frac = torch.tensor(1.0, device=self.device)
                            _pref_margin_mean = torch.tensor(0.0, device=self.device)
                            _pref_need_mean = torch.tensor(0.0, device=self.device)
                            _pref_admiss_mean = torch.tensor(0.0, device=self.device)
                            _pref_target_mean = torch.tensor(0.0, device=self.device)
                            _tri_weight_repair_mean = torch.tensor(0.0, device=self.device)
                            _tri_weight_noisy_mean = torch.tensor(0.0, device=self.device)
                            _tri_weight_stable_mean = torch.tensor(0.0, device=self.device)
                            _pref_inertial_penalty_rho_mean = torch.tensor(0.0, device=self.device)
                            _pref_inertial_penalty_one_mean = torch.tensor(0.0, device=self.device)
                            _pref_enabled = (
                                bool(self.cfg.get("frontres_acceptance_preference_enabled", True))
                                and N_candidate > 0
                                and _n_exec > 0
                            )
                            if _pref_enabled:
                                _pref_margin = max(
                                    float(self.cfg.get("frontres_acceptance_preference_margin", 0.003)),
                                    0.0,
                                )
                                _j_rho = _repair_gain
                                _j_one = _candidate_gain
                                _j_zero = torch.zeros_like(_repair_gain)
                                _c_zero = None
                                _c_one = None
                                if bool(self.cfg.get("frontres_inertial_preference_enabled", False)):
                                    _cmd_for_inertia = _mcmd_rdelta
                                    _have_inertia = (
                                        hasattr(_cmd_for_inertia, "robot_anchor_quat_w")
                                        and hasattr(_cmd_for_inertia, "robot_anchor_ang_vel_w")
                                    )
                                    if _have_inertia:
                                        _robot_q = _cmd_for_inertia.robot_anchor_quat_w[:_n_exec].to(self.device)
                                        _robot_w = _cmd_for_inertia.robot_anchor_ang_vel_w[:_n_exec].to(self.device)
                                        _robot_p = (
                                            _cmd_for_inertia.robot_anchor_pos_w[:_n_exec].to(self.device)
                                            if hasattr(_cmd_for_inertia, "robot_anchor_pos_w")
                                            else None
                                        )
                                        _robot_v = (
                                            _cmd_for_inertia.robot_anchor_lin_vel_w[:_n_exec].to(self.device)
                                            if hasattr(_cmd_for_inertia, "robot_anchor_lin_vel_w")
                                            else None
                                        )
                                        _ang_w = float(self.cfg.get("frontres_inertial_preference_ang_weight", 0.5))

                                        def _branch_compat(_branch_pos, _branch_q):
                                            _rot_err = _quat_to_rotvec_wxyz(
                                                quat_mul(quat_inv(_robot_q), _branch_q)
                                            )[:, :3]
                                            _compat = torch.zeros(_n_exec, device=self.device, dtype=_j_rho.dtype)
                                            if _branch_pos is not None and _robot_p is not None and _robot_v is not None:
                                                _pos_err = _branch_pos - _robot_p
                                                _compat = _compat + (_pos_err * _robot_v).sum(-1) / (
                                                    _pos_err.norm(dim=-1) * _robot_v.norm(dim=-1) + 1e-8
                                                )
                                            _compat = _compat + _ang_w * (_rot_err * _robot_w).sum(-1) / (
                                                _rot_err.norm(dim=-1) * _robot_w.norm(dim=-1) + 1e-8
                                            )
                                            return torch.nan_to_num(_compat, nan=0.0, posinf=0.0, neginf=0.0)

                                        _pos_all = getattr(_cmd_for_inertia, "anchor_pos_w", None)
                                        _quat_all = getattr(_cmd_for_inertia, "anchor_quat_w", None)
                                        if _quat_all is not None:
                                            _noisy_pos = (
                                                _pos_all[_base_start:_base_start + _n_exec].to(self.device)
                                                if _pos_all is not None
                                                else None
                                            )
                                            _rho_pos = (
                                                _pos_all[:_n_exec].to(self.device)
                                                if _pos_all is not None
                                                else None
                                            )
                                            _one_pos = (
                                                _pos_all[_candidate_start:_candidate_start + _n_exec].to(self.device)
                                                if _pos_all is not None and N_candidate > 0
                                                else None
                                            )
                                            _c_zero = _branch_compat(
                                                _noisy_pos,
                                                _quat_all[_base_start:_base_start + _n_exec].to(self.device),
                                            )
                                            _c_rho = _branch_compat(
                                                _rho_pos,
                                                _quat_all[:_n_exec].to(self.device),
                                            )
                                            _c_one = _branch_compat(
                                                _one_pos,
                                                _quat_all[_candidate_start:_candidate_start + _n_exec].to(self.device),
                                            )
                                            _inertial_margin = float(
                                                self.cfg.get("frontres_inertial_preference_margin", 0.05)
                                            )
                                            _inertial_weight = max(
                                                0.0,
                                                float(self.cfg.get("frontres_inertial_preference_weight", 0.0)),
                                            )
                                            _penalty_rho = torch.relu(_c_zero - _c_rho + _inertial_margin)
                                            _penalty_one = torch.relu(_c_zero - _c_one + _inertial_margin)
                                            _j_rho = _j_rho - _inertial_weight * _penalty_rho
                                            _j_one = _j_one - _inertial_weight * _penalty_one
                                            _pref_inertial_penalty_rho_mean = _penalty_rho.mean()
                                            _pref_inertial_penalty_one_mean = _penalty_one.mean()
                                _full_win = (_j_one > _j_rho + _pref_margin) & (_j_one > _j_zero + _pref_margin)
                                _noop_win = (_j_zero > _j_rho + _pref_margin) & (_j_zero > _j_one + _pref_margin)
                                _keep_win = (_j_rho > _j_one + _pref_margin) & (_j_rho > _j_zero + _pref_margin)
                                _pref_gate = (
                                    _oracle_trust * _repair_gate * _learnable_route_mask
                                ).detach().clamp(0.0, 1.0)
                                _target_exec = torch.zeros(_n_exec, 6, device=self.device)
                                _mask_exec = torch.zeros(_n_exec, 6, device=self.device)
                                _rho_current = torch.ones(_n_exec, 6, device=self.device)
                                _task_conf_dim = int(getattr(getattr(self.alg, "policy", None), "task_conf_dim", 2))
                                if actions.shape[-1] >= 12 and _task_conf_dim == 6:
                                    _rho_current = actions[:_n_exec, 6:12].detach().clamp(0.0, 1.0)
                                elif actions.shape[-1] >= 7 and _task_conf_dim == 1:
                                    _rho_current = actions[:_n_exec, 6:7].detach().clamp(0.0, 1.0).expand(-1, 6)
                                _rho_space = str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                                if _rho_space in ("tri_anchor", "tri-anchor", "tri"):
                                    _rho_temp = max(
                                        1e-6,
                                        float(self.cfg.get("frontres_acceptance_rho_target_temp", 0.08)),
                                    )
                                    _target_scalar = torch.sigmoid(
                                        (_j_one - _j_zero - _pref_margin) / _rho_temp
                                    ).clamp(0.0, 1.0)
                                    _target_exec = _target_scalar.view(-1, 1).expand(-1, 6).clone()
                                    _mask_exec = _pref_gate.view(-1, 1).expand(-1, 6).clone()
                                    _tri_alpha = _state_alpha_target[:_n_exec, 0].detach().clamp(0.0, 1.0)
                                    _tri_weight_repair_mean = _target_scalar.mean()
                                    _tri_weight_stable_mean = ((1.0 - _target_scalar) * _tri_alpha).mean()
                                    _tri_weight_noisy_mean = ((1.0 - _target_scalar) * (1.0 - _tri_alpha)).mean()
                                    _state_alpha_mask[:_n_exec, 0] = (
                                        _state_alpha_mask[:_n_exec, 0]
                                        * (1.0 - _target_scalar.detach()).clamp(0.0, 1.0)
                                    )
                                    _pref_full_frac = (_target_scalar > 0.66).float().mean()
                                    _pref_noop_frac = (
                                        ((1.0 - _target_scalar) * (1.0 - _tri_alpha)) > 0.50
                                    ).float().mean()
                                    _pref_keep_frac = (
                                        (_target_scalar >= 0.33) & (_target_scalar <= 0.66)
                                    ).float().mean()
                                    _pref_target_mean = _target_scalar.mean()
                                elif bool(self.cfg.get("frontres_acceptance_direct_target_enabled", False)):
                                    _clean_pos = _a_w[_base_start:_base_start + _n_exec].detach()
                                    _noisy_pos = _a_raw[_base_start:_base_start + _n_exec].detach()
                                    _candidate_clean_pos = _a_w[
                                        _candidate_start:_candidate_start + _n_exec
                                    ].detach()
                                    _candidate_pos = _a_fr[_candidate_start:_candidate_start + _n_exec].detach()
                                    _clean_q = _q_w[_base_start:_base_start + _n_exec].detach()
                                    _noisy_q = _q_raw[_base_start:_base_start + _n_exec].detach()
                                    _candidate_clean_q = _q_w[
                                        _candidate_start:_candidate_start + _n_exec
                                    ].detach()
                                    _candidate_q = _q_fr[_candidate_start:_candidate_start + _n_exec].detach()
                                    _err0_vec = torch.cat(
                                        [
                                            (_clean_pos - _noisy_pos),
                                            _quat_to_rotvec_wxyz(quat_mul(quat_inv(_noisy_q), _clean_q)),
                                        ],
                                        dim=-1,
                                    )
                                    _err1_vec = torch.cat(
                                        [
                                            (_candidate_clean_pos - _candidate_pos),
                                            _quat_to_rotvec_wxyz(
                                                quat_mul(quat_inv(_candidate_q), _candidate_clean_q)
                                            ),
                                        ],
                                        dim=-1,
                                    )
                                    _err0 = _err0_vec.abs()
                                    _err1 = _err1_vec.abs()
                                    _need = ((_err0 - _err1) / _err0.clamp(min=1e-6)).clamp(0.0, 1.0)
                                    _need_min = max(
                                        0.0,
                                        float(self.cfg.get("frontres_acceptance_need_min_error", 0.01)),
                                    )
                                    _need_mask = (_err0 > _need_min).to(_need.dtype)
                                    _admissibility = torch.ones(_n_exec, device=self.device, dtype=_need.dtype)
                                    if _c_zero is not None and _c_one is not None:
                                        _inertial_margin = float(
                                            self.cfg.get("frontres_inertial_preference_margin", 0.05)
                                        )
                                        _temp = max(
                                            1e-6,
                                            float(self.cfg.get("frontres_acceptance_admissibility_temp", 0.20)),
                                        )
                                        _admissibility = torch.sigmoid((_c_one - _c_zero - _inertial_margin) / _temp)
                                    _target_exec = torch.minimum(
                                        _need, _admissibility.view(-1, 1)
                                    ).clamp(0.0, 1.0)
                                    _mask_exec = _need_mask * _pref_gate.view(-1, 1)
                                else:
                                    # Calibrate acceptance around the current write
                                    # instead of forcing a full/no-op binary label.
                                    # This keeps the quartet rollout budget fixed:
                                    # Noisy (0), Projected (rho), Candidate (1).
                                    _calib_step = float(self.cfg.get("frontres_acceptance_calibration_step", 0.5))
                                    _calib_step = max(0.0, min(1.0, _calib_step))
                                    _denom = (_j_one - _j_zero).abs().clamp(min=1e-6)
                                    _target_exec = _rho_current.clone()
                                    _raise = ((_j_one - _j_rho) / _denom).clamp(0.0, 1.0)
                                    _lower = ((_j_zero - _j_rho) / _denom).clamp(0.0, 1.0)
                                    if bool(_full_win.any().detach().item()):
                                        _target_exec[_full_win] = (
                                            _rho_current[_full_win]
                                            + _calib_step * _raise[_full_win].view(-1, 1)
                                        ).clamp(0.0, 1.0)
                                    if bool(_noop_win.any().detach().item()):
                                        _target_exec[_noop_win] = (
                                            _rho_current[_noop_win]
                                            - _calib_step * _lower[_noop_win].view(-1, 1)
                                        ).clamp(0.0, 1.0)
                                    # If Projected wins, current rho is the best observed
                                    # point on the existing quartet and should be anchored.
                                    _trainable = (_full_win | _noop_win | _keep_win).to(_target_exec.dtype)
                                    _mask_exec = _trainable.view(-1, 1) * _pref_gate.view(-1, 1)
                                if bool(self.cfg.get("frontres_per_mode_acceptance_preference_mask", True)):
                                    _mode_dim_mask = self._frontres_mode_dim_mask(
                                        _mode_groups, _n_exec, self.device, _mask_exec.dtype
                                    )
                                    _mask_exec = _mask_exec * _mode_dim_mask
                                _active_dims_cfg = self.cfg.get("frontres_active_task_dims", None)
                                if _active_dims_cfg is not None:
                                    _dim_mask = torch.zeros(6, device=self.device, dtype=_mask_exec.dtype)
                                    for _idx in _active_dims_cfg:
                                        _idx = int(_idx)
                                        if 0 <= _idx < 6:
                                            _dim_mask[_idx] = 1.0
                                        elif 6 <= _idx < 12:
                                            _dim_mask[_idx - 6] = 1.0
                                    _mask_exec = _mask_exec * _dim_mask.view(1, -1)
                                _accept_pref_target[:_n_exec] = _target_exec.detach()
                                _accept_pref_mask[:_n_exec] = _mask_exec.detach()
                                self._frontres_state_alpha_mask_last = float(
                                    _state_alpha_mask[:_n_exec, 0].mean().detach().item()
                                )
                                _active_pref = _mask_exec.sum(dim=-1) > 0
                                if bool(_active_pref.any().detach().item()):
                                    _target_sample = (
                                        (_target_exec * _mask_exec).sum(dim=-1)
                                        / _mask_exec.sum(dim=-1).clamp(min=1e-6)
                                    )
                                    _active_target = _target_sample[_active_pref]
                                    _pref_full_frac = (_active_target > 0.66).float().mean()
                                    _pref_noop_frac = (_active_target < 0.33).float().mean()
                                    _pref_keep_frac = (
                                        ((_active_target >= 0.33) & (_active_target <= 0.66)).float().mean()
                                    )
                                    _pref_target_mean = _active_target.mean()
                                    if bool(self.cfg.get("frontres_acceptance_direct_target_enabled", False)):
                                        _need_sample = (
                                            (_need * _mask_exec).sum(dim=-1)
                                            / _mask_exec.sum(dim=-1).clamp(min=1e-6)
                                        )
                                        _pref_need_mean = _need_sample[_active_pref].mean()
                                        _admiss_sample = (
                                            (_admissibility.view(-1, 1) * _mask_exec).sum(dim=-1)
                                            / _mask_exec.sum(dim=-1).clamp(min=1e-6)
                                        )
                                        _pref_admiss_mean = _admiss_sample[_active_pref].mean()
                                _pref_ignore_frac = (~_active_pref).float().mean()
                                _best = torch.maximum(torch.maximum(_j_one, _j_rho), _j_zero)
                                _second = torch.minimum(
                                    torch.maximum(_j_one, _j_rho),
                                    torch.maximum(torch.minimum(_j_one, _j_rho), _j_zero),
                                )
                                _pref_margin_mean = (_best - _second).mean()
                            self.alg.transition.acceptance_target = _accept_pref_target
                            self.alg.transition.acceptance_mask = _accept_pref_mask
                            _ranking_enabled = bool(self.cfg.get("frontres_candidate_ranking_reward_enabled", True)) and N_candidate > 0
                            if _ranking_enabled:
                                _ranking_under_weight = float(self.cfg.get("frontres_candidate_underwrite_weight", 1.0))
                                _ranking_projection_weight = float(self.cfg.get("frontres_candidate_projection_weight", 0.25))
                                _ranking_harm_weight = float(self.cfg.get("frontres_candidate_harm_weight", 1.0))
                                _under_write = torch.relu(_candidate_gain - _repair_gain) * (_candidate_gain > 0.0).to(_repair_gain.dtype)
                                _ranking_exec = (
                                    _repair_gain
                                    + _ranking_projection_weight * _projection_gain
                                    - _ranking_under_weight * _under_write
                                    - _ranking_harm_weight * torch.relu(-_repair_gain)
                                )
                                _ranking_reward = torch.zeros(N_train, device=self.device)
                                _ranking_reward[:_n_exec] = _exec_gate * _ranking_exec
                            else:
                                _under_write = torch.zeros_like(_repair_gain)
                                _ranking_reward = torch.zeros(N_train, device=self.device)
                            _positive_reward = (
                                _w_exec * _repair_scale * _exec_weight * _r_exec
                                + _w_exec * _repair_scale * _effective_gain_bonus
                                + _w_geom * _r_step
                                + float(self.cfg.get("frontres_candidate_ranking_reward_weight", 1.0)) * _ranking_reward
                                + _w_rescue * _r_rescue
                            )
                            _constraint_penalty = (
                                _w_exec_harm * _harm_penalty
                                + _cost_weight * _intervention_cost
                                + _overcorrection_cost
                                + _under_repair_penalty
                            )
                            r_delta = _reward_progress * _positive_reward - _constraint_progress * _constraint_penalty
                            _r_frontres_log = _exec_frontres.mean()
                            _r_clean_log = _exec_clean.mean()
                            _r_oracle_log = _exec_feasible.mean()
                            _r_base_log   = _exec_perturbed.mean()
                            _r_rescue_log = _r_rescue.mean()
                        else:
                            r_raw_gmt = rewards[_base_start:_base_end].view(-1).clone()
                            r_total   = rewards[:N_train].view(-1)
                            if N_train == N_base:
                                r_delta = r_total - r_raw_gmt
                                _r_base_log = r_raw_gmt.mean()
                            else:
                                r_delta = r_total - r_raw_gmt.mean()
                                _r_base_log = r_raw_gmt.mean()
                            _r_rescue_log = 0.0
                            _r_z = _r_xy = _r_rp = _r_ya = None
                            _actor_gate = None
                            _exec_planar_log = _exec_vertical_log = _exec_task_log = None
                            _r_clean_log = _r_oracle_log = None
                            _frontres_actor_gate = torch.zeros(self.env.num_envs, 1, device=self.device)
                            _frontres_actor_gate[:N_train, 0] = 1.0
                            self.alg.transition.frontres_actor_gate = _frontres_actor_gate
                            _dr_z_abs_log = _dr_xy_abs_log = _dr_rp_abs_log = _dr_yaw_abs_log = None
                            _corr_z_abs_log = _corr_xy_abs_log = _corr_rp_abs_log = _corr_yaw_abs_log = None
                        # ─────────────────────────────────────────────────────────────

                        rewards_mod = rewards.clone()
                        if rewards_mod.dim() == 2:
                            rewards_mod[:N_train] = r_delta.unsqueeze(-1)
                            rewards_mod[N_train:] = 0.0  # zero GMT → V(s) target = 0 → advantage = 0
                        else:
                            rewards_mod[:N_train] = r_delta
                            rewards_mod[N_train:] = 0.0
                        rewards = rewards_mod

                        # Optional smoothness penalty on top of delta reward
                        _lambda_smooth = getattr(self.alg, 'lambda_smooth', 0.0)
                        if _lambda_smooth > 0.0 and _frontres_prev_delta_q is not None:
                            _diff = actions[:N_train] - _frontres_prev_delta_q[:N_train]  # [N_train, A]
                            _smooth_penalty = -_lambda_smooth * _diff.pow(2).mean(dim=-1)  # [N_train]
                            if rewards.dim() == 2:
                                rewards[:N_train] = rewards[:N_train] + _smooth_penalty.unsqueeze(-1)
                            else:
                                rewards[:N_train] = rewards[:N_train] + _smooth_penalty
                            _frontres_smooth_penalty_sum += _smooth_penalty.mean().item()

                        # lambda_reg reward shaping: penalize ||Δ||² to discourage unbounded corrections.
                        # Gated by _dr_done: do NOT apply during DR curriculum ramp-up, so Actor can
                        # freely explore non-zero residuals without the penalty reinforcing the Δ=0 shortcut.
                        # Uses sampled actions (unbiased estimator of mean penalty); mean over joints
                        # so scale is per-joint and consistent with per-step reward magnitude.
                        if _lambda_reg > 0.0 and _dr_done:
                            _reg_penalty = -_lambda_reg * actions[:N_train].pow(2).mean(dim=-1)  # [N_train]
                            if rewards.dim() == 2:
                                rewards[:N_train] = rewards[:N_train] + _reg_penalty.unsqueeze(-1)
                            else:
                                rewards[:N_train] = rewards[:N_train] + _reg_penalty
                            _frontres_reg_penalty_sum += _reg_penalty.mean().item()

                        # Logging accumulators
                        _frontres_rdelta_sum   += r_delta.mean().item()
                        _frontres_baseline_sum += _r_base_log.item()
                        if _r_z is not None:
                            _frontres_r_z_sum          += _r_z.mean().item()
                            _frontres_r_xy_sum         += _r_xy.mean().item()
                            _frontres_r_rp_sum         += _r_rp.mean().item()
                            _frontres_r_yaw_sum        += _r_ya.mean().item()
                            _frontres_r_rescue_sum     += float(_r_rescue_log)
                            _frontres_r_exec_sum       += _r_exec.mean().item()
                            _frontres_r_geom_sum       += _r_step.mean().item()
                            _frontres_intervention_cost_sum += _intervention_cost.mean().item()
                            _frontres_clean_bound_cost_sum += _clean_bound_cost.mean().item()
                            _frontres_clean_bound_side_cost_sum += _side_cost.mean().item()
                            _frontres_over_cost_sum += _over_cost.mean().item()
                            _frontres_under_repair_cost_sum += _under_repair_penalty.mean().item()
                            _frontres_reward_frontres_sum += _r_frontres_log.item()
                            _frontres_reward_clean_sum += _r_clean_log.item()
                            _frontres_reward_candidate_sum += _exec_candidate.mean().item()
                            _frontres_reward_oracle_sum += _r_oracle_log.item()
                            _frontres_exec_planar_sum += _exec_planar_log.item()
                            _frontres_exec_vertical_sum += _exec_vertical_log.item()
                            _frontres_exec_task_sum += _exec_task_log.item()
                            _frontres_damage_gap_sum   += _damage_gap.mean().item()
                            _frontres_oracle_clean_gap_sum += _oracle_clean_gap.mean().item()
                            _frontres_oracle_trust_sum += _oracle_trust.mean().item()
                            _frontres_repair_gain_sum  += _repair_gain.mean().item()
                            _frontres_candidate_gain_sum += _candidate_gain.mean().item()
                            _frontres_projection_gain_sum += _projection_gain.mean().item()
                            _frontres_underwrite_sum += _under_write.mean().item()
                            _frontres_accept_pref_mask_sum += (
                                (_accept_pref_mask[:_n_exec].sum(dim=-1) > 0).float().mean().item()
                            )
                            _frontres_accept_pref_full_sum += _pref_full_frac.item()
                            _frontres_accept_pref_noop_sum += _pref_noop_frac.item()
                            _frontres_accept_pref_keep_sum += _pref_keep_frac.item()
                            _frontres_accept_pref_ignore_sum += _pref_ignore_frac.item()
                            _frontres_accept_pref_margin_sum += _pref_margin_mean.item()
                            _frontres_accept_pref_need_sum += _pref_need_mean.item()
                            _frontres_accept_pref_admiss_sum += _pref_admiss_mean.item()
                            _frontres_accept_pref_target_sum += _pref_target_mean.item()
                            _frontres_inertial_pref_penalty_rho_sum += _pref_inertial_penalty_rho_mean.item()
                            _frontres_inertial_pref_penalty_one_sum += _pref_inertial_penalty_one_mean.item()
                            _frontres_positive_gain_frac_sum += (_repair_gain > 0.0).float().mean().item()
                            _frontres_repair_ratio_sum += _repair_ratio.mean().item()
                            _frontres_exec_signal_sum += _r_exec[:_n_exec].mean().item()
                            _frontres_weighted_exec_signal_sum += (
                                _exec_weight[:_n_exec] * _r_exec[:_n_exec]
                            ).mean().item()
                            _frontres_train_reward_sum += r_delta[:_n_exec].mean().item()
                            _frontres_reward_progress_sum += float(_reward_progress)
                            _frontres_constraint_progress_sum += float(_constraint_progress)
                            _frontres_effective_gain_bonus_sum += _effective_gain_bonus[:_n_exec].mean().item()
                            _frontres_safe_cost_sum += (
                                _safe_gate * _cost_exec
                            ).mean().item()
                            _frontres_repair_cost_sum += (
                                _repair_gate * _cost_exec
                            ).mean().item()
                            _frontres_broken_cost_sum += (
                                _broken_gate * _cost_exec
                            ).mean().item()
                            _eps_fit = 1e-6
                            _mu_sum = _window_mu.sum().clamp(min=_eps_fit)
                            _repair_fit_num = (_window_mu * _repair_gain).sum()
                            _repair_fit_gap = (_window_mu * _damage_gap).sum().clamp(min=_eps_fit)
                            _repair_fit_rate = _repair_fit_num / _repair_fit_gap
                            _repair_fit_gain = _repair_fit_num / _mu_sum
                            _harm_indicator = (_harm_mag > 0.0).float()
                            _harm_rate = (_window_mu * _harm_indicator).sum() / _mu_sum
                            _harm_mag_fit = (_window_mu * _harm_mag).sum() / _mu_sum
                            _safe_mask = (_damage_gap < _safe_gap).float()
                            _broken_mask = (_damage_gap > _broken_gap).float()
                            _safe_den = _safe_mask.sum().clamp(min=_eps_fit)
                            _broken_den = _broken_mask.sum().clamp(min=_eps_fit)
                            _safe_harm_rate = (_safe_mask * _harm_indicator).sum() / _safe_den
                            _broken_harm_rate = (_broken_mask * _harm_indicator).sum() / _broken_den
                            _safe_abstain_cost = (_safe_mask * _cost_exec).sum() / _safe_den
                            _broken_abstain_cost = (_broken_mask * _cost_exec).sum() / _broken_den
                            _behavior_fit_num = (
                                (_window_mu * _repair_gain).sum()
                                + _effective_gain_bonus_exec.sum()
                                - _harm_penalty_exec.sum()
                                - (_cost_gate * _cost_exec).sum()
                            )
                            _behavior_fit_den = (
                                (_window_mu * _damage_gap).sum()
                                + (_cost_gate * _cost_exec).sum()
                            ).clamp(min=_eps_fit)
                            _behavior_fit = _behavior_fit_num / _behavior_fit_den
                            _frontres_behavior_fit_sum += _behavior_fit.item()
                            _frontres_repair_fit_rate_sum += _repair_fit_rate.item()
                            _frontres_repair_fit_gain_sum += _repair_fit_gain.item()
                            _restore_eval_min = float(self.cfg.get("frontres_restore_eval_min_error", 1e-3))
                            _restore_eval_mask = _e_raw > max(_restore_eval_min, 1e-8)
                            if _restore_eval_mask.any():
                                _restore_ratio_rp = 1.0 - (
                                    _e_fr[_restore_eval_mask] / _e_raw[_restore_eval_mask].clamp(min=1e-6)
                                )
                                _frontres_restore_ratio_rp_sum += _restore_ratio_rp.mean().item()
                            else:
                                _frontres_restore_ratio_rp_sum += 0.0
                            _frontres_residual_rp_abs_sum += _e_fr.mean().item()
                            _frontres_corr_roll_bias_sum += _rot_raw_to_fr[:, 0].mean().item()
                            _frontres_corr_pitch_bias_sum += _rot_raw_to_fr[:, 1].mean().item()
                            _frontres_harm_rate_sum += _harm_rate.item()
                            _frontres_harm_mag_sum += _harm_mag_fit.item()
                            _frontres_safe_harm_rate_sum += _safe_harm_rate.item()
                            _frontres_broken_harm_rate_sum += _broken_harm_rate.item()
                            _frontres_safe_abstain_cost_sum += _safe_abstain_cost.item()
                            _frontres_broken_abstain_cost_sum += _broken_abstain_cost.item()
                            _frontres_window_mu_sum    += _window_mu.mean().item()
                            _frontres_safe_frac_sum    += _safe_frac.item()
                            _frontres_repair_frac_sum  += _repair_frac.item()
                            _frontres_broken_frac_sum  += _broken_frac.item()
                            _frontres_candidate_floor_margin_sum += _candidate_floor_margin.mean().item()
                            _frontres_candidate_floor_pass_sum += _candidate_floor_pass_frac.item()
                            _frontres_stable_route_frac_sum += float(
                                getattr(self, "_frontres_stable_route_applied_frac", 0.0)
                            )
                            _frontres_stable_endpoint_frac_sum += float(
                                getattr(self, "_frontres_stable_endpoint_frac", 0.0)
                            )
                            _frontres_tri_weight_repair_sum += float(
                                getattr(self, "_frontres_tri_weight_repair", 0.0)
                            )
                            _frontres_tri_weight_noisy_sum += float(
                                getattr(self, "_frontres_tri_weight_noisy", 1.0)
                            )
                            _frontres_tri_weight_stable_sum += float(
                                getattr(self, "_frontres_tri_weight_stable", 0.0)
                            )
                            _frontres_state_alpha_pred_sum += float(
                                getattr(self, "_frontres_state_alpha_pred_last", 0.0)
                            )
                            _frontres_state_alpha_target_sum += float(
                                getattr(self, "_frontres_state_alpha_target_last", 0.0)
                            )
                            _frontres_state_alpha_mask_sum += float(
                                getattr(self, "_frontres_state_alpha_mask_last", 0.0)
                            )
                            _frontres_state_alpha_route_sum += float(
                                getattr(self, "_frontres_state_alpha_route_last", 0.0)
                            )
                            _frontres_actor_gate_sum   += _actor_gate.mean().item()
                            _frontres_exec_gate_sum    += _exec_gate.mean().item()
                            _frontres_cost_gate_sum    += _cost_gate.mean().item()
                            _frontres_reward_diag_steps += 1
                            _frontres_dr_z_abs_sum     += _dr_z_abs_log.item()
                            _frontres_dr_xy_abs_sum    += _dr_xy_abs_log.item()
                            _frontres_dr_rp_abs_sum    += _dr_rp_abs_log.item()
                            _frontres_dr_yaw_abs_sum   += _dr_yaw_abs_log.item()
                            _frontres_corr_z_abs_sum   += _corr_z_abs_log.item()
                            _frontres_corr_xy_abs_sum  += _corr_xy_abs_log.item()
                            _frontres_corr_rp_abs_sum  += _corr_rp_abs_log.item()
                            _frontres_corr_yaw_abs_sum += _corr_yaw_abs_log.item()
                        # Task-space mode: split into Δpos and Δrpy; joint-space mode: log Δz
                        if _is_task_space_mode:
                            _tc = getattr(self.alg.policy, 'last_task_correction', None)
                            if _tc is not None:
                                _frontres_delta_pos_abs_sum += _tc[:N_train, :3].abs().mean().item()
                                _frontres_delta_rpy_abs_sum += _tc[:N_train, 3:6].abs().mean().item()
                            # Accumulate jump_degree for gate activation monitoring
                            for _cmd_term in (self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env).command_manager._terms.values():
                                if hasattr(_cmd_term, 'jump_degree'):
                                    _frontres_jump_degree_sum += _cmd_term.jump_degree[:N_train].mean().item()
                                    break
                        else:
                            _dz = getattr(self.alg.policy, 'last_delta_z', None)
                            if _dz is not None:
                                _frontres_delta_z_abs_sum += _dz.abs().mean().item()
                        _frontres_shaping_steps += 1
                        # Adaptive DR: count terminations in training envs
                        _frontres_term_count += int((dones[:N_train] > 0).sum().item())
                        _frontres_step_count += N_train

                        # Update previous residual action for smoothness tracking (all N envs)
                        _done_mask = dones.bool().view(-1)
                        if _frontres_prev_delta_q is None:
                            _frontres_prev_delta_q = actions.clone()
                        else:
                            _frontres_prev_delta_q = actions.clone()
                            _frontres_prev_delta_q[_done_mask] = 0.0

                        # Reset low-pass filter state for terminated FrontRES envs.
                        # Without this, the filter carries old correction into the new episode:
                        # new OU=0 → FrontRES predicts ≈0, but filter outputs 0.6×old_value
                        # for 4-5 steps → false anchor_pos/ori terminations immediately.
                        _prev_pos_c = getattr(self, '_prev_pos_correction', None)
                        if _prev_pos_c is not None:
                            _fr_done_mask = _done_mask[:N_train]
                            if _fr_done_mask.any():
                                _prev_pos_c[_fr_done_mask] = 0.0
                                self._prev_pos_correction = _prev_pos_c
                    # ── END FrontRES B1 delta-reward ─────────────────────────────────────

                    obs_dict = infos.get("observations", {})
                    if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
                        obs = obs_dict[self.policy_obs_type].to(self.device)
                    else:
                        obs = obs.to(self.device)

                    # For Stage 1: update raw obs BEFORE normalization for GMT ONNX next step
                    if self.training_type == "supervise":
                        obs_raw_for_gmt = obs.clone()

                    # perform normalization 对本次循环的观测量进行归一化, 用于计算下步动作
                    obs = self._apply_obs_normalizer(obs)
                    if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
                        privileged_obs = self.privileged_obs_normalizer(
                            obs_dict[self.privileged_obs_type].to(self.device))
                    else:
                        privileged_obs = obs
                    if self.teacher_obs_type is not None and self.teacher_obs_type in obs_dict:
                        teacher_obs = self.teacher_obs_normalizer(
                            obs_dict[self.teacher_obs_type].to(self.device))
                    else:
                        teacher_obs = privileged_obs
                    
                    # Extract ref_vel_estimator observations (NO normalization - must match offline training!)
                    if self.ref_vel_estimator_obs_type is not None and self.ref_vel_estimator_obs_type in obs_dict:
                        ref_vel_estimator_obs = obs_dict[self.ref_vel_estimator_obs_type].to(self.device)
                    else: # 提取速度估计器的速度观测量
                        ref_vel_estimator_obs = None

                    # process the step 更新回放池的数据 (奖励值, 完成布尔值, 额外信息)
                    self.alg.process_env_step(rewards, dones, infos)  # stores FrontRES residual actions, not GMT robot actions

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if hasattr(self.alg, "rnd") and self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        
                        # Update rewards
                        if hasattr(self.alg, "rnd") and self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards  # GMT envs contribute 0 (zeroed above)
                        # GMT raw reward tracking (separate, uses pre-zeroed values saved earlier)
                        if _is_frontres:
                            if N_candidate > 0 and r_candidate_gmt is not None:
                                cur_reward_sum_gmt[:N_candidate] += r_candidate_gmt
                            _gmt_base_local = N_candidate
                            cur_reward_sum_gmt[_gmt_base_local:_gmt_base_local + N_base] += r_raw_gmt  # noisy GMT per-step reward
                            if N_clean > 0:
                                _gmt_clean_local = _gmt_base_local + N_base
                                cur_reward_sum_gmt[_gmt_clean_local:_gmt_clean_local + N_clean] += r_clean_gmt
                        
                        # Update episode length
                        cur_episode_length += 1

                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if _is_frontres and len(new_ids) > 0:
                            # B1: split done envs into FrontRES (r_delta) and GMT (raw reward + length)
                            _env_idx  = new_ids[:, 0]
                            _fr_done  = new_ids[_env_idx < N_train]
                            _gmt_done = new_ids[_env_idx >= N_train]
                            if len(_fr_done) > 0:
                                rewbuffer.extend(cur_reward_sum[_fr_done][:, 0].cpu().numpy().tolist())
                                lenbuffer.extend(cur_episode_length[_fr_done][:, 0].cpu().numpy().tolist())
                            if len(_gmt_done) > 0:
                                _gmt_local = _gmt_done[:, 0] - N_train
                                rewbuffer_gmt.extend(cur_reward_sum_gmt[_gmt_local].cpu().numpy().tolist())
                                lenbuffer_gmt.extend(cur_episode_length[_gmt_done][:, 0].cpu().numpy().tolist())
                                _base_mask = (_gmt_local >= N_candidate) & (_gmt_local < N_candidate + N_base)
                                if bool(_base_mask.any().item()):
                                    lenbuffer_gmt_base.extend(
                                        cur_episode_length[_gmt_done[_base_mask]][:, 0].cpu().numpy().tolist()
                                    )
                                    _mix_class = getattr(self, "_frontres_dr_mix_class_train", None)
                                    if _mix_class is not None and N_base > 0:
                                        _base_local = _gmt_local[_base_mask] - N_candidate
                                        _base_local = _base_local.to(device=self.device, dtype=torch.long)
                                        _valid = (_base_local >= 0) & (_base_local < int(_mix_class.numel()))
                                        if bool(_valid.any().item()):
                                            _frontier_mask = _mix_class[_base_local[_valid]] == 1
                                            if bool(_frontier_mask.any().item()):
                                                _frontier_done = _gmt_done[_base_mask][_valid][_frontier_mask]
                                                lenbuffer_gmt_base_frontier.extend(
                                                    cur_episode_length[_frontier_done][:, 0].cpu().numpy().tolist()
                                                )
                                cur_reward_sum_gmt[_gmt_local] = 0
                        elif len(new_ids) > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        # -- intrinsic and extrinsic rewards
                        if hasattr(self.alg, "rnd") and self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns 计算广义优势值
                if self.training_type in ["rl", "mosaic", "frontres"]:
                    self.alg.compute_returns(privileged_obs)

            # update policy Rollout结束, 开始使用buffer计算Loss更新权重
            # Pass current iteration to algorithm for logging (needed by MOSAIC)
            self.alg.current_learning_iteration = it
            if _is_frontres and hasattr(self.alg, "ppo_actor_weight"):
                self.alg.ppo_actor_weight = _ppo_actor_weight_current
            # Pass oracle_mix so MOSAIC scales surrogate by (1 - oracle_mix):
            # PPO contribution ∝ FrontRES causal share of the correction applied.
            self.alg.oracle_mix = getattr(self, '_oracle_mix', 0.0)
            loss_dict = self.alg.update() # 调用mosaic.py中的update()函数进行权重更新

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # expose curriculum state and B1 delta-reward metrics to log() via locals()
            frontres_rdelta_mean   = (_frontres_rdelta_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_baseline_mean = (_frontres_baseline_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_z_mean      = (_frontres_r_z_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_xy_mean     = (_frontres_r_xy_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_rp_mean     = (_frontres_r_rp_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_yaw_mean    = (_frontres_r_yaw_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_rescue_mean = (_frontres_r_rescue_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_exec_mean = (_frontres_r_exec_sum / _frontres_reward_diag_steps
                                    if _is_frontres and _frontres_reward_diag_steps > 0 else None)
            frontres_r_geom_mean = (_frontres_r_geom_sum / _frontres_reward_diag_steps
                                    if _is_frontres and _frontres_reward_diag_steps > 0 else None)
            frontres_intervention_cost_mean = (
                _frontres_intervention_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_clean_bound_cost_mean = (
                _frontres_clean_bound_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_clean_bound_side_cost_mean = (
                _frontres_clean_bound_side_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_over_cost_mean = (
                _frontres_over_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_under_repair_cost_mean = (
                _frontres_under_repair_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_reward_frontres_mean = (
                _frontres_reward_frontres_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_reward_clean_mean = (
                _frontres_reward_clean_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_reward_candidate_mean = (
                _frontres_reward_candidate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_reward_oracle_mean = (
                _frontres_reward_oracle_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_exec_planar_mean = (
                _frontres_exec_planar_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_exec_vertical_mean = (
                _frontres_exec_vertical_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_exec_task_mean = (
                _frontres_exec_task_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_damage_gap_mean = (
                _frontres_damage_gap_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_oracle_clean_gap_mean = (
                _frontres_oracle_clean_gap_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_oracle_trust_mean = (
                _frontres_oracle_trust_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_repair_gain_mean = (
                _frontres_repair_gain_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_candidate_gain_mean = (
                _frontres_candidate_gain_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_projection_gain_mean = (
                _frontres_projection_gain_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_underwrite_mean = (
                _frontres_underwrite_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_accept_pref_mask_mean = (
                _frontres_accept_pref_mask_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_accept_pref_full_mean = (
                _frontres_accept_pref_full_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_accept_pref_noop_mean = (
                _frontres_accept_pref_noop_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_accept_pref_keep_mean = (
                _frontres_accept_pref_keep_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_accept_pref_ignore_mean = (
                _frontres_accept_pref_ignore_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_accept_pref_margin_mean = (
                _frontres_accept_pref_margin_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_accept_pref_need_mean = (
                _frontres_accept_pref_need_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_accept_pref_admiss_mean = (
                _frontres_accept_pref_admiss_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_accept_pref_target_mean = (
                _frontres_accept_pref_target_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_inertial_pref_penalty_rho_mean = (
                _frontres_inertial_pref_penalty_rho_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_inertial_pref_penalty_one_mean = (
                _frontres_inertial_pref_penalty_one_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_positive_gain_frac_mean = (
                _frontres_positive_gain_frac_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_repair_ratio_mean = (
                _frontres_repair_ratio_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_exec_signal_mean = (
                _frontres_exec_signal_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_weighted_exec_signal_mean = (
                _frontres_weighted_exec_signal_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_train_reward_mean = (
                _frontres_train_reward_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_effective_gain_bonus_mean = (
                _frontres_effective_gain_bonus_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_safe_cost_mean = (
                _frontres_safe_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_repair_cost_mean = (
                _frontres_repair_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_broken_cost_mean = (
                _frontres_broken_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_reward_progress_mean = (
                _frontres_reward_progress_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_constraint_progress_mean = (
                _frontres_constraint_progress_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_behavior_fit_mean = (
                _frontres_behavior_fit_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_repair_fit_rate_mean = (
                _frontres_repair_fit_rate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_repair_fit_gain_mean = (
                _frontres_repair_fit_gain_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_restore_ratio_rp_mean = (
                _frontres_restore_ratio_rp_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_residual_rp_abs_mean = (
                _frontres_residual_rp_abs_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_corr_roll_bias_mean = (
                _frontres_corr_roll_bias_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_corr_pitch_bias_mean = (
                _frontres_corr_pitch_bias_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_harm_rate_mean = (
                _frontres_harm_rate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_harm_mag_mean = (
                _frontres_harm_mag_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_safe_harm_rate_mean = (
                _frontres_safe_harm_rate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_broken_harm_rate_mean = (
                _frontres_broken_harm_rate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_safe_abstain_cost_mean = (
                _frontres_safe_abstain_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_broken_abstain_cost_mean = (
                _frontres_broken_abstain_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_window_mu_mean = (
                _frontres_window_mu_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_safe_frac_mean = (
                _frontres_safe_frac_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_repair_frac_mean = (
                _frontres_repair_frac_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_broken_frac_mean = (
                _frontres_broken_frac_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_candidate_floor_margin_mean = (
                _frontres_candidate_floor_margin_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_candidate_floor_pass_mean = (
                _frontres_candidate_floor_pass_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_stable_route_frac_mean = (
                _frontres_stable_route_frac_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_stable_endpoint_frac_mean = (
                _frontres_stable_endpoint_frac_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_tri_weight_repair_mean = (
                _frontres_tri_weight_repair_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_tri_weight_noisy_mean = (
                _frontres_tri_weight_noisy_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_tri_weight_stable_mean = (
                _frontres_tri_weight_stable_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_state_alpha_pred_mean = (
                _frontres_state_alpha_pred_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_state_alpha_target_mean = (
                _frontres_state_alpha_target_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_state_alpha_mask_mean = (
                _frontres_state_alpha_mask_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_state_alpha_route_mean = (
                _frontres_state_alpha_route_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_actor_gate_mean = (
                _frontres_actor_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_exec_gate_mean = (
                _frontres_exec_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_cost_gate_mean = (
                _frontres_cost_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_dr_z_abs_mean = (_frontres_dr_z_abs_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_dr_xy_abs_mean = (_frontres_dr_xy_abs_sum / _frontres_shaping_steps
                                       if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_dr_rp_abs_mean = (_frontres_dr_rp_abs_sum / _frontres_shaping_steps
                                       if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_dr_yaw_abs_mean = (_frontres_dr_yaw_abs_sum / _frontres_shaping_steps
                                        if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_corr_z_abs_mean = (_frontres_corr_z_abs_sum / _frontres_shaping_steps
                                        if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_corr_xy_abs_mean = (_frontres_corr_xy_abs_sum / _frontres_shaping_steps
                                         if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_corr_rp_abs_mean = (_frontres_corr_rp_abs_sum / _frontres_shaping_steps
                                         if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_corr_yaw_abs_mean = (_frontres_corr_yaw_abs_sum / _frontres_shaping_steps
                                          if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_smooth_penalty_mean = (_frontres_smooth_penalty_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_reg_penalty_mean    = (_frontres_reg_penalty_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_survival_rate       = (1.0 - _frontres_term_count / _frontres_step_count
                                            if _is_frontres and _frontres_step_count > 0 else None)
            frontres_delta_pos_abs_mean  = (_frontres_delta_pos_abs_sum / _frontres_shaping_steps
                                            if _is_frontres and _is_task_space_mode and _frontres_shaping_steps > 0 else None)
            frontres_delta_rpy_abs_mean  = (_frontres_delta_rpy_abs_sum / _frontres_shaping_steps
                                            if _is_frontres and _is_task_space_mode and _frontres_shaping_steps > 0 else None)
            frontres_delta_z_abs_mean    = (_frontres_delta_z_abs_sum / _frontres_shaping_steps
                                            if _is_frontres and not _is_task_space_mode and _frontres_shaping_steps > 0 else None)
            frontres_jump_degree_mean    = (_frontres_jump_degree_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)

            # Store r_delta mean for next iteration's PI controller update.
            if frontres_rdelta_mean is not None:
                self._last_r_delta_mean = frontres_rdelta_mean
            if (
                frontres_safe_frac_mean is not None
                and frontres_repair_frac_mean is not None
                and frontres_broken_frac_mean is not None
                and frontres_positive_gain_frac_mean is not None
            ):
                self._last_frontres_boundary_stats = {
                    "safe": float(frontres_safe_frac_mean),
                    "repair": float(frontres_repair_frac_mean),
                    "broken": float(frontres_broken_frac_mean),
                    "positive_gain": float(frontres_positive_gain_frac_mean),
                    "candidate_floor_pass": float(frontres_candidate_floor_pass_mean or 0.0),
                    "stable_route": float(frontres_stable_route_frac_mean or 0.0),
                    "stable_endpoint": float(frontres_stable_endpoint_frac_mean or 0.0),
                }

            if _is_frontres:
                if not _frontres_supervised_restore:
                    self._frontres_update_supervised_controller(
                        loss_dict=loss_dict,
                        positive_gain_frac=frontres_positive_gain_frac_mean,
                        harm_rate=frontres_harm_rate_mean,
                    )
                if hasattr(self.alg, "lambda_supervised"):
                    loss_dict["lambda_supervised"] = float(self.alg.lambda_supervised)

            # DR scale for logging: current value (set by PI controller at top of iteration)
            frontres_dr_scale = (
                float(getattr(self, "_frontres_effective_dr_scale", _dr_scale))
                if _is_frontres else None
            )
            frontres_perturb_modes = (
                ",".join(getattr(self, "_frontres_curriculum_active_modes", ()))
                if _is_frontres else None
            )
            frontres_perturb_complexity = (
                getattr(self, "_frontres_curriculum_complexity", None)
                if _is_frontres else None
            )

            # Removed: staircase advancement logic (replaced by PI controller above)
            _staircase_level_for_log = None
            _staircase_mult_for_log  = None

            # Phase flags for diagnostics (exposed to log() via locals())
            _supervised_warmup_active = False  # runs before main loop, always False here
            _critic_warmup_active     = _critic_warmup

            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())

                # Save model
                if it % self.save_interval == 0:
                    _checkpoint_path = os.path.join(self.log_dir, f"model_{it}.pt")
                    self.save(_checkpoint_path)
                    self._record_frontres_checkpoint_probe(locals(), _checkpoint_path)

            # Clear episode infos
            ep_infos.clear() # 清空记录机器人人得分和存活长度的字典

            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files 防呆设计, 自动扫描本地改动并保存在log_dir
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)

                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            _final_checkpoint_path = os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
            self.save(_final_checkpoint_path)
            self._record_frontres_checkpoint_probe(locals(), _final_checkpoint_path)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size

        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)

                if self.training_type == "supervise":
                    # Stage 1: only log termination-related keys under GMT/ prefix.
                    # Reward and other RL metrics are meaningless here.
                    key_lower = key.lower().replace("/", "_")
                    if any(r in key_lower for r in ["rew", "reward"]):
                        continue  # skip reward metrics entirely
                    # Everything else (e.g. termination reasons) → GMT/ namespace
                    log_key = key if "/" in key else f"GMT/{key}"
                    self.writer.add_scalar(log_key, value, locs["it"])
                    ep_string += f"""{f'GMT {key}:':>{pad}} {value:.4f}\n"""
                else:
                    # log to logger and terminal
                    if "/" in key:
                        self.writer.add_scalar(key, value, locs["it"])
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/" + key, value, locs["it"])
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # -- Stage 1: GMT episode-length as action-completion proxy
        if self.training_type == "supervise" and len(locs["lenbuffer"]) > 0:
            gmt_ep_len = statistics.mean(locs["lenbuffer"])
            self.writer.add_scalar("GMT/mean_episode_length", gmt_ep_len, locs["it"])

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        # Keys suppressed when their controlling lambda is 0 (avoids clutter for unused modes).
        _suppress_if_zero = {"bc_off_policy", "bc_teacher", "lambda_off_policy", "lambda_teacher"}
        # Keys reclassified out of Loss/ for clarity.
        _to_frontres  = {
            "supervised_cos_sim",
            "supervised_mae",
            "supervised_rmse",
            "supervised_rpy_mae",
            "supervised_rpy_rmse",
            "supervised_restore_ratio",
            "supervised_valid_frac",
            "supervised_l_pos",
            "supervised_l_rot",
            "supervised_l_mag",
            "supervised_l_over",
            "supervised_l_smooth",
            "supervised_l_sparse",
            "supervised_l_miss",
            "supervised_l_coeff_smooth",
            "supervised_l_harm",
            "frontres_alpha_mean",
            "frontres_alpha_active_frac",
            "frontres_tau_mean",
            "frontres_tau_active_frac",
            "frontres_rho_pos_mean",
            "frontres_rho_pos_active_frac",
            "frontres_accept_pos_mean",
            "frontres_accept_rpy_mean",
            "frontres_accept_active_frac",
            "frontres_write_ratio",
            "frontres_proposal_ratio",
            "frontres_axis_leakage",
            "state_alpha_loss",
            "lambda_state_alpha",
            "state_alpha_mask_frac",
            "state_alpha_target_mean",
            "state_alpha_pred_mean",
            "state_alpha_abs_err",
            "state_alpha_acc",
        }      # diagnostics, not losses
        _to_curriculum = {"lambda_supervised", "ppo_actor_weight"}  # scheduler state, not a loss
        _frontres_diag_keys = {
            "delta_q_norm_mean",
            "delta_q_norm_std",
            "smooth_metric",
            "lambda_reg",
            "grad_cos_ppo_supervised",
            "grad_norm_ratio_ppo_to_supervised",
        }
        _stage1_dz_keys     = {"loss_dz", "dz_pred_abs", "dz_gt_abs", "dz_mae"}
        for key, value in locs["loss_dict"].items():
            if key in _suppress_if_zero and value == 0.0:
                continue
            if self.training_type == "supervise" and key in _stage1_dz_keys:
                self.writer.add_scalar(f"Stage1/DeltaZ/{key}", value, locs["it"])
            elif isinstance(self.alg.policy, FrontRESActorCritic) and key in _frontres_diag_keys:
                self.writer.add_scalar(f"FrontRES/{key}", value, locs["it"])
            elif key in _to_frontres:
                self.writer.add_scalar(f"FrontRES/{key}", value, locs["it"])
            elif key in _to_curriculum:
                self.writer.add_scalar(f"Curriculum/{key}", value, locs["it"])
            else:
                self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy (not meaningful during supervised learning)
        if self.training_type != "supervise":
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- FrontRES B1 delta-reward + curriculum diagnostics
        if isinstance(self.alg.policy, FrontRESActorCritic):
            _ts_mode = getattr(self.alg.policy, 'num_task_corrections', 0) > 0

            # B1 delta-reward
            if locs.get("frontres_rdelta_mean") is not None:
                self.writer.add_scalar("FrontRES/r_delta_per_step",
                                       locs["frontres_rdelta_mean"], locs["it"])
            if locs.get("frontres_baseline_mean") is not None:
                self.writer.add_scalar("FrontRES/baseline_per_step",
                                       locs["frontres_baseline_mean"], locs["it"])
            for _name in (
                "r_exec", "r_geom", "r_rescue", "intervention_cost",
                "clean_bound_cost", "clean_bound_side_cost", "over_cost", "under_repair_cost",
                "reward_frontres", "reward_clean", "reward_candidate", "reward_oracle",
                "exec_planar", "exec_vertical", "exec_task",
                "damage_gap", "oracle_clean_gap", "oracle_trust",
                "repair_gain", "candidate_gain", "projection_gain", "underwrite",
                "accept_pref_mask", "accept_pref_full", "accept_pref_noop",
                "accept_pref_keep", "accept_pref_ignore", "accept_pref_margin",
                "positive_gain_frac", "repair_ratio",
                "exec_signal", "weighted_exec_signal", "train_reward",
                "effective_gain_bonus", "safe_cost", "repair_cost", "broken_cost",
                "reward_progress", "constraint_progress",
                "behavior_fit", "repair_fit_rate", "repair_fit_gain",
                "restore_ratio_rp", "residual_rp_abs", "corr_roll_bias", "corr_pitch_bias",
                "harm_rate", "harm_mag", "safe_harm_rate", "broken_harm_rate",
                "safe_abstain_cost", "broken_abstain_cost",
                "window_mu", "safe_frac", "repair_frac", "broken_frac",
                "candidate_floor_margin", "candidate_floor_pass", "stable_route_frac",
                "stable_endpoint_frac", "tri_weight_repair", "tri_weight_noisy", "tri_weight_stable",
                "actor_gate", "exec_gate", "cost_gate",
                "r_z", "r_xy", "r_rp", "r_yaw",
                "dr_z_abs", "dr_xy_abs", "dr_rp_abs", "dr_yaw_abs",
                "corr_z_abs", "corr_xy_abs", "corr_rp_abs", "corr_yaw_abs",
            ):
                _value = locs.get(f"frontres_{_name}_mean")
                if _value is not None:
                    self.writer.add_scalar(f"FrontRES/RewardComponents/{_name}",
                                           _value, locs["it"])
            _complexity = locs.get("frontres_perturb_complexity")
            if _complexity is not None:
                _complexity_id = {"single": 1.0, "two": 2.0, "three": 3.0, "full": 4.0}.get(_complexity)
                if _complexity_id is not None:
                    self.writer.add_scalar("FrontRES/PerturbationCurriculum/complexity",
                                           _complexity_id, locs["it"])

            # Correction magnitude (task-space: split Δpos/Δrpy; joint-space: Δz)
            if _ts_mode:
                if locs.get("frontres_delta_pos_abs_mean") is not None:
                    self.writer.add_scalar("FrontRES/delta_pos_abs_mean",
                                           locs["frontres_delta_pos_abs_mean"], locs["it"])
                if locs.get("frontres_delta_rpy_abs_mean") is not None:
                    self.writer.add_scalar("FrontRES/delta_rpy_abs_mean",
                                           locs["frontres_delta_rpy_abs_mean"], locs["it"])
            else:
                if locs.get("frontres_delta_z_abs_mean") is not None:
                    self.writer.add_scalar("FrontRES/delta_z_abs_mean",
                                           locs["frontres_delta_z_abs_mean"], locs["it"])

            # Optional shaping penalties (only log when non-zero)
            if locs.get("frontres_smooth_penalty_mean") not in (None, 0.0):
                self.writer.add_scalar("FrontRES/smooth_penalty_per_step",
                                       locs["frontres_smooth_penalty_mean"], locs["it"])
            if locs.get("frontres_reg_penalty_mean") not in (None, 0.0):
                self.writer.add_scalar("FrontRES/reg_penalty_per_step",
                                       locs["frontres_reg_penalty_mean"], locs["it"])

            # Jump-degree gate
            if locs.get("frontres_jump_degree_mean") is not None:
                self.writer.add_scalar("FrontRES/jump_degree_mean",
                                       locs["frontres_jump_degree_mean"], locs["it"])

            # Curriculum / DR schedule
            if locs.get("frontres_dr_scale") is not None:
                self.writer.add_scalar("Curriculum/dr_scale",
                                       locs["frontres_dr_scale"], locs["it"])
                _frontier_scale = getattr(self, "_frontres_dr_frontier_scale", None)
                if _frontier_scale is not None:
                    self.writer.add_scalar("Curriculum/dr_frontier_scale",
                                           float(_frontier_scale), locs["it"])
                _gmt_safe = getattr(self, "_frontres_gmt_frontier_safe_low", None)
                _gmt_broken = getattr(self, "_frontres_gmt_frontier_broken_high", None)
                _gmt_score = getattr(self, "_frontres_gmt_frontier_probe_score", None)
                _gmt_probe = getattr(self, "_frontres_gmt_frontier_probe_scale", None)
                _gmt_confirmed = getattr(self, "_frontres_gmt_frontier_confirmed", None)
                if _gmt_safe is not None:
                    self.writer.add_scalar("Curriculum/gmt_frontier_safe_low",
                                           float(_gmt_safe), locs["it"])
                if _gmt_broken is not None:
                    self.writer.add_scalar("Curriculum/gmt_frontier_broken_high",
                                           float(_gmt_broken), locs["it"])
                if _gmt_score is not None:
                    self.writer.add_scalar("Curriculum/gmt_frontier_probe_score",
                                           float(_gmt_score), locs["it"])
                if _gmt_probe is not None:
                    self.writer.add_scalar("Curriculum/gmt_frontier_next_probe",
                                           float(_gmt_probe), locs["it"])
                if _gmt_confirmed is not None:
                    self.writer.add_scalar("Curriculum/gmt_frontier_confirmed",
                                           float(_gmt_confirmed), locs["it"])
                _mix_easy = getattr(self, "_frontres_dr_mix_easy_frac", None)
                _mix_frontier = getattr(self, "_frontres_dr_mix_frontier_frac", None)
                _mix_hard = getattr(self, "_frontres_dr_mix_hard_frac", None)
                _mix_mean = getattr(self, "_frontres_dr_mix_mean_scale", None)
                if _mix_easy is not None:
                    self.writer.add_scalar("Curriculum/dr_mix_easy_frac", float(_mix_easy), locs["it"])
                if _mix_frontier is not None:
                    self.writer.add_scalar("Curriculum/dr_mix_frontier_frac", float(_mix_frontier), locs["it"])
                if _mix_hard is not None:
                    self.writer.add_scalar("Curriculum/dr_mix_hard_frac", float(_mix_hard), locs["it"])
                if _mix_mean is not None:
                    self.writer.add_scalar("Curriculum/dr_mix_mean_scale", float(_mix_mean), locs["it"])
            if locs.get("frontres_survival_rate") is not None:
                self.writer.add_scalar("Curriculum/training_survival_rate",
                                       locs["frontres_survival_rate"], locs["it"])
            self.writer.add_scalar("Curriculum/r_delta_ema",
                                   locs.get("_r_delta_ema", 0.0), locs["it"])

            # Δq alpha ramp (legacy joint-space mode only)
        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training (RL metrics: skip during supervised learning to avoid confusing oscillation)
        if self.training_type != "supervise" and len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])

            # everything else
            if isinstance(self.alg.policy, FrontRESActorCritic):
                # B1: rewbuffer = FrontRES r_delta per episode; rewbuffer_gmt = GMT raw reward
                self.writer.add_scalar("Train/mean_r_delta",   statistics.mean(locs["rewbuffer"]),     locs["it"])
                if len(locs.get("rewbuffer_gmt", [])) > 0:
                    self.writer.add_scalar("Train/mean_reward_gmt", statistics.mean(locs["rewbuffer_gmt"]), locs["it"])
                if len(locs.get("lenbuffer_gmt", [])) > 0:
                    self.writer.add_scalar("Train/mean_episode_length_gmt",
                                           statistics.mean(locs["lenbuffer_gmt"]), locs["it"])
            else:
                self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        iter_title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if self.training_type != "supervise" and len(locs["rewbuffer"]) > 0:
            # ── Phase indicator ──────────────────────────────────────────────
            _ts_mode = getattr(self.alg.policy, 'num_task_corrections', 0) > 0
            if _ts_mode:
                _is_warmup = locs.get("_supervised_warmup_active", False)
                _is_critic = locs.get("_critic_warmup_active", False)
                _is_actor_takeover = locs.get("_actor_takeover_active", False)
                _lam = getattr(self.alg, 'lambda_supervised', 0.0)
                _objective = getattr(self.alg, "frontres_training_objective", "")
                if f"{_objective}".lower() == "supervised_restore":
                    _phase = "SUPERVISED RESTORE"
                    _notes = "(PPO/HRL update disabled; fitting clean restoration target)"
                elif f"{_objective}".lower() == "basis_restore":
                    _phase = "BASIS RESTORE"
                    _notes = "(PPO/HRL update disabled; factorized repair coefficients)"
                elif _is_warmup:
                    _phase = "SUPERVISED WARMUP"
                    _notes = "(GMT-only, FrontRES corrections disabled)"
                elif _is_critic:
                    _phase = "CRITIC WARMUP"
                    _paw = locs.get("loss_dict", {}).get("ppo_actor_weight", None)
                    if _paw is not None and _paw <= 0.0:
                        _notes = "(fixed low DR, PPO actor frozen; critic + supervised train)"
                    else:
                        _notes = "(fixed low DR; critic + supervised train)"
                elif _is_actor_takeover:
                    _phase = "ACTOR TAKEOVER"
                    _notes = "(fixed DR; PPO actor weight ramping)"
                elif _lam > 0.5:
                    _phase = "PPO + SUPERVISED ANCHOR"
                    _notes = ""
                elif _lam > 0.15:
                    _phase = "PPO + WEAK SUPERVISION"
                    _notes = ""
                else:
                    _phase = "PPO FINE-TUNING"
                    _notes = ""
            else:
                _phase = "PPO"
                _notes = ""
            _phase_str = f"  PHASE: {_phase}  "
            if _notes:
                _phase_str += f"\n  {_notes}  "

            _is_frontres_policy = isinstance(self.alg.policy, FrontRESActorCritic)
            if _is_frontres_policy:
                log_string = (
                    f"""{'#' * width}\n"""
                    f"""{iter_title.center(width, ' ')}\n"""
                    f"""{_phase_str.center(width, ' ')}\n""")
            else:
                log_string = (
                    f"""{'#' * width}\n"""
                    f"""{iter_title.center(width, ' ')}\n"""
                    f"""{_phase_str.center(width, ' ')}\n\n"""
                    f"""{'─' * 30} PERFORMANCE {'─' * 30}\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                        'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")

            # ── FrontRES: r_delta + cos_sim + curriculum (compact) ──────────
            if _is_frontres_policy:
                log_string += f"""\n{'-' * 12} Performance {'-' * 12}\n"""
                log_string += f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                log_string += f"""{'ep_len_FrontRES:':>{pad}} {statistics.mean(locs['lenbuffer']):.1f}\n"""
                if len(locs.get("lenbuffer_gmt", [])) > 0:
                    log_string += f"""{'ep_len_GMT (baseline):':>{pad}} {statistics.mean(locs['lenbuffer_gmt']):.1f}\n"""
                if locs.get("frontres_perturb_complexity") is not None:
                    log_string += f"""{'perturb curriculum:':>{pad}} """
                    log_string += (
                        f"{locs['frontres_perturb_complexity']} "
                        f"[{locs.get('frontres_perturb_modes', '')}]\n"
                    )
                if locs.get("frontres_dr_scale") is not None:
                    log_string += f"""{'DR scale:':>{pad}} {locs['frontres_dr_scale']:.4f}\n"""
                    _frontier_scale = getattr(self, "_frontres_dr_frontier_scale", None)
                    if _frontier_scale is None:
                        _frontier_scale = locs["frontres_dr_scale"]
                    _mix_mode = getattr(self, "_frontres_dr_mix_mode", None)
                    if _mix_mode is None:
                        _mix_mode = "fixed"
                    log_string += f"""{'DR frontier/mix:':>{pad}} """
                    log_string += f"{_frontier_scale:.4f} / {_mix_mode}\n"
                    _mix_easy = getattr(self, "_frontres_dr_mix_easy_frac", None)
                    _mix_frontier = getattr(self, "_frontres_dr_mix_frontier_frac", None)
                    _mix_hard = getattr(self, "_frontres_dr_mix_hard_frac", None)
                    if _mix_easy is not None and _mix_frontier is not None and _mix_hard is not None:
                        log_string += f"""{'DR train mix e/f/h:':>{pad}} """
                        log_string += f"{float(_mix_easy):.3f} / {float(_mix_frontier):.3f} / {float(_mix_hard):.3f}\n"
                    _mix_mean = getattr(self, "_frontres_dr_mix_mean_scale", None)
                    if _mix_mean is not None:
                        log_string += f"""{'DR train scale mean:':>{pad}} {float(_mix_mean):.4f}\n"""
                    _gmt_score = getattr(self, "_frontres_gmt_frontier_probe_score", None)
                    _gmt_decision = getattr(self, "_frontres_gmt_frontier_decision", None)
                    _gmt_probe = getattr(self, "_frontres_gmt_frontier_probe_scale", None)
                    _gmt_safe = getattr(self, "_frontres_gmt_frontier_safe_low", None)
                    _gmt_broken = getattr(self, "_frontres_gmt_frontier_broken_high", None)
                    if _gmt_score is not None or _gmt_decision is not None:
                        log_string += f"""{'GMT frontier probe:':>{pad}} """
                        _score_s = "n/a" if _gmt_score is None else f"{float(_gmt_score):.3f}"
                        _probe_s = "n/a" if _gmt_probe is None else f"{float(_gmt_probe):.4f}"
                        _decision_s = _gmt_decision or "n/a"
                        _probe_src = getattr(self, "_frontres_gmt_frontier_probe_source", None)
                        _probe_n = getattr(self, "_frontres_gmt_frontier_probe_samples", None)
                        _src_s = "" if _probe_src is None else f", src={_probe_src}"
                        _n_s = "" if _probe_n is None else f", n={int(_probe_n)}"
                        log_string += f"next={_probe_s}, score={_score_s}, decision={_decision_s}{_src_s}{_n_s}\n"
                    if _gmt_safe is not None:
                        log_string += f"""{'GMT bracket safe/broken:':>{pad}} """
                        _broken_s = "open" if _gmt_broken is None else f"{float(_gmt_broken):.4f}"
                        log_string += f"{float(_gmt_safe):.4f} / {_broken_s}\n"
                if locs.get("frontres_survival_rate") is not None:
                    log_string += f"""{'survival rate:':>{pad}} {locs['frontres_survival_rate']:.3f}\n"""

                _frontres_supervised_log = (
                    f"{getattr(self.alg, 'frontres_training_objective', '')}".lower()
                    in ("supervised_restore", "basis_restore", "hsl_hybrid")
                )
                _loss_dict = locs.get("loss_dict", {})

                if _frontres_supervised_log:
                    log_string += f"""\n{'-' * 9} Supervised Restore {'-' * 9}\n"""
                    _cs = _loss_dict.get("supervised_cos_sim", None)
                    if _cs is not None:
                        log_string += f"""{'supervised_cos_sim:':>{pad}} {_cs:.4f}\n"""
                    _sup_restore = _loss_dict.get("supervised_restore_ratio", None)
                    if _sup_restore is not None:
                        log_string += f"""{'restore ratio:':>{pad}} {_sup_restore:+.3f}\n"""
                    if _loss_dict.get("supervised_mae", None) is not None:
                        log_string += f"""{'mae/rmse all:':>{pad}} """
                        log_string += (
                            f"{_loss_dict.get('supervised_mae', 0.0):.5f} / "
                            f"{_loss_dict.get('supervised_rmse', 0.0):.5f}\n"
                        )
                    if _loss_dict.get("supervised_rpy_mae", None) is not None:
                        log_string += f"""{'mae/rmse rpy:':>{pad}} """
                        log_string += (
                            f"{_loss_dict.get('supervised_rpy_mae', 0.0):.5f} / "
                            f"{_loss_dict.get('supervised_rpy_rmse', 0.0):.5f}\n"
                        )
                    if _loss_dict.get("supervised_valid_frac", None) is not None:
                        log_string += f"""{'valid target frac:':>{pad}} {_loss_dict.get('supervised_valid_frac', 0.0):.3f}\n"""
                    if _loss_dict.get("supervised_l_pos", None) is not None:
                        log_string += f"""{'L_pos/L_rot:':>{pad}} """
                        log_string += (
                            f"{_loss_dict.get('supervised_l_pos', 0.0):.6f} / "
                            f"{_loss_dict.get('supervised_l_rot', 0.0):.6f}\n"
                        )
                        log_string += f"""{'L_mag/over/smooth:':>{pad}} """
                        log_string += (
                            f"{_loss_dict.get('supervised_l_mag', 0.0):.6f} / "
                            f"{_loss_dict.get('supervised_l_over', 0.0):.6f} / "
                            f"{_loss_dict.get('supervised_l_smooth', 0.0):.6f}\n"
                        )
                        log_string += f"""{'L_harm/conf:':>{pad}} """
                        log_string += (
                            f"{_loss_dict.get('supervised_l_harm', 0.0):.6f} / "
                            f"{_loss_dict.get('supervised_l_conf', 0.0):.6f}\n"
                        )
                    if f"{getattr(self.alg, 'frontres_training_objective', '')}".lower() in ("basis_restore", "hsl_hybrid"):
                        if int(getattr(getattr(self.alg, "policy", None), "task_conf_dim", 2)) == 1:
                            log_string += f"""{'pos rejoin/active:':>{pad}} """
                            log_string += (
                                f"{_loss_dict.get('frontres_rho_pos_mean', _loss_dict.get('frontres_tau_mean', 0.0)):.3f} / "
                                f"{_loss_dict.get('frontres_rho_pos_active_frac', _loss_dict.get('frontres_tau_active_frac', 0.0)):.3f}\n"
                            )
                        elif int(getattr(getattr(self.alg, "policy", None), "task_conf_dim", 2)) == 6:
                            log_string += f"""{'accept pos/rpy:':>{pad}} """
                            log_string += (
                                f"{_loss_dict.get('frontres_accept_pos_mean', 0.0):.3f} / "
                                f"{_loss_dict.get('frontres_accept_rpy_mean', 0.0):.3f}\n"
                            )
                        else:
                            log_string += f"""{'alpha mean/active:':>{pad}} """
                            log_string += (
                                f"{_loss_dict.get('frontres_alpha_mean', 0.0):.3f} / "
                                f"{_loss_dict.get('frontres_alpha_active_frac', 0.0):.3f}\n"
                            )
                        ratio_label = "proposal ratio/leakage"
                        ratio_value = _loss_dict.get(
                            "frontres_proposal_ratio",
                            _loss_dict.get("frontres_write_ratio", 0.0),
                        )
                        log_string += f"""{ratio_label + ':':>{pad}} """
                        log_string += (
                            f"{ratio_value:.3f} / "
                            f"{_loss_dict.get('frontres_axis_leakage', 0.0):.3f}\n"
                        )
                        log_string += f"""{'L_sparse/miss/csmooth:':>{pad}} """
                        log_string += (
                            f"{_loss_dict.get('supervised_l_sparse', 0.0):.6f} / "
                            f"{_loss_dict.get('supervised_l_miss', 0.0):.6f} / "
                            f"{_loss_dict.get('supervised_l_coeff_smooth', 0.0):.6f}\n"
                        )
                        if locs.get("frontres_candidate_gain_mean") is not None:
                            log_string += f"""{"gain proj/cand/bound:":>{pad}} """
                            log_string += (
                                f"{locs['frontres_repair_gain_mean']:+.4f} / "
                                f"{locs['frontres_candidate_gain_mean']:+.4f} / "
                                f"{locs['frontres_projection_gain_mean']:+.4f} "
                                f"(under={locs['frontres_underwrite_mean']:+.4f})\n"
                            )
                        if locs.get("frontres_candidate_floor_margin_mean") is not None:
                            log_string += f"""{'cand floor pass/margin:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_candidate_floor_pass_mean']:.3f} / "
                                f"{locs['frontres_candidate_floor_margin_mean']:+.4f}\n"
                            )
                        if locs.get("frontres_state_alpha_pred_mean") is not None:
                            log_string += f"""{'state alpha p/t/m/r:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_state_alpha_pred_mean']:.3f} / "
                                f"{locs['frontres_state_alpha_target_mean']:.3f} / "
                                f"{locs['frontres_state_alpha_mask_mean']:.3f} / "
                                f"{locs['frontres_state_alpha_route_mean']:.3f}\n"
                            )
                            if _loss_dict.get("state_alpha_loss", None) is not None:
                                log_string += f"""{'state alpha loss/acc:':>{pad}} """
                                log_string += (
                                    f"{_loss_dict.get('state_alpha_loss', 0.0):.4f} / "
                                    f"{_loss_dict.get('state_alpha_acc', 0.0):.3f}\n"
                                )
                        if locs.get("frontres_candidate_floor_margin_mean") is not None:
                            log_string += f"""{"rho space:":>{pad}} """
                            log_string += f"{self.cfg.get('frontres_rho_space', 'noisy_to_repair')}\n"
                            log_string += f"""{"stable endpoint frac:":>{pad}} """
                            log_string += f"{locs.get('frontres_stable_endpoint_frac_mean', 0.0):.3f}\n"
                            log_string += f"""{'stable route frac:':>{pad}} """
                            log_string += f"{locs.get('frontres_stable_route_frac_mean', 0.0):.3f}\n"
                            if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower() in (
                                "tri_anchor", "tri-anchor", "tri"
                            ):
                                log_string += f"""{'tri weights R/N/S:':>{pad}} """
                                log_string += (
                                    f"{locs.get('frontres_tri_weight_repair_mean', 0.0):.3f} / "
                                    f"{locs.get('frontres_tri_weight_noisy_mean', 0.0):.3f} / "
                                    f"{locs.get('frontres_tri_weight_stable_mean', 0.0):.3f}\n"
                                )
                        if locs.get("frontres_accept_pref_mask_mean") is not None:
                            _rho_space_diag = str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                            if _rho_space_diag in ("stable_to_repair", "stable-repair", "stable"):
                                _pref_label = "accept pref repair/stable/keep/ign:"
                            elif _rho_space_diag in ("tri_anchor", "tri-anchor", "tri"):
                                _pref_label = "accept pref repair/fallback/keep/ign:"
                            else:
                                _pref_label = "accept pref full/noop/keep/ign:"
                            log_string += f"""{_pref_label:>{pad}} """
                            log_string += (
                                f"{locs['frontres_accept_pref_full_mean']:.3f} / "
                                f"{locs['frontres_accept_pref_noop_mean']:.3f} / "
                                f"{locs['frontres_accept_pref_keep_mean']:.3f} / "
                                f"{locs['frontres_accept_pref_ignore_mean']:.3f} "
                                f"(mask={locs['frontres_accept_pref_mask_mean']:.3f}, "
                                f"margin={locs['frontres_accept_pref_margin_mean']:+.4f})\n"
                            )
                        if (
                            bool(self.cfg.get("frontres_acceptance_direct_target_enabled", False))
                            and locs.get("frontres_accept_pref_need_mean") is not None
                        ):
                            _accept_target_diag = locs.get(
                                "frontres_accept_pref_target_mean",
                                _loss_dict.get("acceptance_preference_target_mean", 0.0),
                            )
                            log_string += f"""{'accept need/admiss/tgt:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_accept_pref_need_mean']:.3f} / "
                                f"{locs['frontres_accept_pref_admiss_mean']:.3f} / "
                                f"{_accept_target_diag:.3f}\n"
                            )
                        if (
                            bool(self.cfg.get("frontres_inertial_preference_enabled", False))
                            and locs.get("frontres_inertial_pref_penalty_rho_mean") is not None
                        ):
                            log_string += f"""{"inert pref pen rho/cand:":>{pad}} """
                            log_string += (
                                f"{locs['frontres_inertial_pref_penalty_rho_mean']:.3f} / "
                                f"{locs['frontres_inertial_pref_penalty_one_mean']:.3f}\n"
                            )

                    log_string += f"""\n{'-' * 10} Correction Geometry {'-' * 10}\n"""
                    if locs.get("frontres_delta_pos_abs_mean") is not None:
                        log_string += f"""{'|Δpos|:':>{pad}} {locs['frontres_delta_pos_abs_mean']:.4f} m\n"""
                    if locs.get("frontres_delta_rpy_abs_mean") is not None:
                        log_string += f"""{'|Δrpy|:':>{pad}} {locs['frontres_delta_rpy_abs_mean']:.4f} rad\n"""
                    if locs.get("frontres_restore_ratio_rp_mean") is not None:
                        log_string += f"""{'restore rp/res/bias:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_restore_ratio_rp_mean']:+.3f} / "
                            f"{locs['frontres_residual_rp_abs_mean']:.4f} / "
                            f"{locs['frontres_corr_roll_bias_mean']:+.4f}, "
                            f"{locs['frontres_corr_pitch_bias_mean']:+.4f}\n"
                        )

                    log_string += f"""\n{'-' * 10} Optimization / Update {'-' * 10}\n"""
                    _lam = _loss_dict.get("lambda_supervised", None)
                    if _lam is not None:
                        log_string += f"""{'λ_supervised:':>{pad}} {_lam:.3f}\n"""
                    _paw = _loss_dict.get("ppo_actor_weight", None)
                    if _paw is not None:
                        log_string += f"""{'PPO actor weight:':>{pad}} {_paw:.3f}\n"""
                    _apl = _loss_dict.get("acceptance_preference_loss", None)
                    if _apl is not None:
                        log_string += f"""{'accept pref loss:':>{pad}} {_apl:.4f} """
                        _low_target_label = (
                            "stable"
                            if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                            in ("stable_to_repair", "stable-repair", "stable")
                            else (
                                "noisy"
                                if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                                in ("tri_anchor", "tri-anchor", "tri")
                                else "noop"
                            )
                        )
                        log_string += (
                            f"(λ={_loss_dict.get('lambda_acceptance_preference', 0.0):.3f}, "
                            f"mask={_loss_dict.get('acceptance_preference_mask_frac', 0.0):.3f}, "
                            f"full={_loss_dict.get('acceptance_preference_full_frac', 0.0):.3f}, "
                            f"{_low_target_label}={_loss_dict.get('acceptance_preference_noop_frac', 0.0):.3f}, "
                            f"eff_full={_loss_dict.get('acceptance_preference_effective_full_frac', 0.0):.3f}, "
                            f"fw={_loss_dict.get('acceptance_preference_full_weight', 1.0):.2f}, "
                            f"low_w={_loss_dict.get('acceptance_preference_noop_weight', 1.0):.2f}, "
                            f"γ={_loss_dict.get('acceptance_preference_focal_gamma', 0.0):.1f}, "
                            f"rho={_loss_dict.get('acceptance_preference_rho_mean', 0.0):.3f}, "
                            f"err={_loss_dict.get('acceptance_preference_abs_err', 0.0):.3f}, "
                            f"corr={_loss_dict.get('acceptance_preference_corr', 0.0):+.3f})\n"
                        )
                    _sal = _loss_dict.get("state_alpha_loss", None)
                    if _sal is not None:
                        log_string += f"""{'state alpha loss:':>{pad}} {_sal:.4f} """
                        log_string += (
                            f"(λ={_loss_dict.get('lambda_state_alpha', 0.0):.3f}, "
                            f"mask={_loss_dict.get('state_alpha_mask_frac', 0.0):.3f}, "
                            f"tgt={_loss_dict.get('state_alpha_target_mean', 0.0):.3f}, "
                            f"pred={_loss_dict.get('state_alpha_pred_mean', 0.0):.3f}, "
                            f"acc={_loss_dict.get('state_alpha_acc', 0.0):.3f})\n"
                        )
                    log_string += f"""{'learning rate:':>{pad}} {getattr(self.alg, 'learning_rate', 0.0):.2e}\n"""
                    _objective_name = f"{getattr(self.alg, 'frontres_training_objective', '')}".lower()
                    if _objective_name == "hsl_hybrid":
                        _objective_desc = "HSL ΔSE proposal + PPO 6D acceptance"
                    elif _objective_name == "basis_restore":
                        _objective_desc = "basis supervised only"
                    else:
                        _objective_desc = "supervised only"
                    log_string += f"""{'objective:':>{pad}} {_objective_desc}\n"""

                else:
                    log_string += f"""\n{'-' * 12} Main Reward {'-' * 12}\n"""
                    log_string += f"""{'r_delta (FrontRES):':>{pad}} {statistics.mean(locs['rewbuffer']):.4f}\n"""
                    if len(locs.get("rewbuffer_gmt", [])) > 0:
                        log_string += f"""{'reward_GMT (baseline):':>{pad}} {statistics.mean(locs['rewbuffer_gmt']):.4f}\n"""
                    _cs = _loss_dict.get("supervised_cos_sim", None)
                    if _cs is not None:
                        log_string += f"""{'supervised_cos_sim:':>{pad}} {_cs:.4f}\n"""
                    if locs.get("frontres_delta_pos_abs_mean") is not None:
                        log_string += f"""{'|Δpos|:':>{pad}} {locs['frontres_delta_pos_abs_mean']:.4f} m\n"""
                    if locs.get("frontres_delta_rpy_abs_mean") is not None:
                        log_string += f"""{'|Δrpy|:':>{pad}} {locs['frontres_delta_rpy_abs_mean']:.4f} rad\n"""
                    if locs.get("frontres_r_exec_mean") is not None:
                        if locs.get("frontres_damage_gap_mean") is not None:
                            log_string += f"""{'gap/gain/ratio:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_damage_gap_mean']:+.4f} / "
                                f"{locs['frontres_repair_gain_mean']:+.4f} / "
                                f"{locs['frontres_repair_ratio_mean']:+.4f}\n"
                            )
                            if locs.get("frontres_train_reward_mean") is not None:
                                log_string += f"""{'signal/w_signal/train_r:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_exec_signal_mean']:+.4f} / "
                                    f"{locs['frontres_weighted_exec_signal_mean']:+.4f} / "
                                    f"{locs['frontres_train_reward_mean']:+.4f}\n"
                                )
                                log_string += f"""{'Clean/Cand/Oracle/Trust:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_reward_clean_mean']:+.4f} / "
                                    f"{locs['frontres_reward_candidate_mean']:+.4f} / "
                                    f"{locs['frontres_reward_oracle_mean']:+.4f} / "
                                    f"{locs['frontres_oracle_trust_mean']:.3f}\n"
                                )
                                log_string += f"""{'oracle gap/cost:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_oracle_clean_gap_mean']:+.4f} / "
                                    f"{locs['frontres_clean_bound_cost_mean']:.4f}\n"
                                )
                                log_string += f"""{'side/over/under:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_clean_bound_side_cost_mean']:.4f} / "
                                    f"{locs['frontres_over_cost_mean']:.4f} / "
                                    f"{locs['frontres_under_repair_cost_mean']:.4f}\n"
                                )
                                log_string += f"""{'bonus/legacy S/R/B:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_effective_gain_bonus_mean']:+.4f} / "
                                    f"{locs['frontres_safe_cost_mean']:.4f} / "
                                    f"{locs['frontres_repair_cost_mean']:.4f} / "
                                    f"{locs['frontres_broken_cost_mean']:.4f}\n"
                                )
                                log_string += f"""{'reward/constraint prog:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_reward_progress_mean']:.4f} / "
                                    f"{locs['frontres_constraint_progress_mean']:.4f}\n"
                                )
                            if locs.get("frontres_behavior_fit_mean") is not None:
                                log_string += f"""{'exec legacy fit:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_behavior_fit_mean']:+.3f} / "
                                    f"{locs['frontres_repair_fit_rate_mean']:+.3f} / "
                                    f"{locs['frontres_repair_fit_gain_mean']:+.4f}\n"
                                )
                                log_string += f"""{'restore rp/res/bias:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_restore_ratio_rp_mean']:+.3f} / "
                                    f"{locs['frontres_residual_rp_abs_mean']:.4f} / "
                                    f"{locs['frontres_corr_roll_bias_mean']:+.4f}, "
                                    f"{locs['frontres_corr_pitch_bias_mean']:+.4f}\n"
                                )
                                log_string += f"""{'harm rate/mag:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_harm_rate_mean']:.3f} / "
                                    f"{locs['frontres_harm_mag_mean']:.4f}\n"
                                )
                                log_string += f"""{'safe/broken harm:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_safe_harm_rate_mean']:.3f} / "
                                    f"{locs['frontres_broken_harm_rate_mean']:.3f}\n"
                                )
                                log_string += f"""{'safe/broken abstain:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_safe_abstain_cost_mean']:.4f} / "
                                    f"{locs['frontres_broken_abstain_cost_mean']:.4f}\n"
                                )
                            if locs.get("frontres_positive_gain_frac_mean") is not None:
                                log_string += f"""{'positive_gain_frac:':>{pad}} """
                                log_string += f"{locs['frontres_positive_gain_frac_mean']:.3f}\n"
                            log_string += f"""{'safe/repair/broken frac:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_safe_frac_mean']:.3f} / "
                                f"{locs['frontres_repair_frac_mean']:.3f} / "
                                f"{locs['frontres_broken_frac_mean']:.3f}\n"
                            )
                            if locs.get("frontres_candidate_floor_margin_mean") is not None:
                                log_string += f"""{'cand floor pass/margin:':>{pad}} """
                                log_string += (
                                    f"{locs['frontres_candidate_floor_pass_mean']:.3f} / "
                                    f"{locs['frontres_candidate_floor_margin_mean']:+.4f}\n"
                                )
                                log_string += f"""{"rho space:":>{pad}} """
                                log_string += f"{self.cfg.get('frontres_rho_space', 'noisy_to_repair')}\n"
                                log_string += f"""{"stable endpoint frac:":>{pad}} """
                                log_string += f"{locs.get('frontres_stable_endpoint_frac_mean', 0.0):.3f}\n"
                                log_string += f"""{'stable route frac:':>{pad}} """
                                log_string += f"{locs.get('frontres_stable_route_frac_mean', 0.0):.3f}\n"
                                if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower() in (
                                    "tri_anchor", "tri-anchor", "tri"
                                ):
                                    log_string += f"""{'tri weights R/N/S:':>{pad}} """
                                    log_string += (
                                        f"{locs.get('frontres_tri_weight_repair_mean', 0.0):.3f} / "
                                        f"{locs.get('frontres_tri_weight_noisy_mean', 0.0):.3f} / "
                                        f"{locs.get('frontres_tri_weight_stable_mean', 0.0):.3f}\n"
                                    )

                        log_string += f"""\n{'-' * 12} Detail Reward {'-' * 12}\n"""
                        if locs.get("frontres_r_z_mean") is not None:
                            log_string += f"""{'r_z/r_xy/r_rp/r_yaw:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_r_z_mean']:+.4f} / "
                                f"{locs['frontres_r_xy_mean']:+.4f} / "
                                f"{locs['frontres_r_rp_mean']:+.4f} / "
                                f"{locs['frontres_r_yaw_mean']:+.4f}\n"
                            )
                        log_string += f"""{'repair/geom/rescue/action_cost:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_r_exec_mean']:+.4f} / "
                            f"{locs['frontres_r_geom_mean']:+.4f} / "
                            f"{locs['frontres_r_rescue_mean']:+.4f} / "
                            f"{locs['frontres_intervention_cost_mean']:+.4f}\n"
                        )
                        if locs.get("frontres_reward_frontres_mean") is not None and locs.get("frontres_baseline_mean") is not None:
                            log_string += f"""{'exec reward FR/cand/pert:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_reward_frontres_mean']:+.4f} / "
                                f"{locs.get('frontres_reward_candidate_mean', 0.0):+.4f} / "
                                f"{locs['frontres_baseline_mean']:+.4f}\n"
                            )
                        if locs.get("frontres_candidate_gain_mean") is not None:
                            log_string += f"""{'gain proj/cand/bound:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_repair_gain_mean']:+.4f} / "
                                f"{locs['frontres_candidate_gain_mean']:+.4f} / "
                                f"{locs['frontres_projection_gain_mean']:+.4f} "
                                f"(under={locs['frontres_underwrite_mean']:+.4f})\n"
                            )
                        if locs.get("frontres_accept_pref_mask_mean") is not None:
                            _rho_space_diag = str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                            if _rho_space_diag in ("stable_to_repair", "stable-repair", "stable"):
                                _pref_label = "accept pref repair/stable/keep/ign:"
                            elif _rho_space_diag in ("tri_anchor", "tri-anchor", "tri"):
                                _pref_label = "accept pref repair/fallback/keep/ign:"
                            else:
                                _pref_label = "accept pref full/noop/keep/ign:"
                            log_string += f"""{_pref_label:>{pad}} """
                            log_string += (
                                f"{locs['frontres_accept_pref_full_mean']:.3f} / "
                                f"{locs['frontres_accept_pref_noop_mean']:.3f} / "
                                f"{locs['frontres_accept_pref_keep_mean']:.3f} / "
                                f"{locs['frontres_accept_pref_ignore_mean']:.3f} "
                                f"(mask={locs['frontres_accept_pref_mask_mean']:.3f}, "
                                f"margin={locs['frontres_accept_pref_margin_mean']:+.4f})\n"
                            )
                        if (
                            bool(self.cfg.get("frontres_acceptance_direct_target_enabled", False))
                            and locs.get("frontres_accept_pref_need_mean") is not None
                        ):
                            _accept_target_diag = locs.get(
                                "frontres_accept_pref_target_mean",
                                _loss_dict.get("acceptance_preference_target_mean", 0.0),
                            )
                            log_string += f"""{'accept need/admiss/tgt:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_accept_pref_need_mean']:.3f} / "
                                f"{locs['frontres_accept_pref_admiss_mean']:.3f} / "
                                f"{_accept_target_diag:.3f}\n"
                            )
                        if (
                            bool(self.cfg.get("frontres_inertial_preference_enabled", False))
                            and locs.get("frontres_inertial_pref_penalty_rho_mean") is not None
                        ):
                            log_string += f"""{'inert pref pen rho/cand:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_inertial_pref_penalty_rho_mean']:.3f} / "
                                f"{locs['frontres_inertial_pref_penalty_one_mean']:.3f}\n"
                            )
                        if locs.get("frontres_exec_planar_mean") is not None:
                            log_string += f"""{'exec planar/vertical/task:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_exec_planar_mean']:+.4f} / "
                                f"{locs['frontres_exec_vertical_mean']:+.4f} / "
                                f"{locs['frontres_exec_task_mean']:+.4f}\n"
                            )
                        if locs.get("frontres_window_mu_mean") is not None:
                            log_string += f"""{'exec/cost gate:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_exec_gate_mean']:.3f} / "
                                f"{locs['frontres_cost_gate_mean']:.3f}\n"
                            )

                    log_string += f"""\n{'-' * 10} Optimization / Update {'-' * 10}\n"""
                    if locs.get("frontres_window_mu_mean") is not None:
                        log_string += f"""{'mu (reward window):':>{pad}} {locs['frontres_window_mu_mean']:.3f}\n"""
                        log_string += f"""{'actor sample weight:':>{pad}} {locs['frontres_actor_gate_mean']:.3f}\n"""
                    _gc = _loss_dict.get("grad_cos_ppo_supervised", None)
                    if _gc is not None:
                        _gr = _loss_dict.get("grad_norm_ratio_ppo_to_supervised", 0.0)
                        log_string += f"""{'grad cos PPO/Sup:':>{pad}} {_gc:+.4f} (norm ratio={_gr:.3f})\n"""
                    _rd_ema = locs.get("_r_delta_ema", 0.0)
                    log_string += f"""{'r_delta EMA:':>{pad}} {_rd_ema:.4f}\n"""
                    _lam = _loss_dict.get("lambda_supervised", None)
                    if _lam is not None:
                        log_string += f"""{'λ_supervised:':>{pad}} {_lam:.3f}\n"""
                    _paw = _loss_dict.get("ppo_actor_weight", None)
                    if _paw is not None:
                        log_string += f"""{'PPO actor weight:':>{pad}} {_paw:.3f}\n"""
                    _apl = _loss_dict.get("acceptance_preference_loss", None)
                    if _apl is not None:
                        log_string += f"""{'accept pref loss:':>{pad}} {_apl:.4f} """
                        _low_target_label = (
                            "stable"
                            if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                            in ("stable_to_repair", "stable-repair", "stable")
                            else (
                                "noisy"
                                if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                                in ("tri_anchor", "tri-anchor", "tri")
                                else "noop"
                            )
                        )
                        log_string += (
                            f"(λ={_loss_dict.get('lambda_acceptance_preference', 0.0):.3f}, "
                            f"mask={_loss_dict.get('acceptance_preference_mask_frac', 0.0):.3f}, "
                            f"full={_loss_dict.get('acceptance_preference_full_frac', 0.0):.3f}, "
                            f"{_low_target_label}={_loss_dict.get('acceptance_preference_noop_frac', 0.0):.3f}, "
                            f"eff_full={_loss_dict.get('acceptance_preference_effective_full_frac', 0.0):.3f}, "
                            f"fw={_loss_dict.get('acceptance_preference_full_weight', 1.0):.2f}, "
                            f"low_w={_loss_dict.get('acceptance_preference_noop_weight', 1.0):.2f}, "
                            f"γ={_loss_dict.get('acceptance_preference_focal_gamma', 0.0):.1f}, "
                            f"rho={_loss_dict.get('acceptance_preference_rho_mean', 0.0):.3f}, "
                            f"err={_loss_dict.get('acceptance_preference_abs_err', 0.0):.3f}, "
                            f"corr={_loss_dict.get('acceptance_preference_corr', 0.0):+.3f})\n"
                        )
                    if locs.get("frontres_window_mu_mean") is not None:
                        log_string += f"""{'actor/exec/cost:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_actor_gate_mean']:.3f} / "
                            f"{locs['frontres_exec_gate_mean']:.3f} / "
                            f"{locs['frontres_cost_gate_mean']:.3f}\n"
                        )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                    f"""{iter_title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n""")

            if self.training_type == "supervise":
                log_string += f"""{'─' * 30} STAGE 1 {'─' * 33}\n"""
                if "behavior" in locs["loss_dict"]:
                    log_string += f"""{'behavior loss:':>{pad}} {locs['loss_dict']['behavior']:.4f}\n"""
                if len(locs["lenbuffer"]) > 0:
                    log_string += f"""{'ep length:':>{pad}} {statistics.mean(locs['lenbuffer']):.1f}\n"""
            else:
                log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                for key, value in locs["loss_dict"].items():
                    log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        # Episode_Reward / Metrics / Terminations → wandb only, not console
        _footer_width = 44 if self.training_type == "frontres" else width
        log_string += (
            f"""{'-' * _footer_width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n""")
        
        print(log_string)

    def _record_frontres_checkpoint_probe(self, locs: dict, checkpoint_path: str) -> None:
        """Persist save-time FrontRES probe metrics and keep the best demo checkpoint.

        This is a lightweight checkpoint selector: it records the triplet
        rollout diagnostics already computed for the checkpoint iteration,
        without resetting the simulator or replaying the full training set.
        """
        if self.training_type != "frontres" or self.log_dir is None:
            return

        def _float(name: str, default: float | None = None) -> float | None:
            value = locs.get(name, default)
            if value is None:
                return default
            try:
                if isinstance(value, torch.Tensor):
                    value = value.detach().mean().item()
                return float(value)
            except (TypeError, ValueError):
                return default

        restore_ratio = _float("frontres_restore_ratio_rp_mean")
        if restore_ratio is None:
            return

        residual = _float("frontres_residual_rp_abs_mean", 0.0) or 0.0
        roll_bias = _float("frontres_corr_roll_bias_mean", 0.0) or 0.0
        pitch_bias = _float("frontres_corr_pitch_bias_mean", 0.0) or 0.0
        harm_rate = _float("frontres_harm_rate_mean", 0.0) or 0.0
        harm_mag = _float("frontres_harm_mag_mean", 0.0) or 0.0
        survival = _float("frontres_survival_rate", 1.0)
        r_delta = _float("frontres_rdelta_mean", 0.0) or 0.0
        dr_scale = _float("frontres_dr_scale", None)

        bias_abs = abs(roll_bias) + abs(pitch_bias)
        survival_penalty = 0.0 if survival is None else max(0.0, 1.0 - survival)
        score = (
            restore_ratio
            - 0.25 * harm_rate
            - 2.0 * harm_mag
            - 0.50 * bias_abs
            - 0.10 * residual
            - 2.0 * survival_penalty
        )

        record = {
            "iteration": int(locs.get("it", self.current_learning_iteration)),
            "checkpoint": os.path.basename(checkpoint_path),
            "score": score,
            "restore_ratio_rp": restore_ratio,
            "residual_rp_abs": residual,
            "corr_roll_bias": roll_bias,
            "corr_pitch_bias": pitch_bias,
            "bias_abs": bias_abs,
            "harm_rate": harm_rate,
            "harm_mag": harm_mag,
            "survival_rate": survival,
            "r_delta": r_delta,
            "dr_scale": dr_scale,
            "perturb_modes": locs.get("frontres_perturb_modes"),
            "perturb_complexity": locs.get("frontres_perturb_complexity"),
        }

        probe_path = os.path.join(self.log_dir, "frontres_checkpoint_probe.jsonl")
        with open(probe_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        if self.writer is not None and not self.disable_logs:
            self.writer.add_scalar("FrontRES/CheckpointProbe/demo_score", score, record["iteration"])
            self.writer.add_scalar("FrontRES/CheckpointProbe/restore_ratio_rp", restore_ratio, record["iteration"])
            self.writer.add_scalar("FrontRES/CheckpointProbe/bias_abs", bias_abs, record["iteration"])

        best_score = getattr(self, "_frontres_best_probe_score", None)
        best_meta_path = os.path.join(self.log_dir, "frontres_best_probe.json")
        if best_score is None and os.path.exists(best_meta_path):
            try:
                with open(best_meta_path, "r", encoding="utf-8") as f:
                    best_score = float(json.load(f).get("score"))
                    self._frontres_best_probe_score = best_score
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                best_score = None
        if best_score is None or score > float(best_score):
            self._frontres_best_probe_score = score
            best_path = os.path.join(self.log_dir, "model_best_probe.pt")
            shutil.copyfile(checkpoint_path, best_path)
            with open(best_meta_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, sort_keys=True)
            print(
                "[Runner] New FrontRES probe best: "
                f"score={score:+.4f}, restore_rp={restore_ratio:+.3f}, "
                f"harm={harm_rate:.3f}, bias={bias_abs:.4f} -> {os.path.basename(best_path)}",
                flush=True,
            )

    def save(self, path: str, infos=None):
        # Check if using ResidualActorCritic (special handling)
        if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
            # Save only residual network + critic (GMT is frozen, no need to save)
            model_state_dict = {
                'residual_actor': self.alg.policy.residual_actor.state_dict(),
                'critic': self.alg.policy.critic.state_dict(),}
            if getattr(self.alg.policy, "acceptance_actor", None) is not None:
                model_state_dict['acceptance_actor'] = self.alg.policy.acceptance_actor.state_dict()
            if getattr(self.alg.policy, "state_router_head", None) is not None:
                model_state_dict['state_router_head'] = self.alg.policy.state_router_head.state_dict()
            
            # Save noise std parameter
            if hasattr(self.alg.policy, 'std'):
                model_state_dict['std'] = self.alg.policy.std
            elif hasattr(self.alg.policy, 'log_std'):
                model_state_dict['log_std'] = self.alg.policy.log_std
        else:
            # Standard save: entire policy
            model_state_dict = self.alg.policy.state_dict()

        # -- Save model
        saved_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,}

        # Persist adaptive DR state so resume picks up at the correct scale.
        if hasattr(self, '_dr_scale'):
            saved_dict["dr_scale"] = self._dr_scale
        if hasattr(self, '_dr_prev_error'):
            saved_dict["dr_prev_error"] = self._dr_prev_error
        if getattr(self, '_frontres_boundary_ema', None) is not None:
            saved_dict["frontres_boundary_ema"] = dict(self._frontres_boundary_ema)
        if getattr(self, '_last_frontres_boundary_stats', None) is not None:
            saved_dict["last_frontres_boundary_stats"] = dict(self._last_frontres_boundary_stats)
        if hasattr(self, "_frontres_gmt_frontier_safe_low"):
            saved_dict["frontres_gmt_frontier_safe_low"] = self._frontres_gmt_frontier_safe_low
        if hasattr(self, "_frontres_gmt_frontier_broken_high"):
            saved_dict["frontres_gmt_frontier_broken_high"] = self._frontres_gmt_frontier_broken_high
        if hasattr(self, "_frontres_gmt_frontier_probe_scale"):
            saved_dict["frontres_gmt_frontier_probe_scale"] = self._frontres_gmt_frontier_probe_scale
        if hasattr(self, "_frontres_gmt_frontier_probe_score"):
            saved_dict["frontres_gmt_frontier_probe_score"] = self._frontres_gmt_frontier_probe_score
        if hasattr(self, "_frontres_gmt_frontier_decision"):
            saved_dict["frontres_gmt_frontier_decision"] = self._frontres_gmt_frontier_decision
        if hasattr(self, "_frontres_gmt_frontier_confirmed"):
            saved_dict["frontres_gmt_frontier_confirmed"] = self._frontres_gmt_frontier_confirmed
        if hasattr(self, '_frontres_warmup_complete'):
            saved_dict["frontres_warmup_complete"] = bool(self._frontres_warmup_complete)
        
        # -- Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
            # Save teacher normalizer for MOSAIC
            if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
                if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                    saved_dict["teacher_obs_norm_state_dict"] = self.teacher_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, load_critic: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        self._frontres_warmup_complete = bool(loaded_dict.get("frontres_warmup_complete", False))
        if self._frontres_warmup_complete:
            print("[Runner] Checkpoint marks FrontRES supervised warmup as complete.")

        # ── 断点续训模式控制 ────────────────────────────────────────────────────────
        # is_full_resume=True  (Stage2→Stage2 断点续训): 恢复优化器矩估计+学习率, 保留 std
        # is_full_resume=False (Stage1→Stage2 权重迁移): 仅权重, 重置优化器和 std.
        # Joint-warmup checkpoints are a special case: their critic has already
        # learned E(s)=R_feasible_oracle-R_noisy and should be transferred into RL.
        # load_optimizer 参数仍可从外部显式覆盖（例如强制跳过优化器加载）。
        is_full_resume: bool = self.cfg.get('is_full_resume', True)
        if not is_full_resume:
            load_optimizer = False   # 权重迁移模式：强制跳过优化器，从零初始化 Adam
            load_critic = self._frontres_warmup_complete
        print(f"[Runner] is_full_resume={is_full_resume} → "
              f"load_optimizer={load_optimizer}, load_critic={load_critic}, "
              f"reset_noise_std={not is_full_resume}")

        # Check if using ResidualActorCritic (special handling)
        if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
            # 智能映射：尝试从阶段一 (SuperviseLearning) 提取 student 权重
            if isinstance(self.alg.policy, FrontRESActorCritic) and "student.0.weight" in loaded_dict["model_state_dict"]:
                mapped_dict = {k.replace("student.", ""): v for k, v in loaded_dict["model_state_dict"].items() if k.startswith("student.")}
                try:
                    self.alg.policy.residual_actor.load_state_dict(mapped_dict, strict=True)
                    print("[Runner] Success: Auto-mapped Stage 1 'student' weights to Stage 2 'residual_actor'!")
                except RuntimeError:
                    migrated = False
                    if hasattr(self.alg.policy, "initialize_two_head_from_legacy_state"):
                        migrated = self.alg.policy.initialize_two_head_from_legacy_state(mapped_dict)
                    if migrated:
                        print("[Runner] Migrated Stage 1 'student' weights into FrontRES two-head actor.")
                    else:
                        raise
            else:
                residual_state = loaded_dict["model_state_dict"]["residual_actor"]
                try:
                    self.alg.policy.residual_actor.load_state_dict(residual_state, strict=True)
                except RuntimeError:
                    migrated = False
                    if hasattr(self.alg.policy, "initialize_two_head_from_legacy_state"):
                        migrated = self.alg.policy.initialize_two_head_from_legacy_state(residual_state)
                    if migrated:
                        print("[Runner] Migrated legacy residual_actor weights into FrontRES two-head actor.")
                    else:
                        raise
            if getattr(self.alg.policy, "acceptance_actor", None) is not None:
                if "acceptance_actor" in loaded_dict["model_state_dict"]:
                    self.alg.policy.acceptance_actor.load_state_dict(loaded_dict["model_state_dict"]["acceptance_actor"])
                    print("[Runner] Loaded split FrontRES acceptance_actor from checkpoint.")
                else:
                    migrated = False
                    if hasattr(self.alg.policy, "initialize_acceptance_from_residual_state"):
                        migrated = self.alg.policy.initialize_acceptance_from_residual_state(
                            loaded_dict["model_state_dict"].get("residual_actor", {})
                        )
                    if migrated:
                        print("[Runner] Initialized split acceptance_actor from legacy residual_actor rho rows.")
                    else:
                        print("[Runner] No split acceptance_actor weights found; initialized acceptance head from scratch.")
            elif "acceptance_actor" in loaded_dict["model_state_dict"]:
                print("[Runner] Ignoring split acceptance_actor weights because the active config uses the single two-head FrontRES actor.")
            if getattr(self.alg.policy, "state_router_head", None) is not None:
                if "state_router_head" in loaded_dict["model_state_dict"]:
                    self.alg.policy.state_router_head.load_state_dict(
                        loaded_dict["model_state_dict"]["state_router_head"],
                        strict=True,
                    )
                    print("[Runner] Loaded FrontRES state_router_head from checkpoint.")
                else:
                    print("[Runner] No state_router_head weights found; initialized alpha head from scratch.")

            if load_critic:
                if "critic" in loaded_dict["model_state_dict"]:
                    self.alg.policy.critic.load_state_dict(loaded_dict["model_state_dict"]["critic"])
                else:
                    print("[Runner] No critic weights found. Critic will be initialized from scratch.")
            # Load noise std parameter
            if "std" in loaded_dict["model_state_dict"]:
                self.alg.policy.std.data = loaded_dict["model_state_dict"]["std"].data
            elif "log_std" in loaded_dict["model_state_dict"]:
                self.alg.policy.log_std.data = loaded_dict["model_state_dict"]["log_std"].data
            if load_critic:
                print("[Runner] Loaded residual network + critic from checkpoint (GMT remains frozen)")
            else:
                print("[Runner] Loaded residual network only (skipping critic from checkpoint)")
            resumed_training = True
        else:
            if load_critic:
                # Standard load: entire policy
                resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
            else:
                actor_only_state_dict = {
                    key: value
                    for key, value in loaded_dict["model_state_dict"].items()
                    if not key.startswith("critic.")}
                
                resumed_training = self.alg.policy.load_state_dict(actor_only_state_dict, strict=False)

        # Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        # Load observation normalizers if used
        if self.empirical_normalization:
            if resumed_training:
                # Resuming training: load student obs normalizer
                # For ResidualActorCritic / FrontRESActorCritic, obs_normalizer IS GMT's frozen
                # normalizer — never overwrite it with a checkpoint's normalizer statistics.
                if not isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
                    self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                elif (isinstance(self.alg.policy, FrontRESActorCritic)
                        and self._frontres_gmt_obs_dim is not None
                        and "obs_norm_state_dict" in loaded_dict):
                    # Task-space FrontRES: anchor-error dims [:num_extra] are not
                    # covered by the GMT normalizer.  Restore Stage-1 empirical stats
                    # for those dims when the checkpoint actually contains them.
                    # so Stage 2 sees the same normalized scale that Stage 1 trained on.
                    _s1_sd   = loaded_dict["obs_norm_state_dict"]
                    _s1_mean = _s1_sd.get("_mean", None)  # shape (1, 800)
                    _s1_std  = _s1_sd.get("_std",  None)  # shape (1, 800)
                    if _s1_mean is not None and _s1_std is not None:
                        gmt_dim = self._frontres_gmt_obs_dim
                        obs_dim = int(getattr(self.alg.policy, "num_actor_obs", gmt_dim))
                        num_extra = max(0, obs_dim - gmt_dim)
                        if num_extra > 0 and _s1_mean.shape[-1] >= obs_dim and _s1_std.shape[-1] >= obs_dim:
                            self._frontres_extra_mean = _s1_mean[:, :num_extra].to(self.device)
                            self._frontres_extra_std  = _s1_std[:,  :num_extra].to(self.device)
                            print(f"[Runner] Loaded Stage-1 anchor-error normalizer stats "
                                  f"(dims 0–{num_extra}) for FrontRES task-space.")
                        else:
                            self._frontres_extra_mean = None
                            self._frontres_extra_std = None
                            print("[Runner] Stage-1 checkpoint has no compatible anchor-error "
                                  "normalizer stats; FrontRES extra dims pass through unnormalized.")

                if self.training_type == "mosaic":
                    # For MOSAIC: determine whether to load privileged_obs_normalizer from checkpoint
                    # Only skip loading if teacher_critic was loaded from a separate checkpoint AND is frozen
                    load_privileged_normalizer = load_critic
                    if hasattr(self.alg, 'teacher_critic_checkpoint_path') and self.alg.teacher_critic_checkpoint_path is not None:
                        if hasattr(self.alg, 'teacher_critic_frozen') and self.alg.teacher_critic_frozen:
                            load_privileged_normalizer = False
                            print("[Runner] Keeping privileged_obs_normalizer from teacher_critic_checkpoint (frozen).")

                    if load_privileged_normalizer:
                        # Load critic normalizer from student checkpoint
                        if "privileged_obs_norm_state_dict" in loaded_dict:
                            self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                            print("[Runner] Loaded privileged_obs_normalizer from checkpoint.")
                        else:
                            print("[Runner] WARNING: No privileged_obs_norm_state_dict in checkpoint!")

                    # Load teacher obs normalizer if available (for teacher BC)
                    if "teacher_obs_norm_state_dict" in loaded_dict:
                        self.teacher_obs_normalizer.load_state_dict(loaded_dict["teacher_obs_norm_state_dict"])
                        print("[Runner] Loaded teacher_obs_normalizer from checkpoint.")
                else:
                    # For PPO and Distillation: load both normalizers
                    if load_critic:
                        priv_sd = loaded_dict.get("privileged_obs_norm_state_dict", {})
                        if priv_sd and "_mean" in priv_sd:
                            self.privileged_obs_normalizer.load_state_dict(priv_sd)
                        else:
                            # Stage 1 (SuperviseLearning) checkpoint has no valid
                            # privileged_obs_norm_state_dict — critic normalizer starts fresh.
                            print("[Runner] WARNING: privileged_obs_norm_state_dict missing or invalid — "
                                  "privileged_obs_normalizer starts fresh (expected for Stage 1 → Stage 2 transfer).")
            else:
                # Not resuming (e.g., Distillation after RL): load teacher normalizer
                # For Distillation: the checkpoint's obs_norm is the teacher's normalizer
                if load_critic:
                    self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            if not load_critic:
                print("[Runner] Skipping optimizer load because load_critic=False.")
            else:
                try:
                    # -- algorithm optimizer
                    self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                    print("[Runner] Loaded optimizer state from checkpoint.")
                    # ── 学习率同步 ─────────────────────────────────────────────────────
                    # PPO.update() 每次 epoch 都用 self.alg.learning_rate 覆盖
                    # optimizer.param_groups["lr"]。load_state_dict 已将 param_groups["lr"]
                    # 恢复为 checkpoint 时的值，但 self.alg.learning_rate 仍是配置初始值。
                    # 此处同步，避免第一次 update() 将已恢复的学习率覆盖为初始值。
                    if is_full_resume and hasattr(self.alg, 'learning_rate'):
                        restored_lr = self.alg.optimizer.param_groups[0]['lr']
                        reset_lr = bool(self.cfg.get('reset_lr_on_resume', False))
                        if reset_lr:
                            # lr 被 adaptive schedule 压至下限时（如因 desired_kl 配置错误），
                            # 直接重置为算法配置的初始学习率，避免续训起点过低。
                            config_lr = float(self.alg_cfg.get('learning_rate', 5e-4))
                            self.alg.learning_rate = config_lr
                            for pg in self.alg.optimizer.param_groups:
                                pg['lr'] = config_lr
                            print(f"[Runner] Reset learning_rate → {config_lr:.2e} "
                                  f"(reset_lr_on_resume=True; checkpoint had {restored_lr:.2e})")
                        else:
                            self.alg.learning_rate = restored_lr
                            print(f"[Runner] Synced learning_rate = {restored_lr:.2e} (from optimizer checkpoint)")
                except (ValueError, KeyError) as e:
                    # Optimizer state mismatch (e.g., different parameter groups between stages)
                    # This can happen when:
                    # - Stage 1 had frozen critic (optimizer only has actor params)
                    # - Stage 2 unfreezes critic (optimizer has actor + critic params)
                    print(f"[Runner] WARNING: Could not load optimizer state: {e}")
                    print("[Runner] Optimizer will be initialized from scratch (learning rate, momentum, etc. reset)")
                    print("[Runner] This is expected when transitioning between training stages with different frozen parameters.")

                # -- RND optimizer if used
                if hasattr(self.alg, "rnd") and self.alg.rnd:
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            if is_full_resume:
                self.current_learning_iteration = loaded_dict["iter"]
            else:
                self.current_learning_iteration = 0
                print("[Runner] Stage1→Stage2 cold-start: current_learning_iteration reset to 0.")

        # ── 噪声 std 控制 ──────────────────────────────────────────────────────────
        # is_full_resume=True:  保留 checkpoint 中已自然适应的 std（断点续训）
        # is_full_resume=False: 重置为 init_noise_std（Stage1→Stage2 冷启动）
        # 向后兼容：若 cfg 中显式设置了 reset_noise_std_on_resume，以其为准。
        reset_noise: bool
        if 'reset_noise_std_on_resume' in self.cfg:
            reset_noise = bool(self.cfg.get('reset_noise_std_on_resume'))
            print(f"[Runner] reset_noise_std_on_resume = {reset_noise} (explicit config override)")
        else:
            reset_noise = not is_full_resume   # is_full_resume=True → 不重置; False → 重置
            print(f"[Runner] reset_noise_std = {reset_noise} (derived from is_full_resume={is_full_resume})")

        if reset_noise and (hasattr(self.alg.policy, 'std') or hasattr(self.alg.policy, 'log_std')):
            init_noise_std = self.policy_cfg.get("init_noise_std", 1.0)
            noise_std_type = self.policy_cfg.get("noise_std_type", "scalar")
            num_actions = (self.alg.policy.std.shape[0] if hasattr(self.alg.policy, 'std')
                           else self.alg.policy.log_std.shape[0])
            if noise_std_type == "scalar":
                self.alg.policy.std.data = torch.ones(num_actions, device=self.device) * init_noise_std
                print(f"[Runner] Reset noise std → {init_noise_std}")
            elif noise_std_type == "log":
                self.alg.policy.log_std.data = torch.log(
                    torch.ones(num_actions, device=self.device) * init_noise_std)
                print(f"[Runner] Reset log_std → log({init_noise_std})")
        else:
            if hasattr(self.alg.policy, 'std'):
                print(f"[Runner] Kept noise std from checkpoint = {self.alg.policy.std.mean().item():.4f}")

        # -- Freeze normalizer if specified in config (for stage transitions)
        # This prevents normalizer statistics from drifting when resuming from distillation
        freeze_normalizer = self.cfg.get("freeze_normalizer_on_resume", False)
        print(f"[Runner] freeze_normalizer_on_resume = {freeze_normalizer}")
        if freeze_normalizer and self.empirical_normalization:
            # Freeze obs normalizer
            self.obs_normalizer.eval()
            if hasattr(self.obs_normalizer, 'until'):
                self.obs_normalizer.until = self.obs_normalizer.count  # Stop updating
            print(f"[Runner] Froze obs_normalizer (count={self.obs_normalizer.count})")

            # Freeze privileged obs normalizer
            self.privileged_obs_normalizer.eval()
            if hasattr(self.privileged_obs_normalizer, 'until'):
                self.privileged_obs_normalizer.until = self.privileged_obs_normalizer.count
            print(f"[Runner] Froze privileged_obs_normalizer (count={self.privileged_obs_normalizer.count})")

        # Restore adaptive DR scale so resume continues from the correct DR level.
        # is_full_resume=True  (Stage2断点续训): 恢复 checkpoint 中的 dr_scale
        # is_full_resume=False (Stage1→Stage2冷启动): 忽略 checkpoint dr_scale，
        #   改用 cfg 中的 dr_scale_init（默认 1.0），确保 Stage2 从 Stage1 训练强度出发，
        #   避免 dr_scale=0 时 Stage1 修正策略作用于干净参考导致的即时崩溃。
        if is_full_resume:
            self._dr_scale      = loaded_dict.get("dr_scale",      0.0)
            self._dr_prev_error = loaded_dict.get("dr_prev_error", 0.0)
            if "frontres_boundary_ema" in loaded_dict:
                self._frontres_boundary_ema = dict(loaded_dict["frontres_boundary_ema"])
            if "last_frontres_boundary_stats" in loaded_dict:
                self._last_frontres_boundary_stats = dict(loaded_dict["last_frontres_boundary_stats"])
            self._frontres_gmt_frontier_safe_low = float(
                loaded_dict.get("frontres_gmt_frontier_safe_low", self._dr_scale)
            )
            self._frontres_gmt_frontier_broken_high = loaded_dict.get(
                "frontres_gmt_frontier_broken_high", None
            )
            if self._frontres_gmt_frontier_broken_high is not None:
                self._frontres_gmt_frontier_broken_high = float(self._frontres_gmt_frontier_broken_high)
            self._frontres_gmt_frontier_probe_scale = float(
                loaded_dict.get("frontres_gmt_frontier_probe_scale", self._dr_scale)
            )
            self._frontres_gmt_frontier_probe_score = loaded_dict.get(
                "frontres_gmt_frontier_probe_score", None
            )
            if self._frontres_gmt_frontier_probe_score is not None:
                self._frontres_gmt_frontier_probe_score = float(self._frontres_gmt_frontier_probe_score)
            self._frontres_gmt_frontier_decision = str(
                loaded_dict.get("frontres_gmt_frontier_decision", "resume")
            )
            self._frontres_gmt_frontier_confirmed = float(
                loaded_dict.get("frontres_gmt_frontier_confirmed", self._frontres_gmt_frontier_safe_low)
            )
            print(f"[Runner] Adaptive DR scale restored from checkpoint: {self._dr_scale:.4f}")
        else:
            _dr_init = float(self.cfg.get("dr_scale_init", 1.0))
            self._dr_scale = _dr_init
            self._frontres_boundary_ema = None
            self._last_frontres_boundary_stats = None
            self._frontres_gmt_frontier_safe_low = _dr_init
            self._frontres_gmt_frontier_broken_high = None
            self._frontres_gmt_frontier_probe_scale = _dr_init
            self._frontres_gmt_frontier_probe_score = None
            self._frontres_gmt_frontier_decision = "cold_start"
            self._frontres_gmt_frontier_confirmed = _dr_init
            print(f"[Runner] Stage1→Stage2 cold-start: dr_scale initialised to "
                  f"dr_scale_init={_dr_init:.4f} (ignoring checkpoint value "
                  f"{loaded_dict.get('dr_scale', 0.0):.4f})")

        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        if self.cfg["empirical_normalization"] and device is not None:
            self.obs_normalizer.to(device)

        is_task_space_frontres = (
            isinstance(self.alg.policy, FrontRESActorCritic)
            and getattr(self.alg.policy, "num_task_corrections", 0) > 0
        )

        if is_task_space_frontres:
            def policy(x):  # noqa: E306
                with torch.inference_mode():
                    raw_obs = x.to(self.device)
                    norm_obs = self._apply_obs_normalizer(raw_obs) if self.cfg["empirical_normalization"] else raw_obs
                    if (
                        bool(self.cfg.get("frontres_state_alpha_enabled", True))
                        and hasattr(self.alg.policy, "get_state_router_alpha")
                    ):
                        alpha_obs = norm_obs
                        self._frontres_state_alpha_prob_next = (
                            self.alg.policy.get_state_router_alpha(alpha_obs).view(-1).detach()
                        )
                    correction = self.alg.policy.get_task_correction_inference(norm_obs)
                    self._apply_frontres_task_corrections(correction, correction.shape[0], allow_oracle=False)
                    obs_corr, extras_corr = self.env.get_observations()
                    obs_corr_dict = extras_corr.get("observations", {})
                    if self.policy_obs_type is not None and self.policy_obs_type in obs_corr_dict:
                        obs_corr = obs_corr_dict[self.policy_obs_type]
                    obs_corr = obs_corr.to(self.device)
                    norm_corr = self._apply_obs_normalizer(obs_corr) if self.cfg["empirical_normalization"] else obs_corr
                    return self.alg.policy.get_env_action(norm_corr, correction)
            return policy

        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            policy = lambda x: self.alg.policy.act_inference(self._apply_obs_normalizer(x.to(self.device)))  # noqa: E731
        return policy

    def _apply_obs_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply obs_normalizer, with partial pass-through for FrontRES task-space mode.

        IsaacLab places Optional obs terms (anchor_root_pos_error_w, anchor_root_rpy_error_w)
        BEFORE regular terms in the concatenated obs tensor, so the layout is:
          [0 : num_extra]           = anchor-error dims  (FrontRES-only, NOT in GMT training)
          [num_extra : num_extra+gmt_dim] = GMT-compatible dims (match GMT training obs exactly)

        where num_extra = obs_dim - gmt_dim  (= 30 = 6 dims/frame × 5 frames).

        We therefore normalize the LAST gmt_dim dims with the frozen GMT normalizer and
        optionally normalize the FIRST num_extra dims with Stage-1 empirical stats.
        Output shape is unchanged (800 dims); structure: [extra | gmt_part].
        """
        if self._frontres_gmt_obs_dim is not None and obs.shape[-1] > self._frontres_gmt_obs_dim:
            gmt_dim   = self._frontres_gmt_obs_dim
            num_extra = obs.shape[-1] - gmt_dim          # = 30 (anchor errors at front)
            extra     = obs[:, :num_extra]               # [0:30]   anchor errors
            gmt_part  = self.obs_normalizer(obs[:, num_extra:])  # [30:800] GMT-compatible → normalize
            _s1_mean = getattr(self, '_frontres_extra_mean', None)
            _s1_std  = getattr(self, '_frontres_extra_std',  None)
            if (_s1_mean is not None and _s1_std is not None
                    and _s1_mean.shape[-1] == num_extra
                    and _s1_std.shape[-1] == num_extra):
                extra = (extra - _s1_mean) / (_s1_std + 1e-8)
            return torch.cat([extra, gmt_part], dim=-1)  # [anchor_errors | normalized_gmt]
        return self.obs_normalizer(obs)

    def _mask_frontres_task_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply the configured task-space action cone to correction proposals.

        In task-space mode actions are
        [dx, dy, dz, droll, dpitch, dyaw, alpha...].  The action cone should
        zero inactive correction proposals.  It should not force inactive
        sigmoid coefficients to zero: that value is on the boundary of the
        transformed action distribution and would corrupt PPO log-prob ratios.
        Coefficients on inactive axes are harmless once their proposal is zero.
        """
        active_dims = self.cfg.get("frontres_active_task_dims", None)
        if active_dims is None:
            return actions
        mask = torch.ones(actions.shape[-1], device=actions.device, dtype=actions.dtype)
        proposal_dim = min(6, actions.shape[-1])
        mask[:proposal_dim] = 0.0
        for idx in active_dims:
            idx = int(idx)
            if 0 <= idx < proposal_dim:
                mask[idx] = 1.0
        return actions * mask.view(1, -1)

    def _apply_frontres_task_corrections(
        self,
        task_corr: torch.Tensor | None,
        n_train: int | None = None,
        *,
        allow_oracle: bool = False,
        n_candidate: int = 0,
    ) -> torch.Tensor | None:
        """Write FrontRES ΔSE3 into the motion command before GMT/current env step.

        The policy samples/outputs ΔSE3 from the noisy observation.  This method
        applies the same conservative projection used by training rewards so a
        subsequent observation refresh exposes the corrected reference to GMT.
        """
        if task_corr is None:
            return None
        policy = getattr(getattr(self, "alg", None), "policy", None)
        if not isinstance(policy, FrontRESActorCritic):
            return task_corr
        if getattr(policy, "num_task_corrections", 0) <= 0:
            return task_corr

        task_corr = self._mask_frontres_task_actions(task_corr)
        env_raw = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if not (hasattr(env_raw, "command_manager") and hasattr(env_raw.command_manager, "_terms")):
            return task_corr

        if n_train is None:
            n_train = task_corr.shape[0]
        n_train = max(0, min(int(n_train), task_corr.shape[0], self.env.num_envs))
        n_candidate = max(0, min(int(n_candidate), max(0, self.env.num_envs - n_train)))
        self._frontres_stable_route_applied_frac = 0.0
        self._frontres_stable_endpoint_frac = 0.0
        self._frontres_tri_weight_repair = 0.0
        self._frontres_tri_weight_noisy = 1.0
        self._frontres_tri_weight_stable = 0.0
        self._frontres_stable_route_active_mask = torch.zeros(
            n_train, device=task_corr.device, dtype=torch.bool
        )

        if allow_oracle and self.cfg.get("oracle_curriculum", False):
            for cmd_oracle in env_raw.command_manager._terms.values():
                if hasattr(cmd_oracle, "supervised_target"):
                    sup = cmd_oracle.supervised_target.to(task_corr.device)
                    oracle_full = torch.zeros_like(task_corr)
                    n = min(sup.shape[-1], oracle_full.shape[-1])
                    oracle_full[:, :n] = sup[:, :n]

                    fr_v = task_corr[:n_train, :n]
                    or_v = oracle_full[:n_train, :n]
                    if fr_v.numel() > 0:
                        cos_s = (fr_v * or_v).sum(-1) / (fr_v.norm(dim=-1) * or_v.norm(dim=-1) + 1e-8)
                        ema_alpha = 0.99
                        prev_ema = getattr(self, "_oracle_cos_ema", 0.0)
                        new_ema = ema_alpha * prev_ema + (1.0 - ema_alpha) * float(cos_s.mean())
                        self._oracle_cos_ema = new_ema

                        cos_lo = float(self.cfg.get("oracle_mix_cos_low", 0.3))
                        cos_hi = float(self.cfg.get("oracle_mix_cos_high", 0.85))
                        if new_ema < cos_lo:
                            mix = 1.0
                        elif new_ema < cos_hi:
                            mix = 1.0 - (new_ema - cos_lo) / max(cos_hi - cos_lo, 1e-6)
                        else:
                            mix = 0.0
                        self._oracle_mix = mix
                        if mix > 0.0:
                            task_corr = (1.0 - mix) * task_corr + mix * oracle_full
                    break

        for cmd_term in env_raw.command_manager._terms.values():
            if not hasattr(cmd_term, "_frontres_pos_correction"):
                continue
            pos_corr = task_corr[:n_train, :3].clone()
            rpy_corr = task_corr[:n_train, 3:6].clone()
            objective = str(getattr(self.alg, "frontres_training_objective", "")).lower()
            task_conf_dim = int(getattr(policy, "task_conf_dim", 2))
            acceptance = None
            if task_corr.shape[-1] >= 12 and task_conf_dim == 6:
                acceptance = task_corr[:n_train, 6:12].clone().clamp(0.0, 1.0)
                c_pos = acceptance[:, :3]
                c_rpy = acceptance[:, 3:6]
            elif task_corr.shape[-1] >= 7 and task_conf_dim == 1:
                rho_pos = task_corr[:n_train, 6:7].clone().clamp(0.0, 1.0)
                c_pos = torch.ones_like(rho_pos)
                c_rpy = torch.ones_like(rho_pos)
            else:
                c_pos = task_corr[:n_train, 6:7].clone()
                c_rpy = task_corr[:n_train, 7:8].clone()
            if objective == "supervised_restore":
                c_pos = torch.ones_like(c_pos)
                c_rpy = torch.ones_like(c_rpy)
            rho_space = str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
            stable_to_repair = (
                objective == "hsl_hybrid"
                and acceptance is not None
                and rho_space in ("stable_to_repair", "stable-repair", "stable")
            )
            tri_anchor = (
                objective == "hsl_hybrid"
                and acceptance is not None
                and rho_space in ("tri_anchor", "tri-anchor", "tri")
            )

            z_upper = torch.zeros_like(pos_corr[:, 2])
            if hasattr(cmd_term, "jump_degree"):
                jd = cmd_term.jump_degree[:n_train].to(task_corr.device).clamp(0.0, 1.0)
                contact_gate = (1.0 - jd).unsqueeze(-1)
                pos_corr[:, :2] = pos_corr[:, :2] * contact_gate
                if hasattr(cmd_term, "anchor_penetration_depth"):
                    penetration = cmd_term.anchor_penetration_depth[:n_train].to(task_corr.device)
                    z_upper = jd * penetration

            z_lower = torch.full_like(pos_corr[:, 2], -self.alg.policy.max_delta_pos)
            pos_corr[:, 2] = torch.maximum(pos_corr[:, 2], z_lower)
            pos_corr[:, 2] = torch.minimum(pos_corr[:, 2], z_upper)
            # Candidate rollout remains the full HSL route.  Stable-route
            # substitution is applied only to Projected/HRL envs so Candidate
            # stays a clean evidence source for the next floor test.
            cand_pos_corr = pos_corr[:n_candidate].clone() if n_candidate > 0 else None
            cand_rpy_corr = rpy_corr[:n_candidate].clone() if n_candidate > 0 else None
            stable_pos_corr = None
            stable_rpy_corr = None
            if stable_to_repair:
                stable = self._frontres_stabilizing_candidate_correction(
                    cmd_term, n_train, task_corr.device, task_corr.dtype
                )
                if stable is not None:
                    stable_pos_corr, stable_rpy_corr = stable
                    # New HRL coordinate system:
                    #   old fallback: projected = rho * repair
                    #   active branch: projected = stable + rho * (repair - stable)
                    pos_corr = stable_pos_corr + c_pos * (pos_corr - stable_pos_corr)
                    rpy_corr = stable_rpy_corr + c_rpy * (rpy_corr - stable_rpy_corr)
                    c_pos = torch.ones_like(c_pos)
                    c_rpy = torch.ones_like(c_rpy)
                    self._frontres_stable_endpoint_frac = 1.0
                else:
                    self._frontres_stable_endpoint_frac = 0.0
            elif tri_anchor:
                stable = self._frontres_stabilizing_candidate_correction(
                    cmd_term, n_train, task_corr.device, task_corr.dtype
                )
                if stable is not None:
                    stable_pos_corr, stable_rpy_corr = stable
                    alpha_prob = getattr(self, "_frontres_state_alpha_prob_next", None)
                    if alpha_prob is None:
                        alpha = torch.zeros(n_train, device=task_corr.device, dtype=task_corr.dtype)
                    else:
                        alpha = alpha_prob.to(device=task_corr.device, dtype=task_corr.dtype).view(-1)
                        if alpha.numel() < n_train:
                            alpha = torch.nn.functional.pad(alpha, (0, n_train - alpha.numel()), value=0.0)
                        alpha = alpha[:n_train].clamp(0.0, 1.0)
                    alpha_pos = alpha[:, None]
                    pos_corr = c_pos * pos_corr + (1.0 - c_pos) * alpha_pos * stable_pos_corr
                    rpy_corr = c_rpy * rpy_corr + (1.0 - c_rpy) * alpha_pos * stable_rpy_corr
                    self._frontres_tri_weight_repair = float(
                        0.5 * (c_pos.mean() + c_rpy.mean()).detach().item()
                    )
                    self._frontres_tri_weight_stable = float(
                        0.5
                        * (
                            ((1.0 - c_pos) * alpha_pos).mean()
                            + ((1.0 - c_rpy) * alpha_pos).mean()
                        ).detach().item()
                    )
                    self._frontres_tri_weight_noisy = max(
                        0.0,
                        1.0 - self._frontres_tri_weight_repair - self._frontres_tri_weight_stable,
                    )
                    c_pos = torch.ones_like(c_pos)
                    c_rpy = torch.ones_like(c_rpy)
                    self._frontres_stable_endpoint_frac = 0.0
                else:
                    self._frontres_stable_endpoint_frac = 0.0
                    self._frontres_tri_weight_repair = float(
                        0.5 * (c_pos.mean() + c_rpy.mean()).detach().item()
                    )
                    self._frontres_tri_weight_noisy = max(0.0, 1.0 - self._frontres_tri_weight_repair)
                    self._frontres_tri_weight_stable = 0.0
            if (
                (not tri_anchor)
                and allow_oracle
                and n_candidate > 0
                and bool(self.cfg.get("frontres_stable_route_enabled", True))
            ):
                route_mask = getattr(self, "_frontres_stable_route_next_mask", None)
                if route_mask is not None:
                    route_mask = route_mask.to(device=task_corr.device).view(-1).bool()
                    if route_mask.numel() < n_train:
                        route_mask = torch.nn.functional.pad(route_mask, (0, n_train - route_mask.numel()), value=False)
                    route_mask = route_mask[:n_train]
                    if route_mask.any():
                        if stable_pos_corr is None or stable_rpy_corr is None:
                            stable = self._frontres_stabilizing_candidate_correction(
                                cmd_term, n_train, task_corr.device, task_corr.dtype
                            )
                            if stable is not None:
                                stable_pos_corr, stable_rpy_corr = stable
                        if stable_pos_corr is not None and stable_rpy_corr is not None:
                            pos_corr = torch.where(route_mask[:, None], stable_pos_corr, pos_corr)
                            rpy_corr = torch.where(route_mask[:, None], stable_rpy_corr, rpy_corr)
                            c_pos = torch.where(route_mask[:, None], torch.ones_like(c_pos), c_pos)
                            c_rpy = torch.where(route_mask[:, None], torch.ones_like(c_rpy), c_rpy)
                            self._frontres_stable_route_active_mask = route_mask.detach()
                            self._frontres_stable_route_applied_frac = float(
                                route_mask.to(task_corr.dtype).mean().detach().item()
                            )
                        else:
                            self._frontres_stable_route_active_mask = torch.zeros(
                                n_train, device=task_corr.device, dtype=torch.bool
                            )
                            self._frontres_stable_route_applied_frac = 0.0
                    else:
                        self._frontres_stable_route_active_mask = torch.zeros(
                            n_train, device=task_corr.device, dtype=torch.bool
                        )
                        self._frontres_stable_route_applied_frac = 0.0
                else:
                    self._frontres_stable_route_active_mask = torch.zeros(
                        n_train, device=task_corr.device, dtype=torch.bool
                    )
                    self._frontres_stable_route_applied_frac = 0.0
            else:
                self._frontres_stable_route_active_mask = torch.zeros(
                    n_train, device=task_corr.device, dtype=torch.bool
                )
                self._frontres_stable_route_applied_frac = 0.0
            # Legacy Noisy-to-Repair branch applies rho after route selection.
            # In Stable-to-Repair mode rho has already been absorbed into
            # stable + rho * (repair - stable), so c_pos/c_rpy were reset to 1.
            pos_corr = pos_corr * c_pos
            rpy_corr = rpy_corr * c_rpy

            cmd_term._frontres_pos_correction[:n_train].copy_(pos_corr)
            cmd_term._frontres_quat_correction[:n_train].copy_(_rotvec_to_quat_wxyz(rpy_corr))
            if n_candidate > 0 and cand_pos_corr is not None and cand_rpy_corr is not None:
                cand_start = n_train
                cand_end = cand_start + n_candidate
                cmd_term._frontres_pos_correction[cand_start:cand_end].copy_(cand_pos_corr)
                cmd_term._frontres_quat_correction[cand_start:cand_end].copy_(_rotvec_to_quat_wxyz(cand_rpy_corr))
            self._frontres_update_temporal_reference_cache(cmd_term, n_train)
            zero_start = n_train + n_candidate
            if zero_start < self.env.num_envs:
                cmd_term._frontres_pos_correction[zero_start:].zero_()
                cmd_term._frontres_quat_correction[zero_start:].zero_()
                cmd_term._frontres_quat_correction[zero_start:, 0] = 1.0
        return task_corr

    def _frontres_raw_anchor_pose(self, cmd_term, n: int, device: torch.device, dtype: torch.dtype):
        """Return raw reference pose before FrontRES correction."""
        if hasattr(cmd_term, "anchor_pos_w_raw"):
            raw_pos = cmd_term.anchor_pos_w_raw[:n].to(device=device, dtype=dtype)
        elif hasattr(cmd_term, "anchor_pos_w"):
            raw_pos = (
                cmd_term.anchor_pos_w[:n].to(device=device, dtype=dtype)
                - cmd_term._frontres_pos_correction[:n].to(device=device, dtype=dtype)
            )
        else:
            return None
        if hasattr(cmd_term, "anchor_quat_w_raw"):
            raw_quat = cmd_term.anchor_quat_w_raw[:n].to(device=device, dtype=dtype)
        elif hasattr(cmd_term, "anchor_quat_w"):
            written_q = cmd_term._frontres_quat_correction[:n].to(device=device, dtype=dtype)
            raw_quat = quat_mul(
                cmd_term.anchor_quat_w[:n].to(device=device, dtype=dtype),
                quat_inv(written_q),
            )
        else:
            return None
        raw_quat = raw_quat / raw_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return raw_pos, raw_quat

    def _frontres_stabilizing_candidate_correction(
        self,
        cmd_term,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Build a deterministic stable-manifold candidate for HRL route selection.

        This is not a new policy or rollout branch.  It replaces the Projected
        route with a conservative upright reference when the previous full-HSL
        Candidate rollout fell below the executable floor.
        """
        if n <= 0:
            return None
        pose = self._frontres_raw_anchor_pose(cmd_term, n, device, dtype)
        if pose is None or not hasattr(cmd_term, "robot_anchor_quat_w"):
            return None
        raw_pos, raw_quat = pose
        robot_quat = cmd_term.robot_anchor_quat_w[:n].to(device=device, dtype=dtype)
        _, _, robot_yaw = euler_xyz_from_quat(robot_quat)
        zeros = torch.zeros_like(robot_yaw)
        stable_quat = quat_from_euler_xyz(zeros, zeros, robot_yaw)
        stable_quat = stable_quat / stable_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        stable_corr_quat = quat_mul(quat_inv(raw_quat), stable_quat)
        stable_rpy_corr = _quat_to_rotvec_wxyz(stable_corr_quat)
        stable_pos_corr = torch.zeros_like(raw_pos)

        max_delta_pos = float(getattr(getattr(self.alg, "policy", None), "max_delta_pos", 0.3))
        max_delta_rpy = float(getattr(getattr(self.alg, "policy", None), "max_delta_rpy", 0.1))
        stable_pos_corr = stable_pos_corr.clamp(-max_delta_pos, max_delta_pos)
        stable_rpy_corr = stable_rpy_corr.clamp(-max_delta_rpy, max_delta_rpy)

        active_dims = self.cfg.get("frontres_active_task_dims", None)
        if active_dims is not None:
            active = {int(dim) for dim in active_dims}
            for dim in range(3):
                if dim not in active:
                    stable_pos_corr[:, dim] = 0.0
            for dim in range(3, 6):
                if dim not in active:
                    stable_rpy_corr[:, dim - 3] = 0.0
        return stable_pos_corr, stable_rpy_corr

    def _frontres_temporal_continuity_correction(
        self,
        cmd_term,
        n: int,
        hsl_pos_corr: torch.Tensor,
        hsl_rpy_corr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Build the continuity candidate from the previous refined frame.

        HSL gives the clean-oriented repair for the current raw frame.  This
        helper is kept for legacy temporal-rejoin ablations; the active
        hsl_hybrid branch now uses PPO-owned per-axis acceptance over the HSL
        proposal rather than a temporal position rejoin.
        """
        if n <= 0:
            return None
        pose = self._frontres_raw_anchor_pose(cmd_term, n, hsl_pos_corr.device, hsl_pos_corr.dtype)
        if pose is None:
            return None
        raw_pos, raw_quat = pose

        cache = getattr(self, "_frontres_temporal_ref_cache", None)
        if cache is None:
            return None
        prev_raw_pos = cache.get("raw_pos")
        prev_raw_quat = cache.get("raw_quat")
        prev_ref_pos = cache.get("refined_pos")
        prev_ref_quat = cache.get("refined_quat")
        prev_valid = cache.get("valid")
        if (
            prev_raw_pos is None
            or prev_raw_quat is None
            or prev_ref_pos is None
            or prev_ref_quat is None
            or prev_valid is None
        ):
            return None
        if prev_raw_pos.shape[0] < n or prev_ref_pos.shape[0] < n:
            return None

        prev_raw_pos = prev_raw_pos[:n].to(device=raw_pos.device, dtype=raw_pos.dtype)
        prev_ref_pos = prev_ref_pos[:n].to(device=raw_pos.device, dtype=raw_pos.dtype)
        prev_raw_quat = prev_raw_quat[:n].to(device=raw_pos.device, dtype=raw_pos.dtype)
        prev_ref_quat = prev_ref_quat[:n].to(device=raw_pos.device, dtype=raw_pos.dtype)
        valid = prev_valid[:n].to(device=raw_pos.device).bool()

        raw_delta_pos = raw_pos - prev_raw_pos
        cont_ref_pos = prev_ref_pos + raw_delta_pos
        cont_pos_corr = cont_ref_pos - raw_pos

        raw_delta_quat = quat_mul(quat_inv(prev_raw_quat), raw_quat)
        cont_ref_quat = quat_mul(prev_ref_quat, raw_delta_quat)
        cont_corr_quat = quat_mul(quat_inv(raw_quat), cont_ref_quat)
        cont_rpy_corr = _quat_to_rotvec_wxyz(cont_corr_quat)

        max_delta_pos = float(getattr(getattr(self.alg, "policy", None), "max_delta_pos", 0.3))
        max_delta_rpy = float(getattr(getattr(self.alg, "policy", None), "max_delta_rpy", 0.1))
        cont_pos_corr = cont_pos_corr.clamp(-max_delta_pos, max_delta_pos)
        cont_rpy_corr = cont_rpy_corr.clamp(-max_delta_rpy, max_delta_rpy)

        z_upper = torch.zeros(n, device=raw_pos.device, dtype=raw_pos.dtype)
        if hasattr(cmd_term, "jump_degree") and hasattr(cmd_term, "anchor_penetration_depth"):
            jump_degree = cmd_term.jump_degree[:n].to(raw_pos.device).to(raw_pos.dtype).clamp(0.0, 1.0)
            penetration = cmd_term.anchor_penetration_depth[:n].to(raw_pos.device).to(raw_pos.dtype)
            z_upper = (jump_degree * penetration).clamp(max=max_delta_pos)
        z_lower = torch.full_like(z_upper, -max_delta_pos)
        cont_pos_corr[:, 2] = torch.minimum(torch.maximum(cont_pos_corr[:, 2], z_lower), z_upper)
        return cont_pos_corr, cont_rpy_corr, valid

    def _frontres_update_temporal_reference_cache(self, cmd_term, n: int) -> None:
        """Cache raw/refined reference poses after writing FrontRES corrections."""
        if n <= 0:
            return
        device = cmd_term._frontres_pos_correction.device
        dtype = cmd_term._frontres_pos_correction.dtype
        pose = self._frontres_raw_anchor_pose(cmd_term, n, device, dtype)
        if pose is None:
            return
        raw_pos, raw_quat = pose
        pos_corr = cmd_term._frontres_pos_correction[:n].detach().clone()
        quat_corr = cmd_term._frontres_quat_correction[:n].detach().clone()
        refined_pos = raw_pos.detach().clone() + pos_corr
        refined_quat = quat_mul(raw_quat, quat_corr)
        refined_quat = refined_quat / refined_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cache_size = int(self.env.num_envs)
        cache = getattr(self, "_frontres_temporal_ref_cache", None)
        if cache is None or cache.get("raw_pos", torch.empty(0, device=device)).shape[0] != cache_size:
            cache = {
                "raw_pos": torch.zeros(cache_size, 3, device=device, dtype=dtype),
                "raw_quat": torch.zeros(cache_size, 4, device=device, dtype=dtype),
                "refined_pos": torch.zeros(cache_size, 3, device=device, dtype=dtype),
                "refined_quat": torch.zeros(cache_size, 4, device=device, dtype=dtype),
                "valid": torch.zeros(cache_size, device=device, dtype=torch.bool),
            }
            cache["raw_quat"][:, 0] = 1.0
            cache["refined_quat"][:, 0] = 1.0
            self._frontres_temporal_ref_cache = cache
        cache["raw_pos"][:n].copy_(raw_pos.detach())
        cache["raw_quat"][:n].copy_(raw_quat.detach())
        cache["refined_pos"][:n].copy_(refined_pos.detach())
        cache["refined_quat"][:n].copy_(refined_quat.detach())
        cache["valid"][:n] = True

    def _frontres_invalidate_temporal_reference_cache(self, dones: torch.Tensor | None) -> None:
        """Drop temporal continuity state for environments that just reset."""
        cache = getattr(self, "_frontres_temporal_ref_cache", None)
        if cache is None or dones is None or "valid" not in cache:
            return
        valid = cache["valid"]
        done_mask = dones.to(device=valid.device).view(-1).bool()
        n = min(done_mask.numel(), valid.numel())
        if n <= 0:
            return
        valid[:n] &= ~done_mask[:n]

    def _maybe_print_frontres_restore_debug(
        self,
        it: int,
        rollout_step: int,
        actions: torch.Tensor | None,
        supervised_target: torch.Tensor | None,
        n_train: int,
    ) -> None:
        """Low-frequency consistency print for task-space FrontRES restore."""
        if actions is None or supervised_target is None:
            return
        if rollout_step != 0:
            return
        objective = str(getattr(self.alg, "frontres_training_objective", "")).lower()
        if objective not in ("supervised_restore", "basis_restore", "hsl_hybrid"):
            return
        interval = int(getattr(self.alg, "frontres_restore_debug_print_interval", self.cfg.get("frontres_restore_debug_print_interval", 10)))
        if interval <= 0 or int(it) % interval != 0:
            return
        if getattr(self, "_frontres_restore_debug_last_iter", None) == int(it):
            return
        self._frontres_restore_debug_last_iter = int(it)

        env_raw = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if not (hasattr(env_raw, "command_manager") and hasattr(env_raw.command_manager, "_terms")):
            return
        cmd_term = None
        for term in env_raw.command_manager._terms.values():
            needed = (
                "anchor_quat_w_original",
                "anchor_quat_w_raw",
                "_frontres_quat_correction",
            )
            if all(hasattr(term, name) for name in needed):
                cmd_term = term
                break
        if cmd_term is None:
            return

        n = max(0, min(int(n_train), actions.shape[0], supervised_target.shape[0]))
        if n <= 0:
            return

        raw_q = cmd_term.anchor_quat_w_raw[:n].to(self.device)
        clean_q = cmd_term.anchor_quat_w_original[:n].to(self.device)
        written_q = cmd_term._frontres_quat_correction[:n].to(self.device)
        have_pos_debug = (
            hasattr(cmd_term, "anchor_pos_w_raw")
            and hasattr(cmd_term, "anchor_pos_w_original")
            and hasattr(cmd_term, "_frontres_pos_correction")
        )
        if have_pos_debug:
            raw_p = cmd_term.anchor_pos_w_raw[:n].to(self.device)
            clean_p = cmd_term.anchor_pos_w_original[:n].to(self.device)
            written_p = cmd_term._frontres_pos_correction[:n].to(self.device)
            target_pos = supervised_target[:n, :3].detach()
            pred_pos = actions[:n, :3].detach()
        target = supervised_target[:n, 3:6].detach()
        pred = actions[:n, 3:6].detach()
        task_conf_dim = int(getattr(self.alg.policy, "task_conf_dim", 2))
        if actions.shape[-1] >= 12 and task_conf_dim == 6:
            conf_raw = actions[:n, 6:12].detach()
        elif actions.shape[-1] >= 7 and task_conf_dim == 1:
            conf_raw = actions[:n, 6:7].detach()
        elif actions.shape[-1] >= 8:
            conf_raw = actions[:n, 7:8].detach()
        else:
            conf_raw = torch.ones(n, 1, device=self.device)
        acceptance_hybrid = objective == "hsl_hybrid" and task_conf_dim == 6
        scalar_rejoin_hybrid = objective == "hsl_hybrid" and task_conf_dim == 1
        if objective == "supervised_restore" or scalar_rejoin_hybrid:
            conf_eff = torch.ones_like(conf_raw)
        else:
            conf_eff = conf_raw
        written = _quat_to_rotvec_wxyz(written_q)[:, :3]
        if acceptance_hybrid:
            applied = pred * conf_eff[:, 3:6]
        elif scalar_rejoin_hybrid:
            applied = written
        else:
            applied = pred * conf_eff

        clean_from_raw = _quat_to_rotvec_wxyz(quat_mul(quat_inv(raw_q), clean_q))[:, :3]
        corrected_q = quat_mul(raw_q, written_q)
        corrected_err = _quat_to_rotvec_wxyz(quat_mul(quat_inv(corrected_q), clean_q))[:, :3]
        alt_corrected_q = quat_mul(written_q, raw_q)
        alt_corrected_err = _quat_to_rotvec_wxyz(quat_mul(quat_inv(alt_corrected_q), clean_q))[:, :3]

        def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)

        valid = target.norm(dim=-1) > 1e-4
        if valid.any():
            cos_pred_target = _safe_cos(pred[valid], target[valid]).mean()
            cos_written_target = _safe_cos(written[valid], target[valid]).mean()
            sign_match = (torch.sign(pred[valid, :2]) == torch.sign(target[valid, :2])).float().mean()
        else:
            cos_pred_target = torch.tensor(0.0, device=self.device)
            cos_written_target = torch.tensor(0.0, device=self.device)
            sign_match = torch.tensor(0.0, device=self.device)

        noisy_err_norm = clean_from_raw[:, :2].norm(dim=-1)
        corrected_err_norm = corrected_err[:, :2].norm(dim=-1)
        alt_err_norm = alt_corrected_err[:, :2].norm(dim=-1)
        restore_gain = noisy_err_norm - corrected_err_norm
        if have_pos_debug:
            raw_pos_err = (clean_p - raw_p).norm(dim=-1)
            written_pos_err = (clean_p - (raw_p + written_p)).norm(dim=-1)
            candidate_pos_err = (clean_p - (raw_p + pred_pos)).norm(dim=-1)
            pos_valid = target_pos.norm(dim=-1) > 1e-4
            if pos_valid.any():
                pos_cos_pred = _safe_cos(pred_pos[pos_valid], target_pos[pos_valid]).mean()
                pos_cos_written = _safe_cos(written_p[pos_valid], target_pos[pos_valid]).mean()
            else:
                pos_cos_pred = torch.tensor(0.0, device=self.device)
                pos_cos_written = torch.tensor(0.0, device=self.device)
        candidate_rpy_err = (clean_from_raw[:, :2] - pred[:, :2]).norm(dim=-1)
        projected_rpy_err = (clean_from_raw[:, :2] - applied[:, :2]).norm(dim=-1)
        max_delta_rpy = float(getattr(getattr(self.alg, "policy", None), "max_delta_rpy", 0.4))
        sat_frac = (pred[:, :2].abs() > 0.95 * max_delta_rpy).float().mean()

        prev = getattr(self, "_frontres_restore_debug_prev_applied", None)
        if prev is not None and prev.shape == applied.shape:
            step_jump = (applied[:, :2] - prev[:, :2]).norm(dim=-1).mean()
        else:
            step_jump = torch.tensor(0.0, device=self.device)
        self._frontres_restore_debug_prev_applied = applied.detach().clone()

        def _vec(t: torch.Tensor, idx: int = 0) -> list[float]:
            vals = t[idx, :3].detach().cpu().tolist()
            return [round(float(v), 5) for v in vals]

        def _vec6(t: torch.Tensor) -> list[float]:
            vals = t.detach().mean(dim=0).cpu().tolist()
            return [round(float(v), 5) for v in vals]

        def _stats(t: torch.Tensor) -> str:
            vals = t.detach().reshape(-1)
            if vals.numel() == 0:
                return "mean=0.000 std=0.000 min=0.000 max=0.000"
            return (
                f"mean={float(vals.mean()):.3f} "
                f"std={float(vals.std(unbiased=False)):.3f} "
                f"min={float(vals.min()):.3f} "
                f"max={float(vals.max()):.3f}"
            )

        def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
            aa = a.detach().reshape(-1)
            bb = b.detach().reshape(-1)
            mask = torch.isfinite(aa) & torch.isfinite(bb)
            if int(mask.sum().item()) < 2:
                return 0.0
            aa = aa[mask] - aa[mask].mean()
            bb = bb[mask] - bb[mask].mean()
            denom = (aa.norm() * bb.norm()).clamp(min=1e-8)
            return float((aa * bb).sum() / denom)

        def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> float:
            if mask is None or not bool(mask.any()):
                return 0.0
            return float(t.detach()[mask].mean())

        def _branch_state_metrics(
            branch_pos: torch.Tensor | None,
            branch_quat: torch.Tensor,
            robot_pos: torch.Tensor | None,
            robot_quat: torch.Tensor,
            robot_lin_vel: torch.Tensor | None,
            robot_ang_vel: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            rot_err = _quat_to_rotvec_wxyz(quat_mul(quat_inv(robot_quat), branch_quat))[:, :3]
            rot_dist = rot_err.norm(dim=-1)
            if branch_pos is not None and robot_pos is not None:
                pos_err = branch_pos - robot_pos
                pos_dist = pos_err.norm(dim=-1)
            else:
                pos_err = None
                pos_dist = torch.zeros_like(rot_dist)
            dist = pos_dist + rot_dist

            compat = torch.zeros_like(rot_dist)
            if pos_err is not None and robot_lin_vel is not None:
                compat = compat + (pos_err * robot_lin_vel).sum(-1) / (
                    pos_err.norm(dim=-1) * robot_lin_vel.norm(dim=-1) + 1e-8
                )
            if robot_ang_vel is not None:
                compat = compat + 0.5 * (rot_err * robot_ang_vel).sum(-1) / (
                    rot_err.norm(dim=-1) * robot_ang_vel.norm(dim=-1) + 1e-8
                )
            if robot_ang_vel is not None:
                rp_err = rot_err[:, :2]
                rp_vel = robot_ang_vel[:, :2]
                compat_rp = (rp_err * rp_vel).sum(-1) / (
                    rp_err.norm(dim=-1) * rp_vel.norm(dim=-1) + 1e-8
                )
            else:
                compat_rp = torch.zeros_like(rot_dist)
            return dist, compat, compat_rp

        inertial_debug = None
        have_inertial_debug = (
            hasattr(cmd_term, "robot_anchor_quat_w")
            and hasattr(cmd_term, "robot_anchor_ang_vel_w")
        )
        if have_inertial_debug:
            robot_q = cmd_term.robot_anchor_quat_w[:n].to(self.device)
            robot_w = cmd_term.robot_anchor_ang_vel_w[:n].to(self.device)
            robot_p = None
            robot_v = None
            noisy_p = candidate_p = projected_p = clean_pos_branch = None
            if have_pos_debug and hasattr(cmd_term, "robot_anchor_pos_w"):
                robot_p = cmd_term.robot_anchor_pos_w[:n].to(self.device)
                noisy_p = raw_p
                candidate_p = raw_p + pred_pos
                clean_pos_branch = clean_p
                if acceptance_hybrid:
                    projected_pos_corr = pred_pos * conf_eff[:, :3]
                else:
                    projected_pos_corr = pred_pos * conf_eff
                projected_p = raw_p + projected_pos_corr
                if hasattr(cmd_term, "robot_anchor_lin_vel_w"):
                    robot_v = cmd_term.robot_anchor_lin_vel_w[:n].to(self.device)

            noisy_q = raw_q
            candidate_q = quat_mul(raw_q, _rotvec_to_quat_wxyz(pred))
            projected_q = quat_mul(raw_q, _rotvec_to_quat_wxyz(applied))
            clean_branch_q = clean_q
            d_noisy, c_noisy, crp_noisy = _branch_state_metrics(
                noisy_p, noisy_q, robot_p, robot_q, robot_v, robot_w
            )
            d_proj, c_proj, crp_proj = _branch_state_metrics(
                projected_p, projected_q, robot_p, robot_q, robot_v, robot_w
            )
            d_cand, c_cand, crp_cand = _branch_state_metrics(
                candidate_p, candidate_q, robot_p, robot_q, robot_v, robot_w
            )
            d_clean, c_clean, crp_clean = _branch_state_metrics(
                clean_pos_branch, clean_branch_q, robot_p, robot_q, robot_v, robot_w
            )
            if noisy_p is not None and projected_p is not None:
                d_p0r1, c_p0r1, crp_p0r1 = _branch_state_metrics(
                    noisy_p, candidate_q, robot_p, robot_q, robot_v, robot_w
                )
                d_ppr1, c_ppr1, crp_ppr1 = _branch_state_metrics(
                    projected_p, candidate_q, robot_p, robot_q, robot_v, robot_w
                )
            else:
                d_p0r1 = c_p0r1 = crp_p0r1 = torch.zeros_like(d_noisy)
                d_ppr1 = c_ppr1 = crp_ppr1 = torch.zeros_like(d_noisy)
            margin = 0.05
            angle_inv = (d_proj > d_noisy).float()
            anti_inertia = (c_proj < c_noisy - margin).float()
            anti_cand = (c_cand < c_noisy - margin).float()
            anti_clean = (c_clean < c_noisy - margin).float()
            inertial_debug = {
                "d_noisy": d_noisy,
                "d_proj": d_proj,
                "d_cand": d_cand,
                "d_clean": d_clean,
                "c_noisy": c_noisy,
                "c_proj": c_proj,
                "c_cand": c_cand,
                "c_clean": c_clean,
                "crp_noisy": crp_noisy,
                "crp_proj": crp_proj,
                "crp_cand": crp_cand,
                "crp_clean": crp_clean,
                "d_p0r1": d_p0r1,
                "d_ppr1": d_ppr1,
                "c_p0r1": c_p0r1,
                "c_ppr1": c_ppr1,
                "crp_p0r1": crp_p0r1,
                "crp_ppr1": crp_ppr1,
                "angle_inv": angle_inv,
                "anti_inertia": anti_inertia,
                "anti_cand": anti_cand,
                "anti_clean": anti_clean,
                "inertial_gain": c_proj - c_noisy,
            }

        sample_idx = int(torch.argmax(noisy_err_norm).item())
        if acceptance_hybrid:
            gate_desc = (
                f"rho_pos={float(conf_raw[:, :3].mean()):.3f} "
                f"rho_rpy={float(conf_raw[:, 3:6].mean()):.3f}"
            )
        elif scalar_rejoin_hybrid:
            gate_desc = f"rho_pos={float(conf_raw.mean()):.3f}"
        else:
            gate_desc = f"conf_eff={float(conf_eff.mean()):.3f} conf_raw={float(conf_raw.mean()):.3f}"
        print(
            "[FrontRES restore debug] "
            f"it={int(it)} dr={float(getattr(self, '_dr_scale', 0.0)):.4f} "
            f"n={n} sample={sample_idx} "
            f"cos(pred,target)={float(cos_pred_target):+.4f} "
            f"cos(written,target)={float(cos_written_target):+.4f} "
            f"sign_xy={float(sign_match):.3f} {gate_desc} "
            f"sat={float(sat_frac):.3f} jump={float(step_jump):.5f}",
            flush=True,
        )
        print(
            "[FrontRES restore debug] "
            f"|raw-clean|={float(noisy_err_norm.mean()):.5f} "
            f"|corr-clean|={float(corrected_err_norm.mean()):.5f} "
            f"|altcorr-clean|={float(alt_err_norm.mean()):.5f} "
            f"gain={float(restore_gain.mean()):+.5f}",
            flush=True,
        )
        print(
            "[FrontRES restore debug] rpy counterfactual "
            f"noop={float(noisy_err_norm.mean()):.5f} "
            f"full_candidate={float(candidate_rpy_err.mean()):.5f} "
            f"projected={float(projected_rpy_err.mean()):.5f} "
            f"written={float(corrected_err_norm.mean()):.5f} "
            f"gain_full={float((noisy_err_norm - candidate_rpy_err).mean()):+.5f} "
            f"gain_proj={float((noisy_err_norm - projected_rpy_err).mean()):+.5f}",
            flush=True,
        )
        if have_pos_debug:
            print(
                "[FrontRES restore debug] position "
                f"|raw-clean|={float(raw_pos_err.mean()):.5f} "
                f"|written-clean|={float(written_pos_err.mean()):.5f} "
                f"|candidate-clean|={float(candidate_pos_err.mean()):.5f} "
                f"gain_cand={float((raw_pos_err - candidate_pos_err).mean()):+.5f} "
                f"gain_written={float((raw_pos_err - written_pos_err).mean()):+.5f} "
                f"cos(pred,target)={float(pos_cos_pred):+.4f} "
                f"cos(written,target)={float(pos_cos_written):+.4f}",
                flush=True,
            )
        if acceptance_hybrid:
            gate_mean = conf_raw.mean(dim=-1)
            proposal6 = actions[:n, :6].detach()
            target6 = supervised_target[:n, :6].detach()
            applied6 = proposal6 * conf_raw
            positive_gain = restore_gain > 0.0
            print(
                "[FrontRES gate debug] "
                f"pos({_stats(conf_raw[:, :3])}) "
                f"rpy({_stats(conf_raw[:, 3:6])}) "
                f"gate_gain_pos={_masked_mean(gate_mean, positive_gain):.3f} "
                f"gate_gain_neg={_masked_mean(gate_mean, ~positive_gain):.3f}",
                flush=True,
            )
            print(
                "[FrontRES gate debug] "
                f"corr(gate,damage)={_corr(gate_mean, noisy_err_norm):+.3f} "
                f"corr(gate,gain)={_corr(gate_mean, restore_gain):+.3f} "
                f"corr(gate,|proposal|)={_corr(gate_mean, proposal6.norm(dim=-1)):+.3f} "
                f"|target|={_vec6(target6.abs())} "
                f"|proposal|={_vec6(proposal6.abs())} "
                f"|applied|={_vec6(applied6.abs())}",
                flush=True,
            )
            if inertial_debug is not None:
                print(
                    "[FrontRES inertial debug] "
                    f"D noisy/proj/cand/clean="
                    f"{float(inertial_debug['d_noisy'].mean()):.4f}/"
                    f"{float(inertial_debug['d_proj'].mean()):.4f}/"
                    f"{float(inertial_debug['d_cand'].mean()):.4f}/"
                    f"{float(inertial_debug['d_clean'].mean()):.4f} "
                    f"C noisy/proj/cand/clean="
                    f"{float(inertial_debug['c_noisy'].mean()):+.3f}/"
                    f"{float(inertial_debug['c_proj'].mean()):+.3f}/"
                    f"{float(inertial_debug['c_cand'].mean()):+.3f}/"
                    f"{float(inertial_debug['c_clean'].mean()):+.3f} "
                    f"Crp noisy/proj/cand/clean="
                    f"{float(inertial_debug['crp_noisy'].mean()):+.3f}/"
                    f"{float(inertial_debug['crp_proj'].mean()):+.3f}/"
                    f"{float(inertial_debug['crp_cand'].mean()):+.3f}/"
                    f"{float(inertial_debug['crp_clean'].mean()):+.3f}",
                    flush=True,
                )
                print(
                    "[FrontRES inertial debug] mixed "
                    f"D p0r1/pProjr1="
                    f"{float(inertial_debug['d_p0r1'].mean()):.4f}/"
                    f"{float(inertial_debug['d_ppr1'].mean()):.4f} "
                    f"C p0r1/pProjr1="
                    f"{float(inertial_debug['c_p0r1'].mean()):+.3f}/"
                    f"{float(inertial_debug['c_ppr1'].mean()):+.3f} "
                    f"Crp p0r1/pProjr1="
                    f"{float(inertial_debug['crp_p0r1'].mean()):+.3f}/"
                    f"{float(inertial_debug['crp_ppr1'].mean()):+.3f}",
                    flush=True,
                )
                print(
                    "[FrontRES inertial debug] "
                    f"angle_inv={float(inertial_debug['angle_inv'].mean()):.3f} "
                    f"anti_proj={float(inertial_debug['anti_inertia'].mean()):.3f} "
                    f"anti_cand={float(inertial_debug['anti_cand'].mean()):.3f} "
                    f"anti_clean={float(inertial_debug['anti_clean'].mean()):.3f} "
                    f"corr(gate,inert_gain)={_corr(gate_mean, inertial_debug['inertial_gain']):+.3f} "
                    f"sample_d={float(inertial_debug['d_noisy'][sample_idx]):.4f}->"
                    f"{float(inertial_debug['d_proj'][sample_idx]):.4f} "
                    f"sample_c={float(inertial_debug['c_noisy'][sample_idx]):+.3f}->"
                    f"{float(inertial_debug['c_proj'][sample_idx]):+.3f}",
                    flush=True,
                )
        print(
            "[FrontRES restore debug] sample vectors "
            f"clean_from_raw={_vec(clean_from_raw, sample_idx)} "
            f"target={_vec(target, sample_idx)} "
            f"pred={_vec(pred, sample_idx)} "
            f"applied={_vec(applied, sample_idx)} "
            f"written={_vec(written, sample_idx)} "
            f"residual={_vec(corrected_err, sample_idx)}",
            flush=True,
        )

    def _move_normalizer_to_device(self, device):
        if hasattr(self, 'obs_normalizer') and self.obs_normalizer is not None:
            for param in self.obs_normalizer.parameters():
                param.data = param.data.to(device)
            if hasattr(self.obs_normalizer, '_mean') and self.obs_normalizer._mean is not None:
                self.obs_normalizer._mean = self.obs_normalizer._mean.to(device)
            if hasattr(self.obs_normalizer, '_std') and self.obs_normalizer._std is not None:
                self.obs_normalizer._std = self.obs_normalizer._std.to(device)

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()
            # Teacher normalizer should remain frozen for MOSAIC
            if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
                if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                    self.teacher_obs_normalizer.eval()  # Keep frozen

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()
            # Teacher normalizer should remain frozen for MOSAIC
            if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
                if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                    self.teacher_obs_normalizer.eval()  # Keep frozen

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,}  # total number of processes

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
