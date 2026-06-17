# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation, MOSAIC, FrontRESUnified
from rsl_rl.runners.frontres_alpha_rho_bridge import FrontRESAlphaRhoBridge
from rsl_rl.runners.frontres_action_cone import FrontRESActionCone
from rsl_rl.runners.frontres_checkpointing import (
    load_runner,
    record_frontres_checkpoint_probe,
    save_runner,
)
from rsl_rl.runners.frontres_episode_bookkeeping import update_episode_bookkeeping
from rsl_rl.runners.frontres_executable_floor import resolve_runner_executable_floor
from rsl_rl.runners.frontres_executability import FrontRESExecutabilityScorer
from rsl_rl.runners.frontres_dr_sweep_eval import evaluate_frontres_dr_sweep as run_frontres_dr_sweep_eval
from rsl_rl.runners.frontres_metrics import frontres_boundary_stats
from rsl_rl.runners.frontres_rollout_step import prepare_frontres_rollout_step
from rsl_rl.runners.frontres_hsl_rollout_target import build_frontres_hsl_rollout_target
from rsl_rl.runners.frontres_runtime import (
    apply_frontres_task_corrections,
    apply_obs_normalizer,
    frontres_invalidate_temporal_reference_cache,
    get_inference_policy_runner,
    mask_frontres_task_actions,
    maybe_print_frontres_restore_debug,
)
from rsl_rl.runners.frontres_reward_window import (
    build_frontres_reward_context,
)
from rsl_rl.runners.frontres_reward_diagnostics import (
    initialize_frontres_reward_diagnostic_sums,
    materialize_frontres_reward_diagnostic_means,
)
from rsl_rl.runners.frontres_post_step_connector import apply_frontres_post_step_reward_connector
from rsl_rl.runners.frontres_runner_logging import log_runner
from rsl_rl.runners.frontres_transition_payload import (
    build_and_write_frontres_acceptance_payload,
    write_frontres_actor_gate,
    write_frontres_state_alpha_payload,
)
from rsl_rl.runners.frontres_training_setup import (
    apply_frontres_dr_scale,
    apply_frontres_debug_training_overrides,
    apply_frontres_iteration_dr_controller,
    build_frontres_task_action_mask,
    configure_frontres_pair_layout,
    frontres_curriculum_allowed_bases,
    frontres_ppo_actor_weight_for_iter,
    frontres_warmup_perturbation_mode_groups,
    initialize_frontres_dr_setup,
    maybe_print_frontres_perturbation_curriculum,
    resolve_frontres_mode_state,
    set_frontres_curriculum_modes,
    set_frontres_perturbation_curriculum,
    update_frontres_supervised_controller,
)
from rsl_rl.runners.frontres_warmup import (
    resolve_frontres_warmup_iterations,
    run_frontres_joint_warmup,
)
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
        self._frontres_alpha_rho_bridge = FrontRESAlphaRhoBridge()
        self._frontres_action_cone = FrontRESActionCone(self.cfg, self.alg)
        self._frontres_executability = FrontRESExecutabilityScorer(self.cfg, self.alg, self.device)

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

    def _frontres_executable_floor_values(self) -> tuple[float, float, str]:
        """Return the unified executable floor used by alpha, rho, and diagnostics.

        GMT frontier search finds the boundary in DR-strength space.  This helper
        exposes the corresponding score-space floor.  Until both safe and broken
        GMT score evidence are available, it deliberately falls back to the fixed
        historical threshold for resume stability.
        """
        return resolve_runner_executable_floor(self)

    def _frontres_structured_joint_effective_enabled(self) -> bool:
        """Mirror the algorithm-side structured-rho enable gate in the runner."""
        enabled = getattr(getattr(self, "alg", None), "_structured_joint_rl_enabled", None)
        if callable(enabled):
            return bool(enabled())
        return (
            bool(self.cfg.get("frontres_structured_joint_rl_enabled", False))
            and float(self.cfg.get("frontres_structured_joint_rl_weight", 0.0)) > 0.0
            and str(self.cfg.get("frontres_training_objective", "")).lower() == "hsl_hybrid"
        )

    def evaluate_frontres_dr_sweep(
        self,
        *,
        dr_scales: list[float],
        num_iterations_per_scale: int,
        output_path: str,
        init_at_random_ep_len: bool = True,
    ) -> list[dict]:
        return run_frontres_dr_sweep_eval(
            self,
            dr_scales=dr_scales,
            num_iterations_per_scale=num_iterations_per_scale,
            output_path=output_path,
            init_at_random_ep_len=init_at_random_ep_len,
        )

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
        obs_raw_for_gmt = None
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

        _frontres_mode = resolve_frontres_mode_state(self, FrontRESActorCritic)
        _is_frontres = _frontres_mode.is_frontres
        _frontres_training_objective = _frontres_mode.training_objective
        _frontres_supervised_restore = _frontres_mode.supervised_restore
        _frontres_hsl_restore = _frontres_mode.hsl_restore

        apply_frontres_debug_training_overrides(self, is_frontres=_is_frontres)
        _frontres_mode = resolve_frontres_mode_state(self, FrontRESActorCritic)
        _frontres_training_objective = _frontres_mode.training_objective
        _frontres_supervised_restore = _frontres_mode.supervised_restore
        _frontres_hsl_restore = _frontres_mode.hsl_restore
        _is_task_space_mode = _frontres_mode.is_task_space_mode
        critic_warmup_iters = _frontres_mode.critic_warmup_iters

        print("[Runner] Checking FrontRES mode...", flush=True)
        print(
            f"[Runner] FrontRES mode: is_frontres={_is_frontres}, "
            f"task_space={_is_task_space_mode}",
            flush=True,
        )
        if _is_frontres and critic_warmup_iters > 0:
            print(f"[Runner] Critic warmup (fixed DR scale, Actor may be held): {critic_warmup_iters} iters")

        def _frontres_ppo_actor_weight_for_iter(iteration: int) -> float:
            return frontres_ppo_actor_weight_for_iter(
                self,
                iteration=iteration,
                is_frontres=_is_frontres,
                supervised_restore=_frontres_supervised_restore,
            )

        _frontres_pair_layout = configure_frontres_pair_layout(self, is_frontres=_is_frontres)
        _use_quartet_reward = _frontres_pair_layout.use_quartet_reward
        N_train = _frontres_pair_layout.n_train
        N_candidate = _frontres_pair_layout.n_candidate
        N_base = _frontres_pair_layout.n_base
        N_clean = _frontres_pair_layout.n_clean
        cur_reward_sum_gmt = _frontres_pair_layout.cur_reward_sum_gmt

        _frontres_task_action_mask = build_frontres_task_action_mask(
            self,
            is_task_space_mode=_is_task_space_mode,
        )

        def _mask_frontres_task_actions(_actions: torch.Tensor) -> torch.Tensor:
            return self._mask_frontres_task_actions(_actions)

        _frontres_dr_setup = initialize_frontres_dr_setup(self, is_frontres=_is_frontres)
        _dr_max = _frontres_dr_setup.dr_max
        _dr_min = _frontres_dr_setup.dr_min
        _dr_ema_alpha = _frontres_dr_setup.dr_ema_alpha
        _dr_scale_init = _frontres_dr_setup.dr_scale_init
        _dr_scale = _frontres_dr_setup.dr_scale
        _r_delta_ema = _frontres_dr_setup.r_delta_ema
        _perturb_target = _frontres_dr_setup.perturb_target

        maybe_print_frontres_perturbation_curriculum(self, is_frontres=_is_frontres)

        # FrontRES joint warmup runs before PPO and owns its own rollout/loss diagnostics.
        _warmup_decision = resolve_frontres_warmup_iterations(
            configured_iterations=int(self.cfg.get("supervised_warmup_iterations", 0)),
            start_iter=start_iter,
            warmup_complete=bool(getattr(self, "_frontres_warmup_complete", False)),
        )
        _warmup_iters = _warmup_decision.iterations
        if _warmup_decision.skip_message is not None:
            print(_warmup_decision.skip_message, flush=True)
        run_frontres_joint_warmup(
            self,
            is_frontres=_is_frontres,
            warmup_iters=_warmup_iters,
            dr_scale_init=_dr_scale_init,
            dr_scale=_dr_scale,
            n_train=N_train,
            n_base=N_base,
            n_clean=N_clean,
            perturb_target=_perturb_target,
            curriculum_allowed_bases=frontres_curriculum_allowed_bases,
            set_perturbation_curriculum=set_frontres_perturbation_curriculum,
            set_curriculum_modes=set_frontres_curriculum_modes,
            warmup_perturbation_mode_groups=frontres_warmup_perturbation_mode_groups,
            apply_dr_scale=apply_frontres_dr_scale,
        )
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

            _dr_iter_plan = apply_frontres_iteration_dr_controller(
                self,
                iteration=it,
                is_frontres=_is_frontres,
                frontres_hsl_restore=_frontres_hsl_restore,
                perturb_target=_perturb_target,
                critic_warmup_iters=critic_warmup_iters,
                ppo_actor_weight_current=_ppo_actor_weight_current,
                dr_scale=_dr_scale,
                dr_scale_init=_dr_scale_init,
                dr_min=_dr_min,
                dr_max=_dr_max,
                dr_ema_alpha=_dr_ema_alpha,
                r_delta_ema=_r_delta_ema,
                n_train=N_train,
                n_candidate=N_candidate,
                n_base=N_base,
                n_clean=N_clean,
                lenbuffer_gmt_base=lenbuffer_gmt_base,
                lenbuffer_gmt_base_frontier=lenbuffer_gmt_base_frontier,
                frontres_policy_cls=FrontRESActorCritic,
            )
            _dr_scale = _dr_iter_plan.dr_scale
            _r_delta_ema = _dr_iter_plan.r_delta_ema
            _effective_dr_scale = _dr_iter_plan.effective_dr_scale
            _frontres_dr_mix_mode = _dr_iter_plan.dr_mix_mode
            _mix_diag = _dr_iter_plan.mix_diag
            _critic_warmup = _dr_iter_plan.critic_warmup
            _actor_takeover_active = _dr_iter_plan.actor_takeover_active
            _hsl_boundary_available = _dr_iter_plan.hsl_boundary_available
            _use_boundary = _dr_iter_plan.use_boundary
            _gmt_frontier_score = _dr_iter_plan.gmt_frontier_score
            _gmt_frontier_decision = _dr_iter_plan.gmt_frontier_decision

            # FrontRES reward-shaping state: reset at the start of each rollout.
            _frontres_prev_delta_q: torch.Tensor | None = None  # [N, A] residual action for smoothness penalty
            # Accumulators for wandb logging (per iteration, divided by shaping steps)
            _frontres_diag_sums = initialize_frontres_reward_diagnostic_sums()
            # Termination tracking for training envs (used to compute survival_rate this rollout)
            _frontres_term_count: int = 0
            _frontres_step_count: int = 0
            # reg_penalty activates once dr_scale ≥ 1.0 (base values fully applied).
            # Before that, reg pushing corrections→0 reinforces the no-op shortcut trap.
            _lambda_reg = getattr(self.alg, 'lambda_reg_current', 0.0) if _is_frontres else 0.0
            _dr_done    = _is_frontres and (_dr_scale >= 1.0)

            # Rollout: 训练首先需要积攒数据, 等数据攒够才能调用self.alg.update()更新权重
            with torch.inference_mode(): # 关闭计算图的梯度追踪, 只进行推理
                for _rollout_step in range(self.num_steps_per_env):
                    step_plan = prepare_frontres_rollout_step(
                        self,
                        obs=obs,
                        privileged_obs=privileged_obs,
                        teacher_obs=teacher_obs,
                        ref_vel_estimator_obs=ref_vel_estimator_obs,
                        obs_raw_for_gmt=obs_raw_for_gmt,
                        vel_est_error_buffer=vel_est_error_buffer,
                        iteration=it,
                        rollout_step=_rollout_step,
                        is_frontres=_is_frontres,
                        is_task_space_mode=_is_task_space_mode,
                        n_train=N_train,
                        n_candidate=N_candidate,
                        n_base=N_base,
                        n_clean=N_clean,
                    )
                    actions = step_plan.actions
                    env_actions = step_plan.env_actions
                    _hsl_pos_snapshot = step_plan.hsl_pos_snapshot
                    _hsl_quat_snapshot = step_plan.hsl_quat_snapshot

                    # Step the environment 仿真环境更新观测量/动作评分/序列结束与否/监控数据
                    # NOTE: This is where the environment computes the *next* observation internally.
                    # The result is returned here and then used in the next loop iteration.
                    obs, rewards, dones, infos = self.env.step(env_actions.to(self.env.device))

                    # Move to device
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    if _is_frontres and _is_task_space_mode:
                        frontres_invalidate_temporal_reference_cache(self, dones)

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
                            build_frontres_hsl_rollout_target(
                                self,
                                command=_cmd_hsl,
                                actions=actions,
                                dones=dones,
                                current_pos_correction=_hsl_pos_snapshot,
                                current_quat_correction=_hsl_quat_snapshot,
                                n_train=N_train,
                                n_candidate=N_candidate,
                                n_base=N_base,
                                n_clean=N_clean,
                                quat_to_rotvec_wxyz=_quat_to_rotvec_wxyz,
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
                        _reward_window = None
                        _reward_context = build_frontres_reward_context(
                            self,
                            rewards=rewards,
                            dones=dones,
                            infos=infos,
                            actions=actions,
                            n_train=N_train,
                            n_candidate=N_candidate,
                            n_base=N_base,
                            n_clean=N_clean,
                            is_task_space_mode=_is_task_space_mode,
                            dr_scale=_dr_scale,
                            ppo_actor_weight_current=_ppo_actor_weight_current,
                            quat_to_rotvec_wxyz=_quat_to_rotvec_wxyz,
                            quat_mul_fn=quat_mul,
                            quat_inv_fn=quat_inv,
                            euler_xyz_from_quat_fn=euler_xyz_from_quat,
                            device=self.device,
                        )
                        if _reward_context is None and _is_task_space_mode:
                            raise RuntimeError(
                                "FrontRES task-space reward/evidence requires clean anchor evidence "
                                "(anchor_pos_w_original and anchor_quat_w_original)."
                            )
                        if _reward_context is not None:
                            _candidate_start = _reward_context.candidate_start
                            _candidate_end = _reward_context.candidate_end
                            _base_start = _reward_context.base_start
                            _base_end = _reward_context.base_end
                            _clean_start = _reward_context.clean_start
                            _clean_end = _reward_context.clean_end
                            _n_exec = _reward_context.n_exec
                            _n_pair = _reward_context.n_pair
                            r_raw_gmt = _reward_context.r_raw_gmt
                            r_clean_gmt = _reward_context.r_clean_gmt
                            r_candidate_gmt = _reward_context.r_candidate_gmt
                            r_total = _reward_context.r_total
                            _mcmd_rdelta = _reward_context.cmd
                            _use_clean = _reward_context.use_clean
                            _a_w = _reward_context.a_w
                            _a_raw = _reward_context.a_raw
                            _a_fr = _reward_context.a_fr
                            _q_w = _reward_context.q_w
                            _q_raw = _reward_context.q_raw
                            _q_fr = _reward_context.q_fr
                            _r_step = _reward_context.r_step
                            _r_rescue = _reward_context.r_rescue
                            _r_exec = _reward_context.r_exec
                            _dr_z_abs_log = _reward_context.dr_z_abs_log
                            _dr_xy_abs_log = _reward_context.dr_xy_abs_log
                            _dr_rp_abs_log = _reward_context.dr_rp_abs_log
                            _dr_yaw_abs_log = _reward_context.dr_yaw_abs_log
                            _corr_z_abs_log = _reward_context.corr_z_abs_log
                            _corr_xy_abs_log = _reward_context.corr_xy_abs_log
                            _corr_rp_abs_log = _reward_context.corr_rp_abs_log
                            _corr_yaw_abs_log = _reward_context.corr_yaw_abs_log
                            _rot_raw_to_clean = _reward_context.rot_raw_to_clean
                            _rot_raw_to_fr = _reward_context.rot_raw_to_fr
                            _e_raw = _reward_context.e_raw
                            _e_fr = _reward_context.e_fr
                            _exec_score_all = _reward_context.exec_score_all
                            _exec_components = _reward_context.exec_components
                            _feasible_components = _reward_context.feasible_components
                            _mode_groups = _reward_context.mode_groups
                            _exec_frontres = _reward_context.exec_frontres
                            _exec_candidate = _reward_context.exec_candidate
                            _exec_perturbed = _reward_context.exec_perturbed
                            _exec_clean = _reward_context.exec_clean
                            _exec_feasible = _reward_context.exec_feasible
                            _exec_planar_log = _reward_context.exec_planar_log
                            _exec_vertical_log = _reward_context.exec_vertical_log
                            _exec_task_log = _reward_context.exec_task_log
                            _intervention_cost = _reward_context.intervention_cost
                            _clean_bound_cost = _reward_context.clean_bound_cost
                            _side_cost = _reward_context.side_cost
                            _over_cost = _reward_context.over_cost
                            _overcorrection_cost = _reward_context.overcorrection_cost
                            _under_repair_penalty = _reward_context.under_repair_penalty
                            _action_activity = _reward_context.action_activity
                            _w_exec = _reward_context.w_exec
                            _repair_scale = _reward_context.repair_scale
                            _w_geom = _reward_context.w_geom
                            _w_rescue = _reward_context.w_rescue
                            _w_exec_harm = _reward_context.w_exec_harm
                            _repair_gain = _reward_context.repair_gain
                            _candidate_gain = _reward_context.candidate_gain
                            _projection_gain = _reward_context.projection_gain
                            _oracle_ub_gain = _reward_context.oracle_ub_gain
                            _oracle_ub_pass = _reward_context.oracle_ub_pass
                            _oracle_ub_noisy_win = _reward_context.oracle_ub_noisy_win
                            _oracle_ub_projected_win = _reward_context.oracle_ub_projected_win
                            _oracle_ub_candidate_win = _reward_context.oracle_ub_candidate_win
                            _oracle_ub_feasible_win = _reward_context.oracle_ub_feasible_win
                            _exec_floor = _reward_context.exec_floor
                            _exec_safe_floor = _reward_context.exec_safe_floor
                            _exec_floor_source = _reward_context.exec_floor_source
                            _candidate_floor_margin = _reward_context.candidate_floor_margin
                            _candidate_floor_pass = _reward_context.candidate_floor_pass
                            _candidate_floor_pass_frac = _reward_context.candidate_floor_pass_frac
                            _stable_route_next = _reward_context.stable_route_next
                            _stable_route_active = _reward_context.stable_route_active
                            _reward_window = _reward_context.reward_window
                            _damage_gap = _reward_window.damage_gap
                            _oracle_clean_gap = _reward_window.oracle_clean_gap
                            _oracle_trust = _reward_window.oracle_trust
                            _repair_ratio = _reward_window.repair_ratio
                            _safe_gate = _reward_window.safe_gate
                            _repair_gate = _reward_window.repair_gate
                            _broken_gate = _reward_window.broken_gate
                            _window_mu = _reward_window.window_mu
                            _exec_gate = _reward_window.exec_gate
                            _cost_gate = _reward_window.cost_gate
                            _safe_frac = _reward_window.safe_frac
                            _repair_frac = _reward_window.repair_frac
                            _broken_frac = _reward_window.broken_frac
                            _safe_gap = _reward_window.safe_gap
                            _broken_gap = _reward_window.broken_gap
                            _learnable_route_mask = _reward_window.learnable_route_mask
                            _exec_weight = _reward_window.exec_weight
                            _cost_weight = _reward_window.cost_weight
                            _actor_gate = _reward_window.actor_gate
                            _harm_penalty = _reward_window.harm_penalty
                            _harm_penalty_exec = _reward_window.harm_penalty_exec
                            _harm_mag = _reward_window.harm_mag
                            _cost_exec = _reward_window.cost_exec
                            _effective_gain_bonus = _reward_window.effective_gain_bonus
                            _effective_gain_bonus_exec = _reward_window.effective_gain_bonus_exec
                            _under_repair_penalty = _reward_window.under_repair_penalty
                            _reward_progress = _reward_window.reward_progress
                            _constraint_progress = _reward_window.constraint_progress
                            _frontres_actor_gate = write_frontres_actor_gate(
                                self,
                                n_exec=_n_exec,
                                actor_gate=_actor_gate,
                            )
                            _state_alpha_target, _state_alpha_mask = write_frontres_state_alpha_payload(
                                self,
                                n_exec=_n_exec,
                                exec_perturbed=_exec_perturbed,
                                dones=dones,
                                infos=infos,
                                base_start=_base_start,
                            )
                            _accept_payload = build_and_write_frontres_acceptance_payload(
                                self,
                                actions=actions,
                                n_candidate=N_candidate,
                                n_exec=_n_exec,
                                candidate_start=_candidate_start,
                                base_start=_base_start,
                                a_w=_a_w,
                                a_raw=_a_raw,
                                a_fr=_a_fr,
                                q_w=_q_w,
                                q_raw=_q_raw,
                                q_fr=_q_fr,
                                cmd_for_inertia=_mcmd_rdelta,
                                repair_gain=_repair_gain,
                                candidate_gain=_candidate_gain,
                                repair_gate=_repair_gate,
                                oracle_trust=_oracle_trust,
                                learnable_route_mask=_learnable_route_mask,
                                exec_components=_exec_components,
                                feasible_components=_feasible_components,
                                exec_perturbed=_exec_perturbed,
                                exec_feasible=_exec_feasible,
                                exec_frontres=_exec_frontres,
                                exec_candidate=_exec_candidate,
                                actor_gate=_actor_gate,
                                state_alpha_target=_state_alpha_target,
                                state_alpha_mask=_state_alpha_mask,
                                mode_groups=_mode_groups,
                                quat_to_rotvec_wxyz=_quat_to_rotvec_wxyz,
                                quat_mul_fn=quat_mul,
                                quat_inv_fn=quat_inv,
                            )
                            _accept_pref_target = _accept_payload.accept_target
                            _accept_pref_mask = _accept_payload.accept_mask
                            _pref_full_frac = _accept_payload.pref_full_frac
                            _pref_noop_frac = _accept_payload.pref_noop_frac
                            _pref_keep_frac = _accept_payload.pref_keep_frac
                            _pref_ignore_frac = _accept_payload.pref_ignore_frac
                            _pref_margin_mean = _accept_payload.pref_margin_mean
                            _pref_need_mean = _accept_payload.pref_need_mean
                            _pref_admiss_mean = _accept_payload.pref_admiss_mean
                            _pref_target_mean = _accept_payload.pref_target_mean
                            _tri_weight_repair_mean = _accept_payload.tri_weight_repair_mean
                            _tri_weight_noisy_mean = _accept_payload.tri_weight_noisy_mean
                            _tri_weight_stable_mean = _accept_payload.tri_weight_stable_mean
                            _pref_inertial_penalty_rho_mean = _accept_payload.pref_inertial_penalty_rho_mean
                            _pref_inertial_penalty_one_mean = _accept_payload.pref_inertial_penalty_one_mean
                            _rho_target_planar_mean = _accept_payload.rho_target_planar_mean
                            _rho_target_rp_mean = _accept_payload.rho_target_rp_mean
                            _rho_target_z_mean = _accept_payload.rho_target_z_mean
                            _rho_target_spread_mean = _accept_payload.rho_target_spread_mean
                            _grouped_rho_mask_mean = _accept_payload.grouped_rho_mask_mean
                            _rho_regret_up_planar_mean = _accept_payload.rho_regret_up_planar_mean
                            _rho_regret_up_rp_mean = _accept_payload.rho_regret_up_rp_mean
                            _rho_regret_up_z_mean = _accept_payload.rho_regret_up_z_mean
                            _rho_regret_down_planar_mean = _accept_payload.rho_regret_down_planar_mean
                            _rho_regret_down_rp_mean = _accept_payload.rho_regret_down_rp_mean
                            _rho_regret_down_z_mean = _accept_payload.rho_regret_down_z_mean
                        _post_step = apply_frontres_post_step_reward_connector(
                            self,
                            locs=locals(),
                            rewards=rewards,
                            dones=dones,
                            actions=actions,
                            reward_window=_reward_window if _is_task_space_mode else None,
                            diagnostic_sums=_frontres_diag_sums,
                            prev_delta_q=_frontres_prev_delta_q,
                            term_count=_frontres_term_count,
                            step_count=_frontres_step_count,
                        )
                        rewards = _post_step.rewards
                        _reward_window = _post_step.reward_window
                        r_raw_gmt = _post_step.r_raw_gmt
                        r_candidate_gmt = _post_step.r_candidate_gmt
                        r_clean_gmt = _post_step.r_clean_gmt
                        _frontres_prev_delta_q = _post_step.prev_delta_q
                        _frontres_term_count = _post_step.term_count
                        _frontres_step_count = _post_step.step_count
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

                    update_episode_bookkeeping(
                        self,
                        infos=infos,
                        rewards=rewards,
                        dones=dones,
                        intrinsic_rewards=intrinsic_rewards,
                        ep_infos=ep_infos,
                        rewbuffer=rewbuffer,
                        lenbuffer=lenbuffer,
                        rewbuffer_gmt=rewbuffer_gmt,
                        lenbuffer_gmt=lenbuffer_gmt,
                        lenbuffer_gmt_base=lenbuffer_gmt_base,
                        lenbuffer_gmt_base_frontier=lenbuffer_gmt_base_frontier,
                        erewbuffer=erewbuffer if hasattr(self.alg, "rnd") and self.alg.rnd else None,
                        irewbuffer=irewbuffer if hasattr(self.alg, "rnd") and self.alg.rnd else None,
                        cur_reward_sum=cur_reward_sum,
                        cur_episode_length=cur_episode_length,
                        cur_reward_sum_gmt=cur_reward_sum_gmt,
                        cur_ereward_sum=cur_ereward_sum if hasattr(self.alg, "rnd") and self.alg.rnd else None,
                        cur_ireward_sum=cur_ireward_sum if hasattr(self.alg, "rnd") and self.alg.rnd else None,
                        is_frontres=_is_frontres,
                        n_train=N_train,
                        n_candidate=N_candidate,
                        n_base=N_base,
                        n_clean=N_clean,
                        r_candidate_gmt=r_candidate_gmt if _is_frontres else None,
                        r_raw_gmt=r_raw_gmt if _is_frontres else None,
                        r_clean_gmt=r_clean_gmt if _is_frontres else None,
                    )

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

            _frontres_diag_means = materialize_frontres_reward_diagnostic_means(
                _frontres_diag_sums,
                is_frontres=_is_frontres,
                is_task_space_mode=_is_task_space_mode,
                term_count=_frontres_term_count,
                step_count=_frontres_step_count,
            )
            frontres_rdelta_mean = _frontres_diag_means["frontres_rdelta_mean"]
            frontres_positive_gain_frac_mean = _frontres_diag_means["frontres_positive_gain_frac_mean"]
            frontres_harm_rate_mean = _frontres_diag_means["frontres_harm_rate_mean"]
            _frontres_stats_locs = locals().copy()
            _frontres_stats_locs.update(_frontres_diag_means)

            # Store r_delta mean for next iteration's PI controller update.
            if frontres_rdelta_mean is not None:
                self._last_r_delta_mean = frontres_rdelta_mean
            _frontres_boundary_stats = frontres_boundary_stats(_frontres_stats_locs)
            if _frontres_boundary_stats is not None:
                self._last_frontres_boundary_stats = _frontres_boundary_stats

            if _is_frontres:
                if not _frontres_supervised_restore:
                    update_frontres_supervised_controller(
                        self,
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
            _frontres_log_locs = locals().copy()
            _frontres_log_locs.update(_frontres_diag_means)

            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(_frontres_log_locs)

                # Save model
                if it % self.save_interval == 0:
                    _checkpoint_path = os.path.join(self.log_dir, f"model_{it}.pt")
                    self.save(_checkpoint_path)
                    self._record_frontres_checkpoint_probe(_frontres_log_locs, _checkpoint_path)

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
            self._record_frontres_checkpoint_probe(
                _frontres_log_locs if "_frontres_log_locs" in locals() else locals(),
                _final_checkpoint_path,
            )

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        return log_runner(self, locs, width=width, pad=pad)

    def _record_frontres_checkpoint_probe(self, locs: dict, checkpoint_path: str) -> None:
        return record_frontres_checkpoint_probe(self, locs, checkpoint_path)

    def save(self, path: str, infos=None):
        return save_runner(self, path, infos=infos)

    def load(self, path: str, load_optimizer: bool = True, load_critic: bool = True):
        return load_runner(self, path, load_optimizer=load_optimizer, load_critic=load_critic)

    def get_inference_policy(self, device=None):
        return get_inference_policy_runner(self, device=device)

    def _apply_obs_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
        return apply_obs_normalizer(self, obs)

    def _mask_frontres_task_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return mask_frontres_task_actions(self, actions)

    def _apply_frontres_task_corrections(
        self,
        task_corr: torch.Tensor | None,
        n_train: int | None = None,
        *,
        allow_oracle: bool = False,
        n_candidate: int = 0,
    ) -> torch.Tensor | None:
        return apply_frontres_task_corrections(
            self,
            task_corr,
            n_train=n_train,
            allow_oracle=allow_oracle,
            n_candidate=n_candidate,
        )

    def _maybe_print_frontres_restore_debug(
        self,
        *,
        it: int,
        rollout_step: int,
        actions: torch.Tensor | None,
        supervised_target: torch.Tensor | None,
        n_train: int,
    ) -> None:
        return maybe_print_frontres_restore_debug(
            self,
            it=it,
            rollout_step=rollout_step,
            actions=actions,
            supervised_target=supervised_target,
            n_train=n_train,
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
