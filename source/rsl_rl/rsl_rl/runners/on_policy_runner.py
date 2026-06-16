# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
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
from rsl_rl.runners.frontres_executable_floor import (
    ExecutableFloorState,
    resolve_executable_floor,
    update_executable_floor_stats,
)
from rsl_rl.runners.frontres_executability import FrontRESExecutabilityScorer
from rsl_rl.runners.frontres_dr_sweep_eval import evaluate_frontres_dr_sweep as run_frontres_dr_sweep_eval
from rsl_rl.runners.frontres_oracle import compute_frontres_oracle_upper_bound
from rsl_rl.runners.frontres_metrics import frontres_boundary_stats
from rsl_rl.runners.frontres_rollout_evidence import compute_frontres_rollout_evidence
from rsl_rl.runners.frontres_rollout_step import prepare_frontres_rollout_step
from rsl_rl.runners.frontres_hsl_rollout_target import build_frontres_hsl_rollout_target
from rsl_rl.runners.frontres_runtime import (
    apply_frontres_task_corrections,
    apply_obs_normalizer,
    frontres_invalidate_temporal_reference_cache,
    frontres_raw_anchor_pose,
    frontres_stabilizing_candidate_correction,
    frontres_temporal_continuity_correction,
    frontres_update_temporal_reference_cache,
    get_inference_policy_runner,
    mask_frontres_task_actions,
    maybe_print_frontres_restore_debug,
)
from rsl_rl.runners.frontres_reward_window import (
    build_frontres_reward_window,
)
from rsl_rl.runners.frontres_reward_diagnostics import (
    initialize_frontres_reward_diagnostic_sums,
    materialize_frontres_reward_diagnostic_means,
)
from rsl_rl.runners.frontres_post_step_connector import apply_frontres_post_step_reward_connector
from rsl_rl.runners.frontres_runner_logging import log_runner
from rsl_rl.runners.frontres_transition_payload import (
    apply_frontres_structured_rho_payload,
    build_frontres_non_tri_acceptance_target_payload,
    build_frontres_tri_anchor_rho_payload,
    initialize_frontres_acceptance_payload,
    summarize_frontres_acceptance_payload,
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

    def _frontres_exec_score(self, command, return_components: bool = False):
        return self._frontres_executability.exec_score(command, return_components=return_components)

    def _frontres_feasible_oracle_exec_score(
        self,
        command,
        start: int,
        count: int,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._frontres_executability.feasible_oracle_exec_score(
            command,
            start,
            count,
            return_components=return_components,
        )

    def _frontres_exec_score_for_modes(
        self,
        components: dict[str, torch.Tensor],
        start: int,
        count: int,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...] | None = None,
    ) -> torch.Tensor:
        return self._frontres_executability.exec_score_for_modes(
            components,
            start,
            count,
            mode_groups=mode_groups,
            active_modes=tuple(getattr(self, "_frontres_curriculum_active_modes", ())),
        )

    def _frontres_executable_floor_values(self) -> tuple[float, float, str]:
        """Return the unified executable floor used by alpha, rho, and diagnostics.

        GMT frontier search finds the boundary in DR-strength space.  This helper
        exposes the corresponding score-space floor.  Until both safe and broken
        GMT score evidence are available, it deliberately falls back to the fixed
        historical threshold for resume stability.
        """
        values = resolve_executable_floor(
            self.cfg,
            ExecutableFloorState(
                safe_score_ema=getattr(self, "_frontres_exec_floor_safe_score_ema", None),
                broken_score_ema=getattr(self, "_frontres_exec_floor_broken_score_ema", None),
                safe_count=float(getattr(self, "_frontres_exec_floor_safe_count", 0.0)),
                broken_count=float(getattr(self, "_frontres_exec_floor_broken_count", 0.0)),
            ),
        )
        self._frontres_exec_floor_value_last = values.floor
        self._frontres_exec_floor_safe_last = values.safe_floor
        self._frontres_exec_floor_source_last = values.source
        self._frontres_exec_floor_adaptive_last = values.adaptive
        self._frontres_exec_floor_safe_count_last = values.safe_count
        self._frontres_exec_floor_broken_count_last = values.broken_count
        return values.floor, values.safe_floor, values.source

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

    def _frontres_update_executable_floor_stats(
        self,
        exec_score: torch.Tensor,
        done: torch.Tensor | None = None,
        timeout: torch.Tensor | None = None,
        mix_class: torch.Tensor | None = None,
    ) -> tuple[float, float, str]:
        """Update GMT-score floor statistics and return the active floor."""
        state, values = update_executable_floor_stats(
            self.cfg,
            ExecutableFloorState(
                safe_score_ema=getattr(self, "_frontres_exec_floor_safe_score_ema", None),
                broken_score_ema=getattr(self, "_frontres_exec_floor_broken_score_ema", None),
                safe_count=float(getattr(self, "_frontres_exec_floor_safe_count", 0.0)),
                broken_count=float(getattr(self, "_frontres_exec_floor_broken_count", 0.0)),
            ),
            exec_score,
            done=done,
            timeout=timeout,
            mix_class=mix_class,
            frontier_decision=getattr(self, "_frontres_gmt_frontier_decision", ""),
        )
        for name, value in (
            ("safe_score_ema", state.safe_score_ema),
            ("broken_score_ema", state.broken_score_ema),
            ("safe_count", state.safe_count),
            ("broken_count", state.broken_count),
        ):
            attr = f"_frontres_exec_floor_{name}"
            if value is None:
                if hasattr(self, attr):
                    delattr(self, attr)
            else:
                setattr(self, attr, float(value))
        self._frontres_exec_floor_value_last = values.floor
        self._frontres_exec_floor_safe_last = values.safe_floor
        self._frontres_exec_floor_source_last = values.source
        self._frontres_exec_floor_adaptive_last = values.adaptive
        self._frontres_exec_floor_safe_count_last = values.safe_count
        self._frontres_exec_floor_broken_count_last = values.broken_count
        return values.floor, values.safe_floor, values.source

    def _frontres_project_task_target_to_action_cone(self, command, target: torch.Tensor) -> torch.Tensor:
        return self._frontres_action_cone.project_task_target(command, target)

    def _frontres_mode_dim_mask(
        self,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._frontres_action_cone.mode_dim_mask(mode_groups, count, device, dtype)

    def _frontres_apply_per_mode_supervised_mask(
        self,
        target: torch.Tensor,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
        count: int,
    ) -> torch.Tensor:
        return self._frontres_action_cone.apply_per_mode_supervised_mask(target, mode_groups, count)

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
        return build_frontres_hsl_rollout_target(
            self,
            command=command,
            actions=actions,
            dones=dones,
            current_pos_correction=current_pos_correction,
            current_quat_correction=current_quat_correction,
            n_train=n_train,
            n_candidate=n_candidate,
            n_base=n_base,
            n_clean=n_clean,
            quat_to_rotvec_wxyz=_quat_to_rotvec_wxyz,
        )

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
                            _rollout_evidence = compute_frontres_rollout_evidence(
                                noisy_score=_exec_perturbed,
                                projected_score=_exec_frontres,
                                candidate_score=_exec_candidate,
                            )
                            _repair_gain = _rollout_evidence.repair_gain
                            _candidate_gain = _rollout_evidence.candidate_gain
                            _projection_gain = _rollout_evidence.projection_gain
                            _oracle_ub = compute_frontres_oracle_upper_bound(
                                _exec_perturbed,
                                _exec_frontres,
                                _exec_candidate,
                                _exec_feasible,
                                margin=float(self.cfg.get("frontres_oracle_upper_bound_margin", 0.0)),
                                enabled=bool(self.cfg.get("frontres_oracle_upper_bound_diag_enabled", True)),
                            )
                            _oracle_ub_gain = _oracle_ub.gain
                            _oracle_ub_pass = _oracle_ub.pass_mask
                            _oracle_ub_noisy_win = _oracle_ub.noisy_win
                            _oracle_ub_projected_win = _oracle_ub.projected_win
                            _oracle_ub_candidate_win = _oracle_ub.candidate_win
                            _oracle_ub_feasible_win = _oracle_ub.feasible_win
                            _base_done_for_floor = dones[_base_start:_base_start + _n_exec].view(-1) > 0
                            _timeout_for_floor = infos.get("time_outs", None)
                            if _timeout_for_floor is not None:
                                _timeout_for_floor = _timeout_for_floor.to(self.device).view(-1)
                                _base_timeout_for_floor = (
                                    _timeout_for_floor[_base_start:_base_start + _n_exec] > 0
                                )
                            else:
                                _base_timeout_for_floor = torch.zeros(
                                    _n_exec, device=self.device, dtype=torch.bool
                                )
                            _mix_class_for_floor = getattr(self, "_frontres_dr_mix_class_train", None)
                            _exec_floor, _exec_safe_floor, _exec_floor_source = (
                                self._frontres_update_executable_floor_stats(
                                    _exec_perturbed,
                                    done=_base_done_for_floor,
                                    timeout=_base_timeout_for_floor,
                                    mix_class=_mix_class_for_floor,
                                )
                            )
                            _exec_floor_tensor = torch.full_like(_exec_candidate, float(_exec_floor))
                            # Diagnostic-only executable floor: Candidate is
                            # judged against the unified GMT-calibrated floor,
                            # not against Clean or Noisy.
                            _candidate_floor_margin = _exec_candidate - _exec_floor_tensor
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
                            _reward_window = build_frontres_reward_window(
                                runner=self,
                                cfg=self.cfg,
                                n_train=N_train,
                                n_exec=_n_exec,
                                exec_clean=_exec_clean,
                                exec_perturbed=_exec_perturbed,
                                exec_feasible=_exec_feasible,
                                exec_frontres=_exec_frontres,
                                repair_gain=_repair_gain,
                                mode_groups=_mode_groups,
                                e_raw=_e_raw,
                                e_fr=_e_fr,
                                intervention_cost=_intervention_cost,
                                action_activity=_action_activity,
                                under_repair_penalty=_under_repair_penalty,
                                dr_scale=_dr_scale,
                                ppo_actor_weight_current=_ppo_actor_weight_current,
                                stable_route_active_mask=_stable_route_active,
                                device=self.device,
                            )
                            _r_exec = _reward_window.r_exec
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
                            _accept_payload = initialize_frontres_acceptance_payload(self)
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
                            _structured_joint_requested = self._frontres_structured_joint_effective_enabled()
                            _legacy_pref_enabled = bool(self.cfg.get("frontres_acceptance_preference_enabled", True))
                            _pref_enabled = (
                                (_legacy_pref_enabled or _structured_joint_requested)
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
                                _regret_target_enabled = bool(
                                    self.cfg.get("frontres_acceptance_regret_target_enabled", True)
                                )
                                if _regret_target_enabled:
                                    _regret_mask_floor = float(
                                        self.cfg.get("frontres_acceptance_regret_soft_mask_floor", 1.0)
                                    )
                                    _regret_mask_floor = max(0.0, min(1.0, _regret_mask_floor))
                                    _repair_pref_gate = (
                                        _regret_mask_floor
                                        + (1.0 - _regret_mask_floor) * _repair_gate
                                    ).clamp(0.0, 1.0)
                                    _oracle_pref_floor = float(
                                        self.cfg.get("frontres_acceptance_regret_oracle_trust_floor", 0.25)
                                    )
                                    _oracle_pref_floor = max(0.0, min(1.0, _oracle_pref_floor))
                                    _oracle_pref_gate = (
                                        _oracle_pref_floor
                                        + (1.0 - _oracle_pref_floor) * _oracle_trust
                                    ).clamp(0.0, 1.0)
                                else:
                                    _repair_pref_gate = _repair_gate
                                    _oracle_pref_gate = _oracle_trust
                                _pref_gate = (
                                    _oracle_pref_gate * _repair_pref_gate * _learnable_route_mask
                                ).detach().clamp(0.0, 1.0)
                                _task_conf_dim = int(getattr(getattr(self.alg, "policy", None), "task_conf_dim", 2))
                                _tri_rho_payload = build_frontres_tri_anchor_rho_payload(
                                    cfg=self.cfg,
                                    actions=actions,
                                    n_exec=_n_exec,
                                    task_conf_dim=_task_conf_dim,
                                    j_one=_j_one,
                                    j_zero=_j_zero,
                                    pref_margin=_pref_margin,
                                    pref_gate=_pref_gate,
                                    exec_components=_exec_components,
                                    candidate_start=_candidate_start,
                                    base_start=_base_start,
                                    regret_target_enabled=_regret_target_enabled,
                                    device=self.device,
                                )
                                _target_exec = _tri_rho_payload.target_exec
                                _mask_exec = _tri_rho_payload.mask_exec
                                _rho_current = _tri_rho_payload.rho_current
                                _rho_space = _tri_rho_payload.rho_space
                                _grouped_targets_enabled = _tri_rho_payload.grouped_targets_enabled
                                _rho_direction_dim_from_regret = _tri_rho_payload.rho_direction_dim_from_regret
                                _candidate_planar = _tri_rho_payload.candidate_planar
                                _candidate_rp = _tri_rho_payload.candidate_rp
                                _candidate_z = _tri_rho_payload.candidate_z
                                _projected_planar = _tri_rho_payload.projected_planar
                                _projected_rp = _tri_rho_payload.projected_rp
                                _projected_z = _tri_rho_payload.projected_z
                                _base_planar = _tri_rho_payload.base_planar
                                _base_rp = _tri_rho_payload.base_rp
                                _base_z = _tri_rho_payload.base_z
                                _rho_target_planar_mean = _tri_rho_payload.rho_target_planar_mean
                                _rho_target_rp_mean = _tri_rho_payload.rho_target_rp_mean
                                _rho_target_z_mean = _tri_rho_payload.rho_target_z_mean
                                _rho_target_spread_mean = _tri_rho_payload.rho_target_spread_mean
                                _rho_regret_up_planar_mean = _tri_rho_payload.rho_regret_up_planar_mean
                                _rho_regret_up_rp_mean = _tri_rho_payload.rho_regret_up_rp_mean
                                _rho_regret_up_z_mean = _tri_rho_payload.rho_regret_up_z_mean
                                _rho_regret_down_planar_mean = _tri_rho_payload.rho_regret_down_planar_mean
                                _rho_regret_down_rp_mean = _tri_rho_payload.rho_regret_down_rp_mean
                                _rho_regret_down_z_mean = _tri_rho_payload.rho_regret_down_z_mean
                                _non_tri_acceptance_payload = build_frontres_non_tri_acceptance_target_payload(
                                    cfg=self.cfg,
                                    rho_space=_rho_space,
                                    target_exec=_target_exec,
                                    mask_exec=_mask_exec,
                                    n_exec=_n_exec,
                                    base_start=_base_start,
                                    candidate_start=_candidate_start,
                                    a_w=_a_w,
                                    a_raw=_a_raw,
                                    a_fr=_a_fr,
                                    q_w=_q_w,
                                    q_raw=_q_raw,
                                    q_fr=_q_fr,
                                    c_zero=_c_zero,
                                    c_one=_c_one,
                                    rho_current=_rho_current,
                                    j_one=_j_one,
                                    j_zero=_j_zero,
                                    j_rho=_j_rho,
                                    full_win=_full_win,
                                    noop_win=_noop_win,
                                    keep_win=_keep_win,
                                    pref_margin=_pref_margin,
                                    pref_gate=_pref_gate,
                                    quat_to_rotvec_wxyz=_quat_to_rotvec_wxyz,
                                    quat_mul_fn=quat_mul,
                                    quat_inv_fn=quat_inv,
                                    device=self.device,
                                )
                                _target_exec = _non_tri_acceptance_payload.target_exec
                                _mask_exec = _non_tri_acceptance_payload.mask_exec
                                _need = _non_tri_acceptance_payload.need
                                _admissibility = _non_tri_acceptance_payload.admissibility
                                if bool(self.cfg.get("frontres_per_mode_acceptance_preference_mask", True)):
                                    _mode_dim_mask = self._frontres_mode_dim_mask(
                                        _mode_groups, _n_exec, self.device, _mask_exec.dtype
                                    )
                                    if _regret_target_enabled and _grouped_targets_enabled:
                                        _mode_soft_floor = float(
                                            self.cfg.get(
                                                "frontres_acceptance_regret_per_mode_soft_floor",
                                                1.0,
                                            )
                                        )
                                        _mode_soft_floor = max(0.0, min(1.0, _mode_soft_floor))
                                        _mode_dim_mask = (
                                            _mode_soft_floor
                                            + (1.0 - _mode_soft_floor) * _mode_dim_mask
                                        ).clamp(0.0, 1.0)
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
                                _grouped_rho_mask_mean = _mask_exec.mean()
                                if _rho_space in ("tri_anchor", "tri-anchor", "tri"):
                                    _mask_sum_for_alpha = _mask_exec.sum(dim=-1)
                                    _target_mean_for_alpha = _target_exec.mean(dim=-1).detach().clamp(0.0, 1.0)
                                    _target_active_for_alpha = (
                                        (_target_exec * _mask_exec).sum(dim=-1)
                                        / _mask_sum_for_alpha.clamp(min=1e-6)
                                    ).detach().clamp(0.0, 1.0)
                                    _target_sample_for_alpha = torch.where(
                                        _mask_sum_for_alpha > 0.0,
                                        _target_active_for_alpha,
                                        _target_mean_for_alpha,
                                    )
                                    _tri_alpha_source = getattr(
                                        self, "_frontres_state_alpha_prob_next", None
                                    )
                                    if (
                                        isinstance(_tri_alpha_source, torch.Tensor)
                                        and _tri_alpha_source.numel() > 0
                                    ):
                                        _tri_alpha = _tri_alpha_source.to(
                                            device=self.device,
                                            dtype=_target_exec.dtype,
                                        ).view(-1)
                                        if _tri_alpha.numel() < _n_exec:
                                            _tri_alpha = torch.nn.functional.pad(
                                                _tri_alpha,
                                                (0, _n_exec - _tri_alpha.numel()),
                                                value=0.0,
                                            )
                                        _tri_alpha = _tri_alpha[:_n_exec].detach().clamp(0.0, 1.0)
                                    else:
                                        _tri_alpha = (
                                            _state_alpha_target[:_n_exec, 0]
                                            .detach()
                                            .clamp(0.0, 1.0)
                                        )
                                    _tri_weight_repair_mean = _target_sample_for_alpha.mean()
                                    _tri_weight_stable_mean = (
                                        (1.0 - _target_sample_for_alpha) * _tri_alpha
                                    ).mean()
                                    _tri_weight_noisy_mean = (
                                        (1.0 - _target_sample_for_alpha) * (1.0 - _tri_alpha)
                                    ).mean()
                                _accept_pref_target[:_n_exec] = _target_exec.detach()
                                _accept_pref_mask[:_n_exec] = _mask_exec.detach()
                                _structured_rho_payload = apply_frontres_structured_rho_payload(
                                    self,
                                    accept_target=_accept_pref_target,
                                    accept_mask=_accept_pref_mask,
                                    target_exec=_target_exec,
                                    mask_exec=_mask_exec,
                                    n_exec=_n_exec,
                                    rho_current=_rho_current,
                                    actor_gate=_actor_gate,
                                    exec_perturbed=_exec_perturbed,
                                    exec_feasible=_exec_feasible,
                                    exec_frontres=_exec_frontres,
                                    exec_candidate=_exec_candidate,
                                    state_alpha_target=_state_alpha_target,
                                    rho_space=_rho_space,
                                    grouped_targets_enabled=_grouped_targets_enabled,
                                    feasible_components=_feasible_components,
                                    candidate_planar=_candidate_planar,
                                    candidate_rp=_candidate_rp,
                                    candidate_z=_candidate_z,
                                    projected_planar=_projected_planar,
                                    projected_rp=_projected_rp,
                                    projected_z=_projected_z,
                                    base_planar=_base_planar,
                                    base_rp=_base_rp,
                                    base_z=_base_z,
                                    pref_margin=_pref_margin,
                                )
                                _accept_pref_target = _structured_rho_payload.accept_target
                                _accept_pref_mask = _structured_rho_payload.accept_mask
                                _target_exec = _structured_rho_payload.target_exec
                                _mask_exec = _structured_rho_payload.mask_exec
                                _structured_joint_enabled = _structured_rho_payload.enabled
                                self._frontres_state_alpha_mask_last = float(
                                    _state_alpha_mask[:_n_exec, 0].mean().detach().item()
                                )
                                _accept_payload = summarize_frontres_acceptance_payload(
                                    self,
                                    accept_target=_accept_pref_target,
                                    accept_mask=_accept_pref_mask,
                                    target_exec=_target_exec,
                                    mask_exec=_mask_exec,
                                    structured_joint_enabled=_structured_joint_enabled,
                                    pref_margin=_pref_margin,
                                    need=_need,
                                    admissibility=_admissibility,
                                    j_one=_j_one,
                                    j_rho=_j_rho,
                                    j_zero=_j_zero,
                                    tri_weight_repair_mean=_tri_weight_repair_mean,
                                    tri_weight_noisy_mean=_tri_weight_noisy_mean,
                                    tri_weight_stable_mean=_tri_weight_stable_mean,
                                    pref_inertial_penalty_rho_mean=_pref_inertial_penalty_rho_mean,
                                    pref_inertial_penalty_one_mean=_pref_inertial_penalty_one_mean,
                                    rho_target_planar_mean=_rho_target_planar_mean,
                                    rho_target_rp_mean=_rho_target_rp_mean,
                                    rho_target_z_mean=_rho_target_z_mean,
                                    rho_target_spread_mean=_rho_target_spread_mean,
                                    grouped_rho_mask_mean=_grouped_rho_mask_mean,
                                    rho_regret_up_planar_mean=_rho_regret_up_planar_mean,
                                    rho_regret_up_rp_mean=_rho_regret_up_rp_mean,
                                    rho_regret_up_z_mean=_rho_regret_up_z_mean,
                                    rho_regret_down_planar_mean=_rho_regret_down_planar_mean,
                                    rho_regret_down_rp_mean=_rho_regret_down_rp_mean,
                                    rho_regret_down_z_mean=_rho_regret_down_z_mean,
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
                            self.alg.transition.acceptance_target = _accept_pref_target
                            self.alg.transition.acceptance_mask = _accept_pref_mask
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

    def _frontres_raw_anchor_pose(self, cmd_term, n: int, device: torch.device, dtype: torch.dtype):
        return frontres_raw_anchor_pose(self, cmd_term, n, device, dtype)

    def _frontres_stabilizing_candidate_correction(
        self,
        cmd_term,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return frontres_stabilizing_candidate_correction(self, cmd_term, n, device, dtype)

    def _frontres_temporal_continuity_correction(
        self,
        cmd_term,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return frontres_temporal_continuity_correction(self, cmd_term, n, device, dtype)

    def _frontres_update_temporal_reference_cache(self, cmd_term, n: int) -> None:
        return frontres_update_temporal_reference_cache(self, cmd_term, n)

    def _frontres_invalidate_temporal_reference_cache(self, dones: torch.Tensor | None) -> None:
        return frontres_invalidate_temporal_reference_cache(self, dones)

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
