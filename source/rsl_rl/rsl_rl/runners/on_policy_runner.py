# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from collections.abc import Mapping
from collections import deque
from types import SimpleNamespace

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation, MOSAIC, FrontRESUnified
from rsl_rl.frontres.frontres_action_cone import FrontRESActionCone
from rsl_rl.runners.frontres_checkpointing import (
    capture_v015_hsl_fresh_reload_shadow,
    load_v015_hsl_initializer,
    load_runner,
    record_frontres_checkpoint_probe,
    save_runner,
    verify_v015_hsl_fresh_reload,
)
from rsl_rl.runners.frontres_episode_bookkeeping import update_episode_bookkeeping
from rsl_rl.frontres.frontres_executability import FrontRESExecutabilityScorer
from rsl_rl.runners.frontres_dr_sweep_eval import evaluate_frontres_dr_sweep as run_frontres_dr_sweep_eval
from rsl_rl.runners.frontres_rollout_step import prepare_frontres_rollout_step
from rsl_rl.runners.frontres_segment_runner_boundary import FrontRESSegmentRunnerBoundary
from rsl_rl.runners.frontres_segment_live_probe import (
    run_frontres_v015_local_identity_sentinel as run_frontres_v015_local_identity_sentinel_helper,
    run_frontres_segment_live_probe as run_frontres_segment_live_probe_helper,
    run_frontres_segment_single_update,
)
from rsl_rl.runners.frontres_segment_live_update_loop import (
    run_frontres_segment_live_update_loop as run_frontres_segment_live_update_loop_helper,
    run_frontres_v015_formal_training_update_loop as run_frontres_v015_formal_training_update_loop_helper,
    run_frontres_v015_formal_transaction_update_loop as run_frontres_v015_formal_transaction_update_loop_helper,
)
from rsl_rl.runners.frontres_segment_live_sampler import initialize_frontres_segment_live_sampler
from rsl_rl.runners.frontres_segment_live_training import (
    finalize_frontres_v015_local_sentinel_checkpoint as finalize_frontres_v015_local_sentinel_checkpoint_helper,
    run_frontres_segment_offline_eval as run_frontres_segment_offline_eval_helper,
    run_frontres_segment_sequence_offline_eval as run_frontres_segment_sequence_offline_eval_helper,
    run_frontres_segment_live_training_loop,
    run_frontres_segment_periodic_eval as run_frontres_segment_periodic_eval_helper,
)
from rsl_rl.frontres.task_space_correction import apply_frontres_task_corrections
from rsl_rl.runners.frontres_runtime import (
    apply_obs_normalizer,
    append_frontres_future_intent_context,
    append_frontres_fixed_noisy_future_context,
    build_frontres_v015_deployment_observation,
    get_inference_policy_runner,
    maybe_print_frontres_restore_debug,
    read_frontres_v015_deployment_context,
)
from rsl_rl.modules.frontres_observation_layout import (
    FRONTRES_V015_GMT_SUFFIX_DIM,
    resolve_frontres_future_intent_layout,
    resolve_frontres_v015_observation_authority,
)
from rsl_rl.runners.frontres_runner_logging import log_runner
from rsl_rl.frontres.training_schedule import (
    frontres_curriculum_allowed_bases,
    frontres_warmup_perturbation_mode_groups,
    resolve_frontres_mode_state,
)
from rsl_rl.runners.frontres_training_setup import (
    apply_frontres_debug_training_overrides,
    apply_frontres_dr_scale,
    apply_frontres_iteration_dr_controller,
    configure_frontres_pair_layout,
    initialize_frontres_dr_setup,
    maybe_print_frontres_perturbation_curriculum,
    set_frontres_curriculum_modes,
    set_frontres_perturbation_curriculum,
)
from rsl_rl.runners.frontres_warmup import (
    resolve_frontres_warmup_iterations,
    run_frontres_joint_warmup,
    should_exit_after_frontres_stage1_warmup,
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
)


_FRONTRES_LOG_SKIP_KEYS = {
    "self",
    "obs",
    "extras",
    "obs_dict",
    "privileged_obs",
    "teacher_obs",
    "ref_vel_estimator_obs",
    "obs_raw_for_gmt",
    "actions",
    "env_actions",
    "rewards",
    "dones",
    "infos",
    "step_plan",
    "frontres_truth",
    "frontres_reward",
    "_frontres_stats_locs",
    "_frontres_log_locs",
    "_frontres_diag_sums",
    "_hsl_pos_snapshot",
    "_hsl_quat_snapshot",
}


def _frontres_safe_log_value(value):
    """Detach log data so runner diagnostics do not retain rollout CUDA tensors."""
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() == 0:
            return []
        if detached.numel() == 1:
            return detached.item()
        if detached.numel() <= 32:
            return detached.cpu()
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, deque):
        safe_items = []
        for item in value:
            safe_item = _frontres_safe_log_value(item)
            if safe_item is not None:
                safe_items.append(safe_item)
        return deque(safe_items, maxlen=value.maxlen)
    if isinstance(value, Mapping):
        safe_dict = {}
        for key, item in value.items():
            safe_item = _frontres_safe_log_value(item)
            if safe_item is not None:
                safe_dict[key] = safe_item
        return safe_dict
    if isinstance(value, (list, tuple)):
        safe_items = []
        for item in value:
            safe_item = _frontres_safe_log_value(item)
            if safe_item is not None:
                safe_items.append(safe_item)
        return safe_items
    return None


def _frontres_build_safe_log_locs(
    local_vars: Mapping[str, object],
    diagnostic_means: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the runner log dictionary without keeping live CUDA rollout state."""
    safe_locs = {}
    for key, value in local_vars.items():
        if key in _FRONTRES_LOG_SKIP_KEYS:
            continue
        safe_value = _frontres_safe_log_value(value)
        if safe_value is not None:
            safe_locs[key] = safe_value
    if diagnostic_means is not None:
        for key, value in diagnostic_means.items():
            safe_value = _frontres_safe_log_value(value)
            if safe_value is not None:
                safe_locs[key] = safe_value
    return safe_locs


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
            self.cfg["frontres_specialist_mode"] = "rp"
            self.cfg["frontres_perturbation_channels"] = "rp"
            self.cfg["frontres_exec_task_weight"] = 0.0
            self.cfg["frontres_exec_cone_task_weight"] = 0.0
            print(
                "[Runner] FrontRES specialist mode enabled: rp "
                "(local_rp perturbation; full-6D repair)",
                flush=True,
            )
            return
        if mode not in ("rp_z", "z_rp", "vertical_contact"):
            return

        self.cfg["frontres_specialist_mode"] = "rp_z"
        self.cfg["frontres_perturbation_channels"] = "rp_z"
        self.cfg["frontres_exec_task_weight"] = 0.0
        self.cfg["frontres_exec_cone_task_weight"] = 0.0
        print(
            "[Runner] FrontRES specialist mode enabled: rp_z "
            "(global_z + local_rp perturbation; full-6D repair)",
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
        self._frontres_segment_replay_boundary = FrontRESSegmentRunnerBoundary.from_train_cfg(self.cfg)
        self._frontres_hsl_proposal_context_enabled = bool(
            self.training_type == "frontres"
            and self.cfg.get("frontres_stage1_exit_after_warmup", False)
            and self.alg_cfg.get("frontres_training_objective", "") == "supervised_restore"
        )
        self._frontres_hsl_live_smoke_enabled = bool(
            self._frontres_hsl_proposal_context_enabled
            and self.cfg.get("frontres_hsl_live_smoke_enabled", False)
        )
        if self._frontres_hsl_proposal_context_enabled and (
            self._frontres_segment_replay_boundary.requested
            or bool(self.alg_cfg.get("frontres_v015_formal_transaction_enabled", False))
        ):
            raise ValueError(
                "proposal-only Stage-1 HSL cannot mix with Segment Replay or a Stage-3 formal transaction"
            )
        if self.training_type == "frontres":
            self._frontres_segment_replay_boundary.assert_live_runner_ready()
            frontres_segment_sentinel_log = self._frontres_segment_replay_boundary.sentinel_log()
            if frontres_segment_sentinel_log is not None:
                print(frontres_segment_sentinel_log, flush=True)
            frontres_v015_sentinel_log = self._frontres_segment_replay_boundary.v015_sentinel_log()
            if frontres_v015_sentinel_log is not None:
                print(frontres_v015_sentinel_log, flush=True)
            frontres_segment_probe_log = self._frontres_segment_replay_boundary.probe_log()
            if frontres_segment_probe_log is not None:
                print(frontres_segment_probe_log, flush=True)
            frontres_segment_train_log = self._frontres_segment_replay_boundary.train_log()
            if frontres_segment_train_log is not None:
                print(frontres_segment_train_log, flush=True)

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
        self._frontres_fixed_noisy_future_offsets: tuple[int, ...] = ()
        self._frontres_fixed_noisy_actor_context_dim = 0
        self._frontres_future_intent_layout = None
        self._frontres_future_intent_layout_version: str | None = None
        self._frontres_future_intent_actor_context_dim = 0
        v015_formal_layout_requested = bool(
            self.alg_cfg.get("frontres_v015_formal_transaction_enabled", False)
        )
        # Deployment eval 只复用 v015 layout/checkpoint identity, 不请求 Segment sampler.
        if self.training_type == "frontres" and (
            self._frontres_segment_replay_boundary.requested
            or v015_formal_layout_requested
            or self._frontres_hsl_proposal_context_enabled
        ):
            raw_offsets = self.alg_cfg.get("frontres_future_offsets", None)
            layout_version = self.alg_cfg.get("frontres_future_intent_layout_version", None)
            if raw_offsets is None or isinstance(raw_offsets, (str, bytes)) or layout_version is None:
                raise ValueError(
                    "FrontRES v015 requires explicit frontres_future_offsets and "
                    "frontres_future_intent_layout_version; no legacy default is allowed"
                )
            try:
                layout = resolve_frontres_future_intent_layout(raw_offsets, str(layout_version))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid v015 FrontRES future-intent layout: {exc}") from exc
            self._frontres_future_intent_layout = layout
            self._frontres_future_intent_layout_version = layout.version
            self._frontres_future_intent_actor_context_dim = layout.actor_tail_dim
            configured_prefix = int(self.policy_cfg.get("num_frontres_obs", 0) or 0)
            authority = resolve_frontres_v015_observation_authority(
                environment_obs_dim=num_obs,
                configured_frontres_prefix_dim=configured_prefix,
                actor_tail_dim=self._frontres_future_intent_actor_context_dim,
                gmt_suffix_dim=FRONTRES_V015_GMT_SUFFIX_DIM,
            )
            num_actor_obs = authority.combined_obs_dim
            self.policy_cfg["num_frontres_obs"] = authority.frontres_visible_dim
            print(
                "[Runner] FrontRES v015 future-intent actor layout: "
                f"version={layout.version} H={layout.future_offsets} tail={self._frontres_future_intent_actor_context_dim} "
                f"raw_obs={num_obs} actor_obs={num_actor_obs} "
                f"frontres_obs={authority.frontres_visible_dim} gmt_obs={authority.gmt_suffix_dim}",
                flush=True,
            )

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
        self._frontres_action_cone = FrontRESActionCone(self.cfg, self.alg)
        self._frontres_executability = FrontRESExecutabilityScorer(self.cfg, self.alg, self.device)
        self._frontres_segment_sampler = None
        if self.training_type == "frontres":
            initialize_frontres_segment_live_sampler(self)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        # Track whether task-space FrontRES needs partial obs normalization.
        # Runtime layout for task-space FrontRES is:
        #   [0:num_extra] = FrontRES-only prefix, [num_extra:] = GMT obs.
        # When set, the trailing _frontres_gmt_obs_dim dims are GMT-normalized;
        # leading FrontRES-only prefix optionally uses checkpoint empirical stats.
        self._frontres_gmt_obs_dim: int | None = None
        self._frontres_extra_mean: torch.Tensor | None = None  # (1, K) Stage-1 mean for extra dims
        self._frontres_extra_std:  torch.Tensor | None = None  # (1, K) Stage-1 std  for extra dims
        self._frontres_extra_stats_layout_version: str | None = None
        self._frontres_extra_normalizer: EmpiricalNormalization | None = None

        # Check if using ResidualActorCritic (special handling for GMT normalizer)
        if isinstance(policy, (ResidualActorCritic, FrontRESActorCritic)):
            # Use GMT's frozen normalizer for observations
            if policy.gmt_normalizer is not None:
                self.obs_normalizer = policy.gmt_normalizer
                print(f"[Runner] Using GMT's frozen normalizer for {type(policy).__name__}")
                # Task-space mode: student obs may have FrontRES-only prefix dims beyond
                # what the GMT normalizer expects.  Detect and store the split point.
                if (isinstance(policy, FrontRESActorCritic)
                        and getattr(policy, 'num_task_corrections', 0) > 0):
                    _gmt_mean = getattr(policy.gmt_normalizer, '_mean', None)
                    gmt_norm_dim = _gmt_mean.shape[-1] if _gmt_mean is not None else num_obs
                    if num_obs > gmt_norm_dim:
                        self._frontres_gmt_obs_dim = gmt_norm_dim
                        if self.empirical_normalization:
                            self._frontres_extra_normalizer = EmpiricalNormalization(
                                shape=[num_obs - gmt_norm_dim + self._frontres_future_intent_actor_context_dim],
                                until=1.0e8,
                            ).to(self.device)
                        print(f"[Runner] FrontRES task-space obs layout: first "
                              f"{num_obs - gmt_norm_dim + self._frontres_future_intent_actor_context_dim} "
                              "FrontRES-only dims use prefix stats or pass-through; "
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

    def run_frontres_segment_live_probe(
        self,
        init_at_random_ep_len: bool = True,
    ) -> dict[str, object]:
        return run_frontres_segment_live_probe_helper(
            self,
            init_at_random_ep_len=init_at_random_ep_len,
        )

    def run_frontres_segment_live_update_loop(
        self,
        init_at_random_ep_len: bool = True,
        *,
        runner_learn: bool = False,
    ) -> dict[str, float | int]:
        return run_frontres_segment_live_update_loop_helper(
            self,
            init_at_random_ep_len=init_at_random_ep_len,
            runner_learn=runner_learn,
        )

    def run_frontres_v015_formal_transaction(self) -> object:
        """Run one injected v015 request through the exact-one update owner."""

        return run_frontres_v015_formal_transaction_update_loop_helper(self)

    def run_frontres_v015_formal_training_transaction(
        self,
        init_at_random_ep_len: bool = True,
    ) -> object:
        """Collect and commit one complete ordinary Stage-3 transaction."""

        return run_frontres_v015_formal_training_update_loop_helper(
            self,
            init_at_random_ep_len=init_at_random_ep_len,
        )

    def run_frontres_v015_local_identity_sentinel(
        self,
        init_at_random_ep_len: bool = True,
    ) -> object:
        """Run only the explicit v015 local-scenario sentinel, never a legacy live update path."""

        return run_frontres_v015_local_identity_sentinel_helper(
            self,
            init_at_random_ep_len=init_at_random_ep_len,
        )

    def finalize_frontres_v015_local_sentinel_checkpoint(self, result: object) -> str:
        """Save the exact checkpoint-v6 adjacent to one completed local sentinel."""

        return finalize_frontres_v015_local_sentinel_checkpoint_helper(self, result)

    def learn_frontres_segment_live(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = True,
    ) -> None:
        run_frontres_segment_live_training_loop(
            self,
            num_learning_iterations=num_learning_iterations,
            init_at_random_ep_len=init_at_random_ep_len,
        )

    def run_frontres_segment_periodic_eval(self, *, iteration: int, train_summary: dict) -> dict[str, float]:
        return run_frontres_segment_periodic_eval_helper(self, iteration=iteration, train_summary=train_summary)

    def run_frontres_segment_offline_eval(self, *, num_eval_segments: int, rollout_steps: int) -> dict[str, float]:
        return run_frontres_segment_offline_eval_helper(
            self,
            num_eval_segments=num_eval_segments,
            rollout_steps=rollout_steps,
        )

    def run_frontres_policy_quality_eval(
        self,
        *,
        manifest_path: str,
        hsl_checkpoint_path: str,
        policy_checkpoint_path: str,
        result_path: str,
    ):
        # QUALITY-ID-01: thin connector uses a lazy import so old modes never import quality evaluation.
        from rsl_rl.runners.frontres_policy_quality_eval import run_frontres_policy_quality_eval

        return run_frontres_policy_quality_eval(
            self,
            manifest_path=manifest_path,
            hsl_checkpoint_path=hsl_checkpoint_path,
            policy_checkpoint_path=policy_checkpoint_path,
            result_path=result_path,
        )

    def run_frontres_policy_quality_q2d_eval(
        self,
        *,
        manifest_path: str,
        hsl_checkpoint_path: str,
        policy_checkpoint_path: str,
        result_path: str,
    ):
        """Thin connector for the isolated Q2-D scale evaluator."""
        from rsl_rl.runners.frontres_policy_quality_eval import build_frontres_policy_quality_eval_request
        from rsl_rl.runners.frontres_policy_quality_q2d_eval import run_frontres_policy_quality_q2d_scale_eval

        request = build_frontres_policy_quality_eval_request(
            manifest_path=manifest_path,
            hsl_checkpoint_path=hsl_checkpoint_path,
            policy_checkpoint_path=policy_checkpoint_path,
            result_path=result_path,
        )
        return run_frontres_policy_quality_q2d_scale_eval(
            self,
            request=request,
            result_path=result_path,
        )

    def run_frontres_segment_sequence_offline_eval(
        self,
        *,
        num_eval_sequences: int,
        rollout_steps: int,
        max_preroll_steps: int | None = None,
        sampler_seed: int | None = None,
    ) -> dict[str, float]:
        # FRS3-EVAL-003: keep runner API thin and delegate sequence eval ownership.
        return run_frontres_segment_sequence_offline_eval_helper(
            self,
            num_eval_sequences=num_eval_sequences,
            rollout_steps=rollout_steps,
            max_preroll_steps=max_preroll_steps,
            sampler_seed=sampler_seed,
        )

    def run_frontres_v015_deployment_composition_eval(self, *, config):
        """Thin formal connector for the isolated v015 deployment evaluator."""

        from rsl_rl.runners.frontres_segment_sequence_eval import (
            run_frontres_v015_deployment_composition_eval,
        )

        return run_frontres_v015_deployment_composition_eval(self, config=config)

    def _run_frontres_segment_single_update(self, storage_batch) -> object:
        return run_frontres_segment_single_update(self, storage_batch)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901

        # ------------------- 初始化参数 -------------------

        print("[Runner] learn() entered — initializing logger...", flush=True)
        # 入口阶段：准备 logger、writer 与算法侧日志句柄；这里不产生训练数据。
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

        # 训练前检查：确认 teacher / multi-teacher 等外部依赖已经接好。
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

        # 初始观测：拿到 env 的第一帧 obs，并拆成 policy / critic / teacher / estimator 输入。
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
        if self._frontres_hsl_proposal_context_enabled:
            obs = self._append_frontres_future_intent_context(obs)
        else:
            obs = self._append_frontres_fixed_noisy_future_context(obs)
        obs = self._apply_obs_normalizer(obs) # 三种观测量分别使用不同观测归一器
        print("[Runner] Policy obs normalized.", flush=True)

        # 使用观测量归一化器对观测量进行处理
        print("[Runner] Applying privileged/teacher normalizers...", flush=True)
        if self._frontres_hsl_proposal_context_enabled:
            # Proposal-only HSL has no critic/teacher consumer. Avoid even a
            # running-stat write on those inactive paths.
            privileged_obs = privileged_obs.detach()
            teacher_obs = teacher_obs.detach()
        else:
            privileged_obs = self.privileged_obs_normalizer(privileged_obs)
            teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        print("[Runner] Privileged/teacher obs normalized.", flush=True)

        print("[Runner] Switching modules to train mode...", flush=True)
        self.train_mode() # switch to train mode (for dropout for example)
        print("[Runner] Train mode set.", flush=True)

        # 回合统计缓存：这些 deque 只服务日志和 curriculum 反馈，不直接更新网络权重。
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

        # ------------------- 初始化参数 -------------------

        # ------------------- 预热Critic -------------------

        # Start training
        print("[Runner] Preparing iteration counters...", flush=True)
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        print(f"[Runner] Iteration counters ready: start={start_iter}, total={tot_iter}", flush=True)

        # FrontRES 模式解析：把 config / policy 类型压缩成后续主循环使用的阶段开关。
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

        def _frontres_pipeline_section(label: str, *, rollout_step: int | None = None) -> None:
            if not (
                _is_frontres
                and bool(getattr(self.alg, "frontres_cuda_memory_debug", False))
            ):
                return
            step_text = "n/a" if rollout_step is None else str(rollout_step)
            prefix = f"[FrontRES pipeline] it={getattr(self.alg, 'current_learning_iteration', self.current_learning_iteration)}"
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                try:
                    device = torch.device(self.device)
                    torch.cuda.synchronize(device)
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    allocated = torch.cuda.memory_allocated(device)
                    reserved = torch.cuda.memory_reserved(device)

                    def _gib(value: int) -> float:
                        return float(value) / (1024.0 ** 3)

                    print(
                        f"{prefix} section={label} step={step_text} "
                        f"alloc={_gib(allocated):.2f}GiB "
                        f"reserved={_gib(reserved):.2f}GiB "
                        f"free={_gib(free_bytes):.2f}GiB "
                        f"total={_gib(total_bytes):.2f}GiB",
                        flush=True,
                    )
                    return
                except Exception as exc:
                    print(f"{prefix} section={label} step={step_text} cuda_mem_unavailable={exc}", flush=True)
                    return
            print(f"{prefix} section={label} step={step_text}", flush=True)

        # 四分支 rollout 布局：Train / Candidate / Noisy-GMT / Clean-GMT 的 env 区间在这里确定。
        _frontres_pair_layout = configure_frontres_pair_layout(self, is_frontres=_is_frontres)
        _use_quartet_reward = _frontres_pair_layout.use_quartet_reward
        N_train = _frontres_pair_layout.n_train
        N_candidate = _frontres_pair_layout.n_candidate
        N_base = _frontres_pair_layout.n_base
        N_clean = _frontres_pair_layout.n_clean
        cur_reward_sum_gmt = _frontres_pair_layout.cur_reward_sum_gmt

        # DR curriculum 初值：后续每个 iteration 会基于这一组状态更新 perturbation 强度。
        _frontres_dr_setup = initialize_frontres_dr_setup(self, is_frontres=_is_frontres)
        _dr_max = _frontres_dr_setup.dr_max
        _dr_min = _frontres_dr_setup.dr_min
        _dr_ema_alpha = _frontres_dr_setup.dr_ema_alpha
        _dr_scale_init = _frontres_dr_setup.dr_scale_init
        _dr_scale = _frontres_dr_setup.dr_scale
        _r_delta_ema = _frontres_dr_setup.r_delta_ema
        _perturb_target = _frontres_dr_setup.perturb_target

        maybe_print_frontres_perturbation_curriculum(self, is_frontres=_is_frontres)

        # Proposal-only HSL runs before PPO and owns only the residual-actor
        # current anti-DR target. Critic and executable-energy loss are excluded.
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

        # ------------------- 预热Critic -------------------

        if should_exit_after_frontres_stage1_warmup(
            self.cfg,
            is_frontres=_is_frontres,
            warmup_iters=_warmup_iters,
        ):
            print(
                "[Runner] Stage 1 HSL warmup-only run complete; exiting before PPO loop.",
                flush=True,
            )
            return

        print(
            f"[Runner] Entering PPO loop: start_iter={start_iter}, "
            f"tot_iter={tot_iter}, steps_per_env={self.num_steps_per_env}",
            flush=True,
        )
        for it in range(start_iter, tot_iter):
            start = time.time()
            self.alg.current_learning_iteration = it
            _frontres_pipeline_section("iteration_start")
            # Iteration controller: DR scale and frontier/boundary state are
            # fixed before rollout collection.
            # ------------------- Update Curriculum -------------------

            _dr_iter_plan = apply_frontres_iteration_dr_controller(
                self,
                iteration=it,
                is_frontres=_is_frontres,
                frontres_hsl_restore=_frontres_hsl_restore,
                perturb_target=_perturb_target,
                critic_warmup_iters=critic_warmup_iters,
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
            _hsl_boundary_available = _dr_iter_plan.hsl_boundary_available
            _use_boundary = _dr_iter_plan.use_boundary
            _gmt_frontier_score = _dr_iter_plan.gmt_frontier_score
            _gmt_frontier_decision = _dr_iter_plan.gmt_frontier_decision

            # ------------------- Update Curriculum -------------------

            # Rollout 收集：先用当前 policy 跑 num_steps_per_env 步，写满 storage 后再 update。
            with torch.inference_mode(): # 关闭计算图的梯度追踪, 只进行推理
                _frontres_pipeline_section("rollout_start")
                _pipeline_rollout_interval = max(1, int(self.num_steps_per_env) // 4)
                for _rollout_step in range(self.num_steps_per_env):
                    _pipeline_step_sentinel = (
                        _rollout_step == 0
                        or _rollout_step == self.num_steps_per_env - 1
                        or _rollout_step % _pipeline_rollout_interval == 0
                    )
                    if _pipeline_step_sentinel:
                        _frontres_pipeline_section("step_prepare_before", rollout_step=_rollout_step)

                    # ------------------- Policy Rollout -------------------

                    # Step preparation：policy action 在这里产生；FrontRES task-space 模式还会写入 GMT 可执行动作。
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

                    if _pipeline_step_sentinel:
                        _frontres_pipeline_section("env_step_before", rollout_step=_rollout_step)
                    # Env step：真正推进仿真；从这里开始得到下一帧 obs、reward、done 和诊断 infos。
                    # NOTE: This is where the environment computes the *next* observation internally.
                    # The result is returned here and then used in the next loop iteration.
                    obs, rewards, dones, infos = self.env.step(env_actions.to(self.env.device))
                    if _pipeline_step_sentinel:
                        _frontres_pipeline_section("env_step_after", rollout_step=_rollout_step)

                    # Move to device
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    if (
                        _is_frontres
                        and _is_task_space_mode
                        and bool(self.cfg.get("frontres_hsl_rollout_label_enabled", False))
                    ):
                        raise RuntimeError(
                "FRS-TRAIN-v011 forbids frontres_hsl_rollout_label_enabled on every active Stage-3 route"
                        )
                    
                    # ------------------- Policy Rollout -------------------

                    # ------------------- Reward Compute -------------------

                    # ── GMT baselines ────────────────────────────────────────────────
                    # Noisy-GMT envs share the FrontRES perturbation and receive no
                    # correction. Clean-GMT envs are registered via set_baseline_envs()
                    # so the perturber keeps only the clean diagnostic block unperturbed.

                    # ── FrontRES B1 delta-reward ────────────────────────────────────────
                    # Noisy-GMT envs run the same perturbed reference with zero residual
                    # correction, so their reward is the GMT-only baseline.
                    # r_delta isolates FrontRES contribution to anchor tracking only.
                    # GMT env rewards are zeroed → returns ≈ 0 → advantage ≈ 0 → no policy gradient.
                    
                    # Fix 1: anchor-only r_delta (HRL intrinsic reward)
                    # Global reward (joint_torque, contact, body vel) conflates FrontRES with GMT:
                    #   - FrontRES corrects correctly but causes torque spike → penalises correct action
                    #   - FrontRES correction has no effect → should give 0, not negative

                    # Anchor error is the ONLY thing FrontRES directly controls. Using it as the
                    # intrinsic reward decouples FrontRES credit from GMT behaviour.

                    # Stage 2 is supervised-only: HSL target storage is the training signal.
                    if _is_frontres: 
                        if not _frontres_supervised_restore:
                            raise RuntimeError(
                                "Standard FrontRES rollout reached the retired reward connector; "
                                "use the dedicated Segment Replay route."
                            )
                        rewards = torch.zeros_like(rewards)
                        frontres_reward = SimpleNamespace(
                            rewards=rewards,
                            reward_window=None,
                            r_raw_gmt=None,
                            r_candidate_gmt=None,
                            r_clean_gmt=None,
                            prev_delta_q=actions.clone(),
                            term_count=_frontres_term_count + int(dones[:N_train].sum().item()),
                            step_count=_frontres_step_count + N_train,
                        )
                        _frontres_prev_delta_q = frontres_reward.prev_delta_q
                        _frontres_term_count = frontres_reward.term_count
                        _frontres_step_count = frontres_reward.step_count
                    
                    # ------------------- Reward Compute -------------------

                    # ------------------- logging info -------------------

                    # Observation refresh：把 env 返回的下一帧 obs 归一化，准备下一次 policy forward。
                    obs_dict = infos.get("observations", {})
                    if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
                        obs = obs_dict[self.policy_obs_type].to(self.device)
                    else:
                        obs = obs.to(self.device)

                    # For Stage 1: update raw obs BEFORE normalization for GMT ONNX next step
                    if self.training_type == "supervise":
                        obs_raw_for_gmt = obs.clone()

                    # perform normalization 对本次循环的观测量进行归一化, 用于计算下步动作
                    obs = self._append_frontres_fixed_noisy_future_context(obs)
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

                    # Storage write：algorithm.process_env_step() 将当前 transition 写入 rollout buffer。
                    if _pipeline_step_sentinel:
                        _frontres_pipeline_section("storage_write_before", rollout_step=_rollout_step)
                    self.alg.process_env_step(rewards, dones, infos)  # stores FrontRES residual actions, not GMT robot actions
                    if _pipeline_step_sentinel:
                        _frontres_pipeline_section("storage_write_after", rollout_step=_rollout_step)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if hasattr(self.alg, "rnd") and self.alg.rnd else None

                    # Episode bookkeeping：只更新日志/curriculum 统计，不改变 storage 中的训练样本。
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
                        r_candidate_gmt=frontres_reward.r_candidate_gmt if _is_frontres else None,
                        r_raw_gmt=frontres_reward.r_raw_gmt if _is_frontres else None,
                        r_clean_gmt=frontres_reward.r_clean_gmt if _is_frontres else None,
                    )
                
                    # ------------------- logging info -------------------

                stop = time.time()
                collection_time = stop - start
                start = stop
                _frontres_pipeline_section("rollout_end")

                # Return / advantage：rollout 收集完成后，用 critic 观测计算 GAE/returns。
                if self.training_type in ["rl", "mosaic", "frontres"]:
                    _frontres_pipeline_section("compute_returns_before")
                    self.alg.compute_returns(privileged_obs)
                    _frontres_pipeline_section("compute_returns_after")

            # Algorithm update is the only parameter-update boundary.
            # Pass the current iteration to algorithms that use it for logging.
            self.alg.current_learning_iteration = it
            _frontres_pipeline_section("algorithm_update_before")
            loss_dict = self.alg.update() # 调用mosaic.py中的update()函数进行权重更新
            _frontres_pipeline_section("algorithm_update_after")

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if _is_frontres:
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
            _frontres_log_locs = _frontres_build_safe_log_locs(locals())

            # Log / checkpoint：打印本轮训练状态，并按 save_interval 保存模型。
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                _frontres_pipeline_section("log_before")
                self.log(_frontres_log_locs)
                _frontres_pipeline_section("log_after")

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
                _frontres_log_locs
                if "_frontres_log_locs" in locals()
                else _frontres_build_safe_log_locs(locals()),
                _final_checkpoint_path,
            )

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        return log_runner(self, locs, width=width, pad=pad)

    def _record_frontres_checkpoint_probe(self, locs: dict, checkpoint_path: str) -> None:
        return record_frontres_checkpoint_probe(self, locs, checkpoint_path)

    def save(self, path: str, infos=None):
        return save_runner(self, path, infos=infos)

    def _capture_v015_hsl_fresh_reload_shadow(self):
        return capture_v015_hsl_fresh_reload_shadow(self)

    def _verify_v015_hsl_fresh_reload(self, shadow, **kwargs):
        return verify_v015_hsl_fresh_reload(shadow, **kwargs)

    def load(self, path: str, load_optimizer: bool = True, load_critic: bool = True):
        return load_runner(self, path, load_optimizer=load_optimizer, load_critic=load_critic)

    def load_frontres_v015_hsl_initializer(self, path: str):
        """Initialize only the v015 Stage-3 actor boundary from strict HSL-v1."""

        return load_v015_hsl_initializer(self, path)

    def get_inference_policy(self, device=None):
        return get_inference_policy_runner(self, device=device)

    def _apply_obs_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
        return apply_obs_normalizer(self, obs)

    def _append_frontres_fixed_noisy_future_context(self, obs: torch.Tensor) -> torch.Tensor:
        return append_frontres_fixed_noisy_future_context(self, obs)

    def _append_frontres_future_intent_context(self, obs: torch.Tensor) -> torch.Tensor:
        return append_frontres_future_intent_context(self, obs)

    def _read_frontres_v015_deployment_context(self, env_ids: torch.Tensor | None = None):
        return read_frontres_v015_deployment_context(self, env_ids)

    def _build_frontres_v015_deployment_observation(
        self,
        obs: torch.Tensor,
        *,
        snapshot: dict[str, object] | None = None,
    ) -> torch.Tensor:
        return build_frontres_v015_deployment_observation(self, obs, snapshot=snapshot)

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
            if self._frontres_gmt_obs_dim is not None:
                self.obs_normalizer.eval()
                if hasattr(self.obs_normalizer, "until"):
                    self.obs_normalizer.until = 0
                if self._frontres_extra_normalizer is not None:
                    if self._frontres_extra_mean is None or self._frontres_extra_std is None:
                        self._frontres_extra_normalizer.train()
                    else:
                        self._frontres_extra_normalizer.eval()
            else:
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
            if getattr(self, "_frontres_extra_normalizer", None) is not None:
                self._frontres_extra_normalizer.eval()
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
