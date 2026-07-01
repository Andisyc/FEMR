from __future__ import annotations

from typing import Optional
import importlib.util
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.modules import ActorCritic, FrontRESActorCritic, ResidualActorCritic
from rsl_rl.storage import RolloutStorage

_AUTHORITY_TARGETS_PATH = Path(__file__).resolve().parents[1] / "frontres" / "frontres_authority_targets.py"
_AUTHORITY_TARGETS_SPEC = importlib.util.spec_from_file_location(
    "frontres_authority_targets_algorithm_module",
    _AUTHORITY_TARGETS_PATH,
)
if _AUTHORITY_TARGETS_SPEC is None or _AUTHORITY_TARGETS_SPEC.loader is None:
    raise RuntimeError(f"Could not load FrontRES authority target helper from {_AUTHORITY_TARGETS_PATH}.")
_AUTHORITY_TARGETS_MODULE = importlib.util.module_from_spec(_AUTHORITY_TARGETS_SPEC)
_AUTHORITY_TARGETS_SPEC.loader.exec_module(_AUTHORITY_TARGETS_MODULE)
resolve_frontres_authority_targets = _AUTHORITY_TARGETS_MODULE.resolve_frontres_authority_targets


class FrontRESUnified:
    """FrontRES PPO + supervised ΔSE3 training.

    This class intentionally owns only the pieces FrontRES needs:
    on-policy PPO, the online ΔSE3 supervised auxiliary loss, optional reference
    velocity estimation, and the split-env FrontRES mask.  MOSAIC teacher BC and
    off-policy expert BC are not implemented here by design.
    """

    policy: ActorCritic

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        obs_normalizer: Optional[torch.nn.Module] = None,
        privileged_obs_normalizer: Optional[torch.nn.Module] = None,
        use_estimate_ref_vel: bool = False,
        ref_vel_estimator_checkpoint_path: Optional[str] = None,
        ref_vel_estimator_type: str = "mlp",
        lambda_supervised: float = 0.0,
        lambda_supervised_min: float = 0.05,
        lambda_supervised_decay: float = 0.997,
        supervised_trigger_cosine_sim: float = 0.85,
        supervised_rpy_loss_weight: float = 1.0,
        supervised_conf_loss_weight: float = 0.05,
        supervised_direction_loss_weight: float = 0.1,
        supervised_valid_loss_weight: float = 4.0,
        supervised_magnitude_loss_weight: float = 0.0,
        supervised_over_loss_weight: float = 0.0,
        supervised_smooth_loss_weight: float = 0.0,
        supervised_coeff_sparse_weight: float = 0.0,
        supervised_coeff_miss_weight: float = 0.0,
        supervised_coeff_smooth_weight: float = 0.0,
        supervised_harm_loss_weight: float = 1.0,
        frontres_supervised_lr_schedule: str = "fixed",
        frontres_supervised_lr_start: float | None = None,
        frontres_supervised_lr_peak: float | None = None,
        frontres_supervised_lr_min: float | None = None,
        frontres_supervised_lr_warmup_iters: int = 0,
        frontres_supervised_lr_cosine_iters: int = 1000,
        frontres_restore_debug_print_interval: int = 10,
        ppo_actor_warmup_iterations: int = 0,
        ppo_actor_ramp_iterations: int = 0,
        ppo_advantage_focal_power: float = 0.0,
        frontres_active_task_dims: list[int] | None = None,
        frontres_training_objective: str = "ppo_hrl",
        frontres_segment_replay_enabled: bool = False,
        frontres_segment_live_runner_enabled: bool = False,
        frontres_segment_live_sentinel_only: bool = False,
        frontres_segment_live_probe_only: bool = False,
        frontres_segment_live_storage_write_only: bool = False,
        frontres_segment_live_single_update_only: bool = False,
        frontres_segment_live_update_loop_only: bool = False,
        frontres_segment_live_train_enabled: bool = False,
        frontres_segment_live_update_steps: int = 4,
        frontres_segment_live_fail_on_invalid_update: bool = True,
        frontres_segment_live_min_valid_count: int = 1,
        frontres_segment_live_fail_on_nonfinite: bool = True,
        frontres_hsl_init_enabled: bool = False,
        frontres_segment_k: int = 4,
        frontres_segment_cache_dir: str = "",
        frontres_segment_shard_cache_size: int = 8,
        frontres_segment_include_boundary_diagnostic: bool = False,
        frontres_segment_sampler_global_frac: float = 0.4,
        frontres_segment_sampler_replay_frac: float = 0.5,
        frontres_segment_sampler_review_frac: float = 0.1,
        frontres_segment_reset_mode: str = "auto",
        frontres_acceptance_preference_weight: float = 0.0,
        frontres_acceptance_preference_focal_gamma: float = 0.0,
        frontres_acceptance_preference_balance_min: float = 1.0,
        frontres_acceptance_preference_balance_max: float = 1.0,
        frontres_state_alpha_weight: float = 0.0,
        frontres_structured_joint_rl_enabled: bool = False,
        frontres_structured_joint_rl_weight: float = 0.0,
        frontres_structured_joint_rl_adv_clip: float = 5.0,
        frontres_structured_joint_rl_normalize_advantage: bool = False,
        frontres_structured_joint_rl_loss_mode: str = "ppo_clipped",
        frontres_structured_joint_rl_keep_legacy_bce: bool = False,
        frontres_structured_joint_rl_disable_generic_ppo: bool = True,
        frontres_structured_joint_exec_floor: float = 0.0,
        frontres_structured_joint_rho_retention_weight: float = 0.0,
        frontres_structured_joint_directional_weight: float = 1.0,
        frontres_structured_joint_underwrite_weight: float = 0.0,
        frontres_structured_joint_repair_loss_kind: str = "current_rho_linear",
        frontres_structured_joint_repair_loss_scale: float = 1.0,
        frontres_structured_joint_rho_center: float = 0.5,
        frontres_structured_joint_retention_prior_weight: float = 0.0,
        frontres_structured_joint_floor_penalty_weight: float = 5.0,
        frontres_structured_joint_full_repair_bonus_weight: float = 1.0,
        frontres_structured_joint_prior_loss_weight: float = 0.0,
        frontres_authority_actor_critic_enabled: bool = False,
        frontres_authority_actor_loss_weight: float = 1.0,
        frontres_authority_critic_loss_weight: float = 1.0,
        frontres_authority_actor_warmup_iterations: int = 0,
        frontres_authority_actor_ramp_iterations: int = 0,
        frontres_authority_return_horizon: int = 1,
        frontres_reward_compute_live_debug: bool = False,
        frontres_cuda_memory_debug: bool = False,
        diagnose_gradient_conflict: bool = True,
        hybrid: bool = True,
        use_ppo: bool = True,
        gradient_accumulation_steps: int = 1,
        **disabled_mosaic_kwargs,
    ):
        self._assert_no_mosaic_branches(disabled_mosaic_kwargs)
        if not hybrid:
            raise ValueError("FrontRESUnified supports only hybrid=True PPO+supervised training.")
        if not use_ppo:
            raise ValueError("FrontRESUnified requires use_ppo=True.")
        if gradient_accumulation_steps != 1:
            raise ValueError("FrontRESUnified does not use MOSAIC gradient accumulation.")

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.rnd_optimizer = None
        self.symmetry = None

        if rnd_cfg is not None:
            raise ValueError("FrontRESUnified does not support RND.")
        if symmetry_cfg is not None:
            raise ValueError("FrontRESUnified does not support symmetry augmentation.")

        self.obs_normalizer = obs_normalizer
        self.privileged_obs_normalizer = privileged_obs_normalizer

        self.use_estimate_ref_vel = use_estimate_ref_vel
        self.ref_vel_estimator = None
        self.ref_vel_estimator_obs_shape = None
        if use_estimate_ref_vel:
            if ref_vel_estimator_checkpoint_path is None:
                raise ValueError("ref_vel_estimator_checkpoint_path must be provided when use_estimate_ref_vel=True")
            self._load_ref_vel_estimator(ref_vel_estimator_checkpoint_path, ref_vel_estimator_type)

        self.policy = policy.to(self.device)

        trainable_params = self._collect_trainable_params(policy)
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate)

        self.storage: RolloutStorage = None
        self.transition = RolloutStorage.Transition()

        self.use_ppo = True
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.lambda_supervised = lambda_supervised
        self.lambda_supervised_min = lambda_supervised_min
        self.lambda_supervised_decay_rate = lambda_supervised_decay
        self.supervised_trigger_cosine_sim = supervised_trigger_cosine_sim
        self.supervised_rpy_loss_weight = supervised_rpy_loss_weight
        self.supervised_conf_loss_weight = supervised_conf_loss_weight
        self.supervised_direction_loss_weight = supervised_direction_loss_weight
        self.supervised_valid_loss_weight = supervised_valid_loss_weight
        self.supervised_magnitude_loss_weight = float(supervised_magnitude_loss_weight)
        self.supervised_over_loss_weight = float(supervised_over_loss_weight)
        self.supervised_smooth_loss_weight = float(supervised_smooth_loss_weight)
        self.supervised_coeff_sparse_weight = float(supervised_coeff_sparse_weight)
        self.supervised_coeff_miss_weight = float(supervised_coeff_miss_weight)
        self.supervised_coeff_smooth_weight = float(supervised_coeff_smooth_weight)
        self.supervised_harm_loss_weight = float(supervised_harm_loss_weight)
        self.frontres_supervised_lr_schedule = str(frontres_supervised_lr_schedule).lower()
        self.frontres_supervised_lr_start = float(frontres_supervised_lr_start) if frontres_supervised_lr_start is not None else float(learning_rate)
        self.frontres_supervised_lr_peak = float(frontres_supervised_lr_peak) if frontres_supervised_lr_peak is not None else float(learning_rate)
        self.frontres_supervised_lr_min = float(frontres_supervised_lr_min) if frontres_supervised_lr_min is not None else float(learning_rate)
        self.frontres_supervised_lr_warmup_iters = int(frontres_supervised_lr_warmup_iters)
        self.frontres_supervised_lr_cosine_iters = int(frontres_supervised_lr_cosine_iters)
        self.frontres_restore_debug_print_interval = int(frontres_restore_debug_print_interval)
        self.ppo_actor_warmup_iterations = int(ppo_actor_warmup_iterations)
        self.ppo_actor_ramp_iterations = int(ppo_actor_ramp_iterations)
        self.ppo_advantage_focal_power = float(ppo_advantage_focal_power)
        self.frontres_active_task_dims = frontres_active_task_dims
        self.frontres_training_objective = str(frontres_training_objective).lower()
        self.frontres_segment_replay_enabled = bool(frontres_segment_replay_enabled)
        self.frontres_segment_live_runner_enabled = bool(frontres_segment_live_runner_enabled)
        self.frontres_segment_live_sentinel_only = bool(frontres_segment_live_sentinel_only)
        self.frontres_segment_live_probe_only = bool(frontres_segment_live_probe_only)
        self.frontres_segment_live_storage_write_only = bool(frontres_segment_live_storage_write_only)
        self.frontres_segment_live_single_update_only = bool(frontres_segment_live_single_update_only)
        self.frontres_segment_live_update_loop_only = bool(frontres_segment_live_update_loop_only)
        self.frontres_segment_live_train_enabled = bool(frontres_segment_live_train_enabled)
        self.frontres_segment_live_update_steps = max(1, int(frontres_segment_live_update_steps))
        self.frontres_segment_live_fail_on_invalid_update = bool(frontres_segment_live_fail_on_invalid_update)
        self.frontres_segment_live_min_valid_count = max(0, int(frontres_segment_live_min_valid_count))
        self.frontres_segment_live_fail_on_nonfinite = bool(frontres_segment_live_fail_on_nonfinite)
        self.frontres_hsl_init_enabled = bool(frontres_hsl_init_enabled)
        self.frontres_segment_k = max(1, int(frontres_segment_k))
        self.frontres_segment_cache_dir = str(frontres_segment_cache_dir or "")
        self.frontres_segment_shard_cache_size = max(1, int(frontres_segment_shard_cache_size))
        self.frontres_segment_include_boundary_diagnostic = bool(frontres_segment_include_boundary_diagnostic)
        self.frontres_segment_sampler_global_frac = max(0.0, float(frontres_segment_sampler_global_frac))
        self.frontres_segment_sampler_replay_frac = max(0.0, float(frontres_segment_sampler_replay_frac))
        self.frontres_segment_sampler_review_frac = max(0.0, float(frontres_segment_sampler_review_frac))
        self.frontres_segment_reset_mode = str(frontres_segment_reset_mode).lower()
        if self.frontres_segment_reset_mode not in ("auto", "direct", "preroll"):
            raise ValueError("frontres_segment_reset_mode must be 'auto', 'direct', or 'preroll'")
        if self.frontres_training_objective == "segment_replay_hrl":
            if not self.frontres_segment_replay_enabled:
                raise ValueError("segment_replay_hrl requires frontres_segment_replay_enabled=True")
            if not self.frontres_segment_live_runner_enabled:
                raise NotImplementedError(
                    "segment_replay_hrl is recognized, but live runner integration is disabled. "
                    "Use Step 4-7 toy contract tests until the live Stage 3 connector is integrated."
                )
            if self.frontres_segment_live_sentinel_only:
                print(
                    "[FrontRESUnified] Segment Replay HRL live sentinel initialized; "
                    "PPO/update training remains disabled.",
                    flush=True,
                )
            elif self.frontres_segment_live_probe_only:
                print(
                    "[FrontRESUnified] Segment Replay HRL live probe initialized; "
                    "storage/write and PPO/update training remain disabled.",
                    flush=True,
                )
            elif self.frontres_segment_live_storage_write_only:
                print(
                    "[FrontRESUnified] Segment Replay HRL live storage probe initialized; "
                    "PPO/update training remains disabled.",
                    flush=True,
                )
            elif self.frontres_segment_live_single_update_only:
                print(
                    "[FrontRESUnified] Segment Replay HRL live single-update probe initialized; "
                    "runner will execute exactly one PPO optimizer step and exit.",
                    flush=True,
                )
            elif self.frontres_segment_live_update_loop_only:
                print(
                    "[FrontRESUnified] Segment Replay HRL live update-loop probe initialized; "
                    f"runner will execute {self.frontres_segment_live_update_steps} PPO optimizer steps and exit.",
                    flush=True,
                )
            elif self.frontres_segment_live_train_enabled:
                print(
                    "[FrontRESUnified] Segment Replay HRL live training initialized; "
                    f"runner will execute {self.frontres_segment_live_update_steps} PPO optimizer steps per iteration.",
                    flush=True,
                )
            else:
                raise NotImplementedError(
                    "segment_replay_hrl is recognized, but runner/PPO integration is not wired yet. "
                    "Use Step 4-7 toy contract tests until the live Stage 3 connector is integrated."
                )
        self.frontres_acceptance_preference_weight = float(frontres_acceptance_preference_weight)
        self.frontres_acceptance_preference_focal_gamma = max(
            0.0, float(frontres_acceptance_preference_focal_gamma)
        )
        _balance_min = max(0.0, float(frontres_acceptance_preference_balance_min))
        _balance_max = max(_balance_min, float(frontres_acceptance_preference_balance_max))
        self.frontres_acceptance_preference_balance_min = _balance_min
        self.frontres_acceptance_preference_balance_max = _balance_max
        self.frontres_state_alpha_weight = max(0.0, float(frontres_state_alpha_weight))
        self.frontres_structured_joint_rl_enabled = bool(frontres_structured_joint_rl_enabled)
        self.frontres_structured_joint_rl_weight = max(0.0, float(frontres_structured_joint_rl_weight))
        self.frontres_structured_joint_rl_adv_clip = max(0.0, float(frontres_structured_joint_rl_adv_clip))
        self.frontres_structured_joint_rl_normalize_advantage = bool(
            frontres_structured_joint_rl_normalize_advantage
        )
        self.frontres_structured_joint_rl_loss_mode = str(
            frontres_structured_joint_rl_loss_mode
        ).lower()
        if self.frontres_structured_joint_rl_loss_mode not in ("ppo_clipped", "region_direct"):
            raise ValueError(
                "frontres_structured_joint_rl_loss_mode must be 'ppo_clipped' or 'region_direct'"
            )
        self.frontres_structured_joint_rl_keep_legacy_bce = bool(
            frontres_structured_joint_rl_keep_legacy_bce
        )
        self.frontres_structured_joint_rl_disable_generic_ppo = bool(
            frontres_structured_joint_rl_disable_generic_ppo
        )
        self.frontres_structured_joint_exec_floor = float(frontres_structured_joint_exec_floor)
        self.frontres_structured_joint_rho_retention_weight = max(
            0.0, float(frontres_structured_joint_rho_retention_weight)
        )
        self.frontres_structured_joint_directional_weight = max(
            0.0, float(frontres_structured_joint_directional_weight)
        )
        self.frontres_structured_joint_underwrite_weight = max(
            0.0, float(frontres_structured_joint_underwrite_weight)
        )
        self.frontres_structured_joint_repair_loss_kind = str(
            frontres_structured_joint_repair_loss_kind
        ).lower()
        if self.frontres_structured_joint_repair_loss_kind not in (
            "current_rho_linear",
            "bce_logit",
        ):
            raise ValueError(
                "frontres_structured_joint_repair_loss_kind must be "
                "'current_rho_linear' or 'bce_logit'"
            )
        self.frontres_structured_joint_repair_loss_scale = max(
            0.0, float(frontres_structured_joint_repair_loss_scale)
        )
        self.frontres_structured_joint_rho_center = min(
            1.0, max(0.0, float(frontres_structured_joint_rho_center))
        )
        self.frontres_structured_joint_retention_prior_weight = max(
            0.0, float(frontres_structured_joint_retention_prior_weight)
        )
        self.frontres_structured_joint_floor_penalty_weight = max(
            0.0, float(frontres_structured_joint_floor_penalty_weight)
        )
        self.frontres_structured_joint_full_repair_bonus_weight = max(
            0.0, float(frontres_structured_joint_full_repair_bonus_weight)
        )
        self.frontres_structured_joint_prior_loss_weight = max(
            0.0, float(frontres_structured_joint_prior_loss_weight)
        )
        self.frontres_authority_actor_critic_enabled = bool(frontres_authority_actor_critic_enabled)
        self.frontres_authority_actor_loss_weight = max(0.0, float(frontres_authority_actor_loss_weight))
        self.frontres_authority_critic_loss_weight = max(0.0, float(frontres_authority_critic_loss_weight))
        self.frontres_authority_actor_warmup_iterations = max(0, int(frontres_authority_actor_warmup_iterations))
        self.frontres_authority_actor_ramp_iterations = max(0, int(frontres_authority_actor_ramp_iterations))
        self.frontres_authority_return_horizon = max(1, int(frontres_authority_return_horizon))
        if self.frontres_authority_actor_critic_enabled:
            if self._structured_joint_rl_enabled():
                raise ValueError(
                    "FrontRES authority actor-critic and structured-joint rho loss must not be enabled together."
                )
            if getattr(self.policy, "authority_actor", None) is None:
                raise ValueError("frontres_authority_actor_critic_enabled=True requires policy.authority_actor.")
            if getattr(self.policy, "authority_critic", None) is None:
                raise ValueError("frontres_authority_actor_critic_enabled=True requires policy.authority_critic.")
        self.frontres_reward_compute_live_debug = bool(frontres_reward_compute_live_debug)
        self.frontres_cuda_memory_debug = bool(frontres_cuda_memory_debug)
        self.diagnose_gradient_conflict = bool(diagnose_gradient_conflict)
        self.ppo_actor_weight = 1.0
        self._supervised_decay_triggered = False
        self._supervised_cosine_ema = 0.0
        self._supervised_ema_alpha = 0.05

        self.is_frontres_unified = True
        self._print_init_summary()

    @staticmethod
    def _assert_no_mosaic_branches(kwargs: dict) -> None:
        forbidden_nonzero = {
            "teacher_checkpoint_path": None,
            "teacher_policy": None,
            "teacher_policy_cfg": None,
            "teacher_obs_source_mapping": None,
            "teacher_critic_checkpoint_path": None,
            "expert_trajectory_path": None,
        }
        for key, disabled_value in forbidden_nonzero.items():
            if kwargs.get(key, disabled_value) is not disabled_value:
                raise ValueError(f"FrontRESUnified does not support MOSAIC branch '{key}'.")

        for key in ("lambda_teacher_init", "lambda_teacher_min", "lambda_off_policy", "lambda_off_policy_min"):
            if float(kwargs.get(key, 0.0) or 0.0) != 0.0:
                raise ValueError(f"FrontRESUnified requires {key}=0.0.")

    def _load_ref_vel_estimator(self, checkpoint_path: str, estimator_type: str) -> None:
        print(f"[FrontRESUnified] Loading reference velocity estimator from: {checkpoint_path}")
        if estimator_type == "mlp":
            from rsl_rl.modules import VelocityEstimator

            self.ref_vel_estimator = VelocityEstimator.load(checkpoint_path, device=self.device)
        elif estimator_type == "transformer":
            from rsl_rl.modules import VelocityEstimatorTransformer

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.ref_vel_estimator = VelocityEstimatorTransformer(
                feature_dim=checkpoint.get("feature_dim", 61),
                history_length=checkpoint.get("history_length", 5),
                d_model=checkpoint.get("d_model", 128),
                nhead=checkpoint.get("nhead", 4),
                num_layers=checkpoint.get("num_layers", 2),
            ).to(self.device)
            self.ref_vel_estimator.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError(f"Unknown ref_vel_estimator_type: {estimator_type}. Must be 'mlp' or 'transformer'.")

        self.ref_vel_estimator.eval()
        for param in self.ref_vel_estimator.parameters():
            param.requires_grad = False
        self.ref_vel_estimator_obs_shape = (self.ref_vel_estimator.num_obs,)
        print("[FrontRESUnified] Reference velocity estimator loaded and frozen")

    @staticmethod
    def _collect_trainable_params(policy):
        if isinstance(policy, (ResidualActorCritic, FrontRESActorCritic)):
            params = list(policy.residual_actor.parameters())
            acceptance_actor = getattr(policy, "acceptance_actor", None)
            has_split_acceptance = acceptance_actor is not None
            if has_split_acceptance:
                params.extend(acceptance_actor.parameters())
            state_router_head = getattr(policy, "state_router_head", None)
            has_state_router = state_router_head is not None
            if has_state_router:
                params.extend(state_router_head.parameters())
            authority_actor = getattr(policy, "authority_actor", None)
            has_authority_actor = authority_actor is not None
            if has_authority_actor:
                params.extend(authority_actor.parameters())
            authority_critic = getattr(policy, "authority_critic", None)
            has_authority_critic = authority_critic is not None
            if has_authority_critic:
                params.extend(authority_critic.parameters())
            params.extend(policy.critic.parameters())
            has_trainable_std = False
            if hasattr(policy, "std") and getattr(policy.std, "requires_grad", False):
                params.append(policy.std)
                has_trainable_std = True
            elif hasattr(policy, "log_std") and getattr(policy.log_std, "requires_grad", False):
                params.append(policy.log_std)
                has_trainable_std = True
            suffix = " + policy std" if has_trainable_std else " (fixed policy std)"
            actor_parts = ["residual_actor"]
            if has_split_acceptance:
                actor_parts.append("acceptance_actor")
            if has_state_router:
                actor_parts.append("state_router_head")
            if has_authority_actor:
                actor_parts.append("authority_actor")
            if has_authority_critic:
                actor_parts.append("authority_critic")
            actor_desc = " + ".join(actor_parts)
            print(f"[FrontRESUnified] Optimizer updates {actor_desc} + critic{suffix}")
            return params
        print("[FrontRESUnified] Optimizer updates full policy")
        return policy.parameters()

    def _print_init_summary(self):
        print("=" * 80)
        print("  FrontRESUnified ▸ PPO + Supervised ΔSE3")
        print(f"  Objective={self.frontres_training_objective}")
        if self._authority_actor_critic_enabled():
            print(
                "  L = L_authority_actor_critic(rho | obs, detached ΔSE proposal) "
                "+ λ_sup·L_HSL(proposal)"
            )
        elif self.frontres_training_objective == "supervised_restore":
            print("  L = L_supervised_restore  (PPO/HRL branch kept but disabled for updates)")
        elif self.frontres_training_objective == "basis_restore":
            print("  L = L_proposal + L_written + L_coeff  "
                  "(factorized per-axis repair coefficients; PPO/HRL branch kept but disabled)")
        elif self.frontres_training_objective == "segment_replay_hrl":
            print("  L = Segment Replay HRL  (dedicated runner loop; legacy update disabled)")
        elif self.frontres_training_objective == "hsl_hybrid":
            print("  L = L_PPO(6D acceptance) + λ_sup·L_HSL(proposal)  "
                  "(PPO log-prob is restricted to acceptance; ΔSE proposal is supervised)")
        else:
            print(f"  L = L_PPO + λ_sup({self.lambda_supervised:.2f})·L_supervised")
        print("=" * 80)
        print(f"  LR={self.learning_rate}  clip={self.clip_param}  ent_coef={self.entropy_coef}")
        print(f"  epochs={self.num_learning_epochs}  mini_batches={self.num_mini_batches}")
        print(f"  Supervised  λ={self.lambda_supervised:.3f} → {self.lambda_supervised_min}"
              f"  decay={self.lambda_supervised_decay_rate}"
              f"  trigger_cos={self.supervised_trigger_cosine_sim}"
              f"  rpy_w={self.supervised_rpy_loss_weight}"
              f"  conf_w={self.supervised_conf_loss_weight}"
              f"  dir_w={self.supervised_direction_loss_weight}"
              f"  mag_w={self.supervised_magnitude_loss_weight}"
              f"  over_w={self.supervised_over_loss_weight}"
              f"  smooth_w={self.supervised_smooth_loss_weight}"
              f"  valid_w={self.supervised_valid_loss_weight}")
        print(f"  PPO actor warmup={self.ppo_actor_warmup_iterations}"
              f"  ramp={self.ppo_actor_ramp_iterations}"
              f"  adv_focal_power={self.ppo_advantage_focal_power}")
        if self._structured_joint_rl_enabled():
            print(
                "  Structured Joint RL: rho-only constrained retention "
                f"(weight={self.frontres_structured_joint_rl_weight}, "
                f"floor=runner-adaptive U_floor, fallback={self.frontres_structured_joint_exec_floor}, "
                f"dir_w={self.frontres_structured_joint_directional_weight}, "
                f"under_w={self.frontres_structured_joint_underwrite_weight}, "
                f"repair={self.frontres_structured_joint_repair_loss_kind}, "
                f"rscale={self.frontres_structured_joint_repair_loss_scale}, "
                f"rho_center={self.frontres_structured_joint_rho_center}, "
                f"ret_prior_w={self.frontres_structured_joint_retention_prior_weight}, "
                f"floor_w={self.frontres_structured_joint_floor_penalty_weight}, "
                f"full_w={self.frontres_structured_joint_full_repair_bonus_weight}, "
                f"loss_mode={self.frontres_structured_joint_rl_loss_mode}, "
                f"normalize_adv={self.frontres_structured_joint_rl_normalize_advantage})"
            )
        if self._authority_actor_critic_enabled():
            print(
                "  Authority Actor-Critic: "
                f"actor_w={self.frontres_authority_actor_loss_weight}, "
                f"critic_w={self.frontres_authority_critic_loss_weight}, "
                f"actor_warmup={self.frontres_authority_actor_warmup_iterations}, "
                f"actor_ramp={self.frontres_authority_actor_ramp_iterations}, "
                "target=K-step executable return"
            )
        print("  MOSAIC teacher/off-policy branches: disabled by construction")
        print("=" * 80)

    def _cuda_memory_debug_enabled(self) -> bool:
        return (
            bool(getattr(self, "frontres_cuda_memory_debug", False))
            and torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        )

    @staticmethod
    def _cuda_memory_debug_should_print(label: str, update_idx: int | None) -> bool:
        # Keep the live OOM sentinel useful without flooding every mini-batch.
        if label == "update_entry" or label.startswith("oom_"):
            return True
        if update_idx != 0:
            return False
        return label in {
            "value_backward_after",
            "actor_supervised_backward_after",
            "rho_backward_after",
        }

    def _print_cuda_memory_debug(
        self,
        label: str,
        *,
        update_idx: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        if not self._cuda_memory_debug_enabled():
            return
        if not self._cuda_memory_debug_should_print(label, update_idx):
            return
        try:
            device = torch.device(self.device)
            torch.cuda.synchronize(device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            max_allocated = torch.cuda.max_memory_allocated(device)
            max_reserved = torch.cuda.max_memory_reserved(device)
        except Exception as exc:
            print(f"[FrontRES CUDA mem] label={label} unavailable: {exc}", flush=True)
            return

        def _gib(value: int) -> float:
            return float(value) / (1024.0 ** 3)

        it = int(getattr(self, "current_learning_iteration", 0))
        idx_text = "n/a" if update_idx is None else str(update_idx)
        batch_text = "n/a" if batch_size is None else str(batch_size)
        print(
            "[FrontRES CUDA mem] "
            f"it={it} label={label} update_idx={idx_text} "
            f"batch={batch_text} epochs={self.num_learning_epochs} "
            f"mini_batches={self.num_mini_batches} "
            f"alloc={_gib(allocated):.2f}GiB "
            f"reserved={_gib(reserved):.2f}GiB "
            f"max_alloc={_gib(max_allocated):.2f}GiB "
            f"max_reserved={_gib(max_reserved):.2f}GiB "
            f"free={_gib(free_bytes):.2f}GiB "
            f"total={_gib(total_bytes):.2f}GiB",
            flush=True,
        )

    @staticmethod
    def _frontres_env_int(name: str, default: int) -> int:
        raw = os.environ.get(name, "")
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _maybe_dump_frontres_live_batch(
        self,
        *,
        update_idx: int,
        obs_batch,
        mu_batch,
        actions_batch,
        old_mu_batch,
        old_sigma_batch,
        acceptance_target_batch,
        acceptance_mask_batch,
        rho_prior_authority_batch,
        rho_prior_target_batch,
        structured_metrics,
        original_batch_size: int,
        proposal_delta_se_batch=None,
        authority_action_batch=None,
        authority_return_k_batch=None,
        authority_return_zero_k_batch=None,
        authority_return_one_k_batch=None,
        authority_mask_batch=None,
    ) -> None:
        """Optionally dump one formal live minibatch for offline FrontRES replay tests."""
        dump_path = os.environ.get("FRONTRES_LIVE_BATCH_DUMP", "")
        if not dump_path:
            return
        if bool(getattr(self, "_frontres_live_batch_dump_done", False)):
            return

        current_it = int(getattr(self, "current_learning_iteration", 0))
        target_it_raw = os.environ.get("FRONTRES_LIVE_BATCH_DUMP_IT", "")
        if target_it_raw:
            try:
                if current_it != int(target_it_raw):
                    return
            except ValueError:
                pass

        target_update = self._frontres_env_int("FRONTRES_LIVE_BATCH_DUMP_UPDATE", 0)
        if update_idx != target_update:
            return

        n = int(original_batch_size)
        max_samples = self._frontres_env_int("FRONTRES_LIVE_BATCH_DUMP_MAX", 20000)
        max_samples = max(1, min(max_samples, n))
        sample_slice = slice(0, max_samples)
        conf_dim = int(getattr(self.policy, "task_conf_dim", 1))
        conf_dim = max(1, min(conf_dim, acceptance_target_batch.shape[-1]))
        rho_cols = slice(6, 6 + conf_dim)

        def _cpu(x):
            if x is None:
                return None
            return x.detach().to("cpu")

        payload = {
            "kind": "frontres_live_batch_rho_replay",
            "iteration": current_it,
            "update_idx": int(update_idx),
            "num_samples": int(max_samples),
            "original_batch_size": int(n),
            "obs": _cpu(obs_batch[:n][sample_slice]),
            "rho_mean_raw": _cpu(mu_batch[:n, rho_cols][sample_slice]),
            "rho_mean": _cpu(torch.sigmoid(mu_batch[:n, rho_cols])[sample_slice]),
            "rho_action": _cpu(actions_batch[:n, rho_cols][sample_slice]),
            "old_rho_mean_raw": _cpu(old_mu_batch[:n, rho_cols][sample_slice]),
            "old_rho_sigma": _cpu(old_sigma_batch[:n, rho_cols][sample_slice]),
            "rho_advantage": _cpu(acceptance_target_batch[:n, :conf_dim][sample_slice]),
            "rho_weight": _cpu(acceptance_mask_batch[:n, :conf_dim][sample_slice]),
            "rho_prior_authority": _cpu(rho_prior_authority_batch[:n][sample_slice])
            if rho_prior_authority_batch is not None else None,
            "rho_prior_target": _cpu(rho_prior_target_batch[:n, :conf_dim][sample_slice])
            if rho_prior_target_batch is not None else None,
            "proposal_delta_se": _cpu(proposal_delta_se_batch[:n][sample_slice])
            if proposal_delta_se_batch is not None else None,
            "authority_action": _cpu(authority_action_batch[:n][sample_slice])
            if authority_action_batch is not None else None,
            "authority_return_k": _cpu(authority_return_k_batch[:n][sample_slice])
            if authority_return_k_batch is not None else None,
            "authority_return_zero_k": _cpu(authority_return_zero_k_batch[:n][sample_slice])
            if authority_return_zero_k_batch is not None else None,
            "authority_return_one_k": _cpu(authority_return_one_k_batch[:n][sample_slice])
            if authority_return_one_k_batch is not None else None,
            "authority_mask": _cpu(authority_mask_batch[:n][sample_slice])
            if authority_mask_batch is not None else None,
            "config": {
                "task_conf_dim": conf_dim,
                "loss_mode": str(getattr(self, "frontres_structured_joint_rl_loss_mode", "")),
                "repair_loss_kind": str(getattr(self, "frontres_structured_joint_repair_loss_kind", "")),
                "repair_loss_scale": float(getattr(self, "frontres_structured_joint_repair_loss_scale", 1.0)),
                "prior_loss_weight": float(getattr(self, "frontres_structured_joint_prior_loss_weight", 0.0)),
                "adv_clip": float(getattr(self, "frontres_structured_joint_rl_adv_clip", 0.0)),
                "normalize_advantage": bool(
                    getattr(self, "frontres_structured_joint_rl_normalize_advantage", False)
                ),
                "authority_actor_critic_enabled": bool(self._authority_actor_critic_enabled()),
                "authority_return_horizon": int(getattr(self, "frontres_authority_return_horizon", 1)),
            },
            "live_metrics": {
                key: float(value)
                for key, value in structured_metrics.items()
                if isinstance(value, (int, float))
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
        torch.save(payload, dump_path)
        self._frontres_live_batch_dump_done = True
        print(
            "[FrontRES live batch dump] "
            f"path={dump_path} it={current_it} update_idx={update_idx} "
            f"samples={max_samples}/{n} conf_dim={conf_dim}",
            flush=True,
        )

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        teacher_obs_shape=None,
        ref_vel_estimator_obs_shape=None,
    ):
        if training_type != "frontres":
            raise ValueError(f"FrontRESUnified storage must use training_type='frontres', got {training_type!r}.")
        self.ref_vel_estimator_obs_shape = ref_vel_estimator_obs_shape
        self.storage = RolloutStorage(
            "frontres",
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            None,
            self.device,
            teacher_obs_shape=None,
            ref_vel_estimator_obs_shape=ref_vel_estimator_obs_shape,
        )
        self.storage.yield_batch_indices = self.supervised_smooth_loss_weight > 0

    def act(self, obs, critic_obs, teacher_obs=None, ref_vel_estimator_obs=None, motion_groups=None):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        if self.use_estimate_ref_vel and self.ref_vel_estimator is not None:
            estimator_input = ref_vel_estimator_obs if ref_vel_estimator_obs is not None else obs
            with torch.no_grad():
                estimated_ref_vel = self.ref_vel_estimator(estimator_input)
                self.last_estimated_ref_vel = estimated_ref_vel.clone()
                obs_augmented = torch.cat([obs, estimated_ref_vel], dim=-1)
        else:
            obs_augmented = obs
            self.last_estimated_ref_vel = None

        self.transition.actions = self.policy.act(obs_augmented).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self._get_actor_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.transition.ref_vel_estimator_observations = ref_vel_estimator_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch)

    def update(self):
        if (
            self.frontres_training_objective == "segment_replay_hrl"
            and (
                self.frontres_segment_live_sentinel_only
                or self.frontres_segment_live_probe_only
                or self.frontres_segment_live_storage_write_only
                or self.frontres_segment_live_single_update_only
                or self.frontres_segment_live_update_loop_only
                or self.frontres_segment_live_train_enabled
            )
        ):
            raise NotImplementedError(
                "Stage 3 Segment Replay live mode reached FrontRESUnified.update; "
                "use the dedicated Segment Replay runner loop instead of the legacy full update path."
            )
        self._update_supervised_learning_rate()
        loss_dict = self._update_ppo_supervised()
        if (
            self.frontres_training_objective not in ("supervised_restore", "basis_restore")
            and not bool(getattr(self, "state_supervised_controller_enabled", False))
        ):
            self._step_supervised_lambda(loss_dict.get("supervised_cos_sim", 0.0))
        return loss_dict

    def _update_supervised_learning_rate(self) -> None:
        if self.frontres_training_objective not in ("supervised_restore", "basis_restore", "hsl_hybrid"):
            return
        if self.frontres_supervised_lr_schedule not in ("cosine", "cosine_anneal", "cosine_annealing"):
            return

        it = int(getattr(self, "current_learning_iteration", 0))
        warmup = max(0, self.frontres_supervised_lr_warmup_iters)
        cosine_iters = max(1, self.frontres_supervised_lr_cosine_iters)
        lr_start = self.frontres_supervised_lr_start
        lr_peak = self.frontres_supervised_lr_peak
        lr_min = self.frontres_supervised_lr_min

        if warmup > 0 and it < warmup:
            frac = it / float(max(1, warmup))
            lr = lr_start + (lr_peak - lr_start) * frac
        else:
            frac = min(1.0, max(0.0, (it - warmup) / float(cosine_iters)))
            lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + math.cos(math.pi * frac))

        self.learning_rate = float(lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _step_supervised_lambda(self, cos_sim: float):
        if self.lambda_supervised <= self.lambda_supervised_min:
            return
        self._supervised_cosine_ema = (
            (1.0 - self._supervised_ema_alpha) * self._supervised_cosine_ema
            + self._supervised_ema_alpha * cos_sim
        )
        if not self._supervised_decay_triggered:
            if self._supervised_cosine_ema >= self.supervised_trigger_cosine_sim:
                self._supervised_decay_triggered = True
                print(f"[FrontRESUnified] Supervised λ decay triggered: "
                      f"cos_sim_ema={self._supervised_cosine_ema:.3f} >= "
                      f"{self.supervised_trigger_cosine_sim:.3f}")

        if self._supervised_decay_triggered:
            self.lambda_supervised = max(
                self.lambda_supervised * self.lambda_supervised_decay_rate,
                self.lambda_supervised_min,
            )

    def _update_ppo_supervised(self):
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        if self._cuda_memory_debug_enabled():
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
            self._print_cuda_memory_debug("update_entry")
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_supervised_loss = 0.0
        mean_supervised_cos_sim = 0.0
        mean_supervised_metrics: dict[str, float] = {}
        mean_acceptance_preference_loss = 0.0
        mean_acceptance_preference_metrics: dict[str, float] = {}
        mean_state_alpha_loss = 0.0
        mean_state_alpha_metrics: dict[str, float] = {}
        mean_structured_joint_rl_loss = 0.0
        mean_structured_joint_rl_metrics: dict[str, float] = {}
        mean_authority_loss = 0.0
        mean_authority_metrics: dict[str, float] = {}
        grad_conflict_cos = 0.0
        grad_conflict_norm_ratio = 0.0
        grad_conflict_count = 0
        grad_diag_sums: dict[str, float] = {}
        grad_diag_count = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for update_idx, batch in enumerate(generator):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
                _teacher_obs_batch,
                _teacher_mu_batch,
                _teacher_sigma_batch,
                ref_vel_estimator_obs_batch,
                _motion_groups_batch,
                frontres_mask_batch,
                supervised_target_batch,
                frontres_actor_gate_batch,
                supervised_weight_batch,
                supervised_harm_weight_batch,
                acceptance_target_batch,
                acceptance_mask_batch,
                rho_prior_authority_batch,
                rho_prior_target_batch,
                state_alpha_target_batch,
                state_alpha_mask_batch,
            ) = batch[:28]
            proposal_delta_se_batch = None
            authority_action_batch = None
            authority_log_prob_batch = None
            authority_rho_batch = None
            authority_return_k_batch = None
            authority_return_zero_k_batch = None
            authority_return_one_k_batch = None
            authority_mask_batch = None
            acceptance_action_batch = None
            acceptance_logit_batch = None
            acceptance_prob_batch = None
            acceptance_gt_batch = None
            acceptance_margin_batch = None
            if len(batch) >= 41:
                (
                    proposal_delta_se_batch,
                    authority_action_batch,
                    authority_log_prob_batch,
                    authority_rho_batch,
                    authority_return_k_batch,
                    authority_return_zero_k_batch,
                    authority_return_one_k_batch,
                    authority_mask_batch,
                ) = batch[28:36]
                (
                    acceptance_action_batch,
                    acceptance_logit_batch,
                    acceptance_prob_batch,
                    acceptance_gt_batch,
                    acceptance_margin_batch,
                ) = batch[36:41]
                batch_indices = batch[41] if len(batch) > 41 else None
            elif len(batch) >= 36:
                (
                    proposal_delta_se_batch,
                    authority_action_batch,
                    authority_log_prob_batch,
                    authority_rho_batch,
                    authority_return_k_batch,
                    authority_return_zero_k_batch,
                    authority_return_one_k_batch,
                    authority_mask_batch,
                ) = batch[28:36]
                batch_indices = batch[36] if len(batch) > 36 else None
            elif len(batch) >= 34:
                (
                    proposal_delta_se_batch,
                    authority_action_batch,
                    authority_log_prob_batch,
                    authority_rho_batch,
                    authority_return_k_batch,
                    authority_mask_batch,
                ) = batch[28:34]
                batch_indices = batch[34] if len(batch) > 34 else None
            else:
                batch_indices = batch[28] if len(batch) > 28 else None
            if acceptance_gt_batch is not None:
                acceptance_target_batch = acceptance_gt_batch
            original_batch_size = obs_batch.shape[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.use_estimate_ref_vel and self.ref_vel_estimator is not None:
                with torch.no_grad():
                    estimator_input = ref_vel_estimator_obs_batch if ref_vel_estimator_obs_batch is not None else obs_batch
                    estimated_ref_vel_batch = self.ref_vel_estimator(estimator_input)
                    obs_batch_augmented = torch.cat([obs_batch, estimated_ref_vel_batch], dim=-1)
            else:
                obs_batch_augmented = obs_batch

            structured_rho_active = self._structured_joint_rl_enabled()
            structured_loss_mode = str(
                getattr(self, "frontres_structured_joint_rl_loss_mode", "ppo_clipped")
            ).lower()
            oracle_mix = float(getattr(self, "oracle_mix", 0.0))
            raw_ppo_weight = float(getattr(self, "ppo_actor_weight", 1.0)) * (1.0 - oracle_mix)
            authority_active = self._authority_actor_critic_enabled()
            hsl_acceptance_active = self._active_hsl_acceptance_loss_enabled()
            disable_generic_ppo = (
                authority_active
                or hsl_acceptance_active
                or (
                    structured_rho_active
                    and bool(getattr(self, "frontres_structured_joint_rl_disable_generic_ppo", True))
                )
            )
            # In region-direct structured rho mode, the rho update is a direct
            # logit loss.  It does not use PPO log-prob ratios, so building the
            # log-prob graph only raises the update memory peak.
            ppo_weight = 0.0 if disable_generic_ppo else raw_ppo_weight
            needs_actor_log_prob = (
                self.frontres_training_objective not in ("supervised_restore", "basis_restore")
                and (
                    ppo_weight > 0.0
                    or (structured_rho_active and structured_loss_mode != "region_direct")
                )
            )
            memory_safe_region_direct = (
                structured_rho_active
                and structured_loss_mode == "region_direct"
                and disable_generic_ppo
                and self._ppo_acceptance_only_mode()
            )

            self.optimizer.zero_grad(set_to_none=True)

            if memory_safe_region_direct:
                self._print_cuda_memory_debug(
                    "value_forward_before",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                value_batch = self.policy.evaluate(
                    critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                self._print_cuda_memory_debug(
                    "value_forward_after",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                value_loss = self._compute_value_loss(
                    target_values_batch,
                    returns_batch,
                    value_batch,
                    frontres_mask_batch,
                )
                value_term = self.value_loss_coef * value_loss
                if not torch.isfinite(value_term):
                    self._warn_skip("non-finite value loss", value_term)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self._print_cuda_memory_debug(
                    "value_backward_before",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                try:
                    if value_term.requires_grad:
                        value_term.backward()
                except torch.cuda.OutOfMemoryError:
                    self._print_cuda_memory_debug(
                        "oom_value_backward",
                        update_idx=update_idx,
                        batch_size=original_batch_size,
                    )
                    raise
                self._print_cuda_memory_debug(
                    "value_backward_after",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                value_loss_item = value_loss.item()
                del value_batch, value_term

                self._print_cuda_memory_debug(
                    "actor_supervised_forward_before",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                self.policy.update_distribution(obs_batch_augmented)
                self._print_cuda_memory_debug(
                    "actor_supervised_forward_after",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                actions_log_prob_batch = torch.zeros(
                    original_batch_size,
                    device=self.device,
                    dtype=obs_batch.dtype,
                )
                mu_batch = self.policy.action_mean[:original_batch_size]
                sigma_batch = self.policy.action_std[:original_batch_size]
                entropy_batch = self.policy.entropy[:original_batch_size]
                if (
                    self.frontres_training_objective != "supervised_restore"
                    and self.frontres_training_objective != "basis_restore"
                    and self.desired_kl is not None
                    and self.schedule == "adaptive"
                ):
                    self._adapt_learning_rate(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)
                supervised_loss, sup_cos_sim, sup_metrics = self._compute_supervised_loss(
                    mu_batch,
                    supervised_target_batch,
                    original_batch_size,
                    batch_indices=batch_indices,
                    supervised_weight_batch=supervised_weight_batch,
                    supervised_harm_weight_batch=supervised_harm_weight_batch,
                )
                acceptance_preference_loss, acceptance_preference_metrics = self._compute_acceptance_preference_loss(
                    mu_batch,
                    acceptance_target_batch,
                    acceptance_mask_batch,
                    original_batch_size,
                )
                state_alpha_loss, state_alpha_metrics = self._compute_state_alpha_loss(
                    obs_batch_augmented,
                    state_alpha_target_batch,
                    state_alpha_mask_batch,
                    original_batch_size,
                )
                legacy_preference_weight = 0.0
                legacy_alpha_weight = float(getattr(self, "frontres_state_alpha_weight", 0.0))
                non_acceptance_loss = (
                    -self.entropy_coef * entropy_batch.mean()
                    + self.lambda_supervised * supervised_loss
                    + legacy_preference_weight * acceptance_preference_loss
                    + legacy_alpha_weight * state_alpha_loss
                )
                if not torch.isfinite(non_acceptance_loss):
                    self._warn_skip("non-finite non-acceptance loss", non_acceptance_loss)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self._print_cuda_memory_debug(
                    "actor_supervised_backward_before",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                try:
                    if non_acceptance_loss.requires_grad:
                        non_acceptance_loss.backward()
                except torch.cuda.OutOfMemoryError:
                    self._print_cuda_memory_debug(
                        "oom_actor_supervised_backward",
                        update_idx=update_idx,
                        batch_size=original_batch_size,
                    )
                    raise
                self._print_cuda_memory_debug(
                    "actor_supervised_backward_after",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                base_grads = {
                    p: (p.grad.detach().clone() if p.grad is not None else None)
                    for p in self.policy.parameters()
                }
                entropy_item = entropy_batch.mean().item()
                supervised_loss_item = supervised_loss.item()
                acceptance_preference_loss_item = acceptance_preference_loss.item()
                state_alpha_loss_item = state_alpha_loss.item()
                del (
                    mu_batch,
                    sigma_batch,
                    entropy_batch,
                    supervised_loss,
                    acceptance_preference_loss,
                    state_alpha_loss,
                    non_acceptance_loss,
                )

                self._print_cuda_memory_debug(
                    "rho_forward_before",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                self.policy.update_distribution(obs_batch_augmented)
                self._print_cuda_memory_debug(
                    "rho_forward_after",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                mu_batch = self.policy.action_mean[:original_batch_size]
                sigma_batch = self.policy.action_std[:original_batch_size]
                structured_joint_rl_loss, structured_joint_rl_metrics = self._compute_structured_joint_rl_loss(
                    obs_batch_augmented,
                    mu_batch,
                    actions_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    actions_log_prob_batch,
                    old_actions_log_prob_batch,
                    acceptance_target_batch,
                    acceptance_mask_batch,
                    rho_prior_authority_batch,
                    rho_prior_target_batch,
                    original_batch_size,
                )
                self._maybe_dump_frontres_live_batch(
                    update_idx=update_idx,
                    obs_batch=obs_batch_augmented,
                    mu_batch=mu_batch,
                    actions_batch=actions_batch,
                    old_mu_batch=old_mu_batch,
                    old_sigma_batch=old_sigma_batch,
                    acceptance_target_batch=acceptance_target_batch,
                    acceptance_mask_batch=acceptance_mask_batch,
                    rho_prior_authority_batch=rho_prior_authority_batch,
                    rho_prior_target_batch=rho_prior_target_batch,
                    structured_metrics=structured_joint_rl_metrics,
                    original_batch_size=original_batch_size,
                    proposal_delta_se_batch=proposal_delta_se_batch,
                    authority_action_batch=authority_action_batch,
                    authority_return_k_batch=authority_return_k_batch,
                    authority_return_zero_k_batch=authority_return_zero_k_batch,
                    authority_return_one_k_batch=authority_return_one_k_batch,
                    authority_mask_batch=authority_mask_batch,
                )
                acceptance_only_loss = self.frontres_structured_joint_rl_weight * structured_joint_rl_loss
                if not torch.isfinite(acceptance_only_loss):
                    self._warn_skip("non-finite acceptance loss", acceptance_only_loss)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self._print_cuda_memory_debug(
                    "rho_backward_before",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                try:
                    if acceptance_only_loss.requires_grad:
                        acceptance_only_loss.backward()
                except torch.cuda.OutOfMemoryError:
                    self._print_cuda_memory_debug(
                        "oom_rho_backward",
                        update_idx=update_idx,
                        batch_size=original_batch_size,
                    )
                    raise
                self._print_cuda_memory_debug(
                    "rho_backward_after",
                    update_idx=update_idx,
                    batch_size=original_batch_size,
                )
                self._keep_ppo_grad_on_acceptance_head_only(base_grads)
                grad_diag = self._compute_acceptance_grad_diagnostics(base_grads)
                if grad_diag:
                    for key, value in grad_diag.items():
                        grad_diag_sums[key] = grad_diag_sums.get(key, 0.0) + float(value)
                    grad_diag_count += 1

                if self.is_multi_gpu:
                    self.reduce_parameters()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in self.policy.parameters() if p.requires_grad):
                    self._warn_skip("NaN gradient detected")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss_item
                mean_surrogate_loss += 0.0
                mean_entropy += entropy_item
                mean_supervised_loss += supervised_loss_item
                mean_supervised_cos_sim += sup_cos_sim
                for key, value in sup_metrics.items():
                    mean_supervised_metrics[key] = mean_supervised_metrics.get(key, 0.0) + float(value)
                mean_acceptance_preference_loss += acceptance_preference_loss_item
                for key, value in acceptance_preference_metrics.items():
                    mean_acceptance_preference_metrics[key] = (
                        mean_acceptance_preference_metrics.get(key, 0.0) + float(value)
                    )
                mean_state_alpha_loss += state_alpha_loss_item
                for key, value in state_alpha_metrics.items():
                    mean_state_alpha_metrics[key] = mean_state_alpha_metrics.get(key, 0.0) + float(value)
                mean_structured_joint_rl_loss += structured_joint_rl_loss.item()
                for key, value in structured_joint_rl_metrics.items():
                    mean_structured_joint_rl_metrics[key] = (
                        mean_structured_joint_rl_metrics.get(key, 0.0) + float(value)
                    )
                del (
                    mu_batch,
                    sigma_batch,
                    structured_joint_rl_loss,
                    acceptance_only_loss,
                    base_grads,
                )
                continue

            self.policy.update_distribution(obs_batch_augmented)
            if needs_actor_log_prob:
                actions_log_prob_batch = self._get_actor_log_prob(actions_batch)
            else:
                actions_log_prob_batch = torch.zeros(
                    original_batch_size,
                    device=self.device,
                    dtype=obs_batch.dtype,
                )
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if (
                self.frontres_training_objective != "supervised_restore"
                and self.frontres_training_objective != "basis_restore"
                and self.desired_kl is not None
                and self.schedule == "adaptive"
            ):
                self._adapt_learning_rate(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)

            supervised_loss, sup_cos_sim, sup_metrics = self._compute_supervised_loss(
                mu_batch,
                supervised_target_batch,
                original_batch_size,
                batch_indices=batch_indices,
                supervised_weight_batch=supervised_weight_batch,
                supervised_harm_weight_batch=supervised_harm_weight_batch,
            )
            acceptance_preference_loss, acceptance_preference_metrics = self._compute_acceptance_preference_loss(
                mu_batch,
                acceptance_target_batch,
                acceptance_mask_batch,
                original_batch_size,
            )
            state_alpha_loss, state_alpha_metrics = self._compute_state_alpha_loss(
                obs_batch_augmented,
                state_alpha_target_batch,
                state_alpha_mask_batch,
                original_batch_size,
            )
            structured_joint_rl_loss, structured_joint_rl_metrics = self._compute_structured_joint_rl_loss(
                obs_batch_augmented,
                mu_batch,
                actions_batch,
                old_mu_batch,
                old_sigma_batch,
                actions_log_prob_batch,
                old_actions_log_prob_batch,
                acceptance_target_batch,
                acceptance_mask_batch,
                rho_prior_authority_batch,
                rho_prior_target_batch,
                original_batch_size,
            )
            authority_loss, authority_metrics = self._compute_authority_actor_critic_loss(
                obs_batch_augmented,
                proposal_delta_se_batch,
                authority_action_batch,
                authority_return_k_batch,
                authority_return_zero_k_batch,
                authority_return_one_k_batch,
                authority_mask_batch,
                original_batch_size,
            )
            self._maybe_dump_frontres_live_batch(
                update_idx=update_idx,
                obs_batch=obs_batch_augmented,
                mu_batch=mu_batch,
                actions_batch=actions_batch,
                old_mu_batch=old_mu_batch,
                old_sigma_batch=old_sigma_batch,
                acceptance_target_batch=acceptance_target_batch,
                acceptance_mask_batch=acceptance_mask_batch,
                rho_prior_authority_batch=rho_prior_authority_batch,
                rho_prior_target_batch=rho_prior_target_batch,
                structured_metrics=structured_joint_rl_metrics,
                original_batch_size=original_batch_size,
                proposal_delta_se_batch=proposal_delta_se_batch,
                authority_action_batch=authority_action_batch,
                authority_return_k_batch=authority_return_k_batch,
                authority_return_zero_k_batch=authority_return_zero_k_batch,
                authority_return_one_k_batch=authority_return_one_k_batch,
                authority_mask_batch=authority_mask_batch,
            )

            if self.frontres_training_objective in ("supervised_restore", "basis_restore"):
                loss = supervised_loss
                value_loss = torch.zeros((), device=self.device)
                surrogate_loss = torch.zeros((), device=self.device)
                ppo_weight = 0.0
            else:
                if ppo_weight > 0.0:
                    surrogate_loss, value_loss = self._compute_ppo_losses(
                        actions_log_prob_batch,
                        old_actions_log_prob_batch,
                        advantages_batch,
                        target_values_batch,
                        returns_batch,
                        value_batch,
                        frontres_mask_batch,
                        frontres_actor_gate_batch,
                    )
                else:
                    surrogate_loss = torch.zeros((), device=self.device)
                    value_loss = self._compute_value_loss(
                        target_values_batch,
                        returns_batch,
                        value_batch,
                        frontres_mask_batch,
                    )
                if self.diagnose_gradient_conflict and ppo_weight > 0.0 and self.lambda_supervised > 0.0:
                    _gc, _ratio = self._compute_actor_grad_conflict(
                        surrogate_loss, supervised_loss)
                    if _gc is not None:
                        grad_conflict_cos += _gc
                        grad_conflict_norm_ratio += _ratio
                        grad_conflict_count += 1
                legacy_preference_weight = float(getattr(self, "frontres_acceptance_preference_weight", 0.0))
                legacy_alpha_weight = float(getattr(self, "frontres_state_alpha_weight", 0.0))
                structured_weight = (
                    self.frontres_structured_joint_rl_weight
                    if structured_rho_active
                    else 0.0
                )
                if structured_rho_active and not self.frontres_structured_joint_rl_keep_legacy_bce:
                    legacy_preference_weight = 0.0
                loss = (
                    ppo_weight * surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + self.lambda_supervised * supervised_loss
                    + legacy_preference_weight * acceptance_preference_loss
                    + legacy_alpha_weight * state_alpha_loss
                    + structured_weight * structured_joint_rl_loss
                    + authority_loss
                )

            if not torch.isfinite(loss):
                self._warn_skip("non-finite loss", loss)
                continue

            preference_weight = float(getattr(self, "frontres_acceptance_preference_weight", 0.0))
            structured_weight = (
                self.frontres_structured_joint_rl_weight
                if self._structured_joint_rl_enabled()
                else 0.0
            )
            if self._structured_joint_rl_enabled() and not self.frontres_structured_joint_rl_keep_legacy_bce:
                preference_weight = 0.0
            if self._ppo_acceptance_only_mode() and (
                ppo_weight > 0.0 or preference_weight > 0.0 or structured_weight > 0.0
            ):
                acceptance_only_loss = (
                    ppo_weight * surrogate_loss
                    + preference_weight * acceptance_preference_loss
                    + structured_weight * structured_joint_rl_loss
                )
                non_ppo_loss = loss - acceptance_only_loss
                if non_ppo_loss.requires_grad:
                    non_ppo_loss.backward(retain_graph=True)
                base_grads = {
                    p: (p.grad.detach().clone() if p.grad is not None else None)
                    for p in self.policy.parameters()
                }
                if acceptance_only_loss.requires_grad:
                    acceptance_only_loss.backward()
                    self._keep_ppo_grad_on_acceptance_head_only(base_grads)
                    grad_diag = self._compute_acceptance_grad_diagnostics(base_grads)
                    if grad_diag:
                        for key, value in grad_diag.items():
                            grad_diag_sums[key] = grad_diag_sums.get(key, 0.0) + float(value)
                        grad_diag_count += 1
                else:
                    for p in self.policy.parameters():
                        base = base_grads.get(p)
                        p.grad = None if base is None else base.clone()
            else:
                loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()

            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in self.policy.parameters() if p.requires_grad):
                self._warn_skip("NaN gradient detected")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_supervised_loss += supervised_loss.item()
            mean_supervised_cos_sim += sup_cos_sim
            for key, value in sup_metrics.items():
                mean_supervised_metrics[key] = mean_supervised_metrics.get(key, 0.0) + float(value)
            mean_acceptance_preference_loss += acceptance_preference_loss.item()
            for key, value in acceptance_preference_metrics.items():
                mean_acceptance_preference_metrics[key] = (
                    mean_acceptance_preference_metrics.get(key, 0.0) + float(value)
                )
            mean_state_alpha_loss += state_alpha_loss.item()
            for key, value in state_alpha_metrics.items():
                mean_state_alpha_metrics[key] = mean_state_alpha_metrics.get(key, 0.0) + float(value)
            mean_structured_joint_rl_loss += structured_joint_rl_loss.item()
            for key, value in structured_joint_rl_metrics.items():
                mean_structured_joint_rl_metrics[key] = (
                    mean_structured_joint_rl_metrics.get(key, 0.0) + float(value)
                )
            mean_authority_loss += authority_loss.item()
            for key, value in authority_metrics.items():
                mean_authority_metrics[key] = mean_authority_metrics.get(key, 0.0) + float(value)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_supervised_loss /= num_updates
        mean_supervised_cos_sim /= num_updates
        mean_supervised_metrics = {
            key: value / num_updates for key, value in mean_supervised_metrics.items()
        }
        mean_acceptance_preference_loss /= num_updates
        mean_acceptance_preference_metrics = {
            key: value / num_updates for key, value in mean_acceptance_preference_metrics.items()
        }
        mean_state_alpha_loss /= num_updates
        mean_state_alpha_metrics = {
            key: value / num_updates for key, value in mean_state_alpha_metrics.items()
        }
        mean_structured_joint_rl_loss /= num_updates
        mean_structured_joint_rl_metrics = {
            key: value / num_updates for key, value in mean_structured_joint_rl_metrics.items()
        }
        mean_authority_loss /= num_updates
        mean_authority_metrics = {
            key: value / num_updates for key, value in mean_authority_metrics.items()
        }
        if grad_conflict_count > 0:
            grad_conflict_cos /= grad_conflict_count
            grad_conflict_norm_ratio /= grad_conflict_count
        else:
            grad_conflict_cos = 0.0
            grad_conflict_norm_ratio = 0.0
        mean_grad_diag = {
            key: value / max(1, grad_diag_count) for key, value in grad_diag_sums.items()
        }
        if mean_grad_diag:
            interval = int(getattr(self, "frontres_restore_debug_print_interval", 10))
            it = int(getattr(self, "current_learning_iteration", 0))
            if interval > 0 and it % interval == 0:
                print(
                    "[FrontRES grad debug] "
                    f"it={it} "
                    f"base_res={mean_grad_diag.get('base_residual', 0.0):.3e} "
                    f"base_acc={mean_grad_diag.get('base_acceptance', 0.0):.3e} "
                    f"base_alpha={mean_grad_diag.get('base_state_router', 0.0):.3e} "
                    f"ppo_res={mean_grad_diag.get('ppo_residual', 0.0):.3e} "
                    f"ppo_acc={mean_grad_diag.get('ppo_acceptance', 0.0):.3e} "
                    f"ppo_alpha={mean_grad_diag.get('ppo_state_router', 0.0):.3e} "
                    f"final_res={mean_grad_diag.get('final_residual', 0.0):.3e} "
                    f"final_acc={mean_grad_diag.get('final_acceptance', 0.0):.3e} "
                    f"final_alpha={mean_grad_diag.get('final_state_router', 0.0):.3e}",
                    flush=True,
                )

        self.storage.clear()
        for attr in ("_cached_observations", "_cached_full_policy_obs"):
            if hasattr(self.policy, attr):
                setattr(self.policy, attr, None)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "bc_off_policy": 0.0,
            "bc_teacher": 0.0,
            "lambda_off_policy": 0.0,
            "lambda_teacher": 0.0,
            "supervised_loss": mean_supervised_loss,
            "supervised_cos_sim": mean_supervised_cos_sim,
            "lambda_supervised": self.lambda_supervised,
            "acceptance_preference_loss": mean_acceptance_preference_loss,
            "lambda_acceptance_preference": self.frontres_acceptance_preference_weight,
            "state_alpha_loss": mean_state_alpha_loss,
            "lambda_state_alpha": self.frontres_state_alpha_weight,
            "structured_joint_rl_loss": mean_structured_joint_rl_loss,
            "lambda_structured_joint_rl": self.frontres_structured_joint_rl_weight,
            "lambda_structured_joint_prior": self.frontres_structured_joint_prior_loss_weight,
            "authority_loss": mean_authority_loss,
            "lambda_authority_actor": self.frontres_authority_actor_loss_weight,
            "lambda_authority_actor_effective": (
                self.frontres_authority_actor_loss_weight * self._authority_actor_phase_weight()
            ),
            "lambda_authority_critic": self.frontres_authority_critic_loss_weight,
            "ppo_actor_weight": (
                0.0
                if (
                    self._authority_actor_critic_enabled()
                    or self._active_hsl_acceptance_loss_enabled()
                    or (
                        self._structured_joint_rl_enabled()
                        and bool(getattr(self, "frontres_structured_joint_rl_disable_generic_ppo", True))
                    )
                )
                else float(getattr(self, "ppo_actor_weight", 1.0))
            ),
            "raw_ppo_actor_weight": float(getattr(self, "ppo_actor_weight", 1.0)),
            "grad_cos_ppo_supervised": grad_conflict_cos,
            "grad_norm_ratio_ppo_to_supervised": grad_conflict_norm_ratio,
        }
        loss_dict.update(mean_supervised_metrics)
        loss_dict.update(mean_acceptance_preference_metrics)
        loss_dict.update(mean_state_alpha_metrics)
        loss_dict.update(mean_structured_joint_rl_metrics)
        loss_dict.update(mean_authority_metrics)
        loss_dict.update({f"grad_{key}": value for key, value in mean_grad_diag.items()})
        if self.frontres_training_objective in ("supervised_restore", "basis_restore"):
            loss_dict["ppo_actor_weight"] = 0.0
        return loss_dict

    def _active_hsl_acceptance_loss_enabled(self) -> bool:
        """Return True when FEMR active path trains acceptance by rollout labels."""
        return (
            str(self.frontres_training_objective).lower() == "hsl_hybrid"
            and float(getattr(self, "frontres_acceptance_preference_weight", 0.0)) > 0.0
            and self._ppo_acceptance_only_mode()
            and not self._structured_joint_rl_enabled()
            and not self._authority_actor_critic_enabled()
        )

    def _ppo_selected_action_dims(self):
        """Return the action dimensions that PPO is allowed to update."""
        if self._ppo_acceptance_only_mode():
            return list(range(6, 6 + int(getattr(self.policy, "task_conf_dim", 2))))
        return None

    def _ppo_acceptance_only_mode(self) -> bool:
        """Current hsl_hybrid contract: PPO owns only acceptance coefficients."""
        return (
            str(self.frontres_training_objective).lower() == "hsl_hybrid"
            and getattr(self.policy, "num_task_corrections", 0) > 0
            and int(getattr(self.policy, "task_conf_dim", 2)) in (1, 6)
        )

    def _ppo_rho_pos_only_mode(self) -> bool:
        """Backward-compatible name for older scalar-head checks."""
        return self._ppo_acceptance_only_mode()

    def _ppo_tau_only_mode(self) -> bool:
        """Backward-compatible alias for older checkpoints/scripts."""
        return self._ppo_rho_pos_only_mode()

    def _get_actor_log_prob(self, actions):
        dims = self._ppo_selected_action_dims()
        if dims is not None and hasattr(self.policy, "get_actions_log_prob_selected"):
            return self.policy.get_actions_log_prob_selected(actions, dims)
        return self.policy.get_actions_log_prob(actions)

    def _structured_joint_rl_enabled(self) -> bool:
        return (
            bool(getattr(self, "frontres_structured_joint_rl_enabled", False))
            and float(getattr(self, "frontres_structured_joint_rl_weight", 0.0)) > 0.0
            and self._ppo_acceptance_only_mode()
        )

    def _authority_actor_critic_enabled(self) -> bool:
        return (
            bool(getattr(self, "frontres_authority_actor_critic_enabled", False))
            and (
                float(getattr(self, "frontres_authority_actor_loss_weight", 0.0)) > 0.0
                or float(getattr(self, "frontres_authority_critic_loss_weight", 0.0)) > 0.0
            )
        )

    def _authority_active_task_dim_mask(self, *, device, dtype) -> torch.Tensor | None:
        active_dims = getattr(self, "frontres_active_task_dims", None)
        if active_dims is None:
            return None
        dim_mask = torch.zeros(6, device=device, dtype=dtype)
        for idx in active_dims:
            idx = int(idx)
            if 0 <= idx < 6:
                dim_mask[idx] = 1.0
            elif 6 <= idx < 12:
                dim_mask[idx - 6] = 1.0
        return dim_mask

    def _authority_actor_phase_weight(self) -> float:
        """Return the Stage-2 authority actor takeover weight for the current iteration."""
        if not self._authority_actor_critic_enabled():
            return 0.0
        iteration = max(0, int(getattr(self, "current_learning_iteration", 0)))
        warmup_iters = max(0, int(getattr(self, "frontres_authority_actor_warmup_iterations", 0)))
        ramp_iters = max(0, int(getattr(self, "frontres_authority_actor_ramp_iterations", 0)))
        if iteration < warmup_iters:
            return 0.0
        if ramp_iters <= 0:
            return 1.0
        ramp_step = iteration - warmup_iters + 1
        return min(1.0, max(0.0, float(ramp_step) / float(ramp_iters)))

    def _compute_authority_actor_critic_loss(
        self,
        obs_batch,
        proposal_delta_se_batch,
        authority_action_batch,
        authority_return_k_batch,
        authority_return_zero_k_batch,
        authority_return_one_k_batch,
        authority_mask_batch,
        original_batch_size: int,
    ):
        zero = torch.zeros((), device=self.device)
        metrics = {
            "authority_actor_critic_enabled": 1.0 if self._authority_actor_critic_enabled() else 0.0,
            "authority_actor_loss": 0.0,
            "authority_critic_loss": 0.0,
            "authority_critic_behavior_loss": 0.0,
            "authority_critic_zero_loss": 0.0,
            "authority_critic_one_loss": 0.0,
            "authority_total_loss": 0.0,
            "authority_actor_phase_weight": 0.0,
            "authority_actor_warmup_active": 0.0,
            "authority_actor_ramp_active": 0.0,
            "authority_active_frac": 0.0,
            "authority_k_horizon": float(getattr(self, "frontres_authority_return_horizon", 1)),
            "authority_return_mean": 0.0,
            "authority_return_zero_mean": 0.0,
            "authority_return_one_mean": 0.0,
            "authority_q_behavior_mean": 0.0,
            "authority_q_actor_mean": 0.0,
            "authority_q_zero_mean": 0.0,
            "authority_q_one_mean": 0.0,
            "authority_q_one_minus_zero_mean": 0.0,
            "authority_q_actor_minus_zero_mean": 0.0,
            "authority_target_conflict_frac": 0.0,
            "authority_harmful_full_write_frac": 0.0,
            "authority_actor_reject_loss": 0.0,
            "authority_rho_mean": 0.0,
            "authority_rho_std": 0.0,
            "authority_rho_min": 0.0,
            "authority_rho_max": 0.0,
            "authority_rho_near_zero_frac": 0.0,
            "authority_rho_near_one_frac": 0.0,
            "authority_proposal_abs_mean": 0.0,
        }
        if not self._authority_actor_critic_enabled():
            return zero, metrics
        if (
            proposal_delta_se_batch is None
            or authority_action_batch is None
            or authority_return_k_batch is None
            or authority_mask_batch is None
        ):
            return zero, metrics

        n = int(original_batch_size)
        obs = obs_batch[:n]
        proposal = proposal_delta_se_batch[:n, :6].to(device=self.device, dtype=obs.dtype).detach()
        behavior_rho = authority_action_batch[:n, :6].to(device=self.device, dtype=obs.dtype).detach()
        target_return = authority_return_k_batch[:n, :1].to(device=self.device, dtype=obs.dtype).detach()
        target_zero = (
            authority_return_zero_k_batch[:n, :1].to(device=self.device, dtype=obs.dtype).detach()
            if authority_return_zero_k_batch is not None
            else torch.zeros_like(target_return)
        )
        target_one = (
            authority_return_one_k_batch[:n, :1].to(device=self.device, dtype=obs.dtype).detach()
            if authority_return_one_k_batch is not None
            else target_return
        )
        mask = authority_mask_batch[:n, :1].to(device=self.device, dtype=obs.dtype).detach()
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        denom = mask.sum().clamp(min=1e-6)
        if float(mask.sum().detach().item()) <= 1e-6:
            return zero, metrics

        behavior_rho = torch.nan_to_num(behavior_rho, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        resolved_targets = resolve_frontres_authority_targets(
            behavior_return=target_return,
            zero_return=target_zero,
            one_return=target_one,
            behavior_rho=behavior_rho,
            mask=mask,
        )
        target_return = resolved_targets.behavior_return
        target_zero = resolved_targets.zero_return
        target_one = resolved_targets.one_return
        mask = resolved_targets.mask
        denom = mask.sum().clamp(min=1e-6)
        q_behavior = self.policy.evaluate_authority_q(obs, proposal, behavior_rho, detach_proposal=True)
        active_task_dims = self._authority_active_task_dim_mask(device=self.device, dtype=obs.dtype)
        if active_task_dims is None:
            active_dim_mask = torch.ones(1, behavior_rho.shape[-1], device=self.device, dtype=obs.dtype)
        else:
            active_dim_mask = active_task_dims.view(1, -1).to(device=self.device, dtype=obs.dtype)
        zero_rho_fit = torch.zeros_like(behavior_rho)
        one_rho_fit = torch.ones_like(behavior_rho) * active_dim_mask
        q_zero_fit = self.policy.evaluate_authority_q(obs, proposal, zero_rho_fit, detach_proposal=True)
        q_one_fit = self.policy.evaluate_authority_q(obs, proposal, one_rho_fit, detach_proposal=True)
        behavior_loss = (((q_behavior - target_return) ** 2) * mask).sum() / denom
        zero_loss = (((q_zero_fit - target_zero) ** 2) * mask).sum() / denom
        one_loss = (((q_one_fit - target_one) ** 2) * mask).sum() / denom
        critic_loss = (behavior_loss + zero_loss + one_loss) / 3.0

        actor_rho = self.policy.get_authority_rho(
            obs,
            proposal_delta_se=proposal,
            active_task_dims=active_task_dims,
            detach_proposal=True,
        )
        authority_critic = getattr(self.policy, "authority_critic", None)
        critic_requires_grad = []
        if authority_critic is not None:
            for param in authority_critic.parameters():
                critic_requires_grad.append(param.requires_grad)
                param.requires_grad_(False)
        try:
            q_actor = self.policy.evaluate_authority_q(obs, proposal, actor_rho, detach_proposal=True)
        finally:
            if authority_critic is not None:
                for param, requires_grad in zip(authority_critic.parameters(), critic_requires_grad):
                    param.requires_grad_(requires_grad)
        target_endpoint_delta = (target_one - target_zero).detach()
        critic_endpoint_delta = (q_one_fit - q_zero_fit).detach()
        actor_ready_mask = mask * (target_endpoint_delta * critic_endpoint_delta > 0.0).to(dtype=mask.dtype)
        actor_denom = actor_ready_mask.sum().clamp(min=1e-6)
        actor_loss = -(q_actor * actor_ready_mask).sum() / actor_denom
        reject_loss = torch.zeros((), device=self.device, dtype=actor_loss.dtype)

        critic_weight = float(getattr(self, "frontres_authority_critic_loss_weight", 1.0))
        actor_phase_weight = self._authority_actor_phase_weight()
        actor_weight = float(getattr(self, "frontres_authority_actor_loss_weight", 1.0)) * actor_phase_weight
        total_loss = critic_weight * critic_loss + actor_weight * actor_loss

        with torch.no_grad():
            zero_rho = torch.zeros_like(actor_rho)
            one_rho = torch.ones_like(actor_rho) * active_dim_mask
            q_zero = self.policy.evaluate_authority_q(obs, proposal, zero_rho, detach_proposal=True)
            q_one = self.policy.evaluate_authority_q(obs, proposal, one_rho, detach_proposal=True)
            sample_dim_mask = mask * active_dim_mask
            denom_dim = sample_dim_mask.sum().clamp(min=1e-6)
            active_rho = actor_rho * sample_dim_mask
            active_proposal_abs = proposal.abs() * sample_dim_mask
            rho_mean = active_rho.sum() / denom_dim
            rho_var = (((actor_rho - rho_mean) ** 2) * sample_dim_mask).sum() / denom_dim
            has_dim = sample_dim_mask > 0.0
            rho_for_min = torch.where(has_dim, actor_rho, torch.ones_like(actor_rho))
            rho_for_max = torch.where(has_dim, actor_rho, torch.zeros_like(actor_rho))
            sample_rho = (active_rho.sum(dim=-1, keepdim=True) / sample_dim_mask.sum(dim=-1, keepdim=True).clamp(min=1e-6))
            active_sample_mask = mask > 0.0

            def _bucket_mean(value: torch.Tensor, bucket: torch.Tensor) -> float:
                bucket_mask = (bucket & active_sample_mask).to(dtype=value.dtype)
                bucket_denom = bucket_mask.sum().clamp(min=1e-6)
                return float((value * bucket_mask).sum().detach().item() / bucket_denom.detach().item())

            metrics["authority_actor_loss"] = float(actor_loss.detach().item())
            metrics["authority_critic_loss"] = float(critic_loss.detach().item())
            metrics["authority_critic_behavior_loss"] = float(behavior_loss.detach().item())
            metrics["authority_critic_zero_loss"] = float(zero_loss.detach().item())
            metrics["authority_critic_one_loss"] = float(one_loss.detach().item())
            metrics["authority_total_loss"] = float(total_loss.detach().item())
            metrics["authority_actor_phase_weight"] = float(actor_phase_weight)
            metrics["authority_actor_warmup_active"] = 1.0 if actor_phase_weight <= 0.0 else 0.0
            metrics["authority_actor_ramp_active"] = 1.0 if 0.0 < actor_phase_weight < 1.0 else 0.0
            metrics["authority_active_frac"] = float(mask.mean().detach().item())
            metrics["authority_return_mean"] = float((target_return * mask).sum().detach().item() / denom.detach().item())
            metrics["authority_return_zero_mean"] = float((target_zero * mask).sum().detach().item() / denom.detach().item())
            metrics["authority_return_one_mean"] = float((target_one * mask).sum().detach().item() / denom.detach().item())
            metrics["authority_q_behavior_mean"] = float((q_behavior * mask).sum().detach().item() / denom.detach().item())
            metrics["authority_q_actor_mean"] = float((q_actor * mask).sum().detach().item() / denom.detach().item())
            metrics["authority_q_zero_mean"] = float((q_zero * mask).sum().detach().item() / denom.detach().item())
            metrics["authority_q_one_mean"] = float((q_one * mask).sum().detach().item() / denom.detach().item())
            metrics["authority_q_one_minus_zero_mean"] = float(((q_one - q_zero) * mask).sum().detach().item() / denom.detach().item())
            metrics["authority_q_actor_minus_zero_mean"] = float(((q_actor - q_zero) * mask).sum().detach().item() / denom.detach().item())
            target_endpoint_delta_metric = target_endpoint_delta
            critic_endpoint_delta_metric = critic_endpoint_delta
            ready_accept_mask = actor_ready_mask * (target_endpoint_delta_metric > 0.0).to(dtype=mask.dtype)
            ready_reject_mask = actor_ready_mask * (target_endpoint_delta_metric < 0.0).to(dtype=mask.dtype)
            metrics["authority_actor_ready_frac"] = float((actor_ready_mask.sum() / denom).detach().item())
            metrics["authority_actor_ready_accept_frac"] = float((ready_accept_mask.sum() / denom).detach().item())
            metrics["authority_actor_ready_reject_frac"] = float((ready_reject_mask.sum() / denom).detach().item())
            metrics["authority_target_conflict_frac"] = float(
                ((resolved_targets.conflict_mask * mask).sum() / denom).detach().item()
            )
            metrics["authority_harmful_full_write_frac"] = float(
                ((resolved_targets.harmful_full_write_mask * mask).sum() / denom).detach().item()
            )
            metrics["authority_actor_reject_loss"] = float(reject_loss.detach().item())
            metrics["authority_rho_mean"] = float(rho_mean.detach().item())
            metrics["authority_rho_std"] = float(torch.sqrt(rho_var.clamp(min=0.0)).detach().item())
            metrics["authority_rho_min"] = float(rho_for_min.min().detach().item())
            metrics["authority_rho_max"] = float(rho_for_max.max().detach().item())
            metrics["authority_rho_near_zero_frac"] = float(
                (((actor_rho <= 0.05).to(dtype=actor_rho.dtype) * sample_dim_mask).sum() / denom_dim).detach().item()
            )
            metrics["authority_rho_near_one_frac"] = float(
                (((actor_rho >= 0.95).to(dtype=actor_rho.dtype) * sample_dim_mask).sum() / denom_dim).detach().item()
            )
            metrics["authority_proposal_abs_mean"] = float((active_proposal_abs.sum() / denom_dim).detach().item())
            for idx, name in enumerate(("dx", "dy", "dz", "roll", "pitch", "yaw")):
                dim_mask = sample_dim_mask[:, idx : idx + 1]
                dim_denom = dim_mask.sum().clamp(min=1e-6)
                metrics[f"authority_rho_{name}_mean"] = float(
                    ((actor_rho[:, idx : idx + 1] * dim_mask).sum() / dim_denom).detach().item()
                )
            low_bucket = sample_rho <= 0.25
            mid_bucket = (sample_rho > 0.25) & (sample_rho < 0.75)
            high_bucket = sample_rho >= 0.75
            proposal_sample_abs = (
                active_proposal_abs.sum(dim=-1, keepdim=True)
                / sample_dim_mask.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            )
            for bucket_name, bucket in (("low", low_bucket), ("mid", mid_bucket), ("high", high_bucket)):
                metrics[f"authority_return_{bucket_name}_rho_mean"] = _bucket_mean(target_return, bucket)
                metrics[f"authority_q_actor_{bucket_name}_rho_mean"] = _bucket_mean(q_actor, bucket)
                metrics[f"authority_proposal_abs_{bucket_name}_rho_mean"] = _bucket_mean(proposal_sample_abs, bucket)
        return total_loss, metrics

    def _keep_rl_grad_on_acceptance_and_state_router_only(self, base_grads):
        """Deprecated compatibility alias.

        Structured Joint RL now trains rho only.  Alpha/state_router gradients
        must come only from the separate state-alpha SSL loss.
        """
        return self._keep_ppo_grad_on_acceptance_head_only(base_grads)

    def _keep_ppo_grad_on_acceptance_head_only(self, base_grads):
        """Remove PPO leakage into proposal direction and shared trunk.

        In the preferred two-head implementation, PPO/preference gradients are
        allowed to update only the acceptance head.  In the optional split-MLP
        ablation, they update only the separate acceptance network.  In the
        legacy shared-output MLP, keep non-PPO gradients everywhere and add the
        PPO/preference gradient only to final-layer rows that emit acceptance.
        """
        acceptance_actor = getattr(self.policy, "acceptance_actor", None)
        if acceptance_actor is not None:
            allowed = {p for p in acceptance_actor.parameters()}
            for p in self.policy.parameters():
                base = base_grads.get(p)
                if p not in allowed:
                    p.grad = None if base is None else base.clone()
                    continue
                cur = p.grad
                if cur is None:
                    p.grad = None if base is None else base.clone()
                    continue
                base_tensor = torch.zeros_like(cur) if base is None else base
                p.grad = cur
            return

        final_linear = None
        residual_actor = getattr(self.policy, "residual_actor", None)
        if residual_actor is None:
            return
        acceptance_head = getattr(residual_actor, "acceptance_head", None)
        if acceptance_head is not None:
            allowed = {acceptance_head.weight, acceptance_head.bias}
            for p in self.policy.parameters():
                base = base_grads.get(p)
                if p not in allowed:
                    p.grad = None if base is None else base.clone()
                    continue
                cur = p.grad
                if cur is None:
                    p.grad = None if base is None else base.clone()
                    continue
                p.grad = cur
            return
        for module in residual_actor.modules():
            if isinstance(module, nn.Linear):
                final_linear = module
        if final_linear is None:
            return
        conf_dim = int(getattr(self.policy, "task_conf_dim", 2))
        start = int(getattr(self.policy, "num_task_corrections", 6))
        end = min(start + conf_dim, final_linear.out_features)
        if end <= start:
            return

        allowed = {final_linear.weight, final_linear.bias}
        for p in self.policy.parameters():
            base = base_grads.get(p)
            if p not in allowed:
                p.grad = None if base is None else base.clone()
                continue
            cur = p.grad
            if cur is None:
                p.grad = None if base is None else base.clone()
                continue
            base_tensor = torch.zeros_like(cur) if base is None else base
            ppo_grad = cur - base_tensor
            mask = torch.zeros_like(cur)
            if p is final_linear.weight:
                mask[start:end, :] = 1.0
            else:
                mask[start:end] = 1.0
            p.grad = base_tensor + ppo_grad * mask

    def _keep_ppo_grad_on_rho_pos_head_only(self, base_grads):
        """Backward-compatible alias for the previous scalar-head name."""
        return self._keep_ppo_grad_on_acceptance_head_only(base_grads)

    def _keep_ppo_grad_on_tau_head_only(self, base_grads):
        """Backward-compatible alias for the previous scalar-head name."""
        return self._keep_ppo_grad_on_acceptance_head_only(base_grads)

    def _grad_norm_from_map(self, params, grad_map) -> float:
        norm_sq = torch.tensor(0.0, device=self.device)
        for p in params:
            g = grad_map.get(p) if grad_map is not None else None
            if g is not None:
                norm_sq = norm_sq + g.square().sum()
        return float(norm_sq.sqrt().detach().item())

    def _grad_norm_current(self, params) -> float:
        norm_sq = torch.tensor(0.0, device=self.device)
        for p in params:
            if p.grad is not None:
                norm_sq = norm_sq + p.grad.square().sum()
        return float(norm_sq.sqrt().detach().item())

    def _grad_delta_norm(self, params, base_grads) -> float:
        norm_sq = torch.tensor(0.0, device=self.device)
        for p in params:
            cur = p.grad
            base = base_grads.get(p) if base_grads is not None else None
            if cur is None and base is None:
                continue
            if cur is None:
                delta = -base
            elif base is None:
                delta = cur
            else:
                delta = cur - base
            norm_sq = norm_sq + delta.square().sum()
        return float(norm_sq.sqrt().detach().item())

    def _compute_acceptance_grad_diagnostics(self, base_grads) -> dict[str, float]:
        residual_actor = getattr(self.policy, "residual_actor", None)
        acceptance_actor = getattr(self.policy, "acceptance_actor", None)
        state_router = getattr(self.policy, "state_router_head", None)
        if residual_actor is None and acceptance_actor is None and state_router is None:
            return {}

        two_head_acceptance = getattr(residual_actor, "acceptance_head", None) if residual_actor is not None else None
        if residual_actor is not None and two_head_acceptance is not None:
            acceptance_set = {p for p in two_head_acceptance.parameters()}
            residual_params = [
                p for p in residual_actor.parameters()
                if p.requires_grad and p not in acceptance_set
            ]
        else:
            residual_params = [p for p in residual_actor.parameters() if p.requires_grad] if residual_actor is not None else []
        acceptance_params = [p for p in acceptance_actor.parameters() if p.requires_grad] if acceptance_actor is not None else []
        if not acceptance_params and two_head_acceptance is not None:
            acceptance_params = [p for p in two_head_acceptance.parameters() if p.requires_grad]
        state_router_params = [p for p in state_router.parameters() if p.requires_grad] if state_router is not None else []
        diagnostics: dict[str, float] = {}
        if residual_params:
            diagnostics["base_residual"] = self._grad_norm_from_map(residual_params, base_grads)
            diagnostics["ppo_residual"] = self._grad_delta_norm(residual_params, base_grads)
            diagnostics["final_residual"] = self._grad_norm_current(residual_params)
        if acceptance_params:
            diagnostics["base_acceptance"] = self._grad_norm_from_map(acceptance_params, base_grads)
            diagnostics["ppo_acceptance"] = self._grad_delta_norm(acceptance_params, base_grads)
            diagnostics["final_acceptance"] = self._grad_norm_current(acceptance_params)
        if state_router_params:
            diagnostics["base_state_router"] = self._grad_norm_from_map(state_router_params, base_grads)
            diagnostics["ppo_state_router"] = self._grad_delta_norm(state_router_params, base_grads)
            diagnostics["final_state_router"] = self._grad_norm_current(state_router_params)
        return diagnostics

    def _compute_actor_grad_conflict(self, surrogate_loss, supervised_loss):
        residual_actor = getattr(self.policy, "residual_actor", None)
        if residual_actor is None:
            params = []
        else:
            acceptance_head = getattr(residual_actor, "acceptance_head", None)
            acceptance_params = {p for p in acceptance_head.parameters()} if acceptance_head is not None else set()
            params = [
                p for p in residual_actor.parameters()
                if p.requires_grad and p not in acceptance_params
            ]
        if not params:
            return None, 0.0

        ppo_grads = torch.autograd.grad(
            surrogate_loss, params, retain_graph=True, allow_unused=True)
        sup_grads = torch.autograd.grad(
            supervised_loss, params, retain_graph=True, allow_unused=True)

        dot = torch.tensor(0.0, device=self.device)
        ppo_norm_sq = torch.tensor(0.0, device=self.device)
        sup_norm_sq = torch.tensor(0.0, device=self.device)
        for gp, gs in zip(ppo_grads, sup_grads):
            if gp is None or gs is None:
                continue
            dot = dot + (gp * gs).sum()
            ppo_norm_sq = ppo_norm_sq + gp.square().sum()
            sup_norm_sq = sup_norm_sq + gs.square().sum()

        denom = (ppo_norm_sq.sqrt() * sup_norm_sq.sqrt()).clamp(min=1e-12)
        if float(ppo_norm_sq.detach().item()) <= 0.0 or float(sup_norm_sq.detach().item()) <= 0.0:
            return None, 0.0
        cos = (dot / denom).detach().item()
        ratio = (ppo_norm_sq.sqrt() / sup_norm_sq.sqrt().clamp(min=1e-12)).detach().item()
        return cos, ratio

    def _adapt_learning_rate(self, old_mu_batch, old_sigma_batch, mu_batch, sigma_batch):
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size

            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _compute_value_loss(
        self,
        target_values_batch,
        returns_batch,
        value_batch,
        frontres_mask_batch,
    ):
        has_mask = frontres_mask_batch is not None
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_terms = torch.max(value_losses, value_losses_clipped)
        else:
            value_terms = (returns_batch - value_batch).pow(2)

        if has_mask:
            n = frontres_mask_batch.sum().clamp(min=1.0)
            value_loss = (value_terms * frontres_mask_batch).sum() / n
        else:
            value_loss = value_terms.mean()
        return value_loss

    def _compute_ppo_losses(
        self,
        actions_log_prob_batch,
        old_actions_log_prob_batch,
        advantages_batch,
        target_values_batch,
        returns_batch,
        value_batch,
        frontres_mask_batch,
        frontres_actor_gate_batch=None,
    ):
        has_mask = frontres_mask_batch is not None
        log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))

        advantages = torch.squeeze(advantages_batch)
        focal_power = max(0.0, float(getattr(self, "ppo_advantage_focal_power", 0.0)))
        if focal_power > 0.0:
            focal = advantages.abs().pow(focal_power).clamp(max=25.0)
        else:
            focal = torch.ones_like(advantages)
        surrogate = -advantages * ratio * focal
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * focal
        surrogate_terms = torch.max(surrogate, surrogate_clipped)

        if has_mask:
            if frontres_actor_gate_batch is not None:
                mask_flat = (frontres_mask_batch * frontres_actor_gate_batch).view(-1)
            else:
                mask_flat = frontres_mask_batch.view(-1)
            # Normalize by the sum of sample weights, not by minibatch size.
            # This makes actor_gate redistribute actor-gradient contribution
            # without shrinking the whole PPO step when most samples are outside
            # the repairability window.
            surrogate_loss = (surrogate_terms * mask_flat).sum() / mask_flat.sum().clamp(min=1e-6)
        else:
            surrogate_loss = surrogate_terms.mean()

        value_loss = self._compute_value_loss(
            target_values_batch,
            returns_batch,
            value_batch,
            frontres_mask_batch,
        )
        return surrogate_loss, value_loss

    def _compute_acceptance_preference_loss(
        self,
        mu_batch,
        acceptance_target_batch,
        acceptance_mask_batch,
        original_batch_size,
    ):
        zero = torch.zeros((), device=self.device, dtype=mu_batch.dtype)
        metrics = {
            "acceptance_preference_mask_frac": 0.0,
            "acceptance_preference_target_mean": 0.0,
            "acceptance_preference_full_frac": 0.0,
            "acceptance_preference_noop_frac": 0.0,
            "acceptance_preference_rho_mean": 0.0,
            "acceptance_preference_abs_err": 0.0,
            "acceptance_preference_corr": 0.0,
            "acceptance_preference_focal_gamma": 0.0,
            "acceptance_preference_full_weight": 1.0,
            "acceptance_preference_noop_weight": 1.0,
            "acceptance_preference_effective_full_frac": 0.0,
            "hsl_acceptance_path_enabled": 0.0,
            "hsl_acceptance_loss_enabled": 0.0,
            "hsl_acceptance_gt_mean": 0.0,
            "hsl_acceptance_mask_frac": 0.0,
            "hsl_acceptance_prob_mean": 0.0,
            "hsl_acceptance_abs_err": 0.0,
        }
        metrics["hsl_acceptance_path_enabled"] = 1.0 if self._active_hsl_acceptance_loss_enabled() else 0.0
        if (
            self._structured_joint_rl_enabled()
            and not self.frontres_structured_joint_rl_keep_legacy_bce
        ):
            return zero, metrics
        if (
            self.frontres_acceptance_preference_weight <= 0.0
            or acceptance_target_batch is None
            or acceptance_mask_batch is None
            or not self._ppo_acceptance_only_mode()
            or self._authority_actor_critic_enabled()
        ):
            return zero, metrics

        raw_pred = mu_batch[:original_batch_size]
        if raw_pred.shape[-1] < 7:
            return zero, metrics

        conf_dim = int(getattr(self.policy, "task_conf_dim", 2))
        if conf_dim == 1:
            logits = raw_pred[:, 6:7]
            rho = torch.sigmoid(logits).expand(-1, 6)
        elif conf_dim == 6 and raw_pred.shape[-1] >= 12:
            logits = raw_pred[:, 6:12]
            rho = torch.sigmoid(logits)
        else:
            return zero, metrics

        target = acceptance_target_batch[:original_batch_size, :6].to(
            device=self.device, dtype=rho.dtype
        ).detach()
        mask = acceptance_mask_batch[:original_batch_size, :6].to(
            device=self.device, dtype=rho.dtype
        ).detach()
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)

        active_dims = getattr(self, "frontres_active_task_dims", None)
        if active_dims is not None:
            dim_mask = torch.zeros(6, device=self.device, dtype=mask.dtype)
            for idx in active_dims:
                idx = int(idx)
                if 0 <= idx < 6:
                    dim_mask[idx] = 1.0
                elif 6 <= idx < 12:
                    dim_mask[idx - 6] = 1.0
            mask = mask * dim_mask.view(1, -1)

        denom_raw = mask.sum()
        if float(denom_raw.detach().item()) <= 1e-6:
            return zero, metrics
        loss_terms = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        rho_clamped = rho.clamp(1e-4, 1.0 - 1e-4)
        # Targets may be soft after rollout-calibrated acceptance projection.
        # Treat target mass as "accept" weight and (1-target) as "no-op" mass
        # instead of hard-thresholding every element for class balancing.
        full_indicator = target.to(mask.dtype)
        noop_indicator = (1.0 - target).to(mask.dtype)
        full_active_mask = mask * full_indicator
        noop_active_mask = mask * noop_indicator
        full_mass = full_active_mask.sum()
        noop_mass = noop_active_mask.sum()
        total_mass = (full_mass + noop_mass).clamp(min=1e-6)
        balance_min = float(getattr(self, "frontres_acceptance_preference_balance_min", 1.0))
        balance_max = float(getattr(self, "frontres_acceptance_preference_balance_max", 1.0))
        full_weight = (total_mass / (2.0 * full_mass.clamp(min=1e-6))).clamp(
            min=balance_min, max=balance_max
        )
        noop_weight = (total_mass / (2.0 * noop_mass.clamp(min=1e-6))).clamp(
            min=balance_min, max=balance_max
        )
        class_weight = full_weight * full_indicator + noop_weight * noop_indicator

        gamma = float(getattr(self, "frontres_acceptance_preference_focal_gamma", 0.0))
        if gamma > 0.0:
            pt = target * rho_clamped + (1.0 - target) * (1.0 - rho_clamped)
            focal_weight = (1.0 - pt).pow(gamma)
        else:
            focal_weight = torch.ones_like(mask)
        weighted_mask = mask * class_weight * focal_weight
        denom = weighted_mask.sum().clamp(min=1e-6)
        loss = (loss_terms * weighted_mask).sum() / denom

        sample_active = (mask.sum(dim=-1) > 0).to(rho.dtype)
        sample_denom = sample_active.sum().clamp(min=1e-6)
        target_per_sample = (target * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
        rho_per_sample = (rho.detach() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
        err_per_sample = ((rho.detach() - target).abs() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)

        metrics["acceptance_preference_mask_frac"] = float(sample_active.mean().detach().item())
        metrics["hsl_acceptance_loss_enabled"] = 1.0
        metrics["hsl_acceptance_mask_frac"] = metrics["acceptance_preference_mask_frac"]
        metrics["acceptance_preference_target_mean"] = float(
            (target_per_sample * sample_active).sum().detach().item() / float(sample_denom.detach().item())
        )
        metrics["hsl_acceptance_gt_mean"] = metrics["acceptance_preference_target_mean"]
        metrics["acceptance_preference_full_frac"] = float(
            ((target_per_sample > 0.5).to(rho.dtype) * sample_active).sum().detach().item()
            / float(sample_denom.detach().item())
        )
        metrics["acceptance_preference_noop_frac"] = float(
            ((target_per_sample < 0.5).to(rho.dtype) * sample_active).sum().detach().item()
            / float(sample_denom.detach().item())
        )
        metrics["acceptance_preference_rho_mean"] = float(
            (rho_per_sample * sample_active).sum().detach().item() / float(sample_denom.detach().item())
        )
        metrics["hsl_acceptance_prob_mean"] = metrics["acceptance_preference_rho_mean"]
        metrics["acceptance_preference_abs_err"] = float(
            (err_per_sample * sample_active).sum().detach().item() / float(sample_denom.detach().item())
        )
        metrics["hsl_acceptance_abs_err"] = metrics["acceptance_preference_abs_err"]
        effective_full_mass = (weighted_mask.detach() * full_indicator).sum()
        effective_noop_mass = (weighted_mask.detach() * noop_indicator).sum()
        effective_total = (effective_full_mass + effective_noop_mass).clamp(min=1e-6)
        metrics["acceptance_preference_focal_gamma"] = gamma
        metrics["acceptance_preference_full_weight"] = float(full_weight.detach().item())
        metrics["acceptance_preference_noop_weight"] = float(noop_weight.detach().item())
        metrics["acceptance_preference_effective_full_frac"] = float(
            (effective_full_mass / effective_total).detach().item()
        )
        active = sample_active > 0
        if int(active.sum().detach().item()) > 1:
            rho_active = rho_per_sample[active]
            target_active = target_per_sample[active]
            rho_centered = rho_active - rho_active.mean()
            target_centered = target_active - target_active.mean()
            corr_denom = rho_centered.square().sum().sqrt() * target_centered.square().sum().sqrt()
            if float(corr_denom.detach().item()) > 1e-12:
                metrics["acceptance_preference_corr"] = float(
                    (rho_centered * target_centered).sum().div(corr_denom).detach().item()
                )
        return loss, metrics

    def _compute_structured_joint_rl_loss(
        self,
        obs_batch,
        mu_batch,
        actions_batch,
        old_mu_batch,
        old_sigma_batch,
        actions_log_prob_batch,
        old_actions_log_prob_batch,
        acceptance_target_batch,
        acceptance_mask_batch,
        rho_prior_authority_batch,
        rho_prior_target_batch,
        original_batch_size,
    ):
        zero = obs_batch[:original_batch_size].sum() * 0.0
        metrics = {
            "structured_joint_rl_enabled": 1.0 if self._structured_joint_rl_enabled() else 0.0,
            "structured_joint_rl_mode_region_direct": (
                1.0 if getattr(self, "frontres_structured_joint_rl_loss_mode", "ppo_clipped") == "region_direct" else 0.0
            ),
            "structured_joint_rl_adv_mean": 0.0,
            "structured_joint_rl_adv_abs_mean": 0.0,
            "structured_joint_rl_adv_used_mean": 0.0,
            "structured_joint_rl_weight_mean": 0.0,
            "structured_joint_rl_weight_all_mean": 0.0,
            "structured_joint_rl_rho_adv_mean": 0.0,
            "structured_joint_rl_rho_adv_abs_mean": 0.0,
            "structured_joint_rl_rho_weight_mean": 0.0,
            "structured_joint_rl_rho_weight_all_mean": 0.0,
            "structured_joint_rl_rho_ratio_mean": 1.0,
            "structured_joint_rl_rho_loss": 0.0,
            "structured_joint_rl_repairable_loss": 0.0,
            "structured_joint_rl_boundary_loss": 0.0,
            "structured_joint_rl_repair_loss_scale": float(
                getattr(self, "frontres_structured_joint_repair_loss_scale", 1.0)
            ),
            "structured_joint_rl_repair_loss_is_bce": (
                1.0
                if str(
                    getattr(
                        self,
                        "frontres_structured_joint_repair_loss_kind",
                        "current_rho_linear",
                    )
                ).lower()
                == "bce_logit"
                else 0.0
            ),
            "structured_joint_rl_repairable_authority_mean": 0.0,
            "structured_joint_rl_boundary_authority_mean": 0.0,
            "structured_joint_rl_prior_loss": 0.0,
            "structured_joint_rl_prior_authority_mean": 0.0,
            "structured_joint_rl_prior_target_mean": 0.0,
            "structured_joint_rl_prior_rho_mean": 0.0,
            "structured_joint_rl_rho_mean": 0.0,
            "structured_joint_rl_rho_abs_from_half": 0.0,
            "structured_joint_rl_rho_near_half_frac": 0.0,
            "structured_joint_rl_rho_action_minus_mean_abs": 0.0,
            "structured_joint_rl_rho_action_minus_mean_rho_abs": 0.0,
            "structured_joint_rl_rho_pos_adv_mean": 0.0,
            "structured_joint_rl_rho_neg_adv_mean": 0.0,
            "structured_joint_rl_adv_pos_abs_mean": 0.0,
            "structured_joint_rl_adv_neg_abs_mean": 0.0,
            "structured_joint_rl_rho_repairable_mean": 0.0,
            "structured_joint_rl_rho_boundary_mean": 0.0,
            "structured_joint_rl_repairable_pos_frac": 0.0,
            "structured_joint_rl_rho_pos_dim_mean": 0.0,
            "structured_joint_rl_rho_rpy_dim_mean": 0.0,
            "structured_joint_rl_rho_pos_adv_pos_dim_mean": 0.0,
            "structured_joint_rl_rho_pos_adv_rpy_dim_mean": 0.0,
            "structured_joint_rl_rho_neg_adv_pos_dim_mean": 0.0,
            "structured_joint_rl_rho_neg_adv_rpy_dim_mean": 0.0,
            "structured_joint_rl_adv_pos_frac_pos_dim": 0.0,
            "structured_joint_rl_adv_pos_frac_rpy_dim": 0.0,
            "structured_joint_rl_active_frac_pos_dim": 0.0,
            "structured_joint_rl_active_frac_rpy_dim": 0.0,
            "structured_joint_rl_adv_pos_frac": 0.0,
            "structured_joint_rl_adv_neg_frac": 0.0,
            "structured_joint_rl_adv_near_zero_frac": 0.0,
            "structured_joint_rl_ratio_mean": 1.0,
            "structured_joint_rl_dim_active_mean": 0.0,
        }
        # Structured-rho mode reuses legacy acceptance_* storage fields:
        # target carries rho advantage, mask carries rho loss weight.
        rho_advantage_batch = acceptance_target_batch
        rho_weight_batch = acceptance_mask_batch
        if (
            not self._structured_joint_rl_enabled()
            or rho_advantage_batch is None
            or rho_weight_batch is None
        ):
            return zero, metrics

        n = original_batch_size
        conf_dim = int(getattr(self.policy, "task_conf_dim", 1))
        conf_dim = max(1, min(conf_dim, rho_advantage_batch.shape[-1]))
        rho_dims = list(range(6, 6 + conf_dim))
        carrier = rho_advantage_batch[:n, :conf_dim].to(device=self.device, dtype=obs_batch.dtype).detach()
        weights = rho_weight_batch[:n, :conf_dim].to(device=self.device, dtype=obs_batch.dtype).detach()
        rho_adv_raw = torch.nan_to_num(carrier, nan=0.0, posinf=0.0, neginf=0.0)
        rho_weight = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)

        rho_active = rho_weight > 1e-6
        if not bool(rho_active.any().detach().item()):
            return zero, metrics

        def _prepare_advantage(raw_adv: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
            adv = raw_adv
            clip = float(getattr(self, "frontres_structured_joint_rl_adv_clip", 0.0))
            if clip > 0.0:
                adv = adv.clamp(-clip, clip)
            # Structured rho advantages already carry an absolute signed
            # direction.  Normalizing them inside a minibatch can turn globally
            # useful repair evidence into relative negative samples, so keep it
            # opt-in even for older configs/checkpoints that lack the flag.
            if bool(getattr(self, "frontres_structured_joint_rl_normalize_advantage", False)):
                active_adv = adv[active_mask]
                if int(active_adv.numel()) > 1:
                    adv_mean = active_adv.mean()
                    adv_std = active_adv.std(unbiased=False).clamp(min=1e-6)
                    adv = (adv - adv_mean) / adv_std
            return adv

        loss_mode = str(getattr(self, "frontres_structured_joint_rl_loss_mode", "ppo_clipped")).lower()
        if loss_mode == "region_direct":
            cols = min(
                rho_adv_raw.shape[-1],
                rho_weight.shape[-1],
                max(0, mu_batch.shape[-1] - 6),
            )
            new_rho_logp = None
            old_rho_logp = None
        else:
            if hasattr(self.policy, "get_actions_log_prob_per_dim"):
                new_rho_logp = self.policy.get_actions_log_prob_per_dim(actions_batch[:n], rho_dims)
                new_rho_logp = new_rho_logp.to(device=self.device, dtype=obs_batch.dtype)
            else:
                new_rho_logp = actions_log_prob_batch[:n].view(-1, 1).to(
                    device=self.device, dtype=obs_batch.dtype
                ).expand_as(rho_adv_raw)
            if hasattr(self.policy, "get_actions_log_prob_per_dim_from_stats"):
                old_rho_logp = self.policy.get_actions_log_prob_per_dim_from_stats(
                    actions_batch[:n],
                    old_mu_batch[:n],
                    old_sigma_batch[:n],
                    rho_dims,
                )
                old_rho_logp = old_rho_logp.to(device=self.device, dtype=obs_batch.dtype)
            else:
                old_rho_logp = old_actions_log_prob_batch[:n].view(-1, 1).to(
                    device=self.device, dtype=obs_batch.dtype
                ).expand_as(rho_adv_raw)
            cols = min(
                rho_adv_raw.shape[-1],
                rho_weight.shape[-1],
                new_rho_logp.shape[-1],
                old_rho_logp.shape[-1],
            )
        if cols <= 0:
            return zero, metrics
        rho_adv_raw = rho_adv_raw[:, :cols]
        rho_weight = rho_weight[:, :cols]
        if new_rho_logp is not None:
            new_rho_logp = new_rho_logp[:, :cols]
        if old_rho_logp is not None:
            old_rho_logp = old_rho_logp[:, :cols]
        rho_active = rho_weight > 1e-6
        if not bool(rho_active.any().detach().item()):
            return zero, metrics

        rho_adv = _prepare_advantage(rho_adv_raw, rho_active)

        def _clipped_loss(
            adv: torch.Tensor,
            weight: torch.Tensor,
            log_ratio: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
            ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate = -adv * ratio
            surrogate_clipped = -adv * ratio_clipped
            loss_terms = torch.max(surrogate, surrogate_clipped)
            denom = weight.sum().clamp(min=1e-6)
            return (loss_terms * weight).sum() / denom, ratio

        rho_action = actions_batch[:n, 6:6 + cols].to(device=self.device, dtype=obs_batch.dtype)
        rho_action = rho_action.clamp(1e-6, 1.0 - 1e-6)
        rho_action_raw = torch.log(rho_action / (1.0 - rho_action))
        rho_mean_raw = mu_batch[:n, 6:6 + cols].to(device=self.device, dtype=obs_batch.dtype)
        rho_mean = torch.sigmoid(rho_mean_raw)
        if loss_mode == "region_direct":
            rho_loss = zero
            rho_ratio = torch.ones_like(rho_adv)
        else:
            rho_log_ratio = new_rho_logp - old_rho_logp
            rho_loss, rho_ratio = _clipped_loss(rho_adv, rho_weight, rho_log_ratio)
        prior_loss = zero
        prior_loss_weight = float(getattr(self, "frontres_structured_joint_prior_loss_weight", 0.0))
        prior_authority = None
        prior_target = None
        prior_dim_weight = None
        has_prior_inputs = rho_prior_authority_batch is not None and rho_prior_target_batch is not None
        if has_prior_inputs:
            prior_authority = rho_prior_authority_batch[:n].to(
                device=self.device, dtype=obs_batch.dtype
            ).detach().clamp(0.0, 1.0)
            if prior_authority.ndim == 1:
                prior_authority = prior_authority.view(-1, 1)
            prior_target = rho_prior_target_batch[:n, :cols].to(
                device=self.device, dtype=obs_batch.dtype
            ).detach().clamp(0.0, 1.0)
            prior_dim_weight = (prior_authority[:, :1] * (rho_weight > 1e-6).to(obs_batch.dtype)).clamp(0.0, 1.0)
            metrics["structured_joint_rl_prior_authority_mean"] = float(
                prior_authority.mean().detach().item()
            )
            metrics["structured_joint_rl_boundary_authority_mean"] = metrics[
                "structured_joint_rl_prior_authority_mean"
            ]
            repairable_authority = (1.0 - prior_authority[:, :1]).clamp(0.0, 1.0)
            metrics["structured_joint_rl_repairable_authority_mean"] = float(
                repairable_authority.mean().detach().item()
            )
        else:
            prior_authority = torch.zeros(n, 1, device=self.device, dtype=obs_batch.dtype)
            prior_target = torch.zeros(n, cols, device=self.device, dtype=obs_batch.dtype)
            prior_dim_weight = torch.zeros_like(rho_weight)
            metrics["structured_joint_rl_repairable_authority_mean"] = 1.0
        if loss_mode == "region_direct":
            repairable_authority = (1.0 - prior_authority[:, :1]).clamp(0.0, 1.0)
            repairable_weight = (repairable_authority * (rho_weight > 1e-6).to(obs_batch.dtype)).clamp(0.0, 1.0)
            repairable_loss = zero
            repair_loss_kind = str(
                getattr(self, "frontres_structured_joint_repair_loss_kind", "current_rho_linear")
            ).lower()
            repair_loss_scale = max(
                0.0,
                float(getattr(self, "frontres_structured_joint_repair_loss_scale", 1.0)),
            )
            if bool((repairable_weight > 1e-6).any().detach().item()):
                if repair_loss_kind == "bce_logit":
                    repair_target = (rho_adv > 0.0).to(dtype=obs_batch.dtype)
                    repair_terms = F.binary_cross_entropy_with_logits(
                        rho_mean_raw,
                        repair_target,
                        reduction="none",
                    )
                    repairable_loss = (
                        repair_terms * rho_adv.abs() * repairable_weight
                    ).sum()
                else:
                    repair_loss_kind = "current_rho_linear"
                    repairable_loss = (-rho_adv * rho_mean * repairable_weight).sum()
                repairable_loss = repairable_loss / repairable_weight.sum().clamp(min=1e-6)
            boundary_loss = zero
            if bool((prior_dim_weight > 1e-6).any().detach().item()):
                boundary_error = (rho_mean - prior_target).pow(2)
                boundary_loss = (boundary_error * prior_dim_weight).sum()
                boundary_loss = boundary_loss / prior_dim_weight.sum().clamp(min=1e-6)
                metrics["structured_joint_rl_prior_target_mean"] = float(
                    prior_target[prior_dim_weight > 1e-6].mean().detach().item()
                )
                metrics["structured_joint_rl_prior_rho_mean"] = float(
                    rho_mean[prior_dim_weight > 1e-6].mean().detach().item()
                )
            rho_loss = repair_loss_scale * repairable_loss + prior_loss_weight * boundary_loss
            prior_loss = boundary_loss
            metrics["structured_joint_rl_repairable_loss"] = float(repairable_loss.detach().item())
            metrics["structured_joint_rl_boundary_loss"] = float(boundary_loss.detach().item())
            metrics["structured_joint_rl_prior_loss"] = metrics["structured_joint_rl_boundary_loss"]
            metrics["structured_joint_rl_repair_loss_scale"] = repair_loss_scale
            metrics["structured_joint_rl_repair_loss_is_bce"] = 1.0 if repair_loss_kind == "bce_logit" else 0.0
        elif (
            prior_loss_weight > 0.0
            and has_prior_inputs
        ):
            if bool((prior_dim_weight > 1e-6).any().detach().item()):
                prior_error = (rho_mean - prior_target).pow(2)
                prior_loss = (prior_error * prior_dim_weight).sum() / prior_dim_weight.sum().clamp(min=1e-6)
                metrics["structured_joint_rl_prior_loss"] = float(prior_loss.detach().item())
                metrics["structured_joint_rl_prior_target_mean"] = float(
                    prior_target[prior_dim_weight > 1e-6].mean().detach().item()
                )
                metrics["structured_joint_rl_prior_rho_mean"] = float(
                    rho_mean[prior_dim_weight > 1e-6].mean().detach().item()
                )
        loss = rho_loss if loss_mode == "region_direct" else rho_loss + prior_loss_weight * prior_loss

        metrics["structured_joint_rl_adv_mean"] = float(
            (rho_adv_raw[rho_active]).mean().detach().item()
        ) if bool(rho_active.any().detach().item()) else 0.0
        metrics["structured_joint_rl_adv_abs_mean"] = float(
            (rho_adv_raw[rho_active]).abs().mean().detach().item()
        ) if bool(rho_active.any().detach().item()) else 0.0
        metrics["structured_joint_rl_adv_used_mean"] = float(
            (rho_adv[rho_active]).mean().detach().item()
        ) if bool(rho_active.any().detach().item()) else 0.0
        metrics["structured_joint_rl_weight_mean"] = float(
            rho_weight[rho_active].mean().detach().item()
        ) if bool(rho_active.any().detach().item()) else 0.0
        metrics["structured_joint_rl_weight_all_mean"] = float(rho_weight.mean().detach().item())
        metrics["structured_joint_rl_rho_adv_mean"] = metrics["structured_joint_rl_adv_mean"]
        metrics["structured_joint_rl_rho_adv_abs_mean"] = metrics["structured_joint_rl_adv_abs_mean"]
        metrics["structured_joint_rl_rho_weight_mean"] = metrics["structured_joint_rl_weight_mean"]
        metrics["structured_joint_rl_rho_weight_all_mean"] = metrics["structured_joint_rl_weight_all_mean"]
        metrics["structured_joint_rl_rho_ratio_mean"] = float(
            rho_ratio.detach()[rho_active].mean().item()
        ) if bool(rho_active.any().detach().item()) else 1.0
        metrics["structured_joint_rl_ratio_mean"] = metrics["structured_joint_rl_rho_ratio_mean"]
        metrics["structured_joint_rl_rho_loss"] = float(rho_loss.detach().item())
        metrics["structured_joint_rl_dim_active_mean"] = float(rho_active.float().mean().detach().item())
        if bool(rho_active.any().detach().item()):
            active_adv_raw = rho_adv_raw[rho_active]
            active_rho_mean = rho_mean[rho_active]
            active_action_raw = rho_action_raw[rho_active]
            active_mean_raw = rho_mean_raw[rho_active]
            active_rho_action = rho_action[rho_active]
            metrics["structured_joint_rl_rho_mean"] = float(active_rho_mean.mean().detach().item())
            metrics["structured_joint_rl_rho_abs_from_half"] = float(
                (active_rho_mean - 0.5).abs().mean().detach().item()
            )
            metrics["structured_joint_rl_rho_near_half_frac"] = float(
                ((active_rho_mean - 0.5).abs() < 0.05).float().mean().detach().item()
            )
            metrics["structured_joint_rl_rho_action_minus_mean_abs"] = float(
                (active_action_raw - active_mean_raw).abs().mean().detach().item()
            )
            metrics["structured_joint_rl_rho_action_minus_mean_rho_abs"] = float(
                (active_rho_action - active_rho_mean).abs().mean().detach().item()
            )
            metrics["structured_joint_rl_adv_pos_frac"] = float(
                (active_adv_raw > 1e-6).float().mean().detach().item()
            )
            metrics["structured_joint_rl_adv_neg_frac"] = float(
                (active_adv_raw < -1e-6).float().mean().detach().item()
            )
            metrics["structured_joint_rl_adv_near_zero_frac"] = float(
                (active_adv_raw.abs() <= 1e-6).float().mean().detach().item()
            )
            pos_adv_mask = rho_active & (rho_adv_raw > 1e-6)
            neg_adv_mask = rho_active & (rho_adv_raw < -1e-6)
            if bool(pos_adv_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_pos_adv_mean"] = float(
                    rho_mean[pos_adv_mask].mean().detach().item()
                )
                metrics["structured_joint_rl_adv_pos_abs_mean"] = float(
                    rho_adv_raw[pos_adv_mask].abs().mean().detach().item()
                )
            if bool(neg_adv_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_neg_adv_mean"] = float(
                    rho_mean[neg_adv_mask].mean().detach().item()
                )
                metrics["structured_joint_rl_adv_neg_abs_mean"] = float(
                    rho_adv_raw[neg_adv_mask].abs().mean().detach().item()
                )
            repairable_mask = repairable_weight > 1e-6 if loss_mode == "region_direct" else rho_active
            if bool(repairable_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_repairable_mean"] = float(
                    rho_mean[repairable_mask].mean().detach().item()
                )
                repairable_adv = rho_adv_raw[repairable_mask]
                metrics["structured_joint_rl_repairable_pos_frac"] = float(
                    (repairable_adv > 1e-6).float().mean().detach().item()
                )
            boundary_mask = prior_dim_weight > 1e-6
            if bool(boundary_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_boundary_mean"] = float(
                    rho_mean[boundary_mask].mean().detach().item()
                )
            dim_ids = torch.arange(cols, device=self.device).view(1, cols)
            pos_dim_mask = rho_active & (dim_ids < min(3, cols))
            rpy_dim_mask = rho_active & (dim_ids >= 3)
            pos_dim_all = dim_ids < min(3, cols)
            rpy_dim_all = dim_ids >= 3
            if bool(pos_dim_all.any().detach().item()):
                metrics["structured_joint_rl_active_frac_pos_dim"] = float(
                    rho_active[pos_dim_all.expand_as(rho_active)].float().mean().detach().item()
                )
            if bool(rpy_dim_all.any().detach().item()):
                metrics["structured_joint_rl_active_frac_rpy_dim"] = float(
                    rho_active[rpy_dim_all.expand_as(rho_active)].float().mean().detach().item()
                )
            if bool(pos_dim_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_pos_dim_mean"] = float(
                    rho_mean[pos_dim_mask].mean().detach().item()
                )
                pos_dim_adv = rho_adv_raw[pos_dim_mask]
                metrics["structured_joint_rl_adv_pos_frac_pos_dim"] = float(
                    (pos_dim_adv > 1e-6).float().mean().detach().item()
                )
            if bool(rpy_dim_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_rpy_dim_mean"] = float(
                    rho_mean[rpy_dim_mask].mean().detach().item()
                )
                rpy_dim_adv = rho_adv_raw[rpy_dim_mask]
                metrics["structured_joint_rl_adv_pos_frac_rpy_dim"] = float(
                    (rpy_dim_adv > 1e-6).float().mean().detach().item()
                )
            pos_adv_pos_dim_mask = pos_adv_mask & pos_dim_mask
            pos_adv_rpy_dim_mask = pos_adv_mask & rpy_dim_mask
            neg_adv_pos_dim_mask = neg_adv_mask & pos_dim_mask
            neg_adv_rpy_dim_mask = neg_adv_mask & rpy_dim_mask
            if bool(pos_adv_pos_dim_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_pos_adv_pos_dim_mean"] = float(
                    rho_mean[pos_adv_pos_dim_mask].mean().detach().item()
                )
            if bool(pos_adv_rpy_dim_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_pos_adv_rpy_dim_mean"] = float(
                    rho_mean[pos_adv_rpy_dim_mask].mean().detach().item()
                )
            if bool(neg_adv_pos_dim_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_neg_adv_pos_dim_mean"] = float(
                    rho_mean[neg_adv_pos_dim_mask].mean().detach().item()
                )
            if bool(neg_adv_rpy_dim_mask.any().detach().item()):
                metrics["structured_joint_rl_rho_neg_adv_rpy_dim_mean"] = float(
                    rho_mean[neg_adv_rpy_dim_mask].mean().detach().item()
                )
        if bool(getattr(self, "frontres_reward_compute_live_debug", False)):
            it = int(getattr(self, "current_learning_iteration", 0))
            interval = int(getattr(self, "frontres_restore_debug_print_interval", 10))
            if interval > 0 and it % interval == 0:
                if loss_mode == "region_direct":
                    print(
                        "[FrontRES reward live loss] "
                        f"it={it} mode=region_direct "
                        f"rep={metrics['structured_joint_rl_repairable_loss']:+.4f} "
                        f"bound={metrics['structured_joint_rl_boundary_loss']:.4f} "
                        f"repair_bce={metrics['structured_joint_rl_repair_loss_is_bce']:.0f} "
                        f"rscale={metrics['structured_joint_rl_repair_loss_scale']:.3f} "
                        f"adv={metrics['structured_joint_rl_adv_mean']:+.4f} "
                        f"|adv|={metrics['structured_joint_rl_adv_abs_mean']:.4f} "
                        f"weight={metrics['structured_joint_rl_weight_mean']:.3f} "
                        f"p_auth={metrics['structured_joint_rl_prior_authority_mean']:.3f} "
                        f"r_auth={metrics['structured_joint_rl_repairable_authority_mean']:.3f} "
                        f"p_rho={metrics['structured_joint_rl_prior_rho_mean']:.3f} "
                        f"rho_mean={metrics['structured_joint_rl_rho_mean']:.3f} "
                        f"rho_loss={metrics['structured_joint_rl_rho_loss']:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        "[FrontRES reward live loss] "
                        f"it={it} mode=ppo_clipped "
                        f"adv={metrics['structured_joint_rl_adv_mean']:+.4f} "
                        f"|adv|={metrics['structured_joint_rl_adv_abs_mean']:.4f} "
                        f"weight={metrics['structured_joint_rl_weight_mean']:.3f} "
                        f"prior_loss={metrics['structured_joint_rl_prior_loss']:.4f} "
                        f"prior_auth={metrics['structured_joint_rl_prior_authority_mean']:.3f} "
                        f"prior_rho={metrics['structured_joint_rl_prior_rho_mean']:.3f} "
                        f"rho_mean={metrics['structured_joint_rl_rho_mean']:.3f} "
                        f"rho_|.5|={metrics['structured_joint_rl_rho_abs_from_half']:.3f} "
                        f"rho_act-mu={metrics['structured_joint_rl_rho_action_minus_mean_abs']:.4f} "
                        f"rho_loss={metrics['structured_joint_rl_rho_loss']:.4f}",
                        flush=True,
                    )
        return loss, metrics

    def _compute_state_alpha_loss(
        self,
        obs_batch,
        state_alpha_target_batch,
        state_alpha_mask_batch,
        original_batch_size,
    ):
        zero = obs_batch[:original_batch_size].sum() * 0.0
        metrics = {
            "state_alpha_mask_frac": 0.0,
            "state_alpha_target_mean": 0.0,
            "state_alpha_pred_mean": 0.0,
            "state_alpha_abs_err": 0.0,
            "state_alpha_acc": 0.0,
        }
        if (
            self.frontres_state_alpha_weight <= 0.0
            or state_alpha_target_batch is None
            or state_alpha_mask_batch is None
            or not hasattr(self.policy, "get_state_router_logit")
        ):
            return zero, metrics

        target = state_alpha_target_batch[:original_batch_size, :1].to(
            device=self.device, dtype=obs_batch.dtype
        ).detach()
        mask = state_alpha_mask_batch[:original_batch_size, :1].to(
            device=self.device, dtype=obs_batch.dtype
        ).detach()
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        denom = mask.sum()
        if float(denom.detach().item()) <= 1e-6:
            return zero, metrics

        logits = self.policy.get_state_router_logit(obs_batch[:original_batch_size])
        loss_terms = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        loss = (loss_terms * mask).sum() / denom.clamp(min=1e-6)
        pred = torch.sigmoid(logits.detach())
        active = mask > 0
        pred_active = pred[active]
        target_active = target[active]
        metrics["state_alpha_mask_frac"] = float(mask.mean().detach().item())
        metrics["state_alpha_target_mean"] = float(target_active.mean().detach().item())
        metrics["state_alpha_pred_mean"] = float(pred_active.mean().detach().item())
        metrics["state_alpha_abs_err"] = float((pred_active - target_active).abs().mean().detach().item())
        metrics["state_alpha_acc"] = float(((pred_active >= 0.5) == (target_active >= 0.5)).float().mean().detach().item())
        return loss, metrics

    def _compute_supervised_loss(
        self,
        mu_batch,
        supervised_target_batch,
        original_batch_size,
        batch_indices=None,
        supervised_weight_batch=None,
        supervised_harm_weight_batch=None,
    ):
        supervised_loss = torch.tensor(0.0, device=self.device)
        sup_cos_sim = 0.0
        sup_metrics = {
            "supervised_mae": 0.0,
            "supervised_rmse": 0.0,
            "supervised_rpy_mae": 0.0,
            "supervised_rpy_rmse": 0.0,
            "supervised_restore_ratio": 0.0,
            "supervised_valid_frac": 0.0,
            "supervised_l_pos": 0.0,
            "supervised_l_rot": 0.0,
            "supervised_l_mag": 0.0,
            "supervised_l_over": 0.0,
            "supervised_l_smooth": 0.0,
            "supervised_l_sparse": 0.0,
            "supervised_l_miss": 0.0,
            "supervised_l_coeff_smooth": 0.0,
            "supervised_l_harm": 0.0,
            "supervised_l_conf": 0.0,
            "frontres_alpha_mean": 0.0,
            "frontres_alpha_active_frac": 0.0,
            "frontres_tau_mean": 0.0,
            "frontres_tau_active_frac": 0.0,
            "frontres_rho_pos_mean": 0.0,
            "frontres_rho_pos_active_frac": 0.0,
            "frontres_write_ratio": 0.0,
            "frontres_proposal_ratio": 0.0,
            "frontres_axis_leakage": 0.0,
            "frontres_supervised_weight": 0.0,
        }
        if supervised_target_batch is None or self.lambda_supervised <= 0:
            return supervised_loss, sup_cos_sim, sup_metrics

        mu_dim = mu_batch.shape[-1]
        sup_dim = supervised_target_batch.shape[-1]
        if mu_dim < sup_dim:
            return supervised_loss, sup_cos_sim, sup_metrics

        raw_pred = mu_batch[:original_batch_size]
        target = supervised_target_batch[:original_batch_size]
        if supervised_weight_batch is not None:
            sample_weight = supervised_weight_batch[:original_batch_size].view(-1).to(
                device=self.device, dtype=raw_pred.dtype
            )
        else:
            sample_weight = torch.ones(raw_pred.shape[0], device=self.device, dtype=raw_pred.dtype)
        sample_weight = torch.nan_to_num(sample_weight, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        if supervised_harm_weight_batch is not None:
            harm_weight = supervised_harm_weight_batch[:original_batch_size].view(-1).to(
                device=self.device, dtype=raw_pred.dtype
            )
            harm_weight = torch.nan_to_num(harm_weight, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        else:
            harm_weight = torch.zeros(raw_pred.shape[0], device=self.device, dtype=raw_pred.dtype)

        def _wmean(values: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
            w = sample_weight if weight is None else sample_weight * weight.to(sample_weight.dtype)
            return (values * w).sum() / w.sum().clamp(min=1e-6)

        is_task_space = hasattr(self.policy, "num_task_corrections") and self.policy.num_task_corrections > 0
        if is_task_space:
            proposal = torch.cat([
                torch.tanh(raw_pred[:, :3]) * self.policy.max_delta_pos,
                torch.tanh(raw_pred[:, 3:6]) * self.policy.max_delta_rpy,
            ], dim=-1)
        else:
            proposal = raw_pred

        if hasattr(self.policy, "max_delta_pos") and hasattr(self.policy, "max_delta_rpy"):
            target = torch.cat([
                target[:, :3].clamp(-self.policy.max_delta_pos, self.policy.max_delta_pos),
                target[:, 3:].clamp(-self.policy.max_delta_rpy, self.policy.max_delta_rpy),
            ], dim=-1)

        coeff = None
        scalar_trust = False
        acceptance_only = False
        tau_logits = None
        tau_value = None
        objective = str(self.frontres_training_objective).lower()
        if is_task_space and objective in ("basis_restore", "hsl_hybrid"):
            coeff_dim = int(getattr(self.policy, "task_conf_dim", 2))
            if coeff_dim == 1 and raw_pred.shape[-1] >= 7:
                tau_logits = raw_pred[:, 6:7]
                tau_value = torch.sigmoid(tau_logits)
                coeff = tau_value.expand(-1, proposal.shape[-1])
                scalar_trust = True
                # Legacy scalar path.  The active hsl_hybrid contract now uses
                # six acceptance heads, but this keeps older checkpoints/scripts
                # loadable.
                pred = proposal
            elif coeff_dim == 6 and raw_pred.shape[-1] >= 12:
                coeff = torch.sigmoid(raw_pred[:, 6:12])
                pred = proposal * coeff
                acceptance_only = objective == "hsl_hybrid"
            elif raw_pred.shape[-1] >= 8:
                c_pos = torch.sigmoid(raw_pred[:, 6:7])
                c_rpy = torch.sigmoid(raw_pred[:, 7:8])
                coeff = torch.cat([c_pos.expand(-1, 3), c_rpy.expand(-1, 3)], dim=-1)
                pred = proposal * coeff
            else:
                pred = proposal
        else:
            pred = proposal

        active_dims = getattr(self, "frontres_active_task_dims", None)
        if (
            active_dims is not None
            and is_task_space
        ):
            mask = torch.zeros(pred.shape[-1], device=self.device, dtype=pred.dtype)
            for idx in active_dims:
                idx = int(idx)
                if 0 <= idx < 6:
                    mask[idx] = 1.0
            proposal = proposal * mask.view(1, -1)
            pred = pred * mask.view(1, -1)
            target = target * mask.view(1, -1)

        written_pred = pred
        supervised_pred = proposal if (scalar_trust or acceptance_only) else pred
        target_detached = target.detach()
        target_norm = target_detached.norm(dim=-1)
        valid = target_norm > 1e-4

        pos_valid = target_detached[:, :3].norm(dim=-1) > 1e-4
        rpy_valid = target_detached[:, 3:].norm(dim=-1) > 1e-4
        pos_weight = torch.ones_like(target_norm)
        rpy_weight = torch.ones_like(target_norm)
        valid_weight = float(self.supervised_valid_loss_weight)
        if pos_valid.any():
            pos_weight[pos_valid] = valid_weight
        if rpy_valid.any():
            rpy_weight[rpy_valid] = valid_weight
        pos_weight = pos_weight / pos_weight.mean().clamp(min=1e-6)
        rpy_weight = rpy_weight / rpy_weight.mean().clamp(min=1e-6)

        dir_pred = proposal if coeff is not None else pred
        pos_err = nn.functional.huber_loss(
            dir_pred[:, :3], target_detached[:, :3], reduction="none").mean(dim=-1)
        rpy_err = nn.functional.huber_loss(
            dir_pred[:, 3:], target_detached[:, 3:], reduction="none").mean(dim=-1)
        pos_sup = _wmean(pos_err, pos_weight)
        rpy_sup = _wmean(rpy_err, rpy_weight)
        supervised_loss = pos_sup + self.supervised_rpy_loss_weight * rpy_sup

        if coeff is not None and not scalar_trust and not acceptance_only:
            write_err = nn.functional.huber_loss(written_pred, target_detached, reduction="none").mean(dim=-1)
            supervised_loss = supervised_loss + _wmean(write_err)

        mag_loss = torch.zeros((), device=self.device)
        over_loss = torch.zeros((), device=self.device)
        smooth_loss = torch.zeros((), device=self.device)
        sparse_loss = torch.zeros((), device=self.device)
        miss_loss = torch.zeros((), device=self.device)
        coeff_smooth_loss = torch.zeros((), device=self.device)
        harm_loss = torch.zeros((), device=self.device)
        conf_sup = torch.zeros((), device=self.device)
        if self.supervised_magnitude_loss_weight > 0 and valid.any():
            pred_norm = supervised_pred.norm(dim=-1)
            target_norm_valid = target_detached.norm(dim=-1)
            mag_terms = nn.functional.huber_loss(pred_norm, target_norm_valid, reduction="none")
            mag_loss = _wmean(mag_terms, valid.to(sample_weight.dtype))
            supervised_loss = supervised_loss + self.supervised_magnitude_loss_weight * mag_loss
        if self.supervised_over_loss_weight > 0 and valid.any():
            pred_norm = supervised_pred.norm(dim=-1)
            target_norm_valid = target_detached.norm(dim=-1)
            over_terms = torch.relu(pred_norm - target_norm_valid).square()
            over_loss = _wmean(over_terms, valid.to(sample_weight.dtype))
            supervised_loss = supervised_loss + self.supervised_over_loss_weight * over_loss
        if (
            self.supervised_smooth_loss_weight > 0
            and batch_indices is not None
            and getattr(self.storage, "num_envs", 0) > 0
        ):
            smooth_loss = self._compute_temporal_smooth_loss(
                supervised_pred, target_detached, batch_indices[:original_batch_size])
            supervised_loss = supervised_loss + self.supervised_smooth_loss_weight * smooth_loss

        if coeff is not None:
            target_axis_mask = (target_detached.abs() > 1e-4).to(coeff.dtype)
            if self.supervised_coeff_sparse_weight > 0 and not scalar_trust and not acceptance_only:
                sparse_terms = (coeff * (1.0 - target_axis_mask)).mean(dim=-1)
                sparse_loss = _wmean(sparse_terms)
                supervised_loss = supervised_loss + self.supervised_coeff_sparse_weight * sparse_loss
            if self.supervised_coeff_miss_weight > 0 and not scalar_trust and not acceptance_only:
                miss_terms = ((1.0 - coeff) * target_axis_mask).mean(dim=-1)
                miss_loss = _wmean(miss_terms)
                supervised_loss = supervised_loss + self.supervised_coeff_miss_weight * miss_loss
            if (
                self.supervised_coeff_smooth_weight > 0
                and not acceptance_only
                and batch_indices is not None
                and getattr(self.storage, "num_envs", 0) > 0
            ):
                coeff_smooth_loss = self._compute_temporal_step_loss(
                    coeff, batch_indices[:original_batch_size])
                supervised_loss = supervised_loss + self.supervised_coeff_smooth_weight * coeff_smooth_loss

        if self.supervised_harm_loss_weight > 0 and float(harm_weight.sum().detach().item()) > 0.0:
            # In hsl_hybrid, acceptance is PPO-owned.  The supervised harmful
            # penalty should suppress unsafe proposal directions without
            # directly training the acceptance head.
            harm_base = proposal if acceptance_only else (written_pred if coeff is not None else pred)
            harm_terms = harm_base.square().mean(dim=-1)
            harm_loss = (harm_terms * harm_weight).sum() / harm_weight.sum().clamp(min=1e-6)
            supervised_loss = supervised_loss + self.supervised_harm_loss_weight * harm_loss

        if self.supervised_direction_loss_weight > 0:
            direction_loss = torch.zeros((), device=self.device)
            if pos_valid.any():
                direction_loss = direction_loss + (
                    _wmean(
                        1.0 - nn.functional.cosine_similarity(
                            dir_pred[:, :3], target_detached[:, :3], dim=-1
                        ),
                        pos_valid.to(sample_weight.dtype),
                    )
                )
            if rpy_valid.any():
                direction_loss = direction_loss + (
                    _wmean(
                        1.0 - nn.functional.cosine_similarity(
                            dir_pred[:, 3:], target_detached[:, 3:], dim=-1
                        ),
                        rpy_valid.to(sample_weight.dtype),
                    )
                )
            supervised_loss = supervised_loss + self.supervised_direction_loss_weight * direction_loss

        if (
            scalar_trust
            and tau_logits is not None
            and str(self.frontres_training_objective).lower() != "hsl_hybrid"
            and self.supervised_conf_loss_weight > 0
        ):
            pass_weight = sample_weight.view(-1, 1)
            reject_weight = harm_weight.view(-1, 1)
            pass_loss = torch.zeros((), device=self.device)
            reject_loss = torch.zeros((), device=self.device)
            if float(pass_weight.sum().detach().item()) > 0.0:
                pass_target = torch.ones_like(tau_logits)
                pass_terms = nn.functional.binary_cross_entropy_with_logits(
                    tau_logits, pass_target, reduction="none")
                pass_loss = (pass_terms * pass_weight).sum() / pass_weight.sum().clamp(min=1e-6)
            if float(reject_weight.sum().detach().item()) > 0.0:
                reject_target = torch.zeros_like(tau_logits)
                reject_terms = nn.functional.binary_cross_entropy_with_logits(
                    tau_logits, reject_target, reduction="none")
                reject_loss = (reject_terms * reject_weight).sum() / reject_weight.sum().clamp(min=1e-6)
            conf_sup = pass_loss + reject_loss
            supervised_loss = supervised_loss + self.supervised_conf_loss_weight * conf_sup
        elif (
            hasattr(self.policy, "num_task_corrections")
            and self.policy.num_task_corrections > 0
            and raw_pred.shape[-1] >= 8
            and self.supervised_conf_loss_weight > 0
            and int(getattr(self.policy, "task_conf_dim", 2)) == 2
        ):
            target_conf = valid.view(-1, 1).to(raw_pred.dtype)
            conf_sup = nn.functional.binary_cross_entropy_with_logits(
                raw_pred[:, 6:8], target_conf.expand(-1, 2).detach())
            supervised_loss = supervised_loss + self.supervised_conf_loss_weight * conf_sup

        with torch.no_grad():
            err = supervised_pred - target_detached
            written_err = written_pred - target_detached
            sup_metrics["supervised_mae"] = err.abs().mean().item()
            sup_metrics["supervised_rmse"] = err.square().mean().sqrt().item()
            sup_metrics["supervised_l_pos"] = pos_sup.item()
            sup_metrics["supervised_l_rot"] = rpy_sup.item()
            sup_metrics["supervised_l_mag"] = mag_loss.item()
            sup_metrics["supervised_l_over"] = over_loss.item()
            sup_metrics["supervised_l_smooth"] = smooth_loss.item()
            sup_metrics["supervised_l_sparse"] = sparse_loss.item()
            sup_metrics["supervised_l_miss"] = miss_loss.item()
            sup_metrics["supervised_l_coeff_smooth"] = coeff_smooth_loss.item()
            sup_metrics["supervised_l_harm"] = harm_loss.item()
            sup_metrics["supervised_l_conf"] = conf_sup.item()
            sup_metrics["frontres_supervised_weight"] = sample_weight.mean().item()
            if coeff is not None:
                sup_metrics["frontres_alpha_mean"] = coeff.mean().item()
                sup_metrics["frontres_alpha_active_frac"] = (coeff > 0.5).float().mean().item()
                if scalar_trust:
                    rho_pos = coeff[:, :1]
                    sup_metrics["frontres_rho_pos_mean"] = rho_pos.mean().item()
                    sup_metrics["frontres_rho_pos_active_frac"] = (rho_pos > 0.5).float().mean().item()
                    # Legacy aliases kept for old TensorBoard dashboards.
                    sup_metrics["frontres_tau_mean"] = sup_metrics["frontres_rho_pos_mean"]
                    sup_metrics["frontres_tau_active_frac"] = sup_metrics["frontres_rho_pos_active_frac"]
                elif acceptance_only and coeff.shape[-1] >= 6:
                    sup_metrics["frontres_accept_pos_mean"] = coeff[:, :3].mean().item()
                    sup_metrics["frontres_accept_rpy_mean"] = coeff[:, 3:6].mean().item()
                    sup_metrics["frontres_accept_active_frac"] = (coeff > 0.5).float().mean().item()
                pred_norm_all = written_pred.norm(dim=-1)
                proposal_norm_all = proposal.norm(dim=-1)
                target_norm_all = target_detached.norm(dim=-1).clamp(min=1e-6)
                sup_metrics["frontres_write_ratio"] = (pred_norm_all / target_norm_all).mean().item()
                sup_metrics["frontres_proposal_ratio"] = (proposal_norm_all / target_norm_all).mean().item()
                inactive = (target_detached.abs() <= 1e-4).to(pred.dtype)
                leakage_num = (written_pred.abs() * inactive).sum(dim=-1)
                leakage_den = written_pred.abs().sum(dim=-1).clamp(min=1e-6)
                sup_metrics["frontres_axis_leakage"] = (leakage_num / leakage_den).mean().item()
            if err.shape[-1] >= 6:
                rpy_err_vec = err[:, 3:6]
                sup_metrics["supervised_rpy_mae"] = rpy_err_vec.abs().mean().item()
                sup_metrics["supervised_rpy_rmse"] = rpy_err_vec.square().mean().sqrt().item()
            sup_metrics["supervised_valid_frac"] = valid.float().mean().item()
            if valid.any():
                sup_cos_sim = nn.functional.cosine_similarity(
                    supervised_pred[valid], target[valid], dim=-1).mean().item()
                residual_norm = written_err[valid].norm(dim=-1)
                target_norm_valid = target_detached[valid].norm(dim=-1).clamp(min=1e-6)
                restore_ratio = 1.0 - residual_norm / target_norm_valid
                sup_metrics["supervised_restore_ratio"] = restore_ratio.mean().item()
        return supervised_loss, sup_cos_sim, sup_metrics

    def _compute_temporal_smooth_loss(self, pred, target, batch_indices):
        """Match correction first differences on adjacent rollout samples."""
        if batch_indices is None or batch_indices.numel() < 2:
            return torch.zeros((), device=self.device)
        num_envs = int(getattr(self.storage, "num_envs", 0))
        if num_envs <= 0:
            return torch.zeros((), device=self.device)

        idx = batch_indices.to(device=self.device, dtype=torch.long).view(-1)
        sorted_idx, order = torch.sort(idx)
        pred_sorted = pred[order]
        target_sorted = target[order]

        next_idx = sorted_idx + num_envs
        pos = torch.searchsorted(sorted_idx, next_idx)
        safe_pos = pos.clamp(max=sorted_idx.numel() - 1)
        found = (pos < sorted_idx.numel()) & (sorted_idx[safe_pos] == next_idx)
        if not found.any():
            return torch.zeros((), device=self.device)

        cur = torch.nonzero(found, as_tuple=False).squeeze(-1)
        nxt = pos[cur]
        pred_diff = pred_sorted[nxt] - pred_sorted[cur]
        target_diff = target_sorted[nxt] - target_sorted[cur]
        return nn.functional.huber_loss(pred_diff, target_diff.detach(), reduction="mean")

    def _compute_temporal_step_loss(self, values, batch_indices):
        """Penalize first differences on adjacent rollout samples."""
        if batch_indices is None or batch_indices.numel() < 2:
            return torch.zeros((), device=self.device)
        num_envs = int(getattr(self.storage, "num_envs", 0))
        if num_envs <= 0:
            return torch.zeros((), device=self.device)

        idx = batch_indices.to(device=self.device, dtype=torch.long).view(-1)
        sorted_idx, order = torch.sort(idx)
        values_sorted = values[order]

        next_idx = sorted_idx + num_envs
        pos = torch.searchsorted(sorted_idx, next_idx)
        safe_pos = pos.clamp(max=sorted_idx.numel() - 1)
        found = (pos < sorted_idx.numel()) & (sorted_idx[safe_pos] == next_idx)
        if not found.any():
            return torch.zeros((), device=self.device)

        cur = torch.nonzero(found, as_tuple=False).squeeze(-1)
        nxt = pos[cur]
        return (values_sorted[nxt] - values_sorted[cur]).square().mean()

    def _warn_skip(self, reason: str, loss: torch.Tensor | None = None):
        skip_count = getattr(self, "_nan_skip_count", 0) + 1
        self._nan_skip_count = skip_count
        if skip_count <= 5 or skip_count % 100 == 0:
            suffix = f" ({loss.item():.4g})" if loss is not None else ""
            print(f"[FrontRESUnified] WARNING: {reason}{suffix}, skipping update (skip #{skip_count})")

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
                offset += numel
