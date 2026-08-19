from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.modules import ActorCritic, FrontRESActorCritic, ResidualActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.modules.frontres_observation_layout import (
    FRONTRES_FUTURE_INTENT_LAYOUT_VERSION,
    resolve_frontres_future_intent_layout,
)
from rsl_rl.frontres.frontres_segment_warmup import (
    frontres_k_stage_schedule_fingerprint,
    frontres_k_stage_schedule_tuple,
    normalize_frontres_k_stage_schedule,
    require_frontres_v013_campaign_schedule,
)
from rsl_rl.frontres.frontres_value_normalization import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FRONTRES_VALUE_NORMALIZER_DECAY,
    FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
    FrontRESValueNormalizerState,
)


def _frontres_v013_schedule_owners():
    return (
        normalize_frontres_k_stage_schedule,
        frontres_k_stage_schedule_tuple,
        frontres_k_stage_schedule_fingerprint,
        require_frontres_v013_campaign_schedule,
    )


def validate_frontres_v015_stage3_supervision_config(
    *,
    future_offsets,
    lambda_supervised: float,
    lambda_supervised_min: float,
) -> None:
    """Reject online HSL loss whenever the v015 future-intent route is selected."""

    offsets = tuple(int(value) for value in (future_offsets or ()))
    if not offsets:
        return
    resolve_frontres_future_intent_layout(offsets, FRONTRES_FUTURE_INTENT_LAYOUT_VERSION)
    if abs(float(lambda_supervised)) > 1.0e-12 or abs(float(lambda_supervised_min)) > 1.0e-12:
        raise ValueError(
            "FRS-TRAIN-v021 requires lambda_supervised=0 and lambda_supervised_min=0 "
            "for the v015 future-intent Stage-3 route; HSL is initialization-only"
        )


class FrontRESTrackedAdam(optim.Adam):
    """Count every real v015 optimizer step in persisted optimizer state."""

    _STEP_COUNT_KEY = "frontres_step_count"

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        for group in self.param_groups:
            group[self._STEP_COUNT_KEY] = 0

    @property
    def frontres_step_count(self) -> int:
        counts = []
        for group in self.param_groups:
            value = group.get(self._STEP_COUNT_KEY)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError("v015 Adam state is missing a valid persisted optimizer step counter")
            counts.append(value)
        if not counts or len(set(counts)) != 1:
            raise RuntimeError("v015 Adam parameter groups disagree on the optimizer step counter")
        return counts[0]

    def step(self, closure=None):
        result = super().step(closure=closure)
        next_count = self.frontres_step_count + 1
        for group in self.param_groups:
            group[self._STEP_COUNT_KEY] = next_count
        return result


class FrontRESUnified:
    """FrontRES PPO plus legacy supervised ΔSE3 support.

    This class intentionally owns only the pieces FrontRES needs:
    on-policy PPO, optional legacy ΔSE3 supervision, reference velocity estimation,
    and the split-env FrontRES mask. v015 future-intent Stage 3 rejects the
    online supervision branch; Stage-1 HSL is handled by the warmup owner.
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
        critic_learning_rate: float | None = None,
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
        supervised_direction_loss_weight: float = 0.1,
        supervised_valid_loss_weight: float = 4.0,
        supervised_magnitude_loss_weight: float = 0.0,
        supervised_over_loss_weight: float = 0.0,
        supervised_smooth_loss_weight: float = 0.0,
        supervised_harm_loss_weight: float = 1.0,
        frontres_supervised_lr_schedule: str = "fixed",
        frontres_supervised_lr_start: float | None = None,
        frontres_supervised_lr_peak: float | None = None,
        frontres_supervised_lr_min: float | None = None,
        frontres_supervised_lr_warmup_iters: int = 0,
        frontres_supervised_lr_cosine_iters: int = 1000,
        frontres_restore_debug_print_interval: int = 10,
        frontres_training_objective: str = "supervised_restore",
        frontres_relational_actor_only: bool = False,
        frontres_actor_only_lr_init_transactions: int = 100,
        frontres_actor_only_lr_ramp_transactions: int = 50,
        frontres_segment_replay_enabled: bool = False,
        frontres_policy_quality_eval_only: bool = False,
        frontres_segment_live_runner_enabled: bool = False,
        frontres_segment_live_sentinel_only: bool = False,
        frontres_local_sentinel_only: bool = False,
        frontres_segment_live_probe_only: bool = False,
        frontres_segment_live_storage_write_only: bool = False,
        frontres_segment_live_single_update_only: bool = False,
        frontres_segment_live_update_loop_only: bool = False,
        frontres_segment_live_train_enabled: bool = False,
        frontres_formal_transaction_enabled: bool = False,
        frontres_segment_live_update_steps: int = 4,
        frontres_segment_critic_warmup_iterations: int = 0,
        frontres_segment_actor_warmup_iterations: int = 0,
        frontres_segment_actor_joint_lr: float = 1.0e-6,
        frontres_segment_k_curriculum: tuple[tuple[object, ...], ...] = (),
        frontres_formal_runtime_audit: bool = False,
        frontres_segment_live_fail_on_invalid_update: bool = True,
        frontres_segment_live_min_valid_count: int = 1,
        frontres_segment_live_fail_on_nonfinite: bool = True,
        frontres_hsl_init_enabled: bool = False,
        frontres_hsl_rollout_label_enabled: bool = False,
        frontres_segment_k: int = 8,
        frontres_future_offsets: tuple[int, ...] = (),
        frontres_future_intent_layout_version: str = "frontres-v015-future-intent-q29-v1",
        frontres_segment_max_horizon_k: int = 64,
        frontres_segment_advantage_normalization: str = "scale_only",
        frontres_critic_value_normalization: str = FRONTRES_VALUE_NORMALIZATION_ID,
        frontres_critic_value_normalizer_decay: float = FRONTRES_VALUE_NORMALIZER_DECAY,
        frontres_critic_value_normalizer_scale_floor: float = FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
        frontres_gain_beta: float = 0.02,
        frontres_segment_cache_dir: str = "",
        frontres_segment_shard_cache_size: int = 8,
        frontres_segment_include_boundary_diagnostic: bool = False,
        frontres_segment_sampler_global_frac: float = 0.4,
        frontres_segment_sampler_replay_frac: float = 0.5,
        frontres_segment_sampler_review_frac: float = 0.1,
        frontres_segment_reset_mode: str = "auto",
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
            raise ValueError("FrontRESUnified supports only hybrid=True HSL training.")
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

        relational_actor_only = bool(
            frontres_relational_actor_only
            or str(frontres_training_objective).lower()
            in {"segment_replay_relational", "segment_replay_relational_preference_v014"}
        )
        if relational_actor_only and not frontres_formal_transaction_enabled:
            raise ValueError("relational Actor-only mode requires formal transaction isolation")
        strict_split_lr = bool(frontres_formal_transaction_enabled and not frontres_policy_quality_eval_only)
        if strict_split_lr:
            if str(schedule).lower() != "fixed":
                raise ValueError("FRS-TRAIN-v021 requires schedule='fixed' for Stage-3 training")
            actor_lr = self._require_positive_finite_lr(learning_rate, name="actor_learning_rate")
            critic_lr = (
                None
                if relational_actor_only
                else self._require_positive_finite_lr(critic_learning_rate, name="critic_learning_rate")
            )
            trainable_params = self._collect_trainable_param_groups(
                policy,
                actor_learning_rate=actor_lr,
                critic_learning_rate=critic_lr,
                actor_only=relational_actor_only,
            )
        else:
            actor_lr = self._require_positive_finite_lr(learning_rate, name="learning_rate")
            critic_lr = (
                self._require_positive_finite_lr(critic_learning_rate, name="critic_learning_rate")
                if frontres_formal_transaction_enabled
                else None
            )
            trainable_params = self._collect_trainable_params(policy)
        optimizer_type = FrontRESTrackedAdam if frontres_formal_transaction_enabled else optim.Adam
        self.optimizer = optimizer_type(trainable_params, lr=actor_lr)

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
        self.learning_rate = actor_lr
        self.actor_learning_rate = actor_lr
        self.critic_learning_rate = critic_lr
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.lambda_supervised = lambda_supervised
        self.lambda_supervised_min = lambda_supervised_min
        self.lambda_supervised_decay_rate = lambda_supervised_decay
        self.supervised_trigger_cosine_sim = supervised_trigger_cosine_sim
        self.supervised_rpy_loss_weight = supervised_rpy_loss_weight
        self.supervised_direction_loss_weight = supervised_direction_loss_weight
        self.supervised_valid_loss_weight = supervised_valid_loss_weight
        self.supervised_magnitude_loss_weight = float(supervised_magnitude_loss_weight)
        self.supervised_over_loss_weight = float(supervised_over_loss_weight)
        self.supervised_smooth_loss_weight = float(supervised_smooth_loss_weight)
        self.supervised_harm_loss_weight = float(supervised_harm_loss_weight)
        self.frontres_supervised_lr_schedule = str(frontres_supervised_lr_schedule).lower()
        self.frontres_supervised_lr_start = float(frontres_supervised_lr_start) if frontres_supervised_lr_start is not None else float(learning_rate)
        self.frontres_supervised_lr_peak = float(frontres_supervised_lr_peak) if frontres_supervised_lr_peak is not None else float(learning_rate)
        self.frontres_supervised_lr_min = float(frontres_supervised_lr_min) if frontres_supervised_lr_min is not None else float(learning_rate)
        self.frontres_supervised_lr_warmup_iters = int(frontres_supervised_lr_warmup_iters)
        self.frontres_supervised_lr_cosine_iters = int(frontres_supervised_lr_cosine_iters)
        self.frontres_restore_debug_print_interval = int(frontres_restore_debug_print_interval)
        self.frontres_training_objective = str(frontres_training_objective).lower()
        self.frontres_relational_actor_only = relational_actor_only
        self.frontres_actor_only_lr_init_transactions = int(frontres_actor_only_lr_init_transactions)
        self.frontres_actor_only_lr_ramp_transactions = int(frontres_actor_only_lr_ramp_transactions)
        if not 50 <= self.frontres_actor_only_lr_init_transactions <= 100:
            raise ValueError("v014 Actor LR init transactions must be in [50,100]")
        if self.frontres_actor_only_lr_ramp_transactions < 2:
            raise ValueError("v014 Actor LR ramp transactions must be at least two")
        self.frontres_segment_replay_enabled = bool(frontres_segment_replay_enabled)
        self.frontres_policy_quality_eval_only = bool(frontres_policy_quality_eval_only)
        self.frontres_segment_live_runner_enabled = bool(frontres_segment_live_runner_enabled)
        self.frontres_segment_live_sentinel_only = bool(frontres_segment_live_sentinel_only)
        self.frontres_local_sentinel_only = bool(frontres_local_sentinel_only)
        self.frontres_segment_live_probe_only = bool(frontres_segment_live_probe_only)
        self.frontres_segment_live_storage_write_only = bool(frontres_segment_live_storage_write_only)
        self.frontres_segment_live_single_update_only = bool(frontres_segment_live_single_update_only)
        self.frontres_segment_live_update_loop_only = bool(frontres_segment_live_update_loop_only)
        self.frontres_segment_live_train_enabled = bool(frontres_segment_live_train_enabled)
        self.frontres_formal_transaction_enabled = bool(frontres_formal_transaction_enabled)
        if self.frontres_relational_actor_only:
            self.frontres_method_contract_id = "FRS-METHOD-v026"
            self.frontres_gain_contract_id = "FRS-GAIN-v009"
            self.frontres_optimization_contract_id = (
                "FRS-PPO-v014"
                if self.frontres_training_objective == "segment_replay_relational_preference_v014"
                else "FRS-PPO-v013"
            )
            self.frontres_training_contract_id = "FRS-TRAIN-v025"
        else:
            self.frontres_method_contract_id = "FRS-METHOD-v025"
            self.frontres_gain_contract_id = "FRS-GAIN-v008"
            self.frontres_optimization_contract_id = "FRS-PPO-v012"
            self.frontres_training_contract_id = "FRS-TRAIN-v024"
        self.frontres_dr_curriculum_schema_id = "nested-k-dr-four-class-v1"
        self.frontres_scalar_target_id = "none" if self.frontres_relational_actor_only else "symmetric-log-recovery-aware-utility-v1"
        self.frontres_physics_schema_id = (
            "hierarchical-relational-evidence-v1"
            if self.frontres_relational_actor_only
            else "clean-anchored-contact-zmp-survival-v1"
        )
        self.frontres_grouped_schema_id = (
            "relational-preference-edge-v1" if self.frontres_relational_actor_only else "grouped-all-attempt-scalar-v1"
        )
        self.frontres_critic_value_kind = "inert-legacy-compat" if self.frontres_relational_actor_only else "state_value"
        self.frontres_critic_input_dim = 449
        self.frontres_critic_action_conditioned = False
        self.frontres_critic_target_id = "none" if self.frontres_relational_actor_only else "scenario-current-exact-m4-mean-symlog-v1"
        self.frontres_return_utility_id = "none" if self.frontres_relational_actor_only else "symmetric-log-gain-g0-1-v1"
        self.frontres_return_utility_scale = 1.0
        self.frontres_critic_support_context_id = (
            "none" if self.frontres_relational_actor_only else "action-pre-support-plan-kmax32-v1"
        )
        self.frontres_gradient_clip_identity = (
            "actor-only-relational-v1" if self.frontres_relational_actor_only else "separate-actor-critic-v1"
        )
        self.frontres_critic_value_normalization = str(frontres_critic_value_normalization).lower()
        self.frontres_critic_value_normalizer_decay = float(frontres_critic_value_normalizer_decay)
        self.frontres_critic_value_normalizer_scale_floor = float(frontres_critic_value_normalizer_scale_floor)
        if self.frontres_formal_transaction_enabled and not self.frontres_relational_actor_only:
            if self.frontres_critic_value_normalization != FRONTRES_VALUE_NORMALIZATION_ID:
                raise ValueError(
                    f"FRS-TRAIN-v021 requires frontres_critic_value_normalization={FRONTRES_VALUE_NORMALIZATION_ID}"
                )
            if self.frontres_critic_value_normalizer_decay != FRONTRES_VALUE_NORMALIZER_DECAY:
                raise ValueError("FRS-TRAIN-v021 requires value-normalizer decay=0.9")
            if self.frontres_critic_value_normalizer_scale_floor != FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR:
                raise ValueError("FRS-TRAIN-v021 requires value-normalizer scale floor=1.0")
        self.frontres_critic_value_normalizer_state = FrontRESValueNormalizerState()
        self.frontres_segment_live_update_steps = max(1, int(frontres_segment_live_update_steps))
        self.frontres_segment_critic_warmup_iterations = max(0, int(frontres_segment_critic_warmup_iterations))
        self.frontres_segment_actor_warmup_iterations = max(0, int(frontres_segment_actor_warmup_iterations))
        self.frontres_segment_actor_joint_lr = self._require_positive_finite_lr(
            frontres_segment_actor_joint_lr,
            name="frontres_segment_actor_joint_lr",
        )
        if self.frontres_formal_transaction_enabled and not self.frontres_relational_actor_only and (
            self.actor_learning_rate != 3.0e-7
            or self.frontres_segment_actor_joint_lr != 1.0e-6
            or self.critic_learning_rate != 1.0e-5
        ):
            raise ValueError("FRS-TRAIN-v024 requires Actor LR curriculum 3e-7 -> 1e-6")
        self.frontres_segment_k_curriculum = tuple(tuple(row) for row in frontres_segment_k_curriculum)
        self.frontres_segment_k_curriculum_fingerprint = ""
        if self.frontres_segment_k_curriculum:
            (
                normalize_frontres_k_stage_schedule,
                frontres_k_stage_schedule_tuple,
                frontres_k_stage_schedule_fingerprint,
                require_frontres_v013_campaign_schedule,
            ) = _frontres_v013_schedule_owners()
            normalized_k_schedule = normalize_frontres_k_stage_schedule(
                self.frontres_segment_k_curriculum,
                max_horizon_k=int(frontres_segment_max_horizon_k),
            )
            if self.frontres_formal_transaction_enabled:
                normalized_k_schedule = require_frontres_v013_campaign_schedule(normalized_k_schedule)
            self.frontres_segment_k_curriculum = frontres_k_stage_schedule_tuple(normalized_k_schedule)
            self.frontres_segment_k_curriculum_fingerprint = frontres_k_stage_schedule_fingerprint(
                normalized_k_schedule
            )
        self.frontres_formal_runtime_audit = bool(frontres_formal_runtime_audit)
        self.frontres_segment_live_fail_on_invalid_update = bool(frontres_segment_live_fail_on_invalid_update)
        self.frontres_segment_live_min_valid_count = max(0, int(frontres_segment_live_min_valid_count))
        self.frontres_segment_live_fail_on_nonfinite = bool(frontres_segment_live_fail_on_nonfinite)
        self.frontres_hsl_init_enabled = bool(frontres_hsl_init_enabled)
        self.frontres_hsl_rollout_label_enabled = bool(frontres_hsl_rollout_label_enabled)
        self.frontres_segment_k = max(1, int(frontres_segment_k))
        self.frontres_future_offsets = tuple(int(value) for value in frontres_future_offsets)
        self.frontres_future_intent_layout_version = str(frontres_future_intent_layout_version)
        validate_frontres_v015_stage3_supervision_config(
            future_offsets=self.frontres_future_offsets,
            lambda_supervised=self.lambda_supervised,
            lambda_supervised_min=self.lambda_supervised_min,
        )
        self.frontres_segment_max_horizon_k = max(
            self.frontres_segment_k,
            int(frontres_segment_max_horizon_k),
        )
        self.frontres_segment_advantage_normalization = str(frontres_segment_advantage_normalization).lower()
        self.frontres_gain_beta = float(frontres_gain_beta)
        if not math.isfinite(self.frontres_gain_beta) or self.frontres_gain_beta < 0.0:
            raise ValueError("FRS-GAIN-v008 beta must be finite and non-negative")
        if self.frontres_segment_advantage_normalization not in (
            "none", "scale_only", "standard", "grouped_scale_only", "pairwise_edge"
        ):
            raise ValueError(
                "frontres_segment_advantage_normalization must be one of "
                "'none', 'scale_only', 'standard', or 'grouped_scale_only'"
            )
        if self.frontres_formal_transaction_enabled:
            if self.frontres_gain_beta != 0.02:
                raise ValueError("FRS-GAIN-v008 formal route requires the frozen beta_init=0.02")
            if not self.frontres_segment_k_curriculum:
                raise ValueError("FRS-TRAIN-v021 formal transaction requires an explicit K x M x DR curriculum")
            expected_normalization = "pairwise_edge" if self.frontres_relational_actor_only else "grouped_scale_only"
            if self.frontres_segment_advantage_normalization != expected_normalization:
                raise ValueError("v015 formal transaction requires grouped_scale_only normalization")
            if (
                self.lambda_supervised != 0.0
                or self.lambda_supervised_min != 0.0
                or self.frontres_hsl_init_enabled
                or self.frontres_hsl_rollout_label_enabled
            ):
                raise ValueError("v015 formal transaction rejects HSL and Stage-3 supervised targets")
        if self.frontres_local_sentinel_only:
            if not self.frontres_formal_transaction_enabled:
                raise ValueError("v017 local sentinel requires frontres_formal_transaction_enabled=True")
            if any(
                (
                    self.frontres_segment_live_sentinel_only,
                    self.frontres_segment_live_probe_only,
                    self.frontres_segment_live_storage_write_only,
                    self.frontres_segment_live_single_update_only,
                    self.frontres_segment_live_update_loop_only,
                    self.frontres_segment_live_train_enabled,
                )
            ):
                raise ValueError("v015 local sentinel rejects legacy live mode mixing")
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
            # B1: 区分只读 evaluator 和训练 route, 产出唯一可执行 mode.
            if self.frontres_policy_quality_eval_only:
                evaluation_conflicts = tuple(
                    name
                    for name, enabled in (
                        ("segment_replay", self.frontres_segment_replay_enabled),
                        ("live_runner", self.frontres_segment_live_runner_enabled),
                        ("live_sentinel", self.frontres_segment_live_sentinel_only),
                        ("local_sentinel", self.frontres_local_sentinel_only),
                        ("live_probe", self.frontres_segment_live_probe_only),
                        ("storage_write", self.frontres_segment_live_storage_write_only),
                        ("single_update", self.frontres_segment_live_single_update_only),
                        ("update_loop", self.frontres_segment_live_update_loop_only),
                        ("live_train", self.frontres_segment_live_train_enabled),
                    )
                    if enabled
                )
                if evaluation_conflicts:
                    raise ValueError(
                        "policy-quality evaluation cannot enable Segment Replay/live training modes: "
                        f"{evaluation_conflicts}"
                    )
                if not self.frontres_formal_transaction_enabled:
                    raise ValueError("policy-quality evaluation requires the formal transaction identity")
                print(
                    "[FrontRESUnified] Read-only policy-quality evaluator initialized; "
                    "Segment Replay and optimizer dispatch remain disabled.",
                    flush=True,
                )
            elif not self.frontres_segment_replay_enabled:
                raise ValueError("segment_replay_hrl requires frontres_segment_replay_enabled=True")
            elif not self.frontres_segment_live_runner_enabled:
                raise NotImplementedError(
                    "segment_replay_hrl is recognized, but live runner integration is disabled. "
                    "Use Step 4-7 toy contract tests until the live Stage 3 connector is integrated."
                )
            elif self.frontres_local_sentinel_only:
                print(
                    "[FrontRESUnified] v015 local identity sentinel initialized; "
                    "the dedicated formal route is opt-in and no legacy live mode is active.",
                    flush=True,
                )
            elif self.frontres_segment_live_sentinel_only:
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
        self.frontres_reward_compute_live_debug = bool(frontres_reward_compute_live_debug)
        self.frontres_cuda_memory_debug = bool(frontres_cuda_memory_debug)
        self.diagnose_gradient_conflict = bool(diagnose_gradient_conflict)
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
    def _require_positive_finite_lr(value, *, name: str) -> float:
        if value is None:
            raise ValueError(f"FRS-TRAIN-v021 requires explicit {name}")
        lr = float(value)
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"FRS-TRAIN-v021 requires positive finite {name}")
        return lr

    @staticmethod
    def _collect_trainable_param_groups(
        policy,
        *,
        actor_learning_rate: float,
        critic_learning_rate: float | None,
        actor_only: bool = False,
    ):
        if not isinstance(policy, (ResidualActorCritic, FrontRESActorCritic)):
            raise TypeError("FRS-TRAIN-v021 Stage-3 optimizer requires a FrontRES Actor/Critic policy")
        actor_params = list(policy.residual_actor.parameters())
        critic_params = list(policy.critic.parameters())
        if not actor_params or not critic_params:
            raise ValueError("FrontRES formal optimizer requires non-empty Actor and Critic modules")
        if hasattr(policy, "std") and getattr(policy.std, "requires_grad", False):
            raise ValueError("FRS-TRAIN-v021 task-space policy std must remain fixed")
        if hasattr(policy, "log_std") and getattr(policy.log_std, "requires_grad", False):
            raise ValueError("FRS-TRAIN-v021 task-space policy std must remain fixed")
        actor_ids = {id(parameter) for parameter in actor_params}
        critic_ids = {id(parameter) for parameter in critic_params}
        if actor_ids.intersection(critic_ids):
            raise ValueError("FRS-TRAIN-v021 Actor and Critic optimizer groups must be disjoint")
        if actor_only:
            for parameter in critic_params:
                parameter.requires_grad_(False)
            trainable_ids = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
            if trainable_ids != actor_ids:
                raise ValueError("FRS-TRAIN-v025 Actor-only optimizer must exhaust Actor parameters only")
            print(
                f"[FrontRESUnified] Relational optimizer updates Actor only (lr={actor_learning_rate:.6g}); "
                "Critic is frozen compatibility state"
            )
            return [{"params": actor_params, "lr": actor_learning_rate, "frontres_role": "actor"}]
        trainable_ids = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
        if trainable_ids != actor_ids.union(critic_ids):
            raise ValueError(
                "FRS-TRAIN-v021 Actor and Critic optimizer groups must exhaust all trainable policy parameters"
            )
        print(
            "[FrontRESUnified] Optimizer groups "
            f"actor(lr={actor_learning_rate:.6g}) + critic(lr={critic_learning_rate:.6g}); fixed policy std"
        )
        return [
            {"params": actor_params, "lr": actor_learning_rate, "frontres_role": "actor"},
            {"params": critic_params, "lr": critic_learning_rate, "frontres_role": "critic"},
        ]

    @staticmethod
    def _collect_trainable_params(policy):
        if isinstance(policy, (ResidualActorCritic, FrontRESActorCritic)):
            params = list(policy.residual_actor.parameters())
            params.extend(policy.critic.parameters())
            has_trainable_std = False
            if hasattr(policy, "std") and getattr(policy.std, "requires_grad", False):
                params.append(policy.std)
                has_trainable_std = True
            elif hasattr(policy, "log_std") and getattr(policy.log_std, "requires_grad", False):
                params.append(policy.log_std)
                has_trainable_std = True
            suffix = " + policy std" if has_trainable_std else " (fixed policy std)"
            print(f"[FrontRESUnified] Optimizer updates residual_actor + critic{suffix}")
            return params
        print("[FrontRESUnified] Optimizer updates full policy")
        return policy.parameters()

    def _print_init_summary(self):
        print("=" * 80)
        print("  FrontRESUnified ▸ PPO + Supervised ΔSE3")
        print(f"  Objective={self.frontres_training_objective}")
        if self.frontres_training_objective == "supervised_restore":
            print("  L = L_supervised_restore  (full-6D HSL proposal update)")
        elif self.frontres_training_objective in ("segment_replay_hrl", "segment_replay_relational"):
            if self.frontres_policy_quality_eval_only:
                print("  Mode = read-only policy-quality evaluation  (all updates disabled)")
            elif self.frontres_relational_actor_only:
                print("  L = relational Segment Replay  (Actor-only preference edges; Critic inert)")
            else:
                print("  L = Segment Replay HRL  (dedicated runner loop; legacy update disabled)")
        else:
            raise ValueError(
                f"FrontRESUnified only supports supervised_restore, segment_replay_hrl, or segment_replay_relational, "
                f"got {self.frontres_training_objective!r}"
            )
        print("=" * 80)
        if self.critic_learning_rate is None:
            print(f"  LR={self.learning_rate}  clip={self.clip_param}  ent_coef={self.entropy_coef}")
        else:
            print(
                f"  Actor LR={self.actor_learning_rate}  Critic LR={self.critic_learning_rate} "
                f"schedule={self.schedule}  clip={self.clip_param}  ent_coef={self.entropy_coef}"
            )
        print(f"  epochs={self.num_learning_epochs}  mini_batches={self.num_mini_batches}")
        print(f"  Supervised  λ={self.lambda_supervised:.3f} → {self.lambda_supervised_min}"
              f"  decay={self.lambda_supervised_decay_rate}"
              f"  trigger_cos={self.supervised_trigger_cosine_sim}"
              f"  rpy_w={self.supervised_rpy_loss_weight}"
              f"  dir_w={self.supervised_direction_loss_weight}"
              f"  mag_w={self.supervised_magnitude_loss_weight}"
              f"  over_w={self.supervised_over_loss_weight}"
              f"  smooth_w={self.supervised_smooth_loss_weight}"
              f"  valid_w={self.supervised_valid_loss_weight}")
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
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
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
        return loss_dict

    def _update_supervised_learning_rate(self) -> None:
        if self.frontres_training_objective != "supervised_restore":
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
        """Run the Stage 2 supervised full-6D HSL update."""
        self._update_supervised_learning_rate()
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_supervised_loss = 0.0
        mean_supervised_cos_sim = 0.0
        mean_supervised_metrics: dict[str, float] = {}
        num_updates = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for batch in generator:
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
                _rnd_state_batch,
                _teacher_obs_batch,
                _teacher_mu_batch,
                _teacher_sigma_batch,
                ref_vel_estimator_obs_batch,
                _motion_groups_batch,
                frontres_mask_batch,
                supervised_target_batch,
                supervised_weight_batch,
                supervised_harm_weight_batch,
                *_legacy_batch_fields,
            ) = batch

            batch_indices = _legacy_batch_fields[-1] if _legacy_batch_fields else None
            original_batch_size = obs_batch.shape[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (
                        advantages_batch - advantages_batch.mean()
                    ) / (advantages_batch.std(unbiased=False) + 1e-8)

            if self.use_estimate_ref_vel and self.ref_vel_estimator is not None:
                with torch.no_grad():
                    estimator_input = (
                        ref_vel_estimator_obs_batch
                        if ref_vel_estimator_obs_batch is not None
                        else obs_batch
                    )
                    estimated_ref_vel_batch = self.ref_vel_estimator(estimator_input)
                    obs_batch_augmented = torch.cat(
                        [obs_batch, estimated_ref_vel_batch], dim=-1
                    )
            else:
                obs_batch_augmented = obs_batch

            self.optimizer.zero_grad(set_to_none=True)
            self.policy.update_distribution(obs_batch_augmented)
            value_batch = self.policy.evaluate(
                critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            supervised_loss, sup_cos_sim, sup_metrics = self._compute_supervised_loss(
                mu_batch,
                supervised_target_batch,
                original_batch_size,
                batch_indices=batch_indices,
                supervised_weight_batch=supervised_weight_batch,
                supervised_harm_weight_batch=supervised_harm_weight_batch,
            )

            surrogate_loss = torch.zeros((), device=self.device)
            value_loss = torch.zeros((), device=self.device)
            entropy = torch.zeros((), device=self.device)
            loss = self.lambda_supervised * supervised_loss
            if not torch.isfinite(loss):
                self._warn_skip("non-finite FrontRES loss", loss)
                continue

            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in self.policy.parameters()
                if p.requires_grad
            ):
                self._warn_skip("NaN gradient detected")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            num_updates += 1
            mean_value_loss += float(value_loss.detach().item())
            mean_surrogate_loss += float(surrogate_loss.detach().item())
            mean_entropy += float(entropy.detach().item())
            mean_supervised_loss += float(supervised_loss.detach().item())
            mean_supervised_cos_sim += float(sup_cos_sim)
            for key, value in sup_metrics.items():
                mean_supervised_metrics[key] = (
                    mean_supervised_metrics.get(key, 0.0) + float(value)
                )
            self._step_supervised_lambda(float(sup_cos_sim))

        if num_updates == 0:
            self.storage.clear()
            return {
                "value_function": 0.0,
                "surrogate": 0.0,
                "entropy": 0.0,
                "supervised_loss": 0.0,
                "supervised_cos_sim": 0.0,
                "lambda_supervised": self.lambda_supervised,
            }

        self.storage.clear()
        for attr in ("_cached_observations", "_cached_full_policy_obs"):
            if hasattr(self.policy, attr):
                setattr(self.policy, attr, None)
        scale = 1.0 / float(num_updates)
        return {
            "value_function": mean_value_loss * scale,
            "surrogate": mean_surrogate_loss * scale,
            "entropy": mean_entropy * scale,
            "supervised_loss": mean_supervised_loss * scale,
            "supervised_cos_sim": mean_supervised_cos_sim * scale,
            "lambda_supervised": self.lambda_supervised,
            **{
                key: value * scale
                for key, value in mean_supervised_metrics.items()
            },
        }

    def _compute_supervised_loss(
        self,
        mu_batch,
        supervised_target_batch,
        original_batch_size,
        batch_indices=None,
        supervised_weight_batch=None,
        supervised_harm_weight_batch=None,
    ):
        supervised_loss = torch.zeros((), device=self.device)
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
            "supervised_l_harm": 0.0,
            "frontres_write_ratio": 0.0,
            "frontres_proposal_ratio": 0.0,
            "frontres_axis_leakage": 0.0,
            "frontres_supervised_weight": 0.0,
        }
        if supervised_target_batch is None or self.lambda_supervised <= 0:
            return supervised_loss, sup_cos_sim, sup_metrics

        raw_pred = mu_batch[:original_batch_size]
        target = supervised_target_batch[:original_batch_size].to(
            device=self.device, dtype=raw_pred.dtype
        )
        if raw_pred.shape[-1] < target.shape[-1]:
            return supervised_loss, sup_cos_sim, sup_metrics

        sample_weight = (
            supervised_weight_batch[:original_batch_size].view(-1).to(
                device=self.device, dtype=raw_pred.dtype
            )
            if supervised_weight_batch is not None
            else torch.ones(raw_pred.shape[0], device=self.device, dtype=raw_pred.dtype)
        )
        sample_weight = torch.nan_to_num(
            sample_weight, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=0.0)
        harm_weight = (
            supervised_harm_weight_batch[:original_batch_size].view(-1).to(
                device=self.device, dtype=raw_pred.dtype
            )
            if supervised_harm_weight_batch is not None
            else torch.zeros(raw_pred.shape[0], device=self.device, dtype=raw_pred.dtype)
        )
        harm_weight = torch.nan_to_num(
            harm_weight, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=0.0)

        def _wmean(values: torch.Tensor, weight: torch.Tensor | None = None):
            weights = sample_weight if weight is None else sample_weight * weight
            return (values * weights).sum() / weights.sum().clamp(min=1e-6)

        if self.policy.num_task_corrections > 0:
            if raw_pred.ndim != 2 or int(raw_pred.shape[-1]) != 6:
                raise ValueError(
                    "task-space FrontRES supervision requires exact direct [B,6] actor output, "
                    f"got {tuple(raw_pred.shape)}"
                )
            proposal = raw_pred
            target = target[:, :6]
        else:
            proposal = raw_pred[:, : target.shape[-1]]

        target = target.detach()
        valid = target.norm(dim=-1) > 1e-4
        pos_valid = target[:, :3].norm(dim=-1) > 1e-4
        rpy_valid = target[:, 3:6].norm(dim=-1) > 1e-4

        pos_weight = torch.ones_like(valid, dtype=raw_pred.dtype)
        rpy_weight = torch.ones_like(valid, dtype=raw_pred.dtype)
        if pos_valid.any():
            pos_weight[pos_valid] = float(self.supervised_valid_loss_weight)
        if rpy_valid.any():
            rpy_weight[rpy_valid] = float(self.supervised_valid_loss_weight)
        pos_weight /= pos_weight.mean().clamp(min=1e-6)
        rpy_weight /= rpy_weight.mean().clamp(min=1e-6)

        pos_err = nn.functional.huber_loss(
            proposal[:, :3], target[:, :3], reduction="none"
        ).mean(dim=-1)
        rpy_err = nn.functional.huber_loss(
            proposal[:, 3:6], target[:, 3:6], reduction="none"
        ).mean(dim=-1)
        pos_loss = _wmean(pos_err, pos_weight)
        rpy_loss = _wmean(rpy_err, rpy_weight)
        supervised_loss = pos_loss + self.supervised_rpy_loss_weight * rpy_loss

        magnitude_loss = torch.zeros((), device=self.device)
        over_loss = torch.zeros((), device=self.device)
        smooth_loss = torch.zeros((), device=self.device)
        harm_loss = torch.zeros((), device=self.device)
        if self.supervised_magnitude_loss_weight > 0 and valid.any():
            magnitude_loss = _wmean(
                nn.functional.huber_loss(
                    proposal.norm(dim=-1), target.norm(dim=-1), reduction="none"
                ),
                valid.to(raw_pred.dtype),
            )
            supervised_loss = supervised_loss + self.supervised_magnitude_loss_weight * magnitude_loss
        if self.supervised_over_loss_weight > 0 and valid.any():
            over_loss = _wmean(
                torch.relu(proposal.norm(dim=-1) - target.norm(dim=-1)).square(),
                valid.to(raw_pred.dtype),
            )
            supervised_loss = supervised_loss + self.supervised_over_loss_weight * over_loss
        if (
            self.supervised_smooth_loss_weight > 0
            and batch_indices is not None
            and getattr(self.storage, "num_envs", 0) > 0
        ):
            smooth_loss = self._compute_temporal_smooth_loss(
                proposal, target, batch_indices[:original_batch_size]
            )
            supervised_loss = supervised_loss + self.supervised_smooth_loss_weight * smooth_loss
        if self.supervised_harm_loss_weight > 0 and harm_weight.sum() > 0:
            harm_loss = _wmean(proposal.square().mean(dim=-1), harm_weight)
            supervised_loss = supervised_loss + self.supervised_harm_loss_weight * harm_loss

        if self.supervised_direction_loss_weight > 0:
            direction_loss = torch.zeros((), device=self.device)
            if pos_valid.any():
                direction_loss = direction_loss + _wmean(
                    1.0 - nn.functional.cosine_similarity(
                        proposal[:, :3], target[:, :3], dim=-1
                    ),
                    pos_valid.to(raw_pred.dtype),
                )
            if rpy_valid.any():
                direction_loss = direction_loss + _wmean(
                    1.0 - nn.functional.cosine_similarity(
                        proposal[:, 3:6], target[:, 3:6], dim=-1
                    ),
                    rpy_valid.to(raw_pred.dtype),
                )
            supervised_loss = supervised_loss + self.supervised_direction_loss_weight * direction_loss

        with torch.no_grad():
            error = proposal - target
            target_norm = target.norm(dim=-1).clamp(min=1e-6)
            sup_metrics["supervised_mae"] = error.abs().mean().item()
            sup_metrics["supervised_rmse"] = error.square().mean().sqrt().item()
            sup_metrics["supervised_rpy_mae"] = error[:, 3:6].abs().mean().item()
            sup_metrics["supervised_rpy_rmse"] = error[:, 3:6].square().mean().sqrt().item()
            sup_metrics["supervised_l_pos"] = pos_loss.item()
            sup_metrics["supervised_l_rot"] = rpy_loss.item()
            sup_metrics["supervised_l_mag"] = magnitude_loss.item()
            sup_metrics["supervised_l_over"] = over_loss.item()
            sup_metrics["supervised_l_smooth"] = smooth_loss.item()
            sup_metrics["supervised_l_harm"] = harm_loss.item()
            sup_metrics["supervised_valid_frac"] = valid.float().mean().item()
            sup_metrics["frontres_supervised_weight"] = sample_weight.mean().item()
            sup_metrics["frontres_write_ratio"] = (
                proposal.norm(dim=-1) / target_norm
            ).mean().item()
            sup_metrics["frontres_proposal_ratio"] = sup_metrics["frontres_write_ratio"]
            inactive = (target.abs() <= 1e-4).to(proposal.dtype)
            sup_metrics["frontres_axis_leakage"] = (
                (proposal.abs() * inactive).sum(dim=-1)
                / proposal.abs().sum(dim=-1).clamp(min=1e-6)
            ).mean().item()
            if valid.any():
                sup_cos_sim = nn.functional.cosine_similarity(
                    proposal[valid], target[valid], dim=-1
                ).mean().item()
                sup_metrics["supervised_restore_ratio"] = (
                    1.0 - error[valid].norm(dim=-1) / target_norm[valid]
                ).mean().item()
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
