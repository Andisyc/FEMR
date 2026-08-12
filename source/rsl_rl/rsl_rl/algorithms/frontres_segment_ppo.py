from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from collections.abc import Mapping
from typing import Any

import torch

from rsl_rl.frontres.frontres_formal_runtime_probe import emit_formal_runtime_probe
from rsl_rl.frontres.frontres_return_utility import (
    FRONTRES_RETURN_UTILITY_ID,
    FRONTRES_RETURN_UTILITY_SCALE,
    frontres_symmetric_log_utility,
)
from rsl_rl.frontres.frontres_value_normalization import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FRONTRES_VALUE_NORMALIZER_DECAY,
    FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
    FrontRESValueNormalizerState,
    FrontRESValueNormalizerUpdate,
    preview_frontres_v007_value_normalization,
)


@dataclass(frozen=True)
class FrontRESSegmentPPOConfig:
    clip_param: float = 0.2
    value_clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    use_clipped_value_loss: bool = True
    normalize_advantages: bool = False
    advantage_normalization: str = "none"
    advantage_scale_epsilon: float = 1.0e-8
    max_log_ratio: float = 20.0
    actor_loss_weight: float = 1.0
    critic_target_id: str = "per-attempt-return-v005"
    critic_value_normalization: str = "none"
    critic_value_normalizer_state: FrontRESValueNormalizerState | None = None
    critic_value_normalizer_decay: float = FRONTRES_VALUE_NORMALIZER_DECAY
    critic_value_normalizer_scale_floor: float = FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR
    critic_target_means_by_source: tuple[float, ...] | None = None


@dataclass(frozen=True)
class FrontRESSegmentPPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    valid_mask: torch.Tensor
    segment_ids: torch.Tensor | None = None
    old_means: torch.Tensor | None = None
    old_sigmas: torch.Tensor | None = None
    # Row-aligned critic input carried by the sealed v015 candidate path. The
    # grouped loss does not inspect it; the formal policy adapter owns it.
    privileged_observations: torch.Tensor | None = None
    # Sealed S1b provenance. It remains row-aligned through storage and is
    # consumed only by the candidate grouped-loss mode below.
    transaction_metadata: Any | None = None
    transaction_row_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class FrontRESSegmentPolicyEval:
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor | None = None
    mean: torch.Tensor | None = None
    sigma: torch.Tensor | None = None
    raw_actions: torch.Tensor | None = None
    log_jacobian_contrib: torch.Tensor | None = None


@dataclass(frozen=True)
class FrontRESSegmentPPOResult:
    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    valid_count: int
    valid_frac: float
    clip_frac: float
    approx_kl: float
    ratio_mean: float
    actor_loss_weight: float = 1.0
    ratio_max: float = 0.0
    old_log_prob_mean: float = 0.0
    new_log_prob_mean: float = 0.0
    raw_log_ratio_mean: float = 0.0
    raw_log_ratio_min: float = 0.0
    raw_log_ratio_max: float = 0.0
    pre_update_raw_log_ratio_mean: float = 0.0
    pre_update_raw_log_ratio_min: float = 0.0
    pre_update_raw_log_ratio_max: float = 0.0
    pre_update_clamped_ratio_mean: float = 0.0
    pre_update_clamped_ratio_max: float = 0.0
    advantage_mean: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    advantage_abs_mean: float = 0.0
    advantage_abs_max: float = 0.0
    advantage_abs_top1_frac: float = 0.0
    advantage_scale: float = 1.0
    advantage_sign_flip_count: int = 0
    prepared_advantages: tuple[float, ...] = ()
    grouped_reduction_active: bool = False
    grouped_motion_count: int = 0
    grouped_segment_count: int = 0
    grouped_attempt_count: int = 0
    grouped_valid_step_count: int = 0
    grouped_transaction_advantage_rms: float = 0.0
    grouped_segment_advantage_rms: tuple[float, ...] = ()
    grouped_segment_advantage_scales: tuple[float, ...] = ()
    grouped_motion_keys: tuple[str, ...] = ()
    grouped_segment_keys: tuple[str, ...] = ()
    grouped_attempt_keys: tuple[str, ...] = ()
    grouped_motion_mass_shares: tuple[float, ...] = ()
    grouped_segment_mass_shares: tuple[float, ...] = ()
    grouped_attempt_mass_shares: tuple[float, ...] = ()
    grouped_valid_step_row_indices: tuple[int, ...] = ()
    grouped_valid_step_mass_shares: tuple[float, ...] = ()
    grouped_missing_metadata_count: int = 0
    grouped_nonfinite_group_count: int = 0
    logprob_approx_kl: float = 0.0
    distribution_kl_mean: float = 0.0
    distribution_kl_available: bool = False
    distribution_mean_delta_l2_mean: float = 0.0
    distribution_mean_delta_max_abs: float = 0.0
    old_sigma_min: float = 0.0
    sigma_min: float = 0.0
    post_update_distribution_kl_mean: float = 0.0
    post_update_distribution_kl_available: bool = False
    post_update_logprob_approx_kl: float = 0.0
    post_update_ratio_mean: float = 0.0
    post_update_ratio_max: float = 0.0
    post_update_raw_log_ratio_mean: float = 0.0
    post_update_raw_log_ratio_min: float = 0.0
    post_update_raw_log_ratio_max: float = 0.0
    post_update_clamped_ratio_mean: float = 0.0
    post_update_clamped_ratio_max: float = 0.0
    post_update_clip_frac: float = 0.0
    raw_action_old_mean_l2_mean: float = 0.0
    raw_action_old_mean_abs_max: float = 0.0
    raw_action_old_mean_abs_dim_mean: tuple[float, ...] = ()
    raw_action_old_mean_abs_dim_max: tuple[float, ...] = ()
    old_sigma_dim_mean: tuple[float, ...] = ()
    sigma_dim_mean: tuple[float, ...] = ()
    distribution_mean_delta_dim_mean: tuple[float, ...] = ()
    distribution_mean_delta_abs_dim_max: tuple[float, ...] = ()
    log_ratio_contrib_dim_mean: tuple[float, ...] = ()
    log_ratio_contrib_abs_dim_max: tuple[float, ...] = ()
    log_jacobian_dim_mean: tuple[float, ...] = ()
    log_jacobian_abs_dim_max: tuple[float, ...] = ()
    post_update_raw_action_old_mean_l2_mean: float = 0.0
    post_update_raw_action_old_mean_abs_max: float = 0.0
    post_update_raw_action_old_mean_abs_dim_mean: tuple[float, ...] = ()
    post_update_raw_action_old_mean_abs_dim_max: tuple[float, ...] = ()
    post_update_old_sigma_dim_mean: tuple[float, ...] = ()
    post_update_sigma_dim_mean: tuple[float, ...] = ()
    post_update_distribution_mean_delta_dim_mean: tuple[float, ...] = ()
    post_update_distribution_mean_delta_abs_dim_max: tuple[float, ...] = ()
    post_update_log_ratio_contrib_dim_mean: tuple[float, ...] = ()
    post_update_log_ratio_contrib_abs_dim_max: tuple[float, ...] = ()
    post_update_log_jacobian_dim_mean: tuple[float, ...] = ()
    post_update_log_jacobian_abs_dim_max: tuple[float, ...] = ()
    actor_advantages: tuple[float, ...] = ()
    raw_returns: tuple[float, ...] = ()
    utility_returns: tuple[float, ...] = ()
    return_utility_id: str = "none"
    return_utility_scale: float = 1.0
    critic_value_targets: tuple[float, ...] = ()
    critic_segment_target_means: tuple[float, ...] = ()
    critic_segment_keys: tuple[int, ...] = ()
    critic_raw_value_loss: torch.Tensor | None = None
    critic_value_normalization_id: str = "none"
    critic_value_scale: float = 1.0
    critic_value_normalizer_previous_state: FrontRESValueNormalizerState | None = None
    critic_value_normalizer_candidate_state: FrontRESValueNormalizerState | None = None
    critic_value_normalizer_batch_mean: float = 0.0
    critic_value_normalizer_batch_second_moment: float = 0.0

    @property
    def should_step(self) -> bool:
        return self.valid_count > 0

    def diagnostics(self) -> dict[str, float]:
        return {
            "segment/ppo_total_loss": float(self.total_loss.detach().cpu().item()),
            "segment/ppo_actor_loss": float(self.actor_loss.detach().cpu().item()),
            "segment/ppo_value_loss": float(self.value_loss.detach().cpu().item()),
            "segment/ppo_entropy": float(self.entropy.detach().cpu().item()),
            "segment/ppo_valid_frac": self.valid_frac,
            "segment/ppo_clip_frac": self.clip_frac,
            "segment/ppo_approx_kl": self.approx_kl,
            "segment/ppo_ratio_mean": self.ratio_mean,
            "segment/ppo_actor_loss_weight": self.actor_loss_weight,
            "segment/ppo_ratio_max": self.ratio_max,
            "segment/ppo_old_log_prob_mean": self.old_log_prob_mean,
            "segment/ppo_new_log_prob_mean": self.new_log_prob_mean,
            "segment/ppo_raw_log_ratio_mean": self.raw_log_ratio_mean,
            "segment/ppo_raw_log_ratio_min": self.raw_log_ratio_min,
            "segment/ppo_raw_log_ratio_max": self.raw_log_ratio_max,
            "segment/ppo_pre_update_raw_log_ratio_mean": self.pre_update_raw_log_ratio_mean,
            "segment/ppo_pre_update_raw_log_ratio_min": self.pre_update_raw_log_ratio_min,
            "segment/ppo_pre_update_raw_log_ratio_max": self.pre_update_raw_log_ratio_max,
            "segment/ppo_pre_update_clamped_ratio_mean": self.pre_update_clamped_ratio_mean,
            "segment/ppo_pre_update_clamped_ratio_max": self.pre_update_clamped_ratio_max,
            "segment/ppo_advantage_mean": self.advantage_mean,
            "segment/ppo_advantage_min": self.advantage_min,
            "segment/ppo_advantage_max": self.advantage_max,
            "segment/ppo_advantage_abs_mean": self.advantage_abs_mean,
            "segment/ppo_advantage_abs_max": self.advantage_abs_max,
            "segment/ppo_advantage_abs_top1_frac": self.advantage_abs_top1_frac,
            "segment/ppo_advantage_scale": self.advantage_scale,
            "segment/ppo_advantage_sign_flip_count": float(self.advantage_sign_flip_count),
            "segment/ppo_critic_value_scale": self.critic_value_scale,
            "segment/ppo_critic_value_normalizer_batch_mean": self.critic_value_normalizer_batch_mean,
            "segment/ppo_critic_value_normalizer_batch_second_moment": (
                self.critic_value_normalizer_batch_second_moment
            ),
            "segment/ppo_grouped_reduction_active": float(self.grouped_reduction_active),
            "segment/ppo_grouped_motion_count": float(self.grouped_motion_count),
            "segment/ppo_grouped_segment_count": float(self.grouped_segment_count),
            "segment/ppo_grouped_attempt_count": float(self.grouped_attempt_count),
            "segment/ppo_grouped_valid_step_count": float(self.grouped_valid_step_count),
            "segment/ppo_grouped_transaction_advantage_rms": self.grouped_transaction_advantage_rms,
            "segment/ppo_grouped_missing_metadata_count": float(self.grouped_missing_metadata_count),
            "segment/ppo_grouped_nonfinite_group_count": float(self.grouped_nonfinite_group_count),
            "segment/ppo_logprob_approx_kl": self.logprob_approx_kl,
            "segment/ppo_distribution_kl_mean": self.distribution_kl_mean,
            "segment/ppo_distribution_kl_available": float(self.distribution_kl_available),
            "segment/ppo_distribution_mean_delta_l2_mean": self.distribution_mean_delta_l2_mean,
            "segment/ppo_distribution_mean_delta_max_abs": self.distribution_mean_delta_max_abs,
            "segment/ppo_old_sigma_min": self.old_sigma_min,
            "segment/ppo_sigma_min": self.sigma_min,
            "segment/ppo_post_update_distribution_kl_mean": self.post_update_distribution_kl_mean,
            "segment/ppo_post_update_distribution_kl_available": float(
                self.post_update_distribution_kl_available
            ),
            "segment/ppo_post_update_logprob_approx_kl": self.post_update_logprob_approx_kl,
            "segment/ppo_post_update_ratio_mean": self.post_update_ratio_mean,
            "segment/ppo_post_update_ratio_max": self.post_update_ratio_max,
            "segment/ppo_post_update_raw_log_ratio_mean": self.post_update_raw_log_ratio_mean,
            "segment/ppo_post_update_raw_log_ratio_min": self.post_update_raw_log_ratio_min,
            "segment/ppo_post_update_raw_log_ratio_max": self.post_update_raw_log_ratio_max,
            "segment/ppo_post_update_clamped_ratio_mean": self.post_update_clamped_ratio_mean,
            "segment/ppo_post_update_clamped_ratio_max": self.post_update_clamped_ratio_max,
            "segment/ppo_post_update_clip_frac": self.post_update_clip_frac,
        }


@dataclass(frozen=True)
class FrontRESScalarOptimizerCommitResult:
    """Actual scalar PPO commit facts without a second projection authority."""

    optimizer_candidate_actor_delta_l2: float
    committed_actor_delta_l2: float
    actor_optimizer_state_preserved: bool


@dataclass(frozen=True)
class FrontRESScalarGradientInstallResult:
    """Disjoint parameter identities and finite role-specific clipping facts."""

    actor_parameters: tuple[torch.Tensor, ...]
    critic_parameters: tuple[torch.Tensor, ...]
    actor_pre_clip_norm: float
    actor_post_clip_norm: float
    actor_clip_coefficient: float
    critic_pre_clip_norm: float
    critic_post_clip_norm: float
    critic_clip_coefficient: float
    actor_nonzero_parameter_count: int
    critic_nonzero_parameter_count: int


def _frontres_scalar_parameter_partition(
    policy: Any,
    optimizer_parameters: tuple[torch.Tensor, ...],
    *,
    contract_id: str,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    critic = getattr(policy, "critic", None)
    if not isinstance(critic, torch.nn.Module):
        raise RuntimeError(f"{contract_id} requires one explicit scalar Critic module")
    critic_ids = {id(parameter) for parameter in critic.parameters()}
    actor_parameters = tuple(parameter for parameter in optimizer_parameters if id(parameter) not in critic_ids)
    critic_parameters = tuple(parameter for parameter in optimizer_parameters if id(parameter) in critic_ids)
    if not actor_parameters or not critic_parameters:
        raise RuntimeError(f"{contract_id} requires disjoint actor/std and Critic parameters")
    return actor_parameters, critic_parameters


def _frontres_loss_gradients(
    loss: torch.Tensor,
    parameters: tuple[torch.Tensor, ...],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, values, strict=True)
    )


def _frontres_gradient_norm(parameters: tuple[torch.Tensor, ...]) -> float:
    gradients = tuple(parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None)
    return math.sqrt(sum(float(gradient.square().sum().cpu().item()) for gradient in gradients))


def _frontres_nonzero_gradient_count(parameters: tuple[torch.Tensor, ...]) -> int:
    return sum(
        int(bool((parameter.grad.detach() != 0.0).any().cpu().item()))
        for parameter in parameters
        if parameter.grad is not None
    )


def install_frontres_v006_scalar_gradients(
    policy: Any,
    result: FrontRESSegmentPPOResult,
    cfg: FrontRESSegmentPPOConfig,
    optimizer_parameters: tuple[torch.Tensor, ...],
    *,
    max_grad_norm: float,
) -> FrontRESScalarGradientInstallResult:
    """Install and independently clip Actor/std and state-value Critic gradients."""

    # B1: 按 explicit Critic module identity 分组, 产出互斥且完整的 parameter families.
    actor_parameters, critic_parameters = _frontres_scalar_parameter_partition(
        policy,
        optimizer_parameters,
        contract_id="FRS-PPO-v011",
    )
    max_norm = float(max_grad_norm)
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("FRS-PPO-v009 max_grad_norm must be positive and finite")

    # B2: 分别安装 Actor 与 Critic loss gradients, critic-only 保持 Actor grad 为空.
    actor_frozen = float(cfg.actor_loss_weight) == 0.0
    if actor_frozen:
        for parameter in actor_parameters:
            parameter.grad = None
    else:
        actor_loss = float(cfg.actor_loss_weight) * (
            result.actor_loss - float(cfg.entropy_coef) * result.entropy
        )
        actor_gradients = _frontres_loss_gradients(actor_loss, actor_parameters, retain_graph=True)
        for parameter, gradient in zip(actor_parameters, actor_gradients, strict=True):
            parameter.grad = gradient.detach().clone()
    critic_loss = float(cfg.value_loss_coef) * result.value_loss
    critic_gradients = _frontres_loss_gradients(critic_loss, critic_parameters, retain_graph=False)
    for parameter, gradient in zip(critic_parameters, critic_gradients, strict=True):
        parameter.grad = gradient.detach().clone()

    # B3: 两个 family 各自 clip 并发布 finite norms/coefficients, 不共享 clip coefficient.
    actor_pre = _frontres_gradient_norm(actor_parameters)
    critic_pre = _frontres_gradient_norm(critic_parameters)
    if not actor_frozen:
        torch.nn.utils.clip_grad_norm_(actor_parameters, max_norm)
    torch.nn.utils.clip_grad_norm_(critic_parameters, max_norm)
    actor_post = _frontres_gradient_norm(actor_parameters)
    critic_post = _frontres_gradient_norm(critic_parameters)
    actor_coefficient = 1.0 if actor_pre <= max_norm else max_norm / (actor_pre + 1.0e-6)
    critic_coefficient = 1.0 if critic_pre <= max_norm else max_norm / (critic_pre + 1.0e-6)
    facts = (actor_pre, actor_post, actor_coefficient, critic_pre, critic_post, critic_coefficient)
    if not all(math.isfinite(value) for value in facts):
        raise FloatingPointError(f"FRS-PPO-v009 produced non-finite gradient facts: {facts!r}")
    return FrontRESScalarGradientInstallResult(
        actor_parameters=actor_parameters,
        critic_parameters=critic_parameters,
        actor_pre_clip_norm=actor_pre,
        actor_post_clip_norm=actor_post,
        actor_clip_coefficient=actor_coefficient,
        critic_pre_clip_norm=critic_pre,
        critic_post_clip_norm=critic_post,
        critic_clip_coefficient=critic_coefficient,
        actor_nonzero_parameter_count=_frontres_nonzero_gradient_count(actor_parameters),
        critic_nonzero_parameter_count=_frontres_nonzero_gradient_count(critic_parameters),
    )


def install_frontres_v005_scalar_gradients(
    policy: Any,
    result: FrontRESSegmentPPOResult,
    cfg: FrontRESSegmentPPOConfig,
    optimizer_parameters: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Install disjoint actor and scalar-Critic gradients from one G_total loss."""

    # B1: 以 scalar Critic module identity 分离 Actor/std 与 Critic optimizer parameters.
    actor_parameters, critic_parameters = _frontres_scalar_parameter_partition(
        policy,
        optimizer_parameters,
        contract_id="FRS-PPO-v005",
    )

    # B2: Actor 只接收完整 scalar PPO loss; critic-only phase 保持 Actor/std grad 为空.
    actor_frozen = float(cfg.actor_loss_weight) == 0.0
    if actor_frozen:
        for parameter in actor_parameters:
            parameter.grad = None
    else:
        actor_loss = float(cfg.actor_loss_weight) * (
            result.actor_loss - float(cfg.entropy_coef) * result.entropy
        )
        actor_gradients = _frontres_loss_gradients(actor_loss, actor_parameters, retain_graph=True)
        for parameter, gradient in zip(actor_parameters, actor_gradients, strict=True):
            parameter.grad = gradient.detach().clone()
    # B3: Critic 只拟合 G_total value target, 不恢复独立 Physics gradient authority.
    critic_loss = float(cfg.value_loss_coef) * result.value_loss
    critic_gradients = _frontres_loss_gradients(critic_loss, critic_parameters, retain_graph=False)
    for parameter, gradient in zip(critic_parameters, critic_gradients, strict=True):
        parameter.grad = gradient.detach().clone()
    return actor_parameters, critic_parameters


def step_frontres_v005_scalar_optimizer(
    optimizer: Any,
    actor_parameters: tuple[torch.Tensor, ...],
    parameter_snapshots: dict[int, torch.Tensor],
    *,
    actor_loss_weight: float,
) -> FrontRESScalarOptimizerCommitResult:
    """Run exactly one step and preserve Actor/std state during critic-only recalibration."""

    # B1: 保存 Actor/std parameter 与 optimizer state, 建立 critic-only rollback boundary.
    if not actor_parameters or any(id(parameter) not in parameter_snapshots for parameter in actor_parameters):
        raise RuntimeError("FRS-PPO-v005 requires pre-step Actor/std snapshots")
    state = getattr(optimizer, "state", None)
    step = getattr(optimizer, "step", None)
    if not isinstance(state, Mapping) or not callable(step):
        raise TypeError("FRS-PPO-v005 optimizer must expose state and step()")
    frozen = float(actor_loss_weight) == 0.0
    state_snapshots = {
        id(parameter): (parameter in state, copy.deepcopy(state.get(parameter, {})))
        for parameter in actor_parameters
    } if frozen else None
    # B2: 调用唯一 optimizer step, 记录实际 candidate Actor delta.
    step()
    candidate_delta = torch.cat(
        tuple(
            (parameter.detach() - parameter_snapshots[id(parameter)]).reshape(-1)
            for parameter in actor_parameters
        ),
        dim=0,
    )
    # B3: critic-only 时同时恢复 Actor 参数和 Adam state; joint 时提交 candidate delta.
    if frozen:
        assert state_snapshots is not None
        for parameter in actor_parameters:
            parameter.data.copy_(parameter_snapshots[id(parameter)])
            existed, before = state_snapshots[id(parameter)]
            if existed:
                state[parameter] = copy.deepcopy(before)
            else:
                state.pop(parameter, None)
        committed_delta = torch.zeros_like(candidate_delta)
    else:
        committed_delta = candidate_delta
    return FrontRESScalarOptimizerCommitResult(
        optimizer_candidate_actor_delta_l2=float(candidate_delta.norm().detach().cpu().item()),
        committed_actor_delta_l2=float(committed_delta.norm().detach().cpu().item()),
        actor_optimizer_state_preserved=frozen,
    )


@dataclass(frozen=True)
class _FrontRESSegmentPPOTransactionRows:
    transaction_id: str
    policy_snapshot_id: str
    motion_ids: tuple[str, ...]
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor
    trial_role: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    scenario_ids: tuple[str, ...]


@dataclass(frozen=True)
class _FrontRESSegmentPPOGroupedReduction:
    hierarchy: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]
    prepared_advantages: torch.Tensor
    transaction_advantage_rms: float
    segment_advantage_rms: tuple[float, ...]
    segment_advantage_scales: tuple[float, ...]
    motion_keys: tuple[str, ...]
    segment_keys: tuple[str, ...]
    attempt_keys: tuple[str, ...]
    motion_mass_shares: tuple[float, ...]
    segment_mass_shares: tuple[float, ...]
    attempt_mass_shares: tuple[float, ...]
    valid_step_row_indices: tuple[int, ...]
    valid_step_mass_shares: tuple[float, ...]


@dataclass(frozen=True)
class _FrontRESSegmentValueTargets:
    row_targets: torch.Tensor
    segment_keys: tuple[int, ...]
    segment_means: tuple[float, ...]


def _build_frontres_v006_segment_value_targets(
    batch: FrontRESSegmentPPOBatch,
    rows: _FrontRESSegmentPPOTransactionRows,
    valid: torch.Tensor,
    selected_target_returns: torch.Tensor | None = None,
    target_means_by_source: tuple[float, ...] | None = None,
) -> _FrontRESSegmentValueTargets:
    """Build row-aligned exact-M means after proving one shared state per Segment."""

    # B1: 校验完整 B8 x exact-M rows, 产出每个 Scenario 的共享 state/value group.
    privileged = batch.privileged_observations
    if (
        not isinstance(privileged, torch.Tensor)
        or privileged.ndim != 2
        or tuple(privileged.shape) != (int(batch.actions.shape[0]), 449)
        or privileged.requires_grad
        or not bool(torch.isfinite(privileged).all().item())
    ):
        raise ValueError("FRS-PPO-v011 requires detached finite Critic observations with shape [B,449]")
    selected_sources = rows.source_index[valid]
    segment_keys_tensor = torch.unique(selected_sources, sorted=True)
    if int(segment_keys_tensor.numel()) != 8:
        raise ValueError("FRS-PPO-v011 requires exactly eight Scenario state groups")
    if target_means_by_source is not None and not torch.equal(
        segment_keys_tensor,
        torch.arange(8, device=segment_keys_tensor.device, dtype=segment_keys_tensor.dtype),
    ):
        raise ValueError("FRS-PPO-v011 robust targets require source identities 0..7")
    selected_returns = (
        batch.returns[valid].detach() if selected_target_returns is None else selected_target_returns.detach()
    )
    if tuple(selected_returns.shape) != tuple(batch.returns[valid].shape):
        raise ValueError("FRS-PPO-v009 Segment target returns must remain row-aligned")
    selected_old_values = batch.old_values[valid].detach()
    selected_privileged = privileged[valid].detach()
    selected_trials = rows.trial_index[valid]
    selected_segment_ids = rows.segment_ids[valid]
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1).detach().cpu().tolist()
    selected_scenarios = tuple(rows.scenario_ids[index] for index in valid_indices)
    selected_hashes = tuple(rows.noisy_segment_hashes[index] for index in valid_indices)

    row_targets = torch.empty_like(selected_returns)
    segment_means: list[float] = []
    external_targets: torch.Tensor | None = None
    if target_means_by_source is not None:
        if len(target_means_by_source) != 8:
            raise ValueError("FRS-PPO-v011 requires exactly eight Replay-owned Scenario targets")
        external_targets = selected_returns.new_tensor(target_means_by_source)
        if not bool(torch.isfinite(external_targets).all().item()):
            raise ValueError("FRS-PPO-v011 requires finite Replay-owned Scenario targets")
    expected_m: int | None = None
    for segment_key_tensor in segment_keys_tensor:
        segment_key = int(segment_key_tensor.item())
        local_rows = torch.nonzero(selected_sources == segment_key, as_tuple=False).reshape(-1)
        count = int(local_rows.numel())
        if expected_m is None:
            expected_m = count
        if count < 2 or count != expected_m:
            raise ValueError("FRS-PPO-v011 requires the same exact-M count for all eight Scenarios")
        expected_trials = torch.arange(count, device=selected_trials.device, dtype=torch.long)
        if not torch.equal(torch.sort(selected_trials[local_rows]).values, expected_trials):
            raise ValueError("FRS-PPO-v009 requires unique trial_index=0..M-1 within each Segment")
        if int(torch.unique(selected_segment_ids[local_rows]).numel()) != 1:
            raise ValueError("FRS-PPO-v009 source rows cannot mix Segment identity")
        scenario_values = {selected_scenarios[int(index)] for index in local_rows.detach().cpu().tolist()}
        hash_values = {selected_hashes[int(index)] for index in local_rows.detach().cpu().tolist()}
        if len(scenario_values) != 1 or len(hash_values) != 1:
            raise ValueError("FRS-PPO-v009 Segment rows cannot mix scenario or Noisy hash identity")
        states = selected_privileged[local_rows]
        if not torch.equal(states, states[:1].expand_as(states)):
            raise ValueError("FRS-PPO-v009 requires identical Critic state rows within each Segment")
        old_values = selected_old_values[local_rows]
        if not torch.equal(old_values, old_values[:1].expand_as(old_values)):
            raise ValueError("FRS-PPO-v009 requires one shared old value within each Segment")

        # B2: 仅对该 Segment 的 exact-M realized returns 求均值, 产出一个 state-value target.
        target = (
            selected_returns[local_rows].mean()
            if external_targets is None
            else external_targets[segment_key]
        )
        if not bool(torch.isfinite(target).item()):
            raise FloatingPointError("FRS-PPO-v009 produced a non-finite Segment value target")
        row_targets[local_rows] = target
        segment_means.append(float(target.detach().cpu().item()))

    # B3: 将八个 mean 对齐回 policy rows, 不改逐 attempt Actor advantage 或 row ordering.
    return _FrontRESSegmentValueTargets(
        row_targets=row_targets,
        segment_keys=tuple(int(value) for value in segment_keys_tensor.detach().cpu().tolist()),
        segment_means=tuple(segment_means),
    )


def compute_frontres_segment_ppo_loss(
    policy: Any,
    batch: FrontRESSegmentPPOBatch,
    cfg: FrontRESSegmentPPOConfig | None = None,
) -> FrontRESSegmentPPOResult:
    """计算一个 direct Delta SE(3) Segment PPO loss.

    函数名说明:
        `compute_frontres_segment_ppo_loss` 是 algorithm loss owner, 在已采样 batch
        上计算 clipped surrogate, value loss 和 diagnostics; 它不执行 optimizer
        step, rollback 或 adaptive LR.

    主链路:
        上游: Segment storage 提供同源 action, old distribution, return 和 advantage.
        下游: runner 对 `total_loss` backward/step, 再用同 batch 计算 post-update KL
        和 trust-region decision.

    语义:
        PPO ratio 与 KL 必须比较同一个 raw full-6D distribution. Old tensors 全部
        detach, 梯度只进入当前 FrontRES actor/critic; frozen GMT 不在该图中.

    状态: `grouped_scale_only` 是离线 candidate loss mode. 它要求一个完整 sealed
        transaction carrier, 且不会自行激活 legacy runner, optimizer,
        checkpoint, 或 simulator route.
    """
    # B1: 验证同源 old action/distribution/advantage tuple 并选择 valid rows.
    cfg = FrontRESSegmentPPOConfig() if cfg is None else cfg
    _validate_batch(batch)
    policy_eval = _evaluate_policy(policy, batch)
    _validate_policy_eval(policy_eval, batch)

    # Old policy tensors 在下方 detach, 梯度只流经 current policy evaluation.
    finite = (
        torch.isfinite(policy_eval.log_prob)
        & torch.isfinite(policy_eval.value)
        & torch.isfinite(batch.old_log_probs)
        & torch.isfinite(batch.old_values)
        & torch.isfinite(batch.returns)
        & torch.isfinite(batch.advantages)
    )
    if policy_eval.entropy is not None:
        finite = finite & torch.isfinite(policy_eval.entropy)
    has_distribution_stats = (
        batch.old_means is not None
        and batch.old_sigmas is not None
        and policy_eval.mean is not None
        and policy_eval.sigma is not None
    )
    if has_distribution_stats:
        finite = (
            finite
            & torch.isfinite(batch.old_means).all(dim=-1)
            & torch.isfinite(batch.old_sigmas).all(dim=-1)
            & (batch.old_sigmas > 0.0).all(dim=-1)
            & torch.isfinite(policy_eval.mean).all(dim=-1)
            & torch.isfinite(policy_eval.sigma).all(dim=-1)
            & (policy_eval.sigma > 0.0).all(dim=-1)
        )
    normalization_mode = _advantage_normalization_mode(cfg)
    critic_target_id = str(cfg.critic_target_id)
    if critic_target_id not in {
        "per-attempt-return-v005",
        "segment-exact-m-mean-v1",
        "segment-exact-m-mean-symlog-v1",
        "scenario-compatible-robust-mean-symlog-v1",
    }:
        raise ValueError(f"unsupported FrontRES Critic target identity: {critic_target_id!r}")
    segment_mean_target_ids = {
        "segment-exact-m-mean-v1",
        "segment-exact-m-mean-symlog-v1",
        "scenario-compatible-robust-mean-symlog-v1",
    }
    robust_target_id = "scenario-compatible-robust-mean-symlog-v1"
    if critic_target_id == robust_target_id and cfg.critic_target_means_by_source is None:
        raise ValueError("FRS-PPO-v011 robust Scenario target requires Replay-owned target means")
    if critic_target_id != robust_target_id and cfg.critic_target_means_by_source is not None:
        raise ValueError("Replay-owned target means are valid only for FRS-PPO-v011 robust targets")
    if critic_target_id in segment_mean_target_ids and normalization_mode != "grouped_scale_only":
        raise ValueError("FRS-PPO-v009 Segment mean target requires grouped_scale_only transaction rows")
    transaction_rows: _FrontRESSegmentPPOTransactionRows | None = None
    if normalization_mode == "grouped_scale_only":
        transaction_rows = _transaction_metadata_rows(batch)
        policy_sampled = torch.tensor(
            [role == "policy" for role in transaction_rows.trial_role],
            device=batch.actions.device,
            dtype=torch.bool,
        )
        valid = batch.valid_mask.bool() & finite & policy_sampled
    else:
        valid = batch.valid_mask.bool() & finite
    valid_count = int(valid.sum().item())
    valid_frac = float(valid.float().mean().item()) if valid.numel() else 0.0
    if valid_count == 0:
        zero = (policy_eval.log_prob.sum() + policy_eval.value.sum()) * 0.0
        entropy_zero = zero.detach()
        return FrontRESSegmentPPOResult(
            total_loss=zero,
            actor_loss=zero,
            value_loss=zero,
            entropy=entropy_zero,
            valid_count=0,
            valid_frac=valid_frac,
            clip_frac=0.0,
            approx_kl=0.0,
            ratio_mean=0.0,
            actor_loss_weight=float(cfg.actor_loss_weight),
            grouped_reduction_active=normalization_mode == "grouped_scale_only",
        )

    log_prob = policy_eval.log_prob[valid]
    value = policy_eval.value[valid]
    old_log_prob = batch.old_log_probs[valid].detach()
    old_value = batch.old_values[valid].detach()
    returns = batch.returns[valid].detach()
    advantages = batch.advantages[valid].detach()
    raw_returns = returns.detach().clone()
    utility_returns = returns.detach().clone()
    return_utility_id = "none"
    if critic_target_id in {
        "segment-exact-m-mean-symlog-v1",
        "scenario-compatible-robust-mean-symlog-v1",
    }:
        utility_returns = frontres_symmetric_log_utility(raw_returns)
        expected_advantages = utility_returns - old_value
        if not torch.allclose(advantages, expected_advantages, rtol=0.0, atol=1.0e-6):
            raise ValueError("FRS-PPO-v009 requires row-aligned utility advantage U(G)-V_old")
        advantages = expected_advantages
        return_utility_id = FRONTRES_RETURN_UTILITY_ID
    actor_advantages = advantages.detach().clone()
    value_targets = returns
    segment_target_means: tuple[float, ...] = ()
    segment_target_keys: tuple[int, ...] = ()
    if critic_target_id in segment_mean_target_ids:
        assert transaction_rows is not None
        target_facts = _build_frontres_v006_segment_value_targets(
            batch,
            transaction_rows,
            valid,
            selected_target_returns=(utility_returns if return_utility_id != "none" else returns),
            target_means_by_source=cfg.critic_target_means_by_source,
        )
        value_targets = target_facts.row_targets
        segment_target_means = target_facts.segment_means
        segment_target_keys = target_facts.segment_keys
    value_normalization_id = str(cfg.critic_value_normalization).lower()
    if value_normalization_id not in {"none", FRONTRES_VALUE_NORMALIZATION_ID}:
        raise ValueError(f"unsupported FrontRES Critic value normalization: {value_normalization_id!r}")
    value_normalizer_update: FrontRESValueNormalizerUpdate | None = None
    value_scale = 1.0
    if value_normalization_id == FRONTRES_VALUE_NORMALIZATION_ID:
        if critic_target_id not in segment_mean_target_ids:
            raise ValueError("FRS-PPO-v009 value normalization requires an exact-M Segment mean target")
        if cfg.critic_value_normalizer_state is None:
            raise ValueError("FRS-PPO-v009 value normalization requires prior normalizer state")
        value_normalizer_update = preview_frontres_v007_value_normalization(
            value_targets.new_tensor(segment_target_means),
            cfg.critic_value_normalizer_state,
            decay=cfg.critic_value_normalizer_decay,
            scale_floor=cfg.critic_value_normalizer_scale_floor,
        )
        value_scale = value_normalizer_update.scale
    grouped_reduction: _FrontRESSegmentPPOGroupedReduction | None = None
    if normalization_mode == "grouped_scale_only":
        assert transaction_rows is not None
        grouped_reduction = _build_frontres_grouped_reduction(transaction_rows, valid, advantages)
        advantages = grouped_reduction.prepared_advantages
        advantage_scale = grouped_reduction.transaction_advantage_rms
        advantage_sign_flip_count = 0
    else:
        advantages, advantage_scale, advantage_sign_flip_count = _prepare_advantages(advantages, cfg)

    # B2: 构造 PPO ratio/surrogate/value path, 保留 raw log-ratio 供诊断.
    raw_log_ratio = log_prob - old_log_prob
    log_ratio = raw_log_ratio.clamp(-abs(float(cfg.max_log_ratio)), abs(float(cfg.max_log_ratio)))
    ratio = torch.exp(log_ratio)
    surrogate = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
    clipped_surrogate = clipped_ratio * advantages
    actor_row_loss = -torch.min(surrogate, clipped_surrogate)

    if cfg.use_clipped_value_loss:
        value_clipped = old_value + (value - old_value).clamp(-cfg.value_clip_param, cfg.value_clip_param)
        raw_value_row_loss = 0.5 * torch.max(
            (value - value_targets).square(),
            (value_clipped - value_targets).square(),
        )
    else:
        raw_value_row_loss = 0.5 * (value - value_targets).square()
    value_row_loss = raw_value_row_loss / (value_scale**2)

    if policy_eval.entropy is None:
        entropy_rows = log_prob.new_zeros(log_prob.shape)
    else:
        entropy_rows = policy_eval.entropy[valid]
    if grouped_reduction is None:
        actor_loss = actor_row_loss.mean()
        raw_value_loss = raw_value_row_loss.mean()
        value_loss = value_row_loss.mean()
        entropy = entropy_rows.mean()
    else:
        actor_loss = _reduce_frontres_grouped_rows(actor_row_loss, grouped_reduction.hierarchy)
        raw_value_loss = _reduce_frontres_grouped_rows(raw_value_row_loss, grouped_reduction.hierarchy)
        value_loss = _reduce_frontres_grouped_rows(value_row_loss, grouped_reduction.hierarchy)
        entropy = _reduce_frontres_grouped_rows(entropy_rows, grouped_reduction.hierarchy)
    actor_loss_weight = max(0.0, min(1.0, float(cfg.actor_loss_weight)))
    total_loss = (
        actor_loss_weight * actor_loss
        + cfg.value_loss_coef * value_loss
        - actor_loss_weight * cfg.entropy_coef * entropy
    )
    with torch.no_grad():
        # B3: 在同一次 forward 截获 pre-update diagnostics; runner 在 step 后用同 batch 复算.
        clip_frac = ((ratio - 1.0).abs() > cfg.clip_param).float().mean().item()
        logprob_approx_kl = (old_log_prob - log_prob).mean().item()
        distribution_kl_mean = (
            _distribution_kl_mean(policy_eval, batch, valid).item() if has_distribution_stats else 0.0
        )
        approx_kl = distribution_kl_mean if has_distribution_stats else logprob_approx_kl
        ratio_mean = ratio.mean().item()
        ratio_max = ratio.max().item()
        old_log_prob_mean = old_log_prob.mean().item()
        new_log_prob_mean = log_prob.mean().item()
        raw_log_ratio_mean = raw_log_ratio.mean().item()
        raw_log_ratio_min = raw_log_ratio.min().item()
        raw_log_ratio_max = raw_log_ratio.max().item()
        advantage_mean = advantages.mean().item()
        advantage_min = advantages.min().item()
        advantage_max = advantages.max().item()
        advantage_abs = advantages.abs()
        advantage_abs_mean = advantage_abs.mean().item()
        advantage_abs_max = advantage_abs.max().item()
        advantage_abs_sum = advantage_abs.sum().item()
        advantage_abs_top1_frac = advantage_abs_max / advantage_abs_sum if advantage_abs_sum > 0.0 else 0.0
        distribution_mean_delta_l2_mean = 0.0
        distribution_mean_delta_max_abs = 0.0
        old_sigma_min = 0.0
        sigma_min = 0.0
        raw_action_old_mean_l2_mean = 0.0
        raw_action_old_mean_abs_max = 0.0
        raw_action_old_mean_abs_dim_mean: tuple[float, ...] = ()
        raw_action_old_mean_abs_dim_max: tuple[float, ...] = ()
        old_sigma_dim_mean: tuple[float, ...] = ()
        sigma_dim_mean: tuple[float, ...] = ()
        distribution_mean_delta_dim_mean: tuple[float, ...] = ()
        distribution_mean_delta_abs_dim_max: tuple[float, ...] = ()
        log_ratio_contrib_dim_mean: tuple[float, ...] = ()
        log_ratio_contrib_abs_dim_max: tuple[float, ...] = ()
        log_jacobian_dim_mean: tuple[float, ...] = ()
        log_jacobian_abs_dim_max: tuple[float, ...] = ()
        if has_distribution_stats:
            old_mean = batch.old_means[valid].detach()
            old_sigma = batch.old_sigmas[valid].detach()
            mean = policy_eval.mean[valid].detach()
            sigma = policy_eval.sigma[valid].detach()
            mean_delta = mean - old_mean
            distribution_mean_delta_l2_mean = mean_delta.norm(dim=-1).mean().item()
            distribution_mean_delta_max_abs = mean_delta.abs().max().item()
            old_sigma_min = old_sigma.min().item()
            sigma_min = sigma.min().item()
            old_sigma_dim_mean = _dim_mean_tuple(old_sigma)
            sigma_dim_mean = _dim_mean_tuple(sigma)
            distribution_mean_delta_dim_mean = _dim_mean_tuple(mean_delta)
            distribution_mean_delta_abs_dim_max = _dim_max_tuple(mean_delta.abs())
            if policy_eval.raw_actions is not None:
                raw_actions = policy_eval.raw_actions[valid].detach()
                raw_action_old_mean = raw_actions - old_mean
                raw_action_old_mean_l2_mean = raw_action_old_mean.norm(dim=-1).mean().item()
                raw_action_old_mean_abs_max = raw_action_old_mean.abs().max().item()
                raw_action_old_mean_abs_dim_mean = _dim_mean_tuple(raw_action_old_mean.abs())
                raw_action_old_mean_abs_dim_max = _dim_max_tuple(raw_action_old_mean.abs())
                old_logprob_dim = torch.distributions.Normal(old_mean, old_sigma).log_prob(raw_actions)
                new_logprob_dim = torch.distributions.Normal(mean, sigma).log_prob(raw_actions)
                log_ratio_contrib = new_logprob_dim - old_logprob_dim
                log_ratio_contrib_dim_mean = _dim_mean_tuple(log_ratio_contrib)
                log_ratio_contrib_abs_dim_max = _dim_max_tuple(log_ratio_contrib.abs())
            if policy_eval.log_jacobian_contrib is not None:
                log_jacobian = policy_eval.log_jacobian_contrib[valid].detach()
                log_jacobian_dim_mean = _dim_mean_tuple(log_jacobian)
                log_jacobian_abs_dim_max = _dim_max_tuple(log_jacobian.abs())

    # AUDIT-PPO-01 截获 backward 实际消费的 pre-update loss/distribution state.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-PPO-01",
        actions=batch.actions,
        old_means=batch.old_means,
        old_sigmas=batch.old_sigmas,
        advantages=advantages,
        raw_returns=raw_returns,
        utility_returns=utility_returns,
        valid_mask=valid,
        total_loss=total_loss,
        ratio=ratio,
        distribution_kl_mean=distribution_kl_mean,
    )
    return FrontRESSegmentPPOResult(
        total_loss=total_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        entropy=entropy,
        valid_count=valid_count,
        valid_frac=valid_frac,
        clip_frac=float(clip_frac),
        approx_kl=float(approx_kl),
        ratio_mean=float(ratio_mean),
        actor_loss_weight=float(actor_loss_weight),
        ratio_max=float(ratio_max),
        old_log_prob_mean=float(old_log_prob_mean),
        new_log_prob_mean=float(new_log_prob_mean),
        raw_log_ratio_mean=float(raw_log_ratio_mean),
        raw_log_ratio_min=float(raw_log_ratio_min),
        raw_log_ratio_max=float(raw_log_ratio_max),
        pre_update_raw_log_ratio_mean=float(raw_log_ratio_mean),
        pre_update_raw_log_ratio_min=float(raw_log_ratio_min),
        pre_update_raw_log_ratio_max=float(raw_log_ratio_max),
        pre_update_clamped_ratio_mean=float(ratio_mean),
        pre_update_clamped_ratio_max=float(ratio_max),
        advantage_mean=float(advantage_mean),
        advantage_min=float(advantage_min),
        advantage_max=float(advantage_max),
        advantage_abs_mean=float(advantage_abs_mean),
        advantage_abs_max=float(advantage_abs_max),
        advantage_abs_top1_frac=float(advantage_abs_top1_frac),
        advantage_scale=float(advantage_scale),
        advantage_sign_flip_count=int(advantage_sign_flip_count),
        prepared_advantages=tuple(float(value) for value in advantages.detach().cpu().tolist()),
        actor_advantages=tuple(float(value) for value in actor_advantages.detach().cpu().tolist()),
        raw_returns=tuple(float(value) for value in raw_returns.detach().cpu().tolist()),
        utility_returns=tuple(float(value) for value in utility_returns.detach().cpu().tolist()),
        return_utility_id=return_utility_id,
        return_utility_scale=float(FRONTRES_RETURN_UTILITY_SCALE),
        critic_value_targets=tuple(float(value) for value in value_targets.detach().cpu().tolist()),
        critic_segment_target_means=segment_target_means,
        critic_segment_keys=segment_target_keys,
        critic_raw_value_loss=raw_value_loss,
        critic_value_normalization_id=value_normalization_id,
        critic_value_scale=float(value_scale),
        critic_value_normalizer_previous_state=(
            None if value_normalizer_update is None else value_normalizer_update.previous
        ),
        critic_value_normalizer_candidate_state=(
            None if value_normalizer_update is None else value_normalizer_update.candidate
        ),
        critic_value_normalizer_batch_mean=(
            0.0 if value_normalizer_update is None else value_normalizer_update.batch_mean
        ),
        critic_value_normalizer_batch_second_moment=(
            0.0 if value_normalizer_update is None else value_normalizer_update.batch_second_moment
        ),
        grouped_reduction_active=grouped_reduction is not None,
        grouped_motion_count=0 if grouped_reduction is None else len(grouped_reduction.motion_keys),
        grouped_segment_count=0 if grouped_reduction is None else len(grouped_reduction.segment_keys),
        grouped_attempt_count=0 if grouped_reduction is None else len(grouped_reduction.attempt_keys),
        grouped_valid_step_count=0 if grouped_reduction is None else len(grouped_reduction.valid_step_row_indices),
        grouped_transaction_advantage_rms=(
            0.0 if grouped_reduction is None else grouped_reduction.transaction_advantage_rms
        ),
        grouped_segment_advantage_rms=(
            () if grouped_reduction is None else grouped_reduction.segment_advantage_rms
        ),
        grouped_segment_advantage_scales=(
            () if grouped_reduction is None else grouped_reduction.segment_advantage_scales
        ),
        grouped_motion_keys=() if grouped_reduction is None else grouped_reduction.motion_keys,
        grouped_segment_keys=() if grouped_reduction is None else grouped_reduction.segment_keys,
        grouped_attempt_keys=() if grouped_reduction is None else grouped_reduction.attempt_keys,
        grouped_motion_mass_shares=(
            () if grouped_reduction is None else grouped_reduction.motion_mass_shares
        ),
        grouped_segment_mass_shares=(
            () if grouped_reduction is None else grouped_reduction.segment_mass_shares
        ),
        grouped_attempt_mass_shares=(
            () if grouped_reduction is None else grouped_reduction.attempt_mass_shares
        ),
        grouped_valid_step_row_indices=(
            () if grouped_reduction is None else grouped_reduction.valid_step_row_indices
        ),
        grouped_valid_step_mass_shares=(
            () if grouped_reduction is None else grouped_reduction.valid_step_mass_shares
        ),
        logprob_approx_kl=float(logprob_approx_kl),
        distribution_kl_mean=float(distribution_kl_mean),
        distribution_kl_available=bool(has_distribution_stats),
        distribution_mean_delta_l2_mean=float(distribution_mean_delta_l2_mean),
        distribution_mean_delta_max_abs=float(distribution_mean_delta_max_abs),
        old_sigma_min=float(old_sigma_min),
        sigma_min=float(sigma_min),
        raw_action_old_mean_l2_mean=float(raw_action_old_mean_l2_mean),
        raw_action_old_mean_abs_max=float(raw_action_old_mean_abs_max),
        raw_action_old_mean_abs_dim_mean=raw_action_old_mean_abs_dim_mean,
        raw_action_old_mean_abs_dim_max=raw_action_old_mean_abs_dim_max,
        old_sigma_dim_mean=old_sigma_dim_mean,
        sigma_dim_mean=sigma_dim_mean,
        distribution_mean_delta_dim_mean=distribution_mean_delta_dim_mean,
        distribution_mean_delta_abs_dim_max=distribution_mean_delta_abs_dim_max,
        log_ratio_contrib_dim_mean=log_ratio_contrib_dim_mean,
        log_ratio_contrib_abs_dim_max=log_ratio_contrib_abs_dim_max,
        log_jacobian_dim_mean=log_jacobian_dim_mean,
        log_jacobian_abs_dim_max=log_jacobian_abs_dim_max,
    )


def _dim_mean_tuple(tensor: torch.Tensor) -> tuple[float, ...]:
    if tensor.ndim != 2 or tensor.shape[-1] == 0:
        return ()
    return tuple(float(item) for item in tensor.mean(dim=0).detach().cpu().tolist())


def _dim_max_tuple(tensor: torch.Tensor) -> tuple[float, ...]:
    if tensor.ndim != 2 or tensor.shape[-1] == 0:
        return ()
    return tuple(float(item) for item in tensor.max(dim=0).values.detach().cpu().tolist())


def _advantage_normalization_mode(cfg: FrontRESSegmentPPOConfig) -> str:
    mode = str(cfg.advantage_normalization).lower()
    if cfg.normalize_advantages and mode == "none":
        mode = "standard"
    if mode not in ("none", "standard", "scale_only", "grouped_scale_only"):
        raise ValueError(
            "advantage_normalization must be one of none, standard, scale_only, or grouped_scale_only; "
            f"got {cfg.advantage_normalization!r}"
        )
    return mode


def _transaction_metadata_batch_size(metadata: Any) -> int:
    value = getattr(metadata, "batch_size", None)
    if value is None:
        segment_ids = getattr(metadata, "segment_ids", None)
        if not isinstance(segment_ids, torch.Tensor):
            raise ValueError("transaction metadata requires batch_size or segment_ids")
        value = int(segment_ids.numel())
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("transaction metadata batch_size must be an integer") from exc
    if size <= 0:
        raise ValueError("transaction metadata batch_size must be positive")
    return size


def _transaction_metadata_tensor(
    metadata: Any,
    *,
    name: str,
    metadata_size: int,
    row_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    value = getattr(metadata, name, None)
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or int(value.numel()) != metadata_size:
        raise ValueError(f"transaction metadata {name} must be a rank-1 vector of size {metadata_size}")
    if value.requires_grad:
        raise ValueError(f"transaction metadata {name} must be detached")
    return value.detach().to(device=device, dtype=torch.long)[row_indices.to(device=device)]


def _transaction_metadata_tuple(
    metadata: Any,
    *,
    name: str,
    metadata_size: int,
    row_indices: torch.Tensor,
) -> tuple[str, ...]:
    value = getattr(metadata, name, None)
    if not isinstance(value, tuple) or len(value) != metadata_size or not all(isinstance(item, str) for item in value):
        raise ValueError(f"transaction metadata {name} must be a string tuple of size {metadata_size}")
    selected = tuple(value[int(index)] for index in row_indices.detach().cpu().tolist())
    if any(not item for item in selected):
        raise ValueError(f"transaction metadata {name} must be non-empty")
    return selected


def _transaction_metadata_rows(batch: FrontRESSegmentPPOBatch) -> _FrontRESSegmentPPOTransactionRows:
    metadata = batch.transaction_metadata
    if metadata is None:
        raise ValueError("grouped_scale_only requires sealed transaction metadata")
    validate = getattr(metadata, "validate", None)
    if not callable(validate):
        raise ValueError("transaction metadata requires validate()")
    validate()
    batch_size = int(batch.actions.shape[0])
    metadata_size = _transaction_metadata_batch_size(metadata)
    raw_indices = batch.transaction_row_indices
    if raw_indices is None:
        if metadata_size != batch_size:
            raise ValueError(
                "transaction metadata requires explicit transaction_row_indices when storage rows are a subset"
            )
        row_indices = torch.arange(batch_size, dtype=torch.long)
    else:
        if not isinstance(raw_indices, torch.Tensor) or raw_indices.ndim != 1 or int(raw_indices.numel()) != batch_size:
            raise ValueError("transaction_row_indices must have shape [B]")
        if raw_indices.requires_grad:
            raise ValueError("transaction_row_indices must be detached")
        row_indices = raw_indices.detach().to(device="cpu", dtype=torch.long)
        if bool((row_indices < 0).any()) or bool((row_indices >= metadata_size).any()):
            raise ValueError("transaction_row_indices contain an out-of-range metadata row")
    expected_rows = torch.arange(metadata_size, dtype=torch.long)
    if batch_size != metadata_size or not torch.equal(torch.sort(row_indices).values, expected_rows):
        raise ValueError("grouped_scale_only requires one transaction-complete set of metadata rows")
    if batch.segment_ids is None:
        raise ValueError("grouped_scale_only requires row-aligned segment_ids")
    device = batch.actions.device
    metadata_segment_ids = _transaction_metadata_tensor(
        metadata,
        name="segment_ids",
        metadata_size=metadata_size,
        row_indices=row_indices,
        device=device,
    )
    batch_segment_ids = batch.segment_ids.detach().to(device=device, dtype=torch.long)
    if not torch.equal(metadata_segment_ids, batch_segment_ids):
        raise ValueError("transaction metadata segment_ids do not match the PPO batch rows")
    transaction_id = str(getattr(metadata, "transaction_id", ""))
    policy_snapshot_id = str(getattr(metadata, "policy_snapshot_id", ""))
    if not transaction_id or not policy_snapshot_id:
        raise ValueError("transaction metadata requires transaction_id and policy_snapshot_id")
    rows = _FrontRESSegmentPPOTransactionRows(
        transaction_id=transaction_id,
        policy_snapshot_id=policy_snapshot_id,
        motion_ids=_transaction_metadata_tuple(
            metadata,
            name="motion_ids",
            metadata_size=metadata_size,
            row_indices=row_indices,
        ),
        segment_ids=metadata_segment_ids,
        source_index=_transaction_metadata_tensor(
            metadata,
            name="source_index",
            metadata_size=metadata_size,
            row_indices=row_indices,
            device=device,
        ),
        trial_index=_transaction_metadata_tensor(
            metadata,
            name="trial_index",
            metadata_size=metadata_size,
            row_indices=row_indices,
            device=device,
        ),
        horizon_k=_transaction_metadata_tensor(
            metadata,
            name="horizon_k",
            metadata_size=metadata_size,
            row_indices=row_indices,
            device=device,
        ),
        trial_role=_transaction_metadata_tuple(
            metadata,
            name="trial_role",
            metadata_size=metadata_size,
            row_indices=row_indices,
        ),
        noisy_segment_hashes=_transaction_metadata_tuple(
            metadata,
            name="noisy_segment_hashes",
            metadata_size=metadata_size,
            row_indices=row_indices,
        ),
        scenario_ids=_transaction_metadata_tuple(
            metadata,
            name="scenario_ids",
            metadata_size=metadata_size,
            row_indices=row_indices,
        ),
    )
    if bool((rows.source_index < 0).any()) or bool((rows.trial_index < 0).any()) or bool((rows.horizon_k <= 0).any()):
        raise ValueError("transaction metadata has invalid source, trial, or horizon rows")
    return rows


def _build_frontres_grouped_reduction(
    rows: _FrontRESSegmentPPOTransactionRows,
    valid: torch.Tensor,
    advantages: torch.Tensor,
) -> _FrontRESSegmentPPOGroupedReduction:
    valid_row_indices = tuple(int(index) for index in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist())
    if not valid_row_indices:
        return _FrontRESSegmentPPOGroupedReduction(
            hierarchy=(),
            prepared_advantages=advantages,
            transaction_advantage_rms=0.0,
            segment_advantage_rms=(),
            segment_advantage_scales=(),
            motion_keys=(),
            segment_keys=(),
            attempt_keys=(),
            motion_mass_shares=(),
            segment_mass_shares=(),
            attempt_mass_shares=(),
            valid_step_row_indices=(),
            valid_step_mass_shares=(),
        )
    if int(advantages.numel()) != len(valid_row_indices):
        raise ValueError("grouped advantages must contain exactly the valid policy rows")
    local_index_by_row = {row: local for local, row in enumerate(valid_row_indices)}
    segment_rows: dict[tuple[str, int, int], list[int]] = {}
    attempt_rows: dict[tuple[str, int, int, int], list[int]] = {}
    motion_segments: dict[str, list[tuple[str, int, int]]] = {}
    for row in valid_row_indices:
        motion = rows.motion_ids[row]
        segment_key = (motion, int(rows.segment_ids[row].item()), int(rows.source_index[row].item()))
        attempt_key = (*segment_key, int(rows.trial_index[row].item()))
        if segment_key not in segment_rows:
            segment_rows[segment_key] = []
            motion_segments.setdefault(motion, []).append(segment_key)
        segment_rows[segment_key].append(local_index_by_row[row])
        attempt_rows.setdefault(attempt_key, []).append(local_index_by_row[row])
    transaction_rms_tensor = torch.sqrt(advantages.detach().square().mean())
    if not bool(torch.isfinite(transaction_rms_tensor).item()):
        raise ValueError("grouped_scale_only received a non-finite transaction advantage RMS")
    prepared = advantages.clone()
    segment_keys: list[str] = []
    segment_rms: list[float] = []
    segment_scales: list[float] = []
    for motion, segments in motion_segments.items():
        for segment_key in segments:
            local_rows = segment_rows[segment_key]
            index = torch.tensor(local_rows, device=advantages.device, dtype=torch.long)
            group_rms = torch.sqrt(advantages[index].detach().square().mean())
            denominator = torch.maximum(group_rms, transaction_rms_tensor).detach()
            if not bool(torch.isfinite(denominator).item()):
                raise ValueError("grouped_scale_only produced a non-finite Segment denominator")
            prepared[index] = torch.where(
                denominator > 0.0,
                advantages[index] / denominator,
                torch.zeros_like(advantages[index]),
            )
            segment_keys.append(f"{motion}|segment={segment_key[1]}|source={segment_key[2]}")
            segment_rms.append(float(group_rms.detach().cpu().item()))
            segment_scales.append(float(denominator.detach().cpu().item()))
    hierarchy: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
    motion_keys: list[str] = []
    attempt_keys: list[str] = []
    motion_mass_shares: list[float] = []
    segment_mass_shares: list[float] = []
    attempt_mass_shares: list[float] = []
    valid_step_rows: list[int] = []
    valid_step_mass_shares: list[float] = []
    motion_count = len(motion_segments)
    for motion, segments in motion_segments.items():
        motion_keys.append(motion)
        motion_mass = 1.0 / float(motion_count)
        motion_mass_shares.append(motion_mass)
        segment_hierarchy: list[tuple[tuple[int, ...], ...]] = []
        for segment_key in segments:
            segment_mass = motion_mass / float(len(segments))
            segment_mass_shares.append(segment_mass)
            attempts = [key for key in attempt_rows if key[:3] == segment_key]
            attempt_hierarchy: list[tuple[int, ...]] = []
            for attempt_key in attempts:
                local_rows = tuple(attempt_rows[attempt_key])
                attempt_mass = segment_mass / float(len(attempts))
                attempt_mass_shares.append(attempt_mass)
                attempt_keys.append(f"{motion}|segment={segment_key[1]}|source={segment_key[2]}|trial={attempt_key[3]}")
                attempt_hierarchy.append(local_rows)
                step_mass = attempt_mass / float(len(local_rows))
                for local_row in local_rows:
                    valid_step_rows.append(valid_row_indices[local_row])
                    valid_step_mass_shares.append(step_mass)
            segment_hierarchy.append(tuple(attempt_hierarchy))
        hierarchy.append(tuple(segment_hierarchy))
    original = advantages
    sign_rows = (original != 0.0) & (prepared != 0.0)
    sign_flip_count = int((torch.sign(original[sign_rows]) != torch.sign(prepared[sign_rows])).sum().item())
    if sign_flip_count:
        raise RuntimeError("grouped_scale_only must preserve every nonzero advantage sign")
    return _FrontRESSegmentPPOGroupedReduction(
        hierarchy=tuple(hierarchy),
        prepared_advantages=prepared,
        transaction_advantage_rms=float(transaction_rms_tensor.detach().cpu().item()),
        segment_advantage_rms=tuple(segment_rms),
        segment_advantage_scales=tuple(segment_scales),
        motion_keys=tuple(motion_keys),
        segment_keys=tuple(segment_keys),
        attempt_keys=tuple(attempt_keys),
        motion_mass_shares=tuple(motion_mass_shares),
        segment_mass_shares=tuple(segment_mass_shares),
        attempt_mass_shares=tuple(attempt_mass_shares),
        valid_step_row_indices=tuple(valid_step_rows),
        valid_step_mass_shares=tuple(valid_step_mass_shares),
    )


def _reduce_frontres_grouped_rows(
    row_values: torch.Tensor,
    hierarchy: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...],
) -> torch.Tensor:
    if not hierarchy:
        return row_values.sum() * 0.0
    motion_losses: list[torch.Tensor] = []
    for segment_hierarchy in hierarchy:
        segment_losses: list[torch.Tensor] = []
        for attempt_hierarchy in segment_hierarchy:
            attempt_losses = [
                row_values[torch.tensor(local_rows, device=row_values.device, dtype=torch.long)].mean()
                for local_rows in attempt_hierarchy
            ]
            segment_losses.append(torch.stack(attempt_losses).mean())
        motion_losses.append(torch.stack(segment_losses).mean())
    return torch.stack(motion_losses).mean()


def _prepare_advantages(
    advantages: torch.Tensor,
    cfg: FrontRESSegmentPPOConfig,
) -> tuple[torch.Tensor, float, int]:
    mode = _advantage_normalization_mode(cfg)
    if mode == "grouped_scale_only":
        raise RuntimeError("grouped_scale_only must be prepared through sealed transaction metadata")
    if advantages.numel() <= 1 or mode == "none":
        return advantages, 1.0, 0
    original = advantages
    eps = abs(float(cfg.advantage_scale_epsilon))
    if mode == "standard":
        scale = advantages.std(unbiased=False) + eps
        scaled = (advantages - advantages.mean()) / scale
    else:
        scale = torch.sqrt(advantages.square().mean()).clamp_min(eps)
        scaled = advantages / scale
    sign_rows = (original != 0.0) & (scaled != 0.0)
    sign_flip_count = int((torch.sign(original[sign_rows]) != torch.sign(scaled[sign_rows])).sum().item())
    return scaled, float(scale.detach().cpu().item()), sign_flip_count


def _evaluate_policy(policy: Any, batch: FrontRESSegmentPPOBatch) -> FrontRESSegmentPolicyEval:
    if hasattr(policy, "evaluate_segment_actions"):
        value = policy.evaluate_segment_actions(batch.observations, batch.actions)
    elif callable(policy):
        value = policy(batch.observations, batch.actions)
    else:
        raise TypeError("policy must define evaluate_segment_actions(observations, actions) or be callable")
    if isinstance(value, FrontRESSegmentPolicyEval):
        return value
    if isinstance(value, dict):
        return FrontRESSegmentPolicyEval(
            log_prob=value["log_prob"],
            value=value["value"],
            entropy=value.get("entropy"),
            mean=value.get("mean"),
            sigma=value.get("sigma"),
            raw_actions=value.get("raw_actions"),
            log_jacobian_contrib=value.get("log_jacobian_contrib"),
        )
    raise TypeError(f"unsupported policy evaluation output: {type(value)!r}")


def _masked_entropy(entropy: torch.Tensor | None, valid: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if entropy is None:
        return like.new_zeros(())
    return entropy[valid].mean()


def _distribution_kl_mean(
    policy_eval: FrontRESSegmentPolicyEval,
    batch: FrontRESSegmentPPOBatch,
    valid: torch.Tensor,
) -> torch.Tensor:
    if batch.old_means is None or batch.old_sigmas is None or policy_eval.mean is None or policy_eval.sigma is None:
        return batch.actions.new_zeros(())
    old_mean = batch.old_means[valid].detach()
    old_sigma = batch.old_sigmas[valid].detach()
    mean = policy_eval.mean[valid].detach()
    sigma = policy_eval.sigma[valid].detach()
    kl = torch.sum(
        torch.log(sigma / old_sigma + 1.0e-5)
        + (old_sigma.square() + (old_mean - mean).square()) / (2.0 * sigma.square())
        - 0.5,
        dim=-1,
    )
    return kl.mean()


def _validate_batch(batch: FrontRESSegmentPPOBatch) -> None:
    if batch.actions.ndim != 2 or batch.actions.shape[-1] != 6:
        raise ValueError(f"actions must have shape [B, 6], got {tuple(batch.actions.shape)}")
    batch_size = batch.actions.shape[0]
    if batch.observations.ndim < 2 or batch.observations.shape[0] != batch_size:
        raise ValueError("observations must have batch dimension B matching actions")
    for name in ("old_log_probs", "old_values", "returns", "advantages", "valid_mask"):
        _require_vector(name, getattr(batch, name), batch_size)
    if batch.segment_ids is not None:
        _require_vector("segment_ids", batch.segment_ids, batch_size)
    if (batch.old_means is None) != (batch.old_sigmas is None):
        raise ValueError("old_means and old_sigmas must be provided together")
    if batch.old_means is not None and tuple(batch.old_means.shape) != (batch_size, 6):
        raise ValueError(f"old_means must have shape [B, 6], got {tuple(batch.old_means.shape)}")
    if batch.old_sigmas is not None and tuple(batch.old_sigmas.shape) != (batch_size, 6):
        raise ValueError(f"old_sigmas must have shape [B, 6], got {tuple(batch.old_sigmas.shape)}")


def _validate_policy_eval(policy_eval: FrontRESSegmentPolicyEval, batch: FrontRESSegmentPPOBatch) -> None:
    batch_size = batch.actions.shape[0]
    _require_vector("policy log_prob", policy_eval.log_prob, batch_size)
    _require_vector("policy value", policy_eval.value, batch_size)
    if policy_eval.entropy is not None:
        _require_vector("policy entropy", policy_eval.entropy, batch_size)
    if policy_eval.mean is not None and tuple(policy_eval.mean.shape) != (batch_size, 6):
        raise ValueError(f"policy mean must have shape [B, 6], got {tuple(policy_eval.mean.shape)}")
    if policy_eval.sigma is not None and tuple(policy_eval.sigma.shape) != (batch_size, 6):
        raise ValueError(f"policy sigma must have shape [B, 6], got {tuple(policy_eval.sigma.shape)}")
    if policy_eval.raw_actions is not None and tuple(policy_eval.raw_actions.shape) != (batch_size, 6):
        raise ValueError(f"policy raw_actions must have shape [B, 6], got {tuple(policy_eval.raw_actions.shape)}")
    if policy_eval.log_jacobian_contrib is not None and tuple(policy_eval.log_jacobian_contrib.shape) != (batch_size, 6):
        raise ValueError(
            "policy log_jacobian_contrib must have shape [B, 6], "
            f"got {tuple(policy_eval.log_jacobian_contrib.shape)}"
        )


def _require_vector(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [B], got {tuple(tensor.shape)}")
