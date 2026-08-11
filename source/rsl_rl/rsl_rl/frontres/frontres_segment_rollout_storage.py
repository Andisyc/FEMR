"""Mutable Segment rollout storage Unit of Work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rsl_rl.frontres.frontres_return_utility import frontres_symmetric_log_utility
from rsl_rl.frontres.frontres_segment_storage_records import (
    FrontRESSegmentStorageBatch,
    FrontRESSegmentTransition,
)


@dataclass(frozen=True)
class FrontRESSegmentStorageStats:
    size: int
    capacity: int
    valid_frac: float
    reset_success_frac: float
    reward_mean: float
    advantage_mean: float


class FrontRESSegmentRolloutStorage:
    """Independent Stage 3 storage for Segment Replay HRL.

    Segment rewards are already raw K-step Gain outcomes, so returns preserve
    reward while advantages use the active fixed utility minus stored value.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Iterable[int] | torch.Size,
        action_dim: int = 6,
        privileged_obs_shape: Iterable[int] | torch.Size | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if action_dim != 6:
            raise ValueError(f"Segment Replay HRL action_dim must be 6, got {action_dim}")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs_shape = tuple(obs_shape)
        self.privileged_obs_shape = tuple(privileged_obs_shape) if privileged_obs_shape is not None else None
        self.action_dim = int(action_dim)
        self.step = 0
        self.segment_source: list[str] = []
        self.priority_evidence: list[Any] = []
        self.audit_transaction_id: str | None = None
        self.audit_batch_signature: str | None = None
        self.audit_identity_state = "UNCONFIRMED"
        self.transaction_metadata: Any | None = None

        self.observations = torch.zeros(self.capacity, *self.obs_shape, device=self.device)
        self.privileged_observations = (
            torch.zeros(self.capacity, *self.privileged_obs_shape, device=self.device)
            if self.privileged_obs_shape is not None
            else None
        )
        self.actions = torch.zeros(self.capacity, 6, device=self.device)
        self.old_log_probs = torch.zeros(self.capacity, device=self.device)
        self.old_values = torch.zeros(self.capacity, device=self.device)
        self.old_means = torch.zeros(self.capacity, 6, device=self.device)
        self.old_sigmas = torch.zeros(self.capacity, 6, device=self.device)
        self.rewards = torch.zeros(self.capacity, device=self.device)
        self.returns = torch.zeros(self.capacity, device=self.device)
        self.advantages = torch.zeros(self.capacity, device=self.device)
        self.valid_mask = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.reset_mask = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.segment_ids = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.transaction_row_indices = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)

    def add_transition(self, transition: FrontRESSegmentTransition) -> None:
        transition = self._normalize_transition(transition)
        batch_size = int(transition.actions.shape[0])
        transaction_row_indices = _resolve_transaction_row_indices(transition, batch_size=batch_size)
        identity = (
            transition.audit_transaction_id,
            transition.audit_batch_signature,
            transition.audit_identity_state,
        )
        if self.step == 0:
            self.audit_transaction_id, self.audit_batch_signature, self.audit_identity_state = identity
        elif identity != (self.audit_transaction_id, self.audit_batch_signature, self.audit_identity_state):
            raise ValueError(
                "Segment storage received rows from different rollout transactions: "
                f"existing={(self.audit_transaction_id, self.audit_batch_signature, self.audit_identity_state)!r} "
                f"incoming={identity!r}"
            )
        if self.step == 0:
            self.transaction_metadata = transition.transaction_metadata
        elif transition.transaction_metadata is not self.transaction_metadata:
            raise ValueError("Segment storage requires one identical frozen transaction metadata carrier")
        if self.step + batch_size > self.capacity:
            raise OverflowError("FrontRESSegmentRolloutStorage overflow; call clear() before adding more transitions")
        sl = slice(self.step, self.step + batch_size)
        self.observations[sl].copy_(transition.observations)
        if self.privileged_observations is not None:
            if transition.privileged_observations is None:
                raise ValueError("privileged_observations are required by this storage")
            self.privileged_observations[sl].copy_(transition.privileged_observations)
        self.actions[sl].copy_(transition.actions)
        self.old_log_probs[sl].copy_(transition.old_log_probs)
        self.old_values[sl].copy_(transition.values)
        self.rewards[sl].copy_(transition.rewards)
        returns = transition.returns if transition.returns is not None else transition.rewards
        advantages = frontres_symmetric_log_utility(returns) - transition.values
        if transition.advantages is not None and not torch.allclose(
            transition.advantages,
            advantages,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("TRAIN-v021 storage rejects non-utility carried advantages")
        self.returns[sl].copy_(returns)
        self.advantages[sl].copy_(advantages)
        self.valid_mask[sl].copy_(transition.valid_mask & transition.reset_mask)
        self.reset_mask[sl].copy_(transition.reset_mask)
        self.segment_ids[sl].copy_(transition.segment_ids)
        if transaction_row_indices is not None:
            self.transaction_row_indices[sl].copy_(transaction_row_indices)
        if transition.old_means is not None:
            self.old_means[sl].copy_(transition.old_means)
        if transition.old_sigmas is not None:
            self.old_sigmas[sl].copy_(transition.old_sigmas)
        self.segment_source.extend(transition.segment_source or ("unknown",) * batch_size)
        if transition.priority_evidence is not None:
            self.priority_evidence.append(_detach_evidence(transition.priority_evidence))
        self.step += batch_size

    def write(self, **payload: Any) -> None:
        self.add_transition(self.transition_from_connector_payload(payload))

    def transition_from_connector_payload(self, payload: dict[str, Any]) -> FrontRESSegmentTransition:
        batch = payload["batch"]
        repair_action = payload["repair_action"]
        reward_result = payload["reward_result"]
        reset_result = payload["reset_result"]
        sample = payload.get("sample")
        policy_output = payload.get("policy_output") or payload.get("raw_action")
        observations = _required_attr_or_key(policy_output, "observations")
        old_log_probs = _required_attr_or_key(policy_output, "log_prob")
        values = _required_attr_or_key(policy_output, "value")
        old_means = _optional_attr_or_key(policy_output, "mean")
        old_sigmas = _optional_attr_or_key(policy_output, "sigma")
        return FrontRESSegmentTransition(
            observations=observations,
            actions=repair_action.projected_delta_se,
            old_log_probs=old_log_probs,
            values=values,
            rewards=reward_result.reward,
            valid_mask=reward_result.valid_mask,
            reset_mask=_required_reset_mask(reset_result),
            segment_ids=batch.segment_ids,
            segment_source=getattr(sample, "source", None),
            privileged_observations=_optional_attr_or_key(policy_output, "privileged_observations"),
            old_means=old_means,
            old_sigmas=old_sigmas,
            priority_evidence=payload.get("priority_evidence"),
            audit_transaction_id=payload.get("audit_transaction_id"),
            audit_batch_signature=payload.get("audit_batch_signature"),
            audit_identity_state=str(payload.get("audit_identity_state", "UNCONFIRMED")),
            transaction_metadata=payload.get("transaction_metadata"),
            transaction_row_indices=payload.get("transaction_row_indices"),
        )

    def compute_returns_and_advantages(
        self,
        *,
        reward_steps: torch.Tensor | None = None,
        done_steps: torch.Tensor | None = None,
        horizon: int | torch.Tensor | None = None,
        gamma: float = 1.0,
    ) -> None:
        # QUALITY-CREDIT-01: 检查 canonical Gain steps -> returns/advantages -> PPO batch.
        # Result: PENDING_Q_EVIDENCE.
        # B1: policy-row effective K 与 Gain steps 在 return aggregation 前冻结.
        # B2: sign-preserving credit 与 valid mask 决定每行 advantage.
        # B3: to_ppo_batch 前统计 sign、bucket contribution 与 dominance.
        """按每行 K 和 done 边界累计 Segment return/advantage.

        函数名说明:
            `compute_returns_and_advantages` 是 Segment temporal-credit owner, 从
            canonical per-step Gain 构造 PPO 学习信号; 它不是 Gain 公式或 PPO loss.

        主链路:
            上游: rollout capture 提供 `[T, B]` Gain, done 和 per-row horizon K.
            下游: finalized returns/advantages 经 `to_ppo_batch` 进入 PPO.

        语义:
            每行只累计 alive 且 `offset < K` 的 paired Gain. Advantage 保持
            `return - old_value`, 不混入 full environment reward.
        """
        # B1: 读取 canonical per-step Gain 和每行 effective K.
        storage_slice = slice(0, self.step)
        if reward_steps is None:
            self.returns[storage_slice].copy_(self.rewards[storage_slice])
            utility_returns = frontres_symmetric_log_utility(self.returns[storage_slice])
            self.advantages[storage_slice].copy_(utility_returns - self.old_values[storage_slice])
            return

        if reward_steps.ndim != 2:
            raise ValueError(f"reward_steps must be rank-2 [T, B], got {tuple(reward_steps.shape)}")
        if int(reward_steps.shape[1]) < self.step:
            raise ValueError(f"reward_steps must have at least {self.step} batch entries, got {int(reward_steps.shape[1])}")
        if done_steps is not None and tuple(done_steps.shape) != tuple(reward_steps.shape):
            raise ValueError(f"done_steps shape {tuple(done_steps.shape)} must match reward_steps {tuple(reward_steps.shape)}")

        step_count = int(reward_steps.shape[0])
        if isinstance(horizon, torch.Tensor):
            horizon_k = horizon.to(device=self.device, dtype=torch.long).reshape(-1)
            if int(horizon_k.numel()) != self.step:
                raise ValueError(f"horizon must have {self.step} rows, got {int(horizon_k.numel())}")
            horizon_k = horizon_k.clamp(min=1, max=step_count)
        else:
            scalar_horizon = min(step_count, max(1, int(horizon if horizon is not None else step_count)))
            horizon_k = torch.full((self.step,), scalar_horizon, dtype=torch.long, device=self.device)
        return_horizon = int(horizon_k.max().item())
        rewards = reward_steps[:return_horizon, : self.step].to(device=self.device, dtype=torch.float32)
        if done_steps is None:
            dones = torch.zeros_like(rewards, dtype=torch.bool)
        else:
            dones = done_steps[:return_horizon, : self.step].to(device=self.device).bool()

        # B2: 仅在每行仍 alive 且位于 K 内时累计 discounted Gain.
        returns = torch.zeros(self.step, device=self.device, dtype=torch.float32)
        alive = torch.ones(self.step, device=self.device, dtype=torch.float32)
        discount = 1.0
        gamma_value = float(gamma)
        for offset in range(return_horizon):
            horizon_active = offset < horizon_k
            returns = returns + (discount * alive * horizon_active.float() * rewards[offset])
            alive = alive * (~(dones[offset] & horizon_active)).float()
            discount *= gamma_value

        self.returns[storage_slice].copy_(returns)
        utility_returns = frontres_symmetric_log_utility(self.returns[storage_slice])
        self.advantages[storage_slice].copy_(utility_returns - self.old_values[storage_slice])
        # B3: finalized returns/advantages 已准备好进入 PPO batch conversion.

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 1,
        shuffle: bool = True,
    ):
        if self.step == 0:
            raise RuntimeError("cannot generate mini-batches from empty segment storage")
        if num_mini_batches <= 0:
            raise ValueError(f"num_mini_batches must be positive, got {num_mini_batches}")
        total = self.step
        mini_batch_size = max(1, (total + num_mini_batches - 1) // num_mini_batches)
        for _ in range(num_epochs):
            indices = torch.randperm(total, device=self.device) if shuffle else torch.arange(total, device=self.device)
            for start in range(0, total, mini_batch_size):
                idx = indices[start : min(start + mini_batch_size, total)]
                yield self._batch(idx)

    def full_batch(self) -> FrontRESSegmentStorageBatch:
        return self._batch(torch.arange(self.step, device=self.device))

    def stats(self) -> FrontRESSegmentStorageStats:
        if self.step == 0:
            return FrontRESSegmentStorageStats(0, self.capacity, 0.0, 0.0, 0.0, 0.0)
        active = slice(0, self.step)
        return FrontRESSegmentStorageStats(
            size=self.step,
            capacity=self.capacity,
            valid_frac=float(self.valid_mask[active].float().mean().item()),
            reset_success_frac=float(self.reset_mask[active].float().mean().item()),
            reward_mean=float(self.rewards[active].mean().item()),
            advantage_mean=float(self.advantages[active].mean().item()),
        )

    def clear(self) -> None:
        self.step = 0
        self.segment_source.clear()
        self.priority_evidence.clear()
        self.audit_transaction_id = None
        self.audit_batch_signature = None
        self.audit_identity_state = "UNCONFIRMED"
        self.transaction_metadata = None
        self.transaction_row_indices.fill_(-1)

    def state_dict(self) -> dict[str, Any]:
        active = slice(0, self.step)
        return {
            "step": self.step,
            "observations": self.observations[active].detach().cpu(),
            "privileged_observations": self.privileged_observations[active].detach().cpu()
            if self.privileged_observations is not None
            else None,
            "actions": self.actions[active].detach().cpu(),
            "old_log_probs": self.old_log_probs[active].detach().cpu(),
            "old_values": self.old_values[active].detach().cpu(),
            "old_means": self.old_means[active].detach().cpu(),
            "old_sigmas": self.old_sigmas[active].detach().cpu(),
            "rewards": self.rewards[active].detach().cpu(),
            "returns": self.returns[active].detach().cpu(),
            "advantages": self.advantages[active].detach().cpu(),
            "valid_mask": self.valid_mask[active].detach().cpu(),
            "reset_mask": self.reset_mask[active].detach().cpu(),
            "segment_ids": self.segment_ids[active].detach().cpu(),
            "segment_source": tuple(self.segment_source),
            "priority_evidence": tuple(self.priority_evidence),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        step = int(state["step"])
        if step > self.capacity:
            raise ValueError(f"stored step {step} exceeds capacity {self.capacity}")
        self.clear()
        self.step = step
        active = slice(0, self.step)
        for name in (
            "observations",
            "actions",
            "old_log_probs",
            "old_values",
            "old_means",
            "old_sigmas",
            "rewards",
            "returns",
            "advantages",
            "valid_mask",
            "reset_mask",
            "segment_ids",
        ):
            getattr(self, name)[active].copy_(state[name].to(self.device))
        if self.privileged_observations is not None and state.get("privileged_observations") is not None:
            self.privileged_observations[active].copy_(state["privileged_observations"].to(self.device))
        self.segment_source = list(state.get("segment_source", ("unknown",) * self.step))
        self.priority_evidence = list(state.get("priority_evidence", ()))

    def _batch(self, idx: torch.Tensor) -> FrontRESSegmentStorageBatch:
        privileged = self.privileged_observations[idx] if self.privileged_observations is not None else None
        return FrontRESSegmentStorageBatch(
            observations=self.observations[idx],
            privileged_observations=privileged,
            actions=self.actions[idx],
            old_log_probs=self.old_log_probs[idx],
            old_values=self.old_values[idx],
            old_means=self.old_means[idx],
            old_sigmas=self.old_sigmas[idx],
            rewards=self.rewards[idx],
            returns=self.returns[idx],
            advantages=self.advantages[idx],
            valid_mask=self.valid_mask[idx],
            segment_ids=self.segment_ids[idx],
            audit_transaction_id=self.audit_transaction_id,
            audit_batch_signature=self.audit_batch_signature,
            audit_identity_state=self.audit_identity_state,
            transaction_metadata=self.transaction_metadata,
            transaction_row_indices=(self.transaction_row_indices[idx] if self.transaction_metadata is not None else None),
        )

    def _normalize_transition(self, transition: FrontRESSegmentTransition) -> FrontRESSegmentTransition:
        if transition.actions.ndim != 2 or transition.actions.shape[-1] != 6:
            raise ValueError(f"actions must have shape [B, 6], got {tuple(transition.actions.shape)}")
        batch_size = transition.actions.shape[0]
        _require_batch("observations", transition.observations, batch_size)
        for name in ("old_log_probs", "values", "rewards", "valid_mask", "reset_mask", "segment_ids"):
            _require_vector(name, getattr(transition, name), batch_size)
        if transition.segment_source is not None and len(transition.segment_source) != batch_size:
            raise ValueError("segment_source length must match batch size")
        if transition.privileged_observations is not None:
            _require_batch("privileged_observations", transition.privileged_observations, batch_size)
        for name in ("old_means", "old_sigmas"):
            value = getattr(transition, name)
            if value is not None and tuple(value.shape) != (batch_size, 6):
                raise ValueError(f"{name} must have shape [B, 6], got {tuple(value.shape)}")
        for name in ("returns", "advantages"):
            value = getattr(transition, name)
            if value is not None:
                _require_vector(name, value, batch_size)
        return FrontRESSegmentTransition(
            observations=transition.observations.to(self.device).detach(),
            actions=transition.actions.to(self.device).detach(),
            old_log_probs=transition.old_log_probs.to(self.device).detach(),
            values=transition.values.to(self.device).detach(),
            rewards=transition.rewards.to(self.device).detach(),
            valid_mask=transition.valid_mask.to(self.device).bool().detach(),
            reset_mask=transition.reset_mask.to(self.device).bool().detach(),
            segment_ids=transition.segment_ids.to(self.device, dtype=torch.long).detach(),
            segment_source=transition.segment_source,
            privileged_observations=transition.privileged_observations.to(self.device).detach()
            if transition.privileged_observations is not None
            else None,
            old_means=transition.old_means.to(self.device).detach() if transition.old_means is not None else None,
            old_sigmas=transition.old_sigmas.to(self.device).detach() if transition.old_sigmas is not None else None,
            returns=transition.returns.to(self.device).detach() if transition.returns is not None else None,
            advantages=transition.advantages.to(self.device).detach() if transition.advantages is not None else None,
            priority_evidence=transition.priority_evidence,
            audit_transaction_id=transition.audit_transaction_id,
            audit_batch_signature=transition.audit_batch_signature,
            audit_identity_state=transition.audit_identity_state,
            transaction_metadata=transition.transaction_metadata,
            transaction_row_indices=(
                transition.transaction_row_indices.to(self.device, dtype=torch.long).detach()
                if transition.transaction_row_indices is not None
                else None
            ),
        )


def _resolve_transaction_row_indices(
    transition: FrontRESSegmentTransition,
    *,
    batch_size: int,
) -> torch.Tensor | None:
    """Return the sealed metadata row for every storage row without rebuilding identity."""

    metadata = transition.transaction_metadata
    raw_indices = transition.transaction_row_indices
    if metadata is None:
        if raw_indices is not None:
            raise ValueError("transaction_row_indices require transaction_metadata")
        return None
    validate = getattr(metadata, "validate", None)
    if not callable(validate):
        raise ValueError("transaction metadata requires validate()")
    validate()
    metadata_segment_ids = getattr(metadata, "segment_ids", None)
    metadata_size = int(getattr(metadata, "batch_size", 0) or 0)
    if not isinstance(metadata_segment_ids, torch.Tensor):
        raise ValueError("transaction metadata requires segment_ids")
    if metadata_size <= 0:
        metadata_size = int(metadata_segment_ids.numel())
    if metadata_segment_ids.ndim != 1 or int(metadata_segment_ids.numel()) != metadata_size:
        raise ValueError("transaction metadata segment_ids must be a rank-1 metadata vector")
    if raw_indices is None:
        if metadata_size != batch_size:
            raise ValueError(
                "transaction metadata requires explicit transaction_row_indices when storage rows are a subset"
            )
        row_indices = torch.arange(batch_size, device=transition.segment_ids.device, dtype=torch.long)
    else:
        _require_vector("transaction_row_indices", raw_indices, batch_size)
        if raw_indices.requires_grad:
            raise ValueError("transaction_row_indices must be detached")
        row_indices = raw_indices.detach().to(device=transition.segment_ids.device, dtype=torch.long)
        if bool((row_indices < 0).any()) or bool((row_indices >= metadata_size).any()):
            raise ValueError("transaction_row_indices contain an out-of-range metadata row")
    selected_segment_ids = metadata_segment_ids.detach().to(
        device=transition.segment_ids.device,
        dtype=torch.long,
    )[row_indices]
    if not torch.equal(selected_segment_ids, transition.segment_ids):
        raise ValueError("transaction metadata segment_ids do not match the storage rows")
    return row_indices


def _require_batch(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{name} batch dimension must be {batch_size}, got {tensor.shape[0]}")


def _require_vector(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [B], got {tuple(tensor.shape)}")


def _required_attr_or_key(obj: Any, name: str) -> torch.Tensor:
    value = _optional_attr_or_key(obj, name)
    if value is None:
        raise ValueError(f"connector payload must provide policy {name}")
    return value


def _required_reset_mask(reset_result: Any) -> torch.Tensor:
    value = _optional_attr_or_key(reset_result, "success_mask")
    if value is None:
        value = _optional_attr_or_key(reset_result, "valid_mask")
    if value is None:
        raise ValueError("connector payload reset_result must provide success_mask or valid_mask")
    return value


def _optional_attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _detach_evidence(evidence: Any) -> Any:
    if isinstance(evidence, torch.Tensor):
        return evidence.detach().cpu()
    if isinstance(evidence, dict):
        return {key: _detach_evidence(value) for key, value in evidence.items()}
    if hasattr(evidence, "__dict__"):
        return {
            key: _detach_evidence(value)
            for key, value in vars(evidence).items()
            if not key.startswith("_")
        }
    return evidence
