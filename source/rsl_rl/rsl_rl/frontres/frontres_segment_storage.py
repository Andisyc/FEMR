from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class FrontRESSegmentTransition:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    valid_mask: torch.Tensor
    reset_mask: torch.Tensor
    segment_ids: torch.Tensor
    segment_source: tuple[str, ...] | None = None
    privileged_observations: torch.Tensor | None = None
    old_means: torch.Tensor | None = None
    old_sigmas: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    priority_evidence: Any | None = None


@dataclass(frozen=True)
class FrontRESSegmentStorageBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    valid_mask: torch.Tensor
    segment_ids: torch.Tensor
    action_mask: torch.Tensor | None = None
    privileged_observations: torch.Tensor | None = None
    old_means: torch.Tensor | None = None
    old_sigmas: torch.Tensor | None = None

    def to_ppo_batch(self, batch_cls: Callable[..., Any]) -> Any:
        return batch_cls(
            observations=self.observations,
            actions=self.actions,
            old_log_probs=self.old_log_probs,
            old_values=self.old_values,
            returns=self.returns,
            advantages=self.advantages,
            valid_mask=self.valid_mask,
            segment_ids=self.segment_ids,
            action_mask=self.action_mask,
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

    Segment rewards are already K-step rollout outcomes, so returns default to
    reward and advantages default to reward minus stored value.
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
        self.action_mask = torch.ones(self.capacity, 6, device=self.device)

    def add_transition(self, transition: FrontRESSegmentTransition) -> None:
        transition = self._normalize_transition(transition)
        batch_size = int(transition.actions.shape[0])
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
        advantages = transition.advantages if transition.advantages is not None else returns - transition.values
        self.returns[sl].copy_(returns)
        self.advantages[sl].copy_(advantages)
        self.valid_mask[sl].copy_(transition.valid_mask & transition.reset_mask)
        self.reset_mask[sl].copy_(transition.reset_mask)
        self.segment_ids[sl].copy_(transition.segment_ids)
        if transition.old_means is not None:
            self.old_means[sl].copy_(transition.old_means)
        if transition.old_sigmas is not None:
            self.old_sigmas[sl].copy_(transition.old_sigmas)
        if transition.action_mask is not None:
            self.action_mask[sl].copy_(transition.action_mask)
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
            action_mask=getattr(repair_action, "active_mask", None),
            priority_evidence=payload.get("priority_evidence"),
        )

    def compute_returns_and_advantages(self) -> None:
        active = slice(0, self.step)
        self.returns[active].copy_(self.rewards[active])
        self.advantages[active].copy_(self.returns[active] - self.old_values[active])

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
            "action_mask": self.action_mask[active].detach().cpu(),
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
            "action_mask",
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
            returns=self.returns[idx],
            advantages=self.advantages[idx],
            valid_mask=self.valid_mask[idx],
            segment_ids=self.segment_ids[idx],
            action_mask=self.action_mask[idx],
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
        for name in ("old_means", "old_sigmas", "action_mask"):
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
            action_mask=transition.action_mask.to(self.device).detach() if transition.action_mask is not None else None,
            priority_evidence=transition.priority_evidence,
        )


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
