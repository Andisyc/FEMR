"""Independent Q2-D scale-sweep and policy-mean causality primitives."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch


Q2D_SCALE_FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def _state_hash(module: torch.nn.Module) -> str:
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def scale_route_name(scale: float) -> str:
    if scale < 0 or not torch.isfinite(torch.tensor(scale)):
        raise ValueError("Q2-D action scale must be finite and nonnegative")
    return f"hsl_scale_{scale:.2f}".replace(".", "p")


@dataclass(frozen=True)
class Q2DScaleRouteResult:
    route: str
    scale: float
    initial_state_hash: str
    actions: torch.Tensor
    gain: Any
    execution: Mapping[str, Any]


class ScaledHSLActionAdapter:
    """Scale one frozen HSL action source without changing its observation path."""

    def __init__(self, base_adapter: Any, scale: float) -> None:
        self.base_adapter = base_adapter
        self.scale = float(scale)
        self.route = scale_route_name(self.scale)
        self.checkpoint_identity = getattr(base_adapter, "checkpoint_identity", "")
        self.observation_identity = getattr(base_adapter, "observation_identity", None)

    def action(self, observations: torch.Tensor) -> torch.Tensor:
        base = self.base_adapter.action(observations)
        if not isinstance(base, torch.Tensor) or base.ndim != 2 or base.shape[-1] != 6:
            raise ValueError("Q2-D HSL adapter must emit [batch, 6]")
        scaled = base * self.scale
        if not bool(torch.isfinite(scaled).all()):
            raise ValueError("Q2-D scaled HSL action must remain finite")
        return scaled


def run_q2d_scale_sweep(
    *,
    base_adapter: Any,
    scales: Iterable[float],
    horizon_k: int,
    restore_state: Callable[[], str],
    begin_route: Callable[[str], None],
    observe: Callable[[], torch.Tensor],
    apply_action: Callable[[torch.Tensor], None],
    step: Callable[[], Any],
    compute_gain: Callable[[], Any],
    capture_execution: Callable[[], Mapping[str, Any]],
    isolation_state: Callable[[], str],
    set_audit_identity: Callable[[str, float, str], None] | None = None,
) -> tuple[Q2DScaleRouteResult, ...]:
    """Run scaled HSL routes from one restored state through injected canonical owners."""

    if horizon_k <= 0:
        raise ValueError("Q2-D horizon_k must be positive")
    scale_tuple = tuple(float(scale) for scale in scales)
    if scale_tuple != tuple(sorted(set(scale_tuple))) or not scale_tuple:
        raise ValueError("Q2-D scales must be nonempty, unique, and sorted")
    isolated_before = isolation_state()
    results = []
    expected_state_hash = None
    for scale in scale_tuple:
        adapter = ScaledHSLActionAdapter(base_adapter, scale)
        state_hash = restore_state()
        expected_state_hash = state_hash if expected_state_hash is None else expected_state_hash
        if state_hash != expected_state_hash:
            raise RuntimeError("Q2-D scale routes did not restore the same initial state")
        begin_route(adapter.route)
        if set_audit_identity is not None:
            set_audit_identity(adapter.route, scale, state_hash)
        actions = []
        for _ in range(horizon_k):
            value = adapter.action(observe())
            actions.append(value.detach().clone())
            apply_action(value)
            step()
        results.append(
            Q2DScaleRouteResult(
                route=adapter.route,
                scale=scale,
                initial_state_hash=state_hash,
                actions=torch.stack(actions),
                gain=compute_gain(),
                execution=dict(capture_execution()),
            )
        )
    if isolation_state() != isolated_before:
        raise RuntimeError("Q2-D scale sweep mutated training isolation state")
    return tuple(results)


# B4: QUALITY-CREDIT-01 将 sampled raw action 和 advantage 转成 mean score direction.
def gaussian_mean_score_gradient(
    raw_actions: torch.Tensor,
    old_means: torch.Tensor,
    old_sigmas: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the on-policy mean score direction E[A*(a-mu)/sigma^2]."""

    if raw_actions.shape != old_means.shape or raw_actions.shape != old_sigmas.shape:
        raise ValueError("Q2-D raw action, old mean, and sigma shapes must match")
    if raw_actions.ndim != 2 or raw_actions.shape[-1] != 6:
        raise ValueError("Q2-D score-gradient inputs must be [batch, 6]")
    advantages = advantages.reshape(-1)
    valid_mask = valid_mask.reshape(-1).bool()
    if advantages.numel() != raw_actions.shape[0] or valid_mask.numel() != raw_actions.shape[0]:
        raise ValueError("Q2-D advantage/mask rows must match actions")
    valid = valid_mask & torch.isfinite(advantages) & torch.isfinite(raw_actions).all(-1)
    if not bool(valid.any()):
        raise ValueError("Q2-D score-gradient requires at least one finite valid row")
    sigma = old_sigmas[valid].clamp_min(1e-8)
    score = advantages[valid, None] * (raw_actions[valid] - old_means[valid]) / sigma.square()
    return score.mean(dim=0)

# B4: QUALITY-CREDIT-01 冻结正式 optimizer.step 前的 transaction-complete PPO tuple.
def write_q2d_credit_tuple(
    *,
    result_path: str,
    raw_actions: torch.Tensor,
    bounded_actions: torch.Tensor,
    old_means: torch.Tensor,
    old_sigmas: torch.Tensor,
    gains: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    audit_transaction_id: str | None,
    audit_batch_signature: str | None,
    audit_identity_state: str,
) -> dict[str, Any]:
    """Persist the exact pre-update PPO credit tuple without mutating training state."""

    tensors = {
        "raw_actions": raw_actions,
        "bounded_actions": bounded_actions,
        "old_means": old_means,
        "old_sigmas": old_sigmas,
    }
    if any(value.ndim != 2 or value.shape[-1] != 6 for value in tensors.values()):
        raise ValueError("Q2-D action/distribution tuple fields must be [batch, 6]")
    row_count = int(raw_actions.shape[0])
    if any(tuple(value.shape) != (row_count, 6) for value in tensors.values()):
        raise ValueError("Q2-D action/distribution tuple fields must share shape")
    vectors = {
        "gains": gains,
        "returns": returns,
        "advantages": advantages,
        "valid_mask": valid_mask,
        "segment_ids": segment_ids,
    }
    if any(int(value.reshape(-1).numel()) != row_count for value in vectors.values()):
        raise ValueError("Q2-D credit tuple vectors must match the PPO batch row count")
    if audit_identity_state != "complete" or not audit_transaction_id or not audit_batch_signature:
        raise ValueError("Q2-D credit tuple requires complete rollout transaction identity")
    numeric = (*tensors.values(), gains, returns, advantages)
    if not all(bool(torch.isfinite(value.detach()).all()) for value in numeric):
        raise ValueError("Q2-D credit tuple must be finite")

    score_direction = gaussian_mean_score_gradient(
        raw_actions,
        old_means,
        old_sigmas,
        advantages,
        valid_mask,
    )
    payload = {
        "schema_version": "frontres_policy_quality_q2d_credit_v1",
        "audit_transaction_id": audit_transaction_id,
        "audit_batch_signature": audit_batch_signature,
        "audit_identity_state": audit_identity_state,
        "row_count": row_count,
        "raw_actions": raw_actions.detach().cpu().tolist(),
        "bounded_actions": bounded_actions.detach().cpu().tolist(),
        "old_means": old_means.detach().cpu().tolist(),
        "old_sigmas": old_sigmas.detach().cpu().tolist(),
        "gains": gains.detach().reshape(-1).cpu().tolist(),
        "returns": returns.detach().reshape(-1).cpu().tolist(),
        "advantages": advantages.detach().reshape(-1).cpu().tolist(),
        "valid_mask": valid_mask.detach().bool().reshape(-1).cpu().tolist(),
        "segment_ids": segment_ids.detach().long().reshape(-1).cpu().tolist(),
        "mean_score_direction": score_direction.detach().cpu().tolist(),
    }
    destination = Path(result_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    temporary.replace(destination)
    return payload


def mean_correction_projection(
    before_mean: torch.Tensor,
    after_mean: torch.Tensor,
    current_action: torch.Tensor,
    preferred_action: torch.Tensor,
) -> dict[str, float | bool]:
    """Check whether the mean delta points from current action toward preferred action."""

    for tensor in (before_mean, after_mean, current_action, preferred_action):
        if tensor.shape != before_mean.shape:
            raise ValueError("Q2-D mean/action tensors must share shape")
    delta = (after_mean - before_mean).reshape(-1)
    desired = (preferred_action - current_action).reshape(-1)
    desired_norm = desired.norm()
    projection = torch.dot(delta, desired) / desired_norm.clamp_min(1e-12)
    cosine = torch.dot(delta, desired) / (delta.norm() * desired_norm).clamp_min(1e-12)
    return {
        "mean_delta_l2": float(delta.norm()),
        "desired_delta_l2": float(desired_norm),
        "projection_toward_preferred": float(projection),
        "cosine_toward_preferred": float(cosine),
        "moves_toward_preferred": bool(projection > 0),
    }


# B4: QUALITY-UPDATE-01 在 policy clone 上执行一次 update 并投影 mean delta.
def run_isolated_controlled_update(
    *,
    model: torch.nn.Module,
    observations: torch.Tensor,
    mean_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    update_fn: Callable[[torch.nn.Module], None],
    current_action: torch.Tensor,
    preferred_action: torch.Tensor,
) -> dict[str, Any]:
    """Update a clone once and report mean direction while preserving the source model."""

    source_hash = _state_hash(model)
    clone = copy.deepcopy(model)
    before = mean_fn(clone, observations).detach().clone()
    update_fn(clone)
    after = mean_fn(clone, observations).detach().clone()
    if _state_hash(model) != source_hash:
        raise RuntimeError("Q2-D controlled update mutated the source policy")
    return {
        "before_mean": before,
        "after_mean": after,
        "source_model_unchanged": True,
        "direction": mean_correction_projection(before, after, current_action, preferred_action),
    }
