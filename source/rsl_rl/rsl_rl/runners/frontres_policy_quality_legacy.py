"""Legacy matched-counterfactual policy-quality compatibility."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import torch

from rsl_rl.frontres.frontres_policy_quality_manifest import FrontRESPolicyQualityRouteIdentity
if TYPE_CHECKING:
    from rsl_rl.runners.frontres_policy_quality_interfaces import (
        FrontRESPolicyQualityObservationIdentity,
        FrontRESPolicyQualityRouteHooks,
        FrontRESPolicyQualityRouteResult,
    )
    from rsl_rl.runners.frontres_policy_quality_state import (
        FrontRESPolicyQualityScoringState,
    )

class _FrozenFrontRESCheckpointNormalizer(torch.nn.Module):
    """Apply checkpoint prefix stats and frozen-GMT suffix stats without mutating runner state."""

    def __init__(
        self,
        *,
        suffix_template: torch.nn.Module,
        checkpoint_state: dict[str, Any],
        obs_dim: int,
        gmt_dim: int,
    ) -> None:
        super().__init__()
        suffix_state = dict(checkpoint_state)
        template_state = suffix_template.state_dict()
        for key, template_value in template_state.items():
            value = suffix_state.get(key)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(template_value, torch.Tensor)
                and value.ndim > 0
                and template_value.ndim > 0
                and value.shape != template_value.shape
                and value.shape[:-1] == template_value.shape[:-1]
                and value.shape[-1] >= template_value.shape[-1]
            ):
                suffix_state[key] = value[..., -template_value.shape[-1] :]
        suffix_template.load_state_dict(suffix_state, strict=True)
        extra = _extract_checkpoint_extra_stats(checkpoint_state, obs_dim=obs_dim, gmt_dim=gmt_dim)
        if extra is None:
            extra_mean = torch.zeros(obs_dim - gmt_dim)
            extra_std = torch.ones(obs_dim - gmt_dim)
        else:
            extra_mean, extra_std = extra
        self.suffix = suffix_template
        self.gmt_dim = int(gmt_dim)
        self.register_buffer("extra_mean", extra_mean.detach().clone())
        self.register_buffer("extra_std", extra_std.detach().clone())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        extra, suffix = _split_policy_observations(observations, self.gmt_dim)
        if extra is None:
            return self.suffix(suffix)
        normalized_extra = (extra - self.extra_mean.to(extra)) / (self.extra_std.to(extra) + 1e-8)
        return torch.cat((normalized_extra, self.suffix(suffix)), dim=-1)


def _split_policy_observations(observations: torch.Tensor, gmt_dim: int) -> tuple[torch.Tensor | None, torch.Tensor]:
    if observations.shape[-1] <= int(gmt_dim):
        return None, observations
    return observations[..., : -int(gmt_dim)], observations[..., -int(gmt_dim) :]


def _extract_checkpoint_extra_stats(
    state: Mapping[str, Any], *, obs_dim: int, gmt_dim: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    mean = state.get("_mean")
    std = state.get("_std")
    extra_dim = int(obs_dim) - int(gmt_dim)
    if (
        extra_dim <= 0
        or not isinstance(mean, torch.Tensor)
        or not isinstance(std, torch.Tensor)
        or mean.shape[-1] < int(gmt_dim)
        or std.shape != mean.shape
    ):
        return None
    available = max(0, min(extra_dim, int(mean.shape[-1]) - int(gmt_dim)))
    extra_mean = torch.zeros((*mean.shape[:-1], extra_dim), device=mean.device, dtype=mean.dtype)
    extra_std = torch.ones((*std.shape[:-1], extra_dim), device=std.device, dtype=std.dtype)
    if available:
        extra_mean[..., :available] = mean[..., :available]
        extra_std[..., :available] = std[..., :available]
    return extra_mean, extra_std


class FrozenFrontRESTaskActor:
    """Inference-only actor/normalizer pair loaded without runner training state."""

    def __init__(
        self,
        *,
        route: Literal["hsl", "policy"],
        checkpoint_identity: str,
        actor: torch.nn.Module,
        normalizer: torch.nn.Module,
        observation_identity: FrontRESPolicyQualityObservationIdentity,
    ) -> None:
        if route not in ("hsl", "policy"):
            raise ValueError("FrozenFrontRESTaskActor route must be hsl or policy")
        if not checkpoint_identity.strip():
            raise ValueError("checkpoint_identity must be explicit")
        self.route = route
        self.checkpoint_identity = checkpoint_identity
        self.actor = actor.eval()
        self.normalizer = normalizer.eval()
        self.observation_identity = observation_identity
        for module in (self.actor, self.normalizer):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint_payload(
        cls,
        *,
        route: Literal["hsl", "policy"],
        checkpoint_identity: str,
        checkpoint_payload: Mapping[str, Any],
        actor_template: torch.nn.Module,
        normalizer_template: torch.nn.Module,
        observation_identity: FrontRESPolicyQualityObservationIdentity,
        gmt_obs_dim: int | None = None,
    ) -> FrozenFrontRESTaskActor:
        model_state = checkpoint_payload.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            raise ValueError("quality actor checkpoint requires model_state_dict")
        residual_state = model_state.get("residual_actor")
        if not isinstance(residual_state, Mapping):
            student_state = {
                str(key).removeprefix("student."): value
                for key, value in model_state.items()
                if str(key).startswith("student.")
            }
            residual_state = student_state or None
        if not isinstance(residual_state, Mapping):
            raise ValueError("quality actor checkpoint requires residual_actor or student.* weights")
        normalizer_state = checkpoint_payload.get("obs_norm_state_dict")
        if not isinstance(normalizer_state, Mapping):
            raise ValueError("quality actor checkpoint requires obs_norm_state_dict")

        actor = copy.deepcopy(actor_template)
        normalizer = copy.deepcopy(normalizer_template)
        actor.load_state_dict(dict(residual_state), strict=True)
        if gmt_obs_dim is None or observation_identity.expected_obs_dim <= int(gmt_obs_dim):
            normalizer.load_state_dict(dict(normalizer_state), strict=True)
        else:
            normalizer = _FrozenFrontRESCheckpointNormalizer(
                suffix_template=normalizer,
                checkpoint_state=dict(normalizer_state),
                obs_dim=observation_identity.expected_obs_dim,
                gmt_dim=int(gmt_obs_dim),
            )
        return cls(
            route=route,
            checkpoint_identity=checkpoint_identity,
            actor=actor,
            normalizer=normalizer,
            observation_identity=observation_identity,
        )

    @torch.inference_mode()
    def action(self, observations: torch.Tensor) -> torch.Tensor:
        _validate_observations(observations, self.observation_identity)
        normalized = self.normalizer(observations)
        if not isinstance(normalized, torch.Tensor) or tuple(normalized.shape) != tuple(observations.shape):
            raise ValueError("quality normalizer must preserve observation shape")
        raw = self.actor(normalized[:, : self.observation_identity.actor_input_dim])
        if tuple(raw.shape) != (int(observations.shape[0]), 6) or not bool(torch.isfinite(raw).all()):
            raise ValueError("quality residual actor must emit finite full-6D raw actions")
        return raw


class ZeroFrontRESTaskActor:
    def __init__(self, observation_identity: FrontRESPolicyQualityObservationIdentity) -> None:
        self.route = "zero"
        self.checkpoint_identity = "zero:no-checkpoint"
        self.observation_identity = observation_identity

    def action(self, observations: torch.Tensor) -> torch.Tensor:
        _validate_observations(observations, self.observation_identity)
        return torch.zeros((int(observations.shape[0]), 6), dtype=observations.dtype, device=observations.device)


def run_frontres_policy_quality_counterfactuals(
    runner: Any,
    *,
    snapshot: FrontRESPolicyQualityScoringState,
    comparison_signature: str,
    adapters: tuple[ZeroFrontRESTaskActor | FrozenFrontRESTaskActor, ...],
    hooks: FrontRESPolicyQualityRouteHooks,
    horizon_k: int,
    isolation_state: Callable[[], str],
) -> tuple[FrontRESPolicyQualityRouteResult, ...]:
    """Run zero/HSL/policy through the same restored state and canonical owner hooks."""
    # QUALITY-ID-01: 检查 manifest identity -> restored scoring state -> route identity.
    # Result: Q-E1/Q-E2/Q-E3 OFFLINE PASS; real simulator equality pending Q1-F.
    # B1: 进入 route 前校验 adapter order、observation identity 与 comparison signature.
    # B2: 每条 route 在首次 observation 前恢复 initial_state_hash.
    # B3: 返回前要求三路 comparison signature 与 state hash 完全相同.
    if tuple(adapter.route for adapter in adapters) != ("zero", "hsl", "policy"):
        raise ValueError("quality counterfactual order must be exactly zero, hsl, policy")
    if horizon_k <= 0:
        raise ValueError("quality counterfactual horizon_k must be positive")
    observation_identity = adapters[0].observation_identity
    if any(adapter.observation_identity != observation_identity for adapter in adapters[1:]):
        raise ValueError("zero/HSL/policy observation and normalizer identities must match")
    isolated_before = isolation_state()
    results: list[FrontRESPolicyQualityRouteResult] = []
    from rsl_rl.runners import frontres_policy_quality_eval as quality_owner

    for adapter in adapters:
        # QUALITY-ID-01: 每条 route 的任何 observation/action 前先恢复并验证同一 scoring state.
        # Result: Q-E3 OFFLINE PASS; zero/HSL/policy route identity 共享 comparison/state hash.
        state_identity = quality_owner.restore_frontres_policy_quality_state(
            runner,
            snapshot,
            comparison_signature=comparison_signature,
        )
        if hooks.begin_route is not None:
            hooks.begin_route(adapter.route)
        route_identity = FrontRESPolicyQualityRouteIdentity(
            route=adapter.route,
            checkpoint_identity=adapter.checkpoint_identity,
            state=state_identity,
        )
        route_actions: list[torch.Tensor] = []
        for _ in range(horizon_k):
            observations = hooks.observe()
            actions = adapter.action(observations)
            if tuple(actions.shape) != (int(observations.shape[0]), 6):
                raise ValueError("quality route action must preserve full-6D identity")
            # QUALITY-ACTION-01: actor source 之后、正式 task-space application owner 之前截获 6D action.
            # Result: Q-E3 OFFLINE PASS; zero/HSL/policy source、shape、bounds 与 frozen state 已闭合.
            route_actions.append(actions.detach().clone())
            hooks.apply_action(actions)
            hooks.step()
        # QUALITY-GAIN-01 / QUALITY-EXEC-01: K-step rollout 完成后只消费注入的
        # canonical Gain 与 execution owner; 本模块不复制公式或执行指标.
        # Result: Q-E3 OFFLINE CONNECTIVITY PASS; real owner wiring remains Q1-D/Q1-F.
        results.append(
            quality_owner.FrontRESPolicyQualityRouteResult(
                identity=route_identity,
                observation_identity=observation_identity,
                actions=torch.stack(route_actions, dim=0),
                gain=hooks.compute_gain(),
                execution=hooks.capture_execution(),
            )
        )

    if isolation_state() != isolated_before:
        raise RuntimeError("quality evaluation mutated optimizer/sampler/warmup isolation state")
    signatures = {result.identity.comparison_signature for result in results}
    state_hashes = {result.identity.state.initial_state_hash for result in results}
    if signatures != {comparison_signature} or state_hashes != {snapshot.initial_state_hash}:
        raise RuntimeError("quality counterfactual routes did not share one matched scoring state")
    return tuple(results)


def _validate_observations(
    observations: torch.Tensor,
    identity: FrontRESPolicyQualityObservationIdentity,
) -> None:
    if not isinstance(observations, torch.Tensor) or observations.ndim != 2:
        raise ValueError("quality observations must be a rank-2 tensor")
    if int(observations.shape[1]) != identity.expected_obs_dim:
        raise ValueError(
            f"quality observation dim mismatch: expected {identity.expected_obs_dim}, got {int(observations.shape[1])}"
        )
    if not bool(torch.isfinite(observations).all()):
        raise ValueError("quality observations must be finite")


def run_zero_frontres_preroll(
    step_fn: Callable[[torch.Tensor], Any],
    *,
    num_envs: int,
    steps: int,
    device: torch.device | str,
) -> None:
    """Advance GMT with exact zero Delta SE(3); no policy object enters this boundary."""
    if num_envs <= 0 or steps < 0:
        raise ValueError("zero preroll requires num_envs > 0 and steps >= 0")
    zero_action = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)
    for _ in range(steps):
        step_fn(zero_action.clone())
