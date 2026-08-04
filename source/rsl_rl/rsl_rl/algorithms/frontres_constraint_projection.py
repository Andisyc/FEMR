"""Grouped Physics projection and actual Actor-update authority."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import itertools
import math
from typing import Any

import torch


@dataclass(frozen=True)
class FrontRESConstraintProjectionResult:
    status: str
    direction: torch.Tensor
    active_families: tuple[str, ...]
    gradient_norms: dict[str, float]
    directional_derivatives: dict[str, float]
    intent_direction_norm: float
    projected_direction_norm: float
    dual_coefficients: dict[str, float]
    constraint_gram: tuple[tuple[float, ...], ...]
    intent_directional_derivatives: dict[str, float]
    kkt_max_violation: float
    # Internal optimizer-authority carrier. These vectors use the exact flat
    # actor/std parameter order installed below and are never checkpointed or
    # exposed to the actor.
    constraint_gradient_vectors: dict[str, torch.Tensor]


@dataclass(frozen=True)
class FrontRESActualOptimizerCommitResult:
    """Result of the single PPO-v004 optimizer call and Actor commit guard."""

    projection: FrontRESConstraintProjectionResult
    optimizer_candidate_actor_delta_l2: float
    committed_actor_delta_l2: float
    actor_optimizer_state_preserved: bool


def project_frontres_grouped_constraint_direction(
    intent_direction: torch.Tensor,
    constraint_gradients: dict[str, torch.Tensor],
    constraint_levels: dict[str, float],
    *,
    eps_grad: float = 1.0e-10,
    tolerance: float = 1.0e-8,
) -> FrontRESConstraintProjectionResult:
    """Project one actor direction into the joint Contact/ZMP/survival cone."""

    family_order = ("contact", "zmp", "survival")
    if intent_direction.ndim != 1 or not bool(torch.isfinite(intent_direction).all()):
        raise ValueError("FRS-PPO-v004 requires one finite flattened Intent direction")
    active: list[str] = []
    rows: list[torch.Tensor] = []
    norms: dict[str, float] = {}
    no_direction = False
    for family in family_order:
        gradient = constraint_gradients.get(family)
        level = float(constraint_levels.get(family, float("nan")))
        if not isinstance(gradient, torch.Tensor) or gradient.shape != intent_direction.shape:
            raise ValueError(f"FRS-PPO-v004 requires one aligned {family} constraint gradient")
        if not bool(torch.isfinite(gradient).all()) or not math.isfinite(level) or level < 0.0:
            raise ValueError(f"FRS-PPO-v004 rejects invalid {family} constraint state")
        norm = float(gradient.norm().detach().cpu().item())
        norms[family] = norm
        if level > 0.0:
            active.append(family)
            if norm <= float(eps_grad):
                no_direction = True
            else:
                rows.append(gradient)
    zero = torch.zeros_like(intent_direction)
    gram_rows: tuple[tuple[float, ...], ...] = ()
    intent_dots: dict[str, float] = {}
    if no_direction:
        return FrontRESConstraintProjectionResult(
            status="NO_EMPIRICAL_DIRECTION", direction=zero, active_families=tuple(active),
            gradient_norms=norms, directional_derivatives={family: 0.0 for family in active},
            intent_direction_norm=float(intent_direction.norm().item()), projected_direction_norm=0.0,
            dual_coefficients={family: 0.0 for family in active}, constraint_gram=gram_rows,
            intent_directional_derivatives=intent_dots, kkt_max_violation=0.0,
            constraint_gradient_vectors={key: value.detach() for key, value in constraint_gradients.items()},
        )
    if not rows:
        return FrontRESConstraintProjectionResult(
            status="INTENT_FEASIBLE", direction=intent_direction.clone(), active_families=(),
            gradient_norms=norms, directional_derivatives={},
            intent_direction_norm=float(intent_direction.norm().item()),
            projected_direction_norm=float(intent_direction.norm().item()),
            dual_coefficients={}, constraint_gram=(), intent_directional_derivatives={}, kkt_max_violation=0.0,
            constraint_gradient_vectors={key: value.detach() for key, value in constraint_gradients.items()},
        )
    matrix = torch.stack(rows, dim=0)
    gram = matrix @ matrix.T
    gram_rows = tuple(tuple(float(value) for value in row) for row in gram.detach().cpu().tolist())
    intent_dots = {
        family: float(value) for family, value in zip(active, (matrix @ intent_direction).detach().cpu().tolist(), strict=True)
    }

    def project(seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        best: torch.Tensor | None = None
        best_dual: torch.Tensor | None = None
        best_distance = float("inf")
        count = int(matrix.shape[0])
        for size in range(count + 1):
            for subset in itertools.combinations(range(count), size):
                if not subset:
                    candidate = seed
                    full_dual = torch.zeros(count, device=matrix.device, dtype=matrix.dtype)
                else:
                    index = torch.tensor(subset, device=matrix.device, dtype=torch.long)
                    selected = matrix.index_select(0, index)
                    gram = selected @ selected.T
                    rhs = selected @ seed
                    lambdas = torch.linalg.pinv(gram) @ rhs
                    if bool((lambdas < -float(tolerance)).any()):
                        continue
                    candidate = seed - selected.T @ lambdas.clamp_min(0.0)
                    full_dual = torch.zeros(count, device=matrix.device, dtype=matrix.dtype)
                    full_dual.index_copy_(0, index, lambdas.clamp_min(0.0))
                if not bool(torch.isfinite(candidate).all()) or bool((matrix @ candidate > float(tolerance)).any()):
                    continue
                distance = float((candidate - seed).square().sum().detach().cpu().item())
                if distance < best_distance:
                    best, best_dual, best_distance = candidate, full_dual, distance
        return None if best is None or best_dual is None else (best.clone(), best_dual.clone())

    projected_solution = project(intent_direction)
    if projected_solution is not None:
        projected, dual = projected_solution
        dots = matrix @ projected
        if float(projected.norm().item()) > float(eps_grad) and bool((dots < -float(tolerance)).any()):
            return FrontRESConstraintProjectionResult(
                status="PROJECTED_INTENT", direction=projected, active_families=tuple(active),
                gradient_norms=norms,
                directional_derivatives={family: float(dots[index].item()) for index, family in enumerate(active)},
                intent_direction_norm=float(intent_direction.norm().item()),
                projected_direction_norm=float(projected.norm().item()),
                dual_coefficients={family: float(dual[index].item()) for index, family in enumerate(active)},
                constraint_gram=gram_rows, intent_directional_derivatives=intent_dots,
                kkt_max_violation=float(torch.relu(dots).max().item()),
                constraint_gradient_vectors={key: value.detach() for key, value in constraint_gradients.items()},
            )
    normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(float(eps_grad))
    recovery_solution = project(-normalized.mean(dim=0))
    if recovery_solution is not None:
        recovery, _ = recovery_solution
        dots = matrix @ recovery
        if float(recovery.norm().item()) > float(eps_grad) and bool((dots < -float(tolerance)).any()):
            target_norm = max(
                float(intent_direction.norm().item()),
                float(torch.sqrt(torch.stack([row.norm().square() for row in rows]).mean()).item()),
            )
            recovery = recovery * (target_norm / float(recovery.norm().item()))
        # B5: Gain-v006/PPO-v004 final owner reprojects recovery after norm scaling.
        # B4: 放大后重新投影, 防止容差内的浮点残差被同步放大为真实约束违规.
            postscale_solution = project(recovery)
            if postscale_solution is not None:
                recovery, _ = postscale_solution
                dots = matrix @ recovery
                if (
                    float(recovery.norm().item()) > float(eps_grad)
                    and not bool((dots > float(tolerance)).any())
                    and bool((dots < -float(tolerance)).any())
                ):
                    return FrontRESConstraintProjectionResult(
                        status="CONSTRAINT_RECOVERY", direction=recovery, active_families=tuple(active),
                        gradient_norms=norms,
                        directional_derivatives={family: float(dots[index].item()) for index, family in enumerate(active)},
                        intent_direction_norm=float(intent_direction.norm().item()),
                        projected_direction_norm=float(recovery.norm().item()),
                        dual_coefficients={family: 0.0 for family in active}, constraint_gram=gram_rows,
                        intent_directional_derivatives=intent_dots,
                        kkt_max_violation=float(torch.relu(dots).max().item()),
                        constraint_gradient_vectors={key: value.detach() for key, value in constraint_gradients.items()},
                    )
    return FrontRESConstraintProjectionResult(
        status="NO_COMMON_FIRST_ORDER_DESCENT", direction=zero, active_families=tuple(active),
        gradient_norms=norms, directional_derivatives={family: 0.0 for family in active},
        intent_direction_norm=float(intent_direction.norm().item()), projected_direction_norm=0.0,
        dual_coefficients={family: 0.0 for family in active}, constraint_gram=gram_rows,
        intent_directional_derivatives=intent_dots, kkt_max_violation=0.0,
        constraint_gradient_vectors={key: value.detach() for key, value in constraint_gradients.items()},
    )


def project_frontres_v004_actual_parameter_delta(
    candidate_delta: torch.Tensor,
    gradient_projection: FrontRESConstraintProjectionResult,
    *,
    actor_loss_weight: float,
    eps_grad: float = 1.0e-10,
    tolerance: float = 1.0e-8,
) -> FrontRESConstraintProjectionResult:
    """Commit one Adam candidate delta inside the already accepted Physics cone.

    This is the post-optimizer authority boundary. It never changes the PPO
    surrogate or constraint gradients; it only prevents Adam momentum and
    coordinate-wise preconditioning from bypassing the v004 halfspaces.
    """

    if candidate_delta.ndim != 1 or not bool(torch.isfinite(candidate_delta).all()):
        raise ValueError("FRS-PPO-v004 requires one finite flattened optimizer candidate delta")
    if not 0.0 <= float(actor_loss_weight) <= 1.0:
        raise ValueError("FRS-PPO-v004 actual update requires actor_loss_weight in [0,1]")
    gradients = gradient_projection.constraint_gradient_vectors
    zero = torch.zeros_like(candidate_delta)
    candidate_norm = float(candidate_delta.norm().item())
    frozen = float(actor_loss_weight) == 0.0 or gradient_projection.status in {
        "NO_EMPIRICAL_DIRECTION",
        "NO_COMMON_FIRST_ORDER_DESCENT",
    }
    if frozen:
        return FrontRESConstraintProjectionResult(
            status=gradient_projection.status,
            direction=zero,
            active_families=gradient_projection.active_families,
            gradient_norms=gradient_projection.gradient_norms,
            directional_derivatives={family: 0.0 for family in gradient_projection.active_families},
            intent_direction_norm=float(candidate_delta.norm().item()),
            projected_direction_norm=0.0,
            dual_coefficients=gradient_projection.dual_coefficients,
            constraint_gram=gradient_projection.constraint_gram,
            intent_directional_derivatives={
                family: float(torch.dot(gradients[family], candidate_delta).item())
                for family in gradient_projection.active_families
            },
            kkt_max_violation=0.0,
            constraint_gradient_vectors=gradients,
        )
    if gradient_projection.projected_direction_norm > float(eps_grad) and candidate_norm <= float(eps_grad):
        raise RuntimeError("FRS-PPO-v004 optimizer erased a permitted nonzero Actor direction")
    active = gradient_projection.active_families
    if not active:
        return FrontRESConstraintProjectionResult(
            status=gradient_projection.status,
            direction=candidate_delta.clone(),
            active_families=(),
            gradient_norms=gradient_projection.gradient_norms,
            directional_derivatives={},
            intent_direction_norm=float(candidate_delta.norm().item()),
            projected_direction_norm=float(candidate_delta.norm().item()),
            dual_coefficients={}, constraint_gram=(), intent_directional_derivatives={},
            kkt_max_violation=0.0, constraint_gradient_vectors=gradients,
        )
    matrix = torch.stack(tuple(gradients[family] for family in active), dim=0)
    if matrix.shape[1] != candidate_delta.numel() or not bool(torch.isfinite(matrix).all()):
        raise ValueError("FRS-PPO-v004 actual update received misaligned constraint gradients")

    best: torch.Tensor | None = None
    best_distance = float("inf")
    count = int(matrix.shape[0])
    for size in range(count + 1):
        for subset in itertools.combinations(range(count), size):
            if not subset:
                candidate = candidate_delta
            else:
                index = torch.tensor(subset, device=matrix.device, dtype=torch.long)
                selected = matrix.index_select(0, index)
                lambdas = torch.linalg.pinv(selected @ selected.T) @ (selected @ candidate_delta)
                if bool((lambdas < -float(tolerance)).any()):
                    continue
                candidate = candidate_delta - selected.T @ lambdas.clamp_min(0.0)
            if not bool(torch.isfinite(candidate).all()) or bool((matrix @ candidate > float(tolerance)).any()):
                continue
            distance = float((candidate - candidate_delta).square().sum().item())
            if distance < best_distance:
                best, best_distance = candidate.clone(), distance
    if best is None:
        raise RuntimeError("FRS-PPO-v004 could not project the actual optimizer delta")
    dots = matrix @ best
    # The source gradient already proved strict Physics descent. Adam's much
    # smaller parameter-space step only has to remain inside every active
    # halfspace; a nonzero tangent step is a legal constrained Intent update.
    if float(best.norm().item()) <= float(eps_grad) and candidate_norm > float(eps_grad):
        accepted = gradient_projection.direction
        accepted_norm = float(accepted.norm().item())
        if accepted_norm > float(eps_grad):
            best = accepted * (candidate_norm / accepted_norm)
            dots = matrix @ best
    if float(best.norm().item()) <= float(eps_grad):
        raise RuntimeError("FRS-PPO-v004 actual optimizer delta lost the permitted Actor update")
    kkt = float(torch.relu(dots).max().item())
    if kkt > float(tolerance):
        raise RuntimeError(
            "FRS-PPO-v004 actual optimizer delta violates a Physics halfspace: "
            f"kkt={kkt:.9g} tolerance={float(tolerance):.9g}"
        )
    return FrontRESConstraintProjectionResult(
        status=gradient_projection.status,
        direction=best,
        active_families=active,
        gradient_norms=gradient_projection.gradient_norms,
        directional_derivatives={family: float(dots[index].item()) for index, family in enumerate(active)},
        intent_direction_norm=float(candidate_delta.norm().item()),
        projected_direction_norm=float(best.norm().item()),
        dual_coefficients=gradient_projection.dual_coefficients,
        constraint_gram=gradient_projection.constraint_gram,
        intent_directional_derivatives={
            family: float(torch.dot(gradients[family], candidate_delta).item()) for family in active
        },
        kkt_max_violation=kkt,
        constraint_gradient_vectors=gradients,
    )


def step_frontres_v004_optimizer_with_actor_authority(
    optimizer: Any,
    actor_parameters: tuple[torch.Tensor, ...],
    parameter_snapshots: dict[int, torch.Tensor],
    gradient_projection: FrontRESConstraintProjectionResult,
    *,
    actor_loss_weight: float,
    eps_grad: float = 1.0e-10,
    tolerance: float = 1.0e-8,
) -> FrontRESActualOptimizerCommitResult:
    """Run exactly one shared step, then govern the actual Actor/std increment.

    Critic ownership remains with the caller's installed gradients. This owner
    only prevents optimizer momentum/preconditioning from bypassing the
    accepted PPO-v004 Actor direction.
    """

    if not actor_parameters:
        raise RuntimeError("FRS-PPO-v004 actual update requires Actor/std parameters")
    if any(id(parameter) not in parameter_snapshots for parameter in actor_parameters):
        raise RuntimeError("FRS-PPO-v004 actual update requires pre-step Actor snapshots")
    state = getattr(optimizer, "state", None)
    step = getattr(optimizer, "step", None)
    if not isinstance(state, Mapping) or not callable(step):
        raise TypeError("FRS-PPO-v004 optimizer must expose parameter state and step()")

    must_preserve = float(actor_loss_weight) == 0.0 or gradient_projection.status in {
        "NO_EMPIRICAL_DIRECTION",
        "NO_COMMON_FIRST_ORDER_DESCENT",
    }
    state_snapshots = (
        {
            id(parameter): (parameter in state, copy.deepcopy(state.get(parameter, {})))
            for parameter in actor_parameters
        }
        if must_preserve
        else None
    )

    step()
    candidate_delta = torch.cat(
        tuple(
            (parameter.detach() - parameter_snapshots[id(parameter)]).reshape(-1)
            for parameter in actor_parameters
        ),
        dim=0,
    )
    actual_projection = project_frontres_v004_actual_parameter_delta(
        candidate_delta,
        gradient_projection,
        actor_loss_weight=actor_loss_weight,
        eps_grad=eps_grad,
        tolerance=tolerance,
    )

    if must_preserve:
        if state_snapshots is None:
            raise RuntimeError("FRS-PPO-v004 Actor optimizer-state snapshot is missing")
        for parameter in actor_parameters:
            parameter.data.copy_(parameter_snapshots[id(parameter)])
            existed, before = state_snapshots[id(parameter)]
            if existed:
                state[parameter] = copy.deepcopy(before)
            else:
                state.pop(parameter, None)
    else:
        offset = 0
        for parameter in actor_parameters:
            count = parameter.numel()
            parameter.data.copy_(
                parameter_snapshots[id(parameter)]
                + actual_projection.direction[offset : offset + count].reshape_as(parameter)
            )
            offset += count
        if offset != int(actual_projection.direction.numel()):
            raise RuntimeError("FRS-PPO-v004 actual Actor delta does not match parameter layout")

    return FrontRESActualOptimizerCommitResult(
        projection=actual_projection,
        optimizer_candidate_actor_delta_l2=float(candidate_delta.norm().detach().cpu().item()),
        committed_actor_delta_l2=float(actual_projection.direction.norm().detach().cpu().item()),
        actor_optimizer_state_preserved=must_preserve,
    )


def install_frontres_v004_projected_gradients(
    policy: Any,
    result: FrontRESSegmentPPOResult,
    cfg: FrontRESSegmentPPOConfig,
    optimizer_parameters: tuple[torch.Tensor, ...],
) -> FrontRESConstraintProjectionResult:
    """Install disjoint scalar-Critic and projected actor gradients before one step."""

    critic = getattr(policy, "critic", None)
    if not isinstance(critic, torch.nn.Module):
        raise RuntimeError("FRS-PPO-v004 requires one explicit scalar Critic module")
    critic_ids = {id(parameter) for parameter in critic.parameters()}
    actor_parameters = tuple(parameter for parameter in optimizer_parameters if id(parameter) not in critic_ids)
    critic_parameters = tuple(parameter for parameter in optimizer_parameters if id(parameter) in critic_ids)
    if not actor_parameters or not critic_parameters:
        raise RuntimeError("FRS-PPO-v004 requires disjoint actor/std and Critic parameters")
    if result.constraint_surrogates is None or result.constraint_levels is None:
        raise RuntimeError("FRS-PPO-v004 requires grouped constraint surrogates")

    def gradients(loss: torch.Tensor, parameters: tuple[torch.Tensor, ...], *, retain_graph: bool) -> tuple[torch.Tensor, ...]:
        observed = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
        return tuple(torch.zeros_like(parameter) if gradient is None else gradient for parameter, gradient in zip(parameters, observed, strict=True))

    def flatten(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(tuple(value.reshape(-1) for value in values), dim=0)

    actor_loss = result.actor_loss - float(cfg.entropy_coef) * result.entropy
    intent_descent_gradient = gradients(actor_loss, actor_parameters, retain_graph=True)
    intent_direction = -flatten(intent_descent_gradient)
    constraint_gradients: dict[str, torch.Tensor] = {}
    for family in ("contact", "zmp", "survival"):
        surrogate = result.constraint_surrogates.get(family)
        if not isinstance(surrogate, torch.Tensor):
            raise RuntimeError(f"FRS-PPO-v004 missing {family} constraint surrogate")
        constraint_gradients[family] = flatten(gradients(surrogate, actor_parameters, retain_graph=True))
    projection = project_frontres_grouped_constraint_direction(
        intent_direction,
        constraint_gradients,
        result.constraint_levels,
        eps_grad=cfg.constraint_grad_epsilon,
        tolerance=cfg.projection_tolerance,
    )
    actor_direction = float(cfg.actor_loss_weight) * projection.direction
    actor_frozen = float(cfg.actor_loss_weight) == 0.0 or projection.status in {
        "NO_EMPIRICAL_DIRECTION",
        "NO_COMMON_FIRST_ORDER_DESCENT",
    }
    offset = 0
    for parameter in actor_parameters:
        count = parameter.numel()
        parameter.grad = (
            None
            if actor_frozen
            else (-actor_direction[offset : offset + count].reshape_as(parameter)).detach().clone()
        )
        offset += count
    critic_loss = float(cfg.value_loss_coef) * result.value_loss
    critic_gradients = gradients(critic_loss, critic_parameters, retain_graph=False)
    for parameter, gradient in zip(critic_parameters, critic_gradients, strict=True):
        parameter.grad = gradient.detach().clone()
    return projection
