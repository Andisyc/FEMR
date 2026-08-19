"""S1 gradient and optimizer alignment for the FRS-PPO-v014 candidate.

The pseudo policy keeps one scalar mean weight and one scalar log standard
deviation.  This makes the Gaussian chain rule, global norm clipping, and the
first Adam update independently hand-computable while still invoking the
production preference-loss public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESRelationalPPOBatch,
    compute_frontres_relational_preference_loss,
)


DTYPE = torch.float64
MAX_GRAD_NORM = 0.5
ADAM_EPS = 1.0e-8


class _ScalarGaussianPolicy:
    def __init__(self, *, sigma: float) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.log_std = torch.nn.Parameter(torch.tensor(math.log(sigma), dtype=DTYPE))

    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return (self.weight, self.log_std)

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        mean = self.weight * observations[:, 0]
        sigma = torch.exp(self.log_std)
        log_prob = -0.5 * ((actions[:, 0] - mean) / sigma).square() - self.log_std
        return {
            "log_prob": log_prob,
            "value": torch.zeros_like(log_prob),
            "entropy": None,
        }


@dataclass(frozen=True)
class _Oracle:
    loss: float
    weight_gradient: float
    log_std_gradient: float


@dataclass(frozen=True)
class _Observation:
    loss: float
    weight_gradient: float
    log_std_gradient: float
    gradient_norm: float


@dataclass(frozen=True)
class _OptimizerObservation:
    pre_clip_norm: float
    post_clip_norm: float
    parameter_delta: tuple[float, float]
    parameter_delta_l2: float


def _batch(*, input_scale: float, winner_action: float, loser_action: float) -> FrontRESRelationalPPOBatch:
    return FrontRESRelationalPPOBatch(
        observations=torch.tensor([[input_scale], [input_scale]], dtype=DTYPE),
        actions=torch.tensor(
            [
                [winner_action, 0.0, 0.0, 0.0, 0.0, 0.0],
                [loser_action, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=DTYPE,
        ),
        old_log_probs=torch.zeros(2, dtype=DTYPE),
        valid_mask=torch.ones(2, dtype=torch.bool),
    )


def _independent_oracle(
    *,
    sigma: float,
    input_scale: float,
    winner_action: float,
    loser_action: float,
) -> _Oracle:
    """Closed-form answer at weight=0; no production helper is used."""

    winner_residual = winner_action
    loser_residual = loser_action
    variance = sigma * sigma
    margin = -0.5 * (
        winner_residual * winner_residual - loser_residual * loser_residual
    ) / variance
    loss = math.log1p(math.exp(-margin))
    loss_margin_gradient = -1.0 / (1.0 + math.exp(margin))
    margin_mean_gradient = (winner_residual - loser_residual) / variance
    margin_log_std_gradient = (
        winner_residual * winner_residual - loser_residual * loser_residual
    ) / variance
    return _Oracle(
        loss=loss,
        weight_gradient=loss_margin_gradient * margin_mean_gradient * input_scale,
        log_std_gradient=loss_margin_gradient * margin_log_std_gradient,
    )


def _evaluate(
    *,
    sigma: float,
    input_scale: float,
    winner_action: float,
    loser_action: float,
) -> tuple[_ScalarGaussianPolicy, _Observation]:
    policy = _ScalarGaussianPolicy(sigma=sigma)
    result = compute_frontres_relational_preference_loss(
        policy,
        _batch(
            input_scale=input_scale,
            winner_action=winner_action,
            loser_action=loser_action,
        ),
        ((0, 1),),
    )
    assert result.contract_id == "FRS-PPO-v014"
    assert result.status == "READY"
    assert result.edge_count == 1
    result.total_loss.backward()
    weight_gradient = float(policy.weight.grad)
    log_std_gradient = float(policy.log_std.grad)
    return policy, _Observation(
        loss=float(result.total_loss.detach()),
        weight_gradient=weight_gradient,
        log_std_gradient=log_std_gradient,
        gradient_norm=math.hypot(weight_gradient, log_std_gradient),
    )


def _assert_matches_oracle(
    *,
    sigma: float,
    input_scale: float,
    winner_action: float,
    loser_action: float,
) -> _Observation:
    _policy, observed = _evaluate(
        sigma=sigma,
        input_scale=input_scale,
        winner_action=winner_action,
        loser_action=loser_action,
    )
    expected = _independent_oracle(
        sigma=sigma,
        input_scale=input_scale,
        winner_action=winner_action,
        loser_action=loser_action,
    )
    assert math.isclose(observed.loss, expected.loss, rel_tol=1.0e-10, abs_tol=1.0e-10)
    assert math.isclose(
        observed.weight_gradient,
        expected.weight_gradient,
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    )
    assert math.isclose(
        observed.log_std_gradient,
        expected.log_std_gradient,
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    )
    return observed


def _one_adam_step(*, learning_rate: float, sigma: float, action_gap: float) -> _OptimizerObservation:
    policy = _ScalarGaussianPolicy(sigma=sigma)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=ADAM_EPS,
    )
    optimizer.zero_grad()
    result = compute_frontres_relational_preference_loss(
        policy,
        _batch(input_scale=1.0, winner_action=action_gap, loser_action=0.0),
        ((0, 1),),
    )
    result.total_loss.backward()
    parameters = policy.parameters()
    pre_clip_gradients = tuple(float(parameter.grad) for parameter in parameters)
    pre_clip_norm = math.sqrt(sum(value * value for value in pre_clip_gradients))
    torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
    clipped_gradients = tuple(float(parameter.grad) for parameter in parameters)
    post_clip_norm = math.sqrt(sum(value * value for value in clipped_gradients))
    before = tuple(float(parameter.detach()) for parameter in parameters)
    optimizer.step()
    after = tuple(float(parameter.detach()) for parameter in parameters)
    parameter_delta = tuple(end - start for start, end in zip(before, after))

    # On Adam's first step, bias correction reduces m_hat to g and v_hat to
    # g^2.  This is not the SGD expression ``-lr * g``.
    expected_delta = tuple(
        -learning_rate * gradient / (abs(gradient) + ADAM_EPS)
        for gradient in clipped_gradients
    )
    for observed, expected in zip(parameter_delta, expected_delta):
        assert math.isclose(observed, expected, rel_tol=1.0e-8, abs_tol=1.0e-12)
    assert pre_clip_norm > MAX_GRAD_NORM
    assert math.isclose(post_clip_norm, MAX_GRAD_NORM, rel_tol=5.0e-6, abs_tol=5.0e-7)
    return _OptimizerObservation(
        pre_clip_norm=pre_clip_norm,
        post_clip_norm=post_clip_norm,
        parameter_delta=parameter_delta,
        parameter_delta_l2=math.sqrt(sum(value * value for value in parameter_delta)),
    )


def main() -> None:
    # C1/T-formula+gradient: ordinary asymmetric Gaussian preference.
    ordinary = _assert_matches_oracle(
        sigma=0.15,
        input_scale=1.0,
        winner_action=0.145,
        loser_action=0.0,
    )

    # C2/T-boundary: identical actions have log(2) loss but no policy gradient.
    equal = _assert_matches_oracle(
        sigma=0.15,
        input_scale=1.0,
        winner_action=0.145,
        loser_action=0.145,
    )
    assert math.isclose(equal.loss, math.log(2.0), rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert equal.gradient_norm == 0.0

    # C3/T-scale: with fixed sigma, a preferred action farther from the current
    # mean increases the Gaussian chain-rule gradient.
    action_sweep = [
        _assert_matches_oracle(
            sigma=0.10,
            input_scale=1.0,
            winner_action=action_gap,
            loser_action=0.0,
        )
        for action_gap in (0.05, 0.145, 0.286)
    ]
    assert action_sweep[0].gradient_norm < action_sweep[1].gradient_norm < action_sweep[2].gradient_norm

    # C3/T-scale: the same action evidence becomes sharply conditioned as
    # sigma shrinks because Gaussian derivatives contain 1/sigma^2.
    sigma_sweep = [
        _assert_matches_oracle(
            sigma=sigma,
            input_scale=1.0,
            winner_action=0.145,
            loser_action=0.0,
        )
        for sigma in (0.30, 0.15, 0.075)
    ]
    assert sigma_sweep[0].gradient_norm < sigma_sweep[1].gradient_norm < sigma_sweep[2].gradient_norm

    # C4/T-identity: the mean-network Jacobian scales only the weight gradient;
    # it does not change the distribution-level log-std gradient.
    input_one = _assert_matches_oracle(
        sigma=0.15,
        input_scale=1.0,
        winner_action=0.145,
        loser_action=0.0,
    )
    input_eight = _assert_matches_oracle(
        sigma=0.15,
        input_scale=8.0,
        winner_action=0.145,
        loser_action=0.0,
    )
    assert math.isclose(
        input_eight.weight_gradient,
        8.0 * input_one.weight_gradient,
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    )
    assert math.isclose(
        input_eight.log_std_gradient,
        input_one.log_std_gradient,
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    )

    # Sensitivity: a mutant that mistakes the bounded dL/dmargin for the full
    # parameter gradient omits the Gaussian and network Jacobians.
    adverse = sigma_sweep[-1]
    bounded_margin_only_mutant = 1.0
    assert abs(adverse.weight_gradient) > 10.0 * bounded_margin_only_mutant

    # C5/T-update: exercise the configured clip and two Actor learning rates.
    step_low = _one_adam_step(learning_rate=3.0e-7, sigma=0.05, action_gap=0.286)
    step_high = _one_adam_step(learning_rate=1.0e-6, sigma=0.05, action_gap=0.286)
    assert math.isclose(step_low.pre_clip_norm, step_high.pre_clip_norm, rel_tol=1.0e-12)
    assert math.isclose(step_low.post_clip_norm, step_high.post_clip_norm, rel_tol=1.0e-12)
    assert math.isclose(
        step_high.parameter_delta_l2 / step_low.parameter_delta_l2,
        1.0e-6 / 3.0e-7,
        rel_tol=1.0e-7,
    )

    # Sensitivity: an SGD mutant would multiply the clipped gradient directly
    # by lr.  Adam's first step normalizes each nonzero coordinate instead.
    sgd_mutant_delta = MAX_GRAD_NORM * 1.0e-6
    assert not math.isclose(
        step_high.parameter_delta_l2,
        sgd_mutant_delta,
        rel_tol=1.0e-3,
        abs_tol=1.0e-12,
    )

    print("frontres_relational_preference_optimizer_alignment: MODULE-CORRECT")
    print(
        "action_gap_gradient_norms="
        + ",".join(f"{value.gradient_norm:.6f}" for value in action_sweep)
    )
    print(
        "sigma_gradient_norms="
        + ",".join(f"{value.gradient_norm:.6f}" for value in sigma_sweep)
    )
    print(
        f"optimizer_pre_clip_norm={step_high.pre_clip_norm:.6f} "
        f"post_clip_norm={step_high.post_clip_norm:.6f}"
    )
    print(
        f"adam_delta_l2_lr_3e-7={step_low.parameter_delta_l2:.12f} "
        f"adam_delta_l2_lr_1e-6={step_high.parameter_delta_l2:.12f}"
    )


if __name__ == "__main__":
    main()
