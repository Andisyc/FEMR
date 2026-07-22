from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.algorithms.frontres_segment_ppo import (  # noqa: E402
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    FrontRESSegmentPolicyEval,
    compute_frontres_segment_ppo_loss,
)
WARMUP_PATH = SOURCE_ROOT / "rsl_rl" / "frontres" / "frontres_segment_warmup.py"
WARMUP_SPEC = importlib.util.spec_from_file_location("frontres_segment_warmup_contract_module", WARMUP_PATH)
if WARMUP_SPEC is None or WARMUP_SPEC.loader is None:
    raise RuntimeError(f"Could not load Segment warmup owner from {WARMUP_PATH}")
WARMUP_MODULE = importlib.util.module_from_spec(WARMUP_SPEC)
sys.modules[WARMUP_SPEC.name] = WARMUP_MODULE
WARMUP_SPEC.loader.exec_module(WARMUP_MODULE)
frontres_segment_warmup_phase = WARMUP_MODULE.frontres_segment_warmup_phase
parse_frontres_k_stage_schedule = WARMUP_MODULE.parse_frontres_k_stage_schedule
resolve_frontres_k_stage_identity = WARMUP_MODULE.resolve_frontres_k_stage_identity


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, 6, bias=False)
        self.critic = torch.nn.Linear(2, 1, bias=False)
        self.log_std = torch.nn.Parameter(torch.full((6,), -0.4))
        torch.nn.init.constant_(self.actor.weight, 0.2)
        torch.nn.init.constant_(self.critic.weight, 0.1)

    def evaluate_segment_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> FrontRESSegmentPolicyEval:
        mean = self.actor(observations)
        sigma = self.log_std.exp().expand_as(mean)
        distribution = torch.distributions.Normal(mean, sigma)
        return FrontRESSegmentPolicyEval(
            log_prob=distribution.log_prob(actions).sum(dim=-1),
            value=self.critic(observations).squeeze(-1),
            entropy=distribution.entropy().sum(dim=-1),
            mean=mean,
            sigma=sigma,
            raw_actions=actions,
        )


def _batch(policy: _ToyPolicy) -> FrontRESSegmentPPOBatch:
    obs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    actions = torch.tensor(
        [
            [0.5, -0.1, 0.0, 0.2, -0.2, 0.1],
            [-0.2, 0.1, 0.3, -0.1, 0.0, 0.2],
            [0.3, -0.2, 0.1, 0.0, 0.2, -0.1],
        ]
    )
    with torch.no_grad():
        evaluation = policy.evaluate_segment_actions(obs, actions)
    return FrontRESSegmentPPOBatch(
        observations=obs,
        actions=actions,
        old_log_probs=evaluation.log_prob.detach(),
        old_values=evaluation.value.detach(),
        returns=torch.tensor([0.8, -0.3, 0.5]),
        advantages=torch.tensor([0.7, -0.4, 0.2]),
        valid_mask=torch.ones(3, dtype=torch.bool),
        old_means=evaluation.mean.detach(),
        old_sigmas=evaluation.sigma.detach(),
    )


def _grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().item())


def test_phase_boundaries_are_monotonic() -> None:
    phases = [
        frontres_segment_warmup_phase(
            iteration=iteration,
            critic_warmup_iterations=2,
            actor_warmup_iterations=4,
        )
        for iteration in range(8)
    ]
    assert [phase.name for phase in phases] == [
        "critic_only",
        "critic_only",
        "actor_warmup",
        "actor_warmup",
        "actor_warmup",
        "actor_warmup",
        "joint",
        "joint",
    ]
    weights = [phase.actor_loss_weight for phase in phases]
    assert weights[:2] == [0.0, 0.0]
    assert weights[2:6] == [0.25, 0.5, 0.75, 1.0]
    assert weights[6:] == [1.0, 1.0]
    assert weights == sorted(weights)


def test_critic_only_blocks_actor_and_std_gradients() -> None:
    policy = _ToyPolicy()
    result = compute_frontres_segment_ppo_loss(
        policy,
        _batch(policy),
        FrontRESSegmentPPOConfig(actor_loss_weight=0.0, entropy_coef=0.01),
    )
    result.total_loss.backward()
    assert result.actor_loss_weight == 0.0
    assert _grad_norm(policy.actor.weight) == 0.0
    assert _grad_norm(policy.log_std) == 0.0
    assert _grad_norm(policy.critic.weight) > 0.0


def test_actor_warmup_releases_actor_gradient() -> None:
    policy = _ToyPolicy()
    result = compute_frontres_segment_ppo_loss(
        policy,
        _batch(policy),
        FrontRESSegmentPPOConfig(actor_loss_weight=0.5, entropy_coef=0.01),
    )
    result.total_loss.backward()
    assert result.actor_loss_weight == 0.5
    assert _grad_norm(policy.actor.weight) > 0.0
    assert _grad_norm(policy.log_std) > 0.0
    assert _grad_norm(policy.critic.weight) > 0.0


def test_v009_k_stage_boundaries_and_repeated_critic_only() -> None:
    schedule = parse_frontres_k_stage_schedule("8:2:2:2,16:3:2:1,32:1:1:0", max_horizon_k=32)
    identities = [
        resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=iteration)
        for iteration in range(14)
    ]
    assert [(value.stage_index, value.active_k, value.stage_iteration) for value in identities[:7]] == [
        (0, 8, 0), (0, 8, 1), (0, 8, 2), (0, 8, 3), (0, 8, 4), (0, 8, 5), (1, 16, 0)
    ]
    assert identities[0].phase.name == "critic_only"
    assert identities[6].phase.name == "critic_only"
    assert identities[7].phase.name == "critic_only"
    assert identities[9].phase.name == "actor_warmup"
    assert identities[11].phase.name == "joint"
    assert identities[12].stage_index == 2
    assert identities[12].active_k == 32
    assert identities[12].phase.name == "critic_only"
    assert identities[13].phase.name == "actor_warmup"
    assert len({value.schedule_fingerprint for value in identities}) == 1


def test_v009_schedule_is_deterministic_and_fail_closed() -> None:
    first = parse_frontres_k_stage_schedule("8:2:3:4,16:5:6:0")
    second = parse_frontres_k_stage_schedule("8:2:3:4,16:5:6:0")
    assert first == second
    assert resolve_frontres_k_stage_identity(schedule=first, committed_update_iteration=4).schedule_fingerprint == (
        resolve_frontres_k_stage_identity(schedule=second, committed_update_iteration=4).schedule_fingerprint
    )
    invalid = (
        "",
        "8:2:3",
        "8:0:3:4",
        "8:2:0:4",
        "8:2:3:0,16:2:3:0",
        "16:2:3:4,8:2:3:0",
    )
    for value in invalid:
        try:
            parse_frontres_k_stage_schedule(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid v009 schedule to reject: {value!r}")
    for schedule in (((8.5, 2, 3, 0),), ((8, True, 3, 0),)):
        try:
            WARMUP_MODULE.normalize_frontres_k_stage_schedule(schedule)
        except ValueError:
            pass
        else:
            raise AssertionError("expected non-integer/bool v009 schedule field to reject")
    try:
        parse_frontres_k_stage_schedule("8:2:3:4,64:2:3:0", max_horizon_k=32)
    except ValueError:
        pass
    else:
        raise AssertionError("expected schedule horizon beyond max_horizon_k to reject")


def main() -> None:
    test_phase_boundaries_are_monotonic()
    test_critic_only_blocks_actor_and_std_gradients()
    test_actor_warmup_releases_actor_gradient()
    test_v009_k_stage_boundaries_and_repeated_critic_only()
    test_v009_schedule_is_deterministic_and_fail_closed()
    print("frontres_segment_warmup_contract: ok")


if __name__ == "__main__":
    main()
