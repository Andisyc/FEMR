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
require_frontres_v011_campaign_schedule = WARMUP_MODULE.require_frontres_v011_campaign_schedule
require_frontres_v013_campaign_schedule = WARMUP_MODULE.require_frontres_v013_campaign_schedule
sample_frontres_v013_dr_strength = WARMUP_MODULE.sample_frontres_v013_dr_strength


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
        "actor_ramp",
        "actor_ramp",
        "actor_ramp",
        "actor_ramp",
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


def test_actor_ramp_releases_actor_gradient() -> None:
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


def test_v011_k_m_stage_boundaries_and_repeated_critic_only() -> None:
    schedule = parse_frontres_k_stage_schedule("8:2:2:2:2,16:3:3:2:1,32:4:1:1:0", max_horizon_k=32)
    identities = [
        resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=iteration)
        for iteration in range(14)
    ]
    assert [(value.stage_index, value.active_k, value.active_m, value.stage_iteration) for value in identities[:7]] == [
        (0, 8, 2, 0), (0, 8, 2, 1), (0, 8, 2, 2), (0, 8, 2, 3),
        (0, 8, 2, 4), (0, 8, 2, 5), (1, 16, 3, 0)
    ]
    assert identities[0].phase.name == "critic_only"
    assert identities[6].phase.name == "critic_only"
    assert identities[7].phase.name == "critic_only"
    assert identities[9].phase.name == "actor_ramp"
    assert identities[11].phase.name == "joint"
    assert identities[12].stage_index == 2
    assert identities[12].active_k == 32
    assert identities[12].active_m == 4
    assert identities[12].phase.name == "critic_only"
    assert identities[13].phase.name == "actor_ramp"
    assert len({value.schedule_fingerprint for value in identities}) == 1


def test_v011_schedule_is_deterministic_and_fail_closed() -> None:
    first = parse_frontres_k_stage_schedule("8:2:2:3:4,16:3:5:6:0")
    second = parse_frontres_k_stage_schedule("8:2:2:3:4,16:3:5:6:0")
    assert first == second
    assert resolve_frontres_k_stage_identity(schedule=first, committed_update_iteration=4).schedule_fingerprint == (
        resolve_frontres_k_stage_identity(schedule=second, committed_update_iteration=4).schedule_fingerprint
    )
    invalid = (
        "",
        "8:2:3:4",
        "8:1:2:3:4",
        "8:2:0:3:4",
        "8:2:2:0:4",
        "8:2:2:3:0,16:3:2:3:0",
        "16:2:2:3:4,8:3:2:3:0",
        "8:3:2:3:4,16:2:2:3:0",
    )
    for value in invalid:
        try:
            parse_frontres_k_stage_schedule(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid v011 schedule to reject: {value!r}")
    for schedule in (((8.5, 2, 2, 3, 0),), ((8, True, 2, 3, 0),)):
        try:
            WARMUP_MODULE.normalize_frontres_k_stage_schedule(schedule)
        except ValueError:
            pass
        else:
            raise AssertionError("expected non-integer/bool v011 schedule field to reject")
    try:
        parse_frontres_k_stage_schedule("8:2:2:3:4,64:3:2:3:0", max_horizon_k=32)
    except ValueError:
        pass
    else:
        raise AssertionError("expected schedule horizon beyond max_horizon_k to reject")


def test_v011_campaign_schedule_is_exact_and_checkpoint_bounded() -> None:
    schedule = parse_frontres_k_stage_schedule(
        "8:2:200:500:1300,16:3:300:300:900,32:4:400:300:625"
    )
    assert require_frontres_v011_campaign_schedule(schedule) == schedule
    assert WARMUP_MODULE.FRONTRES_V011_SELECTED_SEGMENT_COUNT == 2
    assert WARMUP_MODULE.FRONTRES_V011_MAX_ABSOLUTE_ITERATION == 8000
    assert WARMUP_MODULE.FRONTRES_V011_REVIEW_BOUNDARIES == (2000, 3500, 4825, 6500, 8000)
    for bad in (
        "8:2:200:500:1300,16:2:300:300:900,32:4:400:300:625",
        "8:2:200:500:1300,16:3:300:300:900,32:4:400:300:624",
    ):
        try:
            require_frontres_v011_campaign_schedule(parse_frontres_k_stage_schedule(bad))
        except ValueError:
            pass
        else:
            raise AssertionError("TRAIN-v011 formal campaign drift must reject")


def _v013_schedule_text() -> str:
    return (
        "8:2:200:500:1300:lower-k8:0.50:linear-joint-v1:1300:2.381,"
        "16:3:300:300:900:lower-k16:0.60:linear-joint-v1:900:2.381,"
        "32:4:400:300:625:lower-k32:0.70:linear-joint-v1:625:2.381"
    )


def test_v013_nested_dr_restart_and_committed_progress() -> None:
    schedule = require_frontres_v013_campaign_schedule(
        parse_frontres_k_stage_schedule(_v013_schedule_text(), max_horizon_k=32)
    )
    at_k8_joint = resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=700)
    assert at_k8_joint.active_k == 8
    assert at_k8_joint.dr_progress == 0.0
    assert at_k8_joint.d_cap == 0.50
    later_k8 = resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=1350)
    assert abs(later_k8.dr_progress - 0.5) < 1.0e-12
    expected_half = 0.5 + 0.5 * (2.381 / 1.10 - 0.5)
    assert abs(later_k8.d_cap - expected_half) < 1.0e-12
    at_k16 = resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=2000)
    assert at_k16.active_k == 16 and at_k16.active_m == 3
    assert at_k16.phase.name == "critic_only"
    assert at_k16.dr_progress == 0.0 and at_k16.d_cap == 0.60
    assert at_k16.dr_stage_fingerprint != at_k8_joint.dr_stage_fingerprint


def test_v013_four_class_sampling_and_no_hidden_defaults() -> None:
    schedule = require_frontres_v013_campaign_schedule(parse_frontres_k_stage_schedule(_v013_schedule_text()))
    identity = resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=1350)
    counts = {"easy": 0, "medium": 0, "hard": 0, "broken": 0}
    for sample_key in range(10_000):
        sample = sample_frontres_v013_dr_strength(identity, sample_key=sample_key)
        counts[sample.class_name] += 1
        assert 0.0 <= sample.strength <= 2.381
        if sample.class_name == "easy":
            assert sample.strength < 0.25 * identity.d_cap
        elif sample.class_name == "medium":
            assert 0.25 * identity.d_cap <= sample.strength < 0.70 * identity.d_cap
        elif sample.class_name == "hard":
            assert 0.70 * identity.d_cap <= sample.strength <= identity.d_cap
        else:
            assert identity.d_cap < sample.strength <= min(1.10 * identity.d_cap, 2.381)
    fractions = {name: count / 10_000 for name, count in counts.items()}
    assert abs(fractions["easy"] - 0.20) < 0.02
    assert abs(fractions["medium"] - 0.30) < 0.02
    assert abs(fractions["hard"] - 0.40) < 0.02
    assert abs(fractions["broken"] - 0.10) < 0.02
    for old_or_incomplete in (
        "8:2:200:500:1300,16:3:300:300:900,32:4:400:300:625",
        "8:2:200:500:1300:lower-k8:0.5::1300:2.381",
    ):
        try:
            require_frontres_v013_campaign_schedule(parse_frontres_k_stage_schedule(old_or_incomplete))
        except ValueError:
            pass
        else:
            raise AssertionError("TRAIN-v013 must reject missing DR identity instead of using defaults")


def main() -> None:
    test_phase_boundaries_are_monotonic()
    test_critic_only_blocks_actor_and_std_gradients()
    test_actor_ramp_releases_actor_gradient()
    test_v011_k_m_stage_boundaries_and_repeated_critic_only()
    test_v011_schedule_is_deterministic_and_fail_closed()
    test_v011_campaign_schedule_is_exact_and_checkpoint_bounded()
    test_v013_nested_dr_restart_and_committed_progress()
    test_v013_four_class_sampling_and_no_hidden_defaults()
    print("frontres_segment_warmup_contract: ok")


if __name__ == "__main__":
    main()
