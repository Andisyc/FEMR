"""Deterministic contracts for Q2-D scale and update causality."""

import importlib.util
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_policy_quality_q2d.py"
SPEC = importlib.util.spec_from_file_location("frontres_policy_quality_q2d_contract_module", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PPO_PATH = ROOT / "source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py"
PPO_SPEC = importlib.util.spec_from_file_location("frontres_policy_quality_q2d_ppo_module", PPO_PATH)
PPO = importlib.util.module_from_spec(PPO_SPEC)
assert PPO_SPEC.loader is not None
sys.modules[PPO_SPEC.name] = PPO
PPO_SPEC.loader.exec_module(PPO)


class _Adapter:
    checkpoint_identity = "frozen-hsl"
    observation_identity = "obs-v1"

    def action(self, observations):
        return torch.ones((observations.shape[0], 6)) * 0.2


class _SegmentPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(1, 6, bias=False)
        self.critic = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.critic.weight)

    def evaluate_segment_actions(self, observations, actions):
        mean = self.actor(observations)
        value = self.critic(observations).squeeze(-1)
        log_prob = -0.5 * (actions - mean).square().sum(-1)
        return PPO.FrontRESSegmentPolicyEval(
            log_prob=log_prob,
            value=value,
            entropy=torch.zeros_like(log_prob),
            mean=mean,
            sigma=torch.ones_like(mean),
            raw_actions=actions,
        )


def main() -> None:
    route = {"name": None, "actions": []}
    restores = []
    isolation = {"value": "unchanged"}
    audit_identities = []

    def restore():
        restores.append(1)
        route["actions"] = []
        return "same-state"

    results = MODULE.run_q2d_scale_sweep(
        base_adapter=_Adapter(),
        scales=MODULE.Q2D_SCALE_FACTORS,
        horizon_k=2,
        restore_state=restore,
        begin_route=lambda name: route.update(name=name),
        observe=lambda: torch.zeros((1, 8)),
        apply_action=lambda action: route["actions"].append(action.clone()),
        step=lambda: None,
        compute_gain=lambda: torch.tensor(float(torch.stack(route["actions"]).mean())),
        capture_execution=lambda: {"steps": len(route["actions"])},
        isolation_state=lambda: isolation["value"],
        set_audit_identity=lambda route_name, scale, state_hash: audit_identities.append(
            (route_name, scale, state_hash)
        ),
    )
    assert len(results) == 6 and len(restores) == 6
    assert torch.equal(results[0].actions, torch.zeros_like(results[0].actions))
    assert torch.allclose(results[4].actions, torch.full_like(results[4].actions, 0.2))
    assert results[0].initial_state_hash == results[-1].initial_state_hash
    assert len(audit_identities) == 6
    assert all(identity[2] == "same-state" for identity in audit_identities)

    raw = torch.tensor([[-1.0] * 6, [1.0] * 6])
    means = torch.zeros_like(raw)
    sigmas = torch.ones_like(raw)
    advantages = torch.tensor([2.0, -1.0])
    score = MODULE.gaussian_mean_score_gradient(raw, means, sigmas, advantages, torch.ones(2, dtype=torch.bool))
    assert bool((score < 0).all())

    credit_path = ROOT / "source/rsl_rl/rsl_rl/tests/.frontres_policy_quality_q2d_credit.json"
    payload = MODULE.write_q2d_credit_tuple(
        result_path=str(credit_path),
        raw_actions=raw,
        bounded_actions=raw * 0.1,
        old_means=means,
        old_sigmas=sigmas,
        gains=torch.tensor([0.5, -0.25]),
        returns=torch.tensor([0.4, -0.2]),
        advantages=advantages,
        valid_mask=torch.ones(2, dtype=torch.bool),
        segment_ids=torch.tensor([11, 12]),
        audit_transaction_id="txn-1",
        audit_batch_signature="batch-1",
        audit_identity_state="complete",
    )
    assert payload["row_count"] == 2
    assert json.loads(credit_path.read_text())["audit_transaction_id"] == "txn-1"
    credit_path.unlink()
    try:
        MODULE.write_q2d_credit_tuple(
            result_path=str(credit_path), raw_actions=raw, bounded_actions=raw,
            old_means=means, old_sigmas=sigmas, gains=torch.ones(2), returns=torch.ones(2),
            advantages=advantages, valid_mask=torch.ones(2, dtype=torch.bool),
            segment_ids=torch.arange(2), audit_transaction_id=None,
            audit_batch_signature=None, audit_identity_state="UNCONFIRMED",
        )
        raise AssertionError("incomplete transaction identity must fail closed")
    except ValueError as exc:
        assert "complete rollout transaction identity" in str(exc)

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    obs = torch.ones((1, 1))

    def update(clone):
        optimizer = torch.optim.SGD(clone.parameters(), lr=0.1)
        optimizer.zero_grad()
        clone(obs).square().mean().backward()
        optimizer.step()

    controlled = MODULE.run_isolated_controlled_update(
        model=model,
        observations=obs,
        mean_fn=lambda active, value: active(value),
        update_fn=update,
        current_action=torch.ones((1, 1)),
        preferred_action=torch.zeros((1, 1)),
    )
    assert controlled["source_model_unchanged"]
    assert controlled["direction"]["moves_toward_preferred"]
    assert torch.equal(model.weight.detach(), torch.ones_like(model.weight))

    policy = _SegmentPolicy()
    sampled_action = torch.full((1, 6), -0.1)
    ppo_batch = PPO.FrontRESSegmentPPOBatch(
        observations=torch.ones((1, 1)),
        actions=sampled_action,
        old_log_probs=-0.5 * sampled_action.square().sum(-1),
        old_values=torch.zeros(1),
        returns=torch.zeros(1),
        advantages=torch.ones(1),
        valid_mask=torch.ones(1, dtype=torch.bool),
        old_means=torch.zeros((1, 6)),
        old_sigmas=torch.ones((1, 6)),
    )

    def canonical_ppo_update(clone):
        optimizer = torch.optim.SGD(clone.parameters(), lr=0.1)
        optimizer.zero_grad()
        result = PPO.compute_frontres_segment_ppo_loss(
            clone,
            ppo_batch,
            PPO.FrontRESSegmentPPOConfig(value_loss_coef=0.0),
        )
        result.total_loss.backward()
        optimizer.step()

    ppo_controlled = MODULE.run_isolated_controlled_update(
        model=policy,
        observations=ppo_batch.observations,
        mean_fn=lambda active, value: active.evaluate_segment_actions(value, sampled_action).mean,
        update_fn=canonical_ppo_update,
        current_action=torch.zeros((1, 6)),
        preferred_action=sampled_action,
    )
    assert ppo_controlled["direction"]["moves_toward_preferred"]
    assert torch.equal(policy.actor.weight.detach(), torch.zeros_like(policy.actor.weight))
    print("PASS: Q2-D scale sweep and controlled mean-direction contracts are closed offline.")


if __name__ == "__main__":
    main()
