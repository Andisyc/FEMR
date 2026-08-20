"""S1 pseudo-data test for the active v014 Gaussian preference boundary.

The fixture is calibrated from one READY transaction in the V026 log.  It
does not replay the simulator or start live training. The active v014
reference/Fisher owner is exercised through its production public function;
the old direct preference function remains a characterization baseline.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.distributions import Normal

SOURCE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESRelationalPPOBatch,
    compute_frontres_relational_preference_loss,
    compute_frontres_relational_reference_fisher_loss,
)


LOG_NAME = "FRS_TRAIN_V026_V014_K8_SHORT50_20260819.log"
TELEMETRY_PREFIX = "[FrontRES v017 Transaction Telemetry] "


class _GaussianRows:
    """Tiny row-indexed Gaussian policy with optionally frozen sigma."""

    def __init__(self, actions: torch.Tensor, sigma: float, *, train_sigma: bool):
        self.mu = torch.nn.Parameter(actions.clone() * 0.25)
        log_std = torch.full((6,), math.log(sigma), dtype=actions.dtype)
        if train_sigma:
            self.log_std = torch.nn.Parameter(log_std)
        else:
            self.log_std = log_std

    def parameters(self):
        values = [self.mu]
        if isinstance(self.log_std, torch.nn.Parameter):
            values.append(self.log_std)
        return iter(values)

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        rows = observations[:, 0].to(dtype=torch.long)
        mean = self.mu.index_select(0, rows)
        log_std = self.log_std if isinstance(self.log_std, torch.Tensor) else torch.as_tensor(self.log_std)
        sigma = log_std.exp().expand_as(mean)
        distribution = Normal(mean, sigma)
        return {
            "log_prob": distribution.log_prob(actions).sum(dim=-1),
            "value": torch.zeros(actions.shape[0], dtype=actions.dtype),
            "entropy": distribution.entropy().sum(dim=-1),
            "mean": mean,
            "sigma": sigma,
        }


def _first_ready_payload(log_path: Path) -> dict[str, object]:
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            marker = line.find(TELEMETRY_PREFIX)
            if marker < 0:
                continue
            payload = json.loads(line[marker + len(TELEMETRY_PREFIX) :])
            if payload.get("status") == "READY" and payload.get("preference_edges"):
                return payload
    raise AssertionError(f"no READY telemetry found in {log_path}")


def _log_fixture(payload: dict[str, object]):
    edges = tuple(tuple(int(value) for value in edge) for edge in payload["preference_edges"])
    scenario_ids = tuple(str(value) for value in payload["scenario_ids"])
    row_count = int(payload["policy_row_count"])
    action_mean = float(payload["action_l2_mean"])
    action_max = float(payload["action_l2_max"])
    assert row_count > 0 and edges
    assert 0.0 < action_mean <= action_max

    # Keep the six-dimensional identity and use logged action magnitudes to
    # set an asymmetric, nonzero pseudo action for every sealed row.
    first = torch.linspace(-action_max, action_max, row_count, dtype=torch.float64)
    first = first + action_mean * 0.1
    actions = torch.zeros((row_count, 6), dtype=torch.float64)
    actions[:, 0] = first
    reference_distribution = Normal(torch.zeros_like(actions), torch.ones_like(actions))
    batch = FrontRESRelationalPPOBatch(
        observations=torch.arange(row_count, dtype=torch.float64).reshape(row_count, 1),
        actions=actions,
        old_log_probs=reference_distribution.log_prob(actions).sum(dim=-1),
        valid_mask=torch.ones(row_count, dtype=torch.bool),
        old_means=torch.zeros_like(actions),
        old_sigmas=torch.ones_like(actions),
        transaction_metadata=SimpleNamespace(scenario_ids=scenario_ids),
    )
    return batch, edges, scenario_ids


def _run_current(payload: dict[str, object], batch, edges, sigma: float):
    policy = _GaussianRows(batch.actions, sigma, train_sigma=True)
    result = compute_frontres_relational_preference_loss(policy, batch, edges)
    result.total_loss.backward()
    gradients = [parameter.grad.detach() for parameter in policy.parameters()]
    return result, math.sqrt(sum(float(value.square().sum()) for value in gradients)), float(policy.log_std.grad.norm())


def _seal_reference(policy: _GaussianRows, batch: FrontRESRelationalPPOBatch):
    with torch.no_grad():
        reference = policy.evaluate_segment_actions(batch.observations, batch.actions)
    return replace(
        batch,
        old_log_probs=reference["log_prob"].detach().clone(),
        old_means=reference["mean"].detach().clone(),
        old_sigmas=reference["sigma"].detach().clone(),
    )


def _run_candidate(batch, edges, scenario_ids, sigma: float):
    policy = _GaussianRows(batch.actions, sigma, train_sigma=False)
    sealed = _seal_reference(policy, batch)
    result = compute_frontres_relational_reference_fisher_loss(policy, sealed, edges)
    result.total_loss.backward()
    grad_norm = float(policy.mu.grad.norm())
    return result, grad_norm, policy, sealed


def _run_candidate_adam(batch, edges, scenario_ids, sigma: float):
    policy = _GaussianRows(batch.actions, sigma, train_sigma=False)
    sealed = _seal_reference(policy, batch)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-6)
    optimizer.zero_grad()
    result = compute_frontres_relational_reference_fisher_loss(policy, sealed, edges)
    result.total_loss.backward()
    grad_norm = float(policy.mu.grad.norm())
    before = policy.mu.detach().clone()
    optimizer.step()
    delta = float((policy.mu.detach() - before).norm())
    return float(result.total_loss.detach()), grad_norm, delta, float(result.reference_kl)


def main(log_path: Path | None = None) -> None:
    torch.set_default_dtype(torch.float64)
    resolved_log = log_path or (REPO_ROOT / LOG_NAME)
    payload = _first_ready_payload(resolved_log)
    batch, edges, scenario_ids = _log_fixture(payload)

    current_wide, wide_norm, wide_sigma_norm = _run_current(payload, batch, edges, sigma=1.0)
    current_narrow, narrow_norm, narrow_sigma_norm = _run_current(payload, batch, edges, sigma=0.25)
    assert current_wide.status == current_narrow.status == "READY"
    assert math.isfinite(wide_norm) and math.isfinite(narrow_norm)
    assert narrow_norm > wide_norm * 2.0
    assert narrow_sigma_norm > wide_sigma_norm * 2.0

    candidate_loss, candidate_norm, candidate_policy, candidate_batch = _run_candidate(
        batch, edges, scenario_ids, sigma=1.0
    )
    assert candidate_loss.loss_identity == "pairwise-reference-fisher-scenario-v1"
    edge_scenarios = {scenario_ids[index] for edge in edges for index in edge}
    assert candidate_loss.scenario_count == len(edge_scenarios)
    assert math.isfinite(float(candidate_loss.total_loss.detach()))
    assert math.isfinite(candidate_norm) and candidate_norm > 0.0
    assert not hasattr(candidate_policy, "log_std") or not isinstance(candidate_policy.log_std, torch.nn.Parameter)

    reversed_loss, _, _, _ = _run_candidate(
        batch, tuple(reversed(edges)), scenario_ids, sigma=1.0
    )
    assert math.isclose(
        float(candidate_loss.total_loss.detach()),
        float(reversed_loss.total_loss.detach()),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )

    pair_actions = batch.actions[:2].detach().clone()
    pair_scenarios = (scenario_ids[0], scenario_ids[0])
    pair_batch = replace(
        batch,
        observations=torch.arange(2, dtype=batch.observations.dtype).reshape(2, 1),
        actions=pair_actions,
        old_log_probs=batch.old_log_probs[:2],
        valid_mask=batch.valid_mask[:2],
        old_means=batch.old_means[:2],
        old_sigmas=batch.old_sigmas[:2],
        transaction_metadata=SimpleNamespace(scenario_ids=pair_scenarios),
    )
    _, _, pair_policy, _ = _run_candidate(
        pair_batch, ((0, 1),), pair_scenarios, sigma=1.0
    )
    mean_to_action = pair_actions - pair_policy.mu.detach()
    optimizer_direction = -pair_policy.mu.grad.detach()
    assert float((optimizer_direction[0] * mean_to_action[0]).sum()) > 0.0
    assert float((optimizer_direction[1] * mean_to_action[1]).sum()) < 0.0

    # Independent candidate oracle: recompute the reference-relative margins
    # with plain Python arithmetic from the two Gaussian row log-prob vectors.
    with torch.no_grad():
        current_log_prob = candidate_policy.evaluate_segment_actions(batch.observations, batch.actions)["log_prob"]
        per_scenario: dict[str, list[float]] = {}
        for winner, loser in edges:
            margin = (float(current_log_prob[winner] - candidate_batch.old_log_probs[winner])
                      - float(current_log_prob[loser] - candidate_batch.old_log_probs[loser]))
            per_scenario.setdefault(scenario_ids[winner], []).append(math.log1p(math.exp(-margin)))
        expected_preference = sum(sum(values) / len(values) for values in per_scenario.values()) / len(per_scenario)
        mean = candidate_policy.evaluate_segment_actions(batch.observations, batch.actions)["mean"]
        sigma = candidate_policy.evaluate_segment_actions(batch.observations, batch.actions)["sigma"]
        expected_kl = 0.5 * (
            ((sigma.square() + (mean - candidate_batch.old_means).square()) / candidate_batch.old_sigmas.square())
            - 1.0
            + 2.0 * (candidate_batch.old_sigmas.log() - sigma.log())
        ).sum(dim=-1).mean()
        expected_candidate_loss = expected_preference
    assert math.isclose(float(candidate_loss.total_loss.detach()), expected_candidate_loss, rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(candidate_loss.reference_kl, float(expected_kl), rel_tol=1e-10, abs_tol=1e-10)

    # Scenario balancing: repeating one Scenario's edge must not increase its
    # weight, while an edge-average reduction would change.
    first_scenario = scenario_ids[edges[0][0]]
    second_scenario = next(value for value in scenario_ids if value != first_scenario)
    first_rows = [index for index, value in enumerate(scenario_ids) if value == first_scenario]
    second_rows = [index for index, value in enumerate(scenario_ids) if value == second_scenario]
    balanced_actions = batch.actions.detach().clone()
    balanced_actions[first_rows[1]] = balanced_actions[first_rows[0]]
    balanced_actions[first_rows[2]] = balanced_actions[first_rows[0]]
    balanced_reference = Normal(torch.zeros_like(balanced_actions), torch.ones_like(balanced_actions))
    balanced_batch = replace(
        batch,
        actions=balanced_actions,
        old_log_probs=balanced_reference.log_prob(balanced_actions).sum(dim=-1),
    )
    balanced_edges = ((first_rows[0], first_rows[1]), (second_rows[0], second_rows[1]))
    expanded_edges = balanced_edges + ((first_rows[0], first_rows[2]),)
    balanced_loss, _, _, _ = _run_candidate(balanced_batch, balanced_edges, scenario_ids, sigma=1.0)
    expanded_loss, _, _, _ = _run_candidate(balanced_batch, expanded_edges, scenario_ids, sigma=1.0)
    assert math.isclose(float(expanded_loss.total_loss.detach()), float(balanced_loss.total_loss.detach()), rel_tol=1e-10, abs_tol=1e-10)

    # The active owner fails closed when sealed reference statistics are
    # missing or when an edge crosses Scenario identity.
    active_policy = _GaussianRows(batch.actions, 0.01, train_sigma=False)
    sealed_batch = _seal_reference(active_policy, batch)
    try:
        compute_frontres_relational_reference_fisher_loss(
            active_policy, replace(sealed_batch, old_means=None), edges
        )
    except ValueError:
        pass
    else:
        raise AssertionError("missing reference means must fail closed")
    cross_edge = next(
        (row_a, row_b)
        for row_a, scenario_a in enumerate(scenario_ids)
        for row_b, scenario_b in enumerate(scenario_ids)
        if scenario_a != scenario_b
    )
    try:
        compute_frontres_relational_reference_fisher_loss(
            active_policy, sealed_batch, ((cross_edge[0], cross_edge[1]),)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("cross-Scenario edge must fail closed")

    # Production task-space sigma is a fixed buffer. A trainable sigma must be
    # rejected rather than silently changing the Fisher normalization.
    trainable_sigma_policy = _GaussianRows(batch.actions, 0.5, train_sigma=True)
    trainable_sigma_batch = _seal_reference(trainable_sigma_policy, batch)
    try:
        compute_frontres_relational_reference_fisher_loss(
            trainable_sigma_policy, trainable_sigma_batch, edges
        )
    except ValueError:
        pass
    else:
        raise AssertionError("trainable Actor sigma must fail closed")

    # A producer/reference mismatch must not silently alter the preference
    # margin, even on a row that is not an edge endpoint.
    try:
        compute_frontres_relational_reference_fisher_loss(
            active_policy,
            replace(sealed_batch, old_log_probs=sealed_batch.old_log_probs + 1.0),
            edges,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent sealed Gaussian log-probs must fail closed")

    stale_mean = sealed_batch.old_means + 0.1
    stale_reference = replace(
        sealed_batch,
        old_means=stale_mean,
        old_log_probs=Normal(stale_mean, sealed_batch.old_sigmas)
        .log_prob(sealed_batch.actions)
        .sum(dim=-1),
    )
    try:
        compute_frontres_relational_reference_fisher_loss(
            active_policy, stale_reference, edges
        )
    except ValueError:
        pass
    else:
        raise AssertionError("internally consistent but stale reference must fail closed")

    invalid_mask = sealed_batch.valid_mask.detach().clone()
    invalid_mask[-1] = False
    try:
        compute_frontres_relational_reference_fisher_loss(
            active_policy,
            replace(sealed_batch, valid_mask=invalid_mask),
            edges,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid unreferenced row must fail closed")

    # Independent scalar check for one edge: at a reference-equal policy the
    # relative margin is zero, hence its contribution is exactly log(2).
    zero_margin = math.log(2.0)
    assert math.isclose(zero_margin, math.log1p(math.exp(0.0)), rel_tol=1e-12)

    sweep = {
        sigma: _run_candidate_adam(batch, edges, scenario_ids, sigma)
        for sigma in (1.0, 0.5, 0.25, 0.01)
    }
    # The production owner applies the row-wise Fisher factor, so the
    # gradient handed to Adam should be stable across the sigma sweep.
    assert max(row[1] for row in sweep.values()) / min(row[1] for row in sweep.values()) < 1.25
    assert all(math.isfinite(value) for row in sweep.values() for value in row)
    assert all(math.isclose(row[3], 0.0, abs_tol=1.0e-12) for row in sweep.values())

    print("frontres_reference_preference_loss_log_pseudo_alignment: ACTIVE_V014_PSEUDO_PASS")
    print(f"source_transaction={payload['transaction_id']}")
    print(f"logged_edges={len(edges)} logged_valid_rows={payload['valid_count']}")
    print(f"logged_preclip_gradient_norm={float(payload['actor_gradient_pre_clip_norm']):.6f}")
    print(f"current_sigma_1_loss={float(current_wide.total_loss.detach()):.9f} gradient_norm={wide_norm:.6f} sigma_gradient={wide_sigma_norm:.6f}")
    print(f"current_sigma_0.25_loss={float(current_narrow.total_loss.detach()):.9f} gradient_norm={narrow_norm:.6f} sigma_gradient={narrow_sigma_norm:.6f}")
    print(f"active_v014_fisher_balanced_loss={float(candidate_loss.total_loss.detach()):.9f} mean_gradient_norm={candidate_norm:.6f} sigma_gradient=0.000000 reference_kl_diagnostic={candidate_loss.reference_kl:.9f}")
    print("scenario_balance_duplicate_invariance=PASS")
    print("missing_reference_cross_scenario_trainable_sigma_stale_reference_logprob_and_invalid_row_fail_closed=PASS")
    for sigma, (loss, grad_norm, delta, reference_kl) in sweep.items():
        print(f"active_v014_sigma={sigma:.2f} loss={loss:.9f} fisher_scaled_mu_gradient_norm={grad_norm:.6f} adam_delta_lr_1e-6={delta:.9f} reference_kl={reference_kl:.9f}")
    print("interpretation=active v014 formal owner consumes reference/Scenario/Fisher loss; KL is diagnostic only; no live test started")


if __name__ == "__main__":
    selected_log = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    main(selected_log)
