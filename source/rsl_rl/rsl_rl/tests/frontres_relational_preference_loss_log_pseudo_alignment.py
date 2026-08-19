"""Log-calibrated RED test for the proposed FRS-PPO-v014 preference loss.

The training log supplies real Scenario identities, BETTER edges, Actor credit,
and action scales.  It does not contain row-level current policy log
probabilities, so this test constructs those values explicitly and reports only
an S1 numerical module result, not a real transaction replay.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESRelationalPPOBatch,
    compute_frontres_relational_preference_loss,
)


LOG_NAME = "FRS_TRAIN_V025_RELATIONAL_1805_CUDA0_20260818_RERUN2.log"
TELEMETRY_PREFIX = "[FrontRES v017 Transaction Telemetry] "
EXPECTED_GAIN_CONTRACT = "FRS-GAIN-v009"
EXPECTED_ACTIVE_OPTIMIZATION_CONTRACT = "FRS-PPO-v013"
EXPECTED_CANDIDATE_CONTRACT = "FRS-PPO-v014"
PREFERENCE_BETA = 1.0


class _RowLogProbabilityPolicy:
    """Controlled policy dependency exposing one trainable log-prob per row."""

    def __init__(self, row_log_probs: list[float]) -> None:
        self.row_log_probs = torch.nn.Parameter(torch.tensor(row_log_probs, dtype=torch.float64))

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        row_ids = observations[:, 0].to(dtype=torch.long)
        log_prob = self.row_log_probs.index_select(0, row_ids)
        return {
            "log_prob": log_prob,
            "value": torch.zeros_like(log_prob),
            "entropy": None,
        }


def _first_ready_telemetry(log_path: Path) -> dict[str, object]:
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            marker = line.find(TELEMETRY_PREFIX)
            if marker < 0:
                continue
            payload = json.loads(line[marker + len(TELEMETRY_PREFIX) :])
            if payload.get("status") == "READY" and payload.get("preference_edges"):
                return payload
    raise AssertionError(f"no READY relational telemetry found in {log_path}")


def _validate_log_identity(payload: dict[str, object]) -> None:
    assert payload["gain_contract_id"] == EXPECTED_GAIN_CONTRACT
    assert payload["optimization_contract_id"] == EXPECTED_ACTIVE_OPTIMIZATION_CONTRACT
    assert payload["active_m"] == 4
    assert payload["critic_target_id"] == "none"
    assert payload["scalar_target_id"] == "none"

    row_count = int(payload["policy_row_count"])
    actor_credit = [float(value) for value in payload["actor_credit"]]
    scenario_ids = [str(value) for value in payload["scenario_ids"]]
    edges = [tuple(int(value) for value in edge) for edge in payload["preference_edges"]]
    assert len(actor_credit) == row_count == len(scenario_ids)

    recomputed_credit = [0.0] * row_count
    for winner, loser in edges:
        assert 0 <= winner < row_count and 0 <= loser < row_count
        assert scenario_ids[winner] == scenario_ids[loser]
        recomputed_credit[winner] += 1.0
        recomputed_credit[loser] -= 1.0
    assert recomputed_credit == actor_credit


def _select_disjoint_logged_edges(payload: dict[str, object]) -> tuple[tuple[int, int], ...]:
    scenario_ids = [str(value) for value in payload["scenario_ids"]]
    selected: list[tuple[int, int]] = []
    used_scenarios: set[str] = set()
    used_rows: set[int] = set()
    for edge in payload["preference_edges"]:
        winner, loser = (int(edge[0]), int(edge[1]))
        scenario_id = scenario_ids[winner]
        if scenario_id in used_scenarios or winner in used_rows or loser in used_rows:
            continue
        selected.append((winner, loser))
        used_scenarios.add(scenario_id)
        used_rows.update((winner, loser))
        if len(selected) == 4:
            break
    assert len(selected) == 4, "log fixture requires four disjoint real Scenario edges"
    return tuple(selected)


def _log_calibrated_fixture(
    payload: dict[str, object],
) -> tuple[FrontRESRelationalPPOBatch, tuple[tuple[int, int], ...], list[float]]:
    logged_edges = _select_disjoint_logged_edges(payload)
    action_mean = float(payload["action_l2_mean"])
    action_max = float(payload["action_l2_max"])
    assert 0.0 < action_mean <= action_max

    row_log_probs: list[float] = []
    actions: list[list[float]] = []
    remapped_edges: list[tuple[int, int]] = []
    for pair_index, _logged_edge in enumerate(logged_edges):
        strength = float(pair_index + 1) / float(len(logged_edges))
        winner_row = len(row_log_probs)
        loser_row = winner_row + 1
        row_log_probs.extend((action_mean * strength, -action_max * strength))
        actions.extend(
            (
                [action_mean * strength, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-action_max * strength, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        )
        remapped_edges.append((winner_row, loser_row))

    row_count = len(row_log_probs)
    stale_distance = math.sqrt(float(payload["actor_gradient_pre_clip_norm"]))
    batch = FrontRESRelationalPPOBatch(
        observations=torch.arange(row_count, dtype=torch.float64).reshape(row_count, 1),
        actions=torch.tensor(actions, dtype=torch.float64),
        old_log_probs=torch.full((row_count,), -stale_distance, dtype=torch.float64),
        valid_mask=torch.ones(row_count, dtype=torch.bool),
    )
    return batch, tuple(remapped_edges), row_log_probs


def _independent_logistic_oracle(
    row_log_probs: list[float], edges: tuple[tuple[int, int], ...]
) -> tuple[float, list[float]]:
    edge_count = float(len(edges))
    expected_loss = 0.0
    expected_gradient = [0.0] * len(row_log_probs)
    for winner, loser in edges:
        margin = PREFERENCE_BETA * (row_log_probs[winner] - row_log_probs[loser])
        expected_loss += math.log1p(math.exp(-margin)) / edge_count
        edge_gradient = -PREFERENCE_BETA / (1.0 + math.exp(margin)) / edge_count
        expected_gradient[winner] += edge_gradient
        expected_gradient[loser] -= edge_gradient
    return expected_loss, expected_gradient


def main(log_path: Path | None = None) -> None:
    resolved_log = log_path or (REPO_ROOT / LOG_NAME)
    payload = _first_ready_telemetry(resolved_log)
    _validate_log_identity(payload)
    batch, edges, row_log_probs = _log_calibrated_fixture(payload)
    expected_loss, expected_gradient = _independent_logistic_oracle(row_log_probs, edges)

    # Sensitivity: the former direct difference objective must not satisfy the
    # logistic preference oracle on this asymmetric log-calibrated fixture.
    direct_difference_mutant = -sum(
        row_log_probs[winner] - row_log_probs[loser] for winner, loser in edges
    ) / float(len(edges))
    assert not math.isclose(direct_difference_mutant, expected_loss, rel_tol=1e-9, abs_tol=1e-9)

    policy = _RowLogProbabilityPolicy(row_log_probs)
    result = compute_frontres_relational_preference_loss(policy, batch, edges)
    result.total_loss.backward()
    observed_loss = float(result.total_loss.detach())
    observed_gradient = [float(value) for value in policy.row_log_probs.grad]

    assert result.contract_id == EXPECTED_CANDIDATE_CONTRACT
    assert result.status == "READY"
    assert result.edge_count == len(edges)
    assert math.isclose(observed_loss, expected_loss, rel_tol=1e-9, abs_tol=1e-9), (
        "FRS-PPO-v014 candidate does not implement the confirmed logistic preference loss: "
        f"observed={observed_loss:.12f}, expected={expected_loss:.12f}, "
        f"direct_difference_mutant={direct_difference_mutant:.12f}"
    )
    for row, (observed, expected) in enumerate(zip(observed_gradient, expected_gradient)):
        assert math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9), (
            f"row {row} preference gradient mismatch: observed={observed}, expected={expected}"
        )

    # old_log_probs are deliberately stale and must not affect a current-policy
    # pairwise preference likelihood.
    fresh_batch = FrontRESRelationalPPOBatch(
        observations=batch.observations,
        actions=batch.actions,
        old_log_probs=torch.zeros_like(batch.old_log_probs),
        valid_mask=batch.valid_mask,
    )
    fresh_policy = _RowLogProbabilityPolicy(row_log_probs)
    fresh = compute_frontres_relational_preference_loss(fresh_policy, fresh_batch, edges)
    assert math.isclose(float(fresh.total_loss.detach()), expected_loss, rel_tol=1e-9, abs_tol=1e-9)

    print("frontres_relational_preference_loss_log_pseudo_alignment: MODULE-CORRECT")
    print(f"source_transaction={payload['transaction_id']}")
    print(f"logged_gradient_pre_clip_norm={float(payload['actor_gradient_pre_clip_norm']):.6f}")
    print(f"observed_loss={observed_loss:.12f}")
    print(f"expected_logistic_loss={expected_loss:.12f}")


if __name__ == "__main__":
    selected_log = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    main(selected_log)
