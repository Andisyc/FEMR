"""Immutable diagnostics for one FRS-GAIN-v009 relational transaction."""

from __future__ import annotations

from dataclasses import dataclass

from rsl_rl.frontres.frontres_segment_evidence import FrontRESSealedRecoveryAwareGainBatch


@dataclass(frozen=True)
class FrontRESRelationalEvaluationReport:
    transaction_id: str
    policy_row_count: int
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    source_statuses: tuple[str, ...]
    preference_edges: tuple[tuple[int, int], ...]
    comparable_pair_count_by_row: tuple[int, ...]

    def validate(self) -> None:
        rows = int(self.policy_row_count)
        if not self.transaction_id or rows <= 0:
            raise ValueError("relational report requires transaction identity and policy rows")
        if len(self.scenario_ids) != rows or len(self.noisy_segment_hashes) != rows:
            raise ValueError("relational report row identity is misaligned")
        if any(not value for value in self.scenario_ids + self.noisy_segment_hashes):
            raise ValueError("relational report requires complete Scenario identity")
        if not self.source_statuses or any(
            value not in {"READY", "NO_COMPARABLE_PAIRS"} for value in self.source_statuses
        ):
            raise ValueError("relational report contains an invalid source status")
        if len(self.comparable_pair_count_by_row) != rows or any(
            isinstance(value, bool) or int(value) < 0 for value in self.comparable_pair_count_by_row
        ):
            raise ValueError("relational report requires nonnegative row-aligned pair counts")
        for winner, loser in self.preference_edges:
            if winner == loser or not (0 <= int(winner) < rows) or not (0 <= int(loser) < rows):
                raise ValueError("relational report contains an invalid preference edge")


def build_frontres_relational_evaluation_report(
    evidence: FrontRESSealedRecoveryAwareGainBatch,
) -> FrontRESRelationalEvaluationReport:
    evidence.validate()
    attempts = evidence.ordered_attempts
    batches = evidence.relational_training_batches()
    counts = tuple(
        int(value)
        for batch in batches
        for value in batch.comparable_pair_count
    )
    report = FrontRESRelationalEvaluationReport(
        transaction_id=attempts[0].transaction_id,
        policy_row_count=len(attempts),
        scenario_ids=tuple(value.scenario_id for value in attempts),
        noisy_segment_hashes=tuple(value.noisy_segment_hash for value in attempts),
        source_statuses=tuple(value.status for value in batches),
        preference_edges=evidence.relational_preference_edges(),
        comparable_pair_count_by_row=counts,
    )
    report.validate()
    return report


__all__ = (
    "FrontRESRelationalEvaluationReport",
    "build_frontres_relational_evaluation_report",
)
