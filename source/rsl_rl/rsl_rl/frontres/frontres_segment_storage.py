"""Compatibility exports for the split FrontRES Segment storage owners.

Active consumers should import the narrow records, evidence, grouped adapter or
mutable rollout owner directly. This module intentionally owns no behavior.
"""

from rsl_rl.frontres.frontres_segment_evidence_legacy import (
    FrontRESV015GainReturnEvidence,
    FrontRESV015OneActionKEvidence,
    FrontRESV015PairedGainFacts,
    build_frontres_v015_gain_return_evidence,
    pair_frontres_v015_gain_facts,
)
from rsl_rl.frontres.frontres_segment_grouped_adapter import (
    build_frontres_v015_grouped_candidate_storage,
)
from rsl_rl.frontres.frontres_segment_rollout_storage import (
    FrontRESSegmentRolloutStorage,
    FrontRESSegmentStorageStats,
)
from rsl_rl.frontres.frontres_segment_storage_records import (
    FrontRESSegmentStorageBatch,
    FrontRESSegmentTransition,
    FrontRESV015GroupedCandidateMetadata,
    FrontRESV015RejectedTransactionEvidence,
)

__all__ = [
    "FrontRESSegmentRolloutStorage",
    "FrontRESSegmentStorageBatch",
    "FrontRESSegmentStorageStats",
    "FrontRESSegmentTransition",
    "FrontRESV015GainReturnEvidence",
    "FrontRESV015GroupedCandidateMetadata",
    "FrontRESV015OneActionKEvidence",
    "FrontRESV015PairedGainFacts",
    "FrontRESV015RejectedTransactionEvidence",
    "build_frontres_v015_gain_return_evidence",
    "build_frontres_v015_grouped_candidate_storage",
    "pair_frontres_v015_gain_facts",
]
