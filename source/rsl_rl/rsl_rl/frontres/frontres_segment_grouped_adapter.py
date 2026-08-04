"""Bind sealed v015 evidence to grouped PPO storage records."""

from __future__ import annotations

from typing import Any

import torch

from rsl_rl.frontres.frontres_segment_storage_records import (
    FrontRESSegmentStorageBatch,
    FrontRESV015GroupedCandidateMetadata,
)
from rsl_rl.frontres.frontres_segment_evidence import FrontRESSealedRecoveryAwareGainBatch
from rsl_rl.frontres.frontres_gain import FrontRESRecoveryAwareGainResult


def build_frontres_v015_grouped_candidate_storage(
    candidate_evidence: Any,
    *,
    transaction_id: str,
    policy_snapshot_id: str,
    motion_ids: tuple[str, ...],
    start_frames: torch.Tensor,
    segment_ids: torch.Tensor,
    source_index: torch.Tensor,
    trial_index: torch.Tensor,
) -> FrontRESSegmentStorageBatch:
    """Bind sealed v015 candidate evidence to one complete metadata-bearing storage batch.

    函数名说明:
        `build_frontres_v015_grouped_candidate_storage` 是 candidate-only storage
        adapter. 它不收集 rollout, 不选择 best-of-M, 不读取 priority 数值, 不执行
        PPO 或 optimizer.

    主链路:
        上游: Step 3B `FrontRESV015GainConsumerEvidence`.
        下游: `to_grouped_ppo_candidate_batch()` 的完整 v015 transaction batch.

    语义:
        每个 ordinary Repair attempt 只得到一个 `[B,6]` policy row. K 的实际
        survival/evidence count 作为 metadata 保留, 不能复制 row 或改变 actor mass.
    """

    # B1: 验证 sealed Gain carrier 与 Repair row 仍是同一 local scenario.
    validate = getattr(candidate_evidence, "validate", None)
    if not callable(validate):
        raise TypeError("v015 grouped candidate adapter requires validated Step 3B evidence")
    validate()
    return_evidence = getattr(candidate_evidence, "return_evidence", None)
    one_action = getattr(candidate_evidence, "one_action", None)
    validate_return = getattr(return_evidence, "validate", None)
    validate_one_action = getattr(one_action, "validate", None)
    if not callable(validate_return) or not callable(validate_one_action):
        raise TypeError("v015 grouped candidate adapter requires return and one-action evidence")
    validate_return()
    validate_one_action()
    repair_rows = getattr(one_action, "policy_row_indices", None)
    if not isinstance(repair_rows, torch.Tensor) or repair_rows.ndim != 1:
        raise ValueError("v015 grouped candidate adapter requires one Repair-row index per policy tuple")
    count = int(return_evidence.policy_actions.shape[0])
    if int(repair_rows.numel()) != count:
        raise ValueError("v015 grouped candidate adapter Repair-row count disagrees with return evidence")
    expected_horizon = one_action.horizon_k.index_select(0, repair_rows.to(dtype=torch.long))
    expected_survival = one_action.survival_steps.index_select(0, repair_rows.to(dtype=torch.long)).to(
        device=return_evidence.policy_actions.device,
        dtype=torch.float32,
    )
    expected_steps = expected_survival.to(dtype=torch.long)
    if (
        not torch.equal(expected_survival, expected_steps.to(dtype=expected_survival.dtype))
        or not torch.equal(expected_horizon.to(device=return_evidence.horizon_k.device, dtype=torch.long), return_evidence.horizon_k)
        or not torch.equal(return_evidence.evidence_valid_step_count, expected_steps)
        or tuple(one_action.scenario_ids[int(row)] for row in repair_rows.tolist()) != return_evidence.scenario_ids
        or tuple(one_action.noisy_segment_hashes[int(row)] for row in repair_rows.tolist()) != return_evidence.noisy_segment_hashes
        or tuple(one_action.x_t_identities[int(row)] for row in repair_rows.tolist()) != return_evidence.x_t_identities
    ):
        raise ValueError("v015 grouped candidate adapter lost one-action local scenario identity or K evidence")

    # B2: 封存 transaction/motion/Segment/trial 与 local scenario row metadata.
    metadata = FrontRESV015GroupedCandidateMetadata(
        transaction_id=transaction_id,
        policy_snapshot_id=policy_snapshot_id,
        motion_ids=motion_ids,
        start_frames=start_frames,
        segment_ids=segment_ids,
        source_index=source_index,
        trial_index=trial_index,
        horizon_k=return_evidence.horizon_k,
        evidence_valid_step_count=return_evidence.evidence_valid_step_count,
        trial_role=("policy",) * count,
        noisy_segment_hashes=return_evidence.noisy_segment_hashes,
        scenario_ids=return_evidence.scenario_ids,
        x_t_identities=return_evidence.x_t_identities,
        intent_q29_provenance=return_evidence.intent_q29_provenance,
        intent_q29_source=return_evidence.intent_q29_source,
    )
    metadata.validate()

    # B3: 构造 one-row storage batch, metadata 只供 grouped candidate path 消费.
    device = return_evidence.policy_actions.device
    storage_batch = FrontRESSegmentStorageBatch(
        observations=return_evidence.policy_observations.detach().clone(),
        privileged_observations=one_action.policy_privileged_observations.detach().clone(),
        actions=return_evidence.policy_actions.detach().clone(),
        old_log_probs=return_evidence.policy_log_probs.detach().clone(),
        old_values=return_evidence.policy_values.detach().clone(),
        rewards=return_evidence.gain_total.detach().clone(),
        returns=return_evidence.return_k.detach().clone(),
        advantages=return_evidence.advantage_k.detach().clone(),
        valid_mask=return_evidence.policy_row_valid.detach().clone(),
        segment_ids=metadata.segment_ids.to(device=device, dtype=torch.long).detach().clone(),
        old_means=return_evidence.policy_means.detach().clone(),
        old_sigmas=return_evidence.policy_sigmas.detach().clone(),
        transaction_metadata=metadata,
        transaction_row_indices=torch.arange(count, device=device, dtype=torch.long),
    )
    return storage_batch


def build_frontres_v017_grouped_candidate_storage(
    evidence: FrontRESSealedRecoveryAwareGainBatch,
    gain: FrontRESRecoveryAwareGainResult,
    *,
    motion_ids: tuple[str, ...],
    start_frames: torch.Tensor,
    intent_q29_provenance: str,
    intent_q29_source: str,
) -> FrontRESSegmentStorageBatch:
    """Bind owner-produced v007 scalars to all valid Repair policy rows."""

    if not isinstance(evidence, FrontRESSealedRecoveryAwareGainBatch):
        raise TypeError("v017 grouped storage requires sealed recovery-aware evidence")
    if not isinstance(gain, FrontRESRecoveryAwareGainResult):
        raise TypeError("v017 grouped storage requires the owner-produced Gain result")
    evidence.validate()
    attempts = evidence.ordered_attempts
    count = len(attempts)
    gain_total = gain.gain_total.detach().clone().reshape(-1)
    if int(gain_total.numel()) != count or not bool(torch.isfinite(gain_total).all()):
        raise ValueError("v017 grouped storage requires one finite G_total per Repair attempt")
    if len(motion_ids) != count or tuple(start_frames.shape) != (count,):
        raise ValueError("v017 grouped storage motion/start identity must align with Repair rows")

    def stack(name: str) -> torch.Tensor:
        return torch.stack([getattr(value, name) for value in attempts], dim=0).detach().clone()

    device = attempts[0].policy_action.device
    source_index = torch.tensor([value.source_index for value in attempts], device=device, dtype=torch.long)
    segment_ids = torch.tensor([value.segment_id for value in attempts], device=device, dtype=torch.long)
    trial_index = torch.tensor([value.trial_index for value in attempts], device=device, dtype=torch.long)
    horizon_k = torch.tensor([value.horizon_k for value in attempts], device=device, dtype=torch.long)
    valid_steps = torch.tensor(
        [int(value.repair.valid_mask[:, 0].sum().item()) for value in attempts],
        device=device,
        dtype=torch.long,
    )
    metadata = FrontRESV015GroupedCandidateMetadata(
        transaction_id=attempts[0].transaction_id,
        policy_snapshot_id=attempts[0].policy_snapshot_id,
        motion_ids=tuple(str(value) for value in motion_ids),
        start_frames=start_frames,
        segment_ids=segment_ids,
        source_index=source_index,
        trial_index=trial_index,
        horizon_k=horizon_k,
        evidence_valid_step_count=valid_steps,
        trial_role=("policy",) * count,
        noisy_segment_hashes=tuple(value.noisy_segment_hash for value in attempts),
        scenario_ids=tuple(value.scenario_id for value in attempts),
        x_t_identities=tuple(value.x_t_identity for value in attempts),
        intent_q29_provenance=str(intent_q29_provenance),
        intent_q29_source=str(intent_q29_source),
    )
    old_values = stack("policy_value").reshape(-1)
    return FrontRESSegmentStorageBatch(
        observations=stack("policy_observation"),
        privileged_observations=stack("policy_privileged_observation"),
        actions=stack("policy_action"),
        old_log_probs=stack("policy_log_prob").reshape(-1),
        old_values=old_values,
        rewards=gain_total,
        returns=gain_total,
        advantages=gain_total - old_values,
        valid_mask=torch.ones(count, device=device, dtype=torch.bool),
        segment_ids=segment_ids,
        old_means=stack("policy_mean"),
        old_sigmas=stack("policy_sigma"),
        transaction_metadata=metadata,
        transaction_row_indices=torch.arange(count, device=device, dtype=torch.long),
    )
