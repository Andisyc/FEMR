"""Historical v015 one-action-K and v006 return evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rsl_rl.frontres.frontres_gain_legacy import evaluate_phase_conditioned_physics
from rsl_rl.frontres.frontres_segment_storage_records import FrontRESV015RejectedTransactionEvidence

@dataclass(frozen=True)
class FrontRESV015OneActionKEvidence:
    """Evidence for one Repair tuple and its frozen-GMT K consequence.

    This is deliberately not a PPO batch: it has no reward, return, advantage,
    optimizer, or legacy ``to_ppo_batch`` path. Formal training may convert it
    into a sealed metadata candidate; held-out evaluation consumes it directly.
    """

    policy_observations: torch.Tensor
    policy_privileged_observations: torch.Tensor
    policy_actions: torch.Tensor
    policy_log_probs: torch.Tensor
    policy_values: torch.Tensor
    policy_means: torch.Tensor
    policy_sigmas: torch.Tensor
    policy_row_indices: torch.Tensor
    t_env_actions: torch.Tensor
    continuation: torch.Tensor
    continuation_valid_mask: torch.Tensor
    frozen_gmt_env_actions: torch.Tensor
    actor_forward_count: int
    later_femr_action_count: int
    horizon_k: torch.Tensor
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    roles: tuple[str, ...]
    intent_q29: torch.Tensor
    intent_q29_provenance: tuple[str, ...]
    intent_q29_source: tuple[str, ...]
    executed_q29_t: torch.Tensor
    executed_q29_t_valid_mask: torch.Tensor
    done_any: torch.Tensor
    survival_steps: torch.Tensor
    physics_expected_support_steps: torch.Tensor
    physics_zmp_repaired_steps: torch.Tensor
    physics_zmp_noisy_steps: torch.Tensor
    physics_contact_repaired_steps: torch.Tensor
    physics_contact_noisy_steps: torch.Tensor
    physics_pair_valid_mask: torch.Tensor
    physics_survival_repaired_steps: torch.Tensor | None = None
    physics_survival_noisy_steps: torch.Tensor | None = None
    evaluation_only_lateral_lean_repaired_steps: torch.Tensor | None = None
    evaluation_only_lateral_lean_noisy_steps: torch.Tensor | None = None

    def validate(self) -> None:
        """Fail closed unless the evidence encodes exactly one Repair policy row per scenario."""

        policy_count = int(self.policy_actions.shape[0])
        role_count = int(self.t_env_actions.shape[0])
        if int(self.actor_forward_count) != 1 or int(self.later_femr_action_count) != 0:
            raise ValueError("v015 one-action evidence requires exactly one actor forward and zero later FEMR actions")
        if policy_count <= 0 or role_count != 2 * policy_count:
            raise ValueError("v015 one-action evidence requires equal Repair/Noisy roles and one Repair policy row per scenario")
        if self.policy_actions.ndim != 2 or int(self.policy_actions.shape[1]) != 6:
            raise ValueError("v015 one-action evidence requires policy_actions [B,6]")
        if self.policy_observations.ndim != 2 or int(self.policy_observations.shape[0]) != policy_count:
            raise ValueError("v015 one-action evidence policy observations must align with Repair rows")
        if (
            self.policy_privileged_observations.ndim != 2
            or int(self.policy_privileged_observations.shape[0]) != policy_count
            or int(self.policy_privileged_observations.shape[1]) <= 0
        ):
            raise ValueError("v015 one-action evidence privileged observations must be non-empty [B,C]")
        vector_fields = {
            "policy_log_probs": self.policy_log_probs,
            "policy_values": self.policy_values,
            "policy_row_indices": self.policy_row_indices,
        }
        for name, value in vector_fields.items():
            if value.ndim != 1 or int(value.numel()) != policy_count:
                raise ValueError(f"v015 one-action evidence {name} must be [B]")
        for name, value in (("policy_means", self.policy_means), ("policy_sigmas", self.policy_sigmas)):
            if tuple(value.shape) != tuple(self.policy_actions.shape):
                raise ValueError(f"v015 one-action evidence {name} must be [B,6]")
        if (
            self.continuation.ndim != 3
            or int(self.continuation.shape[1]) != role_count
            or int(self.continuation.shape[2]) != 65
            or tuple(self.continuation_valid_mask.shape) != tuple(self.continuation.shape[:2])
            or self.frozen_gmt_env_actions.ndim != 3
            or tuple(self.frozen_gmt_env_actions.shape[:2]) != tuple(self.continuation.shape[:2])
        ):
            raise ValueError("v015 one-action evidence requires [K,N,65] C, [K,N] masks, and [K,N,A] frozen GMT actions")
        if tuple(self.horizon_k.shape) != (role_count,) or bool((self.horizon_k <= 0).any()):
            raise ValueError("v015 one-action evidence horizon_k must be positive [N]")
        if int(self.continuation.shape[0]) != int(self.horizon_k.max().item()):
            raise ValueError("v015 one-action evidence K dimension must equal max per-row horizon_k")
        expected_valid = torch.arange(
            int(self.continuation.shape[0]), device=self.horizon_k.device, dtype=torch.long
        ).unsqueeze(1) < self.horizon_k.unsqueeze(0)
        if not torch.equal(self.continuation_valid_mask.to(device=expected_valid.device, dtype=torch.bool), expected_valid):
            raise ValueError("v015 one-action evidence valid mask must exactly encode each K horizon")
        metadata = (self.scenario_ids, self.noisy_segment_hashes, self.x_t_identities, self.roles)
        if any(len(value) != role_count for value in metadata):
            raise ValueError("v015 one-action evidence metadata must cover every Repair/Noisy role row")
        if any(role not in {"repair", "noisy"} for role in self.roles):
            raise ValueError("v015 one-action evidence rejects Clean and legacy quartet roles")
        if (
            self.intent_q29.ndim != 3
            or int(self.intent_q29.shape[0]) != role_count
            or int(self.intent_q29.shape[1]) < 2
            or int(self.intent_q29.shape[2]) != 29
        ):
            raise ValueError("v015 one-action evidence requires intent_q29 [N,H+1,29] with H>=1")
        if tuple(self.executed_q29_t.shape) != (role_count, 29):
            raise ValueError("v015 one-action evidence requires post-t executed_q29_t [N,29]")
        for name, value in (
            ("executed_q29_t_valid_mask", self.executed_q29_t_valid_mask),
            ("done_any", self.done_any),
            ("survival_steps", self.survival_steps),
        ):
            if value.ndim != 1 or int(value.numel()) != role_count:
                raise ValueError(f"v015 one-action evidence {name} must be [N]")
        physics_shape = (int(self.continuation.shape[0]), policy_count)
        if tuple(self.physics_expected_support_steps.shape) != physics_shape + (2,):
            raise ValueError(
                f"v015 one-action evidence physics_expected_support_steps must be [K,B,2]={physics_shape + (2,)}"
            )
        if bool(
            ((self.physics_expected_support_steps != 0.0) & (self.physics_expected_support_steps != 1.0)).any()
        ):
            raise ValueError("v015 one-action evidence expected support must be binary left/right state")
        for name, value in (
            ("physics_zmp_repaired_steps", self.physics_zmp_repaired_steps),
            ("physics_zmp_noisy_steps", self.physics_zmp_noisy_steps),
            ("physics_pair_valid_mask", self.physics_pair_valid_mask),
        ):
            if tuple(value.shape) != physics_shape:
                raise ValueError(f"v015 one-action evidence {name} must be [K,B]={physics_shape}")
        for name, value in (
            ("physics_survival_repaired_steps", self.physics_survival_repaired_steps),
            ("physics_survival_noisy_steps", self.physics_survival_noisy_steps),
            ("evaluation_only_lateral_lean_repaired_steps", self.evaluation_only_lateral_lean_repaired_steps),
            ("evaluation_only_lateral_lean_noisy_steps", self.evaluation_only_lateral_lean_noisy_steps),
        ):
            if value is not None and tuple(value.shape) != physics_shape:
                raise ValueError(f"v015 one-action evidence {name} must be [K,B]={physics_shape}")
        physics_valid = self.physics_pair_valid_mask.bool()
        contact_shape = physics_shape + (2,)
        for name, value in (
            ("physics_contact_repaired_steps", self.physics_contact_repaired_steps),
            ("physics_contact_noisy_steps", self.physics_contact_noisy_steps),
        ):
            if tuple(value.shape) != contact_shape:
                raise ValueError(f"v015 one-action evidence {name} must be [K,B,2]={contact_shape}")
        for name, value, actual_contact in (
            ("physics_zmp_repaired_steps", self.physics_zmp_repaired_steps, self.physics_contact_repaired_steps),
            ("physics_zmp_noisy_steps", self.physics_zmp_noisy_steps, self.physics_contact_noisy_steps),
        ):
            applicable = (
                physics_valid
                & self.physics_expected_support_steps.bool().any(dim=-1)
                & actual_contact.bool().any(dim=-1)
            )
            finite = torch.isfinite(value.float())
            if not bool(finite[applicable].all()) or bool(finite[~applicable].any()):
                raise ValueError(
                    f"v015 one-action evidence {name} must be finite exactly on loaded-support paired-valid K steps"
                )
        if (
            len(self.intent_q29_provenance) != role_count
            or len(self.intent_q29_source) != role_count
            or not bool(torch.isfinite(self.survival_steps.float()).all())
            or bool((self.survival_steps.float() < 0.0).any())
            or bool((self.survival_steps.float() > self.horizon_k.float()).any())
        ):
            raise ValueError("v015 one-action evidence has invalid q29 provenance or K survival evidence")
        repair_rows = torch.tensor(
            [index for index, role in enumerate(self.roles) if role == "repair"],
            device=self.policy_row_indices.device,
            dtype=torch.long,
        )
        if not torch.equal(self.policy_row_indices.to(dtype=torch.long), repair_rows):
            raise ValueError("v015 one-action evidence policy rows must be exactly the Repair role rows")
        support_by_scenario: dict[str, torch.Tensor] = {}
        for policy_col, role_row in enumerate(repair_rows.tolist()):
            scenario_id = str(self.scenario_ids[int(role_row)])
            support = self.physics_expected_support_steps[:, policy_col]
            anchor_support = support_by_scenario.setdefault(scenario_id, support)
            if not torch.equal(anchor_support, support):
                raise ValueError(f"v015 one-action evidence scenario={scenario_id!r} mixes expected support identity")
        rows_by_scenario: dict[str, list[int]] = {}
        for row, scenario_id in enumerate(self.scenario_ids):
            rows_by_scenario.setdefault(str(scenario_id), []).append(row)
        if not rows_by_scenario or len(rows_by_scenario) > policy_count:
            raise ValueError("v015 one-action evidence has an invalid scenario identity partition")
        for scenario_id, rows in rows_by_scenario.items():
            repair_rows_for_scenario = [row for row in rows if self.roles[row] == "repair"]
            noisy_rows_for_scenario = [row for row in rows if self.roles[row] == "noisy"]
            if not repair_rows_for_scenario or len(repair_rows_for_scenario) != len(noisy_rows_for_scenario):
                raise ValueError(
                    f"v015 one-action evidence scenario={scenario_id!r} must have equal Repair/Noisy attempt rows"
                )
            # M 次尝试共享同一 sealed scenario. 逐行验证 I/C/hash/x_t/K, 而不是
            # 错误地要求每条 Repair policy row 都拥有不同 scenario_id.
            anchor = rows[0]
            for row in rows[1:]:
                if (
                    self.noisy_segment_hashes[anchor] != self.noisy_segment_hashes[row]
                    or self.x_t_identities[anchor] != self.x_t_identities[row]
                    or int(self.horizon_k[anchor].item()) != int(self.horizon_k[row].item())
                    or not torch.equal(self.continuation[:, anchor], self.continuation[:, row])
                    or not torch.equal(self.intent_q29[anchor], self.intent_q29[row])
                    or self.intent_q29_provenance[anchor] != self.intent_q29_provenance[row]
                    or self.intent_q29_source[anchor] != self.intent_q29_source[row]
                ):
                    raise ValueError(f"v015 one-action evidence scenario={scenario_id!r} mixes immutable local artifacts")
            provenance = self.intent_q29_provenance[anchor]
            source = self.intent_q29_source[anchor].lower()
            if provenance != "deployment_noisy_q29" or not source or any(
                token in source for token in ("clean", "root", "global")
            ):
                raise ValueError("v015 one-action evidence q29 target must retain deployment/Noisy provenance")
        tensors = (
            self.policy_observations,
            self.policy_privileged_observations,
            self.policy_actions,
            self.policy_log_probs,
            self.policy_values,
            self.policy_means,
            self.policy_sigmas,
            self.t_env_actions,
            self.continuation,
            self.continuation_valid_mask,
            self.frozen_gmt_env_actions,
            self.horizon_k,
            self.intent_q29,
            self.executed_q29_t,
            self.executed_q29_t_valid_mask,
            self.done_any,
            self.survival_steps,
            self.physics_expected_support_steps,
            self.physics_zmp_repaired_steps,
            self.physics_zmp_noisy_steps,
            self.physics_contact_repaired_steps,
            self.physics_contact_noisy_steps,
            self.physics_pair_valid_mask,
        )
        if any(value.requires_grad for value in tensors):
            raise ValueError("v015 one-action evidence must be immutable detached capture data")


@dataclass(frozen=True)
class FrontRESV015PairedGainFacts:
    """将一个 Repair policy row 与其 Noisy baseline 配对给 v005 owner.

    状态: active v015 local storage adapter.
    上游: immutable one-action capture.
    下游: FRS-GAIN-v006 scalar target/vector-constraint input 和 one return/advantage carrier.
    证据: deterministic formal connectivity; live Physics values remain pending.
    """

    policy_observations: torch.Tensor
    policy_actions: torch.Tensor
    policy_log_probs: torch.Tensor
    policy_values: torch.Tensor
    policy_means: torch.Tensor
    policy_sigmas: torch.Tensor
    intent_q29: torch.Tensor
    repaired_q29: torch.Tensor
    noisy_q29: torch.Tensor
    intent_valid_mask: torch.Tensor
    repaired_success: torch.Tensor
    noisy_success: torch.Tensor
    repaired_survival: torch.Tensor
    noisy_survival: torch.Tensor
    repaired_zmp_margin: torch.Tensor
    noisy_zmp_margin: torch.Tensor
    repaired_contact: torch.Tensor
    noisy_contact: torch.Tensor
    repaired_contact_violation: torch.Tensor
    noisy_contact_violation: torch.Tensor
    repaired_zmp_violation: torch.Tensor
    noisy_zmp_violation: torch.Tensor
    physics_valid_step_count: torch.Tensor
    horizon_k: torch.Tensor
    expected_support_steps: torch.Tensor
    repaired_contact_steps: torch.Tensor
    noisy_contact_steps: torch.Tensor
    repaired_zmp_margin_steps: torch.Tensor
    noisy_zmp_margin_steps: torch.Tensor
    physics_pair_valid_mask: torch.Tensor
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    intent_q29_provenance: str
    intent_q29_source: str

    def validate(self) -> None:
        count = int(self.policy_actions.shape[0])
        if count <= 0 or self.policy_actions.ndim != 2 or int(self.policy_actions.shape[1]) != 6:
            raise ValueError("v015 paired gain facts require policy_actions [B,6]")
        if self.policy_observations.ndim != 2 or int(self.policy_observations.shape[0]) != count:
            raise ValueError("v015 paired gain facts observations must align with policy rows")
        for name, value in (
            ("policy_log_probs", self.policy_log_probs),
            ("policy_values", self.policy_values),
            ("intent_valid_mask", self.intent_valid_mask),
            ("repaired_success", self.repaired_success),
            ("noisy_success", self.noisy_success),
            ("repaired_survival", self.repaired_survival),
            ("noisy_survival", self.noisy_survival),
            ("repaired_zmp_margin", self.repaired_zmp_margin),
            ("noisy_zmp_margin", self.noisy_zmp_margin),
            ("repaired_contact", self.repaired_contact),
            ("noisy_contact", self.noisy_contact),
            ("repaired_contact_violation", self.repaired_contact_violation),
            ("noisy_contact_violation", self.noisy_contact_violation),
            ("repaired_zmp_violation", self.repaired_zmp_violation),
            ("noisy_zmp_violation", self.noisy_zmp_violation),
            ("physics_valid_step_count", self.physics_valid_step_count),
            ("horizon_k", self.horizon_k),
        ):
            if value.ndim != 1 or int(value.numel()) != count:
                raise ValueError(f"v015 paired gain facts {name} must be [B]")
        for name, value in (
            ("policy_means", self.policy_means),
            ("policy_sigmas", self.policy_sigmas),
            ("intent_q29", self.intent_q29),
            ("repaired_q29", self.repaired_q29),
            ("noisy_q29", self.noisy_q29),
        ):
            expected = (count, 6) if name.startswith("policy_") else (count, 29)
            if tuple(value.shape) != expected:
                raise ValueError(f"v015 paired gain facts {name} must be {expected}")
        if (
            len(self.scenario_ids) != count
            or len(self.noisy_segment_hashes) != count
            or len(self.x_t_identities) != count
            or not bool((self.horizon_k > 0).all())
            or not bool(torch.isfinite(self.repaired_survival.float()).all())
            or not bool(torch.isfinite(self.noisy_survival.float()).all())
            or bool((self.physics_valid_step_count < 0).any())
            or bool((self.physics_valid_step_count > self.horizon_k).any())
        ):
            raise ValueError("v015 paired gain facts have invalid identity or Physics evidence")
        k_steps = int(self.expected_support_steps.shape[0])
        if (
            tuple(self.expected_support_steps.shape) != (k_steps, count, 2)
            or tuple(self.repaired_contact_steps.shape) != (k_steps, count, 2)
            or tuple(self.noisy_contact_steps.shape) != (k_steps, count, 2)
            or tuple(self.repaired_zmp_margin_steps.shape) != (k_steps, count)
            or tuple(self.noisy_zmp_margin_steps.shape) != (k_steps, count)
            or tuple(self.physics_pair_valid_mask.shape) != (k_steps, count)
        ):
            raise ValueError("FRS-GAIN-v006 paired facts require ordered [K,B] Contact/ZMP evidence")
        valid = self.intent_valid_mask.bool()
        for name, value in (("repaired_contact", self.repaired_contact), ("noisy_contact", self.noisy_contact)):
            finite = torch.isfinite(value.float())
            if not bool(finite[valid].all()) or bool(finite[~valid].any()):
                raise ValueError(f"v015 paired gain facts {name} must be finite exactly on valid policy rows")
        for name, value, actual_contact in (
            ("repaired_zmp_margin", self.repaired_zmp_margin, self.repaired_contact_steps),
            ("noisy_zmp_margin", self.noisy_zmp_margin, self.noisy_contact_steps),
        ):
            applicable = (
                self.physics_pair_valid_mask.bool()
                & self.expected_support_steps.bool().any(dim=-1)
                & actual_contact.bool().any(dim=-1)
            ).any(dim=0) & valid
            finite = torch.isfinite(value.float())
            if not bool(finite[applicable].all()) or bool(finite[~applicable].any()):
                raise ValueError(f"v015 paired gain facts {name} must follow role-specific loaded-support applicability")
        source = self.intent_q29_source.lower()
        if (
            self.intent_q29_provenance != "deployment_noisy_q29"
            or not source
            or any(token in source for token in ("clean", "root", "global"))
        ):
            raise ValueError("v015 paired gain facts reject non-deployment q29 provenance")


# B4: Preserve scalar return plus role-specific Contact/ZMP/survival constraint evidence.
@dataclass(frozen=True)
class FrontRESV015GainReturnEvidence:
    """从唯一 FRS-GAIN-v006 owner 构造 candidate-only one-row return evidence.

    这不是 legacy PPO batch, 不能传入 to_ppo_batch.
    role-specific unloaded support 的 ZMP margin 保持 N/A, 不得填零.
    """

    policy_observations: torch.Tensor
    policy_actions: torch.Tensor
    policy_log_probs: torch.Tensor
    policy_values: torch.Tensor
    policy_means: torch.Tensor
    policy_sigmas: torch.Tensor
    gain_total: torch.Tensor
    intent_gain: torch.Tensor
    physics_gain: torch.Tensor
    repair_cost: torch.Tensor
    repaired_success: torch.Tensor
    noisy_success: torch.Tensor
    repaired_survival: torch.Tensor
    noisy_survival: torch.Tensor
    repaired_zmp_margin: torch.Tensor
    noisy_zmp_margin: torch.Tensor
    repaired_contact: torch.Tensor
    noisy_contact: torch.Tensor
    physics_success_gain: torch.Tensor
    physics_survival_quality_repaired: torch.Tensor
    physics_survival_quality_noisy: torch.Tensor
    physics_survival_gain: torch.Tensor
    physics_zmp_gain: torch.Tensor
    physics_contact_gain: torch.Tensor
    intent_quality_repaired: torch.Tensor
    intent_quality_noisy: torch.Tensor
    physics_admissible_repaired: torch.Tensor
    physics_admissible_noisy: torch.Tensor
    physics_deficit_repaired: torch.Tensor
    physics_deficit_noisy: torch.Tensor
    utility_repaired: torch.Tensor
    utility_noisy: torch.Tensor
    repair_penalty: torch.Tensor
    contact_constraint: torch.Tensor
    zmp_constraint: torch.Tensor
    survival_constraint: torch.Tensor
    zmp_applicable_repaired: torch.Tensor
    zmp_applicable_noisy: torch.Tensor
    zmp_constraint_applicable: torch.Tensor
    contact_constraint_advantage: torch.Tensor
    zmp_constraint_advantage: torch.Tensor
    survival_constraint_advantage: torch.Tensor
    physics_valid_step_count: torch.Tensor
    return_k: torch.Tensor
    advantage_k: torch.Tensor
    policy_row_valid: torch.Tensor
    horizon_k: torch.Tensor
    evidence_valid_step_count: torch.Tensor
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    intent_q29_provenance: str
    intent_q29_source: str
    gain_source: str = "FRS-GAIN-v006-loaded-support-zmp-applicability"
    scalar_target_id: str = "paired-intent-minus-repair-v1"
    constraint_schema_id: str = "contact-loaded-phase_zmp-survival-physical-v2"
    constraint_advantage_state: str = "unsealed"

    def validate(self) -> None:
        count = int(self.policy_actions.shape[0])
        if count <= 0 or tuple(self.policy_actions.shape[1:]) != (6,):
            raise ValueError("v015 return evidence requires policy_actions [B,6]")
        for name, value in (
            ("policy_log_probs", self.policy_log_probs),
            ("policy_values", self.policy_values),
            ("gain_total", self.gain_total),
            ("intent_gain", self.intent_gain),
            ("physics_gain", self.physics_gain),
            ("repair_cost", self.repair_cost),
            ("repaired_success", self.repaired_success),
            ("noisy_success", self.noisy_success),
            ("repaired_survival", self.repaired_survival),
            ("noisy_survival", self.noisy_survival),
            ("repaired_zmp_margin", self.repaired_zmp_margin),
            ("noisy_zmp_margin", self.noisy_zmp_margin),
            ("repaired_contact", self.repaired_contact),
            ("noisy_contact", self.noisy_contact),
            ("physics_success_gain", self.physics_success_gain),
            ("physics_survival_quality_repaired", self.physics_survival_quality_repaired),
            ("physics_survival_quality_noisy", self.physics_survival_quality_noisy),
            ("physics_survival_gain", self.physics_survival_gain),
            ("physics_contact_gain", self.physics_contact_gain),
            ("intent_quality_repaired", self.intent_quality_repaired),
            ("intent_quality_noisy", self.intent_quality_noisy),
            ("physics_admissible_repaired", self.physics_admissible_repaired),
            ("physics_admissible_noisy", self.physics_admissible_noisy),
            ("physics_deficit_repaired", self.physics_deficit_repaired),
            ("physics_deficit_noisy", self.physics_deficit_noisy),
            ("utility_repaired", self.utility_repaired),
            ("utility_noisy", self.utility_noisy),
            ("repair_penalty", self.repair_penalty),
            ("contact_constraint", self.contact_constraint),
            ("zmp_constraint", self.zmp_constraint),
            ("survival_constraint", self.survival_constraint),
            ("zmp_applicable_repaired", self.zmp_applicable_repaired),
            ("zmp_applicable_noisy", self.zmp_applicable_noisy),
            ("zmp_constraint_applicable", self.zmp_constraint_applicable),
            ("contact_constraint_advantage", self.contact_constraint_advantage),
            ("zmp_constraint_advantage", self.zmp_constraint_advantage),
            ("survival_constraint_advantage", self.survival_constraint_advantage),
            ("physics_valid_step_count", self.physics_valid_step_count),
            ("return_k", self.return_k),
            ("advantage_k", self.advantage_k),
            ("policy_row_valid", self.policy_row_valid),
            ("horizon_k", self.horizon_k),
            ("evidence_valid_step_count", self.evidence_valid_step_count),
        ):
            if value.ndim != 1 or int(value.numel()) != count:
                raise ValueError(f"v015 return evidence {name} must be [B]")
        if (
            self.policy_observations.ndim != 2
            or int(self.policy_observations.shape[0]) != count
            or tuple(self.policy_means.shape) != (count, 6)
            or tuple(self.policy_sigmas.shape) != (count, 6)
            or len(self.scenario_ids) != count
            or len(self.noisy_segment_hashes) != count
            or len(self.x_t_identities) != count
            or self.gain_source != "FRS-GAIN-v006-loaded-support-zmp-applicability"
            or self.scalar_target_id != "paired-intent-minus-repair-v1"
            or self.constraint_schema_id != "contact-loaded-phase_zmp-survival-physical-v2"
            or bool((self.horizon_k <= 0).any())
            or bool((self.evidence_valid_step_count < 0).any())
            or bool((self.evidence_valid_step_count > self.horizon_k).any())
            or bool((self.physics_valid_step_count < 0).any())
            or bool((self.physics_valid_step_count > self.horizon_k).any())
        ):
            raise ValueError("v015 return evidence has invalid policy tuple or Gain source")
        valid = self.policy_row_valid.bool()
        constraint_advantages = (
            self.contact_constraint_advantage,
            self.zmp_constraint_advantage,
            self.survival_constraint_advantage,
        )
        if self.constraint_advantage_state == "sealed":
            if any(not bool(torch.isfinite(value[valid]).all()) for value in constraint_advantages):
                raise ValueError("FRS-GAIN-v006 sealed constraint advantages must be finite on valid rows")
        elif self.constraint_advantage_state == "unsealed":
            if any(bool(torch.isfinite(value).any()) for value in constraint_advantages):
                raise ValueError("FRS-GAIN-v006 unsealed constraint advantages must remain UNCONFIRMED")
        else:
            raise ValueError("FRS-GAIN-v006 rejects unknown constraint advantage lifecycle state")
        for name, value in (
            ("gain_total", self.gain_total),
            ("intent_gain", self.intent_gain),
            ("physics_gain", self.physics_gain),
            ("repair_cost", self.repair_cost),
            ("return_k", self.return_k),
            ("advantage_k", self.advantage_k),
            ("repaired_success", self.repaired_success),
            ("noisy_success", self.noisy_success),
            ("repaired_survival", self.repaired_survival),
            ("noisy_survival", self.noisy_survival),
            ("repaired_contact", self.repaired_contact),
            ("noisy_contact", self.noisy_contact),
            ("physics_success_gain", self.physics_success_gain),
            ("physics_survival_quality_repaired", self.physics_survival_quality_repaired),
            ("physics_survival_quality_noisy", self.physics_survival_quality_noisy),
            ("physics_survival_gain", self.physics_survival_gain),
            ("physics_contact_gain", self.physics_contact_gain),
            ("intent_quality_repaired", self.intent_quality_repaired),
            ("intent_quality_noisy", self.intent_quality_noisy),
            ("physics_admissible_repaired", self.physics_admissible_repaired),
            ("physics_admissible_noisy", self.physics_admissible_noisy),
            ("physics_deficit_repaired", self.physics_deficit_repaired),
            ("physics_deficit_noisy", self.physics_deficit_noisy),
            ("utility_repaired", self.utility_repaired),
            ("utility_noisy", self.utility_noisy),
            ("repair_penalty", self.repair_penalty),
        ):
            finite = torch.isfinite(value)
            if not bool(finite[valid].all()) or bool(finite[~valid].any()):
                raise ValueError(f"v015 return evidence {name} must be finite exactly on valid rows")
        if not torch.equal(self.zmp_constraint_applicable.bool(), self.zmp_applicable_repaired.bool()):
            raise ValueError("v015 return evidence PPO ZMP applicability must alias the Repair role")
        if bool(self.zmp_applicable_repaired.bool()[~valid].any()) or bool(
            self.zmp_applicable_noisy.bool()[~valid].any()
        ):
            raise ValueError("v015 return evidence ZMP applicability must be false outside valid policy rows")
        repaired_zmp_applicable = valid & self.zmp_applicable_repaired.bool()
        repaired_zmp_finite = torch.isfinite(self.repaired_zmp_margin)
        if (
            not bool(repaired_zmp_finite[repaired_zmp_applicable].all())
            or bool(repaired_zmp_finite[~repaired_zmp_applicable].any())
        ):
            raise ValueError("v015 return evidence Repair ZMP must follow role-specific loaded-support applicability")
        noisy_zmp_applicable = valid & self.zmp_applicable_noisy.bool()
        noisy_zmp_finite = torch.isfinite(self.noisy_zmp_margin)
        if (
            not bool(noisy_zmp_finite[noisy_zmp_applicable].all())
            or bool(noisy_zmp_finite[~noisy_zmp_applicable].any())
        ):
            raise ValueError("v015 return evidence Noisy ZMP must follow role-specific loaded-support applicability")
        paired_zmp_applicable = repaired_zmp_applicable & noisy_zmp_applicable
        physics_zmp_finite = torch.isfinite(self.physics_zmp_gain)
        if (
            not bool(physics_zmp_finite[paired_zmp_applicable].all())
            or bool(physics_zmp_finite[~paired_zmp_applicable].any())
        ):
            raise ValueError(
                "v015 return evidence paired ZMP gain must be finite exactly when both role margins are applicable"
            )
        source = self.intent_q29_source.lower()
        if (
            self.intent_q29_provenance != "deployment_noisy_q29"
            or not source
            or any(token in source for token in ("clean", "root", "global"))
        ):
            raise ValueError("v015 return evidence rejects non-deployment q29 provenance")


def pair_frontres_v015_gain_facts(evidence: FrontRESV015OneActionKEvidence) -> FrontRESV015PairedGainFacts:
    """按 scenario 内 attempt 顺序配对 Repair/Noisy facts, 不把 Clean C 用作 intent."""

    evidence.validate()
    repair_rows = evidence.policy_row_indices.to(dtype=torch.long)
    noisy_by_scenario: dict[str, list[int]] = {}
    for row, scenario_id in enumerate(evidence.scenario_ids):
        if evidence.roles[row] == "noisy":
            noisy_by_scenario.setdefault(str(scenario_id), []).append(row)
    repair_attempt_index: dict[str, int] = {}
    noisy_rows: list[int] = []
    for repair_row in repair_rows.tolist():
        scenario_id = str(evidence.scenario_ids[int(repair_row)])
        attempt = repair_attempt_index.get(scenario_id, 0)
        matches = noisy_by_scenario.get(scenario_id, ())
        if attempt >= len(matches):
            raise ValueError(
                f"v015 paired gain facts require an attempt-aligned Noisy row for scenario={scenario_id!r}"
            )
        noisy_rows.append(matches[attempt])
        repair_attempt_index[scenario_id] = attempt + 1
    noisy_index = torch.tensor(noisy_rows, device=repair_rows.device, dtype=torch.long)
    provenance = tuple(evidence.intent_q29_provenance[int(row)] for row in repair_rows.tolist())
    source = tuple(evidence.intent_q29_source[int(row)] for row in repair_rows.tolist())
    if len(set(provenance)) != 1 or len(set(source)) != 1:
        raise ValueError("v015 paired gain facts require one q29 provenance/source across the candidate batch")

    def paired_step_mean(values: torch.Tensor) -> torch.Tensor:
        mask = evidence.physics_pair_valid_mask.bool()
        finite = torch.isfinite(values.float())
        usable = mask & finite
        count = usable.sum(dim=0)
        summed = torch.where(usable, values.float(), torch.zeros_like(values.float())).sum(dim=0)
        mean = summed / count.clamp_min(1).to(dtype=summed.dtype)
        return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))

    physics_valid_step_count = evidence.physics_pair_valid_mask.bool().sum(dim=0).to(dtype=torch.long)
    expected_support = evidence.physics_expected_support_steps.detach().bool()
    repair_phase = evaluate_phase_conditioned_physics(
        expected_support,
        evidence.physics_contact_repaired_steps,
        evidence.physics_zmp_repaired_steps,
        evidence.physics_pair_valid_mask,
    )
    noisy_phase = evaluate_phase_conditioned_physics(
        expected_support,
        evidence.physics_contact_noisy_steps,
        evidence.physics_zmp_noisy_steps,
        evidence.physics_pair_valid_mask,
    )
    intent_valid = (
        evidence.executed_q29_t_valid_mask.index_select(0, repair_rows).bool()
        & evidence.executed_q29_t_valid_mask.index_select(0, noisy_index).bool()
        & (physics_valid_step_count > 0)
    ).detach().clone()
    phase_nan = torch.full((int(repair_rows.numel()),), float("nan"), device=repair_rows.device)

    def valid_phase(value: torch.Tensor) -> torch.Tensor:
        return torch.where(intent_valid, value.to(device=repair_rows.device, dtype=torch.float32), phase_nan)

    facts = FrontRESV015PairedGainFacts(
        policy_observations=evidence.policy_observations.detach().clone(),
        policy_actions=evidence.policy_actions.detach().clone(),
        policy_log_probs=evidence.policy_log_probs.detach().clone(),
        policy_values=evidence.policy_values.detach().clone(),
        policy_means=evidence.policy_means.detach().clone(),
        policy_sigmas=evidence.policy_sigmas.detach().clone(),
        intent_q29=evidence.intent_q29.index_select(0, repair_rows)[:, 0].detach().clone(),
        repaired_q29=evidence.executed_q29_t.index_select(0, repair_rows).detach().clone(),
        noisy_q29=evidence.executed_q29_t.index_select(0, noisy_index).detach().clone(),
        intent_valid_mask=intent_valid,
        repaired_success=(~evidence.done_any.index_select(0, repair_rows).bool()).detach().clone(),
        noisy_success=(~evidence.done_any.index_select(0, noisy_index).bool()).detach().clone(),
        repaired_survival=evidence.survival_steps.index_select(0, repair_rows).detach().clone(),
        noisy_survival=evidence.survival_steps.index_select(0, noisy_index).detach().clone(),
        repaired_zmp_margin=paired_step_mean(evidence.physics_zmp_repaired_steps).detach().clone(),
        noisy_zmp_margin=paired_step_mean(evidence.physics_zmp_noisy_steps).detach().clone(),
        repaired_contact=valid_phase(1.0 - repair_phase["contact_violation"]).detach().clone(),
        noisy_contact=valid_phase(1.0 - noisy_phase["contact_violation"]).detach().clone(),
        repaired_contact_violation=valid_phase(repair_phase["contact_violation"]).detach().clone(),
        noisy_contact_violation=valid_phase(noisy_phase["contact_violation"]).detach().clone(),
        repaired_zmp_violation=valid_phase(repair_phase["zmp_violation"]).detach().clone(),
        noisy_zmp_violation=valid_phase(noisy_phase["zmp_violation"]).detach().clone(),
        physics_valid_step_count=physics_valid_step_count.detach().clone(),
        horizon_k=evidence.horizon_k.index_select(0, repair_rows).detach().clone(),
        expected_support_steps=evidence.physics_expected_support_steps.detach().clone(),
        repaired_contact_steps=evidence.physics_contact_repaired_steps.detach().clone(),
        noisy_contact_steps=evidence.physics_contact_noisy_steps.detach().clone(),
        repaired_zmp_margin_steps=evidence.physics_zmp_repaired_steps.detach().clone(),
        noisy_zmp_margin_steps=evidence.physics_zmp_noisy_steps.detach().clone(),
        physics_pair_valid_mask=evidence.physics_pair_valid_mask.detach().clone(),
        scenario_ids=tuple(evidence.scenario_ids[int(row)] for row in repair_rows.tolist()),
        noisy_segment_hashes=tuple(evidence.noisy_segment_hashes[int(row)] for row in repair_rows.tolist()),
        x_t_identities=tuple(evidence.x_t_identities[int(row)] for row in repair_rows.tolist()),
        intent_q29_provenance=provenance[0],
        intent_q29_source=source[0],
    )
    facts.validate()
    return facts


def build_frontres_v015_gain_return_evidence(
    facts: FrontRESV015PairedGainFacts,
    gain_result: Any,
) -> FrontRESV015GainReturnEvidence:
    """只从 v003 为每个 Repair policy row 构造一个 return/advantage carrier."""

    facts.validate()
    count = int(facts.policy_actions.shape[0])
    components: dict[str, torch.Tensor] = {}
    for name in ("gain_total", "intent_gain", "physics_gain", "repair_cost"):
        value = getattr(gain_result, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or int(value.numel()) != count:
            raise ValueError(f"v016 return evidence requires FRS-GAIN-v006 {name} [B]")
        components[name] = value.detach().to(device=facts.policy_values.device, dtype=torch.float32).clone()
    if (
        getattr(gain_result, "intent_q29_provenance", None) != facts.intent_q29_provenance
        or getattr(gain_result, "intent_q29_source", None) != facts.intent_q29_source
    ):
        raise ValueError("v015 return evidence rejects a Gain result with mismatched q29 provenance")
    valid = facts.intent_valid_mask.bool()
    for value in components.values():
        valid = valid & torch.isfinite(value)
    nan = torch.full_like(components["gain_total"], float("nan"))
    masked_components = {
        name: torch.where(valid, value, nan)
        for name, value in components.items()
    }
    constraint_components: dict[str, torch.Tensor] = {}
    for name in ("contact_constraint", "zmp_constraint", "survival_constraint"):
        value = getattr(gain_result, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or int(value.numel()) != count:
            raise ValueError(f"v015 return evidence requires FRS-GAIN-v006 {name} [B]")
        value = value.detach().to(device=facts.policy_values.device, dtype=torch.float32).clone()
        if not bool(torch.isfinite(value[valid]).all()) or bool((value[valid] < 0.0).any()):
            raise ValueError(f"FRS-GAIN-v006 {name} must be finite and nonnegative on valid rows")
        constraint_components[name] = torch.where(valid, value, nan)
    zmp_applicability: dict[str, torch.Tensor] = {}
    for name in ("zmp_applicable_repaired", "zmp_applicable_noisy", "zmp_constraint_applicable"):
        value = getattr(gain_result, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or int(value.numel()) != count:
            raise ValueError(f"v015 return evidence requires FRS-GAIN-v006 {name} [B]")
        value = value.detach().to(device=facts.policy_values.device, dtype=torch.bool).clone()
        zmp_applicability[name] = value & valid
    if not torch.equal(
        zmp_applicability["zmp_constraint_applicable"],
        zmp_applicability["zmp_applicable_repaired"],
    ):
        raise ValueError("FRS-GAIN-v006 PPO ZMP applicability must alias the Repair role")

    scenario_counts = {
        scenario_id: sum(observed == scenario_id for observed in facts.scenario_ids)
        for scenario_id in set(facts.scenario_ids)
    }
    advantages_sealed = all(count >= 2 for count in scenario_counts.values())

    def centered_by_scenario(value: torch.Tensor) -> torch.Tensor:
        result = torch.full_like(value, float("nan"))
        if not advantages_sealed:
            return result
        for scenario_id in sorted(set(facts.scenario_ids)):
            rows = [index for index, observed in enumerate(facts.scenario_ids) if observed == scenario_id]
            index = torch.tensor(rows, device=value.device, dtype=torch.long)
            selected = value.index_select(0, index)
            selected_valid = valid.index_select(0, index)
            if not bool(selected_valid.all()):
                valid_rows = tuple(bool(item) for item in selected_valid.detach().cpu().tolist())
                physics_steps = tuple(
                    int(facts.physics_valid_step_count[row].detach().cpu().item()) for row in rows
                )
                raise FrontRESV015RejectedTransactionEvidence(
                    "FRS-GAIN-v006 rejected the complete transaction before constraint centering: "
                    f"scenario_id={scenario_id!r} valid_repair_attempts={valid_rows} "
                    f"physics_valid_step_count={physics_steps}"
                )
            centered = selected - selected.mean().detach()
            result.index_copy_(0, index, centered)
        return result

    constraint_advantages = {
        name: centered_by_scenario(value)
        for name, value in constraint_components.items()
    }
    physics_components: dict[str, torch.Tensor] = {}
    for name in (
        "physics_success_gain",
        "physics_survival_quality_repaired",
        "physics_survival_quality_noisy",
        "physics_survival_gain",
        "physics_zmp_gain",
        "physics_contact_gain",
        "intent_quality_repaired",
        "intent_quality_noisy",
        "physics_admissible_repaired",
        "physics_admissible_noisy",
        "physics_deficit_repaired",
        "physics_deficit_noisy",
        "utility_repaired",
        "utility_noisy",
        "repair_penalty",
    ):
        value = getattr(gain_result, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or int(value.numel()) != count:
            raise ValueError(f"v016 return evidence requires FRS-GAIN-v006 {name} [B]")
        value = value.detach().to(device=facts.policy_values.device, dtype=torch.float32).clone()
        component_valid = valid
        if name == "physics_zmp_gain":
            component_valid = (
                valid
                & zmp_applicability["zmp_applicable_repaired"]
                & zmp_applicability["zmp_applicable_noisy"]
            )
        physics_components[name] = torch.where(component_valid, value, nan)

    def masked_fact(value: torch.Tensor) -> torch.Tensor:
        return torch.where(
            valid,
            value.detach().to(device=facts.policy_values.device, dtype=torch.float32),
            nan,
        ).clone()

    def masked_zmp_fact(value: torch.Tensor, applicability_name: str) -> torch.Tensor:
        applicable = zmp_applicability[applicability_name]
        return torch.where(
            valid & applicable,
            value.detach().to(device=facts.policy_values.device, dtype=torch.float32),
            nan,
        ).clone()

    return_k = masked_components["gain_total"]
    advantage_k = torch.where(valid, return_k - facts.policy_values.detach().float(), nan)
    survival = facts.repaired_survival.detach().to(device=facts.policy_values.device, dtype=torch.float32)
    if not bool(torch.isfinite(survival).all()) or bool((survival < 0.0).any()):
        raise ValueError("v015 return evidence requires finite non-negative K survival evidence")
    evidence_valid_step_count = survival.to(dtype=torch.long)
    if not torch.equal(survival, evidence_valid_step_count.to(dtype=survival.dtype)):
        raise ValueError("v015 return evidence requires integer K survival-step counts")
    if bool((evidence_valid_step_count > facts.horizon_k).any()):
        raise ValueError("v015 return evidence survival-step count exceeds horizon_k")
    result = FrontRESV015GainReturnEvidence(
        policy_observations=facts.policy_observations.detach().clone(),
        policy_actions=facts.policy_actions.detach().clone(),
        policy_log_probs=facts.policy_log_probs.detach().clone(),
        policy_values=facts.policy_values.detach().clone(),
        policy_means=facts.policy_means.detach().clone(),
        policy_sigmas=facts.policy_sigmas.detach().clone(),
        gain_total=masked_components["gain_total"],
        intent_gain=masked_components["intent_gain"],
        physics_gain=masked_components["physics_gain"],
        repair_cost=masked_components["repair_cost"],
        repaired_success=masked_fact(facts.repaired_success),
        noisy_success=masked_fact(facts.noisy_success),
        repaired_survival=masked_fact(facts.repaired_survival),
        noisy_survival=masked_fact(facts.noisy_survival),
        repaired_zmp_margin=masked_zmp_fact(facts.repaired_zmp_margin, "zmp_applicable_repaired"),
        noisy_zmp_margin=masked_zmp_fact(facts.noisy_zmp_margin, "zmp_applicable_noisy"),
        repaired_contact=masked_fact(facts.repaired_contact),
        noisy_contact=masked_fact(facts.noisy_contact),
        physics_success_gain=physics_components["physics_success_gain"],
        physics_survival_quality_repaired=physics_components["physics_survival_quality_repaired"],
        physics_survival_quality_noisy=physics_components["physics_survival_quality_noisy"],
        physics_survival_gain=physics_components["physics_survival_gain"],
        physics_zmp_gain=physics_components["physics_zmp_gain"],
        physics_contact_gain=physics_components["physics_contact_gain"],
        intent_quality_repaired=physics_components["intent_quality_repaired"],
        intent_quality_noisy=physics_components["intent_quality_noisy"],
        physics_admissible_repaired=physics_components["physics_admissible_repaired"],
        physics_admissible_noisy=physics_components["physics_admissible_noisy"],
        physics_deficit_repaired=physics_components["physics_deficit_repaired"],
        physics_deficit_noisy=physics_components["physics_deficit_noisy"],
        utility_repaired=physics_components["utility_repaired"],
        utility_noisy=physics_components["utility_noisy"],
        repair_penalty=physics_components["repair_penalty"],
        contact_constraint=constraint_components["contact_constraint"],
        zmp_constraint=constraint_components["zmp_constraint"],
        survival_constraint=constraint_components["survival_constraint"],
        zmp_applicable_repaired=zmp_applicability["zmp_applicable_repaired"],
        zmp_applicable_noisy=zmp_applicability["zmp_applicable_noisy"],
        zmp_constraint_applicable=zmp_applicability["zmp_constraint_applicable"],
        contact_constraint_advantage=constraint_advantages["contact_constraint"],
        zmp_constraint_advantage=constraint_advantages["zmp_constraint"],
        survival_constraint_advantage=constraint_advantages["survival_constraint"],
        constraint_advantage_state="sealed" if advantages_sealed else "unsealed",
        physics_valid_step_count=facts.physics_valid_step_count.detach().clone(),
        return_k=return_k,
        advantage_k=advantage_k,
        policy_row_valid=valid,
        horizon_k=facts.horizon_k.detach().clone(),
        evidence_valid_step_count=evidence_valid_step_count.detach().clone(),
        scenario_ids=facts.scenario_ids,
        noisy_segment_hashes=facts.noisy_segment_hashes,
        x_t_identities=facts.x_t_identities,
        intent_q29_provenance=facts.intent_q29_provenance,
        intent_q29_source=facts.intent_q29_source,
    )
    result.validate()
    return result
