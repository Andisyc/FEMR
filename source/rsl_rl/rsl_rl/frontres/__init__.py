# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrontRES domain helpers for runner, algorithm, and diagnostics code."""

from .frontres_action_cone import FrontRESActionCone
from .frontres_dr_curriculum import (
    DRStrengthPlan,
    GMTFrontierState,
    GMTFrontierUpdate,
    PerturbationMixPlan,
    allowed_perturbation_bases,
    choose_perturbation_choices,
    mode_complexity,
    sample_per_env_dr_strength,
    sample_perturbation_mix,
    sample_scalar_dr_strength,
    score_gmt_frontier,
    update_boundary_ema,
    update_gmt_frontier_state,
    warmup_perturbation_mode_groups,
)
from .frontres_executability import (
    FrontRESExecutabilityScorer,
    quat_to_rotvec_wxyz,
    rotvec_to_quat_wxyz,
)
from .frontres_metrics import (
    FrontRESMetricsAccumulator,
    frontres_boundary_stats,
    frontres_metric_mean,
)
from .frontres_rollout_evidence import FrontRESRolloutEvidence, compute_frontres_rollout_evidence
from .perturbation_runtime import (
    apply_frontres_dr_scale,
    apply_frontres_dr_scale_env,
    apply_frontres_family_env_masks,
    snapshot_frontres_perturbation_target,
)
from .runtime_diagnostics import maybe_print_frontres_restore_debug
from .task_space_correction import (
    apply_frontres_task_corrections,
)
from .training_schedule import (
    FrontRESDRIterationPlan,
    FrontRESDRScaleEnvPlan,
    FrontRESDRSetup,
    FrontRESModeState,
    FrontRESPairLayout,
    frontres_curriculum_allowed_bases,
    frontres_curriculum_choices,
    frontres_curriculum_hash,
    frontres_mixed_dr_scale,
    frontres_mixed_dr_scale_env,
    frontres_warmup_perturbation_mode_groups,
    resolve_frontres_mode_state,
)

__all__ = [
    "DRStrengthPlan",
    "FrontRESActionCone",
    "FrontRESDRIterationPlan",
    "FrontRESDRScaleEnvPlan",
    "FrontRESDRSetup",
    "FrontRESExecutabilityScorer",
    "FrontRESMetricsAccumulator",
    "FrontRESModeState",
    "FrontRESPairLayout",
    "FrontRESRolloutEvidence",
    "GMTFrontierState",
    "GMTFrontierUpdate",
    "PerturbationMixPlan",
    "allowed_perturbation_bases",
    "apply_frontres_dr_scale",
    "apply_frontres_dr_scale_env",
    "apply_frontres_family_env_masks",
    "apply_frontres_task_corrections",
    "choose_perturbation_choices",
    "compute_frontres_rollout_evidence",
    "frontres_boundary_stats",
    "frontres_curriculum_allowed_bases",
    "frontres_curriculum_choices",
    "frontres_curriculum_hash",
    "frontres_metric_mean",
    "frontres_mixed_dr_scale",
    "frontres_mixed_dr_scale_env",
    "frontres_warmup_perturbation_mode_groups",
    "maybe_print_frontres_restore_debug",
    "mode_complexity",
    "quat_to_rotvec_wxyz",
    "resolve_frontres_mode_state",
    "rotvec_to_quat_wxyz",
    "sample_per_env_dr_strength",
    "sample_perturbation_mix",
    "sample_scalar_dr_strength",
    "score_gmt_frontier",
    "snapshot_frontres_perturbation_target",
    "update_boundary_ema",
    "update_gmt_frontier_state",
    "warmup_perturbation_mode_groups",
]
