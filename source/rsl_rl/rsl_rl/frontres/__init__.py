# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public FrontRES domain facade.

Exports are loaded on first access so importing a narrow interface does not
eagerly construct IsaacLab, simulator-math, curriculum, and diagnostics
dependencies. Existing ``from rsl_rl.frontres import ...`` call sites retain
their public API while new code can depend on lightweight contract modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "DRStrengthPlan": ".frontres_dr_curriculum",
    "GMTFrontierState": ".frontres_dr_curriculum",
    "GMTFrontierUpdate": ".frontres_dr_curriculum",
    "PerturbationMixPlan": ".frontres_dr_curriculum",
    "allowed_perturbation_bases": ".frontres_dr_curriculum",
    "choose_perturbation_choices": ".frontres_dr_curriculum",
    "mode_complexity": ".frontres_dr_curriculum",
    "sample_per_env_dr_strength": ".frontres_dr_curriculum",
    "sample_perturbation_mix": ".frontres_dr_curriculum",
    "sample_scalar_dr_strength": ".frontres_dr_curriculum",
    "score_gmt_frontier": ".frontres_dr_curriculum",
    "update_boundary_ema": ".frontres_dr_curriculum",
    "update_gmt_frontier_state": ".frontres_dr_curriculum",
    "warmup_perturbation_mode_groups": ".frontres_dr_curriculum",
    "FrontRESExecutabilityScorer": ".frontres_executability",
    "quat_to_rotvec_wxyz": ".frontres_executability",
    "rotvec_to_quat_wxyz": ".frontres_executability",
    "FrontRESMetricsAccumulator": ".frontres_metrics",
    "frontres_boundary_stats": ".frontres_metrics",
    "frontres_metric_mean": ".frontres_metrics",
    "FrontRESRolloutEvidence": ".frontres_rollout_evidence",
    "compute_frontres_rollout_evidence": ".frontres_rollout_evidence",
    "apply_frontres_dr_scale": ".perturbation_runtime",
    "apply_frontres_dr_scale_env": ".perturbation_runtime",
    "apply_frontres_family_env_masks": ".perturbation_runtime",
    "snapshot_frontres_perturbation_target": ".perturbation_runtime",
    "maybe_print_frontres_restore_debug": ".runtime_diagnostics",
    "apply_frontres_task_corrections": ".task_space_correction",
    "FrontRESDRIterationPlan": ".training_schedule",
    "FrontRESDRScaleEnvPlan": ".training_schedule",
    "FrontRESDRSetup": ".training_schedule",
    "FrontRESModeState": ".training_schedule",
    "FrontRESPairLayout": ".training_schedule",
    "frontres_curriculum_allowed_bases": ".training_schedule",
    "frontres_curriculum_choices": ".training_schedule",
    "frontres_curriculum_hash": ".training_schedule",
    "frontres_mixed_dr_scale": ".training_schedule",
    "frontres_mixed_dr_scale_env": ".training_schedule",
    "frontres_warmup_perturbation_mode_groups": ".training_schedule",
    "resolve_frontres_mode_state": ".training_schedule",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
