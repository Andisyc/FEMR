# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Mapping


class FrontRESMetricsAccumulator:
    """Small helpers for FrontRES metric aggregation and live-path state."""

    @staticmethod
    def mean_or_none(enabled: bool, steps: int, total: float) -> float | None:
        if not enabled or steps <= 0:
            return None
        return total / steps

    @staticmethod
    def boundary_stats(
        *,
        safe_frac: float | None,
        repair_frac: float | None,
        broken_frac: float | None,
        positive_gain_frac: float | None,
        candidate_floor_pass: float | None,
        stable_route_frac: float | None,
        stable_endpoint_frac: float | None,
    ) -> dict[str, float] | None:
        """Build the DR boundary-controller stats only when required signals exist."""
        if (
            safe_frac is None
            or repair_frac is None
            or broken_frac is None
            or positive_gain_frac is None
        ):
            return None
        return {
            "safe": float(safe_frac),
            "repair": float(repair_frac),
            "broken": float(broken_frac),
            "positive_gain": float(positive_gain_frac),
            "candidate_floor_pass": float(candidate_floor_pass or 0.0),
            "stable_route": float(stable_route_frac or 0.0),
            "stable_endpoint": float(stable_endpoint_frac or 0.0),
        }


def frontres_metric_mean(enabled: bool, steps: int, total: float) -> float | None:
    """Compatibility helper for converting accumulated FrontRES sums to means."""
    return FrontRESMetricsAccumulator.mean_or_none(enabled, steps, total)


def frontres_boundary_stats(metrics: Mapping[str, float | None]) -> dict[str, float] | None:
    """Build boundary stats from a metrics mapping with stable FrontRES keys."""
    return FrontRESMetricsAccumulator.boundary_stats(
        safe_frac=metrics.get("frontres_safe_frac_mean"),
        repair_frac=metrics.get("frontres_repair_frac_mean"),
        broken_frac=metrics.get("frontres_broken_frac_mean"),
        positive_gain_frac=metrics.get("frontres_positive_gain_frac_mean"),
        candidate_floor_pass=metrics.get("frontres_candidate_floor_pass_mean"),
        stable_route_frac=metrics.get("frontres_stable_route_frac_mean"),
        stable_endpoint_frac=metrics.get("frontres_stable_endpoint_frac_mean"),
    )
